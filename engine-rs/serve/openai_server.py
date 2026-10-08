#!/usr/bin/env python3
"""vLLM/OpenAI-compatible HTTP server for `spark-engine serve` (TP2, greedy speculative decoding).

The engine (both ranks) must already run `serve`; rank0 listens on a Unix socket. This process
owns tokenization, the model's chat template, reasoning (<think>) parsing, stop strings and SSE.

Endpoints (vLLM style):
  GET  /health  /ping  /version  /metrics  /v1/models  /v1/models/{id}
  POST /v1/chat/completions   (stream / non-stream, stream_options.include_usage, stop, stop_token_ids,
                               max_tokens / max_completion_tokens, chat_template_kwargs, reasoning split)
  POST /v1/completions        (prompt: str | [str] | [int] | [[int]], echo, stream)
Images and videos in chat content (OpenAI image_url, vLLM video_url, input_image/input_video, HF/Qwen image/video,
Anthropic base64 source; data URLs, bare base64, http(s), allow-listed local files) go through mm.py; the engine
encodes them with the vision tower (crate::vision). Audio is rejected: the model has no audio encoder.
  POST /tokenize  /detokenize
Up to GLM53_SERVE_MAX_SEQS sequences run concurrently on the engine (default 4); more requests queue. Sampling: temperature (0 = greedy;
default GLM53_SERVE_DEFAULT_TEMPERATURE, 0) and seed are applied by exact speculative sampling; top_p, top_k,
presence/frequency penalties are accepted and not applied. n must be 1.

Usage:
  python3 openai_server.py --socket /tmp/glm53-serve.sock --model-dir <snapshot> --port 8888
"""
import argparse, json, os, queue, re, socket, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import jinja2
from tokenizers import Tokenizer

try:
    import mm as media_mod          # needs numpy/Pillow (+ PyAV for video): serve/venv
except ImportError as _e:           # text-only front end
    media_mod, MEDIA_IMPORT_ERROR = None, _e

EOS = [154820, 154827, 154829]
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
VERSION = "spark-engine M3 (openai-compatible front end)"


class ApiError(Exception):
    def __init__(self, code, message, etype="invalid_request_error", param=None):
        super().__init__(message)
        self.code, self.message, self.etype, self.param = code, message, etype, param


# ----------------------------------------------------------------------------- metrics
# Prometheus text exposition. vllm:* names/semantics follow vLLM v1 so the existing monitoring
# (dashboards, alerts) works unchanged; glm53_* are engine specifics.
TTFT_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]
ITL_BUCKETS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0]
LAT_BUCKETS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0]
TOK_BUCKETS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]


class Histogram:
    def __init__(self, buckets):
        self.b, self.counts, self.sum, self.n = list(buckets), [0] * len(buckets), 0.0, 0

    def observe(self, v, times=1):
        for i, ub in enumerate(self.b):
            if v <= ub:
                self.counts[i] += times
                break
        self.sum += v * times
        self.n += times

    def render(self, name, lab):
        out, acc = [], 0
        for ub, c in zip(self.b, self.counts):
            acc += c
            out.append(f'{name}_bucket{{{lab},le="{ub}"}} {acc}')
        out.append(f'{name}_bucket{{{lab},le="+Inf"}} {self.n}')
        out.append(f"{name}_sum{{{lab}}} {self.sum}")
        out.append(f"{name}_count{{{lab}}} {self.n}")
        return out


class Metrics:
    COUNTERS = [  # name, help
        ("vllm:prompt_tokens_total", "Number of prefill tokens processed."),
        ("vllm:generation_tokens_total", "Number of generation tokens processed."),
        ("vllm:prefix_cache_queries_total", "Prefix cache queries, in terms of number of queried tokens."),
        ("vllm:prefix_cache_hits_total", "Prefix cache hits, in terms of number of cached tokens."),
        ("vllm:prompt_tokens_cached_total", "Number of cached prompt tokens (reused from a sequence store)."),
        ("vllm:num_preemptions_total", "Cumulative number of preemptions (always 0: this engine does not preempt)."),
        ("vllm:spec_decode_num_drafts_total", "Number of speculative verification rounds."),
        ("vllm:spec_decode_num_draft_tokens_total", "Number of draft tokens proposed for verification."),
        ("vllm:spec_decode_num_accepted_tokens_total", "Number of accepted draft tokens."),
        ("glm53_requests_total", "Finished generations (all endpoints)."),
        ("glm53_errors_total", "Requests answered with an error."),
        ("glm53_prefill_seconds_total", "Engine prefill wall time summed over requests."),
        ("glm53_decode_seconds_total", "Engine decode wall time summed over requests."),
        ("glm53_tool_calls_total", "Tool calls parsed from model output."),
    ]
    HISTS = [
        ("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", TTFT_BUCKETS),
        ("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", ITL_BUCKETS),
        ("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.", LAT_BUCKETS),
        ("vllm:request_queue_time_seconds", "Histogram of time spent waiting for a sequence slot.", LAT_BUCKETS),
        ("vllm:request_prefill_time_seconds", "Histogram of time spent in prefill.", LAT_BUCKETS),
        ("vllm:request_decode_time_seconds", "Histogram of time spent in decode.", LAT_BUCKETS),
        ("vllm:request_prompt_tokens", "Number of prefill tokens processed.", TOK_BUCKETS),
        ("vllm:request_generation_tokens", "Number of generation tokens processed.", TOK_BUCKETS),
    ]

    def __init__(self, model):
        self.lab = f'model_name="{model}",engine="0"'
        self.lock = threading.Lock()
        self.c = {n: 0 for n, _ in self.COUNTERS}
        self.success = {r: 0 for r in ("stop", "length", "abort", "tool_calls")}
        self.h = {n: Histogram(b) for n, _, b in self.HISTS}
        self.inflight = 0

    def add(self, name, v):
        with self.lock:
            self.c[name] += v

    def observe(self, name, v, times=1):
        with self.lock:
            self.h[name].observe(v, times)

    def finished(self, reason):
        with self.lock:
            self.success[reason] = self.success.get(reason, 0) + 1

    def render(self, stats, max_seqs):
        lab, out = self.lab, []
        with self.lock:
            c, succ, inflight = dict(self.c), dict(self.success), self.inflight
            hists = [(n, h, self.h[n].render(n, lab)) for n, h, _ in self.HISTS]
        up = stats is not None
        if up:
            running, waiting = stats["running"], stats["waiting"]
            kv_perc = stats["kv_active_tokens"] / max(1, stats["kv_budget_tokens"])
        else:                                   # engine busy/unreachable: front-end estimate
            running, waiting, kv_perc = min(inflight, max_seqs), max(0, inflight - max_seqs), 0.0

        def g(name, help_, v, typ="gauge"):
            out.extend([f"# HELP {name} {help_}", f"# TYPE {name} {typ}", f"{name}{{{lab}}} {v}"])
        g("vllm:num_requests_running", "Number of requests in model execution batches (admitted sequences).", running)
        g("vllm:num_requests_waiting", "Number of requests waiting to be processed.", waiting)
        g("vllm:kv_cache_usage_perc", "KV-cache usage: tokens held by active sequence stores / KV budget (0..1).", kv_perc)
        for n, help_ in self.COUNTERS:
            g(n, help_, c[n], "counter")
        out.extend(["# HELP vllm:request_success_total Count of successfully processed requests.",
                    "# TYPE vllm:request_success_total counter"])
        for r, v in succ.items():
            out.append(f'vllm:request_success_total{{{lab},finished_reason="{r}"}} {v}')
        for (n, help_, _), (_, _, lines) in zip(self.HISTS, hists):
            out.extend([f"# HELP {n} {help_}", f"# TYPE {n} histogram"] + lines)
        g("glm53_engine_stats_up", "1 if the engine answered the stats request for this scrape.", int(up))
        if up:
            g("glm53_decoding_sequences", "Admitted sequences currently decoding (others are prefilling).", stats["decoding"])
            g("glm53_max_sequences", "Configured concurrent sequence limit (GLM53_SERVE_MAX_SEQS).", stats["max_seqs"])
            g("glm53_sequence_stores", "Live sequence stores (active + cached prefixes).", stats["stores"])
            g("glm53_max_sequence_stores", "Configured store limit (GLM53_SERVE_MAX_STORES).", stats["max_stores"])
            g("glm53_kv_allocated_tokens", "KV tokens allocated to all live stores (incl. idle cached prefixes).", stats["kv_allocated_tokens"])
            g("glm53_kv_active_tokens", "KV tokens allocated to stores of admitted sequences.", stats["kv_active_tokens"])
            g("glm53_kv_budget_tokens", "KV budget in tokens (GLM53_SERVE_KV_TOKENS).", stats["kv_budget_tokens"])
            g("glm53_batched_verify", "1 if multi-sequence batched verification is enabled.", int(bool(stats["batch"])))
        return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- engine client
class Engine:
    """One Unix-socket connection multiplexed by request id. The engine runs up to
    GLM53_SERVE_MAX_SEQS sequences concurrently and queues the rest; a reader thread routes
    {"id",queued|delta|done} lines to per-request queues."""

    def __init__(self, path):
        self.path, self.wlock, self.sock = path, threading.Lock(), None
        self.waiters, self.mlock = {}, threading.Lock()

    def _ensure(self):
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.path)
        self.sock = s
        threading.Thread(target=self._reader, args=(s,), daemon=True).start()

    def _reader(self, s):
        f = s.makefile("r", encoding="utf-8")
        try:
            for line in f:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                with self.mlock:
                    q = self.waiters.get(msg.get("id"))
                if q is not None:
                    q.put(msg)
        except OSError:
            pass
        finally:
            with self.wlock:
                if self.sock is s:
                    self.sock = None
            with self.mlock:
                pending = list(self.waiters.values())
            for q in pending:
                q.put({"done": True, "error": "engine connection lost", "lost": True})

    def _send(self, obj):
        with self.wlock:
            self._ensure()
            self.sock.sendall((json.dumps(obj) + "\n").encode())

    def stats(self, timeout=2.0):
        """Engine-side monitoring snapshot, or None if the engine does not answer in time (it replies
        between scheduling steps; a long prefill chunk can delay it)."""
        rid = "stats-" + uuid.uuid4().hex
        q = queue.Queue()
        with self.mlock:
            self.waiters[rid] = q
        try:
            self._send({"id": rid, "stats": True})
            msg = q.get(timeout=timeout)
            return msg.get("stats")
        except (OSError, queue.Empty):
            return None
        finally:
            with self.mlock:
                self.waiters.pop(rid, None)

    def generate(self, ids, max_new, stop_token_ids, on_delta=None, alive=None, sampling=None, mm=None):
        """Runs one sequence. on_delta(list[int]) -> bool (True = cancel); alive() -> False cancels
        (checked every second, so a client that disconnects during a long prefill is cancelled too)."""
        rid = uuid.uuid4().hex
        q = queue.Queue()
        with self.mlock:
            self.waiters[rid] = q
        try:
            try:
                req = {"id": rid, "prompt_ids": ids, "max_new": max_new, "stop_token_ids": stop_token_ids, **(sampling or {})}
                if mm:
                    req["mm"] = mm
                self._send(req)
            except OSError as e:
                raise ApiError(503, f"engine unavailable: {e}", "server_error")
            cancelled = False
            while True:
                try:
                    msg = q.get(timeout=1.0)
                except queue.Empty:
                    if not cancelled and alive is not None and not alive():
                        cancelled = True
                        try:
                            self._send({"cancel": rid})
                        except OSError:
                            pass
                    continue
                if msg.get("done"):
                    if msg.get("lost"):
                        raise ApiError(503, "engine connection lost", "server_error")
                    if "error" in msg:
                        raise ApiError(400, msg["error"])
                    return msg
                if "delta" in msg and on_delta is not None and not cancelled:
                    if on_delta(msg["delta"]):
                        cancelled = True
                        try:
                            self._send({"cancel": rid})
                        except OSError:
                            pass
        finally:
            with self.mlock:
                self.waiters.pop(rid, None)


# ----------------------------------------------------------------------------- text streaming
class TextStream:
    """Incremental detokenization + reasoning split + stop strings.

    feed(ids) returns (reasoning_delta, content_delta, stop_hit). Text that could still be the
    start of '</think>' or of a stop string is held back until it is disambiguated."""

    def __init__(self, tok, reasoning, stops, stop_ids=()):
        self.tok, self.ids, self.text_len = tok, [], 0
        self.stop_ids = set(stop_ids)
        self.hit_stop_id = False
        self.in_reasoning = reasoning
        self.stops = [s for s in stops if s]
        self.reasoning, self.content = "", ""
        self.buf = ""                  # decoded but not yet classified
        self.stopped = False

    def _holdback(self, s, markers):
        keep = 0
        for m in markers:
            for k in range(min(len(m) - 1, len(s)), 0, -1):
                if s.endswith(m[:k]):
                    keep = max(keep, k)
                    break
        return keep

    def feed(self, ids, final=False):
        if self.stopped:
            return "", "", True
        for t in ids:                                     # cut at the first stop token id (excluded)
            if t in self.stop_ids:
                self.hit_stop_id, final = True, True
                break
            if t not in EOS:
                self.ids.append(t)
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        if not final and text.endswith("�"):          # incomplete UTF-8 sequence
            text = text[:-1]
        new, self.text_len = text[self.text_len:], max(self.text_len, len(text))
        self.buf += new
        r_out, c_out = "", ""
        if self.in_reasoning:
            i = self.buf.find(THINK_CLOSE)
            if i >= 0:
                r_out += self.buf[:i]
                self.buf = self.buf[i + len(THINK_CLOSE):].lstrip("\n")
                self.in_reasoning = False
            else:
                keep = 0 if final else self._holdback(self.buf, [THINK_CLOSE])
                r_out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        if not self.in_reasoning:
            for s in self.stops:
                j = (self.content + self.buf).find(s)
                if j >= 0:
                    j -= len(self.content)
                    if j >= 0:
                        c_out += self.buf[:j]
                    self.buf, self.stopped = "", True
                    break
            if not self.stopped:
                keep = 0 if final else self._holdback(self.buf, self.stops)
                c_out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        self.reasoning += r_out
        self.content += c_out
        if self.hit_stop_id:
            self.stopped = True
        return r_out, c_out, self.stopped


TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


class ToolCalls:
    """GLM-5 tool-call extraction from the content stream (same wire format as the chat template):
    <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
    feed(text) returns (content_delta, [new calls]); text that could be the start of <tool_call> is
    held back. Values are kept as strings when the tool schema says string, otherwise JSON-decoded
    when possible (falls back to the raw string)."""

    def __init__(self, tools):
        self.types = {}
        for t in tools or []:
            f = t.get("function", t) if isinstance(t, dict) else {}
            props = ((f.get("parameters") or {}).get("properties") or {}) if isinstance(f, dict) else {}
            self.types[f.get("name")] = {k: (v or {}).get("type") for k, v in props.items() if isinstance(v, dict)}
        self.buf, self.in_call, self.calls = "", False, []

    def _value(self, name, key, raw):
        typ = self.types.get(name, {}).get(key)
        if typ == "string" or (isinstance(typ, list) and "string" in typ and len(typ) == 1):
            return raw
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    def _parse(self, body):
        k = body.find("<arg_key>")
        name = (body if k < 0 else body[:k]).strip()
        args = {m.group(1).strip(): self._value(name, m.group(1).strip(), m.group(2)) for m in _ARG_RE.finditer(body)}
        call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
        self.calls.append(call)
        return call

    def feed(self, text, final=False):
        self.buf += text
        out, new = "", []
        while True:
            if not self.in_call:
                i = self.buf.find(TOOL_OPEN)
                if i >= 0:
                    out += self.buf[:i]
                    self.buf, self.in_call = self.buf[i + len(TOOL_OPEN):], True
                    continue
                keep = 0 if final else TextStream._holdback(None, self.buf, [TOOL_OPEN])
                out += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                break
            j = self.buf.find(TOOL_CLOSE)
            if j >= 0:
                new.append(self._parse(self.buf[:j]))
                self.buf, self.in_call = self.buf[j + len(TOOL_CLOSE):], False
                continue
            if final and self.buf.strip():         # output ended inside a call (length/stop): keep what parsed
                new.append(self._parse(self.buf))
                self.buf, self.in_call = "", False
            break
        if (self.calls or self.in_call) and not out.strip():
            out = ""                                  # whitespace between/around calls
        return out, new


# ----------------------------------------------------------------------------- application
class App:
    def __init__(self, args):
        self.engine = Engine(args.socket)
        d = Path(args.model_dir)
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda v, ensure_ascii=True, indent=None, **_: json.dumps(v, ensure_ascii=ensure_ascii, indent=indent)
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(jinja2.TemplateError(msg))
        tpl = Path(args.chat_template) if args.chat_template else d / "chat_template.jinja"
        self.template = env.from_string(tpl.read_text())
        self.media = None
        if args.vision == "on":
            if media_mod is None:
                raise SystemExit(f"--vision on needs numpy/Pillow/PyAV (serve/venv): {MEDIA_IMPORT_ERROR}")
            pol = media_mod.Policy(timeout=args.media_timeout, image_max_bytes=args.media_max_bytes, video_max_bytes=args.video_max_bytes,
                                   local_root=args.allowed_local_media_path or "", deny_private=args.media_deny_private,
                                   allowed_domains=tuple(d.strip().lower() for d in (args.allowed_media_domains or "").split(",") if d.strip()))
            opts = media_mod.Opts(max_image_tokens=args.max_image_tokens, max_video_tokens=args.max_video_tokens,
                                  video_fps=args.video_fps, video_max_frames=args.video_max_frames, detail_low_tokens=args.detail_low_tokens)
            self.media = media_mod.Media(media_mod.CanvasStore(args.canvas_dir, args.canvas_cache_mb << 20), pol, opts)
            self.media_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="media")
        print(f"[openai-server] template {tpl}; vision {'on' if self.media else 'off'}", flush=True)
        self.name, self.max_len, self.default_max = args.model_name, args.max_model_len, args.max_tokens
        self.metrics = Metrics(self.name)
        self.max_seqs = int(os.environ.get("GLM53_SERVE_MAX_SEQS", "4"))
        self.default_temperature = float(os.environ.get("GLM53_SERVE_DEFAULT_TEMPERATURE", "0"))
        self.started = int(time.time())
        self.m = {"requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
                  "prefill_ms": 0.0, "decode_ms": 0.0, "prefix_hit_tokens": 0}
        self.mlock = threading.Lock()

    # --- prompts
    @staticmethod
    def _text(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                parts.append(p.get("text", ""))
            elif isinstance(p, dict):
                raise ApiError(400, f"content part type {p.get('type')!r} is not supported (text only)")
        return "".join(parts)

    def _content(self, content, media):
        """Template content of a message: the joined string for text-only content (rendering unchanged), else
        text/image/video parts in order; media parts are collected into `media`."""
        if isinstance(content, dict):
            content = [content]
        if content is None or isinstance(content, str) or not any(
                isinstance(p, dict) and media_mod is not None and media_mod.part_kind(p) for p in content):
            return self._text(content)
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append({"type": "text", "text": p})
                continue
            if not isinstance(p, dict):
                continue
            t, kind = p.get("type"), media_mod.part_kind(p)
            if t in ("text", "input_text"):
                parts.append({"type": "text", "text": p.get("text", "")})
            elif kind in ("image", "video"):
                if self.media is None:
                    raise ApiError(400, "image/video input is disabled on this server (--vision off)")
                media.append(p)
                parts.append({"type": kind})
            elif kind == "audio":
                raise ApiError(400, "this model has no audio encoder; audio input is not supported")
            else:
                raise ApiError(400, f"content part type {t!r} is not supported")
        return parts

    def _load_media(self, parts, body):
        kw = dict(body.get("mm_processor_kwargs") or {})
        futs = [self.media_pool.submit(self.media.load, p, kw) for p in parts]
        items = []
        try:
            for f in futs:
                items.append(f.result())
        except media_mod.MediaError as e:
            raise ApiError(e.code, e.message, param="messages")
        return items

    def chat_prompt(self, body):
        """-> (engine ids, reasoning, media specs, plain ids). Media placeholders are salted in the engine ids
        (crate::vision::MM_BASE..2^24); the plain ids keep <|image|> for /tokenize and counting."""
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ApiError(400, "messages must be a non-empty list", param="messages")
        messages, media = [], []
        for m in msgs:
            mm = dict(m)
            mm["content"] = self._content(m.get("content"), media)
            if mm.get("tool_calls"):
                calls = []
                for tc in mm["tool_calls"]:
                    tc = dict(tc)
                    f = dict(tc.get("function") or {})
                    a = f.get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a) if a.strip() else {}
                        except ValueError:
                            raise ApiError(400, "tool_calls[].function.arguments must be a JSON object string", param="messages")
                    f["arguments"] = a if isinstance(a, dict) else {}
                    tc["function"] = f
                    calls.append(tc)
                mm["tool_calls"] = calls
            messages.append(mm)
        kwargs = dict(body.get("chat_template_kwargs") or {})
        thinking = kwargs.pop("enable_thinking", kwargs.pop("thinking", True))
        if body.get("reasoning_effort") and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = body["reasoning_effort"]
        try:
            text = self.template.render(messages=messages, tools=body.get("tools"),
                                        add_generation_prompt=body.get("add_generation_prompt", True), **kwargs)
        except jinja2.TemplateError as e:
            raise ApiError(400, f"chat template error: {e}")
        if thinking is False and text.endswith(THINK_OPEN):
            text += THINK_CLOSE               # this template always opens <think>
        reasoning = text.endswith(THINK_OPEN)
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if not media:
            return ids, reasoning, [], ids
        t0 = time.time()
        items = self._load_media(media, body)
        try:
            ids, specs, plain = media_mod.expand(ids, items, lambda t: self.tok.encode(t, add_special_tokens=False).ids)
        except media_mod.MediaError as e:
            raise ApiError(e.code, e.message, param="messages")
        specs[0]["prep_ms"] = (time.time() - t0) * 1000     # fetch + decode + canvas (cached items: lookup only)
        return ids, reasoning, specs, plain

    def limits(self, body, n_prompt):
        if int(body.get("n", 1) or 1) != 1:
            raise ApiError(400, "only n=1 is supported", param="n")
        room = self.max_len - n_prompt
        if room <= 0:
            raise ApiError(400, f"prompt has {n_prompt} tokens; maximum context length is {self.max_len}", param="messages")
        mx = body.get("max_completion_tokens") or body.get("max_tokens")
        mx = min(room, self.default_max) if mx is None else int(mx)
        if mx < 1:
            raise ApiError(400, "max_tokens must be at least 1", param="max_tokens")
        if mx > room:
            raise ApiError(400, f"This model's maximum context length is {self.max_len} tokens. However, you requested "
                           f"{n_prompt + mx} tokens ({n_prompt} in the messages, {mx} in the completion).", param="max_tokens")
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        stop_ids = [int(t) for t in (body.get("stop_token_ids") or [])]
        # Sampling: temperature only (exact speculative sampling over the full distribution).
        # top_p / top_k / penalties are accepted and not applied.
        t = body.get("temperature")
        t = self.default_temperature if t is None else float(t)
        if t < 0 or t > 100:
            raise ApiError(400, "temperature must be in [0, 100]", param="temperature")
        seed = body.get("seed")
        samp = {"temperature": t}
        if seed is not None:
            samp["seed"] = int(seed) & 0xFFFFFF
        return mx, stops, stop_ids, samp

    def run(self, ids, max_new, stop_ids, stream_text, on_text=None, alive=None, mm=None):
        """Drives one generation; on_text(reasoning_delta, content_delta) may raise to cancel."""
        stop_ids, samp = stop_ids
        mm_items = [s["item"] for s in (mm or [])]
        engine_mm = [{"path": self.media.store.path(s["item"]), "frames": int(s["item"].canvas.shape[0]),
                      "height": int(s["item"].canvas.shape[1]), "width": int(s["item"].canvas.shape[2]),
                      "segments": s["segments"]} for s in (mm or [])] if mm else None
        state = {"n": 0, "t0": time.time(), "last": None}
        mt = self.metrics
        with mt.lock:
            mt.inflight += 1

        def on_delta(delta):
            now = time.time()
            if delta:
                if state["last"] is None:
                    mt.observe("vllm:time_to_first_token_seconds", now - state["t0"])
                else:                               # a round yields several tokens: spread the interval
                    mt.observe("vllm:inter_token_latency_seconds", (now - state["last"]) / len(delta), len(delta))
                state["last"] = now
            state["n"] += len(delta)
            r, c, stop = stream_text.feed(delta)
            if on_text is not None and (r or c):
                try:
                    on_text(r, c)
                except OSError:
                    return True                # client went away: cancel
            return stop

        if mm_items:
            self.media.store.acquire(mm_items)
        try:
            rec = self.engine.generate(ids, max_new, stop_ids, on_delta, alive, samp, engine_mm)
        except ApiError:
            mt.finished("abort")
            raise
        finally:
            with mt.lock:
                mt.inflight -= 1
            if mm_items:
                self.media.store.release(mm_items)
        out = rec["token_ids"]
        # Deltas already streamed; flush whatever remains (held-back text, final partial UTF-8).
        r, c, _ = stream_text.feed(out[state["n"]:], final=True)
        if on_text is not None and (r or c):
            try:
                on_text(r, c)
            except OSError:
                pass
        if stream_text.stopped or (out and out[-1] in stop_ids) or (out and out[-1] in EOS):
            finish = "stop"
        else:
            finish = "length"
        n_out = len(out)
        if stream_text.stopped and not stream_text.hit_stop_id and stream_text.stops:
            # Count only the tokens up to the stop string (engine cancels at round granularity).
            lo, hi = 0, len(stream_text.ids)
            while lo < hi:
                mid = (lo + hi) // 2
                t = self.tok.decode(stream_text.ids[:mid], skip_special_tokens=False)
                if any(x in t for x in stream_text.stops):
                    hi = mid
                else:
                    lo = mid + 1
            n_out = lo
        elif stream_text.hit_stop_id:
            n_out = len(stream_text.ids) + 1
        usage = {"prompt_tokens": len(ids), "completion_tokens": n_out, "total_tokens": len(ids) + n_out}
        hit = rec.get("prefix_hit_tokens", 0) or 0
        for name, v in (("vllm:prompt_tokens_total", len(ids)), ("vllm:generation_tokens_total", n_out),
                        ("vllm:prefix_cache_queries_total", len(ids)), ("vllm:prefix_cache_hits_total", hit),
                        ("vllm:prompt_tokens_cached_total", hit),
                        ("vllm:spec_decode_num_drafts_total", rec.get("rounds", 0) or 0),
                        ("vllm:spec_decode_num_draft_tokens_total", rec.get("drafted_tokens", 0) or 0),
                        ("vllm:spec_decode_num_accepted_tokens_total", rec.get("accepted_drafts", 0) or 0),
                        ("glm53_requests_total", 1),
                        ("glm53_prefill_seconds_total", (rec.get("prefill_ms", 0.0) or 0.0) / 1000),
                        ("glm53_decode_seconds_total", (rec.get("decode_ms", 0.0) or 0.0) / 1000)):
            mt.add(name, v)
        mt.observe("vllm:e2e_request_latency_seconds", time.time() - state["t0"])
        mt.observe("vllm:request_queue_time_seconds", (rec.get("queue_ms", 0.0) or 0.0) / 1000)
        mt.observe("vllm:request_prefill_time_seconds", (rec.get("prefill_ms", 0.0) or 0.0) / 1000)
        mt.observe("vllm:request_decode_time_seconds", (rec.get("decode_ms", 0.0) or 0.0) / 1000)
        mt.observe("vllm:request_prompt_tokens", len(ids))
        mt.observe("vllm:request_generation_tokens", n_out)
        if rec.get("cancelled"):
            finish_reason = "abort"
        else:
            finish_reason = finish
        mt.finished(finish_reason)
        with self.mlock:
            self.m["requests"] += 1
            self.m["prompt_tokens"] += len(ids)
            self.m["completion_tokens"] += n_out
            self.m["prefill_ms"] += rec.get("prefill_ms", 0.0)
            self.m["decode_ms"] += rec.get("decode_ms", 0.0)
            self.m["prefix_hit_tokens"] += rec.get("prefix_hit_tokens", 0)
        timing = {"engine_id": rec.get("id"), **{k: rec.get(k) for k in ("queue_ms", "prefill_ms", "decode_ms", "total_ms", "prefix_hit_tokens", "rounds", "accepted_drafts", "drafted_tokens", "copy_rounds", "copy_drafted", "copy_accepted", "kv_used_tokens", "kv_active_tokens", "kv_budget_tokens", "mm_encode_ms", "mm_encoded_tokens", "disk_restore_ms")}}
        if mm:
            timing["mm_items"] = [{"kind": s["item"].kind, "canvas": list(s["item"].canvas.shape[:3]), "tokens": s["item"].tokens} for s in mm]
            timing["mm_prep_ms"] = round(mm[0].get("prep_ms", 0.0), 1)
        return out, finish, usage, timing

    def model_card(self):
        return {"id": self.name, "object": "model", "created": self.started, "owned_by": "spark-engine",
                "root": self.name, "parent": None, "max_model_len": self.max_len,
                "permission": [{"id": "modelperm-glm53", "object": "model_permission", "created": self.started,
                                "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": False,
                                "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False,
                                "organization": "*", "group": None, "is_blocking": False}]}


# ----------------------------------------------------------------------------- HTTP
def make_handler(app):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "glm53-openai"

        def log_message(self, fmt, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send(self, code, obj, ctype="application/json"):
            data = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def _error(self, e):
            with app.mlock:
                app.m["errors"] += 1
                app.metrics.add("glm53_errors_total", 1)
            self._send(e.code, {"object": "error", "message": e.message, "type": e.etype, "param": e.param, "code": e.code,
                                "error": {"message": e.message, "type": e.etype, "param": e.param, "code": e.code}})

        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True
            self._sse_open, self._last_write = True, time.time()

        def _sse(self, obj):
            payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()
            self._last_write = time.time()

        def _alive(self):
            """False once the client closed its side (peek returns EOF) or a keepalive write fails."""
            import select
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                if r and self.connection.recv(1, socket.MSG_PEEK) == b"":
                    return False
            except OSError:
                return False
            if getattr(self, "_sse_open", False) and time.time() - getattr(self, "_last_write", 0) > 5:
                try:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    self._last_write = time.time()
                except OSError:
                    return False
            return True

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/health", "/ping"):
                # 200 only once the engine accepts connections: start-tp2.sh starts this front end
                # while the engine is still loading (it binds its socket after warm-up).
                try:
                    with app.engine.wlock:
                        app.engine._ensure()
                except OSError:
                    return self._send(503, b"engine starting", "text/plain")
                return self._send(200, b"", "text/plain")
            if p == "/version":
                return self._send(200, {"version": VERSION})
            if p == "/v1/models":
                return self._send(200, {"object": "list", "data": [app.model_card()]})
            if p.startswith("/v1/models/"):
                return self._send(200, app.model_card())
            if p == "/metrics":
                text = app.metrics.render(app.engine.stats(), app.max_seqs)
                return self._send(200, text.encode(), "text/plain; version=0.0.4; charset=utf-8")
            self._error(ApiError(404, f"route {p} not found", "not_found"))

        def do_POST(self):
            p = self.path.split("?")[0]
            try:
                n = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError as e:
                    raise ApiError(400, f"invalid JSON body: {e}")
                if p == "/v1/chat/completions":
                    return self.chat(body)
                if p == "/v1/completions":
                    return self.completions(body)
                if p == "/tokenize":
                    if "messages" in body:
                        _, _, _, ids = app.chat_prompt(body)
                    else:
                        ids = app.tok.encode(body.get("prompt", ""), add_special_tokens=body.get("add_special_tokens", False)).ids
                    return self._send(200, {"count": len(ids), "max_model_len": app.max_len, "tokens": ids})
                if p == "/detokenize":
                    return self._send(200, {"prompt": app.tok.decode(list(body.get("tokens", [])), skip_special_tokens=False)})
                if p in ("/v1/embeddings", "/v1/audio/transcriptions", "/v1/images/generations", "/pooling", "/score", "/rerank"):
                    raise ApiError(400, f"{p} is not supported by this model server", param=None)
                raise ApiError(404, f"route {p} not found", "not_found")
            except ApiError as e:
                self._error(e)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # --- chat
        def chat(self, body):
            ids, reasoning, mm, _ = app.chat_prompt(body)
            max_new, stops, stop_ids, samp = app.limits(body, len(ids))
            rid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            model = body.get("model") or app.name
            ts = TextStream(app.tok, reasoning, stops, stop_ids)
            tools = body.get("tools") if body.get("tool_choice", "auto") != "none" else None
            tp = ToolCalls(tools) if tools else None
            if not body.get("stream"):
                out, finish, usage, timing = app.run(ids, max_new, (stop_ids, samp), ts, None, self._alive, mm)
                content, calls = ts.content, []
                if tp:
                    content, calls = tp.feed(ts.content, final=True)
                    app.metrics.add("glm53_tool_calls_total", len(calls))
                    if calls and finish == "stop":
                        finish = "tool_calls"
                msg = {"role": "assistant", "content": content if (content or not calls) else None, "tool_calls": calls}
                if reasoning:
                    msg["reasoning_content"] = ts.reasoning
                    msg["reasoning"] = ts.reasoning
                return self._send(200, {"id": rid, "object": "chat.completion", "created": created, "model": model,
                                        "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": finish,
                                                     "stop_reason": None, **({"token_ids": list(out)} if body.get("return_token_ids") else {})}],
                                        "usage": usage, "prompt_logprobs": None, "glm53_timing": timing})
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            self._sse_start()
            base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model}

            def chunk(delta, finish=None):
                c = dict(base)
                c["choices"] = [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]
                return c

            self._sse(chunk({"role": "assistant", "content": ""}))

            ncalls = [0]

            def emit_calls(calls):
                for call in calls:
                    app.metrics.add("glm53_tool_calls_total", 1)
                    self._sse(chunk({"tool_calls": [dict(call, index=ncalls[0])]}))
                    ncalls[0] += 1

            def on_text(r, c):
                if r:
                    self._sse(chunk({"reasoning_content": r, "reasoning": r}))
                if c and tp:
                    c, calls = tp.feed(c)
                    if c:
                        self._sse(chunk({"content": c}))
                    emit_calls(calls)
                elif c:
                    self._sse(chunk({"content": c}))

            try:
                out, finish, usage, timing = app.run(ids, max_new, (stop_ids, samp), ts, on_text, self._alive, mm)
                if tp:
                    c, calls = tp.feed("", final=True)
                    if c:
                        self._sse(chunk({"content": c}))
                    emit_calls(calls)
                    if ncalls[0] and finish == "stop":
                        finish = "tool_calls"
                self._sse(chunk({}, finish))
                if include_usage:
                    u = dict(base)
                    u["choices"], u["usage"] = [], usage
                    self._sse(u)
                self._sse("[DONE]")
            except ApiError as e:
                self._sse({"error": {"message": e.message, "type": e.etype, "code": e.code}})
                self._sse("[DONE]")

        # --- completions
        def completions(self, body):
            prompt = body.get("prompt")
            if isinstance(prompt, str):
                prompts = [prompt]
            elif isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
                prompts = [prompt]
            elif isinstance(prompt, list) and prompt:
                prompts = prompt
            else:
                raise ApiError(400, "prompt must be a string, a list of strings, or token ids", param="prompt")
            if body.get("stream") and len(prompts) > 1:
                raise ApiError(400, "streaming supports a single prompt", param="prompt")
            rid, created = "cmpl-" + uuid.uuid4().hex, int(time.time())
            model = body.get("model") or app.name
            encoded = [(pr, app.tok.encode(pr, add_special_tokens=False).ids if isinstance(pr, str) else list(pr)) for pr in prompts]
            for _, ids in encoded:
                if not ids:
                    raise ApiError(400, "prompt is empty", param="prompt")
            if not body.get("stream"):
                choices, total = [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                timing = None
                for i, (pr, ids) in enumerate(encoded):
                    max_new, stops, stop_ids, samp = app.limits(body, len(ids))
                    ts = TextStream(app.tok, False, stops, stop_ids)
                    out, finish, usage, timing = app.run(ids, max_new, (stop_ids, samp), ts, None, self._alive)
                    text = ts.content
                    if body.get("echo"):
                        text = (pr if isinstance(pr, str) else app.tok.decode(ids, skip_special_tokens=False)) + text
                    choices.append({"index": i, "text": text, "logprobs": None, "finish_reason": finish, "stop_reason": None})
                    if body.get("return_token_ids"):
                        choices[-1]["token_ids"] = list(out)
                    for k in total:
                        total[k] += usage[k]
                return self._send(200, {"id": rid, "object": "text_completion", "created": created, "model": model,
                                        "choices": choices, "usage": total, "glm53_timing": timing})
            pr, ids = encoded[0]
            max_new, stops, stop_ids, samp = app.limits(body, len(ids))
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            ts = TextStream(app.tok, False, stops, stop_ids)
            self._sse_start()

            def chunk(text, finish=None):
                return {"id": rid, "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish, "stop_reason": None}]}

            if body.get("echo"):
                self._sse(chunk(pr if isinstance(pr, str) else app.tok.decode(ids, skip_special_tokens=False)))
            try:
                out, finish, usage, timing = app.run(ids, max_new, (stop_ids, samp), ts, lambda r, c: c and self._sse(chunk(c)), self._alive)
                self._sse(chunk("", finish))
                if include_usage:
                    self._sse({"id": rid, "object": "text_completion", "created": created, "model": model,
                               "choices": [], "usage": usage})
                self._sse("[DONE]")
            except ApiError as e:
                self._sse({"error": {"message": e.message, "type": e.etype, "code": e.code}})
                self._sse("[DONE]")

    return H


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--socket", default="/tmp/glm53-serve.sock")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--model-name", default="GLM-5.3-Flash-EXL3")
    p.add_argument("--max-model-len", type=int, default=1048576, help="must match the engine KV budget (GLM53_SERVE_KV_TOKENS)")
    p.add_argument("--max-tokens", type=int, default=8192, help="default max_tokens when the request omits it")
    p.add_argument("--chat-template", default=None, help="chat template file (default: the model's; serve/chat_template_mm.jinja adds media)")
    p.add_argument("--vision", choices=["on", "off"], default="on", help="image/video input (engine GLM53_VISION must match)")
    p.add_argument("--max-image-tokens", type=int, default=8000, help="per-image token cap (model default 8000)")
    p.add_argument("--max-video-tokens", type=int, default=65536, help="default per-video token budget (clients may raise it up to 240000)")
    p.add_argument("--detail-low-tokens", type=int, default=1024, help="token cap for image_url.detail=low")
    p.add_argument("--video-fps", type=float, default=2.0)
    p.add_argument("--video-max-frames", type=int, default=2048)
    p.add_argument("--media-timeout", type=float, default=30.0, help="seconds per remote media fetch")
    p.add_argument("--media-max-bytes", type=int, default=64 << 20, help="image size limit")
    p.add_argument("--video-max-bytes", type=int, default=2 << 30, help="video size limit")
    p.add_argument("--allowed-local-media-path", default="", help="allow file:// and paths below this directory")
    p.add_argument("--allowed-media-domains", default="", help="comma-separated host allow-list for remote media (empty = any)")
    p.add_argument("--media-deny-private", action="store_true", help="also refuse private-network media hosts")
    p.add_argument("--canvas-dir", default="/dev/shm/glm53-mm")
    p.add_argument("--canvas-cache-mb", type=int, default=1024, help="LRU of decoded canvases (in-flight ones are always kept)")
    args = p.parse_args()
    # listen backlog: socketserver default is 5; bursts of 8+ concurrent clients were reset (bench/r26)
    ThreadingHTTPServer.request_queue_size = 128
    server = ThreadingHTTPServer((args.host, args.port), make_handler(App(args)))
    server.daemon_threads = True
    print(f"[openai-server] http://{args.host}:{args.port}/v1 -> {args.socket}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
