#!/usr/bin/env python3
"""OpenAI-compatible front end for `spark-engine qwen-serve` (Qwen3.8-Flash-Next, single GPU).

Reuses openai_server.py (HTTP, SSE, metrics, stop strings, reasoning split) and swaps the model specifics:
EOS ids, the chat template call (enable_thinking / reasoning_effort go to the template, which opens
"<think>\\n" itself), and the qwen3_coder tool-call format:
  <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n</function>\\n</tool_call>
Images and videos (--vision on, the engine needs QWEN_VISION=1): mm.py with the Qwen geometry and placeholders of
mm_qwen.py; the template's <|image_pad|> / <|video_pad|> become salted runs the engine fills from the canvases.

Usage:
  python3 qwen_server.py --socket /tmp/qwen38-serve.sock --model-dir <dir> --port 8888
"""
import json, re, sys, time, uuid

import jinja2
import openai_server as o
if o.media_mod is not None:
    import mm_qwen

o.EOS[:] = [248046, 248044]                     # <|im_end|>, <|endoftext|>
o.VERSION = "spark-engine qwen3.8 (openai-compatible front end)"
_FN = re.compile(r"<function=([^>\n]+)>")
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


class QwenToolCalls(o.ToolCalls):
    def _parse(self, body):
        m = _FN.search(body)
        name = m.group(1).strip() if m else body.strip()
        args = {p.group(1).strip(): self._value(name, p.group(1).strip(), p.group(2)) for p in _PARAM.finditer(body)}
        call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
        self.calls.append(call)
        return call


# The template knows xhigh (default), medium and low; the common API levels map onto them, "none" turns thinking off.
_EFFORT = {"none": None, "off": None, "minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}


class QwenApp(o.App):
    def chat_prompt(self, body):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise o.ApiError(400, "messages must be a non-empty list", param="messages")
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
                            raise o.ApiError(400, "tool_calls[].function.arguments must be a JSON object string", param="messages")
                    f["arguments"] = a if isinstance(a, dict) else {}
                    tc["function"] = f
                    calls.append(tc)
                mm["tool_calls"] = calls
            messages.append(mm)
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if "thinking" in kwargs and "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = kwargs.pop("thinking")
        # thinking level: reasoning_effort (OpenAI), reasoning.effort (Responses / OpenRouter), thinking.type (Anthropic style)
        effort = kwargs.pop("reasoning_effort", None) or body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort")
        th = body.get("thinking")
        if isinstance(th, dict) and th.get("type") == "disabled":
            kwargs.setdefault("enable_thinking", False)
        if effort is not None:
            e = str(effort).strip().lower()
            if e not in _EFFORT:
                raise o.ApiError(400, f"reasoning_effort {effort!r} is not supported (none, minimal, low, medium, high, xhigh, max)", param="reasoning_effort")
            if _EFFORT[e] is None:
                kwargs["enable_thinking"] = False
            else:
                kwargs["reasoning_effort"] = _EFFORT[e]
        try:
            text = self.template.render(messages=messages, tools=body.get("tools"),
                                        add_generation_prompt=body.get("add_generation_prompt", True), **kwargs)
        except jinja2.TemplateError as e:
            raise o.ApiError(400, f"chat template error: {e}")
        reasoning = text.rstrip("\n").endswith(o.THINK_OPEN)
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if not media:
            return ids, reasoning, [], ids
        t0 = time.time()
        items = self._load_media(media, body)
        try:
            ids, specs, plain = o.media_mod.expand(ids, items, lambda t: self.tok.encode(t, add_special_tokens=False).ids)
        except o.media_mod.MediaError as e:
            raise o.ApiError(e.code, e.message, param="messages")
        specs[0]["prep_ms"] = (time.time() - t0) * 1000
        return ids, reasoning, specs, plain


_REQLOG = __import__("os").environ.get("QWEN_REQLOG", "/tmp/qwen38-requests.jsonl")   # "" = off
_tl = __import__("threading").local()


class LoggedApp(QwenApp):
    """Appends one JSON line per request to QWEN_REQLOG: sizes, prefix hit, queue / prefill / TTFT, and where the prompt
    first differs from the closest earlier sequence (its prompt + output), with the text on both sides of that point."""
    seen = __import__("collections").deque(maxlen=64)

    def chat_prompt(self, body):
        r = super().chat_prompt(body)
        msgs = body.get("messages") or []
        _tl.info = {"msgs": len(msgs), "roles": "".join((m.get("role") or "?")[0] for m in msgs if isinstance(m, dict))[-12:],
                    "tools": len(body.get("tools") or []), "effort": body.get("reasoning_effort") or (body.get("chat_template_kwargs") or {}).get("reasoning_effort"),
                    "thinking": (body.get("chat_template_kwargs") or {}).get("enable_thinking"),
                    "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens"), "stream": bool(body.get("stream")),
                    "asst_reasoning_echoed": sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant" and (m.get("reasoning_content") or m.get("reasoning"))),
                    "asst": sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant")}
        return r

    def run(self, ids, max_new, stop_ids, stream_text, on_text=None, alive=None, mm=None):
        t0 = time.time()
        first = {}
        def wrap(r, c):
            first.setdefault("t", time.time())
            if on_text is not None:
                on_text(r, c)
        out, finish, usage, timing = super().run(ids, max_new, stop_ids, stream_text, wrap, alive, mm)
        if _REQLOG:
            try:
                best, bi = 0, None
                for k, prev in enumerate(self.seen):
                    n = 0
                    lim = min(len(prev), len(ids))
                    while n < lim and prev[n] == ids[n]:
                        n += 1
                    if n > best:
                        best, bi = n, k
                rec = {"t": time.strftime("%H:%M:%S"), **getattr(_tl, "info", {}), "prompt": len(ids), "max_new": max_new, "out": len(out),
                       "finish": finish, "hit": timing.get("prefix_hit_tokens"), "queue_ms": round(timing.get("queue_ms") or 0),
                       "prefill_ms": round(timing.get("prefill_ms") or 0), "ttft_ms": round(((first.get("t") or time.time()) - t0) * 1000),
                       "lcp": best}
                if bi is not None and best < len(ids):
                    prev = self.seen[bi]
                    rec["lcp_prev_len"] = len(prev)
                    rec["diverge_prev"] = self.tok.decode(prev[max(0, best - 30):best + 50], skip_special_tokens=False)
                    rec["diverge_now"] = self.tok.decode(ids[max(0, best - 30):best + 50], skip_special_tokens=False)
                self.seen.append(list(ids) + list(out))
                with open(_REQLOG, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:      # logging must never fail a request
                print("[reqlog]", e, flush=True)
        return out, finish, usage, timing


o.ToolCalls = QwenToolCalls
o.App = LoggedApp

if __name__ == "__main__":
    defaults = {"--socket": "/tmp/qwen38-serve.sock", "--model-name": "Qwen3.8-Flash-Next", "--vision": "on",
                "--max-model-len": "1048544", "--max-tokens": "8192", "--canvas-dir": "/dev/shm/qwen38-mm",
                "--max-image-tokens": "16384", "--max-video-tokens": "12288", "--video-max-frames": "768"}
    for k, v in defaults.items():
        if k not in sys.argv:
            sys.argv += [k, v]
    o.main()
