#!/usr/bin/env python3
"""OpenAI-compatible front end for `spark-engine ds41-serve` (DeepSeek-V4.1-Flash, TP3 over three Sparks).

Reuses openai_server.py (HTTP, SSE, metrics, stop strings, reasoning split) and swaps the model specifics:
- prompts come from the checkpoint's own `encoding/encoding.py` (encode_messages: multi-turn, tools, thinking mode,
  numeric reasoning effort, mid-conversation system messages), so the format follows the official reference exactly;
- tool calls are parsed from the V4.1 DSML blocks:
    <｜DSML｜ calls>\\n<｜DSML｜ invoke name="NAME">\\n<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>...
- request conversion follows the official API (deepseek-ai/deepseek-recipe): reasoning effort levels map onto the 1-100
  budget like the official API (minimal/low 50, medium/high/xhigh 75, max/ultra 100; "none" turns thinking off, an
  integer 1-100 passes through), a top-level `thinking: {type: enabled|disabled|adaptive}` decides the mode, thinking
  mode samples at temperature 1.0, the default max_tokens is 8K / 64K thinking / 128K at max effort, tool_choice
  none / required / a named function, response_format json_object / json_schema, tool schemas normalized to
  {name, description, parameters};
- control tokens come only from the template: every added-token string inside request text (content, reasoning,
  tool arguments and schemas) is encoded as ordinary text, so e.g. a file containing "</think>" or DSML read by an
  agent cannot open turns or tool blocks;
- images (--vision on, the engine needs DS41_VISION=1): content image parts go to encode_messages as image blocks (one
  <｜deepseek_image｜> each, in prompt order), mm.py loads them with the DS41 geometry of mm_ds41.py and the placeholders
  become salted spans the engine fills from the canvases. No video.

Usage:
  python3 ds41_server.py --socket /tmp/ds41-serve.sock --model-dir /models/DeepSeek-V4.1-Flash-1017 --port 8888
"""
import json, os, re, sys, time, uuid

from tokenizers import Tokenizer

import openai_server as o
if o.media_mod is not None:
    import mm_ds41  # noqa: F401  (patches mm with the DS41 geometry)

o.VERSION = "spark-engine deepseek-v4.1-flash (openai-compatible front end)"

_ENC = None


def encoding_module(model_dir):
    global _ENC
    if _ENC is None:
        sys.path.insert(0, os.path.join(model_dir, "encoding"))
        import encoding as enc
        _ENC = enc
    return _ENC


_DSML = "｜DSML｜"
# Output parsing knows the control tokens by id, as the official parser does: the ｜DSML｜ and </think> tokens the model
# generates decode to these stand-ins (DsTextStream), so the same text spelled with ordinary tokens (a file being
# quoted, a model repeating markup) stays literal content instead of ending the reasoning or opening a tool call.
_S_DSML, _S_THINK = "\U000FFFF0", "\U000FFFF1"
_INVOKE = re.compile(r'<' + _S_DSML + r' invoke name="([^"]*)">\n?(.*?)</' + _S_DSML + r' invoke>', re.S)
_PARAM = re.compile(r'<' + _S_DSML + r' parameter name="([^"]*)"(?: string="(true|false)")?>(.*?)</' + _S_DSML + r' parameter>', re.S)


def _literal(text, drop_think=False):
    """Stand-ins back to the tokens' text; a stray </think> in the content is dropped (as the official parser does)."""
    return text.replace(_S_DSML, _DSML).replace(_S_THINK, "" if drop_think else "</think>")


_SENT = 0xF0000   # supplementary private-use plane: one stand-in character per added token


class SafeEncoder:
    """Prompt text -> ids where only template-inserted special tokens become control tokens.

    Request strings are escaped before rendering (each added-token string -> one private-use stand-in character),
    so after encode_messages every added-token string left in the prompt was put there by the template. The prompt is
    split at those, they map to their ids, and the text between them (stand-ins restored) is encoded with added-token
    matching off, i.e. as ordinary BPE text. The pre-tokenizer splits at added tokens anyway, so prompts without such
    strings in request text encode exactly as before (tested by bench/ds41/prompt_encode_test.py)."""

    def __init__(self, tok_path):
        with open(tok_path) as f:
            spec = json.load(f)
        added = sorted((t["id"], t["content"]) for t in spec["added_tokens"])
        # the same tokenizer without its added tokens: text in, base BPE out (encode_special_tokens covers only the
        # tokens flagged special, not <｜User｜>, <think>, ｜DSML｜ ...)
        spec["added_tokens"] = []
        self.tok = Tokenizer.from_str(json.dumps(spec))
        self.ids = {c: i for i, c in added}
        names = sorted(self.ids, key=len, reverse=True)
        self.pat = re.compile("|".join(map(re.escape, names)))
        self.esc = {n: chr(_SENT + k) for k, n in enumerate(names)}
        self.unesc = {v: k for k, v in self.esc.items()}
        self.unpat = re.compile("[" + chr(_SENT) + "-" + chr(_SENT + len(names) - 1) + "]")

    def escape(self, v):
        """Strings (dict keys too) inside a request structure with the added-token strings replaced by stand-ins."""
        if isinstance(v, str):
            return self.pat.sub(lambda m: self.esc[m.group()], v) if v else v
        if isinstance(v, list):
            return [self.escape(x) for x in v]
        if isinstance(v, dict):
            return {self.escape(k): self.escape(x) for k, x in v.items()}
        return v

    def unescape(self, s):
        return self.unpat.sub(lambda m: self.unesc[m.group()], s)

    def encode(self, text):
        segs, specials, pos = [], [], 0
        for m in self.pat.finditer(text):
            segs.append(text[pos:m.start()])
            specials.append(self.ids[m.group()])
            pos = m.end()
        segs.append(text[pos:])
        enc = self.tok.encode_batch([self.unescape(x) for x in segs], add_special_tokens=False)
        ids = []
        for k, e in enumerate(enc):
            ids.extend(e.ids)
            if k < len(specials):
                ids.append(specials[k])
        return ids


class DsTextStream(o.TextStream):
    """TextStream with the generated ｜DSML｜ / </think> tokens decoded to stand-ins: the reasoning ends at the real
    </think> token only; reasoning (and content without a tool parser) is handed out with the tokens' text again."""

    def __init__(self, tok, reasoning, stops, stop_ids, ctrl, keep_dsml):
        super().__init__(tok, reasoning, stops, stop_ids)
        self.ctrl, self.keep_dsml = ctrl, keep_dsml
        self.think_close = _S_THINK

    def _decode(self, ids):
        parts, seg = [], []
        for t in ids:
            s = self.ctrl.get(t)
            if s is None:
                seg.append(t)
                continue
            if seg:
                parts.append(self.tok.decode(seg, skip_special_tokens=False))
                seg = []
            parts.append(s)
        if seg:
            parts.append(self.tok.decode(seg, skip_special_tokens=False))
        return "".join(parts)

    def feed(self, ids, final=False):
        r0, c0 = len(self.reasoning), len(self.content)
        r, c, stop = super().feed(ids, final)
        r = _literal(r)
        self.reasoning = self.reasoning[:r0] + r
        if not self.keep_dsml:       # with a tool parser the content keeps the stand-ins until DsToolCalls
            c = _literal(c, drop_think=True)
            self.content = self.content[:c0] + c
        return r, c, stop


class DsToolCalls(o.ToolCalls):
    """DSML tool-call extraction: the content stream is cut at the calls block, each invoke becomes one call."""

    OPEN, CLOSE = "<" + _S_DSML + " calls>", "</" + _S_DSML + " calls>"

    def __init__(self, tools, in_call=False):
        super().__init__(tools)
        self.in_call = in_call    # tool_choice required: the prompt already opened the calls block
        self.done = False         # the official parser drops all output after the calls block

    def _parse_block(self, body):
        out = []
        for m in _INVOKE.finditer(body):
            name, inner = m.group(1), m.group(2)
            args = {}
            for p in _PARAM.finditer(inner):
                key, is_str, raw = p.group(1), p.group(2), _literal(p.group(3))
                if is_str == "true":
                    args[key] = raw
                else:
                    try:
                        args[key] = json.loads(raw)
                    except ValueError:
                        args[key] = self._value(name, key, raw)
            call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
            self.calls.append(call)
            out.append(call)
        return out

    def feed(self, text, final=False):
        if self.done:
            return "", []
        self.buf += text
        out, new = "", []
        while True:
            if self.done:
                self.buf = ""
                break
            if not self.in_call:
                i = self.buf.find(self.OPEN)
                if i >= 0:
                    # the official encoding puts "\n\n" between an assistant's content and its calls block: the model
                    # generates content + "\n\n" + block, so the content handed back must not keep that separator,
                    # or the client's next prompt re-encodes it twice and departs from the generated tokens there
                    # (the engine then cannot resume the stored sequence and recomputes the whole previous output)
                    pre = self.buf[:i]
                    out += pre[:-2] if pre.endswith("\n\n") else pre
                    self.buf, self.in_call = self.buf[i + len(self.OPEN):], True
                    continue
                keep = 0 if final else o.TextStream._holdback(None, self.buf, [self.OPEN])
                if not final:   # trailing newlines may be that separator: held until the next text shows what follows
                    rest = self.buf[:len(self.buf) - keep]
                    keep += len(rest) - len(rest.rstrip("\n"))
                out += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                break
            j = self.buf.find(self.CLOSE)
            if j >= 0:
                new.extend(self._parse_block(self.buf[:j]))
                self.buf, self.in_call, self.done = "", False, True
                continue
            if final and self.buf.strip():
                new.extend(self._parse_block(self.buf))
                self.buf, self.in_call = "", False
            break
        if (self.calls or self.in_call) and not out.strip():
            out = ""
        return _literal(out, drop_think=True), new


_REF = "mm-part:"   # image blocks handed to encode_messages refer to the request's image parts by index
# reasoning effort levels -> the 1-100 budget, as the official API maps them (deepseek-recipe protocol/reasoning.rs +
# v4/dsv41.rs: minimal/low -> Low 50, medium/high -> High 75, xhigh -> Xhigh 75, ultra/max -> Max 100)
_EFFORT = {"minimal": 50, "low": 50, "medium": 75, "high": 75, "xhigh": 75, "ultra": 100, "max": 100}
_MSG_KEYS = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")   # the rest is dropped:
# encode_messages also honours keys that insert control tokens (task, wo_eos, content_blocks, tools, response_format)
_ROLES = ("system", "developer", "user", "assistant", "tool", "latest_reminder")


class DsApp(o.App):
    def __init__(self, args):
        super().__init__(args)
        # requests without "temperature" sample with the official recommendation for DeepSeek-V4.1-Flash(-1017)
        # (temperature 1.0, top_p 1.0; top_p 1.0 is the full distribution, which is what the engine samples);
        # greedy decoding (the shared front end's default 0) fell into endless repetition in long agent sessions
        self.default_temperature = float(os.environ.get("DS41_DEFAULT_TEMPERATURE", "1.0"))
        # thinking mode samples at this temperature whatever the request asks (the official API fixes 1.0 there);
        # empty = use the request's temperature
        t = os.environ.get("DS41_THINKING_TEMPERATURE", "1.0").strip()
        self.thinking_temperature = float(t) if t else None
        self.safe = SafeEncoder(os.path.join(args.model_dir, "tokenizer.json"))
        if os.environ.get("DS41_PROMPT_LOG_DIR"):
            # with the prompt log: also the generated token ids (<time>-<prompt tokens>.out.json), to find where the
            # client's next prompt departs from what was generated (the engine resumes a stored sequence only from its end)
            gen = self.engine.generate

            def logged(ids, *a, **k):
                rec = gen(ids, *a, **k)
                try:
                    d = os.environ["DS41_PROMPT_LOG_DIR"]
                    with open(os.path.join(d, f"{time.time():.3f}-{len(ids)}.out.json"), "w") as f:
                        json.dump({"n_prompt": len(ids), "out_ids": rec.get("token_ids", [])}, f)
                except OSError as e:
                    print(f"[ds41-server] output log: {e}", flush=True)
                return rec
            self.engine.generate = logged
        self.enc = encoding_module(args.model_dir)
        eos = self.tok.token_to_id(self.enc.eos_token)
        o.EOS[:] = [eos]
        o.THINK_OPEN, o.THINK_CLOSE = self.enc.thinking_start_token, self.enc.thinking_end_token
        self.ctrl = {self.tok.token_to_id(_DSML): _S_DSML, self.tok.token_to_id(self.enc.thinking_end_token): _S_THINK}

    @staticmethod
    def _mode(body):
        """-> (thinking, effort 1-100 or None) as the official API resolves them: an explicit thinking setting decides
        the mode (top-level `thinking: {type}`, else chat_template_kwargs thinking / enable_thinking), else effort
        "none" turns it off; thinking without an effort uses high (75)."""
        kwargs = body.get("chat_template_kwargs") or {}
        effort = body.get("reasoning_effort", kwargs.get("reasoning_effort"))
        th = body.get("thinking")
        if isinstance(th, dict) and th.get("type") is not None:
            if th["type"] not in ("enabled", "adaptive", "disabled"):
                raise o.ApiError(400, f"unknown thinking type {th['type']!r}", param="thinking")
            thinking = th["type"] != "disabled"
        elif "enable_thinking" in kwargs or "thinking" in kwargs:
            thinking = kwargs.get("enable_thinking", kwargs.get("thinking")) is not False
        else:
            thinking = not (isinstance(effort, str) and effort.lower() in ("none", "off"))
        if not thinking or effort is None or (isinstance(effort, str) and effort.lower() in ("none", "off")):
            return thinking, (75 if thinking else None)
        if isinstance(effort, str) and not effort.strip().isdigit():
            if effort.lower() not in _EFFORT:
                raise o.ApiError(400, f"unknown reasoning_effort {effort!r}", param="reasoning_effort")
            return True, _EFFORT[effort.lower()]
        try:
            return True, max(1, min(100, int(effort)))
        except (TypeError, ValueError):
            raise o.ApiError(400, f"invalid reasoning_effort {effort!r}", param="reasoning_effort")

    @staticmethod
    def _tools(body, thinking):
        """-> (tools for the prompt, tool_choice required). Each tool normalized to {name, description, parameters} like
        the official API (deepseek-recipe v4/mod.rs tool_definition_json), defer_loading tools left out, a named
        tool_choice keeps only that tool; required / named choices are refused in thinking mode (as officially)."""
        choice = body.get("tool_choice", "auto")
        tools = body.get("tools") or []
        if choice == "none" or not tools:
            return None, False
        out, names = [], set()
        for k, t in enumerate(tools):
            f = t.get("function") if isinstance(t, dict) else None
            if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]:
                raise o.ApiError(400, f"tools[{k}] must be a function tool with a name", param="tools")
            if f["name"] in names:
                raise o.ApiError(400, "tool names must be unique", param="tools")
            names.add(f["name"])
            if t.get("defer_loading") or f.get("defer_loading"):
                continue
            params = f.get("parameters")
            if params is None:
                params = {}
            elif not isinstance(params, dict) or params.get("type") != "object":
                raise o.ApiError(400, f"Invalid schema for function '{f['name']}': schema must be a JSON Schema of "
                                 "'type: \"object\"'", param="tools")
            tool = {"type": "function", "function": {"name": f["name"], "description": f.get("description") or "",
                                                     "parameters": params}}
            if t.get("namespace") is not None:
                tool["namespace"] = t["namespace"]
            out.append(tool)
        required = False
        if isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name")
            if name not in names:
                raise o.ApiError(400, f"tool_choice names an undeclared tool {name!r}", param="tool_choice")
            out, required = [t for t in out if t["function"]["name"] == name], True
        elif choice == "required":
            required = True
        elif choice != "auto":
            raise o.ApiError(400, f"unknown tool_choice {choice!r}", param="tool_choice")
        if required and thinking:
            raise o.ApiError(400, "Thinking mode does not support this tool_choice", param="tool_choice")
        return out or None, required and bool(out)

    def chat_prompt(self, body):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise o.ApiError(400, "messages must be a non-empty list", param="messages")
        thinking, effort = self._mode(body)
        tools, required = self._tools(body, thinking)
        # OpenAI format throughout: the official render_message converts tools and tool_calls itself, and
        # encode_messages merges role "tool" results into user turns (merge_tool_messages)
        messages, media = [], []
        for k, m in enumerate(msgs):
            if not isinstance(m, dict) or m.get("role") not in _ROLES:
                raise o.ApiError(400, f"messages[{k}]: unknown role {m.get('role') if isinstance(m, dict) else m!r}", param="messages")
            mm = {key: m[key] for key in _MSG_KEYS if m.get(key) is not None}
            if mm["role"] == "developer":
                mm["role"] = "system"
            mm["content"] = self._blocks(m.get("content"), media)
            if mm.get("tool_calls"):
                calls = []
                for c in mm["tool_calls"]:
                    c = dict(c, function=dict(c.get("function") or {}))
                    if not isinstance(c["function"].get("arguments"), str):
                        c["function"]["arguments"] = json.dumps(c["function"].get("arguments") or {}, ensure_ascii=False)
                    calls.append(c)
                mm["tool_calls"] = calls
            messages.append(mm)
        # every string from the request is escaped: added-token strings in it stay ordinary text (SafeEncoder)
        messages = self.safe.escape(messages)
        if messages[-1]["role"] == "assistant" and msgs[-1].get("prefix"):
            messages[-1]["wo_eos"] = True          # continue the given assistant prefix (official "prefix": true)
        rf = body.get("response_format") or {}
        fmt = None
        if rf.get("type") == "json_object":
            if not any("json" in json.dumps(m.get("content"), ensure_ascii=False).lower() for m in msgs):
                raise o.ApiError(400, "Prompt must contain the word 'json' in some form to use 'response_format' of type "
                                 "'json_object'.", param="response_format")
            fmt = {"type": "json_object"}
        elif rf.get("type") == "json_schema":
            fmt = (rf.get("json_schema") or {}).get("schema")
            if not isinstance(fmt, dict):
                raise o.ApiError(400, "response_format.json_schema.schema must be an object", param="response_format")
            fmt = self.safe.escape(fmt)
        elif rf.get("type") not in (None, "text"):
            raise o.ApiError(400, f"response_format type {rf.get('type')!r} is not supported", param="response_format")
        if tools or fmt:
            if messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": ""})
            if tools:
                messages[0]["tools"] = self.safe.escape(tools)
            if fmt:
                messages[0]["response_format"] = fmt
        mode = "thinking" if thinking else "chat"
        try:
            text, mmd = self.enc.encode_messages(messages, thinking_mode=mode, reasoning_effort=effort, return_multi_modal_data=True)
        except (AssertionError, ValueError, KeyError, TypeError, NotImplementedError) as e:
            raise o.ApiError(400, f"prompt encoding error: {e}")
        if required:     # tool_choice required / named: the official prompt opens the calls block (dsv41.rs)
            text += "\n\n<" + _DSML + " calls>\n"
        reasoning = text.endswith(o.THINK_OPEN)
        ids = self.safe.encode(text)
        body["_ds41"] = {"thinking": thinking, "effort": effort, "in_call": required}
        self._log_prompt(body, text, ids)
        if not media:
            return ids, reasoning, [], ids
        # the images in prompt order (their blocks carry the index of the content part)
        order = [int(r.get("url", "")[len(_REF):]) for r in mmd["images"] if str(r.get("url", "")).startswith(_REF)]
        if sorted(order) != list(range(len(media))):
            raise o.ApiError(400, "the prompt encoding dropped or reordered an image", param="messages")
        t0 = time.time()
        items = self._load_media([media[i] for i in order], body)
        try:
            ids, specs, plain = o.media_mod.expand(ids, items, lambda t: self.tok.encode(t, add_special_tokens=False).ids)
        except o.media_mod.MediaError as e:
            raise o.ApiError(e.code, e.message, param="messages")
        specs[0]["prep_ms"] = (time.time() - t0) * 1000
        return ids, reasoning, specs, plain

    def tool_parser(self, tools, body):
        return DsToolCalls(tools, in_call=(body.get("_ds41") or {}).get("in_call", False))

    def text_stream(self, reasoning, stops, stop_ids, tools_parsed=False):
        return DsTextStream(self.tok, reasoning, stops, stop_ids, self.ctrl, tools_parsed)

    def limits(self, body, n_prompt):
        mode = body.get("_ds41")     # chat requests (set by chat_prompt); /v1/completions has none
        if mode and body.get("max_completion_tokens") is None and body.get("max_tokens") is None:
            # the official defaults: 8K without thinking, 64K thinking, 128K at max effort (lowered to the room left)
            body = dict(body, max_tokens=max(1, min(self.max_len - n_prompt,
                        (128 << 10) if mode["effort"] == 100 else (64 << 10) if mode["thinking"] else (8 << 10))))
        if mode and mode["thinking"] and self.thinking_temperature is not None:
            body = dict(body, temperature=self.thinking_temperature)
        mx, stops, stop_ids, samp = super().limits(body, n_prompt)
        # sampling without a seed: a fresh one per request (the engine's noise is keyed by (seed, position), so a
        # fixed default would give every retry of the same prompt the same sample)
        if samp["temperature"] > 0 and "seed" not in samp:
            samp["seed"] = int.from_bytes(os.urandom(3), "little")
        return mx, stops, stop_ids, samp

    def _log_prompt(self, body, text, ids):
        """DS41_PROMPT_LOG_DIR (off by default; debugging prefix-cache misses): every encoded prompt as JSON (text,
        token ids, the request's message roles), the newest DS41_PROMPT_LOG_KEEP (200) kept."""
        d = os.environ.get("DS41_PROMPT_LOG_DIR")
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
            name = os.path.join(d, f"{time.time():.3f}-{len(ids)}.json")
            with open(name, "w") as f:
                json.dump({"time": time.time(), "n_tokens": len(ids), "roles": [m.get("role") for m in body.get("messages") or []],
                           "sampling": {k: body.get(k) for k in ("temperature", "top_p", "seed", "max_tokens", "reasoning_effort")},
                           "n_tools": len(body.get("tools") or []), "text": text, "ids": ids}, f, ensure_ascii=False)
            old = sorted(os.listdir(d))[:-int(os.environ.get("DS41_PROMPT_LOG_KEEP", "200"))]
            for x in old:
                os.remove(os.path.join(d, x))
        except OSError as e:
            print(f"[ds41-server] prompt log: {e}", flush=True)

    def _blocks(self, content, media):
        """Message content for encode_messages: the joined text for text-only content (as before), else text and image
        blocks in order; each image part is collected into `media` and its block refers to it by index."""
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list) or o.media_mod is None or not any(isinstance(p, dict) and o.media_mod.part_kind(p) for p in content):
            return self._text(content)
        blocks = []
        for p in content:
            if isinstance(p, str):
                blocks.append({"type": "text", "text": p})
                continue
            if not isinstance(p, dict):
                continue
            t, kind = p.get("type"), o.media_mod.part_kind(p)
            if t in ("text", "input_text"):
                blocks.append({"type": "text", "text": p.get("text", "")})
            elif kind == "image":
                if self.media is None:
                    raise o.ApiError(400, "image input is disabled on this server (--vision off)")
                blocks.append({"type": "image_url", "image_url": {"url": f"{_REF}{len(media)}"}})
                media.append(p)
            elif kind == "video":
                raise o.ApiError(400, "DeepSeek-V4.1 takes images only; video input is not supported")
            elif kind == "audio":
                raise o.ApiError(400, "this model has no audio encoder; audio input is not supported")
            else:
                raise o.ApiError(400, f"content part type {t!r} is not supported")
        return blocks


if __name__ == "__main__":
    o.App = DsApp
    o.ToolCalls = DsToolCalls
    o.main()
