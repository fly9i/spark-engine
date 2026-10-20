#!/usr/bin/env python3
"""OpenAI-compatible front end for `spark-engine ds41-serve` (DeepSeek-V4.1-Flash, TP3 over three Sparks).

Reuses openai_server.py (HTTP, SSE, metrics, stop strings, reasoning split) and swaps the model specifics:
- prompts come from the checkpoint's own `encoding/encoding.py` (encode_messages: multi-turn, tools, thinking mode,
  numeric reasoning effort, mid-conversation system messages), so the format follows the official reference exactly;
- tool calls are parsed from the V4.1 DSML blocks:
    <｜DSML｜ calls>\\n<｜DSML｜ invoke name="NAME">\\n<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>...
- reasoning effort: OpenAI levels map onto the 1-100 budget (low 50, medium 62, high 75, max/xhigh 100), "none" turns
  thinking off (chat mode); an integer 1-100 passes through;
- images (--vision on, the engine needs DS41_VISION=1): content image parts go to encode_messages as image blocks (one
  <｜deepseek_image｜> each, in prompt order), mm.py loads them with the DS41 geometry of mm_ds41.py and the placeholders
  become salted spans the engine fills from the canvases. No video.

Usage:
  python3 ds41_server.py --socket /tmp/ds41-serve.sock --model-dir /models/DeepSeek-V4.1-Flash-1017 --port 8888
"""
import json, os, re, sys, time, uuid

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
_INVOKE = re.compile(r'<' + _DSML + r' invoke name="([^"]*)">\n?(.*?)</' + _DSML + r' invoke>', re.S)
_PARAM = re.compile(r'<' + _DSML + r' parameter name="([^"]*)"(?: string="(true|false)")?>(.*?)</' + _DSML + r' parameter>', re.S)


class DsToolCalls(o.ToolCalls):
    """DSML tool-call extraction: the content stream is cut at the calls block, each invoke becomes one call."""

    OPEN, CLOSE = "<" + _DSML + " calls>", "</" + _DSML + " calls>"

    def _parse_block(self, body):
        out = []
        for m in _INVOKE.finditer(body):
            name, inner = m.group(1), m.group(2)
            args = {}
            for p in _PARAM.finditer(inner):
                key, is_str, raw = p.group(1), p.group(2), p.group(3)
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
        self.buf += text
        out, new = "", []
        while True:
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
                self.buf, self.in_call = self.buf[j + len(self.CLOSE):], False
                continue
            if final and self.buf.strip():
                new.extend(self._parse_block(self.buf))
                self.buf, self.in_call = "", False
            break
        if (self.calls or self.in_call) and not out.strip():
            out = ""
        return out, new


_REF = "mm-part:"   # image blocks handed to encode_messages refer to the request's image parts by index
_EFFORT = {"low": 50, "minimal": 50, "medium": 62, "high": 75, "xhigh": 100, "max": 100}


class DsApp(o.App):
    def __init__(self, args):
        super().__init__(args)
        # requests without "temperature" sample with the official recommendation for DeepSeek-V4.1-Flash(-1017)
        # (temperature 1.0, top_p 1.0; top_p 1.0 is the full distribution, which is what the engine samples);
        # greedy decoding (the shared front end's default 0) fell into endless repetition in long agent sessions
        self.default_temperature = float(os.environ.get("DS41_DEFAULT_TEMPERATURE", "1.0"))
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

    def chat_prompt(self, body):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise o.ApiError(400, "messages must be a non-empty list", param="messages")
        # OpenAI format throughout: the official render_message converts tools and tool_calls itself, and
        # encode_messages merges role "tool" results into user turns (merge_tool_messages)
        messages, media = [], []
        for m in msgs:
            mm = dict(m)
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
        tools = body.get("tools")
        if tools:
            if messages[0].get("role") == "system":
                messages[0]["tools"] = tools
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": tools})
        kwargs = dict(body.get("chat_template_kwargs") or {})
        effort = body.get("reasoning_effort", kwargs.get("reasoning_effort"))
        thinking = kwargs.get("enable_thinking", kwargs.get("thinking", True))
        if isinstance(effort, str) and effort.lower() in ("none", "off"):
            thinking, effort = False, None
        if isinstance(effort, str):
            effort = _EFFORT.get(effort.lower(), 75)
        elif effort is not None:
            effort = max(1, min(100, int(effort)))
        mode = "thinking" if thinking is not False else "chat"
        try:
            text, mmd = self.enc.encode_messages(messages, thinking_mode=mode, reasoning_effort=effort, return_multi_modal_data=True)
        except (AssertionError, ValueError, KeyError, TypeError) as e:
            raise o.ApiError(400, f"prompt encoding error: {e}")
        reasoning = text.endswith(o.THINK_OPEN)
        ids = self.tok.encode(text, add_special_tokens=False).ids
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

    def limits(self, body, n_prompt):
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
