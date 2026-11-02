#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 image input end to end (OpenAI front end): answers checked against known image content. Needs pillow.

  ds41_vision_e2e.py [base_url, default http://127.0.0.1:8888/v1] [chart.png|-] [boundary]
    1. a rendered card with a random code and a color: the model must read the code and name the color;
    2. a follow-up turn on the same conversation: the image span is part of the reused prefix (timing prefix_hit_tokens,
       no image encoded again);
    3. two images in one message (the card and a second card): the model must read both codes in order;
    4. a real chart (optional path): free-form description, printed for a look;
    5. a video part: rejected with 400.
  boundary: also an image after about 3.7K / 4.8K tokens of new random text (each run its own, so nothing comes from the
  prefix cache of an earlier run), the same prefix asked three times.
"""
import os;os.environ['NO_PROXY']=os.environ['no_proxy']='*'
import base64, io, json, random, string, sys, time, urllib.error, urllib.request

from PIL import Image, ImageDraw, ImageFont

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888/v1"
chart = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None


def card(code, color, size=(900, 500)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 96)
    except OSError:
        font = ImageFont.load_default(size=96)
    d.rectangle([40, 40, 260, 460], fill=color)
    d.text((300, 190), code, fill="black", font=font)
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def ask(content, max_tokens=200, expect=200, history=()):
    body = {"model": "x", "messages": [*history, {"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "none"}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        assert e.code == expect, (e.code, e.read()[:300])
        return None, e.code, time.time() - t0
    assert expect == 200, f"expected {expect}"
    return out, 200, time.time() - t0


def text(out):
    return out["choices"][0]["message"]["content"]


def timing(out):
    return out.get("glm53_timing") or {}


def main():
    ok = True
    rng = random.Random(7)
    code1 = "".join(rng.choice(string.ascii_uppercase) for _ in range(3)) + "-" + "".join(rng.choice(string.digits) for _ in range(4))
    code2 = "".join(rng.choice(string.ascii_uppercase) for _ in range(3)) + "-" + "".join(rng.choice(string.digits) for _ in range(4))
    img1, img2 = card(code1, "red"), card(code2, "blue", (1000, 700))

    q1 = [{"type": "image_url", "image_url": {"url": img1}},
          {"type": "text", "text": "What code is written in this image, and what color is the rectangle on the left? Answer briefly."}]
    out, _, dt = ask(q1)
    a = text(out)
    turn1 = [{"role": "user", "content": q1}, {"role": "assistant", "content": a}]
    r1 = code1 in a and "red" in a.lower()
    ok &= r1
    print(f"[1] {dt:.1f}s {'PASS' if r1 else 'FAIL'} want {code1}/red: {a!r}\n    usage {out.get('usage')} timing {json.dumps(timing(out))[:300]}")

    out, _, dt = ask("How many digits does that code contain? Answer with one number.", history=turn1)
    a, tm = text(out), timing(out)
    r2 = "4" in a and (tm.get("prefix_hit_tokens") or 0) > 0 and not tm.get("mm_encoded_tokens")
    ok &= r2
    print(f"[2] {dt:.1f}s {'PASS' if r2 else 'FAIL'} want 4, prefix hit, nothing encoded: {a!r}; prefix_hit_tokens {tm.get('prefix_hit_tokens')}, "
          f"mm_encoded_tokens {tm.get('mm_encoded_tokens')}")

    out, _, dt = ask([{"type": "text", "text": "Here are two images."}, {"type": "image_url", "image_url": {"url": img1}},
                      {"type": "image_url", "image_url": {"url": img2}},
                      {"type": "text", "text": "List the code written in each image, in order, one per line."}])
    a = text(out)
    r3 = code1 in a and code2 in a and a.index(code1) < a.index(code2)
    ok &= r3
    print(f"[3] {dt:.1f}s {'PASS' if r3 else 'FAIL'} want {code1} then {code2}: {a!r}")

    if chart:
        with open(chart, "rb") as f:
            url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        out, _, dt = ask([{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "Describe this chart: title, axes, and the main numbers."}], 400)
        print(f"[4] {dt:.1f}s chart: {text(out)!r}")

    _, code, _ = ask([{"type": "video_url", "video_url": {"url": img1}}, {"type": "text", "text": "hi"}], expect=400)
    print(f"[5] video part -> {code} PASS")
    return ok


def boundary_test(n_words, seed):
    """Text before an image, then the same prefix with other questions. n_words 2450: the image (~3700..4540) crosses the
    4096-token prefill chunk boundary (the second request restores the 3584 checkpoint before it and encodes it again,
    rows over two chunks); 3150: the image (~4750..5590) holds the 5120 checkpoint (restored mid-span, the rest of the
    span from a fresh encode)."""
    rng2 = random.Random(seed)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    filler = " ".join(rng2.choice(words) for _ in range(n_words))
    code = "QZX-" + "".join(rng2.choice(string.digits) for _ in range(4))
    img = card(code, "green", (1400, 1000))
    msgs = lambda q: [{"type": "text", "text": "Ignore this filler: " + filler}, {"type": "image_url", "image_url": {"url": img}},
                      {"type": "text", "text": q}]
    good = True
    for i, (q, want) in enumerate([("What code is written in the image? Answer with the code only.", code),
                                   ("What color is the rectangle in the image? One word.", "green"),
                                   ("Repeat the code from the image, then its last digit.", code)]):
        out, _, dt = ask(msgs(q))
        a, tm = text(out), timing(out)
        # the first request must really encode the image, or a prefix cached by an earlier run hides the path under test
        r = want.lower() in a.lower() and (i > 0 or (tm.get("mm_encoded_tokens") or 0) > 0)
        good &= r
        print(f"[{n_words}.{i}] {dt:.1f}s {'PASS' if r else 'FAIL'} want {want}: {a!r}; prompt {out['usage']['prompt_tokens']}, "
              f"prefix_hit_tokens {tm.get('prefix_hit_tokens')}, mm_encoded_tokens {tm.get('mm_encoded_tokens')}")
    return good


ok = main()
if len(sys.argv) > 3 and sys.argv[3] == "boundary":
    ok = boundary_test(2450, random.randrange(1 << 32)) and ok
    ok = boundary_test(3150, random.randrange(1 << 32)) and ok
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
