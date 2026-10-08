"""Qwen3.8-Flash-Next specifics of the multimodal front end (mm.py is GLM's): importing this module patches mm in place.

Geometry (the model's preprocessor_config.json / video_preprocessor_config.json, Qwen2VL image + Qwen3VL video processors):
  - patch 16, merge 2, temporal patch 2: one merged token per 32 x 32 pixels (per 2 frames of a video);
  - image: smart_resize to multiples of 32 within [65536, 16777216] pixels, bicubic, no padding;
  - video: frames sampled at 2 fps (4..768 frames, uniform linspace), smart_resize of the whole clip within
    [4096, 25165824] pixels over (frames rounded to even) x H x W; an odd frame count repeats the last frame.
Placeholders (the chat template emits <|vision_start|><|image_pad|><|vision_end|> / ... <|video_pad|> ...):
  - image: the single <|image_pad|> becomes the item's salted run (gh/2 * gw/2 ids);
  - video: the single <|video_pad|> becomes, per 2-frame group, "<{t:.1f} seconds>" <|vision_start|> run <|vision_end|>
    (t = the mean time of the group's two frames); the template's outer start / end stay, as in the HF processor.
Salted ids are even for images and odd for videos: the engine maps them back to the pad ids (crate qwen::vision::vocab_id).
"""
import math

import numpy as np
from PIL import Image

import mm

mm.IMAGE_TOKEN, mm.VIDEO_TOKEN = 248056, 248057
mm.BOI, mm.EOI = 248053, 248054
mm.BOV, mm.EOV = 248053, 248054
mm.CANVAS_VERSION = b"qwen38-mm-canvas-v1"
FACTOR = 32
IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS = 65536, 16777216
VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS = 4096, 25165824
VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES = 4, 768
# token budgets as the front end counts them (merged tokens): an image token is 32 x 32 pixels, a video token 2 x 32 x 32
IMAGE_MIN_TOKENS, IMAGE_MAX_TOKENS = IMAGE_MIN_PIXELS // FACTOR ** 2, IMAGE_MAX_PIXELS // FACTOR ** 2       # 64, 16384
VIDEO_MAX_TOKENS = VIDEO_MAX_PIXELS // (2 * FACTOR ** 2)                                                   # 12288

mm.Item.grid_tokens = property(lambda self: (self.canvas.shape[1] // FACTOR) * (self.canvas.shape[2] // FACTOR))


def smart_resize(height, width, min_pixels, max_pixels, frames=1):
    """HF Qwen2VL smart_resize (frames = 1) / Qwen3VL video smart_resize (pixels counted over frames rounded to even)."""
    if height < FACTOR or width < FACTOR:
        s = max(FACTOR / height, FACTOR / width)
        height, width = int(height * s), int(width * s)
    if max(height, width) / min(height, width) > 200:
        raise mm.MediaError(400, f"aspect ratio {max(height, width) / min(height, width):.0f} is above 200")
    t = 1 if frames == 1 else max(2, round(frames / 2) * 2)
    n = 1 if frames == 1 else frames
    hb, wb = max(FACTOR, round(height / FACTOR) * FACTOR), max(FACTOR, round(width / FACTOR) * FACTOR)
    if t * hb * wb > max_pixels:
        beta = math.sqrt(n * height * width / max_pixels)
        hb = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        wb = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif t * hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (n * height * width))
        hb = math.ceil(height * beta / FACTOR) * FACTOR
        wb = math.ceil(width * beta / FACTOR) * FACTOR
    return hb, wb


def _resize(img, h, w):
    if (img.height, img.width) != (h, w):
        img = img.resize((w, h), resample=Image.BICUBIC)
    a = np.asarray(img, dtype=np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return np.ascontiguousarray(a)


def image_item(img, min_tokens, max_tokens):
    h, w = smart_resize(img.height, img.width, min_tokens * FACTOR ** 2, max_tokens * FACTOR ** 2)
    return mm.Item("image", _resize(img, h, w)[None])


def sample_indices(total, vfps, fps, max_frames):
    """HF Qwen3VLVideoProcessor.sample_frames: int(total / vfps * fps) frames, clamped to [4, max_frames] and the total."""
    n = int(total / vfps * fps)
    n = min(max(n, VIDEO_MIN_FRAMES), max_frames, total)
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def _clip(frames_of, idx, height, width, vfps, max_tokens):
    """Canvas of the sampled frames (an odd count repeats the last) and the per-group mean timestamps."""
    h, w = smart_resize(height, width, VIDEO_MIN_PIXELS, max_tokens * 2 * FACTOR ** 2, frames=max(2, len(idx)))
    idx = list(idx) + ([idx[-1]] if len(idx) & 1 else [])
    canvas = np.zeros((len(idx), h, w, 3), dtype=np.uint8)
    frames_of(idx, lambda k, img: canvas.__setitem__(k, _resize(img, h, w)))
    ts = [(idx[i] + idx[i + 1]) / 2 / vfps for i in range(0, len(idx), 2)]
    return mm.Item("video", canvas, ts)


def video_item_from_bytes(b, fps, max_frames, max_tokens, min_tokens=16):
    if mm.av is None:
        raise mm.MediaError(415, "video decoding needs PyAV")
    if mm.sniff(b) in ("svg", "pdf", "audio"):
        raise mm.MediaError(415, "unsupported video type")
    import io
    try:
        c = mm.av.open(io.BytesIO(b))
    except Exception as e:
        raise mm.MediaError(415, f"cannot open video: {e}")
    try:
        if not c.streams.video:
            raise mm.MediaError(415, "no video stream")
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        vfps = float(vs.average_rate or vs.guessed_rate or 0) or 24.0
        total = int(vs.frames or 0)
        if total <= 0:
            dur = float(vs.duration * vs.time_base) if vs.duration else (c.duration / 1e6 if c.duration else 0)
            total = int(round(dur * vfps)) if dur > 0 else 0
        if total <= 0:
            total = sum(1 for _ in c.decode(video=0))
            c.seek(0)
        if total <= 0:
            raise mm.MediaError(415, "video has no frames")
        idx = sample_indices(total, vfps, fps, max_frames)

        def frames_of(want_idx, put):
            want = {}
            for k, i in enumerate(want_idx):
                want.setdefault(i, []).append(k)
            got, last, prev = 0, max(want_idx), None
            for i, fr in enumerate(c.decode(video=0)):
                if i > last:
                    break
                if i in want:
                    img = fr.to_image()
                    for k in want[i]:
                        put(k, img)
                        got += 1
                    prev = img
            if got < len(want_idx):            # frame count overestimated: repeat the last decoded frame
                if prev is None:
                    raise mm.MediaError(415, "video decoding produced no frames")
                for k in range(got, len(want_idx)):
                    put(k, prev)
        return _clip(frames_of, idx, vs.codec_context.height, vs.codec_context.width, vfps, max_tokens)
    finally:
        c.close()


def video_item_from_frames(images, fps, max_tokens, min_tokens=16):
    """Pre-sampled frames (all kept; timestamps i / fps)."""
    if not images:
        raise mm.MediaError(400, "empty frame list")
    w0, h0 = images[0].width, images[0].height
    ims = [im if (im.width, im.height) == (w0, h0) else im.resize((w0, h0), Image.BICUBIC) for im in images]
    return _clip(lambda idx, put: [put(k, ims[i]) for k, i in enumerate(idx)], list(range(len(ims))), h0, w0, fps, max_tokens)


def _tok_opt(d, name, pixels_name, default):
    if d.get(name) is not None:
        return int(d[name])
    if d.get(pixels_name) is not None:
        return max(1, int(d[pixels_name]) // FACTOR ** 2)
    return default


def expand(ids, items, tokenize):
    """Replace the template's single <|image_pad|> / <|video_pad|> (in media order) by salted runs (see the module doc).
    Returns (engine ids, engine mm specs without paths, plain ids with the pad ids)."""
    out, plain, specs, k = [], [], [], 0
    for t in ids:
        if t not in (mm.IMAGE_TOKEN, mm.VIDEO_TOKEN):
            out.append(t)
            plain.append(t)
            continue
        if k >= len(items):
            raise mm.MediaError(400, "more media placeholders than media parts (a literal <|image_pad|>/<|video_pad|> in the text?)")
        it = items[k]
        k += 1
        video = it.kind == "video"
        if t != (mm.VIDEO_TOKEN if video else mm.IMAGE_TOKEN):
            raise mm.MediaError(400, "media placeholders do not match the media parts")
        salt = [(s & ~1) | int(video) for s in mm.salted(it.digest, it.tokens)]
        n = it.grid_tokens
        segs = []
        if not video:
            segs.append([len(out), n, 0])
            out.extend(salt)
            plain.extend([mm.IMAGE_TOKEN] * n)
        else:
            ts = list(it.timestamps[:it.groups])
            while len(ts) < it.groups:
                ts.append(ts[-1] if ts else 0.0)
            for g in range(it.groups):
                stamp = tokenize(f"<{ts[g]:.1f} seconds>")
                out.extend(stamp)
                plain.extend(stamp)
                out.append(mm.BOI)
                plain.append(mm.BOI)
                segs.append([len(out), n, 2 * g])
                out.extend(salt[g * n:(g + 1) * n])
                plain.extend([mm.VIDEO_TOKEN] * n)
                out.append(mm.EOI)
                plain.append(mm.EOI)
        specs.append({"item": it, "segments": segs})
    if k != len(items):
        raise mm.MediaError(400, "fewer media placeholders than media parts (the chat template dropped a media part)")
    return out, specs, plain


mm.smart_resize = None            # GLM geometry must not be used by accident
mm.image_item = image_item
mm.video_item_from_bytes = video_item_from_bytes
mm.video_item_from_frames = video_item_from_frames
mm._tok_opt = _tok_opt
mm.expand = expand
_opts_init = mm.Opts.__init__


def _opts(self, *a, **kw):
    _opts_init(self, *a, **kw)
    self.min_image_tokens = max(self.min_image_tokens, IMAGE_MIN_TOKENS)
    self.video_max_frames = min(self.video_max_frames, VIDEO_MAX_FRAMES)


mm.Opts.__init__ = _opts
