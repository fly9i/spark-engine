"""DeepSeek-V4.1-Flash specifics of the multimodal front end (mm.py is GLM's): importing this module patches mm in place.

Geometry: the checkpoint's own inference/image_processor.py (plan_image_grid / load_image; its pure functions are
mirrored here so the front end does not import torch):
  - patch 14, 3 x 3 downsample: one LLM token per 42 x 42 pixels; small images are first scaled up to 295936 pixels
    (544 x 544), large ones down so the span stays within 1024 tokens;
  - the canvas is the image gray-padded (127) and centered to the planned size (ImageOps.pad, bicubic), uint8;
    rescale / normalize / patchify happen on the GPU (crate::ds41::vision);
  - as the reference, no EXIF orientation is applied (Image.open(...).convert("RGB")).
Placeholders: encode_messages puts one <｜deepseek_image｜> (129264) per image; each becomes the image's salted run of
the whole span, [IMAGE_START] + ([IMAGE] * gw + [IMAGE_NEW_LINE]) * gh + [IMAGE_END] = gh * (gw + 1) + 2 positions.
No video input.
"""
import io
import math

import numpy as np
from PIL import Image, ImageOps

import mm

PATCH, RATIO = 14, 3
MAX_TOKENS, MIN_PIXELS = 1024, 295936
mm.IMAGE_TOKEN, mm.VIDEO_TOKEN = 129264, -1
mm.CANVAS_VERSION = b"ds41-mm-canvas-v1"


def num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height, best_width):
    return math.ceil((best_height // PATCH) / RATIO), math.ceil((best_width // PATCH) / RATIO)


def solve_resize_ratio(height, width, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token (reference)."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = PATCH * RATIO
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / PATCH) * PATCH, math.floor(width * beta / PATCH) * PATCH


def plan_image_grid(width, height, max_n_token=MAX_TOKENS):
    """Reference plan_image_grid (vision_max_wh_ratio None): (n_llm_h, n_llm_w, best_height, best_width)."""
    if 0 < width * height < MIN_PIXELS:
        ratio = (MIN_PIXELS / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_width, best_height = math.ceil(width / PATCH) * PATCH, math.ceil(height / PATCH) * PATCH
    n_llm_h, n_llm_w = llm_grid(best_height, best_width)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def open_image(b):
    """Reference load_image: RGB of the first frame, no EXIF transpose."""
    kind = mm.sniff(b)
    if kind in ("svg", "pdf", "audio"):
        raise mm.MediaError(415, f"unsupported image type ({kind})")
    try:
        with Image.open(io.BytesIO(b)) as source:
            return source.convert("RGB")
    except Exception as e:
        raise mm.MediaError(415, f"cannot decode image: {e}{'' if mm.HEIF else ' (HEIC needs pillow-heif)'}")


def image_item(img, min_tokens, max_tokens):
    _, _, h, w = plan_image_grid(img.width, img.height, max(16, min(max_tokens, MAX_TOKENS)))
    canvas = ImageOps.pad(img, (w, h), color=(127, 127, 127))
    return mm.Item("image", np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))[None])


def _no_video(*_a, **_k):
    raise mm.MediaError(400, "DeepSeek-V4.1 takes images only; video input is not supported")


def expand(ids, items, tokenize):
    """Replace each <｜deepseek_image｜> (in image order) by the image's salted span.
    Returns (engine ids, engine mm specs without paths, plain ids with 129264 over the spans)."""
    out, plain, specs, k = [], [], [], 0
    for t in ids:
        if t != mm.IMAGE_TOKEN:
            out.append(t)
            plain.append(t)
            continue
        if k >= len(items):
            raise mm.MediaError(400, "more image placeholders than images")
        it = items[k]
        k += 1
        n = it.grid_tokens
        specs.append({"item": it, "segments": [[len(out), n, 0]]})
        out.extend(mm.salted(it.digest, n))
        plain.extend([t] * n)
    if k != len(items):
        raise mm.MediaError(400, "fewer image placeholders than images (the prompt encoding dropped an image)")
    return out, specs, plain


mm.Item.grid_tokens = property(lambda self: num_image_tokens(*llm_grid(self.canvas.shape[1], self.canvas.shape[2])))
mm.smart_resize = None            # GLM geometry must not be used by accident
mm.open_image = open_image
mm.image_item = image_item
mm.video_item_from_bytes = _no_video
mm.video_item_from_frames = _no_video
mm.expand = expand
