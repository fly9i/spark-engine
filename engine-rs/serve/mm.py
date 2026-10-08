"""Multimodal input for the GLM-5.3 front end: media sources, decoding, the processor's canvas, salted ids.

The engine side is crate::vision (design: 自研引擎-多模态接入方案-20260930.md). This module turns every image
or video of a chat request into
  - a uint8 canvas (HWC, frames stacked) in /dev/shm: resized and zero-padded exactly like the HF
    Glm5Next processor (PIL backend for images); rescale/normalize/patchify happen on the GPU;
  - salted placeholder ids in [MM_BASE, 2^24) derived from the canvas hash: prefix caching (exact id
    prefixes) then separates different media while the same media hits again;
  - the placeholder runs ("segments") the engine fills: one per image, one per 2-frame video group.
"""
import base64, binascii, hashlib, io, ipaddress, math, os, socket, threading, time, urllib.parse, urllib.request
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF = True
except Exception:          # optional
    HEIF = False
try:
    import av
except Exception:          # optional: videos are rejected without PyAV
    av = None

Image.MAX_IMAGE_PIXELS = 256_000_000   # decompression-bomb guard (canvases are far smaller)

IMAGE_TOKEN, VIDEO_TOKEN = 154854, 154855
BOI, EOI, BOV, EOV = 154830, 154831, 154832, 154833
MM_BASE, MM_TOP = 1 << 20, 1 << 24
CANVAS_VERSION = b"glm53-mm-canvas-v1"


class MediaError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


# ----------------------------------------------------------------------------- processor geometry
def smart_resize(num_frames, height, width, temporal_factor=2, factor=28, min_tokens=16, max_tokens=8000):
    """HF Glm5Next smart_resize: an aligned canvas within the spatio-temporal token budget."""
    ppt = temporal_factor * factor ** 2
    min_pixels, max_pixels = min_tokens * ppt, max_tokens * ppt

    def align(v, f):
        return math.ceil(v / f) * f

    def fit(aligned_frames):
        if max_pixels < aligned_frames * factor ** 2:
            raise MediaError(400, f"token budget {max_tokens} is too small for {num_frames} frames")
        lo, hi = 1, height
        best = (factor, factor)
        while lo <= hi:
            ch = (lo + hi) // 2
            cw = max(1, math.floor(width * ch / height))
            h2, w2 = align(ch, factor), align(cw, factor)
            if aligned_frames * h2 * w2 <= max_pixels:
                best = (h2, w2)
                lo = ch + 1
            else:
                hi = ch - 1
        return best

    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    ah, aw = align(height, factor), align(width, factor)
    budget = aligned_frames * ah * aw
    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        ah, aw = align(max(1, math.ceil(height * scale)), factor), align(max(1, math.ceil(width * scale)), factor)
        budget = aligned_frames * ah * aw
    if budget > max_pixels:
        ah, aw = fit(aligned_frames)
    return ah, aw


def content_size(height, width, target_h, target_w, frames_for_min, min_tokens, temporal_factor=2, factor=28):
    """HF: the content keeps its aspect, is never upscaled once the raw pixels meet the minimum budget; the
    rest of the canvas is zero padding (right/bottom)."""
    ppt = temporal_factor * factor ** 2
    scale = min(target_h / height, target_w / width)
    if frames_for_min * height * width >= ppt * min_tokens:
        scale = min(1.0, scale)
    return max(1, min(target_h, math.floor(height * scale))), max(1, min(target_w, math.floor(width * scale)))


def to_canvas(img, target_h, target_w, content_h, content_w):
    """PIL bicubic resize of the content, zero padding to the canvas: uint8 [H, W, 3]."""
    if (img.height, img.width) != (content_h, content_w):
        img = img.resize((content_w, content_h), resample=Image.BICUBIC)
    a = np.asarray(img, dtype=np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if (content_h, content_w) == (target_h, target_w):
        return np.ascontiguousarray(a)
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    out[:content_h, :content_w] = a
    return out


# ----------------------------------------------------------------------------- sources
@dataclass
class Policy:
    timeout: float = 30.0
    image_max_bytes: int = 64 << 20
    video_max_bytes: int = 2 << 30
    local_root: str = ""            # file:// and plain paths allowed only below this directory ("" = off)
    deny_private: bool = False      # also refuse RFC1918/ULA hosts (loopback/link-local are always refused)
    allowed_domains: tuple = ()     # non-empty: only these hosts (and subdomains)


def _check_host(host, pol):
    if pol.allowed_domains and not any(host == d or host.endswith("." + d) for d in pol.allowed_domains):
        raise MediaError(403, f"media host {host!r} is not in the allowed domains")
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        raise MediaError(400, f"cannot resolve media host {host!r}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved \
                or (pol.deny_private and ip.is_private):
            raise MediaError(403, f"media host {host!r} resolves to a refused address {ip}")


class _Redirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, pol):
        self.pol = pol

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlparse(newurl)
        if u.scheme not in ("http", "https"):
            raise MediaError(400, f"media redirect to unsupported scheme {u.scheme!r}")
        _check_host(u.hostname or "", self.pol)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(src, pol, max_bytes):
    """Bytes of a media source: data URL, http(s) URL, file:// or path (local root only), or bare base64."""
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    if not isinstance(src, str) or not src:
        raise MediaError(400, "media source must be a non-empty string")
    s = src.strip()
    if s.startswith("data:"):
        head, _, data = s.partition(",")
        try:
            b = base64.b64decode(data, validate=False) if ";base64" in head else urllib.parse.unquote_to_bytes(data)
        except (binascii.Error, ValueError) as e:
            raise MediaError(400, f"bad data URL: {e}")
        if len(b) > max_bytes:
            raise MediaError(413, f"media larger than {max_bytes} bytes")
        return b
    u = urllib.parse.urlparse(s)
    if u.scheme in ("http", "https"):
        _check_host(u.hostname or "", pol)
        opener = urllib.request.build_opener(_Redirect(pol))
        req = urllib.request.Request(s, headers={"User-Agent": "spark-engine/1.0"})
        try:
            with opener.open(req, timeout=pol.timeout) as r:
                n = int(r.headers.get("Content-Length") or 0)
                if n > max_bytes:
                    raise MediaError(413, f"media larger than {max_bytes} bytes")
                chunks, total = [], 0
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    total += len(c)
                    if total > max_bytes:
                        raise MediaError(413, f"media larger than {max_bytes} bytes")
                    chunks.append(c)
                return b"".join(chunks)
        except MediaError:
            raise
        except Exception as e:
            raise MediaError(400, f"cannot fetch media {s[:200]}: {e}")
    # A bare base64 JPEG starts with "/9j/": a plain path is only a path when it names an existing file.
    local = u.scheme == "file" or (not u.scheme and s.startswith("/") and len(s) < 4096 and pol.local_root and os.path.isfile(s))
    if local:
        path = urllib.parse.unquote(u.path) if u.scheme == "file" else s
        if not pol.local_root:
            raise MediaError(403, "local media paths are disabled (--allowed-local-media-path)")
        real, root = os.path.realpath(path), os.path.realpath(pol.local_root)
        if os.path.commonpath([real, root]) != root:
            raise MediaError(403, f"media path {path!r} is outside --allowed-local-media-path")
        if os.path.getsize(real) > max_bytes:
            raise MediaError(413, f"media larger than {max_bytes} bytes")
        with open(real, "rb") as f:
            return f.read()
    if u.scheme and len(u.scheme) > 1:
        raise MediaError(400, f"unsupported media URL scheme {u.scheme!r}")
    try:                                              # bare base64
        b = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError):
        raise MediaError(400, "media source is neither a URL nor base64")
    if not b:
        raise MediaError(400, "empty media")
    if len(b) > max_bytes:
        raise MediaError(413, f"media larger than {max_bytes} bytes")
    return b


def sniff(b):
    """Coarse container type from magic bytes: image | video | svg | pdf | audio | unknown."""
    h = b[:64]
    if h.startswith(b"%PDF"):
        return "pdf"
    if h.lstrip()[:5] in (b"<?xml", b"<svg ") or b"<svg" in b[:512]:
        return "svg"
    if h[4:8] == b"ftyp":
        brand = h[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif", b"avis"):
            return "image"
        if brand in (b"M4A ", b"M4B "):
            return "audio"
        return "video"
    ts = len(b) > 376 and b[0] == 0x47 and b[188] == 0x47 and b[376] == 0x47      # MPEG-TS sync bytes (not GIF's "G")
    if h.startswith(b"\x1aE\xdf\xa3") or h.startswith(b"FLV") or (h.startswith(b"RIFF") and h[8:12] == b"AVI ") \
            or h.startswith(b"\x00\x00\x01\xba") or h.startswith(b"\x00\x00\x01\xb3") or ts:
        return "video"
    if h.startswith(b"ID3") or h.startswith(b"fLaC") or h.startswith(b"OggS") or (h.startswith(b"RIFF") and h[8:12] == b"WAVE"):
        return "audio"
    return "image"


def open_image(b):
    """HF load_image: EXIF orientation, then RGB (first frame of animations)."""
    kind = sniff(b)
    if kind in ("svg", "pdf", "audio"):
        raise MediaError(415, f"unsupported image type ({kind})")
    try:
        img = Image.open(io.BytesIO(b))
        img.load()
    except Exception as e:
        hint = "" if HEIF else " (HEIC needs pillow-heif)"
        raise MediaError(415, f"cannot decode image: {e}{hint}")
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# ----------------------------------------------------------------------------- items
@dataclass
class Item:
    kind: str                       # image | video
    canvas: np.ndarray              # uint8 [F, H, W, 3]
    timestamps: list = field(default_factory=list)   # video: seconds per 2-frame group
    digest: bytes = b""

    @property
    def grid_tokens(self):          # merged tokens per image / per video group
        return (self.canvas.shape[1] // 28) * (self.canvas.shape[2] // 28)

    @property
    def groups(self):
        return 1 if self.kind == "image" else self.canvas.shape[0] // 2

    @property
    def tokens(self):
        return self.grid_tokens * self.groups


def image_item(img, min_tokens, max_tokens):
    th, tw = smart_resize(2, img.height, img.width, 2, 28, min_tokens, max_tokens)
    ch, cw = content_size(img.height, img.width, th, tw, 2, min_tokens)
    return Item("image", to_canvas(img, th, tw, ch, cw)[None])


def sample_indices(total, fps, target_fps, max_frames, duration=None):
    """HF Glm5NextVideoProcessor.sample_frames (max_duration 0)."""
    max_idx = total - 1
    duration = duration or round(max_idx / fps) + 1
    max_seconds = int(duration)
    extract_t = min(int(duration * target_fps), max_frames)
    ts = [i / fps for i in range(total)]
    if total < extract_t:
        idx = np.linspace(0, total - 1, extract_t, dtype=int).tolist()
    else:
        idx, cur, inv = [], 0, 1 / target_fps
        for i in range(total):
            if ts[i] >= cur:
                cur += inv
                idx.append(i)
                if cur >= max_seconds:
                    break
    if len(idx) < extract_t:
        start, end = (0, max(total - 1, 0)) if not idx else (idx[0], idx[-1])
        idx = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(idx) > extract_t:
        idx = np.linspace(0, total - 1, extract_t, dtype=int).tolist()
    seen, uniq = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    if not uniq:
        uniq = [0]
    if len(uniq) & 1:
        uniq.append(uniq[-1])
    return uniq


def _video_canvas(n, height, width, max_tokens, min_tokens):
    th, tw = smart_resize(n, height, width, 2, 28, min_tokens, max_tokens)
    ch, cw = content_size(height, width, th, tw, n, min_tokens)
    return th, tw, ch, cw


def video_item_from_bytes(b, fps, max_frames, max_tokens, min_tokens=16):
    """Decode with PyAV (sequential, only sampled frames converted), HF frame sampling and timestamps."""
    if av is None:
        raise MediaError(415, "video decoding needs PyAV")
    kind = sniff(b)
    if kind in ("svg", "pdf", "audio"):
        raise MediaError(415, f"unsupported video type ({kind})")
    try:
        c = av.open(io.BytesIO(b))
    except Exception as e:
        raise MediaError(415, f"cannot open video: {e}")
    try:
        if not c.streams.video:
            raise MediaError(415, "no video stream")
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        vfps = float(vs.average_rate or vs.guessed_rate or 0) or 24.0
        total = int(vs.frames or 0)
        if total <= 0:                        # containers without a frame count (webm, some mkv/ts)
            dur = float(vs.duration * vs.time_base) if vs.duration else (c.duration / 1e6 if c.duration else 0)
            total = int(round(dur * vfps)) if dur > 0 else 0
        if total <= 0:                        # last resort: count by decoding
            total = sum(1 for _ in c.decode(video=0))
            c.seek(0)
        if total <= 0:
            raise MediaError(415, "video has no frames")
        idx = sample_indices(total, vfps, fps, max_frames, total / vfps if vfps else None)
        width, height = vs.codec_context.width, vs.codec_context.height
        th, tw, ch, cw = _video_canvas(len(idx), height, width, max_tokens, min_tokens)
        want = {}
        for k, i in enumerate(idx):
            want.setdefault(i, []).append(k)
        canvas = np.zeros((len(idx), th, tw, 3), dtype=np.uint8)
        got, last = 0, idx[-1]
        for i, fr in enumerate(c.decode(video=0)):
            if i > last:
                break
            if i in want:
                img = fr.to_image()           # RGB
                a = to_canvas(img, th, tw, ch, cw)
                for k in want[i]:
                    canvas[k] = a
                    got += 1
        if got < len(idx):                    # frame count overestimated: repeat the last decoded frame
            if got == 0:
                raise MediaError(415, "video decoding produced no frames")
            canvas[got:] = canvas[got - 1]
    finally:
        c.close()
    ts = [i / vfps for i in idx][::2]
    return Item("video", canvas, ts)


def video_item_from_frames(images, fps, max_tokens, min_tokens=16):
    """Pre-sampled frames (list of images): all kept, padded to an even count; timestamps i / fps."""
    if not images:
        raise MediaError(400, "empty frame list")
    if len(images) & 1:
        images = images + [images[-1]]
    w0, h0 = images[0].width, images[0].height
    th, tw, ch, cw = _video_canvas(len(images), h0, w0, max_tokens, min_tokens)
    canvas = np.stack([to_canvas(im if (im.width, im.height) == (w0, h0) else im.resize((w0, h0), Image.BICUBIC), th, tw, ch, cw)
                       for im in images])
    ts = [i / fps for i in range(len(images))][::2]
    return Item("video", canvas, ts)


def digest(item, extra=b""):
    h = hashlib.sha256(CANVAS_VERSION)
    h.update(item.kind.encode())
    h.update(np.asarray(item.canvas.shape, dtype=np.int64).tobytes())
    h.update(item.canvas.tobytes())
    for t in item.timestamps:
        h.update(f"{t:.6f}".encode())
    h.update(extra)
    return h.digest()


def salted(dig, n):
    """n salted placeholder ids for an item: splitmix64 over (digest, position), mapped into [MM_BASE, 2^24)."""
    seed = np.uint64(int.from_bytes(dig[:8], "little"))
    with np.errstate(over="ignore"):
        x = seed + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (np.uint64(MM_BASE) + x % np.uint64(MM_TOP - MM_BASE)).astype(np.int64).tolist()


# ----------------------------------------------------------------------------- canvas store (/dev/shm)
class CanvasStore:
    """Canvas files for the engine plus an LRU of decoded/resized items keyed by (raw bytes, options):
    a conversation re-sends its history images every turn, which then skips decode and resize. Files of
    requests in flight are never evicted."""

    def __init__(self, directory, cap_bytes):
        self.dir, self.cap = directory, cap_bytes
        os.makedirs(directory, exist_ok=True)
        for f in os.listdir(directory):
            try:
                os.unlink(os.path.join(directory, f))
            except OSError:
                pass
        self.lock = threading.Lock()
        self.by_key = {}        # raw key -> digest hex
        self.items = {}         # digest hex -> Item
        self.files = {}         # digest hex -> [path, bytes, refs, last_used]

    def lookup(self, key):
        with self.lock:
            hx = self.by_key.get(key)
            if hx is None or hx not in self.files:
                return None
            self.files[hx][3] = time.time()
            return self.items[hx]

    def put(self, key, item):
        """Registers the item (writes its canvas file once per digest) and returns the stored item."""
        hx = item.digest.hex()
        with self.lock:
            if hx in self.files:
                self.by_key[key] = hx
                self.files[hx][3] = time.time()
                return self.items[hx]
        path = os.path.join(self.dir, hx + ".u8")
        tmp = path + f".{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            f.write(item.canvas.tobytes())
        os.replace(tmp, path)
        with self.lock:
            if hx not in self.files:
                self.files[hx] = [path, item.canvas.nbytes, 0, time.time()]
                self.items[hx] = item
            self.by_key[key] = hx
            self._evict()
            return self.items[hx]

    def path(self, item):
        return self.files[item.digest.hex()][0]

    def acquire(self, items):
        with self.lock:
            for it in items:
                self.files[it.digest.hex()][2] += 1

    def release(self, items):
        with self.lock:
            for it in items:
                f = self.files.get(it.digest.hex())
                if f:
                    f[2] -= 1
                    f[3] = time.time()
            self._evict()

    GRACE_S = 300    # a file prepared for a request stays until that request acquires it (prompt build -> run)

    def _evict(self):
        total = sum(f[1] for f in self.files.values())
        if total <= self.cap:
            return
        now = time.time()
        for hx, f in sorted(self.files.items(), key=lambda kv: kv[1][3]):
            if total <= self.cap:
                break
            if f[2] > 0 or now - f[3] < self.GRACE_S:
                continue
            try:
                os.unlink(f[0])
            except OSError:
                pass
            total -= f[1]
            del self.files[hx]
            del self.items[hx]
            for k in [k for k, v in self.by_key.items() if v == hx]:
                del self.by_key[k]


# ----------------------------------------------------------------------------- request parts
AUDIO_TYPES = ("audio", "audio_url", "input_audio")


def _src_of(part, kind):
    """Media source of a content part in any of the accepted shapes (OpenAI, vLLM, Responses, HF/Qwen, Anthropic)."""
    t = part.get("type")
    for k in (t, f"{kind}_url", kind, f"input_{kind}"):
        v = part.get(k)
        if isinstance(v, dict):
            v = v.get("url") or v.get("data") or v.get(kind)
        if isinstance(v, (str, list)) and v:
            return v
    s = part.get("source")
    if isinstance(s, dict):
        if s.get("type") == "base64" and s.get("data"):
            return f"data:{s.get('media_type', 'application/octet-stream')};base64,{s['data']}"
        if s.get("url"):
            return s["url"]
    if part.get("url"):
        return part["url"]
    if part.get("data"):
        return part["data"]
    raise MediaError(400, f"content part of type {t!r} has no media source")


def part_kind(part):
    """'image' | 'video' | 'audio' | None for a content part dict."""
    t = part.get("type")
    if t in ("image_url", "image", "input_image"):
        return "image"
    if t in ("video_url", "video", "input_video"):
        return "video"
    if t in AUDIO_TYPES:
        return "audio"
    return None


@dataclass
class Opts:
    min_image_tokens: int = 16
    max_image_tokens: int = 8000
    max_video_tokens: int = 65536
    video_fps: float = 2.0
    video_max_frames: int = 2048
    detail_low_tokens: int = 1024
    video_token_cap: int = 240000     # model limit (processor max_image_tokens of the video processor)


def _tok_opt(d, name, pixels_name, default):
    if d.get(name) is not None:
        return int(d[name])
    if d.get(pixels_name) is not None:
        return max(1, int(d[pixels_name]) // (28 * 28))
    return default


class Media:
    def __init__(self, store, policy, opts):
        self.store, self.pol, self.opts = store, policy, opts

    def load(self, part, req_kwargs):
        """One content part -> Item (cached by raw bytes + options)."""
        kind = part_kind(part)
        o = self.opts
        kw = dict(req_kwargs or {})
        for k in ("min_image_tokens", "max_image_tokens", "min_pixels", "max_pixels", "fps", "max_frames", "max_video_tokens"):
            if k in part:
                kw[k] = part[k]
        src = _src_of(part, kind)
        if kind == "image":
            detail = (part.get("image_url") or {}).get("detail") if isinstance(part.get("image_url"), dict) else part.get("detail")
            mx = _tok_opt(kw, "max_image_tokens", "max_pixels", o.detail_low_tokens if detail == "low" else o.max_image_tokens)
            mx = min(mx, o.max_image_tokens)
            mn = _tok_opt(kw, "min_image_tokens", "min_pixels", o.min_image_tokens)
            mx, mn = max(16, mx), max(1, min(mn, mx))
            raw = fetch(src, self.pol, self.pol.image_max_bytes)
            key = hashlib.sha256(raw).hexdigest() + f":i:{mn}:{mx}"
            it = self.store.lookup(key)
            if it is None:
                it = image_item(open_image(raw), mn, mx)
                it.digest = digest(it)
                it = self.store.put(key, it)
            return it
        if kind == "video":
            fps = float(kw.get("fps") or o.video_fps)
            mf = int(kw.get("max_frames") or o.video_max_frames)
            mx = max(64, min(int(kw.get("max_video_tokens") or o.max_video_tokens), o.video_token_cap))
            if isinstance(src, list):          # pre-sampled frames
                raws = [fetch(s, self.pol, self.pol.image_max_bytes) for s in src]
                key = hashlib.sha256(b"".join(hashlib.sha256(r).digest() for r in raws)).hexdigest() + f":f:{fps}:{mx}"
                it = self.store.lookup(key)
                if it is None:
                    it = video_item_from_frames([open_image(r) for r in raws], fps, mx)
                    it.digest = digest(it)
                    it = self.store.put(key, it)
                return it
            raw = fetch(src, self.pol, self.pol.video_max_bytes)
            key = hashlib.sha256(raw).hexdigest() + f":v:{fps}:{mf}:{mx}"
            it = self.store.lookup(key)
            if it is None:
                try:                           # PyAV also demuxes GIF/APNG/WebP animations
                    it = video_item_from_bytes(raw, fps, mf, mx)
                except MediaError:
                    if sniff(raw) != "image":
                        raise
                    it = self._animated(raw, fps, mx)   # animations PyAV cannot open: frames via PIL
                it.digest = digest(it)
                it = self.store.put(key, it)
            return it
        raise MediaError(400, "this model has no audio encoder; audio input is not supported")

    def _animated(self, raw, fps, mx):
        try:
            im = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise MediaError(415, f"cannot decode video/animation: {e}")
        frames, durs = [], []
        try:
            for i in range(getattr(im, "n_frames", 1)):
                im.seek(i)
                frames.append(im.convert("RGB"))
                durs.append(im.info.get("duration", 100) or 100)
        except EOFError:
            pass
        # Uniform resampling at `fps` over the animation's time line.
        t, acc = [], 0.0
        for d in durs:
            t.append(acc)
            acc += d / 1000.0
        n = max(2, int(acc * fps))
        picks = [min(range(len(t)), key=lambda k: abs(t[k] - j / fps)) for j in range(n)]
        return video_item_from_frames([frames[k] for k in picks], fps, mx)


def expand(ids, items, tokenize):
    """Replace the template's single <|image|>/<|video|> placeholders (in media order) by salted runs.
    Returns (new ids, engine mm spec list without paths, plain ids with 154854 placeholders)."""
    out, plain, specs, k = [], [], [], 0
    for t in ids:
        if t not in (IMAGE_TOKEN, VIDEO_TOKEN):
            out.append(t)
            plain.append(t)
            continue
        if k >= len(items):
            raise MediaError(400, "more media placeholders than media parts (literal <|image|>/<|video|> in the text?)")
        it = items[k]
        k += 1
        want = IMAGE_TOKEN if it.kind == "image" else VIDEO_TOKEN
        if t != want:
            raise MediaError(400, "media placeholders do not match the media parts")
        salt = salted(it.digest, it.tokens)
        n = it.grid_tokens
        segs = []
        if it.kind == "image":
            segs.append([len(out), n, 0])
            out.extend(salt)
            plain.extend([IMAGE_TOKEN] * n)
        else:
            ts = list(it.timestamps[:it.groups])
            while len(ts) < it.groups:
                ts.append(ts[-1] if ts else 0.0)
            for g in range(it.groups):
                out.append(BOI)
                plain.append(BOI)
                segs.append([len(out), n, 2 * g])
                out.extend(salt[g * n:(g + 1) * n])
                plain.extend([IMAGE_TOKEN] * n)
                out.append(EOI)
                plain.append(EOI)
                stamp = tokenize(f"{ts[g]:.1f} seconds")
                out.extend(stamp)
                plain.extend(stamp)
        specs.append({"item": it, "segments": segs})
    if k != len(items):
        raise MediaError(400, "fewer media placeholders than media parts (the chat template dropped a media part)")
    return out, specs, plain
