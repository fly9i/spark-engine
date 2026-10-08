//! Qwen3.8-Flash-Next vision tower (qwen4_exp vision: 27 ViT blocks of 1152, 16 heads, 2D axial RoPE, learned 48x48
//! position grid resampled per image, 2x2 patch merger to the 2560-wide text embedding) and the multimodal prompt
//! plumbing: salted placeholders, encoded rows per prefill chunk, 3D (mRoPE) positions.
//!
//! Weights: `vision_k6.safetensors` (EXL3 6-bit linears with biases; the fused qkv as stored in F16).
//! Data flow (as GLM's crate::vision): the front end resizes every image / video to its canvas (uint8 HWC, frames
//! stacked, sides multiples of 32) and expands each placeholder into salted ids in [MM_BASE, 2^24) (even: image, odd:
//! video), distinct per content, so prefix reuse (plain id `starts_with`) separates different media. The engine encodes
//! the segments a prefill chunk touches straight from the canvas and writes their rows into the chunk's embedding.
use super::ffi;
use super::load::Checkpoint;
use super::model::Exl3;
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};

pub const MM_BASE: i64 = 1 << 20;
pub const IMAGE_PAD: i64 = 248056;
pub const VIDEO_PAD: i64 = 248057;

/// The vocabulary id behind a prompt id: salted placeholders are the image / video pad tokens (the n-gram embedding and the
/// MTP head see the pad ids, as the reference passes input_ids).
pub fn vocab_id(t: i64) -> i64 { if t >= MM_BASE { if t & 1 == 1 { VIDEO_PAD } else { IMAGE_PAD } } else { t } }

/// QWEN_VISION=1: load the vision tower (about 0.5 GB) and accept image / video requests.
pub fn enabled() -> bool { std::env::var("QWEN_VISION").as_deref() == Ok("1") }

struct VBlock { n1w: Tensor, n1b: Tensor, qkv_w: Tensor, qkv_b: Tensor, proj: Exl3, proj_b: Tensor, n2w: Tensor, n2b: Tensor,
                fc1: Exl3, fc1_b: Tensor, fc2: Exl3, fc2_b: Tensor }

pub struct Vision {
    hidden: i64, heads: i64, patch: i64, merge: i64, tps: i64, side: i64,
    pe_w: Tensor, pe_b: Tensor, pos: Tensor, blocks: Vec<VBlock>,
    mn_w: Tensor, mn_b: Tensor, m_fc1: Exl3, m_fc1_b: Tensor, m_fc2: Exl3, m_fc2_b: Tensor,
    inv_freq: Tensor, dev: Device,
}

fn f32t(t: Tensor) -> Tensor { t.to_kind(Kind::Float).contiguous() }

/// QWEN_VIS_EXACT (read per call; default on): the vision tower's EXL3 linears take the GEMM output in fp32 (cuBLASLt) and
/// finish in fp32, instead of the many-row path's fp16 GEMM output. 2026-10-07 vs an fp32 reference (3 images + a video):
/// screen relL2 0.0246 -> 0.0032, worst merged row cos 0.797 -> 0.998, the others 0.0046/0.0036/0.0081 -> 0.0028/0.0029/
/// 0.0069, at +3..10% encode time. The error was the fp16 output, not the fp16 input (a hi/lo input split on top changed
/// nothing). For scale: bf16 activations, as in a bf16 ViT, give relL2 0.0169 and a worst row of 0.926 there.
fn exact_on() -> bool { std::env::var("QWEN_VIS_EXACT").as_deref() != Ok("0") }
/// QWEN_VIS_ATT32=1 (read per call; off): attention in fp32 instead of fp16 (screen relL2 0.0032 -> 0.0022 at +55% time).
fn att32_on() -> bool { std::env::var("QWEN_VIS_ATT32").as_deref() == Ok("1") }

impl Vision {
    pub fn load(ck: &Checkpoint, cfg: &serde_json::Value) -> Self {
        let vc = &cfg["vision_config"];
        let num = |k: &str, d: f64| vc.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        let (hidden, heads, depth) = (num("hidden_size", 1152.) as i64, num("num_heads", 16.) as i64, num("depth", 27.) as usize);
        let (patch, merge, tps) = (num("patch_size", 16.) as i64, num("spatial_merge_size", 2.) as i64, num("temporal_patch_size", 2.) as i64);
        let side = (num("num_position_embeddings", 2304.) as f64).sqrt() as i64;
        let theta = vc.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(|x| x.as_f64()).unwrap_or(10000.);
        let g = |n: &str| ck.get(&format!("model.visual.{n}"));
        let e = |n: &str| Exl3::load(ck, &format!("model.visual.{n}"));
        let pe = g("patch_embed.proj.weight");
        assert_eq!(pe.size(), [hidden, 3, tps, patch, patch], "patch_embed shape");
        let pe_w = f32t(pe.reshape([hidden, 3 * tps * patch * patch]));
        let blocks = (0..depth).map(|i| {
            let b = |n: &str| g(&format!("blocks.{i}.{n}"));
            VBlock { n1w: f32t(b("norm1.weight")), n1b: f32t(b("norm1.bias")), qkv_w: b("attn.qkv.weight").to_kind(Kind::Half).contiguous(),
                     qkv_b: f32t(b("attn.qkv.bias")), proj: e(&format!("blocks.{i}.attn.proj")), proj_b: f32t(b("attn.proj.bias")),
                     n2w: f32t(b("norm2.weight")), n2b: f32t(b("norm2.bias")), fc1: e(&format!("blocks.{i}.mlp.linear_fc1")),
                     fc1_b: f32t(b("mlp.linear_fc1.bias")), fc2: e(&format!("blocks.{i}.mlp.linear_fc2")), fc2_b: f32t(b("mlp.linear_fc2.bias")) }
        }).collect();
        let dev = Device::Cuda(0);
        // axial rope (vllm qwen3_vl: neox, half the head dim): inv_freq = 1 / theta^(arange(0, spatial, 2) / spatial), spatial = head_dim / 2, float32
        let spatial = hidden / heads / 2;
        let ex = Tensor::arange_start_step(0, spatial, 2, (Kind::Float, dev)) / (spatial as f64);
        let inv_freq = Tensor::from(theta).to_kind(Kind::Float).to_device(dev).pow(&ex).reciprocal();
        eprintln!("[qwen-vision] loaded: depth {depth}, hidden {hidden}, heads {heads}, rope theta {theta}");
        Vision { hidden, heads, patch, merge, tps, side, pe_w, pe_b: f32t(g("patch_embed.proj.bias")), pos: f32t(g("pos_embed.weight")), blocks,
                 mn_w: f32t(g("merger.norm.weight")), mn_b: f32t(g("merger.norm.bias")), m_fc1: e("merger.linear_fc1"),
                 m_fc1_b: f32t(g("merger.linear_fc1.bias")), m_fc2: e("merger.linear_fc2"), m_fc2_b: f32t(g("merger.linear_fc2.bias")), inv_freq, dev }
    }

    /// Canvas (uint8 [F, H, W, 3] on the device; F = 1 for an image, 2 G for G video groups) -> patch rows [G*gh*gw, 3*2*16*16]
    /// fp32 in the processor's merge-major order: x / 255, (x - 0.5) / 0.5 (float32), an image duplicated in time.
    pub fn pixels(&self, canvas: &Tensor) -> (Tensor, i64, i64, i64) {
        let s = canvas.size();
        let (f, h, w) = (s[0], s[1], s[2]);
        let (p, m, t) = (self.patch, self.merge, self.tps);
        assert!(h % (p * m) == 0 && w % (p * m) == 0, "canvas {h}x{w} not aligned to {}", p * m);
        let x = (canvas.to_kind(Kind::Float) * (1.0 / 255.0) - 0.5) / 0.5;
        let x = x.permute([0, 3, 1, 2]);   // [F, 3, H, W]
        let (g, x) = if f == 1 { (1, x.unsqueeze(0).expand([1, t, 3, h, w], false)) } else {
            assert_eq!(f % t, 0, "video frames must be a multiple of the temporal patch");
            (f / t, x.reshape([f / t, t, 3, h, w]))
        };
        let (gh, gw) = (h / p, w / p);
        let x = x.reshape([g, t, 3, gh / m, m, p, gw / m, m, p]).permute([0, 3, 6, 4, 7, 2, 1, 5, 8]).reshape([g * gh * gw, 3 * t * p * p]);
        (x.contiguous(), g, gh, gw)
    }

    /// Merge-major (row, col) of the patches of one gh x gw frame.
    fn coords(&self, gh: i64, gw: i64) -> Vec<(i64, i64)> {
        let m = self.merge;
        let mut v = Vec::with_capacity((gh * gw) as usize);
        for br in 0..gh / m { for bc in 0..gw / m { for ir in 0..m { for ic in 0..m { v.push((br * m + ir, bc * m + ic)); } } } }
        v
    }

    /// The learned 48 x 48 position grid resampled to gh x gw (bilinear, align_corners, float32 as vllm
    /// qwen3_vl pos_embed_interpolate_native), merge-major rows [gh*gw, hidden].
    fn pos_embed(&self, gh: i64, gw: i64) -> Tensor {
        let side = self.side;
        let taps = |i: i64, size: i64| -> [(i64, f32); 2] {
            let src = i as f32 * (side - 1) as f32 / ((size - 1).max(1)) as f32;
            let fl = src.floor();
            let t = |o: i64| (((fl as i64) + o).clamp(0, side - 1), (1.0 - (src - fl - o as f32).abs()).max(0.0));
            [t(0), t(1)]
        };
        let (mut idx, mut wt) = (Vec::new(), Vec::new());
        for (r, c) in self.coords(gh, gw) {
            let (hr, hc) = (taps(r, gh), taps(c, gw));
            for &(ti, wi) in &hr { for &(tj, wj) in &hc { idx.push(ti * side + tj); wt.push(wi * wj); } }
        }
        let n = gh * gw;
        let idx = Tensor::from_slice(&idx).to_device(self.dev);
        let wt = Tensor::from_slice(&wt).to_device(self.dev).view([n, 4, 1]);
        (self.pos.index_select(0, &idx).view([n, 4, self.hidden]) * wt).sum_dim_intlist(&[1i64][..], false, Kind::Float)
    }

    /// Axial rotary cos / sin [gh*gw, head_dim] (fp32) for one frame ([h | w | h | w]).
    fn rope(&self, gh: i64, gw: i64) -> (Tensor, Tensor) {
        let (hs, ws): (Vec<f32>, Vec<f32>) = self.coords(gh, gw).into_iter().map(|(r, c)| (r as f32, c as f32)).unzip();
        let hp = Tensor::from_slice(&hs).to_device(self.dev).unsqueeze(1) * self.inv_freq.unsqueeze(0);
        let wp = Tensor::from_slice(&ws).to_device(self.dev).unsqueeze(1) * self.inv_freq.unsqueeze(0);
        let hw = Tensor::cat(&[hp, wp], 1);
        let full = Tensor::cat(&[&hw, &hw], 1);
        (full.cos(), full.sin())
    }

    fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        // x [N, H, D] fp32: x * cos + rotate_half(x) * sin
        let d = x.size()[2];
        let (x1, x2) = (x.narrow(2, 0, d / 2), x.narrow(2, d / 2, d / 2));
        let rot = Tensor::cat(&[x2.neg(), x1], 2);
        x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)
    }

    fn linear_bias(&self, l: &Exl3, x: &Tensor, b: &Tensor) -> Tensor {
        if !exact_on() { return l.fwd(&x.contiguous()) + b.narrow(0, 0, l.n); }
        let (m, k, n) = (x.size()[0], l.k, l.n);
        let (suh, svh) = l.scales();
        let xh = Tensor::empty([m, k], (Kind::Half, self.dev));
        ffi::exl3_had_in(&x.contiguous(), suh, &xh, m, k);
        let part = Tensor::empty([m, n], (Kind::Float, self.dev));
        ffi::lt_mm16(&xh, &l.inner_t(), &part);
        let y = part.empty_like();
        ffi::exl3_finish(&part, 1, svh, &y, m, n);
        y + b.narrow(0, 0, n)
    }

    /// `g` segments (an image, or the groups of a video) of grid gh x gw: patch rows -> merged tokens [g*gh*gw/4, 2560] fp32.
    pub fn forward(&self, px: &Tensor, g: i64, gh: i64, gw: i64) -> Tensor {
        let _ng = tch::no_grad_guard();
        let n = gh * gw;
        let rows = g * n;
        let (heads, hd) = (self.heads, self.hidden / self.heads);
        let mut x = px.matmul(&self.pe_w.tr()) + &self.pe_b;
        let pe = self.pos_embed(gh, gw);
        x += if g > 1 { pe.repeat([g, 1]) } else { pe };
        let (cos, sin) = self.rope(gh, gw);
        let (cos, sin) = if g > 1 { (cos.repeat([g, 1]), sin.repeat([g, 1])) } else { (cos, sin) };
        let scale = (hd as f64).powf(-0.5);
        // Row blocks bound the activations: a 16K-token image has 65536 patch rows, whose full fp32 qkv / MLP intermediates
        // are 0.9..1.1 GiB each (2026-10-07: MemAvailable fell from 29 to 12 GiB for such an image without blocks).
        const BLK: i64 = 8192;
        let blocks = |f: &mut dyn FnMut(i64, i64)| { let mut r0 = 0; while r0 < rows { let nr = (rows - r0).min(BLK); f(r0, nr); r0 += nr; } };
        // attention runs in fp16 (flash) unless QWEN_VIS_ATT32
        let ak = if att32_on() { Kind::Float } else { Kind::Half };
        for b in &self.blocks {
            let (q, k, v) = (Tensor::empty([rows, heads, hd], (ak, self.dev)), Tensor::empty([rows, heads, hd], (ak, self.dev)),
                             Tensor::empty([rows, heads, hd], (ak, self.dev)));
            blocks(&mut |r0, nr| {
                let h = x.narrow(0, r0, nr).layer_norm([self.hidden], Some(&b.n1w), Some(&b.n1b), 1e-6, false);
                let qkv = Tensor::empty([nr, 3 * self.hidden], (Kind::Float, self.dev));
                ffi::lt_mm16(&h.to_kind(Kind::Half), &b.qkv_w, &qkv);
                let qkv = (qkv + &b.qkv_b).view([nr, 3, heads, hd]);
                let (c, sn) = (cos.narrow(0, r0, nr), sin.narrow(0, r0, nr));
                q.narrow(0, r0, nr).copy_(&Self::rotate(&qkv.select(1, 0), &c, &sn));
                k.narrow(0, r0, nr).copy_(&Self::rotate(&qkv.select(1, 1), &c, &sn));
                v.narrow(0, r0, nr).copy_(&qkv.select(1, 2));
            });
            // non-causal attention inside each segment: [g, heads, n, hd]
            let seg = |t: &Tensor| t.view([g, n, heads, hd]).transpose(1, 2).contiguous();
            let o = Tensor::scaled_dot_product_attention(&seg(&q), &seg(&k), &seg(&v), None::<Tensor>, 0.0, false, scale, false);
            drop((q, k, v));
            let o = o.transpose(1, 2).reshape([rows, self.hidden]);
            blocks(&mut |r0, nr| {
                let mut xb = x.narrow(0, r0, nr);
                xb += self.linear_bias(&b.proj, &o.narrow(0, r0, nr).to_kind(Kind::Float), &b.proj_b);
                let h = xb.layer_norm([self.hidden], Some(&b.n2w), Some(&b.n2b), 1e-6, false);
                let a = self.linear_bias(&b.fc1, &h, &b.fc1_b).gelu("tanh");
                xb += self.linear_bias(&b.fc2, &a, &b.fc2_b);
            });
        }
        let x = x.layer_norm([self.hidden], Some(&self.mn_w), Some(&self.mn_b), 1e-6, false).reshape([rows / 4, 4 * self.hidden]);
        let x = self.linear_bias(&self.m_fc1, &x, &self.m_fc1_b).gelu("none");
        self.linear_bias(&self.m_fc2, &x, &self.m_fc2_b)
    }
}

/// One media item of a request: its canvas file and the placeholder runs it fills.
#[derive(Clone, Debug)]
pub struct MmItem {
    pub path: PathBuf, pub frames: i64, pub height: i64, pub width: i64, pub video: bool,
    /// (prompt position, placeholder count, first frame): an image has one segment (frame 0, duplicated in time); a video
    /// has one per 2-frame group.
    pub segments: Vec<(usize, usize, i64)>,
}

/// The request's "mm" array: [{"path","frames","height","width","video","segments":[[pos,len,frame],...]}].
pub fn parse_items(v: &serde_json::Value) -> Result<Vec<MmItem>, String> {
    let Some(a) = v.as_array() else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for it in a {
        let path = PathBuf::from(it["path"].as_str().ok_or("mm item without path")?);
        let get = |k: &str| it[k].as_i64().ok_or(format!("mm item without {k}"));
        let (frames, height, width) = (get("frames")?, get("height")?, get("width")?);
        let video = it["video"].as_bool().unwrap_or(frames > 1);
        if frames < 1 || height < 32 || width < 32 || height % 32 != 0 || width % 32 != 0 || (video && frames % 2 != 0) {
            return Err(format!("mm item has a bad canvas {frames}x{height}x{width}"));
        }
        let meta = std::fs::metadata(&path).map_err(|e| format!("mm canvas {}: {e}", path.display()))?;
        if meta.len() as i64 != frames * height * width * 3 {
            return Err(format!("mm canvas {} has {} bytes, expected {}", path.display(), meta.len(), frames * height * width * 3));
        }
        let tokens_per = (height / 32) * (width / 32);
        let mut segments = Vec::new();
        for s in it["segments"].as_array().ok_or("mm item without segments")? {
            let s = s.as_array().ok_or("bad segment")?;
            let f = |i: usize| s.get(i).and_then(|x| x.as_i64()).ok_or("bad segment");
            let (pos, len, frame) = (f(0)?, f(1)?, f(2)?);
            if pos < 0 || len != tokens_per || frame < 0 || frame >= frames || (video && (frame % 2 != 0 || frame + 2 > frames)) {
                return Err(format!("mm segment ({pos},{len},{frame}) does not fit the canvas"));
            }
            segments.push((pos as usize, len as usize, frame));
        }
        out.push(MmItem { path, frames, height, width, video, segments });
    }
    Ok(out)
}

/// Check a prompt's salted placeholders against its items: every id is a vocabulary id or a placeholder of the right kind,
/// and the placeholder positions are exactly the items' segments.
pub fn validate(ids: &[i64], vocab: i64, items: &[MmItem]) -> Result<(), String> {
    let mut mark = vec![0u8; ids.len()];
    for it in items {
        for &(pos, len, _) in &it.segments {
            if pos + len > ids.len() { return Err("mm segment past the prompt end".into()); }
            for j in pos..pos + len { if mark[j] != 0 { return Err("overlapping mm segments".into()); } mark[j] = 1 + it.video as u8; }
        }
    }
    for (j, &t) in ids.iter().enumerate() {
        let salted = (MM_BASE..1 << 24).contains(&t);
        let want = if salted { 1 + (t & 1) as u8 } else { 0 };
        if want != mark[j] {
            return Err(if salted { format!("placeholder id at {j} has no mm segment of its kind") } else { format!("mm segment covers non-placeholder id {t} at {j}") });
        }
        if !salted && !(0..vocab).contains(&t) { return Err(format!("token id {t} at {j} is outside the vocabulary")); }
    }
    Ok(())
}

/// mRoPE positions (t, h, w) of every prompt token (vllm Qwen3VL _get_mrope_input_positions: text runs advance by one; an image / video group
/// of merged grid gh x gw starting at position s takes (s, s + row, s + col) and advances the position by max(gh, gw)),
/// and the delta of the text after the prompt (its position = token index - delta).
pub fn rope_positions(n: usize, items: &[MmItem]) -> (Vec<[i32; 3]>, i64) {
    let mut seg: Vec<(usize, usize, i64, i64)> = items.iter()
        .flat_map(|it| it.segments.iter().map(move |&(p, l, _)| (p, l, it.height / 32, it.width / 32))).collect();
    seg.sort_unstable();
    let mut out = Vec::with_capacity(n);
    let (mut i, mut cur) = (0usize, 0i64);
    for (p, l, gh, gw) in seg {
        while i < p { out.push([cur as i32; 3]); cur += 1; i += 1; }
        assert_eq!((gh * gw) as usize, l);
        for r in 0..gh { for c in 0..gw { out.push([cur as i32, (cur + r) as i32, (cur + c) as i32]); } }
        i += l;
        cur += gh.max(gw);
    }
    while i < n { out.push([cur as i32; 3]); cur += 1; i += 1; }
    (out, n as i64 - cur)
}

/// Per-sequence encoder state: segments encoded and waiting for the prefill chunks that consume them.
#[derive(Default)]
pub struct MmState { pub items: Vec<MmItem>, done: std::collections::HashMap<(usize, usize), Tensor>, pub encode_ms: f64, pub encoded_tokens: usize }

impl MmState {
    pub fn new(items: Vec<MmItem>) -> Self { MmState { items, ..Default::default() } }

    /// Encode every segment that intersects [begin, end) and is not cached yet, then return (rows of this chunk that are
    /// placeholders, relative to begin; their embeddings [k, 2560] fp32) in position order. Consumed segments are released.
    pub fn chunk_rows(&mut self, vision: &Vision, begin: usize, end: usize) -> Option<(Vec<i64>, Tensor)> {
        let t0 = std::time::Instant::now();
        let enc0 = self.encoded_tokens;
        let mut pieces: Vec<(usize, Tensor)> = Vec::new();
        for (ii, it) in self.items.iter().enumerate() {
            let need: Vec<usize> = (0..it.segments.len())
                .filter(|&si| { let (p, l, _) = it.segments[si]; p < end && p + l > begin && !self.done.contains_key(&(ii, si)) }).collect();
            if !need.is_empty() {
                // video groups share one grid: up to ~16K patches per forward (bounded activations)
                let per = (it.height / 16) * (it.width / 16);
                let batch = ((16384 / per).max(1)) as usize;
                for group in need.chunks(batch) {
                    let canvas = read_frames(it, &group.iter().map(|&si| it.segments[si].2).collect::<Vec<_>>(), vision.dev);
                    let (px, g, gh, gw) = vision.pixels(&canvas);
                    let y = vision.forward(&px, g, gh, gw);
                    let tok = gh * gw / 4;
                    for (k, &si) in group.iter().enumerate() { self.done.insert((ii, si), y.narrow(0, k as i64 * tok, tok)); }
                    self.encoded_tokens += group.len() * tok as usize;
                }
            }
            for (si, &(p, l, _)) in it.segments.iter().enumerate() {
                if p < end && p + l > begin {
                    let (lo, hi) = (p.max(begin), (p + l).min(end));
                    pieces.push((lo, self.done[&(ii, si)].narrow(0, (lo - p) as i64, (hi - lo) as i64)));
                }
            }
        }
        // give large encodes' cached blocks back to the driver (unified memory: the caching allocator would keep them
        // out of MemAvailable, 8.5 GiB after a 16K-token image)
        if self.encoded_tokens > enc0 + 4096 {
            extern "C" { fn rs_empty_cache(); }
            tch::Cuda::synchronize(0);
            unsafe { rs_empty_cache(); }
        }
        let items = &self.items;
        self.done.retain(|&(ii, si), _| { let (p, l, _) = items[ii].segments[si]; p + l > end });
        if pieces.is_empty() { return None; }
        pieces.sort_by_key(|p| p.0);
        let rows: Vec<i64> = pieces.iter().flat_map(|(lo, t)| (0..t.size()[0]).map(move |k| (*lo - begin) as i64 + k)).collect();
        let emb = Tensor::cat(&pieces.into_iter().map(|p| p.1).collect::<Vec<_>>(), 0);
        self.encode_ms += t0.elapsed().as_secs_f64() * 1e3;
        Some((rows, emb))
    }
}

/// Canvas frames of the given segments: an image reads frame 0, a video group frames [f, f + 2).
fn read_frames(it: &MmItem, firsts: &[i64], dev: Device) -> Tensor {
    use std::io::{Read, Seek, SeekFrom};
    let fb = (it.height * it.width * 3) as usize;
    let mut f = std::fs::File::open(&it.path).unwrap_or_else(|e| panic!("mm canvas {}: {e}", it.path.display()));
    let per = if it.video { 2 } else { 1 };
    let mut buf = vec![0u8; fb * per * firsts.len()];
    for (k, &first) in firsts.iter().enumerate() {
        f.seek(SeekFrom::Start(first as u64 * fb as u64)).unwrap();
        f.read_exact(&mut buf[k * fb * per..(k + 1) * fb * per]).unwrap_or_else(|e| panic!("mm canvas {}: {e}", it.path.display()));
    }
    Tensor::from_slice(&buf).view([(per * firsts.len()) as i64, it.height, it.width, 3]).to_device(dev)
}

/// `qwen-vision-probe <model_dir> <canvas.u8> <frames> <height> <width> <out>`: encode one canvas with the serving code and
/// write <out>.pixels.f32 / <out>.embeds.f32 (comparison with an FP32 reference).
pub fn probe(dir: &std::path::Path, canvas: &str, frames: i64, h: i64, w: i64, out: &str) {
    let _g = tch::no_grad_guard();
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let ck = Checkpoint::open(dir, &["vision_k6.safetensors"], Device::Cuda(0));
    let v = Vision::load(&ck, &cfg);
    // a ".f32" input holds patch rows (the HF processor's pixel_values) instead of a uint8 canvas
    let (px, g, gh, gw) = if canvas.ends_with(".f32") {
        let b = std::fs::read(canvas).unwrap();
        let f: Vec<f32> = b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let g = if frames == 1 { 1 } else { frames / 2 };
        let (gh, gw) = (h / 16, w / 16);
        (Tensor::from_slice(&f).view([g * gh * gw, 1536]).to_device(Device::Cuda(0)), g, gh, gw)
    } else {
        let it = MmItem { path: PathBuf::from(canvas), frames, height: h, width: w, video: frames > 1, segments: vec![] };
        let firsts: Vec<i64> = if frames == 1 { vec![0] } else { (0..frames / 2).map(|g| 2 * g).collect() };
        let c = read_frames(&it, &firsts, Device::Cuda(0));
        v.pixels(&c)
    };
    let _ = v.forward(&px, g, gh, gw);
    tch::Cuda::synchronize(0);
    let t0 = std::time::Instant::now();
    let y = v.forward(&px, g, gh, gw);
    tch::Cuda::synchronize(0);
    eprintln!("[qwen-vision-probe] grid {g}x{gh}x{gw}: forward {:.1} ms -> {:?}", t0.elapsed().as_secs_f64() * 1e3, y.size());
    let dump = |t: &Tensor, s: &str| {
        let v = Vec::<f32>::try_from(t.to_kind(Kind::Float).to_device(Device::Cpu).reshape([-1])).unwrap();
        std::fs::write(format!("{out}.{s}.f32"), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    };
    dump(&px, "pixels");
    dump(&y, "embeds");
}
