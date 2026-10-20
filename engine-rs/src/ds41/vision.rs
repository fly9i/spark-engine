//! DeepSeek-V4.1-Flash image input (DS41_VISION=1, rank 0): the DeepSeek-ViT (32 blocks of 1024, 16 heads, 2D RoPE,
//! patch 14, BF16 as stored), the aligner (3x3 pixel-unshuffle + two-layer MLP to the 5120-wide embedding) and the
//! span delimiters, as the official inference/vision.py + model.py `merge_image_embeddings`.
//!
//! An image of canvas H x W (multiples of 14, gray-padded by the front end: serve/mm_ds41.py, official
//! image_processor.py) is a gh x gw token grid (gh = ceil(H / 42), gw = ceil(W / 42)) the prompt sees as
//!   [IMAGE_START] + ([IMAGE] * gw + [IMAGE_NEW_LINE]) * gh + [IMAGE_END]
//! Every position of the span is a salted placeholder (crate::mm_canvas); the backbone treats them as image tokens:
//! the MoE gates route them with `bias_vl`, the engram skips them (n-gram history DEAD, gate shut), as the official
//! model does with `token_types >= 0`.
use super::load::Checkpoint;
use crate::mm_canvas::{read_frames, MmItem};
use tch::{Device, Kind, Tensor};

pub fn enabled() -> bool { std::env::var("DS41_VISION").as_deref() == Ok("1") }

struct Block { n1: Tensor, wqkv: Tensor, bqkv: Tensor, wo: Tensor, bo: Tensor, n2: Tensor, w1: Tensor, w2: Tensor }

pub struct Vision {
    dim: i64, heads: i64, patch: i64, ratio: i64, max_tokens: i64,
    pe_w: Tensor, pe_b: Tensor, blocks: Vec<Block>, norm: Tensor,
    a_w1: Tensor, a_b1: Tensor, a_w2: Tensor, a_b2: Tensor,
    /// [3, dim] BF16: IMAGE_START, IMAGE_END, IMAGE_NEW_LINE
    delims: Tensor,
    inv_freq: Tensor,
    pub dev: Device,
}

impl Vision {
    /// Weights from the original checkpoint (`ck`: the shards do not carry them; ~0.95 GB BF16).
    pub fn load(ck: &Checkpoint, cfg_dir: &std::path::Path) -> Vision {
        let v: serde_json::Value = serde_json::from_reader(std::fs::File::open(cfg_dir.join("inference/config.json")).expect("inference/config.json")).unwrap();
        let i = |k: &str| v[k].as_i64().unwrap_or_else(|| panic!("config {k}"));
        let (layers, dim, heads, patch, ratio) = (i("vision_n_layers"), i("vision_dim"), i("vision_n_heads"), i("vision_patch_size"), i("vision_downsample_ratio"));
        let theta = v["vision_rope_theta"].as_f64().unwrap_or(10000.0);
        let g = |k: &str| ck.get(k, Kind::BFloat16);
        // RMSNorm weights are FP32 parameters in the official module (loaded from BF16: exact)
        let f = |k: &str| ck.get(k, Kind::BFloat16).to_kind(Kind::Float);
        let blocks = (0..layers).map(|l| {
            let b = |n: &str| format!("vision.blocks.{l}.{n}");
            Block { n1: f(&b("norm1.weight")), wqkv: g(&b("attn.wqkv.weight")), bqkv: g(&b("attn.wqkv.bias")), wo: g(&b("attn.wo.weight")),
                    bo: g(&b("attn.wo.bias")), n2: f(&b("norm2.weight")), w1: g(&b("mlp.w1.weight")), w2: g(&b("mlp.w2.weight")) }
        }).collect();
        let dev = ck.dev;
        // inv_freq = 1 / theta^(arange(0, rope_dim, 2) / rope_dim), rope_dim = head_dim / 2, FP32
        let rd = dim / heads / 2;
        let ex = Tensor::arange_start_step(0, rd, 2, (Kind::Float, dev)) / (rd as f64);
        let p = Tensor::from(theta as f32).to_device(dev).pow(&ex);
        let inv_freq = p.ones_like() / p;
        let delims = Tensor::stack(&[g("image_start"), g("image_end"), g("image_newline")], 0);
        let vis = Vision { dim, heads, patch, ratio, max_tokens: i("vision_max_n_token"), pe_w: g("vision.patch_embed.proj.weight"),
            pe_b: g("vision.patch_embed.proj.bias"), blocks, norm: f("vision.norm.weight"), a_w1: g("aligner.w1.weight"), a_b1: g("aligner.w1.bias"),
            a_w2: g("aligner.w2.weight"), a_b2: g("aligner.w2.bias"), delims, inv_freq, dev };
        eprintln!("[ds41-vision] loaded: {layers} blocks of {dim}, {heads} heads, patch {patch}, downsample {ratio}, theta {theta}");
        vis
    }

    /// Token grid (gh, gw) of a canvas H x W.
    pub fn grid(&self, h: i64, w: i64) -> (i64, i64) {
        ((h / self.patch + self.ratio - 1) / self.ratio, (w / self.patch + self.ratio - 1) / self.ratio)
    }
    /// The span length of an item (mm_canvas::parse_items' `seg_len`): images only, canvas sides multiples of the patch.
    pub fn seg_len(&self, it: &MmItem) -> Result<usize, String> {
        if it.frames != 1 { return Err("DeepSeek-V4.1 takes images only (no video)".into()); }
        if it.height % self.patch != 0 || it.width % self.patch != 0 { return Err(format!("image canvas {}x{} is not a multiple of {}", it.height, it.width, self.patch)); }
        let (gh, gw) = self.grid(it.height, it.width);
        let n = gh * (gw + 1) + 2;
        if n > self.max_tokens { return Err(format!("image canvas {}x{} is {n} tokens, above {}", it.height, it.width, self.max_tokens)); }
        Ok(n as usize)
    }

    fn rms(x: &Tensor, w: &Tensor) -> Tensor {
        let xf = x.to_kind(Kind::Float);
        let xf = &xf * (xf.square().mean_dim([-1i64].as_slice(), true, Kind::Float) + 1e-6).rsqrt();
        (w * xf).to_kind(x.kind())
    }

    /// Canvas uint8 [1, H, W, 3] -> patches [nh * nw, 3 * p * p] BF16 (row-major patches, (c, y, x) inside).
    pub fn patches(&self, canvas: &Tensor) -> (Tensor, i64, i64) {
        let s = canvas.size();
        let (h, w, p) = (s[1], s[2], self.patch);
        let (nh, nw) = (h / p, w / p);
        let x = canvas.get(0).permute([2, 0, 1]).to_kind(Kind::Float) / 255.0;
        let x = ((x - 0.5) / 0.5).to_kind(Kind::BFloat16);
        (x.reshape([3, nh, p, nw, p]).permute([1, 3, 0, 2, 4]).reshape([nh * nw, 3 * p * p]), nh, nw)
    }

    /// ViT + aligner: patches [nh * nw, 588] -> aligner rows [gh * gw, dim_llm] BF16 (row-major grid).
    pub fn encode(&self, px: &Tensor, nh: i64, nw: i64) -> Tensor {
        let _g = tch::no_grad_guard();
        let n = nh * nw;
        let (heads, hd) = (self.heads, self.dim / self.heads);
        let mut x = px.linear(&self.pe_w, Some(&self.pe_b));
        // 2D RoPE: [h * inv_freq, w * inv_freq] per patch, rotate-half over the head's two halves
        let hp = Tensor::arange(nh, (Kind::Int64, self.dev)).view([nh, 1]).expand([nh, nw], false).reshape([n, 1]).to_kind(Kind::Float);
        let wp = Tensor::arange(nw, (Kind::Int64, self.dev)).view([1, nw]).expand([nh, nw], false).reshape([n, 1]).to_kind(Kind::Float);
        let fr = Tensor::cat(&[hp * self.inv_freq.unsqueeze(0), wp * self.inv_freq.unsqueeze(0)], 1);
        let (cos, sin) = (fr.cos().unsqueeze(1), fr.sin().unsqueeze(1));    // [n, 1, hd / 2]
        let rot = |t: &Tensor| {
            let tf = t.to_kind(Kind::Float);
            let (a, b) = (tf.narrow(2, 0, hd / 2), tf.narrow(2, hd / 2, hd / 2));
            Tensor::cat(&[&a * &cos - &b * &sin, &b * &cos + &a * &sin], 2).to_kind(t.kind())
        };
        for b in &self.blocks {
            let qkv = Self::rms(&x, &b.n1).linear(&b.wqkv, Some(&b.bqkv));
            let (q, k, v) = (qkv.narrow(1, 0, self.dim), qkv.narrow(1, self.dim, self.dim), qkv.narrow(1, 2 * self.dim, self.dim));
            let (q, k, v) = (rot(&q.view([n, heads, hd])), rot(&k.view([n, heads, hd])), v.reshape([n, heads, hd]));
            // [1, heads, n, hd]: the fused (memory-efficient / flash) kernels need 4-D inputs; the reference's 3-D call takes the
            // math path, which materializes heads x n x n scores (4.4 GB at the largest image)
            let sd = |t: &Tensor| t.transpose(0, 1).unsqueeze(0);
            let o = Tensor::scaled_dot_product_attention(&sd(&q), &sd(&k), &sd(&v), None::<Tensor>, 0.0, false, None::<f64>, false);
            x = x + o.squeeze_dim(0).transpose(0, 1).reshape([n, self.dim]).linear(&b.wo, Some(&b.bo));
            let gu = Self::rms(&x, &b.n2).linear::<Tensor>(&b.w1, None);
            let il = gu.size()[1] / 2;
            x = x + (gu.narrow(1, 0, il).silu() * gu.narrow(1, il, il)).linear::<Tensor>(&b.w2, None);
        }
        let x = Self::rms(&x, &self.norm);
        // aligner: zero-pad the grid to multiples of r, unfold r x r windows (feature c * r * r + ky * r + kx)
        let r = self.ratio;
        let (gh, gw) = ((nh + r - 1) / r, (nw + r - 1) / r);
        let pad = Tensor::zeros([gh * r, gw * r, self.dim], (x.kind(), self.dev));
        let _ = pad.narrow(0, 0, nh).narrow(1, 0, nw).copy_(&x.view([nh, nw, self.dim]));
        let u = pad.view([gh, r, gw, r, self.dim]).permute([0, 2, 4, 1, 3]).reshape([gh * gw, self.dim * r * r]);
        u.linear(&self.a_w1, Some(&self.a_b1)).gelu("none").linear(&self.a_w2, Some(&self.a_b2))
    }

    /// The rows of an image's whole span [gh * (gw + 1) + 2, dim_llm] BF16 from its canvas.
    pub fn span(&self, canvas: &Tensor) -> Tensor {
        let (px, nh, nw) = self.patches(canvas);
        let rows = self.encode(&px, nh, nw);
        let (gh, gw) = self.grid(nh * self.patch, nw * self.patch);
        // rows of [aligner rows; start; end; newline] in span order
        let l = gh * gw;
        let mut idx = Vec::with_capacity((gh * (gw + 1) + 2) as usize);
        idx.push(l);
        for y in 0..gh { idx.extend((0..gw).map(|x| y * gw + x)); idx.push(l + 2); }
        idx.push(l + 1);
        Tensor::cat(&[rows, self.delims.shallow_clone()], 0).index_select(0, &Tensor::from_slice(&idx).to_device(self.dev))
    }

    /// mm_canvas::MmState's encoder: one span per segment (images: one segment, frame 0).
    pub fn encode_segments(&self, it: &MmItem, segs: &[usize]) -> Vec<Tensor> {
        segs.iter().map(|&si| self.span(&read_frames(it, &[it.segments[si].2], 1, self.dev))).collect()
    }
}

/// `ds41-vision-probe <ckpt> <canvas.u8> <height> <width> <out>`: the span rows of one canvas with the serving code path,
/// raw little-endian FP32 in <out>.span.f32 (and the patches in <out>.pixels.f32), for the comparison with the official
/// vision.py (bench/ds41/vision_ref.py).
pub fn probe(args: &[String]) {
    let _g = tch::no_grad_guard();
    let dev = Device::Cuda(0);
    let ckpt = std::path::Path::new(&args[2]);
    let ck = Checkpoint::open(ckpt, dev);
    let v = Vision::load(&ck, ckpt);
    let (h, w): (i64, i64) = (args[4].parse().unwrap(), args[5].parse().unwrap());
    let it = MmItem { path: args[3].clone().into(), frames: 1, height: h, width: w, segments: vec![(0, 0, 0)] };
    eprintln!("[ds41-vision-probe] span length {:?}", v.seg_len(&it));
    let canvas = read_frames(&it, &[0], 1, dev);
    let _ = v.span(&canvas);
    tch::Cuda::synchronize(0);
    let t0 = std::time::Instant::now();
    let y = v.span(&canvas);
    tch::Cuda::synchronize(0);
    eprintln!("[ds41-vision-probe] encode {:.1} ms -> {:?}", t0.elapsed().as_secs_f64() * 1e3, y.size());
    let dump = |t: &Tensor, suffix: &str| {
        let v = Vec::<f32>::try_from(t.to_kind(Kind::Float).to_device(Device::Cpu).reshape([-1])).unwrap();
        std::fs::write(format!("{}.{suffix}.f32", args[6]), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    };
    dump(&v.patches(&canvas).0, "pixels");
    dump(&y, "span");
}
