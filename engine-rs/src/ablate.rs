//! Refusal-direction ablation shared by the GLM and Qwen paths (docs/abliteration.md): removes one direction d (unit
//! vector in the hidden space) from everything the blocks write into the residual streams of the selected layers.
//!
//! SPARK_ABLATE=<file.safetensors> (off when unset): tensor `direction` [hidden] (any float type);
//! SPARK_ABLATE_LAYERS=a-b limits the layers (default all), SPARK_ABLATE_FROM_LAYER=n uses row n of `directions`. Weights that write into the residual stream and are not quantized are
//! orthogonalized at load (`ortho_out`, `ortho_rows`); outputs of quantized projections are projected at run time
//! (`apply`, graph-capturable). The capture helpers record each layer's mixed block input at the last prompt row for
//! computing directions (`spark-engine qwen-refusal-dir` / `glm-refusal-dir`).
use std::sync::OnceLock;
use tch::{Device, Kind, Tensor};

/// d: the direction(s): [hidden] (one for every layer), [k, hidden] orthonormal rows (a subspace, every layer), or with
/// `per_layer` [layers, hidden] (layer i removes row i).
pub struct Ablation { d: Tensor, layers: std::ops::RangeInclusive<usize>, per_layer: bool }
// The direction is written once at load and only read afterwards (all CUDA work on the engine thread).
unsafe impl Sync for Ablation {}

fn parse_layers(s: &str) -> std::ops::RangeInclusive<usize> {
    let (a, b) = s.split_once('-').unwrap_or((s, s));
    a.trim().parse::<usize>().expect("ablation layers")..=b.trim().parse::<usize>().expect("ablation layers")
}

/// The configured ablation (loaded once), None when SPARK_ABLATE is unset.
pub fn get() -> Option<&'static Ablation> {
    static A: OnceLock<Option<Ablation>> = OnceLock::new();
    A.get_or_init(|| {
        let path = std::env::var("SPARK_ABLATE").ok().filter(|p| !p.is_empty())?;
        let tensors = Tensor::read_safetensors(&path).unwrap_or_else(|e| panic!("SPARK_ABLATE {path}: {e}"));
        // SPARK_ABLATE_FROM_LAYER=n: take row n of `directions` instead of the stored choice
        let pick = std::env::var("SPARK_ABLATE_FROM_LAYER").ok().and_then(|v| v.parse::<i64>().ok());
        let d = match pick {
            Some(n) => tensors.iter().find(|(k, _)| k == "directions").unwrap_or_else(|| panic!("SPARK_ABLATE {path}: no `directions`")).1.get(n),
            None => tensors.iter().find(|(k, _)| k == "direction").unwrap_or_else(|| panic!("SPARK_ABLATE {path}: no `direction` tensor")).1.shallow_clone(),
        }.to_kind(Kind::Float).to_device(Device::Cuda(0)).contiguous();
        let d = &d / d.norm();
        let layers = std::env::var("SPARK_ABLATE_LAYERS").ok().map(|s| parse_layers(&s)).unwrap_or(0..=usize::MAX);
        let dirs = || tensors.iter().find(|(k, _)| k == "directions").unwrap_or_else(|| panic!("SPARK_ABLATE {path}: no `directions`")).1
            .to_kind(Kind::Float).to_device(Device::Cuda(0));
        // SPARK_ABLATE_MODE: `single` (default), `per-layer` (layer i removes its own direction), `subspace:a-b:k` (the top k
        // singular directions of layers a..b's directions, removed everywhere)
        let mode = std::env::var("SPARK_ABLATE_MODE").unwrap_or_else(|_| "single".into());
        let (d, per_layer) = if mode == "per-layer" {
            let all = dirs(); let n = all.norm_scalaropt_dim(2, [1i64].as_slice(), true); ((&all / n).contiguous(), true)
        } else if let Some(spec) = mode.strip_prefix("subspace:") {
            let (r, k) = spec.rsplit_once(':').expect("subspace:a-b:k");
            let rr = parse_layers(r); let k: i64 = k.parse().unwrap();
            // rows: every layer's per-position directions when stored, else one direction per layer
            let sel = match tensors.iter().find(|(k, _)| k == "directions_pos") {
                Some((_, t)) => { let t = t.to_kind(Kind::Float); let n = t.size(); t.narrow(0, *rr.start() as i64, (*rr.end() - *rr.start() + 1) as i64).reshape([-1, n[2]]) }
                None => dirs().narrow(0, *rr.start() as i64, (*rr.end() - *rr.start() + 1) as i64),
            };
            let (_, _, v) = Tensor::svd(&sel.to_device(Device::Cpu), true, true);   // v [hidden, n]
            (v.narrow(1, 0, k).tr().contiguous().to_device(Device::Cuda(0)), false)
        } else { (d.unsqueeze(0), false) };
        eprintln!("[ablate] {path}: mode {mode}, {} direction rows of hidden {}, layers {:?}", d.size()[0], d.size()[1], layers);
        Some(Ablation { d, layers, per_layer })
    }).as_ref()
}

impl Ablation {
    pub fn covers(&self, layer: usize) -> bool { self.layers.contains(&layer) }
    /// The direction rows that apply to `layer` ([k, hidden], orthonormal).
    fn rows(&self, layer: usize) -> Tensor { if self.per_layer { self.d.narrow(0, layer.min(self.d.size()[0] as usize - 1) as i64, 1) } else { self.d.shallow_clone() } }
    /// W [hidden, in] of a projection writing into the residual stream: W - D^T (D W) (same kind as W).
    pub fn ortho_out(&self, layer: usize, w: &Tensor) -> Tensor {
        let wf = w.to_kind(Kind::Float);
        let d = self.rows(layer).to_device(w.device());
        (&wf - d.tr().matmul(&d.matmul(&wf))).to_kind(w.kind())
    }
    /// E [rows, hidden] whose rows are hidden vectors (embeddings): E - (E D^T) D.
    pub fn ortho_rows(&self, layer: usize, e: &Tensor) -> Tensor {
        let ef = e.to_kind(Kind::Float);
        let d = self.rows(layer).to_device(e.device());
        (&ef - ef.matmul(&d.tr()).matmul(&d)).to_kind(e.kind())
    }
    /// In place: y [..., hidden] fp32 (rows with stride, contiguous columns) loses its components along the rows of D.
    pub fn apply(&self, layer: usize, y: &Tensor) {
        extern "C" { fn spark_ablate_rows_k(y: *mut std::ffi::c_void, ld: i64, r: i32, d: *const std::ffi::c_void, dim: i32, k: i32, st: *mut std::ffi::c_void) -> i32;
                     fn rs_current_stream() -> *mut std::ffi::c_void; }
        assert_eq!(y.kind(), Kind::Float);
        let s = y.size();
        let (r, dim) = (s[..s.len() - 1].iter().product::<i64>(), s[s.len() - 1]);
        assert_eq!(y.stride()[y.dim() - 1], 1);
        let d = self.rows(layer).contiguous();   // [k, dim], orthonormal unit rows
        let k = d.size()[0];
        if r == 0 || k == 0 { return; }
        // One launch removes all k directions (read each row once); orthonormal D => equals sequential projection.
        assert_eq!(unsafe { spark_ablate_rows_k(y.data_ptr(), y.stride()[y.dim() - 2], r as i32, d.data_ptr(), dim as i32, k as i32, rs_current_stream()) }, 0, "ablate rows");
    }
}

/// Runtime projection for layer `layer` (no-op without ablation or outside the range).
pub fn apply(layer: usize, y: &Tensor) { if let Some(a) = get() { if a.covers(layer) { a.apply(layer, y); } } }

// ---- direction extraction: per-layer block inputs at the last prompt row ----
thread_local! { static CAPTURE: std::cell::RefCell<Option<Vec<Option<Tensor>>>> = const { std::cell::RefCell::new(None) }; }
pub fn capture_start() { CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new())); }
/// The captured rows of one prompt, by layer (a layer seen in several prefill pieces keeps its last piece's row).
pub fn capture_take() -> Vec<Tensor> {
    CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default().into_iter().map(|t| t.expect("layer not captured")).collect())
}
/// Record the last P rows of x [T, hidden] (a block's mixed input) for `layer`, as fp32 on the host [P, hidden]
/// (P = SPARK_ABLATE_POSITIONS, default 8: the end of the prompt, i.e. the assistant turn's opening tokens).
pub fn capture(layer: usize, x: &Tensor) {
    let p = std::env::var("SPARK_ABLATE_POSITIONS").ok().and_then(|v| v.parse::<i64>().ok()).unwrap_or(8).min(x.size()[0]);
    CAPTURE.with(|c| if let Some(v) = c.borrow_mut().as_mut() {
        if v.len() <= layer { v.resize_with(layer + 1, || None); }
        v[layer] = Some(x.narrow(0, x.size()[0] - p, p).to_kind(Kind::Float).to_device(Device::Cpu));
    });
}
pub fn capturing() -> bool { CAPTURE.with(|c| c.borrow().is_some()) }

/// Directions from captured activations ([P, hidden] per prompt and layer): per layer and position
/// mean(harmful) - mean(harmless) -> `pos` [layers, P, hidden] (normalized), the position-averaged direction per layer
/// [layers, hidden] (normalized), and the separation score per layer (|mean difference| / mean activation norm).
pub fn directions(harmful: &[Vec<Tensor>], harmless: &[Vec<Tensor>]) -> (Tensor, Tensor, Vec<f64>) {
    let mean = |set: &[Vec<Tensor>]| -> Tensor { Tensor::stack(&set.iter().map(|v| Tensor::stack(v, 0)).collect::<Vec<_>>(), 0).mean_dim(0, false, Kind::Float) };
    let (mh, mn) = (mean(harmful), mean(harmless));          // [layers, P, hidden]
    let diff = &mh - &mn;
    let pos = &diff / diff.norm_scalaropt_dim(2, [2i64].as_slice(), true);
    let avg = diff.mean_dim(1, false, Kind::Float);          // [layers, hidden]
    let norms = avg.norm_scalaropt_dim(2, [1i64].as_slice(), true);
    let scale = (mh.norm_scalaropt_dim(2, [2i64].as_slice(), false) + mn.norm_scalaropt_dim(2, [2i64].as_slice(), false)).mean_dim(1, false, Kind::Float) / 2.0;
    let score: Vec<f64> = Vec::try_from((norms.squeeze_dim(1) / scale).to_device(Device::Cpu)).unwrap();
    (pos, &avg / norms, score)
}

/// Write `directions`, `scores` and the chosen `direction` (see the refusal-dir tools) to `out`.
pub fn save_directions(harmful: &[Vec<Tensor>], harmless: &[Vec<Tensor>], out: &std::path::Path) {
    let (pos, dirs, score) = directions(harmful, harmless);
    let layers = score.len();
    let pick = std::env::var("SPARK_ABLATE_FROM_LAYER").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or_else(|| {
        (layers / 4..=(3 * layers / 4).min(layers - 1)).max_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap()).unwrap()
    });
    for (i, s) in score.iter().enumerate() { eprintln!("[refusal-dir] layer {i:2}: separation {s:.4}{}", if i == pick { "  <- direction" } else { "" }); }
    let named = [("directions_pos", pos), ("directions", dirs.shallow_clone()), ("scores", Tensor::from_slice(&score).to_kind(Kind::Float)), ("direction", dirs.get(pick as i64)),
                 ("layer", Tensor::from_slice(&[pick as i64]))];
    Tensor::write_safetensors(&named, out).expect("write directions");
    eprintln!("[refusal-dir] wrote {} (direction from layer {pick})", out.display());
}

/// `glm-refusal-dir <model> <harmful.json> <harmless.json> <out.safetensors>` (both TP2 ranks, rank 0 writes): as
/// qwen-refusal-dir for GLM-5.3-Flash. The replicated prefill path is used (sequence-parallel prefill off) so every rank
/// sees the last prompt rows.
pub fn glm_refusal_dir(model: &std::path::Path, harmful: &std::path::Path, harmless: &std::path::Path, out: &std::path::Path) {
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    for flag in ["GLM53_MHC_FUSED", "GLM53_KDA_FUSED", "GLM53_MLA_LATENT"] { std::env::set_var(flag, "1"); }
    std::env::set_var("GLM53_PREFILL_SP", "0");
    let dev = Device::Cuda(0);
    let cfg = crate::config::load(&model.join("config.json")).unwrap();
    let w = crate::weights::ModelWeights::load(model, &cfg, cfg.num_hidden_layers, dev);
    let mut fast = crate::moefast::MoeFast::new(model, cfg.num_hidden_layers, cfg.n_routed_experts, cfg.num_hidden_layers * cfg.n_routed_experts + 16, dev);
    fast.preload_all(cfg.num_hidden_layers, cfg.n_routed_experts, dev);
    fast.assume_hot = true;
    let mut eng = crate::forward::Engine { w, fast: Some(fast), native: None, pool: crate::moe::ExpertPool::new(model, 4) };
    let load = |p: &std::path::Path| -> Vec<Vec<i64>> { serde_json::from_reader(std::fs::File::open(p).expect("prompt ids json")).expect("prompt ids") };
    let mut run = |set: Vec<Vec<i64>>, name: &str| -> Vec<Vec<Tensor>> {
        let t0 = std::time::Instant::now();
        let n = set.len();
        let r = set.iter().map(|ids| {
            capture_start();
            let _ = eng.prefill(&Tensor::from_slice(ids).to_device(dev));
            tch::Cuda::synchronize(0);
            capture_take()
        }).collect();
        eprintln!("[refusal-dir] rank{} {name}: {n} prompts in {:.1}s", tp.rank, t0.elapsed().as_secs_f64());
        r
    };
    let (h, n) = (run(load(harmful), "harmful"), run(load(harmless), "harmless"));
    if tp.rank == 0 { save_directions(&h, &n, out); }
}
