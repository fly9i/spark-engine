//! Refusal-direction ablation shared by the GLM and Qwen paths (docs/abliteration.md): removes one direction d (unit
//! vector in the hidden space) from everything the blocks write into the residual streams of the selected layers.
//!
//! SPARK_ABLATE=<file.safetensors> (off when unset): tensor `direction` [hidden] (any float type);
//! SPARK_ABLATE_LAYERS=a-b limits the layers (default all), SPARK_ABLATE_FROM_LAYER=n uses row n of `directions`. Weights that write into the residual stream and are not quantized are
//! orthogonalized at load (`ortho_out`); embedding rows (at lookup, the drafter keeps the stored table) and outputs of quantized projections are projected at run time
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
        // SPARK_ABLATE_SCALE=a (default 1, 0 < a <= 1): remove only the fraction a of the components (y - a D^T D y, and
        // W - a D^T D W at load), by storing sqrt(a) D: every consumer (apply, ortho_out, ortho_rows) is quadratic in D.
        let scale: f64 = std::env::var("SPARK_ABLATE_SCALE").ok().map(|v| v.parse().expect("SPARK_ABLATE_SCALE")).unwrap_or(1.0);
        assert!(scale > 0.0 && scale <= 1.0, "SPARK_ABLATE_SCALE must be in (0, 1]");
        let d = if scale == 1.0 { d } else { (d * scale.sqrt()).contiguous() };
        eprintln!("[ablate] {path}: mode {mode}, {} direction rows of hidden {}, layers {:?}, scale {scale}", d.size()[0], d.size()[1], layers);
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

/// SPARK_ABLATE_WRITERS=oproj: SPARK_ABLATE edits only the attention o_proj outputs (no routed / shared / dense MLP
/// outputs, no embedding rows). Default: every residual writer.
pub fn oproj_only() -> bool { static O: OnceLock<bool> = OnceLock::new(); *O.get_or_init(|| std::env::var("SPARK_ABLATE_WRITERS").as_deref() == Ok("oproj")) }
/// Runtime projection of a non-attention writer (routed experts) for layer `layer` (no-op without ablation, outside the
/// range, or with SPARK_ABLATE_WRITERS=oproj).
pub fn apply(layer: usize, y: &Tensor) { if oproj_only() { return; } apply_marked(layer, y); }
/// Runtime projection of a writer selected at load (SPX o_proj / down outputs).
pub fn apply_marked(layer: usize, y: &Tensor) { if let Some(a) = get() { if a.covers(layer) { a.apply(layer, y); } } }

/// SPARK_ABLATE_CLAMP=<file.safetensors>: conditional ablation of what every sublayer reads. Tensors `directions`
/// [k, hidden] (orthonormal rows), `tau` / `tau_lo` [layers, 2, k] (site 0: attention input, 1: MLP input) and
/// `tau_final` / `tau_final_lo` [k] (the final hidden before lm_head). A row z of a sublayer input (the MHC pre output)
/// has its component along each direction clipped into [tau_lo, tau]: z - sum_j (relu(c_j - tau_j) - relu(tau_lo_j - c_j)) d_j,
/// c = z.d (two-sided: the sign of an SVD direction is arbitrary). Ordinary text stays
/// below the thresholds (calibrated as high quantiles of its projections), so it is left unchanged; the residual
/// streams themselves (and the drafter's features) are never edited. SPARK_ABLATE_CLAMP_STATS=<dir> (rank0, eager
/// prefill only, while <dir>/ON exists): append every site's raw projections z.D^T (f32, [rows, k]) to
/// <dir>/L<layer>-<site>.f32 (site 2 = final) and each call's row count to <dir>/calls.txt (at layer 0 site 0), for
/// calibration. Independent of SPARK_ABLATE / SPARK_ABLATE_OPROJ.
/// Optional `target` [layers, 2, k] / `target_final` [k]: a triggered component (outside [tau_lo, tau]) is moved to the
/// target (e.g. the harmless median) instead of the nearest threshold: z - sum_j [c_j outside] (c_j - target_j) d_j.
/// SPARK_ABLATE_CLAMP_MIN=m (default 1 = every triggered site edits): count gate over the sublayer inputs of one forward
/// pass -- a row is edited at a triggered site only once m sites (this one included, from layer 0's attention input on)
/// have triggered for it, so isolated threshold crossings in ordinary text pass through (q56 stats: tokens edited
/// 11.7% -> 2.4% at m = 8, harmful prompts' last token still covered). The final site is not gated (its rows are not
/// the pass's rows in prefill). Needs SPARK_ABLATE_CLAMP_FUSE >= 1.
pub struct Clamp { d: Tensor, tau: Tensor, tau_lo: Tensor, tau_final: Tensor, tau_final_lo: Tensor, target: Option<Tensor>, target_final: Option<Tensor>, sites: std::sync::Mutex<std::collections::HashMap<usize, (usize, usize)>>,
                   min: i32, count: Tensor }
/// Rows of one forward pass the count gate can track.
const CLAMP_COUNT_ROWS: i64 = 1 << 20;
unsafe impl Sync for Clamp {}
pub fn clamp() -> Option<&'static Clamp> {
    static C: OnceLock<Option<Clamp>> = OnceLock::new();
    C.get_or_init(|| {
        let path = std::env::var("SPARK_ABLATE_CLAMP").ok().filter(|p| !p.is_empty())?;
        let t = Tensor::read_safetensors(&path).unwrap_or_else(|e| panic!("SPARK_ABLATE_CLAMP {path}: {e}"));
        let get = |n: &str| t.iter().find(|(k, _)| k == n).unwrap_or_else(|| panic!("SPARK_ABLATE_CLAMP {path}: no `{n}`")).1
            .to_kind(Kind::Float).to_device(Device::Cuda(0)).contiguous();
        let (d, tau, tau_lo, tau_final, tau_final_lo) = (get("directions"), get("tau"), get("tau_lo"), get("tau_final"), get("tau_final_lo"));
        assert!(d.dim() == 2 && tau.dim() == 3 && tau.size()[1] == 2 && tau.size()[2] == d.size()[0] && tau_final.size() == [d.size()[0]], "SPARK_ABLATE_CLAMP shapes");
        eprintln!("[ablate] clamp {path}: {} directions, thresholds for {} layers{}", d.size()[0], tau.size()[0],
                  if stats_dir().is_some() { ", STATS mode" } else { "" });
        assert!(tau_lo.size() == tau.size() && tau_final_lo.size() == tau_final.size(), "SPARK_ABLATE_CLAMP lower thresholds");
        let opt = |n: &str| t.iter().find(|(k, _)| k == n).map(|(_, v)| v.to_kind(Kind::Float).to_device(Device::Cuda(0)).contiguous());
        let (target, target_final) = (opt("target"), opt("target_final"));
        assert!(target.as_ref().map_or(true, |x| x.size() == tau.size()) && target_final.as_ref().map_or(true, |x| x.size() == tau_final.size()), "SPARK_ABLATE_CLAMP target shapes");
        if target.is_some() { eprintln!("[ablate] clamp: triggered components move to the stored targets"); }
        let min: i32 = std::env::var("SPARK_ABLATE_CLAMP_MIN").ok().map(|v| v.parse().expect("SPARK_ABLATE_CLAMP_MIN")).unwrap_or(1);
        assert!(min >= 1, "SPARK_ABLATE_CLAMP_MIN >= 1");
        if min > 1 {
            assert!(fuse_mode() >= 1, "SPARK_ABLATE_CLAMP_MIN > 1 needs SPARK_ABLATE_CLAMP_FUSE >= 1 (not STATS mode)");
            eprintln!("[ablate] clamp: count gate, a site edits a row once {min} sites have triggered for it");
        }
        let count = Tensor::zeros([if min > 1 { CLAMP_COUNT_ROWS } else { 0 }], (Kind::Int, Device::Cuda(0)));
        Some(Clamp { d, tau, tau_lo, tau_final, tau_final_lo, target, target_final, sites: std::sync::Mutex::new(Default::default()), min, count })
    }).as_ref()
}
fn stats_dir() -> Option<String> { static S: OnceLock<Option<String>> = OnceLock::new(); S.get_or_init(|| std::env::var("SPARK_ABLATE_CLAMP_STATS").ok().filter(|p| !p.is_empty())).clone() }
/// SPARK_ABLATE_CLAMP_FUSE: 0 = ATen chain (reference), 1 = one kernel per site (shim/ablate.cu), 2 (default) = sublayer
/// inputs clamped inside the fused MHC pre finish (shim/mhc.cu), other sites one kernel. STATS mode always uses 0 (it
/// needs the projections on the host). L1 between the modes (dot-product order).
fn fuse_mode() -> u8 {
    static F: OnceLock<u8> = OnceLock::new();
    *F.get_or_init(|| if stats_dir().is_some() { 0 } else { std::env::var("SPARK_ABLATE_CLAMP_FUSE").ok().and_then(|v| v.parse().ok()).unwrap_or(2) })
}
/// Whether an active clamp is NOT fused into the producing kernels (STATS mode or SPARK_ABLATE_CLAMP_FUSE < 2): the
/// DS41 encoder two-micro-batch prefill overlap must then stay off (its per-batch z rows are locally indexed; the
/// fused path carries the row base in the offset count pointer instead).
pub fn unfused_active() -> bool { clamp().is_some() && fuse_mode() < 2 }
/// Kernel view of one site's clamp (`SparkClamp` in shim/ablate_clamp.cuh).
#[repr(C)]
pub struct ClampArgs { d: *const f32, tau: *const f32, lo: *const f32, target: *const f32, k: i32, count: *mut i32, min: i32, reset: i32 }
impl Clamp {
    /// Register a layer's MHC norm weights (load time): in_ln = site 0, post_ln = site 1.
    pub fn register(&self, ln: &Tensor, layer: usize, site: usize) { self.sites.lock().unwrap().insert(ln.data_ptr() as usize, (layer, site)); }
    fn site(&self, ln: &Tensor) -> Option<(usize, usize)> {
        let (layer, site) = self.sites.lock().unwrap().get(&(ln.data_ptr() as usize)).copied()?;
        ((layer as i64) < self.tau.size()[0]).then_some((layer, site))
    }
    /// `gate`: Some((layer, site)) for a sublayer input (count-gated when SPARK_ABLATE_CLAMP_MIN > 1; layer 0 site 0 starts
    /// the pass), None for the final site; rows = rows of z.
    fn args(&self, tau: &Tensor, lo: &Tensor, target: Option<&Tensor>, gate: Option<(usize, usize)>, rows: i64) -> ClampArgs {
        let gated = self.min > 1 && gate.is_some();
        if gated { assert!(rows <= CLAMP_COUNT_ROWS, "SPARK_ABLATE_CLAMP_MIN: {rows} rows in one pass"); }
        ClampArgs { d: self.d.data_ptr() as *const f32, tau: tau.data_ptr() as *const f32, lo: lo.data_ptr() as *const f32,
                    target: target.map_or(std::ptr::null(), |t| t.data_ptr() as *const f32), k: self.d.size()[0] as i32,
                    count: if gated { self.count.data_ptr() as *mut i32 } else { std::ptr::null_mut() }, min: self.min,
                    reset: i32::from(gate == Some((0, 0))) }
    }
    /// The clamp of the sublayer input normed by `ln` for the fused MHC pre finish (None: no clamp at this site, or
    /// SPARK_ABLATE_CLAMP_FUSE < 2 -- the caller then runs `apply_input` on z).
    pub fn mhc_args(&self, ln: &Tensor, rows: i64) -> Option<ClampArgs> {
        if fuse_mode() < 2 { return None; }
        let (layer, site) = self.site(ln)?;
        let (l, s) = (layer as i64, site as i64);
        Some(self.args(&self.tau.get(l).get(s), &self.tau_lo.get(l).get(s), self.target.as_ref().map(|t| t.get(l).get(s)).as_ref(), Some((layer, site)), rows))
    }
    /// The same fused-site args for the DS41 rms_norm vec epilogue (docs/05-消融.md §3a), with the count-gate rows
    /// based at `off`: the encoder two-micro-batch path's batch-B z rows are the pass's rows [off, off + rows), so its
    /// count pointer is offset (GLM's mhc launcher does the same by bumping the pointer).
    pub fn norm_args(&self, ln: &Tensor, rows: i64) -> Option<ClampArgs> { self.norm_args_off(ln, rows, 0) }
    pub fn norm_args_off(&self, ln: &Tensor, rows: i64, off: i64) -> Option<ClampArgs> {
        let mut a = self.mhc_args(ln, rows)?;
        if !a.count.is_null() { a.count = unsafe { a.count.add(off.max(0) as usize) }; }
        Some(a)
    }
    /// The count gate's per-row counters (None without the gate): DS41's sequence-parallel encoder gathers each rank's
    /// rows of them back so every rank holds the whole pass's counts, as when every rank ran every row.
    pub fn gate_counts(&self) -> Option<&Tensor> { (self.min > 1).then_some(&self.count) }
    /// Test accessors for the ds41-clamp-test fused-epilogue check.
    pub fn site_thresholds(&self, l: usize, s: usize) -> (Tensor, Tensor, Option<Tensor>) {
        (self.tau.get(l as i64).get(s as i64).shallow_clone(), self.tau_lo.get(l as i64).get(s as i64).shallow_clone(),
         self.target.as_ref().map(|t| t.get(l as i64).get(s as i64).shallow_clone()))
    }
    pub fn dirs(&self) -> &Tensor { &self.d }
    pub fn min_m(&self) -> i32 { self.min }
    /// One launch (shim/ablate.cu) on z [rows, hidden] fp32, unit column stride.
    fn clamp_rows(&self, z: &Tensor, tau: &Tensor, lo: &Tensor, target: Option<&Tensor>, gate: Option<(usize, usize)>) {
        extern "C" { fn spark_ablate_clamp_rows(y: *mut std::ffi::c_void, ld: i64, r: i32, d: *const f32, dim: i32, k: i32, tau: *const f32, lo: *const f32,
                                                target: *const f32, count: *mut i32, min: i32, reset: i32, st: *mut std::ffi::c_void) -> i32;
                     fn rs_current_stream() -> *mut std::ffi::c_void; }
        let s = z.size();
        assert!(z.dim() == 2 && z.stride()[1] == 1, "clamp rows");
        let a = self.args(tau, lo, target, gate, s[0]);
        assert_eq!(unsafe { spark_ablate_clamp_rows(z.data_ptr(), z.stride()[0], s[0] as i32, a.d, s[1] as i32, a.k, a.tau, a.lo, a.target, a.count, a.min, a.reset, rs_current_stream()) }, 0, "clamp rows");
    }
    /// The ATen reference given the projections c = z D^T.
    fn clamp_aten(&self, z: &Tensor, c: &Tensor, tau: &Tensor, lo: &Tensor, target: Option<&Tensor>) {
        let e = match target {
            None => (c - tau).relu() - (lo - c).relu(),                    // excess outside [lo, tau]
            Some(t) => (c - t) * (c.gt_tensor(tau).logical_or(&c.lt_tensor(lo))).to_kind(Kind::Float),   // triggered -> target
        };
        let _ = z.shallow_clone().g_sub_(&e.matmul(&self.d));
    }
    fn apply_with(&self, z: &Tensor, tau: &Tensor, lo: &Tensor, target: Option<Tensor>, key: (usize, usize)) {
        if z.size()[0] == 0 { return; }
        assert_eq!(z.kind(), Kind::Float);
        if fuse_mode() >= 1 { return self.clamp_rows(z, tau, lo, target.as_ref(), (key.1 < 2).then_some(key)); }
        let c = z.matmul(&self.d.tr());                                   // [rows, k]
        if let Some(dir) = stats_dir() {
            if crate::tp::world().rank == 0 && !crate::tp::graph::capturing() && std::path::Path::new(&format!("{dir}/ON")).exists() {
                use std::io::Write;
                if key == (0, 0) { let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/calls.txt")).unwrap(); writeln!(f, "{}", z.size()[0]).unwrap(); }
                let v: Vec<f32> = Vec::try_from(c.to_device(Device::Cpu).reshape([-1])).unwrap();
                let _ = std::fs::create_dir_all(&dir);
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/L{}-{}.f32", key.0, key.1)).unwrap();
                f.write_all(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            }
        }
        self.clamp_aten(z, &c, tau, lo, target.as_ref());
    }
    /// In place on a sublayer input z [rows, hidden] fp32 whose MHC norm weight is `ln` (no-op for unregistered weights).
    pub fn apply_input(&self, ln: &Tensor, z: &Tensor) {
        let Some((layer, site)) = self.site(ln) else { return };
        self.apply_with(z, &self.tau.get(layer as i64).get(site as i64), &self.tau_lo.get(layer as i64).get(site as i64),
                        self.target.as_ref().map(|t| t.get(layer as i64).get(site as i64)), (layer, site));
    }
    /// In place on the final hidden z [rows, hidden] fp32 (before lm_head).
    pub fn apply_final(&self, z: &Tensor) { self.apply_with(z, &self.tau_final, &self.tau_final_lo, self.target_final.as_ref().map(|t| t.shallow_clone()), (self.tau.size()[0] as usize, 2)); }
}
/// `clamp-test` (single GPU, SPARK_ABLATE_CLAMP=<file>): the row kernel and the fused MHC pre / post+pre finish epilogue
/// (reg_z 0/1, Sinkhorn split 0/1) against the ATen reference, on synthetic rows whose components straddle one site's
/// thresholds. Prints the worst |diff| / max|z| per case and how many components triggered; panics above 1e-5.
pub fn clamp_test() {
    let c = clamp().expect("clamp-test needs SPARK_ABLATE_CLAMP");
    let dev = Device::Cuda(0);
    let _g = tch::no_grad_guard();
    tch::manual_seed(0);
    let (h, k) = (c.d.size()[1], c.d.size()[0]);
    let (l, s) = (c.tau.size()[0] / 2, 1i64);
    let (tau, lo) = (c.tau.get(l).get(s), c.tau_lo.get(l).get(s));
    let target = c.target.as_ref().map(|t| t.get(l).get(s));
    let mut worst = 0f64;
    let mut check = |name: &str, rows: i64, z: &Tensor, got: &Tensor, want: &Tensor| {
        let cz = z.matmul(&c.d.tr());
        let trig = i64::try_from(cz.gt_tensor(&tau).logical_or(&cz.lt_tensor(&lo)).sum(Kind::Int64)).unwrap();
        let d = f64::try_from((got - want).abs().max()).unwrap() / f64::try_from(z.abs().max()).unwrap();
        worst = worst.max(d);
        eprintln!("[clamp-test] {name:14} rows {rows:5}: rel diff {d:.2e}, triggered {trig}/{}", rows * k);
    };
    // Components spread over [lo - 2w, tau + 2w] (w = tau - lo) on top of a random orthogonal part.
    let spread = |rows: i64, scale: f64| -> Tensor {
        let base = Tensor::randn([rows, h], (Kind::Float, dev)) * scale;
        let w = (&tau - &lo).unsqueeze(0);
        let coef = &lo.unsqueeze(0) + (Tensor::rand([rows, k], (Kind::Float, dev)) * 5.0 - 2.0) * &w;
        &base - base.matmul(&c.d.tr()).matmul(&c.d) + coef.matmul(&c.d)
    };
    for &rows in &[1i64, 3, 8, 16, 64, 4096] {
        let z = spread(rows, 1.0);
        let want = z.copy(); c.clamp_aten(&want, &want.matmul(&c.d.tr()), &tau, &lo, target.as_ref());
        let got = z.copy(); c.clamp_rows(&got, &tau, &lo, target.as_ref(), None);
        check("row kernel", rows, &z, &got, &want);
    }
    // MHC: register a synthetic ln as site (l, s); residual streams carry the spread rows so z lands near the thresholds.
    extern "C" {
        fn rs_mhc_pre_fused(residual: *const f32, fn_: *const std::ffi::c_void, fn_bf16: i32, scale: *const f32, base: *const f32, ln: *const f32, partial: *mut f32,
            z: *mut f32, post: *mut f32, comb: *mut f32, rows: i32, h: i32, rsign: *const std::ffi::c_void, rbk: i32, zr: *mut std::ffi::c_void, clamp: *const ClampArgs) -> i32;
        fn rs_mhc_post_pre_fused(x: *const f32, residual: *const f32, comb: *const f32, post: *const f32, out: *mut f32, fn_: *const std::ffi::c_void, fn_bf16: i32,
            scale: *const f32, base: *const f32, ln: *const f32, partial: *mut f32, z: *mut f32, post_next: *mut f32, comb_next: *mut f32, rows: i32, h: i32,
            rsign: *const std::ffi::c_void, rbk: i32, zr: *mut std::ffi::c_void, clamp: *const ClampArgs) -> i32;
    }
    let ln = Tensor::ones([h], (Kind::Float, dev));
    c.register(&ln, l as usize, s as usize);
    let fn_t = Tensor::randn([24, 4 * h], (Kind::Float, dev)) * 0.01;
    let (scale, base) = (Tensor::ones([3], (Kind::Float, dev)), Tensor::randn([24], (Kind::Float, dev)) * 0.1);
    let p = |x: &Tensor| x.data_ptr() as *const f32;
    let q = |x: &Tensor| x.data_ptr() as *mut f32;
    let null = std::ptr::null_mut();
    for &(reg, split) in &[("0", "0"), ("1", "0"), ("1", "1"), ("0", "1")] {
        unsafe { std::env::set_var("GLM53_MHC_FINISH_REG", reg); std::env::set_var("GLM53_MHC_SINKHORN_SPLIT", split); }
        for &rows in &[1i64, 4, 8, 16, 512] {
            // ln = 1: z = RMS-normed collapse of the streams, ~ the spread row itself (its rms is ~1).
            let zr = spread(rows, 1.0);
            let residual = (zr.unsqueeze(1).repeat([1, 4, 1]) + Tensor::randn([rows, 4, h], (Kind::Float, dev)) * 0.01).contiguous();
            let x = Tensor::randn([rows, h], (Kind::Float, dev)) * 0.01;
            let comb_in = Tensor::rand([rows, 4, 4], (Kind::Float, dev)) * 0.5;
            let post_in = Tensor::rand([rows, 4], (Kind::Float, dev));
            let a = c.args(&tau, &lo, target.as_ref(), None, rows);
            for post_pre in [false, true] {
                let run = |cl: *const ClampArgs| -> Tensor {
                    let partial = Tensor::empty([(4 * h / 128) * rows * 25], (Kind::Float, dev));
                    let z = Tensor::empty([rows, h], (Kind::Float, dev));
                    let (pm, cb) = (Tensor::empty([rows, 4], (Kind::Float, dev)), Tensor::empty([rows, 4, 4], (Kind::Float, dev)));
                    let rc = if post_pre {
                        let out = Tensor::empty([rows, 4, h], (Kind::Float, dev));
                        unsafe { rs_mhc_post_pre_fused(p(&x), p(&residual), p(&comb_in), p(&post_in), q(&out), fn_t.data_ptr(), 0, p(&scale), p(&base), p(&ln), q(&partial), q(&z), q(&pm), q(&cb), rows as i32, h as i32, null, 0, null, cl) }
                    } else {
                        unsafe { rs_mhc_pre_fused(p(&residual), fn_t.data_ptr(), 0, p(&scale), p(&base), p(&ln), q(&partial), q(&z), q(&pm), q(&cb), rows as i32, h as i32, null, 0, null, cl) }
                    };
                    assert_eq!(rc, 0, "clamp-test mhc launch");
                    z
                };
                let z = run(std::ptr::null());
                let want = z.copy(); c.clamp_aten(&want, &want.matmul(&c.d.tr()), &tau, &lo, target.as_ref());
                let got = run(&a);
                check(&format!("{} r{reg}s{split}", if post_pre { "post+pre" } else { "pre" }), rows, &z, &got, &want);
            }
        }
    }
    // Count gate (SPARK_ABLATE_CLAMP_MIN > 1): sites (0,0) [reset] -> (0,1) -> (1,0) on independent rows; the reference
    // edits site i's row iff it triggered and the running count of triggered sites reaches min. Site (0,1) runs both as
    // the row kernel and inside the fused MHC pre finish.
    if c.min > 1 {
        let trig = |z: &Tensor, tau: &Tensor, lo: &Tensor| { let cz = z.matmul(&c.d.tr()); cz.gt_tensor(tau).logical_or(&cz.lt_tensor(lo)).any_dim(1, false).to_kind(Kind::Int) };
        for &rows in &[1i64, 8, 512] {
            for mhc_mid in [false, true] {
                let mut count = Tensor::zeros([rows], (Kind::Int, dev));
                let mut edited = 0i64;
                for (i, key) in [(0usize, 0usize), (0, 1), (1, 0)].into_iter().enumerate() {
                    let (tl, ll) = (c.tau.get(key.0 as i64).get(key.1 as i64), c.tau_lo.get(key.0 as i64).get(key.1 as i64));
                    let tg = c.target.as_ref().map(|t| t.get(key.0 as i64).get(key.1 as i64));
                    let z = spread(rows, 1.0);
                    let tr = trig(&z, &tl, &ll);
                    count = &count + &tr;
                    let mask = tr.gt(0).logical_and(&count.ge(c.min as i64)).unsqueeze(1);
                    edited += i64::try_from(mask.sum(Kind::Int64)).unwrap();
                    let full = z.copy(); c.clamp_aten(&full, &full.matmul(&c.d.tr()), &tl, &ll, tg.as_ref());
                    let want = full.where_self(&mask, &z);
                    let got = if mhc_mid && i == 1 {
                        // ln = 1 and four equal streams: the finish's z is the RMS-normed row, so feed the normed row as the reference.
                        let zn = &z / (z.square().mean_dim(&[-1i64][..], true, Kind::Float) + 1e-5).sqrt();
                        let full = zn.copy(); c.clamp_aten(&full, &full.matmul(&c.d.tr()), &tl, &ll, tg.as_ref());
                        let trn = trig(&zn, &tl, &ll); count = &count - &tr + &trn;
                        let mask = trn.gt(0).logical_and(&count.ge(c.min as i64)).unsqueeze(1);
                        let want = full.where_self(&mask, &zn);
                        let residual = zn.unsqueeze(1).repeat([1, 4, 1]).contiguous();
                        let a = c.args(&tl, &ll, tg.as_ref(), Some(key), rows);
                        let partial = Tensor::empty([(4 * h / 128) * rows * 25], (Kind::Float, dev));
                        let zo = Tensor::empty([rows, h], (Kind::Float, dev));
                        let (pm, cb) = (Tensor::empty([rows, 4], (Kind::Float, dev)), Tensor::empty([rows, 4, 4], (Kind::Float, dev)));
                        assert_eq!(unsafe { rs_mhc_pre_fused(p(&residual), fn_t.data_ptr(), 0, p(&scale), p(&base), p(&ln), q(&partial), q(&zo), q(&pm), q(&cb), rows as i32, h as i32, null, 0, null, &a) }, 0);
                        // pre[] weights scale the collapse; RMS normalisation removes the scale, so zo ~ zn up to rounding.
                        check(&format!("gate mhc s{i}"), rows, &zn, &zo, &want);
                        continue;
                    } else { let g = z.copy(); c.clamp_rows(&g, &tl, &ll, tg.as_ref(), Some(key)); g };
                    check(&format!("gate rows s{i}"), rows, &z, &got, &want);
                }
                let dev_count = c.count.narrow(0, 0, rows).copy();
                assert_eq!(i64::try_from((&dev_count - &count).abs().sum(Kind::Int64)).unwrap(), 0, "clamp-test: device count != reference count");
                eprintln!("[clamp-test] gate min {} rows {rows}{}: {edited} row edits, counts equal", c.min, if mhc_mid { " (mhc mid)" } else { "" });
            }
        }
    }
    eprintln!("[clamp-test] worst rel diff {worst:.2e} ({})", if worst < 1e-5 { "PASS" } else { "FAIL" });
    assert!(worst < 1e-5, "clamp-test");
}
/// The final hidden with the clamp applied when configured (returns z itself otherwise).
pub fn clamp_final(z: Tensor) -> Tensor {
    match clamp() { Some(c) => { let z = if z.kind() == Kind::Float && z.is_contiguous() { z } else { z.to_kind(Kind::Float).contiguous() }; c.apply_final(&z); z } None => z }
}

/// y [..., hidden] fp32 (unit column stride) in place: y - D^T D y for the rows of d [k, hidden] (one launch).
fn project_rows(y: &Tensor, d: &Tensor) {
    extern "C" { fn spark_ablate_rows_k(y: *mut std::ffi::c_void, ld: i64, r: i32, d: *const std::ffi::c_void, dim: i32, k: i32, st: *mut std::ffi::c_void) -> i32;
                 fn rs_current_stream() -> *mut std::ffi::c_void; }
    assert_eq!(y.kind(), Kind::Float);
    let s = y.size();
    let (r, dim) = (s[..s.len() - 1].iter().product::<i64>(), s[s.len() - 1]);
    assert_eq!(y.stride()[y.dim() - 1], 1);
    if r == 0 { return; }
    assert_eq!(unsafe { spark_ablate_rows_k(y.data_ptr(), y.stride()[y.dim() - 2], r as i32, d.data_ptr(), dim as i32, d.size()[0] as i32, rs_current_stream()) }, 0, "ablate rows");
}

/// SPARK_ABLATE_OPROJ=<file.safetensors> (independent of SPARK_ABLATE): tensor `directions` [layers, hidden], row L =
/// sqrt(a_L) d_L (zero row: layer L unedited). Only the attention o_proj of layer L is edited, W -> W - a_L d_L d_L^T W
/// (at load for dense weights, on the outputs at run time for SPX ones): a per-layer rank-1 edit of the form the dealign
/// o_proj transplant uses (recovered offline from the donor's weights).
pub struct OprojEdit { d: Tensor, rows: Vec<bool> }
unsafe impl Sync for OprojEdit {}
pub fn oproj() -> Option<&'static OprojEdit> {
    static O: OnceLock<Option<OprojEdit>> = OnceLock::new();
    O.get_or_init(|| {
        let path = std::env::var("SPARK_ABLATE_OPROJ").ok().filter(|p| !p.is_empty())?;
        let t = Tensor::read_safetensors(&path).unwrap_or_else(|e| panic!("SPARK_ABLATE_OPROJ {path}: {e}"));
        let d = t.iter().find(|(k, _)| k == "directions").unwrap_or_else(|| panic!("SPARK_ABLATE_OPROJ {path}: no `directions`")).1
            .to_kind(Kind::Float).to_device(Device::Cuda(0)).contiguous();
        let norms: Vec<f32> = Vec::try_from(d.norm_scalaropt_dim(2, [1i64].as_slice(), false).to_device(Device::Cpu)).unwrap();
        let rows: Vec<bool> = norms.iter().map(|&n| n > 0.0).collect();
        eprintln!("[ablate] {path}: o_proj edits in {} of {} layers (scale^2 per layer {:?})", rows.iter().filter(|&&b| b).count(), rows.len(),
                  norms.iter().map(|n| (n * n * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        Some(OprojEdit { d, rows })
    }).as_ref()
}
impl OprojEdit {
    pub fn covers(&self, layer: usize) -> bool { self.rows.get(layer).copied().unwrap_or(false) }
    fn row(&self, layer: usize) -> Tensor { self.d.narrow(0, layer as i64, 1) }
    /// In place on o_proj outputs y [..., hidden] fp32.
    pub fn apply(&self, layer: usize, y: &Tensor) { if self.covers(layer) { project_rows(y, &self.row(layer)); } }
    /// The edited dense o_proj weight W [hidden, in] (same kind).
    pub fn ortho_out(&self, layer: usize, w: &Tensor) -> Tensor {
        if !self.covers(layer) { return w.shallow_clone(); }
        let wf = w.to_kind(Kind::Float); let d = self.row(layer).to_device(w.device());
        (&wf - d.tr().matmul(&d.matmul(&wf))).to_kind(w.kind())
    }
}

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
