//! GLM routed experts on the engine's EXL3 MoE kernels (shim/moe_exl3.cuh, config GlmMoe), and `glm-moe-ab`: one layer
//! of real weights, synthetic inputs and routes, timing (CUDA graphs) and the error against an FP32 reference.
use std::path::Path;
use tch::{Device, Kind, Tensor};

extern "C" {
    fn glm_moe_ws_bytes(r: i32, s: i32) -> usize;
    fn glm_moe_experts(x: *const std::ffi::c_void, ldx: i64, r: i32, idx: *const std::ffi::c_void, w: *const std::ffi::c_void,
                       tab: *const std::ffi::c_void, add: *const std::ffi::c_void, out: *mut std::ffi::c_void, ws: *mut std::ffi::c_void,
                       s: i32, suh0: *const std::ffi::c_void, kmask: i32, st: *mut std::ffi::c_void) -> i32;
    fn glm_moe3_ws_bytes(r: i32, s: i32) -> usize;
    fn glm_moe3_experts(x: *const std::ffi::c_void, ldx: i64, r: i32, idx: *const std::ffi::c_void, w: *const std::ffi::c_void,
                        tab: *const std::ffi::c_void, add: *const std::ffi::c_void, out: *mut std::ffi::c_void, ws: *mut std::ffi::c_void,
                        s: i32, kmask: i32, st: *mut std::ffi::c_void) -> i32;
    fn rs_current_stream() -> *mut std::ffi::c_void;
    fn rs_exl3_inner_k(tr: *const std::ffi::c_void, inner: *mut std::ffi::c_void, k: i64, n: i64, bits: i64) -> i32;
}

/// K slices of the decode GEMVs (GLM53_MOE_SLICES, default 4).
fn slices() -> i32 { std::env::var("GLM53_MOE_SLICES").ok().and_then(|v| v.parse().ok()).unwrap_or(4) }

/// Rows per launch on the prefill path (the prefill chunk; per-pair scratch rows * 8 * (4096 + 1024) * 2 bytes).
pub(crate) const PREFILL_ROWS: i64 = 4096;

/// out [R, 4096] fp32 = sum_k w[r][k] * expert_k(x[r]) (+ add). x [R, 4096] fp16, ids [R, 8] int64, weights [R, 8] (any
/// float kind; rounded to fp16 like the cooperative path); tab [10, 288] i64: 9 rows of device pointers (moe_exl3.cuh order)
/// and row 9 the experts' trellis widths K; kmask: bit K set for every K present (1 << 4: the uniform 4-bit kernels).
/// `suh0`: the layer's shared gate/up input scales when the pool proved every expert uses the same ones (prefill then
/// transforms each token once instead of once per routed pair).
pub(crate) fn run_into(tab: &Tensor, kmask: i32, x: &Tensor, ids: &Tensor, weights: &Tensor, out: &Tensor, add: Option<&Tensor>, suh0: Option<&Tensor>) {
    let r = x.size()[0];
    assert_eq!(x.kind(), Kind::Half);
    assert!(x.is_contiguous() && ids.is_contiguous() && out.is_contiguous());
    assert_eq!(ids.size(), [r, 8]);
    assert_eq!(ids.kind(), Kind::Int64);
    assert_eq!(out.size(), [r, 4096]);
    assert_eq!(out.kind(), Kind::Float);
    let w = weights.to_kind(Kind::Half).contiguous();
    let s = slices();
    let add_p = add.map_or(std::ptr::null(), |a| { assert!(a.is_contiguous() && a.kind() == Kind::Float && a.size() == [r, 4096]); a.data_ptr() as *const _ });
    // a table with the experts' widths in row 10 (TP3, crate::moefast::expert_part): the variable-width kernels (GlmMoe3)
    if tab.size()[0] == 11 {
        assert!(suh0.is_none(), "variable-width experts take no shared input scales");
        let ws = Tensor::empty([unsafe { glm_moe3_ws_bytes(r as i32, s) } as i64], (Kind::Uint8, x.device()));
        assert_eq!(unsafe { glm_moe3_experts(x.data_ptr(), x.stride()[0], r as i32, ids.data_ptr(), w.data_ptr(), tab.data_ptr(), add_p,
                                             out.data_ptr(), ws.data_ptr(), s, kmask, rs_current_stream()) }, 0, "GLM MoE experts (TP3 widths)");
        return;
    }
    let ws = Tensor::empty([unsafe { glm_moe_ws_bytes(r as i32, s) } as i64], (Kind::Uint8, x.device()));
    assert_eq!(unsafe { glm_moe_experts(x.data_ptr(), x.stride()[0], r as i32, ids.data_ptr(), w.data_ptr(), tab.data_ptr(), add_p,
                                        out.data_ptr(), ws.data_ptr(), s, suh0.map_or(std::ptr::null(), |t| t.data_ptr() as *const _),
                                        kmask, rs_current_stream()) }, 0, "GLM MoE experts");
}

/// The 9 per-projection pointer tables of one layer (MoeFast order: gate tr, suh, svh, up tr, suh, svh, down tr, suh,
/// svh; [288] each) in moe_exl3.cuh order: gate/up/down trellis, gate/up/down suh, gate/up/down svh; row 9 the experts'
/// trellis widths `kbits`. Returns (table [10, 288], kmask).
pub(crate) fn table(per_proj: &[Tensor], kbits: &[i64]) -> (Tensor, i32) {
    assert_eq!(per_proj.len(), 9);
    let order = [0usize, 3, 6, 1, 4, 7, 2, 5, 8];
    let mut rows: Vec<Tensor> = order.iter().map(|&i| per_proj[i].shallow_clone()).collect();
    rows.push(Tensor::from_slice(kbits).to_device(per_proj[0].device()));
    (Tensor::stack(&rows, 0).contiguous(), kmask(kbits))
}
/// Bit K for every expert trellis width K present (each K in 3..=6).
pub(crate) fn kmask(kbits: &[i64]) -> i32 {
    let m = kbits.iter().fold(0, |m, &k| { assert!((3..=6).contains(&k), "expert trellis K {k} not supported (3..6)"); m | 1 << k });
    // GLM53_MOE_FORCE_MIXED=1 (diagnostic): uniform layers take the mixed-K kernels too (dispatch overhead alone)
    if std::env::var("GLM53_MOE_FORCE_MIXED").as_deref() == Ok("1") { m | 1 << 3 } else { m }
}
/// Trellis width K of a packed EXL3 trellis tensor [k/16, n/16, 16 K].
pub(crate) fn trellis_bits(tr: &Tensor) -> i64 { tr.size()[2] / 16 }

// ---- glm-moe-ab ----

fn hadamard128(dev: Device) -> Tensor {
    let mut h = Tensor::ones([1, 1], (Kind::Float, dev));
    while h.size()[0] < 128 { h = Tensor::cat(&[Tensor::cat(&[&h, &h], 1), Tensor::cat(&[&h, &(-&h)], 1)], 0); }
    h / (128f64).sqrt()
}
/// y = had128(had128(x * suh) @ inner) * svh in FP32 (inner = the decoded EXL3 weights, exact in fp16).
fn ref_linear(x: &Tensor, tr: &Tensor, suh: &Tensor, svh: &Tensor, k: i64, n: i64, h: &Tensor) -> Tensor {
    let inner = Tensor::empty([k, n], (Kind::Half, x.device()));
    assert_eq!(unsafe { rs_exl3_inner_k(tr.data_ptr(), inner.data_ptr(), k, n, trellis_bits(tr)) }, 0, "reconstruct");
    let r = x.size()[0];
    let xh = (x * suh.to_kind(Kind::Float)).view([r, k / 128, 128]).matmul(h).view([r, k]);
    let y = xh.matmul(&inner.to_kind(Kind::Float));
    y.view([r, n / 128, 128]).matmul(h).view([r, n]) * svh.to_kind(Kind::Float)
}

/// `glm-moe-ab <model>`: GLM53_MOE_AB_LAYER (default 3), rank GLM53_MOE_AB_RANK (default 0) of TP GLM53_MOE_AB_WORLD (2; 3: the
/// TP3 6/5/5 split on the variable-width kernels, or the holders with GLM53_TP3_EXPERTS=holders).
pub fn ab(model: &Path) {
    use std::time::Instant;
    tch::set_num_threads(4);
    let _guard = tch::no_grad_guard();
    let dev = Device::Cuda(0);
    let rank = std::env::var("GLM53_MOE_AB_RANK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let layer = std::env::var("GLM53_MOE_AB_LAYER").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(3);
    let world = std::env::var("GLM53_MOE_AB_WORLD").ok().and_then(|v| v.parse().ok()).unwrap_or(2usize);
    let tp = crate::tp::Tp { rank, world };
    let mut pool = crate::moefast::MoeFast::new_with_tp(model, 45, 288, 288, dev, tp);
    pool.ensure_many(layer, &(0..288).collect::<Vec<_>>(), dev);
    pool.assume_hot = true;
    let siv = pool.shared_input_verified(layer);
    let per_proj: Vec<Tensor> = pool.probe_tables().iter().map(Tensor::shallow_clone).collect();
    let slots = Tensor::from_slice(&(0..288).map(|e| pool.slot_of(layer, e)).collect::<Vec<_>>()).to_device(dev);
    let layer_tabs: Vec<Tensor> = per_proj.iter().map(|t| t.index_select(0, &slots)).collect();
    let h = hadamard128(dev);
    let experts = pool.layer_tensors(layer);
    let kbits: Vec<i64> = experts.iter().map(|t| { let k = trellis_bits(&t[0]); assert!(trellis_bits(&t[3]) == k && trellis_bits(&t[6]) == k); k }).collect();
    // the kernel's own table (dummies marked, TP3 widths in row 10)
    let (tab, kmask) = pool.own_table(layer);
    let widths: Vec<i64> = experts.iter().map(|t| t[0].size()[1] * 16).collect();
    let held: Vec<bool> = (0..288).map(|e| crate::moefast::expert_part(layer, e, tp).is_some()).collect();
    println!("table {:?}, widths {:?}", tab.size(), { let mut w = widths.clone(); w.sort(); w.dedup(); w });
    let mut kcount = [0usize; 8];
    for &k in &kbits { kcount[k as usize] += 1; }
    println!("expert K: 3:{} 4:{} 5:{} 6:{} (kmask {kmask:#x})", kcount[3], kcount[4], kcount[5], kcount[6]);
    // skewed routes: expert e drawn with probability ~ 1 / (e + 12) (a fixed popularity order), 8 distinct per row
    let pick = |rows: i64, skew: bool, seed: i64| -> Tensor {
        tch::manual_seed(seed);
        let p = if skew { Tensor::arange(288, (Kind::Float, dev)).g_add_scalar(12.0).reciprocal() } else { Tensor::ones([288], (Kind::Float, dev)) };
        p.unsqueeze(0).expand([rows, 288], false).multinomial(8, false).contiguous()
    };
    println!("layer {layer} rank {rank} shared-input {siv}");
    // GLM53_MOE_AB_PREFILL=rows: prefill path, own grouped GEMMs vs the reference grouped (fat) path, eager timing
    if let Some(rows) = std::env::var("GLM53_MOE_AB_PREFILL").ok().and_then(|v| v.parse::<i64>().ok()) {
        tch::manual_seed(5);
        let x = (Tensor::randn([rows, 4096], (Kind::Float, dev))).to_kind(Kind::Half);
        let ids = Tensor::ones([rows, 288], (Kind::Float, dev)).multinomial(8, false).contiguous();
        let w = (Tensor::rand([rows, 8], (Kind::Float, dev)) + 0.1).to_kind(Kind::Float);
        let mut run = || -> (f64, Tensor) {
            let mut y = pool.expert_grouped(layer, &x, &ids, &w, false);
            tch::Cuda::synchronize(0);
            let mut best = f64::MAX;
            for _ in 0..5 { let t = Instant::now(); y = pool.expert_grouped(layer, &x, &ids, &w, false); tch::Cuda::synchronize(0); best = best.min(t.elapsed().as_secs_f64() * 1e3); }
            (best, y)
        };
        let (t_own, y_own) = run();
        std::env::set_var("GLM53_MOE_NO_SHARED", "1");
        let (t_ns, y_ns) = run();
        std::env::remove_var("GLM53_MOE_NO_SHARED");
        println!("prefill rows {rows}: {t_own:.2} ms (per-pair input transform {t_ns:.2} ms, {})", if y_ns.equal(&y_own) { "bitwise equal" } else { "DIFFERS" });
        return;
    }
    let list = |v: &str, d: &str| -> Vec<i64> { std::env::var(v).unwrap_or_else(|_| d.into()).split(',').map(|t| t.parse().unwrap()).collect() };
    for skew in list("GLM53_MOE_AB_SKEW", "0,1").into_iter().map(|v| v != 0) {
        for rows in list("GLM53_MOE_AB_ROWS", "1,2,3,4,6,8,16,32") {
            let ids = pick(rows, skew, rows * 7 + skew as i64);
            tch::manual_seed(rows + 100);
            let x = Tensor::randn([rows, 4096], (Kind::Float, dev)).to_kind(Kind::Half);
            let w = (Tensor::rand([rows, 8], (Kind::Float, dev)) + 0.1).to_kind(Kind::Half);
            // FP32 reference
            let xf = x.to_kind(Kind::Float);
            let wf = w.to_kind(Kind::Float);
            let ids_h: Vec<i64> = Vec::try_from(ids.view([-1]).to_device(Device::Cpu)).unwrap();
            let mut refo = Tensor::zeros([rows, 4096], (Kind::Float, dev));
            for r in 0..rows {
                for k in 0..8 {
                    let e = ids_h[(r * 8 + k) as usize] as usize;
                    if !held[e] { continue; }   // another rank's expert (TP3 holders): no contribution here
                    let (t, we) = (&experts[e], widths[e]);
                    let xr = xf.narrow(0, r, 1);
                    let g = ref_linear(&xr, &t[0], &t[1], &t[2], 4096, we, &h).clamp_max(10.0);
                    let u = ref_linear(&xr, &t[3], &t[4], &t[5], 4096, we, &h).clamp(-10.0, 10.0);
                    let a = g.silu() * u;
                    let d = ref_linear(&a, &t[6], &t[7], &t[8], we, 4096, &h);
                    refo = refo.index_add(0, &Tensor::from_slice(&[r]).to_device(dev), &(d * wf.narrow(0, r, 1).narrow(1, k, 1)));
                }
            }
            let rel = |y: &Tensor| ((y - &refo).norm().double_value(&[]) / refo.norm().double_value(&[]));
            let own = Tensor::empty([rows, 4096], (Kind::Float, dev));
            run_into(&tab, kmask, &x, &ids, &w, &own, None, None);
            let e_own = rel(&own);
            let uniq = ids.view([-1]).unique_dim(0, false, false, false).0.size()[0];
            // timing: 8 calls (different routes) per graph, replayed; same process, both kernels
            let routes: Vec<Tensor> = (0..8).map(|i| pick(rows, skew, 1000 + rows * 31 + i)).collect();
            let time = |f: &dyn Fn(&Tensor)| -> f64 {
                for r in &routes { f(r); }
                tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();
                for r in &routes { f(r); }
                crate::tp::graph::end().unwrap();
                for _ in 0..3 { crate::tp::graph::replay().unwrap(); }
                tch::Cuda::synchronize(0);
                let mut best = f64::MAX;
                for _ in 0..7 {
                    let t = Instant::now();
                    for _ in 0..10 { crate::tp::graph::replay().unwrap(); }
                    tch::Cuda::synchronize(0);
                    best = best.min(t.elapsed().as_secs_f64() * 1e6 / 80.0);
                }
                crate::tp::graph::destroy();
                best
            };
            // GLM53_MOE_AB_CFGS="cfg;cfg;..": GLM53_MOE_GEMV schedules timed (and checked) in turn
            if let Ok(cfgs) = std::env::var("GLM53_MOE_AB_CFGS") {
                for c in cfgs.split(';') {
                    std::env::set_var("GLM53_MOE_GEMV", c);
                    let o = Tensor::empty([rows, 4096], (Kind::Float, dev));
                    run_into(&tab, kmask, &x, &ids, &w, &o, None, None);
                    let same = o.equal(&own);
                    let t = time(&|r: &Tensor| { let o = Tensor::empty([rows, 4096], (Kind::Float, dev)); run_into(&tab, kmask, &x, r, &w, &o, None, None); });
                    println!("  cfg {c:>18}: {t:7.1} us {}", if same { "bitwise" } else { "DIFFERS" });
                }
                std::env::remove_var("GLM53_MOE_GEMV");
            }
            let t_own = time(&|r: &Tensor| { let o = Tensor::empty([rows, 4096], (Kind::Float, dev)); run_into(&tab, kmask, &x, r, &w, &o, None, None); });
            println!("{} rows {:2} uniq {:3}: {:7.1} us, error vs FP32 {:.5}", if skew { "skew" } else { "unif" }, rows, uniq, t_own, e_own);
        }
    }
}
