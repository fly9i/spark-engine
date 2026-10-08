//! CUDA-graph decoder for one sequence store: the speculative round runs as graph replays only.
//!   verify  : target forward of [next, d1..dk] (k + 1 rows, state not written)
//!   commit n: replay of the first n verify rows into the GDN states / conv windows / PLE window (n = 1..k+1)
//!   mtp n   : MTP forward of n rows (n = 1 for draft steps, 1..k+1 for the catch-up)
//!   batch   : 1..8 sequences together (k + 1 rows each): verify, one commit graph whose per-sequence accepted
//!             row counts are read from a device buffer, and the MTP chain (catch-up on the verify stacks, then
//!             k - 1 draft steps with on-device argmax) that yields the next round's drafts
//! Inputs (ids, position, packed PLE rings, MTP hidden stacks) are copied into fixed buffers before a replay;
//! outputs stay in the graphs' pools and are read before the next replay of the same graph.
use super::model::{Model, Pending, Pick, Seq, H, D};
use crate::tp::graph;
use tch::{Device, Kind, Tensor};

/// Copy into a fixed (graph input) buffer: tch's copy_ wants &mut, the buffer handle is shared.
fn put(dst: &Tensor, src: &Tensor) { let mut d = dst.shallow_clone(); d.copy_(src); }

fn dev() -> Device { Device::Cuda(0) }

pub struct Verify { g: graph::Owned, ids: Tensor, pos: Tensor, packed: Option<Tensor>, pub logits: Tensor, pub x: Tensor, pend: Pending }
pub struct MtpG { g: graph::Owned, hidden: Tensor, ids: Tensor, pos: Tensor, pub logits: Tensor, pub x: Tensor }

pub struct Decoder { pub k: usize, verify: Verify, commits: Vec<graph::Owned>, mtp: Vec<MtpG> }

/// Capture into the memory pool shared by all batch graphs (they replay one after another; every tensor read across
/// graphs is kept alive by its owner), with one cuBLAS workspace for all graphs.
fn capture_shared<T>(f: impl FnOnce() -> T) -> (graph::Owned, T) {
    extern "C" { fn rs_graph_begin_pool(key: i64) -> i32; }
    fn cap<T>(f: impl FnOnce() -> T) -> (graph::Owned, T) {
        tch::Cuda::synchronize(0);
        assert_eq!(unsafe { rs_graph_begin_pool(1) }, 0, "graph begin");
        let out = f();
        graph::end().expect("graph end");
        (graph::Owned::take(), out)
    }
    std::env::set_var("GLM53_GRAPH_SHARED_WS", "1");
    // a pool whose graphs are all gone cannot be captured into again: one small graph keeps it for the process
    static KEEPER: std::sync::Once = std::sync::Once::new();
    KEEPER.call_once(|| {
        let t = Tensor::zeros([1], (Kind::Float, dev()));
        let (g, ()) = cap(|| { let _ = t.shallow_clone().fill_(1.0); });
        std::mem::forget(g);
        std::mem::forget(t);
    });
    cap(f)
}

/// (allocated, reserved) bytes of the CUDA caching allocator
pub fn cuda_mem() -> (i64, i64) {
    extern "C" { fn rs_cuda_memory(a: *mut i64, r: *mut i64); }
    let (mut a, mut r) = (0, 0);
    unsafe { rs_cuda_memory(&mut a, &mut r) };
    (a, r)
}

fn capture<T>(f: impl FnOnce() -> T) -> (graph::Owned, T) {
    tch::Cuda::synchronize(0);
    graph::begin().expect("graph begin");
    let out = f();
    graph::end().expect("graph end");
    (graph::Owned::take(), out)
}

impl Decoder {
    /// Capture all graphs for `seq` (its tensors are baked in: the store must keep them, `Model::reset` zeroes in place).
    pub fn capture(m: &Model, seq: &mut Seq, k: usize) -> Self {
        let r = k as i64 + 1;
        let ids = Tensor::zeros([r], (Kind::Int64, dev()));
        let pos = Tensor::zeros([1], (Kind::Int, dev()));
        let packed = m.ple_row_bytes().map(|b| Tensor::zeros([r * b], (Kind::Uint8, dev())));
        assert!(seq.pending.is_none());
        let (g, (logits, x)) = capture(|| m.run_dev(seq, &ids, &pos, packed.as_ref(), r, false, false));
        let pend = seq.pending.take().expect("verify capture leaves the pending inputs");
        let verify = Verify { g, ids, pos, packed, logits, x, pend };
        let mut commits = Vec::new();
        for n in 1..=r {
            let (g, ()) = capture(|| m.commit_with(seq, &verify.pend, n));
            commits.push(g);
        }
        let mut mtp = Vec::new();
        if m.mtp.is_some() {
            for n in 1..=r {
                let hidden = Tensor::zeros([n, H, D], (Kind::Float, dev()));
                let ids = Tensor::zeros([n], (Kind::Int64, dev()));
                let pos = Tensor::zeros([1], (Kind::Int, dev()));
                let (g, (logits, x)) = capture(|| m.mtp_dev(seq, &hidden, &ids, &pos, true));
                mtp.push(MtpG { g, hidden, ids, pos, logits, x });
            }
        }
        Decoder { k, verify, commits, mtp }
    }

    /// Verify rows [next, drafts..] at the sequence position (state untouched); returns (logits [k+1, V], stacks).
    pub fn verify(&self, m: &Model, seq: &Seq, rows: &[i64]) -> (&Tensor, &Tensor) {
        assert_eq!(rows.len(), self.k + 1);
        let v = &self.verify;
        put(&v.ids, &Tensor::from_slice(rows));
        put(&v.pos, &Tensor::from_slice(&[seq.pos as i32]));
        if let (Some(buf), Some(h)) = (&v.packed, m.ple_pack(seq, rows)) { put(buf, &Tensor::from_slice(&h)); }
        v.g.replay();
        (&v.logits, &v.x)
    }

    /// Accept the first n verify rows (device replay + host bookkeeping).
    pub fn commit(&self, seq: &mut Seq, rows: &[i64], n: usize) {
        self.commits[n - 1].replay();
        seq.ids.extend_from_slice(&rows[..n]);
        seq.pos += n as i64;
    }

    /// MTP forward of hidden [n, 4, 2560] (device) with next tokens `ids` at positions pos0..; returns
    /// (last-row logits [1, V], output stacks [n, 4, 2560]).
    pub fn mtp(&self, hidden: &Tensor, ids: &[i64], pos0: i64) -> (&Tensor, &Tensor) {
        let g = &self.mtp[ids.len() - 1];
        put(&g.hidden, hidden);
        put(&g.ids, &Tensor::from_slice(ids));
        put(&g.pos, &Tensor::from_slice(&[pos0 as i32]));
        g.g.replay();
        (&g.logits, &g.x)
    }
}

/// Draft length policy. QWEN_SPEC_CUMCONF=θ (default 0.7; L2: drafts only, the output is unchanged):
/// the chain keeps drafting while some sequence's product of draft confidences (draft-head softmax max) stays >= θ,
/// up to QWEN_SPEC_KMAX (default 10) drafts, and each sequence verifies its drafts up to the last one inside the product
/// bound. QWEN_SPEC_CUMCONF=0: QWEN_SPEC_K (default 3) drafts every round.
#[derive(Clone, Copy, Debug)]
pub struct Policy { pub k: usize, pub kmax: usize, pub theta: Option<f64>, pub all: bool }
impl Policy {
    pub fn from_env() -> Self {
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
        let k = num("QWEN_SPEC_K").unwrap_or(3.0) as usize;
        let theta = Some(num("QWEN_SPEC_CUMCONF").unwrap_or(0.7)).filter(|&t| t > 0.0);
        let kmax = num("QWEN_SPEC_KMAX").map_or(if theta.is_some() { 10 } else { k }, |v| v as usize).max(1);
        Policy { k: k.max(1), kmax, theta, all: false }
    }
    /// Most drafts a chain produces.
    pub fn cap(&self) -> usize { if self.theta.is_some() { self.kmax } else { self.k } }
    /// Calibrated draft confidence: the draft head's softmax maximum mapped to the measured acceptance rate by the
    /// piecewise-linear table QWEN_SPEC_CALIB ("c:p,c:p,..", ascending c; "fit" = CALIB; unset / "0" = identity, the default:
    /// 2026-10-07 θ 0.45 / 0.55 / 0.65 on calibrated confidences vs 0.7 on raw ones, same-process A/B over 10 prompts and
    /// B = 4: within +-2% either way, no consistent gain).
    pub fn cal(c: f64) -> f64 {
        static T: std::sync::OnceLock<Vec<(f64, f64)>> = std::sync::OnceLock::new();
        let t = T.get_or_init(|| std::env::var("QWEN_SPEC_CALIB").map(|v| if v == "fit" { CALIB.to_string() } else { v }).unwrap_or_default().split(',')
            .filter_map(|kv| { let (a, b) = kv.split_once(':')?; Some((a.trim().parse().ok()?, b.trim().parse().ok()?)) }).collect());
        if t.is_empty() { return c; }
        if c <= t[0].0 { return t[0].1 * c / t[0].0.max(1e-9); }
        for w in t.windows(2) {
            if c <= w[1].0 { return w[0].1 + (w[1].1 - w[0].1) * (c - w[0].0) / (w[1].0 - w[0].0); }
        }
        t[t.len() - 1].1
    }
    /// The policy for b sequences verified together, within DEC_ROWS verify rows in total (the decode kernels take up
    /// to 64 rows per weight pass). Same-process A/B 2026-10-07 (B = 2 / 4, fixed k = 3 -> adaptive with the `all`
    /// rule): prose +1% / -0.5%, mixed prompts -1.3%, structured +36% / +13%; the `max` rule lost 6..25% on prose.
    /// QWEN_SPEC_CUMCONF_BATCH: unset = the single-sequence θ, a number in (0, 1) = its own θ, 0 = fixed k.
    pub fn for_batch(&self, b: usize) -> Policy {
        let mut p = *self;
        if b > 1 {
            match std::env::var("QWEN_SPEC_CUMCONF_BATCH").ok().map(|v| v.parse::<f64>().unwrap_or(0.0)) {
                None => {}
                Some(t) if t > 0.0 && t < 1.0 && p.theta.is_some() => p.theta = Some(t),
                Some(t) if t == 1.0 => {}
                _ => { p.theta = None; p.kmax = p.k; }
            }
            // QWEN_SPEC_BATCH_RULE=max: draft / verify as deep as the most confident sequence; default (all): at least
            // k, deeper only while every sequence is inside the bound
            p.all = std::env::var("QWEN_SPEC_BATCH_RULE").as_deref() != Ok("max");
        }
        if p.theta.is_some() { p.kmax = p.kmax.min((super::model::DEC_ROWS as usize / b).saturating_sub(1)).max(1); }
        p
    }
    /// The first chain of a sequence (eager, right after its prefill): at most k drafts.
    pub fn first(&self) -> Policy { Policy { kmax: self.kmax.min(self.k), ..*self } }
    /// Drafts of one sequence worth verifying (at least 1).
    pub fn verify_len(&self, conf: &[f64]) -> usize {
        let Some(th) = self.theta else { return conf.len().min(self.k).max(1) };
        let (mut d, mut p) = (0, 1.0);
        while d < conf.len() { p *= conf[d]; if p < th { break; } d += 1; }
        d.max(1)
    }
    /// Draft one more step: below the cap and (fixed k, or some sequence still inside the bound on all its drafts;
    /// `all`: at least k, then every sequence inside the bound).
    pub fn more(&self, confs: &[Vec<f64>]) -> bool {
        let n = confs[0].len();
        let inside = |c: &Vec<f64>| c.iter().product::<f64>() >= self.theta.unwrap_or(0.0);
        n < self.cap() && (self.theta.is_none() || if self.all { n < self.k || confs.iter().all(inside) } else { confs.iter().any(inside) })
    }
    /// Verify width (drafts per sequence) of a batch from the sequences' verify lengths. `all`: k, or the cap when
    /// every sequence's verify length reaches halfway from k to the cap (two widths per batch: few graphs to keep).
    pub fn width(&self, lens: &[usize]) -> usize {
        if self.all && self.theta.is_some() {
            let m = *lens.iter().min().unwrap();
            if self.kmax > self.k && 2 * m >= self.k + self.kmax { self.kmax } else { self.k.min(self.kmax) }
        } else { *lens.iter().max().unwrap() }
    }
}

/// Default calibration table (see `Policy::cal`).
/// 2026-10-07, 982 rounds of 10 drafts (20 prompts, every draft verified): acceptance of a draft (given the earlier ones
/// accepted) by its softmax maximum: 0.2-0.3: 0.13, 0.5-0.6: 0.38, 0.7-0.8: 0.48, 0.9-0.95: 0.66, 0.95-0.98: 0.72, >= 0.98: 0.95.
const CALIB: &str = "0.15:0.05,0.25:0.13,0.35:0.24,0.45:0.33,0.55:0.38,0.65:0.42,0.75:0.5,0.85:0.66,0.92:0.69,0.965:0.72,0.99:0.95,1:0.97";

/// Mapped pinned host memory. On GB10 it is the same DRAM as device memory: graph inputs are written by the host and
/// outputs read by it in place (no copy launches; tch's blocking copies synchronized the stream on every input).
struct Mapped { host: *mut u8, dev: *mut u8, bytes: usize }
extern "C" {
    fn cudaHostAlloc(p: *mut *mut std::ffi::c_void, size: usize, flags: u32) -> i32;
    fn cudaHostGetDevicePointer(d: *mut *mut std::ffi::c_void, h: *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaFreeHost(p: *mut std::ffi::c_void) -> i32;
}
impl Mapped {
    fn new(bytes: usize) -> Self {
        let bytes = bytes.max(16);
        let (mut h, mut d) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(unsafe { cudaHostAlloc(&mut h, bytes, 3 /* portable | mapped */) }, 0, "cudaHostAlloc");
        assert_eq!(unsafe { cudaHostGetDevicePointer(&mut d, h, 0) }, 0, "cudaHostGetDevicePointer");
        unsafe { std::ptr::write_bytes(h as *mut u8, 0, bytes) };
        Mapped { host: h.cast(), dev: d.cast(), bytes }
    }
    /// Device tensor over [off, off + numel * size) (row-major).
    fn tensor(&self, off: usize, shape: &[i64], kind: Kind) -> Tensor {
        let n: i64 = shape.iter().product();
        assert!(off + n as usize * kind.elt_size_in_bytes() <= self.bytes);
        unsafe { Tensor::from_blob(self.dev.add(off), shape, &[], kind, dev()) }
    }
    fn put<T: Copy>(&self, off: usize, v: &[T]) {
        assert!(off + std::mem::size_of_val(v) <= self.bytes);
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), self.host.add(off) as *mut T, v.len()) };
    }
    fn get<T: Copy + Default>(&self, off: usize, n: usize) -> Vec<T> {
        assert!(off + n * std::mem::size_of::<T>() <= self.bytes);
        let mut v = vec![T::default(); n];
        unsafe { std::ptr::copy_nonoverlapping(self.host.add(off) as *const T, v.as_mut_ptr(), n) };
        v
    }
}
impl Drop for Mapped { fn drop(&mut self) { unsafe { cudaFreeHost(self.host.cast()) }; } }

/// Offsets of 16-byte aligned fields in a Mapped block.
struct Lay(usize);
impl Lay { fn take(&mut self, bytes: usize) -> usize { let o = self.0; self.0 += (bytes + 15) / 16 * 16; o } }

/// Batched verify / commit / MTP chain for a fixed set of sequences (their tensors are baked in). Per verify width
/// r1 (rows per sequence, captured on first use): verify; accept (greedy on the device: target argmax, accepted counts
/// within kk[g], catch-up tokens and rows); commit; MTP catch-up on the verify stacks, which yields the first next
/// draft. Then one MTP draft-step graph replayed per further draft (its inputs are fed back on the device), so the
/// chain length is decided between replays. Inputs and outputs live in mapped host memory. A greedy round replays
/// verify, accept, commit and catch-up back to back and synchronizes once (plus once per further draft step).
pub struct BatchG { nseg: i64, parts: Vec<Option<Part>>, step: Option<StepG> }
struct Part { r1: i64, io: Mapped, o: PartOff, g: graph::Owned, logits: Tensor, x: Tensor, accept: graph::Owned,
              commit: graph::Owned, catch: Option<graph::Owned>, nd_dev: Tensor, nd_map: Tensor, _keep: (Pending, Vec<Tensor>) }
/// Offsets in Part::io: inputs ids [b*r1] i64, pos [b] i32, kk [b] i32, packed rings; accept outputs / host-path inputs
/// nd [b] i32, cids [b*r1] i64, sel [b] i64; output greedy [b*r1] i64.
struct PartOff { ids: usize, pos: usize, kk: usize, packed: usize, nd: usize, cids: usize, sel: usize, greedy: usize }
/// Draft step: hidden [b, 4, 2560] and tokens [b] in, (token, confidence) appended to out[cnt] (mapped: spos, out).
struct StepG { g: graph::Owned, io: Mapped, kc: usize, hid: Tensor, d: Tensor, spos: Tensor, out: Tensor, cnt: Tensor }

fn pick_pair(m: &Model, lg: &Tensor) -> (Tensor, Tensor) {
    let (d, c) = m.draft_pick(lg);
    let pair = Tensor::stack(&[d.to_kind(Kind::Double), c.to_kind(Kind::Double)], 1);
    (d, pair)
}

/// Wait for the current stream (not the side stream).
fn sync() { extern "C" { fn rs_stream_sync() -> i32; } assert_eq!(unsafe { rs_stream_sync() }, 0, "stream sync"); }

/// QWEN_COMMIT_SIDE=1 (experimental, off: 2026-10-07 the overlapped commit changed verified output in 1 of 6 runs,
/// a race not yet found): the state commit runs on a side stream next to the MTP chain; the next verify joins it.
/// QWEN_COMMIT_SIDE=2: side stream, joined right away (diagnostic).
fn side(f: impl FnOnce()) {
    extern "C" { fn rs_side_begin() -> i32; fn rs_side_end() -> i32; }
    let mode = std::env::var("QWEN_COMMIT_SIDE").unwrap_or_default();
    if mode != "1" && mode != "2" { return f(); }
    assert_eq!(unsafe { rs_side_begin() }, 0, "side stream begin");
    f();
    assert_eq!(unsafe { rs_side_end() }, 0, "side stream end");
    if mode == "2" { side_join(); }
}
/// Make the current stream wait for the side-stream work issued so far.
pub fn side_join() { extern "C" { fn rs_side_join() -> i32; } assert_eq!(unsafe { rs_side_join() }, 0, "side stream join"); }

/// Per sequence: drafts and their confidences.
pub type Drafts = Vec<(Vec<i64>, Vec<f64>)>;

impl BatchG {
    /// `kc`: most drafts per chain.
    pub fn capture(m: &Model, seqs: &[&Seq], kc: usize) -> Self {
        let b = seqs.len() as i64;
        let step = m.mtp.is_some().then(|| {
            let mut l = Lay(0);
            let (o_spos, o_out) = (l.take(4 * b as usize), l.take(16 * kc * b as usize));
            let io = Mapped::new(l.0);
            let spos = io.tensor(o_spos, &[b], Kind::Int);
            let out = io.tensor(o_out, &[kc as i64, b, 2], Kind::Double);
            let hid = Tensor::zeros([b, H, D], (Kind::Float, dev()));
            let d = Tensor::zeros([b], (Kind::Int64, dev()));
            let cnt = Tensor::zeros([1], (Kind::Int64, dev()));
            let ones: Vec<(i64, i64)> = (0..b).map(|g| (g, 1)).collect();
            let (g, ()) = capture_shared(|| {
                let (lg, xs) = m.mtp_batch_dev(seqs, &hid, &d, &spos, &ones, Pick::All);
                let (d2, pair) = pick_pair(m, &lg);
                let _ = out.shallow_clone().index_copy_(0, &cnt, &pair.unsqueeze(0));
                put(&d, &d2);
                put(&hid, &xs);
                let _ = spos.shallow_clone().g_add_scalar_(1);
                let _ = cnt.shallow_clone().g_add_scalar_(1);
            });
            StepG { g, io, kc, hid, d, spos, out, cnt }
        });
        let mut bg = BatchG { nseg: b, parts: (0..=kc + 1).map(|_| None).collect(), step };
        bg.ensure(m, seqs, kc as i64 + 1);
        bg
    }

    /// Capture the graphs of verify width r1 now (they are otherwise captured on first use).
    pub fn prepare(&mut self, m: &Model, seqs: &[&Seq], r1: usize) { self.ensure(m, seqs, r1 as i64); }

    fn ensure(&mut self, m: &Model, seqs: &[&Seq], r1: i64) {
        assert!(r1 >= 2, "verify width");
        if (r1 as usize) >= self.parts.len() { self.parts.resize_with(r1 as usize + 1, || None); }
        if self.parts[r1 as usize].is_some() { return; }
        let b = self.nseg;
        let (bu, ru) = (b as usize, r1 as usize);
        let rb = m.ple_row_bytes().unwrap_or(0) as usize;
        let mut l = Lay(0);
        let o = PartOff { ids: l.take(8 * bu * ru), pos: l.take(4 * bu), kk: l.take(4 * bu), packed: l.take(bu * ru * rb),
                          nd: l.take(4 * bu), cids: l.take(8 * bu * ru), sel: l.take(8 * bu), greedy: l.take(8 * bu * ru) };
        let io = Mapped::new(l.0);
        let ids = io.tensor(o.ids, &[b * r1], Kind::Int64);
        let pos = io.tensor(o.pos, &[b], Kind::Int);
        let kk = io.tensor(o.kk, &[b], Kind::Int);
        let packed = (rb > 0).then(|| io.tensor(o.packed, &[(bu * ru * rb) as i64], Kind::Uint8));
        // accepted row counts: written by the accept graph (or copied from the host-written mapped slot) into device
        // memory, which the commit reads (possibly from the side stream)
        let nd_map = io.tensor(o.nd, &[b], Kind::Int);
        let nd = Tensor::zeros([b], (Kind::Int, dev()));
        let cids = io.tensor(o.cids, &[b * r1], Kind::Int64);
        let sel = io.tensor(o.sel, &[b], Kind::Int64);
        let greedy = io.tensor(o.greedy, &[b * r1], Kind::Int64);
        let rows: Vec<(i64, i64)> = (0..b).map(|g| (g * r1, r1)).collect();
        let (g, (logits, x, pend)) = capture_shared(|| m.run_batch_dev(seqs, &ids, &pos, packed.as_ref(), &rows, false, false));
        // greedy accept on the device (the host repeats the same rule on the copied-out tokens)
        let spos = self.step.as_ref().map(|s| s.spos.shallow_clone());
        let (accept, ()) = capture_shared(|| {
            let gt = logits.argmax(-1, false);
            put(&greedy, &gt);
            let gv = gt.view([b, r1]);
            let rv = ids.view([b, r1]);
            let eq = gv.narrow(1, 0, r1 - 1).eq_tensor(&rv.narrow(1, 1, r1 - 1)).to_kind(Kind::Int64);
            let acc = eq.cumprod(1, Kind::Int64).sum_dim_intlist(1, false, Kind::Int64).minimum(&kk.to_kind(Kind::Int64));
            put(&nd, &(&acc + 1));
            let shifted = Tensor::cat(&[rv.narrow(1, 1, r1 - 1), rv.narrow(1, r1 - 1, 1)], 1);
            let bonus = gv.gather(1, &acc.unsqueeze(1), false);
            let col = Tensor::arange(r1, (Kind::Int64, dev())).unsqueeze(0);
            put(&cids, &bonus.expand([b, r1], false).where_self(&col.eq_tensor(&acc.unsqueeze(1)), &shifted).view([-1]));
            put(&sel, &(Tensor::arange_start_step(0, b * r1, r1, (Kind::Int64, dev())) + &acc));
            if let Some(sp) = &spos { put(sp, &(&pos + &acc + 1)); }
        });
        // own pool: the commit replays on the side stream, concurrently with graphs of the shared pool
        let (commit, ()) = capture(|| m.commit_segs(seqs, &pend, &rows, Some(&nd)));
        let catch = self.step.as_ref().map(|st| {
            let (g, ()) = capture_shared(|| {
                let (lg, xc) = m.mtp_batch_dev(seqs, &x, &cids, &pos, &rows, Pick::Rows(&sel));
                let (d, pair) = pick_pair(m, &lg);
                let _ = st.out.get(0).shallow_clone().copy_(&pair);
                put(&st.d, &d);
                put(&st.hid, &xc.index_select(0, &sel));
                let _ = st.cnt.shallow_clone().fill_(1);
            });
            g
        });
        let keep = vec![ids, pos, kk, cids, sel, greedy].into_iter().chain(packed).collect();
        self.parts[r1 as usize] = Some(Part { r1, io, o, g, logits, x, accept, commit, catch, nd_dev: nd, nd_map, _keep: (pend, keep) });
    }

    fn part(&self, r1: usize) -> &Part { self.parts[r1].as_ref().expect("verify width not captured") }

    /// Rows of all captured verify widths (their outputs and pending inputs stay allocated: about 5 MB a row).
    pub fn rows_held(&self) -> i64 { self.parts.iter().flatten().map(|p| p.r1 * self.nseg).sum() }

    /// Write the verify inputs of rows[g] (= [next, d1..], the same width r1 for all) of seqs[g] and replay the verify.
    fn verify_replay(&mut self, m: &Model, seqs: &[&Seq], rows: &[Vec<i64>], kk: &[usize]) -> usize {
        assert_eq!(rows.len() as i64, self.nseg);
        let r1 = rows[0].len();
        side_join();
        self.ensure(m, seqs, r1 as i64);
        let p = self.part(r1);
        let flat: Vec<i64> = rows.iter().flat_map(|r| { assert_eq!(r.len(), r1); r.iter().copied() }).collect();
        p.io.put(p.o.ids, &flat);
        p.io.put(p.o.pos, &seqs.iter().map(|s| s.pos as i32).collect::<Vec<_>>());
        p.io.put(p.o.kk, &kk.iter().map(|&v| v as i32).collect::<Vec<_>>());
        if m.ple_row_bytes().is_some() {
            let h: Vec<u8> = seqs.iter().zip(rows).flat_map(|(s, r)| m.ple_pack(s, r).unwrap()).collect();
            p.io.put(p.o.packed, &h);
        }
        p.g.replay();
        r1
    }

    /// Greedy round: verify rows[g] (width r1 for all) with at most kk[g] acceptable drafts, accept, commit and MTP
    /// catch-up on the device, one synchronization. Returns (target argmax [b][r1], verify stacks [b*r1, 4, 2560]);
    /// the host applies the same accept rule. Follow with `drafts`.
    pub fn round_greedy(&mut self, m: &Model, seqs: &[&Seq], rows: &[Vec<i64>], kk: &[usize]) -> (Vec<Vec<i64>>, Tensor) {
        let r1 = self.verify_replay(m, seqs, rows, kk);
        let p = self.part(r1);
        p.accept.replay();
        side(|| p.commit.replay());
        if let Some(c) = &p.catch { c.replay(); }
        sync();
        let g: Vec<i64> = p.io.get(p.o.greedy, self.nseg as usize * r1);
        (g.chunks(r1).map(|c| c.to_vec()).collect(), p.x.shallow_clone())
    }

    /// Verify only (sampling path): returns (logits [b*r1, V], verify stacks); then `commit_host`.
    pub fn verify(&mut self, m: &Model, seqs: &[&Seq], rows: &[Vec<i64>], kk: &[usize]) -> (Tensor, Tensor) {
        let r1 = self.verify_replay(m, seqs, rows, kk);
        let p = self.part(r1);
        (p.logits.shallow_clone(), p.x.shallow_clone())
    }

    /// Host-decided accept of the last verify (width r1): commit n[g] rows per sequence and run the MTP catch-up at the
    /// old positions p0[g] with tokens ids[g] (the r1 tokens following the verify rows, rows past n[g] - 1 arbitrary).
    pub fn commit_host(&self, r1: usize, ids: &[Vec<i64>], p0: &[i64], n: &[usize]) {
        let p = self.part(r1);
        let b = self.nseg as usize;
        p.io.put(p.o.nd, &n.iter().map(|&v| v as i32).collect::<Vec<_>>());
        put(&p.nd_dev, &p.nd_map);
        side(|| p.commit.replay());
        if let (Some(c), Some(st)) = (&p.catch, &self.step) {
            let flat: Vec<i64> = ids.iter().flat_map(|r| { assert_eq!(r.len(), r1); r.iter().copied() }).collect();
            p.io.put(p.o.cids, &flat);
            p.io.put(p.o.sel, &(0..b).map(|g| (g * r1 + n[g] - 1) as i64).collect::<Vec<_>>());
            st.io.put(0, &(0..b).map(|g| (p0[g] + n[g] as i64) as i32).collect::<Vec<_>>());
            c.replay();
        }
    }

    /// After the catch-up: draft steps while the policy asks for more; returns per sequence (drafts, confidences).
    /// look[g]: a prompt-lookup proposal (tokens, per-token confidence) for sequence g; when it starts with the first MTP
    /// draft it replaces that sequence's chain (which then needs no further steps). Returns which sequences use theirs.
    pub fn drafts(&self, pol: &Policy, look: &[Option<(Vec<i64>, f64)>]) -> (Drafts, Vec<bool>) {
        let st = self.step.as_ref().expect("no MTP head");
        let b = self.nseg as usize;
        let out_off = (4 * b + 15) / 16 * 16;   // StepG layout: spos, then out
        let read = |j: usize| -> Vec<f64> { st.io.get(out_off + j * 16 * b, 2 * b) };
        let mut res: Drafts = vec![(vec![], vec![]); b];
        let push = |v: Vec<f64>, res: &mut Drafts| for g in 0..b { res[g].0.push(v[2 * g] as i64); res[g].1.push(Policy::cal(v[2 * g + 1])); };
        assert!(pol.cap() <= st.kc);
        let mut used = vec![false; b];
        let take = |res: &mut Drafts, used: &[bool]| for g in 0..b {
            if let (true, Some((v, p))) = (used[g], &look.get(g).and_then(|l| l.as_ref())) {
                let c0 = res[g].1[0].max(*p);
                let n = v.len().min(pol.cap());
                res[g] = (v[..n].to_vec(), (0..n).map(|i| if i == 0 { c0 } else { *p }).collect());
            }
        };
        let agree = |res: &Drafts| -> Vec<bool> { (0..b).map(|g| matches!(look.get(g), Some(Some((v, _))) if v.first() == res[g].0.first())).collect() };
        if pol.theta.is_none() {   // fixed length: all steps, one synchronization
            for _ in 1..pol.cap() { st.g.replay(); }
            sync();
            for j in 0..pol.cap() { push(read(j), &mut res); }
            used = agree(&res);
            take(&mut res, &used);
            return (res, used);
        }
        sync();
        push(read(0), &mut res);
        used = agree(&res);
        let mut j = 1;
        loop {
            let open: Vec<Vec<f64>> = (0..b).filter(|&g| !used[g]).map(|g| res[g].1.clone()).collect();
            if open.is_empty() || !pol.more(&open) { break; }
            st.g.replay();
            sync();
            push(read(j), &mut res);
            j += 1;
        }
        take(&mut res, &used);
        (res, used)
    }
}
