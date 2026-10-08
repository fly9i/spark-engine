// SPDX-License-Identifier: MIT
//! Fast full expert preload (GLM53_FAST_PRELOAD, default on; =0 old per-tensor path in moefast).
//!
//! Every MoE layer's experts are laid out, for this rank, as one arena: per expert, per projection
//! (gate, up, down) the rank slice of the trellis, then suh, then svh (the same TP split as
//! moefast::load_expert_raw). A reader thread fills one pinned host buffer per layer from the checkpoint
//! shards (one O_DIRECT read per expert of its contiguous ~10 MiB region, slices copied out on a small
//! worker pool). An upload thread copies each buffer to one device arena with an SM copy kernel (the
//! pinned buffer is the same DRAM on GB10, read at device-memory speed) and recycles it. The serve path starts both at
//! process start, so disk reads overlap process-group setup and the non-expert load; the pool then
//! only builds views and tables. Install order, slot numbering and every resident byte are those of
//! the old path; GLM53_LOAD_CHECK=1 compares each expert with the old reader.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

use crate::safetensors::{Entry, ShardIndex};
use crate::tp::Tp;

pub(crate) fn enabled() -> bool { std::env::var("GLM53_FAST_PRELOAD").as_deref() != Ok("0") }

const ALIGN: usize = 4096;
const SEG_ALIGN: usize = 512;
#[cfg(target_arch = "aarch64")]
const O_DIRECT: i32 = 0o200000;
#[cfg(not(target_arch = "aarch64"))]
const O_DIRECT: i32 = 0o40000;

fn pad(n: usize, a: usize) -> usize { n.div_ceil(a) * a }

/// MemAvailable in bytes (0 if unreadable).
fn mem_available() -> i64 {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse::<i64>().ok())).map_or(0, |k| k * 1024)
}

/// Source entries of one expert projection.
struct ProjPlan { tr: Entry, suh: Entry, svh: Entry, down: bool }

/// This rank's slice geometry, identical for every expert of every MoE layer.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Layout {
    pub tr_shape: [[usize; 3]; 3],
    pub suh_len: [usize; 3],
    pub svh_len: [usize; 3],
    /// Byte offsets of (trellis, suh, svh) of each projection inside an expert record.
    pub offs: [[usize; 3]; 3],
    pub expert_bytes: usize,
    pub n_experts: usize,
    pub layer_bytes: usize,
}

impl Layout {
    fn of(p: &[ProjPlan; 3], tp: Tp, n_experts: usize) -> Layout {
        let w = tp.world;
        let mut tr_shape = [[0; 3]; 3];
        let (mut suh_len, mut svh_len, mut offs) = ([0; 3], [0; 3], [[0; 3]; 3]);
        let mut at = 0;
        for (pi, q) in p.iter().enumerate() {
            let s = &q.tr.shape;
            let (suh, svh) = (q.suh.nbytes / 2, q.svh.nbytes / 2);
            (tr_shape[pi], suh_len[pi], svh_len[pi]) = if w == 1 { ([s[0], s[1], s[2]], suh, svh) }
                else if q.down { ([s[0] / w, s[1], s[2]], suh / w, svh) }
                else { ([s[0], s[1] / w, s[2]], suh, svh / w) };
            let sizes = [tr_shape[pi].iter().product::<usize>() * 2, suh_len[pi] * 2, svh_len[pi] * 2];
            for (j, n) in sizes.into_iter().enumerate() { offs[pi][j] = at; at += pad(n, SEG_ALIGN); }
        }
        Layout { tr_shape, suh_len, svh_len, offs, expert_bytes: at, n_experts, layer_bytes: pad(at * n_experts, ALIGN) }
    }
    fn seg_len(&self, pi: usize, j: usize) -> usize {
        [self.tr_shape[pi].iter().product::<usize>(), self.suh_len[pi], self.svh_len[pi]][j] * 2
    }
}

/// Owned per-expert source plan; None when the checkpoint lacks the expected tensors or dtypes, or the
/// geometry is not uniform (the caller then keeps the old path).
struct Plan { layers: Vec<usize>, experts: Vec<Vec<[ProjPlan; 3]>>, layout: Layout }

fn plan(idx: &ShardIndex, n_layers: usize, n_experts: usize, tp: Tp) -> Option<Plan> {
    let layers: Vec<usize> = (0..n_layers)
        .filter(|l| idx.entries.contains_key(&format!("model.language_model.layers.{l}.mlp.experts.0.gate_proj.trellis")))
        .collect();
    let mut experts = Vec::with_capacity(layers.len());
    for &l in &layers {
        let mut v = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let p = format!("model.language_model.layers.{l}.mlp.experts.{e}");
            let mut projs = Vec::with_capacity(3);
            for w in ["gate_proj", "up_proj", "down_proj"] {
                let get = |k: &str| idx.entries.get(&format!("{p}.{w}.{k}")).cloned();
                let (Some(tr), Some(suh), Some(svh)) = (get("trellis"), get("suh"), get("svh")) else { return None };
                if tr.dtype != "I16" || tr.shape.len() != 3 || suh.dtype != "F16" || svh.dtype != "F16" { return None; }
                projs.push(ProjPlan { tr, suh, svh, down: w == "down_proj" });
            }
            v.push(<[ProjPlan; 3]>::try_from(projs).ok()?);
        }
        experts.push(v);
    }
    let layout = Layout::of(experts.first()?.first()?, tp, n_experts);
    let divisible = |p: &ProjPlan| tp.world == 1 || if p.down { p.tr.shape[0] % tp.world == 0 && p.suh.nbytes % (2 * tp.world) == 0 }
        else { p.tr.shape[1] % tp.world == 0 && p.svh.nbytes % (2 * tp.world) == 0 };
    if !experts.iter().flatten().all(|p| p.iter().all(divisible) && Layout::of(p, tp, n_experts) == layout) { return None; }
    Some(Plan { layers, experts, layout })
}

/// Reusable host buffer with an ALIGN-aligned window (O_DIRECT source/destination).
#[derive(Default)]
struct AlignedBuf { raw: Vec<u8>, start: usize }
impl AlignedBuf {
    fn with_len(len: usize) -> Self { let mut b = Self::default(); b.window(len); b }
    fn window(&mut self, len: usize) -> &mut [u8] {
        if self.raw.len() < len + ALIGN { self.raw = vec![0u8; len + ALIGN]; }
        self.start = (ALIGN - self.raw.as_ptr() as usize % ALIGN) % ALIGN;
        &mut self.raw[self.start..self.start + len]
    }
}

/// Layer buffer: pinned (cudaHostAlloc, so the device copy runs at DRAM speed) or aligned heap memory.
pub(crate) struct HostBuf { ptr: *mut u8, len: usize, pinned: bool, _heap: Option<AlignedBuf> }
unsafe impl Send for HostBuf {}
extern "C" {
    fn glm53_host_pin_alloc(p: *mut *mut std::ffi::c_void, bytes: usize) -> i32;
    fn glm53_host_pin_free(p: *mut std::ffi::c_void, bytes: usize);
    fn cudaStreamCreateWithFlags(s: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(s: *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(s: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpyAsync(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, n: usize, kind: i32, s: *mut std::ffi::c_void) -> i32;
}
impl HostBuf {
    pub(crate) fn pinned(len: usize) -> Self {
        let mut p = std::ptr::null_mut();
        // Mapped: the upload kernel reads it directly (rs_pinned_to_device). Anonymous mmap + cudaHostRegister
        // (GLM53_HOST_REGISTER, default on) so kernel compaction never isolates these ~1.3 GB of pinned staging
        // pages; GLM53_HOST_REGISTER=0 reverts to cudaHostAlloc (/dev/zero shmem, the old scattered behaviour).
        if unsafe { glm53_host_pin_alloc(&mut p, len) } == 0 && !p.is_null() && p as usize % ALIGN == 0 {
            return Self { ptr: p.cast(), len, pinned: true, _heap: None };
        }
        if !p.is_null() { unsafe { glm53_host_pin_free(p, len) }; }
        Self::heap(len)
    }
    pub(crate) fn heap(len: usize) -> Self {
        let mut b = AlignedBuf::with_len(len);
        let ptr = b.window(len).as_mut_ptr();
        Self { ptr, len, pinned: false, _heap: Some(b) }
    }
    fn bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) } }
    pub(crate) fn len(&self) -> usize { self.len }

    /// Copy the first `bytes` of this buffer into bytes [off, off+bytes) of the contiguous device tensor `dst` and
    /// wait for it (the buffer is reused right after). Pinned: SM copy kernel over 16-byte words (`off` must be
    /// 16-byte aligned; a rounded-up tail stays inside `dst`'s allocation, which the caching allocator rounds to
    /// 512 bytes); otherwise an ordinary copy.
    pub(crate) fn copy_to(&self, dst: &Tensor, off: usize, bytes: usize) {
        let total = dst.numel() as usize * dst.kind().elt_size_in_bytes();
        assert!(bytes <= self.len && dst.is_contiguous() && off + bytes <= total);
        if bytes == 0 { return; }
        if self.pinned && self.len >= pad(bytes, 16) && off % 16 == 0 {
            extern "C" { fn rs_pinned_to_device(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, bytes: u64, stream: *mut std::ffi::c_void) -> i32; }
            // Legacy default stream: ordered with the caller's work on it; synchronized before returning.
            let d = unsafe { dst.data_ptr().cast::<u8>().add(off) };
            assert_eq!(unsafe { rs_pinned_to_device(d.cast(), self.ptr.cast(), pad(bytes, 16) as u64, std::ptr::null_mut()) }, 0, "pinned upload");
            assert_eq!(unsafe { cudaStreamSynchronize(std::ptr::null_mut()) }, 0, "pinned upload");
        } else {
            let src = unsafe { Tensor::from_blob(self.ptr, &[bytes as i64], &[1], Kind::Uint8, Device::Cpu) };
            let _ = dst.view_dtype(Kind::Uint8).view([-1]).narrow(0, off as i64, bytes as i64).copy_(&src);
        }
    }
}
impl Drop for HostBuf {
    fn drop(&mut self) { if self.pinned { unsafe { glm53_host_pin_free(self.ptr.cast(), self.len) }; } }
}

/// Shards opened once; `true` = O_DIRECT (buffered fallback if the filesystem refuses it).
struct DirectFiles { files: HashMap<String, (std::fs::File, bool)> }
impl DirectFiles {
    fn open(dir: &Path, plan: &Plan) -> Self {
        use std::os::unix::fs::OpenOptionsExt;
        let mut files = HashMap::new();
        for p in plan.experts.iter().flatten().flatten() {
            for name in [&p.tr.file, &p.suh.file, &p.svh.file] {
                if files.contains_key(name) { continue; }
                let path = dir.join(name);
                let f = match std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(&path) {
                    Ok(f) => (f, true),
                    Err(_) => (std::fs::File::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display())), false),
                };
                files.insert(name.clone(), f);
            }
        }
        Self { files }
    }
    fn direct(&self) -> usize { self.files.values().filter(|f| f.1).count() }
    /// Read [off, off+len) of `file` into `buf`; returns the position of `off` inside `buf.raw`.
    fn read(&self, file: &str, off: usize, len: usize, buf: &mut AlignedBuf) -> usize {
        let (f, direct) = &self.files[file];
        let (a0, a1) = if *direct { (off & !(ALIGN - 1), pad(off + len, ALIGN)) } else { (off, off + len) };
        let dst = buf.window(a1 - a0);
        let done = read_full(f, dst, a0 as u64).unwrap_or_else(|e| panic!("read {file} @{a0}: {e}"));
        assert!(done >= off + len - a0, "short read {file} @{off} len {len}: {done} of {}", off + len - a0);
        buf.start + (off - a0)
    }
}

/// pread until `dst` is full or EOF (an O_DIRECT tail may end inside the last aligned block).
fn read_full(f: &std::fs::File, dst: &mut [u8], at: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    let mut done = 0;
    while done < dst.len() {
        let n = f.read_at(&mut dst[done..], at + done as u64)?;
        if n == 0 { break; }
        done += n;
    }
    Ok(done)
}

/// Write this rank's slices of one expert into `dst` (one expert record of the layout) from the shards.
/// Each expert's trellis tensors are adjacent, so the pieces merge into one ~10 MiB read.
fn fill_from_shards(files: &DirectFiles, p: &[ProjPlan; 3], lay: &Layout, tp: Tp, bufs: &mut Vec<AlignedBuf>, dst: &mut [u8]) {
    const GAP: usize = 1 << 20;
    let (rank, world) = (tp.rank, tp.world);
    // Source byte range of every piece: (file, offset, len), in (tr, suh, svh) x projection order.
    let mut pieces: Vec<(&str, usize, usize)> = Vec::with_capacity(9);
    for (pi, q) in p.iter().enumerate() {
        let tr_len = lay.seg_len(pi, 0);
        let tr_off = if q.down && world > 1 { q.tr.offset + rank * tr_len } else { q.tr.offset };
        pieces.push((&q.tr.file, tr_off, if q.down || world == 1 { tr_len } else { q.tr.nbytes }));
        let (suh, svh) = (lay.seg_len(pi, 1), lay.seg_len(pi, 2));
        pieces.push((&q.suh.file, q.suh.offset + if q.down && world > 1 { rank * suh } else { 0 }, suh));
        pieces.push((&q.svh.file, q.svh.offset + if !q.down && world > 1 { rank * svh } else { 0 }, svh));
    }
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_by_key(|&i| (pieces[i].0, pieces[i].1));
    let mut regions: Vec<(&str, usize, usize)> = Vec::new();
    let mut at = vec![(0usize, 0usize); pieces.len()];
    for &i in &order {
        let (f, off, len) = pieces[i];
        match regions.last_mut() {
            Some(r) if r.0 == f && off <= r.2 + GAP => r.2 = r.2.max(off + len),
            _ => regions.push((f, off, off + len)),
        }
        let r = regions.len() - 1;
        at[i] = (r, off - regions[r].1);
    }
    if bufs.len() < regions.len() { bufs.resize_with(regions.len(), Default::default); }
    let starts: Vec<usize> = regions.iter().zip(bufs.iter_mut()).map(|(&(f, a, b), buf)| files.read(f, a, b - a, buf)).collect();
    let src = |i: usize| { let (r, o) = at[i]; let s = starts[r] + o; &bufs[r].raw[s..s + pieces[i].2] };
    for (pi, q) in p.iter().enumerate() {
        let o = lay.offs[pi];
        let tr = src(pi * 3);
        if q.down || world == 1 {
            dst[o[0]..o[0] + tr.len()].copy_from_slice(tr);
        } else {
            // Column split: each of the kt rows keeps its nth-wide slice.
            let [kt, nth, wd] = lay.tr_shape[pi];
            let (row, part) = (nth * world * wd * 2, nth * wd * 2);
            for r in 0..kt {
                let s = r * row + rank * part;
                dst[o[0] + r * part..o[0] + (r + 1) * part].copy_from_slice(&tr[s..s + part]);
            }
        }
        for j in 1..3 { let s = src(pi * 3 + j); dst[o[j]..o[j] + s.len()].copy_from_slice(s); }
    }
}

/// Fill a whole layer buffer from the shards on the rayon pool (padding zeroed).
/// Worker pool of the expert reads (GLM53_PRELOAD_THREADS, default 8): enough concurrent ~10 MiB reads to
/// keep the NVMe busy while leaving cores to the main thread's non-expert load running at the same time.
fn read_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("GLM53_PRELOAD_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).clamp(1, 64);
        rayon::ThreadPoolBuilder::new().num_threads(n).thread_name(|i| format!("expert-read-{i}")).build().expect("expert read pool")
    })
}

/// Fill consecutive expert records (one per entry of `experts`) at the front of `buf` on the rayon pool.
fn fill_records(files: &DirectFiles, experts: &[[ProjPlan; 3]], lay: &Layout, tp: Tp, buf: &mut [u8]) {
    buf[..lay.expert_bytes * experts.len()].par_chunks_mut(lay.expert_bytes).zip(experts.par_iter())
        .for_each_init(Vec::new, |bufs, (dst, p)| { dst.fill(0); fill_from_shards(files, p, lay, tp, bufs, dst) });
}

/// GLM53_LOAD_CHECK=1: each of the `n` records at the front of `buf` (experts e0..e0+n of `layer`) must equal
/// the old reader's upload values.
fn check_records(idx: &ShardIndex, layer: usize, e0: usize, n: usize, lay: &Layout, tp: Tp, buf: &[u8]) {
    buf[..lay.expert_bytes * n].par_chunks(lay.expert_bytes).enumerate().for_each(|(i, rec)| {
        let e = e0 + i;
        let old = crate::moefast::load_expert_raw(idx, layer, e, tp);
        for (pi, (tr, shape, suh, svh)) in old.iter().enumerate() {
            let ctx = format!("GLM53_LOAD_CHECK: layer {layer} expert {e} projection {pi}");
            let o = lay.offs[pi];
            assert_eq!(shape.iter().map(|&d| d as usize).collect::<Vec<_>>(), lay.tr_shape[pi].to_vec(), "{ctx}: shape");
            let tr_bytes: &[u8] = unsafe { std::slice::from_raw_parts(tr.as_ptr().cast(), tr.len() * 2) };
            assert!(rec[o[0]..o[0] + tr_bytes.len()] == *tr_bytes, "{ctx}: trellis differs");
            for (j, src) in [(1, suh), (2, svh)] {
                let h = Vec::<i16>::try_from(&Tensor::from_slice(src).to_kind(Kind::Half).view_dtype(Kind::Int16)).unwrap();
                let hb: &[u8] = unsafe { std::slice::from_raw_parts(h.as_ptr().cast(), h.len() * 2) };
                assert!(hb.len() == lay.seg_len(pi, j) && rec[o[j]..o[j] + hb.len()] == *hb, "{ctx}: scales differ");
            }
        }
    });
}

/// Experts e0..e0+n of one layer in a pinned staging buffer, with their gate/up suh values.
struct Chunk { layer: usize, e0: usize, n: usize, buf: HostBuf, suh: Vec<(Vec<f32>, Vec<f32>)> }

/// Staging chunk size: pinned host memory is not reclaimable, so it stays small (two ~256 MiB buffers rather than
/// two whole ~1.8 GB layers) to keep the load's peak under the serving footprint.
const CHUNK_BYTES: usize = 256 << 20;

/// One device-resident layer arena, plus the gate/up suh values the shared-input certificate observes.
pub(crate) struct Loaded { pub layer: usize, pub arena: Tensor, pub suh: Vec<(Vec<f32>, Vec<f32>)> }

#[derive(Default)]
struct Stats { read_s: f64, upload_s: f64, alloc_s: f64, alloc_wait_s: f64, gate_s: f64, wait_s: f64, pinned: usize, direct: String }

/// Background expert reader + uploader; consumed layer by layer by MoeFast::preload_all_with.
pub struct ExpertReader {
    rx: Option<Receiver<Loaded>>,
    handles: Vec<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    stats: Arc<Mutex<Stats>>,
    pub(crate) layers: Vec<usize>,
    pub(crate) n_experts: usize,
    pub(crate) tp: Tp,
    pub(crate) layout: Layout,
    t0: Instant,
}

impl ExpertReader {
    /// Start reading/uploading every MoE layer's experts for rank `tp` (None: disabled, CPU device, or a
    /// checkpoint the fast path does not cover).
    pub fn start(dir: &Path, n_layers: usize, n_experts: usize, tp: Tp, dev: Device, after: Vec<Arc<crate::safetensors::Done>>) -> Option<Self> {
        if !enabled() || !dev.is_cuda() { return None; }
        let t0 = Instant::now();
        let idx = ShardIndex::scan(dir).ok()?;
        let plan = plan(&idx, n_layers, n_experts, tp)?;
        let files = DirectFiles::open(&idx.dir, &plan);
        let (layers, layout) = (plan.layers.clone(), plan.layout.clone());
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(Stats {
            direct: format!("O_DIRECT {}/{} shards", files.direct(), files.files.len()), ..Default::default() }));
        // Two pinned chunk buffers: the reader fills one while the uploader copies the other.
        let per_chunk = (CHUNK_BYTES / layout.expert_bytes).clamp(1, layout.n_experts);
        let (free_tx, free_rx) = sync_channel::<HostBuf>(2);
        let (full_tx, full_rx) = sync_channel::<Chunk>(1);
        let (out_tx, out_rx) = std::sync::mpsc::channel::<Loaded>();
        let check = crate::weights::load_check_enabled();
        // Arena allocations run on their own thread, one layer ahead of the uploads: each ~1.8 GB allocation maps
        // (and clears) its pages, ~0.1-0.2 s, off the upload path. Bounded: allocating every arena up front
        // (76 GB) pushed the prefetched non-expert pages out of the page cache before the main thread read them.
        let (arena_tx, arena_rx) = sync_channel::<Tensor>(1);
        let allocator = {
            let (stop, stats, n, bytes, after) = (stop.clone(), stats.clone(), layers.len(), layout.layer_bytes as i64, after.clone());
            std::thread::spawn(move || {
                // Not during process-group setup (driver work at process start slowed it 2.4 -> 7 s).
                for a in &after { a.wait(); }
                const MARGIN: i64 = 6 << 30;
                for i in 0..n {
                    if stop.load(Ordering::Relaxed) { return; }
                    // Leave the not-yet-read prefetched pages alone: MemAvailable counts them as reclaimable, and read-once
                    // pages are the kernel's first victims. Each pending byte also becomes about one byte of non-expert
                    // device memory when the main thread uploads it, so allocate only while MemAvailable minus twice the
                    // pending bytes still fits every remaining arena (plus a margin for staging); otherwise wait.
                    let tg = Instant::now();
                    let remaining = (n - i) as i64 * bytes;
                    loop {
                        let pending = crate::weights::cache_pending_bytes();
                        if pending <= 0 || mem_available() - 2 * pending > remaining + MARGIN || stop.load(Ordering::Relaxed) { break; }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    stats.lock().unwrap().gate_s += tg.elapsed().as_secs_f64();
                    let t = Instant::now();
                    let a = Tensor::empty([bytes], (Kind::Uint8, dev));
                    stats.lock().unwrap().alloc_s += t.elapsed().as_secs_f64();
                    if arena_tx.send(a).is_err() { return; }
                }
            })
        };
        let reader = {
            let (stop, stats, lay) = (stop.clone(), stats.clone(), layout.clone());
            std::thread::spawn(move || {
                // Disk and host memory go to the (smaller, earlier-needed) non-expert prefetch first.
                for a in &after { a.wait(); }
                stats.lock().unwrap().wait_s = t0.elapsed().as_secs_f64();
                let mut made = 0;
                let f32s = |b: &[u8]| b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect::<Vec<f32>>();
                for (li, &layer) in plan.layers.iter().enumerate() {
                    for e0 in (0..lay.n_experts).step_by(per_chunk) {
                        if stop.load(Ordering::Relaxed) { return; }
                        let n = per_chunk.min(lay.n_experts - e0);
                        let mut buf = if made < 2 { made += 1; let b = HostBuf::pinned(per_chunk * lay.expert_bytes); stats.lock().unwrap().pinned += b.pinned as usize; b }
                            else { match free_rx.recv() { Ok(b) => b, Err(_) => return } };
                        let t = Instant::now();
                        read_pool().install(|| fill_records(&files, &plan.experts[li][e0..e0 + n], &lay, tp, buf.bytes_mut()));
                        if check { check_records(&idx, layer, e0, n, &lay, tp, buf.bytes()); }
                        let suh = buf.bytes()[..lay.expert_bytes * n].chunks(lay.expert_bytes).map(|r| {
                            let g = &r[lay.offs[0][1]..lay.offs[0][1] + lay.seg_len(0, 1)];
                            let u = &r[lay.offs[1][1]..lay.offs[1][1] + lay.seg_len(1, 1)];
                            (f32s(g), f32s(u))
                        }).collect();
                        stats.lock().unwrap().read_s += t.elapsed().as_secs_f64();
                        if full_tx.send(Chunk { layer, e0, n, buf, suh }).is_err() { return; }
                    }
                }
            })
        };
        let uploader = {
            let (stop, stats) = (stop.clone(), stats.clone());
            let layout = layout.clone();
            std::thread::spawn(move || {
                let _guard = tch::no_grad_guard();
                // Own non-blocking stream: the copies neither wait for nor stall the main thread's work on the
                // default stream. Each arena is complete (stream synchronized) before it is handed over.
                let mut stream = std::ptr::null_mut();
                assert_eq!(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }, 0, "upload stream");
                let eb = layout.expert_bytes;
                assert_eq!(eb % 16, 0, "expert record size must be 16-byte aligned for the upload kernel");
                let mut cur: Option<(Tensor, Vec<(Vec<f32>, Vec<f32>)>)> = None;
                for c in full_rx {
                    if stop.load(Ordering::Relaxed) { break; }
                    let t = Instant::now();
                    if c.e0 == 0 {
                        let Ok(arena) = arena_rx.recv() else { break };
                        cur = Some((arena, Vec::with_capacity(layout.n_experts)));
                    }
                    stats.lock().unwrap().alloc_wait_s += t.elapsed().as_secs_f64();
                    let (arena, suh) = cur.as_mut().expect("chunks arrive in order");
                    let (dst, len) = (unsafe { arena.data_ptr().cast::<u8>().add(c.e0 * eb) }, c.n * eb);
                    if c.buf.pinned {
                        extern "C" { fn rs_pinned_to_device(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, bytes: u64, stream: *mut std::ffi::c_void) -> i32; }
                        assert_eq!(unsafe { rs_pinned_to_device(dst.cast(), c.buf.ptr.cast(), len as u64, stream) }, 0, "expert upload kernel");
                    } else {
                        const H2D: i32 = 1;
                        assert_eq!(unsafe { cudaMemcpyAsync(dst.cast(), c.buf.ptr.cast(), len, H2D, stream) }, 0, "expert upload");
                    }
                    assert_eq!(unsafe { cudaStreamSynchronize(stream) }, 0, "expert upload");   // the buffer is refilled next
                    stats.lock().unwrap().upload_s += t.elapsed().as_secs_f64();
                    let _ = free_tx.send(c.buf);
                    suh.extend(c.suh);
                    if c.e0 + c.n == layout.n_experts {
                        let (arena, suh) = cur.take().unwrap();
                        if out_tx.send(Loaded { layer: c.layer, arena, suh }).is_err() { break; }
                    }
                }
                unsafe { cudaStreamDestroy(stream) };
            })
        };
        Some(Self { rx: Some(out_rx), handles: vec![reader, uploader, allocator], stop, stats, layers, n_experts, tp, layout, t0 })
    }

    pub(crate) fn recv(&self) -> Loaded { self.rx.as_ref().unwrap().recv().expect("expert reader stopped (see the panic above)") }

    pub(crate) fn summary(&self) -> String {
        let s = self.stats.lock().unwrap();
        format!("{}, pinned buffers {}/2, reader started after {:.1}s, read {:.1}s, upload {:.1}s (of which waiting for arenas {:.1}s; arena allocation {:.1}s on its own thread, held back {:.1}s for the prefetched pages), pool reached {:.1}s after start", s.direct, s.pinned, s.wait_s, s.read_s, s.upload_s, s.alloc_wait_s, s.alloc_s, s.gate_s, self.t0.elapsed().as_secs_f64())
    }

    /// Views of expert `e`'s nine tensors (tr, suh, svh x gate/up/down) in a layer arena.
    pub(crate) fn views(&self, arena: &Tensor, e: usize) -> Vec<Tensor> {
        let lay = &self.layout;
        let base = e * lay.expert_bytes;
        let mut v = Vec::with_capacity(9);
        for pi in 0..3 {
            let seg = |j: usize, kind: Kind| arena.narrow(0, (base + lay.offs[pi][j]) as i64, lay.seg_len(pi, j) as i64).view_dtype(kind);
            v.push(seg(0, Kind::Int16).view(lay.tr_shape[pi].map(|d| d as i64).as_slice()));
            v.push(seg(1, Kind::Half));
            v.push(seg(2, Kind::Half));
        }
        v
    }
}

impl Drop for ExpertReader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.rx = None;
        for h in self.handles.drain(..) {
            // A reader panic already reached the consumer as a closed channel; do not panic again in drop.
            let _ = h.join();
        }
    }
}
