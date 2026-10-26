//! Engram n-gram memory of DeepSeek-V4.1-Flash: the hash ids of every position (official `engram.py`
//! NgramHashState, constants exported by reference/ds41/export_engram.py) and the table rows, read from the
//! checkpoint (the 2 x 98 GB tables never enter GPU memory): prefill-sized gathers by explicit page reads (`io`),
//! decode's few rows through the mmap (MADV_RANDOM).
use super::load::Checkpoint;
use serde_json::Value;
use std::path::Path;
use tch::{Kind, Tensor};

pub struct Hash {
    pub layer_ids: Vec<usize>,
    max_ngram: usize,
    n_heads: usize,
    pad: i64,
    primes: Vec<Vec<Vec<i64>>>,     // [layer][ngram-1][head]
    offsets: Vec<Vec<i64>>,         // [layer][col]
    multipliers: Vec<Vec<i64>>,     // [layer][shift]
    token_map: Vec<i64>,
    pub head_dim: i64,
}

/// A token that takes no part in an n-gram (image spans).
pub const DEAD: i64 = -1;

impl Hash {
    pub fn load(path: &Path) -> Hash {
        let v: Value = serde_json::from_reader(std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
        let ints = |x: &Value| x.as_array().unwrap().iter().map(|y| y.as_i64().unwrap()).collect::<Vec<_>>();
        Hash {
            layer_ids: ints(&v["layer_ids"]).into_iter().map(|x| x as usize).collect(),
            max_ngram: v["max_ngram_size"].as_u64().unwrap() as usize,
            n_heads: v["n_heads"].as_u64().unwrap() as usize,
            pad: v["pad_id"].as_i64().unwrap(),
            primes: v["primes"].as_array().unwrap().iter().map(|l| l.as_array().unwrap().iter().map(ints).collect()).collect(),
            offsets: v["offsets"].as_array().unwrap().iter().map(ints).collect(),
            multipliers: v["multipliers"].as_array().unwrap().iter().map(ints).collect(),
            token_map: ints(&v["token_map"]),
            head_dim: v["head_dim"].as_i64().unwrap(),
        }
    }
    pub fn cols(&self) -> usize { (self.max_ngram - 1) * self.n_heads }
    /// Compressed id of a token; DEAD for DEAD and for ids outside the vocabulary (salted image placeholders,
    /// crate::mm_canvas: image positions take no part in an n-gram).
    pub fn compress(&self, id: i64) -> i64 { if id < 0 || id as usize >= self.token_map.len() { DEAD } else { self.token_map[id as usize] } }
    /// Hash ids [n_layers][cols] of the position whose compressed history ends with `hist` (oldest first, the
    /// position's own token last; fewer than max_ngram entries at the start of the sequence).
    pub fn at(&self, hist: &[i64]) -> Vec<Vec<i64>> {
        let mut out = vec![Vec::with_capacity(self.cols()); self.layer_ids.len()];
        self.at_into(hist, &mut out);
        out
    }
    /// `at`, appended to `out[layer]` (prefill: a chunk's hash ids without allocations per position).
    pub fn at_into(&self, hist: &[i64], out: &mut [Vec<i64>]) {
        let n = self.max_ngram;
        let mut toks = vec![self.pad; n];
        let mut blocked = false;
        for (shift, t) in toks.iter_mut().enumerate() {
            let src = if shift < hist.len() { hist[hist.len() - 1 - shift] } else { DEAD };
            blocked = blocked || shift >= hist.len() || src == DEAD;
            *t = if blocked { self.pad } else { src };
        }
        for (li, out) in out.iter_mut().enumerate().take(self.layer_ids.len()) {
            let m = &self.multipliers[li];
            let mut rolling = toks[0].wrapping_mul(m[0]);
            let mut c = 0;
            for i in 1..n {
                rolling ^= toks[i].wrapping_mul(m[i]);
                for h in 0..self.n_heads { out.push(rolling.rem_euclid(self.primes[li][i - 1][h]) + self.offsets[li][c]); c += 1; }
            }
        }
    }
}

/// Host buffers of the prefill gathers, recycled: a 4096-token chunk's rows are 24 MiB (weights) + 0.75 MiB (scales) per
/// engram layer, allocated and freed every chunk on changing threads (10-19: with glibc's dynamic mmap threshold these
/// stayed in the malloc arenas, host RSS +0.6 GiB per 54k-token cold prefill). Buffers of at least POOL_MIN bytes go
/// back here after the upload (at most POOL_MAX of them, ~100 MiB: both layers' weights and scales for this chunk
/// and the next); smaller ones (decode) are plain allocations.
const POOL_MIN: usize = 1 << 20;
const POOL_MAX: usize = 8;
static POOL: std::sync::Mutex<Vec<Vec<u8>>> = std::sync::Mutex::new(Vec::new());
/// A zeroed buffer of `n` bytes (from the pool when large).
pub(crate) fn buf(n: usize) -> Vec<u8> {
    if n < POOL_MIN { return vec![0u8; n]; }
    let mut v = {
        let mut p = POOL.lock().unwrap();
        // the smallest that fits, else the largest (grown below)
        let i = p.iter().enumerate().filter(|(_, b)| b.capacity() >= n).min_by_key(|(_, b)| b.capacity()).map(|(i, _)| i)
            .or_else(|| p.iter().enumerate().max_by_key(|(_, b)| b.capacity()).map(|(i, _)| i));
        i.map(|i| p.swap_remove(i)).unwrap_or_default()
    };
    v.clear();
    v.resize(n, 0);
    v
}
/// Returns a buffer from `buf` to the pool.
pub(crate) fn give(v: Vec<u8>) {
    if v.capacity() < POOL_MIN { return; }
    let mut p = POOL.lock().unwrap();
    p.push(v);
    if p.len() > POOL_MAX {
        let i = p.iter().enumerate().min_by_key(|(_, b)| b.capacity()).map(|(i, _)| i).unwrap();
        p.swap_remove(i);
    }
}

/// Table rows of one engram layer, dequantized like ParallelEngramEmbedding: [n, head_dim] BF16 for row ids `rows`.
extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }
const MADV_WILLNEED: i32 = 3;

/// DS41_ENGRAM_STATS=1: every 200 calls, the mean host time of a gather and its row count.
fn stats(rows: usize, t: std::time::Instant) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("DS41_ENGRAM_STATS").is_ok_and(|v| v == "1")) { return; }
    static ACC: std::sync::Mutex<(usize, usize, f64)> = std::sync::Mutex::new((0, 0, 0.0));
    if rows > 4096 { eprintln!("[ds41-engram] gather of {rows} rows: {:.1} ms", t.elapsed().as_secs_f64() * 1e3); return; }
    let mut a = ACC.lock().unwrap();
    a.0 += 1; a.1 += rows; a.2 += t.elapsed().as_secs_f64() * 1e3;
    if a.0 == 200 { eprintln!("[ds41-engram] {} gathers: mean {:.1} rows, {:.3} ms", a.0, a.1 as f64 / 200.0, a.2 / 200.0); *a = (0, 0, 0.0); }
}

pub fn rows(ck: &Checkpoint, layer: usize, rows: &[i64], head_dim: i64) -> Tensor {
    // rows a background read already has (the next chunk's, read while this one ran); otherwise through the mapping
    // (warm pages are a plain copy there; a cold first chunk costs about the same either way)
    let (wbuf, sbuf) = match io::take(ck, layer, rows) {
        Some(h) => h.join().expect("engram prefetch thread"),
        None => { let (w, s) = tables(ck, layer); gather(w, s, rows, head_dim) }
    };
    upload(wbuf, sbuf, rows.len() as i64, head_dim, ck.dev)
}

/// The weight and scale tables of an engram layer (mapped for the life of the process).
fn tables(ck: &Checkpoint, layer: usize) -> (&'static [u8], &'static [u8]) {
    let (wn, sn) = (format!("layers.{layer}.engram.embed.weight"), format!("layers.{layer}.engram.embed.scale"));
    // SAFETY: the checkpoint (and its maps) lives as long as the model, i.e. the process
    unsafe { (std::mem::transmute::<&[u8], &'static [u8]>(ck.bytes(&wn)), std::mem::transmute::<&[u8], &'static [u8]>(ck.bytes(&sn))) }
}

/// The host gather of a later engram layer's rows started on a background thread (the encoder layers before it run
/// meanwhile); `upload` the result when the layer is reached.
pub fn rows_async(ck: &Checkpoint, layer: usize, rows: Vec<i64>, head_dim: i64) -> std::thread::JoinHandle<(Vec<u8>, Vec<u8>)> {
    if let Some(h) = io::take(ck, layer, &rows) { return h; }
    let (w, s) = tables(ck, layer);
    std::thread::spawn(move || gather(w, s, &rows, head_dim))
}

/// `rows` as the act_quant operands of their BF16 values (E4M3 bytes [n, head_dim], UE8M0 bytes [n, head_dim / 32]).
pub fn rows_q(ck: &Checkpoint, layer: usize, rows: &[i64], head_dim: i64) -> (Tensor, Tensor) {
    let (wbuf, sbuf) = match io::take(ck, layer, rows) {
        Some(h) => h.join().expect("engram prefetch thread"),
        None => { let (w, s) = tables(ck, layer); gather(w, s, rows, head_dim) }
    };
    upload_q(wbuf, sbuf, ck.dev)
}

/// The gathered bytes uploaded and requantized on the device (kern::requant_fp8): the bytes act_quant gives `upload`'s
/// BF16 rows, with no FP32 / BF16 copy of them; the buffers go back to the pool.
pub fn upload_q(wbuf: Vec<u8>, sbuf: Vec<u8>, dev: tch::Device) -> (Tensor, Tensor) {
    let dev_bytes = |b: &Vec<u8>| unsafe { Tensor::from_blob(b.as_ptr(), &[b.len() as i64], &[], Kind::Uint8, tch::Device::Cpu) }.to_device(dev);
    let (wd, sd) = (dev_bytes(&wbuf), dev_bytes(&sbuf));
    give(wbuf);
    give(sbuf);
    super::kern::requant_fp8(&wd, &sd)
}

/// BF16 values of E4M3 bytes + E8M0 bytes per 32 ([.., K] and [.., K / 32]).
pub fn dequant(q: &Tensor, s: &Tensor) -> Tensor {
    let (r, k) = (q.size()[0], q.size()[1]);
    let wt = q.contiguous().view_dtype(Kind::Float8e4m3fn).to_kind(Kind::Float).view([r, k / 32, 32]);
    (wt * super::ops::e8m0(&s.contiguous()).view([r, k / 32, 1])).view([r, k]).to_kind(Kind::BFloat16)
}

/// Rows as BF16 [n, head_dim] on `dev` from the gathered FP8 bytes and E8M0 scales; the buffers go back to the pool.
pub fn upload(wbuf: Vec<u8>, sbuf: Vec<u8>, n: i64, head_dim: i64, dev: tch::Device) -> Tensor {
    // straight from the buffers (no CPU tensor copy first); a blocking copy from pageable memory, so the buffers are free
    // again when it returns
    let dev_bytes = |b: &Vec<u8>| unsafe { Tensor::from_blob(b.as_ptr(), &[b.len() as i64], &[], Kind::Uint8, tch::Device::Cpu) }.to_device(dev);
    let (wd, sd) = (dev_bytes(&wbuf), dev_bytes(&sbuf));
    give(wbuf);
    give(sbuf);
    let wt = wd.view_dtype(Kind::Float8e4m3fn).to_kind(Kind::Float).view([n, head_dim / 32, 32]);
    let st = super::ops::e8m0(&sd).view([n, head_dim / 32, 1]);
    (wt * st).view([n, head_dim]).to_kind(Kind::BFloat16)
}

/// Host gather of table rows: FP8 bytes [n * head_dim] and scale bytes [n * head_dim / 32].
pub fn gather(w: &[u8], s: &[u8], rows: &[i64], head_dim: i64) -> (Vec<u8>, Vec<u8>) {
    let t0 = std::time::Instant::now();
    let (wb, sb) = (head_dim as usize, head_dim as usize / 32);
    let mut wbuf = buf(rows.len() * wb);
    let mut sbuf = buf(rows.len() * sb);
    // the tables live on NVMe (page cache only partly; a serving node has ~5 GB of it): rows not cached fault one by
    // one, so gather from many threads (DS41_ENGRAM_THREADS, default 256) to keep the device queue deep: 98k new rows
    // (4096 prompt tokens, one layer) 750 ms with 32 threads, 530 ms with 256 (~370k page reads/s)
    let threads = std::env::var("DS41_ENGRAM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256usize).max(1);
    // few rows (decode): start every row's read at once (async readahead of the pages), then copy; many rows
    // (prefill): one syscall per row cost more than the reads (0.8 s per 8k tokens), the copy threads fault in parallel
    let page = 4096usize;
    for &r in rows.iter().take(if rows.len() <= 1024 { rows.len() } else { 0 }) {
        let r = r as usize;
        for (buf, b) in [(w, wb), (s, sb)] {
            let a = buf.as_ptr() as usize + r * b;
            let p0 = a & !(page - 1);
            unsafe { madvise(p0 as *mut std::ffi::c_void, a + b - p0, MADV_WILLNEED); }
        }
    }
    let per = rows.len().div_ceil(threads).max(64);
    std::thread::scope(|sc| {
        for ((rs, wo), so) in rows.chunks(per).zip(wbuf.chunks_mut(per * wb)).zip(sbuf.chunks_mut(per * sb)) {
            sc.spawn(move || {
                for (i, &r) in rs.iter().enumerate() {
                    let r = r as usize;
                    wo[i * wb..(i + 1) * wb].copy_from_slice(&w[r * wb..(r + 1) * wb]);
                    so[i * sb..(i + 1) * sb].copy_from_slice(&s[r * sb..(r + 1) * sb]);
                }
            });
        }
    });
    stats(rows.len(), t0);
    (wbuf, sbuf)
}

/// Reads the table rows of `rows` (per engram layer) into the page cache from a background thread pool, so a later
/// `rows` call finds them there (prefill: the next chunk's rows load while the current chunk computes).
pub fn prefetch(ck: &Checkpoint, rows: Vec<(usize, Vec<i64>)>, head_dim: i64) {
    match io::mode() {
        io::Mode::Mmap => {}
        // the next chunk's pages: asynchronous kernel readahead into the page cache (no faults, no copies)
        io::Mode::Fadvise => {
            let ts: Vec<(std::sync::Arc<io::Table>, Vec<i64>)> = rows.into_iter().filter_map(|(l, r)| io::table(ck, l).map(|t| (t, r))).collect();
            std::thread::spawn(move || for (t, r) in ts { t.readahead(&r, head_dim) });
            return;
        }
        // the rows themselves, read in the background and kept until the chunk's gather asks for exactly them
        io::Mode::Pread | io::Mode::Direct => { for (l, r) in rows { io::prefetch_rows(ck, l, r, head_dim); } return; }
    }
    let mut addrs: Vec<(usize, usize)> = Vec::new();   // (address, bytes) of every row of both tables
    for (layer, rs) in &rows {
        let (w, s) = (ck.bytes(&format!("layers.{layer}.engram.embed.weight")), ck.bytes(&format!("layers.{layer}.engram.embed.scale")));
        let (wb, sb) = (head_dim as usize, head_dim as usize / 32);
        for &r in rs {
            addrs.push((w.as_ptr() as usize + r as usize * wb, wb));
            addrs.push((s.as_ptr() as usize + r as usize * sb, sb));
        }
    }
    if addrs.is_empty() { return; }
    // the mapping lives as long as the model (the whole process)
    std::thread::spawn(move || {
        let threads = 16usize;
        let per = addrs.len().div_ceil(threads);
        std::thread::scope(|sc| {
            for part in addrs.chunks(per) {
                sc.spawn(move || {
                    let mut sum = 0u8;
                    for &(a, n) in part {
                        let mut p = a;
                        while p < a + n { sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(p as *const u8) }); p = (p & !4095) + 4096; }
                    }
                    std::hint::black_box(sum);
                });
            }
        });
    });
}

/// Explicit reads of the engram tables for prefill-sized gathers (DS41_ENGRAM_IO). The mmap path faults every missing
/// page separately (~100k pages/s on a cold rank0 with the CPU at 60%; a 4096-token chunk touches ~390k), and every
/// path through the page cache competes with the engine for memory: rank0 has ~5.7 GB of it, a 30k-token cold prompt
/// reads ~5 GB, and once it is full reclaim slowed rank0's later chunks by 0.7-0.9 s each. `direct` (default): the
/// next chunk's rows are read in the background while this one computes (`prefetch_rows`): sorted by address, grouped
/// into page runs, read with O_DIRECT by a thread pool into small per-thread buffers and copied out (no page cache),
/// and handed over when the chunk asks for exactly those rows (`take`). Gathers nothing prefetched (a request's first
/// chunk) go through the mapping: warm pages are a plain copy there (a 98k-row O_DIRECT read is ~450 ms whether the
/// rows were read before or not). `pread`: the background reads through the page cache. `fadvise`: the next chunk's
/// pages read ahead by the kernel into the page cache. `mmap`: the original path. All return the same bytes.
pub mod io {
    use super::Checkpoint;
    use std::collections::HashMap;
    use std::fs::File;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    use std::sync::{atomic::{AtomicUsize, Ordering}, Arc, Mutex, OnceLock};
    use std::thread::JoinHandle;

    const PAGE: usize = 4096;
    /// Pages per read at most.
    const RUN: usize = 64;

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Mode { Mmap, Fadvise, Pread, Direct }
    pub fn mode() -> Mode {
        static M: OnceLock<Mode> = OnceLock::new();
        *M.get_or_init(|| match std::env::var("DS41_ENGRAM_IO").as_deref() {
            Ok("mmap") => Mode::Mmap, Ok("fadvise") => Mode::Fadvise, Ok("pread") => Mode::Pread, Ok("direct") | Err(_) => Mode::Direct,
            Ok(v) => panic!("DS41_ENGRAM_IO must be direct, pread, fadvise or mmap (got {v})"),
        })
    }
    /// Background reads of the next chunk's rows (pread / direct) rather than page-cache readahead.
    pub fn explicit() -> bool { matches!(mode(), Mode::Pread | Mode::Direct) }
    /// Gathers of at least this many rows are prefetched (DS41_ENGRAM_IO_MIN, 4096); decode's few rows keep the mmap
    /// path (page-cache hits, no thread pool round).
    pub fn min_rows() -> usize {
        static N: OnceLock<usize> = OnceLock::new();
        *N.get_or_init(|| std::env::var("DS41_ENGRAM_IO_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4096))
    }
    fn threads() -> usize {
        static N: OnceLock<usize> = OnceLock::new();
        *N.get_or_init(|| std::env::var("DS41_ENGRAM_IO_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(256usize).max(1))
    }

    /// One engram layer's weight [rows, head_dim] FP8 and scale [rows, head_dim / 32] tables in their file.
    pub struct Table { file: File, w0: usize, s0: usize }

    fn key(ck: &Checkpoint, layer: usize) -> (usize, usize) { (ck as *const Checkpoint as usize, layer) }

    /// The layer's tables (opened once per process); None in the mmap mode or when the tables are missing.
    pub fn table(ck: &Checkpoint, layer: usize) -> Option<Arc<Table>> {
        if mode() == Mode::Mmap { return None; }
        static T: OnceLock<Mutex<HashMap<(usize, usize), Arc<Table>>>> = OnceLock::new();
        let mut m = T.get_or_init(Default::default).lock().unwrap();
        if let Some(t) = m.get(&key(ck, layer)) { return Some(t.clone()); }
        let (wn, sn) = (format!("layers.{layer}.engram.embed.weight"), format!("layers.{layer}.engram.embed.scale"));
        if !ck.has(&wn) || !ck.has(&sn) { return None; }
        let ((wp, w0), (sp, s0)) = (ck.file_of(&wn), ck.file_of(&sn));
        assert_eq!(wp, sp, "engram weight and scale tables in different files");
        let mut o = std::fs::OpenOptions::new();
        o.read(true);
        if mode() == Mode::Direct { o.custom_flags(super::super::load::O_DIRECT); }
        let file = o.open(wp).unwrap_or_else(|e| panic!("open {}: {e}", wp.display()));
        let t = Arc::new(Table { file, w0, s0 });
        m.insert(key(ck, layer), t.clone());
        Some(t)
    }

    type Gathered = JoinHandle<(Vec<u8>, Vec<u8>)>;
    /// Background gathers by layer: (rows, handle), the newest last; a few per layer (this chunk's and the next).
    fn pending() -> &'static Mutex<HashMap<(usize, usize), Vec<(Vec<i64>, Gathered)>>> {
        static P: OnceLock<Mutex<HashMap<(usize, usize), Vec<(Vec<i64>, Gathered)>>>> = OnceLock::new();
        P.get_or_init(Default::default)
    }
    /// A background gather's completion, for the one queued after it (`prefetch_rows`).
    #[derive(Default)]
    struct Done { m: Mutex<bool>, c: std::sync::Condvar }
    impl Done {
        fn wait(&self) { let mut d = self.m.lock().unwrap(); while !*d { d = self.c.wait(d).unwrap(); } }
    }
    /// Sets `Done` when dropped (also when the gather panics: the next one is not left waiting).
    struct Signal(Arc<Done>);
    impl Drop for Signal { fn drop(&mut self) { *self.0.m.lock().unwrap() = true; self.0.c.notify_all(); } }

    /// Starts reading `rows` of the layer in the background (explicit modes, prefill-sized gathers only). The
    /// background gathers run one at a time in the order asked (each waits for the one before): a prompt's first chunk
    /// asks for layer 1's rows, then layer 14's, then the next chunk's, and layer 1 is needed first (after layer 0,
    /// ~50 ms at 4096 tokens). The device's random reads scale with the queue depth up to ~256 (`threads`: ~1.15M 4 KiB
    /// reads/s on every node, 64 gave ~0.4M); two gathers at once split it, so layer 1's rows came at the time both
    /// were done (~80 ms each at 256 threads, a 60-90 ms stall in layer 1 of every 4k request), alone at ~56 ms.
    pub fn prefetch_rows(ck: &Checkpoint, layer: usize, rows: Vec<i64>, head_dim: i64) {
        if !explicit() || rows.len() < min_rows() { return; }
        let Some(t) = table(ck, layer) else { return };
        static LANE: Mutex<Option<Arc<Done>>> = Mutex::new(None);
        let me = Arc::new(Done::default());
        let prev = LANE.lock().unwrap().replace(me.clone());
        let r = rows.clone();
        let h = std::thread::spawn(move || {
            let _s = Signal(me);
            if let Some(p) = prev { p.wait(); }
            t.gather(&r, head_dim)
        });
        let mut p = pending().lock().unwrap();
        let v = p.entry(key(ck, layer)).or_default();
        v.push((rows, h));
        if v.len() > 3 { v.remove(0); }   // never asked for (an aborted request): its thread finishes, the result drops
    }
    /// The background gather of exactly these rows, if one was started.
    pub fn take(ck: &Checkpoint, layer: usize, rows: &[i64]) -> Option<Gathered> {
        let mut p = pending().lock().unwrap();
        let hit = p.get_mut(&key(ck, layer)).and_then(|v| v.iter().position(|(r, _)| r.as_slice() == rows).map(|i| v.remove(i).1));
        if rows.len() >= min_rows() && std::env::var("DS41_ENGRAM_STATS").is_ok_and(|v| v == "1") {
            eprintln!("[ds41-engram] layer {layer} gather of {} rows: {}", rows.len(), if hit.is_some() { "prefetched" } else { "not prefetched" });
        }
        hit
    }

    /// A page-aligned buffer (O_DIRECT needs aligned memory).
    struct Buf { p: *mut u8, n: usize }
    impl Buf {
        fn new(n: usize) -> Buf {
            let p = unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(n, PAGE).unwrap()) };
            assert!(!p.is_null(), "engram read buffer");
            Buf { p, n }
        }
        fn slice(&mut self, n: usize) -> &mut [u8] { assert!(n <= self.n); unsafe { std::slice::from_raw_parts_mut(self.p, n) } }
    }
    impl Drop for Buf { fn drop(&mut self) { unsafe { std::alloc::dealloc(self.p, std::alloc::Layout::from_size_align(self.n, PAGE).unwrap()) } } }
    /// The output rows, written by the pool's threads at disjoint offsets.
    #[derive(Clone, Copy)]
    struct Out(*mut u8);
    unsafe impl Send for Out {}
    unsafe impl Sync for Out {}
    impl Out { fn ptr(self) -> *mut u8 { self.0 } }

    /// A row piece: file address, length, destination (true: scale buffer) and its offset there.
    #[derive(Clone, Copy)]
    struct Item { addr: usize, len: u32, scale: bool, dst: usize }

    impl Table {
        /// The rows' weight and scale pieces sorted by file address, and the reads covering them: (first page, pages,
        /// items lo..hi), each item inside one read, consecutive pages merged up to RUN pages.
        fn plan(&self, rows: &[i64], head_dim: i64) -> (Vec<Item>, Vec<(usize, usize, usize, usize)>) {
            let (wb, sb) = (head_dim as usize, head_dim as usize / 32);
            let mut items = Vec::with_capacity(rows.len() * 2);
            for (i, &r) in rows.iter().enumerate() {
                let r = r as usize;
                items.push(Item { addr: self.w0 + r * wb, len: wb as u32, scale: false, dst: i * wb });
                items.push(Item { addr: self.s0 + r * sb, len: sb as u32, scale: true, dst: i * sb });
            }
            items.sort_unstable_by_key(|x| x.addr);
            let mut runs: Vec<(usize, usize, usize, usize)> = Vec::new();
            for (k, it) in items.iter().enumerate() {
                let (p0, p1) = (it.addr / PAGE, (it.addr + it.len as usize - 1) / PAGE + 1);
                if let Some(r) = runs.last_mut() {
                    let end = r.0 + r.1;
                    if p0 <= end && p1.max(end) - r.0 <= RUN { r.1 = p1.max(end) - r.0; r.3 = k + 1; continue; }
                }
                runs.push((p0, p1 - p0, k, k + 1));
            }
            (items, runs)
        }

        /// The rows' weight bytes [n * head_dim] and scale bytes [n * head_dim / 32], as `super::gather`.
        pub fn gather(&self, rows: &[i64], head_dim: i64) -> (Vec<u8>, Vec<u8>) {
            let t0 = std::time::Instant::now();
            let (items, runs) = self.plan(rows, head_dim);
            let (wb, sb) = (head_dim as usize, head_dim as usize / 32);
            let mut wbuf = super::buf(rows.len() * wb);
            let mut sbuf = super::buf(rows.len() * sb);
            let (wo, so) = (Out(wbuf.as_mut_ptr()), Out(sbuf.as_mut_ptr()));
            let next = AtomicUsize::new(0);
            std::thread::scope(|sc| {
                for _ in 0..threads().min(runs.len()) {
                    let (items, runs, next) = (&items, &runs, &next);
                    sc.spawn(move || {
                        let mut buf = Buf::new(RUN * PAGE);
                        loop {
                            let k = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(p0, n, lo, hi)) = runs.get(k) else { break };
                            let b = buf.slice(n * PAGE);
                            let mut got = 0usize;
                            while got < b.len() {
                                let m = self.file.read_at(&mut b[got..], (p0 * PAGE + got) as u64).expect("engram table read");
                                if m == 0 { break; }   // the file's end (its last page is partial)
                                got += m;
                            }
                            for it in &items[lo..hi] {
                                let a = it.addr - p0 * PAGE;
                                let dst = if it.scale { so.ptr() } else { wo.ptr() };
                                // SAFETY: every item has its own destination range inside the output buffers
                                unsafe { std::ptr::copy_nonoverlapping(b.as_ptr().add(a), dst.add(it.dst), it.len as usize) };
                            }
                        }
                    });
                }
            });
            super::stats(rows.len(), t0);
            (wbuf, sbuf)
        }

        /// Asynchronous readahead of the rows' pages into the page cache (posix_fadvise WILLNEED per read).
        pub fn readahead(&self, rows: &[i64], head_dim: i64) {
            extern "C" { fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32; }
            const POSIX_FADV_WILLNEED: i32 = 3;
            use std::os::unix::io::AsRawFd;
            for (p0, n, _, _) in self.plan(rows, head_dim).1 {
                unsafe { posix_fadvise(self.file.as_raw_fd(), (p0 * PAGE) as i64, (n * PAGE) as i64, POSIX_FADV_WILLNEED); }
            }
        }
    }
}
