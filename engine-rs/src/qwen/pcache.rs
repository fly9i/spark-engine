// SPDX-License-Identifier: MIT
//! Persistent prefix cache for Qwen3.8-Flash-Next on the local NVMe (QWEN_PCACHE=1, serving only). Mirrors the GLM
//! engine's prefix cache (src/pcache.rs) — same O_DIRECT pinned double-buffered writer, per-chunk checksums, JSON
//! tensor-table header, streaming prefix hash, LRU byte-budget index, binary+model+numerics identity — but single
//! node (no TP replication, no segments): one `.snap` file holds a whole prompt checkpoint (token ids + the flat
//! state tensor list `ckpt_save` produces, ~100-120 MB). A background thread writes every boundary checkpoint the
//! serve loop takes; a later request whose prompt extends a cached prefix restores it instead of re-prefilling it,
//! even after the in-memory store holding it was reused.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{mpsc, Arc};
use std::cell::RefCell;
use std::time::Instant;
use serde_json::{json, Value};
use tch::{Device, Kind, Tensor};

const FORMAT: u64 = 1;
const ALIGN: usize = 4096;
const CHUNK: usize = 16 << 20;
#[cfg(target_arch = "aarch64")] const O_DIRECT: i32 = 0o200000;
#[cfg(not(target_arch = "aarch64"))] const O_DIRECT: i32 = 0o40000;
const PENDING: u8 = 0; const OK: u8 = 1; const FAILED: u8 = 2; const CANCELLED: u8 = 3;

extern "C" {
    fn cudaSetDevice(d: i32) -> i32;
    fn cudaStreamCreateWithFlags(s: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaStreamSynchronize(s: *mut std::ffi::c_void) -> i32;
    fn cudaStreamWaitEvent(s: *mut std::ffi::c_void, e: *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaEventCreateWithFlags(e: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaEventRecord(e: *mut std::ffi::c_void, s: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(e: *mut std::ffi::c_void) -> i32;
    fn cudaEventDestroy(e: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpyAsync(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, n: usize, kind: i32, s: *mut std::ffi::c_void) -> i32;
    fn cudaHostRegister(p: *mut std::ffi::c_void, n: usize, flags: u32) -> i32;
    fn rs_current_stream() -> *mut std::ffi::c_void;
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut std::ffi::c_void;
    fn statvfs(path: *const std::ffi::c_char, buf: *mut u64) -> i32;
}
const D2H: i32 = 2; const H2D: i32 = 1;

pub(crate) fn requested() -> bool { std::env::var("QWEN_PCACHE").as_deref() == Ok("1") }
pub(crate) fn enabled() -> bool { PC.with(|p| p.borrow().is_some()) }
fn env_u64(k: &str, d: u64) -> u64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
fn pad(n: usize) -> usize { n.div_ceil(ALIGN) * ALIGN }
fn log() -> bool { std::env::var("QWEN_PCACHE_LOG").as_deref() == Ok("1") }

// ---- hashing / checksum (identical to the GLM cache) ----
fn mix(mut x: u64) -> u64 { x ^= x >> 30; x = x.wrapping_mul(0xbf58476d1ce4e5b9); x ^= x >> 27; x = x.wrapping_mul(0x94d049bb133111eb); x ^ x >> 31 }
#[derive(Clone, Copy)] struct PrefixHash { a: u64, b: u64, n: u64 }
impl PrefixHash {
    fn new() -> Self { Self { a: 0x243f6a8885a308d3, b: 0x13198a2e03707344, n: 0 } }
    fn push(&mut self, t: i64) {
        let m = mix(t as u64 ^ self.n.wrapping_mul(0x9e3779b97f4a7c15));
        self.a = (self.a ^ m).wrapping_mul(0xff51afd7ed558ccd).rotate_left(27);
        self.b = self.b.wrapping_add(mix(m ^ 0xa4093822299f31d0)).wrapping_mul(0xc4ceb9fe1a85ec53).rotate_left(31); self.n += 1;
    }
    fn digest(&self) -> u128 { ((mix(self.a ^ self.n) as u128) << 64) | mix(self.b ^ self.n.rotate_left(32)) as u128 }
}
fn hash_ids(ids: &[i64]) -> u128 { let mut h = PrefixHash::new(); for &t in ids { h.push(t); } h.digest() }
fn checksum(b: &[u8]) -> u64 {
    let mut s = 0x9e3779b97f4a7c15u64;
    for w in b.chunks_exact(8) { s = (s.rotate_left(5) ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x100000001b3); }
    mix(s ^ b.len() as u64)
}
fn kind_name(k: Kind) -> &'static str { match k { Kind::Uint8 => "u8", Kind::Int8 => "i8", Kind::Int => "i32", Kind::Int64 => "i64", Kind::Half => "f16", Kind::Float => "f32", Kind::BFloat16 => "bf16", Kind::Bool => "bool", k => panic!("prefix cache: unsupported kind {k:?}") } }
fn kind_of(s: &str) -> Option<Kind> { Some(match s { "u8" => Kind::Uint8, "i8" => Kind::Int8, "i32" => Kind::Int, "i64" => Kind::Int64, "f16" => Kind::Half, "f32" => Kind::Float, "bf16" => Kind::BFloat16, "bool" => Kind::Bool, _ => return None }) }
fn nbytes(t: &Tensor) -> usize { t.numel() * t.kind().elt_size_in_bytes() }

// ---- pinned staging ----
struct Pinned { ptr: *mut u8, len: usize }
unsafe impl Send for Pinned {} unsafe impl Sync for Pinned {}
impl Pinned {
    fn new(len: usize) -> Self {
        const PROT_RW: i32 = 3; const MAP_PRIVATE_ANON_POPULATE: i32 = 0x02 | 0x20 | 0x8000;
        let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_RW, MAP_PRIVATE_ANON_POPULATE, -1, 0) };
        assert!(p as isize != -1, "prefix cache: mmap {len}");
        assert_eq!(unsafe { cudaHostRegister(p, len, 1) }, 0, "prefix cache: cudaHostRegister");
        Self { ptr: p.cast(), len }
    }
    fn slice(&self, n: usize) -> &mut [u8] { assert!(n <= self.len); unsafe { std::slice::from_raw_parts_mut(self.ptr, n) } }
}
fn stream() -> usize { let mut s = std::ptr::null_mut(); assert_eq!(unsafe { cudaStreamCreateWithFlags(&mut s, 1) }, 0, "prefix cache stream"); s as usize }

// ---- file format ----
#[derive(Clone)] struct Region { name: String, kind: Kind, shape: Vec<i64>, off: u64, bytes: usize }
fn table(regions: &[Region], sums: &[Vec<u64>]) -> Value {
    Value::Array(regions.iter().zip(sums).map(|(r, s)| json!({"name": r.name, "kind": kind_name(r.kind), "shape": r.shape, "off": r.off, "bytes": r.bytes,
        "sums": s.iter().map(|v| format!("{v:016x}")).collect::<Vec<_>>()})).collect())
}
fn layout(regions: &mut [Region], fixed: &Value) -> (usize, u64) {
    let dummy: Vec<Vec<u64>> = regions.iter().map(|r| vec![0; r.bytes.div_ceil(CHUNK)]).collect();
    for r in regions.iter_mut() { r.off = u64::MAX / 2; }
    let mut probe = fixed.clone(); probe["tensors"] = table(regions, &dummy);
    let head = pad(8 + serde_json::to_vec(&probe).unwrap().len() + 64);
    let mut off = head as u64; for r in regions.iter_mut() { r.off = off; off += pad(r.bytes) as u64; }
    (head, off)
}
fn read_header(path: &Path) -> Option<Value> {
    use std::io::Read; let mut f = std::fs::File::open(path).ok()?;
    let mut n = [0u8; 8]; f.read_exact(&mut n).ok()?; let n = u64::from_le_bytes(n) as usize;
    if n > 64 << 20 { return None; }
    let mut b = vec![0u8; n]; f.read_exact(&mut b).ok()?; serde_json::from_slice(&b).ok()
}
fn regions_of(h: &Value) -> Option<Vec<(Region, Vec<u64>)>> {
    h["tensors"].as_array()?.iter().map(|t| Some((Region { name: t["name"].as_str()?.to_string(), kind: kind_of(t["kind"].as_str()?)?,
        shape: t["shape"].as_array()?.iter().map(|v| v.as_i64()).collect::<Option<Vec<_>>>()?, off: t["off"].as_u64()?, bytes: t["bytes"].as_u64()? as usize },
        t["sums"].as_array()?.iter().map(|v| u64::from_str_radix(v.as_str()?, 16).ok()).collect::<Option<Vec<_>>>()?))).collect()
}

// ---- writer thread (D2H of device tensors, O_DIRECT, double buffered, tmp+rename) ----
struct FilePlan { path: PathBuf, head: usize, size: u64, fixed: Value, regions: Vec<Region>, srcs: Vec<usize>, state: Arc<AtomicU8>, cancel: Arc<AtomicBool> }
enum Job { Write { event: usize, f: FilePlan, done: Arc<AtomicBool>, keep: Vec<Tensor> }, Delete(PathBuf), Touch(PathBuf), Flush(mpsc::Sender<()>) }
struct Writer { stream: usize, bufs: [Pinned; 2], dir: PathBuf, margin: u64, events: [usize; 2] }
impl Writer {
    fn run(self, rx: mpsc::Receiver<Job>, back: mpsc::Sender<Vec<Tensor>>) {
        unsafe { cudaSetDevice(0); }
        for job in rx { match job {
            Job::Write { event, f, done, keep } => {
                let waited = unsafe { cudaStreamWaitEvent(self.stream as _, event as _, 0) } == 0;
                if f.cancel.load(Ordering::Acquire) { f.state.store(CANCELLED, Ordering::Release); }
                else if !waited { eprintln!("[qpcache] {}: cudaStreamWaitEvent failed", f.path.display()); f.state.store(FAILED, Ordering::Release); }
                else { let t = Instant::now(); match self.write(&f) {
                    Ok(()) => { f.state.store(OK, Ordering::Release); if let Ok(d) = std::fs::File::open(&self.dir) { let _ = d.sync_all(); }
                        if log() { eprintln!("[qpcache] wrote {} ({:.1} MiB) in {:.1} ms", f.path.display(), f.size as f64 / 1048576., t.elapsed().as_secs_f64() * 1e3); } }
                    Err(e) => { eprintln!("[qpcache] write {} failed: {e}", f.path.display()); let _ = std::fs::remove_file(f.path.with_extension("tmp")); f.state.store(FAILED, Ordering::Release); }
                } }
                unsafe { cudaEventDestroy(event as _); }
                done.store(true, Ordering::Release); let _ = back.send(keep);
            }
            Job::Delete(p) => { let _ = std::fs::remove_file(&p); let _ = std::fs::remove_file(p.with_extension("tmp")); }
            Job::Touch(p) => { if let Ok(f) = std::fs::File::options().write(true).open(&p) { let _ = f.set_modified(std::time::SystemTime::now()); } }
            Job::Flush(tx) => { let _ = tx.send(()); }
        } }
    }
    fn write(&self, f: &FilePlan) -> std::io::Result<()> {
        use std::os::unix::fs::{FileExt, OpenOptionsExt};
        let mut sv = [0u64; 16]; let c = std::ffi::CString::new(self.dir.as_os_str().as_encoded_bytes()).unwrap();
        if unsafe { statvfs(c.as_ptr(), sv.as_mut_ptr()) } == 0 && sv[1] * sv[4] < f.size + self.margin { return Err(std::io::Error::other("disk headroom below QWEN_PCACHE_FREE_GIB")); }
        let tmp = f.path.with_extension("tmp");
        let file = std::fs::OpenOptions::new().write(true).create(true).truncate(true).custom_flags(O_DIRECT).open(&tmp)?;
        let chunks: Vec<(usize, usize, usize)> = f.regions.iter().enumerate().flat_map(|(i, r)| (0..r.bytes.div_ceil(CHUNK)).map(move |k| (i, k * CHUNK, (r.bytes - k * CHUNK).min(CHUNK)))).collect();
        let cuda = |rc: i32, what: &str| if rc == 0 { Ok(()) } else { Err(std::io::Error::other(format!("{what}: CUDA error {rc}"))) };
        let mut sums: Vec<Vec<u64>> = f.regions.iter().map(|_| Vec::new()).collect();
        let issue = |k: usize| -> std::io::Result<()> { let (i, o, n) = chunks[k]; let b = &self.bufs[k % 2];
            cuda(unsafe { cudaMemcpyAsync(b.ptr.cast(), (f.srcs[i] + o) as *const std::ffi::c_void, n, D2H, self.stream as _) }, "D2H")?;
            cuda(unsafe { cudaEventRecord(self.events[k % 2] as _, self.stream as _) }, "event record") };
        if !chunks.is_empty() { issue(0)?; }
        for k in 0..chunks.len() {
            cuda(unsafe { cudaEventSynchronize(self.events[k % 2] as _) }, "D2H sync")?;
            let (i, o, n) = chunks[k]; let b = &self.bufs[k % 2]; let len = pad(n);
            b.slice(len)[n..].fill(0); let buf = b.slice(len); sums[i].push(checksum(buf));
            if k + 1 < chunks.len() { issue(k + 1)?; }
            file.write_all_at(buf, f.regions[i].off + o as u64)?;
        }
        let mut h = f.fixed.clone(); h["tensors"] = table(&f.regions, &sums);
        let js = serde_json::to_vec(&h).unwrap();
        if 8 + js.len() > f.head { return Err(std::io::Error::other("header larger than reserved")); }
        let hb = self.bufs[0].slice(f.head); hb.fill(0); hb[..8].copy_from_slice(&(js.len() as u64).to_le_bytes()); hb[8..8 + js.len()].copy_from_slice(&js);
        file.write_all_at(hb, 0)?; file.sync_data()?; drop(file);
        std::fs::rename(&tmp, &f.path)?; Ok(())
    }
}

// ---- index ----
struct Entry { key: u128, len: usize, bytes: u64, used: u64, path: PathBuf, state: Arc<AtomicU8>, cancel: Arc<AtomicBool> }
struct Pending { tag: u64, done: Arc<AtomicBool> }
#[derive(Default)] struct Stats { saves: u64, skipped_dup: u64, skipped_budget: u64, evicted: u64, restores: u64, restore_fail: u64, written_bytes: u64 }
struct Pc {
    dir: PathBuf, identity: String, cap: u64, min: usize, margin: u64,
    entries: Vec<Entry>, clock: u64, total: u64,
    read: [Pinned; 2], rstream: usize, revents: [usize; 2],
    tx: mpsc::Sender<Job>, back: mpsc::Receiver<Vec<Tensor>>, pending: Vec<Pending>, stats: Stats,
}
thread_local! { static PC: RefCell<Option<Pc>> = const { RefCell::new(None) }; }

fn identity(binary: &str, model: &Path) -> String {
    let skip = ["QWEN_SERVE_", "QWEN_PCACHE", "QWEN_KV_TOKENS", "QWEN_MEMGUARD", "QWEN_ASSETS", "QWEN_REQLOG", "QWEN_TIMING",
                "QWEN_KEEP_SHARD_CACHE", "QWEN_NGRAM_CACHE_MB", "QWEN_NGRAM_HOT", "QWEN_PLE_SERIAL", "QWEN_BATCH", "QWEN_TAIL", "QWEN_STATE_DUMP", "QWEN_DUMP"];
    let mut env: Vec<(String, String)> = std::env::vars().filter(|(k, _)| k.starts_with("QWEN_") && !skip.iter().any(|s| k.starts_with(s))).collect(); env.sort();
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).display().to_string();
    format!("format={FORMAT};binary={binary};model={};env={env:?}", canon(model))
}
fn short(s: &str) -> String { use sha2::Digest; sha2::Sha256::digest(s.as_bytes())[..8].iter().map(|b| format!("{b:02x}")).collect() }
fn mtime(p: &Path) -> u64 { std::fs::metadata(p).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as u64) }

/// Serve startup (QWEN_PCACHE=1): scan the identity's directory, load the valid `.snap` index in LRU order, start the writer.
pub(crate) fn init(model: &Path) {
    let t0 = Instant::now();
    let root = PathBuf::from(std::env::var("QWEN_PCACHE_DIR").unwrap_or_else(|_| "/tmp/qwen38-prefix-cache".into()));
    let cap = env_u64("QWEN_PCACHE_GIB", 100) << 30; let min = env_u64("QWEN_PCACHE_MIN", 1024) as usize;
    let binary = std::fs::read(std::env::current_exe().unwrap()).map(|b| { use sha2::Digest; sha2::Sha256::digest(&b).iter().map(|x| format!("{x:02x}")).collect::<String>() }).expect("prefix cache: read the engine binary");
    let identity = identity(&binary, model); let dir = root.join(short(&identity));
    std::fs::create_dir_all(&dir).expect("prefix cache directory");
    let _ = std::fs::write(dir.join("identity.txt"), &identity);
    let mut found: Vec<(u128, usize, u64, PathBuf, u64)> = Vec::new();   // key, len, bytes, path, mtime
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p = e.path(); let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.ends_with(".tmp") { let _ = std::fs::remove_file(&p); continue; }
        if !name.ends_with(".snap") { continue; }
        match read_header(&p).filter(|h| h["identity"].as_str() == Some(identity.as_str()) && h["format"].as_u64() == Some(FORMAT)) {
            Some(h) => found.push((u128::from_str_radix(h["key"].as_str().unwrap_or("0"), 16).unwrap_or(0), h["len"].as_u64().unwrap_or(0) as usize,
                                   e.metadata().map(|m| m.len()).unwrap_or(0), p.clone(), mtime(&p))),
            None => { let _ = std::fs::remove_file(&p); }
        }
    }
    found.sort_by_key(|x| x.4);
    let (tx, rx) = mpsc::channel(); let (btx, brx) = mpsc::channel();
    let margin = env_u64("QWEN_PCACHE_FREE_GIB", 20) << 30;
    let ev = || { let mut e = std::ptr::null_mut(); assert_eq!(unsafe { cudaEventCreateWithFlags(&mut e, 2) }, 0); e as usize };
    let writer = Writer { stream: stream(), bufs: [Pinned::new(CHUNK), Pinned::new(CHUNK)], dir: dir.clone(), margin, events: [ev(), ev()] };
    std::thread::Builder::new().name("qpcache-writer".into()).spawn(move || writer.run(rx, btx)).unwrap();
    let mut pc = Pc { dir: dir.clone(), identity, cap, min, margin, entries: Vec::new(), clock: 0, total: 0,
        read: [Pinned::new(CHUNK), Pinned::new(CHUNK)], rstream: stream(), revents: [ev(), ev()],
        tx, back: brx, pending: Vec::new(), stats: Stats::default() };
    for (key, len, bytes, path, _) in found { pc.clock += 1; pc.total += bytes;
        pc.entries.push(Entry { key, len, bytes, used: pc.clock, path, state: Arc::new(AtomicU8::new(OK)), cancel: Arc::new(AtomicBool::new(false)) }); }
    let before = pc.entries.len(); pc.evict_to(0);
    eprintln!("[qpcache] {} entries ({} evicted to fit), {:.2} GiB of {:.1} GiB at {} ({:.2}s)", pc.entries.len(), before - pc.entries.len(),
        pc.total as f64 / (1u64 << 30) as f64, pc.cap as f64 / (1u64 << 30) as f64, dir.display(), t0.elapsed().as_secs_f64());
    PC.with(|p| *p.borrow_mut() = Some(pc));
}

impl Pc {
    fn evict_to(&mut self, extra: u64) {
        while self.total + extra > self.cap {
            let Some(i) = self.entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i) else { break };
            let e = self.entries.remove(i); e.cancel.store(true, Ordering::Release); self.total -= e.bytes;
            let _ = self.tx.send(Job::Delete(e.path)); self.stats.evicted += 1;
        }
    }
    fn reap(&mut self) { while self.back.try_recv().is_ok() {} self.pending.retain(|p| !p.done.load(Ordering::Acquire)); }
}

pub(crate) fn poll() { PC.with(|p| if let Some(pc) = p.borrow_mut().as_mut() { pc.reap(); }); }
/// Wait until no background write still reads store `tag`'s checkpoint tensors (before they are overwritten).
pub(crate) fn fence(tag: u64) {
    PC.with(|p| if let Some(pc) = p.borrow_mut().as_mut() {
        pc.reap(); if !pc.pending.iter().any(|j| j.tag == tag) { return; }
        while pc.pending.iter().any(|j| j.tag == tag) {
            if let Err(mpsc::RecvTimeoutError::Disconnected) = pc.back.recv_timeout(std::time::Duration::from_millis(1)).map(drop) { pc.pending.clear(); break; }
            pc.pending.retain(|p| !p.done.load(Ordering::Acquire));
        }
    });
}
pub(crate) fn shutdown(secs: u64) {
    PC.with(|p| if let Some(pc) = p.borrow_mut().as_mut() {
        let (tx, rx) = mpsc::channel(); let _ = pc.tx.send(Job::Flush(tx));
        let ok = rx.recv_timeout(std::time::Duration::from_secs(secs)).is_ok(); pc.reap();
        eprintln!("[qpcache] shutdown: writes {}", if ok { "flushed" } else { "still pending (timeout)" });
    });
}
pub(crate) fn stats() -> Value {
    PC.with(|p| p.borrow().as_ref().map_or(Value::Null, |pc| { let s = &pc.stats; json!({"entries": pc.entries.len(), "bytes": pc.total, "cap": pc.cap,
        "pending_writes": pc.pending.len(), "saves": s.saves, "skipped_dup": s.skipped_dup, "skipped_budget": s.skipped_budget, "evicted": s.evicted,
        "restores": s.restores, "restore_failures": s.restore_fail, "written_bytes": s.written_bytes}) }))
}

/// Save checkpoint of prefix `ids` with its state `tensors` (right after ckpt_save). `tag`: the store (see fence).
pub(crate) fn save(ids: &[i64], tensors: &[Tensor], tag: u64) {
    PC.with(|p| { let mut b = p.borrow_mut(); let Some(pc) = b.as_mut() else { return }; pc.reap();
        if ids.len() < pc.min { return; }
        let key = hash_ids(ids); pc.clock += 1; let clock = pc.clock;
        if let Some(e) = pc.entries.iter_mut().find(|e| e.key == key && e.len == ids.len()) { e.used = clock; pc.stats.skipped_dup += 1; let _ = pc.tx.send(Job::Touch(e.path.clone())); return; }
        let mut regions = Vec::new(); let mut srcs = Vec::new(); let mut keep = Vec::new();
        let idb: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
        // ids first (Host bytes staged through a kept tensor so every region is a device source)
        let idt = Tensor::from_slice(&ids.iter().map(|&x| x).collect::<Vec<i64>>()).to_device(Device::Cuda(0));
        regions.push(Region { name: "ids".into(), kind: Kind::Int64, shape: vec![ids.len() as i64], off: 0, bytes: idb.len() }); srcs.push(idt.data_ptr() as usize); keep.push(idt);
        for (i, t) in tensors.iter().enumerate() {
            let t = if t.is_contiguous() { t.shallow_clone() } else { t.contiguous() };
            regions.push(Region { name: format!("t{i}"), kind: t.kind(), shape: t.size(), off: 0, bytes: nbytes(&t) }); srcs.push(t.data_ptr() as usize); keep.push(t);
        }
        let fixed = json!({"format": FORMAT, "identity": pc.identity, "key": format!("{key:032x}"), "len": ids.len()});
        let (head, size) = layout(&mut regions, &fixed); let _ = head;
        if size > pc.cap { pc.stats.skipped_budget += 1; return; }
        pc.clock += 1; let used = pc.clock;
        let path = pc.dir.join(format!("e-{key:032x}.snap"));
        let state = Arc::new(AtomicU8::new(PENDING)); let cancel = Arc::new(AtomicBool::new(false));
        pc.entries.push(Entry { key, len: ids.len(), bytes: size, used, path: path.clone(), state: state.clone(), cancel: cancel.clone() });
        pc.total += size; pc.evict_to(0);
        let (head, _) = layout(&mut regions, &fixed);
        let f = FilePlan { path, head, size, fixed, regions, srcs, state, cancel };
        let mut event = std::ptr::null_mut(); assert_eq!(unsafe { cudaEventCreateWithFlags(&mut event, 2) }, 0);
        assert_eq!(unsafe { cudaEventRecord(event, rs_current_stream()) }, 0, "prefix cache: record");
        let done = Arc::new(AtomicBool::new(false)); pc.pending.push(Pending { tag, done: done.clone() });
        pc.stats.saves += 1; pc.stats.written_bytes += size;
        let _ = pc.tx.send(Job::Write { event: event as usize, f, done, keep });
        if log() { eprintln!("[qpcache] save {} tokens: {:.1} MiB, total {:.2} GiB, {} entries", ids.len(), size as f64 / 1048576., pc.total as f64 / (1u64 << 30) as f64, pc.entries.len()); }
    });
}
/// The longest finished entry that is a (<=) prefix of `q`, at least `min_len`, longer than `longer_than`. (id index, len).
pub(crate) fn lookup(q: &[i64], longer_than: usize, min_len: usize) -> Option<(usize, usize)> {
    PC.with(|p| { let b = p.borrow(); let pc = b.as_ref()?;
        let mut cand: Vec<(usize, &Entry)> = pc.entries.iter().enumerate().filter(|(_, e)| e.len <= q.len() && e.len > longer_than && e.len >= min_len && e.state.load(Ordering::Acquire) == OK).collect();
        if cand.is_empty() { return None; }
        cand.sort_by_key(|(_, e)| e.len);
        let mut h = PrefixHash::new(); let mut at = 0usize; let mut best: Option<(usize, usize)> = None;
        for (i, e) in cand { while at < e.len { h.push(q[at]); at += 1; } if h.digest() == e.key { best = Some((i, e.len)); } }
        best
    })
}
/// Read entry `idx` from disk into fresh device tensors: (ids, state tensors in save order). Verifies every chunk's
/// checksum and the token ids against `prompt`'s prefix; returns None (and drops the entry) on any mismatch.
pub(crate) fn restore(idx: usize, prompt: &[i64], dev: Device) -> Option<(Vec<i64>, Vec<Tensor>)> {
    PC.with(|p| {
        let t0 = Instant::now();
        let (path, key, len, state) = { let b = p.borrow(); let pc = b.as_ref()?; let e = pc.entries.get(idx)?; (e.path.clone(), e.key, e.len, e.state.clone()) };
        if state.load(Ordering::Acquire) != OK { return None; }
        let fail = |p: &RefCell<Option<Pc>>| { if let Some(pc) = p.borrow_mut().as_mut() { pc.stats.restore_fail += 1; if let Some(i) = pc.entries.iter().position(|e| e.path == path) { let e = pc.entries.remove(i); pc.total -= e.bytes; let _ = pc.tx.send(Job::Delete(e.path)); } } None::<(Vec<i64>, Vec<Tensor>)> };
        let h = match read_header(&path) { Some(h) => h, None => return fail(p) };
        let regions = match regions_of(&h) { Some(r) => r, None => return fail(p) };
        use std::os::unix::fs::{FileExt, OpenOptionsExt};
        let file = match std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(&path) { Ok(f) => f, Err(_) => return fail(p) };
        let mut ids: Vec<i64> = Vec::new(); let mut out: Vec<Tensor> = Vec::new();
        let b = p.borrow(); let pc = b.as_ref().unwrap(); let (rs, rev) = (pc.rstream, pc.revents);
        for (r, sums) in &regions {
            let dst = if r.name == "ids" { None } else { Some(Tensor::empty(&r.shape, (r.kind, dev))) };
            let dptr = dst.as_ref().map(|t| t.data_ptr() as usize);
            let mut idbytes: Vec<u8> = Vec::new();
            let nch = r.bytes.div_ceil(CHUNK);
            for k in 0..nch {
                let o = k * CHUNK; let n = (r.bytes - o).min(CHUNK); let len_p = pad(n); let buf = pc.read[k % 2].slice(len_p);
                if file.read_exact_at(buf, r.off + o as u64).is_err() { drop(b); return fail(p); }
                if checksum(buf) != sums[k] { drop(b); return fail(p); }
                match dptr { Some(dp) => { if unsafe { cudaMemcpyAsync((dp + o) as *mut std::ffi::c_void, buf.as_ptr() as *const _, n, H2D, rs as _) } != 0 { drop(b); return fail(p); }
                                            unsafe { cudaEventRecord(rev[k % 2] as _, rs as _); cudaEventSynchronize(rev[k % 2] as _); } }
                             None => idbytes.extend_from_slice(&buf[..n]) }
            }
            match dst { Some(t) => out.push(t), None => ids = idbytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect() }
        }
        drop(b);
        if ids.len() != len || hash_ids(&ids) != key || prompt.len() < len || prompt[..len] != ids[..] { return fail(p); }
        unsafe { cudaStreamSynchronize(rs as _); }
        PC.with(|p| if let Some(pc) = p.borrow_mut().as_mut() { pc.stats.restores += 1; pc.clock += 1; if let Some(e) = pc.entries.get_mut(idx) { e.used = pc.clock; } });
        if log() { eprintln!("[qpcache] restore {len} tokens in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3); }
        Some((ids, out))
    })
}
