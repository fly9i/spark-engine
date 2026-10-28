//! Checkpoint read-ahead shared by the model loaders: byte ranges of shard files read in order by one thread with
//! O_DIRECT into host buffers on transparent huge pages, which the GPU reads in place (GB10: CPU and GPU share the
//! memory), and a scatter copy of a range's tensors to their places on the device in one launch.
//! Page-cache reads (mmap faults, buffered pread) ran at ~0.5..1.5 GB/s here, O_DIRECT at ~10..12 GB/s; a
//! cudaMemcpyAsync from pinned memory holds the calling thread for the copy (~34 GB/s), a kernel reading the host
//! buffer does not.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

pub const ALIGN: usize = 4096;
#[cfg(target_arch = "aarch64")] pub const O_DIRECT: i32 = 0o200000;
#[cfg(not(target_arch = "aarch64"))] pub const O_DIRECT: i32 = 0o40000;

extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    fn cudaEventCreateWithFlags(e: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaEventRecord(e: *mut std::ffi::c_void, st: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(e: *mut std::ffi::c_void) -> i32;
    fn cudaEventDestroy(e: *mut std::ffi::c_void) -> i32;
    fn rs_current_stream() -> *mut std::ffi::c_void;
    fn spark_copy_segments(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, seg: *const i64, n: i32, chunks: i64, st: *mut std::ffi::c_void) -> i32;
}

/// Anonymous memory on transparent huge pages (the GPU reads it in place, GB10's pageable memory access at ~95 GB/s;
/// pinning 2.8 GB with cudaHostAlloc took 1.5 s and freeing it 0.7 s, faulting it in by 4 KiB pages ~0.7 s, by 2 MiB
/// pages ~0.04 s).
pub struct PageBuf { p: *mut u8, n: usize }
unsafe impl Send for PageBuf {}
impl PageBuf {
    pub fn new(n: usize) -> PageBuf {
        const PROT_RW: i32 = 3; const MAP_PRIVATE_ANON: i32 = 0x22; const MADV_HUGEPAGE: i32 = 14;
        let p = unsafe { mmap(std::ptr::null_mut(), n, PROT_RW, MAP_PRIVATE_ANON, -1, 0) };
        assert!(!p.is_null() && p as isize != -1, "mmap {n} B");
        unsafe { madvise(p, n, MADV_HUGEPAGE); }
        PageBuf { p: p as *mut u8, n }
    }
    pub fn as_mut(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.p, self.n) } }
}
impl Drop for PageBuf { fn drop(&mut self) { unsafe { munmap(self.p.cast(), self.n); } } }

/// A CUDA event recorded on the current stream (the copies out of a buffer), waited for by the reader thread.
struct Event(*mut std::ffi::c_void);
unsafe impl Send for Event {}
impl Event {
    fn record() -> Event {
        let mut e = std::ptr::null_mut();
        assert_eq!(unsafe { cudaEventCreateWithFlags(&mut e, 2 /* cudaEventDisableTiming */) }, 0, "cudaEventCreate");
        assert_eq!(unsafe { cudaEventRecord(e, rs_current_stream()) }, 0, "cudaEventRecord");
        Event(e)
    }
    fn sync(&self) { assert_eq!(unsafe { cudaEventSynchronize(self.0) }, 0, "cudaEventSynchronize"); }
}
impl Drop for Event { fn drop(&mut self) { unsafe { cudaEventDestroy(self.0); } } }

/// Bytes [a, b) of `path`.
#[derive(Clone, Debug)]
pub struct Region { pub path: PathBuf, pub a: usize, pub b: usize }

/// Regions read ahead strictly in list order into `nbuf` buffers of the largest region's size (region i into buffer
/// i % nbuf). The consumer asks for regions in increasing order; asking for region i gives back every earlier one it
/// held (after the copies already queued out of them have run: an event), and buffer i % nbuf is refilled with region
/// i + nbuf once region i is given back.
pub struct Readahead {
    ready: Receiver<(usize, usize)>,
    release: Option<Sender<Event>>,
    held: BTreeMap<usize, usize>,
    cur: Option<usize>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// time the consumer waited for the reader, the time since the start, bytes read (reported at the end)
    waited: std::time::Duration, t0: std::time::Instant, bytes: usize, tag: &'static str,
}
impl Readahead {
    pub fn start(regions: Vec<Region>, nbuf: usize, tag: &'static str) -> Readahead {
        let nbuf = nbuf.max(1);
        // each region from its aligned start, plus 16 readable bytes past the end (spark_copy_segments)
        let span = |r: &Region| (r.b + 16 - (r.a & !(ALIGN - 1))).div_ceil(ALIGN) * ALIGN;
        let cap = regions.iter().map(span).max().unwrap_or(0);
        let bytes = regions.iter().map(|r| r.b - r.a).sum();
        let (tx_ready, ready) = channel();
        let (release, rx_release) = channel::<Event>();
        let stop = Arc::new(AtomicBool::new(false));
        let st = stop.clone();
        let thread = std::thread::spawn(move || {
            use std::os::unix::fs::{FileExt, OpenOptionsExt};
            const READ: usize = 64 << 20;
            let mut bufs: Vec<PageBuf> = Vec::new();
            for (i, r) in regions.iter().enumerate() {
                let k = i % nbuf;
                if bufs.len() <= k { bufs.push(PageBuf::new(cap)); }
                if i >= nbuf {
                    // given back in order: the next release is region i - nbuf's, the holder of buffer k
                    match rx_release.recv() { Ok(ev) => ev.sync(), Err(_) => return }
                }
                let Ok(f) = std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(&r.path) else { return };
                let a0 = r.a & !(ALIGN - 1);
                let (need, end) = (r.b - a0, span(r));
                let dst = bufs[k].as_mut();
                let mut got = 0usize;
                while got < need {
                    if st.load(Ordering::Relaxed) { return; }
                    let e = (got + READ).min(end);
                    match f.read_at(&mut dst[got..e], (a0 + got) as u64) { Ok(0) | Err(_) => return, Ok(n) => got += n }
                }
                if tx_ready.send((i, dst.as_ptr() as usize + (r.a - a0))).is_err() { return; }
            }
            // the buffers stay until the consumer is done with them (Drop: device synchronized, release sender dropped)
            while rx_release.recv().is_ok() {}
        });
        Readahead { ready, release: Some(release), held: Default::default(), cur: None, stop, thread: Some(thread),
                    waited: Default::default(), t0: std::time::Instant::now(), bytes, tag }
    }
    /// Host address of region i's first byte (valid until a later region is asked for); None when region i was already
    /// given back or the reader failed (the caller then reads the bytes another way).
    pub fn get(&mut self, i: usize) -> Option<*const u8> {
        if self.cur.is_some_and(|c| i < c) { return None; }
        if self.cur != Some(i) {
            loop {
                while let Ok((ri, p)) = self.ready.try_recv() { self.held.insert(ri, p); }
                let done: Vec<usize> = self.held.keys().copied().filter(|&ri| ri < i).collect();
                for ri in done {
                    self.held.remove(&ri);
                    let _ = self.release.as_ref()?.send(Event::record());
                }
                if self.held.contains_key(&i) { break; }
                let tw = std::time::Instant::now();
                let r = self.ready.recv();
                self.waited += tw.elapsed();
                let (ri, p) = r.ok()?;
                self.held.insert(ri, p);
            }
            self.cur = Some(i);
        }
        Some(self.held[&i] as *const u8)
    }
}
impl Drop for Readahead {
    fn drop(&mut self) {
        eprintln!("[{}] read-ahead: {:.1} GB, the loader waited {:.1}s of {:.1}s for reads", self.tag, self.bytes as f64 / 1e9,
                  self.waited.as_secs_f64(), self.t0.elapsed().as_secs_f64());
        // every copy out of the buffers ran before the reader frees them
        tch::Cuda::synchronize(0);
        self.stop.store(true, Ordering::Relaxed);
        self.release.take();
        if let Some(t) = self.thread.take() { let _ = t.join(); }
    }
}

/// Copies segments (source offset from `src`, destination offset from `dst` (16-aligned), bytes) on the current stream
/// in one launch. `src` must be readable 16 bytes past each segment (a Readahead region is).
pub fn copy_segments(dst: *mut u8, src: *const u8, segs: &[(usize, usize, usize)], dev: tch::Device) {
    if segs.is_empty() { return; }
    let mut t = Vec::with_capacity(4 * segs.len() + 4);
    let mut chunks = 0i64;
    for &(s, d, n) in segs {
        assert_eq!(d % 16, 0, "copy_segments: destination offset {d} not 16-aligned");
        t.extend_from_slice(&[s as i64, d as i64, n as i64, chunks]);
        chunks += n.div_ceil(16) as i64;
    }
    t.extend_from_slice(&[0, 0, 0, chunks]);
    // the table on the device; dropped after the launch, its memory reused only by later work on this stream
    let table = tch::Tensor::from_slice(&t).to_device(dev);
    assert_eq!(unsafe { spark_copy_segments(dst.cast(), src.cast(), table.data_ptr() as *const i64, segs.len() as i32, chunks, rs_current_stream()) },
               0, "spark_copy_segments");
}
