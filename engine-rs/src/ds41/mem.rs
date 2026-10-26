//! Memory budget of a DS41 serving rank (docs/07 D5). Serving puts the memory left after the warm-up into the KV pool
//! (crate::kv_pool, VMM: outside the caching allocator, whose cap is lowered by it: spare / lower_cap), so the ballast
//! below then holds little or nothing. GB10 has one 121 GB memory for CPU and GPU and the nodes run
//! without swap, so an engine that keeps growing takes the memory the OS needs and the node stops answering (seen
//! twice). Like vLLM, the engine therefore takes its whole budget up front and never exceeds it:
//! - after loading, the CUDA caching allocator is capped at (reserved now + MemAvailable - DS41_MEM_MARGIN_GB (5));
//!   an allocation beyond fails inside the engine (CUDA OOM) instead of starving the OS;
//! - after the warm-up (and after each later graph capture) the free part of the budget is occupied by a held
//!   "ballast" tensor (up to the cap minus a small headroom): the warm-up's peak stays in the allocator's cache for the
//!   eager steps, the rest is taken. The ballast is never handed to other allocations, so it cannot be split up and
//!   pinned by them (an occupation freed into the cache was, and then blocked graph captures: 9 GiB cached but unusable);
//!   Later occupations keep the warm-up's eager working set free too (the capture below empties the cache);
//! - before a new graph capture (graphs allocate from private pools) the ballast is dropped and the cache emptied,
//!   and retaken after.
use std::sync::atomic::{AtomicI64, Ordering};

static CAP: AtomicI64 = AtomicI64::new(0);
/// The warm-up's eager working set (cached but unused at the first occupation): every later occupation (after a graph
/// capture emptied the cache) leaves this much free under the cap, or the next long prefill chunk fails.
static EAGER: AtomicI64 = AtomicI64::new(-1);
static WARMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Serving with a KV pool: no ballast (the spare memory went to the pool; what is left on the larger ranks stays free
/// for the OS, the allocator's cap still bounds the engine).
static NO_BALLAST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn no_ballast() { NO_BALLAST.store(true, Ordering::SeqCst); }
/// During the warm-up nothing is occupied (graph captures still empty the cache).
pub fn warming(on: bool) { WARMING.store(on, Ordering::SeqCst); }
/// The one memory pool of every DS41 graph (row graphs and DSpark draft graphs): with a pool per captured segment, the
/// 81 segments of each row signature each held their own blocks (several GiB over a few dozen signatures). Segments
/// replay one after another on one stream; their temporaries may share memory, their outputs are held by the caller.
pub const GRAPH_POOL: i64 = 0xd541;
thread_local! { static BALLAST: std::cell::RefCell<Option<tch::Tensor>> = const { std::cell::RefCell::new(None) }; }
/// Left free under the cap beyond the warm-up's cached peak (DS41_MEM_HEADROOM_GB, 2): eager growth, small blocks.
fn headroom() -> i64 { (std::env::var("DS41_MEM_HEADROOM_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(2.0) * (1u64 << 30) as f64) as i64 }

extern "C" {
    fn rs_cuda_set_memory_fraction(fraction: f64, total: *mut i64) -> i32;
    fn rs_empty_cache();
}

fn mem_available() -> i64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    s.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0) * 1024
}
fn gib(b: i64) -> f64 { b as f64 / (1u64 << 30) as f64 }

/// Sets the cap (once, after loading). DS41_MEM_MARGIN_GB=0 disables the budget.
pub fn set_cap(rank: usize) {
    let margin = std::env::var("DS41_MEM_MARGIN_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(5.0);
    RANK.store(rank as i64, Ordering::SeqCst);
    if margin <= 0.0 { return; }
    let (_, reserved) = crate::tp::graph::memory();
    let avail = mem_available();
    let cap = reserved + avail - (margin * (1u64 << 30) as f64) as i64;
    let mut total = 0i64;
    assert_eq!(unsafe { rs_cuda_set_memory_fraction(1.0, &mut total) }, 0, "cudaMemGetInfo");
    let frac = (cap as f64 / total as f64).clamp(0.05, 1.0);
    assert_eq!(unsafe { rs_cuda_set_memory_fraction(frac, &mut total) }, 0);
    CAP.store(cap, Ordering::SeqCst);
    eprintln!("[ds41-mem] rank{rank}: reserved {:.1} GiB, MemAvailable {:.1} GiB, margin {margin} GiB -> allocator cap {:.1} GiB ({:.3} of {:.1} GiB)",
              gib(reserved), gib(avail), gib(cap), frac, gib(total));
}

/// After the warm-up (its eager peak cached): the bytes of the budget the allocator does not need, i.e. what occupy
/// would hold: cap - reserved - headroom - the eager working set not cached right now. 0 without a budget.
pub fn spare() -> i64 {
    let cap = CAP.load(Ordering::SeqCst);
    if cap == 0 { return 0; }
    let (allocated, reserved) = crate::tp::graph::memory();
    let cached = reserved - allocated;
    if EAGER.load(Ordering::SeqCst) < 0 { EAGER.store(cached, Ordering::SeqCst); }
    (cap - reserved - headroom() - (EAGER.load(Ordering::SeqCst) - cached).max(0)).max(0)
}
/// Takes `bytes` out of the allocator's budget (memory handed to the KV pool, outside the caching allocator), so the
/// engine's total stays within the budget.
pub fn lower_cap(rank: usize, bytes: i64) {
    let cap = CAP.load(Ordering::SeqCst);
    if cap == 0 || bytes <= 0 { return; }
    let cap = cap - bytes;
    let mut total = 0i64;
    assert_eq!(unsafe { rs_cuda_set_memory_fraction(1.0, &mut total) }, 0, "cudaMemGetInfo");
    let frac = (cap as f64 / total as f64).clamp(0.05, 1.0);
    assert_eq!(unsafe { rs_cuda_set_memory_fraction(frac, &mut total) }, 0);
    CAP.store(cap, Ordering::SeqCst);
    eprintln!("[ds41-mem] rank{rank}: allocator cap lowered by {:.2} GiB (the KV pool and the spare memory left to the OS) -> {:.1} GiB", gib(bytes), gib(cap));
}

/// Occupies the free part of the budget (cached in the allocator).
pub fn occupy() {
    let before = RSS_AT_RELEASE.swap(0, Ordering::SeqCst);
    if before > 0 {
        let (n, now) = (CAPTURES.fetch_add(1, Ordering::SeqCst) + 1, rss_anon());
        eprintln!("[ds41-mem] rank{}: graph capture {n}: host anon {:+.0} MiB -> {:.2} GiB", RANK.load(Ordering::SeqCst), mib(now - before), gib(now));
    }
    let cap = CAP.load(Ordering::SeqCst);
    if cap == 0 || WARMING.load(Ordering::SeqCst) || NO_BALLAST.load(Ordering::SeqCst) { return; }
    BALLAST.with(|b| {
        let mut b = b.borrow_mut();
        if b.is_some() { return; }
        let (allocated, reserved) = crate::tp::graph::memory();
        let cached = reserved - allocated;
        if EAGER.load(Ordering::SeqCst) < 0 { EAGER.store(cached, Ordering::SeqCst); }
        let want = cap - reserved - headroom() - (EAGER.load(Ordering::SeqCst) - cached).max(0);
        if want < (256 << 20) { return; }
        let dev = tch::Device::Cuda(0);
        match tch::Tensor::f_empty([want], (tch::Kind::Uint8, dev)) {
            Ok(t) => {
                eprintln!("[ds41-mem] occupied {:.1} GiB (in use {:.1} GiB, eager cache {:.1} GiB kept, headroom {:.1} GiB, cap {:.1} GiB)",
                    gib(want), gib(allocated), gib(cached), gib(headroom()), gib(cap));
                *b = Some(t)
            }
            Err(_) => eprintln!("[ds41-mem] occupy of {:.1} GiB failed", gib(want)),
        }
    });
}

/// Returns the occupied (cached, unused) memory before a graph capture.
pub fn release() {
    RSS_AT_RELEASE.store(rss_anon(), Ordering::SeqCst);
    if CAP.load(Ordering::SeqCst) == 0 { return; }
    BALLAST.with(|b| { b.borrow_mut().take(); });
    unsafe { rs_empty_cache() }
}

/// The caching allocator's (allocation retries, cudaFree calls) so far.
pub fn alloc_stats() -> (i64, i64) {
    extern "C" { fn rs_cuda_alloc_stats(r: *mut i64, f: *mut i64) -> i32; }
    let (mut r, mut f) = (0i64, 0i64);
    unsafe { rs_cuda_alloc_stats(&mut r, &mut f) };
    (r, f)
}
pub fn report(rank: usize) {
    let (a, r) = crate::tp::graph::memory();
    eprintln!("[ds41-mem] rank{rank}: allocated {:.1} GiB, reserved {:.1} GiB, MemAvailable {:.1} GiB, {}", gib(a), gib(r), gib(mem_available()), host());
}

// The host side of the process is not under the allocator's cap: on GB10 it comes out of the same memory, so its growth
// eats the margin left to the OS (seen 10-19: rank0 MemAvailable 5.5 GiB after the warm-up -> 0.1 GiB, host anon 8.7 GB).
// Observation only: graph captures log their host growth, a watcher logs every change of the host side.
static RANK: AtomicI64 = AtomicI64::new(0);
static CAPTURES: AtomicI64 = AtomicI64::new(0);
static RSS_AT_RELEASE: AtomicI64 = AtomicI64::new(0);
fn mib(b: i64) -> f64 { b as f64 / (1u64 << 20) as f64 }

/// RssAnon of this process (bytes).
pub fn rss_anon() -> i64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines().find(|l| l.starts_with("RssAnon:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0) * 1024
}

#[repr(C)]
struct MallInfo2 { arena: usize, ordblks: usize, smblks: usize, hblks: usize, hblkhd: usize, usmblks: usize, fsmblks: usize, uordblks: usize, fordblks: usize, keepcost: usize }
extern "C" { fn mallinfo2() -> MallInfo2; }

/// The host side: RssAnon and glibc malloc (in use in the arenas, free in the arenas, mmapped chunks).
fn host() -> String {
    let m = unsafe { mallinfo2() };
    format!("host anon {:.2} GiB (malloc in use {:.2}, free in arenas {:.2}, mmapped {:.2} GiB; {} graph captures)",
        gib(rss_anon()), gib(m.uordblks as i64), gib(m.fordblks as i64), gib(m.hblkhd as i64), CAPTURES.load(Ordering::SeqCst))
}

/// Logs the memory every DS41_MEM_WATCH_SECS (60; 0 off) when the host anon moved by 64 MiB or more since the last line.
pub fn watch(rank: usize) {
    RANK.store(rank as i64, Ordering::SeqCst);
    let secs = std::env::var("DS41_MEM_WATCH_SECS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(60);
    if secs == 0 { return; }
    let _ = std::thread::Builder::new().name("ds41-memwatch".into()).spawn(move || {
        let mut last = rss_anon();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            let now = rss_anon();
            if (now - last).abs() < (64 << 20) { continue; }
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) % 86400;
            let (a, r) = crate::tp::graph::memory();
            eprintln!("[ds41-mem] {:02}:{:02}:{:02}Z rank{rank}: host anon {:+.0} MiB; allocated {:.1} GiB, reserved {:.1} GiB, MemAvailable {:.2} GiB, {}",
                t / 3600, t / 60 % 60, t % 60, mib(now - last), gib(a), gib(r), gib(mem_available()), host());
            last = now;
        }
    });
}

/// For the request line: host anon and MemAvailable.
pub fn brief() -> String { format!("host anon {:.2} GiB, MemAvailable {:.2} GiB", gib(rss_anon()), gib(mem_available())) }
