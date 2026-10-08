//! Loading stages must not retain unused host allocator pages indefinitely.
//! This releases only free glibc arena pages, never live tensors or CUDA pools.
use serde_json::{json,Value};

pub fn snapshot()->Value {
    let status=std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kib=|key:&str|status.lines().find_map(|l|l.strip_prefix(key))
        .and_then(|v|v.split_whitespace().next()).and_then(|v|v.parse::<u64>().ok());
    let stat=std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields:Vec<_>=stat.rsplit_once(')').map(|(_,v)|v.split_whitespace().collect()).unwrap_or_default();
    let field=|index:usize|fields.get(index).and_then(|v|v.parse::<u64>().ok());
    let pressure=std::fs::read_to_string("/proc/pressure/memory").unwrap_or_default();
    let total=|kind:&str|pressure.lines().find(|l|l.starts_with(kind))
        .and_then(|l|l.split_whitespace().find_map(|v|v.strip_prefix("total=")))
        .and_then(|v|v.parse::<u64>().ok());
    json!({"rss_kib":kib("VmRSS:"),"anon_kib":kib("RssAnon:"),"file_kib":kib("RssFile:"),
        "minor_faults":field(7),"major_faults":field(9),"memory_some_us":total("some "),"memory_full_us":total("full ")})
}

/// Startup timeline: `[load] +<s since the first stage> <wall clock> <stage>` on stderr, so both ranks'
/// logs line up and a restart can be split into its phases without a profiler.
pub fn stage(name:&str) {
    static T0:std::sync::OnceLock<std::time::Instant>=std::sync::OnceLock::new();
    let t=T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64();
    let wall=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d|d.as_secs_f64()).unwrap_or(0.);
    let avail=std::fs::read_to_string("/proc/meminfo").ok().and_then(|m|m.lines().find_map(|l|l.strip_prefix("MemAvailable:"))
        .and_then(|v|v.split_whitespace().next()).and_then(|v|v.parse::<f64>().ok())).map_or(-1.,|k|k/1048576.);
    eprintln!("[load] +{t:.2}s unix {wall:.2} {name} [avail {avail:.1} GiB]");
}

/// stage() plus the CUDA caching allocator's allocated/reserved bytes (only once CUDA is initialized).
pub fn stage_mem(name:&str) {
    extern "C"{fn rs_cuda_memory(allocated:*mut i64,reserved:*mut i64);}
    let (mut a,mut r)=(0i64,0i64);unsafe{rs_cuda_memory(&mut a,&mut r)};
    stage(&format!("{name} (allocated {:.2} GiB, reserved {:.2} GiB)",a as f64/(1u64<<30) as f64,r as f64/(1u64<<30) as f64));
}

/// Bytes this thread has read from storage (page-cache misses and direct reads), from /proc.
pub fn thread_read_bytes()->u64 {
    let tid=unsafe{libc_gettid()};
    std::fs::read_to_string(format!("/proc/self/task/{tid}/io")).ok().and_then(|s|s.lines().find_map(|l|l.strip_prefix("read_bytes:"))
        .and_then(|v|v.trim().parse().ok())).unwrap_or(0)
}
extern "C" {#[link_name="gettid"] fn libc_gettid()->i32;}

pub fn finish_loading(phase:&str) {
    if std::env::var("GLM53_HOST_TRIM").as_deref()!=Ok("1"){return;}
    let before=snapshot();let now=std::time::Instant::now();
    extern "C" {fn malloc_trim(pad:usize)->i32;}
    // glibc owns the arena synchronization and identifies unallocated pages.
    // No process-wide cache drop and no CUDA allocator purge is performed.
    let released=unsafe{malloc_trim(0)};let elapsed_ms=now.elapsed().as_secs_f64()*1000.;
    eprintln!("[host-trim] {}",json!({"phase":phase,"before":before,"after":snapshot(),"returned":released,"elapsed_ms":elapsed_ms}));
}

/// Probe-only allocator telemetry, outside the timed interval. Reset only the
/// allocator's peak counters; this never frees a tensor, pool or CUDA graph.
/// Driver free/total on GB10 are not a measurement of unique physical traffic.
pub fn cuda_snapshot(reset_peak:bool)->Value {
    extern "C" {fn rs_memory_stats(reset:i32,out:*mut i64)->i32;}
    let mut values=[0i64;8];
    assert_eq!(unsafe{rs_memory_stats(i32::from(reset_peak),values.as_mut_ptr())},0,"CUDA memory telemetry");
    let names=["allocated_current","allocated_peak","reserved_current","reserved_peak",
        "alloc_retries","ooms","driver_free","driver_total"];
    Value::Object(names.into_iter().zip(values).map(|(name,value)|(name.into(),json!(value))).collect())
}

/// A5 (GLM53_RELEASE_FILE_CACHE=1): after all weights are resident on the device, the
/// read-only safetensors mappings are only a fallback path. Unmap their pages from this
/// process (madvise DONTNEED; a later read simply faults them back from the file) and ask
/// the kernel to drop the files' clean page cache, so the unified memory is not held by
/// ~20 GB of duplicate weight bytes that kswapd must otherwise reclaim under pressure.
pub fn release_file_cache(dirs:&[&std::path::Path]) {
    if std::env::var("GLM53_RELEASE_FILE_CACHE").as_deref()!=Ok("1"){return;}
    extern "C" {fn madvise(addr:*mut std::ffi::c_void,len:usize,advice:i32)->i32;fn posix_fadvise(fd:i32,offset:i64,len:i64,advice:i32)->i32;}
    const MADV_DONTNEED:i32=4;const POSIX_FADV_DONTNEED:i32=4;
    let before=snapshot();let now=std::time::Instant::now();
    let roots:Vec<std::path::PathBuf>=dirs.iter().filter_map(|d|std::fs::canonicalize(d).ok()).collect();
    // HF snapshots are symlinks into blobs/: match mappings by the resolved file path.
    let mut files=std::collections::BTreeSet::new();
    for root in &roots {if let Ok(rd)=std::fs::read_dir(root){for e in rd.flatten(){let p=e.path();
        if p.extension().map_or(false,|e|e=="safetensors"){if let Ok(c)=std::fs::canonicalize(&p){files.insert(c);}}}}}
    let maps=std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let (mut ranges,mut bytes)=(0usize,0usize);
    for line in maps.lines() {
        let mut it=line.split_whitespace();
        let (Some(range),Some(_perm),Some(_off),Some(_dev),Some(_inode),Some(path))=(it.next(),it.next(),it.next(),it.next(),it.next(),it.next()) else {continue};
        if !files.contains(std::path::Path::new(path)) {continue;}
        let Some((a,b))=range.split_once('-') else {continue};
        let (a,b)=(usize::from_str_radix(a,16).unwrap(),usize::from_str_radix(b,16).unwrap());
        if unsafe{madvise(a as *mut _,b-a,MADV_DONTNEED)}==0 {ranges+=1;bytes+=b-a;}
    }
    let mut dropped=0;
    for f in &files {if let Ok(file)=std::fs::File::open(f){use std::os::unix::io::AsRawFd;if unsafe{posix_fadvise(file.as_raw_fd(),0,0,POSIX_FADV_DONTNEED)}==0{dropped+=1;}}}
    eprintln!("[file-cache] {}",json!({"mapped_ranges":ranges,"mapped_bytes":bytes,"files_dropped":dropped,"before":before,"after":snapshot(),"elapsed_ms":now.elapsed().as_secs_f64()*1000.}));
}
