// SPDX-License-Identifier: MIT
//! safetensors 分片索引:一次扫描 header,张量按需 mmap 读取。

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

#[derive(Clone, Debug)]
pub struct Entry {
    pub file: String,
    pub offset: usize,
    pub nbytes: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Background page-cache warmer from ShardIndex::prefetch.
pub struct Prefetch { stop: std::sync::Arc<std::sync::atomic::AtomicBool>, handles: Vec<std::thread::JoinHandle<()>>, pub bytes: usize, pub done: std::sync::Arc<Done> }

/// Set when every prefetch worker has finished (or stopped); other readers can wait for it to give
/// the prefetched bytes the disk first.
#[derive(Default)]
pub struct Done { left: std::sync::Mutex<usize>, cv: std::sync::Condvar }
impl Done {
    /// Signal that completes after `n` finish_one calls.
    pub fn new(n: usize) -> std::sync::Arc<Self> { std::sync::Arc::new(Done { left: std::sync::Mutex::new(n), cv: Default::default() }) }
    pub fn wait(&self) { let mut l = self.left.lock().unwrap(); while *l > 0 { l = self.cv.wait(l).unwrap(); } }
    pub fn finish_one(&self) { let mut l = self.left.lock().unwrap(); *l -= 1; if *l == 0 { self.cv.notify_all(); } }
}
impl Drop for Prefetch {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in self.handles.drain(..) { let _ = h.join(); }
    }
}

/// Cloning is cheap (shared, read-only): one scan per directory per process, see `scan`.
#[derive(Clone)]
pub struct ShardIndex {
    pub dir: PathBuf,
    pub entries: std::sync::Arc<HashMap<String, Entry>>,
    mmaps: std::sync::Arc<HashMap<String, Mmap>>, // 扫描时全部预映射(之后 &self 只读,rayon 可并行)
}

fn read_header(path: &Path) -> io::Result<(usize, serde_json::Map<String, serde_json::Value>)> {
    let f = fs::File::open(path)?;
    let mut reader = io::BufReader::new(f);
    let mut buf8 = [0u8; 8];
    io::Read::read_exact(&mut reader, &mut buf8)?;
    let hlen = u64::from_le_bytes(buf8) as usize;
    let mut hdr = vec![0u8; hlen];
    io::Read::read_exact(&mut reader, &mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok((8 + hlen, v.as_object().cloned().unwrap_or_default()))
}

impl ShardIndex {
    /// Index of `dir`, scanned (headers parsed, shards mapped) once per process: the target weights, the
    /// expert pool and the fallback pool of one serve process used to parse the same ~120 headers each.
    pub fn scan(dir: &Path) -> io::Result<Self> {
        static CACHE: std::sync::Mutex<Option<HashMap<PathBuf, ShardIndex>>> = std::sync::Mutex::new(None);
        let key = fs::canonicalize(dir)?;
        if let Some(i) = CACHE.lock().unwrap().get_or_insert_with(HashMap::new).get(&key) { return Ok(i.clone()); }
        let i = Self::scan_uncached(dir)?;
        CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, i.clone());
        Ok(i)
    }

    fn scan_uncached(dir: &Path) -> io::Result<Self> {
        let mut entries = HashMap::new();
        let mut files: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n=="model.safetensors" || (n.starts_with("model-") && n.ends_with(".safetensors")))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        for path in files {
            let fname = path.file_name().unwrap().to_string_lossy().to_string();
            let (base, hdr) = read_header(&path)?;
            for (name, meta) in &hdr {
                if name == "__metadata__" {
                    continue;
                }
                let offs = meta
                    .get("data_offsets")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>())
                    .unwrap_or_default();
                if offs.len() != 2 {
                    continue;
                }
                let shape = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
                    .unwrap_or_default();
                entries.insert(
                    name.clone(),
                    Entry {
                        file: fname.clone(),
                        offset: base + offs[0] as usize,
                        nbytes: (offs[1] - offs[0]) as usize,
                        dtype: meta
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        shape,
                    },
                );
            }
        }
        let files: std::collections::HashSet<String> = entries.values().map(|e: &Entry| e.file.clone()).collect();
        let mut mmaps = HashMap::new();
        for f in files {
            let file = fs::File::open(dir.join(&f))?;
            mmaps.insert(f, unsafe { Mmap::map(&file)? });
        }
        Ok(Self { dir: dir.to_path_buf(), entries: std::sync::Arc::new(entries), mmaps: std::sync::Arc::new(mmaps) })
    }

    fn mmap(&self, file: &str) -> io::Result<&Mmap> {
        self.mmaps
            .get(file)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, file.to_string()))
    }

    /// 读取任意张量为 f32(BF16 位重释 / F32 直取)。shape 顺带返回。
    pub fn get_f32(&self, name: &str) -> io::Result<(Vec<f32>, Vec<usize>)> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))?
            .clone();
        let m = self.mmap(&e.file)?;
        let bytes = &m[e.offset..e.offset + e.nbytes];
        let v = match e.dtype.as_str() {
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("dtype {other} 未支持(get_f32)"),
                ))
            }
        };
        Ok((v, e.shape.clone()))
    }

    /// 读取 int16 张量(如 trellis)。
    pub fn get_i16(&self, name: &str) -> io::Result<Vec<i16>> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))?
            .clone();
        let m = self.mmap(&e.file)?;
        let bytes = &m[e.offset..e.offset + e.nbytes];
        Ok(bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    /// Warm the page cache for the tensors `keep` selects, ahead of a sequential consumer of the mappings:
    /// regions (merged per file, gaps <= 1 MiB) sorted by `order(name)` then file offset, read by `threads`
    /// workers with buffered pread in that order. Consumer faults then hit cached pages (or wait on the
    /// in-flight read) instead of one ~0.7 GB/s readahead stream. Dropping the handle stops and joins it.
    pub fn prefetch(&self, threads: usize, keep: impl Fn(&str) -> bool, order: impl Fn(&str) -> i64) -> Prefetch {
        let mut v: Vec<(i64, &str, usize, usize)> = self.entries.iter()
            .filter(|(n, _)| keep(n))
            .map(|(n, e)| (order(n), e.file.as_str(), e.offset, e.offset + e.nbytes)).collect();
        v.sort();
        let mut regions: Vec<(std::path::PathBuf, usize, usize)> = Vec::new();
        let mut last: Option<(i64, &str)> = None;
        for (k, f, a, b) in v {
            match regions.last_mut() {
                Some(r) if last == Some((k, f)) && a <= r.2 + (1 << 20) => r.2 = r.2.max(b),
                _ => regions.push((self.dir.join(f), a, b)),
            }
            last = Some((k, f));
        }
        let bytes: usize = regions.iter().map(|r| r.2 - r.1).sum();
        let regions = std::sync::Arc::new(regions);
        let next = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done = std::sync::Arc::new(Done { left: std::sync::Mutex::new(threads.max(1)), cv: Default::default() });
        let handles = (0..threads.max(1)).map(|_| {
            let (regions, next, stop, done) = (regions.clone(), next.clone(), stop.clone(), done.clone());
            std::thread::spawn(move || {
                struct Finish(std::sync::Arc<Done>);
                impl Drop for Finish { fn drop(&mut self) { self.0.finish_one(); } }
                let _finish = Finish(done);
                use std::os::unix::fs::FileExt;
                let mut buf = vec![0u8; 8 << 20];
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= regions.len() || stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                    let (path, a, b) = &regions[i];
                    let Ok(f) = fs::File::open(path) else { continue }; // the consumer reports real errors
                    let mut off = *a;
                    while off < *b && !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let n = (*b - off).min(buf.len());
                        match f.read_at(&mut buf[..n], off as u64) { Ok(0) | Err(_) => break, Ok(r) => off += r }
                    }
                }
            })
        }).collect();
        Prefetch { stop, handles, bytes, done }
    }

    /// Read `dst.len()` bytes of tensor `name` starting `off` bytes into it, with pread. Copying cached file pages
    /// into a staging buffer, then uploading from there, is far faster on this host than letting the device copy
    /// fault the file mapping in (~1 GB/s).
    pub fn read_range(&self, name: &str, off: usize, dst: &mut [u8]) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        let e = self.entries.get(name).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))?;
        assert!(off + dst.len() <= e.nbytes, "read past the end of {name}");
        fs::File::open(self.dir.join(&e.file))?.read_exact_at(dst, (e.offset + off) as u64)
    }

    /// Drop tensor `name`'s pages from the page cache: consumed once, they would otherwise compete with the
    /// device allocations that follow (a partial first/last page is re-read by a neighbour if needed).
    pub fn drop_cache(&self, name: &str) {
        let Some(e) = self.entries.get(name) else { return };
        let Ok(f) = fs::File::open(self.dir.join(&e.file)) else { return };
        extern "C" { fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32; }
        use std::os::unix::io::AsRawFd;
        unsafe { posix_fadvise(f.as_raw_fd(), e.offset as i64, e.nbytes as i64, 4) };
    }

    /// 张量原始字节切片(零拷贝,供并行装载/H2D 直传)。
    pub fn get_bytes(&self, name: &str) -> io::Result<&[u8]> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))?;
        let m = self.mmap(&e.file)?;
        Ok(&m[e.offset..e.offset + e.nbytes])
    }
}
