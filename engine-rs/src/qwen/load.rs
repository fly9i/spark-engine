//! Checkpoint access for Qwen3.8 (EXL3 safetensors shards + the n-gram ring table): tensors are copied
//! from the mmap straight to the GPU (no anonymous host copy), the n-gram table stays mmap'd (MADV_RANDOM).
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};

pub struct TensorRef { file: usize, offset: usize, len: usize, pub dtype: String, pub shape: Vec<i64> }

pub struct Checkpoint {
    pub dir: PathBuf,
    pub cfg: Value,
    maps: Vec<Mmap>,
    names: Vec<String>,
    pub tensors: HashMap<String, TensorRef>,
    pub dev: Device,
    cache: std::cell::RefCell<HashMap<String, Tensor>>,
    staging: std::cell::RefCell<Option<Tensor>>,
    /// read-ahead of the `preload` prefixes (`read_ahead`): the reader and, per prefix, its regions (index, file, start)
    ahead: std::cell::RefCell<Option<(crate::readahead::Readahead, HashMap<String, Vec<(usize, usize, usize)>>)>>,
}

fn header(path: &Path) -> (usize, serde_json::Map<String, Value>) {
    use std::io::Read;
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8).unwrap();
    let n = u64::from_le_bytes(b8) as usize;
    let mut h = vec![0u8; n];
    f.read_exact(&mut h).unwrap();
    let v: Value = serde_json::from_slice(&h).expect("safetensors header");
    (8 + n, v.as_object().unwrap().clone())
}

pub fn kind_of(dtype: &str) -> Kind {
    match dtype {
        "F32" => Kind::Float, "F16" => Kind::Half, "BF16" => Kind::BFloat16, "I16" => Kind::Int16,
        "I32" => Kind::Int, "I64" => Kind::Int64, "U8" => Kind::Uint8, "I8" => Kind::Int8,
        d => panic!("unsupported dtype {d}"),
    }
}

impl Checkpoint {
    pub fn open(dir: &Path, extra_files: &[&str], dev: Device) -> Self {
        let cfg: Value = serde_json::from_reader(File::open(dir.join("config.json")).expect("config.json")).unwrap();
        let idx: Value = serde_json::from_reader(File::open(dir.join("model.safetensors.index.json")).expect("index")).unwrap();
        let mut files: Vec<String> = idx["weight_map"].as_object().unwrap().values().map(|v| v.as_str().unwrap().to_string()).collect();
        files.sort();
        files.dedup();
        files.extend(extra_files.iter().map(|s| s.to_string()));
        let mut ck = Checkpoint { dir: dir.into(), cfg, maps: vec![], names: vec![], tensors: HashMap::new(), dev,
                                  cache: Default::default(), staging: Default::default(), ahead: Default::default() };
        for (fi, name) in files.iter().enumerate() {
            let path = dir.join(name);
            let (base, h) = header(&path);
            let f = File::open(&path).unwrap();
            let map = unsafe { Mmap::map(&f) }.expect("mmap shard");
            for (k, v) in h {
                if k == "__metadata__" { continue; }
                let off = v["data_offsets"].as_array().unwrap();
                let (a, b) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
                let shape = v["shape"].as_array().unwrap().iter().map(|d| d.as_i64().unwrap()).collect();
                ck.tensors.insert(k, TensorRef { file: fi, offset: base + a, len: b - a, dtype: v["dtype"].as_str().unwrap().into(), shape });
            }
            ck.maps.push(map);
            ck.names.push(name.clone());
        }
        ck
    }

    pub fn text_cfg(&self) -> &Value { &self.cfg["text_config"] }
    pub fn has(&self, k: &str) -> bool { self.tensors.contains_key(k) }

    pub fn bytes(&self, k: &str) -> &[u8] {
        let t = self.tensors.get(k).unwrap_or_else(|| panic!("missing tensor {k}"));
        &self.maps[t.file][t.offset..t.offset + t.len]
    }

    /// Tensor `k` on the device (or on the CPU with `cpu`), with its checkpoint dtype.
    pub fn get_on(&self, k: &str, dev: Device) -> Tensor {
        let t = &self.tensors[k];
        let b = self.bytes(k);
        let kind = kind_of(&t.dtype);
        let mut strides = vec![1i64; t.shape.len()];
        for i in (0..t.shape.len().saturating_sub(1)).rev() { strides[i] = strides[i + 1] * t.shape[i + 1]; }
        let view = unsafe { Tensor::from_blob(b.as_ptr(), &t.shape, &strides, kind, Device::Cpu) };
        if dev == Device::Cpu { view.copy() } else { view.to_device(dev) }
    }
    pub fn get(&self, k: &str) -> Tensor {
        if let Some(t) = self.cache.borrow_mut().remove(k) { return t; }
        self.get_on(k, self.dev)
    }

    fn preload_keys(&self, prefix: &str) -> Vec<&String> {
        let mut keys: Vec<&String> = self.tensors.keys()
            .filter(|k| k.starts_with(prefix) && !k.ends_with("ngram_embedding.trellis")).collect();
        keys.sort();
        keys
    }

    /// Start reading the tensors of `prefixes` (each the argument of a later `preload`, in the order they will be
    /// preloaded) ahead with O_DIRECT (QWEN_LOAD_AHEAD, default on; 0: `preload` copies out of the mmap). A prefix's
    /// tensors lie together in its shard (a layer, ~1.35 GB): one region per file it spans.
    pub fn read_ahead(&self, prefixes: &[String]) {
        if !self.dev.is_cuda() || std::env::var("QWEN_LOAD_AHEAD").as_deref() == Ok("0") { return; }
        let mut regions = Vec::new();
        let mut map = HashMap::new();
        for p in prefixes {
            let mut per: std::collections::BTreeMap<usize, (usize, usize)> = Default::default();
            for k in self.preload_keys(p) {
                let t = &self.tensors[k];
                let e = per.entry(t.file).or_insert((t.offset, t.offset + t.len));
                e.0 = e.0.min(t.offset);
                e.1 = e.1.max(t.offset + t.len);
            }
            let v = per.into_iter().map(|(f, (a, b))| {
                regions.push(crate::readahead::Region { path: self.dir.join(&self.names[f]), a, b });
                (regions.len() - 1, f, a)
            }).collect();
            map.insert(p.clone(), v);
        }
        *self.ahead.borrow_mut() = Some((crate::readahead::Readahead::start(regions, 3, "qwen"), map));
    }
    /// Ends the read-ahead (its buffers are freed).
    pub fn finish_ahead(&self) { self.ahead.borrow_mut().take(); }

    /// Bulk-load every tensor whose name starts with `prefix` into one device allocation (tensors at 256-byte aligned
    /// offsets in name order, views of it): with `read_ahead` one scatter copy per region out of the read-ahead
    /// buffer, otherwise parallel copies from the mmap into a pinned staging buffer and one H2D copy per 256 MiB.
    pub fn preload(&self, prefix: &str) {
        use rayon::prelude::*;
        const STAGE: usize = 256 << 20;
        let keys = self.preload_keys(prefix);
        let align = |n: usize| (n + 255) / 256 * 256;
        let total: usize = keys.iter().map(|k| align(self.tensors[*k].len)).sum();
        if total == 0 { return; }
        let gpu = Tensor::empty([total as i64], (Kind::Uint8, self.dev));
        if self.preload_ahead(prefix, &keys, &gpu) {
            // QWEN_LOAD_CHECK=1: every tensor's bytes against the mmap (slow: the shards are read twice)
            let check = std::env::var("QWEN_LOAD_CHECK").as_deref() == Ok("1");
            let mut o = 0usize;
            let mut cache = self.cache.borrow_mut();
            for k in &keys {
                let t = &self.tensors[*k];
                if check {
                    let want = Tensor::from_slice(self.bytes(k)).to_device(self.dev);
                    assert!(gpu.narrow(0, o as i64, t.len as i64).equal(&want), "read-ahead bytes of {k} differ from the mmap");
                }
                cache.insert((*k).clone(), gpu.narrow(0, o as i64, t.len as i64).view_dtype(kind_of(&t.dtype)).view(t.shape.as_slice()));
                o += align(t.len);
            }
            return;
        }
        let mut stg = self.staging.borrow_mut();
        let staging = stg.get_or_insert_with(|| Tensor::empty([STAGE as i64], (Kind::Uint8, Device::Cpu)).pin_memory(self.dev));
        let sp = staging.data_ptr() as usize;
        let mut off = 0usize;
        let mut i = 0;
        while i < keys.len() {
            // a batch of tensors that fits the staging buffer (a single larger tensor goes alone, in pieces)
            let mut j = i;
            let mut used = 0usize;
            while j < keys.len() && (j == i || used + align(self.tensors[keys[j]].len) <= STAGE) {
                used += align(self.tensors[keys[j]].len);
                j += 1;
            }
            if used > STAGE {
                let b = self.bytes(keys[i]);
                let mut done = 0;
                while done < b.len() {
                    let n = (b.len() - done).min(STAGE);
                    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr().add(done), sp as *mut u8, n); }
                    gpu.narrow(0, (off + done) as i64, n as i64).copy_(&staging.narrow(0, 0, n as i64));
                    done += n;
                }
            } else {
                let mut offs = Vec::with_capacity(j - i);
                let mut o = 0usize;
                for k in &keys[i..j] { offs.push(o); o += align(self.tensors[*k].len); }
                let srcs: Vec<(usize, usize, usize)> = keys[i..j].iter().zip(offs.iter())
                    .map(|(k, &o)| { let b = self.bytes(k); (b.as_ptr() as usize, b.len(), o) }).collect();
                srcs.par_iter().for_each(|&(src, len, o)| unsafe {
                    std::ptr::copy_nonoverlapping(src as *const u8, (sp + o) as *mut u8, len);
                });
                gpu.narrow(0, off as i64, used as i64).copy_(&staging.narrow(0, 0, used as i64));
            }
            let mut o = off;
            let mut cache = self.cache.borrow_mut();
            for k in &keys[i..j] {
                let t = &self.tensors[*k];
                let v = gpu.narrow(0, o as i64, t.len as i64).view_dtype(kind_of(&t.dtype)).view(t.shape.as_slice());
                cache.insert((*k).clone(), v);
                o += align(t.len);
            }
            off += used;
            i = j;
        }
    }
    /// The read-ahead half of `preload`: false when `prefix` was not read ahead (or its regions are gone).
    fn preload_ahead(&self, prefix: &str, keys: &[&String], gpu: &Tensor) -> bool {
        let mut ah = self.ahead.borrow_mut();
        let Some((ra, map)) = ah.as_mut() else { return false };
        let Some(regs) = map.get(prefix) else { return false };
        let align = |n: usize| (n + 255) / 256 * 256;
        let mut dst = Vec::with_capacity(keys.len());
        let mut o = 0usize;
        for k in keys { dst.push(o); o += align(self.tensors[*k].len); }
        for &(ri, f, a) in regs {
            let Some(src) = ra.get(ri) else { return false };
            let segs: Vec<(usize, usize, usize)> = keys.iter().zip(&dst).filter(|(k, _)| self.tensors[**k].file == f)
                .map(|(k, &d)| { let t = &self.tensors[*k]; (t.offset - a, d, t.len) }).collect();
            crate::readahead::copy_segments(gpu.data_ptr() as *mut u8, src, &segs, self.dev);
        }
        true
    }
    pub fn clear_cache(&self) { self.cache.borrow_mut().clear(); }

    /// Raw pointer + length of a tensor in the mmap (for row gathers from the n-gram table).
    pub fn raw_ptr(&self, k: &str) -> (*const u8, usize) { let b = self.bytes(k); (b.as_ptr(), b.len()) }

    /// Drop the page cache of every shard except `keep` (their tensors are on the device now). After a load the cached
    /// shards fill most of the memory, and the first large device allocations then wait for the kernel to reclaim and
    /// compact it: a fresh process's first 12K-token prefill took 7.6..17 s instead of 6.7 s.
    pub fn release_shards(&self, keep: &[&str]) {
        use std::os::unix::io::AsRawFd;
        extern "C" {
            fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        }
        let mut bytes = 0usize;
        for (map, name) in self.maps.iter().zip(&self.names) {
            if keep.iter().any(|k| name.ends_with(k)) { continue; }
            unsafe { madvise(map.as_ptr() as *mut _, map.len(), 4 /* MADV_DONTNEED */); }
            if let Ok(f) = File::open(self.dir.join(name)) { unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 4 /* POSIX_FADV_DONTNEED */); } }
            bytes += map.len();
        }
        eprintln!("[qwen] released the page cache of {:.1} GB of loaded shards", bytes as f64 / 1e9);
    }

    pub fn advise_random(&self, k: &str) {
        extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }
        let (p, n) = self.raw_ptr(k);
        let page = 4096usize;
        let a = (p as usize) & !(page - 1);
        unsafe { madvise(a as *mut _, n + (p as usize - a), 1 /* MADV_RANDOM */); }
    }
}
