//! Checkpoint access for DeepSeek-V4.1-Flash (official HF safetensors): shards are mmap'd and only
//! this rank's slice of each tensor is copied to the GPU (row slices are contiguous; column slices are gathered row by
//! row through a host staging buffer). Decoding helpers for the stored formats:
//! - FP8 E4M3 with one E8M0 scale per 32x32 block (non-expert linears): dequantized exactly to BF16 (an E4M3 value
//!   times a power of two is exactly representable in BF16);
//! - MXFP4 (E2M1 pairs, low nibble first, one E8M0 scale per 32 along K): routed experts, kept packed;
//! - E8M0 scales: 2^(byte-127).
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};

pub struct TensorRef { file: usize, offset: usize, len: usize, pub dtype: String, pub shape: Vec<i64> }

pub struct Checkpoint {
    pub dir: PathBuf, maps: Vec<Mmap>, files: Vec<PathBuf>, pub tensors: HashMap<String, TensorRef>, pub dev: Device,
    /// A per-rank shard written by `ds41-shard`: every tensor is already this rank's slice, so `rows`/`cols` return
    /// it whole. `source` is the original checkpoint (engram tables stay there).
    pub sliced: bool,
    pub source: Option<PathBuf>,
    /// While recording (`ds41-shard`), every tensor handed out is kept under its name.
    record: std::cell::RefCell<Option<Vec<(String, Tensor)>>>,
    /// Shard loads to the GPU: the files opened with O_DIRECT and a pinned staging buffer (`whole_direct`).
    direct: Vec<Option<File>>,
    staging: std::cell::RefCell<Option<Tensor>>,
}

/// Bytes per O_DIRECT read of the shard loader (DS41_LOAD_CHUNK_MB, 64).
fn chunk_bytes() -> usize { std::env::var("DS41_LOAD_CHUNK_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(64) << 20 }
const ALIGN: usize = 4096;
#[cfg(target_arch = "aarch64")] pub(super) const O_DIRECT: i32 = 0o200000;
#[cfg(not(target_arch = "aarch64"))] pub(super) const O_DIRECT: i32 = 0o40000;

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

fn elt(dtype: &str) -> usize {
    match dtype { "F32" | "I32" => 4, "BF16" | "F16" => 2, "F8_E4M3" | "F8_E8M0" | "I8" | "U8" => 1, "I64" => 8, d => panic!("dtype {d}") }
}

impl Checkpoint {
    pub fn open(dir: &Path, dev: Device) -> Self {
        let idx: Value = serde_json::from_reader(File::open(dir.join("model.safetensors.index.json")).expect("index")).unwrap();
        let mut files: Vec<String> = idx["weight_map"].as_object().unwrap().values().map(|v| v.as_str().unwrap().to_string()).collect();
        files.sort();
        files.dedup();
        let meta = &idx["metadata"]["ds41_shard"];
        let mut ck = Checkpoint { dir: dir.into(), maps: vec![], files: vec![], tensors: HashMap::new(), dev, sliced: meta.is_object(),
                                  source: meta["source"].as_str().map(PathBuf::from), record: Default::default(),
                                  direct: vec![], staging: Default::default() };
        // shards go to the GPU through O_DIRECT reads (page-cache reads of the mmap ran at ~0.5 GB/s, O_DIRECT ~10 GB/s)
        let use_direct = ck.sliced && dev.is_cuda() && std::env::var("DS41_DIRECT_LOAD").map_or(true, |v| v != "0");
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
            ck.files.push(path.clone());
            ck.direct.push(if use_direct {
                use std::os::unix::fs::OpenOptionsExt;
                std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(&path).ok()
            } else { None });
        }
        ck
    }
    pub fn has(&self, k: &str) -> bool { self.tensors.contains_key(k) }
    /// The files holding none of the tensors `keep`.
    pub fn files_without(&self, keep: &[String]) -> Vec<PathBuf> {
        let held: std::collections::HashSet<usize> = keep.iter().filter_map(|k| self.tensors.get(k)).map(|t| t.file).collect();
        self.files.iter().enumerate().filter(|(i, _)| !held.contains(i)).map(|(_, p)| p.clone()).collect()
    }
    pub fn info(&self, k: &str) -> &TensorRef { self.tensors.get(k).unwrap_or_else(|| panic!("missing tensor {k}")) }
    pub fn bytes(&self, k: &str) -> &[u8] { let t = self.info(k); &self.maps[t.file][t.offset..t.offset + t.len] }
    /// The file holding tensor `k` and the tensor's byte offset in it (explicit reads instead of the mapping).
    pub fn file_of(&self, k: &str) -> (&Path, usize) { let t = self.info(k); (&self.files[t.file], t.offset) }
    /// madvise the shard holding `k` as random access (engram tables: row lookups, no readahead).
    pub fn advise_random(&self, k: &str) {
        let t = self.info(k);
        let _ = self.maps[t.file].advise(memmap2::Advice::Random);
    }

    pub fn start_record(&self) { *self.record.borrow_mut() = Some(Vec::new()); }
    pub fn take_record(&self) -> Vec<(String, Tensor)> { self.record.borrow_mut().take().unwrap_or_default() }
    fn keep(&self, k: &str, t: Tensor) -> Tensor {
        if let Some(r) = self.record.borrow_mut().as_mut() { r.push((k.to_string(), t.shallow_clone())); }
        t
    }

    /// Rows [r0, r0+n) of a 2-D (or 1-D) tensor as raw bytes on the device, viewed as `kind` with shape [n, cols].
    pub fn rows(&self, k: &str, r0: i64, n: i64, kind: Kind) -> Tensor {
        let t = self.info(k);
        if self.sliced {
            assert_eq!(t.shape[0], n, "{k}: shard holds {} rows, {n} requested", t.shape[0]);
            return self.whole(k, kind);
        }
        let row_bytes = t.len / t.shape[0] as usize;
        assert!(r0 >= 0 && r0 + n <= t.shape[0], "{k}: rows {r0}+{n} of {}", t.shape[0]);
        let src = &self.maps[t.file][t.offset + r0 as usize * row_bytes..t.offset + (r0 + n) as usize * row_bytes];
        let cols = (row_bytes / elt(&t.dtype)) as i64;
        let out = upload(src, &[n, cols], &t.dtype, kind, self.dev);
        let out = if t.shape.len() == 1 { out.view([n]) } else { out };
        self.keep(k, out)
    }
    /// The stored tensor in full (shard path).
    fn whole(&self, k: &str, kind: Kind) -> Tensor {
        let t = self.info(k);
        if let Some(Some(f)) = self.direct.get(t.file) {
            if t.len > 0 { return from_raw(self.read_direct(f, t.offset, t.len), &t.shape, &t.dtype, kind); }
        }
        let src = &self.maps[t.file][t.offset..t.offset + t.len];
        upload(src, &t.shape, &t.dtype, kind, self.dev)
    }
    /// Whole tensor (`rows` over all rows), shape as stored.
    pub fn get(&self, k: &str, kind: Kind) -> Tensor {
        if self.sliced { return self.whole(k, kind); }
        let t = self.info(k);
        let src = &self.maps[t.file][t.offset..t.offset + t.len];
        let out = upload(src, &t.shape, &t.dtype, kind, self.dev);
        self.keep(k, out)
    }
    /// Columns [c0, c0+n) (in stored elements) of a 2-D tensor, gathered row by row on the host.
    pub fn cols(&self, k: &str, c0: i64, n: i64, kind: Kind) -> Tensor {
        let t = self.info(k);
        if self.sliced {
            assert_eq!(t.shape[1], n, "{k}: shard holds {} columns, {n} requested", t.shape[1]);
            return self.whole(k, kind);
        }
        assert_eq!(t.shape.len(), 2, "{k}: column slice of a non-matrix");
        let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        assert!(c0 >= 0 && (c0 + n) as usize <= cols, "{k}: cols {c0}+{n} of {cols}");
        let e = elt(&t.dtype);
        let src = &self.maps[t.file][t.offset..t.offset + t.len];
        let mut buf = vec![0u8; rows * n as usize * e];
        let (a, w) = (c0 as usize * e, n as usize * e);
        for r in 0..rows { buf[r * w..(r + 1) * w].copy_from_slice(&src[r * cols * e + a..r * cols * e + a + w]); }
        let out = upload(&buf, &[rows as i64, n], &t.dtype, kind, self.dev);
        self.keep(k, out)
    }
}

impl Checkpoint {
    /// Bytes [offset, offset + len) of a file into a new device Uint8 tensor: aligned O_DIRECT reads into the pinned
    /// staging buffer, each copied to its place on the device.
    fn read_direct(&self, f: &File, offset: usize, len: usize) -> Tensor {
        use std::os::unix::fs::FileExt;
        let chunk = chunk_bytes();
        let mut st = self.staging.borrow_mut();
        let staging = st.get_or_insert_with(|| Tensor::empty([(chunk + 2 * ALIGN) as i64], (Kind::Uint8, Device::Cpu)).pin_memory(self.dev));
        let buf = unsafe { std::slice::from_raw_parts_mut(staging.data_ptr() as *mut u8, chunk + 2 * ALIGN) };
        let dst = Tensor::empty([len as i64], (Kind::Uint8, self.dev));
        let mut done = 0usize;
        while done < len {
            let pos = offset + done;
            let a = pos & !(ALIGN - 1);
            let head = pos - a;
            let want = (chunk - head).min(len - done);
            let span = (head + want).div_ceil(ALIGN) * ALIGN;
            let mut got = 0usize;
            while got < head + want {
                let n = f.read_at(&mut buf[got..span], (a + got) as u64).expect("O_DIRECT shard read");
                if n == 0 { panic!("shard read past the end of the file"); }
                got += n;
            }
            let _ = dst.narrow(0, done as i64, want as i64).copy_(&staging.narrow(0, head as i64, want as i64));
            done += want;
        }
        dst
    }
}

/// Host bytes -> device tensor of the stored dtype, then reinterpreted as `kind` (same element size) or converted
/// (BF16/F32 sources to a wider/narrower float kind).
fn upload(src: &[u8], shape: &[i64], dtype: &str, kind: Kind, dev: Device) -> Tensor {
    from_raw(Tensor::from_slice(src).to_device(dev), shape, dtype, kind)
}
/// Raw bytes (a Uint8 tensor) viewed as the stored dtype and shape, then as `kind`.
fn from_raw(raw: Tensor, shape: &[i64], dtype: &str, kind: Kind) -> Tensor {
    let stored = match dtype {
        "BF16" => raw.view_dtype(Kind::BFloat16),
        "F16" => raw.view_dtype(Kind::Half),
        "F32" => raw.view_dtype(Kind::Float),
        "F8_E4M3" => raw.view_dtype(Kind::Float8e4m3fn),
        "F8_E8M0" | "U8" => raw,
        "I8" => raw.view_dtype(Kind::Int8),
        "I32" => raw.view_dtype(Kind::Int),
        "I64" => raw.view_dtype(Kind::Int64),
        d => panic!("dtype {d}"),
    }.reshape(shape);
    if stored.kind() == kind { return stored; }
    match (stored.kind(), kind) {
        (Kind::Uint8 | Kind::Int8 | Kind::Float8e4m3fn, Kind::Uint8 | Kind::Int8 | Kind::Float8e4m3fn) => stored.view_dtype(kind),
        (Kind::BFloat16 | Kind::Float | Kind::Half, Kind::BFloat16 | Kind::Float | Kind::Half) => stored.to_kind(kind),
        (a, b) => panic!("cannot load {a:?} as {b:?}"),
    }
}

pub use super::ops::e8m0;

/// FP8 E4M3 weight [n, k] with E8M0 scales [ceil(n/32), ceil(k/32)] -> exact BF16 [n, k].
pub fn dequant_fp8(w: &Tensor, s: &Tensor) -> Tensor {
    let (n, k) = (w.size()[0], w.size()[1]);
    let sc = e8m0(s).repeat_interleave_self_int(32, 0, None).narrow(0, 0, n).repeat_interleave_self_int(32, 1, None).narrow(1, 0, k);
    (w.to_kind(Kind::Float) * sc).to_kind(Kind::BFloat16)
}

/// E2M1 magnitudes by code (bit 3 = sign).
pub fn fp4_table(dev: Device) -> Tensor {
    Tensor::from_slice(&[0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).to_device(dev)
}
/// MXFP4 [n, k/2] packed (Uint8, low nibble first) with E8M0 scales [n, k/32] -> `kind` [n, k] (exact in BF16).
pub fn dequant_mxfp4(w: &Tensor, s: &Tensor, table: &Tensor, kind: Kind) -> Tensor {
    let n = w.size()[0];
    let lo = w.bitwise_and(15).to_kind(Kind::Int64);
    let hi = w.bitwise_right_shift_tensor_scalar(4).to_kind(Kind::Int64);
    let v = Tensor::stack(&[table.index_select(0, &lo.view(-1)).view_as(&lo), table.index_select(0, &hi.view(-1)).view_as(&hi)], -1).view([n, -1]);
    let k = v.size()[1];
    (v.view([n, k / 32, 32]) * e8m0(s).unsqueeze(-1)).view([n, k]).to_kind(kind)
}
