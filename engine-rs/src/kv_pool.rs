// SPDX-License-Identifier: MIT
//! Serving KV pool (GLM: GLM53_KV_POOL=1, sized by GLM53_MEM_UTIL; DeepSeek-V4.1 serving: always, VMM, sized by the
//! memory left after the warm-up, ds41/serve.rs). Like vLLM's gpu_memory_utilization: after the
//! weights are resident, the context-proportional state of every sequence store (per MLA layer: the latent rows
//! and the DSA index pool rows) is allocated once, as large as the memory target allows, and stores are carved
//! out of it in granule-aligned row ranges. The KV budget in tokens follows from the memory, the footprint is
//! fixed from startup, and a machine that cannot hold it fails at startup rather than under load.
//!
//! A store holds a [`Lease`]; dropping the store returns its range. Carved ranges are zero-filled, like the
//! freshly allocated state they replace, so pooled and non-pooled stores hold the same values.
//!
//! Backends:
//! - contiguous (default): one tensor per layer and kind; a lease is a contiguous row range (first fit). A store
//!   cannot grow, and free space can be fragmented.
//! - VMM (GLM53_KV_VMM=1, shim/vmm.cpp): a fixed set of store slots; every slot reserves, per layer and kind, the
//!   virtual range of the whole budget, and physical chunks of one shared pool are mapped behind the rows a lease
//!   uses. A lease grows in place ([`KvPool::grow`]: the rows it had keep their address and contents), any store can
//!   use the whole budget, and there is no fragmentation: the only limit is the number of free chunks.

use std::cell::RefCell;
use std::rc::Rc;

use tch::{Device, Kind, Tensor};

pub(crate) fn enabled() -> bool { std::env::var("GLM53_KV_POOL").as_deref() == Ok("1") }
pub(crate) fn vmm_enabled() -> bool { std::env::var("GLM53_KV_VMM").as_deref() == Ok("1") }

extern "C" {
    fn rs_vmm_init(dev: i32, min_chunk_bytes: u64, chunk_bytes: *mut u64) -> i32;
    fn rs_vmm_create(chunks: i64) -> i32;
    fn rs_vmm_region(bytes: u64, base: *mut u64) -> i64;
    fn rs_vmm_resize(id: i64, bytes: u64) -> i32;
    fn rs_vmm_free_chunks() -> i64;
}

/// One per-token tensor of a store: `width` elements of `kind` per row, one row per `div` tokens (GLM: an MLA layer's
/// latent rows, div 1, and its DSA pool rows, div 4, FP32; DeepSeek-V4.1: each KV source's compressed KV and index
/// keys, div = its compression ratio).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Arr { pub div: i64, pub width: i64, pub kind: Kind }
impl Arr {
    fn bytes(&self, rows: i64) -> i64 { rows / self.div * self.width * self.kind.elt_size_in_bytes() as i64 }
}

/// VMM backend: per slot, per array, the (region id, full-budget tensor).
struct Vmm {
    chunk: i64,
    created: i64,
    slots: Vec<Vec<(i64, Tensor)>>,
    used: RefCell<Vec<bool>>,
}

/// The per-token tensors of every store, carved from one budget of `rows` tokens.
pub(crate) struct KvPool {
    pub rows: i64,
    pub granule: i64,
    arrs: Vec<Arr>,
    whole: Vec<Tensor>,                // contiguous backend: one tensor per array
    free: RefCell<Vec<(i64, i64)>>,    // (start, len) in rows, sorted by start, coalesced (contiguous backend)
    vmm: Option<Vmm>,
}

/// A store's rows. Contiguous backend: rows `start..start+len` of the pool tensors; VMM backend: rows `0..len` of
/// slot `slot` (`start` = 0).
pub(crate) struct Lease { pool: Rc<KvPool>, pub start: i64, pub len: i64, slot: usize }
impl Drop for Lease {
    fn drop(&mut self) { self.pool.release(self); }
}

thread_local! { static POOL: RefCell<Option<Rc<KvPool>>> = const { RefCell::new(None) }; }
pub(crate) fn installed() -> Option<Rc<KvPool>> { POOL.with(|p| p.borrow().clone()) }

/// GLM's per MLA layer (latent width, latent kind, DSA dim) as arrays: latent rows, then DSA pool rows (FP32, div 4).
fn glm_arrs(layers: &[(i64, Kind, i64)]) -> Vec<Arr> {
    layers.iter().flat_map(|&(w, k, d)| [Arr { div: 1, width: w, kind: k }, Arr { div: 4, width: d, kind: Kind::Float }]).collect()
}

impl KvPool {
    /// GLM: `layers`: (latent width, latent kind, DSA dim) of each MLA layer in state order. `slots`: stores (VMM only).
    pub fn new(dev: Device, layers: &[(i64, Kind, i64)], rows: i64, granule: i64, slots: usize) -> Rc<Self> {
        let min_chunk = std::env::var("GLM53_KV_VMM_CHUNK_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(2) << 20;
        Self::with_arrays(dev, &glm_arrs(layers), rows, granule, slots, vmm_enabled(), min_chunk)
    }
    /// Any model: the per-token arrays, `rows` tokens of budget in granules of `granule` tokens (a multiple of every
    /// array's div), `slots` stores (VMM), VMM backend or contiguous, VMM chunks of at least `min_chunk` bytes.
    pub fn with_arrays(dev: Device, arrs: &[Arr], rows: i64, granule: i64, slots: usize, vmm: bool, min_chunk: u64) -> Rc<Self> {
        assert!(rows % granule == 0 && rows > 0 && arrs.iter().all(|a| granule % a.div == 0), "KV pool rows must be whole granules");
        if vmm {
            let Device::Cuda(d) = dev else { panic!("the VMM KV pool needs a CUDA device") };
            // Chunks: the whole budget in one store, each per-array tensor rounded up to whole chunks, plus that
            // rounding for every further slot so that `slots` stores can always map `rows` rows between them.
            let mut chunk = 0u64;
            assert_eq!(unsafe { rs_vmm_init(d as i32, min_chunk, &mut chunk) }, 0, "KV pool: VMM init");
            let need = Self::chunks_of(arrs, rows, chunk as i64) + arrs.len() as i64 * (slots as i64 - 1).max(0);
            assert_eq!(unsafe { rs_vmm_create(need) }, 0, "KV pool: VMM create {need} chunks");
            let slots_v = (0..slots).map(|_| arrs.iter().map(|a| {
                let shape = [rows / a.div, a.width];
                let mut base = 0u64;
                let id = unsafe { rs_vmm_region(a.bytes(rows) as u64, &mut base) };
                assert!(id >= 0, "KV pool: VMM reserve");
                // torch checks at creation that the pointer belongs to the device: map one chunk for that moment.
                assert_eq!(unsafe { rs_vmm_resize(id, 1) }, 0, "KV pool: VMM probe map");
                let t = unsafe { Tensor::from_blob(base as *const u8, &shape, &[shape[1], 1], a.kind, dev) };
                assert_eq!(unsafe { rs_vmm_resize(id, 0) }, 0, "KV pool: VMM probe unmap");
                (id, t)
            }).collect()).collect();
            return Rc::new(Self { rows, granule, arrs: arrs.to_vec(), whole: Vec::new(), free: RefCell::new(Vec::new()),
                vmm: Some(Vmm { chunk: chunk as i64, created: need, slots: slots_v, used: RefCell::new(vec![false; slots]) }) });
        }
        let whole = arrs.iter().map(|a| Tensor::zeros([rows / a.div, a.width], (a.kind, dev))).collect();
        Rc::new(Self { rows, granule, arrs: arrs.to_vec(), whole, free: RefCell::new(vec![(0, rows)]), vmm: None })
    }
    pub fn install(p: Rc<Self>) { POOL.with(|s| *s.borrow_mut() = Some(p)); }
    /// VMM: the physical memory of the pool (every chunk created at startup), 0 for the contiguous backend.
    pub fn vmm_bytes(&self) -> i64 { self.vmm.as_ref().map_or(0, |v| v.created * v.chunk) }

    /// GLM: bytes per token of all layers (latent row + a quarter DSA pool row).
    pub fn bytes_per_token(layers: &[(i64, Kind, i64)]) -> i64 {
        layers.iter().map(|&(w, k, d)| w * k.elt_size_in_bytes() as i64 + d).sum()
    }
    /// Bytes per token of `arrs` (fractional rows of div > 1 arrays rounded down per token; exact per granule).
    pub fn arr_bytes_per_granule(arrs: &[Arr], granule: i64) -> i64 { arrs.iter().map(|a| a.bytes(granule)).sum() }
    /// Chunks that `rows` rows take in one slot (every per-array tensor rounded up to whole chunks).
    fn chunks_of(arrs: &[Arr], rows: i64, chunk: i64) -> i64 {
        arrs.iter().map(|a| (a.bytes(rows) + chunk - 1) / chunk).sum()
    }
    fn vmm_rows_free(&self, v: &Vmm) -> i64 {
        // Largest whole-granule row count whose chunks fit in the free pool.
        let free = unsafe { rs_vmm_free_chunks() };
        let (mut lo, mut hi) = (0i64, self.rows / self.granule);
        while lo < hi { let m = (lo + hi + 1) / 2; if Self::chunks_of(&self.arrs, m * self.granule, v.chunk) <= free { lo = m } else { hi = m - 1 } }
        lo * self.granule
    }

    /// Largest store, in rows, that can be allocated now (contiguous: largest free range; VMM: free chunks, 0 when
    /// every slot is taken).
    pub fn largest_free(&self) -> i64 {
        match &self.vmm {
            Some(v) => if v.used.borrow().iter().all(|&u| u) { 0 } else { self.vmm_rows_free(v) },
            None => self.free.borrow().iter().map(|r| r.1).max().unwrap_or(0),
        }
    }
    pub fn fits(&self, cap: i64) -> bool { self.largest_free() >= cap }
    /// Whether `lease` can grow in place (VMM only).
    pub fn growable(&self) -> bool { self.vmm.is_some() }
    /// Whether `lease` can grow to `cap` rows now.
    pub fn can_grow(&self, lease: &Lease, cap: i64) -> bool {
        let Some(v) = &self.vmm else { return false };
        cap <= self.rows && Self::chunks_of(&self.arrs, cap, v.chunk) - Self::chunks_of(&self.arrs, lease.len, v.chunk) <= unsafe { rs_vmm_free_chunks() }
    }

    /// First fit; `cap` must be whole granules.
    pub fn alloc(self: &Rc<Self>, cap: i64) -> Option<Lease> {
        assert!(cap > 0 && cap % self.granule == 0, "KV pool leases are whole granules");
        if let Some(v) = &self.vmm {
            if !self.fits(cap) { return None; }
            let slot = v.used.borrow().iter().position(|&u| !u)?;
            self.map(v, slot, cap);
            v.used.borrow_mut()[slot] = true;
            return Some(Lease { pool: self.clone(), start: 0, len: cap, slot });
        }
        let mut f = self.free.borrow_mut();
        let i = f.iter().position(|r| r.1 >= cap)?;
        let (start, len) = f[i];
        if len == cap { f.remove(i); } else { f[i] = (start + cap, len - cap); }
        Some(Lease { pool: self.clone(), start, len: cap, slot: 0 })
    }
    fn map(&self, v: &Vmm, slot: usize, cap: i64) {
        for (a, t) in self.arrs.iter().zip(&v.slots[slot]) {
            assert_eq!(unsafe { rs_vmm_resize(t.0, a.bytes(cap) as u64) }, 0, "KV pool: map rows");
        }
    }
    /// VMM: grow `lease` in place to `cap` rows (rows below the old length keep address and contents; the new rows
    /// are zero-filled). The caller rebuilds the store's views ([`KvPool::views_of`]) and drops graphs sized by the
    /// old capacity. Returns false (nothing changed) when the free chunks do not suffice.
    pub fn grow(&self, lease: &mut Lease, cap: i64) -> bool {
        assert!(cap % self.granule == 0 && cap >= lease.len, "KV pool grows by whole granules");
        let Some(v) = &self.vmm else { return false };
        if !self.can_grow(lease, cap) { return false; }
        self.map(v, lease.slot, cap);
        for (a, t) in self.arrs.iter().zip(&v.slots[lease.slot]) {
            let _ = t.1.narrow(0, lease.len / a.div, (cap - lease.len) / a.div).zero_();
        }
        lease.len = cap;
        true
    }
    /// VMM: shrink `lease` to `cap` rows (> 0, whole granules), giving the chunks past it back to the pool. Nothing may
    /// still read the dropped rows (this drains the device first).
    pub fn shrink(&self, lease: &mut Lease, cap: i64) {
        assert!(cap > 0 && cap % self.granule == 0 && cap <= lease.len, "KV pool shrinks to whole granules");
        let Some(v) = &self.vmm else { return };
        extern "C" { fn cudaDeviceSynchronize() -> i32; }
        assert_eq!(unsafe { cudaDeviceSynchronize() }, 0, "KV pool: synchronize before unmapping");
        self.map(v, lease.slot, cap);
        lease.len = cap;
    }
    fn release(&self, lease: &Lease) {
        if let Some(v) = &self.vmm {
            // Nothing may still read the rows being unmapped (the caller fenced background writes; this drains kernels).
            extern "C" { fn cudaDeviceSynchronize() -> i32; }
            assert_eq!(unsafe { cudaDeviceSynchronize() }, 0, "KV pool: synchronize before unmapping");
            self.map(v, lease.slot, 0);
            v.used.borrow_mut()[lease.slot] = false;
            return;
        }
        let (start, len) = (lease.start, lease.len);
        let mut f = self.free.borrow_mut();
        let i = f.iter().position(|r| r.0 > start).unwrap_or(f.len());
        f.insert(i, (start, len));
        // Coalesce with the neighbours.
        if i + 1 < f.len() && f[i].0 + f[i].1 == f[i + 1].0 { f[i].1 += f[i + 1].1; f.remove(i + 1); }
        if i > 0 && f[i - 1].0 + f[i - 1].1 == f[i].0 { f[i - 1].1 += f[i].1; f.remove(i); }
    }

    /// Views of every array's rows for `lease` (array order), not zeroed: [len / div, width] each.
    pub fn arr_views(&self, lease: &Lease) -> Vec<Tensor> {
        self.arrs.iter().enumerate().map(|(i, a)| match &self.vmm {
            Some(v) => v.slots[lease.slot][i].1.narrow(0, 0, lease.len / a.div),
            None => self.whole[i].narrow(0, lease.start / a.div, lease.len / a.div),
        }).collect()
    }
    /// GLM: views of MLA layer `layer`'s rows for `lease`: (latent [len, width], DSA pools [len/4, dim]), not zeroed.
    pub fn views_of(&self, layer: usize, lease: &Lease) -> (Tensor, Tensor) {
        let view = |i: usize| {
            let a = &self.arrs[i];
            match &self.vmm {
                Some(v) => v.slots[lease.slot][i].1.narrow(0, 0, lease.len / a.div),
                None => self.whole[i].narrow(0, lease.start / a.div, lease.len / a.div),
            }
        };
        (view(2 * layer), view(2 * layer + 1))
    }
    /// GLM: zero-filled views of MLA layer `layer`'s rows for `lease`: (latent [len, width], DSA pools [len/4, dim]).
    pub fn views(&self, layer: usize, lease: &Lease) -> (Tensor, Tensor) {
        let (mut l, mut p) = self.views_of(layer, lease);
        let _ = l.zero_();
        let _ = p.zero_();
        (l, p)
    }
    /// Rows still free (contiguous: sum of free ranges; VMM: as `largest_free`).
    pub fn free_rows(&self) -> i64 {
        match &self.vmm { Some(v) => self.vmm_rows_free(v), None => self.free.borrow().iter().map(|r| r.1).sum() }
    }
}

/// MemTotal and MemAvailable in bytes.
pub(crate) fn meminfo() -> (i64, i64) {
    let m = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let get = |k: &str| m.lines().find_map(|l| l.strip_prefix(k)).and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<i64>().ok()).map_or(0, |kb| kb * 1024);
    (get("MemTotal:"), get("MemAvailable:"))
}
