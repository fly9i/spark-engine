// SPDX-License-Identifier: MIT
//! Serving KV pool (GLM53_KV_POOL=1, sized by GLM53_MEM_UTIL). Like vLLM's gpu_memory_utilization: after the
//! weights are resident, the context-proportional state of every sequence store (per MLA layer: the latent rows
//! and the DSA index pool rows) is allocated once, as large as the memory target allows, and stores are carved
//! out of it in granule-aligned row ranges. The KV budget in tokens follows from the memory, the footprint is
//! fixed from startup, and a machine that cannot hold it fails at startup rather than under load.
//!
//! A store holds a [`Lease`]; dropping the store returns its range. Carved ranges are zero-filled, like the
//! freshly allocated state they replace, so pooled and non-pooled stores hold the same values.

use std::cell::RefCell;
use std::rc::Rc;

use tch::{Device, Kind, Tensor};

pub(crate) fn enabled() -> bool { std::env::var("GLM53_KV_POOL").as_deref() == Ok("1") }

/// Per MLA layer: latent rows [rows, width] and DSA index pool rows [rows/4, dim] (FP32).
pub(crate) struct KvPool {
    pub rows: i64,
    pub granule: i64,
    latent: Vec<Tensor>,
    pools: Vec<Tensor>,
    free: RefCell<Vec<(i64, i64)>>, // (start, len) in rows, sorted by start, coalesced
}

pub(crate) struct Lease { pool: Rc<KvPool>, pub start: i64, pub len: i64 }
impl Drop for Lease {
    fn drop(&mut self) { self.pool.release(self.start, self.len); }
}

thread_local! { static POOL: RefCell<Option<Rc<KvPool>>> = const { RefCell::new(None) }; }
pub(crate) fn installed() -> Option<Rc<KvPool>> { POOL.with(|p| p.borrow().clone()) }

impl KvPool {
    /// `layers`: (latent width, latent kind, DSA dim) of each MLA layer in state order.
    pub fn new(dev: Device, layers: &[(i64, Kind, i64)], rows: i64, granule: i64) -> Rc<Self> {
        assert!(granule % 4 == 0 && rows % granule == 0 && rows > 0, "KV pool rows must be whole granules");
        let latent = layers.iter().map(|&(w, k, _)| Tensor::zeros([rows, w], (k, dev))).collect();
        let pools = layers.iter().map(|&(_, _, d)| Tensor::zeros([rows / 4, d], (Kind::Float, dev))).collect();
        Rc::new(Self { rows, granule, latent, pools, free: RefCell::new(vec![(0, rows)]) })
    }
    pub fn install(p: Rc<Self>) { POOL.with(|s| *s.borrow_mut() = Some(p)); }

    /// Bytes per token of all layers (latent row + a quarter DSA pool row).
    pub fn bytes_per_token(layers: &[(i64, Kind, i64)]) -> i64 {
        layers.iter().map(|&(w, k, d)| w * k.elt_size_in_bytes() as i64 + d).sum()
    }

    /// Largest contiguous free range, in rows.
    pub fn largest_free(&self) -> i64 { self.free.borrow().iter().map(|r| r.1).max().unwrap_or(0) }
    pub fn fits(&self, cap: i64) -> bool { self.largest_free() >= cap }

    /// First fit; `cap` must be whole granules.
    pub fn alloc(self: &Rc<Self>, cap: i64) -> Option<Lease> {
        assert!(cap > 0 && cap % self.granule == 0, "KV pool leases are whole granules");
        let mut f = self.free.borrow_mut();
        let i = f.iter().position(|r| r.1 >= cap)?;
        let (start, len) = f[i];
        if len == cap { f.remove(i); } else { f[i] = (start + cap, len - cap); }
        Some(Lease { pool: self.clone(), start, len: cap })
    }
    fn release(&self, start: i64, len: i64) {
        let mut f = self.free.borrow_mut();
        let i = f.iter().position(|r| r.0 > start).unwrap_or(f.len());
        f.insert(i, (start, len));
        // Coalesce with the neighbours.
        if i + 1 < f.len() && f[i].0 + f[i].1 == f[i + 1].0 { f[i].1 += f[i + 1].1; f.remove(i + 1); }
        if i > 0 && f[i - 1].0 + f[i - 1].1 == f[i].0 { f[i - 1].1 += f[i].1; f.remove(i); }
    }

    /// Zero-filled views of MLA layer `layer`'s rows for `lease`: (latent [len, width], DSA pools [len/4, dim]).
    pub fn views(&self, layer: usize, lease: &Lease) -> (Tensor, Tensor) {
        let mut l = self.latent[layer].narrow(0, lease.start, lease.len);
        let mut p = self.pools[layer].narrow(0, lease.start / 4, lease.len / 4);
        let _ = l.zero_();
        let _ = p.zero_();
        (l, p)
    }
    pub fn free_rows(&self) -> i64 { self.free.borrow().iter().map(|r| r.1).sum() }
}

/// MemTotal and MemAvailable in bytes.
pub(crate) fn meminfo() -> (i64, i64) {
    let m = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let get = |k: &str| m.lines().find_map(|l| l.strip_prefix(k)).and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<i64>().ok()).map_or(0, |kb| kb * 1024);
    (get("MemTotal:"), get("MemAvailable:"))
}
