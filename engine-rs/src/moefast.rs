// SPDX-License-Identifier: MIT
//! mgemm 池 v2:全局槽位 + 指针表。
//! 语义(2026-09-21 两次实证修正,vs exllamav3_ext 反汇编签名+内核文档):
//! - sel 值 = 指针表的**槽位索引**(非专家 ID),int64 [1,k]
//! - **K 参数 = trellis 词组数 shape[-1]/16 = 4**(内核模板索引,与专家数无关;
//!   误传专家数=8 → rel≈√2 垃圾,误传 in_features → No kernel)
//! - 指针表按投影分 9 张(gate/up/down × tr/suh/svh);共用一张会被 last-writer-wins
//!   污染成只指向 down_proj(rel=54.5 垃圾)
//! - A 3-D:共享 x 用 [1,1,K];down 逐专家行用 [k,1,K]
//! - suh/svh 表项指向 fp16 尺度;mcg=1/mul1=0;rows=-1 自动
//! - 指针表指向的张量必须**永久保活**;单选 k=1 路径有内核怪癖,固定用 ≥2

use std::collections::HashMap;

use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

use crate::safetensors::ShardIndex;

use crate::moe::SWIGLU_LIMIT;

extern "C" {
    fn rs_exl3_mgemm_probe(a: *const std::ffi::c_void, c: *mut std::ffi::c_void, yh: *mut std::ffi::c_void,
        tr: *const std::ffi::c_void, sh: *const std::ffi::c_void, sv: *const std::ffi::c_void, sel: *const std::ffi::c_void,
        k: i64, ki: i64, no: i64, rows: i64, cap: i64, fp32: i32) -> i32;
    fn rs_exl3_mgemm2(a_p: *const u8, c_p: *mut u8, yh_p: *mut u8,
                      ptrtr_p: *const u8, ptrsuh_p: *const u8, ptrsvh_p: *const u8,
                      sel_p: *const u8, k_sel: i64, kk: i64, nn: i64,
                      rows_a: i64, kt: i64, nt: i64, cap: i64) -> i32;
}

/// 指针表布局:每投影一组 (tr, suh, svh) — 9 张表。
/// 修复 2026-09-21:此前 gate/up/down 共用一组表,ensure 循环内对同一 slot
/// 连写 3 次,last-writer-wins → slot 只指向 down_proj,gate/up GEMM 读错权重
/// (fast-smoke rel = 54.5 实证)。每投影独立一组表后各自指向正确权重。
const PTR_TABLES: usize = 9; // [proj(3)][tr, suh, svh]

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ScratchMode { Zeros, Empty, Reuse }

impl ScratchMode {
    pub fn label(self) -> &'static str {
        match self { Self::Zeros => "zeros", Self::Empty => "empty", Self::Reuse => "reuse" }
    }
    fn from_env() -> Self {
        match std::env::var("GLM53_MOE_SCRATCH").as_deref().unwrap_or("zeros") {
            "zeros" => Self::Zeros, "empty" => Self::Empty, "reuse" => Self::Reuse,
            other => panic!("invalid GLM53_MOE_SCRATCH={other}; expected zeros/empty/reuse"),
        }
    }
}

struct Scratch {
    key: (i64, i64, u8, Device),
    buffers: [Tensor; 5], // yh, yh_down, gate, up, down
}

/// Pool-owned proof for the optional shared gate/up input transform. Validation
/// uses checkpoint bytes before upload; no GPU readback or offline report is
/// trusted. Raw f32 equality is deliberately stricter than equal Half rounding.
struct SharedInputCertificate {
    canonical: Option<Vec<u32>>,
    seen: Vec<bool>,
    distinct: usize,
    matching: bool,
}
impl SharedInputCertificate {
    fn new(experts:usize)->Self {
        Self{canonical:None,seen:vec![false;experts],distinct:0,matching:true}
    }
    fn observe(&mut self,expert:usize,gate:&[f32],up:&[f32]) {
        assert!(expert<self.seen.len(),"shared-input expert ID out of range");
        if !self.seen[expert] {self.seen[expert]=true;self.distinct+=1;}
        // Equal finite source bits imply equal Half bits. Also reject finite
        // f32 values that would overflow the actual Half scale on upload.
        let pair_matches=gate.len()==4096 && up.len()==gate.len() && gate.iter().zip(up).all(|(&g,&u)|
            g.is_finite() && u.is_finite() && half::f16::from_f32(g).is_finite() && g.to_bits()==u.to_bits());
        if !pair_matches {self.matching=false;return;}
        if let Some(canonical)=&self.canonical {
            if !gate.iter().zip(canonical).all(|(&g,&bits)|g.to_bits()==bits) {self.matching=false;}
        } else {self.canonical=Some(gate.iter().map(|v|v.to_bits()).collect());}
        // A mismatch permanently invalidates this layer in this pool, even if
        // a later reload happens to restore matching scales.
    }
    fn verified(&self,resident:usize)->bool {
        self.seen.len()==288 && self.distinct==288 && resident==288 && self.matching && self.canonical.is_some()
    }
}

pub struct MoeFast {
    idx: ShardIndex,
    ptrs: [Tensor; PTR_TABLES], // [cap] i64 × 9
    alive: Vec<Option<Vec<Tensor>>>, // slot → 9 张量保活(tr,suh,svh × 3 投影)
    owner: Vec<Option<(usize, usize)>>, // slot → (layer, e)(逐出时清 slot_table 用)
    slot_table: Tensor, // [n_layers, n_experts] i64 设备侧选路表,-1 = 未驻留
    where_: HashMap<(usize, usize), usize>, // (layer, e) → slot
    lru: std::collections::VecDeque<usize>,
    free: Vec<usize>,
    pub cap: usize,
    pub misses: u64, // ensure 冷装载计数(诊断用)
    pub assume_hot: bool, // 图模式:跳过缺失检查(回放中 miss 会静默算错,勿开除非全驻留)
    /// Diagnostic C dtype: 0=native, 1=down FP32, 2=all projections FP32.
    /// C dtype also controls kernel intermediate-sum and post-transform storage.
    pub projection_precision: u8,
    pub(crate) scratch_mode: ScratchMode,
    // Calls on one pool must be serialized on the same CUDA stream. The engine
    // already requires this; graph capture/replay retains these stable addresses.
    scratch: Option<Scratch>,
    own_tables:HashMap<usize,Tensor>,
    shared_input:Vec<SharedInputCertificate>,
    resident_per_layer:Vec<usize>,
    tp: crate::tp::Tp,
    _keep: (Tensor, Tensor), // dummy 张量保活
}

type RawProj = (Vec<i16>, Vec<i64>, Vec<f32>, Vec<f32>); // (trellis, shape, suh, svh)

/// 纯读取(可 rayon 并行):页缓存/NVMe 页入在这里发生。
/// TP 分片(对齐 exllamav3 TP 规则):gate/up 列切(N 半),down 行切(K 半);
/// suh 按输入维切、svh 按输出维切。
pub(crate) fn load_expert_raw(idx: &ShardIndex, layer: usize, e: usize, tp: crate::tp::Tp) -> [RawProj; 3] {
    let p = format!("model.language_model.layers.{layer}.mlp.experts.{e}");
    let (rank, world) = (tp.rank, tp.world);
    return ["gate_proj", "up_proj", "down_proj"].map(|w| {
        let tr = idx.get_i16(&format!("{p}.{w}.trellis")).expect("tr");
        let s = idx.entries[&format!("{p}.{w}.trellis")].shape.clone();
        let (suh, _) = idx.get_f32(&format!("{p}.{w}.suh")).expect("suh");
        let (svh, _) = idx.get_f32(&format!("{p}.{w}.svh")).expect("svh");
        if world == 1 {
            return (tr, s.iter().map(|&d| d as i64).collect(), suh, svh);
        }
        let (kt, nt, wd) = (s[0], s[1], s[2]);
        if w == "down_proj" {
            // 行切:K 维(trellis 第 0 维)取半
            let kth = kt / world;
            let lo = rank * kth;
            let tr2 = tr[lo * nt * wd..(lo + kth) * nt * wd].to_vec();
            let suh2 = suh[rank * kth * 16..(rank + 1) * kth * 16].to_vec();
            (tr2, vec![kth as i64, nt as i64, wd as i64], suh2, svh)
        } else {
            // 列切:N 维(trellis 第 1 维)取半
            let nth = nt / world;
            let lo = rank * nth;
            let mut tr2 = Vec::with_capacity(kt * nth * wd);
            for r in 0..kt {
                tr2.extend_from_slice(&tr[(r * nt + lo) * wd..(r * nt + lo + nth) * wd]);
            }
            let svh2 = svh[rank * nth * 16..(rank + 1) * nth * 16].to_vec();
            (tr2, vec![kt as i64, nth as i64, wd as i64], suh, svh2)
        }
    });
}

impl MoeFast {
    pub fn new(dir: &std::path::Path, n_layers: usize, n_experts: usize, cap: usize, dev: Device) -> Self {
        let tp = crate::tp::world();
        Self::new_with_tp(dir, n_layers, n_experts, cap, dev, tp)
    }

    pub(crate) fn new_with_tp(dir: &std::path::Path, n_layers: usize, n_experts: usize, cap: usize, dev: Device, tp: crate::tp::Tp) -> Self {
        let idx = ShardIndex::scan(dir).expect("scan");
        let cap = cap.max(16);
        let dtr = Tensor::zeros([256, 128, 64], (tch::Kind::Int16, dev));
        let ds = Tensor::zeros([4096], (Kind::Half, dev));
        let ptrs: [Tensor; PTR_TABLES] = std::array::from_fn(|i| {
            let dummy = if i % 3 == 0 { &dtr } else { &ds };
            Tensor::full([cap as i64], dummy.data_ptr() as i64, (Kind::Int64, dev))
        });
        Self {
            idx,
            ptrs,
            alive: (0..cap).map(|_| None).collect(),
            owner: (0..cap).map(|_| None).collect(),
            slot_table: Tensor::full([n_layers as i64, n_experts as i64], -1i64, (Kind::Int64, dev)),
            where_: HashMap::new(),
            lru: Default::default(),
            free: (0..cap).rev().collect(),
            cap,
            misses: 0,
            assume_hot: false,
            projection_precision: 0,
            scratch_mode: ScratchMode::from_env(),
            scratch: None,
            own_tables:HashMap::new(),
            shared_input:(0..n_layers).map(|_|SharedInputCertificate::new(n_experts)).collect(),
            resident_per_layer:vec![0;n_layers],
            tp,
            _keep: (dtr, ds),
        }
    }

    /// Diagnostic callers retain this pool until all GPU operations finish.
    pub(crate) fn probe_tables(&self) -> &[Tensor; PTR_TABLES] { &self.ptrs }

    /// O(1) proof tied to this pool's immutable scale tensors and current layer
    /// residency. A local 288-expert probe is sufficient; other layers need not
    /// be loaded. Callers must retain this pool without mutation through replay.
    pub(crate) fn shared_input_verified(&self,layer:usize)->bool {
        self.tp.world==2 && self.shared_input.get(layer).zip(self.resident_per_layer.get(layer))
            .is_some_and(|(proof,&resident)|proof.verified(resident))
    }

    /// Expert tensors of one layer (per expert: gate tr, suh, svh, up tr, suh, svh, down tr, suh, svh).
    pub(crate) fn layer_tensors(&self,layer:usize)->Vec<Vec<Tensor>> {
        (0..).take_while(|e|self.where_.contains_key(&(layer,*e)))
            .map(|e|self.alive[self.where_[&(layer,e)]].as_ref().unwrap().iter().map(Tensor::shallow_clone).collect()).collect()
    }
    /// moe_exl3.cuh pointer table [9, 288] of one layer (cached; the pool is fully resident).
    /// Built for every resident layer on first use, from the host-side pointers of the resident tensors (no device
    /// work: safe before any capture; a miss during capture is an error).
    fn own_table(&mut self,layer:usize)->Tensor {
        if let Some(t)=self.own_tables.get(&layer){return t.shallow_clone();}
        assert!(self.assume_hot,"own MoE requires a fully resident pool");
        assert!(!crate::tp::graph::capturing(),"own MoE table cache miss during graph capture");
        let dev=self.slot_table.device();
        let layers:std::collections::BTreeSet<usize>=self.where_.keys().map(|&(l,_)|l).collect();
        for l in layers {
            let n=(0..).take_while(|e|self.where_.contains_key(&(l,*e))).count();
            assert_eq!(n,288,"own MoE: layer {l} has {n} resident experts");
            let mut host=vec![0i64;9*n];
            for (row,&j) in [0usize,3,6,1,4,7,2,5,8].iter().enumerate() {
                for e in 0..n {host[row*n+e]=self.alive[self.where_[&(l,e)]].as_ref().unwrap()[j].data_ptr() as i64;}
            }
            self.own_tables.insert(l,Tensor::from_slice(&host).view([9,n as i64]).to_device(dev));
        }
        self.own_tables[&layer].shallow_clone()
    }
    /// Prepare the per-layer expert tables before graph capture.
    pub(crate) fn prepare_layer(&mut self,layer:usize) {let _=self.own_table(layer);}
    /// Routed experts of decode / verify rows (moe_exl3.cuh): x [R, 4096] fp16, ids [R, 8], weights [R, 8] -> [R, 4096] FP32.
    pub fn expert_cooperative(&mut self,layer:usize,x:&Tensor,ids:&Tensor,weights:&Tensor)->Tensor {
        let out=Tensor::empty([x.size()[0],4096],(Kind::Float,x.device()));
        crate::moe_own::run_into(&self.own_table(layer),x,ids,weights,&out,None,None);crate::ablate::apply(layer,&out);out
    }
    /// expert_cooperative into a caller-provided contiguous FP32 [rows, 4096] (e.g. the first half of a packed
    /// collective buffer). The kernel assigns every output element, so `out` needs no initialization.
    pub fn expert_cooperative_into(&mut self,layer:usize,x:&Tensor,ids:&Tensor,weights:&Tensor,out:&Tensor) {
        crate::moe_own::run_into(&self.own_table(layer),x,ids,weights,out,None,None);crate::ablate::apply(layer,out);
    }

    /// Prefill-only expert grouping. Host routing transfer happens once per
    /// layer, then each expert projection serves all its assigned token rows.
    /// The scatter writes unique (token,route-slot) rows: no floating atomics.
    pub fn expert_grouped(&mut self,layer:usize,x:&Tensor,ids:&Tensor,weights:&Tensor,reconstruct:bool)->Tensor {
        self.expert_grouped_add(layer,x,ids,weights,reconstruct,None)
    }
    /// `add` (P3b): FP32 [rows,4096] partial added after the fixed-order routed reduction, inside
    /// the reduce kernel (fl(routed+add), same as the separate add). Fat path only.
    pub fn expert_grouped_add(&mut self,layer:usize,x:&Tensor,ids:&Tensor,weights:&Tensor,reconstruct:bool,add:Option<&Tensor>)->Tensor {
        assert!(self.assume_hot);assert_eq!(self.tp.world,2);
        let rows=x.size()[0];assert_eq!(ids.size(),[rows,8]);
        // The grouped GEMM path of moe_exl3.cuh in launches of PREFILL_ROWS rows, the shared partial `add` added after the
        // fixed-order slot sum. GLM53_PREFILL_GROUPED=recon keeps the reconstruct tier (decoded weights + cuBLAS).
        if !reconstruct || std::env::var("GLM53_MOE_RECON").as_deref()!=Ok("1") {
            let tab=self.own_table(layer);
            // shared-input layers: expert 0's gate suh stands for all (proven equal at load)
            let suh0=(self.shared_input_verified(layer)&&std::env::var("GLM53_MOE_NO_SHARED").is_err()).then(||self.alive[self.where_[&(layer,0)]].as_ref().unwrap()[1].shallow_clone());
            let x=x.to_kind(Kind::Half).contiguous();let ids=ids.contiguous();
            let out=Tensor::empty([rows,4096],(Kind::Float,x.device()));
            let mut r0=0;
            while r0<rows {
                let n=(rows-r0).min(crate::moe_own::PREFILL_ROWS);
                let a=add.map(|a|a.narrow(0,r0,n).contiguous());
                crate::moe_own::run_into(&tab,&x.narrow(0,r0,n),&ids.narrow(0,r0,n),&weights.narrow(0,r0,n),&out.narrow(0,r0,n),a.as_ref(),suh0.as_ref());
                r0+=n;
            }
            crate::ablate::apply(layer,&out);
            return out;
        }
        let host:Vec<i64>=Vec::try_from(&ids.reshape([-1]).to_device(Device::Cpu)).unwrap();
        let mut groups=std::collections::BTreeMap::<usize,Vec<i64>>::new();
        for (i,e) in host.into_iter().enumerate(){groups.entry(e as usize).or_default().push(i as i64);}
        // The same route metadata is consumed by input gather and output scatter.
        // Upload all expert slices once; views retain the packed allocation until
        // this serial invocation's consumers have been queued on the same stream.
        assert!(add.is_none(),"grouped add requires the fat MoE path");
        let packed=if std::env::var("GLM53_GROUPED_INDEX_PACK").as_deref()==Ok("1") {
            let mut indices=Vec::with_capacity((rows*16) as usize);
            indices.extend(groups.values().flat_map(|v|v.iter().copied()));
            indices.extend(groups.values().flat_map(|v|v.iter().map(|i|i/8)));
            Some(Tensor::from_slice(&indices).view([2,rows*8]).to_device(x.device()))
        }else{None};
        let mut packed_offset=0;
        let mut result=Tensor::empty([rows*8,4096],(Kind::Half,x.device()));
        // P2a: experts are independent (disjoint result rows); optional stream pool.
        let streams=match std::env::var("GLM53_PREFILL_EXPERT_STREAMS").ok().and_then(|v|v.parse::<i32>().ok()).unwrap_or(0){1=>4,n=>n.clamp(0,8)};
        extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;fn rs_stream_join(n:i32)->i32;}
        if streams>1{assert_eq!(unsafe{rs_stream_fork(streams)},0);}
        for (turn,(expert,assignments)) in groups.into_iter().enumerate() {
            if streams>1{assert_eq!(unsafe{rs_stream_set(turn as i32%streams)},0);}
            let slot=*self.where_.get(&(layer,expert)).expect("resident expert");
            let tensors=self.alive[slot].as_ref().unwrap();
            let count=assignments.len() as i64;
            let (assignments,input_rows)=if let Some(ref packed)=packed {
                (packed.get(0).narrow(0,packed_offset,count),packed.get(1).narrow(0,packed_offset,count))
            }else{
                let a=Tensor::from_slice(&assignments).to_device(x.device());
                let r=a.floor_divide_scalar(8);(a,r)
            };
            packed_offset+=count;
            let input=x.index_select(0,&input_rows);
            let project=|a:&Tensor,projection:usize| {
                let (tr,sh,sv)=(&tensors[projection*3],&tensors[projection*3+1],&tensors[projection*3+2]);
                let (m,k,n)=(a.size()[0],sh.size()[0],sv.size()[0]);
                let out=Tensor::empty([m,n],(Kind::Half,a.device()));
                extern "C" {
                    fn rs_exl3_recon_gemm(x:*const std::ffi::c_void,y:*mut std::ffi::c_void,tr:*const std::ffi::c_void,sh:*const std::ffi::c_void,sv:*const std::ffi::c_void,m:i64,k:i64,n:i64)->i32;
                    fn rs_exl3_gemm(x:*const u8,y:*mut u8,xh:*mut u8,tr:*const u8,sh:*const u8,sv:*const u8,m:i64,k:i64,n:i64,kt:i64,nt:i64)->i32;
                }
                let min_rows=std::env::var("GLM53_PREFILL_RECON_MIN_ROWS").ok().map(|x|x.parse::<i64>().unwrap()).unwrap_or(1);
                let rc=if reconstruct && m>=min_rows {unsafe{rs_exl3_recon_gemm(a.data_ptr(),out.data_ptr(),tr.data_ptr(),sh.data_ptr(),sv.data_ptr(),m,k,n)}}else{
                    let scratch=Tensor::empty_like(a);
                    unsafe{rs_exl3_gemm(a.data_ptr().cast(),out.data_ptr().cast(),scratch.data_ptr().cast(),tr.data_ptr().cast(),sh.data_ptr().cast(),sv.data_ptr().cast(),m,k,n,k/16,n/16)}
                };assert_eq!(rc,0,"grouped expert projection");out
            };
            let gate=project(&input,0);let up=project(&input,1);
            let activation=grouped_swiglu(&gate,&up,std::env::var("GLM53_GROUPED_SWIGLU").as_deref()==Ok("1"));
            let down=project(&activation,2);
            let _=result.index_copy_(0,&assignments,&down);
        }
        if streams>1{assert_eq!(unsafe{rs_stream_join(streams)},0);}
        grouped_reduce(&result,weights,std::env::var("GLM53_GROUPED_REDUCE").as_deref()==Ok("1"))
    }

    /// P2b: one grouped launch per phase for all routed experts of a layer (vendored
    /// exllamav3 fat-MoE kernels). gate/up in FP32 before SwiGLU; per-assignment Half
    /// down rows feed the same fixed-order FP32 grouped_reduce (deterministic).
    /// Diagnostic only: prove that no previous scratch contents reach the result.
    pub(crate) fn poison_scratch(&mut self) {
        if let Some(s) = &mut self.scratch {
            for t in &mut s.buffers { let _ = t.fill_(f64::NAN); }
        }
    }

    fn projection_scratch(&mut self, k: i64, width: i64, dev: Device) -> [Tensor; 5] {
        assert!(self.projection_precision <= 2);
        let key = (k, width, self.projection_precision, dev);
        let mode = self.scratch_mode;
        let alloc = || {
            let gu = if self.projection_precision == 2 { Kind::Float } else { Kind::Half };
            let down = if self.projection_precision != 0 { Kind::Float } else { Kind::Half };
            [(4096, Kind::Half), (width, Kind::Half), (width, gu), (width, gu), (4096, down)]
                .map(|(n, kind)| if mode == ScratchMode::Zeros {
                    Tensor::zeros([k, 1, n], (kind, dev))
                } else { Tensor::empty([k, 1, n], (kind, dev)) })
        };
        if mode != ScratchMode::Reuse { return alloc(); }
        if self.scratch.as_ref().map(|s| s.key) != Some(key) {
            self.scratch = Some(Scratch { key, buffers: alloc() });
        }
        self.scratch.as_ref().unwrap().buffers.each_ref().map(Tensor::shallow_clone)
    }

    pub(crate) fn probe_selection(&mut self, layer: usize, selected: &[usize], dev: Device) -> Tensor {
        self.ensure_many(layer, selected, dev);
        Tensor::from_slice(&selected.iter().map(|&e| self.slot_of(layer,e)).collect::<Vec<_>>()).view([1,-1]).to_device(dev)
    }

    /// 装入一个专家到指定结构(raw 已在手)。返回 slot。
    /// Claim a slot for a nonresident expert (evicting the LRU one when the pool is full).
    fn claim_slot(&mut self, layer: usize, e: usize) -> usize {
        assert!(!self.where_.contains_key(&(layer,e)),"install requires a nonresident expert");
        self.own_tables.remove(&layer);
        let slot = if let Some(s) = self.free.pop() {
            s
        } else {
            let s = self.lru.pop_front().expect("lru");
            self.alive[s] = None;
            // 清设备侧选路表 + 找回该槽的 (layer, e) 并从索引删除
            if let Some((ol, oe)) = self.owner[s].take() {
                let _ = self.slot_table.get(ol as i64).get(oe as i64).fill_(-1);
                self.where_.remove(&(ol, oe));
                self.resident_per_layer[ol]-=1;
                self.own_tables.remove(&ol);
            }
            s
        };
        self.misses += 1;
        slot
    }

    /// Record a filled slot. `tensors` = (tr, suh, svh) x gate/up/down, index i feeds pointer table i.
    /// `tables`: write its pointer-table entries and slot-table cell now (the batched preload writes
    /// every table once at the end instead, see write_tables).
    fn commit_slot(&mut self, layer: usize, e: usize, slot: usize, tensors: Vec<Tensor>, tables: bool) {
        // 每个投影写自己那组表:ptrs[pi*3 + {0=tr, 1=suh, 2=svh}]
        if tables {
            for (i, t) in tensors.iter().enumerate() {
                let _ = self.ptrs[i].narrow(0, slot as i64, 1).fill_(t.data_ptr() as i64);
            }
        }
        self.alive[slot] = Some(tensors);
        self.owner[slot] = Some((layer, e));
        self.where_.insert((layer, e), slot);
        self.resident_per_layer[layer]+=1;
        self.lru.push_back(slot);
        if tables {
            let _ = self.slot_table.get(layer as i64).get(e as i64).fill_(slot as i64);
        }
    }

    /// 装入一个专家到指定结构(raw 已在手)。返回 slot。
    fn install(&mut self, layer: usize, e: usize, raws: [RawProj; 3], dev: Device) -> i64 {
        assert!(!self.where_.contains_key(&(layer,e)),"install requires a nonresident expert");
        self.shared_input[layer].observe(e,&raws[0].2,&raws[1].2);
        let slot = self.claim_slot(layer, e);
        let mut tensors: Vec<Tensor> = Vec::with_capacity(9);
        for (raw, s, suh, svh) in raws.into_iter() {
            tensors.push(Tensor::from_slice(&raw).view(s.as_slice()).to_device(dev));
            tensors.push(Tensor::from_slice(&suh).to_kind(Kind::Half).to_device(dev));
            tensors.push(Tensor::from_slice(&svh).to_kind(Kind::Half).to_device(dev));
        }
        self.commit_slot(layer, e, slot, tensors, true);
        slot as i64
    }

    /// Write every resident slot's pointer-table entries and slot-table cell in one upload per table.
    fn write_tables(&mut self) {
        let dev = self.slot_table.device();
        for i in 0..PTR_TABLES {
            let mut host = Vec::<i64>::try_from(&self.ptrs[i].to_device(Device::Cpu)).unwrap();
            for (slot, a) in self.alive.iter().enumerate() {
                if let Some(t) = a { host[slot] = t[i].data_ptr() as i64; }
            }
            let _ = self.ptrs[i].copy_(&Tensor::from_slice(&host).to_device(dev));
        }
        let (nl, ne) = (self.slot_table.size()[0], self.slot_table.size()[1]);
        let mut st = Vec::<i64>::try_from(&self.slot_table.reshape([-1]).to_device(Device::Cpu)).unwrap();
        for (&(l, e), &slot) in &self.where_ { st[l * ne as usize + e] = slot as i64; }
        let _ = self.slot_table.copy_(&Tensor::from_slice(&st).view([nl, ne]).to_device(dev));
    }

    fn ensure(&mut self, layer: usize, e: usize, dev: Device) -> i64 {
        if let Some(&s) = self.where_.get(&(layer, e)) {
            return s as i64;
        }
        let raw = load_expert_raw(&self.idx, layer, e, self.tp);
        self.install(layer, e, raw, dev)
    }

    /// 批量装入(并行磁盘读取;GPU 上传仍串行)。
    pub fn ensure_many(&mut self, layer: usize, experts: &[usize], dev: Device) {
        let mut experts = experts.to_vec();
        experts.sort_unstable();
        experts.dedup();
        assert!(experts.len() <= self.cap, "本批专家数超过池容量");
        // 将本轮命中的槽移到队尾，整个批次不会逐出它们。
        for e in &experts {
            if let Some(&slot) = self.where_.get(&(layer, *e)) {
                self.lru.retain(|&s| s != slot);
                self.lru.push_back(slot);
            }
        }
        let missing: Vec<usize> = experts
            .iter()
            .copied()
            .filter(|&e| !self.where_.contains_key(&(layer, e)))
            .collect();
        if missing.is_empty() {
            return;
        }
        let tp = self.tp;
        let raws: Vec<(usize, [RawProj; 3])> = missing
            .par_iter()
            .map(|&e| (e, load_expert_raw(&self.idx, layer, e, tp)))
            .collect();
        for (e, r) in raws {
            self.install(layer, e, r, dev);
        }
    }

    /// 全量预载(TP2 图模式前提):每层全部专家分片驻留。rayon 并行磁盘读。
    pub fn preload_all(&mut self, n_layers: usize, n_experts: usize, dev: Device) {
        self.preload_all_with(None, n_layers, n_experts, dev)
    }

    /// preload_all consuming a reader started earlier (serve starts it at process start so the disk reads
    /// overlap the rest of the load). Without one, the fast path starts its own; a reader that does not
    /// match this pool (rank, layers) is dropped and the pool loads by itself.
    pub fn preload_all_with(&mut self, reader: Option<crate::expert_load::ExpertReader>, n_layers: usize, n_experts: usize, dev: Device) {
        assert!(self.cap >= self.expected_total(), "池容量 {} 小于全驻留所需 {}，拒绝预载", self.cap, self.expected_total());
        let t0 = std::time::Instant::now();
        let all: Vec<usize> = (0..n_experts).collect();
        // 跳过 dense 层(无 experts 键)
        let layers: Vec<usize> = (0..n_layers)
            .filter(|l| self.idx.entries.contains_key(&format!("model.language_model.layers.{l}.mlp.experts.0.gate_proj.trellis")))
            .collect();
        let skipped = n_layers - layers.len();
        let reader = if self.where_.is_empty() {
            reader.filter(|r| r.tp == self.tp && r.layers == layers && r.n_experts == n_experts)
                .or_else(|| crate::expert_load::ExpertReader::start(&self.idx.dir, n_layers, n_experts, self.tp, dev, Vec::new()))
        } else { None };
        if let Some(r) = reader {
            let (mut wait_s, mut view_s) = (0f64, 0f64);
            for &l in &layers {
                let t = std::time::Instant::now();
                let got = r.recv();
                wait_s += t.elapsed().as_secs_f64();
                assert_eq!(got.layer, l, "expert reader out of order");
                let t = std::time::Instant::now();
                for (e, (g, u)) in got.suh.iter().enumerate() {
                    self.shared_input[l].observe(e, g, u);
                    let slot = self.claim_slot(l, e);
                    self.commit_slot(l, e, slot, r.views(&got.arena, e), false);
                }
                view_s += t.elapsed().as_secs_f64();
            }
            self.write_tables();
            eprintln!("[moefast] fast preload: {}, pool waited {wait_s:.1}s, views {view_s:.1}s", r.summary());
        } else {
            for &l in &layers {
                self.ensure_many(l, &all, dev);
            }
        }
        assert!(self.is_fully_loaded(), "预载范围未覆盖全部专家");
        crate::host_memory::finish_loading("expert-preload");
        crate::host_memory::release_file_cache(&[self.idx.dir.as_path()]);
        println!("[moefast] 全量预载完成:{} 专家分片驻留(跳过 {} 个 dense 层),{:.1}s", self.where_.len(), skipped, t0.elapsed().as_secs_f32());
        let checked=self.shared_input.iter().filter(|proof|proof.distinct>0).count();
        let certified=(0..self.shared_input.len()).filter(|&layer|self.shared_input_verified(layer)).count();
        println!("[moefast] shared-input runtime proof: {certified}/{checked} layers have 288 matching resident experts");
    }

    /// slot 查询(要求已驻留;先 ensure_many)。
    pub fn slot_of(&self, layer: usize, e: usize) -> i64 {
        self.where_[&(layer, e)] as i64
    }

    /// 当前驻留专家数。
    pub fn resident_count(&self) -> usize {
        self.where_.len()
    }

    /// 是否全驻留(图模式前提)。
    pub fn is_fully_loaded(&self) -> bool {
        self.resident_count() >= self.expected_total() && self.idx.entries.keys()
            .filter(|k| k.starts_with("model.language_model.layers.")
                && k.contains(".mlp.experts.") && k.ends_with(".gate_proj.trellis"))
            .all(|k| {
                let parts: Vec<_> = k.split('.').collect();
                let l: usize = parts[3].parse().expect("layer id");
                let e: usize = parts[6].parse().expect("expert id");
                l >= self.slot_table.size()[0] as usize || self.where_.contains_key(&(l,e))
            })
    }

    /// 期望总量(含专家的层 × 每居专家数;扫描索引一次)。
    pub fn expected_total(&self) -> usize {
        self.idx.entries.keys()
            .filter(|k| k.starts_with("model.language_model.layers.")
                && k.contains(".mlp.experts.") && k.ends_with(".gate_proj.trellis")
                && k.split('.').nth(3).and_then(|v| v.parse::<i64>().ok())
                    .map(|l| l < self.slot_table.size()[0]).unwrap_or(false))
            .count()
    }

    /// 设备侧选路:topi [k] i64(GPU)→ sel 槽位 [1,k] i64(GPU)。
    /// 未驻留时回退:读 topi 到 host、ensure、重收集。稳态每层仅 1 次布尔同步。
    pub fn sel_device(&mut self, layer: usize, topi: &Tensor, dev: Device) -> Tensor {
        let row = self.slot_table.get(layer as i64);
        let sel = row.index_select(0, topi);
        if self.assume_hot || std::env::var("GLM53_ASSUME_HOT").is_ok() {
            return sel.unsqueeze(0); // 跳过缺失检查(仅基准/图模式;未驻留会算错)
        }
        let missing = sel.lt(0).any().int64_value(&[]) != 0;
        if missing {
            let host: Vec<i64> = topi.to_device(Device::Cpu).try_into().expect("topi");
            let es: Vec<usize> = host.iter().map(|&e| e as usize).collect();
            self.ensure_many(layer, &es, dev);
            let row = self.slot_table.get(layer as i64);
            return row.index_select(0, topi).unsqueeze(0);
        }
        sel.unsqueeze(0)
    }

    /// 逐投影探针:只跑 proj(0=gate,1=up,2=down) 一次 mgemm,返回 C [k,1,N]。
    /// 用于定位合批路径的错误级(2026-09-21:整链 rel=1.45 → 二分用)。
    pub fn proj_probe(&mut self, layer: usize, selected: &[usize], proj: usize, a: &Tensor, dev: Device) -> Tensor {
        assert!(proj < 3);
        self.ensure_many(layer, selected, dev);
        let slots: Vec<i64> = selected.iter().map(|&e| self.slot_of(layer, e)).collect();
        let k = selected.len() as i64;
        let sel_t = Tensor::from_slice(&slots).view([1, k]).to_device(dev);
        let (kk, nn, rows) = match proj {
            0 | 1 => (4096i64, 2048i64, 1i64),
            _ => (2048, 4096, k),
        };
        let yh = Tensor::zeros([k, 1, kk], (Kind::Half, dev));
        let c = Tensor::zeros([k, 1, nn], (Kind::Half, dev));
        let (p1, p2, p3) = (&self.ptrs[proj * 3], &self.ptrs[proj * 3 + 1], &self.ptrs[proj * 3 + 2]);
        let rc = unsafe {
            rs_exl3_mgemm2(a.data_ptr() as *const u8, c.data_ptr() as *mut u8, yh.data_ptr() as *mut u8,
                           p1.data_ptr() as *const u8, p2.data_ptr() as *const u8, p3.data_ptr() as *const u8,
                           sel_t.data_ptr() as *const u8, k, kk, nn, rows, 64 / 16, nn / 16, self.cap as i64)
        };
        assert!(rc == 0, "proj_probe gemm rc={rc}");
        c
    }

    /// 单 token 批量专家前向:x16 [1,4096] fp16 → [1,4096] fp32。
    pub fn expert_batch(&mut self, layer: usize, x16: &Tensor, selected: &[usize], wts: &Tensor, dev: Device) -> Tensor {
        self.ensure_many(layer, selected, dev);
        let slots: Vec<i64> = selected.iter().map(|&e| self.where_[&(layer, e)] as i64).collect();
        let sel_t = Tensor::from_slice(&slots).view([1, -1]).to_device(dev);
        self.expert_batch_sel(x16, &sel_t, wts, dev)
    }

    /// sel 已在设备上的批量前向(主路径,配合 sel_device 零 host 往返)。
    /// TP 下各 rank 算本地半:gate/up 列切出 [k,1,2048/world],down 行切入 [k,1,2048/world],
    /// 加权和为部分和 → 调用方负责 allreduce。
    pub fn expert_batch_sel(&mut self, x16: &Tensor, sel_t: &Tensor, wts: &Tensor, dev: Device) -> Tensor {
        assert_eq!(x16.size()[0],1);
        self.expert_multi_sel(x16,sel_t,wts,dev)
    }

    /// Flatten (token, selected expert) into independent mgemm matrix slots.
    /// Each slot has its own input, Hadamard scratch and output; weights remain EXL3.
    /// Kept opt-in because auto tile/grid selection changes with matrix batch size.
    pub fn expert_multi_sel(&mut self,x16:&Tensor,sel_t:&Tensor,wts:&Tensor,dev:Device)->Tensor {
        let tokens=x16.size()[0];let per_token=sel_t.size()[1];
        assert!((1..=16).contains(&tokens));assert_eq!(sel_t.size()[0],tokens);
        assert_eq!(wts.numel() as i64,tokens*per_token);
        let k=tokens*per_token;let sel_t=sel_t.reshape([1,k]).contiguous();
        let world = self.tp.world as i64;
        let n_gu = 2048 / world;  // gate/up 输出宽(列切)
        let k_d = 2048 / world;   // down 输入宽(行切)
        // EXL3 writes YH and the first partial sum before reading either buffer.
        // Empty/reused storage avoids five redundant zero-fill kernels per call.
        let [yh, yh_d, g, u, d] = self.projection_scratch(k, n_gu, dev);

        let call = |a: &Tensor, c: &Tensor, kk: i64, nn: i64, rows: i64,
                    ptrtr: &Tensor, ptrsuh: &Tensor, ptrsvh: &Tensor,
                    sel: &Tensor, yh: &Tensor, k: i64| -> bool {
            unsafe {
                if c.kind() == Kind::Float {
                    return rs_exl3_mgemm_probe(a.data_ptr(), c.data_ptr(), yh.data_ptr(),
                        ptrtr.data_ptr(), ptrsuh.data_ptr(), ptrsvh.data_ptr(), sel.data_ptr(),
                        k, kk, nn, rows, self.cap as i64, 1) == 0;
                }
                rs_exl3_mgemm2(a.data_ptr() as *const u8, c.data_ptr() as *mut u8, yh.data_ptr() as *mut u8,
                               ptrtr.data_ptr() as *const u8, ptrsuh.data_ptr() as *const u8, ptrsvh.data_ptr() as *const u8,
                               sel.data_ptr() as *const u8, k, kk, nn, rows, 64 / 16, nn / 16, self.cap as i64) == 0
            }
        };

        let a3=if tokens==1{x16.unsqueeze(0)}else{
            x16.unsqueeze(1).expand([tokens,per_token,4096],false).contiguous().view([k,1,4096])
        };
        let input_rows=if tokens==1{1}else{k};
        {
            let (g1, g2, g3) = (&self.ptrs[0], &self.ptrs[1], &self.ptrs[2]);
            let (u1, u2, u3) = (&self.ptrs[3], &self.ptrs[4], &self.ptrs[5]);
            assert!(call(&a3, &g, 4096, n_gu, input_rows, g1, g2, g3, &sel_t, &yh, k), "gate");
            assert!(call(&a3, &u, 4096, n_gu, input_rows, u1, u2, u3, &sel_t, &yh, k), "up");
        }
        let g32 = g.to_kind(Kind::Float).clamp(f64::NEG_INFINITY, SWIGLU_LIMIT);
        let u32 = u.to_kind(Kind::Float).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let act = (g32.silu() * u32).to_kind(Kind::Half); // [k,1,2048/world]
        {
            let (d1, d2, d3) = (&self.ptrs[6], &self.ptrs[7], &self.ptrs[8]);
            assert!(call(&act, &d, k_d, 4096, k, d1, d2, d3, &sel_t, &yh_d, k), "down");
        }
        let d32 = d.to_kind(Kind::Float).squeeze_dim(1); // [k,4096]
        if tokens==1 {wts.reshape([1,k]).to_kind(Kind::Float).matmul(&d32)}
        else {wts.reshape([tokens,1,per_token]).to_kind(Kind::Float).bmm(&d32.view([tokens,per_token,4096])).squeeze_dim(1)}
    }
}

pub(crate) fn route_group_gpu_enabled()->bool {std::env::var("GLM53_ROUTE_GROUP_GPU").as_deref()==Ok("1")}
/// Sticky device flag for out-of-range expert ids / segment overflow; checked by probes and at
/// the end of a prefill chunk (no per-layer sync).
pub(crate) fn route_error_flag(dev:Device)->Tensor {
    thread_local!{static FLAG:std::cell::RefCell<Option<Tensor>>=const{std::cell::RefCell::new(None)};}
    FLAG.with(|f|f.borrow_mut().get_or_insert_with(||Tensor::zeros([1],(Kind::Int,dev))).shallow_clone())
}
pub(crate) fn route_error_check(dev:Device) {
    if !route_group_gpu_enabled(){return;}
    route_error_check_forced(dev);
}
pub(crate) fn route_error_check_forced(dev:Device) {
    let v=i32::try_from(route_error_flag(dev).to_device(Device::Cpu)).unwrap();
    assert_eq!(v,0,"device route grouping error {v} (1: expert id out of range, 2: segment capacity)");
}
pub(crate) fn grouped_reduce_add(result:&Tensor,weights:&Tensor,add:&Tensor)->Tensor {
    let rows=weights.size()[0];assert_eq!(weights.size(),[rows,8]);
    assert!(result.is_contiguous());assert_eq!(result.kind(),Kind::Half);
    assert_eq!(add.size(),[rows,4096]);assert_eq!(add.kind(),Kind::Float);assert!(add.is_contiguous());
    let weights=weights.to_kind(Kind::Half).contiguous();let out=Tensor::empty([rows,4096],(Kind::Float,result.device()));
    extern "C" {fn rs_grouped_reduce_add(x:*const std::ffi::c_void,w:*const std::ffi::c_void,add:*const f32,y:*mut f32,rows:i32)->i32;}
    assert_eq!(unsafe{rs_grouped_reduce_add(result.data_ptr(),weights.data_ptr(),add.data_ptr().cast(),out.data_ptr().cast(),rows as i32)},0);out
}

/// Preserve the FP16 route-weight rounding and libtorch's four accumulators.
pub(crate) fn grouped_reduce(result:&Tensor,weights:&Tensor,fused:bool)->Tensor {
    let rows=weights.size()[0];assert_eq!(weights.size(),[rows,8]);
    assert!(result.is_contiguous());assert_eq!(result.kind(),Kind::Half);
    if !fused {return (result.view([rows,8,4096]).to_kind(Kind::Float)*weights.to_kind(Kind::Half).to_kind(Kind::Float).unsqueeze(-1))
        .sum_dim_intlist(&[1i64][..],false,Kind::Float);}
    let weights=weights.to_kind(Kind::Half).contiguous();let out=Tensor::empty([rows,4096],(Kind::Float,result.device()));
    extern "C" {fn rs_grouped_reduce(x:*const std::ffi::c_void,w:*const std::ffi::c_void,y:*mut f32,rows:i32)->i32;}
    assert_eq!(unsafe{rs_grouped_reduce(result.data_ptr(),weights.data_ptr(),out.data_ptr().cast(),rows as i32)},0);out
}

pub(crate) fn grouped_swiglu(gate:&Tensor,up:&Tensor,fused:bool)->Tensor {
    assert_eq!(gate.size(),up.size());assert_eq!(gate.kind(),Kind::Half);assert_eq!(up.kind(),Kind::Half);
    if !fused {return (gate.to_kind(Kind::Float).clamp(f64::NEG_INFINITY,SWIGLU_LIMIT).silu()
        * up.to_kind(Kind::Float).clamp(-SWIGLU_LIMIT,SWIGLU_LIMIT)).to_kind(Kind::Half);}
    assert!(gate.is_contiguous() && up.is_contiguous());let out=Tensor::empty_like(gate);
    extern "C" {fn rs_grouped_swiglu(gate:*const std::ffi::c_void,up:*const std::ffi::c_void,out:*mut std::ffi::c_void,count:i32,limit:f32)->i32;}
    assert_eq!(unsafe{rs_grouped_swiglu(gate.data_ptr(),up.data_ptr(),out.data_ptr(),gate.numel() as i32,SWIGLU_LIMIT as f32)},0);out
}

#[cfg(test)]
mod shared_input_proof_tests {
    use super::SharedInputCertificate;

    fn scales(value:f32)->Vec<f32>{vec![value;4096]}
    fn complete(proof:&mut SharedInputCertificate,scale:&[f32]) {
        for expert in 0..288{proof.observe(expert,scale,scale);}
    }

    #[test]
    fn requires_all_distinct_experts_and_current_residency() {
        let scale=scales(1.);let mut proof=SharedInputCertificate::new(288);
        for expert in 0..287{proof.observe(expert,&scale,&scale);}
        for _ in 0..300{proof.observe(0,&scale,&scale);}
        assert_eq!(proof.distinct,287);
        assert!(!proof.verified(288),"duplicate loads cannot certify the missing expert");
        proof.observe(287,&scale,&scale);
        assert!(proof.verified(288));
        assert!(!proof.verified(287),"eviction invalidates current residency");
        proof.observe(287,&scale,&scale);
        assert_eq!(proof.distinct,288);assert!(proof.verified(288));
    }

    #[test]
    fn mismatch_on_reload_revokes_proof_permanently() {
        let scale=scales(1.);let mut proof=SharedInputCertificate::new(288);
        complete(&mut proof,&scale);assert!(proof.verified(288));
        let mut changed=scale.clone();changed[23]=f32::from_bits(1f32.to_bits()+1);
        // Both projections still agree with each other and round to the same
        // Half as before. The stricter source-bit proof must still reject it.
        assert_eq!(half::f16::from_f32(changed[23]),half::f16::from_f32(scale[23]));
        proof.observe(17,&changed,&changed);assert!(!proof.verified(288));
        proof.observe(17,&scale,&scale);assert!(!proof.verified(288));
    }

    #[test]
    fn rejects_gate_up_mismatch_signed_zero_and_nonfinite_upload() {
        let scale=scales(1.);
        for invalid in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY,f32::MAX] {
            let mut bad=scale.clone();bad[9]=invalid;
            let mut proof=SharedInputCertificate::new(288);proof.observe(0,&bad,&bad);
            complete(&mut proof,&scale);assert!(!proof.verified(288));
        }
        let positive=scales(0.);let negative=scales(-0.);
        let mut proof=SharedInputCertificate::new(288);proof.observe(0,&positive,&negative);
        complete(&mut proof,&positive);assert!(!proof.verified(288));
        let mut other=scale.clone();other[1]=2.;
        let mut proof=SharedInputCertificate::new(288);proof.observe(0,&scale,&other);
        complete(&mut proof,&scale);assert!(!proof.verified(288));
    }

    #[test]
    fn rejects_incomplete_shape_and_wrong_expert_geometry() {
        let scale=scales(1.);
        for len in [0,4095,4097] {
            let short=vec![1.;len];let mut proof=SharedInputCertificate::new(288);
            proof.observe(0,&short,&short);complete(&mut proof,&scale);
            assert!(!proof.verified(288));
        }
        let mut proof=SharedInputCertificate::new(16);
        for expert in 0..16{proof.observe(expert,&scale,&scale);}
        assert!(!proof.verified(16));assert!(!proof.verified(288));
    }

    #[test]
    fn layer_and_pool_canonical_values_are_independent() {
        let one=scales(1.);let two=scales(2.);
        let mut layers=[SharedInputCertificate::new(288),SharedInputCertificate::new(288)];
        complete(&mut layers[0],&one);complete(&mut layers[1],&two);
        assert!(layers.iter().all(|p|p.verified(288)));
        layers[0].observe(1,&two,&two);
        assert!(!layers[0].verified(288));assert!(layers[1].verified(288));
        let mut other_pool=SharedInputCertificate::new(288);complete(&mut other_pool,&two);
        assert!(other_pool.verified(288));
    }
}
