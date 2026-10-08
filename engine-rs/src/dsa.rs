//! Incremental GLM indexer. Four-token pools, top-512 pools, visible partial tail.
//! Mirrors the independently verified M0 indexer semantics, including pool-end self visibility.
use tch::{Tensor,Kind,Device};

/// Verifier-only execution regime. Capture/replay callers choose this from a
/// checked host extent; the index construction still reads each device pos.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum TreeSelection {Ranked,AllVisible}
pub fn tree_selection(base_len:i64,max_depth:i64)->TreeSelection {
    assert!(base_len>=0&&max_depth>0);
    if std::env::var("GLM53_DSA_ALL_VISIBLE").as_deref()==Ok("1") &&
        base_len.checked_add(max_depth).is_some_and(|n|n<=2051) {
        TreeSelection::AllVisible
    }else{TreeSelection::Ranked}
}

pub struct Weights {
    pub q: Tensor,
    pub k: Tensor,
    pub norm_w: Tensor,
    pub norm_b: Tensor,
    pub score: Tensor,
    pub ape: Tensor,
    pub gate: Tensor,
}

pub struct State {
    pub pools: Tensor,
    pub tail_k: Tensor,
    pub tail_gate: Tensor,
    pool_positions:Tensor,slot_offsets:Tensor,tail_offsets:Tensor,
    all_tokens:Tensor,
}

fn prefill_topk_fast()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_DSA_PREFILL_TOPK_FAST").as_deref()==Ok("1"))}
/// GLM53_DSA_PREFILL_TOPK_CHECK=1 (diagnostic): recompute every fast prefill block through the ATen path and compare.
fn prefill_topk_check()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_DSA_PREFILL_TOPK_CHECK").as_deref()==Ok("1"))}
pub(crate) fn prefill_score_fused_enabled()->bool {std::env::var("GLM53_DSA_PREFILL_SCORE_FUSED").as_deref()==Ok("1")}
pub struct Projected {pub k:Tensor,pub gate:Tensor,pub q:Tensor,pub mixing:Tensor}
pub struct KeyProjected {pub k:Tensor,pub gate:Tensor}
/// x @ w^T. BF16-resident weights (GLM53_DSA_INDEX_BF16=1): decode rows (<= 16) use the FP32-FMA kernel on
/// the widened BF16 values (L1 vs TF32 cuBLAS); larger rows widen to FP32 and keep the original matmul
/// (identical values, so prefill is unchanged).
fn proj(x:&Tensor,w:&Tensor)->Tensor {
    // GLM53_C12_INDEX=1: C12-coded indexer weights (decode rows; row-count invariant, so ahead of the 8-row pieces).
    if let Some(y)=crate::c12::try_run(x,w,0) {return y;}
    if w.kind()!=Kind::BFloat16 {return x.matmul(&w.transpose(0,1));}
    let (m,k,n)=(x.size()[0],x.size()[1],w.size()[0]);
    // Proposal 3: bf16w_rows<16> differs per row from <4>/<8> (which agree); verify batches over 8 rows run in
    // 8-row pieces so every row takes the <=8-row arithmetic.
    if crate::forward::verify_invariant() && (9..=64).contains(&m) && x.device().is_cuda() {
        return Tensor::cat(&crate::forward::invariant_pieces(m).into_iter().map(|(r,n)|proj(&x.narrow(0,r,n),w)).collect::<Vec<_>>(),0);
    }
    // GLM53_DSA_INDEX_BF16_KERNEL=0 keeps the TF32 matmul on the widened weights (bitwise the FP32 baseline):
    // the in-process L1 gate toggles only the arithmetic, with identical resident weights.
    if std::env::var("GLM53_DSA_INDEX_BF16_KERNEL").as_deref()!=Ok("0")
        && x.device().is_cuda() && x.kind()==Kind::Float && x.dim()==2 && (1..=16).contains(&m) && k%256==0 && x.stride()[1]==1 && x.stride()[0]%4==0 && w.is_contiguous() {
        let y=Tensor::empty([m,n],(Kind::Float,x.device()));
        extern "C"{fn rs_bf16w_rows(x:*const f32,ldx:i32,w:*const std::ffi::c_void,y:*mut f32,m:i32,n:i32,k:i32)->i32;}
        assert_eq!(unsafe{rs_bf16w_rows(x.data_ptr().cast(),x.stride()[0] as i32,w.data_ptr(),y.data_ptr().cast(),m as i32,n as i32,k as i32)},0,"BF16 indexer projection");
        return y;
    }
    // GLM53_DSA_INDEX_HALF_PREFILL=1 (L1): prefill-sized rows on Half tensor cores (x rounded to Half: the same 10-bit
    // input mantissa as the TF32 GEMM it replaces; weights exact in Half, checked once) with FP32 accumulation/output,
    // instead of widening the BF16 weight to FP32 on every call for a TF32 GEMM.
    if m>16 && x.device().is_cuda() && std::env::var("GLM53_DSA_INDEX_HALF_PREFILL").as_deref()==Ok("1") {
        if let Some(wh)=half_copy(w) {return crate::weights::mm16_partial(x,&wh);}
    }
    x.matmul(&w.to_kind(Kind::Float).transpose(0,1))
}
thread_local!{static HALF_W:std::cell::RefCell<std::collections::HashMap<usize,Option<Tensor>>>=std::cell::RefCell::new(std::collections::HashMap::new());}
/// Half copy of a BF16 indexer weight, when every value is exact in Half (else None, and the caller keeps TF32).
fn half_copy(w:&Tensor)->Option<Tensor> {
    HALF_W.with(|c|c.borrow_mut().entry(w.data_ptr() as usize).or_insert_with(||{
        let h=w.to_kind(Kind::Half).contiguous();
        if h.to_kind(Kind::BFloat16).equal(w) {Some(h)} else {eprintln!("[dsa] indexer weight {:?} not exact in Half: TF32 kept",w.size());None}
    }).as_ref().map(Tensor::shallow_clone))
}
impl Weights {
    pub fn project_keys(&self,x:&Tensor)->KeyProjected {
        // Proposal 3: ATen's mean_dim reduction split depends on the row count; verify batches over 8 rows are
        // normalised in 8-row pieces (the <=8-row arithmetic).
        let m=x.size()[0];
        if crate::forward::verify_invariant() && (9..=64).contains(&m) && x.device().is_cuda() {
            let parts:Vec<KeyProjected>=crate::forward::invariant_pieces(m).into_iter().map(|(r,n)|self.project_keys(&x.narrow(0,r,n))).collect();
            return KeyProjected{k:Tensor::cat(&parts.iter().map(|p|p.k.shallow_clone()).collect::<Vec<_>>(),0),
                gate:Tensor::cat(&parts.iter().map(|p|p.gate.shallow_clone()).collect::<Vec<_>>(),0)};
        }
        let grouped=crate::c12::try_run_rows_out(x,&[&self.k,&self.gate],0);
        // GLM53_DSA_KEY_FUSED=1 (L1): the LayerNorm below and the gate's contiguous copy in one launch (dataflow.cu dsa_key_ln).
        if let Some(y)=&grouped {
            static FUSED:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
            if *FUSED.get_or_init(||std::env::var("GLM53_DSA_KEY_FUSED").as_deref()==Ok("1")) && y.kind()==Kind::Float && y.is_contiguous()
                && self.norm_w.kind()==Kind::Float && self.norm_b.kind()==Kind::Float && self.norm_w.is_contiguous() && self.norm_b.is_contiguous() {
                let (m,dim,gc)=(y.size()[0],self.k.size()[0],self.gate.size()[0]);
                let k=Tensor::empty([m,dim],(Kind::Float,y.device()));let g=Tensor::empty([m,gc],(Kind::Float,y.device()));
                extern "C"{fn rs_dsa_key_ln(y:*const f32,ldy:i32,dim:i32,gcols:i32,w:*const f32,b:*const f32,k:*mut f32,g:*mut f32,rows:i32)->i32;}
                assert_eq!(unsafe{rs_dsa_key_ln(y.data_ptr().cast(),y.size()[1] as i32,dim as i32,gc as i32,self.norm_w.data_ptr().cast(),self.norm_b.data_ptr().cast(),
                    k.data_ptr().cast(),g.data_ptr().cast(),m as i32)},0,"DSA key LayerNorm");
                return KeyProjected{k,gate:g};
            }
        }
        let (raw,gate)=match grouped {
            Some(y)=>{let n=self.k.size()[0];(y.narrow(1,0,n),Some(y.narrow(1,n,self.gate.size()[0]).contiguous()))},None=>(proj(x,&self.k),None)};
        let centered=&raw-raw.mean_dim(&[-1i64][..],true,Kind::Float);
        let variance=(&centered*&centered).mean_dim(&[-1i64][..],true,Kind::Float);
        KeyProjected{k:centered*(variance+1e-6).rsqrt()*&self.norm_w+&self.norm_b,
            gate:gate.unwrap_or_else(||proj(x,&self.gate))}
    }
    pub fn project(&self,x:&Tensor,q_resid:&Tensor)->Projected {
        let t=x.size()[0];let dim=self.k.size()[0];let heads=self.q.size()[0]/dim;
        let KeyProjected{k,gate}=self.project_keys(x);
        Projected{k,gate,
            q:proj(q_resid,&self.q).view([t,heads,dim]),
            mixing:proj(x,&self.score).view([t,heads,1])*(heads as f64).powf(-0.5)}
    }
}

impl State {
    /// Prefill-only: construct complete pools once, then score query blocks
    /// against them with a per-query causal mask. Preserve the provisional
    /// final pool and the stale slots of the rolling tail for exact checkout.
    /// GLM53_DSA_PREFILL_TOPK_FAST=1 (L0, widths >= 1024): the exact fast top-k over each row's visible prefix plus the token
    /// expansion (dsa_topk_rows + dsa_index_expand_rows), replacing masked_fill + ATen topk + the ids/tail arithmetic and cat
    /// of append_chunk. `scores` [n, active] raw or already masked: only [0, complete) of each row is read.
    #[allow(clippy::too_many_arguments)]
    fn fast_select(&self,scores:&Tensor,n:i64,active:i64,kk:i64,width:i64,start:i64,first:i64,complete:&Tensor)->Option<Tensor> {
        if !(active>=1024 && kk==512 && width==512 && prefill_topk_fast()) {return None;}
        assert!(scores.is_contiguous() && scores.size()==[n,active] && scores.kind()==Kind::Float);
        let dev=scores.device();
        let pos=Tensor::arange(n,(Kind::Int64,dev))+start+first;
        let sel=Tensor::empty([n,512],(Kind::Int64,dev));let out=Tensor::empty([n,4*512+3],(Kind::Int64,dev));
        extern "C"{fn rs_dsa_prefill_topk_expand(scores:*const f32,pos:*const i64,selected:*mut i64,out:*mut i64,pools:i32,k:i32,rows:i32)->i32;}
        assert_eq!(unsafe{rs_dsa_prefill_topk_expand(scores.data_ptr().cast(),pos.data_ptr().cast(),sel.data_ptr().cast(),out.data_ptr().cast(),active as i32,512,n as i32)},0,"DSA prefill fast top-k");
        if prefill_topk_check() {
            let visible=self.pool_positions.narrow(0,0,active).unsqueeze(0).lt_tensor(&complete.unsqueeze(1));
            let ids=scores.masked_fill(&visible.logical_not(),f32::MIN as f64).topk(kk,1,true,true).1;
            let valid=ids.lt_tensor(&complete.unsqueeze(1));
            let tokens=(&ids.unsqueeze(-1)*4+&self.slot_offsets).masked_fill(&valid.logical_not().unsqueeze(-1),-1).view([n,-1]);
            let len=Tensor::arange(n,(Kind::Int64,dev))+start+first+1;
            let tail=complete.unsqueeze(1)*4+&self.tail_offsets;
            let tail=tail.masked_fill(&self.tail_offsets.unsqueeze(0).ge_tensor(&len.remainder(4).unsqueeze(1)),-1);
            let same=Tensor::cat(&[tokens,tail],1).eq_tensor(&out).all().int64_value(&[])==1;
            eprintln!("[dsa-prefill-topk] {} start {start} first {first} rows {n} active {active}",if same{"ok"}else{"MISMATCH"});
        }
        Some(out)
    }
    pub fn append_chunk(&mut self,w:&Weights,p:&Projected,start:i64)->Tensor {
        let t=p.k.size()[0];let dim=p.k.size()[1];let offset=start%4;
        let keys=Tensor::cat(&[self.tail_k.narrow(0,0,offset),p.k.shallow_clone()],0);
        let gates=Tensor::cat(&[self.tail_gate.narrow(0,0,offset),p.gate.shallow_clone()],0);
        let full=(offset+t)/4;let remainder=(offset+t)%4;
        if full>0 {
            let k=keys.narrow(0,0,full*4).view([full,4,dim]);
            let g=gates.narrow(0,0,full*4).view([full,4,dim]);
            let pools=((g+&w.ape).softmax(1,Kind::Float)*&k).sum_dim_intlist(&[1i64][..],false,Kind::Float);
            self.pools.narrow(0,start/4,full).copy_(&pools);
            self.tail_k.copy_(&k.get(full-1));self.tail_gate.copy_(&gates.narrow(0,(full-1)*4,4));
        }
        if remainder>0 {
            self.tail_k.narrow(0,0,remainder).copy_(&keys.narrow(0,full*4,remainder));
            self.tail_gate.narrow(0,0,remainder).copy_(&gates.narrow(0,full*4,remainder));
            let pool=((&self.tail_gate+&w.ape).softmax(0,Kind::Float)*&self.tail_k).sum_dim_intlist(&[0i64][..],true,Kind::Float);
            self.pools.narrow(0,start/4+full,1).copy_(&pool);
        }
        let mut result=Vec::new();
        // GLM53_DSA_PREFILL_QBLOCK=auto|N (L0: scores and top-k are per query row): queries per block. Default 128;
        // auto sizes the block to keep the [block, pools] score matrix near 64M floats (1024 at <= 64K pools).
        let qb:i64=match std::env::var("GLM53_DSA_PREFILL_QBLOCK").as_deref() {
            Ok("auto")=>((64i64<<20)/self.pools.size()[0].max(1)).clamp(128,1024)/128*128,
            Ok(v)=>v.parse().unwrap_or(128),_=>128};
        for first in (0..t).step_by(qb as usize) {
            let n=(t-first).min(qb);
            let complete=(Tensor::arange(n,(Kind::Int64,p.k.device()))+start+first+1).floor_divide_scalar(4);
            // The CPU knows the prefill extent. Retain the original score width
            // for topk padding/tie semantics, but never read future pool contents.
            let bounded=std::env::var("GLM53_DSA_PREFILL_LIMIT").as_deref()==Ok("1");
            let active=if bounded{(start+first+n)/4}else{self.pools.size()[0]};
            // P5: fused TF32 index scores over the active pools only; topk on that width, padded to
            // the fixed 512 slots with an index that is never visible (-1 after the valid mask).
            if bounded && prefill_score_fused_enabled() && dim==128 && p.q.size()[1]==32 && active>0 {
                let q=p.q.narrow(0,first,n).contiguous();let mixing=p.mixing.narrow(0,first,n).contiguous();
                assert_eq!(self.pools.kind(),Kind::Float);assert!(self.pools.is_contiguous());
                let scores=Tensor::empty([n,active],(Kind::Float,p.k.device()));
                let width=512.min(self.pools.size()[0]);let kk=width.min(active);
                let mut ids=if std::env::var("GLM53_DSA_PREFILL_SCORE_TILED").as_deref()==Ok("1") {
                    // Tiled kernel writes the visibility mask itself (pools >= this query's completed count -> MIN).
                    extern "C"{fn rs_dsa_prefill_scores_masked(q:*const f32,mixing:*const f32,pools:*const f32,out:*mut f32,n:i32,active:i32,first_pos:i64)->i32;}
                    assert_eq!(unsafe{rs_dsa_prefill_scores_masked(q.data_ptr().cast(),mixing.data_ptr().cast(),self.pools.data_ptr().cast(),scores.data_ptr().cast(),n as i32,active as i32,start+first)},0,"DSA prefill scores");
                    if let Some(out)=self.fast_select(&scores,n,active,kk,width,start,first,&complete) {result.push(out);continue;}
                    scores.topk(kk,1,true,true).1
                } else {
                    extern "C"{fn rs_dsa_prefill_scores(q:*const f32,mixing:*const f32,pools:*const f32,out:*mut f32,n:i32,active:i32)->i32;}
                    assert_eq!(unsafe{rs_dsa_prefill_scores(q.data_ptr().cast(),mixing.data_ptr().cast(),self.pools.data_ptr().cast(),scores.data_ptr().cast(),n as i32,active as i32)},0,"DSA prefill scores");
                    if let Some(out)=self.fast_select(&scores,n,active,kk,width,start,first,&complete) {result.push(out);continue;}
                    let visible=self.pool_positions.narrow(0,0,active).unsqueeze(0).lt_tensor(&complete.unsqueeze(1));
                    scores.masked_fill(&visible.logical_not(),f32::MIN as f64).topk(kk,1,true,true).1
                };
                if kk<width {ids=Tensor::cat(&[ids,Tensor::full([n,width-kk],self.pools.size()[0],(Kind::Int64,p.k.device()))],1);}
                let valid=ids.lt_tensor(&complete.unsqueeze(1));
                let tokens=(&ids.unsqueeze(-1)*4+&self.slot_offsets).masked_fill(&valid.logical_not().unsqueeze(-1),-1).view([n,-1]);
                let len=Tensor::arange(n,(Kind::Int64,p.k.device()))+start+first+1;
                let tail=complete.unsqueeze(1)*4+&self.tail_offsets;
                let tail=tail.masked_fill(&self.tail_offsets.unsqueeze(0).ge_tensor(&len.remainder(4).unsqueeze(1)),-1);
                result.push(Tensor::cat(&[tokens,tail],1));
                continue;
            }
            let scores=if bounded {
                let scores=Tensor::full([n,self.pools.size()[0]],f32::MIN as f64,(Kind::Float,p.k.device()));
                if active>0 {
                    let head=(p.q.narrow(0,first,n).matmul(&self.pools.narrow(0,0,active).transpose(0,1))*(dim as f64).powf(-0.5)).relu();
                    let value=(head*p.mixing.narrow(0,first,n)).sum_dim_intlist(&[1i64][..],false,Kind::Float);
                    scores.narrow(1,0,active).copy_(&value);
                }scores
            }else {
                let head=(p.q.narrow(0,first,n).matmul(&self.pools.transpose(0,1))*(dim as f64).powf(-0.5)).relu();
                (head*p.mixing.narrow(0,first,n)).sum_dim_intlist(&[1i64][..],false,Kind::Float)
            };
            let visible=self.pool_positions.unsqueeze(0).lt_tensor(&complete.unsqueeze(1));
            let ids=scores.masked_fill(&visible.logical_not(),f32::MIN as f64).topk(512.min(self.pools.size()[0]),1,true,true).1;
            let valid=ids.lt_tensor(&complete.unsqueeze(1));
            let tokens=(&ids.unsqueeze(-1)*4+&self.slot_offsets).masked_fill(&valid.logical_not().unsqueeze(-1),-1).view([n,-1]);
            let len=Tensor::arange(n,(Kind::Int64,p.k.device()))+start+first+1;
            let tail=complete.unsqueeze(1)*4+&self.tail_offsets;
            let tail=tail.masked_fill(&self.tail_offsets.unsqueeze(0).ge_tensor(&len.remainder(4).unsqueeze(1)),-1);
            result.push(Tensor::cat(&[tokens,tail],1));
        }
        Tensor::cat(&result,0)
    }

    pub fn new(capacity: i64, dim: i64, dev: Device) -> Self {
        Self::with_pools(capacity, dim, dev, Tensor::zeros([(capacity+3)/4,dim],(Kind::Float,dev)))
    }
    /// `new` with the pool rows supplied (zero-filled [(capacity+3)/4, dim] FP32, e.g. a KV pool view).
    pub(crate) fn with_pools(capacity: i64, dim: i64, dev: Device, pools: Tensor) -> Self {
        assert!(capacity>0);
        assert_eq!(pools.size(),[(capacity+3)/4,dim],"DSA pool rows");assert_eq!(pools.kind(),Kind::Float);
        let pool_positions=Tensor::arange((capacity+3)/4,(Kind::Int64,dev));
        let slots=4*512.min((capacity+3)/4);
        // Reuse the existing ascending pool-index storage when it is long
        // enough. This immutable view must exist before graph capture, so
        // every sibling shares it and no replay launches another arange.
        let all_tokens=if pool_positions.numel() as i64>=slots{pool_positions.narrow(0,0,slots)}
            else{Tensor::arange(slots,(Kind::Int64,dev))};
        Self {pool_positions,all_tokens,
            slot_offsets:Tensor::arange(4,(Kind::Int64,dev)),tail_offsets:Tensor::arange(3,(Kind::Int64,dev)),
            pools,
            tail_k:Tensor::zeros([4,dim],(Kind::Float,dev)),
            tail_gate:Tensor::zeros([4,dim],(Kind::Float,dev)) }
    }
    pub fn snapshot(&self) -> Self {
        Self{pools:self.pools.copy(),tail_k:self.tail_k.copy(),tail_gate:self.tail_gate.copy(),
            all_tokens:self.all_tokens.shallow_clone(),
            pool_positions:self.pool_positions.shallow_clone(),slot_offsets:self.slot_offsets.shallow_clone(),tail_offsets:self.tail_offsets.shallow_clone()}
    }
    /// Shallow alias (same storage); used when a state is the verifier graph's own base.
    pub(crate) fn alias(&self)->Self {
        Self{pools:self.pools.shallow_clone(),tail_k:self.tail_k.shallow_clone(),tail_gate:self.tail_gate.shallow_clone(),
            all_tokens:self.all_tokens.shallow_clone(),pool_positions:self.pool_positions.shallow_clone(),
            slot_offsets:self.slot_offsets.shallow_clone(),tail_offsets:self.tail_offsets.shallow_clone()}
    }
    /// Shares pools (and index constants) with self; fresh tails for a chain verifier node.
    pub(crate) fn chain_child(&self)->Self {
        Self{pools:self.pools.shallow_clone(),tail_k:Tensor::empty_like(&self.tail_k),tail_gate:Tensor::empty_like(&self.tail_gate),
            all_tokens:self.all_tokens.shallow_clone(),pool_positions:self.pool_positions.shallow_clone(),
            slot_offsets:self.slot_offsets.shallow_clone(),tail_offsets:self.tail_offsets.shallow_clone()}
    }
    pub fn restore(&mut self, src: &Self) {
        self.pools.copy_(&src.pools);self.tail_k.copy_(&src.tail_k);self.tail_gate.copy_(&src.tail_gate);
    }
    pub fn snapshot_active(&self,len:&Tensor)->Self {
        let pools=Tensor::empty_like(&self.pools);
        crate::mla_latent::active_copy(&pools,&self.pools,len,4);
        Self{pools,tail_k:self.tail_k.copy(),tail_gate:self.tail_gate.copy(),
            all_tokens:self.all_tokens.shallow_clone(),
            pool_positions:self.pool_positions.shallow_clone(),slot_offsets:self.slot_offsets.shallow_clone(),tail_offsets:self.tail_offsets.shallow_clone()}
    }
    /// Pools' active rows only; tails are left uninitialized for a fused node append to fill.
    pub(crate) fn snapshot_pools_for_append(&self,len:&Tensor)->Self {
        let pools=Tensor::empty_like(&self.pools);
        crate::mla_latent::active_copy(&pools,&self.pools,len,4);
        Self{pools,tail_k:Tensor::empty_like(&self.tail_k),tail_gate:Tensor::empty_like(&self.tail_gate),
            all_tokens:self.all_tokens.shallow_clone(),
            pool_positions:self.pool_positions.shallow_clone(),slot_offsets:self.slot_offsets.shallow_clone(),tail_offsets:self.tail_offsets.shallow_clone()}
    }
    /// Score part of append_score, reading pools already updated by the fused node append.
    pub(crate) fn score_row(&self,w:&Weights,p:&Projected,row:i64,pos:&Tensor)->Tensor {
        let dim=w.k.size()[0];let q=p.q.get(row);let mixing=p.mixing.get(row);
        let fused=matches!(std::env::var("GLM53_DSA_SCORE_FUSED").as_deref(),Ok("1"|"2"|"3"|"4"|"5")) && dim==128 && q.size()[0]==32;
        if fused{score_fused(&q,&self.pools,&mixing,pos)}else {
            let head_score=(q.matmul(&self.pools.transpose(0,1))*(dim as f64).powf(-0.5)).relu();
            (head_score*mixing).sum_dim_intlist(&[0i64][..],false,Kind::Float)
        }
    }
    pub fn restore_active(&mut self,src:&Self,len:&Tensor) {
        crate::mla_latent::active_copy(&self.pools,&src.pools,len,4);
        self.tail_k.copy_(&src.tail_k);self.tail_gate.copy_(&src.tail_gate);
    }
    /// pos is the zero-based position of x, on device, fixed address during graph replay.
    pub fn append_select(&mut self,w:&Weights,x:&Tensor,q_resid:&Tensor,pos:&Tensor)->Tensor {
        let p=w.project(x,q_resid);
        self.append_projected(w,&p,0,pos)
    }
    pub fn append_projected(&mut self,w:&Weights,p:&Projected,row:i64,pos:&Tensor)->Tensor {
        let dim=w.k.size()[0];
        let constants=std::env::var("GLM53_STATIC_TENSORS").as_deref()==Ok("1");
        let k=p.k.narrow(0,row,1);let gate=p.gate.narrow(0,row,1);
        self.append_keys(w,&k,&gate,pos);
        let q=p.q.get(row);
        let mixing=p.mixing.get(row);
        let fused=matches!(std::env::var("GLM53_DSA_SCORE_FUSED").as_deref(),Ok("1"|"2"|"3"|"4"|"5")) && dim==128 && q.size()[0]==32;
        let scores=if fused{score_fused(&q,&self.pools,&mixing,pos)}else {
            let head_score=(q.matmul(&self.pools.transpose(0,1))*(dim as f64).powf(-0.5)).relu();
            (head_score*mixing).sum_dim_intlist(&[0i64][..],false,Kind::Float)
        };
        // Ranked-only bookkeeping: score arithmetic and sorted topk stay
        // unchanged; runtime pos remains a device input in both kernels.
        if let Some(tokens)=crate::dsa_index::try_ranked(&scores,pos) {return tokens;}
        let len=pos+1;
        let complete=len.floor_divide_scalar(4);
        let positions=if constants{self.pool_positions.shallow_clone()}else{Tensor::arange(self.pools.size()[0],(Kind::Int64,k.device()))};
        let visible=positions.lt_tensor(&complete);
        let scores=scores.masked_fill(&visible.logical_not(),f32::MIN as f64);
        let selected=scores.topk(512.min(self.pools.size()[0]),0,true,true).1;
        let valid=selected.lt_tensor(&complete);
        let slots=if constants{self.slot_offsets.shallow_clone()}else{Tensor::arange(4,(Kind::Int64,k.device()))};
        let tokens=(&selected.unsqueeze(1)*4+slots)
            .masked_fill(&valid.logical_not().unsqueeze(1),-1).reshape([-1]);
        let offsets=if constants{self.tail_offsets.shallow_clone()}else{Tensor::arange(3,(Kind::Int64,k.device()))};
        let tail=&complete*4+&offsets;
        let tail=tail.masked_fill(&offsets.ge_tensor(&len.remainder(4)),-1);
        Tensor::cat(&[tokens,tail],0)
    }
    /// Candidate seam: same row-local append/GEMM/scale/relu/reduce as
    /// append_projected. Ranking is delayed; no next-state update uses it.
    /// Keep append_projected itself byte-for-byte unchanged for flag0.
    pub(crate) fn append_score(&mut self,w:&Weights,p:&Projected,row:i64,pos:&Tensor)->Tensor {
        let dim=w.k.size()[0];
        let k=p.k.narrow(0,row,1);let gate=p.gate.narrow(0,row,1);
        self.append_keys(w,&k,&gate,pos);
        let q=p.q.get(row);
        let mixing=p.mixing.get(row);
        let fused=matches!(std::env::var("GLM53_DSA_SCORE_FUSED").as_deref(),Ok("1"|"2"|"3"|"4"|"5")) && dim==128 && q.size()[0]==32;
        let scores=if fused{score_fused(&q,&self.pools,&mixing,pos)}else {
            let head_score=(q.matmul(&self.pools.transpose(0,1))*(dim as f64).powf(-0.5)).relu();
            (head_score*mixing).sum_dim_intlist(&[0i64][..],false,Kind::Float)
        };
        scores
    }
    pub(crate) fn append_keys(&mut self,w:&Weights,k:&Tensor,gate:&Tensor,pos:&Tensor) {
        let offset=pos.remainder(4);
        let _=self.tail_k.index_copy_(0,&offset,k);
        let _=self.tail_gate.index_copy_(0,&offset,gate);
        let pool=((&self.tail_gate+&w.ape).softmax(0,Kind::Float)*&self.tail_k)
            .sum_dim_intlist(&[0i64][..],true,Kind::Float);
        let pool_pos=pos.floor_divide_scalar(4);
        // Incomplete pool content is provisional and is masked out below. Once
        // the fourth token arrives every tail slot belongs to this pool.
        let _=self.pools.index_copy_(0,&pool_pos,&pool);
    }
    /// Caller must qualify the whole tree before capture/replay. This mode
    /// preserves complete-pool + invalid-padding + three-tail-slot layout,
    /// but changes full-pool order to ascending and is not bitwise equivalent.
    pub fn append_all_visible(&mut self,w:&Weights,p:&KeyProjected,row:i64,pos:&Tensor)->Tensor {
        self.append_keys(w,&p.k.narrow(0,row,1),&p.gate.narrow(0,row,1),pos);
        let ids=&self.all_tokens;
        let len=pos+1;
        let complete=len.floor_divide_scalar(4);
        let full_len=&complete*4;
        let tokens=ids.masked_fill(&ids.ge_tensor(&full_len),-1);
        let tail=&full_len+&self.tail_offsets;
        let tail=tail.masked_fill(&self.tail_offsets.ge_tensor(&len.remainder(4)),-1);
        Tensor::cat(&[tokens,tail],0)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batched_pool_causality_and_partial_tail() {
        tch::set_num_threads(1);tch::manual_seed(78);let dev=Device::Cpu;let dim=4;
        let w=Weights{q:Tensor::zeros([8,4],(Kind::Float,dev)),k:Tensor::zeros([dim,4],(Kind::Float,dev)),
            norm_w:Tensor::ones([dim],(Kind::Float,dev)),norm_b:Tensor::zeros([dim],(Kind::Float,dev)),
            score:Tensor::zeros([2,4],(Kind::Float,dev)),ape:Tensor::randn([4,dim],(Kind::Float,dev)),gate:Tensor::zeros([dim,4],(Kind::Float,dev))};
        let p=Projected{k:Tensor::randn([32,dim],(Kind::Float,dev)),gate:Tensor::randn([32,dim],(Kind::Float,dev)),q:Tensor::randn([32,2,dim],(Kind::Float,dev)),mixing:Tensor::randn([32,2,1],(Kind::Float,dev))};
        for start in [0,1,3,4,7]{for count in [1,2,3,4,5,9]{
            let mut serial=State::new(32,dim,dev);
            for i in 0..start{let _=serial.append_projected(&w,&p,i,&Tensor::from_slice(&[i]));}
            let mut batch=serial.snapshot();let part=Projected{k:p.k.narrow(0,start,count),gate:p.gate.narrow(0,start,count),q:p.q.narrow(0,start,count),mixing:p.mixing.narrow(0,start,count)};
            let ids=batch.append_chunk(&w,&part,start);
            for i in 0..count{
                let expected=serial.append_projected(&w,&p,start+i,&Tensor::from_slice(&[start+i]));
                let valid=|x:Tensor|{let v:Vec<i64>=Vec::try_from(&x).unwrap();let mut v:Vec<_>=v.into_iter().filter(|&i|i>=0).collect();v.sort();v};
                assert_eq!(valid(ids.get(i)),valid(expected));assert_eq!(valid(ids.get(i)),(0..=start+i).collect::<Vec<_>>());
            }
            assert!(batch.tail_k.equal(&serial.tail_k)&&batch.tail_gate.equal(&serial.tail_gate));
            assert!((&batch.pools-&serial.pools).abs().max().double_value(&[])<1e-6);
        }}
    }
}

pub(crate) fn score_fused(q:&Tensor,pools:&Tensor,mixing:&Tensor,pos:&Tensor)->Tensor {
    assert_eq!(q.size(),[32,128]);assert_eq!(pools.size()[1],128);
    for t in [q,pools,mixing]{assert!(t.is_contiguous());assert_eq!(t.kind(),Kind::Float);}
    assert!(pos.is_contiguous());assert_eq!(pos.numel(),1);assert_eq!(pos.kind(),Kind::Int64);
    let out=Tensor::empty([pools.size()[0]],(Kind::Float,q.device()));
    extern "C" {fn rs_dsa_score(q:*const f32,pools:*const f32,mixing:*const f32,pos:*const i64,out:*mut f32,capacity:i32,mode:i32)->i32;}
    assert_eq!(unsafe{rs_dsa_score(q.data_ptr().cast(),pools.data_ptr().cast(),mixing.data_ptr().cast(),pos.data_ptr().cast(),out.data_ptr().cast(),pools.size()[0] as i32,std::env::var("GLM53_DSA_SCORE_FUSED").ok().and_then(|v|v.parse().ok()).unwrap_or(1))},0);out
}

fn probe_index_tree(w:&Weights,x:&Tensor,cq:&Tensor,base:&State,pos:&Tensor,parents:&[Option<usize>],selection:TreeSelection)->(Tensor,Vec<State>) {
    let p=(selection==TreeSelection::Ranked).then(||w.project(x,cq));
    let k=(selection==TreeSelection::AllVisible).then(||w.project_keys(x));
    let mut states:Vec<State>=Vec::new();let mut positions:Vec<Tensor>=Vec::new();let mut ids=Vec::new();
    for (i,&parent) in parents.iter().enumerate() {
        let mut state=parent.map_or(base,|p|&states[p]).snapshot();
        let position=parent.map_or_else(||pos.shallow_clone(),|p|&positions[p]+1);
        ids.push(match selection {
            TreeSelection::Ranked=>state.append_projected(w,p.as_ref().unwrap(),i as i64,&position),
            TreeSelection::AllVisible=>state.append_all_visible(w,k.as_ref().unwrap(),i as i64,&position),
        });states.push(state);positions.push(position);
    }
    (Tensor::stack(&ids,0),states)
}

/// Bounded local correctness gate. Ascending full-pool order intentionally
/// changes FP32 accumulation; membership/state/graph semantics must be exact,
/// while attention is independently checked against a double-precision oracle.
pub fn all_visible_probe(out:&std::path::Path) {
    use serde_json::json;
    assert!(!crate::tp::is_tp(),"DSA local probe expects a single GPU");
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let dev=Device::Cuda(0);
    std::fs::create_dir_all(out).unwrap();tch::manual_seed(92351);
    std::env::set_var("GLM53_DSA_ALL_VISIBLE","1");std::env::set_var("GLM53_DSA_SCORE_FUSED","0");
    let capacity=20480;let dim=128;
    let w=Weights{q:Tensor::randn([4096,16],(Kind::Float,dev))*0.02,k:Tensor::randn([dim,16],(Kind::Float,dev))*0.02,
        norm_w:Tensor::ones([dim],(Kind::Float,dev)),norm_b:Tensor::zeros([dim],(Kind::Float,dev)),
        score:Tensor::randn([32,16],(Kind::Float,dev))*0.02,ape:Tensor::randn([4,dim],(Kind::Float,dev)),gate:Tensor::randn([dim,16],(Kind::Float,dev))*0.02};
    let equivalent=|a:&Tensor,b:&Tensor|a.eq_tensor(b).logical_or(&a.isnan().logical_and(&b.isnan())).all().int64_value(&[])!=0;
    let check_states=|a:&[State],b:&[State]|{
        assert_eq!(a.len(),b.len());
        for (a,b) in a.iter().zip(b){assert!(equivalent(&a.pools,&b.pools)&&a.tail_k.equal(&b.tail_k)&&a.tail_gate.equal(&b.tail_gate),"DSA key/pool/tail state mismatch");}
    };
    assert_eq!(tree_selection(2043,8),TreeSelection::AllVisible);
    assert_eq!(tree_selection(2044,8),TreeSelection::Ranked);
    assert_eq!(tree_selection(2050,1),TreeSelection::AllVisible);
    assert_eq!(tree_selection(2051,1),TreeSelection::Ranked);
    let layouts=vec![(0..8).map(|i:usize|i.checked_sub(1)).collect::<Vec<_>>(),
        vec![None,Some(0),Some(0),Some(1),Some(2),Some(1),Some(2),Some(6)],vec![None]];
    let mut records=Vec::new();
    std::fs::write(out.join("dsa-all-visible.json"),r#"{"gate":false,"complete":false,"cases":[]}"#).unwrap();
    for parents in layouts {
        let n=parents.len() as i64;let mut depths=Vec::new();
        for &parent in &parents{depths.push(parent.map_or(1i64,|p|depths[p]+1));}
        let max_depth=*depths.iter().max().unwrap();
        let mut base=State::new(capacity,dim,dev);
        let pool_values=Tensor::randn_like(&base.pools)*0.1;let tail_k=Tensor::randn_like(&base.tail_k);
        let tail_gate=Tensor::randn_like(&base.tail_gate);let mut pos=Tensor::zeros([1],(Kind::Int64,dev));
        let original_x=Tensor::randn([n,16],(Kind::Float,dev));let mut x=original_x.copy();let cq=Tensor::randn_like(&x);
        let latent_values=Tensor::randn([capacity,512],(Kind::Half,dev))*0.1;let mut latent=latent_values.copy();
        let q=Tensor::randn([32,1,512],(Kind::Float,dev))*0.3;
        let mut graphs:Vec<(TreeSelection,crate::tp::graph::Owned,Tensor,Vec<State>,Tensor)>=Vec::new();
        for start in [0i64,1,3,127,2043,2044,2047,2048,2050,2051,2052,7,0] {
            let selection=tree_selection(start,max_depth);let _=pos.fill_(start);
            base.pools.copy_(&pool_values);let active=(start+3)/4;
            let _=base.pools.narrow(0,active,base.pools.size()[0]-active).fill_(f64::NAN);
            base.tail_k.copy_(&tail_k);base.tail_gate.copy_(&tail_gate);
            x.copy_(&(&original_x*if start%2==0{1.}else{-1.}));
            latent.copy_(&latent_values);let visible=start+max_depth;
            let _=latent.narrow(0,visible,capacity-visible).fill_(f64::NAN);
            let before=base.snapshot();
            let reference=probe_index_tree(&w,&x,&cq,&base,&pos,&parents,TreeSelection::Ranked);
            let expected=probe_index_tree(&w,&x,&cq,&base,&pos,&parents,selection);
            check_states(&expected.1,&reference.1);
            assert!(expected.0.sort(-1,false).0.equal(&reference.0.sort(-1,false).0),"AllVisible changed selected set start={start}");
            if selection==TreeSelection::AllVisible {
                for (i,&depth) in depths.iter().enumerate() {
                    let ids:Vec<i64>=Vec::try_from(expected.0.get(i as i64).to_device(Device::Cpu)).unwrap();
                    assert_eq!(ids.into_iter().filter(|&id|id>=0).collect::<Vec<_>>(),(0..start+depth).collect::<Vec<_>>());
                }
            }
            let cached=graphs.iter().position(|g|g.0==selection);let hit=cached.is_some();
            let index=if let Some(i)=cached{i}else{
                let _=probe_index_tree(&w,&x,&cq,&base,&pos,&parents,selection);
                let _=crate::mla_latent::sparse_attention(&q,&latent,&expected.0.get(n-1));tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();
                let (ids,states)=probe_index_tree(&w,&x,&cq,&base,&pos,&parents,selection);
                let attention=crate::mla_latent::sparse_attention(&q,&latent,&ids.get(n-1));
                crate::tp::graph::end().unwrap();
                graphs.push((selection,crate::tp::graph::Owned::take(),ids,states,attention));assert!(graphs.len()<=2);graphs.len()-1
            };
            let (_,graph,ids,states,attention)=&graphs[index];graph.replay();
            assert!(ids.equal(&expected.0),"DSA graph changed-length indices mismatch");check_states(states,&expected.1);
            assert!(equivalent(&base.pools,&before.pools)&&base.tail_k.equal(&before.tail_k)&&base.tail_gate.equal(&before.tail_gate),"DSA graph mutated base");
            let eager=crate::mla_latent::sparse_attention(&q,&latent,&expected.0.get(n-1));
            assert!(attention.equal(&eager)&&attention.isfinite().all().int64_value(&[])!=0,"DSA graph attention/input length mismatch");
            let old=crate::mla_latent::sparse_attention(&q,&latent,&reference.0.get(n-1));
            let delta=(attention-&old).abs().max().double_value(&[]);
            let old_relative=((attention-&old).norm()/old.norm().clamp_min(1e-30)).double_value(&[]);
            let mut oracle_relative=None;
            if selection==TreeSelection::AllVisible {
                let len=start+depths[depths.len()-1];let values=latent.narrow(0,0,len).to_kind(Kind::Double);
                let query=q.to_kind(Kind::Double);let probabilities=(query.matmul(&values.transpose(0,1))/16.).softmax(-1,Kind::Double);
                let oracle=probabilities.matmul(&values);
                let relative=((attention.to_kind(Kind::Double)-&oracle).norm()/oracle.norm().clamp_min(1e-30)).double_value(&[]);
                // Same 2e-5 relative bound as the existing sparse-attention probe.
                assert!(relative<2e-5,"AllVisible attention differs from full-visible FP64 oracle: {relative}");oracle_relative=Some(relative);
            }
            records.push(json!({"parents":parents,"start":start,"max_depth":max_depth,"selection":format!("{selection:?}"),
                "graph_hit":hit,"live_graphs":graphs.len(),"same_set":true,"same_input_states_exact":true,"graph_eager_exact":true,
                "nan_suffix_safe":true,"order_exact":expected.0.equal(&reference.0),"attention_max_abs_vs_ranked":delta,
                "attention_relative_vs_ranked":old_relative,"attention_relative_vs_double":oracle_relative}));
            std::fs::write(out.join("dsa-all-visible.json"),serde_json::to_string_pretty(&json!({"gate":false,"complete":false,"cases":records,
                "scope":"local synthetic projection dimensions; full-model numerical/task/acceptance qualification still required"})).unwrap()).unwrap();
        }
    }
    std::fs::write(out.join("dsa-all-visible.json"),serde_json::to_string_pretty(&json!({"gate":true,"complete":true,"cases":records,
        "scope":"local synthetic projection dimensions; full-model numerical/task/acceptance qualification still required"})).unwrap()).unwrap();
    eprintln!("[dsa-all-visible] membership/state, 2051/2052 boundary, sibling trees, two-regime graph restore and NaN gates passed");
}

/// P5 gate: fused prefill index scores vs the ATen TF32 path on random inputs (score error and
/// top-512 set overlap per query).
pub fn prefill_score_probe() {
    for (tiled,f16) in [("1","0"),("1","1")] {std::env::set_var("GLM53_DSA_PREFILL_SCORE_TILED",tiled);std::env::set_var("GLM53_DSA_PREFILL_SCORE_F16",f16);
        eprintln!("[dsa-prefill-score] tiled={tiled} f16={f16}");prefill_score_probe_once();}
}
fn prefill_score_probe_once() {
    let _g=tch::no_grad_guard();let dev=Device::Cuda(0);tch::manual_seed(5);
    crate::tp::set_tf32(true);
    for &(n,active) in &[(128i64,37i64),(128,512),(128,2048),(128,25000)] {
        let q=Tensor::randn([n,32,128],(Kind::Float,dev));let pools=Tensor::randn([active+100,128],(Kind::Float,dev))*0.5;
        let mixing=Tensor::randn([n,32,1],(Kind::Float,dev))*(32f64).powf(-0.5);
        let head=(q.matmul(&pools.narrow(0,0,active).transpose(0,1))*(128f64).powf(-0.5)).relu();
        let aten=(head*&mixing).sum_dim_intlist(&[1i64][..],false,Kind::Float);
        let out=Tensor::empty([n,active],(Kind::Float,dev));
        extern "C"{fn rs_dsa_prefill_scores(q:*const f32,mixing:*const f32,pools:*const f32,out:*mut f32,n:i32,active:i32)->i32;}
        assert_eq!(unsafe{rs_dsa_prefill_scores(q.data_ptr().cast(),mixing.data_ptr().cast(),pools.data_ptr().cast(),out.data_ptr().cast(),n as i32,active as i32)},0);
        let rel=f64::try_from((&out-&aten).norm()/aten.norm()).unwrap();
        let k=512.min(active);let a=aten.topk(k,1,true,true).1;let b=out.topk(k,1,true,true).1;
        let mut overlap=0f64;
        for i in 0..n {let x:Vec<i64>=Vec::try_from(a.get(i)).unwrap();let y:std::collections::HashSet<i64>=Vec::<i64>::try_from(b.get(i)).unwrap().into_iter().collect();
            overlap+=x.iter().filter(|v|y.contains(v)).count() as f64/k as f64;}
        eprintln!("[dsa-prefill-score] n {n} active {active}: rel {rel:.2e} top{k} overlap {:.4}",overlap/n as f64);
    }
    for &active in &[12500i64,25000,50000,75000] {
        let n=128i64;let q=Tensor::randn([n,32,128],(Kind::Float,dev));let pools=Tensor::randn([active,128],(Kind::Float,dev))*0.5;
        let mixing=Tensor::randn([n,32,1],(Kind::Float,dev));let out=Tensor::empty([n,active],(Kind::Float,dev));
        extern "C"{fn rs_dsa_prefill_scores(q:*const f32,mixing:*const f32,pools:*const f32,out:*mut f32,n:i32,active:i32)->i32;}
        let time=|f:&dyn Fn()|{f();tch::Cuda::synchronize(0);let t0=std::time::Instant::now();for _ in 0..16{f();}tch::Cuda::synchronize(0);t0.elapsed().as_secs_f64()*1000./16.};
        let ms_score=time(&||{assert_eq!(unsafe{rs_dsa_prefill_scores(q.data_ptr().cast(),mixing.data_ptr().cast(),pools.data_ptr().cast(),out.data_ptr().cast(),n as i32,active as i32)},0);});
        let complete=Tensor::arange(n,(Kind::Int64,dev))+active-n;let pos=Tensor::arange(active,(Kind::Int64,dev));
        let ms_mask=time(&||{let v=pos.unsqueeze(0).lt_tensor(&complete.unsqueeze(1));let _=out.masked_fill(&v.logical_not(),f32::MIN as f64);});
        let ms_topk=time(&||{let _=out.topk(512,1,true,true);});
        eprintln!("[dsa-prefill-score] 128 queries x {active} pools: score {ms_score:.2} ms, mask {ms_mask:.2} ms, topk512 {ms_topk:.2} ms (x16 per 2K chunk per layer)");
    }
    crate::tp::set_tf32(false);
    let q=Tensor::randn([128,32,128],(Kind::Float,dev));let pools=Tensor::randn([200,128],(Kind::Float,dev));let mixing=Tensor::randn([128,32,1],(Kind::Float,dev));
    let exact=((q.to_kind(Kind::Double).matmul(&pools.to_kind(Kind::Double).transpose(0,1))*(128f64).powf(-0.5)).relu()*mixing.to_kind(Kind::Double)).sum_dim_intlist(&[1i64][..],false,Kind::Double);
    let out=Tensor::empty([128,200],(Kind::Float,dev));
    extern "C"{fn rs_dsa_prefill_scores(q:*const f32,mixing:*const f32,pools:*const f32,out:*mut f32,n:i32,active:i32)->i32;}
    assert_eq!(unsafe{rs_dsa_prefill_scores(q.data_ptr().cast(),mixing.data_ptr().cast(),pools.data_ptr().cast(),out.data_ptr().cast(),128,200)},0);
    let tf32=((q.matmul(&pools.transpose(0,1))*(128f64).powf(-0.5)).relu()*&mixing).sum_dim_intlist(&[1i64][..],false,Kind::Float);
    let e=|x:&Tensor|f64::try_from((x.to_kind(Kind::Double)-&exact).norm()/exact.norm()).unwrap();
    eprintln!("[dsa-prefill-score] vs FP64: fused {:.2e}, ATen FP32 (TF32 off) {:.2e}",e(&out),e(&tf32));
}
