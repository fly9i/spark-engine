//! Single-engine prefix state pool. Entries own mutable tensors; checkouts are copies.
//! Cache identity is tied to one Engine and one frozen execution configuration.
use crate::forward::{Engine,DecodeStates,LayerState,snapshot};
use tch::Tensor;

struct Entry { ids:Vec<i64>, logits:Tensor, state:DecodeStates, used:u64 }

pub struct PrefixPool {
    entries:Vec<Entry>, slots:usize, clock:u64,
    pub hits:usize, pub misses:usize, pub evictions:usize,
}

impl PrefixPool {
    pub fn new(slots:usize)->Self {Self{entries:Vec::new(),slots,clock:0,hits:0,misses:0,evictions:0}}
    pub fn len(&self)->usize {self.entries.len()}
    pub fn checkout(&mut self,ids:&[i64])->Option<(usize,Tensor,DecodeStates)> {
        let best=self.entries.iter().enumerate().filter(|(_,e)|ids.starts_with(&e.ids))
            .max_by_key(|(_,e)|e.ids.len()).map(|(i,_)|i);
        self.clock+=1;
        if let Some(i)=best {
            self.hits+=1;let e=&mut self.entries[i];e.used=self.clock;
            Some((e.ids.len(),e.logits.copy(),snapshot(&e.state)))
        } else {self.misses+=1;None}
    }
    pub fn insert(&mut self,ids:&[i64],logits:&Tensor,state:&DecodeStates) {
        if self.slots==0{return;}
        assert!(!ids.is_empty());self.clock+=1;
        let same=self.entries.iter().position(|e|e.ids==ids);
        let target=same.or_else(||if self.entries.len()==self.slots {
            self.evictions+=1;
            self.entries.iter().enumerate().min_by_key(|(_,e)|e.used).map(|(i,_)|i)
        }else{None});
        if let Some(i)=target {
            let e=&mut self.entries[i];
            // Reuse pool allocations only when mutable shapes match exactly.
            if compatible(&e.state,state) {crate::forward::restore(&mut e.state,state);}
            else {e.state=snapshot(state);}
            e.ids=ids.to_vec();e.logits=logits.copy();e.used=self.clock;
        } else {self.entries.push(Entry{ids:ids.to_vec(),logits:logits.copy(),state:snapshot(state),used:self.clock});}
    }
}

fn compatible(a:&DecodeStates,b:&DecodeStates)->bool {
    a.0.len()==b.0.len() && a.0.iter().zip(&b.0).all(|(a,b)|match(a,b) {
        (LayerState::Kda(a),LayerState::Kda(b))=>a.h.size()==b.h.size()&&a.conv.size()==b.conv.size(),
        (LayerState::Mla(a),LayerState::Mla(b))=>a.k.size()==b.k.size()&&a.v.size()==b.v.size(),
        (LayerState::MlaG(a),LayerState::MlaG(b))=>a.max_t==b.max_t&&a.k.size()==b.k.size(),
        (LayerState::MlaLatent(a),LayerState::MlaLatent(b))=>a.capacity==b.capacity&&a.latent.size()==b.latent.size(),
        _=>false,
    })
}

/// Owns the engine so cached states cannot be accidentally applied to another model.
/// Changing execution flags after constructing a Session is unsupported.
pub struct Session {pub engine:Engine,pub prefixes:PrefixPool,pub chunk_size:usize,signature:Vec<Option<String>>}
pub(crate) fn signature()->Vec<Option<String>> {
    ["GLM53_RELEASE_FILE_CACHE","GLM53_C12","GLM53_C12_INDEX","GLM53_C12_KS","GLM53_C12_QU","GLM53_C12_PREFILL","GLM53_C12_FREE_SOURCE","GLM53_C12_BIG_STAGES","GLM53_C12_L0CHECK","GLM53_C12_MIN_ROWS","GLM53_C12_MAX_ROWS","GLM53_PREFILL_SP","GLM53_MLA_PREFILL_F16","GLM53_PREFILL_MOE_SUM1","GLM53_FP8_PREFILL_HALF","GLM53_DSA_PREFILL_SCORE_F16","GLM53_DSA_PREFILL_SCORE_TILED","GLM53_DSA_PREFILL_SCORE_FUSED","GLM53_KV_FP8","GLM53_MLA_CHAIN_SHARED","GLM53_CHAIN_SHARED_BASE","GLM53_MHC_POST_PRE_FUSED","GLM53_HALF_INPUT_CACHE","GLM53_PREFILL_FAT_MOE","GLM53_MHC_PRE_LARGE","GLM53_MLA_PREFILL_TC","GLM53_KDA_PREFILL_FUSED","GLM53_PREFILL_EXPERT_STREAMS","GLM53_RDMA_AR","GLM53_DRAFT_FUSED_NORM","GLM53_MHC_PRE_TC","GLM53_HALF_SKINNY_K1536","GLM53_KDA_CORR_WARP","GLM53_NORM_FUSED","GLM53_SPEC_CONF_TAU_PCT","GLM53_DRAFT_APPEND_GRAPH","GLM53_DRAFT_CONF_TRUNC","GLM53_SPEC_CONF_TAU","GLM53_HALF_SKINNY","GLM53_DRAFT_GRAPH","GLM53_STATE_INPLACE","GLM53_MLA_HALF_BMM","GLM53_ROUTER_FUSED","GLM53_DSA_NODE_FUSED","GLM53_KDA_GATE_FUSED","GLM53_MHC_PRE_FUSED","GLM53_FP8_QKV_FUSED","GLM53_FP8_SKINNY","GLM53_FP8_SKINNY_KS","GLM53_SHARED_GU_FUSED","GLM53_FP8_SMALL_TRANSPOSE","GLM53_FP8_SMALL_PAD","GLM53_DRAFT_FINAL_NORM_SELECT","GLM53_DRAFT_CONV_FUSED","GLM53_DRAFT_SELECTOR_FUSED","GLM53_ACCEPTED_FEATURE_VIEW","GLM53_DSA_POSITION_CAPTURE","GLM53_DSA_TOPK_BATCH","GLM53_DSA_INDEX_FUSED","GLM53_COOP_PREFETCH_MASK","GLM53_KDA_CONV_DEFERRED","GLM53_MOE_INPUT_HALF_REUSE","GLM53_KDA_CORRECTION_REPLAY","GLM53_MHC_POST_FOUR_STREAMS","GLM53_MHC_POST_PACKED","GLM53_COOP_PREFETCH","GLM53_MLA_WEIGHT_CACHE","GLM53_TARGET_TOP1_TP","GLM53_COOP_SHARED_INPUT","GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT","GLM53_FP8_SPLITS","GLM53_COOP_LOCAL_GU","GLM53_COOP_TRANSPOSE","GLM53_COOP_CANDIDATE","GLM53_MOE_NO_COPY","GLM53_KDA_CONV_CHAIN","GLM53_KDA_CHAIN_NORM","GLM53_KDA_CHAIN_RECURRENT","GLM53_COOP_CANDIDATE_SO","GLM53_COOP_CANDIDATE_SHA","GLM53_FP8_SHARED","GLM53_FP8_MLA","GLM53_FP8_HEAD","GLM53_HOST_TRIM","GLM53_GROUPED_INDEX_PACK","GLM53_FP8_EPILOGUE","GLM53_GROUPED_SWIGLU","GLM53_FP8_LARGE","GLM53_DSA_SCORE_FUSED","GLM53_PREFIX_ADMISSION","GLM53_DSA_PREFILL_LIMIT","GLM53_GROUPED_REDUCE","GLM53_MLA_ACTIVE_COPY","GLM53_PREFILL_MIN_ROWS","GLM53_FP8_WMMA","GLM53_MLA_PREFILL_DENSE","GLM53_PREFILL_RECON_MIN_ROWS","GLM53_DENSE_FP8","GLM53_KDA_FP8","GLM53_TF32","GLM53_PREFILL_LAST_LOGITS","GLM53_COOP_GEOMETRY","GLM53_DENSE_LT","GLM53_DENSE_LT_TABLE","GLM53_PREFILL_GROUPED","GLM53_KDA_SEQUENCE","GLM53_MLA_SPARSE_FUSED","GLM53_MLA_PREFILL_BATCHED","GLM53_PREFILL_COOP","GLM53_MHC_FUSED","GLM53_MHC_POST_FUSED","GLM53_KDA_FUSED","GLM53_KDA_FORK_FUSED","GLM53_STATIC_TENSORS","GLM53_MLA_LATENT","GLM53_MLA_SCORE_2D","GLM53_MLA_SCORE_2D_SCOPE","GLM53_MLA_SCORE_2D_MIN_ROWS","GLM53_MAX_CONTEXT",
     "GLM53_DENSE_TP","GLM53_TP_MOE_PACK","GLM53_TP_MOE_PACK_MAX_ROWS","GLM53_TP_SMALL_COMM","GLM53_TP_SMALL_COMM_ACTIVE","GLM53_VOCAB_TP","GLM53_DENSE_GEMV","GLM53_DENSE_SMALL","GLM53_MOE_SCRATCH","GLM53_PREFILL_BATCH","GLM53_MOE_BATCH","GLM53_MOE_COOP","GLM53_COOP_SO","GLM53_MOE_COOP_PERSISTENT","GLM53_COOP_PERSISTENT_SO","GLM53_COOP_PERSISTENT_SHA","GLM53_COOP_PERSISTENT_CTAS"].map(|s|std::env::var(s).ok()).to_vec()
}
impl Session {
    pub fn new(engine:Engine,slots:usize,chunk_size:usize)->Self {
        assert!(chunk_size>0);
        Self{engine,prefixes:PrefixPool::new(slots),chunk_size,signature:signature()}
    }
    /// Returns last prompt logits and independent continuation state. Prefix can
    /// end on any chunk/token boundary; no token is replayed on an exact hit.
    pub fn prepare(&mut self,ids:&[i64])->(Tensor,DecodeStates,usize) {
        assert_eq!(self.signature,signature(),"session execution configuration changed");
        assert!(!ids.is_empty(),"empty prompt");
        if crate::mla_latent::enabled(){assert!(ids.len()<=crate::mla_latent::capacity() as usize,"prompt exceeds latent capacity");}
        let cached=self.prefixes.checkout(ids);
        let (mut consumed,mut last,mut state)=match cached {
            Some((n,l,s))=>(n,Some(l),Some(s)),None=>(0,None,None),
        };
        let hit=consumed;
        while consumed<ids.len() {
            let end=(consumed+self.chunk_size).min(ids.len());
            let input=Tensor::from_slice(&ids[consumed..end]).to_device(self.engine.w.device);
            let (logits,s)={let(l,s,_)=self.engine.prefill_record_last(&input,state.take(),false);(l,s)};
            consumed=end;last=Some(logits.get(logits.size()[0]-1));state=Some(s);
            if admit_chunk(ids.len()-consumed,self.chunk_size,self.prefixes.slots) {
                self.prefixes.insert(&ids[..consumed],last.as_ref().unwrap(),state.as_ref().unwrap());
            }
        }
        (last.unwrap(),state.unwrap(),hit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Kind,Device};
    fn state(value:f64)->DecodeStates {
        let mut k=crate::kda::KdaState::with_heads(Device::Cpu,1);let _=k.h.fill_(value);
        DecodeStates(vec![LayerState::Kda(k)])
    }
    #[test]
    fn doomed_chunk_admission_preserves_future_hits_and_states() {
        tch::set_num_threads(1);let logits=Tensor::zeros([3],(Kind::Float,Device::Cpu));
        for slots in 0..=4 {for chunk in [1usize,3,7] {
            let mut baseline=PrefixPool::new(slots);let mut pruned=PrefixPool::new(slots);
            let requests:Vec<Vec<i64>>=vec![(0..23).collect(),(0..23).collect(),(0..24).collect(),(50..62).collect(),(0..23).collect(),(50..70).collect(),(0..9).collect()];
            for ids in requests {
                let a=baseline.checkout(&ids).map(|x|x.0).unwrap_or(0);
                let b=pruned.checkout(&ids).map(|x|x.0).unwrap_or(0);assert_eq!(a,b);
                let mut end=a;
                while end<ids.len() {end=(end+chunk).min(ids.len());let s=state(end as f64);
                    baseline.insert(&ids[..end],&logits,&s);
                    if (ids.len()-end).div_ceil(chunk)<slots {pruned.insert(&ids[..end],&logits,&s);}
                }
                assert_eq!(baseline.entries.len(),pruned.entries.len());
                for e in &baseline.entries {
                    let other=pruned.entries.iter().find(|p|p.ids==e.ids).expect("same surviving keys");
                    assert!(e.logits.equal(&other.logits));assert_eq!(crate::forward::states_max_diff(&e.state,&other.state),0.);
                }
            }
        }}
    }
    #[test]
    fn longest_prefix_lru_and_independent_checkout() {
        let mut p=PrefixPool::new(2);let l=Tensor::zeros([3],(Kind::Float,Device::Cpu));
        let a=state(1.);p.insert(&[1,2],&l,&a);p.insert(&[1,2,3],&l,&state(2.));
        let (n,_,mut s)=p.checkout(&[1,2,3,4]).unwrap();assert_eq!(n,3);
        if let LayerState::Kda(k)=&mut s.0[0]{let _=k.h.fill_(99.);}
        assert_eq!(crate::forward::states_max_diff(&p.checkout(&[1,2,3]).unwrap().2,&state(2.)),0.);
        assert!(p.checkout(&[1]).is_none());assert!(p.checkout(&[2,1]).is_none());
        p.insert(&[7,8],&l,&state(3.));assert_eq!(p.evictions,1);
        assert!(p.checkout(&[1,2]).is_none());assert!(p.checkout(&[1,2,3]).is_some());
        let address=if let LayerState::Kda(k)=&p.entries[0].state.0[0]{k.h.data_ptr()}else{unreachable!()};
        p.insert(&[7,8],&l,&state(4.));
        let new_address=if let LayerState::Kda(k)=&p.entries[0].state.0[0]{k.h.data_ptr()}else{unreachable!()};
        assert_eq!(address,new_address);assert_eq!(p.len(),2);
    }
}

/// A synchronous prepare has no intervening checkout. If at least `slots`
/// newer, distinct chunks follow, this entry cannot survive the request.
/// Longest-prefix checkout ensures all remaining chunk keys are new.
pub(crate) fn admit_chunk(remaining:usize,chunk:usize,slots:usize)->bool {
    std::env::var("GLM53_PREFIX_ADMISSION").as_deref()!=Ok("1") || remaining.div_ceil(chunk)<slots
}
