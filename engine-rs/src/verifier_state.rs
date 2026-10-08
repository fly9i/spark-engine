//! Temporary verifier output. Only materialize() returns a request state.
//! Deferred records never enter prefill, step, prefix caches, or public snapshots.
use crate::forward::{DecodeStates,LayerState,snapshot,snapshot_layer};

pub enum Layer {
    Kda(crate::kda_correction::ChainRecord),
    Full(Vec<LayerState>),
}

pub enum VerifierStates {
    Full(Vec<DecodeStates>),
    Deferred { tokens:usize, layers:Vec<Layer> },
}

impl VerifierStates {
    pub fn len(&self)->usize {
        match self {Self::Full(s)=>s.len(),Self::Deferred{tokens,..}=>*tokens}
    }
    pub fn into_full(self)->Vec<DecodeStates> {
        match self {Self::Full(s)=>s,Self::Deferred{..}=>panic!("deferred verifier state crossed a full-state API")}
    }
    /// `node` includes the anchor. No selected node means unchanged base;
    /// zero accepted drafts normally means Some(0), not None.
    pub fn materialize(&self,base:&DecodeStates,node:Option<usize>)->DecodeStates {
        let Some(node)=node else {return snapshot(base);};
        assert!(node<self.len());
        match self {
            Self::Full(states)=>snapshot(&states[node]),
            Self::Deferred{layers,..}=>{
                assert_eq!(layers.len(),base.0.len());
                let records:Vec<_>=layers.iter().filter_map(|s|match s {Layer::Kda(r)=>Some(r),_=>None}).collect();
                let committed=if records.is_empty(){Vec::new()}else{crate::kda_correction::materialize(&records,node)};
                assert_eq!(committed.len(),records.len());
                let mut committed=committed.into_iter();
                let state=DecodeStates(layers.iter().map(|s|match s {
                    Layer::Kda(_)=>LayerState::Kda(committed.next().unwrap()),
                    Layer::Full(states)=>{assert_eq!(states.len(),self.len());snapshot_layer(&states[node])},
                }).collect());
                assert!(committed.next().is_none());state
            }
        }
    }
    /// Diagnostic only: materialize one node at a time, never hold all full H.
    pub fn assert_exact(&self,other:&Self,base:&DecodeStates,context:&str) {
        assert_eq!(self.len(),other.len());
        for node in 0..self.len() {
            let a=self.materialize(base,Some(node));let b=other.materialize(base,Some(node));
            assert_eq!(crate::forward::states_max_diff(&a,&b),0.,"{context} node={node}");
            for (a,b) in a.0.iter().zip(&b.0) {
                if let (LayerState::Kda(a),LayerState::Kda(b))=(a,b) {
                    for (a,b) in [(&a.h,&b.h),(&a.conv,&b.conv)] {
                        assert!(a.contiguous().view_dtype(tch::Kind::Int).equal(&b.contiguous().view_dtype(tch::Kind::Int)),
                            "{context} KDA bit pattern node={node}");
                    }
                }
            }
        }
    }
}
