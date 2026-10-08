//! Optional Ranked DSA integer/mask bookkeeping. No score or topk rewrite.
use tch::{Tensor,Kind};

pub(crate) fn enabled()->bool {
    match std::env::var("GLM53_DSA_INDEX_FUSED") {
        Ok(v)=>match v.as_str(){"0"=>false,"1"=>true,_=>panic!("GLM53_DSA_INDEX_FUSED must be 0 or 1")},
        Err(std::env::VarError::NotPresent)=>false,Err(_)=>panic!("GLM53_DSA_INDEX_FUSED must be 0 or 1"),
    }
}
pub(crate) fn layout(scores:&Tensor,pos:&Tensor)->bool {
    scores.device().is_cuda()&&scores.kind()==Kind::Float&&scores.dim()==1&&scores.is_contiguous()&&
        scores.size()[0]>=1&&scores.size()[0]<=0x7fffff00&&pos.kind()==Kind::Int64&&pos.numel()==1&&
        pos.is_contiguous()&&pos.device()==scores.device()
}

// The caller owns disjoint output allocations. Narrowed contiguous guarded
// output views are accepted for the probe. No input tensor is mutated.
pub(crate) fn mask_into(scores:&Tensor,pos:&Tensor,out:&Tensor) {
    assert!(layout(scores,pos));assert_eq!(out.size(),scores.size());assert_eq!(out.kind(),Kind::Float);
    assert!(out.is_contiguous());assert_eq!(out.device(),scores.device());assert_ne!(out.data_ptr(),scores.data_ptr());
    extern "C" {fn rs_dsa_index_mask(scores:*const f32,pos:*const i64,out:*mut f32,pools:i32)->i32;}
    assert_eq!(unsafe{rs_dsa_index_mask(scores.data_ptr().cast(),pos.data_ptr().cast(),out.data_ptr().cast(),scores.size()[0] as i32)},0);
}
pub(crate) fn expand_into(selected:&Tensor,pos:&Tensor,out:&Tensor) {
    assert_eq!(selected.kind(),Kind::Int64);assert_eq!(selected.dim(),1);assert!(selected.is_contiguous());
    let k=selected.size()[0];assert!((1..=512).contains(&k));
    assert!(selected.device().is_cuda());assert_eq!(pos.kind(),Kind::Int64);assert_eq!(pos.numel(),1);assert!(pos.is_contiguous());
    assert_eq!(pos.device(),selected.device());assert_eq!(out.kind(),Kind::Int64);assert_eq!(out.size(),[4*k+3]);
    assert!(out.is_contiguous());assert_eq!(out.device(),selected.device());assert_ne!(out.data_ptr(),selected.data_ptr());
    extern "C" {fn rs_dsa_index_expand(selected:*const i64,pos:*const i64,out:*mut i64,k:i32)->i32;}
    assert_eq!(unsafe{rs_dsa_index_expand(selected.data_ptr().cast(),pos.data_ptr().cast(),out.data_ptr().cast(),k as i32)},0);
}

/// Shape/device metadata only: never synchronously read pos. append_projected
/// already has a checked sequence extent; valid pos is nonnegative and fits
/// its token cache. A fresh masked buffer preserves the scores input exactly.
pub(crate) fn try_ranked(scores:&Tensor,pos:&Tensor)->Option<Tensor> {
    if !enabled()||!layout(scores,pos) {return None;}
    let masked=Tensor::empty_like(scores);mask_into(scores,pos,&masked);
    // Same dimension, k, largest=true and sorted=true as the original path.
    let selected=masked.topk(512.min(scores.size()[0]),0,true,true).1;
    let out=Tensor::empty([selected.size()[0]*4+3],(Kind::Int64,scores.device()));
    expand_into(&selected,pos,&out);Some(out)
}

#[path="dsa_index_probe.rs"] mod local_probe;
pub fn probe(out:&std::path::Path) {local_probe::run(out);}
