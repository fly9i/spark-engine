//! Default-off pre-append position capture in the existing mask producer.
//! No floating arithmetic or ranking changes; old kernel/ABI remain available.
use tch::{Tensor,Kind};
pub(crate) fn enabled()->bool {
    match std::env::var("GLM53_DSA_POSITION_CAPTURE") {
        Ok(v)=>match v.as_str(){"0"=>false,"1"=>true,_=>panic!("GLM53_DSA_POSITION_CAPTURE must be 0 or 1")},
        Err(std::env::VarError::NotPresent)=>false,
        Err(_)=>panic!("GLM53_DSA_POSITION_CAPTURE must be 0 or 1"),
    }
}
extern "C" {fn rs_dsa_index_mask_capture(scores:*const f32,pos:*const i64,out:*mut f32,captured:*mut i64,pools:i32)->i32;}
/// `captured` must be an independent caller-owned single-row sidecar. The
/// mutable input length may advance immediately AFTER this enqueue on the
/// current stream. No host length read or deferred input alias is retained.
pub(crate) fn mask_into(scores:&Tensor,pos:&Tensor,out:&Tensor,captured:&Tensor) {
    assert!(crate::dsa_index::layout(scores,pos));
    assert_eq!(out.size(),scores.size());assert_eq!(out.kind(),Kind::Float);
    assert!(out.is_contiguous()&&out.device()==scores.device());
    assert!(captured.kind()==Kind::Int64&&captured.numel()==1&&captured.is_contiguous()&&captured.device()==scores.device());
    assert_ne!(out.data_ptr(),scores.data_ptr());assert_ne!(captured.data_ptr(),pos.data_ptr());
    assert_ne!(captured.data_ptr(),scores.data_ptr());assert_ne!(captured.data_ptr(),out.data_ptr());
    let status=unsafe{rs_dsa_index_mask_capture(scores.data_ptr().cast(),pos.data_ptr().cast(),
        out.data_ptr().cast(),captured.data_ptr().cast(),scores.size()[0] as i32)};
    assert_eq!(status,0,"DSA mask/position capture launch failed");
}
