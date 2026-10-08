//! Only the final drafter residual+norm; default off. The caller supplies the
//! unchanged original Norm implementation, including its precision boundaries.
use tch::{Kind,Tensor};
use std::path::Path;

pub fn enabled()->bool {
    match std::env::var("GLM53_DRAFT_FINAL_NORM_SELECT").as_deref() {
        Ok("1")=>true,Ok("0")|Err(_)=>false,
        Ok(value)=>panic!("GLM53_DRAFT_FINAL_NORM_SELECT must be 0 or 1, got {value}"),
    }
}

/// Preserve full-N conversion/add/norm shapes and strides: selecting rows before
/// ATen reduction failed the first L graph raw-bit gate. Delete only the unused
/// BF16 residual cast, then perform the original final narrow view.
pub(crate) fn try_hidden<F>(x:&Tensor,residual:&Tensor,normalize:F)->Option<Tensor>
where F:FnOnce(&Tensor)->Tensor {
    if !enabled(){return None;}
    let shape=x.size();
    if shape.len()!=2 || !(2..=8).contains(&shape[0]) || shape[1]!=4096 ||
        residual.size()!=shape || residual.device()!=x.device() ||
        ![Kind::BFloat16,Kind::Float].contains(&x.kind()) || residual.kind()!=x.kind() {
        return None;
    }
    let sum=x.to_kind(Kind::Float)+residual.to_kind(Kind::Float);
    Some(normalize(&sum).narrow(0,1,shape[0]-1))
}

#[path="draft_final_norm_probe.rs"]
mod probe;
pub(crate) use probe::norm_check;

/// Uses the resident real final Norm and complete original proposal producers.
/// Other flags (including Conv) are preserved; caller may run again in a separately
/// qualified combination. This does not turn an env value into proof of Conv dispatch.
pub fn check(drafter:&crate::dflash::Drafter,target:&crate::weights::ModelWeights,out:&Path) {
    drafter.final_norm_select_check(&out.join("norm"));
    probe::proposal_check(drafter,target,&out.join("proposal"));
}
