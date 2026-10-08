//! Bounded default-off Conv::convolve only; projection/norm/residual unchanged.
use tch::{Kind,Tensor};
pub(crate) fn enabled()->bool {
    match std::env::var("GLM53_DRAFT_CONV_FUSED") {
        Ok(v)=>match v.as_str(){"0"=>false,"1"=>true,_=>panic!("GLM53_DRAFT_CONV_FUSED must be 0 or 1")},
        Err(std::env::VarError::NotPresent)=>false,Err(_)=>panic!("GLM53_DRAFT_CONV_FUSED must be 0 or 1"),
    }
}
fn eligible(x:&Tensor,delta:&Tensor,base:&Tensor,side:i64)->bool {
    (0..=1).contains(&side)&&x.device().is_cuda()&&x.kind()==Kind::BFloat16&&x.dim()==2&&
        (1..=8).contains(&x.size()[0])&&x.size()[1]==4096&&x.is_contiguous()&&
        base.device()==x.device()&&base.kind()==Kind::BFloat16&&base.size()==[2,2,4096]&&base.is_contiguous()&&
        delta.device()==x.device()&&delta.kind()==Kind::BFloat16&&delta.size()==[x.size()[0],2,256]&&
        delta.stride().iter().all(|&s|s>0)
}
// Private only: production allocates a fresh output; tests pass disjoint guards.
fn into(x:&Tensor,delta:&Tensor,base:&Tensor,side:i64,out:&Tensor) {
    assert!(eligible(x,delta,base,side));assert_eq!(out.size(),x.size());assert_eq!(out.kind(),Kind::BFloat16);
    assert_eq!(out.device(),x.device());assert!(out.is_contiguous());
    for input in [x,delta,base]{assert_ne!(out.data_ptr(),input.data_ptr());}
    let ds=delta.stride();
    extern "C" {fn rs_draft_conv(x:*const std::ffi::c_void,delta:*const std::ffi::c_void,base:*const std::ffi::c_void,
        out:*mut std::ffi::c_void,n:i32,side:i32,ds0:i64,ds1:i64,ds2:i64)->i32;}
    assert_eq!(unsafe{rs_draft_conv(x.data_ptr(),delta.data_ptr(),base.data_ptr(),out.data_ptr(),
        x.size()[0] as i32,side as i32,ds[0],ds[1],ds[2])},0);
}
fn try_convolve_enabled(on:bool,x:&Tensor,delta:&Tensor,base:&Tensor,side:i64)->Option<Tensor> {
    if !on||!eligible(x,delta,base,side){return None;}
    let out=Tensor::empty(x.size().as_slice(),(Kind::BFloat16,x.device()));into(x,delta,base,side,&out);Some(out)
}
pub(crate) fn try_convolve(x:&Tensor,delta:&Tensor,base:&Tensor,side:i64)->Option<Tensor> {
    try_convolve_enabled(enabled(),x,delta,base,side)
}
/// One proposal owns these host-only counts. Cache its immutable flag once;
/// increment fused only after the FFI-backed Some path actually returns.
/// No globals, atomics, device readback or per-call environment lookup.
pub(crate) struct Dispatch {pub calls:usize,pub fused:usize,on:bool}
impl Dispatch {
    pub(crate) fn new()->Self {Self{calls:0,fused:0,on:enabled()}}
    pub(crate) fn try_convolve(&mut self,x:&Tensor,delta:&Tensor,base:&Tensor,side:i64)->Option<Tensor> {
        self.calls+=1;
        let out=try_convolve_enabled(self.on,x,delta,base,side);
        self.fused+=usize::from(out.is_some());out
    }
}
/// Resident qualification changes only Conv and the explicitly enumerated final
/// norm setting, restoring both before returning to the caller's session.
pub fn check(drafter:&crate::dflash::Drafter,target:&crate::weights::ModelWeights,out:&std::path::Path) {
    crate::dflash::conv_fused_check(drafter,target,out);
}
// Exact current dflash expression. This diagnostic oracle does not replace the
// original fallback, and materializes the BF16 intermediate tensors as before.
fn old(x:&Tensor,delta:&Tensor,base:&Tensor,side:i64)->Tensor {
    let n=x.size()[0];let blocks=x.view([n,256,16]);
    let coefficients=base.get(side).view([1,2,256,16])+delta.unsqueeze(-1);
    let shifted=Tensor::cat(&[Tensor::zeros([1,256,16],(x.kind(),x.device())),blocks.narrow(0,0,n-1)],0);
    (coefficients.select(1,0)*blocks+coefficients.select(1,1)*shifted).view([n,4096])
}
#[path="draft_conv_probe.rs"] mod local_probe;
pub fn probe(draft:&std::path::Path,out:&std::path::Path){local_probe::run(draft,out);}
