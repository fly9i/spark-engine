//! Default-off path selection from the original drafter's edges and IDs.
use tch::{Kind,Tensor};
pub(crate) fn enabled()->bool {
    match std::env::var("GLM53_DRAFT_SELECTOR_FUSED") {
        Ok(v)=>match v.as_str(){"0"=>false,"1"=>true,_=>panic!("GLM53_DRAFT_SELECTOR_FUSED must be 0 or 1")},
        Err(std::env::VarError::NotPresent)=>false,Err(_)=>panic!("GLM53_DRAFT_SELECTOR_FUSED must be 0 or 1"),
    }
}
fn layout(edges:&Tensor,ids:&Tensor)->bool {
    edges.device().is_cuda()&&edges.kind()==Kind::Float&&edges.dim()==3&&
        (1..=7).contains(&edges.size()[0])&&edges.size()[1..]==[16,16]&&edges.is_contiguous()&&
        ids.device()==edges.device()&&ids.kind()==Kind::Int64&&ids.size()==[edges.size()[0],16]&&ids.is_contiguous()
}
// Private callers provide disjoint output allocation; narrowed contiguous
// guard views are accepted by the local probe. No host read of input values.
fn into(edges:&Tensor,ids:&Tensor,out:&Tensor) {
    assert!(layout(edges,ids));assert_eq!(out.size(),[edges.size()[0]]);assert_eq!(out.kind(),Kind::Int64);
    assert_eq!(out.device(),edges.device());assert!(out.is_contiguous());assert_ne!(out.data_ptr(),ids.data_ptr());
    extern "C" {fn rs_draft_selector(edges:*const f32,ids:*const i64,out:*mut i64,steps:i32)->i32;}
    assert_eq!(unsafe{rs_draft_selector(edges.data_ptr().cast(),ids.data_ptr().cast(),out.data_ptr().cast(),edges.size()[0] as i32)},0);
}
pub(crate) fn try_path(edges:&Tensor,ids:&Tensor)->Option<Tensor> {
    if !enabled()||!layout(edges,ids){return None;}
    let out=Tensor::empty([edges.size()[0]],(Kind::Int64,edges.device()));into(edges,ids,&out);Some(out)
}
// Exact original sequence, used only as a diagnostic oracle. Production
// fallback remains in dflash.rs rather than silently changing its contract.
fn old_path(edges:&Tensor,ids:&Tensor)->Tensor {
    let mut previous=Tensor::zeros([1],(Kind::Int64,edges.device()));let mut tokens=Vec::new();
    for t in 0..edges.size()[0] {
        let best=edges.get(t).index_select(0,&previous).argmax(-1,false);
        tokens.push(ids.get(t).index_select(0,&best));previous=best;
    }
    Tensor::cat(&tokens,0)
}
#[path="draft_selector_probe.rs"] mod local_probe;
pub fn probe(out:&std::path::Path){local_probe::run(out);}
/// Optional borrowed resident gate, after dflash.rs wiring is applied.
pub(crate) fn check(drafter:&crate::dflash::Drafter,target:&crate::weights::ModelWeights,out:&std::path::Path) {
    local_probe::check(drafter,target,out);
}
