//! Native Half shared GU only. Preserve original two GEMMs and every Float
//! activation operation; return exactly the Half input consumed by native down.
use tch::{Kind,Tensor};
pub(crate) fn enabled()->bool {
    match std::env::var("GLM53_SHARED_GU_FUSED").as_deref(){Ok("1")=>true,Ok("0")|Err(_)=>false,Ok(v)=>panic!("invalid GLM53_SHARED_GU_FUSED={v}")}
}
/// Metadata/flags only. No GPU work or speculative backend execution.
pub(crate) fn metadata_eligible(x:&Tensor,wg:&Tensor,wu:&Tensor,wd:&Tensor)->bool {
    if !enabled() || crate::root_probe::retain_f32() || crate::root_probe::full_f32() || crate::root_probe::recording(){return false;}
    if !x.device().is_cuda() || x.dim()!=2 || !(1..=32).contains(&x.size()[0]) || x.size()[1]!=4096 ||
        ![Kind::Half,Kind::Float].contains(&x.kind()){return false;}
    for w in [wg,wu,wd]{if w.device()!=x.device()||w.kind()!=Kind::Half||!(w.is_contiguous()||crate::c12::is_coded(w))||crate::dense_fp8::registered_enabled(w){return false;}}
    if wg.size()!=[1024,4096] || wu.size()!=wg.size() || wd.size()!=[4096,1024]{return false;}
    // Ask the very same dispatcher predicates with the original Float input.
    // A global GEMV/small flag does not mean these particular shapes use it.
    for w in [wg,wu] {
        if crate::dense_lt::eligible(x,w,false).is_some() || crate::gemv::small_eligible(x,w) || crate::gemv::eligible(x,w){return false;}
    }
    // Original down consumes a fresh contiguous Float [T,1024] product. Only
    // metadata is needed here: no fake GPU tensor/allocation/copy. Both down
    // consumers (row_mm16 / mm16_partial) are supported by this private seam.
    let rows=x.size()[0];let dev=x.device();
    if [false,true].into_iter().any(|partial|crate::dense_lt::eligible_meta(dev,rows,wd,partial).is_some()) ||
        crate::gemv::small_eligible_meta(dev,Kind::Float,rows,1024,true,wd) ||
        crate::gemv::eligible_meta(dev,Kind::Float,rows,1024,true,wd){return false;}
    true
}
fn write(g:&Tensor,u:&Tensor,out:&Tensor,stages:Option<&Tensor>) {
    assert_eq!(g.kind(),Kind::Half);assert_eq!(u.kind(),Kind::Half);assert_eq!(out.kind(),Kind::Half);
    assert_eq!(g.size(),u.size());assert_eq!(g.size(),out.size());assert!(g.is_contiguous()&&u.is_contiguous()&&out.is_contiguous());
    assert!(g.device().is_cuda());assert_eq!(g.device(),u.device());assert_eq!(g.device(),out.device());
    assert!((1..=4194304).contains(&g.numel()));
    if let Some(s)=stages{assert_eq!(s.kind(),Kind::Float);assert_eq!(s.numel(),4*g.numel());assert!(s.is_contiguous());assert_eq!(s.device(),g.device());}
    extern "C"{fn rs_shared_gu(g:*const std::ffi::c_void,u:*const std::ffi::c_void,out:*mut std::ffi::c_void,stages:*mut f32,count:i32)->i32;}
    let ptr=stages.map_or(std::ptr::null_mut(),|s|s.data_ptr().cast());
    assert_eq!(unsafe{rs_shared_gu(g.data_ptr(),u.data_ptr(),out.data_ptr(),ptr,g.numel() as i32)},0);
}
/// Fresh output ownership; graph callers keep input/weight owners as before.
pub(crate) fn try_activation(x:&Tensor,wg:&Tensor,wu:&Tensor,wd:&Tensor,half:Option<&Tensor>)->Option<Tensor> {
    if !metadata_eligible(x,wg,wu,wd){return None;}
    // Half D7: one C12 launch for the gate/up group (each half bitwise equal to its own launch), packed epilogue.
    if let Some(gu)=crate::c12::try_run_rows_out(x,&[wg,wu],2) {
        let (rows,n)=(gu.size()[0],wg.size()[0]);
        // GLM53_SHARED_GU_F32=1: FP32 copies of the same Half values, so an FP32-input consumer (the C12 down
        // projection) needs no Half->Float conversion launch; its in-kernel Half rounding restores the same operand.
        if std::env::var("GLM53_SHARED_GU_F32").as_deref()==Ok("1") {
            let out=Tensor::empty([rows,n],(Kind::Float,x.device()));
            extern "C"{fn rs_shared_gu_packed_f32(gu:*const std::ffi::c_void,out:*mut f32,rows:i32,n:i32)->i32;}
            assert_eq!(unsafe{rs_shared_gu_packed_f32(gu.data_ptr(),out.data_ptr().cast(),rows as i32,n as i32)},0);return Some(out);
        }
        let out=Tensor::empty([rows,n],(Kind::Half,x.device()));
        extern "C"{fn rs_shared_gu_packed(gu:*const std::ffi::c_void,out:*mut std::ffi::c_void,rows:i32,n:i32)->i32;}
        assert_eq!(unsafe{rs_shared_gu_packed(gu.data_ptr(),out.data_ptr(),rows as i32,n as i32)},0);return Some(out);
    }
    let g=crate::weights::shared_native_half(x,wg,half);
    let u=crate::weights::shared_native_half(x,wu,half);
    let out=Tensor::empty(g.size().as_slice(),(Kind::Half,x.device()));write(&g,&u,&out,None);Some(out)
}
#[path="shared_gu_probe.rs"]mod probe;
pub fn run(model:&std::path::Path,out:&std::path::Path){probe::run(model,out);}

#[cfg(test)]
mod dispatch_tests {
    use tch::{Device,Kind,Tensor};
    #[test]
    fn shared_native_dispatch_under_enabled_gemv_and_small() {
        struct Restore(Vec<(&'static str,Option<std::ffi::OsString>)>);
        impl Drop for Restore{fn drop(&mut self){for(k,v)in &self.0{match v{Some(v)=>std::env::set_var(k,v),None=>std::env::remove_var(k)}}}}
        let flags=["GLM53_DENSE_GEMV","GLM53_DENSE_SMALL"];
        let _restore=Restore(flags.into_iter().map(|k|(k,std::env::var_os(k))).collect());
        for key in flags{std::env::set_var(key,"1");}
        // CPU allocations contain no read data. These functions inspect only
        // metadata; Device::Cuda is a value, not a CUDA context/tensor creation.
        let gu=Tensor::empty([1024,4096],(Kind::Half,Device::Cpu));
        let down=Tensor::empty([4096,1024],(Kind::Half,Device::Cpu));
        let other=Tensor::empty([4096,4096],(Kind::Half,Device::Cpu));
        for rows in [1,2,8,32]{
            for (w,k) in [(&gu,4096),(&down,1024)]{
                assert!(!crate::gemv::eligible_meta(Device::Cuda(0),Kind::Float,rows,k,true,w));
                assert!(!crate::gemv::small_eligible_meta(Device::Cuda(0),Kind::Float,rows,k,true,w));
            }
        }
        assert!(crate::gemv::eligible_meta(Device::Cuda(0),Kind::Float,1,4096,true,&other));
        assert!(crate::gemv::small_eligible_meta(Device::Cuda(0),Kind::Float,2,4096,true,&other));
        for rows in [2,8,32]{assert!(!crate::gemv::eligible_meta(Device::Cuda(0),Kind::Float,rows,4096,true,&other));}
        for rows in [1,8,32]{assert!(!crate::gemv::small_eligible_meta(Device::Cuda(0),Kind::Float,rows,4096,true,&other));}
        assert!(!crate::gemv::eligible_meta(Device::Cuda(0),Kind::Half,1,4096,true,&other));
        assert!(!crate::gemv::eligible_meta(Device::Cuda(0),Kind::Float,1,4096,false,&other));
        assert!(!crate::gemv::eligible_meta(Device::Cpu,Kind::Float,1,4096,true,&other));
        for key in flags{std::env::set_var(key,"0");}
        assert!(!crate::gemv::eligible_meta(Device::Cuda(0),Kind::Float,1,4096,true,&other));
        assert!(!crate::gemv::small_eligible_meta(Device::Cuda(0),Kind::Float,2,4096,true,&other));
    }
}
