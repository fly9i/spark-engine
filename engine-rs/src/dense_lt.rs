//! Shape-qualified cuBLASLt alternative, FP16 operands and FP32 accumulation.
use tch::{Tensor,Kind};
use std::collections::HashMap;
fn table()-> &'static HashMap<String,i32> {
    static TABLE:std::sync::OnceLock<HashMap<String,i32>>=std::sync::OnceLock::new();
    TABLE.get_or_init(||std::env::var("GLM53_DENSE_LT_TABLE").ok().map(|p|serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()).unwrap_or_default())
}
/// Same shape table lookup without allocating a synthetic activation tensor.
pub(crate) fn eligible_meta(device:tch::Device,rows:i64,w:&Tensor,partial:bool)->Option<i32> {
    if std::env::var("GLM53_DENSE_LT").as_deref()!=Ok("1") || !device.is_cuda() || w.kind()!=Kind::Half || !w.is_contiguous() || w.dim()!=2{return None;}
    let key=format!("{},{},{},{}",rows,w.size()[0],w.size()[1],i32::from(partial));
    table().get(&key).copied()
}
pub fn eligible(x:&Tensor,w:&Tensor,partial:bool)->Option<i32> {
    if x.dim()!=2{return None;}eligible_meta(x.device(),x.size()[0],w,partial)
}
pub fn run(x:&Tensor,w:&Tensor,partial:bool,algorithm:i32)->Tensor {
    let x=x.to_kind(Kind::Half).contiguous();let y=Tensor::empty([x.size()[0],w.size()[0]],(if partial{Kind::Float}else{Kind::Half},x.device()));
    extern "C"{fn rs_lt_mm16(x:*const std::ffi::c_void,w:*const std::ffi::c_void,y:*mut std::ffi::c_void,m:i32,n:i32,k:i32,fp32:i32,algorithm:i32)->i32;}
    assert_eq!(unsafe{rs_lt_mm16(x.data_ptr(),w.data_ptr(),y.data_ptr(),x.size()[0] as i32,w.size()[0] as i32,w.size()[1] as i32,i32::from(partial),algorithm)},0,"Lt launch");y.to_kind(Kind::Float)
}
