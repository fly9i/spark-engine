//! Experimental weight-only FP8: extra quantization, not an EXL3 format change.
//! Cache owns the source tensors so allocator pointer reuse cannot alias weights.
//! One resident model per process; cached weights have process lifetime.
use std::{cell::RefCell,collections::HashMap};
use tch::{Tensor,Kind};
/// q4: GLM53_DRAFT_Q4 copy of a DFlash2 weight (shim/draft_q4.cu), encoded from the BF16 source at registration.
struct Entry {source:Tensor,flag:&'static str,quant:Option<(Tensor,Tensor)>,q4:Option<(Tensor,Tensor)>}
thread_local!{static CACHE:RefCell<HashMap<usize,Entry>>=RefCell::new(HashMap::new());}
// Row-concatenated FP8 weights for producers sharing one input (KDA q/k/v). Member cache
// entries become row views of the same storage, so residency does not grow.
thread_local!{static FUSED:RefCell<HashMap<Vec<usize>,(Tensor,Tensor)>>=RefCell::new(HashMap::new());}
/// Read-only dispatch proof; no lazy quantization or GPU work.
pub(crate) fn registered_enabled(w:&Tensor)->bool {
    CACHE.with(|c|c.borrow().get(&(w.data_ptr() as usize)).is_some_and(|e|
        e.source.size()==w.size() && e.source.stride()==w.stride() && e.source.kind()==w.kind() &&
        std::env::var(e.flag).as_deref()==Ok("1")))
}
/// GLM53_DRAFT_Q4=1 (L2, draft side only): DFlash2 body weights (every GLM53_DRAFT_FP8_* class but the head, whose
/// INT4 copy and FP8 rescore stay) are also held as affine 4-bit, read by decode-sized calls (1..=32 rows).
fn draft_q4_enabled()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_DRAFT_Q4").as_deref()==Ok("1"))}
fn draft_q4_of(w:&Tensor,flag:&str)->Option<(Tensor,Tensor)> {
    extern "C"{fn glm53_draft_q4_ok(n:i32,k:i32)->i32;fn rs_draft_q4_encode(w:*const std::ffi::c_void,n:i32,k:i32,mse:i32,sm:*mut std::ffi::c_void,q:*mut std::ffi::c_void)->i32;}
    if !draft_q4_enabled() || w.kind()!=Kind::BFloat16 || !flag.starts_with("GLM53_DRAFT_") || flag=="GLM53_DRAFT_FP8_HEAD" || std::env::var(flag).as_deref()!=Ok("1") {return None;}
    let (n,k)=(w.size()[0],w.size()[1]);
    if unsafe{glm53_draft_q4_ok(n as i32,k as i32)}==0 {return None;}
    let sm=Tensor::empty([n,k/128,2],(Kind::Float,w.device()));let q=Tensor::empty([n,k/2],(Kind::Uint8,w.device()));
    let mse=i32::from(std::env::var("GLM53_DRAFT_Q4_MSE").as_deref()!=Ok("0"));
    assert_eq!(unsafe{rs_draft_q4_encode(w.data_ptr(),n as i32,k as i32,mse,sm.data_ptr(),q.data_ptr())},0,"draft Q4 encode");
    Some((q,sm))
}
fn run_q4(x:&Tensor,q:&Tensor,sm:&Tensor,partial:bool)->Tensor {
    // FP32 inputs are rounded to BF16 in-kernel and BF16 outputs written directly (the same RN conversions as
    // x.to_kind(BF16) / y.to_kind(BF16)): no separate cast launches.
    let xf=x.kind()==Kind::Float;
    let input=if xf||x.kind()==Kind::BFloat16 {x.contiguous()} else {x.to_kind(Kind::BFloat16).contiguous()};
    let xf=xf&&input.kind()==Kind::Float;let (m,n,k)=(x.size()[0],q.size()[0],q.size()[1]*2);
    let y=Tensor::empty([m,n],(if partial{Kind::Float}else{Kind::BFloat16},x.device()));
    extern "C"{fn rs_draft_q4_gemm2(x:*const std::ffi::c_void,xf:i32,q:*const std::ffi::c_void,sm:*const std::ffi::c_void,y:*mut std::ffi::c_void,yb:i32,m:i32,n:i32,k:i32)->i32;}
    assert_eq!(unsafe{rs_draft_q4_gemm2(input.data_ptr(),i32::from(xf),q.data_ptr(),sm.data_ptr(),y.data_ptr(),i32::from(!partial),m as i32,n as i32,k as i32)},0,"draft Q4 GEMM");
    y
}
pub fn register_weight(w:&Tensor,flag:&'static str) {
    if ![Kind::Half,Kind::BFloat16].contains(&w.kind()) || w.dim()!=2 || !w.is_contiguous() || w.size()[0]<512 || w.size()[1]<512{return;}
    CACHE.with(|c|{c.borrow_mut().entry(w.data_ptr() as usize).or_insert_with(||Entry{source:w.shallow_clone(),flag,quant:None,q4:draft_q4_of(w,flag)});});
}
pub fn register(layers:&[crate::weights::LayerWeights]) {
    for layer in layers {
        // GLM53_DENSE_MLP_FP8=0 (diagnostic): keep the dense MLP layers (0-2) in Half while KDA follows GLM53_DENSE_FP8.
        if std::env::var("GLM53_DENSE_MLP_FP8").as_deref()!=Ok("0") {
            if let Some(w)=&layer.dense{for t in [&w.wg,&w.wu,&w.wd]{register_weight(t,"GLM53_DENSE_FP8");}}}
        // Item 4: EXL3 KDA layers never use FP8 copies of q/k/v/o.
        // GLM53_KDA_FP8=0: KDA q/k/v/o stay at source precision (Half of the BF16 checkpoint) while the dense MLP layers
        // follow GLM53_DENSE_FP8 (report 10.29: KDA FP8 is nearly all of r11's drift from source precision).
        if let Some(w)=&layer.kda{if std::env::var("GLM53_KDA_FP8").as_deref()!=Ok("0"){for t in [&w.wq,&w.wk,&w.wv,&w.wo]{register_weight(t,"GLM53_DENSE_FP8");}}}
        if let Some(w)=&layer.moe{for t in [&w.sh_wg,&w.sh_wu,&w.sh_wd]{register_weight(t,"GLM53_FP8_SHARED");}}
        if let Some(w)=&layer.mla{for t in [&w.q_a,&w.q_b,&w.kv_a,&w.wo]{register_weight(t,"GLM53_FP8_MLA");}}
    }
}
pub fn quantize(w:&Tensor)->(Tensor,Tensor) {
    let w=w.to_kind(Kind::Float);let scale=w.abs().amax([1],true).clamp_min(1e-12)/448.;
    let quant=(&w/&scale).clamp(-448.,448.).to_kind(Kind::Float8e4m3fn);(quant,scale.view([-1]))
}
fn placeholder(w:&Tensor)->bool {w.dim()==2 && w.stride().iter().all(|&s|s==0)}
pub(crate) fn free_sources_enabled()->bool {std::env::var("GLM53_FP8_FREE_SOURCE").as_deref()==Ok("1")}
/// Memory reclaim (L0): quantize a registered weight now and return a stride-0 shape
/// placeholder that becomes the cache key; dropping the caller's original frees its storage.
/// Any non-FP8 fallback on a placeholder panics (see weights::assert_materialized).
pub(crate) fn compact(w:&Tensor)->Option<Tensor> {
    if !free_sources_enabled() {return None;}
    CACHE.with(|c|{let mut c=c.borrow_mut();
        let e=c.get(&(w.data_ptr() as usize))?;
        if e.source.size()!=w.size() || std::env::var(e.flag).as_deref()!=Ok("1") {return None;}
        let flag=e.flag;let quant=e.quant.as_ref().map(|(a,b)|(a.shallow_clone(),b.shallow_clone())).unwrap_or_else(||quantize(w));
        let q4=c.remove(&(w.data_ptr() as usize)).and_then(|e|e.q4);
        let ph=Tensor::zeros([1],(w.kind(),w.device())).expand(w.size().as_slice(),false);
        c.insert(ph.data_ptr() as usize,Entry{source:ph.shallow_clone(),flag,quant:Some(quant),q4});
        Some(ph)})
}
pub fn try_run(x:&Tensor,w:&Tensor,partial:bool)->Option<Tensor> {
    if !x.device().is_cuda()||x.dim()!=2||w.kind()!=Kind::Half||!(w.is_contiguous()||placeholder(w)){return None;}
    // P9 (GLM53_FP8_PREFILL_HALF=1, needs FP8_FREE_SOURCE=0): prefill-sized inputs use the retained
    // Half weights directly instead of expanding the FP8 copy on every call; decode keeps FP8.
    if x.size()[0]>128 && !placeholder(w) && std::env::var("GLM53_FP8_PREFILL_HALF").as_deref()==Ok("1") {return None;}
    cached(x,w,partial)
}
/// P1b (GLM53_FP8_BIG=1): prefill-sized rows (> GLM53_FP8_BIG_MIN, default 128) read the resident FP8
/// weight directly in a tensor-core GEMM with the per-row scale and Half rounding fused
/// (shim/fp8_big.cu). No Half weight is kept or expanded, so GLM53_FP8_FREE_SOURCE=1 can drop the
/// retained sources. Takes precedence over GLM53_FP8_PREFILL_HALF. L1 against the expand path.
pub(crate) fn big_enabled()->bool {std::env::var("GLM53_FP8_BIG").as_deref()==Ok("1")}
fn big_min_rows()->i64 {std::env::var("GLM53_FP8_BIG_MIN").ok().and_then(|v|v.parse().ok()).unwrap_or(128)}
fn big_stages()->i32 {std::env::var("GLM53_FP8_BIG_STAGES").ok().and_then(|v|v.parse().ok()).unwrap_or(3)}
/// (quant, scale) of a registered, enabled Half weight; quantizes lazily outside graph capture.
pub(crate) fn quant_of(w:&Tensor)->Option<(Tensor,Tensor)> {
    if w.kind()!=Kind::Half || w.dim()!=2 || !(w.is_contiguous()||placeholder(w)) {return None;}
    CACHE.with(|c|{let mut c=c.borrow_mut();let e=c.get_mut(&(w.data_ptr() as usize))?;
        if e.source.size()!=w.size() || e.source.stride()!=w.stride() || e.source.kind()!=w.kind() || std::env::var(e.flag).as_deref()!=Ok("1"){return None;}
        let (q,s)=e.quant.get_or_insert_with(||{assert!(!crate::tp::graph::capturing(),"FP8 quant cache miss during graph capture");quantize(w)});
        Some((q.shallow_clone(),s.shallow_clone()))})
}
/// The FP8 (q, s) of a registered BF16 weight (drafter), quantizing on first use outside capture.
pub(crate) fn quant_of_bf16(w:&Tensor)->Option<(Tensor,Tensor)> {
    if w.kind()!=Kind::BFloat16 || w.dim()!=2 || !(w.is_contiguous()||placeholder(w)) {return None;}
    CACHE.with(|c|{let mut c=c.borrow_mut();let e=c.get_mut(&(w.data_ptr() as usize))?;
        if e.source.size()!=w.size() || e.source.stride()!=w.stride() || e.source.kind()!=w.kind() || std::env::var(e.flag).as_deref()!=Ok("1"){return None;}
        let (q,s)=e.quant.get_or_insert_with(||{assert!(!crate::tp::graph::capturing(),"FP8 quant cache miss during graph capture");quantize(w)});
        Some((q.shallow_clone(),s.shallow_clone()))})
}
/// try_big would run for (x, w) (GLM53_PREFILL_HALF_GLUE: callers that skip it must know).
pub(crate) fn big_eligible(x:&Tensor,w:&Tensor)->bool {big_rows(x) && quant_of(w).is_some_and(|(q,_)|big_shape_ok(&q,x.size()[1]))}
pub(crate) fn big_rows(x:&Tensor)->bool {big_enabled() && x.device().is_cuda() && x.dim()==2 && x.size()[0]>big_min_rows()}
fn big_shape_ok(q:&Tensor,k:i64)->bool {q.size()[0]%128==0 && q.size()[1]%32==0 && q.size()[1]==k}
/// out[:, :] (row stride may exceed its width: column slice of a wider buffer) = scale * (xh @ fp8(q)^T).
pub(crate) fn big_into(xh:&Tensor,q:&Tensor,s:&Tensor,out:&Tensor,rounded:bool) {
    assert_eq!(xh.kind(),Kind::Half);assert!(xh.is_contiguous());assert_eq!(q.kind(),Kind::Float8e4m3fn);assert!(q.is_contiguous());
    assert_eq!(s.kind(),Kind::Float);assert!(s.is_contiguous());assert_eq!(out.kind(),Kind::Float);assert_eq!(out.stride()[1],1);
    let (m,k,n)=(xh.size()[0],xh.size()[1],q.size()[0]);assert_eq!(q.size()[1],k);assert_eq!(out.size(),[m,n]);assert_eq!(s.numel() as i64,n);
    extern "C"{fn rs_fp8_big(x:*const std::ffi::c_void,w:*const std::ffi::c_void,s:*const f32,y:*mut f32,m:i32,n:i32,k:i32,ldy:i32,rounded:i32,stages:i32)->i32;}
    assert_eq!(unsafe{rs_fp8_big(xh.data_ptr(),q.data_ptr(),s.data_ptr().cast(),out.data_ptr().cast(),m as i32,n as i32,k as i32,out.stride()[0] as i32,i32::from(rounded),big_stages())},0,"FP8 big GEMM");
}
pub(crate) fn try_big(x:&Tensor,w:&Tensor,partial:bool,half:Option<&Tensor>)->Option<Tensor> {
    if !big_rows(x) {return None;}
    let (q,s)=quant_of(w)?;
    if !big_shape_ok(&q,x.size()[1]) {return None;}
    let xh=half.map(Tensor::shallow_clone).unwrap_or_else(||crate::weights::half_input_pub(x)).contiguous();
    let y=Tensor::empty([x.size()[0],q.size()[0]],(Kind::Float,x.device()));
    big_into(&xh,&q,&s,&y,!partial);Some(y)
}
/// Row-concatenated producers sharing x (KDA q/k/v): each writes its own column slice.
pub(crate) fn try_big_cat(x:&Tensor,ws:&[&Tensor])->Option<Tensor> {
    if !big_rows(x) {return None;}
    let parts:Option<Vec<(Tensor,Tensor)>>=ws.iter().map(|w|quant_of(w).filter(|(q,_)|big_shape_ok(q,x.size()[1]))).collect();
    let parts=parts?;
    let total:i64=parts.iter().map(|(q,_)|q.size()[0]).sum();
    let out=Tensor::empty([x.size()[0],total],(Kind::Float,x.device()));
    let xh=crate::weights::half_input_pub(x).contiguous();let mut off=0;
    for (q,s) in &parts {let n=q.size()[0];big_into(&xh,q,s,&out.narrow(1,off,n),true);off+=n;}
    Some(out)
}
pub fn try_bf16(x:&Tensor,w:&Tensor,partial:bool)->Option<Tensor> {
    if !x.device().is_cuda()||x.dim()!=2||w.kind()!=Kind::BFloat16||!(w.is_contiguous()||placeholder(w)){return None;}
    cached(x,w,partial)
}
fn cached(x:&Tensor,w:&Tensor,partial:bool)->Option<Tensor> {
    CACHE.with(|c|{let mut c=c.borrow_mut();let e=c.get_mut(&(w.data_ptr() as usize))?;
        // A rank-zero narrow view can share a pointer with a larger registered
        // tensor. Pointer identity alone does not identify its shape/layout.
        if e.source.size()!=w.size() || e.source.stride()!=w.stride() || e.source.kind()!=w.kind() || std::env::var(e.flag).as_deref()!=Ok("1"){return None;}
        if w.kind()==Kind::BFloat16 && (1..=32).contains(&x.size()[0]) {if let Some((q,sm))=&e.q4 {return Some(run_q4(x,q,sm,partial));}}
        let (q,s)=e.quant.get_or_insert_with(||{assert!(!crate::tp::graph::capturing(),"FP8 quant cache miss during graph capture");quantize(w)});
        Some(if w.kind()==Kind::BFloat16 {run_bf16(x,q,s,partial)}else{run(x,q,s,partial)})})
}
/// BF16 activations and BF16 output boundaries, FP32 partials before TP SUM.
/// No FP16 activation cast: BF16's exponent range must remain supported.
pub fn run_bf16(x:&Tensor,q:&Tensor,s:&Tensor,partial:bool)->Tensor {
    let input=x.to_kind(Kind::BFloat16).contiguous();
    let y=Tensor::empty([x.size()[0],q.size()[0]],(Kind::Float,x.device()));
    let max_rows=if std::env::var("GLM53_DRAFT_SKINNY32").as_deref()==Ok("1"){32}else{16};
    if skinny_enabled() && (2..=max_rows).contains(&x.size()[0]) && q.size()[0]%16==0 && q.size()[1]%64==0 {
        // W04 kernel reads the BF16 activation directly; no Float staging copy.
        extern "C"{fn rs_fp8_bf16_skinny(x:*const std::ffi::c_void,w:*const std::ffi::c_void,s:*const f32,y:*mut f32,m:i32,n:i32,k:i32)->i32;}
        assert_eq!(unsafe{rs_fp8_bf16_skinny(input.data_ptr(),q.data_ptr(),s.data_ptr().cast(),y.data_ptr().cast(),x.size()[0] as i32,q.size()[0] as i32,q.size()[1] as i32)},0);
    } else if x.size()[0]<=16 {
        let input=input.to_kind(Kind::Float);
        extern "C"{fn rs_fp8_bf16(x:*const f32,w:*const std::ffi::c_void,s:*const f32,y:*mut f32,m:i32,n:i32,k:i32)->i32;}
        assert_eq!(unsafe{rs_fp8_bf16(input.data_ptr().cast(),q.data_ptr(),s.data_ptr().cast(),y.data_ptr().cast(),x.size()[0] as i32,q.size()[0] as i32,q.size()[1] as i32)},0);
    } else {
        // Same quantized weights for long append/prefill. A fallback to the
        // unquantized source would silently change the model across shapes.
        let weight=q.to_kind(Kind::BFloat16).contiguous();
        extern "C"{fn rs_bf16_partial(x:*const u8,w:*const u8,y:*mut f32,m:i32,n:i32,k:i32,ldw:i32)->i32;}
        assert_eq!(unsafe{rs_bf16_partial(input.data_ptr().cast(),weight.data_ptr().cast(),y.data_ptr().cast(),x.size()[0] as i32,q.size()[0] as i32,q.size()[1] as i32,q.size()[1] as i32)},0);
        epilogue_inplace(&y,s,false);
    }
    if partial{y}else{y.to_kind(Kind::BFloat16)}
}
pub fn run(x:&Tensor,q:&Tensor,s:&Tensor,partial:bool)->Tensor {
    // Prefill uses the very same quantized weights as decode. Expand only this
    // projection to FP16 and retain FP32 accumulation/scaling, rather than
    // silently switching back to the original unquantized weights for m > 16.
    let large=std::env::var("GLM53_FP8_LARGE").unwrap_or_else(|_|"0".into());
    if x.size()[0]>16 && (["1","3","4"].contains(&large.as_str()) || (large=="2" && x.size()[0]<=64) || (large=="5" && x.size()[0]<=128)) {
        let input=x.to_kind(Kind::Float).contiguous();let y=Tensor::empty([x.size()[0],q.size()[0]],(Kind::Float,x.device()));
        extern "C"{fn rs_fp8_large(x:*const f32,w:*const std::ffi::c_void,s:*const f32,y:*mut f32,m:i32,n:i32,k:i32,rounded:i32,mode:i32)->i32;}
        assert_eq!(unsafe{rs_fp8_large(input.data_ptr().cast(),q.data_ptr(),s.data_ptr().cast(),y.data_ptr().cast(),x.size()[0] as i32,q.size()[0] as i32,q.size()[1] as i32,i32::from(!partial),large.parse().unwrap())},0);return y;
    }
    if x.size()[0]>16 {
        let input=crate::weights::half_input_pub(x).contiguous();let weight=q.to_kind(Kind::Half).contiguous();
        let y=Tensor::empty([x.size()[0],q.size()[0]],(Kind::Float,x.device()));
        extern "C"{fn rs_mm16_f32(x:*const u8,w:*const u8,y:*mut u8,m:i32,n:i32,k:i32)->i32;}
        assert_eq!(unsafe{rs_mm16_f32(input.data_ptr().cast(),weight.data_ptr().cast(),y.data_ptr().cast(),x.size()[0] as i32,q.size()[0] as i32,q.size()[1] as i32)},0);
        if std::env::var("GLM53_FP8_EPILOGUE").as_deref()==Ok("1") {
            // Fresh private GEMM output: scale and preserve the Half rounding
            // boundary in place, before exposing it to any downstream consumer.
            epilogue_inplace(&y,s,!partial);return y;
        }
        let y=y*s.unsqueeze(0);return if partial{y}else{y.to_kind(Kind::Half).to_kind(Kind::Float)};
    }
    let x=x.to_kind(Kind::Float).contiguous();let y=Tensor::empty([x.size()[0],q.size()[0]],(Kind::Float,x.device()));
    extern "C"{fn rs_fp8_dense(x:*const f32,w:*const std::ffi::c_void,s:*const f32,y:*mut f32,m:i32,n:i32,k:i32,rounded:i32)->i32;}
    assert_eq!(unsafe{rs_fp8_dense(x.data_ptr().cast(),q.data_ptr(),s.data_ptr().cast(),y.data_ptr().cast(),x.size()[0] as i32,q.size()[0] as i32,q.size()[1] as i32,i32::from(!partial))},0);y
}

pub(crate) fn epilogue_inplace(y:&Tensor,scale:&Tensor,rounded:bool) {
    assert_eq!(y.kind(),Kind::Float);assert_eq!(scale.kind(),Kind::Float);
    assert!(y.is_contiguous()&&scale.is_contiguous());assert_eq!(scale.numel() as i64,y.size()[1]);
    extern "C"{fn rs_fp8_epilogue(y:*mut f32,scale:*const f32,rows:i32,cols:i32,rounded:i32)->i32;}
    assert_eq!(unsafe{rs_fp8_epilogue(y.data_ptr().cast(),scale.data_ptr().cast(),y.size()[0] as i32,y.size()[1] as i32,i32::from(rounded))},0);
}

/// Exact host dispatcher flags. Quantized weight cache identity is unchanged:
/// these flags change only small-kernel instruction/layout and must bind graphs.
/// W04 candidate kernel (L1: summation order differs). Binds graphs like the small flags.
pub(crate) fn skinny_enabled()->bool {std::env::var("GLM53_FP8_SKINNY").as_deref()==Ok("1")}
/// Full FP8 dispatch identity for captured graphs: small-kernel flags plus skinny arm.
/// Also carries the fused mhc_pre bit (bit 2): any flag that changes captured launches must bind graphs.
pub(crate) fn graph_signature()->((bool,bool),u64) {(small_kernel_signature(),(crate::tp::ar_fused_enabled() as u64)<<32|((std::env::var("GLM53_MOE_PACK_DIRECT").as_deref()==Ok("1")) as u64)<<33|((std::env::var("GLM53_KDA_GATE_SMEM").as_deref()==Ok("1")) as u64)<<34|((std::env::var("GLM53_KDA_CONV_CACHED").as_deref()==Ok("1")) as u64)<<35|((std::env::var("GLM53_KDA_CONV_SILU").as_deref()==Ok("1")) as u64)<<36|((std::env::var("GLM53_SHARED_GU_F32").as_deref()==Ok("1")) as u64)<<37|((std::env::var("GLM53_ROUTER_HALF_W").as_deref()==Ok("1")) as u64)<<38|(crate::mla_latent::node_batch_enabled() as u64)<<39|((std::env::var("GLM53_ROUTER_ONE").as_deref()==Ok("1")) as u64)<<40|((std::env::var("GLM53_MHC_FINISH_REG").as_deref()==Ok("1")) as u64)<<41|((std::env::var("GLM53_DRAFT_INT4_LAYERS").as_deref()==Ok("1")) as u64)<<42|((std::env::var("GLM53_KDA_GATE_ONE").as_deref()==Ok("1")) as u64)<<43|(crate::dsa_topk::fast_enabled() as u64)<<44|(crate::sampling::fused_enabled() as u64)<<45|(crate::mla_latent::bmm_c12_enabled() as u64)<<46|((std::env::var("GLM53_KDA_CORR_WARP").as_deref()==Ok("5")) as u64)<<47|((std::env::var("GLM53_KDA_CONV_L2").as_deref()==Ok("1")) as u64)<<48|(crate::c12::q8_on() as u64)<<49|(crate::c12::q4_on() as u64)<<51|((std::env::var("GLM53_DRAFT_Q4").as_deref()==Ok("1")) as u64)<<50|(crate::kda::gate_side_enabled() as u64)<<52|((std::env::var("GLM53_MOE_ROUTE_SIDE").as_deref()==Ok("1")) as u64)<<53|(crate::mla_latent::index_side_enabled() as u64)<<54|u64::from(skinny_enabled() as u32|(qkv_fused_enabled() as u32)<<1|(crate::mhc::pre_fused_enabled() as u32)<<2|(dsa_score_mode() as u32)<<3|(crate::kda::gate_fused_enabled() as u32)<<6|(crate::mla_latent::node_fused_enabled() as u32)<<7|(crate::moe::router_fused_enabled() as u32)<<8|(crate::mla_latent::half_bmm_enabled() as u32)<<9|(crate::spec_probe::state_inplace_enabled() as u32)<<10|(crate::weights::half_skinny_enabled() as u32)<<11|(crate::kda::norm_fused_enabled() as u32)<<12|((std::env::var("GLM53_KDA_CORR_WARP").as_deref()==Ok("1")) as u32)<<13|((std::env::var("GLM53_HALF_SKINNY_K1536").as_deref()==Ok("1")) as u32)<<14|((std::env::var("GLM53_MHC_PRE_TC").as_deref()==Ok("1")) as u32)<<15|((std::env::var("GLM53_STATIC_TENSORS").as_deref()==Ok("1")) as u32)<<16|((std::env::var("GLM53_RDMA_AR").as_deref()==Ok("1")) as u32)<<17|(crate::mla_latent::chain_shared_enabled() as u32)<<18|(crate::mla_latent::kv_fp8_enabled() as u32)<<19|((std::env::var("GLM53_MOE_SHARED_STREAM").as_deref()==Ok("1")) as u32)<<20|((std::env::var("GLM53_SHARED_GU_ROWS").as_deref()==Ok("1")) as u32)<<21|((std::env::var("GLM53_DSA_INDEX_BF16_KERNEL").as_deref()!=Ok("0")) as u32)<<22|((std::env::var("GLM53_DSA_SCORE_MULTI").as_deref()==Ok("1")) as u32)<<23|((std::env::var("GLM53_MHC_POST_PRE_DECODE").as_deref()==Ok("1")) as u32)<<24|((std::env::var("GLM53_PDL").as_deref()==Ok("1")) as u32)<<25|(crate::tp::ar_prefetch_enabled() as u32)<<26|(crate::forward::verify_invariant() as u32)<<27|(crate::c12::enabled() as u32)<<28|((std::env::var("GLM53_KDA_CORR_WARP").as_deref()==Ok("2")) as u32)<<29|((std::env::var("GLM53_KDA_CORR_WARP").as_deref()==Ok("3")) as u32)<<30|((std::env::var("GLM53_KDA_CORR_WARP").as_deref()==Ok("4")) as u32)<<31))}
fn dsa_score_mode()->u8 {std::env::var("GLM53_DSA_SCORE_FUSED").ok().and_then(|v|v.parse::<u8>().ok()).filter(|v|*v<=5).unwrap_or(0)}
/// One launch for row-concatenated producers; requires the skinny kernel (per-row split-K
/// layout is independent of N, so each output equals the separate skinny call).
pub(crate) fn qkv_fused_enabled()->bool {skinny_enabled()&&std::env::var("GLM53_FP8_QKV_FUSED").as_deref()==Ok("1")}
/// Returns [rows, sum(N_i)] exactly as cat(mm16(x,w_i)) would, or None when not eligible.
pub fn try_run_rows(x:&Tensor,ws:&[&Tensor])->Option<Tensor> {
    if !qkv_fused_enabled()||!x.device().is_cuda()||x.dim()!=2||!(2..=16).contains(&x.size()[0]) {return None;}
    let keys:Vec<usize>=ws.iter().map(|w|w.data_ptr() as usize).collect();
    let (q,s)=FUSED.with(|f|f.borrow().get(&keys).map(|(q,s)|(q.shallow_clone(),s.shallow_clone()))).or_else(||{
        let parts:Option<Vec<(Tensor,Tensor)>>=CACHE.with(|c|{let mut c=c.borrow_mut();ws.iter().map(|w|{
            let e=c.get_mut(&(w.data_ptr() as usize))?;
            if e.source.size()!=w.size()||e.source.stride()!=w.stride()||e.source.kind()!=Kind::Half||std::env::var(e.flag).as_deref()!=Ok("1"){return None;}
            let (q,s)=e.quant.get_or_insert_with(||{assert!(!crate::tp::graph::capturing(),"FP8 quant cache miss during graph capture");quantize(w)});Some((q.shallow_clone(),s.shallow_clone()))}).collect()});
        let parts=parts?;
        assert!(!crate::tp::graph::capturing(),"FP8 fused QKV cache miss during graph capture");
        let q=Tensor::cat(&parts.iter().map(|p|&p.0).collect::<Vec<_>>(),0);
        let s=Tensor::cat(&parts.iter().map(|p|&p.1).collect::<Vec<_>>(),0);
        drop(parts);
        CACHE.with(|c|{let mut c=c.borrow_mut();let mut offset=0;for w in ws {
            let n=w.size()[0];let e=c.get_mut(&(w.data_ptr() as usize)).unwrap();
            e.quant=Some((q.narrow(0,offset,n),s.narrow(0,offset,n)));offset+=n;}});
        FUSED.with(|f|f.borrow_mut().insert(keys.clone(),(q.shallow_clone(),s.shallow_clone())));
        Some((q,s))})?;
    Some(run(x,&q,&s,false))
}
pub(crate) fn small_kernel_signature()->(bool,bool) {
    (std::env::var("GLM53_FP8_SMALL_TRANSPOSE").as_deref()==Ok("1"),
     std::env::var("GLM53_FP8_SMALL_PAD").as_deref()==Ok("1"))
}
