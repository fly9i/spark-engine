//! Proposal 2 (GLM53_C12=1): lossless 12-bit coding of the resident Half weights (shim/c12.cuh).
//! Decode-sized calls (1..=GLM53_C12_MAX_ROWS rows, default 32) read the coded copy: 0.75 of the Half bytes.
//! The decoded MMA operands equal the Half weights exactly (the coding is lossless on the BF16 value of each
//! resident Half, and decoding rounds back with the same RN conversion), so:
//! - 2..8 rows: bitwise equal to skinny_half with the same split-K (the default Half path, GLM53_HALF_SKINNY=1);
//! - every row's value is independent of the row count (1..32), unlike gemv (1 row) / cuBLAS (> 8 rows): L1
//!   against those two, and row-count invariant (proposal 3).
//! The Half sources stay resident for prefill-sized calls (cuBLAS). Coded copies are process-lifetime, keyed by the
//! source pointer and checked against its size/stride like dense_fp8. Row-concatenated groups (KDA q/k/v, gate/up)
//! are encoded once; members are row views of the group storage (escape CSR offsets are absolute).
use std::{cell::RefCell,collections::HashMap,sync::OnceLock};
use tch::{Tensor,Kind};

/// s: Some for the Q8 format (GLM53_Q8, shim/q8.cuh): m8 then holds the int8 weights in m8 order, s the FP32 scale
/// per (row, 128 k); e4/eb/ptr/col/val are empty placeholders.
/// q4: Some for the dense Q4 format (GLM53_DENSE_Q4_SET, shim/q4d.cuh): m8 then holds the packed nibbles [N,K/2], q4 the
/// half2 (scale, minimum) per row and 64 k.
struct Coded {source:Tensor,m8:Tensor,e4:Tensor,eb:Tensor,ptr:Tensor,col:Tensor,val:Tensor,s:Option<Tensor>,q4:Option<Tensor>,t:Option<Tensor>}
impl Coded {
    fn view(&self,src:&Tensor,r0:i64,n:i64)->Coded {Coded{source:src.shallow_clone(),m8:self.m8.narrow(0,r0,n),e4:self.e4.narrow(0,r0,n),
        eb:self.eb.narrow(0,r0,n),ptr:self.ptr.narrow(0,r0,n+1),col:self.col.shallow_clone(),val:self.val.shallow_clone(),
        s:self.s.as_ref().map(|s|s.narrow(0,r0,n)),q4:self.q4.as_ref().map(|t|t.narrow(0,r0,n)),
        t:self.t.as_ref().map(|t|{assert!(r0%16==0&&n%16==0,"tiled Q8 view off the 16-row grid");let per=self.m8.size()[1]/128*2112;t.narrow(0,r0/16*per,n/16*per)})}}
    fn launch(&self,x:&Tensor,y:&Tensor,m:i64,out:i32,ks:i32) {
        extern "C"{fn rs_c12_gemm(x:*const f32,m8:*const u8,e4:*const u8,eb:*const u8,ptr:*const i32,col:*const i32,val:*const u16,n:i32,k:i32,y:*mut std::ffi::c_void,m:i32,out:i32,ks:i32)->i32;}
        if let Some(sm)=&self.q4 {
            let (n,k)=(self.m8.size()[0],self.m8.size()[1]*2);
            extern "C"{fn rs_q4d_gemm(x:*const f32,q:*const u8,sm:*const std::ffi::c_void,n:i32,k:i32,y:*mut std::ffi::c_void,m:i32,out:i32,ks:i32)->i32;}
            assert_eq!(unsafe{rs_q4d_gemm(x.data_ptr().cast(),self.m8.data_ptr().cast(),sm.data_ptr(),n as i32,k as i32,y.data_ptr(),m as i32,out,ks)},0,"Q4 GEMM");
            return;
        }
        let (n,k)=(self.m8.size()[0],self.m8.size()[1]);
        if let Some(t)=&self.t {
            // GLM53_Q8_TILED: same values and arithmetic as the rs_q8_gemm calls below, tiled layout (L0).
            let (n,k)=(self.m8.size()[0],self.m8.size()[1]);
            let xh=(m>8 && q8_xhalf_on()).then(||xhalf_of(x));
            extern "C"{fn rs_q8t_gemm(x:*const std::ffi::c_void,t:*const u8,n:i32,k:i32,y:*mut std::ffi::c_void,m:i32,out:i32,ks:i32,xhalf:i32)->i32;}
            let xp=xh.as_ref().map_or(x.data_ptr(),|h|h.data_ptr());
            assert_eq!(unsafe{rs_q8t_gemm(xp,t.data_ptr().cast(),n as i32,k as i32,y.data_ptr(),m as i32,out,ks,i32::from(xh.is_some()))},0,"Q8 tiled GEMM");
            return;
        }
        if let Some(s)=&self.s {
            if m>8 && q8_xhalf_on() {
                let xh=xhalf_of(x);
                extern "C"{fn rs_q8_gemm_xh(x:*const std::ffi::c_void,q:*const u8,s:*const f32,n:i32,k:i32,y:*mut std::ffi::c_void,m:i32,out:i32,ks:i32)->i32;}
                assert_eq!(unsafe{rs_q8_gemm_xh(xh.data_ptr(),self.m8.data_ptr().cast(),s.data_ptr().cast(),n as i32,k as i32,y.data_ptr(),m as i32,out,ks)},0,"Q8 GEMM (Half X)");
                return;
            }
            extern "C"{fn rs_q8_gemm(x:*const f32,q:*const u8,s:*const f32,n:i32,k:i32,y:*mut std::ffi::c_void,m:i32,out:i32,ks:i32)->i32;}
            assert_eq!(unsafe{rs_q8_gemm(x.data_ptr().cast(),self.m8.data_ptr().cast(),s.data_ptr().cast(),n as i32,k as i32,y.data_ptr(),m as i32,out,ks)},0,"Q8 GEMM");
            return;
        }
        if m>24 && xhalf_on() {
            let xh=xhalf_of(x);
            extern "C"{fn rs_c12_gemm_xh(x:*const std::ffi::c_void,m8:*const u8,e4:*const u8,eb:*const u8,ptr:*const i32,col:*const i32,val:*const u16,n:i32,k:i32,y:*mut std::ffi::c_void,m:i32,out:i32,ks:i32)->i32;}
            assert_eq!(unsafe{rs_c12_gemm_xh(xh.data_ptr(),self.m8.data_ptr().cast(),self.e4.data_ptr().cast(),self.eb.data_ptr().cast(),self.ptr.data_ptr().cast(),
                self.col.data_ptr().cast(),self.val.data_ptr().cast(),n as i32,k as i32,y.data_ptr(),m as i32,out,ks)},0,"C12 GEMM (Half X)");
            return;
        }
        assert_eq!(unsafe{rs_c12_gemm(x.data_ptr().cast(),self.m8.data_ptr().cast(),self.e4.data_ptr().cast(),self.eb.data_ptr().cast(),self.ptr.data_ptr().cast(),
            self.col.data_ptr().cast(),self.val.data_ptr().cast(),n as i32,k as i32,y.data_ptr(),m as i32,out,ks)},0,"C12 GEMM");
    }
}
/// GLM53_C12_XHALF=1 (L0): C12 GEMMs of more than 24 rows read a Half copy of X (the RN values the kernel would make
/// itself) with 16-byte loads. The copy is reused for the next GEMM of the same input (same storage and version; the
/// entry keeps x alive so its address cannot be recycled); never across or inside graph capture.
/// GLM53_Q8_XHALF=1 (L0 against the Q8 path): Q8 GEMMs of more than 8 rows read the Half copy of X (xhalf_of), which
/// removes the per-warp FP32 load + Half conversion of X that made 32-row calls compute-bound (N=12288: 409 -> 265 us).
fn q8_xhalf_on()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||std::env::var("GLM53_Q8_XHALF").as_deref()==Ok("1"))}
fn xhalf_on()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||std::env::var("GLM53_C12_XHALF").as_deref()==Ok("1"))}
fn xhalf_of(x:&Tensor)->Tensor {
    thread_local!{static LAST:RefCell<Option<(Tensor,i64,Tensor)>>=const{RefCell::new(None)};}
    let x=x.contiguous();
    if crate::tp::graph::capturing() {LAST.with(|l|*l.borrow_mut()=None);return x.to_kind(Kind::Half);}
    extern "C"{fn rs_tensor_version(t:*const std::ffi::c_void)->i64;}
    let version=unsafe{rs_tensor_version(x.as_ptr().cast())};
    LAST.with(|l|{let mut l=l.borrow_mut();
        if let Some((src,v,h))=l.as_ref() {
            if src.data_ptr()==x.data_ptr() && src.size()==x.size() && src.kind()==x.kind() && *v==version {return h.shallow_clone();}
        }
        let h=x.to_kind(Kind::Half);*l=Some((x.shallow_clone(),version,h.shallow_clone()));h})
}
thread_local!{static CACHE:RefCell<HashMap<usize,Coded>>=RefCell::new(HashMap::new());}
thread_local!{static GROUPS:RefCell<HashMap<Vec<usize>,Coded>>=RefCell::new(HashMap::new());}

pub(crate) fn enabled()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||std::env::var("GLM53_C12").as_deref()==Ok("1"))}
/// GLM53_Q8=1 (needs GLM53_C12=1): decode-sized calls of the weight classes in GLM53_Q8_SET (comma list of dense, kda,
/// shared, mla, index, head; default all) read a near-lossless int8 copy (shim/q8.cuh) instead of the lossless C12 one.
/// Lossy (L3): every operand is Half_RN(q*scale), relative RMS error logged per class at load.
pub(crate) fn q8_on()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||enabled() && std::env::var("GLM53_Q8").as_deref()==Ok("1"))}
fn q8_for(class:&str)->bool {
    static SET:OnceLock<Vec<String>>=OnceLock::new();
    q8_on() && SET.get_or_init(||std::env::var("GLM53_Q8_SET").unwrap_or_else(|_|"dense,kda,shared,mla,index,head".into())
        .split(',').map(|c|c.trim().to_string()).filter(|c|!c.is_empty()).collect()).iter().any(|c|c==class)
}
/// GLM53_C12_MIN_ROWS (default 1; 2 = L0 check: 1-row calls keep gemv, so with MAX_ROWS=8 every C12 call replaces a
/// skinny_half call and outputs must be bitwise those of the Half path).
fn min_rows()->i64 {static M:OnceLock<i64>=OnceLock::new();*M.get_or_init(||std::env::var("GLM53_C12_MIN_ROWS").ok().map(|v|v.parse().expect("GLM53_C12_MIN_ROWS")).unwrap_or(1))}
pub(crate) fn index_enabled()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||enabled() && std::env::var("GLM53_C12_INDEX").as_deref()==Ok("1"))}
fn max_rows()->i64 {static M:OnceLock<i64>=OnceLock::new();*M.get_or_init(||std::env::var("GLM53_C12_MAX_ROWS").ok().map(|v|v.parse().expect("GLM53_C12_MAX_ROWS")).unwrap_or(32))}
/// skinny_half's split-K choice (weights.rs half_skinny), so 2..8-row outputs stay bitwise equal to it.
fn ks_for(n:i64)->i32 {if n<=2048{8}else{4}}
/// Split-K of a launch: skinny_half's choice (bitwise equal to it for 2..8 rows), or GLM53_C12_KS=8 wherever K allows
/// (3-4% faster on the large shapes in c12_bench; L1: another summation split).
fn launch_ks(n:i64,k:i64)->i32 {
    static KS8:OnceLock<bool>=OnceLock::new();
    if *KS8.get_or_init(||std::env::var("GLM53_C12_KS").as_deref()==Ok("8")) && (k/32)%8==0 && (k/32/8)%4==0 {8} else {ks_for(n)}
}
fn shape_ok(n:i64,k:i64)->bool {let chunks=k/32;n%16==0 && k%128==0 && chunks%(ks_for(n) as i64)==0 && (chunks/ks_for(n) as i64)%4==0}

/// Q8 copy of a contiguous Half [N,K] (BF16-valued) weight; accumulates (squared error, squared source) per class.
fn encode_q8(src:&Tensor,class:&str)->Coded {
    assert!(!crate::tp::graph::capturing(),"Q8 encode during graph capture");
    let (n,k)=(src.size()[0],src.size()[1]);let dev=src.device();
    let b=src.to_kind(Kind::BFloat16).contiguous();
    let s=Tensor::empty([n,k/128],(Kind::Float,dev));let q=Tensor::empty([n,k],(Kind::Uint8,dev));
    let err=Tensor::zeros([2],(Kind::Double,dev));
    let mse=i32::from(std::env::var("GLM53_Q8_MSE").as_deref()!=Ok("0"));
    extern "C"{fn rs_q8_encode(w:*const std::ffi::c_void,n:i32,k:i32,mse:i32,s:*mut f32,q:*mut std::ffi::c_void,err:*mut f64)->i32;}
    assert_eq!(unsafe{rs_q8_encode(b.data_ptr(),n as i32,k as i32,mse,s.data_ptr().cast(),q.data_ptr(),err.data_ptr().cast())},0,"Q8 encode");
    let (e2,r2)=(err.double_value(&[0]),err.double_value(&[1]));
    Q8STATS.with(|m|{let mut m=m.borrow_mut();let e=m.entry(class.to_string()).or_insert((0.,0.,0));e.0+=e2;e.1+=r2;e.2+=n*k;});
    STATS.with(|st|{let mut st=st.borrow_mut();st.0+=n*k;st.2+=n*k+n*(k/128)*4;});
    let z=|shape:&[i64],kind:Kind|Tensor::zeros(shape,(kind,dev));
    if q8_tiled_on() {
        // GLM53_Q8_TILED=1: repack into the tiled layout; q and s are released (shape-only placeholders keep sizes and views).
        let t=Tensor::empty([n/16*(k/128)*2112],(Kind::Uint8,dev));
        extern "C"{fn rs_q8t_pack(q:*const std::ffi::c_void,s:*const f32,n:i32,k:i32,t:*mut std::ffi::c_void)->i32;}
        assert_eq!(unsafe{rs_q8t_pack(q.data_ptr(),s.data_ptr().cast(),n as i32,k as i32,t.data_ptr())},0,"Q8 tile pack");
        let ph=|shape:&[i64],kind:Kind|Tensor::zeros([1],(kind,dev)).expand(shape,false);
        return Coded{source:src.shallow_clone(),m8:ph(&[n,k],Kind::Uint8),e4:z(&[n,0],Kind::Uint8),eb:z(&[n],Kind::Uint8),ptr:z(&[n+1],Kind::Int),col:z(&[1],Kind::Int),val:z(&[1],Kind::Int16),
            s:Some(ph(&[n,k/128],Kind::Float)),q4:None,t:Some(t)};
    }
    Coded{source:src.shallow_clone(),m8:q,e4:z(&[n,0],Kind::Uint8),eb:z(&[n],Kind::Uint8),ptr:z(&[n+1],Kind::Int),col:z(&[1],Kind::Int),val:z(&[1],Kind::Int16),s:Some(s),q4:None,t:None}
}
/// GLM53_Q8_TILED=1 (L0 against the Q8 path): Q8 weights stored in skinny_q8t's tiled layout (contiguous 512 B warp loads).
fn q8_tiled_on()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||std::env::var("GLM53_Q8_TILED").as_deref()==Ok("1"))}
thread_local!{static Q8STATS:RefCell<std::collections::BTreeMap<String,(f64,f64,i64)>>=RefCell::new(std::collections::BTreeMap::new());}
/// GLM53_DENSE_Q4_SET (comma list of the classes above; default none): those classes read an affine 4-bit copy
/// (shim/q4d.cuh, 4.5 bits per weight) instead; takes precedence over GLM53_Q8. Lossy (L3), error logged per class.
fn q4_for(class:&str)->bool {
    static SET:OnceLock<Vec<String>>=OnceLock::new();
    enabled() && SET.get_or_init(||std::env::var("GLM53_DENSE_Q4_SET").unwrap_or_default().split(',').map(|c|c.trim().to_string()).filter(|c|!c.is_empty()).collect()).iter().any(|c|c==class)
}
pub(crate) fn q4_on()->bool {enabled() && std::env::var("GLM53_DENSE_Q4_SET").is_ok_and(|v|!v.trim().is_empty())}
fn encode_q4(src:&Tensor,class:&str)->Coded {
    assert!(!crate::tp::graph::capturing(),"Q4 encode during graph capture");
    let (n,k)=(src.size()[0],src.size()[1]);let dev=src.device();
    let b=src.to_kind(Kind::BFloat16).contiguous();
    let sm=Tensor::empty([n,k/64,2],(Kind::Half,dev));let q=Tensor::empty([n,k/2],(Kind::Uint8,dev));
    let err=Tensor::zeros([2],(Kind::Double,dev));
    let mse=i32::from(std::env::var("GLM53_DENSE_Q4_MSE").as_deref()!=Ok("0"));
    extern "C"{fn rs_q4d_encode(w:*const std::ffi::c_void,n:i32,k:i32,mse:i32,sm:*mut std::ffi::c_void,q:*mut std::ffi::c_void,err:*mut f64)->i32;}
    assert_eq!(unsafe{rs_q4d_encode(b.data_ptr(),n as i32,k as i32,mse,sm.data_ptr(),q.data_ptr(),err.data_ptr().cast())},0,"Q4 encode");
    let (e2,r2)=(err.double_value(&[0]),err.double_value(&[1]));
    Q8STATS.with(|m|{let mut m=m.borrow_mut();let e=m.entry(format!("q4:{class}")).or_insert((0.,0.,0));e.0+=e2;e.1+=r2;e.2+=n*k;});
    STATS.with(|st|{let mut st=st.borrow_mut();st.0+=n*k;st.2+=n*k/2+n*(k/64)*4;});
    let z=|shape:&[i64],kind:Kind|Tensor::zeros(shape,(kind,dev));
    Coded{source:src.shallow_clone(),m8:q,e4:z(&[n,0],Kind::Uint8),eb:z(&[n],Kind::Uint8),ptr:z(&[n+1],Kind::Int),col:z(&[1],Kind::Int),val:z(&[1],Kind::Int16),s:None,q4:Some(sm),t:None}
}
fn encode_class(src:&Tensor,class:&str)->Coded {if q4_for(class) {encode_q4(src,class)} else if q8_for(class) {encode_q8(src,class)} else {encode(src)}}
/// Encode a contiguous Half [N,K] (or the row concatenation of several with equal K).
fn encode(src:&Tensor)->Coded {
    assert!(!crate::tp::graph::capturing(),"C12 encode during graph capture");
    let (n,k)=(src.size()[0],src.size()[1]);let dev=src.device();
    // Half -> BF16 is exact here: a normal Half of a BF16 checkpoint value has <= 8 significant bits, and a Half
    // subnormal (the RN image of a smaller BF16 value) has fewer; both are BF16-representable.
    let b=src.to_kind(Kind::BFloat16).contiguous();
    let back=b.to_kind(src.kind());assert!(back.equal(src),"C12: resident weight is not BF16-representable");drop(back);
    let eb=Tensor::empty([n],(Kind::Uint8,dev));let cnt=Tensor::empty([n],(Kind::Int,dev));
    extern "C"{fn rs_c12_stats(w:*const std::ffi::c_void,n:i32,k:i32,eb:*mut u8,cnt:*mut i32)->i32;
        fn rs_c12_pack(w:*const std::ffi::c_void,n:i32,k:i32,eb:*const u8,ptr:*const i32,m8:*mut u8,e4:*mut u8,col:*mut i32,val:*mut u16)->i32;}
    assert_eq!(unsafe{rs_c12_stats(b.data_ptr(),n as i32,k as i32,eb.data_ptr().cast(),cnt.data_ptr().cast())},0);
    let ptr=Tensor::cat(&[Tensor::zeros([1],(Kind::Int64,dev)),cnt.cumsum(0,Kind::Int64)],0);
    let total=ptr.int64_value(&[n]);assert!(total<i32::MAX as i64);
    let ptr=ptr.to_kind(Kind::Int);
    let m8=Tensor::empty([n,k],(Kind::Uint8,dev));let e4=Tensor::empty([n,k/2],(Kind::Uint8,dev));
    let col=Tensor::empty([total.max(1)],(Kind::Int,dev));let val=Tensor::empty([total.max(1)],(Kind::Int16,dev));
    assert_eq!(unsafe{rs_c12_pack(b.data_ptr(),n as i32,k as i32,eb.data_ptr().cast(),ptr.data_ptr().cast(),m8.data_ptr().cast(),e4.data_ptr().cast(),col.data_ptr().cast(),val.data_ptr().cast())},0);
    // Always-on proof: every weight decoded by the GEMM's c12_dec8 equals the Half RN of its BF16 source value.
    let bad=Tensor::zeros([1],(Kind::Int64,dev));
    extern "C"{fn rs_c12_verify(w:*const std::ffi::c_void,m8:*const u8,e4:*const u8,eb:*const u8,ptr:*const i32,col:*const i32,val:*const u16,n:i32,k:i32,bad:*mut u64)->i32;}
    assert_eq!(unsafe{rs_c12_verify(b.data_ptr(),m8.data_ptr().cast(),e4.data_ptr().cast(),eb.data_ptr().cast(),ptr.data_ptr().cast(),col.data_ptr().cast(),val.data_ptr().cast(),n as i32,k as i32,bad.data_ptr().cast())},0);
    let mismatches=bad.int64_value(&[0]);assert_eq!(mismatches,0,"C12: {mismatches} of {} decoded weights differ from the source [{n},{k}]",n*k);
    STATS.with(|s|{let mut s=s.borrow_mut();s.0+=n*k;s.1+=total;s.2+=n*k*3/2+n*5+total*6;});
    Coded{source:src.shallow_clone(),m8,e4,eb,ptr,col,val,s:None,q4:None,t:None}
}
/// Coded tensors (m8, e4, eb, ptr, col, val) of a Half [N,K] for kernels outside this module's GEMM registry
/// (the MLA per-head bmm); load-time verified like every other coding.
pub(crate) fn encode_parts(src:&Tensor)->[Tensor;6] {let c=encode(src);[c.m8,c.e4,c.eb,c.ptr,c.col,c.val]}
thread_local!{static STATS:RefCell<(i64,i64,i64)>=const{RefCell::new((0,0,0))};}   // weights, escapes, coded bytes

/// Half weights, or FP32-resident weights whose values are BF16 (the DSA indexer projections, GLM53_C12_INDEX=1).
fn eligible_weight(w:&Tensor)->bool {
    (w.kind()==Kind::Half || (w.kind()==Kind::Float && w.dim()==2 && w.to_kind(Kind::BFloat16).to_kind(Kind::Float).equal(w))) && w.dim()==2 && w.is_contiguous() && w.device().is_cuda() && !w.stride().iter().all(|&s|s==0)
        && shape_ok(w.size()[0],w.size()[1]) && !crate::dense_fp8::registered_enabled(w)
}
fn single(w:&Tensor,class:&str) {
    if !eligible_weight(w) || CACHE.with(|c|c.borrow().contains_key(&(w.data_ptr() as usize))) {return;}
    let c=encode_class(w,class);CACHE.with(|m|m.borrow_mut().insert(w.data_ptr() as usize,c));
}
/// Row-concatenated producers sharing one input: one coded storage, members are row views. The group's split-K
/// (from the total N) must equal each member's, otherwise members are encoded on their own.
fn group(ws:&[&Tensor],class:&str) {
    let total:i64=ws.iter().map(|w|w.size()[0]).sum();
    if ws.iter().any(|w|!eligible_weight(w)||w.size()[1]!=ws[0].size()[1]) || !shape_ok(total,ws[0].size()[1])
        || ws.iter().any(|w|ks_for(w.size()[0])!=ks_for(total)) {for w in ws{single(w,class);}return;}
    let keys:Vec<usize>=ws.iter().map(|w|w.data_ptr() as usize).collect();
    if GROUPS.with(|g|g.borrow().contains_key(&keys)) {return;}
    // The group entry must not keep the concatenated Half copy (encode() records its source): w36 measured +3.8 GiB/rank.
    let cat=Tensor::cat(ws,0);let mut g=encode_class(&cat,class);g.source=Tensor::zeros([1],(Kind::Half,cat.device()));drop(cat);
    let mut r0=0;for w in ws {let n=w.size()[0];let v=g.view(w,r0,n);CACHE.with(|m|m.borrow_mut().insert(w.data_ptr() as usize,v));r0+=n;}
    GROUPS.with(|m|m.borrow_mut().insert(keys,g));
}
pub fn register(layers:&[crate::weights::LayerWeights],lm_head:&Tensor) {
    if !enabled() {return;}
    let t0=std::time::Instant::now();
    for l in layers {
        if let Some(w)=&l.dense {group(&[&w.wg,&w.wu],"dense");single(&w.wd,"dense");}
        if let Some(w)=&l.kda {group(&[&w.wq,&w.wk,&w.wv],"kda");single(&w.wo,"kda");}
        if let Some(w)=&l.moe {group(&[&w.sh_wg,&w.sh_wu],"shared");single(&w.sh_wd,"shared");}
        if let Some(w)=&l.mla {group(&[&w.q_a,&w.kv_a],"mla");for t in [&w.q_b,&w.wo]{single(t,"mla");}
            // DSA indexer projections (FP32-resident BF16 values; TF32 cuBLAS today): C12 reads 12 of their 32 bits. Its
            // Half-rounded input keeps TF32's 10-bit input mantissa (L1). Keys and gate share x: one launch.
            if index_enabled() {if let Some(ix)=&w.indexer {group(&[&ix.k,&ix.gate],"index");single(&ix.q,"index");single(&ix.score,"index");}}}
    }
    single(lm_head,"head");
    tch::Cuda::synchronize(0);
    // Encoding temporaries (BF16 copies, group concatenations, round-trip checks) would otherwise stay in the caching
    // allocator (w36: r13b resident +8.8 GiB against r13a for 4.77 GiB of coded data).
    extern "C"{fn rs_empty_cache();}unsafe{rs_empty_cache();}
    let (n,e,b)=STATS.with(|s|*s.borrow());
    eprintln!("[c12] coded (every weight decode-verified bitwise) {:.2} GiB of Half weights into {:.2} GiB ({:.1}%), escapes {:.2e}, {} tensors, {:.1} s",
        n as f64*2./(1u64<<30) as f64,b as f64/(1u64<<30) as f64,100.*b as f64/(n as f64*2.),e as f64/n.max(1) as f64,
        CACHE.with(|c|c.borrow().len()),t0.elapsed().as_secs_f64());
    Q8STATS.with(|m|for (class,(e2,r2,n)) in m.borrow().iter() {
        eprintln!("[q8] {class}: {:.2} G weights ({}), relative RMS error {:.3e}",*n as f64/1e9,if class.starts_with("q4:"){"affine 4-bit, half2 range per 64"}else{"int8 + fp32 scale per 128"},(e2/r2.max(1e-30)).sqrt());});
}

/// GLM53_C12_L0CHECK=1 (diagnostic): serve only the calls the Half path sends to skinny_half (weights.rs half_skinny:
/// 2..8 rows, K in {1024,4096} (+1536 with GLM53_HALF_SKINNY_K1536), N <= 16384), so outputs must be bitwise unchanged.
fn l0check()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||std::env::var("GLM53_C12_L0CHECK").as_deref()==Ok("1"))}
fn l0check_ok(x:&Tensor,n:i64)->bool {
    if !l0check() {return true;}
    let k=x.size()[1];let k1536=std::env::var("GLM53_HALF_SKINNY_K1536").as_deref()==Ok("1");
    crate::weights::half_skinny_enabled() && (2..=8).contains(&x.size()[0]) && ([1024,4096].contains(&k)||(k1536&&k==1536)) && n<=16384
}
fn rows_ok(x:&Tensor)->bool {
    enabled() && x.device().is_cuda() && x.dim()==2 && (min_rows()..=max_rows()).contains(&x.size()[0])
        && !crate::root_probe::retain_f32() && !crate::root_probe::full_f32() && !crate::root_probe::recording()
}
fn float_input(x:&Tensor)->Option<Tensor> {
    if x.kind()==Kind::Float && x.is_contiguous() {Some(x.shallow_clone())} else if matches!(x.kind(),Kind::Float|Kind::Half) {Some(x.to_kind(Kind::Float).contiguous())} else {None}
}
/// out: 0 FP32 partial, 1 FP32 of the Half-rounded value (mm16), 2 Half.
pub(crate) fn try_run(x:&Tensor,w:&Tensor,out:i32)->Option<Tensor> {
    if !rows_ok(x) || x.size()[1]!=w.size()[1] {return None;}
    CACHE.with(|c|{let c=c.borrow();let e=c.get(&(w.data_ptr() as usize))?;
        if e.source.size()!=w.size() || e.source.stride()!=w.stride() || e.source.kind()!=w.kind() {return None;}
        if !l0check_ok(x,w.size()[0]) {return None;}
        let x=float_input(x)?;let (m,n)=(x.size()[0],w.size()[0]);
        let y=Tensor::empty([m,n],(if out==2{Kind::Half}else{Kind::Float},x.device()));
        e.launch(&x,&y,m,out,launch_ks(n,w.size()[1]));Some(y)})
}
/// try_run into a caller-provided contiguous output of the right shape and dtype (out mode as try_run). false: not taken.
pub(crate) fn try_run_into(x:&Tensor,w:&Tensor,out:i32,y:&Tensor)->bool {
    if !rows_ok(x) || x.size()[1]!=w.size()[1] || !y.is_contiguous() || y.size()!=[x.size()[0],w.size()[0]]
        || y.kind()!=(if out==2{Kind::Half}else{Kind::Float}) {return false;}
    CACHE.with(|c|{let c=c.borrow();let Some(e)=c.get(&(w.data_ptr() as usize)) else {return false};
        if e.source.size()!=w.size() || e.source.stride()!=w.stride() || e.source.kind()!=w.kind() {return false;}
        if !l0check_ok(x,w.size()[0]) {return false;}
        let Some(x)=float_input(x) else {return false};let m=x.size()[0];
        e.launch(&x,y,m,out,launch_ks(w.size()[0],w.size()[1]));true})
}
/// [rows, sum N_i] = cat(mm16(x,w_i)) for a registered group, in one launch.
pub(crate) fn try_run_rows(x:&Tensor,ws:&[&Tensor])->Option<Tensor> {try_run_rows_out(x,ws,1)}
/// Group launch with an explicit output mode (see try_run); out 2 gives a Half [rows, sum N_i].
pub(crate) fn try_run_rows_out(x:&Tensor,ws:&[&Tensor],out:i32)->Option<Tensor> {
    if !rows_ok(x) {return None;}
    let keys:Vec<usize>=ws.iter().map(|w|w.data_ptr() as usize).collect();
    if !CACHE.with(|c|{let c=c.borrow();ws.iter().all(|w|c.get(&(w.data_ptr() as usize)).is_some_and(|e|e.source.size()==w.size()&&e.source.stride()==w.stride()))}) {return None;}
    GROUPS.with(|g|{let g=g.borrow();let e=g.get(&keys)?;
        if ws.iter().any(|w|!l0check_ok(x,w.size()[0])) {return None;}
        let x=float_input(x)?;let (m,n)=(x.size()[0],e.m8.size()[0]);
        let y=Tensor::empty([m,n],(if out==2{Kind::Half}else{Kind::Float},x.device()));e.launch(&x,&y,m,out,launch_ks(n,e.m8.size()[1]));Some(y)})
}

// ---- Prefill (GLM53_C12_PREFILL=1): rows above the decode range read the coded weight in a tensor-core GEMM
// (shim/c12_big.cuh) instead of cuBLAS on the Half source; with GLM53_C12_FREE_SOURCE=1 the Half sources of coded
// weights are then released (stride-0 placeholders; any non-C12 use panics in mm16's placeholder assert).
fn prefill_enabled()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||enabled() && std::env::var("GLM53_C12_PREFILL").as_deref()==Ok("1"))}
fn big_stages()->i32 {std::env::var("GLM53_C12_BIG_STAGES").ok().and_then(|v|v.parse().ok()).unwrap_or(4)}
/// try_big would run for (x, w).
pub(crate) fn big_eligible(x:&Tensor,w:&Tensor)->bool {
    big_rows(x) && x.size()[1]==w.size()[1] && CACHE.with(|c|c.borrow().get(&(w.data_ptr() as usize)).is_some_and(|e|e.source.size()==w.size() && e.source.stride()==w.stride() && big_shape_ok(w.size()[0],w.size()[1])))
}
fn big_rows(x:&Tensor)->bool {
    prefill_enabled() && x.device().is_cuda() && x.dim()==2 && x.size()[0]>max_rows()
        && !crate::root_probe::retain_f32() && !crate::root_probe::full_f32() && !crate::root_probe::recording()
}
impl Coded {
    /// out[:, :] (row stride may exceed its width) = half(x) . W^T.
    fn big_into(&self,xh:&Tensor,out:&Tensor,rounded:bool) {
        assert!(self.s.is_none()&&self.q4.is_none(),"Q8/Q4 weights have no prefill GEMM (GLM53_C12_PREFILL with GLM53_Q8 / GLM53_DENSE_Q4_SET)");
        extern "C"{fn rs_c12_big(x:*const std::ffi::c_void,m8:*const u8,e4:*const u8,eb:*const u8,ptr:*const i32,col:*const i32,val:*const u16,
            y:*mut f32,m:i32,n:i32,k:i32,ldy:i32,rounded:i32,stages:i32)->i32;}
        let (n,k)=(self.m8.size()[0],self.m8.size()[1]);let m=xh.size()[0];
        assert_eq!(xh.kind(),Kind::Half);assert!(xh.is_contiguous());assert_eq!(xh.size()[1],k);
        assert_eq!(out.kind(),Kind::Float);assert_eq!(out.stride()[1],1);assert_eq!(out.size(),[m,n]);
        assert_eq!(unsafe{rs_c12_big(xh.data_ptr(),self.m8.data_ptr().cast(),self.e4.data_ptr().cast(),self.eb.data_ptr().cast(),self.ptr.data_ptr().cast(),
            self.col.data_ptr().cast(),self.val.data_ptr().cast(),out.data_ptr().cast(),m as i32,n as i32,k as i32,out.stride()[0] as i32,i32::from(rounded),big_stages())},0,"C12 big GEMM");
    }
}
fn big_shape_ok(n:i64,k:i64)->bool {n%128==0 && k%128==0}
/// mm16 (partial=false: Half-rounded) / mm16_partial (partial=true) for prefill-sized rows.
pub(crate) fn try_big(x:&Tensor,w:&Tensor,partial:bool,half:Option<&Tensor>)->Option<Tensor> {
    if !big_rows(x) || x.size()[1]!=w.size()[1] {return None;}
    CACHE.with(|c|{let c=c.borrow();let e=c.get(&(w.data_ptr() as usize))?;
        if e.source.size()!=w.size() || e.source.stride()!=w.stride() || !big_shape_ok(w.size()[0],w.size()[1]) {return None;}
        let xh=half.map(Tensor::shallow_clone).unwrap_or_else(||crate::weights::half_input_pub(x)).contiguous();
        let y=Tensor::empty([x.size()[0],w.size()[0]],(Kind::Float,x.device()));e.big_into(&xh,&y,!partial);Some(y)})
}
/// cat(mm16(x,w_i)) for a coded group (P1a-Half's KDA q/k/v): one launch over the group's rows.
pub(crate) fn try_big_cat(x:&Tensor,ws:&[&Tensor])->Option<Tensor> {
    if !big_rows(x) {return None;}
    let keys:Vec<usize>=ws.iter().map(|w|w.data_ptr() as usize).collect();
    if !CACHE.with(|c|{let c=c.borrow();ws.iter().all(|w|c.get(&(w.data_ptr() as usize)).is_some_and(|e|e.source.size()==w.size()&&e.source.stride()==w.stride()))}) {return None;}
    GROUPS.with(|g|{let g=g.borrow();let e=g.get(&keys)?;
        if !big_shape_ok(e.m8.size()[0],e.m8.size()[1]) || x.size()[1]!=e.m8.size()[1] {return None;}
        let xh=crate::weights::half_input_pub(x).contiguous();
        let y=Tensor::empty([x.size()[0],e.m8.size()[0]],(Kind::Float,x.device()));e.big_into(&xh,&y,true);Some(y)})
}
/// AR prefetch (GLM53_AR_PREFETCH): the device regions a decode call on w reads first, up to `budget` bytes: the leading
/// rows of its coded arrays (C12: m8, e4, eb; Q8: q, s), in proportion. Empty when w is not coded.
pub(crate) fn prefetch_regions(w:&Tensor,budget:i64)->Vec<(usize,i64)> {
    CACHE.with(|c|{let c=c.borrow();let Some(e)=c.get(&(w.data_ptr() as usize)) else {return Vec::new()};
        if e.source.size()!=w.size() {return Vec::new();}
        let (n,k)=(e.m8.size()[0],e.m8.size()[1]);
        let mut parts:Vec<(usize,i64)>=vec![(e.m8.data_ptr() as usize,k)];
        if let Some(sm)=&e.q4 {parts.push((sm.data_ptr() as usize,(k*2/64)*4));}
        else if let Some(s)=&e.s {parts.push((s.data_ptr() as usize,(k/128)*4));}
        else {parts.push((e.e4.data_ptr() as usize,k/2));parts.push((e.eb.data_ptr() as usize,1));}
        let per_row:i64=parts.iter().map(|p|p.1).sum();let rows=(budget/per_row.max(1)).clamp(0,n);
        parts.into_iter().map(|(p,b)|(p,b*rows)).filter(|r|r.1>0).collect()})
}
/// A coded weight (source or released placeholder): lets metadata checks accept placeholders.
pub(crate) fn is_coded(w:&Tensor)->bool {
    enabled() && CACHE.with(|c|c.borrow().get(&(w.data_ptr() as usize)).is_some_and(|e|e.source.size()==w.size()&&e.source.stride()==w.stride()))
}
thread_local!{static REMAP:RefCell<HashMap<usize,usize>>=RefCell::new(HashMap::new());}
pub(crate) fn free_sources_enabled()->bool {prefill_enabled() && std::env::var("GLM53_C12_FREE_SOURCE").as_deref()==Ok("1")}
/// Replace a coded weight by a stride-0 placeholder (the cache entry is re-keyed to it); returns the placeholder.
pub(crate) fn compact(w:&Tensor)->Option<Tensor> {
    if !free_sources_enabled() {return None;}
    let old=w.data_ptr() as usize;
    CACHE.with(|c|{let mut c=c.borrow_mut();
        let e=c.get(&old)?;if e.source.size()!=w.size() || e.source.stride()!=w.stride() || !big_shape_ok(w.size()[0],w.size()[1]) {return None;}
        let mut e=c.remove(&old).unwrap();
        let ph=Tensor::zeros([1],(w.kind(),w.device())).expand(w.size().as_slice(),false);
        e.source=ph.shallow_clone();c.insert(ph.data_ptr() as usize,e);
        REMAP.with(|r|r.borrow_mut().insert(old,ph.data_ptr() as usize));Some(ph)})
}
/// After compaction: group keys follow their members' placeholders.
pub(crate) fn finish_compact() {
    let remap=REMAP.with(|r|std::mem::take(&mut *r.borrow_mut()));if remap.is_empty() {return;}
    GROUPS.with(|g|{let mut g=g.borrow_mut();let old=std::mem::take(&mut *g);
        for (k,v) in old {let nk:Vec<usize>=k.iter().map(|p|*remap.get(p).unwrap_or(p)).collect();g.insert(nk,v);}});
    extern "C"{fn rs_empty_cache();}unsafe{rs_empty_cache();}
}
