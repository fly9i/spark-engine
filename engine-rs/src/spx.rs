//! SPX1 non-expert weight format (v1): per-tensor detection, load-time repack, decode GEMV, prefill dequant + cuBLAS.
//!
//! On disk, for a checkpoint linear N ([out, k] BF16 in the original):
//!   N.spx_q    U8  [out, ceil(k*b/8)]  per row, the k codes c in [0, 2^b-1] as a little-endian bitstream
//!   N.spx_s    F16 [out, k/64]         per-group scale
//!   N.spx_sign I8  [k]                 random signs (+1/-1)
//!   file metadata "spx1" = JSON {N: {"bits": b, "group": 64, "rot_block": B, "version": 1}}
//! W_rot[o,i] = (c[o,i] - (2^b-1)/2) * s[o,i/64]; y = W_rot . x_rot, x_rot = blockdiag(H_B)(sign (.) x), H_B the orthonormal
//! Sylvester Hadamard. A tensor is SPX when N.spx_q is in the shard index (later files override earlier ones) and
//! GLM53_SPX is not 0; otherwise the existing path loads N unchanged (old checkpoints never reach this module).
//!
//! Resident form: only the tiled codes + scales (shim/spx.cu layout, (256b+64)/2048 bytes per weight) and the int8 sign
//! vector. The model field holds a stride-0 Half placeholder of the local [n, k] shape, the key of this module's
//! registry (any non-SPX consumer trips the existing "FP8-freed weight" placeholder asserts instead of reading it).
//! - rows 1..=GLM53_SPX_MAX_ROWS (default 32): rotate x once (spx_rot, FP32 FWHT -> Half), then spx_gemv over the codes
//!   (the odd integers 2c-(2^b-1) as exact Half MMA operands, FP32 accumulate per 64-k group, scaled by s/2 in FP32). Row-concatenated producers sharing one input (KDA q/k/v, dense and shared gate/up) are
//!   registered as one group: one rotation, one launch.
//! - more rows (prefill): rotate x, dequantize W_rot to Half into a transient scratch (<= GLM53_SPX_SCRATCH_MIB, default
//!   256, per piece) and run the existing cuBLAS Half GEMM (no resident Half copy). Next step (not done): an INT8/FP8
//!   activation MMA reading the codes directly (see `prefill`).
//! - kv_b (only used in absorbed form) and the drafter's copy of the LM head are reconstructed at load as
//!   BF16(W_rot R^T) in Half (`reconstruct_*`), the same values a fake-quant BF16 checkpoint would hold.
use std::{cell::RefCell,collections::HashMap,sync::{OnceLock,atomic::{AtomicBool,Ordering}}};
use tch::{Device,Kind,Tensor};
use crate::safetensors::ShardIndex;

extern "C" {
    fn rs_spx_gemv(x:*const std::ffi::c_void,t:*const std::ffi::c_void,n:i32,k:i32,bits:i32,y:*mut std::ffi::c_void,m:i32,out:i32,ks:i32)->i32;
    fn rs_spx_dequant(t:*const std::ffi::c_void,n:i32,k:i32,bits:i32,w:*mut std::ffi::c_void,ldw:i64,f32:i32)->i32;
    fn rs_spx_pack(src:*const std::ffi::c_void,rs:i64,k0:i64,sc:*const std::ffi::c_void,n:i32,k:i32,bits:i32,out:*mut std::ffi::c_void)->i32;
    fn rs_spx_rot(x:*const std::ffi::c_void,ldx:i64,in_kind:i32,sign:*const std::ffi::c_void,m:i32,k:i32,bk:i32,y:*mut std::ffi::c_void,ldy:i64,out_kind:i32,sign_first:i32)->i32;
    fn rs_mm16_f32(x:*const u8,w:*const u8,y:*mut u8,m:i32,n:i32,k:i32)->i32;
    fn rs_spx_kda_onorm_rot(o:*const std::ffi::c_void,on:*const std::ffi::c_void,onl:i32,sg2:*const std::ffi::c_void,out:*mut std::ffi::c_void,rows:i32,heads:i32,sign:*const std::ffi::c_void,bk:i32,xr:*mut std::ffi::c_void)->i32;
    fn rs_spx_swiglu_rot(y:*const std::ffi::c_void,m:i32,n:i32,limit:f32,sign:*const std::ffi::c_void,bk:i32,act:*mut std::ffi::c_void,xr:*mut std::ffi::c_void)->i32;
    fn rs_spx_rot8(x:*const std::ffi::c_void,ldx:i64,in_kind:i32,sign:*const std::ffi::c_void,m:i32,k:i32,bk:i32,q:*mut std::ffi::c_void,sx:*mut std::ffi::c_void)->i32;
    fn rs_spx_quant8(x:*const std::ffi::c_void,m:i32,k:i32,q:*mut std::ffi::c_void,sx:*mut std::ffi::c_void,sumx:*mut std::ffi::c_void)->i32;
    fn rs_spx_mm8(q:*const std::ffi::c_void,sx:*const std::ffi::c_void,sumx:*const std::ffi::c_void,t:*const std::ffi::c_void,m:i32,n:i32,k:i32,bits:i32,y:*mut std::ffi::c_void,ldy:i64,out:i32)->i32;
    fn rs_empty_cache();
}

/// GLM53_SPX=0: ignore every .spx_* tensor (the BF16 originals must then be in the checkpoint).
pub(crate) fn enabled()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||std::env::var("GLM53_SPX").as_deref()!=Ok("0"))}
fn max_rows()->i64 {static M:OnceLock<i64>=OnceLock::new();*M.get_or_init(||std::env::var("GLM53_SPX_MAX_ROWS").ok().map(|v|v.parse().expect("GLM53_SPX_MAX_ROWS")).unwrap_or(32).clamp(0,32))}
/// D7 (GLM53_SPX_MM8, default off): prefill rows (> max_rows) of b <= 6 SPX weights on INT8 tensor cores (shim/spx_mm8.cuh):
/// x_rot quantized per token to INT8, codes read from the tiles, no Half dequant scratch. Value "1": every class; else a
/// comma list of classes (attn, mla, dense, shared, head).
fn mm8_on(class:&str,bits:i64)->bool {
    static C:OnceLock<Option<Vec<String>>>=OnceLock::new();
    let c=C.get_or_init(||std::env::var("GLM53_SPX_MM8").ok().filter(|v|!v.is_empty()&&v!="0").map(|v|v.split(',').map(str::to_string).collect()));
    bits<=6 && c.as_ref().is_some_and(|l|l.iter().any(|x|x=="1"||x==class||class=="test"))
}
/// Per-token INT8 quantization of x_rot (Half [m, k]): (x_q int8 [m, k], sx FP32 [m], group sums int32 [m, k/64]).
fn quant8(xr:&Tensor)->(Tensor,Tensor,Tensor) {
    let (m,k)=(xr.size()[0],xr.size()[1]);let dev=xr.device();
    let q=Tensor::empty([m,k],(Kind::Int8,dev));let sx=Tensor::empty([m],(Kind::Float,dev));let sum=Tensor::empty([m,k/64],(Kind::Int,dev));
    assert_eq!(unsafe{rs_spx_quant8(xr.data_ptr(),m as i32,k as i32,q.data_ptr(),sx.data_ptr(),sum.data_ptr())},0,"SPX quant8");
    (q,sx,sum)
}
/// D7: x [m, k] (FP32 or Half, unit column stride) -> (x_rot quantized per row to INT8, row scales), fused (spx8::rot8).
fn rot8(x:&Tensor,sign:&Tensor,rot:i64)->(Tensor,Tensor) {
    let x=if x.stride()[1]==1 && matches!(x.kind(),Kind::Float|Kind::Half) {x.shallow_clone()} else {x.to_kind(Kind::Float).contiguous()};
    let (m,k)=(x.size()[0],x.size()[1]);let dev=x.device();
    let q=Tensor::empty([m,k],(Kind::Int8,dev));let sx=Tensor::empty([m],(Kind::Float,dev));
    assert_eq!(unsafe{rs_spx_rot8(x.data_ptr(),x.stride()[0],i32::from(x.kind()==Kind::Half),sign.data_ptr(),m as i32,k as i32,rot as i32,q.data_ptr(),sx.data_ptr())},0,"SPX rot8");
    (q,sx)
}
thread_local!{
    /// One pre-rotated input: (input data pointer, its shape, the consuming weight's sign pointer, x_rot Half). Written
    /// by `swiglu_rot` right before its consumer runs; consumed (taken) by the next try_mm / try_mm_into of a weight with
    /// that sign on that exact input, and dropped by any later write, so a reused address can never hit.
    static PREROT:RefCell<Option<(usize,Vec<i64>,usize,Tensor)>>=const{RefCell::new(None)};
}
thread_local!{
    /// O5 step 3: the last fused MHC pre output (z pointer, shape, its ln weight pointer), and per ln weight the sign and
    /// rotation block of the SPX weight that rotated that z (learned the first time it does, from the separate rotation).
    static MHC_Z:RefCell<Option<(usize,Vec<i64>,usize)>>=const{RefCell::new(None)};
    static MHC_ROT:RefCell<HashMap<usize,(Tensor,i64)>>=RefCell::new(HashMap::new());
}
fn mhc_rot_on()->bool {static E:OnceLock<bool>=OnceLock::new();*E.get_or_init(||std::env::var("GLM53_SPX_MHC_ROT").as_deref()==Ok("1"))}
/// GLM53_SPX_MHC_ROT=1 (L0): the fused MHC pre (site weight `ln`, t decode rows of width h) also writes the rotated
/// input of the SPX projection known to read its z: Some((sign, block, x_rot Half [t, h] to fill)).
pub(crate) fn mhc_rot_plan(ln:&Tensor,t:i64,h:i64)->Option<(Tensor,i64,Tensor)> {
    if !any() || !mhc_rot_on() || t<1 || t>max_rows() || crate::ablate::clamp().is_some() {return None;}
    let (sign,bk)=MHC_ROT.with(|m|m.borrow().get(&(ln.data_ptr() as usize)).map(|(s,b)|(s.shallow_clone(),*b)))?;
    if h%256!=0 || h>8192 || bk<8 || bk>h || bk&(bk-1)!=0 || sign.size()!=[h] {return None;}
    Some((sign,bk,Tensor::empty([t,h],(Kind::Half,ln.device()))))
}
/// After the fused MHC pre launch: remember z for learning, and publish the rotated input for its consumer.
pub(crate) fn mhc_rot_done(z:&Tensor,ln:&Tensor,rot:Option<(Tensor,i64,Tensor)>) {
    if !any() || !mhc_rot_on() {return;}
    MHC_Z.with(|m|*m.borrow_mut()=Some((z.data_ptr() as usize,z.size(),ln.data_ptr() as usize)));
    if let Some((sign,_,xr))=rot {PREROT.with(|p|*p.borrow_mut()=Some((z.data_ptr() as usize,z.size(),sign.data_ptr() as usize,xr)));}
}
/// x is about to be rotated for a decode-row SPX weight (sign, block): when x is the last MHC pre output, record the pair
/// for that MHC site (first consumer wins).
fn learn_mhc(x:&Tensor,sign:&Tensor,rot:i64) {
    if !mhc_rot_on() || x.size()[0]>max_rows() {return;}
    let site=MHC_Z.with(|m|m.borrow().as_ref().and_then(|(zp,shape,lp)|(*zp==x.data_ptr() as usize && *shape==x.size()).then_some(*lp)));
    if let Some(lp)=site {MHC_ROT.with(|m|{let mut m=m.borrow_mut();
        if !m.contains_key(&lp) {m.insert(lp,(sign.shallow_clone(),rot));if m.len()==1 || m.len()%32==0 {eprintln!("[spx] MHC rotation sites learned: {} (block {rot})",m.len());}}});}
}
/// Pre-rotated input when published for exactly this input and sign, else the rotation (learning MHC sites on the way).
fn rotated(x:&Tensor,sign:&Tensor,rot:i64)->Tensor {
    take_prerot(x,sign).unwrap_or_else(||{learn_mhc(x,sign,rot);rotate(x,sign,rot)})
}
thread_local!{
    /// SPX projections writing into the residual stream of a layer SPARK_ABLATE covers (o_proj / down_proj): key -> layer.
    static ABLATE_OUT:RefCell<HashMap<usize,usize>>=RefCell::new(HashMap::new());
}
/// SPARK_ABLATE (crate::ablate) on an SPX residual writer `w` of `layer`: every output of `w` (decode GEMV, prefill
/// INT8 / dequant paths, Half or FP32) loses the ablated directions at run time, as the routed experts' outputs do.
pub(crate) fn mark_residual_writer(w:&Tensor,layer:usize) {ABLATE_OUT.with(|m|{m.borrow_mut().insert(w.data_ptr() as usize,layer);});}
thread_local!{static OPROJ_OUT:RefCell<HashMap<usize,usize>>=RefCell::new(HashMap::new());}
/// SPARK_ABLATE_OPROJ on an SPX attention o_proj of `layer`: its outputs get the per-layer edit (before SPARK_ABLATE, the
/// order of the dense weights' load-time edits).
pub(crate) fn mark_oproj_writer(w:&Tensor,layer:usize) {OPROJ_OUT.with(|m|{m.borrow_mut().insert(w.data_ptr() as usize,layer);});}
fn ablate_output(w:&Tensor,y:&Tensor) {
    let key=w.data_ptr() as usize;
    let edit=OPROJ_OUT.with(|m|m.borrow().get(&key).copied());let abl=ABLATE_OUT.with(|m|m.borrow().get(&key).copied());
    if edit.is_none() && abl.is_none() {return;}
    let run=|f:&Tensor|{if let Some(l)=edit {crate::ablate::oproj().unwrap().apply(l,f);} if let Some(l)=abl {crate::ablate::apply_marked(l,f);}};
    if y.kind()==Kind::Float {run(y);return;}
    let f=y.to_kind(Kind::Float);run(&f);y.shallow_clone().copy_(&f);
}
fn take_prerot(x:&Tensor,sign:&Tensor)->Option<Tensor> {
    PREROT.with(|p|{let mut p=p.borrow_mut();
        let hit=p.as_ref().is_some_and(|(xp,shape,sp,_)|*xp==x.data_ptr() as usize && *shape==x.size() && *sp==sign.data_ptr() as usize);
        if hit {p.take().map(|e|e.3)} else {None}})
}
/// Shared-expert clamp-SwiGLU of the SPX gate/up group output y [m, 2n] (FP32) for an SPX down projection `w_down`:
/// one kernel writes the activation (returned, the same values as clamp/silu/mul) and the down projection's rotated
/// input, which its GEMV then reuses (one launch instead of five). None (caller keeps the ATen path) unless w_down is
/// SPX and m is a decode row count (prefill rows go to the INT8 / dequant paths, which rotate themselves).
pub(crate) fn swiglu_rot(y:&Tensor,n:i64,w_down:&Tensor,limit:f64)->Option<Tensor> {
    if !any() || std::env::var("GLM53_SPX_SWIGLU_ROT").as_deref()==Ok("0") {return None;}
    let m=y.size()[0];
    if m==0 || m>max_rows() || y.kind()!=Kind::Float || !y.is_contiguous() || y.size()[1]!=2*n {return None;}
    CACHE.with(|c|{let c=c.borrow();let e=c.get(&(w_down.data_ptr() as usize))?;
        if e.key.size()!=w_down.size() || e.k!=n || n%256!=0 || e.rot>n {return None;}
        let act=Tensor::empty([m,n],(Kind::Float,y.device()));let xr=Tensor::empty([m,n],(Kind::Half,y.device()));
        assert_eq!(unsafe{rs_spx_swiglu_rot(y.data_ptr(),m as i32,n as i32,limit as f32,e.sign.data_ptr(),e.rot as i32,act.data_ptr(),xr.data_ptr())},0,"SPX swiglu_rot");
        PREROT.with(|p|*p.borrow_mut()=Some((act.data_ptr() as usize,act.size(),e.sign.data_ptr() as usize,xr)));
        Some(act)})
}
/// KDA output norm + gate (kda_onorm_gate: o, sg2 FP32 [t, heads, 128] contiguous) for an SPX o_proj `wo`: one kernel
/// writes the hidden (same values) and wo's rotated input for the next try_mm of hidden.reshape([t, heads * 128]).
/// None (caller keeps its kernel) unless wo is SPX over exactly this width and t is a decode row count.
pub(crate) fn kda_onorm_rot(o:&Tensor,onorm:&Tensor,sg2:&Tensor,heads:i64,wo:&Tensor)->Option<Tensor> {
    if !any() || std::env::var("GLM53_SPX_ONORM_ROT").as_deref()==Ok("0") {return None;}
    let t=o.size()[0];
    if t==0 || t>max_rows() || heads>32 || !o.is_contiguous() || !sg2.is_contiguous() || o.kind()!=Kind::Float || sg2.kind()!=Kind::Float {return None;}
    CACHE.with(|c|{let c=c.borrow();let e=c.get(&(wo.data_ptr() as usize))?;
        if e.key.size()!=wo.size() || e.k!=heads*128 || e.rot>heads*128 {return None;}
        let out=Tensor::empty([t,heads,128],(Kind::Float,o.device()));let xr=Tensor::empty([t,heads*128],(Kind::Half,o.device()));
        assert_eq!(unsafe{rs_spx_kda_onorm_rot(o.data_ptr(),onorm.data_ptr(),onorm.numel() as i32,sg2.data_ptr(),out.data_ptr(),t as i32,heads as i32,e.sign.data_ptr(),e.rot as i32,xr.data_ptr())},0,"SPX kda_onorm_rot");
        PREROT.with(|p|*p.borrow_mut()=Some((out.data_ptr() as usize,vec![t,heads*128],e.sign.data_ptr() as usize,xr)));
        Some(out)})
}
fn scratch_bytes()->i64 {static M:OnceLock<i64>=OnceLock::new();*M.get_or_init(||std::env::var("GLM53_SPX_SCRATCH_MIB").ok().map(|v|v.parse().expect("GLM53_SPX_SCRATCH_MIB")).unwrap_or(256i64)<<20)}
/// Set once anything is registered: every hook is a single relaxed load for checkpoints without SPX tensors.
static ANY:AtomicBool=AtomicBool::new(false);
#[inline] fn any()->bool {ANY.load(Ordering::Relaxed)}

pub(crate) fn tile_bytes(bits:i64)->i64 {256*bits+64}

/// One SPX weight (or a row-concatenated group): tiles [n/16 * k/128 * TB] u8, sign int8 [k] on the device.
struct Spx {key:Tensor,tiles:Tensor,sign:Tensor,n:i64,k:i64,bits:i64,rot:i64,class:&'static str}
impl Spx {
    fn view(&self,key:&Tensor,r0:i64,n:i64)->Spx {
        let per=self.k/128*tile_bytes(self.bits);
        Spx{key:key.shallow_clone(),tiles:self.tiles.narrow(0,r0/16*per,n/16*per),sign:self.sign.shallow_clone(),n,k:self.k,bits:self.bits,rot:self.rot,class:self.class}
    }
    fn same_input(&self,o:&Spx)->bool {
        self.k==o.k && self.rot==o.rot && (self.sign.data_ptr()==o.sign.data_ptr() || self.sign.equal(&o.sign))
    }
}
thread_local!{
    static TEST_STEP:std::cell::Cell<i64>=const{std::cell::Cell::new(0)};   // selftest: force multi-piece prefill
    static CACHE:RefCell<HashMap<usize,Spx>>=RefCell::new(HashMap::new());
    static GROUPS:RefCell<HashMap<Vec<usize>,Spx>>=RefCell::new(HashMap::new());
    static META:RefCell<HashMap<String,serde_json::Map<String,serde_json::Value>>>=RefCell::new(HashMap::new());
    static STATS:RefCell<std::collections::BTreeMap<String,(i64,i64,i64)>>=RefCell::new(Default::default());   // class -> (weights, tile bytes, tensors)
}

/// How a TP rank holds its part: Rows = output rows split (column-parallel), Cols = input k split (row-parallel), as a
/// crate::tp::Part of the dimension.
#[derive(Clone,Copy,Debug)]
pub(crate) enum Split {Full,Rows(crate::tp::Part),Cols(crate::tp::Part)}

/// N has SPX tensors in this index (and SPX is enabled).
pub(crate) fn present(idx:&ShardIndex,name:&str)->bool {enabled() && idx.entries.contains_key(&format!("{name}.spx_q"))}
/// N's BF16 original is shadowed by SPX tensors (the load never reads it; excluded from the page-cache warm-up).
pub(crate) fn shadowed(idx:&ShardIndex,name:&str)->bool {enabled() && !name.contains(".spx_") && idx.entries.contains_key(&format!("{name}.spx_q"))}

struct Meta {bits:i64,rot:i64,out:i64,k:i64}
fn meta(idx:&ShardIndex,name:&str)->Meta {
    let q=idx.entries.get(&format!("{name}.spx_q")).unwrap_or_else(||panic!("SPX: {name}.spx_q missing"));
    let file=q.file.clone();
    let m=META.with(|c|c.borrow_mut().entry(file.clone()).or_insert_with(||idx.file_metadata(&file).unwrap_or_else(|e|panic!("SPX: metadata of {file}: {e}"))).clone());
    let spec=m.get("spx1").and_then(|v|v.as_str()).unwrap_or_else(||panic!("SPX: {file} has {name}.spx_q but no \"spx1\" metadata"));
    let spec:serde_json::Value=serde_json::from_str(spec).unwrap_or_else(|e|panic!("SPX: {file} \"spx1\" metadata: {e}"));
    let e=spec.get(name).unwrap_or_else(||panic!("SPX: {file} \"spx1\" metadata has no entry for {name}"));
    let get=|k:&str|e.get(k).and_then(|v|v.as_i64()).unwrap_or_else(||panic!("SPX: {name} metadata lacks {k}"));
    let (bits,group,rot,version)=(get("bits"),get("group"),get("rot_block"),get("version"));
    assert_eq!(version,1,"SPX: {name} version {version} (this engine reads v1)");
    assert_eq!(group,64,"SPX: {name} group {group} (v1 is 64)");
    assert!([4,5,6,8].contains(&bits),"SPX: {name} bits {bits}");
    assert!(rot>=8 && rot<=8192 && rot&(rot-1)==0,"SPX: {name} rot_block {rot}");
    assert_eq!(q.dtype,"U8","SPX: {name}.spx_q dtype");assert_eq!(q.shape.len(),2,"SPX: {name}.spx_q rank");
    let (out,rb)=(q.shape[0] as i64,q.shape[1] as i64);
    let sg=idx.entries.get(&format!("{name}.spx_sign")).unwrap_or_else(||panic!("SPX: {name}.spx_sign missing"));
    assert_eq!(sg.dtype,"I8","SPX: {name}.spx_sign dtype");assert_eq!(sg.shape.len(),1);
    let k=sg.shape[0] as i64;
    assert_eq!(rb,(k*bits+7)/8,"SPX: {name}.spx_q row bytes for k {k} at {bits} bits");
    let s=idx.entries.get(&format!("{name}.spx_s")).unwrap_or_else(||panic!("SPX: {name}.spx_s missing"));
    assert_eq!(s.dtype,"F16","SPX: {name}.spx_s dtype");assert_eq!(s.shape,vec![out as usize,(k/64) as usize],"SPX: {name}.spx_s shape");
    if let Some(o)=idx.entries.get(name) {assert_eq!(o.shape,vec![out as usize,k as usize],"SPX: {name} shape differs from its original");}
    Meta{bits,rot,out,k}
}

/// (out, k) of an SPX tensor and its rotation block (the input-dimension unit a TP split must not cut).
pub(crate) fn shape_rot(idx:&ShardIndex,name:&str)->(i64,i64,i64) {let m=meta(idx,name);(m.out,m.k,m.rot)}

/// A tensor's bytes [off, off+n) on the device (u8). Staged like weights::upload_stored.
fn bytes(idx:&ShardIndex,name:&str,off:usize,n:usize,dev:Device)->Tensor {crate::weights::upload_bytes(idx,name,off,n,dev)}

/// Codes of rows [r0, r0+rows) and input columns [k0, k0+kl) (k0 % 64 == 0) packed into tiles. `codes`: u8 [rows, rb] (the
/// full on-disk rows), `scales`: Half [rows, K/64] (full rows).
pub(crate) fn pack(codes:&Tensor,scales:&Tensor,bits:i64,k0:i64,kl:i64)->Tensor {
    let (rows,rb)=(codes.size()[0],codes.size()[1]);
    assert!(rows%16==0 && kl%128==0 && k0%64==0,"SPX pack: rows {rows} k {kl} k0 {k0} (need rows%16, k%128, k0%64)");
    let sc=scales.narrow(1,k0/64,kl/64).contiguous();assert_eq!(sc.kind(),Kind::Half);
    let codes=codes.contiguous();
    let tiles=Tensor::empty([rows/16*(kl/128)*tile_bytes(bits)],(Kind::Uint8,codes.device()));
    assert_eq!(unsafe{rs_spx_pack(codes.data_ptr(),rb,k0,sc.data_ptr(),rows as i32,kl as i32,bits as i32,tiles.data_ptr())},0,"SPX pack");
    tiles
}

fn class_of(name:&str)->&'static str {
    if name=="lm_head.weight" {"head"} else if name.contains(".shared_experts.") {"shared"} else if name.contains(".mlp.") {"dense"}
    else if name.contains("q_b_proj")||name.contains("kv_b_proj") {"mla"} else {"attn"}
}

/// Load N (SPX) for this rank: returns the stride-0 Half placeholder [n_local, k_local] registered here.
pub(crate) fn load(idx:&ShardIndex,name:&str,split:Split,dev:Device)->Tensor {
    let e=load_entry(idx,name,split,dev);
    let key=Tensor::zeros([1],(Kind::Half,dev)).expand([e.n,e.k],false);
    let e=Spx{key:key.shallow_clone(),..e};
    STATS.with(|s|{let mut s=s.borrow_mut();let v=s.entry(e.class.to_string()).or_insert((0,0,0));v.0+=e.n*e.k;v.1+=e.tiles.numel() as i64+e.sign.numel() as i64;v.2+=1;});
    CACHE.with(|c|c.borrow_mut().insert(key.data_ptr() as usize,e));
    ANY.store(true,Ordering::Relaxed);
    key
}
fn load_entry(idx:&ShardIndex,name:&str,split:Split,dev:Device)->Spx {
    assert!(dev.is_cuda(),"SPX tensors load on CUDA only");
    let m=meta(idx,name);let rb=(m.k*m.bits+7)/8;
    let (r0,rows,k0,kl)=match split {
        Split::Full=>(0,m.out,0,m.k),
        Split::Rows(p)=>{let (r0,n)=p.range(m.out);(r0,n,0,m.k)}
        Split::Cols(p)=>{let (k0,n)=p.range(m.k);(0,m.out,k0,n)}
    };
    assert!(kl%m.rot==0 && k0%m.rot==0,"SPX: {name} rot_block {} does not divide this rank's input slice [{k0}, {}) ({split:?})",m.rot,k0+kl);
    let codes=bytes(idx,&format!("{name}.spx_q"),(r0*rb) as usize,(rows*rb) as usize,dev).view([rows,rb]);
    let scales=bytes(idx,&format!("{name}.spx_s"),(r0*(m.k/64)*2) as usize,(rows*(m.k/64)*2) as usize,dev).view_dtype(Kind::Half).view([rows,m.k/64]);
    let sign=bytes(idx,&format!("{name}.spx_sign"),0,m.k as usize,dev).view_dtype(Kind::Int8).narrow(0,k0,kl).contiguous();
    assert_eq!(sign.abs().ne(1).sum(Kind::Int64).int64_value(&[]),0,"SPX: {name}.spx_sign holds values other than +1/-1");
    let tiles=pack(&codes,&scales,m.bits,k0,kl);
    for t in ["spx_q","spx_s","spx_sign"] {crate::weights::consumed(idx,&format!("{name}.{t}"));}
    Spx{key:Tensor::zeros([1],(Kind::Half,dev)),tiles,sign,n:rows,k:kl,bits:m.bits,rot:m.rot,class:class_of(name)}
}

/// kv_b (absorbed form only) and similar: BF16(W_rot R^T) as a Half [n, k] tensor (Split as `load`).
pub(crate) fn load_reconstructed(idx:&ShardIndex,name:&str,split:Split,dev:Device)->Tensor {
    let e=load_entry(idx,name,split,dev);let w=reconstruct_entry(&e);
    STATS.with(|s|{let mut s=s.borrow_mut();let v=s.entry("reconstructed (Half resident)".to_string()).or_insert((0,0,0));v.0+=e.n*e.k;v.1+=e.n*e.k*2;v.2+=1;});
    w
}
/// BF16(W_rot R^T) of a registered weight as Half [n, k] (the drafter's LM head); None when w is not SPX.
pub(crate) fn reconstruct(w:&Tensor)->Option<Tensor> {
    if !any() {return None;}
    CACHE.with(|c|{let c=c.borrow();let e=c.get(&(w.data_ptr() as usize))?;if e.key.size()!=w.size() {return None;}Some(reconstruct_entry(e))})
}
fn reconstruct_entry(e:&Spx)->Tensor {
    let dev=e.tiles.device();let out=Tensor::empty([e.n,e.k],(Kind::Half,dev));
    let step=((64i64<<20)/(e.k*4)/16*16).max(16);let per=e.k/128*tile_bytes(e.bits);
    let mut r0=0;
    while r0<e.n {
        let n=step.min(e.n-r0);
        let wr=Tensor::empty([n,e.k],(Kind::Float,dev));
        assert_eq!(unsafe{rs_spx_dequant(e.tiles.narrow(0,r0/16*per,n/16*per).data_ptr(),n as i32,e.k as i32,e.bits as i32,wr.data_ptr(),e.k,1)},0,"SPX dequant");
        let w=Tensor::empty([n,e.k],(Kind::Float,dev));
        assert_eq!(unsafe{rs_spx_rot(wr.data_ptr(),e.k,0,e.sign.data_ptr(),n as i32,e.k as i32,e.rot as i32,w.data_ptr(),e.k,0,0)},0,"SPX unrotate");
        out.narrow(0,r0,n).copy_(&w.to_kind(Kind::BFloat16));
        r0+=n;
    }
    out
}

/// Row-concatenated producers sharing one input: one tile storage (members become row views), one rotation and launch.
pub(crate) fn group(ws:&[&Tensor]) {
    if !any() {return;}
    let keys:Vec<usize>=ws.iter().map(|w|w.data_ptr() as usize).collect();
    let ok=CACHE.with(|c|{let c=c.borrow();
        let es:Vec<&Spx>=keys.iter().filter_map(|k|c.get(k)).collect();
        es.len()==ws.len() && es.iter().all(|e|e.bits==es[0].bits && e.same_input(es[0]))});
    if !ok || GROUPS.with(|g|g.borrow().contains_key(&keys)) {return;}
    let g=CACHE.with(|c|{let c=c.borrow();let es:Vec<&Spx>=keys.iter().map(|k|&c[k]).collect();
        let tiles=Tensor::cat(&es.iter().map(|e|e.tiles.shallow_clone()).collect::<Vec<_>>(),0);
        Spx{key:Tensor::zeros([1],(Kind::Half,tiles.device())),tiles,sign:es[0].sign.shallow_clone(),n:es.iter().map(|e|e.n).sum(),k:es[0].k,bits:es[0].bits,rot:es[0].rot,class:es[0].class}});
    let mut r0=0;
    for (w,k) in ws.iter().zip(&keys) {let n=w.size()[0];let v=g.view(w,r0,n);CACHE.with(|c|c.borrow_mut().insert(*k,v));r0+=n;}
    GROUPS.with(|m|m.borrow_mut().insert(keys,g));
}
pub(crate) fn register_groups(layers:&[crate::weights::LayerWeights]) {
    if !any() {return;}
    for l in layers {
        if let Some(w)=&l.kda {group(&[&w.wq,&w.wk,&w.wv]);}
        if let Some(w)=&l.dense {group(&[&w.wg,&w.wu]);}
        if let Some(w)=&l.moe {group(&[&w.sh_wg,&w.sh_wu]);}
    }
    tch::Cuda::synchronize(0);unsafe{rs_empty_cache();}
}
pub(crate) fn is_spx(w:&Tensor)->bool {any() && CACHE.with(|c|c.borrow().get(&(w.data_ptr() as usize)).is_some_and(|e|e.key.size()==w.size()&&e.key.stride()==w.stride()))}

/// Resident bytes of the SPX copies (tiles + signs), by class; printed once after the load.
pub(crate) fn report() {
    if !any() {return;}
    let mut total=(0i64,0i64);
    STATS.with(|s|for (class,(w,b,n)) in s.borrow().iter() {
        eprintln!("[spx] {class}: {n} tensors, {:.3} G weights (local), {:.3} GiB resident ({:.2} bits/weight)",*w as f64/1e9,*b as f64/(1u64<<30) as f64,*b as f64*8./(*w).max(1) as f64);
        total.0+=w;total.1+=b;});
    eprintln!("[spx] total {:.3} G weights in {:.3} GiB; groups {}, max rows {} (GLM53_SPX_MAX_ROWS), prefill scratch <= {} MiB",
        total.0 as f64/1e9,total.1 as f64/(1u64<<30) as f64,GROUPS.with(|g|g.borrow().len()),max_rows(),scratch_bytes()>>20);
}

// ---------------------------------------------------------------- execution

fn split_k(n:i64,k:i64)->i32 {
    static KS:OnceLock<i32>=OnceLock::new();
    let env=*KS.get_or_init(||std::env::var("GLM53_SPX_KS").ok().map(|v|v.parse().expect("GLM53_SPX_KS")).unwrap_or(0));
    let q=k/128;
    if env>0 {return (env as i64).min(q).max(1) as i32;}
    // spx-selftest sweep (GB10, rows 1..16): split K 8 ways once a warp keeps >= 1.5 quads, else 4 / 2.
    let _=n;
    if q>=12 {8} else if q>=8 {4} else if q>=2 {2} else {1}
}

/// x [m, k] (FP32 or Half, unit column stride) -> x_rot Half [m, k] contiguous.
fn rotate(x:&Tensor,sign:&Tensor,rot:i64)->Tensor {
    if x.size()[0]==0 {return Tensor::empty([0,x.size()[1]],(Kind::Half,x.device()));}
    let x=if x.stride()[1]==1 && matches!(x.kind(),Kind::Float|Kind::Half) {x.shallow_clone()} else {x.to_kind(Kind::Float).contiguous()};
    let (m,k)=(x.size()[0],x.size()[1]);
    let y=Tensor::empty([m,k],(Kind::Half,x.device()));
    assert_eq!(unsafe{rs_spx_rot(x.data_ptr(),x.stride()[0],i32::from(x.kind()==Kind::Half),sign.data_ptr(),m as i32,k as i32,rot as i32,y.data_ptr(),k,1,1)},0,"SPX rotate");
    y
}
fn out_kind(out:i32)->Kind {if out==2 {Kind::Half} else {Kind::Float}}
impl Spx {
    /// This weight applied to the rotated input xr: [m, n] of the out kind.
    fn run(&self,xr:&Tensor,out:i32)->Tensor {
        let m=xr.size()[0];
        if m==0 {return Tensor::empty([0,self.n],(out_kind(out),xr.device()));}
        if self.mm8_ok(m) {let y=Tensor::empty([m,self.n],(out_kind(out),xr.device()));self.mm8_into(xr,out,&y);return y;}   // (k % 256: TP3 slices of 2688)
        if m>max_rows() && self.n<=self.prefill_step() {
            // Whole weight in one scratch piece: the exact result expressions of the Half path (mm16 / mm16_partial).
            let w=self.dequant_rows(0,self.n);
            if out==0 {
                let y=Tensor::empty([m,self.n],(Kind::Float,xr.device()));
                assert_eq!(unsafe{rs_mm16_f32(xr.data_ptr().cast(),w.data_ptr().cast(),y.data_ptr().cast(),m as i32,self.n as i32,self.k as i32)},0,"SPX prefill GEMM");
                return y;
            }
            let p=xr.matmul(&w.transpose(0,1));
            return if out==1 {p.to_kind(Kind::Float)} else {p};
        }
        let y=Tensor::empty([m,self.n],(out_kind(out),xr.device()));self.run_into(xr,out,&y);y
    }
    /// y [m, n] (contiguous, out kind) = this weight applied to the rotated input xr.
    fn run_into(&self,xr:&Tensor,out:i32,y:&Tensor) {
        let m=xr.size()[0];
        if m==0 {return;}
        if self.mm8_ok(m) {self.mm8_into(xr,out,y);return;}
        if m<=max_rows() {
            assert_eq!(unsafe{rs_spx_gemv(xr.data_ptr(),self.tiles.data_ptr(),self.n as i32,self.k as i32,self.bits as i32,y.data_ptr(),m as i32,out,split_k(self.n,self.k))},0,"SPX GEMV");
        } else {self.prefill(xr,out,y);}
    }
    /// D7 from the un-rotated input: one fused rotation + quantization, then the INT8 GEMM.
    fn mm8x(&self,x:&Tensor,out:i32)->Tensor {
        let (q,sx)=rot8(x,&self.sign,self.rot);let y=Tensor::empty([x.size()[0],self.n],(out_kind(out),x.device()));self.mm8q(&q,&sx,out,&y);y
    }
    fn mm8q(&self,q:&Tensor,sx:&Tensor,out:i32,y:&Tensor) {
        assert!(y.is_contiguous());let m=q.size()[0];
        assert_eq!(unsafe{rs_spx_mm8(q.data_ptr(),sx.data_ptr(),std::ptr::null(),self.tiles.data_ptr(),m as i32,self.n as i32,self.k as i32,self.bits as i32,y.data_ptr(),self.n,out)},0,"SPX mm8");
    }
    fn mm8_ok(&self,m:i64)->bool {m>max_rows() && mm8_on(self.class,self.bits) && self.k%256==0 && self.k<=8192}
    /// D7: y [m, n] (contiguous, out kind) on INT8 tensor cores (activations quantized per token).
    fn mm8_into(&self,xr:&Tensor,out:i32,y:&Tensor) {
        let (q,sx,sum)=quant8(xr);let m=xr.size()[0];
        assert!(y.is_contiguous());
        assert_eq!(unsafe{rs_spx_mm8(q.data_ptr(),sx.data_ptr(),sum.data_ptr(),self.tiles.data_ptr(),m as i32,self.n as i32,self.k as i32,self.bits as i32,y.data_ptr(),self.n,out)},0,"SPX mm8");
    }
    fn prefill_step(&self)->i64 {
        let t=TEST_STEP.with(|c|c.get());if t>0 {return t;}
        (scratch_bytes()/(self.k*2)/16*16).max(16)
    }
    /// Half W_rot of rows [r0, r0+n) (a transient scratch from the caching allocator).
    fn dequant_rows(&self,r0:i64,n:i64)->Tensor {
        let per=self.k/128*tile_bytes(self.bits);
        let w=Tensor::empty([n,self.k],(Kind::Half,self.tiles.device()));
        assert_eq!(unsafe{rs_spx_dequant(self.tiles.narrow(0,r0/16*per,n/16*per).data_ptr(),n as i32,self.k as i32,self.bits as i32,w.data_ptr(),self.k,0)},0,"SPX dequant");
        w
    }
    /// Prefill v1: transient Half W_rot pieces (<= GLM53_SPX_SCRATCH_MIB) and the cuBLAS Half GEMM the Half path uses.
    /// Next step: an INT8 (or FP8) activation MMA kernel reading the tiles directly (x_rot quantized per row and block),
    /// which removes the dequant write + re-read and doubles the MMA rate; it would hook in here (and in `run`), keyed on m.
    fn prefill(&self,xr:&Tensor,out:i32,y:&Tensor) {
        let (m,dev)=(xr.size()[0],xr.device());let step=self.prefill_step();
        let mut r0=0;
        while r0<self.n {
            let n=step.min(self.n-r0);let w=self.dequant_rows(r0,n);
            if out==0 {
                let dst=Tensor::empty([m,n],(Kind::Float,dev));
                assert_eq!(unsafe{rs_mm16_f32(xr.data_ptr().cast(),w.data_ptr().cast(),dst.data_ptr().cast(),m as i32,n as i32,self.k as i32)},0,"SPX prefill GEMM");
                y.narrow(1,r0,n).copy_(&dst);
            } else {y.narrow(1,r0,n).copy_(&xr.matmul(&w.transpose(0,1)));}   // Half result; copy_ widens exactly for out 1
            r0+=n;
        }
    }
}

fn rows_ok(x:&Tensor,k:i64)->bool {
    x.device().is_cuda() && x.dim()==2 && x.size()[1]==k
        && !crate::root_probe::retain_f32() && !crate::root_probe::full_f32() && !crate::root_probe::recording()
}
/// mm16 (out 1: FP32 of the Half-rounded result), mm16_partial (out 0: FP32), Half (out 2) of an SPX weight.
pub(crate) fn try_mm(x:&Tensor,w:&Tensor,out:i32)->Option<Tensor> {
    if !any() {return None;}
    CACHE.with(|c|{let c=c.borrow();let e=c.get(&(w.data_ptr() as usize))?;
        if e.key.size()!=w.size() || e.key.stride()!=w.stride() {return None;}
        assert!(rows_ok(x,e.k),"SPX weight {:?} reached with input {:?} {:?} (or a diagnostic precision mode)",w.size(),x.size(),x.kind());
        let y=if e.mm8_ok(x.size()[0]) {e.mm8x(x,out)} else {let xr=rotated(x,&e.sign,e.rot);e.run(&xr,out)};
        ablate_output(w,&y);Some(y)})
}
/// try_mm into a caller-provided contiguous [rows, n] output of the out kind.
pub(crate) fn try_mm_into(x:&Tensor,w:&Tensor,out:i32,y:&Tensor)->bool {
    if !any() {return false;}
    CACHE.with(|c|{let c=c.borrow();let Some(e)=c.get(&(w.data_ptr() as usize)) else {return false};
        if e.key.size()!=w.size() || e.key.stride()!=w.stride() {return false;}
        assert!(rows_ok(x,e.k),"SPX weight {:?} reached with input {:?}",w.size(),x.size());
        assert!(y.is_contiguous() && y.size()==[x.size()[0],e.n] && y.kind()==out_kind(out),"SPX into: output layout");
        if e.mm8_ok(x.size()[0]) {let (q,sx)=rot8(x,&e.sign,e.rot);e.mm8q(&q,&sx,out,y);} else {let xr=rotated(x,&e.sign,e.rot);e.run_into(&xr,out,y);}
        ablate_output(w,y);true})
}
/// [rows, sum n_i] = cat(w_i applied to x): one launch for a registered group, else one shared rotation (when the members
/// share sign and block) and a launch per member. None when any member is not SPX.
pub(crate) fn try_rows(x:&Tensor,ws:&[&Tensor],out:i32)->Option<Tensor> {
    if !any() {return None;}
    let keys:Vec<usize>=ws.iter().map(|w|w.data_ptr() as usize).collect();
    let members=CACHE.with(|c|{let c=c.borrow();keys.iter().zip(ws).all(|(k,w)|c.get(k).is_some_and(|e|e.key.size()==w.size()&&e.key.stride()==w.stride()))});
    if !members {return None;}
    assert!(ABLATE_OUT.with(|m|{let m=m.borrow();keys.iter().all(|k|!m.contains_key(k))}) && OPROJ_OUT.with(|m|{let m=m.borrow();keys.iter().all(|k|!m.contains_key(k))}),"SPX row group with a SPARK_ABLATE residual writer");
    if let Some(y)=GROUPS.with(|g|{let g=g.borrow();let e=g.get(&keys)?;
        assert!(rows_ok(x,e.k),"SPX group reached with input {:?}",x.size());
        if e.mm8_ok(x.size()[0]) {return Some(e.mm8x(x,out));}
        let xr=rotated(x,&e.sign,e.rot);Some(e.run(&xr,out))}) {return Some(y);}
    CACHE.with(|c|{let c=c.borrow();let es:Vec<&Spx>=keys.iter().map(|k|&c[k]).collect();
        assert!(rows_ok(x,es[0].k),"SPX rows reached with input {:?}",x.size());
        if !es.iter().all(|e|e.same_input(es[0])) {return None;}
        let total:i64=es.iter().map(|e|e.n).sum();
        let y=Tensor::empty([x.size()[0],total],(out_kind(out),x.device()));let mut off=0;
        if es.iter().all(|e|e.mm8_ok(x.size()[0])) {   // one fused rotation + quantization for every member
            let (q,sx)=rot8(x,&es[0].sign,es[0].rot);
            for e in es {let p=Tensor::empty([x.size()[0],e.n],(out_kind(out),x.device()));e.mm8q(&q,&sx,out,&p);y.narrow(1,off,e.n).copy_(&p);off+=e.n;}
            return Some(y);
        }
        let xr=rotated(x,&es[0].sign,es[0].rot);
        for e in es {y.narrow(1,off,e.n).copy_(&e.run(&xr,out));off+=e.n;}
        Some(y)})
}

/// Files of `model` holding SPX tensors (name, size, mtime), for cache identities; None without SPX tensors (or GLM53_SPX=0).
pub(crate) fn fingerprint(model:&std::path::Path)->Option<String> {
    if !enabled() {return None;}
    let idx=ShardIndex::scan(model).ok()?;
    let mut files:Vec<String>=idx.entries.iter().filter(|(n,_)|n.ends_with(".spx_q")).map(|(_,e)|e.file.clone()).collect();
    files.sort();files.dedup();
    if files.is_empty() {return None;}
    Some(files.iter().map(|f|{let m=std::fs::metadata(model.join(f)).ok();
        format!("{f}:{}:{}",m.as_ref().map_or(0,|m|m.len()),m.and_then(|m|m.modified().ok()).and_then(|t|t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0,|d|d.as_nanos()))}).collect::<Vec<_>>().join(","))
}
/// SPX tensors in the index that no loader consumed (e.g. an unsupported linear): warned once after the load.
pub(crate) fn warn_unused(idx:&ShardIndex,n_layers:usize,used:&dyn Fn(&str)->bool) {
    if !enabled() {return;}
    let layer_of=|n:&str|n.strip_prefix("model.language_model.layers.").and_then(|r|r.split('.').next()).and_then(|i|i.parse::<usize>().ok());
    let mut extra:Vec<&String>=idx.entries.keys().filter(|n|n.ends_with(".spx_q")).filter(|n|layer_of(n).map_or(true,|l|l<n_layers))
        .filter(|n|!used(n.trim_end_matches(".spx_q"))).collect();
    extra.sort();
    if !extra.is_empty() {eprintln!("[spx] WARNING: {} SPX tensors are not supported by this engine and were ignored (their BF16 originals load instead), e.g. {:?}",extra.len(),&extra[..extra.len().min(4)]);}
}

// ---------------------------------------------------------------- self test (spx-selftest)

/// CPU reference unpack of one on-disk row bitstream: codes k0 .. k0+k.
fn unpack_row(row:&[u8],bits:i64,k0:i64,k:i64,out:&mut [f32],scales:&[half::f16]) {
    let mid=((1i64<<bits)-1) as f32/2.;
    for i in 0..k {
        let bit=((k0+i)*bits) as usize;let by=bit>>3;let sh=bit&7;
        let mut v=row[by] as u32;if sh+bits as usize>8 {v|=(row[by+1] as u32)<<8;}
        let c=((v>>sh)&((1u32<<bits)-1)) as f32;
        out[i as usize]=(c-mid)*scales[((k0+i)/64) as usize].to_f32();
    }
}
fn pack_bits(codes:&[u8],bits:i64)->Vec<u8> {
    let mut v=vec![0u8;(codes.len()*bits as usize+7)/8];
    for (i,&c) in codes.iter().enumerate() {let bit=i*bits as usize;let w=(c as u32)<<(bit&7);v[bit>>3]|=w as u8;if (bit&7)+bits as usize>8 {v[(bit>>3)+1]|=(w>>8) as u8;}}
    v
}
fn hadamard(b:i64,dev:Device)->Tensor {
    let mut h=Tensor::ones([1,1],(Kind::Double,dev));
    while h.size()[0]<b {h=Tensor::cat(&[Tensor::cat(&[&h,&h],1),Tensor::cat(&[&h,&(-&h)],1)],0);}
    h/(b as f64).sqrt()
}
struct Synth {codes:Tensor,scales:Tensor,sign:Tensor,wrot:Tensor}
/// Random on-disk tensors for [n, kfull] at `bits`; wrot = the reference W_rot (FP64, CPU unpack) of columns [k0, k0+k).
fn synth(n:i64,kfull:i64,k0:i64,k:i64,bits:i64,seed:u64,dev:Device)->Synth {
    let mut st=seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut rnd=||{st^=st<<13;st^=st>>7;st^=st<<17;st};
    let rb=(kfull*bits+7)/8;
    let mut raw=vec![0u8;(n*rb) as usize];let mut sc=vec![half::f16::ZERO;(n*(kfull/64)) as usize];
    for r in 0..n as usize {
        let codes:Vec<u8>=(0..kfull).map(|_|(rnd()%(1u64<<bits)) as u8).collect();
        raw[r*rb as usize..(r+1)*rb as usize].copy_from_slice(&pack_bits(&codes,bits));
        for g in 0..(kfull/64) as usize {sc[r*(kfull/64) as usize+g]=half::f16::from_f32((1e-3+(rnd()%1000) as f32*2e-5)*if bits==8{0.1}else{1.});}
    }
    let sign:Vec<i8>=(0..kfull).map(|_|if rnd()&1==1{1}else{-1}).collect();
    let mut w=vec![0f32;(n*k) as usize];
    for r in 0..n as usize {unpack_row(&raw[r*rb as usize..],bits,k0,k,&mut w[r*k as usize..(r+1)*k as usize],&sc[r*(kfull/64) as usize..]);}
    Synth{codes:Tensor::from_slice(&raw).view([n,rb]).to_device(dev),
        scales:Tensor::from_slice(&sc.iter().map(|h|h.to_bits() as i16).collect::<Vec<_>>()).view([n,kfull/64]).to_device(dev).view_dtype(Kind::Half),
        sign:Tensor::from_slice(&sign).to_device(dev).narrow(0,k0,k).contiguous(),
        wrot:Tensor::from_slice(&w).view([n,k]).to_kind(Kind::Double).to_device(dev)}
}
fn rel(a:&Tensor,b:&Tensor)->f64 {let a=a.to_kind(Kind::Double);let b=b.to_kind(Kind::Double);((&a-&b).norm().double_value(&[]))/(b.norm().double_value(&[])+1e-30)}
fn maxabs_rel(a:&Tensor,b:&Tensor)->f64 {let a=a.to_kind(Kind::Double);let b=b.to_kind(Kind::Double);(&a-&b).abs().max().double_value(&[])/(b.abs().max().double_value(&[])+1e-30)}

/// `spx-selftest [OUT.json]`: kernels vs FP64 references on synthetic data for every bit width, model shape and row count,
/// then GEMV timings (rotation included) over a working set larger than L2. Single GPU, no checkpoint.
pub fn selftest(out:Option<&std::path::Path>) {
    let dev=Device::Cuda(0);let _g=tch::no_grad_guard();
    let shapes:[(&str,i64,i64,i64,bool);10]=[   // name, n, k (local), rot_block, row-split (codes taken at k0 = k of a [n, 2k] tensor)
        ("kda_qkv_group",12288,4096,4096,false),("kda_o",4096,4096,4096,true),("mla_q_b",8192,1536,512,false),("mla_o",4096,8192,4096,true),
        ("dense_gu_group",12288,4096,4096,false),("dense_down",4096,6144,2048,true),("shared_gu_group",2048,4096,4096,false),
        ("shared_down",4096,1024,1024,true),("head_shard",77440,4096,4096,false),("mla_kv_b",16384,512,512,false)];
    let rows_list=[1i64,2,4,8,16,32,64,512];
    let mut results=Vec::new();let mut worst=0f64;
    for bits in [4i64,5,6,8] {
        for &(name,n,k,rot,rowsplit) in &shapes {
            // Correctness on at most 8192 rows (the reference is FP64); timing below on the full shape.
            let nt=n.min(8192);let (kfull,k0)=if rowsplit {(2*k,k)} else {(k,0)};
            let s=synth(nt,kfull,k0,k,bits,(bits*1000+n+k) as u64,dev);
            let tiles=pack(&s.codes,&s.scales,bits,k0,k);
            let e=Spx{key:Tensor::zeros([1],(Kind::Half,dev)),tiles,sign:s.sign.shallow_clone(),n:nt,k,bits,rot,class:"test"};
            // Dequant (Half operand) equals Half_RN(W_rot) exactly; reconstruction vs FP64 W_rot R^T.
            let wd=Tensor::empty([nt,k],(Kind::Half,dev));
            assert_eq!(unsafe{rs_spx_dequant(e.tiles.data_ptr(),nt as i32,k as i32,bits as i32,wd.data_ptr(),k,0)},0);
            let wh=s.wrot.to_kind(Kind::Float).to_kind(Kind::Half);
            let deq_mismatch=wd.view_dtype(Kind::Int16).ne_tensor(&wh.view_dtype(Kind::Int16)).sum(Kind::Int64).int64_value(&[]);
            let h=hadamard(rot,dev);let sg=s.sign.to_kind(Kind::Double);
            let wfull=(s.wrot.view([nt,k/rot,rot]).matmul(&h).view([nt,k]))*&sg;   // W_rot R^T
            let recon=reconstruct_entry(&e);let recon_rel=rel(&recon,&wfull);
            let mut rec=serde_json::json!({"bits":bits,"shape":name,"n":n,"k":k,"rot_block":rot,"row_split_k0":k0,"dequant_half_mismatches":deq_mismatch,"reconstruct_rel_l2":recon_rel,"rows":{}});
            assert_eq!(deq_mismatch,0,"{name} b{bits}: dequant differs from Half_RN(W_rot)");
            assert!(recon_rel<5e-3,"{name} b{bits}: reconstruction rel {recon_rel}");
            for &m in &rows_list {
                let x=Tensor::randn([m,k],(Kind::Float,dev));
                let xrot=((&x.to_kind(Kind::Double)*&sg).view([m,k/rot,rot]).matmul(&h)).view([m,k]);
                let yref=xrot.matmul(&s.wrot.transpose(0,1));                                    // exact FP64
                let xr=rotate(&x,&e.sign,rot);
                // GEMM reference on the kernel's own rotated Half input (the rotation is checked separately: rot_maxabs_rel).
                let xh=xr.to_kind(Kind::Double);
                // Kernel semantics: decode applies the exact weights to Half(x_rot); prefill Half(W_rot) . Half(x_rot) (cuBLAS).
                let yhalf=if m<=max_rows() {xh.matmul(&s.wrot.transpose(0,1))} else {xh.matmul(&wh.to_kind(Kind::Double).transpose(0,1))};
                let rot_err=maxabs_rel(&xr,&xrot);assert!(rot_err<2e-3,"{name}: rotation error {rot_err}");
                let mut r=serde_json::Map::new();r.insert("rot_maxabs_rel".into(),rot_err.into());
                for out in [0,1,2] {
                    let y=e.run(&xr,out);
                    let (a,b)=(rel(&y,&yref),rel(&y,&yhalf));worst=worst.max(b);
                    r.insert(format!("out{out}_rel_vs_fp64"),a.into());r.insert(format!("out{out}_rel_vs_kernel_semantics"),b.into());
                    let tol=if out==0 {1e-5} else {1e-3};
                    assert!(b<tol,"{name} b{bits} rows {m} out {out}: rel {b} vs the reference of the kernel semantics");
                }
                rec["rows"][m.to_string()]=serde_json::Value::Object(r);
            }
            {   // multi-piece prefill (scratch smaller than the weight) equals the single-piece result
                let x=Tensor::randn([96,k],(Kind::Float,dev));let xr=rotate(&x,&e.sign,rot);
                let mut worst_piece=0f64;
                for out in [0,1,2] {let a=e.run(&xr,out);TEST_STEP.with(|c|c.set(((nt/3)/16*16).max(16)));let b=e.run(&xr,out);TEST_STEP.with(|c|c.set(0));
                    let r=rel(&b,&a);worst_piece=worst_piece.max(r);assert!(r<1e-3,"{name} b{bits}: multi-piece prefill differs (out {out}) rel {r}");}
                rec["prefill_pieces_rel"]=worst_piece.into();
            }
            eprintln!("[spx-selftest] b{bits} {name:16} [{nt},{k}] B{rot} k0 {k0}: dequant exact, recon {recon_rel:.1e}, rows 1..512 ok");
            results.push(rec);
        }
    }
    // Timing: full shapes, working set > 300 MB of tiles (copies cycled), rotation included per call.
    let mut timing=Vec::new();
    for bits in [4i64,5,6,8] {
        for &(name,n,k,rot,_) in &shapes {
            let tb=n/16*(k/128)*tile_bytes(bits);let copies=((320i64<<20)/tb).clamp(2,64);
            let ws:Vec<Tensor>=(0..copies).map(|_|Tensor::randint(256,[tb],(Kind::Uint8,dev))).collect();
            let sign=Tensor::ones([k],(Kind::Int8,dev));
            for &m in &[1i64,2,4,8,16,32] {
                let x=Tensor::randn([m,k],(Kind::Float,dev));
                let mut best=(f64::MAX,0);
                let kss:Vec<i32>=[1,2,4,8].into_iter().filter(|&s|(k/128)>=s as i64).collect();
                for ks in kss.iter().copied().chain(std::iter::once(0)) {
                    let ksv=if ks==0 {split_k(n,k)} else {ks};
                    let y=Tensor::empty([m,n],(Kind::Float,dev));
                    let run=|i:usize|{let xr=rotate(&x,&sign,rot);
                        assert_eq!(unsafe{rs_spx_gemv(xr.data_ptr(),ws[i%ws.len()].data_ptr(),n as i32,k as i32,bits as i32,y.data_ptr(),m as i32,1,ksv)},0);};
                    for i in 0..8 {run(i);}tch::Cuda::synchronize(0);
                    let iters=(ws.len()*4).max(40);let t0=std::time::Instant::now();
                    for i in 0..iters {run(i);}tch::Cuda::synchronize(0);
                    let us=t0.elapsed().as_secs_f64()*1e6/iters as f64;
                    if ks==0 {
                        let gbs=tb as f64/us*1e-3;
                        timing.push(serde_json::json!({"bits":bits,"shape":name,"n":n,"k":k,"rows":m,"ks":ksv,"us":us,"weight_GBps":gbs,"best_ks":best.1,"best_us":best.0}));
                        eprintln!("[spx-time] b{bits} {name:16} rows {m:2} ks {ksv} {us:8.1} us {gbs:6.1} GB/s (best ks {} {:.1} us)",best.1,best.0);
                    } else if us<best.0 {best=(us,ks);}
                }
            }
        }
    }
    // Production Q8 (tiled, GLM53_Q8_TILED/XHALF, C12_KS=8 split rule) on the same shapes, for the per-round comparison.
    let mut q8=Vec::new();
    extern "C"{fn rs_q8t_gemm(x:*const std::ffi::c_void,t:*const u8,n:i32,k:i32,y:*mut std::ffi::c_void,m:i32,out:i32,ks:i32,xhalf:i32)->i32;}
    for &(name,n,k,_,_) in &shapes {
        if name=="mla_kv_b" {continue;}   // kv_b is read by the latent C12 bmm, not a Q8 GEMM
        let tb=n/16*(k/128)*2112;let copies=((320i64<<20)/tb).clamp(2,64);
        let ws:Vec<Tensor>=(0..copies).map(|_|Tensor::randint(256,[tb],(Kind::Uint8,dev))).collect();
        let ks=if (k/32)%8==0 && (k/32/8)%4==0 {8} else if n<=2048 {8} else {4};
        for &m in &[1i64,2,4,8,16,32] {
            let x=Tensor::randn([m,k],(Kind::Float,dev));let y=Tensor::empty([m,n],(Kind::Float,dev));
            let run=|i:usize|{let xh=(m>8).then(||x.to_kind(Kind::Half));let xp=xh.as_ref().map_or(x.data_ptr(),|h|h.data_ptr());
                assert_eq!(unsafe{rs_q8t_gemm(xp,ws[i%ws.len()].data_ptr().cast(),n as i32,k as i32,y.data_ptr(),m as i32,1,ks,i32::from(m>8))},0);};
            for i in 0..8 {run(i);}tch::Cuda::synchronize(0);
            let iters=(ws.len()*4).max(40);let t0=std::time::Instant::now();
            for i in 0..iters {run(i);}tch::Cuda::synchronize(0);
            let us=t0.elapsed().as_secs_f64()*1e6/iters as f64;
            q8.push(serde_json::json!({"shape":name,"n":n,"k":k,"rows":m,"ks":ks,"us":us,"weight_GBps":tb as f64/us*1e-3}));
            eprintln!("[q8t-time] {name:16} rows {m:2} ks {ks} {us:8.1} us {:6.1} GB/s",tb as f64/us*1e-3);
        }
    }
    let j=serde_json::json!({"correctness":results,"worst_rel_vs_kernel_semantics":worst,"timing":timing,"q8t_timing":q8});
    if let Some(p)=out {std::fs::write(p,serde_json::to_string_pretty(&j).unwrap()).unwrap();}
    eprintln!("[spx-selftest] PASS (worst rel vs the reference of the kernel semantics {worst:.2e})");
}

/// `spx-mm8-test`: D7 INT8 prefill kernel vs FP64 on synthetic tiles (b 4/5/6, model shapes, rows 64/200/4096), the
/// activation-quantization error vs the Half path, and timing at 4096 rows (quant8 included) vs dequant + cuBLAS.
pub fn mm8_test() {
    let dev=Device::Cuda(0);let _g=tch::no_grad_guard();
    let shapes:[(&str,i64,i64,i64,bool);9]=[("kda_qkv_group",12288,4096,4096,false),("kda_o",4096,4096,4096,true),("mla_q_b",8192,1536,512,false),
        ("mla_o",4096,8192,4096,true),("dense_gu_group",12288,4096,4096,false),("dense_down",4096,6144,2048,true),("shared_gu_group",2048,4096,4096,false),
        ("shared_down",4096,1024,1024,true),("head_shard",77440,4096,4096,false)];
    for bits in [6i64,5,4] {
        for &(name,n,k,rot,rowsplit) in &shapes {
            let nt=n.min(4096);let (kfull,k0)=if rowsplit {(2*k,k)} else {(k,0)};
            let s=synth(nt,kfull,k0,k,bits,(bits*1000+n+k) as u64,dev);
            let e=Spx{key:Tensor::zeros([1],(Kind::Half,dev)),tiles:pack(&s.codes,&s.scales,bits,k0,k),sign:s.sign.shallow_clone(),n:nt,k,bits,rot,class:"test"};
            let mut line=format!("[mm8] b{bits} {name:16} [{nt},{k}]");
            for &m in &[64i64,200,4096] {
                let x=Tensor::randn([m,k],(Kind::Float,dev));let xr=rotate(&x,&e.sign,rot);
                let (q,sx,_)=quant8(&xr);
                let xd=q.to_kind(Kind::Double)*sx.to_kind(Kind::Double).unsqueeze(1);
                let yref=xd.matmul(&s.wrot.transpose(0,1));                 // the kernel's exact semantics
                let yfull=xr.to_kind(Kind::Double).matmul(&s.wrot.transpose(0,1));   // un-quantized activations
                let y=Tensor::empty([m,nt],(Kind::Float,dev));e.mm8_into(&xr,0,&y);
                let (a,b)=(rel(&y,&yref),rel(&y,&yfull));
                assert!(a<1e-5,"{name} b{bits} rows {m}: mm8 rel {a} vs its FP64 semantics");
                for out in [1,2] {let yo=Tensor::empty([m,nt],(out_kind(out),dev));e.mm8_into(&xr,out,&yo);let d=rel(&yo,&y);assert!(d<1e-3,"{name} out {out}: rel {d}");}
                line+=&format!(" | rows {m}: exact {a:.1e} a8 {b:.2e}");
            }
            eprintln!("{line}");
        }
    }
    // timing, 4096 rows, full shapes: mm8 (quant8 included) vs the Half prefill path (dequant pieces + cuBLAS)
    for bits in [6i64,5] {
        for &(name,n,k,rot,_) in &shapes {
            let tb=n/16*(k/128)*tile_bytes(bits);
            let tiles=Tensor::randint(256,[tb],(Kind::Uint8,dev));
            // valid scales (random bytes could be Inf/NaN halves): rewrite the 64-byte scale block of every tile
            let tv=tiles.view([-1,tile_bytes(bits)]);let _=tv.narrow(1,256*bits,64).copy_(&Tensor::full([tv.size()[0],32],0x1400i64,(Kind::Int16,dev)).view_dtype(Kind::Uint8).view([-1,64]));
            let e=Spx{key:Tensor::zeros([1],(Kind::Half,dev)),tiles,sign:Tensor::ones([k],(Kind::Int8,dev)),n,k,bits,rot,class:"test"};
            let m=4096;let x=Tensor::randn([m,k],(Kind::Float,dev));let xr=rotate(&x,&e.sign,rot);
            let time=|f:&dyn Fn()|->f64 {for _ in 0..2{f();}tch::Cuda::synchronize(0);let t0=std::time::Instant::now();for _ in 0..5{f();}tch::Cuda::synchronize(0);t0.elapsed().as_secs_f64()*1e3/5.};
            let y=Tensor::empty([m,n],(Kind::Float,dev));
            let t8=time(&||e.mm8_into(&xr,0,&y));
            let t8f=time(&||{let (q,sx)=rot8(&x,&e.sign,rot);e.mm8q(&q,&sx,0,&y);});
            let th_rot=time(&||{let xr=rotate(&x,&e.sign,rot);e.prefill(&xr,0,&y);});
            let th=time(&||e.prefill(&xr,0,&y));
            let tf=2.*m as f64*n as f64*k as f64;
            eprintln!("[mm8-time] b{bits} {name:16} [{n},{k}] rows {m}: int8 {t8:8.2} ms ({:6.1} TOPS) | half {th:8.2} ms ({:6.1} TFLOPS) | x{:.2} || with rotation: rot8+int8 {t8f:7.2} ms vs rot+half {th_rot:7.2} ms x{:.2}",tf/t8*1e-9,tf/th*1e-9,th/t8,th_rot/t8f);
        }
    }
}

/// `spx-swiglu-test`: swiglu_rot vs clamp/silu/mul + spx_rot (bitwise), rows 1..32, n 1024 / 2048, rotation blocks; and
/// kda_onorm_rot vs kda_onorm_gate + spx_rot (bitwise).
pub fn swiglu_test() {
    let dev=Device::Cuda(0);let _g=tch::no_grad_guard();
    for &(n,bk) in &[(1024i64,1024i64),(1024,512),(2048,2048),(6144,2048)] {
        for &m in &[1i64,3,8,16,32] {
            let y=(Tensor::randn([m,2*n],(Kind::Float,dev))*6.0).contiguous();
            let sign=(Tensor::randint(2,[n],(Kind::Int8,dev))*2-1).to_kind(Kind::Int8);
            let g=y.narrow(1,0,n).clamp(f64::NEG_INFINITY,10.0);let u=y.narrow(1,n,n).clamp(-10.0,10.0);
            let a0=g.silu()*u;let x0=rotate(&a0,&sign,bk);
            let act=Tensor::empty([m,n],(Kind::Float,dev));let xr=Tensor::empty([m,n],(Kind::Half,dev));
            assert_eq!(unsafe{rs_spx_swiglu_rot(y.data_ptr(),m as i32,n as i32,10.0,sign.data_ptr(),bk as i32,act.data_ptr(),xr.data_ptr())},0);
            let da=act.view_dtype(Kind::Int).ne_tensor(&a0.contiguous().view_dtype(Kind::Int)).sum(Kind::Int64).int64_value(&[]);
            let dx=xr.view_dtype(Kind::Int16).ne_tensor(&x0.view_dtype(Kind::Int16)).sum(Kind::Int64).int64_value(&[]);
            eprintln!("[swiglu-rot] n {n} bk {bk} rows {m}: act mismatches {da}, x_rot mismatches {dx}");
            if n==1024 && bk==1024 {
                let time=|f:&dyn Fn()|->f64 {for _ in 0..20{f();}tch::Cuda::synchronize(0);let t0=std::time::Instant::now();for _ in 0..500{f();}tch::Cuda::synchronize(0);t0.elapsed().as_secs_f64()*1e6/500.};
                let ta=time(&||{let g=y.narrow(1,0,n).clamp(f64::NEG_INFINITY,10.0);let u=y.narrow(1,n,n).clamp(-10.0,10.0);let a0=g.silu()*u;let _=rotate(&a0,&sign,bk);});
                let tb=time(&||{assert_eq!(unsafe{rs_spx_swiglu_rot(y.data_ptr(),m as i32,n as i32,10.0,sign.data_ptr(),bk as i32,act.data_ptr(),xr.data_ptr())},0);});
                eprintln!("[swiglu-rot-time] rows {m}: separate {ta:.1} us, fused {tb:.1} us (eager, launch included)");
            }
        }
    }
    extern "C"{fn rs_kda_onorm_gate(o:*const f32,onorm:*const f32,len:i32,sg2:*const f32,out:*mut f32,groups:i32)->i32;}
    for &(heads,bk,onl) in &[(32i64,4096i64,128i64),(32,4096,4096),(16,2048,128),(32,1024,128)] {
        for &m in &[1i64,4,16,32] {
            let o=Tensor::randn([m,heads,128],(Kind::Float,dev));let sg2=Tensor::rand([m,heads,128],(Kind::Float,dev));
            let on=Tensor::randn([onl],(Kind::Float,dev))+1.0;let sign=(Tensor::randint(2,[heads*128],(Kind::Int8,dev))*2-1).to_kind(Kind::Int8);
            let h0=Tensor::empty([m,heads,128],(Kind::Float,dev));
            assert_eq!(unsafe{rs_kda_onorm_gate(o.data_ptr().cast(),on.data_ptr().cast(),onl as i32,sg2.data_ptr().cast(),h0.data_ptr().cast(),(m*heads) as i32)},0);
            let x0=rotate(&h0.reshape([m,heads*128]),&sign,bk);
            let h1=Tensor::empty([m,heads,128],(Kind::Float,dev));let x1=Tensor::empty([m,heads*128],(Kind::Half,dev));
            assert_eq!(unsafe{rs_spx_kda_onorm_rot(o.data_ptr(),on.data_ptr(),onl as i32,sg2.data_ptr(),h1.data_ptr(),m as i32,heads as i32,sign.data_ptr(),bk as i32,x1.data_ptr())},0);
            let dh=h1.view_dtype(Kind::Int).ne_tensor(&h0.view_dtype(Kind::Int)).sum(Kind::Int64).int64_value(&[]);
            let dx=x1.view_dtype(Kind::Int16).ne_tensor(&x0.view_dtype(Kind::Int16)).sum(Kind::Int64).int64_value(&[]);
            eprintln!("[onorm-rot] heads {heads} bk {bk} onorm {onl} rows {m}: hidden mismatches {dh}, x_rot mismatches {dx}");
            if heads==32 && bk==4096 && onl==128 {
                let time=|f:&dyn Fn()|->f64 {for _ in 0..20{f();}tch::Cuda::synchronize(0);let t0=std::time::Instant::now();for _ in 0..500{f();}tch::Cuda::synchronize(0);t0.elapsed().as_secs_f64()*1e6/500.};
                let ta=time(&||{assert_eq!(unsafe{rs_kda_onorm_gate(o.data_ptr().cast(),on.data_ptr().cast(),onl as i32,sg2.data_ptr().cast(),h0.data_ptr().cast(),(m*heads) as i32)},0);let _=rotate(&h0.reshape([m,heads*128]),&sign,bk);});
                let tb=time(&||{assert_eq!(unsafe{rs_spx_kda_onorm_rot(o.data_ptr(),on.data_ptr(),onl as i32,sg2.data_ptr(),h1.data_ptr(),m as i32,heads as i32,sign.data_ptr(),bk as i32,x1.data_ptr())},0);});
                eprintln!("[onorm-rot-time] rows {m}: separate {ta:.1} us, fused {tb:.1} us (eager, launch included)");
            }
        }
    }
}
