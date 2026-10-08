//! DFlash2 BF16 drafter with optional head/MLP TP; target remains EXL3.
//! Grouped convolution and greedy selector walk of the DFlash2 drafter.
use std::path::Path;
use tch::{Tensor,Kind,Device};

/// GLM53_DRAFT_INT4_LAYERS=1 (draft side, L2): drafter projections with K a multiple of 1024 up to 4096 and at most 8
/// rows read the INT4 copy (tensor-core kernel, draft_head4.cu) instead of FP8; same output boundary as run_bf16
/// (BF16 unless `partial`). Copies are built at load (prepare_head_int4), never during capture.
fn int4_linear(x:&Tensor,w:&Tensor,partial:bool)->Option<Tensor> {
    if std::env::var("GLM53_DRAFT_INT4_LAYERS").as_deref()!=Ok("1") {return None;}
    let (m,k)=(x.size()[0],x.size()[1]);
    if x.dim()!=2 || !(1..=8).contains(&m) || k%1024!=0 || k>4096 {return None;}
    let (q,s)=crate::dense_fp8::quant_of_bf16(w)?;
    if q.size()[0]%8!=0 {return None;}
    let (q4,s4)=head4_of(&q,&s)?;
    let x=x.to_kind(Kind::BFloat16).contiguous();let n=q.size()[0];
    let y=Tensor::empty([m,n],(Kind::Float,x.device()));
    extern "C"{fn rs_draft_head_int4(x:*const std::ffi::c_void,q4:*const std::ffi::c_void,s:*const f32,y:*mut f32,m:i32,n:i32,k:i32)->i32;}
    assert_eq!(unsafe{rs_draft_head_int4(x.data_ptr(),q4.data_ptr(),s4.data_ptr().cast(),y.data_ptr().cast(),m as i32,n as i32,k as i32)},0,"INT4 drafter linear");
    Some(if partial{y}else{y.to_kind(Kind::BFloat16)})
}
fn linear(x:&Tensor,w:&Tensor)->Tensor {
    if let Some(y)=int4_linear(x,w,false){return y;}
    if let Some(y)=crate::dense_fp8::try_bf16(x,w,false){return y;}
    assert!(!w.stride().iter().all(|&s|s==0),"FP8-freed drafter weight reached a non-FP8 path");
    x.to_kind(Kind::BFloat16).matmul(&w.transpose(0,1))
}
thread_local!{static HEAD4:std::cell::RefCell<std::collections::HashMap<usize,(Tensor,Tensor)>>=std::cell::RefCell::new(std::collections::HashMap::new());}
struct Norm {original:Tensor,expanded:Tensor}
impl Norm {fn new(original:Tensor)->Self {let expanded=original.to_kind(Kind::Float);Self{original,expanded}}}
fn norm(x:&Tensor,w:&Norm)->Tensor {
    let weight=if std::env::var("GLM53_DRAFT_NORM_CACHE").as_deref()==Ok("1"){w.expanded.shallow_clone()}else{w.original.to_kind(Kind::Float)};
    let f=x.to_kind(Kind::Float);let ms=(&f*&f).mean_dim(&[-1i64][..],true,Kind::Float);
    (f*(ms+1e-5).rsqrt()*weight).to_kind(Kind::BFloat16)
}
fn add_norm(x:&Tensor,residual:&Tensor,w:&Norm)->(Tensor,Tensor) {
    let sum=x.to_kind(Kind::Float)+residual.to_kind(Kind::Float);
    (norm(&sum,w),sum.to_kind(Kind::BFloat16))
}
/// The norm parameters are tiny and naturally cache-resident. Measure their
/// conversion/launch cost separately from unchanged projection weight traffic.
pub fn norm_cache_probe(out:&Path) {
    use serde_json::json;use std::time::Instant;
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let dev=Device::Cuda(0);tch::manual_seed(923703);
    let mut shapes=Vec::new();
    for _ in 0..5 {shapes.extend([[8,4096],[8,4096],[256,128],[64,128]]);}
    shapes.push([8,4096]);shapes.push([4,4096]);for _ in 0..5{shapes.push([32,128]);}
    let mut inputs:Vec<_>=shapes.iter().map(|s|Tensor::randn(*s,(Kind::BFloat16,dev))).collect();
    let norms:Vec<_>=shapes.iter().map(|s|Norm::new(Tensor::randn([s[1]],(Kind::BFloat16,dev)))).collect();
    let mut rounds=Vec::new();
    for cached in [false,true,true,false] {
        std::env::set_var("GLM53_DRAFT_NORM_CACHE",if cached{"1"}else{"0"});
        for (x,w) in inputs.iter().zip(&norms){let _=norm(x,w);}tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let outputs:Vec<_>=inputs.iter().zip(&norms).map(|(x,w)|norm(x,w)).collect();crate::tp::graph::end().unwrap();
        for value in [0.,1.,-3.] {
            for x in &mut inputs{let _=x.fill_(value);}crate::tp::graph::replay().unwrap();
            std::env::set_var("GLM53_DRAFT_NORM_CACHE","0");
            for ((x,w),y) in inputs.iter().zip(&norms).zip(&outputs){assert!(y.equal(&norm(x,w)),"norm cache graph changed result");}
            std::env::set_var("GLM53_DRAFT_NORM_CACHE",if cached{"1"}else{"0"});
        }
        for _ in 0..3{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
        let mut samples=Vec::new();for _ in 0..5 {let t=Instant::now();for _ in 0..128{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);samples.push(t.elapsed().as_secs_f64()*1e6/128.);}
        crate::tp::graph::destroy();rounds.push(json!({"cached":cached,"us_per_27_norms":samples}));
    }
    std::env::set_var("GLM53_DRAFT_NORM_CACHE","0");std::fs::create_dir_all(out).unwrap();
    std::fs::write(out.join("norm-cache.json"),serde_json::to_string_pretty(&json!({"shape_list":shapes,"exact":true,"synthetic_values":true,"rounds":rounds})).unwrap()).unwrap();
}

fn rope_tables(positions:&Tensor,inv:Option<&Tensor>)->(Tensor,Tensor) {
    let inv=inv.map(Tensor::shallow_clone).unwrap_or_else(||(Tensor::arange(64,(Kind::Float,positions.device()))*(-10000f64.ln()/64.)).exp());
    let angle=positions.to_kind(Kind::Float).unsqueeze(1)*inv.unsqueeze(0);
    (angle.cos().to_kind(Kind::BFloat16).unsqueeze(1).to_kind(Kind::Float),
     angle.sin().to_kind(Kind::BFloat16).unsqueeze(1).to_kind(Kind::Float))
}
fn rope(x:&Tensor,positions:&Tensor,cached:Option<&(Tensor,Tensor)>)->Tensor {
    let computed;let (cos,sin)=match cached{Some(t)=>t,None=>{computed=rope_tables(positions,None);&computed}};
    let a=x.narrow(-1,0,64).to_kind(Kind::Float);let b=x.narrow(-1,64,64).to_kind(Kind::Float);
    Tensor::cat(&[&a*cos-&b*sin,&b*cos+&a*sin],-1).to_kind(Kind::BFloat16)
}

fn fused_norm()->bool {std::env::var("GLM53_DRAFT_FUSED_NORM").as_deref()==Ok("1")}
/// L2 drafter fusion: (norm(x+r) as BF16, BF16(x+r)); r=None normalizes x and returns x as residual.
fn add_norm_f(x:&Tensor,r:Option<&Tensor>,w:&Norm)->(Tensor,Tensor) {
    let x=x.to_kind(Kind::BFloat16).contiguous();let (rows,d)=(x.size()[0],x.size()[1]);
    let z=Tensor::empty([rows,d],(Kind::BFloat16,x.device()));
    let rout=if r.is_some(){Tensor::empty([rows,d],(Kind::BFloat16,x.device()))}else{x.shallow_clone()};
    let rr=r.map(|t|t.to_kind(Kind::BFloat16).contiguous());
    extern "C"{fn rs_draft_add_norm(x:*const std::ffi::c_void,r:*const std::ffi::c_void,w:*const f32,z:*mut std::ffi::c_void,rout:*mut std::ffi::c_void,rows:i32,d:i32)->i32;}
    assert_eq!(unsafe{rs_draft_add_norm(x.data_ptr(),rr.as_ref().map_or(std::ptr::null(),|t|t.data_ptr() as *const _),w.expanded.data_ptr().cast(),z.data_ptr(),
        if r.is_some(){rout.data_ptr()}else{std::ptr::null_mut()},rows as i32,d as i32)},0);
    (z,rout)
}
fn head_norm_rope(y:&Tensor,w:&Norm,positions:&Tensor,inv:&Tensor)->Tensor {
    let y=y.to_kind(Kind::BFloat16).contiguous();let (n,h)=(y.size()[0],y.size()[1]);assert_eq!(y.size()[2],128);
    let out=Tensor::empty([n,h,128],(Kind::BFloat16,y.device()));let pos=positions.to_kind(Kind::Int64).contiguous();
    extern "C"{fn rs_draft_head_norm_rope(y:*const std::ffi::c_void,w:*const f32,pos:*const i64,inv:*const f32,out:*mut std::ffi::c_void,n:i32,heads:i32)->i32;}
    assert_eq!(unsafe{rs_draft_head_norm_rope(y.data_ptr(),w.expanded.data_ptr().cast(),pos.data_ptr().cast(),inv.data_ptr().cast(),out.data_ptr(),n as i32,h as i32)},0);out
}
fn rope_cache()->bool {std::env::var("GLM53_DRAFT_ROPE_CACHE").as_deref()==Ok("1")}

// Group the four query heads into the M dimension of each KV-head GEMM.
// The absolute-distance mask includes later tokens within the draft block.
pub(crate) fn attention(q:&Tensor,ck:&Tensor,cv:&Tensor,k:&Tensor,v:&Tensor,visible:&Tensor,grouped:bool)->Tensor {
    let n=q.size()[0];let total=ck.size()[0]+k.size()[0];let (qh,kvh)=(q.size()[1],k.size()[1]);
    let k=Tensor::cat(&[ck,k],0);let v=Tensor::cat(&[cv,v],0);
    let (q,k,v)=if grouped {
        (q.transpose(0,1).to_kind(Kind::Float).contiguous().view([kvh,4*n,128]),
         k.transpose(0,1).to_kind(Kind::Float),v.transpose(0,1).to_kind(Kind::Float))
    }else {
        (q.transpose(0,1).to_kind(Kind::Float),
         k.repeat_interleave_self_int(4,1,None).transpose(0,1).to_kind(Kind::Float),
         v.repeat_interleave_self_int(4,1,None).transpose(0,1).to_kind(Kind::Float))
    };
    let score=(q.matmul(&k.transpose(1,2))/128f64.sqrt()).view([qh,n,total]);
    let prob=score.masked_fill(&visible.logical_not().unsqueeze(0),f64::NEG_INFINITY).softmax(-1,Kind::Float);
    let prob=if grouped{prob.view([kvh,4*n,total])}else{prob};
    let value=if grouped && std::env::var("GLM53_DRAFT_GQA_PRECISE").as_deref()==Ok("1") {bmm_fp32(&prob,&v)}else{prob.matmul(&v)};
    value.view([qh,n,128]).transpose(0,1).reshape([n,qh*128]).to_kind(Kind::BFloat16)
}

fn bmm_fp32(a:&Tensor,b:&Tensor)->Tensor {
    assert_eq!(a.kind(),Kind::Float);assert_eq!(b.kind(),Kind::Float);
    let out=Tensor::empty([a.size()[0],a.size()[1],b.size()[2]],(Kind::Float,a.device()));
    extern "C" {fn rs_bmm_fp32(a:*const f32,b:*const f32,out:*mut f32,ashape:*const i64,astride:*const i64,bshape:*const i64,bstride:*const i64)->i32;}
    assert_eq!(unsafe{rs_bmm_fp32(a.data_ptr().cast(),b.data_ptr().cast(),out.data_ptr().cast(),a.size().as_ptr(),a.stride().as_ptr(),b.size().as_ptr(),b.stride().as_ptr())},0);out
}

// Keep the original eight-query head geometry while sharing the KV allocation.
// Four strided BMMs consume the four Q groups; this is a diagnostic alternative
// to changing M from n to 4*n. No KV repeat_interleave is performed.
pub(crate) fn attention_shared_heads(q:&Tensor,ck:&Tensor,cv:&Tensor,k:&Tensor,v:&Tensor,visible:&Tensor)->Tensor {
    let n=q.size()[0];let total=ck.size()[0]+n;let (qh,kvh)=(q.size()[1],k.size()[1]);
    let q=q.transpose(0,1).to_kind(Kind::Float).view([kvh,4,n,128]);
    let k=Tensor::cat(&[ck,k],0).transpose(0,1).to_kind(Kind::Float);
    let v=Tensor::cat(&[cv,v],0).transpose(0,1).to_kind(Kind::Float);
    let scores=Tensor::empty([kvh,4,n,total],(Kind::Float,q.device()));
    for g in 0..4 {scores.select(1,g).copy_(&(bmm_fp32(&q.select(1,g),&k.transpose(1,2))/128f64.sqrt()));}
    let prob=scores.masked_fill(&visible.logical_not().view([1,1,n,total]),f64::NEG_INFINITY).softmax(-1,Kind::Float);
    let out=Tensor::empty([kvh,4,n,128],(Kind::Float,q.device()));
    for g in 0..4 {out.select(1,g).copy_(&bmm_fp32(&prob.select(1,g),&v));}
    out.view([qh,n,128]).transpose(0,1).reshape([n,qh*128]).to_kind(Kind::BFloat16)
}
fn attention_oracle(q:&Tensor,ck:&Tensor,cv:&Tensor,k:&Tensor,v:&Tensor,visible:&Tensor)->Tensor {
    let n=q.size()[0];let qh=q.size()[1];
    let q=q.transpose(0,1).to_kind(Kind::Double);
    let k=Tensor::cat(&[ck,k],0).repeat_interleave_self_int(4,1,None).transpose(0,1).to_kind(Kind::Double);
    let v=Tensor::cat(&[cv,v],0).repeat_interleave_self_int(4,1,None).transpose(0,1).to_kind(Kind::Double);
    let prob=(q.matmul(&k.transpose(1,2))/128f64.sqrt()).masked_fill(&visible.logical_not().unsqueeze(0),f64::NEG_INFINITY).softmax(-1,Kind::Double);
    prob.matmul(&v).transpose(0,1).reshape([n,qh*128]).to_kind(Kind::BFloat16)
}

pub(crate) fn attention_fused(q:&Tensor,ck:&Tensor,cv:&Tensor,k:&Tensor,v:&Tensor)->Tensor {
    for t in [q,ck,cv,k,v]{assert!(t.is_contiguous());assert_eq!(t.kind(),Kind::BFloat16);}
    let n=q.size()[0];let history=ck.size()[0];let (qh,kvh)=(q.size()[1],k.size()[1]);assert_eq!(qh,4*kvh);
    let out=Tensor::empty([n,qh*128],(Kind::BFloat16,q.device()));
    extern "C" {fn rs_draft_attention(q:*const std::ffi::c_void,ck:*const std::ffi::c_void,cv:*const std::ffi::c_void,k:*const std::ffi::c_void,v:*const std::ffi::c_void,out:*mut std::ffi::c_void,history:i32,n:i32,kvh:i32)->i32;}
    assert_eq!(unsafe{rs_draft_attention(q.data_ptr(),ck.data_ptr(),cv.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),history as i32,n as i32,kvh as i32)},0);out
}

struct Conv {base:Tensor,projection:Tensor}
impl Conv {
    fn convolve(&self,x:&Tensor,delta:&Tensor,side:i64,dispatch:&mut crate::draft_conv::Dispatch)->Tensor {
        if let Some(out)=dispatch.try_convolve(x,delta,&self.base,side){return out;}
        let n=x.size()[0];let blocks=x.view([n,256,16]);
        let coefficients=self.base.get(side).view([1,2,256,16])+delta.unsqueeze(-1);
        let shifted=Tensor::cat(&[Tensor::zeros([1,256,16],(x.kind(),x.device())),blocks.narrow(0,0,n-1)],0);
        (coefficients.select(1,0)*blocks+coefficients.select(1,1)*shifted).view([n,4096])
    }
    fn prepare(&self,x:&Tensor,dispatch:&mut crate::draft_conv::Dispatch)->(Tensor,Tensor) {
        let delta=linear(x,&self.projection).view([x.size()[0],2,2,256]);
        (self.convolve(x,&delta.select(1,0),0,dispatch),delta.select(1,1))
    }
}
struct Layer {q:Tensor,k:Tensor,v:Tensor,o:Tensor,qn:Norm,kn:Norm,
    input_norm:Norm,post_norm:Norm,gate:Tensor,up:Tensor,down:Tensor,attn_conv:Conv,mlp_conv:Conv}

pub struct Context {pub len:i64,pub start:i64,kv:Vec<(Tensor,Tensor)>,storage:Option<Vec<(Tensor,Tensor)>>,cursor:i64,slot:Option<std::rc::Rc<DraftSlot>>}
impl Drop for Context {fn drop(&mut self){if let Some(s)=&self.slot{s.busy.set(false);}}}
impl Context {
    pub fn equal(&self,other:&Self)->bool {
        self.len==other.len && self.start==other.start && self.kv.len()==other.kv.len() &&
            self.kv.iter().zip(&other.kv).all(|((a,b),(c,d))|a.equal(c)&&b.equal(d))
    }

    /// Resident bytes of the visible drafter KV (checkpoint-library accounting).
    pub fn kv_bytes(&self)->i64 {self.kv.iter().map(|(k,v)|(k.numel()+v.numel()) as i64*2).sum()}
    /// Cache entries never alias mutable KV storage with a continuation.
    pub fn snapshot(&self)->Self {
        Self{len:self.len,start:self.start,kv:self.kv.iter().map(|(k,v)|(k.copy(),v.copy())).collect(),storage:None,cursor:0,slot:None}
    }
    /// Prefix cache (crate::pcache): the visible K/V per layer, and a detached context (like `snapshot`) rebuilt from them.
    pub(crate) fn kv(&self)->&[(Tensor,Tensor)] {&self.kv}
    pub(crate) fn from_parts(len:i64,start:i64,kv:Vec<(Tensor,Tensor)>)->Self {Self{len,start,kv,storage:None,cursor:0,slot:None}}
}
fn mlp_tp()->bool {crate::tp::is_tp() && std::env::var("GLM53_DRAFT_MLP_TP").as_deref()==Ok("1")}
fn head_tp()->bool {crate::tp::is_tp() && std::env::var("GLM53_DRAFT_HEAD_TP").as_deref()==Ok("1")}
fn topk_tp()->bool {std::env::var("GLM53_DRAFT_TOPK_TP").as_deref()==Ok("1")}
fn buffered()->bool {std::env::var("GLM53_DRAFT_KV_BUFFER").as_deref()==Ok("1")}
fn buffer_min_context()->i64 {
    let n=std::env::var("GLM53_DRAFT_KV_BUFFER_MIN_CONTEXT").ok().map(|v|v.parse::<i64>().unwrap()).unwrap_or(0);
    assert!(n>=0);n
}


/// W08: one fixed-address KV slab shared by successive request contexts, so a captured
/// proposal graph stays valid across requests. Only one live context may own it.
pub(crate) fn conf_tau()->Option<f64> {
    if std::env::var("GLM53_DRAFT_CONF_TRUNC").as_deref()!=Ok("1"){return None;}
    if let Some(p)=std::env::var("GLM53_SPEC_CONF_TAU_PCT").ok().and_then(|v|v.parse::<f64>().ok()){return Some(p/100.);}
    Some(std::env::var("GLM53_SPEC_CONF_TAU").ok().and_then(|v|v.parse::<f64>().ok()).unwrap_or(0.7))
}
fn path_conf(unary:&Tensor,ids:&Tensor,path:&Tensor)->Tensor {
    let p=unary.softmax(-1,Kind::Float);let hit=ids.eq_tensor(&path.view([-1,1])).to_kind(Kind::Float);
    (p*hit).sum_dim_intlist(&[-1i64][..],false,Kind::Float)
}
pub(crate) fn graph_enabled()->bool {std::env::var("GLM53_DRAFT_GRAPH").as_deref()==Ok("1")}
pub(crate) struct DraftSlot {k:Vec<Tensor>,v:Vec<Tensor>,busy:std::cell::Cell<bool>,meta:Tensor,ids:Tensor,
    part_o:Tensor,part_ml:Tensor,graph:std::cell::RefCell<Vec<ProposeGraph>>,warmed:std::cell::Cell<bool>,
    feat:Tensor,ameta:Tensor,dtemp:Tensor,dkeys:Tensor,appends:std::cell::RefCell<Vec<Vec<(crate::tp::graph::Owned,String)>>>,append_warmed:std::cell::RefCell<Vec<bool>>}
#[allow(clippy::type_complexity)]
struct ManyGraph {graph:crate::tp::graph::Owned,key:Vec<usize>,signature:String,_slots:Vec<std::rc::Rc<DraftSlot>>,
    outs:Vec<(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,bool)>,paths:Tensor,confs:Tensor,cc:usize,cf:usize,used:u64}
thread_local!{static MANY_GRAPHS:std::cell::RefCell<Vec<ManyGraph>>=const{std::cell::RefCell::new(Vec::new())};}
struct ProposeGraph {_graph:crate::tp::graph::Owned,signature:String,ids:Tensor,unary:Tensor,edges:Tensor,path:Tensor,hidden:Tensor,conf:Tensor,
    selector_fused:bool,final_norm_selected:bool,conv_calls:usize,conv_fused_calls:usize}
fn draft_signature()->String {
    let mut v:Vec<(String,String)>=std::env::vars().filter(|(k,_)|k.starts_with("GLM53_DRAFT_")||k=="GLM53_FP8_SKINNY"||k=="GLM53_RDMA_AR"||k=="GLM53_TP_SMALL_COMM"||k=="GLM53_TP_SMALL_COMM_ACTIVE").collect();
    v.sort();format!("{v:?}")
}
fn draft_attention_dev(q:&Tensor,sk:&Tensor,sv:&Tensor,k:&Tensor,v:&Tensor,slot:&DraftSlot)->Tensor {
    for t in [q,sk,sv,k,v]{assert!(t.is_contiguous());assert_eq!(t.kind(),Kind::BFloat16);}
    let (qh,kvh)=(q.size()[1],k.size()[1]);assert_eq!(q.size(),[8,qh,128]);assert_eq!(k.size(),[8,kvh,128]);assert_eq!(qh,4*kvh);
    let out=Tensor::empty([8,qh*128],(Kind::BFloat16,q.device()));
    extern "C"{fn rs_draft_attn_dev(q:*const std::ffi::c_void,sk:*const std::ffi::c_void,sv:*const std::ffi::c_void,nk:*const std::ffi::c_void,nv:*const std::ffi::c_void,
        meta:*const i64,part_o:*mut f32,part_ml:*mut f32,out:*mut std::ffi::c_void,kvh:i32)->i32;}
    assert_eq!(unsafe{rs_draft_attn_dev(q.data_ptr(),sk.data_ptr(),sv.data_ptr(),k.data_ptr(),v.data_ptr(),slot.meta.data_ptr().cast(),
        slot.part_o.data_ptr().cast(),slot.part_ml.data_ptr().cast(),out.data_ptr(),kvh as i32)},0,"draft device attention");out
}
pub struct Candidates {pub ids:Tensor,pub unary:Tensor,pub edges:Tensor,pub path:Vec<i64>,pub hidden:Tensor,
    /// Softmax (over the top-16) confidence of each chosen path token; filled when requested.
    pub conf:Option<Vec<f32>>,
    /// True only when this proposal's path used the validated fused FFI path.
    pub selector_fused:bool,
    /// True only when the selected-row terminal norm returned this proposal hidden.
    pub final_norm_selected:bool,
    /// Counts from this proposal's actual Conv dispatches, not environment flags.
    pub conv_calls:usize,pub conv_fused_calls:usize}
pub struct Drafter {layers:Vec<Layer>,fc:Tensor,hidden_norm:Norm,final_norm:Norm,
    predecessor:Tensor,successor:Tensor,selector:Tensor,lm_head:Tensor,head_shard:Option<(i64,i64)>,mlp_sharded:bool,position_ids:Tensor,rotary_inv:Tensor,pub mask_id:i64,pub block_size:i64,slot:std::cell::RefCell<Vec<std::rc::Rc<DraftSlot>>>,
    /// C3: query / KV heads held by this rank (32/8 full; 16/4 with GLM53_DRAFT_ATTN_TP=1 on TP2).
    qh:i64,kvh:i64,attn_tp:bool}

impl Drafter {
    pub(crate) fn final_norm_select_check(&self,out:&Path) {
        crate::draft_final_norm::norm_check(|sum|norm(sum,&self.final_norm),self.fc.device(),out);
    }
    pub fn load_target(dir:&Path,target:&crate::weights::ModelWeights)->Self {
        // Separate from the runtime A/B flag so a resident full head can qualify
        // both paths. Production can omit the replicated head entirely.
        let mut draft=if std::env::var("GLM53_DRAFT_HEAD_SHARD_LOAD").as_deref()==Ok("1") {
            assert!(head_tp(), "sharded drafter head requires TP execution");
            let shard=target.vocab_shard.expect("sharded drafter head requires vocab TP");
            let mut draft=Self::load(dir,&target.lm_head);draft.head_shard=Some(shard);draft
        } else {Self::load(dir,&target.drafter_head())};
        if std::env::var("GLM53_DRAFT_MLP_SHARD_LOAD").as_deref()==Ok("1") {
            assert!(mlp_tp());let tp=crate::tp::world();
            for layer in &mut draft.layers {
                let size=layer.gate.size()[0];assert_eq!(size%tp.world as i64,0);
                let n=size/tp.world as i64;let start=n*tp.rank as i64;
                layer.gate=layer.gate.narrow(0,start,n).copy();layer.up=layer.up.narrow(0,start,n).copy();
                layer.down=layer.down.narrow(1,start,n).copy();
            }
            draft.mlp_sharded=true;
        }
        // C3 (GLM53_DRAFT_ATTN_TP=1, TP2): attention head-parallel. q/k/v keep this rank's heads (rows), o its
        // input columns; the o projection's FP32 partials are summed across ranks. Slot KV holds local heads.
        if std::env::var("GLM53_DRAFT_ATTN_TP").as_deref()==Ok("1") {
            let tp=crate::tp::world();assert!(tp.world==2,"drafter attention TP is TP2 only");
            let (qh,kvh)=(32/tp.world as i64,8/tp.world as i64);
            for layer in &mut draft.layers {
                layer.q=layer.q.narrow(0,tp.rank as i64*qh*128,qh*128).copy();
                layer.k=layer.k.narrow(0,tp.rank as i64*kvh*128,kvh*128).copy();layer.v=layer.v.narrow(0,tp.rank as i64*kvh*128,kvh*128).copy();
                layer.o=layer.o.narrow(1,tp.rank as i64*qh*128,qh*128).copy();
            }
            draft.qh=qh;draft.kvh=kvh;draft.attn_tp=true;
        }
        for layer in &draft.layers {
            for w in [&layer.gate,&layer.up,&layer.down]{crate::dense_fp8::register_weight(w,"GLM53_DRAFT_FP8_MLP");}
            for w in [&layer.q,&layer.k,&layer.v,&layer.o]{crate::dense_fp8::register_weight(w,"GLM53_DRAFT_FP8_ATTN");}
            for w in [&layer.attn_conv.projection,&layer.mlp_conv.projection]{crate::dense_fp8::register_weight(w,"GLM53_DRAFT_FP8_CONV");}
        }
        crate::dense_fp8::register_weight(&draft.fc,"GLM53_DRAFT_FP8_FC");
        crate::dense_fp8::register_weight(&draft.lm_head,"GLM53_DRAFT_FP8_HEAD");
        if crate::dense_fp8::free_sources_enabled() {
            let take=|t:&mut Tensor|{if let Some(ph)=crate::dense_fp8::compact(t){*t=ph;}};
            for layer in &mut draft.layers {
                for t in [&mut layer.gate,&mut layer.up,&mut layer.down,&mut layer.q,&mut layer.k,&mut layer.v,&mut layer.o,
                          &mut layer.attn_conv.projection,&mut layer.mlp_conv.projection]{take(t);}
            }
            take(&mut draft.fc);take(&mut draft.lm_head);
        }
        crate::weights::release_load_buffer();
        crate::host_memory::finish_loading("drafter-load");
        draft.prepare_head_int4();
        draft
    }
    /// Attention output projection; with head-parallel attention the rank partials are summed in FP32 first.
    fn o_proj(&self,x:&Tensor,w:&Tensor)->Tensor {
        if !self.attn_tp {return linear(x,w);}
        let y=int4_linear(x,w,true).or_else(||crate::dense_fp8::try_bf16(x,w,true)).unwrap_or_else(||x.to_kind(Kind::BFloat16).matmul(&w.transpose(0,1)).to_kind(Kind::Float));
        crate::tp::allreduce(&y);y.to_kind(Kind::BFloat16)
    }
    fn mlp(&self,layer:&Layer,z:&Tensor)->Tensor {
        if mlp_tp() {
            let tp=crate::tp::world();
            let (gate,up,down)=if self.mlp_sharded {
                (layer.gate.shallow_clone(),layer.up.shallow_clone(),layer.down.shallow_clone())
            }else {
                let n=layer.gate.size()[0]/tp.world as i64;let start=n*tp.rank as i64;
                (layer.gate.narrow(0,start,n),layer.up.narrow(0,start,n),layer.down.narrow(1,start,n))
            };
            let activation=if fused_norm() {
                let (g,u)=(linear(z,&gate).to_kind(Kind::BFloat16).contiguous(),linear(z,&up).to_kind(Kind::BFloat16).contiguous());
                let out=Tensor::empty_like(&g);
                extern "C"{fn rs_draft_silu_mul(g:*const std::ffi::c_void,u:*const std::ffi::c_void,out:*mut std::ffi::c_void,n:i64)->i32;}
                assert_eq!(unsafe{rs_draft_silu_mul(g.data_ptr(),u.data_ptr(),out.data_ptr(),g.numel() as i64)},0);out
            } else {(linear(z,&gate).to_kind(Kind::Float).silu()*linear(z,&up).to_kind(Kind::Float)).to_kind(Kind::BFloat16).contiguous()};
            let out=if let Some(y)=crate::dense_fp8::try_bf16(&activation,&down,true){y}else{
            let out=Tensor::empty([z.size()[0],down.size()[0]],(Kind::Float,z.device()));
            extern "C" {fn rs_bf16_partial(x:*const u8,w:*const u8,y:*mut f32,m:i32,n:i32,k:i32,ldw:i32)->i32;}
            assert_eq!(unsafe{rs_bf16_partial(activation.data_ptr().cast(),down.data_ptr().cast(),out.data_ptr().cast(),
                z.size()[0] as i32,down.size()[0] as i32,down.size()[1] as i32,down.stride()[0] as i32)},0);out};
            crate::tp::allreduce(&out);out.to_kind(Kind::BFloat16)
        }else {
            assert!(!self.mlp_sharded,"cannot disable MLP TP after sharded load");
            let activation=(linear(z,&layer.gate).to_kind(Kind::Float).silu()*linear(z,&layer.up).to_kind(Kind::Float)).to_kind(Kind::BFloat16);
            linear(&activation,&layer.down)
        }
    }

    fn local_logits(&self,hidden:&Tensor)->(i64,i64,Tensor) {
            assert!(head_tp(),"distributed candidates require head TP");
            let tp=crate::tp::world();
            let (start,total,weight)=if let Some((start,total))=self.head_shard {
                (start,total,self.lm_head.shallow_clone())
            } else {
                let total=self.lm_head.size()[0];assert_eq!(total%tp.world as i64,0);
                let count=total/tp.world as i64;let start=count*tp.rank as i64;
                (start,total,self.lm_head.narrow(0,start,count))
            };
            let local=linear(hidden,&weight).to_kind(Kind::Float);
            (start,total,local)
    }
    /// GLM53_DRAFT_HEAD_INT4=1 (draft side, L2): coarse INT4 scores of this rank's head slice preselect 64 candidates
    /// per row; those rows of the FP8 head are rescored with the same kernel and BF16 output boundary as the full
    /// pass (per-row values do not depend on the row count), and the top 16 of them are returned. Target outputs do not
    /// depend on the drafts (exact speculative sampling), only acceptance can move.
    fn topk_local_int4(&self,hidden:&Tensor)->Option<(i64,i64,Tensor,Tensor)> {
        if std::env::var("GLM53_DRAFT_HEAD_INT4").as_deref()!=Ok("1") {return None;}
        let tp=crate::tp::world();
        let (start,total,weight)=if let Some((start,total))=self.head_shard {(start,total,self.lm_head.shallow_clone())} else {
            let total=self.lm_head.size()[0];let count=total/tp.world as i64;let start=count*tp.rank as i64;(start,total,self.lm_head.narrow(0,start,count))};
        let (q,s)=crate::dense_fp8::quant_of_bf16(&weight)?;
        let (n,k)=(q.size()[0],q.size()[1]);let m=hidden.size()[0];
        if !(1..=8).contains(&m) || k%1024!=0 || hidden.size()[1]!=k {return None;}
        let (q4,s4)=head4_of(&q,&s)?;
        let x=hidden.to_kind(Kind::BFloat16).contiguous();
        let coarse=Tensor::empty([m,n],(Kind::Float,x.device()));
        extern "C"{fn rs_draft_head_int4(x:*const std::ffi::c_void,q4:*const std::ffi::c_void,s:*const f32,y:*mut f32,m:i32,n:i32,k:i32)->i32;}
        assert_eq!(unsafe{rs_draft_head_int4(x.data_ptr(),q4.data_ptr(),s4.data_ptr().cast(),coarse.data_ptr().cast(),m as i32,n as i32,k as i32)},0,"INT4 draft head");
        let (_,cand)=coarse.topk(64,-1,true,false);
        let flat=cand.reshape([-1]);
        let wq=q.index_select(0,&flat);let ws=s.index_select(0,&flat);
        let y=crate::dense_fp8::run_bf16(&x,&wq,&ws,false).to_kind(Kind::Float);       // [m, m*64]
        let own=y.view([m,m,64]).diagonal(0,0,1).transpose(0,1);                      // [m, 64]: row r's own candidates
        let (values,pos)=own.topk(16,-1,true,true);
        Some((start,total,values,cand.gather(-1,&pos,false)))
    }
    /// Builds the INT4 head copy at load (never during graph capture).
    fn prepare_layers_int4(&self) {
        if std::env::var("GLM53_DRAFT_INT4_LAYERS").as_deref()!=Ok("1") {return;}
        let mut n=0;
        for l in &self.layers {for w in [&l.q,&l.k,&l.v,&l.o,&l.gate,&l.up,&l.down,&l.attn_conv.projection,&l.mlp_conv.projection] {
            let k=w.size()[1];if k%1024!=0||k>4096 {continue;}
            if let Some((q,s))=crate::dense_fp8::quant_of_bf16(w) {if q.size()[0]%8==0 && head4_of(&q,&s).is_some(){n+=1;}}}}
        eprintln!("[draft] INT4 layer copies ready: {n} projections");
    }
    pub(crate) fn prepare_head_int4(&self) {
        self.prepare_layers_int4();
        if std::env::var("GLM53_DRAFT_HEAD_INT4").as_deref()!=Ok("1") || !topk_tp() {return;}
        let tp=crate::tp::world();
        let weight=if self.head_shard.is_some() {self.lm_head.shallow_clone()} else {
            let count=self.lm_head.size()[0]/tp.world as i64;self.lm_head.narrow(0,count*tp.rank as i64,count)};
        if let Some((q,s))=crate::dense_fp8::quant_of_bf16(&weight) {let _=head4_of(&q,&s);eprintln!("[draft] INT4 head copy ready: {:?}",q.size());}
    }
}
fn head4_of(q:&Tensor,s:&Tensor)->Option<(Tensor,Tensor)> {
        let (n,k)=(q.size()[0],q.size()[1]);
        let key=q.data_ptr() as usize;
        let cached=HEAD4.with(|c|c.borrow().get(&key).map(|(a,b)|(a.shallow_clone(),b.shallow_clone())));
        let (q4,s4)=match cached {Some(v)=>v,None=>{
            if crate::tp::graph::capturing() {return None;}
            let _g=tch::no_grad_guard();let dev=q.device();
            let q4=Tensor::empty([n,k/2],(Kind::Uint8,dev));let s4=Tensor::empty([n,k/128],(Kind::Float,dev));
            for r0 in (0..n).step_by(8192) {
                let rows=(n-r0).min(8192);
                let wf=(q.narrow(0,r0,rows).to_kind(Kind::Float)*s.narrow(0,r0,rows).unsqueeze(1)).view([rows,k/128,128]);
                let sc=(wf.abs().amax([-1],false)/7.).clamp_min(1e-12);
                let qi=((wf/sc.unsqueeze(-1)).round().clamp(-8.,7.)+8.).to_kind(Kind::Uint8).view([rows,k/2,2]);
                let packed=qi.select(-1,0).bitwise_or_tensor(&(qi.select(-1,1)*16));
                q4.narrow(0,r0,rows).copy_(&packed);s4.narrow(0,r0,rows).copy_(&sc);
            }
            // GLM53_DRAFT_HEAD4_TILED=1 (draft side, L0): the MMA head kernel reads a tiled copy (C side no-op when off).
            extern "C"{fn rs_draft_head4_tile(q4:*mut std::ffi::c_void,n:i32,k:i32)->i32;}
            assert_eq!(unsafe{rs_draft_head4_tile(q4.data_ptr(),n as i32,k as i32)},0,"INT4 head retile");
            HEAD4.with(|c|c.borrow_mut().insert(key,(q4.shallow_clone(),s4.shallow_clone())));(q4,s4)}};
        Some((q4,s4))
}
impl Drafter {
    fn topk(&self,hidden:&Tensor)->(Tensor,Tensor) {
        if !topk_tp(){return self.logits(hidden).topk(16,-1,true,true);}
        let tp=crate::tp::world();
        let (start,total,values,ids)=match self.topk_local_int4(hidden) {Some(v)=>v,None=>{
            let (start,total,local)=self.local_logits(hidden);let (v,i)=local.topk(16,-1,true,true);(start,total,v,i)}};
        assert!(total<(1<<24),"FP32 candidate IDs must be exactly representable");
        let rows=hidden.size()[0];
        // Disjoint lanes preserve scores exactly in a SUM. Only 2*7*16*2
        // FP32 values cross the fabric in TP2, instead of 7*154880 logits.
        let packed=Tensor::zeros([tp.world as i64,rows,16,2],(Kind::Float,hidden.device()));
        let lane=packed.get(tp.rank as i64);lane.select(-1,0).copy_(&values);
        lane.select(-1,1).copy_(&(ids+start).to_kind(Kind::Float));
        crate::tp::allreduce(&packed);
        let candidates=packed.permute([1,0,2,3]).reshape([rows,tp.world as i64*16,2]);
        let (scores,indices)=candidates.select(-1,0).topk(16,-1,true,true);
        let ids=candidates.select(-1,1).gather(-1,&indices,false).to_kind(Kind::Int64);
        // Equal-score membership/order can differ from a monolithic topk;
        // selector paths and whole-request acceptance require their own gate.
        (scores,ids)
    }
    fn logits(&self,hidden:&Tensor)->Tensor {
        if head_tp() {
            let (start,total,local)=self.local_logits(hidden);
            let result=Tensor::zeros([hidden.size()[0],total],(Kind::Float,hidden.device()));
            result.narrow(1,start,local.size()[1]).copy_(&local);
            crate::tp::allreduce(&result);result
        } else {
            assert!(self.head_shard.is_none(),"cannot disable TP after loading sharded head");
            linear(hidden,&self.lm_head).to_kind(Kind::Float)
        }
    }
    pub fn load(dir:&Path,target_head:&Tensor)->Self {
        let cfg:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        assert_eq!(cfg["hidden_size"],4096);assert_eq!(cfg["num_hidden_layers"],5);
        assert_eq!(cfg["dflash_config"]["block_size"],8);assert_eq!(cfg["is_causal"],false);
        let mut idx=crate::safetensors::ShardIndex::scan(dir).unwrap();let dev=target_head.device();
        let mut get=|name:&str|{
            // GLM53_FAST_LOAD: a BF16 checkpoint tensor goes up as stored (BF16 -> F32 -> BF16 is the identity).
            let old=|idx:&crate::safetensors::ShardIndex|{let (v,shape)=idx.get_f32(name).unwrap_or_else(|e|panic!("DFlash {name}: {e}"));
                Tensor::from_slice(&v).view(shape.iter().map(|&v|v as i64).collect::<Vec<_>>().as_slice()).to_kind(Kind::BFloat16).to_device(dev)};
            if crate::weights::fast_load_enabled() && dev.is_cuda() && idx.entries.get(name).is_some_and(|e|e.dtype=="BF16") {
                let t=crate::weights::upload_stored(&idx,name,dev);
                if crate::weights::load_check_enabled() {
                    assert!(old(&idx).view_dtype(Kind::Int16).equal(&t.view_dtype(Kind::Int16)),"GLM53_LOAD_CHECK: DFlash {name} differs from the old load");
                }
                return t;
            }
            old(&idx)};
        let mut layers=Vec::new();
        for i in 0..5 {
            let p=format!("layers.{i}");
            layers.push(Layer{q:get(&format!("{p}.self_attn.q_proj.weight")),k:get(&format!("{p}.self_attn.k_proj.weight")),
                v:get(&format!("{p}.self_attn.v_proj.weight")),o:get(&format!("{p}.self_attn.o_proj.weight")),
                qn:Norm::new(get(&format!("{p}.self_attn.q_norm.weight"))),kn:Norm::new(get(&format!("{p}.self_attn.k_norm.weight"))),
                input_norm:Norm::new(get(&format!("{p}.input_layernorm.weight"))),post_norm:Norm::new(get(&format!("{p}.post_attention_layernorm.weight"))),
                gate:get(&format!("{p}.mlp.gate_proj.weight")),up:get(&format!("{p}.mlp.up_proj.weight")),down:get(&format!("{p}.mlp.down_proj.weight")),
                attn_conv:Conv{base:get(&format!("{p}.attention_conv.base_kernel")),projection:get(&format!("{p}.attention_conv.kernel_projection.weight"))},
                mlp_conv:Conv{base:get(&format!("{p}.mlp_conv.base_kernel")),projection:get(&format!("{p}.mlp_conv.kernel_projection.weight"))}});
        }
        Self{layers,fc:get("fc.weight"),hidden_norm:Norm::new(get("hidden_norm.weight")),final_norm:Norm::new(get("norm.weight")),
            predecessor:get("candidate_selector.predecessor_codebook"),successor:get("candidate_selector.successor_codebook"),
            selector:get("candidate_selector.hidden_projection.weight"),lm_head:target_head.to_kind(Kind::BFloat16),
            head_shard:None,mlp_sharded:false,position_ids:Tensor::arange(crate::mla_latent::capacity().max(2056),(Kind::Int64,dev)),
            rotary_inv:(Tensor::arange(64,(Kind::Float,dev))*(-10000f64.ln()/64.)).exp(),mask_id:154856,block_size:8,slot:std::cell::RefCell::new(Vec::new()),qh:32,kvh:8,attn_tp:false}
    }
    fn positions(&self,n:i64)->Tensor {
        if rope_cache() && n<=self.position_ids.size()[0] {self.position_ids.narrow(0,0,n)}
        else {Tensor::arange(n,(Kind::Int64,self.fc.device()))}
    }
    pub fn empty_context(&self)->Context {
        Context{len:0,start:0,kv:self.layers.iter().map(|_|{
            let z=||Tensor::zeros([0,self.kvh,128],(Kind::BFloat16,self.fc.device()));(z(),z())}).collect(),storage:None,cursor:0,slot:None}
    }
    fn append_body(&self,slot:&DraftSlot,n:i64) {
        let x=norm(&linear(&slot.feat.narrow(0,0,n),&self.fc),&self.hidden_norm);
        let positions=self.positions(n)+slot.ameta.narrow(0,0,1);
        let rotary=rope_cache().then(||rope_tables(&positions,Some(&self.rotary_inv)));
        let rows=self.positions(n)+slot.ameta.narrow(0,1,1);
        for (i,layer) in self.layers.iter().enumerate() {
            let k=if fused_norm(){head_norm_rope(&linear(&x,&layer.k).view([n,self.kvh,128]),&layer.kn,&positions,&self.rotary_inv)}
                else{rope(&norm(&linear(&x,&layer.k).view([n,self.kvh,128]),&layer.kn),&positions,rotary.as_ref())};
            let v=linear(&x,&layer.v).view([n,self.kvh,128]);
            let _=slot.k[i].shallow_clone().index_copy_(0,&rows,&k);let _=slot.v[i].shallow_clone().index_copy_(0,&rows,&v);
        }
    }
    /// W08: steady-state append (1..8 rows) replays a per-row-count graph writing straight into
    /// the slot slab; compaction and host view bookkeeping stay outside the graph.
    fn append_graph(&self,context:&mut Context,features:&Tensor)->bool {
        let Some(slot)=context.slot.clone() else {return false};
        let n=features.size()[0];
        if !(1..=8).contains(&n) || context.storage.is_none() || features.kind()!=Kind::Float || context.kv.is_empty() {return false;}
        let old_len=context.len-context.start;
        let compact=context.cursor+n>4096;
        let offset=if compact {old_len} else {context.cursor};
        if compact {
            for ((a,b),(old_k,old_v)) in context.storage.as_ref().unwrap().iter().zip(&context.kv) {
                a.narrow(0,0,old_len).copy_(&old_k.copy());b.narrow(0,0,old_len).copy_(&old_v.copy());
            }
        }
        // Replayed index_copy_ writes slab rows [offset, offset+n): they must stay inside the slab, and the
        // context's visible views must be this slot's rows ending at cursor.
        let esz=slot.k[0].size()[1]*128*2;let view=&context.kv[0].0;
        let vbegin=(view.data_ptr() as i64-slot.k[0].data_ptr() as i64)/esz;
        if std::env::var("GLM53_DRAFT_META_LOG").as_deref()==Ok("1") {
            eprintln!("[draft-append] slot {:p} len {} start {} cursor {} n {} compact {} offset {} old_len {} view_begin {} view_rows {}",std::rc::Rc::as_ptr(&slot),context.len,context.start,context.cursor,n,compact,offset,old_len,vbegin,view.size()[0]);
        }
        assert!(offset>=0 && offset+n<=4096 && old_len==view.size()[0],"draft append out of slab: offset {offset} n {n} old_len {old_len} view {}",view.size()[0]);
        assert!(compact || vbegin+view.size()[0]==context.cursor,"draft context views not at cursor: begin {vbegin} rows {} cursor {}",view.size()[0],context.cursor);
        slot.feat.narrow(0,0,n).copy_(features);
        crate::tp::upload(&slot.ameta,&[context.len,offset]);
        let signature=draft_signature();let idx=(n-1) as usize;
        let found=slot.appends.borrow()[idx].iter().any(|(_,s)|*s==signature);
        if !found {
            if slot.appends.borrow()[idx].len()>=4 {slot.appends.borrow_mut()[idx].remove(0);}
            if !slot.append_warmed.borrow()[idx]{self.append_body(&slot,n);slot.append_warmed.borrow_mut()[idx]=true;}
            tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();self.append_body(&slot,n);crate::tp::graph::end().unwrap();
            slot.appends.borrow_mut()[idx].push((crate::tp::graph::Owned::take(),signature.clone()));
        }
        slot.appends.borrow()[idx].iter().find(|(_,s)|*s==signature).unwrap().0.replay();
        let end=offset+n;let keep=(old_len+n).min(2048);
        for ((a,b),(old_k,old_v)) in context.storage.as_ref().unwrap().iter().zip(&mut context.kv) {
            *old_k=a.narrow(0,end-keep,keep);*old_v=b.narrow(0,end-keep,keep);
        }
        context.cursor=end;context.len+=n;context.start=(context.len-2048).max(0);
        true
    }
    /// C2 (GLM53_SERVE_DRAFT_BATCH=1): steady-state appends of several slot-backed contexts (1..8 rows each)
    /// in one forward: the fc projection and each layer's K/V projections read their weights once; RoPE uses
    /// each context's own positions and every context writes its own slab rows. Slab bookkeeping (compaction,
    /// views, cursor) is exactly append_graph's. Contexts that do not qualify fall back to append().
    pub fn append_many(&self,items:&mut [(&mut Context,Tensor)]) {
        let qualifies=|c:&Context,f:&Tensor|{let n=f.size()[0];c.slot.is_some() && (1..=8).contains(&n) && c.storage.is_some() && f.kind()==Kind::Float
            && !c.kv.is_empty() && f.size()[1]==20480};
        let ok:Vec<bool>=items.iter().map(|(c,f)|qualifies(c,f)).collect();
        if ok.iter().filter(|&&b|b).count()<2 {for (c,f) in items.iter_mut() {self.append(c,f);}return;}
        // Host bookkeeping per context (as append_graph): compaction and the slab offset.
        let mut plan:Vec<(usize,i64,i64,i64)>=Vec::new();   // (item, n, offset, len before)
        for (idx,(context,features)) in items.iter_mut().enumerate() {
            if !ok[idx] {continue;}
            let slot=context.slot.clone().unwrap();let n=features.size()[0];
            let old_len=context.len-context.start;let compact=context.cursor+n>4096;let offset=if compact {old_len} else {context.cursor};
            if compact {for ((a,b),(old_k,old_v)) in context.storage.as_ref().unwrap().iter().zip(&context.kv) {
                a.narrow(0,0,old_len).copy_(&old_k.copy());b.narrow(0,0,old_len).copy_(&old_v.copy());}}
            let esz=slot.k[0].size()[1]*128*2;let view=&context.kv[0].0;let vbegin=(view.data_ptr() as i64-slot.k[0].data_ptr() as i64)/esz;
            assert!(offset>=0 && offset+n<=4096 && old_len==view.size()[0],"draft append out of slab");
            assert!(compact || vbegin+view.size()[0]==context.cursor,"draft context views not at cursor");
            plan.push((idx,n,offset,context.len));
        }
        let dev=self.fc.device();
        let feats=Tensor::cat(&plan.iter().map(|&(i,..)|items[i].1.shallow_clone()).collect::<Vec<_>>(),0);
        let total=feats.size()[0];
        let x=norm(&linear(&feats,&self.fc),&self.hidden_norm);
        let positions=Tensor::cat(&plan.iter().map(|&(_,n,_,len)|self.positions(n)+len).collect::<Vec<_>>(),0);
        let rotary=rope_cache().then(||rope_tables(&positions,Some(&self.rotary_inv)));
        let rows:Vec<Tensor>=plan.iter().map(|&(_,n,off,_)|Tensor::arange(n,(Kind::Int64,dev))+off).collect();
        for (li,layer) in self.layers.iter().enumerate() {
            let k=if fused_norm(){head_norm_rope(&linear(&x,&layer.k).view([total,self.kvh,128]),&layer.kn,&positions,&self.rotary_inv)}
                else{rope(&norm(&linear(&x,&layer.k).view([total,self.kvh,128]),&layer.kn),&positions,rotary.as_ref())};
            let v=linear(&x,&layer.v).view([total,self.kvh,128]);
            let mut first=0;
            for (p,&(i,n,..)) in plan.iter().enumerate() {
                let slot=items[i].0.slot.clone().unwrap();
                let _=slot.k[li].shallow_clone().index_copy_(0,&rows[p],&k.narrow(0,first,n));
                let _=slot.v[li].shallow_clone().index_copy_(0,&rows[p],&v.narrow(0,first,n));first+=n;
            }
        }
        for &(i,n,offset,_) in &plan {
            let context=&mut *items[i].0;let old_len=context.len-context.start;
            let end=offset+n;let keep=(old_len+n).min(2048);
            for ((a,b),(old_k,old_v)) in context.storage.as_ref().unwrap().iter().zip(&mut context.kv) {
                *old_k=a.narrow(0,end-keep,keep);*old_v=b.narrow(0,end-keep,keep);
            }
            context.cursor=end;context.len+=n;context.start=(context.len-2048).max(0);
        }
        for (idx,(c,f)) in items.iter_mut().enumerate() {if !ok[idx] {self.append(c,f);}}
    }
    pub fn append(&self,context:&mut Context,features:&Tensor) {
        assert_eq!(features.size()[1],20480);let n=features.size()[0];assert!(n>0);
        if graph_enabled() && std::env::var("GLM53_DRAFT_APPEND_GRAPH").as_deref()==Ok("1") && self.append_graph(context,features) {return;}
        let trim=std::env::var("GLM53_DRAFT_APPEND_TRIM").as_deref()==Ok("1");
        let replace_window=trim && n>=2048;
        let projected_n=if trim{n.min(2048)}else{n};
        let skipped=n-projected_n;
        let x=norm(&linear(&features.narrow(0,skipped,projected_n),&self.fc),&self.hidden_norm);
        let positions=self.positions(projected_n)+context.len+skipped;
        let rotary=rope_cache().then(||rope_tables(&positions,Some(&self.rotary_inv)));
        // Enable the slab at a measured context length; short requests can keep
        // compact storage. A boundary crossing initializes from the visible KV.
        let use_buffer=(buffered() && context.len+n>=buffer_min_context()) || (graph_enabled() && context.slot.is_some());
        let old_len=context.len-context.start;
        if graph_enabled() && context.storage.is_none() && context.slot.is_none() {
            if let Some(slot)=self.claim_slot(x.device()) {
                // Seed the fixed slab exactly as the lazily allocated buffer path would.
                if !replace_window {for ((a,b),(k,v)) in slot.k.iter().zip(&slot.v).zip(&context.kv) {a.narrow(0,0,old_len).copy_(k);b.narrow(0,0,old_len).copy_(v);}}
                context.storage=Some(slot.k.iter().zip(&slot.v).map(|(a,b)|(a.shallow_clone(),b.shallow_clone())).collect());
                context.cursor=if replace_window{0}else{old_len};context.slot=Some(slot);
            }
        }
        let use_buffer=use_buffer || context.slot.is_some();
        // Keep 2048 visible positions; amortize compaction using a 4096-row slab.
        // A restored prefix starts with owned compact KV and initializes lazily.
        if use_buffer && context.storage.is_none() {
            let storage:Vec<_>=context.kv.iter().map(|(k,v)|{
                let a=Tensor::empty([4096,self.kvh,128],(Kind::BFloat16,x.device()));
                let b=Tensor::empty_like(&a);
                if !replace_window{a.narrow(0,0,old_len).copy_(k);b.narrow(0,0,old_len).copy_(v);}(a,b)
            }).collect();context.storage=Some(storage);context.cursor=if replace_window{0}else{old_len};
        }
        let append_n=n.min(2048);
        let compact=use_buffer && !replace_window && context.cursor+append_n>4096;
        let offset=if replace_window{0}else if compact {old_len} else {context.cursor};
        for (i,(layer,(old_k,old_v))) in self.layers.iter().zip(&mut context.kv).enumerate() {
            let k=rope(&norm(&linear(&x,&layer.k).view([projected_n,self.kvh,128]),&layer.kn),&positions,rotary.as_ref());
            let v=linear(&x,&layer.v).view([projected_n,self.kvh,128]);
            if use_buffer {
                let (a,b)=&context.storage.as_ref().unwrap()[i];
                if compact {a.narrow(0,0,old_len).copy_(&old_k.copy());b.narrow(0,0,old_len).copy_(&old_v.copy());}
                a.narrow(0,offset,append_n).copy_(&k.narrow(0,projected_n-append_n,append_n));
                b.narrow(0,offset,append_n).copy_(&v.narrow(0,projected_n-append_n,append_n));
                let end=offset+append_n;let keep=(old_len+n).min(2048);
                *old_k=a.narrow(0,end-keep,keep);*old_v=b.narrow(0,end-keep,keep);
            } else if replace_window {
                // This append replaces the whole visible window: no old KV is
                // consumed, including by the next proposal or prefix snapshot.
                *old_k=k;*old_v=v;
            } else {
                let k=Tensor::cat(&[&*old_k,&k],0);let v=Tensor::cat(&[&*old_v,&v],0);
                let keep=k.size()[0].min(2048);let start=k.size()[0]-keep;
                *old_k=k.narrow(0,start,keep).contiguous();*old_v=v.narrow(0,start,keep).contiguous();
            }
        }
        if use_buffer {context.cursor=offset+append_n;}else{context.storage=None;context.cursor=0;}
        context.len+=n;context.start=(context.len-2048).max(0);
    }
    fn claim_slot(&self,dev:Device)->Option<std::rc::Rc<DraftSlot>> {
        // One fixed-address slot (KV slab + draft/append graphs) per concurrent context.
        let mut cell=self.slot.borrow_mut();
        if let Some(free)=cell.iter().find(|s|!s.busy.get()) {free.busy.set(true);return Some(free.clone());}
        let max=std::env::var("GLM53_DRAFT_SLOTS").ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(8);
        if cell.len()>=max {return None;}
        let slot={
            let mk=||Tensor::zeros([4096,self.kvh,128],(Kind::BFloat16,dev));
            std::rc::Rc::new(DraftSlot{k:(0..self.layers.len()).map(|_|mk()).collect(),v:(0..self.layers.len()).map(|_|mk()).collect(),
                busy:std::cell::Cell::new(false),meta:Tensor::zeros([4],(Kind::Int64,dev)),ids:Tensor::zeros([8],(Kind::Int64,dev)),
                part_o:Tensor::zeros([65*256*128],(Kind::Float,dev)),part_ml:Tensor::zeros([65*256*2],(Kind::Float,dev)),
                graph:std::cell::RefCell::new(Vec::new()),warmed:std::cell::Cell::new(false),
                feat:Tensor::zeros([8,20480],(Kind::Float,dev)),ameta:Tensor::zeros([2],(Kind::Int64,dev)),
                dtemp:Tensor::zeros([7],(Kind::Float,dev)),dkeys:Tensor::zeros([7],(Kind::Int64,dev)),
                appends:std::cell::RefCell::new((0..8).map(|_|Vec::new()).collect()),append_warmed:std::cell::RefCell::new(vec![false;8])})};
        slot.busy.set(true);cell.push(slot.clone());Some(slot)
    }
    /// Fixed-shape body: device-side anchor, positions and history range only.
    fn propose_body(&self,slot:&DraftSlot,target:&crate::weights::ModelWeights)->(Tensor,Tensor,Tensor,Tensor,Tensor,bool,bool,usize,usize) {
        let n=8i64;let dev=target.device;
        let mut x=target.embed_tokens(&slot.ids).to_kind(Kind::BFloat16);
        let positions=self.positions(n)+slot.meta.narrow(0,0,1);
        let rotary=Some(rope_tables(&positions,Some(&self.rotary_inv)));
        let mut residual:Option<Tensor>=None;let mut conv=crate::draft_conv::Dispatch::new();
        let fused=fused_norm();
        for (i,layer) in self.layers.iter().enumerate() {
            let (z,r)=if fused {let r=residual.take();add_norm_f(&x,r.as_ref(),&layer.input_norm)} else {match residual.take(){Some(r)=>add_norm(&x,&r,&layer.input_norm),None=>(norm(&x,&layer.input_norm),x.shallow_clone())}};
            let (z,delta)=layer.attn_conv.prepare(&z,&mut conv);
            let (q,k)=if fused {(head_norm_rope(&linear(&z,&layer.q).view([n,self.qh,128]),&layer.qn,&positions,&self.rotary_inv),
                                 head_norm_rope(&linear(&z,&layer.k).view([n,self.kvh,128]),&layer.kn,&positions,&self.rotary_inv))}
                else {(rope(&norm(&linear(&z,&layer.q).view([n,self.qh,128]),&layer.qn),&positions,rotary.as_ref()).contiguous(),
                       rope(&norm(&linear(&z,&layer.k).view([n,self.kvh,128]),&layer.kn),&positions,rotary.as_ref()).contiguous())};
            let v=linear(&z,&layer.v).view([n,self.kvh,128]).contiguous();
            let attention=draft_attention_dev(&q,&slot.k[i],&slot.v[i],&k,&v,slot);
            let attn=layer.attn_conv.convolve(&self.o_proj(&attention,&layer.o),&delta,1,&mut conv);
            let (z,r)=if fused {add_norm_f(&attn,Some(&r),&layer.post_norm)} else {add_norm(&attn,&r,&layer.post_norm)};let (z,delta)=layer.mlp_conv.prepare(&z,&mut conv);
            x=layer.mlp_conv.convolve(&self.mlp(layer,&z),&delta,1,&mut conv);residual=Some(r);
        }
        let residual=residual.unwrap();
        let (hidden,final_norm_selected)=if let Some(h)=crate::draft_final_norm::try_hidden(&x,&residual,|sum|norm(sum,&self.final_norm)) {
            (h,true)
        }else{(add_norm(&x,&residual,&self.final_norm).0.narrow(0,1,n-1),false)};
        let (unary,ids)=self.topk(&hidden);
        let projected=linear(&hidden,&self.selector);
        let successors=self.successor.index_select(0,&ids.reshape([-1])).view([n-1,16,256]);
        let predecessors=Tensor::cat(&[slot.ids.narrow(0,0,1).view([1,1]).expand([1,16],false),ids.narrow(0,0,n-2)],0);
        let predecessors=self.predecessor.index_select(0,&predecessors.reshape([-1])).view([n-1,16,256]);
        let mut edges=(predecessors*projected.unsqueeze(1)).bmm(&successors.transpose(1,2)).to_kind(Kind::Float)+unary.unsqueeze(1);
        if crate::sampling::coupled() {edges=edges+crate::sampling::candidate_noise(&ids,&slot.dtemp,&slot.dkeys).unsqueeze(1);}
        let (path,selector_fused)=if let Some(path)=crate::draft_selector::try_path(&edges,&ids) {(path,true)}else {
            let mut previous=Tensor::zeros([1],(Kind::Int64,dev));let mut tokens=Vec::new();
            for t in 0..n-1 {let best=edges.get(t).index_select(0,&previous).argmax(-1,false);tokens.push(ids.get(t).index_select(0,&best));previous=best;}
            (Tensor::cat(&tokens,0),false)
        };
        (ids,unary,edges,path,hidden,selector_fused,final_norm_selected,conv.calls,conv.fused)
    }
    fn propose_graph(&self,context:&Context,anchor:i64,target:&crate::weights::ModelWeights)->Option<Candidates> {
        let slot=context.slot.as_ref()?;
        if self.block_size!=8 || context.kv.is_empty() {return None;}
        let esz=slot.k[0].size()[1]*128*2;let base=slot.k[0].data_ptr() as i64;let view=&context.kv[0].0;
        let count=view.size()[0];if count>2048 {return None;}
        let begin=if count==0{0}else{(view.data_ptr() as i64-base)/esz};
        if count>0 && !(0..4096).contains(&begin) {return None;}
        for (i,(k,_)) in context.kv.iter().enumerate(){if count>0 {assert_eq!((k.data_ptr() as i64-slot.k[i].data_ptr() as i64)/esz,begin,"draft slab views must align");}}
        let meta=[context.len,context.start,begin,count];
        if std::env::var("GLM53_DRAFT_META_LOG").as_deref()==Ok("1") {
            eprintln!("[draft-meta] slot {:p} len {} start {} begin {} count {} kv0 {:?} storage {}",std::rc::Rc::as_ptr(slot),context.len,context.start,begin,count,view.size(),context.storage.is_some());
        }
        crate::tp::upload(&slot.meta,&meta);
        let mut ids=vec![self.mask_id;8];ids[0]=anchor;crate::tp::upload(&slot.ids,&ids);
        if crate::sampling::coupled() {let (t,k)=crate::sampling::draft_row(0);crate::tp::upload(&slot.dtemp,&[t;7]);crate::tp::upload(&slot.dkeys,&k);}
        let signature=draft_signature();
        // Keep one graph per draft configuration (ABBA arms alternate signatures).
        let found=slot.graph.borrow().iter().position(|g|g.signature==signature);
        if found.is_none() {
            if slot.graph.borrow().len()>=4 {slot.graph.borrow_mut().remove(0);}
            if !slot.warmed.get(){let _=self.propose_body(slot,target);slot.warmed.set(true);}
            tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();
            let (ids,unary,edges,path,hidden,sf,fn_,cc,cf)=self.propose_body(slot,target);
            let conf=path_conf(&unary,&ids,&path);
            crate::tp::graph::end().unwrap();
            slot.graph.borrow_mut().push(ProposeGraph{_graph:crate::tp::graph::Owned::take(),signature:signature.clone(),ids,unary,edges,path,hidden,conf,
                selector_fused:sf,final_norm_selected:fn_,conv_calls:cc,conv_fused_calls:cf});
        }
        let graphs=slot.graph.borrow();let g=graphs.iter().find(|g|g.signature==signature).unwrap();g._graph.replay();
        let path=Vec::<i64>::try_from(g.path.to_device(Device::Cpu)).unwrap();
        let conf=conf_tau().map(|_|Vec::<f32>::try_from(g.conf.to_device(Device::Cpu)).unwrap());
        Some(Candidates{ids:g.ids.shallow_clone(),unary:g.unary.shallow_clone(),edges:g.edges.shallow_clone(),path,hidden:g.hidden.shallow_clone(),conf,
            selector_fused:g.selector_fused,final_norm_selected:g.final_norm_selected,conv_calls:g.conv_calls,conv_fused_calls:g.conv_fused_calls})
    }
    /// Device-attention metadata of a slot-backed context: [len, start, slab begin, count].
    fn slot_meta(context:&Context,slot:&DraftSlot)->Option<[i64;4]> {
        if context.kv.is_empty() {return None;}
        let esz=slot.k[0].size()[1]*128*2;let base=slot.k[0].data_ptr() as i64;let view=&context.kv[0].0;
        let count=view.size()[0];if count>2048 {return None;}
        let begin=if count==0{0}else{(view.data_ptr() as i64-base)/esz};
        if count>0 && !(0..4096).contains(&begin) {return None;}
        for (i,(k,_)) in context.kv.iter().enumerate(){if count>0 {assert_eq!((k.data_ptr() as i64-slot.k[i].data_ptr() as i64)/esz,begin,"draft slab views must align");}}
        Some([context.len,context.start,begin,count])
    }
    fn conv_prepare_many(conv:&Conv,x:&Tensor,s:i64,d:&mut crate::draft_conv::Dispatch)->(Tensor,Tensor) {
        let delta=linear(x,&conv.projection).view([x.size()[0],2,2,256]);let d0=delta.select(1,0);
        let out=Tensor::cat(&(0..s).map(|g|conv.convolve(&x.narrow(0,g*8,8),&d0.narrow(0,g*8,8),0,d)).collect::<Vec<_>>(),0);
        (out,delta.select(1,1))
    }
    fn conv_convolve_many(conv:&Conv,x:&Tensor,delta:&Tensor,side:i64,s:i64,d:&mut crate::draft_conv::Dispatch)->Tensor {
        Tensor::cat(&(0..s).map(|g|conv.convolve(&x.narrow(0,g*8,8),&delta.narrow(0,g*8,8),side,d)).collect::<Vec<_>>(),0)
    }
    /// C2 (GLM53_SERVE_DRAFT_BATCH=1): proposals of 2..4 slot-backed contexts in one forward, so the
    /// drafter weights and head are read once per round instead of once per sequence. Embedding, norms,
    /// projections, MLP (one TP sum), head top-k and the selector projection run on all rows; attention
    /// (each slot's own KV, device kernel) and the block-local convolution run per context. Drafter-side
    /// only (L2): target outputs never depend on the draft; acceptance may move by rounding.
    /// propose_many's forward with every per-round input read from the slots' device buffers (ids, meta), so it can be
    /// captured: same operations and operands as the eager propose_many (positions = arange + meta len on device instead
    /// of on the host: the same integers). Returns per sequence (ids, unary, edges, path, hidden, conf, selector_fused).
    #[allow(clippy::type_complexity)]
    fn propose_many_body(&self,slots:&[std::rc::Rc<DraftSlot>],target:&crate::weights::ModelWeights)->(Vec<(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,bool)>,usize,usize) {
        let s=slots.len();let n=8i64;let rows=s as i64*n;let dev=target.device;
        let ids_in=Tensor::cat(&slots.iter().map(|sl|sl.ids.shallow_clone()).collect::<Vec<_>>(),0);
        let mut x=target.embed_tokens(&ids_in).to_kind(Kind::BFloat16);
        let positions=Tensor::cat(&slots.iter().map(|sl|self.positions(n)+sl.meta.narrow(0,0,1)).collect::<Vec<_>>(),0);
        let rotary=Some(rope_tables(&positions,Some(&self.rotary_inv)));
        let mut residual:Option<Tensor>=None;let mut conv=crate::draft_conv::Dispatch::new();
        let fused=fused_norm();
        for (i,layer) in self.layers.iter().enumerate() {
            let (z,r)=if fused {let r=residual.take();add_norm_f(&x,r.as_ref(),&layer.input_norm)} else {match residual.take(){Some(r)=>add_norm(&x,&r,&layer.input_norm),None=>(norm(&x,&layer.input_norm),x.shallow_clone())}};
            let (z,delta)=Self::conv_prepare_many(&layer.attn_conv,&z,s as i64,&mut conv);
            let (q,k)=if fused {(head_norm_rope(&linear(&z,&layer.q).view([rows,self.qh,128]),&layer.qn,&positions,&self.rotary_inv),
                                 head_norm_rope(&linear(&z,&layer.k).view([rows,self.kvh,128]),&layer.kn,&positions,&self.rotary_inv))}
                else {(rope(&norm(&linear(&z,&layer.q).view([rows,self.qh,128]),&layer.qn),&positions,rotary.as_ref()).contiguous(),
                       rope(&norm(&linear(&z,&layer.k).view([rows,self.kvh,128]),&layer.kn),&positions,rotary.as_ref()).contiguous())};
            let v=linear(&z,&layer.v).view([rows,self.kvh,128]).contiguous();
            let attention=Tensor::cat(&(0..s).map(|g|{let sl=|t:&Tensor|t.narrow(0,g as i64*n,n).contiguous();
                draft_attention_dev(&sl(&q),&slots[g].k[i],&slots[g].v[i],&sl(&k),&sl(&v),&slots[g])}).collect::<Vec<_>>(),0);
            let attn=Self::conv_convolve_many(&layer.attn_conv,&self.o_proj(&attention,&layer.o),&delta,1,s as i64,&mut conv);
            let (z,r)=if fused {add_norm_f(&attn,Some(&r),&layer.post_norm)} else {add_norm(&attn,&r,&layer.post_norm)};
            let (z,delta)=Self::conv_prepare_many(&layer.mlp_conv,&z,s as i64,&mut conv);
            x=Self::conv_convolve_many(&layer.mlp_conv,&self.mlp(layer,&z),&delta,1,s as i64,&mut conv);residual=Some(r);
        }
        let residual=residual.unwrap();
        let hidden=add_norm(&x,&residual,&self.final_norm).0.view([s as i64,n,4096]).narrow(1,1,n-1).reshape([rows-s as i64,4096]);
        let (unary_all,ids_all)=self.topk(&hidden);
        let projected_all=linear(&hidden,&self.selector);
        let mut out=Vec::with_capacity(s);
        for g in 0..s {
            let r=|t:&Tensor|t.narrow(0,g as i64*(n-1),n-1);
            let (ids,unary,projected,hidden)=(r(&ids_all),r(&unary_all),r(&projected_all),r(&hidden));
            let successors=self.successor.index_select(0,&ids.reshape([-1])).view([n-1,16,256]);
            let predecessors=Tensor::cat(&[slots[g].ids.narrow(0,0,1).view([1,1]).expand([1,16],false),ids.narrow(0,0,n-2)],0);
            let predecessors=self.predecessor.index_select(0,&predecessors.reshape([-1])).view([n-1,16,256]);
            let edges=(predecessors*projected.unsqueeze(1)).bmm(&successors.transpose(1,2)).to_kind(Kind::Float)+unary.unsqueeze(1);
            let (path,selector_fused)=if let Some(path)=crate::draft_selector::try_path(&edges,&ids.contiguous()) {(path,true)}else {
                let mut previous=Tensor::zeros([1],(Kind::Int64,dev));let mut tokens=Vec::new();
                for t in 0..n-1 {let best=edges.get(t).index_select(0,&previous).argmax(-1,false);tokens.push(ids.get(t).index_select(0,&best));previous=best;}
                (Tensor::cat(&tokens,0),false)
            };
            let conf=path_conf(&unary,&ids,&path);
            out.push((ids,unary,edges,path,hidden,conf,selector_fused));
        }
        (out,conv.calls,conv.fused)
    }
    /// GLM53_DRAFT_MANY_GRAPH=1 (L0 against the eager propose_many, greedy rounds only): propose_many captured once per
    /// (slot combination, draft configuration) and replayed with the slots' ids/meta uploaded; one host copy of all
    /// paths and confidences per round.
    fn propose_many_graph(&self,contexts:&[&Context],anchors:&[i64],target:&crate::weights::ModelWeights)->Option<Vec<Candidates>> {
        if crate::sampling::coupled() {return None;}
        let s=contexts.len();let mut slots=Vec::with_capacity(s);
        for (c,&a) in contexts.iter().zip(anchors) {let slot=c.slot.as_ref()?;let meta=Self::slot_meta(c,slot)?;
            crate::tp::upload(&slot.meta,&meta);let mut ids=vec![self.mask_id;8];ids[0]=a;crate::tp::upload(&slot.ids,&ids);slots.push(slot.clone());}
        let key:Vec<usize>=slots.iter().map(|sl|std::rc::Rc::as_ptr(sl) as usize).collect();let signature=draft_signature();
        MANY_GRAPHS.with(|m|{
            let mut m=m.borrow_mut();
            let clock={let c=m.iter().map(|g|g.used).max().unwrap_or(0)+1;c};
            let pos=m.iter().position(|g|g.key==key&&g.signature==signature);
            let i=match pos {Some(i)=>i,None=>{
                let limit=std::env::var("GLM53_DRAFT_MANY_GRAPHS").ok().and_then(|v|v.parse::<usize>().ok()).unwrap_or(8).max(1);
                while m.len()>=limit {let lru=(0..m.len()).min_by_key(|&i|m[i].used).unwrap();m.remove(lru);}
                let _=self.propose_many_body(&slots,target);tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();
                let (outs,cc,cf)=self.propose_many_body(&slots,target);
                let paths=Tensor::cat(&outs.iter().map(|o|o.3.reshape([-1])).collect::<Vec<_>>(),0);
                let confs=Tensor::cat(&outs.iter().map(|o|o.5.reshape([-1])).collect::<Vec<_>>(),0);
                crate::tp::graph::end().unwrap();
                m.push(ManyGraph{graph:crate::tp::graph::Owned::take(),key:key.clone(),signature:signature.clone(),_slots:slots.clone(),outs,paths,confs,cc,cf,used:clock});
                m.len()-1}};
            let g=&mut m[i];g.used=clock;g.graph.replay();
            let paths=Vec::<i64>::try_from(g.paths.to_device(Device::Cpu)).unwrap();
            let confs=conf_tau().map(|_|Vec::<f32>::try_from(g.confs.to_device(Device::Cpu)).unwrap());
            let mut out=Vec::with_capacity(s);let (mut po,mut co)=(0usize,0usize);
            for o in &g.outs {
                let pl=o.3.numel();let cl=o.5.numel();
                out.push(Candidates{ids:o.0.shallow_clone(),unary:o.1.shallow_clone(),edges:o.2.shallow_clone(),path:paths[po..po+pl].to_vec(),hidden:o.4.shallow_clone(),
                    conf:confs.as_ref().map(|v|v[co..co+cl].to_vec()),selector_fused:o.6,final_norm_selected:false,conv_calls:g.cc,conv_fused_calls:g.cf});
                po+=pl;co+=cl;
            }
            Some(out)
        })
    }
    pub fn propose_many(&self,contexts:&[&Context],anchors:&[i64],target:&crate::weights::ModelWeights)->Option<Vec<Candidates>> {
        let s=contexts.len();if self.block_size!=8 || !(2..=4).contains(&s) || anchors.len()!=s {return None;}
        if std::env::var("GLM53_DRAFT_MANY_GRAPH").as_deref()==Ok("1") {if let Some(c)=self.propose_many_graph(contexts,anchors,target) {
            // GLM53_DRAFT_MANY_CHECK=1: the eager proposals of the same round must equal the replayed ones.
            if std::env::var("GLM53_DRAFT_MANY_CHECK").as_deref()==Ok("1") {
                let e=self.propose_many_eager(contexts,anchors,target).unwrap();
                use std::sync::atomic::{AtomicU64,Ordering::Relaxed};static OK:AtomicU64=AtomicU64::new(0);static BAD:AtomicU64=AtomicU64::new(0);
                let same=c.iter().zip(&e).all(|(a,b)|a.path==b.path && a.conf==b.conf && a.edges.equal(&b.edges) && a.hidden.equal(&b.hidden));
                if same {let o=OK.fetch_add(1,Relaxed)+1;if o%100==0{eprintln!("[draft-many-check] ok {o} bad {}",BAD.load(Relaxed));}}
                else {let b=BAD.fetch_add(1,Relaxed)+1;eprintln!("[draft-many-check] MISMATCH (bad {b}) paths {:?} vs {:?}",c.iter().map(|x|x.path.clone()).collect::<Vec<_>>(),e.iter().map(|x|x.path.clone()).collect::<Vec<_>>());}
            }
            return Some(c);}}
        self.propose_many_eager(contexts,anchors,target)
    }
    fn propose_many_eager(&self,contexts:&[&Context],anchors:&[i64],target:&crate::weights::ModelWeights)->Option<Vec<Candidates>> {
        let s=contexts.len();
        let mut slots=Vec::with_capacity(s);let mut metas=Vec::with_capacity(s);
        for c in contexts {let slot=c.slot.as_ref()?;metas.push(Self::slot_meta(c,slot)?);slots.push(slot.clone());}
        for (slot,meta) in slots.iter().zip(&metas) {slot.meta.shallow_clone().copy_(&Tensor::from_slice(meta));}
        let n=8i64;let rows=s as i64*n;let dev=target.device;
        let mut ids=Vec::with_capacity(rows as usize);for &a in anchors {ids.push(a);ids.extend(std::iter::repeat(self.mask_id).take(7));}
        let mut x=target.embed_tokens(&Tensor::from_slice(&ids).to_device(dev)).to_kind(Kind::BFloat16);
        let positions=Tensor::cat(&contexts.iter().map(|c|self.positions(n)+c.len).collect::<Vec<_>>(),0);
        let rotary=Some(rope_tables(&positions,Some(&self.rotary_inv)));
        let mut residual:Option<Tensor>=None;let mut conv=crate::draft_conv::Dispatch::new();
        let fused=fused_norm();
        for (i,layer) in self.layers.iter().enumerate() {
            let (z,r)=if fused {let r=residual.take();add_norm_f(&x,r.as_ref(),&layer.input_norm)} else {match residual.take(){Some(r)=>add_norm(&x,&r,&layer.input_norm),None=>(norm(&x,&layer.input_norm),x.shallow_clone())}};
            let (z,delta)=Self::conv_prepare_many(&layer.attn_conv,&z,s as i64,&mut conv);
            let (q,k)=if fused {(head_norm_rope(&linear(&z,&layer.q).view([rows,self.qh,128]),&layer.qn,&positions,&self.rotary_inv),
                                 head_norm_rope(&linear(&z,&layer.k).view([rows,self.kvh,128]),&layer.kn,&positions,&self.rotary_inv))}
                else {(rope(&norm(&linear(&z,&layer.q).view([rows,self.qh,128]),&layer.qn),&positions,rotary.as_ref()).contiguous(),
                       rope(&norm(&linear(&z,&layer.k).view([rows,self.kvh,128]),&layer.kn),&positions,rotary.as_ref()).contiguous())};
            let v=linear(&z,&layer.v).view([rows,self.kvh,128]).contiguous();
            let attention=Tensor::cat(&(0..s).map(|g|{let sl=|t:&Tensor|t.narrow(0,g as i64*n,n).contiguous();
                draft_attention_dev(&sl(&q),&slots[g].k[i],&slots[g].v[i],&sl(&k),&sl(&v),&slots[g])}).collect::<Vec<_>>(),0);
            let attn=Self::conv_convolve_many(&layer.attn_conv,&self.o_proj(&attention,&layer.o),&delta,1,s as i64,&mut conv);
            let (z,r)=if fused {add_norm_f(&attn,Some(&r),&layer.post_norm)} else {add_norm(&attn,&r,&layer.post_norm)};
            let (z,delta)=Self::conv_prepare_many(&layer.mlp_conv,&z,s as i64,&mut conv);
            x=Self::conv_convolve_many(&layer.mlp_conv,&self.mlp(layer,&z),&delta,1,s as i64,&mut conv);residual=Some(r);
        }
        let residual=residual.unwrap();
        let tm=std::env::var("GLM53_ROUND_TIMING").as_deref()==Ok("1");
        let t_layers=if tm {tch::Cuda::synchronize(0);Some(std::time::Instant::now())} else {None};
        let hidden=add_norm(&x,&residual,&self.final_norm).0.view([s as i64,n,4096]).narrow(1,1,n-1).reshape([rows-s as i64,4096]);
        let (unary_all,ids_all)=self.topk(&hidden);
        let projected_all=linear(&hidden,&self.selector);
        let mut out=Vec::with_capacity(s);
        let one_d2h=std::env::var("GLM53_PROPOSE_ONE_D2H").as_deref()==Ok("1");
        let (mut dev_path,mut dev_conf):(Vec<Tensor>,Vec<Option<Tensor>>)=(Vec::new(),Vec::new());
        for g in 0..s {
            let r=|t:&Tensor|t.narrow(0,g as i64*(n-1),n-1);
            let (ids,unary,projected,hidden)=(r(&ids_all),r(&unary_all),r(&projected_all),r(&hidden));
            let successors=self.successor.index_select(0,&ids.reshape([-1])).view([n-1,16,256]);
            let predecessors=Tensor::cat(&[Tensor::full([1,16],anchors[g],(Kind::Int64,dev)),ids.narrow(0,0,n-2)],0);
            let predecessors=self.predecessor.index_select(0,&predecessors.reshape([-1])).view([n-1,16,256]);
            let mut edges=(predecessors*projected.unsqueeze(1)).bmm(&successors.transpose(1,2)).to_kind(Kind::Float)+unary.unsqueeze(1);
            if crate::sampling::coupled() {edges=edges+crate::sampling::candidate_noise_row(&ids.contiguous(),g).unsqueeze(1);}
            let (path,selector_fused)=if let Some(path)=crate::draft_selector::try_path(&edges,&ids.contiguous()) {(path,true)}else {
                let mut previous=Tensor::zeros([1],(Kind::Int64,dev));let mut tokens=Vec::new();
                for t in 0..n-1 {let best=edges.get(t).index_select(0,&previous).argmax(-1,false);tokens.push(ids.get(t).index_select(0,&best));previous=best;}
                (Tensor::cat(&tokens,0),false)
            };
            if one_d2h {
                // GLM53_PROPOSE_ONE_D2H=1 (L0): the same per-sequence path/conf tensors, copied to the host once for all
                // sequences below instead of two synchronizing copies per sequence.
                dev_conf.push(conf_tau().map(|_|path_conf(&unary,&ids,&path)));dev_path.push(path);
                out.push(Candidates{ids,unary,edges,path:Vec::new(),hidden,conf:None,selector_fused,final_norm_selected:false,conv_calls:conv.calls,conv_fused_calls:conv.fused});
                continue;
            }
            let conf=conf_tau().map(|_|Vec::<f32>::try_from(path_conf(&unary,&ids,&path).to_device(Device::Cpu)).unwrap());
            let path=Vec::<i64>::try_from(path.to_device(Device::Cpu)).unwrap();
            out.push(Candidates{ids,unary,edges,path,hidden,conf,selector_fused,final_norm_selected:false,conv_calls:conv.calls,conv_fused_calls:conv.fused});
        }
        if one_d2h {
            let lens:Vec<i64>=dev_path.iter().map(|p|p.numel() as i64).collect();
            let paths=Vec::<i64>::try_from(Tensor::cat(&dev_path,0).to_device(Device::Cpu)).unwrap();
            let confs:Option<Vec<f32>>=dev_conf.iter().all(|c|c.is_some()).then(||{let v:Vec<Tensor>=dev_conf.iter().map(|c|c.as_ref().unwrap().reshape([-1])).collect();
                Vec::<f32>::try_from(Tensor::cat(&v,0).to_device(Device::Cpu)).unwrap()});
            let clen:Vec<i64>=dev_conf.iter().map(|c|c.as_ref().map_or(0,|t|t.numel() as i64)).collect();
            let (mut po,mut co)=(0usize,0usize);
            for (g,c) in out.iter_mut().enumerate() {
                c.path=paths[po..po+lens[g] as usize].to_vec();po+=lens[g] as usize;
                if let Some(v)=&confs {c.conf=Some(v[co..co+clen[g] as usize].to_vec());co+=clen[g] as usize;}
            }
        }
        if let Some(t)=t_layers {eprintln!("[round-timing] propose_many tail (topk+selector+D2H) {:.2} ms",t.elapsed().as_secs_f64()*1000.);}
        Some(out)
    }
    pub fn propose(&self,context:&Context,anchor:i64,target:&crate::weights::ModelWeights)->Candidates {
        if graph_enabled() { if let Some(c)=self.propose_graph(context,anchor,target) {return c;} }
        let n=self.block_size;let dev=target.device;
        let mut ids=vec![self.mask_id;n as usize];ids[0]=anchor;
        let mut x=target.embed_tokens(&Tensor::from_slice(&ids).to_device(dev)).to_kind(Kind::BFloat16);
        let positions=self.positions(n)+context.len;
        let rotary=rope_cache().then(||rope_tables(&positions,Some(&self.rotary_inv)));
        let keys_pos=self.positions(context.len-context.start+n)+context.start;
        let visible=(&positions.unsqueeze(1)-&keys_pos.unsqueeze(0)).abs().lt(2048);
        let mut residual:Option<Tensor>=None;let mut conv=crate::draft_conv::Dispatch::new();
        for (layer,(ck,cv)) in self.layers.iter().zip(&context.kv) {
            let (z,r)=match residual.take(){Some(r)=>add_norm(&x,&r,&layer.input_norm),None=>(norm(&x,&layer.input_norm),x.shallow_clone())};
            let (z,delta)=layer.attn_conv.prepare(&z,&mut conv);
            let q=rope(&norm(&linear(&z,&layer.q).view([n,self.qh,128]),&layer.qn),&positions,rotary.as_ref());
            let k=rope(&norm(&linear(&z,&layer.k).view([n,self.kvh,128]),&layer.kn),&positions,rotary.as_ref());
            let v=linear(&z,&layer.v).view([n,self.kvh,128]);
            let mode=std::env::var("GLM53_DRAFT_GQA").unwrap_or_else(|_|"0".into());
            assert!(["0","1","2","3","4","5"].contains(&mode.as_str()));
            let attention=if std::env::var("GLM53_DRAFT_ATTN_ORACLE").as_deref()==Ok("1") {
                attention_oracle(&q,ck,cv,&k,&v,&visible)
            }else if mode=="2" || (["3","5"].contains(&mode.as_str()) && ck.size()[0]<=std::env::var("GLM53_DRAFT_GQA_SHORT_MAX").ok().map(|v|v.parse::<i64>().unwrap()).unwrap_or(64)) {
                attention_fused(&q,ck,cv,&k,&v)
            }else if ["4","5"].contains(&mode.as_str()) {attention_shared_heads(&q,ck,cv,&k,&v,&visible)}
            else {attention(&q,ck,cv,&k,&v,&visible,mode!="0")};
            let attn=layer.attn_conv.convolve(&self.o_proj(&attention,&layer.o),&delta,1,&mut conv);
            let (z,r)=add_norm(&attn,&r,&layer.post_norm);let (z,delta)=layer.mlp_conv.prepare(&z,&mut conv);
            // Fused SiLU-and-mul rounds once; don't round SiLU separately to BF16.
            x=layer.mlp_conv.convolve(&self.mlp(layer,&z),&delta,1,&mut conv);residual=Some(r);
        }
        let residual=residual.unwrap();
        let (hidden,final_norm_selected)=if let Some(h)=crate::draft_final_norm::try_hidden(&x,&residual,|sum|norm(sum,&self.final_norm)) {
            (h,true)
        }else{(add_norm(&x,&residual,&self.final_norm).0.narrow(0,1,n-1),false)};
        let (unary,ids)=self.topk(&hidden);
        let projected=linear(&hidden,&self.selector);
        let successors=self.successor.index_select(0,&ids.reshape([-1])).view([n-1,16,256]);
        let predecessors=Tensor::cat(&[Tensor::full([1,16],anchor,(Kind::Int64,dev)),ids.narrow(0,0,n-2)],0);
        let predecessors=self.predecessor.index_select(0,&predecessors.reshape([-1])).view([n-1,16,256]);
        let mut edges=(predecessors*projected.unsqueeze(1)).bmm(&successors.transpose(1,2)).to_kind(Kind::Float)+unary.unsqueeze(1);
        if crate::sampling::coupled() {edges=edges+crate::sampling::candidate_noise_row(&ids.contiguous(),0).unsqueeze(1);}
        let (path,selector_fused)=if let Some(path)=crate::draft_selector::try_path(&edges,&ids) {(path,true)}else {
            let mut previous=Tensor::zeros([1],(Kind::Int64,dev));let mut tokens=Vec::new();
            for t in 0..n-1 {let best=edges.get(t).index_select(0,&previous).argmax(-1,false);tokens.push(ids.get(t).index_select(0,&best));previous=best;}
            (Tensor::cat(&tokens,0),false)
        };
        let conf=conf_tau().map(|_|Vec::<f32>::try_from(path_conf(&unary,&ids,&path).to_device(Device::Cpu)).unwrap());
        let path=Vec::<i64>::try_from(path.to_device(Device::Cpu)).unwrap();
        Candidates{ids,unary,edges,path,hidden,conf,selector_fused,final_norm_selected,conv_calls:conv.calls,conv_fused_calls:conv.fused}
    }
}

/// Local primitive gate loads only actual norm.weight, no head or TP setup.
pub fn final_norm_select_probe(draft:&Path,out:&Path) {
    assert!(!crate::tp::is_tp());tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let mut idx=crate::safetensors::ShardIndex::scan(draft).unwrap();let(v,shape)=idx.get_f32("norm.weight").unwrap();
    assert_eq!(shape,vec![4096]);let w=Norm::new(Tensor::from_slice(&v).to_kind(Kind::BFloat16).to_device(Device::Cuda(0)));
    crate::draft_final_norm::norm_check(|sum|norm(sum,&w),Device::Cuda(0),out);
}

pub fn probe(target:&Path,draft:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();
    let dev=Device::Cuda(0);let cfg=crate::config::load(&target.join("config.json")).unwrap();
    let target=crate::weights::ModelWeights::load(target,&cfg,0,dev);
    let draft=Drafter::load(draft,&target.drafter_head());let mut context=draft.empty_context();
    tch::manual_seed(20260922);let features=Tensor::randn([2053,20480],(Kind::Float,dev))*0.1;
    let mut archives=vec![("features".to_owned(),features.to_device(Device::Cpu))];
    let mut pos=0;
    for n in [5,11,2037] {
        draft.append(&mut context,&features.narrow(0,pos,n));pos+=n;
        let c=draft.propose(&context,13041,&target);
        assert!(c.hidden.isfinite().all().int64_value(&[])!=0);assert_eq!(c.path.len(),7);
        archives.extend([(format!("hidden_{pos}"),c.hidden.to_device(Device::Cpu)),
            (format!("ids_{pos}"),c.ids.to_device(Device::Cpu)),(format!("unary_{pos}"),c.unary.to_device(Device::Cpu)),
            (format!("edges_{pos}"),c.edges.to_device(Device::Cpu)),(format!("path_{pos}"),Tensor::from_slice(&c.path))]);
        eprintln!("[dflash-probe] context={pos} start={} path={:?}",context.start,c.path);
    }
    Tensor::save_multi(&archives,out.join("drafter.pt")).unwrap();
}

/// Real BF16 head + full drafter, both ranks, including window/compaction boundaries.
pub fn opt_probe(target:&Path,draft:&Path,out:&Path) {
    use std::time::Instant;
    use serde_json::json;
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);
    let cfg=crate::config::load(&target.join("config.json")).unwrap();
    let target=crate::weights::ModelWeights::load(target,&cfg,0,dev);
    std::env::remove_var("GLM53_DRAFT_HEAD_SHARD_LOAD");
    std::env::remove_var("GLM53_DRAFT_MLP_SHARD_LOAD");std::env::set_var("GLM53_DRAFT_MLP_TP","0");std::env::set_var("GLM53_DRAFT_TOPK_TP","0");
    let drafter=Drafter::load_target(draft,&target);tch::manual_seed(20260924);
    let hidden=Tensor::randn([7,4096],(Kind::BFloat16,dev));
    let mut input=hidden.copy();let mut head_cases=Vec::new();
    for n in [1,3,7] {
        let input=input.narrow(0,0,n);
        std::env::set_var("GLM53_DRAFT_HEAD_TP","0");let reference=drafter.logits(&input);
        std::env::set_var("GLM53_DRAFT_HEAD_TP","1");let actual=drafter.logits(&input);
        let difference=(&actual-&reference).abs();
        let max_abs=difference.max().double_value(&[]);
        let top1_equal=actual.argmax(-1,false).equal(&reference.argmax(-1,false));
        let exact=actual.equal(&reference);
        std::env::set_var("GLM53_DRAFT_TOPK_TP","1");let (scores,candidates)=drafter.topk(&input);
        assert!(scores.equal(&reference.topk(16,-1,true,true).0),"distributed topk scores");
        assert!(scores.equal(&reference.gather(-1,&candidates,false)),"distributed topk IDs");
        for _ in 0..3{let _=drafter.topk(&input);}tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let (graph_scores,graph_ids)=drafter.topk(&input);crate::tp::graph::end().unwrap();
        for z in [&hidden.narrow(0,0,n),&(-hidden.narrow(0,0,n)),&Tensor::zeros_like(&input)] {
            let mut dest=input.shallow_clone();dest.copy_(z);crate::tp::graph::replay().unwrap();
            let (s,i)=drafter.topk(z);assert!(graph_scores.equal(&s)&&graph_ids.equal(&i),"distributed topk graph");
            let all=drafter.logits(z);assert!(s.equal(&all.topk(16,-1,true,true).0)&&s.equal(&all.gather(-1,&i,false)));
        }
        crate::tp::graph::destroy();let mut dest=input.shallow_clone();dest.copy_(&hidden.narrow(0,0,n));
        std::env::set_var("GLM53_DRAFT_TOPK_TP","0");
        assert!(actual.isfinite().all().int64_value(&[])!=0);assert!(top1_equal,"head top1");
        let mut rounds=Vec::new();
        for on in [false,true,true,false] {
            std::env::set_var("GLM53_DRAFT_HEAD_TP",if on{"1"}else{"0"});
            for _ in 0..3{let _=drafter.logits(&input);}tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let y=drafter.logits(&input);crate::tp::graph::end().unwrap();
            for z in [&hidden.narrow(0,0,n),&(-hidden.narrow(0,0,n)),&Tensor::zeros_like(&input)] {
                let mut dest=input.shallow_clone();dest.copy_(z);crate::tp::graph::replay().unwrap();
                assert!(y.equal(&drafter.logits(z)),"head changed-input graph");
            }
            let mut dest=input.shallow_clone();dest.copy_(&hidden.narrow(0,0,n));
            crate::tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));tch::Cuda::synchronize(0);
            let begin=Instant::now();for _ in 0..64{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"head_tp":on,"graph_us":begin.elapsed().as_secs_f64()*1e6/64.}));crate::tp::graph::destroy();
        }
        head_cases.push(json!({"rows":n,"exact":exact,"max_abs":max_abs,"top1_equal":top1_equal,"rounds":rounds}));
    }
    std::env::set_var("GLM53_DRAFT_HEAD_TP","0");
    let mut baseline=drafter.empty_context();let mut candidate=drafter.empty_context();
    let mut contexts=Vec::new();
    // Cross 2048 visibility and 4096 storage boundaries, including a >window append.
    for n in [5,11,2031,1,1,7,2040,8,2053] {
        let features=Tensor::randn([n,20480],(Kind::Float,dev))*0.1;
        std::env::set_var("GLM53_DRAFT_KV_BUFFER","0");std::env::set_var("GLM53_DRAFT_ROPE_CACHE","0");drafter.append(&mut baseline,&features);
        std::env::set_var("GLM53_DRAFT_KV_BUFFER","1");std::env::set_var("GLM53_DRAFT_ROPE_CACHE","1");drafter.append(&mut candidate,&features);
        assert_eq!((baseline.len,baseline.start),(candidate.len,candidate.start));
        for ((a,b),(c,d)) in baseline.kv.iter().zip(&candidate.kv){assert!(a.equal(c)&&b.equal(d),"buffer KV len={}",baseline.len);}
        std::env::set_var("GLM53_DRAFT_HEAD_TP","0");std::env::set_var("GLM53_DRAFT_ROPE_CACHE","0");let reference=drafter.propose(&baseline,13041,&target);
        std::env::set_var("GLM53_DRAFT_HEAD_TP","1");std::env::set_var("GLM53_DRAFT_ROPE_CACHE","1");let actual=drafter.propose(&candidate,13041,&target);
        assert!(reference.hidden.equal(&actual.hidden),"buffer hidden");
        let exact=reference.unary.equal(&actual.unary)&&reference.ids.equal(&actual.ids)&&reference.edges.equal(&actual.edges);
        let path_equal=reference.path==actual.path;
        assert!(path_equal,"candidate path changed at context {}",baseline.len);
        std::env::set_var("GLM53_DRAFT_TOPK_TP","1");let merged=drafter.propose(&candidate,13041,&target);
        std::env::set_var("GLM53_DRAFT_TOPK_TP","0");
        assert!(merged.hidden.equal(&actual.hidden)&&merged.unary.equal(&actual.unary));
        let all=drafter.logits(&merged.hidden);assert!(merged.unary.equal(&all.gather(-1,&merged.ids,false)));
        // Checkout must be independently mutable, and lazily rebuild its slab.
        let mut checkout=candidate.snapshot();let saved=candidate.snapshot();
        drafter.append(&mut checkout,&features.narrow(0,0,1));
        assert_eq!(checkout.len,candidate.len+1);
        for ((a,b),(c,d)) in candidate.kv.iter().zip(&saved.kv){assert!(a.equal(c)&&b.equal(d),"checkout aliases pool");}
        contexts.push(json!({"append_rows":n,"len":baseline.len,"start":baseline.start,"kv_exact":true,"proposal_exact":exact,"path_equal":path_equal,
            "distributed_topk_scores_exact":true,"distributed_topk_ids_exact":merged.ids.equal(&actual.ids),"distributed_topk_path_equal":merged.path==actual.path}));
        eprintln!("[draft-opt] rank{} context={} exact={exact}",tp.rank,baseline.len);
    }
    let mut mlp_cases=Vec::new();
    for rows in [1,3,7,8] {
        let z=Tensor::randn([rows,4096],(Kind::BFloat16,dev))*0.1;
        for (i,layer) in drafter.layers.iter().enumerate() {
            std::env::set_var("GLM53_DRAFT_MLP_TP","0");let expected=drafter.mlp(layer,&z);
            std::env::set_var("GLM53_DRAFT_MLP_TP","1");let actual=drafter.mlp(layer,&z);
            let error=((actual.to_kind(Kind::Float)-expected.to_kind(Kind::Float)).norm()/expected.to_kind(Kind::Float).norm()).double_value(&[]);
            assert!(error<0.01,"drafter MLP TP layer {i}: {error}");
            mlp_cases.push(json!({"layer":i,"rows":rows,"exact":actual.equal(&expected),"relative_l2":error}));
        }
    }
    std::env::set_var("GLM53_DRAFT_MLP_TP","0");let original=drafter.propose(&baseline,13041,&target);
    std::env::set_var("GLM53_DRAFT_MLP_TP","1");let split=drafter.propose(&baseline,13041,&target);
    let mlp_path_equal=original.path==split.path;
    let mlp_hidden_rel=((original.hidden.to_kind(Kind::Float)-split.hidden.to_kind(Kind::Float)).norm()/original.hidden.to_kind(Kind::Float).norm()).double_value(&[]);
    assert!(mlp_hidden_rel<0.02);std::env::set_var("GLM53_DRAFT_MLP_TP","0");
    // Whole proposal and append costs, both arms include communication and sync.
    let features=Tensor::randn([8,20480],(Kind::Float,dev))*0.1;let mut perf=Vec::new();
    for flag in ["GLM53_DRAFT_HEAD_TP","GLM53_DRAFT_KV_BUFFER","GLM53_DRAFT_ROPE_CACHE","GLM53_DRAFT_MLP_TP","GLM53_DRAFT_TOPK_TP"] {
        std::env::set_var("GLM53_DRAFT_HEAD_TP",if flag=="GLM53_DRAFT_TOPK_TP"{"1"}else{"0"});std::env::set_var("GLM53_DRAFT_KV_BUFFER","0");
        for on in [false,true,true,false] {
            std::env::set_var(flag,if on{"1"}else{"0"});let mut context=baseline.snapshot();
            for _ in 0..3{if flag!="GLM53_DRAFT_KV_BUFFER"{let _=drafter.propose(&context,13041,&target);}else{drafter.append(&mut context,&features);}}
            crate::tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));tch::Cuda::synchronize(0);
            let begin=Instant::now();
            for _ in 0..32{if flag!="GLM53_DRAFT_KV_BUFFER"{let _=drafter.propose(&context,13041,&target);}else{drafter.append(&mut context,&features);}}
            tch::Cuda::synchronize(0);perf.push(json!({"flag":flag,"enabled":on,"us":begin.elapsed().as_secs_f64()*1e6/32.}));
        }
    }
    std::env::set_var("GLM53_DRAFT_MLP_TP","0");
    std::env::set_var("GLM53_DRAFT_HEAD_TP","1");std::env::set_var("GLM53_DRAFT_HEAD_SHARD_LOAD","1");
    let sharded=Drafter::load_target(draft,&target);
    let (start,total)=sharded.head_shard.unwrap();
    assert!(sharded.lm_head.equal(&drafter.lm_head.narrow(0,start,sharded.lm_head.size()[0])));
    let full=drafter.propose(&baseline,13041,&target);let local=sharded.propose(&baseline,13041,&target);
    assert!(full.hidden.equal(&local.hidden)&&full.ids.equal(&local.ids)&&full.edges.equal(&local.edges));assert_eq!(full.path,local.path);
    std::env::set_var("GLM53_DRAFT_MLP_TP","1");std::env::set_var("GLM53_DRAFT_MLP_SHARD_LOAD","1");
    let loaded=Drafter::load_target(draft,&target);
    for (a,b) in drafter.layers.iter().zip(&loaded.layers) {
        let n=a.gate.size()[0]/2;let start=n*tp.rank as i64;
        assert!(b.gate.equal(&a.gate.narrow(0,start,n)) && b.up.equal(&a.up.narrow(0,start,n)) && b.down.equal(&a.down.narrow(1,start,n)));
    }
    let sliced=drafter.propose(&baseline,13041,&target);let materialized=loaded.propose(&baseline,13041,&target);
    let loaded_mlp_hidden_rel=((sliced.hidden.to_kind(Kind::Float)-materialized.hidden.to_kind(Kind::Float)).norm()/sliced.hidden.to_kind(Kind::Float).norm()).double_value(&[]);
    assert!(loaded_mlp_hidden_rel<0.02);
    std::fs::write(out.join(format!("rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({"head":head_cases,"contexts":contexts,"perf":perf,
        "buffer_min_context":buffer_min_context(),"loaded_mlp_hidden_relative_l2":loaded_mlp_hidden_rel,"loaded_mlp_path_equal":sliced.path==materialized.path,
        "distributed_candidate_payload_bytes":tp.world*7*16*2*4,"full_logits_payload_bytes":7*total*4,
        "mlp_weight_shards_exact":true,"mlp_cases":mlp_cases,"mlp_path_equal":mlp_path_equal,"mlp_hidden_relative_l2":mlp_hidden_rel,"sharded_load_exact":true,"head_bytes_per_rank":sharded.lm_head.numel()*2,"replicated_head_bytes":total*4096*2})).unwrap()).unwrap();
}

/// Real drafter weights: trim before projection, absolute RoPE and slab rollover.
pub fn dataflow_probe(target:&Path,draft:&Path,out:&Path) {
    use serde_json::json;
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);let dev=Device::Cuda(0);
    let cfg=crate::config::load(&target.join("config.json")).unwrap();
    let weights=crate::weights::ModelWeights::load(target,&cfg,0,dev);
    let drafter=Drafter::load_target(draft,&weights);tch::manual_seed(9226);let mut records=Vec::new();
    for buffered_mode in [false,true] {
        std::env::set_var("GLM53_DRAFT_KV_BUFFER",if buffered_mode{"1"}else{"0"});
        let mut baseline=drafter.empty_context();let mut candidate=drafter.empty_context();
        for n in [1,20,43,64,1920,1,7,2049,4097,8,2048,8] {
            let features=Tensor::randn([n,20480],(Kind::Float,dev))*0.1;
            std::env::set_var("GLM53_DRAFT_APPEND_TRIM","0");drafter.append(&mut baseline,&features);
            std::env::set_var("GLM53_DRAFT_APPEND_TRIM","1");drafter.append(&mut candidate,&features);
            assert_eq!((baseline.len,baseline.start),(candidate.len,candidate.start));
            let mut max_rel=0f64;
            for ((a,b),(c,d)) in baseline.kv.iter().zip(&candidate.kv) {for (a,b) in [(a,c),(b,d)] {
                let rel=((a.to_kind(Kind::Float)-b.to_kind(Kind::Float)).norm()/a.to_kind(Kind::Float).norm().clamp_min(1e-12)).double_value(&[]);max_rel=max_rel.max(rel);
            }}
            assert!(max_rel<0.002,"trim KV numerical drift {max_rel}");
            std::env::set_var("GLM53_DRAFT_GQA","0");let expected=drafter.propose(&baseline,13041,&weights);
            let trim_only=drafter.propose(&candidate,13041,&weights);
            let trim_rel=((trim_only.hidden.to_kind(Kind::Float)-expected.hidden.to_kind(Kind::Float)).norm()/expected.hidden.to_kind(Kind::Float).norm()).double_value(&[]);
            eprintln!("[draft-dataflow] rank{} n={n} len={} buffer={buffered_mode} kv_rel={max_rel} trim_hidden_rel={trim_rel}",tp.rank,candidate.len);
            assert!(trim_rel<0.02,"append trim alone: {trim_rel}");
            std::env::set_var("GLM53_DRAFT_ATTN_ORACLE","1");let oracle=drafter.propose(&baseline,13041,&weights);std::env::remove_var("GLM53_DRAFT_ATTN_ORACLE");
            let oracle_rel=|a:&Tensor|((a.to_kind(Kind::Float)-oracle.hidden.to_kind(Kind::Float)).norm()/oracle.hidden.to_kind(Kind::Float).norm()).double_value(&[]);
            let baseline_oracle_rel=oracle_rel(&expected.hidden);
            for mode in ["1","2","3","4","5"] {
                std::env::set_var("GLM53_DRAFT_GQA",mode);let actual=drafter.propose(&candidate,13041,&weights);
                let rel=((actual.hidden.to_kind(Kind::Float)-expected.hidden.to_kind(Kind::Float)).norm()/expected.hidden.to_kind(Kind::Float).norm()).double_value(&[]);
                eprintln!("[draft-dataflow] mode={mode} hidden_rel={rel}");
                records.push(json!({"baseline_oracle_relative":baseline_oracle_rel,"candidate_oracle_relative":oracle_rel(&actual.hidden),"original_hidden_screen_pass":rel<0.02,
                    "trim_hidden_relative":trim_rel,"buffered":buffered_mode,"n":n,"len":candidate.len,"start":candidate.start,"gqa":mode,
                    "kv_exact":candidate.equal(&baseline),"kv_relative":max_rel,"hidden_relative":rel,"path_equal":actual.path==expected.path}));
                std::fs::write(out.join(format!("draft-dataflow-rank{}.json",tp.rank)),serde_json::to_string_pretty(&records).unwrap()).unwrap();
                // Keep failed exploratory modes in the evidence; the original
                // 2% screen applies to the new geometry-preserving candidate.
                if mode=="5"{assert!(rel<0.02,"drafter hidden drift len={} mode={mode}: {rel}",candidate.len);}
                assert!(rel.is_finite());
            }
            // A restored prefix must detach from the original slab on append.
            if n==4097 {candidate=candidate.snapshot();baseline=baseline.snapshot();}
            std::fs::write(out.join(format!("draft-dataflow-rank{}.json",tp.rank)),serde_json::to_string_pretty(&records).unwrap()).unwrap();
        }
    }
    eprintln!("[draft-dataflow] rank{} trim/slab/restore/selector passed",tp.rank);
}

#[path="draft_conv_proposal.rs"]
mod conv_probe;
pub(crate) use conv_probe::check as conv_fused_check;

/// `draft-head-probe` (single GPU, synthetic weights of the real per-rank head shape [77440, 4096], 8 rows): time of
/// the FP8 head + top-16 against the INT4 two-stage path step by step (coarse INT4 kernel, top-64, gather, FP8 rescore,
/// top-16), and how often the two paths' top-16 id sets agree.
pub fn head_probe() {
    use tch::Device;let dev=Device::Cuda(0);let _g=tch::no_grad_guard();
    std::env::set_var("GLM53_DRAFT_FP8_HEAD","1");std::env::set_var("GLM53_FP8_SKINNY","1");
    tch::manual_seed(11);let (n,k,m)=(77440i64,4096i64,8i64);
    let w=(Tensor::randn([n,k],(Kind::Float,dev))*0.02).to_kind(Kind::BFloat16);
    crate::dense_fp8::register_weight(&w,"GLM53_DRAFT_FP8_HEAD");
    let (q,s)=crate::dense_fp8::quant_of_bf16(&w).expect("fp8 head");
    let (q4,s4)=head4_of(&q,&s).unwrap();
    let x=(Tensor::randn([m,k],(Kind::Float,dev))).to_kind(Kind::BFloat16);
    let time=|f:&dyn Fn()->Tensor|->f64{for _ in 0..3{let _=f();}tch::Cuda::synchronize(0);let t=std::time::Instant::now();for _ in 0..50{let _=f();}tch::Cuda::synchronize(0);t.elapsed().as_secs_f64()*1e6/50.};
    let full=||{let y=crate::dense_fp8::run_bf16(&x,&q,&s,false).to_kind(Kind::Float);y.topk(16,-1,true,true).1};
    let coarse=||{let c=Tensor::empty([m,n],(Kind::Float,dev));
        extern "C"{fn rs_draft_head_int4(x:*const std::ffi::c_void,q4:*const std::ffi::c_void,s:*const f32,y:*mut f32,m:i32,n:i32,k:i32)->i32;}
        assert_eq!(unsafe{rs_draft_head_int4(x.data_ptr(),q4.data_ptr(),s4.data_ptr().cast(),c.data_ptr().cast(),m as i32,n as i32,k as i32)},0);c};
    let c=coarse();let cand=c.topk(64,-1,true,false).1;let flat=cand.reshape([-1]);
    let wq=q.index_select(0,&flat);let ws=s.index_select(0,&flat);
    eprintln!("[head-probe] FP8 head only: {:.1} us",time(&||crate::dense_fp8::run_bf16(&x,&q,&s,false)));
    eprintln!("[head-probe] FP8 head + top16: {:.1} us",time(&full));
    eprintln!("[head-probe] INT4 coarse kernel: {:.1} us",time(&coarse));
    eprintln!("[head-probe] top64 of coarse: {:.1} us",time(&||c.topk(64,-1,true,false).1));
    eprintln!("[head-probe] gather 512 rows: {:.1} us",time(&||{let a=q.index_select(0,&flat);let _=s.index_select(0,&flat);a}));
    eprintln!("[head-probe] FP8 rescore 512: {:.1} us",time(&||crate::dense_fp8::run_bf16(&x,&wq,&ws,false)));
    let two=||{let c=coarse();let cand=c.topk(64,-1,true,false).1;let flat=cand.reshape([-1]);
        let y=crate::dense_fp8::run_bf16(&x,&q.index_select(0,&flat),&s.index_select(0,&flat),false).to_kind(Kind::Float);
        let own=y.view([m,m,64]).diagonal(0,0,1).transpose(0,1);cand.gather(-1,&own.topk(16,-1,true,true).1,false)};
    eprintln!("[head-probe] two-stage total: {:.1} us",time(&two));
    let a=full().sort(-1,false).0;let b=two().sort(-1,false).0;
    eprintln!("[head-probe] top-16 id sets equal in {}/{} rows",i64::try_from(a.eq_tensor(&b).all_dim(-1,false).sum(Kind::Int64)).unwrap(),m);
}
