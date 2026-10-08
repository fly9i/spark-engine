//! Diagnostic isolation of dense TP, before the first routed expert.
use std::{cell::RefCell, collections::{HashMap,HashSet}, path::Path, sync::atomic::{AtomicBool,Ordering}};
use tch::{Device,Kind,Tensor};
use serde_json::{json,Value};
use crate::{weights::{self,LayerWeights,ModelWeights},mhc};
static ROUND_F32_INPUT: AtomicBool=AtomicBool::new(false);
pub fn round_f32_input()->bool { ROUND_F32_INPUT.load(Ordering::Relaxed) }
static FULL_F32: AtomicBool=AtomicBool::new(false);
pub fn full_f32()->bool { FULL_F32.load(Ordering::Relaxed) }
extern "C" { fn rs_disable_tf32() -> i32; }
static RETAIN_F32: AtomicBool=AtomicBool::new(false);
pub fn retain_f32()->bool { RETAIN_F32.load(Ordering::Relaxed) }
static RECORD: AtomicBool=AtomicBool::new(false);
pub(crate) fn recording()->bool {RECORD.load(Ordering::Relaxed)}
thread_local! { static CAPTURE: RefCell<(HashSet<usize>,Vec<(usize,Tensor,Tensor)>)>=RefCell::new((HashSet::new(),Vec::new())); }
pub fn record(x:&Tensor,w:&Tensor,y:&Tensor) {
    if !RECORD.load(Ordering::Relaxed) {return;}
    CAPTURE.with(|c| {let mut c=c.borrow_mut();let key=w.data_ptr() as usize;
        if c.0.insert(key) {c.1.push((key,x.copy(),y.copy()));}
    });
}
/// Isolated local diagnostic capture. Production record() remains unchanged;
/// nested captures and panic unwinding restore the previous recorder exactly.
pub(crate) fn capture_once<T>(f:impl FnOnce()->T)->(T,Vec<(usize,Tensor,Tensor)>) {
    struct Restore {enabled:bool,previous:Option<(HashSet<usize>,Vec<(usize,Tensor,Tensor)>)>}
    impl Drop for Restore {fn drop(&mut self) {
        RECORD.store(self.enabled,Ordering::Relaxed);
        CAPTURE.with(|c|*c.borrow_mut()=self.previous.take().unwrap());
    }}
    let previous=CAPTURE.with(|c|std::mem::take(&mut *c.borrow_mut()));
    let _restore=Restore{enabled:RECORD.swap(true,Ordering::Relaxed),previous:Some(previous)};
    let value=f();let captures=CAPTURE.with(|c|std::mem::take(&mut c.borrow_mut().1));
    (value,captures)
}
fn f(t:Tensor)->f64 {f64::try_from(t).unwrap()}
fn diff(a:&Tensor,b:&Tensor)->Value {
    assert_eq!(a.size(),b.size());
    let a=a.to_kind(Kind::Double);let b=b.to_kind(Kind::Double);let d=&a-&b;
    assert!(f(a.isfinite().all().to_kind(Kind::Double))==1. && f(b.isfinite().all().to_kind(Kind::Double))==1.);
    json!({"relative_l2":f((&d*&d).sum(Kind::Double).sqrt())/f((&a*&a).sum(Kind::Double).sqrt()).max(1e-30),
        "max_abs":f(d.abs().max()),"unequal_fraction":f(a.ne_tensor(&b).to_kind(Kind::Double).mean(Kind::Double))})
}
// Enumerate actual model tensors and their declared partition axes, including replicated fields.
fn fields(l:&LayerWeights)->Vec<(String,&Tensor,Option<i64>)> {
    let mut out=Vec::new();
    macro_rules! add {($w:ident,$prefix:expr,$axis:expr,$($field:ident),+) => {$(out.push((format!("{}.{}",$prefix,stringify!($field)),&$w.$field,$axis));)+};}
    let h=&l.hc;
    add!(h,"hc",None,attn_fn,attn_scale,attn_base,ffn_fn,ffn_scale,ffn_base,in_ln,post_ln);
    if let Some(w)=&l.kda {
        add!(w,"kda",Some(0),wq,wk,wv,wb,fb,gb,conv_q,conv_k,conv_v,dt_bias,a_log);
        add!(w,"kda",Some(1),wo);add!(w,"kda",None,fa,ga,o_norm);
    }
    if let Some(w)=&l.mla {
        add!(w,"mla",Some(0),q_b,kv_b);add!(w,"mla",Some(1),wo);
        add!(w,"mla",None,q_a,kv_a,q_a_ln,kv_a_ln);
    }
    if let Some(w)=&l.dense {add!(w,"dense",Some(0),wg,wu);add!(w,"dense",Some(1),wd);}
    if let Some(w)=&l.moe {add!(w,"moe",Some(0),sh_wg,sh_wu);add!(w,"moe",Some(1),sh_wd);add!(w,"moe",None,w_gate,bias);}
    out
}
struct Trace {z:Tensor,a:Tensor,states:Vec<Tensor>,z2:Tensor,m:Option<Tensor>,residual:Tensor}
fn attention(l:&LayerWeights,x:&Tensor)->(Tensor,Vec<Tensor>) {
    if let Some(w)=&l.kda {
        let (y,s)=crate::kda::kda_forward_state(w,x,crate::kda::KdaState::with_heads(x.device(),w.wq.size()[0]/128));
        (y,vec![s.h,s.conv])
    } else {
        let w=l.mla.as_ref().unwrap();
        let (y,s)=crate::mla::mla_forward_state(w,x,crate::mla::MlaState::with_heads(x.device(),w.q_b.size()[0]/256));
        (y,vec![s.k,s.v])
    }
}
fn mlp(l:&LayerWeights,x:&Tensor)->Tensor {
    let w=l.dense.as_ref().unwrap();
    weights::row_mm16(&(weights::mm16(x,&w.wg).silu()*weights::mm16(x,&w.wu)),&w.wd)
}
fn prefix(w:&ModelWeights,ids:&Tensor)->Vec<Trace> {
    let mut r=mhc::hc_expand(&w.embed_tokens(ids));let mut out=Vec::new();
    for l in w.layers.iter().take(4) {
        let (pre,z)=mhc::mhc_pre(&r,&l.hc.attn_fn,&l.hc.attn_scale,&l.hc.attn_base,&l.hc.in_ln);
        let (a,states)=attention(l,&z);r=mhc::mhc_post(&a,&r,&pre);
        let (pre,z2)=mhc::mhc_pre(&r,&l.hc.ffn_fn,&l.hc.ffn_scale,&l.hc.ffn_base,&l.hc.post_ln);
        let m=if l.dense.is_some(){Some(mlp(l,&z2))}else{None};
        if let Some(m)=&m {r=mhc::mhc_post(m,&r,&pre);}
        out.push(Trace{z,a,states,z2,m,residual:r.shallow_clone()});
    }
    out
}
fn projections(w:&ModelWeights)->Vec<Value> {
    let map:HashMap<usize,(String,&Tensor,Option<i64>)>=w.layers.iter().enumerate().flat_map(|(i,l)|fields(l).into_iter().map(move |(s,t,d)|(t.data_ptr() as usize,(format!("L{i}.{s}"),t,d)))).collect();
    let captures=CAPTURE.with(|c| std::mem::take(&mut c.borrow_mut().1));let mut out=Vec::new();
    for (ptr,x,base) in captures {
        let (name,ww,axis)=map.get(&ptr).unwrap();
        let Some(axis)=axis else {continue};
        let d=ww.size()[*axis as usize]/2;
        let s0=ww.narrow(*axis,0,d).copy();let s1=ww.narrow(*axis,d,d).copy();
        let split=if *axis==0 {Tensor::cat(&[weights::mm16(&x,&s0),weights::mm16(&x,&s1)],1)} else {
            {let y=weights::mm16_partial(&x.narrow(1,0,d),&s0)+weights::mm16_partial(&x.narrow(1,d,d),&s1);
                if retain_f32(){y}else{y.to_kind(Kind::Half).to_kind(Kind::Float)}}
        };
        let y_full_f32=weights::mm16_partial(&x,ww);
        let full_f32_round=y_full_f32.to_kind(Kind::Half).to_kind(Kind::Float);
        // Uniform output samples plus worst split-vs-base output rows. CPU double evaluates
        // the actual operands used by each precision mode, avoiding a different checkpoint / TF32 reference.
        let n=ww.size()[0];
        let uniform=Tensor::arange(32,(Kind::Int64,x.device()))*(n/32);
        let worst=(&base-&split).abs().max_dim(0,false).0.topk(32,0,true,true).1;
        let ix=Tensor::cat(&[uniform,worst],0);
        let xd=if full_f32() && !round_f32_input(){x.shallow_clone()}else{x.to_kind(Kind::Half)}.to_device(Device::Cpu).to_kind(Kind::Double);
        let wd=ww.index_select(0,&ix).to_device(Device::Cpu).to_kind(Kind::Double);
        let reference=xd.matmul(&wd.transpose(0,1));
        let rounded=reference.to_kind(Kind::Half).to_kind(Kind::Double);
        let sample=|a:&Tensor|a.index_select(1,&ix).to_device(Device::Cpu).to_kind(Kind::Double);
        let base_s=sample(&base);let split_s=sample(&split);let f32_s=sample(&full_f32_round);
        out.push(json!({"name":name,"axis":axis,"input_shape":x.size(),"weight_shape":ww.size(),
            "split_vs_base":diff(&base,&split),"full_f32_round_vs_base":diff(&base,&full_f32_round),
            "reference_output_indices":Vec::<i64>::try_from(ix.to_device(Device::Cpu)).unwrap(),
            "base_vs_double":diff(&reference,&base_s),"split_vs_double":diff(&reference,&split_s),
            "base_vs_correct_half":diff(&rounded,&base_s),"split_vs_correct_half":diff(&rounded,&split_s),
            "full_f32_vs_correct_half":diff(&rounded,&f32_s),"full_f32_vs_double":diff(&reference,&sample(&y_full_f32))}));
    }
    out
}
fn local_state(l:&LayerWeights,s:&Tensor,index:usize,rank:i64)->Tensor {
    if l.kda.is_some(){if index==0{s.narrow(0,rank*32,32).copy()}else{
        Tensor::cat(&(0..3).map(|part|s.narrow(1,part*8192+rank*4096,4096)).collect::<Vec<_>>(),1)
    }}else{s.narrow(1,rank*32,32).copy()}
}
pub fn set_precision(mode:&str) {
    assert!(["native","fp32-output","fp32-compute","fp32-rounded-input"].contains(&mode));
    let fp32_output=mode!="native";
    RETAIN_F32.store(fp32_output,Ordering::Relaxed);
    FULL_F32.store(mode=="fp32-compute" || mode=="fp32-rounded-input",Ordering::Relaxed);
    ROUND_F32_INPUT.store(mode=="fp32-rounded-input",Ordering::Relaxed);
    if full_f32() {assert_eq!(unsafe{rs_disable_tf32()},0);}
}
pub fn run(dir:&Path,suite:&Path,outdir:&Path,case_name:&str,mode:&str) {
    set_precision(mode);
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);assert!(weights::w_fp16());
    tch::set_num_threads(8);
    std::env::remove_var("GLM53_DENSE_TP");
    let cfg=crate::config::load(&dir.join("config.json")).unwrap();
    let mut w=ModelWeights::load(dir,&cfg,45,Device::Cuda(0));
    let suite:Value=serde_json::from_str(&std::fs::read_to_string(suite).unwrap()).unwrap();
    let case=suite["cases"].as_array().unwrap().iter().find(|c|c["name"]==case_name).unwrap();
    let ids:Vec<i64>=case["prompt_ids"].as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect();
    let ids=Tensor::from_slice(&ids).to_device(w.device);
    RECORD.store(true,Ordering::Relaxed);let base=prefix(&w,&ids);RECORD.store(false,Ordering::Relaxed);
    let proj=projections(&w);
    eprintln!("[root] rank{} {} projection comparisons complete",tp.rank,proj.len());
    let mut audit=Vec::new();
    for (i,l) in w.layers.iter_mut().enumerate() {
        let old:Vec<_>=fields(l).into_iter().map(|(n,t,d)|(n,t.shallow_clone(),d)).collect();
        weights::shard_dense_layer(l,tp.rank,tp.world);
        for ((name,original,axis),(new_name,local,new_axis)) in old.iter().zip(fields(l)) {
            assert_eq!(name,&new_name);assert_eq!(axis,&new_axis);
            let expected=if let Some(d)=axis{original.narrow(*d,tp.rank as i64*(original.size()[*d as usize]/2),original.size()[*d as usize]/2)}else{original.shallow_clone()};
            let exact=expected.equal(local);assert!(exact,"shard mismatch L{i}.{name}");
            audit.push(json!({"layer":i,"name":name,"axis":axis,"original_shape":original.size(),"local_shape":local.size(),"exact":exact}));
        }
    }
    eprintln!("[root] rank{} {} exact tensor checks complete",tp.rank,audit.len());
    std::env::set_var("GLM53_DENSE_TP","1");
    let natural=prefix(&w,&ids);let mut layers=Vec::new();
    for (i,l) in w.layers.iter().take(4).enumerate(){
        let (a,ss)=attention(l,&base[i].z);
        let states:Vec<Value>=ss.iter().enumerate().map(|(j,s)|diff(&local_state(l,&base[i].states[j],j,tp.rank as i64),s)).collect();
        let m=l.dense.as_ref().map(|_|diff(base[i].m.as_ref().unwrap(),&mlp(l,&base[i].z2)));
        layers.push(json!({"layer":i,"same_input_attention":diff(&base[i].a,&a),"same_input_states":states,"same_input_mlp":m,
            "propagated_attn_input":diff(&base[i].z,&natural[i].z),"propagated_attn_output":diff(&base[i].a,&natural[i].a),
            "propagated_ffn_input":diff(&base[i].z2,&natural[i].z2),"propagated_residual":diff(&base[i].residual,&natural[i].residual)}));
    }
    let m=w.layers[3].moe.as_ref().unwrap();
    let score=|x:&Tensor|x.matmul(&m.w_gate.transpose(0,1)).sigmoid()+&m.bias;
    let a=score(&base[3].z2);let b=score(&natural[3].z2);let (values,ai)=a.topk(9,-1,true,true);let bi=b.topk(8,-1,true,true).1;
    let delta=(&a-&b).abs().max_dim(-1,false).0;let margin=values.select(1,7)-values.select(1,8);
    let changed=ai.narrow(1,0,8).sort(-1,false).0.ne_tensor(&bi.sort(-1,false).0).any_dim(-1,false);
    let mut positions=Vec::new();for t in 0..a.size()[0]{positions.push(json!({"position":t,"changed":changed.int64_value(&[t])!=0,
        "baseline_gap8_9":margin.double_value(&[t]),"max_score_perturbation":delta.double_value(&[t]),
        "baseline_top9":Vec::<i64>::try_from(ai.get(t).to_device(Device::Cpu)).unwrap(),"candidate_top8":Vec::<i64>::try_from(bi.get(t).to_device(Device::Cpu)).unwrap()}));}
    let report=json!({"case":case_name,"precision_mode":mode,"retain_fp32_projection_output":retain_f32(),"rank":tp.rank,"scope":"all 45 dense weight layers; prefix through layer3 router, no routed experts; FP32 states; eager",
        "weight_audit":audit,"projections":proj,"layers":layers,"first_router":{"score_diff":diff(&a,&b),"positions":positions}});
    std::fs::create_dir_all(outdir).unwrap();std::fs::write(outdir.join(format!("{case_name}-rank{}.json",tp.rank)),serde_json::to_string_pretty(&report).unwrap()).unwrap();
    eprintln!("[root] rank{} finished",tp.rank);
}
