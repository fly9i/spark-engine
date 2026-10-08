//! Full-model precision ablation and same-input/state attention diagnostics.
use std::{cell::RefCell,collections::HashMap,path::Path,sync::atomic::{AtomicBool,AtomicUsize,Ordering}};
use serde_json::{json,Value};
use tch::{Device,Kind,Tensor};
use crate::forward::{Engine,LayerState};
static ACTIVE:AtomicBool=AtomicBool::new(false);
static HEAD:AtomicUsize=AtomicUsize::new(0);
static TRACE:AtomicBool=AtomicBool::new(false);
static ATTN:AtomicBool=AtomicBool::new(false);
thread_local! {
    static INPUT_REPLAY:RefCell<Option<(Vec<Tensor>,usize)>>=const{RefCell::new(None)};
    static NEXT_INPUT:RefCell<Option<Tensor>>=const{RefCell::new(None)};
    static CACHE:RefCell<HashMap<usize,Tensor>>=RefCell::new(HashMap::new());
    static SELECTED:RefCell<Vec<Tensor>>=const{RefCell::new(Vec::new())};
    static ROUTER:RefCell<Vec<(Tensor,Tensor)>>=const{RefCell::new(Vec::new())};
    static ATTENTION:RefCell<(String,Vec<(String,Tensor)>)>=const{RefCell::new((String::new(),Vec::new()))};
}
pub fn native_head(w:&Tensor)->bool { ACTIVE.load(Ordering::Relaxed) && w.data_ptr() as usize==HEAD.load(Ordering::Relaxed) }
pub fn begin_layer(){if ACTIVE.load(Ordering::Relaxed){CACHE.with(|c|c.borrow_mut().clear());}}
pub fn float_weight(w:&Tensor)->Tensor {
    if !ACTIVE.load(Ordering::Relaxed){return w.to_kind(Kind::Float);}
    CACHE.with(|c|c.borrow_mut().entry(w.data_ptr() as usize).or_insert_with(||w.to_kind(Kind::Float)).shallow_clone())
}
/// True when router inputs/scores are traced or replaced (fused router must defer to ATen).
pub(crate) fn router_hooks_active()->bool {TRACE.load(Ordering::Relaxed)||INPUT_REPLAY.with(|p|p.borrow().is_some())}
pub fn record_router(h:&Tensor,s:&Tensor){
    INPUT_REPLAY.with(|p|{if let Some((plan,index))=p.borrow_mut().as_mut(){
        let x=plan.get(*index).expect("expert-input replay exhausted");assert_eq!(x.size(),h.size());
        NEXT_INPUT.with(|n|{assert!(n.borrow().is_none(),"unused expert input");*n.borrow_mut()=Some(x.shallow_clone());});*index+=1;
    }});
    if TRACE.load(Ordering::Relaxed){ROUTER.with(|r|r.borrow_mut().push((h.shallow_clone(),s.shallow_clone())));}}
pub fn record_selection(ids:&Tensor){if TRACE.load(Ordering::Relaxed){SELECTED.with(|s|s.borrow_mut().push(ids.shallow_clone()));}}
/// A routed-only input override is queued (the next expert_input is not z's plain Half).
pub(crate) fn next_input_pending()->bool {NEXT_INPUT.with(|n|n.borrow().is_some())}
pub fn expert_input(x:&Tensor)->Tensor {
    NEXT_INPUT.with(|n|n.borrow_mut().take()).unwrap_or_else(||x.to_kind(Kind::Half))
}
/// Scoped routed-only override for local dataflow checks; no normal-path work.
pub(crate) fn with_expert_input<T>(input:&Tensor,f:impl FnOnce()->T)->T {
    struct Restore(Option<Tensor>);
    impl Drop for Restore {fn drop(&mut self) {
        NEXT_INPUT.with(|n|*n.borrow_mut()=self.0.take());
    }}
    let _restore=Restore(NEXT_INPUT.with(|n|n.borrow_mut().replace(input.shallow_clone())));
    let value=f();
    NEXT_INPUT.with(|n|assert!(n.borrow().is_none(),"scoped expert input was not consumed"));
    value
}
fn state_archive(out:&mut Vec<(String,Tensor)>,prefix:&str,s:&LayerState){
    match s {
        LayerState::Kda(s)=>{out.push((format!("{prefix}_h"),s.h.to_device(Device::Cpu)));out.push((format!("{prefix}_conv"),s.conv.to_device(Device::Cpu)));},
        LayerState::Mla(s)=>{out.push((format!("{prefix}_k"),s.k.to_device(Device::Cpu)));out.push((format!("{prefix}_v"),s.v.to_device(Device::Cpu)));},
        _=>panic!("deep diagnostic requires eager states"),
    }
}
pub fn before_attention(i:usize,x:&Tensor,s:Option<&LayerState>){if ATTN.load(Ordering::Relaxed){ATTENTION.with(|a|{
    let mut a=a.borrow_mut();let prefix=format!("{}_L{i:02}",a.0);a.1.push((format!("{prefix}_input"),x.to_device(Device::Cpu)));
    if let Some(s)=s{state_archive(&mut a.1,&format!("{prefix}_pre"),s);}
});}}
pub fn after_attention(i:usize,y:&Tensor,s:&LayerState){if ATTN.load(Ordering::Relaxed){ATTENTION.with(|a|{
    let mut a=a.borrow_mut();let prefix=format!("{}_L{i:02}",a.0);a.1.push((format!("{prefix}_output"),y.to_device(Device::Cpu)));state_archive(&mut a.1,&format!("{prefix}_post"),s);
});}}
fn number(t:Tensor)->f64 {f64::try_from(t).unwrap()}
fn relative(a:&Tensor,b:&Tensor)->Tensor {
    let a=a.to_kind(Kind::Double);let b=b.to_kind(Kind::Double);
    assert_eq!(a.size(),b.size());assert_eq!(number(a.isfinite().all().to_kind(Kind::Double)),1.);assert_eq!(number(b.isfinite().all().to_kind(Kind::Double)),1.);
    (&a-&b).norm_scalaropt_dim(2.,[-1],false)/a.norm_scalaropt_dim(2.,[-1],false).clamp_min(1e-30)
}
fn diff(a:&Tensor,b:&Tensor)->Value {let r=relative(&a.reshape([1,-1]),&b.reshape([1,-1]));json!({"relative_l2":number(r.get(0)),"max_abs":number((a-b).abs().max())})}
fn ints(v:&Value)->Vec<i64>{v.as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).collect()}
fn write(path:&Path,v:&Value){std::fs::write(path,serde_json::to_string_pretty(v).unwrap()).unwrap();}
fn teacher(eng:&mut Engine,case:&Value,capture:bool)->(Vec<(String,Tensor)>,Vec<(String,Tensor)>){
    let ids=ints(&case["prompt_ids"]);let reference=ints(&case["reference_ids"]);
    assert!(!reference.is_empty());
    ROUTER.with(|r|r.borrow_mut().clear());SELECTED.with(|s|s.borrow_mut().clear());TRACE.store(true,Ordering::Relaxed);
    if capture {ATTENTION.with(|a|*a.borrow_mut()=("prefill".into(),Vec::new()));ATTN.store(true,Ordering::Relaxed);}
    let (logits,mut state)=eng.prefill(&Tensor::from_slice(&ids).to_device(eng.w.device));
    ATTN.store(false,Ordering::Relaxed);
    let mut lg=logits.get(logits.size()[0]-1);let mut rows=Vec::new();
    let target=case["capture_teacher_input"].as_u64().unwrap_or(0) as usize;
    for (p,&tok) in reference.iter().enumerate(){
        rows.push(lg.to_device(Device::Cpu));
        if p+1<reference.len(){
            if capture && p==target {ATTENTION.with(|a|a.borrow_mut().0="decode".into());ATTN.store(true,Ordering::Relaxed);}
            lg=eng.step(tok,&mut state);ATTN.store(false,Ordering::Relaxed);
        }
    }
    TRACE.store(false,Ordering::Relaxed);begin_layer();
    let mut archive=vec![("logits".into(),Tensor::stack(&rows,0))];
    for (i,(h,s)) in ROUTER.with(|r|std::mem::take(&mut *r.borrow_mut())).into_iter().enumerate(){
        archive.push((format!("input_{i:05}"),h.to_device(Device::Cpu)));archive.push((format!("score_{i:05}"),s.to_device(Device::Cpu)));
    }
    let selected=SELECTED.with(|s|std::mem::take(&mut *s.borrow_mut()));
    assert_eq!(selected.len(),42*reference.len());
    for (i,ids) in selected.into_iter().enumerate(){archive.push((format!("selected_{i:05}"),ids.to_device(Device::Cpu)));}
    let attn=if capture {ATTENTION.with(|a|std::mem::take(&mut a.borrow_mut().1))}else{Vec::new()};
    (archive,attn)
}
fn compare(gold:&HashMap<String,Tensor>,cand:&HashMap<String,Tensor>,case:&Value)->Value {
    let reference=ints(&case["reference_ids"]);let mut out=crate::evaluation::distribution(&gold["logits"],&cand["logits"],&reference);
    let mut routes=Vec::new();let mut first=Vec::new();
    for call in 0..42*reference.len(){
        let key=format!("score_{call:05}");let a=&gold[&key];let b=&cand[&key];
        let av=a.topk(9,-1,true,true).0;
        let selected=|m:&HashMap<String,Tensor>,score:&Tensor|m.get(&format!("selected_{call:05}")).map(Tensor::shallow_clone).unwrap_or_else(||score.to_device(Device::Cuda(0)).topk(8,-1,true,true).1.to_device(Device::Cpu));
        let ai=selected(gold,a);let bi=selected(cand,b);
        let same=ai.sort(-1,false).0.eq_tensor(&bi.sort(-1,false).0).all_dim(-1,false);
        let delta=(a-b).abs().max_dim(-1,false).0;let gap=av.select(1,7)-av.select(1,8);
        let hs=relative(&gold[&format!("input_{call:05}")],&cand[&format!("input_{call:05}")]);
        let mut changed=Vec::new();
        for t in 0..a.size()[0]{if same.int64_value(&[t])==0 {
            let g=gap.double_value(&[t]);let d=delta.double_value(&[t]);assert!(g<=2.*d+1e-12,"top-k flip violates score perturbation bound");
            changed.push(json!({"token":t,"gap8_9":g,"max_score_delta":d,"router_input_relative_l2":hs.double_value(&[t])}));
        }}
        let phase=if call<42{"prefill"}else{"decode"};let step=if call<42{None}else{Some(call/42-1)};
        if !changed.is_empty(){first.push(json!({"layer":call%42+3,"phase":phase,"teacher_input":step,"first_changed_token":changed[0]}));}
        routes.push(json!({"layer":call%42+3,"phase":phase,"teacher_input":step,"tokens":a.size()[0],"changed_count":changed.len(),
            "input_relative_l2_mean":number(hs.mean(Kind::Double)),"input_relative_l2_max":number(hs.max()),"score_delta_max":number(delta.max()),"flips":changed}));
    }
    out["routing"]=json!(routes);out["first_divergence_by_execution_order"]=json!(first.first());out
}
fn state_from(gold:&HashMap<String,Tensor>,p:&str,kda:bool,rank:usize,dev:Device)->LayerState {
    let rank=rank as i64;
    if kda{
        let h=gold[&format!("{p}_h")].narrow(0,rank*32,32).to_device(dev);
        let c=&gold[&format!("{p}_conv")];let conv=Tensor::cat(&(0..3).map(|j|c.narrow(1,j*8192+rank*4096,4096)).collect::<Vec<_>>(),1).to_device(dev);
        LayerState::Kda(crate::kda::KdaState{h,conv})
    }else{
        let k=gold[&format!("{p}_k")].narrow(1,rank*32,32).to_device(dev);
        let v=gold[&format!("{p}_v")].narrow(1,rank*32,32).to_device(dev);let len=k.size()[0];
        LayerState::Mla(crate::mla::MlaState{k,v,len})
    }
}
fn attention_check(eng:&Engine,path:&Path)->Vec<Value>{
    let gold:HashMap<_,_>=Tensor::load_multi_with_device(path,Device::Cpu).unwrap().into_iter().collect();
    let dev=eng.w.device;let rank=crate::tp::world().rank;let mut results=Vec::new();
    for phase in ["prefill","decode"]{for (i,l) in eng.w.layers.iter().enumerate(){
        begin_layer();let p=format!("{phase}_L{i:02}");let x=gold[&format!("{p}_input")].to_device(dev);
        let mut state=if phase=="decode" {state_from(&gold,&format!("{p}_pre"),l.kda.is_some(),rank,dev)}else if l.kda.is_some(){
            LayerState::Kda(crate::kda::KdaState::with_heads(dev,32))
        }else{LayerState::Mla(crate::mla::MlaState::with_heads(dev,32))};
        let y=match &mut state {
            LayerState::Kda(s)=>{let mut rows=Vec::new();for t in 0..x.size()[0]{rows.push(crate::kda::kda_step(l.kda.as_ref().unwrap(),&x.get(t).unsqueeze(0),s));}Tensor::cat(&rows,0)},
            LayerState::Mla(s)=>{let mut rows=Vec::new();for t in 0..x.size()[0]{rows.push(crate::mla::mla_step(l.mla.as_ref().unwrap(),&x.get(t).unsqueeze(0),s));}Tensor::cat(&rows,0)},
            _=>unreachable!(),
        };
        let expected=state_from(&gold,&format!("{p}_post"),l.kda.is_some(),rank,dev);
        let sd=match(&expected,&state){
            (LayerState::Kda(a),LayerState::Kda(b))=>vec![diff(&a.h,&b.h),diff(&a.conv,&b.conv)],
            (LayerState::Mla(a),LayerState::Mla(b))=>vec![diff(&a.k,&b.k),diff(&a.v,&b.v)],_=>unreachable!()};
        results.push(json!({"layer":i,"phase":phase,"output":diff(&gold[&format!("{p}_output")],&y.to_device(Device::Cpu)),"states":sd}));
    }}begin_layer();results
}
pub fn run(dir:&Path,suite_path:&Path,out:&Path){
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);assert!(crate::weights::w_fp16());
    for key in ["GLM53_GRAPH","GLM53_PROFILE","GLM53_PROFILE2","GLM53_NO_SELDEV","GLM53_NO_FAST","GLM53_NO_FAST_PREFILL"]{assert!(std::env::var(key).is_err());}
    tch::set_num_threads(4);std::fs::create_dir_all(out).unwrap();
    let suite:Value=serde_json::from_str(&std::fs::read_to_string(suite_path).unwrap()).unwrap();let cases=suite["cases"].as_array().unwrap();
    let cfg=crate::config::load(&dir.join("config.json")).unwrap();assert_eq!(cfg.num_hidden_layers,45);
    std::env::set_var("GLM53_DENSE_TP","0");crate::root_probe::set_precision("fp32-compute");crate::root_probe::set_precision("native");let dev=Device::Cuda(0);
    let w=crate::weights::ModelWeights::load(dir,&cfg,45,dev);
    HEAD.store(w.lm_head.data_ptr() as usize,Ordering::Relaxed);ACTIVE.store(true,Ordering::Relaxed);
    let mut fast=crate::moefast::MoeFast::new(dir,45,cfg.n_routed_experts,45*cfg.n_routed_experts+16,dev);
    fast.preload_all(45,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};let misses=eng.fast.as_ref().unwrap().misses;
    let mut all=Vec::new();
    for layout in ["replicated","dense"]{
        if layout=="dense"{begin_layer();tch::Cuda::synchronize(0);for l in &mut eng.w.layers{crate::weights::shard_dense_layer(l,tp.rank,tp.world);}std::env::set_var("GLM53_DENSE_TP","1");}
        for mode in ["native","fp32-compute","fp32-rounded-input"]{
            crate::root_probe::set_precision(mode);begin_layer();let folder=out.join(mode).join(layout);std::fs::create_dir_all(&folder).unwrap();
            for case in cases{
                let name=case["name"].as_str().unwrap();let capture=case["attention_check"].as_bool().unwrap_or(false);
                let t=std::time::Instant::now();let (archive,attn)=teacher(&mut eng,case,capture&&layout=="replicated");
                Tensor::save_multi(&archive,folder.join(format!("{name}.pt"))).unwrap();
                if !attn.is_empty(){Tensor::save_multi(&attn,folder.join(format!("{name}-attention.pt"))).unwrap();}
                if layout=="dense"{
                    let g:HashMap<_,_>=Tensor::load_multi_with_device(out.join(mode).join("replicated").join(format!("{name}.pt")),Device::Cpu).unwrap().into_iter().collect();
                    let c:HashMap<_,_>=archive.into_iter().collect();let mut result=compare(&g,&c,case);result["name"]=json!(name);result["mode"]=json!(mode);
                    if capture{result["same_input_state_attention"]=json!(attention_check(&eng,&out.join(mode).join("replicated").join(format!("{name}-attention.pt"))));}
                    println!("[deep-metrics] rank={} mode={mode} case={name} KL={} raw_max={} top1={}",tp.rank,result["kl_baseline_to_candidate_nats"]["mean"],result["raw_logits_relative_l2"]["max"],result["top1_equal_fraction"]);
                    all.push(result);write(&out.join("metrics.json"),&json!({"rank":tp.rank,"results":all,"lm_head":"native in all arms","experts":"same EXL3 fast path and FP16 inputs in all arms","graph":false,"teacher_only":true}));
                }
                println!("[deep-case] rank={} layout={layout} mode={mode} case={name} seconds={:.2}",tp.rank,t.elapsed().as_secs_f64());
            }
        }
    }
    assert_eq!(eng.fast.as_ref().unwrap().misses,misses);crate::root_probe::set_precision("native");begin_layer();ACTIVE.store(false,Ordering::Relaxed);
    println!("[deep-complete] rank={} cold_loads=0",tp.rank);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recorded_tie_selection_is_authoritative() {
        let case=json!({"reference_ids":[0]});let mut gold=HashMap::new();let mut cand=HashMap::new();
        gold.insert("logits".into(),Tensor::zeros([1,2],(Kind::Float,Device::Cpu)));
        cand.insert("logits".into(),Tensor::zeros([1,2],(Kind::Float,Device::Cpu)));
        for call in 0..42 {
            for m in [&mut gold,&mut cand] {
                m.insert(format!("input_{call:05}"),Tensor::ones([1,4],(Kind::Float,Device::Cpu)));
                m.insert(format!("score_{call:05}"),Tensor::from_slice(&[9f32,8.,7.,6.,5.,4.,3.,2.,2.]).view([1,9]));
                m.insert(format!("selected_{call:05}"),Tensor::arange(8,(Kind::Int64,Device::Cpu)).view([1,8]));
            }
        }
        cand.insert("selected_00000".into(),Tensor::from_slice(&[0i64,1,2,3,4,5,6,8]).view([1,8]));
        let result=compare(&gold,&cand,&case);
        assert_eq!(result["routing"][0]["changed_count"],1);
        assert_eq!(result["routing"][0]["flips"][0]["gap8_9"],0.);
        assert_eq!(result["routing"][0]["flips"][0]["max_score_delta"],0.);
    }
    #[test]
    fn router_trace_identifies_decode_boundary_flip() {
        let case=json!({"reference_ids":[0,1]});let mut gold=HashMap::new();let mut cand=HashMap::new();
        gold.insert("logits".into(),Tensor::from_slice(&[1f32,0.,0.,1.]).view([2,2]));
        cand.insert("logits".into(),Tensor::from_slice(&[1f32,0.,0.,1.]).view([2,2]));
        for call in 0..84 {
            let input=Tensor::ones([1,4],(Kind::Float,Device::Cpu));let s=Tensor::from_slice(&[9f32,8.,7.,6.,5.,4.,3.,2.,1.]).view([1,9]);
            gold.insert(format!("input_{call:05}"),input.copy());cand.insert(format!("input_{call:05}"),input.copy());
            gold.insert(format!("score_{call:05}"),s.copy());
            gold.insert(format!("selected_{call:05}"),s.topk(8,-1,true,true).1);
            let b=s.copy();if call==42{let _=b.narrow(1,8,1).fill_(2.25);}
            cand.insert(format!("selected_{call:05}"),b.topk(8,-1,true,true).1);
            cand.insert(format!("score_{call:05}"),b);
        }
        let m=compare(&gold,&cand,&case);let first=&m["first_divergence_by_execution_order"];
        assert_eq!(first["layer"],3);assert_eq!(first["phase"],"decode");assert_eq!(first["teacher_input"],0);
        assert_eq!(m["routing"][42]["changed_count"],1);assert_eq!(m["top1_equal_fraction"],1.);
        assert_eq!(m["routing"][42]["flips"][0]["gap8_9"],1.);
    }
}

/// Controlled intervention on remaining routed-expert input propagation. Not valid deployment.
pub fn input_replay_run(dir:&Path,suite_path:&Path,gold_root:&Path,out:&Path){
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);tch::set_num_threads(4);
    for key in ["GLM53_GRAPH","GLM53_PROFILE","GLM53_PROFILE2","GLM53_NO_SELDEV","GLM53_NO_FAST","GLM53_NO_FAST_PREFILL"]{assert!(std::env::var(key).is_err());}
    assert!(crate::weights::w_fp16());
    std::fs::create_dir_all(out).unwrap();let suite:Value=serde_json::from_str(&std::fs::read_to_string(suite_path).unwrap()).unwrap();
    let cfg=crate::config::load(&dir.join("config.json")).unwrap();assert_eq!(cfg.num_hidden_layers,45);let dev=Device::Cuda(0);
    std::env::set_var("GLM53_DENSE_TP","1");crate::root_probe::set_precision("fp32-compute");
    let w=crate::weights::ModelWeights::load(dir,&cfg,45,dev);HEAD.store(w.lm_head.data_ptr() as usize,Ordering::Relaxed);ACTIVE.store(true,Ordering::Relaxed);
    let mut fast=crate::moefast::MoeFast::new(dir,45,cfg.n_routed_experts,45*cfg.n_routed_experts+16,dev);fast.preload_all(45,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};let misses=eng.fast.as_ref().unwrap().misses;let mut results=Vec::new();
    for case in suite["cases"].as_array().unwrap(){let name=case["name"].as_str().unwrap();if !["hello","structured_count"].contains(&name){continue;}
        let gold:HashMap<_,_>=Tensor::load_multi_with_device(gold_root.join("replicated").join(format!("{name}.pt")),Device::Cpu).unwrap().into_iter().collect();
        for mode in ["natural","fixed_expert_inputs"]{
            if mode=="fixed_expert_inputs"{let plan=(0..42*ints(&case["reference_ids"]).len()).map(|i|gold[&format!("input_{i:05}")].to_kind(Kind::Half).to_device(dev)).collect();INPUT_REPLAY.with(|p|*p.borrow_mut()=Some((plan,0)));}
            let (archive,_)=teacher(&mut eng,case,false);
            if mode=="fixed_expert_inputs"{INPUT_REPLAY.with(|p|{let (plan,n)=p.borrow_mut().take().unwrap();assert_eq!(plan.len(),n);});NEXT_INPUT.with(|n|assert!(n.borrow().is_none()));}
            let cand:HashMap<_,_>=archive.into_iter().collect();let mut result=compare(&gold,&cand,case);
            if mode=="natural"{let previous:HashMap<_,_>=Tensor::load_multi_with_device(gold_root.join("dense").join(format!("{name}.pt")),Device::Cpu).unwrap().into_iter().collect();assert!(previous["logits"].equal(&cand["logits"]),"natural rerun changed");result["previous_natural_logits_bitwise_equal"]=json!(true);}
            let saved:Vec<_>=cand.iter().map(|(k,v)|(k.as_str(),v)).collect();Tensor::save_multi(&saved,out.join(format!("{name}-{mode}.pt"))).unwrap();
            result["name"]=json!(name);result["mode"]=json!(mode);println!("[input-replay] rank={} case={name} mode={mode} KL={} rawmax={}",tp.rank,result["kl_baseline_to_candidate_nats"]["mean"],result["raw_logits_relative_l2"]["max"]);
            results.push(result);write(&out.join("metrics.json"),&json!({"rank":tp.rank,"results":results,"note":"Diagnostic only: replay baseline quantized inputs for routed experts, retain natural routing and current router coefficients. This removes all input perturbations into routed experts, not just quantization effects."}));
        }
    }
    assert_eq!(eng.fast.as_ref().unwrap().misses,misses);crate::root_probe::set_precision("native");begin_layer();ACTIVE.store(false,Ordering::Relaxed);println!("[input-replay-complete] rank={} cold_loads=0",tp.rank);
}

/// Recompute old summaries with CUDA top-k, and validate against actual historical selections.
pub fn recompare(suite_path:&Path,root:&Path,historical:&Path){
    tch::set_num_threads(4);let suite:Value=serde_json::from_str(&std::fs::read_to_string(suite_path).unwrap()).unwrap();
    let old:Value=serde_json::from_str(&std::fs::read_to_string(root.join("metrics.json")).unwrap()).unwrap();let mut results=Vec::new();let mut checks=Vec::new();
    for case in suite["cases"].as_array().unwrap(){let name=case["name"].as_str().unwrap();for mode in ["native","fp32-compute","fp32-rounded-input"]{
        let mut arms=Vec::new();for layout in ["replicated","dense"]{
            let mut m:HashMap<_,_>=Tensor::load_multi_with_device(root.join(mode).join(layout).join(format!("{name}.pt")),Device::Cpu).unwrap().into_iter().collect();
            let historic=if mode=="native"{Some(Tensor::load_multi_with_device(historical.join(layout).join(format!("{name}.pt")),Device::Cpu).unwrap().into_iter().collect::<HashMap<_,_>>())}else{None};
            for call in 0..42*ints(&case["reference_ids"]).len(){let ids=m[&format!("score_{call:05}")].to_device(Device::Cuda(0)).topk(8,-1,true,true).1.to_device(Device::Cpu);
                if let Some(h)=&historic{assert!(ids.equal(&h[&format!("route_ids_{call:05}")]),"CUDA reconstruction differs from actual GPU selection: {name}/{layout}/{call}");}
                m.insert(format!("selected_{call:05}"),ids);
            }
            if historic.is_some(){checks.push(json!({"name":name,"layout":layout,"actual_historical_ids_exact":true}));}
            arms.push(m);
        }
        let mut result=compare(&arms[0],&arms[1],case);result["name"]=json!(name);result["mode"]=json!(mode);
        let previous=old["results"].as_array().unwrap().iter().find(|r|r["name"]==name&&r["mode"]==mode).unwrap();
        if let Some(v)=previous.get("same_input_state_attention"){result["same_input_state_attention"]=v.clone();}
        results.push(result);
    }}
    write(&root.join("metrics.json"),&json!({"rank":old["rank"],"results":results,"selection_source":"CUDA reconstruction from saved float32 scores, same shapes; exact validated against historical actual GPU IDs for all native cases","selection_validation":checks,"lm_head":"native in all arms","experts":"same EXL3 fast path and FP16 inputs in all arms","graph":false,"teacher_only":true}));
    println!("[recompare] historical GPU selections all exact");
}

/// Same-model 2x2 experiment: dense arithmetic and EXL3 output rounding varied independently.
pub fn expert_precision_run(dir:&Path,suite_path:&Path,historical:&Path,out:&Path){
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);assert!(crate::weights::w_fp16());
    for key in ["GLM53_GRAPH","GLM53_PROFILE","GLM53_PROFILE2","GLM53_NO_SELDEV","GLM53_NO_FAST","GLM53_NO_FAST_PREFILL"]{assert!(std::env::var(key).is_err());}
    tch::set_num_threads(4);std::fs::create_dir_all(out).unwrap();
    let suite:Value=serde_json::from_str(&std::fs::read_to_string(suite_path).unwrap()).unwrap();
    let cfg=crate::config::load(&dir.join("config.json")).unwrap();assert_eq!(cfg.num_hidden_layers,45);let dev=Device::Cuda(0);
    std::env::set_var("GLM53_DENSE_TP","0");crate::root_probe::set_precision("fp32-compute");crate::root_probe::set_precision("native");
    let w=crate::weights::ModelWeights::load(dir,&cfg,45,dev);HEAD.store(w.lm_head.data_ptr() as usize,Ordering::Relaxed);ACTIVE.store(true,Ordering::Relaxed);
    let mut fast=crate::moefast::MoeFast::new(dir,45,cfg.n_routed_experts,45*cfg.n_routed_experts+16,dev);fast.preload_all(45,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(dir,4)};let misses=eng.fast.as_ref().unwrap().misses;let mut results=Vec::new();
    for layout in ["replicated","dense"]{
        if layout=="dense"{begin_layer();tch::Cuda::synchronize(0);for l in &mut eng.w.layers{crate::weights::shard_dense_layer(l,tp.rank,tp.world);}std::env::set_var("GLM53_DENSE_TP","1");}
        for dense in ["native","fp32-compute"]{for expert in ["native","all_proj_fp32"]{
            crate::root_probe::set_precision(dense);begin_layer();eng.fast.as_mut().unwrap().projection_precision=if expert=="native"{0}else{2};
            let mode=format!("{dense}--{expert}");let folder=out.join(&mode).join(layout);std::fs::create_dir_all(&folder).unwrap();
            for case in suite["cases"].as_array().unwrap(){
                let name=case["name"].as_str().unwrap();let start=std::time::Instant::now();let (archive,_)=teacher(&mut eng,case,false);let cand:HashMap<_,_>=archive.into_iter().collect();
                if expert=="native"{
                    let old:HashMap<_,_>=Tensor::load_multi_with_device(historical.join(dense).join(layout).join(format!("{name}.pt")),Device::Cpu).unwrap().into_iter().collect();
                    assert!(old["logits"].equal(&cand["logits"]),"native control no longer bitwise reproduces historical run: {mode}/{layout}/{name}");
                }
                Tensor::save_multi(&cand.iter().map(|(k,v)|(k.as_str(),v)).collect::<Vec<_>>(),folder.join(format!("{name}.pt"))).unwrap();
                if layout=="dense"{
                    let gold:HashMap<_,_>=Tensor::load_multi_with_device(out.join(&mode).join("replicated").join(format!("{name}.pt")),Device::Cpu).unwrap().into_iter().collect();let mut result=compare(&gold,&cand,case);result["name"]=json!(name);result["mode"]=json!(mode);result["native_control_bitwise_equal"]=json!(expert=="native");
                    println!("[expert-deep-metrics] rank={} mode={mode} case={name} KL={} TVmax={} rawmax={}",tp.rank,result["kl_baseline_to_candidate_nats"]["mean"],result["total_variation"]["max"],result["raw_logits_relative_l2"]["max"]);
                    results.push(result);write(&out.join("metrics.json"),&json!({"rank":tp.rank,"results":results,"note":"2x2 dense precision / EXL3 output precision; same quantized expert weights, natural routing, native LM head, eager teacher forcing"}));
                }
                println!("[expert-deep-case] rank={} mode={mode} layout={layout} case={name} seconds={:.2}",tp.rank,start.elapsed().as_secs_f64());
            }
        }}
    }
    assert_eq!(eng.fast.as_ref().unwrap().misses,misses);eng.fast.as_mut().unwrap().projection_precision=0;crate::root_probe::set_precision("native");begin_layer();ACTIVE.store(false,Ordering::Relaxed);
    println!("[expert-deep-complete] rank={} cold_loads=0",tp.rank);
}
