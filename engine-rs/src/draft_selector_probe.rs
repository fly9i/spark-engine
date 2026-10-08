//! Independent local path gate; no model load or TP init in run().
use super::{into,old_path,try_path};
use tch::{Tensor,Kind,Device};
use serde_json::{json,Value};
use std::{path::Path,time::Instant};
struct Restore(Option<std::ffi::OsString>);
impl Restore {fn new()->Self{Self(std::env::var_os("GLM53_DRAFT_SELECTOR_FUSED"))}}
impl Drop for Restore {fn drop(&mut self){if let Some(v)=&self.0{std::env::set_var("GLM53_DRAFT_SELECTOR_FUSED",v);}
    else{std::env::remove_var("GLM53_DRAFT_SELECTOR_FUSED");}}}
fn flag(on:bool){std::env::set_var("GLM53_DRAFT_SELECTOR_FUSED",if on{"1"}else{"0"});}
fn exact(a:&Tensor,b:&Tensor,label:&str) {
    assert_eq!(a.kind(),b.kind(),"{label}");assert_eq!(a.size(),b.size(),"{label}");
    let same=if a.kind()==Kind::Float {a.contiguous().view_dtype(Kind::Int).equal(&b.contiguous().view_dtype(Kind::Int))}else{a.equal(b)};
    assert!(same,"{label}");
}
fn fixture(steps:i64,mode:usize,turn:usize)->(Tensor,Tensor) {
    let patterns=[0x7f800000u32,0xff800000,0x7fc12345,0xffc45678,0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff];
    let mut score=Vec::<i32>::new();let mut ids=Vec::new();
    for t in 0..steps {for p in 0..16 {for c in 0..16 {
        let winner=(p*5+t*3+turn as i64)%16;
        let bits=match mode {
            0=>{let v=if c==winner{100.}else{-(c as f32)};v.to_bits()},
            1=>if (c+turn as i64)%2==0 {0}else{0x80000000},
            2=>0xff800000,3=>0x7f800000,
            4=>if c==winner||c==(winner+3)%16 {0x7fc12345}else{(c as f32).to_bits()},
            5=>patterns[((t*256+p*16+c) as usize+turn)%patterns.len()],
            6=>if c==winner{1u32}else{0x80000001},
            7=>0xff7fffff,
            _=>unreachable!(),
        };score.push(bits as i32);
    }}for c in 0..16 {ids.push(match (t*16+c+turn as i64)%5 {0=>i64::MIN,1=>i64::MAX,_=>t*1000+c*17+turn as i64});}}
    (Tensor::from_slice(&score).view_dtype(Kind::Float).view([steps,16,16]).to_device(Device::Cuda(0)),
        Tensor::from_slice(&ids).view([steps,16]).to_device(Device::Cuda(0)))
}
// Independent serial scan: preserve the first NaN, otherwise the first max.
fn cpu_path(edges:&Tensor,ids:&Tensor)->Tensor {
    let e:Vec<f32>=Vec::try_from(edges.flatten(0,-1).to_device(Device::Cpu)).unwrap();
    let id:Vec<i64>=Vec::try_from(ids.flatten(0,-1).to_device(Device::Cpu)).unwrap();let mut prev=0usize;let mut path=Vec::new();
    for t in 0..edges.size()[0] as usize {
        let row=&e[(t*16+prev)*16..(t*16+prev+1)*16];let mut best=0;
        for c in 1..16 {if !row[best].is_nan()&&(row[c].is_nan()||row[c]>row[best]){best=c;}}
        path.push(id[t*16+best]);prev=best;
    }
    Tensor::from_slice(&path).to_device(edges.device())
}
fn one_shape(steps:i64)->Value {
    let dev=Device::Cuda(0);let guard=64;
    let score_storage=Tensor::full([steps*256+2*guard],f64::NAN,(Kind::Float,dev));
    let id_storage=Tensor::full([steps*16+2*guard],-92399i64,(Kind::Int64,dev));
    let storage=Tensor::full([steps+2*guard],-73125i64,(Kind::Int64,dev));
    let mut edges=score_storage.narrow(0,guard,steps*256).view([steps,16,16]);
    let mut ids=id_storage.narrow(0,guard,steps*16).view([steps,16]);let mut out=storage.narrow(0,guard,steps);
    let (e,i)=fixture(steps,0,0);edges.copy_(&e);ids.copy_(&i);into(&edges,&ids,&out);let _=old_path(&edges,&ids);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();into(&edges,&ids,&out);crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
    let mut records=Vec::new();let mut previous:Option<(Tensor,Tensor)>=None;
    for turn in 0..4 {for mode in 0..8 {
        let (e,i)=fixture(steps,mode,turn);edges.copy_(&e);ids.copy_(&i);
        let before_e=score_storage.copy();let before_i=id_storage.copy();let _=out.fill_(-67231i64);
        graph.replay();let gold=old_path(&edges,&ids);exact(&out,&gold,"ATen path including NaN/tie order");exact(&out,&cpu_path(&edges,&ids),"independent CPU scan");
        for g in [storage.narrow(0,0,guard),storage.narrow(0,guard+steps,guard)]{assert!(g.eq(-73125i64).all().int64_value(&[])!=0);}
        exact(&score_storage,&before_e,"edge inputs and guard bits unchanged");exact(&id_storage,&before_i,"candidate IDs and guards unchanged");
        flag(true);let actual=try_path(&edges,&ids).unwrap();exact(&actual,&gold,"public fresh allocation");
        flag(false);assert!(try_path(&edges,&ids).is_none());
        if let Some((a,b))=&previous{exact(a,b,"previous output survives changed replay/new allocation");}previous=Some((actual,gold));
        records.push(json!({"turn":turn,"mode":mode,"graph_exact":true,"cpu_exact":true,"input_raw_bits":true,"output_guards":true}));
    }}
    tch::Cuda::synchronize(0);drop(graph);if let Some((a,b))=&previous{exact(a,b,"public output survives graph destruction");}
    flag(true);assert!(try_path(&edges.transpose(1,2),&ids).is_none());assert!(try_path(&edges.to_kind(Kind::Half),&ids).is_none());
    assert!(try_path(&edges,&ids.to_kind(Kind::Int)).is_none());assert!(try_path(&edges.to_device(Device::Cpu),&ids.to_device(Device::Cpu)).is_none());
    let strided=Tensor::zeros([steps,32],(Kind::Int64,dev)).slice(1,0,32,2);assert!(try_path(&edges,&strided).is_none());
    json!({"steps":steps,"records":records,"fallback_checked":true,"owner_lifetime":true})
}
fn timing()->Value {
    let inputs:Vec<_>=(0..16).map(|i|fixture(7,0,i)).collect();let gold:Vec<_>=inputs.iter().map(|(e,i)|old_path(e,i)).collect();let mut rounds=Vec::new();
    for execution in ["graph","eager"] {for block in 0..2 {for (arm,on) in [false,true,true,false].into_iter().enumerate() {
        flag(on);let call=|(e,i):&(Tensor,Tensor)|if on{try_path(e,i).unwrap()}else{old_path(e,i)};
        for input in &inputs{let _=call(input);}tch::Cuda::synchronize(0);
        let (graph,outputs)=if execution=="graph" {
            crate::tp::graph::begin().unwrap();let outputs:Vec<_>=inputs.iter().map(call).collect();crate::tp::graph::end().unwrap();
            (Some(crate::tp::graph::Owned::take()),outputs)
        }else{(None,inputs.iter().map(call).collect::<Vec<_>>())};
        if let Some(g)=&graph{g.replay();}for (a,b) in outputs.iter().zip(&gold){exact(a,b,"timing path");}
        let mut samples=Vec::new();for _ in 0..5 {
            tch::Cuda::synchronize(0);let start=Instant::now();for _ in 0..4 {
                if let Some(g)=&graph{g.replay();}else{for input in &inputs{let _=call(input);}}
            }tch::Cuda::synchronize(0);samples.push(start.elapsed().as_secs_f64()*1e6/(4*16) as f64);
        }
        rounds.push(json!({"execution":execution,"block":block,"arm":arm,"fused":on,"us_per_path":samples}));drop(graph);drop(outputs);
    }}}
    json!({"sets":16,"edge_id_workset_bytes":16*(7*16*16*4+7*16*8),"rounds":rounds,
        "scope":"precomputed synthetic edges; device path only, eager batches exclude per-proposal D2H synchronization, not whole drafter speed"})
}
pub(super) fn run(out:&Path) {
    assert!(!crate::tp::is_tp());let _restore=Restore::new();let _guard=tch::no_grad_guard();tch::set_num_threads(4);
    std::fs::create_dir_all(out).unwrap();let path=out.join("draft-selector-local.json");
    std::fs::write(&path,r#"{"complete":false,"gate":false}"#).unwrap();std::fs::remove_file(out.join("draft-selector-timing.json")).ok();
    std::env::remove_var("GLM53_DRAFT_SELECTOR_FUSED");assert!(!super::enabled());let mut cases=Vec::new();
    for steps in 1..=7 {cases.push(one_shape(steps));std::fs::write(&path,serde_json::to_string_pretty(&json!({"complete":false,"gate":false,"cases":cases})).unwrap()).unwrap();}
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"complete":true,"gate":true,"cases":cases,"scope":"local synthetic exact path only; real-producer/TP request gate still required"})).unwrap()).unwrap();
    std::fs::write(out.join("draft-selector-timing.json"),serde_json::to_string_pretty(&timing()).unwrap()).unwrap();
}
/// Real original head/edge producer; local check additionally proves dispatch.
pub(super) fn check(drafter:&crate::dflash::Drafter,target:&crate::weights::ModelWeights,out:&Path) {
    let _restore=Restore::new();let _guard=tch::no_grad_guard();tch::manual_seed(2026092317);std::fs::create_dir_all(out).unwrap();
    let path=out.join(format!("draft-selector-real-rank{}.json",crate::tp::world().rank));let mut report=json!({"gate":false,"cases":[],"scope":"real drafter; synthetic target features; unchanged head/edges and path parity, not natural task quality"});
    std::fs::write(&path,serde_json::to_string_pretty(&report).unwrap()).unwrap();
    let mut context=drafter.empty_context();let mut length=0;
    for goal in [0i64,3,129] {
        if goal>length{let feature=Tensor::randn([goal-length,20480],(Kind::Float,target.device))*0.05;drafter.append(&mut context,&feature);length=goal;}
        let snapshot=context.snapshot();flag(false);let a=drafter.propose(&context,13041,target);flag(true);let b=drafter.propose(&context,13041,target);
        assert!(!a.selector_fused,"off reference must use the original path");
        assert!(b.selector_fused,"real producer must execute the fused selector, not merely enable its flag");
        for (x,y) in [(&a.ids,&b.ids),(&a.edges,&b.edges),(&a.unary,&b.unary)]{exact(x,y,"real producer must remain unchanged");}
        assert_eq!(a.hidden.kind(),Kind::BFloat16);assert_eq!(b.hidden.kind(),Kind::BFloat16);
        assert!(a.hidden.contiguous().view_dtype(Kind::Int16).equal(&b.hidden.contiguous().view_dtype(Kind::Int16)));assert_eq!(a.path,b.path);assert!(context.equal(&snapshot));
        let direct=try_path(&a.edges,&a.ids).expect("actual producer layout must qualify");exact(&direct,&old_path(&a.edges,&a.ids),"real edges direct kernel");
        assert_eq!(Vec::<i64>::try_from(direct.to_device(Device::Cpu)).unwrap(),a.path);
        report["cases"].as_array_mut().unwrap().push(json!({"history":goal,"path":a.path,"producer_bits_exact":true,"direct_kernel_exact":true,
            "reference_selector_fused":a.selector_fused,"candidate_selector_fused":b.selector_fused,"context_unchanged":true}));
        std::fs::write(&path,serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
    report["gate"]=json!(true);std::fs::write(&path,serde_json::to_string_pretty(&report).unwrap()).unwrap();
}
