//! Borrowed resident-drafter gate. No additional checkpoint load or TP setup.
//! Synthetic Float target features, real FC/norm/KV/proposal consumers.
use super::{AcceptedFeatures,eligible};
use crate::{dflash::{Context,Drafter,Candidates},weights::ModelWeights,speculative::Node};
use tch::{Tensor,Kind};
use serde_json::{json,Value};
use std::path::Path;
struct Restore(Option<std::ffi::OsString>);
impl Restore {fn new()->Self {Self(std::env::var_os("GLM53_ACCEPTED_FEATURE_VIEW"))}}
impl Drop for Restore {fn drop(&mut self){if let Some(v)=&self.0 {std::env::set_var("GLM53_ACCEPTED_FEATURE_VIEW",v);}
    else {std::env::remove_var("GLM53_ACCEPTED_FEATURE_VIEW");}}}
fn flag(on:bool){std::env::set_var("GLM53_ACCEPTED_FEATURE_VIEW",if on{"1"}else{"0"});}
fn exact(a:&Tensor,b:&Tensor,label:&str) {
    assert_eq!(a.kind(),b.kind(),"{label}");assert_eq!(a.size(),b.size(),"{label}");
    let kind=match a.kind(){Kind::Float=>Kind::Int,Kind::BFloat16|Kind::Half=>Kind::Int16,_=>a.kind()};
    assert!(a.contiguous().view_dtype(kind).equal(&b.contiguous().view_dtype(kind)),"{label} raw bits");
}
fn candidates(a:&Candidates,b:&Candidates) {
    exact(&a.ids,&b.ids,"proposal IDs");exact(&a.hidden,&b.hidden,"proposal hidden");
    exact(&a.edges,&b.edges,"proposal edges");exact(&a.unary,&b.unary,"proposal unary");assert_eq!(a.path,b.path);
}
fn contexts(drafter:&Drafter,target:&ModelWeights,a:&Context,b:&Context)->Value {
    assert!(a.equal(b));let ca=a.snapshot();let cb=b.snapshot();
    let x=drafter.propose(a,13041,target);let y=drafter.propose(b,13041,target);candidates(&x,&y);
    assert!(a.equal(&ca)&&b.equal(&cb),"proposal mutated accepted context");
    json!({"context_equal":true,"proposal_ids_hidden_edges_unary_raw_bits":true,"path":x.path})
}
fn rows(features:&Tensor,indices:&[usize])->AcceptedFeatures {
    AcceptedFeatures::Rows(indices.iter().map(|&i|features.narrow(0,i as i64,1)).collect())
}
fn chain(n:usize)->Vec<Node> {(0..n).map(|i|Node{parent:i.checked_sub(1),token:13041+i as i64}).collect()}
fn save(out:&Path,report:&Value) {
    let path=out.join(format!("feature-prefix-rank{}.json",crate::tp::world().rank));
    let temp=path.with_extension("json.tmp");std::fs::write(&temp,serde_json::to_string_pretty(report).unwrap()).unwrap();
    std::fs::rename(temp,path).unwrap();
}

pub(super) fn run(drafter:&Drafter,target:&ModelWeights,out:&Path) {
    let _restore=Restore::new();let _guard=tch::no_grad_guard();let dev=target.device;assert!(dev.is_cuda());
    // Both TP ranks must feed the same synthetic feature stream to the real
    // sharded consumers, independently of any preceding qualification probes.
    tch::manual_seed(2026092311);
    std::fs::create_dir_all(out).unwrap();let mut report=json!({"gate":false,"rank":crate::tp::world().rank,
        "scope":"synthetic feature storage + actual borrowed drafter FC/norm/KV/propose; not full target selection or performance",
        "feature_seed":2026092311i64,"cases":[]});save(out,&report);
    std::env::remove_var("GLM53_ACCEPTED_FEATURE_VIEW");assert!(!super::enabled());
    // Pure data-movement gate includes exceptional bit patterns. Do not feed
    // these into the real drafter, which expects finite model features.
    let patterns=[0u32,0x80000000,0x7fc12345,0xffc45678,0x7f800000,0xff800000,1,0x80000001,0x7f7fffff,0xff7fffff];
    let raw:Vec<i32>=(0..8*20480).map(|i|patterns[i%patterns.len()] as i32).collect();
    let special=Tensor::from_slice(&raw).view_dtype(Kind::Float).reshape([8,20480]).to_device(dev);
    for accepted in 1..=8 {
        let ids=(0..accepted).collect::<Vec<_>>();let a=rows(&special,&ids).join();let b=AcceptedFeatures::prefix(&special,accepted).join();
        exact(&a,&b,"exceptional prefix bit copy");assert_eq!(b.data_ptr(),special.data_ptr());assert_ne!(a.data_ptr(),special.data_ptr());
    }
    flag(true);assert!(eligible(&chain(8),&special));flag(false);assert!(!eligible(&chain(8),&special));flag(true);
    for parents in [vec![None,Some(0),Some(0)],vec![None,None,Some(1)]] {
        let nodes:Vec<_>=parents.iter().map(|&parent|Node{parent,token:0}).collect();
        assert!(!eligible(&nodes,&special.narrow(0,0,nodes.len() as i64)));
    }
    assert!(!eligible(&chain(8),&special.to_kind(Kind::BFloat16)));
    assert!(!eligible(&chain(8),&special.transpose(0,1)));
    let strided=Tensor::zeros([8,40960],(Kind::Float,dev)).slice(1,0,40960,2);
    assert!(!eligible(&chain(8),&strided));
    let mut base_context=drafter.empty_context();
    drafter.append(&mut base_context,&(Tensor::randn([3,20480],(Kind::Float,dev))*0.05));
    let base_snapshot=base_context.snapshot();
    for accepted in [1usize,2,4,8] {
        let mut source=Tensor::randn([8,20480],(Kind::Float,dev))*0.05;
        let storage=Tensor::full([10,20480],731.25,(Kind::Float,dev));let mut features=storage.narrow(0,1,8);
        features.copy_(&source);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();features.copy_(&source);crate::tp::graph::end().unwrap();
        let graph=crate::tp::graph::Owned::take();let mut turns=Vec::new();
        for turn in 0..3 {
            // Finite contents change while the graph's addresses stay fixed.
            let changed=Tensor::randn([8,20480],(Kind::Float,dev))*(0.03+turn as f64*0.01);
            source.copy_(&changed);graph.replay();let before=features.copy();
            let ids=(0..accepted).collect::<Vec<_>>();let old=rows(&features,&ids);let new=AcceptedFeatures::prefix(&features,accepted);
            let old_join=old.join();let new_join=new.join();exact(&old_join,&new_join,"changed graph prefix");
            assert_eq!(new_join.data_ptr(),features.data_ptr());assert_ne!(old_join.data_ptr(),features.data_ptr());
            let mut ca=base_context.snapshot();let mut cb=base_context.snapshot();
            drafter.append(&mut ca,&old_join);drafter.append(&mut cb,&new_join);
            exact(&features,&before,"append must not mutate producer features");
            let snapshot_a=ca.snapshot();let snapshot_b=cb.snapshot();
            // The next producer overwrite is enqueued after all append reads
            // on the same stream. The old source view may change, contexts may not.
            let _=features.fill_(f64::NAN);let _=source.fill_(0.125+turn as f64*0.03125);graph.replay();
            assert!(ca.equal(&snapshot_a)&&cb.equal(&snapshot_b),"accepted context aliases overwritten producer features");
            let result=contexts(drafter,target,&ca,&cb);
            assert!(storage.get(0).eq(731.25).all().int64_value(&[])!=0);
            assert!(storage.get(9).eq(731.25).all().int64_value(&[])!=0);
            turns.push(json!({"turn":turn,"context_after_source_poison_and_replay":result,"both_storage_guards":true}));
        }
        // Terminal accepted features are intentionally consumed after request
        // timing. No further producer replay may precede this append. Drop all
        // other owner handles to demonstrate the narrow handle preserves storage.
        source.copy_(&(Tensor::randn([8,20480],(Kind::Float,dev))*0.05));graph.replay();
        let old=rows(&features,&(0..accepted).collect::<Vec<_>>()).join();let pending=AcceptedFeatures::prefix(&features,accepted);
        tch::Cuda::synchronize(0);drop(graph);drop(features);drop(storage);drop(source);
        let mut ca=base_context.snapshot();let mut cb=base_context.snapshot();
        drafter.append(&mut ca,&old);drafter.append(&mut cb,&pending.join());drop(pending);
        let terminal=contexts(drafter,target,&ca,&cb);
        report["cases"].as_array_mut().unwrap().push(json!({"accepted_rows":accepted,"turns":turns,
            "terminal_delayed_append_after_graph_and_source_drop":terminal,
            "removed_cat_allocation_bytes":accepted*20480*4,"removed_logical_read_write_bytes":accepted*20480*8}));
        save(out,&report);
    }
    assert!(base_context.equal(&base_snapshot),"accepted append changed shared original prefix context");
    report["base_context_unchanged"]=json!(true);
    report["exceptional_copy_raw_bits_rows1_to8"]=json!(true);
    report["serial_and_tree_note"]=json!("Rows.join always cat; complete target selection/rejection/EOS paths require independent spec ABBA gate");
    report["gate"]=json!(true);save(out,&report);
}
