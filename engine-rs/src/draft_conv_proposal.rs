//! Child of dflash: inspect complete private context storage in qualification
//! without extending the production Context API or introducing readback there.
use super::{Drafter,Context};
use tch::{Tensor,Kind};
use serde_json::{json,Value};
use std::{path::Path,ffi::OsString};
const CONV:&str="GLM53_DRAFT_CONV_FUSED";
const FINAL:&str="GLM53_DRAFT_FINAL_NORM_SELECT";
struct Restore(Vec<(&'static str,Option<OsString>)>);
impl Restore {fn new()->Self{Self([CONV,FINAL].into_iter().map(|k|(k,std::env::var_os(k))).collect())}}
impl Drop for Restore {fn drop(&mut self){for(k,v)in &self.0{match v{Some(v)=>std::env::set_var(k,v),None=>std::env::remove_var(k)}}}}
fn conv(on:bool){std::env::set_var(CONV,if on{"1"}else{"0"});}
fn exact(a:&Tensor,b:&Tensor,label:&str) {
    assert_eq!(a.kind(),b.kind(),"{label}");assert_eq!(a.size(),b.size(),"{label}");
    let bits=match a.kind(){Kind::BFloat16|Kind::Half=>Kind::Int16,Kind::Float=>Kind::Int,Kind::Int64=>Kind::Int64,_=>panic!("unexpected {label} dtype")};
    assert!(a.contiguous().view_dtype(bits).equal(&b.contiguous().view_dtype(bits)),"{label}: raw bits differ");
}
struct Buffer {bits:Tensor,address:usize,stride:Vec<i64>}
impl Buffer {
    fn take(t:&Tensor)->Self{Self{bits:t.copy(),address:t.data_ptr() as usize,stride:t.stride()}}
    fn check(&self,t:&Tensor){assert_eq!(self.address,t.data_ptr() as usize);assert_eq!(self.stride,t.stride());exact(t,&self.bits,"context buffer");}
}
struct ContextAudit {len:i64,start:i64,cursor:i64,kv:Vec<(Buffer,Buffer)>,storage:Option<Vec<(Buffer,Buffer)>>}
impl ContextAudit {
    fn take(c:&Context)->Self {Self{len:c.len,start:c.start,cursor:c.cursor,
        kv:c.kv.iter().map(|(k,v)|(Buffer::take(k),Buffer::take(v))).collect(),
        storage:c.storage.as_ref().map(|s|s.iter().map(|(k,v)|(Buffer::take(k),Buffer::take(v))).collect())}}
    fn check(&self,c:&Context) {
        assert_eq!((self.len,self.start,self.cursor),(c.len,c.start,c.cursor));assert_eq!(self.kv.len(),c.kv.len());
        for ((a,b),(k,v)) in self.kv.iter().zip(&c.kv){a.check(k);b.check(v);}
        match (&self.storage,&c.storage) {
            (None,None)=>(),(Some(a),Some(b))=>{assert_eq!(a.len(),b.len());for ((k0,v0),(k,v)) in a.iter().zip(b){k0.check(k);v0.check(v);}},
            _=>panic!("proposal changed context storage ownership"),
        }
    }
}
fn save(path:&Path,report:&Value){let tmp=path.with_extension("json.tmp");std::fs::write(&tmp,serde_json::to_string_pretty(report).unwrap()).unwrap();std::fs::rename(tmp,path).unwrap();}

pub(crate) fn check(drafter:&Drafter,target:&crate::weights::ModelWeights,out:&Path) {
    let _guard=tch::no_grad_guard();let before=crate::session::signature();let before_spec=crate::spec_session::signature();
    let restore=Restore::new();tch::manual_seed(2026092323);
    std::fs::create_dir_all(out).unwrap();let path=out.join(format!("draft-conv-real-rank{}.json",crate::tp::world().rank));
    let incoming:std::collections::BTreeMap<_,_>=std::env::vars().filter(|(k,_)|k.starts_with("GLM53_")).collect();
    let mut report=json!({"complete":false,"gate":false,"rank":crate::tp::world().rank,"cases":[],
        "scope":"real resident drafter/target embedding; synthetic target features; all proposal outputs raw bits and full context storage; no timing",
        "incoming_flags":incoming,"other_execution_flags_preserved":true});save(&path,&report);
    let expected=4*drafter.layers.len();assert_eq!(expected,20,"gate targets the actual five-layer, two-Conv/two-side drafter");
    assert!((2..=8).contains(&drafter.block_size));
    conv(false);let a_signature=crate::session::signature();let a_spec=crate::spec_session::signature();
    conv(true);assert_ne!(a_signature,crate::session::signature(),"Conv absent from target/session identity");
    assert_ne!(a_spec,crate::spec_session::signature(),"Conv absent from speculative runtime identity");
    // Final norm is isolated within each Conv comparison, then both settings
    // are exercised. FP8 / selector / cache flags are retained from the caller.
    for final_norm in [false,true] {
        std::env::set_var(FINAL,if final_norm{"1"}else{"0"});
        let mut context=drafter.empty_context();let mut length=0;
        for history in [0i64,3,129] {
            if history>length {drafter.append(&mut context,&(Tensor::randn([history-length,20480],(Kind::Float,target.device))*0.05));length=history;}
            let audit=ContextAudit::take(&context);
            conv(false);let a=drafter.propose(&context,13041,target);audit.check(&context);
            conv(true);let b=drafter.propose(&context,13041,target);audit.check(&context);
            assert_eq!((a.conv_calls,a.conv_fused_calls),(expected,0),"off arm dispatch");
            assert_eq!((b.conv_calls,b.conv_fused_calls),(expected,expected),"candidate must actually fuse every Conv call");
            assert_eq!(a.final_norm_selected,final_norm);assert_eq!(b.final_norm_selected,final_norm);
            assert_eq!(a.selector_fused,b.selector_fused);assert_eq!(a.path,b.path);
            exact(&a.ids,&b.ids,"proposal ids");exact(&a.hidden,&b.hidden,"proposal hidden");
            exact(&a.unary,&b.unary,"proposal unary");exact(&a.edges,&b.edges,"proposal edges");
            assert!(a.hidden.isfinite().all().int64_value(&[])!=0,"real finite proposal unexpectedly nonfinite");
            report["cases"].as_array_mut().unwrap().push(json!({"history":history,"final_norm_selected":final_norm,
                "reference_calls":a.conv_calls,"reference_fused_calls":a.conv_fused_calls,"candidate_calls":b.conv_calls,"candidate_fused_calls":b.conv_fused_calls,
                "ids_hidden_unary_edges_raw_bits":true,"path_equal":true,"context_full_storage_raw_bits_and_owner_unchanged":true,
                "context_len":context.len,"context_start":context.start,"context_cursor":context.cursor,
                "context_has_storage":context.storage.is_some(),"selector_fused":a.selector_fused,"path":a.path}));save(&path,&report);
        }
    }
    assert_eq!(report["cases"].as_array().unwrap().len(),6);
    drop(restore);assert_eq!(before,crate::session::signature());assert_eq!(before_spec,crate::spec_session::signature());
    report["complete"]=json!(true);report["gate"]=json!(true);report["case_count"]=json!(6);
    report["configuration_identity_checked"]=json!(true);report["environment_restored"]=json!(true);save(&path,&report);
}
