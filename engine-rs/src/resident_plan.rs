//! Explicit, bounded benchmark jobs borrowing one already-resident model.
//! No scheduler, model reload, service hook, or implicit environment trigger.
use std::{collections::{BTreeMap,HashSet},ffi::OsString,path::Path,time::Instant};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use tch::{Device,Kind,Tensor};
use crate::{forward::Engine,dflash::Drafter};

// Runtime switches already exercised by spec_probe/perf_probe. Loading format,
// TP partition, capacity, model path, Lt table and SO identities are excluded.
const BOOLEAN:&[&str]=&[
    "GLM53_PREFILL_SP","GLM53_MLA_PREFILL_F16","GLM53_PREFILL_MOE_SUM1","GLM53_FP8_PREFILL_HALF","GLM53_DSA_PREFILL_SCORE_F16","GLM53_DSA_PREFILL_SCORE_TILED","GLM53_DSA_PREFILL_SCORE_FUSED","GLM53_KV_FP8","GLM53_MLA_CHAIN_SHARED","GLM53_CHAIN_SHARED_BASE","GLM53_MHC_POST_PRE_FUSED","GLM53_HALF_INPUT_CACHE","GLM53_PREFILL_FAT_MOE","GLM53_MHC_PRE_LARGE","GLM53_MLA_PREFILL_TC","GLM53_KDA_PREFILL_FUSED","GLM53_PREFILL_EXPERT_STREAMS","GLM53_RDMA_AR","GLM53_DRAFT_FUSED_NORM","GLM53_MHC_PRE_TC","GLM53_HALF_SKINNY_K1536","GLM53_KDA_CORR_WARP","GLM53_NORM_FUSED","GLM53_DRAFT_APPEND_GRAPH","GLM53_DRAFT_CONF_TRUNC","GLM53_HALF_SKINNY","GLM53_SPEC_LOG_CONF","GLM53_SPEC_LOG_CAND","GLM53_DRAFT_GRAPH","GLM53_STATE_INPLACE","GLM53_MLA_HALF_BMM","GLM53_ROUTER_FUSED","GLM53_DSA_NODE_FUSED","GLM53_KDA_GATE_FUSED","GLM53_MHC_PRE_FUSED","GLM53_FP8_QKV_FUSED","GLM53_FP8_SKINNY","GLM53_SHARED_GU_FUSED","GLM53_FP8_SMALL_TRANSPOSE","GLM53_FP8_SMALL_PAD",
    "GLM53_DRAFT_FINAL_NORM_SELECT","GLM53_DRAFT_CONV_FUSED","GLM53_DRAFT_SELECTOR_FUSED",
    "GLM53_ACCEPTED_FEATURE_VIEW",
    "GLM53_DSA_POSITION_CAPTURE","GLM53_DSA_TOPK_BATCH","GLM53_DSA_INDEX_FUSED",
    "GLM53_KDA_CONV_DEFERRED","GLM53_MOE_INPUT_HALF_REUSE","GLM53_KDA_CORRECTION_REPLAY",
    "GLM53_MHC_POST_FOUR_STREAMS","GLM53_MHC_POST_PACKED","GLM53_MHC_POST_FUSED",
    "GLM53_KDA_FUSED","GLM53_KDA_FORK_FUSED","GLM53_KDA_CONV_CHAIN","GLM53_KDA_CHAIN_NORM","GLM53_KDA_CHAIN_RECURRENT",
    "GLM53_MOE_NO_COPY","GLM53_DRAFT_NORM_CACHE","GLM53_TARGET_TOP1_TP","GLM53_MLA_WEIGHT_CACHE",
    "GLM53_COOP_PREFETCH","GLM53_COOP_SHARED_INPUT","GLM53_COOP_LOCAL_GU","GLM53_COOP_TRANSPOSE","GLM53_COOP_CANDIDATE",
    "GLM53_DSA_ALL_VISIBLE","GLM53_DSA_VISIBLE_DIRECT","GLM53_DSA_PREFILL_LIMIT",
    "GLM53_DENSE_FP8","GLM53_KDA_FP8","GLM53_DENSE_MLP_FP8","GLM53_FP8_SHARED","GLM53_FP8_MLA","GLM53_FP8_HEAD",
    "GLM53_DRAFT_FP8_MLP","GLM53_DRAFT_FP8_ATTN","GLM53_DRAFT_FP8_CONV","GLM53_DRAFT_FP8_FC","GLM53_DRAFT_FP8_HEAD",
    "GLM53_DENSE_LT","GLM53_TF32","GLM53_DENSE_SMALL","GLM53_MLA_ACTIVE_COPY",
    "GLM53_MLA_PREFILL_BATCHED","GLM53_MLA_SCORE_2D","GLM53_PREFILL_COOP","GLM53_PREFILL_BATCH",
    "GLM53_GROUPED_INDEX_PACK","GLM53_GROUPED_SWIGLU","GLM53_GROUPED_REDUCE","GLM53_FP8_EPILOGUE",
    "GLM53_TP_MOE_PACK","GLM53_TP_SMALL_COMM_ACTIVE","GLM53_STATIC_TENSORS",
    "GLM53_DRAFT_HEAD_TP","GLM53_DRAFT_KV_BUFFER","GLM53_DRAFT_ROPE_CACHE","GLM53_DRAFT_MLP_TP","GLM53_DRAFT_TOPK_TP",
    "GLM53_SPEC_SKIP_SERIAL_BASELINE","GLM53_SPEC_REQUIRE_ABBA_EXACT","GLM53_SPEC_STAGE_PROFILE","GLM53_SPEC_LAYOUT_ABBA",
];
const PHASES:&[&str]=&[
    "fp8-small-transpose","fp8-small-pad",
    "feature-prefix",
    "dsa-index",
    "conv-deferred","half-input","kda-correction","mhc-four","mhc-packed","depth4-abba","depth6-abba",
    "target-top1","mla-cache","coop-prefetch","memory-combined","dsa-direct","coop-shared-input","sham",
    "dsa-visible","memory-core","memory-bundle","kda-chain","kda-norm","norm-cache","kda-conv","coop-transpose",
    "moe-no-copy","fp8-shared","fp8-mla","fp8-head","fp8-draft-mlp","fp8-draft-attn","fp8-draft-conv",
    "fp8-draft-fc","fp8-draft-head","dsa-fused","dsa-tiled","dsa-tensor","gqa","gqa-fused","gqa-auto","gqa-shared",
    "dsa-limit","grouped-reduce","active-copy","prefill128","fp8","sparse","lt","geometry0","geometry2","tf32",
    "depth3","depth4","depth5","depth6","depth7","adaptive",
];
const MODES:&[&str]=&["chain","tree","batch-chain","batch-tree","batch-graph-chain","batch-graph-tree","batch-adaptive-chain"];

fn number(key:&str)->Option<(usize,usize)> {Some(match key {
    "GLM53_SPEC_TOKENS"=>(8,4096),"GLM53_SPEC_ROUNDS"=>(1,16),"GLM53_SPEC_MAX_DRAFT"=>(1,7),
    "GLM53_SPEC_FULL_WARMUPS"=>(0,4),"GLM53_SPEC_GRAPH_SLOTS"=>(1,8),"GLM53_SPEC_TREE_CAPTURE_AFTER"=>(1,64),
    "GLM53_SPEC_TREE_CAPTURE_MIN_REMAINING"=>(1,4096),"GLM53_TP_MOE_PACK_MAX_ROWS"=>(1,4096),
    "GLM53_DRAFT_KV_BUFFER_MIN_CONTEXT"=>(0,1048576),"GLM53_PREFILL_MIN_ROWS"=>(1,4096),
    "GLM53_MLA_SCORE_2D_MIN_ROWS"=>(1,4096),"GLM53_PREFILL_RECON_MIN_ROWS"=>(1,4096),
    _=>return None,
})}
fn choices(key:&str)->Option<&'static [&'static str]> {Some(match key {
    "GLM53_COOP_PREFETCH_MASK"=>&["a","b","ab"],"GLM53_COOP_GEOMETRY"=>&["0","1","2"],
    "GLM53_KDA_SEQUENCE"=>&["0","1","2","3","4","5","6"],"GLM53_FP8_LARGE"=>&["0","1","2","3","4","5"],
    "GLM53_MLA_SPARSE_FUSED"=>&["0","1","2","3","4","5","6"],
    "GLM53_DRAFT_GQA"=>&["0","1","2","3","4","5"],"GLM53_DSA_SCORE_FUSED"=>&["0","1","2","3","4","5"],
    "GLM53_FP8_SPLITS"|"GLM53_DRAFT_FP8_SPLITS"=>&["1","2","4","8","16","32"],
    "GLM53_MLA_SCORE_2D_SCOPE"=>&["all","tree","decode","prefill"],
    "GLM53_SPEC_CONF_TAU"=>&["0.3","0.5","0.6","0.7","0.8","0.9"],
    "GLM53_SPEC_CONF_TAU_PCT"=>&["30","40","50","55","60","65","70","75","80","85","90"],
    _=>return None,
})}
fn optimization(key:&str)->bool {
    (BOOLEAN.contains(&key)&&!key.starts_with("GLM53_SPEC_")) || choices(key).is_some() ||
        ["GLM53_SPEC_MAX_DRAFT","GLM53_PREFILL_MIN_ROWS"].contains(&key)
}
fn known(key:&str)->bool {
    BOOLEAN.contains(&key)||number(key).is_some()||choices(key).is_some()||
        ["GLM53_R5_SWEEP","GLM53_SPEC_MODES","GLM53_SPEC_SUITE","GLM53_SPEC_ABBA_FLAG",
         "GLM53_SPEC_ABBA_EXTRA","GLM53_SPEC_ABBA_ON","GLM53_SPEC_ABBA_OFF"].contains(&key)
}
fn validate_value(key:&str,value:Option<&str>)->Result<(),String> {
    if !known(key) {return Err(format!("resident job cannot modify {key}; loading/TP/NCCL/paths/SO identity are immutable"));}
    let Some(v)=value else {return Ok(());};
    if v.len()>4096||v.contains('\0') {return Err(format!("invalid value size/NUL for {key}"));}
    let valid=if BOOLEAN.contains(&key) {["0","1"].contains(&v)}
        else if let Some((lo,hi))=number(key) {v.parse::<usize>().map_or(false,|n|(lo..=hi).contains(&n))}
        else if let Some(values)=choices(key) {values.contains(&v)}
        else {match key {
            "GLM53_R5_SWEEP"=>!v.is_empty()&&v.split(',').count()<=16&&v.split(',').all(|p|PHASES.contains(&p)),
            "GLM53_SPEC_MODES"=>!v.is_empty()&&v.split(',').count()<=7&&v.split(',').all(|m|MODES.contains(&m)),
            "GLM53_SPEC_ABBA_FLAG"=>optimization(v),
            "GLM53_SPEC_ABBA_EXTRA"=>v.is_empty()||v.split(',').all(optimization),
            "GLM53_SPEC_ABBA_OFF"|"GLM53_SPEC_ABBA_ON"=>v.parse::<usize>().map_or(false,|n|n<=4096),
            "GLM53_SPEC_SUITE"=>!v.is_empty(),_=>false,
        }};
    if valid {Ok(())}else {Err(format!("invalid resident job value {key}={v:?}"))}
}

struct Job {name:String,kind:String,env:BTreeMap<String,Option<String>>,spec:Value}
fn parse(value:&Value)->Result<Vec<Job>,String> {
    let object=value.as_object().ok_or("resident plan must be an object")?;
    if object.keys().any(|k|k!="jobs") {return Err("resident plan accepts only jobs".into());}
    let jobs=object.get("jobs").and_then(Value::as_array).ok_or("jobs must be an array")?;
    if !(1..=16).contains(&jobs.len()) {return Err("resident plan requires 1..16 jobs".into());}
    let mut seen=HashSet::new();let mut result=Vec::new();
    for spec in jobs {
        let job=spec.as_object().ok_or("job must be an object")?;
        if job.keys().any(|k|!["name","kind","env"].contains(&k.as_str())) {return Err("job accepts only name/kind/env".into());}
        let name=job.get("name").and_then(Value::as_str).ok_or("job name must be a string")?;
        if name.is_empty()||name.len()>64||!name.bytes().all(|c|c.is_ascii_alphanumeric()||c==b'_'||c==b'-')||!seen.insert(name.to_owned()) {
            return Err("job names must be unique, 1..64 ASCII letters/digits/_/-; no paths".into());
        }
        let kind=job.get("kind").and_then(Value::as_str).ok_or("job kind must be spec or sweep")?;
        if !["spec","sweep"].contains(&kind) {return Err("job kind must be spec or sweep".into());}
        let fields=job.get("env").and_then(Value::as_object).ok_or("job env must be an object")?;
        let mut env=BTreeMap::new();
        for (k,v) in fields {
            let val=if v.is_null(){None}else{Some(v.as_str().ok_or("env values must be strings or null")?.to_owned())};
            validate_value(k,val.as_deref())?;env.insert(k.clone(),val);
        }
        result.push(Job{name:name.into(),kind:kind.into(),env,spec:spec.clone()});
    }
    Ok(result)
}

/// Snapshot ALL GLM53 variables, including variables a nested sweep creates.
/// Restoring only job.env would leak SPEC_MAX_DRAFT/ABBA flags into later jobs.
struct Environment(BTreeMap<OsString,OsString>);
impl Environment {
    fn capture()->Self {Self(std::env::vars_os().filter(|(k,_)|k.to_string_lossy().starts_with("GLM53_")).collect())}
}
impl Drop for Environment {fn drop(&mut self) {
    let keys:Vec<_>=std::env::vars_os().filter(|(k,_)|k.to_string_lossy().starts_with("GLM53_")).map(|(k,_)|k).collect();
    for k in keys {std::env::remove_var(k);}
    for (k,v) in &self.0 {std::env::set_var(k,v);}
    crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
}}
fn visible_env()->BTreeMap<String,String> {
    std::env::vars().filter(|(k,_)|k.starts_with("GLM53_")).collect()
}
fn sha(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn read_bounded(path:&Path,limit:u64)->Result<Vec<u8>,String> {
    let size=std::fs::metadata(path).map_err(|e|format!("{}: {e}",path.display()))?.len();
    if size>limit {return Err(format!("{} exceeds {limit} bytes",path.display()));}
    let bytes=std::fs::read(path).map_err(|e|format!("{}: {e}",path.display()))?;
    if bytes.len() as u64>limit {return Err(format!("{} grew beyond {limit} bytes",path.display()));}Ok(bytes)
}
fn publish(path:&Path,value:&Value) {
    let tmp=path.with_extension("json.tmp");std::fs::write(&tmp,serde_json::to_vec_pretty(value).unwrap()).unwrap();
    std::fs::rename(tmp,path).unwrap();
}

/// One small, out-of-measurement TP SUM compares the complete 256-bit digest.
/// Each rank owns a separate packet row; bytes are exact FP32 integers. This
/// avoids numerical hash reductions and lets both ranks detect disagreement.
fn agree(value:&Value,dev:Device)->String {
    let tp=crate::tp::world();assert!(tp.world>=1&&tp.world<=4);
    let digest=Sha256::digest(serde_json::to_vec(value).unwrap());
    let data:Vec<f32>=digest.iter().map(|&v|v as f32).collect();
    let packet=Tensor::zeros([tp.world as i64,32],(Kind::Float,dev));
    packet.get(tp.rank as i64).copy_(&Tensor::from_slice(&data).to_device(dev));
    crate::tp::allreduce(&packet);
    let rows:Vec<f32>=Vec::try_from(packet.to_device(Device::Cpu).reshape([-1])).unwrap();
    assert!(rows.chunks_exact(32).all(|v|v==data.as_slice()),
        "resident plan/rank configuration mismatch; refusing divergent jobs/collectives");
    format!("{digest:x}")
}

fn suite_evidence()->Result<Value,String> {
    let path=std::env::var("GLM53_SPEC_SUITE").ok();
    let source=path.as_deref().unwrap_or("bench/m0-refs.json");
    let bytes=read_bounded(Path::new(source),16*1024*1024)?;
    // Parse now, before either rank enters model computation.
    let _:Value=serde_json::from_slice(&bytes).map_err(|e|format!("invalid local suite {source}: {e}"))?;
    Ok(json!({"path":source,"sha256":sha(&bytes),"bytes":bytes.len(),"builtin_refs":path.is_none()}))
}

fn effective_validation(env:&BTreeMap<String,String>)->Result<(),String> {
    for (key,value) in env {if known(key){validate_value(key,Some(value))?;}}
    // ABBA is an indirect setter. Check both arms against the destination
    // flag's domain, not merely whether the arm text looks like an integer.
    let target=env.get("GLM53_SPEC_ABBA_FLAG").map(String::as_str).or_else(||
        (env.get("GLM53_SPEC_LAYOUT_ABBA").map(String::as_str)==Some("1")).then_some("GLM53_MLA_SCORE_2D"));
    if let Some(target)=target {
        let mut targets=vec![target];
        if let Some(extra)=env.get("GLM53_SPEC_ABBA_EXTRA") {targets.extend(extra.split(',').filter(|v|!v.is_empty()));}
        for flag in targets {for (key,default) in [("GLM53_SPEC_ABBA_OFF","0"),("GLM53_SPEC_ABBA_ON","1")] {
            validate_value(flag,Some(env.get(key).map_or(default,String::as_str)))?;
        }}
    }
    Ok(())
}

/// Explicit local JSON plan only; caller decides whether an opt-in path exists.
/// Measurement kernels/timers remain wholly inside the existing probe entries.
pub fn run(eng:&mut Engine,drafter:&Drafter,path:&Path,out:&Path) {
    let tp=crate::tp::world();assert_eq!(tp.world,2,"resident benchmark plan currently qualifies TP2 only");
    assert!(eng.w.device.is_cuda());assert!(eng.fast.as_ref().expect("resident experts required").assume_hot);
    std::fs::create_dir_all(out).unwrap();
    let manifest_path=out.join(format!("resident-plan-rank{}.json",tp.rank));
    let mut manifest=json!({"gate":false,"complete":false,"rank":tp.rank,"world":tp.world,"path":path,
        "scope":"benchmark orchestration using an already resident target/drafter; orchestration wall time is NOT a performance sample",
        "timing_scope":"individual probe result JSONs own warmup/ABBA/timing; rank handshakes and manifests are outside their timers",
        "jobs":[]});publish(&manifest_path,&manifest);
    let loaded=read_bounded(path,1024*1024).and_then(|bytes|serde_json::from_slice::<Value>(&bytes).map_err(|e|e.to_string()));
    let parsed=loaded.as_ref().map_err(Clone::clone).and_then(parse);
    // Validate the entire plan on both ranks before starting the first job.
    let preflight=json!({"plan":loaded.as_ref().ok(),"error":parsed.as_ref().err(),
        "signature":crate::spec_session::signature(),"world":tp.world});
    let plan_digest=agree(&preflight,eng.w.device);
    let jobs=parsed.unwrap_or_else(|e|panic!("invalid resident plan: {e}"));
    manifest["plan"]=loaded.unwrap();manifest["plan_agreement_sha256"]=json!(plan_digest);
    manifest["caller_glm53_environment"]=json!(visible_env());publish(&manifest_path,&manifest);
    let root=path.parent().unwrap_or_else(||Path::new("."));
    let initial_misses=eng.fast.as_ref().unwrap().misses;
    for (index,job) in jobs.iter().enumerate() {
        let restore=Environment::capture();
        for (key,value) in &job.env {match value {
            Some(value)=>{
                if key=="GLM53_SPEC_SUITE" {
                    let p=Path::new(value);let resolved=if p.is_absolute(){p.to_path_buf()}else{root.join(p)};
                    std::env::set_var(key,resolved.as_os_str());
                }else{std::env::set_var(key,value);}
            },None=>std::env::remove_var(key),
        }}
        let effective=visible_env();
        let suite=suite_evidence();
        let valid=effective_validation(&effective);
        let common:BTreeMap<_,_>=effective.iter().filter(|(k,_)|known(k)).map(|(k,v)|(k.clone(),v.clone())).collect();
        let fingerprint=json!({"job":job.spec,"index":index,"signature":crate::spec_session::signature(),
            "effective_dynamic_parameters":common,"suite":suite.as_ref().ok(),"suite_error":suite.as_ref().err(),
            "effective_validation_error":valid.as_ref().err(),
            "mode":std::env::var("GLM53_SPEC_MODES").ok(),"depth":std::env::var("GLM53_SPEC_MAX_DRAFT").ok()});
        let job_out=out.join(format!("{index:02}-{}",job.name));std::fs::create_dir_all(&job_out).unwrap();
        let job_path=job_out.join(format!("job-manifest-rank{}.json",tp.rank));
        let mut result=json!({"gate":false,"complete":false,"index":index,"spec":job.spec,"rank":tp.rank,
            "signature":crate::spec_session::signature(),"effective_glm53_environment":effective,
            "suite":suite.as_ref().ok(),"suite_error":suite.as_ref().err(),"effective_validation_error":valid.as_ref().err(),
            "output":job_out,"timing_scope":"orchestration elapsed below is not a performance sample; use probe result files"});
        publish(&job_path,&result);manifest["jobs"].as_array_mut().unwrap().push(result.clone());publish(&manifest_path,&manifest);
        let digest=agree(&fingerprint,eng.w.device);suite.unwrap_or_else(|e|panic!("resident suite: {e}"));
        valid.unwrap_or_else(|e|panic!("resident parameters: {e}"));
        result["agreement_sha256"]=json!(digest);publish(&job_path,&result);
        tch::Cuda::synchronize(0);crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
        let started=Instant::now();
        match job.kind.as_str() {"spec"=>crate::spec_probe::check(eng,drafter,&job_out),
            "sweep"=>crate::perf_probe::sweep_resident(eng,drafter,&job_out),_=>unreachable!()}
        tch::Cuda::synchronize(0);
        let wall_ms=started.elapsed().as_secs_f64()*1000.;
        assert_eq!(eng.fast.as_ref().unwrap().misses,initial_misses,"resident plan unexpectedly loaded experts");
        result["probe_returned_successfully"]=json!(true);result["job_wall_ms_not_performance_sample"]=json!(wall_ms);
        result["post_job_glm53_environment_before_restore"]=json!(visible_env());
        drop(restore);
        let restored=visible_env();assert_eq!(restored,serde_json::from_value::<BTreeMap<String,String>>(manifest["caller_glm53_environment"].clone()).unwrap(),
            "resident job environment was not fully restored");
        // A completion handshake prevents one rank entering the next job while
        // the peer is still in a prior job. No request/graph state is retained.
        agree(&json!({"job":index,"completed":true,"signature":crate::spec_session::signature()}),eng.w.device);
        result["environment_restored"]=json!(true);result["gate"]=json!(true);result["complete"]=json!(true);
        publish(&job_path,&result);manifest["jobs"][index]=result;publish(&manifest_path,&manifest);
    }
    manifest["gate"]=json!(true);manifest["complete"]=json!(true);publish(&manifest_path,&manifest);
}

#[cfg(test)] mod tests {
    use super::*;
    fn plan(env:Value)->Value {json!({"jobs":[{"name":"local","kind":"spec","env":env}]})}
    #[test] fn accepts_bounded_dynamic_jobs() {
        assert!(parse(&plan(json!({"GLM53_MOE_INPUT_HALF_REUSE":"1","GLM53_SPEC_ABBA_FLAG":"GLM53_KDA_CONV_DEFERRED",
            "GLM53_SPEC_ABBA_EXTRA":"GLM53_MOE_INPUT_HALF_REUSE","GLM53_SPEC_SUITE":"suite.json","GLM53_SPEC_ROUNDS":"4"}))).is_ok());
        assert!(parse(&plan(json!({"GLM53_SPEC_ABBA_FLAG":null}))).is_ok());
    }
    #[test] fn refuses_loading_transport_or_indirect_env_mutation() {
        for key in ["GLM53_TP_RANK","GLM53_TP_WORLD","GLM53_MASTER_PORT","NCCL_IB_HCA","GLM53_MODEL",
            "GLM53_COOP_CANDIDATE_SO","GLM53_COOP_CANDIDATE_SHA","GLM53_DRAFT_MLP_SHARD_LOAD","GLM53_DENSE_TP"] {
            let env=json!({key:"1"});assert!(parse(&plan(env)).is_err(),"{key}");
            assert!(validate_value("GLM53_SPEC_ABBA_FLAG",Some(key)).is_err(),"indirect {key}");
        }
    }
    #[test] fn refuses_bad_shape_names_values_and_unknown_jobs() {
        assert!(parse(&json!({"jobs":[]})).is_err());
        assert!(parse(&json!({"jobs":(0..17).map(|i|json!({"name":format!("job{i}"),"kind":"spec","env":{}})).collect::<Vec<_>>()})).is_err());
        for name in ["../escape","a/b","","a.b"] {assert!(parse(&json!({"jobs":[{"name":name,"kind":"spec","env":{}}]})).is_err());}
        assert!(parse(&plan(json!({"GLM53_MOE_INPUT_HALF_REUSE":true}))).is_err());
        assert!(parse(&plan(json!({"GLM53_SPEC_MAX_DRAFT":"8"}))).is_err());
        assert!(parse(&plan(json!({"GLM53_R5_SWEEP":"half-input,unknown"}))).is_err());
        assert!(parse(&json!({"jobs":[{"name":"x","kind":"shell","env":{}}]})).is_err());
    }
}
