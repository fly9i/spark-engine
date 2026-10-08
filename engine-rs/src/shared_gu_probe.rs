//! Bounded source-only probe: primitive Float stages, native seam and real TP shards.
use super::{write,try_activation};
use tch::{Kind,Tensor,Device};
use serde_json::{json,Value};
use std::{path::Path,ffi::OsString,time::Instant};
const FLAG:&str="GLM53_SHARED_GU_FUSED";
const FLAGS:[&str;7]=[FLAG,"GLM53_DENSE_LT","GLM53_DENSE_LT_TABLE","GLM53_DENSE_GEMV","GLM53_DENSE_SMALL","GLM53_FP8_SHARED","GLM53_TF32"];
struct Restore{env:Vec<(&'static str,Option<OsString>)>,precision:&'static str,tf32:bool}
impl Restore{fn new()->Self{Self{env:FLAGS.into_iter().map(|f|(f,std::env::var_os(f))).collect(),
    precision:if crate::root_probe::full_f32(){if crate::root_probe::round_f32_input(){"fp32-rounded-input"}else{"fp32-compute"}}else if crate::root_probe::retain_f32(){"fp32-output"}else{"native"},
    tf32:std::env::var("GLM53_TF32").as_deref()==Ok("1")}}}
impl Drop for Restore{fn drop(&mut self){crate::root_probe::set_precision(self.precision);crate::tp::set_tf32(self.tf32);for(f,v)in &self.env{match v{Some(v)=>std::env::set_var(f,v),None=>std::env::remove_var(f)}}}}
fn flag(on:bool){std::env::set_var(FLAG,if on{"1"}else{"0"});}
fn exact(a:&Tensor,b:&Tensor,label:&str){assert_eq!(a.size(),b.size(),"{label}");assert_eq!(a.kind(),b.kind(),"{label}");let k=if a.kind()==Kind::Half{Kind::Int16}else{assert_eq!(a.kind(),Kind::Float);Kind::Int};assert!(a.contiguous().view_dtype(k).equal(&b.contiguous().view_dtype(k)),"{label}: strict raw bits including NaN payload and +/-0");}
fn save(path:&Path,r:&Value){let tmp=path.with_extension("tmp");std::fs::write(&tmp,serde_json::to_vec_pretty(r).unwrap()).unwrap();std::fs::rename(tmp,path).unwrap();}
fn oracle(g:&Tensor,u:&Tensor)->(Tensor,Tensor){let g=g.to_kind(Kind::Float).clamp(f64::NEG_INFINITY,10.);let u=u.to_kind(Kind::Float).clamp(-10.,10.);let silu=g.silu();let product=&silu*&u;(product.to_kind(Kind::Half),Tensor::stack(&[g,u,silu,product],0))}
fn all_half()->Tensor{let bits:Vec<i16>=(0..65536u32).map(|x|x as u16 as i16).collect();Tensor::from_slice(&bits).view_dtype(Kind::Half).to_device(Device::Cuda(0))}
fn primitive()->Value{
    let dev=Device::Cuda(0);let mut g=all_half();let mut u=Tensor::zeros_like(&g);
    let guard=64;let owner=Tensor::full([65536+2*guard],317.,(Kind::Half,dev));let mut out=owner.narrow(0,guard,65536);
    let mut stages=Tensor::empty([4,65536],(Kind::Float,dev));write(&g,&u,&out,Some(&stages));let _=oracle(&g,&u);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();write(&g,&u,&out,Some(&stages));crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
    let values=[0.0f64,-0.,1.,-1.,10.,-10.,f64::INFINITY,f64::NEG_INFINITY,f64::NAN,0.000000059604644775390625,-0.000000059604644775390625,65504.,-65504.,0.333251953125,-0.333251953125];
    let mut cases=Vec::new();
    for direction in 0..2{for (vindex,&value)in values.iter().enumerate(){
        if direction==0{g.copy_(&all_half());let _=u.fill_(value);}else{u.copy_(&all_half());let _=g.fill_(value);}
        let before_g=g.copy();let before_u=u.copy();let _=out.fill_(f64::NAN);let _=stages.fill_(f64::NAN);
        graph.replay();let(y,s)=oracle(&g,&u);exact(&out,&y,"exhaustive Half output");exact(&stages,&s,"Float clamp/SiLU/product stages");
        exact(&g,&before_g,"gate unchanged");exact(&u,&before_u,"up unchanged");
        assert!(owner.narrow(0,0,guard).eq(317.).all().int64_value(&[])!=0);assert!(owner.narrow(0,guard+65536,guard).eq(317.).all().int64_value(&[])!=0);
        cases.push(json!({"direction":direction,"constant_index":vindex,"raw_bits":true,"patterns":65536}));
    }}
    // Finite final replay detects stale NaN output and preserves a result after graph drop.
    let _=g.fill_(-0.);let _=u.fill_(1.);graph.replay();let(y,s)=oracle(&g,&u);exact(&out,&y,"negative-zero restore");exact(&stages,&s,"negative-zero Float stages");
    let bits=out.copy();tch::Cuda::synchronize(0);drop(graph);drop(g);drop(u);exact(&out,&bits,"output survives owners");
    json!({"cases":cases,"final_negative_zero_replay":true,"strict_nan_payload":true,"note":"Each operand exhausts all Half encodings against 15 boundary constants; not all 2^32 input pairs"})
}
fn model(wg:Tensor,wu:Tensor,wd:Tensor)->crate::weights::MoeMeta{let opt=(Kind::Float,wg.device());crate::weights::MoeMeta{w_gate:Tensor::empty([0],opt),bias:Tensor::empty([0],opt),sh_wg:wg,sh_wu:wu,sh_wd:wd}}
fn activation(m:&crate::weights::MoeMeta,x:&Tensor)->Tensor{let half=x.to_kind(Kind::Half);crate::moe::shared_activation_with_half(m,x,Some(&half))}
fn full(m:&crate::weights::MoeMeta,x:&Tensor)->Tensor{crate::weights::mm16_partial(&activation(m,x),&m.sh_wd)}
fn changed(rows:i64,turn:usize)->Tensor{let dev=Device::Cuda(0);match turn{
    1=>Tensor::full([rows,4096],-0.,(Kind::Float,dev)),
    2=>{let x=Tensor::randn([rows,4096],(Kind::Float,dev))*0.1;let _=x.narrow(1,0,1).fill_(f64::NAN);x},
    3=>Tensor::full([rows,4096],65504.,(Kind::Float,dev)),
    _=>Tensor::randn([rows,4096],(Kind::Float,dev))*0.125}}
fn real_graph(m:&crate::weights::MoeMeta,rows:i64)->Value{
    let owner=Tensor::full([rows+2,8192],f64::NAN,(Kind::Float,m.sh_wg.device()));let mut x=owner.narrow(0,1,rows).slice(1,0,8192,2);x.copy_(&changed(rows,0));
    flag(false);let _=full(m,&x);flag(true);let h=x.to_kind(Kind::Half);assert!(try_activation(&x,&m.sh_wg,&m.sh_wu,&m.sh_wd,Some(&h)).is_some());let _=full(m,&x);tch::Cuda::synchronize(0);
    flag(false);crate::tp::graph::begin().unwrap();let a=activation(m,&x);let ya=crate::weights::mm16_partial(&a,&m.sh_wd);crate::tp::graph::end().unwrap();let ga=crate::tp::graph::Owned::take();
    flag(true);crate::tp::graph::begin().unwrap();let b=activation(m,&x);let yb=crate::weights::mm16_partial(&b,&m.sh_wd);crate::tp::graph::end().unwrap();let gb=crate::tp::graph::Owned::take();
    assert_eq!(a.kind(),Kind::Float);assert_eq!(b.kind(),Kind::Half);let mut turns=Vec::new();
    for turn in 0..5{x.copy_(&changed(rows,turn));let saved=owner.copy();let _=a.shallow_clone().fill_(f64::NAN);let _=b.shallow_clone().fill_(f64::NAN);let _=ya.shallow_clone().fill_(f64::NAN);let _=yb.shallow_clone().fill_(f64::NAN);
        ga.replay();gb.replay();flag(false);let gold=activation(m,&x);let down=crate::weights::mm16_partial(&gold,&m.sh_wd);
        exact(&a,&gold,"old graph Float activation");exact(&b,&gold.to_kind(Kind::Half),"new graph Half activation");exact(&ya,&down,"old down partial");exact(&yb,&down,"new down partial");exact(&owner,&saved,"strided input and NaN padding");turns.push(json!({"turn":turn,"activation_half_and_partial_raw_bits":true}));}
    let saved=yb.copy();tch::Cuda::synchronize(0);drop(ga);drop(gb);drop(x);drop(owner);exact(&yb,&saved,"captured partial owns storage");json!({"rows":rows,"turns":turns})
}
fn fallbacks(m:&crate::weights::MoeMeta)->Value{
    let x=changed(8,0);let half=x.to_kind(Kind::Half);flag(true);let mut cases=Vec::new();
    // Match formal GEMV=1. These actual TP2 shared shapes miss GEMV/small,
    // including T1: GU N1024<1536; down K1024 is not supported by either.
    let mut dispatch=Vec::new();
    for rows in [1i64,2,8,32]{let input=changed(rows,0);let down=Tensor::zeros([rows,1024],(Kind::Float,x.device()));
        for small in [false,true]{std::env::set_var("GLM53_DENSE_SMALL",if small{"1"}else{"0"});
            for w in [&m.sh_wg,&m.sh_wu]{assert!(!crate::gemv::eligible(&input,w));assert!(!crate::gemv::small_eligible(&input,w));}
            assert!(!crate::gemv::eligible(&down,&m.sh_wd));assert!(!crate::gemv::small_eligible(&down,&m.sh_wd));
            flag(false);let old=full(m,&input);flag(true);let half=input.to_kind(Kind::Half);
            assert!(try_activation(&input,&m.sh_wg,&m.sh_wu,&m.sh_wd,Some(&half)).is_some());
            exact(&full(m,&input),&old,"GEMV1 actual shared native dispatch");
            dispatch.push(json!({"rows":rows,"gemv_enabled":true,"small_enabled":small,"GU_and_down_native":true,"candidate_hit":true,"down_raw_bits":true}));
        }
    }
    std::env::set_var("GLM53_DENSE_SMALL","0");cases.push("GEMV1-shared-native".into());cases.push("SMALL1-shared-native".into());
    // Real alternate shape predicates still fire; shared bounded guard rejects
    // them. This is dispatch evidence, not a timing/quality test of GEMV/small.
    let wide=Tensor::empty([4096,4096],(Kind::Half,x.device()));
    assert!(crate::gemv::eligible(&changed(1,0),&wide));assert!(!crate::gemv::eligible(&changed(2,0),&wide));
    assert!(try_activation(&changed(1,0),&wide,&m.sh_wu,&m.sh_wd,None).is_none());
    std::env::set_var("GLM53_DENSE_SMALL","1");assert!(crate::gemv::small_eligible(&changed(2,0),&wide));
    assert!(try_activation(&changed(2,0),&wide,&m.sh_wu,&m.sh_wd,None).is_none());std::env::set_var("GLM53_DENSE_SMALL","0");
    cases.push("GEMV1-actual-alternate-hit".into());cases.push("SMALL1-actual-alternate-hit".into());
    std::env::set_var("GLM53_DENSE_LT","1");assert!(crate::dense_lt::eligible(&x,&m.sh_wg,false).is_some());
    assert!(try_activation(&x,&m.sh_wg,&m.sh_wu,&m.sh_wd,Some(&half)).is_none());std::env::set_var("GLM53_DENSE_LT","0");cases.push("LT-actual-shape-table-hit".into());
    for mode in ["fp32-output","fp32-compute","fp32-rounded-input"]{crate::root_probe::set_precision(mode);assert!(try_activation(&x,&m.sh_wg,&m.sh_wu,&m.sh_wd,Some(&half)).is_none());cases.push(mode.to_owned());}crate::root_probe::set_precision("native");
    for w in [&m.sh_wg,&m.sh_wu,&m.sh_wd]{crate::dense_fp8::register_weight(w,"GLM53_FP8_SHARED");}
    std::env::set_var("GLM53_FP8_SHARED","1");assert!(crate::dense_fp8::registered_enabled(&m.sh_wg));assert!(try_activation(&x,&m.sh_wg,&m.sh_wu,&m.sh_wd,Some(&half)).is_none());std::env::set_var("GLM53_FP8_SHARED","0");cases.push("registered-fp8".into());
    let (_,records)=crate::root_probe::capture_once(||{assert!(try_activation(&x,&m.sh_wg,&m.sh_wu,&m.sh_wd,Some(&half)).is_none());activation(m,&x)});assert_eq!(records.len(),2);
    for (_,input,y)in &records{exact(input,&x,"diagnostic original Float input");assert_eq!(y.kind(),Kind::Float);}cases.push("recording-original-Float".into());
    let f=m.sh_wg.to_kind(Kind::Float);assert!(try_activation(&x,&f,&m.sh_wu,&m.sh_wd,Some(&half)).is_none());
    let large=changed(33,0);assert!(try_activation(&large,&m.sh_wg,&m.sh_wu,&m.sh_wd,None).is_none());
    flag(false);assert!(try_activation(&x,&m.sh_wg,&m.sh_wu,&m.sh_wd,Some(&half)).is_none());json!({"cases":cases,"dispatch":dispatch,"nonhalf_and_rows33_and_flag0":true})
}
fn timing(work:&[(usize,i64,crate::weights::MoeMeta)])->Value{let mut records=Vec::new();for rows in [1i64,8,32]{let inputs:Vec<_>=work.iter().map(|_|changed(rows,0)).collect();
    for block in 0..2{for (arm,on)in [false,true,true,false].into_iter().enumerate(){flag(on);
        for ((_,_,m),x)in work.iter().zip(&inputs){let _=full(m,x);}tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let outputs:Vec<_>=work.iter().zip(&inputs).map(|((_,_,m),x)|full(m,x)).collect();crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
        for _ in 0..3{graph.replay();}tch::Cuda::synchronize(0);let mut samples=Vec::new();
        for _ in 0..5{let start=Instant::now();for _ in 0..4{graph.replay();}tch::Cuda::synchronize(0);samples.push(start.elapsed().as_secs_f64()*1e6/(4*work.len())as f64);}
        records.push(json!({"rows":rows,"block":block,"arm":arm,"fused":on,"us_per_shared_GU_activation_down":samples}));drop(graph);drop(outputs);
    }}}
    json!({"workset_real_layers":8,"tp_shards_per_layer":2,"matrices_per_rotation":48,"weight_bytes":16*24*1024*1024u64,"rounds":records,"gemv_enabled":std::env::var("GLM53_DENSE_GEMV").as_deref()==Ok("1"),"scope":"G/U original GEMMs + activation + down FP32 partial; GEMV1 in both arms; no TP collective; two raw ABBA blocks no sample trimming"})}
pub(super) fn run(model_path:&Path,out:&Path){assert!(!crate::tp::is_tp());tch::set_num_threads(4);let _guard=tch::no_grad_guard();let _restore=Restore::new();tch::manual_seed(2026092323);
    for f in FLAGS{std::env::set_var(f,"0");}std::env::set_var("GLM53_DENSE_GEMV","1");crate::root_probe::set_precision("native");crate::tp::set_tf32(false);
    // Process-local diagnostic table, initialized only when LT1 is tested. No
    // Lt GEMM is launched with this synthetic algorithm id.
    std::fs::create_dir_all(out).unwrap();let lt=out.join("dispatch-lt-table.json");
    std::fs::write(&lt,r#"{"8,1024,4096,0":0}"#).unwrap();std::env::set_var("GLM53_DENSE_LT_TABLE",&lt);
    std::fs::create_dir_all(out).unwrap();let path=out.join("shared-gu-local.json");let mut report=json!({"gate":false,"complete":false,"real_cases":[],"scope":"source-only draft until parent runs this CLI; real weights/synthetic activations; no full model qualification"});save(&path,&report);
    report["primitive"]=primitive();save(&path,&report);
    let idx=crate::safetensors::ShardIndex::scan(model_path).unwrap();let mut work=Vec::new();
    for layer in 3..11{let mut whole=Vec::new();for name in ["gate_proj","up_proj","down_proj"]{let key=format!("model.language_model.layers.{layer}.mlp.shared_experts.{name}.weight");let(v,s)=idx.get_f32(&key).unwrap();let shape:Vec<i64>=s.into_iter().map(|n|n as i64).collect();whole.push(Tensor::from_slice(&v).view(shape.as_slice()).to_kind(Kind::Half).to_device(Device::Cuda(0)));}
        for shard in 0..2{let wg=whole[0].narrow(0,shard*1024,1024).contiguous();let wu=whole[1].narrow(0,shard*1024,1024).contiguous();let wd=whole[2].narrow(1,shard*1024,1024).contiguous();work.push((layer,shard,model(wg,wu,wd)));}}
    drop(idx);report["fallback"]=fallbacks(&work[0].2);save(&path,&report);
    for (layer,shard,m)in &work{for rows in [1i64,2,8,32]{let mut record=real_graph(m,rows);record["layer"]=json!(layer);record["shard"]=json!(shard);report["real_cases"].as_array_mut().unwrap().push(record);save(&path,&report);}}
    report["numerical_gate"]=json!(true);save(&path,&report);report["timing"]=timing(&work);report["gate"]=json!(true);report["complete"]=json!(true);save(&path,&report);
}
