//! Optional bandwidth-oriented GEMV; no change to weight or activation precision.
use tch::{Tensor,Kind,Device};
use std::{path::Path,time::Instant};
use serde_json::json;

extern "C" {fn rs_dense_gemv(x:*const f32,w:*const u8,y:*mut f32,n:i32,k:i32,round_output:i32)->i32;}
/// Metadata-only counterpart used before a producer changes input dtype.
pub(crate) fn eligible_meta(device:Device,kind:Kind,rows:i64,cols:i64,contiguous:bool,w:&Tensor)->bool {
    std::env::var("GLM53_DENSE_GEMV").as_deref()==Ok("1") && device!=Device::Cpu &&
        kind==Kind::Float && w.kind()==Kind::Half && w.dim()==2 &&
        rows==1 && cols==w.size()[1] && w.size()[1]%8==0 &&
        w.size()[0]>=1536 && [128,1536,4096].contains(&w.size()[1]) &&
        contiguous && w.is_contiguous()
}
pub fn eligible(x:&Tensor,w:&Tensor)->bool {
    x.dim()==2 && eligible_meta(x.device(),x.kind(),x.size()[0],x.size()[1],x.is_contiguous(),w)
}
pub fn run(x:&Tensor,w:&Tensor,round:bool)->Tensor {
    let y=Tensor::empty([1,w.size()[0]],(Kind::Float,x.device()));
    assert_eq!(unsafe{rs_dense_gemv(x.data_ptr().cast(),w.data_ptr().cast(),y.data_ptr().cast(),
        w.size()[0] as i32,w.size()[1] as i32,round as i32)},0,"dense GEMV launch");y
}
fn reference(x:&Tensor,w:&Tensor,round:bool)->Tensor {
    if round {x.to_kind(Kind::Half).matmul(&w.transpose(0,1)).to_kind(Kind::Float)}
    else {crate::weights::mm16_partial(x,w)}
}
fn rel(a:&Tensor,b:&Tensor)->f64 {
    let d=a-b;((&d*&d).sum(Kind::Float)/(b*b).sum(Kind::Float).clamp_min(1e-30)).sqrt().double_value(&[])
}
pub fn probe(model:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::env::remove_var("GLM53_DENSE_GEMV");
    std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);tch::manual_seed(20260923);
    let mut idx=crate::safetensors::ShardIndex::scan(model).unwrap();let mut cases=Vec::new();
    let prefix="model.language_model.layers";
    let shapes=[("lm_head.weight".to_string(),-1),
        (format!("{prefix}.0.self_attn.q_proj.weight"),0),
        (format!("{prefix}.0.self_attn.o_proj.weight"),1),
        (format!("{prefix}.0.self_attn.f_b_proj.weight"),0),
        (format!("{prefix}.0.mlp.gate_proj.weight"),0),
        (format!("{prefix}.0.mlp.down_proj.weight"),1),
        (format!("{prefix}.3.self_attn.q_a_proj.weight"),-1),
        (format!("{prefix}.3.self_attn.q_b_proj.weight"),0),
        (format!("{prefix}.3.self_attn.kv_a_proj_with_mqa.weight"),-1),
        (format!("{prefix}.3.self_attn.o_proj.weight"),1),
        (format!("{prefix}.3.mlp.shared_experts.gate_proj.weight"),0),
        (format!("{prefix}.3.mlp.shared_experts.down_proj.weight"),1)];
    for (name,axis) in shapes {
        let (v,s)=idx.get_f32(&name).unwrap();let shape:Vec<i64>=s.iter().map(|&x|x as i64).collect();
        let whole=Tensor::from_slice(&v).view(shape.as_slice()).to_kind(Kind::Half).to_device(dev);drop(v);
        for rank in 0..if axis<0{1}else{2} {
            let w=if axis<0{whole.shallow_clone()}else{let n=whole.size()[axis as usize]/2;whole.narrow(axis,rank*n,n).contiguous()};
            let x=Tensor::randn([1,w.size()[1]],(Kind::Float,dev));let mut input=x.copy();
            for round in [false,true] {
                let expected=reference(&input,&w,round);let actual=run(&input,&w,round);
                let error=rel(&actual,&expected);assert!(error<if round{0.001}else{0.00001},"{name} error={error}");
                let sample=w.narrow(0,0,w.size()[0].min(256));
                let exact=input.to_kind(Kind::Half).to_kind(Kind::Double).matmul(&sample.to_kind(Kind::Double).transpose(0,1)).to_kind(Kind::Float);
                let exact=if round{exact.to_kind(Kind::Half).to_kind(Kind::Float)}else{exact};
                let oracle_error=rel(&actual.narrow(1,0,sample.size()[0]),&exact);
                assert!(oracle_error<if round{0.001}else{0.00001});
                let mut rounds=Vec::new();
                for custom in [false,true,true,false] {
                    let op=|x:&Tensor|if custom{run(x,&w,round)}else{reference(x,&w,round)};
                    for _ in 0..3{let _=op(&input);}tch::Cuda::synchronize(0);
                    crate::tp::graph::begin().unwrap();let y=op(&input);crate::tp::graph::end().unwrap();
                    for z in [&x,&(-&x),&(&x*0.125),&Tensor::zeros_like(&x)] {
                        input.copy_(z);crate::tp::graph::replay().unwrap();assert!(y.equal(&op(z)),"changed-input graph {name}");
                    }
                    input.copy_(&x);tch::Cuda::synchronize(0);let t=Instant::now();
                    for _ in 0..64{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                    rounds.push(json!({"custom":custom,"graph_us":t.elapsed().as_secs_f64()*1e6/64.}));
                    crate::tp::graph::destroy();
                }
                cases.push(json!({"weight":name,"rank_slice":rank,"shape":w.size(),"round_output":round,
                    "relative_l2":error,"fp64_reference_relative_l2":oracle_error,"rounds":rounds}));
            }
        }
        eprintln!("[gemv-probe] checked {name}");
        std::fs::write(out.join("local.json"),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
    }
}

pub(crate) fn small_eligible_meta(device:Device,kind:Kind,rows:i64,cols:i64,contiguous:bool,w:&Tensor)->bool {
    std::env::var("GLM53_DENSE_SMALL").as_deref()==Ok("1") && device!=Device::Cpu &&
        kind==Kind::Float && w.kind()==Kind::Half && w.dim()==2 &&
        rows==2 && cols==w.size()[1] && [4096,6144].contains(&w.size()[0]) &&
        [128,4096,6144].contains(&w.size()[1]) && w.size()[1]%8==0 &&
        contiguous && w.is_contiguous()
}
pub fn small_eligible(x:&Tensor,w:&Tensor)->bool {
    x.dim()==2 && small_eligible_meta(x.device(),x.kind(),x.size()[0],x.size()[1],x.is_contiguous(),w)
}
pub fn small(x:&Tensor,w:&Tensor,round:bool)->Tensor {
    extern "C" {fn rs_dense_small(x:*const f32,w:*const u8,y:*mut f32,m:i32,n:i32,k:i32,round:i32)->i32;}
    let y=Tensor::empty([x.size()[0],w.size()[0]],(Kind::Float,x.device()));
    assert_eq!(unsafe{rs_dense_small(x.data_ptr().cast(),w.data_ptr().cast(),y.data_ptr().cast(),
        x.size()[0] as i32,w.size()[0] as i32,w.size()[1] as i32,round as i32)},0);y
}
pub fn small_probe(model:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let dev=Device::Cuda(0);
    std::env::remove_var("GLM53_DENSE_SMALL");std::fs::create_dir_all(out).unwrap();
    let mut idx=crate::safetensors::ShardIndex::scan(model).unwrap();let mut cases=Vec::new();
    for (name,axis) in [("0.self_attn.q_proj.weight",0),("0.self_attn.f_b_proj.weight",0),
        ("0.mlp.gate_proj.weight",0),("0.mlp.down_proj.weight",1),("3.self_attn.q_a_proj.weight",-1),
        ("3.self_attn.q_b_proj.weight",0),("3.self_attn.kv_a_proj_with_mqa.weight",-1),
        ("3.self_attn.o_proj.weight",1),("3.mlp.shared_experts.gate_proj.weight",0),
        ("3.mlp.shared_experts.down_proj.weight",1)] {
        let (v,shape)=idx.get_f32(&format!("model.language_model.layers.{name}")).unwrap();
        let w=Tensor::from_slice(&v).view([shape[0] as i64,shape[1] as i64]).to_kind(Kind::Half).to_device(dev);drop(v);
        let w=if axis<0{w}else{let n=w.size()[axis as usize]/2;w.narrow(axis,crate::tp::world().rank as i64*n,n).contiguous()};
        for rows in [2,3,4,6,8] {for round in [false,true] {
            tch::manual_seed(4500+rows);let x=Tensor::randn([rows,w.size()[1]],(Kind::Float,dev));let mut input=x.copy();
            let a=small(&x,&w,round);let b=reference(&x,&w,round);let error=rel(&a,&b);
            assert!(error<if round{0.001}else{0.00001},"small-N {name}/{rows}: {error}");
            let ww=w.narrow(0,0,w.size()[0].min(128));
            let oracle=x.to_kind(Kind::Half).to_kind(Kind::Double).matmul(&ww.to_kind(Kind::Double).transpose(0,1)).to_kind(Kind::Float);
            let oracle=if round{oracle.to_kind(Kind::Half).to_kind(Kind::Float)}else{oracle};
            let oracle_error=rel(&a.narrow(1,0,ww.size()[0]),&oracle);assert!(oracle_error<if round{0.001}else{0.00001});
            let mut rounds=Vec::new();
            for custom in [false,true,true,false] {
                let call=|x:&Tensor|if custom{small(x,&w,round)}else{reference(x,&w,round)};
                for _ in 0..3 {let _=call(&input);}tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();let y=call(&input);crate::tp::graph::end().unwrap();
                for z in [&x,&(-&x),&(&x*0.125),&Tensor::zeros_like(&x)] {
                    input.copy_(z);crate::tp::graph::replay().unwrap();assert!(y.equal(&call(z)),"small-N graph");
                }
                input.copy_(&x);tch::Cuda::synchronize(0);let t=Instant::now();
                for _ in 0..64{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                rounds.push(json!({"custom":custom,"graph_us":t.elapsed().as_secs_f64()*1e6/64.}));crate::tp::graph::destroy();
            }
            cases.push(json!({"weight":name,"shape":w.size(),"rows":rows,"round_output":round,"relative_l2":error,
                "oracle_relative_l2":oracle_error,"rounds":rounds}));
        }}
        std::fs::write(out.join(format!("small-rank{}.json",crate::tp::world().rank)),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
        eprintln!("[small-N] checked {name}");
    }
}
