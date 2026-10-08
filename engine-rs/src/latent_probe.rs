//! Real checkpoint, TP head-slice, long-context and CUDA graph diagnostics.
use std::{path::Path,time::Instant};
use tch::{Tensor,Kind,Device};
use serde_json::json;

pub fn run(dir:&Path,out:&Path) {
    tch::set_num_threads(4);
    let _guard=tch::no_grad_guard();
    std::env::set_var("GLM53_MLA_LATENT","1");
    assert_eq!(crate::tp::init_from_env().world,1,"local slice probe must run without NCCL");
    std::fs::create_dir_all(out).unwrap();
    let dev=Device::Cuda(0);
    let cfg=crate::config::load(&dir.join("config.json")).unwrap();
    let checkpoints=[0,1,3,4,511,512,2047,2048,2051,2103,2111];
    let mut cases=Vec::new();
    for rank in 0..2 {
        let mut model=crate::weights::ModelWeights::load(dir,&cfg,4,dev);
        crate::weights::shard_dense_layer(&mut model.layers[3],rank,2);
        let w=model.layers[3].mla.as_ref().unwrap();
        tch::manual_seed(20260922);
        let x=Tensor::randn([2112,4096],(Kind::Float,dev))*0.02;
        let mut state=crate::mla_latent::State::new(w,2112);
        let mut archives=vec![("x".to_owned(),x.to_device(Device::Cpu))];
        let mut cq=Vec::new();let mut snapshots=Vec::new();
        let start=Instant::now();
        for pos in 0..2112 {
            if pos==508 || pos==2044 {snapshots.push((pos,state.snapshot()));}
            let (y,selected,q)=crate::mla_latent::step_record(w,&x.narrow(0,pos,1),&mut state);
            cq.push(q);
            if checkpoints.contains(&pos) {
                assert!(y.isfinite().all().int64_value(&[])!=0);
                let v:Vec<i64>=Vec::try_from(&selected.to_device(Device::Cpu)).unwrap();
                let mut valid:Vec<_>=v.into_iter().filter(|&i|i>=0).collect();
                assert!(valid.iter().all(|&i|i<=pos));
                let n=valid.len();valid.sort();valid.dedup();assert_eq!(valid.len(),n);
                assert_eq!(n,((pos+1)/4).min(512) as usize*4+((pos+1)%4) as usize);
                if pos<2048 {assert_eq!(valid,(0..=pos).collect::<Vec<_>>());}
                archives.push((format!("selected_{pos}"),selected.to_device(Device::Cpu)));
                archives.push((format!("output_{pos}"),y.to_device(Device::Cpu)));
            }
        }
        tch::Cuda::synchronize(0);
        let sequential_ms=start.elapsed().as_secs_f64()*1000.;
        archives.push(("q_resid".into(),Tensor::cat(&cq,0).to_device(Device::Cpu)));
        archives.push(("latent".into(),state.latent.to_device(Device::Cpu)));
        archives.push(("pools".into(),state.index.pools.to_device(Device::Cpu)));
        let mut replays=Vec::new();
        for (pos,snap) in snapshots {
            let mut input=x.narrow(0,pos,1).copy();
            state.restore(&snap);state.ensure_room(16);
            let mut expected=Vec::new();
            for t in pos..pos+16 {expected.push(crate::mla_latent::step(w,&x.narrow(0,t,1),&mut state));}
            let expected_state=state.snapshot();
            state.restore(&snap);
            let _=crate::mla_latent::step(w,&input,&mut state);
            tch::Cuda::synchronize(0);
            state.restore(&snap);
            crate::tp::graph::begin().unwrap();
            let output=crate::mla_latent::step(w,&input,&mut state);
            crate::tp::graph::end().unwrap();
            let mut error=0f64;
            for _ in 0..2 {
                state.restore(&snap);
                for (offset,t) in (pos..pos+16).enumerate() {
                    input.copy_(&x.narrow(0,t,1));crate::tp::graph::replay().unwrap();
                    error=error.max(f64::try_from((&output-&expected[offset]).abs().max()).unwrap());
                }
                assert_eq!(state.max_diff(&expected_state),0.,"graph state differs at {pos}");
            }
            crate::tp::graph::destroy();
            assert_eq!(error,0.,"graph output differs at {pos}");
            replays.push(json!({"start":pos,"steps":16,"rounds":2,"max_abs":error,"state_max_abs":0}));
        }
        Tensor::save_multi(&archives,out.join(format!("slice{rank}.pt"))).unwrap();
        cases.push(json!({"slice":rank,"positions":2112,"checkpoints":checkpoints,"sequential_ms":sequential_ms,"replays":replays,
            "latent_bytes":2112*512*2,"index_bytes":(528*128+2*4*128)*4,
            "derived_weights_bytes":32*512*512*4}));
        std::fs::write(out.join("local.json"),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
        eprintln!("[latent-probe] slice{rank} PASS 2112 tokens + graph boundaries, {sequential_ms:.1} ms");
    }
}
