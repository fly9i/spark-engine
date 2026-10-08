//! TP vocabulary shards: actual weight/lookup equivalence and live graph collectives.
use std::{path::Path,time::Instant};
use tch::{Tensor,Kind,Device};
use serde_json::json;
pub fn run(model:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);tch::manual_seed(823);
    let cfg=crate::config::load(&model.join("config.json")).unwrap();
    std::env::remove_var("GLM53_VOCAB_TP");let replicated=crate::weights::ModelWeights::load(model,&cfg,0,dev);
    std::env::set_var("GLM53_VOCAB_TP","1");let sharded=crate::weights::ModelWeights::load(model,&cfg,0,dev);
    assert!(replicated.lm_head.equal(&sharded.drafter_head()),"reconstructed drafter head differs");
    let v=replicated.embed.size()[0];let mut ids=Tensor::from_slice(&[0,v/2-1,v/2,v-1,154856]).to_device(dev);
    assert!(replicated.embed_tokens(&ids).equal(&sharded.embed_tokens(&ids)));
    tch::Cuda::synchronize(0);crate::tp::graph::begin().unwrap();let embedded=sharded.embed_tokens(&ids);crate::tp::graph::end().unwrap();
    for tokens in [[v-1,v/2,v/2-1,0,154856],[1,2,3,4,5],[154856,v-2,1,v/2+1,v/2-2]] {
        ids.copy_(&Tensor::from_slice(&tokens).to_device(dev));crate::tp::graph::replay().unwrap();
        assert!(embedded.equal(&replicated.embed_tokens(&ids)),"lookup graph failed across vocabulary boundary");
    }
    crate::tp::graph::destroy();let mut cases=Vec::new();
    for rows in [1,2,8] {for custom in [false,true] {
        std::env::set_var("GLM53_DENSE_GEMV",if custom{"1"}else{"0"});
        let x=Tensor::randn([rows,4096],(Kind::Float,dev));let mut input=x.copy();
        let expected=replicated.logits(&x);let actual=sharded.logits(&x);let diff=&actual-&expected;
        let rel=((&diff*&diff).sum(Kind::Float)/(&expected*&expected).sum(Kind::Float)).sqrt().double_value(&[]);
        assert!(rel<0.001);if rows==1&&custom{assert!(actual.equal(&expected),"custom per-row GEMV must preserve vocabulary slicing exactly");}
        let mut rounds=Vec::new();
        for split in [false,true,true,false] {
            let w=if split{&sharded}else{&replicated};let _=w.logits(&input);tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let output=w.logits(&input);crate::tp::graph::end().unwrap();
            for z in [&x,&(-&x),&(&x*0.125)]{input.copy_(z);crate::tp::graph::replay().unwrap();assert!(output.equal(&w.logits(z)));}
            input.copy_(&x);for _ in 0..3{crate::tp::graph::replay().unwrap();}
            tch::Cuda::synchronize(0);let started=Instant::now();
            for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"sharded":split,"graph_us":started.elapsed().as_secs_f64()*1e6/32.}));crate::tp::graph::destroy();
        }
        cases.push(json!({"rows":rows,"custom_gemv":custom,"relative_l2":rel,
            "top1_equal":actual.argmax(-1,false).equal(&expected.argmax(-1,false)),"rounds":rounds}));
    }}
    let bytes=replicated.embed.numel()*4+replicated.lm_head.numel()*2;
    std::fs::write(out.join(format!("vocab-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({"rank":tp.rank,"cases":cases,
        "replicated_bytes":bytes,"sharded_bytes":bytes/2,"embedding_exact":true,"drafter_head_exact":true})).unwrap()).unwrap();
    eprintln!("[vocab-probe] rank{} PASS real lookup/head, two-rank changed-input graphs",tp.rank);
}
