//! Bounded performance candidates: retain numerical evidence separately from timing.
use std::{path::Path,time::Instant};
use tch::{Tensor,Kind,Device};
use serde_json::json;
fn error(a:&Tensor,b:&Tensor)->(f64,f64) {
    (f64::try_from((a-b).abs().max()).unwrap(),f64::try_from((a-b).norm()/b.norm().clamp_min(1e-12)).unwrap())
}
pub fn cache(model:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();
    let dev=Device::Cuda(0);tch::manual_seed(701);let mut records=Vec::new();
    for capacity in [4096i64,20480] {for (kind,width,divisor) in [(Kind::Half,512,1),(Kind::Float,128,4)] {
        let rows=(capacity+divisor-1)/divisor;
        let source=Tensor::randn([rows,width],(kind,dev));let mut dest=Tensor::full_like(&source,f64::NAN);
        let mut len=Tensor::zeros([1],(Kind::Int64,dev));
        crate::mla_latent::active_copy(&dest,&source,&len,divisor as i32);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();crate::mla_latent::active_copy(&dest,&source,&len,divisor as i32);crate::tp::graph::end().unwrap();
        for n in [0,1,3,4,5,127,2047,2048,capacity,7,0] {
            let _=dest.fill_(f64::NAN);let _=len.fill_(n);crate::tp::graph::replay().unwrap();let active=(n+divisor-1)/divisor;
            assert!(dest.narrow(0,0,active).equal(&source.narrow(0,0,active)),"active prefix {n}");
            assert!(dest.narrow(0,active,rows-active).isnan().all().int64_value(&[])!=0,"inactive suffix overwritten {n}");
        }
        crate::tp::graph::destroy();
        for n in [128,capacity-1] {let _=len.fill_(n);let mut rounds=Vec::new();
            for active in [false,true,true,false] {
                crate::tp::graph::begin().unwrap();
                if active{crate::mla_latent::active_copy(&dest,&source,&len,divisor as i32);}else{dest.copy_(&source);}
                crate::tp::graph::end().unwrap();for _ in 0..5{crate::tp::graph::replay().unwrap();}
                tch::Cuda::synchronize(0);let start=Instant::now();for _ in 0..64{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                rounds.push(json!({"active":active,"us":start.elapsed().as_secs_f64()*1e6/64.}));crate::tp::graph::destroy();
            }
            records.push(json!({"capacity":capacity,"len":n,"divisor":divisor,"rounds":rounds,"graph_dynamic_extent_exact":true,"suffix_untouched":true}));
        }
    }}
    std::fs::write(out.join("cache-copy.json"),serde_json::to_string_pretty(&records).unwrap()).unwrap();
    let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let mut weights=crate::weights::ModelWeights::load(model,&cfg,4,dev);crate::weights::shard_dense_layer(&mut weights.layers[3],0,2);
    let w=weights.layers[3].mla.as_ref().unwrap();let capacity=20480;let x=Tensor::randn([2070,4096],(Kind::Float,dev))*0.02;
    std::env::set_var("GLM53_PREFILL_BATCH","1");std::env::set_var("GLM53_MLA_PREFILL_BATCHED","1");std::env::set_var("GLM53_MLA_SPARSE_FUSED","1");
    let mut cases=Vec::new();
    for start in [0i64,1,3,4,2047,2048] {
        std::env::set_var("GLM53_MLA_ACTIVE_COPY","0");let mut base=crate::mla_latent::State::new(w,capacity);
        if start>0{let _=crate::mla_latent::chunk(w,&x.narrow(0,0,start),&mut base);}
        let mut expected=base.snapshot();let reference=crate::mla_latent::chunk(w,&x.narrow(0,start,9),&mut expected);
        std::env::set_var("GLM53_MLA_ACTIVE_COPY","1");let mut state=base.snapshot();
        // Poison both unused suffixes. DSA must mask even NaNs in future pools.
        let _=state.latent.narrow(0,start,capacity-start).fill_(f64::NAN);
        let pool_rows=(start+3)/4;let _=state.index.pools.narrow(0,pool_rows,state.index.pools.size()[0]-pool_rows).fill_(f64::NAN);
        let actual=crate::mla_latent::chunk(w,&x.narrow(0,start,9),&mut state);
        assert!(actual.equal(&reference) && state.max_diff(&expected)==0.,"poisoned batch start={start}");
        // Restore a shorter branch into a longer allocation, then use the
        // incremental path. It must not reveal stale future cache entries.
        state.restore(&base);let mut serial=base.snapshot();
        for j in 0..9{let input=x.narrow(0,start+j,1);let a=crate::mla_latent::step(w,&input,&mut state);let b=crate::mla_latent::step(w,&input,&mut serial);assert!(a.equal(&b) && a.isfinite().all().int64_value(&[])!=0);}
        assert_eq!(state.max_diff(&serial),0.);
        // Capture from a short state, then replay with a different device len
        // after branch restore, crossing the pool boundary.
        state.restore(&base);let mut input=x.narrow(0,start,1).copy();let mut sink=base.snapshot();
        let _=crate::mla_latent::step(w,&input,&mut sink);sink.restore(&base);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();sink.restore(&state);let output=crate::mla_latent::step(w,&input,&mut sink);crate::tp::graph::end().unwrap();
        for j in 0..9{input.copy_(&x.narrow(0,start+j,1));let mut eager=state.snapshot();let y=crate::mla_latent::step(w,&input,&mut eager);crate::tp::graph::replay().unwrap();assert!(output.equal(&y));assert_eq!(sink.max_diff(&eager),0.);state.restore(&eager);}
        crate::tp::graph::destroy();cases.push(json!({"start":start,"poisoned_suffix_exact":true,"short_restore_exact":true,"changed_len_graph_exact":true}));
    }
    std::fs::write(out.join("cache-state.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
    eprintln!("[perf-cache] dynamic prefix copy, poisoned inactive suffix, branch restore and changed-length graph passed");
}
pub fn kernels(model:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();
    let dev=Device::Cuda(0);tch::manual_seed(20260923);let mut data=Vec::new();
    for t in [1,7,64,256,2048] {
        let h=32;let initial=Tensor::randn([h,128,128],(Kind::Float,dev))*0.01;
        let norm=|x:Tensor|{let s=(&x*&x).sum_dim_intlist(&[-1i64][..],true,Kind::Float).sqrt();x/s};
        let q=norm(Tensor::randn([t,h,128],(Kind::Float,dev)))/128f64.sqrt();
        let k=norm(Tensor::randn([t,h,128],(Kind::Float,dev)));
        let v=Tensor::randn([t,h,128],(Kind::Float,dev));let beta=Tensor::rand([t,h],(Kind::Float,dev));
        let decay=Tensor::rand([t,h,128],(Kind::Float,dev))*0.1+0.9;
        let mut baseline=initial.copy();let mut state=initial.copy();
        let expected=Tensor::stack(&(0..t).map(|i|crate::kda::recurrent(&mut baseline,&q.get(i),&k.get(i),&v.get(i),&beta.get(i),&decay.get(i),true)).collect::<Vec<_>>(),0);
        let actual=crate::kda::sequence(&mut state,&q,&k,&v,&beta,&decay);
        let e=error(&actual,&expected);assert!(actual.equal(&expected) && state.equal(&baseline),"KDA sequence {t}: {e:?}");
        let mut rounds=Vec::new();
        for fused in [false,true,true,false] {
            state.copy_(&initial);tch::Cuda::synchronize(0);let now=Instant::now();
            if fused{let _=crate::kda::sequence(&mut state,&q,&k,&v,&beta,&decay);}else{for i in 0..t{let _=crate::kda::recurrent(&mut state,&q.get(i),&k.get(i),&v.get(i),&beta.get(i),&decay.get(i),true);}}
            tch::Cuda::synchronize(0);rounds.push(json!({"fused":fused,"ms":now.elapsed().as_secs_f64()*1000.}));
        }
        data.push(json!({"kernel":"kda-sequence","tokens":t,"exact":true,"error":e,"rounds":rounds}));
        std::env::set_var("GLM53_KDA_SEQUENCE","2");state.copy_(&initial);
        tch::Cuda::synchronize(0);let start=Instant::now();let warp=crate::kda::sequence(&mut state,&q,&k,&v,&beta,&decay);tch::Cuda::synchronize(0);
        let ms=start.elapsed().as_secs_f64()*1000.;let e=error(&warp,&expected);let se=error(&state,&baseline);
        assert!(e.0<2e-6 && se.0<2e-6,"KDA warp {e:?}, {se:?}");
        data.push(json!({"kernel":"kda-warp-sequence","tokens":t,"error":e,"state_error":se,"ms":ms}));
        std::env::set_var("GLM53_KDA_SEQUENCE","0");
    }
    for &(queries,visible) in &[(1,1),(1,21),(1,512),(1,2048),(8,21),(8,512),(8,2048),(64,512),(256,2048),(2048,2048)] {
        let heads=32;let slots=2051;let latent=Tensor::randn([4096,512],(Kind::Half,dev));
        let q=Tensor::randn([heads,queries,512],(Kind::Float,dev))*0.3;
        let ids:Vec<i64>=(0..queries).flat_map(|i|(0..slots).map(move|s|if s<visible{(s*17+i)%4096}else{-1})).collect();
        let ids=Tensor::from_slice(&ids).view([queries,slots]).to_device(dev);
        let reference=|input:&Tensor|Tensor::cat(&(0..queries).map(|i|{
            let selected=ids.get(i);let c=latent.index_select(0,&selected.clamp_min(0)).to_kind(Kind::Float);
            let s=input.narrow(1,i,1).matmul(&c.transpose(0,1))/16.;
            s.masked_fill(&selected.lt(0).view([1,1,-1]),f64::NEG_INFINITY).softmax(-1,Kind::Float).matmul(&c)
        }).collect::<Vec<_>>(),1);
        let expected=reference(&q);let actual=crate::mla_latent::sparse_attention(&q,&latent,&ids);
        let e=error(&actual,&expected);assert!(e.1<2e-5 && actual.isfinite().all().int64_value(&[])!=0,"sparse attention {e:?}");
        let mut input=q.copy();let _=crate::mla_latent::sparse_attention(&input,&latent,&ids);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let graphed=crate::mla_latent::sparse_attention(&input,&latent,&ids);crate::tp::graph::end().unwrap();
        for x in [&q,&(-&q),&q]{input.copy_(x);crate::tp::graph::replay().unwrap();assert!(graphed.equal(&crate::mla_latent::sparse_attention(x,&latent,&ids)));}
        crate::tp::graph::destroy();let mut rounds=Vec::new();
        for fused in [false,true,true,false]{tch::Cuda::synchronize(0);let now=Instant::now();for _ in 0..3{let _=if fused{crate::mla_latent::sparse_attention(&q,&latent,&ids)}else{reference(&q)};}tch::Cuda::synchronize(0);rounds.push(json!({"fused":fused,"ms":now.elapsed().as_secs_f64()*1000./3.}));}
        data.push(json!({"kernel":"sparse-latent","queries":queries,"visible":visible,"error":e,"changed_graph_exact":true,"rounds":rounds}));
        for tf32 in [false,true] {
            crate::tp::set_tf32(tf32);let gather=||crate::mla_latent::prefill_gather_attention(&q,&latent,&ids);
            let actual=gather();let e=error(&actual,&expected);assert!(e.1<if tf32{0.01}else{2e-5},"gather MLA {e:?}");
            tch::Cuda::synchronize(0);let start=Instant::now();for _ in 0..3{let _=gather();}tch::Cuda::synchronize(0);
            data.push(json!({"kernel":"gather-latent","queries":queries,"visible":visible,"tf32":tf32,"error":e,"ms":start.elapsed().as_secs_f64()*1000./3.}));
        }
        crate::tp::set_tf32(false);
        {std::env::set_var("GLM53_MLA_PREFILL_F16","1");
         let actual=crate::mla_latent::shared_attention(&q,&latent,&ids,8);let e=error(&actual,&expected);
         tch::Cuda::synchronize(0);let start=Instant::now();for _ in 0..3{let _=crate::mla_latent::shared_attention(&q,&latent,&ids,8);}tch::Cuda::synchronize(0);
         let ms=start.elapsed().as_secs_f64()*1000./3.;std::env::set_var("GLM53_MLA_PREFILL_F16","0");
         eprintln!("[perf-kernels] latent f16 queries {queries} visible {visible}: rel {:.2e} ms {ms:.3}",e.1);
         assert!(e.1<5e-3,"FP16 latent attention {e:?}");
         data.push(json!({"kernel":"shared-latent","mode":10,"queries":queries,"visible":visible,"error":e,"ms":ms}));}
        for mode in [3,4,5,6,7,8] {
            let actual=crate::mla_latent::shared_attention(&q,&latent,&ids,mode);let e=error(&actual,&expected);
            assert!(e.1<2e-5,"shared latent mode={mode} {e:?}");
            if mode==6{assert!(actual.equal(&crate::mla_latent::shared_attention(&q,&latent,&ids,4)),"vector loads changed shared attention arithmetic");}
            input.copy_(&q);tch::Cuda::synchronize(0);crate::tp::graph::begin().unwrap();
            let graphed=crate::mla_latent::shared_attention(&input,&latent,&ids,mode);crate::tp::graph::end().unwrap();
            for x in [&q,&(-&q),&q]{input.copy_(x);crate::tp::graph::replay().unwrap();assert!(graphed.equal(&crate::mla_latent::shared_attention(x,&latent,&ids,mode)));}
            crate::tp::graph::destroy();tch::Cuda::synchronize(0);let start=Instant::now();
            for _ in 0..3{let _=crate::mla_latent::shared_attention(&q,&latent,&ids,mode);}tch::Cuda::synchronize(0);
            data.push(json!({"kernel":"shared-latent","mode":mode,"queries":queries,"visible":visible,"error":e,"changed_graph_exact":true,"ms":start.elapsed().as_secs_f64()*1000./3.}));
        }
    }
    std::fs::write(out.join("kernels.json"),serde_json::to_string_pretty(&data).unwrap()).unwrap();
    let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let mut weights=crate::weights::ModelWeights::load(model,&cfg,4,dev);crate::weights::shard_dense_layer(&mut weights.layers[3],0,2);
    let w=weights.layers[3].mla.as_ref().unwrap();let x=Tensor::randn([2177,4096],(Kind::Float,dev))*0.02;
    let mut old=crate::mla_latent::State::new(w,4096);let mut new=old.snapshot();let mut pos=0;let mut cases=Vec::new();
    for n in [1,3,60,1985,128] {
        std::env::set_var("GLM53_MLA_PREFILL_BATCHED","0");std::env::set_var("GLM53_MLA_SPARSE_FUSED","0");
        let before=old.snapshot();
        let input=x.narrow(0,pos,n);let expected=crate::mla_latent::chunk(w,&input,&mut old);
        std::env::set_var("GLM53_MLA_PREFILL_BATCHED","1");let actual=crate::mla_latent::chunk(w,&input,&mut new);
        let e=error(&actual,&expected);let state_error=new.max_diff(&old);
        cases.push(json!({"start":pos,"rows":n,"output_error":e,"state_max_abs":state_error}));
        std::fs::write(out.join("mla-batch.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
        assert!(e.1<0.003 && state_error<2e-6,"MLA batch {pos}+{n}: {e:?} state {state_error}");
        for mode in ["1","2","3","4","5","6","7","8"] {for tf32 in [false,true] {
            crate::tp::set_tf32(tf32);std::env::set_var("GLM53_MLA_PREFILL_DENSE",mode);let mut state=before.snapshot();
            let actual=crate::mla_latent::chunk(w,&input,&mut state);let e=error(&actual,&expected);let se=state.max_diff(&old);
            assert!(e.1<0.015 && se<0.02,"dense MLA tf32={tf32}: {e:?} state {se}");
            cases.push(json!({"start":pos,"rows":n,"attention_mode":mode,"tf32":tf32,"output_error":e,"state_max_abs":se}));
        }}
        crate::tp::set_tf32(false);std::env::set_var("GLM53_MLA_PREFILL_DENSE","0");pos+=n;
    }
    std::fs::write(out.join("mla-batch.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
    eprintln!("[perf-kernels] KDA exact, sparse attention, changed graph and real MLA pool boundaries passed");
}

pub fn prefill(model:&Path,suite:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;let misses=fast.misses;
    let mut engine=crate::forward::Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    let suite:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(suite).unwrap()).unwrap();
    let ids:Vec<i64>=suite["cases"][0]["prompt_ids"].as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).collect();
    let mut results=Vec::new();
    let flags=["GLM53_KDA_SEQUENCE","GLM53_MLA_SPARSE_FUSED","GLM53_MLA_PREFILL_BATCHED","GLM53_MLA_PREFILL_DENSE"];
    let saved_tf32=std::env::var("GLM53_TF32").unwrap_or_else(|_|"0".into());
    let lengths=std::env::var("GLM53_PREFILL_LENGTHS").ok().map(|s|s.split(',').map(|n|n.parse::<usize>().unwrap()).collect::<Vec<_>>()).unwrap_or_else(||vec![64,ids.len()]);
    let variants=std::env::var("GLM53_PREFILL_VARIANTS").unwrap_or_else(|_|"baseline,kda,sparse,batched,all,all,baseline".into());
    for rows in lengths {
        let input=Tensor::from_slice(&ids[..rows]).to_device(dev);
        let mut reference=None;
        for full_variant in variants.split(',') {
            // "base+ext+ext": P-series prefill extensions on top of a base variant.
            let mut parts=full_variant.split('+');let variant=parts.next().unwrap();let extensions:Vec<&str>=parts.collect();
            for flag in flags{std::env::set_var(flag,"0");}
            for flag in ["GLM53_MHC_PRE_LARGE","GLM53_KDA_PREFILL_FUSED","GLM53_PREFILL_EXPERT_STREAMS","GLM53_PREFILL_FAT_MOE","GLM53_HALF_INPUT_CACHE","GLM53_MHC_POST_PRE_FUSED","GLM53_DSA_PREFILL_SCORE_FUSED","GLM53_PREFILL_MOE_SUM1","GLM53_MLA_PREFILL_F16","GLM53_PREFILL_SP","GLM53_PREFILL_FIRST_CONTIG","GLM53_FP8_PREFILL_HALF","GLM53_DSA_PREFILL_SCORE_TILED"]{std::env::set_var(flag,"0");}
            std::env::set_var("GLM53_PREFILL_GROUPED","0");
            std::env::set_var("GLM53_TF32",&saved_tf32);
            std::env::set_var("GLM53_PREFILL_RECON_MIN_ROWS","1");
            if ["warp","direct","recon","hybrid32","hybrid64","dense","dense-tf32","gather","gather-tf32","shared3","shared4","shared5","shared6","shared7","shared8"].contains(&variant){
                std::env::set_var(flags[0],"2");std::env::set_var(flags[1],"1");std::env::set_var(flags[2],"1");
                if variant!="warp"{std::env::set_var("GLM53_PREFILL_GROUPED",if variant=="direct"{"direct"}else{"recon"});}
                if variant.starts_with("hybrid"){std::env::set_var("GLM53_PREFILL_RECON_MIN_ROWS",variant.trim_start_matches("hybrid"));}
                if variant.starts_with("dense"){std::env::set_var(flags[3],"1");}
                if variant.starts_with("gather"){std::env::set_var(flags[3],"2");}
                if variant.starts_with("shared"){std::env::set_var(flags[3],variant.trim_start_matches("shared"));}
                if variant.ends_with("-tf32"){std::env::set_var("GLM53_TF32","1");}
            }
            if variant=="kda"||variant=="all"{std::env::set_var(flags[0],"1");}
            if variant=="sparse"||variant=="all"{std::env::set_var(flags[1],"1");}
            if variant=="batched"||variant=="all"{std::env::set_var(flags[2],"1");}
            if ["r5","dsa-limit","grouped-reduce","r6","r6-full","r6-swiglu","r6-reuse"].contains(&variant) {
                std::env::set_var(flags[0],"2");std::env::set_var(flags[1],"1");std::env::set_var(flags[2],"1");std::env::set_var(flags[3],"6");
                std::env::set_var("GLM53_PREFILL_GROUPED","recon");std::env::set_var("GLM53_PREFILL_RECON_MIN_ROWS","64");
                std::env::set_var("GLM53_FP8_LARGE",if ["r6-full","r6-swiglu","r6-reuse"].contains(&variant){"5"}else{"0"});
                for flag in ["GLM53_GROUPED_INDEX_PACK","GLM53_FP8_EPILOGUE"] {std::env::set_var(flag,if variant=="r6-reuse"{"1"}else{"0"});}
                std::env::set_var("GLM53_GROUPED_SWIGLU",if ["r6-swiglu","r6-reuse"].contains(&variant){"1"}else{"0"});
                std::env::set_var("GLM53_DSA_PREFILL_LIMIT",if variant=="dsa-limit"||variant=="r6"||["r6-full","r6-swiglu","r6-reuse"].contains(&variant){"1"}else{"0"});
                std::env::set_var("GLM53_GROUPED_REDUCE",if variant=="grouped-reduce"||variant=="r6"||["r6-full","r6-swiglu","r6-reuse"].contains(&variant){"1"}else{"0"});
            }
            for e in &extensions {
                match *e {
                    "mla8"=>std::env::set_var(flags[3],"8"),
                    "mhc"=>std::env::set_var("GLM53_MHC_PRE_LARGE","1"),
                    "kda"=>std::env::set_var("GLM53_KDA_PREFILL_FUSED","1"),
                    "fat"=>std::env::set_var("GLM53_PREFILL_FAT_MOE","1"),
                    "hc"=>std::env::set_var("GLM53_HALF_INPUT_CACHE","1"),
                    "k3"=>std::env::set_var(flags[0],"3"),
                    "k4"=>std::env::set_var(flags[0],"4"),
                    "pp"=>std::env::set_var("GLM53_MHC_POST_PRE_FUSED","1"),
                    "ps"=>std::env::set_var("GLM53_DSA_PREFILL_SCORE_FUSED","1"),
                    "ms1"=>std::env::set_var("GLM53_PREFILL_MOE_SUM1","1"),
                    "f16"=>std::env::set_var("GLM53_MLA_PREFILL_F16","1"),
                    "sp"=>std::env::set_var("GLM53_PREFILL_SP","1"),
                    "fc"=>std::env::set_var("GLM53_PREFILL_FIRST_CONTIG","1"),
                    "ph"=>std::env::set_var("GLM53_FP8_PREFILL_HALF","1"),
                    "tl"=>std::env::set_var("GLM53_DSA_PREFILL_SCORE_TILED","1"),
                    s if s.starts_with('s')=>std::env::set_var("GLM53_PREFILL_EXPERT_STREAMS",&s[1..]),
                    other=>panic!("unknown prefill extension {other}"),
                }
            }
            crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
            extern "C"{fn rs_memory_stats(reset:i32,out:*mut i64)->i32;}
            let host_before=crate::host_memory::snapshot();
            let mut memory_before=[0i64;8];assert_eq!(unsafe{rs_memory_stats(1,memory_before.as_mut_ptr())},0);
            tch::Cuda::synchronize(0);let now=Instant::now();let (logits,state,_)=engine.prefill_record_last(&input,None,false);tch::Cuda::synchronize(0);let ms=now.elapsed().as_secs_f64()*1000.;
            let last=logits.get(logits.size()[0]-1);let top=last.argmax(-1,false).int64_value(&[]);
            let (max,rel)=if let Some(ref r)=reference{error(&last,r)}else{reference=Some(last.copy());(0.,0.)};
            let mut memory_after=[0i64;8];assert_eq!(unsafe{rs_memory_stats(0,memory_after.as_mut_ptr())},0);
            results.push(json!({"rows":rows,"variant":full_variant,"ms":ms,"tok_s":rows as f64*1000./ms,"last_top1":top,"last_max_abs":max,"last_relative":rel,
                "memory_before":memory_before,"memory_after":memory_after,"host_before":host_before,"host_after":crate::host_memory::snapshot(),
                "memory_fields":["allocated_current","allocated_peak","reserved_current","reserved_peak","alloc_retries","ooms","driver_free","driver_total"]}));
            std::fs::write(out.join(format!("prefill-rank{}.json",tp.rank)),serde_json::to_string_pretty(&results).unwrap()).unwrap();
            eprintln!("[prefill-r5] rank{} {rows} {full_variant} {ms:.2} ms top1={top} max_abs={max:.3e} rel={rel:.3e}",tp.rank);drop(state);
        }
    }
    assert_eq!(misses,engine.fast.as_ref().unwrap().misses);
}

pub fn grouped(model:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);tch::manual_seed(99);
    let mut pool=crate::moefast::MoeFast::new_with_tp(model,4,288,304,dev,crate::tp::Tp{rank:0,world:2});
    pool.ensure_many(3,&(0..288).collect::<Vec<_>>(),dev);pool.assume_hot=true;let mut cases=Vec::new();
    for rows in [8,32,128,512,2048] {
        let x=(Tensor::randn([rows,4096],(Kind::Float,dev))*0.03).to_kind(Kind::Half);
        let ids:Vec<i64>=(0..rows).flat_map(|t|(0..8).map(move|e|(t*19+e*31)%288)).collect();
        let ids=Tensor::from_slice(&ids).view([rows,8]).to_device(dev);
        let weights=Tensor::rand([rows,8],(Kind::Float,dev)).softmax(-1,Kind::Float)*2.5;
        let coop=|pool:&mut crate::moefast::MoeFast|Tensor::cat(&(0..rows).step_by(32).map(|i|pool.expert_cooperative(3,&x.narrow(0,i,(rows-i).min(32)),&ids.narrow(0,i,(rows-i).min(32)),&weights.narrow(0,i,(rows-i).min(32)))).collect::<Vec<_>>(),0);
        let expected=coop(&mut pool);let mut rounds=Vec::new();
        for mode in ["coop","direct","recon","fat","fat","recon","direct","coop"] {
            std::env::set_var("GLM53_PREFILL_FAT_MOE",if mode=="fat"{"1"}else{"0"});
            tch::Cuda::synchronize(0);let start=Instant::now();
            let y=if mode=="coop"{coop(&mut pool)}else{pool.expert_grouped(3,&x,&ids,&weights,mode=="recon")};
            tch::Cuda::synchronize(0);let ms=start.elapsed().as_secs_f64()*1000.;let e=error(&y,&expected);
            rounds.push(json!({"mode":mode,"ms":ms,"error":e}));
            assert!(e.1<0.01,"grouped rows {rows}, mode {mode}: {e:?}");
        }
        cases.push(json!({"rows":rows,"experts":288,"rounds":rounds}));
        std::fs::write(out.join("grouped.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
    }
}

pub fn dense(model:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);
    let cfg=crate::config::load(&model.join("config.json")).unwrap();let mut weights=crate::weights::ModelWeights::load(model,&cfg,4,dev);
    for layer in &mut weights.layers{crate::weights::shard_dense_layer(layer,0,2);}
    let mut matrices=std::collections::BTreeMap::new();
    for layer in &weights.layers {
        let mut add=|w:&Tensor|{if w.kind()==Kind::Half && w.size().len()==2 && w.size()[0]>=512 && w.size()[1]>=512{matrices.entry((w.size()[0],w.size()[1])).or_insert_with(||w.shallow_clone());}};
        if let Some(w)=&layer.kda{for x in [&w.wq,&w.wk,&w.wv,&w.wo,&w.fa,&w.fb,&w.ga,&w.gb]{add(x);}}
        if let Some(w)=&layer.mla{for x in [&w.q_a,&w.q_b,&w.kv_a,&w.wo]{add(x);}}
        if let Some(w)=&layer.dense{for x in [&w.wg,&w.wu,&w.wd]{add(x);}}
        if let Some(w)=&layer.moe{for x in [&w.sh_wg,&w.sh_wu,&w.sh_wd]{add(x);}}
    }
    let head=weights.lm_head.narrow(0,0,weights.lm_head.size()[0]/2).contiguous();matrices.insert((head.size()[0],head.size()[1]),head);
    let mut cases=Vec::new();let mut selected=std::collections::BTreeMap::<String,i32>::new();
    for ((n,k),w) in matrices {for m in [1,2,8] {for partial in [false,true] {
        tch::manual_seed(99);let x=Tensor::randn([m,k],(Kind::Float,dev))*0.03;
        let baseline=||if partial{crate::weights::mm16_partial(&x,&w)}else{crate::weights::mm16(&x,&w)};
        let expected=baseline();
        extern "C" {fn rs_lt_count(m:i32,n:i32,k:i32,fp32:i32)->i32;}
        let count=unsafe{rs_lt_count(m as i32,n as i32,k as i32,i32::from(partial))};assert!(count>0);
        let mut rounds=Vec::new();let mut best=(f64::INFINITY,-1);let mut base_us=0.;
        for algo in std::iter::once(-1).chain(0..count.min(16)).chain(std::iter::once(-1)) {
            let call=||if algo<0{baseline()}else{crate::dense_lt::run(&x,&w,partial,algo)};
            let actual=call();let e=error(&actual,&expected);assert!(e.1<0.002,"Lt {m}/{n}/{k} {algo}: {e:?}");
            for _ in 0..2{let _=call();}tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let _y=call();crate::tp::graph::end().unwrap();
            for _ in 0..3{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let start=Instant::now();
            for _ in 0..16{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let us=start.elapsed().as_secs_f64()*1e6/16.;crate::tp::graph::destroy();
            rounds.push(json!({"algorithm":algo,"us":us,"error":e}));
            if algo<0{base_us+=us/2.;}else if us<best.0{best=(us,algo);}
        }
        if best.0<base_us*0.95{selected.insert(format!("{m},{n},{k},{}",i32::from(partial)),best.1);}
        cases.push(json!({"m":m,"n":n,"k":k,"partial":partial,"baseline_us":base_us,"best_us":best.0,"best_algorithm":best.1,"rounds":rounds}));
        std::fs::write(out.join("dense.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
        std::fs::write(out.join("lt-table.json"),serde_json::to_string_pretty(&selected).unwrap()).unwrap();
    }}}
}


pub fn sweep(model:&Path,draft:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);let dev=Device::Cuda(0);
    let cfg=crate::config::load(&model.join("config.json")).unwrap();let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=crate::forward::Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
    sweep_resident(&mut eng,&drafter,out);
}

/// Benchmark-only entry for staged tests sharing the same resident checkpoint.
/// No timing or request state is shared implicitly between phases.
pub(crate) fn sweep_resident(eng:&mut crate::forward::Engine,drafter:&crate::dflash::Drafter,out:&Path) {
    let tp=crate::tp::world();assert_eq!(tp.world,2);
    let phases=std::env::var("GLM53_R5_SWEEP").unwrap_or_else(|_|"sparse,lt,geometry0,geometry2,tf32,fp8,depth3,depth5".into());
    for phase in phases.split(',') {
        eprintln!("[r5-sweep] rank{} phase={phase}",tp.rank);
        if phase.starts_with("fp8-small-") {
            assert_eq!(std::env::var("GLM53_DENSE_FP8").as_deref(),Ok("1"),"FP8 small sweep must retain target FP8 in both arms");
            assert!(std::env::var("GLM53_FP8_WMMA").ok().and_then(|v|v.parse::<i32>().ok()).unwrap_or(0)>0,"mode0 does not exercise small WMMA");
        }
        if phase=="dsa-topk" {
            assert!(crate::dsa_index::enabled(),"DSA TopK sweep requires fused index bookkeeping");
            assert_ne!(std::env::var("GLM53_DSA_ALL_VISIBLE").as_deref(),Ok("1"),"DSA TopK sweep must remain Ranked");
        }
        let flag=match phase{"dsa-topk"=>"GLM53_DSA_TOPK_BATCH","fp8-small-transpose"=>"GLM53_FP8_SMALL_TRANSPOSE","fp8-small-pad"=>"GLM53_FP8_SMALL_PAD","feature-prefix"=>"GLM53_ACCEPTED_FEATURE_VIEW","dsa-index"=>"GLM53_DSA_INDEX_FUSED","conv-deferred"=>"GLM53_KDA_CONV_DEFERRED","half-input"=>"GLM53_MOE_INPUT_HALF_REUSE","kda-correction"=>"GLM53_KDA_CORRECTION_REPLAY","mhc-four"=>"GLM53_MHC_POST_FOUR_STREAMS","mhc-packed"=>"GLM53_MHC_POST_PACKED","depth4-abba"|"depth6-abba"=>"GLM53_SPEC_MAX_DRAFT","target-top1"=>"GLM53_TARGET_TOP1_TP","mla-cache"=>"GLM53_MLA_WEIGHT_CACHE","coop-prefetch"=>"GLM53_COOP_PREFETCH","memory-combined"=>"GLM53_KDA_CONV_CHAIN","dsa-direct"=>"GLM53_DSA_VISIBLE_DIRECT","coop-shared-input"=>"GLM53_COOP_SHARED_INPUT","sham"=>"GLM53_DRAFT_NORM_CACHE","dsa-visible"=>"GLM53_DSA_ALL_VISIBLE","memory-core"|"memory-bundle"=>"GLM53_KDA_CONV_CHAIN","kda-chain"=>"GLM53_KDA_CHAIN_RECURRENT","kda-norm"=>"GLM53_KDA_CHAIN_NORM","norm-cache"=>"GLM53_DRAFT_NORM_CACHE","kda-conv"=>"GLM53_KDA_CONV_CHAIN","coop-transpose"=>"GLM53_COOP_CANDIDATE","moe-no-copy"=>"GLM53_MOE_NO_COPY","fp8-shared"=>"GLM53_FP8_SHARED","fp8-mla"=>"GLM53_FP8_MLA","fp8-head"=>"GLM53_FP8_HEAD","fp8-draft-mlp"=>"GLM53_DRAFT_FP8_MLP","fp8-draft-attn"=>"GLM53_DRAFT_FP8_ATTN","fp8-draft-conv"=>"GLM53_DRAFT_FP8_CONV","fp8-draft-fc"=>"GLM53_DRAFT_FP8_FC","fp8-draft-head"=>"GLM53_DRAFT_FP8_HEAD","dsa-fused"|"dsa-tiled"|"dsa-tensor"=>"GLM53_DSA_SCORE_FUSED","gqa"|"gqa-fused"|"gqa-auto"|"gqa-shared"=>"GLM53_DRAFT_GQA","dsa-limit"=>"GLM53_DSA_PREFILL_LIMIT","grouped-reduce"=>"GLM53_GROUPED_REDUCE","active-copy"=>"GLM53_MLA_ACTIVE_COPY","prefill128"=>"GLM53_PREFILL_MIN_ROWS","fp8"=>"GLM53_DENSE_FP8","sparse"=>"GLM53_MLA_SPARSE_FUSED","lt"=>"GLM53_DENSE_LT","geometry0"|"geometry2"=>"GLM53_COOP_GEOMETRY","tf32"=>"GLM53_TF32","depth3"|"depth4"|"depth5"|"depth6"|"depth7"|"adaptive"=>"",_=>panic!("unknown sweep phase")};
        std::env::set_var("GLM53_SPEC_ABBA_OFF",if phase.ends_with("-abba"){"7"}else if phase.starts_with("geometry"){"1"}else{"0"});
        std::env::set_var("GLM53_SPEC_ABBA_ON",match phase{"depth4-abba"=>"4","depth6-abba"=>"6","sham"=>"0","dsa-tensor"=>"5","dsa-tiled"=>"2","gqa-shared"=>"5","gqa-fused"=>"2","gqa-auto"=>"3","prefill128"=>"128","geometry0"=>"0","geometry2"=>"2",_=>"1"});
        if !flag.is_empty(){std::env::set_var("GLM53_SPEC_ABBA_FLAG",flag);std::env::set_var("GLM53_SPEC_ROUNDS","4");}
        else{std::env::remove_var("GLM53_SPEC_ABBA_FLAG");std::env::set_var("GLM53_SPEC_ROUNDS","2");std::env::set_var("GLM53_SPEC_FULL_WARMUPS","1");std::env::set_var("GLM53_SPEC_MAX_DRAFT",match phase{"depth3"=>"3","depth4"=>"4","depth6"=>"6","depth7"=>"7",_=>"5"});}
        let saved_mode=std::env::var("GLM53_SPEC_MODES").unwrap_or_else(|_|"batch-graph-chain".into());
        if phase=="adaptive"{std::env::set_var("GLM53_SPEC_MODES","batch-adaptive-chain");std::env::set_var("GLM53_SPEC_MAX_DRAFT","7");}
        let dependencies:Vec<_>=if phase=="conv-deferred"{vec!["GLM53_KDA_FUSED","GLM53_KDA_FORK_FUSED","GLM53_KDA_CONV_CHAIN","GLM53_KDA_CHAIN_NORM","GLM53_KDA_CHAIN_RECURRENT","GLM53_KDA_CORRECTION_REPLAY"]}else if phase=="kda-correction"{vec!["GLM53_KDA_FUSED","GLM53_KDA_FORK_FUSED","GLM53_KDA_CONV_CHAIN","GLM53_KDA_CHAIN_NORM","GLM53_KDA_CHAIN_RECURRENT"]}else if phase=="dsa-direct"{vec!["GLM53_DSA_ALL_VISIBLE"]}else if phase=="coop-prefetch" || phase=="coop-shared-input"{vec!["GLM53_COOP_CANDIDATE"]}else if phase=="kda-norm"{vec!["GLM53_KDA_CONV_CHAIN"]}else if phase=="kda-chain"{vec!["GLM53_KDA_CONV_CHAIN","GLM53_KDA_CHAIN_NORM"]}else{vec![]};
        let saved_dependencies:Vec<_>=dependencies.iter().map(|f|(*f,std::env::var(f).ok())).collect();
        for flag in dependencies{std::env::set_var(flag,"1");}
        let saved_extra=std::env::var("GLM53_SPEC_ABBA_EXTRA").ok();
        if phase=="memory-bundle" || phase=="memory-core" || phase=="memory-combined" {
            std::env::set_var("GLM53_SPEC_ABBA_EXTRA",if phase=="memory-combined" {
                "GLM53_KDA_CHAIN_NORM,GLM53_KDA_CHAIN_RECURRENT,GLM53_MOE_NO_COPY,GLM53_DRAFT_NORM_CACHE,GLM53_TARGET_TOP1_TP,GLM53_MLA_WEIGHT_CACHE"
            }else if phase=="memory-core" {
                "GLM53_KDA_CHAIN_NORM,GLM53_KDA_CHAIN_RECURRENT,GLM53_MOE_NO_COPY,GLM53_DRAFT_NORM_CACHE"
            }else{"GLM53_KDA_CHAIN_NORM,GLM53_KDA_CHAIN_RECURRENT,GLM53_MOE_NO_COPY,GLM53_DRAFT_NORM_CACHE,GLM53_COOP_CANDIDATE"});
        }
        crate::spec_probe::check(eng,drafter,&out.join(phase));
        match saved_extra{Some(v)=>std::env::set_var("GLM53_SPEC_ABBA_EXTRA",v),None=>std::env::remove_var("GLM53_SPEC_ABBA_EXTRA")};
        for (f,saved) in saved_dependencies{match saved{Some(v)=>std::env::set_var(f,v),None=>std::env::remove_var(f)}}
        std::env::set_var("GLM53_SPEC_MODES",saved_mode);
        std::env::set_var("GLM53_SPEC_MAX_DRAFT","7");
    }
    std::env::remove_var("GLM53_SPEC_ABBA_FLAG");std::env::remove_var("GLM53_SPEC_ABBA_OFF");std::env::remove_var("GLM53_SPEC_ABBA_ON");
}

pub fn fp8(model:&Path,out:&Path){
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);
    let cfg=crate::config::load(&model.join("config.json")).unwrap();let mut weights=crate::weights::ModelWeights::load(model,&cfg,4,dev);
    for l in &mut weights.layers{crate::weights::shard_dense_layer(l,0,2);}
    let kda=weights.layers[0].kda.as_ref().unwrap();let dense=weights.layers[0].dense.as_ref().unwrap();
    if std::env::var("GLM53_FP8_LARGE_PROBE").as_deref()==Ok("2") {
        let mut cases=Vec::new();
        for w in [&kda.wq,&kda.wo,&dense.wg,&dense.wd] {
            let (q,s)=crate::dense_fp8::quantize(w);let packed:Vec<_>=(0..16).map(|_|q.copy()).collect();
            for m in [64,128] {
                let x=Tensor::randn([m,w.size()[1]],(Kind::Float,dev))*0.03;
                std::env::set_var("GLM53_FP8_LARGE","0");let expected=crate::dense_fp8::run(&x,&q,&s,true);
                std::env::set_var("GLM53_FP8_LARGE","5");let actual=crate::dense_fp8::run(&x,&q,&s,true);
                let e=error(&actual,&expected);assert!(e.1<2e-5);let mut rounds=Vec::new();
                for mode in ["0","5","5","0"] {
                    std::env::set_var("GLM53_FP8_LARGE",mode);for p in &packed{let _=crate::dense_fp8::run(&x,p,&s,true);}tch::Cuda::synchronize(0);
                    crate::tp::graph::begin().unwrap();for p in &packed{let _=crate::dense_fp8::run(&x,p,&s,true);}crate::tp::graph::end().unwrap();
                    for _ in 0..3{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let now=Instant::now();
                    for _ in 0..8{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                    rounds.push(json!({"mode":mode,"us_per_projection":now.elapsed().as_secs_f64()*1e6/(8.*16.)}));crate::tp::graph::destroy();
                }
                cases.push(json!({"m":m,"shape":w.size(),"weight_working_set_bytes":q.numel()*16,"error":e,"rounds":rounds}));
                std::fs::write(out.join("fp8-large-rotate.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
            }
        }return;
    }
    if std::env::var("GLM53_FP8_LARGE_PROBE").as_deref()==Ok("1") {
        let mut cases=Vec::new();
        for w in [&kda.wq,&kda.wo,&dense.wg,&dense.wd] {
            let (q,s)=crate::dense_fp8::quantize(w);
            for m in [17,64,128,512,2048] {
                let x=Tensor::randn([m,w.size()[1]],(Kind::Float,dev))*0.03;
                std::env::set_var("GLM53_FP8_LARGE","0");let expected=crate::dense_fp8::run(&x,&q,&s,true);let mut rounds=Vec::new();
                for mode in ["0","1","3","4","4","3","1","0"] {
                    std::env::set_var("GLM53_FP8_LARGE",mode);
                    let y=crate::dense_fp8::run(&x,&q,&s,true);let e=error(&y,&expected);assert!(e.1<2e-5,"FP8 large m={m}, {e:?}");
                    let mut input=x.copy();crate::tp::graph::begin().unwrap();let y=crate::dense_fp8::run(&input,&q,&s,true);crate::tp::graph::end().unwrap();
                    for z in [&x,&(-&x),&x]{input.copy_(z);crate::tp::graph::replay().unwrap();assert!(y.equal(&crate::dense_fp8::run(z,&q,&s,true)));}
                    for _ in 0..3{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let now=Instant::now();
                    for _ in 0..8{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                    rounds.push(json!({"mode":mode,"us":now.elapsed().as_secs_f64()*1e6/8.,"error":e}));crate::tp::graph::destroy();
                }
                cases.push(json!({"m":m,"shape":w.size(),"rounds":rounds,"graph_changed_input_exact":true}));
                std::fs::write(out.join("fp8-large.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
            }
        }return;
    }
    let mut cases=Vec::new();
    for w in [&kda.wq,&kda.wo,&dense.wg,&dense.wd] {
        let (q,s)=crate::dense_fp8::quantize(w);
        for m in [1,2,8,16,17,64,256] {
            tch::manual_seed(99);let x=Tensor::randn([m,w.size()[1]],(Kind::Float,dev))*0.03;
            let gold=x.to_kind(Kind::Half).to_kind(Kind::Float).matmul(&q.to_kind(Kind::Float).transpose(0,1))*s.unsqueeze(0);
            let y=crate::dense_fp8::run(&x,&q,&s,true);let e=error(&y,&gold);assert!(e.1<2e-5,"FP8 arithmetic {e:?}");
            let base=crate::weights::mm16_partial(&x,w);let quant_error=error(&y,&base);assert!(quant_error.1<0.06,"FP8 quantization {quant_error:?}");
            let mut input=x.copy();let mut rounds=Vec::new();
            for fp8 in [false,true,true,false] {
                let call=|x:&Tensor|if fp8{crate::dense_fp8::run(x,&q,&s,true)}else{crate::weights::mm16_partial(x,w)};
                for _ in 0..3{let _=call(&input);}tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();let actual=call(&input);crate::tp::graph::end().unwrap();
                for changed in [&x,&(-&x),&x]{input.copy_(changed);crate::tp::graph::replay().unwrap();assert!(actual.equal(&call(changed)));}
                tch::Cuda::synchronize(0);let start=Instant::now();for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let us=start.elapsed().as_secs_f64()*1e6/32.;crate::tp::graph::destroy();rounds.push(json!({"fp8":fp8,"us":us}));
            }
            cases.push(json!({"m":m,"shape":w.size(),"arithmetic_error":e,"quantization_error":quant_error,"graph_changed_input_exact":true,"rounds":rounds}));
            std::fs::write(out.join("fp8.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
            if m==1 || m==8 {
                // Rotate independent allocations: a single repeatedly replayed
                // matrix can fit FP8 in L2 while FP16 spills, overstating the
                // gain for a full model with different weights at every layer.
                let originals:Vec<_>=(0..16).map(|_|w.copy()).collect();
                let packed:Vec<_>=(0..16).map(|_|q.copy()).collect();
                let mut streaming=Vec::new();
                for fp8 in [false,true,true,false] {
                    let call=|i:usize|if fp8{crate::dense_fp8::run(&x,&packed[i],&s,true)}else{crate::weights::mm16_partial(&x,&originals[i])};
                    for i in 0..16{let _=call(i);}tch::Cuda::synchronize(0);
                    crate::tp::graph::begin().unwrap();for i in 0..16{let _=call(i);}crate::tp::graph::end().unwrap();
                    for _ in 0..3{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                    let start=Instant::now();for _ in 0..8{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                    streaming.push(json!({"fp8":fp8,"us_per_projection":start.elapsed().as_secs_f64()*1e6/128.}));crate::tp::graph::destroy();
                }
                cases.push(json!({"m":m,"shape":w.size(),"streaming_matrices":16,"rounds":streaming}));
                std::fs::write(out.join("fp8.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
            }
        }
    }
}

/// Memory-flow candidates: masks, ties, BF16 layout and deterministic reductions.
pub fn dataflow(out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();
    let dev=Device::Cuda(0);tch::manual_seed(90222);let mut records=Vec::new();
    for rows in [1,7,128,512,2048] {
        let x=Tensor::randn([rows*8,4096],(Kind::Half,dev));
        let w=Tensor::randn([rows,8],(Kind::Float,dev));
        let expected=crate::moefast::grouped_reduce(&x,&w,false);
        let actual=crate::moefast::grouped_reduce(&x,&w,true);
        assert!(actual.equal(&expected),"grouped reduction {rows}: {:?}",error(&actual,&expected));
        let mut rounds=Vec::new();
        for fused in [false,true,true,false] {
            let _=crate::moefast::grouped_reduce(&x,&w,fused);tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let y=crate::moefast::grouped_reduce(&x,&w,fused);crate::tp::graph::end().unwrap();
            for _ in 0..5{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let now=Instant::now();
            for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"fused":fused,"us":now.elapsed().as_secs_f64()*1e6/32.}));assert!(y.equal(&expected));crate::tp::graph::destroy();
        }
        records.push(json!({"op":"grouped-reduce","rows":rows,"exact":true,"rounds":rounds}));
    }
    for tf32 in [false,true] {crate::tp::set_tf32(tf32);
        for history in [0,1,21,64,96,128,160,192,256,511,2047,2048] {
            let n=8;let q=Tensor::randn([n,32,128],(Kind::BFloat16,dev));
            let ck=Tensor::randn([history,8,128],(Kind::BFloat16,dev));let cv=Tensor::randn_like(&ck);
            let k=Tensor::randn([n,8,128],(Kind::BFloat16,dev));let v=Tensor::randn_like(&k);
            let visible=(Tensor::arange(n,(Kind::Int64,dev)).unsqueeze(1)+history-Tensor::arange(history+n,(Kind::Int64,dev)).unsqueeze(0)).abs().lt(2048);
            let expected=crate::dflash::attention(&q,&ck,&cv,&k,&v,&visible,false);
            let actual=crate::dflash::attention(&q,&ck,&cv,&k,&v,&visible,true);let e=error(&actual,&expected);
            assert!(e.1<0.001,"GQA {history} tf32={tf32}: {e:?}");let mut rounds=Vec::new();
            for grouped in [false,true,true,false] {
                let _=crate::dflash::attention(&q,&ck,&cv,&k,&v,&visible,grouped);tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();let y=crate::dflash::attention(&q,&ck,&cv,&k,&v,&visible,grouped);crate::tp::graph::end().unwrap();
                for _ in 0..5{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let now=Instant::now();
                for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                rounds.push(json!({"grouped":grouped,"us":now.elapsed().as_secs_f64()*1e6/32.}));assert!(error(&y,&expected).1<0.001);crate::tp::graph::destroy();
            }
            records.push(json!({"op":"draft-gqa","history":history,"tf32":tf32,"error":e,"exact":actual.equal(&expected),"rounds":rounds}));
            let actual=crate::dflash::attention_fused(&q,&ck,&cv,&k,&v);let e=error(&actual,&expected);
            assert!(e.1<0.005,"fused GQA {history}, tf32={tf32}: {e:?}");
            let mut input=q.copy();let _=crate::dflash::attention_fused(&input,&ck,&cv,&k,&v);tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let y=crate::dflash::attention_fused(&input,&ck,&cv,&k,&v);crate::tp::graph::end().unwrap();
            for changed in [&q,&(-&q),&Tensor::zeros_like(&q)] {
                input.copy_(changed);crate::tp::graph::replay().unwrap();assert!(y.equal(&crate::dflash::attention_fused(changed,&ck,&cv,&k,&v)));
            }
            input.copy_(&q);for _ in 0..5{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let now=Instant::now();
            for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            let us=now.elapsed().as_secs_f64()*1e6/32.;crate::tp::graph::destroy();
            records.push(json!({"op":"draft-gqa-fused","history":history,"tf32":tf32,"error":e,"graph_changed_input_exact":true,"us":us}));
            let actual=crate::dflash::attention_shared_heads(&q,&ck,&cv,&k,&v,&visible);let e=error(&actual,&expected);assert!(e.1<0.005);
            crate::tp::graph::begin().unwrap();let shared=crate::dflash::attention_shared_heads(&input,&ck,&cv,&k,&v,&visible);crate::tp::graph::end().unwrap();
            for z in [&q,&(-&q),&q]{input.copy_(z);crate::tp::graph::replay().unwrap();assert!(shared.equal(&crate::dflash::attention_shared_heads(z,&ck,&cv,&k,&v,&visible)));}
            for _ in 0..5{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let now=Instant::now();
            for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            let us=now.elapsed().as_secs_f64()*1e6/32.;crate::tp::graph::destroy();
            records.push(json!({"op":"draft-gqa-shared-heads","history":history,"tf32":tf32,"error":e,"graph_changed_input_exact":true,"us":us}));
            std::fs::write(out.join("dataflow.json"),serde_json::to_string_pretty(&records).unwrap()).unwrap();
        }
    }
    crate::tp::set_tf32(false);
    let mut q=Tensor::randn([32,128],(Kind::Float,dev));let mut pools=Tensor::randn([5120,128],(Kind::Float,dev));
    let mixing=Tensor::randn([32,1],(Kind::Float,dev));let mut pos=Tensor::zeros([1],(Kind::Int64,dev));
    let _=crate::dsa::score_fused(&q,&pools,&mixing,&pos);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();let output=crate::dsa::score_fused(&q,&pools,&mixing,&pos);crate::tp::graph::end().unwrap();
    for len in [0,1,3,4,5,127,2047,2048,2049,16000,20480,7,0] {
        let _=pos.fill_(len-1);q.copy_(&Tensor::randn_like(&q));let complete=len/4;
        let _=pools.narrow(0,complete,5120-complete).fill_(f64::NAN);
        pools.narrow(0,0,complete).copy_(&Tensor::randn([complete,128],(Kind::Float,dev)));
        crate::tp::graph::replay().unwrap();assert!(output.equal(&crate::dsa::score_fused(&q,&pools,&mixing,&pos)));
        let saved=std::env::var("GLM53_DSA_SCORE_FUSED").unwrap_or_else(|_|"1".into());
        std::env::set_var("GLM53_DSA_SCORE_FUSED","1");let original=crate::dsa::score_fused(&q,&pools,&mixing,&pos);
        for mode in ["2","3","4"] {
            std::env::set_var("GLM53_DSA_SCORE_FUSED",mode);let tiled=crate::dsa::score_fused(&q,&pools,&mixing,&pos);assert!(original.equal(&tiled),"DSA tiling changed arithmetic len={len} mode={mode}");
        }
        std::env::set_var("GLM53_DSA_SCORE_FUSED","5");let tensor=crate::dsa::score_fused(&q,&pools,&mixing,&pos);
        let native_error=if complete>0{error(&tensor.narrow(0,0,complete),&original.narrow(0,0,complete))}else{(0.,0.)};
        assert!(native_error.1<2e-6 || native_error.0<2e-6,"compensated DSA {len} {native_error:?}");
        std::env::set_var("GLM53_DSA_SCORE_FUSED",saved);
        let expected=((q.matmul(&pools.narrow(0,0,complete).transpose(0,1))/128f64.sqrt()).relu()*&mixing).sum_dim_intlist(&[0i64][..],false,Kind::Float);
        let e=if complete>0{error(&output.narrow(0,0,complete),&expected)}else{(0.,0.)};
        assert!(e.1<2e-6 || e.0<2e-6,"DSA fused {len} {e:?}");
        assert!(output.narrow(0,complete,5120-complete).eq(f32::MIN as f64).all().int64_value(&[])!=0);
        records.push(json!({"op":"dsa-fused","len":len,"error":e,"graph_changed_len_exact":true,"poison_ignored":true}));
    }
    crate::tp::graph::destroy();
    pools.copy_(&Tensor::randn_like(&pools));
    for tf32 in [false,true] {crate::tp::set_tf32(tf32);for len in [128,2048,20480] {
        let _=pos.fill_(len-1);let visible=Tensor::arange(5120,(Kind::Int64,dev)).lt(len/4);let mut rounds=Vec::new();
        for mode in ["0","2","3","4","5","5","4","3","2","0"] {
            std::env::set_var("GLM53_DSA_SCORE_FUSED",mode);let fused=mode!="0";
            let call=||if fused{crate::dsa::score_fused(&q,&pools,&mixing,&pos)}else{
                ((q.matmul(&pools.transpose(0,1))/128f64.sqrt()).relu()*&mixing).sum_dim_intlist(&[0i64][..],false,Kind::Float).masked_fill(&visible.logical_not(),f32::MIN as f64)
            };
            let _=call();tch::Cuda::synchronize(0);crate::tp::graph::begin().unwrap();let _output=call();crate::tp::graph::end().unwrap();
            for _ in 0..5{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let now=Instant::now();
            for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"mode":mode,"us":now.elapsed().as_secs_f64()*1e6/32.}));crate::tp::graph::destroy();
        }
        records.push(json!({"op":"dsa-score-timing","len":len,"tf32":tf32,"rounds":rounds}));
    }}
    crate::tp::set_tf32(false);
    let dim=128;let w=crate::dsa::Weights{q:Tensor::zeros([4096,dim],(Kind::Float,dev)),k:Tensor::zeros([dim,dim],(Kind::Float,dev)),
        norm_w:Tensor::ones([dim],(Kind::Float,dev)),norm_b:Tensor::zeros([dim],(Kind::Float,dev)),score:Tensor::zeros([32,dim],(Kind::Float,dev)),
        ape:Tensor::randn([4,dim],(Kind::Float,dev)),gate:Tensor::zeros([dim,dim],(Kind::Float,dev))};
    for start in [0,1,3,4,2047,2048,4095,16000] {for n in [1,7,128,256] {for tied in [false,true] {
        let mut base=crate::dsa::State::new(20480,dim,dev);base.pools.copy_(&Tensor::randn_like(&base.pools));
        base.tail_k.copy_(&Tensor::randn_like(&base.tail_k));base.tail_gate.copy_(&Tensor::randn_like(&base.tail_gate));
        let _=base.pools.narrow(0,(start+3)/4,5120-(start+3)/4).fill_(f64::NAN);
        let p=crate::dsa::Projected{k:Tensor::randn([n,dim],(Kind::Float,dev)),gate:Tensor::randn([n,dim],(Kind::Float,dev)),
            q:Tensor::randn([n,32,dim],(Kind::Float,dev)),mixing:if tied{Tensor::zeros([n,32,1],(Kind::Float,dev))}else{Tensor::randn([n,32,1],(Kind::Float,dev))}};
        std::env::set_var("GLM53_DSA_PREFILL_LIMIT","0");let expected=base.snapshot().append_chunk(&w,&p,start);
        std::env::set_var("GLM53_DSA_PREFILL_LIMIT","1");let actual=base.snapshot().append_chunk(&w,&p,start);
        // GEMM shape changes can reorder near-ties; report exact ordering too.
        assert!(actual.sort(-1,false).0.equal(&expected.sort(-1,false).0),"DSA membership start={start} n={n} tied={tied}");
        records.push(json!({"op":"dsa-limit","start":start,"n":n,"tied":tied,"same_membership":true,"same_order":actual.equal(&expected)}));
    }}}
    std::env::remove_var("GLM53_DSA_PREFILL_LIMIT");
    std::fs::write(out.join("dataflow.json"),serde_json::to_string_pretty(&records).unwrap()).unwrap();
    eprintln!("[perf-dataflow] reduction, GQA, DSA boundaries/ties passed");
}


pub fn swiglu(out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();
    let dev=Device::Cuda(0);let mut records=Vec::new();tch::manual_seed(9227);
    let bits:Vec<i16>=(0..65536).map(|v|v as i16).collect();let gate=Tensor::from_slice(&bits).to_device(dev).view_dtype(Kind::Half).view([64,1024]);
    let up=gate.roll([917],[1]);
    let a=crate::moefast::grouped_swiglu(&gate,&up,false);let b=crate::moefast::grouped_swiglu(&gate,&up,true);
    let equal=a.eq_tensor(&b).logical_or(&a.isnan().logical_and(&b.isnan()));assert!(equal.all().int64_value(&[])!=0,"SwiGLU exhaustive half bit patterns");
    for rows in [1,7,17,56,64,128,512,2048] {
        let mut gate=(Tensor::randn([rows,1024],(Kind::Float,dev))*15.).to_kind(Kind::Half);let up=(Tensor::randn_like(&gate)*7.).to_kind(Kind::Half);
        let expected=crate::moefast::grouped_swiglu(&gate,&up,false);let y=crate::moefast::grouped_swiglu(&gate,&up,true);
        assert!(y.equal(&expected),"SwiGLU rows={rows}: {:?}",error(&y.to_kind(Kind::Float),&expected.to_kind(Kind::Float)));let mut rounds=Vec::new();
        for fused in [false,true,true,false] {
            let _=crate::moefast::grouped_swiglu(&gate,&up,fused);tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let y=crate::moefast::grouped_swiglu(&gate,&up,fused);crate::tp::graph::end().unwrap();
            let original=gate.copy();for changed in [&original,&(-&original),&original]{gate.copy_(changed);crate::tp::graph::replay().unwrap();assert!(y.equal(&crate::moefast::grouped_swiglu(changed,&up,fused)));}
            for _ in 0..5{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);let now=Instant::now();for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"fused":fused,"us":now.elapsed().as_secs_f64()*1e6/32.}));crate::tp::graph::destroy();
        }
        records.push(json!({"rows":rows,"exact":true,"exhaustive_half_bit_patterns":true,"rounds":rounds}));
        std::fs::write(out.join("swiglu.json"),serde_json::to_string_pretty(&records).unwrap()).unwrap();
    }
}

pub fn cross_stage(model:&Path,out:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();std::fs::create_dir_all(out).unwrap();
    let dev=Device::Cuda(0);tch::manual_seed(9238);let mut records=Vec::new();
    let equivalent=|a:&Tensor,b:&Tensor|a.eq_tensor(b).logical_or(&a.isnan().logical_and(&b.isnan())).all().int64_value(&[])!=0;
    for (rows,cols) in [(17,4096),(64,6144),(128,4096),(512,6144),(2048,4096),(2048,6144)] {for rounded in [false,true] {
        let raw=Tensor::randn([rows,cols],(Kind::Float,dev))*70000.;
        let scale=Tensor::rand([cols],(Kind::Float,dev))*0.03;
        let original=raw.copy();let original_scale=scale.copy();
        let run=|fused:bool|{let y=raw.copy();if fused{crate::dense_fp8::epilogue_inplace(&y,&scale,rounded);y}else{let y=y*&scale.unsqueeze(0);if rounded{y.to_kind(Kind::Half).to_kind(Kind::Float)}else{y}}};
        assert!(equivalent(&run(false),&run(true)));let mut rounds=Vec::new();
        for fused in [false,true,true,false] {
            let _=run(fused);tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let y=run(fused);crate::tp::graph::end().unwrap();
            // Tensor handles can be copied without allocating/changing captured addresses.
            for value in [1.,-1.,0.] {raw.shallow_clone().copy_(&(&original*value));scale.shallow_clone().copy_(&(&original_scale*value));crate::tp::graph::replay().unwrap();assert!(equivalent(&y,&run(false)));}
            raw.shallow_clone().copy_(&original);scale.shallow_clone().copy_(&original_scale);
            for _ in 0..4 {crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            let start=Instant::now();for _ in 0..24 {crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"fused":fused,"us":start.elapsed().as_secs_f64()*1e6/24.}));crate::tp::graph::destroy();
        }
        let _=raw.shallow_clone().fill_(f64::INFINITY);let _=scale.shallow_clone().fill_(0.);assert!(equivalent(&run(false),&run(true)));
        records.push(json!({"op":"fp8-epilogue","rows":rows,"cols":cols,"rounded":rounded,"exact":true,"changed_graph_and_nonfinite":true,"rounds":rounds}));
    }}
    // Full GEMM + post-processing, actual dimensions and the same quantized weights.
    std::env::set_var("GLM53_FP8_LARGE","0");
    for rows in [17,128,512] {let x=Tensor::randn([rows,4096],(Kind::Float,dev))*0.03;
        let (q,s)=crate::dense_fp8::quantize(&(Tensor::randn([4096,4096],(Kind::Float,dev))*0.01).to_kind(Kind::Half));
        for partial in [false,true] {std::env::set_var("GLM53_FP8_EPILOGUE","0");let a=crate::dense_fp8::run(&x,&q,&s,partial);
            std::env::set_var("GLM53_FP8_EPILOGUE","1");let b=crate::dense_fp8::run(&x,&q,&s,partial);assert!(a.equal(&b));
        }
    }
    let mut pool=crate::moefast::MoeFast::new_with_tp(model,4,288,304,dev,crate::tp::Tp{rank:0,world:2});
    pool.ensure_many(3,&(0..288).collect::<Vec<_>>(),dev);pool.assume_hot=true;
    std::env::set_var("GLM53_GROUPED_REDUCE","1");std::env::set_var("GLM53_GROUPED_SWIGLU","1");
    std::env::set_var("GLM53_PREFILL_RECON_MIN_ROWS","64");
    for rows in [1,128,512,2048] {for experts in [1,32,288] {
        let x=(Tensor::randn([rows,4096],(Kind::Float,dev))*0.03).to_kind(Kind::Half);
        let host:Vec<i64>=(0..rows).flat_map(|t|(0..8).map(move|e|(t*19+e*31)%experts)).collect();
        let ids=Tensor::from_slice(&host).view([rows,8]).to_device(dev);
        let weights=Tensor::rand([rows,8],(Kind::Float,dev)).softmax(-1,Kind::Float)*2.5;
        std::env::set_var("GLM53_GROUPED_INDEX_PACK","0");let expected=pool.expert_grouped(3,&x,&ids,&weights,true);
        std::env::set_var("GLM53_GROUPED_INDEX_PACK","1");let actual=pool.expert_grouped(3,&x,&ids,&weights,true);assert!(actual.equal(&expected),"route index pack rows={rows} experts={experts}");
        let mut rounds=Vec::new();
        for packed in [false,true,true,false] {std::env::set_var("GLM53_GROUPED_INDEX_PACK",if packed{"1"}else{"0"});
            tch::Cuda::synchronize(0);let now=Instant::now();let y=pool.expert_grouped(3,&x,&ids,&weights,true);tch::Cuda::synchronize(0);
            let ms=now.elapsed().as_secs_f64()*1000.;assert!(y.equal(&expected));rounds.push(json!({"packed":packed,"ms":ms}));
        }
        records.push(json!({"op":"grouped-index-pack","rows":rows,"experts":experts,"exact":true,"rounds":rounds}));
    }}
    std::fs::write(out.join("cross-stage.json"),serde_json::to_string_pretty(&records).unwrap()).unwrap();
    eprintln!("[cross-stage] FP8 epilogue/graph and real expert gather/scatter exact");
}
