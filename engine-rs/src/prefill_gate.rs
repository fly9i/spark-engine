//! Prefill L0 gate + in-process ABBA (TP2, both ranks, the serving model setup).
//! GLM53_GATE_SUITE: suite JSON (cases[0].prompt_ids); GLM53_GATE_ROWS (default 2048);
//! GLM53_GATE_ARMS: "name:FLAG=V,FLAG=V;name2:..." (every flag named by any arm is reset to its
//! first arm's value before each run); GLM53_GATE_ORDER: arm names, e.g. "a,b,b,a,a,b,b,a".
//! GLM53_GATE_CHUNKS: rows per prefill call (default = rows; e.g. 1024 runs 2 chunked calls).
//! Every arm's outputs (last logits, all layer states, features) are compared bitwise with the
//! first arm's; the gate fails unless the arm is listed in GLM53_GATE_LOSSY.
use std::path::Path;
use std::time::Instant;
use tch::{Device,Kind,Tensor};
use serde_json::json;
use crate::forward::{Engine,LayerState,DecodeStates};

fn state_tensors(s:&DecodeStates,rows:i64)->Vec<(String,Tensor)> {
    let mut out=Vec::new();
    for (i,l) in s.0.iter().enumerate() {match l {
        LayerState::Kda(k)=>{out.push((format!("L{i}.h"),k.h.shallow_clone()));out.push((format!("L{i}.conv"),k.conv.shallow_clone()));}
        LayerState::MlaLatent(m)=>{
            out.push((format!("L{i}.latent"),m.latent.narrow(0,0,rows)));
            out.push((format!("L{i}.pools"),m.index.pools.narrow(0,0,(rows+3)/4)));
            out.push((format!("L{i}.tail_k"),m.index.tail_k.shallow_clone()));out.push((format!("L{i}.tail_gate"),m.index.tail_gate.shallow_clone()));
            out.push((format!("L{i}.len"),m.len.shallow_clone()));}
        _=>panic!("gate expects KDA / latent MLA states")}}
    out
}
fn barrier(){let t=Tensor::zeros([1],(Kind::Float,Device::Cuda(0)));crate::tp::allreduce(&t);tch::Cuda::synchronize(0);}
fn bits_equal(a:&Tensor,b:&Tensor)->bool {
    a.size()==b.size() && a.kind()==b.kind() && {
        let (a,b)=if a.kind()==Kind::Float8e4m3fn{(a.view_dtype(Kind::Uint8),b.view_dtype(Kind::Uint8))}else{(a.shallow_clone(),b.shallow_clone())};
        // NaN-safe bit equality: compare raw bytes.
        let (a,b)=(a.contiguous().view_dtype(Kind::Uint8),b.contiguous().view_dtype(Kind::Uint8));a.equal(&b)}
}

pub fn run(model:&Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    for flag in ["GLM53_MHC_FUSED","GLM53_KDA_FUSED","GLM53_MLA_LATENT"]{std::env::set_var(flag,"1");}
    std::env::set_var("GLM53_PREFILL_BATCH","1");
    let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    let suite=std::env::var("GLM53_GATE_SUITE").unwrap_or_else(|_|"bench/p1-prefill/the-suite.json".into());
    let suite:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(&suite).unwrap()).unwrap();
    let all:Vec<i64>=suite["cases"][0]["prompt_ids"].as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).collect();
    let rows:usize=std::env::var("GLM53_GATE_ROWS").ok().and_then(|v|v.parse().ok()).unwrap_or(2048);
    assert!(rows<=all.len(),"suite has {} ids",all.len());
    let chunk:usize=std::env::var("GLM53_GATE_CHUNKS").ok().and_then(|v|v.parse().ok()).unwrap_or(rows);
    let lossy:Vec<String>=std::env::var("GLM53_GATE_LOSSY").unwrap_or_default().split(',').filter(|s|!s.is_empty()).map(String::from).collect();
    let arms:Vec<(String,Vec<(String,String)>)>=std::env::var("GLM53_GATE_ARMS").expect("GLM53_GATE_ARMS").split(';').filter(|s|!s.is_empty()).map(|a|{
        let (name,flags)=a.split_once(':').unwrap_or((a,""));
        (name.to_string(),flags.split(',').filter(|s|!s.is_empty()).map(|kv|{let (k,v)=kv.split_once('=').unwrap();(k.to_string(),v.to_string())}).collect())}).collect();
    let mut defaults:Vec<(String,String)>=Vec::new();
    for (_,fl) in &arms {for (k,_) in fl {if !defaults.iter().any(|(d,_)|d==k){defaults.push((k.clone(),std::env::var(k).unwrap_or_else(|_|"0".into())));}}}
    let order:Vec<String>=std::env::var("GLM53_GATE_ORDER").unwrap_or_else(|_|arms.iter().map(|a|a.0.clone()).collect::<Vec<_>>().join(",")).split(',').map(String::from).collect();
    let input=Tensor::from_slice(&all[..rows]).to_device(dev);
    let mut reference:Option<(Tensor,Vec<(String,Tensor)>,Vec<Tensor>)>=None;
    let mut results=Vec::new();let mut ok=true;
    for (run,name) in order.iter().enumerate() {
        let arm=arms.iter().find(|a|&a.0==name).unwrap_or_else(||panic!("unknown arm {name}"));
        for (k,v) in &defaults {std::env::set_var(k,v);}
        for (k,v) in &arm.1 {std::env::set_var(k,v);}
        crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
        tch::Cuda::synchronize(0);barrier();
        let t0=Instant::now();let mut state:Option<DecodeStates>=None;let mut feats:Vec<Vec<Tensor>>=Vec::new();let mut logits=Tensor::zeros([1],(Kind::Float,dev));
        let mut done=0;
        while done<rows {
            let n=chunk.min(rows-done);
            let (l,s,f)=eng.prefill_record_last(&input.narrow(0,done as i64,n as i64),state.take(),true);
            state=Some(s);feats.push(f);logits=l;done+=n;
        }
        tch::Cuda::synchronize(0);let ms=t0.elapsed().as_secs_f64()*1000.;
        let state=state.unwrap();
        let last=logits.get(logits.size()[0]-1).copy();
        let features:Vec<Tensor>=(0..feats[0].len()).map(|j|Tensor::cat(&feats.iter().map(|f|&f[j]).collect::<Vec<_>>(),0)).collect();
        let st=state_tensors(&state,rows as i64);
        let mut diff=Vec::new();
        if let Some((rl,rs,rf))=&reference {
            if !bits_equal(&last,rl){diff.push("logits".to_string());}
            for ((n,a),(_,b)) in st.iter().zip(rs.iter()) {if !bits_equal(a,b){diff.push(n.clone());}}
            if features.len()!=rf.len(){diff.push("features.len".into());}
            for (j,(a,b)) in features.iter().zip(rf.iter()).enumerate() {if !bits_equal(a,b){diff.push(format!("feature{j}"));}}
        } else {
            reference=Some((last.copy(),st.iter().map(|(n,t)|(n.clone(),t.copy())).collect(),features.iter().map(|t|t.copy()).collect()));
        }
        let max_abs=reference.as_ref().map_or(0.,|(rl,_,_)|(&last-rl).abs().max().double_value(&[]));
        // Cross-process fingerprint: position-weighted sums of the raw bytes of every output tensor.
        let fp=|t:&Tensor|->i64{let b=t.contiguous().view_dtype(Kind::Uint8).reshape([-1]).to_kind(Kind::Int64);
            let w=Tensor::arange(b.numel() as i64,(Kind::Int64,b.device())).remainder(65521)+1;(b*w).sum(Kind::Int64).int64_value(&[])};
        let mut hash:i64=fp(&last);for (_,t) in &st {hash=hash.wrapping_mul(1_000_003).wrapping_add(fp(t));}
        for t in &features {hash=hash.wrapping_mul(1_000_003).wrapping_add(fp(t));}
        if !diff.is_empty() && !lossy.contains(name) {ok=false;}
        eprintln!("[prefill-gate] rank{} run {run} arm {name} rows {rows} chunk {chunk} ms {ms:.2} exact {} diffs {} logits_max_abs {max_abs:.3e} fingerprint {hash:016x} {}",tp.rank,diff.is_empty(),diff.len(),
            diff.iter().take(6).cloned().collect::<Vec<_>>().join(","));
        results.push(json!({"run":run,"arm":name,"rows":rows,"chunk":chunk,"ms":ms,"exact":diff.is_empty(),"diffs":diff,"logits_max_abs":max_abs}));
        drop(state);
    }
    for (k,v) in &defaults {std::env::set_var(k,v);}
    let mut summary=serde_json::Map::new();
    for (name,_) in &arms {
        let mut v:Vec<f64>=results.iter().filter(|r|r["arm"]==json!(name)).map(|r|r["ms"].as_f64().unwrap()).collect();
        v.sort_by(|a,b|a.partial_cmp(b).unwrap());
        // first occurrence of each arm includes shape/allocator warm-up; report both
        if !v.is_empty(){summary.insert(name.clone(),json!({"runs":v.len(),"median_ms":v[v.len()/2],"min_ms":v[0]}));}
    }
    eprintln!("[prefill-gate] rank{} summary {} gate {}",tp.rank,serde_json::Value::Object(summary.clone()),if ok{"PASS"}else{"FAIL"});
    if let Ok(out)=std::env::var("GLM53_GATE_OUT") {
        std::fs::write(format!("{out}-rank{}.json",tp.rank),serde_json::to_string_pretty(&json!({"results":results,"summary":summary,"pass":ok})).unwrap()).unwrap();
    }
    barrier();
    assert!(ok,"prefill gate failed");
}

/// Needs GLM53_RDMA_BIG_INIT=1; toggles GLM53_RDMA_BIG per call.
pub fn rdma_big_probe() {
    let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);let dev=Device::Cuda(0);
    tch::manual_seed(11+tp.rank as i64);let mut ok=true;
    for &rows in &[64i64,256,512,1024,2048,4096] {
        let x=Tensor::randn([2*rows,4096],(Kind::Float,dev));let h=Tensor::randn([rows,4096],(Kind::Float,dev));let hh=h.to_kind(Kind::Half);
        let mut res=Vec::new();
        for big in ["0","1"] {
            std::env::set_var("GLM53_RDMA_BIG",big);
            let rs=crate::tp::reduce_scatter_rows(&x);let ag=crate::tp::all_gather_rows(&h);let agh=crate::tp::all_gather_rows(&hh);
            barrier();
            let mut t=[0f64;3];
            for it in 0..25 {
                let t0=Instant::now();let _=crate::tp::reduce_scatter_rows(&x);tch::Cuda::synchronize(0);let a=t0.elapsed().as_secs_f64();
                let t1=Instant::now();let _=crate::tp::all_gather_rows(&h);tch::Cuda::synchronize(0);let b=t1.elapsed().as_secs_f64();
                let t2=Instant::now();let _=crate::tp::all_gather_rows(&hh);tch::Cuda::synchronize(0);let c=t2.elapsed().as_secs_f64();
                if it>=5 {t[0]+=a;t[1]+=b;t[2]+=c;}
            }
            res.push((rs,ag,agh,t.map(|v|v/20.*1e6)));
        }
        let eq=bits_equal(&res[0].0,&res[1].0)&&bits_equal(&res[0].1,&res[1].1)&&bits_equal(&res[0].2,&res[1].2);ok&=eq;
        let mib=(rows*4096*4) as f64/1048576.;
        println!("rdma-big rank{} rows {rows:5} ({mib:5.1} MiB/rank) bitwise {eq} | reduce_scatter nccl {:7.0} us rdma {:7.0} us | all_gather f32 nccl {:7.0} rdma {:7.0} | f16 nccl {:7.0} rdma {:7.0}",
            tp.rank,res[0].3[0],res[1].3[0],res[0].3[1],res[1].3[1],res[0].3[2],res[1].3[2]);
    }
    std::env::set_var("GLM53_RDMA_BIG","0");barrier();
    println!("rdma-big-probe rank{} {}",tp.rank,if ok{"PASS"}else{"FAIL"});assert!(ok);
}

/// L1-a local numeric check: DSA indexer projections with the real weights of every MLA layer, TF32 cuBLAS
/// (current) and the BF16-weight FP32-FMA kernel, each against an FP64 reference. Activations are unit
/// normal (the projections consume RMS-normalized rows). Reports max relative error per layer/weight.
pub fn index_bf16_probe(model:&Path) {
    let _guard=tch::no_grad_guard();let dev=Device::Cuda(0);tch::manual_seed(5);
    let idx=crate::safetensors::ShardIndex::scan(model).unwrap();
    let mut layers:Vec<usize>=(0..45).filter(|l|idx.get_f32(&format!("model.language_model.layers.{l}.self_attn.indexer.wk.weight")).is_ok()).collect();layers.sort();
    let mut worst=[0f64;2];
    for l in layers {
        let p=format!("model.language_model.layers.{l}.self_attn.indexer");
        for (name,k) in [("wq_b",1536i64),("wk",4096),("weights_proj",4096),("index_kpool_compress_gate",4096)] {
            let (v,shape)=idx.get_f32(&format!("{p}.{name}.weight").replace(".index_kpool_compress_gate.weight",".index_kpool_compress_gate")).unwrap();
            let w=Tensor::from_slice(&v).view([shape[0] as i64,shape[1] as i64]).to_device(dev);assert_eq!(w.size()[1],k);
            let wb=w.to_kind(Kind::BFloat16);assert!(wb.to_kind(Kind::Float).equal(&w),"{name} not BF16-exact");
            for m in [1i64,2,5,8,16] {
                let x=Tensor::randn([m,k],(Kind::Float,dev));
                let reference=x.to_kind(Kind::Double).matmul(&w.to_kind(Kind::Double).transpose(0,1));
                crate::tp::set_tf32(true);let tf32=x.matmul(&w.transpose(0,1));
                let y=Tensor::empty([m,w.size()[0]],(Kind::Float,dev));
                extern "C"{fn rs_bf16w_rows(x:*const f32,ldx:i32,w:*const std::ffi::c_void,y:*mut f32,m:i32,n:i32,k:i32)->i32;}
                assert_eq!(unsafe{rs_bf16w_rows(x.data_ptr().cast(),k as i32,wb.data_ptr(),y.data_ptr().cast(),m as i32,w.size()[0] as i32,k as i32)},0);
                let scale=reference.abs().max().double_value(&[]).max(1e-30);
                let e=|t:&Tensor|(t.to_kind(Kind::Double)-&reference).abs().max().double_value(&[])/scale;
                let (a,b)=(e(&tf32),e(&y));worst[0]=worst[0].max(a);worst[1]=worst[1].max(b);
                if m==8 {println!("index-bf16 layer {l:2} {name:26} m {m:2}: max|err|/max|ref| TF32 {a:.3e}  FP32-FMA {b:.3e}");}
            }
        }
    }
    println!("index-bf16-probe worst over layers/weights/m: TF32 {:.3e}  FP32-FMA(BF16 weights) {:.3e}",worst[0],worst[1]);
}

/// P1b local check: FP8 big GEMM (shim/fp8_big.cu) against the FP8 expand path (FP8 -> Half weight,
/// cuBLAS, scale epilogue) at the per-rank prefill shapes, plus timing of the retained-Half cuBLAS GEMM
/// (current service), the expand path and the big kernel (3/4/5 stages). Weights rotate over a working
/// set larger than L2 (GLM53_PROBE_SET_MB, default 256). GLM53_PROBE_M: comma list of row counts.
pub fn fp8_big_probe() {
    let dev=Device::Cuda(0);tch::manual_seed(7);
    let ms:Vec<i64>=std::env::var("GLM53_PROBE_M").unwrap_or_else(|_|"2048".into()).split(',').map(|v|v.parse().unwrap()).collect();
    let set_mb:i64=std::env::var("GLM53_PROBE_SET_MB").ok().and_then(|v|v.parse().ok()).unwrap_or(256);
    // (name, N, K) per rank under TP2.
    let shapes=[("kda.qkv",4096,4096),("kda.o",4096,4096),("mla.q_a",1536,4096),("mla.q_b",8192,1536),("mla.kv_a",512,4096),
        ("mla.o",4096,8192),("shared.gu",1024,4096),("shared.down",4096,1024),("dense.gu",6144,4096),("dense.down",4096,6144)];
    let time=|f:&mut dyn FnMut(usize),count:usize,iters:usize|->f64 {
        for i in 0..count.min(4) {f(i);} tch::Cuda::synchronize(0);
        let t=Instant::now();for i in 0..iters {f(i%count);} tch::Cuda::synchronize(0);t.elapsed().as_secs_f64()*1e3/iters as f64};
    for &m in &ms { for (name,n,k) in shapes {
        let count=((set_mb<<20)/(n*k)).clamp(2,64) as usize;
        let quant:Vec<(Tensor,Tensor)>=(0..count).map(|_|crate::dense_fp8::quantize(&(Tensor::randn([n,k],(Kind::Float,dev))*0.02).to_kind(Kind::Half))).collect();
        let halfw:Vec<Tensor>=quant.iter().map(|(q,s)|(q.to_kind(Kind::Float)*s.unsqueeze(1)).to_kind(Kind::Half)).collect();
        let x=Tensor::randn([m,k],(Kind::Float,dev));let xh=x.to_kind(Kind::Half).contiguous();
        let mut worst=0f64;let mut worst_p=0f64;let mut mism=0i64;
        let check_codes:Vec<String>=std::env::var("GLM53_PROBE_STAGES").unwrap_or_else(|_|"4".into()).split(',').map(String::from).collect();
        for code in &check_codes {std::env::set_var("GLM53_FP8_BIG_STAGES",code);
        for (q,s) in quant.iter().take(2) {
            for rounded in [true,false] {
                let r=crate::dense_fp8::run(&x,q,s,!rounded);
                let y=Tensor::empty([m,n],(Kind::Float,dev));crate::dense_fp8::big_into(&xh,q,s,&y,rounded);
                let d=(&y-&r).abs().max().double_value(&[])/r.abs().max().double_value(&[]).max(1e-30);
                if rounded {worst=worst.max(d);mism+=y.ne_tensor(&r).sum(Kind::Int64).int64_value(&[]);} else {worst_p=worst_p.max(d);}
            }
        }}
        std::env::remove_var("GLM53_FP8_BIG_STAGES");
        let t_half=time(&mut |i|{let _=xh.matmul(&halfw[i].transpose(0,1));},count,40);
        let t_exp=time(&mut |i|{let _=crate::dense_fp8::run(&x,&quant[i].0,&quant[i].1,false);},count,40);
        let y=Tensor::empty([m,n],(Kind::Float,dev));
        let mut tb=Vec::new();
        let codes:Vec<String>=std::env::var("GLM53_PROBE_STAGES").unwrap_or_else(|_|"3,4,5".into()).split(',').map(String::from).collect();
        for st in &codes {std::env::set_var("GLM53_FP8_BIG_STAGES",st);
            tb.push(time(&mut |i|crate::dense_fp8::big_into(&xh,&quant[i].0,&quant[i].1,&y,true),count,40));}
        std::env::remove_var("GLM53_FP8_BIG_STAGES");
        let tf=2.0*(m*n*k) as f64/1e9;
        let per:String=codes.iter().zip(&tb).map(|(c,t)|format!(" k{c} {t:.3}")).collect();
        println!("[fp8-big] m {m} {name} n {n} k {k} set {count} | rel_max rounded {worst:.2e} partial {worst_p:.2e} differing {mism}/{} ({} variants) | half {t_half:.3} ms ({:.1} TF/s) expand {t_exp:.3} | big{per} ms ({:.1} TF/s best)",
            2*m*n,check_codes.len(),tf/t_half,tf/tb.iter().cloned().fold(f64::MAX,f64::min));
    }}
}

/// Item 5 local check: a chain of FP8 skinny GEMMs (real decode shape: M rows x 4096 -> 4096, 64 distinct weights,
/// ~1 GiB, beyond L2) captured into one CUDA graph with GLM53_PDL off/on; replay timing and bitwise outputs.
pub fn pdl_probe() {
    let dev=Device::Cuda(0);let _g=tch::no_grad_guard();tch::manual_seed(11);
    std::env::set_var("GLM53_FP8_SKINNY","1");
    let m:i64=std::env::var("GLM53_PROBE_M").ok().and_then(|v|v.parse().ok()).unwrap_or(2);
    let layers=64usize;
    // Low-rank pairs 4096->512->4096: the narrow producer occupies few SMs, so the PDL-launched consumer becomes
    // resident while the producer is still reducing/writing X; a read of X above griddepcontrol.wait then races.
    let quant:Vec<(Tensor,Tensor)>=(0..layers).map(|i|{let (n,k)=if i%2==0 {(512,4096)} else {(4096,512)};
        let sd=1.0/(k as f64).sqrt();crate::dense_fp8::quantize(&(Tensor::randn([n,k],(Kind::Float,dev))*sd).to_kind(Kind::Half))}).collect();
    let x0=Tensor::randn([m,4096],(Kind::Float,dev)).to_kind(Kind::BFloat16);
    // Normalise only every 8th layer: the other GEMMs read X written directly by the previous skinny GEMM, whose
    // early trigger is the PDL hazard (a norm kernel in between never triggers early and hides it).
    // BF16 path (IN=2, the serving BF16-resident boundary): skinny -> BF16 cast -> skinny. The cast never triggers
    // early; its implicit trigger at exit gives no memory visibility, so X reads must still follow griddepcontrol.wait.
    let run_chain=|x:&Tensor|->Tensor{let mut y=x.shallow_clone();for (i,(q,s)) in quant.iter().enumerate() {y=crate::dense_fp8::run_bf16(&y,q,s,false);
        if i%8==7 {let n=(&y.to_kind(Kind::Float)*&y.to_kind(Kind::Float)).mean_dim(&[-1i64][..],true,Kind::Float);y=(&y.to_kind(Kind::Float)*(n+1e-6).rsqrt()).to_kind(Kind::BFloat16);}}y};
    let mut results=Vec::new();
    for arm in ["0","1","1","0"] {
        std::env::set_var("GLM53_PDL",arm);
        let input=x0.copy();let _=run_chain(&input);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let out=run_chain(&input);crate::tp::graph::end().unwrap();
        let g=crate::tp::graph::Owned::take();
        for _ in 0..3 {g.replay();} tch::Cuda::synchronize(0);
        // every replay must reproduce the eager arm-0 result, not just the captured one
        let mut stable=true;
        let t=Instant::now();for _ in 0..20 {g.replay();} tch::Cuda::synchronize(0);
        let ms=t.elapsed().as_secs_f64()*1000./20.;
        for _ in 0..200 {g.replay();tch::Cuda::synchronize(0);stable&=out.equal(&results.first().map(|(_,o):&(&str,Tensor)|o.shallow_clone()).unwrap_or(out.copy()));}
        println!("[pdl-probe] m {m} pdl {arm} chain of {layers} BF16 skinny GEMMs (4096<->512 pairs): {ms:.3} ms per replay ({:.1} us per GEMM), 200 replays match arm 1: {stable}",ms*1000./layers as f64);
        results.push((arm,out.copy()));
    }
    println!("[pdl-probe] outputs bitwise equal across arms: {}",results.iter().all(|(_,o)|o.equal(&results[0].1)));
}

/// Item 5 feasibility: per "layer", a 27 us spin (allreduce wait stand-in) then an FP8 skinny GEMM (4096x4096,
/// M rows) over 64 distinct weights. Arm "pf P": during the spin, a side stream bulk-prefetches the first P MiB of
/// that layer's weight into L2. Eager, CUDA-event timed per chain. GLM53_PROBE_M rows (default 2).
pub fn l2pf_probe() {
    let dev=Device::Cuda(0);let _g=tch::no_grad_guard();tch::manual_seed(13);std::env::set_var("GLM53_FP8_SKINNY","1");
    let m:i64=std::env::var("GLM53_PROBE_M").ok().and_then(|v|v.parse().ok()).unwrap_or(2);
    let spin_ns:i64=std::env::var("GLM53_PROBE_SPIN_NS").ok().and_then(|v|v.parse().ok()).unwrap_or(27000);
    let quant:Vec<(Tensor,Tensor)>=(0..64).map(|_|crate::dense_fp8::quantize(&(Tensor::randn([4096,4096],(Kind::Float,dev))*0.02).to_kind(Kind::Half))).collect();
    // GLM53_PROBE_BF16=1: BF16 activations (IN=2, the serving BF16-resident path) instead of Float (IN=0).
    let bf16=std::env::var("GLM53_PROBE_BF16").as_deref()==Ok("1");
    let x=if bf16 {Tensor::randn([m,4096],(Kind::Float,dev)).to_kind(Kind::BFloat16)} else {Tensor::randn([m,4096],(Kind::Float,dev))};
    extern "C"{fn rs_spin_ns(ns:i64)->i32;fn rs_l2_prefetch(p:*const std::ffi::c_void,bytes:i64)->i32;
        fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;fn rs_stream_join(n:i32)->i32;}
    let chain=|pf_mib:i64|{for (q,s) in &quant {
            if pf_mib>0 {unsafe{assert_eq!(rs_stream_fork(1),0);assert_eq!(rs_stream_set(0),0);
                assert_eq!(rs_l2_prefetch(q.data_ptr(),pf_mib<<20),0);assert_eq!(rs_stream_set(-1),0);}}
            unsafe{assert_eq!(rs_spin_ns(spin_ns),0);}
            if pf_mib>0 {unsafe{assert_eq!(rs_stream_join(1),0);}}
            let _=if bf16 {crate::dense_fp8::run_bf16(&x,q,s,true)} else {crate::dense_fp8::run(&x,q,s,false)};}};
    for arm in [0i64,4,8,12,16,0,4,8,12,16] {
        chain(arm);tch::Cuda::synchronize(0);
        let t=Instant::now();for _ in 0..5 {chain(arm);} tch::Cuda::synchronize(0);
        let us=t.elapsed().as_secs_f64()*1e6/5./64.;
        println!("[l2pf] m {m} bf16 {bf16} spin {} us prefetch {arm} MiB: {us:.1} us per spin+GEMM",spin_ns/1000);
    }
}

/// Proposal 3: mhc_pre row-count dependence in isolation (single GPU). Random residual [8,4,4096] and BF16/FP32 fn;
/// z/post/comb of rows 0..k from mhc_pre(first k rows) vs mhc_pre(all 8 rows), for k = 1..8.
pub fn mhc_rows_probe() {
    let dev=Device::Cuda(0);let _g=tch::no_grad_guard();tch::manual_seed(5);
    for (flag,val) in [("GLM53_MHC_PRE_FUSED","1"),("GLM53_MHC_PRE_TC","1")] {std::env::set_var(flag,val);}
    let res=Tensor::randn([8,4,4096],(Kind::Float,dev));
    let scale=Tensor::randn([3],(Kind::Float,dev))*0.1;let base=Tensor::randn([24],(Kind::Float,dev))*0.1;let ln=Tensor::ones([4096],(Kind::Float,dev));
    for fk in [Kind::BFloat16,Kind::Float] {
        let fn_=(Tensor::randn([24,16384],(Kind::Float,dev))*0.01).to_kind(fk);
        let (pa,za)=crate::mhc::mhc_pre(&res,&fn_,&scale,&base,&ln);
        for k in 1..8i64 {
            let (pk,zk)=crate::mhc::mhc_pre(&res.narrow(0,0,k).contiguous(),&fn_,&scale,&base,&ln);
            let n=|a:&Tensor|a.narrow(0,0,k);
            println!("[mhc-rows] fn {fk:?} k {k} vs 8: z {} post {} comb {} (max |dz| {:.2e})",n(&za).equal(&zk),n(&pa.post_mix).equal(&pk.post_mix),n(&pa.comb).equal(&pk.comb),
                (n(&za)-&zk).abs().max().double_value(&[]));
        }
    }
}
