//! Exact vocabulary-sharded verifier top-1, without a full-vocabulary SUM.
//! MaxOps and ArgMaxOps use the same GreaterOrNan rule in ATen: NaNs win,
//! then the larger value, and the lowest index wins ties (including +/-0).
//! Equal, contiguous shards ordered by rank make a rank tie the global-ID tie.
use tch::{Kind,Tensor};

pub(crate) fn enabled()->bool {
    std::env::var("GLM53_TARGET_TOP1_TP").as_deref()==Ok("1")
}

fn check_layout(local:&Tensor,start:i64,total:i64,rank:usize,world:usize) {
    assert_eq!(local.kind(),Kind::Float,"top1 packet must preserve FP32 head values");
    assert_eq!(local.dim(),2);assert!(local.size()[0]>0&&local.size()[1]>0);
    assert!(world>0&&rank<world);assert!(total>0&&total<(1i64<<24),"FP32 packet IDs must be exact");
    assert_eq!(total,local.size()[1]*world as i64,"requires equal vocabulary shards");
    assert_eq!(start,local.size()[1]*rank as i64,"requires contiguous rank-ordered shards");
}

fn packet(local:&Tensor,start:i64,total:i64,rank:usize,world:usize)->Tensor {
    check_layout(local,start,total,rank,world);
    let (values,ids)=local.max_dim(1,false);
    let result=Tensor::zeros([world as i64,local.size()[0],2],(Kind::Float,local.device()));
    let lane=result.get(rank as i64);
    lane.select(1,0).copy_(&values);
    lane.select(1,1).copy_(&(ids+start).to_kind(Kind::Float));
    result
}

fn choose(packets:&Tensor)->Tensor {
    // Non-owner lanes are zero before SUM. Each score (including NaN/Inf)
    // has exactly one owner: SUM never combines different ranks' head scores.
    // SUM can change -0 to +0, as full logits SUM can, without changing argmax.
    let rank=packets.select(2,0).argmax(0,false);
    packets.select(2,1).gather(0,&rank.unsqueeze(0),false).squeeze_dim(0).to_kind(Kind::Int64)
}

pub(crate) fn from_local(local:&Tensor,start:i64,total:i64)->Tensor {
    let tp=crate::tp::world();
    check_layout(local,start,total,tp.rank,tp.world);
    if tp.world==1 {return local.argmax(-1,false);}
    let packets=packet(local,start,total,tp.rank,tp.world);
    crate::tp::allreduce(&packets);
    choose(&packets)
}

/// GLM53_GUMBEL_FUSED: from_local(perturb(local)) with the noise evaluated inside the top-1 kernel (bitwise the same
/// packet; see gumbel_argmax_part). `temps`/`keys` are the sampling parameter buffers.
pub(crate) fn from_local_gumbel(local:&Tensor,start:i64,total:i64,temps:&Tensor,keys:&Tensor)->Tensor {
    let tp=crate::tp::world();
    check_layout(local,start,total,tp.rank,tp.world);assert!(local.is_contiguous());
    let rows=local.size()[0];let dev=local.device();
    let part_v=Tensor::empty([rows*32],(Kind::Float,dev));let part_i=Tensor::empty([rows*32],(Kind::Int,dev));
    let packets=Tensor::empty([tp.world as i64,rows,2],(Kind::Float,dev));
    extern "C"{fn rs_gumbel_argmax_packet(values:*const f32,rows:i32,width:i32,start:i64,temps:*const f32,keys:*const u64,part_v:*mut f32,part_i:*mut i32,packet:*mut f32,world:i32,rank:i32)->i32;}
    assert_eq!(unsafe{rs_gumbel_argmax_packet(local.data_ptr().cast(),rows as i32,local.size()[1] as i32,start,temps.data_ptr().cast(),keys.data_ptr().cast(),
        part_v.data_ptr().cast(),part_i.data_ptr().cast(),packets.data_ptr().cast(),tp.world as i32,tp.rank as i32)},0,"fused gumbel top-1");
    if tp.world==1 {return packets.select(2,1).get(0).to_kind(Kind::Int64);}
    crate::tp::allreduce(&packets);
    choose(&packets)
}

// Kept here for a bounded probe of exactly the old communication/selection
// path from identical precomputed local logits, independent of head GEMM.
fn full_from_local(local:&Tensor,start:i64,total:i64)->Tensor {
    let full=Tensor::zeros([local.size()[0],total],(local.kind(),local.device()));
    full.narrow(1,start,local.size()[1]).copy_(local);
    crate::tp::allreduce(&full);full.argmax(-1,false)
}

fn simulate(full:&Tensor,world:usize)->Tensor {
    let width=full.size()[1]/world as i64;
    assert_eq!(full.size()[1],width*world as i64);
    let packets:Vec<_>=(0..world).map(|rank|packet(&full.narrow(1,rank as i64*width,width),
        rank as i64*width,full.size()[1],rank,world)).collect();
    let mut combined=packets[0].shallow_clone();
    for next in &packets[1..] {combined=&combined+next;}
    choose(&combined)
}

/// Adversarial rows: negative-only, all -Inf, all zero, cross-shard ties,
/// multiple NaNs, NaN versus +Inf, +/-0, tiny normals and subnormals.
fn fixtures(world:usize,width:usize,rotation:usize)->Vec<Vec<f32>> {
    let total=world*width;assert!(world>=2&&width>=4);
    let mut rows=vec![vec![-19.;total];12];
    rows[0][total-1]=-1.;rows[1].fill(f32::NEG_INFINITY);
    rows[2].fill(0.);rows[3][width-1]=7.;rows[3][width]=7.;
    rows[4][width-1]=f32::NAN;rows[4][width+1]=f32::NAN;
    rows[5][0]=f32::INFINITY;rows[5][total-1]=f32::NAN;
    rows[6][width-1]=-0.;rows[6][width]=0.;
    rows[7][width-1]=0.;rows[7][width]=-0.;
    rows[8][width-1]=f32::INFINITY;rows[8][total-1]=f32::INFINITY;
    rows[9].fill(f32::NEG_INFINITY);rows[9][total-1]=-f32::MAX;
    rows[10].fill(-f32::MIN_POSITIVE);rows[10][total-1]=f32::MIN_POSITIVE;
    rows[11].fill(-f32::from_bits(1));rows[11][width]=f32::from_bits(1);
    for row in &mut rows {row.rotate_right(rotation%total);}
    rows
}

/// Single GPU: synthetic TP2/3/4 reduction-rule/graph checks only. TP2:
/// actual NCCL synthetic checks, real checkpoint head, changed-input graphs,
/// and ABBA fixed-head-workset timing. TP3/4 networking is not exercised.
pub fn probe(model:&std::path::Path,out:&std::path::Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let tp=crate::tp::init_from_env();crate::tp::set_fp32_accum();
    if tp.world==1 {check_impl(None,out);return;}
    let old=FlagRestore::new(&["GLM53_VOCAB_TP"]);std::env::set_var("GLM53_VOCAB_TP","1");
    let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,0,tch::Device::Cuda(0));
    check(&w,out);drop(old);
}

/// Reuse an already resident model during qualification. This changes no
/// weights or caller precision flags; FP8_HEAD/TOP1 flags are restored on exit.
pub fn check(w:&crate::weights::ModelWeights,out:&std::path::Path) {
    check_impl(Some(w),out)
}

struct FlagRestore(Vec<(&'static str,Option<String>)>);
impl FlagRestore {
    fn new(keys:&[&'static str])->Self {Self(keys.iter().map(|&key|(key,std::env::var(key).ok())).collect())}
}
impl Drop for FlagRestore {
    fn drop(&mut self) {for (key,value) in &self.0 {match value {Some(v)=>std::env::set_var(key,v),None=>std::env::remove_var(key)}}}
}

fn check_impl(weights:Option<&crate::weights::ModelWeights>,out:&std::path::Path) {
    use tch::Device;use serde_json::json;use std::time::Instant;
    let _guard=tch::no_grad_guard();let tp=crate::tp::world();
    std::fs::create_dir_all(out).unwrap();let dev=Device::Cuda(0);
    let _restore=FlagRestore::new(&["GLM53_TARGET_TOP1_TP","GLM53_FP8_HEAD"]);
    let flags:std::collections::BTreeMap<_,_>=std::env::vars().filter(|(k,_)|k.starts_with("GLM53_")||k.starts_with("NCCL_")).collect();
    let dest=out.join(format!("head-select-rank{}.json",tp.rank));
    std::fs::write(&dest,r#"{"complete":false,"gate":false}"#).unwrap();
    let mut cases=Vec::new();
    let worlds:Vec<usize>=if tp.world==1 {vec![2,3,4]} else {vec![tp.world]};
    for world in worlds {for width in [7usize,1024] {
        let total=(world*width) as i64;
        let mut input=Tensor::zeros([12,if tp.world==1{total}else{width as i64}],(Kind::Float,dev));
        let run=|x:&Tensor|if tp.world==1 {simulate(x,world)} else {from_local(x,(tp.rank*width) as i64,total)};
        let _=run(&input);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let mut result=run(&input);crate::tp::graph::end().unwrap();
        let graph=crate::tp::graph::Owned::take();
        for rotation in [0usize,1,width-1,width,width+1,world*width-1,0] {
            let rows=fixtures(world,width,rotation);let flat:Vec<_>=rows.into_iter().flatten().collect();
            let full=Tensor::from_slice(&flat).view([12,total]).to_device(dev);
            let local=if tp.world==1 {full.shallow_clone()} else {full.narrow(1,(tp.rank*width) as i64,width as i64)};
            input.copy_(&local);
            let expected=if tp.world==1 {full.argmax(-1,false)} else {full_from_local(&input,(tp.rank*width) as i64,total)};
            assert!(run(&input).equal(&expected),"top1 eager edge cases world={world} width={width} rotation={rotation}");
            let _=result.fill_(-777);graph.replay();
            assert!(result.equal(&expected),"top1 changed-input graph/canary world={world} width={width} rotation={rotation}");
            cases.push(json!({"kind":"adversarial","world":world,"local_width":width,"rotation":rotation,
                "rows":12,"eager_exact":true,"graph_exact":true,"network":tp.world>1}));
        }
    }}
    let mut timing=Vec::new();
    if tp.world>1 {
        assert_eq!(tp.world,2,"TP3/4 hardware is not qualified");
        let w=weights.expect("TP2 check needs the resident head");assert_eq!(w.device,dev);
        assert!(!crate::root_probe::full_f32(),"FP8 head probe requires normal head precision dispatch");
        assert_eq!(w.lm_head.kind(),Kind::Half,"FP8_HEAD 0/1 check requires registered FP16 head");
        assert!(w.lm_head.is_contiguous()&&w.lm_head.size()[0]>=512&&w.lm_head.size()[1]>=512);
        crate::dense_fp8::register_weight(&w.lm_head,"GLM53_FP8_HEAD");
        std::env::set_var("GLM53_TARGET_TOP1_TP","1");
        let (start,total)=w.vocab_shard.expect("head probe needs vocabulary TP");
        for fp8 in [false,true] {
        std::env::set_var("GLM53_FP8_HEAD",if fp8{"1"}else{"0"});tch::manual_seed(923713);
        for rows in [1i64,2,8,32] {
            let original=Tensor::randn([rows,w.lm_head.size()[1]],(Kind::Float,dev));
            let mut input=original.copy();
            assert_eq!(crate::dense_fp8::try_run(&input,&w.lm_head,false).is_some(),fp8,
                "FP8_HEAD toggle must actually select the registered head cache");
            // Quantize/register any immutable FP8 weight before capture, so
            // graph replay never depends on allocations created inside capture.
            for _ in 0..3 {let _=w.logits(&input);let _=w.predictions(&input);}
            tch::Cuda::synchronize(0);crate::tp::graph::begin().unwrap();let mut result=w.predictions(&input);
            crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
            for scale in [1.,-1.,0.,0.125,1.] {
                input.copy_(&(&original*scale));let expected=w.logits(&input).argmax(-1,false);
                assert!(w.predictions(&input).equal(&expected),"real head eager top1 rows={rows} scale={scale}");
                let _=result.fill_(-777);graph.replay();assert!(result.equal(&expected),"real head graph top1");
                cases.push(json!({"kind":"checkpoint_head","rows":rows,"fp8_head":fp8,"scale":scale,"eager_exact":true,"graph_exact":true}));
            }
            drop(graph);input.copy_(&original);
            // This intentionally repeats the sole model head, which is the
            // production head working set. It is not a multi-layer GEMM claim.
            let local=crate::weights::mm16(&input,&w.lm_head);
            for include_gemm in [false,true] {for candidate in [false,true,true,false] {
                let call=||if include_gemm {
                    if candidate {w.predictions(&input)} else {w.logits(&input).argmax(-1,false)}
                } else if candidate {from_local(&local,start,total)} else {full_from_local(&local,start,total)};
                for _ in 0..3 {let _=call();}tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();let selected=call();crate::tp::graph::end().unwrap();
                let graph=crate::tp::graph::Owned::take();for _ in 0..5 {graph.replay();}
                assert!(selected.equal(&w.logits(&input).argmax(-1,false)));
                tch::Cuda::synchronize(0);let clock=Instant::now();
                for _ in 0..64 {graph.replay();}tch::Cuda::synchronize(0);
                timing.push(json!({"rows":rows,"fp8_head":fp8,"include_gemm":include_gemm,"candidate":candidate,
                    "graph_us":clock.elapsed().as_secs_f64()*1e6/64.,"rounds":64,
                    "full_collective_bytes":rows*total*4,"packet_collective_bytes":tp.world as i64*rows*2*4}));
            }}
        }}
    }
    let binary=std::env::current_exe().unwrap();let digest=std::process::Command::new("sha256sum").arg(&binary).output().unwrap();
    assert!(digest.status.success());
    let binary_sha=String::from_utf8(digest.stdout).unwrap().split_whitespace().next().unwrap().to_owned();
    std::fs::write(&dest,serde_json::to_string_pretty(&json!({"complete":true,"gate":true,"rank":tp.rank,"world":tp.world,
        "binary":binary,"binary_sha256":binary_sha,"flags":flags,"cases":cases,"timing":timing,
        "scope":"TP1 simulates TP2/3/4 rules on one GPU; TP2 additionally tests actual checkpoint head/NCCL; synthetic activations",
        "matrix_compute_unchanged":true,"timing_scope":"fixed real head workset; take slower rank per ABBA arm; end-to-end acceptance pending"})).unwrap()).unwrap();
    eprintln!("[head-select] rank {} PASS exact eager/graph top1, network world {}",tp.rank,tp.world);
}

#[cfg(test)]
mod tests {
    use super::*;
    // Independent scalar oracle: scan global token IDs in ascending order,
    // preserve the first maximum and stop at the first NaN.
    fn scan(values:&[f32])->usize {
        assert!(!values.is_empty());let mut best=0;
        for i in 0..values.len() {
            if values[i].is_nan() {return i;}
            if values[i]>values[best] {best=i;}
        }best
    }
    fn distributed(values:&[f32],world:usize)->usize {
        let width=values.len()/world;assert_eq!(values.len(),world*width);
        let ids:Vec<_>=values.chunks_exact(width).enumerate().map(|(r,x)|r*width+scan(x)).collect();
        ids[scan(&ids.iter().map(|&i|values[i]).collect::<Vec<_>>())]
    }
    #[test] fn tp234_adversarial_exact_rules() {
        for world in [2usize,3,4] {for width in [4usize,7,127] {for rotation in 0..world*width {
            for row in fixtures(world,width,rotation) {assert_eq!(distributed(&row,world),scan(&row));}
        }}}
    }
    #[test] fn pytorch_cpu_matches_packet_rules() {
        for world in [2usize,3,4] {for width in [7usize,1024] {for rotation in [0,1,width-1,width,width+1] {
            let rows=fixtures(world,width,rotation);let expected:Vec<i64>=rows.iter().map(|r|scan(r) as i64).collect();
            let full=Tensor::from_slice(&rows.into_iter().flatten().collect::<Vec<_>>()).view([12,(world*width) as i64]);
            let reference=Tensor::from_slice(&expected);
            assert!(full.argmax(-1,false).equal(&reference));assert!(simulate(&full,world).equal(&reference));
        }}}
    }
    #[test] fn lowest_id_for_nan_inf_and_signed_zero() {
        assert_eq!(scan(&[f32::INFINITY,f32::NAN,f32::NAN]),1);
        assert_eq!(scan(&[f32::NEG_INFINITY,f32::NEG_INFINITY]),0);
        assert_eq!(scan(&[-0.,0.]),0);assert_eq!(scan(&[0.,-0.]),0);
    }
    #[test] fn packet_ids_preserve_exact_float_integer_range() {
        for id in [0i64,1,65535,154879,(1<<24)-1] {assert_eq!((id as f32) as i64,id);}
    }
    #[test] #[should_panic(expected="rank-ordered")]
    fn reject_out_of_order_shards() {
        check_layout(&Tensor::zeros([1,7],(Kind::Float,tch::Device::Cpu)),0,14,1,2);
    }
}
