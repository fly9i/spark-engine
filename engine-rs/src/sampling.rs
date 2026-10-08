//! Temperature sampling by exact speculative sampling for a greedy drafter.
//! Each verifier row adds Gumbel noise T*G to its target logits before the (possibly vocabulary-
//! sharded) argmax, so every "prediction" is a sample from softmax(l/T). A greedy draft token x
//! is accepted iff the sample equals x (probability p(x)); on mismatch the sample is distributed as
//! p restricted to y != x, which is the rejection residual for a deterministic draft. The output
//! distribution is therefore exactly the target's. G is a hash of (request seed, output position,
//! global vocabulary id), so a given seed yields the same text for any draft depth/acceptance.
//! Temperature-0 rows add exact zeros (argmax unchanged).
use std::cell::RefCell;
use tch::{Device,Kind,Tensor};

pub const MAX_ROWS:usize=64;
struct Noise {buf:Tensor,dirty:Vec<bool>,start:i64,width:i64,early:Option<Vec<(f32,u64)>>,
    /// GLM53_GUMBEL_FUSED: per-row temperature [64] f32 and key [64] u64 read by the fused top-1 kernel.
    params:Option<(Tensor,Tensor)>}
thread_local!{static NOISE:RefCell<Option<Noise>>=RefCell::new(None);}

/// Allocate the persistent noise buffer (before any verifier graph is captured).
/// `start,width`: the vocabulary columns this rank's head produces before selection.
pub fn enable(start:i64,width:i64,device:Device) {
    // Both representations exist so graphs captured with and without GLM53_GUMBEL_FUSED can coexist (in-process A/B);
    // the parameters are always written, the [rows, width] buffer only while the flag is off.
    let params=Some((Tensor::zeros([MAX_ROWS as i64],(Kind::Float,device)),Tensor::zeros([MAX_ROWS as i64],(Kind::Int64,device))));
    let buf=Tensor::zeros([MAX_ROWS as i64,width],(Kind::Float,device));
    NOISE.with(|n|*n.borrow_mut()=Some(Noise{buf,dirty:vec![false;MAX_ROWS],start,width,early:None,params}));
}
pub fn enabled()->bool {NOISE.with(|n|n.borrow().is_some())}
pub fn fused_enabled()->bool {std::env::var("GLM53_GUMBEL_FUSED").as_deref()==Ok("1")}
/// Fused mode: (temps, keys, start, width) for the verifier top-1 kernel; None when sampling is off or not fused.
pub(crate) fn fused_params()->Option<(Tensor,Tensor,i64,i64)> {
    if !fused_enabled() {return None;}
    NOISE.with(|n|n.borrow().as_ref().and_then(|n|n.params.as_ref().map(|(t,k)|(t.shallow_clone(),k.shallow_clone(),n.start,n.width))))
}
pub fn key(seed:u64,position:u64)->u64 {
    let mut x=seed.wrapping_mul(0x9e3779b97f4a7c15)^position.wrapping_add(0x632be59bd9b4e019);
    x^=x>>31;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>29;x
}
fn fill(out:&Tensor,col_offset:i64,temps:&[f32],keys:&[u64]) {
    extern "C"{fn rs_gumbel_noise(out:*mut f32,stride:i64,rows:i32,cols:i32,col_offset:i64,temps:*const f32,keys:*const u64)->i32;}
    assert_eq!(out.kind(),Kind::Float);assert_eq!(out.stride()[1],1);assert!(temps.len()<=MAX_ROWS&&temps.len()==keys.len());
    assert_eq!(unsafe{rs_gumbel_noise(out.data_ptr().cast(),out.stride()[0],temps.len() as i32,out.size()[1] as i32,col_offset,temps.as_ptr(),keys.as_ptr())},0,"gumbel noise");
}
/// GLM53_NOISE_EARLY=1: the next round's noise (a pure function of seed, output position and vocabulary id) is filled on
/// a side stream while the drafter runs on the main stream; set_rows then only joins when its rows are a prefix of the
/// early rows (same values, same buffer).
pub fn early_enabled()->bool {std::env::var("GLM53_NOISE_EARLY").as_deref()==Ok("1")}
extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;fn rs_stream_join(n:i32)->i32;}
/// Fill `rows` (a superset: the maximum draft depth + 1) on a forked side stream. The fork orders it after every
/// earlier main-stream reader of the buffer (the previous verifier replay).
pub fn prefill_rows(rows:&[(f32,u64)]) {
    NOISE.with(|n|{
        let mut n=n.borrow_mut();let Some(n)=n.as_mut() else {return};
        if fused_enabled() {return;}
        if n.early.is_some() {assert_eq!(unsafe{rs_stream_join(1)},0);n.early=None;}
        if rows.is_empty() || rows.len()>MAX_ROWS || rows.iter().enumerate().all(|(i,r)|r.0<=0.&&!n.dirty[i]) {return;}
        let temps:Vec<f32>=rows.iter().map(|r|r.0.max(0.)).collect();let keys:Vec<u64>=rows.iter().map(|r|r.1).collect();
        assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);
        fill(&n.buf.narrow(0,0,rows.len() as i64),n.start,&temps,&keys);
        assert_eq!(unsafe{rs_stream_set(-1)},0);
        for (i,t) in temps.iter().enumerate() {n.dirty[i]=*t>0.;}
        n.early=Some(rows.iter().map(|r|(r.0.max(0.),r.1)).collect());
    })
}
/// Set rows 0..rows.len() of the verifier noise: (temperature, key). Skips the launch when every
/// row is greedy and already zero.
pub fn set_rows(rows:&[(f32,u64)]) {
    NOISE.with(|n|{
        let mut n=n.borrow_mut();let Some(n)=n.as_mut() else {assert!(rows.iter().all(|r|r.0<=0.),"sampling not enabled");return};
        if let Some((t,k))=&n.params {
            assert!(rows.len()<=MAX_ROWS,"sampling rows {} > {}",rows.len(),MAX_ROWS);
            let temps:Vec<f32>=rows.iter().map(|r|r.0.max(0.)).collect();let keys:Vec<u64>=rows.iter().map(|r|r.1).collect();
            extern "C"{fn rs_gumbel_params(temps:*mut f32,keys:*mut u64,rows:i32,t:*const f32,k:*const u64)->i32;}
            assert_eq!(unsafe{rs_gumbel_params(t.data_ptr().cast(),k.data_ptr().cast(),rows.len() as i32,temps.as_ptr(),keys.as_ptr())},0,"gumbel params");
            if fused_enabled() {return;}
        }
        if let Some(early)=n.early.take() {
            assert_eq!(unsafe{rs_stream_join(1)},0);
            if rows.len()<=early.len() && rows.iter().zip(&early).all(|(a,b)|a.0.max(0.)==b.0&&a.1==b.1) {return;}
        }
        assert!(rows.len()<=MAX_ROWS,"sampling rows {} > {}",rows.len(),MAX_ROWS);
        if rows.iter().enumerate().all(|(i,r)|r.0<=0.&&!n.dirty[i]) {return;}
        let temps:Vec<f32>=rows.iter().map(|r|r.0.max(0.)).collect();let keys:Vec<u64>=rows.iter().map(|r|r.1).collect();
        fill(&n.buf.narrow(0,0,rows.len() as i64),n.start,&temps,&keys);
        for (i,t) in temps.iter().enumerate() {n.dirty[i]=*t>0.;}
    })
}
/// Head values plus the row noise (columns = this rank's head slice). Used inside captured graphs:
/// the buffer address is fixed.
pub fn perturb(values:&Tensor)->Tensor {
    NOISE.with(|n|match n.borrow().as_ref() {
        Some(n)=>{let r=values.size()[0];assert!(r as usize<=MAX_ROWS);assert_eq!(values.size()[1],n.width);
            assert!(!fused_enabled(),"GLM53_GUMBEL_FUSED: the noise buffer is not maintained; use head_select::from_local_gumbel");
            values+n.buf.narrow(0,0,r)}
        None=>values.shallow_clone(),
    })
}
/// Sample one token from full logits [V] (replicated on every rank): prefill's last row.
pub fn sample_full(logits:&Tensor,temp:f32,key:u64)->i64 {
    if temp<=0. {return logits.argmax(-1,false).int64_value(&[]);}
    let l=logits.to_kind(Kind::Float).reshape([1,-1]);let noise=Tensor::empty_like(&l);
    fill(&noise,0,&[temp],&[key]);(l+noise).argmax(-1,false).int64_value(&[0])
}

/// `sampling-probe`: empirical check of the Gumbel noise against softmax(l/T) on a realistic-width
/// vocabulary (logits with a heavy tail), plus finiteness of the noise. Sharded columns (two halves
/// perturbed separately, then argmax of the maxima) must pick the same ids as the full row.
pub fn probe() {
    let dev=Device::Cuda(0);let _g=tch::no_grad_guard();let v=154880i64;let n=20000u64;
    tch::manual_seed(7);
    // realistic head: a handful of strong candidates above a wide random tail
    let l=Tensor::randn([v],(Kind::Float,dev))*1.5;
    for (i,b) in [(71855i64,9.0),(3,8.6),(120000,8.2),(77440,7.9),(5,7.5),(154000,7.0),(40000,6.0)] {let _=l.get(i).fill_(b);}
    for &t in &[0.7f32,1.0] {
        let probs=(&l/t as f64).softmax(-1,Kind::Float);
        let mut counts=std::collections::HashMap::<i64,u64>::new();let mut shard_mismatch=0;let mut nonfinite=0i64;
        let noise=Tensor::empty([1,v],(Kind::Float,dev));
        for i in 0..n {
            let k=key(1234,i);fill(&noise,0,&[t],&[k]);
            nonfinite+=noise.isfinite().logical_not().sum(Kind::Int64).int64_value(&[]);
            let y=(l.reshape([1,-1])+&noise).argmax(-1,false).int64_value(&[0]);
            *counts.entry(y).or_default()+=1;
            if i<500 {  // shard check: two halves with their own column offsets
                let h=v/2;let a=Tensor::empty([1,h],(Kind::Float,dev));let b=Tensor::empty([1,h],(Kind::Float,dev));
                fill(&a,0,&[t],&[k]);fill(&b,h,&[t],&[k]);
                let pa=l.narrow(0,0,h).reshape([1,-1])+a;let pb=l.narrow(0,h,h).reshape([1,-1])+b;
                let (ma,ia)=pa.max_dim(1,false);let (mb,ib)=pb.max_dim(1,false);
                let ys=if mb.double_value(&[0])>ma.double_value(&[0]) {ib.int64_value(&[0])+h} else {ia.int64_value(&[0])};
                if ys!=y {shard_mismatch+=1;}
            }
        }
        let (top_p,top_i)=probs.topk(8,-1,true,true);let top_i:Vec<i64>=Vec::try_from(top_i.to_device(Device::Cpu)).unwrap();
        let top_p:Vec<f32>=Vec::try_from(top_p.to_device(Device::Cpu)).unwrap();
        let mut chi=0.;let mut covered=0.;
        for (i,p) in top_i.iter().zip(&top_p) {let c=*counts.get(i).unwrap_or(&0) as f64;let e=*p as f64*n as f64;covered+=*p as f64;
            chi+=(c-e)*(c-e)/e.max(1e-9);println!("T={t} id {i:>6} p {:.4} expected {:>8.1} observed {c:>6}",p,e);}
        let rest_obs=n as f64-top_i.iter().map(|i|*counts.get(i).unwrap_or(&0) as f64).sum::<f64>();let rest_exp=(1.-covered)*n as f64;
        chi+=(rest_obs-rest_exp).powi(2)/rest_exp.max(1e-9);
        println!("T={t} rest expected {rest_exp:.1} observed {rest_obs}; chi2(8 dof)={chi:.2} (p=0.001 critical 26.1); nonfinite noise {nonfinite}; shard mismatches {shard_mismatch}/500");
    }
}

/// Coupled drafting (GLM53_DRAFT_COUPLED=1, L2): the drafter's path walk adds, to each of its 16 candidates at
/// draft position t, the very noise T*G(key(seed, g+1+t), id) that the verifier row for that output position adds
/// to the target logits, so draft and target take the Gumbel-max of the same noise (maximal-coupling style
/// agreement at T>0). The target sample depends only on (seed, position, id): outputs are identical with the flag
/// on or off; only acceptance changes. Temperature 0 adds exact zeros.
pub fn coupled()->bool {std::env::var("GLM53_DRAFT_COUPLED").as_deref()==Ok("1")}
thread_local!{static DRAFT_ROWS:RefCell<Vec<(f32,[i64;7])>>=const{RefCell::new(Vec::new())};}
/// Per proposal (in call order: single propose uses entry 0, propose_many entry i): temperature and the 7 keys.
pub fn set_draft_rows(rows:Vec<(f32,[i64;7])>) {DRAFT_ROWS.with(|r|*r.borrow_mut()=rows);}
pub fn clear_draft_rows() {DRAFT_ROWS.with(|r|r.borrow_mut().clear());}
pub fn draft_row(i:usize)->(f32,[i64;7]) {DRAFT_ROWS.with(|r|r.borrow().get(i).copied().unwrap_or((0.,[0;7])))}
/// Keys for draft positions 1..=7 of a round whose anchor sits at generated index g.
pub fn draft_keys(seed:u64,g:u64)->[i64;7] {let mut k=[0i64;7];for t in 0..7 {k[t]=key(seed,g+1+t as u64) as i64;}k}
/// [positions,16] noise at candidate ids (device temps [>=positions] f32, keys [>=positions] i64).
pub fn candidate_noise(ids:&Tensor,temps:&Tensor,keys:&Tensor)->Tensor {
    assert_eq!(ids.kind(),Kind::Int64);assert_eq!(ids.size()[1],16);assert!(ids.is_contiguous());
    let positions=ids.size()[0];let out=Tensor::empty([positions,16],(Kind::Float,ids.device()));
    extern "C"{fn rs_gumbel_candidates(ids:*const i64,temps:*const f32,keys:*const i64,out:*mut f32,positions:i32)->i32;}
    assert_eq!(unsafe{rs_gumbel_candidates(ids.data_ptr().cast(),temps.data_ptr().cast(),keys.data_ptr().cast(),out.data_ptr().cast(),positions as i32)},0,"candidate noise");
    out
}
/// Eager-path noise for proposal `i` (fresh device tensors).
pub fn candidate_noise_row(ids:&Tensor,i:usize)->Tensor {
    let (t,k)=draft_row(i);let dev=ids.device();
    candidate_noise(ids,&Tensor::from_slice(&[t;7]).to_device(dev),&Tensor::from_slice(&k).to_device(dev))
}

/// `gumbel-fused-probe`: GLM53_GUMBEL_FUSED top-1 vs the old noise fill -> values+noise -> argmax on one rank's head slice
/// (width 77440, column offset of rank 1), bitwise on the chosen id; heavy ties, NaN/Inf fixtures, T in {0, 0.7, 1, 1.5};
/// then timing of both.
pub fn fused_probe() {
    let dev=Device::Cuda(0);let _g=tch::no_grad_guard();let width=77440i64;let start=77440i64;
    let t_buf=Tensor::zeros([MAX_ROWS as i64],(Kind::Float,dev));let k_buf=Tensor::zeros([MAX_ROWS as i64],(Kind::Int64,dev));
    let old=|l:&Tensor,temps:&[f32],keys:&[u64]|{let noise=Tensor::empty_like(l);fill(&noise,start,temps,keys);(l+noise).argmax(-1,false)};
    let mut cases=0;let mut bad=0;
    for trial in 0..400u64 {
        let rows=1+(trial%8) as i64;tch::manual_seed(trial as i64);
        let mut l=Tensor::randn([rows,width],(Kind::Float,dev))*(1.+(trial%5) as f64);
        if trial%7==3 {l=l.round();}                                      // many exact ties
        if trial%11==5 {let _=l.narrow(1,0,1000).fill_(f64::from(9.5f32));} // a wide flat top
        if trial%13==6 {let _=l.get(0).get(1234).fill_(f64::NAN);let _=l.get(0).get(99).fill_(f64::NAN);}
        if trial%17==8 {let _=l.get(0).get(4321).fill_(f64::INFINITY);let _=l.get(0).get(4000).fill_(f64::INFINITY);}
        if trial%19==9 {let _=l.fill_(f64::NEG_INFINITY);}
        let l=l.contiguous();
        let temps:Vec<f32>=(0..rows).map(|r|[0f32,0.7,1.0,1.5][((trial+r as u64)%4) as usize]).collect();
        let keys:Vec<u64>=(0..rows).map(|r|key(trial*31+7,r as u64+trial)).collect();
        let a=old(&l.shallow_clone(),&temps,&keys);
        let b=fused_at(&l,start,&temps,&keys,&t_buf,&k_buf);
        cases+=1;if !a.equal(&b) {bad+=1;eprintln!("[gumbel-fused] mismatch trial {trial} rows {rows}: {:?} vs {:?}",Vec::<i64>::try_from(&a).unwrap(),Vec::<i64>::try_from(&b).unwrap());}
    }
    eprintln!("[gumbel-fused] {cases} cases, {bad} mismatching");
    for rows in [1i64,2,4,8] {
        let l=Tensor::randn([rows,width],(Kind::Float,dev))*3.;let temps=vec![1f32;rows as usize];let keys:Vec<u64>=(0..rows as u64).collect();
        let time=|f:&dyn Fn()|{for _ in 0..20{f();}tch::Cuda::synchronize(0);let t=std::time::Instant::now();for _ in 0..300{f();}tch::Cuda::synchronize(0);t.elapsed().as_secs_f64()*1e6/300.};
        let a=time(&||{let _=old(&l,&temps,&keys);});let b=time(&||{let _=fused_at(&l,start,&temps,&keys,&t_buf,&k_buf);});
        eprintln!("[gumbel-fused] rows {rows}: old fill+add+argmax {a:.1} us, fused {b:.1} us");
    }
    assert_eq!(bad,0,"fused Gumbel top-1 differs");eprintln!("[gumbel-fused] PASS");
}
/// Fused top-1 on a slice whose global columns start at `start` (world 1: returns local column ids).
fn fused_at(l:&Tensor,start:i64,temps:&[f32],keys:&[u64],t_buf:&Tensor,k_buf:&Tensor)->Tensor {
    extern "C"{fn rs_gumbel_params(temps:*mut f32,keys:*mut u64,rows:i32,t:*const f32,k:*const u64)->i32;
        fn rs_gumbel_argmax_packet(values:*const f32,rows:i32,width:i32,start:i64,temps:*const f32,keys:*const u64,part_v:*mut f32,part_i:*mut i32,packet:*mut f32,world:i32,rank:i32)->i32;}
    assert_eq!(unsafe{rs_gumbel_params(t_buf.data_ptr().cast(),k_buf.data_ptr().cast(),temps.len() as i32,temps.as_ptr(),keys.as_ptr())},0);
    let rows=l.size()[0];let dev=l.device();
    let pv=Tensor::empty([rows*32],(Kind::Float,dev));let pi=Tensor::empty([rows*32],(Kind::Int,dev));let pk=Tensor::empty([1,rows,2],(Kind::Float,dev));
    assert_eq!(unsafe{rs_gumbel_argmax_packet(l.data_ptr().cast(),rows as i32,l.size()[1] as i32,start,t_buf.data_ptr().cast(),k_buf.data_ptr().cast(),
        pv.data_ptr().cast(),pi.data_ptr().cast(),pk.data_ptr().cast(),1,0)},0);
    pk.select(2,1).get(0).to_kind(Kind::Int64)-start
}
