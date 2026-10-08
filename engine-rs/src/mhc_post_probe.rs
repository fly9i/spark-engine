// SPDX-License-Identifier: MIT
//! Exact old-CUDA arithmetic gates for four-stream and packed-consumer post.
//! Synthetic tensors only; full-model/TP SUM quality remains a separate gate.
use super::{mhc_post,mhc_post_packed,post_cuda_into,PreOut};
use serde_json::json;
use std::{path::Path,time::Instant};
use tch::{Device,Kind,Tensor};

pub(super) struct FlagGuard(Vec<(&'static str,Option<std::ffi::OsString>)>);
impl FlagGuard {
    pub(super) fn new()->Self {
        Self(["GLM53_MHC_POST_FUSED","GLM53_MHC_POST_FOUR_STREAMS","GLM53_MHC_POST_PACKED"]
            .into_iter().map(|key|(key,std::env::var_os(key))).collect())
    }
}
impl Drop for FlagGuard {fn drop(&mut self){for (key,value) in &self.0 {
    if let Some(v)=value {std::env::set_var(key,v);}else {std::env::remove_var(key);}
}}}

fn flags(four:bool,packed:bool) {
    std::env::set_var("GLM53_MHC_POST_FUSED","1");
    std::env::set_var("GLM53_MHC_POST_FOUR_STREAMS",if four{"1"}else{"0"});
    std::env::set_var("GLM53_MHC_POST_PACKED",if packed{"1"}else{"0"});
}

// Finite values (including +/-0), infinities and subnormals compare their bits.
// IEEE NaN payload selection is not the contract; require identical NaN masks
// and report separately whether even payload bits were identical.
fn exact(a:&Tensor,b:&Tensor,label:&str)->bool {
    assert_eq!(a.size(),b.size());
    let bits=a.contiguous().view_dtype(Kind::Int).eq_tensor(&b.contiguous().view_dtype(Kind::Int));
    let both_nan=a.isnan().logical_and(&b.isnan());
    assert!(bits.logical_or(&both_nan).all().int64_value(&[])!=0,"post finite/zero/inf bits or NaN mask changed: {label}");
    bits.all().int64_value(&[])!=0
}

fn pattern(shape:&[i64],mode:usize,turn:usize,salt:usize)->Tensor {
    let count:usize=shape.iter().map(|&v|v as usize).product();
    let extreme=[0.0f32,-0.,1.,-1.,f32::MIN_POSITIVE,-f32::MIN_POSITIVE,
        f32::from_bits(1),f32::from_bits(0x80000001),1e-20,-1e-20,1e20,-1e20,
        f32::MAX/16.,-f32::MAX/16.,65504.,65520.,-65520.,1.00048828125,-1.00048828125,
        f32::INFINITY,f32::NEG_INFINITY];
    let values:Vec<f32>=(0..count).map(|i| {
        let j=i+turn*13+salt;
        match mode {
            0=>((j*37%1024) as f32-512.)/256.,
            1=>extreme[j%extreme.len()],
            2=>if j%19==0 {f32::from_bits(0x7fc12345)}else{((j*17%256) as f32-128.)/64.},
            _=>unreachable!(),
        }
    }).collect();
    Tensor::from_slice(&values).view(shape).to_device(Device::Cuda(0))
}

struct Fixture {
    rows:i64,hidden:i64,
    residual_base:Tensor,residual_write:Tensor,residual:Tensor,
    x_base:Tensor,x:Tensor,
    comb_base:Tensor,post_base:Tensor,pre:PreOut,
    packed:Tensor,
}
impl Fixture {
    fn new(rows:i64,hidden:i64,layout:usize)->Self {
        let opts=(Kind::Float,Device::Cuda(0));
        let residual_base=match layout {
            0=>Tensor::empty([rows,4,hidden],opts),
            1=>Tensor::empty([rows,1,hidden],opts),
            2=>Tensor::empty([rows*2,8,hidden*2+5],opts),
            _=>unreachable!(),
        };
        let residual_write=if layout==2 {residual_base.slice(0,0,rows*2,2).slice(1,0,8,2).slice(2,2,2+hidden*2,2)}else{residual_base.shallow_clone()};
        let residual=if layout==1 {residual_write.expand([rows,4,hidden],false)}else{residual_write.shallow_clone()};
        let x_base=Tensor::empty([rows,hidden*2+3],opts);
        let x=x_base.slice(1,1,1+hidden*2,2);
        let comb_base=Tensor::empty([rows,4,8],opts);let comb=comb_base.slice(2,0,8,2);
        let post_base=Tensor::empty([rows,8],opts);let post_mix=post_base.slice(1,1,8,2);
        Self{rows,hidden,residual_base,residual_write,residual,x_base,x,comb_base,post_base,
            pre:PreOut{comb,post_mix},packed:Tensor::empty([rows*2,hidden],opts)}
    }
    fn fill(&mut self,mode:usize,turn:usize) {
        for b in [&self.residual_base,&self.x_base,&self.comb_base,&self.post_base] {let _=b.shallow_clone().fill_(f64::NAN);}
        self.residual_write.copy_(&pattern(&self.residual_write.size(),mode,turn,0));
        self.x.copy_(&pattern(&self.x.size(),mode,turn,2));
        // Nontrivial coefficients distinguish all output streams. Turn 4 also
        // moves NaNs into coefficients, not merely branch/residual data.
        let coeff_mode=if mode==2 && turn==4 {2}else{0};
        self.pre.comb.copy_(&pattern(&self.pre.comb.size(),coeff_mode,turn,7));
        self.pre.post_mix.copy_(&pattern(&self.pre.post_mix.size(),coeff_mode,turn,11));
        self.packed.narrow(0,0,self.rows).copy_(&pattern(&[self.rows,self.hidden],mode,turn,23));
        self.packed.narrow(0,self.rows,self.rows).copy_(&pattern(&[self.rows,self.hidden],mode,turn,31));
    }
    fn branch(&self,round:bool)->Tensor {
        let shared=self.packed.narrow(0,self.rows,self.rows);
        let shared=if round {shared.to_kind(Kind::Half).to_kind(Kind::Float)}else{shared};
        self.packed.narrow(0,0,self.rows)+shared
    }
    fn old(&self,packed:bool,round:bool)->Tensor {
        flags(false,false);
        mhc_post(&if packed {self.branch(round)}else{self.x.shallow_clone()},&self.residual,&self.pre)
    }
    fn public(&self,packed:bool,round:bool,four:bool)->Tensor {
        flags(four,packed);
        if packed {mhc_post_packed(&self.packed,&self.residual,&self.pre,round)}else{mhc_post(&self.x,&self.residual,&self.pre)}
    }
    fn write_output(&self,y:&Tensor,packed:bool,round:bool,four:bool) {
        let x=if packed {self.packed.shallow_clone()}else{self.x.contiguous()};
        let comb=self.pre.comb.contiguous();let post=self.pre.post_mix.contiguous();
        post_cuda_into(&x,&self.residual,&comb,&post,y,packed,round,four);
    }
    fn snapshot(&self)->Vec<Tensor> {
        [&self.residual_base,&self.x_base,&self.comb_base,&self.post_base,&self.packed].into_iter().map(Tensor::copy).collect()
    }
    fn unchanged(&self,before:&[Tensor]) {
        for (now,old) in [&self.residual_base,&self.x_base,&self.comb_base,&self.post_base,&self.packed].into_iter().zip(before) {
            assert!(now.view_dtype(Kind::Int).equal(&old.view_dtype(Kind::Int)),"post modified an input or its padding");
        }
    }
}

fn check_case(rows:i64,hidden:i64,layout:usize)->serde_json::Value {
    let mut f=Fixture::new(rows,hidden,layout);f.fill(0,0);
    let n=rows*4*hidden;let guard=128;
    let storage=Tensor::empty([n+guard*2],(Kind::Float,Device::Cuda(0)));
    let output=storage.narrow(0,guard,n).view([rows,4,hidden]);
    let sentinel=713.0f64;
    let mut all_payload_bits=true;let mut checks=0;
    // Packed off/on is independent of the four-stream output mapping.
    for (packed,round,four) in [(false,false,true),(true,false,false),(true,false,true),(true,true,false),(true,true,true)] {
        let old=f.old(packed,round);let actual=f.public(packed,round,four);
        all_payload_bits&=exact(&actual,&old,"public entry");
        // Warm contiguous-copy/kernel paths before capture, then use the same
        // output allocation for NaN and bounds canaries on every replay.
        f.write_output(&output,packed,round,four);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();f.write_output(&output,packed,round,four);crate::tp::graph::end().unwrap();
        for turn in 0..5 {
            let mode=match turn {0|3=>0,1=>1,_=>2};f.fill(mode,turn);
            let before=f.snapshot();let _=storage.shallow_clone().fill_(sentinel);let _=output.shallow_clone().fill_(f64::NAN);
            crate::tp::graph::replay().unwrap();let reference=f.old(packed,round);
            all_payload_bits&=exact(&output,&reference,"changed graph inputs");
            if mode==0 {assert!(output.isfinite().all().int64_value(&[])!=0,"NaN padding or stale output escaped");}
            assert!(storage.narrow(0,0,guard).eq(sentinel).all().int64_value(&[])!=0,"left output guard overwritten");
            assert!(storage.narrow(0,guard+n,guard).eq(sentinel).all().int64_value(&[])!=0,"right output guard overwritten");
            f.unchanged(&before);checks+=1;
        }
        crate::tp::graph::destroy();f.fill(0,0);
    }
    json!({"rows":rows,"hidden":hidden,"layout":(["contiguous","broadcast_stream","strided_all_dims_offset"][layout]),
        "graph_checks":checks,"non_nan_bits_exact":true,"nan_masks_exact":true,"all_nan_payload_bits_exact":all_payload_bits,
        "output_guards":true,"nan_padding_ignored":true,"inputs_unchanged":true,"captured_input_and_coefficients_changed":true})
}

fn timing(rows:i64)->serde_json::Value {
    let hidden=4096;let per_set=(rows*hidden*(4+2+4)*4) as usize;
    // At least 64 MiB logical rotating tensors; do not repeatedly hit one tiny
    // residual in cache. Includes input, output and packed lanes, not weights.
    let sets=((64*1024*1024+per_set-1)/per_set).clamp(32,512);
    let fixtures:Vec<_>=(0..sets).map(|s| {
        let mut f=Fixture::new(rows,hidden,0);f.fill(0,s);
        // Numeric fixtures exercise strided coefficient copies. Performance
        // uses production's contiguous comb/post, packed before graph capture.
        f.pre.comb=f.pre.comb.contiguous();f.pre.post_mix=f.pre.post_mix.contiguous();f
    }).collect();
    let mut rounds=Vec::new();
    for (target_packed,target_four) in [(false,true),(true,false),(true,true)] {
        for candidate in [false,true,true,false] {
            let (packed,four)=if candidate {(target_packed,target_four)}else{(false,false)};
            flags(four,packed);
            let run=|f:&Fixture|if packed {mhc_post_packed(&f.packed,&f.residual,&f.pre,true)}else{mhc_post(&f.branch(true),&f.residual,&f.pre)};
            for f in &fixtures {let _=run(f);}tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let outputs:Vec<_>=fixtures.iter().map(run).collect();crate::tp::graph::end().unwrap();
            for _ in 0..3 {crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            let mut samples=Vec::new();for _ in 0..5 {
                let begin=Instant::now();for _ in 0..4 {crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
                samples.push(begin.elapsed().as_secs_f64()*1e6/(4*sets) as f64);
            }
            crate::tp::graph::destroy();drop(outputs);
            rounds.push(json!({"comparison_packed":target_packed,"comparison_four":target_four,"candidate":candidate,
                "packed":packed,"four_streams":four,"us_per_layer":samples}));
        }
    }
    json!({"rows":rows,"hidden":hidden,"sets":sets,"logical_input_output_bytes_lower_bound":per_set*sets,
        "scope":"shared Half round + routed add + post; synthetic rotating tensors, no TP collective","rounds":rounds})
}

pub(super) fn run(out:&Path) {
    let _guard=FlagGuard::new();let _no_grad=tch::no_grad_guard();let mut cases=Vec::new();
    // Deliberately interleave larger/smaller graph extents, with non-warp/tile
    // tails as well as the real H4096. Each shape owns its graph and output.
    for rows in [32,1,8,2,31,3] {for hidden in [1,33,511,4096] {for layout in 0..3 {
        cases.push(check_case(rows,hidden,layout));
        std::fs::write(out.join("post-four-numeric.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
    }}}
    for layout in [0,1] {cases.push(check_case(2048,4096,layout));}
    std::fs::write(out.join("post-four-numeric.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
    let mut times=Vec::new();for rows in [1,8,32] {
        times.push(timing(rows));std::fs::write(out.join("post-four-timing.json"),serde_json::to_string_pretty(&times).unwrap()).unwrap();
    }
    eprintln!("[mhc-post-four] PASS {} shapes/layouts; finite bits, NaN masks, strided/changed-input graphs, canaries; timing arms complete",cases.len());
}
