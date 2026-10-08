// SPDX-License-Identifier: MIT
//! MHC(流形约束超连接)tch 实现。语义与 engine/glm53/mhc.py 逐行对齐(M0 验收)。
//! 张量形态:residual [T,4,4096](流 = 全宽 hidden)。

use tch::{Kind, Tensor};

const HC_EPS: f64 = 1e-6;
const POST_MULT: f64 = 2.0;
const SINKHORN_REPEAT: usize = 20;
const RMS_EPS: f64 = 1e-5;

pub(crate) fn fused_enabled() -> bool {
    std::env::var("GLM53_MHC_FUSED").as_deref() == Ok("1")
}

pub(crate) fn pre_fused_enabled()->bool {std::env::var("GLM53_MHC_PRE_FUSED").as_deref()==Ok("1")}
/// Prefill rows (>16): split-TF32 partials + the same finish kernel (L1 vs ATen FP32).
pub(crate) fn pre_large_enabled()->bool {std::env::var("GLM53_MHC_PRE_LARGE").as_deref()==Ok("1")}

pub(crate) fn sinkhorn(mut comb: Tensor, fused: bool) -> Tensor {
    if fused && comb.device().is_cuda() {
        assert_eq!(comb.kind(), Kind::Float);
        assert_eq!(&comb.size()[1..], &[4,4]);
        let input = comb.contiguous();
        let output = Tensor::empty_like(&input);
        extern "C" { fn rs_mhc_sinkhorn(input: *const f32, output: *mut f32, rows: i32) -> i32; }
        assert_eq!(unsafe { rs_mhc_sinkhorn(input.data_ptr() as *const f32,
            output.data_ptr() as *mut f32, input.size()[0] as i32) }, 0, "Sinkhorn CUDA launch");
        return output;
    }
    comb = &comb / (&comb.sum_dim_intlist(&[-2i64][..], true, Kind::Float) + HC_EPS);
    for _ in 0..SINKHORN_REPEAT - 1 {
        comb = &comb / (&comb.sum_dim_intlist(&[-1i64][..], true, Kind::Float) + HC_EPS);
        comb = &comb / (&comb.sum_dim_intlist(&[-2i64][..], true, Kind::Float) + HC_EPS);
    }
    comb
}

pub struct PreOut {
    pub post_mix: Tensor, // [T,4]
    pub comb: Tensor,     // [T,4,4]
}

/// 展开:embed [T,H] → [T,4,H](广播复制)。
pub fn hc_expand(x: &Tensor) -> Tensor {
    x.unsqueeze(1).expand([x.size()[0], 4, x.size()[1]], false)
}

/// 收缩:最终 head = 流均值。
pub fn hc_contract(s: &Tensor) -> Tensor {
    s.mean_dim(&[1i64][..], false, Kind::Float)
}

fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Tensor {
    let v = x.to_kind(Kind::Float);
    let sq = &v * &v;
    let ms = sq.mean_dim(&[-1i64][..], true, Kind::Float);
    w.to_kind(Kind::Float) * (v * (ms + eps).rsqrt())
}

/// mhc_pre:residual → (post, comb, z)。fn/scale/base/ln 按站点(attn/ffn)传入。
pub fn mhc_pre(
    residual: &Tensor,
    fn_t: &Tensor,
    scale_t: &Tensor,
    base_t: &Tensor,
    ln_t: &Tensor,
) -> (PreOut, Tensor) {
    let (t, n, h) = (residual.size()[0], residual.size()[1], residual.size()[2]);
    if pre_fused_enabled() && residual.device().is_cuda() && n==4 && ((1..=16).contains(&t) || (t>16 && pre_large_enabled())) && residual.kind()==Kind::Float
        && residual.is_contiguous() && [scale_t,base_t,ln_t].iter().all(|w|w.kind()==Kind::Float&&w.is_contiguous())
        && [Kind::Float,Kind::BFloat16].contains(&fn_t.kind()) && fn_t.is_contiguous()
        && fn_t.size()==[24,n*h] && ln_t.size()==[h] {
        // W05: two kernels replace ~30 ATen launches; mixes/sq-sum reduction order changes (L1).
        let dev=residual.device();
        let partial=Tensor::empty([(n*h/128)*t*25],(Kind::Float,dev));
        let z=Tensor::empty([t,h],(Kind::Float,dev));let post_mix=Tensor::empty([t,n],(Kind::Float,dev));
        let comb=Tensor::empty([t,n,n],(Kind::Float,dev));
        extern "C"{fn rs_mhc_pre_fused(residual:*const f32,fn_:*const std::ffi::c_void,fn_bf16:i32,scale:*const f32,base:*const f32,ln:*const f32,partial:*mut f32,z:*mut f32,post:*mut f32,comb:*mut f32,rows:i32,h:i32)->i32;}
        let p=|x:&Tensor|x.data_ptr() as *const f32;let q=|x:&Tensor|x.data_ptr() as *mut f32;
        assert_eq!(unsafe{rs_mhc_pre_fused(p(residual),fn_t.data_ptr(),i32::from(fn_t.kind()==Kind::BFloat16),p(scale_t),p(base_t),p(ln_t),q(&partial),q(&z),q(&post_mix),q(&comb),t as i32,h as i32)},0,"fused mhc_pre launch");
        return (PreOut{post_mix,comb},z);
    }
    let x = residual.reshape([t, n * h]).to_kind(Kind::Float);
    // M4: BF16-resident fn widens exactly (fallback / non-fused paths only).
    let mixes_raw = x.matmul(&fn_t.to_kind(Kind::Float).transpose(0, 1));
    // 无权重 RMS(先归一再线性 = 后归一,线性齐次)
    let xsq = &x * &x;
    let sqrsum = xsq.sum_dim_intlist(&[1i64][..], true, Kind::Float);
    let rstd = (sqrsum / (n * h) as f64 + RMS_EPS).rsqrt();
    let mixes = mixes_raw * rstd;

    let pre = (&mixes.slice(1, 0, n, 1) * &scale_t.get(0) + base_t.slice(0, 0, n, 1))
        .sigmoid()
        + HC_EPS;
    let post_mix = (&mixes.slice(1, n, 2 * n, 1) * &scale_t.get(1) + base_t.slice(0, n, 2 * n, 1))
        .sigmoid()
        * POST_MULT;
    let comb_raw = mixes
        .slice(1, 2 * n, 2 * n + n * n, 1)
        .reshape([t, n, n])
        * scale_t.get(2)
        + base_t
            .slice(0, 2 * n, 2 * n + n * n, 1)
            .reshape([1, n, n]);
    let comb = sinkhorn(comb_raw.softmax(-1, Kind::Float) + HC_EPS, fused_enabled());
    let collapsed =
        (pre.unsqueeze(-1) * residual.to_kind(Kind::Float)).sum_dim_intlist(&[1i64][..], false, Kind::Float);
    let z = rmsnorm(&collapsed, ln_t, RMS_EPS);
    
    (PreOut { post_mix, comb }, z)
}

/// mhc_post:新残差 = combᵀ·residual + post ⊙ x。x [T,H]。
pub fn mhc_post(x: &Tensor, residual: &Tensor, pre: &PreOut) -> Tensor {
    if x.device().is_cuda() && std::env::var("GLM53_MHC_POST_FUSED").as_deref()==Ok("1") {
        assert_eq!(residual.size()[1],4);assert_eq!(x.size(),[residual.size()[0],residual.size()[2]]);
        for t in [x,residual,&pre.comb,&pre.post_mix]{assert_eq!(t.kind(),Kind::Float);}
        let x=x.contiguous();let comb=pre.comb.contiguous();let post=pre.post_mix.contiguous();
        let y=Tensor::empty(residual.size().as_slice(),(Kind::Float,x.device()));
        post_cuda_into(&x,residual,&comb,&post,&y,false,false,four_streams_enabled());return y;
    }

    let mixed = pre
        .comb
        .transpose(-2, -1)
        .matmul(&residual.to_kind(Kind::Float));
    pre.post_mix.unsqueeze(-1) * x.to_kind(Kind::Float).unsqueeze(-2) + mixed
}

/// I3 step 2: fused four-stream post of a pending TP sum (`tp::PendingSum`, x = this rank's partial); any other shape
/// or flag combination materializes the sum and takes `mhc_post` (same result either way).
pub fn mhc_post_pending(x:&Tensor,pending:Option<crate::tp::PendingSum>,residual:&Tensor,pre:&PreOut)->Tensor {
    let Some(p)=pending else {return mhc_post(x,residual,pre)};
    let ok=x.device().is_cuda() && std::env::var("GLM53_MHC_POST_FUSED").as_deref()==Ok("1") && four_streams_enabled()
        && x.is_contiguous() && x.kind()==Kind::Float && residual.size().len()==3 && residual.size()[1]==4
        && x.size()==[residual.size()[0],residual.size()[2]] && x.data_ptr() as usize==p.ptr && x.numel() as i64==p.numel;
    if !ok {crate::tp::materialize(p);return mhc_post(x,residual,pre);}
    post_ar_into(x,residual,pre,false,p.round_half)
}
/// The packed MoE lanes [routed, shared] after `tp::allreduce_send`: same as mhc_post_packed on the reduced lanes.
pub(crate) fn mhc_post_packed_pending(packed:&Tensor,residual:&Tensor,pre:&PreOut,round_shared:bool)->Tensor {
    assert!(packed_enabled() && four_streams_enabled(),"fused allreduce post requires the packed four-stream post");
    assert_eq!(packed.size(),[2*residual.size()[0],residual.size()[2]]);
    post_ar_into(packed,residual,pre,true,round_shared)
}
fn post_ar_into(x:&Tensor,residual:&Tensor,pre:&PreOut,packed:bool,round:bool)->Tensor {
    let (rows,hidden)=(residual.size()[0],residual.size()[2]);let stride=residual.stride();
    assert!(x.is_contiguous() && x.kind()==Kind::Float && x.device().is_cuda());
    for t in [residual,&pre.comb,&pre.post_mix] {assert_eq!(t.kind(),Kind::Float);assert_eq!(t.device(),x.device());}
    let comb=pre.comb.contiguous();let post=pre.post_mix.contiguous();
    assert_eq!(comb.size(),[rows,4,4]);assert_eq!(post.size(),[rows,4]);
    let y=Tensor::empty(residual.size().as_slice(),(Kind::Float,x.device()));
    extern "C" {fn rs_mhc_post_four_ar(x:*const f32,residual:*const f32,comb:*const f32,post:*const f32,out:*mut f32,
        rows:i32,hidden:i32,s0:i64,s1:i64,s2:i64,packed:i32,round:i32)->i32;}
    let p=|t:&Tensor|t.data_ptr() as *const f32;
    assert_eq!(unsafe{rs_mhc_post_four_ar(p(x),p(residual),p(&comb),p(&post),y.data_ptr().cast(),rows as i32,hidden as i32,
        stride[0],stride[1],stride[2],i32::from(packed),i32::from(round))},0,"fused allreduce MHC post");
    y
}

fn post_flag(name:&str)->bool {
    match std::env::var(name) {
        Ok(value)=>match value.as_str() {"1"=>true,"0"=>false,_=>panic!("{name} must be 0 or 1")},
        Err(std::env::VarError::NotPresent)=>false,
        Err(_)=>panic!("{name} must be 0 or 1"),
    }
}

pub(crate) fn four_streams_enabled()->bool {post_flag("GLM53_MHC_POST_FOUR_STREAMS")}

/// A separate opt-in: callers leave the existing finish_tp -> mhc_post path
/// untouched when this is false or the original TP-pack shape is ineligible.
pub(crate) fn packed_enabled()->bool {
    std::env::var("GLM53_MHC_POST_FUSED").as_deref()==Ok("1") && post_flag("GLM53_MHC_POST_PACKED")
}

/// packed holds two already globally reduced FP32 lanes [routed, shared].
/// round_shared is the original round_row_output precision decision; it cannot
/// be inferred from packed's FP32 dtype. The output never aliases residual.
pub(crate) fn mhc_post_packed(packed:&Tensor,residual:&Tensor,pre:&PreOut,round_shared:bool)->Tensor {
    assert!(packed_enabled(),"packed post requires explicit opt-in and fused post");
    assert!(packed.device().is_cuda());
    assert_eq!(residual.size().len(),3);assert_eq!(residual.size()[1],4);
    let (rows,hidden)=(residual.size()[0],residual.size()[2]);
    assert_eq!(packed.size(),[2*rows,hidden]);assert!(packed.is_contiguous(),"packed post must consume existing contiguous lanes");
    for t in [packed,residual,&pre.comb,&pre.post_mix] {assert_eq!(t.kind(),Kind::Float);assert_eq!(t.device(),packed.device());}
    let comb=pre.comb.contiguous();let post=pre.post_mix.contiguous();
    let y=Tensor::empty(residual.size().as_slice(),(Kind::Float,packed.device()));
    post_cuda_into(packed,residual,&comb,&post,&y,true,round_shared,four_streams_enabled());y
}

// Caller-supplied output is private to this module/probe: graph canary tests
// exercise the exact kernel without adding a post-launch copy to the candidate.
fn post_cuda_into(x:&Tensor,residual:&Tensor,comb:&Tensor,post:&Tensor,y:&Tensor,packed:bool,round_shared:bool,four:bool) {
    let (rows,hidden)=(residual.size()[0],residual.size()[2]);let stride=residual.stride();
    assert!(rows>0 && hidden>0 && rows.checked_mul(hidden).and_then(|v|v.checked_mul(4)).is_some_and(|v|v<=i32::MAX as i64));
    assert_eq!(residual.size(),[rows,4,hidden]);assert_eq!(y.size(),[rows,4,hidden]);
    assert_eq!(x.size(),[rows*if packed {2}else{1},hidden]);
    assert_eq!(comb.size(),[rows,4,4]);assert_eq!(post.size(),[rows,4]);
    for t in [x,residual,comb,post,y] {assert_eq!(t.kind(),Kind::Float);assert_eq!(t.device(),x.device());}
    for t in [x,comb,post,y] {assert!(t.is_contiguous());}
    assert_ne!(y.data_ptr(),residual.data_ptr(),"post output cannot overwrite residual");
    extern "C" {
        fn rs_mhc_post(x:*const f32,residual:*const f32,comb:*const f32,post:*const f32,out:*mut f32,
            rows:i32,hidden:i32,s0:i64,s1:i64,s2:i64)->i32;
        fn rs_mhc_post_four(x:*const f32,residual:*const f32,comb:*const f32,post:*const f32,out:*mut f32,
            rows:i32,hidden:i32,s0:i64,s1:i64,s2:i64)->i32;
        fn rs_mhc_post_packed(x:*const f32,residual:*const f32,comb:*const f32,post:*const f32,out:*mut f32,
            rows:i32,hidden:i32,s0:i64,s1:i64,s2:i64,round_shared:i32,four_streams:i32)->i32;
    }
    let p=|t:&Tensor|t.data_ptr() as *const f32;
    let rc=unsafe {
        if packed {rs_mhc_post_packed(p(x),p(residual),p(comb),p(post),y.data_ptr().cast(),rows as i32,hidden as i32,
            stride[0],stride[1],stride[2],i32::from(round_shared),i32::from(four))}
        else if four {rs_mhc_post_four(p(x),p(residual),p(comb),p(post),y.data_ptr().cast(),rows as i32,hidden as i32,
            stride[0],stride[1],stride[2])}
        else {rs_mhc_post(p(x),p(residual),p(comb),p(post),y.data_ptr().cast(),rows as i32,hidden as i32,
            stride[0],stride[1],stride[2])}
    };
    assert_eq!(rc,0,"MHC post CUDA launch");
}

#[path="mhc_post_probe.rs"]
mod post_probe_detail;

pub fn post_probe(out:&std::path::Path) {
    use tch::Device;use std::time::Instant;use serde_json::json;
    let _flags=post_probe_detail::FlagGuard::new();
    std::env::set_var("GLM53_MHC_POST_FOUR_STREAMS","0");std::env::set_var("GLM53_MHC_POST_PACKED","0");
    let _guard=tch::no_grad_guard();let dev=Device::Cuda(0);let mut cases=Vec::new();tch::manual_seed(2491);
    for rows in [1,2,8,32,2048] {for broadcast in [false,true] {
        let x=Tensor::randn([rows,4096],(Kind::Float,dev));
        let residual=if broadcast{hc_expand(&x)}else{Tensor::randn([rows,4,4096],(Kind::Float,dev))};
        let original=residual.copy();let mut input=residual.shallow_clone();
        let pre=PreOut{comb:Tensor::rand([rows,4,4],(Kind::Float,dev)).softmax(-1,Kind::Float),post_mix:Tensor::rand([rows,4],(Kind::Float,dev))};
        std::env::set_var("GLM53_MHC_POST_FUSED","0");let reference=mhc_post(&x,&input,&pre);
        std::env::set_var("GLM53_MHC_POST_FUSED","1");let actual=mhc_post(&x,&input,&pre);
        let max_abs=(&actual-&reference).abs().max().double_value(&[]);
        assert!(max_abs<2e-6,"MHC post {rows}/{broadcast}: {max_abs}");
        let oracle=pre.comb.to_kind(Kind::Double).transpose(-2,-1).matmul(&input.to_kind(Kind::Double))+
            pre.post_mix.to_kind(Kind::Double).unsqueeze(-1)*x.to_kind(Kind::Double).unsqueeze(1);
        let oracle_error=(actual.to_kind(Kind::Double)-oracle).abs().max().double_value(&[]);assert!(oracle_error<2e-6);
        let mut rounds=Vec::new();
        for on in [false,true,true,false] {
            std::env::set_var("GLM53_MHC_POST_FUSED",if on{"1"}else{"0"});
            for _ in 0..3{let _=mhc_post(&x,&input,&pre);}tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let y=mhc_post(&x,&input,&pre);crate::tp::graph::end().unwrap();
            if !broadcast {for z in [&original,&(-&original),&Tensor::zeros_like(&original)] {
                input.copy_(z);crate::tp::graph::replay().unwrap();assert!(y.equal(&mhc_post(&x,z,&pre)));
            }input.copy_(&original);}else{crate::tp::graph::replay().unwrap();assert!(y.equal(&mhc_post(&x,&input,&pre)));}
            tch::Cuda::synchronize(0);let begin=Instant::now();for _ in 0..64{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"fused":on,"graph_us":begin.elapsed().as_secs_f64()*1e6/64.}));crate::tp::graph::destroy();
        }
        cases.push(json!({"rows":rows,"broadcast":broadcast,"max_abs":max_abs,"oracle_max_abs":oracle_error,"rounds":rounds}));
    }}
    std::fs::create_dir_all(out).unwrap();std::fs::write(out.join("post.json"),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
    post_probe_detail::run(out);
}

/// W05 local gate: fused mhc_pre vs the ATen path on real hc weights, rows 1..16,
/// plus changed-input CUDA Graph replay. Diagnostic only.
pub fn pre_probe(model:&std::path::Path,out:&std::path::Path) {
    use serde_json::json;
    let _g=tch::no_grad_guard();let dev=tch::Device::Cuda(0);std::fs::create_dir_all(out).unwrap();
    let mut idx=crate::safetensors::ShardIndex::scan(model).unwrap();let mut cases=Vec::new();
    let get=|idx:&mut crate::safetensors::ShardIndex,n:&str|{let (v,s)=idx.get_f32(n).unwrap();
        Tensor::from_slice(&v).view(s.iter().map(|&x|x as i64).collect::<Vec<_>>().as_slice()).to_device(dev)};
    let set=|on:bool|std::env::set_var("GLM53_MHC_PRE_FUSED",if on{"1"}else{"0"});
    for layer in [0i64,3,15,30,44] {
        for site in ["attn","ffn"] {
            let p=format!("model.language_model.layers.{layer}");
            let f=get(&mut idx,&format!("{p}.hc_{site}_fn"));let sc=get(&mut idx,&format!("{p}.hc_{site}_scale"));
            let b=get(&mut idx,&format!("{p}.hc_{site}_base"));
            let ln=get(&mut idx,&format!("{p}.{}.weight",if site=="attn"{"input_layernorm"}else{"post_attention_layernorm"}));
            std::env::set_var("GLM53_MHC_PRE_LARGE","1");
            for rows in [1i64,2,3,5,8,9,16,17,64,2048] {
                tch::manual_seed(900+rows+layer);
                for scale in [0.05f64,1.,20.] {
                    let r=(Tensor::randn([rows,4,4096],(Kind::Float,dev))*scale).contiguous();
                    set(false);let (a,za)=mhc_pre(&r,&f,&sc,&b,&ln);
                    set(true);let (c,zc)=mhc_pre(&r,&f,&sc,&b,&ln);
                    let d=|x:&Tensor,y:&Tensor|f64::try_from((x-y).abs().max()).unwrap();
                    let rel=|x:&Tensor,y:&Tensor|f64::try_from((x-y).norm()/y.norm().clamp_min(1e-30)).unwrap();
                    let (dz,dp,dc)=(d(&zc,&za),d(&c.post_mix,&a.post_mix),d(&c.comb,&a.comb));
                    let tol=if std::env::var("GLM53_MHC_PRE_TC").as_deref()==Ok("1"){1e-3}else{1e-4};
                    assert!(zc.isfinite().all().int64_value(&[])!=0 && rel(&zc,&za)<tol && dp<tol && dc<tol,"mhc pre fused mismatch {layer}/{site}/{rows}/{scale}: {dz} {dp} {dc}");
                    cases.push(json!({"layer":layer,"site":site,"rows":rows,"scale":scale,"z_max_abs":dz,"z_rel":rel(&zc,&za),"post_max_abs":dp,"comb_max_abs":dc}));
                }
                // graph replay with changed inputs (candidate path)
                if rows>64{continue;}
                set(true);let mut input=Tensor::randn([rows,4,4096],(Kind::Float,dev));let orig=input.copy();
                let _=mhc_pre(&input,&f,&sc,&b,&ln);tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();let (gp,gz)=mhc_pre(&input,&f,&sc,&b,&ln);crate::tp::graph::end().unwrap();
                for k in [0.5f64,-2.,0.] {
                    input.copy_(&(&orig*k));crate::tp::graph::replay().unwrap();let (ep,ez)=mhc_pre(&input,&f,&sc,&b,&ln);
                    assert!(gz.equal(&ez)&&gp.comb.equal(&ep.comb)&&gp.post_mix.equal(&ep.post_mix),"graph replay changed input");
                }
                crate::tp::graph::destroy();
            }
            eprintln!("[mhc-pre] layer {layer} {site} ok");
        }
    }
    set(false);
    // timing: 90 calls (one graph) fused vs ATen
    let f=get(&mut idx,"model.language_model.layers.3.hc_attn_fn");let sc=get(&mut idx,"model.language_model.layers.3.hc_attn_scale");
    let b=get(&mut idx,"model.language_model.layers.3.hc_attn_base");let ln=get(&mut idx,"model.language_model.layers.3.input_layernorm.weight");
    let mut timing=Vec::new();
    for rows in [1i64,8] {
        let r=Tensor::randn([rows,4,4096],(Kind::Float,dev));
        for on in [false,true,true,false] {
            set(on);for _ in 0..3{let _=mhc_pre(&r,&f,&sc,&b,&ln);}tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();for _ in 0..90{let _=mhc_pre(&r,&f,&sc,&b,&ln);}crate::tp::graph::end().unwrap();
            for _ in 0..3{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            let t0=std::time::Instant::now();for _ in 0..10{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            timing.push(json!({"rows":rows,"fused":on,"us_per_call":t0.elapsed().as_secs_f64()*1e6/900.}));crate::tp::graph::destroy();
        }
    }
    for on in [false,true,true,false] {
        let r=Tensor::randn([2048,4,4096],(Kind::Float,dev));set(on);
        for _ in 0..2{let _=mhc_pre(&r,&f,&sc,&b,&ln);}tch::Cuda::synchronize(0);
        let t0=std::time::Instant::now();for _ in 0..20{let _=mhc_pre(&r,&f,&sc,&b,&ln);}tch::Cuda::synchronize(0);
        timing.push(json!({"rows":2048,"fused":on,"eager":true,"us_per_call":t0.elapsed().as_secs_f64()*1e6/20.}));
    }
    set(false);
    std::fs::write(out.join("mhc-pre-probe.json"),serde_json::to_string_pretty(&json!({"gate":true,"cases":cases,"timing":timing})).unwrap()).unwrap();
    eprintln!("[mhc-pre] PASS {} cases",cases.len());
}

/// P1d (GLM53_MHC_POST_PRE_FUSED=1): residual' = mhc_post(x, residual, prev) and the next
/// mhc_pre(residual', ...) in two kernels instead of three passes over residual'. Bitwise equal
/// to the separate fused post (four streams, non-packed) + large-row fused pre. Falls back to
/// the separate calls when any precondition does not hold.
pub fn mhc_post_pre(x:&Tensor,residual:&Tensor,prev:&PreOut,fn_t:&Tensor,scale_t:&Tensor,base_t:&Tensor,ln_t:&Tensor)->(Tensor,PreOut,Tensor) {
    if !post_pre_eligible(x,residual,fn_t,scale_t,base_t,ln_t) {
        let r=mhc_post(x,residual,prev);let (p,z)=mhc_pre(&r,fn_t,scale_t,base_t,ln_t);return (r,p,z);
    }
    post_pre_fused(x,residual,prev,fn_t,scale_t,base_t,ln_t)
}
/// mhc_post_pre whose x is a pending TP sum (I3 step 2): the fused post consumes it where mhc_post_pre would run the
/// separate post + pre; the fused post+pre kernel gets the materialized sum.
pub fn mhc_post_pre_pending(x:&Tensor,pending:Option<crate::tp::PendingSum>,residual:&Tensor,prev:&PreOut,fn_t:&Tensor,scale_t:&Tensor,base_t:&Tensor,ln_t:&Tensor)->(Tensor,PreOut,Tensor) {
    let Some(p)=pending else {return mhc_post_pre(x,residual,prev,fn_t,scale_t,base_t,ln_t)};
    if post_pre_eligible(x,residual,fn_t,scale_t,base_t,ln_t) {crate::tp::materialize(p);return post_pre_fused(x,residual,prev,fn_t,scale_t,base_t,ln_t);}
    let r=mhc_post_pending(x,Some(p),residual,prev);let (p,z)=mhc_pre(&r,fn_t,scale_t,base_t,ln_t);(r,p,z)
}
fn post_pre_eligible(x:&Tensor,residual:&Tensor,fn_t:&Tensor,scale_t:&Tensor,base_t:&Tensor,ln_t:&Tensor)->bool {
    let t=residual.size()[0];
    // D3 (GLM53_MHC_POST_PRE_DECODE=1): decode rows <= 8 with the TF32 pre partial (GLM53_MHC_PRE_TC=1) use the
    // fused kernel's plain-TF32 variant, bitwise the separate mhc_post + mhc_pre of that path.
    let decode=t<=8 && std::env::var("GLM53_MHC_POST_PRE_DECODE").as_deref()==Ok("1") && std::env::var("GLM53_MHC_PRE_TC").as_deref()==Ok("1") && pre_fused_enabled();
    let eligible=((std::env::var("GLM53_MHC_POST_PRE_FUSED").as_deref()==Ok("1") && t>16 && pre_fused_enabled() && pre_large_enabled()) || decode)
        && std::env::var("GLM53_MHC_POST_FUSED").as_deref()==Ok("1") && four_streams_enabled()
        && residual.device().is_cuda() && residual.kind()==Kind::Float && residual.is_contiguous() && residual.size().len()==3 && residual.size()[1]==4
        && residual.size()[2]%128==0 && x.kind()==Kind::Float && x.size()==[t,residual.size()[2]]
        && [scale_t,base_t,ln_t].iter().all(|w|w.kind()==Kind::Float&&w.is_contiguous())
        && [Kind::Float,Kind::BFloat16].contains(&fn_t.kind()) && fn_t.is_contiguous()
        && fn_t.size()==[24,4*residual.size()[2]] && ln_t.size()==[residual.size()[2]];
    eligible
}
fn post_pre_fused(x:&Tensor,residual:&Tensor,prev:&PreOut,fn_t:&Tensor,scale_t:&Tensor,base_t:&Tensor,ln_t:&Tensor)->(Tensor,PreOut,Tensor) {
    let t=residual.size()[0];let h=residual.size()[2];let dev=residual.device();
    let x=x.contiguous();let comb=prev.comb.contiguous();let post=prev.post_mix.contiguous();
    let out=Tensor::empty([t,4,h],(Kind::Float,dev));
    let partial=Tensor::empty([(4*h/128)*t*25],(Kind::Float,dev));
    let z=Tensor::empty([t,h],(Kind::Float,dev));let post_mix=Tensor::empty([t,4],(Kind::Float,dev));let comb_next=Tensor::empty([t,4,4],(Kind::Float,dev));
    extern "C"{fn rs_mhc_post_pre_fused(x:*const f32,residual:*const f32,comb:*const f32,post:*const f32,out:*mut f32,fn_:*const std::ffi::c_void,fn_bf16:i32,scale:*const f32,base:*const f32,ln:*const f32,partial:*mut f32,z:*mut f32,post_next:*mut f32,comb_next:*mut f32,rows:i32,h:i32)->i32;}
    let p=|x:&Tensor|x.data_ptr() as *const f32;let q=|x:&Tensor|x.data_ptr() as *mut f32;
    assert_eq!(unsafe{rs_mhc_post_pre_fused(p(&x),p(residual),p(&comb),p(&post),q(&out),fn_t.data_ptr(),i32::from(fn_t.kind()==Kind::BFloat16),p(scale_t),p(base_t),p(ln_t),q(&partial),q(&z),q(&post_mix),q(&comb_next),t as i32,h as i32)},0,"fused mhc post+pre");
    (out,PreOut{post_mix,comb:comb_next},z)
}

/// I3 step 2 gate (TP2, GLM53_AR_FUSED=1 and GLM53_RDMA_AR=1 set before start): the send-only allreduce consumed by the
/// fused four-stream post equals allreduce -> (round) -> post bitwise, for the attention lane (optional Half round) and
/// the packed MoE lanes (optional shared round), rows 1-8, contiguous and strided residuals, eager over > 2 ring laps and
/// inside one CUDA graph with changing inputs, interleaved with ordinary allreduces and the materialize fallback.
pub fn ar_fused_probe(out:&std::path::Path) {
    use tch::Device;use serde_json::json;
    let _guard=tch::no_grad_guard();let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    assert!(crate::tp::ar_fused_enabled(),"set GLM53_AR_FUSED=1 and GLM53_RDMA_AR=1");
    std::env::set_var("GLM53_MHC_POST_FUSED","1");std::env::set_var("GLM53_MHC_POST_PACKED","1");std::env::set_var("GLM53_MHC_POST_FOUR_STREAMS","1");
    let dev=Device::Cuda(0);std::fs::create_dir_all(out).unwrap();let h=4096i64;let mut cases=Vec::new();
    // One step: returns (reference, fused) outputs for the same inputs. x is this rank's partial.
    let step=|x:&Tensor,residual:&Tensor,pre:&PreOut,packed:bool,round:bool|->(Tensor,Tensor) {
        let a=x.copy();crate::tp::allreduce(&a);
        let reference=if packed {mhc_post_packed(&a,residual,pre,round)} else {
            let a=if round {a.to_kind(Kind::Half).to_kind(Kind::Float)} else {a};mhc_post(&a,residual,pre)};
        let b=x.copy();assert!(crate::tp::allreduce_send(&b));
        let fused=if packed {mhc_post_packed_pending(&b,residual,pre,round)} else {
            mhc_post_pending(&b,Some(crate::tp::PendingSum{ptr:b.data_ptr() as usize,numel:b.numel() as i64,round_half:round}),residual,pre)};
        (reference,fused)
    };
    let inputs=|rows:i64,packed:bool,strided:bool,seed:i64|->(Tensor,Tensor,PreOut) {
        tch::manual_seed(seed*7919+tp.rank as i64);
        let x=Tensor::randn([rows*if packed{2}else{1},h],(Kind::Float,dev))*0.37;
        tch::manual_seed(seed);   // residual / mixes identical on both ranks, as in the model
        let residual=if strided {Tensor::randn([rows,h,4],(Kind::Float,dev)).transpose(1,2)} else {Tensor::randn([rows,4,h],(Kind::Float,dev))};
        let pre=PreOut{post_mix:Tensor::rand([rows,4],(Kind::Float,dev)),comb:Tensor::rand([rows,4,4],(Kind::Float,dev))};
        (x,residual,pre)
    };
    let mut n=0;
    for lap in 0..3 {for rows in 1..=8i64 {for packed in [false,true] {for round in [false,true] {
        let strided=!packed && lap==1;
        let (x,residual,pre)=inputs(rows,packed,strided,n);let (r,f)=step(&x,&residual,&pre,packed,round);
        tch::Cuda::synchronize(0);assert!(r.equal(&f),"fused allreduce post differs rows={rows} packed={packed} round={round} strided={strided}");n+=1;
        if lap==2 && rows==3 {   // fallback: send-only, then materialize in place == allreduce (+round)
            let mut a=x.copy();crate::tp::allreduce(&a);if round && !packed {a.copy_(&a.to_kind(Kind::Half).to_kind(Kind::Float));}
            let b=x.copy();assert!(crate::tp::allreduce_send(&b));
            crate::tp::materialize(crate::tp::PendingSum{ptr:b.data_ptr() as usize,numel:b.numel() as i64,round_half:round&&!packed});
            tch::Cuda::synchronize(0);assert!(a.equal(&b),"materialize differs");
        }
    }}}}
    eprintln!("[ar-fused-probe] rank{} eager {n} cases (3 ring laps) PASS",tp.rank);
    cases.push(json!({"eager_cases":n}));
    // Graph: 20 fused steps (> one ring lap per replay, odd count so slots shift between replays), inputs changed per replay.
    for rows in [1i64,4,8] {
        let set:Vec<_>=(0..20).map(|k|{let packed=k%2==1;inputs(rows,packed,false,1000+k)}).collect();
        let xs:Vec<Tensor>=set.iter().map(|s|s.0.copy()).collect();
        let run=|xs:&[Tensor]|->Vec<(Tensor,Tensor)> {xs.iter().zip(&set).enumerate().map(|(k,(x,s))|step(x,&s.1,&s.2,k%2==1,k%3==0)).collect()};
        for _ in 0..2 {let _=run(&xs);}tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let outs=run(&xs);crate::tp::graph::end().unwrap();
        for scale in [1.0f64,-2.0,0.5,3.0,0.0] {
            for (x,s) in xs.iter().zip(&set) {x.shallow_clone().copy_(&(&s.0*scale));}
            crate::tp::graph::replay().unwrap();tch::Cuda::synchronize(0);
            for (k,(r,f)) in outs.iter().enumerate() {assert!(r.equal(f),"graph fused post differs rows={rows} step={k} scale={scale}");}
            // an odd number of ordinary allreduces shifts the ring phase for the next replay
            crate::tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));
        }
        crate::tp::graph::destroy();
        eprintln!("[ar-fused-probe] rank{} graph rows={rows} 20 steps x 5 replays PASS",tp.rank);
        cases.push(json!({"graph_rows":rows,"steps":20,"replays":5}));
    }
    std::fs::write(out.join(format!("ar-fused-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({"rank":tp.rank,"pass":true,"cases":cases})).unwrap()).unwrap();
}
