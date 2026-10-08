// SPDX-License-Identifier: MIT
//! KDA(gated delta net)tch 实现 — 逐行对齐 engine/glm53/kda.py(M0 验收,vs GPU 内核 6e-6)。
//! 语义要点:状态 h[H,Dv,Dk],衰减 exp(λ) 乘 K 列;β 层内预 sigmoid;
//! 卷积带 silu;q/k l2norm(eps 1e-6)且 q×D^-0.5;o_norm = w·RMS(o)·sigmoid(g2)。

use tch::{Kind, Tensor};

use crate::weights::{mm16, KdaWeights};

pub fn fused_enabled()->bool {std::env::var("GLM53_KDA_FUSED").as_deref()==Ok("1")}

/// Inputs [H,D], beta [H], decay [H,D]. Both paths update state in place.
pub(crate) fn recurrent(h:&mut Tensor,q:&Tensor,k:&Tensor,v:&Tensor,beta:&Tensor,decay:&Tensor,fused:bool)->Tensor {
    if fused && h.device().is_cuda() {
        assert_eq!(h.size()[1..],[128,128]);
        for t in [&*h,q,k,v,beta,decay] {assert!(t.is_contiguous());assert_eq!(t.kind(),Kind::Float);}
        let out=Tensor::empty_like(v);
        extern "C" {fn rs_kda_recurrent(h:*mut f32,q:*const f32,k:*const f32,v:*const f32,
            beta:*const f32,decay:*const f32,out:*mut f32,heads:i32)->i32;}
        let p=|t:&Tensor|t.data_ptr() as *const f32;
        assert_eq!(unsafe{rs_kda_recurrent(h.data_ptr() as *mut f32,p(q),p(k),p(v),p(beta),p(decay),out.data_ptr() as *mut f32,h.size()[0] as i32)},0);
        return out;
    }
    let h1=&*h*decay.unsqueeze(1);
    let hk=h1.matmul(&k.unsqueeze(-1)).squeeze_dim(-1);
    let h2=&h1+(beta.unsqueeze(-1)*(v-hk)).unsqueeze(-1)*k.unsqueeze(1);
    h.copy_(&h2);
    h2.matmul(&q.unsqueeze(-1)).squeeze_dim(-1)
}

fn fork_recurrent(prior:&Tensor,q:&Tensor,k:&Tensor,v:&Tensor,beta:&Tensor,decay:&Tensor)->(Tensor,Tensor) {
    if fused_enabled() && prior.device().is_cuda() && std::env::var("GLM53_KDA_FORK_FUSED").as_deref()==Ok("1") {
        assert_eq!(prior.size()[1..],[128,128]);
        for t in [prior,q,k,v,beta,decay]{assert!(t.is_contiguous());assert_eq!(t.kind(),Kind::Float);}
        let next=Tensor::empty_like(prior);let out=Tensor::empty_like(v);
        extern "C" {fn rs_kda_fork(source:*const f32,h:*mut f32,q:*const f32,k:*const f32,v:*const f32,beta:*const f32,decay:*const f32,out:*mut f32,heads:i32)->i32;}
        let p=|t:&Tensor|t.data_ptr() as *const f32;
        assert_eq!(unsafe{rs_kda_fork(p(prior),next.data_ptr().cast(),p(q),p(k),p(v),p(beta),p(decay),out.data_ptr().cast(),prior.size()[0] as i32)},0);
        (out,next)
    }else{let mut next=prior.copy();let out=recurrent(&mut next,q,k,v,beta,decay,fused_enabled());(out,next)}
}

pub fn causal_conv1d_pub(x: &Tensor, w: &Tensor) -> Tensor {
    // x [T,C], w [C,K] → y [T,C]。左 pad K-1 个零,无偏置。
    let (t, c) = (x.size()[0], x.size()[1]);
    let k = w.size()[1];
    let mut xp = Tensor::zeros([k - 1 + t, c], (x.kind(), x.device()));
    let _ = xp.narrow(0, k - 1, t).copy_(x);
    let mut y = Tensor::zeros([t, c], (x.kind(), x.device()));
    for j in 0..k {
        y = y + w.select(1, j).unsqueeze(0) * xp.narrow(0, j, t);
    }
    y
}

fn l2(x: &Tensor) -> Tensor {
    let sq = x * x;
    x / (sq.sum_dim_intlist(&[-1i64][..], true, Kind::Float) + 1e-6).sqrt()
}

pub(crate) fn sequence(h:&mut Tensor,q:&Tensor,k:&Tensor,v:&Tensor,beta:&Tensor,decay:&Tensor)->Tensor {
    let inputs=[q,k,v,beta,decay].map(|x|x.contiguous());
    assert!(h.is_contiguous());assert_eq!(h.kind(),Kind::Float);
    for x in &inputs {assert_eq!(x.kind(),Kind::Float);}
    let out=Tensor::empty_like(&inputs[2]);
    extern "C" {fn rs_kda_sequence(h:*mut f32,q:*const f32,k:*const f32,v:*const f32,b:*const f32,d:*const f32,o:*mut f32,heads:i32,steps:i32)->i32; fn rs_kda_warp_sequence(h:*mut f32,q:*const f32,k:*const f32,v:*const f32,b:*const f32,d:*const f32,o:*mut f32,heads:i32,steps:i32)->i32;}
    let p=|i:usize|inputs[i].data_ptr() as *const f32;
    let call=if matches!(std::env::var("GLM53_KDA_SEQUENCE").as_deref(),Ok("2")|Ok("3")|Ok("4")|Ok("5")|Ok("6")){rs_kda_warp_sequence}else{rs_kda_sequence};
    assert_eq!(unsafe{call(h.data_ptr().cast(),p(0),p(1),p(2),p(3),p(4),out.data_ptr().cast(),h.size()[0] as i32,q.size()[0] as i32)},0);
    out
}

pub fn kda_forward(w: &KdaWeights, x: &Tensor) -> Tensor {
    // x [T,4096] fp32 → [T,4096]
    let t = x.size()[0];
    let (hh, dd) = (w.wq.size()[0] / 128, 128i64);
    let sc = 128f64.powf(-0.5);

    let q = l2(
        &causal_conv1d_pub(&mm16(x, &w.wq), &w.conv_q)
            .silu()
            .view([t, hh, dd]),
    ) * sc;
    let k = l2(
        &causal_conv1d_pub(&mm16(x, &w.wk), &w.conv_k)
            .silu()
            .view([t, hh, dd]),
    );
    let v = causal_conv1d_pub(&mm16(x, &w.wv), &w.conv_v)
        .silu()
        .view([t, hh, dd]);

    let beta = x.matmul(&w.wb.transpose(0, 1)).sigmoid(); // [T,H]
    let g1 = mm16(&mm16(x, &w.fa), &w.fb).view([t, hh, dd]);
    let a = w.decay_base().unsqueeze(0).unsqueeze(-1);
    let lam = -5.0f64 / ((a * (g1 + w.dt_bias.unsqueeze(0))).neg().exp() + 1.0);
    let g2 = mm16(&mm16(x, &w.ga), &w.gb).view([t, hh, dd]);

    let mut hstate = Tensor::zeros([hh, dd, dd], (Kind::Float, x.device())); // [H,Dv,Dk]
    let mut outs = Vec::with_capacity(t as usize);
    for ti in 0..t {
        hstate = &hstate * lam.get(ti).unsqueeze(1).exp(); // 衰减 exp(λ) 乘 K 列
        let kt = k.get(ti); // [H,D]
        let vt = v.get(ti);
        let hk = hstate.matmul(&kt.unsqueeze(-1)).squeeze_dim(-1); // [H,Dv]
        hstate = &hstate
            + (beta.get(ti).unsqueeze(-1) * (vt - &hk)).unsqueeze(-1) * kt.unsqueeze(1);
        outs.push(hstate.matmul(&q.get(ti).unsqueeze(-1)).squeeze_dim(-1));
    }
    let o = Tensor::stack(&outs, 0); // [T,H,D]
    let osq = &o * &o;
    let on = o * (osq.mean_dim(&[-1i64][..], true, Kind::Float) + 1e-5).rsqrt();
    let o = &w.o_norm * on * g2.sigmoid();
    crate::weights::row_mm16(&o.reshape([t, hh * dd]), &w.wo)
}

// ───────────────────────── M1.3:单步解码(增量) ─────────────────────────

pub struct KdaState {
    pub h: Tensor,        // [H,Dv,Dk] 递归状态
    pub conv: Tensor,     // [3, 3*8192] 卷积尾态(q|k|v 拼接,K-1=3 行)
}

impl KdaState {
    pub fn new(dev: tch::Device) -> Self {
        Self::with_heads(dev, 64)
    }

    pub fn with_heads(dev: tch::Device, heads: i64) -> Self {
        Self {
            h: Tensor::zeros([heads, 128, 128], (Kind::Float, dev)),
            conv: Tensor::zeros([3, 3 * heads * 128], (Kind::Float, dev)),
        }
    }
}

/// 单步:x [1,4096](该 token 的层输入)→ [1,4096];更新 h/conv 状态。
/// 状态原地更新(st.h 地址固定)——CUDA Graph 捕获要求。
pub fn kda_step(w: &KdaWeights, x: &Tensor, st: &mut KdaState) -> Tensor {
    let (hh, dd) = (w.wq.size()[0] / 128, 128i64);
    let sc = 128f64.powf(-0.5);

    // 三个投影一次算:[1, 3*8192]
    let xqkv = Tensor::cat(
        &[
            mm16(x, &w.wq),
            mm16(x, &w.wk),
            mm16(x, &w.wv),
        ],
        1,
    );
    // depthwise conv K=4:y = Σ_j w[:,j]·state[j](state 尾 3 行 + 当前行)
    let rows = Tensor::cat(&[st.conv.shallow_clone(), xqkv.shallow_clone()], 0); // [4, C]
    // depthwise 权重按段拼接:[3C, K]
    let wall = w.convolution_weights();
    let conv_out = wall.select(1, 0).unsqueeze(0) * rows.get(0).unsqueeze(0)
        + wall.select(1, 1).unsqueeze(0) * rows.get(1).unsqueeze(0)
        + wall.select(1, 2).unsqueeze(0) * rows.get(2).unsqueeze(0)
        + wall.select(1, 3).unsqueeze(0) * rows.get(3).unsqueeze(0);
    // 更新尾态(滑窗)
    let _ = st.conv.copy_(&rows.narrow(0, 1, 3));

    // silu → 拆 q/k/v → l2norm/scale
    let act = conv_out.silu();
    let qkv: Vec<Tensor> = act.split_with_sizes([hh * dd, hh * dd, hh * dd].to_vec(), 1);
    let l2 = |x: &Tensor| { let sq = x * x; x / (sq.sum_dim_intlist(&[-1i64][..], true, Kind::Float) + 1e-6).sqrt() };
    let q = l2(&qkv[0].view([1, hh, dd])) * sc;
    let k = l2(&qkv[1].view([1, hh, dd]));
    let v = qkv[2].view([1, hh, dd]);

    let beta = x.matmul(&w.wb.transpose(0, 1)).sigmoid(); // [1,H]
    let g1 = mm16(&mm16(x, &w.fa), &w.fb).view([1, hh, dd]);
    let a = w.decay_base().unsqueeze(0).unsqueeze(-1);
    let lam = -5.0f64 / ((a * (g1 + w.dt_bias.unsqueeze(0))).neg().exp() + 1.0);
    let g2 = mm16(&mm16(x, &w.ga), &w.gb).view([1, hh, dd]);

    // 递归一步:衰减 K 列 → β 校正 → 输出(原地写回 st.h,地址固定)
    let k1 = k.get(0); // [H,D]
    let v1 = v.get(0);
    let q1 = q.get(0);
    let o = recurrent(&mut st.h,&q1,&k1,&v1,&beta.get(0),&lam.get(0).exp(),fused_enabled()).unsqueeze(0);

    let osq = &o * &o;
    let on = o * (osq.mean_dim(&[-1i64][..], true, Kind::Float) + 1e-5).rsqrt();
    let out = &w.o_norm * on * g2.sigmoid();
    let h2 = out.reshape([1, hh * dd]);
    crate::weights::row_mm16(&h2, &w.wo)
}


/// 全序列前向 + 终态(h + conv 尾 3 行)供增量解码接续。

pub fn kda_forward_state(w: &KdaWeights, x: &Tensor, mut st: KdaState) -> (Tensor, KdaState) {

    if std::env::var("GLM53_PREFILL_BATCH").as_deref()==Ok("1") {
        return kda_chunk(w,x,st);
    }
    // 与 kda_forward 相同的语义,但用逐 token 更新状态(慢于批式但一次拿全状态)
    let tt = x.size()[0];
    let mut outs = Vec::with_capacity(tt as usize);
    for ti in 0..tt {
        let o = kda_step(w, &x.get(ti).unsqueeze(0), &mut st);
        outs.push(o.squeeze_dim(0));
    }
    (Tensor::stack(&outs, 0), st)
}

/// Batch projections and final row reduction; recurrence remains strictly causal.
fn kda_chunk(w:&KdaWeights,x:&Tensor,mut st:KdaState)->(Tensor,KdaState) {
    let t=x.size()[0];let heads=w.wq.size()[0]/128;assert!(t>0);
    // GLM53_PREFILL_HALF_GLUE: the plain-Half q/k/v results stay three Half tensors (no FP32 cat); see kda_chunk_fused.
    let half_glue=crate::weights::half_glue_enabled() && prefill_fused_enabled() && t>8;
    let mut hparts:Option<Vec<Tensor>>=None;
    let projected=crate::dense_fp8::try_run_rows(x,&[&w.wq,&w.wk,&w.wv]).or_else(||crate::c12::try_run_rows(x,&[&w.wq,&w.wk,&w.wv]))
        .or_else(||{if half_glue {if let Some(p)=crate::weights::mm16_half_parts(x,&[&w.wq,&w.wk,&w.wv]) {hparts=Some(p);return Some(Tensor::empty([0],(Kind::Float,x.device())));}} None})
        .or_else(||crate::weights::mm16_cat_into(x,&[&w.wq,&w.wk,&w.wv]))
        .unwrap_or_else(||Tensor::cat(&[mm16(x,&w.wq),mm16(x,&w.wk),mm16(x,&w.wv)],1));
    if prefill_fused_enabled() && t>8 {return kda_chunk_fused(w,x,st,projected,hparts,None);}
    let rows=Tensor::cat(&[&st.conv,&projected],0);
    let wall=w.convolution_weights();
    let conv=wall.select(1,0)*rows.narrow(0,0,t)+wall.select(1,1)*rows.narrow(0,1,t)
        +wall.select(1,2)*rows.narrow(0,2,t)+wall.select(1,3)*rows.narrow(0,3,t);
    st.conv.copy_(&rows.narrow(0,t,3));
    let act=conv.silu();let parts=act.split(heads*128,1);
    let q=l2(&parts[0].view([t,heads,128]))/128f64.sqrt();
    let k=l2(&parts[1].view([t,heads,128]));let v=parts[2].view([t,heads,128]);
    let beta=x.matmul(&w.wb.transpose(0,1)).sigmoid();
    let g1=mm16(&mm16(x,&w.fa),&w.fb).view([t,heads,128]);
    let a=w.decay_base().view([1,heads,1]);
    let lam=-5f64/((a*(g1+w.dt_bias.unsqueeze(0))).neg().exp()+1.);
    let decay=lam.exp();let g2=mm16(&mm16(x,&w.ga),&w.gb).view([t,heads,128]);
    let min_rows=std::env::var("GLM53_PREFILL_MIN_ROWS").ok().map(|s|s.parse::<i64>().unwrap()).unwrap_or(1);
    let o=if t>=min_rows && fused_enabled() && matches!(std::env::var("GLM53_KDA_SEQUENCE").as_deref(),Ok("1")|Ok("2")|Ok("3")|Ok("4")|Ok("5")|Ok("6")) {
        sequence(&mut st.h,&q,&k,&v,&beta,&decay)
    }else{
        let outputs:Vec<_>=(0..t).map(|i|recurrent(&mut st.h,&q.get(i),&k.get(i),&v.get(i).contiguous(),
            &beta.get(i),&decay.get(i),fused_enabled())).collect();Tensor::stack(&outputs,0)
    };let ms=(&o*&o).mean_dim(&[-1i64][..],true,Kind::Float);
    let hidden=&w.o_norm*(o*(ms+1e-5).rsqrt())*g2.sigmoid();

    ({let h2=hidden.reshape([t,heads*128]);crate::weights::row_mm16(&h2,&w.wo)},st)
}

fn onorm_sig_fused(w:&KdaWeights,heads:i64)->bool {
    std::env::var("GLM53_KDA_ONORM_SIGMOID").as_deref()==Ok("1") && w.o_norm.kind()==Kind::Float && w.o_norm.is_contiguous()
        && (w.o_norm.numel() as i64==128 || w.o_norm.numel() as i64==heads*128)
}
/// GLM53_PREFILL_AG_OVERLAP: kda_chunk_fused's row-wise front (q/k/v and the gate chains as Half GEMM results), so a
/// rank can project its own rows while the other rank's rows are still being gathered. Row-split Half GEMMs are
/// bitwise the full ones (GLM53_ROWSPLIT_CHECK), so cat_rank_rows of the two fronts is the full front (L0).
pub struct KdaFront {qkv:Vec<Tensor>,g1:Tensor,g2:Option<Tensor>}
pub fn front(w:&KdaWeights,x:&Tensor)->Option<KdaFront> {
    let heads=w.wq.size()[0]/128;
    if !crate::weights::half_glue_enabled() || !prefill_fused_enabled() || x.size()[0]<=128 {return None;}
    let qkv=crate::weights::mm16_half_parts(x,&[&w.wq,&w.wk,&w.wv])?;
    let g1=crate::weights::mm16_chain_half(x,&w.fa,&w.fb)?;
    let g2=if onorm_sig_fused(w,heads) {Some(crate::weights::mm16_chain_half(x,&w.ga,&w.gb)?)} else {None};
    Some(KdaFront{qkv,g1,g2})
}
pub fn front_cat(own:KdaFront,other:KdaFront)->KdaFront {
    let c=|a:&Tensor,b:&Tensor|crate::tp::cat_rank_rows(a,b);
    KdaFront{qkv:own.qkv.iter().zip(other.qkv.iter()).map(|(a,b)|c(a,b)).collect(),g1:c(&own.g1,&other.g1),
        g2:match (own.g2,other.g2) {(Some(a),Some(b))=>Some(c(&a,&b)),_=>None}}
}
/// kda_forward_state with the front already computed for all rows of x.
pub fn kda_forward_state_front(w:&KdaWeights,x:&Tensor,st:KdaState,f:KdaFront)->(Tensor,KdaState) {

    kda_chunk_fused(w,x,st,Tensor::empty([0],(Kind::Float,x.device())),Some(f.qkv),Some((f.g1,f.g2)))
}
pub(crate) fn prefill_fused_enabled()->bool {std::env::var("GLM53_KDA_PREFILL_FUSED").as_deref()==Ok("1")}
/// P1: kda_chunk for prefill rows without the [T+3,C] concat and the ATen elementwise
/// chains. Convolution (bitwise, same kernel as the chain path), SiLU (ATen), l2 (warp
/// sum, L1), decay (ATen op sequence), recurrence (unchanged), o-norm gate (warp sum, L1).
fn kda_chunk_fused(w:&KdaWeights,x:&Tensor,mut st:KdaState,projected:Tensor,hparts:Option<Vec<Tensor>>,pre:Option<(Tensor,Option<Tensor>)>)->(Tensor,KdaState) {
    let (pre_g1,pre_g2)=match pre {Some((a,b))=>(Some(a),b),None=>(None,None)};
    let t=x.size()[0];let heads=w.wq.size()[0]/128;let width=3*heads*128;
    let wall=w.convolution_weights().contiguous();assert_eq!(wall.size(),[width,4]);
    assert!(st.conv.is_contiguous());
    let o3=||Tensor::empty([t,heads,128],(Kind::Float,x.device()));let (q,k,v)=(o3(),o3(),o3());
    extern "C" {fn rs_kda_conv_silu_l2(base:*const f32,projected:*const f32,wall:*const f32,q:*mut f32,k:*mut f32,v:*mut f32,t:i32,h:i32,scale:f32)->i32;
        fn rs_kda_conv_silu_l2_h(base:*const f32,p0:*const std::ffi::c_void,p1:*const std::ffi::c_void,p2:*const std::ffi::c_void,wall:*const f32,q:*mut f32,k:*mut f32,v:*mut f32,t:i32,h:i32,scale:f32)->i32;
        fn rs_kda_decay_rows(g1:*const f32,a:*const f32,dt:*const f32,out:*mut f32,total:i64,heads:i32)->i32;
        fn rs_kda_decay_rows_h(g1:*const std::ffi::c_void,a:*const f32,dt:*const f32,out:*mut f32,total:i64,heads:i32)->i32;}
    if let Some(p)=&hparts {
        // GLM53_PREFILL_HALF_GLUE: the same kernel reading the three Half results (the FP32 cat held their widening).
        let hw=heads*128;for pp in p {assert!(pp.is_contiguous());assert_eq!(pp.size(),[t,hw]);assert_eq!(pp.kind(),Kind::Half);}
        assert_eq!(unsafe{rs_kda_conv_silu_l2_h(st.conv.data_ptr().cast(),p[0].data_ptr(),p[1].data_ptr(),p[2].data_ptr(),wall.data_ptr().cast(),q.data_ptr().cast(),k.data_ptr().cast(),v.data_ptr().cast(),t as i32,heads as i32,128f32.powf(-0.5))},0);
        for (i,pp) in p.iter().enumerate() {let _=st.conv.narrow(1,i as i64*hw,hw).copy_(&pp.narrow(0,t-3,3));}
    } else {
        let projected=projected.contiguous();assert_eq!(projected.size(),[t,width]);assert_eq!(projected.kind(),Kind::Float);
        // conv -> SiLU -> l2 in one pass; reads the old tail state before it is replaced.
        assert_eq!(unsafe{rs_kda_conv_silu_l2(st.conv.data_ptr().cast(),projected.data_ptr().cast(),wall.data_ptr().cast(),q.data_ptr().cast(),k.data_ptr().cast(),v.data_ptr().cast(),t as i32,heads as i32,128f32.powf(-0.5))},0);
        // The last three convolution inputs become the tail state (t > 8 >= 3).
        st.conv.copy_(&projected.narrow(0,t-3,3));
    }
    let beta=x.matmul(&w.wb.transpose(0,1)).sigmoid();
    let a=w.decay_base().contiguous();let dt=w.dt_bias.contiguous();
    assert_eq!(a.kind(),Kind::Float);assert_eq!(dt.kind(),Kind::Float);assert_eq!(dt.numel() as i64,heads*128);
    let decay=Tensor::empty([t,heads,128],(Kind::Float,x.device()));
    let glue=crate::weights::half_glue_enabled();
    if let Some(g1)=pre_g1.or_else(||glue.then(||crate::weights::mm16_chain_half(x,&w.fa,&w.fb)).flatten()) {
        let g1=g1.contiguous();assert_eq!(g1.size(),[t,heads*128]);
        assert_eq!(unsafe{rs_kda_decay_rows_h(g1.data_ptr(),a.data_ptr().cast(),dt.data_ptr().cast(),decay.data_ptr().cast(),t*heads*128,heads as i32)},0);
    } else {
        let g1=mm16(&mm16(x,&w.fa),&w.fb).contiguous();assert_eq!(g1.kind(),Kind::Float);assert_eq!(g1.size(),[t,heads*128]);
        assert_eq!(unsafe{rs_kda_decay_rows(g1.data_ptr().cast(),a.data_ptr().cast(),dt.data_ptr().cast(),decay.data_ptr().cast(),t*heads*128,heads as i32)},0);
    }
    // P10 (GLM53_KDA_ONORM_SIGMOID=1): keep the raw g2; the o-norm gate kernel applies the same sigmoid in place.
    let sig_fused=std::env::var("GLM53_KDA_ONORM_SIGMOID").as_deref()==Ok("1") && w.o_norm.kind()==Kind::Float && w.o_norm.is_contiguous()
        && (w.o_norm.numel() as i64==128 || w.o_norm.numel() as i64==heads*128);
    // GLM53_PREFILL_HALF_GLUE with the fused sigmoid: g2 stays the Half GEMM result; when wo takes the plain Half GEMM,
    // the gate kernel writes the Half hidden that GEMM's input conversion would make.
    let g2h=if pre_g2.is_some() {pre_g2.map(|g|g.contiguous())} else if glue && sig_fused {crate::weights::mm16_chain_half(x,&w.ga,&w.gb).map(|g|g.contiguous())} else {None};
    let half_out=g2h.is_some()
        && crate::weights::plain_half_prefill(&Tensor::empty([t,heads*128],(Kind::Float,x.device())),&w.wo);
    let g2=match &g2h {Some(_)=>Tensor::empty([0],(Kind::Float,x.device())),None=>mm16(&mm16(x,&w.ga),&w.gb).view([t,heads,128])};
    let sg2=if sig_fused {g2} else {g2.sigmoid()};
    let o=if fused_enabled() && matches!(std::env::var("GLM53_KDA_SEQUENCE").as_deref(),Ok("1")|Ok("2")|Ok("3")|Ok("4")|Ok("5")|Ok("6")) {
        sequence(&mut st.h,&q,&k,&v,&beta,&decay)
    }else{
        let outputs:Vec<_>=(0..t).map(|i|recurrent(&mut st.h,&q.get(i),&k.get(i),&v.get(i),&beta.get(i),&decay.get(i),fused_enabled())).collect();Tensor::stack(&outputs,0)
    };
    if let Some(g2h)=&g2h {
        let o=o.contiguous();assert_eq!(o.size(),[t,heads,128]);assert_eq!(o.kind(),Kind::Float);assert_eq!(g2h.size(),[t,heads*128]);
        if half_out {
            let out=Tensor::empty([t,heads*128],(Kind::Half,o.device()));
            extern "C"{fn rs_kda_onorm_gate_sig_hh(o:*const f32,onorm:*const f32,len:i32,g2:*const std::ffi::c_void,out:*mut std::ffi::c_void,groups:i32)->i32;}
            assert_eq!(unsafe{rs_kda_onorm_gate_sig_hh(o.data_ptr().cast(),w.o_norm.data_ptr().cast(),w.o_norm.numel() as i32,g2h.data_ptr(),out.data_ptr(),(t*heads) as i32)},0);
            return (crate::weights::row_mm16(&out,&w.wo),st);
        }
        let g2f=g2h.to_kind(Kind::Float);let out=Tensor::empty([t,heads,128],(Kind::Float,o.device()));
        extern "C"{fn rs_kda_onorm_gate_sig(o:*const f32,onorm:*const f32,len:i32,g2:*const f32,out:*mut f32,groups:i32)->i32;}
        assert_eq!(unsafe{rs_kda_onorm_gate_sig(o.data_ptr().cast(),w.o_norm.data_ptr().cast(),w.o_norm.numel() as i32,g2f.data_ptr().cast(),out.data_ptr().cast(),(t*heads) as i32)},0);

        return ({let h2=out.reshape([t,heads*128]);crate::weights::row_mm16(&h2,&w.wo)},st);
    }
    let hidden=if sig_fused && o.kind()==Kind::Float && o.size()==[t,heads,128] {
        let o=o.contiguous();let g2=sg2.contiguous();let out=Tensor::empty([t,heads,128],(Kind::Float,o.device()));
        extern "C"{fn rs_kda_onorm_gate_sig(o:*const f32,onorm:*const f32,len:i32,g2:*const f32,out:*mut f32,groups:i32)->i32;}
        assert_eq!(unsafe{rs_kda_onorm_gate_sig(o.data_ptr().cast(),w.o_norm.data_ptr().cast(),w.o_norm.numel() as i32,g2.data_ptr().cast(),out.data_ptr().cast(),(t*heads) as i32)},0);
        out
    } else {onorm_gate_any(w,&o,&if sig_fused {sg2.sigmoid()} else {sg2},heads)};

    ({let h2=hidden.reshape([t,heads*128]);crate::weights::row_mm16(&h2,&w.wo)},st)
}
fn onorm_gate_any(w:&KdaWeights,o:&Tensor,sg2:&Tensor,heads:i64)->Tensor {
    let t=o.size()[0];
    if o.kind()==Kind::Float && o.size()==[t,heads,128] && sg2.size()==[t,heads,128] && w.o_norm.is_contiguous() && w.o_norm.kind()==Kind::Float
        && (w.o_norm.numel() as i64==128 || w.o_norm.numel() as i64==heads*128) {
        let o=o.contiguous();let sg2=sg2.contiguous();let out=Tensor::empty([t,heads,128],(Kind::Float,o.device()));
        extern "C"{fn rs_kda_onorm_gate(o:*const f32,onorm:*const f32,len:i32,sg2:*const f32,out:*mut f32,groups:i32)->i32;}
        assert_eq!(unsafe{rs_kda_onorm_gate(o.data_ptr().cast(),w.o_norm.data_ptr().cast(),w.o_norm.numel() as i32,sg2.data_ptr().cast(),out.data_ptr().cast(),(t*heads) as i32)},0);
        return out;
    }
    let ms=(o*o).mean_dim(&[-1i64][..],true,Kind::Float);
    &w.o_norm*(o*(ms+1e-5).rsqrt())*sg2
}

/// Only the first root followed by its immediate predecessor is a fixed chain.
/// A sibling, second root, or larger tree must keep parent-indexed histories.
fn bounded_chain(parents:&[Option<usize>])->bool {
    !parents.is_empty() && parents.len()<=16 &&
        parents.iter().enumerate().all(|(i,&p)|p==i.checked_sub(1))
}

/// Capture identity may query eligibility without tensors or device reads.
/// A per-layer tensor contract is checked separately by tree_deferred before
/// any projection runs. Old tree APIs always continue returning full states.
pub(crate) fn deferred_eligible(parents:&[Option<usize>])->bool {
    bounded_chain(parents) && crate::kda_correction::enabled() && fused_enabled() &&
        ["GLM53_KDA_FORK_FUSED","GLM53_KDA_CONV_CHAIN","GLM53_KDA_CHAIN_NORM","GLM53_KDA_CHAIN_RECURRENT"]
            .iter().all(|name|std::env::var(name).as_deref()==Ok("1"))
}

/// Read the immutable base and batched projections directly, without building
/// a four-row temporary for every node. Each returned convolution state is an
/// independent slice of newly owned storage, never an alias of the inputs.
pub(crate) fn chain_convolution(base:&Tensor,projected:&Tensor,wall:&Tensor)->(Tensor,Tensor) {
    let (act,states)=chain_convolution_impl(base,projected,wall,true);
    (act,states.unwrap())
}

pub(crate) fn chain_convolution_deferred(base:&Tensor,projected:&Tensor,wall:&Tensor)
    ->(Tensor,crate::kda_correction::ConvRecord) {
    let (act,states)=chain_convolution_impl(base,projected,wall,false);
    assert!(states.is_none());
    (act,crate::kda_correction::ConvRecord::Deferred {
        base:base.shallow_clone(),projected:projected.shallow_clone(),
    })
}

fn chain_convolution_impl(base:&Tensor,projected:&Tensor,wall:&Tensor,write_states:bool)->(Tensor,Option<Tensor>) {
    assert_eq!(projected.dim(),2);assert_eq!(base.dim(),2);assert_eq!(wall.dim(),2);
    let t=projected.size()[0];let width=projected.size()[1];
    assert!((1..=16).contains(&t)&&width>0);assert_eq!(base.size(),[3,width]);
    assert_eq!(wall.size(),[width,4]);
    for x in [base,projected,wall] {
        assert!(x.is_contiguous());assert_eq!(x.kind(),Kind::Float);
        assert_eq!(x.device(),projected.device());
    }
    assert!(projected.device().is_cuda());
    let conv=Tensor::empty_like(projected);
    let states=write_states.then(||Tensor::empty([t,3,width],(Kind::Float,projected.device())));
    extern "C" {fn rs_kda_conv_chain(base:*const f32,projected:*const f32,wall:*const f32,
        conv:*mut f32,states:*mut f32,tokens:i32,width:i32)->i32;}
    assert_eq!(unsafe{rs_kda_conv_chain(base.data_ptr().cast(),projected.data_ptr().cast(),wall.data_ptr().cast(),
        conv.data_ptr().cast(),states.as_ref().map_or(std::ptr::null_mut(),|s|s.data_ptr().cast()),
        t.try_into().unwrap(),width.try_into().unwrap())},0);
    // Reuse the existing SiLU implementation; only its elementwise launch is
    // batched. In particular, do not replace it by a different exp intrinsic.
    // GLM53_KDA_CONV_SILU=1: the convolution kernel already applied the same SiLU.
    if std::env::var("GLM53_KDA_CONV_SILU").as_deref()==Ok("1") {return (conv,states);}
    (conv.silu(),states)
}

/// GLM53_KDA_CONV_L2=1: deferred convolution + SiLU + q/k/v normalization in one launch (kda_conv_l2; bitwise the
/// kda_conv_chain<false,true> + kda_chain_l2 pair). Needs the deferred record, the in-kernel SiLU and the fused norm.
fn conv_l2_fused(base:&Tensor,projected:&Tensor,wall:&Tensor,heads:i64)->Option<(Tensor,Tensor,Tensor)> {
    let on=|k:&str|std::env::var(k).as_deref()==Ok("1");
    if !(on("GLM53_KDA_CONV_L2") && on("GLM53_KDA_CONV_DEFERRED") && on("GLM53_KDA_CONV_SILU") && norm_fused_enabled()) {return None;}
    let t=projected.size()[0];let width=projected.size()[1];
    if !(1..=16).contains(&t) || width!=3*heads*128 || base.size()!=[3,width] || wall.size()!=[width,4]
        || [base,projected,wall].iter().any(|x|!x.is_contiguous()||x.kind()!=Kind::Float||x.device()!=projected.device()) || !projected.device().is_cuda() {return None;}
    let o=||Tensor::empty([t,heads,128],(Kind::Float,projected.device()));let (q,k,v)=(o(),o(),o());
    extern "C"{fn rs_kda_conv_l2(base:*const f32,projected:*const f32,weights:*const f32,q:*mut f32,k:*mut f32,v:*mut f32,t:i32,h:i32,scale:f32)->i32;}
    assert_eq!(unsafe{rs_kda_conv_l2(base.data_ptr().cast(),projected.data_ptr().cast(),wall.data_ptr().cast(),q.data_ptr().cast(),k.data_ptr().cast(),
        v.data_ptr().cast(),t as i32,heads as i32,128f32.powf(-0.5))},0,"KDA conv+l2");
    Some((q,k,v))
}
/// Batch only the independent row/head normalizations. Keep the same last-axis
/// reduction, epsilon, sqrt/division and query scaling as the per-node path.
/// The Q/K slices have gaps between tokens; the results must be contiguous for
/// fork_recurrent's raw-pointer contract when a single token is selected.
pub(crate) fn norm_fused_enabled()->bool {std::env::var("GLM53_NORM_FUSED").as_deref()==Ok("1")}
/// W05c: q = l2(act_q)*128^-0.5, k = l2(act_k), v = act_v in one launch (L1: l2 sum order).
fn chain_normalization_fused(act:&Tensor,heads:i64)->Option<(Tensor,Tensor,Tensor)> {
    if !norm_fused_enabled() || !act.is_contiguous() || act.kind()!=Kind::Float || !act.device().is_cuda() {return None;}
    let t=act.size()[0];let o=|| Tensor::empty([t,heads,128],(Kind::Float,act.device()));let (q,k,v)=(o(),o(),o());
    extern "C"{fn rs_kda_chain_l2(act:*const f32,q:*mut f32,k:*mut f32,v:*mut f32,t:i32,h:i32,scale:f32)->i32;}
    assert_eq!(unsafe{rs_kda_chain_l2(act.data_ptr().cast(),q.data_ptr().cast(),k.data_ptr().cast(),v.data_ptr().cast(),t as i32,heads as i32,128f32.powf(-0.5))},0);
    Some((q,k,v))
}
fn chain_normalization(act:&Tensor,heads:i64)->(Tensor,Tensor) {
    if let Some((q,k,_))=chain_normalization_fused(act,heads){return (q,k);}
    let t=act.size()[0];assert_eq!(act.size(),[t,3*heads*128]);
    let q=(l2(&act.narrow(1,0,heads*128).view([t,heads,128]))*128f64.powf(-0.5)).contiguous();
    let k=l2(&act.narrow(1,heads*128,heads*128).view([t,heads,128])).contiguous();
    (q,k)
}

/// Keep all committed-prefix choices as independent states. This only removes
/// repeated reads of H between fixed-chain nodes, not their output snapshots.
pub(crate) fn chain_recurrent(base:&Tensor,q:&Tensor,k:&Tensor,v:&Tensor,beta:&Tensor,decay:&Tensor)->(Tensor,Tensor) {
    let t=q.size()[0];let heads=base.size()[0];
    assert!((1..=16).contains(&t));assert_eq!(base.size(),[heads,128,128]);
    for x in [q,k,v,decay] {assert_eq!(x.size(),[t,heads,128]);}
    assert_eq!(beta.size(),[t,heads]);
    for x in [base,q,k,v,beta,decay] {
        assert!(x.is_contiguous());assert_eq!(x.kind(),Kind::Float);assert_eq!(x.device(),base.device());
    }
    assert!(base.device().is_cuda());
    let states=Tensor::empty([t,heads,128,128],(Kind::Float,base.device()));
    let out=Tensor::empty_like(v);
    extern "C" {fn rs_kda_recurrent_chain(base:*const f32,q:*const f32,k:*const f32,v:*const f32,
        beta:*const f32,decay:*const f32,states:*mut f32,out:*mut f32,heads:i32,tokens:i32)->i32;}
    assert_eq!(unsafe{rs_kda_recurrent_chain(base.data_ptr().cast(),q.data_ptr().cast(),k.data_ptr().cast(),v.data_ptr().cast(),
        beta.data_ptr().cast(),decay.data_ptr().cast(),states.data_ptr().cast(),out.data_ptr().cast(),
        heads.try_into().unwrap(),t.try_into().unwrap())},0);
    (out,states)
}

/// Verifier-only opt-in. None means the caller must use the existing full
/// tree path; no placeholder H or partially executed projection is returned.

/// The fused gate kernel takes at most 8 rows; per-row math is independent of the split.
fn gate_values_split(w:&KdaWeights,x:&Tensor,heads:i64)->(Tensor,Tensor,Tensor) {
    let t=x.size()[0];if t<=8 {return gate_values(w,x,heads);}
    let split=||{let parts:Vec<_>=(0..t).step_by(8).map(|r|gate_values(w,&x.narrow(0,r,(t-r).min(8)),heads)).collect();
        (Tensor::cat(&parts.iter().map(|p|&p.0).collect::<Vec<_>>(),0),Tensor::cat(&parts.iter().map(|p|&p.1).collect::<Vec<_>>(),0),
         Tensor::cat(&parts.iter().map(|p|&p.2).collect::<Vec<_>>(),0))};
    if t<=32 && gate_wide() {if let Some(v)=gate_fused_rows(w,x,heads,32){
        if wide_check_on() {let s=split();wide_check("kda-gate",&Tensor::cat(&[v.0.reshape([-1]),v.1.reshape([-1]),v.2.reshape([-1])],0),
            &Tensor::cat(&[s.0.reshape([-1]),s.1.reshape([-1]),s.2.reshape([-1])],0));}
        return v;}}
    split()
}
/// GLM53_WIDE_CHECK=1: every wide (9..32-row) call is recomputed through the 8-row split and compared bitwise.
pub(crate) fn wide_check_on()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_WIDE_CHECK").as_deref()==Ok("1")) && !crate::tp::graph::capturing()}
pub(crate) fn wide_check(name:&str,a:&Tensor,b:&Tensor) {
    use std::sync::atomic::{AtomicU64,Ordering::Relaxed};static OK:AtomicU64=AtomicU64::new(0);static BAD:AtomicU64=AtomicU64::new(0);
    let same=a.size()==b.size() && a.eq_tensor(b).all().int64_value(&[])==1;
    let (o,bad)=if same {(OK.fetch_add(1,Relaxed)+1,BAD.load(Relaxed))} else {(OK.load(Relaxed),BAD.fetch_add(1,Relaxed)+1)};
    if !same {let d=(a-b).abs().max().double_value(&[]);eprintln!("[wide-check] MISMATCH {name} rows-shape {:?} max_abs {d:e} (ok {o} bad {bad})",a.size());}
    else if o%500==0 {eprintln!("[wide-check] ok {o} bad {bad}");}
}
/// Shared by every verifier-tree path so graph (deferred) and eager references stay identical.
fn gate_values(w:&KdaWeights,x:&Tensor,heads:i64)->(Tensor,Tensor,Tensor) {
    let t=x.size()[0];
    if let Some(v)=gate_fused(w,x,heads){return v;}
    let beta=x.matmul(&w.wb.transpose(0,1)).sigmoid();
    let g1=mm16(&mm16(x,&w.fa),&w.fb).view([t,heads,128]);
    let a=w.decay_base().view([1,heads,1]);
    let decay=(-5f64/((a*(g1+w.dt_bias.unsqueeze(0))).neg().exp()+1.)).exp();
    let g2=mm16(&mm16(x,&w.ga),&w.gb).view([t,heads,128]);
    (beta,decay,g2.sigmoid())
}
fn onorm_gate(w:&KdaWeights,o:&Tensor,sg2:&Tensor,heads:i64)->Tensor {
    let t=o.size()[0];
    if gate_fused_enabled() && o.kind()==Kind::Float && o.size()==[t,heads,128] && sg2.size()==[t,heads,128]
        && w.o_norm.is_contiguous() && w.o_norm.kind()==Kind::Float && (w.o_norm.numel() as i64==128 || w.o_norm.numel() as i64==heads*128) {
        let o=o.contiguous();let sg2=sg2.contiguous();
        let out=Tensor::empty([t,heads,128],(Kind::Float,o.device()));
        extern "C"{fn rs_kda_onorm_gate(o:*const f32,onorm:*const f32,len:i32,sg2:*const f32,out:*mut f32,groups:i32)->i32;}
        assert_eq!(unsafe{rs_kda_onorm_gate(o.data_ptr().cast(),w.o_norm.data_ptr().cast(),w.o_norm.numel() as i32,sg2.data_ptr().cast(),out.data_ptr().cast(),(t*heads) as i32)},0);
        return out;
    }
    let ms=(o*o).mean_dim(&[-1i64][..],true,Kind::Float);
    &w.o_norm*(o*(ms+1e-5).rsqrt())*sg2
}
fn gate_wide()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_KDA_GATE_WIDE").as_deref()==Ok("1"))}
pub(crate) fn gate_fused_enabled()->bool {std::env::var("GLM53_KDA_GATE_FUSED").as_deref()==Ok("1")}
/// W05: beta, decay and sigmoid(g2) in two kernels (L1: projection summation order and
/// FP32 FMA instead of TF32 for b_proj). Returns None when shapes/dtypes do not match.
fn gate_fused(w:&KdaWeights,x:&Tensor,heads:i64)->Option<(Tensor,Tensor,Tensor)> {gate_fused_rows(w,x,heads,8)}
/// GLM53_KDA_GATE_WIDE: only callers that otherwise split into fused 8-row pieces pass max 32 (L0 for them); the
/// tree path keeps max 8, so its 9..32-row windows stay on the ATen projections.
fn gate_fused_rows(w:&KdaWeights,x:&Tensor,heads:i64,max:i64)->Option<(Tensor,Tensor,Tensor)> {
    if !gate_fused_enabled() || x.size()[0]>max {return None;}
    gate_fused_raw(x,&w.fa,&w.ga,&w.wb,&w.fb,&w.gb,&w.a_log,&w.dt_bias,heads)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_fused_raw(x:&Tensor,fa:&Tensor,ga:&Tensor,wb:&Tensor,fb:&Tensor,gb:&Tensor,a_log:&Tensor,dt_bias:&Tensor,heads:i64)->Option<(Tensor,Tensor,Tensor)> {
    struct W<'a>{fa:&'a Tensor,ga:&'a Tensor,wb:&'a Tensor,fb:&'a Tensor,gb:&'a Tensor,a_log:&'a Tensor,dt_bias:&'a Tensor}
    let w=W{fa,ga,wb,fb,gb,a_log,dt_bias};
    let t=x.size()[0];let k=x.size()[1];let rank=w.fa.size()[0];let n=heads*128;
    let ok=(1..=(if gate_wide(){32}else{8})).contains(&t) && k%1024==0 && rank==128 && w.fa.kind()==Kind::Half && w.ga.kind()==Kind::Half
        && w.fb.kind()==Kind::Half && w.gb.kind()==Kind::Half && w.wb.kind()==Kind::Float
        && w.fa.size()==[rank,k] && w.ga.size()==[rank,k] && w.fb.size()==[n,rank] && w.gb.size()==[n,rank]
        && w.wb.size()==[heads,k] && w.dt_bias.numel() as i64==n && w.a_log.numel() as i64==heads
        && w.dt_bias.kind()==Kind::Float && w.a_log.kind()==Kind::Float
        && [w.fa,w.ga,w.fb,w.gb,w.wb,w.dt_bias,w.a_log].iter().all(|t|t.is_contiguous()) && x.is_contiguous() && x.kind()==Kind::Float;
    if !ok {return None;}
    let dev=x.device();
    // mid [t,2*rank] followed by the K-quarter partials [4,t,2*rank+heads] (scratch only).
    let mid=Tensor::empty([t*2*rank+4*t*(2*rank+heads)],(Kind::Float,dev));let beta=Tensor::empty([t,heads],(Kind::Float,dev));
    let decay=Tensor::empty([t,heads,128],(Kind::Float,dev));let sg2=Tensor::empty([t,heads,128],(Kind::Float,dev));
    extern "C"{fn rs_kda_gate(x:*const f32,fa:*const std::ffi::c_void,ga:*const std::ffi::c_void,wb:*const f32,fb:*const std::ffi::c_void,gb:*const std::ffi::c_void,a_log:*const f32,dt_bias:*const f32,mid:*mut f32,beta:*mut f32,decay:*mut f32,sg2:*mut f32,rows:i32,k:i32,rank:i32,nb:i32,n:i32)->i32;}
    assert_eq!(unsafe{rs_kda_gate(x.data_ptr().cast(),w.fa.data_ptr(),w.ga.data_ptr(),w.wb.data_ptr().cast(),w.fb.data_ptr(),w.gb.data_ptr(),
        w.a_log.data_ptr().cast(),w.dt_bias.data_ptr().cast(),mid.data_ptr().cast(),beta.data_ptr().cast(),decay.data_ptr().cast(),sg2.data_ptr().cast(),
        t as i32,k as i32,rank as i32,heads as i32,n as i32)},0,"KDA gate fusion");
    Some((beta,decay,sg2))
}

/// Keep the projection, convolution, normalization and output math identical
/// to tree's existing conv+norm+recurrent-chain branch.
/// GLM53_KDA_GATE_SIDE=1 helpers. The side stream (process pool) must never run cuBLAS: only the fused gate kernels run
/// there (<= 8 rows, or <= 32 with GLM53_KDA_GATE_WIDE, exactly what gate_values_split would launch); when they do not
/// apply, the stream is joined and the gates are computed on the main stream as before.
type Gates=(Tensor,Tensor,Tensor);
fn side_gates_start(w:&KdaWeights,x:&Tensor,heads:i64)->Option<Option<Gates>> {
    let t=x.size()[0];let max=if t<=8 {8} else if gate_wide() {32} else {0};
    if !gate_side_enabled() || t>max {return None;}
    extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;}
    assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);
    let g=gate_fused_rows(w,x,heads,max);assert_eq!(unsafe{rs_stream_set(-1)},0);Some(g)
}
fn side_gates_finish(g:Option<Option<Gates>>,w:&KdaWeights,x:&Tensor,heads:i64)->Gates {
    if g.is_some() {extern "C"{fn rs_stream_join(n:i32)->i32;}assert_eq!(unsafe{rs_stream_join(1)},0);}
    match g.flatten() {Some(v)=>v,None=>gate_values_split(w,x,heads)}
}
pub(crate) fn gate_side_enabled()->bool {std::env::var("GLM53_KDA_GATE_SIDE").as_deref()==Ok("1")}
pub(crate) fn tree_deferred(w:&KdaWeights,x:&Tensor,base:&KdaState,parents:&[Option<usize>])
    ->Option<(Tensor,crate::kda_correction::ChainRecord)> {
    if !deferred_eligible(parents) || !x.device().is_cuda() || x.kind()!=Kind::Float ||
        !x.is_contiguous() || x.dim()!=2 || w.wq.dim()!=2 || x.size()[0] as usize!=parents.len() {
        return None;
    }
    let width=w.wq.size()[0];let heads=width/128;
    if width%128!=0 || !(1..=1024).contains(&heads) || x.size()[1]!=w.wq.size()[1] ||
        base.h.size()!=[heads,128,128] || base.conv.size()!=[3,3*heads*128] ||
        [&base.h,&base.conv].iter().any(|t|t.kind()!=Kind::Float || !t.is_contiguous() || t.device()!=x.device()) {
        return None;
    }
    let t=x.size()[0];
    // GLM53_KDA_GATE_SIDE=1 (L0, schedule only): the gate chain (x only) runs on a forked side stream while the
    // bandwidth-bound q/k/v projection runs on the main stream; joined before the convolution consumes both.
    let gates=side_gates_start(w,x,heads);
    let projected=crate::dense_fp8::try_run_rows(x,&[&w.wq,&w.wk,&w.wv]).or_else(||crate::c12::try_run_rows(x,&[&w.wq,&w.wk,&w.wv])).unwrap_or_else(||Tensor::cat(&[mm16(x,&w.wq),mm16(x,&w.wk),mm16(x,&w.wv)],1));
    crate::forward::probe_point(0,"kda_proj",&projected);
    let wall=w.convolution_weights();
    let (beta,decay,g2s)=side_gates_finish(gates,w,x,heads);
    crate::forward::probe_point(0,"kda_beta",&beta);crate::forward::probe_point(0,"kda_decay",&decay);crate::forward::probe_point(0,"kda_g2s",&g2s);
    if let Some((q,k,v))=conv_l2_fused(&base.conv,&projected,&wall,heads) {
        let conv=crate::kda_correction::ConvRecord::Deferred{base:base.conv.shallow_clone(),projected:projected.shallow_clone()};
        crate::forward::probe_point(0,"kda_q",&q);crate::forward::probe_point(0,"kda_k",&k);
        let beta=beta.contiguous();let decay=decay.contiguous();
        let (o,correction)=crate::kda_correction::produce(&base.h,&q,&k,&v,&beta,&decay);
        crate::forward::probe_point(0,"kda_o",&o);
        let hidden=onorm_gate(w,&o,&g2s,heads);
        crate::forward::probe_point(0,"kda_hidden",&hidden);
        let output={let h2=hidden.reshape([t,heads*128]);crate::weights::row_mm16(&h2,&w.wo)};
        return Some((output,crate::kda_correction::ChainRecord{base_h:base.h.shallow_clone(),k,decay,correction,conv}));
    }
    let (act,conv)=if std::env::var("GLM53_KDA_CONV_DEFERRED").as_deref()==Ok("1") {
        chain_convolution_deferred(&base.conv,&projected,&wall)
    }else {
        let (act,states)=chain_convolution(&base.conv,&projected,&wall);
        (act,crate::kda_correction::ConvRecord::Full(states))
    };
    let (q,k,v)=chain_normalization_fused(&act,heads).unwrap_or_else(||{let (q,k)=chain_normalization(&act,heads);
        (q,k,act.narrow(1,2*heads*128,heads*128).view([t,heads,128]).contiguous())});
    crate::forward::probe_point(0,"kda_act",&act);crate::forward::probe_point(0,"kda_q",&q);crate::forward::probe_point(0,"kda_k",&k);
    let beta=beta.contiguous();let decay=decay.contiguous();
    let (o,correction)=crate::kda_correction::produce(&base.h,&q,&k,&v,&beta,&decay);
    crate::forward::probe_point(0,"kda_o",&o);
    let hidden=onorm_gate(w,&o,&g2s,heads);
    crate::forward::probe_point(0,"kda_hidden",&hidden);
    let output={let h2=hidden.reshape([t,heads*128]);crate::weights::row_mm16(&h2,&w.wo)};
    Some((output,crate::kda_correction::ChainRecord {
        base_h:base.h.shallow_clone(),k,decay,correction,conv,
    }))
}

/// Multi-sequence verifier (serving batch): the row-wise work (Q/K/V projection, gates, L2
/// normalization, output norm/gate and the WO projection with its TP sum) runs once over the rows of
/// all sequences, so the layer weights are read once per round; only the state-dependent convolution
/// chain and recurrence run per sequence. Each sequence keeps its own deferred record.
pub(crate) fn tree_deferred_multi(w:&KdaWeights,x:&Tensor,bases:&[&KdaState],segs:&[(usize,usize)],parents_all:&[Vec<Option<usize>>])
    ->Option<(Tensor,Vec<crate::kda_correction::ChainRecord>)> {
    if std::env::var("GLM53_KDA_MULTI").as_deref()==Ok("0") || !x.device().is_cuda() || x.kind()!=Kind::Float || !x.is_contiguous() || x.dim()!=2
        || w.wq.dim()!=2 || std::env::var("GLM53_KDA_CONV_DEFERRED").as_deref()!=Ok("1") {return None;}
    let width=w.wq.size()[0];let heads=width/128;
    if width%128!=0 || x.size()[1]!=w.wq.size()[1] {return None;}
    for (g,b) in bases.iter().enumerate() {
        if !deferred_eligible(&parents_all[g]) || parents_all[g].len()!=segs[g].1 || b.h.size()!=[heads,128,128] || b.conv.size()!=[3,3*heads*128]
            || [&b.h,&b.conv].iter().any(|t|t.kind()!=Kind::Float || !t.is_contiguous() || t.device()!=x.device()) {return None;}
    }
    let t=x.size()[0];
    let gates=side_gates_start(w,x,heads);
    let projected=crate::dense_fp8::try_run_rows(x,&[&w.wq,&w.wk,&w.wv]).or_else(||crate::c12::try_run_rows(x,&[&w.wq,&w.wk,&w.wv])).unwrap_or_else(||Tensor::cat(&[mm16(x,&w.wq),mm16(x,&w.wk),mm16(x,&w.wv)],1));
    let (beta,decay,g2s)=side_gates_finish(gates,w,x,heads);
    let beta=beta.contiguous();let decay=decay.contiguous();
    let wall=w.convolution_weights();
    // GLM53_KDA_MULTI_LAUNCH=1 (L0): the per-sequence convolution chains and correction chains as one launch each over
    // the stacked rows (kda_conv_chain_multi / kda_correction_chain_thread_multi; per-row arithmetic unchanged), and no
    // cat of the per-sequence outputs.
    let multi_launch=std::env::var("GLM53_KDA_MULTI_LAUNCH").as_deref()==Ok("1") && segs.len()<=8 && segs.iter().all(|s|(1..=16).contains(&s.1));
    #[repr(C)] struct SeqTab{base:[*const f32;8],first:[i32;8],len:[i32;8],n:i32}
    let tab=|ptrs:&dyn Fn(usize)->*const f32|{let mut t=SeqTab{base:[std::ptr::null();8],first:[0;8],len:[0;8],n:segs.len() as i32};
        for (g,&(first,len)) in segs.iter().enumerate() {t.base[g]=ptrs(g);t.first[g]=first as i32;t.len[g]=len as i32;}t};
    let mut convs=Vec::with_capacity(segs.len());
    let act=if multi_launch {
        let projected=projected.contiguous();let wall=wall.contiguous();let width=projected.size()[1];
        assert_eq!(wall.size(),[width,4]);assert_eq!(projected.kind(),Kind::Float);
        for b in bases {assert_eq!(b.conv.size(),[3,width]);}
        let act=Tensor::empty_like(&projected);let tb=tab(&|g|bases[g].conv.data_ptr() as *const f32);
        extern "C"{fn rs_kda_conv_chain_multi(table:*const std::ffi::c_void,projected:*const f32,weights:*const f32,conv:*mut f32,tokens:i32,width:i32)->i32;}
        assert_eq!(unsafe{rs_kda_conv_chain_multi((&tb as *const SeqTab).cast(),projected.data_ptr().cast(),wall.data_ptr().cast(),act.data_ptr().cast(),t as i32,width as i32)},0,"KDA conv chain multi");
        for (g,&(first,len)) in segs.iter().enumerate() {
            convs.push(crate::kda_correction::ConvRecord::Deferred{base:bases[g].conv.shallow_clone(),projected:projected.narrow(0,first as i64,len as i64)});
        }
        if wide_check_on() {let parts:Vec<Tensor>=segs.iter().enumerate().map(|(g,&(first,len))|chain_convolution_deferred(&bases[g].conv,&projected.narrow(0,first as i64,len as i64),&wall).0).collect();
            wide_check("kda-conv-multi",&act,&Tensor::cat(&parts,0));}
        act
    } else {
    let mut acts=Vec::with_capacity(segs.len());
    for (g,&(first,len)) in segs.iter().enumerate() {
        let (act,conv)=chain_convolution_deferred(&bases[g].conv,&projected.narrow(0,first as i64,len as i64),&wall);
        acts.push(act);convs.push(conv);
    }
    Tensor::cat(&acts,0)};
    let (q,k,v)=chain_normalization_fused(&act,heads).unwrap_or_else(||{let (q,k)=chain_normalization(&act,heads);
        (q,k,act.narrow(1,2*heads*128,heads*128).view([t,heads,128]).contiguous())});
    let mut records=Vec::with_capacity(segs.len());
    let o=if multi_launch {
        let (q,k,v)=(q.contiguous(),k.contiguous(),v.contiguous());
        for x in [&q,&k,&v,&decay] {assert_eq!(x.size(),[t,heads,128]);assert_eq!(x.kind(),Kind::Float);}
        assert_eq!(beta.size(),[t,heads]);
        for b in bases {assert_eq!(b.h.size(),[heads,128,128]);}
        let out=Tensor::empty_like(&v);let corr=Tensor::empty_like(&v);let tb=tab(&|g|bases[g].h.data_ptr() as *const f32);
        extern "C"{fn rs_kda_correction_chain_multi(table:*const std::ffi::c_void,q:*const f32,k:*const f32,v:*const f32,beta:*const f32,decay:*const f32,correction:*mut f32,out:*mut f32,heads:i32)->i32;}
        assert_eq!(unsafe{rs_kda_correction_chain_multi((&tb as *const SeqTab).cast(),q.data_ptr().cast(),k.data_ptr().cast(),v.data_ptr().cast(),beta.data_ptr().cast(),
            decay.data_ptr().cast(),corr.data_ptr().cast(),out.data_ptr().cast(),heads as i32)},0,"KDA correction multi");
        if wide_check_on() {
            let (mut os,mut cs)=(Vec::new(),Vec::new());
            for (g,&(first,len)) in segs.iter().enumerate() {let r=|x:&Tensor|x.narrow(0,first as i64,len as i64);
                let (o,c)=crate::kda_correction::produce(&bases[g].h,&r(&q),&r(&k),&r(&v),&r(&beta),&r(&decay));os.push(o);cs.push(c);}
            wide_check("kda-corr-multi",&Tensor::cat(&[out.reshape([-1]),corr.reshape([-1])],0),&Tensor::cat(&[Tensor::cat(&os,0).reshape([-1]),Tensor::cat(&cs,0).reshape([-1])],0));
        }
        for (g,(&(first,len),conv)) in segs.iter().zip(convs).enumerate() {
            let r=|x:&Tensor|x.narrow(0,first as i64,len as i64);
            records.push(crate::kda_correction::ChainRecord{base_h:bases[g].h.shallow_clone(),k:r(&k),decay:r(&decay),correction:r(&corr),conv});
        }
        out
    } else {
    let mut outs=Vec::with_capacity(segs.len());
    for (g,(&(first,len),conv)) in segs.iter().zip(convs).enumerate() {
        let r=|x:&Tensor|x.narrow(0,first as i64,len as i64);
        let kg=r(&k);let dg=r(&decay);
        let (o,correction)=crate::kda_correction::produce(&bases[g].h,&r(&q),&kg,&r(&v),&r(&beta),&dg);
        outs.push(o);
        records.push(crate::kda_correction::ChainRecord{base_h:bases[g].h.shallow_clone(),k:kg,decay:dg,correction,conv});
    }
    Tensor::cat(&outs,0)};
    let hidden=onorm_gate(w,&o,&g2s,heads);
    Some(({let h2=hidden.reshape([t,heads*128]);crate::weights::row_mm16(&h2,&w.wo)},records))
}

/// Batched projections with an independent recurrent/conv state for every tree node.
pub fn tree(w:&KdaWeights,x:&Tensor,base:&KdaState,parents:&[Option<usize>])->(Tensor,Vec<KdaState>) {
    let t=x.size()[0];let heads=w.wq.size()[0]/128;assert_eq!(t as usize,parents.len());
    let projected=crate::dense_fp8::try_run_rows(x,&[&w.wq,&w.wk,&w.wv]).or_else(||crate::c12::try_run_rows(x,&[&w.wq,&w.wk,&w.wv])).unwrap_or_else(||Tensor::cat(&[mm16(x,&w.wq),mm16(x,&w.wk),mm16(x,&w.wv)],1));
    let wall=w.convolution_weights();
    let (beta,decay,g2s)=gate_values(w,x,heads);
    let conv_chain=(x.device().is_cuda() && bounded_chain(parents) &&
        std::env::var("GLM53_KDA_CONV_CHAIN").as_deref()==Ok("1"))
        .then(||chain_convolution(&base.conv,&projected,&wall));
    let normalized=if std::env::var("GLM53_KDA_CHAIN_NORM").as_deref()==Ok("1") {
        conv_chain.as_ref().map(|(act,_)|chain_normalization(act,heads))
    }else{None};
    let recurrent_chain=if fused_enabled() && std::env::var("GLM53_KDA_FORK_FUSED").as_deref()==Ok("1") &&
        std::env::var("GLM53_KDA_CHAIN_RECURRENT").as_deref()==Ok("1") {
        normalized.as_ref().map(|(q,k)|{
            let act=&conv_chain.as_ref().unwrap().0;
            let v=act.narrow(1,2*heads*128,heads*128).view([t,heads,128]).contiguous();
            chain_recurrent(&base.h,q,k,&v,&beta.contiguous(),&decay.contiguous())
        })
    }else{None};
    let mut states:Vec<KdaState>=Vec::new();let mut outputs=Vec::new();
    for (i,&parent) in parents.iter().enumerate() {
        assert!(parent.map_or(true,|p|p<i));let prior=parent.map_or(base,|p|&states[p]);
        let (act,conv_state)=if let Some((act,conv_states))=&conv_chain {
            (act.get(i as i64),conv_states.get(i as i64))
        }else {
        let rows=Tensor::cat(&[&prior.conv,&projected.narrow(0,i as i64,1)],0);
        let conv=wall.select(1,0)*rows.get(0)+wall.select(1,1)*rows.get(1)
            +wall.select(1,2)*rows.get(2)+wall.select(1,3)*rows.get(3);
            (conv.silu(),rows.narrow(0,1,3))
        };
        let parts=act.split(heads*128,0);
        let (q,k)=if let Some((q,k))=&normalized {(q.get(i as i64),k.get(i as i64))}
        else{(l2(&parts[0].view([heads,128]))*128f64.powf(-0.5),l2(&parts[1].view([heads,128])))};
        let v=parts[2].view([heads,128]);
        let (output,h)=if let Some((o,h))=&recurrent_chain {(o.get(i as i64),h.get(i as i64))}
        else{fork_recurrent(&prior.h,&q,&k,&v,&beta.get(i as i64),&decay.get(i as i64))};
        outputs.push(output);
        // Keep the fallback copy at its original point after recurrence.
        let conv=if conv_chain.is_some(){conv_state}else{conv_state.copy()};
        states.push(KdaState{h,conv});
    }
    let o=if let Some((o,_))=&recurrent_chain{o.shallow_clone()}else{Tensor::stack(&outputs,0)};
    let hidden=onorm_gate(w,&o,&g2s,heads);
    ({let h2=hidden.reshape([t,heads*128]);crate::weights::row_mm16(&h2,&w.wo)},states)
}

pub fn fork_probe(out:&std::path::Path) {
    use std::time::Instant;use tch::Device;use serde_json::json;
    let _guard=tch::no_grad_guard();let dev=Device::Cuda(0);let mut cases=Vec::new();
    std::env::set_var("GLM53_KDA_FUSED","1");tch::manual_seed(2419);
    for heads in [1,32,64] {
        let original=Tensor::randn([heads,128,128],(Kind::Float,dev))*0.1;let mut input=original.copy();
        let q=Tensor::randn([heads,128],(Kind::Float,dev))*0.01;let k=Tensor::randn_like(&q)*0.01;
        let v=Tensor::randn_like(&q);let beta=Tensor::rand([heads],(Kind::Float,dev));let decay=Tensor::rand_like(&q);
        let mut rounds=Vec::new();
        for on in [false,true,true,false] {
            std::env::set_var("GLM53_KDA_FORK_FUSED",if on{"1"}else{"0"});
            let call=|h:&Tensor|fork_recurrent(h,&q,&k,&v,&beta,&decay);
            for _ in 0..3 {let _=call(&input);}tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let (y,next)=call(&input);crate::tp::graph::end().unwrap();
            for z in [&original,&(-&original),&Tensor::zeros_like(&original)] {
                input.copy_(z);crate::tp::graph::replay().unwrap();
                let mut expected=z.copy();let result=recurrent(&mut expected,&q,&k,&v,&beta,&decay,true);
                assert!(input.equal(z)&&next.equal(&expected)&&y.equal(&result),"fork changed-input/state isolation");
            }
            input.copy_(&original);tch::Cuda::synchronize(0);let begin=Instant::now();
            for _ in 0..128{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"fused_fork":on,"graph_us":begin.elapsed().as_secs_f64()*1e6/128.}));crate::tp::graph::destroy();
        }
        cases.push(json!({"heads":heads,"exact":true,"rounds":rounds}));
    }
    std::fs::create_dir_all(out).unwrap();std::fs::write(out.join("fork.json"),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
}

/// Original per-node convolution, without projections/recurrence, for the
/// chain probe's independent state and floating-point reference.
fn chain_convolution_reference(base:&Tensor,projected:&Tensor,wall:&Tensor)->(Tensor,Tensor) {
    let (acts,states)=chain_convolution_reference_parts(base,projected,wall);
    (Tensor::stack(&acts,0),Tensor::stack(&states,0))
}
fn chain_convolution_reference_parts(base:&Tensor,projected:&Tensor,wall:&Tensor)->(Vec<Tensor>,Vec<Tensor>) {
    let mut prior=base.shallow_clone();let mut acts=Vec::new();let mut states=Vec::new();
    for i in 0..projected.size()[0] {
        let rows=Tensor::cat(&[&prior,&projected.narrow(0,i,1)],0);
        let conv=wall.select(1,0)*rows.get(0)+wall.select(1,1)*rows.get(1)
            +wall.select(1,2)*rows.get(2)+wall.select(1,3)*rows.get(3);
        acts.push(conv.silu());prior=rows.narrow(0,1,3).copy();states.push(prior.shallow_clone());
    }
    (acts,states)
}

fn chain_recurrent_reference(base:&Tensor,q:&Tensor,k:&Tensor,v:&Tensor,beta:&Tensor,decay:&Tensor)->(Tensor,Tensor) {
    let mut previous=base.shallow_clone();let mut outputs=Vec::new();let mut states=Vec::new();
    for i in 0..q.size()[0] {
        let mut next=previous.copy();
        outputs.push(recurrent(&mut next,&q.get(i),&k.get(i),&v.get(i),&beta.get(i),&decay.get(i),true));
        states.push(next.shallow_clone());previous=next;
    }
    (Tensor::stack(&outputs,0),Tensor::stack(&states,0))
}

fn chain_normalization_reference(act:&Tensor,heads:i64)->(Tensor,Tensor) {
    let mut q=Vec::new();let mut k=Vec::new();
    for i in 0..act.size()[0] {
        let parts=act.get(i).split(heads*128,0);
        q.push(l2(&parts[0].view([heads,128]))*128f64.powf(-0.5));
        k.push(l2(&parts[1].view([heads,128])));
    }
    (Tensor::stack(&q,0),Tensor::stack(&k,0))
}

/// Standalone one-GPU probe; the owner runs this before TP2 request ABBA.
/// Numeric coverage includes every supported length, poisoned unused rows,
/// changing graph inputs, disjoint states, and real checkpoint KDA weights.
pub fn conv_chain_probe(model:&std::path::Path,out:&std::path::Path) {
    use tch::Device;use serde_json::json;use std::time::Instant;
    assert!(!crate::tp::is_tp(),"local probe must not use a TP2 launch environment");
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();let dev=Device::Cuda(0);
    std::fs::create_dir_all(out).unwrap();tch::manual_seed(20260923);
    let equal=|a:&(Tensor,Tensor),b:&(Tensor,Tensor)|{
        assert!(a.0.equal(&b.0),"chain primary output mismatch");
        assert!(a.1.equal(&b.1),"chain state/secondary output mismatch");
        assert!(a.0.isfinite().all().int64_value(&[])!=0);
    };
    let mut numeric=Vec::new();
    for heads in [1,32,64] {for t in 1..=8 {
        let width=3*heads*128;
        let original=Tensor::randn([8,width],(Kind::Float,dev));
        let base_original=Tensor::randn([3,width],(Kind::Float,dev));
        let wall=Tensor::randn([width,4],(Kind::Float,dev))*0.1;
        let mut storage=Tensor::full([11,width],f64::NAN,(Kind::Float,dev));
        let mut projected=storage.narrow(0,0,t);let mut base=base_original.copy();
        for scale in [0.,1.,1e6,1e-20] {
            let _=storage.fill_(f64::NAN);projected.copy_(&(&original.narrow(0,0,t)*scale));
            base.copy_(&(&base_original*scale));
            let gold=chain_convolution_reference(&base,&projected,&wall);
            let candidate=chain_convolution(&base,&projected,&wall);equal(&candidate,&gold);
            equal(&chain_normalization(&candidate.0,heads),&chain_normalization_reference(&gold.0,heads));
            // Return slices may share an allocation, but never a writable range.
            if t>1 {
                let prior=candidate.1.get(0).copy();let _=candidate.1.get(t-1).fill_(f64::NAN);
                assert!(candidate.1.get(0).equal(&prior));
            }
        }
        base.copy_(&base_original);projected.copy_(&original.narrow(0,0,t));
        let _=chain_convolution(&base,&projected,&wall);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();
        let captured=chain_convolution(&base,&projected,&wall);
        let captured_norm=chain_normalization(&captured.0,heads);
        crate::tp::graph::end().unwrap();
        for scale in [1.,-1.,0.,100.] {
            let _=storage.fill_(f64::NAN);projected.copy_(&(&original.narrow(0,0,t)*scale));
            base.copy_(&(&base_original*scale));crate::tp::graph::replay().unwrap();
            let expected=chain_convolution_reference(&base,&projected,&wall);equal(&captured,&expected);
            equal(&captured_norm,&chain_normalization_reference(&expected.0,heads));
            assert!(base.equal(&(&base_original*scale)),"base overwritten");
            assert!(projected.equal(&(&original.narrow(0,0,t)*scale)),"projected overwritten");
        }
        crate::tp::graph::destroy();
        // Every recurrence state must agree with independent fused steps,
        // including graph replay after H and Q change at the same addresses.
        let original_h=Tensor::randn([heads,128,128],(Kind::Float,dev))*0.1;
        let mut input_h=original_h.copy();
        let original_q=Tensor::randn([t,heads,128],(Kind::Float,dev))*0.01;let mut q=original_q.copy();
        let k=Tensor::randn_like(&q)*0.01;let v=Tensor::randn_like(&q);
        let beta=Tensor::rand([t,heads],(Kind::Float,dev));let decay=Tensor::rand_like(&q);
        let candidate=chain_recurrent(&input_h,&q,&k,&v,&beta,&decay);
        equal(&candidate,&chain_recurrent_reference(&input_h,&q,&k,&v,&beta,&decay));
        let _=chain_recurrent(&input_h,&q,&k,&v,&beta,&decay);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let captured=chain_recurrent(&input_h,&q,&k,&v,&beta,&decay);
        crate::tp::graph::end().unwrap();
        for scale in [1.,-1.,0.,100.] {
            input_h.copy_(&(&original_h*scale));q.copy_(&(&original_q*scale));
            let _=captured.1.shallow_clone().fill_(f64::NAN);crate::tp::graph::replay().unwrap();
            equal(&captured,&chain_recurrent_reference(&input_h,&q,&k,&v,&beta,&decay));
            assert!(input_h.equal(&(&original_h*scale)),"chain recurrence modified base");
            if t>1 {
                let first=captured.1.get(0).copy();let _=captured.1.get(t-1).fill_(f64::NAN);
                assert!(captured.1.get(0).equal(&first),"chain H states alias writable ranges");
            }
        }
        crate::tp::graph::destroy();
        numeric.push(json!({"heads":heads,"tokens":t,"exact":true,"scales":[0.,1.,1e6,1e-20],"changed_graph_input":true,"inactive_nan":true,"disjoint_states":true,
            "batch_norm_exact":true,"batch_norm_changed_graph_input":true,
            "recurrent_output_and_all_states_exact":true,"recurrent_changed_graph_input":true,"recurrent_disjoint_states":true}));
    }}
    std::fs::write(out.join("conv-local.json"),serde_json::to_string_pretty(&json!({"cases":numeric})).unwrap()).unwrap();
    eprintln!("[kda-conv-chain] 24 shapes, extreme values, graph input and isolation passed");

    // Real layer-0 projected values, from actual embedding and MHC inputs.
    // The full local layer also exercises reduction/recurrence unchanged.
    std::env::set_var("GLM53_KDA_FUSED","1");std::env::set_var("GLM53_KDA_FORK_FUSED","1");
    std::env::set_var("GLM53_STATIC_TENSORS","1");
    let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let weights=crate::weights::ModelWeights::load(model,&cfg,1,dev);
    let layer=&weights.layers[0];let w=layer.kda.as_ref().unwrap();let heads=w.wq.size()[0]/128;
    let ids=Tensor::from_slice(&[154822i64,154824,154826,25062,287,29905,13041,13]).to_device(dev);
    let residual=crate::mhc::hc_expand(&weights.embed_tokens(&ids));
    let (_,z)=crate::mhc::mhc_pre(&residual,&layer.hc.attn_fn,&layer.hc.attn_scale,&layer.hc.attn_base,&layer.hc.in_ln);
    let mut base=KdaState::with_heads(dev,heads);
    for i in 0..3 {let _=kda_step(w,&z.narrow(0,i,1),&mut base);}
    let h_original=base.h.copy();let conv_original=base.conv.copy();let mut full=Vec::new();
    let mut shapes:Vec<Vec<Option<usize>>>=(1..=8).map(|t|(0..t).map(|i:usize|i.checked_sub(1)).collect()).collect();
    shapes.push(vec![None,Some(0),Some(0),Some(1),Some(2)]);
    shapes.push(vec![None,None,Some(1),Some(0)]);
    shapes.push((0..9).map(|i:usize|i.checked_sub(1)).collect());
    let compare=|actual:&Tensor,expected:&Tensor|{
        assert!(actual.isfinite().all().int64_value(&[])!=0,"nonfinite KDA candidate");
        let delta=actual-expected;
        let max_abs=delta.abs().max().double_value(&[]);
        let relative=(delta.norm()/expected.norm().clamp_min(1e-30)).double_value(&[]);
        (actual.equal(expected),max_abs,relative)
    };
    let mut all_norm_exact=true;let mut all_recurrent_exact=true;
    for parents in shapes {
        let t=parents.len() as i64;
        let x=if t<=8{z.narrow(0,0,t)}else{Tensor::cat(&[&z,&z.narrow(0,0,1)],0)};
        std::env::set_var("GLM53_KDA_CHAIN_NORM","0");
        std::env::set_var("GLM53_KDA_CHAIN_RECURRENT","0");
        std::env::set_var("GLM53_KDA_CONV_CHAIN","0");let expected=tree(w,&x,&base,&parents);
        let mut modes=Vec::new();
        for (conv,norm,recurrent) in [(true,false,false),(true,true,false),(true,true,true),(false,true,true),(true,false,true)] {
            std::env::set_var("GLM53_KDA_CONV_CHAIN",if conv{"1"}else{"0"});
            std::env::set_var("GLM53_KDA_CHAIN_NORM",if norm{"1"}else{"0"});
            std::env::set_var("GLM53_KDA_CHAIN_RECURRENT",if recurrent{"1"}else{"0"});
            let actual=tree(w,&x,&base,&parents);let (output_exact,output_abs,output_rel)=compare(&actual.0,&expected.0);
            let mut h_exact=true;let mut conv_exact=true;let mut h_abs=0f64;let mut h_rel=0f64;
            for (a,b) in actual.1.iter().zip(&expected.1) {
                let (exact,abs,rel)=compare(&a.h,&b.h);h_exact&=exact;h_abs=h_abs.max(abs);h_rel=h_rel.max(rel);
                conv_exact&=a.conv.equal(&b.conv);
            }
            let exact=output_exact&&h_exact&&conv_exact;
            let norm_effective=conv&&norm&&bounded_chain(&parents);
            let recurrent_effective=norm_effective&&recurrent;
            if recurrent_effective{all_recurrent_exact&=exact;}
            else if norm_effective{all_norm_exact&=exact;}
            else{assert!(exact,"real KDA conv-only/fallback mismatch");}
            assert!(conv_exact,"normalization must not alter convolution states");
            assert!(base.h.equal(&h_original)&&base.conv.equal(&conv_original),"real base mutated");
            modes.push(json!({"conv_chain":conv,"chain_norm":norm,"chain_recurrent":recurrent,"norm_effective":norm_effective,"recurrent_effective":recurrent_effective,
                "output_exact":output_exact,"h_exact":h_exact,"conv_exact":conv_exact,
                "output_max_abs":output_abs,"output_relative_l2":output_rel,"h_max_abs":h_abs,"h_max_relative_l2":h_rel}));
        }
        full.push(json!({"tokens":t,"parents":parents,"eligible":bounded_chain(&parents),"modes":modes}));
    }
    std::fs::write(out.join("conv-real.json"),serde_json::to_string_pretty(&json!({"heads":heads,"cases":full,"chain_norm_all_exact":all_norm_exact,
        "chain_recurrent_all_exact":all_recurrent_exact,"gate":all_norm_exact&&all_recurrent_exact,
        "chain_norm_status":if all_norm_exact{"exact in tested real-layer cases"}else{"numerical differences observed; requires separate qualification"},
        "source":"actual layer 0 weights, embedding/MHC values and advanced KDA state"})).unwrap()).unwrap();
    eprintln!("[kda-conv-chain] real conv-only/fallback exact; batched normalization all exact: {all_norm_exact}");
    assert!(all_norm_exact,"batched KDA normalization changed output/state; see conv-real.json (gate=false)");
    assert!(all_recurrent_exact,"chained KDA recurrence changed output/state; see conv-real.json (gate=false)");

    let projected=Tensor::cat(&[mm16(&z,&w.wq),mm16(&z,&w.wk),mm16(&z,&w.wv)],1);
    let wall=w.convolution_weights();
    // Model loading above is single-rank. Select the first 32 heads of each
    // Q/K/V segment so timing uses the real TP2 local shape, not full 64 heads.
    assert!(heads>=32);let local_heads=32;
    let shard=|x:&Tensor,axis:i64|{
        let parts:Vec<_>=x.split(heads*128,axis).iter()
            .map(|p|p.narrow(axis,0,local_heads*128)).collect();Tensor::cat(&parts,axis)
    };
    let projected=shard(&projected,1);let wall=shard(&wall,0);let conv_base=shard(&base.conv,1);
    let mut timing=Vec::new();
    for t in [1,5,8] {
        // A 64-instance working set distinguishes a single hot scratch block
        // from the many per-layer states used by the real verifier.
        let inputs:Vec<_>=(0..64).map(|_|(conv_base.copy(),projected.narrow(0,0,t).copy(),wall.copy())).collect();
        let working_bytes=inputs.iter().map(|(a,b,c)|(a.numel()+b.numel()+c.numel())*4).sum::<usize>();
        let mut rounds=Vec::new();
        for fused in [false,true,true,false] {
            let run=|a:&Tensor,b:&Tensor,c:&Tensor|{
                if fused{let (x,s)=chain_convolution(a,b,c);vec![x,s]}
                else{let (mut x,s)=chain_convolution_reference_parts(a,b,c);x.extend(s);x}
            };
            for (a,b,c) in &inputs {let _=run(a,b,c);}tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();
            let outputs:Vec<_>=inputs.iter().map(|(a,b,c)|run(a,b,c)).collect();
            crate::tp::graph::end().unwrap();for _ in 0..3{crate::tp::graph::replay().unwrap();}
            tch::Cuda::synchronize(0);let begin=Instant::now();
            for _ in 0..32{crate::tp::graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"fused":fused,"us_per_layer":begin.elapsed().as_secs_f64()*1e6/(32.*64.)}));
            crate::tp::graph::destroy();drop(outputs);
        }
        timing.push(json!({"tokens":t,"heads":local_heads,"rotating_inputs":64,"input_bytes":working_bytes,"rounds":rounds}));
    }
    std::env::set_var("GLM53_KDA_CONV_CHAIN","0");
    std::env::set_var("GLM53_KDA_CHAIN_NORM","0");
    std::env::set_var("GLM53_KDA_CHAIN_RECURRENT","0");
    std::fs::write(out.join("conv-timing.json"),serde_json::to_string_pretty(&json!({"scope":"local ABBA; does not establish whole-request throughput","cases":timing})).unwrap()).unwrap();
}

/// W05 local gate: fused KDA gate/o-norm kernels vs the ATen formulas on real rank-0 shards.
pub fn gate_probe(model:&std::path::Path,out:&std::path::Path) {
    use serde_json::json;let _g=tch::no_grad_guard();let dev=tch::Device::Cuda(0);std::fs::create_dir_all(out).unwrap();
    let mut idx=crate::safetensors::ShardIndex::scan(model).unwrap();let mut cases=Vec::new();let mut ws=Vec::new();
    let mut get=|n:&str,kind:Kind|{let (v,s)=idx.get_f32(n).unwrap();
        Tensor::from_slice(&v).view(s.iter().map(|&x|x as i64).collect::<Vec<_>>().as_slice()).to_kind(kind).to_device(dev)};
    for layer in [0i64,4,30] {
        let p=format!("model.language_model.layers.{layer}.self_attn");
        let fa=get(&format!("{p}.f_a_proj.weight"),Kind::Half);let ga=get(&format!("{p}.g_a_proj.weight"),Kind::Half);
        let fb=get(&format!("{p}.f_b_proj.weight"),Kind::Half).narrow(0,0,4096).contiguous();
        let gb=get(&format!("{p}.g_b_proj.weight"),Kind::Half).narrow(0,0,4096).contiguous();
        let wb=get(&format!("{p}.b_proj.weight"),Kind::Float).narrow(0,0,32).contiguous();
        let dt=get(&format!("{p}.dt_bias"),Kind::Float).view([64,128]).narrow(0,0,32).contiguous();
        let al=get(&format!("{p}.A_log"),Kind::Float).narrow(0,0,32).contiguous();
        let on=get(&format!("{p}.o_norm.weight"),Kind::Float).contiguous();
        for rows in 1..=8i64 { for scale in [0.1f64,1.,8.] {
            tch::manual_seed(77+rows+layer);let x=(Tensor::randn([rows,4096],(Kind::Float,dev))*scale).contiguous();
            let (b1,d1,s1)=gate_fused_raw(&x,&fa,&ga,&wb,&fb,&gb,&al,&dt,32).expect("eligible");
            // GLM53_KDA_GATE_ONE must reproduce the two-stage kernels bit for bit.
            std::env::set_var("GLM53_KDA_GATE_ONE","1");
            let (b2,d2,s2)=gate_fused_raw(&x,&fa,&ga,&wb,&fb,&gb,&al,&dt,32).expect("eligible");
            std::env::remove_var("GLM53_KDA_GATE_ONE");
            assert!(b1.equal(&b2)&&d1.equal(&d2)&&s1.equal(&s2),"GLM53_KDA_GATE_ONE not bitwise: layer {layer} rows {rows} scale {scale}");
            let beta=x.matmul(&wb.transpose(0,1)).sigmoid();
            let g1=mm16(&mm16(&x,&fa),&fb).view([rows,32,128]);
            let decay=(-5f64/((al.exp().view([1,32,1])*(g1+dt.unsqueeze(0))).neg().exp()+1.)).exp();
            let sg2=mm16(&mm16(&x,&ga),&gb).view([rows,32,128]).sigmoid();
            let o=Tensor::randn([rows,32,128],(Kind::Float,dev));
            let ms=(&o*&o).mean_dim(&[-1i64][..],true,Kind::Float);let href=&on*(&o*(ms+1e-5).rsqrt())*&sg2;
            let hf=Tensor::empty([rows,32,128],(Kind::Float,dev));
            extern "C"{fn rs_kda_onorm_gate(o:*const f32,onorm:*const f32,len:i32,sg2:*const f32,out:*mut f32,groups:i32)->i32;}
            assert_eq!(unsafe{rs_kda_onorm_gate(o.data_ptr().cast(),on.data_ptr().cast(),on.numel() as i32,sg2.data_ptr().cast(),hf.data_ptr().cast(),(rows*32) as i32)},0);
            let d=|a:&Tensor,b:&Tensor|f64::try_from((a-b).abs().max()).unwrap();
            let r=json!({"layer":layer,"rows":rows,"scale":scale,"beta":d(&b1,&beta),"decay":d(&d1,&decay),"sg2":d(&s1,&sg2),"onorm_rel":f64::try_from((&hf-&href).norm()/href.norm()).unwrap(),
                "decay_rel":f64::try_from((&d1-&decay).norm()/decay.norm()).unwrap()});
            assert!(r["beta"].as_f64().unwrap()<1e-3 && r["decay_rel"].as_f64().unwrap()<1e-3 && r["sg2"].as_f64().unwrap()<1e-2 && r["onorm_rel"].as_f64().unwrap()<1e-5,"KDA gate mismatch {r}");
            cases.push(r);
        }}
        eprintln!("[kda-gate] layer {layer} ok (one-kernel bitwise)");
        ws.push((fa,ga,wb,fb,gb,al,dt));
    }
    // Timing: two-stage vs one kernel, rotating the three layers' weights, 1/2/4/8 rows.
    let mut timing=Vec::new();
    for rows in [1i64,2,4,8] {
        let x=(Tensor::randn([rows,4096],(Kind::Float,dev))).contiguous();
        for mode in ["0","1","0","1"] {
            std::env::set_var("GLM53_KDA_GATE_ONE",mode);
            for (fa,ga,wb,fb,gb,al,dt) in &ws {let _=gate_fused_raw(&x,fa,ga,wb,fb,gb,al,dt,32);}
            tch::Cuda::synchronize(0);let t0=std::time::Instant::now();let iters=600;
            for i in 0..iters {let (fa,ga,wb,fb,gb,al,dt)=&ws[i%ws.len()];let _=gate_fused_raw(&x,fa,ga,wb,fb,gb,al,dt,32);}
            tch::Cuda::synchronize(0);let us=t0.elapsed().as_secs_f64()*1e6/iters as f64;
            eprintln!("[kda-gate] rows {rows} one={mode} {us:.2} us/call (host+alloc incl.)");
            timing.push(json!({"rows":rows,"one":mode,"us":us}));
        }
    }
    std::env::remove_var("GLM53_KDA_GATE_ONE");
    std::fs::write(out.join("kda-gate-probe.json"),serde_json::to_string_pretty(&json!({"gate":true,"cases":cases,"timing":timing})).unwrap()).unwrap();
    eprintln!("[kda-gate] PASS {}",cases.len());
}

/// P1 gate: kda_chunk fused prefill path vs the ATen path on real sharded weights,
/// chunks of 9/300/2048 rows continuing the same state; outputs, H and conv tail.
pub fn prefill_probe(model:&std::path::Path,out:&std::path::Path) {
    use serde_json::json;
    let _g=tch::no_grad_guard();let dev=tch::Device::Cuda(0);std::fs::create_dir_all(out).unwrap();tch::manual_seed(77);
    let cfg=crate::config::load(&model.join("config.json")).unwrap();let mut weights=crate::weights::ModelWeights::load(model,&cfg,4,dev);
    for layer in &mut weights.layers{crate::weights::shard_dense_layer(layer,0,2);}
    let mut cases=Vec::new();
    for (li,layer) in weights.layers.iter().enumerate() {
        let Some(w)=&layer.kda else {continue};let heads=w.wq.size()[0]/128;
        let mut a=KdaState::with_heads(dev,heads);let mut b=KdaState::with_heads(dev,heads);
        for (ci,&n) in [9i64,300,2048].iter().enumerate() {
            let x=Tensor::randn([n,4096],(Kind::Float,dev))*0.05;
            std::env::set_var("GLM53_KDA_PREFILL_FUSED","0");let (ya,na)=kda_chunk(w,&x,a);a=na;
            std::env::set_var("GLM53_KDA_PREFILL_FUSED","1");let (yb,nb)=kda_chunk(w,&x,b);b=nb;
            let d=|p:&Tensor,q:&Tensor|f64::try_from((p-q).abs().max()).unwrap();
            let rel=|p:&Tensor,q:&Tensor|f64::try_from((p-q).norm()/q.norm().clamp_min(1e-30)).unwrap();
            let (dy,ry,dh,dc)=(d(&yb,&ya),rel(&yb,&ya),d(&b.h,&a.h),d(&b.conv,&a.conv));
            eprintln!("[kda-prefill] layer {li} chunk {ci} rows {n}: out max {dy:.3e} rel {ry:.3e} H max {dh:.3e} conv {dc:.3e} exact={}",yb.equal(&ya));
            assert!(yb.isfinite().all().int64_value(&[])!=0 && ry<1e-3 && dc==0.,"kda prefill fused mismatch");
            cases.push(json!({"layer":li,"rows":n,"out_max_abs":dy,"out_rel":ry,"h_max_abs":dh,"conv_max_abs":dc,"exact":yb.equal(&ya)}));
        }
        {// conv+SiLU+l2 kernel versus conv_chain -> ATen SiLU -> chain_l2 (must be bitwise).
            let t=2048i64;let width=3*heads*128;let projected=Tensor::randn([t,width],(Kind::Float,dev));
            let base=Tensor::randn([3,width],(Kind::Float,dev));let wall=w.convolution_weights().contiguous();
            let conv=Tensor::empty_like(&projected);
            extern "C" {fn rs_kda_conv_chain(base:*const f32,projected:*const f32,wall:*const f32,conv:*mut f32,states:*mut f32,tokens:i32,width:i32)->i32;
                fn rs_kda_chain_l2(act:*const f32,q:*mut f32,k:*mut f32,v:*mut f32,t:i32,h:i32,scale:f32)->i32;
                fn rs_kda_conv_silu_l2(base:*const f32,projected:*const f32,wall:*const f32,q:*mut f32,k:*mut f32,v:*mut f32,t:i32,h:i32,scale:f32)->i32;}
            assert_eq!(unsafe{rs_kda_conv_chain(base.data_ptr().cast(),projected.data_ptr().cast(),wall.data_ptr().cast(),conv.data_ptr().cast(),std::ptr::null_mut(),t as i32,width as i32)},0);
            let act=conv.silu();let o3=||Tensor::empty([t,heads,128],(Kind::Float,dev));
            let (q0,k0,v0,q1,k1,v1)=(o3(),o3(),o3(),o3(),o3(),o3());let sc=128f32.powf(-0.5);
            assert_eq!(unsafe{rs_kda_chain_l2(act.data_ptr().cast(),q0.data_ptr().cast(),k0.data_ptr().cast(),v0.data_ptr().cast(),t as i32,heads as i32,sc)},0);
            assert_eq!(unsafe{rs_kda_conv_silu_l2(base.data_ptr().cast(),projected.data_ptr().cast(),wall.data_ptr().cast(),q1.data_ptr().cast(),k1.data_ptr().cast(),v1.data_ptr().cast(),t as i32,heads as i32,sc)},0);
            let exact=q0.equal(&q1)&&k0.equal(&k1)&&v0.equal(&v1);
            eprintln!("[kda-prefill] conv+silu+l2 bitwise={exact} max {:.3e}",f64::try_from((&q0-&q1).abs().max()).unwrap().max(f64::try_from((&v0-&v1).abs().max()).unwrap()));
            cases.push(json!({"conv_silu_l2_bitwise":exact}));
        }
        {// KDA sequence mode 3 (two rows/warp + prefetch) must equal mode 2 bitwise.
            let (t,h)=(2048i64,heads);let n=|s:&[i64]|Tensor::randn(s,(Kind::Float,dev));
            let nrm=|x:Tensor|{let s=(&x*&x).sum_dim_intlist(&[-1i64][..],true,Kind::Float).sqrt();x/s};
            let (q,k,v)=(nrm(n(&[t,h,128]))*0.088,nrm(n(&[t,h,128])),n(&[t,h,128]));
            let beta=Tensor::rand([t,h],(Kind::Float,dev));let decay=Tensor::rand([t,h,128],(Kind::Float,dev))*0.1+0.9;
            let init=n(&[h,128,128])*0.01;let mut res=Vec::new();let mut times=Vec::new();
            for mode in ["2","3","4","4","3","2"] {std::env::set_var("GLM53_KDA_SEQUENCE",mode);let mut st=init.copy();tch::Cuda::synchronize(0);
                let t0=std::time::Instant::now();let o=sequence(&mut st,&q,&k,&v,&beta,&decay);tch::Cuda::synchronize(0);
                times.push((mode,t0.elapsed().as_secs_f64()*1000.));res.push((o,st));}
            let exact=(1..6).all(|i|res[0].0.equal(&res[i].0)&&res[0].1.equal(&res[i].1));
            eprintln!("[kda-prefill] sequence mode3 bitwise={exact} times {times:?}");assert!(exact,"KDA sequence mode 3 differs");
            std::env::set_var("GLM53_KDA_SEQUENCE","2");cases.push(json!({"sequence3_bitwise":exact,"times":format!("{times:?}")}));
        }
        let x=Tensor::randn([2048,4096],(Kind::Float,dev))*0.05;let mut timing=Vec::new();
        for on in [false,true,true,false] {
            std::env::set_var("GLM53_KDA_PREFILL_FUSED",if on{"1"}else{"0"});
            let st=KdaState::with_heads(dev,heads);let _=kda_chunk(w,&x,st);tch::Cuda::synchronize(0);
            let t0=std::time::Instant::now();for _ in 0..5{let st=KdaState::with_heads(dev,heads);let _=kda_chunk(w,&x,st);}tch::Cuda::synchronize(0);
            timing.push(json!({"fused":on,"ms":t0.elapsed().as_secs_f64()*1000./5.}));
        }
        eprintln!("[kda-prefill] timing {timing:?}");cases.push(json!({"layer":li,"timing":timing}));
        break;
    }
    std::env::set_var("GLM53_KDA_PREFILL_FUSED","0");
    std::fs::write(out.join("kda-prefill-probe.json"),serde_json::to_string_pretty(&cases).unwrap()).unwrap();
    eprintln!("[kda-prefill] PASS");
}

/// Item 3 timing: prefill recurrence alone (heads 32, T rows), mode 4 vs mode 5 variants, bitwise vs mode 4.
pub fn seq_probe() {
    let _g=tch::no_grad_guard();let dev=tch::Device::Cuda(0);tch::manual_seed(5);
    let (h,t)=(32i64,std::env::var("GLM53_PROBE_T").ok().and_then(|v|v.parse().ok()).unwrap_or(2048i64));
    let n=|s:&[i64]|Tensor::randn(s,(Kind::Float,dev));
    let nrm=|x:Tensor|{let s=(&x*&x).sum_dim_intlist(&[-1i64][..],true,Kind::Float).sqrt();x/s};
    let (q,k,v)=(nrm(n(&[t,h,128]))*0.088,nrm(n(&[t,h,128])),n(&[t,h,128]));
    let beta=Tensor::rand([t,h],(Kind::Float,dev));let decay=Tensor::rand([t,h,128],(Kind::Float,dev))*0.1+0.9;
    let init=n(&[h,128,128])*0.01;
    let run=|mode:&str,var:&str|->(Tensor,Tensor,f64){std::env::set_var("GLM53_KDA_SEQUENCE",mode);std::env::set_var("GLM53_KDA_SEQ5",var);
        let mut st=init.copy();let _=sequence(&mut st,&q,&k,&v,&beta,&decay);
        let mut ms=Vec::new();let mut o=Tensor::zeros([1],(Kind::Float,dev));
        for _ in 0..7 {st=init.copy();tch::Cuda::synchronize(0);let t0=std::time::Instant::now();o=sequence(&mut st,&q,&k,&v,&beta,&decay);tch::Cuda::synchronize(0);ms.push(t0.elapsed().as_secs_f64()*1000.);}
        ms.sort_by(|a,b|a.partial_cmp(b).unwrap());(o,st,ms[3])};
    let (o4,s4,t4)=run("4","41");println!("[kda-seq] T {t} mode4 {t4:.3} ms ({:.2} us/step)",t4*1000./t as f64);
    for var in ["24","81"] {let (o,s,tm)=run("5",var);
        println!("[kda-seq] mode5 var {var} {tm:.3} ms bitwise {}",o.equal(&o4)&&s.equal(&s4));}
    let rel=|a:&Tensor,b:&Tensor|f64::try_from((a-b).abs().max()/b.abs().max()).unwrap();
    for var in ["44","48","24","28","84","88"] {std::env::set_var("GLM53_KDA_SEQ6",var);let (o,s,tm)=run("6","41");
        println!("[kda-seq] mode6 var {var} {tm:.3} ms ({:.2} us/step) out rel {:.2e} state rel {:.2e}",tm*1000./t as f64,rel(&o,&o4),rel(&s,&s4));}
}

