// SPDX-License-Identifier: MIT
//! DSA 层 MLA(tch)— 对齐 engine/glm53/mla.py(M0 验收)。
//! 无 RoPE;kv_b 输出逐头交错拆 k/v;全因果注意力,scale = dk^-0.5。

use tch::{Kind, Tensor};

use crate::weights::{mm16, MlaWeights};

fn rmsnorm(x: &Tensor, w: &Tensor) -> Tensor {
    let v = x.to_kind(Kind::Float);
    let sq = &v * &v;
    let ms = sq.mean_dim(&[-1i64][..], true, Kind::Float);
    w.to_kind(Kind::Float) * (v * (ms + 1e-5).rsqrt())
}

pub fn mla_forward(w: &MlaWeights, x: &Tensor) -> Tensor {
    // x [T,4096] → [T,4096]
    let t = x.size()[0];
    let (hh, dk, dv) = (w.q_b.size()[0] / 256, 256i64, 256i64);

    let cq = rmsnorm(&mm16(x, &w.q_a), &w.q_a_ln); // [T,1536]
    let q = mm16(&cq, &w.q_b).view([t, hh, dk]);
    let ckv = rmsnorm(&mm16(x, &w.kv_a), &w.kv_a_ln); // [T,512]
    let kv = mm16(&ckv, &w.kv_b).view([t, hh, dk + dv]);
    let k = kv.narrow(2, 0, dk);
    let v = kv.narrow(2, dk, dv);

    // att [H,T,T] = q·kᵀ / sqrt(dk)
    let att = q
        .transpose(0, 1)
        .matmul(&k.permute([1, 2, 0]))
        / (dk as f64).powf(0.5);
    let causal = Tensor::ones([t, t], (Kind::Bool, x.device())).tril(0);
    let att = att.masked_fill(&causal.logical_not(), f64::NEG_INFINITY);
    let att = att.softmax(-1, Kind::Float);
    let o = att.matmul(&v.transpose(0, 1)).transpose(0, 1); // [T,H,dv]
    crate::weights::row_mm16(&o.reshape([t, hh * dv]), &w.wo)
}

// ───────────────────────── M1.3:单步解码(增量) ─────────────────────────

pub struct MlaState {
    pub k: Tensor, // [T,H,dk] 累积
    pub v: Tensor, // [T,H,dv]
    pub len: i64,
}

impl MlaState {
    pub fn new(dev: tch::Device) -> Self {
        Self::with_heads(dev, 64)
    }

    pub fn with_heads(dev: tch::Device, heads: i64) -> Self {
        Self {
            k: Tensor::zeros([0, heads, 256], (tch::Kind::Float, dev)),
            v: Tensor::zeros([0, heads, 256], (tch::Kind::Float, dev)),
            len: 0,
        }
    }
}

/// 单步:x [1,4096] → [1,4096];K/V 追加进缓存。
pub fn mla_step(w: &MlaWeights, x: &Tensor, st: &mut MlaState) -> Tensor {
    let (hh, dk, dv) = (w.q_b.size()[0] / 256, 256i64, 256i64);
    let cq = rmsnorm(&mm16(x, &w.q_a), &w.q_a_ln);
    let q = mm16(&cq, &w.q_b).view([1, hh, dk]);
    let ckv = rmsnorm(&mm16(x, &w.kv_a), &w.kv_a_ln);
    let kv = mm16(&ckv, &w.kv_b).view([1, hh, dk + dv]);
    let k_t = kv.narrow(2, 0, dk); // [1,H,dk]
    let v_t = kv.narrow(2, dk, dv);

    st.k = Tensor::cat(&[&st.k, &k_t], 0);
    st.v = Tensor::cat(&[&st.v, &v_t], 0);
    st.len += 1;

    // att [H,1,T] = q·Kᵀ/sqrt(dk),全历史(无掩码:新 token 看全部)
    let att = q
        .transpose(0, 1)
        .matmul(&st.k.permute([1, 2, 0]))
        / (dk as f64).powf(0.5);
    let att = att.softmax(-1, tch::Kind::Float);
    let o = att.matmul(&st.v.transpose(0, 1)).transpose(0, 1); // [1,H,dv]
    crate::weights::row_mm16(&o.reshape([1, hh * dv]), &w.wo)
}


/// 全序列前向 + KV 状态。
pub fn mla_forward_state(w: &MlaWeights, x: &Tensor, mut st: MlaState) -> (Tensor, MlaState) {
    let tt = x.size()[0];
    let mut outs = Vec::with_capacity(tt as usize);
    for ti in 0..tt {
        let o = mla_step(w, &x.get(ti).unsqueeze(0), &mut st);
        outs.push(o.squeeze_dim(0));
    }
    (Tensor::stack(&outs, 0), st)
}

// ─────────────────── M1②:图兼容 MLA 状态(定长窗口 + 原地写)───────────────────

/// 图兼容 MLA 状态:KV 预分配 [max_t,H,d] fp16,设备端 len 计数。
/// 注意:短上下文专用(max_t 默认 512);长上下文的正确解是压缩 latent 缓存(M2)。
pub struct MlaStateG {
    pub k: Tensor,   // [max_t,H,dk] fp16
    pub v: Tensor,   // [max_t,H,dv] fp16
    pub len: Tensor, // [1] i64 设备端(图内自增,掩码由此派生)
    pub max_t: i64,
}

impl MlaStateG {
    pub fn new(dev: tch::Device, max_t: i64) -> Self {
        Self::with_heads(dev, max_t, 64)
    }

    pub fn with_heads(dev: tch::Device, max_t: i64, heads: i64) -> Self {
        Self {
            k: Tensor::zeros([max_t, heads, 256], (Kind::Half, dev)),
            v: Tensor::zeros([max_t, heads, 256], (Kind::Half, dev)),
            len: Tensor::zeros([1], (tch::Kind::Int64, dev)),
            max_t,
        }
    }
    /// 从 MlaState(变长)迁入(prefill 后调用)。
    pub fn from_state(st: &MlaState, max_t: i64) -> Self {
        let dev = st.k.device();
        assert!(st.len <= max_t, "prefill 超出图窗口");
        let mut g = Self::with_heads(dev, max_t, st.k.size()[1]);
        let t = st.len;
        if t > 0 {
            let _ = g.k.narrow(0, 0, t).copy_(&st.k.narrow(0, 0, t).to_kind(Kind::Half));
            let _ = g.v.narrow(0, 0, t).copy_(&st.v.narrow(0, 0, t).to_kind(Kind::Half));
            let _ = g.len.fill_(t);
        }
        g
    }
}

/// 图兼容单步:KV 定长窗口 + 设备端 len 派生掩码;全窗口注意力(短上下文)。
pub fn mla_step_g(w: &MlaWeights, x: &Tensor, st: &mut MlaStateG) -> Tensor {
    let (hh, dk, dv) = (w.q_b.size()[0] / 256, 256i64, 256i64);
    let cq = rmsnorm(&mm16(x, &w.q_a), &w.q_a_ln);
    let q = mm16(&cq, &w.q_b).view([1, hh, dk]);
    let ckv = rmsnorm(&mm16(x, &w.kv_a), &w.kv_a_ln);
    let kv = mm16(&ckv, &w.kv_b).view([1, hh, dk + dv]);
    let k_t = kv.narrow(2, 0, dk).to_kind(Kind::Half); // [1,H,dk]
    let v_t = kv.narrow(2, dk, dv).to_kind(Kind::Half);

    // 设备端位置写入 + len 自增(图内,地址固定)
    let _ = st.k.index_copy_(0, &st.len, &k_t);
    let _ = st.v.index_copy_(0, &st.len, &v_t);
    let _ = st.len.copy_(&(&st.len + 1));

    // 全窗口注意力 + 掩码(arange < len)
    let kf = st.k.to_kind(Kind::Float);
    let vf = st.v.to_kind(Kind::Float);
    let att = q
        .transpose(0, 1)
        .matmul(&kf.permute([1, 2, 0]))
        / (dk as f64).powf(0.5); // [H,1,max_t]
    let mask = Tensor::arange(st.max_t, (tch::Kind::Int64, x.device()))
        .lt_tensor(&st.len); // [max_t] bool
    let att = att.masked_fill(&mask.logical_not().view([1, 1, st.max_t]), f64::NEG_INFINITY);
    let att = att.softmax(-1, Kind::Float);
    let o = att.matmul(&vf.transpose(0, 1)).transpose(0, 1); // [1,H,dv]
    crate::weights::row_mm16(&o.reshape([1, hh * dv]), &w.wo)
}
