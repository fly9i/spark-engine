// SPDX-License-Identifier: MIT
//! MoE 路由 + 专家池(tch)— 对齐 engine/glm53/moe.py(M0 验收)。
//! 路由:noaux_tc + sigmoid,top-8,renorm ×2.5;专家:clamp-SwiGLU(limit 10);
//! shared experts 直加。专家权重由 EXL3 trellis 按需解码(Rust 并行 inner + tch Hadamard)。

use std::collections::HashMap;

use tch::{Device, Kind, Tensor};

use crate::safetensors::ShardIndex;
use crate::weights::MoeMeta;

pub const ROUTED_SCALING: f64 = 2.5;
pub const SWIGLU_LIMIT: f64 = 10.0;

/// 路由:noaux_tc + sigmoid → (专家 id [T,k], 权重 [T,k])。
pub(crate) fn router_fused_enabled()->bool {std::env::var("GLM53_ROUTER_FUSED").as_deref()==Ok("1")}
/// route() plus, with GLM53_ROUTER_HALF_W=1 on the fused router, the Half RN copy of the weights written by the same
/// select kernel (for the cooperative expert kernel, which would otherwise convert them in its own launch).
pub(crate) fn route_h(h: &Tensor, w_gate: &Tensor, bias: &Tensor, topk: i64) -> (Tensor, Tensor, Option<Tensor>) {
    if std::env::var("GLM53_ROUTER_HALF_W").as_deref()==Ok("1") {
        let wh=Tensor::empty([h.size()[0],8],(Kind::Half,h.device()));
        HALF_W.with(|c|*c.borrow_mut()=Some(wh.shallow_clone()));
        let (ids,w)=route(h,w_gate,bias,topk);
        let used=HALF_W.with(|c|c.borrow_mut().take()).is_none();
        return (ids,w,used.then_some(wh));
    }
    let (ids,w)=route(h,w_gate,bias,topk);(ids,w,None)
}
/// GLM53_ROUTER_WIDE=1: the fused router also for 17..32 rows (batched multi-stream verify), per row the same
/// arithmetic as the <=16-row fused router (L1 against the ATen TF32 router those batches used before).
/// Only inside route_h_multi (the batched multi-stream verify): prefill windows of 17..32 rows keep the ATen router.
fn router_wide()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    *E.get_or_init(||std::env::var("GLM53_ROUTER_WIDE").as_deref()==Ok("1")) && MULTI_VERIFY.with(|c|c.get())}
thread_local!{static MULTI_VERIFY:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}
pub(crate) fn route_h_multi(h:&Tensor,w_gate:&Tensor,bias:&Tensor,topk:i64)->(Tensor,Tensor,Option<Tensor>) {
    MULTI_VERIFY.with(|c|c.set(true));let r=route_h(h,w_gate,bias,topk);MULTI_VERIFY.with(|c|c.set(false));r
}
thread_local!{static HALF_W:std::cell::RefCell<Option<Tensor>>=const{std::cell::RefCell::new(None)};}
/// GLM53_MOE_ROUTE_SIDE=1: whether route() takes the fused router kernels (no cuBLAS) for these inputs, so it may run on a
/// pool side stream. The same conditions as route()'s fused branch for at most 16 rows.
pub(crate) fn route_side_ok(h:&Tensor,w_gate:&Tensor,bias:&Tensor)->bool {route_side_ok_rows(h,w_gate,bias,16)}
/// route_h_multi: up to 32 rows with GLM53_ROUTER_WIDE (its fused wide router), else 16.
pub(crate) fn route_side_ok_multi(h:&Tensor,w_gate:&Tensor,bias:&Tensor)->bool {
    let wide=std::env::var("GLM53_ROUTER_WIDE").as_deref()==Ok("1");route_side_ok_rows(h,w_gate,bias,if wide{32}else{16})
}
fn route_side_ok_rows(h:&Tensor,w_gate:&Tensor,bias:&Tensor,max:i64)->bool {
    std::env::var("GLM53_MOE_ROUTE_SIDE").as_deref()==Ok("1") && router_fused_enabled() && h.device().is_cuda() && h.kind()==Kind::Float && h.dim()==2
        && (1..=max).contains(&h.size()[0]) && h.is_contiguous() && [Kind::Float,Kind::BFloat16].contains(&w_gate.kind()) && w_gate.is_contiguous()
        && bias.kind()==Kind::Float && bias.is_contiguous() && w_gate.size()[1]==h.size()[1] && h.size()[1]%256==0 && w_gate.size()[0]<=320
        && bias.numel() as i64==w_gate.size()[0] && !crate::evaluation::replay_active() && !crate::deep_probe::router_hooks_active()
}
pub fn route(h: &Tensor, w_gate: &Tensor, bias: &Tensor, topk: i64) -> (Tensor, Tensor) {
    if router_fused_enabled() && topk==8 && h.device().is_cuda() && h.kind()==Kind::Float && h.dim()==2 && (1..=if router_wide(){32}else{16}).contains(&h.size()[0])
        && h.is_contiguous() && [Kind::Float,Kind::BFloat16].contains(&w_gate.kind()) && w_gate.is_contiguous() && bias.kind()==Kind::Float && bias.is_contiguous()
        && w_gate.size()[1]==h.size()[1] && h.size()[1]%256==0 && w_gate.size()[0]<=320 && bias.numel() as i64==w_gate.size()[0]
        && !crate::evaluation::replay_active() && !crate::deep_probe::router_hooks_active() {
        // W05: two launches replace GEMM+splitK+sigmoid+add+topk+sort+gather+sum+div+scale (L1: FP32
        // FMA logits instead of TF32; ties resolve to the lower expert id).
        let (t,e,k)=(h.size()[0],w_gate.size()[0],h.size()[1]);let dev=h.device();
        extern "C"{fn glm53_router_split()->i32;}
        let partial=Tensor::empty([t,e,unsafe{glm53_router_split()} as i64],(Kind::Float,dev));
        let ids=Tensor::empty([t,8],(Kind::Int64,dev));let w=Tensor::empty([t,8],(Kind::Float,dev));
        extern "C"{fn rs_router_fused(h:*const f32,w:*const std::ffi::c_void,w_bf16:i32,bias:*const f32,partial:*mut f32,ids:*mut i64,weights:*mut f32,rows:i32,experts:i32,k:i32,scaling:f32)->i32;}
        if let Some(wh)=HALF_W.with(|c|c.borrow_mut().take()) {
            extern "C"{fn rs_router_fused_h(h:*const f32,w:*const std::ffi::c_void,w_bf16:i32,bias:*const f32,partial:*mut f32,ids:*mut i64,weights:*mut f32,wh:*mut std::ffi::c_void,rows:i32,experts:i32,k:i32,scaling:f32)->i32;}
            assert_eq!(unsafe{rs_router_fused_h(h.data_ptr().cast(),w_gate.data_ptr(),i32::from(w_gate.kind()==Kind::BFloat16),bias.data_ptr().cast(),partial.data_ptr().cast(),ids.data_ptr().cast(),w.data_ptr().cast(),
                wh.data_ptr(),t as i32,e as i32,k as i32,ROUTED_SCALING as f32)},0,"fused router (half weights)");
        } else {
        assert_eq!(unsafe{rs_router_fused(h.data_ptr().cast(),w_gate.data_ptr(),i32::from(w_gate.kind()==Kind::BFloat16),bias.data_ptr().cast(),partial.data_ptr().cast(),ids.data_ptr().cast(),w.data_ptr().cast(),
            t as i32,e as i32,k as i32,ROUTED_SCALING as f32)},0,"fused router");
        }
        if t>16 && crate::kda::wide_check_on() {
            let a=route(&h.narrow(0,0,16),w_gate,bias,topk);let b=route(&h.narrow(0,16,t-16),w_gate,bias,topk);
            crate::kda::wide_check("router-ids",&ids,&Tensor::cat(&[a.0,b.0],0));crate::kda::wide_check("router-w",&w,&Tensor::cat(&[a.1,b.1],0));
        }
        crate::evaluation::record_route(&ids,&w);
        return (ids,w);
    }
    let logits = h.to_kind(Kind::Float).matmul(&w_gate.to_kind(Kind::Float).transpose(0, 1)); // [T,E] (M4: BF16 widens exactly)
    let scores = logits.sigmoid();
    let sel = &scores + bias;
    crate::deep_probe::record_router(h,&sel);
    let (topi, fixed_weights) = crate::evaluation::replay_route(sel.topk(topk, -1, true, true).1);
    crate::deep_probe::record_selection(&topi);
    let w = fixed_weights.unwrap_or_else(|| {
        let w = scores.gather(-1, &topi, false);
        let w = &w / w.sum_dim_intlist(&[-1i64][..], true, Kind::Float).clamp_min(1e-20);
        w * ROUTED_SCALING
    });
    crate::evaluation::record_route(&topi, &w);
    (topi, w)
}

/// 单专家:clamp-SwiGLU。x [T,in];W 为 K-major [in,out]。
pub fn expert_forward(x: &Tensor, wg: &Tensor, wu: &Tensor, wd: &Tensor) -> Tensor {
    let g = x.matmul(wg).clamp(f64::NEG_INFINITY, SWIGLU_LIMIT);
    let u = x.matmul(wu).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
    (g.silu() * u).matmul(wd)
}

/// shared experts(线性权重全量驻留)。
pub fn shared_forward(m: &MoeMeta, h: &Tensor) -> Tensor {
    crate::weights::row_mm16(&shared_activation(m,h), &m.sh_wd)
}

fn shared_activation(m:&MoeMeta,h:&Tensor)->Tensor {
    shared_activation_with_half(m,h,None)
}

pub(crate) fn shared_activation_with_half(m:&MoeMeta,h:&Tensor,half:Option<&Tensor>)->Tensor {
    if let Some(y)=crate::shared_gu::try_activation(h,&m.sh_wg,&m.sh_wu,&m.sh_wd,half){return y;}
    // D7 (GLM53_SHARED_GU_ROWS=1): FP8 gate and up in one row-concatenated skinny launch (the KDA QKV
    // mechanism: per-row split-K layout does not depend on N, each half equals its separate call).
    if std::env::var("GLM53_SHARED_GU_ROWS").as_deref()==Ok("1") {
        if let Some(y)=crate::dense_fp8::try_run_rows(h,&[&m.sh_wg,&m.sh_wu]) {
            let n=m.sh_wg.size()[0];
            let g=y.narrow(1,0,n).clamp(f64::NEG_INFINITY, SWIGLU_LIMIT);
            let u=y.narrow(1,n,n).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
            return g.silu()*u;
        }
    }
    if let Some(y)=crate::c12::try_run_rows(h,&[&m.sh_wg,&m.sh_wu]) {
        let n=m.sh_wg.size()[0];
        let g=y.narrow(1,0,n).clamp(f64::NEG_INFINITY, SWIGLU_LIMIT);
        let u=y.narrow(1,n,n).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        return g.silu()*u;
    }
    let g = crate::weights::mm16_with_half(h, &m.sh_wg,half).clamp(f64::NEG_INFINITY, SWIGLU_LIMIT);
    let u = crate::weights::mm16_with_half(h, &m.sh_wu,half).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
    g.silu()*u
}

pub(crate) fn shared_half_enabled()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_PREFILL_SHARED_HALF").as_deref()==Ok("1"))}
/// GLM53_PREFILL_SHARED_HALF (L0): mm16_partial(shared_activation_with_half(m, x, Some(xh)), sh_wd) for prefill rows on
/// the plain Half GEMM path, without the FP32 gate/up tensors: Half GEMMs, one SwiGLU kernel writing the Half input of
/// the down GEMM. None when any projection would take another backend.
pub(crate) fn shared_partial_half(m:&MoeMeta,xh:&Tensor)->Option<Tensor> {
    if !shared_half_enabled() || xh.kind()!=Kind::Half || !xh.is_contiguous() {return None;}
    if !crate::weights::plain_half_prefill(xh,&m.sh_wg) || !crate::weights::plain_half_prefill(xh,&m.sh_wu) {return None;}
    let g=xh.matmul(&m.sh_wg.transpose(0,1));let u=xh.matmul(&m.sh_wu.transpose(0,1));
    if !g.is_contiguous() || !u.is_contiguous() {return None;}
    let act=Tensor::empty_like(&g);
    if !crate::weights::plain_half_prefill(&act,&m.sh_wd) {return None;}
    extern "C"{fn rs_shared_swiglu_hh(g:*const std::ffi::c_void,u:*const std::ffi::c_void,out:*mut std::ffi::c_void,n:i64,lim:f32)->i32;}
    assert_eq!(unsafe{rs_shared_swiglu_hh(g.data_ptr(),u.data_ptr(),act.data_ptr(),g.numel() as i64,SWIGLU_LIMIT as f32)},0,"shared swiglu");
    Some(crate::weights::mm16_partial(&act,&m.sh_wd))
}
pub(crate) fn input_half_reuse_enabled()->bool {
    match std::env::var("GLM53_MOE_INPUT_HALF_REUSE") {
        Ok(v)=>match v.as_str(){"1"=>true,"0"=>false,_=>panic!("GLM53_MOE_INPUT_HALF_REUSE must be 0 or 1")},
        Err(std::env::VarError::NotPresent)=>false,Err(_)=>panic!("GLM53_MOE_INPUT_HALF_REUSE must be 0 or 1"),
    }
}

pub fn tp_pack_enabled()->bool {
    crate::tp::world().world==2 && crate::tp::dense_enabled() &&
        std::env::var("GLM53_TP_MOE_PACK").as_deref()==Ok("1")
}

fn tp_pack_for_rows(rows:i64)->bool {
    if !tp_pack_enabled(){return false;}
    let max_rows=std::env::var("GLM53_TP_MOE_PACK_MAX_ROWS").ok()
        .map(|s|s.parse::<i64>().expect("TP pack max rows must be an integer")).unwrap_or(if crate::forward::verify_invariant(){32}else{8});   // proposal 3: one packed layout for every verify batch
    assert!((1..=4096).contains(&max_rows),"TP pack max rows outside 1..4096");
    rows<=max_rows
}

/// Two independent TP partial sums, one collective. Never sum the local
/// routed/shared lanes together: shared output rounds only AFTER reduction.
pub fn finish_tp(routed:Tensor,m:&MoeMeta,h:&Tensor)->Tensor {
    finish_tp_with_half(routed,m,h,None)
}

/// Sequence-parallel prefill: routed + shared FP32 partials, reduce-scattered (this rank's rows).
pub(crate) fn sp_finish(routed:Tensor,m:&MoeMeta,h:&Tensor)->Tensor {sp_finish_with_half(routed,m,h,None)}
pub(crate) fn sp_finish_with_half(routed:Tensor,m:&MoeMeta,h:&Tensor,half:Option<&Tensor>)->Tensor {
    let shared=crate::weights::mm16_partial(&shared_activation_with_half(m,h,half),&m.sh_wd);
    crate::tp::reduce_scatter_rows(&(routed+shared).contiguous())
}
fn finish_tp_with_half(routed:Tensor,m:&MoeMeta,h:&Tensor,half:Option<&Tensor>)->Tensor {
    // P6 (GLM53_PREFILL_MOE_SUM1=1, prefill rows > 64): add the shared expert's FP32 partial to the
    // routed partial and reduce once (one 32 MB collective instead of two per MoE layer). Drops the
    // shared lane's Half rounding and changes summation order (L1).
    if routed.size()[0]>64 && crate::tp::is_tp() && std::env::var("GLM53_PREFILL_MOE_SUM1").as_deref()==Ok("1") {
        let shared=crate::weights::mm16_partial(&shared_activation_with_half(m,h,half),&m.sh_wd);
        let y=routed+shared;crate::tp::allreduce(&y);return y;
    }
    if !tp_pack_for_rows(routed.size()[0]) {
        crate::tp::allreduce(&routed);
        return routed+crate::weights::row_mm16(&shared_activation_with_half(m,h,half),&m.sh_wd);
    }
    let rows=routed.size()[0];
    let packed=tp_packed_sum(routed,m,h,half);
    packed.narrow(0,0,rows)+crate::weights::round_row_output(packed.narrow(0,rows,rows),&m.sh_wd)
}

/// GLM53_MOE_ROUTE_SIDE, unpacked rows (tp_pack_for_rows false): finish_tp_post_with_half with the shared expert's
/// activation computed earlier by the caller (shared_activation_with_half of the same inputs: the same values).
pub(crate) fn unpacked_rows(rows:i64)->bool {!crate::mhc::packed_enabled() || !tp_pack_for_rows(rows)}
pub(crate) fn finish_unpacked_with_act(routed:Tensor,m:&MoeMeta,act:&Tensor,residual:&Tensor,pre:&crate::mhc::PreOut)->Tensor {
    assert!(unpacked_rows(routed.size()[0]) && routed.size()[0]<=64);
    crate::tp::allreduce(&routed);
    crate::mhc::mhc_post(&(routed+crate::weights::row_mm16(act,&m.sh_wd)),residual,pre)
}
/// I3 step 3 (GLM53_MOE_PACK_DIRECT=1, needs the fused allreduce post): rows where finish_tp_post_with_half would take
/// the packed send-only path, so the caller may let the routed kernel fill the first half of the packed buffer.
pub(crate) fn pack_direct_eligible(rows:i64)->bool {
    std::env::var("GLM53_MOE_PACK_DIRECT").as_deref()==Ok("1") && crate::tp::ar_fused_enabled() && crate::mhc::packed_enabled()
        && crate::mhc::four_streams_enabled() && tp_pack_for_rows(rows) && (1..=32).contains(&rows)
        && !crate::root_probe::retain_f32() && !crate::root_probe::full_f32() && !crate::root_probe::recording()
}
/// `packed` [2*rows, H] FP32 whose first half already holds the routed partial: the shared down projection writes the
/// second half in place, then the same send-only collective and fused packed post as finish_tp_post_with_half.
pub(crate) fn finish_packed_direct(packed:Tensor,m:&MoeMeta,h:&Tensor,half:Option<&Tensor>,residual:&Tensor,pre:&crate::mhc::PreOut)->Tensor {
    shared_into_packed(&packed,m,h,half);
    finish_packed_after_shared(packed,m,residual,pre)
}
/// The shared expert's FP32 partial into the second half of the packed collective buffer (finish_packed_direct's first step).
pub(crate) fn shared_into_packed(packed:&Tensor,m:&MoeMeta,h:&Tensor,half:Option<&Tensor>) {
    let rows=packed.size()[0]/2;
    crate::weights::mm16_partial_into(&shared_activation_with_half(m,h,half),&m.sh_wd,&packed.narrow(0,rows,rows));
}
/// finish_packed_direct after both halves of `packed` are written: the packed collective and packed MHC post.
pub(crate) fn finish_packed_after_shared(packed:Tensor,m:&MoeMeta,residual:&Tensor,pre:&crate::mhc::PreOut)->Tensor {
    let round_shared=m.sh_wd.kind()==Kind::Half && !crate::root_probe::retain_f32();
    if crate::tp::allreduce_send(&packed) {return crate::mhc::mhc_post_packed_pending(&packed,residual,pre,round_shared);}
    crate::tp::allreduce(&packed);crate::mhc::mhc_post_packed(&packed,residual,pre,round_shared)
}

/// Produce the existing two-lane collective layout once, for either consumer.
/// The shared lane must retain FP32 until the collective has completed.
fn tp_packed_sum(routed:Tensor,m:&MoeMeta,h:&Tensor,half:Option<&Tensor>)->Tensor {
    let packed=tp_packed_partial(routed,m,h,half);
    crate::tp::allreduce(&packed);
    packed
}
fn tp_packed_partial(routed:Tensor,m:&MoeMeta,h:&Tensor,half:Option<&Tensor>)->Tensor {
    let shared=crate::weights::mm16_partial(&shared_activation_with_half(m,h,half),&m.sh_wd);
    assert_eq!(routed.kind(),Kind::Float);assert_eq!(shared.kind(),Kind::Float);
    assert_eq!(routed.size(),shared.size());
    Tensor::cat(&[routed,shared],0)
}

/// D2: side-stream schedule for the shared expert. Eligible exactly where the packed two-lane
/// collective + packed MHC post would be used (so the joined path is the same arithmetic).
pub(crate) fn shared_stream_eligible(rows:i64)->bool {
    std::env::var("GLM53_MOE_SHARED_STREAM").as_deref()==Ok("1") && crate::mhc::packed_enabled() && tp_pack_for_rows(rows)
        && !crate::root_probe::retain_f32() && !crate::root_probe::full_f32() && !crate::root_probe::recording()
}
/// The side stream comes from the process stream pool, shared by every graph; it must never run a
/// cuBLAS call (PyTorch keys cuBLAS workspaces by stream, see the illegal-address investigation).
/// Only the FP8 shared-expert path (own kernels for <= 16 rows) qualifies.
pub(crate) fn shared_stream_kernels_ok(m:&MoeMeta,rows:i64)->bool {
    rows<=16 && [&m.sh_wg,&m.sh_wu,&m.sh_wd].iter().all(|w|crate::dense_fp8::registered_enabled(w))
        && std::env::var("GLM53_SHARED_GU_FUSED").as_deref()!=Ok("1")
}
pub(crate) struct SideShared(Tensor);
/// Fork one side stream from the current stream, compute the shared FP32 partial there, and return
/// to the main stream. The caller must pass the result to `finish_packed_post_joined` (which joins).
/// Side-allocated blocks are only reused by later side work, which is always ordered after a new
/// fork (main -> side), so the main-stream consumers of this output finish first.
pub(crate) fn shared_partial_on_side(m:&MoeMeta,h:&Tensor,half:Option<&Tensor>)->SideShared {
    extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;}
    assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);
    let shared=crate::weights::mm16_partial(&shared_activation_with_half(m,h,half),&m.sh_wd);
    assert_eq!(unsafe{rs_stream_set(-1)},0);
    SideShared(shared)
}
/// Join the side stream, then the same packed collective and packed MHC post as
/// finish_tp_post_with_half (cat order routed|shared, shared rounded after the reduction).
pub(crate) fn finish_packed_post_joined(routed:Tensor,shared:SideShared,m:&MoeMeta,residual:&Tensor,pre:&crate::mhc::PreOut)->Tensor {
    extern "C"{fn rs_stream_join(n:i32)->i32;}
    assert_eq!(unsafe{rs_stream_join(1)},0);
    let shared=shared.0;
    assert_eq!(routed.kind(),Kind::Float);assert_eq!(shared.kind(),Kind::Float);assert_eq!(routed.size(),shared.size());
    let packed=Tensor::cat(&[routed,shared],0);
    let round_shared=m.sh_wd.kind()==Kind::Half && !crate::root_probe::retain_f32();
    if crate::mhc::four_streams_enabled() && crate::tp::allreduce_send(&packed) {return crate::mhc::mhc_post_packed_pending(&packed,residual,pre,round_shared);}
    crate::tp::allreduce(&packed);
    crate::mhc::mhc_post_packed(&packed,residual,pre,round_shared)
}

/// Verifier consumer: perform the original shared Half rounding and routed
/// addition inside MHC post, eliminating their otherwise unused intermediates.
pub fn finish_tp_post(routed:Tensor,m:&MoeMeta,h:&Tensor,residual:&Tensor,pre:&crate::mhc::PreOut)->Tensor {
    finish_tp_post_with_half(routed,m,h,None,residual,pre)
}

pub(crate) fn finish_tp_post_with_half(routed:Tensor,m:&MoeMeta,h:&Tensor,half:Option<&Tensor>,residual:&Tensor,pre:&crate::mhc::PreOut)->Tensor {
    if !crate::mhc::packed_enabled() ||
        !h.device().is_cuda() || !tp_pack_for_rows(routed.size()[0]) {
        return crate::mhc::mhc_post(&finish_tp_with_half(routed,m,h,half),residual,pre);
    }
    let round_shared=m.sh_wd.kind()==Kind::Half && !crate::root_probe::retain_f32();
    if crate::tp::ar_fused_enabled() && crate::mhc::four_streams_enabled() {
        let packed=tp_packed_partial(routed,m,h,half);
        if crate::tp::allreduce_send(&packed) {return crate::mhc::mhc_post_packed_pending(&packed,residual,pre,round_shared);}
        crate::tp::allreduce(&packed);return crate::mhc::mhc_post_packed(&packed,residual,pre,round_shared);
    }
    let packed=tp_packed_sum(routed,m,h,half);
    crate::mhc::mhc_post_packed(&packed,residual,pre,round_shared)
}

/// 专家驻留池:EXL3 trellis → fp16 tch 权重,LRU(条目数上限)。
pub struct ExpertPool {
    idx: ShardIndex,
    table: Vec<u16>, // MCG 码本
    cache: HashMap<(usize, usize), (Tensor, Tensor, Tensor)>,
    order: std::collections::VecDeque<(usize, usize)>,
    pub cap: usize,
    pub decoded: u64,
    pub hits: u64,
}

impl ExpertPool {
    pub fn new(dir: &std::path::Path, cap: usize) -> Self {
        let idx = ShardIndex::scan(dir).expect("scan");
        Self {
            idx,
            table: crate::exl3::mcg_table().to_vec(),
            cache: HashMap::new(),
            order: Default::default(),
            cap,
            decoded: 0,
            hits: 0,
        }
    }

    /// 解码单个投影:全 GPU 链(镜像 exl3_torch.decode_weight_t)。
    /// trellis u16 位 → device int64 → u32 组装 → nibble 抽取 → 词组装 →
    /// F 查表 → Hadamard(K 中间维转置右乘)×suh×Hadamard(N)×svh。
    fn decode_proj(&mut self, prefix: &str, dev: Device) -> Tensor {
        use tch::Kind;
        let trellis = self.idx.get_i16(&format!("{prefix}.trellis")).expect("trellis");
        let shape = self.idx.entries[&format!("{prefix}.trellis")].shape.clone();
        let (kt, nt) = (shape[0], shape[1]);
        let (suh, _) = self.idx.get_f32(&format!("{prefix}.suh")).expect("suh");
        let (svh, _) = self.idx.get_f32(&format!("{prefix}.svh")).expect("svh");

        // trellis i16 → device(位模式保持,后续按位组装)
        let tr = Tensor::from_slice(&trellis).to_device(dev).to_kind(Kind::Int);
        let tr = tr.reshape([kt as i64, nt as i64, 64]);
        let w16 = tr.bitwise_and(0xFFFF);
        let u32s = w16.slice(2, 0, 64, 2)
            .bitwise_or_tensor(&w16.slice(2, 1, 64, 2).bitwise_left_shift(&Tensor::from(16))); // [Kt,Nt,32]

        // nibble:[Kt,Nt,256]
        let pos = Tensor::arange(256, (Kind::Int64, dev));
        let sh: Tensor = Tensor::from(28i64) - 4i64 * pos.remainder(8i64);
        let idx8 = pos.bitwise_right_shift(&Tensor::from(3i64)).unsqueeze(0).unsqueeze(0).expand([kt as i64, nt as i64, 256], false);
        let nib = u32s
            .gather(2, &idx8, false)
            .bitwise_right_shift(&sh.unsqueeze(0).unsqueeze(0))
            .bitwise_and(0xF);

        // NIBMAP [256] 拍平到 device:cell(r,c) 在输出 (k*16+r, n*16+c) 处,
        // 生成 [Kt,Nt,16,16] 的 4-nibble 索引矩阵
        let mut src_flat = Vec::with_capacity(256 * 4);
        for r in 0..16 {
            for c in 0..16 {
                for s in 0..4 {
                    src_flat.push(crate::nibmap::NIBMAP[r][c][s] as i64);
                }
            }
        }
        let src4 = Tensor::from_slice(&src_flat)
            .view([1, 1, 256, 4])
            .to_device(dev);
        let mk_idx = |s: i64| -> Tensor {
            src4.select(3, s)
                .expand([kt as i64, nt as i64, 256], false)
        };
        let word = &nib.gather(2, &mk_idx(0), false)
            + &nib.gather(2, &mk_idx(1), false).bitwise_left_shift(&Tensor::from(4))
            + &nib.gather(2, &mk_idx(2), false).bitwise_left_shift(&Tensor::from(8))
            + &nib.gather(2, &mk_idx(3), false).bitwise_left_shift(&Tensor::from(12)); // [Kt,Nt,256]

        // F 表查值:word [Kt,Nt,256] → flat gather
        let f_dev = Tensor::from_slice(&{
            let mut v = Vec::with_capacity(65536);
            for b in &self.table {
                v.push(half::f16::from_bits(*b).to_f32());
            }
            v
        })
        .to_device(dev);
        let inner = f_dev
            .index_select(0, &word.reshape([-1]))
            .reshape([kt as i64, nt as i64, 16, 16])
            .permute([0, 2, 1, 3])
            .reshape([(kt * 16) as i64, (nt * 16) as i64])
            .to_kind(Kind::Float);

        let suh_t = Tensor::from_slice(&suh).to_kind(Kind::Float).to_device(dev);
        let svh_t = Tensor::from_slice(&svh).to_kind(Kind::Float).to_device(dev);
        let h128 = hadamard_128(dev);
        let (kk, nn) = (inner.size()[0], inner.size()[1]);
        let w = inner
            .view([kk / 128, 128, nn])
            .transpose(1, 2)
            .matmul(&h128.transpose(0, 1))
            .transpose(1, 2)
            .reshape([kk, nn]);
        let w = w * suh_t.unsqueeze(1);
        let w = w
            .view([kk, nn / 128, 128])
            .matmul(&h128)
            .reshape([kk, nn]);
        &w * svh_t.unsqueeze(0)
    }

    pub fn expert(&mut self, layer: usize, e: usize, dev: Device) -> &(Tensor, Tensor, Tensor) {
        let key = (layer, e);
        if self.cache.contains_key(&key) {
            self.hits += 1;
            // LRU:移到队尾
            self.order.retain(|k| k != &key);
            self.order.push_back(key);
        } else {
            let p = format!("model.language_model.layers.{layer}.mlp.experts.{e}");
            let wg = self.decode_proj(&format!("{p}.gate_proj"), dev);
            let wu = self.decode_proj(&format!("{p}.up_proj"), dev);
            let wd = self.decode_proj(&format!("{p}.down_proj"), dev);
            self.decoded += 1;
            if self.cache.len() >= self.cap {
                if let Some(old) = self.order.pop_front() {
                    self.cache.remove(&old);
                }
            }
            self.cache.insert(key, (wg, wu, wd));
            self.order.push_back(key);
        }
        &self.cache[&key]
    }
}

/// 原生专家池:每专家 3 个投影的 (trellis, suh, svh) 驻 GPU。
/// 口径:单机整专家 ≈ 12MiB(trellis 3×4MiB + 尺度);6.02MiB 是 TP2 每节点半份,
/// 单机 cap 预算必须按 12MiB 算(cap 2048 ≈ 24GiB)。经 exl3_gemm_gr 直算(算力换带宽主路径)。
pub struct NativeExpertPool {
    idx: ShardIndex,
    cache: HashMap<(usize, usize), Vec<(Tensor, Tensor, Tensor)>>, // [gate, up, down]
    order: std::collections::VecDeque<(usize, usize)>,
    pub cap: usize,
}

extern "C" {
    fn rs_exl3_gemm(x_p: *const u8, y_p: *mut u8, xh_p: *mut u8,
                    tr_p: *const u8, suh_p: *const u8, svh_p: *const u8,
                    rows: i64, k: i64, n: i64, kt: i64, nt: i64) -> i32;
}

impl NativeExpertPool {
    pub fn new(dir: &std::path::Path, cap: usize) -> Self {
        Self { idx: ShardIndex::scan(dir).expect("scan"), cache: HashMap::new(), order: Default::default(), cap }
    }

    pub fn expert(&mut self, layer: usize, e: usize, dev: Device) -> &Vec<(Tensor, Tensor, Tensor)> {
        if !self.cache.contains_key(&(layer, e)) {
            let p = format!("model.language_model.layers.{layer}.mlp.experts.{e}");
            let mut v = Vec::with_capacity(3);
            for w in ["gate_proj", "up_proj", "down_proj"] {
                let raw = self.idx.get_i16(&format!("{p}.{w}.trellis")).expect("tr");
                let s = self.idx.entries[&format!("{p}.{w}.trellis")].shape.clone();
                let (suh, _) = self.idx.get_f32(&format!("{p}.{w}.suh")).expect("suh");
                let (svh, _) = self.idx.get_f32(&format!("{p}.{w}.svh")).expect("svh");
                v.push((
                    Tensor::from_slice(&raw).view([s[0] as i64, s[1] as i64, 64]).to_device(dev),
                    Tensor::from_slice(&suh).to_kind(Kind::Half).to_device(dev),
                    Tensor::from_slice(&svh).to_kind(Kind::Half).to_device(dev),
                ));
            }
            if self.cache.len() >= self.cap {
                if let Some(old) = self.order.pop_front() {
                    self.cache.remove(&old);
                }
            }
            self.cache.insert((layer, e), v);
            self.order.push_back((layer, e));
        }
        &self.cache[&(layer, e)]
    }

    /// 直算单投影:x [R,K] fp16 → y [R,N] fp16(预分配)。xh 为工作区。
    pub fn gemm(x: &Tensor, xh: &Tensor, y: &Tensor, proj: &(Tensor, Tensor, Tensor)) -> bool {
        let (tr, suh, svh) = proj;
        let (kt, nt) = (tr.size()[0], tr.size()[1]);
        let (k, n) = (kt * 16, nt * 16);
        let rows = x.size()[0];
        let rc = unsafe {
            rs_exl3_gemm(x.data_ptr() as *const u8, y.data_ptr() as *mut u8, xh.data_ptr() as *mut u8,
                         tr.data_ptr() as *const u8, suh.data_ptr() as *const u8, svh.data_ptr() as *const u8,
                         rows, k, n, kt, nt)
        };
        rc == 0
    }

    /// 完整 clamp-SwiGLU(与 expert_forward 语义一致,fp16 出 → fp32)。
    pub fn expert_forward(x16: &Tensor, projs: &[(Tensor, Tensor, Tensor); 3], dev: Device) -> Tensor {
        let rows = x16.size()[0];
        let (kg, ng) = { let (kt, nt) = (projs[0].0.size()[0], projs[0].0.size()[1]); (kt * 16, nt * 16) };
        let (ku, nu) = { let (kt, nt) = (projs[1].0.size()[0], projs[1].0.size()[1]); (kt * 16, nt * 16) };
        let (kd, nd) = { let (kt, nt) = (projs[2].0.size()[0], projs[2].0.size()[1]); (kt * 16, nt * 16) };
        let xh = Tensor::empty_like(x16);
        let g = Tensor::zeros([rows, ng], (Kind::Half, dev));
        let u = Tensor::zeros([rows, nu], (Kind::Half, dev));
        let out = Tensor::zeros([rows, nd], (Kind::Half, dev));
        assert!(Self::gemm(x16, &xh, &g, &projs[0]), "gate gemm");
        assert!(Self::gemm(x16, &xh, &u, &projs[1]), "up gemm");
        // clamp-SwiGLU:gate 只 max-clamp,up ±clamp(与 Python 一致)
        let g = g.to_kind(Kind::Float).clamp(f64::NEG_INFINITY, SWIGLU_LIMIT);
        let u = u.to_kind(Kind::Float).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        let act = (g.silu() * u).to_kind(Kind::Half);
        assert!(Self::gemm(&act, &xh, &out, &projs[2]), "down gemm");
        out.to_kind(Kind::Float)
    }
}

fn hadamard_128(dev: Device) -> Tensor {
    // 与 exl3_torch._tables 相同:Sylvester 构造 / sqrt(128)
    let mut h = Tensor::ones([1, 1], (Kind::Float, Device::Cpu));
    while h.size()[0] < 128 {
        h = Tensor::cat(
            &[
                Tensor::cat(&[&h, &h], 1),
                Tensor::cat(&[&h, &(-&h)], 1),
            ],
            0,
        );
    }
    (h / 128f64.powf(0.5)).to_device(dev)
}

/// Real shared weights, rank-distinct routed inputs, and changed-input graphs.
/// Inputs are copied on every invocation, so repeated SUM cannot overflow them.
pub fn tp_pack_probe(model:&std::path::Path,out:&std::path::Path) {
    use serde_json::json;
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);
    assert!(crate::tp::dense_enabled());let dev=Device::Cuda(0);
    tch::manual_seed(20260924);std::fs::create_dir_all(out).unwrap();
    let idx=ShardIndex::scan(model).unwrap();
    let load=|proj:&str,axis:i64| {
        let (v,s)=idx.get_f32(&format!("model.language_model.layers.3.mlp.shared_experts.{proj}.weight")).unwrap();
        let shape:Vec<i64>=s.iter().map(|&v|v as i64).collect();
        let full=Tensor::from_slice(&v).view(shape.as_slice()).to_kind(Kind::Half);
        let n=shape[axis as usize]/2;
        full.narrow(axis,tp.rank as i64*n,n).contiguous().to_device(dev)
    };
    let m=MoeMeta{w_gate:Tensor::new(),bias:Tensor::new(),sh_wg:load("gate_proj",0),
        sh_wu:load("up_proj",0),sh_wd:load("down_proj",1)};
    let mut cases=Vec::new();
    for rows in [1,2,8,16,32,128,2048] {
        let x=Tensor::randn([rows,4096],(Kind::Float,dev));
        let r=Tensor::randn([rows,4096],(Kind::Float,dev))*(tp.rank+1) as f64;
        let mut input=x.copy();let mut routed=r.copy();let mut rounds=Vec::new();
        for on in [false,true,true,false] {
            std::env::set_var("GLM53_TP_MOE_PACK",if on{"1"}else{"0"});
            for _ in 0..3 {let _=finish_tp(routed.copy(),&m,&input);}
            tch::Cuda::synchronize(0);crate::tp::graph::begin().unwrap();
            let y=finish_tp(routed.copy(),&m,&input);crate::tp::graph::end().unwrap();
            for scale in [1.,-1.,0.125,0.] {
                input.copy_(&(&x*scale));routed.copy_(&(&r*scale));
                crate::tp::graph::replay().unwrap();
                std::env::set_var("GLM53_TP_MOE_PACK","0");
                let gold=finish_tp(routed.copy(),&m,&input);
                assert!(y.isfinite().all().int64_value(&[])!=0);
                assert!(y.equal(&gold),"packing differs rows={rows} on={on} scale={scale} max={}",(&y-&gold).abs().max().double_value(&[]));
                std::env::set_var("GLM53_TP_MOE_PACK",if on{"1"}else{"0"});
            }
            input.copy_(&x);routed.copy_(&r);tch::Cuda::synchronize(0);
            crate::tp::allreduce(&Tensor::zeros([1],(Kind::Float,dev)));tch::Cuda::synchronize(0);
            let started=std::time::Instant::now();
            for _ in 0..128 {crate::tp::graph::replay().unwrap();}
            tch::Cuda::synchronize(0);
            rounds.push(json!({"packed":on,"effective_packed":tp_pack_for_rows(rows),"graph_us":started.elapsed().as_secs_f64()*1e6/128.}));
            crate::tp::graph::destroy();
        }
        cases.push(json!({"rows":rows,"exact":true,"changed_inputs_per_graph":4,"rounds":rounds}));
        std::fs::write(out.join(format!("local-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({"rank":tp.rank,"cases":cases})).unwrap()).unwrap();
        eprintln!("[tp-pack-probe] rank{} rows={rows} exact PASS",tp.rank);
    }
}
