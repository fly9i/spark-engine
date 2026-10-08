// SPDX-License-Identifier: MIT
//! 全栈前向(tch)— 对齐 engine/glm53/model.py 的 forward/greedy(M0 验收)。
//! 接线:embed → hc_expand → 45 层(延迟 post)→ contract → norm → lm_head。

use tch::{Kind, Tensor};

use crate::mhc::{hc_contract, hc_expand, mhc_post, mhc_pre, PreOut};
use crate::moe::ExpertPool;
use crate::weights::ModelWeights;

pub struct Engine {
    pub w: ModelWeights,
    pub pool: ExpertPool,
    pub native: Option<crate::moe::NativeExpertPool>,
    pub fast: Option<crate::moefast::MoeFast>,
}

impl Engine {
    /// 整段前向(验收/对照用;fast 可用时 MoE 也走 mgemm,TP 逐层 allreduce)。
    pub fn forward(&mut self, ids: &Tensor) -> Tensor {
        // ids [T] → logits [T,V]
        let x = self.w.embed_tokens(ids); // [T,4096]
        let mut residual = hc_expand(&x); // [T,4,4096]
        let n = self.w.layers.len();
        let mut deferred: Option<(PreOut, Tensor)> = None; // (ffn pre, mlp 输出)

        for (i, layer) in self.w.layers.iter().enumerate() {
            crate::deep_probe::begin_layer();
            let hc = &layer.hc;
            if let Some((pre, m)) = deferred.take() {
                residual = mhc_post(&m, &residual, &pre);
            }
            // attn 站点
            let (pre, z) = mhc_pre(&residual, &hc.attn_fn, &hc.attn_scale, &hc.attn_base, &hc.in_ln);
            let a = if let Some(kw) = &layer.kda {
                crate::kda::kda_forward(kw, &z)
            } else if let Some(mw) = &layer.mla {
                crate::mla::mla_forward(mw, &z)
            } else {
                panic!("层 {i} 无注意力权重");
            };
            residual = mhc_post(&a, &residual, &pre);

            // ffn 站点
            let (pre2, z2) = mhc_pre(&residual, &hc.ffn_fn, &hc.ffn_scale, &hc.ffn_base, &hc.post_ln);
            let m = if let Some(dm) = &layer.dense {
                let g = crate::weights::mm16(&z2, &dm.wg).silu();
                crate::weights::row_mm16(&(g * crate::weights::mm16(&z2, &dm.wu)), &dm.wd)
            } else if let Some(mm) = &layer.moe {
                let (topi, wts) = crate::moe::route(&z2, &mm.w_gate, &mm.bias, 8);
                let mut y = Tensor::zeros(
                    [z2.size()[0], z2.size()[1]],
                    (Kind::Float, z2.device()),
                );
                if let Some(np) = self.native.as_mut() {
                    // 原生路径:trellis 直算(算力换带宽,fp16 出 fp32 累加)
                    let x16 = z2.to_kind(Kind::Half);
                    for t in 0..z2.size()[0] {
                        let one = x16.get(t).unsqueeze(0);
                        for ki in 0..topi.size()[1] {
                            let e = topi.get(t).get(ki).int64_value(&[]) as usize;
                            let pv = np.expert(i, e, z2.device());
                            let (a, b, c) = &pv[0];
                            let (d, e2, f) = &pv[1];
                            let (g, h, i2) = &pv[2];
                            let projs = [(a.shallow_clone(), b.shallow_clone(), c.shallow_clone()),
                                         (d.shallow_clone(), e2.shallow_clone(), f.shallow_clone()),
                                         (g.shallow_clone(), h.shallow_clone(), i2.shallow_clone())];
                            let eo = crate::moe::NativeExpertPool::expert_forward(&one, &projs, z2.device());
                            let contrib = wts.get(t).get(ki) * eo;
                            let old = y.narrow(0, t, 1);
                            let _ = y.narrow(0, t, 1).copy_(&(old + contrib));
                        }
                    }
                } else {
                for t in 0..z2.size()[0] {
                    for ki in 0..topi.size()[1] {
                        let e = topi.get(t).get(ki).int64_value(&[]) as usize;
                        let (wg, wu, wd) = self.pool.expert(layer_plan_idx(layer, i), e, z2.device());
                        let one = z2.get(t).unsqueeze(0);
                        let eo = crate::moe::expert_forward(&one, wg, wu, wd);
                        let contrib = wts.get(t).get(ki) * eo;
                        let old = y.narrow(0, t, 1);
                        let _ = y.narrow(0, t, 1).copy_(&(old + contrib));
                    }
                }
                }
                y + crate::moe::shared_forward(mm, &z2)
            } else {
                panic!("层 {i} 无 MLP 权重");
            };

            if i == n - 1 {
                residual = mhc_post(&m, &residual, &pre2);
            } else {
                deferred = Some((pre2, m));
            }
        }
        let s = hc_contract(&residual); // [T,4096]
        let z = rmsnorm_final(&s, &self.w.final_norm);
        self.w.logits(&z)
    }

    pub fn greedy(&mut self, ids: &[i64], max_new: usize) -> Vec<i64> {
        let mut cur: Vec<i64> = ids.to_vec();
        let mut out = Vec::with_capacity(max_new);
        let dev = self.w.device;
        for _ in 0..max_new {
            let input = Tensor::from_slice(&cur).to_device(dev);
            let logits = self.forward(&input);
            let nxt = logits.get(logits.size()[0] - 1).argmax(-1, false).int64_value(&[]);
            out.push(nxt);
            cur.push(nxt);
        }
        out
    }
}

fn layer_plan_idx(_layer: &crate::weights::LayerWeights, i: usize) -> usize {
    i
}

/// Proposal 3: final contract + RMSNorm whose per-row arithmetic does not depend on the row count (ATen's mean_dim
/// picks its reduction split by the number of outputs). Stream mean as explicit adds; one block per row (rms_rows).
fn final_norm_invariant(residual:&Tensor,w:&Tensor)->Tensor {
    let s=residual.to_kind(Kind::Float);
    let x=((s.select(1,0)+s.select(1,1))+(s.select(1,2)+s.select(1,3)))*0.25;let x=x.contiguous();
    let w=w.to_kind(Kind::Float).contiguous();let y=Tensor::empty_like(&x);
    extern "C"{fn rs_rms_rows(x:*const f32,w:*const f32,y:*mut f32,rows:i32,d:i32,eps:f32)->i32;}
    assert_eq!(unsafe{rs_rms_rows(x.data_ptr().cast(),w.data_ptr().cast(),y.data_ptr().cast(),x.size()[0] as i32,x.size()[1] as i32,1e-5)},0);y
}
fn rmsnorm_final(x: &Tensor, w: &Tensor) -> Tensor {
    let v = x.to_kind(Kind::Float);
    let sq = &v * &v;
    let ms = sq.mean_dim(&[-1i64][..], true, Kind::Float);
    w.to_kind(Kind::Float) * (v * (ms + 1e-5).rsqrt())
}

// ───────────────────────── M1.3:增量解码 ─────────────────────────

pub enum LayerState {
    Kda(crate::kda::KdaState),
    Mla(crate::mla::MlaState),
    MlaG(crate::mla::MlaStateG), // 图兼容(定长窗口)
    MlaLatent(crate::mla_latent::State),
}

pub struct DecodeStates(pub Vec<LayerState>);

impl Engine {
    /// Empty decode state with an explicit latent MLA capacity (serving: per-sequence storage).
    /// fresh_states with every MLA layer's context-proportional rows taken from the KV pool lease.
    pub(crate) fn fresh_states_pooled(&self,pool:&crate::kv_pool::KvPool,lease:&crate::kv_pool::Lease)->DecodeStates {
        let dev=self.w.device;let mut mla=0usize;
        DecodeStates(self.w.layers.iter().map(|layer| if let Some(kw)=&layer.kda {
            LayerState::Kda(crate::kda::KdaState::with_heads(dev,kw.wq.size()[0]/128))
        } else {
            let (latent,pools)=pool.views(mla,lease);mla+=1;
            LayerState::MlaLatent(crate::mla_latent::State::from_pool(layer.mla.as_ref().unwrap(),lease.len,latent,pools))
        }).collect())
    }
    /// KV pool layout of this model: (latent width, latent kind, DSA dim) of each MLA layer in state order.
    pub(crate) fn kv_pool_layers(&self)->Vec<(i64,Kind,i64)> {
        let (w,k)=crate::mla_latent::State::latent_layout();
        self.w.layers.iter().filter_map(|l|l.mla.as_ref()).map(|m|(w,k,m.indexer.as_ref().expect("latent MLA needs indexer weights").k.size()[0])).collect()
    }
    pub(crate) fn fresh_states(&self,capacity:i64)->DecodeStates {
        assert!(crate::mla_latent::enabled(),"fresh_states needs latent MLA");
        let dev=self.w.device;
        DecodeStates(self.w.layers.iter().map(|layer| if let Some(kw)=&layer.kda {
            LayerState::Kda(crate::kda::KdaState::with_heads(dev,kw.wq.size()[0]/128))
        } else {
            LayerState::MlaLatent(crate::mla_latent::State::new(layer.mla.as_ref().unwrap(),capacity))
        }).collect())
    }
}
/// Reset a state in place to "no tokens" (same storage, so captured graphs stay valid).
/// Rows beyond the length are never read; tails, recurrent state and convolution are zeroed.
pub(crate) fn reset_states(ds:&DecodeStates) {
    for l in &ds.0 {
        match l {
            LayerState::Kda(k)=>{let _=k.h.shallow_clone().zero_();let _=k.conv.shallow_clone().zero_();}
            LayerState::MlaLatent(m)=>{let _=m.len.shallow_clone().zero_();let _=m.index.tail_k.shallow_clone().zero_();let _=m.index.tail_gate.shallow_clone().zero_();}
            _=>panic!("reset_states supports KDA and latent MLA only"),
        }
    }
}
/// Committed token count of a state (reads one device scalar).
pub(crate) fn states_len(ds:&DecodeStates)->i64 {
    ds.0.iter().find_map(|l|match l{LayerState::MlaLatent(m)=>Some(m.len.int64_value(&[0])),_=>None}).expect("latent MLA layer")
}

impl Engine {
    /// Batched tree verifier forward. Dense/MHC projections batch across nodes;
    /// attention states inherit only the indicated parent, never adjacent rows.
    pub fn tree_forward(&mut self,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],collect:bool)
        ->(Tensor,Vec<DecodeStates>,Vec<Tensor>) {
        self.tree_forward_impl(ids,base,parents,collect,true)
    }
    /// Capture caller must preflight capacity before capture and every replay.
    pub(crate) fn tree_forward_impl(&mut self,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],collect:bool,check_room:bool)
        ->(Tensor,Vec<DecodeStates>,Vec<Tensor>) {
        self.tree_forward_selected(ids,base,parents,collect,crate::dsa::TreeSelection::Ranked,check_room)
    }
    pub(crate) fn tree_forward_selected(&mut self,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],collect:bool,
        selection:crate::dsa::TreeSelection,check_room:bool)->(Tensor,Vec<DecodeStates>,Vec<Tensor>) {
        self.tree_forward_output_selected(ids,base,parents,collect,selection,check_room,false)
    }
    /// Returns device Int64 [nodes] predictions, with the same states/features
    /// as the logits interface. Capture consumers must keep this output mode
    /// fixed for the graph lifetime; the flag is part of the session signature.
    pub(crate) fn tree_forward_predictions_selected(&mut self,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],collect:bool,
        selection:crate::dsa::TreeSelection,check_room:bool)->(Tensor,Vec<DecodeStates>,Vec<Tensor>) {
        self.tree_forward_output_selected(ids,base,parents,collect,selection,check_room,true)
    }
    fn tree_forward_output_selected(&mut self,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],collect:bool,
        selection:crate::dsa::TreeSelection,check_room:bool,predictions:bool)->(Tensor,Vec<DecodeStates>,Vec<Tensor>) {
        let (output,states,features)=self.tree_forward_collected::<false>(ids,base,parents,collect,selection,check_room,predictions,false,None);
        (output,states.into_full(),features)
    }
    /// Only speculative verification may request deferred KDA records. Every
    /// legacy raw-logits/quality API above keeps its full-state contract.
    pub(crate) fn verifier_forward_selected(&mut self,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],collect:bool,
        selection:crate::dsa::TreeSelection,check_room:bool,predictions:bool)
        ->(Tensor,crate::verifier_state::VerifierStates,Vec<Tensor>) {
        let deferred=ids.device().is_cuda() && crate::kda::deferred_eligible(parents);
        self.tree_forward_collected::<false>(ids,base,parents,collect,selection,check_room,predictions,deferred,None)
    }
    /// Diagnostic monomorphization: actual model layer inputs, no hot-path hook.
    pub(crate) fn tree_dsa_topk_probe(&mut self,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],records:&mut Vec<serde_json::Value>) {
        let _=self.tree_forward_collected::<true>(ids,base,parents,false,crate::dsa::TreeSelection::Ranked,true,false,false,Some(records));
    }
    fn tree_forward_collected<const DSA_PROBE:bool>(&mut self,ids:&Tensor,base:&DecodeStates,parents:&[Option<usize>],collect:bool,
        selection:crate::dsa::TreeSelection,check_room:bool,predictions:bool,deferred:bool,mut dsa_probe:Option<&mut Vec<serde_json::Value>>)
        ->(Tensor,crate::verifier_state::VerifierStates,Vec<Tensor>) {
        use crate::verifier_state::{VerifierStates,Layer as VerifierLayer};
        assert!(crate::mla_latent::enabled(),"tree verification requires latent MLA");
        let count=ids.size()[0] as usize;assert!(count>0);assert_eq!(count,parents.len());
        assert_eq!(base.0.len(),self.w.layers.len());
        let mut residual=hc_expand(&self.w.embed_tokens(ids));
        // Proposal 3 (GLM53_VERIFY_INVARIANT=1): the expanded view is not contiguous, so layer 0's mhc_pre fell back
        // to ATen (cuBLAS picks its algorithm by row count: 2-3 rows differ from 4-8). Materialize it (rows x 64 KB)
        // so layer 0 uses the same fused kernel as every other layer. L1 versus the fallback.
        if verify_invariant() {residual=residual.contiguous();}
        let mut states:Vec<_>=if deferred{Vec::new()}else{(0..count).map(|_|DecodeStates(Vec::new())).collect()};
        let mut deferred_layers=Vec::new();let mut features=Vec::new();
        probe_point(0,"embed_resid",&residual);
        for (i,layer) in self.w.layers.iter().enumerate() {
            let (pre,z)=mhc_pre(&residual,&layer.hc.attn_fn,&layer.hc.attn_scale,&layer.hc.attn_base,&layer.hc.in_ln);
            probe_layer(i);probe_point(i,"pre_z",&z);probe_point(i,"pre_post",&pre.post_mix);probe_point(i,"pre_comb",&pre.comb);
            // AR prefetch: the attention o_proj all-reduce is followed by the FFN MHC-pre and the router/dense gate.
            if let Some(mm)=&layer.moe {crate::tp::set_ar_prefetch(&[(&layer.hc.ffn_fn,i64::MAX),(&mm.w_gate,i64::MAX)]);}
            else if let Some(dm)=&layer.dense {let wg=crate::dense_fp8::quant_of(&dm.wg).map(|q|q.0);crate::tp::set_ar_prefetch(&[(&layer.hc.ffn_fn,i64::MAX),(wg.as_ref().unwrap_or(&dm.wg),i64::MAX)]);}
            // I3 step 2 (GLM53_AR_FUSED=1): the o-projection's TP sum is formed by the MHC post below.
            let (attention,pending)=crate::tp::with_deferred_sum(||match &base.0[i] {
                LayerState::Kda(base)=>{
                    let record=if deferred{crate::kda::tree_deferred(layer.kda.as_ref().unwrap(),&z,base,parents)}else{None};
                    if let Some((o,record))=record {
                        deferred_layers.push(VerifierLayer::Kda(record));o
                    }else{
                        let (o,s)=crate::kda::tree(layer.kda.as_ref().unwrap(),&z,base,parents);
                        if deferred {deferred_layers.push(VerifierLayer::Full(s.into_iter().map(LayerState::Kda).collect()));}
                        else {for (dst,src) in states.iter_mut().zip(s){dst.0.push(LayerState::Kda(src));}}
                        o
                    }
                },
                LayerState::MlaLatent(base)=>{
                    if DSA_PROBE {dsa_probe.as_deref_mut().unwrap().push(crate::dsa_topk::real_layer(i,layer.mla.as_ref().unwrap(),&z,base,parents));}
                    let (o,s)=crate::mla_latent::tree_impl_selected(layer.mla.as_ref().unwrap(),&z,base,parents,selection,check_room);
                    if deferred {deferred_layers.push(VerifierLayer::Full(s.into_iter().map(LayerState::MlaLatent).collect()));}
                    else {for (dst,src) in states.iter_mut().zip(s){dst.0.push(LayerState::MlaLatent(src));}}o
                },_=>panic!("tree verification requires KDA/latent state"),
            });
            crate::tp::clear_ar_prefetch();
            let pending=probe_settle(pending);
            probe_point(i,"attn_in",&z);probe_point(i,"attn_out",&attention);
            let (r,pre,z)=crate::mhc::mhc_post_pre_pending(&attention,pending,&residual,&pre,&layer.hc.ffn_fn,&layer.hc.ffn_scale,&layer.hc.ffn_base,&layer.hc.post_ln);
            residual=r;probe_point(i,"ffn_in",&z);
            // AR prefetch: the MoE / dense-down all-reduce is followed by the next layer's MHC-pre and first projection.
            if crate::tp::ar_prefetch_enabled() {if let Some(next)=self.w.layers.get(i+1) {
                let first=if let Some(k)=&next.kda {crate::dense_fp8::quant_of(&k.wq).map(|q|q.0).unwrap_or_else(||k.wq.shallow_clone())}
                    else if let Some(m)=&next.mla {crate::dense_fp8::quant_of(&m.q_a).map(|q|q.0).unwrap_or_else(||m.q_a.shallow_clone())}
                    else {next.hc.attn_fn.shallow_clone()};
                // A C12/Q8-coded first projection is read from its coded copy, not the Half source: prefetch those bytes.
                let coded=crate::c12::prefetch_regions(&first,crate::tp::ar_prefetch_bytes()-next.hc.attn_fn.numel() as i64*next.hc.attn_fn.kind().elt_size_in_bytes() as i64);
                if coded.is_empty() {crate::tp::set_ar_prefetch(&[(&next.hc.attn_fn,i64::MAX),(&first,i64::MAX)]);}
                else {crate::tp::set_ar_prefetch(&[(&next.hc.attn_fn,i64::MAX)]);crate::tp::add_ar_prefetch_raw(coded);}
            }}
            // GLM53_MOE_ROUTE_SIDE=1 (L0, schedule only): the router (fused kernels) runs on a side stream while the main
            // stream writes the shared expert's partial into the packed buffer; joined before the routed experts.
            let route_side=layer.moe.as_ref().is_some_and(|mm|count<=16 && std::env::var("GLM53_MOE_COOP").as_deref()==Ok("1")
                && crate::moe::pack_direct_eligible(count as i64) && !(crate::moe::shared_stream_eligible(count as i64)&&crate::moe::shared_stream_kernels_ok(mm,count as i64))
                && crate::moe::route_side_ok(&z,&mm.w_gate,&mm.bias));
            residual=if route_side {
                let mm=layer.moe.as_ref().unwrap();
                let fast=self.fast.as_mut().expect("tree requires resident EXL3");assert!(fast.assume_hot);
                let shared_half=crate::moe::input_half_reuse_enabled().then(||z.to_kind(Kind::Half));
                extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;fn rs_stream_join(n:i32)->i32;}
                assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);
                let (ids,weights,weights_half)=crate::moe::route_h(&z,&mm.w_gate,&mm.bias,8);
                assert_eq!(unsafe{rs_stream_set(-1)},0);
                let p=Tensor::empty([2*count as i64,4096],(Kind::Float,z.device()));
                crate::moe::shared_into_packed(&p,mm,&z,shared_half.as_ref());
                assert_eq!(unsafe{rs_stream_join(1)},0);
                crate::route_trace::record(i,&ids);
                let coop_w=weights_half.as_ref().unwrap_or(&weights);
                let input=crate::deep_probe::expert_input(shared_half.as_ref().unwrap_or(&z));
                fast.expert_cooperative_into(i,&input,&ids.contiguous(),coop_w,&p.narrow(0,0,count as i64));
                probe_point(i,"routed",&p.narrow(0,0,count as i64));probe_point(i,"route_w",&weights);
                crate::moe::finish_packed_after_shared(p,mm,&residual,&pre)
            } else if let Some(mm)=&layer.moe {
                let fast=self.fast.as_mut().expect("tree requires resident EXL3");assert!(fast.assume_hot);
                let (ids,weights,weights_half)=crate::moe::route_h(&z,&mm.w_gate,&mm.bias,8);
                crate::route_trace::record(i,&ids);
                // GLM53_ROUTER_HALF_W=1: the cooperative kernel takes the router's Half copy (its own conversion is then a no-op).
                let coop_w=weights_half.as_ref().unwrap_or(&weights);
                // This canonical Half belongs to original z, independently of
                // any routed-only deep_probe replay input. It remains alive
                // through the shared gate/up consumers on the same stream.
                let shared_half=crate::moe::input_half_reuse_enabled().then(||z.to_kind(Kind::Half));
                let input=crate::deep_probe::expert_input(shared_half.as_ref().unwrap_or(&z));let mut outputs=Vec::new();
                // D2 (GLM53_MOE_SHARED_STREAM=1): the shared expert (same input, independent until the packed
                // collective) runs on a forked side stream while the routed cooperative kernels run on the main
                // stream. Same kernels and arithmetic; only the schedule changes (L0).
                let shared_early=(crate::moe::shared_stream_eligible(count as i64)&&crate::moe::shared_stream_kernels_ok(mm,count as i64)).then(||crate::moe::shared_partial_on_side(mm,&z,shared_half.as_ref()));
                // I3 step 3 (GLM53_MOE_PACK_DIRECT=1): the routed kernel and the shared down projection write the two halves
                // of the collective's packed buffer directly (no cat); same values, same kernels (L0).
                let direct=(shared_early.is_none() && std::env::var("GLM53_MOE_COOP").as_deref()==Ok("1") && crate::moe::pack_direct_eligible(count as i64))
                    .then(||Tensor::empty([2*count as i64,4096],(Kind::Float,z.device())));
                if let Some(p)=&direct {fast.expert_cooperative_into(i,&input,&ids.contiguous(),coop_w,&p.narrow(0,0,count as i64));}
                else if std::env::var("GLM53_MOE_COOP").as_deref()==Ok("1") {
                    for start in (0..count as i64).step_by(32) {
                        let n=(count as i64-start).min(32);
                        outputs.push(fast.expert_cooperative(i,&input.narrow(0,start,n),&ids.narrow(0,start,n).contiguous(),&coop_w.narrow(0,start,n)));
                    }
                } else if std::env::var("GLM53_MOE_BATCH").as_deref()==Ok("1") {
                    for start in (0..count as i64).step_by(16) {
                        let n=(count as i64-start).min(16);
                        let sel=fast.sel_device(i,&ids.narrow(0,start,n).reshape([-1]),self.w.device).view([n,8]);
                        outputs.push(fast.expert_multi_sel(&input.narrow(0,start,n),&sel,&weights.narrow(0,start,n),self.w.device));
                    }
                } else {for row in 0..count as i64 {
                    let sel=fast.sel_device(i,&ids.get(row),self.w.device);
                    outputs.push(fast.expert_batch_sel(&input.narrow(0,row,1),&sel,&weights.get(row),self.w.device));
                }}
                if let Some(p)=direct {
                    probe_point(i,"routed",&p.narrow(0,0,count as i64));probe_point(i,"route_w",&weights);
                    crate::moe::finish_packed_direct(p,mm,&z,shared_half.as_ref(),&residual,&pre)
                } else {
                let routed=if outputs.len()==1 && std::env::var("GLM53_MOE_NO_COPY").as_deref()==Ok("1"){outputs.pop().unwrap()}else{Tensor::cat(&outputs,0)};
                probe_point(i,"routed",&routed);probe_point(i,"route_w",&weights);
                if let Some(shared)=shared_early {crate::moe::finish_packed_post_joined(routed,shared,mm,&residual,&pre)}
                else {crate::moe::finish_tp_post_with_half(routed,mm,&z,shared_half.as_ref(),&residual,&pre)}
                }
            }else{mhc_post(&mlp_forward_f(i,layer,&z,&mut self.native,&mut self.pool),&residual,&pre)};
            crate::tp::clear_ar_prefetch();probe_point(i,"resid",&residual);
            if collect&&[5,14,24,33,42].contains(&i){features.push(hc_contract(&residual));}
        }
        let z=if verify_invariant(){final_norm_invariant(&residual,&self.w.final_norm)}else{rmsnorm_final(&hc_contract(&residual),&self.w.final_norm)};
        let head=if predictions { self.w.predictions(&z) } else { self.w.logits(&z) };
        probe_point(99,"final_z",&z);probe_point(99,"head",&head);
        let states=if deferred{VerifierStates::Deferred{tokens:count,layers:deferred_layers}}else{VerifierStates::Full(states)};
        (head,states,features)
    }
    /// Multi-sequence verifier (serving): rows of several independent chains are verified in one forward.
    /// MHC/dense/MoE/head run on all rows (one weight read); attention runs per sequence on its own base
    /// state (KDA deferred records, latent MLA chain nodes). segs[g] = (first row, rows) of sequence g.
    pub(crate) fn tree_forward_multi(&mut self,ids:&Tensor,bases:&[&DecodeStates],segs:&[(usize,usize)],parents_all:&[Vec<Option<usize>>],
        selections:&[crate::dsa::TreeSelection],collect:bool,predictions:bool,check_room:bool)
        ->(Tensor,Vec<crate::verifier_state::VerifierStates>,Vec<Tensor>) {
        use crate::verifier_state::{VerifierStates,Layer as VerifierLayer};
        assert!(crate::mla_latent::enabled(),"tree verification requires latent MLA");
        let count=ids.size()[0] as usize;assert!(count>0);assert_eq!(segs.iter().map(|s|s.1).sum::<usize>(),count);
        for b in bases {assert_eq!(b.0.len(),self.w.layers.len());}
        let mut seq_layers:Vec<Vec<VerifierLayer>>=(0..segs.len()).map(|_|Vec::new()).collect();
        let mut residual=hc_expand(&self.w.embed_tokens(ids));
        if verify_invariant() {residual=residual.contiguous();}   // proposal 3: same as the single-sequence verify
        let mut features=Vec::new();
        // GLM53_MULTI_TIMING=1: synchronized per-category wall time (diagnostic only).
        let timing=std::env::var("GLM53_MULTI_TIMING").as_deref()==Ok("1");let mut acc=[0f64;4];let mut mark=std::time::Instant::now();
        let mut tick=|k:usize,acc:&mut [f64;4]|{if timing{tch::Cuda::synchronize(0);acc[k]+=mark.elapsed().as_secs_f64()*1000.;mark=std::time::Instant::now();}};
        for (i,layer) in self.w.layers.iter().enumerate() {
            let (pre,z)=mhc_pre(&residual,&layer.hc.attn_fn,&layer.hc.attn_scale,&layer.hc.attn_base,&layer.hc.in_ln);
            probe_layer(i);probe_point(i,"pre_z",&z);
            tick(0,&mut acc);
            let (attention,pending)=crate::tp::with_deferred_sum(||{
            let mut outs=Vec::with_capacity(segs.len());
            // KDA layer: row-wise work batched over all sequences (one weight read), recurrence per sequence.
            let kda_bases:Option<Vec<&crate::kda::KdaState>>=bases.iter().map(|b|match &b.0[i]{LayerState::Kda(k)=>Some(k),_=>None}).collect();
            let multi=kda_bases.as_ref().filter(|_|segs.len()>1).and_then(|kb|crate::kda::tree_deferred_multi(layer.kda.as_ref().unwrap(),&z,kb,segs,parents_all));
            let mla_bases:Option<Vec<&crate::mla_latent::State>>=bases.iter().map(|b|match &b.0[i]{LayerState::MlaLatent(m)=>Some(m),_=>None}).collect();
            if let Some((o,records)) = multi {
                for (g,r) in records.into_iter().enumerate() {seq_layers[g].push(VerifierLayer::Kda(r));}
                outs.push(o);
            } else if let Some(mb)=mla_bases.filter(|_|segs.len()>1 && std::env::var("GLM53_MLA_MULTI").as_deref()!=Ok("0")) {
                let (o,states)=crate::mla_latent::tree_multi_selected(layer.mla.as_ref().unwrap(),&z,&mb,segs,parents_all,selections,check_room);
                for (g,s) in states.into_iter().enumerate() {seq_layers[g].push(VerifierLayer::Full(s.into_iter().map(LayerState::MlaLatent).collect()));}
                outs.push(o);
            } else {
            for (g,&(first,len)) in segs.iter().enumerate() {
                let zg=z.narrow(0,first as i64,len as i64);let parents=&parents_all[g];
                match &bases[g].0[i] {
                    LayerState::Kda(base)=>{
                        let (o,record)=crate::kda::tree_deferred(layer.kda.as_ref().unwrap(),&zg,base,parents).expect("multi-sequence verify needs deferred KDA chains");
                        seq_layers[g].push(VerifierLayer::Kda(record));outs.push(o);
                    },
                    LayerState::MlaLatent(base)=>{
                        let (o,s)=crate::mla_latent::tree_impl_selected(layer.mla.as_ref().unwrap(),&zg,base,parents,selections[g],check_room);
                        seq_layers[g].push(VerifierLayer::Full(s.into_iter().map(LayerState::MlaLatent).collect()));outs.push(o);
                    },_=>panic!("tree verification requires KDA/latent state"),
                }
            }
            }
            tick(1,&mut acc);
            if outs.len()==1{outs.pop().unwrap()}else{Tensor::cat(&outs,0)}
            });
            let pending=probe_settle(pending);
            probe_point(i,"attn_out",&attention);
            residual=crate::mhc::mhc_post_pending(&attention,pending,&residual,&pre);
            probe_point(i,"post_resid",&residual);
            let (pre,z)=mhc_pre(&residual,&layer.hc.ffn_fn,&layer.hc.ffn_scale,&layer.hc.ffn_base,&layer.hc.post_ln);
            probe_point(i,"ffn_in",&z);
            tick(0,&mut acc);
            // GLM53_MOE_ROUTE_SIDE=1 (multi): router on the side stream while the main stream computes the shared expert
            // (packed second half when the direct path applies, else its activation); same arithmetic as below.
            let route_side_m=layer.moe.as_ref().is_some_and(|mm|count<=32 && std::env::var("GLM53_MOE_COOP").as_deref()==Ok("1")
                && !(crate::moe::shared_stream_eligible(count as i64)&&crate::moe::shared_stream_kernels_ok(mm,count as i64))
                && (crate::moe::pack_direct_eligible(count as i64) || crate::moe::unpacked_rows(count as i64))
                && crate::moe::route_side_ok_multi(&z,&mm.w_gate,&mm.bias));
            residual=if route_side_m {
                let mm=layer.moe.as_ref().unwrap();
                let fast=self.fast.as_mut().expect("tree requires resident EXL3");assert!(fast.assume_hot);
                let shared_half=crate::moe::input_half_reuse_enabled().then(||z.to_kind(Kind::Half));
                extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;fn rs_stream_join(n:i32)->i32;}
                assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);
                let (ids,weights,weights_half)=crate::moe::route_h_multi(&z,&mm.w_gate,&mm.bias,8);
                assert_eq!(unsafe{rs_stream_set(-1)},0);
                let direct=crate::moe::pack_direct_eligible(count as i64).then(||Tensor::empty([2*count as i64,4096],(Kind::Float,z.device())));
                let act=match &direct {Some(p)=>{crate::moe::shared_into_packed(p,mm,&z,shared_half.as_ref());None},
                    None=>Some(crate::moe::shared_activation_with_half(mm,&z,shared_half.as_ref()))};
                assert_eq!(unsafe{rs_stream_join(1)},0);
                crate::route_trace::record(i,&ids);
                let coop_w=weights_half.as_ref().unwrap_or(&weights);
                let input=crate::deep_probe::expert_input(shared_half.as_ref().unwrap_or(&z));
                if let Some(p)=direct {
                    fast.expert_cooperative_into(i,&input,&ids.contiguous(),coop_w,&p.narrow(0,0,count as i64));
                    probe_point(i,"routed",&p.narrow(0,0,count as i64));probe_point(i,"route_w",&weights);
                    crate::moe::finish_packed_after_shared(p,mm,&residual,&pre)
                } else {
                    let mut outputs=Vec::new();
                    for start in (0..count as i64).step_by(32) {
                        let n=(count as i64-start).min(32);
                        outputs.push(fast.expert_cooperative(i,&input.narrow(0,start,n),&ids.narrow(0,start,n).contiguous(),&coop_w.narrow(0,start,n)));
                    }
                    let routed=if outputs.len()==1 && std::env::var("GLM53_MOE_NO_COPY").as_deref()==Ok("1"){outputs.pop().unwrap()}else{Tensor::cat(&outputs,0)};
                    probe_point(i,"routed",&routed);probe_point(i,"route_w",&weights);
                    crate::moe::finish_unpacked_with_act(routed,mm,act.as_ref().unwrap(),&residual,&pre)
                }
            } else if let Some(mm)=&layer.moe {
                let fast=self.fast.as_mut().expect("tree requires resident EXL3");assert!(fast.assume_hot);
                let (ids,weights,weights_half)=crate::moe::route_h_multi(&z,&mm.w_gate,&mm.bias,8);
                crate::route_trace::record(i,&ids);
                // GLM53_ROUTER_HALF_W=1: the cooperative kernel takes the router's Half copy (its own conversion is then a no-op).
                let coop_w=weights_half.as_ref().unwrap_or(&weights);
                // This canonical Half belongs to original z, independently of
                // any routed-only deep_probe replay input. It remains alive
                // through the shared gate/up consumers on the same stream.
                let shared_half=crate::moe::input_half_reuse_enabled().then(||z.to_kind(Kind::Half));
                let input=crate::deep_probe::expert_input(shared_half.as_ref().unwrap_or(&z));let mut outputs=Vec::new();
                // D2 (GLM53_MOE_SHARED_STREAM=1): the shared expert (same input, independent until the packed
                // collective) runs on a forked side stream while the routed cooperative kernels run on the main
                // stream. Same kernels and arithmetic; only the schedule changes (L0).
                let shared_early=(crate::moe::shared_stream_eligible(count as i64)&&crate::moe::shared_stream_kernels_ok(mm,count as i64)).then(||crate::moe::shared_partial_on_side(mm,&z,shared_half.as_ref()));
                // I3 step 3 (GLM53_MOE_PACK_DIRECT=1): the routed kernel and the shared down projection write the two halves
                // of the collective's packed buffer directly (no cat); same values, same kernels (L0).
                let direct=(shared_early.is_none() && std::env::var("GLM53_MOE_COOP").as_deref()==Ok("1") && crate::moe::pack_direct_eligible(count as i64))
                    .then(||Tensor::empty([2*count as i64,4096],(Kind::Float,z.device())));
                if let Some(p)=&direct {fast.expert_cooperative_into(i,&input,&ids.contiguous(),coop_w,&p.narrow(0,0,count as i64));}
                else if std::env::var("GLM53_MOE_COOP").as_deref()==Ok("1") {
                    for start in (0..count as i64).step_by(32) {
                        let n=(count as i64-start).min(32);
                        outputs.push(fast.expert_cooperative(i,&input.narrow(0,start,n),&ids.narrow(0,start,n).contiguous(),&coop_w.narrow(0,start,n)));
                    }
                } else if std::env::var("GLM53_MOE_BATCH").as_deref()==Ok("1") {
                    for start in (0..count as i64).step_by(16) {
                        let n=(count as i64-start).min(16);
                        let sel=fast.sel_device(i,&ids.narrow(0,start,n).reshape([-1]),self.w.device).view([n,8]);
                        outputs.push(fast.expert_multi_sel(&input.narrow(0,start,n),&sel,&weights.narrow(0,start,n),self.w.device));
                    }
                } else {for row in 0..count as i64 {
                    let sel=fast.sel_device(i,&ids.get(row),self.w.device);
                    outputs.push(fast.expert_batch_sel(&input.narrow(0,row,1),&sel,&weights.get(row),self.w.device));
                }}
                if let Some(p)=direct {
                    probe_point(i,"routed",&p.narrow(0,0,count as i64));probe_point(i,"route_w",&weights);
                    crate::moe::finish_packed_direct(p,mm,&z,shared_half.as_ref(),&residual,&pre)
                } else {
                let routed=if outputs.len()==1 && std::env::var("GLM53_MOE_NO_COPY").as_deref()==Ok("1"){outputs.pop().unwrap()}else{Tensor::cat(&outputs,0)};
                probe_point(i,"routed",&routed);probe_point(i,"route_w",&weights);
                if let Some(shared)=shared_early {crate::moe::finish_packed_post_joined(routed,shared,mm,&residual,&pre)}
                else {crate::moe::finish_tp_post_with_half(routed,mm,&z,shared_half.as_ref(),&residual,&pre)}
                }
            }else{mhc_post(&mlp_forward_f(i,layer,&z,&mut self.native,&mut self.pool),&residual,&pre)};
            probe_point(i,"resid",&residual);
            tick(2,&mut acc);
            if collect&&[5,14,24,33,42].contains(&i){features.push(hc_contract(&residual));}
        }
        let z=if verify_invariant(){final_norm_invariant(&residual,&self.w.final_norm)}else{rmsnorm_final(&hc_contract(&residual),&self.w.final_norm)};
        let head=if predictions { self.w.predictions(&z) } else { self.w.logits(&z) };
        tick(3,&mut acc);
        if timing {eprintln!("[multi-timing] seqs {} rows {} mhc_ms {:.1} attn_ms {:.1} ffn_ms {:.1} head_ms {:.1}",segs.len(),count,acc[0],acc[1],acc[2],acc[3]);}
        let states=seq_layers.into_iter().zip(segs).map(|(layers,&(_,len))|VerifierStates::Deferred{tokens:len,layers}).collect();
        (head,states,features)
    }
}

/// GLM53_NVTX=1: named NVTX ranges around prefill stages (kernel attribution in nsys; otherwise free).
fn nvtx_on()->bool {std::env::var("GLM53_NVTX").as_deref()==Ok("1")}
struct Nvtx(bool);
impl Nvtx {fn new(on:bool,name:&str)->Self{if on{extern "C"{fn rs_nvtx_push(n:*const std::ffi::c_char);}let c=std::ffi::CString::new(name).unwrap();unsafe{rs_nvtx_push(c.as_ptr())};}Nvtx(on)}}
impl Drop for Nvtx {fn drop(&mut self){if self.0{extern "C"{fn rs_nvtx_pop();}unsafe{rs_nvtx_pop()};}}}
/// Proposal 3: row pieces for batch-invariant verify ops: at most 8 rows and never a single row (1-row inputs take
/// other kernels: fp8 gemv, ATen single-output reductions). m <= 8 is one piece; a remainder of 1 becomes 7 + 2.
pub fn invariant_pieces(m:i64)->Vec<(i64,i64)> {
    if m<=8 {return vec![(0,m)];}
    let mut v:Vec<(i64,i64)>=(0..m).step_by(8).map(|r|(r,(m-r).min(8))).collect();
    if v.last().unwrap().1==1 {let n=v.len();v[n-2].1=7;v[n-1]=(v[n-1].0-1,2);}
    v
}
pub fn verify_invariant()->bool {static V:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*V.get_or_init(||std::env::var("GLM53_VERIFY_INVARIANT").as_deref()==Ok("1"))}
thread_local! {static PROBE:std::cell::RefCell<Option<Vec<(usize,&'static str,Tensor)>>>=std::cell::RefCell::new(None);}
/// Row-invariance audit (verify-invariance): copies of per-layer intermediates while enabled; no-op otherwise.
pub fn probe_begin(){PROBE.with(|p|*p.borrow_mut()=Some(Vec::new()));}
pub fn probe_take()->Vec<(usize,&'static str,Tensor)>{PROBE.with(|p|p.borrow_mut().take().unwrap_or_default())}
thread_local! {static PROBE_LAYER:std::cell::Cell<usize>=std::cell::Cell::new(0);}
/// Inner taps (inside attention modules) are attributed to the layer set here by the verify loops.
pub fn probe_layer(i:usize){PROBE_LAYER.with(|c|c.set(i));}
#[inline] pub fn probe_inner(tag:&'static str,t:&Tensor){probe_point(PROBE_LAYER.with(|c|c.get()),tag,t);}
/// A pending TP sum (I3 step 2) is materialized when probes record the attention output.
fn probe_settle(p:Option<crate::tp::PendingSum>)->Option<crate::tp::PendingSum> {
    match p {Some(v) if PROBE.with(|x|x.borrow().is_some())=>{crate::tp::materialize(v);None},_=>p}
}
#[inline] pub fn probe_point(i:usize,tag:&'static str,t:&Tensor){PROBE.with(|p|if let Some(v)=p.borrow_mut().as_mut(){v.push((i,tag,t.copy()));});}
impl Engine {
    /// prefill:整段 prompt 一次前向,建立全部注意力状态。
    pub fn prefill(&mut self, ids: &Tensor) -> (Tensor, DecodeStates) {
        self.prefill_with(ids, None)
    }

    /// 可续跑的 prefill:ds 为 Some 时在既有状态上继续(分块 prefill 等价性契约)。
    pub fn prefill_with(&mut self, ids: &Tensor, ds: Option<DecodeStates>) -> (Tensor, DecodeStates) {
        let (logits,state,_)=self.prefill_record(ids,ds,false);
        (logits,state)
    }

    pub fn prefill_record(&mut self,ids:&Tensor,ds:Option<DecodeStates>,collect:bool)->(Tensor,DecodeStates,Vec<Tensor>) {
        self.prefill_record_impl(ids,ds,collect,false)
    }
    /// Request paths need only the final prompt logits; diagnostics can still
    /// ask for all rows through prefill_record. Feature/state coverage is full.
    pub fn prefill_record_last(&mut self,ids:&Tensor,ds:Option<DecodeStates>,collect:bool)->(Tensor,DecodeStates,Vec<Tensor>) {
        self.prefill_record_impl(ids,ds,collect,std::env::var("GLM53_PREFILL_LAST_LOGITS").as_deref()==Ok("1"))
    }
    /// P8 (GLM53_PREFILL_SP=1): sequence-parallel prefill for TP2. The residual stream is split by
    /// rows (rank r owns rows [r*T/2,(r+1)*T/2)); MHC runs on the owned rows only, attention/FFN
    /// inputs are all-gathered, and row-parallel outputs are reduce-scattered (same bytes as the
    /// all-reduce, half of the row-wise work). Per-row arithmetic equals the replicated path; the
    /// MoE lane uses the MOE_SUM1 combine (routed + shared partials, one collective).
    fn sp_eligible(&self,t:i64)->bool {
        std::env::var("GLM53_PREFILL_SP").as_deref()==Ok("1") && crate::tp::world().world==2 && t>=128 && t%2==0 && crate::tp::dense_enabled()
            && crate::mla_latent::enabled() && self.fast.as_ref().map_or(false,|f|f.assume_hot)
            && !["","0"].contains(&std::env::var("GLM53_PREFILL_GROUPED").unwrap_or_default().as_str())
    }
    fn prefill_record_sp(&mut self,ids:&Tensor,ds:Option<DecodeStates>,collect:bool)->Option<(Tensor,DecodeStates,Vec<Tensor>)> {
        let t=ids.size()[0];let tp=crate::tp::world();
        if let Some(s)=ds.as_ref().and_then(|d|d.0.iter().find_map(|s|match s {LayerState::MlaLatent(s)=>Some(s),_=>None})) {s.ensure_room(t);}
        let half=t/2;let lo=tp.rank as i64*half;let sp_rows=half;
        let x=self.w.embed_prefill(ids);
        let mut residual=hc_expand(&x.narrow(0,lo,half)).contiguous();
        let n=self.w.layers.len();
        let mut prior=ds.map(|d|{assert_eq!(d.0.len(),n);d.0.into_iter()});
        let mut deferred:Option<(crate::mhc::PreOut,Tensor)>=None;
        let mut states:Vec<LayerState>=Vec::with_capacity(n);let mut features=Vec::new();
        let grouped=std::env::var("GLM53_PREFILL_GROUPED").unwrap_or_default();
        let flag=|k:&str|std::env::var(k).as_deref()==Ok("1");
        let (half_reuse,shared_fuse,feature_reuse)=(flag("GLM53_PREFILL_HALF_REUSE"),flag("GLM53_PREFILL_SHARED_FUSE"),flag("GLM53_PREFILL_FEATURE_REUSE"));
        let half_gather=flag("GLM53_PREFILL_SP_HALF_GATHER");
        let half_z=flag("GLM53_PREFILL_SP_HALF_Z");
        let ag_overlap=crate::tp::ag_overlap_enabled();
        let mut feature_pending=false;let nv=nvtx_on();
        for (i,layer) in self.w.layers.iter().enumerate() {
            let _mhc1=Nvtx::new(nv,"mhc-attn");
            let (pre,z_half)=if let Some((prev,m))=deferred.take() {
                let (r,p,z)=crate::mhc::mhc_post_pre(&m,&residual,&prev,&layer.hc.attn_fn,&layer.hc.attn_scale,&layer.hc.attn_base,&layer.hc.in_ln);
                residual=r;(p,z)
            } else {mhc_pre(&residual,&layer.hc.attn_fn,&layer.hc.attn_scale,&layer.hc.attn_base,&layer.hc.in_ln)};
            if feature_pending {features.push(crate::tp::all_gather_rows(&hc_contract(&residual)));feature_pending=false;}
            drop(_mhc1);
            // GLM53_PREFILL_SP_HALF_Z=1 (L1): gather z as Half (half the bytes). Its Half-GEMM consumers round it to Half
            // anyway (same values); the TF32 consumers (KDA beta, DSA indexer) see the same 10-bit mantissa.
            // GLM53_PREFILL_AG_OVERLAP (L0): for KDA layers the gather of z runs while this rank projects its own rows
            // (kda::front); the other rank's rows are projected after it lands and the fronts are joined in rank order.
            let mut kfront:Option<crate::kda::KdaFront>=None;
            let z={let _g=Nvtx::new(nv,"gather-z");
                if ag_overlap && !half_z && layer.kda.is_some() {
                    let kw=layer.kda.as_ref().unwrap();
                    let pg=crate::tp::all_gather_rows_start(&z_half);
                    let own=crate::kda::front(kw,&z_half);
                    let z=pg.finish();
                    if let Some(own)=own {
                        let other=z.narrow(0,(1-tp.rank as i64)*half,half);
                        kfront=Some(crate::kda::front_cat(own,crate::kda::front(kw,&other).expect("KDA front for the gathered rows")));
                    }
                    z
                } else if half_z {crate::tp::all_gather_rows(&z_half.to_kind(tch::Kind::Half)).to_kind(tch::Kind::Float)} else {crate::tp::all_gather_rows(&z_half)}};
            let _att=Nvtx::new(nv,if layer.kda.is_some(){"kda"}else{"mla"});
            let old=prior.as_mut().map(|p|p.next().unwrap());
            let (a_half,ls)=crate::weights::with_sp_scatter(|| if let Some(kw)=&layer.kda {
                let st0=match old {Some(LayerState::Kda(s))=>s,Some(_)=>panic!("层 {i} 状态类型不符"),None=>crate::kda::KdaState::with_heads(z.device(),kw.wq.size()[0]/128)};
                let (o,st)=match kfront.take() {Some(f)=>crate::kda::kda_forward_state_front(kw,&z,st0,f),None=>crate::kda::kda_forward_state(kw,&z,st0)};(o,LayerState::Kda(st))
            } else {
                let w=layer.mla.as_ref().unwrap();
                let mut st=match old {Some(LayerState::MlaLatent(s))=>s,Some(_)=>panic!("latent MLA requires latent state"),None=>crate::mla_latent::State::new(w,crate::mla_latent::capacity())};
                let o=crate::mla_latent::chunk(w,&z,&mut st);(o,LayerState::MlaLatent(st))
            });
            assert_eq!(a_half.size()[0],half,"SP attention must return this rank's rows");
            drop(_att);
            states.push(ls);
            let (r2,pre2,z2_half)={let _g=Nvtx::new(nv,"mhc-ffn");crate::mhc::mhc_post_pre(&a_half,&residual,&pre,&layer.hc.ffn_fn,&layer.hc.ffn_scale,&layer.hc.ffn_base,&layer.hc.post_ln)};
            residual=r2;
            // P7 (GLM53_PREFILL_SP_HALF_GATHER=1, L1: the router GEMM runs on this rank's rows): route the owned rows,
            // then all-gather the Half expert input and the packed routes ([rows,16] FP32: ids exact, weights) instead
            // of the FP32 z2. Every z2 consumer except the router reads the Half rows.
            let mut shared_own:Option<Tensor>=None;
            let pre_routed=if half_gather && layer.moe.is_some() && !crate::deep_probe::next_input_pending() {
                let mm=layer.moe.as_ref().unwrap();
                let (ti,wi)={let _g=Nvtx::new(nv,"route");crate::moe::route(&z2_half,&mm.w_gate,&mm.bias,8)};
                let packed=Tensor::cat(&[ti.to_kind(tch::Kind::Float),wi.to_kind(tch::Kind::Float)],1).contiguous();
                let x16_half=z2_half.to_kind(tch::Kind::Half);
                let _g=Nvtx::new(nv,"gather-z2");
                // GLM53_PREFILL_AG_OVERLAP: the shared expert's own-row partial is computed while x16 and the routes are
                // gathered; the other rows' partial follows (row-split GEMMs are bitwise the full ones).
                let (x16,routes)=if ag_overlap && shared_fuse {
                    let pgx=crate::tp::all_gather_rows_start(&x16_half);let pgr=crate::tp::all_gather_rows_start(&packed);
                    shared_own=Some({let _g=Nvtx::new(nv,"shared-own");crate::weights::mm16_partial(&crate::moe::shared_activation_with_half(mm,&x16_half.to_kind(tch::Kind::Float),Some(&x16_half)),&mm.sh_wd)});
                    (pgx.finish(),pgr.finish())
                } else {(crate::tp::all_gather_rows(&x16_half),crate::tp::all_gather_rows(&packed))};
                Some((x16,routes.narrow(1,0,8).to_kind(tch::Kind::Int64),routes.narrow(1,8,8).contiguous()))
            } else {None};
            // GLM53_PREFILL_SHARED_HALF: with pre-routed rows and the fused shared path, z2's FP32 widening of x16 has no
            // consumer (the shared expert reads x16); it is made only if the shared expert falls back.
            let skip_z2=crate::moe::shared_half_enabled() && shared_fuse && pre_routed.is_some();
            let z2=if skip_z2 {Tensor::empty([0],(tch::Kind::Float,x.device()))} else if let Some((x16,_,_))=pre_routed.as_ref() {x16.to_kind(tch::Kind::Float)}
                else {let _g=Nvtx::new(nv,"gather-z2");crate::tp::all_gather_rows(&z2_half)};
            let _ffn=Nvtx::new(nv,if layer.moe.is_some(){"moe"}else{"dense-mlp"});
            let m_half=if let Some(mm)=&layer.moe {
                let overridden=crate::deep_probe::next_input_pending();
                let (x16,topi,wts)=if let Some(v)=pre_routed {v} else {
                    let (topi,wts)={let _g=Nvtx::new(nv,"route");crate::moe::route(&z2,&mm.w_gate,&mm.bias,8)};
                    (crate::deep_probe::expert_input(&z2),topi,wts)};
                // P8 (GLM53_PREFILL_HALF_REUSE=1): the shared expert consumes the routed lane's Half z2
                // (same RN conversion) instead of converting z2 again.
                let half=(half_reuse&&!overridden).then_some(&x16);
                if shared_fuse {
                    // P3b (GLM53_PREFILL_SHARED_FUSE=1): shared partial first, then the grouped reduce adds
                    // it after the fixed 8-slot sum and writes the reduce-scatter input directly.
                    let shared={let _g=Nvtx::new(nv,"shared");match shared_own.take() {
                        Some(own)=>{let ox=x16.narrow(0,(1-tp.rank as i64)*sp_rows,sp_rows).contiguous();
                            let other=crate::weights::mm16_partial(&crate::moe::shared_activation_with_half(mm,&ox.to_kind(tch::Kind::Float),Some(&ox)),&mm.sh_wd);
                            crate::tp::cat_rank_rows(&own,&other)}
                        None=>match crate::moe::shared_partial_half(mm,&x16) {Some(p)=>p,None=>{
                            let z2f=if skip_z2 {x16.to_kind(tch::Kind::Float)} else {z2.shallow_clone()};
                            crate::weights::mm16_partial(&crate::moe::shared_activation_with_half(mm,&z2f,half),&mm.sh_wd)}}}};
                    let y={let _g=Nvtx::new(nv,"experts");self.fast.as_mut().unwrap().expert_grouped_add(i,&x16,&topi,&wts,grouped=="recon",Some(&shared))};
                    drop(shared);
                    let _g=Nvtx::new(nv,"rs-moe");crate::tp::reduce_scatter_rows(&y)
                } else {
                    let y=self.fast.as_mut().unwrap().expert_grouped(i,&x16,&topi,&wts,grouped=="recon");
                    crate::moe::sp_finish_with_half(y.contiguous(),mm,&z2,half)
                }
            } else {
                crate::weights::with_sp_scatter(||mlp_forward_f(i,layer,&z2,&mut self.native,&mut self.pool))
            };
            if collect && [5,14,24,33,42].contains(&i) {
                // P6 (GLM53_PREFILL_FEATURE_REUSE=1): the next layer's fused post+pre writes the same
                // residual (bitwise equal to mhc_post); contract that instead of a second mhc_post.
                if feature_reuse && i<n-1 {feature_pending=true;}
                else {features.push(crate::tp::all_gather_rows(&hc_contract(&mhc_post(&m_half,&residual,&pre2))));}
            }
            if i==n-1 {residual=mhc_post(&m_half,&residual,&pre2);} else {deferred=Some((pre2,m_half));}
        }
        crate::moefast::route_error_check(self.w.device);
        let s=hc_contract(&residual);
        let last_row=s.narrow(0,half-1,1);
        let s_last=crate::tp::all_gather_rows(&last_row).narrow(0,1,1);
        let sq=&s_last*&s_last;let ms=sq.mean_dim(&[-1i64][..],true,tch::Kind::Float);
        let z=&self.w.final_norm*(s_last*(ms+1e-5).rsqrt());
        let logits=self.w.logits(&z);
        Some((logits,DecodeStates(states),features))
    }
    fn prefill_record_impl(&mut self,ids:&Tensor,ds:Option<DecodeStates>,collect:bool,last:bool)->(Tensor,DecodeStates,Vec<Tensor>) {
        if last && self.sp_eligible(ids.size()[0]) {
            return self.prefill_record_sp(ids,ds,collect).expect("SP prefill");
        }
        self.prefill_record_impl_inner(ids,ds,collect,last)
    }
    fn prefill_record_impl_inner(&mut self,ids:&Tensor,ds:Option<DecodeStates>,collect:bool,last:bool)->(Tensor,DecodeStates,Vec<Tensor>) {
        let mut features=Vec::new();
        let timing=std::env::var("GLM53_PREFILL_TIMING").as_deref()==Ok("1");
        let mut clock=std::time::Instant::now();
        let mut stamp=|layer:usize,stage:&str|{if timing{tch::Cuda::synchronize(0);eprintln!("[prefill-stage] rows={} layer={layer} stage={stage} ms={:.6}",ids.size()[0],clock.elapsed().as_secs_f64()*1000.);clock=std::time::Instant::now();}};
        if crate::mla_latent::enabled() {
            if let Some(s)=ds.as_ref().and_then(|d|d.0.iter().find_map(|s|match s {
                LayerState::MlaLatent(s)=>Some(s),_=>None,
            })) { s.ensure_room(ids.size()[0]); }
            else { assert!(ids.size()[0]<=crate::mla_latent::capacity(),"prompt exceeds latent capacity"); }
        }
        let x = self.w.embed_prefill(ids);
        let mut residual = hc_expand(&x);
        // Diagnostic: the SP path materializes the first residual (fused MHC pre on layer 0).
        if std::env::var("GLM53_PREFILL_FIRST_CONTIG").as_deref()==Ok("1") {residual=residual.contiguous();}
        let n = self.w.layers.len();
        let mut prior=ds.map(|d|{assert_eq!(d.0.len(),n);d.0.into_iter()});
        let mut deferred: Option<(crate::mhc::PreOut, Tensor)> = None;
        let mut states: Vec<LayerState> = Vec::with_capacity(n);
        for (i, layer) in self.w.layers.iter().enumerate() {
            crate::deep_probe::begin_layer();
            let (pre, z) = if let Some((prev, m)) = deferred.take() {
                let (r,p,z)=crate::mhc::mhc_post_pre(&m,&residual,&prev,&layer.hc.attn_fn,&layer.hc.attn_scale,&layer.hc.attn_base,&layer.hc.in_ln);
                residual=r;(p,z)
            } else {mhc_pre(&residual, &layer.hc.attn_fn, &layer.hc.attn_scale, &layer.hc.attn_base, &layer.hc.in_ln)};
            if crate::ablate::capturing() {crate::ablate::capture(i,&z);}
            crate::deep_probe::before_attention(i,&z,None);stamp(i,"input-mhc");
            let old=prior.as_mut().map(|p|p.next().unwrap());
            let (a, ls) = if let Some(kw) = &layer.kda {
                let st0 = match old {
                    Some(LayerState::Kda(s)) => s,
                    Some(_) => panic!("层 {i} 状态类型不符"),
                    None => crate::kda::KdaState::with_heads(z.device(), kw.wq.size()[0] / 128),
                };
                let (o, st) = crate::kda::kda_forward_state(kw, &z, st0);
                (o, LayerState::Kda(st))
            } else if crate::mla_latent::enabled() {
                let w=layer.mla.as_ref().unwrap();
                let mut st=match old {
                    Some(LayerState::MlaLatent(s))=>s,
                    Some(_)=>panic!("latent MLA requires latent state"),
                    None=>crate::mla_latent::State::new(w,crate::mla_latent::capacity()),
                };
                let output=crate::mla_latent::chunk(w,&z,&mut st);
                (output,LayerState::MlaLatent(st))
            } else {
                let st0 = match old {
                    Some(LayerState::Mla(s)) => s,
                    Some(_) => panic!("层 {i} 状态类型不符"),
                    None => crate::mla::MlaState::with_heads(z.device(), layer.mla.as_ref().unwrap().q_b.size()[0] / 256),
                };
                let (o, st) = crate::mla::mla_forward_state(layer.mla.as_ref().unwrap(), &z, st0);
                (o, LayerState::Mla(st))
            };
            stamp(i,if layer.kda.is_some(){"kda"}else{"mla"});
            crate::deep_probe::after_attention(i,&a,&ls);
            states.push(ls);
            let (r2, pre2, z2) = crate::mhc::mhc_post_pre(&a,&residual,&pre,&layer.hc.ffn_fn,&layer.hc.ffn_scale,&layer.hc.ffn_base,&layer.hc.post_ln);
            residual = r2;
            let m = if self.fast.is_some() && layer.moe.is_some()
                && std::env::var("GLM53_NO_FAST_PREFILL").is_err()
            {
                // Resident TP2 prefill can reuse EXL3 expert weights across up
                // to 32 rows. Keep the routed/shared reduction after all rows.
                let mm = layer.moe.as_ref().unwrap();
                let (topi, wts) = crate::moe::route(&z2, &mm.w_gate, &mm.bias, 8);
                let x16 = crate::deep_probe::expert_input(&z2);
                let tt = z2.size()[0];
                let dev = self.w.device;
                let cooperative=self.fast.as_ref().unwrap().assume_hot
                    && crate::tp::world().world==2
                    && std::env::var("GLM53_PREFILL_COOP").as_deref()==Ok("1");
                let grouped=std::env::var("GLM53_PREFILL_GROUPED").unwrap_or_else(|_|"0".into());
                assert!(["0","direct","recon"].contains(&grouped.as_str()));
                // the grouped prefill MoE (moe_exl3.cuh) wins from ~64 rows
                let is_grouped=cooperative && tt>=64 && grouped!="0";
                let y=if is_grouped {
                    self.fast.as_mut().unwrap().expert_grouped(i,&x16,&topi,&wts,grouped=="recon")
                }else {Tensor::empty([tt,4096],(Kind::Float,z2.device()))};
                // Each non-grouped row is assigned exactly once below.
                let mut t=if is_grouped{tt}else{0};
                while t<tt {
                    let rows=(tt-t).min(32);
                    if cooperative && rows>=8 {
                        let yb=self.fast.as_mut().unwrap().expert_cooperative(i,
                            &x16.narrow(0,t,rows).contiguous(),
                            &topi.narrow(0,t,rows).contiguous(),&wts.narrow(0,t,rows));
                        y.narrow(0,t,rows).copy_(&yb);t+=rows;continue;
                    }
                    let f = self.fast.as_mut().unwrap();
                    let sel_t=if f.assume_hot && std::env::var("GLM53_PREFILL_BATCH").as_deref()==Ok("1") {
                        f.sel_device(i,&topi.get(t),dev)
                    } else {
                    let host: Vec<i64> = topi.get(t).to_device(tch::Device::Cpu).try_into().expect("topi");
                    let es: Vec<usize> = host.iter().map(|&e| e as usize).collect();
                    f.ensure_many(i, &es, dev);
                    let slots: Vec<i64> = es.iter().map(|&e| f.slot_of(i, e)).collect();
                    Tensor::from_slice(&slots).view([1, -1]).to_device(dev)
                    };
                    let yb = f.expert_batch_sel(&x16.get(t).unsqueeze(0), &sel_t, &wts.get(t), dev);
                    let _ = y.narrow(0, t, 1).copy_(&yb);
                    t+=1;
                }
                crate::moe::finish_tp(y.contiguous(),mm,&z2)
            } else {
                mlp_forward_f(i, layer, &z2, &mut self.native, &mut self.pool)
            };
            stamp(i,"ffn-mhc");
            if collect && [5,14,24,33,42].contains(&i) {features.push(hc_contract(&mhc_post(&m,&residual,&pre2)));}
            if i == n - 1 {
                residual = mhc_post(&m, &residual, &pre2);
            } else {
                deferred = Some((pre2, m));
            }
        }
        let s = hc_contract(&residual);
        let sq = &s * &s;
        let ms = sq.mean_dim(&[-1i64][..], true, tch::Kind::Float);
        let z = &self.w.final_norm * (s * (ms + 1e-5).rsqrt());
        let z=if last{z.narrow(0,z.size()[0]-1,1)}else{z};
        let logits=self.w.logits(&z);stamp(n,"head");
        (logits, DecodeStates(states),features)
    }

    /// step:单 token O(1) 前向,返回该位置 logits。
    pub fn step(&mut self, tok: i64, ds: &mut DecodeStates) -> Tensor {
        if let Some(s)=ds.0.iter().find_map(|s|match s {LayerState::MlaLatent(s)=>Some(s),_=>None}) {
            s.ensure_room(1);
        }
        let ids = Tensor::from_slice(&[tok]).to_device(self.w.device);
        self.step_buf(&ids, ds)
    }

    /// step 的设备缓冲版(图捕获用:tok 从固定地址读取)。
    pub fn step_buf(&mut self, ids: &Tensor, ds: &mut DecodeStates) -> Tensor {
        self.step_record(ids,ds,false).0
    }

    pub fn step_record(&mut self,ids:&Tensor,ds:&mut DecodeStates,collect:bool)->(Tensor,Vec<Tensor>) {
        let mut features=Vec::new();
        let prof = std::env::var("GLM53_PROFILE").is_ok();
        let sync = |dev: tch::Device| { if prof { tch::Cuda::synchronize(match dev { tch::Device::Cuda(i) => i as i64, _ => -1 }); } };
        let dev0 = self.w.device;
        let t0 = std::time::Instant::now();
        let x = self.w.embed_tokens(&ids);
        let mut residual = hc_expand(&x);
        let n = self.w.layers.len();
        let mut deferred: Option<(crate::mhc::PreOut, Tensor)> = None;
        let (mut t_attn, mut t_mlp, mut t_hc) = (0.0f64, 0.0f64, 0.0f64);
        let mut tm = std::time::Instant::now();
        for (i, layer) in self.w.layers.iter().enumerate() {
            crate::deep_probe::begin_layer();
            if let Some((pre, m)) = deferred.take() {
                residual = mhc_post(&m, &residual, &pre);
            }
            let (pre, z) = mhc_pre(&residual, &layer.hc.attn_fn, &layer.hc.attn_scale, &layer.hc.attn_base, &layer.hc.in_ln);
            if crate::ablate::capturing() {crate::ablate::capture(i,&z);}
            if prof { sync(dev0); t_hc += tm.elapsed().as_secs_f64(); tm = std::time::Instant::now(); }
            crate::deep_probe::before_attention(i,&z,Some(&ds.0[i]));
            let a = match &mut ds.0[i] {
                LayerState::Kda(st) => crate::kda::kda_step(layer.kda.as_ref().unwrap(), &z, st),
                LayerState::Mla(st) => crate::mla::mla_step(layer.mla.as_ref().unwrap(), &z, st),
                LayerState::MlaG(st) => crate::mla::mla_step_g(layer.mla.as_ref().unwrap(), &z, st),
                LayerState::MlaLatent(st) => crate::mla_latent::step(layer.mla.as_ref().unwrap(), &z, st),
            };
            crate::deep_probe::after_attention(i,&a,&ds.0[i]);
            if prof { sync(dev0); t_attn += tm.elapsed().as_secs_f64(); tm = std::time::Instant::now(); }
            residual = mhc_post(&a, &residual, &pre);
            let (pre2, z2) = mhc_pre(&residual, &layer.hc.ffn_fn, &layer.hc.ffn_scale, &layer.hc.ffn_base, &layer.hc.post_ln);
            let m = if z2.size()[0] == 1 && layer.moe.is_some() && self.fast.is_some() {
                // M1.4:mgemm 合批 + 设备侧选路(GLM53_NO_SELDEV=1 回退 host 选路)
                let mm = layer.moe.as_ref().unwrap();
                let (topi, wts) = crate::moe::route(&z2, &mm.w_gate, &mm.bias, 8);
                let x16 = crate::deep_probe::expert_input(&z2);
                let f = self.fast.as_mut().unwrap();
                let sel = if std::env::var("GLM53_NO_SELDEV").is_ok() {
                    let host: Vec<i64> = topi.get(0).to_device(tch::Device::Cpu).try_into().expect("topi");
                    let es: Vec<usize> = host.iter().map(|&e| e as usize).collect();
                    f.ensure_many(i, &es, self.w.device);
                    let slots: Vec<i64> = es.iter().map(|&e| f.slot_of(i, e)).collect();
                    Tensor::from_slice(&slots).view([1, -1]).to_device(self.w.device)
                } else {
                    f.sel_device(i, &topi.get(0), self.w.device)
                };
                let w8 = wts.get(0);
                let prof2 = std::env::var("GLM53_PROFILE2").is_ok();
                let tb = std::time::Instant::now();
                let yb = f.expert_batch_sel(&x16, &sel, &w8, self.w.device);
                if crate::moe::tp_pack_enabled() {
                    crate::moe::finish_tp(yb.to_kind(Kind::Float).contiguous(),mm,&z2)
                } else {
                let (mut tq_ar, tq_batch) = (0.0f64, tb.elapsed().as_secs_f64());
                let yb = if crate::tp::is_tp() {
                    let ta = std::time::Instant::now();
                    let yb32 = yb.to_kind(tch::Kind::Float).contiguous();
                    crate::tp::allreduce(&yb32); // 专家行切部分和 → 全和
                    if prof2 {
                        tch::Cuda::synchronize(match self.w.device { tch::Device::Cuda(i) => i as i64, _ => -1 });
                    }
                    tq_ar = ta.elapsed().as_secs_f64();
                    yb32
                } else {
                    yb
                };
                let y = yb + crate::moe::shared_forward(mm, &z2);
                if prof2 && i == 20 {
                    println!("[prof2] L20 batch {:.2}ms allreduce+drain {:.2}ms",
                             tq_batch * 1e3, tq_ar * 1e3);
                }
                y
                }
            } else {
                mlp_forward_f(i, layer, &z2, &mut self.native, &mut self.pool)
            };
            if prof { sync(dev0); t_mlp += tm.elapsed().as_secs_f64(); tm = std::time::Instant::now(); }
            if collect && [5,14,24,33,42].contains(&i) {features.push(hc_contract(&mhc_post(&m,&residual,&pre2)));}
            if i == n - 1 {
                residual = mhc_post(&m, &residual, &pre2);
            } else {
                deferred = Some((pre2, m));
            }
        }
        let s = hc_contract(&residual);
        let sq = &s * &s;
        let ms = sq.mean_dim(&[-1i64][..], true, tch::Kind::Float);
        let z = &self.w.final_norm * (s * (ms + 1e-5).rsqrt());
        let out = self.w.logits(&z).get(0);
        if prof {
            sync(dev0);
            let total = t0.elapsed().as_secs_f64();
            println!("[prof] 步 {:.1}ms | attn {:.1} mlp {:.1} mhc-pre {:.1} 其余 {:.1}",
                     total * 1e3, t_attn * 1e3, t_mlp * 1e3, t_hc * 1e3,
                     (total - t_attn - t_mlp - t_hc) * 1e3);
        }
        (out,features)
    }

}

pub fn mlp_forward_f(
    i: usize,
    layer: &crate::weights::LayerWeights,
    z2: &Tensor,
    native: &mut Option<crate::moe::NativeExpertPool>,
    pool: &mut crate::moe::ExpertPool,
) -> Tensor {
        use tch::Kind;
        if let Some(dm) = &layer.dense {
            let g = crate::weights::mm16(z2, &dm.wg).silu();
            crate::weights::row_mm16(&(g * crate::weights::mm16(z2, &dm.wu)), &dm.wd)
        } else if let Some(mm) = &layer.moe {
            let (topi, wts) = crate::moe::route(z2, &mm.w_gate, &mm.bias, 8);
            let mut y = Tensor::zeros([z2.size()[0], z2.size()[1]], (Kind::Float, z2.device()));
            if let Some(np) = native.as_mut() {
                let x16 = z2.to_kind(Kind::Half);
                for t in 0..z2.size()[0] {
                    let one = x16.get(t).unsqueeze(0);
                    for ki in 0..topi.size()[1] {
                        let e = topi.get(t).get(ki).int64_value(&[]) as usize;
                        let pv = np.expert(i, e, z2.device());
                        let (a, b, c) = &pv[0];
                        let (d, e2, f) = &pv[1];
                        let (g, h, i2) = &pv[2];
                        let projs = [(a.shallow_clone(), b.shallow_clone(), c.shallow_clone()),
                                     (d.shallow_clone(), e2.shallow_clone(), f.shallow_clone()),
                                     (g.shallow_clone(), h.shallow_clone(), i2.shallow_clone())];
                        let eo = crate::moe::NativeExpertPool::expert_forward(&one, &projs, z2.device());
                        let contrib = wts.get(t).get(ki) * eo;
                        let old = y.narrow(0, t, 1);
                        let _ = y.narrow(0, t, 1).copy_(&(old + contrib));
                    }
                }
            } else {
                for t in 0..z2.size()[0] {
                    for ki in 0..topi.size()[1] {
                        let e = topi.get(t).get(ki).int64_value(&[]) as usize;
                        let (wg, wu, wd) = pool.expert(i, e, z2.device());
                        let one = z2.get(t).unsqueeze(0);
                        let eo = crate::moe::expert_forward(&one, wg, wu, wd);
                        let contrib = wts.get(t).get(ki) * eo;
                        let old = y.narrow(0, t, 1);
                        let _ = y.narrow(0, t, 1).copy_(&(old + contrib));
                    }
                }
            }
            y + crate::moe::shared_forward(mm, z2)
        } else {
            panic!("层 {i} 无 MLP 权重")
        }
    }

impl Engine {
    /// M1.3 增量贪心:prefill 一次 + N 步 O(1)。
    /// GLM53_GRAPH=1 时走图模式(TP2 + 全专家驻留前提,见 greedy_incremental_graph)。
    pub fn greedy_incremental(&mut self, ids: &[i64], max_new: usize) -> Vec<i64> {
        if std::env::var("GLM53_GRAPH").is_ok() {
            return self.greedy_incremental_graph(ids, max_new);
        }
        self.greedy_incremental_dbg(ids, max_new, false).0
    }

    /// M1② 图模式:prefill → 状态转图兼容(MLA 定长窗,KDA 原地) → 2 步 eager 暖内核
    /// → 捕获一步 → 回放。前提:TP2 且全专家驻留(preload_all;回放中 miss 会静默
    /// 算错——sel=-1 被内核跳过或读错槽,无任何报错)。
    /// 窗口上限 GLM53_GRAPH_WIN 默认 512。
    pub fn greedy_incremental_graph(&mut self, ids: &[i64], max_new: usize) -> Vec<i64> {
        self.greedy_incremental_graph_dbg(ids, max_new, false).0
    }

    pub fn greedy_incremental_graph_dbg(&mut self, ids: &[i64], max_new: usize, keep_logits: bool) -> (Vec<i64>, Vec<Tensor>) {
        if max_new == 0 { return (Vec::new(), Vec::new()); }
        assert!(!ids.is_empty(), "prompt 不能为空");
        if max_new <= 3 { return self.greedy_incremental_dbg(ids, max_new, keep_logits); }
        assert!(self.fast.is_some(), "图模式需要 fast 专家池");
        assert!(std::env::var("GLM53_NO_SELDEV").is_err()
            && std::env::var("GLM53_PROFILE").is_err()
            && std::env::var("GLM53_PROFILE2").is_err(), "图模式禁止 host 选路/同步 profiling");
        assert!(crate::tp::is_tp(),
            "GLM53_GRAPH 仅支持 TP2:单机 fast_cap 装不下全部专家,LRU 逐出会在回放期静默错算");
        if let Some(f) = self.fast.as_ref() {
            assert!(f.is_fully_loaded(),
                "图模式要求专家全驻留(先 preload_all);当前 {}/{}",
                f.resident_count(), f.expected_total());
        }
        let dev = self.w.device;
        let dev_i = match dev { tch::Device::Cuda(i) => i as i64, _ => -1 };
        let max_t: i64 = if crate::mla_latent::enabled() {crate::mla_latent::capacity()}
            else {std::env::var("GLM53_GRAPH_WIN").ok().and_then(|s| s.parse().ok()).unwrap_or(512)};
        assert!(ids.len() as i64 + max_new as i64 <= max_t, "超出图窗口 {max_t}");
        let ids_t = Tensor::from_slice(ids).to_device(dev);
        let (logits, ds) = self.prefill(&ids_t);
        let mut dsg = DecodeStates(ds.0.into_iter().map(|s| match s {
            LayerState::Kda(k) => LayerState::Kda(k),
            LayerState::Mla(m) => LayerState::MlaG(crate::mla::MlaStateG::from_state(&m, max_t)),
            LayerState::MlaG(g) => LayerState::MlaG(g),
            LayerState::MlaLatent(s) => LayerState::MlaLatent(s),
        }).collect());
        let old_assume_hot = self.fast.as_ref().unwrap().assume_hot;
        if let Some(f) = self.fast.as_mut() {
            f.assume_hot = true; // 图内不许 host 同步;调用方须保证全驻留
        }
        let mut tok_buf = Tensor::zeros([1], (Kind::Int64, dev));
        let mut out: Vec<i64> = Vec::with_capacity(max_new);
        let mut logs = Vec::new();
        let mut lg = logits.get(logits.size()[0] - 1);
        // 暖 2 步 eager(算子/autotune/图池形状稳定)
        for _ in 0..2 {
            let nxt = lg.argmax(-1, false).int64_value(&[]);
            out.push(nxt);
            if keep_logits { logs.push(lg.copy()); }
            let _ = tok_buf.copy_(&Tensor::from_slice(&[nxt]).to_device(dev));
            lg = self.step_buf(&tok_buf, &mut dsg);
        }
        let nxt = lg.argmax(-1, false).int64_value(&[]);
        out.push(nxt);
        if keep_logits { logs.push(lg.copy()); }
        let _ = tok_buf.copy_(&Tensor::from_slice(&[nxt]).to_device(dev));
        tch::Cuda::synchronize(dev_i);
        // 捕获(不执行)
        crate::tp::graph::begin().expect("graph begin");
        let lg_g = self.step_buf(&tok_buf, &mut dsg);
        crate::tp::graph::end().expect("graph end");
        // 回放循环
        let t0 = std::time::Instant::now();
        let mut n_replay = 0usize;
        while out.len() < max_new {
            crate::tp::graph::replay().expect("replay");
            n_replay += 1;
            let nxt = lg_g.argmax(-1, false).int64_value(&[]);
            out.push(nxt);
            if keep_logits { logs.push(lg_g.copy()); }
            if out.len() >= max_new { break; }
            let _ = tok_buf.copy_(&Tensor::from_slice(&[nxt]).to_device(dev));
        }
        let el = t0.elapsed().as_secs_f64();
        if n_replay > 0 {
            println!("[graph] 回放 {} 步,{:.2} ms/token", n_replay, el * 1e3 / n_replay as f64);
        }
        self.fast.as_mut().unwrap().assume_hot = old_assume_hot;
        (out, logs)
    }

    pub fn greedy_incremental_dbg(&mut self, ids: &[i64], max_new: usize, keep_logits: bool) -> (Vec<i64>, Vec<Tensor>) {
        if max_new == 0 { return (Vec::new(), Vec::new()); }
        assert!(!ids.is_empty(), "prompt 不能为空");
        let ids_t = Tensor::from_slice(ids).to_device(self.w.device);
        let (logits, mut ds) = self.prefill(&ids_t);
        let mut lg_cur = logits.get(logits.size()[0] - 1); // 产出下一个 token 的 logits
        let mut out = Vec::with_capacity(max_new);
        let mut logs = Vec::new();
        for step_i in 0..max_new {
            let nxt = lg_cur.argmax(-1, false).int64_value(&[]);
            out.push(nxt);
            if keep_logits { logs.push(lg_cur.shallow_clone()); }
            if step_i + 1 == max_new { break; }
            lg_cur = self.step(nxt, &mut ds);
        }
        (out, logs)
    }
}

/// 状态深拷贝(回滚/分支契约:tch 的 clone 是浅拷贝共享存储,必须显式复制)。
pub fn snapshot(ds: &DecodeStates) -> DecodeStates {
    crate::spec_probe::settle_side_commit();
    DecodeStates(ds.0.iter().map(snapshot_layer).collect())
}

pub(crate) fn snapshot_layer(s:&LayerState)->LayerState {
    let cp = |t: &Tensor| { let mut c = Tensor::empty_like(t); let _ = c.copy_(t); c };
    match s {
        LayerState::Kda(k) => LayerState::Kda(crate::kda::KdaState { h: cp(&k.h), conv: cp(&k.conv) }),
        LayerState::Mla(m) => LayerState::Mla(crate::mla::MlaState { k: cp(&m.k), v: cp(&m.v), len: m.len }),
        LayerState::MlaG(g) => LayerState::MlaG(crate::mla::MlaStateG {
            k: cp(&g.k), v: cp(&g.v), len: cp(&g.len), max_t: g.max_t }),
        LayerState::MlaLatent(s) => LayerState::MlaLatent(s.snapshot()),
    }
}

/// 两组状态逐层最大绝对差(等价性检查)。
pub fn states_max_diff(a: &DecodeStates, b: &DecodeStates) -> f64 {
    if a.0.len() != b.0.len() { return f64::INFINITY; }
    let mut worst = 0.0f64;
    for (sa, sb) in a.0.iter().zip(b.0.iter()) {
        let d = |x: &Tensor, y: &Tensor| -> f64 {
            if x.size() != y.size() { return f64::INFINITY; }
            if x.numel() == 0 { return 0.0; }
            let value = f64::try_from((x - y).abs().max()).unwrap_or(f64::INFINITY);
            if value.is_finite() { value } else { f64::INFINITY }
        };
        match (sa, sb) {
            (LayerState::Kda(x), LayerState::Kda(y)) => {
                worst = worst.max(d(&x.h, &y.h)).max(d(&x.conv, &y.conv));
            }
            (LayerState::Mla(x), LayerState::Mla(y)) => {
                worst = worst.max(d(&x.k, &y.k)).max(d(&x.v, &y.v));
                if x.len != y.len { return f64::INFINITY; }
            }
            (LayerState::MlaG(x), LayerState::MlaG(y)) => {
                worst = worst.max(d(&x.k, &y.k)).max(d(&x.v, &y.v)).max(d(&x.len, &y.len));
            }
            (LayerState::MlaLatent(x),LayerState::MlaLatent(y)) => {worst=worst.max(x.max_diff(y));}
            _ => return f64::INFINITY,
        }
    }
    worst
}

pub fn layer_plan_idx_public(_l: &crate::weights::LayerWeights) -> usize { 0 }

pub(crate) fn restore(dst: &mut DecodeStates, src: &DecodeStates) {
    crate::spec_probe::settle_side_commit();
    assert_eq!(dst.0.len(), src.0.len());
    for (d,s) in dst.0.iter_mut().zip(&src.0) {
        match (d,s) {
            (LayerState::Kda(d),LayerState::Kda(s)) => { d.h.copy_(&s.h); d.conv.copy_(&s.conv); }
            (LayerState::MlaG(d),LayerState::MlaG(s)) => { d.k.copy_(&s.k); d.v.copy_(&s.v); d.len.copy_(&s.len); }
            (LayerState::MlaLatent(d),LayerState::MlaLatent(s)) => d.restore(s),
            (LayerState::Mla(d),LayerState::Mla(s)) => { assert_eq!(d.k.size(),s.k.size()); d.k.copy_(&s.k); d.v.copy_(&s.v); d.len=s.len; }
            _ => panic!("state restore requires matching state types"),
        }
    }
}
