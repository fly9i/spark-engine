//! Compressed latent MLA with incremental DSA selection. Bounded, graph-compatible cache.
//! Absorbed products reassociate FP arithmetic; compare quality, not raw-logit bit identity.
use tch::{Tensor,Kind};
use crate::{weights::{MlaWeights,mm16,row_mm16},dsa};

pub fn enabled()->bool {std::env::var("GLM53_MLA_LATENT").as_deref()==Ok("1")}
fn active_copy_enabled()->bool {std::env::var("GLM53_MLA_ACTIVE_COPY").as_deref()==Ok("1")}
/// Copy only initialized rows; all consumers must mask the inactive suffix.
/// The extent stays on device and can change on every graph replay.
pub(crate) fn active_copy(dst:&Tensor,src:&Tensor,len:&Tensor,divisor:i32) {
    assert_eq!(src.size(),dst.size());assert_eq!(src.kind(),dst.kind());
    assert!(src.is_contiguous() && dst.is_contiguous());assert_eq!(src.size().len(),2);
    assert_eq!(len.kind(),Kind::Int64);assert_eq!(len.numel(),1);
    let bytes=match src.kind(){Kind::Half=>2,Kind::Float=>4,Kind::Uint8=>1,_=>panic!("unsupported cache kind")};
    let width=src.size()[1]*bytes;assert_eq!(width%16,0);
    extern "C"{fn rs_active_cache_copy(src:*const std::ffi::c_void,dst:*mut std::ffi::c_void,len:*const i64,capacity:i32,row_vectors:i32,divisor:i32)->i32;}
    assert_eq!(unsafe{rs_active_cache_copy(src.data_ptr(),dst.data_ptr(),len.data_ptr().cast(),src.size()[0].try_into().unwrap(),(width/16).try_into().unwrap(),divisor)},0);
}
/// A sliced [heads, 1, latent] batch has gaps between heads. Explicitly
/// removing the singleton dimension selects shared-key GEMM instead of
/// strided batched GEMV. Both operands and accumulation remain FP32.
fn batch_scores(absorbed:&Tensor,row:i64,c:&Tensor,is_tree:bool)->Tensor {
    let q=absorbed.narrow(1,row,1);
    let configured=std::env::var("GLM53_MLA_SCORE_2D").as_deref()==Ok("1");
    let selected=if configured {
        let scope=std::env::var("GLM53_MLA_SCORE_2D_SCOPE").unwrap_or_else(|_|"all".into());
        let min_rows=std::env::var("GLM53_MLA_SCORE_2D_MIN_ROWS").ok().map(|v|v.parse::<i64>().unwrap()).unwrap_or(1);
        assert!(min_rows>0);
        let scope_ok=match scope.as_str(){"all"=>true,"tree"=>is_tree,"prefill"=>!is_tree,_=>panic!("invalid MLA score scope")};
        scope_ok && absorbed.size()[1]>=min_rows
    }else{false};
    if absorbed.size()[1]>1 && selected {
        q.squeeze_dim(1).matmul(&c.transpose(0,1)).unsqueeze(1)/16.
    } else {q.matmul(&c.transpose(0,1))/16.}
}
/// GLM53_KV_FP8=1: latent rows are 512 FP8 e4m3 + 4 FP32 tile scales (528 bytes, L3).
pub(crate) fn kv_fp8_enabled()->bool {std::env::var("GLM53_KV_FP8").as_deref()==Ok("1")}
pub(crate) const KV8_ROW:i64=528;
fn is_fp8(latent:&Tensor)->bool {latent.kind()==Kind::Uint8}
fn assert_half_latent(latent:&Tensor,path:&str) {assert!(!is_fp8(latent),"{path} does not support GLM53_KV_FP8; use the fused attention paths");}
/// Store FP16 latent rows [n,512] at device position `pos` (+row).
pub(crate) fn store_rows(latent:&Tensor,rows:&Tensor,pos:&Tensor) {
    if !is_fp8(latent) {let n=rows.size()[0];if n==1{let _=latent.shallow_clone().index_copy_(0,pos,rows);}
        else{let idx=pos+Tensor::arange(n,(Kind::Int64,pos.device()));let _=latent.shallow_clone().index_copy_(0,&idx,rows);}return;}
    let rows=rows.to_kind(Kind::Half).contiguous();assert_eq!(rows.size()[1],512);assert_eq!(pos.kind(),Kind::Int64);
    extern "C"{fn rs_latent_fp8_store(rows:*const std::ffi::c_void,n:i32,pos_dev:*const i64,pos_host:i64,latent:*mut std::ffi::c_void)->i32;}
    assert_eq!(unsafe{rs_latent_fp8_store(rows.data_ptr(),rows.size()[0] as i32,pos.data_ptr().cast(),0,latent.data_ptr())},0,"FP8 latent store");
}
/// Store FP16 latent rows [n,512] starting at host position `start`.
pub(crate) fn store_rows_at(latent:&Tensor,rows:&Tensor,start:i64) {
    if !is_fp8(latent) {latent.narrow(0,start,rows.size()[0]).copy_(rows);return;}
    let rows=rows.to_kind(Kind::Half).contiguous();assert_eq!(rows.size()[1],512);
    extern "C"{fn rs_latent_fp8_store(rows:*const std::ffi::c_void,n:i32,pos_dev:*const i64,pos_host:i64,latent:*mut std::ffi::c_void)->i32;}
    assert_eq!(unsafe{rs_latent_fp8_store(rows.data_ptr(),rows.size()[0] as i32,std::ptr::null(),start,latent.data_ptr())},0,"FP8 latent store");
}
/// Selected rows follow DSA layout: valid full-pool slots, invalid padding,
/// then three independent tail slots. Arbitrary holes are not accepted.
pub(crate) fn node_batch_enabled()->bool {std::env::var("GLM53_MLA_NODE_BATCH").as_deref()==Ok("1")}
pub(crate) fn sparse_attention(q:&Tensor,latent:&Tensor,selected:&Tensor)->Tensor {sparse_attention_splits(q,latent,selected,None)}
/// `splits`: None = the size rule below; Some(n) forces n (batched chain nodes keep the per-node split count).
pub(crate) fn sparse_attention_splits(q:&Tensor,latent:&Tensor,selected:&Tensor,splits:Option<i64>)->Tensor {
    let (heads,queries)=(q.size()[0],q.size()[1]);
    assert_eq!(q.size()[2],512);assert_eq!(q.kind(),Kind::Float);
    assert_eq!(q.stride()[2],1);assert_eq!(q.stride()[1],512);
    assert!(latent.is_contiguous());assert!(latent.kind()==Kind::Half||is_fp8(latent));
    assert!(selected.is_contiguous());assert_eq!(selected.kind(),Kind::Int64);
    let slots=selected.numel() as i64/queries;assert_eq!(selected.numel() as i64,queries*slots);
    let out=Tensor::empty([heads,queries,512],(Kind::Float,q.device()));
    let splits=splits.unwrap_or(if queries*heads<192{8}else{1});
    let scratch=Tensor::empty([if splits>1{splits*heads*queries*514}else{0}],(Kind::Float,q.device()));
    extern "C" {fn rs_latent_attention(q:*const f32,c:*const std::ffi::c_void,ids:*const i64,out:*mut f32,scratch:*mut f32,h:i32,t:i32,s:i32,stride:i32,splits:i32)->i32;
        fn rs_latent_attention_fp8(q:*const f32,c:*const std::ffi::c_void,ids:*const i64,out:*mut f32,scratch:*mut f32,h:i32,t:i32,s:i32,stride:i32,splits:i32)->i32;}
    let call=if is_fp8(latent){rs_latent_attention_fp8}else{rs_latent_attention};
    assert_eq!(unsafe{call(q.data_ptr().cast(),latent.data_ptr(),selected.data_ptr().cast(),out.data_ptr().cast(),scratch.data_ptr().cast(),heads as i32,queries as i32,slots as i32,q.stride()[0] as i32,splits as i32)},0);
    out
}
pub(crate) fn shared_attention(q:&Tensor,latent:&Tensor,selected:&Tensor,mode:i32)->Tensor {
    assert!((3..=8).contains(&mode));assert_eq!(q.kind(),Kind::Float);assert_eq!(q.size()[2],512);
    let mode=if is_fp8(latent){assert_eq!(mode,8,"GLM53_KV_FP8 prefill needs GLM53_MLA_PREFILL_TC=1 (mode 8)");9}else{mode};
    // P7: FP16 tensor-core attention (Q and probabilities in FP16, FP32 accumulation).
    let mode=if (mode==8||mode==9) && std::env::var("GLM53_MLA_PREFILL_F16").as_deref()==Ok("1"){mode+2}else{mode};
    assert_eq!(q.stride()[1],512);assert_eq!(q.stride()[2],1);
    assert!(latent.kind()==Kind::Half||is_fp8(latent));assert!(latent.is_contiguous());
    assert_eq!(selected.kind(),Kind::Int64);assert!(selected.is_contiguous());
    let (heads,queries)=(q.size()[0],q.size()[1]);let slots=selected.numel() as i64/queries;
    assert_eq!(selected.numel() as i64,queries*slots);assert!(slots>=3);
    let out=Tensor::empty([heads,queries,512],(Kind::Float,q.device()));
    extern "C"{fn rs_shared_latent(q:*const f32,c:*const std::ffi::c_void,ids:*const i64,out:*mut f32,h:i32,t:i32,s:i32,stride:i32,mode:i32)->i32;}
    assert_eq!(unsafe{rs_shared_latent(q.data_ptr().cast(),latent.data_ptr(),selected.data_ptr().cast(),out.data_ptr().cast(),heads as i32,queries as i32,slots as i32,q.stride()[0] as i32,mode)},0);
    out
}
// Large-prefill alternative: shared-key GEMMs, with the SAME DSA selected
// positions. Dense arithmetic does not make unselected/future keys visible.
fn prefill_dense_attention(q:&Tensor,latent:&Tensor,ids:&Tensor,visible_len:i64)->Tensor {
    let c=latent.narrow(0,0,visible_len).to_kind(Kind::Float);
    let heads=q.size()[0];let t=q.size()[1];let slots=ids.size()[1];let mut outputs=Vec::new();
    for start in (0..t).step_by(128) {
        let n=(t-start).min(128);let selected=ids.narrow(0,start,n);
        let index=selected.clamp_min(0).unsqueeze(0).expand([heads,n,slots],false);
        let score=q.narrow(1,start,n).contiguous().view([heads*n,512]).matmul(&c.transpose(0,1)).view([heads,n,visible_len])/16.;
        let chosen=score.gather(2,&index,false).masked_fill(&selected.lt(0).unsqueeze(0),f64::NEG_INFINITY).softmax(-1,Kind::Float);
        let mut weights=Tensor::zeros([heads,n,visible_len],(Kind::Float,q.device()));
        let _=weights.scatter_add_(2,&index,&chosen);
        outputs.push(weights.view([heads*n,visible_len]).matmul(&c).view([heads,n,512]));
    }
    Tensor::cat(&outputs,1)
}
fn fused_attention()->bool {std::env::var("GLM53_MLA_SPARSE_FUSED").as_deref()==Ok("1")}
pub fn capacity()->i64 {
    let n=std::env::var("GLM53_MAX_CONTEXT").ok().map(|s|s.parse().expect("GLM53_MAX_CONTEXT integer")).unwrap_or(4096);
    assert!(n>0,"GLM53_MAX_CONTEXT must be positive");n
}
/// q_a and kv_a read the same x: one C12 group launch when both are coded (their slices are bitwise the separate
/// projections: same split-K), otherwise the two projections.
fn qa_kva(w:&MlaWeights,x:&Tensor)->(Tensor,Tensor) {
    if let Some(y)=crate::c12::try_run_rows(x,&[&w.q_a,&w.kv_a]) {let n=w.q_a.size()[0];return (y.narrow(1,0,n),y.narrow(1,n,w.kv_a.size()[0]));}
    (mm16(x,&w.q_a),mm16(x,&w.kv_a))
}
fn norm(x:Tensor,w:&Tensor)->Tensor {
    if crate::kda::norm_fused_enabled() && x.kind()==Kind::Float && x.dim()==2 && x.stride()[1]==1 && x.stride()[0]>=x.size()[1] && x.device().is_cuda()
        && w.kind()==Kind::Float && w.is_contiguous() && w.numel() as i64==x.size()[1] {
        // Row-strided input allowed (q_a / kv_a slices of one C12 group output): same per-row arithmetic.
        let y=Tensor::empty([x.size()[0],x.size()[1]],(Kind::Float,x.device()));
        extern "C"{fn rs_rms_rows_ld(x:*const f32,ldx:i32,w:*const f32,y:*mut f32,rows:i32,d:i32,eps:f32)->i32;}
        assert_eq!(unsafe{rs_rms_rows_ld(x.data_ptr().cast(),x.stride()[0] as i32,w.data_ptr().cast(),y.data_ptr().cast(),x.size()[0] as i32,x.size()[1] as i32,1e-5)},0);
        return y;
    }
    let x=x.to_kind(Kind::Float);
    let ms=(&x*&x).mean_dim(&[-1i64][..],true,Kind::Float);
    x*(ms+1e-5).rsqrt()*w
}

pub struct State {
    pub latent:Tensor,
    pub index:dsa::State,
    pub len:Tensor,
    pub capacity:i64,
    /// GLM53_MLA_CHAIN_SHARED: a chain verifier node aliases its base's latent and pools;
    /// later nodes may overwrite pool row (len-1)/4, so the node keeps its own copy of it.
    pub(crate) pool_row:Option<Tensor>,
    // Immutable derived projection weights shared by snapshots. Cache accounting
    // reports these separately; they are not per-token KV storage.
    pub wk:Tensor,
    pub wv:Tensor,
}
pub(crate) fn half_bmm_enabled()->bool {std::env::var("GLM53_MLA_HALF_BMM").as_deref()==Ok("1")}
thread_local!{
    // kv_b ptr -> (owner, wk^T Half [heads,512,256], wv^T Half [heads,256,512]); built outside capture.
    static LATENT_HALF:std::cell::RefCell<std::collections::HashMap<usize,(Tensor,Tensor,Tensor)>>=std::cell::RefCell::new(std::collections::HashMap::new());
}
fn prefill_strided()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_MLA_PREFILL_STRIDED").as_deref()==Ok("1"))}
#[allow(clippy::too_many_arguments)]
fn bmm_tf32(a:&Tensor,lda:i64,sa:i64,b:&Tensor,ldb:i64,sb:i64,c:&Tensor,ldc:i64,sc:i64,m:i64,n:i64,k:i64,batch:i64) {
    assert!([a,b,c].iter().all(|t|t.kind()==Kind::Float));
    extern "C"{fn rs_bmm_f32_tf32(a:*const f32,lda:i64,sa:i64,b:*const f32,ldb:i64,sb:i64,c:*mut f32,ldc:i64,sc:i64,m:i32,n:i32,k:i32,batch:i32)->i32;}
    assert_eq!(unsafe{rs_bmm_f32_tf32(a.data_ptr().cast(),lda,sa,b.data_ptr().cast(),ldb,sb,c.data_ptr().cast(),ldc,sc,m as i32,n as i32,k as i32,batch as i32)},0,"MLA prefill strided bmm");
}
pub(crate) fn ensure_latent_half(w:&MlaWeights) {
    if !half_bmm_enabled(){return;}
    let key=w.kv_b.data_ptr() as usize;
    if LATENT_HALF.with(|c|c.borrow().contains_key(&key)) && (!crate::c12::enabled() || LATENT_C12.with(|c|c.borrow().contains_key(&key))) {return;}
    assert!(!crate::tp::graph::capturing(),"LATENT_HALF cache miss during graph capture");
    let heads=w.q_b.size()[0]/256;assert_eq!(w.kv_b.kind(),Kind::Half);
    let kv=w.kv_b.view([heads,512,512]);
    let wk_t=kv.narrow(1,0,256).transpose(1,2).contiguous();   // [heads,512,256]: W[n=latent][k=d]
    let wv_t=kv.narrow(1,256,256).contiguous();                // [heads,256,512]: W[n=v][k=latent]
    // The coding exists whenever C12 is on (about 12 MiB per MLA layer and rank) so graphs with and without
    // GLM53_MLA_BMM_C12 can be captured in one process (in-process A/B); the flag only selects the kernel.
    if crate::c12::enabled() && !LATENT_C12.with(|c|c.borrow().contains_key(&key)) {
        let n=|t:&Tensor|t.size()[1];
        let ck=crate::c12::encode_parts(&wk_t.reshape([heads*n(&wk_t),256]));let cv=crate::c12::encode_parts(&wv_t.reshape([heads*n(&wv_t),512]));
        LATENT_C12.with(|c|c.borrow_mut().insert(key,(w.kv_b.shallow_clone(),ck,cv)));
    }
    LATENT_HALF.with(|c|{c.borrow_mut().entry(key).or_insert((w.kv_b.shallow_clone(),wk_t,wv_t));});
}
/// GLM53_MLA_BMM_C12=1 (with GLM53_C12=1): the absorb/expand products read a C12 coding of the per-head Half weights
/// (0.75 of the bytes); bitwise the latent_half_bmm result (same MMA operands and order).
/// GLM53_MLA_BMM_WIDE=1 (L0): the C12 absorb/expand kernel takes up to 32 rows per call (2 or 4 token tiles), so batched
/// verify windows read each head's weights once instead of once per 8-row piece.
fn bmm_wide()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||bmm_c12_enabled()&&std::env::var("GLM53_MLA_BMM_WIDE").as_deref()==Ok("1"))}
pub(crate) fn bmm_c12_enabled()->bool {crate::c12::enabled() && std::env::var("GLM53_MLA_BMM_C12").as_deref()==Ok("1")}
thread_local!{
    // kv_b ptr -> (owner, coded wk^T [heads*512,256], coded wv^T [heads*256,512]).
    static LATENT_C12:std::cell::RefCell<std::collections::HashMap<usize,(Tensor,[Tensor;6],[Tensor;6])>>=std::cell::RefCell::new(std::collections::HashMap::new());
}
fn c12_bmm(x:&Tensor,c:&[Tensor;6],heads:i64,n:i64,k:i64)->Tensor {
    let m=x.size()[1];assert_eq!(x.size(),[heads,m,k]);assert_eq!(x.stride()[2],1);
    let y=Tensor::empty([heads,m,n],(Kind::Float,x.device()));
    extern "C"{fn rs_latent_c12_bmm(x:*const f32,xh:i64,xm:i64,m8:*const std::ffi::c_void,e4:*const std::ffi::c_void,eb:*const std::ffi::c_void,ptr:*const i32,col:*const i32,
        val:*const std::ffi::c_void,heads:i32,n:i32,k:i32,y:*mut f32,m:i32)->i32;}
    assert_eq!(unsafe{rs_latent_c12_bmm(x.data_ptr().cast(),x.stride()[0],x.stride()[1],c[0].data_ptr(),c[1].data_ptr(),c[2].data_ptr(),c[3].data_ptr().cast(),
        c[4].data_ptr().cast(),c[5].data_ptr(),heads as i32,n as i32,k as i32,y.data_ptr().cast(),m as i32)},0,"latent C12 bmm");y
}
/// x [heads,m,K] (any head/row strides, unit inner stride) times W^T per head -> [heads,m,N] FP32.
fn half_bmm(x:&Tensor,w:&Tensor)->Tensor {
    let (heads,m,k)=(x.size()[0],x.size()[1],x.size()[2]);let n=w.size()[1];
    assert_eq!(w.size(),[heads,n,k]);assert_eq!(x.stride()[2],1);assert!(w.is_contiguous());
    let y=Tensor::empty([heads,m,n],(Kind::Float,x.device()));
    extern "C"{fn rs_latent_half_bmm(x:*const f32,xh:i64,xm:i64,w:*const std::ffi::c_void,wh:i64,y:*mut f32,heads:i32,m:i32,n:i32,k:i32)->i32;}
    assert_eq!(unsafe{rs_latent_half_bmm(x.data_ptr().cast(),x.stride()[0],x.stride()[1],w.data_ptr(),n*k,y.data_ptr().cast(),heads as i32,m as i32,n as i32,k as i32)},0,"latent half bmm");y
}
/// Verifier-tree absorbed/output products; falls back to the FP32 bmm when not eligible.
fn absorb(w:&MlaWeights,q:&Tensor,wk:&Tensor)->Tensor {
    // The Half/C12 path takes at most 8 rows (dim 1): split longer chains (per-row math is independent of the split).
    let wide=bmm_wide() && q.size()[1]<=32 && q.device().is_cuda() && q.kind()==Kind::Float && q.stride()[2]==1 && q.stride()[0]%4==0 && q.stride()[1]%4==0
        && LATENT_C12.with(|c|c.borrow().contains_key(&(w.kv_b.data_ptr() as usize)));
    if half_bmm_enabled() && q.size()[1]>8 && q.size()[1]<=64 && !wide {
        let t=q.size()[1];return Tensor::cat(&(0..t).step_by(8).map(|r|absorb(w,&q.narrow(1,r,(t-r).min(8)),wk)).collect::<Vec<_>>(),1);
    }
    let key=w.kv_b.data_ptr() as usize;
    let half=half_bmm_enabled() && q.device().is_cuda() && q.kind()==Kind::Float && ((1..=8).contains(&q.size()[1])||wide) && q.stride()[2]==1
        && q.stride()[0]%4==0 && q.stride()[1]%4==0;
    if half && bmm_c12_enabled() { if let Some(r)=LATENT_C12.with(|c|c.borrow().get(&key).map(|e|c12_bmm(q,&e.1,q.size()[0],512,256))) {
        if wide && q.size()[1]>8 && crate::kda::wide_check_on() {let t=q.size()[1];
            crate::kda::wide_check("mla-absorb",&r,&Tensor::cat(&(0..t).step_by(8).map(|r0|absorb(w,&q.narrow(1,r0,(t-r0).min(8)),wk)).collect::<Vec<_>>(),1));}
        return r;} }
    if half { if let Some(wt)=LATENT_HALF.with(|c|c.borrow().get(&key).map(|e|e.1.shallow_clone())) {return half_bmm(q,&wt);} }
    q.bmm(wk)
}
fn expand(w:&MlaWeights,a:&Tensor,wv:&Tensor)->Tensor {
    // Proposal 3: the Half path takes at most 8 rows (dim 1); split larger verify batches like absorb does.
    if crate::forward::verify_invariant() && a.size()[1]>8 && a.size()[1]<=64 {
        let t=a.size()[1];return Tensor::cat(&crate::forward::invariant_pieces(t).into_iter().map(|(r,n)|expand(w,&a.narrow(1,r,n),wv)).collect::<Vec<_>>(),1);
    }
    let wide=bmm_wide() && a.size()[1]<=32 && a.device().is_cuda() && a.kind()==Kind::Float && LATENT_C12.with(|c|c.borrow().contains_key(&(w.kv_b.data_ptr() as usize)));
    if half_bmm_enabled() && a.size()[1]>8 && a.size()[1]<=64 && !wide {
        let t=a.size()[1];return Tensor::cat(&(0..t).step_by(8).map(|r|expand(w,&a.narrow(1,r,(t-r).min(8)),wv)).collect::<Vec<_>>(),1);
    }
    let key=w.kv_b.data_ptr() as usize;
    let a=if half_bmm_enabled(){a.contiguous()}else{a.shallow_clone()};
    let half=half_bmm_enabled() && a.device().is_cuda() && a.kind()==Kind::Float && ((1..=8).contains(&a.size()[1])||wide);
    if half && bmm_c12_enabled() { if let Some(r)=LATENT_C12.with(|c|c.borrow().get(&key).map(|e|c12_bmm(&a,&e.2,a.size()[0],256,512))) {
        if wide && a.size()[1]>8 && crate::kda::wide_check_on() {let t=a.size()[1];
            crate::kda::wide_check("mla-expand",&r,&Tensor::cat(&(0..t).step_by(8).map(|r0|expand(w,&a.narrow(1,r0,(t-r0).min(8)),wv)).collect::<Vec<_>>(),1));}
        return r;} }
    if half { if let Some(wt)=LATENT_HALF.with(|c|c.borrow().get(&key).map(|e|e.2.shallow_clone())) {return half_bmm(&a,&wt);} }
    a.bmm(wv)
}
pub(crate) fn chain_shared_enabled()->bool {std::env::var("GLM53_MLA_CHAIN_SHARED").as_deref()==Ok("1")}
pub(crate) fn node_fused_enabled()->bool {std::env::var("GLM53_DSA_NODE_FUSED").as_deref()==Ok("1")}
/// One launch: child tails = parent tails with slot pos%4 replaced, provisional pool at pos/4,
/// latent row at pos, len=pos+1. Parent is read-only; child latent/pools hold active copies.
fn node_append(src:&State,st:&mut State,w:&dsa::Weights,k:&Tensor,gate:&Tensor,row:&Tensor) {
    let dim=st.index.tail_k.size()[1];let lat=if is_fp8(&st.latent){0}else{st.latent.size()[1]};
    for t in [k,gate,row,&w.ape,&src.index.tail_k,&src.index.tail_gate,&st.index.pools,&st.latent] {assert!(t.is_contiguous());}
    assert_eq!(k.numel() as i64,dim);assert_eq!(row.numel() as i64,512);assert_eq!(row.kind(),Kind::Half);assert_eq!(w.ape.size(),[4,dim]);
    extern "C"{fn rs_dsa_node_append_row(parent_len:*const i64,ptk:*const f32,ptg:*const f32,ctk:*mut f32,ctg:*mut f32,cpools:*mut f32,ape:*const f32,
        krow:*const f32,grow:*const f32,clatent:*mut std::ffi::c_void,lrow:*const std::ffi::c_void,clen:*mut i64,dim:i32,lat:i32,crow:*mut f32)->i32;}
    let f=|t:&Tensor|t.data_ptr() as *const f32;let g=|t:&Tensor|t.data_ptr() as *mut f32;
    let crow=st.pool_row.as_ref().map_or(std::ptr::null_mut(),|r|{assert!(r.is_contiguous());g(r)});
    assert_eq!(unsafe{rs_dsa_node_append_row(src.len.data_ptr().cast(),f(&src.index.tail_k),f(&src.index.tail_gate),g(&st.index.tail_k),g(&st.index.tail_gate),
        g(&st.index.pools),f(&w.ape),f(k),f(gate),st.latent.data_ptr(),row.data_ptr(),st.len.data_ptr().cast(),dim as i32,lat as i32,crow)},0,"DSA node append");
    if is_fp8(&st.latent) {store_rows(&st.latent,&row.view([1,-1]),&src.len);}
}
impl State {
    /// Latent row width and dtype of a store's latent tensor (KV pool layout).
    pub(crate) fn latent_layout()->(i64,Kind) {if kv_fp8_enabled(){(KV8_ROW,Kind::Uint8)}else{(512,Kind::Half)}}
    /// `new` over KV-pool views (zero-filled latent [capacity, width] and DSA pools [capacity/4, dim]).
    pub(crate) fn from_pool(w:&MlaWeights,capacity:i64,latent:Tensor,pools:Tensor)->Self {
        ensure_latent_half(w);
        let dev=w.kv_b.device();let (wk,wv)=w.latent_projections();
        let (width,kind)=Self::latent_layout();
        assert_eq!(latent.size(),[capacity,width],"pooled latent rows");assert_eq!(latent.kind(),kind);
        let dim=w.indexer.as_ref().expect("latent MLA needs indexer weights").k.size()[0];
        Self{latent,index:dsa::State::with_pools(capacity,dim,dev,pools),len:Tensor::zeros([1],(Kind::Int64,dev)),capacity,pool_row:None,wk,wv}
    }
    pub fn new(w:&MlaWeights,capacity:i64)->Self {
        ensure_latent_half(w);
        let dev=w.kv_b.device();let (wk,wv)=w.latent_projections();
        Self{latent:if kv_fp8_enabled(){Tensor::zeros([capacity,KV8_ROW],(Kind::Uint8,dev))}else{Tensor::zeros([capacity,512],(Kind::Half,dev))},
            index:dsa::State::new(capacity,w.indexer.as_ref().expect("latent MLA needs indexer weights").k.size()[0],dev),
            len:Tensor::zeros([1],(Kind::Int64,dev)),capacity,pool_row:None,
            wk,wv}
    }
    pub fn snapshot(&self)->Self {
        if active_copy_enabled(){
            let latent=Tensor::empty_like(&self.latent);active_copy(&latent,&self.latent,&self.len,1);
            let st=Self{latent,index:self.index.snapshot_active(&self.len),len:self.len.copy(),capacity:self.capacity,pool_row:None,
                wk:self.wk.shallow_clone(),wv:self.wv.shallow_clone()};
            self.patch_pool_row(&st);return st;
        }
        let st=Self{latent:self.latent.copy(),index:self.index.snapshot(),len:self.len.copy(),capacity:self.capacity,pool_row:None,
            wk:self.wk.shallow_clone(),wv:self.wv.shallow_clone()};
        self.patch_pool_row(&st);st
    }
    pub(crate) fn alias(&self)->Self {
        Self{latent:self.latent.shallow_clone(),index:self.index.alias(),len:self.len.shallow_clone(),capacity:self.capacity,
            pool_row:self.pool_row.as_ref().map(Tensor::shallow_clone),wk:self.wk.shallow_clone(),wv:self.wv.shallow_clone()}
    }
    /// W07: bring self (== the verifier base, length L) up to `node` (length L+new):
    /// copy only the new latent rows, the touched pool rows, both tails and len.
    pub(crate) fn commit_from(&self,node:&Self,new_rows:i64) {
        assert_eq!(self.capacity,node.capacity);assert!((1..=16).contains(&new_rows));
        if node.latent.data_ptr()==self.latent.data_ptr() {
            // Chain-shared node: latent rows and completed pool rows are already in place;
            // only the node's last (possibly overwritten) pool row, tails and len remain.
            assert_eq!(node.index.pools.data_ptr(),self.index.pools.data_ptr());
            node.patch_pool_row(self);
            self.index.tail_k.shallow_clone().copy_(&node.index.tail_k);self.index.tail_gate.shallow_clone().copy_(&node.index.tail_gate);
            self.len.shallow_clone().copy_(&node.len);
            return;
        }
        extern "C"{fn rs_range_cache_copy(src:*const std::ffi::c_void,dst:*mut std::ffi::c_void,lo:*const i64,hi:*const i64,capacity:i32,row_vectors:i32,divisor:i32,max_rows:i32)->i32;}
        let rv=|t:&Tensor|{let bytes=match t.kind(){Kind::Half=>2,Kind::Float=>4,Kind::Uint8=>1,_=>panic!()};(t.size()[1]*bytes/16) as i32};
        assert_eq!(unsafe{rs_range_cache_copy(node.latent.data_ptr(),self.latent.data_ptr(),self.len.data_ptr().cast(),node.len.data_ptr().cast(),
            self.latent.size()[0] as i32,rv(&self.latent),1,new_rows as i32)},0);
        assert_eq!(unsafe{rs_range_cache_copy(node.index.pools.data_ptr(),self.index.pools.data_ptr(),self.len.data_ptr().cast(),node.len.data_ptr().cast(),
            self.index.pools.size()[0] as i32,rv(&self.index.pools),4,(new_rows/4+2) as i32)},0);
        self.index.tail_k.shallow_clone().copy_(&node.index.tail_k);self.index.tail_gate.shallow_clone().copy_(&node.index.tail_gate);
        self.len.shallow_clone().copy_(&node.len);
    }
    /// D8: the chain-shared commit as one batched-kernel entry (None: not chain-shared / unsupported layout).
    pub(crate) fn commit_entry(&self,node:&Self,new_rows:i64)->Option<MlaCommitEntry> {
        assert_eq!(self.capacity,node.capacity);assert!((1..=16).contains(&new_rows));
        if node.latent.data_ptr()!=self.latent.data_ptr() {return None;}
        assert_eq!(node.index.pools.data_ptr(),self.index.pools.data_ptr());
        let ok=|a:&Tensor,b:&Tensor|a.kind()==Kind::Float&&b.kind()==Kind::Float&&a.is_contiguous()&&b.is_contiguous()&&a.size()==b.size();
        if !ok(&self.index.tail_k,&node.index.tail_k)||!ok(&self.index.tail_gate,&node.index.tail_gate)
            ||self.len.kind()!=Kind::Int64||node.len.kind()!=Kind::Int64||self.index.pools.kind()!=Kind::Float||!self.index.pools.is_contiguous() {return None;}
        let (row_ptr,row_floats,has)=match &node.pool_row {
            Some(r)=>{assert!(r.is_contiguous()&&r.kind()==Kind::Float&&r.numel() as i64==self.index.pools.size()[1]);(r.data_ptr() as *const f32,r.numel() as i32,1)}
            None=>(std::ptr::null(),0,0)};
        Some(MlaCommitEntry{node_len:node.len.data_ptr() as *const i64,dst_pools:self.index.pools.data_ptr() as *mut f32,pool_row:row_ptr,row_floats,
            src_tk:node.index.tail_k.data_ptr() as *const f32,dst_tk:self.index.tail_k.data_ptr() as *mut f32,tk_floats:self.index.tail_k.numel() as i32,
            src_tg:node.index.tail_gate.data_ptr() as *const f32,dst_tg:self.index.tail_gate.data_ptr() as *mut f32,tg_floats:self.index.tail_gate.numel() as i32,
            dst_len:self.len.data_ptr() as *mut i64,has_row:has})
    }
    /// Active latent/pools copies only; tails and len are produced by node_append.
    pub(crate) fn snapshot_for_append(&self)->Self {
        let latent=Tensor::empty_like(&self.latent);active_copy(&latent,&self.latent,&self.len,1);
        let st=Self{latent,index:self.index.snapshot_pools_for_append(&self.len),len:Tensor::empty_like(&self.len),capacity:self.capacity,
            pool_row:None,wk:self.wk.shallow_clone(),wv:self.wv.shallow_clone()};
        self.patch_pool_row(&st);st
    }
    pub fn restore(&mut self,src:&Self) {
        assert_eq!(self.capacity,src.capacity);
        if active_copy_enabled(){active_copy(&self.latent,&src.latent,&src.len,1);self.index.restore_active(&src.index,&src.len);}
        else{self.latent.copy_(&src.latent);self.index.restore(&src.index);}
        self.len.copy_(&src.len);
        src.patch_pool_row(self);
    }
    /// Write this chain-shared node's own copy of pool row (len-1)/4 into `dst` (no-op otherwise).
    pub(crate) fn patch_pool_row(&self,dst:&Self) {
        let Some(row)=&self.pool_row else {return};
        let idx=(&self.len-1).floor_divide_scalar(4).clamp_min(0);
        let _=dst.index.pools.shallow_clone().index_copy_(0,&idx,row);
    }
    /// Chain child sharing this state's latent/pools storage (GLM53_MLA_CHAIN_SHARED).
    pub(crate) fn chain_child(&self)->Self {
        Self{latent:self.latent.shallow_clone(),index:self.index.chain_child(),len:Tensor::empty_like(&self.len),capacity:self.capacity,
            pool_row:Some(Tensor::empty([1,self.index.pools.size()[1]],(Kind::Float,self.latent.device()))),
            wk:self.wk.shallow_clone(),wv:self.wv.shallow_clone()}
    }
    pub fn max_diff(&self,other:&Self)->f64 {
        if self.capacity!=other.capacity{return f64::INFINITY;}
        let len=self.len.int64_value(&[0]);
        if len!=other.len.int64_value(&[0]) || len<0 || len>self.capacity{return f64::INFINITY;}
        // Unused suffix is not part of the state. Restoring a shorter branch
        // can leave stale bytes there; consumers must never make them visible.
        let pools=|st:&Self|{let p=st.index.pools.narrow(0,0,(len+3)/4).copy();
            if let (Some(row),true)=(&st.pool_row,len>0){let _=p.narrow(0,(len-1)/4,1).copy_(row);}p};
        let a=[self.latent.narrow(0,0,len),pools(self),self.index.tail_k.shallow_clone(),self.index.tail_gate.shallow_clone()];
        let b=[other.latent.narrow(0,0,len),pools(other),other.index.tail_k.shallow_clone(),other.index.tail_gate.shallow_clone()];
        a.into_iter().zip(b).filter(|(a,_)|a.numel()>0)
            .map(|(a,b)|{let d=f64::try_from((a-b).abs().max()).unwrap();if d.is_finite(){d}else{f64::INFINITY}})
            .fold(0.,f64::max)
    }
    pub fn ensure_room(&self,count:i64) {
        let len=self.len.int64_value(&[0]);
        assert!(count>=0 && len+count<=self.capacity,"latent capacity exceeded: {len}+{count}>{}",self.capacity);
    }
}

/// Caller checks capacity outside graph capture/replay (once per request/block).
pub fn step(w:&MlaWeights,x:&Tensor,st:&mut State)->Tensor {
    step_record(w,x,st).0
}

/// Chunked prefill: batch weight reads, preserve per-token DSA causality.
pub(crate) fn prefill_gather_attention(q:&Tensor,latent:&Tensor,ids:&Tensor)->Tensor {
    let mut output=Vec::new();let tokens=q.size()[1];let slots=ids.size()[1];
    // One key matrix per query, shared by all local heads. Bound temporary
    // expansion to 32 queries; never materialize T x heads x slots x latent.
    for start in (0..tokens).step_by(32) {
        let n=(tokens-start).min(32);let selected=ids.narrow(0,start,n);
        let c=latent.index_select(0,&selected.clamp_min(0).reshape([-1]))
            .view([n,slots,512]).to_kind(Kind::Float);
        let queries=q.narrow(1,start,n).transpose(0,1).contiguous();
        let scores=queries.bmm(&c.transpose(1,2))/16.;
        let probabilities=scores.masked_fill(&selected.lt(0).unsqueeze(1),f64::NEG_INFINITY).softmax(-1,Kind::Float);
        output.push(probabilities.bmm(&c));
    }
    Tensor::cat(&output,0).transpose(0,1).contiguous()
}

pub fn chunk(w:&MlaWeights,x:&Tensor,st:&mut State)->Tensor {
    let t=x.size()[0];assert!(t>0);st.ensure_room(t);
    if std::env::var("GLM53_PREFILL_BATCH").as_deref()!=Ok("1") {
        let rows:Vec<_>=(0..t).map(|i|step(w,&x.narrow(0,i,1),st)).collect();
        return Tensor::cat(&rows,0);
    }
    let heads=w.q_b.size()[0]/256;
    let (qa_out,kva_out)=qa_kva(w,x);let cq=norm(qa_out,&w.q_a_ln);
    let strided=prefill_strided() && st.wk.is_contiguous() && st.wv.is_contiguous() && st.wk.kind()==Kind::Float && st.wv.kind()==Kind::Float;
    let qb=mm16(&cq,&w.q_b);let q=qb.view([t,heads,256]).transpose(0,1);
    let latent=norm(kva_out,&w.kv_a_ln).to_kind(Kind::Half);
    let index=w.indexer.as_ref().unwrap();let projected=index.project(x,&cq);
    // GLM53_MLA_PREFILL_STRIDED=1 (L1): q read in place ([t, heads*256], batch stride 256) by a strided TF32 GEMM.
    let absorbed=if strided && qb.is_contiguous() {let a=Tensor::empty([heads,t,512],(Kind::Float,x.device()));
        bmm_tf32(&qb,heads*256,256,&st.wk,512,256*512,&a,512,t*512,t,512,256,heads);a} else {q.bmm(&st.wk)};
    let min_rows=std::env::var("GLM53_PREFILL_MIN_ROWS").ok().map(|s|s.parse::<i64>().unwrap()).unwrap_or(1);
    if x.size()[0]>=min_rows && std::env::var("GLM53_MLA_PREFILL_BATCHED").as_deref()==Ok("1") {
        let start=st.len.int64_value(&[0]);
        store_rows_at(&st.latent,&latent,start);
        let selected=st.index.append_chunk(index,&projected,start);
        // P1: tensor-core form of the shared-latent attention (L1); overrides the mode.
        let tc=std::env::var("GLM53_MLA_PREFILL_TC").as_deref()==Ok("1");
        let attended=if tc {shared_attention(&absorbed,&st.latent,&selected,8)} else {match std::env::var("GLM53_MLA_PREFILL_DENSE").as_deref() {
            Ok("1")=>{assert_half_latent(&st.latent,"dense prefill");prefill_dense_attention(&absorbed,&st.latent,&selected,start+t)},
            Ok("2")=>{assert_half_latent(&st.latent,"gather prefill");prefill_gather_attention(&absorbed,&st.latent,&selected)},
            Ok(mode @ ("3"|"4"|"5"|"6"|"7"|"8"))=>shared_attention(&absorbed,&st.latent,&selected,mode.parse().unwrap()),
            _=>sparse_attention(&absorbed,&st.latent,&selected),
        }};
        st.len.copy_(&(&st.len+t));
        // P9 (GLM53_PREFILL_MLA_HALF_OUT=1): one strided copy writes the Half row-major hidden that the o_proj
        // GEMM consumes (the same RN conversion), instead of an FP32 reshape copy followed by a Half copy.
        if t>128 && std::env::var("GLM53_PREFILL_MLA_HALF_OUT").as_deref()==Ok("1") && w.wo.kind()==Kind::Half {
            let y=attended.bmm(&st.wv);
            let hidden=Tensor::empty([t,heads,256],(Kind::Half,y.device()));let _=hidden.shallow_clone().copy_(&y.transpose(0,1));
            return row_mm16(&hidden.view([t,heads*256]),&w.wo);
        }
        if strided && attended.is_contiguous() && attended.size()==[heads,t,512] && attended.kind()==Kind::Float {
            // the expand written straight into o_proj's [t, heads*256] layout (no transpose/reshape copy)
            let hidden=Tensor::empty([t,heads*256],(Kind::Float,x.device()));
            bmm_tf32(&attended,512,t*512,&st.wv,256,512*256,&hidden,heads*256,256,t,256,512,heads);
            return row_mm16(&hidden,&w.wo);
        }
        let hidden=attended.bmm(&st.wv).transpose(0,1).reshape([t,heads*256]);
        return row_mm16(&hidden,&w.wo);
    }
    let mut rows=Vec::new();
    for i in 0..t {
        store_rows(&st.latent,&latent.narrow(0,i,1),&st.len);
        let selected=st.index.append_projected(index,&projected,i,&st.len);
        if fused_attention() {
            rows.push(sparse_attention(&absorbed.narrow(1,i,1),&st.latent,&selected));
            st.len.copy_(&(&st.len+1));continue;
        }
        let visible=selected.ge(0);
        let c={assert_half_latent(&st.latent,"unfused MLA attention");st.latent.index_select(0,&selected.clamp_min(0)).to_kind(Kind::Float)};
        let scores=batch_scores(&absorbed,i,&c,false);
        let probs=scores.masked_fill(&visible.logical_not().view([1,1,-1]),f64::NEG_INFINITY).softmax(-1,Kind::Float);
        rows.push(probs.matmul(&c));
        st.len.copy_(&(&st.len+1));
    }
    let hidden=Tensor::cat(&rows,1).bmm(&st.wv).transpose(0,1).reshape([t,heads*256]);
    row_mm16(&hidden,&w.wo)
}

/// Every branch inherits only its ancestors' latent/indexer state.
pub fn tree(w:&MlaWeights,x:&Tensor,base:&State,parents:&[Option<usize>])->(Tensor,Vec<State>) {
    tree_impl(w,x,base,parents,true)
}

pub(crate) fn tree_impl(w:&MlaWeights,x:&Tensor,base:&State,parents:&[Option<usize>],check_room:bool)->(Tensor,Vec<State>) {
    tree_impl_selected(w,x,base,parents,dsa::TreeSelection::Ranked,check_room)
}

pub(crate) fn index_side_enabled()->bool {std::env::var("GLM53_MLA_INDEX_SIDE").as_deref()==Ok("1")}
pub(crate) fn tree_impl_selected(w:&MlaWeights,x:&Tensor,base:&State,parents:&[Option<usize>],selection:dsa::TreeSelection,check_room:bool)->(Tensor,Vec<State>) {
    let t=x.size()[0];assert_eq!(t as usize,parents.len());let heads=w.q_b.size()[0]/256;
    let (qa_out,kva_out)=qa_kva(w,x);let cq=norm(qa_out,&w.q_a_ln);
    let index=w.indexer.as_ref().unwrap();
    // GLM53_MLA_INDEX_SIDE=1 (L0, schedule only): the DSA indexer projections (x, cq only; C12/Q8-coded, no cuBLAS on the
    // pool side stream) run on a side stream while the main stream does q_b and the absorb; joined before tree_core.
    let index_side=selection==dsa::TreeSelection::Ranked && index_side_enabled() && crate::c12::index_enabled() && t<=16;
    extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;fn rs_stream_join(n:i32)->i32;}
    let side_projected=index_side.then(||{assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);
        let p=index.project(x,&cq);assert_eq!(unsafe{rs_stream_set(-1)},0);p});
    let qb=mm16(&cq,&w.q_b);
    let absorbed=absorb(w,&qb.view([t,heads,256]).transpose(0,1),&base.wk);
    let latent=norm(kva_out,&w.kv_a_ln).to_kind(Kind::Half);
    if side_projected.is_some() {assert_eq!(unsafe{rs_stream_join(1)},0);}
    let projected=side_projected.or_else(||(selection==dsa::TreeSelection::Ranked).then(||index.project(x,&cq)));
    let keys=(selection==dsa::TreeSelection::AllVisible).then(||index.project_keys(x));
    mla_taps(&cq,&qb,&absorbed,&latent,projected.as_ref());
    let (rows,states)=tree_core(w,x,base,parents,selection,check_room,&absorbed,&latent,projected.as_ref(),keys.as_ref());
    let rows_cat=if rows.len()==1 {rows[0].shallow_clone()} else {Tensor::cat(&rows,1)};crate::forward::probe_inner("mla_attn",&rows_cat.transpose(0,1));
    let hidden=expand(w,&rows_cat,&base.wv).transpose(0,1).reshape([t,heads*256]);
    crate::forward::probe_inner("mla_hidden",&hidden);
    (row_mm16(&hidden,&w.wo),states)
}
/// Multi-sequence verifier (serving batch): Q/KV/indexer projections, the output expansion and WO (with
/// its TP sum) run once over the rows of all sequences; latent/DSA appends, selection and sparse
/// attention run per sequence on its own base state (views of the batched projections).
pub(crate) fn tree_multi_selected(w:&MlaWeights,x:&Tensor,bases:&[&State],segs:&[(usize,usize)],parents_all:&[Vec<Option<usize>>],
    selections:&[dsa::TreeSelection],check_room:bool)->(Tensor,Vec<Vec<State>>) {
    let t=x.size()[0];let heads=w.q_b.size()[0]/256;
    assert!(bases.iter().all(|b|b.wk.data_ptr()==bases[0].wk.data_ptr()&&b.wv.data_ptr()==bases[0].wv.data_ptr()),"MLA absorbed weights differ between sequences");
    let (qa_out,kva_out)=qa_kva(w,x);let cq=norm(qa_out,&w.q_a_ln);
    let index=w.indexer.as_ref().unwrap();
    // GLM53_MLA_INDEX_SIDE=1 (multi): indexer projections on the side stream (C12-coded, <= 32 rows), as in tree_impl_selected.
    let index_side=selections.contains(&dsa::TreeSelection::Ranked) && index_side_enabled() && crate::c12::index_enabled() && t<=32;
    extern "C"{fn rs_stream_fork(n:i32)->i32;fn rs_stream_set(i:i32)->i32;fn rs_stream_join(n:i32)->i32;}
    let side_projected=index_side.then(||{assert_eq!(unsafe{rs_stream_fork(1)},0);assert_eq!(unsafe{rs_stream_set(0)},0);
        let p=index.project(x,&cq);assert_eq!(unsafe{rs_stream_set(-1)},0);p});
    let qb=mm16(&cq,&w.q_b);let q=qb.view([t,heads,256]).transpose(0,1);
    // absorb's Half path takes at most 8 rows; per-row math is independent of the split.
    let absorbed=if t<=8 || (bmm_wide() && t<=32) {absorb(w,&q,&bases[0].wk)} else {
        Tensor::cat(&(0..t).step_by(8).map(|r|absorb(w,&q.narrow(1,r,(t-r).min(8)),&bases[0].wk)).collect::<Vec<_>>(),1)};
    let latent=norm(kva_out,&w.kv_a_ln).to_kind(Kind::Half);
    if side_projected.is_some() {assert_eq!(unsafe{rs_stream_join(1)},0);}
    let projected=side_projected.or_else(||selections.contains(&dsa::TreeSelection::Ranked).then(||index.project(x,&cq)));
    let keys=selections.contains(&dsa::TreeSelection::AllVisible).then(||index.project_keys(x));
    mla_taps(&cq,&qb,&absorbed,&latent,projected.as_ref());
    if let Some((rows_cat,all_states))=multi_launch_core(x,bases,segs,parents_all,selections,check_room,&absorbed,&latent,projected.as_ref(),index) {
        if crate::kda::wide_check_on() {
            // the per-sequence path again on fresh chain children (its appends and stores write the same values)
            let mut refr=Vec::new();
            for (g,&(first,len)) in segs.iter().enumerate() {
                let (f,n)=(first as i64,len as i64);let r=|a:&Tensor|a.narrow(0,f,n);
                let pg=projected.as_ref().map(|p|dsa::Projected{k:r(&p.k),gate:r(&p.gate),q:r(&p.q),mixing:r(&p.mixing)});
                let (rows,_)=tree_core(w,&r(x),bases[g],&parents_all[g],selections[g],check_room,&absorbed.narrow(1,f,n),&r(&latent),pg.as_ref(),None);
                refr.extend(rows);
            }
            crate::kda::wide_check("mla-multi",&rows_cat,&Tensor::cat(&refr,1));
        }
        crate::forward::probe_inner("mla_attn",&rows_cat.transpose(0,1));
        let hidden=expand(w,&rows_cat,&bases[0].wv).transpose(0,1).reshape([t,heads*256]);
        crate::forward::probe_inner("mla_hidden",&hidden);
        return (row_mm16(&hidden,&w.wo),all_states);
    }
    let mut all_rows=Vec::new();let mut all_states=Vec::new();
    for (g,&(first,len)) in segs.iter().enumerate() {
        let (f,n)=(first as i64,len as i64);let r=|a:&Tensor|a.narrow(0,f,n);
        let pg=(selections[g]==dsa::TreeSelection::Ranked).then(||{let p=projected.as_ref().unwrap();dsa::Projected{k:r(&p.k),gate:r(&p.gate),q:r(&p.q),mixing:r(&p.mixing)}});
        let kg=(selections[g]==dsa::TreeSelection::AllVisible).then(||{let p=keys.as_ref().unwrap();dsa::KeyProjected{k:r(&p.k),gate:r(&p.gate)}});
        let (rows,states)=tree_core(w,&r(x),bases[g],&parents_all[g],selections[g],check_room,&absorbed.narrow(1,f,n),&r(&latent),pg.as_ref(),kg.as_ref());
        all_rows.extend(rows);all_states.push(states);
    }
    let rows_cat=Tensor::cat(&all_rows,1);crate::forward::probe_inner("mla_attn",&rows_cat.transpose(0,1));
    let hidden=expand(w,&rows_cat,&bases[0].wv).transpose(0,1).reshape([t,heads*256]);
    crate::forward::probe_inner("mla_hidden",&hidden);
    (row_mm16(&hidden,&w.wo),all_states)
}
/// GLM53_MLA_MULTI_LAUNCH=1 (L0): tree_core's node-batched FP8 chain path for every sequence of a multi-sequence verify at
/// once: chain children per sequence as before, then one launch each for the node appends, the FP8 latent stores, the
/// DSA scores (into one stacked masked buffer), the position capture, the fast top-k, the token expand and the sparse
/// attention over the stacked queries (each sequence reading its own state). Every kernel keeps the per-sequence
/// arithmetic, so rows and states equal the per-sequence tree_core's. None (nothing launched) unless every sequence takes
/// that path.
#[allow(clippy::too_many_arguments)]
fn multi_launch_core(x:&Tensor,bases:&[&State],segs:&[(usize,usize)],parents_all:&[Vec<Option<usize>>],selections:&[dsa::TreeSelection],
    check_room:bool,absorbed:&Tensor,latent:&Tensor,projected:Option<&dsa::Projected>,index:&dsa::Weights)->Option<(Tensor,Vec<Vec<State>>)> {
    static ON:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
    if !*ON.get_or_init(||std::env::var("GLM53_MLA_MULTI_LAUNCH").as_deref()==Ok("1")) {return None;}
    let n=segs.len();let t=x.size()[0];
    let score_mode=std::env::var("GLM53_DSA_SCORE_FUSED").ok().and_then(|v|v.parse::<i32>().ok()).unwrap_or(0);
    if check_room || n<2 || n>8 || t>64 || !(node_fused_enabled()&&active_copy_enabled()&&chain_shared_enabled()&&node_batch_enabled()&&fused_attention())
        || std::env::var("GLM53_DSA_SCORE_MULTI").as_deref()!=Ok("1") || !(2..=4).contains(&score_mode)
        || !crate::dsa_position::enabled() || !crate::dsa_topk::fast_enabled() {return None;}
    let p=projected?;
    if p.q.size()[1..]!=[32,128] || !p.q.is_contiguous() || !p.mixing.is_contiguous() || p.q.kind()!=Kind::Float || p.mixing.kind()!=Kind::Float
        || !p.k.is_contiguous() || !p.gate.is_contiguous() || latent.kind()!=Kind::Half || !latent.is_contiguous() || latent.size()[1]!=512 {return None;}
    let capacity=bases[0].index.pools.size()[0];let dim=bases[0].index.tail_k.size()[1];
    if capacity<1024 || bases[0].index.pools.size()[1]!=128 || absorbed.stride()[1]!=512 || absorbed.stride()[2]!=1 {return None;}
    for (g,&(first,len)) in segs.iter().enumerate() {
        let b=bases[g];
        if selections[g]!=dsa::TreeSelection::Ranked || !(2..=8).contains(&len) || !is_fp8(&b.latent) || b.index.pools.size()[0]!=capacity
            || b.index.tail_k.size()[1]!=dim || parents_all[g].iter().enumerate().any(|(i,&q)|q!=i.checked_sub(1))
            || !crate::dsa_topk::eligible(selections[g],&x.narrow(0,first as i64,len as i64),&b.index.pools,&b.len) {return None;}
    }
    let dev=x.device();
    // chain children and their parents' lengths (tree_core's node-batched branch)
    let mut states:Vec<Vec<State>>=Vec::with_capacity(n);
    #[repr(C)] #[derive(Clone,Copy)] struct Ptrs{parent_len:*const i64,ptk:*const f32,ptg:*const f32,ctk:*mut f32,ctg:*mut f32,cpools:*mut f32,krow:*const f32,grow:*const f32,clen:*mut i64,crow:*mut f32}
    let null=Ptrs{parent_len:std::ptr::null(),ptk:std::ptr::null(),ptg:std::ptr::null(),ctk:std::ptr::null_mut(),ctg:std::ptr::null_mut(),cpools:std::ptr::null_mut(),
        krow:std::ptr::null(),grow:std::ptr::null(),clen:std::ptr::null_mut(),crow:std::ptr::null_mut()};
    let mut ptrs=vec![null;64];let mut nodes=[0i32;8];let mut pos_ptrs:Vec<*const i64>=Vec::with_capacity(t as usize);
    for (g,&(first,len)) in segs.iter().enumerate() {
        let b=bases[g];let st:Vec<State>=(0..len).map(|_|b.chain_child()).collect();
        for i in 0..len {
            let src=if i==0{b}else{&st[i-1]};let c=&st[i];let row=(first+i) as i64;
            for z in [&src.index.tail_k,&src.index.tail_gate,&c.index.pools] {assert!(z.is_contiguous());}
            pos_ptrs.push(src.len.data_ptr() as *const i64);
            ptrs[g*8+i]=Ptrs{parent_len:src.len.data_ptr() as *const i64,ptk:src.index.tail_k.data_ptr() as *const f32,ptg:src.index.tail_gate.data_ptr() as *const f32,
                ctk:c.index.tail_k.data_ptr() as *mut f32,ctg:c.index.tail_gate.data_ptr() as *mut f32,cpools:c.index.pools.data_ptr() as *mut f32,
                krow:p.k.get(row).data_ptr() as *const f32,grow:p.gate.get(row).data_ptr() as *const f32,clen:c.len.data_ptr() as *mut i64,
                crow:c.pool_row.as_ref().map_or(std::ptr::null_mut(),|r|{assert!(r.is_contiguous());r.data_ptr() as *mut f32})};
        }
        nodes[g]=len as i32;states.push(st);
    }
    assert!(index.ape.is_contiguous());
    extern "C"{fn rs_dsa_node_append_chain_multi(ptrs:*const std::ffi::c_void,nodes:*const i32,nseq:i32,ape:*const f32,dim:i32)->i32;
        fn rs_latent_fp8_store_multi(rows:*const std::ffi::c_void,table:*const std::ffi::c_void)->i32;
        fn rs_dsa_score_multi_seq(q:*const f32,mixing:*const f32,table:*const std::ffi::c_void,out:*mut f32,capacity:i32,mode:i32)->i32;
        fn rs_dsa_capture_rows64(pos:*const *const i64,captured:*mut i64,rows:i32)->i32;
        fn rs_dsa_topk_rows(masked:*const f32,pos:*const i64,selected:*mut i64,pools:i32,k:i32,rows:i32)->i32;
        fn rs_dsa_index_expand_rows(selected:*const i64,pos:*const i64,out:*mut i64,k:i32,rows:i32)->i32;
        fn rs_latent_attention_fp8_multi(q:*const f32,table:*const std::ffi::c_void,selected:*const i64,out:*mut f32,scratch:*mut f32,heads:i32,queries:i32,slots:i32,stride:i32,splits:i32)->i32;}
    assert_eq!(unsafe{rs_dsa_node_append_chain_multi(ptrs.as_ptr().cast(),nodes.as_ptr(),n as i32,index.ape.data_ptr().cast(),dim as i32)},0,"DSA chain node append (multi)");
    #[repr(C)] struct Store{latent:[*mut u8;8],pos:[*const i64;8],first:[i32;8],n:[i32;8],nseq:i32}
    let mut sto=Store{latent:[std::ptr::null_mut();8],pos:[std::ptr::null();8],first:[0;8],n:[0;8],nseq:n as i32};
    #[repr(C)] struct Score{pools:[*const f32;8],p:[[*const i64;8];8],t:[i32;8],first:[i32;8],n:i32}
    let mut sc=Score{pools:[std::ptr::null();8],p:[[std::ptr::null();8];8],t:[0;8],first:[0;8],n:n as i32};
    #[repr(C)] struct Lat{latent:[*const u8;8],first:[i32;8],n:i32}
    let mut lt=Lat{latent:[std::ptr::null();8],first:[0;8],n:n as i32};
    for (g,&(first,len)) in segs.iter().enumerate() {
        let b=bases[g];sto.latent[g]=b.latent.data_ptr() as *mut u8;sto.pos[g]=b.len.data_ptr() as *const i64;sto.first[g]=first as i32;sto.n[g]=len as i32;
        sc.pools[g]=b.index.pools.data_ptr() as *const f32;sc.t[g]=len as i32;sc.first[g]=first as i32;
        for i in 0..len {sc.p[g][i]=pos_ptrs[first+i];}
        assert!(states[g].iter().all(|s|s.latent.data_ptr()==b.latent.data_ptr()));
        lt.latent[g]=b.latent.data_ptr() as *const u8;lt.first[g]=first as i32;
    }
    assert_eq!(unsafe{rs_latent_fp8_store_multi(latent.data_ptr(),(&sto as *const Store).cast())},0,"FP8 latent store (multi)");
    let masked=Tensor::empty([t,capacity],(Kind::Float,dev));
    assert_eq!(unsafe{rs_dsa_score_multi_seq(p.q.data_ptr().cast(),p.mixing.data_ptr().cast(),(&sc as *const Score).cast(),masked.data_ptr().cast(),capacity as i32,score_mode)},0,"DSA score (multi)");
    let side=Tensor::empty([t],(Kind::Int64,dev));
    assert_eq!(unsafe{rs_dsa_capture_rows64(pos_ptrs.as_ptr(),side.data_ptr().cast(),t as i32)},0,"DSA position capture (multi)");
    let k=512.min(capacity);let selected=Tensor::empty([t,k],(Kind::Int64,dev));
    assert_eq!(unsafe{rs_dsa_topk_rows(masked.data_ptr().cast(),side.data_ptr().cast(),selected.data_ptr().cast(),capacity as i32,k as i32,t as i32)},0,"DSA fast topk (multi)");
    let sel=Tensor::empty([t,4*k+3],(Kind::Int64,dev));
    assert_eq!(unsafe{rs_dsa_index_expand_rows(selected.data_ptr().cast(),side.data_ptr().cast(),sel.data_ptr().cast(),k as i32,t as i32)},0);
    let heads=absorbed.size()[0];let splits=8i64;
    let out=Tensor::empty([heads,t,512],(Kind::Float,dev));let scratch=Tensor::empty([splits*heads*t*514],(Kind::Float,dev));
    assert_eq!(unsafe{rs_latent_attention_fp8_multi(absorbed.data_ptr().cast(),(&lt as *const Lat).cast(),sel.data_ptr().cast(),out.data_ptr().cast(),scratch.data_ptr().cast(),
        heads as i32,t as i32,(4*k+3) as i32,absorbed.stride()[0] as i32,splits as i32)},0,"latent attention (multi)");
    Some((out,states))
}
#[allow(clippy::too_many_arguments)]
fn tree_core(w:&MlaWeights,x:&Tensor,base:&State,parents:&[Option<usize>],selection:dsa::TreeSelection,check_room:bool,
    absorbed:&Tensor,latent:&Tensor,projected:Option<&dsa::Projected>,keys:Option<&dsa::KeyProjected>)->(Vec<Tensor>,Vec<State>) {
    let t=x.size()[0];assert_eq!(t as usize,parents.len());
    let index=w.indexer.as_ref().unwrap();
    let mut depths=Vec::new();
    for (i,&p) in parents.iter().enumerate(){assert!(p.map_or(true,|p|p<i));depths.push(p.map_or(1,|p|depths[p]+1));}
    if check_room {
        let len=base.len.int64_value(&[0]);let depth=*depths.iter().max().unwrap();
        assert!(len>=0 && len+depth<=base.capacity,"latent capacity exceeded");
        if selection==dsa::TreeSelection::AllVisible {assert!(len+depth<=2051,"AllVisible tree crosses pool selection boundary");}
    }
    let mut states:Vec<State>=Vec::new();let mut rows=Vec::new();

    if crate::dsa_topk::eligible(selection,x,&base.index.pools,&base.len) {
        let mut batch=crate::dsa_topk::MaskedRows::new(t,base.index.pools.size()[0],x.device());
        // Finish each independent branch state before its descendants. The
        // output of attention is not part of a parent's indexer/latent state.
        let node_fused=node_fused_enabled()&&active_copy_enabled();
        // C1 (GLM53_DSA_SCORE_MULTI=1): a chain's score rows in one pass over the pools, after all appends.
        // Rows below node i's complete count are never rewritten by later chain nodes (they only complete
        // pools at or beyond it), so the deferred scores equal the per-node ones bitwise.
        let score_mode=std::env::var("GLM53_DSA_SCORE_FUSED").ok().and_then(|v|v.parse::<i32>().ok()).unwrap_or(0);
        let multi=node_fused && chain_shared_enabled() && std::env::var("GLM53_DSA_SCORE_MULTI").as_deref()==Ok("1")
            && (2..=4).contains(&score_mode) && t<=8 && parents.iter().enumerate().all(|(i,&p)|p==i.checked_sub(1))
            && projected.is_some_and(|p|p.q.size()[1..]==[32,128] && p.q.is_contiguous() && p.mixing.is_contiguous() && p.q.kind()==Kind::Float && p.mixing.kind()==Kind::Float)
            && base.index.pools.size()[1]==128;
        let mut deferred_pos:Vec<Tensor>=Vec::new();
        // GLM53_MLA_NODE_BATCH=1 (chain, FP8 latent): all node appends in one launch and the chain's latent rows in one
        // FP8 store (node i's row lands at base len + i, as its own store would put it).
        if node_fused && multi && node_batch_enabled() && is_fp8(&base.latent) && (2..=8).contains(&t) {
            let p=projected.unwrap();let dim=base.index.tail_k.size()[1];
            for _ in 0..t {states.push(base.chain_child());}
            #[repr(C)] struct Ptrs{parent_len:*const i64,ptk:*const f32,ptg:*const f32,ctk:*mut f32,ctg:*mut f32,cpools:*mut f32,krow:*const f32,grow:*const f32,clen:*mut i64,crow:*mut f32}
            let mut ptrs=Vec::with_capacity(t as usize);let (ks,gs):(Vec<Tensor>,Vec<Tensor>)=(0..t).map(|i|(p.k.get(i),p.gate.get(i))).unzip();
            for i in 0..t as usize {
                let src=if i==0{base}else{&states[i-1]};let st=&states[i];
                for x in [&ks[i],&gs[i],&src.index.tail_k,&src.index.tail_gate,&st.index.pools] {assert!(x.is_contiguous());}
                assert_eq!(ks[i].numel() as i64,dim);
                deferred_pos.push(batch.position_input(&src.len));
                ptrs.push(Ptrs{parent_len:src.len.data_ptr() as *const i64,ptk:src.index.tail_k.data_ptr() as *const f32,ptg:src.index.tail_gate.data_ptr() as *const f32,
                    ctk:st.index.tail_k.data_ptr() as *mut f32,ctg:st.index.tail_gate.data_ptr() as *mut f32,cpools:st.index.pools.data_ptr() as *mut f32,
                    krow:ks[i].data_ptr() as *const f32,grow:gs[i].data_ptr() as *const f32,clen:st.len.data_ptr() as *mut i64,
                    crow:st.pool_row.as_ref().map_or(std::ptr::null_mut(),|r|{assert!(r.is_contiguous());r.data_ptr() as *mut f32})});
            }
            assert!(index.ape.is_contiguous());
            extern "C"{fn rs_dsa_node_append_chain(ptrs:*const std::ffi::c_void,nodes:i32,ape:*const f32,dim:i32)->i32;}
            assert_eq!(unsafe{rs_dsa_node_append_chain(ptrs.as_ptr().cast(),t as i32,index.ape.data_ptr().cast(),dim as i32)},0,"DSA chain node append");
            store_rows(&base.latent,&latent.narrow(0,0,t),&base.len);
        } else {
        for (i,&parent) in parents.iter().enumerate() {
            if node_fused && multi {
                let src=parent.map_or(base,|p|&states[p]);
                let mut st=base.chain_child();
                let pos=batch.position_input(&src.len);
                let p=projected.unwrap();
                node_append(src,&mut st,index,&p.k.get(i as i64),&p.gate.get(i as i64),&latent.get(i as i64));
                deferred_pos.push(pos);states.push(st);continue;
            }
            if node_fused {
                let src=parent.map_or(base,|p|&states[p]);
                // A chain node (parent = previous node, first = base) can share the base storage.
                let chain=chain_shared_enabled() && parent==i.checked_sub(1);
                let mut st=if chain{base.chain_child()}else{src.snapshot_for_append()};
                let pos=batch.position_input(&src.len);
                let p=projected.unwrap();
                node_append(src,&mut st,index,&p.k.get(i as i64),&p.gate.get(i as i64),&latent.get(i as i64));
                let scores=st.index.score_row(index,p,i as i64,&pos);
                batch.push(&scores,pos);states.push(st);continue;
            }
            let mut st=parent.map_or(base,|p|&states[p]).snapshot();
            // Mask captures this view before the following len update; old
            // mode still returns an independent copy. RankedRows owns the
            // captured sidecar, never the mutable state length view.
            let pos=batch.position_input(&st.len);
            store_rows(&st.latent,&latent.narrow(0,i as i64,1),&pos);
            let scores=st.index.append_score(index,projected.unwrap(),i as i64,&pos);
            batch.push(&scores,pos); // writes directly into one shared masked row
            st.len.copy_(&(&st.len+1));states.push(st);
        }
        }
        if multi {
            let p=projected.unwrap();let capacity=base.index.pools.size()[0];
            let pre=if node_batch_enabled() && batch.width()==capacity && deferred_pos.len() as i64==t {batch.prescore_target(t)} else {None};
            let out=pre.as_ref().map(|m|m.shallow_clone()).unwrap_or_else(||Tensor::empty([t,capacity],(Kind::Float,x.device())));
            let ptrs:Vec<*const i64>=deferred_pos.iter().map(|q|{assert!(q.kind()==Kind::Int64&&q.numel()==1);q.data_ptr() as *const i64}).collect();
            extern "C"{fn rs_dsa_score_multi(q:*const f32,pools:*const f32,mixing:*const f32,pos:*const *const i64,t:i32,out:*mut f32,capacity:i32,mode:i32)->i32;}
            assert_eq!(unsafe{rs_dsa_score_multi(p.q.data_ptr().cast(),base.index.pools.data_ptr().cast(),p.mixing.data_ptr().cast(),ptrs.as_ptr(),t as i32,
                out.data_ptr().cast(),capacity as i32,score_mode)},0,"DSA multi-node score");
            if pre.is_some() {batch.push_prescored(&deferred_pos);}
            else if !(node_batch_enabled() && batch.push_all(&out,&deferred_pos)) {
                for (i,pos) in deferred_pos.into_iter().enumerate() {batch.push(&out.get(i as i64),pos);}
            }
        }
        let ranked=batch.finish();
        // GLM53_MLA_NODE_BATCH=1: every node of a chain reads the same shared latent storage (all appends above are
        // already enqueued), so one expand over all rows and one attention call with queries = t replace t of each.
        // Per (query, head, part) CTA and per-row expand are unchanged; the split count stays the per-node one (8).
        let batched=(node_batch_enabled() && fused_attention() && t>=2 && states.iter().all(|s|s.latent.data_ptr()==states[0].latent.data_ptr()))
            .then(||ranked.tokens_all()).flatten();
        if let Some(sel)=batched {
            rows.push(sparse_attention_splits(absorbed,&states[0].latent,&sel,Some(8)));
        } else {
        for i in 0..t {
            let st=&states[i as usize];let selected=ranked.tokens(i);
            if fused_attention() {
                rows.push(sparse_attention(&absorbed.narrow(1,i,1),&st.latent,&selected));
                continue;
            }
            let c={assert_half_latent(&st.latent,"unfused MLA attention");st.latent.index_select(0,&selected.clamp_min(0)).to_kind(Kind::Float)};
            let score=batch_scores(&absorbed,i,&c,true);
            let prob=score.masked_fill(&selected.lt(0).view([1,1,-1]),f64::NEG_INFINITY).softmax(-1,Kind::Float);
            rows.push(prob.matmul(&c));
        }
        }
    } else {
    for (i,&parent) in parents.iter().enumerate() {
        let mut st=parent.map_or(base,|p|&states[p]).snapshot();
        store_rows(&st.latent,&latent.narrow(0,i as i64,1),&st.len);
        if selection==dsa::TreeSelection::AllVisible && fused_attention() && std::env::var("GLM53_DSA_VISIBLE_DIRECT").as_deref()==Ok("1") {
            let p=keys.unwrap();
            st.index.append_keys(index,&p.k.narrow(0,i as i64,1),&p.gate.narrow(0,i as i64,1),&st.len);
            assert_half_latent(&st.latent,"DSA visible direct");rows.push(crate::dsa_direct::attention(&absorbed.narrow(1,i as i64,1),&st.latent,&st.len));
            st.len.copy_(&(&st.len+1));states.push(st);continue;
        }
        let selected=match selection {
            dsa::TreeSelection::Ranked=>st.index.append_projected(index,projected.unwrap(),i as i64,&st.len),
            dsa::TreeSelection::AllVisible=>st.index.append_all_visible(index,keys.unwrap(),i as i64,&st.len),
        };
        if fused_attention() {
            rows.push(sparse_attention(&absorbed.narrow(1,i as i64,1),&st.latent,&selected));
            st.len.copy_(&(&st.len+1));states.push(st);continue;
        }
        let c={assert_half_latent(&st.latent,"unfused MLA attention");st.latent.index_select(0,&selected.clamp_min(0)).to_kind(Kind::Float)};
        let score=batch_scores(&absorbed,i as i64,&c,true);
        let prob=score.masked_fill(&selected.lt(0).view([1,1,-1]),f64::NEG_INFINITY).softmax(-1,Kind::Float);
        rows.push(prob.matmul(&c));st.len.copy_(&(&st.len+1));states.push(st);
    }
    }
    (rows,states)
}

pub(crate) fn step_record(w:&MlaWeights,x:&Tensor,st:&mut State)->(Tensor,Tensor,Tensor) {
    assert_eq!(x.size()[0],1);
    let heads=w.q_b.size()[0]/256;
    let (qa_out,kva_out)=qa_kva(w,x);let cq=norm(qa_out,&w.q_a_ln);
    let q=mm16(&cq,&w.q_b).view([heads,1,256]);
    let latent=norm(kva_out,&w.kv_a_ln).to_kind(Kind::Half);
    store_rows(&st.latent,&latent,&st.len);
    let selected=st.index.append_select(w.indexer.as_ref().unwrap(),x,&cq,&st.len);
    let q_abs=q.bmm(&st.wk);
    let latent_out=if fused_attention(){sparse_attention(&q_abs,&st.latent,&selected)}else{
    let visible=selected.ge(0);
    let selected_latent={assert_half_latent(&st.latent,"unfused MLA attention");st.latent.index_select(0,&selected.clamp_min(0)).to_kind(Kind::Float)};
    let scores=q_abs.matmul(&selected_latent.transpose(0,1))/16.;
    let probs=scores.masked_fill(&visible.logical_not().view([1,1,-1]),f64::NEG_INFINITY).softmax(-1,Kind::Float);
    probs.matmul(&selected_latent)};
    let output=latent_out.bmm(&st.wv).reshape([1,heads*256]);
    st.len.copy_(&(&st.len+1));
    (row_mm16(&output,&w.wo),selected,cq)
}

/// Bounded cache identity checks use CPU tensors; no model mutation is needed.
fn weight_cache_identity_probe()->Vec<serde_json::Value> {
    use serde_json::json;
    let dev=tch::Device::Cpu;let empty=||Tensor::zeros([1],(Kind::Float,dev));
    let original=std::env::var("GLM53_MLA_WEIGHT_CACHE").ok();std::env::set_var("GLM53_MLA_WEIGHT_CACHE","1");
    let mut w=MlaWeights{indexer:None,q_a:empty(),q_b:Tensor::zeros([512,1],(Kind::Float,dev)),
        kv_a:empty(),kv_b:((Tensor::arange(1024*512,(Kind::Float,dev)).remainder(113)-56.)*0.001).view([1024,512]).to_kind(Kind::Half),
        wo:empty(),q_a_ln:empty(),kv_a_ln:empty(),latent_cache:std::cell::RefCell::new(None)};
    let mut records=Vec::new();
    let check=|w:&MlaWeights,name:&str,records:&mut Vec<serde_json::Value>| {
        let kv=w.kv_b.view([w.q_b.size()[0]/256,512,512]);
        let gold_k=kv.narrow(1,0,256).to_kind(Kind::Float).contiguous();
        let gold_v=kv.narrow(1,256,256).transpose(1,2).to_kind(Kind::Float).contiguous();
        let first=w.latent_projections();let second=w.latent_projections();
        assert!(first.0.equal(&gold_k)&&first.1.equal(&gold_v),"MLA cache identity/value {name}");
        assert_eq!(first.0.data_ptr(),second.0.data_ptr());assert_eq!(first.1.data_ptr(),second.1.data_ptr());
        assert!(first.0.is_contiguous()&&first.1.is_contiguous());
        records.push(json!({"case":name,"shape":w.kv_b.size(),"stride":w.kv_b.stride(),"dtype":format!("{:?}",w.kv_b.kind()),"exact":true,"shared_read_only":true}));first
    };
    let a=check(&w,"initial-half",&mut records);
    let source=w.kv_b.data_ptr();let _=w.kv_b.fill_(0.125);
    let b=check(&w,"same-pointer-inplace-version",&mut records);
    assert_eq!(source,w.kv_b.data_ptr());assert_ne!(a.0.data_ptr(),b.0.data_ptr());assert_ne!(a.1.data_ptr(),b.1.data_ptr());
    w.kv_b=w.kv_b.copy();let c=check(&w,"replaced-storage",&mut records);assert_ne!(b.0.data_ptr(),c.0.data_ptr());
    w.kv_b=w.kv_b.to_kind(Kind::Float);let d=check(&w,"changed-dtype",&mut records);assert_ne!(c.0.data_ptr(),d.0.data_ptr());
    let wide=(Tensor::arange(1024*1024,(Kind::Float,dev)).remainder(101)*0.001).view([1024,1024]).to_kind(Kind::Half);
    w.kv_b=wide.as_strided([1024,512],[512,1],None);let e=check(&w,"contiguous-view",&mut records);let source=w.kv_b.data_ptr();
    w.kv_b=wide.as_strided([1024,512],[1024,1],None);let f=check(&w,"same-pointer-changed-stride",&mut records);
    assert_eq!(source,w.kv_b.data_ptr());assert_ne!(e.0.data_ptr(),f.0.data_ptr());
    w.invalidate_latent_cache();assert!(w.latent_cache.borrow().is_none());
    w.kv_b=w.kv_b.narrow(0,0,512).copy();w.q_b=w.q_b.narrow(0,0,256).copy();
    let _=check(&w,"changed-local-head-shard",&mut records);
    std::env::set_var("GLM53_MLA_WEIGHT_CACHE","0");let _=w.latent_projections();assert!(w.latent_cache.borrow().is_none());
    records.push(json!({"case":"disabled-owner-release","exact":true}));
    match original{Some(v)=>std::env::set_var("GLM53_MLA_WEIGHT_CACHE",v),None=>std::env::remove_var("GLM53_MLA_WEIGHT_CACHE")};records
}

/// Real full-engine gate, usable with a resident TP2 engine. Two independently
/// initialized prefills share only immutable wk/wv, including through a tree
/// graph, while every writable state retains its own storage.
pub fn weight_cache_probe(eng:&mut crate::forward::Engine,out:&std::path::Path) {
    use serde_json::json;
    use crate::forward::LayerState;
    let _guard=tch::no_grad_guard();assert!(enabled());assert!(capacity()>=8);
    std::fs::create_dir_all(out).unwrap();let rank=crate::tp::world().rank;
    let path=out.join(format!("mla-weight-cache-rank{rank}.json"));
    std::fs::write(&path,r#"{"gate":false,"complete":false}"#).unwrap();
    let identity=weight_cache_identity_probe();let original=std::env::var("GLM53_MLA_WEIGHT_CACHE").ok();
    let ids=Tensor::from_slice(&[154822i64,154824,154826,13041]).to_device(eng.w.device);
    std::env::set_var("GLM53_MLA_WEIGHT_CACHE","0");
    let (gold_l,gold,gold_f)=eng.prefill_record_last(&ids,None,true);
    std::env::set_var("GLM53_MLA_WEIGHT_CACHE","1");
    let (a_l,a,a_f)=eng.prefill_record_last(&ids,None,true);
    let (b_l,mut b,b_f)=eng.prefill_record_last(&ids,None,true);
    assert!(a_l.equal(&gold_l)&&b_l.equal(&gold_l),"MLA cache prefill logits");
    let gold_f=Tensor::cat(&gold_f,1);
    assert!(Tensor::cat(&a_f,1).equal(&gold_f)&&Tensor::cat(&b_f,1).equal(&gold_f),"MLA cache prefill features");
    assert_eq!(crate::forward::states_max_diff(&gold,&a),0.);assert_eq!(crate::forward::states_max_diff(&a,&b),0.);
    let mut layers=Vec::new();let mut bytes=0usize;
    for (i,((gold,a),b)) in gold.0.iter().zip(&a.0).zip(&b.0).enumerate() {
        match (gold,a,b) {
            (LayerState::MlaLatent(g),LayerState::MlaLatent(a),LayerState::MlaLatent(b))=>{
                assert_eq!(a.wk.data_ptr(),b.wk.data_ptr());assert_eq!(a.wv.data_ptr(),b.wv.data_ptr());
                assert!(a.wk.equal(&g.wk)&&a.wv.equal(&g.wv));
                for (x,y) in [(&a.latent,&b.latent),(&a.len,&b.len),(&a.index.pools,&b.index.pools),
                    (&a.index.tail_k,&b.index.tail_k),(&a.index.tail_gate,&b.index.tail_gate)]{assert_ne!(x.data_ptr(),y.data_ptr());}
                bytes+=(a.wk.numel()+a.wv.numel())*4;
                layers.push(json!({"layer":i,"kind":"mla","derived_shared":true,"mutable_state_distinct":true}));
            },
            (LayerState::Kda(_),LayerState::Kda(a),LayerState::Kda(b))=>{
                assert_ne!(a.h.data_ptr(),b.h.data_ptr());assert_ne!(a.conv.data_ptr(),b.conv.data_ptr());
                layers.push(json!({"layer":i,"kind":"kda","mutable_state_distinct":true}));
            },_=>panic!("weight cache probe requires latent/KDA states"),
        }
    }
    // Existing full-model graph uses these shared immutable addresses. Its
    // evolving input and all branch states must remain exact against eager.
    let mut input=Tensor::from_slice(&[13041i64,13042,13043,13044]).to_device(eng.w.device);
    let parents=[None,Some(0),Some(0),Some(1)];
    let _=eng.tree_forward(&input,&a,&parents,true);tch::Cuda::synchronize(0);
    crate::tp::graph::begin().unwrap();let (graph_l,graph_s,graph_f)=eng.tree_forward_impl(&input,&a,&parents,true,false);
    crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
    for tokens in [[13041i64,13042,13043,13044],[0,13043,13042,13041]] {
        input.copy_(&Tensor::from_slice(&tokens).to_device(eng.w.device));graph.replay();
        let (eager_l,eager_s,eager_f)=eng.tree_forward(&input,&a,&parents,true);
        assert!(graph_l.equal(&eager_l)&&Tensor::cat(&graph_f,1).equal(&Tensor::cat(&eager_f,1)));
        for (g,e) in graph_s.iter().zip(&eager_s){assert_eq!(crate::forward::states_max_diff(g,e),0.);}
    }
    // Mutating a separately initialized request cannot touch a prior request
    // or its shared derived weights. Do not mutate wk/wv, which are read-only.
    for layer in &mut b.0 {match layer {
        LayerState::Kda(s)=>{let _=s.h.get(0).get(0).get(0).fill_(99.);let _=s.conv.get(0).get(0).fill_(98.);},
        LayerState::MlaLatent(s)=>{let _=s.latent.get(0).get(0).fill_(97.);let _=s.index.pools.get(0).get(0).fill_(96.);
            let _=s.index.tail_k.get(0).get(0).fill_(95.);let _=s.index.tail_gate.get(0).get(0).fill_(94.);let _=s.len.fill_(1);},
        _=>unreachable!(),
    }}
    assert_eq!(crate::forward::states_max_diff(&gold,&a),0.,"independent request mutation leaked");
    for (g,a) in gold.0.iter().zip(&a.0) {if let (LayerState::MlaLatent(g),LayerState::MlaLatent(a))=(g,a) {assert!(g.wk.equal(&a.wk)&&g.wv.equal(&a.wv));}}
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"gate":true,"complete":true,"identity_cases":identity,"layers":layers,
        "independent_prefills_exact":true,"changed_input_sibling_graph_exact":true,"writable_mutation_isolated":true,
        "derived_bytes_per_independent_initialization_before":bytes,"scope":"Model-owned read-only layout cache only; mutable state capacity is unchanged; no throughput claim"})).unwrap()).unwrap();
    match original{Some(v)=>std::env::set_var("GLM53_MLA_WEIGHT_CACHE",v),None=>std::env::remove_var("GLM53_MLA_WEIGHT_CACHE")};
    eprintln!("[mla-weight-cache] rank{rank} exact prefills/graph, source identity invalidation and mutable isolation passed; shared bytes={bytes}");
}

/// Probe-only projection of actual layer input; identical to tree's indexer
/// producer. No model reload and no synthetic activation substitution.
pub(crate) fn dsa_topk_project(w:&MlaWeights,x:&Tensor)->dsa::Projected {
    let cq=norm(mm16(x,&w.q_a),&w.q_a_ln);
    w.indexer.as_ref().unwrap().project(x,&cq)
}

/// W03 local gate: half_bmm vs FP64 reference on Half-rounded inputs, strided q, rows 1..8.
pub fn half_bmm_probe(out:&std::path::Path) {
    use serde_json::json;let _g=tch::no_grad_guard();let dev=tch::Device::Cuda(0);std::fs::create_dir_all(out).unwrap();
    let mut cases=Vec::new();
    for (n,k) in [(512i64,256i64),(256,512)] { for m in 1..=8i64 {
        tch::manual_seed(5+m+n);
        let w=(Tensor::randn([32,n,k],(Kind::Float,dev))*0.05).to_kind(Kind::Half);
        // strided like q.view([t,32,k]).transpose(0,1)
        let base=Tensor::randn([m,32,k],(Kind::Float,dev));let x=base.transpose(0,1);
        let y=half_bmm(&x,&w);
        let r=x.to_kind(Kind::Half).to_kind(Kind::Double).bmm(&w.to_kind(Kind::Double).transpose(1,2));
        let rel=f64::try_from((y.to_kind(Kind::Double)-&r).norm()/r.norm()).unwrap();
        assert!(rel<1e-5,"half bmm {n}x{k} m={m}: {rel}");
        cases.push(json!({"n":n,"k":k,"m":m,"rel":rel}));
    }}
    // C12 variant: bitwise vs half_bmm on BF16-representable Half weights, then timing over an 11-layer working set.
    let mut timing=Vec::new();
    for (n,k) in [(512i64,256i64),(256,512)] {
        let layers:Vec<(Tensor,[Tensor;6])>=(0..11).map(|i|{tch::manual_seed(900+i+n);
            let w=(Tensor::randn([32,n,k],(Kind::Float,dev))*0.05).to_kind(Kind::BFloat16).to_kind(Kind::Half).contiguous();
            let c=crate::c12::encode_parts(&w.reshape([32*n,k]));(w,c)}).collect();
        for m in 1..=8i64 {
            let base=Tensor::randn([m,32,k],(Kind::Float,dev));let x=base.transpose(0,1);
            for (w,c) in &layers[..2] {let a=half_bmm(&x,w);let b=c12_bmm(&x,c,32,n,k);assert!(a.equal(&b),"C12 bmm not bitwise {n}x{k} m={m}");}
            let time=|f:&dyn Fn(usize)|{for i in 0..22{f(i%11);}tch::Cuda::synchronize(0);let t=std::time::Instant::now();for i in 0..440{f(i%11);}tch::Cuda::synchronize(0);t.elapsed().as_secs_f64()*1e6/440.};
            let th=time(&|i|{let _=half_bmm(&x,&layers[i].0);});let tc=time(&|i|{let _=c12_bmm(&x,&layers[i].1,32,n,k);});
            eprintln!("[half-bmm] {n}x{k} m={m}: half {th:.1} us, c12 {tc:.1} us (bitwise equal)");
            timing.push(json!({"n":n,"k":k,"m":m,"half_us":th,"c12_us":tc}));
        }
    }
    std::fs::write(out.join("half-bmm-probe.json"),serde_json::to_string_pretty(&json!({"gate":true,"cases":cases,"c12_timing":timing})).unwrap()).unwrap();
    eprintln!("[half-bmm] PASS {}",cases.len());
}

/// D8 entry (layout = MlaCommitEntry in shim/dataflow.cu).
#[repr(C)]
pub(crate) struct MlaCommitEntry {node_len:*const i64,dst_pools:*mut f32,pool_row:*const f32,row_floats:i32,
    src_tk:*const f32,dst_tk:*mut f32,tk_floats:i32,src_tg:*const f32,dst_tg:*mut f32,tg_floats:i32,dst_len:*mut i64,has_row:i32}
pub(crate) fn commit_many(entries:&[MlaCommitEntry]) {
    extern "C"{fn rs_mla_commit_many(entries:*const std::ffi::c_void,n:i32)->i32;}
    for chunk in entries.chunks(16) {
        assert_eq!(unsafe{rs_mla_commit_many(chunk.as_ptr().cast(),chunk.len() as i32)},0,"fused MLA commit");
    }
}

/// Row-invariance audit taps (rows-first layouts); no-op unless forward::probe_begin was called.
fn mla_taps(cq:&Tensor,qb:&Tensor,absorbed:&Tensor,latent:&Tensor,projected:Option<&dsa::Projected>) {
    use crate::forward::probe_inner as tap;
    tap("mla_cq",cq);tap("mla_qb",qb);tap("mla_absorbed",&absorbed.transpose(0,1));tap("mla_latent",latent);
    if let Some(p)=projected {tap("mla_idx_q",&p.q);tap("mla_idx_k",&p.k);tap("mla_idx_gate",&p.gate);tap("mla_idx_mixing",&p.mixing);}
}
