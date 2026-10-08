// SPDX-License-Identifier: MIT
//! TP2(M1.5):c10d/NCCL 进程组包装。env:GLM53_TP_RANK/GLM53_TP_WORLD/
//! GLM53_MASTER_ADDR/GLM53_MASTER_PORT;NCCL_* 由启动脚本注入(见 env-tp2.sh)。

use std::sync::OnceLock;
use std::ffi::CString;

use tch::Tensor;

extern "C" {
    fn rs_pg_init(rank: i32, world: i32, master_addr: *const u8, port: i32) -> i32;
    fn rs_allreduce(p: *mut u8, numel: i64, dtype: i32) -> i32;
    fn rs_runtime_shutdown() -> i32;
    fn rs_set_fp32_accum() -> i32;
    fn rs_graph_begin() -> i32;
    fn rs_graph_end() -> i32;
    fn rs_graph_replay() -> i32;
}

/// CUDA Graph 捕获/回放(全步图,M1②)。
pub mod graph {
    /// Exclusive owner; no Clone, and captures remain serialized on one thread.
    pub struct Owned {id:i64,_thread:std::marker::PhantomData<std::rc::Rc<()>>}
    impl Owned {
        pub fn take()->Self {
            extern "C" {fn rs_graph_take()->i64;}
            let id=unsafe{rs_graph_take()};assert!(id>0,"no completed capture to own");Self{id,_thread:std::marker::PhantomData}
        }
        pub fn replay(&self) {
            extern "C" {fn rs_graph_replay_owned(id:i64)->i32;}
            assert_eq!(unsafe{rs_graph_replay_owned(self.id)},0,"owned graph replay");
        }
    }
    impl Drop for Owned {fn drop(&mut self){
        extern "C" {fn rs_graph_drop_owned(id:i64)->i32;}
        assert_eq!(unsafe{rs_graph_drop_owned(self.id)},0,"owned graph drop");
    }}

    pub fn owned_probe() {
        use tch::{Tensor,Kind,Device};
        let mut inputs=Vec::new();let mut outputs=Vec::new();let mut graphs=Vec::new();
        for i in 0..4 {
            let input=Tensor::ones([1024],(Kind::Float,Device::Cuda(0)));tch::Cuda::synchronize(0);
            begin().unwrap();let output=&input*(i+1) as f64;end().unwrap();
            graphs.push(Owned::take());inputs.push(input);outputs.push(output);
        }
        for turn in 0..12 {for i in [3usize,0,2,1] {
            let value=turn as f64-4.;let _=inputs[i].fill_(value);graphs[i].replay();
            assert_eq!(outputs[i].min().double_value(&[]),value*(i+1) as f64);
            assert_eq!(outputs[i].max().double_value(&[]),value*(i+1) as f64);
        }}
        graphs.remove(1);let _=inputs.remove(1);let _=outputs.remove(1);
        begin().unwrap();let legacy=&inputs[0]+1.;end().unwrap();replay().unwrap();
        assert_eq!(legacy.min().double_value(&[]),8.);destroy();
        graphs[2].replay();assert_eq!(outputs[2].min().double_value(&[]),28.);
        eprintln!("[owned-graphs] PASS independent 4 captures/48 replays, one eviction, legacy coexistence");
    }
    /// `graph-workspace-probe`: regression for the pooled-capture-stream cuBLAS workspace bug.
    /// 33 graphs each hold a TF32 split-K GEMM (M=8,N=32,K=4096, the DSA `mixing` shape). With
    /// pooled capture streams (GLM53_GRAPH_OWN_STREAM=0) graph 32 shares graph 0's stream and thus
    /// its workspace; dropping graph 0 clears that workspace and the next begin()'s emptyCache
    /// returns it to the driver, so replaying graph 32 must fault. With per-capture streams every
    /// replay must match its eager reference.
    pub fn workspace_probe() {
        use tch::{Tensor,Kind,Device};
        let _g=tch::no_grad_guard();let dev=Device::Cuda(0);crate::tp::set_tf32(true);tch::manual_seed(7);
        let w=Tensor::randn([32,4096],(Kind::Float,dev));
        let (mut inputs,mut outputs,mut graphs,mut refs)=(Vec::new(),Vec::new(),Vec::new(),Vec::new());
        for i in 0..33 {
            let x=Tensor::randn([8,4096],(Kind::Float,dev))*(1.+i as f64*0.01);
            let r=x.matmul(&w.transpose(0,1));tch::Cuda::synchronize(0);
            begin().unwrap();let y=x.matmul(&w.transpose(0,1));end().unwrap();
            graphs.push(Some(Owned::take()));inputs.push(x);outputs.push(y);refs.push(r);
        }
        for g in graphs.iter().flatten() {g.replay();}tch::Cuda::synchronize(0);
        graphs[0]=None;                                   // drop graph 0 (reset clears its stream's workspace)
        begin().unwrap();let _t=&inputs[1]+1.;end().unwrap();destroy();   // begin() runs emptyCache
        let big:Vec<Tensor>=(0..8).map(|_|Tensor::full([64<<20],7.,(Kind::Float,dev))).collect();  // reuse freed memory
        for round in 0..3 {for i in 1..33 {
            graphs[i].as_ref().unwrap().replay();tch::Cuda::synchronize(0);
            let d=(&outputs[i]-&refs[i]).abs().max().double_value(&[]);
            assert!(d<1e-2,"graph {i} round {round}: replay differs from eager by {d}");
        }}
        drop(big);
        eprintln!("[graph-workspace] PASS 33 TF32 split-K graphs, drop graph 0 + emptyCache, 96 replays (own stream={})",
            std::env::var("GLM53_GRAPH_OWN_STREAM").as_deref()!=Ok("0"));
    }
    /// True while the current stream is capturing (see rs_is_capturing).
    pub fn capturing()->bool {extern "C"{fn rs_is_capturing()->i32;}let r=unsafe{rs_is_capturing()};assert!(r>=0,"cudaStreamIsCapturing");r==1}
    pub fn memory()->(i64,i64) {
        extern "C" {fn rs_cuda_memory(allocated:*mut i64,reserved:*mut i64);}
        let (mut allocated,mut reserved)=(0,0);
        unsafe{rs_cuda_memory(&mut allocated,&mut reserved);}(allocated,reserved)
    }
    /// Repeated captures must reclaim retired pools without invalidating live
    /// graph inputs. This exercises the native shim, not Python's graph helper.
    pub fn churn_probe() {
        use tch::{Tensor,Kind,Device};
        let dev=Device::Cuda(0);let mut input=Tensor::ones([1],(Kind::Float,dev));
        let mut reserved=Vec::new();
        for round in 0..24 {
            tch::Cuda::synchronize(0);begin().unwrap();
            let output=Tensor::ones([64,1024,1024],(Kind::Float,dev))*&input;
            end().unwrap();
            for value in [0.,3.,-0.25] {
                let _=input.fill_(value);replay().unwrap();
                assert_eq!(output.min().double_value(&[]),value);
                assert_eq!(output.max().double_value(&[]),value);
            }
            tch::Cuda::synchronize(0);destroy();drop(output);
            let (a,r)=memory();reserved.push(r);
            eprintln!("[graph-churn] round={round} allocated={a} reserved={r}");
        }
        assert!(reserved.iter().max().unwrap()-reserved.iter().min().unwrap()<128*1024*1024,
            "retired graph pools accumulate: {reserved:?}");
        eprintln!("[graph-churn] PASS 24 captures / 72 changed-input replays");
    }
    pub fn destroy() {
        extern "C" { fn rs_graph_destroy() -> i32; }
        assert_eq!(unsafe { rs_graph_destroy() }, 0);
    }
    pub fn begin() -> Result<(), String> {
        let rc = unsafe { super::rs_graph_begin() };
        if rc == 0 { Ok(()) } else { Err("graph_begin 失败".into()) }
    }
    #[track_caller]
    pub fn end() -> Result<(), String> {
        let rc = unsafe { super::rs_graph_end() };
        if rc != 0 { return Err("graph_end 失败".into()); }
        audit_pool(std::panic::Location::caller());
        Ok(())
    }
    /// GLM53_GRAPH_POOL_AUDIT=1: report every capture site that leaves live tensors in its graph's memory pool (bytes
    /// still allocated right after the capture): the graph outputs/state that must move to buffers allocated before
    /// capture before graph pools can be shared. Printed once per site and size.
    fn audit_pool(site:&std::panic::Location) {
        if std::env::var("GLM53_GRAPH_POOL_AUDIT").as_deref()!=Ok("1") {return;}
        extern "C" { fn rs_graph_pool_live_bytes() -> i64; }
        let live=unsafe{rs_graph_pool_live_bytes()};
        use std::sync::Mutex;
        static SEEN:Mutex<Vec<(String,i64)>>=Mutex::new(Vec::new());
        let key=(format!("{}:{}",site.file(),site.line()),live);
        let mut seen=SEEN.lock().unwrap();
        if !seen.contains(&key) {eprintln!("[graph-pool-audit] {} live {} bytes",key.0,live);seen.push(key);}
    }
    pub fn replay() -> Result<(), String> {
        let rc = unsafe { super::rs_graph_replay() };
        if rc == 0 { Ok(()) } else { Err("graph_replay 失败".into()) }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Tp {
    pub rank: usize,
    pub world: usize,
}

static TP: OnceLock<Tp> = OnceLock::new();

pub fn world() -> Tp {
    TP.get().copied().unwrap_or(Tp { rank: 0, world: 1 })
}

pub fn is_tp() -> bool {
    world().world > 1
}

/// This process's rank and world from the environment, without initializing anything (init_from_env
/// reads the same variables): lets rank-sliced disk reads start before the process group is up.
pub fn env_tp() -> Tp {
    let world: usize = std::env::var("GLM53_TP_WORLD").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let rank: usize = std::env::var("GLM53_TP_RANK").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    if world == 1 { Tp { rank: 0, world: 1 } } else { Tp { rank, world } }
}

/// 从环境初始化(幂等)。world=1 时不碰 NCCL。
pub fn init_from_env() -> Tp {
    *TP.get_or_init(|| {
        let world: usize = std::env::var("GLM53_TP_WORLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        let rank: usize = std::env::var("GLM53_TP_RANK")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        assert!(world == 1 || world == 2, "仅支持 TP1/TP2");
        assert!(rank < world, "rank 必须小于 world");
        if world == 1 {
            return Tp { rank: 0, world: 1 };
        }
        let addr = std::env::var("GLM53_MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".into());
        let port: i32 = std::env::var("GLM53_MASTER_PORT")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(29631);
        let addr = CString::new(addr).expect("MASTER_ADDR 含 NUL");
        assert!((1..=65535).contains(&port), "master port 超出范围");
        let rc = unsafe { rs_pg_init(rank as i32, world as i32, addr.as_ptr().cast(), port) };
        assert!(rc == 0, "pg_init 失败 rc={rc}");
        eprintln!("[tp] rank {rank}/{world} NCCL 就绪");
        Tp { rank, world }
    })
}

/// fp16 GEMM fp32 累加(引擎启动时调用一次)。
pub fn set_tf32(enabled:bool){extern "C"{fn rs_set_tf32(enabled:i32)->i32;}assert_eq!(unsafe{rs_set_tf32(i32::from(enabled))},0);}
pub fn set_fp32_accum() {
    set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
    unsafe {
        rs_set_fp32_accum();
    }
}

/// 原地 allreduce(SUM),fp32/fp16;host 不阻塞(流依赖)。
thread_local! {
    static AR_PREFETCH: std::cell::RefCell<Vec<(usize, i64)>> = std::cell::RefCell::new(Vec::new());
}
/// GLM53_AR_PREFETCH=1: weight regions registered by the verify layer loop are bulk-prefetched into L2
/// (cp.async.bulk.prefetch, fire-and-forget, one tiny kernel on the current stream) right before the next
/// all-reduce, so the network wait of that all-reduce streams the next GEMM's weights. Hint only: no numerics change.
pub fn ar_prefetch_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("GLM53_AR_PREFETCH").as_deref() == Ok("1"))
}
/// Prefetch budget per all-reduce (GLM53_AR_PREFETCH_MB, default 8 MiB: the ~27 us wait at ~300 GB/s).
pub fn ar_prefetch_bytes() -> i64 {
    static B: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("GLM53_AR_PREFETCH_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(8i64) << 20)
}
/// Registers regions for the next all-reduce on this thread (replaces any unconsumed ones). Regions beyond the
/// budget are clipped in order.
pub fn set_ar_prefetch(regions: &[(&Tensor, i64)]) {
    if !ar_prefetch_enabled() { return; }
    let mut left = ar_prefetch_bytes();
    let v: Vec<(usize, i64)> = regions.iter().filter_map(|(t, bytes)| {
        if !t.is_contiguous() || !t.device().is_cuda() { return None; }   // stride-0 placeholders (FP8_FREE_SOURCE) own no such bytes
        let b = (*bytes).min(t.numel() as i64 * t.kind().elt_size_in_bytes() as i64).min(left);
        left -= b.max(0);
        (b > 0).then(|| (t.data_ptr() as usize, b))
    }).collect();
    AR_PREFETCH.with(|p| *p.borrow_mut() = v);
}
/// Appends raw device regions (already budgeted by the caller) to the next all-reduce's prefetch.
pub fn add_ar_prefetch_raw(regions:Vec<(usize,i64)>) {
    if !ar_prefetch_enabled() { return; }
    AR_PREFETCH.with(|p| p.borrow_mut().extend(regions.into_iter().filter(|r| r.1 > 0)));
}
pub fn clear_ar_prefetch() {
    if ar_prefetch_enabled() { AR_PREFETCH.with(|p| p.borrow_mut().clear()); }
}
fn issue_ar_prefetch() {
    if !ar_prefetch_enabled() { return; }
    extern "C" { fn rs_l2_prefetch(p: *const std::ffi::c_void, bytes: i64) -> i32; }
    AR_PREFETCH.with(|p| for (ptr, bytes) in p.borrow_mut().drain(..) {
        assert_eq!(unsafe { rs_l2_prefetch(ptr as *const std::ffi::c_void, bytes) }, 0, "l2 prefetch");
    });
}

pub fn allreduce(t: &Tensor) {
    if !is_tp() {
        return;
    }
    settle_pending();
    issue_ar_prefetch();
    let dtype = match t.kind() {
        tch::Kind::Float => 2,
        tch::Kind::Half => 0,
        other => panic!("allreduce dtype {other:?} 未支持"),
    };
    let rc = unsafe { rs_allreduce(t.data_ptr() as *mut u8, t.numel() as i64, dtype) };
    assert!(rc == 0, "allreduce 失败 rc={rc}");
}

/// I3 step 2 (GLM53_AR_FUSED=1, default off): the lean RDMA allreduce only sends and waits; its single consumer (the
/// four-stream MHC post, `mhc::post_ar`) forms rank0+rank1 from the local partial and the peer's recv slot, applying the
/// same rounding in the same order as the separate reduction + round + post (L0). The peer slot is only valid until this
/// rank's next allreduce, so a pending sum must be consumed or materialized before any other collective.
static AR_FUSED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);   // 2: not read yet
pub fn ar_fused_enabled()->bool {
    use std::sync::atomic::Ordering::Relaxed;
    match AR_FUSED.load(Relaxed) {0=>false,1=>true,_=>{
        let v=is_tp() && std::env::var("GLM53_AR_FUSED").as_deref() == Ok("1") && std::env::var("GLM53_RDMA_AR").as_deref() == Ok("1");
        AR_FUSED.store(v as u8,Relaxed);v}}
}
/// Re-read cached environment switches (in-process A/B, `spec_probe::row_timing` with GLM53_I1_AB). Only between
/// captures: graphs already captured keep the kernels they recorded.
pub fn reset_flag_cache() {AR_FUSED.store(2,std::sync::atomic::Ordering::Relaxed);}
/// Send-only allreduce of `t` (contiguous FP32). false: not eligible, nothing launched (use `allreduce`).
pub fn allreduce_send(t:&Tensor)->bool {
    if !ar_fused_enabled() || t.kind()!=tch::Kind::Float || !t.is_contiguous() || !t.device().is_cuda() { return false; }
    settle_pending();
    issue_ar_prefetch();
    extern "C" { fn rs_allreduce_send(p:*mut u8,numel:i64)->i32; }
    match unsafe { rs_allreduce_send(t.data_ptr() as *mut u8, t.numel() as i64) } {0=>true,1=>false,rc=>panic!("send-only allreduce rc={rc}")}
}
/// An unsummed partial whose peer half waits in the recv slot, with the rounding its consumer must apply.
#[derive(Clone,Copy,PartialEq,Debug)]
pub struct PendingSum {pub ptr:usize,pub numel:i64,pub round_half:bool}
thread_local! {
    static DEFER_SUM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PENDING_SUM: std::cell::Cell<Option<PendingSum>> = const { std::cell::Cell::new(None) };
}
/// Runs `f` (an attention block) letting its row-parallel output projection defer the TP sum; returns the pending
/// record when the returned tensor is exactly that unsummed partial. Anything else is materialized first.
pub fn with_deferred_sum(f:impl FnOnce()->Tensor)->(Tensor,Option<PendingSum>) {
    if !ar_fused_enabled() { return (f(),None); }
    DEFER_SUM.with(|d|d.set(true));let r=f();DEFER_SUM.with(|d|d.set(false));
    let p=PENDING_SUM.with(|c|c.take());
    match p {
        Some(p) if r.data_ptr() as usize==p.ptr && r.numel() as i64==p.numel && r.is_contiguous() && r.kind()==tch::Kind::Float => (r,Some(p)),
        Some(p) => {materialize(p);(r,None)},
        None => (r,None),
    }
}
/// Inside `with_deferred_sum`: send-only reduce of a row-parallel partial, recorded as pending. false: not deferred.
pub fn defer_sum(y:&Tensor,round_half:bool)->bool {
    if !DEFER_SUM.with(|d|d.get()) { return false; }
    if let Some(p)=PENDING_SUM.with(|c|c.take()) { materialize(p); }
    if !allreduce_send(y) { return false; }
    PENDING_SUM.with(|c|c.set(Some(PendingSum{ptr:y.data_ptr() as usize,numel:y.numel() as i64,round_half})));true
}
/// A pending sum inside a deferral scope must be complete before any later collective reuses the sequence counter.
fn settle_pending() {
    if ar_fused_enabled() { if let Some(p)=PENDING_SUM.with(|c|c.take()) { materialize(p); } }
}
pub fn settle_pending_pub(){settle_pending();}
/// Fallback consumer: writes rank0+rank1 (and the pending rounding) into the partial in place.
pub fn materialize(p:PendingSum) {
    extern "C" { fn rs_ar_materialize(p:*mut f32,numel:i64,round_half:i32)->i32; }
    assert_eq!(unsafe { rs_ar_materialize(p.ptr as *mut f32,p.numel,i32::from(p.round_half)) },0,"allreduce materialize");
}

/// TP2 reduce-scatter (sum): `t` [2*r, ...] contiguous -> this rank's rows [r, ...].
pub fn reduce_scatter_rows(t:&Tensor)->Tensor {
    settle_pending();
    assert!(is_tp());assert!(t.is_contiguous());assert_eq!(t.size()[0]%2,0);
    extern "C"{fn rs_reduce_scatter(i:*mut u8,o:*mut u8,n:i64,dtype:i32)->i32;}
    let mut shape=t.size();shape[0]/=2;let out=Tensor::empty(shape.as_slice(),(t.kind(),t.device()));
    let dtype=match t.kind(){tch::Kind::Float=>2,tch::Kind::Half=>0,k=>panic!("reduce_scatter {k:?}")};
    assert_eq!(unsafe{rs_reduce_scatter(t.data_ptr() as *mut u8,out.data_ptr() as *mut u8,out.numel() as i64,dtype)},0,"reduce_scatter");out
}
/// TP2 all-gather of row shards: `t` [r, ...] -> [2*r, ...] (rank 0 rows first).
pub fn all_gather_rows(t:&Tensor)->Tensor {
    settle_pending();
    assert!(is_tp());let t=t.contiguous();
    extern "C"{fn rs_all_gather(i:*mut u8,o:*mut u8,n:i64,dtype:i32)->i32;}
    let mut shape=t.size();shape[0]*=2;let out=Tensor::empty(shape.as_slice(),(t.kind(),t.device()));
    let dtype=match t.kind(){tch::Kind::Float=>2,tch::Kind::Half=>0,k=>panic!("all_gather {k:?}")};
    assert_eq!(unsafe{rs_all_gather(t.data_ptr() as *mut u8,out.data_ptr() as *mut u8,t.numel() as i64,dtype)},0,"all_gather");out
}
/// An all-gather in flight (GLM53_PREFILL_AG_OVERLAP): `finish` makes the current stream wait and returns the rows
/// in rank order, exactly as all_gather_rows.
pub struct PendingGather {out:Tensor,_input:Tensor,handle:i32}
pub fn all_gather_rows_start(t:&Tensor)->PendingGather {
    settle_pending();
    assert!(is_tp());let t=t.contiguous();
    extern "C"{fn rs_all_gather_async(i:*mut u8,o:*mut u8,n:i64,dtype:i32)->i32;}
    let mut shape=t.size();shape[0]*=2;let out=Tensor::empty(shape.as_slice(),(t.kind(),t.device()));
    let dtype=match t.kind(){tch::Kind::Float=>2,tch::Kind::Half=>0,k=>panic!("all_gather {k:?}")};
    let handle=unsafe{rs_all_gather_async(t.data_ptr() as *mut u8,out.data_ptr() as *mut u8,t.numel() as i64,dtype)};
    assert!(handle>=0,"async all_gather");PendingGather{out,_input:t,handle}
}
impl PendingGather {
    pub fn finish(self)->Tensor {
        extern "C"{fn rs_work_wait(h:i32)->i32;}
        assert_eq!(unsafe{rs_work_wait(self.handle)},0,"all_gather wait");self.out
    }
}
/// Rows of `own` (this rank's) and `other` in rank order, concatenated along dim 0.
pub fn cat_rank_rows(own:&Tensor,other:&Tensor)->Tensor {
    if world().rank==0 {Tensor::cat(&[own,other],0)} else {Tensor::cat(&[other,own],0)}
}
pub fn ag_overlap_enabled()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_PREFILL_AG_OVERLAP").as_deref()==Ok("1"))}
/// Opt-in until TP2 numerical/performance qualification is complete.
pub fn dense_enabled() -> bool {
    is_tp() && std::env::var("GLM53_DENSE_TP").as_deref() == Ok("1")
}

pub fn dense_sum(t: Tensor) -> Tensor {
    if dense_enabled() { allreduce(&t); }
    t
}

pub fn small_comm_enabled()->bool {
    world().world==2 && std::env::var("GLM53_TP_SMALL_COMM").as_deref()==Ok("1") &&
        std::env::var("GLM53_TP_SMALL_COMM_ACTIVE").map_or(true,|v|v=="1")
}

pub fn shutdown() {
    assert_eq!(unsafe { rs_runtime_shutdown() }, 0, "runtime shutdown failed");
}

/// Exercise both communicators in one CUDA graph, including threshold edges,
/// changing inputs and returning to the small communicator after a large SUM.
pub fn network_probe(out:&std::path::Path) {
    use tch::{Device,Kind,Tensor};use serde_json::json;
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let tp=init_from_env();assert_eq!(tp.world,2);assert!(small_comm_enabled());
    let dev=Device::Cuda(0);std::fs::create_dir_all(out).unwrap();let mut cases=Vec::new();
    for (kind,small_n,large_n) in [(Kind::Float,1,65537),(Kind::Float,4096,262144),
        (Kind::Float,65536,65537),(Kind::Half,131072,131073)] {
        let make=|n:i64|Tensor::arange(n,(Kind::Float,dev)).remainder(65)-32.;
        let s=make(small_n);let l=make(large_n);
        let mut si=(&s+tp.rank as f64).to_kind(kind);let mut li=(&l+tp.rank as f64).to_kind(kind);
        let mut rounds=Vec::new();
        for active in [false,true,true,false] {
            std::env::set_var("GLM53_TP_SMALL_COMM_ACTIVE",if active{"1"}else{"0"});
            let op=|s:&Tensor,l:&Tensor| {
                let a=s.copy();let b=l.copy();allreduce(&a);allreduce(&b);allreduce(&a);(a,b)
            };
            for _ in 0..3{let _=op(&si,&li);}tch::Cuda::synchronize(0);
            graph::begin().unwrap();let (a,b)=op(&si,&li);graph::end().unwrap();
            for scale in [1.,-1.,0.25,0.] {
                si.copy_(&((&s+tp.rank as f64)*scale).to_kind(kind));
                li.copy_(&((&l+tp.rank as f64)*scale).to_kind(kind));
                graph::replay().unwrap();
                let expected_a=((&s*2.+1.)*(scale*2.)).to_kind(kind);
                let expected_b=((&l*2.+1.)*scale).to_kind(kind);
                assert!(a.equal(&expected_a) && b.equal(&expected_b),"mixed communicator graph SUM differs");
                assert!(a.isfinite().all().int64_value(&[])!=0 && b.isfinite().all().int64_value(&[])!=0);
            }
            si.copy_(&(&s+tp.rank as f64).to_kind(kind));li.copy_(&(&l+tp.rank as f64).to_kind(kind));
            tch::Cuda::synchronize(0);allreduce(&Tensor::zeros([1],(Kind::Float,dev)));tch::Cuda::synchronize(0);
            let start=std::time::Instant::now();for _ in 0..256{graph::replay().unwrap();}tch::Cuda::synchronize(0);
            rounds.push(json!({"small_comm":active,"graph_us":start.elapsed().as_secs_f64()*1e6/256.}));
            graph::destroy();
        }
        cases.push(json!({"dtype":format!("{kind:?}"),"small_elements":small_n,"large_elements":large_n,"exact":true,"rounds":rounds}));
        std::fs::write(out.join(format!("network-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({"rank":tp.rank,"cases":cases})).unwrap()).unwrap();
        eprintln!("[tp-network-probe] rank{} {kind:?} {small_n}/{large_n} mixed graphs exact PASS",tp.rank);
    }
}

/// C1: pure TP2 allreduce latency inside a CUDA graph (both ranks aligned by a barrier
/// allreduce before timing). Sizes match verifier payloads: rows x 4096 FP32, and 2x packed.
pub fn allreduce_latency_probe(out:&std::path::Path) {
    use tch::{Device,Kind,Tensor};use serde_json::json;
    let _guard=tch::no_grad_guard();let tp=init_from_env();assert_eq!(tp.world,2);
    let dev=Device::Cuda(0);std::fs::create_dir_all(out).unwrap();let mut cases=Vec::new();
    let filler=Tensor::ones([12<<20],(Kind::Float,dev));
    let rdma=std::env::var("GLM53_RDMA_AR_INIT").as_deref()==Ok("1");
    if rdma {
        // bitwise check vs NCCL on random data (different per rank)
        for rows in [1i64,2,3,5,8,16] {
            tch::manual_seed(100+rows+tp.rank as i64);let x=Tensor::randn([rows,4096],(Kind::Float,dev));
            std::env::set_var("GLM53_RDMA_AR","0");let a=x.copy();allreduce(&a);
            std::env::set_var("GLM53_RDMA_AR","1");let b=x.copy();allreduce(&b);
            tch::Cuda::synchronize(0);
            assert!(a.equal(&b),"lean RDMA allreduce differs from NCCL rows={rows}");
            // graph replay with changing input
            let g=x.copy();crate::tp::graph::begin().unwrap();allreduce(&g);crate::tp::graph::end().unwrap();
            for k in [1.0f64,-3.0,0.5] {let _=g.shallow_clone().copy_(&(&x*k));crate::tp::graph::replay().unwrap();
                std::env::set_var("GLM53_RDMA_AR","0");let r=(&x*k).copy();allreduce(&r);std::env::set_var("GLM53_RDMA_AR","1");
                tch::Cuda::synchronize(0);assert!(g.equal(&r),"lean RDMA graph replay differs rows={rows}");}
            crate::tp::graph::destroy();
        }
        eprintln!("[allreduce-latency] rank{} lean RDMA bitwise == NCCL (eager + graph) PASS",tp.rank);
    }
    for mode in if rdma{vec!["nccl","rdma"]}else{vec!["nccl"]} {
    std::env::set_var("GLM53_RDMA_AR",if mode=="rdma"{"1"}else{"0"});
    for rows in [1i64,2,3,4,8,16] {
        for gap in [false,true] {
            let x=Tensor::ones([rows,4096],(Kind::Float,dev));
            let op=|| {for _ in 0..64 {if gap{let _=filler.sum(Kind::Float);} allreduce(&x);}};
            op();tch::Cuda::synchronize(0);
            graph::begin().unwrap();op();graph::end().unwrap();
            for _ in 0..3{graph::replay().unwrap();}tch::Cuda::synchronize(0);
            let mut v=Vec::new();
            for _ in 0..7 {
                allreduce(&Tensor::zeros([1],(Kind::Float,dev)));tch::Cuda::synchronize(0);
                let t=std::time::Instant::now();graph::replay().unwrap();tch::Cuda::synchronize(0);
                v.push(t.elapsed().as_secs_f64()*1e6/64.);
            }
            // filler-only baseline to subtract
            let base=if gap {
                graph::destroy();
                graph::begin().unwrap();for _ in 0..64{let _=filler.sum(Kind::Float);}graph::end().unwrap();
                for _ in 0..3{graph::replay().unwrap();}tch::Cuda::synchronize(0);
                let t=std::time::Instant::now();for _ in 0..5{graph::replay().unwrap();}tch::Cuda::synchronize(0);
                t.elapsed().as_secs_f64()*1e6/320.
            } else {0.};
            graph::destroy();
            v.sort_by(|a,b|a.partial_cmp(b).unwrap());
            cases.push(json!({"mode":mode,"rows":rows,"bytes":rows*4096*4,"with_48MB_read_between":gap,"us_per_iteration_median":v[3],"us_min":v[0],"filler_us":base,"allreduce_us_est":v[3]-base}));
            eprintln!("[allreduce-latency] rank{} {mode} rows={rows} gap={gap} {:.1} us/iter (filler {:.1})",tp.rank,v[3],base);
        }
    }
    }
    // Prefill-sized NCCL allreduces (eager, main group; protocol from NCCL_PROTO if set).
    std::env::set_var("GLM53_RDMA_AR","0");
    for rows in [256i64,1024,2048,4096] {
        let x=Tensor::ones([rows,4096],(Kind::Float,dev));for _ in 0..3{allreduce(&x);}tch::Cuda::synchronize(0);
        let mut v=Vec::new();
        for _ in 0..5 {let t=std::time::Instant::now();for _ in 0..10{allreduce(&x);}tch::Cuda::synchronize(0);v.push(t.elapsed().as_secs_f64()*1e6/10.);}
        v.sort_by(|a,b|a.partial_cmp(b).unwrap());let bytes=rows*4096*4;
        cases.push(json!({"mode":"nccl-large","proto":std::env::var("NCCL_PROTO").ok(),"rows":rows,"bytes":bytes,"us_median":v[2],"algbw_gbps":bytes as f64/v[2]/1e3}));
        eprintln!("[allreduce-latency] rank{} nccl-large proto={:?} rows={rows} {:.1} us ({:.1} GB/s)",tp.rank,std::env::var("NCCL_PROTO").ok(),v[2],bytes as f64/v[2]/1e3);
    }
    std::fs::write(out.join(format!("allreduce-latency-rank{}.json",tp.rank)),serde_json::to_string_pretty(&json!({"cases":cases})).unwrap()).unwrap();
}

/// Stream-ordered upload of a small host slice into device tensor `dst` (same numel and dtype) without the
/// stream drain of a blocking pageable copy (ATen copy_ from pageable memory ends in cudaStreamSynchronize,
/// so the GPU idles while the host enqueues the next graph). The source is pinned; PyTorch's caching host
/// allocator records the copy's event and keeps the block alive until it completes. Outside capture only;
/// Opt-in (GLM53_ASYNC_UPLOAD=1): serve ABBA showed no gain (bench/w1), default keeps the blocking copy. Same bytes either way (L0).
pub fn upload<T:tch::kind::Element>(dst:&Tensor,v:&[T]) {
    let blocking=std::env::var("GLM53_ASYNC_UPLOAD").as_deref()!=Ok("1") || !dst.device().is_cuda() || graph::capturing();
    let src=Tensor::from_slice(v);
    assert_eq!(src.numel(),dst.numel(),"upload size");assert_eq!(src.kind(),dst.kind(),"upload dtype");assert!(dst.is_contiguous());
    if blocking {dst.shallow_clone().copy_(&src.view(dst.size().as_slice()));return;}
    let pinned=src.pin_memory(dst.device()).view(dst.size().as_slice());
    // Pinned -> device with non_blocking (no stream drain), then an ordinary device copy into `dst`.
    let staged=pinned.to_device_(dst.device(),dst.kind(),true,false);
    dst.shallow_clone().copy_(&staged);
}
