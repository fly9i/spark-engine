// SPDX-License-Identifier: MIT
//! M1.2 权重注册表:检查点 → tch 张量(非专家部分全量驻留 GPU fp32)。
//! 专家保持 trellis 形态,由 moe 模块按需解码(位预算驻留池)。
//! 语义与 engine/glm53/model.py 完全一致(M0 已验收)。

use std::{path::Path,cell::RefCell};

use tch::{Device, Kind, Tensor};

use crate::config::{Config, LayerPlan};
use crate::safetensors::ShardIndex;

pub struct HcParams {
    pub attn_fn: Tensor,    // [24, 16384]
    pub attn_scale: Tensor, // [3]
    pub attn_base: Tensor,  // [24]
    pub ffn_fn: Tensor,
    pub ffn_scale: Tensor,
    pub ffn_base: Tensor,
    pub in_ln: Tensor,  // [4096]
    pub post_ln: Tensor, // [4096]
}

pub struct KdaWeights {
    pub wq: Tensor, pub wk: Tensor, pub wv: Tensor, pub wo: Tensor, // [8192,4096]×3,[4096,8192]
    pub wb: Tensor, // [64,4096]
    pub fa: Tensor, pub fb: Tensor, // [128,4096],[8192,128]
    pub ga: Tensor, pub gb: Tensor, // [128,4096],[8192,128]
    pub conv_q: Tensor, pub conv_k: Tensor, pub conv_v: Tensor, // [8192,4]
    pub dt_bias: Tensor, // [64,128]
    pub a_log: Tensor,   // [64]
    pub o_norm: Tensor,  // [4096]
    pub constants: Option<(Tensor,Tensor)>, // exp(A_log), concatenated conv weights after TP
}

impl KdaWeights {
    pub fn prepare_constants(&mut self) {
        self.constants=Some((self.a_log.exp(),Tensor::cat(&[&self.conv_q,&self.conv_k,&self.conv_v],0)));
    }
    pub fn decay_base(&self)->Tensor {
        if std::env::var("GLM53_STATIC_TENSORS").as_deref()==Ok("1") {
            if let Some((a,_))=&self.constants{return a.shallow_clone();}
        }
        self.a_log.exp()
    }
    pub fn convolution_weights(&self)->Tensor {
        // GLM53_KDA_CONV_CACHED=1: the load-time concatenation (same values) instead of a cat per call.
        if std::env::var("GLM53_STATIC_TENSORS").as_deref()==Ok("1") || std::env::var("GLM53_KDA_CONV_CACHED").as_deref()==Ok("1") {
            if let Some((_,w))=&self.constants{return w.shallow_clone();}
        }
        Tensor::cat(&[&self.conv_q,&self.conv_k,&self.conv_v],0)
    }
}

pub struct MlaWeights {
    pub indexer: Option<crate::dsa::Weights>,
    pub q_a: Tensor, pub q_b: Tensor, // [1536,4096],[16384,1536]
    pub kv_a: Tensor, // [512,4096]
    pub kv_b: Tensor, // [32768,512]
    pub wo: Tensor,   // [4096,16384]
    pub q_a_ln: Tensor, pub kv_a_ln: Tensor, // [1536],[512]
    pub(crate) latent_cache:RefCell<Option<MlaLatentCache>>,
}

/// Model-owned immutable projection layout. The source owner prevents allocator
/// address reuse; frozen metadata/version also catch same-storage replacement.
pub(crate) struct MlaLatentCache {
    _source:Tensor,ptr:usize,shape:Vec<i64>,stride:Vec<i64>,kind:Kind,device:Device,
    version:i64,heads:i64,wk:Tensor,wv:Tensor,
}
impl MlaWeights {
    fn build_latent_projections(&self)->(Tensor,Tensor) {
        let heads=self.q_b.size()[0]/256;
        let kv=self.kv_b.view([heads,512,512]);
        // Preserve exactly State::new's dtype boundary and contiguous layouts.
        (kv.narrow(1,0,256).to_kind(Kind::Float).contiguous(),
            kv.narrow(1,256,256).transpose(1,2).to_kind(Kind::Float).contiguous())
    }
    pub(crate) fn latent_projections(&self)->(Tensor,Tensor) {
        if std::env::var("GLM53_MLA_WEIGHT_CACHE").as_deref()!=Ok("1") {
            // An ABBA arm may disable this after an enabled arm. Drop only the
            // model's cache ownership; previously returned state views survive.
            self.latent_cache.borrow_mut().take();
            return self.build_latent_projections();
        }
        let heads=self.q_b.size()[0]/256;let source=&self.kv_b;
        let ptr=source.data_ptr() as usize;let shape=source.size();let stride=source.stride();
        let kind=source.kind();let device=source.device();let version=source.internal_version();
        let mut cache=self.latent_cache.borrow_mut();
        let valid=cache.as_ref().is_some_and(|c|c.ptr==ptr&&c.shape==shape&&c.stride==stride&&c.kind==kind&&c.device==device&&c.version==version&&c.heads==heads);
        if !valid {
            let (wk,wv)=self.build_latent_projections();
            *cache=Some(MlaLatentCache{_source:source.shallow_clone(),ptr,shape,stride,kind,device,version,heads,wk,wv});
        }
        let cache=cache.as_ref().unwrap();(cache.wk.shallow_clone(),cache.wv.shallow_clone())
    }
    pub(crate) fn invalidate_latent_cache(&mut self) {self.latent_cache.get_mut().take();}
}

pub struct DenseMlp {
    pub wg: Tensor, pub wu: Tensor, pub wd: Tensor, // [12288,4096]×2,[4096,12288]
}

pub struct MoeMeta {
    pub w_gate: Tensor, // [256,4096] router
    pub bias: Tensor,   // [256]
    pub sh_wg: Tensor, pub sh_wu: Tensor, pub sh_wd: Tensor, // shared experts 线性权重
}

pub struct LayerWeights {
    pub plan: LayerPlanOwned,
    pub hc: HcParams,
    pub kda: Option<KdaWeights>,
    pub mla: Option<MlaWeights>,
    pub dense: Option<DenseMlp>,
    pub moe: Option<MoeMeta>,
}

pub struct LayerPlanOwned {
    pub attn: String, // "kda" | "dsa"
    pub mlp: String,  // "dense" | "sparse"
}

pub struct ModelWeights {
    pub embed: Tensor, // [V,4096]
    pub final_norm: Tensor,
    pub lm_head: Tensor, // [V,4096]
    pub layers: Vec<LayerWeights>,
    pub device: Device,
    pub vocab_shard: Option<(i64,i64)>, // (first global row, full vocabulary)
}

/// M4 (GLM53_BF16_RESIDENT=1): keep a BF16 checkpoint tensor as BF16 (the FP32 copy it replaces is
/// its exact widening; consumers widen in-kernel). Asserts the source really is BF16-exact.
pub(crate) fn bf16_resident_enabled()->bool {std::env::var("GLM53_BF16_RESIDENT").as_deref()==Ok("1")}
fn load_bf16_resident(idx: &mut ShardIndex, name: &str, dev: Device) -> Tensor {
    let t=load(idx,name,dev);
    if !bf16_resident_enabled() {return t;}
    let b=t.to_kind(Kind::BFloat16);
    assert!(b.to_kind(Kind::Float).equal(&t),"{name} is not BF16-exact; cannot keep it BF16-resident");
    b
}
/// L1-a (GLM53_DSA_INDEX_BF16=1): DSA indexer projections BF16-resident (exact source values).
fn load_index_bf16(idx: &mut ShardIndex, name: &str, dev: Device) -> Tensor {
    let t=load(idx,name,dev);
    if std::env::var("GLM53_DSA_INDEX_BF16").as_deref()!=Ok("1") {return t;}
    let b=t.to_kind(Kind::BFloat16);
    assert!(b.to_kind(Kind::Float).equal(&t),"{name} is not BF16-exact");
    b
}
/// Startup breakdown of ModelWeights::load (ns): tensor loads, ablit, per-layer shard/prepare, index scan.
static LOAD_NS:[std::sync::atomic::AtomicU64;6]=[const{std::sync::atomic::AtomicU64::new(0)};6];
static LOAD_BYTES:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
thread_local!{static LOAD_BUF:std::cell::RefCell<Option<crate::expert_load::HostBuf>>=const{std::cell::RefCell::new(None)};}
fn load_ns(i:usize,t:std::time::Instant) {LOAD_NS[i].fetch_add(t.elapsed().as_nanos() as u64,std::sync::atomic::Ordering::Relaxed);}
/// GLM53_FAST_LOAD (default on; =0 old path): upload each checkpoint tensor in its stored dtype (pread into a
/// reused host buffer, page cache warmed ahead in load order) and widen on the device. BF16/F16 -> F32 is exact
/// either side, so the resident FP32 tensor is bit-identical to the old CPU conversion, without the CPU FP32
/// copy (2x the bytes). GLM53_LOAD_CHECK=1 also builds the old tensor and asserts equal bits.
pub(crate) fn fast_load_enabled()->bool {std::env::var("GLM53_FAST_LOAD").as_deref()!=Ok("0")}
pub(crate) fn load_check_enabled()->bool {std::env::var("GLM53_LOAD_CHECK").as_deref()==Ok("1")}
fn load(idx: &mut ShardIndex, name: &str, dev: Device) -> Tensor {
    let t0=std::time::Instant::now();
    if let Some(e)=idx.entries.get(name) {LOAD_BYTES.fetch_add(e.nbytes as u64,std::sync::atomic::Ordering::Relaxed);}
    let t=load_timed(idx,name,dev);load_ns(0,t0);t
}
fn load_timed(idx: &mut ShardIndex, name: &str, dev: Device) -> Tensor {
    // A CPU destination would alias the mapping: keep the owning copy there.
    if !fast_load_enabled() || !dev.is_cuda() {return load_slow(idx,name,dev);}
    let t=upload_stored(idx,name,dev).to_kind(Kind::Float);
    if load_check_enabled() {
        let old=load_slow(idx,name,dev);
        assert!(old.view_dtype(Kind::Int).equal(&t.view_dtype(Kind::Int)),"GLM53_LOAD_CHECK: {name} differs from the old load");
    }
    t
}
/// One checkpoint tensor on `dev` (CUDA) in its stored dtype (BF16/F16/F32): pread into a reused 64 MiB pinned
/// host buffer, synchronous SM-kernel copies from there.
pub(crate) fn upload_stored(idx: &ShardIndex, name: &str, dev: Device) -> Tensor {
    assert!(dev.is_cuda());
    let e=idx.entries.get(name).unwrap_or_else(||panic!("权重缺失 {name}: not found"));
    let kind=match e.dtype.as_str() {"BF16"=>Kind::BFloat16,"F16"=>Kind::Half,"F32"=>Kind::Float,
        other=>panic!("权重缺失 {name}: dtype {other} 未支持(get_f32)")};
    let shape:Vec<i64>=e.shape.iter().map(|d|*d as i64).collect();
    let nbytes=e.nbytes;
    assert_eq!(nbytes as i64,shape.iter().product::<i64>()*kind.elt_size_in_bytes() as i64,"{name} size");
    let dst=Tensor::empty(shape.as_slice(),(kind,dev));
    const STAGING:usize=64<<20;
    LOAD_BUF.with(|b|{
        let mut b=b.borrow_mut();
        // One small pinned staging buffer (pinned memory is not reclaimable: large tensors go up in 64 MiB pieces,
        // so the load's peak stays under the serving footprint); the device copy is an SM kernel.
        if b.is_none() {*b=Some(crate::expert_load::HostBuf::pinned(STAGING));}
        let buf=b.as_mut().unwrap();
        let mut off=0;
        while off<nbytes {
            let n=(nbytes-off).min(STAGING);
            let t_read=std::time::Instant::now();
            idx.read_range(name,off,&mut buf.bytes_mut()[..n]).unwrap_or_else(|err|panic!("权重缺失 {name}: {err}"));
            load_ns(4,t_read);let t_up=std::time::Instant::now();
            buf.copy_to(&dst,off,n);
            load_ns(5,t_up);
            off+=n;
        }
        idx.drop_cache(name);
        CACHE_PENDING.fetch_sub(nbytes as i64,std::sync::atomic::Ordering::Relaxed);
    });
    dst
}
/// Release the host staging buffer of upload_stored (up to one LM head of bytes).
pub(crate) fn release_load_buffer() {LOAD_BUF.with(|b|*b.borrow_mut()=None);}
fn load_slow(idx: &mut ShardIndex, name: &str, dev: Device) -> Tensor {
    let (v, s) = idx
        .get_f32(name)
        .unwrap_or_else(|e| panic!("权重缺失 {name}: {e}"));
    let shape: Vec<i64> = s.iter().map(|d| *d as i64).collect();
    Tensor::from_slice(&v)
        .to_kind(Kind::Float)
        .view(shape.as_slice())
        .to_device(dev)
}

/// 大投影权重的驻留精度开关:GLM53_W_FP16=0 → fp32(严格数值档,M0 级);
/// 默认 fp16(性能档,逐层 parity 3-5e-4,长链 greedy 可能边际翻转——归“数值精度”类分歧)。
pub fn w_fp16() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| std::env::var("GLM53_W_FP16").map(|v| v != "0").unwrap_or(true))
}

/// 大投影权重以 fp16 常驻(M1③:非专家权重读取带宽减半,36→18GiB)。
/// 小参数/路由/范数/卷积/HC 仍 fp32(精度敏感或体量无关紧要)。
/// GLM53_ABLIT=1 (default off): load-time o_proj transplant from an abliterated donor checkpoint (the published
/// "o_proj transplant" edit; files from scripts/fetch_ablit_transplant.py): layers GLM53_ABLIT_LAYERS (default 15-44; 45
/// is the checkpoint MTP block, not loaded here) take the donor's BF16 o_proj byte-for-byte from
/// GLM53_ABLIT_DIR/L{i}.bin, verified against MANIFEST.json (sha256, shape, dtype). Any mismatch or
/// missing file aborts the load: a silent fallback would serve a different model. Applied to the full
/// tensor before TP sharding / FP8 registration, so every later path sees the edited weight.
fn ablit_layers()->Option<(std::path::PathBuf,std::ops::RangeInclusive<usize>)> {
    if std::env::var("GLM53_ABLIT").as_deref()!=Ok("1") {return None;}
    let dir=std::env::var("GLM53_ABLIT_DIR").expect("GLM53_ABLIT=1 needs GLM53_ABLIT_DIR (scripts/fetch_ablit_transplant.py)");
    let spec=std::env::var("GLM53_ABLIT_LAYERS").unwrap_or_else(|_|"15-44".into());
    let (a,b)=spec.split_once('-').unwrap_or((spec.as_str(),spec.as_str()));
    let (a,b)=(a.trim().parse::<usize>().expect("GLM53_ABLIT_LAYERS"),b.trim().parse::<usize>().expect("GLM53_ABLIT_LAYERS"));
    Some((std::path::PathBuf::from(dir),a..=b))
}
/// Donor files read and hashed on background threads (GLM53_FAST_LOAD), one per layer, started with the
/// weight load; ablit_wo takes its layer's result (same bytes and digest as reading it inline).
type AblitRead=std::io::Result<(Vec<u8>,String)>;
static ABLIT_READS:std::sync::Mutex<Option<std::collections::HashMap<usize,std::thread::JoinHandle<AblitRead>>>>=std::sync::Mutex::new(None);
/// Returns the signal that all donor files have been read (None: ablit off or already started).
fn ablit_prefetch_start()->Option<std::sync::Arc<crate::safetensors::Done>> {
    let (dir,range)=ablit_layers()?;
    if ABLIT_READS.lock().unwrap().is_some() {return None;}   // already started (start_early_prefetch)
    let read=crate::safetensors::Done::new(range.clone().count());
    let reads=range.map(|layer|{let path=dir.join(format!("L{layer}.bin"));let read=read.clone();(layer,std::thread::spawn(move||{
        use sha2::{Digest,Sha256};
        let bytes=std::fs::read(&path);read.finish_one();let bytes=bytes?;
        // Read once: drop the donor's page cache right away (it would otherwise compete with the expert arenas).
        if let Ok(f)=std::fs::File::open(&path) {use std::os::unix::io::AsRawFd;extern "C"{fn posix_fadvise(fd:i32,o:i64,l:i64,a:i32)->i32;}unsafe{posix_fadvise(f.as_raw_fd(),0,0,4)};}let digest:String=Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect();Ok((bytes,digest))
    }))}).collect();
    *ABLIT_READS.lock().unwrap()=Some(reads);
    Some(read)
}
/// Page-cache warm-up of the loaded layers' non-expert tensors, in load order (see ModelWeights::load).
fn prefetch_weights(idx:&ShardIndex,n_layers:usize)->crate::safetensors::Prefetch {
    let layer_of=|n:&str|n.strip_prefix("model.language_model.layers.").and_then(|r|r.split('.').next()).and_then(|i|i.parse::<usize>().ok());
    idx.prefetch(8,|n|(n.starts_with("model.language_model.") || n=="lm_head.weight") && !n.contains(".mlp.experts.")
        && layer_of(n).map_or(true,|i|i<n_layers),|n|layer_of(n).map_or(0,|i|i as i64+1))
}
/// Bytes warmed into the page cache for the non-expert and drafter loads and not yet read by them. The expert
/// arena allocations keep this much of MemAvailable untouched (see expert_load), so the arenas never push the
/// warmed pages out before the main thread reads them (seen as ~6 GB of re-reads and a 14 s load when a restart
/// began with less free memory).
static CACHE_PENDING:std::sync::atomic::AtomicI64=std::sync::atomic::AtomicI64::new(0);
pub(crate) fn cache_pending_bytes()->i64 {CACHE_PENDING.load(std::sync::atomic::Ordering::Relaxed)}
static EARLY_PREFETCH:std::sync::Mutex<Option<(std::path::PathBuf,usize,crate::safetensors::Prefetch)>>=std::sync::Mutex::new(None);
/// GLM53_FAST_LOAD: start the non-expert prefetch and ablit donor reads at process start; ModelWeights::load
/// adopts them. Returns the prefetch's completion signal (the expert reader waits for it, so the smaller
/// critical-path non-expert bytes get the disk first).
/// `draft_dir` (the drafter checkpoint, all tensors) is warmed right after, held until finish_early_prefetch.
pub(crate) fn start_early_prefetch(dir:&Path,n_layers:usize,draft_dir:&Path)->Vec<std::sync::Arc<crate::safetensors::Done>> {
    if !fast_load_enabled() {return Vec::new();}
    let Ok(idx)=ShardIndex::scan(dir) else {return Vec::new()};
    let ablit=ablit_prefetch_start();
    let p=prefetch_weights(&idx,n_layers);let mut done=vec![p.done.clone()];done.extend(ablit);
    CACHE_PENDING.fetch_add(p.bytes as i64,std::sync::atomic::Ordering::Relaxed);
    *EARLY_PREFETCH.lock().unwrap()=Some((idx.dir.clone(),n_layers,p));
    if let Ok(d)=ShardIndex::scan(draft_dir) {
        let p=d.prefetch(4,|_|true,|_|0);done.push(p.done.clone());
        CACHE_PENDING.fetch_add(p.bytes as i64,std::sync::atomic::Ordering::Relaxed);*DRAFT_PREFETCH.lock().unwrap()=Some(p);
    }
    done
}
static DRAFT_PREFETCH:std::sync::Mutex<Option<crate::safetensors::Prefetch>>=std::sync::Mutex::new(None);
/// Join the drafter prefetch (after the drafter has loaded).
pub(crate) fn finish_early_prefetch() {DRAFT_PREFETCH.lock().unwrap().take();CACHE_PENDING.store(0,std::sync::atomic::Ordering::Relaxed);}
fn ablit_take(layer:usize)->Option<AblitRead> {
    let h=ABLIT_READS.lock().unwrap().as_mut()?.remove(&layer)?;
    Some(h.join().expect("GLM53_ABLIT prefetch thread"))
}
/// SPARK_ABLATE (crate::ablate): a projection writing into the residual stream, orthogonalized for layer `layer`.
fn ablate_out(layer:usize,w:Tensor)->Tensor {
    match crate::ablate::get() {Some(a) if a.covers(layer)=>a.ortho_out(layer,&w),_=>w}
}
fn ablit_wo(layer:usize,stock:Tensor)->Tensor {
    let t0=std::time::Instant::now();let t=ablit_wo_timed(layer,stock);load_ns(1,t0);t
}
fn ablit_wo_timed(layer:usize,stock:Tensor)->Tensor {
    use sha2::{Digest,Sha256};
    let Some((dir,range))=ablit_layers() else {return stock};
    if !range.contains(&layer) {return stock;}
    let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(dir.join("MANIFEST.json")).expect("GLM53_ABLIT: MANIFEST.json")).expect("GLM53_ABLIT: MANIFEST.json parse");
    let entry=&manifest["tensors"][layer.to_string()];
    assert!(entry.is_object(),"GLM53_ABLIT: layer {layer} missing from MANIFEST");
    assert_eq!(entry["dtype"].as_str(),Some("BF16"),"GLM53_ABLIT: layer {layer} dtype");
    let shape:Vec<i64>=entry["shape"].as_array().expect("shape").iter().map(|v|v.as_i64().unwrap()).collect();
    assert_eq!(shape,stock.size(),"GLM53_ABLIT: layer {layer} shape differs from the checkpoint o_proj");
    let (bytes,digest)=match ablit_take(layer) {
        Some(r)=>r.unwrap_or_else(|e|panic!("GLM53_ABLIT: L{layer}.bin: {e}")),
        None=>{
            let bytes=std::fs::read(dir.join(format!("L{layer}.bin"))).unwrap_or_else(|e|panic!("GLM53_ABLIT: L{layer}.bin: {e}"));
            let digest:String=Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect();(bytes,digest)
        }
    };
    assert_eq!(bytes.len() as i64,shape.iter().product::<i64>()*2,"GLM53_ABLIT: layer {layer} size");
    assert_eq!(Some(digest.as_str()),entry["sha256"].as_str(),"GLM53_ABLIT: layer {layer} sha256 mismatch");
    let donor=Tensor::from_data_size(&bytes,&shape,Kind::BFloat16).to_device(stock.device());
    let edited=donor.to_kind(stock.kind());
    let sf=stock.to_kind(Kind::Float);let rel=((&edited.to_kind(Kind::Float)-&sf).norm().double_value(&[]))/sf.norm().double_value(&[]);
    eprintln!("[ablit] layer {layer} o_proj {:?} transplanted (sha256 ok) rel_l2_vs_stock {rel:.4}",shape);
    edited
}
fn load_h(idx: &mut ShardIndex, name: &str, dev: Device) -> Tensor {
    let t = load(idx, name, dev);
    if w_fp16() { t.to_kind(Kind::Half) } else { t }
}

/// 权重自适应 GEMM:w fp16 → fp16 计算;w fp32 → fp32 直通。
pub fn mm16(x: &Tensor, w: &Tensor) -> Tensor {
    mm16_with_half(x,w,None)
}

/// A per-call, read-only view of exactly x.to_kind(Half), produced by this
/// forward's caller. Never substitute it for x before backend selection or
/// diagnostics: FP8/GEMV and full-F32 modes consume the original FP32 tensor.
pub(crate) fn mm16_with_half(x:&Tensor,w:&Tensor,half:Option<&Tensor>)->Tensor {
    if let Some(h)=half {
        assert_eq!(h.kind(),Kind::Half);assert_eq!(h.size(),x.size());assert_eq!(h.device(),x.device());
        assert!(h.is_contiguous(),"shared Half input must preserve the native matrix layout");
    }
    let y = if w.kind() == Kind::Half && crate::root_probe::retain_f32() && !crate::deep_probe::native_head(w) {
        mm16_partial(x,w)
    } else { mm16_impl(x,w,half) };
    crate::root_probe::record(x,w,&y);
    y
}

/// Private shared-GU bridge. Caller already excludes alternate backends and
/// diagnostics; preserve the original native matmul, retaining its Half result.
pub(crate) fn shared_native_half(x:&Tensor,w:&Tensor,half:Option<&Tensor>)->Tensor {
    assert!(!crate::root_probe::retain_f32() && !crate::root_probe::full_f32() && !crate::root_probe::recording());
    assert_eq!(w.kind(),Kind::Half);
    if let Some(h)=half {assert_eq!(h.kind(),Kind::Half);assert_eq!(h.size(),x.size());assert_eq!(h.device(),x.device());assert!(h.is_contiguous());}
    if let Some(y)=crate::c12::try_run(x,w,2){return y;}
    if let Some(y)=half_skinny(x,w,2){return y;}
    half.map(Tensor::shallow_clone).unwrap_or_else(||x.to_kind(Kind::Half)).matmul(&w.transpose(0,1))
}


pub(crate) fn half_skinny_enabled()->bool {std::env::var("GLM53_HALF_SKINNY").as_deref()==Ok("1")}
/// W02b: Half weights with K in {1024,4096}, 2..8 rows; out 0=FP32, 1=Half-rounded FP32, 2=Half.
pub(crate) fn half_skinny(x:&Tensor,w:&Tensor,out:i32)->Option<Tensor> {
    if !half_skinny_enabled() || crate::root_probe::retain_f32() || crate::root_probe::full_f32() || crate::root_probe::recording() {return None;}
    if !x.device().is_cuda() || x.dim()!=2 || !(2..=8).contains(&x.size()[0]) || w.kind()!=Kind::Half || w.dim()!=2 || !w.is_contiguous() {return None;}
    let (m,k,n)=(x.size()[0],x.size()[1],w.size()[0]);
    let k1536=std::env::var("GLM53_HALF_SKINNY_K1536").as_deref()==Ok("1");
    if w.size()[1]!=k || !([1024,4096].contains(&k) || (k1536&&k==1536)) || n%16!=0 || n>16384 {return None;}
    let x=if x.kind()==Kind::Float && x.is_contiguous() {x.shallow_clone()} else if x.kind()==Kind::Float || x.kind()==Kind::Half {x.to_kind(Kind::Float).contiguous()} else {return None;};
    let y=Tensor::empty([m,n],(if out==2{Kind::Half}else{Kind::Float},x.device()));
    let ks=if n<=2048{8}else{4};
    extern "C"{fn rs_half_skinny(x:*const f32,w:*const std::ffi::c_void,y:*mut std::ffi::c_void,m:i32,n:i32,k:i32,out:i32,ks:i32)->i32;}
    assert_eq!(unsafe{rs_half_skinny(x.data_ptr().cast(),w.data_ptr(),y.data_ptr(),m as i32,n as i32,k as i32,out,ks)},0,"half skinny");
    Some(y)
}
/// P1c (GLM53_HALF_INPUT_CACHE=1): prefill-sized inputs feed several Half GEMMs; keep
/// one exact x.to_kind(Half) conversion. The entry holds a shallow clone of x (its storage
/// cannot be recycled) and the storage version counter (any in-place write invalidates).
pub(crate) fn half_input_pub(x:&Tensor)->Tensor {half_input(x)}
fn half_input(x:&Tensor)->Tensor {
    thread_local!{static LAST:std::cell::RefCell<Option<(Tensor,i64,Tensor)>>=const{std::cell::RefCell::new(None)};}
    if std::env::var("GLM53_HALF_INPUT_CACHE").as_deref()!=Ok("1") || x.dim()!=2 || x.size()[0]<64 || !x.device().is_cuda() {return x.to_kind(Kind::Half);}
    extern "C"{fn rs_tensor_version(t:*const std::ffi::c_void)->i64;}
    let version=unsafe{rs_tensor_version(x.as_ptr().cast())};
    LAST.with(|l|{
        let mut l=l.borrow_mut();
        if let Some((src,v,h))=l.as_ref() {
            if src.data_ptr()==x.data_ptr() && src.size()==x.size() && src.stride()==x.stride() && src.kind()==x.kind() && *v==version {return h.shallow_clone();}
        }
        let h=x.to_kind(Kind::Half);*l=Some((x.shallow_clone(),version,h.shallow_clone()));h
    })
}
fn mm16_impl(x: &Tensor, w: &Tensor, half:Option<&Tensor>) -> Tensor {
    if let Some(y)=crate::dense_fp8::try_big(x,w,false,half){return y;}
    if let Some(y)=crate::dense_fp8::try_run(x,w,false){return y;}
    if let Some(y)=crate::c12::try_run(x,w,1){return y;}
    if let Some(y)=crate::c12::try_big(x,w,false,half){return y;}
    if let Some(a)=crate::dense_lt::eligible(x,w,false){return crate::dense_lt::run(x,w,false,a);}
    if crate::gemv::small_eligible(x,w) {return crate::gemv::small(x,w,true);}
    if crate::gemv::eligible(x,w) {return crate::gemv::run(x,w,true);}
    if let Some(y)=half_skinny(x,w,1){return y;}
    assert!(!w.stride().iter().all(|&s|s==0),"FP8-freed weight reached a non-FP8 path");
    if w.kind() == Kind::Half {
        let xh=half.map(Tensor::shallow_clone).unwrap_or_else(||half_input(x));
        rowsplit_check("mm16",&xh,w);
        xh.matmul(&w.transpose(0, 1))
            .to_kind(Kind::Float)
    } else {
        x.matmul(&w.transpose(0, 1))
    }
}

fn conv_w(idx: &mut ShardIndex, name: &str, dev: Device) -> Tensor {
    // conv1d 权重 [C,1,K] → 去掉 kernel 维前的 1 → [C,K]
    load(idx, name, dev).squeeze_dim(1)
}

/// Multimodal rows of the next prefill chunk (serve): the chunk positions holding salted placeholders
/// (both ranks) and, on the rank that encoded them, their vision embeddings (FP32 [k,4096]).
pub struct MmInject {pub rows:Tensor,pub values:Option<Tensor>}
thread_local!{static PREFILL_MM:std::cell::RefCell<Option<MmInject>>=const{std::cell::RefCell::new(None)};}
/// Arm the injection for the next prefill embedding (consumed by `embed_prefill`).
pub fn set_prefill_mm(x:Option<MmInject>) {PREFILL_MM.with(|p|*p.borrow_mut()=x);}

impl ModelWeights {
    /// Prefill embedding: `embed_tokens`, or with an armed multimodal injection, the placeholder rows zeroed
    /// on every rank and filled with the vision rows before the embedding allreduce (x + 0: exact), so both
    /// ranks receive them without another collective.
    pub fn embed_prefill(&self,ids:&Tensor)->Tensor {
        let Some(inj)=PREFILL_MM.with(|p|p.borrow_mut().take()) else {return self.embed_tokens(ids)};
        let real=ids.index_fill(0,&inj.rows,crate::vision::IMAGE_TOKEN);
        let fill=|y:&mut Tensor|{let _=y.index_fill_(0,&inj.rows,0.);if let Some(v)=&inj.values {let _=y.index_copy_(0,&inj.rows,v);}};
        if let Some((start,total))=self.vocab_shard {
            let rows=self.embed.size()[0];
            let local=(&real-start).clamp(0,rows-1);
            let invalid=real.lt(start).logical_or(&real.ge(start+rows));
            let mut y=self.embed.index_select(0,&local).masked_fill(&invalid.unsqueeze(1),0.);
            let _=Tensor::zeros([total],(Kind::Int64,real.device())).index_select(0,&real);
            fill(&mut y);
            crate::tp::allreduce(&y);y
        } else {
            let mut y=self.embed.index_select(0,&real);
            if crate::tp::world().world>1 {
                // Replicated table: only the vision rows need the other rank's contribution.
                let k=inj.rows.size()[0];
                let v=inj.values.as_ref().map(|v|v.copy()).unwrap_or_else(||Tensor::zeros([k,y.size()[1]],(y.kind(),y.device())));
                crate::tp::allreduce(&v);let _=y.index_copy_(0,&inj.rows,&v);
            } else {fill(&mut y);}
            y
        }
    }
    pub fn embed_tokens(&self,ids:&Tensor)->Tensor {
        if let Some((start,total))=self.vocab_shard {
            let rows=self.embed.size()[0];
            let local=(ids-start).clamp(0,rows-1);
            let invalid=ids.lt(start).logical_or(&ids.ge(start+rows));
            let y=self.embed.index_select(0,&local).masked_fill(&invalid.unsqueeze(1),0.);
            // A shard must not turn an invalid global token into a valid zero row.
            // index_select on this tiny lookup preserves the normal GPU bounds check.
            let _=Tensor::zeros([total],(Kind::Int64,ids.device())).index_select(0,ids);
            crate::tp::allreduce(&y);y
        } else {self.embed.index_select(0,ids)}
    }
/// This rank's head GEMM. Proposal 3 (GLM53_VERIFY_INVARIANT=1): verify batches over 8 rows run in 8-row pieces so
    /// every row takes the <=8-row skinny arithmetic (the 9..16-row tile gives per-row different rounding; greedy argmax
    /// rarely notices, T>0 Gumbel sampling at near-ties does).
    fn head_local(&self,x:&Tensor)->Tensor {
        let m=x.size()[0];
        if crate::forward::verify_invariant() && (9..=64).contains(&m) {
            return Tensor::cat(&crate::forward::invariant_pieces(m).into_iter().map(|(r,n)|mm16(&x.narrow(0,r,n),&self.lm_head)).collect::<Vec<_>>(),0);
        }
        mm16(x,&self.lm_head)
    }
    pub fn logits(&self,x:&Tensor)->Tensor {
        let local=self.head_local(x);
        if let Some((start,total))=self.vocab_shard {
            let y=Tensor::zeros([x.size()[0],total],(local.kind(),local.device()));
            y.narrow(1,start,self.lm_head.size()[0]).copy_(&local);crate::tp::allreduce(&y);y
        } else {local}
    }
    /// Exact verifier top-1. Keep the local head GEMM and its precision intact;
    /// the opt-in path replaces only full-vocabulary assembly/communication.
    pub(crate) fn predictions(&self,x:&Tensor)->Tensor {
        // With serving sampling enabled, rows carry Gumbel noise (zeros for greedy rows).
        if !crate::head_select::enabled() { return crate::sampling::perturb(&self.logits(x)).argmax(-1,false); }
        if let Some((start,total))=self.vocab_shard {
            if let Some((t,k,nstart,nwidth))=crate::sampling::fused_params() {
                let local=self.head_local(x);assert_eq!((nstart,nwidth),(start,local.size()[1]),"noise columns = head slice");
                return crate::head_select::from_local_gumbel(&local.contiguous(),start,total,&t,&k);
            }
            crate::head_select::from_local(&crate::sampling::perturb(&self.head_local(x)),start,total)
        } else { crate::sampling::perturb(&self.logits(x)).argmax(-1,false) }
    }
    /// Drafter uses a replicated BF16 vocabulary head. Gather only at load time.
    pub fn drafter_head(&self)->Tensor {
        if let Some((start,total))=self.vocab_shard {
            let full=Tensor::zeros([total,self.lm_head.size()[1]],(self.lm_head.kind(),self.device));
            full.narrow(0,start,self.lm_head.size()[0]).copy_(&self.lm_head);crate::tp::allreduce(&full);full
        }else{self.lm_head.shallow_clone()}
    }
    pub fn load(dir: &Path, cfg: &Config, n_layers: usize, dev: Device) -> Self {
        let disk0=crate::host_memory::thread_read_bytes();
        let t_scan=std::time::Instant::now();
        let mut idx = ShardIndex::scan(dir).expect("scan");load_ns(3,t_scan);
        // GLM53_FAST_LOAD: warm the non-expert tensors of the loaded layers in load order, read ablit donors.
        let prefetch=fast_load_enabled().then(||{
            let early=EARLY_PREFETCH.lock().unwrap().take().filter(|(d,n,_)|*d==idx.dir && *n==n_layers);
            early.map(|(_,_,p)|p).unwrap_or_else(||{let _=ablit_prefetch_start();prefetch_weights(&idx,n_layers)})
        });
        let p = "model.language_model";
        let embed = load(&mut idx, &format!("{p}.embed_tokens.weight"), dev);
        let embed = match crate::ablate::get() { Some(a) => a.ortho_rows(0, &embed), None => embed };
        let final_norm = load(&mut idx, &format!("{p}.norm.weight"), dev);
        let lm_head = load_h(&mut idx, "lm_head.weight", dev);
        let mut layers = Vec::with_capacity(n_layers);
        for plan in cfg.plans.iter().take(n_layers) {
            let i = plan.idx;
            let l = format!("{p}.layers.{i}");
            let hc = HcParams {
                attn_fn: load_bf16_resident(&mut idx, &format!("{l}.hc_attn_fn"), dev),
                attn_scale: load(&mut idx, &format!("{l}.hc_attn_scale"), dev),
                attn_base: load(&mut idx, &format!("{l}.hc_attn_base"), dev),
                ffn_fn: load_bf16_resident(&mut idx, &format!("{l}.hc_ffn_fn"), dev),
                ffn_scale: load(&mut idx, &format!("{l}.hc_ffn_scale"), dev),
                ffn_base: load(&mut idx, &format!("{l}.hc_ffn_base"), dev),
                in_ln: load(&mut idx, &format!("{l}.input_layernorm.weight"), dev),
                post_ln: load(&mut idx, &format!("{l}.post_attention_layernorm.weight"), dev),
            };
            let sa = format!("{l}.self_attn");
            let (kda, mla) = if plan.attn == "kda" {
                (
                    Some(KdaWeights {
                        constants:None,
                        wq: load_h(&mut idx, &format!("{sa}.q_proj.weight"), dev),
                        wk: load_h(&mut idx, &format!("{sa}.k_proj.weight"), dev),
                        wv: load_h(&mut idx, &format!("{sa}.v_proj.weight"), dev),
                        wo: ablate_out(i, ablit_wo(i, load_h(&mut idx, &format!("{sa}.o_proj.weight"), dev))),
                        wb: load(&mut idx, &format!("{sa}.b_proj.weight"), dev),
                        fa: load_h(&mut idx, &format!("{sa}.f_a_proj.weight"), dev),
                        fb: load_h(&mut idx, &format!("{sa}.f_b_proj.weight"), dev),
                        ga: load_h(&mut idx, &format!("{sa}.g_a_proj.weight"), dev),
                        gb: load_h(&mut idx, &format!("{sa}.g_b_proj.weight"), dev),
                        conv_q: conv_w(&mut idx, &format!("{sa}.q_conv1d.weight"), dev),
                        conv_k: conv_w(&mut idx, &format!("{sa}.k_conv1d.weight"), dev),
                        conv_v: conv_w(&mut idx, &format!("{sa}.v_conv1d.weight"), dev),
                        dt_bias: load(&mut idx, &format!("{sa}.dt_bias"), dev).view([64, 128]),
                        a_log: load(&mut idx, &format!("{sa}.A_log"), dev),
                        o_norm: load(&mut idx, &format!("{sa}.o_norm.weight"), dev),
                    }),
                    None,
                )
            } else {
                let qp = format!("{l}.self_attn");
                (
                    None,
                    Some(MlaWeights {
                        latent_cache:RefCell::new(None),
                        indexer: if crate::mla_latent::enabled() {
                            let p=format!("{qp}.indexer");
                            Some(crate::dsa::Weights{
                                q:load_index_bf16(&mut idx,&format!("{p}.wq_b.weight"),dev),
                                k:load_index_bf16(&mut idx,&format!("{p}.wk.weight"),dev),
                                norm_w:load(&mut idx,&format!("{p}.k_norm.weight"),dev),
                                norm_b:load(&mut idx,&format!("{p}.k_norm.bias"),dev),
                                score:load_index_bf16(&mut idx,&format!("{p}.weights_proj.weight"),dev),
                                ape:load(&mut idx,&format!("{p}.index_kpool_compress_ape"),dev),
                                gate:load_index_bf16(&mut idx,&format!("{p}.index_kpool_compress_gate"),dev),
                            })
                        } else {None},
                        q_a: load_h(&mut idx, &format!("{qp}.q_a_proj.weight"), dev),
                        q_b: load_h(&mut idx, &format!("{qp}.q_b_proj.weight"), dev),
                        kv_a: load_h(&mut idx, &format!("{qp}.kv_a_proj_with_mqa.weight"), dev),
                        kv_b: load_h(&mut idx, &format!("{qp}.kv_b_proj.weight"), dev),
                        wo: ablate_out(i, ablit_wo(i, load_h(&mut idx, &format!("{qp}.o_proj.weight"), dev))),
                        q_a_ln: load(&mut idx, &format!("{qp}.q_a_layernorm.weight"), dev),
                        kv_a_ln: load(&mut idx, &format!("{qp}.kv_a_layernorm.weight"), dev),
                    }),
                )
            };
            let (dense, moe) = if plan.mlp == "dense" {
                let m = format!("{l}.mlp");
                (
                    Some(DenseMlp {
                        wg: load_h(&mut idx, &format!("{m}.gate_proj.weight"), dev),
                        wu: load_h(&mut idx, &format!("{m}.up_proj.weight"), dev),
                        wd: ablate_out(i, load_h(&mut idx, &format!("{m}.down_proj.weight"), dev)),
                    }),
                    None,
                )
            } else {
                let m = format!("{l}.mlp");
                (
                    None,
                    Some(MoeMeta {
                        w_gate: load_bf16_resident(&mut idx, &format!("{m}.gate.weight"), dev),
                        bias: load(&mut idx, &format!("{m}.gate.e_score_correction_bias"), dev),
                        sh_wg: load_h(&mut idx, &format!("{m}.shared_experts.gate_proj.weight"), dev),
                        sh_wu: load_h(&mut idx, &format!("{m}.shared_experts.up_proj.weight"), dev),
                        sh_wd: ablate_out(i, load_h(&mut idx, &format!("{m}.shared_experts.down_proj.weight"), dev)),
                    }),
                )
            };
            let mut layer = LayerWeights {
                plan: LayerPlanOwned {
                    attn: plan.attn.to_string(),
                    mlp: plan.mlp.to_string(),
                },
                hc,
                kda,
                mla,
                dense,
                moe,
            };
            let t_prep=std::time::Instant::now();
            if crate::tp::dense_enabled() {
                let tp = crate::tp::world();
                shard_dense_layer(&mut layer, tp.rank, tp.world);
            }
            if let Some(w)=layer.kda.as_mut(){w.prepare_constants();}
            if crate::mla_latent::enabled()&&std::env::var("GLM53_MLA_WEIGHT_CACHE").as_deref()==Ok("1") {
                if let Some(w)=layer.mla.as_ref(){let _=w.latent_projections();}
            }
            load_ns(2,t_prep);
            layers.push(layer);
        }
        let ns=|i:usize|LOAD_NS[i].load(std::sync::atomic::Ordering::Relaxed) as f64*1e-9;
        eprintln!("[load] model weights: index scan {:.1}s, tensor loads {:.1}s (pread {:.1}s, upload {:.1}s; {:.1} GB, main thread disk reads {:.1} GB, ablit excluded), ablit {:.1}s, shard/prepare {:.1}s",
            ns(3),ns(0),ns(4),ns(5),LOAD_BYTES.load(std::sync::atomic::Ordering::Relaxed) as f64*1e-9,(crate::host_memory::thread_read_bytes()-disk0) as f64*1e-9,ns(1),ns(2));
        if let Some(p)=prefetch {eprintln!("[load] model weights: prefetched {:.1} GB",p.bytes as f64*1e-9);}
        *ABLIT_READS.lock().unwrap()=None;
        release_load_buffer();
        crate::host_memory::stage("model weights: tensors resident");
        let tp=crate::tp::world();
        let vocab_shard=if tp.world>1&&std::env::var("GLM53_VOCAB_TP").as_deref()==Ok("1") {
            assert_eq!(embed.size()[0],lm_head.size()[0]);assert_eq!(embed.size()[0]%tp.world as i64,0);
            Some((embed.size()[0]/tp.world as i64*tp.rank as i64,embed.size()[0]))
        }else{None};
        let (embed,lm_head)=if vocab_shard.is_some(){(shard(&embed,0,tp.rank,tp.world),shard(&lm_head,0,tp.rank,tp.world))}else{(embed,lm_head)};
        crate::dense_fp8::register(&layers);
        if crate::dense_fp8::free_sources_enabled() {
            let mut freed=0i64;
            let mut take=|t:&mut Tensor|{if let Some(ph)=crate::dense_fp8::compact(t){freed+=t.numel() as i64*2;*t=ph;}};
            for layer in &mut layers {
                if let Some(k)=layer.kda.as_mut(){for t in [&mut k.wq,&mut k.wk,&mut k.wv,&mut k.wo]{take(t);}}
                if let Some(d)=layer.dense.as_mut(){for t in [&mut d.wg,&mut d.wu,&mut d.wd]{take(t);}}
                if let Some(m)=layer.moe.as_mut(){for t in [&mut m.sh_wg,&mut m.sh_wu,&mut m.sh_wd]{take(t);}}
                if let Some(m)=layer.mla.as_mut(){for t in [&mut m.q_a,&mut m.q_b,&mut m.kv_a,&mut m.wo]{take(t);}}
            }
            eprintln!("[fp8-free-source] released {:.2} GiB of Half sources",freed as f64/(1u64<<30) as f64);
        }
        crate::dense_fp8::register_weight(&lm_head,"GLM53_FP8_HEAD");
        // After every FP8 registration/compaction: C12 codes only the weights that stay at source precision.
        crate::c12::register(&layers,&lm_head);
        if crate::c12::free_sources_enabled() {
            // The LM head keeps its Half source (the drafter's full head is assembled from it).
            let mut freed=0i64;
            let mut take=|t:&mut Tensor|{if let Some(ph)=crate::c12::compact(t){freed+=t.numel() as i64*2;*t=ph;}};
            for layer in &mut layers {
                if let Some(k)=layer.kda.as_mut(){for t in [&mut k.wq,&mut k.wk,&mut k.wv,&mut k.wo]{take(t);}}
                if let Some(d)=layer.dense.as_mut(){for t in [&mut d.wg,&mut d.wu,&mut d.wd]{take(t);}}
                if let Some(m)=layer.moe.as_mut(){for t in [&mut m.sh_wg,&mut m.sh_wu,&mut m.sh_wd]{take(t);}}
                if let Some(m)=layer.mla.as_mut(){for t in [&mut m.q_a,&mut m.q_b,&mut m.kv_a,&mut m.wo]{take(t);}}
            }
            crate::c12::finish_compact();
            eprintln!("[c12-free-source] released {:.2} GiB of Half sources",freed as f64/(1u64<<30) as f64);
        }
        Self { embed, final_norm, lm_head, layers, device: dev, vocab_shard }
    }
}

/// Materialize the local shard; a narrow view alone would retain the full allocation.
fn shard(t: &Tensor, dim: i64, rank: usize, world: usize) -> Tensor {
    assert!(world > 0 && rank < world);
    assert_eq!(t.size()[dim as usize] % world as i64, 0);
    let width = t.size()[dim as usize] / world as i64;
    t.narrow(dim, rank as i64 * width, width).copy()
}

/// Head-parallel attention and intermediate-parallel MLP; residual stays replicated.
/// q_a/kv_a, low-rank f_a/g_a, norms, router, embed and lm_head stay replicated.
pub fn shard_dense_layer(layer: &mut LayerWeights, rank: usize, world: usize) {
    if let Some(w) = layer.kda.as_mut() {
        w.constants=None;
        for t in [&mut w.wq, &mut w.wk, &mut w.wv, &mut w.wb,
                  &mut w.fb, &mut w.gb, &mut w.conv_q, &mut w.conv_k,
                  &mut w.conv_v, &mut w.dt_bias, &mut w.a_log] {
            *t = shard(t, 0, rank, world);
        }
        w.wo = shard(&w.wo, 1, rank, world);
    }
    if let Some(w) = layer.mla.as_mut() {
        w.invalidate_latent_cache();
        w.q_b = shard(&w.q_b, 0, rank, world);
        // kv_b rows are interleaved [head, k|v], so contiguous head shards preserve layout.
        w.kv_b = shard(&w.kv_b, 0, rank, world);
        w.wo = shard(&w.wo, 1, rank, world);
    }
    if let Some(w) = layer.dense.as_mut() {
        w.wg = shard(&w.wg, 0, rank, world);
        w.wu = shard(&w.wu, 0, rank, world);
        w.wd = shard(&w.wd, 1, rank, world);
    }
    if let Some(w) = layer.moe.as_mut() {
        w.sh_wg = shard(&w.sh_wg, 0, rank, world);
        w.sh_wu = shard(&w.sh_wu, 0, rank, world);
        w.sh_wd = shard(&w.sh_wd, 1, rank, world);
    }
}

extern "C" {
    fn rs_mm16_f32(x: *const u8, w: *const u8, y: *mut u8, m: i32, n: i32, k: i32) -> i32;
}

/// FP16 inputs and weights, FP32 output: do not round each rank's partial sum.
/// mm16_partial written into `out` (contiguous FP32 [rows, N]): the C12 kernel writes it directly; any other path
/// computes mm16_partial and copies (same values).
pub fn mm16_partial_into(x:&Tensor,w:&Tensor,out:&Tensor) {
    assert!(out.is_contiguous() && out.kind()==Kind::Float && out.size()==[x.size()[0],w.size()[0]]);
    if !crate::root_probe::full_f32() && w.kind()==Kind::Half && !crate::dense_fp8::registered_enabled(w) && crate::c12::try_run_into(x,w,0,out) {return;}
    let _=out.shallow_clone().copy_(&mm16_partial(x,w));
}
pub fn mm16_partial(x: &Tensor, w: &Tensor) -> Tensor {
    if crate::root_probe::full_f32() {
        let x = if crate::root_probe::round_f32_input() { x.to_kind(Kind::Half).to_kind(Kind::Float) } else { x.to_kind(Kind::Float) };
        return x.matmul(&crate::deep_probe::float_weight(w).transpose(0,1));
    }
    if w.kind() != Kind::Half { return mm16(x,w); }
    if let Some(y)=crate::dense_fp8::try_big(x,w,true,None){return y;}
    if let Some(y)=crate::dense_fp8::try_run(x,w,true){return y;}
    if let Some(y)=crate::c12::try_run(x,w,0){return y;}
    if let Some(y)=crate::c12::try_big(x,w,true,None){return y;}
    if let Some(a)=crate::dense_lt::eligible(x,w,true){return crate::dense_lt::run(x,w,true,a);}
    if crate::gemv::small_eligible(x,w) {return crate::gemv::small(x,w,false);}
    if crate::gemv::eligible(x,w) {return crate::gemv::run(x,w,false);}
    if let Some(y)=half_skinny(x,w,0){return y;}
    assert!(!w.stride().iter().all(|&s|s==0),"FP8-freed weight reached a non-FP8 path");
    let x = x.to_kind(Kind::Half).contiguous();
    let w = w.contiguous();
    if x.device() == tch::Device::Cpu {
        return x.to_kind(Kind::Float).matmul(&w.to_kind(Kind::Float).transpose(0,1));
    }
    let y = Tensor::empty([x.size()[0],w.size()[0]],(Kind::Float,x.device()));
    let rc = unsafe { rs_mm16_f32(x.data_ptr().cast(),w.data_ptr().cast(),y.data_ptr().cast(),
        x.size()[0] as i32,w.size()[0] as i32,w.size()[1] as i32) };
    assert_eq!(rc,0,"row-parallel GEMM failed");
    if rowsplit_on() && x.size()[0]>=256 && x.size()[0]%2==0 && !crate::tp::graph::capturing() {
        let h=x.size()[0]/2;let part=|r:i64|{let xs=x.narrow(0,r,h).contiguous();let ys=Tensor::empty([h,w.size()[0]],(Kind::Float,x.device()));
            assert_eq!(unsafe{rs_mm16_f32(xs.data_ptr().cast(),w.data_ptr().cast(),ys.data_ptr().cast(),h as i32,w.size()[0] as i32,w.size()[1] as i32)},0);ys};
        rowsplit_report("mm16_f32",&y,&Tensor::cat(&[part(0),part(h)],0),x.size()[0],&w);
    }
    y
}
/// GLM53_ROWSPLIT_CHECK=1 (diagnostic): every prefill Half GEMM is recomputed as two row halves and compared bitwise,
/// to qualify row-split GEMMs (communication overlap) as L0.
fn rowsplit_on()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_ROWSPLIT_CHECK").as_deref()==Ok("1"))}
fn rowsplit_check(name:&str,xh:&Tensor,w:&Tensor) {
    if !rowsplit_on() || xh.dim()!=2 || xh.size()[0]<256 || xh.size()[0]%2!=0 || crate::tp::graph::capturing() {return;}
    let h=xh.size()[0]/2;let wt=w.transpose(0,1);
    let full=xh.matmul(&wt);let split=Tensor::cat(&[xh.narrow(0,0,h).matmul(&wt),xh.narrow(0,h,h).matmul(&wt)],0);
    rowsplit_report(name,&full,&split,xh.size()[0],w);
}
fn rowsplit_report(name:&str,a:&Tensor,b:&Tensor,rows:i64,w:&Tensor) {
    use std::sync::atomic::{AtomicU64,Ordering::Relaxed};static OK:AtomicU64=AtomicU64::new(0);static BAD:AtomicU64=AtomicU64::new(0);
    let same=a.eq_tensor(b).all().int64_value(&[])==1;
    if same {let o=OK.fetch_add(1,Relaxed)+1;if o%200==0{eprintln!("[rowsplit] ok {o} bad {}",BAD.load(Relaxed));}}
    else {let n=BAD.fetch_add(1,Relaxed)+1;if n<=40{eprintln!("[rowsplit] MISMATCH {name} rows {rows} w {:?} max_abs {:e}",w.size(),(a.to_kind(Kind::Float)-b.to_kind(Kind::Float)).abs().max().double_value(&[]));}}
}

thread_local!{static SP_SCATTER:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};}
/// Sequence-parallel prefill: inside `f`, row-parallel projections reduce-scatter their FP32
/// partials (this rank keeps its half of the rows) instead of all-reducing them.
pub(crate) fn with_sp_scatter<R>(f:impl FnOnce()->R)->R {
    SP_SCATTER.with(|c|c.set(true));let r=f();SP_SCATTER.with(|c|c.set(false));r
}
/// Match unsharded output precision: reduce fp32 partials, then round once.
/// P1a-Half (GLM53_PREFILL_QKV_INTO=1): prefill projections that take mm16's Half-GEMM path (retained Half
/// weights, GLM53_FP8_PREFILL_HALF) write their Half result, widened, straight into column slices of one
/// [rows, sum N] FP32 output: the same values as cat(mm16(x,w_i)) without the per-projection FP32
/// intermediates and the cat. Returns None when any projection would take another backend.
pub(crate) fn mm16_cat_into(x:&Tensor,ws:&[&Tensor])->Option<Tensor> {
    if std::env::var("GLM53_PREFILL_QKV_INTO").as_deref()!=Ok("1") {return None;}
    if !x.device().is_cuda() || x.dim()!=2 || x.size()[0]<=128 || crate::root_probe::retain_f32() || crate::root_probe::full_f32() || crate::root_probe::recording() {return None;}
    if let Some(y)=crate::dense_fp8::try_big_cat(x,ws){return Some(y);}
    if let Some(y)=crate::c12::try_big_cat(x,ws){return Some(y);}
    if std::env::var("GLM53_FP8_PREFILL_HALF").as_deref()!=Ok("1") || std::env::var("GLM53_DENSE_LT").as_deref()==Ok("1") {return None;}
    for w in ws {if w.kind()!=Kind::Half || w.dim()!=2 || !w.is_contiguous() || w.stride().iter().all(|&s|s==0) || w.size()[1]!=x.size()[1] {return None;}}
    // Same backend decision as mm16_impl for these rows: dense_fp8::try_run declines (> 128 rows, retained Half
    // weights, GLM53_FP8_PREFILL_HALF), DENSE_LT is off, GEMV/skinny only serve <= 16 rows.
    if crate::gemv::small_eligible(x,ws[0]) || crate::gemv::eligible(x,ws[0]) {return None;}
    let rows=x.size()[0];let total:i64=ws.iter().map(|w|w.size()[0]).sum();
    let out=Tensor::empty([rows,total],(Kind::Float,x.device()));
    let xh=half_input(x);let mut off=0;
    for w in ws {let n=w.size()[0];let _=out.narrow(1,off,n).copy_(&xh.matmul(&w.transpose(0,1)));off+=n;}
    Some(out)
}
/// GLM53_PREFILL_HALF_GLUE=1 (L0): prefill glue without FP32 round trips of Half GEMM results.
pub(crate) fn half_glue_enabled()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_PREFILL_HALF_GLUE").as_deref()==Ok("1"))}
/// mm16(x, w) for prefill rows would take mm16_impl's plain Half matmul (the conditions mm16_cat_into states).
pub(crate) fn plain_half_prefill(x:&Tensor,w:&Tensor)->bool {
    x.device().is_cuda() && x.dim()==2 && x.size()[0]>128 && !crate::root_probe::retain_f32() && !crate::root_probe::full_f32() && !crate::root_probe::recording()
        && std::env::var("GLM53_FP8_PREFILL_HALF").as_deref()==Ok("1") && std::env::var("GLM53_DENSE_LT").as_deref()!=Ok("1")
        && w.kind()==Kind::Half && w.dim()==2 && w.is_contiguous() && !w.stride().iter().all(|&s|s==0) && w.size()[1]==x.size()[1]
        && !crate::dense_fp8::big_eligible(x,w) && !crate::c12::big_eligible(x,w) && !crate::gemv::small_eligible(x,w) && !crate::gemv::eligible(x,w)
}
/// mm16_cat_into's projections as separate Half results (the values its FP32 slices hold, before widening).
pub(crate) fn mm16_half_parts(x:&Tensor,ws:&[&Tensor])->Option<Vec<Tensor>> {
    if !half_glue_enabled() || std::env::var("GLM53_PREFILL_QKV_INTO").as_deref()!=Ok("1") || !ws.iter().all(|w|plain_half_prefill(x,w)) {return None;}
    let xh=half_input(x);Some(ws.iter().map(|w|xh.matmul(&w.transpose(0,1))).collect())
}
/// mm16(&mm16(x,w1),w2) as Half: the inner result's FP32 widening and Half narrowing are exact, so both are skipped.
pub(crate) fn mm16_chain_half(x:&Tensor,w1:&Tensor,w2:&Tensor)->Option<Tensor> {
    if !half_glue_enabled() || !plain_half_prefill(x,w1) {return None;}
    let y1=half_input(x).matmul(&w1.transpose(0,1));
    if !plain_half_prefill(&y1,w2) {return None;}
    Some(y1.matmul(&w2.transpose(0,1)))
}
pub fn row_mm16(x: &Tensor, w: &Tensor) -> Tensor {
    if SP_SCATTER.with(|c|c.get()) && crate::tp::dense_enabled() {
        if let Some(y)=row_mm16_rs_split(x,w) {return round_row_output(y,w);}
        return round_row_output(crate::tp::reduce_scatter_rows(&mm16_partial(x,w).contiguous()),w);
    }
    if !crate::tp::dense_enabled() { return mm16(x,w); }
    let y = mm16_partial(x,w);
    if defer_row_sum(&y,w) { return y; }
    round_row_output(crate::tp::dense_sum(y),w)
}
/// GLM53_PREFILL_SP_RS_SPLIT=n (2 or 4; L1: each column part is its own GEMM): the SP row-parallel projection in n
/// column parts, each part's reduce-scatter issued asynchronously so it overlaps the next part's GEMM; the parts are
/// concatenated afterwards (same 2-rank sums). Half weights on the cuBLAS prefill path only.
fn row_mm16_rs_split(x:&Tensor,w:&Tensor)->Option<Tensor> {
    static N:std::sync::OnceLock<i64>=std::sync::OnceLock::new();
    let parts=*N.get_or_init(||std::env::var("GLM53_PREFILL_SP_RS_SPLIT").ok().and_then(|v|v.parse().ok()).filter(|&n:&i64|n==2||n==4).unwrap_or(0));
    if parts==0 || w.kind()!=Kind::Half || w.dim()!=2 || !w.is_contiguous() || w.stride().iter().all(|&s|s==0) || x.size()[0]<256 || x.size()[0]%2!=0 {return None;}
    let n=w.size()[0];if n%(parts*64)!=0 {return None;}
    let step=n/parts;let rows=x.size()[0];
    extern "C"{fn rs_reduce_scatter_async(i:*mut u8,o:*mut u8,n:i64,dtype:i32)->i32;fn rs_work_wait(h:i32)->i32;}
    let mut keep=Vec::new();let mut outs=Vec::new();let mut handles=Vec::new();
    for j in 0..parts {
        let y=mm16_partial(x,&w.narrow(0,j*step,step)).contiguous();assert_eq!(y.size(),[rows,step]);
        let o=Tensor::empty([rows/2,step],(Kind::Float,x.device()));
        crate::tp::settle_pending_pub();
        let h=unsafe{rs_reduce_scatter_async(y.data_ptr() as *mut u8,o.data_ptr() as *mut u8,o.numel() as i64,2)};assert!(h>=0,"async reduce_scatter");
        keep.push(y);outs.push(o);handles.push(h);
    }
    for h in handles {assert_eq!(unsafe{rs_work_wait(h)},0,"reduce_scatter wait");}
    drop(keep);
    Some(Tensor::cat(&outs,1))
}
/// I3 step 2: inside `tp::with_deferred_sum`, leave the TP sum and round_row_output's rounding to the consumer
/// (only where round_row_output would use the in-place Half round or no rounding). true: `y` is a pending partial.
fn defer_row_sum(y:&Tensor,w:&Tensor)->bool {
    let round=w.kind()==Kind::Half && !crate::root_probe::retain_f32();
    if round && !(crate::kda::norm_fused_enabled() && y.kind()==Kind::Float && y.is_contiguous() && y.device().is_cuda()) { return false; }
    crate::tp::defer_sum(y,round)
}

pub(crate) fn round_row_output(y:Tensor,w:&Tensor)->Tensor {
    if w.kind() == Kind::Half && !crate::root_probe::retain_f32() {
        if crate::kda::norm_fused_enabled() && y.kind()==Kind::Float && y.is_contiguous() && y.device().is_cuda() {
            extern "C"{fn rs_round_half_inplace(y:*mut f32,n:i64)->i32;}
            assert_eq!(unsafe{rs_round_half_inplace(y.data_ptr().cast(),y.numel() as i64)},0);return y;
        }
        y.to_kind(Kind::Half).to_kind(Kind::Float)
    } else { y }
}
