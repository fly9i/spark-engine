// SPDX-License-Identifier: MIT
//! Persistent prefix cache on the local NVMe (GLM53_PCACHE=1, serving only). Every prompt/boundary checkpoint the
//! serve loop takes (≥ GLM53_PCACHE_MIN tokens) is written to disk right away by a background thread, so a restart,
//! a crash or an evicted store no longer loses it; a later request whose prompt extends a cached prefix restores it
//! (admission action 6) instead of prefilling it again.
//!
//! Layout (per rank, `GLM53_PCACHE_DIR/<identity>/rank<r>/`):
//! - `s-<id>.seg`: one segment = GLM53_PCACHE_SEG (4096) tokens of every MLA layer's latent rows and DSA pool rows.
//!   Segments are shared by all snapshots of one store lineage (multi-turn: turn N reuses turn N-1's segments and only
//!   writes the new ones). Sharing follows provenance, not token hashes: a store remembers which segments hold exactly
//!   its rows, so two independent prefills of the same tokens (possibly different chunk splits, L1) never mix.
//! - `e-<key>-<which>.snap`: a checkpoint: token ids, KDA h/conv, each MLA layer's len/tails/partial pool row, the
//!   drafter context, the last-row logits, and the rows after the last full segment; the header lists its segments.
//! Files are written O_DIRECT through a small pinned ring (no page cache: on GB10 it would take unified memory from
//! the GPU), as `*.tmp`, fdatasync'ed and renamed; every 16 MiB chunk carries a checksum checked on restore.
//!
//! Replication: the index (entries, segments, LRU clock, byte budget) is a replicated state machine like the
//! checkpoint library: both ranks save at the same prefill points and evict in the same order. Writes are local
//! and asynchronous; a restore waits for its own rank's write, and both ranks agree (one allreduce) on success,
//! otherwise both fall back to a cold prefill and drop the entry. At startup the two ranks exchange their valid
//! files over the doorbell TCP stream and keep the intersection (so a crash between the two ranks' writes is safe).
//!
//! Identity: the engine binary's SHA256 (strict: any rebuild starts a new cache), model and drafter snapshot paths,
//! TP world, segment size, and every numerics-relevant GLM53_* setting. Both ranks must run the same binary (the start
//! script checks it); caches of other identities are kept (a rollback finds its own) within half the cap.
//!
//! The machinery (files, writer and reader threads, index, replication) is crate::pcache_core, shared with the
//! DeepSeek-V4.1 serve loop; this module describes GLM's checkpoint to it (row arrays `lat.<l>` div 1 and `pool.<l>` div
//! 4 rounded up; fixed regions last, kda, mla, draft; header fields draft_len / draft_start) and keeps GLM's interface.

use std::path::Path;
use std::cell::RefCell;
use std::collections::HashMap;
use serde_json::{json,Value};
use tch::{Device,Tensor};
use crate::pcache_core::{self as core,Rows,Part,Snapshot};

pub(crate) fn requested()->bool {std::env::var("GLM53_PCACHE").as_deref()==Ok("1")}
fn env_u64(k:&str,d:u64)->u64 {std::env::var(k).ok().and_then(|v|v.parse().ok()).unwrap_or(d)}

thread_local!{static PC:RefCell<Option<core::Cache>>=const{RefCell::new(None)};}
pub(crate) fn enabled()->bool {PC.with(|p|p.borrow().is_some())}
/// Segment size in tokens (0 when the cache is off).
pub(crate) fn seg_tokens()->i64 {PC.with(|p|p.borrow().as_ref().map_or(0,|p|p.seg_tokens()))}

fn identity(binary:&str,model:&Path,draft:&Path,world:usize,seg:i64)->String {
    // Scheduling, memory, logging, loading and transport settings do not change the saved values.
    let skip=["GLM53_SERVE_","GLM53_MEM_","GLM53_KV_POOL","GLM53_HOST_","GLM53_RELEASE_FILE_CACHE","GLM53_NCCL_","GLM53_PCACHE","GLM53_TP_RANK",
        "GLM53_MASTER_","GLM53_ENGINE_BIN","GLM53_FAST_","GLM53_PRELOAD_","GLM53_LOAD_","GLM53_GRAPH_EMPTY_CACHE","GLM53_DRAFT_TRACE","GLM53_DRAFT_EXPORT",
        "GLM53_PREFIX_ADMISSION","GLM53_SPEC_GRAPH_SLOTS","GLM53_DEPTH_RULE","GLM53_LOG"];
    let mut env:Vec<(String,String)>=std::env::vars().filter(|(k,_)|k.starts_with("GLM53_")&&!skip.iter().any(|s|k.starts_with(s))).collect();env.sort();
    let canon=|p:&Path|std::fs::canonicalize(p).unwrap_or_else(|_|p.to_path_buf()).display().to_string();
    // SPX checkpoints (an override file in the same model directory changes the weights): the SPX files' names, sizes and
    // mtimes join the identity. Unchanged for checkpoints without SPX tensors.
    let spx=crate::spx::fingerprint(model).map(|f|format!(";spx={f}")).unwrap_or_default();
    // SPARK_ABLATE* (crate::ablate) changes every saved value: its settings and the direction file's content hash join the
    // identity. Unchanged when no SPARK_ABLATE variable is set.
    let mut abl:Vec<(String,String)>=std::env::vars().filter(|(k,_)|k.starts_with("SPARK_ABLATE")).collect();abl.sort();
    let abl=if abl.is_empty() {String::new()} else {
        // Content hash (not mtime: the two ranks' copies are written at different times and must still agree).
        let f=["SPARK_ABLATE","SPARK_ABLATE_OPROJ","SPARK_ABLATE_CLAMP"].iter().map(|v|std::env::var(v).ok().map(|f|std::fs::read(&f).map(|b|{use sha2::Digest;let d=sha2::Sha256::digest(&b);d[..16].iter().map(|x|format!("{x:02x}")).collect::<String>()}).unwrap_or_else(|_|"missing".into())).unwrap_or_default()).collect::<Vec<_>>().join("/");
        format!(";ablate={abl:?};ablate_file={f}")};
    format!("format={};binary={binary};model={};draft={};world={world};seg={seg};env={env:?}{spx}{abl}",core::FORMAT,canon(model),canon(draft))
}

/// Startup (every rank, after the doorbell connections and before serving): scan this rank's files, agree with the
/// other ranks on the common valid set and its LRU order (rank0's), delete the rest, start the writer thread.
pub(crate) fn init(rank:usize,world:usize,model:&Path,draft:&Path,links:&mut [std::net::TcpStream]) {
    let seg=env_u64("GLM53_PCACHE_SEG",4096) as i64;assert!(seg>=64&&seg%64==0,"GLM53_PCACHE_SEG must be a multiple of 64");
    let cfg=core::Config{root:std::env::var("GLM53_PCACHE_DIR").unwrap_or_else(|_|"prefix-cache".into()).into(),seg,
        cap:env_u64("GLM53_PCACHE_GIB",100)<<30,min:env_u64("GLM53_PCACHE_MIN",1024) as usize,free_margin:env_u64("GLM53_PCACHE_FREE_GIB",20)<<30,
        stage_mb:env_u64("GLM53_PCACHE_STAGE_MB",256),readers:env_u64("GLM53_PCACHE_READERS",4) as usize,
        log:std::env::var("GLM53_PCACHE_LOG").as_deref()==Ok("1"),tag:"[pcache]",
        test_delay_ms:std::env::var("GLM53_PCACHE_TEST_DELAY_MS").ok().and_then(|v|v.parse::<u64>().ok())};
    let binary=core::binary_sha256();
    let identity=identity(&binary,model,draft,world,seg);
    // rank0: one stream per worker (rank order); a worker: its stream to rank0
    let link=if world<=1 {core::Link::Solo} else if rank==0 {core::Link::Lead(links.iter_mut().map(|l|l as &mut dyn core::Duplex).collect())} else {core::Link::Follow(&mut links[0])};
    let pc=core::open(rank,cfg,identity,binary,link);
    PC.with(|p|*p.borrow_mut()=pc);
}

/// Serve loop housekeeping (between steps, both ranks): release the tensors of finished writes, move staged restore
/// chunks to the device.
pub(crate) fn poll() {PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {pc.poll();});}
/// A restore is in flight on this rank (the serve loop keeps the other rank stepping so it can copy its chunks).
pub(crate) fn loading() -> bool {PC.with(|p|p.borrow().as_ref().is_some_and(|pc|pc.loading()))}
/// Wait until no background write still reads store `tag`'s rows (before they are overwritten or returned to the pool).
pub(crate) fn fence(tag:u64) {PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {pc.fence(tag);});}
/// Flush pending writes (shutdown), up to `secs`.
pub(crate) fn shutdown(secs:u64) {PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {pc.shutdown(secs);});}
pub(crate) fn stats()->Value {PC.with(|p|p.borrow().as_ref().map_or(Value::Null,|pc|pc.stats()))}

/// What one checkpoint consists of (device tensors of the store and its checkpoint).
pub(crate) struct Parts<'a> {
    pub kda:&'a [(Tensor,Tensor)],pub mla:&'a [(Tensor,Tensor,Tensor,Tensor)],pub draft:&'a crate::dflash::Context,pub last:&'a Tensor,
    /// The store's per-MLA-layer latent [capacity, width] and DSA pools [capacity/4, dim] (rows below the prefix are final).
    pub latent:Vec<Tensor>,pub pools:Vec<Tensor>,
}
/// GLM's row arrays: per MLA layer its latent rows (one per token) and DSA pool rows (one per 4 tokens, the last,
/// incomplete one included).
fn rows_of(latent:&[Tensor],pools:&[Tensor])->Vec<Rows> {
    latent.iter().zip(pools).enumerate().flat_map(|(l,(lat,pool))|[Rows{name:format!("lat.{l}"),t:lat.shallow_clone(),div:1,ceil:false},
        Rows{name:format!("pool.{l}"),t:pool.shallow_clone(),div:4,ceil:true}]).collect()
}
/// Save checkpoint `which` of prefix `ids` (both ranks, right after the checkpoint is taken). `segs`: the store's
/// segment provenance, extended with the segments written here. `tag`: the store (see `fence`).
pub(crate) fn save(ids:&[i64],which:usize,parts:Parts,tag:u64,segs:&mut Vec<u64>)->Option<u64> {
    PC.with(|p|{let mut b=p.borrow_mut();let pc=b.as_mut()?;
        let mut fixed:Vec<(String,Part)>=vec![("last".into(),Part::Dev(parts.last.shallow_clone()))];
        for (i,(h,c)) in parts.kda.iter().enumerate() {fixed.push((format!("kda.{i}.h"),Part::Dev(h.shallow_clone())));fixed.push((format!("kda.{i}.conv"),Part::Dev(c.shallow_clone())));}
        for (i,(len,tk,tg,pr)) in parts.mla.iter().enumerate() {
            for (n,t) in [("len",len),("tk",tk),("tg",tg),("prow",pr)] {fixed.push((format!("mla.{i}.{n}"),Part::Dev(t.shallow_clone())));}}
        for (j,(k,v)) in parts.draft.kv().iter().enumerate() {fixed.push((format!("draft.{j}.k"),Part::Dev(k.shallow_clone())));fixed.push((format!("draft.{j}.v"),Part::Dev(v.shallow_clone())));}
        let mut meta=serde_json::Map::new();meta.insert("draft_len".into(),json!(parts.draft.len));meta.insert("draft_start".into(),json!(parts.draft.start));
        pc.save(Snapshot{ids,which,rows:rows_of(&parts.latent,&parts.pools),fixed,meta},tag,segs)
    })
}

/// Both ranks: a request reuses checkpoint `which` of prefix `ids` from memory: refresh its disk entry in the LRU
/// order (the persistent copy of what is hot in memory must not be the first to go).
pub(crate) fn touch(ids:&[i64],which:usize) {PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {pc.touch(ids,which);});}
/// rank0: the longest cached entry that is a prefix of `q` (strict, or exact with its last-row logits), longer than
/// `longer_than` and at least `min_len`. Returns (entry id, length).
pub(crate) fn lookup(q:&[i64],longer_than:usize,min_len:usize)->Option<(u64,usize)> {
    PC.with(|p|p.borrow().as_ref()?.lookup(q,longer_than,min_len,|which|which==0))
}
/// rank0: the longest prefix `q` shares with a cached entry it then diverges from (the fork point), 0 when none. An entry
/// that is itself a prefix of `q` is a `lookup` hit, not a fork.
pub(crate) fn fork_point(q:&[i64])->usize {PC.with(|p|p.borrow().as_ref().map_or(0,|pc|pc.fork_point(q)))}
pub(crate) use core::common_prefix;

/// A restored checkpoint (device copies), for the caller to install into the store.
pub(crate) struct Restored {pub which:usize,pub ids:Vec<i64>,pub kda:Vec<(Tensor,Tensor)>,pub mla:Vec<(Tensor,Tensor,Tensor,Tensor)>,pub draft:crate::dflash::Context,pub last:Tensor,pub segs:Vec<u64>}
/// An asynchronous restore (crate::pcache_core::Loading) of a GLM checkpoint.
pub(crate) struct Loading {inner:core::Loading,layers:usize}
impl Loading {
    /// Every chunk is read (or the load failed): `finish` only copies what is still staged.
    pub(crate) fn ready(&self)->bool {self.inner.ready()}
    /// Copy what is left, then assemble the checkpoint; `prompt` must extend the saved ids. None on any local failure.
    pub(crate) fn finish(self,prompt:&[i64])->Option<Restored> {
        let layers=self.layers;
        let assemble=move |l:core::Loaded|->Result<Restored,String>{
            let core::Loaded{which,ids,mut owned,meta,segs}=l;
            let draft_len=meta["draft_len"].as_i64().ok_or("draft len")?;let draft_start=meta["draft_start"].as_i64().ok_or("draft start")?;
            let mut take=|n:String|owned.remove(&n).ok_or(format!("missing {n}"));
            let kda=(0..).map_while(|i|{let h=take(format!("kda.{i}.h")).ok()?;Some((h,take(format!("kda.{i}.conv")).ok()?))}).collect::<Vec<_>>();
            let mut mla=Vec::new();
            while let Ok(len)=take(format!("mla.{}.len",mla.len())) {let i=mla.len();mla.push((len,take(format!("mla.{i}.tk"))?,take(format!("mla.{i}.tg"))?,take(format!("mla.{i}.prow"))?));}
            let mut kv=Vec::new();
            while let Ok(k)=take(format!("draft.{}.k",kv.len())) {let j=kv.len();kv.push((k,take(format!("draft.{j}.v"))?));}
            let last=take("last".into())?;
            if mla.len()!=layers {return Err("layer count mismatch".into());}
            Ok(Restored{which,ids,kda,mla,draft:crate::dflash::Context::from_parts(draft_len,draft_start,kv),last,segs})
        };
        let inner=self.inner;
        // (the cache is borrowed for the bookkeeping only: draining the staged chunks does not touch it)
        PC.with(|p|{let mut b=p.borrow_mut();let pc=b.as_mut()?;inner.finish(prompt,pc,assemble)})
    }
    /// Abandon (cancelled request): stop the readers, free the staged slots, discard the result.
    pub(crate) fn abandon(self) {self.inner.abandon();}
}
/// Start restoring entry `id` (this rank's files) into the store rows `latent`/`pools` (per MLA layer, as in `Parts`)
/// and new device tensors for the checkpoint. The device copies run on the serve thread's stream, after whatever
/// initialized the destinations. `touch`: a real restore (LRU and statistics); false for the read-back self-test.
pub(crate) fn start(id:u64,latent:&[Tensor],pools:&[Tensor],dev:Device,touch:bool)->Loading {
    let rows=rows_of(latent,pools);
    let inner=PC.with(|p|{let mut b=p.borrow_mut();b.as_mut().map(|pc|pc.start(id,&rows,&HashMap::new(),&[],dev,touch))});
    // (cache off: a load that fails at once)
    let inner=inner.unwrap_or_else(||core::Loading::failed("prefix cache off"));
    Loading{inner,layers:latent.len()}
}
/// Synchronous restore (self-test): start + finish.
pub(crate) fn restore(id:u64,prompt:&[i64],latent:&[Tensor],pools:&[Tensor],dev:Device,touch:bool)->Option<Restored> {start(id,latent,pools,dev,touch).finish(prompt)}
/// Both ranks: agree on a restore's outcome (one small allreduce). Returns true only if every rank succeeded.
pub(crate) fn agree(ok:bool,dev:Device)->bool {core::agree(ok,dev)}
/// Both ranks: drop entry `id` after a failed restore, with every entry that shares one of its segments (the failing
/// part is unknown here: a damaged segment would fail each of them in turn, one cold prefill per request).
pub(crate) fn drop_entry(id:u64) {PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {pc.drop_entry(id);});}
