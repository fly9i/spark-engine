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

use std::path::{Path,PathBuf};
use std::sync::atomic::{AtomicBool,AtomicU8,Ordering};
use std::sync::{Arc,mpsc};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;
use serde_json::{json,Value};
use tch::{Device,Kind,Tensor};

const FORMAT:u64=1;
const ALIGN:usize=4096;
const CHUNK:usize=16<<20;
#[cfg(target_arch="aarch64")] const O_DIRECT:i32=0o200000;
#[cfg(not(target_arch="aarch64"))] const O_DIRECT:i32=0o40000;
const PENDING:u8=0;const OK:u8=1;const FAILED:u8=2;const CANCELLED:u8=3;

extern "C" {
    fn cudaSetDevice(d:i32)->i32;
    fn cudaStreamCreateWithFlags(s:*mut *mut std::ffi::c_void,flags:u32)->i32;
    fn cudaStreamSynchronize(s:*mut std::ffi::c_void)->i32;
    fn cudaStreamWaitEvent(s:*mut std::ffi::c_void,e:*mut std::ffi::c_void,flags:u32)->i32;
    fn cudaEventCreateWithFlags(e:*mut *mut std::ffi::c_void,flags:u32)->i32;
    fn cudaEventRecord(e:*mut std::ffi::c_void,s:*mut std::ffi::c_void)->i32;
    fn cudaEventSynchronize(e:*mut std::ffi::c_void)->i32;
    fn cudaEventDestroy(e:*mut std::ffi::c_void)->i32;
    fn cudaMemcpyAsync(dst:*mut std::ffi::c_void,src:*const std::ffi::c_void,n:usize,kind:i32,s:*mut std::ffi::c_void)->i32;
    fn cudaHostRegister(p:*mut std::ffi::c_void,n:usize,flags:u32)->i32;
    fn rs_current_stream()->*mut std::ffi::c_void;
    fn mmap(addr:*mut std::ffi::c_void,len:usize,prot:i32,flags:i32,fd:i32,off:i64)->*mut std::ffi::c_void;
    fn statvfs(path:*const std::ffi::c_char,buf:*mut u64)->i32;
}
const D2H:i32=2;const H2D:i32=1;

pub(crate) fn requested()->bool {std::env::var("GLM53_PCACHE").as_deref()==Ok("1")}
fn env_u64(k:&str,d:u64)->u64 {std::env::var(k).ok().and_then(|v|v.parse().ok()).unwrap_or(d)}
fn pad(n:usize)->usize {n.div_ceil(ALIGN)*ALIGN}

// ---------------------------------------------------------------- hashing
fn mix(mut x:u64)->u64 {x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^x>>31}
/// Streaming 128-bit hash of a token prefix (entries are also verified token by token on restore).
#[derive(Clone,Copy)] struct PrefixHash {a:u64,b:u64,n:u64}
impl PrefixHash {
    fn new()->Self {Self{a:0x243f6a8885a308d3,b:0x13198a2e03707344,n:0}}
    fn push(&mut self,t:i64) {
        let m=mix(t as u64^self.n.wrapping_mul(0x9e3779b97f4a7c15));
        self.a=(self.a^m).wrapping_mul(0xff51afd7ed558ccd).rotate_left(27);
        self.b=self.b.wrapping_add(mix(m^0xa4093822299f31d0)).wrapping_mul(0xc4ceb9fe1a85ec53).rotate_left(31);self.n+=1;
    }
    fn digest(&self)->u128 {((mix(self.a^self.n) as u128)<<64)|mix(self.b^self.n.rotate_left(32)) as u128}
}
fn hash_ids(ids:&[i64])->u128 {let mut h=PrefixHash::new();for &t in ids {h.push(t);}h.digest()}
/// Checksum of a (zero padded) chunk: rotate-xor-multiply over 64-bit words.
fn checksum(b:&[u8])->u64 {
    let mut s=0x9e3779b97f4a7c15u64;
    for w in b.chunks_exact(8) {s=(s.rotate_left(5)^u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x100000001b3);}
    mix(s^b.len() as u64)
}

fn kind_name(k:Kind)->&'static str {match k {Kind::Uint8=>"u8",Kind::Int8=>"i8",Kind::Int=>"i32",Kind::Int64=>"i64",Kind::Half=>"f16",Kind::Float=>"f32",Kind::BFloat16=>"bf16",Kind::Bool=>"bool",k=>panic!("prefix cache: unsupported kind {k:?}")}}
fn kind_of(s:&str)->Option<Kind> {Some(match s {"u8"=>Kind::Uint8,"i8"=>Kind::Int8,"i32"=>Kind::Int,"i64"=>Kind::Int64,"f16"=>Kind::Half,"f32"=>Kind::Float,"bf16"=>Kind::BFloat16,"bool"=>Kind::Bool,_=>return None})}
fn nbytes(t:&Tensor)->usize {t.numel()*t.kind().elt_size_in_bytes()}

// ---------------------------------------------------------------- pinned staging
struct Pinned {ptr:*mut u8,len:usize}
unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}
impl Pinned {
    /// Anonymous pages pinned with cudaHostRegister (compaction leaves them alone, see shim/host_pin.cuh). Kept for the
    /// process lifetime.
    fn new(len:usize)->Self {
        const PROT_RW:i32=3;const MAP_PRIVATE_ANON_POPULATE:i32=0x02|0x20|0x8000;
        let p=unsafe{mmap(std::ptr::null_mut(),len,PROT_RW,MAP_PRIVATE_ANON_POPULATE,-1,0)};
        assert!(p as isize!=-1,"prefix cache: mmap {len}");
        assert_eq!(unsafe{cudaHostRegister(p,len,1)},0,"prefix cache: cudaHostRegister");
        Self{ptr:p.cast(),len}
    }
    fn slice(&self,n:usize)->&mut [u8] {assert!(n<=self.len);unsafe{std::slice::from_raw_parts_mut(self.ptr,n)}}
}
fn stream()->usize {let mut s=std::ptr::null_mut();assert_eq!(unsafe{cudaStreamCreateWithFlags(&mut s,1)},0,"prefix cache stream");s as usize}

// ---------------------------------------------------------------- file format
/// One tensor region of a file: its table entry and where its bytes come from (write) or go (read).
#[derive(Clone)] struct Region {name:String,kind:Kind,shape:Vec<i64>,off:u64,bytes:usize}
enum Src {Dev(usize),Host(Arc<Vec<u8>>)}
fn table(regions:&[Region],sums:&[Vec<u64>])->Value {
    Value::Array(regions.iter().zip(sums).map(|(r,s)|json!({"name":r.name,"kind":kind_name(r.kind),"shape":r.shape,"off":r.off,"bytes":r.bytes,
        "sums":s.iter().map(|v|format!("{v:016x}")).collect::<Vec<_>>()})).collect())
}
/// Place `regions` after the header (sized to hold the table with its checksums): (header bytes, file size).
fn layout(regions:&mut [Region],fixed:&Value)->(usize,u64) {
    // Size the table with 19-digit offsets and full-width checksums, then place the regions after it.
    let dummy:Vec<Vec<u64>>=regions.iter().map(|r|vec![0;r.bytes.div_ceil(CHUNK)]).collect();
    for r in regions.iter_mut() {r.off=u64::MAX/2;}
    let mut probe=fixed.clone();probe["tensors"]=table(regions,&dummy);
    let head=pad(8+serde_json::to_vec(&probe).unwrap().len()+64);
    let mut off=head as u64;for r in regions.iter_mut() {r.off=off;off+=pad(r.bytes) as u64;}
    (head,off)
}
fn read_header(path:&Path)->Option<Value> {
    use std::io::Read;let mut f=std::fs::File::open(path).ok()?;
    let mut n=[0u8;8];f.read_exact(&mut n).ok()?;let n=u64::from_le_bytes(n) as usize;
    if n>64<<20 {return None;}
    let mut b=vec![0u8;n];f.read_exact(&mut b).ok()?;serde_json::from_slice(&b).ok()
}
fn regions_of(h:&Value)->Option<Vec<(Region,Vec<u64>)>> {
    h["tensors"].as_array()?.iter().map(|t|Some((Region{name:t["name"].as_str()?.to_string(),kind:kind_of(t["kind"].as_str()?)?,
        shape:t["shape"].as_array()?.iter().map(|v|v.as_i64()).collect::<Option<Vec<_>>>()?,off:t["off"].as_u64()?,bytes:t["bytes"].as_u64()? as usize},
        t["sums"].as_array()?.iter().map(|v|u64::from_str_radix(v.as_str()?,16).ok()).collect::<Option<Vec<_>>>()?))).collect()
}

// ---------------------------------------------------------------- writer thread
struct FilePlan {path:PathBuf,head:usize,size:u64,fixed:Value,regions:Vec<Region>,srcs:Vec<Src>,state:Arc<AtomicU8>,cancel:Arc<AtomicBool>}
enum Job {
    Write {event:usize,files:Vec<FilePlan>,done:Arc<AtomicBool>,keep:Vec<Tensor>},
    Delete(PathBuf),
    Touch(PathBuf),
    Flush(mpsc::Sender<()>),
}
struct Writer {stream:usize,bufs:[Pinned;2],dir:PathBuf,margin:u64,events:[usize;2]}
impl Writer {
    fn run(self,rx:mpsc::Receiver<Job>,back:mpsc::Sender<Vec<Tensor>>) {
        unsafe{cudaSetDevice(0);}
        for job in rx {match job {
            Job::Write{event,files,done,keep}=>{
                let waited=unsafe{cudaStreamWaitEvent(self.stream as _,event as _,0)}==0;
                let mut wrote=false;
                for f in &files {
                    if f.cancel.load(Ordering::Acquire) {f.state.store(CANCELLED,Ordering::Release);continue;}
                    if !waited {eprintln!("[pcache] {}: cudaStreamWaitEvent failed",f.path.display());f.state.store(FAILED,Ordering::Release);continue;}
                    let t=Instant::now();
                    match self.write(f) {
                        Ok(())=>{f.state.store(OK,Ordering::Release);wrote=true;
                            if std::env::var("GLM53_PCACHE_LOG").as_deref()==Ok("1") {eprintln!("[pcache] wrote {} ({:.1} MiB) in {:.1} ms",f.path.display(),f.size as f64/1048576.,t.elapsed().as_secs_f64()*1e3);}}
                        Err(e)=>{eprintln!("[pcache] write {} failed: {e}",f.path.display());let _=std::fs::remove_file(f.path.with_extension("tmp"));f.state.store(FAILED,Ordering::Release);}
                    }
                }
                if wrote {if let Ok(d)=std::fs::File::open(&self.dir) {let _=d.sync_all();}}
                unsafe{cudaEventDestroy(event as _);}
                done.store(true,Ordering::Release);let _=back.send(keep);
            }
            Job::Delete(p)=>{let _=std::fs::remove_file(&p);let _=std::fs::remove_file(p.with_extension("tmp"));}
            Job::Touch(p)=>{if let Ok(f)=std::fs::File::options().write(true).open(&p) {let _=f.set_modified(std::time::SystemTime::now());}}
            Job::Flush(tx)=>{let _=tx.send(());}
        }}
    }
    fn write(&self,f:&FilePlan)->std::io::Result<()> {
        use std::os::unix::fs::{OpenOptionsExt,FileExt};
        // Local disk headroom (not part of the replicated budget): a refused write fails this rank's copy only.
        let mut sv=[0u64;16];let c=std::ffi::CString::new(self.dir.as_os_str().as_encoded_bytes()).unwrap();
        if unsafe{statvfs(c.as_ptr(),sv.as_mut_ptr())}==0 && sv[1]*sv[4]<f.size+self.margin {return Err(std::io::Error::other("disk headroom below GLM53_PCACHE_FREE_GIB"));}
        let tmp=f.path.with_extension("tmp");
        let file=std::fs::OpenOptions::new().write(true).create(true).truncate(true).custom_flags(O_DIRECT).open(&tmp)?;
        // Chunk list over all regions: (region, offset in region, length). D2H of chunk i+1 overlaps the write of chunk i.
        let chunks:Vec<(usize,usize,usize)>=f.regions.iter().enumerate().flat_map(|(i,r)|
            (0..r.bytes.div_ceil(CHUNK)).map(move |k|(i,k*CHUNK,(r.bytes-k*CHUNK).min(CHUNK)))).collect();
        let cuda=|rc:i32,what:&str|if rc==0 {Ok(())} else {Err(std::io::Error::other(format!("{what}: CUDA error {rc}")))};
        let mut sums:Vec<Vec<u64>>=f.regions.iter().map(|_|Vec::new()).collect();
        let issue=|k:usize|->std::io::Result<()>{let (i,o,n)=chunks[k];let b=&self.bufs[k%2];
            match &f.srcs[i] {
                Src::Dev(p)=>cuda(unsafe{cudaMemcpyAsync(b.ptr.cast(),(*p+o) as *const std::ffi::c_void,n,D2H,self.stream as _)},"D2H")?,
                Src::Host(v)=>b.slice(n).copy_from_slice(&v[o..o+n]),
            }
            cuda(unsafe{cudaEventRecord(self.events[k%2] as _,self.stream as _)},"event record")};
        if !chunks.is_empty() {issue(0)?;}
        for k in 0..chunks.len() {
            cuda(unsafe{cudaEventSynchronize(self.events[k%2] as _)},"D2H sync")?;
            let (i,o,n)=chunks[k];let b=&self.bufs[k%2];let len=pad(n);
            b.slice(len)[n..].fill(0);
            let buf=b.slice(len);
            sums[i].push(checksum(buf));
            // Chunk k's buffer is written below; chunk k+1 goes to the other buffer (its previous chunk k-1 is written).
            if k+1<chunks.len() {issue(k+1)?;}
            file.write_all_at(buf,f.regions[i].off+o as u64)?;
        }
        let mut h=f.fixed.clone();h["tensors"]=table(&f.regions,&sums);
        let js=serde_json::to_vec(&h).unwrap();
        if 8+js.len()>f.head {return Err(std::io::Error::other("header larger than reserved"));}
        let hb=self.bufs[0].slice(f.head);hb.fill(0);hb[..8].copy_from_slice(&(js.len() as u64).to_le_bytes());hb[8..8+js.len()].copy_from_slice(&js);
        file.write_all_at(hb,0)?;
        file.sync_data()?;drop(file);
        std::fs::rename(&tmp,&f.path)?;
        Ok(())
    }
}

// ---------------------------------------------------------------- replicated index
struct Entry {id:u64,key:u128,len:usize,which:usize,bytes:u64,segs:Vec<u64>,used:u64,path:PathBuf,state:Arc<AtomicU8>,cancel:Arc<AtomicBool>}
struct Seg {refs:u32,bytes:u64,path:PathBuf,state:Arc<AtomicU8>,cancel:Arc<AtomicBool>}
struct Pending {tag:u64,done:Arc<AtomicBool>}
#[derive(Default)] struct Stats {saves:u64,skipped_dup:u64,skipped_budget:u64,evicted:u64,restores:u64,restore_fail:u64,restored_tokens:u64,restore_ms:f64,written_bytes:u64}
struct Pc {
    rank:usize,dir:PathBuf,identity:String,binary:String,cap:u64,min:usize,seg:i64,
    entries:Vec<Entry>,segs:HashMap<u64,Seg>,next_id:u64,next_seg:u64,clock:u64,total:u64,
    tx:mpsc::Sender<Job>,back:mpsc::Receiver<Vec<Tensor>>,pending:Vec<Pending>,
    stage:Arc<Stage>,reader_threads:usize,loads:Vec<std::sync::Weak<LoadShared>>,stats:Stats,
}
thread_local!{static PC:RefCell<Option<Pc>>=const{RefCell::new(None)};}
pub(crate) fn enabled()->bool {PC.with(|p|p.borrow().is_some())}
/// Segment size in tokens (0 when the cache is off).
pub(crate) fn seg_tokens()->i64 {PC.with(|p|p.borrow().as_ref().map_or(0,|p|p.seg))}

fn identity(binary:&str,model:&Path,draft:&Path,world:usize,seg:i64)->String {
    // Scheduling, memory, logging, loading and transport settings do not change the saved values.
    let skip=["GLM53_SERVE_","GLM53_MEM_","GLM53_KV_POOL","GLM53_HOST_","GLM53_RELEASE_FILE_CACHE","GLM53_NCCL_","GLM53_PCACHE","GLM53_TP_RANK",
        "GLM53_MASTER_","GLM53_ENGINE_BIN","GLM53_FAST_","GLM53_PRELOAD_","GLM53_LOAD_","GLM53_GRAPH_EMPTY_CACHE","GLM53_DRAFT_TRACE","GLM53_DRAFT_EXPORT",
        "GLM53_PREFIX_ADMISSION","GLM53_SPEC_GRAPH_SLOTS","GLM53_DEPTH_RULE","GLM53_LOG"];
    let mut env:Vec<(String,String)>=std::env::vars().filter(|(k,_)|k.starts_with("GLM53_")&&!skip.iter().any(|s|k.starts_with(s))).collect();env.sort();
    let canon=|p:&Path|std::fs::canonicalize(p).unwrap_or_else(|_|p.to_path_buf()).display().to_string();
    format!("format={FORMAT};binary={binary};model={};draft={};world={world};seg={seg};env={env:?}",canon(model),canon(draft))
}
fn short(s:&str)->String {use sha2::Digest;let d=sha2::Sha256::digest(s.as_bytes());d[..8].iter().map(|b|format!("{b:02x}")).collect()}
fn dir_bytes(d:&Path)->u64 {std::fs::read_dir(d).map(|r|r.flatten().map(|e|{let m=e.metadata();match m {Ok(m) if m.is_dir()=>dir_bytes(&e.path()),Ok(m)=>m.len(),_=>0}}).sum()).unwrap_or(0)}
fn mtime(p:&Path)->u64 {std::fs::metadata(p).and_then(|m|m.modified()).ok().and_then(|t|t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0,|d|d.as_nanos() as u64)}

fn send_msg(s:&mut std::net::TcpStream,v:&Value) {use std::io::Write;let b=serde_json::to_vec(v).unwrap();s.write_all(&(b.len() as u64).to_le_bytes()).unwrap();s.write_all(&b).unwrap();}
fn recv_msg(s:&mut std::net::TcpStream)->Value {use std::io::Read;let mut n=[0u8;8];s.read_exact(&mut n).expect("prefix cache handshake");
    let mut b=vec![0u8;u64::from_le_bytes(n) as usize];s.read_exact(&mut b).expect("prefix cache handshake");serde_json::from_slice(&b).unwrap()}

/// Startup (both ranks, after the doorbell connection and before serving): scan this rank's files, agree with the
/// other rank on the common valid set and its LRU order (rank0's), delete the rest, start the writer thread.
pub(crate) fn init(rank:usize,world:usize,model:&Path,draft:&Path,link:&mut std::net::TcpStream) {
    let t0=Instant::now();
    let root=PathBuf::from(std::env::var("GLM53_PCACHE_DIR").unwrap_or_else(|_|"prefix-cache".into()));
    let seg=env_u64("GLM53_PCACHE_SEG",4096) as i64;assert!(seg>=64&&seg%64==0,"GLM53_PCACHE_SEG must be a multiple of 64");
    let cap=env_u64("GLM53_PCACHE_GIB",100)<<30;let min=env_u64("GLM53_PCACHE_MIN",1024) as usize;
    let binary=std::fs::read(std::env::current_exe().unwrap()).map(|b|{use sha2::Digest;sha2::Sha256::digest(&b).iter().map(|x|format!("{x:02x}")).collect::<String>()}).expect("prefix cache: read the engine binary");
    let identity=identity(&binary,model,draft,world,seg);let id_dir=short(&identity);
    let dir=root.join(&id_dir).join(format!("rank{rank}"));std::fs::create_dir_all(&dir).expect("prefix cache directory");
    let _=std::fs::write(root.join(&id_dir).join("identity.txt"),&identity);
    // Scan: temporaries are unfinished writes; entries/segments need a readable header with this identity.
    let (mut ents,mut segs):(Vec<Value>,HashMap<u64,u64>)=(Vec::new(),HashMap::new());
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p=e.path();let name=p.file_name().unwrap().to_string_lossy().to_string();
        let len=e.metadata().map(|m|m.len()).unwrap_or(0);
        if name.ends_with(".tmp") {let _=std::fs::remove_file(&p);continue;}
        let h=read_header(&p).filter(|h|h["identity"].as_str()==Some(identity.as_str())&&h["format"].as_u64()==Some(FORMAT));
        match (h,name.split('.').last()) {
            (Some(h),Some("seg")) if h["seg"].as_u64().is_some()=>{segs.insert(h["seg"].as_u64().unwrap(),len);}
            (Some(h),Some("snap"))=>ents.push(json!({"name":name,"key":h["key"],"len":h["len"],"which":h["which"],"segs":h["segs"],"bytes":len,"mtime":mtime(&p)})),
            _=>{let _=std::fs::remove_file(&p);}
        }
    }
    let valid=|e:&Value,segs:&HashMap<u64,u64>|e["segs"].as_array().is_some_and(|a|a.iter().all(|s|s.as_u64().is_some_and(|s|segs.contains_key(&s))));
    ents.retain(|e|valid(e,&segs));
    // Other identities (older profiles / numerics versions) count against the cap: their oldest directories are
    // dropped (locally) while they hold more than half of it. This identity's budget uses the larger rank's rest.
    let others:Vec<(PathBuf,u64,u64)>={let mut v:Vec<_>=std::fs::read_dir(&root).map(|r|r.flatten().filter(|e|e.path().is_dir()&&e.file_name().to_string_lossy()!=id_dir)
        .map(|e|{let p=e.path().join(format!("rank{rank}"));let b=dir_bytes(&p);(p,b,mtime(&e.path()))}).filter(|x|x.1>0).collect()).unwrap_or_default();v.sort_by_key(|x|x.2);v};
    let mut others_left=others.iter().map(|o|o.1).sum::<u64>();
    for (p,b,_) in &others {if others_left<=cap/2 {break;}let _=std::fs::remove_dir_all(p);others_left-=b;eprintln!("[pcache] removed older cache {} ({:.1} GiB)",p.display(),*b as f64/(1u64<<30) as f64);}
    let mine=json!({"identity":identity,"entries":ents,"segs":segs.iter().map(|(k,v)|json!([k,v])).collect::<Vec<_>>(),"others":others_left});
    let plan=if rank==0 {
        let peer=if world>1 {recv_msg(link)} else {mine.clone()};
        let plan=if peer["identity"]!=mine["identity"] {
            eprintln!("[pcache] identity differs between ranks: cache disabled");json!({"ok":false})
        } else {
            let peer_segs:HashMap<u64,u64>=peer["segs"].as_array().unwrap().iter().map(|v|(v[0].as_u64().unwrap(),v[1].as_u64().unwrap())).collect();
            let peer_names:HashMap<String,&Value>=peer["entries"].as_array().unwrap().iter().map(|e|(e["name"].as_str().unwrap().to_string(),e)).collect();
            let mut keep:Vec<&Value>=ents.iter().filter(|e|peer_names.get(e["name"].as_str().unwrap()).is_some_and(|p|p["segs"]==e["segs"]&&p["len"]==e["len"])
                &&valid(e,&peer_segs)).collect();
            keep.sort_by_key(|e|e["mtime"].as_u64().unwrap());
            let other=mine["others"].as_u64().unwrap().max(peer["others"].as_u64().unwrap());
            let next_seg=segs.keys().chain(peer_segs.keys()).max().map_or(0,|m|m+1);
            json!({"ok":true,"entries":keep.iter().map(|e|json!({"name":e["name"],"key":e["key"],"len":e["len"],"which":e["which"],"segs":e["segs"],"bytes":e["bytes"]})).collect::<Vec<_>>(),
                "seg_bytes":segs.iter().map(|(k,v)|json!([k,v])).collect::<Vec<_>>(),"others":other,"next_seg":next_seg})
        };
        if world>1 {send_msg(link,&plan);}plan
    } else {send_msg(link,&mine);recv_msg(link)};
    if plan["ok"].as_bool()!=Some(true) {return;}
    let cap_mine=cap.saturating_sub(plan["others"].as_u64().unwrap().min(cap/2));
    let (tx,rx)=mpsc::channel();let (btx,brx)=mpsc::channel();
    let margin=env_u64("GLM53_PCACHE_FREE_GIB",20)<<30;
    let ev=||{let mut e=std::ptr::null_mut();assert_eq!(unsafe{cudaEventCreateWithFlags(&mut e,2)},0);e as usize};
    let writer=Writer{stream:stream(),bufs:[Pinned::new(CHUNK),Pinned::new(CHUNK)],dir:dir.clone(),margin,events:[ev(),ev()]};
    std::thread::Builder::new().name("pcache-writer".into()).spawn(move||writer.run(rx,btx)).unwrap();
    let slots=(env_u64("GLM53_PCACHE_STAGE_MB",256) as usize).div_ceil(CHUNK>>20).max(2);
    let stage=Arc::new(Stage{slots:(0..slots).map(|_|Pinned::new(CHUNK)).collect(),free:std::sync::Mutex::new((0..slots).collect()),cv:std::sync::Condvar::new()});
    let reader_threads=env_u64("GLM53_PCACHE_READERS",4).clamp(1,16) as usize;
    let mut pc=Pc{rank,dir:dir.clone(),identity,binary,cap:cap_mine,min,seg,entries:Vec::new(),segs:HashMap::new(),next_id:0,next_seg:plan["next_seg"].as_u64().unwrap(),
        clock:0,total:0,tx,back:brx,pending:Vec::new(),stage,reader_threads,loads:Vec::new(),stats:Stats::default()};
    let seg_bytes:HashMap<u64,u64>=plan["seg_bytes"].as_array().unwrap().iter().map(|v|(v[0].as_u64().unwrap(),v[1].as_u64().unwrap())).collect();
    let kept:std::collections::HashSet<String>=plan["entries"].as_array().unwrap().iter().map(|e|e["name"].as_str().unwrap().to_string()).collect();
    for e in plan["entries"].as_array().unwrap() {
        let segs_e:Vec<u64>=e["segs"].as_array().unwrap().iter().map(|s|s.as_u64().unwrap()).collect();
        for s in &segs_e {let b=seg_bytes[s];let seg=pc.segs.entry(*s).or_insert_with(||{pc.total+=b;Seg{refs:0,bytes:b,path:dir.join(format!("s-{s:016x}.seg")),
            state:Arc::new(AtomicU8::new(OK)),cancel:Arc::new(AtomicBool::new(false))}});seg.refs+=1;}
        let bytes=e["bytes"].as_u64().unwrap();pc.total+=bytes;pc.clock+=1;
        pc.entries.push(Entry{id:pc.next_id,key:u128::from_str_radix(e["key"].as_str().unwrap(),16).unwrap(),len:e["len"].as_u64().unwrap() as usize,
            which:e["which"].as_u64().unwrap() as usize,bytes,segs:segs_e,used:pc.clock,path:dir.join(e["name"].as_str().unwrap()),
            state:Arc::new(AtomicU8::new(OK)),cancel:Arc::new(AtomicBool::new(false))});pc.next_id+=1;
    }
    // Everything else of this identity goes.
    for f in std::fs::read_dir(&dir).unwrap().flatten() {let n=f.file_name().to_string_lossy().to_string();
        let keep=if n.ends_with(".snap") {kept.contains(&n)} else if let Some(h)=n.strip_prefix("s-").and_then(|s|s.strip_suffix(".seg")) {
            u64::from_str_radix(h,16).is_ok_and(|id|pc.segs.contains_key(&id))} else {false};
        if !keep {let _=std::fs::remove_file(f.path());}}
    let before=pc.entries.len();pc.evict_to(0,None);
    eprintln!("[pcache] rank{rank} {} entries ({} evicted to fit), {} segments, {:.2} GiB of {:.1} GiB at {} ({:.2}s)",pc.entries.len(),before-pc.entries.len(),
        pc.segs.len(),pc.total as f64/(1u64<<30) as f64,pc.cap as f64/(1u64<<30) as f64,dir.display(),t0.elapsed().as_secs_f64());
    PC.with(|p|*p.borrow_mut()=Some(pc));
}

impl Pc {
    /// Evict least recently used entries until `extra` more bytes fit (never `protect`). Replicated.
    fn evict_to(&mut self,extra:u64,protect:Option<u64>) {
        while self.total+extra>self.cap {
            let Some(i)=self.entries.iter().enumerate().filter(|(_,e)|Some(e.id)!=protect).min_by_key(|(_,e)|e.used).map(|(i,_)|i) else {break};
            self.remove(i);self.stats.evicted+=1;
        }
    }
    fn remove(&mut self,i:usize) {
        let e=self.entries.remove(i);e.cancel.store(true,Ordering::Release);self.total-=e.bytes;let _=self.tx.send(Job::Delete(e.path.clone()));
        for s in &e.segs {let gone={let seg=self.segs.get_mut(s).unwrap();seg.refs-=1;seg.refs==0};
            if gone {let seg=self.segs.remove(s).unwrap();seg.cancel.store(true,Ordering::Release);self.total-=seg.bytes;let _=self.tx.send(Job::Delete(seg.path));}}
    }
    fn reap(&mut self) {while let Ok(keep)=self.back.try_recv() {drop(keep);}self.pending.retain(|p|!p.done.load(Ordering::Acquire));}
}
/// Serve loop housekeeping (between steps, both ranks): release the tensors of finished writes, move staged restore
/// chunks to the device.
pub(crate) fn poll() {PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {pc.reap();pump_all(pc);});}
/// A restore is in flight on this rank (the serve loop keeps the other rank stepping so it can copy its chunks).
pub(crate) fn loading() -> bool {PC.with(|p|p.borrow().as_ref().is_some_and(|pc|pc.loads.iter().any(|w|w.strong_count()>0)))}
/// Wait until no background write still reads store `tag`'s rows (before they are overwritten or returned to the pool).
pub(crate) fn fence(tag:u64) {
    PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {
        pc.reap();if !pc.pending.iter().any(|j|j.tag==tag) {return;}
        let t=Instant::now();
        while pc.pending.iter().any(|j|j.tag==tag) {
            // A dead writer no longer reads anything.
            if let Err(mpsc::RecvTimeoutError::Disconnected)=pc.back.recv_timeout(std::time::Duration::from_millis(1)).map(drop) {pc.pending.clear();break;}
            pc.pending.retain(|p|!p.done.load(Ordering::Acquire));}
        if std::env::var("GLM53_PCACHE_LOG").as_deref()==Ok("1") {eprintln!("[pcache] fence store {tag}: waited {:.1} ms",t.elapsed().as_secs_f64()*1e3);}
    });
}
/// Flush pending writes (shutdown), up to `secs`.
pub(crate) fn shutdown(secs:u64) {
    PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {
        let (tx,rx)=mpsc::channel();let _=pc.tx.send(Job::Flush(tx));
        let ok=rx.recv_timeout(std::time::Duration::from_secs(secs)).is_ok();pc.reap();
        eprintln!("[pcache] rank{} shutdown: writes {}",pc.rank,if ok {"flushed"} else {"still pending (timeout)"});
    });
}
pub(crate) fn stats()->Value {
    PC.with(|p|p.borrow().as_ref().map_or(Value::Null,|pc|{let s=&pc.stats;json!({"entries":pc.entries.len(),"segments":pc.segs.len(),"bytes":pc.total,"cap":pc.cap,
        "pending_writes":pc.pending.len(),"saves":s.saves,"skipped_dup":s.skipped_dup,"skipped_budget":s.skipped_budget,"evicted":s.evicted,"restores":s.restores,
        "restore_failures":s.restore_fail,"restored_tokens":s.restored_tokens,"restore_ms":s.restore_ms,"written_bytes":s.written_bytes})}))
}

/// What one checkpoint consists of (device tensors of the store and its checkpoint).
pub(crate) struct Parts<'a> {
    pub kda:&'a [(Tensor,Tensor)],pub mla:&'a [(Tensor,Tensor,Tensor,Tensor)],pub draft:&'a crate::dflash::Context,pub last:&'a Tensor,
    /// The store's per-MLA-layer latent [capacity, width] and DSA pools [capacity/4, dim] (rows below the prefix are final).
    pub latent:Vec<Tensor>,pub pools:Vec<Tensor>,
}
fn seg_regions(parts_latent:&[Tensor],parts_pools:&[Tensor],a:i64,n:i64,pa:i64,pn:i64)->(Vec<Region>,Vec<Src>,Vec<Tensor>) {
    let (mut r,mut s,mut k)=(Vec::new(),Vec::new(),Vec::new());
    for (l,(lat,pool)) in parts_latent.iter().zip(parts_pools).enumerate() {
        for (name,t) in [(format!("lat.{l}"),lat.narrow(0,a,n)),(format!("pool.{l}"),pool.narrow(0,pa,pn))] {
            assert!(t.is_contiguous());r.push(Region{name,kind:t.kind(),shape:t.size(),off:0,bytes:nbytes(&t)});s.push(Src::Dev(t.data_ptr() as usize));k.push(t);
        }
    }
    (r,s,k)
}
/// Save checkpoint `which` of prefix `ids` (both ranks, right after the checkpoint is taken). `segs`: the store's
/// segment provenance, extended with the segments written here. `tag`: the store (see `fence`).
pub(crate) fn save(ids:&[i64],which:usize,parts:Parts,tag:u64,segs:&mut Vec<u64>)->Option<u64> {
    PC.with(|p|{let mut b=p.borrow_mut();let Some(pc)=b.as_mut() else {return None};
        pc.reap();
        let plen=ids.len();if plen<pc.min {return None;}
        let key=hash_ids(ids);
        pc.clock+=1;let clock=pc.clock;
        if let Some(e)=pc.entries.iter_mut().find(|e|e.key==key&&e.len==plen&&e.which==which) {e.used=clock;pc.stats.skipped_dup+=1;let _=pc.tx.send(Job::Touch(e.path.clone()));return None;}
        let s=pc.seg;let nfull=(plen as i64/s) as usize;
        // Provenance: keep the prefix of segments that still exist (an evicted one ends it).
        let valid=segs.iter().take_while(|id|pc.segs.contains_key(id)).count();segs.truncate(valid.min(nfull));
        let complete=(plen as i64+3)/4;let fixed_seg=json!({"format":FORMAT,"identity":pc.identity,"binary":pc.binary,"seg":0u64,"start":0i64,"tokens":s});
        // New segments.
        let mut files=Vec::new();let mut keep=Vec::new();let mut new_segs=Vec::new();let mut bytes=0u64;
        for k in segs.len()..nfull {
            let id=pc.next_seg+new_segs.len() as u64;let a=k as i64*s;
            let (mut regions,srcs,kk)=seg_regions(&parts.latent,&parts.pools,a,s,a/4,s/4);
            let mut fixed=fixed_seg.clone();fixed["seg"]=json!(id);fixed["start"]=json!(a);
            let (head,size)=layout(&mut regions,&fixed);bytes+=size;keep.extend(kk);
            files.push(FilePlan{path:pc.dir.join(format!("s-{id:016x}.seg")),head,size,fixed,regions,srcs,state:Arc::new(AtomicU8::new(PENDING)),cancel:Arc::new(AtomicBool::new(false))});
            new_segs.push(id);
        }
        // The snapshot file: ids, fixed-size state, rows after the last full segment.
        let (mut regions,mut srcs)=(Vec::new(),Vec::new());
        let add=|name:String,t:&Tensor,regions:&mut Vec<Region>,srcs:&mut Vec<Src>,keep:&mut Vec<Tensor>|{
            let t=if t.is_contiguous() {t.shallow_clone()} else {t.contiguous()};
            regions.push(Region{name,kind:t.kind(),shape:t.size(),off:0,bytes:nbytes(&t)});srcs.push(Src::Dev(t.data_ptr() as usize));keep.push(t);};
        let idb:Vec<u8>=ids.iter().flat_map(|t|t.to_le_bytes()).collect();
        regions.push(Region{name:"ids".into(),kind:Kind::Int64,shape:vec![plen as i64],off:0,bytes:idb.len()});srcs.push(Src::Host(Arc::new(idb)));
        let tail=plen as i64-nfull as i64*s;let pa=nfull as i64*s/4;
        let (r,sr,kk)=seg_regions(&parts.latent,&parts.pools,nfull as i64*s,tail,pa,complete-pa);regions.extend(r);srcs.extend(sr);keep.extend(kk);
        add("last".into(),parts.last,&mut regions,&mut srcs,&mut keep);
        for (i,(h,c)) in parts.kda.iter().enumerate() {add(format!("kda.{i}.h"),h,&mut regions,&mut srcs,&mut keep);add(format!("kda.{i}.conv"),c,&mut regions,&mut srcs,&mut keep);}
        for (i,(len,tk,tg,pr)) in parts.mla.iter().enumerate() {
            for (n,t) in [("len",len),("tk",tk),("tg",tg),("prow",pr)] {add(format!("mla.{i}.{n}"),t,&mut regions,&mut srcs,&mut keep);}}
        for (j,(k,v)) in parts.draft.kv().iter().enumerate() {add(format!("draft.{j}.k"),k,&mut regions,&mut srcs,&mut keep);add(format!("draft.{j}.v"),v,&mut regions,&mut srcs,&mut keep);}
        let all_segs:Vec<u64>=segs.iter().copied().chain(new_segs.iter().copied()).collect();
        let fixed=json!({"format":FORMAT,"identity":pc.identity,"binary":pc.binary,"key":format!("{key:032x}"),"len":plen,"which":which,"segs":all_segs,"seg_tokens":s,
            "draft_len":parts.draft.len,"draft_start":parts.draft.start,"created":format!("{:016x}",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs())});
        let (head,size)=layout(&mut regions,&fixed);bytes+=size;
        if bytes>pc.cap {pc.stats.skipped_budget+=1;return None;}
        // Reused segments count for this entry before anything is evicted.
        for id in segs.iter() {pc.segs.get_mut(id).unwrap().refs+=1;}
        let id=pc.next_id;pc.next_id+=1;pc.next_seg+=new_segs.len() as u64;
        let path=pc.dir.join(format!("e-{key:032x}-{which}.snap"));
        let entry_file=FilePlan{path:path.clone(),head,size,fixed,regions,srcs,state:Arc::new(AtomicU8::new(PENDING)),cancel:Arc::new(AtomicBool::new(false))};
        for (f,&sid) in files.iter().zip(&new_segs) {pc.segs.insert(sid,Seg{refs:1,bytes:f.size,path:f.path.clone(),state:f.state.clone(),cancel:f.cancel.clone()});pc.total+=f.size;}
        pc.entries.push(Entry{id,key,len:plen,which,bytes:size,segs:all_segs.clone(),used:clock,path,state:entry_file.state.clone(),cancel:entry_file.cancel.clone()});pc.total+=size;
        pc.evict_to(0,Some(id));
        // Eviction may have taken this entry's own segments only if it was itself evicted (protected): not possible.
        files.push(entry_file);
        let mut event=std::ptr::null_mut();assert_eq!(unsafe{cudaEventCreateWithFlags(&mut event,2)},0);
        assert_eq!(unsafe{cudaEventRecord(event,rs_current_stream())},0,"prefix cache: record");
        let done=Arc::new(AtomicBool::new(false));pc.pending.push(Pending{tag,done:done.clone()});
        pc.stats.saves+=1;pc.stats.written_bytes+=bytes;
        let _=pc.tx.send(Job::Write{event:event as usize,files,done,keep});
        *segs=all_segs;
        if std::env::var("GLM53_PCACHE_LOG").as_deref()==Ok("1") {eprintln!("[pcache] save {plen} tokens which {which}: {} new segments, {:.1} MiB, total {:.2} GiB, {} entries",
            new_segs.len(),bytes as f64/1048576.,pc.total as f64/(1u64<<30) as f64,pc.entries.len());}
        Some(id)
    })
}

/// Both ranks: a request reuses checkpoint `which` of prefix `ids` from memory: refresh its disk entry in the LRU
/// order (the persistent copy of what is hot in memory must not be the first to go).
pub(crate) fn touch(ids:&[i64],which:usize) {
    PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {
        if ids.len()<pc.min {return;}
        let key=hash_ids(ids);pc.clock+=1;let clock=pc.clock;
        if let Some(e)=pc.entries.iter_mut().find(|e|e.key==key&&e.len==ids.len()&&e.which==which) {e.used=clock;let _=pc.tx.send(Job::Touch(e.path.clone()));}
    });
}
/// rank0: the longest cached entry that is a prefix of `q` (strict, or exact with its last-row logits), longer than
/// `longer_than` and at least `min_len`. Returns (entry id, length).
pub(crate) fn lookup(q:&[i64],longer_than:usize,min_len:usize)->Option<(u64,usize)> {
    PC.with(|p|{let b=p.borrow();let pc=b.as_ref()?;
        let mut cands:Vec<&Entry>=pc.entries.iter().filter(|e|e.len<=q.len()&&(e.len<q.len()||e.which==0)&&e.len>longer_than&&e.len>=min_len
            &&e.state.load(Ordering::Acquire)==OK&&e.segs.iter().all(|s|pc.segs[s].state.load(Ordering::Acquire)==OK)).collect();
        if cands.is_empty() {return None;}
        cands.sort_by_key(|e|e.len);
        let mut h=PrefixHash::new();let mut at=0usize;let mut best:Option<&Entry>=None;
        for e in cands {while at<e.len {h.push(q[at]);at+=1;}
            if h.digest()==e.key && best.map_or(true,|b|e.len>b.len||(e.len==b.len&&e.which==0)) {best=Some(e);}}
        best.map(|e|(e.id,e.len))
    })
}

/// A restored checkpoint (device copies), for the caller to install into the store.
pub(crate) struct Restored {pub which:usize,pub ids:Vec<i64>,pub kda:Vec<(Tensor,Tensor)>,pub mla:Vec<(Tensor,Tensor,Tensor,Tensor)>,pub draft:crate::dflash::Context,pub last:Tensor,pub segs:Vec<u64>}
/// Pinned staging ring for restores (GLM53_PCACHE_STAGE_MB, default 256): reader threads fill free slots from disk,
/// the serve thread copies filled slots to the device on its own stream between steps and frees them.
struct Stage {slots:Vec<Pinned>,free:std::sync::Mutex<Vec<usize>>,cv:std::sync::Condvar}
/// One restore's shared progress: filled slots waiting for their device copies (slot, [(offset in slot, destination, bytes)]).
struct LoadShared {stage:Arc<Stage>,ready:std::sync::Mutex<std::collections::VecDeque<(usize,Vec<(usize,usize,usize)>)>>,read_done:AtomicBool,cancel:AtomicBool}
impl LoadShared {
    /// Serve thread: copy every filled slot to its destination on the current stream, wait, free the slots. `discard`:
    /// free them without copying (abandoned load). Returns the number of slots handled.
    fn pump(&self,discard:bool)->Result<usize,String> {
        let batch:Vec<(usize,Vec<(usize,usize,usize)>)>=self.ready.lock().unwrap().drain(..).collect();
        if batch.is_empty() {return Ok(0);}
        let mut r=Ok(batch.len());
        if !discard {
            let st=unsafe{rs_current_stream()};
            'copy: for (slot,pieces) in &batch {for &(off,dst,n) in pieces {
                if unsafe{cudaMemcpyAsync(dst as *mut std::ffi::c_void,self.stage.slots[*slot].ptr.add(off).cast(),n,H2D,st)}!=0 {r=Err("H2D copy".to_string());break 'copy;}}}
            if unsafe{cudaStreamSynchronize(st)}!=0 {r=Err("H2D sync".into());}
        }
        self.stage.free.lock().unwrap().extend(batch.iter().map(|b|b.0));self.stage.cv.notify_all();
        r
    }
}
/// An asynchronous restore (P2): reader threads read and checksum the files into the staging ring in the background;
/// the serve thread moves filled slots into the store rows / new checkpoint tensors between its steps (`poll`), where the
/// device is idle (copies issued from other threads while sequences decode ran ~30x slower, bench/pcache). The target
/// rows and tensors belong to the loading sequence only; `finish` drains the rest and assembles the checkpoint.
pub(crate) struct Loading {
    id:u64,which:usize,plen:usize,segs:Vec<u64>,draft_len:i64,draft_start:i64,layers:usize,
    owned:HashMap<String,Tensor>,shared:Option<Arc<LoadShared>>,job:Option<std::thread::JoinHandle<Result<(),String>>>,err:Option<String>,
    t0:Instant,waited_ms:f64,bytes:usize,touch:bool,
}
impl Loading {
    /// Every chunk is read (or the load failed): `finish` only copies what is still staged.
    pub(crate) fn ready(&self)->bool {self.err.is_some()||self.shared.as_ref().is_some_and(|s|s.read_done.load(Ordering::Acquire))}
    fn drain(&mut self,discard:bool)->Result<(),String> {
        if let Some(e)=self.err.take() {if let Some(s)=&self.shared {s.cancel.store(true,Ordering::Release);}let _=self.settle();return Err(e);}
        let Some(sh)=self.shared.clone() else {return Err("no load".into())};
        if discard {sh.cancel.store(true,Ordering::Release);}
        let mut res=Ok(());
        loop {
            let done=sh.read_done.load(Ordering::Acquire);
            if let Err(e)=sh.pump(discard||res.is_err()) {res=Err(e);sh.cancel.store(true,Ordering::Release);}
            if done && sh.ready.lock().unwrap().is_empty() {break;}
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        let joined=self.settle();res.and(joined)
    }
    fn settle(&mut self)->Result<(),String> {match self.job.take() {Some(h)=>h.join().unwrap_or_else(|_|Err("reader thread panicked".into())),None=>Ok(())}}
    /// Copy what is left, then assemble the checkpoint; `prompt` must extend the saved ids. None on any local failure.
    pub(crate) fn finish(mut self,prompt:&[i64])->Option<Restored> {
        let (id,plen,which)=(self.id,self.plen,self.which);
        let r=(||->Result<Restored,String>{
            self.drain(false)?;
            let ids=Vec::<i64>::try_from(self.owned.remove("ids").ok_or("missing ids")?).map_err(|e|e.to_string())?;
            if prompt.len()<plen||prompt[..plen]!=ids[..] {return Err("token ids differ from the prompt".into());}
            let owned=&mut self.owned;let mut take=|n:String|owned.remove(&n).ok_or(format!("missing {n}"));
            let kda=(0..).map_while(|i|{let h=take(format!("kda.{i}.h")).ok()?;Some((h,take(format!("kda.{i}.conv")).ok()?))}).collect::<Vec<_>>();
            let mut mla=Vec::new();
            while let Ok(len)=take(format!("mla.{}.len",mla.len())) {let i=mla.len();mla.push((len,take(format!("mla.{i}.tk"))?,take(format!("mla.{i}.tg"))?,take(format!("mla.{i}.prow"))?));}
            let mut kv=Vec::new();
            while let Ok(k)=take(format!("draft.{}.k",kv.len())) {let j=kv.len();kv.push((k,take(format!("draft.{j}.v"))?));}
            let last=take("last".into())?;
            if mla.len()!=self.layers {return Err("layer count mismatch".into());}
            Ok(Restored{which,ids,kda,mla,draft:crate::dflash::Context::from_parts(self.draft_len,self.draft_start,kv),last,segs:self.segs.clone()})
        })();
        let ms=self.t0.elapsed().as_secs_f64()*1e3;
        PC.with(|p|{let mut b=p.borrow_mut();let pc=b.as_mut()?;
            match r {
                Ok(r)=>{
                    if std::env::var("GLM53_PCACHE_LOG").as_deref()==Ok("1") {eprintln!("[pcache] rank{} restore {plen} tokens which {which}: {:.1} MiB in {ms:.1} ms (waited {:.1} ms for the write)",
                        pc.rank,self.bytes as f64/1048576.,self.waited_ms);}
                    if self.touch {
                        pc.clock+=1;let clock=pc.clock;
                        if let Some(e)=pc.entries.iter_mut().find(|e|e.id==id) {e.used=clock;let _=pc.tx.send(Job::Touch(e.path.clone()));}
                        pc.stats.restores+=1;pc.stats.restored_tokens+=plen as u64;pc.stats.restore_ms+=ms;
                    }
                    Some(r)}
                Err(m)=>{eprintln!("[pcache] rank{} restore of entry {id} ({plen} tokens) failed: {m}",pc.rank);None}
            }})
    }
    /// Abandon (cancelled request): stop the readers, free the staged slots, discard the result.
    pub(crate) fn abandon(mut self) {let _=self.drain(true);}
}
impl Drop for Loading {fn drop(&mut self) {if self.job.is_some() {let _=self.drain(true);}}}
/// Start restoring entry `id` (this rank's files) into the store rows `latent`/`pools` (per MLA layer, as in `Parts`)
/// and new device tensors for the checkpoint. The device copies run on the serve thread's stream, after whatever
/// initialized the destinations. `touch`: a real restore (LRU and statistics); false for the read-back self-test.
pub(crate) fn start(id:u64,latent:&[Tensor],pools:&[Tensor],dev:Device,touch:bool)->Loading {
    let t0=Instant::now();
    let mut ld=Loading{id,which:0,plen:0,segs:Vec::new(),draft_len:0,draft_start:0,layers:latent.len(),owned:HashMap::new(),shared:None,job:None,
        err:None,t0,waited_ms:0.,bytes:0,touch};
    let r=PC.with(|p|->Result<(),String>{let mut b=p.borrow_mut();let pc=b.as_mut().ok_or("prefix cache off")?;
        pc.reap();
        let i=pc.entries.iter().position(|e|e.id==id).ok_or("no such entry")?;
        // This rank's writes of the entry and its segments must have finished (rank0 only picks written entries; the
        // other rank's copy is normally done too, else this waits for it).
        let states:Vec<Arc<AtomicU8>>=std::iter::once(pc.entries[i].state.clone()).chain(pc.entries[i].segs.iter().map(|s|pc.segs[s].state.clone())).collect();
        while states.iter().any(|s|s.load(Ordering::Acquire)==PENDING) {match pc.back.recv_timeout(std::time::Duration::from_millis(1)) {
            Ok(k)=>drop(k),Err(mpsc::RecvTimeoutError::Timeout)=>{},
            Err(mpsc::RecvTimeoutError::Disconnected)=>return Err("writer thread is gone".into())}}
        pc.reap();
        ld.waited_ms=t0.elapsed().as_secs_f64()*1e3;
        if states.iter().any(|s|s.load(Ordering::Acquire)!=OK) {return Err("local write failed".into());}
        let e=&pc.entries[i];let s=pc.seg;(ld.plen,ld.which,ld.segs)=(e.len,e.which,e.segs.clone());let plen=e.len;
        let h=read_header(&e.path).ok_or("unreadable header")?;
        if h["identity"].as_str()!=Some(pc.identity.as_str())||h["len"].as_u64()!=Some(plen as u64)||h["which"].as_u64()!=Some(e.which as u64)
            ||h["segs"]!=json!(e.segs) {return Err("header does not match the index".into());}
        ld.draft_len=h["draft_len"].as_i64().ok_or("draft len")?;ld.draft_start=h["draft_start"].as_i64().ok_or("draft start")?;
        let regs=regions_of(&h).ok_or("bad tensor table")?;
        if !regs.iter().any(|r|r.0.name=="ids") {return Err("missing ids".into());}
        // Reads: (file, region, sums, destination pointer, host destination). Rows go straight into the store.
        let mut reads:Vec<(PathBuf,Region,Vec<u64>,usize,bool)>=Vec::new();
        let check=|r:&Region,t:&Tensor|->Result<(),String>{if r.kind!=t.kind()||r.shape!=t.size() {Err(format!("{}: {:?} {:?} vs {:?} {:?}",r.name,r.kind,r.shape,t.kind(),t.size()))} else {Ok(())}};
        let layer=|n:&str|->Result<usize,String>{n.parse::<usize>().ok().filter(|&l|l<latent.len()).ok_or(format!("bad layer {n}"))};
        for (k,sid) in e.segs.iter().enumerate() {
            let sp=pc.segs.get(sid).ok_or("segment gone")?.path.clone();
            let sh=read_header(&sp).ok_or("unreadable segment header")?;
            if sh["identity"].as_str()!=Some(pc.identity.as_str())||sh["seg"].as_u64()!=Some(*sid)||sh["start"].as_i64()!=Some(k as i64*s) {return Err("segment header mismatch".into());}
            for (r,sums) in regions_of(&sh).ok_or("bad segment table")? {
                let (kind,l)=r.name.split_once('.').ok_or("bad segment tensor")?;let l=layer(l)?;
                let dst=if kind=="lat" {latent[l].narrow(0,k as i64*s,s)} else {pools[l].narrow(0,k as i64*s/4,s/4)};
                check(&r,&dst)?;let p=dst.data_ptr() as usize;reads.push((sp.clone(),r,sums,p,false));
            }
        }
        let nfull=e.segs.len() as i64;if nfull*s>plen as i64 {return Err("segments exceed the prefix".into());}
        let tail=plen as i64-nfull*s;let complete=(plen as i64+3)/4;let pa=nfull*s/4;
        for (r,sums) in regs {
            let dst=if let Some(l)=r.name.strip_prefix("lat.") {latent[layer(l)?].narrow(0,nfull*s,tail)}
                else if let Some(l)=r.name.strip_prefix("pool.") {pools[layer(l)?].narrow(0,pa,complete-pa)}
                else {let t=if r.name=="ids" {Tensor::empty([plen as i64],(Kind::Int64,Device::Cpu))} else {Tensor::empty(r.shape.as_slice(),(r.kind,dev))};
                    ld.owned.insert(r.name.clone(),t.shallow_clone());t};
            check(&r,&dst)?;
            let host=r.name=="ids";reads.push((e.path.clone(),r,sums,dst.data_ptr() as usize,host));
        }
        ld.bytes=reads.iter().map(|r|r.1.bytes).sum();
        let sh=Arc::new(LoadShared{stage:pc.stage.clone(),ready:std::sync::Mutex::new(Default::default()),read_done:AtomicBool::new(false),cancel:AtomicBool::new(false)});
        pc.loads.retain(|w|w.strong_count()>0);pc.loads.push(Arc::downgrade(&sh));
        ld.shared=Some(sh.clone());
        let threads=pc.reader_threads;
        ld.job=Some(std::thread::Builder::new().name("pcache-reader".into()).spawn(move||{
            let r=read_all(&sh,threads,&reads);if r.is_err() {sh.cancel.store(true,Ordering::Release);}
            sh.read_done.store(true,Ordering::Release);r
        }).map_err(|e|e.to_string())?);
        Ok(())
    });
    if let Err(e)=r {ld.err=Some(e);}
    ld
}
/// Synchronous restore (self-test): start + finish.
pub(crate) fn restore(id:u64,prompt:&[i64],latent:&[Tensor],pools:&[Tensor],dev:Device,touch:bool)->Option<Restored> {start(id,latent,pools,dev,touch).finish(prompt)}
/// Serve thread: move staged chunks of every running restore to the device (`poll`).
fn pump_all(pc:&mut Pc) {
    pc.loads.retain(|w|w.strong_count()>0);
    for w in &pc.loads {if let Some(sh)=w.upgrade() {if !sh.cancel.load(Ordering::Acquire) {let _=sh.pump(false).map_err(|e|{sh.cancel.store(true,Ordering::Release);e});}}}
}
/// Parallel O_DIRECT reads into free staging slots, checksum per chunk; device-bound pieces are queued for the serve
/// thread, the token ids are copied to their CPU tensor right here. Adjacent regions of a file (a segment holds 22 small
/// per-layer regions) share one slot-sized read.
fn read_all(sh:&LoadShared,threads:usize,reads:&[(PathBuf,Region,Vec<u64>,usize,bool)])->Result<(),String> {
    use std::os::unix::fs::{OpenOptionsExt,FileExt};
    let mut files:HashMap<&PathBuf,std::fs::File>=HashMap::new();
    for (p,..) in reads {if !files.contains_key(p) {files.insert(p,std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(p).map_err(|e|format!("open {}: {e}",p.display()))?);}}
    for (_,r,sums,..) in reads {if sums.len()!=r.bytes.div_ceil(CHUNK) {return Err(format!("{}: checksum count",r.name));}}
    // Pieces (read, chunk k, bytes) in file order; a read = (file, offset, padded length, pieces with their slot offsets).
    let mut order:Vec<usize>=(0..reads.len()).collect();order.sort_by(|&a,&b|(&reads[a].0,reads[a].1.off).cmp(&(&reads[b].0,reads[b].1.off)));
    let mut plan:Vec<(&PathBuf,u64,usize,Vec<(usize,usize,usize,usize)>)>=Vec::new();
    for i in order {let r=&reads[i];
        for k in 0..r.1.bytes.div_ceil(CHUNK) {
            let n=(r.1.bytes-k*CHUNK).min(CHUNK);let off=r.1.off+(k*CHUNK) as u64;
            match plan.last_mut() {
                Some(c) if c.0==&r.0 && c.1+c.2 as u64==off && c.2+pad(n)<=CHUNK=>{c.3.push((c.2,i,k,n));c.2+=pad(n);}
                _=>plan.push((&r.0,off,pad(n),vec![(0,i,k,n)])),
            }
        }}
    let next=std::sync::atomic::AtomicUsize::new(0);let files=&files;
    // GLM53_PCACHE_LOG=1: per-phase time over all reader threads (read, checksum, waiting for a free slot).
    let phase=[std::sync::atomic::AtomicU64::new(0),std::sync::atomic::AtomicU64::new(0),std::sync::atomic::AtomicU64::new(0)];let t0=Instant::now();
    let errs:Vec<String>=std::thread::scope(|sc|{
        let hs:Vec<_>=(0..threads).map(|_|{let next=&next;let plan=&plan;let phase=&phase;sc.spawn(move||->Result<(),String>{
            let lap=|t:&mut Instant,i:usize|{phase[i].fetch_add(t.elapsed().as_micros() as u64,Ordering::Relaxed);*t=Instant::now();};
            loop {
                if sh.cancel.load(Ordering::Acquire) {return Ok(());}
                let c=next.fetch_add(1,Ordering::Relaxed);let Some((p,off,len,pieces))=plan.get(c) else {return Ok(())};
                let mut t=Instant::now();
                let slot={let mut f=sh.stage.free.lock().unwrap();
                    loop {if let Some(x)=f.pop() {break x;} if sh.cancel.load(Ordering::Acquire) {return Ok(());}
                        f=sh.stage.cv.wait_timeout(f,std::time::Duration::from_millis(5)).unwrap().0;}};
                lap(&mut t,2);
                // Test only: slow every slot load down (cancellation / concurrency tests).
                if let Some(ms)=std::env::var("GLM53_PCACHE_TEST_DELAY_MS").ok().and_then(|v|v.parse::<u64>().ok()) {std::thread::sleep(std::time::Duration::from_millis(ms));}
                let b=sh.stage.slots[slot].slice(*len);
                let res=(||{files[p].read_exact_at(b,*off).map_err(|e|format!("read {} @{off}: {e}",p.display()))?;lap(&mut t,0);
                    let mut dev=Vec::new();
                    for &(so,i,k,n) in pieces {let (_,r,sums,dst,host)=&reads[i];
                        if checksum(&b[so..so+pad(n)])!=sums[k] {return Err(format!("checksum mismatch in {} {} chunk {k}",p.display(),r.name));}
                        if *host {unsafe{std::ptr::copy_nonoverlapping(b.as_ptr().add(so),(*dst as *mut u8).add(k*CHUNK),n);}} else {dev.push((so,*dst+k*CHUNK,n));}}
                    lap(&mut t,1);Ok(dev)})();
                match res {
                    Ok(dev) if !dev.is_empty()=>sh.ready.lock().unwrap().push_back((slot,dev)),
                    r=>{sh.stage.free.lock().unwrap().push(slot);sh.stage.cv.notify_all();r?;}
                }
            }})}).collect();
        hs.into_iter().filter_map(|h|h.join().unwrap().err()).collect()});
    if std::env::var("GLM53_PCACHE_LOG").as_deref()==Ok("1") {let ms=|i:usize|phase[i].load(Ordering::Relaxed) as f64/1e3;
        eprintln!("[pcache] read {} slot loads in {:.1} ms wall ({threads} threads): read {:.1} ms, checksum {:.1} ms, waiting for a slot {:.1} ms",plan.len(),t0.elapsed().as_secs_f64()*1e3,ms(0),ms(1),ms(2));}
    if errs.is_empty() {Ok(())} else {Err(errs.join("; "))}
}
/// Both ranks: agree on a restore's outcome (one small allreduce). Returns true only if every rank succeeded.
pub(crate) fn agree(ok:bool,dev:Device)->bool {
    let t=Tensor::from_slice(&[if ok {0f32} else {1f32}]).to_device(dev);crate::tp::allreduce(&t);t.double_value(&[0])==0.
}
/// Both ranks: drop entry `id` after a failed restore, with every entry that shares one of its segments (the failing
/// part is unknown here: a damaged segment would fail each of them in turn, one cold prefill per request).
pub(crate) fn drop_entry(id:u64) {
    PC.with(|p|if let Some(pc)=p.borrow_mut().as_mut() {pc.stats.restore_fail+=1;
        let Some(segs)=pc.entries.iter().find(|e|e.id==id).map(|e|e.segs.clone()) else {return};
        while let Some(i)=pc.entries.iter().position(|e|e.id==id||e.segs.iter().any(|s|segs.contains(s))) {pc.remove(i);}
        eprintln!("[pcache] rank{} dropped entry {id} and those sharing its {} segments ({} entries left)",pc.rank,segs.len(),pc.entries.len());
    });
}
