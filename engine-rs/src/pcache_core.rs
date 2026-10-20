// SPDX-License-Identifier: MIT
//! Persistent prefix cache on the local NVMe, model-independent core (GLM: crate::pcache, DeepSeek-V4.1:
//! crate::ds41::pcache). A model describes a sequence checkpoint as data:
//! - row arrays ([`Rows`]): per-token tensors, one row per `div` tokens (GLM: an MLA layer's latent rows, div 1, and its
//!   DSA pool rows, div 4; DeepSeek-V4.1: each KV source's compressed KV and index keys, div = its ratio). They are
//!   written in segments of `seg` tokens shared by every checkpoint of one store lineage (multi-turn: turn N reuses turn
//!   N-1's segments and only writes the new ones), and the rows after the last full segment go into the checkpoint;
//! - fixed regions ([`Part`]): everything else, in the checkpoint file;
//! - extra header fields (`meta`).
//!
//! Layout (per rank, `<root>/<identity>/rank<r>/`): `s-<id>.seg` segments, `e-<key>-<which>.snap` checkpoints. Files are
//! written O_DIRECT through a small pinned ring (no page cache: on GB10 it would take unified memory from the GPU), as
//! `*.tmp`, fdatasync'ed and renamed; every 16 MiB chunk carries a checksum checked on restore. Restores read with
//! parallel O_DIRECT readers into a pinned staging ring; the serve thread copies staged chunks to the device between
//! its steps.
//!
//! Replication: the index (entries, segments, LRU clock, byte budget) is a replicated state machine: every rank saves at
//! the same points and evicts in the same order. Writes are local and asynchronous; a restore waits for its own rank's
//! write and the ranks agree on success ([`agree`]). At startup the ranks exchange their valid files (rank 0 with every
//! other rank over a byte stream) and keep the intersection, so a crash between the ranks' writes is safe.
//!
//! Identity (the model's: binary SHA-256, model paths, numerics settings) selects the directory; caches of other
//! identities are kept (a rollback finds its own) within half the cap.
//!
//! Extracted from GLM's crate::pcache without changing its files (engine-rs/testdata/pcache-golden-manifest.txt,
//! `pcache-golden`).

use std::path::{Path,PathBuf};
use std::sync::atomic::{AtomicBool,AtomicU8,Ordering};
use std::sync::{Arc,mpsc};
use std::collections::HashMap;
use std::time::Instant;
use serde_json::{json,Value};
use tch::{Device,Kind,Tensor};

pub(crate) const FORMAT:u64=1;
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

fn pad(n:usize)->usize {n.div_ceil(ALIGN)*ALIGN}

// ---------------------------------------------------------------- settings
/// The cache's settings (the model's adapter reads them from its environment variables).
pub(crate) struct Config {
    /// cache root (the identity's directory goes below it)
    pub root:PathBuf,
    /// segment size in tokens (a multiple of every row array's div)
    pub seg:i64,
    /// byte budget of this identity (other identities may hold up to half of it)
    pub cap:u64,
    /// shortest prefix saved, in tokens
    pub min:usize,
    /// local disk space kept free (a write that would cut into it fails on this rank only)
    pub free_margin:u64,
    pub stage_mb:u64,
    pub readers:usize,
    pub log:bool,
    /// log prefix, e.g. "[pcache]"
    pub tag:&'static str,
    /// tests only: slow every staged read down
    pub test_delay_ms:Option<u64>,
}
/// SHA-256 of the running engine binary (part of every identity: any rebuild starts a new cache).
pub(crate) fn binary_sha256()->String {
    std::fs::read(std::env::current_exe().unwrap()).map(|b|{use sha2::Digest;sha2::Sha256::digest(&b).iter().map(|x|format!("{x:02x}")).collect::<String>()}).expect("prefix cache: read the engine binary")
}

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
/// The token ids of a snapshot file (its `ids` region, `len` of them), for `fork_point`. Unchecked: a wrong value only
/// moves where a later prefill takes a fork checkpoint, never what is restored.
fn read_ids(path:&Path,len:usize)->Option<Arc<Vec<u32>>> {
    use std::os::unix::fs::FileExt;
    let h=read_header(path)?;let (r,_)=regions_of(&h)?.into_iter().find(|r|r.0.name=="ids")?;
    if r.bytes!=len*8 {return None;}
    let mut b=vec![0u8;r.bytes];std::fs::File::open(path).ok()?.read_exact_at(&mut b,r.off).ok()?;
    Some(Arc::new(b.chunks_exact(8).map(|c|i64::from_le_bytes(c.try_into().unwrap()) as u32).collect()))
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
struct Writer {stream:usize,bufs:[Pinned;2],dir:PathBuf,margin:u64,events:[usize;2],log:bool,tag:&'static str}
impl Writer {
    fn run(self,rx:mpsc::Receiver<Job>,back:mpsc::Sender<Vec<Tensor>>) {
        unsafe{cudaSetDevice(0);}
        let tag=self.tag;
        for job in rx {match job {
            Job::Write{event,files,done,keep}=>{
                let waited=unsafe{cudaStreamWaitEvent(self.stream as _,event as _,0)}==0;
                let mut wrote=false;
                for f in &files {
                    if f.cancel.load(Ordering::Acquire) {f.state.store(CANCELLED,Ordering::Release);continue;}
                    if !waited {eprintln!("{tag} {}: cudaStreamWaitEvent failed",f.path.display());f.state.store(FAILED,Ordering::Release);continue;}
                    let t=Instant::now();
                    match self.write(f) {
                        Ok(())=>{f.state.store(OK,Ordering::Release);wrote=true;
                            if self.log {eprintln!("{tag} wrote {} ({:.1} MiB) in {:.1} ms",f.path.display(),f.size as f64/1048576.,t.elapsed().as_secs_f64()*1e3);}}
                        Err(e)=>{eprintln!("{tag} write {} failed: {e}",f.path.display());let _=std::fs::remove_file(f.path.with_extension("tmp"));f.state.store(FAILED,Ordering::Release);}
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
        if unsafe{statvfs(c.as_ptr(),sv.as_mut_ptr())}==0 && sv[1]*sv[4]<f.size+self.margin {return Err(std::io::Error::other("disk headroom below the free-space margin"));}
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

// ---------------------------------------------------------------- what a checkpoint holds
/// A per-token tensor of a store: row i covers tokens [i div, (i + 1) div). `t`: the store's rows (rows below the prefix
/// are final). `ceil`: the checkpoint also holds the last, incomplete row (GLM's DSA pool); else only complete rows.
pub(crate) struct Rows {pub name:String,pub t:Tensor,pub div:i64,pub ceil:bool}
impl Rows {
    /// Rows the checkpoint of a `plen`-token prefix holds.
    fn end(&self,plen:i64)->i64 {if self.ceil {(plen+self.div-1)/self.div} else {plen/self.div}}
}
/// A fixed region of a checkpoint: device tensor, or host bytes (restored into a CPU tensor when its name is listed as
/// host on restore).
pub(crate) enum Part {Dev(Tensor),Host{kind:Kind,shape:Vec<i64>,bytes:Arc<Vec<u8>>}}
/// Checkpoint `which` of the prefix `ids`.
pub(crate) struct Snapshot<'a> {
    pub ids:&'a [i64],pub which:usize,
    pub rows:Vec<Rows>,
    pub fixed:Vec<(String,Part)>,
    /// extra header fields
    pub meta:serde_json::Map<String,Value>,
}

// ---------------------------------------------------------------- replicated index
/// `ids`: the entry's token ids are the first `len` of it (rank0 only, for `fork_point`; None on other ranks or when the
/// file's ids could not be read). Entries of one lineage share the longest one's buffer (`share_ids`).
struct Entry {id:u64,key:u128,len:usize,which:usize,bytes:u64,segs:Vec<u64>,used:u64,path:PathBuf,state:Arc<AtomicU8>,cancel:Arc<AtomicBool>,ids:Option<Arc<Vec<u32>>>}
struct Seg {refs:u32,bytes:u64,path:PathBuf,state:Arc<AtomicU8>,cancel:Arc<AtomicBool>}
struct Pending {tag:u64,done:Arc<AtomicBool>}
#[derive(Default)] struct Stats {saves:u64,skipped_dup:u64,skipped_budget:u64,evicted:u64,restores:u64,restore_fail:u64,restored_tokens:u64,restore_ms:f64,written_bytes:u64}
pub(crate) struct Cache {
    rank:usize,dir:PathBuf,identity:String,binary:String,cap:u64,min:usize,seg:i64,log:bool,tag:&'static str,test_delay_ms:Option<u64>,
    entries:Vec<Entry>,segs:HashMap<u64,Seg>,next_id:u64,next_seg:u64,clock:u64,total:u64,
    tx:mpsc::Sender<Job>,back:mpsc::Receiver<Vec<Tensor>>,pending:Vec<Pending>,
    stage:Arc<Stage>,reader_threads:usize,loads:Vec<std::sync::Weak<LoadShared>>,stats:Stats,
}

pub(crate) fn short(s:&str)->String {use sha2::Digest;let d=sha2::Sha256::digest(s.as_bytes());d[..8].iter().map(|b|format!("{b:02x}")).collect()}
fn dir_bytes(d:&Path)->u64 {std::fs::read_dir(d).map(|r|r.flatten().map(|e|{let m=e.metadata();match m {Ok(m) if m.is_dir()=>dir_bytes(&e.path()),Ok(m)=>m.len(),_=>0}}).sum()).unwrap_or(0)}
fn mtime(p:&Path)->u64 {std::fs::metadata(p).and_then(|m|m.modified()).ok().and_then(|t|t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0,|d|d.as_nanos() as u64)}

/// A byte stream between rank 0 and another rank (startup agreement).
pub(crate) trait Duplex: std::io::Read+std::io::Write {}
impl<T: std::io::Read+std::io::Write> Duplex for T {}
/// This rank's part in the startup agreement: alone, rank 0 with a stream to every other rank, or another rank.
pub(crate) enum Link<'a> {Solo,Lead(Vec<&'a mut dyn Duplex>),Follow(&'a mut dyn Duplex)}
fn send_msg(s:&mut dyn Duplex,v:&Value) {let b=serde_json::to_vec(v).unwrap();s.write_all(&(b.len() as u64).to_le_bytes()).unwrap();s.write_all(&b).unwrap();s.flush().unwrap();}
fn recv_msg(s:&mut dyn Duplex)->Value {let mut n=[0u8;8];s.read_exact(&mut n).expect("prefix cache handshake");
    let mut b=vec![0u8;u64::from_le_bytes(n) as usize];s.read_exact(&mut b).expect("prefix cache handshake");serde_json::from_slice(&b).unwrap()}

/// Rank 0's plan from its own scan (`mine`, with its entries `ents` and segments `segs`) and every other rank's: the
/// entries all ranks hold (same segments and length, every segment valid everywhere) in rank 0's LRU order.
fn plan(mine:&Value,ents:&[Value],segs:&HashMap<u64,u64>,peers:&[Value],tag:&str)->Value {
    let valid=|e:&Value,segs:&HashMap<u64,u64>|e["segs"].as_array().is_some_and(|a|a.iter().all(|s|s.as_u64().is_some_and(|s|segs.contains_key(&s))));
    if peers.iter().any(|p|p["identity"]!=mine["identity"]) {eprintln!("{tag} identity differs between ranks: cache disabled");return json!({"ok":false});}
    let peer_segs:Vec<HashMap<u64,u64>>=peers.iter().map(|p|p["segs"].as_array().unwrap().iter().map(|v|(v[0].as_u64().unwrap(),v[1].as_u64().unwrap())).collect()).collect();
    let peer_names:Vec<HashMap<String,&Value>>=peers.iter().map(|p|p["entries"].as_array().unwrap().iter().map(|e|(e["name"].as_str().unwrap().to_string(),e)).collect()).collect();
    let mut keep:Vec<&Value>=ents.iter().filter(|e|peer_names.iter().zip(&peer_segs).all(|(names,psegs)|
        names.get(e["name"].as_str().unwrap()).is_some_and(|p|p["segs"]==e["segs"]&&p["len"]==e["len"])&&valid(e,psegs))).collect();
    keep.sort_by_key(|e|e["mtime"].as_u64().unwrap());
    let other=peers.iter().fold(mine["others"].as_u64().unwrap(),|a,p|a.max(p["others"].as_u64().unwrap()));
    let next_seg=segs.keys().chain(peer_segs.iter().flat_map(|s|s.keys())).max().map_or(0,|m|m+1);
    json!({"ok":true,"entries":keep.iter().map(|e|json!({"name":e["name"],"key":e["key"],"len":e["len"],"which":e["which"],"segs":e["segs"],"bytes":e["bytes"]})).collect::<Vec<_>>(),
        "seg_bytes":segs.iter().map(|(k,v)|json!([k,v])).collect::<Vec<_>>(),"others":other,"next_seg":next_seg})
}

/// Startup (every rank, before serving): scan this rank's files, agree with the other ranks on the common valid set and
/// its LRU order (rank0's), delete the rest, start the writer thread. None: disabled (identities differ).
pub(crate) fn open(rank:usize,cfg:Config,identity:String,binary:String,link:Link)->Option<Cache> {
    let t0=Instant::now();let tag=cfg.tag;
    let (root,seg,cap,min)=(cfg.root.clone(),cfg.seg,cfg.cap,cfg.min);
    let id_dir=short(&identity);
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
    // dropped (locally) while they hold more than half of it. This identity's budget uses the largest rank's rest.
    let others:Vec<(PathBuf,u64,u64)>={let mut v:Vec<_>=std::fs::read_dir(&root).map(|r|r.flatten().filter(|e|e.path().is_dir()&&e.file_name().to_string_lossy()!=id_dir)
        .map(|e|{let p=e.path().join(format!("rank{rank}"));let b=dir_bytes(&p);(p,b,mtime(&e.path()))}).filter(|x|x.1>0).collect()).unwrap_or_default();v.sort_by_key(|x|x.2);v};
    let mut others_left=others.iter().map(|o|o.1).sum::<u64>();
    for (p,b,_) in &others {if others_left<=cap/2 {break;}let _=std::fs::remove_dir_all(p);others_left-=b;eprintln!("{tag} removed older cache {} ({:.1} GiB)",p.display(),*b as f64/(1u64<<30) as f64);}
    let mine=json!({"identity":identity,"entries":ents,"segs":segs.iter().map(|(k,v)|json!([k,v])).collect::<Vec<_>>(),"others":others_left});
    let plan=match link {
        Link::Solo=>plan(&mine,&ents,&segs,&[mine.clone()],tag),
        Link::Lead(mut ls)=>{
            let peers:Vec<Value>=ls.iter_mut().map(|l|recv_msg(&mut **l)).collect();
            let p=plan(&mine,&ents,&segs,&peers,tag);
            for l in ls.iter_mut() {send_msg(&mut **l,&p);}
            p
        }
        Link::Follow(l)=>{send_msg(l,&mine);recv_msg(l)}
    };
    if plan["ok"].as_bool()!=Some(true) {return None;}
    let cap_mine=cap.saturating_sub(plan["others"].as_u64().unwrap().min(cap/2));
    let (tx,rx)=mpsc::channel();let (btx,brx)=mpsc::channel();
    let ev=||{let mut e=std::ptr::null_mut();assert_eq!(unsafe{cudaEventCreateWithFlags(&mut e,2)},0);e as usize};
    let writer=Writer{stream:stream(),bufs:[Pinned::new(CHUNK),Pinned::new(CHUNK)],dir:dir.clone(),margin:cfg.free_margin,events:[ev(),ev()],log:cfg.log,tag};
    std::thread::Builder::new().name("pcache-writer".into()).spawn(move||writer.run(rx,btx)).unwrap();
    let slots=(cfg.stage_mb as usize).div_ceil(CHUNK>>20).max(2);
    let stage=Arc::new(Stage{slots:(0..slots).map(|_|Pinned::new(CHUNK)).collect(),free:std::sync::Mutex::new((0..slots).collect()),cv:std::sync::Condvar::new()});
    let mut pc=Cache{rank,dir:dir.clone(),identity,binary,cap:cap_mine,min,seg,log:cfg.log,tag,test_delay_ms:cfg.test_delay_ms,entries:Vec::new(),segs:HashMap::new(),next_id:0,
        next_seg:plan["next_seg"].as_u64().unwrap(),clock:0,total:0,tx,back:brx,pending:Vec::new(),stage,reader_threads:cfg.readers.clamp(1,16),loads:Vec::new(),stats:Stats::default()};
    let seg_bytes:HashMap<u64,u64>=plan["seg_bytes"].as_array().unwrap().iter().map(|v|(v[0].as_u64().unwrap(),v[1].as_u64().unwrap())).collect();
    let kept:std::collections::HashSet<String>=plan["entries"].as_array().unwrap().iter().map(|e|e["name"].as_str().unwrap().to_string()).collect();
    for e in plan["entries"].as_array().unwrap() {
        let segs_e:Vec<u64>=e["segs"].as_array().unwrap().iter().map(|s|s.as_u64().unwrap()).collect();
        for s in &segs_e {let b=seg_bytes[s];let seg=pc.segs.entry(*s).or_insert_with(||{pc.total+=b;Seg{refs:0,bytes:b,path:dir.join(format!("s-{s:016x}.seg")),
            state:Arc::new(AtomicU8::new(OK)),cancel:Arc::new(AtomicBool::new(false))}});seg.refs+=1;}
        let bytes=e["bytes"].as_u64().unwrap();pc.total+=bytes;pc.clock+=1;
        pc.entries.push(Entry{id:pc.next_id,key:u128::from_str_radix(e["key"].as_str().unwrap(),16).unwrap(),len:e["len"].as_u64().unwrap() as usize,
            which:e["which"].as_u64().unwrap() as usize,bytes,segs:segs_e,used:pc.clock,path:dir.join(e["name"].as_str().unwrap()),
            state:Arc::new(AtomicU8::new(OK)),cancel:Arc::new(AtomicBool::new(false)),ids:None});pc.next_id+=1;
        if rank==0 {let e=pc.entries.last_mut().unwrap();e.ids=read_ids(&e.path,e.len);}
    }
    if rank==0 {
        let mut order:Vec<usize>=(0..pc.entries.len()).collect();order.sort_by_key(|&i|std::cmp::Reverse(pc.entries[i].len));
        for i in order {if let Some(b)=pc.entries[i].ids.clone() {if Arc::strong_count(&b)==2 {pc.share_ids(i,&b[..]);}}}
    }
    // Everything else of this identity goes.
    for f in std::fs::read_dir(&dir).unwrap().flatten() {let n=f.file_name().to_string_lossy().to_string();
        let keep=if n.ends_with(".snap") {kept.contains(&n)} else if let Some(h)=n.strip_prefix("s-").and_then(|s|s.strip_suffix(".seg")) {
            u64::from_str_radix(h,16).is_ok_and(|id|pc.segs.contains_key(&id))} else {false};
        if !keep {let _=std::fs::remove_file(f.path());}}
    let before=pc.entries.len();pc.evict_to(0,None);
    eprintln!("{tag} rank{rank} {} entries ({} evicted to fit), {} segments, {:.2} GiB of {:.1} GiB at {} ({:.2}s)",pc.entries.len(),before-pc.entries.len(),
        pc.segs.len(),pc.total as f64/(1u64<<30) as f64,pc.cap as f64/(1u64<<30) as f64,dir.display(),t0.elapsed().as_secs_f64());
    Some(pc)
}

impl Cache {
    /// Segment size in tokens.
    pub(crate) fn seg_tokens(&self)->i64 {self.seg}
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
    /// Serve loop housekeeping (between steps, every rank): release the tensors of finished writes, move staged restore
    /// chunks to the device.
    pub(crate) fn poll(&mut self) {self.reap();self.pump_all();}
    /// A restore is in flight on this rank.
    pub(crate) fn loading(&self)->bool {self.loads.iter().any(|w|w.strong_count()>0)}
    /// Wait until no background write still reads store `tag`'s rows (before they are overwritten or returned to a pool).
    pub(crate) fn fence(&mut self,tag:u64) {
        self.reap();if !self.pending.iter().any(|j|j.tag==tag) {return;}
        let t=Instant::now();
        while self.pending.iter().any(|j|j.tag==tag) {
            // A dead writer no longer reads anything.
            if let Err(mpsc::RecvTimeoutError::Disconnected)=self.back.recv_timeout(std::time::Duration::from_millis(1)).map(drop) {self.pending.clear();break;}
            self.pending.retain(|p|!p.done.load(Ordering::Acquire));}
        if self.log {eprintln!("{} fence store {tag}: waited {:.1} ms",self.tag,t.elapsed().as_secs_f64()*1e3);}
    }
    /// Flush pending writes (shutdown), up to `secs`.
    pub(crate) fn shutdown(&mut self,secs:u64) {
        let (tx,rx)=mpsc::channel();let _=self.tx.send(Job::Flush(tx));
        let ok=rx.recv_timeout(std::time::Duration::from_secs(secs)).is_ok();self.reap();
        eprintln!("{} rank{} shutdown: writes {}",self.tag,self.rank,if ok {"flushed"} else {"still pending (timeout)"});
    }
    pub(crate) fn stats(&self)->Value {
        let s=&self.stats;json!({"entries":self.entries.len(),"segments":self.segs.len(),"bytes":self.total,"cap":self.cap,
            "pending_writes":self.pending.len(),"saves":s.saves,"skipped_dup":s.skipped_dup,"skipped_budget":s.skipped_budget,"evicted":s.evicted,"restores":s.restores,
            "restore_failures":s.restore_fail,"restored_tokens":s.restored_tokens,"restore_ms":s.restore_ms,"written_bytes":s.written_bytes})
    }

    /// Save checkpoint `which` of prefix `ids` (every rank, right after the checkpoint is taken). `segs`: the store's
    /// segment provenance, extended with the segments written here. `tag`: the store (see `fence`).
    pub(crate) fn save(&mut self,snap:Snapshot,tag:u64,segs:&mut Vec<u64>)->Option<u64> {
        let pc=self;
        pc.reap();
        let ids=snap.ids;let which=snap.which;
        let plen=ids.len();if plen<pc.min {return None;}
        let key=hash_ids(ids);
        pc.clock+=1;let clock=pc.clock;
        if let Some(e)=pc.entries.iter_mut().find(|e|e.key==key&&e.len==plen&&e.which==which) {e.used=clock;pc.stats.skipped_dup+=1;let _=pc.tx.send(Job::Touch(e.path.clone()));return None;}
        let s=pc.seg;let nfull=(plen as i64/s) as usize;
        assert!(snap.rows.iter().all(|r|s%r.div==0),"prefix cache: segment size not a multiple of a row divisor");
        // Provenance: keep the prefix of segments that still exist (an evicted one ends it).
        let valid=segs.iter().take_while(|id|pc.segs.contains_key(id)).count();segs.truncate(valid.min(nfull));
        let fixed_seg=json!({"format":FORMAT,"identity":pc.identity,"binary":pc.binary,"seg":0u64,"start":0i64,"tokens":s});
        // Rows [a, a + n) tokens of every row array, as regions (named after the arrays).
        let row_regions=|a:i64,end:&dyn Fn(&Rows)->i64|->(Vec<Region>,Vec<Src>,Vec<Tensor>){
            let (mut r,mut sr,mut k)=(Vec::new(),Vec::new(),Vec::new());
            for rw in &snap.rows {
                let r0=a/rw.div;let t=rw.t.narrow(0,r0,end(rw)-r0);
                assert!(t.is_contiguous());r.push(Region{name:rw.name.clone(),kind:t.kind(),shape:t.size(),off:0,bytes:nbytes(&t)});sr.push(Src::Dev(t.data_ptr() as usize));k.push(t);
            }
            (r,sr,k)};
        // New segments.
        let mut files=Vec::new();let mut keep=Vec::new();let mut new_segs=Vec::new();let mut bytes=0u64;
        for k in segs.len()..nfull {
            let id=pc.next_seg+new_segs.len() as u64;let a=k as i64*s;
            let (mut regions,srcs,kk)=row_regions(a,&|rw:&Rows|(a+s)/rw.div);
            let mut fixed=fixed_seg.clone();fixed["seg"]=json!(id);fixed["start"]=json!(a);
            let (head,size)=layout(&mut regions,&fixed);bytes+=size;keep.extend(kk);
            files.push(FilePlan{path:pc.dir.join(format!("s-{id:016x}.seg")),head,size,fixed,regions,srcs,state:Arc::new(AtomicU8::new(PENDING)),cancel:Arc::new(AtomicBool::new(false))});
            new_segs.push(id);
        }
        // The snapshot file: ids, the rows after the last full segment, the fixed regions.
        let (mut regions,mut srcs)=(Vec::new(),Vec::new());
        let idb:Vec<u8>=ids.iter().flat_map(|t|t.to_le_bytes()).collect();
        regions.push(Region{name:"ids".into(),kind:Kind::Int64,shape:vec![plen as i64],off:0,bytes:idb.len()});srcs.push(Src::Host(Arc::new(idb)));
        let (r,sr,kk)=row_regions(nfull as i64*s,&|rw:&Rows|rw.end(plen as i64));regions.extend(r);srcs.extend(sr);keep.extend(kk);
        for (name,p) in &snap.fixed {
            match p {
                Part::Dev(t)=>{let t=if t.is_contiguous() {t.shallow_clone()} else {t.contiguous()};
                    regions.push(Region{name:name.clone(),kind:t.kind(),shape:t.size(),off:0,bytes:nbytes(&t)});srcs.push(Src::Dev(t.data_ptr() as usize));keep.push(t);}
                Part::Host{kind,shape,bytes:b}=>{regions.push(Region{name:name.clone(),kind:*kind,shape:shape.clone(),off:0,bytes:b.len()});srcs.push(Src::Host(b.clone()));}
            }
        }
        let all_segs:Vec<u64>=segs.iter().copied().chain(new_segs.iter().copied()).collect();
        let mut fixed=json!({"format":FORMAT,"identity":pc.identity,"binary":pc.binary,"key":format!("{key:032x}"),"len":plen,"which":which,"segs":all_segs,"seg_tokens":s,
            "created":format!("{:016x}",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs())});
        for (k,v) in &snap.meta {fixed[k.as_str()]=v.clone();}
        let (head,size)=layout(&mut regions,&fixed);bytes+=size;
        if bytes>pc.cap {pc.stats.skipped_budget+=1;return None;}
        // Reused segments count for this entry before anything is evicted.
        for id in segs.iter() {pc.segs.get_mut(id).unwrap().refs+=1;}
        let id=pc.next_id;pc.next_id+=1;pc.next_seg+=new_segs.len() as u64;
        let path=pc.dir.join(format!("e-{key:032x}-{which}.snap"));
        let entry_file=FilePlan{path:path.clone(),head,size,fixed,regions,srcs,state:Arc::new(AtomicU8::new(PENDING)),cancel:Arc::new(AtomicBool::new(false))};
        for (f,&sid) in files.iter().zip(&new_segs) {pc.segs.insert(sid,Seg{refs:1,bytes:f.size,path:f.path.clone(),state:f.state.clone(),cancel:f.cancel.clone()});pc.total+=f.size;}
        let eids=(pc.rank==0).then(||pc.ids_buffer(ids));
        let shared=eids.as_ref().is_some_and(|b|b.len()>plen);
        pc.entries.push(Entry{id,key,len:plen,which,bytes:size,segs:all_segs.clone(),used:clock,path,state:entry_file.state.clone(),cancel:entry_file.cancel.clone(),ids:eids});pc.total+=size;
        if pc.rank==0&&!shared {let i=pc.entries.len()-1;pc.share_ids(i,ids);}
        pc.evict_to(0,Some(id));
        // Eviction may have taken this entry's own segments only if it was itself evicted (protected): not possible.
        files.push(entry_file);
        let mut event=std::ptr::null_mut();assert_eq!(unsafe{cudaEventCreateWithFlags(&mut event,2)},0);
        assert_eq!(unsafe{cudaEventRecord(event,rs_current_stream())},0,"prefix cache: record");
        let done=Arc::new(AtomicBool::new(false));pc.pending.push(Pending{tag,done:done.clone()});
        pc.stats.saves+=1;pc.stats.written_bytes+=bytes;
        let _=pc.tx.send(Job::Write{event:event as usize,files,done,keep});
        *segs=all_segs;
        if pc.log {eprintln!("{} save {plen} tokens which {which}: {} new segments, {:.1} MiB, total {:.2} GiB, {} entries",pc.tag,
            new_segs.len(),bytes as f64/1048576.,pc.total as f64/(1u64<<30) as f64,pc.entries.len());}
        Some(id)
    }

    /// Every rank: a request reuses checkpoint `which` of prefix `ids` from memory: refresh its disk entry in the LRU
    /// order (the persistent copy of what is hot in memory must not be the first to go).
    pub(crate) fn touch(&mut self,ids:&[i64],which:usize) {
        if ids.len()<self.min {return;}
        let key=hash_ids(ids);self.clock+=1;let clock=self.clock;
        if let Some(e)=self.entries.iter_mut().find(|e|e.key==key&&e.len==ids.len()&&e.which==which) {e.used=clock;let _=self.tx.send(Job::Touch(e.path.clone()));}
    }
    /// The longest cached entry that is a prefix of `q` (strict, or exact when `exact_ok(which)`), longer than
    /// `longer_than` and at least `min_len`. Returns (entry id, length).
    pub(crate) fn lookup(&self,q:&[i64],longer_than:usize,min_len:usize,exact_ok:impl Fn(usize)->bool)->Option<(u64,usize)> {
        let pc=self;
        let mut cands:Vec<&Entry>=pc.entries.iter().filter(|e|e.len<=q.len()&&(e.len<q.len()||exact_ok(e.which))&&e.len>longer_than&&e.len>=min_len
            &&e.state.load(Ordering::Acquire)==OK&&e.segs.iter().all(|s|pc.segs[s].state.load(Ordering::Acquire)==OK)).collect();
        if cands.is_empty() {return None;}
        cands.sort_by_key(|e|e.len);
        let mut h=PrefixHash::new();let mut at=0usize;let mut best:Option<&Entry>=None;
        for e in cands {while at<e.len {h.push(q[at]);at+=1;}
            if h.digest()==e.key && best.map_or(true,|b|e.len>b.len||(e.len==b.len&&e.which==0)) {best=Some(e);}}
        best.map(|e|(e.id,e.len))
    }
    /// rank0: the longest prefix `q` shares with a cached entry it then diverges from (the fork point), 0 when none. An
    /// entry that is itself a prefix of `q` is a `lookup` hit, not a fork.
    pub(crate) fn fork_point(&self,q:&[i64])->usize {
        self.entries.iter().filter_map(|e|e.ids.as_ref().map(|ids|&ids[..e.len])).map(|ids|common_prefix(q,ids)).filter(|&(l,whole)|!whole&&l<q.len()).map(|x|x.0).max().unwrap_or(0)
    }
    /// rank0: the id buffer of a new entry: an existing one that starts with `ids` (a shorter checkpoint of a cached
    /// lineage), else a new one.
    fn ids_buffer(&self,ids:&[i64])->Arc<Vec<u32>> {
        let mut seen=std::collections::HashSet::new();
        for b in self.entries.iter().filter_map(|e|e.ids.as_ref()) {
            if b.len()<ids.len()||!seen.insert(Arc::as_ptr(b)) {continue;}
            if b.iter().zip(ids).all(|(&x,&t)|x as i64==t) {return b.clone();}
        }
        Arc::new(ids.iter().map(|&t|t as u32).collect())
    }
    /// rank0: entries whose ids are a prefix of `src` (entry `i`'s ids, the tokens of its buffer) take entry `i`'s buffer,
    /// so a lineage keeps its ids once: a prompt of L tokens with a checkpoint every 512 held ~L²/1024 ids, 39 MB at
    /// 100k (10-19). Matched by the entries' prefix hashes in one pass over `src`, like `lookup`.
    fn share_ids<T:Copy+Into<i64>>(&mut self,i:usize,src:&[T]) {
        let Some(buf)=self.entries[i].ids.clone() else {return};
        let mut cands:Vec<usize>=(0..self.entries.len()).filter(|&j|j!=i&&self.entries[j].len<=src.len()
            &&self.entries[j].ids.as_ref().is_some_and(|b|!Arc::ptr_eq(b,&buf))).collect();
        cands.sort_by_key(|&j|self.entries[j].len);
        let mut h=PrefixHash::new();let mut at=0usize;
        for j in cands {let e=&mut self.entries[j];while at<e.len {h.push(src[at].into());at+=1;}
            if h.digest()==e.key {e.ids=Some(buf.clone());}}
    }
    /// Every rank: drop entry `id` after a failed restore, with every entry that shares one of its segments (the failing
    /// part is unknown here: a damaged segment would fail each of them in turn, one cold prefill per request).
    pub(crate) fn drop_entry(&mut self,id:u64) {
        self.stats.restore_fail+=1;
        let Some(segs)=self.entries.iter().find(|e|e.id==id).map(|e|e.segs.clone()) else {return};
        while let Some(i)=self.entries.iter().position(|e|e.id==id||e.segs.iter().any(|s|segs.contains(s))) {self.remove(i);}
        eprintln!("{} rank{} dropped entry {id} and those sharing its {} segments ({} entries left)",self.tag,self.rank,segs.len(),self.entries.len());
    }

    /// Start restoring entry `id` (this rank's files): row arrays straight into `rows` (the target store's, same names
    /// and divisors), fixed regions named in `into` into those tensors, the others into new tensors (on the CPU when
    /// named in `host`, else on `dev`). The device copies run on the serve thread's stream, after whatever initialized
    /// the destinations. `touch`: a real restore (LRU and statistics); false for a read-back self-test.
    pub(crate) fn start(&mut self,id:u64,rows:&[Rows],into:&HashMap<String,Tensor>,host:&[&str],dev:Device,touch:bool)->Loading {
        let t0=Instant::now();
        let mut ld=Loading{id,which:0,plen:0,segs:Vec::new(),meta:Value::Null,owned:HashMap::new(),shared:None,job:None,err:None,t0,waited_ms:0.,bytes:0,touch};
        let pc=self;
        let r=(||->Result<(),String>{
            pc.reap();
            let i=pc.entries.iter().position(|e|e.id==id).ok_or("no such entry")?;
            // This rank's writes of the entry and its segments must have finished (rank0 only picks written entries; the
            // other ranks' copies are normally done too, else this waits for them).
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
            let regs=regions_of(&h).ok_or("bad tensor table")?;
            ld.meta=h;
            if !regs.iter().any(|r|r.0.name=="ids") {return Err("missing ids".into());}
            let by_name:HashMap<&str,&Rows>=rows.iter().map(|r|(r.name.as_str(),r)).collect();
            // Reads: (file, region, sums, destination pointer, host destination). Rows go straight into the store.
            let mut reads:Vec<(PathBuf,Region,Vec<u64>,usize,bool)>=Vec::new();
            let check=|r:&Region,t:&Tensor|->Result<(),String>{if r.kind!=t.kind()||r.shape!=t.size() {Err(format!("{}: {:?} {:?} vs {:?} {:?}",r.name,r.kind,r.shape,t.kind(),t.size()))} else {Ok(())}};
            for (k,sid) in e.segs.iter().enumerate() {
                let sp=pc.segs.get(sid).ok_or("segment gone")?.path.clone();
                let sh=read_header(&sp).ok_or("unreadable segment header")?;
                if sh["identity"].as_str()!=Some(pc.identity.as_str())||sh["seg"].as_u64()!=Some(*sid)||sh["start"].as_i64()!=Some(k as i64*s) {return Err("segment header mismatch".into());}
                for (r,sums) in regions_of(&sh).ok_or("bad segment table")? {
                    let rw=by_name.get(r.name.as_str()).ok_or(format!("bad segment tensor {}",r.name))?;
                    let dst=rw.t.narrow(0,k as i64*s/rw.div,s/rw.div);
                    check(&r,&dst)?;let p=dst.data_ptr() as usize;reads.push((sp.clone(),r,sums,p,false));
                }
            }
            let nfull=e.segs.len() as i64;if nfull*s>plen as i64 {return Err("segments exceed the prefix".into());}
            for (r,sums) in regs {
                let dst=if let Some(rw)=by_name.get(r.name.as_str()) {let r0=nfull*s/rw.div;rw.t.narrow(0,r0,rw.end(plen as i64)-r0)}
                    else if let Some(t)=into.get(&r.name) {t.shallow_clone()}
                    else {let t=if r.name=="ids" {Tensor::empty([plen as i64],(Kind::Int64,Device::Cpu))}
                        else if host.contains(&r.name.as_str()) {Tensor::empty(r.shape.as_slice(),(r.kind,Device::Cpu))} else {Tensor::empty(r.shape.as_slice(),(r.kind,dev))};
                        ld.owned.insert(r.name.clone(),t.shallow_clone());t};
                check(&r,&dst)?;
                let on_host=r.name=="ids"||(!by_name.contains_key(r.name.as_str())&&!into.contains_key(&r.name)&&host.contains(&r.name.as_str()));
                if !dst.is_contiguous() {return Err(format!("{}: destination not contiguous",r.name));}
                reads.push((e.path.clone(),r,sums,dst.data_ptr() as usize,on_host));
            }
            ld.bytes=reads.iter().map(|r|r.1.bytes).sum();
            let sh=Arc::new(LoadShared{stage:pc.stage.clone(),ready:std::sync::Mutex::new(Default::default()),read_done:AtomicBool::new(false),cancel:AtomicBool::new(false)});
            pc.loads.retain(|w|w.strong_count()>0);pc.loads.push(Arc::downgrade(&sh));
            ld.shared=Some(sh.clone());
            let (threads,delay)=(pc.reader_threads,pc.test_delay_ms);
            ld.job=Some(std::thread::Builder::new().name("pcache-reader".into()).spawn(move||{
                let r=read_all(&sh,threads,&reads,delay);if r.is_err() {sh.cancel.store(true,Ordering::Release);}
                sh.read_done.store(true,Ordering::Release);r
            }).map_err(|e|e.to_string())?);
            Ok(())
        })();
        if let Err(e)=r {ld.err=Some(e);}
        ld
    }
    /// Serve thread: move staged chunks of every running restore to the device (`poll`).
    fn pump_all(&mut self) {
        self.loads.retain(|w|w.strong_count()>0);
        for w in &self.loads {if let Some(sh)=w.upgrade() {if !sh.cancel.load(Ordering::Acquire) {let _=sh.pump(false).map_err(|e|{sh.cancel.store(true,Ordering::Release);e});}}}
    }
}
/// Length of the common prefix of `q` and `ids`, and whether it is all of `ids`.
pub(crate) fn common_prefix<T:Copy+Into<i64>>(q:&[i64],ids:&[T])->(usize,bool) {
    let l=q.iter().zip(ids).take_while(|(a,b)|**a==(**b).into()).count();(l,l==ids.len())
}

/// Pinned staging ring for restores: reader threads fill free slots from disk, the serve thread copies filled slots to
/// the device on its own stream between steps and frees them.
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
/// What a finished restore hands to the model's adapter: the token ids and every region not restored in place.
pub(crate) struct Loaded {pub which:usize,pub ids:Vec<i64>,pub owned:HashMap<String,Tensor>,pub meta:Value,pub segs:Vec<u64>}
/// An asynchronous restore: reader threads read and checksum the files into the staging ring in the background; the
/// serve thread moves filled slots into the store rows / new tensors between its steps (`poll`), where the device is idle
/// (copies issued from other threads while sequences decode ran ~30x slower, bench/pcache). The target rows and tensors
/// belong to the loading sequence only; `finish` drains the rest and hands the result over.
pub(crate) struct Loading {
    id:u64,which:usize,plen:usize,segs:Vec<u64>,meta:Value,
    owned:HashMap<String,Tensor>,shared:Option<Arc<LoadShared>>,job:Option<std::thread::JoinHandle<Result<(),String>>>,err:Option<String>,
    t0:Instant,waited_ms:f64,bytes:usize,touch:bool,
}
impl Loading {
    /// A load that failed before it started (e.g. the cache is off).
    pub(crate) fn failed(why:&str)->Self {
        Loading{id:0,which:0,plen:0,segs:Vec::new(),meta:Value::Null,owned:HashMap::new(),shared:None,job:None,err:Some(why.into()),t0:Instant::now(),waited_ms:0.,bytes:0,touch:false}
    }
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
    /// Copy what is left, check the token ids against `prompt` (which must extend them), let `assemble` build the
    /// model's checkpoint from the result, then (success only) refresh the entry. None on any local failure.
    pub(crate) fn finish<T>(mut self,prompt:&[i64],pc:&mut Cache,assemble:impl FnOnce(Loaded)->Result<T,String>)->Option<T> {
        let (id,plen,which)=(self.id,self.plen,self.which);
        let r=(||->Result<T,String>{
            self.drain(false)?;
            let ids=Vec::<i64>::try_from(self.owned.remove("ids").ok_or("missing ids")?).map_err(|e|e.to_string())?;
            if prompt.len()<plen||prompt[..plen]!=ids[..] {return Err("token ids differ from the prompt".into());}
            assemble(Loaded{which,ids,owned:std::mem::take(&mut self.owned),meta:std::mem::take(&mut self.meta),segs:self.segs.clone()})
        })();
        let ms=self.t0.elapsed().as_secs_f64()*1e3;
        match r {
            Ok(r)=>{
                if pc.log {eprintln!("{} rank{} restore {plen} tokens which {which}: {:.1} MiB in {ms:.1} ms (waited {:.1} ms for the write)",
                    pc.tag,pc.rank,self.bytes as f64/1048576.,self.waited_ms);}
                if self.touch {
                    pc.clock+=1;let clock=pc.clock;
                    if let Some(e)=pc.entries.iter_mut().find(|e|e.id==id) {e.used=clock;let _=pc.tx.send(Job::Touch(e.path.clone()));}
                    pc.stats.restores+=1;pc.stats.restored_tokens+=plen as u64;pc.stats.restore_ms+=ms;
                }
                Some(r)}
            Err(m)=>{eprintln!("{} rank{} restore of entry {id} ({plen} tokens) failed: {m}",pc.tag,pc.rank);None}
        }
    }
    /// Abandon (cancelled request): stop the readers, free the staged slots, discard the result.
    pub(crate) fn abandon(mut self) {let _=self.drain(true);}
}
impl Drop for Loading {fn drop(&mut self) {if self.job.is_some() {let _=self.drain(true);}}}
/// Parallel O_DIRECT reads into free staging slots, checksum per chunk; device-bound pieces are queued for the serve
/// thread, host-bound ones are copied to their CPU tensor right here. Adjacent regions of a file share one slot-sized
/// read.
fn read_all(sh:&LoadShared,threads:usize,reads:&[(PathBuf,Region,Vec<u64>,usize,bool)],delay:Option<u64>)->Result<(),String> {
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
    let errs:Vec<String>=std::thread::scope(|sc|{
        let hs:Vec<_>=(0..threads).map(|_|{let next=&next;let plan=&plan;sc.spawn(move||->Result<(),String>{
            loop {
                if sh.cancel.load(Ordering::Acquire) {return Ok(());}
                let c=next.fetch_add(1,Ordering::Relaxed);let Some((p,off,len,pieces))=plan.get(c) else {return Ok(())};
                let slot={let mut f=sh.stage.free.lock().unwrap();
                    loop {if let Some(x)=f.pop() {break x;} if sh.cancel.load(Ordering::Acquire) {return Ok(());}
                        f=sh.stage.cv.wait_timeout(f,std::time::Duration::from_millis(5)).unwrap().0;}};
                if let Some(ms)=delay {std::thread::sleep(std::time::Duration::from_millis(ms));}
                let b=sh.stage.slots[slot].slice(*len);
                let res=(||{files[p].read_exact_at(b,*off).map_err(|e|format!("read {} @{off}: {e}",p.display()))?;
                    let mut dev=Vec::new();
                    for &(so,i,k,n) in pieces {let (_,r,sums,dst,host)=&reads[i];
                        if checksum(&b[so..so+pad(n)])!=sums[k] {return Err(format!("checksum mismatch in {} {} chunk {k}",p.display(),r.name));}
                        if *host {unsafe{std::ptr::copy_nonoverlapping(b.as_ptr().add(so),(*dst as *mut u8).add(k*CHUNK),n);}} else {dev.push((so,*dst+k*CHUNK,n));}}
                    Ok(dev)})();
                match res {
                    Ok(dev) if !dev.is_empty()=>sh.ready.lock().unwrap().push_back((slot,dev)),
                    r=>{sh.stage.free.lock().unwrap().push(slot);sh.stage.cv.notify_all();r?;}
                }
            }})}).collect();
        hs.into_iter().filter_map(|h|h.join().unwrap().err()).collect()});
    if errs.is_empty() {Ok(())} else {Err(errs.join("; "))}
}
/// Every rank: agree on a restore's outcome (one small allreduce). Returns true only if every rank succeeded.
pub(crate) fn agree(ok:bool,dev:Device)->bool {
    let t=Tensor::from_slice(&[if ok {0f32} else {1f32}]).to_device(dev);crate::tp::allreduce(&t);t.double_value(&[0])==0.
}
