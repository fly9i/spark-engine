// SPDX-License-Identifier: MIT
//! `pcache-golden <out_dir>`: GLM prefix cache regression test without the GLM model (synthetic state). Two threads
//! play rank 0 and rank 1 (the cache is thread-local) over a loopback TCP pair: startup agreement, a save of 200 tokens,
//! a save of 300 tokens extending it (shared segments), a duplicate save, a touch, a flush, a restore compared bit by
//! bit, a lookup. Then `<out_dir>/manifest.txt` lists every file of the cache: its header without the build-dependent
//! fields (identity, binary, created) and the SHA-256 of everything after the header. A refactor of the cache must
//! leave the manifest unchanged (run the old and the new binary, diff the manifests; the reference is
//! engine-rs/testdata/pcache-golden-manifest.txt).
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};

fn pattern(n: i64, seed: i64, kind: Kind, dev: Device) -> Tensor {
    // deterministic, thread-safe (no global generator): ((i * 7919 + seed * 104729) mod 2003) / 97 - 10
    ((Tensor::arange(n, (Kind::Int64, dev)) * 7919 + seed * 104729).remainder(2003).to_kind(Kind::Float) / 97.0 - 10.0).to_kind(kind)
}

struct State { kda: Vec<(Tensor, Tensor)>, mla: Vec<(Tensor, Tensor, Tensor, Tensor)>, draft: crate::dflash::Context, last: Tensor, latent: Vec<Tensor>, pools: Vec<Tensor> }

fn state(dev: Device, cap: i64) -> State {
    let (layers, width, dim) = (3i64, 96i64, 32i64);
    let latent = (0..layers).map(|l| pattern(cap * width, 10 + l, Kind::BFloat16, dev).view([cap, width])).collect();
    let pools = (0..layers).map(|l| pattern(cap / 4 * dim, 20 + l, Kind::Float, dev).view([cap / 4, dim])).collect();
    let kda = (0..2).map(|i| (pattern(4 * 8 * 8, 30 + i, Kind::Float, dev).view([4, 8, 8]), pattern(4 * 16, 40 + i, Kind::BFloat16, dev).view([4, 16]))).collect();
    let mla = (0..layers).map(|i| (Tensor::from_slice(&[17 + i]).to_device(dev), pattern(4 * dim, 50 + i, Kind::Float, dev).view([4, dim]),
        pattern(4 * dim, 60 + i, Kind::Float, dev).view([4, dim]), pattern(dim, 70 + i, Kind::Float, dev).view([1, dim]))).collect();
    let kv = (0..2).map(|j| (pattern(16 * 64, 80 + j, Kind::BFloat16, dev).view([16, 64]), pattern(16 * 64, 90 + j, Kind::BFloat16, dev).view([16, 64]))).collect();
    State { kda, mla, draft: crate::dflash::Context::from_parts(7, 3, kv), last: pattern(128, 99, Kind::Float, dev).view([1, 128]), latent, pools }
}

fn parts(s: &State) -> crate::pcache::Parts<'_> {
    crate::pcache::Parts { kda: &s.kda, mla: &s.mla, draft: &s.draft, last: &s.last,
        latent: s.latent.iter().map(|t| t.shallow_clone()).collect(), pools: s.pools.iter().map(|t| t.shallow_clone()).collect() }
}

fn rank_main(rank: usize, mut link: TcpStream, model: PathBuf) -> Result<(), String> {
    let dev = Device::Cuda(0);
    let _g = tch::no_grad_guard();
    crate::pcache::init(rank, 2, &model, &model, std::slice::from_mut(&mut link));
    if !crate::pcache::enabled() { return Err("cache not enabled after init".into()); }
    let cap = 512i64;
    let st = state(dev, cap);
    let ids_b: Vec<i64> = (0..300).map(|i| 1000 + (i * 37) % 5000).collect();
    let ids_a = &ids_b[..200];
    let mut segs = Vec::new();
    let a = crate::pcache::save(ids_a, 1, parts(&st), 0, &mut segs);
    let b = crate::pcache::save(&ids_b, 0, parts(&st), 0, &mut segs);
    let dup = crate::pcache::save(ids_a, 1, parts(&st), 0, &mut segs.clone());
    crate::pcache::touch(ids_a, 1);
    crate::pcache::shutdown(60);
    crate::pcache::poll();
    let (Some(_a), Some(b), None) = (a, b, dup) else { return Err(format!("saves: a {a:?} b {b:?} dup {dup:?}")) };
    // restore B into zeroed rows and compare
    let latent: Vec<Tensor> = st.latent.iter().map(|t| t.zeros_like()).collect();
    let pools: Vec<Tensor> = st.pools.iter().map(|t| t.zeros_like()).collect();
    let mut prompt = ids_b.clone();
    prompt.push(7);
    let r = crate::pcache::restore(b, &prompt, &latent, &pools, dev, true).ok_or("restore failed")?;
    let rows = 300i64;
    for l in 0..latent.len() {
        if !latent[l].narrow(0, 0, rows).equal(&st.latent[l].narrow(0, 0, rows)) { return Err(format!("latent {l} differs")); }
        if !pools[l].narrow(0, 0, (rows + 3) / 4).equal(&st.pools[l].narrow(0, 0, (rows + 3) / 4)) { return Err(format!("pool {l} differs")); }
    }
    let same = r.ids == ids_b && r.which == 0 && r.last.equal(&st.last) && r.draft.equal(&st.draft)
        && r.kda.iter().zip(&st.kda).all(|(x, y)| x.0.equal(&y.0) && x.1.equal(&y.1))
        && r.mla.iter().zip(&st.mla).all(|(x, y)| x.0.equal(&y.0) && x.1.equal(&y.1) && x.2.equal(&y.2) && x.3.equal(&y.3));
    if !same { return Err("restored checkpoint differs".into()); }
    if rank == 0 {
        let hit = crate::pcache::lookup(&prompt, 0, 16);
        if hit != Some((b, 300)) { return Err(format!("lookup: {hit:?}")); }
    }
    eprintln!("[pcache-golden] rank{rank}: ok; stats {}", crate::pcache::stats());
    // both ranks flushed: rank 1 keeps a copy of the cache for the manifest (rank 0 signals over the link, framed like
    // the cache's own handshake so the next init reads cleanly)
    {
        use std::io::{Read, Write};
        if rank == 0 { link.write_all(&0u64.to_le_bytes()).map_err(|e| e.to_string())?; }
        else {
            let mut n = [0u8; 8];
            link.read_exact(&mut n).map_err(|e| e.to_string())?;
            let root = PathBuf::from(std::env::var("GLM53_PCACHE_DIR").unwrap());
            let st = std::process::Command::new("cp").arg("-a").arg(&root).arg(root.with_file_name("cache-phase1")).status().map_err(|e| e.to_string())?;
            if !st.success() { return Err("copy of the cache failed".into()); }
        }
    }
    // restart: rank 1 loses the segment only B has; after the startup agreement both ranks keep A only
    if rank == 1 {
        let dir = std::fs::read_dir(std::env::var("GLM53_PCACHE_DIR").unwrap()).unwrap().flatten().find(|e| e.path().is_dir()).unwrap().path().join("rank1");
        let mut segs: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "seg")).collect();
        segs.sort();
        std::fs::remove_file(segs.last().unwrap()).map_err(|e| e.to_string())?;
    }
    crate::pcache::init(rank, 2, &model, &model, std::slice::from_mut(&mut link));
    let (hit, n) = crate::pcache::lookup(&prompt, 0, 16).ok_or("lookup after restart: none")?;
    if n != 200 { return Err(format!("lookup after restart: {n} tokens")); }
    let latent: Vec<Tensor> = st.latent.iter().map(|t| t.zeros_like()).collect();
    let pools: Vec<Tensor> = st.pools.iter().map(|t| t.zeros_like()).collect();
    let r = crate::pcache::restore(hit, &prompt, &latent, &pools, dev, true).ok_or("restore after restart failed")?;
    for l in 0..latent.len() {
        if !latent[l].narrow(0, 0, 200).equal(&st.latent[l].narrow(0, 0, 200)) || !pools[l].narrow(0, 0, 50).equal(&st.pools[l].narrow(0, 0, 50)) {
            return Err(format!("layer {l} differs after restart"));
        }
    }
    if r.ids[..] != ids_b[..200] || r.which != 1 { return Err("restored A differs after restart".into()); }
    eprintln!("[pcache-golden] rank{rank}: restart ok; stats {}", crate::pcache::stats());
    Ok(())
}

fn manifest(root: &Path) -> String {
    let mut files: Vec<PathBuf> = Vec::new();
    fn walk(d: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(r) = std::fs::read_dir(d) { for e in r.flatten() { let p = e.path(); if p.is_dir() { walk(&p, out) } else { out.push(p) } } }
    }
    walk(root, &mut files);
    // the identity directory's name depends on the binary: list by rank directory and file name
    let rel = |p: &Path| { let c: Vec<String> = p.components().rev().take(2).map(|c| c.as_os_str().to_string_lossy().to_string()).collect(); format!("{}/{}", c[1], c[0]) };
    files.retain(|p| p.extension().is_some_and(|e| e == "seg" || e == "snap"));
    files.sort_by_key(|p| rel(p));
    let mut out = String::new();
    for p in files {
        let b = std::fs::read(&p).unwrap();
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        let mut h: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
        for k in ["identity", "binary", "created"] { h.as_object_mut().unwrap().remove(k); }
        let head = h["tensors"].as_array().and_then(|t| t.first()).and_then(|t| t["off"].as_u64()).unwrap_or(b.len() as u64) as usize;
        use sha2::Digest;
        let d = sha2::Sha256::digest(&b[head..]);
        out += &format!("{} size {} head {} data {}\n  {}\n", rel(&p), b.len(), head, d.iter().map(|x| format!("{x:02x}")).collect::<String>(), h);
    }
    out
}

pub fn run(out: &Path) {
    let root = out.join("cache");
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(out.join("cache-phase1"));
    let model = out.join("model");
    std::fs::create_dir_all(&model).unwrap();
    for (k, v) in [("GLM53_PCACHE_DIR", root.to_str().unwrap()), ("GLM53_PCACHE_SEG", "64"), ("GLM53_PCACHE_MIN", "16"), ("GLM53_PCACHE_FREE_GIB", "0"),
                   ("GLM53_PCACHE_STAGE_MB", "32"), ("GLM53_PCACHE_GIB", "1")] {
        std::env::set_var(k, v);
    }
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let m1 = model.clone();
    let h = std::thread::spawn(move || rank_main(1, TcpStream::connect(addr).unwrap(), m1));
    let (s0, _) = l.accept().unwrap();
    let r0 = rank_main(0, s0, model.clone());
    let r1 = h.join().unwrap();
    let man = format!("== after the saves\n{}== after the restart\n{}", manifest(&out.join("cache-phase1")), manifest(&root));
    std::fs::File::create(out.join("manifest.txt")).unwrap().write_all(man.as_bytes()).unwrap();
    println!("rank0 {r0:?}\nrank1 {r1:?}\n{} files in the manifest ({})", man.lines().filter(|l| !l.starts_with("  ")).count(), out.join("manifest.txt").display());
    if r0.is_err() || r1.is_err() { std::process::exit(1); }
}
