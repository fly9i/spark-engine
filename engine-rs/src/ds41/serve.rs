//! `ds41-serve <ckpt> <socket> <engram_hash.json>`: DeepSeek-V4.1-Flash serving loop, TP over the ranks of
//! GLM53_TP_WORLD (docs/07 D5). Same newline-JSON protocol as the GLM / Qwen serve loops, so serve/ds41_server.py
//! reuses the OpenAI front end:
//!   request  {"id","prompt_ids","max_new","stop_token_ids","temperature","seed"[,"mm"]} | {"cancel":id} | {"id","stats":true}
//!            ("mm": image canvases and the salted placeholder runs they fill, crate::mm_canvas; DS41_VISION=1, super::vision)
//!   replies  {"id","queued":true} | {"id","delta":[ids]} | {"id","done":true,"token_ids",...timings}
//! Rank 0 owns the socket and the schedule; every step it sends the same command to the other ranks over a TCP control
//! channel (GLM53_MASTER_ADDR, port GLM53_MASTER_PORT + 31), and all ranks run the same forward (the collectives line
//! up); the logits are identical on every rank after the vocabulary sum, rank 0 samples.
//! v0: no speculative decoding, one token per decoding sequence per step, sequences in turn; prompts in chunks of
//! DS41_PREFILL_CHUNK (2048) interleaved with decode steps. Stores (DS41_SERVE_STORES) keep their committed history: a
//! prompt extending a store's history only prefills the suffix (the state at any position is complete, so the
//! continuation is exact).
//! KV pool (crate::kv_pool, VMM): the context-proportional state of every store (compressed KV and index keys) lives in
//! one budget sized after the warm-up from the memory the allocator does not need (the smallest over the ranks); the
//! allocator's cap is lowered by it, so the engine's footprint stays fixed. A store's lease grows at admission to the
//! request's prompt + max_new (DS41_KV_GRANULE tokens granules); idle stores give theirs back (LRU, their prefixes stay
//! on the NVMe prefix cache) when it runs short, else the request waits. A request may use up to the whole budget
//! (DS41_SERVE_CAP: an optional lower per-request limit; the model's 1M positions at most).
use super::model::{Model, Seq};
use super::pcache;
use crate::pcache_core::{self, Snapshot};

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;
use tch::{Device, Kind, Tensor};

fn env_i64(k: &str, d: i64) -> i64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

/// `segs`: the prefix cache segments holding exactly this store's rows (its lineage, crate::pcache_core).
struct Store { seq: Seq, lease: Option<crate::kv_pool::Lease>, history: Vec<i64>, busy: bool, used: u64, segs: Vec<u64> }

/// Before a store's rows are reset, overwritten or unmapped: no background prefix cache write may still read them.
fn pc_fence(pc: &mut Option<pcache_core::Cache>, stores: &mut [Store], s: usize) {
    if let Some(p) = pc.as_mut() { p.fence(s as u64); }
    stores[s].segs.clear();
}

/// Sets store `s`'s lease to `cap` tokens (grow: new rows zeroed; shrink: only after a reset) and re-attaches its views;
/// every rank applies the same calls in the same order. false: not enough free chunks (nothing changed).
fn kv_set(m: &Model, stores: &mut [Store], s: usize, cap: i64) -> bool {
    let pool = crate::kv_pool::installed().expect("KV pool");
    let st = &mut stores[s];
    let lease = st.lease.as_mut().expect("store without a lease");
    if cap > lease.len { if !pool.grow(lease, cap) { return false; } }
    else if cap < lease.len { pool.shrink(lease, cap); }
    let views = pool.arr_views(lease);
    m.attach_kv(&mut st.seq, &views, lease.len);
    true
}
/// Evicts idle store `s`: its state is reset and its lease shrinks to one granule (its prefix stays on the NVMe cache).
fn kv_evict(m: &Model, pc: &mut Option<pcache_core::Cache>, stores: &mut [Store], s: usize) {
    pc_fence(pc, stores, s);
    m.reset_seq(&mut stores[s].seq);
    stores[s].history.clear();
    let g = crate::kv_pool::installed().expect("KV pool").granule;
    let _ = kv_set(m, stores, s, g);
}

struct Active {
    id: Value, store: usize, prompt: Vec<i64>, done_prompt: usize, max_new: usize, stop_ids: Vec<i64>, temp: f32, seed: u64,
    out: Vec<i64>, sent: usize, hit: usize, t_queue: Instant, queue_ms: f64, prefill_ms: f64, t_first: Option<Instant>, cancelled: bool,
    rounds: usize, drafted: usize, accepted: usize, pf_started: bool,
    /// prefix cache: a checkpoint is saved after this many prompt tokens (where the prompt diverges from a cached or held
    /// prompt: e.g. an agent's shared system prompt and tools before each new conversation's first message)
    fork_at: usize,
    /// where the prefix came from (log)
    src: &'static str,
    /// image input (rank 0): canvases and their encoded spans waiting for the chunks that consume them
    mm: Option<crate::mm_canvas::MmState>,
}

/// A speculative verify in flight on this rank (between the `sverify` and `scommit` steps).
struct SpecPending { sidx: Vec<usize>, rbs: Vec<super::spec::Rollback>, hid: Tensor, offs: Vec<i64>, p0: Vec<i64>, hlen: Vec<usize> }

/// The stores' sequences in ascending store order (the listed ones).
fn seqs_of<'a>(stores: &'a mut [Store], sidx: &[usize]) -> Vec<&'a mut Seq> {
    let mut v = Vec::new();
    for (i, st) in stores.iter_mut().enumerate() { if sidx.contains(&i) { v.push(&mut st.seq); } }
    v
}
fn spec_draft(m: &Model, stores: &mut [Store], sidx: &[usize], toks: &[i64], temps: &[f32], seeds: &[u64]) -> (Vec<Vec<i64>>, Vec<Vec<f32>>) {
    let seqs = seqs_of(stores, sidx);
    m.mtp_draft(&seqs, toks, temps, seeds)
}
fn spec_verify(m: &Model, stores: &mut [Store], sidx: &[usize], rows: &[Vec<i64>]) -> (Tensor, SpecPending) {
    let rbs: Vec<_> = sidx.iter().zip(rows).map(|(&i, r)| m.spec_checkpoint(&stores[i].seq, r.len() as i64)).collect();
    let p0: Vec<i64> = sidx.iter().map(|&i| stores[i].seq.pos).collect();
    let hlen: Vec<usize> = sidx.iter().map(|&i| stores[i].history.len()).collect();
    let (lg, hid) = { let mut seqs = seqs_of(stores, sidx); m.forward_rows(&mut seqs, rows) };
    for (&i, r) in sidx.iter().zip(rows) { stores[i].history.extend_from_slice(r); }
    let offs: Vec<i64> = rows.iter().scan(0i64, |a, r| { let o = *a; *a += r.len() as i64; Some(o) }).collect();
    (lg, SpecPending { sidx: sidx.to_vec(), rbs, hid: hid.expect("no target hidden"), offs, p0, hlen })
}
fn spec_commit(m: &Model, stores: &mut [Store], sp: SpecPending, keep: &[i64]) {
    for (((b, &i), rb), &k) in sp.sidx.iter().enumerate().zip(sp.rbs).zip(keep) {
        m.spec_rollback(&mut stores[i].seq, rb, k);
        stores[i].history.truncate(sp.hlen[b] + k as usize);
        m.mtp_seed(&mut stores[i].seq, &sp.hid.narrow(0, sp.offs[b], k), sp.p0[b]);
    }
}
/// DS41_STEP_STATS=1: every 100 decode steps, the mean time of a scheduler iteration before the decode step
/// (requests, prefill chunks), of the decode step (synchronized) and the whole period between steps, and the batch.
fn step_stats(t_loop: Instant, t_dec: Instant, b: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("DS41_STEP_STATS").is_ok_and(|v| v == "1")) { return; }
    tch::Cuda::synchronize(0);
    thread_local! { static ACC: std::cell::RefCell<(usize, f64, f64, f64, usize, Option<Instant>)> = const { std::cell::RefCell::new((0, 0.0, 0.0, 0.0, 0, None)) }; }
    ACC.with(|a| {
        let mut a = a.borrow_mut();
        let now = Instant::now();
        a.0 += 1;
        a.1 += (t_dec - t_loop).as_secs_f64() * 1e3;
        a.2 += (now - t_dec).as_secs_f64() * 1e3;
        if let Some(p) = a.5 { a.3 += (now - p).as_secs_f64() * 1e3; }
        a.4 += b;
        a.5 = Some(now);
        if a.0 == 100 {
            eprintln!("[ds41-step] 100 steps: batch {:.1}, before {:.2} ms, decode {:.2} ms, period {:.2} ms", a.4 as f64 / 100.0, a.1 / 100.0, a.2 / 100.0, a.3 / 99.0);
            *a = (0, 0.0, 0.0, 0.0, 0, Some(now));
        }
    });
}
/// Speculative decoding only while at most DS41_SPEC_MAX_BATCH (4) sequences decode together: every verified row reads
/// its own experts (~6 ms), so with several sequences plain batched steps give more tokens per second; the plain step
/// keeps the DSpark windows seeded, so the batch can switch either way.
fn spec_max_batch() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("DS41_SPEC_MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(4))
}
/// Verify cost model of a speculative round (DS41_SPEC_COST="base,seq,row" ms, default 22,0.5,2.5: a verify of R rows of B
/// sequences costs about base + seq B + row R; every row reads its own routed experts, every sequence runs its own
/// attention middle): the draft counts that maximize expected tokens per millisecond, with sigmoid(confidence) of a
/// draft as its chance of acceptance given its predecessors; each sequence gets its own count. Only the number of
/// verified drafts changes, never the output. The old 24,0.6,5 was fitted to plain steps of 1, 2, 4 and 8 sequences
/// (29.5 / 35.7 / 46.7 / 64.6 ms); after the decode work of 2026-10-27 (26.3 / 33.5 / 43.7 / 57.0 ms) the extra rows
/// are relatively cheaper, and the row cost was set by service decode_bench (median of 2, prose / structured / code
/// tok/s): c1 59.7 / 113.3 / 100.3 -> 62.1 / 113.5 / 101.6, c2 82.8 / 156.0 / 125.0 -> 86.2 / 157.2 / 132.5,
/// c4 103.8 / 196.5 / 146.3 -> 110.0 / 197.6 / 154.1 (row 3.0: prose c1 63.4 but c2 / code no better; 2.0: prose c1 62.6).
fn spec_cost() -> (f64, f64, f64) {
    static C: std::sync::OnceLock<(f64, f64, f64)> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        let v: Vec<f64> = std::env::var("DS41_SPEC_COST").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
        if v.len() == 3 { (v[0], v[1], v[2]) } else { (22.0, 0.5, 2.5) }
    })
}
/// The acceptance model of choose_drafts: P(draft accepted | its predecessors accepted) = sigmoid(a conf + b), (a, b)
/// per temperature class (greedy, sampled), DS41_SPEC_CALIB="a0,b0;a1,b1" (default 1,0;1,0: the confidence head as is;
/// fit with bench/ds41/spec_calib.py on DS41_SPEC_LOG rounds). Only the number of verified drafts changes.
fn spec_calib(temp: f32) -> (f64, f64) {
    static C: std::sync::OnceLock<[(f64, f64); 2]> = std::sync::OnceLock::new();
    let c = C.get_or_init(|| {
        let mut c = [(1.0, 0.0); 2];
        if let Ok(v) = std::env::var("DS41_SPEC_CALIB") {
            for (i, part) in v.split(';').take(2).enumerate() {
                let x: Vec<f64> = part.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                if x.len() == 2 { c[i] = (x[0], x[1]); }
            }
        }
        c
    });
    c[(temp > 0.0) as usize]
}
fn choose_drafts(confs: &[Vec<f32>], maxd: &[usize], temps: &[f32]) -> Vec<usize> {
    if std::env::var("DS41_SPEC_MIN_P").is_ok() {   // the fixed-threshold rule (comparison)
        return confs.iter().zip(maxd).map(|(c, &m)| super::spec::spec_len(c, m)).collect();
    }
    let (base, seq, row) = spec_cost();
    let b = confs.len();
    // P[s][i]: probability that the first i + 1 drafts of sequence s are all accepted
    let p: Vec<Vec<f64>> = confs.iter().zip(maxd).zip(temps).map(|((c, &m), &t)| {
        let (a, bb) = spec_calib(t);
        let mut acc = 1.0;
        c.iter().take(m).map(|&x| { acc *= 1.0 / (1.0 + (-(a * x as f64 + bb)).exp()); acc }).collect()
    }).collect();
    // every sequence its own count: rows are added (the most likely next draft first) while a row's expected gain beats
    // the round's current rate (the verify graphs depend on the total row count only)
    let mut d = vec![0usize; b];
    let (mut gain, mut cost) = (b as f64, base + (seq + row) * b as f64);
    loop {
        let best = (0..b).filter(|&s| d[s] < p[s].len()).max_by(|&x, &y| p[x][d[x]].partial_cmp(&p[y][d[y]]).unwrap());
        let Some(s) = best else { break };
        let g = p[s][d[s]];
        if (gain + g) / (cost + row) <= gain / cost { break; }
        gain += g; cost += row; d[s] += 1;
    }
    d
}

/// DS41_SPEC_LOG=<file> (rank 0): one JSON line per sequence and round {t, conf (every draft's confidence), d (verified),
/// k (accepted)}, for fitting the acceptance model of choose_drafts (bench/ds41/spec_calib.py)
fn spec_log(temp: f32, conf: &[f32], d: usize, k: usize) {
    static F: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let f = F.get_or_init(|| std::env::var("DS41_SPEC_LOG").ok().map(|p| std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(p).expect("DS41_SPEC_LOG"))));
    if let Some(f) = f { let _ = writeln!(f.lock().unwrap(), "{}", json!({"t": temp, "conf": conf, "d": d, "k": k})); }
}
fn spec_on() -> bool { std::env::var("DS41_SPEC").map_or(true, |v| v != "0") }

struct Conn(std::sync::Arc<std::sync::Mutex<Option<UnixStream>>>);
fn write_line(conn: &mut Conn, v: &Value) {
    if let Some(s) = conn.0.lock().unwrap().as_mut() { let _ = s.write_all(format!("{v}\n").as_bytes()); }
}

pub(super) fn sample(lg: &Tensor, temp: f32, seed: u64, pos: i64) -> i64 {
    if temp <= 0.0 { return lg.argmax(-1, false).int64_value(&[0]); }
    extern "C" { fn rs_gumbel_noise(out: *mut f32, stride: i64, rows: i32, cols: i32, col_offset: i64, temps: *const f32, keys: *const u64) -> i32; }
    let l = lg.to_kind(Kind::Float).contiguous();
    let noise = Tensor::empty_like(&l);
    let temps = [temp];
    let keys = [crate::sampling::key(seed, pos as u64)];
    assert_eq!(unsafe { rs_gumbel_noise(noise.data_ptr().cast(), noise.stride()[0], 1, l.size()[1] as i32, 0, temps.as_ptr(), keys.as_ptr()) }, 0);
    (l + noise).argmax(-1, false).int64_value(&[0])
}

/// Control channel: rank 0 -> every other rank, one JSON line per step.
enum Ctrl { Lead(Vec<TcpStream>), Follow(BufReader<TcpStream>), Solo }
impl Ctrl {
    fn open(rank: usize, world: usize) -> Ctrl {
        if world == 1 { return Ctrl::Solo; }
        let port = env_i64("GLM53_MASTER_PORT", 29961) + 31;
        if rank == 0 {
            let l = TcpListener::bind(("0.0.0.0", port as u16)).expect("ctrl bind");
            let mut peers: Vec<(usize, TcpStream)> = (1..world).map(|_| {
                let (mut s, _) = l.accept().expect("ctrl accept");
                let mut b = [0u8; 1];
                std::io::Read::read_exact(&mut s, &mut b).unwrap();
                s.set_nodelay(true).unwrap();
                (b[0] as usize, s)
            }).collect();
            peers.sort_by_key(|p| p.0);
            Ctrl::Lead(peers.into_iter().map(|p| p.1).collect())
        } else {
            let addr = std::env::var("GLM53_MASTER_ADDR").expect("GLM53_MASTER_ADDR");
            let mut s = (0..600).find_map(|_| TcpStream::connect((addr.as_str(), port as u16)).ok().or_else(|| { std::thread::sleep(std::time::Duration::from_millis(200)); None }))
                .expect("ctrl connect");
            s.set_nodelay(true).unwrap();
            s.write_all(&[rank as u8]).unwrap();
            Ctrl::Follow(BufReader::new(s))
        }
    }
    fn send(&mut self, v: &Value) {
        if let Ctrl::Lead(ps) = self { let line = format!("{v}\n"); for p in ps.iter_mut() { p.write_all(line.as_bytes()).expect("ctrl send"); } }
    }
    fn recv(&mut self) -> Value {
        match self { Ctrl::Follow(r) => { let mut s = String::new(); r.read_line(&mut s).expect("ctrl recv"); serde_json::from_str(&s).expect("ctrl json") } _ => unreachable!() }
    }
}

/// One forward step every rank runs: `ids` appended to store `s` (reset first when asked); returns the logits of the
/// last row.
/// `last`: the prompt's last chunk (logits wanted). With CED (default) prompt chunks run the encoder only and the last
/// one closes with the decoder replay (Model::prefill_ced); the logits of other chunks are not computed.
/// `vis`: the image rows of this chunk (rank 0; crate::mm_canvas::MmState::chunk_rows).
#[allow(clippy::too_many_arguments)]
/// One row-graph / draft-graph warm-up item (DS41_WARM_DEFER: the batched shapes captured after the engine is ready).
#[derive(Clone)]
enum Warm { Draft(usize), Rows(Vec<usize>) }
/// The tokens the warm-up feeds store i.
fn warm_toks(n: usize) -> Vec<i64> { (0..n as i64).map(|i| 1234 + i).collect() }
/// Runs `w` on `stores` (the warm-up's stores, or the small private ones of the deferred items): the draft of B stores,
/// or a verify (plain forward without speculation) of `lens` rows per store, committing one row each.
fn warm_run(m: &Model, stores: &mut [Store], w: &Warm, spec: bool) -> usize {
    let toks = warm_toks(stores.len());
    match w {
        Warm::Draft(b) => {
            let sidx: Vec<usize> = (0..*b).collect();
            spec_draft(m, stores, &sidx, &toks[..*b], &vec![0.0; *b], &vec![0; *b]).0[0].len()
        }
        Warm::Rows(lens) => {
            let b = lens.len();
            let sidx: Vec<usize> = (0..b).collect();
            let rows: Vec<Vec<i64>> = lens.iter().enumerate().map(|(i, &k)| (0..k as i64).map(|j| toks[i] + j).collect()).collect();
            if spec {
                let (_, sp) = spec_verify(m, stores, &sidx, &rows);
                spec_commit(m, stores, sp, &vec![1; b]);
            } else {
                let mut seqs = seqs_of(stores, &sidx);
                let _ = m.forward_rows(&mut seqs, &rows);
            }
            0
        }
    }
}
/// The private stores of the deferred warm-up items (small KV of their own, outside the pool), primed with 64 tokens
/// like the warm-up's.
fn warm_stores(m: &Model, n: usize) -> Vec<Store> {
    let mut st: Vec<Store> = (0..n).map(|_| Store { seq: m.new_seq(1024), lease: None, history: vec![], busy: false, used: 0, segs: vec![] }).collect();
    let ids: Vec<i64> = (0..64).map(|i| 1000 + (i * 7919) % 100000).collect();
    for s in 0..n { let _ = step(m, &mut st, s, &ids, true, true, &[], None); }
    st
}
/// Graph memory set aside per graph of the deferred warm-up when the KV pool is sized (DS41_WARM_GRAPH_MB, 104: its 23
/// graphs, batched row signatures and draft graphs, took 2.0-2.4 GiB per rank).
fn warm_graph_bytes() -> i64 { std::env::var("DS41_WARM_GRAPH_MB").ok().and_then(|v| v.parse::<i64>().ok()).unwrap_or(104) << 20 }

fn step(m: &Model, stores: &mut [Store], s: usize, ids: &[i64], reset: bool, last: bool, ck: &[i64], vis: Option<(Vec<i64>, Tensor)>) -> Tensor {
    if reset { m.reset_seq(&mut stores[s].seq); stores[s].history.clear(); }
    stores[s].seq.ck_req = ck.to_vec();
    stores[s].seq.vis = vis;
    // short pieces through the row graphs (decoder state complete), prefill chunks eagerly (last row's logits only);
    // the DSpark stages' window is seeded from the target hidden states either way. Pieces with image positions always
    // eagerly (the row graphs embed vocabulary ids only); decided from the ids, the same on every rank.
    let img = ids.iter().any(|&t| crate::mm_canvas::is_salted(t));
    let seq = &stores[s].seq;
    let ced = super::model::ced_on() && (seq.dec_pos != seq.pos || ids.len() > 8 || img);
    let (lg, hid) = if ced {
        match m.prefill_ced(&mut stores[s].seq, ids, last) {
            Some(x) => (m.logits(&x), m.last_hidden.borrow_mut().take()),
            None => (Tensor::zeros([1, 1], (tch::Kind::Float, m.dev)), None),
        }
    } else if ids.len() <= 8 && !img {
        let (lg, hid) = m.forward_rows(&mut [&mut stores[s].seq], &[ids.to_vec()]);
        (lg.narrow(0, lg.size()[0] - 1, 1), hid)
    } else {
        let x = m.forward(&mut stores[s].seq, ids, Some(&[ids.len() as i64 - 1]));
        (m.logits(&x), m.last_hidden.borrow_mut().take())
    };
    if let Some(h) = hid { let p0 = stores[s].seq.pos - h.size()[0]; m.mtp_seed(&mut stores[s].seq, &h, p0); }
    stores[s].history.extend_from_slice(ids);
    stores[s].seq.ck_req.clear();
    lg
}
/// Every rank, after a prefill chunk: save the checkpoints captured inside it (ascending), as chunk-end saves would.
fn pc_save_caps(m: &Model, pc: &mut Option<pcache_core::Cache>, stores: &mut [Store], s: usize) {
    let caps = std::mem::take(&mut stores[s].seq.ck_cap);
    let Some(p) = pc.as_mut() else { return };
    for (pos, cap) in caps {
        let Store { seq, history, segs, .. } = &mut stores[s];
        let Some((fixed, meta)) = m.pcache_fixed_at(seq, pos, &cap) else { continue };
        let _ = p.save(Snapshot { ids: &history[..pos as usize], which: 0, rows: m.pcache_rows(seq), fixed, meta }, s as u64, segs);
    }
}

/// One decode token for each listed store (ascending store index) in one batch; logits [B, vocab] in that order.
fn step_batch(m: &Model, stores: &mut [Store], sidx: &[usize], ids: &[i64]) -> Tensor {
    let mut seqs: Vec<&mut Seq> = Vec::new();
    for (i, st) in stores.iter_mut().enumerate() {
        if sidx.contains(&i) { seqs.push(&mut st.seq); }
    }
    let per: Vec<Vec<i64>> = ids.iter().map(|&t| vec![t]).collect();
    let (lg, hid) = m.forward_rows(&mut seqs, &per);
    for (k, &i) in sidx.iter().enumerate() {
        stores[i].history.push(ids[k]);
        if let Some(h) = &hid { let p0 = stores[i].seq.pos - 1; m.mtp_seed(&mut stores[i].seq, &h.narrow(0, k as i64, 1), p0); }
    }
    lg
}

/// Prefix cache restore of entry `id` (holding exactly `ids`) into store `s`, on every rank (its lease already holds
/// them); the ranks agree (allreduce) and on any failure all reset the store and drop the entry.
fn pc_restore(m: &Model, pc: &mut Option<pcache_core::Cache>, stores: &mut [Store], s: usize, id: u64, ids: &[i64]) -> bool {
    let t0 = Instant::now();
    pc_fence(pc, stores, s);
    m.reset_seq(&mut stores[s].seq);
    stores[s].history.clear();
    let Some(p) = pc.as_mut() else { return false };
    let rows = m.pcache_rows(&stores[s].seq);
    let ld = p.start(id, &rows, &std::collections::HashMap::new(), &pcache::HOST, m.dev, true);
    let seq = &mut stores[s].seq;
    let segs = ld.finish(ids, p, |l| { let segs = l.segs.clone(); m.pcache_install(seq, l).map(|_| segs) });
    let all = pcache_core::agree(segs.is_some(), m.dev);
    if all {
        stores[s].history = ids.to_vec();
        stores[s].segs = segs.unwrap();
        eprintln!("[ds41-serve] pcache restore of {} tokens into store {s} in {:.0} ms", ids.len(), t0.elapsed().as_secs_f64() * 1e3);
    } else {
        // (a rank whose restore succeeded resets too: the store must be the same everywhere)
        m.reset_seq(&mut stores[s].seq);
        stores[s].segs.clear();
        p.drop_entry(id);
    }
    all
}
fn pc_save(m: &Model, pc: &mut Option<pcache_core::Cache>, stores: &mut [Store], s: usize) {
    if let Some(p) = pc.as_mut() {
        let Store { seq, history, segs, .. } = &mut stores[s];
        let (fixed, meta) = m.pcache_fixed(seq);
        let _ = p.save(Snapshot { ids: history, which: 0, rows: m.pcache_rows(seq), fixed, meta }, s as u64, segs);
    }
}
/// The prefix cache's startup agreement over the control connections.
fn pc_open(rank: usize, world: usize, ctrl: &mut Ctrl, ckpt: &Path, hash: &Path) -> Option<pcache_core::Cache> {
    /// A follower's control connection as a byte stream: reads through its line reader (nothing buffered is lost)
    struct Buffered<'a>(&'a mut BufReader<TcpStream>);
    impl std::io::Read for Buffered<'_> { fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> { self.0.read(b) } }
    impl std::io::Write for Buffered<'_> {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> { self.0.get_mut().write(b) }
        fn flush(&mut self) -> std::io::Result<()> { self.0.get_mut().flush() }
    }
    match ctrl {
        Ctrl::Solo => pcache::open(rank, world, ckpt, hash, pcache_core::Link::Solo),
        Ctrl::Lead(ps) => pcache::open(rank, world, ckpt, hash, pcache_core::Link::Lead(ps.iter_mut().map(|p| p as &mut dyn pcache_core::Duplex).collect())),
        Ctrl::Follow(r) => { let mut b = Buffered(r); pcache::open(rank, world, ckpt, hash, pcache_core::Link::Follow(&mut b)) }
    }
}

pub fn serve(ckpt: &Path, socket: &Path, hash: &Path) {
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    crate::tp::set_tf32(false);
    let dev = Device::Cuda(0);
    let t0 = Instant::now();
    // image input: the vision encoder on rank 0 (from the original checkpoint), before the memory budget is taken; it
    // loads on the thread that opens that checkpoint, while the shards load
    let want_vision = tp.rank == 0 && super::vision::enabled();
    let cfg_dir = ckpt.to_path_buf();
    let (m, vision) = Model::load_then(ckpt, tp.rank, tp.world, usize::MAX, Some(hash), dev,
                                       move |t| want_vision.then(|| super::vision::Vision::load(t, &cfg_dir)));
    let n_stores = env_i64("DS41_SERVE_STORES", 4) as usize;
    let granule = env_i64("DS41_KV_GRANULE", 256);
    // 8192: two 4096-token sub-chunks per step (Model::prefill_ced, DS41_CED_SUB), their routed experts in one call --
    // the same bits as 4096-token chunks (10-25: cold ~29.6k-token prompts 7345 -> ~7120 ms; 8192 one-piece chunks were
    // slower: 1285 vs 1159 tok/s at 8k, before the sub-chunks)
    let chunk = env_i64("DS41_PREFILL_CHUNK", 8192) as usize;
    let eos: Vec<i64> = std::env::var("DS41_EOS").unwrap_or_else(|_| "1".into()).split(',').map(|v| v.parse().unwrap()).collect();
    // warm-up stores: their own small KV (the pool is sized after the warm-up)
    let warm_cap = (chunk as i64 + 1024 + granule - 1) / granule * granule;
    let mut stores: Vec<Store> = (0..n_stores).map(|_| Store { seq: m.new_seq(warm_cap), lease: None, history: vec![], busy: false, used: 0, segs: vec![] }).collect();
    let mut ctrl = Ctrl::open(tp.rank, tp.world);
    let mut pc = if pcache::requested() { pc_open(tp.rank, tp.world, &mut ctrl, ckpt, hash) } else { None };
    let pc_min = pcache::min_tokens();
    let (al, rs) = crate::tp::graph::memory();
    eprintln!("[ds41-serve] rank{} loaded in {:.0}s; {n_stores} stores; CUDA allocated {:.1} GiB, reserved {:.1} GiB",
              tp.rank, t0.elapsed().as_secs_f64(), al as f64 / (1u64 << 30) as f64, rs as f64 / (1u64 << 30) as f64);
    let spec = spec_on() && m.mtp.is_some();
    // memory budget (ds41/mem.rs): cap, warm up within it, then occupy the rest of the budget; every rank runs the
    // same steps. First the row graphs serving uses (DS41_WARM_GRAPHS, default on; captured now instead of in the first
    // requests): one store with 1..8 rows (prompt tails, verify), B stores with the same 1..drafts+1 rows each (batched
    // decode, uniform verify), and the draft graph of every batch size. Captures empty the allocator's cache, so the
    // largest eager piece, a full prefill chunk, runs last: its peak stays cached as the eager working set.
    super::mem::set_cap(tp.rank);
    let (mut deferred, mut deferred_graphs): (Vec<Warm>, usize) = (Vec::new(), 0);
    {
        let tw = Instant::now();
        super::mem::warming(true);
        let ids: Vec<i64> = (0..chunk as i64).map(|i| 1000 + (i * 7919) % 100000).collect();
        for s in 0..n_stores { let _ = step(&m, &mut stores, s, &ids[..64.min(ids.len())], true, true, &[], None); }
        // the vision encoder at its largest image (rank 0; no collectives): kernels chosen, its working set cached
        if let Some(v) = &vision { let _ = v.span(&Tensor::zeros([1, 1302, 1302, 3], (Kind::Uint8, dev))); }
        let warm_graphs = std::env::var("DS41_WARM_GRAPHS").map_or(true, |v| v != "0");
        // DS41_WARM_DEFER (default on; 0: all now): only the single-store shapes now, the batched ones (B >= 2) when the
        // engine is first idle after it is ready (their graph memory is set aside when the KV pool is sized below)
        let defer = warm_graphs && std::env::var("DS41_WARM_DEFER").map_or(true, |v| v != "0");
        let mut nd = 0;
        // row graphs by total rows: 1..8 (one store: prompt tails, verify) and with speculation up to (drafts + 1) per
        // store for every batch size; the draft graph of every batch size
        let plan_b = |b: usize, nd: usize| -> Vec<Warm> {
            let spec_b = spec && b <= spec_max_batch();
            let mut v = Vec::new();
            if spec_b { v.push(Warm::Draft(b)); }
            v.push(Warm::Rows(vec![1; b]));
            if !warm_graphs { return v; }
            if b == 1 {
                for k in 2..=8 { v.push(Warm::Rows(vec![k])); }
            } else if spec_b {
                // totals b + 1 ..= b (nd + 1): the extra rows spread over the stores
                for extra in 1..=b * nd { v.push(Warm::Rows((0..b).map(|i| 1 + extra / b + usize::from(i < extra % b)).collect())); }
            }
            v
        };
        for b in (1..=n_stores).rev() {
            if !warm_graphs && b < n_stores { break; }
            if defer && b > 1 { continue; }
            let spec_b = spec && b <= spec_max_batch();
            if spec_b { nd = nd.max(warm_run(&m, &mut stores, &Warm::Draft(b), spec)); }
            for w in plan_b(b, nd).iter().filter(|w| !matches!(w, Warm::Draft(_))) { warm_run(&m, &mut stores, w, spec); }
        }
        if defer { deferred = (2..=n_stores).rev().flat_map(|b| plan_b(b, nd)).collect(); }
        // the graphs those items will capture: a draft graph per batch size, a row graph per new total row count (the
        // row graphs are keyed by the total; the startup items captured 1..=8)
        let mut seen: std::collections::HashSet<usize> = plan_b(1, nd).iter().filter_map(|w| match w { Warm::Rows(l) => Some(l.iter().sum()), _ => None }).collect();
        deferred_graphs = deferred.iter().filter(|w| match w { Warm::Draft(_) => true, Warm::Rows(l) => seen.insert(l.iter().sum()) }).count();
        let _ = step(&m, &mut stores, 0, &ids, true, true, &[], None);
        for st in stores.iter_mut() { st.seq = m.new_seq_fixed(); st.history.clear(); }
        tch::Cuda::synchronize(0);
        super::mem::warming(false);
        eprintln!("[ds41-serve] rank{} warm-up in {:.1}s", tp.rank, tw.elapsed().as_secs_f64());
    }
    // loading and the warm-up are done: drop the file pages of the original checkpoint (m.tables maps all its files; rank 0
    // read the vision encoder from them) except the engram tables, which decode reads through the mapping, and return
    // glibc's free pages (GLM53_RELEASE_FILE_CACHE / GLM53_HOST_TRIM, start-ds41.sh turns both on; a later read faults the
    // pages back). The shard directory `ckpt` went to the GPU with O_DIRECT reads and is not mapped.
    let engram: Vec<String> = m.c.engram_layers.iter().flat_map(|l| ["weight", "scale"].map(|x| format!("layers.{l}.engram.embed.{x}"))).collect();
    crate::host_memory::release_files(&m.tables.files_without(&engram));
    crate::host_memory::finish_loading("ds41-warm-up");
    // KV pool: the memory the allocator does not need after the warm-up (the smallest over the ranks, so every rank
    // carves the same leases), less the VMM chunk rounding of every store's arrays; the allocator's cap drops by it
    let arrs = m.kv_arrays();
    let per = crate::kv_pool::KvPool::arr_bytes_per_granule(&arrs, granule);
    let chunk_min: u64 = 2 << 20;
    let slack = (arrs.len() * n_stores) as i64 * chunk_min as i64 * 2;
    // (the deferred warm-up's graphs: their memory stays under the cap, outside the pool)
    let spare = (super::mem::spare() - deferred_graphs as i64 * warm_graph_bytes()).max(0);
    if !deferred.is_empty() {
        eprintln!("[ds41-serve] rank{} deferred warm-up: {} items, {deferred_graphs} graphs, {:.2} GiB set aside", tp.rank, deferred.len(),
                  (deferred_graphs as i64 * warm_graph_bytes()) as f64 / (1u64 << 30) as f64);
    }
    let mine = ((spare - slack).max(0) / per) as f32;
    let units = {
        let t = Tensor::zeros([tp.world as i64], (Kind::Float, dev));
        let _ = t.get(tp.rank as i64).fill_(mine as f64);
        crate::tp::allreduce(&t);
        Vec::<f32>::try_from(t.to_device(Device::Cpu)).unwrap()
    };
    // DS41_KV_TOKENS (1.1M): the pool's size; the rest of the spare memory stays free for the OS and the host side
    // (the allocator's working set grows into its headroom during long prefills)
    let want = (env_i64("DS41_KV_TOKENS", 1_100_000) + granule - 1) / granule;
    let rows = units.iter().fold(f32::MAX, |a, &b| a.min(b)).min(want as f32) as i64 * granule;
    assert!(rows >= granule * n_stores as i64, "KV pool: {rows} tokens on the smallest rank ({units:?} granules of {granule}), below one granule per store");
    let pool = crate::kv_pool::KvPool::with_arrays(dev, &arrs, rows, granule, n_stores, true, chunk_min);
    // the allocator keeps what it needs (in use + the warm-up's eager working set + headroom); the pool takes its share
    // of the spare memory and the rest stays free for the OS
    super::mem::lower_cap(tp.rank, spare.max(pool.vmm_bytes()));
    crate::kv_pool::KvPool::install(pool.clone());
    for s in 0..n_stores {
        stores[s].lease = Some(pool.alloc(granule).expect("KV pool: a granule per store"));
        assert!(kv_set(&m, &mut stores, s, granule));
    }
    let max_pos = 1i64 << 20;
    let cap = env_i64("DS41_SERVE_CAP", i64::MAX).min(rows).min(max_pos);
    eprintln!("[ds41-serve] rank{} KV pool {rows} tokens ({:.2} GiB, {} B per {granule} tokens; ranks {units:?} granules), max context {cap}",
              tp.rank, pool.vmm_bytes() as f64 / (1u64 << 30) as f64, per);
    super::mem::no_ballast();
    super::mem::report(tp.rank);
    super::mem::watch(tp.rank);
    if tp.rank != 0 {
        let mut pending: Option<SpecPending> = None;
        let mut wstores: Vec<Store> = Vec::new();
        loop {
            let v = ctrl.recv();
            match v["op"].as_str() {
                Some("warm") => {
                    let k = v["k"].as_u64().unwrap() as usize;
                    if wstores.is_empty() { wstores = warm_stores(&m, n_stores); }
                    warm_run(&m, &mut wstores, &deferred[k], spec);
                    if k + 1 == deferred.len() { wstores.clear(); }
                }
                Some("fwd") => {
                    let ids: Vec<i64> = v["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
                    let (st, reset) = (v["store"].as_u64().unwrap() as usize, v["reset"].as_bool().unwrap_or(false));
                    if reset { pc_fence(&mut pc, &mut stores, st); }
                    let ck: Vec<i64> = v["ck"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
                    let _ = step(&m, &mut stores, st, &ids, reset, v["last"].as_bool().unwrap_or(true), &ck, None);
                    pc_save_caps(&m, &mut pc, &mut stores, st);
                }
                Some("dec") => {
                    let sidx: Vec<usize> = v["stores"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                    let ids: Vec<i64> = v["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
                    let _ = step_batch(&m, &mut stores, &sidx, &ids);
                }
                Some("pf") => {
                    let ids = |k: &str| -> Vec<i64> { v[k].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect() };
                    m.engram_prefetch(&ids("prev"), &ids("next"));
                }
                Some("sdraft") => {
                    let sidx: Vec<usize> = v["stores"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                    let toks: Vec<i64> = v["toks"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
                    let temps: Vec<f32> = v["temps"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
                    let seeds: Vec<u64> = v["seeds"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
                    let _ = spec_draft(&m, &mut stores, &sidx, &toks, &temps, &seeds);
                }
                Some("sverify") => {
                    let sidx: Vec<usize> = v["stores"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                    let rows: Vec<Vec<i64>> = v["rows"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()).collect();
                    pending = Some(spec_verify(&m, &mut stores, &sidx, &rows).1);
                }
                Some("scommit") => {
                    let keep: Vec<i64> = v["keep"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
                    spec_commit(&m, &mut stores, pending.take().expect("scommit without sverify"), &keep);
                }
                Some("restore") => {
                    let ids: Vec<i64> = v["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
                    let _ = pc_restore(&m, &mut pc, &mut stores, v["store"].as_u64().unwrap() as usize, v["id"].as_u64().unwrap(), &ids);
                }
                Some("save") => pc_save(&m, &mut pc, &mut stores, v["store"].as_u64().unwrap() as usize),
                Some("kv") => assert!(kv_set(&m, &mut stores, v["store"].as_u64().unwrap() as usize, v["cap"].as_i64().unwrap()), "KV pool: follower grow failed"),
                Some("evict") => kv_evict(&m, &mut pc, &mut stores, v["store"].as_u64().unwrap() as usize),
                Some("quit") => { if let Some(p) = pc.as_mut() { p.shutdown(10); } return }
                _ => panic!("ctrl: {v}"),
            }
            if let Some(p) = pc.as_mut() { p.poll(); }
        }
    }
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket).expect("bind socket");
    eprintln!("[ds41-serve] ready on {} (max context {cap})", socket.display());
    let (tx, rx) = mpsc::channel::<String>();
    let shared = std::sync::Arc::new(std::sync::Mutex::new(None::<UnixStream>));
    let mut conn = Conn(shared.clone());
    let conn_gen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    {
        let gen = conn_gen.clone();
        std::thread::spawn(move || {
            for s in listener.incoming() {
                let Ok(s) = s else { continue };
                let Ok(rd) = s.try_clone() else { continue };
                let g = gen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                *shared.lock().unwrap() = Some(s);
                if g > 1 { let _ = tx.send(format!("{{\"connected\":{g}}}")); }
                let tx = tx.clone();
                std::thread::spawn(move || {
                    for line in BufReader::new(rd).lines() { match line { Ok(l) => { if tx.send(l).is_err() { return; } } Err(_) => break } }
                    let _ = tx.send(format!("{{\"disconnected\":{g}}}"));
                });
            }
        });
    }
    let mut queue: VecDeque<(Value, Vec<i64>, usize, Vec<i64>, f32, u64, Instant, Option<crate::mm_canvas::MmState>)> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();
    let mut tick = 0u64;
    let mut turn = 0usize;
    #[allow(clippy::too_many_arguments)]
    let run = |ctrl: &mut Ctrl, pc: &mut Option<pcache_core::Cache>, stores: &mut Vec<Store>, s: usize, ids: &[i64], reset: bool, last: bool, ck: &[i64],
               vis: Option<(Vec<i64>, Tensor)>| -> Tensor {
        ctrl.send(&json!({"op": "fwd", "store": s, "ids": ids, "reset": reset, "last": last, "ck": ck}));
        if reset { pc_fence(pc, stores, s); }
        let lg = step(&m, stores, s, ids, reset, last, ck, vis);
        pc_save_caps(&m, pc, stores, s);
        lg
    };
    // DS41_MIX_DECODE_SHARE (0..0.9, default 0): while a prompt prefills and other sequences decode, the decode steps'
    // share of the time (a prefill chunk of 8192 runs ~1.5 s; with the default one decode step follows each chunk, so a
    // decoding stream gets a token or one speculative round per chunk). With a share f the next chunk waits until the
    // decode steps since the last chunk took f / (1 - f) of its time. Only the timing changes (the chunks and every
    // computation are the same), and a prompt prefilling alone is not affected.
    let mix_share = std::env::var("DS41_MIX_DECODE_SHARE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0).clamp(0.0, 0.9);
    let (mut last_pf_ms, mut dec_since_pf_ms) = (0.0f64, 0.0f64);
    let (mut wk, mut wt, mut wstores) = (0usize, None::<Instant>, Vec::<Store>::new());
    loop {
        let t_loop = Instant::now();
        if let Some(p) = pc.as_mut() { p.poll(); }
        // ---- requests
        loop {
            let line = if active.is_empty() && queue.is_empty() && wk < deferred.len() {
                // idle: the next deferred warm-up item (every rank runs it), unless a request is waiting
                match rx.try_recv() {
                    Ok(l) => l,
                    Err(mpsc::TryRecvError::Empty) => {
                        // (sent first: the followers run the same collectives)
                        ctrl.send(&json!({"op": "warm", "k": wk}));
                        if wk == 0 { wt = Some(Instant::now()); wstores = warm_stores(&m, n_stores); }
                        warm_run(&m, &mut wstores, &deferred[wk], spec);
                        wk += 1;
                        if wk == deferred.len() {
                            wstores.clear();
                            eprintln!("[ds41-serve] deferred warm-up: {} items in {:.1}s", deferred.len(), wt.map_or(0.0, |t| t.elapsed().as_secs_f64()));
                            super::mem::report(tp.rank);
                        }
                        continue;
                    }
                    Err(_) => { ctrl.send(&json!({"op": "quit"})); return }
                }
            } else if active.is_empty() && queue.is_empty() {
                match rx.recv() { Ok(l) => l, Err(_) => { ctrl.send(&json!({"op": "quit"})); return } }
            } else { match rx.try_recv() { Ok(l) => l, Err(mpsc::TryRecvError::Empty) => break, Err(_) => { ctrl.send(&json!({"op": "quit"})); return } } };
            let v: Value = match serde_json::from_str(line.trim()) { Ok(v) => v, Err(e) => { write_line(&mut conn, &json!({"done": true, "error": format!("bad json: {e}")})); continue; } };
            if v.get("shutdown").is_some() { ctrl.send(&json!({"op": "quit"})); if let Some(p) = pc.as_mut() { p.shutdown(10); } return; }
            let gone = v.get("connected").is_some()
                || v.get("disconnected").and_then(|g| g.as_u64()) == Some(conn_gen.load(std::sync::atomic::Ordering::SeqCst) as u64);
            if gone { for a in active.iter_mut() { a.cancelled = true; } queue.clear(); continue; }
            if v.get("disconnected").is_some() { continue; }
            if let Some(c) = v.get("cancel") {
                for a in active.iter_mut() { if &a.id == c { a.cancelled = true; } }
                queue.retain(|q| { if &q.0 == c { write_line(&mut conn, &json!({"id": c, "done": true, "cancelled": true, "token_ids": []})); false } else { true } });
                continue;
            }
            if v.get("stats").is_some() {
                let decoding = active.iter().filter(|a| a.done_prompt >= a.prompt.len()).count();
                let held: i64 = stores.iter().filter(|s| s.busy).map(|s| s.seq.pos).sum();
                let alloc: i64 = stores.iter().map(|s| s.lease.as_ref().map_or(0, |l| l.len)).sum();
                write_line(&mut conn, &json!({"id": v["id"], "stats": {"running": active.len(), "waiting": queue.len(), "decoding": decoding,
                    "max_seqs": n_stores, "stores": n_stores, "max_stores": n_stores, "kv_allocated_tokens": alloc,
                    "kv_active_tokens": held, "kv_budget_tokens": rows, "batch": false, "pcache": pc.as_ref().map_or(Value::Null, |p| p.stats())}}));
                continue;
            }
            let ids: Vec<i64> = v["prompt_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
            let max_new = v["max_new"].as_u64().unwrap_or(256) as usize;
            if ids.is_empty() || ids.len() + max_new > cap as usize {
                write_line(&mut conn, &json!({"id": v["id"], "done": true, "error": format!("invalid request: {} prompt tokens + {} new must fit {} tokens", ids.len(), max_new, cap)}));
                continue;
            }
            // image input: the canvases and their placeholder runs; every id a vocabulary id or one of those placeholders
            let mm = match (v.get("mm").filter(|x| x.as_array().is_some_and(|a| !a.is_empty())), &vision) {
                (None, _) => Ok(None),
                (Some(_), None) => Err("image input is off on this engine (DS41_VISION=1)".to_string()),
                (Some(x), Some(vis)) => crate::mm_canvas::parse_items(x, |it| vis.seg_len(it)).map(Some),
            };
            let mm = mm.and_then(|items| crate::mm_canvas::validate(&ids, m.c.vocab, items.as_deref().unwrap_or(&[])).map(|_| items));
            let mm = match mm {
                Ok(items) => items.map(crate::mm_canvas::MmState::new),
                Err(e) => { write_line(&mut conn, &json!({"id": v["id"], "done": true, "error": format!("invalid request: {e}")})); continue; }
            };
            let stop: Vec<i64> = v["stop_token_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
            let temp = v["temperature"].as_f64().unwrap_or(0.0) as f32;
            let seed = v["seed"].as_u64().unwrap_or(0);
            write_line(&mut conn, &json!({"id": v["id"], "queued": true}));
            queue.push_back((v["id"].clone(), ids, max_new, stop, temp, seed, Instant::now(), mm));
        }
        // ---- admit: the idle store sharing the longest committed prefix, else the least recently used idle store
        while let Some(q) = queue.front() {
            let free: Vec<usize> = (0..stores.len()).filter(|&i| !stores[i].busy).collect();
            if free.is_empty() { break; }
            let prompt = &q.1;
            let best = free.iter().copied().max_by_key(|&i| {
                let h = &stores[i].history;
                let ok = !h.is_empty() && h.len() < prompt.len() && prompt.starts_with(h);
                (if ok { h.len() } else { 0 }, u64::MAX - stores[i].used)
            }).unwrap();
            // KV: the store's lease grows to prompt + max_new; idle stores (LRU) give theirs back until it fits, else
            // the request waits for a running one to finish
            let need = ((prompt.len() + q.2) as i64 + granule - 1) / granule * granule;
            let mut fits = stores[best].lease.as_ref().unwrap().len >= need;
            while !fits {
                if crate::kv_pool::installed().unwrap().can_grow(stores[best].lease.as_ref().unwrap(), need) {
                    ctrl.send(&json!({"op": "kv", "store": best, "cap": need}));
                    fits = kv_set(&m, &mut stores, best, need);
                    assert!(fits, "KV pool: grow failed after can_grow");
                    break;
                }
                let victim = (0..stores.len()).filter(|&i| i != best && !stores[i].busy && stores[i].lease.as_ref().unwrap().len > granule)
                    .min_by_key(|&i| stores[i].used);
                let Some(v) = victim else { break };
                ctrl.send(&json!({"op": "evict", "store": v}));
                kv_evict(&m, &mut pc, &mut stores, v);
            }
            if !fits { break; }
            let h = &stores[best].history;
            let mut hit = if !h.is_empty() && h.len() < prompt.len() && prompt.starts_with(h) { h.len() } else { 0 };
            let mut src = if hit > 0 { "store" } else { "cold" };
            // a longer prefix on the NVMe prefix cache: restore it into the store (all ranks)
            if let Some((id, len)) = pc.as_ref().and_then(|p| p.lookup(prompt, hit, pc_min, |_| false)) {
                let ids = prompt[..len].to_vec();
                ctrl.send(&json!({"op": "restore", "store": best, "id": id, "ids": ids}));
                hit = if pc_restore(&m, &mut pc, &mut stores, best, id, &ids) { src = "disk"; len } else { src = "cold"; 0 };
            }
            // fork point: the longest prefix this prompt shares with a cached prompt or a store's history and then leaves;
            // the prefill saves a checkpoint there, so the next prompt with that prefix restores it (GLM's fork checkpoint)
            let fork = pc.as_ref().map_or(0, |p| p.fork_point(prompt)).max(stores.iter().map(|st| {
                let (l, whole) = crate::pcache_core::common_prefix(prompt, &st.history);
                if whole { 0 } else { l }
            }).max().unwrap_or(0));
            let fork_at = if pc.is_some() && fork >= pc_min && fork > hit && fork + 1 < prompt.len() { fork } else { usize::MAX };
            let (id, prompt, max_new, stop_ids, temp, seed, tq, mm) = queue.pop_front().unwrap();
            tick += 1;
            stores[best].busy = true;
            stores[best].used = tick;
            if hit == 0 { stores[best].history.clear(); }
            active.push(Active { id, store: best, done_prompt: hit, prompt, max_new, stop_ids, temp, seed, out: vec![], sent: 0, hit,
                                 queue_ms: tq.elapsed().as_secs_f64() * 1e3, t_queue: tq, prefill_ms: 0.0, t_first: None, cancelled: false,
                                 rounds: 0, drafted: 0, accepted: 0, pf_started: false, fork_at, src, mm });
        }
        if active.is_empty() { continue; }
        // ---- one prefill chunk (prefilling sequences in turn)
        let pre: Vec<usize> = (0..active.len()).filter(|&i| active[i].done_prompt < active[i].prompt.len() && !active[i].cancelled).collect();
        let decoding = active.iter().any(|a| !a.cancelled && a.done_prompt >= a.prompt.len() && !a.out.is_empty() && a.out.len() < a.max_new
            && !eos.contains(a.out.last().unwrap()) && !a.stop_ids.contains(a.out.last().unwrap()));
        let pf_turn = mix_share == 0.0 || !decoding || dec_since_pf_ms >= mix_share / (1.0 - mix_share) * last_pf_ms;
        if !pre.is_empty() && pf_turn {
            turn = (turn + 1) % pre.len();
            let a = &mut active[pre[turn]];
            let t = Instant::now();
            // prefix cache: the snapshot is taken after all but the last prompt token (an identical prompt later still
            // has one token to run for its logits); the chunk is cut there
            let save_at = if pc.is_some() && a.prompt.len() >= pc_min { a.prompt.len() - 1 } else { usize::MAX };
            // the chunk starting at prompt position d (the next one's too: its engram prefetch must ask for exactly its
            // rows, else layer 1 gathers them synchronously, a cold NVMe read for new text, +0.2-0.3 s at 32k)
            let chunk_at = |d: usize| {
                let mut n = chunk.min(a.prompt.len() - d);
                for cut in [a.fork_at, save_at] { if d < cut && d + n > cut { n = cut - d; } }
                n
            };
            let n = chunk_at(a.done_prompt);
            // checkpoints every DS41_PCACHE_EVERY (512) prompt tokens, captured inside the chunk (no cut: the chunk keeps its
            // size), so a later prompt sharing a prefix restores the last one on its first repeat (each adds the fixed-size
            // state, ~10 MB; the rows are the shared segments). Encoder-only chunks only (the prompt's last chunk replays the
            // decoder and is saved at its end anyway).
            let every = env_i64("DS41_PCACHE_EVERY", 512).max(0) as usize;
            let ck: Vec<i64> = if pc.is_some() && every > 0 && a.prompt.len() >= pc_min && a.done_prompt + n < a.prompt.len() {
                ((a.done_prompt / every + 1) * every..a.done_prompt + n).step_by(every).filter(|&p| p >= pc_min).map(|p| p as i64).collect()
            } else { vec![] };
            let reset = a.done_prompt == 0;
            let ids = a.prompt[a.done_prompt..a.done_prompt + n].to_vec();
            // the first chunk run for this request: its own engram rows load from NVMe in the background too (the
            // tables mostly miss the page cache; layer 14's rows were a synchronous 0.8 s for 4096 new tokens), so
            // they overlap the layers before; then the next chunk's rows load while this chunk runs (every rank)
            if !a.pf_started && ids.len() >= 256 {
                a.pf_started = true;
                let prev = &a.prompt[a.done_prompt.saturating_sub(3)..a.done_prompt];
                ctrl.send(&json!({"op": "pf", "prev": prev, "next": &ids}));
                m.engram_prefetch(prev, &ids);
            }
            let d1 = a.done_prompt + n;
            let nxt = &a.prompt[d1..if d1 < a.prompt.len() { d1 + chunk_at(d1) } else { d1 }];
            if !nxt.is_empty() {
                ctrl.send(&json!({"op": "pf", "prev": &ids[ids.len().saturating_sub(3)..], "next": nxt}));
                m.engram_prefetch(&ids, nxt);
            }
            let (lo, hi) = (a.done_prompt, a.done_prompt + n);
            let vis = match (a.mm.as_mut(), &vision) { (Some(st), Some(v)) => st.chunk_rows(lo, hi, |it, segs| v.encode_segments(it, segs)), _ => None };
            let lg = run(&mut ctrl, &mut pc, &mut stores, a.store, &ids, reset, a.done_prompt + n == a.prompt.len(), &ck, vis);
            a.done_prompt += n;
            if a.done_prompt == save_at || a.done_prompt == a.fork_at {
                ctrl.send(&json!({"op": "save", "store": a.store}));
                pc_save(&m, &mut pc, &mut stores, a.store);
            }
            if a.done_prompt == a.prompt.len() {
                let pos = stores[a.store].seq.pos;
                a.out.push(sample(&lg, a.temp, a.seed, pos));
                a.t_first = Some(Instant::now());
            }
            tch::Cuda::synchronize(0);
            a.prefill_ms += t.elapsed().as_secs_f64() * 1e3;
            (last_pf_ms, dec_since_pf_ms) = (t.elapsed().as_secs_f64() * 1e3, 0.0);
            if std::env::var("DS41_PREFILL_STATS").is_ok_and(|v| v == "1") {
                let (ar, af) = super::mem::alloc_stats();
                eprintln!("[ds41-serve] prefill chunk of {n} rows (last {}): {:.1} ms (allocator retries {ar}, cudaFree {af})", a.done_prompt == a.prompt.len(), t.elapsed().as_secs_f64() * 1e3);
            }
        }
        // ---- one decode step of every decoding sequence, batched (ascending store index on every rank)
        let t_dec = Instant::now();
        let mut dec: Vec<(usize, usize)> = Vec::new();   // (store, active index)
        for (i, a) in active.iter().enumerate() {
            if a.cancelled || a.done_prompt < a.prompt.len() || a.out.is_empty() || a.out.len() >= a.max_new { continue; }
            let last = *a.out.last().unwrap();
            if eos.contains(&last) || a.stop_ids.contains(&last) { continue; }
            dec.push((a.store, i));
        }
        if !dec.is_empty() && spec && dec.len() <= spec_max_batch() {
            // DSpark: draft for every decoding sequence, verify the confident prefix of each in one batch, commit
            dec.sort();
            let sidx: Vec<usize> = dec.iter().map(|d| d.0).collect();
            let toks: Vec<i64> = dec.iter().map(|d| *active[d.1].out.last().unwrap()).collect();
            let temps: Vec<f32> = dec.iter().map(|d| active[d.1].temp).collect();
            let seeds: Vec<u64> = dec.iter().map(|d| active[d.1].seed).collect();
            ctrl.send(&json!({"op": "sdraft", "stores": sidx, "toks": toks, "temps": temps, "seeds": seeds}));
            let (drafts, confs) = spec_draft(&m, &mut stores, &sidx, &toks, &temps, &seeds);
            // draft counts: each sequence its own, by the cost model (choose_drafts)
            let rooms: Vec<usize> = dec.iter().map(|&(st, i)| {
                let a = &active[i];
                (a.max_new - a.out.len()).saturating_sub(1).min((stores[st].seq.cap - stores[st].seq.pos - 1).max(0) as usize)
            }).collect();
            let maxd: Vec<usize> = (0..dec.len()).map(|b| drafts[b].len().min(rooms[b])).collect();
            let chosen = choose_drafts(&confs, &maxd, &temps);
            let rows: Vec<Vec<i64>> = (0..dec.len()).map(|b| {
                let d = chosen[b].min(drafts[b].len());
                let mut r = vec![toks[b]];
                r.extend(&drafts[b][..d]);
                r
            }).collect();
            ctrl.send(&json!({"op": "sverify", "stores": sidx, "rows": rows}));
            let (lg, sp) = spec_verify(&m, &mut stores, &sidx, &rows);
            let mut keep = Vec::new();
            let dq = m.draft_q.borrow_mut().take();
            for (b, &(_, i)) in dec.iter().enumerate() {
                let a = &mut active[i];
                let (off, d) = (sp.offs[b], rows[b].len() - 1);
                // sampled sequences under DS41_SPEC_RS: the rejection-sampling verdict of every row (spec::rs_tokens)
                let ys = match &dq {
                    Some(q) if a.temp > 0.0 && super::spec::rs_on() => Some(m.rs_tokens(&lg.narrow(0, off, d as i64 + 1), &q.select(1, b as i64), &rows[b][1..], a.temp, a.seed, sp.p0[b])),
                    _ => None,
                };
                let mut k = 0;
                let mut emit = Vec::new();
                loop {
                    let y = match &ys { Some(v) => v[k], None => sample(&lg.narrow(0, off + k as i64, 1), a.temp, a.seed, sp.p0[b] + k as i64 + 1) };
                    emit.push(y);
                    if k < d && rows[b][k + 1] == y && !eos.contains(&y) && !a.stop_ids.contains(&y) { k += 1; } else { break; }
                }
                a.rounds += 1;
                a.drafted += d;
                a.accepted += k;
                spec_log(a.temp, &confs[b], d, k);
                if let Some(q) = &dq {
                    super::spec::spec_stats(&lg.narrow(0, off, d as i64 + 1), &q.select(1, b as i64), &rows[b][1..], a.temp,
                                            super::spec::rs_draft_seed(a.seed, a.temp, sp.p0[b]), sp.p0[b], k);
                }
                a.out.extend(emit);
                keep.push(k as i64 + 1);
            }
            ctrl.send(&json!({"op": "scommit", "keep": keep}));
            spec_commit(&m, &mut stores, sp, &keep);
        } else if !dec.is_empty() {
            dec.sort();
            let sidx: Vec<usize> = dec.iter().map(|d| d.0).collect();
            let ids: Vec<i64> = dec.iter().map(|d| *active[d.1].out.last().unwrap()).collect();
            ctrl.send(&json!({"op": "dec", "stores": sidx, "ids": ids}));
            let lg = step_batch(&m, &mut stores, &sidx, &ids);
            for (k, &(st, i)) in dec.iter().enumerate() {
                let pos = stores[st].seq.pos;
                let a = &mut active[i];
                a.out.push(sample(&lg.narrow(0, k as i64, 1), a.temp, a.seed, pos));
            }
        }
        if !dec.is_empty() { step_stats(t_loop, t_dec, dec.len()); dec_since_pf_ms += t_dec.elapsed().as_secs_f64() * 1e3; }
        // ---- deltas, stop conditions, retirement
        let mut i = 0;
        while i < active.len() {
            let a = &mut active[i];
            let mut finished = a.cancelled;
            if let Some(e) = a.out.iter().position(|t| eos.contains(t) || a.stop_ids.contains(t)) { a.out.truncate(e + 1); finished = true; }
            if a.out.len() >= a.max_new { a.out.truncate(a.max_new); finished = true; }
            if a.out.len() > a.sent {
                write_line(&mut conn, &json!({"id": a.id, "delta": &a.out[a.sent..]}));
                a.sent = a.out.len();
            }
            if !finished { i += 1; continue; }
            let a = active.remove(i);
            if super::model::profiling() {
                let mut p = super::model::take_prof();
                p.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
                eprintln!("[ds41-serve] profile (ms): {}", p.iter().map(|(n, ms)| format!("{n} {ms:.1}")).collect::<Vec<_>>().join(", "));
            }
            // the store keeps its state: history (every id run through the model) always equals the committed positions,
            // also after a cancel mid-prompt, so a later request extending it continues exactly
            stores[a.store].busy = false;
            let total = a.t_queue.elapsed().as_secs_f64() * 1e3;
            eprintln!("[ds41-serve] request: prompt {} tokens{}, prefix {} from {}{}, prefill {:.0} ms, TTFT {:.0} ms, {} tokens out; {}",
                a.prompt.len(), a.mm.as_ref().map_or(String::new(), |x| format!(" ({} images, {} image tokens encoded in {:.0} ms)", x.items.len(), x.encoded_tokens, x.encode_ms)),
                a.hit, a.src, if a.fork_at != usize::MAX { format!(", fork checkpoint at {}", a.fork_at) } else { String::new() },
                a.prefill_ms, a.t_first.map_or(0.0, |t| (t - a.t_queue).as_secs_f64() * 1e3), a.out.len(), super::mem::brief());
            let decode_ms = a.t_first.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
            write_line(&mut conn, &json!({"id": a.id, "done": true, "token_ids": a.out, "cancelled": a.cancelled,
                "queue_ms": a.queue_ms, "prefill_ms": a.prefill_ms, "decode_ms": decode_ms, "total_ms": total,
                "prefix_hit_tokens": a.hit, "rounds": if a.rounds > 0 { a.rounds } else { a.out.len() }, "accepted_drafts": a.accepted, "drafted_tokens": a.drafted,
                "mm_encode_ms": a.mm.as_ref().map(|x| x.encode_ms), "mm_encoded_tokens": a.mm.as_ref().map(|x| x.encoded_tokens)}));
        }
    }
}
