//! `qwen-serve <model_dir> <socket>`: single-GPU serving loop for Qwen3.8 (same newline-JSON protocol as the
//! GLM serve loop, so serve/qwen_server.py reuses the OpenAI front end).
//!   request  {"id","prompt_ids","max_new","stop_token_ids","temperature","seed"} | {"cancel":id} | {"id","stats":true}
//!   replies  {"id","queued":true} | {"id","delta":[ids]} (each round) | {"id","done":true,"token_ids",...timings}
//! Up to QWEN_SERVE_MAX_SEQS (default 8) sequences are active. Each turn runs one prefill chunk (prefilling sequences in turn)
//! and one speculative round of all decoding sequences together: graph::BatchG per set of stores (singletons captured
//! at start-up, larger sets on first use) runs the batched verify, commit and MTP draft chain; the host syncs once
//! for the verify tokens and once for the next drafts. QWEN_SERVE_GRAPH=0: eager, one sequence at a time.
//! Stores (QWEN_SERVE_STORES, capacity QWEN_SERVE_CAP tokens) keep their committed history; a prompt that extends a
//! store's history only prefills the suffix (the recurrent GDN state cannot rewind, so only exact prefixes reuse).
use super::graph::{BatchG, Policy};
use std::collections::HashMap;
use super::model::{prefill_macro, Ckpt, Model, Seq};
use super::spec::EOS;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;
use tch::{Device, Kind, Tensor};

fn env_i64(k: &str, d: i64) -> i64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

/// A sequence store: its range [base, base + len) of the KV pool (len 0: none), the committed history for prefix reuse.
/// `ckpt`: the prompt checkpoint of its last request (see `prefill_step`), valid while `ckpt.ids` is not empty.
struct Store { seq: Option<Seq>, history: Vec<i64>, pend_h: Option<Tensor>, busy: bool, used: u64, base: i64, len: i64, ckpt: Option<Ckpt> }
impl Store {
    fn seq(&self) -> &Seq { self.seq.as_ref().expect("store without a KV range") }
    fn seq_mut(&mut self) -> &mut Seq { self.seq.as_mut().expect("store without a KV range") }
    fn drop_ckpt(&mut self) { if let Some(c) = self.ckpt.as_mut() { c.ids.clear(); } }
    /// Give up the KV range (and with it the history and the checkpoint).
    fn release(&mut self) { self.seq = None; self.history.clear(); self.pend_h = None; self.len = 0; self.ckpt = None; }
}

/// Positions reserved for the output when a request is admitted: min(max_new, QWEN_SERVE_RESERVE, default 32768). A longer
/// output grows the range while it decodes (`grow`), so a client's max_tokens near the context limit does not take the
/// whole pool (and turn YaRN on) for every request.
fn reserve(max_new: usize) -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    max_new.min(*R.get_or_init(|| env_i64("QWEN_SERVE_RESERVE", 32768).max(256) as usize))
}

/// Prompt checkpoints (QWEN_SERVE_CKPT=0: off) for prompts of at least QWEN_SERVE_CKPT_MIN (default 1024) tokens before the
/// boundary.
fn ckpt_min() -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *R.get_or_init(|| if std::env::var("QWEN_SERVE_CKPT").as_deref() == Ok("0") { usize::MAX } else { env_i64("QWEN_SERVE_CKPT_MIN", 1024) as usize })
}
const IM_START: i64 = 248045;

/// Give store `s` a range of `need` positions keeping its contents: in place when the positions after its range are free or
/// held by idle stores (which give theirs up), else moved to a free gap (idle stores give theirs up, least recently used
/// first; its rows are copied). The RoPE mode cannot change (the cached keys carry it). False when neither fits beside the
/// busy stores.
#[allow(clippy::too_many_arguments)]
fn grow(m: &Model, pool: &super::model::KvPool, stores: &mut [Store], s: usize, need: i64, total: i64,
        graphs: &mut HashMap<Vec<usize>, BatchG>, graph_use: &mut HashMap<Vec<usize>, u64>) -> bool {
    let (base, len) = (stores[s].base, stores[s].len);
    if len >= need { return true; }
    if len == 0 || super::model::yarn_for(need - 16) != super::model::yarn_for(len - 16) { return false; }
    let overlaps = |st: &Store, b: i64, l: i64| st.len > 0 && st.base < b + l && st.base + st.len > b;
    let blockers: Vec<usize> = (0..stores.len()).filter(|&j| j != s && overlaps(&stores[j], base, need)).collect();
    let (nb, give, copy) = if base + need <= total && blockers.iter().all(|&j| !stores[j].busy) {
        (base, blockers, false)
    } else {
        // moved: a gap beside every other range, its own included (the copy must not overlap)
        let mut idle: Vec<usize> = (0..stores.len()).filter(|&j| j != s && !stores[j].busy && stores[j].len > 0).collect();
        idle.sort_by_key(|&j| stores[j].used);
        let mut give: Vec<usize> = Vec::new();
        let mut gap = find_gap(stores, usize::MAX, total, need);
        while gap.is_none() && !idle.is_empty() {
            give.push(idle.remove(0));
            let saved: Vec<i64> = give.iter().map(|&j| std::mem::replace(&mut stores[j].len, 0)).collect();
            gap = find_gap(stores, usize::MAX, total, need);
            for (&j, l) in give.iter().zip(saved) { stores[j].len = l; }
        }
        let Some(nb) = gap else { return false };
        (nb, give, true)
    };
    tch::Cuda::synchronize(0);
    for &j in give.iter().chain([&s]) {
        graphs.retain(|k, _| !k.contains(&j));
        graph_use.retain(|k, _| !k.contains(&j));
    }
    for &j in &give { stores[j].release(); }
    m.reseat(stores[s].seq_mut(), pool, nb, need, copy);
    stores[s].base = nb;
    stores[s].len = need;
    eprintln!("[qwen-serve] store {s} grows to [{nb}, {}) ({need} tokens{}){}", nb + need, if copy { ", moved" } else { "" },
              if give.is_empty() { String::new() } else { format!("; stores {give:?} gave theirs up") });
    true
}

/// KV ranges are handed out in multiples of this many positions.
const GRANULE: i64 = 16384;
/// Pool range a request of `prompt + max_new` tokens needs (16 positions for the MTP draft steps past the end, 16 to spare).
fn range_for(tokens: usize) -> i64 { ((tokens as i64 + 32) + GRANULE - 1) / GRANULE * GRANULE }

/// First gap of `len` positions in the pool beside the ranges of the other stores.
fn find_gap(stores: &[Store], skip: usize, total: i64, len: i64) -> Option<i64> {
    let mut used: Vec<(i64, i64)> = stores.iter().enumerate().filter(|(i, s)| *i != skip && s.len > 0).map(|(_, s)| (s.base, s.len)).collect();
    used.sort_unstable();
    let mut at = 0;
    for (b, l) in used {
        if b - at >= len { return Some(at); }
        at = at.max(b + l);
    }
    (total - at >= len).then_some(at)
}

enum Phase {
    Prefill { done: usize, mtp_pend: Option<(Tensor, i64)>, last: Option<Tensor> },
    Decode { next: i64, drafts: Vec<i64>, confs: Vec<f64> },
}

struct Active {
    id: Value, store: usize, prompt: Vec<i64>, max_new: usize, stop_ids: Vec<i64>, temp: f32, seed: u64,
    phase: Phase, out: Vec<i64>, sent: usize, t_queue: Instant, t_start: Instant, prefill_ms: f64, queue_ms: f64,
    hit: usize, rounds: usize, drafted: usize, accepted: usize, cancelled: bool,
    look: super::lookup::Lookup, look_used: bool, look_prop: Option<Vec<i64>>,
    mm: Option<super::vision::MmState>,
}

/// The front end's connection. A new connection replaces it (a front-end restart keeps the engine, its stores and graphs);
/// the requests of a replaced or closed connection are cancelled.
struct Conn(std::sync::Arc<std::sync::Mutex<Option<UnixStream>>>);
fn write_line(conn: &mut Conn, v: &Value) {
    if let Some(s) = conn.0.lock().unwrap().as_mut() { let _ = s.write_all(format!("{v}\n").as_bytes()); }
}

fn sample_rows(lg: &Tensor, temp: f32, seed: u64, pos0: i64) -> Vec<i64> {
    if temp <= 0.0 { return Vec::<i64>::try_from(lg.argmax(-1, false).to_device(Device::Cpu)).unwrap(); }
    extern "C" { fn rs_gumbel_noise(out: *mut f32, stride: i64, rows: i32, cols: i32, col_offset: i64, temps: *const f32, keys: *const u64) -> i32; }
    let l = lg.to_kind(Kind::Float).contiguous();
    let r = l.size()[0];
    let noise = Tensor::empty_like(&l);
    let temps = vec![temp; r as usize];
    let keys: Vec<u64> = (0..r).map(|i| crate::sampling::key(seed, (pos0 + i) as u64)).collect();
    assert_eq!(unsafe { rs_gumbel_noise(noise.data_ptr().cast(), noise.stride()[0], r as i32, l.size()[1] as i32, 0, temps.as_ptr(), keys.as_ptr()) }, 0);
    Vec::<i64>::try_from((l + noise).argmax(-1, false).to_device(Device::Cpu)).unwrap()
}

pub fn serve(dir: &Path, socket: &Path) {
    let _guard = tch::no_grad_guard();
    let pol = Policy::from_env();
    let k = pol.k;
    let max_seqs = env_i64("QWEN_SERVE_MAX_SEQS", 8).clamp(1, 8) as usize;
    let n_stores = env_i64("QWEN_SERVE_STORES", 8).max(max_seqs as i64) as usize;
    // KV pool (QWEN_KV_TOKENS, default 1048576 = the former 8 stores x 131072, the same memory): every store gets an equal
    // share at start-up; a request that needs more takes a larger range (idle stores give theirs up), up to the whole pool
    let total = (env_i64("QWEN_KV_TOKENS", 1 << 20) / GRANULE).max(n_stores as i64) * GRANULE;
    let share = total / GRANULE / n_stores as i64 * GRANULE;
    let cap = total - 32;   // longest request (prompt + new tokens)
    let t0 = Instant::now();
    let m = Model::load(dir);
    let pool = m.new_kv_pool(total);
    if super::pcache::requested() { super::pcache::init(dir); }
    eprintln!("[qwen-serve] loaded in {:.1}s; KV pool {} tokens ({} stores of {} at start), {} concurrent, drafts {:?}", t0.elapsed().as_secs_f64(),
              total, n_stores, share, max_seqs, pol);
    let mut stores: Vec<Store> = (0..n_stores).map(|i| Store { seq: Some(m.new_seq_in(&pool, i as i64 * share, share)), history: vec![], pend_h: None,
                                                               busy: false, used: 0, base: i as i64 * share, len: share, ckpt: None }).collect();
    // warm-up: one short generation (cuBLAS handles, allocator pools)
    {
        let mut st = super::spec::Stats::default();
        let _ = super::spec::generate(&m, stores[0].seq_mut(), &[248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198], 8, k, &mut st);
        m.reset(stores[0].seq_mut());
    }
    let (al, rs) = super::graph::cuda_mem();
    eprintln!("[qwen-serve] before graphs: CUDA allocated {:.1} GiB, reserved {:.1} GiB", al as f64 / (1u64 << 30) as f64, rs as f64 / (1u64 << 30) as f64);
    let graphs_on = m.mtp.is_some() && std::env::var("QWEN_SERVE_GRAPH").map_or(true, |v| v != "0");
    let mut batch_graphs: HashMap<Vec<usize>, BatchG> = HashMap::new();
    let mut graph_use: HashMap<Vec<usize>, u64> = HashMap::new();
    if graphs_on {
        let t = Instant::now();
        for (i, st) in stores.iter().enumerate() {
            let p1 = pol.for_batch(1);
            let mut bg = BatchG::capture(&m, &[st.seq()], p1.cap());
            for kk in 1..p1.cap() { bg.prepare(&m, &[st.seq()], kk + 1); }   // every verify width of a single sequence
            batch_graphs.insert(vec![i], bg);
        }
        let (al, rs) = super::graph::cuda_mem();
        eprintln!("[qwen-serve] captured decode graphs for {} stores in {:.1}s (CUDA allocated {:.1} GiB, reserved {:.1} GiB)",
                  n_stores, t.elapsed().as_secs_f64(), al as f64 / (1u64 << 30) as f64, rs as f64 / (1u64 << 30) as f64);
    }
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket).expect("bind socket");
    eprintln!("[qwen-serve] ready on {}", socket.display());
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
    let mut queue: VecDeque<(Value, Vec<i64>, usize, Vec<i64>, f32, u64, Instant, Vec<super::vision::MmItem>)> = VecDeque::new();
    let mut active: Vec<Active> = Vec::new();
    let mut tick: u64 = 0;
    let mut turn = 0usize;
    loop {
        super::pcache::poll();
        // ---- requests
        loop {
            let line = if active.is_empty() && queue.is_empty() {
                match rx.recv() { Ok(l) => l, Err(_) => return }
            } else { match rx.try_recv() { Ok(l) => l, Err(mpsc::TryRecvError::Empty) => break, Err(_) => return } };
            let v: Value = match serde_json::from_str(line.trim()) { Ok(v) => v, Err(e) => { write_line(&mut conn, &json!({"done": true, "error": format!("bad json: {e}")})); continue; } };
            if v.get("shutdown").is_some() { super::pcache::shutdown(10); return; }
            // a new front end (or the current one gone): its predecessor's requests have no client any more
            let gone = v.get("connected").is_some()
                || v.get("disconnected").and_then(|g| g.as_u64()) == Some(conn_gen.load(std::sync::atomic::Ordering::SeqCst) as u64);
            if gone {
                for a in active.iter_mut() { a.cancelled = true; }
                if !queue.is_empty() || !active.is_empty() { eprintln!("[qwen-serve] front end reconnected/closed: cancelled {} active, {} queued", active.len(), queue.len()); }
                queue.clear();
                continue;
            }
            if v.get("disconnected").is_some() { continue; }
            if let Some(c) = v.get("cancel") {
                for a in active.iter_mut() { if &a.id == c { a.cancelled = true; } }
                queue.retain(|q| { if &q.0 == c { write_line(&mut conn, &json!({"id": c, "done": true, "cancelled": true, "token_ids": []})); false } else { true } });
                continue;
            }
            if v.get("stats").is_some() {
                // the fields of the front end's /metrics (as GLM's engine reports them); KV in tokens of store ranges
                let held = |busy_only: bool| -> i64 { stores.iter().filter(|s| s.busy || !busy_only).map(|s| s.len).sum() };
                let decoding = active.iter().filter(|a| matches!(a.phase, Phase::Decode { .. })).count();
                write_line(&mut conn, &json!({"id": v["id"], "stats": {"active": active.len(), "queued": queue.len(), "running": active.len(),
                    "waiting": queue.len(), "decoding": decoding, "max_seqs": max_seqs, "stores": stores.iter().filter(|s| s.len > 0).count(),
                    "max_stores": n_stores, "capacity": cap, "kv_pool_tokens": total, "kv_budget_tokens": total,
                    "kv_allocated_tokens": held(false), "kv_active_tokens": held(true), "batch": graphs_on, "pcache": super::pcache::stats()}}));
                continue;
            }
            let ids: Vec<i64> = v["prompt_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
            let max_new = v["max_new"].as_u64().unwrap_or(256) as usize;
            if ids.is_empty() || range_for(ids.len() + reserve(max_new)) > total {
                write_line(&mut conn, &json!({"id": v["id"], "done": true, "error": format!("invalid request: {} prompt tokens + {} new must fit {} tokens", ids.len(), max_new, cap)}));
                continue;
            }
            // images / videos: canvas files and the placeholder runs of the (salted) prompt ids
            let items = match v.get("mm") {
                None => Vec::new(),
                Some(mm) if m.vision.is_none() => { let _ = mm; write_line(&mut conn, &json!({"id": v["id"], "done": true, "error": "image/video input needs QWEN_VISION=1"})); continue; }
                Some(mm) => match super::vision::parse_items(mm).and_then(|it| super::vision::validate(&ids, m.vocab, &it).map(|_| it)) {
                    Ok(it) => it,
                    Err(e) => { write_line(&mut conn, &json!({"id": v["id"], "done": true, "error": format!("invalid mm request: {e}")})); continue; }
                },
            };
            if items.is_empty() && ids.iter().any(|&t| t >= super::vision::MM_BASE) {
                write_line(&mut conn, &json!({"id": v["id"], "done": true, "error": "placeholder ids without mm items"}));
                continue;
            }
            let stops: Vec<i64> = v["stop_token_ids"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
            let temp = v["temperature"].as_f64().unwrap_or(0.0) as f32;
            let seed = v["seed"].as_u64().unwrap_or(0);
            write_line(&mut conn, &json!({"id": v["id"], "queued": true}));
            queue.push_back((v["id"].clone(), ids, max_new, stops, temp, seed, Instant::now(), items));
        }
        // ---- admission: longest exact-prefix store whose range fits, else a free store whose range fits and has the same RoPE
        // mode, else a new range for the least recently used free store (other idle stores give up theirs, least recently used
        // first, until a gap fits); a request that does not fit beside the busy stores waits
        super::graph::side_join();   // a retired store's last commit may still run on the side stream
        while active.len() < max_seqs && !queue.is_empty() {
            let need = range_for(queue[0].1.len() + reserve(queue[0].2));
            let mode = super::model::yarn_for(need - 16);
            let free: Vec<usize> = (0..n_stores).filter(|&i| !stores[i].busy).collect();
            // the longest prefix a free store holds: its committed history, or its prompt checkpoint (`via` 1 / 2); a store
            // that holds it but is too small grows when that fits beside the busy stores
            let mut best: Option<(usize, usize, u8)> = None;
            {
                let q = &queue[0].1;
                let pre = |h: &[i64]| !h.is_empty() && h.len() < q.len() && q[..h.len()] == h[..];
                for &i in &free {
                    let st = &stores[i];
                    if st.len == 0 { continue; }
                    if pre(&st.history) && best.map_or(true, |b| st.history.len() > b.1) { best = Some((i, st.history.len(), 1)); }
                    if let Some(c) = &st.ckpt { if pre(&c.ids) && best.map_or(true, |b| c.ids.len() > b.1) { best = Some((i, c.ids.len(), 2)); } }
                }
            }
            if let Some((i, _, _)) = best {
                if !grow(&m, &pool, &mut stores, i, need, total, &mut batch_graphs, &mut graph_use) { best = None; }
            }
            let free: Vec<usize> = (0..n_stores).filter(|&i| !stores[i].busy).collect();
            let pick = best.map(|b| b.0).or_else(|| {
                // with no history to extend, prefer a store whose set with the busy ones has graphs already (each new
                // set of stores needs its own captures), then the least recently used
                let busy: Vec<usize> = (0..n_stores).filter(|&i| stores[i].busy).collect();
                let captured = |i: usize| { let mut k = busy.clone(); k.push(i); k.sort_unstable(); batch_graphs.contains_key(&k) };
                free.iter().copied().filter(|&i| stores[i].len >= need && super::model::yarn_for(stores[i].len - 16) == mode)
                    .min_by_key(|&i| (!captured(i), stores[i].used))
            });
            // no in-memory prefix: the longest cached checkpoint on disk that extends into this prompt (QWEN_PCACHE)
            let disk = (best.is_none() && super::pcache::enabled()).then(|| super::pcache::lookup(&queue[0].1, 0, ckpt_min())).flatten();
            let via = best.map_or(if disk.is_some() { 3 } else { 0 }, |b| b.2);
            let si = match pick {
                Some(i) => i,
                None => {
                    let Some(s) = free.iter().copied().min_by_key(|&i| stores[i].used) else { break };
                    let mut gap = find_gap(&stores, s, total, need);
                    let mut idle: Vec<usize> = free.iter().copied().filter(|&i| i != s && stores[i].len > 0).collect();
                    idle.sort_by_key(|&i| stores[i].used);
                    let mut give: Vec<usize> = Vec::new();
                    while gap.is_none() && !idle.is_empty() {
                        give.push(idle.remove(0));
                        let saved: Vec<i64> = give.iter().map(|&j| std::mem::replace(&mut stores[j].len, 0)).collect();
                        gap = find_gap(&stores, s, total, need);
                        for (&j, l) in give.iter().zip(saved) { stores[j].len = l; }
                    }
                    let Some(base) = gap else { break };
                    tch::Cuda::synchronize(0);
                    for &j in give.iter().chain([&s]) { super::pcache::fence(j as u64); }
                    for &j in give.iter().chain([&s]) {
                        batch_graphs.retain(|k, _| !k.contains(&j));
                        graph_use.retain(|k, _| !k.contains(&j));
                        stores[j].release();
                    }
                    let st = &mut stores[s];
                    st.seq = Some(m.new_seq_in(&pool, base, need));
                    st.base = base; st.len = need;
                    eprintln!("[qwen-serve] store {} takes KV range [{}, {}) ({} tokens, {}){}", s, base, base + need, need,
                              if mode == 1 { "YaRN" } else { "native RoPE" },
                              if give.is_empty() { String::new() } else { format!("; stores {:?} gave theirs up", give) });
                    s
                }
            };
            let (id, prompt, max_new, stop_ids, temp, seed, tq, items) = queue.pop_front().unwrap();
            let st = &mut stores[si];
            st.busy = true; tick += 1; st.used = tick;
            let mut hit = 0;
            let mut mtp_pend = None;
            if via == 2 {
                // back to the prompt checkpoint: positions from there on are rewritten
                let c = st.ckpt.take().unwrap();
                let h = m.ckpt_restore(st.seq_mut(), &c);
                hit = c.ids.len();
                mtp_pend = Some((h, hit as i64 - 1));
                st.history = c.ids.clone();
                st.pend_h = None;
                st.ckpt = Some(c);
            } else if via == 1 && !st.history.is_empty() && st.history.len() < prompt.len() && prompt[..st.history.len()] == st.history[..] {
                hit = st.history.len();
                mtp_pend = st.pend_h.take().map(|h| (h, st.history.len() as i64 - 1));
            } else if via == 3 {
                // make sure no pending write is still reading this store's old checkpoint before we overwrite its state
                super::pcache::fence(si as u64);
                match disk.and_then(|(idx, _)| super::pcache::restore(idx, &prompt, Device::Cuda(0))) {
                    Some((ids, t)) => {
                        let c = Ckpt::from_parts(ids, t);
                        let h = m.ckpt_restore(st.seq_mut(), &c);
                        hit = c.ids.len();
                        mtp_pend = Some((h, hit as i64 - 1));
                        st.history = c.ids.clone();
                        st.pend_h = None;
                        st.ckpt = Some(c);
                    }
                    None => { m.reset(st.seq_mut()); st.history.clear(); st.pend_h = None; st.drop_ckpt(); }
                }
            } else {
                m.reset(st.seq_mut());
                st.history.clear();
                st.pend_h = None;
                st.drop_ckpt();
            }
            // mRoPE positions of the whole prompt (text only: none, the position is the token index)
            let (rp, delta) = if items.is_empty() { (Vec::new(), 0) } else { super::vision::rope_positions(prompt.len(), &items) };
            m.set_mrope(st.seq(), &rp, delta);
            let mm = (!items.is_empty()).then(|| super::vision::MmState::new(items));
            active.push(Active { id, store: si, prompt, max_new, stop_ids, temp, seed, phase: Phase::Prefill { done: hit, mtp_pend, last: None },
                                 out: vec![], sent: 0, t_queue: tq, t_start: Instant::now(), prefill_ms: 0.0, queue_ms: tq.elapsed().as_secs_f64() * 1e3,
                                 hit, rounds: 0, drafted: 0, accepted: 0, cancelled: false, look: Default::default(), look_used: false, look_prop: None, mm });
        }
        if active.is_empty() { continue; }
        // ---- one prefill chunk (prefilling sequences in turn)
        let pre: Vec<usize> = (0..active.len()).filter(|&i| matches!(active[i].phase, Phase::Prefill { .. }) && !active[i].cancelled).collect();
        if !pre.is_empty() {
            turn = (turn + 1) % pre.len();
            let a = &mut active[pre[turn]];
            super::graph::side_join();
            prefill_step(&m, &pol, a, &mut stores[a.store]);
            if a.out.len() > a.sent {   // first token out before the next decode round
                write_line(&mut conn, &json!({"id": a.id, "delta": &a.out[a.sent..]}));
                a.sent = a.out.len();
            }
        }
        // ---- a decoding sequence about to pass its range grows it; one that cannot ends here (finish "length")
        for i in 0..active.len() {
            if !matches!(active[i].phase, Phase::Decode { .. }) || active[i].cancelled { continue; }
            let si = active[i].store;
            if stores[si].seq().pos + 64 <= stores[si].seq().cap { continue; }
            let want = stores[si].len + GRANULE;
            if !grow(&m, &pool, &mut stores, si, want, total, &mut batch_graphs, &mut graph_use) {
                eprintln!("[qwen-serve] store {si} cannot grow past {} positions: output ends at {} tokens", stores[si].len, active[i].out.len());
                active[i].max_new = active[i].out.len();
            }
        }
        // ---- one speculative round of every decoding sequence
        let dec: Vec<usize> = (0..active.len()).filter(|&i| matches!(active[i].phase, Phase::Decode { .. }) && !active[i].cancelled
                                                              && active[i].out.len() < active[i].max_new).collect();
        if !dec.is_empty() {
            if graphs_on {
                tick += 1;
                let mut key: Vec<usize> = dec.iter().map(|&i| active[i].store).collect();
                key.sort_unstable();
                graph_use.insert(key.clone(), tick);
                decode_batch(&m, &pol, &mut active, &dec, &mut stores, &mut batch_graphs);
                evict_graphs(&mut batch_graphs, &graph_use, &key);
            }
            else { for &i in &dec { let a = &mut active[i]; decode_one(&m, &pol, a, &mut stores[a.store]); } }
        }
        // ---- deltas, stop conditions, retirement
        let mut i = 0;
        while i < active.len() {
            let a = &mut active[i];
            let mut finished = a.cancelled;
            if let Some(e) = a.out.iter().position(|t| EOS.contains(t) || a.stop_ids.contains(t)) { a.out.truncate(e + 1); finished = true; }
            if a.out.len() >= a.max_new { a.out.truncate(a.max_new); finished = true; }
            if a.out.len() > a.sent {
                write_line(&mut conn, &json!({"id": a.id, "delta": &a.out[a.sent..]}));
                a.sent = a.out.len();
            }
            if !finished { i += 1; continue; }
            let a = active.remove(i);
            let st = &mut stores[a.store];
            st.busy = false;
            // committed history = prompt + generated tokens except the last (not yet run through the target)
            let committed = st.seq().pos as usize;
            let mut hist = a.prompt.clone();
            hist.extend_from_slice(&a.out);
            hist.truncate(committed);
            st.history = if a.cancelled || matches!(a.phase, Phase::Prefill { .. }) || hist.len() != committed { vec![] } else { hist };
            if st.history.is_empty() { st.pend_h = None; }
            let total = a.t_queue.elapsed().as_secs_f64() * 1e3;
            write_line(&mut conn, &json!({"id": a.id, "done": true, "token_ids": a.out, "cancelled": a.cancelled,
                "queue_ms": a.queue_ms, "prefill_ms": a.prefill_ms, "decode_ms": total - a.queue_ms - a.prefill_ms, "total_ms": total,
                "prefix_hit_tokens": a.hit, "rounds": a.rounds, "accepted_drafts": a.accepted, "drafted_tokens": a.drafted,
                "mm_encode_ms": a.mm.as_ref().map(|x| x.encode_ms), "mm_encoded_tokens": a.mm.as_ref().map(|x| x.encoded_tokens)}));
        }
    }
}

/// One prefill chunk of `a` (target + MTP catch-up); the last chunk samples the first token and the first drafts.
fn prefill_step(m: &Model, pol: &Policy, a: &mut Active, st: &mut Store) {
    let Phase::Prefill { done, mtp_pend, last } = &mut a.phase else { return };
    let mut end = (*done + prefill_macro() as usize).min(a.prompt.len());
    // Prompt checkpoint at the start of the last message header (the final <|im_start|>): the next turn of a conversation
    // repeats everything before it, while what follows (this reply as the template renders it back: thinking dropped or
    // re-spaced, tool calls re-serialized) rarely matches the generated tokens, and the recurrent state cannot rewind.
    let b = a.prompt.iter().rposition(|&t| t == IM_START).unwrap_or(0);
    let at_ckpt = b >= ckpt_min() && *done < b && b <= end;
    if at_ckpt { end = b; }
    let c = a.prompt[*done..end].to_vec();
    let p0 = st.seq().pos;
    let n = c.len() as i64;
    let vrows = match (&mut a.mm, &m.vision) { (Some(mm), Some(v)) => mm.chunk_rows(v, *done, end), _ => None };
    let (lg, x) = m.prefill_lm_mm(st.seq_mut(), &c, vrows);
    // MTP catch-up for this chunk (the pending row from the previous chunk / store gets c[0])
    let mut hs = Vec::new();
    let mut ids = Vec::new();
    let mut start = p0;
    if let Some((h, pos)) = mtp_pend.take() { hs.push(h); ids.push(c[0]); start = pos; }
    if n > 1 { hs.push(x.narrow(0, 0, n - 1)); ids.extend_from_slice(&c[1..]); }
    let cut = a.hit as i64 + m.mtp_tail_from(a.prompt.len() - a.hit);
    super::spec::mtp_catch_up(m, st.seq_mut(), hs, ids, start, cut);
    *mtp_pend = Some((x.narrow(0, n - 1, 1).contiguous(), p0 + n - 1));
    *last = Some(lg);
    *done = end;
    if at_ckpt {
        let mut c = st.ckpt.take(); m.ckpt_save(st.seq(), &mtp_pend.as_ref().unwrap().0, &mut c);
        if let Some(cp) = &c { if super::pcache::enabled() { super::pcache::save(&cp.ids, cp.tensors(), a.store as u64); } }
        st.ckpt = c;
    }
    if end == a.prompt.len() {
        let next = sample_rows(last.as_ref().unwrap(), a.temp, a.seed, st.seq().pos)[0];
        let (h, pos) = mtp_pend.take().unwrap();
        let (lg_d, mul) = m.mtp_run(st.seq_mut(), &h, &[next], pos, true);
        let (drafts, confs) = draft_chain(m, st.seq_mut(), &lg_d, &mul, &pol.first());
        a.prefill_ms = a.t_start.elapsed().as_secs_f64() * 1e3;
        a.out.push(next);
        a.phase = Phase::Decode { next, drafts, confs };
    }
}

/// Eager drafts d1..: d1 from the catch-up logits `lg`, then MTP steps from its last stack at seq.pos.. while the
/// policy asks for more; returns (drafts, confidences).
fn draft_chain(m: &Model, seq: &mut Seq, lg: &Tensor, mul: &Tensor, pol: &Policy) -> (Vec<i64>, Vec<f64>) {
    let p = seq.pos;
    let pick = |lg: &Tensor| { let (t, c) = m.draft_pick(lg); (t.int64_value(&[0]), super::graph::Policy::cal(c.double_value(&[0]))) };
    let (t, c) = pick(lg);
    let (mut d, mut cf) = (vec![t], vec![c]);
    let mut multi = mul.narrow(0, mul.size()[0] - 1, 1).contiguous();
    while pol.more(&[cf.clone()]) {
        let j = d.len();
        let (l2, m2) = m.mtp_run(seq, &multi, &[d[j - 1]], p + j as i64 - 1, true);
        let (t, c) = pick(&l2);
        d.push(t); cf.push(c);
        multi = m2;
    }
    (d, cf)
}

/// After the verify: accept against the target tokens g (at most kk drafts), extend the output; returns the
/// accepted draft count.
fn accept(a: &mut Active, d: &[i64], kk: usize, g: &[i64]) -> usize {
    let mut acc = 0;
    while acc < kk && g[acc] == d[acc] { acc += 1; }
    let bonus = g[acc];
    a.rounds += 1; a.drafted += kk; a.accepted += acc;
    a.out.extend_from_slice(&d[..acc]);
    a.out.push(bonus);
    if let Phase::Decode { next, .. } = &mut a.phase { *next = bonus; }
    acc
}

/// Drafts that may still be accepted (the output must not pass max_new).
fn kk_of(k: usize, a: &Active) -> usize { k.min(a.max_new.saturating_sub(a.out.len())).max(1) }

/// One eager speculative round of one sequence.
fn decode_one(m: &Model, pol: &Policy, a: &mut Active, st: &mut Store) {
    let p = st.seq().pos;
    let Phase::Decode { next, drafts, confs } = &a.phase else { unreachable!() };
    let d = drafts[..pol.verify_len(confs).min(drafts.len())].to_vec();
    let kk = kk_of(d.len(), a);
    let mut rows = vec![*next];
    rows.extend_from_slice(&d);
    let (lg, x) = m.run(st.seq_mut(), &rows, false, false);
    let g = sample_rows(&lg, a.temp, a.seed, p + 1);
    let acc = accept(a, &d, kk, &g);
    m.commit(st.seq_mut(), acc as i64 + 1);
    let mut ids = d[..acc].to_vec();
    ids.push(g[acc]);
    let (l3, m3) = m.mtp_run(st.seq_mut(), &x.narrow(0, 0, acc as i64 + 1).contiguous(), &ids, p, true);
    let (nd, nc) = draft_chain(m, st.seq_mut(), &l3, &m3, pol);
    if let Phase::Decode { drafts, confs, .. } = &mut a.phase { *drafts = nd; *confs = nc; }
    st.pend_h = Some(x.narrow(0, acc as i64, 1).copy());
}

/// One speculative round of the decoding sequences: batched verify (width = the longest verify length of the
/// policy among them, plus the next token), accept, commit and MTP chain (graphs). All-greedy rounds accept on the
/// device and synchronize once before the draft steps; rounds with a sampling sequence accept on the host.
fn decode_batch(m: &Model, pol: &Policy, active: &mut [Active], dec: &[usize], stores: &mut [Store], graphs: &mut HashMap<Vec<usize>, BatchG>) {
    let mut clk = Clock::new();
    let pol = &pol.for_batch(dec.len());
    let mut order = dec.to_vec();
    order.sort_by_key(|&i| active[i].store);
    let key: Vec<usize> = order.iter().map(|&i| active[i].store).collect();
    // (sequences that just joined bring chains of other lengths: the width stays within the shortest)
    let dr = |i: usize| { let Phase::Decode { drafts, confs, .. } = &active[i].phase else { unreachable!() }; (drafts.len(), confs) };
    let kv = pol.width(&order.iter().map(|&i| kk_of(pol.verify_len(dr(i).1), &active[i])).collect::<Vec<_>>())
        .min(order.iter().map(|&i| dr(i).0).min().unwrap());
    let r1 = kv + 1;
    let rows: Vec<Vec<i64>> = order.iter().map(|&i| {
        let Phase::Decode { next, drafts, .. } = &active[i].phase else { unreachable!() };
        let mut r = vec![*next];
        r.extend_from_slice(&drafts[..kv]);
        r
    }).collect();
    let kks: Vec<usize> = order.iter().map(|&i| kk_of(kv, &active[i])).collect();
    let ps: Vec<i64> = key.iter().map(|&s| stores[s].seq().pos).collect();
    if !graphs.contains_key(&key) {
        let t = Instant::now();
        let seqs: Vec<&Seq> = key.iter().map(|&s| stores[s].seq()).collect();
        graphs.insert(key.clone(), BatchG::capture(m, &seqs, pol.cap()));
        eprintln!("[qwen-serve] captured batch graph for stores {:?} in {:.0} ms", key, t.elapsed().as_secs_f64() * 1e3);
    }
    let bg = graphs.get_mut(&key).unwrap();
    let greedy = order.iter().all(|&i| active[i].temp <= 0.0);
    let (gs, x) = {
        let seqs: Vec<&Seq> = key.iter().map(|&s| stores[s].seq()).collect();
        if greedy { bg.round_greedy(m, &seqs, &rows, &kks) } else {
            let (lg, x) = bg.verify(m, &seqs, &rows, &kks);
            let am = Vec::<i64>::try_from(lg.argmax(-1, false).to_device(Device::Cpu)).unwrap();
            let gs = order.iter().enumerate().map(|(g, &i)| {
                let a = &active[i];
                if a.temp <= 0.0 { am[g * r1..(g + 1) * r1].to_vec() }
                else { sample_rows(&lg.narrow(0, (g * r1) as i64, r1 as i64), a.temp, a.seed, ps[g] + 1) }
            }).collect();
            (gs, x)
        }
    };
    let mut ns = Vec::new();
    let mut cids = Vec::new();
    for (g, &i) in order.iter().enumerate() {
        let acc = accept(&mut active[i], &rows[g][1..], kks[g], &gs[g]);
        let lu = active[i].look_used;
        active[i].look.record(lu, acc + 1);
        if let Some(p) = active[i].look_prop.take() {
            let truth: Vec<i64> = rows[g][1..=acc].iter().copied().chain([gs[g][acc]]).collect();
            active[i].look.observe_shadow(&p, &truth);
        }
        ns.push(acc + 1);
        // catch-up tokens: row i reads the token at position p + i + 1 (accepted drafts, then the bonus)
        cids.push((0..r1).map(|i| if i < acc { rows[g][i + 1] } else if i == acc { gs[g][acc] } else { rows[g][(i + 1).min(kv)] }).collect::<Vec<_>>());
    }
    clk.lap(1, dec.len());
    if !greedy { bg.commit_host(r1, &cids, &ps, &ns); }
    clk.lap(2, dec.len());
    for (g, &s) in key.iter().enumerate() {
        let st = &mut stores[s];
        st.seq_mut().ids.extend_from_slice(&rows[g][..ns[g]]);
        st.seq_mut().pos += ns[g] as i64;
        // the verify buffers are reused by the next round: keep an owned copy of the history's last stack
        st.pend_h = Some(x.narrow(0, (g * r1 + ns[g] - 1) as i64, 1).copy());
    }
    // prompt-lookup proposals continuing after each sequence's new next token
    let looks: Vec<Option<(Vec<i64>, f64)>> = order.iter().enumerate().map(|(g, &i)| {
        if !super::lookup::enabled() { return None; }
        let a = &mut active[i];
        let Phase::Decode { next, .. } = &a.phase else { unreachable!() };
        let p = a.look.propose(&stores[key[g]].seq().ids, *next, pol.cap());
        a.look_prop = p.as_ref().map(|(v, _)| v.clone());
        p.filter(|_| a.look.usable())
    }).collect();
    let (next, used) = bg.drafts(pol, &looks);
    for (g, &i) in order.iter().enumerate() {
        if let Phase::Decode { drafts, confs, .. } = &mut active[i].phase { (*drafts, *confs) = next[g].clone(); }
        active[i].look_used = used[g];
    }
    clk.lap(3, dec.len());
}

/// Keep the graphs of several stores within QWEN_SERVE_GRAPH_MB (default 3000; ~5 MB per captured verify row):
/// drop the least recently used sets (single stores are kept).
fn evict_graphs(graphs: &mut HashMap<Vec<usize>, BatchG>, used: &HashMap<Vec<usize>, u64>, current: &[usize]) {
    let budget = env_i64("QWEN_SERVE_GRAPH_MB", 3000) * 1_000_000;
    let held = |g: &HashMap<Vec<usize>, BatchG>| g.iter().filter(|(k, _)| k.len() > 1).map(|(_, b)| b.rows_held() * 5_000_000).sum::<i64>();
    while held(graphs) > budget {
        let Some(k) = graphs.keys().filter(|k| k.len() > 1 && k.as_slice() != current).min_by_key(|k| used.get(*k).copied().unwrap_or(0)).cloned() else { break };
        tch::Cuda::synchronize(0);
        graphs.remove(&k);
        eprintln!("[qwen-serve] dropped batch graphs of stores {:?}", k);
    }
}

/// QWEN_SERVE_TIMING=1: synchronized phase times of batched rounds (drafts, verify, commit, catch-up) per batch size,
/// printed every 100 rounds of a size.
struct Clock { on: bool, t: Instant }
thread_local! { static ACC: std::cell::RefCell<[[f64; 5]; 5]> = const { std::cell::RefCell::new([[0.0; 5]; 5]) }; }
impl Clock {
    fn new() -> Self {
        let on = std::env::var("QWEN_SERVE_TIMING").is_ok();
        if on { tch::Cuda::synchronize(0); }
        Clock { on, t: Instant::now() }
    }
    fn lap(&mut self, phase: usize, n: usize) {
        if !self.on { return; }
        tch::Cuda::synchronize(0);
        let ms = self.t.elapsed().as_secs_f64() * 1e3;
        self.t = Instant::now();
        ACC.with(|a| {
            let mut a = a.borrow_mut();
            a[n][phase] += ms;
            if phase == 3 {
                a[n][4] += 1.0;
                if a[n][4] >= 100.0 {
                    let r = a[n][4];
                    eprintln!("[qwen-serve] batch {} per round ms: verify (greedy: + accept, commit, catch-up) {:.2} host commit {:.2} draft steps {:.2}",
                              n, a[n][1] / r, a[n][2] / r, a[n][3] / r);
                    a[n] = [0.0; 5];
                }
            }
        });
    }
}
