//! `qwen-gen <model_dir> <ids.json> <n_new> [logits_out.f32]`: prefill the prompt in chains of 16 rows,
//! then greedy-decode n_new tokens one row at a time; optionally dump the prompt logits (f32, [T, vocab]).
use super::model::Model;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use tch::{Device, Kind, Tensor};

pub fn gen(dir: &Path, ids_path: &Path, n_new: usize, dump: Option<&Path>) {
    let _guard = tch::no_grad_guard();
    let t0 = Instant::now();
    // QWEN_TAIL_ARMS="A;B;.." (each a comma list of VAR=value) with QWEN_TAIL=n: in one process, for every arm and every prompt
    // of the comma list `ids_path`, prefill all but the last n tokens and run those n (QWEN_TAIL_STEP rows at a time),
    // dumping their logits to <dump>_<arm index>_<prompt index>.f32
    if let (Ok(arms), Some(d)) = (std::env::var("QWEN_TAIL_ARMS"), dump) {
        let m = Model::load(dir);
        eprintln!("[qwen] load {:.1}s", t0.elapsed().as_secs_f64());
        let tail: usize = std::env::var("QWEN_TAIL").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        let step: usize = std::env::var("QWEN_TAIL_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(tail);
        let prompts: Vec<Vec<i64>> = ids_path.to_str().unwrap().split(',').map(|f| serde_json::from_reader(std::fs::File::open(f).expect("ids json")).expect("ids")).collect();
        for (ai, arm) in arms.split(';').enumerate() {
            let saved: Vec<(String, Option<String>)> = arm.split(',').filter(|kv| kv.contains('=')).map(|kv| {
                let (a, b) = kv.split_once('=').unwrap();
                let old = std::env::var(a).ok();
                std::env::set_var(a, b);
                (a.to_string(), old)
            }).collect();
            for (pi, ids) in prompts.iter().enumerate() {
                let mut seq = m.new_seq(ids.len() as i64 + 64);
                let _ = m.prefill(&mut seq, &ids[..ids.len() - tail]);
                // QWEN_STATE_DUMP=dir: the sequence state after the prefill, one file per layer and tensor
                if let Ok(dir) = std::env::var("QWEN_STATE_DUMP") { m.dump_state(&seq, &format!("{dir}/st_{ai}_{pi}")); }
                let mut lg = Vec::new();
                for c in ids[ids.len() - tail..].chunks(step.max(1)) { lg.push(m.run(&mut seq, c, true, false).0); }
                let v: Vec<f32> = Vec::try_from(Tensor::cat(&lg, 0).to_device(Device::Cpu).to_kind(Kind::Float).flatten(0, -1)).unwrap();
                let path = format!("{}_{}_{}.f32", d.display(), ai, pi);
                std::fs::write(&path, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }).unwrap();
                eprintln!("[qwen] arm {} '{}' prompt {}: {} tail rows -> {}", ai, arm, pi, tail, path);
            }
            for (a, old) in saved { match old { Some(v) => std::env::set_var(&a, v), None => std::env::remove_var(&a) } }
        }
        return;
    }
    let ids: Vec<i64> = serde_json::from_reader(std::fs::File::open(ids_path).expect("ids json")).expect("ids");
    let m = Model::load(dir);
    eprintln!("[qwen] load {:.1}s", t0.elapsed().as_secs_f64());
    let cap: i64 = std::env::var("QWEN_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    let mut seq = m.new_seq(cap.max(ids.len() as i64 + n_new as i64 + 16));
    let mut logits = Vec::new();
    let chunked = std::env::var("QWEN_PREFILL_STEP").is_ok();
    // QWEN_TAIL=n: prefill all but the last n tokens, then run those n as one chunk with all their logits
    let tail: usize = std::env::var("QWEN_TAIL").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(ids.len().saturating_sub(1));
    // QWEN_PREFILL_REPS=n: n - 1 timed warm-up prefills first (page cache of the n-gram table, allocator pools)
    // QWEN_PREFILL_AB=VAR: the warm-up reps alternate VAR=1 / unset (same-process A/B of a per-launch switch)
    let ab = std::env::var("QWEN_PREFILL_AB").ok();
    // (VAR may be a comma list: all set to 1 / unset together; VAR=value items are set to value / unset)
    let set_ab = |on: bool| if let Some(v) = &ab {
        for item in v.split(',') {
            let (k, val) = item.split_once('=').unwrap_or((item, "1"));
            if on { std::env::set_var(k, val); } else { std::env::remove_var(k); }
        }
    };
    for i in 1..std::env::var("QWEN_PREFILL_REPS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1) {
        set_ab(i % 2 == 1);
        let mut s = m.new_seq(seq.cap);
        let t = Instant::now();
        let _ = m.prefill(&mut s, &ids);
        tch::Cuda::synchronize(0);
        eprintln!("[qwen] warm-up prefill {} {:.2}s{}", i, t.elapsed().as_secs_f64(),
                  ab.as_ref().map_or(String::new(), |v| format!(" ({v}={})", if i % 2 == 1 { "1" } else { "unset" })));
    }
    set_ab(false);
    let t1 = Instant::now();
    if chunked { for chunk in ids.chunks(16) { logits.push(m.step(&mut seq, chunk, true)); } }
    else if tail > 0 {
        let _ = m.prefill(&mut seq, &ids[..ids.len() - tail]);
        // QWEN_TAIL_STEP=n: the tail in chains of n rows (the decode / verify kernels, as speculative rounds of n - 1 drafts)
        let step: usize = std::env::var("QWEN_TAIL_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(tail);
        for c in ids[ids.len() - tail..].chunks(step.max(1)) { logits.push(m.run(&mut seq, c, true, false).0); }
    }
    else { logits.push(m.prefill(&mut seq, &ids)); }
    tch::Cuda::synchronize(0);
    let pf = t1.elapsed().as_secs_f64();
    eprintln!("[qwen] prefill {} tokens {:.2}s ({:.0} tok/s)", ids.len(), pf, ids.len() as f64 / pf);
    if let Some([h, c, t, ns]) = m.ngram_stats() {
        let n = (h + c + t).max(1) as f64;
        eprintln!("[qwen] n-gram rows (all prefills): hot {:.1}%  cache {:.1}%  table {:.1}%  pack {:.0} ms total",
                  100.0 * h as f64 / n, 100.0 * c as f64 / n, 100.0 * t as f64 / n, ns as f64 / 1e6);
    }
    let all = Tensor::cat(&logits, 0);
    if let Some(p) = dump {
        let keep: i64 = std::env::var("QWEN_DUMP_LAST").ok().and_then(|v| v.parse().ok()).unwrap_or(all.size()[0]);
        let rows = all.size()[0];
        let sel = all.narrow(0, rows - keep.min(rows), keep.min(rows));
        let v: Vec<f32> = Vec::try_from(sel.to_device(Device::Cpu).to_kind(Kind::Float).flatten(0, -1)).unwrap();
        let mut f = std::fs::File::create(p).unwrap();
        f.write_all(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }).unwrap();
    }
    let mut next = all.get(all.size()[0] - 1).argmax(0, false).int64_value(&[]);
    let mut out = vec![next];
    let t2 = Instant::now();
    for _ in 1..n_new {
        if next == 248046 || next == 248044 { break; }
        let lg = m.step(&mut seq, &[next], true);
        next = lg.get(0).argmax(0, false).int64_value(&[]);
        out.push(next);
    }
    let dt = t2.elapsed().as_secs_f64();
    eprintln!("[qwen] decode {} tokens {:.2}s ({:.1} tok/s, no graphs/spec)", out.len() - 1, dt, (out.len() - 1) as f64 / dt);
    println!("OUT_IDS {}", serde_json::to_string(&out).unwrap());
}

/// `qwen-spec <model_dir> <ids.json> <n_new> <k>`: plain greedy vs MTP speculative greedy on the same prompt;
/// prints both outputs' agreement, acceptance and decode speed.
pub fn spec(dir: &Path, ids_path: &Path, n_new: usize, k: usize) {
    use super::spec::{generate, generate_graph, generate_plain, Stats};
    let _guard = tch::no_grad_guard();
    let ids: Vec<i64> = serde_json::from_reader(std::fs::File::open(ids_path).expect("ids json")).expect("ids");
    let m = Model::load(dir);
    let cap = ids.len() as i64 + n_new as i64 + 64;
    let t = Instant::now();
    let plain = if std::env::var("QWEN_SKIP_PLAIN").is_ok() { vec![] } else { let mut s = m.new_seq(cap); generate_plain(&m, &mut s, &ids, n_new) };
    tch::Cuda::synchronize(0);
    let tp = t.elapsed().as_secs_f64();
    let mut st = Stats::default();
    let t = Instant::now();
    let graphs = std::env::var("QWEN_GRAPH").map_or(true, |v| v != "0");
    let fast = { let mut s = m.new_seq(cap); if graphs { generate_graph(&m, &mut s, &ids, n_new, k, &mut st) } else { generate(&m, &mut s, &ids, n_new, k, &mut st) } };
    tch::Cuda::synchronize(0);
    let ts = t.elapsed().as_secs_f64();
    let same = plain.iter().zip(fast.iter()).take_while(|(a, b)| a == b).count();
    eprintln!("[qwen-spec] plain {} tok {:.2}s | spec {} tok {:.2}s | identical prefix {}/{} | rounds {} accepted {}/{} ({:.2} tok/round)",
              plain.len(), tp, fast.len(), ts, same, plain.len().min(fast.len()), st.rounds, st.accepted, st.drafted,
              fast.len() as f64 / st.rounds.max(1) as f64);
    if st.ms[1] > 0.0 {
        let r = st.rounds.max(1) as f64;
        eprintln!("[qwen-spec] per round ms: draft {:.2} verify {:.2} commit {:.2} catch-up {:.2}", st.ms[0] / r, st.ms[1] / r, st.ms[2] / r, st.ms[3] / r);
    }
    println!("PLAIN {}", serde_json::to_string(&plain).unwrap());
    println!("SPEC {}", serde_json::to_string(&fast).unwrap());
}

/// `qwen-batch <model_dir> <ids.json,ids.json,..> <rounds> <k>`: one sequence per prompt (1..4), decoded together
/// with the serving round (graph::BatchG: batched verify, commit, MTP draft chain; greedy); synchronized phase times.
/// The draft policy comes from QWEN_SPEC_* (k from the argument unless QWEN_SPEC_K is set).
/// QWEN_BATCH_ARMS="A;B;.." (each a comma list of VAR=value): one run per arm in the same process (fresh sequences).
/// QWEN_BATCH_LOG=file: one JSON line per sequence and round {"conf":[..],"acc":a,"w":verified drafts}.
pub fn batch(dir: &Path, files: &str, rounds: usize, k: usize) {
    let _guard = tch::no_grad_guard();
    // prompt groups separated by '|' run one after the other (each group is one batch)
    let groups: Vec<Vec<Vec<i64>>> = files.split('|').map(|grp| grp.split(',')
        .map(|f| serde_json::from_reader(std::fs::File::open(f).expect("ids json")).expect("ids")).collect()).collect();
    let m = Model::load(dir);
    if std::env::var("QWEN_SPEC_K").is_err() { std::env::set_var("QWEN_SPEC_K", k.to_string()); }
    let arms = std::env::var("QWEN_BATCH_ARMS").unwrap_or_default();
    let arms: Vec<&str> = if arms.is_empty() { vec![""] } else { arms.split(';').collect() };
    let mut log = std::env::var("QWEN_BATCH_LOG").ok().map(|p| std::fs::File::create(p).unwrap());
    for arm in arms {
        let saved: Vec<(String, Option<String>)> = arm.split(',').filter(|kv| kv.contains('=')).map(|kv| {
            let (a, b) = kv.split_once('=').unwrap();
            let old = std::env::var(a).ok();
            std::env::set_var(a, b);
            (a.to_string(), old)
        }).collect();
        for prompts in &groups { batch_arm(&m, prompts, rounds, arm, log.as_mut()); }
        for (a, old) in saved { match old { Some(v) => std::env::set_var(&a, v), None => std::env::remove_var(&a) } }
    }
}

fn batch_arm(m: &Model, prompts: &[Vec<i64>], rounds: usize, arm: &str, mut log: Option<&mut std::fs::File>) {
    use super::graph::{BatchG, Policy};
    use super::model::Seq;
    let pol = Policy::from_env().for_batch(prompts.len());
    let cap = prompts.iter().map(|p| p.len()).max().unwrap() as i64 + (rounds as i64 + 2) * (pol.cap() as i64 + 1) + 64;
    let mut seqs: Vec<Seq> = prompts.iter().map(|_| m.new_seq(cap)).collect();
    let mut state = Vec::new();   // (next, drafts, confs)
    for (s, p) in seqs.iter_mut().zip(prompts) {
        let t_pf = Instant::now();
        let (next, h) = super::spec::prefill(m, s, p);
        tch::Cuda::synchronize(0);
        eprintln!("[qwen-batch] prefill {} tokens (+ MTP catch-up) {:.2}s", p.len(), t_pf.elapsed().as_secs_f64());
        let (lg, mut multi) = m.mtp_run(s, &h, &[next], s.pos - 1, true);
        let pick = |lg: &Tensor| { let (t, c) = m.draft_pick(lg); (t.int64_value(&[0]), super::graph::Policy::cal(c.double_value(&[0]))) };
        let (t, c) = pick(&lg);
        let (mut d, mut cf) = (vec![t], vec![c]);
        while pol.first().more(&[cf.clone()]) {
            let j = d.len();
            let (l2, x2) = m.mtp_run(s, &multi, &[d[j - 1]], s.pos + j as i64 - 1, true);
            let (t, c) = pick(&l2);
            d.push(t); cf.push(c);
            multi = x2;
        }
        state.push((next, d, cf));
    }
    let t_cap = Instant::now();
    let mut bg = BatchG::capture(m, &seqs.iter().collect::<Vec<_>>(), pol.cap());
    let mut ms = [0.0f64; 3];
    let mut toks = 0usize;
    let mut rows_v = 0usize;
    let mut hist = vec![0usize; pol.cap() + 2];
    let mut ended = vec![false; prompts.len()];
    let mut looks: Vec<super::lookup::Lookup> = prompts.iter().map(|_| Default::default()).collect();
    let mut look_used = vec![false; prompts.len()];
    let mut look_rounds = 0usize;
    let mut props: Vec<Option<(Vec<i64>, f64)>> = vec![None; prompts.len()];
    let sync = || tch::Cuda::synchronize(0);
    sync();
    eprintln!("[qwen-batch] arm '{}' {:?}: capture {:.0} ms", arm, pol, t_cap.elapsed().as_secs_f64() * 1e3);
    let t_all = Instant::now();
    let mut done_rounds = 0;
    let lapping = std::env::var("QWEN_BATCH_NOLAP").is_err();   // QWEN_BATCH_NOLAP=1: no phase syncs (nsys)
    for _ in 0..rounds {
        if ended.iter().all(|&e| e) { break; }   // every sequence reached EOS
        done_rounds += 1;
        let mut t = Instant::now();
        let mut lap = |i: usize, t: &mut Instant| { if lapping { sync(); ms[i] += t.elapsed().as_secs_f64() * 1e3; *t = Instant::now(); } };
        let kv = pol.width(&state.iter().map(|(_, _, c)| pol.verify_len(c)).collect::<Vec<_>>()).min(state.iter().map(|s| s.1.len()).min().unwrap());
        let r1 = kv + 1;
        hist[kv] += 1;
        let rows: Vec<Vec<i64>> = state.iter().map(|(n, d, _)| { let mut r = vec![*n]; r.extend_from_slice(&d[..kv]); r }).collect();
        // QWEN_HOST_ACCEPT=1 (A/B): the host-path accept (verify, argmax copy, host-written commit / catch-up inputs)
        let host_acc = std::env::var("QWEN_HOST_ACCEPT").is_ok();
        let gt: Vec<Vec<i64>> = if host_acc {
            let (lg, _) = bg.verify(m, &seqs.iter().collect::<Vec<_>>(), &rows, &vec![kv; seqs.len()]);
            Vec::<i64>::try_from(lg.argmax(-1, false).to_device(Device::Cpu)).unwrap().chunks(r1).map(|c| c.to_vec()).collect()
        } else { bg.round_greedy(m, &seqs.iter().collect::<Vec<_>>(), &rows, &vec![kv; seqs.len()]).0 };
        lap(0, &mut t);
        let mut ns = Vec::new();
        let mut cids = Vec::new();
        for g in 0..seqs.len() {
            let mut a = 0;
            while a < kv && gt[g][a] == rows[g][a + 1] { a += 1; }
            if look_used[g] { look_rounds += 1; }
            looks[g].record(look_used[g], a + 1);
            if let Some((p, _)) = &props[g] {
                let truth: Vec<i64> = rows[g][1..=a].iter().copied().chain([gt[g][a]]).collect();
                looks[g].observe_shadow(p, &truth);
            }
            ns.push(a + 1);
            rows_v += r1;
            if let (Some(f), false) = (log.as_mut(), ended[g]) {
                let _ = writeln!(f, "{}", serde_json::json!({"conf": state[g].2, "acc": a, "w": kv}));
            }
            ended[g] |= rows[g][1..=a].iter().chain([&gt[g][a]]).any(|t| super::spec::EOS.contains(t));
            cids.push((0..r1).map(|i| if i < a { rows[g][i + 1] } else if i == a { gt[g][a] } else { rows[g][(i + 1).min(kv)] }).collect::<Vec<_>>());
        }
        if host_acc { bg.commit_host(r1, &cids, &seqs.iter().map(|s| s.pos).collect::<Vec<_>>(), &ns); }
        lap(1, &mut t);
        for (g, s) in seqs.iter_mut().enumerate() {
            s.ids.extend_from_slice(&rows[g][..ns[g]]);
            s.pos += ns[g] as i64;
            toks += ns[g];
        }
        props = (0..seqs.len()).map(|g| {
            if !super::lookup::enabled() { return None; }
            looks[g].propose(&seqs[g].ids, cids[g][ns[g] - 1], pol.cap())
        }).collect();
        let usable: Vec<Option<(Vec<i64>, f64)>> = (0..seqs.len()).map(|g| props[g].clone().filter(|_| looks[g].usable())).collect();
        let (next, used) = bg.drafts(&pol, &usable);
        look_used = used;
        for g in 0..seqs.len() { state[g] = (cids[g][ns[g] - 1], next[g].0.clone(), next[g].1.clone()); }
        lap(2, &mut t);
    }
    let dt = t_all.elapsed().as_secs_f64();
    let r = done_rounds as f64;
    eprintln!("[qwen-batch] arm '{}': {} seqs, {} rounds: {:.2} tok/round, {:.1} tok/s aggregate, {:.2} verify rows/seq/round, widths {:?}",
              arm, seqs.len(), done_rounds, toks as f64 / r, toks as f64 / dt, rows_v as f64 / r / seqs.len() as f64, &hist[1..]);
    eprintln!("[qwen-batch] arm '{}' per round ms: verify + accept + commit + catch-up {:.2} host {:.2} draft steps {:.2}; lookup drafts in {} sequence-rounds",
              arm, ms[0] / r, ms[1] / r, ms[2] / r, look_rounds);
    let outs: Vec<&[i64]> = seqs.iter().zip(prompts).map(|(s, p)| &s.ids[p.len()..]).collect();
    println!("BATCH_OUT {} {}", arm, serde_json::to_string(&outs).unwrap());
}


/// `qwen-refusal-dir <model_dir> <harmful.json> <harmless.json> <out.safetensors>`: refusal directions for SPARK_ABLATE
/// (docs/abliteration.md). Each JSON file is a list of prompts as token ids (chat template applied, generation prompt
/// included; scripts/refusal_prompts.py). Every prompt is prefilled alone; each layer's mixed block input at the last
/// row is recorded. Per layer: d = mean(harmful) - mean(harmless), normalized. Writes `directions` [layers, hidden],
/// `scores` [layers] (|mean difference| / mean activation norm) and `direction` (the best-scoring layer in the middle
/// half of the stack, or SPARK_ABLATE_FROM_LAYER).
pub fn refusal_dir(dir: &Path, harmful: &Path, harmless: &Path, out: &Path) {
    let _guard = tch::no_grad_guard();
    let m = Model::load(dir);
    let load = |p: &Path| -> Vec<Vec<i64>> { serde_json::from_reader(std::fs::File::open(p).expect("prompt ids json")).expect("prompt ids") };
    let run = |set: Vec<Vec<i64>>, name: &str| -> Vec<Vec<Tensor>> {
        let t0 = Instant::now();
        let n = set.len();
        let r = set.iter().map(|ids| {
            let mut seq = m.new_seq(ids.len() as i64 + 64);
            crate::ablate::capture_start();
            let _ = m.prefill(&mut seq, ids);
            tch::Cuda::synchronize(0);
            crate::ablate::capture_take()
        }).collect();
        eprintln!("[refusal-dir] {name}: {n} prompts in {:.1}s", t0.elapsed().as_secs_f64());
        r
    };
    let (h, n) = (run(load(harmful), "harmful"), run(load(harmless), "harmless"));
    crate::ablate::save_directions(&h, &n, out);
}


/// `qwen-pcache-selftest <model_dir>`: prove the SSD prefix cache round-trips a checkpoint losslessly, with no network.
/// Prefill a prefix, take an in-memory checkpoint, write it to disk, read it back, and compare the token ids and every
/// state tensor (including the pending hidden row, which ckpt_save stores) bit for bit. Identical tensors mean the disk
/// checkpoint restores to exactly the in-memory one, so decoding from either is identical by construction. Uses a
/// private QWEN_PCACHE_DIR, so it never touches a serving cache.
pub fn pcache_selftest(dir: &Path) {
    use super::model::Ckpt;
    let _guard = tch::no_grad_guard();
    std::env::set_var("QWEN_PCACHE", "1");
    let tmp = std::env::var("QWEN_PCACHE_SELFTEST_DIR").unwrap_or_else(|_| format!("/tmp/qpc-selftest-{}", std::process::id()));
    std::env::set_var("QWEN_PCACHE_DIR", &tmp);
    std::env::set_var("QWEN_PCACHE_MIN", "16");
    let m = Model::load(dir);
    super::pcache::init(dir);
    let prefix: Vec<i64> = (0..2048i64).map(|i| (i.wrapping_mul(2654435761) & 0x7fff) % 150000 + 10).collect();
    let n = prefix.len() as i64;
    let mut seq = m.new_seq(n + 64);
    let (_lg, x) = m.prefill_lm(&mut seq, &prefix);
    let pend = x.narrow(0, n - 1, 1).contiguous();
    let mut slot: Option<Ckpt> = None;
    m.ckpt_save(&seq, &pend, &mut slot);
    let c_mem = slot.unwrap();
    super::pcache::save(&c_mem.ids, c_mem.tensors(), 0);
    super::pcache::fence(0);
    let (idx, len) = super::pcache::lookup(&prefix, 0, 16).expect("selftest: entry not found on disk");
    assert_eq!(len, prefix.len(), "selftest: cached length");
    let (ids, t) = super::pcache::restore(idx, &prefix, Device::Cuda(0)).expect("selftest: restore failed");
    assert_eq!(ids, c_mem.ids, "selftest: ids differ");
    let mem = c_mem.tensors();
    assert_eq!(mem.len(), t.len(), "selftest: tensor count {} vs {}", mem.len(), t.len());
    let mut bad = 0;
    for (i, (a, b)) in mem.iter().zip(&t).enumerate() {
        assert_eq!(a.size(), b.size(), "selftest: tensor {i} shape {:?} vs {:?}", a.size(), b.size());
        if !a.equal(b) { bad += 1;
            let d = (a.to_kind(tch::Kind::Float) - b.to_kind(tch::Kind::Float)).abs().max().double_value(&[]);
            eprintln!("[selftest] tensor {i} {:?} DIFFERS (max abs {d})", a.size()); }
    }
    assert_eq!(bad, 0, "selftest: {bad} of {} state tensors differ", mem.len());
    let _ = std::fs::remove_dir_all(&tmp);
    println!("PCACHE-SELFTEST OK: {} tokens, ids + {} state tensors bitwise identical disk<->memory", prefix.len(), mem.len());
}
