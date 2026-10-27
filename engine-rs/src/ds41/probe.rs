//! `ds41-golden <ckpt> <golden.safetensors> <engram_hash.json|-> <out.json> [decode_tail]`: runs the golden ids
//! (bench/ds41/ref/ref_stream.py) through the engine, TP over the ranks of GLM53_TP_WORLD, and compares the logits:
//!   A. one chunk at position 0 (prefill) vs the golden logits of every position;
//!   B. prefill of all but the last `decode_tail` (default 4) tokens, then one token per step (decode path: rings,
//!      compressor state, cached keys) vs the golden logits of those positions.
//! Reports per mode: greedy (top-1) agreement, KL(golden || engine) mean / max, max |logit diff|.
//! `ds41-hash-check <engram_hash.json>`: the engine's n-gram hash ids against the official ones stored in the file.
use super::engram::Hash;
use super::model::{Model, Seq};
use serde_json::json;
use std::path::Path;
use tch::{Device, Kind, Tensor};

fn compare(name: &str, golden: &Tensor, ours: &Tensor) -> serde_json::Value {
    let g = golden.to_kind(Kind::Float);
    let o = ours.to_kind(Kind::Float).to_device(g.device());
    let lg = g.log_softmax(-1, Kind::Float);
    let lo = o.log_softmax(-1, Kind::Float);
    let kl = (lg.exp() * (&lg - &lo)).sum_dim_intlist([-1].as_slice(), false, Kind::Float);   // per row
    let agree = g.argmax(-1, false).eq_tensor(&o.argmax(-1, false)).to_kind(Kind::Float);
    let maxdiff = (&g - &o).abs().max().double_value(&[]);
    let r = json!({"mode": name, "rows": g.size()[0], "top1_agree": agree.mean(Kind::Float).double_value(&[]),
        "kl_mean": kl.mean(Kind::Float).double_value(&[]), "kl_max": kl.max().double_value(&[]), "max_abs_logit_diff": maxdiff});
    eprintln!("[ds41-golden] {r}");
    r
}

pub fn golden(ckpt: &Path, golden: &Path, hash: &str, out: &Path, decode_tail: i64) {
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    crate::tp::set_tf32(false);
    let dev = Device::Cuda(0);
    let gold: std::collections::HashMap<String, Tensor> = Tensor::read_safetensors(golden).expect("golden").into_iter().collect();
    let ids: Vec<i64> = Vec::<i64>::try_from(&gold["ids"]).unwrap();
    let logits = gold.get("logits").expect("golden without full logits (T > 1024)");
    let n_layers = std::env::var("DS41_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let m = Model::load(ckpt, tp.rank, tp.world, n_layers, (hash != "-").then(|| Path::new(hash)), dev);
    let t = ids.len() as i64;
    let cap = (t + 255) / 256 * 256;
    let mut results = Vec::new();
    // A. whole sequence in one chunk
    let mut seq = m.new_seq(cap);
    let t0 = std::time::Instant::now();
    let x = m.forward(&mut seq, &ids, None);
    let la = m.logits(&x);
    tch::Cuda::synchronize(0);
    let ta = t0.elapsed().as_secs_f64();
    // per-layer residual vs the golden hidden states (DS41_DUMP_LAYERS)
    let dumps = super::model::DUMP.with(|d| std::mem::take(&mut *d.borrow_mut()));
    for (l, h) in dumps {
        if l >= 1000 { Tensor::write_safetensors(&[("h", &h)], format!("/tmp/ds41_dump_{l}.safetensors")).unwrap(); continue; }
        if let Some(g) = gold.get(&format!("hidden.{l}")) {
            let g = g.to_kind(Kind::Float);
            let rel = ((&h - &g).square().sum(Kind::Float) / g.square().sum(Kind::Float)).sqrt().double_value(&[]);
            let maxd = (&h - &g).abs().max().double_value(&[]);
            eprintln!("[ds41-golden] layer {l} residual: rel L2 {rel:.3e}, max |diff| {maxd:.3e}");
            results.push(json!({"mode": "hidden", "layer": l, "rel_l2": rel, "max_abs": maxd}));
        }
    }
    results.push(compare("prefill", logits, &la));
    // A2. CED prefill (encoder over all, decoder replaying the last window, chunks of 512): the last position's
    // logits against the reference and against the exact engine prefill, then decode of the reference tail
    {
        let tail = decode_tail.min(t - 1).max(1);
        let head = (t - tail) as usize;
        let mut seq = m.new_seq(cap);
        let gc = std::env::var("DS41_GOLDEN_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(512);
        let chunks: Vec<&[i64]> = ids[..head].chunks(gc).collect();
        let mut x = None;
        for (i, c) in chunks.iter().enumerate() { if let Some(v) = m.prefill_ced(&mut seq, c, i + 1 == chunks.len()) { x = Some(v); } }
        let lc = m.logits(&x.unwrap());
        results.push(compare("ced_last_vs_ref", &logits.narrow(0, head as i64 - 1, 1), &lc));
        results.push(compare("ced_last_vs_exact", &la.narrow(0, head as i64 - 1, 1), &lc));
        let mut rows = Vec::new();
        for i in head..t as usize { rows.push(m.step_logits(&mut seq, &ids[i..i + 1])); }
        results.push(compare("ced_decode_vs_ref", &logits.narrow(0, head as i64, tail), &Tensor::cat(&rows, 0)));
    }
    // B. prefill then single-token decode of the tail
    let tail = decode_tail.min(t - 1).max(1);
    let mut seq = m.new_seq(cap);
    let head = (t - tail) as usize;
    let _ = m.forward(&mut seq, &ids[..head], Some(&[head as i64 - 1]));
    let mut rows = Vec::new();
    for i in head..t as usize { rows.push(m.step_logits(&mut seq, &ids[i..i + 1])); }
    let lb = Tensor::cat(&rows, 0);
    results.push(compare("decode", &logits.narrow(0, head as i64, tail), &lb));
    // engine vs engine: decode rows against the same rows of the one-chunk run (state carried correctly)
    results.push(compare("decode_vs_prefill", &la.narrow(0, head as i64, tail), &lb));
    // the single decode again on a fresh sequence: must reproduce itself exactly (no races)
    {
        let mut seq = m.new_seq(cap);
        let _ = m.forward(&mut seq, &ids[..head], Some(&[head as i64 - 1]));
        let mut rows = Vec::new();
        for i in head..t as usize { rows.push(m.step_logits(&mut seq, &ids[i..i + 1])); }
        let d = (Tensor::cat(&rows, 0) - &lb).abs().max().double_value(&[]);
        eprintln!("[ds41-golden] single decode repeated: max |logit diff| {d:.3e}");
    }
    // batched decode: two copies of the sequence decoded together must reproduce the single decode exactly
    let mut pair: Vec<Seq> = (0..2).map(|_| m.new_seq(cap)).collect();
    for sq in pair.iter_mut() { let _ = m.forward(sq, &ids[..head], Some(&[head as i64 - 1])); }
    let mut brows: Vec<Tensor> = Vec::new();
    // DS41_DEBUG_ROWS=1: the first place the batched step's row 0 differs from the single step (same token)
    let dbg_single = if std::env::var("DS41_DEBUG_ROWS").is_ok_and(|v| v == "1") {
        let mut seq = m.new_seq(cap);
        let _ = m.forward(&mut seq, &ids[..head], Some(&[head as i64 - 1]));
        let _ = super::model::take_rows_debug();
        Some((head..t as usize).map(|i| { let _ = m.step_logits(&mut seq, &ids[i..i + 1]); super::model::take_rows_debug() }).collect::<Vec<_>>())
    } else { None };
    for i in head..t as usize {
        let mut refs: Vec<&mut Seq> = pair.iter_mut().collect();
        brows.push(m.decode_logits(&mut refs, &[ids[i], ids[i]]));
        if let Some(a) = dbg_single.as_ref() {
            let b = super::model::take_rows_debug();
            let first = a[i - head].iter().zip(&b).find_map(|((na, ta), (_, tb))| {
                let d = (ta - tb).abs().max().double_value(&[]);
                (d != 0.0).then(|| format!("{na}: max |diff| {d:.3e}"))
            });
            eprintln!("[ds41-golden] rows debug step {}: {} records, first difference {}", i - head, b.len(), first.unwrap_or("none".into()));
        }
    }
    let bl = Tensor::stack(&brows, 0);                                  // [tail, 2, vocab]
    let d0 = (bl.select(1, 0) - &lb).abs().max().double_value(&[]);
    let d1 = (bl.select(1, 1) - &lb).abs().max().double_value(&[]);
    eprintln!("[ds41-golden] batched decode (B=2) vs single: max |logit diff| row0 {d0:.3e} row1 {d1:.3e}");
    results.push(json!({"mode": "batched_vs_single", "max_abs_logit_diff": [d0, d1]}));
    if tp.rank == 0 {
        std::fs::write(out, serde_json::to_string_pretty(&json!({"T": t, "world": tp.world, "prefill_s": ta, "results": results})).unwrap()).unwrap();
    }
}

pub fn hash_check(path: &Path) {
    let h = Hash::load(path);
    let v: serde_json::Value = serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    let ids: Vec<i64> = v["check"]["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
    let want = &v["check"]["hashes"];   // [L][n_layers][cols]
    let mut hist = Vec::new();
    let mut bad = 0;
    for (p, &id) in ids.iter().enumerate() {
        hist.push(h.compress(id));
        let got = h.at(&hist[hist.len().saturating_sub(4)..]);
        for (li, row) in got.iter().enumerate() {
            let w: Vec<i64> = want[p][li].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
            if *row != w { bad += 1; if bad < 4 { eprintln!("pos {p} layer {li}: {row:?} != {w:?}"); } }
        }
    }
    eprintln!("[ds41-hash-check] {} positions, {} mismatching rows", ids.len(), bad);
    assert_eq!(bad, 0);
}

/// `ds41-bench <ckpt> <engram_hash.json|-> <prompt_len> <decode_n> [chunk]`: prefill a pseudo-random prompt in chunks,
/// then greedy-decode `decode_n` tokens; reports prefill tok/s and decode ms/token (rank 0). DS41_PROFILE=1 adds the
/// per-section breakdown (synchronizing around every section, so the totals are slower than the plain run).
pub fn bench(ckpt: &Path, hash: &str, prompt_len: usize, decode_n: usize, chunk: usize) {
    use super::model::{profiling, take_prof};
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    crate::tp::set_tf32(false);
    let dev = Device::Cuda(0);
    let m = Model::load(ckpt, tp.rank, tp.world, usize::MAX, (hash != "-").then(|| Path::new(hash)), dev);
    let mut rng = 0x9E3779B97F4A7C15u64;
    // DS41_BENCH_IDS=<file>: the prompt's token ids from a file (whitespace-separated, repeated to prompt_len; real text
    // routes unevenly, unlike the random ids)
    let ids: Vec<i64> = match std::env::var("DS41_BENCH_IDS") {
        Ok(f) => {
            let v: Vec<i64> = std::fs::read_to_string(&f).expect("DS41_BENCH_IDS").split_whitespace().map(|x| x.parse().expect("token id")).collect();
            assert!(!v.is_empty(), "DS41_BENCH_IDS: no ids");
            v.iter().cycle().take(prompt_len).copied().collect()
        }
        Err(_) => (0..prompt_len).map(|_| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng % 120000) as i64 + 1000 }).collect(),
    };
    let cap = ((prompt_len + decode_n) as i64 + 255) / 256 * 256;
    let report = |what: &str, total_ms: f64| {
        if tp.rank != 0 { let _ = take_prof(); return; }
        let mut p = take_prof();
        if !profiling() { return; }
        p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let sum: f64 = p.iter().map(|e| e.1).sum();
        let line: Vec<String> = p.iter().map(|(n, ms)| format!("{n} {ms:.1}")).collect();
        eprintln!("[ds41-bench] {what} profile (ms; sections {sum:.1} of {total_ms:.1}): {}", line.join(", "));
    };
    for round in 0..2 {
        let mut seq = m.new_seq(cap);
        let t0 = std::time::Instant::now();
        let mut x = None;
        let chunks: Vec<&[i64]> = ids.chunks(chunk).collect();
        for (i, c) in chunks.iter().enumerate() {
            let n = c.len() as i64;
            if let Some(nx) = chunks.get(i + 1) { m.engram_prefetch(c, nx); }
            if super::model::ced_on() { if let Some(v) = m.prefill_ced(&mut seq, c, i + 1 == chunks.len()) { x = Some(v); } }
            else { x = Some(m.forward(&mut seq, c, Some(&[n - 1]))); }
        }
        let lg = m.logits(&x.unwrap());
        tch::Cuda::synchronize(0);
        let pre_ms = t0.elapsed().as_secs_f64() * 1e3;
        report("prefill", pre_ms);
        let mut tok = lg.argmax(-1, false).int64_value(&[0]);
        let mut steps = Vec::new();
        let mut toks = vec![tok];
        for _ in 0..decode_n {
            let t = std::time::Instant::now();
            tok = m.step_logits(&mut seq, &[tok]).argmax(-1, false).int64_value(&[0]);
            steps.push(t.elapsed().as_secs_f64() * 1e3);
            toks.push(tok);
        }
        // DS41_BENCH_DUMP=<file> (rank 0, round 1): the prompt's last logits (FP32 bytes) and the greedy tokens after it,
        // to compare prefill paths bit for bit
        if let (0, 0, Ok(f)) = (round, tp.rank, std::env::var("DS41_BENCH_DUMP")) {
            let v: Vec<f32> = Vec::<f32>::try_from(lg.to_kind(Kind::Float).to_device(Device::Cpu).view(-1)).unwrap();
            let mut b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            b.extend(toks.iter().flat_map(|x| x.to_le_bytes()));
            std::fs::write(&f, b).expect("DS41_BENCH_DUMP");
            eprintln!("[ds41-bench] dumped to {f}: tokens {toks:?}");
        }
        report("decode", steps.iter().sum());
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if steps.is_empty() { 0.0 } else { steps[steps.len() / 2] };
        // batched decode: B sequences sharing the prompt state (copies of the prefilled sequence)
        // (rows start from different tokens, so they route to different experts as independent requests do;
        // DS41_BENCH_BATCH=2,4 the batch sizes, 0 none)
        let batches: Vec<usize> = match std::env::var("DS41_BENCH_BATCH") {
            Ok(v) if v == "0" => vec![],
            Ok(v) => v.split(',').map(|b| b.trim().parse().expect("DS41_BENCH_BATCH")).collect(),
            Err(_) => vec![2, 4],
        };
        for b in batches {
            let mut seqs: Vec<Seq> = (0..b).map(|_| m.new_seq(cap)).collect();
            for sq in seqs.iter_mut() { for c in ids.chunks(chunk) { let n = c.len() as i64; let _ = m.forward(sq, c, Some(&[n - 1])); } }
            let mut toks: Vec<i64> = (0..b as i64).map(|i| if i == 0 { tok } else { 1000 + (tok + 7919 * i) % 120000 }).collect();
            let mut st = Vec::new();
            for _ in 0..decode_n.min(32) {
                let t = std::time::Instant::now();
                let mut refs: Vec<&mut Seq> = seqs.iter_mut().collect();
                let lg = m.decode_logits(&mut refs, &toks);
                let am: Vec<i64> = Vec::<i64>::try_from(lg.argmax(-1, false).to_device(Device::Cpu)).unwrap();
                toks = am;
                st.push(t.elapsed().as_secs_f64() * 1e3);
            }
            st.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = st.get(st.len() / 2).copied().unwrap_or(0.0);
            if tp.rank == 0 { eprintln!("[ds41-bench] round {round}: batched decode B={b}: median {med:.2} ms/step = {:.1} tok/s total", b as f64 * 1e3 / med.max(1e-9)); }
        }
        if tp.rank == 0 {
            eprintln!("[ds41-bench] round {round}: prefill {prompt_len} tokens (chunk {chunk}) {pre_ms:.0} ms = {:.0} tok/s; decode {decode_n}: median {med:.2} ms/token = {:.1} tok/s",
                prompt_len as f64 / pre_ms * 1e3, 1e3 / med.max(1e-9));
        }
    }
}

/// `ds41-spec <ckpt> <engram_hash.json> <golden.safetensors> <n_new> [temperature]` (DS41_MTP=1): the golden ids as the
/// prompt, then n_new tokens by plain decoding and by DSpark speculative decoding (same sampling keys); reports how
/// many tokens agree (exact for the method up to the verify chunk's own summation order), the accepted drafts per
/// round and the speed of both.
pub fn spec(ckpt: &Path, hash: &str, golden: &Path, n_new: usize, temp: f32) {
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    crate::tp::set_tf32(false);
    let dev = Device::Cuda(0);
    let gold: std::collections::HashMap<String, Tensor> = Tensor::read_safetensors(golden).expect("golden").into_iter().collect();
    let mut ids: Vec<i64> = Vec::<i64>::try_from(&gold["ids"]).unwrap();
    if let Some(n) = std::env::var("DS41_SPEC_PROMPT").ok().and_then(|v| v.parse::<usize>().ok()) { ids.truncate(n); }
    let m = Model::load(ckpt, tp.rank, tp.world, usize::MAX, (hash != "-").then(|| Path::new(hash)), dev);
    assert!(m.mtp.is_some(), "DS41_MTP=1 required");
    let cap = ((ids.len() + n_new + 64) as i64 + 255) / 256 * 256;
    let seed = 1234u64;
    let prefill = |seq: &mut Seq| -> Tensor {
        let mut lg = None;
        for c in ids.chunks(2048) {
            let n = c.len() as i64;
            let x = m.forward(seq, c, Some(&[n - 1]));
            if let Some(h) = m.last_hidden.borrow_mut().take() { let p0 = seq.pos - h.size()[0]; m.mtp_seed(seq, &h, p0); }
            lg = Some(m.logits(&x));
        }
        lg.unwrap()
    };
    // DS41_SPEC_SWEEP=p1,p2,..: one round per DS41_SPEC_MIN_P value (after a warm-up round with the first)
    let sweep: Vec<String> = std::env::var("DS41_SPEC_SWEEP").map(|v| v.split(',').map(|x| x.to_string()).collect()).unwrap_or_default();
    let n_rounds = if sweep.is_empty() { 2 } else { sweep.len() + 1 };
    for round in 0..n_rounds {
        if !sweep.is_empty() {
            let v = &sweep[round.saturating_sub(1)];
            std::env::set_var("DS41_SPEC_MIN_P", v);
            if tp.rank == 0 { eprintln!("[ds41-spec] round {round}: DS41_SPEC_MIN_P={v}"); }
        }
        // plain
        let mut seq = m.new_seq(cap);
        let lg = prefill(&mut seq);
        let mut tok = super::serve::sample(&lg, temp, seed, seq.pos);
        let mut plain = vec![tok];
        tch::Cuda::synchronize(0);
        let t0 = std::time::Instant::now();
        while plain.len() < n_new {
            let p = seq.pos + 1;
            tok = super::serve::sample(&m.step_logits(&mut seq, &[tok]), temp, seed, p);
            plain.push(tok);
        }
        tch::Cuda::synchronize(0);
        let t_plain = t0.elapsed().as_secs_f64();
        // speculative
        let mut seq = m.new_seq(cap);
        let lg = prefill(&mut seq);
        let mut tok = super::serve::sample(&lg, temp, seed, seq.pos);
        let mut spec = vec![tok];
        let mut hist = vec![0usize; 8];
        let mut rounds = 0;
        let mut drafted = 0;
        tch::Cuda::synchronize(0);
        let t0 = std::time::Instant::now();
        while spec.len() < n_new {
            let (out, nd) = m.spec_round(&mut seq, tok, temp, seed);
            drafted += nd;
            hist[(out.len() - 1).min(7)] += 1;
            rounds += 1;
            tok = *out.last().unwrap();
            spec.extend(out);
        }
        tch::Cuda::synchronize(0);
        let t_spec = t0.elapsed().as_secs_f64();
        let pr = super::model::take_prof();
        if tp.rank == 0 && super::model::profiling() { eprintln!("[ds41-spec] round {round} profile (ms over {rounds} rounds): {pr:?}"); }
        spec.truncate(n_new);
        let lead = plain.iter().zip(&spec).take_while(|(a, b)| a == b).count();
        let same = plain.iter().zip(&spec).filter(|(a, b)| a == b).count();
        // teacher-forced plain decoding along the speculative output: where its greedy token differs, the logit gap
        // between its choice and the speculative token (near ties = summation-order noise of the verify chunks)
        let mut seq = m.new_seq(cap);
        let _ = prefill(&mut seq);
        let mut gaps = Vec::new();
        for i in 0..spec.len() - 1 {
            let lg = m.step_logits(&mut seq, &[spec[i]]);
            let am = lg.argmax(-1, false).int64_value(&[0]);
            if am != spec[i + 1] && temp == 0.0 {
                gaps.push((i + 1, lg.double_value(&[0, am]) - lg.double_value(&[0, spec[i + 1]])));
            }
        }
        if tp.rank == 0 { eprintln!("[ds41-spec] round {round}: teacher-forced plain disagrees at {} of {} positions: (position, logit gap) {:?}", gaps.len(), spec.len() - 1, gaps); }
        if tp.rank == 0 {
            eprintln!("[ds41-spec] round {round} temp {temp}: plain {:.1} tok/s, spec {:.1} tok/s ({rounds} rounds, {:.2} tokens/round, {:.2} drafts verified/round, accepted-drafts histogram {:?})",
                n_new as f64 / t_plain, n_new as f64 / t_spec, n_new as f64 / rounds as f64, drafted as f64 / rounds as f64, &hist[..7]);
            eprintln!("[ds41-spec] round {round}: identical leading tokens {lead}/{n_new}, positionwise equal {same}/{n_new}");
        }
    }
}

/// `ds41-dspark-golden <ckpt> <engram_hash.json> <dspark.safetensors>` (DS41_MTP=1): the draft of bench/ds41/ref/
/// ref_dspark.py (official DSparkBlock) against the engine's: prefill the prompt, seed the stages, draft from the
/// golden's token; reports draft ids, confidences and the per-step draft logits (top-1 agreement, KL).
pub fn dspark_golden(ckpt: &Path, hash: &str, golden: &Path) {
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    crate::tp::set_tf32(false);
    let dev = Device::Cuda(0);
    let gold: std::collections::HashMap<String, Tensor> = Tensor::read_safetensors(golden).expect("golden").into_iter().collect();
    let ids: Vec<i64> = Vec::<i64>::try_from(&gold["ids"]).unwrap();
    let t = gold["t"].int64_value(&[0]);
    let m = Model::load(ckpt, tp.rank, tp.world, usize::MAX, (hash != "-").then(|| Path::new(hash)), dev);
    let cap = ((ids.len() + 64) as i64 + 255) / 256 * 256;
    let mut seq = m.new_seq(cap);
    let mut lg = None;
    for c in ids.chunks(2048) {
        let n = c.len() as i64;
        let x = m.forward(&mut seq, c, Some(&[n - 1]));
        if let Some(h) = m.last_hidden.borrow_mut().take() { let p0 = seq.pos - h.size()[0]; m.mtp_seed(&mut seq, &h, p0); }
        lg = Some(m.logits(&x));
    }
    let main_t = lg.unwrap().argmax(-1, false).int64_value(&[0]);
    *m.draft_debug.borrow_mut() = Some(Vec::new());
    let (d, cf) = m.mtp_draft(&[&mut seq], &[t], &[0.0], &[0]);
    let rows = m.draft_debug.borrow_mut().take().unwrap();
    let ours = Tensor::cat(&rows, 0);
    let refl = gold["draft_logits"].to_device(dev);
    let r = compare("draft_logits", &refl, &ours);
    if tp.rank == 0 {
        eprintln!("[ds41-dspark-golden] main greedy token {main_t} (golden {t}); drafts {:?} (golden {:?})", d[0], Vec::<i64>::try_from(&gold["draft_ids"]).unwrap());
        eprintln!("[ds41-dspark-golden] confidence {:?} (golden {:?})", cf[0], Vec::<f32>::try_from(&gold["confidence"]).unwrap());
        eprintln!("[ds41-dspark-golden] {r}");
    }
}

/// `ds41-ar-bench`: the TP allreduce at prefill sizes (FP32 [rows, 5120]), time per call and effective rate, plus a
/// bitwise check against the rank-ordered sum computed locally from every rank's known input.
pub fn ar_bench() {
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    let dev = Device::Cuda(0);
    // decode-size latency: 100 back-to-back all-reduces of 1..4 rows inside one CUDA graph (as in the decode graphs)
    for (i, n) in [4i64, 1024, 5120, 20480].into_iter().enumerate() {
        use crate::tp::graph::{begin_pool, end, Owned};
        let x = Tensor::ones([1, n], (Kind::Float, dev));
        crate::tp::allreduce(&x);
        tch::Cuda::synchronize(0);
        begin_pool(0xa11 + i as i64).expect("capture begin");
        for _ in 0..100 { crate::tp::allreduce(&x); }
        end().expect("capture end");
        let g = Owned::take();
        g.replay();
        tch::Cuda::synchronize(0);
        let t = std::time::Instant::now();
        for _ in 0..10 { g.replay(); }
        tch::Cuda::synchronize(0);
        let us = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
        if tp.rank == 0 { eprintln!("[ds41-ar-bench] graph: {n} floats ({:.1} KB): {us:.1} us per all-reduce", n as f64 * 4.0 / 1024.0); }
    }
    // decode-size row all-gather (rdma_arn rows, ag): each rank's [T, 608] BF16 block (a third of the joint q / kv
    // projection's columns), 100 back-to-back inside one graph
    for (i, t) in [1i64, 2, 4, 8].into_iter().enumerate() {
        use crate::tp::graph::{begin_pool, end, Owned};
        extern "C" { fn rs_sp_all_gather(i: *const std::ffi::c_void, o: *mut std::ffi::c_void, bytes: i64, async_: i32) -> i32; }
        let x = (Tensor::arange(t * 608, (Kind::Float, dev)) + 1000.0 * tp.rank as f64).to_kind(Kind::BFloat16);
        let out = Tensor::zeros([tp.world as i64 * t * 608], (Kind::BFloat16, dev));
        let ag = || assert_eq!(unsafe { rs_sp_all_gather(x.data_ptr(), out.data_ptr(), t * 608 * 2, 0) }, 0, "rs_sp_all_gather");
        ag();
        tch::Cuda::synchronize(0);
        let want = Tensor::cat(&(0..tp.world).map(|r| (Tensor::arange(t * 608, (Kind::Float, dev)) + 1000.0 * r as f64).to_kind(Kind::BFloat16)).collect::<Vec<_>>(), 0);
        let same = out.equal(&want);
        begin_pool(0xa21 + i as i64).expect("capture begin");
        for _ in 0..100 { ag(); }
        end().expect("capture end");
        let g = Owned::take();
        g.replay();
        tch::Cuda::synchronize(0);
        let t0 = std::time::Instant::now();
        for _ in 0..10 { g.replay(); }
        tch::Cuda::synchronize(0);
        let us = t0.elapsed().as_secs_f64() * 1e6 / 1000.0;
        if tp.rank == 0 { eprintln!("[ds41-ar-bench] graph: all-gather [{t} x 608] BF16 per rank: {us:.1} us, correct: {same}"); }
    }
    for rows in [1i64, 64, 512, 2048, 4096] {
        let n = rows * 5120;
        let mk = |r: usize| Tensor::arange(n, (Kind::Float, dev)).sin() * (r as f64 + 1.0) + 0.1 * r as f64;
        let want = (mk(0) + mk(1)) + mk(2);
        let x = mk(tp.rank);
        crate::tp::allreduce(&x);
        let same = x.equal(&want);
        let iters = if rows >= 2048 { 10 } else { 50 };
        tch::Cuda::synchronize(0);
        let t = std::time::Instant::now();
        for _ in 0..iters { let y = mk(tp.rank); crate::tp::allreduce(&y); }
        tch::Cuda::synchronize(0);
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        if tp.rank == 0 {
            eprintln!("[ds41-ar-bench] rows {rows:5} ({:6.1} MB): {us:9.1} us/call (incl. input), {:5.1} GB/s of buffer, bitwise rank-ordered: {same}",
                      n as f64 * 4.0 / 1e6, n as f64 * 4.0 / us / 1e3);
        }
    }
}

/// `ds41-ckpt-test <ckpt> <engram_hash> <p> [len]` (every rank; DS41 prefix cache): the state captured at position p inside
/// one prefill chunk of `len` (4096) tokens against the state of a prefill cut at p (window rings, compressor open
/// groups, CED tail, compressed rows, n-gram history), then both continued with the same 300 tokens: last-row logits.
pub fn ckpt_test(ckpt: &Path, hash: &str, p: i64, len: i64) {
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    crate::tp::set_tf32(false);
    let dev = Device::Cuda(0);
    let m = Model::load(ckpt, tp.rank, tp.world, usize::MAX, (hash != "-").then(|| Path::new(hash)), dev);
    let mut rng = 0x2545F4914F6CDD1Du64;
    let ids: Vec<i64> = (0..len + 300).map(|_| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng % 120000) as i64 + 1000 }).collect();
    let cap = 8192;
    let mut a = m.new_seq(cap);
    a.ck_req = vec![p];
    let _ = m.prefill_ced(&mut a, &ids[..len as usize], false);
    let capt = a.ck_cap.remove(&p).expect("no capture");
    let mut b = m.new_seq(cap);
    let _ = m.prefill_ced(&mut b, &ids[..p as usize], false);
    let mut report = Vec::new();
    let mut cmp = |name: String, x: &Tensor, y: &Tensor| {
        let d = (x.to_kind(Kind::Float) - y.to_kind(Kind::Float)).abs().max().double_value(&[]);
        if d != 0.0 || name.starts_with("win.0") { report.push(format!("{name} {d:.3e}")); }
        d
    };
    let mut worst = 0f64;
    for (l, w) in &capt.win { worst = worst.max(cmp(format!("win.{l}"), w, &b.win[*l])); }
    for (s, (k, sc)) in &capt.cs {
        let r = p % m.c.compress[*s];
        if r > 0 { worst = worst.max(cmp(format!("csk.{s}"), &k.narrow(0, 0, r), &b.cstate[s].0.narrow(0, 0, r))); worst = worst.max(cmp(format!("css.{s}"), &sc.narrow(0, 0, r), &b.cstate[s].1.narrow(0, 0, r))); }
    }
    let (ch, cp) = capt.ced.as_ref().expect("no ced capture");
    let (bh, bp, be) = b.ced_tail.as_ref().unwrap();
    assert_eq!(*be, p);
    worst = worst.max(cmp("ced_h".into(), ch, bh));
    worst = worst.max(cmp("ced_pm".into(), cp, bp));
    for (s, t) in &b.ckv { let r = m.c.compress[*s]; worst = worst.max(cmp(format!("ckv.{s}"), &a.ckv[s].narrow(0, 0, p / r), &t.narrow(0, 0, p / r))); worst = worst.max(cmp(format!("ikv.{s}"), &a.ikv[s].narrow(0, 0, p / r), &b.ikv[s].narrow(0, 0, p / r))); }
    let hist_same = a.hist[..p as usize] == b.hist[..];
    // continue: a fresh sequence from the capture (as a restore would build it) vs b, the same 300 tokens
    let mut c2 = m.new_seq(cap);
    for (l, w) in c2.win.iter_mut().enumerate() { let _ = w.copy_(capt.win.get(&l).unwrap_or(&a.win[l])); }
    for (s, (k, sc)) in c2.cstate.iter_mut() { let (ck, cs) = &capt.cs[s]; let _ = k.copy_(ck); let _ = sc.copy_(cs); }
    for (s, t) in c2.ckv.iter_mut() { let r = m.c.compress[*s]; let _ = t.narrow(0, 0, p / r).copy_(&a.ckv[s].narrow(0, 0, p / r)); let _ = c2.ikv.get_mut(s).unwrap().narrow(0, 0, p / r).copy_(&a.ikv[s].narrow(0, 0, p / r)); }
    for (i, w) in c2.mtp_win.iter_mut().enumerate() { let _ = w.copy_(&a.mtp_win[i]); }
    c2.hist = a.hist[..p as usize].to_vec();
    c2.pos = p; c2.hw = p; c2.dec_pos = a.dec_pos;
    c2.ced_tail = Some((ch.shallow_clone(), cp.shallow_clone(), p));
    let rest = &ids[p as usize..(p + 300) as usize];
    // a checkpoint with dec_pos + window <= p stores no decoder rings (Model::pcache_win): a restore leaves them as they
    // were reset; filled with NaN instead, the continuation must give the same logits (never read)
    let enc = (m.c.n_layers / 2) as usize;
    let canary = (a.dec_pos + m.c.window <= p).then(|| {
        let mut c3 = m.new_seq(cap);
        for (l, w) in c3.win.iter_mut().enumerate() {
            if l < enc { let _ = w.copy_(capt.win.get(&l).unwrap_or(&a.win[l])); } else { let _ = w.fill_(f64::NAN); }
        }
        for (s, (k, sc)) in c3.cstate.iter_mut() { let (ck, cs) = &capt.cs[s]; let _ = k.copy_(ck); let _ = sc.copy_(cs); }
        for (s, t) in c3.ckv.iter_mut() { let r = m.c.compress[*s]; let _ = t.narrow(0, 0, p / r).copy_(&a.ckv[s].narrow(0, 0, p / r)); let _ = c3.ikv.get_mut(s).unwrap().narrow(0, 0, p / r).copy_(&a.ikv[s].narrow(0, 0, p / r)); }
        for (i, w) in c3.mtp_win.iter_mut().enumerate() { let _ = w.copy_(&a.mtp_win[i]); }
        c3.hist = a.hist[..p as usize].to_vec();
        c3.pos = p; c3.hw = p; c3.dec_pos = a.dec_pos;
        c3.ced_tail = Some((ch.shallow_clone(), cp.shallow_clone(), p));
        let l3 = m.logits(&m.prefill_ced(&mut c3, rest, true).unwrap());
        let bad = c3.win.iter().map(|w| w.isnan().sum(Kind::Int64).int64_value(&[])).sum::<i64>();
        (l3, bad)
    });
    let la = m.logits(&m.prefill_ced(&mut c2, rest, true).unwrap());
    if let Some((l3, bad)) = &canary {
        let d = (&la - l3).abs().max().double_value(&[]);
        if tp.rank == 0 { eprintln!("[ds41-ckpt-test] NaN decoder rings (dec_pos {} <= p {p} - window): continued logits max |diff| {d:.3e}, NaN left in rings {bad}", a.dec_pos); }
        assert!(d == 0.0 && *bad == 0, "decoder rings of a checkpoint without them were read");
    }
    let lb = m.logits(&m.prefill_ced(&mut b, rest, true).unwrap());
    let ld = (&la - &lb).abs().max().double_value(&[]);
    // chunking noise: the same tokens in one chunk of p + 300
    let mut f = m.new_seq(cap);
    let lf = m.logits(&m.prefill_ced(&mut f, &ids[..(p + 300) as usize], true).unwrap());
    let kl = |x: &Tensor, y: &Tensor| { let (px, py) = (x.log_softmax(-1, Kind::Float), y.log_softmax(-1, Kind::Float)); (px.exp() * (&px - &py)).sum(Kind::Float).double_value(&[]) };
    if tp.rank == 0 {
        eprintln!("[ds41-ckpt-test] KL: capture-continued vs cut-at-p {:.4}, one-chunk vs cut-at-p {:.4} (chunking noise), capture vs one-chunk {:.4}", kl(&lb, &la), kl(&lb, &lf), kl(&lf, &la));
        eprintln!("[ds41-ckpt-test] p {p} inside a {len}-token chunk: worst state |diff| {worst:.3e}, history {}; {}", if hist_same { "same" } else { "DIFFERENT" }, report.join(", "));
        eprintln!("[ds41-ckpt-test] continued 300 tokens: last-row logits max |diff| {ld:.3e} (top-1 {} vs {})", la.argmax(-1, false).int64_value(&[0]), lb.argmax(-1, false).int64_value(&[0]));
    }
}

/// Prefill one prompt into a fresh sequence, chunked at DS41_PREFILL_CHUNK (default 2048), nothing generated: the
/// shared body of the calibration / extraction feeds below.
fn prefill_prompt(m: &Model, ids: &[i64]) {
    let cap = ((ids.len() as i64 + 255) / 256) * 256;
    let mut seq = m.new_seq(cap);
    let chunk = std::env::var("DS41_PREFILL_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048);
    let n = ids.len();
    for (i, c) in ids.chunks(chunk).enumerate() {
        let _ = m.prefill_ced(&mut seq, c, (i + 1) * chunk >= n);
    }
    tch::Cuda::synchronize(0);
}

/// `ds41-refusal-dir <ckpt> <harmful_ids.json> <harmless_ids.json> <engram_hash.json> <out.safetensors>` (all TP3
/// ranks, rank 0 writes): the GLM `glm-refusal-dir` flow for DS41 (docs/05-消融.md §2/§3). Every prompt is prefilled
/// (CED, eager) with the ablate capture on; per layer the raw attention-input z's last SPARK_ABLATE_POSITIONS rows
/// (default 8, the assistant turn's opening tokens) are recorded, and mean(harmful) - mean(harmless) per layer and
/// position gives `directions_pos` / `directions` / `scores` / `direction`. Run with SPARK_ABLATE_CLAMP unset and
/// DS41_OVERLAP_ROWS=0 (the two-micro-batch path bypasses layer_forward, where the capture hook sits).
pub fn refusal_dir(ckpt: &Path, harmful: &Path, harmless: &Path, hash: &Path, out: &Path) {
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    crate::tp::set_tf32(false);
    let m = Model::load(ckpt, tp.rank, tp.world, usize::MAX, Some(hash), Device::Cuda(0));
    let load = |p: &Path| -> Vec<Vec<i64>> { serde_json::from_reader(std::fs::File::open(p).unwrap_or_else(|e| panic!("{p:?}: {e}"))).unwrap() };
    let mut run = |set: &[Vec<i64>], name: &str| -> Vec<Vec<Tensor>> {
        let t0 = std::time::Instant::now();
        let r: Vec<Vec<Tensor>> = set.iter().map(|ids| {
            crate::ablate::capture_start();
            prefill_prompt(&m, ids);
            crate::ablate::capture_take()
        }).collect();
        eprintln!("[ds41-refusal-dir] rank{} {name}: {} prompts in {:.1}s", tp.rank, set.len(), t0.elapsed().as_secs_f64());
        r
    };
    let h = run(&load(harmful), "harmful");
    let n = run(&load(harmless), "harmless");
    if tp.rank == 0 { crate::ablate::save_directions(&h, &n, out); }
}

/// `ds41-feed <ckpt> <engram_hash.json> <ids.json>` (all TP3 ranks): teacher-forced prefill of every prompt, no
/// generation -- the calibration feed for SPARK_ABLATE_CLAMP_STATS (docs/05-消融.md §3): the clamp STATS mode itself
/// appends every site's projections to <dir>/L<layer>-<site>.f32 and the row counts to <dir>/calls.txt (rank 0). Use a
/// +-inf threshold skeleton file so nothing is edited while recording, and keep the feed order (normal corpus first,
/// then the 256 harmless prompts) that clamp_fit.py expects.
pub fn feed(ckpt: &Path, hash: &Path, ids_json: &Path) {
    let _g = tch::no_grad_guard();
    let tp = crate::tp::init_from_env();
    crate::tp::set_tf32(false);
    let m = Model::load(ckpt, tp.rank, tp.world, usize::MAX, Some(hash), Device::Cuda(0));
    let set: Vec<Vec<i64>> = serde_json::from_reader(std::fs::File::open(ids_json).unwrap_or_else(|e| panic!("{ids_json:?}: {e}"))).unwrap();
    let t0 = std::time::Instant::now();
    for (i, ids) in set.iter().enumerate() {
        prefill_prompt(&m, ids);
        if i % 32 == 0 || i + 1 == set.len() { eprintln!("[ds41-feed] rank{} {}/{} ({:.1}s)", tp.rank, i + 1, set.len(), t0.elapsed().as_secs_f64()); }
    }
    eprintln!("[ds41-feed] rank{} {} prompts, {} tokens, in {:.1}s", tp.rank, set.len(), set.iter().map(|v| v.len()).sum::<usize>(), t0.elapsed().as_secs_f64());
}

/// `ds41-clamp-test` (single GPU, SPARK_ABLATE_CLAMP=<file> [SPARK_ABLATE_CLAMP_MIN=m]): the fused rms_norm vec
/// epilogue (shim/ds41_ops.cu, docs/05-消融.md §3a) against the ATen reference -- fp32 z = w * x * rstd from the same
/// BF16 inputs, per-site clamp with the count gate and the reset at site (0, 0), two chained sites like one layer.
pub fn clamp_norm_test() {
    let _g = tch::no_grad_guard();
    let c = crate::ablate::clamp().expect("SPARK_ABLATE_CLAMP");
    let dev = Device::Cuda(0);
    tch::manual_seed(42);
    let (rows, n) = (512i64, 5120i64);
    let (w0, w1) = (Tensor::randn([n], (Kind::BFloat16, dev)) * 0.5, Tensor::randn([n], (Kind::BFloat16, dev)) * 0.5);
    c.register(&w0, 0, 0);
    c.register(&w1, 0, 1);
    let mut z = Tensor::randn([rows, n], (Kind::BFloat16, dev)) * 2.0;
    let mut count = vec![0i32; rows as usize];
    let mut worst = 0f64;
    for site in [0usize, 1usize] {
        let w = if site == 0 { &w0 } else { &w1 };
        let y = super::kern::rms_norm_cl(&z, w, 1e-5, &c.norm_args(w, rows).expect("site args")).expect("fused path");
        let xf = z.to_kind(Kind::Float);
        let rstd = (xf.square().mean_dim([-1].as_slice(), true, Kind::Float) + 1e-5).rsqrt();
        let z32 = &xf * w.to_kind(Kind::Float) * &rstd;
        let (tau, lo, tg) = c.site_thresholds(0, site);
        let cc = z32.matmul(&c.dirs().tr());                                   // [rows, k]
        let trig = cc.gt_tensor(&tau).logical_or(&cc.lt_tensor(&lo));
        let e = match &tg {
            Some(t) => (&cc - t) * trig.to_kind(Kind::Float),
            None => (&cc - &tau).relu() - (&lo - &cc).relu(),
        };
        let trig_n: Vec<i64> = Vec::try_from(trig.to_kind(Kind::Int64).sum_dim_intlist([1].as_slice(), false, Kind::Int64).to_device(Device::Cpu).reshape([-1])).unwrap();
        let mut ev: Vec<Tensor> = Vec::new();
        for r in 0..rows as usize {
            let t = (trig_n[r] > 0) as i32;
            let base = if site == 0 { 0 } else { count[r] };                    // reset at layer-0 attention input
            let nnow = base + t;
            count[r] = nnow;
            ev.push(if nnow >= c.min_m() { e.narrow(0, r as i64, 1) } else { Tensor::zeros([1, cc.size()[1]], (Kind::Float, dev)) });
        }
        let zref = &z32 - &Tensor::cat(&ev, 0).matmul(c.dirs());
        // the kernel stores BF16: compare against the rounded reference (the fp32 clamp orders differ slightly, so a
        // few elements may sit on the other side of a rounding boundary -- one ulp at most)
        let zb = zref.to_kind(Kind::BFloat16);
        let mism = y.ne_tensor(&zb).to_kind(Kind::Float).mean(Kind::Float).double_value(&[]);
        let d = (y.to_kind(Kind::Float) - &zref).square().sum(Kind::Float).double_value(&[]).sqrt()
            / zref.square().sum(Kind::Float).double_value(&[]).sqrt();
        eprintln!("[ds41-clamp-test] site {site}: rel-vs-fp32 {d:.2e}, bf16 mismatched {mism:.2e} (min {}, k {})", c.min_m(), c.dirs().size()[0]);
        worst = worst.max(mism);
        z = y;
    }
    eprintln!("[ds41-clamp-test] worst bf16 mismatch rate {worst:.2e} ({})", if worst < 2e-3 { "PASS" } else { "FAIL" });
    assert!(worst < 2e-3, "ds41-clamp-test");
}
