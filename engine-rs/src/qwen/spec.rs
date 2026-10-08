//! Greedy speculative decoding with the native MTP head (k drafts per round, chain verify, replay commit).
//! Round: drafts d1..dk (MTP, autoregressive), target verifies [next, d1..dk] without writing state,
//! accepts the agreeing prefix a, commits a + 1 rows, then the MTP catches up on the accepted rows
//! (target stacks + next tokens at the target positions), which also yields the next round's d1.
use super::model::{prefill_macro, Model, Seq};
use tch::{Device, Tensor};

pub const EOS: [i64; 2] = [248046, 248044];

fn argmax_rows(lg: &Tensor) -> Vec<i64> {
    Vec::<i64>::try_from(lg.argmax(-1, false).to_device(Device::Cpu)).unwrap()
}

#[derive(Default, Debug)]
pub struct Stats { pub rounds: usize, pub drafted: usize, pub accepted: usize, pub ms: [f64; 4] }

/// QWEN_TIMING=1: synchronize around the round phases (draft, verify, commit, catch-up) and accumulate ms.
struct Clock { on: bool, t: std::time::Instant }
impl Clock {
    fn new() -> Self { Clock { on: std::env::var("QWEN_TIMING").is_ok(), t: std::time::Instant::now() } }
    fn lap(&mut self, acc: &mut f64) {
        if !self.on { return; }
        tch::Cuda::synchronize(0);
        *acc += self.t.elapsed().as_secs_f64() * 1e3;
        self.t = std::time::Instant::now();
    }
}

/// Prefill `prompt` (target + MTP catch-up per chunk); returns (first sampled token, MTP last stack, its position,
/// target stack of the last prompt row).
pub fn prefill(m: &Model, seq: &mut Seq, prompt: &[i64]) -> (i64, Tensor) {
    let mut pending: Option<(Tensor, i64)> = None;
    let mut last = None;
    let cut = seq.pos + m.mtp_tail_from(prompt.len());
    for c in prompt.chunks(prefill_macro() as usize) {
        let p0 = seq.pos;
        let n = c.len() as i64;
        let (lg, x) = m.prefill_lm(seq, c);
        if m.mtp.is_some() {
            let mut hs = Vec::new();
            let mut ids = Vec::new();
            let mut start = p0;
            if let Some((h, pos)) = pending.take() { hs.push(h); ids.push(c[0]); start = pos; }
            if n > 1 { hs.push(x.narrow(0, 0, n - 1)); ids.extend_from_slice(&c[1..]); }
            mtp_catch_up(m, seq, hs, ids, start, cut);
            pending = Some((x.narrow(0, n - 1, 1).contiguous(), p0 + n - 1));
        }
        last = Some(lg);
    }
    let next = argmax_rows(&last.unwrap())[0];
    (next, pending.map(|(h, p)| { let _ = p; h }).unwrap_or_else(|| Tensor::zeros([1], (tch::Kind::Float, Device::Cuda(0)))))
}

/// MTP catch-up of prompt rows (hidden stacks hs, next tokens ids, from position start), skipping rows before `cut`
/// (QWEN_MTP_PREFILL_TAIL; their MTP cache rows zeroed).
pub fn mtp_catch_up(m: &Model, seq: &mut Seq, hs: Vec<Tensor>, ids: Vec<i64>, start: i64, cut: i64) {
    if ids.is_empty() { return; }
    let skip = (cut - start).clamp(0, ids.len() as i64);
    m.mtp_clear(seq, start, start + skip);
    if skip as usize == ids.len() { return; }
    let hidden = Tensor::cat(&hs, 0);
    let hidden = hidden.narrow(0, skip, ids.len() as i64 - skip).contiguous();
    m.mtp_prefill(seq, &hidden, &ids[skip as usize..], start + skip);
}

pub fn generate(m: &Model, seq: &mut Seq, prompt: &[i64], n_new: usize, k: usize, stats: &mut Stats) -> Vec<i64> {
    let (mut next, h_last) = prefill(m, seq, prompt);
    let mut out = vec![next];
    // d1 from the last prompt row (target stack at P-1, token t_P = next)
    let (mut lg_d, mut mul) = m.mtp_run(seq, &h_last, &[next], seq.pos - 1, true);
    let mut clk = Clock::new();
    clk.lap(&mut 0.0);
    while out.len() < n_new && !EOS.contains(&next) {
        let p = seq.pos;
        let mut d = vec![m.draft_token(argmax_rows(&lg_d)[0])];
        let mut multi = mul.narrow(0, mul.size()[0] - 1, 1).contiguous();
        for j in 1..k {
            let (l2, m2) = m.mtp_run(seq, &multi, &[d[j - 1]], p + j as i64 - 1, true);
            d.push(m.draft_token(argmax_rows(&l2)[0]));
            multi = m2;
        }
        clk.lap(&mut stats.ms[0]);
        let mut rows = vec![next];
        rows.extend_from_slice(&d);
        let (lg, x) = m.run(seq, &rows, false, false);
        let g = argmax_rows(&lg);
        clk.lap(&mut stats.ms[1]);
        let mut a = 0;
        while a < k && g[a] == d[a] { a += 1; }
        let bonus = g[a];
        m.commit(seq, a as i64 + 1);
        clk.lap(&mut stats.ms[2]);
        stats.rounds += 1; stats.drafted += k; stats.accepted += a;
        out.extend_from_slice(&d[..a]);
        out.push(bonus);
        next = bonus;
        if let Some(e) = out.iter().position(|t| EOS.contains(t)) { out.truncate(e + 1); break; }
        // MTP catch-up on the accepted rows: (target stack at p + i, token p + i + 1)
        let mut ids = d[..a].to_vec();
        ids.push(bonus);
        let (l3, m3) = m.mtp_run(seq, &x.narrow(0, 0, a as i64 + 1).contiguous(), &ids, p, true);
        lg_d = l3; mul = m3;
        clk.lap(&mut stats.ms[3]);
    }
    out.truncate(n_new);
    out
}

/// Plain greedy decoding (one row per step), the reference for the speculative output.
pub fn generate_plain(m: &Model, seq: &mut Seq, prompt: &[i64], n_new: usize) -> Vec<i64> {
    let mut lg = m.prefill(seq, prompt);
    let mut out = Vec::new();
    while out.len() < n_new {
        let t = argmax_rows(&lg)[0];
        out.push(t);
        if EOS.contains(&t) { break; }
        lg = m.step(seq, &[t], true);
    }
    out
}

/// Same round as `generate`, every forward a CUDA-graph replay (graphs captured once for `seq`).
pub fn generate_graph(m: &Model, seq: &mut Seq, prompt: &[i64], n_new: usize, k: usize, stats: &mut Stats) -> Vec<i64> {
    use super::graph::Decoder;
    let (mut next, h_last) = prefill(m, seq, prompt);
    let dec = Decoder::capture(m, seq, k);
    let mut out = vec![next];
    let (lg0, x0) = dec.mtp(&h_last, &[next], seq.pos - 1);
    let mut d1 = m.draft_token(argmax_rows(lg0)[0]);
    let mut multi = x0.shallow_clone();
    let mut clk = Clock::new();
    clk.lap(&mut 0.0);
    while out.len() < n_new && !EOS.contains(&next) {
        let p = seq.pos;
        let mut d = vec![d1];
        for j in 1..k {
            let (l2, x2) = dec.mtp(&multi, &[d[j - 1]], p + j as i64 - 1);
            d.push(m.draft_token(argmax_rows(l2)[0]));
            multi = x2.shallow_clone();
        }
        clk.lap(&mut stats.ms[0]);
        let mut rows = vec![next];
        rows.extend_from_slice(&d);
        let (lg, x) = dec.verify(m, seq, &rows);
        let g = argmax_rows(lg);
        clk.lap(&mut stats.ms[1]);
        let mut a = 0;
        while a < k && g[a] == d[a] { a += 1; }
        let bonus = g[a];
        dec.commit(seq, &rows, a + 1);
        clk.lap(&mut stats.ms[2]);
        stats.rounds += 1; stats.drafted += k; stats.accepted += a;
        out.extend_from_slice(&d[..a]);
        out.push(bonus);
        next = bonus;
        if let Some(e) = out.iter().position(|t| EOS.contains(t)) { out.truncate(e + 1); break; }
        let mut ids = d[..a].to_vec();
        ids.push(bonus);
        let (l3, x3) = dec.mtp(&x.narrow(0, 0, a as i64 + 1), &ids, p);
        d1 = m.draft_token(argmax_rows(l3)[0]);
        multi = x3.narrow(0, a as i64, 1);
        clk.lap(&mut stats.ms[3]);
    }
    out.truncate(n_new);
    out
}
