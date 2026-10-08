//! Prompt-lookup drafts (L2: drafts only, the verified output is unchanged). The proposal is the continuation of the
//! latest earlier occurrence of the sequence's last QWEN_LOOKUP_N (default 3) tokens, prompt and output alike; it is used
//! when it starts with the MTP head's first draft, and then replaces the rest of that sequence's MTP chain (no further
//! draft steps for it). Its per-token confidence for the draft-length policy is the sequence's running acceptance rate of
//! its proposals, measured every round against the tokens that actually followed (whether the proposal was used or not);
//! below QWEN_LOOKUP_MIN (default 0.6) proposals are not used. QWEN_LOOKUP=0: off.
use std::collections::HashMap;

pub struct Lookup { n: usize, map: HashMap<u64, usize>, upto: usize, hits: f64, tries: f64,
                    /// running tokens per round with lookup drafts / with MTP drafts, and rounds seen
                    per_l: Option<f64>, per_m: Option<f64>, rounds: u64 }

pub fn enabled() -> bool { std::env::var("QWEN_LOOKUP").as_deref() != Ok("0") }

impl Default for Lookup { fn default() -> Self { Self::new() } }

impl Lookup {
    pub fn new() -> Self {
        let n = std::env::var("QWEN_LOOKUP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(3usize).clamp(1, 8);
        // prior: 9 of 10 accepted
        Lookup { n, map: HashMap::new(), upto: 0, hits: 9.0, tries: 10.0, per_l: None, per_m: None, rounds: 0 }
    }
    fn key(t: &[i64]) -> u64 { t.iter().fold(0xcbf29ce484222325u64, |h, &v| (h ^ v as u64).wrapping_mul(0x100000001b3)) }

    /// Index the n-grams of hist (each followed by a token: the next one of hist, or `next` for the last).
    fn update(&mut self, hist: &[i64]) {
        let n = self.n;
        for e in self.upto.max(n - 1)..hist.len() { self.map.insert(Self::key(&hist[e + 1 - n..=e]), e); }
        self.upto = self.upto.max(hist.len());
    }

    /// The sequence so far is hist ++ [next] (append-only across calls; the drafts continue after `next`): up to `max`
    /// tokens following the latest earlier occurrence of its last n tokens, with the per-token confidence.
    pub fn propose(&mut self, hist: &[i64], next: i64, max: usize) -> Option<(Vec<i64>, f64)> {
        let n = self.n;
        if hist.len() < n || max == 0 { return None; }
        self.update(hist);
        let at = |i: usize| if i < hist.len() { hist[i] } else { next };
        let len = hist.len() + 1;
        let tail: Vec<i64> = (len - n..len).map(at).collect();
        let &e = self.map.get(&Self::key(&tail))?;
        if (0..n).any(|i| at(e + 1 - n + i) != tail[i]) { return None; }   // hash collision
        let v: Vec<i64> = (e + 1..(e + 1 + max).min(len)).map(at).collect();
        (!v.is_empty()).then(|| (v, self.conf()))
    }

    /// Running per-token acceptance of lookup drafts (recent rounds weigh more).
    pub fn conf(&self) -> f64 { (self.hits / self.tries).clamp(0.3, 0.98) }

    /// After a round, with the round's proposal (used or not) and the tokens that actually followed (accepted drafts and the
    /// bonus): the proposal's matching prefix counts as accepted, its first mismatch as a rejection.
    pub fn observe_shadow(&mut self, proposal: &[i64], truth: &[i64]) {
        let m = proposal.iter().zip(truth).take_while(|(a, b)| a == b).count();
        let n = proposal.len().min(truth.len());
        if n == 0 { return; }
        self.hits = self.hits * 0.8 + m as f64;
        self.tries = self.tries * 0.8 + (m + (m < n) as usize) as f64;
    }

    /// Whether a proposal of this sequence should be used: its running per-token acceptance is at least QWEN_LOOKUP_MIN
    /// (default 0.6) and rounds with lookup drafts have yielded at least 1 / 1.1 as many tokens as rounds with MTP drafts (every
    /// 16th round the other source is tried, so both estimates stay current). 2026-10-07: converting JSON to YAML, lookup
    /// proposals agree with the first MTP draft and then diverge (5.0 tokens per round vs 7.5 with the MTP chain).
    pub fn usable(&self) -> bool {
        static MIN: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
        if self.conf() < *MIN.get_or_init(|| std::env::var("QWEN_LOOKUP_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6)) { return false; }
        // (a lookup round runs no MTP draft steps: about 10% cheaper at one sequence)
        let better = match (self.per_l, self.per_m) { (Some(l), Some(m)) => 1.1 * l >= m, (None, _) => true, (Some(_), None) => true };
        if self.rounds % 16 == 15 { !better } else { better }
    }

    /// After a round: whether its drafts came from the lookup, and the tokens it produced (accepted drafts + 1).
    pub fn record(&mut self, used: bool, tokens: usize) {
        let ema = |v: &mut Option<f64>| *v = Some(v.map_or(tokens as f64, |x| 0.7 * x + 0.3 * tokens as f64));
        if used { ema(&mut self.per_l) } else { ema(&mut self.per_m) }
        self.rounds += 1;
    }
}
