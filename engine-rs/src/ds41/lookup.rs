//! Prompt Lookup / N-gram draft extension for DS41 speculative decoding (DS41_PROMPT_LOOKUP, default on).
//!
//! Deterministic across all TP ranks: reads only `store.history`, the pending token `tok`, and DSpark's `(drafts, confs)`,
//! which are identical on every rank under `DS41_CTRL_LOCAL`. Never changes the target verification rule, so emitted tokens
//! remain 100% L0 bit-for-bit identical to plain decoding.

pub fn prompt_lookup_on() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("DS41_PROMPT_LOOKUP").map_or(true, |v| v != "0"))
}

fn lookup_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("DS41_LOOKUP_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(12))
}

fn lookup_window() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("DS41_LOOKUP_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(16384))
}

fn override_p() -> f32 {
    static P: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("DS41_LOOKUP_OVERRIDE_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.35))
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}

/// Find the longest suffix of `query` (length in `min_n..=max_n`) that occurs in `haystack` with at least one token
/// following the match. Ties are broken by the most recent occurrence in `haystack`.
pub fn find_ngram_continuation(
    haystack: &[i64],
    query: &[i64],
    min_n: usize,
    max_n: usize,
    max_ext: usize,
    window: usize,
) -> Option<(usize, Vec<i64>)> {
    let qlen = query.len();
    let hlen = haystack.len();
    if max_ext == 0 || min_n == 0 || qlen < min_n || hlen <= min_n {
        return None;
    }
    let max_n = max_n.min(qlen).min(hlen - 1);
    let last_tok = query[qlen - 1];
    let lo = hlen.saturating_sub(window).max(min_n);
    let mut best_n = 0usize;
    let mut best_end = 0usize;
    // `end` is the index of the first continuation token (`haystack[end - n .. end]` matched `query[qlen - n .. qlen]`).
    // Scanning backward breaks ties in favor of the most recent match.
    for end in (lo..hlen).rev() {
        if haystack[end - 1] != last_tok {
            continue;
        }
        let lim = max_n.min(end);
        let mut n = 1usize;
        while n < lim && haystack[end - 1 - n] == query[qlen - 1 - n] {
            n += 1;
        }
        if n >= min_n && n > best_n {
            best_n = n;
            best_end = end;
            if best_n == max_n {
                break;
            }
        }
    }
    if best_n == 0 {
        return None;
    }
    let take = max_ext.min(hlen - best_end);
    Some((best_n, haystack[best_end..best_end + take].to_vec()))
}

fn build_query(history: &[i64], tok: i64, prefix: &[i64], cap: usize) -> Vec<i64> {
    let mut q = Vec::with_capacity(cap);
    let need = cap.saturating_sub(1 + prefix.len());
    let h0 = history.len().saturating_sub(need);
    q.extend_from_slice(&history[h0..]);
    q.push(tok);
    q.extend_from_slice(prefix);
    if q.len() > cap {
        q.drain(..q.len() - cap);
    }
    q
}

/// Extend (or override low-confidence tail of) DSpark's drafts using n-gram matches in `history`.
/// `max_cap`: caller cap (e.g., to stay within pre-warmed row graph totals at higher batch sizes).
#[allow(dead_code)]
pub fn apply_lookup(
    history: &[i64],
    tok: i64,
    drafts: &mut Vec<i64>,
    confs: &mut Vec<f32>,
    room: usize,
    temp: f32,
) {
    apply_lookup_capped(history, tok, drafts, confs, room, lookup_max(), temp);
}

pub fn apply_lookup_capped(
    history: &[i64],
    tok: i64,
    drafts: &mut Vec<i64>,
    confs: &mut Vec<f32>,
    room: usize,
    cap: usize,
    temp: f32,
) {
    if !prompt_lookup_on() || room == 0 || history.len() < 4 {
        return;
    }
    let max_total = lookup_max().min(cap).min(room);
    if max_total == 0 {
        return;
    }
    let win = lookup_window();
    // Mode 2 (greedy only): if DSpark loses confidence early, override from that point with a strong >=4-gram match.
    if temp <= 0.0 {
        let thr = override_p();
        if let Some(k) = confs.iter().position(|&c| sigmoid(c) < thr) {
            if k < max_total {
                let q = build_query(history, tok, &drafts[..k], 8);
                if let Some((n, cont)) = find_ngram_continuation(history, &q, 4, 8, max_total - k, win) {
                    if !cont.is_empty() && cont[0] != drafts[k] {
                        drafts.truncate(k);
                        confs.truncate(k);
                        let base_p = if n >= 5 { 0.88f32 } else { 0.82f32 };
                        for (i, &t) in cont.iter().enumerate() {
                            drafts.push(t);
                            confs.push(logit((base_p * 0.96f32.powi(i as i32)).clamp(0.50, 0.90)));
                        }
                        return;
                    }
                }
            }
        }
    }
    // Mode 1: extend past DSpark's drafts when the chain is still confident and `room` allows more drafts.
    if max_total <= drafts.len() {
        return;
    }
    if confs.last().is_some_and(|&c| sigmoid(c) < 0.40) {
        return;
    }
    let q = build_query(history, tok, drafts, 8);
    if let Some((n, cont)) = find_ngram_continuation(history, &q, 3, 8, max_total - drafts.len(), win) {
        let mut base_p = if n >= 5 { 0.90f32 } else if n == 4 { 0.85f32 } else { 0.78f32 };
        if temp > 0.0 {
            base_p *= 0.75;
        }
        for (i, &t) in cont.iter().enumerate() {
            drafts.push(t);
            confs.push(logit((base_p * 0.97f32.powi(i as i32)).clamp(0.45, 0.92)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_and_most_recent_match() {
        // Pattern [10, 20, 30] appears twice: once with continuation [99], later with [40, 50, 60].
        let hist = vec![1, 10, 20, 30, 99, 2, 10, 20, 30, 40, 50, 60, 7, 8];
        let (n, cont) = find_ngram_continuation(&hist, &[10, 20, 30], 3, 8, 5, 1024).unwrap();
        assert_eq!(n, 3);
        assert_eq!(cont, vec![40, 50, 60, 7, 8]);
        // Longer match wins over more recent shorter match.
        let hist2 = vec![5, 10, 20, 30, 40, 111, 222, 9, 20, 30, 40, 333];
        let (n2, cont2) = find_ngram_continuation(&hist2, &[10, 20, 30, 40], 3, 8, 2, 1024).unwrap();
        assert_eq!(n2, 4);
        assert_eq!(cont2, vec![111, 222]);
    }

    #[test]
    fn tail_extension_and_room_cap() {
        let hist = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut drafts = vec![4, 5, 6, 7, 8];
        let mut confs = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        apply_lookup(&hist, 3, &mut drafts, &mut confs, 8, 0.0);
        assert_eq!(drafts, vec![4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(confs.len(), 8);
    }

    #[test]
    fn low_confidence_override_requires_4gram() {
        let hist = vec![100, 101, 102, 103, 104, 105, 106, 200, 201, 100, 101, 102];
        let mut drafts = vec![999, 998, 997, 996, 995];
        let mut confs = vec![-2.0, 2.0, 2.0, 2.0, 2.0]; // sigmoid(-2.0) ~= 0.12 < 0.35
        // Query ends with [100, 101, 102, 103] -> 4-gram match at index 0..4 -> overrides with [104, 105, 106, 200, 201, 100]
        apply_lookup(&hist, 103, &mut drafts, &mut confs, 5, 0.0);
        assert_eq!(drafts, vec![104, 105, 106, 200, 201]);
        assert_eq!(confs.len(), 5);
    }
}
