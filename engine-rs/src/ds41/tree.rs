//! DSpark confidence-guided tree verification for DS41 speculative decoding (DS41_SPEC_TREE, default on).
//!
//! Each sequence's verify input is a small rooted tree (`VerifyTree`) with node 0 as the pending token at `seq.pos`,
//! stored in topological order (`parents[0] = -1`, `parents[u] < u`). When a sequence has no alternate branches,
//! `is_chain()` is true and verification takes the existing linear path with zero overhead.

/// DS41_SPEC_TREE (default on; 0 off): allow low-confidence draft steps to add top-2 alternate branches
/// (plus alt0's Markov continuation chain) into a verified tree when the cost model predicts higher tok/ms.
pub fn spec_tree_on() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("DS41_SPEC_TREE").map_or(true, |v| v != "0"))
}

/// Maximum batch size for tree branching (DS41_TREE_MAX_BATCH, default 1: c1 has the most GPU headroom for extra
/// branch rows; at c2+ two sequences' main chains already saturate MoE expert reads).
pub fn tree_max_batch() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("DS41_TREE_MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 2))
}

/// Conservative multiplier on the estimated top-2 branch acceptance probability (DS41_TREE_SCALE, default 0.75).
fn tree_scale() -> f64 {
    static S: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("DS41_TREE_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.75))
}

/// Maximum total nodes (including root) in one sequence's verify tree (DS41_TREE_MAX_NODES, default 12).
fn tree_max_nodes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("DS41_TREE_MAX_NODES").ok().and_then(|v| v.parse().ok()).unwrap_or(12).clamp(2, 16))
}

/// DS41_ALT1_CHAIN (default off; 1 on): also capture alt_toks[1]'s Markov continuation chain in the same C12 weight pass.
pub fn alt1_chain_on() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("DS41_ALT1_CHAIN").is_ok_and(|v| v == "1"))
}

/// Top-2 draft branch candidates captured for one sequence alongside DSpark's primary chain.
#[derive(Clone, Debug)]
pub struct DraftBranch {
    /// 2nd-best token at each primary step `j = 0..blk` (parent is primary step `j - 1`, or root for `j == 0`).
    pub alt_toks: Vec<i64>,
    /// Score margin `val1 - val2` at each primary step `j = 0..blk` (`>= 0`; large/inf means negligible alt).
    pub alt_margins: Vec<f32>,
    /// Markov continuation tokens `[alt0_1, alt0_2, ...]` following `alt_toks[0]` at steps `1..blk`.
    pub alt0_chain: Vec<i64>,
    /// Confidence logits for `alt0_chain`.
    pub alt0_confs: Vec<f32>,
    /// Markov continuation tokens `[alt1_2, alt1_3, ...]` following `alt_toks[1]` at steps `2..blk`.
    pub alt1_chain: Vec<i64>,
    /// Confidence logits for `alt1_chain`.
    pub alt1_confs: Vec<f32>,
    /// Optional parallel branch injected from n-gram prompt/global lookup: `(depth, tokens, conf_logits)`.
    pub lookup_branch: Option<(usize, Vec<i64>, Vec<f32>)>,
}

/// Verification tree for one sequence: node 0 is the pending token (`depth = 0`, `parent = -1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyTree {
    pub ids: Vec<i64>,
    pub parents: Vec<i32>,
    pub depths: Vec<i32>,
}

impl VerifyTree {
    pub fn chain(ids: Vec<i64>) -> Self {
        let n = ids.len();
        assert!(n >= 1, "empty VerifyTree");
        let parents: Vec<i32> = (0..n).map(|u| u as i32 - 1).collect();
        let depths: Vec<i32> = (0..n).map(|u| u as i32).collect();
        VerifyTree { ids, parents, depths }
    }

    pub fn len(&self) -> usize { self.ids.len() }

    pub fn is_chain(&self) -> bool {
        self.parents.iter().enumerate().all(|(u, &p)| p == u as i32 - 1)
    }

    pub fn max_depth(&self) -> usize {
        self.depths.iter().copied().max().unwrap_or(0) as usize
    }

    /// Ancestor node indices from root (depth 0) to `u` (depth `depths[u]`), inclusive.
    pub fn ancestors(&self, u: usize) -> Vec<usize> {
        let mut path = Vec::with_capacity(self.depths[u] as usize + 1);
        let mut cur = u as i32;
        while cur >= 0 {
            path.push(cur as usize);
            cur = self.parents[cur as usize];
        }
        path.reverse();
        path
    }

    /// Per-node ancestor table `[u][d]` = chunk row index of `u`'s ancestor at depth `d` (`0 <= d <= depths[u]`).
    pub fn anc_table(&self) -> Vec<[u8; 16]> {
        let mut tab = vec![[0u8; 16]; self.len()];
        for u in 0..self.len() {
            let anc = self.ancestors(u);
            assert!(anc.len() <= 16, "VerifyTree depth {} exceeds 15", anc.len() - 1);
            for (d, &a) in anc.iter().enumerate() {
                tab[u][d] = a as u8;
            }
        }
        tab
    }

    /// Primary chain membership: root 0 and the first child of each primary-chain node (unique depth per primary node).
    pub fn main_chain_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; self.len()];
        mask[0] = true;
        let mut cur = 0i32;
        for u in 1..self.len() {
            if self.parents[u] == cur {
                mask[u] = true;
                cur = u as i32;
            }
        }
        mask
    }

    /// Length of the primary chain (`main_chain_mask().iter().filter(|&&b| b).count()`).
    pub fn main_chain_len(&self) -> usize {
        self.main_chain_mask().iter().filter(|&&b| b).count()
    }
}

#[derive(Clone)]
struct Cand {
    parent_key: usize, // index into per-seq candidate list (0 = root)
    tok: i64,
    depth: usize,
    prob: f64,
    is_branch: bool,
}

/// Cost-model greedy tree builder across a batch of sequences (generalizes linear `choose_drafts`).
///
/// Guarantees:
/// - When `spec_tree_on()` is false or all `branches` are `None` or no branch beats the cost model, returns pure
///   linear chains identical to `choose_drafts`.
/// - Total rows across the batch never exceed `max_rows`.
/// - If any sequence gets a branch, total batch rows stay `<= max_rows.min(16)` (so `MidGraph` runs all 32
///   non-source layers inside captured CUDA graphs).
#[allow(clippy::too_many_arguments)]
pub fn build_trees(
    toks: &[i64],
    drafts: &[Vec<i64>],
    confs: &[Vec<f32>],
    branches: &[Option<DraftBranch>],
    rooms: &[usize],
    temps: &[f32],
    max_rows: usize,
    cost_cfg: (f64, f64, f64),
    calib: impl Fn(f32) -> (f64, f64),
) -> Vec<VerifyTree> {
    let b = toks.len();
    let (base, seq_c, row_c) = cost_cfg;
    let tree_ok = spec_tree_on() && b <= tree_max_batch();
    let scale = tree_scale();
    let max_nodes = tree_max_nodes();

    // Per-sequence pool of potential nodes; entry 0 is always the already-selected root.
    let mut pools: Vec<Vec<Cand>> = Vec::with_capacity(b);
    let mut chosen_node_id: Vec<Vec<Option<usize>>> = Vec::with_capacity(b);
    let mut trees: Vec<VerifyTree> = Vec::with_capacity(b);

    for s in 0..b {
        let (a, bb) = calib(temps[s]);
        let room = rooms[s];
        let m = drafts[s].len().min(room);
        let mut pool = vec![Cand { parent_key: 0, tok: toks[s], depth: 0, prob: 1.0, is_branch: false }];
        let mut main_keys = Vec::with_capacity(m);
        let mut p_main = Vec::with_capacity(m);
        let mut acc = 1.0f64;
        for j in 0..m {
            let p_step = 1.0 / (1.0 + (-(a * confs[s][j] as f64 + bb)).exp());
            acc *= p_step;
            let parent_key = if j == 0 { 0 } else { main_keys[j - 1] };
            let key = pool.len();
            pool.push(Cand { parent_key, tok: drafts[s][j], depth: j + 1, prob: acc, is_branch: false });
            main_keys.push(key);
            p_main.push((p_step, acc));
        }

        if tree_ok {
            if let Some(br) = &branches[s] {
                let nb = br.alt_toks.len().min(m);
                let tscale = if temps[s] > 0.0 { 0.5 * scale } else { scale };
                for j in 0..nb {
                    let alt = br.alt_toks[j];
                    let margin = br.alt_margins[j];
                    if alt < 0 || alt == drafts[s][j] || !margin.is_finite() || margin < 0.0 || margin > 4.0 {
                        continue;
                    }
                    let p_parent = if j == 0 { 1.0 } else { p_main[j - 1].1 };
                    let p1 = p_main[j].0;
                    let denom = if temps[s] > 0.0 { temps[s].max(0.25) as f64 } else { 1.0 };
                    let ratio = (-(margin as f64) / denom).exp().min(1.0);
                    let p_alt = p_parent * (1.0 - p1).min(p1 * ratio) * tscale;
                    if p_alt <= 0.01 {
                        continue;
                    }
                    let parent_key = if j == 0 { 0 } else { main_keys[j - 1] };
                    let alt_key = pool.len();
                    pool.push(Cand { parent_key, tok: alt, depth: j + 1, prob: p_alt, is_branch: true });

                    // For j == 0 or j == 1, attach the pre-computed Markov continuation chain (zero extra draft weight reads).
                    let cont: Option<(&[i64], &[f32], usize)> = if j == 0 && !br.alt0_chain.is_empty() {
                        Some((&br.alt0_chain, &br.alt0_confs, 2))
                    } else if j == 1 && !br.alt1_chain.is_empty() {
                        Some((&br.alt1_chain, &br.alt1_confs, 3))
                    } else {
                        None
                    };
                    if let Some((ctoks, cconfs, base_d)) = cont {
                        let mut c_acc = p_alt;
                        let mut pkey = alt_key;
                        for (ci, (&ctok, &cconf)) in ctoks.iter().zip(cconfs).enumerate() {
                            let d = ci + base_d;
                            if d > room || ctok < 0 {
                                break;
                            }
                            let ps = 1.0 / (1.0 + (-(a * cconf as f64 + bb)).exp());
                            c_acc *= ps;
                            if c_acc <= 0.01 {
                                break;
                            }
                            let ck = pool.len();
                            pool.push(Cand { parent_key: pkey, tok: ctok, depth: d, prob: c_acc, is_branch: true });
                            pkey = ck;
                        }
                    }
                }
                // Optional lookup branch at depth 0 (when lookup disagreed with DSpark's main chain at moderate confidence).
                if let Some((0, ltoks, lconfs)) = &br.lookup_branch {
                    if let Some(&first_t) = ltoks.first() {
                        let dup = pool.iter().any(|c| c.parent_key == 0 && c.tok == first_t);
                        if !dup {
                            let mut c_acc = 1.0f64;
                            let mut pkey = 0usize;
                            for (ci, (&ctok, &cconf)) in ltoks.iter().zip(lconfs).enumerate() {
                                let d = ci + 1;
                                if d > room || ctok < 0 {
                                    break;
                                }
                                let ps = 1.0 / (1.0 + (-(a * cconf as f64 + bb)).exp());
                                c_acc *= ps;
                                if c_acc <= 0.01 {
                                    break;
                                }
                                let ck = pool.len();
                                pool.push(Cand { parent_key: pkey, tok: ctok, depth: d, prob: c_acc, is_branch: true });
                                pkey = ck;
                            }
                        }
                    }
                }
            }
        }

        let mut cid = vec![None; pool.len()];
        cid[0] = Some(0);
        pools.push(pool);
        chosen_node_id.push(cid);
        trees.push(VerifyTree { ids: vec![toks[s]], parents: vec![-1], depths: vec![0] });
    }

    let mut total_rows = b;
    let mut has_any_branch = false;
    let (mut gain, mut cost) = (b as f64, base + (seq_c + row_c) * b as f64);

    loop {
        let row_lim = if has_any_branch { max_rows.min(16) } else { max_rows };
        if total_rows >= row_lim {
            break;
        }
        let mut best: Option<(usize, usize, f64)> = None;
        for s in 0..b {
            for k in 1..pools[s].len() {
                if chosen_node_id[s][k].is_some() {
                    continue;
                }
                let c = &pools[s][k];
                if chosen_node_id[s][c.parent_key].is_none() {
                    continue;
                }
                // Respect MidGraph <= 16 row cap and per-sequence tree cap for branches.
                if c.is_branch && (total_rows + 1 > max_rows.min(16) || trees[s].len() >= max_nodes) {
                    continue;
                }
                if !c.is_branch && has_any_branch && trees[s].len() >= max_nodes {
                    continue;
                }
                if best.is_none_or(|(_, _, bp)| c.prob.total_cmp(&bp).is_gt()) {
                    best = Some((s, k, c.prob));
                }
            }
        }
        let Some((s, k, g)) = best else { break };
        if (gain + g) / (cost + row_c) <= gain / cost {
            break;
        }
        let c = &pools[s][k];
        let pid = chosen_node_id[s][c.parent_key].unwrap() as i32;
        let nid = trees[s].ids.len();
        trees[s].ids.push(c.tok);
        trees[s].parents.push(pid);
        trees[s].depths.push(c.depth as i32);
        chosen_node_id[s][k] = Some(nid);
        if c.is_branch {
            has_any_branch = true;
        }
        total_rows += 1;
        gain += g;
        cost += row_c;
    }

    // Reorder nodes within each tree so primary chain precedes branches while preserving topological order
    // (`parents[u] < u`), ensuring `main_chain_mask()` marks the exact primary prefix.
    for t in &mut trees {
        if t.is_chain() {
            continue;
        }
        // Build order: walk primary chain (first child at each depth from root), then all remaining nodes in old order.
        let mut order = Vec::with_capacity(t.len());
        let mut in_main = vec![false; t.len()];
        let mut cur = 0i32;
        order.push(0usize);
        in_main[0] = true;
        for u in 1..t.len() {
            if t.parents[u] == cur {
                order.push(u);
                in_main[u] = true;
                cur = u as i32;
            }
        }
        for u in 1..t.len() {
            if !in_main[u] {
                order.push(u);
            }
        }
        let mut inv = vec![0usize; t.len()];
        for (new_u, &old_u) in order.iter().enumerate() {
            inv[old_u] = new_u;
        }
        let new_ids: Vec<i64> = order.iter().map(|&u| t.ids[u]).collect();
        let new_parents: Vec<i32> = order.iter().map(|&u| if t.parents[u] < 0 { -1 } else { inv[t.parents[u] as usize] as i32 }).collect();
        let new_depths: Vec<i32> = order.iter().map(|&u| t.depths[u]).collect();
        t.ids = new_ids;
        t.parents = new_parents;
        t.depths = new_depths;
    }

    trees
}

/// Walk `tree` against target sampled tokens `ys` (`ys[u]` follows node `u`), stopping on mismatch or stop token.
/// Returns `(accepted_node_indices, emitted_tokens)`:
/// - `accepted_node_indices` starts with `0` and has length `k + 1` (`k` accepted draft tokens).
/// - `emitted_tokens` has length `k + 1` (`k` accepted drafts followed by the next pending token `ys[last]`).
pub fn accept_tree_path(tree: &VerifyTree, ys: &[i64], stops: &[i64]) -> (Vec<usize>, Vec<i64>) {
    let mut path = vec![0usize];
    let mut emit = Vec::new();
    let mut cur = 0usize;
    loop {
        let y = ys[cur];
        emit.push(y);
        if stops.contains(&y) {
            break;
        }
        let next = (cur + 1..tree.len()).find(|&c| tree.parents[c] == cur as i32 && tree.ids[c] == y);
        match next {
            Some(c) => {
                cur = c;
                path.push(cur);
            }
            None => break,
        }
    }
    (path, emit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_confidence_stays_pure_chain() {
        let br = DraftBranch {
            alt_toks: vec![90, 91, 92, 93, 94],
            alt_margins: vec![5.0, 5.0, 5.0, 5.0, 5.0],
            alt0_chain: vec![95, 96],
            alt0_confs: vec![2.0, 2.0],
            alt1_chain: vec![],
            alt1_confs: vec![],
            lookup_branch: None,
        };
        let trees = build_trees(
            &[10],
            &[vec![11, 12, 13, 14, 15]],
            &[vec![3.5, 3.5, 3.5, 3.5, 3.5]],
            &[Some(br)],
            &[10],
            &[0.0],
            32,
            (22.0, 0.5, 2.5),
            |_| (1.0, 0.0),
        );
        assert!(trees[0].is_chain());
        assert_eq!(trees[0].ids, vec![10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn uncertain_step_branches_and_accepts_alt_chain() {
        // Step 0 has conf=0.0 (p1=0.5) and tiny margin=0.1 -> alt0 + alt0_chain should be included.
        let br = DraftBranch {
            alt_toks: vec![20, 21, 22],
            alt_margins: vec![0.1, 3.0, 3.0],
            alt0_chain: vec![201, 202],
            alt0_confs: vec![3.0, 3.0],
            alt1_chain: vec![],
            alt1_confs: vec![],
            lookup_branch: None,
        };
        let trees = build_trees(
            &[10],
            &[vec![11, 12, 13]],
            &[vec![0.0, 2.0, 2.0]],
            &[Some(br)],
            &[10],
            &[0.0],
            32,
            (22.0, 0.5, 2.5),
            |_| (1.0, 0.0),
        );
        let t = &trees[0];
        assert!(!t.is_chain());
        // Topological order: parents[u] < u for all u > 0
        for u in 1..t.len() {
            assert!(t.parents[u] >= 0 && (t.parents[u] as usize) < u);
            assert_eq!(t.depths[u], t.depths[t.parents[u] as usize] + 1);
        }
        // Verify alt0 and alt0_chain are present
        let alt0 = t.ids.iter().position(|&id| id == 20).expect("alt0 present");
        let alt1 = t.ids.iter().position(|&id| id == 201).expect("alt0_1 present");
        assert_eq!(t.parents[alt0], 0);
        assert_eq!(t.parents[alt1], alt0 as i32);
        // Simulate target picking alt0 -> alt0_1 -> 999
        let mut ys = vec![777; t.len()];
        ys[0] = 20;
        ys[alt0] = 201;
        ys[alt1] = 999;
        let (path, emit) = accept_tree_path(t, &ys, &[1]);
        assert_eq!(path, vec![0, alt0, alt1]);
        assert_eq!(emit, vec![20, 201, 999]);
    }

    #[test]
    fn stop_token_cuts_tree_acceptance() {
        let t = VerifyTree {
            ids: vec![10, 11, 12, 20, 21],
            parents: vec![-1, 0, 1, 0, 3],
            depths: vec![0, 1, 2, 1, 2],
        };
        // Target emits 20 (branch) then 1 (EOS) at node 3
        let ys = vec![20, 0, 0, 1, 99];
        let (path, emit) = accept_tree_path(&t, &ys, &[1]);
        assert_eq!(path, vec![0, 3]);
        assert_eq!(emit, vec![20, 1]);
    }
}
