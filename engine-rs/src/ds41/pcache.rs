//! Persistent NVMe prefix cache of DeepSeek-V4.1-Flash serving (DS41_PCACHE=1; docs/07 D5) on the engine's shared
//! cache core (crate::pcache_core, also under GLM's crate::pcache): at the end of every prompt of at least
//! DS41_PCACHE_MIN (1024) tokens each rank saves the sequence state; a later prompt extending a cached prefix restores
//! it instead of prefilling it again (multi-turn chats across store evictions and restarts).
//!
//! What a checkpoint holds (Model::pcache_rows / pcache_fixed / pcache_install):
//! - row arrays `ckv.<src>` / `ikv.<src>` per KV source (compressed KV and index keys, one row per `ratio` tokens,
//!   complete groups only): written in segments of DS41_PCACHE_SEG (4096) tokens shared along a store's lineage (a
//!   multi-turn chat only writes the new segments), straight from / back into the store's KV pool rows;
//! - fixed regions: window rings (`win`, stacked), the compressor's open groups (`csk.<src>` / `css.<src>`), the DSpark
//!   windows (`mtp_win`), the CED encoder tail (`ced_h` / `ced_pm`), the n-gram history (`hist`, host);
//! - header fields pos, dec_pos, ced_end.
//!
//! The state is replicated on every rank (the attention KV is not split by TP), so every rank saves the same entries at
//! the same points (the core's index is a replicated state machine; the startup agreement runs over the serve loop's
//! control connections) and restores the same entry; the ranks agree on a restore's success with one allreduce.
//! Identity: engine binary SHA-256, checkpoint and engram table paths, TP world, segment size and the numerics-relevant
//! DS41_* settings; directory `DS41_PCACHE_DIR` (default ./ds41-prefix-cache). LRU within DS41_PCACHE_GB (100).
use crate::pcache_core::{self as core, Config, Link};
use std::path::Path;

fn env_u64(k: &str, d: u64) -> u64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }
pub fn requested() -> bool { std::env::var("DS41_PCACHE").as_deref() == Ok("1") }
pub fn min_tokens() -> usize { env_u64("DS41_PCACHE_MIN", 1024) as usize }
/// Regions restored into CPU tensors.
pub const HOST: [&str; 1] = ["hist"];

fn identity(binary: &str, ckpt: &Path, hash: &Path, world: usize, seg: i64) -> String {
    // scheduling, memory, logging and transport settings do not change the saved values
    let skip = ["DS41_SERVE_", "DS41_PCACHE", "DS41_PREFILL_STATS", "DS41_STEP_STATS", "DS41_WARM_GRAPHS", "DS41_EOS", "DS41_MEM_", "DS41_KV_",
        "DS41_SPEC", "DS41_GRAPH_MAX", "DS41_ENGRAM_STATS", "DS41_ENGRAM_THREADS", "DS41_ENGRAM_IO", "DS41_DEBUG", "DS41_PROFILE", "DS41_HTTP", "DS41_MASTER",
        "DS41_DRAFT_HEAD_FP8", "DS41_INDEX_MB", "DS41_BENCH", "DS41_ROOT", "DS41_ENV", "DS41_PY", "DS41_ENGINE_BIN", "DS41_EXTRA_ENV", "DS41_MODEL_NAME", "DS41_VISION",
        // start-ds41.sh's own inputs: only rank 0 inherits them from the caller's shell (the others are started over ssh),
        // and what they select is in the identity already (shard root, engram hash) or passed on as SPARK_ABLATE_*
        "DS41_WEIGHTS", "DS41_CKPT", "DS41_SHARDS", "DS41_ENGRAM_HASH", "DS41_ABLATE_"];
    // the clamp ablation changes the saved values: SPARK_ABLATE_* is part of the identity
    let mut env: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| (k.starts_with("DS41_") && !skip.iter().any(|s| k.starts_with(s))) || k.starts_with("SPARK_ABLATE_")).collect();
    env.sort();
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).display().to_string();
    // a rank's shard directory (<root>/rank<r>): the root, the same on every rank
    let ckpt = match ckpt.file_name().and_then(|n| n.to_str()) { Some(n) if n.starts_with("rank") => ckpt.parent().unwrap_or(ckpt), _ => ckpt };
    format!("format={};model=ds41;binary={binary};ckpt={};engram={};world={world};seg={seg};env={env:?}", core::FORMAT, canon(ckpt), canon(hash))
}

/// Startup (every rank, after the control connections, before the warm-up so its pinned buffers count before the memory
/// budget): None when off or when the ranks' identities differ.
pub fn open(rank: usize, world: usize, ckpt: &Path, hash: &Path, link: Link) -> Option<core::Cache> {
    let seg = env_u64("DS41_PCACHE_SEG", 4096) as i64;
    assert!(seg >= 64 && seg % 64 == 0, "DS41_PCACHE_SEG must be a multiple of 64");
    let cfg = Config {
        root: std::env::var("DS41_PCACHE_DIR").unwrap_or_else(|_| "ds41-prefix-cache".into()).into(), seg,
        cap: env_u64("DS41_PCACHE_GB", 100) << 30, min: min_tokens(), free_margin: env_u64("DS41_PCACHE_FREE_GB", 20) << 30,
        // (unified memory: the staging ring is pinned host memory taken from the GPU's budget)
        stage_mb: env_u64("DS41_PCACHE_STAGE_MB", 128), readers: env_u64("DS41_PCACHE_READERS", 4) as usize,
        log: std::env::var("DS41_PCACHE_LOG").as_deref() == Ok("1"), tag: "[ds41-pcache]", test_delay_ms: None,
    };
    let binary = core::binary_sha256();
    let id = identity(&binary, ckpt, hash, world, seg);
    core::open(rank, cfg, id, binary, link)
}
