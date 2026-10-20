//! Model-independent multimodal prompt plumbing (the engine side of serve/mm.py): canvases, salted placeholders,
//! encoded rows per prefill chunk. A model supplies its geometry (how many prompt positions one segment of a canvas
//! fills) and its encoder (a canvas segment -> the rows of those positions); the rest is shared.
//!
//! - The front end resizes / pads every image (video: frames stacked) to the model's canvas (uint8 HWC) in /dev/shm and
//!   expands each placeholder into *salted* ids in [MM_BASE, MM_TOP): never a vocabulary id and distinct per media
//!   content, so prefix reuse (plain id `starts_with`) separates different media while the same media hits again.
//! - Rank 0 encodes the segments a prefill chunk touches straight from the canvas file and hands the chunk's rows to
//!   the model, which writes them into the embedding before the vocabulary-parallel embed sum (the other ranks hold
//!   zeros there: a salted id is outside every rank's vocabulary slice).
//!
//! Users: DeepSeek-V4.1 (crate::ds41::vision). GLM (crate::vision) and Qwen (crate::qwen::vision) still carry their
//! own copies of this plumbing (same request format); moving them here needs their regression runs (bench/mm).
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};

/// Salted placeholder ids live in [MM_BASE, MM_TOP) (serve/mm.py `salted`).
pub const MM_BASE: i64 = 1 << 20;
pub const MM_TOP: i64 = 1 << 24;
pub fn is_salted(t: i64) -> bool { (MM_BASE..MM_TOP).contains(&t) }

/// One media item of a request (rank 0): its canvas file and the placeholder runs it fills.
#[derive(Clone, Debug)]
pub struct MmItem {
    pub path: PathBuf, pub frames: i64, pub height: i64, pub width: i64,
    /// (prompt position, placeholder count, first frame): an image has one segment (frame 0).
    pub segments: Vec<(usize, usize, i64)>,
}

/// Parse the request's "mm" array: [{"path","frames","height","width","segments":[[pos,len,frame],...]}].
/// `seg_len(item)`: the placeholder count one segment of this canvas fills, or an error when the model does not
/// accept the canvas.
pub fn parse_items(v: &serde_json::Value, seg_len: impl Fn(&MmItem) -> Result<usize, String>) -> Result<Vec<MmItem>, String> {
    let Some(a) = v.as_array() else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for it in a {
        let path = PathBuf::from(it["path"].as_str().ok_or("mm item without path")?);
        let get = |k: &str| it[k].as_i64().ok_or(format!("mm item without {k}"));
        let mut item = MmItem { path, frames: get("frames")?, height: get("height")?, width: get("width")?, segments: Vec::new() };
        if item.frames < 1 || item.height < 1 || item.width < 1 { return Err(format!("mm item has a bad canvas {}x{}x{}", item.frames, item.height, item.width)); }
        let len = seg_len(&item)?;
        let meta = std::fs::metadata(&item.path).map_err(|e| format!("mm canvas {}: {e}", item.path.display()))?;
        let want = item.frames * item.height * item.width * 3;
        if meta.len() as i64 != want { return Err(format!("mm canvas {} has {} bytes, expected {want}", item.path.display(), meta.len())); }
        for s in it["segments"].as_array().ok_or("mm item without segments")? {
            let s = s.as_array().ok_or("bad segment")?;
            let f = |i: usize| s.get(i).and_then(|x| x.as_i64()).ok_or("bad segment");
            let (pos, n, frame) = (f(0)?, f(1)?, f(2)?);
            if pos < 0 || n as usize != len || frame < 0 || frame >= item.frames { return Err(format!("mm segment ({pos},{n},{frame}) does not fit the canvas")); }
            item.segments.push((pos as usize, n as usize, frame));
        }
        out.push(item);
    }
    Ok(out)
}

/// Check the salted placeholders of a prompt against its items: every id is a vocabulary id or a salted placeholder,
/// and the placeholder positions are exactly the items' segments.
pub fn validate(ids: &[i64], vocab: i64, items: &[MmItem]) -> Result<(), String> {
    let mut mark = vec![false; ids.len()];
    for it in items {
        for &(pos, len, _) in &it.segments {
            if pos + len > ids.len() { return Err("mm segment past the prompt end".into()); }
            for m in &mut mark[pos..pos + len] { if *m { return Err("overlapping mm segments".into()); } *m = true; }
        }
    }
    for (j, &t) in ids.iter().enumerate() {
        let salted = is_salted(t);
        if salted != mark[j] {
            return Err(if salted { format!("placeholder id at {j} has no mm segment") } else { format!("mm segment covers non-placeholder id {t} at {j}") });
        }
        if !salted && !(0..vocab).contains(&t) { return Err(format!("token id {t} at {j} is outside the vocabulary")); }
    }
    Ok(())
}

/// Canvas frames [k * per, H, W, 3] uint8 on `dev` for the segments starting at frames `firsts` (`per` frames each).
pub fn read_frames(it: &MmItem, firsts: &[i64], per: i64, dev: Device) -> Tensor {
    use std::io::{Read, Seek, SeekFrom};
    let fb = (it.height * it.width * 3) as usize;
    let mut f = std::fs::File::open(&it.path).unwrap_or_else(|e| panic!("mm canvas {}: {e}", it.path.display()));
    let per_b = fb * per as usize;
    let mut buf = vec![0u8; per_b * firsts.len()];
    for (k, &first) in firsts.iter().enumerate() {
        f.seek(SeekFrom::Start(first as u64 * fb as u64)).unwrap();
        f.read_exact(&mut buf[k * per_b..(k + 1) * per_b]).unwrap_or_else(|e| panic!("mm canvas {}: {e}", it.path.display()));
    }
    Tensor::from_slice(&buf).view([per * firsts.len() as i64, it.height, it.width, 3]).to_device(dev)
}

/// Per-sequence encoder cache (rank 0): encoded segments waiting for the chunks that consume them.
#[derive(Default)]
pub struct MmState { pub items: Vec<MmItem>, done: std::collections::HashMap<(usize, usize), Tensor>, pub encode_ms: f64, pub encoded_tokens: usize }

impl MmState {
    pub fn new(items: Vec<MmItem>) -> Self { MmState { items, ..Default::default() } }

    /// The rows of the placeholders in prompt positions [begin, end): (their offsets in the chunk, rows [k, dim] FP32
    /// in position order), None when the chunk has none. Segments are encoded on first use, `encode(item, segment
    /// indices)` -> one [len, dim] tensor per segment; segments ending at or before `end` are released.
    pub fn chunk_rows(&mut self, begin: usize, end: usize, mut encode: impl FnMut(&MmItem, &[usize]) -> Vec<Tensor>) -> Option<(Vec<i64>, Tensor)> {
        let t0 = std::time::Instant::now();
        let mut pieces: Vec<(usize, Tensor)> = Vec::new();
        for (ii, it) in self.items.iter().enumerate() {
            let touches = |si: usize| { let (p, l, _) = it.segments[si]; p < end && p + l > begin };
            let need: Vec<usize> = (0..it.segments.len()).filter(|&si| touches(si) && !self.done.contains_key(&(ii, si))).collect();
            if !need.is_empty() {
                let ys = encode(it, &need);
                assert_eq!(ys.len(), need.len(), "mm encoder: one tensor per segment");
                for (&si, y) in need.iter().zip(ys) {
                    assert_eq!(y.size()[0] as usize, it.segments[si].1, "mm encoder: rows of a segment");
                    self.encoded_tokens += it.segments[si].1;
                    self.done.insert((ii, si), y);
                }
            }
            for si in (0..it.segments.len()).filter(|&si| touches(si)) {
                let (p, l, _) = it.segments[si];
                let (lo, hi) = (p.max(begin), (p + l).min(end));
                pieces.push((lo, self.done[&(ii, si)].narrow(0, (lo - p) as i64, (hi - lo) as i64)));
            }
        }
        let items = &self.items;
        self.done.retain(|&(ii, si), _| { let (p, l, _) = items[ii].segments[si]; p + l > end });
        if pieces.is_empty() { return None; }
        pieces.sort_by_key(|p| p.0);
        let mut offs = Vec::new();
        for (lo, y) in &pieces { offs.extend((0..y.size()[0]).map(|k| (lo - begin) as i64 + k)); }
        let rows = Tensor::cat(&pieces.into_iter().map(|p| p.1).collect::<Vec<_>>(), 0).to_kind(Kind::Float);
        self.encode_ms += t0.elapsed().as_secs_f64() * 1e3;
        Some((offs, rows))
    }
}
