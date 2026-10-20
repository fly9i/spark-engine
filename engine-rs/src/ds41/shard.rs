//! `ds41-shard <ckpt> <out_dir> <rank> <world>`: writes this rank's share of the checkpoint as its own safetensors
//! files (one per layer + globals), the exact tensors `Model::load` takes from the full checkpoint (recorded while
//! running the same loader on the CPU), so loading a shard is one contiguous read per tensor instead of row-by-row
//! slices of 510 GB. The index marks the directory as a shard (`metadata.ds41_shard`) and names the source checkpoint
//! (engram tables are read from it). `inference/config.json` is copied alongside.
use super::cfg::{Cfg, Split};
use super::load::{fp4_table, Checkpoint};
use super::model::Model;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use tch::{Device, Kind};

pub fn shard(ckpt: &Path, out: &Path, rank: usize, world: usize) {
    let _g = tch::no_grad_guard();
    let c = Cfg::load(ckpt);
    let sp = Split::new(&c, rank, world);
    let ck = Checkpoint::open(ckpt, Device::Cpu);
    std::fs::create_dir_all(out.join("inference")).unwrap();
    std::fs::copy(ckpt.join("inference/config.json"), out.join("inference/config.json")).unwrap();
    let fp4 = fp4_table(Device::Cpu);
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    let mut write = |file: &str, recs: Vec<(String, tch::Tensor)>| {
        let mut seen = std::collections::HashSet::new();
        let recs: Vec<(String, tch::Tensor)> = recs.into_iter().filter(|(k, _)| seen.insert(k.clone())).map(|(k, t)| (k, t.contiguous())).collect();
        for (k, _) in &recs { map.insert(k.clone(), file.to_string()); }
        let refs: Vec<(&str, &tch::Tensor)> = recs.iter().map(|(k, t)| (k.as_str(), t)).collect();
        tch::Tensor::write_safetensors(&refs, out.join(file)).expect("write shard file");
    };
    let t0 = std::time::Instant::now();
    ck.start_record();
    let _ = ck.rows("embed.weight", sp.v0, sp.v1 - sp.v0, Kind::BFloat16);
    let _ = ck.rows("head.weight", sp.v0, sp.v1 - sp.v0, Kind::BFloat16);
    let _ = ck.get("norm.weight", Kind::BFloat16);
    write("global.safetensors", ck.take_record());
    for l in 0..c.n_layers {
        ck.start_record();
        let _ = Model::load_layer(&ck, &c, &sp, l, &fp4);
        write(&format!("layer{l:02}.safetensors"), ck.take_record());
        eprintln!("[ds41-shard] rank{rank}/{world} layer {l} written ({:.0}s)", t0.elapsed().as_secs_f64());
    }
    let idx = json!({"metadata": {"ds41_shard": {"rank": rank, "world": world, "source": ckpt.canonicalize().unwrap().to_string_lossy(),
        "groups": [sp.g0, sp.g1], "inter": [sp.i0, sp.i1], "vocab": [sp.v0, sp.v1]}}, "weight_map": map});
    std::fs::write(out.join("model.safetensors.index.json"), serde_json::to_string_pretty(&idx).unwrap()).unwrap();
    eprintln!("[ds41-shard] rank{rank}/{world} done in {:.0}s -> {}", t0.elapsed().as_secs_f64(), out.display());
}

/// `ds41-shard-mtp <ckpt> <shard_dir> <rank> <world>`: appends the DSpark stages (what `spec::Mtp::load` takes) to an
/// existing shard directory as `mtp.safetensors` and adds them to its index.
pub fn shard_mtp(ckpt: &Path, out: &Path, rank: usize, world: usize) {
    let _g = tch::no_grad_guard();
    let c = Cfg::load(ckpt);
    let sp = Split::new(&c, rank, world);
    let ck = Checkpoint::open(ckpt, Device::Cpu);
    let fp4 = fp4_table(Device::Cpu);
    let t0 = std::time::Instant::now();
    ck.start_record();
    let _ = super::spec::Mtp::load(&ck, &c, &sp, &fp4, Device::Cpu);
    let mut seen = std::collections::HashSet::new();
    let recs: Vec<(String, tch::Tensor)> = ck.take_record().into_iter().filter(|(k, _)| seen.insert(k.clone())).map(|(k, t)| (k, t.contiguous())).collect();
    let refs: Vec<(&str, &tch::Tensor)> = recs.iter().map(|(k, t)| (k.as_str(), t)).collect();
    tch::Tensor::write_safetensors(&refs, out.join("mtp.safetensors")).expect("write mtp shard");
    let ip = out.join("model.safetensors.index.json");
    let mut idx: serde_json::Value = serde_json::from_reader(std::fs::File::open(&ip).expect("shard index")).unwrap();
    for (k, _) in &recs { idx["weight_map"][k] = json!("mtp.safetensors"); }
    std::fs::write(&ip, serde_json::to_string_pretty(&idx).unwrap()).unwrap();
    eprintln!("[ds41-shard] rank{rank}/{world} DSpark stages ({} tensors) appended in {:.0}s", recs.len(), t0.elapsed().as_secs_f64());
}

/// `ds41-shard-pack <shard_dir>`: rewrites the files of a shard that hold routed experts so that every block's experts
/// are stored as the fused kernel's six packed stacks (`<p>.ffn.experts.packed.{w1,w1s,w3,w3s,w2,w2s}`, kern::moe_pack
/// layout) instead of one tensor per expert and matrix: loading then reads ~100 large tensors per layer instead of
/// ~2300 small ones and skips the stack + repack on the GPU. Other tensors are copied unchanged (byte types stored
/// as U8, which the loader reads the same way).
pub fn shard_pack(dir: &Path) {
    let _g = tch::no_grad_guard();
    let t0 = std::time::Instant::now();
    let ck = Checkpoint::open(dir, Device::Cpu);
    let ip = dir.join("model.safetensors.index.json");
    let mut idx: serde_json::Value = serde_json::from_reader(std::fs::File::open(&ip).expect("shard index")).unwrap();
    let wm: BTreeMap<String, String> = idx["weight_map"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
    let mut files: Vec<String> = wm.values().cloned().collect();
    files.sort();
    files.dedup();
    let kind_of = |d: &str| match d { "BF16" => Kind::BFloat16, "F16" => Kind::Half, "F32" => Kind::Float, "I32" => Kind::Int, "I64" => Kind::Int64, _ => Kind::Uint8 };
    for file in files {
        let names: Vec<&String> = wm.iter().filter(|(_, f)| **f == file).map(|(k, _)| k).collect();
        if !names.iter().any(|k| k.contains(".ffn.experts.") && !k.contains(".packed.")) { continue; }
        let mut out: Vec<(String, tch::Tensor)> = Vec::new();
        let mut prefixes: Vec<String> = names.iter().filter_map(|k| k.split_once(".experts.").map(|(p, _)| p.to_string())).collect();
        prefixes.sort();
        prefixes.dedup();
        for k in &names {
            if k.contains(".ffn.experts.") { continue; }
            out.push((k.to_string(), ck.get(k, kind_of(&ck.info(k).dtype))));
        }
        for p in &prefixes {
            let ne = names.iter().filter(|k| k.starts_with(&format!("{p}.experts.")) && k.ends_with(".w1.weight")).count();
            let st = |m: &str, s: &str| tch::Tensor::stack(&(0..ne).map(|e| ck.get(&format!("{p}.experts.{e}.{m}.{s}"), Kind::Uint8)).collect::<Vec<_>>(), 0);
            for m in ["w1", "w3", "w2"] {
                let (w, s) = super::kern::moe_pack(&st(m, "weight"), &st(m, "scale"));
                out.push((format!("{p}.experts.packed.{m}"), w));
                out.push((format!("{p}.experts.packed.{m}s"), s));
            }
        }
        let tmp = dir.join(format!("{file}.tmp"));
        let refs: Vec<(&str, &tch::Tensor)> = out.iter().map(|(k, t)| (k.as_str(), t)).collect();
        tch::Tensor::write_safetensors(&refs, &tmp).expect("write packed shard file");
        std::fs::rename(&tmp, dir.join(&file)).unwrap();
        let map = idx["weight_map"].as_object_mut().unwrap();
        for k in &names { if k.contains(".ffn.experts.") { map.remove(k.as_str()); } }
        for (k, _) in &out { map.insert(k.clone(), json!(file.clone())); }
        std::fs::write(&ip, serde_json::to_string_pretty(&idx).unwrap()).unwrap();
        eprintln!("[ds41-shard-pack] {file}: {} blocks packed ({:.0}s)", prefixes.len(), t0.elapsed().as_secs_f64());
    }
}
