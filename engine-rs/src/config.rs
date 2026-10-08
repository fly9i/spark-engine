// SPDX-License-Identifier: MIT
//! config.json 解析(轻量,只取 M0 需要的字段)。

use std::fs;
use std::io;
use std::path::Path;
use serde_json::Value;

pub struct LayerPlan {
    pub idx: usize,
    pub attn: &'static str, // "kda" | "dsa"
    pub mlp: &'static str,  // "dense" | "sparse"
}

pub struct Config {
    pub num_hidden_layers: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub plans: Vec<LayerPlan>,
}

pub fn load(path: &Path) -> io::Result<Config> {
    let root: Value = serde_json::from_str(&fs::read_to_string(path)?)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let tc = &root["text_config"];
    let lt = tc["layer_types"].as_array().expect("layer_types");
    let mt = tc["mlp_layer_types"].as_array().expect("mlp_layer_types");
    let plans = lt
        .iter()
        .zip(mt.iter())
        .enumerate()
        .map(|(i, (a, m))| LayerPlan {
            idx: i,
            attn: if a.as_str() == Some("linear_attention") { "kda" } else { "dsa" },
            mlp: if m.as_str() == Some("sparse") { "sparse" } else { "dense" },
        })
        .collect();
    Ok(Config {
        num_hidden_layers: lt.len(),
        n_routed_experts: tc["n_routed_experts"].as_u64().expect("n_routed_experts") as usize,
        num_experts_per_tok: tc["num_experts_per_tok"].as_u64().unwrap_or(8) as usize,
        plans,
    })
}
