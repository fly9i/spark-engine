//! Qwen3.8-Flash-Next (qwen4_exp, EXL3 4.05 bpw) on one GB10: kernels in shim/qwen_*.cu.
pub mod ffi;
pub mod load;
pub mod model;
pub mod probe;
pub mod spec;
pub mod serve;
pub mod graph;
pub mod lookup;
pub mod pcache;
pub mod vision;
