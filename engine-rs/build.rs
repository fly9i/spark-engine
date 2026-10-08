// SPDX-License-Identifier: MIT
//! Compiles the C++ / CUDA shim (engine kernels, the exllamav3_ext bridge, c10d) and links the engine.
use std::env;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::process::Command;

/// Objects are cached by content: hash of the source and of every header in shim/ (any header change recompiles
/// all), stored outside OUT_DIR so they survive build-script reruns; missing ones compile in parallel.
fn cache_dir() -> String {
    let d = format!("{}/target/obj-cache", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn headers_hash() -> u64 {
    let mut names: Vec<_> = std::fs::read_dir("shim").unwrap().flatten().map(|e| e.path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("cuh") | Some("h") | Some("hpp"))).collect();
    names.sort();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for n in names { std::fs::read(&n).unwrap().hash(&mut h); }
    h.finish()
}
fn cached_object(source: &str, flags: &[String], hh: u64) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::fs::read(source).unwrap().hash(&mut h);
    flags.hash(&mut h);
    hh.hash(&mut h);
    let stem = Path::new(source).file_stem().unwrap().to_str().unwrap();
    format!("{}/{}-{:016x}.o", cache_dir(), stem, h.finish())
}
/// Compile (program, flags + [-c source -o object]) for each job whose cached object is missing, BUILD_JOBS (default 8)
/// at a time (each nvcc / torch-header compile takes GBs of memory).
fn compile_all(jobs: &[(String, Vec<String>, String, String)]) {
    let par: usize = env::var("BUILD_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(8).max(1);
    let todo: Vec<_> = jobs.iter().filter(|j| !Path::new(&j.3).exists()).collect();
    for batch in todo.chunks(par) {
        let children: Vec<_> = batch.iter().map(|(prog, flags, src, obj)| {
            let tmp = format!("{obj}.tmp");
            let child = Command::new(prog).args(flags).args(["-c", src.as_str(), "-o", tmp.as_str()]).spawn().expect("compiler");
            (child, tmp, obj.clone(), src.clone())
        }).collect();
        for (mut c, tmp, obj, src) in children {
            assert!(c.wait().unwrap().success(), "compilation of {src} failed");
            std::fs::rename(&tmp, &obj).unwrap();
        }
    }
}

/// Build inputs (environment):
/// - LIBTORCH: the PyTorch 2.13 (cu130) installation to compile and link against, e.g. the `torch` package directory
///   of a Python environment (`python -c "import torch, os; print(os.path.dirname(torch.__file__))"`). Required.
/// - CUDA_HOME: CUDA 13 toolkit (default /usr/local/cuda).
/// - EXL3_LIB_DIR: directory with libexllamav3_ext.so (exllamav3 v1.4.9, scripts/build-exllamav3.sh; default
///   ../third_party/exllamav3/lib). Also an rpath entry, next to $ORIGIN/../lib.
/// - PYTHON_INCLUDE: Python 3.12 headers (default /usr/include/python3.12).
/// - SPARK_CUDA_ARCH: SM version to compile for (default 121, GB10).
/// NCCL's header template comes from ../third_party/nccl (scripts/fetch-deps.sh).
fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let third = format!("{manifest}/../third_party");
    let libtorch = env::var("LIBTORCH").expect("LIBTORCH: path of the PyTorch 2.13 cu130 installation (see docs/build.md)");
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let exl3 = env::var("EXL3_LIB_DIR").unwrap_or_else(|_| format!("{third}/exllamav3/lib"));
    let pyinc = env::var("PYTHON_INCLUDE").unwrap_or_else(|_| "/usr/include/python3.12".into());
    let arch = env::var("SPARK_CUDA_ARCH").unwrap_or_else(|_| "121".into());
    for v in ["LIBTORCH", "CUDA_HOME", "EXL3_LIB_DIR", "PYTHON_INCLUDE", "SPARK_CUDA_ARCH"] { println!("cargo:rerun-if-env-changed={v}"); }
    let out = env::var("OUT_DIR").unwrap();
    // ProcessGroupNCCL::Options embeds ncclConfig_t: match libtorch's build ABI (NCCL 2.29.7), not the newer runtime wheel.
    let nccl_in = format!("{third}/nccl/nccl.h.in");
    let header = std::fs::read_to_string(&nccl_in).unwrap_or_else(|_| panic!("{nccl_in} missing: run scripts/fetch-deps.sh"))
        .replace("${nccl:Major}", "2").replace("${nccl:Minor}", "29")
        .replace("${nccl:Patch}", "7").replace("${nccl:Suffix}", "")
        .replace("${nccl:Version}", "22907");
    std::fs::write(format!("{out}/nccl.h"), header).expect("generate NCCL header");
    let hh = headers_hash();
    let cxx: Vec<String> = ["-O2", "-std=c++17", "-fPIC", "-DUSE_C10D_NCCL"].iter().map(|s| s.to_string())
        .chain([format!("-I{out}"), format!("-I{libtorch}/include"), format!("-I{libtorch}/include/torch/csrc/api/include"),
                format!("-I{pyinc}"), format!("-I{cuda}/include")]).collect();
    let nv: Vec<String> = ["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "--fmad=false"].iter().map(|s| s.to_string())
        .chain([format!("-gencode=arch=compute_{arch},code=sm_{arch}")]).collect();
    let nvcc = format!("{cuda}/bin/nvcc");
    let kernels = ["mhc","kda","gemv","latent","fp8","dataflow","kda_conv_chain","kda_correction","dsa_index","draft_selector","draft_conv","shared_gu","rdma_ar","rdma_big","fp8_big","l2pf","c12","draft_head4","draft_q4","qwen_exl3","qwen_gdn","qwen_qsa","qwen_moe","qwen_hc","qwen_ple","qwen_f16","qwen_q8","glm_moe","ablate"];
    let mut jobs: Vec<(String, Vec<String>, String, String)> = Vec::new();
    for src in ["shim/shim.cpp", "shim/lt.cpp"] {
        jobs.push(("c++".into(), cxx.clone(), src.into(), cached_object(src, &cxx, hh)));
    }
    for k in kernels {
        let src = format!("shim/{k}.cu");
        let obj = cached_object(&src, &nv, hh);
        jobs.push((nvcc.clone(), nv.clone(), src, obj));
    }
    compile_all(&jobs);
    cc::Build::new().object(&jobs[0].3).object(&jobs[1].3).compile("exl3shim");
    for (i, k) in kernels.iter().enumerate() {
        cc::Build::new().object(&jobs[2 + i].3).compile(&format!("{k}cuda"));
        println!("cargo:rerun-if-changed=shim/{k}.cu");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib:{exl3}");
    println!("cargo:rustc-link-search=native={libtorch}/lib");
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-search=native={exl3}");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    for h in std::fs::read_dir("shim").unwrap().flatten().map(|e| e.path()) {
        if matches!(h.extension().and_then(|e| e.to_str()), Some("cuh") | Some("h") | Some("cpp")) { println!("cargo:rerun-if-changed={}", h.display()); }
    }
    for l in ["cudart", "cublas", "cublasLt", "ibverbs", "exllamav3_ext", "torch", "torch_cpu", "torch_cuda", "c10", "c10_cuda"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    println!("cargo:rerun-if-changed={nccl_in}");
}
