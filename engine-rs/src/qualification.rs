//! Reuse one resident model across independent final acceptance stages.
use std::path::Path;
use tch::Device;

const TARGET_FP8_FLAGS:[&str;3]=["GLM53_FP8_SHARED","GLM53_FP8_MLA","GLM53_FP8_HEAD"];
const TARGET_FP8_VARIANTS:[(&str,[bool;3]);4]=[
    ("shared",[true,false,false]),("mla",[false,true,false]),
    ("head",[false,false,true]),("combined",[true,true,true]),
];
fn target_fp8_settings(enabled:[bool;3])->[(&'static str,&'static str,&'static str);3] {
    // Every comparison explicitly disables all three in its baseline. In
    // particular, the previous single-group candidate cannot leak into the
    // next comparison, nor can flags inherited from the launch profile.
    std::array::from_fn(|i|(TARGET_FP8_FLAGS[i],"0",if enabled[i]{"1"}else{"0"}))
}
struct TargetFp8Environment([Option<String>;3]);
impl Drop for TargetFp8Environment {
    fn drop(&mut self) {
        for (flag,value) in TARGET_FP8_FLAGS.into_iter().zip(&self.0) {
            match value {Some(value)=>std::env::set_var(flag,value),None=>std::env::remove_var(flag)}
        }
    }
}

/// Isolate the new target quantization from the existing r6 precision profile.
/// This records numerical/routing/task evidence; it does not invent a quality
/// threshold or replace candidate-specific graph and speculative acceptance checks.
fn target_fp8_check(eng:&mut crate::forward::Engine,suite:&Path,out:&Path) {
    use serde_json::json;
    let before=crate::session::signature();
    let saved=TargetFp8Environment(TARGET_FP8_FLAGS.map(|flag|std::env::var(flag).ok()));
    let precision:std::collections::BTreeMap<_,_>=["GLM53_W_FP16","GLM53_DENSE_TP","GLM53_VOCAB_TP",
        "GLM53_DENSE_FP8","GLM53_TF32","GLM53_FP8_WMMA","GLM53_FP8_SPLITS","GLM53_FP8_LARGE","GLM53_FP8_EPILOGUE"]
        .into_iter().map(|flag|(flag,std::env::var(flag).ok())).collect();
    let incoming:std::collections::BTreeMap<_,_>=TARGET_FP8_FLAGS.into_iter().zip(saved.0.iter().cloned()).collect();
    std::fs::create_dir_all(out).unwrap();
    let mut completed=Vec::new();
    for (name,enabled) in TARGET_FP8_VARIANTS {
        let settings=target_fp8_settings(enabled);
        crate::evaluation::settings_check(eng,suite,&out.join(name),&settings);
        completed.push(json!({"variant":name,"settings":settings,"metrics":format!("{name}/metrics.json")}));
        std::fs::write(out.join(format!("manifest-rank{}.json",crate::tp::world().rank)),
            serde_json::to_string_pretty(&json!({
                "suite":suite,"rank":crate::tp::world().rank,"unchanged_precision":precision,
                "incoming_target_fp8_flags":incoming,"completed":completed,
                "all_variants_complete":completed.len()==TARGET_FP8_VARIANTS.len(),
                "quality_acceptance":"not asserted; inspect teacher-forced logits, routing, natural generation and task results",
                "reference_note":"Both arms retain the same EXL3 expert checkpoint and r6 settings; only the listed non-expert FP8 flags change. These add quantization and do not restore prior quantization loss.",
                "metrics_owner_rank":0,"performance_measurement":false
            })).unwrap()).unwrap();
    }
    drop(saved);
    assert_eq!(before,crate::session::signature(),"target FP8 diagnostics changed the caller's configuration");
    eprintln!("[target-fp8-quality] rank{} four diagnostic comparisons completed; quality/performance acceptance remains separate",crate::tp::world().rank);
}

/// Existing quantized weights; change only the small-M instruction/layout arm.
/// Candidate selection remains explicit and is not promoted by this diagnostic.
fn small_fp8_check(eng:&mut crate::forward::Engine,drafter:&crate::dflash::Drafter,suite:&Path,out:&Path) {
    use serde_json::json;
    struct Environment(std::collections::BTreeMap<std::ffi::OsString,std::ffi::OsString>);
    impl Drop for Environment {fn drop(&mut self) {
        let keys:Vec<_>=std::env::vars_os().filter(|(k,_)|k.to_string_lossy().starts_with("GLM53_")).map(|(k,_)|k).collect();
        for key in keys {std::env::remove_var(key);}
        for (key,value) in &self.0 {std::env::set_var(key,value);}
        crate::tp::set_tf32(std::env::var("GLM53_TF32").as_deref()==Ok("1"));
    }}
    let before=crate::session::signature();
    let saved=Environment(std::env::vars_os().filter(|(k,_)|k.to_string_lossy().starts_with("GLM53_")).collect());
    let candidate=crate::dense_fp8::small_kernel_signature();assert!(candidate!=(false,false),"select a candidate FP8 small arm explicitly");
    assert_eq!(std::env::var("GLM53_DENSE_FP8").as_deref(),Ok("1"),"qualification must exercise registered target FP8 weights");
    let mode=std::env::var("GLM53_FP8_WMMA").ok().and_then(|v|v.parse::<i32>().ok()).unwrap_or(0);
    assert!(mode>0,"WMMA mode0 silently falls back to SIMT");
    assert!(!crate::root_probe::retain_f32() && !crate::root_probe::full_f32(),"FP32 diagnostic override can bypass the target FP8 path");
    // This first integrated gate isolates target KDA/dense. BF16 compatibility
    // is covered locally; drafter FP8 needs an independent full-model gate.
    for flag in ["GLM53_DRAFT_FP8_ATTN","GLM53_DRAFT_FP8_CONV","GLM53_DRAFT_FP8_FC","GLM53_DRAFT_FP8_HEAD","GLM53_DRAFT_FP8_MLP"] {
        assert_ne!(std::env::var(flag).as_deref(),Ok("1"),"target FP8 small qualification keeps drafter FP8 off");
    }
    let tp=crate::tp::world();assert_eq!(tp.world,2);
    let packet=tch::Tensor::zeros([2,2],(tch::Kind::Float,eng.w.device));
    packet.get(tp.rank as i64).copy_(&tch::Tensor::from_slice(&[if candidate.0{1.0f32}else{0.0f32},if candidate.1{1.0f32}else{0.0f32}]).to_device(eng.w.device));
    crate::tp::allreduce(&packet);
    assert!(packet.get(0).equal(&packet.get(1)),"ranks selected different FP8 small arms");
    for flag in ["GLM53_KDA_FORK_FUSED","GLM53_KDA_CONV_CHAIN","GLM53_KDA_CHAIN_NORM","GLM53_KDA_CHAIN_RECURRENT","GLM53_KDA_CORRECTION_REPLAY"] {
        std::env::set_var(flag,"1");
    }
    std::fs::create_dir_all(out).unwrap();let path=out.join(format!("small-fp8-rank{}.json",tp.rank));
    std::fs::write(&path,r#"{"gate":false}"#).unwrap();
    let settings=[("GLM53_FP8_SMALL_TRANSPOSE","0",if candidate.0{"1"}else{"0"}),
                  ("GLM53_FP8_SMALL_PAD","0",if candidate.1{"1"}else{"0"})];
    if crate::dsa_topk::enabled(){assert!(crate::dsa_index::enabled(),"TopK batch qualification requires index fusion");}
    crate::verifier_probe::correction_check(eng,drafter,&out.join("verifier"));
    crate::spec_probe::fp8_small_graph_check(eng,&out.join("graph-identity"));
    if crate::shared_gu::enabled(){crate::spec_probe::shared_gu_graph_check(eng,&out.join("shared-gu-graphs"));}
    crate::spec_probe::graph_check(eng,&out.join("topology-graphs"));
    crate::evaluation::settings_check(eng,suite,&out.join("task-quality"),&settings);
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"gate":true,"rank":tp.rank,"candidate_flags":candidate,
        "reference_flags":[false,false],"verifier_raw_bits_and_states_passed":true,"graph_identity_passed":true,
        "task_quality_completed":true,"quality_acceptance":"not asserted; inspect task-quality metrics and run speculative ABBA acceptance",
        "scope":"existing target FP8 quantization, selected small kernel arm; correctness and quality diagnostics, not performance"})).unwrap()).unwrap();
    drop(saved);assert_eq!(before,crate::session::signature(),"FP8 small diagnostic leaked configuration");
}

pub fn run(model:&Path,draft:&Path,suite:&Path,out:&Path) {
    // Explicit preflight before allocating/loading the main model.
    for p in [model.join("config.json"),draft.join("config.json"),draft.join("model.safetensors"),suite.to_owned()] {
        assert!(p.is_file(),"missing input {}",p.display());
    }
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    let tp=crate::tp::init_from_env();assert_eq!(tp.world,2);assert!(crate::tp::dense_enabled());
    for f in ["GLM53_MLA_LATENT","GLM53_MHC_FUSED","GLM53_KDA_FUSED"]{std::env::set_var(f,"1");}
    let dev=Device::Cuda(0);let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let w=crate::weights::ModelWeights::load(model,&cfg,cfg.num_hidden_layers,dev);
    let mut fast=crate::moefast::MoeFast::new(model,cfg.num_hidden_layers,cfg.n_routed_experts,cfg.num_hidden_layers*cfg.n_routed_experts+16,dev);
    fast.preload_all(cfg.num_hidden_layers,cfg.n_routed_experts,dev);fast.assume_hot=true;
    let mut eng=crate::forward::Engine{w,fast:Some(fast),native:None,pool:crate::moe::ExpertPool::new(model,4)};
    // Before every early-return branch: reuse this resident engine, isolate
    // flags with each probe's RAII, and do not silently accept flag0 as a gate.
    if crate::dsa_topk::enabled() || std::env::var("GLM53_QUALIFY_DSA_TOPK").as_deref()==Ok("1") {
        assert!(crate::dsa_topk::enabled() && crate::dsa_index::enabled(),"DSA TopK qualification requires both explicit candidates");
        crate::dsa_topk::real_check(&mut eng,suite,&out.join("dsa-topk-real"));
        crate::spec_probe::dsa_topk_graph_check(&mut eng,&out.join("dsa-topk-graphs"));
    }
    if crate::dsa_position::enabled() || std::env::var("GLM53_QUALIFY_DSA_POSITION").as_deref()==Ok("1") {
        assert!(crate::dsa_position::enabled()&&crate::dsa_topk::enabled()&&crate::dsa_index::enabled(),
            "DSA position qualification requires capture + TopK batch + fused index");
        crate::spec_probe::dsa_position_graph_check(&mut eng,&out.join("dsa-position-graphs"));
    }
    if std::env::var("GLM53_QUALIFY_FP8_SKINNY").as_deref()==Ok("1") {
        // W04 is L1 (summation order changes) and only engages for 2..16-row verification
        // batches, so evidence comes from the speculative resident ABBA plan (tokens,
        // acceptance, rounds, task outputs per arm), not from prefill/M1 teacher forcing.
        let plan=std::env::var_os("GLM53_RESIDENT_SWEEP_PLAN").expect("FP8 skinny qualification needs a resident plan");
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        crate::resident_plan::run(&mut eng,&drafter,Path::new(&plan),&out.join("resident-plan"));
        return;
    }
    let final_norm_gate=std::env::var("GLM53_QUALIFY_DRAFT_FINAL_NORM").as_deref()==Ok("1");
    let conv_gate=std::env::var("GLM53_QUALIFY_DRAFT_CONV").as_deref()==Ok("1");
    if std::env::var("GLM53_QUALIFY_FP8_SMALL").as_deref()==Ok("1") {
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        if final_norm_gate {crate::draft_final_norm::check(&drafter,&eng.w,&out.join("draft-final-norm"));}
        if conv_gate {crate::draft_conv::check(&drafter,&eng.w,&out.join("draft-conv-real"));}
        small_fp8_check(&mut eng,&drafter,suite,&out.join("fp8-small"));
        if let Some(plan)=std::env::var_os("GLM53_RESIDENT_SWEEP_PLAN") {
            crate::resident_plan::run(&mut eng,&drafter,Path::new(&plan),&out.join("resident-plan"));
        }
        return;
    }
    let selector_gate=std::env::var("GLM53_QUALIFY_DRAFT_SELECTOR").as_deref()==Ok("1");
    if std::env::var("GLM53_QUALIFY_CORRECTION").as_deref()==Ok("1") {
        for flag in ["GLM53_KDA_FORK_FUSED","GLM53_KDA_CONV_CHAIN","GLM53_KDA_CHAIN_NORM","GLM53_KDA_CHAIN_RECURRENT",
            "GLM53_KDA_CORRECTION_REPLAY","GLM53_TARGET_TOP1_TP","GLM53_MLA_WEIGHT_CACHE"] {
            std::env::set_var(flag,"1");
        }
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        if final_norm_gate {crate::draft_final_norm::check(&drafter,&eng.w,&out.join("draft-final-norm"));}
        if conv_gate {crate::draft_conv::check(&drafter,&eng.w,&out.join("draft-conv-real"));}
        if selector_gate {crate::draft_selector::check(&drafter,&eng.w,&out.join("draft-selector-real"));}
        if std::env::var("GLM53_QUALIFY_FEATURE_PREFIX").as_deref()==Ok("1") {
            crate::feature_prefix::probe(&drafter,&eng.w,&out.join("feature-prefix-local"));
            crate::spec_probe::feature_prefix_selection_check(&mut eng,&drafter,&out.join("feature-prefix-selection"));
        }
        let index_gate=std::env::var("GLM53_QUALIFY_DSA_INDEX").as_deref()==Ok("1");
        if index_gate {
            assert!(crate::dsa_index::enabled(),"index qualification requires the candidate path");
            std::env::set_var("GLM53_DSA_ALL_VISIBLE","0");
            std::env::set_var("GLM53_DSA_VISIBLE_DIRECT","0");
        }
        if crate::dsa_topk::enabled(){assert!(crate::dsa_index::enabled(),"TopK batch qualification requires index fusion");}
        crate::verifier_probe::correction_check(&mut eng,&drafter,&out.join("correction"));
        crate::spec_probe::graph_check(&mut eng,&out.join("correction-graphs"));
        if crate::shared_gu::enabled(){crate::spec_probe::shared_gu_graph_check(&mut eng,&out.join("shared-gu-graphs"));}
        std::env::set_var("GLM53_DSA_ALL_VISIBLE","1");
        std::env::set_var("GLM53_DSA_VISIBLE_DIRECT","1");
        crate::spec_probe::all_visible_graph_check(&mut eng,&out.join("correction-dsa-graphs"));
        crate::spec_session::check(&mut eng,&drafter,&out.join("correction-prefix"));
        if index_gate {crate::spec_probe::dsa_index_graph_check(&mut eng,&out.join("index-toggle-graphs"));}
        if let Some(plan)=std::env::var_os("GLM53_RESIDENT_SWEEP_PLAN") {
            crate::resident_plan::run(&mut eng,&drafter,Path::new(&plan),&out.join("resident-plan"));
        }
        return;
    }
    if std::env::var("GLM53_QUALIFY_FEATURE_PREFIX").as_deref()==Ok("1") || selector_gate || final_norm_gate || conv_gate {
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        if std::env::var("GLM53_QUALIFY_FEATURE_PREFIX").as_deref()==Ok("1") {
            crate::feature_prefix::probe(&drafter,&eng.w,&out.join("feature-prefix-local"));
            crate::spec_probe::feature_prefix_selection_check(&mut eng,&drafter,&out.join("feature-prefix-selection"));
        }
        if final_norm_gate {crate::draft_final_norm::check(&drafter,&eng.w,&out.join("draft-final-norm"));}
        if conv_gate {crate::draft_conv::check(&drafter,&eng.w,&out.join("draft-conv-real"));}
        if selector_gate {crate::draft_selector::check(&drafter,&eng.w,&out.join("draft-selector-real"));}
        if let Some(plan)=std::env::var_os("GLM53_RESIDENT_SWEEP_PLAN") {
            crate::resident_plan::run(&mut eng,&drafter,Path::new(&plan),&out.join("resident-plan"));
        }
        return;
    }
    if std::env::var("GLM53_QUALIFY_DATAFLOW").as_deref()==Ok("1") {
        crate::head_select::check(&eng.w,&out.join("head-select"));
        crate::mla_latent::weight_cache_probe(&mut eng,&out.join("mla-weight-cache"));
        let saved=std::env::var("GLM53_TARGET_TOP1_TP").ok();
        std::env::set_var("GLM53_TARGET_TOP1_TP","1");
        crate::spec_probe::graph_check(&mut eng,&out.join("head-tree-graphs"));
        match saved {Some(v)=>std::env::set_var("GLM53_TARGET_TOP1_TP",v),None=>std::env::remove_var("GLM53_TARGET_TOP1_TP")};
        return;
    }
    if std::env::var("GLM53_QUALIFY_DSA").as_deref()==Ok("1") {
        std::env::set_var("GLM53_DSA_ALL_VISIBLE","1");
        crate::spec_probe::all_visible_graph_check(&mut eng,&out.join("dsa-graphs"));
        crate::dsa_quality::run(&mut eng,suite,&out.join("dsa-quality"));
        return;
    }
    if std::env::var("GLM53_QUALIFY_TARGET_FP8").as_deref()==Ok("1") {
        target_fp8_check(&mut eng,suite,&out.join("target-fp8-quality"));return;
    }
    if std::env::var("GLM53_QUALIFY_R6").as_deref()==Ok("1") || std::env::var("GLM53_QUALIFY_R5").as_deref()==Ok("1") || std::env::var("GLM53_QUALIFY_RELEASE").as_deref()==Ok("1") || std::env::var("GLM53_QUALIFY_RUNTIME").as_deref()==Ok("1") {
        let flags:Vec<_>=["GLM53_MHC_POST_FUSED","GLM53_KDA_FORK_FUSED","GLM53_STATIC_TENSORS",
            "GLM53_MOE_COOP_PERSISTENT","GLM53_DENSE_SMALL","GLM53_MLA_SCORE_2D","GLM53_PREFILL_COOP"].into_iter()
            .filter(|f|std::env::var(f).as_deref()==Ok("1")).collect();
        assert!(!flags.is_empty(),"release must select qualified target candidates");
        if std::env::var("GLM53_QUALIFY_R6").as_deref()==Ok("1") {
            let names=["GLM53_GROUPED_INDEX_PACK","GLM53_FP8_EPILOGUE","GLM53_GROUPED_SWIGLU","GLM53_FP8_LARGE","GLM53_DSA_PREFILL_LIMIT","GLM53_GROUPED_REDUCE","GLM53_DSA_SCORE_FUSED"];
            let values:Vec<_>=names.iter().map(|n|std::env::var(n).unwrap_or_else(|_|"0".into())).collect();
            let settings:Vec<_>=names.iter().zip(&values).map(|(&n,v)|(n,"0",v.as_str())).collect();
            crate::evaluation::settings_check(&mut eng,suite,&out.join("quality"),&settings);
        }else if std::env::var("GLM53_QUALIFY_R5").as_deref()==Ok("1") {
            let names=["GLM53_MLA_ACTIVE_COPY","GLM53_PREFILL_MIN_ROWS","GLM53_MLA_PREFILL_DENSE","GLM53_DENSE_FP8","GLM53_KDA_SEQUENCE","GLM53_MLA_SPARSE_FUSED","GLM53_MLA_PREFILL_BATCHED","GLM53_PREFILL_GROUPED","GLM53_PREFILL_LAST_LOGITS","GLM53_DENSE_LT","GLM53_TF32"];
            let values:Vec<_>=names.iter().map(|n|std::env::var(n).unwrap_or_else(|_|"0".into())).collect();
            let settings:Vec<_>=names.iter().zip(&values).map(|(&n,v)|(n,"0",v.as_str())).collect();
            crate::evaluation::settings_check(&mut eng,suite,&out.join("quality"),&settings);
        }else if std::env::var("GLM53_QUALIFY_RUNTIME").as_deref()!=Ok("1") {
            crate::evaluation::flags_check(&mut eng,suite,&out.join("quality"),&flags);
        }
        crate::spec_probe::graph_check(&mut eng,&out.join("graphs"));
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        crate::spec_session::check(&mut eng,&drafter,&out.join("prefix"));
        let normal_suite=std::env::var("GLM53_SPEC_SUITE").unwrap();
        std::env::remove_var("GLM53_SPEC_ABBA_FLAG");
        std::env::set_var("GLM53_SPEC_ROUNDS","1");
        std::env::set_var("GLM53_SPEC_FULL_WARMUPS","0");
        std::env::set_var("GLM53_SPEC_SUITE",std::env::var("GLM53_SPEC_QUALITY_SUITE").unwrap());
        crate::spec_probe::check(&mut eng,&drafter,&out.join("spec-quality"));
        std::env::set_var("GLM53_SPEC_SUITE",normal_suite);
        std::env::set_var("GLM53_SPEC_ROUNDS","2");std::env::set_var("GLM53_SPEC_FULL_WARMUPS","1");
        crate::spec_probe::check(&mut eng,&drafter,&out.join("spec-perf"));
        eprintln!("[release] rank{} final combination completed",tp.rank);return;
    }
    if std::env::var("GLM53_QUALIFY_FINISH").as_deref()==Ok("1") {
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        crate::spec_session::check(&mut eng,&drafter,&out.join("prefix"));
        let normal_suite=std::env::var("GLM53_SPEC_SUITE").unwrap();
        let phases=std::env::var("GLM53_FINISH_PHASES").unwrap_or_else(|_|"small,persistent,static,rope,mlp,kv-long,mla-tree,mla-prefill,depths,trees,adaptive".into());
        let mla_scope=std::env::var("GLM53_MLA_SCORE_2D_SCOPE").ok();
        for phase in phases.split(',') {
            eprintln!("[finish] rank{} phase={phase}",tp.rank);
            std::env::set_var("GLM53_SPEC_SUITE",&normal_suite);
            std::env::set_var("GLM53_SPEC_MODES","batch-graph-chain");
            std::env::set_var("GLM53_SPEC_MAX_DRAFT","7");
            std::env::set_var("GLM53_SPEC_ROUNDS","4");
            std::env::remove_var("GLM53_SPEC_ABBA_FLAG");
            match phase {
                "prefill-quality"=>{
                    crate::evaluation::flags_check(&mut eng,suite,&out.join(phase),&["GLM53_PREFILL_COOP"]);
                    std::env::set_var("GLM53_PREFILL_COOP","0");
                },
                "small"=>crate::spec_probe::small_tree_check(&mut eng,&out.join(phase)),
                "static"=>{
                    crate::evaluation::flags_check(&mut eng,suite,&out.join("static-quality"),&["GLM53_STATIC_TENSORS"]);
                    std::env::set_var("GLM53_BENCH_GRAPH_ONLY","1");
                    let small=std::env::var("GLM53_TP_SMALL_COMM_ACTIVE").unwrap();std::env::set_var("GLM53_TP_SMALL_COMM_ACTIVE","0");
                    crate::kernel_bench::full_check(&mut eng,&out.join("static-m1"),"GLM53_STATIC_TENSORS");
                    std::env::set_var("GLM53_TP_SMALL_COMM_ACTIVE",small);std::env::set_var("GLM53_STATIC_TENSORS","1");
                },
                "depths"=>{
                    std::env::set_var("GLM53_SPEC_ROUNDS","2");std::env::set_var("GLM53_SPEC_FULL_WARMUPS","1");
                    for depth in [3,5,7] {std::env::set_var("GLM53_SPEC_MAX_DRAFT",depth.to_string());
                        crate::spec_probe::check(&mut eng,&drafter,&out.join(format!("depth-{depth}")));}
                },
                "trees"|"adaptive"=>{
                    std::env::set_var("GLM53_SPEC_ROUNDS","2");std::env::set_var("GLM53_SPEC_FULL_WARMUPS","1");
                    if phase=="trees" {
                        std::env::set_var("GLM53_SPEC_SUITE",std::env::var("GLM53_ROUTE_SUITE").unwrap());
                        std::env::set_var("GLM53_SPEC_MODES","batch-tree,batch-graph-tree");
                    } else {std::env::set_var("GLM53_SPEC_MODES","batch-adaptive-chain");}
                    crate::spec_probe::check(&mut eng,&drafter,&out.join(phase));
                },
                "prefill-coop"|"prefill-coop-long"|"mla-prefill-long"|"topk"|"static-spec"|"persistent"|"rope"|"mlp"|"kv-long"|"mla-tree"|"mla-prefill"=>{
                    let flag=match phase{"prefill-coop"|"prefill-coop-long"=>"GLM53_PREFILL_COOP","topk"=>"GLM53_DRAFT_TOPK_TP","static-spec"=>"GLM53_STATIC_TENSORS","persistent"=>"GLM53_MOE_COOP_PERSISTENT","rope"=>"GLM53_DRAFT_ROPE_CACHE","mlp"=>"GLM53_DRAFT_MLP_TP", "kv-long"=>"GLM53_DRAFT_KV_BUFFER",_=>"GLM53_MLA_SCORE_2D"};
                    if phase=="kv-long" || phase=="mla-prefill-long" || phase=="prefill-coop-long" {std::env::set_var("GLM53_SPEC_SUITE",std::env::var("GLM53_LONG_SPEC_SUITE").unwrap());}
                    if phase.starts_with("mla-"){std::env::set_var("GLM53_MLA_SCORE_2D_SCOPE",if phase=="mla-prefill-long"{"prefill"}else{phase.trim_start_matches("mla-")});}
                    std::env::set_var("GLM53_SPEC_ABBA_FLAG",flag);
                    crate::spec_probe::check(&mut eng,&drafter,&out.join(phase));
                    std::env::set_var(flag,"0");
                    match &mla_scope {Some(scope)=>std::env::set_var("GLM53_MLA_SCORE_2D_SCOPE",scope),None=>std::env::remove_var("GLM53_MLA_SCORE_2D_SCOPE")}
                },
                _=>panic!("unknown finish phase {phase}"),
            }
            std::env::remove_var("GLM53_SPEC_FULL_WARMUPS");
        }
        eprintln!("[finish] rank{} completed",tp.rank);return;
    }
    if std::env::var("GLM53_QUALIFY_NEXT").as_deref()==Ok("1") {
        let flags=["GLM53_MHC_POST_FUSED","GLM53_KDA_FORK_FUSED","GLM53_MOE_COOP_PERSISTENT"];
        for flag in flags{std::env::set_var(flag,"1");}
        crate::spec_probe::graph_check(&mut eng,&out.join("graphs"));
        crate::evaluation::flags_check(&mut eng,suite,&out.join("quality"),&flags);
        for flag in flags{std::env::set_var(flag,"0");}
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        for flag in flags {
            std::env::set_var("GLM53_SPEC_ABBA_FLAG",flag);std::env::set_var("GLM53_SPEC_ROUNDS","4");
            crate::spec_probe::check(&mut eng,&drafter,&out.join(flag));std::env::set_var(flag,"0");
        }
        std::env::remove_var("GLM53_SPEC_ABBA_FLAG");
        // Actual graph-verifier routing, explicitly separated from timings.
        if let Ok(route_suite)=std::env::var("GLM53_ROUTE_SUITE") {
            let saved=std::env::var("GLM53_SPEC_SUITE").ok();
            std::env::set_var("GLM53_SPEC_SUITE",route_suite);std::env::set_var("GLM53_SPEC_ROUNDS","1");
            std::env::set_var("GLM53_SPEC_ROUTE_TRACE",out.join("routes"));
            crate::spec_probe::check(&mut eng,&drafter,&out.join("route-diagnostic"));
            std::env::remove_var("GLM53_SPEC_ROUTE_TRACE");
            match saved{Some(p)=>std::env::set_var("GLM53_SPEC_SUITE",p),None=>std::env::remove_var("GLM53_SPEC_SUITE")}
        }
        drop(drafter);
        std::env::set_var("GLM53_TP_SMALL_COMM_ACTIVE","0");std::env::set_var("GLM53_MOE_COOP","0");
        std::env::set_var("GLM53_BENCH_GRAPH_ONLY","1");
        for flag in ["GLM53_MHC_POST_FUSED","GLM53_SCRATCH_EMPTY"] {
            crate::kernel_bench::full_check(&mut eng,&out.join("m1-perf").join(flag),flag);
        }
        eprintln!("[m2-checks] rank{} next stages completed",tp.rank);return;
    }
    if std::env::var("GLM53_QUALIFY_TREE_GRAPHS").as_deref()==Ok("1") {
        crate::spec_probe::graph_check(&mut eng,&out.join("graphs"));
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        std::env::set_var("GLM53_SPEC_ROUNDS","2");
        for depth in [3,5,7] {
            std::env::set_var("GLM53_SPEC_MAX_DRAFT",depth.to_string());
            crate::spec_probe::check(&mut eng,&drafter,&out.join(format!("depth-{depth}")));
        }
        return;
    }
    if std::env::var("GLM53_QUALIFY_DRAFT").as_deref()==Ok("1") {
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        crate::spec_session::check(&mut eng,&drafter,&out.join("prefix"));
        for flag in ["GLM53_DRAFT_HEAD_TP","GLM53_DRAFT_KV_BUFFER"] {
            std::env::set_var("GLM53_SPEC_ABBA_FLAG",flag);
            std::env::set_var("GLM53_SPEC_ROUNDS","4");
            crate::spec_probe::check(&mut eng,&drafter,&out.join(flag));
            std::env::set_var(flag,"1");
        }
        std::env::remove_var("GLM53_SPEC_ABBA_FLAG");
        eprintln!("[m2-checks] rank{} drafter stages completed",tp.rank);return;
    }
    if std::env::var("GLM53_QUALIFY_TP_PACK").as_deref()==Ok("1") {
        let flag=if std::env::var("GLM53_QUALIFY_TP_NETWORK").as_deref()==Ok("1") {
            assert!(crate::tp::small_comm_enabled(),"small communicator must be initialized and active");
            "GLM53_TP_SMALL_COMM_ACTIVE"
        }else{"GLM53_TP_MOE_PACK"};
        let perf_only=std::env::var("GLM53_QUALIFY_TP_PACK_PERF_ONLY").as_deref()==Ok("1");
        std::env::set_var(flag,"1");
        if !perf_only {crate::tree_probe::check(&mut eng,&out.join("tree"));}
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        std::env::set_var("GLM53_SPEC_ABBA_FLAG",flag);
        std::env::set_var("GLM53_SPEC_ROUNDS","4");
        crate::spec_probe::check(&mut eng,&drafter,&out.join("spec"));
        drop(drafter);std::env::remove_var("GLM53_SPEC_ABBA_FLAG");
        if !perf_only {crate::evaluation::flags_check(&mut eng,suite,&out.join("quality"),&[flag]);}
        std::env::set_var("GLM53_BENCH_GRAPH_ONLY","1");
        crate::kernel_bench::full_check(&mut eng,&out.join("m1-perf"),flag);
        eprintln!("[m2-checks] rank{} {flag} stages completed",tp.rank);
        return;
    }
    if std::env::var("GLM53_QUALIFY_MLA_LAYOUT").as_deref()==Ok("1") {
        std::env::set_var("GLM53_MLA_SCORE_2D","1");
        crate::tree_probe::check(&mut eng,&out.join("tree"));
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        std::env::set_var("GLM53_SPEC_LAYOUT_ABBA","1");
        std::env::set_var("GLM53_SPEC_ROUNDS","4");
        crate::spec_probe::check(&mut eng,&drafter,&out.join("spec"));
        drop(drafter);
        std::env::remove_var("GLM53_SPEC_LAYOUT_ABBA");
        crate::evaluation::flags_check(&mut eng,suite,&out.join("quality"),&["GLM53_MLA_SCORE_2D"]);
        eprintln!("[m2-checks] rank{} MLA layout stages completed",tp.rank);
        return;
    }
    crate::tree_probe::check(&mut eng,&out.join("tree"));
    if std::env::var("GLM53_QUALIFY_GEMV").as_deref()==Ok("1") {
        std::env::set_var("GLM53_DENSE_GEMV","1");
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        crate::spec_probe::check(&mut eng,&drafter,&out.join("spec"));
        drop(drafter);
        std::env::set_var("GLM53_PREFILL_BATCH","1");
        crate::evaluation::flags_check(&mut eng,suite,&out.join("quality"),&["GLM53_DENSE_GEMV"]);
        std::env::set_var("GLM53_BENCH_GRAPH_ONLY","1");
        crate::kernel_bench::full_check(&mut eng,&out.join("gemv-perf"),"GLM53_DENSE_GEMV");
        std::env::set_var("GLM53_DENSE_GEMV","1");
    } else {
        crate::evaluation::combined_check(&mut eng,suite,&out.join("quality"));
        std::env::remove_var("GLM53_PREFILL_BATCH");
        let drafter=crate::dflash::Drafter::load_target(draft,&eng.w);
        crate::spec_probe::check(&mut eng,&drafter,&out.join("spec"));
    }
    eprintln!("[m2-checks] rank{} all stages completed",tp.rank);
}
