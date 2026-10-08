//! Verifier-only DSA order diagnostics. Serial settings_check cannot exercise
//! AllVisible. Both arms teacher-force the same eight-node chains and keep
//! independent committed states so accumulated order error remains visible.
use std::path::Path;
use serde_json::{json,Value};
use tch::{Device,Kind,Tensor};
use crate::{dsa::TreeSelection,forward::{Engine,DecodeStates,snapshot}};

fn ints(v:&Value)->Vec<i64> {v.as_array().unwrap().iter().map(|v|v.as_i64().unwrap()).collect()}
fn stats(x:&Tensor)->Value {
    let values:Vec<f64>=Vec::try_from(x.to_kind(Kind::Double).view([-1])).unwrap();
    assert!(!values.is_empty()&&values.iter().all(|v|v.is_finite()));
    let mut sorted=values.clone();sorted.sort_by(f64::total_cmp);
    json!({"mean":values.iter().sum::<f64>()/values.len() as f64,
        "p95":sorted[((sorted.len() as f64*0.95).ceil() as usize-1).min(sorted.len()-1)],
        "max":sorted[sorted.len()-1],"per_position":values})
}
fn metrics(a:&Tensor,b:&Tensor,reference:&[i64])->Value {
    assert_eq!(a.size(),b.size());assert_eq!(a.size()[0] as usize,reference.len());
    assert_eq!(a.device(),Device::Cpu);assert_eq!(b.device(),Device::Cpu);
    let a=a.to_kind(Kind::Float);let b=b.to_kind(Kind::Float);
    assert!(a.isfinite().all().int64_value(&[])!=0&&b.isfinite().all().int64_value(&[])!=0);
    let la=a.log_softmax(-1,Kind::Float);let lb=b.log_softmax(-1,Kind::Float);
    let p=la.exp();let diff=&a-&b;
    let kl=(&p*(&la-&lb)).sum_dim_intlist(&[-1i64][..],false,Kind::Float);
    let relative=diff.norm_scalaropt_dim(2.,[-1],false)/a.norm_scalaropt_dim(2.,[-1],false).clamp_min(1e-30);
    let top_a=a.argmax(-1,false);let top_b=b.argmax(-1,false);
    let agree=top_a.eq_tensor(&top_b).to_kind(Kind::Float);
    let ids=Tensor::from_slice(reference).unsqueeze(-1);
    let nll_a=la.gather(-1,&ids,false).neg().squeeze_dim(-1);
    let nll_b=lb.gather(-1,&ids,false).neg().squeeze_dim(-1);
    json!({"positions":reference.len(),"arithmetic":"FP32 logits, FP32 log_softmax/KL/reductions; JSON scalar conversion only uses FP64",
        "exact_logits":a.equal(&b),"top1_equal_fraction":agree.mean(Kind::Float).double_value(&[]),
        "top1_ranked":Vec::<i64>::try_from(top_a).unwrap(),"top1_candidate":Vec::<i64>::try_from(top_b).unwrap(),
        "raw_logits_max_abs":stats(&diff.abs().max_dim(-1,false).0),"raw_logits_relative_l2":stats(&relative),
        "kl_ranked_to_candidate_nats":stats(&kl),"reference_nll_ranked":stats(&nll_a),
        "reference_nll_candidate":stats(&nll_b),"reference_nll_delta":stats(&(&nll_b-&nll_a))})
}
fn block(eng:&mut Engine,base:&DecodeStates,ids:&[i64],selection:TreeSelection)->(Tensor,DecodeStates) {
    let parents:Vec<_>=(0..ids.len()).map(|i|i.checked_sub(1)).collect();
    let (logits,mut states,_)=eng.tree_forward_selected(&Tensor::from_slice(ids).to_device(eng.w.device),base,&parents,false,selection,true);
    // Only the committed last node survives the block. Discard other writable
    // branch states before evaluating the other arm to bound device workspace.
    let committed=states.pop().unwrap();(logits.to_kind(Kind::Float).to_device(Device::Cpu),committed)
}

/// Diagnostic only: no invented numerical-quality threshold. A wholly Ranked
/// long-prefix control MUST be bitexact, and non-finite results always fail.
/// Natural-generation tokens/acceptance remain a separate spec ABBA experiment.
pub fn run(eng:&mut Engine,suite:&Path,out:&Path) {
    assert!(crate::mla_latent::enabled());let _guard=tch::no_grad_guard();
    let suite:Value=serde_json::from_str(&std::fs::read_to_string(suite).unwrap()).unwrap();
    let mut cases=suite["cases"].as_array().unwrap().clone();assert!(!cases.is_empty());
    let limit=std::env::var("GLM53_DSA_QUALITY_POSITIONS").ok().map(|v|v.parse::<usize>().unwrap()).unwrap_or(64);
    assert!((2..=256).contains(&limit));
    // The normal diagnostic suite has no >2052-token prefix. Add an explicit
    // synthetic fallback control, labelled separately from task quality.
    if !cases.iter().any(|c|c["prompt_ids"].as_array().unwrap().len()>=2052 && c["reference_ids"].as_array().unwrap().len()>1) {
        let source=ints(&cases[0]["prompt_ids"]);let mut prompt=source.clone();assert!(!source.is_empty());
        while prompt.len()<2052 {prompt.extend(source.iter().copied().take(2052-prompt.len()));}
        let mut reference=ints(&cases[0]["reference_ids"]);assert!(!reference.is_empty());
        while reference.len()<9 {reference.push(reference[0]);}reference.truncate(9);
        cases.push(json!({"name":"dsa_fallback_2052_synthetic_control","prompt_ids":prompt,"reference_ids":reference,"dsa_synthetic_control":true}));
    }
    let original=std::env::var("GLM53_DSA_ALL_VISIBLE").ok();std::env::set_var("GLM53_DSA_ALL_VISIBLE","1");
    std::fs::create_dir_all(out).unwrap();let rank=crate::tp::world().rank;
    let output=out.join(format!("dsa-tree-quality-rank{rank}.json"));let mut records=Vec::new();let mut long_controls=0;
    std::fs::write(&output,r#"{"complete":false,"mathematical_gate":false,"quality_gate":null,"cases":[]}"#).unwrap();
    for case in cases {
        let prompt=ints(&case["prompt_ids"]);let all_refs=ints(&case["reference_ids"]);
        let reference=&all_refs[..all_refs.len().min(limit)];assert!(!prompt.is_empty()&&!reference.is_empty());
        assert!(prompt.len()+reference.len()-1<=crate::mla_latent::capacity() as usize);
        let (prefix,mut ranked,_)=eng.prefill_record_last(&Tensor::from_slice(&prompt).to_device(eng.w.device),None,false);
        let mut candidate=snapshot(&ranked);
        let first=prefix.get(prefix.size()[0]-1).to_kind(Kind::Float).to_device(Device::Cpu).unsqueeze(0);
        let mut ranked_logits=vec![first.shallow_clone()];let mut candidate_logits=vec![first];
        let mut blocks=Vec::new();let mut all_visible_blocks=0;let mut committed=0;
        let long_control=prompt.len()>=2052;
        for start in (0..reference.len().saturating_sub(1)).step_by(8) {
            let end=(start+8).min(reference.len()-1);let input=&reference[start..end];
            let selection=crate::dsa::tree_selection((prompt.len()+committed) as i64,input.len() as i64);
            if selection==TreeSelection::AllVisible{all_visible_blocks+=1;}
            let (a,next_a)=block(eng,&ranked,input,TreeSelection::Ranked);ranked=next_a;
            let (b,next_b)=block(eng,&candidate,input,selection);candidate=next_b;
            let state_max_abs=crate::forward::states_max_diff(&ranked,&candidate);
            assert!(state_max_abs.is_finite(),"non-finite DSA tree states");
            if long_control {assert_eq!(selection,TreeSelection::Ranked);assert!(a.equal(&b),"Ranked fallback logits changed");assert_eq!(state_max_abs,0.,"Ranked fallback states changed");}
            blocks.push(json!({"prefix_len":prompt.len()+committed,"nodes":input.len(),"selection":format!("{selection:?}"),
                "cumulative_state_max_abs":state_max_abs,"metrics":metrics(&a,&b,&reference[start+1..end+1])}));
            committed+=input.len();ranked_logits.push(a);candidate_logits.push(b);
        }
        if long_control&&!blocks.is_empty(){long_controls+=1;}
        let a=Tensor::cat(&ranked_logits,0);let b=Tensor::cat(&candidate_logits,0);
        let archive=out.join(format!("case-{:03}-rank{rank}.pt",records.len()));
        Tensor::save_multi(&[("ranked_logits",&a),("candidate_logits",&b),("reference_ids",&Tensor::from_slice(reference))],&archive).unwrap();
        records.push(json!({"name":case["name"],"prompt_len":prompt.len(),"reference_positions":reference.len(),"source_reference_positions":all_refs.len(),
            "synthetic_control":case["dsa_synthetic_control"].as_bool().unwrap_or(false),"long_ranked_exact_control":long_control,
            "all_visible_blocks":all_visible_blocks,"independent_state_continuation":true,"blocks":blocks,"metrics":metrics(&a,&b,reference),
            "logits_archive":archive}));
        std::fs::write(&output,serde_json::to_string_pretty(&json!({"complete":false,"mathematical_gate":false,"quality_gate":null,"cases":records})).unwrap()).unwrap();
        eprintln!("[dsa-tree-quality] rank{rank} {} prefix={} positions={} all_visible_blocks={all_visible_blocks}",case["name"],prompt.len(),reference.len());
    }
    assert!(long_controls>0);
    std::fs::write(&output,serde_json::to_string_pretty(&json!({"complete":true,"mathematical_gate":true,"quality_gate":null,
        "long_ranked_exact_controls":long_controls,"max_teacher_positions":limit,"cases":records,
        "scope":"Teacher-forced eight-node tree verifier with independent accumulated states. Ascending selection changes FP32 accumulation order. Numerical/task acceptance is not asserted; natural generation and speculative acceptance require separate ABBA."})).unwrap()).unwrap();
    if let Some(value)=original{std::env::set_var("GLM53_DSA_ALL_VISIBLE",value);}else{std::env::remove_var("GLM53_DSA_ALL_VISIBLE");}
}
