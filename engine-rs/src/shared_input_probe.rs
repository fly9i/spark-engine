//! Local exact dataflow gate for the canonical routed/shared Half input.
//! Reads only layer 3 shared gate/up weights; no Engine or routed expert load.
use std::{ffi::OsString,path::Path,time::Instant};
use serde_json::{json,Value};
use tch::{Device,Kind,Tensor};

const FLAGS:[&str;10]=["GLM53_MOE_INPUT_HALF_REUSE","GLM53_FP8_SHARED",
    "GLM53_DENSE_LT","GLM53_DENSE_LT_TABLE","GLM53_DENSE_GEMV","GLM53_DENSE_SMALL",
    "GLM53_FP8_LARGE","GLM53_FP8_WMMA","GLM53_FP8_SPLITS","GLM53_TF32"];
struct Restore {env:Vec<(&'static str,Option<OsString>)>,precision:&'static str,tf32:bool}
impl Restore {
    fn new()->Self {
        let precision=if crate::root_probe::full_f32() {
            if crate::root_probe::round_f32_input(){"fp32-rounded-input"}else{"fp32-compute"}
        }else if crate::root_probe::retain_f32(){"fp32-output"}else{"native"};
        Self{env:FLAGS.into_iter().map(|k|(k,std::env::var_os(k))).collect(),precision,
            tf32:std::env::var("GLM53_TF32").as_deref()==Ok("1")}
    }
}
impl Drop for Restore {fn drop(&mut self) {
    crate::root_probe::set_precision(self.precision);crate::tp::set_tf32(self.tf32);
    for (k,v) in &self.env {if let Some(v)=v {std::env::set_var(k,v);}else {std::env::remove_var(k);}}
}}

fn mode(name:&str) {
    std::env::set_var("GLM53_FP8_SHARED",if name.starts_with("fp8"){"1"}else{"0"});
    std::env::set_var("GLM53_DENSE_LT",if name.starts_with("lt"){"1"}else{"0"});
    crate::root_probe::set_precision(match name {
        "fp32-output"|"fp8-fp32-output"|"lt-fp32-output"=>"fp32-output",
        "fp32-compute"=>"fp32-compute","fp32-rounded-input"=>"fp32-rounded-input",_=>"native",
    });
}
fn reuse(on:bool) {std::env::set_var("GLM53_MOE_INPUT_HALF_REUSE",if on{"1"}else{"0"});}

fn exact(a:&Tensor,b:&Tensor,label:&str)->bool {
    assert_eq!(a.kind(),b.kind(),"{label}");assert_eq!(a.size(),b.size(),"{label}");
    let kind=if a.kind()==Kind::Half {Kind::Int16}else{Kind::Int};
    let same=a.contiguous().view_dtype(kind).eq_tensor(&b.contiguous().view_dtype(kind));
    assert!(same.logical_or(&a.isnan().logical_and(&b.isnan())).all().int64_value(&[])!=0,
        "shared input changed finite/signed-zero/Inf bits or NaN mask: {label}");
    same.all().int64_value(&[])!=0
}

fn pattern(rows:i64,cols:i64,turn:usize)->Tensor {
    let edge=[0.0f32,-0.,f32::from_bits(1),-f32::from_bits(1),1.00048828125,
        -1.00048828125,65504.,-65504.,0.00006103515625,-0.00006103515625];
    let values:Vec<f32>=(0..rows*cols).map(|i| {
        let j=i as usize+turn*101;
        match turn {
            1=>if j%113==0 {f32::from_bits(0x7fc12345)}else{((j*17%997) as f32-498.)/997.},
            3=>edge[j%edge.len()],4=>0.,
            _=>((j*37%997) as f32-498.)/997.,
        }
    }).collect();
    Tensor::from_slice(&values).view([rows,cols]).to_device(Device::Cuda(0))
}

struct Outputs {routed:Tensor,gate:Tensor,up:Tensor,activation:Tensor}
impl Outputs {
    fn tensors(&self)->[&Tensor;4] {[&self.routed,&self.gate,&self.up,&self.activation]}
    fn compare(&self,other:&Self,label:&str)->bool {
        let mut bits=true;for (a,b) in self.tensors().into_iter().zip(other.tensors()) {bits&=exact(a,b,label);}bits
    }
    fn poison(&self) {for t in self.tensors(){let _=t.shallow_clone().fill_(f64::NAN);}}
}

// The same producer ordering as forward::tree: canonical Half is based on x,
// while deep replay can replace only the routed view. Original x is always the
// dispatch/diagnostic argument to both shared projections.
fn pair(x:&Tensor,wg:&Tensor,wu:&Tensor)->Outputs {
    let half=crate::moe::input_half_reuse_enabled().then(||x.to_kind(Kind::Half));
    let routed=crate::deep_probe::expert_input(half.as_ref().unwrap_or(x));
    let gate=crate::weights::mm16_with_half(x,wg,half.as_ref());
    let up=crate::weights::mm16_with_half(x,wu,half.as_ref());
    let activation=gate.clamp(f64::NEG_INFINITY,crate::moe::SWIGLU_LIMIT).silu()
        *up.clamp(-crate::moe::SWIGLU_LIMIT,crate::moe::SWIGLU_LIMIT);
    Outputs{routed,gate,up,activation}
}

fn evidence(x:&Tensor,w:&Tensor,name:&str)->Value {
    let partial=crate::root_probe::retain_f32();
    let fp8=crate::dense_fp8::try_run(x,w,partial);
    let lt=crate::dense_lt::eligible(x,w,partial);
    let actual=crate::weights::mm16(x,w);
    let backend=if crate::root_probe::full_f32() {
        assert!(fp8.is_none()&&lt.is_none());"full-f32"
    }else if name.starts_with("fp8") {
        let y=fp8.as_ref().expect("FP8 proof unexpectedly fell back");exact(&actual,y,"FP8 dispatch proof");"registered-shared-fp8"
    }else if name.starts_with("lt") {
        assert!(fp8.is_none());let algorithm=lt.expect("Lt table did not cover this exact shape/partial mode");
        let y=crate::dense_lt::run(x,w,partial,algorithm);exact(&actual,&y,"Lt dispatch proof");"qualified-lt"
    }else {
        assert!(fp8.is_none()&&lt.is_none());
        assert!(!crate::gemv::eligible(x,w)&&!crate::gemv::small_eligible(x,w));
        if w.kind()!=Kind::Half {"native-f32"}else if partial {"native-half-f32-output"}else {"native-half"}
    };
    json!({"backend":backend,"weight_kind":format!("{:?}",w.kind()),"partial":partial,"lt_algorithm":lt,
        "fp8_registered_and_hit":fp8.is_some(),"dispatch_input_kind":format!("{:?}",x.kind())})
}

fn diagnostics(x:&Tensor,wg:&Tensor,wu:&Tensor)->Value {
    let mut all_bits=true;
    for on in [false,true] {
        reuse(on);let (y,records)=crate::root_probe::capture_once(||pair(x,wg,wu));
        assert_eq!(records.len(),2);
        for ((ptr,record_x,record_y),(w,output)) in records.iter().zip([(wg,&y.gate),(wu,&y.up)]) {
            assert_eq!(*ptr,w.data_ptr() as usize);assert_eq!(record_x.kind(),Kind::Float);
            all_bits&=exact(record_x,x,"root probe must retain original Float input");
            all_bits&=exact(record_y,output,"root probe output");
        }
    }
    assert!(x.ne_tensor(&x.to_kind(Kind::Half).to_kind(Kind::Float)).any().int64_value(&[])!=0,
        "diagnostic input must expose accidental pre-rounding");
    json!({"original_float_record_bits_exact":all_bits,"records_per_arm":2,"rounding_sensitive_input":true})
}

fn check_case(wg:&Tensor,wu:&Tensor,rank_slice:i64,rows:i64,layout:usize,name:&str)->Value {
    mode(name);let cols=wg.size()[1];
    let base=Tensor::empty([rows,if layout==0 {cols}else{cols*2+3}],(Kind::Float,wg.device()));
    let mut x=if layout==0 {base.shallow_clone()}else{base.slice(1,1,cols*2+1,2)};
    let _=base.shallow_clone().fill_(f64::NAN);x.copy_(&pattern(rows,cols,0));
    let backend=evidence(&x,wg,name);let up_backend=evidence(&x,wu,name);
    let recorded=diagnostics(&x,wg,wu);
    reuse(false);let old=pair(&x,wg,wu);reuse(true);let new=pair(&x,wg,wu);
    let mut payload_exact=new.compare(&old,"eager enabled versus old");
    let canonical=x.to_kind(Kind::Half);let canonical_before=canonical.copy();
    let _=crate::weights::mm16_with_half(&x,wg,Some(&canonical));
    payload_exact&=exact(&canonical,&canonical_before,"shared input is read-only");
    let mut wrong=canonical.copy();let _=wrong.fill_(0.);
    let actual_wrong=crate::weights::mm16_with_half(&x,wg,Some(&wrong));
    if name=="native" {assert!(actual_wrong.ne_tensor(&old.gate).any().int64_value(&[])!=0,"native cache fixture insensitive");}
    else {payload_exact&=exact(&actual_wrong,&old.gate,"fallback must ignore the optional Half operand");}
    // A clearly different replay input must reach only routed, never shared.
    let override_input=(&x*-0.375+0.125).to_kind(Kind::Half).contiguous();
    let replay=crate::deep_probe::with_expert_input(&override_input,||pair(&x,wg,wu));
    payload_exact&=exact(&replay.routed,&override_input,"routed override");
    for (a,b) in [(&replay.gate,&new.gate),(&replay.up,&new.up),(&replay.activation,&new.activation)] {
        payload_exact&=exact(a,b,"shared must ignore routed override");
    }
    let plain=pair(&x,wg,wu);payload_exact&=plain.compare(&new,"scoped replay cleared");
    // Warm all matmul plans before capture; preserve both graph owners and
    // outputs, then alternate replays with changing inputs and NaN output fill.
    tch::Cuda::synchronize(0);reuse(false);
    crate::tp::graph::begin().unwrap();let graph_old_y=pair(&x,wg,wu);crate::tp::graph::end().unwrap();
    let graph_old=crate::tp::graph::Owned::take();reuse(true);
    crate::tp::graph::begin().unwrap();let graph_new_y=pair(&x,wg,wu);crate::tp::graph::end().unwrap();
    let graph_new=crate::tp::graph::Owned::take();
    for turn in 0..5 {
        x.copy_(&pattern(rows,cols,turn));let before=base.copy();
        graph_old_y.poison();graph_new_y.poison();
        if turn%2==0 {graph_new.replay();graph_old.replay();}else {graph_old.replay();graph_new.replay();}
        reuse(false);let eager=pair(&x,wg,wu);
        payload_exact&=graph_new_y.compare(&graph_old_y,"changed graph enabled versus old");
        payload_exact&=graph_new_y.compare(&eager,"changed graph versus eager old");
        payload_exact&=exact(&base,&before,"input or padding modified");
        if matches!(turn,0|2|4) {for y in graph_new_y.tensors() {assert!(y.isfinite().all().int64_value(&[])!=0,"stale NaN output/padding consumed");}}
    }
    tch::Cuda::synchronize(0);drop(graph_new);drop(graph_old);
    json!({"rank_slice":rank_slice,"rows":rows,"weight_shape":wg.size(),"mode":name,
        "layout":(["contiguous","strided_offset_nan_padding"][layout]),"gate_backend":backend,"up_backend":up_backend,
        "diagnostic":recorded,"graph_replays":5,"exact_finite_zero_inf_bits":true,"nan_masks_exact":true,
        "all_nan_payload_bits_exact":payload_exact,"routed_override_isolated":true,"shared_half_readonly":true,
        "fallback_ignores_optional_half":name!="native","input_padding_unchanged":true,
        "nan_output_canary":true,"alternating_graph_owners":true})
}

fn write_report(out:&Path,cases:&[Value],gate:bool) {
    let value=json!({"gate":gate,"scope":"real layer3 TP2 shared gate/up slices; synthetic FP32 activations; local eager and changed-input graphs, no collective or whole-model claim",
        "tf32":false,"fp8_weight_scope":"registered GLM53_FP8_SHARED","cases":cases,
        "expected_cases":144,"lt_table_scope":"fresh dedicated CLI process, local algorithm index 0 for exact shapes and both output kinds",
        "shape_order":[32,1,8,2],"finite_bits_contract":"exact; NaN mask mandatory, payload equality separately recorded"});
    let p=out.join("shared-input.json");let tmp=out.join("shared-input.json.tmp");
    std::fs::write(&tmp,serde_json::to_string_pretty(&value).unwrap()).unwrap();std::fs::rename(tmp,p).unwrap();
}

fn timing(gate:&Tensor,up:&Tensor,rows:i64)->Value {
    mode("native");let sets=16;
    // 16 independent real-weight copies alternate both TP2 slices: 256 MiB
    // of GU operands. This intentionally exceeds L2, unlike one hot GU pair.
    // It is an address-rotation test, not 16 distinct model layers.
    let fixtures:Vec<_>=(0..sets).map(|i| {
        let slice=(i%2) as i64;
        let g=gate.narrow(0,slice*1024,1024).contiguous().to_device(Device::Cuda(0));
        let u=up.narrow(0,slice*1024,1024).contiguous().to_device(Device::Cuda(0));
        let x=pattern(rows,4096,i*2);(x,g,u)
    }).collect();
    let references:Vec<_>=fixtures.iter().map(|(x,g,u)| {reuse(false);let old=pair(x,g,u);
        reuse(true);assert!(pair(x,g,u).compare(&old,"rotating real-weight timing precheck"));old}).collect();
    let mut rounds=Vec::new();
    for block in 0..2 {for (arm_index,on) in [false,true,true,false].into_iter().enumerate() {
        reuse(on);
        for (x,g,u) in &fixtures {let _=pair(x,g,u);}tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();
        let outputs:Vec<_>=fixtures.iter().map(|(x,g,u)|pair(x,g,u)).collect();
        crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
        graph.replay();
        for (a,b) in outputs.iter().zip(&references) {assert!(a.compare(b,"timing graph matches old bits"));}
        for _ in 0..3 {graph.replay();}tch::Cuda::synchronize(0);
        let mut samples=Vec::new();for _ in 0..5 {
            let started=Instant::now();for _ in 0..4 {graph.replay();}tch::Cuda::synchronize(0);
            samples.push(started.elapsed().as_secs_f64()*1e6/(4*sets) as f64);
        }
        rounds.push(json!({"abba_block":block,"arm_index":arm_index,"arm":if on{"B"}else{"A"},
            "half_reuse":on,"us_per_layer":samples,"replays_per_sample":4,"sets_per_replay":sets}));
        drop(graph);drop(outputs);
    }}
    json!({"rows":rows,"sets":sets,"weight_copies":sets*2,"distinct_source_layers":1,"rank_slices":[0,1],
        "weight_workset_bytes":sets*2*1024*4096*2,"input_bytes":sets*rows as usize*4096*4,
        "source_expected_input_casts_per_layer":{"A":3,"B":1},"rounds":rounds,
        "scope":"local native Half shared gate/up plus unchanged clamp-SwiGLU and routed Half producer; rotating 256 MiB real GU weight addresses; no experts/down/TP collective; all raw samples retained",
        "numeric_bits_checked_before_timing":true})
}

/// CLI: shared-input-probe MODEL OUT. Standalone local GPU; world=1 suffices.
/// Both TP2 rank slices are checked locally. Use a fresh process because the
/// production Lt table is OnceLock and FP8 registered weights have process life.
pub fn run(model:&Path,out:&Path) {
    let _restore=Restore::new();let _guard=tch::no_grad_guard();tch::set_num_threads(4);
    std::fs::create_dir_all(out).unwrap();let mut cases=Vec::new();write_report(out,&cases,false);
    std::fs::write(out.join("shared-input-timing.json"),"{\"complete\":false,\"numeric_gate\":false,\"cases\":[]}\n").unwrap();
    std::env::set_var("GLM53_TF32","0");crate::tp::init_from_env();crate::tp::set_fp32_accum();
    for key in ["GLM53_DENSE_GEMV","GLM53_DENSE_SMALL","GLM53_FP8_LARGE"] {std::env::set_var(key,"0");}
    std::env::set_var("GLM53_FP8_WMMA","3");std::env::set_var("GLM53_FP8_SPLITS","1");
    let idx=crate::safetensors::ShardIndex::scan(model).unwrap();
    let load=|suffix:&str| {
        let (v,shape)=idx.get_f32(&format!("model.language_model.layers.3.mlp.shared_experts.{suffix}_proj.weight")).unwrap();
        assert_eq!(shape.len(),2);Tensor::from_slice(&v).view([shape[0] as i64,shape[1] as i64]).to_kind(Kind::Half)
    };
    let gate=load("gate");let up=load("up");assert_eq!(gate.size(),up.size());assert_eq!(gate.size(),[2048,4096]);
    let mut table=serde_json::Map::new();
    for rows in [1,2,8,32] {for partial in [0,1] {table.insert(format!("{rows},1024,4096,{partial}"),json!(0));}}
    let table_path=out.join("lt-table-local.json");std::fs::write(&table_path,serde_json::to_string_pretty(&table).unwrap()).unwrap();
    std::env::set_var("GLM53_DENSE_LT_TABLE",&table_path);
    std::env::remove_var("GLM53_MOE_INPUT_HALF_REUSE");assert!(!crate::moe::input_half_reuse_enabled(),"reuse default must be off");
    for rank_slice in 0..2i64 {
        let wg=gate.narrow(0,rank_slice*1024,1024).contiguous().to_device(Device::Cuda(0));
        let wu=up.narrow(0,rank_slice*1024,1024).contiguous().to_device(Device::Cuda(0));
        crate::dense_fp8::register_weight(&wg,"GLM53_FP8_SHARED");crate::dense_fp8::register_weight(&wu,"GLM53_FP8_SHARED");
        let fg=wg.to_kind(Kind::Float);let fu=wu.to_kind(Kind::Float);
        for rows in [32,1,8,2] {for layout in 0..2 {for name in ["native","fp8","lt","non-half",
            "fp32-output","fp32-compute","fp32-rounded-input","fp8-fp32-output","lt-fp32-output"] {
            let (g,u)=if name=="non-half" {(&fg,&fu)}else{(&wg,&wu)};
            cases.push(check_case(g,u,rank_slice,rows,layout,name));write_report(out,&cases,false);
            eprintln!("[shared-input] slice={rank_slice} rows={rows} layout={layout} mode={name} passed");
        }}}
    }
    assert_eq!(cases.len(),144);write_report(out,&cases,true);
    let mut timed=Vec::new();
    for rows in [1,8,32] {
        timed.push(timing(&gate,&up,rows));
        std::fs::write(out.join("shared-input-timing.json"),serde_json::to_string_pretty(&json!({
            "complete":timed.len()==3,"numeric_gate":true,"cases":timed})).unwrap()).unwrap();
    }
}
