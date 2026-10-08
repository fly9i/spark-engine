//! Fixed-chain KDA correction records. No full per-node H is materialized by
//! the producer; selected state is reconstructed with the original FP32 RN
//! operations into a newly owned slab. This module never changes DecodeStates.
use std::{path::Path, ffi::c_void};
use tch::{Tensor, Kind, Device};
use serde_json::json;

pub(crate) fn enabled() -> bool {
    std::env::var("GLM53_KDA_CORRECTION_REPLAY").as_deref() == Ok("1")
}

/// Full histories or immutable existing inputs. Both variants materialize an
/// independent selected state, never a writable alias of graph/input buffers.
pub(crate) enum ConvRecord {
    Full(Tensor),
    Deferred { base:Tensor, projected:Tensor },
}

impl ConvRecord {
    fn validate(&self,tokens:i64,heads:i64,dev:Device) {
        let check=|t:&Tensor| {
            assert_eq!(t.kind(),Kind::Float);assert!(t.is_contiguous());assert_eq!(t.device(),dev);
        };
        match self {
            Self::Full(states)=>{assert_eq!(states.size(),[tokens,3,3*heads*128]);check(states);},
            Self::Deferred{base,projected}=>{
                assert_eq!(base.size(),[3,3*heads*128]);assert_eq!(projected.size(),[tokens,3*heads*128]);
                check(base);check(projected);
            },
        }
    }

    fn shallow_clone(&self)->Self {
        match self {
            Self::Full(s)=>Self::Full(s.shallow_clone()),
            Self::Deferred{base,projected}=>Self::Deferred{base:base.shallow_clone(),projected:projected.shallow_clone()},
        }
    }

    fn materialize(&self,node:usize)->Tensor {
        assert!(node<16);
        let node=node as i64;
        match self {
            Self::Full(states)=>{assert!(node<states.size()[0]);states.get(node).copy()},
            Self::Deferred{base,projected}=>{
                assert!(node<projected.size()[0]);
                // Virtual concat(base[3],projected[T])[node+1 .. node+4].
                // A copy remains required even for one contiguous range:
                // selected state must survive producer replay and mutation.
                if node>=2 {return projected.narrow(0,node-2,3).copy();}
                let from_base=2-node;
                let result=Tensor::empty_like(base);
                result.narrow(0,0,from_base).copy_(&base.narrow(0,node+1,from_base));
                result.narrow(0,from_base,node+1).copy_(&projected.narrow(0,0,node+1));
                result
            },
        }
    }
}

/// All tensors are owned handles. base_h/k/decay and deferred conv inputs are
/// immutable aliases; correction/full histories belong to this run or graph.
/// Nothing in this record may be overwritten before selected commit.
pub(crate) struct ChainRecord {
    pub(crate) base_h: Tensor,
    pub(crate) k: Tensor,
    pub(crate) decay: Tensor,
    pub(crate) correction: Tensor,
    pub(crate) conv: ConvRecord,
}

impl ChainRecord {
    pub(crate) fn tokens(&self) -> usize { self.k.size()[0] as usize }

    fn validate(&self) -> (i64, i64) {
        assert_eq!(self.base_h.dim(), 3);
        let heads = self.base_h.size()[0];
        assert!((1..=1024).contains(&heads));
        assert_eq!(self.base_h.size(), [heads, 128, 128]);
        assert_eq!(self.k.dim(), 3);
        let tokens = self.k.size()[0];
        assert!((1..=16).contains(&tokens));
        for value in [&self.k, &self.decay, &self.correction] {
            assert_eq!(value.size(), [tokens, heads, 128]);
        }
        let dev = self.base_h.device();
        assert!(dev.is_cuda());
        self.conv.validate(tokens,heads,dev);
        for value in [&self.base_h, &self.k, &self.decay, &self.correction] {
            assert_eq!(value.kind(), Kind::Float);
            assert!(value.is_contiguous());
            assert_eq!(value.device(), dev);
        }
        (heads, tokens)
    }
}

/// Keep the same 128-thread reduction tree as kda::chain_recurrent. Inputs are
/// contiguous [T,H,128], except beta [T,H], base [H,128,128]. Return (O,u).
pub(crate) fn produce(base: &Tensor, q: &Tensor, k: &Tensor, v: &Tensor,
    beta: &Tensor, decay: &Tensor) -> (Tensor, Tensor) {
    assert_eq!(base.dim(), 3); assert_eq!(q.dim(), 3);
    let heads = base.size()[0]; let tokens = q.size()[0];
    assert!((1..=1024).contains(&heads) && (1..=16).contains(&tokens));
    assert_eq!(base.size(), [heads, 128, 128]);
    for value in [q, k, v, decay] { assert_eq!(value.size(), [tokens, heads, 128]); }
    assert_eq!(beta.size(), [tokens, heads]);
    assert!(base.device().is_cuda());
    for value in [base, q, k, v, beta, decay] {
        assert_eq!(value.kind(), Kind::Float); assert!(value.is_contiguous());
        assert_eq!(value.device(), base.device());
    }
    let out = Tensor::empty_like(v); let correction = Tensor::empty_like(v);
    extern "C" {
        fn rs_kda_correction_chain(base: *const f32, q: *const f32, k: *const f32,
            v: *const f32, beta: *const f32, decay: *const f32, correction: *mut f32,
            out: *mut f32, heads: i32, tokens: i32) -> i32;
    }
    assert_eq!(unsafe { rs_kda_correction_chain(base.data_ptr().cast(), q.data_ptr().cast(),
        k.data_ptr().cast(), v.data_ptr().cast(), beta.data_ptr().cast(), decay.data_ptr().cast(),
        correction.data_ptr().cast(), out.data_ptr().cast(), heads as i32, tokens as i32) }, 0);
    (out, correction)
}

/// The CUDA wrapper copies these HOST descriptors by value into one launch's
/// bounded parameter block. It must not retain this pointer after returning.
#[repr(C)]
struct ReplayEntry {
    base: *const f32,
    k: *const f32,
    decay: *const f32,
    correction: *const f32,
    dst_offset: i64,
    heads: i32,
    tokens: i32,
}
const MAX_LAYERS: usize = 64;
const _: () = assert!(std::mem::size_of::<ReplayEntry>() == 48);

fn descriptors(records: &[&ChainRecord], steps: usize) -> (Vec<ReplayEntry>, i64) {
    assert!(!records.is_empty() && records.len() <= MAX_LAYERS);
    assert!(steps <= 16);
    let device = records[0].base_h.device();
    let mut offset = 0i64;
    let entries = records.iter().map(|record| {
        let (heads, tokens) = record.validate();
        assert_eq!(record.base_h.device(), device);
        assert!(steps <= tokens as usize, "KDA commit exceeds valid record length");
        let entry = ReplayEntry { base: record.base_h.data_ptr().cast(), k: record.k.data_ptr().cast(),
            decay: record.decay.data_ptr().cast(), correction: record.correction.data_ptr().cast(),
            dst_offset: offset, heads: heads as i32, tokens: tokens as i32 };
        offset = offset.checked_add(heads * 128 * 128).expect("KDA slab extent overflow");
        entry
    }).collect();
    (entries, offset)
}

fn commit_into(entries: &[ReplayEntry], steps: usize, dst: &Tensor) {
    assert_eq!(dst.kind(), Kind::Float); assert!(dst.is_contiguous() && dst.device().is_cuda());
    extern "C" {
        fn rs_kda_correction_commit(entries: *const c_void, layers: i32, steps: i32,
            dst: *mut f32, elements: i64) -> i32;
    }
    assert_eq!(unsafe { rs_kda_correction_commit(entries.as_ptr().cast(), entries.len() as i32,
        steps as i32, dst.data_ptr().cast(), dst.numel() as i64) }, 0);
}

/// Exactly replay `steps` inputs (including an anchor when present). Every
/// returned H owns a disjoint view of a NEW slab. steps=0 copies base without
/// reading any u/k/decay content; it is useful for the explicit zero-depth gate.
pub(crate) fn commit_h(records: &[&ChainRecord], steps: usize) -> Vec<Tensor> {
    assert!(steps <= 16);
    if records.is_empty() { return Vec::new(); }
    let (entries, elements) = descriptors(records, steps);
    let slab = Tensor::empty([elements], (Kind::Float, records[0].base_h.device()));
    commit_into(&entries, steps, &slab);
    entries.iter().map(|entry| slab.narrow(0, entry.dst_offset, entry.heads as i64 * 128 * 128)
        .view([entry.heads as i64, 128, 128])).collect()
}

/// W07: replay `steps` inputs directly into each record's base H (the verifier graph's
/// base), and write the selected conv window into `conv_dst`. Same arithmetic as commit_h.
pub(crate) fn commit_inplace(records:&[&ChainRecord],node:usize,conv_dst:&[&Tensor]) {
    assert!(node<16&&!records.is_empty()&&records.len()==conv_dst.len());
    let (entries,_)=descriptors(records,node+1);
    extern "C"{fn rs_kda_correction_commit_inplace(entries:*const c_void,layers:i32,steps:i32)->i32;}
    assert_eq!(unsafe{rs_kda_correction_commit_inplace(entries.as_ptr().cast(),entries.len() as i32,(node+1) as i32)},0,"KDA in-place commit");
    if conv_commit_one(records,node,conv_dst) {return;}
    // The new window may read the old base rows; materialize first, then copy.
    for (record,dst) in records.iter().zip(conv_dst){let c=record.conv.materialize(node);dst.shallow_clone().copy_(&c);}
}

/// GLM53_KDA_CONV_COMMIT_ONE=1: every layer's selected convolution window in one launch (kda_conv_commit_many, pure
/// copies) instead of ~3 ATen copies per layer, which the host issued one by one (w71: ~108 copies, ~0.43 ms of GPU idle
/// per round). Deferred records only; false leaves everything to the per-layer path.
fn conv_commit_one(records:&[&ChainRecord],node:usize,conv_dst:&[&Tensor])->bool {
    if std::env::var("GLM53_KDA_CONV_COMMIT_ONE").as_deref()!=Ok("1") || records.len()>64 {return false;}
    #[repr(C)] struct Entry {base:*const f32,projected:*const f32,dst:*mut f32}
    let mut entries=Vec::with_capacity(records.len());let mut width=-1i64;
    for (record,dst) in records.iter().zip(conv_dst) {
        let ConvRecord::Deferred{base,projected}=&record.conv else {return false};
        let w=base.size()[1];
        if width<0 {width=w;}
        if w!=width || base.size()!=[3,w] || projected.dim()!=2 || projected.size()[1]!=w || (node as i64)>=projected.size()[0] || dst.size()!=[3,w]
            || [base,projected,*dst].iter().any(|t|!t.is_contiguous()||t.kind()!=Kind::Float||t.device()!=base.device()) {return false;}
        entries.push(Entry{base:base.data_ptr() as *const f32,projected:projected.data_ptr() as *const f32,dst:dst.data_ptr() as *mut f32});
    }
    extern "C"{fn rs_kda_conv_commit_many(entries:*const c_void,layers:i32,node:i32,width:i32)->i32;}
    assert_eq!(unsafe{rs_kda_conv_commit_many(entries.as_ptr().cast(),entries.len() as i32,node as i32,width as i32)},0,"KDA conv commit");
    true
}

/// Called only for Some(selected_node). Generic zero accepted nodes must keep
/// using snapshot(base); zero accepted DRAFTS with an anchor means node==0.
pub(crate) fn materialize(records: &[&ChainRecord], node: usize) -> Vec<crate::kda::KdaState> {
    assert!(node < 16);
    assert!(records.iter().all(|record| node < record.tokens()));
    commit_h(records, node + 1).into_iter().zip(records).map(|(h, record)| {
        crate::kda::KdaState { h, conv: record.conv.materialize(node) }
    }).collect()
}

fn bitexact(actual: &Tensor, expected: &Tensor, label: &str) {
    assert_eq!(actual.kind(), Kind::Float, "{label}");
    assert_eq!(actual.size(), expected.size(), "{label}");
    assert!(actual.isfinite().all().int64_value(&[]) != 0, "{label}: nonfinite actual");
    assert!(expected.isfinite().all().int64_value(&[]) != 0, "{label}: nonfinite reference");
    assert!(actual.view_dtype(Kind::Int).equal(&expected.view_dtype(Kind::Int)), "{label}: FP32 bits differ");
}

struct Inputs { base: Tensor, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, decay: Tensor }
fn inputs(heads: i64, tokens: i64, dev: Device) -> Inputs {
    Inputs { base: Tensor::randn([heads,128,128], (Kind::Float,dev)) * 0.1,
        q: Tensor::randn([tokens,heads,128], (Kind::Float,dev)) * 0.01,
        k: Tensor::randn([tokens,heads,128], (Kind::Float,dev)) * 0.01,
        v: Tensor::randn([tokens,heads,128], (Kind::Float,dev)),
        beta: Tensor::rand([tokens,heads], (Kind::Float,dev)),
        decay: Tensor::rand([tokens,heads,128], (Kind::Float,dev)) }
}
fn run_producer(input: &Inputs) -> (Tensor, ChainRecord) {
    let (out, correction) = produce(&input.base, &input.q, &input.k, &input.v, &input.beta, &input.decay);
    let (tokens,heads) = (input.q.size()[0],input.base.size()[0]);
    let conv_states = Tensor::arange(tokens*3*3*heads*128,(Kind::Float,input.base.device()))
        .view([tokens,3,3*heads*128]);
    (out, ChainRecord { base_h: input.base.shallow_clone(), k: input.k.shallow_clone(),
        decay: input.decay.shallow_clone(), correction, conv:ConvRecord::Full(conv_states) })
}
fn reference(input: &Inputs) -> (Tensor, Tensor) {
    crate::kda::chain_recurrent(&input.base,&input.q,&input.k,&input.v,&input.beta,&input.decay)
}
fn independent_steps(input: &Inputs) -> (Tensor, Tensor) {
    let mut h = input.base.copy(); let mut outputs = Vec::new(); let mut states = Vec::new();
    for t in 0..input.q.size()[0] {
        outputs.push(crate::kda::recurrent(&mut h,&input.q.get(t),&input.k.get(t),&input.v.get(t),
            &input.beta.get(t),&input.decay.get(t),true));
        states.push(h.copy());
    }
    (Tensor::stack(&outputs,0),Tensor::stack(&states,0))
}
fn poisoned_suffix(record: &ChainRecord, steps: usize) -> ChainRecord {
    let poison = |x: &Tensor| {
        let out = x.copy(); let count = x.size()[0] - steps as i64;
        if count > 0 { let _ = out.narrow(0,steps as i64,count).fill_(f64::NAN); }
        out
    };
    ChainRecord { base_h: record.base_h.shallow_clone(), k: poison(&record.k), decay: poison(&record.decay),
        correction: poison(&record.correction), conv: record.conv.shallow_clone() }
}

fn conv_bits(actual:&Tensor,expected:&Tensor,label:&str) {
    assert_eq!(actual.kind(),Kind::Float);assert_eq!(actual.size(),expected.size());
    assert!(actual.contiguous().view_dtype(Kind::Int).equal(&expected.contiguous().view_dtype(Kind::Int)),
        "{label}: conv bit patterns differ");
}

/// Independent state oracle uses an explicit virtual-history concatenation,
/// only in this diagnostic. Production constructs no concatenation or dtype
/// conversion and never exposes a mutable slice of the saved raw projections.
fn conv_deferred_probe(dev:Device)->Vec<serde_json::Value> {
    let mut cases=Vec::new();
    for heads in [1i64,32,64] {for tokens in 1i64..=8 {
        let width=3*heads*128;
        let original_base=Tensor::randn([3,width],(Kind::Float,dev));
        let original=Tensor::randn([tokens,width],(Kind::Float,dev));
        let wall=Tensor::randn([width,4],(Kind::Float,dev))*0.1;
        let mut base=original_base.copy();
        let mut storage=Tensor::full([tokens+3,width],f64::NAN,(Kind::Float,dev));
        let mut projected=storage.narrow(0,0,tokens);
        for scale in [1.,-1.,0.,100.] {
            base.copy_(&(&original_base*scale));let _=storage.fill_(f64::NAN);projected.copy_(&(&original*scale));
            let (expected_act,full)=crate::kda::chain_convolution(&base,&projected,&wall);
            let (act,record)=crate::kda::chain_convolution_deferred(&base,&projected,&wall);
            record.validate(tokens,heads,dev);assert!(matches!(&record,ConvRecord::Deferred{..}));
            bitexact(&act,&expected_act,"conv activation-only versus full stores");
            let history=Tensor::cat(&[&base,&projected],0);
            for node in 0..tokens {
                let expected=history.narrow(0,node+1,3);
                conv_bits(&full.get(node),&expected,"old conv history oracle");
                let selected=record.materialize(node as usize);
                conv_bits(&selected,&expected,"deferred selected conv history");
                // Neither future projected rows nor expired base rows may be
                // read by a selected state, even if their values are NaN.
                let poisoned_base=base.copy();let poisoned_projected=projected.copy();
                let _=poisoned_base.narrow(0,0,(node+1).min(3)).fill_(f64::NAN);
                if node+1<tokens {let _=poisoned_projected.narrow(0,node+1,tokens-node-1).fill_(f64::NAN);}
                let poisoned=ConvRecord::Deferred{base:poisoned_base,projected:poisoned_projected};
                conv_bits(&poisoned.materialize(node as usize),&expected,"expired base / future projected NaN");
                // Copies own storage, including the contiguous projected case.
                let second=record.materialize(node as usize);assert_ne!(selected.data_ptr(),second.data_ptr());
                let _=selected.shallow_clone().fill_(f64::NAN);
                conv_bits(&second,&expected,"independent selected conv state");
            }
            conv_bits(&base,&(&original_base*scale),"conv preserved base");
            conv_bits(&projected,&(&original*scale),"conv preserved projected");
        }

        // Copying raw history must preserve +/-0, subnormals and NaN payloads.
        // Nonfinite values are deliberately not passed to finite activation checks.
        let patterns=[0u32,0x80000000,1,0x80000001,0x7fc01234,0xffc05678,0x7f800000,0xff800000];
        let bits:Vec<i32>=patterns.iter().map(|&x|x as i32).collect();
        let raw=Tensor::from_slice(&bits).to_device(dev).view_dtype(Kind::Float)
            .repeat([(tokens+3)*width/8]).view([tokens+3,width]);
        let raw_record=ConvRecord::Deferred{base:raw.narrow(0,0,3),projected:raw.narrow(0,3,tokens)};
        for node in 0..tokens {conv_bits(&raw_record.materialize(node as usize),&raw.narrow(0,node+1,3),"raw history payload copy");}

        base.copy_(&original_base);projected.copy_(&original);
        let _=crate::kda::chain_convolution_deferred(&base,&projected,&wall);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();
        let (captured,record)=crate::kda::chain_convolution_deferred(&base,&projected,&wall);
        crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
        let _=record.materialize(0);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let captured_selected=record.materialize(0);
        crate::tp::graph::end().unwrap();let select_graph=crate::tp::graph::Owned::take();
        let mut saved:Option<(Tensor,Tensor)>=None;
        for scale in [1.,-1.,0.,100.] {
            base.copy_(&(&original_base*scale));let _=storage.fill_(f64::NAN);projected.copy_(&(&original*scale));
            let _=captured.shallow_clone().fill_(f64::NAN);
            graph.replay();select_graph.replay();
            let (act,full)=crate::kda::chain_convolution(&base,&projected,&wall);
            bitexact(&captured,&act,"changed-input activation-only graph");
            conv_bits(&captured_selected,&full.get(0),"captured selected conv / changed base");
            for node in (0..tokens).rev() {conv_bits(&record.materialize(node as usize),&full.get(node),"graph long then short selection");}
            if let Some((old,expected))=&saved {conv_bits(old,expected,"old selected conv survives replay");}
            saved=Some((record.materialize((tokens-1) as usize),full.get(tokens-1).copy()));
        }
        // Exercise the real H+conv materializer, not only the enum helper.
        let input=inputs(heads,tokens,dev);let (_,mut combined)=run_producer(&input);
        combined.conv=record.shallow_clone();let full_h=reference(&input).1;
        let history=Tensor::cat(&[&base,&projected],0);
        for node in 0..tokens {
            let state=materialize(&[&combined],node as usize);
            bitexact(&state[0].h,&full_h.get(node),"combined H/conv record");
            conv_bits(&state[0].conv,&history.narrow(0,node+1,3),"combined H/conv selected history");
        }
        drop(combined);drop(select_graph);drop(graph);
        // The record retains tensor owners after all external input handles
        // and graph owners are gone. A materialized state also survives it.
        drop(projected);drop(storage);drop(base);
        conv_bits(&record.materialize((tokens-1) as usize),&history.narrow(0,tokens,3),"record retained raw input owners");
        drop(record);
        if let Some((old,expected))=&saved {conv_bits(old,expected,"conv state survives graph/input/record drop");}
        cases.push(json!({"heads":heads,"tokens":tokens,"all_selected_nodes_bitexact":true,
            "activation_only_matches_full":true,"independent_concat_oracle":true,
            "future_and_expired_nan_safe":true,"raw_zero_subnormal_nan_payload_bits":true,
            "changed_input_graph":true,"captured_selection":true,"long_then_short_selection":true,
            "combined_h_conv_materialize":true,"selected_state_and_record_owner_isolation":true}));
    }}
    cases
}

/// Standalone local-GPU gate. Not a timing benchmark or TP2 qualification.
/// Every supported T and selected depth are checked against existing kernels;
/// the final JSON gate remains false if any finite/bitwise/state test fails.
pub fn run_local_probe(out: &Path) {
    assert!(!crate::tp::is_tp(), "KDA correction local probe requires a single-rank environment");
    tch::set_num_threads(4); let _guard=tch::no_grad_guard(); tch::manual_seed(923711);
    std::fs::create_dir_all(out).unwrap();
    let path = out.join("kda-correction-local.json");
    std::fs::write(&path,r#"{"complete":false,"gate":false,"cases":[]}"#).unwrap();
    let dev=Device::Cuda(0); let mut cases=Vec::new();
    for heads in [1i64,32,64] { for tokens in 1i64..=8 {
        let original=inputs(heads,tokens,dev);
        for (name,scale,beta_override,decay_override) in [
            ("normal",1.,None,None), ("large",1e6,None,None), ("tiny",1e-35,None,None),
            ("signed_zero",0.,None,None), ("beta_zero_decay_one",1.,Some(0.),Some(1.)),
            ("beta_one_decay_one",1.,Some(1.),Some(1.)), ("near_zero_decay",1.,Some(1.),Some(1e-30))] {
            let input=Inputs { base:&original.base*scale, q:original.q.shallow_clone(), k:original.k.shallow_clone(),
                v:&original.v*scale, beta:beta_override.map_or_else(||original.beta.shallow_clone(),|v|Tensor::full_like(&original.beta,v)),
                decay:decay_override.map_or_else(||original.decay.shallow_clone(),|v|Tensor::full_like(&original.decay,v)) };
            let base_before=input.base.copy(); let gold=reference(&input);
            let (actual,record)=run_producer(&input); bitexact(&actual,&gold.0,"producer O");
            // A separate existing per-token kernel is also an oracle, not a
            // second call to the candidate producer/replay implementation.
            if name=="normal" { let serial=independent_steps(&input);
                bitexact(&gold.0,&serial.0,"original chain versus independent O");
                bitexact(&gold.1,&serial.1,"original chain versus independent H"); }
            for steps in 0..=tokens as usize {
                let poisoned=poisoned_suffix(&record,steps);
                let result=commit_h(&[&poisoned],steps);
                let expected=if steps==0 {input.base.shallow_clone()} else {gold.1.get(steps as i64-1)};
                bitexact(&result[0],&expected,"selected H / inactive NaN suffix");
                if steps>0 { let state=materialize(&[&poisoned],steps-1);
                    bitexact(&state[0].h,&expected,"materialized H");
                    bitexact(&state[0].conv,&record.conv.materialize(steps-1),"materialized conv"); }
            }
            bitexact(&input.base,&base_before,"producer and commit preserve base");
            cases.push(json!({"heads":heads,"tokens":tokens,"scenario":name,"all_depths_0_through_t_exact":true,
                "producer_o_bitexact":true,"inactive_record_nan_safe":true,"base_unchanged":true}));
        }
        std::fs::write(&path,serde_json::to_string_pretty(&json!({"complete":false,"gate":false,"cases":cases})).unwrap()).unwrap();
    }}

    // Real TP2 state shape, 34 independently valued layer records. Also verify
    // a mixed-head pack, proving offsets are not silently hardcoded to 32.
    let mut packs=Vec::new();
    for (name,head_counts) in [("tp2_34_layers",vec![32i64;34]),("mixed_heads",vec![1,2,32,64,1])] {
        let input:Vec<_>=head_counts.iter().map(|&h|inputs(h,8,dev)).collect();
        let base_before:Vec<_>=input.iter().map(|x|x.base.copy()).collect();
        let gold:Vec<_>=input.iter().map(reference).collect();
        let records:Vec<_>=input.iter().map(|x|run_producer(x).1).collect();
        for steps in 0..=8 {
            let poisoned:Vec<_>=records.iter().map(|r|poisoned_suffix(r,steps)).collect();
            let refs:Vec<_>=poisoned.iter().collect(); let result=commit_h(&refs,steps);
            for i in 0..records.len() {
                let expected=if steps==0 {input[i].base.shallow_clone()} else {gold[i].1.get(steps as i64-1)};
                bitexact(&result[i],&expected,"packed layer H");
            }
            let (entries,elements)=descriptors(&refs,steps);
            let guarded=Tensor::full([elements+2],12345.25,(Kind::Float,dev));
            commit_into(&entries,steps,&guarded.narrow(0,1,elements));
            assert_eq!(guarded.double_value(&[0]),12345.25);
            assert_eq!(guarded.double_value(&[elements+1]),12345.25);
            for (entry,value) in entries.iter().zip(&result) {
                bitexact(&guarded.narrow(0,1+entry.dst_offset,value.numel() as i64).view(value.size().as_slice()),value,"slab offsets and canary");
            }
        }
        let refs:Vec<_>=records.iter().collect();
        let one=materialize(&refs,0); let another=materialize(&refs,0);
        let original_next=one[1].h.copy(); let original_conv=another[0].conv.copy();
        assert_ne!(one[0].h.data_ptr(),another[0].h.data_ptr());
        let _=one[0].h.shallow_clone().fill_(f64::NAN); let _=one[0].conv.shallow_clone().fill_(f64::NAN);
        bitexact(&another[0].h,&gold[0].1.get(0),"independent selected slab");
        bitexact(&one[1].h,&original_next,"disjoint layer slab ranges");
        bitexact(&another[0].conv,&original_conv,"independent selected conv");
        for (r,x) in records.iter().zip(&base_before) {bitexact(&r.base_h,x,"packed base isolation");}
        packs.push(json!({"name":name,"heads":head_counts,"layers":records.len(),"all_depths_bitexact":true,
            "slab_canary":true,"independent_selected_states":true}));
    }

    let mut graph_cases=Vec::new();
    for tokens in 1i64..=8 {
        let original=inputs(32,tokens,dev);
        let mut input=Inputs {base:original.base.copy(),q:original.q.copy(),k:original.k.copy(),v:original.v.copy(),
            beta:original.beta.copy(),decay:original.decay.copy()};
        let _=run_producer(&input); tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let (captured,record)=run_producer(&input);
        crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
        let mut saved_selected=None;
        for scale in [1.,-1.,0.,100.] {
            input.base.copy_(&(&original.base*scale)); input.q.copy_(&(&original.q*if scale<0.{-1.}else{1.}));
            input.k.copy_(&(&original.k*if scale==100.{0.5}else{1.})); input.v.copy_(&(&original.v*scale));
            input.beta.copy_(&(&original.beta*if scale==0.{0.}else{1.}));
            input.decay.copy_(&(&original.decay*if scale==100.{0.5}else{1.}));
            let _=record.correction.shallow_clone().fill_(f64::NAN); graph.replay();
            let gold=reference(&input); bitexact(&captured,&gold.0,"changed-input producer graph");
            for steps in (0..=tokens as usize).rev() {
                let result=commit_h(&[&record],steps);
                let expected=if steps==0 {input.base.shallow_clone()} else {gold.1.get(steps as i64-1)};
                bitexact(&result[0],&expected,"changed-input graph selected H");
            }
            if let Some((old,expected))=&saved_selected {bitexact(old,expected,"next graph replay preserves prior selected slab");}
            saved_selected=Some((commit_h(&[&record],tokens as usize).remove(0),gold.1.get(tokens-1).copy()));
        }
        // Commit itself can be captured at a fixed depth; dynamic selection in
        // the real verifier remains a host argument after predictions arrive.
        let _=commit_h(&[&record],1);tch::Cuda::synchronize(0);
        crate::tp::graph::begin().unwrap();let captured_h=commit_h(&[&record],1);
        crate::tp::graph::end().unwrap();let commit_graph=crate::tp::graph::Owned::take();
        input.base.copy_(&original.base);input.q.copy_(&original.q);input.k.copy_(&original.k);
        input.v.copy_(&original.v);input.beta.copy_(&original.beta);input.decay.copy_(&original.decay);
        graph.replay();commit_graph.replay();let gold=reference(&input);
        bitexact(&captured_h[0],&gold.1.get(0),"captured commit after producer replay");
        drop(commit_graph);drop(graph);drop(record);
        if let Some((old,expected))=&saved_selected {bitexact(old,expected,"graph drop preserves selected slab");}
        graph_cases.push(json!({"tokens":tokens,"heads":32,"changing_base_qkv_beta_decay":true,
            "all_selected_depths_bitexact":true,"long_then_short_selection":true,
            "captured_commit":true,"selected_state_survives_replay_and_graph_drop":true}));
    }
    let conv_cases=conv_deferred_probe(dev);
    std::fs::write(&path,serde_json::to_string_pretty(&json!({"complete":true,"gate":true,"cases":cases,
        "packs":packs,"graphs":graph_cases,"conv_deferred":conv_cases,
        "comparison":"FP32 bits; finite arithmetic required; raw conv copies additionally check NaN payload bits",
        "scope":"H1 local producer/commit kernels only; full model, verifier selection, drafter and TP2 ABBA remain separate gates"})).unwrap()).unwrap();
    eprintln!("[kda-correction] local O/H/conv bits, all depths, 34-layer slab, NaN, graph and ownership gates passed");
}
