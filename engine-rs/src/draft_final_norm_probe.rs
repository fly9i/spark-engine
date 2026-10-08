//! Real final Norm parameters, synthetic activation layouts, unchanged ATen oracle.
use tch::{Tensor,Kind,Device};
use serde_json::{json,Value};
use std::{path::{Path,PathBuf},ffi::OsString,time::Instant,cell::RefCell};
const FINAL:&str="GLM53_DRAFT_FINAL_NORM_SELECT";
const CACHE:&str="GLM53_DRAFT_NORM_CACHE";
const PATTERNS:[&str;10]=["normal","opposite_signed_zero","mixed_nonfinite","discarded_anchor_nonfinite","normal_changed",
    "same_negative_zero","finite_extreme_only","positive_inf_only","negative_inf_only","single_nan_only"];
thread_local!{static DIAGNOSTIC:RefCell<Option<(PathBuf,Value)>>=const{RefCell::new(None)};}
struct DiagnosticScope(Option<(PathBuf,Value)>);
impl DiagnosticScope{fn new(path:&Path)->Self{Self(DIAGNOSTIC.with(|d|d.replace(Some((path.to_owned(),json!({"stage":"initialization"}))))))}}
impl Drop for DiagnosticScope{fn drop(&mut self){DIAGNOSTIC.with(|d|{d.replace(self.0.take());});}}
fn diagnostic_context(value:Value){DIAGNOSTIC.with(|d|{if let Some((_,v))=d.borrow_mut().as_mut(){*v=value;}});}
fn raw(t:&Tensor)->Vec<u32>{match t.kind(){
    Kind::BFloat16=>Vec::<i16>::try_from(t.contiguous().view_dtype(Kind::Int16).flatten(0,-1).to_device(Device::Cpu)).unwrap().into_iter().map(|v|v as u16 as u32).collect(),
    Kind::Float=>Vec::<i32>::try_from(t.contiguous().view_dtype(Kind::Int).flatten(0,-1).to_device(Device::Cpu)).unwrap().into_iter().map(|v|v as u32).collect(),
    _=>panic!("unsupported diagnostic dtype"),
}}
fn raw_difference(a:&Tensor,b:&Tensor)->Value{
    let aa=raw(a);let bb=raw(b);let width=*a.size().last().unwrap() as usize;
    let decode=|v:u32|if a.kind()==Kind::BFloat16{f32::from_bits(v<<16)}else{f32::from_bits(v)};
    let different:Vec<_>=aa.iter().zip(&bb).enumerate().filter(|(_,(&x,&y))|x!=y).collect();
    let first:Vec<_>=different.iter().take(32).map(|(i,(&x,&y))|json!({"flat_index":i,"row":i/width,"column":i%width,
        "actual_raw_hex":format!("{x:08x}"),"reference_raw_hex":format!("{y:08x}"),
        "actual_value":format!("{:?}",decode(x)),"reference_value":format!("{:?}",decode(y))})).collect();
    json!({"mismatch_count":different.len(),"all_mismatches_both_nan":!different.is_empty()&&different.iter().all(|(_,(&x,&y))|decode(x).is_nan()&&decode(y).is_nan()),
        "first_differences":first,"actual_shape":a.size(),"reference_shape":b.size(),"actual_stride":a.stride(),"reference_stride":b.stride(),
        "actual_ptr":a.data_ptr() as usize,"reference_ptr":b.data_ptr() as usize,"dtype":format!("{:?}",a.kind())})
}
fn equal_bits(a:&Tensor,b:&Tensor)->bool{
    assert_eq!(a.kind(),b.kind());assert_eq!(a.size(),b.size());
    let k=match a.kind(){Kind::Float=>Kind::Int,Kind::BFloat16=>Kind::Int16,_=>panic!("unexpected dtype")};
    a.contiguous().view_dtype(k).equal(&b.contiguous().view_dtype(k))
}
struct Restore(Vec<(&'static str,Option<OsString>)>);
impl Restore{fn new()->Self{Self([FINAL,CACHE].into_iter().map(|f|(f,std::env::var_os(f))).collect())}}
impl Drop for Restore{fn drop(&mut self){for(f,v)in &self.0{match v{Some(v)=>std::env::set_var(f,v),None=>std::env::remove_var(f)}}}}
fn flag(on:bool){std::env::set_var(FINAL,if on{"1"}else{"0"});}
fn exact(a:&Tensor,b:&Tensor,label:&str){
    assert_eq!(a.kind(),b.kind(),"{label}");assert_eq!(a.size(),b.size(),"{label}");
    let same=equal_bits(a,b);
    if !same {let diff=raw_difference(a,b);DIAGNOSTIC.with(|d|{if let Some((path,context))=d.borrow().as_ref(){
        save(path,&json!({"gate":false,"label":label,"context":context,"raw_difference":diff}));
    }});eprintln!("[final-norm-raw-difference] {label}: {diff}");}
    assert!(same,"{label}: raw bits, including +/-0 and NaN payloads; see failure.json");
}
fn old<F:Fn(&Tensor)->Tensor>(x:&Tensor,r:&Tensor,normalize:&F)->Tensor{
    let sum=x.to_kind(Kind::Float)+r.to_kind(Kind::Float);
    // Materialize the old unused BF16 residual on purpose: this is the exact
    // prior producer path, not a baseline silently receiving half the candidate.
    let output=normalize(&sum);let unused_residual=sum.to_kind(Kind::BFloat16);
    let selected=output.narrow(0,1,x.size()[0]-1);drop(unused_residual);selected
}
// Retired first-L candidate retained solely for optional diagnosis; never used
// by production dispatch or treated as passing the exact gate when it differs.
fn retired_early_slice<F:Fn(&Tensor)->Tensor>(x:&Tensor,r:&Tensor,normalize:&F)->Tensor{
    let n=x.size()[0];let sum=x.narrow(0,1,n-1).to_kind(Kind::Float)+r.narrow(0,1,n-1).to_kind(Kind::Float);normalize(&sum)
}
fn fixture(rows:i64,layout:usize,kind:Kind,dev:Device)->(Tensor,Tensor){
    match layout{
        0=>{let owner=Tensor::full([rows,4096],731.25,(kind,dev));let view=owner.shallow_clone();(owner,view)},
        1=>{let owner=Tensor::full([rows+2,4096],731.25,(kind,dev));let view=owner.narrow(0,1,rows);(owner,view)},
        2=>{let owner=Tensor::full([2*rows+3,4096],731.25,(kind,dev));let view=owner.slice(0,1,1+2*rows,2);(owner,view)},
        3=>{let owner=Tensor::full([rows+2,8200],731.25,(kind,dev));let view=owner.narrow(0,1,rows).slice(1,3,8195,2);(owner,view)},
        _=>unreachable!(),
    }
}
fn changed(rows:i64,turn:usize,kind:Kind,dev:Device)->(Tensor,Tensor){
    if turn==0||turn==4{
        return(Tensor::randn([rows,4096],(Kind::Float,dev)).to_kind(kind)*0.125,
               Tensor::randn([rows,4096],(Kind::Float,dev)).to_kind(kind)*0.0625);
    }
    if turn==5{
        // No flip: both operands are -0 in every retained lane. The former
        // alternating +/-0 case alone mostly exercised opposite-sign addition.
        let raw=vec![0x80000000u32 as i32;(rows*4096) as usize];
        let a=Tensor::from_slice(&raw).view_dtype(Kind::Float).view([rows,4096]).to_kind(kind).to_device(dev);
        return(a.shallow_clone(),a.copy());
    }
    if turn==6{
        // Finite large/small values are isolated from NaN and Inf; maxima stay
        // finite through BF16 conversion, FP32 addition and the squared mean.
        let patterns=[65504f32,-65504.,1.,-1.,f32::from_bits(0x00010000),-f32::from_bits(0x00010000),0.,-0.];
        let raw:Vec<f32>=(0..rows*4096).map(|i|patterns[i as usize%patterns.len()]).collect();
        let a=Tensor::from_slice(&raw).view([rows,4096]).to_kind(kind).to_device(dev);
        let b=&a*0.25;return(a,b);
    }
    if (7..=9).contains(&turn){
        let a=Tensor::full([rows,4096],0.125,(kind,dev));let b=Tensor::full([rows,4096],-0.03125,(kind,dev));
        let value=match turn{7=>f64::INFINITY,8=>f64::NEG_INFINITY,9=>f64::NAN,_=>unreachable!()};
        let _=a.select(1,0).fill_(value);return(a,b);
    }
    let patterns=if turn==1{vec![0u32,0x80000000]}else{
        vec![0x7fc12345u32,0xffc45678,0x7f800000,0xff800000,0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff]
    };
    let raw:Vec<i32>=(0..rows*4096).map(|i|patterns[i as usize%patterns.len()] as i32).collect();
    let a=Tensor::from_slice(&raw).view_dtype(Kind::Float).view([rows,4096]).to_kind(kind).to_device(dev);
    let b=a.flip([1]);
    if turn==3{
        let a=Tensor::full([rows,4096],0.125,(kind,dev));let b=Tensor::full([rows,4096],-0.03125,(kind,dev));
        let _=a.get(0).fill_(f64::NAN);let _=b.get(0).fill_(f64::INFINITY);(a,b)
    }else{(a,b)}
}
fn assert_fixture(x:&Tensor,r:&Tensor,turn:usize){
    let n=x.size()[0];let sum=x.to_kind(Kind::Float)+r.to_kind(Kind::Float);
    let kept=sum.narrow(0,1,n-1);
    match turn{
        5=>{assert!(x.to_kind(Kind::Float).contiguous().view_dtype(Kind::Int).eq(i32::MIN as i64).all().int64_value(&[])!=0);
            assert!(r.to_kind(Kind::Float).contiguous().view_dtype(Kind::Int).eq(i32::MIN as i64).all().int64_value(&[])!=0);
            assert!(kept.view_dtype(Kind::Int).eq(i32::MIN as i64).all().int64_value(&[])!=0,"(-0)+(-0) fixture must retain -0");},
        6=>{assert!(x.isfinite().all().int64_value(&[])!=0&&r.isfinite().all().int64_value(&[])!=0);
            assert!(kept.isfinite().all().int64_value(&[])!=0&&(&kept*&kept).isfinite().all().int64_value(&[])!=0);
            assert!(kept.abs().max().double_value(&[])>=60000.);},
        7|8=>{assert_eq!(x.isinf().sum(Kind::Int64).int64_value(&[]),n);
            assert_eq!(x.isnan().sum(Kind::Int64).int64_value(&[]),0);
            assert!(r.isfinite().all().int64_value(&[])!=0);
            assert!(x.select(1,0).eq(if turn==7{f64::INFINITY}else{f64::NEG_INFINITY}).all().int64_value(&[])!=0);
            assert!(kept.narrow(1,1,4095).isfinite().all().int64_value(&[])!=0);},
        9=>{assert_eq!(x.isnan().sum(Kind::Int64).int64_value(&[]),n);
            assert_eq!(x.isinf().sum(Kind::Int64).int64_value(&[]),0);
            assert!(r.isfinite().all().int64_value(&[])!=0);
            assert!(kept.narrow(1,1,4095).isfinite().all().int64_value(&[])!=0);},
        _=>{},
    }
}
fn save(path:&Path,report:&Value){let tmp=path.with_extension("json.tmp");
    std::fs::write(&tmp,serde_json::to_string_pretty(report).unwrap()).unwrap();std::fs::rename(tmp,path).unwrap();}

/// normalize must invoke dflash::norm with the actual frozen norm.weight. It
/// must not look at row 0 or normalize across the token dimension.
pub(crate) fn norm_check<F:Fn(&Tensor)->Tensor>(normalize:F,dev:Device,out:&Path){
    let _restore=Restore::new();let _guard=tch::no_grad_guard();tch::manual_seed(2026092321);
    std::fs::create_dir_all(out).unwrap();let path=out.join("final-norm-local.json");
    let _diagnostic=DiagnosticScope::new(&out.join("failure.json"));
    let diagnose_early=std::env::var("GLM53_FINAL_NORM_DIAG_EARLY").as_deref()==Ok("1");
    let mut report=json!({"complete":false,"gate":false,"cases":[],"timing":[],
        "scope":"actual final norm.weight; synthetic activations; FP32 ATen add/mean/rsqrt/weight and BF16 boundary unchanged",
        "candidate":"full_N_original_shapes_delete_unused_residual_cast_only","retired_early_slice_diagnostic":diagnose_early,
        "layouts":["contiguous","contiguous_offset","row_stride2","offset_and_column_stride2"],"patterns":PATTERNS});save(&path,&report);
    std::env::remove_var(FINAL);assert!(!super::enabled());
    for cached in [false,true]{std::env::set_var(CACHE,if cached{"1"}else{"0"});
        for rows in 2..=8{for layout in 0..4{for kind in [Kind::BFloat16,Kind::Float]{
            let (xo,mut x)=fixture(rows,layout,kind,dev);let(ro,mut r)=fixture(rows,layout,kind,dev);
            let case_context=json!({"cached":cached,"rows":rows,"layout":layout,"dtype":format!("{kind:?}"),"stage":"capture",
                "x_stride":x.stride(),"r_stride":r.stride(),"x_ptr":x.data_ptr() as usize,"r_ptr":r.data_ptr() as usize});
            diagnostic_context(case_context.clone());report["active_case"]=case_context;report["active_turn"]=Value::Null;save(&path,&report);
            let(a,b)=changed(rows,0,kind,dev);x.copy_(&a);r.copy_(&b);
            let view=x.narrow(0,1,rows-1);
            assert_eq!(view.data_ptr() as usize,x.data_ptr() as usize+x.stride()[0] as usize*if kind==Kind::Float{4}else{2});
            flag(false);assert!(super::try_hidden(&x,&r,&normalize).is_none());
            let _=old(&x,&r,&normalize);flag(true);let _=super::try_hidden(&x,&r,&normalize).unwrap();tch::Cuda::synchronize(0);
            flag(false);crate::tp::graph::begin().unwrap();let old_out=old(&x,&r,&normalize);crate::tp::graph::end().unwrap();let ga=crate::tp::graph::Owned::take();
            flag(true);crate::tp::graph::begin().unwrap();let new_out=super::try_hidden(&x,&r,&normalize).unwrap();crate::tp::graph::end().unwrap();let gb=crate::tp::graph::Owned::take();
            let retired=if diagnose_early{
                let _=retired_early_slice(&x,&r,&normalize);tch::Cuda::synchronize(0);
                crate::tp::graph::begin().unwrap();let y=retired_early_slice(&x,&r,&normalize);crate::tp::graph::end().unwrap();
                Some((crate::tp::graph::Owned::take(),y))
            }else{None};
            let mut turns=Vec::new();
            for turn in 0..PATTERNS.len(){
                let ctx=json!({"cached":cached,"rows":rows,"layout":layout,"dtype":format!("{kind:?}"),"turn":turn,"pattern":PATTERNS[turn],
                    "x_stride":x.stride(),"r_stride":r.stride(),"x_ptr":x.data_ptr() as usize,"r_ptr":r.data_ptr() as usize});
                diagnostic_context(ctx.clone());report["active_turn"]=ctx;save(&path,&report);
                let(a,b)=changed(rows,turn,kind,dev);x.copy_(&a);r.copy_(&b);let before_x=xo.copy();let before_r=ro.copy();
                assert_fixture(&x,&r,turn);
                // Poison both independent graph outputs, then replay changed
                // inputs on the same stream. Their whole visible ranges must be written.
                let _=old_out.shallow_clone().fill_(731.25);let _=new_out.shallow_clone().fill_(-831.5);
                ga.replay();gb.replay();let reference=old(&x,&r,&normalize);
                exact(&old_out,&reference,"old graph replay");exact(&new_out,&reference,"selected graph replay");
                let fresh=super::try_hidden(&x,&r,&normalize).unwrap();exact(&fresh,&reference,"selected eager");
                exact(&xo,&before_x,"source plus guards unchanged");exact(&ro,&before_r,"residual plus guards unchanged");
                if turn==3{assert!(new_out.isfinite().all().int64_value(&[])!=0,"poisoned discarded anchor contaminated retained rows");}
                if turn==5||turn==6{assert!(new_out.isfinite().all().int64_value(&[])!=0,"finite-only boundary yielded nonfinite result");}
                if turn==7||turn==8{assert!(new_out.select(1,0).isnan().all().int64_value(&[])!=0);
                    assert!(new_out.narrow(1,1,4095).isfinite().all().int64_value(&[])!=0,"isolated Inf must not obscure all finite output columns");}
                if turn==9{assert!(new_out.isnan().all().int64_value(&[])!=0,"single NaN input should poison its own row reduction");}
                assert_ne!(fresh.data_ptr(),new_out.data_ptr(),"public output aliases captured writable output");
                let retired_result=retired.as_ref().map(|(g,y)|{
                    let _=y.shallow_clone().fill_(-731.25);g.replay();let eager=retired_early_slice(&x,&r,&normalize);
                    let graph_exact=equal_bits(y,&reference);let eager_exact=equal_bits(&eager,&reference);
                    json!({"retired_graph_matches_original":graph_exact,"retired_eager_matches_original":eager_exact,
                        "graph_difference":if graph_exact{None}else{Some(raw_difference(y,&reference))},
                        "eager_difference":if eager_exact{None}else{Some(raw_difference(&eager,&reference))}})
                });
                turns.push(json!({"turn":turn,"pattern":PATTERNS[turn],"fixture_assertions_passed":true,"eager_graph_raw_bits":true,"input_guard_bits":true,"source_offsets_preserved":true,
                    "retired_early_slice":retired_result}));
            }
            let fresh=super::try_hidden(&x,&r,&normalize).unwrap();let saved=fresh.copy();
            let _=x.fill_(0.25);let _=r.fill_(-0.125);ga.replay();gb.replay();exact(&fresh,&saved,"fresh result survives subsequent replay");
            let captured_bits=new_out.copy();
            tch::Cuda::synchronize(0);drop(retired);drop(ga);drop(gb);drop(view);drop(x);drop(r);drop(xo);drop(ro);
            exact(&fresh,&saved,"result survives graph/source lifetime");
            exact(&new_out,&captured_bits,"captured output survives graph/source lifetime");
            report["cases"].as_array_mut().unwrap().push(json!({"cached":cached,"rows":rows,"layout":layout,
                "dtype":format!("{kind:?}"),"turns":turns,"independent_output_owner":true}));save(&path,&report);
        }}}
        // Small local ABBA only. No claim of full drafter speed, and no reuse of
        // this norm timing to apportion the unrelated BF16 matrix-multiply cost.
        let inputs:Vec<_>=(0..16).map(|_|changed(8,0,Kind::BFloat16,dev)).collect();
        for block in 0..2{for(arm,on)in [false,true,true,false].into_iter().enumerate(){
            flag(on);let call=|(x,r):&(Tensor,Tensor)|if on{super::try_hidden(x,r,&normalize).unwrap()}else{old(x,r,&normalize)};
            for input in &inputs{let _=call(input);}tch::Cuda::synchronize(0);
            crate::tp::graph::begin().unwrap();let outputs:Vec<_>=inputs.iter().map(call).collect();crate::tp::graph::end().unwrap();let graph=crate::tp::graph::Owned::take();
            for _ in 0..3{graph.replay();}tch::Cuda::synchronize(0);let mut times=Vec::new();
            for _ in 0..5{let start=Instant::now();for _ in 0..16{graph.replay();}tch::Cuda::synchronize(0);times.push(start.elapsed().as_secs_f64()*1e6/256.);}
            report["timing"].as_array_mut().unwrap().push(json!({"cached":cached,"block":block,"arm":arm,"selected":on,"us_per_final_norm":times}));
            drop(graph);drop(outputs);save(&path,&report);
        }}
    }
    flag(true);
    let x=Tensor::zeros([1,4096],(Kind::BFloat16,dev));assert!(super::try_hidden(&x,&x,&normalize).is_none());
    let x=Tensor::zeros([9,4096],(Kind::BFloat16,dev));assert!(super::try_hidden(&x,&x,&normalize).is_none());
    let x=Tensor::zeros([8,4095],(Kind::BFloat16,dev));assert!(super::try_hidden(&x,&x,&normalize).is_none());
    assert_eq!(report["cases"].as_array().unwrap().len(),112);
    report["complete"]=json!(true);report["gate"]=json!(true);report["case_count"]=json!(112);
    report["changed_graph_checks"]=json!(112*PATTERNS.len());report["fallback_shape_checked"]=json!(true);save(&path,&report);
}

pub(super) fn proposal_check(drafter:&crate::dflash::Drafter,target:&crate::weights::ModelWeights,out:&Path){
    let _restore=Restore::new();let _guard=tch::no_grad_guard();tch::manual_seed(2026092322);
    std::fs::create_dir_all(out).unwrap();let path=out.join(format!("final-norm-proposal-rank{}.json",crate::tp::world().rank));
    let mut report=json!({"complete":false,"gate":false,"cases":[],"rank":crate::tp::world().rank,
        "scope":"real resident drafter; synthetic target features; final norm flag isolated with cached norm0/1",
        "conv_flag_unchanged":std::env::var("GLM53_DRAFT_CONV_FUSED").ok()});save(&path,&report);
    for cached in [false,true]{std::env::set_var(CACHE,if cached{"1"}else{"0"});
        let mut context=drafter.empty_context();let mut length=0;
        for goal in [0i64,3,129]{
            if goal>length{drafter.append(&mut context,&(Tensor::randn([goal-length,20480],(Kind::Float,target.device))*0.05));length=goal;}
            let snapshot=context.snapshot();flag(false);let a=drafter.propose(&context,13041,target);flag(true);let b=drafter.propose(&context,13041,target);
            assert!(!a.final_norm_selected && b.final_norm_selected,"actual final norm dispatch must differ");
            assert!(a.ids.equal(&b.ids));exact(&a.hidden,&b.hidden,"proposal hidden");exact(&a.unary,&b.unary,"proposal unary");exact(&a.edges,&b.edges,"proposal edges");
            assert_eq!(a.path,b.path);assert_eq!(a.selector_fused,b.selector_fused);assert!(context.equal(&snapshot));
            report["cases"].as_array_mut().unwrap().push(json!({"cached":cached,"history":goal,"path":a.path,
                "hidden_unary_edges_raw_bits":true,"ids_path_equal":true,"context_unchanged":true,
                "reference_selected":a.final_norm_selected,"candidate_selected":b.final_norm_selected}));save(&path,&report);
        }
    }
    report["complete"]=json!(true);report["gate"]=json!(true);save(&path,&report);
}
