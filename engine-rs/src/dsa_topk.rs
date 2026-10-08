//! Default-off Ranked verifier TopK batching. Existing row-local score math,
//! mask/expand kernels and attention consumers stay unchanged. Exact tie/order
//! parity is a qualification requirement, not a promise of the TopK API.
use tch::{Tensor,Kind,Device};
use crate::dsa::TreeSelection;

pub(crate) fn enabled()->bool {
    match std::env::var("GLM53_DSA_TOPK_BATCH") {
        Ok(v)=>match v.as_str(){"0"=>false,"1"=>true,_=>panic!("GLM53_DSA_TOPK_BATCH must be0 or1")},
        Err(std::env::VarError::NotPresent)=>false,
        Err(_)=>panic!("GLM53_DSA_TOPK_BATCH must be0 or1"),
    }
}

pub(crate) fn fast_enabled()->bool {std::env::var("GLM53_DSA_TOPK_FAST").as_deref()==Ok("1")}

/// Metadata-only admission: no device length read and no mutation on fallback.
/// The caller already checked capacity and topological parent order. Siblings
/// are allowed: each row owns its state and immutable original position.
pub(crate) fn eligible(selection:TreeSelection,x:&Tensor,pools:&Tensor,pos:&Tensor)->bool {
    enabled()&&crate::dsa_index::enabled()&&selection==TreeSelection::Ranked&&
        x.device().is_cuda()&&x.kind()==Kind::Float&&x.dim()==2&&
        (2..=16).contains(&x.size()[0])&&x.is_contiguous()&&
        pools.device()==x.device()&&pools.kind()==Kind::Float&&pools.dim()==2&&pools.is_contiguous()&&
        (1..=0x7fffff00).contains(&pools.size()[0])&&pools.size()[1]>0&&
        pos.device()==x.device()&&pos.kind()==Kind::Int64&&pos.numel()==1&&pos.is_contiguous()
}

/// One bounded buffer. Every row is fully overwritten by the existing mask
/// kernel before TopK; inactive capacity is -FLT_MAX, never zero or -Inf.
pub(crate) struct MaskedRows {masked:Tensor,positions:Vec<Tensor>,sidecar:Option<Tensor>}
impl MaskedRows {
    pub(crate) fn new(rows:i64,pools:i64,dev:Device)->Self {
        assert!((2..=16).contains(&rows)&&(1..=0x7fffff00).contains(&pools)&&dev.is_cuda());
        Self{masked:Tensor::empty([rows,pools],(Kind::Float,dev)),positions:Vec::with_capacity(rows as usize),
            sidecar:crate::dsa_position::enabled().then(||Tensor::empty([rows],(Kind::Int64,dev)))}
    }
    /// New mode borrows length only until the current-stream mask enqueue;
    /// old mode retains the original independent copy for delayed expand.
    /// Allocation choice is fixed by this instance, not a later env lookup.
    pub(crate) fn position_input(&self,len:&Tensor)->Tensor {
        if self.sidecar.is_some(){len.shallow_clone()}else{len.copy()}
    }
    /// On the old path `pos` must own a pre-append copy. On the capture path it
    /// may alias mutable len: this call enqueues its capture before returning,
    /// and retains ONLY the sidecar row, never that mutable input alias.
    pub(crate) fn push(&mut self,scores:&Tensor,pos:Tensor) {
        let row=self.positions.len() as i64;assert!(row<self.masked.size()[0]);
        assert_eq!(scores.size(),[self.masked.size()[1]]);
        if let Some(sidecar)=&self.sidecar {
            let captured=sidecar.narrow(0,row,1);
            crate::dsa_position::mask_into(scores,&pos,&self.masked.get(row),&captured);
            self.positions.push(captured);
        } else {
            crate::dsa_index::mask_into(scores,&pos,&self.masked.get(row));
            self.positions.push(pos);
        }
    }
    /// push() for every row of `scores` [rows, pools] in one launch (sidecar mode only; row r equals push(scores[r], pos[r])).
    /// false: not eligible, nothing done.
    pub(crate) fn push_all(&mut self,scores:&Tensor,pos:&[Tensor])->bool {
        let Some(side)=&self.sidecar else {return false};
        let rows=self.masked.size()[0];
        if !self.positions.is_empty() || pos.len() as i64!=rows || rows>8 || scores.size()!=self.masked.size() || !scores.is_contiguous() || scores.kind()!=Kind::Float
            || pos.iter().any(|p|p.kind()!=Kind::Int64||p.numel()!=1||!p.is_contiguous()) {return false;}
        let ptrs:Vec<*const i64>=pos.iter().map(|p|p.data_ptr() as *const i64).collect();
        extern "C"{fn rs_dsa_index_mask_capture_rows(scores:*const f32,pos:*const *const i64,masked:*mut f32,captured:*mut i64,pools:i32,rows:i32)->i32;}
        assert_eq!(unsafe{rs_dsa_index_mask_capture_rows(scores.data_ptr().cast(),ptrs.as_ptr(),self.masked.data_ptr().cast(),side.data_ptr().cast(),
            self.masked.size()[1] as i32,rows as i32)},0,"DSA row-batched mask capture");
        for r in 0..rows {self.positions.push(side.narrow(0,r,1));}
        true
    }
    /// GLM53_DSA_TOPK_FAST with the position sidecar: the masked buffer itself, for a scorer that writes -FLT_MAX at and
    /// beyond each row's complete count (dsa_score_multi); follow with push_prescored.
    pub(crate) fn width(&self)->i64 {self.masked.size()[1]}
    pub(crate) fn prescore_target(&self,rows:i64)->Option<Tensor> {
        (fast_enabled() && self.sidecar.is_some() && self.positions.is_empty() && self.masked.size()[0]==rows).then(||self.masked.shallow_clone())
    }
    /// The rows of prescore_target() are final: capture the positions only (no copy of the scores).
    pub(crate) fn push_prescored(&mut self,pos:&[Tensor]) {
        let side=self.sidecar.as_ref().expect("prescored rows need the sidecar");let rows=self.masked.size()[0];
        assert!(self.positions.is_empty() && pos.len() as i64==rows && rows<=8);
        assert!(pos.iter().all(|p|p.kind()==Kind::Int64&&p.numel()==1&&p.is_contiguous()));
        let ptrs:Vec<*const i64>=pos.iter().map(|p|p.data_ptr() as *const i64).collect();
        extern "C"{fn rs_dsa_capture_rows(pos:*const *const i64,captured:*mut i64,rows:i32)->i32;}
        assert_eq!(unsafe{rs_dsa_capture_rows(ptrs.as_ptr(),side.data_ptr().cast(),rows as i32)},0,"DSA position capture");
        for r in 0..rows {self.positions.push(side.narrow(0,r,1));}
    }
    pub(crate) fn finish(self)->RankedRows {self.finish_inner(false).1}
    fn finish_outputs(self)->(Tensor,RankedRows) {self.finish_inner(true)}
    fn finish_inner(self,want_values:bool)->(Tensor,RankedRows) {
        assert_eq!(self.positions.len(),self.masked.size()[0] as usize);
        // This is the only ranking change: one [T,capacity] dimension1 TopK.
        // Same width/k/largest/sorted per row; exact ties still require GPU gate.
        if let Some(side)=self.sidecar.as_ref().filter(|_|fast_enabled()) {
            // GLM53_DSA_TOPK_FAST: same indices and order as the sorted largest topk below (value desc, ties by index
            // asc), reading only each row's valid prefix. Values are not produced (no consumer outside probes).
            let (rows,pools)=(self.masked.size()[0],self.masked.size()[1]);let k=512.min(pools);
            if side.is_contiguous() && side.numel() as i64==rows && pools>=1024 {
                let selected=Tensor::empty([rows,k],(Kind::Int64,self.masked.device()));
                extern "C"{fn rs_dsa_topk_rows(masked:*const f32,pos:*const i64,selected:*mut i64,pools:i32,k:i32,rows:i32)->i32;}
                assert_eq!(unsafe{rs_dsa_topk_rows(self.masked.data_ptr().cast(),side.data_ptr().cast(),selected.data_ptr().cast(),pools as i32,k as i32,rows as i32)},0,"DSA fast topk");
                let values=if want_values {self.masked.gather(1,&selected,false)} else {Tensor::empty([0],(Kind::Float,self.masked.device()))};
                return (values,RankedRows{selected,positions:self.positions,_sidecar:self.sidecar});
            }
        }
        let (values,selected)=self.masked.topk(512.min(self.masked.size()[1]),1,true,true);
        (values,RankedRows{selected,positions:self.positions,_sidecar:self.sidecar})
    }
}

pub(crate) struct RankedRows {selected:Tensor,positions:Vec<Tensor>,_sidecar:Option<Tensor>}
impl RankedRows {
    /// All rows' expanded token ids [rows, 4k+3] in one launch (row r equals tokens(r)); None without the
    /// position sidecar (positions are then separate tensors).
    pub(crate) fn tokens_all(&self)->Option<Tensor> {
        let side=self._sidecar.as_ref()?;
        if !self.selected.is_contiguous() || self.selected.kind()!=Kind::Int64 || !side.is_contiguous() {return None;}
        let (rows,k)=(self.selected.size()[0],self.selected.size()[1]);
        if side.numel() as i64!=rows || !(1..=512).contains(&k) {return None;}
        let out=Tensor::empty([rows,4*k+3],(Kind::Int64,self.selected.device()));
        extern "C" {fn rs_dsa_index_expand_rows(selected:*const i64,pos:*const i64,out:*mut i64,k:i32,rows:i32)->i32;}
        assert_eq!(unsafe{rs_dsa_index_expand_rows(self.selected.data_ptr().cast(),side.data_ptr().cast(),out.data_ptr().cast(),k as i32,rows as i32)},0);
        Some(out)
    }
    pub(crate) fn tokens(&self,row:i64)->Tensor {
        assert!((0..self.selected.size()[0]).contains(&row));
        let selected=self.selected.get(row);
        assert!(selected.is_contiguous());
        let out=Tensor::empty([selected.size()[0]*4+3],(Kind::Int64,selected.device()));
        crate::dsa_index::expand_into(&selected,&self.positions[row as usize],&out);
        out
    }
}

#[path="dsa_topk_probe.rs"] mod local_probe;
pub(crate) use local_probe::{run as probe,real_layer,real_check};

#[path="dsa_position_probe.rs"] mod position_probe_impl;
pub(crate) use position_probe_impl::run as position_probe;

/// dsa-topk-fast-probe: GLM53_DSA_TOPK_FAST kernel vs ATen sorted largest topk on masked rows (ties, NaN, +-0, -FLT_MAX
/// inside the valid prefix, every complete count edge), bitwise on indices; then timing at the serving width.
pub fn fast_probe(out:&std::path::Path) {
    use serde_json::json;let _g=tch::no_grad_guard();let dev=Device::Cuda(0);std::fs::create_dir_all(out).unwrap();
    extern "C"{fn rs_dsa_topk_rows(masked:*const f32,pos:*const i64,selected:*mut i64,pools:i32,k:i32,rows:i32)->i32;}
    let run=|masked:&Tensor,pos:&Tensor|{let (rows,pools)=(masked.size()[0],masked.size()[1]);let k=512.min(pools);
        let sel=Tensor::empty([rows,k],(Kind::Int64,dev));
        assert_eq!(unsafe{rs_dsa_topk_rows(masked.data_ptr().cast(),pos.data_ptr().cast(),sel.data_ptr().cast(),pools as i32,k as i32,rows as i32)},0);sel};
    let mut cases=0;let mut fails=Vec::new();
    for pools in [1024i64,2048,5120,10240] { for dist in 0..7 { for rows in [1i64,3,8] {
        tch::manual_seed(1000+pools+dist*10+rows);
        let mut s=match dist {
            0=>Tensor::randn([rows,pools],(Kind::Float,dev)),
            1=>(Tensor::randn([rows,pools],(Kind::Float,dev))*3.).round(),            // heavy ties
            2=>Tensor::zeros([rows,pools],(Kind::Float,dev)),                          // all equal
            3=>Tensor::randn([rows,pools],(Kind::Float,dev)).relu()*0.01,              // many +0
            4=>{let z=Tensor::randn([rows,pools],(Kind::Float,dev)).relu();(&z*-1.)}   // many -0 and negatives
            5=>{let z=Tensor::randn([rows,pools],(Kind::Float,dev));let m=Tensor::rand([rows,pools],(Kind::Float,dev)).lt(0.05);z.masked_fill(&m,-3.4028234663852886e38)}
            _=>{let z=Tensor::randn([rows,pools],(Kind::Float,dev));let m=Tensor::rand([rows,pools],(Kind::Float,dev)).lt(0.01);z.masked_fill(&m,f64::NAN)}
        };
        s=s.contiguous();
        for complete in [0i64,1,7,300,511,512,513,1000,4000,pools-1,pools] { if complete>pools {continue;}
            let cs:Vec<i64>=(0..rows).map(|r|(complete-r*37).max(0)).collect();
            let pos=Tensor::from_slice(&cs.iter().map(|c|4*c+2).collect::<Vec<_>>()).to_device(dev);
            let idx=Tensor::arange(pools,(Kind::Int64,dev)).view([1,pools]);
            let valid=idx.lt_tensor(&Tensor::from_slice(&cs).to_device(dev).view([rows,1]));
            let masked=s.where_self(&valid,&Tensor::full([1],-3.4028234663852886e38,(Kind::Float,dev))).contiguous();
            let (_,gold)=masked.topk(512.min(pools),1,true,true);
            let got=run(&masked,&pos);cases+=1;
            if !gold.equal(&got) {let d=i64::try_from(gold.ne_tensor(&got).sum(Kind::Int64)).unwrap();
                if fails.len()<2 {
                    let r=i64::try_from(gold.ne_tensor(&got).any_dim(1,false).to_kind(Kind::Int64).argmax(0,false)).unwrap();
                    let g:Vec<i64>=Vec::try_from(gold.get(r).to_device(Device::Cpu)).unwrap();let o:Vec<i64>=Vec::try_from(got.get(r).to_device(Device::Cpu)).unwrap();
                    let row:Vec<f32>=Vec::try_from(masked.get(r).to_device(Device::Cpu)).unwrap();
                    let p=g.iter().zip(&o).position(|(a,b)|a!=b).unwrap();let lo=p.saturating_sub(4);let hi=(p+12).min(g.len());
                    let show=|v:&[i64]|v[lo..hi].iter().map(|&i|format!("{i}:{:?}({:08x})",row[i as usize],row[i as usize].to_bits())).collect::<Vec<_>>();
                    eprintln!("[dsa-topk-fast] mismatch pools {pools} dist {dist} complete {complete} row {r} at {p}\n gold {:?}\n ours {:?}",show(&g),show(&o));}
                fails.push(json!({"pools":pools,"dist":dist,"rows":rows,"complete":complete,"diff":d}));}
        }
    }}}
    eprintln!("[dsa-topk-fast] {} cases, {} mismatching: {:?}",cases,fails.len(),fails.iter().take(12).collect::<Vec<_>>());
    // timing at the serving capacity (5120 pools), rows 2/4/8, valid prefix 500/2000/5120
    let mut timing=Vec::new();
    for complete in [500i64,2000,5120] { for rows in [2i64,4,8] {
        let pools=5120;let masked=Tensor::randn([rows,pools],(Kind::Float,dev));
        let pos=Tensor::full([rows],4*complete+2,(Kind::Int64,dev));
        let time=|f:&dyn Fn()|{for _ in 0..20{f();}tch::Cuda::synchronize(0);let t=std::time::Instant::now();for _ in 0..400{f();}tch::Cuda::synchronize(0);t.elapsed().as_secs_f64()*1e6/400.};
        let a=time(&||{let _=masked.topk(512,1,true,true);});let b=time(&||{let _=run(&masked,&pos);});
        eprintln!("[dsa-topk-fast] complete {complete} rows {rows}: aten {a:.1} us, fast {b:.1} us");
        timing.push(json!({"complete":complete,"rows":rows,"aten_us":a,"fast_us":b}));
    }}
    std::fs::write(out.join("dsa-topk-fast.json"),serde_json::to_string_pretty(&json!({"cases":cases,"fails":fails,"timing":timing})).unwrap()).unwrap();
    assert!(fails.is_empty(),"fast topk differs from ATen");
    eprintln!("[dsa-topk-fast] PASS");
}
