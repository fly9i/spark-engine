//! GLM-5.3 vision tower (`glm5_next_vision`, BF16 as stored) and the multimodal prefill plumbing.
//!
//! Data flow (design: 自研引擎-多模态接入方案-20260930.md):
//!  - The front end resizes/pads every image or video to its aligned canvas (uint8 HWC, frames stacked) in
//!    /dev/shm and expands each placeholder into *salted* ids in [MM_BASE, 2^24): exact through the FP32
//!    broadcast, never a vocabulary id, and distinct per media content, so prefix identity (plain token-id
//!    `starts_with`) separates different images while identical ones still hit.
//!  - rank0 encodes only the segments (an image, or one 2-frame video group = one attention segment) that
//!    intersect the current prefill chunk, straight from the uint8 canvas: rescale/normalize/patchify on the
//!    GPU (no FP32 patch tensor from the host), 24 ViT blocks, 2x2 downsample, merger.
//!  - The rows go into the chunk's embedding before the existing vocab-parallel embed allreduce: both ranks
//!    zero the placeholder rows, rank0 writes the vision rows, the allreduce (x + 0) distributes them.
//!    No extra collective; the 45 text layers see an ordinary embedding (the text path is NoPE).
use std::path::{Path,PathBuf};
use tch::{Device,Kind,Tensor};

/// First salted placeholder id; salted ids live in [MM_BASE, 2^24).
pub const MM_BASE:i64=1<<20;
pub const IMAGE_TOKEN:i64=154854;

struct Block {n1:Tensor,n2:Tensor,qkv_w:Tensor,qkv_b:Tensor,qn:Tensor,kn:Tensor,proj_w:Tensor,proj_b:Tensor,
    gate_w:Tensor,gate_b:Tensor,up_w:Tensor,up_b:Tensor,down_w:Tensor,down_b:Tensor}

pub struct Vision {
    hidden:i64,heads:i64,patch:i64,merge:i64,tps:i64,out:i64,eps:f64,limit:f64,
    pe_w:Tensor,pe_b:Tensor,blocks:Vec<Block>,post:Tensor,
    /// 2x2 downsample conv as a GEMM: weight [out, kh, kw, c] so the merge-major rows are read as a view.
    ds_w:Tensor,ds_b:Tensor,
    m_proj:Tensor,m_ln_w:Tensor,m_ln_b:Tensor,m_gate:Tensor,m_up:Tensor,m_down:Tensor,
    inv_freq:Tensor,mean:Tensor,std:Tensor,dev:Device,
}

fn num(v:&serde_json::Value,k:&str,d:f64)->f64 {v.get(k).and_then(|x|x.as_f64()).unwrap_or(d)}

/// GLM53_VISION=1 (profile m3-mm1 and later): load the vision tower on rank0 of `serve`. Off keeps the text-only
/// footprint and KV pool of the older profiles.
pub fn enabled()->bool {std::env::var("GLM53_VISION").as_deref()==Ok("1")}

impl Vision {
    pub fn load(dir:&Path,dev:Device)->Vision {
        let cfg:serde_json::Value=serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let vc=&cfg["vision_config"];
        let pp:serde_json::Value=std::fs::read(dir.join("processor_config.json")).ok().and_then(|b|serde_json::from_slice(&b).ok()).unwrap_or_default();
        let ip=&pp["image_processor"];
        let idx=crate::safetensors::ShardIndex::scan(dir).expect("scan vision shards");
        let g=|n:&str|crate::weights::upload_stored(&idx,&format!("model.visual.{n}"),dev);
        let hidden=num(vc,"hidden_size",1024.) as i64;let heads=num(vc,"num_heads",16.) as i64;
        let depth=num(vc,"depth",24.) as usize;let patch=num(vc,"patch_size",14.) as i64;
        let merge=num(vc,"spatial_merge_size",2.) as i64;let tps=num(vc,"temporal_patch_size",2.) as i64;
        let out=num(vc,"out_hidden_size",4096.) as i64;
        let theta=vc.get("rope_parameters").and_then(|r|r.get("rope_theta")).and_then(|x|x.as_f64()).unwrap_or(10000.);
        let pe_w=g("patch_embed.proj.weight");let pe_b=g("patch_embed.proj.bias");
        assert_eq!(pe_w.size(),[hidden,3,tps,patch,patch],"patch_embed shape");
        let pe_w=pe_w.reshape([hidden,3*tps*patch*patch]);
        let blocks=(0..depth).map(|i|{let b=|n:&str|g(&format!("blocks.{i}.{n}"));
            Block{n1:b("norm1.weight"),n2:b("norm2.weight"),qkv_w:b("attn.qkv.weight"),qkv_b:b("attn.qkv.bias"),
                qn:b("attn.q_norm.weight"),kn:b("attn.k_norm.weight"),proj_w:b("attn.proj.weight"),proj_b:b("attn.proj.bias"),
                gate_w:b("mlp.gate_proj.weight"),gate_b:b("mlp.gate_proj.bias"),up_w:b("mlp.up_proj.weight"),up_b:b("mlp.up_proj.bias"),
                down_w:b("mlp.down_proj.weight"),down_b:b("mlp.down_proj.bias")}}).collect();
        let ds=g("downsample.weight");assert_eq!(ds.size(),[out,hidden,merge,merge],"downsample shape");
        let ds_w=ds.permute([0,2,3,1]).reshape([out,merge*merge*hidden]).contiguous();
        let head_dim=hidden/heads;let spatial=head_dim/2;
        // HF: inv_freq = 1 / theta^(arange(0, spatial, 2) / spatial), float32.
        let e=Tensor::arange_start_step(0,spatial,2,(Kind::Float,dev))/(spatial as f64);
        let inv_freq=Tensor::from(theta).to_kind(Kind::Float).to_device(dev).pow(&e).reciprocal();
        let vec3=|k:&str,d:[f64;3]|->Tensor {let v:Vec<f32>=ip.get(k).and_then(|x|x.as_array()).map(|a|a.iter().map(|x|x.as_f64().unwrap() as f32).collect())
            .unwrap_or_else(||d.iter().map(|&x|x as f32).collect());Tensor::from_slice(&v).to_device(dev)};
        let v=Vision{hidden,heads,patch,merge,tps,out,eps:num(vc,"rms_norm_eps",1e-5),limit:num(vc,"swiglu_limit",10.),
            pe_w,pe_b,blocks,post:g("post_layernorm.weight"),ds_w,ds_b:g("downsample.bias"),
            m_proj:g("merger.proj.weight"),m_ln_w:g("merger.post_projection_norm.weight"),m_ln_b:g("merger.post_projection_norm.bias"),
            m_gate:g("merger.gate_proj.weight"),m_up:g("merger.up_proj.weight"),m_down:g("merger.down_proj.weight"),
            inv_freq,mean:vec3("image_mean",[0.48145466,0.4578275,0.40821073]),std:vec3("image_std",[0.26862954,0.26130258,0.27577711]),dev};
        eprintln!("[vision] loaded: depth {depth}, hidden {hidden}, heads {heads}, out {out}, theta {theta}");
        v
    }

    /// HF Glm5NextRMSNorm: FP32 statistics, cast back, then weight * x (input dtype).
    fn rms(&self,x:&Tensor,w:&Tensor)->Tensor {
        let xf=x.to_kind(Kind::Float);
        let var=xf.pow_tensor_scalar(2).mean_dim(&[-1i64][..],true,Kind::Float);
        let y=(xf*(var+self.eps).rsqrt()).to_kind(x.kind());
        w*y
    }

    fn swiglu(&self,gate:Tensor,up:Tensor)->Tensor {
        let gate=gate.clamp_max(self.limit);let up=up.clamp(-self.limit,self.limit);
        gate.silu()*up
    }

    /// Canvas (uint8 [F,H,W,3] on the device; F = 1 for an image, 2*G for G video groups) -> patch rows
    /// [G*gh*gw, 3*tps*p*p] BF16, merge-major, exactly the HF processor's order and arithmetic: rescale in
    /// float64 then float32 ((x * 1/255)), normalize in float32 ((x - mean) / std), cast to BF16.
    pub fn pixels(&self,canvas:&Tensor)->(Tensor,i64,i64,i64) {
        let s=canvas.size();let (f,h,w)=(s[0],s[1],s[2]);
        let (p,m,t)=(self.patch,self.merge,self.tps);
        assert!(h%(p*m)==0&&w%(p*m)==0,"canvas {h}x{w} not aligned to {}",p*m);
        let x=(canvas.to_kind(Kind::Double)*(1.0/255.0)).to_kind(Kind::Float);
        let x=((x-&self.mean)/&self.std).to_kind(Kind::BFloat16);        // [F,H,W,3]
        let x=x.permute([0,3,1,2]);                                        // [F,3,H,W]
        let (g,x)=if f==1 {(1,x.unsqueeze(0).expand([1,t,3,h,w],false))} else {
            assert_eq!(f%t,0,"video frames must be a multiple of the temporal patch");(f/t,x.reshape([f/t,t,3,h,w]))};
        let (gh,gw)=(h/p,w/p);
        let x=x.reshape([g,t,3,gh/m,m,p,gw/m,m,p]).permute([0,3,6,4,7,2,1,5,8]).reshape([g*gh*gw,3*t*p*p]);
        (x,g,gh,gw)
    }

    /// Rotary cos/sin [gh*gw, head_dim] (FP32) for one segment grid, merge-major positions (HF axial 2D rope).
    fn rope(&self,gh:i64,gw:i64)->(Tensor,Tensor) {
        let m=self.merge;let dev=self.dev;
        let hp=Tensor::arange(gh,(Kind::Int64,dev)).view([gh,1]).expand([gh,gw],false);
        let wp=Tensor::arange(gw,(Kind::Int64,dev)).view([1,gw]).expand([gh,gw],false);
        let order=|t:Tensor|t.reshape([gh/m,m,gw/m,m]).permute([0,2,1,3]).reshape([-1]);
        let fh=order(hp).to_kind(Kind::Float).unsqueeze(1)*self.inv_freq.unsqueeze(0);
        let fw=order(wp).to_kind(Kind::Float).unsqueeze(1)*self.inv_freq.unsqueeze(0);
        let hw=Tensor::cat(&[fh,fw],1);let full=Tensor::cat(&[&hw,&hw],1);
        (full.cos(),full.sin())
    }

    fn rotate(&self,x:&Tensor,cos:&Tensor,sin:&Tensor)->Tensor {
        // x [N,H,D] BF16 -> FP32 rotation -> BF16 (HF apply_rotary_pos_emb_vision).
        let xf=x.to_kind(Kind::Float);let d=xf.size()[2];
        let x1=xf.narrow(2,0,d/2);let x2=xf.narrow(2,d/2,d/2);
        let rot=Tensor::cat(&[x2.neg(),x1],2);
        (&xf*cos.unsqueeze(1)+rot*sin.unsqueeze(1)).to_kind(x.kind())
    }

    /// `g` segments of identical grid (gh, gw): patch rows -> merged tokens [g*gh*gw/4, out] BF16.
    pub fn forward(&self,pixels:&Tensor,g:i64,gh:i64,gw:i64)->Tensor {
        let n=gh*gw;let rows=g*n;assert_eq!(pixels.size()[0],rows);
        let (heads,hd)=(self.heads,self.hidden/self.heads);
        let mut x=pixels.linear(&self.pe_w,Some(&self.pe_b));
        let (cos,sin)=self.rope(gh,gw);
        let (cos,sin)=if g>1 {(cos.repeat([g,1]),sin.repeat([g,1]))} else {(cos,sin)};
        let scale=(hd as f64).powf(-0.5);
        for b in &self.blocks {
            let h=self.rms(&x,&b.n1);
            let qkv=h.linear(&b.qkv_w,Some(&b.qkv_b)).view([rows,3,heads,hd]);
            let q=self.rms(&qkv.select(1,0),&b.qn);let k=self.rms(&qkv.select(1,1),&b.kn);let v=qkv.select(1,2);
            let q=self.rotate(&q,&cos,&sin);let k=self.rotate(&k,&cos,&sin);
            // Non-causal attention inside each segment: [g, heads, n, hd].
            let seg=|t:Tensor|t.view([g,n,heads,hd]).transpose(1,2);
            let o=Tensor::scaled_dot_product_attention(&seg(q),&seg(k),&seg(v.contiguous()),None::<Tensor>,0.0,false,scale,false);
            let o=o.transpose(1,2).reshape([rows,self.hidden]);
            x=x+o.linear(&b.proj_w,Some(&b.proj_b));
            let h=self.rms(&x,&b.n2);
            let a=self.swiglu(h.linear(&b.gate_w,Some(&b.gate_b)),h.linear(&b.up_w,Some(&b.up_b)));
            x=x+a.linear(&b.down_w,Some(&b.down_b));
        }
        let x=self.rms(&x,&self.post);
        // Downsample: every 4 consecutive rows are one 2x2 merge window (merge-major order).
        let x=x.reshape([rows/(self.merge*self.merge),self.merge*self.merge*self.hidden]).linear(&self.ds_w,Some(&self.ds_b));
        let x=x.linear(&self.m_proj,None::<Tensor>);
        let x=x.layer_norm([self.out],Some(&self.m_ln_w),Some(&self.m_ln_b),1e-5,false).gelu("none");
        let a=self.swiglu(x.linear(&self.m_gate,None::<Tensor>),x.linear(&self.m_up,None::<Tensor>));
        a.linear(&self.m_down,None::<Tensor>)
    }

    /// Which SDPA backend a segment of `n` patches would use (0 math, 1 flash, 2 efficient, 3 cudnn...).
    pub fn sdp_choice(&self,n:i64)->i64 {
        let hd=self.hidden/self.heads;let q=Tensor::zeros([1,self.heads,n,hd],(Kind::BFloat16,self.dev));
        Tensor::internal_fused_sdp_choice(&q,&q,&q,None::<Tensor>,0.0,false,(hd as f64).powf(-0.5),false)
    }
}

/// One media item of a request (rank0): its canvas file and the placeholder runs it fills.
#[derive(Clone,Debug)]
pub struct MmItem {pub path:PathBuf,pub frames:i64,pub height:i64,pub width:i64,
    /// (prompt position, placeholder count, first frame): an image has one segment (frame 0, duplicated
    /// temporally); a video has one per 2-frame group.
    pub segments:Vec<(usize,usize,i64)>}

/// Parse the request's "mm" array: [{"path","frames","height","width","segments":[[pos,len,frame],...]}].
pub fn parse_items(v:&serde_json::Value)->Result<Vec<MmItem>,String> {
    let Some(a)=v.as_array() else {return Ok(Vec::new())};
    let mut out=Vec::new();
    for it in a {
        let path=PathBuf::from(it["path"].as_str().ok_or("mm item without path")?);
        let get=|k:&str|it[k].as_i64().ok_or(format!("mm item without {k}"));
        let (frames,height,width)=(get("frames")?,get("height")?,get("width")?);
        if frames<1||height<28||width<28||height%28!=0||width%28!=0 {return Err(format!("mm item has a bad canvas {frames}x{height}x{width}"));}
        let meta=std::fs::metadata(&path).map_err(|e|format!("mm canvas {}: {e}",path.display()))?;
        if meta.len() as i64!=frames*height*width*3 {return Err(format!("mm canvas {} has {} bytes, expected {}",path.display(),meta.len(),frames*height*width*3));}
        let tokens_per=(height/28)*(width/28);
        let mut segments=Vec::new();
        for s in it["segments"].as_array().ok_or("mm item without segments")? {
            let s=s.as_array().ok_or("bad segment")?;
            let f=|i:usize|s.get(i).and_then(|x|x.as_i64()).ok_or("bad segment");
            let (pos,len,frame)=(f(0)?,f(1)?,f(2)?);
            if pos<0||len!=tokens_per||frame<0||frame>=frames||(frames>1&&(frame%2!=0||frame+2>frames)) {return Err(format!("mm segment ({pos},{len},{frame}) does not fit the canvas"));}
            segments.push((pos as usize,len as usize,frame));
        }
        out.push(MmItem{path,frames,height,width,segments});
    }
    Ok(out)
}

/// Check the salted placeholders of a prompt against its items: every id is a vocabulary id or a salted
/// placeholder, and the placeholder positions are exactly the items' segments.
pub fn validate(ids:&[i64],vocab:i64,items:&[MmItem])->Result<(),String> {
    let mut mark=vec![false;ids.len()];
    for it in items {for &(pos,len,_) in &it.segments {
        if pos+len>ids.len() {return Err("mm segment past the prompt end".into());}
        for j in pos..pos+len {if mark[j] {return Err("overlapping mm segments".into());}mark[j]=true;}}}
    for (j,&t) in ids.iter().enumerate() {
        let salted=(MM_BASE..1<<24).contains(&t);
        if salted!=mark[j] {return Err(if salted {format!("placeholder id at {j} has no mm segment")} else {format!("mm segment covers non-placeholder id {t} at {j}")});}
        if !salted&&!(0..vocab).contains(&t) {return Err(format!("token id {t} at {j} is outside the vocabulary"));}
    }
    Ok(())
}

/// Per-sequence encoder cache (rank0): encoded segments waiting for the chunks that consume them.
#[derive(Default)]
pub struct MmState {pub items:Vec<MmItem>,done:std::collections::HashMap<(usize,usize),Tensor>,pub encode_ms:f64,pub encoded_tokens:usize}

impl MmState {
    pub fn new(items:Vec<MmItem>)->Self {MmState{items,..Default::default()}}

    /// Encode every segment that intersects [begin, end) and is not cached yet (same item: one batched call
    /// per group of segments), then return the placeholder rows of this chunk in position order as FP32
    /// [k, out]. Segments ending at or before `end` are released.
    pub fn chunk_rows(&mut self,vision:&Vision,begin:usize,end:usize)->Option<Tensor> {
        let t0=std::time::Instant::now();
        let mut pieces:Vec<(usize,Tensor)>=Vec::new();
        for (ii,it) in self.items.iter().enumerate() {
            let need:Vec<usize>=(0..it.segments.len()).filter(|&si|{let (p,l,_)=it.segments[si];p<end&&p+l>begin&&!self.done.contains_key(&(ii,si))}).collect();
            if !need.is_empty() {
                // Video groups share one grid: batch up to ~16K patches per forward (bounded activations).
                let per=(it.height/14)*(it.width/14);let batch=((16384/per).max(1)) as usize;
                for group in need.chunks(batch) {
                    let canvas=read_frames(it,group.iter().map(|&si|it.segments[si].2).collect::<Vec<_>>().as_slice(),vision.dev);
                    let (px,g,gh,gw)=vision.pixels(&canvas);
                    let y=vision.forward(&px,g,gh,gw);
                    let tok=(gh*gw/4) as i64;
                    for (k,&si) in group.iter().enumerate() {self.done.insert((ii,si),y.narrow(0,k as i64*tok,tok));}
                    self.encoded_tokens+=group.len()*tok as usize;
                }
            }
            for (si,&(p,l,_)) in it.segments.iter().enumerate() {
                if p<end&&p+l>begin {
                    let lo=p.max(begin);let hi=(p+l).min(end);
                    let y=self.done.get(&(ii,si)).unwrap();
                    pieces.push((lo,y.narrow(0,(lo-p) as i64,(hi-lo) as i64)));
                }
            }
        }
        // Release consumed segments.
        let items=&self.items;self.done.retain(|&(ii,si),_|{let (p,l,_)=items[ii].segments[si];p+l>end});
        if pieces.is_empty() {return None;}
        pieces.sort_by_key(|p|p.0);
        let rows=Tensor::cat(&pieces.into_iter().map(|p|p.1).collect::<Vec<_>>(),0).to_kind(Kind::Float);
        self.encode_ms+=t0.elapsed().as_secs_f64()*1000.;
        Some(rows)
    }
}

/// Canvas frames for the given segments: images read frame 0 once; video groups read frames [f, f+2).
fn read_frames(it:&MmItem,firsts:&[i64],dev:Device)->Tensor {
    use std::io::{Read,Seek,SeekFrom};
    let fb=(it.height*it.width*3) as usize;
    let mut f=std::fs::File::open(&it.path).unwrap_or_else(|e|panic!("mm canvas {}: {e}",it.path.display()));
    let per=if it.frames==1 {1} else {2};
    let mut buf=vec![0u8;fb*per*firsts.len()];
    for (k,&first) in firsts.iter().enumerate() {
        f.seek(SeekFrom::Start(first as u64*fb as u64)).unwrap();
        f.read_exact(&mut buf[k*fb*per..(k+1)*fb*per]).unwrap_or_else(|e|panic!("mm canvas {}: {e}",it.path.display()));
    }
    Tensor::from_slice(&buf).view([(per*firsts.len()) as i64,it.height,it.width,3]).to_device(dev)
}

/// Placeholder rows (positions within the chunk) of `ids`, in order.
pub fn placeholder_rows(ids:&[i64])->Vec<i64> {ids.iter().enumerate().filter(|(_,&t)|t>=MM_BASE).map(|(i,_)|i as i64).collect()}

/// `vision-probe <model_dir> <canvas.u8> <frames> <height> <width> <out.pt>`: encode one canvas (an image when
/// frames == 1, else a video of frames/2 groups) with the serving code path and save pixels and outputs, for
/// the HF reference comparison (bench/mm).
pub fn probe(args:&[String]) {
    let _g=tch::no_grad_guard();let dev=Device::Cuda(0);
    let v=Vision::load(Path::new(&args[2]),dev);
    let (frames,h,w)=(args[4].parse::<i64>().unwrap(),args[5].parse::<i64>().unwrap(),args[6].parse::<i64>().unwrap());
    let it=MmItem{path:PathBuf::from(&args[3]),frames,height:h,width:w,segments:Vec::new()};
    let firsts:Vec<i64>=if frames==1 {vec![0]} else {(0..frames/2).map(|g|2*g).collect()};
    let canvas=read_frames(&it,&firsts,dev);
    let (px,g,gh,gw)=v.pixels(&canvas);
    eprintln!("[vision-probe] grid g {g} gh {gh} gw {gw} rows {} sdp_choice {}",px.size()[0],v.sdp_choice(gh*gw));
    let _=v.forward(&px,g,gh,gw);tch::Cuda::synchronize(0);
    let t0=std::time::Instant::now();let y=v.forward(&px,g,gh,gw);tch::Cuda::synchronize(0);
    eprintln!("[vision-probe] forward {:.1} ms -> {:?}",t0.elapsed().as_secs_f64()*1000.,y.size());
    // Raw little-endian FP32 (BF16 values widened exactly): <out>.pixels.f32, <out>.embeds.f32.
    let dump=|t:&Tensor,suffix:&str|{let v=Vec::<f32>::try_from(t.to_kind(Kind::Float).to_device(Device::Cpu).reshape([-1])).unwrap();
        let bytes:Vec<u8>=v.iter().flat_map(|x|x.to_le_bytes()).collect();std::fs::write(format!("{}.{suffix}.f32",args[7]),bytes).unwrap();};
    dump(&px,"pixels");dump(&y,"embeds");
    println!("{}",serde_json::json!({"pixels":px.size(),"embeds":y.size(),"grid":[g,gh,gw]}));
}
