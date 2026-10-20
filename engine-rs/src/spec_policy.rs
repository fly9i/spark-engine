//! Deterministic per-request depth policy. Costs are calibrated on both ranks
//! before generation; only common accepted-token counts update the policy.
pub struct Policy {pub costs:[f64;3],survival:[f64;7],pub rounds:usize}
impl Policy {
    pub fn new(costs:[f64;3])->Self {
        assert!(costs.iter().all(|c|c.is_finite()&&*c>0.));Self{costs,survival:[1.;7],rounds:0}
    }
    pub fn depth(&self)->usize {
        // Observe the long tail regularly; a shallow policy cannot infer it.
        if self.rounds<4 || self.rounds%8==0{return 7;}
        [3usize,5,7].into_iter().enumerate().max_by(|(i,a),(j,b)|{
            let score=|d:usize,c:f64|(1.+self.survival[..d].iter().sum::<f64>())/c;
            score(*a,self.costs[*i]).total_cmp(&score(*b,self.costs[*j]))
        }).unwrap().1
    }
    pub fn observe(&mut self,accepted:usize,budget:usize) {
        assert!(accepted<=budget && budget<=7);
        for i in 0..budget {self.survival[i]=0.8*self.survival[i]+0.2*if accepted>i{1.}else{0.};}
        self.rounds+=1;
    }
}

/// Single-sequence chain depth from the drafter's per-position path confidence (L2: target output unchanged;
/// with GLM53_VERIFY_INVARIANT=1 the tokens are identical for any depth).
/// Default: stop at the first position with conf < tau. GLM53_DEPTH_RULE=rate: pick d maximizing
/// E[emitted tokens | d] - LAMBDA * T(d+1), E from the calibrated prefix-acceptance probabilities below
/// (optimal for tokens per unit time when LAMBDA is the achieved rate). Deterministic in conf, so both ranks agree.
pub fn chain_depth(conf:&[f32],max:usize,tau:f64)->usize {chain_depth_online(conf,max,tau,None)}
/// chain_depth with a request's online calibration (rate rule only; None or GLM53_DEPTH_ONLINE unset: the table as is).
pub fn chain_depth_online(conf:&[f32],max:usize,tau:f64,online:Option<&Online>)->usize {
    if rate_rule() {return rate_depth_scaled(conf,max,online.filter(|_|online_enabled()).map_or(1.0,|o|o.factor()));}
    let mut d=0;while d<max && (conf[d] as f64)>=tau {d+=1;}d.max(1).min(max)
}
fn rate_rule()->bool {static R:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*R.get_or_init(||std::env::var("GLM53_DEPTH_RULE").as_deref()==Ok("rate"))}
// Calibration: bench/depth-sim-20260929/gen_table.py on the w10 trace (thinking on, T=1, 100 prompts, 56,474 rounds).
// P[i][b]: P(draft i accepted | drafts < i accepted, conf in bin b of 20). COST_MS: w71 (r13p) round wall per verified
// row count. LAMBDA: optimal rate for these costs (Dinkelbach). 5-fold CV vs tau 0.7: +1.04% (95% CI +0.91..+1.16).
const LAMBDA:f64=0.035621;
const COST_MS:[f64;9]=[0.,0.,64.82,71.92,77.86,85.46,91.79,95.70,103.37];
const P:[[f32;20];7]=[
  [0.0765,0.0854,0.1328,0.1638,0.1758,0.2157,0.2332,0.2688,0.3162,0.3268,0.3473,0.3953,0.4400,0.4527,0.4985,0.5203,0.5830,0.6277,0.6946,0.9234],
  [0.2987,0.3149,0.3086,0.2795,0.3035,0.3380,0.3217,0.3044,0.3696,0.3765,0.3987,0.3972,0.3910,0.4743,0.5343,0.5362,0.5636,0.6162,0.6820,0.9156],
  [0.3573,0.3582,0.3140,0.3097,0.3372,0.3456,0.3520,0.3427,0.3365,0.4010,0.4046,0.4491,0.4635,0.4833,0.4907,0.5495,0.5620,0.6101,0.6901,0.9228],
  [0.3732,0.3450,0.3857,0.3230,0.3322,0.3431,0.3594,0.3376,0.3345,0.3836,0.4534,0.4307,0.4835,0.4463,0.5336,0.5183,0.5845,0.6215,0.6733,0.9311],
  [0.4000,0.4105,0.3770,0.3209,0.4181,0.3731,0.3532,0.3571,0.4272,0.4089,0.4839,0.4440,0.4562,0.5240,0.5670,0.5683,0.5678,0.6675,0.7179,0.9353],
  [0.4147,0.3991,0.3694,0.4295,0.3778,0.4028,0.4060,0.4179,0.4545,0.4729,0.4526,0.4615,0.4748,0.4966,0.5319,0.6173,0.6284,0.6353,0.7052,0.9351],
  [0.4452,0.3836,0.5446,0.4750,0.3956,0.4400,0.5079,0.5224,0.4557,0.5412,0.4937,0.5269,0.5000,0.4667,0.5752,0.5867,0.6214,0.6701,0.7413,0.9369],
];
/// GLM53_DEPTH_ONLINE=1: per-request online calibration of the rate table (the vLLM DSD idea, per request). A request
/// keeps decayed sums of the outcomes O and the table's predictions P of the drafts it verified (draft i counts when the
/// drafts before it were accepted); the depth choice scales the table's conditional probabilities by
/// c = (O + K) / (P + K), clamped to [0.5, 2]. K (GLM53_DEPTH_ONLINE_PRIOR, default 4) keeps a new request at the table;
/// the decay per round is GLM53_DEPTH_ONLINE_DECAY (default 0.9). Deterministic in the common accepted counts and the
/// drafter's confidences (rank1 adopts rank0's depth anyway, GLM53_SERVE_DRAFT_SYNC).
#[derive(Clone,Copy,Default,Debug)] pub struct Online {o:f64,p:f64,pub rounds:usize}
pub fn online_enabled()->bool {static E:std::sync::OnceLock<bool>=std::sync::OnceLock::new();*E.get_or_init(||std::env::var("GLM53_DEPTH_ONLINE").as_deref()==Ok("1"))}
fn online_param(name:&str,default:f64)->f64 {std::env::var(name).ok().and_then(|v|v.parse().ok()).unwrap_or(default)}
impl Online {
    pub fn factor(&self)->f64 {
        static K:std::sync::OnceLock<f64>=std::sync::OnceLock::new();let k=*K.get_or_init(||online_param("GLM53_DEPTH_ONLINE_PRIOR",4.));
        ((self.o+k)/(self.p+k)).clamp(0.5,2.0)
    }
    /// After a verified chain: depth drafts with confidences conf, `accepted` of them accepted (a prefix).
    pub fn observe(&mut self,conf:&[f32],depth:usize,accepted:usize) {
        static G:std::sync::OnceLock<f64>=std::sync::OnceLock::new();let g=*G.get_or_init(||online_param("GLM53_DEPTH_ONLINE_DECAY",0.9));
        let t=table();self.o*=g;self.p*=g;self.rounds+=1;
        for i in 0..depth.min(conf.len()).min(7) {
            if i>accepted {break;}
            self.p+=t.p[i][bin(conf[i])] as f64;self.o+=if accepted>i {1.} else {0.};
        }
    }
}
/// GLM53_DEPTH_TABLE=FILE (JSON {"lambda": f64, "cost_ms": [9 x f64], "p": [7 x [20 x f32]]}): a recalibrated rate
/// table (other drafter / engine costs, see spx1 dtable_fit.py); unset = the built-in table above.
struct Table {lambda:f64,cost:[f64;9],p:[[f32;20];7]}
fn table()->&'static Table {
    static T:std::sync::OnceLock<Table>=std::sync::OnceLock::new();
    T.get_or_init(||{
        let Ok(path)=std::env::var("GLM53_DEPTH_TABLE") else {return Table{lambda:LAMBDA,cost:COST_MS,p:P}};
        let v:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e|panic!("GLM53_DEPTH_TABLE {path}: {e}"))).expect("GLM53_DEPTH_TABLE json");
        let f=|x:&serde_json::Value|x.as_f64().expect("GLM53_DEPTH_TABLE number");
        let cost:Vec<f64>=v["cost_ms"].as_array().expect("cost_ms").iter().map(f).collect();
        let rows:Vec<Vec<f64>>=v["p"].as_array().expect("p").iter().map(|r|r.as_array().expect("p row").iter().map(f).collect()).collect();
        assert!(cost.len()==9 && rows.len()==7 && rows.iter().all(|r|r.len()==20 && r.iter().all(|x|(0.0..=1.0).contains(x))),"GLM53_DEPTH_TABLE shape");
        let mut t=Table{lambda:f(&v["lambda"]),cost:[0.;9],p:[[0.;20];7]};
        t.cost.copy_from_slice(&cost);for (i,r) in rows.iter().enumerate() {for (b,x) in r.iter().enumerate() {t.p[i][b]=*x as f32;}}
        assert!(t.lambda>0. && t.lambda.is_finite());
        eprintln!("[depth] rate table from {path}: lambda {:.6}",t.lambda);t
    })
}
pub fn rate_depth(conf:&[f32],max:usize)->usize {rate_depth_scaled(conf,max,1.0)}
fn bin(c:f32)->usize {((c.max(0.)*20.) as usize).min(19)}
/// rate_depth with the table's conditional acceptance probabilities scaled by `scale` (capped at 0.98).
pub fn rate_depth_scaled(conf:&[f32],max:usize,scale:f64)->usize {
    let max=max.min(7).min(conf.len());
    if max<=1 {return max;}
    let t=table();
    let (mut best,mut bd,mut surv,mut e)=(f64::NEG_INFINITY,1,1f64,1f64);
    for d in 1..=max {
        let b=bin(conf[d-1]);
        surv*=if scale==1.0 {t.p[d-1][b] as f64} else {(t.p[d-1][b] as f64*scale).min(0.98)};e+=surv;
        let v=e-t.lambda*t.cost[d+1];
        if v>best {best=v;bd=d;}
    }
    bd
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn rate_depth_bounds(){
        assert_eq!(rate_depth(&[1.;7],7),7);
        assert_eq!(rate_depth(&[0.;7],7),1);
        assert_eq!(rate_depth(&[1.;7],3),3);
        assert_eq!(rate_depth(&[1.;7],1),1);
        // a confident first draft followed by an unconfident tail stops early
        assert!(rate_depth(&[0.99,0.02,0.02,0.02,0.02,0.02,0.02],7)<=2);
    }
    #[test] fn high_acceptance_amortizes_larger_batches(){
        let mut p=Policy::new([120.,150.,180.]);for _ in 0..7{p.observe(7,7);}assert_eq!(p.depth(),7);
    }
    #[test] fn low_acceptance_shortens_but_keeps_exploring(){
        let mut p=Policy::new([120.,150.,180.]);for _ in 0..23{p.observe(1,7);}assert_eq!(p.depth(),3);
        p.observe(1,3);assert_eq!(p.depth(),7);
        for _ in 0..23{p.observe(7,7);}assert_eq!(p.depth(),7);
    }
}
