//! Coupled target/drafter prefix cache. Entries belong to these borrowed models;
//! all mutable state is copied on checkout. No HTTP scheduler is implied.
use crate::{forward::{Engine,DecodeStates,snapshot},dflash::{Drafter,Context}};
use tch::Tensor;
struct Entry {ids:Vec<i64>,logits:Tensor,target:DecodeStates,draft:Context,used:u64}
pub struct Session<'a> {
    pub engine:&'a mut Engine,pub drafter:&'a Drafter,entries:Vec<Entry>,slots:usize,
    chunk:usize,clock:u64,signature:Vec<Option<String>>,pub hits:usize,pub misses:usize,pub evictions:usize,
}
pub(crate) fn signature()->Vec<Option<String>> {
    let mut s=crate::session::signature();
    s.extend(["GLM53_DRAFT_NORM_CACHE","GLM53_FP8_SPLITS","GLM53_DRAFT_FP8_SPLITS","GLM53_DRAFT_FP8_MLP","GLM53_DRAFT_FP8_ATTN","GLM53_DRAFT_FP8_CONV","GLM53_DRAFT_FP8_FC","GLM53_DRAFT_FP8_HEAD"].map(|k|std::env::var(k).ok()));
    s.extend(["GLM53_DRAFT_GQA_SHORT_MAX","GLM53_DRAFT_ATTN_ORACLE","GLM53_DRAFT_GQA_PRECISE","GLM53_DRAFT_APPEND_TRIM","GLM53_DRAFT_GQA","GLM53_DRAFT_HEAD_TP","GLM53_DRAFT_HEAD_SHARD_LOAD","GLM53_DRAFT_TOPK_TP","GLM53_DRAFT_KV_BUFFER","GLM53_DRAFT_KV_BUFFER_MIN_CONTEXT","GLM53_DRAFT_ROPE_CACHE","GLM53_DRAFT_MLP_TP","GLM53_DRAFT_MLP_SHARD_LOAD","GLM53_SPEC_GRAPH_SLOTS","GLM53_SPEC_TREE_CAPTURE_AFTER","GLM53_SPEC_TREE_CAPTURE_MIN_REMAINING"].map(|k|std::env::var(k).ok()));s
}
impl<'a> Session<'a> {
    pub fn new(engine:&'a mut Engine,drafter:&'a Drafter,slots:usize,chunk:usize)->Self {
        assert!(chunk>0);Self{engine,drafter,entries:Vec::new(),slots,chunk,clock:0,
            signature:signature(),hits:0,misses:0,evictions:0}
    }
    fn insert(&mut self,ids:&[i64],logits:&Tensor,target:&DecodeStates,draft:&Context) {
        if self.slots==0{return;}
        assert_eq!(draft.len,ids.len() as i64);self.clock+=1;
        let entry=Entry{ids:ids.to_vec(),logits:logits.copy(),target:snapshot(target),draft:draft.snapshot(),used:self.clock};
        if let Some(i)=self.entries.iter().position(|e|e.ids==ids){self.entries[i]=entry;}
        else if self.entries.len()<self.slots{self.entries.push(entry);}
        else {let i=self.entries.iter().enumerate().min_by_key(|(_,e)|e.used).unwrap().0;
            self.entries[i]=entry;self.evictions+=1;}
    }
    pub fn prepare(&mut self,ids:&[i64])->(Tensor,DecodeStates,Context,usize) {
        assert_eq!(signature(),self.signature,"spec session execution configuration changed");
        assert!(!ids.is_empty()&&ids.len()<=crate::mla_latent::capacity() as usize);
        self.clock+=1;
        let best=self.entries.iter().enumerate().filter(|(_,e)|ids.starts_with(&e.ids))
            .max_by_key(|(_,e)|e.ids.len()).map(|(i,_)|i);
        let (mut consumed,mut logits,mut target,mut draft)=if let Some(i)=best {
            self.hits+=1;let e=&mut self.entries[i];e.used=self.clock;
            (e.ids.len(),Some(e.logits.copy()),Some(snapshot(&e.target)),e.draft.snapshot())
        }else {self.misses+=1;(0,None,None,self.drafter.empty_context())};
        let hit=consumed;
        while consumed<ids.len() {
            let end=(consumed+self.chunk).min(ids.len());
            let input=Tensor::from_slice(&ids[consumed..end]).to_device(self.engine.w.device);
            let (l,s,f)=self.engine.prefill_record_last(&input,target.take(),true);
            self.drafter.append(&mut draft,&Tensor::cat(&f,1));
            consumed=end;logits=Some(l.get(l.size()[0]-1));target=Some(s);
            if crate::session::admit_chunk(ids.len()-consumed,self.chunk,self.slots) {
                self.insert(&ids[..consumed],logits.as_ref().unwrap(),target.as_ref().unwrap(),&draft);
            }
        }
        (logits.unwrap(),target.unwrap(),draft,hit)
    }
}
/// Same chunking is required for bitwise comparisons of batched prefill.
pub fn check(engine:&mut Engine,drafter:&Drafter,out:&std::path::Path) {
    let refs:serde_json::Value=serde_json::from_str(&std::fs::read_to_string("bench/m0-refs.json").unwrap()).unwrap();
    let ids:Vec<i64>=refs["hello"]["prompt_ids"].as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).collect();
    let ids=&ids[..ids.len().min(12)];assert!(ids.len()>2);let split=ids.len()/2;
    let mut session=Session::new(engine,drafter,2,split);
    let (base_l,base_s,base_d,hit)=session.prepare(&ids[..split]);assert_eq!(hit,0);
    let (l,s,mut d,hit)=session.prepare(&ids[..split]);assert_eq!(hit,split);
    assert!(l.equal(&base_l));assert_eq!(crate::forward::states_max_diff(&s,&base_s),0.);assert!(d.equal(&base_d));
    let (extended_l,extended_s,extended_d,hit)=session.prepare(&ids[..split+1]);assert_eq!(hit,split);
    // Mutating a checkout must not change either the shorter or longer cache.
    let input=Tensor::from_slice(&[ids[split]]).to_device(session.engine.w.device);
    let (expected_l,expected_s,f)=session.engine.prefill_record(&input,Some(base_s),true);
    session.drafter.append(&mut d,&Tensor::cat(&f,1));
    assert!(expected_l.get(0).equal(&extended_l));assert_eq!(crate::forward::states_max_diff(&expected_s,&extended_s),0.);assert!(d.equal(&extended_d));
    let (_,_,old,hit)=session.prepare(&ids[..split]);assert_eq!(hit,split);assert!(old.equal(&base_d));
    let anchor=extended_l.argmax(-1,false).int64_value(&[]);
    let a=session.drafter.propose(&d,anchor,&session.engine.w);
    let b=session.drafter.propose(&extended_d,anchor,&session.engine.w);
    assert!(a.hidden.equal(&b.hidden)&&a.ids.equal(&b.ids)&&a.edges.equal(&b.edges));assert_eq!(a.path,b.path);
    let different=vec![0;split];let _=session.prepare(&different);assert_eq!(session.evictions,1);
    let (_,_,_,hit)=session.prepare(&ids[..split+1]);assert_eq!(hit,split,"recent shorter prefix survives LRU");
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(out.join(format!("prefix-rank{}.json",crate::tp::world().rank)),serde_json::json!({
        "exact":true,"hits":session.hits,"misses":session.misses,"evictions":session.evictions,
        "checked":"target logits/state and drafter KV/selector; longest prefix; checkout isolation; LRU"}).to_string()).unwrap();
}
