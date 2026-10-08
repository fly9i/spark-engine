//! Optional view of the accepted prefix of one fixed-chain feature matrix.
//! Rows preserves the legacy cat even with one row; Prefix carries an explicit
//! producer proof, never an inference from adjacent allocation addresses.
use tch::{Kind,Tensor};
use crate::speculative::Node;

pub(crate) fn enabled()->bool {
    match std::env::var("GLM53_ACCEPTED_FEATURE_VIEW") {
        Ok(v)=>match v.as_str(){"0"=>false,"1"=>true,_=>panic!("GLM53_ACCEPTED_FEATURE_VIEW must be 0 or 1")},
        Err(std::env::VarError::NotPresent)=>false,Err(_)=>panic!("GLM53_ACCEPTED_FEATURE_VIEW must be 0 or 1"),
    }
}

pub(crate) fn fixed_chain(nodes:&[Node])->bool {
    (1..=8).contains(&nodes.len()) && nodes.iter().enumerate().all(|(i,n)|n.parent==i.checked_sub(1))
}

/// The existing greedy branch walk, shared by the real selector and CPU
/// contract tests. The visitor receives (node, accepted position); no new
/// index vector or tensor is allocated. Rejection, budget and EOS order stay
/// identical to the legacy serial verifier's acceptance contract.
pub(crate) fn walk_selected(nodes:&[Node],predictions:&[i64],next:i64,budget:usize,stop:&[i64],
    mut visit:impl FnMut(usize,usize))->(Option<usize>,i64) {
    assert_eq!(nodes.len(),predictions.len());
    let mut parent=None;let mut prediction=next;let mut accepted=0;
    while accepted<budget {
        let found=nodes.iter().enumerate().find(|(_,n)|n.parent==parent&&n.token==prediction).map(|(i,_)|i);
        if let Some(i)=found {
            visit(i,accepted);accepted+=1;prediction=predictions[i];parent=Some(i);
            if stop.contains(&nodes[i].token){break;}
        }else{break;}
    }
    (parent,prediction)
}

/// Metadata only; no tensor allocation, copy, dtype conversion or CUDA read.
/// The only optimized case is a single fixed-chain [T,20480] Float matrix.
pub(crate) fn eligible(nodes:&[Node],features:&Tensor)->bool {
    enabled() && fixed_chain(nodes) && features.kind()==Kind::Float && features.device().is_cuda() &&
        features.size()==[nodes.len() as i64,20480] && features.is_contiguous()
}

pub(crate) enum AcceptedFeatures {
    Rows(Vec<Tensor>),
    Prefix(Tensor),
}
impl AcceptedFeatures {
    pub(crate) fn empty()->Self {Self::Rows(Vec::new())}
    pub(crate) fn is_prefix(&self)->bool {matches!(self,Self::Prefix(_))}
    /// Producer passes only after fixed_chain + accepted indices 0..rows have
    /// been checked. Keep narrow's owning Tensor handle until append is queued.
    pub(crate) fn prefix(features:&Tensor,rows:usize)->Self {
        assert!(rows>0 && rows<=8 && features.dim()==2 && rows as i64<=features.size()[0]);
        assert_eq!(features.size()[1],20480);assert_eq!(features.kind(),Kind::Float);assert!(features.is_contiguous());
        assert!(features.device().is_cuda());
        Self::Prefix(features.narrow(0,0,rows as i64))
    }
    /// Rows always retains cat, including serial and non-chain singletons.
    /// Prefix must be read before its producer graph next overwrites storage.
    pub(crate) fn join(&self)->Tensor {
        match self {
            Self::Rows(rows)=>{assert!(!rows.is_empty());Tensor::cat(rows,0)},
            Self::Prefix(view)=>view.shallow_clone(),
        }
    }
}

#[path="feature_prefix_probe.rs"] mod local_probe;
pub(crate) fn probe(drafter:&crate::dflash::Drafter,target:&crate::weights::ModelWeights,out:&std::path::Path) {
    local_probe::run(drafter,target,out);
}

/// Standalone real-drafter consumer gate: only target embedding/head weights
/// are loaded. A resident qualification can instead call probe() directly.
pub(crate) fn run(model:&std::path::Path,draft:&std::path::Path,out:&std::path::Path) {
    tch::set_num_threads(4);let _guard=tch::no_grad_guard();
    crate::tp::init_from_env();let dev=tch::Device::Cuda(0);
    let cfg=crate::config::load(&model.join("config.json")).unwrap();
    let target=crate::weights::ModelWeights::load(model,&cfg,0,dev);
    let drafter=crate::dflash::Drafter::load_target(draft,&target);
    probe(&drafter,&target,out);
}

#[cfg(test)] mod tests {
    use super::{fixed_chain,walk_selected};
    use crate::speculative::{self,Node,Target};
    fn nodes(parents:&[Option<usize>])->Vec<Node> {
        parents.iter().map(|&parent|Node{parent,token:0}).collect()
    }
    #[test] fn only_bounded_fixed_chains_qualify() {
        for n in 1..=8 {assert!(fixed_chain(&nodes(&(0..n).map(|i:usize|i.checked_sub(1)).collect::<Vec<_>>())));}
        for p in [vec![],vec![None,None],vec![None,Some(0),Some(0)],
            vec![None,Some(0),Some(1),Some(1)],(0..9).map(|i:usize|i.checked_sub(1)).collect()] {
            assert!(!fixed_chain(&nodes(&p)));
        }
    }

    // Independent reference: the existing serial verifier forks and executes
    // a stateful target for every node, while our production selector consumes
    // an already computed prediction vector. Checking only a prefix formula
    // would miss EOS's updated next token, sibling choice and None semantics.
    struct Toy;
    impl Target for Toy {
        type State=Vec<i64>;type Aux=Vec<i64>;
        fn fork(&self,state:&Self::State)->Self::State {state.clone()}
        fn step(&mut self,state:&mut Self::State,token:i64)->(i64,Self::Aux) {
            state.push(token);((state.iter().sum::<i64>()+state.len() as i64)%17,state.clone())
        }
    }
    fn check_selection(proposal:&[Node],next:i64,budget:usize,stop:&[i64]) {
        let base=vec![2,3];let mut toy=Toy;let mut states:Vec<Vec<i64>>=Vec::new();let mut predictions=Vec::new();
        for node in proposal {
            let mut state=node.parent.map_or_else(||base.clone(),|p|states[p].clone());
            predictions.push(toy.step(&mut state,node.token).0);states.push(state);
        }
        // Serial topology validation requires a budget at least the maximum
        // proposal depth. Truncate its accepted path independently afterward.
        let serial=speculative::verify_until(&mut toy,&base,next,proposal,proposal.len(),stop);
        let keep=budget.min(serial.tokens.len());let expected=&serial.tokens[..keep];
        let mut expected_state=base.clone();let mut expected_next=next;
        for &token in expected {expected_next=toy.step(&mut expected_state,token).0;}
        let mut visited=Vec::new();
        let (parent,actual_next)=walk_selected(proposal,&predictions,next,budget,stop,|i,pos|{
            assert_eq!(pos,visited.len());visited.push(i);
        });
        assert_eq!(visited.iter().map(|&i|proposal[i].token).collect::<Vec<_>>(),expected);
        assert_eq!(actual_next,expected_next);
        assert_eq!(parent,visited.last().copied());
        assert_eq!(parent.map_or(&base,|p|&states[p]),&expected_state);
        if fixed_chain(proposal) {assert_eq!(visited,(0..keep).collect::<Vec<_>>());}
        assert_eq!(base,vec![2,3]);
    }
    #[test] fn production_selection_matches_serial_at_every_rejection_budget_and_eos() {
        let mut toy=Toy;let mut state=vec![2,3];let mut next=7;let mut gold=Vec::new();
        for _ in 0..8 {gold.push(next);next=toy.step(&mut state,next).0;}
        for count in 1..=8 {for reject in 0..=count {
            let mut proposal=gold[..count].to_vec();if reject<count {proposal[reject]+=97;}
            let proposal=speculative::chain(&proposal);
            for budget in 0..=count+1 {
                check_selection(&proposal,7,budget,&[]);
                for eos in 0..count {check_selection(&proposal,7,budget,&[proposal[eos].token]);}
            }
        }}
    }
    #[test] fn alternate_roots_and_siblings_keep_their_selected_row_order() {
        let proposal=vec![Node{parent:None,token:1},Node{parent:None,token:7},
            Node{parent:Some(0),token:9},Node{parent:Some(1),token:15},
            Node{parent:Some(1),token:0}];
        assert!(!fixed_chain(&proposal));
        for budget in 0..=4 {for stop in [vec![],vec![7],vec![15],vec![99]] {
            check_selection(&proposal,7,budget,&stop);
        }}
        let mut rows=Vec::new();let (parent,next)=walk_selected(&proposal,&[9,15,8,13,1],7,3,&[],|i,_|rows.push(i));
        assert_eq!(rows,vec![1,3]);assert_eq!(parent,Some(3));assert_eq!(next,13);
    }
    #[test] fn empty_or_unmatched_anchor_never_produces_a_prefix() {
        check_selection(&[],7,0,&[]);check_selection(&[],7,8,&[]);
        check_selection(&speculative::chain(&[1,2,3]),7,3,&[]);
        let proposal=speculative::chain(&[7]);
        check_selection(&proposal,7,0,&[]);check_selection(&proposal,7,1,&[7]);
    }
}
