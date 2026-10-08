//! Greedy chain/tree reference verification with branch-local recurrent state.
//! This is the correctness path; serial node evaluation is not a throughput claim.
pub struct Node {pub parent:Option<usize>,pub token:i64}
pub trait Target {
    type State;
    type Aux;
    fn fork(&self,state:&Self::State)->Self::State;
    fn step(&mut self,state:&mut Self::State,token:i64)->(i64,Self::Aux);
}
pub struct Verified<S,A> {pub tokens:Vec<i64>,pub state:S,pub next:i64,pub aux:Vec<A>,pub evaluated:usize}

pub fn verify<T:Target>(target:&mut T,base:&T::State,next:i64,nodes:&[Node],budget:usize)->Verified<T::State,T::Aux> {
    verify_until(target,base,next,nodes,budget,&[])
}
pub fn verify_until<T:Target>(target:&mut T,base:&T::State,next:i64,nodes:&[Node],budget:usize,stop:&[i64])->Verified<T::State,T::Aux> {
    // Validate topology before executing any target work.
    let mut depths=Vec::new();
    for (i,n) in nodes.iter().enumerate() {
        assert!(n.parent.map_or(true,|p|p<i),"tree must be topologically ordered");
        let depth=n.parent.map_or(1,|p|depths[p]+1);assert!(depth<=budget,"proposal exceeds remaining budget");
        assert!(!nodes[..i].iter().any(|p|p.parent==n.parent&&p.token==n.token),"duplicate sibling token");
        depths.push(depth);
    }
    let mut states=Vec::new();let mut predictions=Vec::new();let mut aux=Vec::new();
    for node in nodes {
        let mut state=target.fork(node.parent.map_or(base,|p|&states[p]));
        let (prediction,feature)=target.step(&mut state,node.token);
        states.push(state);predictions.push(prediction);aux.push(Some(feature));
    }
    let mut parent=None;let mut prediction=next;let mut accepted=Vec::new();let mut features=Vec::new();
    while accepted.len()<budget {
        let found=nodes.iter().enumerate().find(|(_,n)|n.parent==parent&&n.token==prediction).map(|(i,_)|i);
        if let Some(i)=found {accepted.push(nodes[i].token);features.push(aux[i].take().unwrap());prediction=predictions[i];parent=Some(i);
            if stop.contains(&nodes[i].token){break;}}
        else {break;}
    }
    let state=target.fork(parent.map_or(base,|p|&states[p]));
    Verified{tokens:accepted,state,next:prediction,aux:features,evaluated:nodes.len()}
}

pub fn chain(tokens:&[i64])->Vec<Node> {tokens.iter().enumerate().map(|(i,&token)|Node{parent:i.checked_sub(1),token}).collect()}

#[cfg(test)]
mod tests {
    use super::*;
    struct Toy;
    impl Target for Toy {
        type State=Vec<i64>;type Aux=Vec<i64>;
        fn fork(&self,s:&Vec<i64>)->Vec<i64>{s.clone()}
        fn step(&mut self,s:&mut Vec<i64>,token:i64)->(i64,Vec<i64>){s.push(token);((s.iter().sum::<i64>()+s.len() as i64)%11,s.clone())}
    }
    #[test]
    fn rejection_at_every_depth_restores_exact_committed_prefix() {
        let base=vec![2,3];let mut toy=Toy;let mut state=base.clone();let mut next=7;let mut good=Vec::new();
        for _ in 0..7 {good.push(next);next=toy.step(&mut state,next).0;}
        for reject in 0..=7 {
            let mut proposal=good.clone();if reject<7{proposal[reject]=(proposal[reject]+1)%11;}
            let v=verify(&mut toy,&base,7,&chain(&proposal),7);
            assert_eq!(v.tokens,good[..reject]);assert_eq!(v.state,[base.clone(),good[..reject].to_vec()].concat());
            let expected=if reject<7{good[reject]}else{next};assert_eq!(v.next,expected);
            assert_eq!(base,vec![2,3]);
        }
    }
    #[test]
    fn sibling_recurrences_do_not_leak_and_alternate_branch_commits() {
        let mut toy=Toy;let base=vec![2,3];
        let nodes=vec![Node{parent:None,token:1},Node{parent:None,token:7},
            Node{parent:Some(0),token:9},Node{parent:Some(1),token:4},Node{parent:Some(1),token:0}];
        let v=verify(&mut toy,&base,7,&nodes,2);
        assert_eq!(v.tokens,vec![7,4]);assert_eq!(v.state,vec![2,3,7,4]);
        assert_eq!(v.aux,vec![vec![2,3,7],vec![2,3,7,4]]);assert_eq!(v.next,9);
    }
    #[test]
    fn zero_budget_and_empty_tree_leave_state_unchanged() {
        let mut toy=Toy;let v=verify(&mut toy,&vec![3],8,&[],0);
        assert_eq!(v.next,8);assert_eq!(v.state,vec![3]);assert!(v.tokens.is_empty());
    }
    #[test]
    fn eos_commits_only_through_the_stop_token() {
        let mut toy=Toy;let v=verify_until(&mut toy,&vec![2,3],7,&chain(&[7,4,9]),3,&[4]);
        assert_eq!(v.tokens,vec![7,4]);assert_eq!(v.state,vec![2,3,7,4]);assert_eq!(v.next,9);
    }
}
