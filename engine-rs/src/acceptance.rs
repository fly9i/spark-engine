//! Strict token comparison; numerical diagnostics never change this verdict.
pub fn compare(out: &[i64], reference: &[i64], requested: usize) -> (Option<usize>, usize, bool) {
    let matched = out.iter().zip(reference).take_while(|(a,b)| a == b).count();
    let first = (matched < out.len().min(reference.len())).then_some(matched);
    let required = requested.min(reference.len());
    let pass = !reference.is_empty() && first.is_none()
        && matched >= required && matched >= reference.len().min(12);
    (first, matched, pass)
}

#[cfg(test)]
mod tests {
    use super::compare;
    #[test]
    fn mismatch_after_twelve_never_passes() {
        let r: Vec<i64> = (0..24).collect(); let mut o = r.clone(); o[16]=999;
        assert_eq!(compare(&o,&r,24),(Some(16),16,false));
    }
    #[test]
    fn early_stop_does_not_satisfy_requested_span() {
        let r: Vec<i64> = (0..24).collect();
        assert!(!compare(&r[..12],&r,24).2);
        assert!(compare(&r[..12],&r,12).2);
        assert!(compare(&r,&r,24).2);
        assert!(!compare(&[],&[],0).2);
    }
}
