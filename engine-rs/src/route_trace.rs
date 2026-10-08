//! Opt-in graph output taps for actual speculative routes. No CPU copies during
//! capture, no instrumentation in the default path. Diagnostic runs are not timings.
use std::cell::RefCell;
use tch::{Tensor,Device};
thread_local! {static TAPS:RefCell<Option<Vec<(usize,Tensor)>>>=RefCell::new(None);}
pub fn begin(){TAPS.with(|v|*v.borrow_mut()=Some(Vec::new()));}
pub fn record(layer:usize,ids:&Tensor){TAPS.with(|v|{if let Some(v)=v.borrow_mut().as_mut(){v.push((layer,ids.shallow_clone()));}});}
pub fn take()->Vec<(usize,Tensor)>{TAPS.with(|v|v.borrow_mut().take().unwrap_or_default())}
pub fn save(taps:&[(usize,Tensor)],path:&std::path::Path) {
    let data:Vec<_>=taps.iter().map(|(layer,ids)|(format!("layer_{layer}"),ids.to_device(Device::Cpu))).collect();
    Tensor::save_multi(&data,path).unwrap();
}
