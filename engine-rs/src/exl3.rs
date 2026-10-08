// SPDX-License-Identifier: MIT
//! EXL3/MCG 解码(Rust 参考实现)。语义与 engine/glm53/exl3.py 一致,验收 = golden hash。

use rayon::prelude::*;

use crate::nibmap::NIBMAP;
use half::f16;

const MCG_MULTIPLIER: u32 = 0xCBAC_1FED;
const LOP3_B: u32 = 0x8FFF_8FFF;
const LOP3_C: u32 = 0x3B60_3B60;

/// 全码本:65536 个 16 位词 → fp16 位模式(带 fp16 加法舍入,与 numpy 路径一致)。
pub fn mcg_table() -> [u16; 65536] {
    let mut t = [0u16; 65536];
    for (w, slot) in t.iter_mut().enumerate() {
        let w = w as u32;
        let p = w.wrapping_mul(MCG_MULTIPLIER);
        let y = (p & LOP3_B) ^ LOP3_C;
        let lo = f16::from_bits((y & 0xFFFF) as u16);
        let hi = f16::from_bits((y >> 16) as u16);
        *slot = f16::from_f32(lo.to_f32() + hi.to_f32()).to_bits();
    }
    t
}

/// trellis int16[Kt*64 × Nt](tile 行主序:外层 k,内层 n)→ inner fp16 位模式
/// 的行主序 [Kt*16, Nt*16](K-major)。tile(k,n) 位于 [k*16+r, n*16+c]。
///
/// 输入布局:trellis 原始张量 [Kt, Nt, 64] 展平;每 tile 64 个 i16。
pub fn decode_inner(trellis: &[i16], kt: usize, nt: usize, f: &[u16; 65536]) -> Vec<u16> {
    assert_eq!(trellis.len(), kt * nt * 64, "trellis 尺寸不匹配");
    let rows = kt * 16;
    let cols = nt * 16;
    let mut out = vec![0u16; rows * cols];
    let mut u32s = [0u32; 32];
    let mut nib = [0u32; 256];
    for k in 0..kt {
        for n in 0..nt {
            let tile = &trellis[(k * nt + n) * 64..(k * nt + n + 1) * 64];
            for j in 0..32 {
                u32s[j] = (tile[2 * j] as u32 & 0xFFFF) | ((tile[2 * j + 1] as u32 & 0xFFFF) << 16);
            }
            for p in 0..256 {
                nib[p] = (u32s[p / 8] >> (28 - 4 * (p % 8))) & 0xF;
            }
            for r in 0..16usize {
                for c in 0..16usize {
                    let s = &NIBMAP[r][c];
                    let word = nib[s[0] as usize]
                        | (nib[s[1] as usize] << 4)
                        | (nib[s[2] as usize] << 8)
                        | (nib[s[3] as usize] << 12);
                    out[(k * 16 + r) * cols + (n * 16 + c)] = f[word as usize];
                }
            }
        }
    }
    out
}

/// 并行版:tile 级 rayon 并行(16384 tile 独立,自调度)。
/// 位级语义与串行版完全一致(golden hash 锁定)。
pub fn decode_inner_par(trellis: &[i16], kt: usize, nt: usize, f: &[u16; 65536]) -> Vec<u16> {
    assert_eq!(trellis.len(), kt * nt * 64, "trellis 尺寸不匹配");
    let rows = kt * 16;
    let cols = nt * 16;
    let mut out = vec![0u16; rows * cols];
    out.par_chunks_mut(cols * 16)            // 每 k 一个条带(16 行)
        .enumerate()
        .for_each(|(k, band)| {
            let mut u32s = [0u32; 32];
            let mut nib = [0u32; 256];
            for n in 0..nt {
                let tile = &trellis[(k * nt + n) * 64..(k * nt + n + 1) * 64];
                for j in 0..32 {
                    u32s[j] = (tile[2 * j] as u32 & 0xFFFF) | ((tile[2 * j + 1] as u32 & 0xFFFF) << 16);
                }
                for p in 0..256 {
                    nib[p] = (u32s[p / 8] >> (28 - 4 * (p % 8))) & 0xF;
                }
                for r in 0..16usize {
                    for c in 0..16usize {
                        let s = &NIBMAP[r][c];
                        let word = nib[s[0] as usize]
                            | (nib[s[1] as usize] << 4)
                            | (nib[s[2] as usize] << 8)
                            | (nib[s[3] as usize] << 12);
                        band[r * cols + (n * 16 + c)] = f[word as usize];
                    }
                }
            }
        });
    out
}
