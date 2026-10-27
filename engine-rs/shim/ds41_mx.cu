// SPDX-License-Identifier: MIT
// DeepSeek-V4.1-Flash kernels on the native block-scaled tensor-core MMA of GB10 (sm_121a; docs/07 D4):
//   mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0
// A = activations as E4M3 bytes with one UE8M0 scale per 32 along K (exactly the official act_quant), B = the routed
// experts' MXFP4 (E2M1, one UE8M0 per 32 along K, as stored), FP32 accumulation. One instruction covers one 32-wide
// scale block of both operands, so every product and scale is applied exactly; only the summation order differs from
// the dequantize + FP32 reference (L1). Measured on GB10: an E2M1 operand sits in bits [5:2] of its byte container;
// scale threads: A's row of lane l is l/4 + 8 (l & 1), B's column of lane l is l/4 (CUTLASS SM120 SFA/SFB layouts).
//
// Routed experts (ds41_moe_mx): the semantics of ds41_moe.cu (grouping by expert, gate/up + SwiGLU + act_quant, down,
// slot-order combine) with these operands:
// - weights in the "mx" tile layout (kern::moe_pack_mx): per tile of 8 rows and 128-wide K step 512 bytes, lane (g, q)
//   of the warp reads 16 contiguous bytes: word b = bytes [16 b + 4 q, +4) of row g's 64-byte step chunk (32-block b,
//   K positions 8 q .. 8 q + 7 of it); scales per tile and step: 8 rows x 4 blocks (lane reads its row's word);
// - in a 32-block, the lane's 8 K positions split into the even ones (low nibbles: B fragment register 0, A bytes
//   0, 2, 4, 6) and the odd ones (high nibbles: register 1, A bytes 1, 3, 5, 7): A and B use the same K mapping;
// - x and the intermediate h as E4M3 bytes + UE8M0 bytes per 32 (ds41_act_quant_fp8; up's epilogue writes h so).
#include <map>
#include <tuple>
#include "ds41_pdl.cuh"
#include "ds41_hc_job.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace {
constexpr int EMAX = 384, H = 5120;

__device__ __forceinline__ void mma_mx(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
                                       uint32_t sfa, uint32_t sfb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }
// UE8M0 scale byte and E4M3 byte of act_quant (kernel.py: amax >= 1e-4, scale = 2^ceil(log2(amax / 448)))
__device__ __forceinline__ uint32_t aq_scale(float amax) {
  amax = fmaxf(amax, 1e-4f);
  const unsigned bits = __float_as_uint(amax * (1.0f / 448.0f));
  return (unsigned)((int)((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0));   // biased exponent = UE8M0 byte
}
__device__ __forceinline__ uint8_t aq_byte(float v, uint32_t e) {
  const float s = __uint_as_float(e << 23);
  const __nv_fp8_e4m3 f8(fminf(fmaxf(v / s, -448.f), 448.f));
  return f8.__x;
}
// A fragment of one 32-block: the lane's 8 bytes (K 8q .. 8q+7) split into even / odd positions
__device__ __forceinline__ void a_split(uint2 v, uint32_t& even, uint32_t& odd) {
  even = __byte_perm(v.x, v.y, 0x6420);
  odd = __byte_perm(v.x, v.y, 0x7531);
}
// B fragment of one 32-block from the lane's packed word (8 E2M1): low nibbles (even K) and high nibbles (odd K), in
// bits [5:2] of their bytes
__device__ __forceinline__ void b_split(uint32_t w, uint32_t& even, uint32_t& odd) {
  even = (w & 0x0F0F0F0Fu) << 2;
  odd = ((w >> 4) & 0x0F0F0F0Fu) << 2;
}

// ---- Routed-expert scales in a 2-bit coding (DS41_MX_S2, default on; lossless, the MMA gets the same scale bytes).
// The checkpoint's expert E8M0 scales are nearly constant (99.9% are 120 or 121; within a row of a rank's slice the
// span exceeds 3 for ~4e-5 of the rows), so per tile (8 weight rows) the coding keeps a base byte per row and per
// 128-wide K step one byte of four 2-bit offsets per row (one per 32-block): 8 bytes per tile and step instead of 32
// (expert bytes read -4.4%, scale memory / 4). A tile with any row spanning more than 3 keeps its raw words.
// Blob for T tiles of S steps (kern::moe_scales_s2): desc [T] x 16 bytes (bytes 0-7 the rows' bases, 8-11 the raw
// index (int32, -1: coded)), codes [T][S][8 rows] (byte = c0 | c1 << 2 | c2 << 4 | c3 << 6), raw [n_raw][S][8][4] (the
// raw layout of those tiles). The word of 4 scales of a row (block b in byte b) = base * 0x01010101 + spread(code).
struct ScaleV { const uint8_t* raw; const uint8_t* code; const uint4* desc; };
inline ScaleV scale_v(const void* s, bool c2, long long tiles, int steps) {
  const uint8_t* b = (const uint8_t*)s;
  if (!c2) return ScaleV{b, nullptr, nullptr};
  return ScaleV{b + tiles * 16 + tiles * steps * 8, b + tiles * 16, (const uint4*)b};
}
__device__ __forceinline__ uint32_t s2_spread(uint32_t c) { return (c & 3u) | ((c & 0xCu) << 6) | ((c & 0x30u) << 12) | ((c & 0xC0u) << 18); }
// where row g's scale words of a tile come from, steps from s0 on: raw (p: words, 32 bytes per step) or coded (p: code
// bytes, 8 per step; bmul = base * 0x01010101)
struct SSrc { const uint8_t* p; uint32_t bmul; bool raw; };
template <bool C2> __device__ __forceinline__ SSrc s_src(const ScaleV& v, long long tile, int steps, int s0, int g) {
  if (!C2) return SSrc{v.raw + (tile * steps + s0) * 32 + g * 4, 0u, true};
  const uint4 d = __ldg(v.desc + tile);
  const int ri = (int)d.z;
  if (ri >= 0) return SSrc{v.raw + ((long long)ri * steps + s0) * 32 + g * 4, 0u, true};
  return SSrc{v.code + (tile * steps + s0) * 8 + g, (((g < 4 ? d.x : d.y) >> (8 * (g & 3))) & 0xFFu) * 0x01010101u, false};
}
__device__ __forceinline__ uint32_t s_word(const SSrc& c, int s) {
  return c.raw ? __ldg(reinterpret_cast<const uint32_t*>(c.p + s * 32)) : c.bmul + s2_spread(__ldg(c.p + s * 8));
}

struct Item { int e, off, n, pad; };

__global__ __launch_bounds__(1024) void group_kernel(const int64_t* __restrict__ idx, int P, int GR, int E, int* __restrict__ G,
                                                    Item* __restrict__ items, int* __restrict__ pairmap, int* __restrict__ prow) {
  pdl_entry();
  __shared__ int cnt[EMAX], fill[EMAX], eoff[EMAX];
  for (int e = threadIdx.x; e < E; e += blockDim.x) { cnt[e] = 0; fill[e] = 0; }
  __syncthreads();
  for (int p = threadIdx.x; p < P; p += blockDim.x) atomicAdd(&cnt[(int)idx[p]], 1);
  __syncthreads();
  {
    typedef cub::BlockScan<int, 1024> Scan;
    __shared__ typename Scan::TempStorage tmp;
    const int e = threadIdx.x;
    const int c = e < E ? cnt[e] : 0, ni = (c + GR - 1) / GR;
    int off, gi, total;
    Scan(tmp).ExclusiveSum(c, off);
    __syncthreads();
    Scan(tmp).ExclusiveSum(ni, gi, total);
    if (e < E) {
      eoff[e] = off;
      for (int j = 0; j < ni; ++j) items[gi + j] = Item{e, off + j * GR, min(GR, c - j * GR), 0};
    }
    if (e == 0) *G = total;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < P; p += blockDim.x) {
    const int e = (int)idx[p];
    const int pos = eoff[e] + atomicAdd(&fill[e], 1);
    pairmap[p] = pos;
    prow[pos] = p;
  }
}

// gate/up + SwiGLU + act_quant -> h E4M3 [P, I] + scales [P, I/32]. 4 warps, MT m16 tiles; KS = 1: grid (I / 128, Gmax),
// warp = 32 columns over all of K; KS = 4 (decode): grid (I / 32, Gmax), warps split K, warp 0 sums in warp order.
template <int MT, int KS, bool C2>
__global__ __launch_bounds__(KS > 4 ? 32 * KS : 128) void up_mx(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs, const uint8_t* __restrict__ w1,
    const ScaleV s1, const uint8_t* __restrict__ w3, const ScaleV s3, int I,
    const float* __restrict__ rw, const int* __restrict__ G, const Item* __restrict__ items, const int* __restrict__ prow,
    uint8_t* __restrict__ hq, uint8_t* __restrict__ hs, int topk) {
  pdl_entry();
  if ((int)blockIdx.y >= *G) return;
  const Item it = items[blockIdx.y];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int j0 = KS == 1 ? blockIdx.x * 128 + warp * 32 : blockIdx.x * 32;
  constexpr int ALL = H / 128, STEPS = ALL / KS;
  const int s0 = KS == 1 ? 0 : warp * STEPS;
  const uint8_t* wr[8]; SSrc sr[8];
  for (int t = 0; t < 8; ++t) {
    const long long tile = ((long long)it.e * I + j0 + 8 * (t & 3)) / 8;
    wr[t] = (t < 4 ? w1 : w3) + (tile * ALL + s0) * 512 + (g * 4 + q) * 16;
    sr[t] = s_src<C2>(t < 4 ? s1 : s3, tile, ALL, s0, g);
  }
  // A rows (source token of the pair; rows past the item read row 0 and are not stored), and the scale row this lane
  // provides for each m tile
  const uint8_t* xa[MT][2]; const uint8_t* xsa[MT];
  for (int m = 0; m < MT; ++m) {
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m * 16 + hh * 8 + g;
      xa[m][hh] = xq + (long long)(r < it.n ? prow[it.off + r] / topk : 0) * H + s0 * 128 + 8 * q;
    }
    const int rs = m * 16 + (lane >> 2) + 8 * (lane & 1);
    xsa[m] = xs + (long long)(rs < it.n ? prow[it.off + rs] / topk : 0) * (H / 32) + s0 * 4;
  }
  float acc[MT][8][4];
  for (int m = 0; m < MT; ++m) for (int t = 0; t < 8; ++t) for (int c = 0; c < 4; ++c) acc[m][t][c] = 0.f;
  uint4 bw[8]; uint32_t bs[8];
  for (int t = 0; t < 8; ++t) { bw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t])); bs[t] = s_word(sr[t], 0); }
  constexpr bool PF = MT <= 2;   // prefetch the next step's weights (registers allow it up to 2 m tiles)
  for (int s = 0; s < STEPS; ++s) {
    uint4 nw[PF ? 8 : 1]; uint32_t ns[PF ? 8 : 1];
    if (PF && s + 1 < STEPS)
      for (int t = 0; t < 8; ++t) { nw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t] + (s + 1) * 512)); ns[t] = s_word(sr[t], s + 1); }
    if (!PF && s > 0)
      for (int t = 0; t < 8; ++t) { bw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t] + s * 512)); bs[t] = s_word(sr[t], s); }
    uint32_t sfa4[MT];
    for (int m = 0; m < MT; ++m) sfa4[m] = __ldg(reinterpret_cast<const uint32_t*>(xsa[m] + s * 4));
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      uint32_t a[MT][4];
      for (int m = 0; m < MT; ++m) {
        const uint2 r0 = __ldg(reinterpret_cast<const uint2*>(xa[m][0] + s * 128 + b * 32));
        const uint2 r1 = __ldg(reinterpret_cast<const uint2*>(xa[m][1] + s * 128 + b * 32));
        a_split(r0, a[m][0], a[m][2]);
        a_split(r1, a[m][1], a[m][3]);
      }
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const uint32_t w = b == 0 ? bw[t].x : b == 1 ? bw[t].y : b == 2 ? bw[t].z : bw[t].w;
        uint32_t b0, b1;
        b_split(w, b0, b1);
        const uint32_t sfb = (bs[t] >> (8 * b)) & 0xFF;
        for (int m = 0; m < MT; ++m) mma_mx(acc[m][t], a[m][0], a[m][1], a[m][2], a[m][3], b0, b1, (sfa4[m] >> (8 * b)) & 0xFF, sfb);
      }
    }
    if (PF && s + 1 < STEPS) for (int t = 0; t < 8; ++t) { bw[t] = nw[t]; bs[t] = ns[t]; }
  }
  if (KS > 1) {
    __shared__ float red[KS > 1 ? KS - 1 : 1][MT * 32][32];
    if (warp > 0)
      for (int m = 0; m < MT; ++m) for (int t = 0; t < 8; ++t) for (int c = 0; c < 4; ++c) red[warp - 1][(m * 8 + t) * 4 + c][lane] = acc[m][t][c];
    __syncthreads();
    if (warp > 0) return;
    for (int w = 0; w < KS - 1; ++w)
      for (int m = 0; m < MT; ++m) for (int t = 0; t < 8; ++t) for (int c = 0; c < 4; ++c) acc[m][t][c] += red[w][(m * 8 + t) * 4 + c][lane];
  }
  for (int m = 0; m < MT; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m * 16 + hh * 8 + g;
      const float wgt = r < it.n ? rw[prow[it.off + r]] : 0.f;
      float v[8], amax = 0.f;
      for (int t = 0; t < 4; ++t)
        for (int c = 0; c < 2; ++c) {
          const float gt = fminf(bf16r(acc[m][t][hh * 2 + c]), 10.f);
          const float up = fminf(fmaxf(bf16r(acc[m][t + 4][hh * 2 + c]), -10.f), 10.f);
          const float y = bf16r(gt / (1.f + expf(-gt)) * up * wgt);
          v[t * 2 + c] = y;
          amax = fmaxf(amax, fabsf(y));
        }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      const uint32_t e = aq_scale(amax);
      if (r < it.n) {
        uint8_t* hr = hq + (long long)(it.off + r) * I + j0 + 2 * q;
        for (int t = 0; t < 4; ++t)
          *reinterpret_cast<uint16_t*>(hr + 8 * t) = (uint16_t)aq_byte(v[2 * t], e) | ((uint16_t)aq_byte(v[2 * t + 1], e) << 8);
        if (q == 0) hs[(long long)(it.off + r) * (I / 32) + j0 / 32] = (uint8_t)e;
      }
    }
}

// down: out [P, H] FP32 = h . w2_e^T. grid (H / 128, Gmax), 4 warps of 32 output columns.
template <int MT, bool C2>
__global__ __launch_bounds__(128) void down_mx(const uint8_t* __restrict__ hq, const uint8_t* __restrict__ hs, const uint8_t* __restrict__ w2,
    const ScaleV s2, int I, const int* __restrict__ G, const Item* __restrict__ items, float* __restrict__ out) {
  pdl_entry();
  const int bi = blockIdx.y, bc = blockIdx.x;
  if (bi >= *G) return;
  const Item it = items[bi];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int n0 = bc * 128 + warp * 32;
  const int steps = I / 128;
  const uint8_t* wr[4]; SSrc sr[4];
  for (int t = 0; t < 4; ++t) {
    const long long tile = ((long long)it.e * H + n0 + 8 * t) / 8;
    wr[t] = w2 + tile * steps * 512 + (g * 4 + q) * 16;
    sr[t] = s_src<C2>(s2, tile, steps, 0, g);
  }
  const uint8_t* ha[MT][2]; const uint8_t* hsa[MT];
  for (int m = 0; m < MT; ++m) {
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m * 16 + hh * 8 + g;
      ha[m][hh] = hq + (long long)(it.off + (r < it.n ? r : 0)) * I + 8 * q;
    }
    const int rs = m * 16 + (lane >> 2) + 8 * (lane & 1);
    hsa[m] = hs + (long long)(it.off + (rs < it.n ? rs : 0)) * (I / 32);
  }
  float acc[MT][4][4];
  for (int m = 0; m < MT; ++m) for (int t = 0; t < 4; ++t) for (int c = 0; c < 4; ++c) acc[m][t][c] = 0.f;
  // the next step's weights load while this step computes (MT <= 2; registers)
  constexpr bool PF = MT <= 2;
  uint4 bw[4]; uint32_t bs[4];
  for (int t = 0; t < 4; ++t) { bw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t])); bs[t] = s_word(sr[t], 0); }
  for (int s = 0; s < steps; ++s) {
    uint4 nw[PF ? 4 : 1]; uint32_t ns[PF ? 4 : 1];
    if (PF && s + 1 < steps)
      for (int t = 0; t < 4; ++t) { nw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t] + (s + 1) * 512)); ns[t] = s_word(sr[t], s + 1); }
    if (!PF && s > 0)
      for (int t = 0; t < 4; ++t) { bw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t] + s * 512)); bs[t] = s_word(sr[t], s); }
    uint32_t sfa4[MT];
    for (int m = 0; m < MT; ++m) sfa4[m] = *reinterpret_cast<const uint32_t*>(hsa[m] + s * 4);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      uint32_t a[MT][4];
      for (int m = 0; m < MT; ++m) {
        const uint2 r0 = *reinterpret_cast<const uint2*>(ha[m][0] + s * 128 + b * 32);
        const uint2 r1 = *reinterpret_cast<const uint2*>(ha[m][1] + s * 128 + b * 32);
        a_split(r0, a[m][0], a[m][2]);
        a_split(r1, a[m][1], a[m][3]);
      }
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const uint32_t w = b == 0 ? bw[t].x : b == 1 ? bw[t].y : b == 2 ? bw[t].z : bw[t].w;
        uint32_t b0, b1;
        b_split(w, b0, b1);
        const uint32_t sfb = (bs[t] >> (8 * b)) & 0xFF;
        for (int m = 0; m < MT; ++m) mma_mx(acc[m][t], a[m][0], a[m][1], a[m][2], a[m][3], b0, b1, (sfa4[m] >> (8 * b)) & 0xFF, sfb);
      }
    }
    if (PF && s + 1 < steps) for (int t = 0; t < 4; ++t) { bw[t] = nw[t]; bs[t] = ns[t]; }
  }
  for (int m = 0; m < MT; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m * 16 + hh * 8 + g;
      if (r >= it.n) continue;
      float* o = out + (long long)(it.off + r) * H + n0 + 2 * q;
      for (int t = 0; t < 4; ++t) *reinterpret_cast<float2*>(o + 8 * t) = make_float2(acc[m][t][hh * 2], acc[m][t][hh * 2 + 1]);
    }
}

// (columns [c0, c0 + hn) of y; out holds those columns of every pair, hn per row)
__global__ void combine_kernel(const float* __restrict__ out, const int* __restrict__ pairmap, const float* __restrict__ add,
                               float* __restrict__ y, int R, int topk, int c0, int hn) {
  pdl_entry();
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)R * (hn / 4)) return;
  const int r = (int)(i / (hn / 4)), c = c0 + (int)(i % (hn / 4)) * 4;
  float4 acc = add ? *reinterpret_cast<const float4*>(add + (long long)r * H + c) : make_float4(0.f, 0.f, 0.f, 0.f);
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = 0; k < topk; ++k) {
    const float4 v = *reinterpret_cast<const float4*>(out + (long long)pairmap[r * topk + k] * hn + (c - c0));
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  acc.x += s.x; acc.y += s.y; acc.z += s.z; acc.w += s.w;
  *reinterpret_cast<float4*>(y + (long long)r * H + c) = acc;
}

// act_quant of BF16 rows into E4M3 bytes + UE8M0 bytes per 32 (one warp per block of 32)
__global__ void act_quant_fp8_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ q, uint8_t* __restrict__ s, long long blocks) {
  pdl_entry();
  const long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (b >= blocks) return;
  const float v = __bfloat162float(x[b * 32 + lane]);
  float a = fabsf(v);
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const uint32_t e = aq_scale(a);
  q[b * 32 + lane] = aq_byte(v, e);
  if (lane == 0) s[b] = (uint8_t)e;
}

// act_quant of rows stored as E4M3 bytes + E8M0 bytes per 32 (the engram tables): the value is the stored byte times
// the scale built from its exponent bits (ops::e8m0: byte 0 gives 0), rounded to BF16 as engram::upload's dequantize
// did, then act_quant_fp8_kernel's scale and byte. The prefill GEMM gets the operands act_quant gives the dequantized
// BF16 rows, without that FP32 / BF16 round trip through memory (26 MB read + written per 4K-token engram layer
// instead of ~450 MB).
__global__ void requant_fp8_kernel(const uint8_t* __restrict__ wq, const uint8_t* __restrict__ ws, uint8_t* __restrict__ q,
                                   uint8_t* __restrict__ s, long long blocks) {
  pdl_entry();
  const long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (b >= blocks) return;
  __nv_fp8_e4m3 f;
  f.__x = wq[b * 32 + lane];
  const float v = __bfloat162float(__float2bfloat16((float)f * __uint_as_float((uint32_t)ws[b] << 23)));
  float a = fabsf(v);
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const uint32_t e = aq_scale(a);
  q[b * 32 + lane] = aq_byte(v, e);
  if (lane == 0) s[b] = (uint8_t)e;
}

// rows per work item: 16 for decode / verify, 32 otherwise

// ---- FP8 GEMM on the block-scaled MMA (prefill linears): y [M, N] FP32 = A . W^T with A E4M3 [M, K] + UE8M0 [M, K/32]
// (act_quant of the input) and W E4M3 [N, K] + E8M0 per 32x32 block [ceil(N/32), K/32] (the checkpoint's FP8 linears).
// CTA 128 x 128, 8 warps as 2 (M) x 4 (N), warp 64 x 32 = 4 m16 x 4 n8 tiles; K in steps of 64 (two scale blocks)
// through shared memory, cp.async double-buffered. Per 32-block the same even / odd K split as the experts (A and B
// bytes through __byte_perm), so every product and scale is exact; FP32 accumulation.
__device__ __forceinline__ void mma_e4(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
                                       uint32_t sfa, uint32_t sfb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}
__device__ __forceinline__ void cp16(void* smem, const void* g, bool ok) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(a), "l"(g), "r"(ok ? 16 : 0));
}
constexpr int GM = 128, GN = 128, GK = 64, GP = GK + 16;   // tile, K step, padded smem row (bytes)
// B16: y is BF16 (each FP32 sum rounded to nearest, as converting the FP32 output afterwards: same bits, one pass less)
template <bool B16>
__global__ __launch_bounds__(256) void fp8_gemm_mx(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ as_,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, void* __restrict__ yv, int M, int N, int K) {
  float* y = reinterpret_cast<float*>(yv);
  __shared__ __align__(16) uint8_t sa[2][GM * GP], sb[2][GN * GP];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.x * GM, n0 = blockIdx.y * GN;
  const int nk = K / GK, kb = K / 32;
  auto load = [&](int stage, int k0) {
    // 128 rows x 64 bytes per operand = 512 chunks of 16 bytes: 2 per thread each
    for (int i = tid; i < GM * GK / 16; i += 256) {
      const int r = i >> 2, c = (i & 3) * 16;
      const int gm = m0 + r, gn = n0 + r;
      cp16(&sa[stage][r * GP + c], aq + (long long)(gm < M ? gm : 0) * K + k0 + c, gm < M);
      cp16(&sb[stage][r * GP + c], w + (long long)(gn < N ? gn : 0) * K + k0 + c, gn < N);
    }
    asm volatile("cp.async.commit_group;\n");
  };
  float acc[4][4][4];
  for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  // this lane's scale rows / columns
  int arow[4]; for (int mt = 0; mt < 4; ++mt) arow[mt] = min(m0 + wm * 64 + mt * 16 + (lane >> 2) + 8 * (lane & 1), M - 1);
  int bcol[4]; for (int nt = 0; nt < 4; ++nt) bcol[nt] = min(n0 + wn * 32 + nt * 8 + (lane >> 2), N - 1);
  load(0, 0);
  for (int ks = 0; ks < nk; ++ks) {
    const int st = ks & 1;
    if (ks + 1 < nk) { load(st ^ 1, (ks + 1) * GK); asm volatile("cp.async.wait_group 1;\n"); }
    else asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const int kblk = ks * 2 + b;
      uint32_t a[4][4];
      for (int mt = 0; mt < 4; ++mt) {
        const uint8_t* p0 = &sa[st][(wm * 64 + mt * 16 + g) * GP + b * 32 + 8 * q];
        const uint2 r0 = *reinterpret_cast<const uint2*>(p0), r1 = *reinterpret_cast<const uint2*>(p0 + 8 * GP);
        a_split(r0, a[mt][0], a[mt][2]);
        a_split(r1, a[mt][1], a[mt][3]);
      }
      uint32_t sfa[4]; for (int mt = 0; mt < 4; ++mt) sfa[mt] = __ldg(as_ + (long long)arow[mt] * kb + kblk);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const uint2 rb = *reinterpret_cast<const uint2*>(&sb[st][(wn * 32 + nt * 8 + g) * GP + b * 32 + 8 * q]);
        uint32_t b0, b1;
        a_split(rb, b0, b1);
        const uint32_t sfb = __ldg(ws + (long long)(bcol[nt] >> 5) * kb + kblk);
        for (int mt = 0; mt < 4; ++mt) mma_e4(acc[mt][nt], a[mt][0], a[mt][1], a[mt][2], a[mt][3], b0, b1, sfa[mt], sfb);
      }
    }
    __syncthreads();
  }
  for (int mt = 0; mt < 4; ++mt)
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m0 + wm * 64 + mt * 16 + hh * 8 + g;
      if (r >= M) continue;
      for (int nt = 0; nt < 4; ++nt) {
        const int c = n0 + wn * 32 + nt * 8 + 2 * q;
        if (c < N) {
          if (B16) *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(yv) + (long long)r * N + c) = __floats2bfloat162_rn(acc[mt][nt][hh * 2], acc[mt][nt][hh * 2 + 1]);
          else *reinterpret_cast<float2*>(y + (long long)r * N + c) = make_float2(acc[mt][nt][hh * 2], acc[mt][nt][hh * 2 + 1]);
        }
      }
    }
}

// fp8_gemm_mx with the same MMA sequence per output (32-blocks in K order, the same fragments and scales: bitwise
// equal) and: the scales one K step ahead in registers (two adjacent UE8M0 bytes per row / column block in one 16-bit
// load; fp8_gemm_mx reads them from global memory per 32-block inside the MMA loop, B's once per n tile though all of a
// warp's 32 columns share one), one barrier per K step (load the next step after the barrier that ends the previous
// step's reads), a warp tile of 64 x 8 NT (WMW x WNW warps).
template <bool B16, int WMW, int WNW, int NT>
__global__ __launch_bounds__(32 * WMW * WNW) void fp8_gemm_mx2(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ as_,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, void* __restrict__ yv, int M, int N, int K) {
  constexpr int BM = 64 * WMW, BN = 8 * NT * WNW, NTH = 32 * WMW * WNW, NB = NT / 4;   // NB: 32-col scale blocks per warp
  __shared__ __align__(16) uint8_t sa[2][BM * GP], sb[2][BN * GP];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int wm = warp / WNW, wn = warp % WNW;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int nk = K / GK, kb = K / 32;
  auto load = [&](int stage, int k0) {
    for (int i = tid; i < BM * GK / 16; i += NTH) {
      const int r = i >> 2, c = (i & 3) * 16, gm = m0 + r;
      cp16(&sa[stage][r * GP + c], aq + (long long)(gm < M ? gm : 0) * K + k0 + c, gm < M);
    }
    for (int i = tid; i < BN * GK / 16; i += NTH) {
      const int r = i >> 2, c = (i & 3) * 16, gn = n0 + r;
      cp16(&sb[stage][r * GP + c], w + (long long)(gn < N ? gn : 0) * K + k0 + c, gn < N);
    }
    asm volatile("cp.async.commit_group;\n");
  };
  float acc[4][NT][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  const uint16_t* pa[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) pa[mt] = reinterpret_cast<const uint16_t*>(as_ + (long long)min(m0 + wm * 64 + mt * 16 + g + 8 * (lane & 1), M - 1) * kb);
  const uint16_t* pb[NB];
#pragma unroll
  for (int j = 0; j < NB; ++j) pb[j] = reinterpret_cast<const uint16_t*>(ws + (long long)(min(n0 + wn * 8 * NT + j * 32 + g, N - 1) >> 5) * kb);
  uint16_t ca[4], cb[NB];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) ca[mt] = __ldg(pa[mt]);
#pragma unroll
  for (int j = 0; j < NB; ++j) cb[j] = __ldg(pb[j]);
  load(0, 0);
  for (int ks = 0; ks < nk; ++ks) {
    const int st = ks & 1;
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    if (ks + 1 < nk) load(st ^ 1, (ks + 1) * GK);
    uint16_t na[4], nb[NB];
    const int kn = ks + 1 < nk ? ks + 1 : ks;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) na[mt] = __ldg(pa[mt] + kn);
#pragma unroll
    for (int j = 0; j < NB; ++j) nb[j] = __ldg(pb[j] + kn);
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      uint32_t a[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const uint8_t* p0 = &sa[st][(wm * 64 + mt * 16 + g) * GP + b * 32 + 8 * q];
        const uint2 r0 = *reinterpret_cast<const uint2*>(p0), r1 = *reinterpret_cast<const uint2*>(p0 + 8 * GP);
        a_split(r0, a[mt][0], a[mt][2]);
        a_split(r1, a[mt][1], a[mt][3]);
      }
      uint32_t sfa[4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) sfa[mt] = (ca[mt] >> (8 * b)) & 0xFF;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const uint2 rb = *reinterpret_cast<const uint2*>(&sb[st][(wn * 8 * NT + nt * 8 + g) * GP + b * 32 + 8 * q]);
        uint32_t b0, b1;
        a_split(rb, b0, b1);
        const uint32_t sfb = (cb[nt / 4] >> (8 * b)) & 0xFF;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) mma_e4(acc[mt][nt], a[mt][0], a[mt][1], a[mt][2], a[mt][3], b0, b1, sfa[mt], sfb);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) ca[mt] = na[mt];
#pragma unroll
    for (int j = 0; j < NB; ++j) cb[j] = nb[j];
  }
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m0 + wm * 64 + mt * 16 + hh * 8 + g;
      if (r >= M) continue;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int c = n0 + wn * 8 * NT + nt * 8 + 2 * q;
        if (c < N) {
          if (B16) *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(yv) + (long long)r * N + c) = __floats2bfloat162_rn(acc[mt][nt][hh * 2], acc[mt][nt][hh * 2 + 1]);
          else *reinterpret_cast<float2*>(reinterpret_cast<float*>(yv) + (long long)r * N + c) = make_float2(acc[mt][nt][hh * 2], acc[mt][nt][hh * 2 + 1]);
        }
      }
    }
}
// fp8_gemm_mx2 (the same MMA sequence per output: bitwise equal) with the operand rows PB bytes apart in shared memory
// and NS stages (dynamic shared memory). fp8_gemm_mx2's 80-byte rows (GP): the 64-bit fragment loads of a half warp
// (rows g = 0..3, 8 bytes at 8 q) fall on 4-byte banks 20 g + 2 q + {0, 1}, so rows 0 and 3 share banks 0-3 (two-way
// conflicts); 96-byte rows (banks 24 g + ..) are conflict-free.
// PB == 0: 64-byte rows (no padding) with the 16-byte chunks of rows with bit 1 set swapped in pairs (chunk ^ 2): a half
// warp's rows g = 0..3 then cover banks 0-7, 16-23, 8-15, 24-31 (conflict-free), a third less shared memory than 96.
template <bool B16, int WMW, int WNW, int NT, int NS, int PB>
__global__ __launch_bounds__(32 * WMW * WNW) void fp8_gemm_mx3(const uint8_t* __restrict__ aq, const uint8_t* __restrict__ as_,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, void* __restrict__ yv, int M, int N, int K) {
  constexpr int BM = 64 * WMW, BN = 8 * NT * WNW, NTH = 32 * WMW * WNW, NB = NT / 4, RB = PB ? PB : GK;
  // byte offset of (row r, byte c of the K step) in a stage
  auto at = [](int r, int c) { return PB ? r * PB + c : r * GK + ((((c >> 4) ^ (((r >> 1) & 1) << 1))) << 4) + (c & 15); };
  extern __shared__ __align__(16) uint8_t g3_smem[];
  uint8_t* const sa0 = g3_smem;                     // [NS][BM * RB]
  uint8_t* const sb0 = g3_smem + NS * BM * RB;      // [NS][BN * RB]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int wm = warp / WNW, wn = warp % WNW;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int nk = K / GK, kb = K / 32;
  auto load = [&](int stage, int k0) {
    uint8_t* sa = sa0 + stage * BM * RB;
    uint8_t* sb = sb0 + stage * BN * RB;
    for (int i = tid; i < BM * GK / 16; i += NTH) {
      const int r = i >> 2, c = (i & 3) * 16, gm = m0 + r;
      cp16(&sa[at(r, c)], aq + (long long)(gm < M ? gm : 0) * K + k0 + c, gm < M);
    }
    for (int i = tid; i < BN * GK / 16; i += NTH) {
      const int r = i >> 2, c = (i & 3) * 16, gn = n0 + r;
      cp16(&sb[at(r, c)], w + (long long)(gn < N ? gn : 0) * K + k0 + c, gn < N);
    }
  };
  float acc[4][NT][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  const uint16_t* pa[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) pa[mt] = reinterpret_cast<const uint16_t*>(as_ + (long long)min(m0 + wm * 64 + mt * 16 + g + 8 * (lane & 1), M - 1) * kb);
  const uint16_t* pb[NB];
#pragma unroll
  for (int j = 0; j < NB; ++j) pb[j] = reinterpret_cast<const uint16_t*>(ws + (long long)(min(n0 + wn * 8 * NT + j * 32 + g, N - 1) >> 5) * kb);
  uint16_t ca[4], cb[NB];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) ca[mt] = __ldg(pa[mt]);
#pragma unroll
  for (int j = 0; j < NB; ++j) cb[j] = __ldg(pb[j]);
#pragma unroll
  for (int s = 0; s < NS - 1; ++s) { if (s < nk) load(s, s * GK); asm volatile("cp.async.commit_group;\n"); }
  for (int ks = 0; ks < nk; ++ks) {
    const int st = ks % NS;
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 2) : "memory");
    __syncthreads();
    if (ks + NS - 1 < nk) load((ks + NS - 1) % NS, (ks + NS - 1) * GK);
    asm volatile("cp.async.commit_group;\n");
    const uint8_t* sa = sa0 + st * BM * RB;
    const uint8_t* sb = sb0 + st * BN * RB;
    uint16_t na[4], nb[NB];
    const int kn = ks + 1 < nk ? ks + 1 : ks;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) na[mt] = __ldg(pa[mt] + kn);
#pragma unroll
    for (int j = 0; j < NB; ++j) nb[j] = __ldg(pb[j] + kn);
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      uint32_t a[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const int ra = wm * 64 + mt * 16 + g;   // (ra + 8: the same swizzle)
        const uint2 r0 = *reinterpret_cast<const uint2*>(&sa[at(ra, b * 32 + 8 * q)]), r1 = *reinterpret_cast<const uint2*>(&sa[at(ra + 8, b * 32 + 8 * q)]);
        a_split(r0, a[mt][0], a[mt][2]);
        a_split(r1, a[mt][1], a[mt][3]);
      }
      uint32_t sfa[4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) sfa[mt] = (ca[mt] >> (8 * b)) & 0xFF;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const uint2 rb = *reinterpret_cast<const uint2*>(&sb[at(wn * 8 * NT + nt * 8 + g, b * 32 + 8 * q)]);
        uint32_t b0, b1;
        a_split(rb, b0, b1);
        const uint32_t sfb = (cb[nt / 4] >> (8 * b)) & 0xFF;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) mma_e4(acc[mt][nt], a[mt][0], a[mt][1], a[mt][2], a[mt][3], b0, b1, sfa[mt], sfb);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) ca[mt] = na[mt];
#pragma unroll
    for (int j = 0; j < NB; ++j) cb[j] = nb[j];
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m0 + wm * 64 + mt * 16 + hh * 8 + g;
      if (r >= M) continue;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int c = n0 + wn * 8 * NT + nt * 8 + 2 * q;
        if (c < N) {
          if (B16) *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(yv) + (long long)r * N + c) = __floats2bfloat162_rn(acc[mt][nt][hh * 2], acc[mt][nt][hh * 2 + 1]);
          else *reinterpret_cast<float2*>(reinterpret_cast<float*>(yv) + (long long)r * N + c) = make_float2(acc[mt][nt][hh * 2], acc[mt][nt][hh * 2 + 1]);
        }
      }
    }
}
template <int WMW, int WNW, int NT, int NS, int PB>
int launch_g3(const void* aq, const void* as_, const void* w, const void* ws, void* y, int M, int N, int K, int bf16out, cudaStream_t st) {
  constexpr int BM = 64 * WMW, BN = 8 * NT * WNW, SM = NS * (BM + BN) * (PB ? PB : GK);
  static const bool attr = cudaFuncSetAttribute(fp8_gemm_mx3<true, WMW, WNW, NT, NS, PB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) == cudaSuccess &&
                           cudaFuncSetAttribute(fp8_gemm_mx3<false, WMW, WNW, NT, NS, PB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) == cudaSuccess;
  if (!attr) return 3;
  const dim3 gr((M + BM - 1) / BM, (N + BN - 1) / BN);
  if (bf16out) fp8_gemm_mx3<true, WMW, WNW, NT, NS, PB><<<gr, 32 * WMW * WNW, SM, st>>>((const uint8_t*)aq, (const uint8_t*)as_, (const uint8_t*)w, (const uint8_t*)ws, y, M, N, K);
  else fp8_gemm_mx3<false, WMW, WNW, NT, NS, PB><<<gr, 32 * WMW * WNW, SM, st>>>((const uint8_t*)aq, (const uint8_t*)as_, (const uint8_t*)w, (const uint8_t*)ws, y, M, N, K);
  return (int)cudaGetLastError();
}

int g_gemm_version = 0;   // 0: default; ds41_gemm_version (benches) forces one

// ---- Prefill gate/up as a grouped GEMM: items of up to 64 rows (one expert), block = 64 rows x 128 intermediate
// columns (16 w1 + 16 w3 tiles of 8), 8 warps (2 row halves x 4 groups of 32 columns: the up_mx<2, 1> warp tile). K in
// steps of 128 through shared memory, UG_NS stages in flight (cp.async): every weight byte is read once per block for
// all its rows (up_mx: once per 32 rows, into registers, one step ahead). Packed weight blocks keep the lane order of
// the global layout (lane l's 16 bytes of a tile/step block), A rows are gathered by prow. Same MMA sequence per
// output as up_mx<2, 1> (bitwise equal), same epilogue.
constexpr int UG_NS = 3, UG_AP = 144;
// RT rows per block (64, or 128: an expert's rows up to 128 share one read of its weights), RT / 32 row groups x 8 / (RT
// / 32) column groups of warps (the 64-row block: 2 x 4 warps of 32 rows x 32 columns, up_mx<2, 1>'s tile); NS stages.
template <int RT> struct UgStageT { uint8_t a[RT][UG_AP]; uint32_t as[RT]; uint4 b[32][32]; uint32_t bs[32][8]; uint8_t bc[32][8]; };   // bc: 2-bit codes (C2)
using UgStage = UgStageT<64>;
__device__ __forceinline__ void cpa16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cpa4(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void cpa8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

template <int RT, int NS, bool C2>
__global__ __launch_bounds__(256, 1) void up_gemm(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs, const uint8_t* __restrict__ w1,
    const ScaleV s1, const uint8_t* __restrict__ w3, const ScaleV s3, int I,
    const float* __restrict__ rw, const int* __restrict__ G, const Item* __restrict__ items, const int* __restrict__ prow,
    uint8_t* __restrict__ hq, uint8_t* __restrict__ hs, int topk) {
  constexpr int WMR = RT / 32, WNC = 8 / WMR, TPW = 16 / WNC, AJ = RT * 8 / 256;   // TPW: n8 tiles per warp and matrix
  pdl_entry();
  if ((int)blockIdx.y >= *G) return;
  extern __shared__ __align__(16) unsigned char ug_smem[];
  UgStageT<RT>* st = reinterpret_cast<UgStageT<RT>*>(ug_smem);
  const Item it = items[blockIdx.y];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int wm = warp / WNC, wn = warp % WNC;
  const int jb = blockIdx.x * 128;
  constexpr int ALL = H / 128;
  // this thread's load jobs: A rows (AJ x 16 B), A scales (threads < RT), B blocks (4 x 16 B), B scales (threads < 64)
  int a_row[AJ]; const uint8_t* a_src[AJ]; bool a_ok[AJ];
  for (int j = 0; j < AJ; ++j) {
    const int i = tid + 256 * j, r = i >> 3;
    a_row[j] = r; a_ok[j] = r < it.n;
    a_src[j] = xq + (long long)(a_ok[j] ? prow[it.off + r] / topk : 0) * H + (i & 7) * 16;
  }
  const uint8_t* as_src = xs + (long long)(tid < RT && tid < it.n ? prow[it.off + tid] / topk : 0) * (H / 32);
  const uint8_t* b_src[4];
  for (int j = 0; j < 4; ++j) {
    const int i = tid + 256 * j, t = i >> 5, l = i & 31;
    const long long tile = ((long long)it.e * I + jb + 8 * (t & 15)) / 8;
    b_src[j] = (t < 16 ? w1 : w3) + tile * ALL * 512 + l * 16;
  }
  // B scales: raw words (threads < 64: half a tile's 32 bytes each) or a coded tile's 8 code bytes (even threads)
  const uint8_t* bs_src = nullptr; bool bs_raw = true;
  if (tid < 64) {
    const int t = tid >> 1;
    const long long tile = ((long long)it.e * I + jb + 8 * (t & 15)) / 8;
    const ScaleV& v = t < 16 ? s1 : s3;
    if (!C2) bs_src = v.raw + tile * ALL * 32 + (tid & 1) * 16;
    else {
      const int ri = (int)__ldg(v.desc + tile).z;
      bs_raw = ri >= 0;
      bs_src = bs_raw ? v.raw + (long long)ri * ALL * 32 + (tid & 1) * 16 : v.code + tile * ALL * 8;
    }
  }
  auto load = [&](int s, UgStageT<RT>& b) {
    for (int j = 0; j < AJ; ++j) cpa16(&b.a[a_row[j]][((tid + 256 * j) & 7) * 16], a_src[j] + s * 128, a_ok[j]);
    if (tid < RT) cpa4(&b.as[tid], as_src + s * 4, tid < it.n);
    for (int j = 0; j < 4; ++j) { const int i = tid + 256 * j; cpa16(&b.b[i >> 5][i & 31], b_src[j] + s * 512, true); }
    if (tid < 64) {
      if (bs_raw) cpa16(reinterpret_cast<uint8_t*>(&b.bs[tid >> 1][0]) + (tid & 1) * 16, bs_src + s * 32, true);
      else if (!(tid & 1)) cpa8(&b.bc[tid >> 1][0], bs_src + s * 8);
    }
  };
  // this lane's row base (x 0x01010101) of each of its tiles, and which of them are raw
  uint32_t bmul[2 * TPW]; uint32_t rawm = 0;
  if (C2)
    for (int t = 0; t < 2 * TPW; ++t) {
      const int tb = (t < TPW ? 0 : 16) + wn * TPW + (t % TPW);
      const long long tile = ((long long)it.e * I + jb + 8 * (tb & 15)) / 8;
      const uint4 d = __ldg((tb < 16 ? s1 : s3).desc + tile);
      if ((int)d.z >= 0) rawm |= 1u << t;
      bmul[t] = (((g < 4 ? d.x : d.y) >> (8 * (g & 3))) & 0xFFu) * 0x01010101u;
    }
  float acc[2][2 * TPW][4];
  for (int m = 0; m < 2; ++m) for (int t = 0; t < 2 * TPW; ++t) for (int c = 0; c < 4; ++c) acc[m][t][c] = 0.f;
#pragma unroll
  for (int s = 0; s < NS - 1; ++s) { load(s, st[s]); cpa_commit(); }
  for (int s = 0; s < ALL; ++s) {
    cpa_wait<NS - 2>();
    __syncthreads();
    if (s + NS - 1 < ALL) load(s + NS - 1, st[(s + NS - 1) % NS]);
    cpa_commit();
    const UgStageT<RT>& b = st[s % NS];
    uint4 bw[2 * TPW]; uint32_t bsc[2 * TPW];
#pragma unroll
    for (int t = 0; t < 2 * TPW; ++t) {
      const int tb = (t < TPW ? 0 : 16) + wn * TPW + (t % TPW);
      bw[t] = b.b[tb][lane]; bsc[t] = (!C2 || ((rawm >> t) & 1)) ? b.bs[tb][g] : bmul[t] + s2_spread(b.bc[tb][g]);
    }
    uint32_t sfa4[2];
    for (int m = 0; m < 2; ++m) sfa4[m] = b.as[wm * 32 + m * 16 + (lane >> 2) + 8 * (lane & 1)];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      uint32_t a[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int r0 = wm * 32 + m * 16 + g;
        a_split(*reinterpret_cast<const uint2*>(&b.a[r0][kb * 32 + 8 * q]), a[m][0], a[m][2]);
        a_split(*reinterpret_cast<const uint2*>(&b.a[r0 + 8][kb * 32 + 8 * q]), a[m][1], a[m][3]);
      }
#pragma unroll
      for (int t = 0; t < 2 * TPW; ++t) {
        const uint32_t w = kb == 0 ? bw[t].x : kb == 1 ? bw[t].y : kb == 2 ? bw[t].z : bw[t].w;
        uint32_t b0, b1;
        b_split(w, b0, b1);
        const uint32_t sfb = (bsc[t] >> (8 * kb)) & 0xFF;
        for (int m = 0; m < 2; ++m) mma_mx(acc[m][t], a[m][0], a[m][1], a[m][2], a[m][3], b0, b1, (sfa4[m] >> (8 * kb)) & 0xFF, sfb);
      }
    }
  }
  cpa_wait<0>();
  // epilogue (up_mx): SwiGLU of gate / up rounded to BF16, routing weight, act_quant per 32 columns (TPW / 4 groups)
#pragma unroll
  for (int gq = 0; gq < TPW / 4; ++gq) {
    const int j0 = jb + wn * 8 * TPW + gq * 32;
    for (int m = 0; m < 2; ++m)
      for (int hh = 0; hh < 2; ++hh) {
        const int r = wm * 32 + m * 16 + hh * 8 + g;
        const float wgt = r < it.n ? rw[prow[it.off + r]] : 0.f;
        float v[8], amax = 0.f;
        for (int t = 0; t < 4; ++t)
          for (int c = 0; c < 2; ++c) {
            const float gt = fminf(bf16r(acc[m][gq * 4 + t][hh * 2 + c]), 10.f);
            const float up = fminf(fmaxf(bf16r(acc[m][TPW + gq * 4 + t][hh * 2 + c]), -10.f), 10.f);
            const float y = bf16r(gt / (1.f + expf(-gt)) * up * wgt);
            v[t * 2 + c] = y;
            amax = fmaxf(amax, fabsf(y));
          }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        const uint32_t e = aq_scale(amax);
        if (r < it.n) {
          uint8_t* hr = hq + (long long)(it.off + r) * I + j0 + 2 * q;
          for (int t = 0; t < 4; ++t)
            *reinterpret_cast<uint16_t*>(hr + 8 * t) = (uint16_t)aq_byte(v[2 * t], e) | ((uint16_t)aq_byte(v[2 * t + 1], e) << 8);
          if (q == 0) hs[(long long)(it.off + r) * (I / 32) + j0 / 32] = (uint8_t)e;
        }
      }
  }
}

// ---- Prefill down as a grouped GEMM (the up_gemm scheme): block = one item's 64 rows x 128 output columns (16 w2 tiles
// of 8), 8 warps (2 row halves x 4 groups of 32 columns), I in steps of 128 through shared memory, UG_NS stages in flight.
// down_mx<4> reads the item's A rows from global memory in every warp and the weights into registers without prefetch;
// here each block loads both once per step. Same MMA sequence per output as down_mx (bitwise equal).
template <int RT> struct DgStageT { uint8_t a[RT][UG_AP]; uint32_t as[RT]; uint4 b[16][32]; uint32_t bs[16][8]; uint8_t bc[16][8]; };
template <int RT, int NS, bool C2>
__global__ __launch_bounds__(256, 1) void down_gemm(const uint8_t* __restrict__ hq, const uint8_t* __restrict__ hs, const uint8_t* __restrict__ w2,
    const ScaleV s2, int I, const int* __restrict__ G, const Item* __restrict__ items, float* __restrict__ out, int c0, int ldo) {
  constexpr int WMR = RT / 32, WNC = 8 / WMR, TPW = 16 / WNC, AJ = RT * 8 / 256;
  pdl_entry();
  if ((int)blockIdx.y >= *G) return;
  extern __shared__ __align__(16) unsigned char dg_smem[];
  DgStageT<RT>* st = reinterpret_cast<DgStageT<RT>*>(dg_smem);
  const Item it = items[blockIdx.y];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int wm = warp / WNC, wn = warp % WNC;
  const int nb = c0 + blockIdx.x * 128;   // output columns [c0, ..) into out rows of ldo (a column pass, see moe_mx)
  const int steps = I / 128;
  // load jobs: A rows (AJ x 16 B), A scales (threads < RT), B blocks (2 x 16 B), B scales (threads < 32)
  int a_row[AJ]; const uint8_t* a_src[AJ]; bool a_ok[AJ];
  for (int j = 0; j < AJ; ++j) {
    const int i = tid + 256 * j, r = i >> 3;
    a_row[j] = r; a_ok[j] = r < it.n;
    a_src[j] = hq + (long long)(it.off + (a_ok[j] ? r : 0)) * I + (i & 7) * 16;
  }
  const uint8_t* as_src = hs + (long long)(it.off + (tid < RT && tid < it.n ? tid : 0)) * (I / 32);
  const uint8_t* b_src[2];
  for (int j = 0; j < 2; ++j) {
    const int i = tid + 256 * j, t = i >> 5, l = i & 31;
    const long long tile = ((long long)it.e * H + nb + 8 * t) / 8;
    b_src[j] = w2 + tile * steps * 512 + l * 16;
  }
  const uint8_t* bs_src = nullptr; bool bs_raw = true;   // (as up_gemm)
  if (tid < 32) {
    const int t = tid >> 1;
    const long long tile = ((long long)it.e * H + nb + 8 * t) / 8;
    if (!C2) bs_src = s2.raw + tile * steps * 32 + (tid & 1) * 16;
    else {
      const int ri = (int)__ldg(s2.desc + tile).z;
      bs_raw = ri >= 0;
      bs_src = bs_raw ? s2.raw + (long long)ri * steps * 32 + (tid & 1) * 16 : s2.code + tile * steps * 8;
    }
  }
  auto load = [&](int s, DgStageT<RT>& b) {
    for (int j = 0; j < AJ; ++j) cpa16(&b.a[a_row[j]][((tid + 256 * j) & 7) * 16], a_src[j] + s * 128, a_ok[j]);
    if (tid < RT) cpa4(&b.as[tid], as_src + s * 4, tid < it.n);
    for (int j = 0; j < 2; ++j) { const int i = tid + 256 * j; cpa16(&b.b[i >> 5][i & 31], b_src[j] + s * 512, true); }
    if (tid < 32) {
      if (bs_raw) cpa16(reinterpret_cast<uint8_t*>(&b.bs[tid >> 1][0]) + (tid & 1) * 16, bs_src + s * 32, true);
      else if (!(tid & 1)) cpa8(&b.bc[tid >> 1][0], bs_src + s * 8);
    }
  };
  uint32_t bmul[TPW]; uint32_t rawm = 0;
  if (C2)
    for (int t = 0; t < TPW; ++t) {
      const long long tile = ((long long)it.e * H + nb + 8 * (wn * TPW + t)) / 8;
      const uint4 d = __ldg(s2.desc + tile);
      if ((int)d.z >= 0) rawm |= 1u << t;
      bmul[t] = (((g < 4 ? d.x : d.y) >> (8 * (g & 3))) & 0xFFu) * 0x01010101u;
    }
  float acc[2][TPW][4];
  for (int m = 0; m < 2; ++m) for (int t = 0; t < TPW; ++t) for (int c = 0; c < 4; ++c) acc[m][t][c] = 0.f;
#pragma unroll
  for (int s = 0; s < NS - 1; ++s) { if (s < steps) load(s, st[s]); cpa_commit(); }
  for (int s = 0; s < steps; ++s) {
    cpa_wait<NS - 2>();
    __syncthreads();
    if (s + NS - 1 < steps) load(s + NS - 1, st[(s + NS - 1) % NS]);
    cpa_commit();
    const DgStageT<RT>& b = st[s % NS];
    uint4 bw[TPW]; uint32_t bsc[TPW];
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      bw[t] = b.b[wn * TPW + t][lane];
      bsc[t] = (!C2 || ((rawm >> t) & 1)) ? b.bs[wn * TPW + t][g] : bmul[t] + s2_spread(b.bc[wn * TPW + t][g]);
    }
    uint32_t sfa4[2];
    for (int m = 0; m < 2; ++m) sfa4[m] = b.as[wm * 32 + m * 16 + (lane >> 2) + 8 * (lane & 1)];
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      uint32_t a[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int r0 = wm * 32 + m * 16 + g;
        a_split(*reinterpret_cast<const uint2*>(&b.a[r0][kb * 32 + 8 * q]), a[m][0], a[m][2]);
        a_split(*reinterpret_cast<const uint2*>(&b.a[r0 + 8][kb * 32 + 8 * q]), a[m][1], a[m][3]);
      }
#pragma unroll
      for (int t = 0; t < TPW; ++t) {
        const uint32_t w = kb == 0 ? bw[t].x : kb == 1 ? bw[t].y : kb == 2 ? bw[t].z : bw[t].w;
        uint32_t b0, b1;
        b_split(w, b0, b1);
        const uint32_t sfb = (bsc[t] >> (8 * kb)) & 0xFF;
        for (int m = 0; m < 2; ++m) mma_mx(acc[m][t], a[m][0], a[m][1], a[m][2], a[m][3], b0, b1, (sfa4[m] >> (8 * kb)) & 0xFF, sfb);
      }
    }
  }
  cpa_wait<0>();
  const int n0 = nb + wn * 8 * TPW;
  for (int m = 0; m < 2; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      const int r = wm * 32 + m * 16 + hh * 8 + g;
      if (r >= it.n) continue;
      float* o = out + (long long)(it.off + r) * ldo + (n0 - c0) + 2 * q;
      for (int t = 0; t < TPW; ++t) *reinterpret_cast<float2*>(o + 8 * t) = make_float2(acc[m][t][hh * 2], acc[m][t][hh * 2 + 1]);
    }
}

// (Not GLM's fp8.cu / fp8_skinny.cuh: those take FP32 / Half activations and one scale per weight row; DeepSeek's FP8
// linears are act_quant E4M3 activations and 32x32 E8M0 weight blocks, the block-scaled MMA's own operands.)
// ---- FP8 GEMV for a few rows (decode batches, speculative verify) on the block-scaled MMA: y [m, n] FP32 = A . W^T,
// m <= 16 (one m16 tile, rows >= m are zeros), A E4M3 + UE8M0 per 32 (act_quant), W E4M3 [n, k] + E8M0 per 32x32.
// Warp = one n8 tile (8 weight rows). Weights stream like the CUDA-core GEMV: every lane reads 16 contiguous bytes of
// its row per 64-byte span (two 32-blocks; lanes q = 0, 1 hold block 0, q = 2, 3 block 1), and one shuffle with lane
// q ^ 2 gives every lane the 8 bytes of each block the MMA wants from it (block 0: q -> bytes [8 q, 8 q + 8) of the
// block, from lane q >> 1's chunk half q & 1; block 1 likewise from lanes 2, 3). The cost hardly depends on m (the
// CUDA-core GEMV's grows linearly: 1.85x at m = 2, 3.6x at m = 4).
// S > 1 (few weight rows: too few warps to stream at full bandwidth): the block's 8 warps take 8 / S row tiles, the S
// warps of a tile each a contiguous 1 / S of the K spans, summed in shared memory in slice order (deterministic; S
// depends on the shape only, so a row's result does not depend on m).
template <int U, int S>
// aq null: x BF16 [m, k] is act-quantized by the CTA into shared memory first (one warp per 32-block, the same scale
// and bytes as act_quant_fp8_kernel; dynamic shared memory m k + m k / 32 bytes), so no separate quantization kernel;
// else aq / as_ are the quantized input in global memory. ob16: y is BF16 (rounded sums).
// job: an mHC mix job run by the grid's first job.ctas CTAs (ds41_hc_job.cuh; kind 0: none)
__global__ __launch_bounds__(256) void fp8_gemv_mx(const __nv_bfloat16* __restrict__ x, const uint8_t* aq, const uint8_t* as_,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, void* __restrict__ yv, int m, int n, int k, int ob16, HcJob job) {
  float* y = reinterpret_cast<float*>(yv);
  extern __shared__ __align__(16) uint8_t gq_smem[];
  pdl_trigger();
  if (hcj::run(job)) return;
  const int bx = blockIdx.x - (job.kind ? job.ctas : 0);
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int wid = threadIdx.x >> 5, slice = wid % S;
  const int n0 = (bx * (8 / S) + wid / S) * 8;
  const int kb = k / 32;
  const int ns = kb / 2, s_lo = slice * ns / S, s_hi = (slice + 1) * ns / S;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int row = min(n0 + g, n - 1);
  const uint8_t* wr = w + (long long)row * k + 16 * q;
  const uint8_t* wsr = ws + (long long)(min(n0, n - 8) >> 5) * kb;
  // weights of U spans (and their scales) into registers
  auto ldw = [&](uint4* wv, uint32_t (*sb)[2], int sp0) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int sp = sp0 + u;
      if (sp < s_hi) {
        wv[u] = __ldg(reinterpret_cast<const uint4*>(wr + sp * 64));
        sb[u][0] = __ldg(wsr + 2 * sp); sb[u][1] = __ldg(wsr + 2 * sp + 1);
      }
    }
  };
  // the first weights load while the previous kernel still runs (PDL); activations only after the wait
  uint4 wv[U]; uint32_t sb[U][2];
  if (n0 < n) ldw(wv, sb, s_lo);
  pdl_wait();
  if (!aq) {
    uint8_t* qs = gq_smem;
    uint8_t* ss = gq_smem + (long long)m * k;
    // 4 lanes per 32-block, 8 values each: 64 blocks per pass of the CTA
    const int nb = m * kb;
    for (int b0 = (threadIdx.x >> 2); b0 < ((nb + 63) / 64) * 64; b0 += 64) {
      const bool ok = b0 < nb;
      const uint4 raw = ok ? *reinterpret_cast<const uint4*>(x + (long long)b0 * 32 + 8 * (threadIdx.x & 3)) : make_uint4(0, 0, 0, 0);
      const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&raw);
      float f[8], a = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) { f[j] = __bfloat162float(hv[j]); a = fmaxf(a, fabsf(f[j])); }
      a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
      a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
      const uint32_t e = aq_scale(a);
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) { lo |= (uint32_t)aq_byte(f[j], e) << (8 * j); hi |= (uint32_t)aq_byte(f[j + 4], e) << (8 * j); }
      if (ok) {
        *reinterpret_cast<uint2*>(qs + (long long)b0 * 32 + 8 * (threadIdx.x & 3)) = make_uint2(lo, hi);
        if ((threadIdx.x & 3) == 0) ss[b0] = (uint8_t)e;
      }
    }
    __syncthreads();
    aq = qs; as_ = ss;
  }
  if (n0 < n) {
    const int r0 = g, r1 = g + 8, rs = (lane >> 2) + 8 * (lane & 1);
    const uint8_t* a0p = aq + (long long)(r0 < m ? r0 : 0) * k + 8 * q;
    const uint8_t* a1p = aq + (long long)(r1 < m ? r1 : 0) * k + 8 * q;
    const bool z0 = r0 >= m, z1 = r1 >= m;
    const uint8_t* asp = as_ + (long long)(rs < m ? rs : 0) * kb;
    // per 64-byte span: lane holds bytes [16 q, 16 q + 16); block c (0, 1) of the span needs, at lane q, bytes
    // [8 q, 8 q + 8) of the block = span bytes [32 c + 8 q, +8): held by lane 2 c + (q >> 1), half q & 1
    const int src_hi = q >> 1;                                  // which lane of the block's pair holds my bytes
    const int half = q & 1;
    for (int sp0 = s_lo; sp0 < s_hi; sp0 += U) {
      uint2 x0[U][2], x1[U][2]; uint32_t sa[U][2];
  #pragma unroll
      for (int u = 0; u < U; ++u) {
        const int sp = sp0 + u;
        if (sp < s_hi) {
          for (int c = 0; c < 2; ++c) {
            const int b = 2 * sp + c;
            x0[u][c] = z0 ? make_uint2(0, 0) : *reinterpret_cast<const uint2*>(a0p + b * 32);
            x1[u][c] = z1 ? make_uint2(0, 0) : *reinterpret_cast<const uint2*>(a1p + b * 32);
            sa[u][c] = asp[b];
          }
        }
      }
      // the next U spans' weights load while these compute
      uint4 nw[U]; uint32_t nsb[U][2];
      if (sp0 + U < s_hi) ldw(nw, nsb, sp0 + U);
  #pragma unroll
      for (int u = 0; u < U; ++u) {
        if (sp0 + u >= s_hi) break;
  #pragma unroll
        for (int c = 0; c < 2; ++c) {
          // my 8 bytes of block c sit in quad lane 2 c + (q >> 1), half q & 1 (every lane shuffles: uniform)
          const int src = (lane & ~3) + 2 * c + src_hi;
          const uint32_t vx = __shfl_sync(0xffffffffu, wv[u].x, src), vy = __shfl_sync(0xffffffffu, wv[u].y, src);
          const uint32_t vz = __shfl_sync(0xffffffffu, wv[u].z, src), vw = __shfl_sync(0xffffffffu, wv[u].w, src);
          const uint32_t lo = half ? vz : vx, hi = half ? vw : vy;
          uint32_t a0, a1, a2, a3, bb0, bb1;
          a_split(x0[u][c], a0, a2);
          a_split(x1[u][c], a1, a3);
          a_split(make_uint2(lo, hi), bb0, bb1);
          mma_e4(acc, a0, a1, a2, a3, bb0, bb1, sa[u][c], sb[u][c]);
        }
      }
#pragma unroll
      for (int u = 0; u < U; ++u) { wv[u] = nw[u]; sb[u][0] = nsb[u][0]; sb[u][1] = nsb[u][1]; }
    }
  }
  if (S > 1) {
    __shared__ float4 red[8][32];
    red[wid][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    if (slice) return;
    for (int j = 1; j < S; ++j) {
      const float4 o = red[wid + j][lane];
      acc[0] += o.x; acc[1] += o.y; acc[2] += o.z; acc[3] += o.w;
    }
  }
  if (n0 >= n) return;
  const int r0 = g, r1 = g + 8;
  const int c = n0 + 2 * q;
  if (c < n) {
    if (ob16) {
      __nv_bfloat16* yb = reinterpret_cast<__nv_bfloat16*>(yv);
      if (r0 < m) *reinterpret_cast<__nv_bfloat162*>(yb + (long long)r0 * n + c) = __floats2bfloat162_rn(acc[0], acc[1]);
      if (r1 < m) *reinterpret_cast<__nv_bfloat162*>(yb + (long long)r1 * n + c) = __floats2bfloat162_rn(acc[2], acc[3]);
    } else {
      if (r0 < m) *reinterpret_cast<float2*>(y + (long long)r0 * n + c) = make_float2(acc[0], acc[1]);
      if (r1 < m) *reinterpret_cast<float2*>(y + (long long)r1 * n + c) = make_float2(acc[2], acc[3]);
    }
  }
}


// ---- BF16 GEMV for a few rows (vocabulary head of verify / draft rows): y [m, n] FP32 = x . W^T, x BF16 [m, k] (m <=
// 16), W BF16 [n, k]: BF16 MMA m16n8k16 (exact products, FP32 sums). Warp = 8 weight rows; per 32-wide K span every
// lane reads 16 bytes (K 8q .. 8q+7) of its row and of the x rows and feeds two MMAs (K 8q .. 8q+3, then 8q+4 .. 8q+7
// as the logical (2q, 2q+1, 2q+8, 2q+9)): A and B use the same mapping.
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int U>
__global__ __launch_bounds__(256) void bf16_gemv_mx(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                   float* __restrict__ y, int m, int n, int k) {
  pdl_entry();
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 8;
  if (n0 >= n) return;
  const int r0 = g, r1 = g + 8;
  const bool z0 = r0 >= m, z1 = r1 >= m;
  const __nv_bfloat16* x0p = x + (long long)(z0 ? 0 : r0) * k + 8 * q;
  const __nv_bfloat16* x1p = x + (long long)(z1 ? 0 : r1) * k + 8 * q;
  const __nv_bfloat16* wr = w + (long long)min(n0 + g, n - 1) * k + 8 * q;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int spans = k / 32;
  for (int s0 = 0; s0 < spans; s0 += U) {
    uint4 wv[U], a[U], b[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int sp = s0 + u;
      if (sp < spans) {
        wv[u] = __ldg(reinterpret_cast<const uint4*>(wr + sp * 32));
        a[u] = z0 ? make_uint4(0, 0, 0, 0) : __ldg(reinterpret_cast<const uint4*>(x0p + sp * 32));
        b[u] = z1 ? make_uint4(0, 0, 0, 0) : __ldg(reinterpret_cast<const uint4*>(x1p + sp * 32));
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (s0 + u >= spans) break;
      mma_bf16(acc, a[u].x, b[u].x, a[u].y, b[u].y, wv[u].x, wv[u].y);
      mma_bf16(acc, a[u].z, b[u].z, a[u].w, b[u].w, wv[u].z, wv[u].w);
    }
  }
  const int c = n0 + 2 * q;
  if (c < n) {
    if (!z0) *reinterpret_cast<float2*>(y + (long long)r0 * n + c) = make_float2(acc[0], acc[1]);
    if (!z1) *reinterpret_cast<float2*>(y + (long long)r1 * n + c) = make_float2(acc[2], acc[3]);
  }
}

// ---- FP8 weight x BF16 activations for a few rows (wo_a: the official BF16 einsum with an FP8 weight): y [m, n] FP32,
// m <= 16, x BF16 rows of stride xs elements; rpg > 0: weight row r reads input elements [(r / rpg) k, +k) of its row
// (block-diagonal groups in one launch). W E4M3 [n, k] + E8M0 per 32x32, converted to BF16 times its scale in registers
// (exact: 3 mantissa bits, a power-of-two scale), BF16 MMA m16n8k16 (exact products, FP32 sums). Warp = one n8 tile;
// per 64-wide span every lane reads 16 bytes (K 16 q .. 16 q + 15) of its weight row and the same K of its two x rows
// and feeds four MMAs (K 16q + 4j .. +3 as the logical 2q, 2q+1, 2q+8, 2q+9, A and B alike). S warps split K as in
// fp8_gemv_mx (summed in slice order; S depends on the shape only, so a row's result does not depend on m). Before,
// a CUDA-core GEMV whose cost grew with m (decode B = 4: 62 -> 140 us per layer).
template <int S>
__global__ __launch_bounds__(256) void fp8_gemv_bx(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                                  float* __restrict__ y, int m, int n, int k, int xs, int rpg) {
  pdl_trigger();
  constexpr int U = 4;
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int wid = threadIdx.x >> 5, slice = wid % S;
  const int n0 = (blockIdx.x * (8 / S) + wid / S) * 8;
  const int kb = k / 32, ns = k / 64, s_lo = slice * ns / S, s_hi = (slice + 1) * ns / S;
  const uint8_t* wr = w + (long long)min(n0 + g, n - 1) * k + 16 * q;
  const uint8_t* wsr = ws + (long long)(min(n0, n - 8) >> 5) * kb + (q >> 1);
  auto ldw = [&](uint4* wv, uint32_t* sb, int sp0) {
#pragma unroll
    for (int u = 0; u < U; ++u)
      if (sp0 + u < s_hi) { wv[u] = __ldg(reinterpret_cast<const uint4*>(wr + (sp0 + u) * 64)); sb[u] = __ldg(wsr + 2 * (sp0 + u)); }
  };
  uint4 wv[U]; uint32_t sb[U];
  if (n0 < n) ldw(wv, sb, s_lo);
  pdl_wait();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  if (n0 < n) {
    const int r0 = g, r1 = g + 8;
    const bool z0 = r0 >= m, z1 = r1 >= m;
    const long long xo = (rpg ? (long long)(n0 / rpg) * k : 0) + 16 * q;
    const __nv_bfloat16* x0p = x + (long long)(z0 ? 0 : r0) * xs + xo;
    const __nv_bfloat16* x1p = x + (long long)(z1 ? 0 : r1) * xs + xo;
    for (int sp0 = s_lo; sp0 < s_hi; sp0 += U) {
      uint4 a[U][2], b[U][2];
#pragma unroll
      for (int u = 0; u < U; ++u)
        if (sp0 + u < s_hi)
          for (int h = 0; h < 2; ++h) {
            a[u][h] = z0 ? make_uint4(0, 0, 0, 0) : __ldg(reinterpret_cast<const uint4*>(x0p + (sp0 + u) * 64 + 8 * h));
            b[u][h] = z1 ? make_uint4(0, 0, 0, 0) : __ldg(reinterpret_cast<const uint4*>(x1p + (sp0 + u) * 64 + 8 * h));
          }
      uint4 nw[U]; uint32_t nsb[U];
      if (sp0 + U < s_hi) ldw(nw, nsb, sp0 + U);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (sp0 + u >= s_hi) break;
        const float sc = __uint_as_float(sb[u] << 23);
        const uint32_t ww[4] = {wv[u].x, wv[u].y, wv[u].z, wv[u].w};
        uint32_t wb[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(ww[j >> 1] >> (16 * (j & 1))), __NV_E4M3);
          const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
          const __nv_bfloat162 v = __floats2bfloat162_rn(f.x * sc, f.y * sc);
          wb[j] = *reinterpret_cast<const uint32_t*>(&v);
        }
        const uint32_t aw[8] = {a[u][0].x, a[u][0].y, a[u][0].z, a[u][0].w, a[u][1].x, a[u][1].y, a[u][1].z, a[u][1].w};
        const uint32_t bw[8] = {b[u][0].x, b[u][0].y, b[u][0].z, b[u][0].w, b[u][1].x, b[u][1].y, b[u][1].z, b[u][1].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) mma_bf16(acc, aw[2 * j], bw[2 * j], aw[2 * j + 1], bw[2 * j + 1], wb[2 * j], wb[2 * j + 1]);
      }
#pragma unroll
      for (int u = 0; u < U; ++u) { wv[u] = nw[u]; sb[u] = nsb[u]; }
    }
  }
  if (S > 1) {
    __shared__ float4 red[8][32];
    red[wid][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    if (slice) return;
    for (int j = 1; j < S; ++j) {
      const float4 o = red[wid + j][lane];
      acc[0] += o.x; acc[1] += o.y; acc[2] += o.z; acc[3] += o.w;
    }
  }
  if (n0 >= n) return;
  const int r0 = g, r1 = g + 8, c = n0 + 2 * q;
  if (r0 < m) *reinterpret_cast<float2*>(y + (long long)r0 * n + c) = make_float2(acc[0], acc[1]);
  if (r1 < m) *reinterpret_cast<float2*>(y + (long long)r1 * n + c) = make_float2(acc[2], acc[3]);
}

// prefill (R > 64): 64-row items for the grouped GEMM (up_gemm; DS41_MOE_GEMM=0 the earlier 32-row up_mx items)
inline bool moe_gemm(int R) { static const bool on = !(std::getenv("DS41_MOE_GEMM") && std::getenv("DS41_MOE_GEMM")[0] == '0'); return on && R > 64; }
// DS41_MOE_ROWS (prefill items, 64 default, or 128: an expert's rows up to 128 share one read of its weights; measured no faster)
inline int gemm_rows() { static const int r = std::getenv("DS41_MOE_ROWS") && std::atoi(std::getenv("DS41_MOE_ROWS")) == 128 ? 128 : 64; return r; }
inline int gr_of(int R, int topk, int E) { (void)topk; (void)E; return R <= 8 ? 16 : moe_gemm(R) ? gemm_rows() : 32; }   // (64-row up_mx items: register-bound, 2.4x slower)
inline int gmax_of(int R, int topk, int E) { const int P = R * topk; return (P < E ? P : E) + P / gr_of(R, topk, E) + 1; }
inline size_t al(size_t b) { return (b + 255) & ~(size_t)255; }
} // namespace

// DS41_MX_S2: the routed-expert scales handed to ds41_moe_mx* are 2-bit coded blobs (set by the loader)
static int g_s2 = 0;
// the expert kernels of one ds41_moe_mx_ex call (C2: the scales in the 2-bit coding)
template <bool C2>
int moe_launch(const uint8_t* X, const uint8_t* XS, int R, const float* rw, const uint8_t* W1, ScaleV S1, const uint8_t* W3, ScaleV S3,
               const uint8_t* W2, ScaleV S2, int I, const float* add, float* y, int topk, int GR, int Gm, int np, int hn, const int* G,
               const Item* items, const int* pairmap, const int* prow, uint8_t* hq, uint8_t* hs, float* out, cudaStream_t st) {
  // R = 1: 8 warps per block split K (decode 31.8 -> 31.3 ms; DS41_UP_KS=4 the earlier 4)
  static const int up_ks = std::getenv("DS41_UP_KS") ? std::atoi(std::getenv("DS41_UP_KS")) : 8;
  if (GR == 16 && up_ks == 8) {
    pdl_launch((up_mx<1, 8, C2>), dim3(dim3(I / 32, Gm)), dim3(256), 0, st, X, XS, W1, S1, W3, S3, I, rw, G, items, prow, hq, hs, topk);
    pdl_launch((down_mx<1, C2>), dim3(dim3(H / 128, Gm)), dim3(128), 0, st, hq, hs, W2, S2, I, G, items, out);
  } else if (GR == 16) {
    pdl_launch((up_mx<1, 4, C2>), dim3(dim3(I / 32, Gm)), dim3(128), 0, st, X, XS, W1, S1, W3, S3, I, rw, G, items, prow, hq, hs, topk);
    pdl_launch((down_mx<1, C2>), dim3(dim3(H / 128, Gm)), dim3(128), 0, st, hq, hs, W2, S2, I, G, items, out);
  } else if (GR == 64 || GR == 128) {
    // DS41_DOWN_GEMM=0: down_mx<4> (the same MMA sequence per output, bitwise equal; 64-row items only)
    static const bool dg = !(std::getenv("DS41_DOWN_GEMM") && std::getenv("DS41_DOWN_GEMM")[0] == '0');
#define MOE_GEMM(RT, UNS, DNS) do { \
      const int usm = UNS * (int)sizeof(UgStageT<RT>), dsm = DNS * (int)sizeof(DgStageT<RT>); \
      static const bool attr = cudaFuncSetAttribute(up_gemm<RT, UNS, C2>, cudaFuncAttributeMaxDynamicSharedMemorySize, usm) == cudaSuccess && \
                               cudaFuncSetAttribute(down_gemm<RT, DNS, C2>, cudaFuncAttributeMaxDynamicSharedMemorySize, dsm) == cudaSuccess; \
      if (!attr) return 3; \
      pdl_launch((up_gemm<RT, UNS, C2>), dim3(dim3(I / 128, Gm)), dim3(256), usm, st, X, XS, W1, S1, W3, S3, I, rw, G, items, prow, hq, hs, topk); \
      if (np > 1 || dg || RT != 64) { \
        for (int pc = 0; pc < np; ++pc) { \
          pdl_launch((down_gemm<RT, DNS, C2>), dim3(dim3(hn / 128, Gm)), dim3(256), dsm, st, hq, hs, W2, S2, I, G, items, out, pc * hn, hn); \
          if (np > 1) pdl_launch(combine_kernel, dim3((unsigned)(((long long)R * (hn / 4) + 255) / 256)), dim3(256), 0, st, (const float*)out, (const int*)pairmap, add, y, R, topk, pc * hn, hn); } } \
      else pdl_launch((down_mx<4, C2>), dim3(dim3(H / 128, Gm)), dim3(128), 0, st, hq, hs, W2, S2, I, G, items, out); } while (0)
    if (GR == 128) MOE_GEMM(128, 2, 3); else MOE_GEMM(64, 3, 3);
#undef MOE_GEMM
  } else if (GR == 32) {
    pdl_launch((up_mx<2, 1, C2>), dim3(dim3(I / 128, Gm)), dim3(128), 0, st, X, XS, W1, S1, W3, S3, I, rw, G, items, prow, hq, hs, topk);
    pdl_launch((down_mx<2, C2>), dim3(dim3(H / 128, Gm)), dim3(128), 0, st, hq, hs, W2, S2, I, G, items, out);
  } else {
    pdl_launch((up_mx<4, 1, C2>), dim3(dim3(I / 128, Gm)), dim3(128), 0, st, X, XS, W1, S1, W3, S3, I, rw, G, items, prow, hq, hs, topk);
    pdl_launch((down_mx<4, C2>), dim3(dim3(H / 128, Gm)), dim3(128), 0, st, hq, hs, W2, S2, I, G, items, out);
  }
  return 0;
}

extern "C" {
// y [M, N] FP32 = A . W^T: A E4M3 [M, K] + UE8M0 [M, K/32], W E4M3 [N, K] + E8M0 [ceil(N/32), K/32]; K % 64 == 0, N % 8 == 0
int ds41_fp8_gemm_mx_ex(const void* aq, const void* as_, const void* w, const void* ws, void* y, int M, int N, int K, int bf16out, cudaStream_t st) {
  if (M < 1 || K % GK || N % 8) return 1;
  // DS41_GEMM_SHAPES=1: count the calls per (M, N, K), printed every 100 calls (to tune on the real shapes)
  static const bool shapes = std::getenv("DS41_GEMM_SHAPES") && std::getenv("DS41_GEMM_SHAPES")[0] == '1';
  if (shapes) {
    static std::map<std::tuple<int, int, int>, int> cnt; static int calls = 0;
    ++cnt[std::make_tuple(M, N, K)];
    if (++calls % 100 == 0) { for (auto& kv : cnt) fprintf(stderr, "[ds41-gemm-shape] M %d N %d K %d: %d\n", std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first), kv.second); fprintf(stderr, "[ds41-gemm-shape] --\n"); }
  }
  static const int env_v = std::getenv("DS41_GEMM_V") ? std::atoi(std::getenv("DS41_GEMM_V")) : 3;
  const int v = g_gemm_version ? g_gemm_version : env_v;
// (measured on the TP3 prefill shapes at 2048 / 4096 rows: 2 stages with two blocks per SM beat 3-4 stages and a 64 x 64
// warp tile; DS41_GEMM_V=1 the original kernel)
  // default (3): fp8_gemm_mx3 with swizzled 64-byte rows and 3 stages (two blocks per SM); a 64-row block tile up to 256
  // rows (the decoder replay's 128: twice the blocks). Measured on the shapes of a 4k CED prefill (ds41-gemm-bench, calls
  // weighted): fp8_gemm_mx2 100.2 ms, this 86.7 ms; padded 96-byte rows with 2 stages 94.1, 3-4 stages with padding or
  // 4 stages (one block per SM) and 64 x 64 warp tiles slower. Bitwise equal to v2 on every shape. DS41_GEMM_V=2: v2.
  if (v == 3) return M <= 256 ? launch_g3<1, 4, 4, 3, 0>(aq, as_, w, ws, y, M, N, K, bf16out, st)
                              : launch_g3<2, 4, 4, 3, 0>(aq, as_, w, ws, y, M, N, K, bf16out, st);
  if (v != 1) {
    const dim3 gr((M + GM - 1) / GM, (N + GN - 1) / GN);
    if (bf16out) fp8_gemm_mx2<true, 2, 4, 4><<<gr, 256, 0, st>>>((const uint8_t*)aq, (const uint8_t*)as_, (const uint8_t*)w, (const uint8_t*)ws, y, M, N, K);
    else fp8_gemm_mx2<false, 2, 4, 4><<<gr, 256, 0, st>>>((const uint8_t*)aq, (const uint8_t*)as_, (const uint8_t*)w, (const uint8_t*)ws, y, M, N, K);
    return (int)cudaGetLastError();
  }
  if (bf16out) { fp8_gemm_mx<true><<<dim3((M + GM - 1) / GM, (N + GN - 1) / GN), 256, 0, st>>>((const uint8_t*)aq, (const uint8_t*)as_, (const uint8_t*)w, (const uint8_t*)ws, y, M, N, K); return (int)cudaGetLastError(); }
  fp8_gemm_mx<false><<<dim3((M + GM - 1) / GM, (N + GN - 1) / GN), 256, 0, st>>>((const uint8_t*)aq, (const uint8_t*)as_, (const uint8_t*)w,
                                                                       (const uint8_t*)ws, y, M, N, K);
  return (int)cudaGetLastError();
}
int ds41_gemm_version(int v) { g_gemm_version = v; return 0; }
int ds41_fp8_gemm_mx(const void* aq, const void* as_, const void* w, const void* ws, float* y, int M, int N, int K, cudaStream_t st) {
  return ds41_fp8_gemm_mx_ex(aq, as_, w, ws, y, M, N, K, 0, st);
}
// the shared memory limit of the in-kernel quantization (once, before any graph capture: not allowed inside one)
int ds41_fp8_gemv_mx_init() {
  static int done = 0;
  if (done) return 0;
  for (auto f : {fp8_gemv_mx<4, 1>, fp8_gemv_mx<4, 2>, fp8_gemv_mx<4, 4>, fp8_gemv_mx<4, 8>})
    if (cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 92 * 1024)) return 4;
  done = 1;
  return 0;
}
int ds41_fp8_gemv_mx_job(const void* x, const void* aq, const void* as_, const void* w, const void* ws, void* y, int m, int n, int k, int ob16, const HcJob* jp, cudaStream_t st);
int ds41_fp8_gemv_mx_q(const void* x, const void* aq, const void* as_, const void* w, const void* ws, void* y, int m, int n, int k, int ob16, cudaStream_t st) {
  HcJob none{}; return ds41_fp8_gemv_mx_job(x, aq, as_, w, ws, y, m, n, k, ob16, &none, st);
}
// ds41_fp8_gemv_mx_q with an mHC mix job (ds41_hc_job.cuh; job->kind 0: none) in the grid's first CTAs
int ds41_fp8_gemv_mx_job(const void* x, const void* aq, const void* as_, const void* w, const void* ws, void* y, int m, int n, int k, int ob16, const HcJob* jp, cudaStream_t st) {
  if (m < 1 || m > 16 || k % 64 || n % 8) return 1;
  HcJob job = *jp;
  job.ctas = hcj::ctas(job.kind, job.T);
  if (job.kind && (job.T < 1 || job.T > 16 || (4 * job.D) % (hcj::SLICES * 2 * hcj::PTH))) return 1;
  // split K 8 ways while every warp keeps >= 2 spans (cold weights, m = 1: wq_b 69.6 -> 67.2 us, wo_b 70.5 -> 67.7,
  // wq_a [1280 x 5120] 38.6 -> 32.3; DS41_GEMV_SPLIT forces S)
  static const int force = std::getenv("DS41_GEMV_SPLIT") ? std::atoi(std::getenv("DS41_GEMV_SPLIT")) : 0;
  int S = 8;
  if (force > 0) S = force;
  else while (S > 1 && k / 64 / S < 2) S /= 2;
  const int sm = aq ? 0 : m * k + m * k / 32;
  if (sm > 48 * 1024 && ds41_fp8_gemv_mx_init()) return 4;
#define GEMV_MX(SS) pdl_launch((fp8_gemv_mx<4, SS>), dim3((n / 8 + 8 / SS - 1) / (8 / SS) + job.ctas), dim3(256), (size_t)sm, st, (const __nv_bfloat16*)x, (const uint8_t*)aq, (const uint8_t*)as_, (const uint8_t*)w, (const uint8_t*)ws, y, m, n, k, ob16, job)
  switch (S) { case 1: GEMV_MX(1); break; case 2: GEMV_MX(2); break; case 4: GEMV_MX(4); break; default: GEMV_MX(8); }
#undef GEMV_MX
  return (int)cudaGetLastError();
}
// largest m k quantized in shared memory (else the caller quantizes first and passes aq / as_)
size_t ds41_fp8_gemv_mx_smem_max() { return 92 * 1024; }
// y [m, n] FP32 = x . W^T, FP8 W [n, k] with BF16 x (row stride xs elements; rpg > 0: block-diagonal groups of rpg
// weight rows, group g reading input elements [g k, +k)); m <= 16, k % 64 == 0, n % 8 == 0
int ds41_fp8_gemv_bx(const void* x, const void* w, const void* ws, float* y, int m, int n, int k, int xs, int rpg, cudaStream_t st) {
  if (m < 1 || m > 16 || k % 64 || n % 8 || (rpg && rpg % 8)) return 1;
  int S = 8;
  while (S > 1 && k / 64 / S < 2) S /= 2;
#define GEMV_BX(SS) pdl_launch((fp8_gemv_bx<SS>), dim3((n / 8 + 8 / SS - 1) / (8 / SS)), dim3(256), 0, st, (const __nv_bfloat16*)x, (const uint8_t*)w, (const uint8_t*)ws, y, m, n, k, xs, rpg)
  switch (S) { case 1: GEMV_BX(1); break; case 2: GEMV_BX(2); break; case 4: GEMV_BX(4); break; default: GEMV_BX(8); }
#undef GEMV_BX
  return (int)cudaGetLastError();
}
int ds41_bf16_gemv_mx(const void* x, const void* w, float* y, int m, int n, int k, cudaStream_t st) {
  if (m < 1 || m > 16 || k % 32 || n % 8) return 1;
  const int warps = 8;
  pdl_launch((bf16_gemv_mx<4>), dim3((n / 8 + warps - 1) / warps), dim3(32 * warps), 0, st, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, y, m, n, k);
  return (int)cudaGetLastError();
}
int ds41_requant_fp8(const void* wq, const void* ws, void* q, void* s, long long numel, cudaStream_t st) {
  if (numel % 32) return 1;
  const long long blocks = numel / 32;
  pdl_launch(requant_fp8_kernel, dim3((unsigned)((blocks + 7) / 8)), dim3(256), 0, st, (const uint8_t*)wq, (const uint8_t*)ws, (uint8_t*)q, (uint8_t*)s, blocks);
  return (int)cudaGetLastError();
}
int ds41_act_quant_fp8(const void* x, void* q, void* s, long long numel, cudaStream_t st) {
  if (numel % 32) return 1;
  const long long blocks = numel / 32;
  pdl_launch(act_quant_fp8_kernel, dim3((unsigned)((blocks + 7) / 8)), dim3(256), 0, st, (const __nv_bfloat16*)x, (uint8_t*)q, (uint8_t*)s, blocks);
  return (int)cudaGetLastError();
}
// prefill items (grouped GEMMs): the down output of every pair, FP32 [P, H], is at most 512 MiB -- more pairs (two
// chunks in one call, Model::moe_groups) run down + combine in column passes over one buffer of H / passes columns
// (each output's MMA sequence and each sum the same; w2's rows split between the passes, read once): the eager peak
// (and the memory left to the page cache) of a 4096-row call
inline int col_passes(int R, int topk, int E) {
  const size_t P = (size_t)R * topk;
  if (!(gr_of(R, topk, E) == 64 || gr_of(R, topk, E) == 128)) return 1;
  // DS41_MOE_OUT_MB (512): the cap (MiB) of that buffer; 1024 keeps a two-chunk (8192-row) call in one pass, so its
  // combine can run inside the reduce-scatter (Model moe_rs)
  static const size_t cap = (size_t)(std::getenv("DS41_MOE_OUT_MB") ? std::atoi(std::getenv("DS41_MOE_OUT_MB")) : 512) << 20;
  int n = 1;
  while (4 * P * H / n > cap && n < 4) n *= 2;
  return n;
}
void ds41_moe_mx_set_s2(int on) { g_s2 = on; }
int ds41_moe_mx_get_s2() { return g_s2; }
size_t ds41_moe_mx_ws_bytes(int R, int I, int topk, int E) {
  const size_t P = (size_t)R * topk;
  return al(16) + al(sizeof(Item) * gmax_of(R, topk, E)) + 2 * al(4 * P) + al(P * I) + al(P * I / 32) + al(4 * P * H / col_passes(R, topk, E));
}
// xq [R, H] E4M3 bytes + xs [R, H/32] UE8M0 (act_quant of x), idx / rw [R, topk], weights in the mx tile layout,
// add [R, H] FP32 or null; y [R, H] FP32.
// skip_combine: everything but the final combine (ds41_moe_mx_combine runs it later, e.g. after the shared expert ran
// on a parallel stream)
int ds41_moe_mx_ex(const void* xq, const void* xs, int R, const int64_t* idx, const float* rw, const void* w1, const void* s1, const void* w3,
                   const void* s3, const void* w2, const void* s2, int I, const float* add, float* y, void* ws, int topk, int E, int skip_combine,
                   cudaStream_t st) {
  if (R < 1 || I % 128 != 0 || E > EMAX || topk < 1) return 1;
  const int P = R * topk, GR = gr_of(R, topk, E), Gm = gmax_of(R, topk, E);
  const int np = col_passes(R, topk, E), hn = H / np;   // (column passes combine as they go)
  if (skip_combine && np > 1) return 1;
  char* p = (char*)ws;
  int* G = (int*)p; p += al(16);
  Item* items = (Item*)p; p += al(sizeof(Item) * Gm);
  int* pairmap = (int*)p; p += al(4 * (size_t)P);
  int* prow = (int*)p; p += al(4 * (size_t)P);
  uint8_t* hq = (uint8_t*)p; p += al((size_t)P * I);
  uint8_t* hs = (uint8_t*)p; p += al((size_t)P * I / 32);
  float* out = (float*)p;
  pdl_launch(group_kernel, dim3(1), dim3(1024), 0, st, idx, P, GR, E, G, items, pairmap, prow);
  const uint8_t *X = (const uint8_t*)xq, *XS = (const uint8_t*)xs;
  const ScaleV S1 = scale_v(s1, g_s2, (long long)E * I / 8, H / 128), S3 = scale_v(s3, g_s2, (long long)E * I / 8, H / 128);
  const ScaleV S2 = scale_v(s2, g_s2, (long long)E * H / 8, I / 128);
  const int lrc = (g_s2 ? moe_launch<true> : moe_launch<false>)(X, XS, R, rw, (const uint8_t*)w1, S1, (const uint8_t*)w3, S3, (const uint8_t*)w2, S2, I,
                                                             add, y, topk, GR, Gm, np, hn, G, items, pairmap, prow, hq, hs, out, st);
  if (lrc) return lrc;
  if (skip_combine || np > 1) return (int)cudaGetLastError();
  const long long n4 = (long long)R * (H / 4);
  pdl_launch(combine_kernel, dim3((unsigned)((n4 + 255) / 256)), dim3(256), 0, st, out, pairmap, add, y, R, topk, 0, H);
  return (int)cudaGetLastError();
}
int ds41_moe_mx(const void* xq, const void* xs, int R, const int64_t* idx, const float* rw, const void* w1, const void* s1, const void* w3,
                const void* s3, const void* w2, const void* s2, int I, const float* add, float* y, void* ws, int topk, int E, cudaStream_t st) {
  return ds41_moe_mx_ex(xq, xs, R, idx, rw, w1, s1, w3, s3, w2, s2, I, add, y, ws, topk, E, 0, st);
}
// the combine of a ds41_moe_mx_ex(skip_combine) call with the same R, I, topk, E and workspace
// rows [r0, r0 + nr) of ds41_moe_mx_combine (add / y: the whole [R, H]; each row's sum is its own, so the rows in
// parts are bitwise the whole call: a part's reduce-scatter can start while the next part combines)
int ds41_moe_mx_combine_rows(void* ws, int R, int I, int topk, int E, const float* add, float* y, int r0, int nr, cudaStream_t st) {
  if (r0 < 0 || nr < 1 || r0 + nr > R) return 1;
  const int P = R * topk, Gm = gmax_of(R, topk, E);
  char* p = (char*)ws;
  p += al(16);
  p += al(sizeof(Item) * Gm);
  int* pairmap = (int*)p; p += al(4 * (size_t)P);
  p += al(4 * (size_t)P);
  p += al((size_t)P * I);
  p += al((size_t)P * I / 32);
  float* out = (float*)p;
  const long long n4 = (long long)nr * (H / 4);
  pdl_launch(combine_kernel, dim3((unsigned)((n4 + 255) / 256)), dim3(256), 0, st, (const float*)out, (const int*)pairmap + (long long)r0 * topk,
             add ? add + (long long)r0 * H : nullptr, y + (long long)r0 * H, nr, topk, 0, H);
  return (int)cudaGetLastError();
}
// the down outputs out [P, H] FP32 and pairmap [P] in a call's workspace (a combine done elsewhere: the reduce-scatter)
int ds41_moe_mx_ws_ptrs(void* ws, int R, int I, int topk, int E, float** out, int** pairmap) {
  const int P = R * topk, Gm = gmax_of(R, topk, E);
  char* p = (char*)ws;
  p += al(16);
  p += al(sizeof(Item) * Gm);
  *pairmap = (int*)p; p += al(4 * (size_t)P);
  p += al(4 * (size_t)P);
  p += al((size_t)P * I);
  p += al((size_t)P * I / 32);
  *out = (float*)p;
  return 0;
}
// column passes of a call (> 1: down and combine interleave inside ds41_moe_mx_ex; no skip_combine)
int ds41_moe_mx_col_passes(int R, int topk, int E) { return col_passes(R, topk, E); }
int ds41_moe_mx_combine(void* ws, int R, int I, int topk, int E, const float* add, float* y, cudaStream_t st) {
  const int P = R * topk, Gm = gmax_of(R, topk, E);
  char* p = (char*)ws;
  p += al(16);
  p += al(sizeof(Item) * Gm);
  int* pairmap = (int*)p; p += al(4 * (size_t)P);
  p += al(4 * (size_t)P);
  p += al((size_t)P * I);
  p += al((size_t)P * I / 32);
  float* out = (float*)p;
  const long long n4 = (long long)R * (H / 4);
  pdl_launch(combine_kernel, dim3((unsigned)((n4 + 255) / 256)), dim3(256), 0, st, (const float*)out, (const int*)pairmap, add, y, R, topk, 0, H);
  return (int)cudaGetLastError();
}
}
