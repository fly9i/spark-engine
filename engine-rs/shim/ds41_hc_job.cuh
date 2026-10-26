// SPDX-License-Identifier: MIT
// DeepSeek-V4.1 mHC mixes of a few rows (decode / verify) as side jobs of the GEMV that follows them (horizontal
// fusion, docs/07 10-26): the first CTAs of the GEMV's grid run the mix's partial sums (hc_part_kernel<1, 64>: 160
// slices of K, four per 256-thread CTA) or its finalize (hc_fin_kernel<true>: one CTA per token), the other CTAs the
// GEMV. The single-CTA latency-bound mix kernels then leave the critical path (their inputs are ready one or two
// kernels before; their outputs are first read by the sublayer's hc_post). Each job does the operations of its kernel
// in the same order: per slice the same thread / K striding, warp shuffle tree and warp order; per component the same
// lane striding over the slices and shuffle tree; the same gate / softmax / Sinkhorn functions (mhc_common.cuh).
#pragma once
#include "pdl.cuh"
#include "mhc_common.cuh"
#include <cuda_bf16.h>

struct HcJob {
  int kind;            // 0 none, 1 partial sums, 2 finalize
  int ctas;            // the grid's first `ctas` CTAs run the job
  const __nv_bfloat16* h; const float* fn;   // kind 1: h [T, 4, D] BF16, fn [24, 4 D]
  float* part;         // [160][T][25]
  int T, D;
  const float* scale; const float* base;     // kind 2
  float* pre; float* post; float* comb;
  float norm_eps, eps; int iters;
};
namespace hcj {
constexpr int HC = 4, MIX = 24, SLICES = 160, PTH = 64;
// CTAs a job takes in a 256-thread grid
__host__ __device__ inline int ctas(int kind, int T) { return kind == 1 ? T * (SLICES / 4) : kind == 2 ? T : 0; }

// hc_part_kernel<1, 64> for (token t, slice s) on the 64 threads tid of a group; red: the group's [2][25] floats.
// Every thread of the CTA calls it (the __syncthreads are the CTA's).
__device__ __forceinline__ void part(const HcJob& j, int t, int s, int tid, float (*red)[MIX + 1]) {
  const int K = HC * j.D, ks = K / SLICES, kb = s * ks;
  float acc[MIX + 1];
#pragma unroll
  for (int b = 0; b <= MIX; ++b) acc[b] = 0.f;
  for (int k = kb + tid * 2; k < kb + ks; k += PTH * 2) {
    const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(j.h + (long long)t * K + k));
    acc[MIX] += xv.x * xv.x + xv.y * xv.y;
#pragma unroll
    for (int b = 0; b < MIX; ++b) {
      const float2 f = *reinterpret_cast<const float2*>(j.fn + (long long)b * K + k);
      acc[b] += xv.x * f.x + xv.y * f.y;
    }
  }
  const int lane = tid & 31, warp = tid >> 5;
#pragma unroll
  for (int b = 0; b <= MIX; ++b) {
    float v = acc[b];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[warp][b] = v;
  }
  __syncthreads();
  for (int b = tid; b <= MIX; b += PTH) {
    float v = 0.f;
    for (int w = 0; w < PTH / 32; ++w) v += red[w][b];
    j.part[((long long)s * j.T + t) * (MIX + 1) + b] = v;
  }
}
// One job CTA of kind 1 (256 threads: slices 4 c .. 4 c + 3 of token t, c = cta % 40).
__device__ __forceinline__ void part_cta(const HcJob& j, int cta) {
  __shared__ float red[4][PTH / 32][MIX + 1];
  const int t = cta / (SLICES / 4), g = threadIdx.x / PTH;
  part(j, t, (cta % (SLICES / 4)) * 4 + g, threadIdx.x % PTH, red[g]);
}
// hc_fin_kernel<true> of token t with 256 threads (component b by warp b % 8, as warp b does there).
__device__ __forceinline__ void fin_cta(const HcJob& j, int t) {
  __shared__ float ms[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int S = SLICES, K = HC * j.D, T = j.T;
  for (int b = warp; b <= MIX; b += 8) {
    float v = 0.f;
    for (int s = lane; s < S; s += 32) v += j.part[((long long)s * T + t) * (MIX + 1) + b];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) ms[b] = v;
  }
  __syncthreads();
  if (warp) return;
  const float val = lane <= MIX ? ms[lane] : 0.f;
  const float rs = rsqrtf(__shfl_sync(0xffffffffu, val, MIX) / K + j.norm_eps);
  if (lane < HC) j.pre[t * HC + lane] = mhc_sigmoid(val * rs * j.scale[0] + j.base[lane]) + j.eps;
  else if (lane < 2 * HC) j.post[t * HC + lane - HC] = 2.f * mhc_sigmoid(val * rs * j.scale[1] + j.base[lane]);
  const int e = lane & 15;
  const float c = __shfl_sync(0xffffffffu, val, 2 * HC + e) * rs * j.scale[2] + j.base[2 * HC + e];
  const float v = mhc_sinkhorn16(mhc_row_softmax16(c, j.eps), j.iters > 0 ? 2 * j.iters - 1 : 0, j.eps);
  if (lane < 16) j.comb[t * HC * HC + e] = v;
}
// The job part of a fused kernel: true when this CTA ran a job (it then returns). Waits for the previous grid first.
__device__ __forceinline__ bool run(const HcJob& j) {
  if (j.kind == 0 || (int)blockIdx.x >= j.ctas) return false;
  pdl_wait();
  if (j.kind == 1) part_cta(j, blockIdx.x); else fin_cta(j, blockIdx.x);
  return true;
}
}  // namespace hcj
