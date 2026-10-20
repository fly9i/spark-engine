// SPDX-License-Identifier: MIT
// mHC (hc_mult 4) pieces shared by GLM (mhc.cu) and DeepSeek-V4.1 (ds41_hc.cu): the gate sigmoid and the 4x4 comb
// matrix's row softmax and Sinkhorn, one element per lane of a 16-lane group (lane & 15 = 4 row + col), in the
// reference's operation order (pairwise shuffle sums, then divide), so either model's results are unchanged.
#pragma once

__device__ __forceinline__ float mhc_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// softmax over the 4 entries of the element's row, + eps
__device__ __forceinline__ float mhc_row_softmax16(float x, float eps) {
  float mx = fmaxf(x, __shfl_xor_sync(0xffffffff, x, 1, 16));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2, 16));
  const float ex = expf(x - mx);
  float sum = ex + __shfl_xor_sync(0xffffffff, ex, 1, 16);
  sum += __shfl_xor_sync(0xffffffff, sum, 2, 16);
  return ex / sum + eps;
}

// `steps` alternating normalizations, columns first (column, row, column, ...): v / (sum + eps). DeepSeek's
// hc_split_sinkhorn with `iters` iterations is 2 iters - 1 steps (GLM: 39).
__device__ __forceinline__ float mhc_sinkhorn16(float v, int steps, float eps) {
  for (int step = 0; step < steps; ++step) {
    const int stride = (step % 2 == 0) ? 4 : 1;
    float t = v + __shfl_xor_sync(0xffffffff, v, stride, 16);
    t += __shfl_xor_sync(0xffffffff, t, 2 * stride, 16);
    v = v / (t + eps);
  }
  return v;
}
