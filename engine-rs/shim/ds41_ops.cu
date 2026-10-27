// SPDX-License-Identifier: MIT
// DeepSeek-V4.1-Flash small fused ops (docs/07 D4), replacing chains of libtorch elementwise kernels in decode:
// ds41_rms_norm: y = bf16(w * (x * rsqrt(mean(x^2) + eps))) per row, FP32 math (ops.rs rms_norm; BF16 x and w).
// ds41_rope: rotate the last rd dims of every row [n, H, d] (BF16) as adjacent complex pairs by pos[i] * freqs[j]
//   (cos/sin in FP32 like Rope::at), products and sums rounded separately like the libtorch expression
//   re*c - im*s, re*s + im*c (inverse: s -> -s); the first d - rd dims are copied. Out of place.
#include "ds41_pdl.cuh"
#include "ablate_clamp.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {
__global__ void rms_norm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                                int n, float eps) {
  pdl_entry();
  const __nv_bfloat16* xr = x + (long long)blockIdx.x * n;
  __nv_bfloat16* yr = y + (long long)blockIdx.x * n;
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = __bfloat162float(xr[i]); s += v * v; }
  __shared__ float red[32];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) red[0] = rsqrtf(v / n + eps);
  }
  __syncthreads();
  const float r = red[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    yr[i] = __float2bfloat16(__fmul_rn(__bfloat162float(w[i]), __fmul_rn(__bfloat162float(xr[i]), r)));
}

// rms_norm for n % 8 == 0, n <= 8 * 2 * 512: every thread keeps up to two 8-element vectors of the row in registers
// (one read of x; the generic kernel above reads it twice), 16-byte loads and stores. With a clamp (docs/05-消融.md
// §3a, the GLM FUSE=2 approach): the row's normalized fp32 values are clamped in registers before the BF16 store --
// per-thread partial dots c_j = z . d_j, the block reduction + count gate of shim/ablate_clamp.cuh, then the update
// subtracted on a second (L2-cached) read of d. Zero extra kernels, no fp32 round-trip of z; cl.d == null is the
// original kernel (same fmuls, same rounding).
__global__ void rms_norm_vec_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                                    int n, float eps, SparkClamp cl) {
  pdl_entry();
  const int nv = n / 8;
  const uint4* xr = reinterpret_cast<const uint4*>(x + (long long)blockIdx.x * n);
  uint4* yr = reinterpret_cast<uint4*>(y + (long long)blockIdx.x * n);
  uint4 v[2];
  float s = 0.f;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = threadIdx.x + u * blockDim.x;
    v[u] = i < nv ? xr[i] : make_uint4(0, 0, 0, 0);
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v[u]);
#pragma unroll
    for (int j = 0; j < 8; ++j) { const float f = __bfloat162float(e[j]); s += f * f; }
  }
  __shared__ float red[32];
  __shared__ float cred[SPARK_CLAMP_MAX_K * 32];
  __shared__ float ce[SPARK_CLAMP_MAX_K];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (threadIdx.x == 0) red[0] = rsqrtf(t / n + eps);
  }
  __syncthreads();
  const float r = red[0];
  float acc[SPARK_CLAMP_MAX_K];
  float zf[2][8];
#pragma unroll
  for (int j = 0; j < SPARK_CLAMP_MAX_K; ++j) acc[j] = 0.f;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = threadIdx.x + u * blockDim.x;
    const bool live = i < nv;
    const uint4 wv = live ? reinterpret_cast<const uint4*>(w)[i] : make_uint4(0, 0, 0, 0);
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&v[u]);
    const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float z = live ? __fmul_rn(__bfloat162float(we[j]), __fmul_rn(__bfloat162float(e[j]), r)) : 0.f;
      zf[u][j] = z;
      if (cl.d) {
#pragma unroll
        for (int q = 0; q < SPARK_CLAMP_MAX_K; ++q) if (q < cl.k) acc[q] += z * __ldg(cl.d + (long long)q * n + i * 8 + j);
      }
    }
  }
  if (cl.d) spark_clamp_reduce(cl, acc, cred, ce, blockIdx.x);   // contains __syncthreads; uniform (cl.d is block-uniform)
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = threadIdx.x + u * blockDim.x;
    if (i >= nv) break;
    uint4 o; __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float z = zf[u][j];
      if (cl.d) {
#pragma unroll
        for (int q = 0; q < SPARK_CLAMP_MAX_K; ++q) if (q < cl.k) z -= ce[q] * __ldg(cl.d + (long long)q * n + i * 8 + j);
      }
      oe[j] = __float2bfloat16(z);
    }
    yr[i] = o;
  }
}

// The clamped sublayer norm for decode-sized calls (DS41_RMS_CLP, default on): rms_norm_vec_kernel (R = 1) /
// rms_norm_cl2_kernel (R = 2) with the k <= KM clamp directions and w loaded into registers (16-byte loads) before the
// PDL wait -- they are constants -- instead of 2 x 16 x k scalar loads per thread inside the dot and update loops (a
// chain of L2 round trips: 14-16 us for one row, on the critical path twice per layer). Per row the same arithmetic in
// the same order and expression forms (sums of squares, reductions, acc[q] += z * d in (u, j, q) order, block reduction
// + count gate row by row, z -= e[q] * d, roundings): bitwise both kernels (ds41-kernel-test).
template <int KM, int R>
__global__ void __launch_bounds__(320) rms_norm_clp_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                           __nv_bfloat16* __restrict__ y, int rows, int n, float eps, SparkClamp cl) {
  pdl_trigger();
  const int nv = n / 8;
  float dv[KM][2][8];
  uint4 wv[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = threadIdx.x + u * blockDim.x;
    const bool live = i < nv;
    wv[u] = live ? __ldg(reinterpret_cast<const uint4*>(w) + i) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int q = 0; q < KM; ++q) {
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
      if (live && q < cl.k) { a = __ldg(reinterpret_cast<const float4*>(cl.d + (long long)q * n + i * 8)); b = __ldg(reinterpret_cast<const float4*>(cl.d + (long long)q * n + i * 8 + 4)); }
      dv[q][u][0] = a.x; dv[q][u][1] = a.y; dv[q][u][2] = a.z; dv[q][u][3] = a.w; dv[q][u][4] = b.x; dv[q][u][5] = b.y; dv[q][u][6] = b.z; dv[q][u][7] = b.w;
    }
  }
  pdl_wait();
  const int r0 = blockIdx.x * R;
  uint4 v[R][2];
  float s[R];
#pragma unroll
  for (int rr = 0; rr < R; ++rr) {
    const int row = (r0 + rr < rows) ? r0 + rr : r0;
    const uint4* xr = reinterpret_cast<const uint4*>(x + (long long)row * n);
    s[rr] = 0.f;
#pragma unroll
    for (int u = 0; u < 2; ++u) { const int i = threadIdx.x + u * blockDim.x; v[rr][u] = i < nv ? xr[i] : make_uint4(0, 0, 0, 0); }
  }
#pragma unroll
  for (int u = 0; u < 2; ++u)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int rr = 0; rr < R; ++rr) { const float f = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&v[rr][u])[j]); s[rr] += f * f; }
  __shared__ float red[R][32];
  __shared__ float cred[SPARK_CLAMP_MAX_K * 32];
  __shared__ float ce[SPARK_CLAMP_MAX_K];
  for (int o = 16; o; o >>= 1)
#pragma unroll
    for (int rr = 0; rr < R; ++rr) s[rr] += __shfl_xor_sync(0xffffffffu, s[rr], o);
  if ((threadIdx.x & 31) == 0)
#pragma unroll
    for (int rr = 0; rr < R; ++rr) red[rr][threadIdx.x >> 5] = s[rr];
  __syncthreads();
  if (threadIdx.x < 32) {
    float t[R];
#pragma unroll
    for (int rr = 0; rr < R; ++rr) t[rr] = threadIdx.x < (blockDim.x >> 5) ? red[rr][threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int rr = 0; rr < R; ++rr) t[rr] += __shfl_xor_sync(0xffffffffu, t[rr], o);
    __syncwarp();
    if (threadIdx.x == 0)
#pragma unroll
      for (int rr = 0; rr < R; ++rr) red[rr][0] = rsqrtf(t[rr] / n + eps);
  }
  __syncthreads();
  float rr_[R];
#pragma unroll
  for (int rr = 0; rr < R; ++rr) rr_[rr] = red[rr][0];
  float acc[R][KM];
  float zf[R][2][8];
#pragma unroll
  for (int rr = 0; rr < R; ++rr)
#pragma unroll
    for (int q = 0; q < KM; ++q) acc[rr][q] = 0.f;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = threadIdx.x + u * blockDim.x;
    const bool live = i < nv;
    const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv[u]);
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int rr = 0; rr < R; ++rr) {
        const float z = live ? __fmul_rn(__bfloat162float(we[j]), __fmul_rn(__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&v[rr][u])[j]), rr_[rr])) : 0.f;
        zf[rr][u][j] = z;
#pragma unroll
        for (int q = 0; q < KM; ++q) if (q < cl.k) acc[rr][q] += z * dv[q][u][j];
      }
  }
  float e[R][KM];
#pragma unroll
  for (int rr = 0; rr < R; ++rr) {
    if (rr > 0 && r0 + rr >= rows) { for (int q = 0; q < KM; ++q) e[rr][q] = 0.f; continue; }
    spark_clamp_reduce(cl, acc[rr], cred, ce, r0 + rr);
#pragma unroll
    for (int q = 0; q < KM; ++q) e[rr][q] = q < cl.k ? ce[q] : 0.f;
    if (R > 1) __syncthreads();   // ce is rewritten by the next row's reduction
  }
#pragma unroll
  for (int rr = 0; rr < R; ++rr) {
    if (r0 + rr >= rows) break;
    uint4* yr = reinterpret_cast<uint4*>(y + (long long)(r0 + rr) * n);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int i = threadIdx.x + u * blockDim.x;
      if (i >= nv) break;
      uint4 o; __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float z = zf[rr][u][j];
#pragma unroll
        for (int q = 0; q < KM; ++q) if (q < cl.k) z -= e[rr][q] * dv[q][u][j];
        oe[j] = __float2bfloat16(z);
      }
      yr[i] = o;
    }
  }
}

// rms_norm_vec_kernel with a clamp of at most KM directions for two rows per CTA (DS41_RMS_CL2, default on): each
// direction element read once for both rows (the clamp's d reads, not x, dominated: 2 x K x n FP32 per row). Per row
// exactly rms_norm_vec_kernel's arithmetic: the same thread -> element mapping, sums of squares, reductions, partial
// dots in (u, j) order, block reduction + count gate (row by row), updates and roundings.
template <int KM>
__global__ void rms_norm_cl2_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                                    int rows, int n, float eps, SparkClamp cl) {
  pdl_entry();
  const int nv = n / 8;
  const int r0 = blockIdx.x * 2;
  const bool has1 = r0 + 1 < rows;
  const uint4* xr0 = reinterpret_cast<const uint4*>(x + (long long)r0 * n);
  const uint4* xr1 = reinterpret_cast<const uint4*>(x + (long long)(has1 ? r0 + 1 : r0) * n);
  uint4 v0[2], v1[2];
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = threadIdx.x + u * blockDim.x;
    v0[u] = i < nv ? xr0[i] : make_uint4(0, 0, 0, 0);
    v1[u] = i < nv ? xr1[i] : make_uint4(0, 0, 0, 0);
    const __nv_bfloat16* e0 = reinterpret_cast<const __nv_bfloat16*>(&v0[u]);
    const __nv_bfloat16* e1 = reinterpret_cast<const __nv_bfloat16*>(&v1[u]);
#pragma unroll
    for (int j = 0; j < 8; ++j) { const float f0 = __bfloat162float(e0[j]); s0 += f0 * f0; const float f1 = __bfloat162float(e1[j]); s1 += f1 * f1; }
  }
  __shared__ float red[2][32];
  __shared__ float cred[SPARK_CLAMP_MAX_K * 32];
  __shared__ float ce[SPARK_CLAMP_MAX_K];
  for (int o = 16; o; o >>= 1) { s0 += __shfl_xor_sync(0xffffffffu, s0, o); s1 += __shfl_xor_sync(0xffffffffu, s1, o); }
  if ((threadIdx.x & 31) == 0) { red[0][threadIdx.x >> 5] = s0; red[1][threadIdx.x >> 5] = s1; }
  __syncthreads();
  if (threadIdx.x < 32) {
    float t0 = threadIdx.x < (blockDim.x >> 5) ? red[0][threadIdx.x] : 0.f, t1 = threadIdx.x < (blockDim.x >> 5) ? red[1][threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) { t0 += __shfl_xor_sync(0xffffffffu, t0, o); t1 += __shfl_xor_sync(0xffffffffu, t1, o); }
    __syncwarp();
    if (threadIdx.x == 0) { red[0][0] = rsqrtf(t0 / n + eps); red[1][0] = rsqrtf(t1 / n + eps); }
  }
  __syncthreads();
  const float rr0 = red[0][0], rr1 = red[1][0];
  float acc0[KM], acc1[KM];
  float zf0[2][8], zf1[2][8];
#pragma unroll
  for (int j = 0; j < KM; ++j) acc0[j] = acc1[j] = 0.f;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = threadIdx.x + u * blockDim.x;
    const bool live = i < nv;
    const uint4 wv = live ? reinterpret_cast<const uint4*>(w)[i] : make_uint4(0, 0, 0, 0);
    const __nv_bfloat16* e0 = reinterpret_cast<const __nv_bfloat16*>(&v0[u]);
    const __nv_bfloat16* e1 = reinterpret_cast<const __nv_bfloat16*>(&v1[u]);
    const __nv_bfloat16* we = reinterpret_cast<const __nv_bfloat16*>(&wv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float z0 = live ? __fmul_rn(__bfloat162float(we[j]), __fmul_rn(__bfloat162float(e0[j]), rr0)) : 0.f;
      const float z1 = live ? __fmul_rn(__bfloat162float(we[j]), __fmul_rn(__bfloat162float(e1[j]), rr1)) : 0.f;
      zf0[u][j] = z0; zf1[u][j] = z1;
#pragma unroll
      for (int q = 0; q < KM; ++q) if (q < cl.k) { const float dv = __ldg(cl.d + (long long)q * n + i * 8 + j); acc0[q] += z0 * dv; acc1[q] += z1 * dv; }
    }
  }
  float ce0[KM], ce1[KM];
  spark_clamp_reduce(cl, acc0, cred, ce, r0);
#pragma unroll
  for (int q = 0; q < KM; ++q) ce0[q] = q < cl.k ? ce[q] : 0.f;
  if (has1) {
    spark_clamp_reduce(cl, acc1, cred, ce, r0 + 1);
#pragma unroll
    for (int q = 0; q < KM; ++q) ce1[q] = q < cl.k ? ce[q] : 0.f;
  }
  uint4* yr0 = reinterpret_cast<uint4*>(y + (long long)r0 * n);
  uint4* yr1 = reinterpret_cast<uint4*>(y + (long long)(r0 + 1) * n);
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int i = threadIdx.x + u * blockDim.x;
    if (i >= nv) break;
    uint4 o0, o1; __nv_bfloat16* oe0 = reinterpret_cast<__nv_bfloat16*>(&o0); __nv_bfloat16* oe1 = reinterpret_cast<__nv_bfloat16*>(&o1);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float z0 = zf0[u][j], z1 = zf1[u][j];
#pragma unroll
      for (int q = 0; q < KM; ++q) if (q < cl.k) { const float dv = __ldg(cl.d + (long long)q * n + i * 8 + j); z0 -= ce0[q] * dv; z1 -= ce1[q] * dv; }
      oe0[j] = __float2bfloat16(z0); oe1[j] = __float2bfloat16(z1);
    }
    yr0[i] = o0;
    if (has1) yr1[i] = o1;
  }
}

// grid (rows, H): one CTA per (row, head), one thread per rotary pair; the plain dims are copied
__global__ void rope_kernel(const __nv_bfloat16* x, const int64_t* __restrict__ pos, const float* __restrict__ freqs,
                            __nv_bfloat16* y, int H, int d, int rd, int inverse) {
  pdl_entry();
  const int i = blockIdx.x;
  const long long base = ((long long)i * H + blockIdx.y) * d;
  const int plain = d - rd;
  if (x != y)   // (in place: the plain dims stay)
    for (int e = threadIdx.x * 8; e < plain; e += blockDim.x * 8)
      *reinterpret_cast<uint4*>(y + base + e) = *reinterpret_cast<const uint4*>(x + base + e);
  const int j = threadIdx.x;
  if (j < rd / 2) {
    const float p = (float)pos[i];
    const float ang = __fmul_rn(p, freqs[j]);
    const float cs = cosf(ang), sn = inverse ? -sinf(ang) : sinf(ang);
    const long long o = base + plain + 2 * j;
    const float re = __bfloat162float(x[o]), im = __bfloat162float(x[o + 1]);
    y[o] = __float2bfloat16(__fsub_rn(__fmul_rn(re, cs), __fmul_rn(im, sn)));
    y[o + 1] = __float2bfloat16(__fadd_rn(__fmul_rn(re, sn), __fmul_rn(im, cs)));
  }
}
// rope_kernel then act_quant32 (shim/ds41.cu: per 32 elements, UE8M0 scale, E4M3 rounding, dequantized BF16) in one
// pass (attention output before wo_a: the inverse rotation and the quantization each read and wrote all of o): one warp
// per 32-block, the rotary pairs computed exactly as rope_kernel (both elements of a pair lie in one block).
__global__ void rope_act_quant_kernel(const __nv_bfloat16* __restrict__ x, const int64_t* __restrict__ pos, const float* __restrict__ freqs,
                                      __nv_bfloat16* __restrict__ out, long long blocks, int H, int d, int rd, int inverse) {
  pdl_entry();
  const long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (b >= blocks) return;
  const long long e = b * 32 + lane;
  const int c = (int)(e % d), plain = d - rd;
  float v;
  if (c < plain) v = __bfloat162float(x[e]);
  else {
    const int j = (c - plain) / 2, odd = (c - plain) & 1;
    const long long o = e - odd;
    const float p = (float)pos[e / ((long long)H * d)];
    const float ang = __fmul_rn(p, freqs[j]);
    const float cs = cosf(ang), sn = inverse ? -sinf(ang) : sinf(ang);
    const float re = __bfloat162float(x[o]), im = __bfloat162float(x[o + 1]);
    v = __bfloat162float(odd ? __float2bfloat16(__fadd_rn(__fmul_rn(re, sn), __fmul_rn(im, cs))) : __float2bfloat16(__fsub_rn(__fmul_rn(re, cs), __fmul_rn(im, sn))));
  }
  float a = fabsf(v);
  for (int k = 16; k; k >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, k));
  a = fmaxf(a, 1e-4f);
  const float t = a * (1.0f / 448.0f);
  const unsigned bits = __float_as_uint(t);
  const int ex = (int)((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0);
  const float sc = __uint_as_float((unsigned)(ex + 127) << 23);
  const float q = fminf(fmaxf(v / sc, -448.f), 448.f);
  const __nv_fp8_e4m3 f8(q);
  out[e] = __float2bfloat16(float(f8) * sc);
}
// MoE routing (model.py Gate, sqrtsoftplus): s = sqrt(softplus(logit)) (softplus threshold 20 like torch), experts =
// top-k of s + bias (descending; lower index first on ties), weights = s[experts] / (sum + 1e-20) (k > 1) * scale.
// One warp per row, E <= 384 (12 per lane), k <= 8.
__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int64_t* __restrict__ idx,
                             float* __restrict__ wts, int T, int E, int K, float scale) {
  pdl_entry();
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= T) return;
  constexpr int PER = 12;
  float sc[PER], b[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int e = lane + 32 * i;
    if (e < E) {
      const float x = logits[(long long)row * E + e];
      sc[i] = sqrtf(x > 20.f ? x : log1pf(expf(x)));
      b[i] = sc[i] + bias[e];
    } else { sc[i] = 0.f; b[i] = -INFINITY; }
  }
  float picked[8]; int pid[8];
  for (int k = 0; k < K; ++k) {
    float bv = -INFINITY; int bi = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < PER; ++i) { const int e = lane + 32 * i; if (b[i] > bv || (b[i] == bv && e < bi)) { bv = b[i]; bi = e; } }
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    pid[k] = bi;
    float s = 0.f;
    // (static indices: a dynamic index would put sc / b in local memory)
#pragma unroll
    for (int i = 0; i < PER; ++i) if (lane + 32 * i == bi) { s = sc[i]; b[i] = -INFINITY; }
    picked[k] = __shfl_sync(0xffffffffu, s, bi & 31);
  }
  if (lane == 0) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += picked[k];
    for (int k = 0; k < K; ++k) {
      idx[(long long)row * K + k] = pid[k];
      wts[(long long)row * K + k] = (K > 1 ? picked[k] / (sum + 1e-20f) : picked[k]) * scale;
    }
  }
}

// Shared-expert SwiGLU on the stacked w1/w3 output gu [T, 2 I] (FP32 GEMV sums): gate / up rounded to BF16 like the
// Linear outputs, clamp (gate <= lim, |up| <= lim), silu(gate) * up rounded to BF16, then act_quant (UE8M0 per 32).
// One warp per 32 outputs.
__global__ void swiglu_quant_kernel(const float* __restrict__ gu, __nv_bfloat16* __restrict__ out, int T, int I, float lim) {
  pdl_entry();
  const long long blk = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (blk >= (long long)T * (I / 32)) return;
  const int row = (int)(blk / (I / 32)), j = (int)(blk % (I / 32)) * 32 + lane;
  float g = __bfloat162float(__float2bfloat16(gu[(long long)row * 2 * I + j]));
  float u = __bfloat162float(__float2bfloat16(gu[(long long)row * 2 * I + I + j]));
  if (lim > 0.f) { g = fminf(g, lim); u = fminf(fmaxf(u, -lim), lim); }
  const float v = __bfloat162float(__float2bfloat16(g / (1.f + expf(-g)) * u));
  float a = fabsf(v);
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  a = fmaxf(a, 1e-4f);
  const unsigned bits = __float_as_uint(a * (1.0f / 448.0f));
  const int e = (int)((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0);
  const float sc = __uint_as_float((unsigned)(e + 127) << 23);
  const __nv_fp8_e4m3 f8(fminf(fmaxf(v / sc, -448.f), 448.f));
  out[(long long)row * I + j] = __float2bfloat16(float(f8) * sc);
}
// FP4 E2M1 fake quantization per `block` (16 or 32) of BF16 x (ops.rs fp4_quant): e4m3 = 0: UE8M0 scale
// 2^ceil(log2(max(amax, 6 * 2^-126) / 6)), 1: E4M3 scale e4m3(max(amax, 6 * 2^-9) / 6); q = e2m1(clamp(x / s, -6, 6))
// (ties as to_e2m1: up at 0.75, 1.75, 3.5, down at 0.25, 1.25, 2.5, 5); out = bf16(q * s). The scalar divisions are
// multiplications by the FP32 reciprocal like libtorch's division by a CPU scalar. One thread per element.
__global__ void fp4_quant_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, long long n, int block, int e4m3) {
  pdl_entry();
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const float v = i < n ? __bfloat162float(x[i]) : 0.f;
  float a = fabsf(v);
  for (int o = block >> 1; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  if (i >= n) return;
  float s;
  if (e4m3) {
    const float t = fmaxf(a, 6.f * 0x1p-9f) * (1.0f / 6.0f);
    s = float(__nv_fp8_e4m3(t));
  } else {
    const unsigned bits = __float_as_uint(fmaxf(a, 6.f * 0x1p-126f) * (1.0f / 6.0f));
    const int e = (int)((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0);
    s = __uint_as_float((unsigned)(e + 127) << 23);
  }
  const float c = fminf(fmaxf(v / s, -6.f), 6.f);
  const float m = fabsf(c);
  float q = 0.f;
  if (m > 0.25f) q = 0.5f;
  if (m >= 0.75f) q = 1.f;
  if (m > 1.25f) q = 1.5f;
  if (m >= 1.75f) q = 2.f;
  if (m > 2.5f) q = 3.f;
  if (m >= 3.5f) q = 4.f;
  if (m > 5.f) q = 6.f;
  y[i] = __float2bfloat16(copysignf(q, c) * s);
}
// Engram gate (model.rs Model::engram after the wkv projection): per (row t, stream c) of h [T, hc, D] BF16 with
// key = kv[t, c D : (c + 1) D], value = kv[t, hc D :] (BF16): rstd = rsqrt(mean(h^2) + eps) rsqrt(mean(key^2) + eps),
// dot = sum(h qk key) rstd / sqrt(D), gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))), out = bf16(h + gate value).
// One CTA per (t, c), fixed summation order: a row's result does not depend on T (libtorch's reductions do).
__global__ void engram_gate_kernel(const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ kv, const float* __restrict__ qk,
                                   int qk_stride, __nv_bfloat16* __restrict__ out, int hc, int D, float eps, float inv_sqrt_d) {
  pdl_entry();
  const int t = blockIdx.x / hc, c = blockIdx.x % hc;
  const __nv_bfloat16* hr = h + ((long long)t * hc + c) * D;
  const __nv_bfloat16* kr = kv + (long long)t * (hc + 1) * D + (long long)c * D;
  const __nv_bfloat16* vr = kv + (long long)t * (hc + 1) * D + (long long)hc * D;
  const float* q = qk + (long long)c * qk_stride;
  float sh = 0.f, sk = 0.f, sd = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float a = __bfloat162float(hr[i]), k = __bfloat162float(kr[i]);
    sh = __fadd_rn(sh, __fmul_rn(a, a)); sk = __fadd_rn(sk, __fmul_rn(k, k)); sd = __fadd_rn(sd, __fmul_rn(__fmul_rn(a, q[i]), k));
  }
  __shared__ float red[3][32];
  for (int o = 16; o; o >>= 1) {
    sh += __shfl_xor_sync(0xffffffffu, sh, o); sk += __shfl_xor_sync(0xffffffffu, sk, o); sd += __shfl_xor_sync(0xffffffffu, sd, o);
  }
  const int wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
  if ((threadIdx.x & 31) == 0) { red[0][wid] = sh; red[1][wid] = sk; red[2][wid] = sd; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.f, b = 0.f, d = 0.f;
    for (int w = 0; w < nw; ++w) { a += red[0][w]; b += red[1][w]; d += red[2][w]; }
    const float rstd = rsqrtf(a * (1.f / D) + eps) * rsqrtf(b * (1.f / D) + eps);
    const float dot = d * rstd * inv_sqrt_d;
    const float g = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    red[0][0] = 1.f / (1.f + expf(-g));
  }
  __syncthreads();
  const float gate = red[0][0];
  __nv_bfloat16* orow = out + ((long long)t * hc + c) * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x)
    orow[i] = __float2bfloat16(__fadd_rn(__bfloat162float(hr[i]), __fmul_rn(gate, __bfloat162float(vr[i]))));
}
// Window commit (model.rs attn_mid): ring[pos[i] % w] = kv[i] for the last min(t, w) rows i, d % 8 == 0. grid rows.
__global__ void win_commit_kernel(__nv_bfloat16* __restrict__ ring, const __nv_bfloat16* __restrict__ kv, const int64_t* __restrict__ pos,
                                  int t, int w, int d) {
  pdl_entry();
  const int i = t - min(t, w) + blockIdx.x;
  const long long slot = pos[i] % w;
  for (int c = threadIdx.x * 8; c < d; c += blockDim.x * 8)
    *reinterpret_cast<uint4*>(ring + slot * d + c) = *reinterpret_cast<const uint4*>(kv + (long long)i * d + c);
}
// DS41_MID_GRAPH: win_commit_kernel for the rows of several sequences in one launch (row i of kv into its sequence's
// ring rings[i], slot pos[i] % w; at most 16 rows, all of them kept); rings is a device array rewritten before each replay
__global__ void win_commit_rows_kernel(const unsigned long long* __restrict__ rings, const __nv_bfloat16* __restrict__ kv,
                                       const int64_t* __restrict__ pos, int w, int d) {
  pdl_entry();
  const int i = blockIdx.x;
  __nv_bfloat16* ring = reinterpret_cast<__nv_bfloat16*>(rings[i]);
  const long long slot = pos[i] % w;
  for (int c = threadIdx.x * 8; c < d; c += blockDim.x * 8)
    *reinterpret_cast<uint4*>(ring + slot * d + c) = *reinterpret_cast<const uint4*>(kv + (long long)i * d + c);
}
} // namespace

extern "C" {
// pinned host memory for staged uploads (cudaHostAlloc: torch's pin_memory(device) warns on every call)
void* ds41_host_alloc(size_t n) { void* p = nullptr; return cudaHostAlloc(&p, n, cudaHostAllocDefault) ? nullptr : p; }
int ds41_host_free(void* p) { return (int)cudaFreeHost(p); }
// dst <- src (n bytes) by the GPU; src may be pinned host memory (GB10: the GPU reads it in place, where cudaMemcpyAsync
// from pinned memory kept the calling thread for the copy's duration)
}  // extern "C"
template <class V>
__global__ void copy_vec_kernel(V* __restrict__ d, const V* __restrict__ s, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) d[i] = s[i];
}
template <class V>
static void copy_vec(void* dst, const void* src, long long n, cudaStream_t st) {
  if (n <= 0) return;
  const long long b = (n + 255) / 256;
  copy_vec_kernel<V><<<(int)(b < 4096 ? b : 4096), 256, 0, st>>>((V*)dst, (const V*)src, n);
}
extern "C" {
int ds41_copy_bytes(void* dst, const void* src, long long n, cudaStream_t st) {
  if (n <= 0) return 0;
  // the widest unit dst and src share the alignment of: bytes up to src's boundary, that unit, bytes for the rest
  const uintptr_t x = (uintptr_t)dst ^ (uintptr_t)src;
  const int u = (x & 15) == 0 ? 16 : (x & 7) == 0 ? 8 : (x & 3) == 0 ? 4 : 1;
  long long h = (u - ((uintptr_t)src & (u - 1))) & (u - 1);
  if (h > n) h = n;
  const long long nv = (n - h) / u, tail = (n - h) - nv * u;
  char* d = (char*)dst; const char* s = (const char*)src;
  copy_vec<unsigned char>(d, s, h, st);
  if (u == 16) copy_vec<uint4>(d + h, s + h, nv, st);
  else if (u == 8) copy_vec<uint2>(d + h, s + h, nv, st);
  else if (u == 4) copy_vec<unsigned int>(d + h, s + h, nv, st);
  else copy_vec<unsigned char>(d + h, s + h, nv, st);
  copy_vec<unsigned char>(d + h + nv * u, s + h + nv * u, tail, st);
  return (int)cudaGetLastError();
}
int ds41_win_commit_rows(const void* rings, const void* kv, const int64_t* pos, int t, int w, int d, cudaStream_t st) {
  if (t < 1 || t > w || d % 8) return 1;
  pdl_launch(win_commit_rows_kernel, dim3(t), dim3(64), 0, st, (const unsigned long long*)rings, (const __nv_bfloat16*)kv, pos, w, d);
  return (int)cudaGetLastError();
}
int ds41_route(const float* logits, const float* bias, int64_t* idx, float* w, int T, int E, int K, float scale, cudaStream_t st) {
  if (T < 1) return 0;
  if (E > 384 || K > 8 || K < 1) return 1;
  pdl_launch(route_kernel, dim3((T + 3) / 4), dim3(128), 0, st, logits, bias, idx, w, T, E, K, scale);
  return (int)cudaGetLastError();
}
int ds41_swiglu_quant(const float* gu, void* out, int T, int I, float lim, cudaStream_t st) {
  if (T < 1) return 0;
  if (I % 32) return 1;
  const long long blocks = (long long)T * (I / 32);
  pdl_launch(swiglu_quant_kernel, dim3((unsigned)((blocks + 7) / 8)), dim3(256), 0, st, gu, (__nv_bfloat16*)out, T, I, lim);
  return (int)cudaGetLastError();
}
int ds41_fp4_quant(const void* x, void* y, long long n, int block, int e4m3, cudaStream_t st) {
  if (n < 1) return 0;
  if ((block != 16 && block != 32) || n % block) return 1;
  pdl_launch(fp4_quant_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, st, (const __nv_bfloat16*)x, (__nv_bfloat16*)y, n, block, e4m3);
  return (int)cudaGetLastError();
}
int ds41_engram_gate(const void* h, const void* kv, const float* qk, int qk_stride, void* out, int T, int hc, int D, float eps, cudaStream_t st) {
  if (T < 1) return 0;
  pdl_launch(engram_gate_kernel, dim3(T * hc), dim3(256), 0, st, (const __nv_bfloat16*)h, (const __nv_bfloat16*)kv, qk, qk_stride,
             (__nv_bfloat16*)out, hc, D, eps, 1.f / sqrtf((float)D));
  return (int)cudaGetLastError();
}
int ds41_win_commit(void* ring, const void* kv, const int64_t* pos, int t, int w, int d, cudaStream_t st) {
  if (t < 1) return 0;
  if (d % 8) return 1;
  pdl_launch(win_commit_kernel, dim3(t < w ? t : w), dim3(64), 0, st, (__nv_bfloat16*)ring, (const __nv_bfloat16*)kv, pos, t, w, d);
  return (int)cudaGetLastError();
}
int ds41_rms_norm(const void* x, const void* w, void* y, int rows, int n, float eps, cudaStream_t st) {
  if (rows < 1) return 0;
  if (n % 8 == 0 && n <= 8 * 2 * 512 && !std::getenv("DS41_RMS_GENERIC")) {
    const int nv = n / 8, threads = ((nv + 1) / 2 + 31) / 32 * 32;
    pdl_launch(rms_norm_vec_kernel, dim3(rows), dim3(threads), 0, st, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)y, n, eps, SparkClamp{});
    return (int)cudaGetLastError();
  }
  const int threads = n >= 2048 ? 512 : n >= 512 ? 256 : 128;
  pdl_launch(rms_norm_kernel, dim3(rows), dim3(threads), 0, st, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)y, n, eps);
  return (int)cudaGetLastError();
}
/// The clamped-site variant (docs/05-消融.md §3a): BF16 x/w and a vec-eligible n (the 5120 sublayer-input shapes),
/// else -1 and the caller falls back to the row kernel. `cl`'s count pointer may be offset to the caller's row base
/// (the encoder two-micro-batch path).
int ds41_rms_norm_cl(const void* x, const void* w, void* y, int rows, int n, float eps, const SparkClamp* cl, cudaStream_t st) {
  if (rows < 1) return 0;
  if (!cl || !cl->d || n % 8 || n > 8 * 2 * 512 || std::getenv("DS41_RMS_GENERIC")) return -1;
  const int nv = n / 8, threads = ((nv + 1) / 2 + 31) / 32 * 32;
  const SparkClamp c = *cl;
  static const bool two = !(std::getenv("DS41_RMS_CL2") && std::getenv("DS41_RMS_CL2")[0] == '0');
  static const bool clp = !(std::getenv("DS41_RMS_CLP") && std::getenv("DS41_RMS_CLP")[0] == '0');
  // one row per CTA (the R = 2 instance agreed bitwise in the stand-alone test but made batched decode nondeterministic
  // under PDL in the clamp golden, cause not found: not used)
  if (clp && c.k <= 4 && threads <= 320 && rows <= 16) {
    pdl_launch((rms_norm_clp_kernel<4, 1>), dim3(rows), dim3(threads), 0, st, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)y, rows, n, eps, c);
    return (int)cudaGetLastError();
  }
  if (two && c.k <= 4 && rows >= 2) {
    pdl_launch(rms_norm_cl2_kernel<4>, dim3((rows + 1) / 2), dim3(threads), 0, st, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)y, rows, n, eps, c);
    return (int)cudaGetLastError();
  }
  pdl_launch(rms_norm_vec_kernel, dim3(rows), dim3(threads), 0, st, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)y, n, eps, c);
  return (int)cudaGetLastError();
}
int ds41_rope(const void* x, const int64_t* pos, const float* freqs, void* y, int rows, int H, int d, int rd, int inverse, cudaStream_t st) {
  if (rows < 1) return 0;
  if (rd % 2 || rd > d || (d - rd) % 8 || rd / 2 > 128) return 1;
  pdl_launch(rope_kernel, dim3(dim3(rows, H)), dim3(128), 0, st, (const __nv_bfloat16*)x, pos, freqs, (__nv_bfloat16*)y, H, d, rd, inverse);
  return (int)cudaGetLastError();
}
// act_quant32(rope(x)) [rows, H, d] -> y (rope_act_quant_kernel; bitwise the two kernels in turn)
int ds41_rope_act_quant(const void* x, const int64_t* pos, const float* freqs, void* y, int rows, int H, int d, int rd, int inverse, cudaStream_t st) {
  if (rows < 1) return 0;
  if (rd % 2 || rd > d || (d - rd) % 8 || d % 32) return 1;
  const long long blocks = (long long)rows * H * d / 32;
  pdl_launch(rope_act_quant_kernel, dim3((unsigned)((blocks + 7) / 8)), dim3(256), 0, st, (const __nv_bfloat16*)x, pos, freqs, (__nv_bfloat16*)y, blocks, H, d, rd, inverse);
  return (int)cudaGetLastError();
}
}

// PDL suspended during prefill (ds41_pdl.cuh): returns the previous value
extern "C" int ds41_pdl_suspend = 0;
extern "C" int ds41_pdl_prefill(int on) { const int was = ds41_pdl_suspend; ds41_pdl_suspend = on; return was; }

// ds41-kernel-test: the clamped norm kernels on random rows, thresholds tight enough that the clamp and the count gate
// trigger: old vec (1 row / CTA), old cl2 (2 rows / CTA), clp<4,1>, clp<4,2>; outputs and counts compared bitwise
// (mismatching elements returned: [0] clp1 vs vec, [1] clp2 vs vec, [2] cl2 vs vec, [3] count mismatches)
extern "C" int ds41_rms_clp_selftest(int rows, int n, int k, int minc, long long* out) {
  const int nv = n / 8, threads = ((nv + 1) / 2 + 31) / 32 * 32;
  std::vector<__nv_bfloat16> hx((size_t)rows * n), hw(n);
  std::vector<float> hd((size_t)k * n), ht(k), hl(k);
  unsigned s = 12345;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 32768.f - 1.f; };
  for (auto& v : hx) v = __float2bfloat16(rnd() * 3.f);
  for (auto& v : hw) v = __float2bfloat16(1.f + 0.2f * rnd());
  for (auto& v : hd) v = rnd() * 0.02f;
  for (int j = 0; j < k; ++j) { ht[j] = 0.05f * (j - 1); hl[j] = -0.05f; }
  __nv_bfloat16 *x, *w, *y[4]; float *d, *t, *l; int* cnt[4];
  cudaMalloc(&x, hx.size() * 2); cudaMalloc(&w, n * 2); cudaMalloc(&d, hd.size() * 4); cudaMalloc(&t, k * 4); cudaMalloc(&l, k * 4);
  for (int i = 0; i < 4; ++i) { cudaMalloc(&y[i], hx.size() * 2); cudaMalloc(&cnt[i], rows * 4); cudaMemset(cnt[i], 0, rows * 4); }
  cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(w, hw.data(), n * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(t, ht.data(), k * 4, cudaMemcpyHostToDevice); cudaMemcpy(l, hl.data(), k * 4, cudaMemcpyHostToDevice);
  for (int pass = 0; pass < 3; ++pass)   // three sites of a pass: reset on the first, the count accumulates
    for (int i = 0; i < 4; ++i) {
      SparkClamp c{d, t, l, nullptr, k, cnt[i], minc, pass == 0};
      if (i == 0) rms_norm_vec_kernel<<<rows, threads>>>(x, w, y[i], n, 1e-6f, c);
      if (i == 1) rms_norm_clp_kernel<4, 1><<<rows, threads>>>(x, w, y[i], rows, n, 1e-6f, c);
      if (i == 2) rms_norm_clp_kernel<4, 2><<<(rows + 1) / 2, threads>>>(x, w, y[i], rows, n, 1e-6f, c);
      if (i == 3) rms_norm_cl2_kernel<4><<<(rows + 1) / 2, threads>>>(x, w, y[i], rows, n, 1e-6f, c);
    }
  if (cudaDeviceSynchronize()) return 1;
  std::vector<uint16_t> hy[4]; std::vector<int> hc[4];
  for (int i = 0; i < 4; ++i) { hy[i].resize(hx.size()); hc[i].resize(rows); cudaMemcpy(hy[i].data(), y[i], hx.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hc[i].data(), cnt[i], rows * 4, cudaMemcpyDeviceToHost); }
  for (int i = 0; i < 4; ++i) out[i] = 0;
  for (size_t e = 0; e < hx.size(); ++e) { out[0] += hy[1][e] != hy[0][e]; out[1] += hy[2][e] != hy[0][e]; out[2] += hy[3][e] != hy[0][e]; }
  for (int r = 0; r < rows; ++r) out[3] += (hc[1][r] != hc[0][r]) + (hc[2][r] != hc[0][r]) + (hc[3][r] != hc[0][r]);
  long long trig = 0; for (int r = 0; r < rows; ++r) trig += hc[0][r];
  out[4] = trig;
  cudaFree(x); cudaFree(w); cudaFree(d); cudaFree(t); cudaFree(l); for (int i = 0; i < 4; ++i) { cudaFree(y[i]); cudaFree(cnt[i]); }
  return 0;
}
