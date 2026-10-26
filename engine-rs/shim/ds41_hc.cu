// SPDX-License-Identifier: MIT
// DeepSeek-V4.1-Flash manifold-constrained hyper-connections (hc_mult 4), fused (docs/07 D4). Semantics of the v0 ops
// (model.rs hc_mixes / hc_pre / hc_post, ops.rs hc_split_sinkhorn), FP32 throughout; only summation orders differ.
//
// ds41_hc_mix: h [T, 4, D] BF16, fn [24, 4D] FP32, scale [3], base [24] ->
//   m = (h_flat . fn^T) * rsqrt(mean(h_flat^2) + eps_norm);  pre = sigmoid(m[0:4] s0 + b) + eps;
//   post = 2 sigmoid(m[4:8] s1 + b);  comb = sinkhorn(softmax_k(m[8:24] s2 + b) + eps)   (comb[j][k], row j)
//   Partial kernel: CTA of 256 threads per TOK tokens and one of S slices of K (fn rows read once per CTA; S > 1 for
//   few tokens), block reduction -> part[s][t][25]; finalize: one thread per token sums the slices in order and runs
//   the 4x4 Sinkhorn (column normalization first, `iters` alternating steps as hc_split_sinkhorn).
// Shared with GLM (mhc.cu): gates, row softmax and Sinkhorn (mhc_common.cuh). Not shared: the partial sums / pre / post
// kernels (GLM keeps the residual streams in FP32 with its own bitwise-qualified fused variants: TP allreduce consumer,
// SPX rotation, packed lanes, Half rounding; here the streams are BF16, so one kernel would change one side's order).
// ds41_hc_pre: out [T, D] BF16 = sum_j pre[t][j] h[t][j]    (j ascending, FP32)
// ds41_hc_post: y [T, 4, D] BF16, y[t][k] = post[t][k] x[t] + sum_j comb[t][j][k] res[t][j]
#include "ds41_pdl.cuh"
#include "mhc_common.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>

namespace {
constexpr int HC = 4, MIX = 24, THREADS = 256;

template <int TOK, int TH = THREADS>
__global__ __launch_bounds__(TH) void hc_part_kernel(const __nv_bfloat16* __restrict__ h, const float* __restrict__ fn,
    float* __restrict__ part, int T, int D) {
  pdl_entry();
  const int K = HC * D, S = gridDim.y, ks = K / S, kb = blockIdx.y * ks;
  const int t0 = blockIdx.x * TOK;
  float acc[TOK][MIX + 1];
#pragma unroll
  for (int a = 0; a < TOK; ++a)
#pragma unroll
    for (int b = 0; b <= MIX; ++b) acc[a][b] = 0.f;
  for (int k = kb + threadIdx.x * 2; k < kb + ks; k += TH * 2) {
    float2 xv[TOK];
#pragma unroll
    for (int a = 0; a < TOK; ++a) {
      const int t = t0 + a;
      xv[a] = t < T ? __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h + (long long)t * K + k)) : make_float2(0.f, 0.f);
      acc[a][MIX] += xv[a].x * xv[a].x + xv[a].y * xv[a].y;
    }
#pragma unroll
    for (int b = 0; b < MIX; ++b) {
      const float2 f = *reinterpret_cast<const float2*>(fn + (long long)b * K + k);
#pragma unroll
      for (int a = 0; a < TOK; ++a) acc[a][b] += xv[a].x * f.x + xv[a].y * f.y;
    }
  }
  // block reduction: warp shuffle, then across the 8 warps
  __shared__ float red[TH / 32][TOK][MIX + 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int a = 0; a < TOK; ++a)
#pragma unroll
    for (int b = 0; b <= MIX; ++b) {
      float v = acc[a][b];
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[warp][a][b] = v;
    }
  __syncthreads();
  for (int i = threadIdx.x; i < TOK * (MIX + 1); i += TH) {
    const int a = i / (MIX + 1), b = i % (MIX + 1), t = t0 + a;
    if (t >= T) continue;
    float v = 0.f;
    for (int w = 0; w < TH / 32; ++w) v += red[w][a][b];
    part[((long long)blockIdx.y * T + t) * (MIX + 1) + b] = v;
  }
}

// Many tokens (prefill): a tiled GEMM over K. CTA = 32 tokens x the 25 outputs (24 mixes + the square sum), K in
// chunks of 128 through shared memory (x BF16 32 x 128, fn FP32 24 x 128); 256 threads, each up to 4 (token, output)
// pairs (800 pairs per CTA). Writes part[0][t][25] (S = 1).
constexpr int TT = 32, TK = 128;
__global__ __launch_bounds__(256) void hc_part_tiled(const __nv_bfloat16* __restrict__ h, const float* __restrict__ fn,
                                                    float* __restrict__ part, int T, int D) {
  __shared__ float xs[TT][TK + 1];
  __shared__ float fs[MIX][TK + 1];
  const int K = HC * D, t0 = blockIdx.x * TT, tid = threadIdx.x;
  constexpr int PAIRS = TT * (MIX + 1);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int pt[4], po[4];
  for (int j = 0; j < 4; ++j) { const int p = tid + 256 * j; pt[j] = p < PAIRS ? p / (MIX + 1) : -1; po[j] = p < PAIRS ? p % (MIX + 1) : 0; }
  for (int k0 = 0; k0 < K; k0 += TK) {
    for (int i = tid; i < TT * TK / 2; i += 256) {
      const int r = i / (TK / 2), c = (i % (TK / 2)) * 2, t = t0 + r;
      const float2 v = t < T ? __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h + (long long)t * K + k0 + c)) : make_float2(0.f, 0.f);
      xs[r][c] = v.x; xs[r][c + 1] = v.y;
    }
    for (int i = tid; i < MIX * TK; i += 256) { const int r = i / TK, c = i % TK; fs[r][c] = fn[(long long)r * K + k0 + c]; }
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
      if (pt[j] < 0) continue;
      const float* xr = xs[pt[j]];
      float a = acc[j];
      if (po[j] == MIX) { for (int c = 0; c < TK; ++c) a += xr[c] * xr[c]; }
      else { const float* fr = fs[po[j]]; for (int c = 0; c < TK; ++c) a += xr[c] * fr[c]; }
      acc[j] = a;
    }
    __syncthreads();
  }
  for (int j = 0; j < 4; ++j)
    if (pt[j] >= 0 && t0 + pt[j] < T) part[(long long)(t0 + pt[j]) * (MIX + 1) + po[j]] = acc[j];
}

// Many tokens (prefill), register-blocked: CTA = 64 tokens, 128 threads; thread (tg, mg) = tokens 4 tg .. 4 tg + 3 x
// mixes 3 mg .. 3 mg + 2 (12 accumulators; mg 0 also the 4 square sums). K in stages of 64 through shared memory, x
// stored K-major as FP32 (one float4 per thread per k). part[0][t][25] (S = 1). The FP32 sums run along K per
// accumulator (order differs from the generic kernel's lane/warp tree; same values up to rounding).
constexpr int RT = 64, RK = 64;
__global__ __launch_bounds__(128) void hc_part_reg(const __nv_bfloat16* __restrict__ h, const float* __restrict__ fn,
                                                  float* __restrict__ part, int T, int D) {
  pdl_entry();
  __shared__ __align__(16) float xs[RK][RT + 4];
  __shared__ float fs[RK][MIX + 1];
  const int K = HC * D, t0 = blockIdx.x * RT, tid = threadIdx.x, tg = tid >> 3, mg = tid & 7;
  // K stages of this CTA's slice (blockIdx.y of gridDim.y)
  const int nst = K / RK, st0 = blockIdx.y * nst / gridDim.y, st1 = (blockIdx.y + 1) * nst / gridDim.y;
  float acc[4][3], sq[4];
#pragma unroll
  for (int a = 0; a < 4; ++a) { sq[a] = 0.f; for (int b = 0; b < 3; ++b) acc[a][b] = 0.f; }
  for (int k0 = st0 * RK; k0 < st1 * RK; k0 += RK) {
    // x: 64 tokens x 64 k (BF16 pairs), transposed into xs[k][t]
#pragma unroll 4
    for (int i = tid; i < RT * RK / 2; i += 128) {
      const int r = i / (RK / 2), c = (i % (RK / 2)) * 2, t = t0 + r;
      const float2 v = t < T ? __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h + (long long)t * K + k0 + c)) : make_float2(0.f, 0.f);
      xs[c][r] = v.x; xs[c + 1][r] = v.y;
    }
    for (int i = tid; i < MIX * RK; i += 128) { const int m = i / RK, c = i % RK; fs[c][m] = fn[(long long)m * K + k0 + c]; }
    __syncthreads();
#pragma unroll 8
    for (int c = 0; c < RK; ++c) {
      const float4 x = *reinterpret_cast<const float4*>(&xs[c][4 * tg]);
      const float f0 = fs[c][3 * mg], f1 = fs[c][3 * mg + 1], f2 = fs[c][3 * mg + 2];
      const float xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        acc[a][0] = fmaf(xv[a], f0, acc[a][0]); acc[a][1] = fmaf(xv[a], f1, acc[a][1]); acc[a][2] = fmaf(xv[a], f2, acc[a][2]);
        if (mg == 0) sq[a] = fmaf(xv[a], xv[a], sq[a]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int t = t0 + 4 * tg + a;
    if (t >= T) continue;
    float* pr = part + ((long long)blockIdx.y * T + t) * (MIX + 1);
    pr[3 * mg] = acc[a][0]; pr[3 * mg + 1] = acc[a][1]; pr[3 * mg + 2] = acc[a][2];
    if (mg == 0) pr[MIX] = sq[a];
  }
}

// hc_part_reg with the same FMA sequence per accumulator (3 mixes per thread, mg 0 also the square sums, the same K
// slices; fmaf along K in order: bitwise equal), latency hidden: more CTAs and threads (T = 683 gave hc_part_reg 66
// CTAs of 4 warps on 48 SMs), x kept as its BF16 rows and fn rows FP32 in shared memory, both loaded by cp.async
// R2S stages ahead (hc_part_reg loads, transposes with 8-way bank conflicts, then waits on every stage). The thread
// mapping (TPT tokens per thread) is free: each accumulator's FMA chain is the same.
constexpr int R2K = 64, R2S = 3;
__device__ __forceinline__ void hc_cp16(void* s, const void* g, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"((unsigned)__cvta_generic_to_shared(s)), "l"(g), "r"(ok ? 16 : 0));
}
// TPT tokens per thread (thread = TPT tokens x 3 mixes): the CTA is (R2T / TPT) x 8 threads.
template <int R2T, int TPT>
__global__ __launch_bounds__(R2T / TPT * 8) void hc_part_reg2(const __nv_bfloat16* __restrict__ h, const float* __restrict__ fn,
                                                       float* __restrict__ part, int T, int D) {
  constexpr int NTH = R2T / TPT * 8, XP = R2K + 8, FP = R2K + 4;   // padded rows: x 144 bytes, fn 272 (the 8 mg lanes' rows 3 mg on distinct banks)
  pdl_entry();
  __shared__ __align__(16) __nv_bfloat16 xs[R2S][R2T][XP];
  __shared__ __align__(16) float fs[R2S][MIX][FP];
  const int K = HC * D, t0 = blockIdx.x * R2T, tid = threadIdx.x, tg = tid >> 3, mg = tid & 7;
  const int nst = K / R2K, st0 = blockIdx.y * nst / gridDim.y, st1 = (blockIdx.y + 1) * nst / gridDim.y, ns = st1 - st0;
  auto load = [&](int s, int k0) {
    for (int i = tid; i < R2T * R2K / 8; i += NTH) {        // x: 8 bf16 per 16-byte copy
      const int r = i / (R2K / 8), c = (i % (R2K / 8)) * 8, t = t0 + r;
      hc_cp16(&xs[s][r][c], h + (long long)(t < T ? t : 0) * K + k0 + c, t < T);
    }
    for (int i = tid; i < MIX * R2K / 4; i += NTH) {         // fn: 4 floats per copy
      const int m = i / (R2K / 4), c = (i % (R2K / 4)) * 4;
      hc_cp16(&fs[s][m][c], fn + (long long)m * K + k0 + c, true);
    }
  };
  float acc[TPT][3], sq[TPT];
#pragma unroll
  for (int a = 0; a < TPT; ++a) { sq[a] = 0.f; for (int b = 0; b < 3; ++b) acc[a][b] = 0.f; }
#pragma unroll
  for (int s = 0; s < R2S - 1; ++s) { if (s < ns) load(s, (st0 + s) * R2K); asm volatile("cp.async.commit_group;\n"); }
  for (int j = 0; j < ns; ++j) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(R2S - 2) : "memory");
    __syncthreads();
    if (j + R2S - 1 < ns) load((j + R2S - 1) % R2S, (st0 + j + R2S - 1) * R2K);
    asm volatile("cp.async.commit_group;\n");
    const int s = j % R2S;
#pragma unroll 4
    for (int c = 0; c < R2K; c += 2) {
      float2 xv[TPT];
#pragma unroll
      for (int a = 0; a < TPT; ++a) xv[a] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xs[s][TPT * tg + a][c]));
      const float2 f0 = *reinterpret_cast<const float2*>(&fs[s][3 * mg][c]), f1 = *reinterpret_cast<const float2*>(&fs[s][3 * mg + 1][c]),
                   f2 = *reinterpret_cast<const float2*>(&fs[s][3 * mg + 2][c]);
#pragma unroll
      for (int a = 0; a < TPT; ++a) {
        acc[a][0] = fmaf(xv[a].x, f0.x, acc[a][0]); acc[a][1] = fmaf(xv[a].x, f1.x, acc[a][1]); acc[a][2] = fmaf(xv[a].x, f2.x, acc[a][2]);
        if (mg == 0) sq[a] = fmaf(xv[a].x, xv[a].x, sq[a]);
      }
#pragma unroll
      for (int a = 0; a < TPT; ++a) {
        acc[a][0] = fmaf(xv[a].y, f0.y, acc[a][0]); acc[a][1] = fmaf(xv[a].y, f1.y, acc[a][1]); acc[a][2] = fmaf(xv[a].y, f2.y, acc[a][2]);
        if (mg == 0) sq[a] = fmaf(xv[a].y, xv[a].y, sq[a]);
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#pragma unroll
  for (int a = 0; a < TPT; ++a) {
    const int t = t0 + TPT * tg + a;
    if (t >= T) continue;
    float* pr = part + ((long long)blockIdx.y * T + t) * (MIX + 1);
    pr[3 * mg] = acc[a][0]; pr[3 * mg + 1] = acc[a][1]; pr[3 * mg + 2] = acc[a][2];
    if (mg == 0) pr[MIX] = sq[a];
  }
}

// Finalize: component b < 25 of one token (summed over the S slices) sits in lane b of one warp; that warp computes
// pre (lanes 0-3), post (lanes 4-7) and the 4x4 Sinkhorn with one element per lane (lane & 15 = 4 j + k, row j;
// row / column sums by shuffles inside groups of 16): the alternating normalizations run in parallel.
// WIDE (few tokens, many slices): one CTA per token, warp b sums component b over the slices (fixed lane order and
// shuffle tree: deterministic); otherwise one warp per token, lane b sums component b over the slices in order.
template <bool WIDE>
__global__ void hc_fin_kernel(const float* __restrict__ part, int S, const float* __restrict__ scale, const float* __restrict__ base,
    float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb, int T, int D, float norm_eps, float eps, int iters) {
  pdl_entry();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int t = WIDE ? blockIdx.x : blockIdx.x * (blockDim.x >> 5) + warp;
  const int K = HC * D;
  float val = 0.f;
  if (WIDE) {
    __shared__ float ms[32];
    if (warp <= MIX) {
      float v = 0.f;
      for (int s = lane; s < S; s += 32) v += part[((long long)s * T + t) * (MIX + 1) + warp];
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) ms[warp] = v;
    }
    __syncthreads();
    if (warp) return;
    val = lane <= MIX ? ms[lane] : 0.f;
  } else {
    if (t >= T) return;
    if (lane <= MIX) for (int s = 0; s < S; ++s) val += part[((long long)s * T + t) * (MIX + 1) + lane];
  }
  const float rs = rsqrtf(__shfl_sync(0xffffffffu, val, MIX) / K + norm_eps);
  if (lane < HC) pre[t * HC + lane] = mhc_sigmoid(val * rs * scale[0] + base[lane]) + eps;
  else if (lane < 2 * HC) post[t * HC + lane - HC] = 2.f * mhc_sigmoid(val * rs * scale[1] + base[lane]);
  const int e = lane & 15;
  const float c = __shfl_sync(0xffffffffu, val, 2 * HC + e) * rs * scale[2] + base[2 * HC + e];
  // softmax over k (row) + eps, then the Sinkhorn: `iters` iterations = 2 iters - 1 alternating steps, columns first
  // (shared with GLM: mhc_common.cuh)
  const float v = mhc_sinkhorn16(mhc_row_softmax16(c, eps), iters > 0 ? 2 * iters - 1 : 0, eps);
  if (lane < 16) comb[t * HC * HC + e] = v;
}

__global__ void hc_pre_kernel(const __nv_bfloat16* __restrict__ h, const float* __restrict__ pre, __nv_bfloat16* __restrict__ out, int T, int D) {
  pdl_entry();
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // pair index
  if (i >= (long long)T * D / 2) return;
  const int t = (int)(i / (D / 2)), d = (int)(i % (D / 2)) * 2;
  float2 s = make_float2(0.f, 0.f);
  for (int j = 0; j < HC; ++j) {
    const float p = pre[t * HC + j];
    const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h + ((long long)t * HC + j) * D + d));
    s.x += p * v.x; s.y += p * v.y;
  }
  *reinterpret_cast<__nv_bfloat162*>(out + (long long)t * D + d) = __floats2bfloat162_rn(s.x, s.y);
}

__global__ void hc_post_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ res, const float* __restrict__ post,
                               const float* __restrict__ comb, __nv_bfloat16* __restrict__ y, int T, int D) {
  pdl_entry();
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // pair index over [T, D/2]
  if (i >= (long long)T * D / 2) return;
  const int t = (int)(i / (D / 2)), d = (int)(i % (D / 2)) * 2;
  const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + (long long)t * D + d));
  float2 r[HC];
  for (int j = 0; j < HC; ++j) r[j] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(res + ((long long)t * HC + j) * D + d));
  for (int k = 0; k < HC; ++k) {
    float2 b = make_float2(0.f, 0.f);
    for (int j = 0; j < HC; ++j) { const float c = comb[(t * HC + j) * HC + k]; b.x += c * r[j].x; b.y += c * r[j].y; }
    const float p = post[t * HC + k];
    *reinterpret_cast<__nv_bfloat162*>(y + ((long long)t * HC + k) * D + d) = __floats2bfloat162_rn(p * xv.x + b.x, p * xv.y + b.y);
  }
}
} // namespace

extern "C" {
// slices of K for T tokens (K / S must stay a multiple of 2 * THREADS)
// few tokens: 160 slices of 64 threads (160 CTAs: the fn rows stream from all SMs; 40 x 256 left 8 idle and took 15 us
// per call at T = 1), else 40 / 8 / 1 slices of 256 threads
static bool hc_narrow(int T, int D) { return T <= 16 && (HC * D) % (160 * 2 * 64) == 0 && !std::getenv("DS41_HC_WIDE_PART"); }
// many tokens (register-blocked path): K split so that the 64-token CTAs number about 192
static bool hc_reg(int T, int D) { return T > 64 && (HC * D) % RK == 0 && !std::getenv("DS41_HC_PART4"); }
// DS41_HC_REG1=1: hc_part_reg (hc_part_reg2 is bitwise equal)
static bool hc_reg2() { static const bool on = !(std::getenv("DS41_HC_REG1") && std::getenv("DS41_HC_REG1")[0] == '1'); return on; }
// hc_part_reg2 shapes by T (bitwise equal whichever; measured, random rows, D 5120): T <= 256 32 tokens x 2 per thread
// (the replay's 128: 157 -> 44 us), T <= 1536 16 x 1 (a part's 683 local rows: 356 -> 180 us, 1366: 736 -> 341),
// above 64 x 2 (2048: 501 -> 502, hc_part_reg's level)
static void hc_part_launch(const void* h, const float* fn, float* part, int T, int D, int S, cudaStream_t st) {
  const __nv_bfloat16* x = (const __nv_bfloat16*)h;
  if (!hc_reg2()) pdl_launch(hc_part_reg, dim3(dim3((T + RT - 1) / RT, S)), dim3(128), 0, st, x, fn, part, T, D);
  else if (T <= 256) pdl_launch((hc_part_reg2<32, 2>), dim3(dim3((T + 31) / 32, S)), dim3(128), 0, st, x, fn, part, T, D);
  else if (T <= 1536) pdl_launch((hc_part_reg2<16, 1>), dim3(dim3((T + 15) / 16, S)), dim3(128), 0, st, x, fn, part, T, D);
  else pdl_launch((hc_part_reg2<64, 2>), dim3(dim3((T + 63) / 64, S)), dim3(256), 0, st, x, fn, part, T, D);
}
static int hc_reg_slices(int T) { const int blocks = (T + RT - 1) / RT; const int s = (192 + blocks - 1) / blocks; return s < 1 ? 1 : s > 8 ? 8 : s; }
int ds41_hc_slices(int T, int D) {
  if (hc_narrow(T, D)) return 160;
  if (hc_reg(T, D)) return hc_reg_slices(T);
  const int K = HC * D; int S = T <= 16 ? 40 : T <= 64 ? 8 : 1; while (S > 1 && K % (S * 2 * THREADS)) --S; return S;
}
// part: workspace of ds41_hc_slices(T, D) * T * 25 floats
// ds41_hc_mix of T rows exactly as a call over `rows` rows would compute them (the kernel and K slices that `rows` select;
// sequence-parallel prefill: each rank's share of a chunk keeps the full chunk's sums). Only the register-blocked path
// (rows > 64): its per-token result depends on S alone. part: S(rows) * T * 25 floats.
int ds41_hc_mix_as(const void* h, const float* fn, const float* scale, const float* base, float* pre, float* post, float* comb,
                   float* part, int T, int rows, int D, float norm_eps, float eps, int iters, cudaStream_t st) {
  if (T < 1 || D % 2 || !hc_reg(rows, D)) return 1;
  const int S = hc_reg_slices(rows);
  hc_part_launch(h, fn, part, T, D, S, st);
  pdl_launch((hc_fin_kernel<false>), dim3((T + 3) / 4), dim3(128), 0, st, part, S, scale, base, pre, post, comb, T, D, norm_eps, eps, iters);
  return (int)cudaGetLastError();
}
int ds41_hc_mix(const void* h, const float* fn, const float* scale, const float* base, float* pre, float* post, float* comb,
                float* part, int T, int D, float norm_eps, float eps, int iters, cudaStream_t st) {
  if (T < 1 || D % 2) return 1;
  const int S = ds41_hc_slices(T, D);
  if (hc_narrow(T, D)) pdl_launch((hc_part_kernel<1, 64>), dim3(dim3(T, S)), dim3(64), 0, st, (const __nv_bfloat16*)h, fn, part, T, D);
  else if (T <= 16) pdl_launch((hc_part_kernel<1>), dim3(dim3(T, S)), dim3(THREADS), 0, st, (const __nv_bfloat16*)h, fn, part, T, D);
  else if (hc_reg(T, D))   // many tokens: register-blocked, K split in S slices
    hc_part_launch(h, fn, part, T, D, S, st);
  else if (T <= 256) pdl_launch((hc_part_kernel<4>), dim3(dim3((T + 3) / 4, S)), dim3(THREADS), 0, st, (const __nv_bfloat16*)h, fn, part, T, D);
  else if (std::getenv("DS41_HC_TILED")) hc_part_tiled<<<(T + TT - 1) / TT, 256, 0, st>>>((const __nv_bfloat16*)h, fn, part, T, D);
  else pdl_launch((hc_part_kernel<4>), dim3(dim3((T + 3) / 4, S)), dim3(THREADS), 0, st, (const __nv_bfloat16*)h, fn, part, T, D);
  if (S > 8) pdl_launch((hc_fin_kernel<true>), dim3(T), dim3(1024), 0, st, part, S, scale, base, pre, post, comb, T, D, norm_eps, eps, iters);
  else pdl_launch((hc_fin_kernel<false>), dim3((T + 3) / 4), dim3(128), 0, st, part, S, scale, base, pre, post, comb, T, D, norm_eps, eps, iters);
  return (int)cudaGetLastError();
}
int ds41_hc_pre(const void* h, const float* pre, void* out, int T, int D, cudaStream_t st) {
  const long long n = (long long)T * D / 2;
  pdl_launch(hc_pre_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, st, (const __nv_bfloat16*)h, pre, (__nv_bfloat16*)out, T, D);
  return (int)cudaGetLastError();
}
int ds41_hc_post(const void* x, const void* res, const float* post, const float* comb, void* y, int T, int D, cudaStream_t st) {
  const long long n = (long long)T * D / 2;
  pdl_launch(hc_post_kernel, dim3((unsigned)((n + 255) / 256)), dim3(256), 0, st, (const __nv_bfloat16*)x, (const __nv_bfloat16*)res, post, comb, (__nv_bfloat16*)y, T, D);
  return (int)cudaGetLastError();
}
}
