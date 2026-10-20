// SPDX-License-Identifier: MIT
// DeepSeek-V4.1-Flash routed experts (docs/07 D4), this rank's slice of the expert intermediate dimension I:
//   y[r] = sum_k (in slot order) w2_e . q(bf16(silu(bf16(w1_e . x_r)) * bf16(w3_e . x_r) * route_w[r][k]))   (FP32)
// with e = idx[r][k], SwiGLU limit 10 (gate <= 10, -10 <= up <= 10), q = act_quant (UE8M0 per 32) as the official
// Expert.forward + Linear. x is the act-quantized input [R, H] BF16; the partial sum over the I slice is summed across
// ranks by the caller.
//
// Numerics: MXFP4 weights (E2M1 x E8M0 per 32) are exact in BF16, the act-quantized activations (E4M3 x power of two)
// too, so BF16 MMA with FP32 accumulation forms every product exactly; only the summation order differs from the
// dequantize + FP32 matmul reference (L1).
//
// Layout: the P = R * 6 (row, expert) pairs get compact rows grouped by expert (group_kernel, one CTA; the order of
// rows inside an expert does not matter: every compact row is computed independently). Items = at most GR rows of one
// expert. up_kernel: gate and up of 32 intermediate columns per warp (4 + 4 n8 tiles) for all rows of an item, then
// SwiGLU, route weight and act_quant in registers (32 columns = one quant block) -> h [P, I] BF16. down_kernel:
// out [P, H] FP32 = h . w2_e^T. combine_kernel: y[r] = sum over k of out[pairmap[r][k]] in slot order (deterministic).
//
// Weight layout (packed at load, kern.rs moe_pack): every matrix [rows, K] as tiles of 8 rows; per tile and 128-wide
// K step the 8 rows x 64 bytes are contiguous in lane order (row g, quarter q: 16 bytes), the scales likewise
// (8 rows x 4): a warp's read of one tile and step is 512 contiguous bytes, and a tile streams contiguously over K.
//   w: [rows / 8][K / 128][8][4][16] bytes, s: [rows / 8][K / 128][8][4]
// MMA fragment K mapping (m16n8k16): the logical k of a lane's fragment registers ({2q, 2q+1, 2q+8, 2q+9}, q = lane % 4)
// is mapped to the physical k = base + 32 q + 4 i + {0, 1, 2, 3} for substep i: A and B use the same mapping, so the
// products are the same, and every lane reads 16 contiguous weight bytes (32 FP4 values = one scale block) per step.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cub/block/block_scan.cuh>

namespace {
constexpr int EMAX = 384, H = 5120;   // experts (384 main, 128 DSpark) and top-k (6 / 3) are runtime parameters

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// E8M0 scale as BF16x2 (2^(b-127); b = 0 is the BF16 subnormal 2^-127)
__device__ __forceinline__ uint32_t scale2(uint8_t b) { const uint32_t h = b ? (uint32_t)b << 7 : 0x40u; return h | (h << 16); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }
// byte (two E2M1, low nibble first) -> BF16x2 (low half = low nibble)
__device__ __forceinline__ void fill_lut(uint32_t* lut) {
  const float F[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    __nv_bfloat162 v = __floats2bfloat162_rn(F[b & 15], F[b >> 4]);
    lut[b] = *reinterpret_cast<uint32_t*>(&v);
  }
}
// act_quant of one value given its block's absolute max (kernel.py: UE8M0 scale, E4M3 RN, dequantized)
__device__ __forceinline__ float aq(float v, float amax) {
  amax = fmaxf(amax, 1e-4f);
  const unsigned bits = __float_as_uint(amax * (1.0f / 448.0f));
  const int e = (int)((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0);
  const float s = __uint_as_float((unsigned)(e + 127) << 23);
  const __nv_fp8_e4m3 f8(fminf(fmaxf(v / s, -448.f), 448.f));
  return float(f8) * s;
}

struct Item { int e, off, n, pad; };

// grouping of the P pairs by expert; items of at most GR rows. One CTA.
__global__ __launch_bounds__(1024) void group_kernel(const int64_t* __restrict__ idx, int P, int GR, int E, int* __restrict__ G,
                                                    Item* __restrict__ items, int* __restrict__ pairmap, int* __restrict__ prow) {
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

// gate/up + SwiGLU + act_quant. 4 warps, MT m16 tiles. KS = 1: grid (I / 128, Gmax), warp: 32 columns j0.. of w1 and
// w3 over all of K. KS = 4 (decode): grid (I / 32, Gmax), the 4 warps take a quarter of K each for the same 32 columns
// and warp 0 sums the partials in warp order (deterministic).
template <int MT, int KS>
__global__ __launch_bounds__(128) void up_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w1,
    const uint8_t* __restrict__ s1, const uint8_t* __restrict__ w3, const uint8_t* __restrict__ s3, int I,
    const float* __restrict__ rw, const int* __restrict__ G, const Item* __restrict__ items, const int* __restrict__ prow,
    __nv_bfloat16* __restrict__ h, int topk) {
  __shared__ uint32_t lut[256];
  if ((int)blockIdx.y >= *G) return;
  fill_lut(lut);
  __syncthreads();
  const Item it = items[blockIdx.y];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int j0 = KS == 1 ? blockIdx.x * 128 + warp * 32 : blockIdx.x * 32;
  constexpr int STEPS = H / 128 / KS;    // steps per warp
  const int s0 = KS == 1 ? 0 : warp * STEPS;
  // weight rows of this lane: tile t (0..3 gate, 4..7 up) row j0 + 8 (t & 3) + g
  const uint8_t* wr[8]; const uint8_t* sr[8];
  for (int t = 0; t < 8; ++t) {
    const long long tile = ((long long)it.e * I + j0 + 8 * (t & 3)) / 8;
    wr[t] = (t < 4 ? w1 : w3) + (tile * (H / 128) + s0) * 512 + g * 64 + q * 16;
    sr[t] = (t < 4 ? s1 : s3) + (tile * (H / 128) + s0) * 32 + g * 4 + q;
  }
  // input rows (source row of the pair; rows past the item read row 0 and are not stored)
  const __nv_bfloat16* xr[MT][2];
  for (int m = 0; m < MT; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m * 16 + hh * 8 + g;
      xr[m][hh] = x + (long long)(r < it.n ? prow[it.off + r] / topk : 0) * H + q * 32 + s0 * 128;
    }
  float acc[MT][8][4];
  for (int m = 0; m < MT; ++m) for (int t = 0; t < 8; ++t) for (int c = 0; c < 4; ++c) acc[m][t][c] = 0.f;
  uint4 bw[8]; uint8_t bs[8];
  for (int t = 0; t < 8; ++t) { bw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t])); bs[t] = __ldg(sr[t]); }
  for (int s = 0; s < STEPS; ++s) {
    uint4 nw[8]; uint8_t ns[8];
    if (s + 1 < STEPS)
      for (int t = 0; t < 8; ++t) { nw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t] + (s + 1) * 512)); ns[t] = __ldg(sr[t] + (s + 1) * 32); }
    uint32_t sc[8];
    for (int t = 0; t < 8; ++t) sc[t] = scale2(bs[t]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint2 a[MT][2];
      for (int m = 0; m < MT; ++m)
        for (int hh = 0; hh < 2; ++hh) a[m][hh] = __ldg(reinterpret_cast<const uint2*>(xr[m][hh] + s * 128 + 4 * i));
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&bw[t]);
        const uint32_t b0 = hmul2(lut[b[2 * i]], sc[t]), b1 = hmul2(lut[b[2 * i + 1]], sc[t]);
        for (int m = 0; m < MT; ++m) mma(acc[m][t], a[m][0].x, a[m][1].x, a[m][0].y, a[m][1].y, b0, b1);
      }
    }
    if (s + 1 < STEPS) for (int t = 0; t < 8; ++t) { bw[t] = nw[t]; bs[t] = ns[t]; }
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
  // SwiGLU (official: linear outputs rounded to BF16, clamp, silu(gate) * up * route weight, rounded to BF16), act_quant
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
      if (r < it.n) {
        __nv_bfloat16* hr = h + (long long)(it.off + r) * I + j0 + 2 * q;
        for (int t = 0; t < 4; ++t)
          *reinterpret_cast<__nv_bfloat162*>(hr + 8 * t) = __floats2bfloat162_rn(aq(v[2 * t], amax), aq(v[2 * t + 1], amax));
      }
    }
}

// down: out [P, H] FP32 = h . w2_e^T. grid (H / 128, Gmax), 4 warps; warp: 32 output columns (4 n8 tiles).
template <int MT>
__global__ __launch_bounds__(128) void down_kernel(const __nv_bfloat16* __restrict__ h, const uint8_t* __restrict__ w2,
    const uint8_t* __restrict__ s2, int I, const int* __restrict__ G, const Item* __restrict__ items, float* __restrict__ out) {
  __shared__ uint32_t lut[256];
  if ((int)blockIdx.y >= *G) return;
  fill_lut(lut);
  __syncthreads();
  const Item it = items[blockIdx.y];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const int n0 = blockIdx.x * 128 + warp * 32;
  const int steps = I / 128;
  const uint8_t* wr[4]; const uint8_t* sr[4];
  for (int t = 0; t < 4; ++t) {
    const long long tile = ((long long)it.e * H + n0 + 8 * t) / 8;
    wr[t] = w2 + tile * steps * 512 + g * 64 + q * 16;
    sr[t] = s2 + tile * steps * 32 + g * 4 + q;
  }
  const __nv_bfloat16* hr[MT][2];
  for (int m = 0; m < MT; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m * 16 + hh * 8 + g;
      hr[m][hh] = h + (long long)(it.off + (r < it.n ? r : 0)) * I + q * 32;
    }
  float acc[MT][4][4];
  for (int m = 0; m < MT; ++m) for (int t = 0; t < 4; ++t) for (int c = 0; c < 4; ++c) acc[m][t][c] = 0.f;
  for (int s = 0; s < steps; ++s) {
    uint4 bw[4]; uint32_t sc[4];
    for (int t = 0; t < 4; ++t) { bw[t] = __ldg(reinterpret_cast<const uint4*>(wr[t] + s * 512)); sc[t] = scale2(__ldg(sr[t] + s * 32)); }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint2 a[MT][2];
      for (int m = 0; m < MT; ++m)
        for (int hh = 0; hh < 2; ++hh) a[m][hh] = *reinterpret_cast<const uint2*>(hr[m][hh] + s * 128 + 4 * i);
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&bw[t]);
        const uint32_t b0 = hmul2(lut[b[2 * i]], sc[t]), b1 = hmul2(lut[b[2 * i + 1]], sc[t]);
        for (int m = 0; m < MT; ++m) mma(acc[m][t], a[m][0].x, a[m][1].x, a[m][0].y, a[m][1].y, b0, b1);
      }
    }
  }
  for (int m = 0; m < MT; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      const int r = m * 16 + hh * 8 + g;
      if (r >= it.n) continue;
      float* o = out + (long long)(it.off + r) * H + n0 + 2 * q;
      for (int t = 0; t < 4; ++t) *reinterpret_cast<float2*>(o + 8 * t) = make_float2(acc[m][t][hh * 2], acc[m][t][hh * 2 + 1]);
    }
}

// y[r] = sum_k out[pairmap[r * 6 + k]] (k ascending) (+ add[r])
__global__ void combine_kernel(const float* __restrict__ out, const int* __restrict__ pairmap, const float* __restrict__ add,
                               float* __restrict__ y, int R, int topk) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // float4 index
  if (i >= (long long)R * (H / 4)) return;
  const int r = (int)(i / (H / 4)), c = (int)(i % (H / 4)) * 4;
  float4 acc = add ? *reinterpret_cast<const float4*>(add + (long long)r * H + c) : make_float4(0.f, 0.f, 0.f, 0.f);
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = 0; k < topk; ++k) {
    const float4 v = *reinterpret_cast<const float4*>(out + (long long)pairmap[r * topk + k] * H + c);
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  acc.x += s.x; acc.y += s.y; acc.z += s.z; acc.w += s.w;
  *reinterpret_cast<float4*>(y + (long long)r * H + c) = acc;
}

inline int gr_of(int R) { return R <= 8 ? 16 : 32; }
inline int gmax_of(int R, int topk, int E) { const int P = R * topk; return (P < E ? P : E) + P / gr_of(R) + 1; }
inline size_t al(size_t b) { return (b + 255) & ~(size_t)255; }
} // namespace

extern "C" {
// workspace for R rows and local intermediate I
size_t ds41_moe_ws_bytes(int R, int I, int topk, int E) {
  const size_t P = (size_t)R * topk;
  return al(16) + al(sizeof(Item) * gmax_of(R, topk, E)) + 2 * al(4 * P) + al(2 * P * I) + al(4 * P * H);
}
// xq [R, H] BF16 (act-quantized), idx [R, topk] int64, rw [R, topk] FP32 (route weights incl. route_scale); stacked weights
// w1/w3 [E, I, H/2] + scales [E, I, H/32], w2 [E, H, I/2] + [E, H, I/32] (Uint8) in the packed tile layout above;
// add [R, H] FP32 or null; y [R, H].
int ds41_moe(const void* xq, int R, const int64_t* idx, const float* rw, const void* w1, const void* s1, const void* w3,
             const void* s3, const void* w2, const void* s2, int I, const float* add, float* y, void* ws, int topk, int E, cudaStream_t st) {
  if (R < 1 || I % 128 != 0 || E > EMAX || topk < 1) return 1;
  const int P = R * topk, GR = gr_of(R), Gm = gmax_of(R, topk, E);
  char* p = (char*)ws;
  int* G = (int*)p; p += al(16);
  Item* items = (Item*)p; p += al(sizeof(Item) * Gm);
  int* pairmap = (int*)p; p += al(4 * (size_t)P);
  int* prow = (int*)p; p += al(4 * (size_t)P);
  __nv_bfloat16* h = (__nv_bfloat16*)p; p += al(2 * (size_t)P * I);
  float* out = (float*)p;
  group_kernel<<<1, 1024, 0, st>>>(idx, P, GR, E, G, items, pairmap, prow);
  const __nv_bfloat16* x = (const __nv_bfloat16*)xq;
  const uint8_t *W1 = (const uint8_t*)w1, *S1 = (const uint8_t*)s1, *W3 = (const uint8_t*)w3, *S3 = (const uint8_t*)s3;
  const uint8_t *W2 = (const uint8_t*)w2, *S2 = (const uint8_t*)s2;
  if (GR == 16) {
    up_kernel<1, 4><<<dim3(I / 32, Gm), 128, 0, st>>>(x, W1, S1, W3, S3, I, rw, G, items, prow, h, topk);
    down_kernel<1><<<dim3(H / 128, Gm), 128, 0, st>>>(h, W2, S2, I, G, items, out);
  } else {
    up_kernel<2, 1><<<dim3(I / 128, Gm), 128, 0, st>>>(x, W1, S1, W3, S3, I, rw, G, items, prow, h, topk);
    down_kernel<2><<<dim3(H / 128, Gm), 128, 0, st>>>(h, W2, S2, I, G, items, out);
  }
  const long long n4 = (long long)R * (H / 4);
  combine_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, st>>>(out, pairmap, add, y, R, topk);
  return (int)cudaGetLastError();
}
}
