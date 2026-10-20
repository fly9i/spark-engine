// SPDX-License-Identifier: MIT
// DeepSeek-V4.1-Flash sparse attention (docs/07 D4), semantics of ops.rs sparse_attn / kernel.py sparse_attn:
//   q [T, H, 512] BF16, kv pool [N, 512] BF16 (key = value), idx [T, K] Int64 (-1 = none), sink [H] FP32:
//   s = (q . kv[idx]) * scale (FP32), mx = max over valid (clamped to -1e30), p = exp(s - mx),
//   o = sum bf16(p) kv / (sum p + exp(sink - mx))  -> BF16.
// One CTA (8 warps) per query token, all its H <= 32 local heads at once (the KV rows are shared by the heads).
// Two passes over the gathered rows in tiles of 64 (shared memory): pass 1 the row maxima, pass 2 recomputes the same
// scores (same MMA sequence, bitwise equal), rounds p to BF16 exactly as the reference and accumulates PV and the
// denominator. No online rescaling, so p is rounded against the final maximum like the reference.
// Few tokens (decode, verify): the tiles are split over S CTAs per token (MODE 1: partial maxima; MODE 2: the global
// maximum from the partials, partial PV and denominators; finalize: sums the S partials in order and divides).
#include "ds41_pdl.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cub/block/block_scan.cuh>
#include "topk_select.cuh"

namespace {
constexpr int D = 512, DP = D + 8, TILE = 64, TP = TILE + 8, WARPS = 8;

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }
// 16-byte asynchronous global -> shared copy (zero-filled when !valid): a thread's copies are all in flight at once
// (a plain load -> store loop waits for each load in turn: 16 serial round trips per 64-row tile)
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

// The attention pool as up to three row ranges read in place: [0, na) a, [na, na + nb) b, the rest c. W > 0: the
// first W indices of every query are its sliding window (keys pos - W + 1 .. pos; a = the ring of the W latest committed
// keys by slot pos % W, b = the chunk's keys from position `start`), computed here; idx then holds only the remaining
// K - W indices per query.
// lo: window keys before position lo are not attended (SWA bounded replay truncates the window to the replayed
// segment; 0 = no truncation).
struct Pools { const __nv_bfloat16 *a, *b, *c; int na, nb; int W; long long start; const int64_t* pos; long long lo; };

// CACHE (MODE 0, at most MAXT tiles): pass 1 keeps the warp's scores of every tile in registers, pass 2 uses them
// instead of recomputing q . k (the same values: computed once), so pass 2 only loads the tiles for P V.
constexpr int MAXT = 10;
// MODE 3 (few tokens): the S CTAs of a token in one launch, synchronized twice through a per-token counter and
// generation (sense-reversing barrier; the launcher only uses it when every CTA can be resident at once): maxima, then
// partial PV and denominators, then each CTA finalizes heads s, s + S, ... exactly like attn_fin_kernel.
constexpr int MAXTOK = 16, MAXSPLIT = 16;
// Several sequences in one launch (MODE 3 only; decode / verify of a batch): every token its own pools, row of
// indices, key count and split count (the ones its sequence would get alone, so the result is the same bits);
// workspace slots per token MAXSPLIT apart.
struct Multi { Pools pl[MAXTOK]; const int64_t* idx[MAXTOK]; int K[MAXTOK]; int S[MAXTOK]; };
__device__ unsigned int g_bar_cnt[MAXTOK];
__device__ unsigned int g_bar_gen[MAXTOK];
__device__ __forceinline__ void token_barrier(int t, int S) {
  __threadfence();     // this CTA's partial results visible to the others
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned int* gen = &g_bar_gen[t];
    const unsigned int g = *gen;
    __threadfence();
    if (atomicAdd(&g_bar_cnt[t], 1u) == (unsigned)S - 1) { g_bar_cnt[t] = 0; __threadfence(); atomicAdd(&g_bar_gen[t], 1u); }
    else while (*gen == g) __nanosleep(64);
    __threadfence();
  }
  __syncthreads();
}
template <int MT, int MODE, bool CACHE = false, bool MULTI = false>
__global__ __launch_bounds__(WARPS * 32, 1) void sparse_attn_kernel(const __nv_bfloat16* __restrict__ q, Pools pl0, const int64_t* __restrict__ idx, const float* __restrict__ sink, __nv_bfloat16* __restrict__ out, int H, int K0, float scale,
    float* __restrict__ pmx, float* __restrict__ pden, float* __restrict__ po, const __grid_constant__ Multi mt) {
  pdl_entry();
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* kvs = reinterpret_cast<__nv_bfloat16*>(smem);          // [TILE][DP]
  __nv_bfloat16* qs = kvs + TILE * DP;                                    // [H][DP]
  __nv_bfloat16* ps = qs + H * DP;                                        // [32][TP]
  float* red = reinterpret_cast<float*>(ps + 32 * TP);                   // [WARPS][32]
  float* mxs = red + WARPS * 32;                                          // [32]
  __shared__ int rows[TILE];
  const int t = MODE == 3 ? blockIdx.y : blockIdx.x, sp_id = MODE == 3 ? blockIdx.x : blockIdx.y;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, qd = lane & 3;
  const Pools pl = MULTI ? mt.pl[t] : pl0;
  const int K = MULTI ? mt.K[t] : K0;
  const int S = MULTI ? mt.S[t] : MODE == 3 ? gridDim.x : gridDim.y;
  if (MULTI && sp_id >= S) return;                                       // (the grid has the batch's largest split)
  const int SS = MULTI ? MAXSPLIT : S;                                   // workspace slots per token
  const int K2 = K - pl.W;                                               // explicit indices per query
  const int64_t* ix = MULTI ? mt.idx[t] : idx + (long long)t * K2;
  const long long pt = pl.W ? pl.pos[t] : 0;
  for (int i = threadIdx.x; i < H * D / 8; i += blockDim.x) {
    const int h = i / (D / 8), c = (i % (D / 8)) * 8;
    cp16(qs + h * DP + c, q + ((long long)t * H + h) * D + c, true);
  }
  const int ntiles_all = (K + TILE - 1) / TILE, per = (ntiles_all + S - 1) / S;
  const int tl0 = MODE ? sp_id * per : 0, ntiles = MODE ? min(ntiles_all, tl0 + per) : ntiles_all;
  auto load_tile = [&](int tl) {
    if (threadIdx.x < TILE) {
      const int k = tl * TILE + threadIdx.x;
      int src = -1;
      if (k < pl.W) {   // window key k of the query: position pt - (W - 1) + k (ring slot, or the chunk row past the ring)
        const long long kp = pt - (pl.W - 1) + k;
        src = kp < 0 || kp < pl.lo ? -1 : kp >= pl.start ? (int)(kp - pl.start) + pl.W : (int)(kp % pl.W);
      } else if (k < K) src = (int)ix[k - pl.W];
      rows[threadIdx.x] = src;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TILE * D / 8; i += blockDim.x) {
      const int r = i / (D / 8), c = (i % (D / 8)) * 8;
      const int src = rows[r];
      const __nv_bfloat16* row = src < 0 ? pl.a : src < pl.na ? pl.a + (long long)src * D : src < pl.na + pl.nb ? pl.b + (long long)(src - pl.na) * D
                                                                                                         : pl.c + (long long)(src - pl.na - pl.nb) * D;
      cp16(kvs + r * DP + c, row + c, src >= 0);
    }
    cp_wait_all();     // (also the query rows on the first call)
    __syncthreads();
  };
  // scores of this warp's 8 tile rows (n8 tile `warp`) for all heads: s[m][0..3] (rows g / g+8 of m tile, cols 2qd, 2qd+1)
  auto scores = [&](float (&s)[MT][4]) {
    for (int m = 0; m < MT; ++m) for (int c = 0; c < 4; ++c) s[m][c] = 0.f;
    const __nv_bfloat16* kr = kvs + (warp * 8 + g) * DP + 2 * qd;
#pragma unroll 4
    for (int k0 = 0; k0 < D; k0 += 16) {
      const uint32_t b0 = ld32(kr + k0), b1 = ld32(kr + k0 + 8);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int h0 = m * 16 + g, h1 = h0 + 8;
        const uint32_t a0 = h0 < H ? ld32(qs + h0 * DP + k0 + 2 * qd) : 0u, a2 = h0 < H ? ld32(qs + h0 * DP + k0 + 8 + 2 * qd) : 0u;
        const uint32_t a1 = h1 < H ? ld32(qs + h1 * DP + k0 + 2 * qd) : 0u, a3 = h1 < H ? ld32(qs + h1 * DP + k0 + 8 + 2 * qd) : 0u;
        mma(s[m], a0, a1, a2, a3, b0, b1);
      }
    }
    for (int m = 0; m < MT; ++m) for (int c = 0; c < 4; ++c) s[m][c] *= scale;
  };
  // ---- pass 1: maxima
  float mx[MT][2];
  for (int m = 0; m < MT; ++m) mx[m][0] = mx[m][1] = -INFINITY;
  float sc[CACHE ? MAXT : 1][MT][4];
  if (CACHE) {
#pragma unroll
    for (int i = 0; i < MAXT; ++i) {
      const int tl = tl0 + i;
      if (tl >= ntiles) break;
      load_tile(tl);
      scores(sc[i]);
      const bool v0 = rows[warp * 8 + 2 * qd] >= 0, v1 = rows[warp * 8 + 2 * qd + 1] >= 0;
      for (int m = 0; m < MT; ++m) {
        if (v0) { mx[m][0] = fmaxf(mx[m][0], sc[i][m][0]); mx[m][1] = fmaxf(mx[m][1], sc[i][m][2]); }
        if (v1) { mx[m][0] = fmaxf(mx[m][0], sc[i][m][1]); mx[m][1] = fmaxf(mx[m][1], sc[i][m][3]); }
      }
      __syncthreads();
    }
  }
  if (!CACHE && MODE != 2) for (int tl = tl0; tl < ntiles; ++tl) {
    load_tile(tl);
    float s[MT][4];
    scores(s);
    const bool v0 = rows[warp * 8 + 2 * qd] >= 0, v1 = rows[warp * 8 + 2 * qd + 1] >= 0;
    for (int m = 0; m < MT; ++m) {
      if (v0) { mx[m][0] = fmaxf(mx[m][0], s[m][0]); mx[m][1] = fmaxf(mx[m][1], s[m][2]); }
      if (v1) { mx[m][0] = fmaxf(mx[m][0], s[m][1]); mx[m][1] = fmaxf(mx[m][1], s[m][3]); }
    }
    __syncthreads();
  }
  for (int m = 0; m < MT; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      float v = mx[m][hh];
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      if (qd == 0) red[warp * 32 + m * 16 + hh * 8 + g] = v;
    }
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = -INFINITY;
    if (MODE == 2) for (int sp = 0; sp < S; ++sp) v = fmaxf(v, pmx[((long long)t * SS + sp) * 32 + threadIdx.x]);
    else for (int w = 0; w < WARPS; ++w) v = fmaxf(v, red[w * 32 + threadIdx.x]);
    if (MODE == 1 || MODE == 3) pmx[((long long)t * SS + sp_id) * 32 + threadIdx.x] = v;
    mxs[threadIdx.x] = fmaxf(v, -1e30f);
  }
  if (MODE == 1) return;
  if (MODE == 3) {
    token_barrier(t, S);
    if (threadIdx.x < 32) {
      float v = -INFINITY;
      for (int sp = 0; sp < S; ++sp) v = fmaxf(v, __ldcg(&pmx[((long long)t * SS + sp) * 32 + threadIdx.x]));
      mxs[threadIdx.x] = fmaxf(v, -1e30f);
    }
  }
  __syncthreads();
  float mrow[MT][2];
  for (int m = 0; m < MT; ++m) { mrow[m][0] = mxs[m * 16 + g]; mrow[m][1] = mxs[m * 16 + 8 + g]; }
  // ---- pass 2: p, denominator, PV. Warp w accumulates output dims [64 w, 64 w + 64) (8 n8 tiles) of all heads.
  float o[MT][8][4];
  for (int m = 0; m < MT; ++m) for (int n = 0; n < 8; ++n) for (int c = 0; c < 4; ++c) o[m][n][c] = 0.f;
  float den[MT][2];
  for (int m = 0; m < MT; ++m) den[m][0] = den[m][1] = 0.f;
#pragma unroll
  for (int i = 0; i < (CACHE ? MAXT : ntiles - tl0); ++i) {
    const int tl = tl0 + i;
    if (CACHE && tl >= ntiles) break;
    if (!(CACHE && ntiles - tl0 == 1)) load_tile(tl);   // one cached tile: still in shared memory from pass 1
    float s[MT][4];
    if (CACHE) { for (int m = 0; m < MT; ++m) for (int c = 0; c < 4; ++c) s[m][c] = sc[CACHE ? i : 0][m][c]; }
    else scores(s);
    const bool v0 = rows[warp * 8 + 2 * qd] >= 0, v1 = rows[warp * 8 + 2 * qd + 1] >= 0;
    for (int m = 0; m < MT; ++m)
      for (int hh = 0; hh < 2; ++hh) {
        const float p0 = v0 ? expf(s[m][hh * 2] - mrow[m][hh]) : 0.f, p1 = v1 ? expf(s[m][hh * 2 + 1] - mrow[m][hh]) : 0.f;
        den[m][hh] += p0 + p1;
        *reinterpret_cast<__nv_bfloat162*>(ps + (m * 16 + hh * 8 + g) * TP + warp * 8 + 2 * qd) = __floats2bfloat162_rn(p0, p1);
      }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TILE; kk += 16) {
      uint32_t a[MT][4];
      for (int m = 0; m < MT; ++m) {
        const __nv_bfloat16* pr = ps + (m * 16 + g) * TP + kk + 2 * qd;
        a[m][0] = ld32(pr); a[m][1] = ld32(pr + 8 * TP); a[m][2] = ld32(pr + 8); a[m][3] = ld32(pr + 8 * TP + 8);
      }
#pragma unroll
      for (int n = 0; n < 8; n += 2) {
        // x4.trans: lanes 0-15 rows kk..kk+15 at column n0, lanes 16-31 the same rows at n0 + 8
        const int n0 = warp * 64 + n * 8;
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, kvs + (kk + (lane & 15)) * DP + n0 + (lane >> 4) * 8);
        for (int m = 0; m < MT; ++m) { mma(o[m][n], a[m][0], a[m][1], a[m][2], a[m][3], b0, b1); mma(o[m][n + 1], a[m][0], a[m][1], a[m][2], a[m][3], b2, b3); }
      }
    }
    __syncthreads();
  }
  // denominators: lanes of a quad, then warps (in order)
  for (int m = 0; m < MT; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      float v = den[m][hh];
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      if (qd == 0) red[warp * 32 + m * 16 + hh * 8 + g] = v;
    }
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = 0.f;
    for (int w = 0; w < WARPS; ++w) v += red[w * 32 + threadIdx.x];
    const int h = threadIdx.x;
    if (MODE == 2 || MODE == 3) pden[((long long)t * SS + sp_id) * 32 + h] = v;
    else mxs[h] = h < H ? v + expf(sink[h] - mxs[h]) : 1.f;
  }
  __syncthreads();
  if (MODE == 2 || MODE == 3) {   // partial PV [t][split][h][512]
    for (int m = 0; m < MT; ++m)
      for (int hh = 0; hh < 2; ++hh) {
        const int h = m * 16 + hh * 8 + g;
        if (h >= H) continue;
        float* orow = po + (((long long)t * SS + sp_id) * H + h) * D + warp * 64 + 2 * qd;
        for (int n = 0; n < 8; ++n) *reinterpret_cast<float2*>(orow + n * 8) = make_float2(o[m][n][hh * 2], o[m][n][hh * 2 + 1]);
      }
    if (MODE == 2) return;
    // MODE 3: every split's parts written; this CTA finalizes heads sp_id, sp_id + S, ... (attn_fin_kernel's sums)
    token_barrier(t, S);
    // all threads over (head, column pair); every split's value loaded before the sums (S <= MAXSPLIT, in order)
    const int nh = (H - sp_id + S - 1) / S;
    if (threadIdx.x < nh) {
      const int h = sp_id + threadIdx.x * S;
      float pd[MAXSPLIT];
#pragma unroll
      for (int sp = 0; sp < MAXSPLIT; ++sp) pd[sp] = sp < S ? __ldcg(&pden[((long long)t * SS + sp) * 32 + h]) : 0.f;
      float den = 0.f;
#pragma unroll
      for (int sp = 0; sp < MAXSPLIT; ++sp) if (sp < S) den += pd[sp];
      red[threadIdx.x] = den + expf(sink[h] - mxs[h]);
    }
    __syncthreads();
#pragma unroll 2
    for (int i = threadIdx.x; i < nh * (D / 2); i += blockDim.x) {
      const int j = i / (D / 2), d = (i % (D / 2)) * 2, h = sp_id + j * S;
      float2 v[MAXSPLIT];
#pragma unroll
      for (int sp = 0; sp < MAXSPLIT; ++sp) if (sp < S) v[sp] = __ldcg(reinterpret_cast<const float2*>(po + (((long long)t * SS + sp) * H + h) * D + d));
      float2 a = make_float2(0.f, 0.f);
#pragma unroll
      for (int sp = 0; sp < MAXSPLIT; ++sp) if (sp < S) { a.x += v[sp].x; a.y += v[sp].y; }
      const float den = red[j];
      *reinterpret_cast<__nv_bfloat162*>(out + ((long long)t * H + h) * D + d) = __floats2bfloat162_rn(a.x / den, a.y / den);
    }
    return;
  }
  for (int m = 0; m < MT; ++m)
    for (int hh = 0; hh < 2; ++hh) {
      const int h = m * 16 + hh * 8 + g;
      if (h >= H) continue;
      const float dn = mxs[h];
      __nv_bfloat16* orow = out + ((long long)t * H + h) * D + warp * 64 + 2 * qd;
      for (int n = 0; n < 8; ++n)
        *reinterpret_cast<__nv_bfloat162*>(orow + n * 8) = __floats2bfloat162_rn(o[m][n][hh * 2] / dn, o[m][n][hh * 2 + 1] / dn);
    }
}

// out[t][h] = sum_s po[t][s][h] / (sum_s pden[t][s][h] + exp(sink[h] - mx)), mx = max_s pmx (clamped)
// (MULTI: the token's own split count, slots MAXSPLIT apart)
template <bool MULTI = false>
__global__ void attn_fin_kernel(const float* __restrict__ pmx, const float* __restrict__ pden, const float* __restrict__ po,
                                const float* __restrict__ sink, __nv_bfloat16* __restrict__ out, int H, int S0, const __grid_constant__ Multi mt) {
  pdl_entry();
  const int t = blockIdx.x, h = blockIdx.y;
  const int S = MULTI ? mt.S[t] : S0, SS = MULTI ? MAXSPLIT : S0;
  float mx = -INFINITY, den = 0.f;
  for (int sp = 0; sp < S; ++sp) { mx = fmaxf(mx, pmx[((long long)t * SS + sp) * 32 + h]); den += pden[((long long)t * SS + sp) * 32 + h]; }
  mx = fmaxf(mx, -1e30f);
  den += expf(sink[h] - mx);
  for (int d = threadIdx.x * 2; d < D; d += blockDim.x * 2) {
    float2 a = make_float2(0.f, 0.f);
    for (int sp = 0; sp < S; ++sp) { const float2 v = *reinterpret_cast<const float2*>(po + (((long long)t * SS + sp) * H + h) * D + d); a.x += v.x; a.y += v.y; }
    *reinterpret_cast<__nv_bfloat162*>(out + ((long long)t * H + h) * D + d) = __floats2bfloat162_rn(a.x / den, a.y / den);
  }
}

// Indexer scores (model.py Indexer, the BF16 einsum): out[t][c] = bf16(sum_h bf16(relu(bf16(q[t][h] . k[c])) * w[t][h]))
// (FP32 head sum), -inf for c >= (pos[t] + 1) / ratio (not yet compressed for that query). 32 heads x 128 dims.
// grid (ceil(Tc / 64), T), 4 warps of 16 compressed rows; q of the token in shared memory.
__global__ __launch_bounds__(128) void index_score_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ w, const int64_t* __restrict__ pos, int ratio, __nv_bfloat16* __restrict__ out, int Tc) {
  constexpr int NH = 32, ID = 128, QP = ID + 8;
  __shared__ __align__(16) __nv_bfloat16 qs[NH * QP];
  const int t = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, qd = lane & 3;
  for (int i = threadIdx.x; i < NH * ID / 8; i += blockDim.x) {
    const int h = i / (ID / 8), c = (i % (ID / 8)) * 8;
    *reinterpret_cast<uint4*>(qs + h * QP + c) = *reinterpret_cast<const uint4*>(q + ((long long)t * NH + h) * ID + c);
  }
  __syncthreads();
  const int c0 = blockIdx.x * 64 + warp * 16;
  const int r0 = min(c0 + g, Tc - 1), r1 = min(c0 + g + 8, Tc - 1);
  const __nv_bfloat16* k0 = k + (long long)r0 * ID + 2 * qd;
  const __nv_bfloat16* k1 = k + (long long)r1 * ID + 2 * qd;
  float acc[4][4];
  for (int n = 0; n < 4; ++n) for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
  for (int kk = 0; kk < ID; kk += 16) {
    const uint32_t a0 = ld32(k0 + kk), a1 = ld32(k1 + kk), a2 = ld32(k0 + kk + 8), a3 = ld32(k1 + kk + 8);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const __nv_bfloat16* qr = qs + (n * 8 + g) * QP + kk + 2 * qd;
      mma(acc[n], a0, a1, a2, a3, ld32(qr), ld32(qr + 8));
    }
  }
  float sum[2] = {0.f, 0.f};
  for (int n = 0; n < 4; ++n)
    for (int c = 0; c < 4; ++c) {
      const int h = n * 8 + 2 * qd + (c & 1);
      const float v = fmaxf(__bfloat162float(__float2bfloat16(acc[n][c])), 0.f);
      sum[c >> 1] += __bfloat162float(__float2bfloat16(v * __bfloat162float(w[(long long)t * NH + h])));
    }
  for (int hh = 0; hh < 2; ++hh) {
    sum[hh] += __shfl_xor_sync(0xffffffffu, sum[hh], 1);
    sum[hh] += __shfl_xor_sync(0xffffffffu, sum[hh], 2);
  }
  if (qd == 0) {
    const long long lens = (pos[t] + 1) / ratio;
    for (int hh = 0; hh < 2; ++hh) {
      const int c = c0 + g + hh * 8;
      if (c < Tc) out[(long long)t * Tc + c] = c >= lens ? __float2bfloat16(-INFINITY) : __float2bfloat16(sum[hh]);
    }
  }
}

// Indexer scores for many query rows (prefill chunks): the same values as index_score_kernel (each element: the same
// MMA sequence over the 128 dims, the same BF16 roundings, the same 32-head summation order), but the compressed keys
// are shared: a CTA of 8 warps takes 8 query rows (warp = row, its 32 heads' q held as MMA B fragments in registers)
// and streams KS keys through shared memory in tiles of 64 (cp.async double buffer); co-resident CTAs (grid x = row
// groups) work on the same key range, so the keys come from DRAM about once instead of once per query row (prefill at
// 640k context: ~335 GB of key reads per index layer and chunk with one CTA column per row).
constexpr int IS_TILE = 64, IS_KS = 2048, IS_KP = 128 + 8;
__global__ __launch_bounds__(256) void index_score_tiled_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ w, const int64_t* __restrict__ pos, int ratio, __nv_bfloat16* __restrict__ out, int T, int Tc) {
  constexpr int NH = 32, ID = 128;
  __shared__ __align__(16) __nv_bfloat16 ks[2][IS_TILE * IS_KP];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, qd = lane & 3;
  const int t = blockIdx.x * 8 + warp;
  const bool live = t < T;
  const int tq = live ? t : T - 1;
  // B fragments of this row's q: [kk step][n tile][2]
  uint32_t bq[ID / 16][4][2];
#pragma unroll
  for (int kk = 0; kk < ID / 16; ++kk)
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const __nv_bfloat16* qr = q + ((long long)tq * NH + n * 8 + g) * ID + kk * 16 + 2 * qd;
      bq[kk][n][0] = ld32(qr); bq[kk][n][1] = ld32(qr + 8);
    }
  float wv[4][2];
#pragma unroll
  for (int n = 0; n < 4; ++n) for (int c = 0; c < 2; ++c) wv[n][c] = __bfloat162float(w[(long long)tq * NH + n * 8 + 2 * qd + c]);
  const long long lens = (pos[tq] + 1) / ratio;
  const int kb = blockIdx.y * IS_KS, ke = min(Tc, kb + IS_KS);
  const int ntiles = (ke - kb + IS_TILE - 1) / IS_TILE;
  auto load = [&](int tl, int buf) {
    // 64 keys x 256 bytes: 1024 16-byte pieces, 4 per thread; rows past Tc read row Tc - 1 (as the per-row kernel)
    for (int i = threadIdx.x; i < IS_TILE * ID / 8; i += blockDim.x) {
      const int r = i / (ID / 8), c = (i % (ID / 8)) * 8;
      const int kr = min(kb + tl * IS_TILE + r, Tc - 1);
      cp16(ks[buf] + r * IS_KP + c, k + (long long)kr * ID + c, true);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  if (ntiles > 0) load(0, 0);
  for (int tl = 0; tl < ntiles; ++tl) {
    const int buf = tl & 1;
    if (tl + 1 < ntiles) { load(tl + 1, buf ^ 1); asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#pragma unroll 1
    for (int m = 0; m < IS_TILE / 16; ++m) {
      const __nv_bfloat16* k0 = ks[buf] + (m * 16 + g) * IS_KP + 2 * qd;
      const __nv_bfloat16* k1 = k0 + 8 * IS_KP;
      float acc[4][4];
      for (int n = 0; n < 4; ++n) for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
      for (int kk = 0; kk < ID / 16; ++kk) {
        const uint32_t a0 = ld32(k0 + kk * 16), a1 = ld32(k1 + kk * 16), a2 = ld32(k0 + kk * 16 + 8), a3 = ld32(k1 + kk * 16 + 8);
#pragma unroll
        for (int n = 0; n < 4; ++n) mma(acc[n], a0, a1, a2, a3, bq[kk][n][0], bq[kk][n][1]);
      }
      float sum[2] = {0.f, 0.f};
      for (int n = 0; n < 4; ++n)
        for (int c = 0; c < 4; ++c) {
          const float v = fmaxf(__bfloat162float(__float2bfloat16(acc[n][c])), 0.f);
          sum[c >> 1] += __bfloat162float(__float2bfloat16(v * wv[n][c & 1]));
        }
      for (int hh = 0; hh < 2; ++hh) {
        sum[hh] += __shfl_xor_sync(0xffffffffu, sum[hh], 1);
        sum[hh] += __shfl_xor_sync(0xffffffffu, sum[hh], 2);
      }
      if (live && qd == 0)
        for (int hh = 0; hh < 2; ++hh) {
          const int c = kb + tl * IS_TILE + m * 16 + g + hh * 8;
          if (c < ke) out[(long long)t * Tc + c] = c >= lens ? __float2bfloat16(-INFINITY) : __float2bfloat16(sum[hh]);
        }
    }
    __syncthreads();
  }
}

// The indexer's selection (model.py: topk of the scores, then the indices in ascending order, as pool indices):
// out[t] = the k largest scores of row t (ties to the lower index) in ascending index order, + offset, -1 where the
// index >= (pos[t] + 1) / ratio. One CTA per row: two 8-bit radix passes over order-preserving 16-bit keys
// (topk_threshold) find the k-th largest key and how many of its equals to take, then block scans in index order place the selected indices
// (replaces topk + sort + add + compare + masked_fill: 5+ launches, the sort alone ~22 us).
constexpr int TK_THREADS = 1024;
__device__ __forceinline__ unsigned tk_key(__nv_bfloat16 v) {
  const unsigned b = __bfloat16_as_ushort(v);
  return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}
__global__ __launch_bounds__(TK_THREADS) void topk_idx_kernel(const __nv_bfloat16* __restrict__ score, const int64_t* __restrict__ pos, int ratio,
                                                             int Tc, int k, long long offset, int64_t* __restrict__ out) {
  pdl_entry();
  typedef cub::BlockScan<int, TK_THREADS> Scan;
  __shared__ typename Scan::TempStorage tmp;
  __shared__ TopkSelectSmem ts;
  __shared__ int s_carry[2];
  const int t = blockIdx.x;
  const __nv_bfloat16* row = score + (long long)t * Tc;
  const long long lens = (pos[t] + 1) / ratio;
  // the k-th largest key and how many of its equals to take after all larger ones (shared with GLM's DSA top-k)
  unsigned thr; int need_i;
  topk_threshold<2, TK_THREADS>(ts, [&](int i) { return tk_key(row[i]); }, Tc, k, 0, 0u, thr, need_i);
  const unsigned need = (unsigned)need_i;
  if (threadIdx.x == 0) { s_carry[0] = 0; s_carry[1] = 0; }
  __syncthreads();
  for (int c0 = 0; c0 < Tc; c0 += blockDim.x) {
    const int i = c0 + threadIdx.x;
    const unsigned key = i < Tc ? tk_key(row[i]) : 0u;
    const int eq = i < Tc && key == thr, gt = i < Tc && key > thr;
    if (__syncthreads_count(eq | gt) == 0) continue;   // (long rows: most chunks hold no selected key)
    int eqr, eqt;
    Scan(tmp).ExclusiveSum(eq, eqr, eqt);
    __syncthreads();
    const int sel = gt || (eq && (unsigned)(s_carry[0] + eqr) < need);
    int sr, st;
    Scan(tmp).ExclusiveSum(sel, sr, st);
    if (sel) out[(long long)t * k + s_carry[1] + sr] = i >= lens ? -1 : (long long)i + offset;
    __syncthreads();
    if (threadIdx.x == 0) { s_carry[0] += eqt; s_carry[1] += st; }
    __syncthreads();
  }
}

inline size_t smem_bytes(int H) { return (size_t)(TILE * DP + H * DP + 32 * TP) * 2 + (WARPS * 32 + 32) * 4; }
} // namespace

// shared memory needed for H heads (the caller falls back to the libtorch path when the device cannot give it)
extern "C" size_t ds41_sparse_attn_smem(int H) { return smem_bytes(H); }
// workspace for the split path (T <= 16): T * 16 * (64 + H * 512) floats; ws = null disables the split
extern "C" size_t ds41_sparse_attn_ws_bytes(int T, int H) { return T <= 16 ? (size_t)T * 16 * (64 + (size_t)H * D) * 4 : 0; }
// pools a / b / c with na / nb rows (b, c may be null), W window keys per query from `pos` / `start` (0: none), idx
// [T, K - W] the other indices.
extern "C" int ds41_sparse_attn(const void* q, const void* a, int na, const void* b, int nb, const void* c, int W, long long start,
                                long long lo, const int64_t* pos, const int64_t* idx, const float* sink, void* out, int T, int H, int K,
                                float scale, void* ws, cudaStream_t st) {
  if (!b) nb = 0;
  if (!c && !b) na = 0x7fffffff;
  Pools pl{(const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (const __nv_bfloat16*)c, na, nb, W, start, pos, lo};
  if (T < 1 || H < 1 || H > 32 || K < 1) return 1;
  const size_t sm = smem_bytes(H);
  const int tiles = (K + TILE - 1) / TILE;
  const int S = (T <= 16 && ws) ? (tiles < 16 ? tiles : 16) : 1;
  float* pmx = (float*)ws; float* pden = pmx + (size_t)T * S * 32; float* po = pden + (size_t)T * S * 32;
  const __nv_bfloat16* Q = (const __nv_bfloat16*)q; __nv_bfloat16* O = (__nv_bfloat16*)out;
  // MODE 3 needs all T * S CTAs resident at once (one per SM: the shared memory); DS41_ATTN_FUSED=0 the 3-kernel path
  static int sms = 0; if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  static const bool fused_on = !(std::getenv("DS41_ATTN_FUSED") && std::getenv("DS41_ATTN_FUSED")[0] == '0');
  const bool fused = fused_on && T <= MAXTOK && T * S <= sms;
  static const Multi none{};
#define DS41_ATTN_LAUNCH(MT) do { \
    static size_t set[3] = {0, 0, 0}; \
    if (sm > set[0]) { \
      if (cudaFuncSetAttribute(sparse_attn_kernel<MT, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) || \
          cudaFuncSetAttribute(sparse_attn_kernel<MT, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) || \
          cudaFuncSetAttribute(sparse_attn_kernel<MT, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) || \
          cudaFuncSetAttribute(sparse_attn_kernel<MT, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) || \
          cudaFuncSetAttribute(sparse_attn_kernel<MT, 3, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)) { cudaGetLastError(); return 2; } \
      set[0] = sm; } \
    if (S > 1 && fused && (tiles + S - 1) / S <= MAXT) pdl_launch((sparse_attn_kernel<MT, 3, true>), dim3(dim3(S, T)), dim3(WARPS * 32), sm, st, Q, pl, idx, sink, O, H, K, scale, pmx, pden, po, none); \
    else if (S == 1 && tiles <= MAXT) pdl_launch((sparse_attn_kernel<MT, 0, true>), dim3(T), dim3(WARPS * 32), sm, st, Q, pl, idx, sink, O, H, K, scale, nullptr, nullptr, nullptr, none); \
    else if (S == 1) pdl_launch((sparse_attn_kernel<MT, 0>), dim3(T), dim3(WARPS * 32), sm, st, Q, pl, idx, sink, O, H, K, scale, nullptr, nullptr, nullptr, none); \
    else { pdl_launch((sparse_attn_kernel<MT, 1>), dim3(dim3(T, S)), dim3(WARPS * 32), sm, st, Q, pl, idx, sink, O, H, K, scale, pmx, pden, po, none); \
           pdl_launch((sparse_attn_kernel<MT, 2>), dim3(dim3(T, S)), dim3(WARPS * 32), sm, st, Q, pl, idx, sink, O, H, K, scale, pmx, pden, po, none); \
           pdl_launch(attn_fin_kernel<false>, dim3(dim3(T, H)), dim3(128), 0, st, pmx, pden, po, sink, O, H, S, none); } } while (0)
  if (H <= 16) DS41_ATTN_LAUNCH(1); else DS41_ATTN_LAUNCH(2);
#undef DS41_ATTN_LAUNCH
  return (int)cudaGetLastError();
}

// Several sequences' rows in one launch (decode / verify batches): q / out [T, H, 512] of all rows (sequence s owns rows
// [off_s, off_s + n), in order), pos [T] of all rows, W window keys; per sequence its ring a (na rows), chunk keys b
// (n rows), compressed pool c (or null), start, lo and indices idx [n, K - W]. Every token gets the split count its
// sequence would get alone (the same bits); <= 16 rows, each sequence's keys over >= 2 tiles, else 2 (the caller runs
// the sequences one by one). DS41_ATTN_MULTI=2 always takes the three-launch path (tests).
struct Ds41AttnSeq { const void* a; const void* b; const void* c; const int64_t* idx; long long start, lo; int na, n, K, pad; };
extern "C" int ds41_sparse_attn_multi(const void* q, const Ds41AttnSeq* seqs, int nseq, int W, const int64_t* pos, const float* sink, void* out,
                                      int H, float scale, void* ws, cudaStream_t st) {
  static int sms = 0; if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  static const bool fused_on = !(std::getenv("DS41_ATTN_FUSED") && std::getenv("DS41_ATTN_FUSED")[0] == '0');
  if (!fused_on || !ws || H < 1 || H > 32 || nseq < 1) return 2;
  Multi mt{};
  int T = 0, ctas = 0, smax = 1;
  for (int i = 0; i < nseq; ++i) {
    const Ds41AttnSeq& d = seqs[i];
    const int tiles = (d.K + TILE - 1) / TILE, S = tiles < 16 ? tiles : 16;
    if (d.n < 1 || d.K < 1 || T + d.n > MAXTOK || S < 2) return 2;
    Pools pl{(const __nv_bfloat16*)d.a, (const __nv_bfloat16*)d.b, (const __nv_bfloat16*)d.c, d.na, d.b ? d.n : 0, W, d.start, pos, d.lo};
    if (!d.c && !d.b) pl.na = 0x7fffffff;
    for (int r = 0; r < d.n; ++r, ++T) { mt.pl[T] = pl; mt.idx[T] = d.idx + (long long)r * (d.K - W); mt.K[T] = d.K; mt.S[T] = S; }
    ctas += d.n * S;
    smax = S > smax ? S : smax;
  }
  // all CTAs resident at once: the fused split path (as each sequence alone when its own CTAs fit, and its tiles per
  // split stay within the score cache); else partial maxima / partial PV / finalize (the same sums in the same order)
  static const bool force3 = std::getenv("DS41_ATTN_MULTI") && std::getenv("DS41_ATTN_MULTI")[0] == '2';
  bool fused = ctas <= sms && !force3;
  for (int i = 0; i < T; ++i) { const int tiles = (mt.K[i] + TILE - 1) / TILE; fused = fused && (tiles + mt.S[i] - 1) / mt.S[i] <= MAXT; }
  const size_t sm = smem_bytes(H);
  float* pmx = (float*)ws; float* pden = pmx + (size_t)T * MAXSPLIT * 32; float* po = pden + (size_t)T * MAXSPLIT * 32;
  const __nv_bfloat16* Q = (const __nv_bfloat16*)q; __nv_bfloat16* O = (__nv_bfloat16*)out;
  const Pools none{};
#define DS41_ATTN_MULTI(MT) do { \
    static size_t set = 0; \
    if (sm > set) { \
      if (cudaFuncSetAttribute(sparse_attn_kernel<MT, 3, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) || \
          cudaFuncSetAttribute(sparse_attn_kernel<MT, 1, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) || \
          cudaFuncSetAttribute(sparse_attn_kernel<MT, 2, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)) { cudaGetLastError(); return 2; } \
      set = sm; } \
    if (fused) pdl_launch((sparse_attn_kernel<MT, 3, true, true>), dim3(dim3(smax, T)), dim3(WARPS * 32), sm, st, Q, none, (const int64_t*)nullptr, sink, O, H, 0, scale, pmx, pden, po, mt); \
    else { pdl_launch((sparse_attn_kernel<MT, 1, false, true>), dim3(dim3(T, smax)), dim3(WARPS * 32), sm, st, Q, none, (const int64_t*)nullptr, sink, O, H, 0, scale, pmx, pden, po, mt); \
           pdl_launch((sparse_attn_kernel<MT, 2, false, true>), dim3(dim3(T, smax)), dim3(WARPS * 32), sm, st, Q, none, (const int64_t*)nullptr, sink, O, H, 0, scale, pmx, pden, po, mt); \
           pdl_launch(attn_fin_kernel<true>, dim3(dim3(T, H)), dim3(128), 0, st, pmx, pden, po, sink, O, H, 0, mt); } } while (0)
  if (H <= 16) DS41_ATTN_MULTI(1); else DS41_ATTN_MULTI(2);
#undef DS41_ATTN_MULTI
  return (int)cudaGetLastError();
}

extern "C" int ds41_topk_idx(const void* score, const int64_t* pos, int ratio, int T, int Tc, int k, long long offset, int64_t* out, cudaStream_t st) {
  if (T < 1 || k < 1 || k > Tc) return 1;
  pdl_launch(topk_idx_kernel, dim3(T), dim3(TK_THREADS), 0, st, (const __nv_bfloat16*)score, pos, ratio, Tc, k, offset, out);
  return (int)cudaGetLastError();
}
extern "C" int ds41_index_score(const void* q, const void* k, const void* w, const int64_t* pos, int ratio, void* out, int T, int Tc, cudaStream_t st) {
  if (T < 1 || Tc < 1) return 0;
  // many rows (prefill, verify): shared key tiles (DS41_INDEX_TILED=0 the per-row kernel); the same values either way
  static const bool tiled = !(std::getenv("DS41_INDEX_TILED") && std::getenv("DS41_INDEX_TILED")[0] == '0');
  if (tiled && T >= 8) {
    index_score_tiled_kernel<<<dim3((T + 7) / 8, (Tc + IS_KS - 1) / IS_KS), 256, 0, st>>>((const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)w, pos, ratio, (__nv_bfloat16*)out, T, Tc);
    return (int)cudaGetLastError();
  }
  index_score_kernel<<<dim3((Tc + 63) / 64, T), 128, 0, st>>>((const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)w, pos, ratio,
                                                            (__nv_bfloat16*)out, Tc);
  return (int)cudaGetLastError();
}
