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
#include <cstring>
#include <cstdlib>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
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
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }
// 16-byte asynchronous global -> shared copy (zero-filled when !valid): a thread's copies are all in flight at once
// (a plain load -> store loop waits for each load in turn: 16 serial round trips per 64-row tile)
__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }

// The attention pool as up to three row ranges read in place: [0, na) a, [na, na + nb) b, the rest c. W > 0: the
// first W indices of every query are its sliding window (keys pos - W + 1 .. pos; a = the ring of the W latest committed
// keys by slot pos % W, b = the chunk's keys from position `start`), computed here; idx then holds only the remaining
// K - W indices per query.
// lo: window keys before position lo are not attended (SWA bounded replay truncates the window to the replayed
// segment; 0 = no truncation).
struct Pools { const __nv_bfloat16 *a, *b, *c; int na, nb; int W; long long start; const int64_t* pos; long long lo;
               int pipe = 0; };

// CACHE (MODE 0, at most MAXT tiles): pass 1 keeps the warp's scores of every tile in registers, pass 2 uses them
// instead of recomputing q . k (the same values: computed once), so pass 2 only loads the tiles for P V.
// pipe (MODE 0 CACHE; prefill): every tile's rows resolved once up front (both passes), the tile
// buffer streamed in halves so the next tile loads while this one computes: pass 1 by column halves (the scores' k
// range [0, 256) then [256, 512)), pass 2 by row halves (P V's kk = 0, 16 then 32, 48). The same MMAs in the same
// order (the same bits); no room for a second tile buffer (one CTA per SM, 99 KB).
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
// DEVM (with MULTI; DS41_MID_GRAPH): the rows' pools and index pointers come from a device copy of Multi passed in `idx`
// (rewritten before every replay of a captured graph), with each row's K and split count (the grid has MAXSPLIT splits)
template <int MT, int MODE, bool CACHE = false, bool MULTI = false, bool DEVM = false>
__global__ __launch_bounds__(WARPS * 32, 1) void sparse_attn_kernel(const __nv_bfloat16* __restrict__ q, Pools pl0, const int64_t* __restrict__ idx, const float* __restrict__ sink, __nv_bfloat16* __restrict__ out, int H, int K0, float scale,
    float* __restrict__ pmx, float* __restrict__ pden, float* __restrict__ po, const __grid_constant__ Multi mt) {
  pdl_entry();
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* kvs = reinterpret_cast<__nv_bfloat16*>(smem);          // [TILE][DP]
  __nv_bfloat16* qs = kvs + TILE * DP;                                    // [H][DP]
  __nv_bfloat16* ps = qs + H * DP;                                        // [32][TP]
  float* red = reinterpret_cast<float*>(ps + 32 * TP);                   // [WARPS][32]
  float* mxs = red + WARPS * 32;                                          // [32]
  __shared__ int rows_all[MAXT * TILE];   // pipe: the source rows of every tile; else rows[] of the current one
  int* const rows = rows_all;
  const int t = MODE == 3 ? blockIdx.y : blockIdx.x, sp_id = MODE == 3 ? blockIdx.x : blockIdx.y;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, qd = lane & 3;
  const Multi* const md = DEVM ? reinterpret_cast<const Multi*>(idx) : nullptr;
  const Pools pl = MULTI ? (DEVM ? md->pl[t] : mt.pl[t]) : pl0;
  const int K = MULTI ? (DEVM ? md->K[t] : mt.K[t]) : K0;
  const int S = MULTI ? (DEVM ? md->S[t] : mt.S[t]) : MODE == 3 ? gridDim.x : gridDim.y;
  if (MULTI && sp_id >= S) return;                                       // (the grid has the batch's largest split)
  const int SS = MULTI ? MAXSPLIT : S;                                   // workspace slots per token
  const int K2 = K - pl.W;                                               // explicit indices per query
  const int64_t* ix = MULTI ? (DEVM ? md->idx[t] : mt.idx[t]) : idx + (long long)t * K2;
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
  const bool pipe = MODE == 0 && CACHE && pl.pipe;
  const int ntp = ntiles - tl0;   // pipe: tiles (<= MAXT)
  if (pipe) {
    for (int k = threadIdx.x; k < ntp * TILE; k += blockDim.x) {
      int src = -1;
      if (k < pl.W) {
        const long long kp = pt - (pl.W - 1) + k;
        src = kp < 0 || kp < pl.lo ? -1 : kp >= pl.start ? (int)(kp - pl.start) + pl.W : (int)(kp % pl.W);
      } else if (k < K) src = (int)ix[k - pl.W];
      rows_all[k] = src;
    }
    __syncthreads();
  }
  // pipe: tile i's columns [256 h, 256 h + 256) of all rows (cols = true) or rows [32 h, 32 h + 32) (cols = false), one
  // group. 8 threads per row (128 contiguous bytes per instruction and row); a thread resolves its row's address once
  // (cols: rows tid / 8 and 32 + tid / 8, 4 chunks each; rows: row 32 h + tid / 8, 8 chunks).
  auto issue = [&](int i, int h, bool cols) {
    if (i < ntp) {
#pragma unroll
      for (int q = 0; q < (cols ? 2 : 1); ++q) {
        const int r = cols ? q * (TILE / 2) + (threadIdx.x >> 3) : h * (TILE / 2) + (threadIdx.x >> 3);
        const int src = rows_all[i * TILE + r];
        const __nv_bfloat16* row = src < 0 ? pl.a : src < pl.na ? pl.a + (long long)src * D : src < pl.na + pl.nb ? pl.b + (long long)(src - pl.na) * D
                                                                                                           : pl.c + (long long)(src - pl.na - pl.nb) * D;
        const int c0 = (cols ? h * (D / 2) : 0) + (threadIdx.x & 7) * 8;
#pragma unroll
        for (int e = 0; e < (cols ? 4 : 8); ++e) cp16(kvs + r * DP + c0 + e * 64, row + c0 + e * 64, src >= 0);
      }
    }
    cp_commit();   // (empty past the last tile: the waits stay "all but the newest group")
  };
  // ldmatrix x4 per 16 x 16 A block (lane's row m 16 + (lane & 15) of q, clamped to the last head: rows >= H only feed
  // the outputs of heads >= H, never stored), and per two k steps of the warp's 8 key rows for B: the same fragments as
  // scores' ld32s.
  auto scores_part = [&](float (&s)[MT][4], int k0b) {
    const __nv_bfloat16* kr = kvs + (warp * 8 + (lane & 7)) * DP + (lane >> 3) * 8;
    const __nv_bfloat16* qr[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) qr[m] = qs + min(m * 16 + (lane & 15), H - 1) * DP + (lane >> 4) * 8;
#pragma unroll 2
    for (int k0 = k0b; k0 < k0b + D / 2; k0 += 32) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4(b0, b1, b2, b3, kr + k0);   // (k0, k0 + 8, k0 + 16, k0 + 24) of rows warp 8 + 0..7
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, qr[m] + k0);
        mma(s[m], a0, a1, a2, a3, b0, b1);
      }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, qr[m] + k0 + 16);
        mma(s[m], a0, a1, a2, a3, b2, b3);
      }
    }
  };
  // ---- pass 1: maxima
  float mx[MT][2];
  for (int m = 0; m < MT; ++m) mx[m][0] = mx[m][1] = -INFINITY;
  float sc[CACHE ? MAXT : 1][MT][4];
  if (pipe) {
    issue(0, 0, true); issue(0, 1, true);   // (the query rows join the first group)
#pragma unroll
    for (int i = 0; i < MAXT; ++i) {
      if (i >= ntp) break;
      for (int m = 0; m < MT; ++m) for (int c = 0; c < 4; ++c) sc[i][m][c] = 0.f;
      cp_wait1(); __syncthreads();
      scores_part(sc[i], 0);
      __syncthreads();
      issue(i + 1, 0, true);
      cp_wait1(); __syncthreads();
      scores_part(sc[i], D / 2);
      for (int m = 0; m < MT; ++m) for (int c = 0; c < 4; ++c) sc[i][m][c] *= scale;
      const bool v0 = rows_all[i * TILE + warp * 8 + 2 * qd] >= 0, v1 = rows_all[i * TILE + warp * 8 + 2 * qd + 1] >= 0;
      for (int m = 0; m < MT; ++m) {
        if (v0) { mx[m][0] = fmaxf(mx[m][0], sc[i][m][0]); mx[m][1] = fmaxf(mx[m][1], sc[i][m][2]); }
        if (v1) { mx[m][0] = fmaxf(mx[m][0], sc[i][m][1]); mx[m][1] = fmaxf(mx[m][1], sc[i][m][3]); }
      }
      __syncthreads();
      if (i + 1 < ntp) issue(i + 1, 1, true);
    }
    cp_wait_all(); __syncthreads();
    if (ntp > 1) { issue(0, 0, false); issue(0, 1, false); }   // pass 2's first tile (one tile: still in the buffer)
  } else if (CACHE) {
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
  // P V of rows [kb, kb + 32) of the tile in the buffer (p in ps)
  auto pv_part = [&](int kb) {
#pragma unroll
    for (int kk = kb; kk < kb + TILE / 2; kk += 16) {
      uint32_t a[MT][4];
      for (int m = 0; m < MT; ++m) ldsm_x4(a[m][0], a[m][1], a[m][2], a[m][3], ps + (m * 16 + (lane & 15)) * TP + kk + (lane >> 4) * 8);
#pragma unroll
      for (int n = 0; n < 8; n += 2) {
        const int n0 = warp * 64 + n * 8;
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, kvs + (kk + (lane & 15)) * DP + n0 + (lane >> 4) * 8);
        for (int m = 0; m < MT; ++m) { mma(o[m][n], a[m][0], a[m][1], a[m][2], a[m][3], b0, b1); mma(o[m][n + 1], a[m][0], a[m][1], a[m][2], a[m][3], b2, b3); }
      }
    }
  };
  if (pipe) {
#pragma unroll
    for (int i = 0; i < MAXT; ++i) {
      if (i >= ntp) break;
      const bool v0 = rows_all[i * TILE + warp * 8 + 2 * qd] >= 0, v1 = rows_all[i * TILE + warp * 8 + 2 * qd + 1] >= 0;
      for (int m = 0; m < MT; ++m)
        for (int hh = 0; hh < 2; ++hh) {
          const float p0 = v0 ? expf(sc[i][m][hh * 2] - mrow[m][hh]) : 0.f, p1 = v1 ? expf(sc[i][m][hh * 2 + 1] - mrow[m][hh]) : 0.f;
          den[m][hh] += p0 + p1;
          *reinterpret_cast<__nv_bfloat162*>(ps + (m * 16 + hh * 8 + g) * TP + warp * 8 + 2 * qd) = __floats2bfloat162_rn(p0, p1);
        }
      if (ntp > 1) cp_wait1();
      __syncthreads();
      pv_part(0);
      __syncthreads();
      if (ntp > 1) { issue(i + 1, 0, false); cp_wait1(); }
      __syncthreads();
      pv_part(TILE / 2);
      __syncthreads();
      if (ntp > 1 && i + 1 < ntp) issue(i + 1, 1, false);
    }
  } else
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
    pv_part(0);
    pv_part(TILE / 2);
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
// (MULTI: the token's own split count, slots MAXSPLIT apart; DEVM: the split counts of the device Multi `md`)
template <bool MULTI = false, bool DEVM = false>
__global__ void attn_fin_kernel(const float* __restrict__ pmx, const float* __restrict__ pden, const float* __restrict__ po,
                                const float* __restrict__ sink, __nv_bfloat16* __restrict__ out, int H, int S0, const __grid_constant__ Multi mt,
                                const Multi* __restrict__ md) {
  pdl_entry();
  const int t = blockIdx.x, h = blockIdx.y;
  const int S = MULTI ? (DEVM ? md->S[t] : mt.S[t]) : S0, SS = MULTI ? MAXSPLIT : S0;
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
// KT > 1 (index_score_kt): a CTA takes KT groups of 64 keys (group j at blockIdx.x * 64 KT + 64 j), q staged once for all
// of them and the key loads of the groups independent: the same per-key MMA sequence and epilogue (bitwise KT = 1)
template <int KT>
__global__ __launch_bounds__(128) void index_score_kt(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ w, const int64_t* __restrict__ pos, int ratio, __nv_bfloat16* __restrict__ out, int Tc,
    const bool* __restrict__ cand) {
  constexpr int NH = 32, ID = 128, QP = ID + 8;
  __shared__ __align__(16) __nv_bfloat16 qs[NH * QP];
  const int t = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, qd = lane & 3;
  for (int i = threadIdx.x; i < NH * ID / 8; i += blockDim.x) {
    const int h = i / (ID / 8), c = (i % (ID / 8)) * 8;
    *reinterpret_cast<uint4*>(qs + h * QP + c) = *reinterpret_cast<const uint4*>(q + ((long long)t * NH + h) * ID + c);
  }
  __syncthreads();
  const bool* cr = cand ? cand + (long long)t * Tc : nullptr;
  float wt[4][2];
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int e = 0; e < 2; ++e) wt[n][e] = __bfloat162float(w[(long long)t * NH + n * 8 + 2 * qd + e]);
  const long long lens = (pos[t] + 1) / ratio;
#pragma unroll
  for (int j = 0; j < KT; ++j) {
    const int c0 = blockIdx.x * 64 * KT + j * 64 + warp * 16;
    if (c0 >= Tc) break;
    // candidate blocks (later index layers): a tile of 16 keys none of which is a candidate is -inf without scoring
    if (cr && !__any_sync(0xffffffffu, lane < 16 && c0 + lane < Tc && cr[c0 + lane])) {
      if (qd == 0) for (int hh = 0; hh < 2; ++hh) { const int c = c0 + g + hh * 8; if (c < Tc) out[(long long)t * Tc + c] = __float2bfloat16(-INFINITY); }
      continue;
    }
    const int r0 = min(c0 + g, Tc - 1), r1 = min(c0 + g + 8, Tc - 1);
    const __nv_bfloat16* k0 = k + (long long)r0 * ID + 2 * qd;
    const __nv_bfloat16* k1 = k + (long long)r1 * ID + 2 * qd;
    uint32_t a[ID / 16][4];
#pragma unroll
    for (int kk = 0; kk < ID; kk += 16) { a[kk / 16][0] = ld32(k0 + kk); a[kk / 16][1] = ld32(k1 + kk); a[kk / 16][2] = ld32(k0 + kk + 8); a[kk / 16][3] = ld32(k1 + kk + 8); }
    float acc[4][4];
    for (int n = 0; n < 4; ++n) for (int c = 0; c < 4; ++c) acc[n][c] = 0.f;
#pragma unroll
    for (int kk = 0; kk < ID; kk += 16) {
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const __nv_bfloat16* qr = qs + (n * 8 + g) * QP + kk + 2 * qd;
        mma(acc[n], a[kk / 16][0], a[kk / 16][1], a[kk / 16][2], a[kk / 16][3], ld32(qr), ld32(qr + 8));
      }
    }
    float sum[2] = {0.f, 0.f};
    for (int n = 0; n < 4; ++n)
      for (int c = 0; c < 4; ++c) {
        const float v = fmaxf(__bfloat162float(__float2bfloat16(acc[n][c])), 0.f);
        sum[c >> 1] += __bfloat162float(__float2bfloat16(v * wt[n][c & 1]));
      }
    for (int hh = 0; hh < 2; ++hh) {
      sum[hh] += __shfl_xor_sync(0xffffffffu, sum[hh], 1);
      sum[hh] += __shfl_xor_sync(0xffffffffu, sum[hh], 2);
    }
    if (qd == 0) {
      for (int hh = 0; hh < 2; ++hh) {
        const int c = c0 + g + hh * 8;
        if (c < Tc) out[(long long)t * Tc + c] = (c >= lens || (cr && !cr[c])) ? __float2bfloat16(-INFINITY) : __float2bfloat16(sum[hh]);
      }
    }
  }
}
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
// ds41-attn-bench: a kernel variant forced (0: the default)
static int g_attn_version = 0;
extern "C" int ds41_attn_version(int v) { g_attn_version = v; return 0; }
extern "C" int ds41_sparse_attn(const void* q, const void* a, int na, const void* b, int nb, const void* c, int W, long long start,
                                long long lo, const int64_t* pos, const int64_t* idx, const float* sink, void* out, int T, int H, int K,
                                float scale, void* ws, cudaStream_t st) {
  if (!b) nb = 0;
  if (!c && !b) na = 0x7fffffff;
  Pools pl{(const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (const __nv_bfloat16*)c, na, nb, W, start, pos, lo};
  // the pipelined MODE 0 tile loads (DS41_ATTN_PIPE=0 the one-tile-at-a-time loop; the same bits)
  static const bool pipe_on = !(std::getenv("DS41_ATTN_PIPE") && std::getenv("DS41_ATTN_PIPE")[0] == '0');
  pl.pipe = pipe_on;
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
           pdl_launch(attn_fin_kernel<false>, dim3(dim3(T, H)), dim3(128), 0, st, pmx, pden, po, sink, O, H, S, none, (const Multi*)nullptr); } } while (0)
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
// md_dev: launch the DEVM kernels reading pools / indices from that device Multi (graph capture); host_out: only write
// this call's Multi into host_out (the bytes to upload to md_dev before a replay) and launch nothing
static int sparse_attn_multi_impl(const void* q, const Ds41AttnSeq* seqs, int nseq, int W, const int64_t* pos, const float* sink, void* out,
                                  int H, float scale, void* ws, cudaStream_t st, const void* md_dev, void* host_out, int* fused_out = nullptr) {
  static int sms = 0; if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  static const bool fused_on = !(std::getenv("DS41_ATTN_FUSED") && std::getenv("DS41_ATTN_FUSED")[0] == '0');
  if (!fused_on || (!ws && !host_out) || H < 1 || H > 32 || nseq < 1) return 2;
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
  // host_out: 0, or 1 when some row's tiles per split exceed MAXT (a graph captured with the fused launch no longer applies)
  if (host_out) { memcpy(host_out, &mt, sizeof(Multi)); for (int i = 0; i < T; ++i) if ((((mt.K[i] + TILE - 1) / TILE) + mt.S[i] - 1) / mt.S[i] > MAXT) return 1; return 0; }
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
    if (md_dev) { \
      static size_t setd = 0; \
      if (sm > setd) { \
        if (cudaFuncSetAttribute(sparse_attn_kernel<MT, 3, true, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) || \
            cudaFuncSetAttribute(sparse_attn_kernel<MT, 1, false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) || \
            cudaFuncSetAttribute(sparse_attn_kernel<MT, 2, false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm)) { cudaGetLastError(); return 2; } \
        setd = sm; } \
      const int64_t* MD = (const int64_t*)md_dev; \
      const bool fd = T * MAXSPLIT <= sms && fused && !force3; \
      if (fused_out) *fused_out = fd; \
      if (fd) pdl_launch((sparse_attn_kernel<MT, 3, true, true, true>), dim3(dim3(MAXSPLIT, T)), dim3(WARPS * 32), sm, st, Q, none, MD, sink, O, H, 0, scale, pmx, pden, po, mt); \
      else { pdl_launch((sparse_attn_kernel<MT, 1, false, true, true>), dim3(dim3(T, MAXSPLIT)), dim3(WARPS * 32), sm, st, Q, none, MD, sink, O, H, 0, scale, pmx, pden, po, mt); \
             pdl_launch((sparse_attn_kernel<MT, 2, false, true, true>), dim3(dim3(T, MAXSPLIT)), dim3(WARPS * 32), sm, st, Q, none, MD, sink, O, H, 0, scale, pmx, pden, po, mt); \
             pdl_launch((attn_fin_kernel<true, true>), dim3(dim3(T, H)), dim3(128), 0, st, pmx, pden, po, sink, O, H, 0, mt, (const Multi*)md_dev); } \
      break; } \
    if (fused) pdl_launch((sparse_attn_kernel<MT, 3, true, true>), dim3(dim3(smax, T)), dim3(WARPS * 32), sm, st, Q, none, (const int64_t*)nullptr, sink, O, H, 0, scale, pmx, pden, po, mt); \
    else { pdl_launch((sparse_attn_kernel<MT, 1, false, true>), dim3(dim3(T, smax)), dim3(WARPS * 32), sm, st, Q, none, (const int64_t*)nullptr, sink, O, H, 0, scale, pmx, pden, po, mt); \
           pdl_launch((sparse_attn_kernel<MT, 2, false, true>), dim3(dim3(T, smax)), dim3(WARPS * 32), sm, st, Q, none, (const int64_t*)nullptr, sink, O, H, 0, scale, pmx, pden, po, mt); \
           pdl_launch(attn_fin_kernel<true>, dim3(dim3(T, H)), dim3(128), 0, st, pmx, pden, po, sink, O, H, 0, mt, (const Multi*)nullptr); } } while (0)
  if (H <= 16) DS41_ATTN_MULTI(1); else DS41_ATTN_MULTI(2);
#undef DS41_ATTN_MULTI
  return (int)cudaGetLastError();
}

extern "C" int ds41_sparse_attn_multi(const void* q, const Ds41AttnSeq* seqs, int nseq, int W, const int64_t* pos, const float* sink, void* out,
                                      int H, float scale, void* ws, cudaStream_t st) {
  return sparse_attn_multi_impl(q, seqs, nseq, W, pos, sink, out, H, scale, ws, st, nullptr, nullptr);
}
// DS41_MID_GRAPH: the same launch reading the rows' pools / indices from md_dev (a device Multi of ds41_attn_multi_bytes();
// fill it with ds41_attn_multi_build for this step's sequences before every replay)
// (*fused_out: whether the fused single launch was captured)
extern "C" int ds41_sparse_attn_multi_dev(const void* q, const Ds41AttnSeq* seqs, int nseq, int W, const int64_t* pos, const float* sink, void* out,
                                          int H, float scale, void* ws, const void* md_dev, int* fused_out, cudaStream_t st) {
  return sparse_attn_multi_impl(q, seqs, nseq, W, pos, sink, out, H, scale, ws, st, md_dev, nullptr, fused_out);
}
extern "C" int ds41_attn_multi_build(const Ds41AttnSeq* seqs, int nseq, int W, const int64_t* pos, int H, void* host_out) {
  return sparse_attn_multi_impl(nullptr, seqs, nseq, W, pos, nullptr, nullptr, H, 1.f, nullptr, nullptr, nullptr, host_out);
}
extern "C" size_t ds41_attn_multi_bytes() { return sizeof(Multi); }
extern "C" int ds41_h2d(void* dst, const void* src, size_t n, cudaStream_t st) { return (int)cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, st); }
// staging-buffer fences (DS41_MID_GRAPH): an event recorded after a call's uploads; the host waits on it before
// rewriting that buffer (the uploads read pinned memory when they run, not when issued)
extern "C" void* ds41_event_new() { cudaEvent_t e = nullptr; return cudaEventCreateWithFlags(&e, cudaEventDisableTiming) ? nullptr : (void*)e; }
extern "C" int ds41_event_record(void* e, cudaStream_t st) { return (int)cudaEventRecord((cudaEvent_t)e, st); }
extern "C" int ds41_event_sync(void* e) { return (int)cudaEventSynchronize((cudaEvent_t)e); }
extern "C" int ds41_event_free(void* e) { return (int)cudaEventDestroy((cudaEvent_t)e); }

// topk_idx for a few long rows (decode / verify at long contexts: one CTA per row scanned a 1M-wide row three times on
// one SM, ~0.3 ms per index layer at 512k): each row split into slices of TKW_SL keys, one CTA per (slice, row). The
// same selection and output as topk_idx_kernel (the k largest keys, ties to the lower index, ascending index order,
// -1 at index >= lens): the two 8-bit radix passes add slice histograms into a global one, a one-CTA-per-row step finds
// the digit / count still needed (the same scan as topk_threshold), then every slice counts its keys above / equal to
// the threshold and writes its selected indices at the offset the earlier slices' counts give. Not in topk_select.cuh:
// its users take one CTA per row; this is the DS41 output format (pool indices + offset, -1 past lens).
constexpr int TKW_SL = 16384, TKW_NT = 256;
struct TkwRow { unsigned prefix; int need; };
__global__ __launch_bounds__(TKW_NT) void tkw_hist_kernel(const __nv_bfloat16* __restrict__ score, int Tc, unsigned* __restrict__ hist,
                                                          const TkwRow* __restrict__ st, int pass) {
  __shared__ unsigned h[256];
  const int t = blockIdx.y, c0 = blockIdx.x * TKW_SL, c1 = min(Tc, c0 + TKW_SL);
  const __nv_bfloat16* row = score + (long long)t * Tc;
  h[threadIdx.x] = 0;
  __syncthreads();
  const unsigned want = pass ? st[t].prefix : 0u;
  for (int i = c0 + threadIdx.x; i < c1; i += TKW_NT) {
    const unsigned key = tk_key(row[i]);
    if (pass == 0) atomicAdd(&h[key >> 8], 1u);
    else if ((key >> 8) == want) atomicAdd(&h[key & 255u], 1u);
  }
  __syncthreads();
  if (h[threadIdx.x]) atomicAdd(&hist[t * 256 + threadIdx.x], h[threadIdx.x]);
}
// digit of this pass: the count from the top reaches need; hist[t] zeroed for the next pass
__global__ __launch_bounds__(32) void tkw_digit_kernel(unsigned* __restrict__ hist, TkwRow* __restrict__ st, int k, int pass) {
  const int t = blockIdx.x, lane = threadIdx.x;
  unsigned* hh = hist + t * 256;
  unsigned c[8]; unsigned s = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) { c[j] = hh[255 - 8 * lane - j]; s += c[j]; }
  unsigned inc = s;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) { const unsigned v = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += v; }
  const unsigned exc = inc - s, nd = pass ? (unsigned)st[t].need : (unsigned)k;
  const unsigned hit = __ballot_sync(0xffffffffu, inc >= nd);
  const int first = __ffs(hit) - 1;
  __syncwarp();
  if (lane == first) {
    unsigned run = exc; int j = 0;
    for (; j < 8; ++j) { if (run + c[j] >= nd) break; run += c[j]; }
    const unsigned digit = 255u - 8u * lane - j;
    st[t].prefix = pass ? (st[t].prefix << 8) | digit : digit;
    st[t].need = (int)(nd - run);
  }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 8; ++j) hh[8 * lane + j] = 0;
}
// per slice: keys above / equal to the threshold
__global__ __launch_bounds__(TKW_NT) void tkw_count_kernel(const __nv_bfloat16* __restrict__ score, int Tc, const TkwRow* __restrict__ st,
                                                           int2* __restrict__ cnt) {
  const int t = blockIdx.y, c0 = blockIdx.x * TKW_SL, c1 = min(Tc, c0 + TKW_SL);
  const __nv_bfloat16* row = score + (long long)t * Tc;
  const unsigned thr = st[t].prefix;
  int gt = 0, eq = 0;
  for (int i = c0 + threadIdx.x; i < c1; i += TKW_NT) { const unsigned key = tk_key(row[i]); gt += key > thr; eq += key == thr; }
  typedef cub::BlockReduce<int, TKW_NT> Red;
  __shared__ typename Red::TempStorage tg, te;
  gt = Red(tg).Sum(gt); eq = Red(te).Sum(eq);
  if (threadIdx.x == 0) cnt[t * gridDim.x + blockIdx.x] = make_int2(gt, eq);
}
// per slice: the selected indices in index order, after those of the earlier slices
__global__ __launch_bounds__(TKW_NT) void tkw_write_kernel(const __nv_bfloat16* __restrict__ score, const int64_t* __restrict__ pos, int ratio,
                                                           int Tc, int k, long long offset, const TkwRow* __restrict__ st,
                                                           const int2* __restrict__ cnt, int64_t* __restrict__ out) {
  typedef cub::BlockScan<int, TKW_NT> Scan;
  __shared__ typename Scan::TempStorage tmp;
  __shared__ int s_carry[2];
  const int t = blockIdx.y, p = blockIdx.x, c0 = p * TKW_SL, c1 = min(Tc, c0 + TKW_SL);
  const unsigned thr = st[t].prefix, need = (unsigned)st[t].need;
  if (threadIdx.x == 0) {
    int gtb = 0, eqb = 0;
    for (int j = 0; j < p; ++j) { const int2 v = cnt[t * gridDim.x + j]; gtb += v.x; eqb += v.y; }
    s_carry[0] = eqb; s_carry[1] = gtb + min(eqb, (int)need);
  }
  __syncthreads();
  const int2 mine = cnt[t * gridDim.x + p];
  if (mine.x == 0 && (mine.y == 0 || (unsigned)s_carry[0] >= need)) return;
  const __nv_bfloat16* row = score + (long long)t * Tc;
  const long long lens = (pos[t] + 1) / ratio;
  for (int b0 = c0; b0 < c1; b0 += TKW_NT) {
    const int i = b0 + threadIdx.x;
    const unsigned key = i < c1 ? tk_key(row[i]) : 0u;
    const int eq = i < c1 && key == thr, gt = i < c1 && key > thr;
    if (__syncthreads_count(eq | gt) == 0) continue;
    int eqr, eqt;
    Scan(tmp).ExclusiveSum(eq, eqr, eqt);
    __syncthreads();
    const int sel = gt || (eq && (unsigned)(s_carry[0] + eqr) < need);
    int sr, stt;
    Scan(tmp).ExclusiveSum(sel, sr, stt);
    if (sel) out[(long long)t * k + s_carry[1] + sr] = i >= lens ? -1 : (long long)i + offset;
    __syncthreads();
    if (threadIdx.x == 0) { s_carry[0] += eqt; s_carry[1] += stt; }
    __syncthreads();
  }
}
// workspace of the wide path (eager only: grown with cudaMalloc, never during a graph capture)
static int g_tkw_force = -1;   // (ds41-kernel-test: 0 off, 1 on, -1 DS41_TOPK_WIDE_MIN)
extern "C" void ds41_topk_wide_force(int v) { g_tkw_force = v; }
static unsigned* g_tkw_hist = nullptr; static TkwRow* g_tkw_st = nullptr; static int2* g_tkw_cnt = nullptr; static size_t g_tkw_cap = 0;
static bool tkw_run(const __nv_bfloat16* score, const int64_t* pos, int ratio, int T, int Tc, int k, long long offset, int64_t* out, cudaStream_t st) {
  static const int wide_min = std::getenv("DS41_TOPK_WIDE_MIN") ? std::atoi(std::getenv("DS41_TOPK_WIDE_MIN")) : 65536;
  if (g_tkw_force == 0 || (g_tkw_force < 0 && (wide_min <= 0 || Tc < wide_min)) || T > 16) return false;
  cudaStreamCaptureStatus cs;
  if (cudaStreamIsCapturing(st, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) return false;
  const int P = (Tc + TKW_SL - 1) / TKW_SL;
  const size_t need_cap = (size_t)T * P;
  if (need_cap > g_tkw_cap || !g_tkw_hist) {
    cudaStreamSynchronize(st);
    cudaFree(g_tkw_hist); cudaFree(g_tkw_st); cudaFree(g_tkw_cnt);
    const size_t cap = need_cap > 16 * 128 ? need_cap : 16 * 128;   // 16 rows of up to 2M keys
    if (cudaMalloc(&g_tkw_hist, 16 * 256 * sizeof(unsigned)) || cudaMalloc(&g_tkw_st, 16 * sizeof(TkwRow)) || cudaMalloc(&g_tkw_cnt, cap * sizeof(int2))) {
      g_tkw_hist = nullptr; g_tkw_cap = 0; return false;
    }
    cudaMemset(g_tkw_hist, 0, 16 * 256 * sizeof(unsigned));   // the digit kernel leaves it zeroed after each pass
    g_tkw_cap = cap;
  }
  const dim3 g(P, T);
  tkw_hist_kernel<<<g, TKW_NT, 0, st>>>(score, Tc, g_tkw_hist, g_tkw_st, 0);
  tkw_digit_kernel<<<T, 32, 0, st>>>(g_tkw_hist, g_tkw_st, k, 0);
  tkw_hist_kernel<<<g, TKW_NT, 0, st>>>(score, Tc, g_tkw_hist, g_tkw_st, 1);
  tkw_digit_kernel<<<T, 32, 0, st>>>(g_tkw_hist, g_tkw_st, k, 1);
  tkw_count_kernel<<<g, TKW_NT, 0, st>>>(score, Tc, g_tkw_st, g_tkw_cnt);
  tkw_write_kernel<<<g, TKW_NT, 0, st>>>(score, pos, ratio, Tc, k, offset, g_tkw_st, g_tkw_cnt, out);
  return true;
}
extern "C" int ds41_topk_idx(const void* score, const int64_t* pos, int ratio, int T, int Tc, int k, long long offset, int64_t* out, cudaStream_t st) {
  if (T < 1 || k < 1 || k > Tc) return 1;
  // few long rows (DS41_TOPK_WIDE_MIN: least width, 65536; 0 off): the multi-CTA path, the same output (1M-wide row:
  // 0.87 ms -> 0.06 ms; at 32k the six launches lose)
  if (tkw_run((const __nv_bfloat16*)score, pos, ratio, T, Tc, k, offset, out, st)) return (int)cudaGetLastError();
  pdl_launch(topk_idx_kernel, dim3(T), dim3(TK_THREADS), 0, st, (const __nv_bfloat16*)score, pos, ratio, Tc, k, offset, out);
  return (int)cudaGetLastError();
}
// Key-split indexer (decode at long contexts; model.rs index_key_split): the pack / unpack / final steps around the
// all-gather of each rank's top k, one launch each instead of ~20 small libtorch ops per index layer.
// pack: pairs[t][j] = (score bits of local index loc[t][j] (-inf bits for -1), loc + k0 or -1)
__global__ void ks_pack_kernel(const __nv_bfloat16* __restrict__ s, int n, const int64_t* __restrict__ loc, int k, int k0, int2* __restrict__ out, int T) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * k) return;
  const int t = i / k;
  const long long l = loc[i];
  const unsigned short bits = l >= 0 ? __bfloat16_as_ushort(s[(long long)t * n + l]) : (unsigned short)0xFF80u;
  out[i] = make_int2((int)bits, l >= 0 ? (int)(l + k0) : -1);
}
// unpack: all [W][T][k] pairs -> vals [T][W k] BF16 and idx [T][W k] in rank order
__global__ void ks_unpack_kernel(const int2* __restrict__ all, int W, int T, int k, __nv_bfloat16* __restrict__ vals, int* __restrict__ idx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= W * T * k) return;
  const int r = i / (T * k), t = (i / k) % T, j = i % k;
  const int2 p = all[i];
  const long long o = (long long)t * W * k + (long long)r * k + j;
  vals[o] = __ushort_as_bfloat16((unsigned short)p.x);
  idx[o] = p.y;
}
// final: ids[t][j] = idx[t][sel[t][j]] + offset (-1 stays -1)
__global__ void ks_final_kernel(const int64_t* __restrict__ sel, const int* __restrict__ idx, int wk, int k, long long offset, int64_t* __restrict__ out, int T) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * k) return;
  const int t = i / k;
  const int g = idx[(long long)t * wk + sel[i]];
  out[i] = g < 0 ? -1 : (long long)g + offset;
}
extern "C" int ds41_ks_pack(const void* s, int n, const int64_t* loc, int k, int k0, void* out, int T, cudaStream_t st) {
  ks_pack_kernel<<<(T * k + 255) / 256, 256, 0, st>>>((const __nv_bfloat16*)s, n, loc, k, k0, (int2*)out, T);
  return (int)cudaGetLastError();
}
extern "C" int ds41_ks_unpack(const void* all, int W, int T, int k, void* vals, int* idx, cudaStream_t st) {
  ks_unpack_kernel<<<(W * T * k + 255) / 256, 256, 0, st>>>((const int2*)all, W, T, k, (__nv_bfloat16*)vals, idx);
  return (int)cudaGetLastError();
}
extern "C" int ds41_ks_final(const int64_t* sel, const int* idx, int wk, int k, long long offset, int64_t* out, int T, cudaStream_t st) {
  ks_final_kernel<<<(T * k + 255) / 256, 256, 0, st>>>(sel, idx, wk, k, offset, out, T);
  return (int)cudaGetLastError();
}
// cand (Bool [T, Tc] or null): the scores of the non-candidate keys are -inf (the per-row kernel skips whole tiles of
// them: index_score_kt); the tiled kernel takes no cand (returns 2 then: the caller masks)
static int g_index_kt_force = 0;   // (ds41-kernel-test)
extern "C" void ds41_index_score_kt_force(int kt) { g_index_kt_force = kt; }
static int g_index_tiled_min_force = 0;   // (ds41-kernel-test)
extern "C" void ds41_index_tiled_min_force(int v) { g_index_tiled_min_force = v; }
extern "C" int ds41_index_score_cand(const void* q, const void* k, const void* w, const int64_t* pos, int ratio, const bool* cand, void* out, int T, int Tc, cudaStream_t st);
extern "C" int ds41_index_score(const void* q, const void* k, const void* w, const int64_t* pos, int ratio, void* out, int T, int Tc, cudaStream_t st) {
  return ds41_index_score_cand(q, k, w, pos, ratio, nullptr, out, T, Tc, st);
}
extern "C" int ds41_index_score_cand(const void* q, const void* k, const void* w, const int64_t* pos, int ratio, const bool* cand, void* out, int T, int Tc, cudaStream_t st) {
  if (T < 1 || Tc < 1) return 0;
  // many rows (prefill, verify): shared key tiles (DS41_INDEX_TILED=0 the per-row kernel); the same values either way.
  // DS41_INDEX_TILED_LONG (default on): also few rows against many keys (decode / verify at long contexts, where the
  // per-row kernel reads every key once per row and its 131k+ key ranges miss L2: 512k keys, 1 / 7 rows 0.98 / 6.9 ms
  // per-row, 0.77 / 0.79 ms tiled; below ~64k keys the tiled kernel's ~0.1 ms floor loses)
  static const bool tiled = !(std::getenv("DS41_INDEX_TILED") && std::getenv("DS41_INDEX_TILED")[0] == '0');
  static const bool tiled_long = !(std::getenv("DS41_INDEX_TILED_LONG") && std::getenv("DS41_INDEX_TILED_LONG")[0] == '0');
  const int tl_force = g_index_tiled_min_force;   // (ds41-kernel-test: 99 never, 1 always for few rows)
  const bool long_rows = tl_force ? T >= tl_force : (tiled_long && !cand && (Tc >= 131072 || (Tc >= 65536 && T >= 5)));
  if (tiled && (T >= 8 || long_rows)) {
    if (cand) return 2;
    index_score_tiled_kernel<<<dim3((T + 7) / 8, (Tc + IS_KS - 1) / IS_KS), 256, 0, st>>>((const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)w, pos, ratio, (__nv_bfloat16*)out, T, Tc);
    return (int)cudaGetLastError();
  }
  // DS41_INDEX_KT (1; 2 / 4 / 8: index_score_kt, bitwise, measured within noise: the kernel is at ~150 GB/s at 131k keys)
  static const int kt_env = std::getenv("DS41_INDEX_KT") ? std::atoi(std::getenv("DS41_INDEX_KT")) : 1;
  const int kt0 = g_index_kt_force ? g_index_kt_force : kt_env;
  const int kt = cand && kt0 < 2 ? 4 : kt0;
  const auto qq = (const __nv_bfloat16*)q; const auto kq = (const __nv_bfloat16*)k; const auto wq = (const __nv_bfloat16*)w; const auto oq = (__nv_bfloat16*)out;
  switch (kt) {
    case 2: index_score_kt<2><<<dim3((Tc + 127) / 128, T), 128, 0, st>>>(qq, kq, wq, pos, ratio, oq, Tc, cand); break;
    case 4: index_score_kt<4><<<dim3((Tc + 255) / 256, T), 128, 0, st>>>(qq, kq, wq, pos, ratio, oq, Tc, cand); break;
    case 8: index_score_kt<8><<<dim3((Tc + 511) / 512, T), 128, 0, st>>>(qq, kq, wq, pos, ratio, oq, Tc, cand); break;
    default: index_score_kernel<<<dim3((Tc + 63) / 64, T), 128, 0, st>>>(qq, kq, wq, pos, ratio, oq, Tc);
  }
  return (int)cudaGetLastError();
}
