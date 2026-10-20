// SPDX-License-Identifier: MIT
// Shared top-k threshold search for one row per CTA (GLM DSA dsa_topk_rows, DeepSeek-V4.1 topk_idx): radix select over
// order-preserving unsigned keys, DIGITS 8-bit digits from the top. Finds T, the k-th largest of the keys key(i), i < n,
// plus `pad` virtual copies of `padkey` (a known tail, e.g. GLM's -FLT_MAX mask, counted without scanning it), and
// `need`, how many keys equal to T the top k takes after all keys above T. Which of the equal keys to take and the
// output order are up to the caller.
// All NT threads of the block call it (NT a multiple of 32, >= 256); the key loader is called for i < n only.
#pragma once

struct TopkSelectSmem { unsigned hist[256]; unsigned prefix, mask; int need; };

template <int DIGITS, int NT, class KeyAt>
__device__ __forceinline__ void topk_threshold(TopkSelectSmem& sm, KeyAt key_at, int n, int k, int pad, unsigned padkey, unsigned& T, int& need) {
  static_assert(NT % 32 == 0 && NT >= 256, "topk_threshold: block size");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) { sm.prefix = 0; sm.mask = 0; sm.need = k; }
  for (int d = DIGITS - 1; d >= 0; --d) {
    if (tid < 256) sm.hist[tid] = 0;
    __syncthreads();
    const unsigned prefix = sm.prefix, mask = sm.mask;
    const int sh = 8 * d;
    for (int i = tid; i < n; i += NT) { const unsigned key = key_at(i); if ((key & mask) == prefix) atomicAdd(&sm.hist[(key >> sh) & 255u], 1u); }
    if (tid == 0 && pad > 0 && (padkey & mask) == prefix) atomicAdd(&sm.hist[(padkey >> sh) & 255u], (unsigned)pad);
    __syncthreads();
    if (warp == 0) {
      // lane l owns digits 255-8l .. 248-8l (descending); find the digit where the running count reaches need
      unsigned c[8]; unsigned s = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) { c[j] = sm.hist[255 - 8 * lane - j]; s += c[j]; }
      unsigned inc = s;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) { const unsigned v = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += v; }
      const unsigned exc = inc - s, nd = (unsigned)sm.need;
      const unsigned hit = __ballot_sync(0xffffffffu, inc >= nd);
      const int first = __ffs(hit) - 1;
      if (lane == first) {
        unsigned run = exc; int j = 0;
        for (; j < 8; ++j) { if (run + c[j] >= nd) break; run += c[j]; }
        const unsigned digit = 255u - 8u * lane - j;
        sm.prefix = prefix | (digit << sh); sm.mask = mask | (255u << sh); sm.need = (int)(nd - run);
      }
    }
    __syncthreads();
  }
  T = sm.prefix; need = sm.need;
}
