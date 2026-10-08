#pragma once
// Routed EXL3 MoE experts shared by the Qwen3.8-Flash-Next and GLM-5.3-Flash engines: grouping by expert on the
// device (fixed grids, graph-capturable), grouped EXL3 products for gate/up and down, SwiGLU, and a per-row weighted
// sum in slot order (deterministic). Everything model-specific is a compile-time config C (see QwenMoe / GlmMoe):
//   E experts, TOPK routes per row, H hidden size, I expert intermediate size (local to the rank under TP),
//   CBK EXL3 codebook (2 = mul1, 1 = mcg), XT input type, WT route weight type, IT route index type, act(g, u).
//
// Expert table per layer, tab[9][E] device pointers: gate_tr, up_tr, down_tr, gate_suh, up_suh, down_suh,
// gate_svh, up_svh, down_svh.
// Layout: the P = R * TOPK routed (row, expert) pairs get compact rows ordered by expert, then row (a stable counting
// sort); pairmap[r * TOPK + k] = that row. Work items cover at most GR rows of one expert (item g: expert gexp[g],
// first row goff[g], gcnt[g] rows). Decode / verify chains (R <= 64): GR 16, EXL3 GEMV with K split in S slices.
// Prefill chunks: GR 128, a GEMM kernel (A tiles in shared memory, every decoded weight tile used by 8 row tiles).
#include "qwen_exl3.cuh"
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

namespace moex {

struct GemvCfgT { int tpw, pf, warps, s1, s2, ctas, apf, fused; };
struct QwenMoe
{
    static constexpr int E = 512, TOPK = 10, H = 2560, I = 640, CBK = 2;
    using XT = float; using WT = float; using IT = int;
    __device__ __forceinline__ static float act(float a, float u) { return a / (1.0f + __expf(-a)) * u; }
    static constexpr bool PERSIST = false, FINISH_COMBINE = false, PDL = false, APF = false, YD16 = false;
    static constexpr const char* GEMV_ENV = "QWEN_MOE_GEMV";
    static GemvCfgT gemv_default(int S, int) { return {2, 4, 4, S, S, 0, 0, 0}; }
};
// GLM-5.3-Flash under TP2: 288 local experts, top-8, hidden 4096, expert intermediate 1024 per rank (2048 / 2), mcg
// codebook; SwiGLU with the checkpoint's limit 10 (gate <= 10, -10 <= up <= 10), as in the reference expert path.
struct GlmMoe
{
    static constexpr int E = 288, TOPK = 8, H = 4096, I = 1024, CBK = 1;
    using XT = half; using WT = half; using IT = int64_t;
    __device__ __forceinline__ static float act(float a, float u)
    {
        a = fminf(a, 10.0f);
        u = fminf(fmaxf(u, -10.0f), 10.0f);
        return a / (1.0f + __expf(-a)) * u;
    }
    // YD16: prefill per-pair expert outputs in fp16 before the fixed-order slot sum (as the reference grouped path stores them)
    static constexpr bool PERSIST = true, FINISH_COMBINE = true, PDL = true, APF = true, YD16 = true;
    static constexpr const char* GEMV_ENV = "GLM53_MOE_GEMV";
    // glm-moe-ab (2026-10-08, layer 3, same process vs the cooperative library): up to 4 rows the fused kernel (stage 1,
    // mid and stage 2 in one persistent grid), above it the separate persistent GEMVs; 4 n-tiles per warp, 4 k-tiles ahead.
    static GemvCfgT gemv_default(int, int R)
    {
        if (R <= 1) return {4, 4, 4, 2, 1, 0, 0, 1};
        if (R <= 2) return {4, 4, 4, 4, 1, 0, 0, 1};
        if (R <= 4) return {4, 4, 4, 2, 2, 0, 0, 1};
        return {4, 4, 4, 2, 1, 0, 0, 0};
    }
};

constexpr int CB = 64;   // rows per counting block

// Programmatic dependent launch (C::PDL): every decode kernel waits for its predecessor's completion first thing and
// lets its successor launch right away, so launch latency overlaps the predecessor's tail. Without the launch attribute
// (Qwen) griddepcontrol.wait returns at once.
__device__ __forceinline__ void pdl_enter()
{
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}
template <class C, typename... KArgs, typename... Args>
void klaunch(void (*k)(KArgs...), dim3 grid, dim3 block, cudaStream_t st, Args... args)
{
    if constexpr (C::PDL)
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = 0; cfg.stream = st;
        cudaLaunchAttribute a[1];
        a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = a; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, k, ((KArgs) args)...);
    }
    else k<<<grid, block, 0, st>>>(((KArgs) args)...);
}

template <class C, bool CG>
__device__ __forceinline__ void mid_row_block(const float* __restrict__ part, int S, const uint64_t* __restrict__ tab, int P, int e,
                                              int row, int b, int lane, half* __restrict__ xh2);
inline int sm_count()
{
    static int n = [] { int d = 0, v = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d); return v; }();
    return n;
}
__device__ __forceinline__ float tof(float v) { return v; }
__device__ __forceinline__ float tof(half v) { return __half2float(v); }
// 16 consecutive input values from x + off (off < 0: zeros)
__device__ __forceinline__ void ld16(const float* x, int64_t off, float* xa)
{
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const float4 v = off >= 0 ? __ldg((const float4*) (x + off) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
        xa[4 * i] = v.x; xa[4 * i + 1] = v.y; xa[4 * i + 2] = v.z; xa[4 * i + 3] = v.w;
    }
}
__device__ __forceinline__ void ld16(const half* x, int64_t off, float* xa)
{
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        const uint4 u = off >= 0 ? __ldg((const uint4*) (x + off) + i) : make_uint4(0u, 0u, 0u, 0u);
        const half2* h = (const half2*) &u;
        #pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 f = __half22float2(h[j]); xa[8 * i + 2 * j] = f.x; xa[8 * i + 2 * j + 1] = f.y; }
    }
}

// ---- grouping: stable counting sort of the pairs by expert (rows ascending within an expert) ----
// count: grid nb = ceil(R / CB) x E threads (thread = expert): per-block counts cblk[b][e]
template <class C>
__global__ __launch_bounds__(512) void count_kernel(const typename C::IT* __restrict__ idx, int R, int* __restrict__ cblk)
{
    __shared__ int sidx[CB * C::TOPK];
    const int b = blockIdx.x, e = threadIdx.x, r0 = b * CB, nr = min(CB, R - r0);
    for (int i = e; i < nr * C::TOPK; i += C::E) sidx[i] = (int) idx[r0 * C::TOPK + i];
    __syncthreads();
    int c = 0;
    for (int i = 0; i < nr * C::TOPK; ++i) c += sidx[i] == e;
    cblk[b * C::E + e] = c;
}

// scan: one block of E threads: per expert the block offsets (cblk becomes exclusive prefix over blocks), expert row
// offsets eoff[e], items (GR rows at most) in expert order: G, gexp, goff, gcnt
template <class C>
__global__ __launch_bounds__(512) void scan_kernel(int* __restrict__ cblk, int nb, int GR, int* __restrict__ eoff,
                                                   int* __restrict__ G, int* __restrict__ gexp, int* __restrict__ goff,
                                                   int* __restrict__ gcnt)
{
    __shared__ int s_rows[C::E], s_items[C::E];
    const int e = threadIdx.x;
    int tot = 0;
    for (int b = 0; b < nb; ++b) { int c = cblk[b * C::E + e]; cblk[b * C::E + e] = tot; tot += c; }
    s_rows[e] = tot;
    s_items[e] = (tot + GR - 1) / GR;
    __syncthreads();
    if (e == 0)
    {
        int ra = 0, ia = 0;
        for (int i = 0; i < C::E; ++i)
        {
            int nr = s_rows[i], ni = s_items[i];
            s_rows[i] = ra; s_items[i] = ia;
            ra += nr; ia += ni;
        }
        *G = ia;
    }
    __syncthreads();
    eoff[e] = s_rows[e];
    for (int i = 0, it = s_items[e]; i * GR < tot; ++i, ++it)
    {
        gexp[it] = e;
        goff[it] = s_rows[e] + i * GR;
        gcnt[it] = min(GR, tot - i * GR);
    }
}

// fill: grid nb x E (thread = expert): pairmap and the source row of every compact row
template <class C>
__global__ __launch_bounds__(512) void fill_kernel(const typename C::IT* __restrict__ idx, int R, const int* __restrict__ cblk,
                                                   const int* __restrict__ eoff, int* __restrict__ pairmap, int* __restrict__ prow)
{
    __shared__ int sidx[CB * C::TOPK];
    const int b = blockIdx.x, e = threadIdx.x, r0 = b * CB, nr = min(CB, R - r0);
    for (int i = e; i < nr * C::TOPK; i += C::E) sidx[i] = (int) idx[r0 * C::TOPK + i];
    __syncthreads();
    int pos = eoff[e] + cblk[b * C::E + e];
    for (int i = 0; i < nr * C::TOPK; ++i)
        if (sidx[i] == e) { pairmap[r0 * C::TOPK + i] = pos; prow[pos] = r0 + i / C::TOPK; ++pos; }
}

// xh[t][row][K] = had128(x[src] * suh_t[e]) over the P compact rows (stage 1: src = source row of the pair;
// stage 2, x_is_act: src = the compact row itself). Warp per (t, row, kb).
template <class C, typename XT>
__global__ void had_grouped_kernel(const XT* __restrict__ x, int64_t ldx, int x_is_act, const uint64_t* __restrict__ tab,
                                   int suh_slot0, int T, int P, const int* __restrict__ prow, const int* __restrict__ pexp,
                                   int K, half* __restrict__ xh)
{
    pdl_enter();
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int nb = K / 128;
    int kb = warp % nb, rest = warp / nb;
    int row = rest % P, t = rest / P;
    if (t >= T) return;
    const half* suh = (const half*) tab[(suh_slot0 + t) * C::E + pexp[row]];
    int64_t src = x_is_act ? (int64_t) row : (int64_t) prow[row];
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int k = kb * 128 + lane * 4 + i;
        v[i] = tof(x[src * ldx + k]) * __half2float(suh[k]);
    }
    qexl3::fwht128(v, lane);
    half* d = xh + ((int64_t) t * P + row) * K + kb * 128 + lane * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = __float2half_rn(v[i]);
}

// expert of every compact row (from the items)
static __global__ void pexp_kernel(const int* __restrict__ G, const int* __restrict__ gexp, const int* __restrict__ goff,
                            const int* __restrict__ gcnt, int* __restrict__ pexp)
{
    int g = blockIdx.x;
    if (g >= *G) return;
    for (int j = threadIdx.x; j < gcnt[g]; j += blockDim.x) pexp[goff[g] + j] = gexp[g];
}

// decode / verify: grid (N/128, S, T * Gmax), GEMV per item; part[s][t][row][N]
template <class C, int BITS>
__global__ __launch_bounds__(128) void gemv_grouped_kernel(const half* __restrict__ xh, const uint64_t* __restrict__ tab,
                                                           int tr_slot0, const int* __restrict__ G, const int* __restrict__ gexp,
                                                           const int* __restrict__ goff, const int* __restrict__ gcnt, int Gmax,
                                                           int T, int P, int K, int N, int kts, float* __restrict__ part)
{
    int tg = blockIdx.z, t = tg / Gmax, g = tg % Gmax;
    if (g >= *G) return;
    const uint32_t* tr = (const uint32_t*) tab[(tr_slot0 + t) * C::E + gexp[g]];
    const int64_t r0 = (int64_t) t * P + goff[g];
    qexl3::gemv_block<BITS, 2, 4, 4, 1, C::CBK>(xh + r0 * K, tr, part + ((int64_t) blockIdx.y * T * P + r0) * N, gcnt[g], K, N, blockIdx.x,
                            blockIdx.y * kts, kts);
}

// decode / verify, persistent: a fixed grid (the resident CTAs) loops over the tasks (t, item g, slice s, n-block), n-block
// fastest (neighbouring CTAs read neighbouring trellis bytes). Per task exactly gemv_grouped_kernel's gemv_block, so the
// partials are bitwise equal for any grid, tile shape or prefetch depth; only the schedule and the bytes in flight differ.
template <class C, int BITS, int TPW, int PF, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void gemv_persist_kernel(const half* __restrict__ xh, const uint64_t* __restrict__ tab,
                                                                  int tr_slot0, const int* __restrict__ G, const int* __restrict__ gexp,
                                                                  const int* __restrict__ goff, const int* __restrict__ gcnt, int T, int P,
                                                                  int K, int N, int S, int kts, float* __restrict__ part)
{
    pdl_enter();
    const int nb = N / (16 * WARPS * TPW), gn = *G, total = T * gn * S * nb;
    for (int task = blockIdx.x; task < total; task += gridDim.x)
    {
        const int nblk = task % nb;
        int rest = task / nb;
        const int s = rest % S;
        rest /= S;
        const int g = rest % gn, t = rest / gn;
        const uint32_t* tr = (const uint32_t*) tab[(tr_slot0 + t) * C::E + gexp[g]];
        const int64_t r0 = (int64_t) t * P + goff[g];
        qexl3::gemv_block<BITS, TPW, PF, WARPS, 1, C::CBK, false>(xh + r0 * K, tr, part + ((int64_t) s * T * P + r0) * N, gcnt[g], K, N, nblk,
                                                           s * kts, kts);
    }
}

// EXPERIMENT (timing only, wrong values): gemv_block with one 16-byte load per lane covering 4 n-tiles (512 contiguous
// bytes per warp instruction); lane l uses component c of its load as "its" word of tile c. Same decode and MMA work.
template <int BITS, int PF, int WARPS, int CBK>
__device__ __forceinline__ void gemv_block_wide(const half* __restrict__ xh, const uint32_t* __restrict__ tr,
                                                float* __restrict__ out, int M, int K, int N, int nblk, int kt0, int kts)
{
    constexpr int TPW = 4, NW = 8 * BITS;
    static_assert(NW == 32, "4-bit only");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int Nt = N / 16, Kt = K / 16;
    const int nt0 = nblk * (WARPS * TPW) + warp * TPW;
    const int kt1 = min(Kt, kt0 + kts);
    const int ar = lane >> 2, ak = (lane & 3) * 2;
    float c[TPW][2][4] = {};
    uint4 pf[PF];
    auto load = [&](int kt) { return __ldcs((const uint4*) (tr + ((int64_t) kt * Nt + nt0) * NW) + lane); };
    #pragma unroll
    for (int d = 0; d < PF; ++d) if (kt0 + d < kt1) pf[d] = load(kt0 + d);
    for (int kb = kt0; kb < kt1; kb += PF)
    {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            int kt = kb + d;
            if (kt >= kt1) break;
            const uint4 cur = pf[d];
            if (kt + PF < kt1) pf[d] = load(kt + PF);
            uint32_t a[4];
            const half* x0 = xh + (int64_t) ar * K + kt * 16 + ak;
            const half* x1 = x0 + (int64_t) 8 * K;
            a[0] = ar < M ? *(const uint32_t*) x0 : 0u;
            a[1] = ar + 8 < M ? *(const uint32_t*) x1 : 0u;
            a[2] = ar < M ? *(const uint32_t*) (x0 + 8) : 0u;
            a[3] = ar + 8 < M ? *(const uint32_t*) (x1 + 8) : 0u;
            const uint32_t w4[4] = {cur.x, cur.y, cur.z, cur.w};
            #pragma unroll
            for (int t = 0; t < TPW; ++t)
            {
                uint32_t b[4];
                qexl3::decode_tile<BITS, CBK>(w4[t], 0u, lane, b);
                qexl3::mma16816(c[t][0], a, b[0], b[1]);
                qexl3::mma16816(c[t][1], a, b[2], b[3]);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < TPW; ++t)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int n = (nt0 + t) * 16 + h * 8 + (lane & 3) * 2;
            if (ar < M) { out[(int64_t) ar * N + n] = c[t][h][0]; out[(int64_t) ar * N + n + 1] = c[t][h][1]; }
            if (ar + 8 < M) { out[(int64_t) (ar + 8) * N + n] = c[t][h][2]; out[(int64_t) (ar + 8) * N + n + 1] = c[t][h][3]; }
        }
}
template <class C, int BITS, int PF, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void gemv_wide_kernel(const half* __restrict__ xh, const uint64_t* __restrict__ tab,
                                                               int tr_slot0, const int* __restrict__ G, const int* __restrict__ gexp,
                                                               const int* __restrict__ goff, const int* __restrict__ gcnt, int T, int P,
                                                               int K, int N, int S, int kts, float* __restrict__ part)
{
    const int nb = N / (16 * WARPS * 4), gn = *G, total = T * gn * S * nb;
    for (int task = blockIdx.x; task < total; task += gridDim.x)
    {
        const int nblk = task % nb;
        int rest = task / nb;
        const int s = rest % S;
        rest /= S;
        const int g = rest % gn, t = rest / gn;
        const uint32_t* tr = (const uint32_t*) tab[(tr_slot0 + t) * C::E + gexp[g]];
        const int64_t r0 = (int64_t) t * P + goff[g];
        gemv_block_wide<BITS, PF, WARPS, C::CBK>(xh + r0 * K, tr, part + ((int64_t) s * T * P + r0) * N, gcnt[g], K, N, nblk, s * kts, kts);
    }
}

// Decode stages 1 (gate/up), mid and 2 (down) in one persistent kernel. Tasks: stage 1 (t, item g, slice s, n-block)
// then stage 2 (item g, slice s, n-block), each exactly a gemv_block as in gemv_persist_kernel. The CTA that completes
// an item's last stage-1 task (counter cnt1[g]) runs mid for the item (mid_row_block, as mid_grouped_kernel) and sets
// rdy[g]; stage-2 tasks of g wait for rdy[g]. Deadlock-free: all CTAs are co-resident (grid <= occupancy) and every CTA
// takes its tasks in increasing order, so it has finished all its stage-1 tasks before it waits on any item.
template <class C, int BITS, int TPW, int PF, int WARPS, bool APF>
__global__ __launch_bounds__(WARPS * 32) void moe_fused_kernel(const half* __restrict__ xh1, half* __restrict__ xh2,
                                                               const uint64_t* __restrict__ tab, const int* __restrict__ G,
                                                               const int* __restrict__ gexp, const int* __restrict__ goff,
                                                               const int* __restrict__ gcnt, int P, int S1, int kts1, int S2, int kts2,
                                                               float* __restrict__ part1, float* __restrict__ part2, int* __restrict__ cnt1,
                                                               int* __restrict__ rdy)
{
    pdl_enter();
    constexpr int H = C::H, I = C::I, NB1 = I / (16 * WARPS * TPW), NB2 = H / (16 * WARPS * TPW);
    const int gn = *G, T1 = 2 * gn * S1 * NB1, total = T1 + gn * S2 * NB2;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ int s_last;
    for (int task = blockIdx.x; task < total; task += gridDim.x)
    {
        if (task < T1)
        {
            const int nblk = task % NB1;
            int rest = task / NB1;
            const int s = rest % S1;
            rest /= S1;
            const int g = rest % gn, t = rest / gn;
            const int64_t r0 = (int64_t) t * P + goff[g];
            qexl3::gemv_block<BITS, TPW, PF, WARPS, 1, C::CBK, APF>(xh1 + r0 * H, (const uint32_t*) tab[t * C::E + gexp[g]],
                                                                    part1 + ((int64_t) s * 2 * P + r0) * I, gcnt[g], H, I, nblk, s * kts1, kts1);
            __syncthreads();
            if (threadIdx.x == 0) { __threadfence(); s_last = atomicAdd(&cnt1[g], 1) == 2 * S1 * NB1 - 1; }
            __syncthreads();
            if (s_last)
            {
                __threadfence();
                const int e = gexp[g], m = gcnt[g], o = goff[g];
                for (int j = warp; j < m * (I / 128); j += WARPS) mid_row_block<C, true>(part1, S1, tab, P, e, o + j / (I / 128), j % (I / 128), lane, xh2);
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0) atomicExch(&rdy[g], 1);
            }
        }
        else
        {
            const int t2 = task - T1, nblk = t2 % NB2, rest = t2 / NB2, s = rest % S2, g = rest / S2;
            if (threadIdx.x == 0) { while (*(volatile int*) &rdy[g] == 0) __nanosleep(32); __threadfence(); }
            __syncthreads();
            const int64_t r0 = goff[g];
            qexl3::gemv_block<BITS, TPW, PF, WARPS, 1, C::CBK, APF>(xh2 + r0 * I, (const uint32_t*) tab[2 * C::E + gexp[g]],
                                                                    part2 + ((int64_t) s * P + r0) * H, gcnt[g], I, H, nblk, s * kts2, kts2);
        }
        __syncthreads();
    }
}
template <class C, int BITS, int TPW, int PF, int WARPS, bool APF>
bool launch_fused(int tpw, int pf, int warps, int apf, int ctas, const half* xh1, half* xh2, const uint64_t* tab, const int* G,
                  const int* gexp, const int* goff, const int* gcnt, int Gmax, int P, int S1, int S2, float* part1, float* part2, int* cnt1,
                  int* rdy, cudaStream_t st)
{
    if (tpw != TPW || pf != PF || warps != WARPS || (apf != 0) != APF) return false;
    auto k = moe_fused_kernel<C, BITS, TPW, PF, WARPS, APF>;
    static int occ = [&] { int n = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, WARPS * 32, 0); return n; }();
    const int per_sm = ctas > 0 && ctas < occ ? ctas : occ;
    const int NB1 = C::I / (16 * WARPS * TPW), NB2 = C::H / (16 * WARPS * TPW);
    const int tasks = 2 * Gmax * S1 * NB1 + Gmax * S2 * NB2;
    const int grid = min(tasks, per_sm * sm_count());
    klaunch<C>(k, grid, WARPS * 32, st, xh1, xh2, tab, G, gexp, goff, gcnt, P, S1, (C::H / 16 + S1 - 1) / S1, S2, (C::I / 16 + S2 - 1) / S2,
               part1, part2, cnt1, rdy);
    return true;
}

// Decode GEMV schedule (MOE_GEMV="tpw,pf,warps,s1,s2,ctas" for tuning: n-tiles per warp, k-tiles prefetched, warps per
// CTA, K slices of stage 1 (gate/up) and stage 2 (down), resident CTAs per SM; 0 CTAs: the non-persistent grid).
using GemvCfg = GemvCfgT;
template <class C> GemvCfg gemv_cfg(int S, int R)
{
    GemvCfg c = C::gemv_default(S, R);
    if (const char* e = getenv(C::GEMV_ENV)) sscanf(e, "%d,%d,%d,%d,%d,%d,%d,%d", &c.tpw, &c.pf, &c.warps, &c.s1, &c.s2, &c.ctas, &c.apf, &c.fused);
    return c;
}
template <class C, int BITS, int TPW, int PF, int WARPS>
bool launch_persist(const GemvCfg& c, const half* xh, const uint64_t* tab, int tr_slot0, const int* G, const int* gexp, const int* goff,
                    const int* gcnt, int Gmax, int T, int P, int K, int N, int S, float* part, cudaStream_t st)
{
    if (c.tpw != TPW && !(c.tpw == 104 && TPW == 4) || c.pf != PF || c.warps != WARPS) return false;
    const int kts = (K / 16 + S - 1) / S, nb = N / (16 * WARPS * TPW);
    const int tasks = T * Gmax * S * nb, grid = c.ctas > 0 ? min(tasks, c.ctas * sm_count()) : tasks;
    if (c.tpw == 104)   // experiment: wide loads (timing only)
    {
        gemv_wide_kernel<C, BITS, PF, WARPS><<<grid, WARPS * 32, 0, st>>>(xh, tab, tr_slot0, G, gexp, goff, gcnt, T, P, K, N, S, kts, part);
        return true;
    }
    klaunch<C>(gemv_persist_kernel<C, BITS, TPW, PF, WARPS>, grid, WARPS * 32, st, xh, tab, tr_slot0, G, gexp, goff, gcnt, T, P, K, N, S, kts, part);
    return true;
}
template <class C, int BITS>
void launch_decode_products(const GemvCfg& c, const half* xh, const uint64_t* tab, int tr_slot0, const int* G, const int* gexp,
                            const int* goff, const int* gcnt, int Gmax, int T, int P, int K, int N, int S, float* part, cudaStream_t st)
{
#define LP(a, b, w) launch_persist<C, BITS, a, b, w>(c, xh, tab, tr_slot0, G, gexp, goff, gcnt, Gmax, T, P, K, N, S, part, st)
    if (LP(2, 4, 4) || LP(4, 4, 4) || LP(8, 4, 4) || LP(2, 8, 4) || LP(4, 8, 4) || LP(8, 2, 4) || LP(4, 4, 8) || LP(8, 4, 8) || LP(4, 2, 8)
        || LP(2, 4, 8) || LP(8, 8, 4) || LP(4, 6, 4)) return;
#undef LP
    fprintf(stderr, "[moe] no GEMV instance tpw %d pf %d warps %d\n", c.tpw, c.pf, c.warps);
    abort();
}

// prefill: grid (N/128, T * Gmax) x 128 threads. Item rows (<= 128) of A = xh staged through shared memory in
// k-chunks of 64 (cp.async, double buffered); warp w decodes its 2 n16 tiles of each k16 step once and runs them
// against all 8 row tiles (ldmatrix A fragments). part[t][row][N] (no K split). Per output element the same mma
// sequence as the GEMV (k ascending), so results are bitwise equal to gemv_grouped_kernel with S = 1.
constexpr int GM = 128, KC = 64, SA = KC + 8;
__device__ __forceinline__ void cp16(void* s, const void* g)
{
    unsigned a = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(g));
}
// FIN (svh_slot >= 0): the finish (had128 over the block's 128 columns * svh) runs in the epilogue and y gets the
// finished rows (the same fwht128 lane layout as finish_grouped_kernel, so bitwise equal).
template <class C, int BITS, int WARPS = 4, typename OT = float>
__global__ __launch_bounds__(WARPS * 32) void gemm_grouped_kernel(const half* __restrict__ xh, const uint64_t* __restrict__ tab,
                                                           int tr_slot0, const int* __restrict__ G, const int* __restrict__ gexp,
                                                           const int* __restrict__ goff, const int* __restrict__ gcnt, int Gmax,
                                                           int P, int K, int N, OT* __restrict__ part, int svh_slot)
{
    constexpr int NW = 8 * BITS, TW = (KC / 16) * 8 * NW;   // trellis words of a k-chunk: 4 k-tiles x 8 n16 tiles
    __shared__ __align__(16) half sa[2][GM * SA];
    __shared__ __align__(16) uint32_t sw[2][TW];
    const int tg = blockIdx.y, t = tg / Gmax, g = tg % Gmax;
    if (g >= *G) return;
    const int M = gcnt[g];
    const int64_t r0 = (int64_t) t * P + goff[g];
    const half* A = xh + r0 * K;
    const uint32_t* tr = (const uint32_t*) tab[(tr_slot0 + t) * C::E + gexp[g]];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int TPW = 8 / WARPS, NT = WARPS * 32;   // n16 tiles per warp, threads
    const int Nt = N / 16, Kt = K / 16, nt0 = blockIdx.x * 8 + warp * TPW;
    const int MT = (M + 15) / 16;
    auto load = [&](int kc, int buf)
    {
        // rows < M, 8 x 16 B per row
        for (int i = tid; i < M * (KC / 8); i += NT)
        {
            int r = i / (KC / 8), c = i % (KC / 8);
            cp16(&sa[buf][r * SA + c * 8], A + (int64_t) r * K + kc * KC + c * 8);
        }
        // trellis: per k-tile the block's 8 n16 tiles are contiguous (8 * NW words)
        for (int i = tid; i < TW / 4; i += NT)
        {
            int kk = i / (8 * NW / 4), j = i % (8 * NW / 4);
            cp16(&sw[buf][kk * 8 * NW + j * 4], tr + ((int64_t) (kc * (KC / 16) + kk) * Nt + blockIdx.x * 8) * NW + j * 4);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float c[8][TPW][2][4] = {};
    const int nkc = K / KC;
    load(0, 0);
    for (int kc = 0; kc < nkc; ++kc)
    {
        if (kc + 1 < nkc) { load(kc + 1, (kc + 1) & 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const half* S = sa[kc & 1];
        const uint32_t* W = sw[kc & 1];
        // decode the chunk's B fragments first (independent of the mma chain: more work in flight)
        uint32_t bb[KC / 16][TPW][4];
        #pragma unroll
        for (int kk = 0; kk < KC / 16; ++kk)
            #pragma unroll
            for (int u = 0; u < TPW; ++u)
            {
                const uint32_t* p = W + (kk * 8 + warp * TPW + u) * NW;
                qexl3::decode_tile<BITS, C::CBK>(p[lane], NW > 32 && lane + 32 < NW ? p[32 + lane] : 0u, lane, bb[kk][u]);
            }
        #pragma unroll
        for (int kk = 0; kk < KC / 16; ++kk)
        {
            uint32_t (&b)[TPW][4] = bb[kk];
            #pragma unroll
            for (int mt = 0; mt < 8; ++mt)
            {
                if (mt >= MT) break;
                uint32_t a[4];
                const half* ap = S + (mt * 16 + (lane & 15)) * SA + kk * 16 + (lane >> 4) * 8;
                unsigned sp = (unsigned) __cvta_generic_to_shared(ap);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sp));
                #pragma unroll
                for (int u = 0; u < TPW; ++u)
                {
                    qexl3::mma16816(c[mt][u][0], a, b[u][0], b[u][1]);
                    qexl3::mma16816(c[mt][u][1], a, b[u][2], b[u][3]);
                }
            }
        }
        __syncthreads();
    }
    const int ar = lane >> 2;
    OT* out = part + r0 * N;
    if (svh_slot >= 0)
    {
        // epilogue per row tile: fragments -> shared [16][128] fp32 -> warp per row: had128 * svh
        float* sf = (float*) sa[0];
        const half* svh = (const half*) tab[(svh_slot + t) * C::E + gexp[g]];
        #pragma unroll
        for (int mt = 0; mt < 8; ++mt)
        {
            if (mt >= MT) break;
            #pragma unroll
            for (int u = 0; u < TPW; ++u)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    const int col = (warp * TPW + u) * 16 + h * 8 + (lane & 3) * 2;
                    sf[ar * 128 + col] = c[mt][u][h][0]; sf[ar * 128 + col + 1] = c[mt][u][h][1];
                    sf[(ar + 8) * 128 + col] = c[mt][u][h][2]; sf[(ar + 8) * 128 + col + 1] = c[mt][u][h][3];
                }
            __syncthreads();
            for (int rr = warp; rr < 16; rr += WARPS)
            {
                const int row = mt * 16 + rr;
                if (row >= M) break;
                float v[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = sf[rr * 128 + lane * 4 + i];
                qexl3::fwht128(v, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    const int n = blockIdx.x * 128 + lane * 4 + i;
                    out[(int64_t) row * N + n] = (OT) (v[i] * __half2float(svh[n]));
                }
            }
            __syncthreads();
        }
        return;
    }
    #pragma unroll
    for (int mt = 0; mt < 8; ++mt)
    {
        if (mt >= MT) break;
        #pragma unroll
        for (int u = 0; u < TPW; ++u)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int rr = mt * 16 + ar;
                const int n = (nt0 + u) * 16 + h * 8 + (lane & 3) * 2;
                if (rr < M) { out[(int64_t) rr * N + n] = (OT) c[mt][u][h][0]; out[(int64_t) rr * N + n + 1] = (OT) c[mt][u][h][1]; }
                if (rr + 8 < M) { out[(int64_t) (rr + 8) * N + n] = (OT) c[mt][u][h][2]; out[(int64_t) (rr + 8) * N + n + 1] = (OT) c[mt][u][h][3]; }
            }
    }
}

// Prefill gate + up with the input transform fused (no xh1 in memory). grid (I / 128, 2 * Gmax) x 256 threads:
// block = (n-block of 128 columns, half of an item = <= 64 rows). Per k-chunk of 128 (one Hadamard block) the
// block's rows are transformed into shared memory for gate and up (had128(x * suh) -> fp16, exactly as
// had_grouped_kernel), then warp w multiplies its n16 tile of both matrices against the 4 row tiles. 35 KB of
// shared memory: two blocks per SM, so one block's transform overlaps the other's MMAs. Epilogue per row tile:
// had128 * svh for gate and up, silu(gate) * up, * suh_down, had128 -> xh2 (fp16) at the block's 128 columns
// (= one Hadamard block of the down input). Same operations and fwht128 lane layout as the unfused stages
// (had_grouped, products, finish, act, had_grouped), so bitwise equal.
constexpr int H2 = 64, SA2 = 128 + 8;
template <class C, int BITS>
__global__ __launch_bounds__(256, 2) void gux_kernel(const typename C::XT* __restrict__ x, int64_t ldx, const uint64_t* __restrict__ tab,
                                                     const int* __restrict__ G, const int* __restrict__ gexp, const int* __restrict__ goff,
                                                     const int* __restrict__ gcnt, const int* __restrict__ prow, half* __restrict__ xh2)
{
    constexpr int NW = 8 * BITS, K = C::H, N = C::I, Nt = N / 16;
    __shared__ __align__(16) half sg[H2 * SA2], su[H2 * SA2];
    const int g = blockIdx.y >> 1, hf = blockIdx.y & 1;
    if (g >= *G) return;
    const int M = min(H2, gcnt[g] - hf * H2);
    if (M <= 0) return;
    const int e = gexp[g], MT = (M + 15) / 16;
    const int64_t r0 = goff[g] + hf * H2;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nt = blockIdx.x * 8 + warp;
    const half* suh_g = (const half*) tab[3 * C::E + e];
    const half* suh_u = (const half*) tab[4 * C::E + e];
    const uint32_t* tr_g = (const uint32_t*) tab[0 * C::E + e];
    const uint32_t* tr_u = (const uint32_t*) tab[1 * C::E + e];
    float cg[4][2][4] = {}, cu[4][2][4] = {};
    uint32_t wg[2], wu[2];
    auto ldw = [&](int kt)
    {
        const uint32_t* pg = tr_g + ((int64_t) kt * Nt + nt) * NW;
        const uint32_t* pu = tr_u + ((int64_t) kt * Nt + nt) * NW;
        wg[0] = __ldcs(pg + lane); wg[1] = NW > 32 && lane + 32 < NW ? __ldcs(pg + 32 + lane) : 0u;
        wu[0] = __ldcs(pu + lane); wu[1] = NW > 32 && lane + 32 < NW ? __ldcs(pu + 32 + lane) : 0u;
    };
    // transform layout: 8 lanes per row (part p = lane % 8 holds elements 16p..16p+15), 4 rows per warp pass, the
    // warp's rows r = warp + 8 (4 pass + lane / 8) < M; gate then up from the same loaded x values. fwht128_16: the
    // same butterflies in the same order as fwht128 (bitwise equal) with 12 shuffles per row instead of 20.
    const int part = lane & 7;
    int64_t xoff[2];
    #pragma unroll
    for (int ps = 0; ps < 2; ++ps) { const int r = warp + 8 * (4 * ps + (lane >> 3)); xoff[ps] = r < M ? (int64_t) prow[r0 + r] * ldx : -1; }
    ldw(0);
    for (int kb = 0; kb < K / 128; ++kb)
    {
        __syncthreads();
        const int k0 = kb * 128 + part * 16;
        #pragma unroll 1
        for (int ps = 0; ps < 2; ++ps)
        {
            if (warp + 32 * ps >= M) break;   // no row of this pass in the warp (warp-uniform)
            const bool live = xoff[ps] >= 0;
            const int r = warp + 8 * (4 * ps + (lane >> 3));
            float xa[16];
            ld16(x, live ? xoff[ps] + k0 : -1, xa);
            #pragma unroll
            for (int mtx = 0; mtx < 2; ++mtx)
            {
                const half* suh = mtx ? suh_u : suh_g;
                half* dst = mtx ? su : sg;
                const uint4 h0 = *(const uint4*) (suh + k0), h1 = *(const uint4*) (suh + k0 + 8);
                const half* hs0 = (const half*) &h0;
                const half* hs1 = (const half*) &h1;
                float v[16];
                #pragma unroll
                for (int j = 0; j < 8; ++j) { v[j] = xa[j] * __half2float(hs0[j]); v[8 + j] = xa[8 + j] * __half2float(hs1[j]); }
                qexl3::fwht128_16(v, part);
                if (live)
                {
                    uint4 o[2];
                    half2* ho = (half2*) o;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) ho[j] = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
                    *(uint4*) (dst + r * SA2 + part * 16) = o[0];
                    *(uint4*) (dst + r * SA2 + part * 16 + 8) = o[1];
                }
            }
        }
        __syncthreads();
        #pragma unroll 2
        for (int kk = 0; kk < 8; ++kk)
        {
            const int kt = kb * 8 + kk;
            const uint32_t cg0 = wg[0], cg1 = wg[1], cu0 = wu[0], cu1 = wu[1];
            if (kt + 1 < K / 16) ldw(kt + 1);
            uint32_t bg[4], bu[4];
            qexl3::decode_tile<BITS, C::CBK>(cg0, cg1, lane, bg);
            qexl3::decode_tile<BITS, C::CBK>(cu0, cu1, lane, bu);
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
            {
                if (mt >= MT) break;
                uint32_t a[4];
                const int ro = (mt * 16 + (lane & 15)) * SA2 + kk * 16 + (lane >> 4) * 8;
                unsigned sp = (unsigned) __cvta_generic_to_shared(sg + ro);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sp));
                qexl3::mma16816(cg[mt][0], a, bg[0], bg[1]);
                qexl3::mma16816(cg[mt][1], a, bg[2], bg[3]);
                sp = (unsigned) __cvta_generic_to_shared(su + ro);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sp));
                qexl3::mma16816(cu[mt][0], a, bu[0], bu[1]);
                qexl3::mma16816(cu[mt][1], a, bu[2], bu[3]);
            }
        }
    }
    __syncthreads();
    float* fg = (float*) sg;
    float* fu = (float*) su;
    const half* svh_g = (const half*) tab[6 * C::E + e];
    const half* svh_u = (const half*) tab[7 * C::E + e];
    const half* suh_d = (const half*) tab[5 * C::E + e];
    const int ar = lane >> 2;
    for (int mt = 0; mt < MT; ++mt)
    {
        #pragma unroll
        for (int m2 = 0; m2 < 4; ++m2)
            if (m2 == mt)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    const int col = warp * 16 + h * 8 + (lane & 3) * 2;
                    fg[ar * 128 + col] = cg[m2][h][0]; fg[ar * 128 + col + 1] = cg[m2][h][1];
                    fg[(ar + 8) * 128 + col] = cg[m2][h][2]; fg[(ar + 8) * 128 + col + 1] = cg[m2][h][3];
                    fu[ar * 128 + col] = cu[m2][h][0]; fu[ar * 128 + col + 1] = cu[m2][h][1];
                    fu[(ar + 8) * 128 + col] = cu[m2][h][2]; fu[(ar + 8) * 128 + col + 1] = cu[m2][h][3];
                }
        __syncthreads();
        for (int rr = warp; rr < 16; rr += 8)
        {
            const int row = mt * 16 + rr;
            if (row >= M) break;
            float vg[4], vu[4], v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { vg[i] = fg[rr * 128 + lane * 4 + i]; vu[i] = fu[rr * 128 + lane * 4 + i]; }
            qexl3::fwht128(vg, lane);
            qexl3::fwht128(vu, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = blockIdx.x * 128 + lane * 4 + i;
                float a = vg[i] * __half2float(svh_g[n]), u = vu[i] * __half2float(svh_u[n]);
                float act = C::act(a, u);
                v[i] = act * __half2float(suh_d[n]);
            }
            qexl3::fwht128(v, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) xh2[(r0 + row) * N + blockIdx.x * 128 + lane * 4 + i] = __float2half_rn(v[i]);
        }
        __syncthreads();
    }
}

template <class C, int BITS>
__global__ __launch_bounds__(256, 2) void gux_shared_kernel(const half* __restrict__ xh1, const uint64_t* __restrict__ tab,
                                                     const int* __restrict__ G, const int* __restrict__ gexp, const int* __restrict__ goff,
                                                     const int* __restrict__ gcnt, const int* __restrict__ prow, half* __restrict__ xh2)
{
    constexpr int NW = 8 * BITS, K = C::H, N = C::I, Nt = N / 16;
    __shared__ __align__(16) half sa[2][H2 * SA2];
    const int g = blockIdx.y >> 1, hf = blockIdx.y & 1;
    if (g >= *G) return;
    const int M = min(H2, gcnt[g] - hf * H2);
    if (M <= 0) return;
    const int e = gexp[g], MT = (M + 15) / 16;
    const int64_t r0 = goff[g] + hf * H2;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nt = blockIdx.x * 8 + warp;
    const uint32_t* tr_g = (const uint32_t*) tab[0 * C::E + e];
    const uint32_t* tr_u = (const uint32_t*) tab[1 * C::E + e];
    float cg[4][2][4] = {}, cu[4][2][4] = {};
    uint32_t wg[2], wu[2];
    auto ldw = [&](int kt)
    {
        const uint32_t* pg = tr_g + ((int64_t) kt * Nt + nt) * NW;
        const uint32_t* pu = tr_u + ((int64_t) kt * Nt + nt) * NW;
        wg[0] = __ldcs(pg + lane); wg[1] = NW > 32 && lane + 32 < NW ? __ldcs(pg + 32 + lane) : 0u;
        wu[0] = __ldcs(pu + lane); wu[1] = NW > 32 && lane + 32 < NW ? __ldcs(pu + 32 + lane) : 0u;
    };
    // A = the per-token transformed input (shared-input layers: one had128(x * suh) per token for every expert),
    // gathered by token in k-blocks of 128 (one Hadamard block) with cp.async, double buffered.
    auto load_a = [&](int kb, int buf)
    {
        for (int i = threadIdx.x; i < M * 16; i += 256)
        {
            const int r = i >> 4, c = i & 15;
            cp16(&sa[buf][r * SA2 + c * 8], xh1 + (int64_t) prow[r0 + r] * K + kb * 128 + c * 8);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    ldw(0);
    load_a(0, 0);
    for (int kb = 0; kb < K / 128; ++kb)
    {
        __syncthreads();   // every warp is done with k-block kb - 1, whose buffer the next load overwrites
        if (kb + 1 < K / 128) { load_a(kb + 1, (kb + 1) & 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const half* sg = sa[kb & 1];
        #pragma unroll 2
        for (int kk = 0; kk < 8; ++kk)
        {
            const int kt = kb * 8 + kk;
            const uint32_t cg0 = wg[0], cg1 = wg[1], cu0 = wu[0], cu1 = wu[1];
            if (kt + 1 < K / 16) ldw(kt + 1);
            uint32_t bg[4], bu[4];
            qexl3::decode_tile<BITS, C::CBK>(cg0, cg1, lane, bg);
            qexl3::decode_tile<BITS, C::CBK>(cu0, cu1, lane, bu);
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
            {
                if (mt >= MT) break;
                uint32_t a[4];
                const int ro = (mt * 16 + (lane & 15)) * SA2 + kk * 16 + (lane >> 4) * 8;
                unsigned sp = (unsigned) __cvta_generic_to_shared(sg + ro);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sp));
                qexl3::mma16816(cg[mt][0], a, bg[0], bg[1]);
                qexl3::mma16816(cg[mt][1], a, bg[2], bg[3]);
                qexl3::mma16816(cu[mt][0], a, bu[0], bu[1]);
                qexl3::mma16816(cu[mt][1], a, bu[2], bu[3]);
            }
        }
    }
    __syncthreads();
    float* fg = (float*) sa[0];
    float* fu = (float*) sa[1];
    const half* svh_g = (const half*) tab[6 * C::E + e];
    const half* svh_u = (const half*) tab[7 * C::E + e];
    const half* suh_d = (const half*) tab[5 * C::E + e];
    const int ar = lane >> 2;
    for (int mt = 0; mt < MT; ++mt)
    {
        #pragma unroll
        for (int m2 = 0; m2 < 4; ++m2)
            if (m2 == mt)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    const int col = warp * 16 + h * 8 + (lane & 3) * 2;
                    fg[ar * 128 + col] = cg[m2][h][0]; fg[ar * 128 + col + 1] = cg[m2][h][1];
                    fg[(ar + 8) * 128 + col] = cg[m2][h][2]; fg[(ar + 8) * 128 + col + 1] = cg[m2][h][3];
                    fu[ar * 128 + col] = cu[m2][h][0]; fu[ar * 128 + col + 1] = cu[m2][h][1];
                    fu[(ar + 8) * 128 + col] = cu[m2][h][2]; fu[(ar + 8) * 128 + col + 1] = cu[m2][h][3];
                }
        __syncthreads();
        for (int rr = warp; rr < 16; rr += 8)
        {
            const int row = mt * 16 + rr;
            if (row >= M) break;
            float vg[4], vu[4], v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { vg[i] = fg[rr * 128 + lane * 4 + i]; vu[i] = fu[rr * 128 + lane * 4 + i]; }
            qexl3::fwht128(vg, lane);
            qexl3::fwht128(vu, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = blockIdx.x * 128 + lane * 4 + i;
                float a = vg[i] * __half2float(svh_g[n]), u = vu[i] * __half2float(svh_u[n]);
                float act = C::act(a, u);
                v[i] = act * __half2float(suh_d[n]);
            }
            qexl3::fwht128(v, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) xh2[(r0 + row) * N + blockIdx.x * 128 + lane * 4 + i] = __float2half_rn(v[i]);
        }
        __syncthreads();
    }
}

// y[t][row][N] = had128(sum_s part[s][t][row]) * svh_t[e]; warp per (t, row, nb)
template <class C>
__global__ void finish_grouped_kernel(const float* __restrict__ part, int S, const uint64_t* __restrict__ tab, int svh_slot0,
                                      int T, int P, const int* __restrict__ pexp, int N, float* __restrict__ y)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int nb = N / 128;
    int b = warp % nb, rest = warp / nb;
    int row = rest % P, t = rest / P;
    if (t >= T) return;
    const half* svh = (const half*) tab[(svh_slot0 + t) * C::E + pexp[row]];
    float v[4] = {};
    for (int s = 0; s < S; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] += part[(((int64_t) s * T + t) * P + row) * N + b * 128 + lane * 4 + i];
    qexl3::fwht128(v, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int n = b * 128 + lane * 4 + i;
        y[((int64_t) t * P + row) * N + n] = v[i] * __half2float(svh[n]);
    }
}

// Decode / verify (one counting block, R * TOPK <= CB * TOPK pairs): count, scan, fill and pexp in one block of E threads
// (thread = expert); the same outputs as count_kernel / scan_kernel / fill_kernel / pexp_kernel with nb = 1.
template <class C>
__global__ __launch_bounds__(512) void prep_small_kernel(const typename C::IT* __restrict__ idx, int R, int GR, int* __restrict__ eoff,
                                                         int* __restrict__ G, int* __restrict__ gexp, int* __restrict__ goff,
                                                         int* __restrict__ gcnt, int* __restrict__ pairmap, int* __restrict__ prow,
                                                         int* __restrict__ pexp, int* __restrict__ zero = nullptr, int nzero = 0)
{
    pdl_enter();
    for (int i = threadIdx.x; i < nzero; i += blockDim.x) zero[i] = 0;
    __shared__ int sidx[CB * C::TOPK];
    __shared__ int s_rows[C::E], s_items[C::E];
    const int e = threadIdx.x, n = R * C::TOPK;
    for (int i = e; i < n; i += C::E) sidx[i] = idx[i];
    __syncthreads();
    int tot = 0;
    for (int i = 0; i < n; ++i) tot += sidx[i] == e;
    // exclusive prefix sums of rows and items over the experts (block scan; the same integers as a serial scan)
    __shared__ int w_rows[32], w_items[32];
    const int lane = e & 31, wp = e >> 5;
    int ir = tot, ii = (tot + GR - 1) / GR;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        const int a = __shfl_up_sync(0xffffffffu, ir, o), b = __shfl_up_sync(0xffffffffu, ii, o);
        if (lane >= o) { ir += a; ii += b; }
    }
    if (lane == 31) { w_rows[wp] = ir; w_items[wp] = ii; }
    __syncthreads();
    if (wp == 0)
    {
        constexpr int NWP = (C::E + 31) / 32;
        int a = lane < NWP ? w_rows[lane] : 0, b = lane < NWP ? w_items[lane] : 0;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            const int c = __shfl_up_sync(0xffffffffu, a, o), d = __shfl_up_sync(0xffffffffu, b, o);
            if (lane >= o) { a += c; b += d; }
        }
        if (lane < NWP) { w_rows[lane] = a; w_items[lane] = b; }
        if (lane == NWP - 1) *G = b;
    }
    __syncthreads();
    s_rows[e] = ir - tot + (wp ? w_rows[wp - 1] : 0);
    s_items[e] = ii - (tot + GR - 1) / GR + (wp ? w_items[wp - 1] : 0);
    eoff[e] = s_rows[e];
    for (int i = 0, it = s_items[e]; i * GR < tot; ++i, ++it)
    {
        gexp[it] = e;
        goff[it] = s_rows[e] + i * GR;
        gcnt[it] = min(GR, tot - i * GR);
    }
    int pos = s_rows[e];
    for (int i = 0; i < n; ++i)
        if (sidx[i] == e) { pairmap[i] = pos; prow[pos] = i / C::TOPK; pexp[pos] = e; ++pos; }
}

// Decode: finish of gate and up (T = 2), silu(gate) * up and the down input transform in one pass; warp per
// (compact row, 128-block of the I columns); the same operations as finish_grouped_kernel, act_kernel and
// had_grouped_kernel (stage 2) in sequence (bitwise equal). xh2 [P, I] fp16.
template <class C>
__global__ void mid_grouped_kernel(const float* __restrict__ part, int S, const uint64_t* __restrict__ tab, int P,
                                   const int* __restrict__ pexp, half* __restrict__ xh2)
{
    pdl_enter();
    constexpr int N = C::I;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int b = warp % (N / 128), row = warp / (N / 128);
    if (row >= P) return;
    mid_row_block<C, false>(part, S, tab, P, pexp[row], row, b, lane, xh2);
}
template <class C, bool CG>
__device__ __forceinline__ void mid_row_block(const float* __restrict__ part, int S, const uint64_t* __restrict__ tab, int P, int e,
                                              int row, int b, int lane, half* __restrict__ xh2)
{
    constexpr int N = C::I;
    float g[2][4];
    #pragma unroll
    for (int t = 0; t < 2; ++t)
    {
        const half* svh = (const half*) tab[(6 + t) * C::E + e];
        float v[4] = {};
        for (int s = 0; s < S; ++s)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const float* q = part + (((int64_t) s * 2 + t) * P + row) * N + b * 128 + lane * 4 + i;
                v[i] += CG ? __ldcg(q) : *q;
            }
        qexl3::fwht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) g[t][i] = v[i] * __half2float(svh[b * 128 + lane * 4 + i]);
    }
    const half* suh = (const half*) tab[5 * C::E + e];
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const float a = g[0][i], u = g[1][i];
        const float act = C::act(a, u);
        v[i] = act * __half2float(suh[b * 128 + lane * 4 + i]);
    }
    qexl3::fwht128(v, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) xh2[(int64_t) row * N + b * 128 + lane * 4 + i] = __float2half_rn(v[i]);
}

// act[row][n] = silu(gate) * up (gate rows t = 0, up rows t = 1 of y)
template <class C>
__global__ void act_kernel(const float* __restrict__ y, int P, int N, float* __restrict__ act)
{
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) P * N) return;
    float a = y[i], u = y[(int64_t) P * N + i];
    act[i] = C::act(a, u);
}

// out[r][n] = add[r][n] + sum_k w[r][k] * yd[pairmap[r][k]][n], k in slot order
__device__ __forceinline__ float4 ld4(const float* p) { return *(const float4*) p; }
__device__ __forceinline__ float4 ld4(const half* p)
{
    const uint2 u = *(const uint2*) p;
    const float2 a = __half22float2(*(const __half2*) &u.x), b = __half22float2(*(const __half2*) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
template <class C, typename YT>
__global__ void combine_kernel(const YT* __restrict__ yd, const int* __restrict__ pairmap, const typename C::WT* __restrict__ w,
                               const float* __restrict__ add, float* __restrict__ out, int N)
{
    __shared__ int pm[C::TOPK];
    __shared__ float pw[C::TOPK];
    int r = blockIdx.x;
    if (threadIdx.x < C::TOPK) { pm[threadIdx.x] = pairmap[r * C::TOPK + threadIdx.x]; pw[threadIdx.x] = tof(w[r * C::TOPK + threadIdx.x]); }
    __syncthreads();
    for (int n = threadIdx.x * 4; n < N; n += blockDim.x * 4)
    {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int k = 0; k < C::TOPK; ++k)
        {
            float4 v = ld4(yd + (int64_t) pm[k] * N + n);
            acc.x += pw[k] * v.x; acc.y += pw[k] * v.y; acc.z += pw[k] * v.z; acc.w += pw[k] * v.w;
        }
        float4 a = add ? *(const float4*) (add + (int64_t) r * N + n) : make_float4(0.f, 0.f, 0.f, 0.f);
        *(float4*) (out + (int64_t) r * N + n) = make_float4(a.x + acc.x, a.y + acc.y, a.z + acc.z, a.w + acc.w);
    }
}

// Prefill gate + up for shared-input layers (one transformed row per token, xh1 [R, H]): a whole item (<= 128 rows)
// per CTA, A rows gathered by token and both matrices' trellis words staged by cp.async in k-chunks of KC (double
// buffered); 16 warps: warps 0-7 the gate n16 tile w of the CTA's 128 columns, warps 8-15 the up tile; every decoded
// tile serves the 8 row tiles. Epilogue per row tile: had128 * svh for gate and up, act, * suh_down, had128 -> xh2
// (the same operations as gux_kernel, bitwise equal).
template <class C, int BITS, int STAGES>
__global__ __launch_bounds__(512) void gu_gemm_kernel(const half* __restrict__ xh1, const uint64_t* __restrict__ tab,
                                                      const int* __restrict__ G, const int* __restrict__ gexp, const int* __restrict__ goff,
                                                      const int* __restrict__ gcnt, const int* __restrict__ prow, half* __restrict__ xh2)
{
    constexpr int NW = 8 * BITS, K = C::H, N = C::I, Nt = N / 16, TW = (KC / 16) * 8 * NW;
    extern __shared__ __align__(16) unsigned char gu_smem[];
    half* sa = (half*) gu_smem;                                           // [STAGES][GM * SA]
    uint32_t* sw = (uint32_t*) (gu_smem + STAGES * GM * SA * sizeof(half));   // [STAGES][gate TW | up TW]
    const int g = blockIdx.y;
    if (g >= *G) return;
    const int M = gcnt[g], e = gexp[g], MT = (M + 15) / 16;
    const int64_t r0 = goff[g];
    const uint32_t* trg = (const uint32_t*) tab[0 * C::E + e];
    const uint32_t* tru = (const uint32_t*) tab[1 * C::E + e];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, mtx = warp >> 3, wn = warp & 7;
    auto load = [&](int kc, int buf)
    {
        half* A = sa + buf * GM * SA;   // buf < STAGES
        for (int i = tid; i < M * (KC / 8); i += 512)
        {
            const int r = i / (KC / 8), c = i % (KC / 8);
            cp16(&A[r * SA + c * 8], xh1 + (int64_t) prow[r0 + r] * K + kc * KC + c * 8);
        }
        uint32_t* W = sw + buf * 2 * TW;
        for (int i = tid; i < 2 * TW / 4; i += 512)
        {
            const int m = i / (TW / 4), j = i % (TW / 4), kk = j / (8 * NW / 4), jj = j % (8 * NW / 4);
            cp16(&W[m * TW + kk * 8 * NW + jj * 4], (m ? tru : trg) + ((int64_t) (kc * (KC / 16) + kk) * Nt + blockIdx.x * 8) * NW + jj * 4);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float c[8][2][4] = {};
    constexpr int nkc = K / KC;
    // STAGES-deep cp.async pipeline: chunk kc + STAGES - 1 is issued into the buffer of chunk kc - 1, which every warp
    // has finished (the barrier at the top of iteration kc)
    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) { if (i < nkc) load(i, i); else asm volatile("cp.async.commit_group;\n" ::); }
    for (int kc = 0; kc < nkc; ++kc)
    {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();
        if (kc + STAGES - 1 < nkc) load(kc + STAGES - 1, (kc + STAGES - 1) % STAGES); else asm volatile("cp.async.commit_group;\n" ::);
        const half* S = sa + (kc % STAGES) * GM * SA;
        const uint32_t* W = sw + (kc % STAGES) * 2 * TW + mtx * TW;
        uint32_t bb[KC / 16][4];
        #pragma unroll
        for (int kk = 0; kk < KC / 16; ++kk)
        {
            const uint32_t* p = W + (kk * 8 + wn) * NW;
            qexl3::decode_tile<BITS, C::CBK>(p[lane], NW > 32 && lane + 32 < NW ? p[32 + lane] : 0u, lane, bb[kk]);
        }
        #pragma unroll
        for (int kk = 0; kk < KC / 16; ++kk)
            #pragma unroll
            for (int mt = 0; mt < 8; ++mt)
            {
                if (mt >= MT) break;
                uint32_t a[4];
                const half* ap = S + (mt * 16 + (lane & 15)) * SA + kk * 16 + (lane >> 4) * 8;
                unsigned sp = (unsigned) __cvta_generic_to_shared(ap);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sp));
                qexl3::mma16816(c[mt][0], a, bb[kk][0], bb[kk][1]);
                qexl3::mma16816(c[mt][1], a, bb[kk][2], bb[kk][3]);
            }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    // epilogue per row tile: gate (fs[0]) and up (fs[1]) fragments -> shared [16][128] fp32 -> warp per row
    float* fs = (float*) gu_smem;
    const half* svh_g = (const half*) tab[6 * C::E + e];
    const half* svh_u = (const half*) tab[7 * C::E + e];
    const half* suh_d = (const half*) tab[5 * C::E + e];
    const int ar = lane >> 2;
    for (int mt = 0; mt < MT; ++mt)
    {
        #pragma unroll
        for (int m2 = 0; m2 < 8; ++m2)
            if (m2 == mt)
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    float* f = fs + mtx * 16 * 128;
                    const int col = wn * 16 + h * 8 + (lane & 3) * 2;
                    f[ar * 128 + col] = c[m2][h][0]; f[ar * 128 + col + 1] = c[m2][h][1];
                    f[(ar + 8) * 128 + col] = c[m2][h][2]; f[(ar + 8) * 128 + col + 1] = c[m2][h][3];
                }
        __syncthreads();
        const int row = mt * 16 + warp;
        if (row < M)
        {
            float vg[4], vu[4], v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { vg[i] = fs[warp * 128 + lane * 4 + i]; vu[i] = fs[16 * 128 + warp * 128 + lane * 4 + i]; }
            qexl3::fwht128(vg, lane);
            qexl3::fwht128(vu, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = blockIdx.x * 128 + lane * 4 + i;
                float a = vg[i] * __half2float(svh_g[n]), u = vu[i] * __half2float(svh_u[n]);
                float act = C::act(a, u);
                v[i] = act * __half2float(suh_d[n]);
            }
            qexl3::fwht128(v, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) xh2[(r0 + row) * N + blockIdx.x * 128 + lane * 4 + i] = __float2half_rn(v[i]);
        }
        __syncthreads();
    }
}
// Paired variant: 8 warps, warp w computes both the gate and the up n16 tile w (one A fragment feeds 4 MMAs).
template <class C, int BITS, int STAGES, int KCG = KC, int MINB = 1, int MTN = 8>
__global__ __launch_bounds__(256, MINB) void gu2_gemm_kernel(const half* __restrict__ xh1, const uint64_t* __restrict__ tab,
                                                      const int* __restrict__ G, const int* __restrict__ gexp, const int* __restrict__ goff,
                                                      const int* __restrict__ gcnt, const int* __restrict__ prow, half* __restrict__ xh2)
{
    constexpr int NW = 8 * BITS, K = C::H, N = C::I, Nt = N / 16, SAG = KCG + 8, TW = (KCG / 16) * 8 * NW, GMR = 16 * MTN, PARTS = GM / GMR;
    extern __shared__ __align__(16) unsigned char gu_smem[];
    half* sa = (half*) gu_smem;                                           // [STAGES][GMR * SAG]
    uint32_t* sw = (uint32_t*) (gu_smem + STAGES * GMR * SAG * sizeof(half));   // [STAGES][gate TW | up TW]
    // MTN row tiles per CTA: an item (<= GM rows) is split into GM / (16 MTN) parts along blockIdx.y
    const int g = blockIdx.y / PARTS, hf = blockIdx.y % PARTS;
    if (g >= *G) return;
    const int M = min(GMR, gcnt[g] - hf * GMR), e = gexp[g], MT = (M + 15) / 16;
    if (M <= 0) return;
    const int64_t r0 = goff[g] + hf * GMR;
    const uint32_t* trg = (const uint32_t*) tab[0 * C::E + e];
    const uint32_t* tru = (const uint32_t*) tab[1 * C::E + e];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wn = warp;
    auto load = [&](int kc, int buf)
    {
        half* A = sa + buf * GMR * SAG;   // buf < STAGES
        for (int i = tid; i < M * (KCG / 8); i += 256)
        {
            const int r = i / (KCG / 8), c = i % (KCG / 8);
            cp16(&A[r * SAG + c * 8], xh1 + (int64_t) prow[r0 + r] * K + kc * KCG + c * 8);
        }
        uint32_t* W = sw + buf * 2 * TW;
        for (int i = tid; i < 2 * TW / 4; i += 256)
        {
            const int m = i / (TW / 4), j = i % (TW / 4), kk = j / (8 * NW / 4), jj = j % (8 * NW / 4);
            cp16(&W[m * TW + kk * 8 * NW + jj * 4], (m ? tru : trg) + ((int64_t) (kc * (KCG / 16) + kk) * Nt + blockIdx.x * 8) * NW + jj * 4);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float c[2][MTN][2][4] = {};
    constexpr int nkc = K / KCG;
    // STAGES-deep cp.async pipeline: chunk kc + STAGES - 1 is issued into the buffer of chunk kc - 1, which every warp
    // has finished (the barrier at the top of iteration kc)
    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) { if (i < nkc) load(i, i); else asm volatile("cp.async.commit_group;\n" ::); }
    for (int kc = 0; kc < nkc; ++kc)
    {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();
        if (kc + STAGES - 1 < nkc) load(kc + STAGES - 1, (kc + STAGES - 1) % STAGES); else asm volatile("cp.async.commit_group;\n" ::);
        const half* S = sa + (kc % STAGES) * GMR * SAG;
        const uint32_t* W = sw + (kc % STAGES) * 2 * TW;
        uint32_t bb[KCG / 16][2][4];
        #pragma unroll
        for (int kk = 0; kk < KCG / 16; ++kk)
        {
            #pragma unroll
            for (int m = 0; m < 2; ++m)
            {
                const uint32_t* p = W + m * TW + (kk * 8 + wn) * NW;
                qexl3::decode_tile<BITS, C::CBK>(p[lane], NW > 32 && lane + 32 < NW ? p[32 + lane] : 0u, lane, bb[kk][m]);
            }
        }
        #pragma unroll
        for (int kk = 0; kk < KCG / 16; ++kk)
            #pragma unroll
            for (int mt = 0; mt < MTN; ++mt)
            {
                if (mt >= MT) break;
                uint32_t a[4];
                const half* ap = S + (mt * 16 + (lane & 15)) * SAG + kk * 16 + (lane >> 4) * 8;
                unsigned sp = (unsigned) __cvta_generic_to_shared(ap);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sp));
                #pragma unroll
                for (int m = 0; m < 2; ++m)
                {
                    qexl3::mma16816(c[m][mt][0], a, bb[kk][m][0], bb[kk][m][1]);
                    qexl3::mma16816(c[m][mt][1], a, bb[kk][m][2], bb[kk][m][3]);
                }
            }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    // epilogue per row tile: gate (fs[0]) and up (fs[1]) fragments -> shared [16][128] fp32 -> warp per row
    float* fs = (float*) gu_smem;
    const half* svh_g = (const half*) tab[6 * C::E + e];
    const half* svh_u = (const half*) tab[7 * C::E + e];
    const half* suh_d = (const half*) tab[5 * C::E + e];
    const int ar = lane >> 2;
    for (int mt = 0; mt < MT; ++mt)
    {
        #pragma unroll
        for (int m2 = 0; m2 < MTN; ++m2)
            if (m2 == mt)
                #pragma unroll
                for (int m = 0; m < 2; ++m)
                    #pragma unroll
                    for (int h = 0; h < 2; ++h)
                    {
                        float* f = fs + m * 16 * 128;
                        const int col = wn * 16 + h * 8 + (lane & 3) * 2;
                        f[ar * 128 + col] = c[m][m2][h][0]; f[ar * 128 + col + 1] = c[m][m2][h][1];
                        f[(ar + 8) * 128 + col] = c[m][m2][h][2]; f[(ar + 8) * 128 + col + 1] = c[m][m2][h][3];
                    }
        __syncthreads();
        for (int rr = warp; rr < 16; rr += 8)
        {
            const int row = mt * 16 + rr;
            if (row >= M) break;
            float vg[4], vu[4], v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) { vg[i] = fs[rr * 128 + lane * 4 + i]; vu[i] = fs[16 * 128 + rr * 128 + lane * 4 + i]; }
            qexl3::fwht128(vg, lane);
            qexl3::fwht128(vu, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = blockIdx.x * 128 + lane * 4 + i;
                float a = vg[i] * __half2float(svh_g[n]), u = vu[i] * __half2float(svh_u[n]);
                float act = C::act(a, u);
                v[i] = act * __half2float(suh_d[n]);
            }
            qexl3::fwht128(v, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) xh2[(r0 + row) * N + blockIdx.x * 128 + lane * 4 + i] = __float2half_rn(v[i]);
        }
        __syncthreads();
    }
}
template <int STAGES, int KCG = KC, int ROWS = GM> constexpr size_t gu_smem_bytes()   // 4-bit
{
    constexpr size_t a = STAGES * (ROWS * (KCG + 8) * sizeof(half) + 2 * (KCG / 16) * 8 * 32 * sizeof(uint32_t));
    return a > 2 * 16 * 128 * sizeof(float) ? a : 2 * 16 * 128 * sizeof(float);   // the epilogue's fp32 row tiles
}

// finish of the down products + combine in one pass: warp per (row r, 128-block b); for k in slot order the pair's S
// partials are summed in order, fwht128, * svh, then acc += w * y; out = add + acc. The same operations as
// finish_grouped_kernel (stage 2) followed by combine_kernel<float> (bitwise equal), without yd in memory.
template <class C>
__global__ void finish_combine_kernel(const float* __restrict__ part, int S, const uint64_t* __restrict__ tab, int P,
                                      const int* __restrict__ pairmap, const int* __restrict__ pexp, const typename C::WT* __restrict__ w,
                                      const float* __restrict__ add, float* __restrict__ out, int R)
{
    pdl_enter();
    constexpr int N = C::H, NB = N / 128;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int b = warp % NB, r = warp / NB;
    if (r >= R) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < C::TOPK; ++k)
    {
        const int row = pairmap[r * C::TOPK + k];
        const half* svh = (const half*) tab[8 * C::E + pexp[row]];
        const float wk = tof(w[r * C::TOPK + k]);
        float v[4] = {};
        for (int s = 0; s < S; ++s)
        {
            const float4 p = *(const float4*) (part + ((int64_t) s * P + row) * N + b * 128 + lane * 4);
            v[0] += p.x; v[1] += p.y; v[2] += p.z; v[3] += p.w;
        }
        qexl3::fwht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += wk * (v[i] * __half2float(svh[b * 128 + lane * 4 + i]));
    }
    const int64_t o = (int64_t) r * N + b * 128 + lane * 4;
    const float4 a = add ? *(const float4*) (add + o) : make_float4(0.f, 0.f, 0.f, 0.f);
    *(float4*) (out + o) = make_float4(a.x + acc[0], a.y + acc[1], a.z + acc[2], a.w + acc[3]);
}

// rows per work item: 16 for decode / verify chains (S > 1, up to 64 rows: several sequences), 128 for prefill
// chunks (S = 1 above 16 rows; GEMM)
inline int group_rows(int R, int S) { return R > 64 || (R > 16 && S == 1) ? GM : 16; }

template <class C, int BITS>
void launch_products(const half* xh, const uint64_t* tab, int tr_slot0, const int* G, const int* gexp, const int* goff,
                     const int* gcnt, int Gmax, int GR, int T, int P, int K, int N, int S, float* part, cudaStream_t st)
{
    if (GR == GM)
        gemm_grouped_kernel<C, BITS><<<dim3(N / 128, T * Gmax), 128, 0, st>>>(xh, tab, tr_slot0, G, gexp, goff, gcnt, Gmax, P, K, N, part, -1);
    else
    {
        int kts = (K / 16 + S - 1) / S;
        gemv_grouped_kernel<C, BITS><<<dim3(N / 128, S, T * Gmax), 128, 0, st>>>(xh, tab, tr_slot0, G, gexp, goff, gcnt, Gmax, T, P,
                                                                             K, N, kts, part);
    }
}

template <class C>
int max_slices(int S, int R)
{
    if (!C::PERSIST) return S;
    const GemvCfg c = gemv_cfg<C>(S, R);
    return c.s1 > c.s2 ? c.s1 : c.s2;
}

template <class C>
int moe_gmax(int R, int S)
{
    int pairs = R * C::TOPK, items = pairs / group_rows(R, S) + C::E;
    return pairs < items ? pairs : items;
}

// Routed experts for R rows. ws: device scratch (ws_bytes); S = K slices of the decode GEMV (prefill: 1).
template <class C>
size_t ws_bytes(int R, int S)
{
    constexpr size_t H = C::H, I = C::I;
    size_t P = (size_t) R * C::TOPK, Gmax = (size_t) moe_gmax<C>(R, S), nb = (R + CB - 1) / CB;
    size_t b = 0;
    b += 4 * (1 + 3 * Gmax + nb * C::E + C::E + 3 * P) + 6 * 256;          // G, items, cblk, eoff, pairmap, prow, pexp
    if (group_rows(R, S) == GM)                                             // prefill: fused stages
        return b + P * I * 2 + 256 + P * H * 4 + 256 + (size_t) R * H * 2 + 256;   // xh2, yd, per-token xh1
    b += 2 * P * H * 2 + 256;                                               // xh stage 1
    b += (size_t) max_slices<C>(S, R) * P * (2 * I > H ? 2 * I : H) * 4 + 256; // partials (max of both stages)
    if (C::PERSIST) b += (size_t) max_slices<C>(S, R) * P * H * 4 + 256 + 8 * Gmax + 256;   // fused: stage-2 partials, counters
    b += 2 * P * I * 4 + 256;                                               // y stage 1
    b += P * I * 4 + 256;                                                   // act
    b += P * I * 2 + 256;                                                   // xh stage 2
    b += P * H * 4 + 256;                                                   // yd
    return b;
}

// x [R, H] (row stride ldx), idx / w [R, TOPK], out [R, H] fp32 (+ add [R, H] fp32 if given).
// shared: the layer's experts share one gate/up input scale vector suh0 (device pointer; prefill transforms each token once).
template <class C>
int experts(const void* x, int64_t ldx, int R, const void* idx, const void* w, const void* tab, const void* add, void* out,
            void* ws, int S, cudaStream_t st, const void* suh0 = nullptr)
{
    const bool shared = suh0 != nullptr;
    constexpr int H = C::H, I = C::I, E = C::E;
    using XT = typename C::XT;
    using IT = typename C::IT;
    using WT = typename C::WT;
    const int P = R * C::TOPK, Gmax = moe_gmax<C>(R, S), GR = group_rows(R, S), nb = (R + CB - 1) / CB;
    if (GR == GM) S = 1;
    char* p = (char*) ws;
    auto take = [&](size_t n) { char* q = p; p += (n + 255) / 256 * 256; return q; };
    int* G = (int*) take(4);
    int* gexp = (int*) take(4 * (size_t) Gmax);
    int* goff = (int*) take(4 * (size_t) Gmax);
    int* gcnt = (int*) take(4 * (size_t) Gmax);
    int* cblk = (int*) take(4 * (size_t) nb * E);
    int* eoff = (int*) take(4 * E);
    int* pairmap = (int*) take(4 * (size_t) P);
    int* prow = (int*) take(4 * (size_t) P);
    int* pexp = (int*) take(4 * (size_t) P);
    const uint64_t* tab_ = (const uint64_t*) tab;
    if (GR == GM)
    {
        half* xh2 = (half*) take((size_t) P * I * 2);
        void* yd = take((size_t) P * H * 4);
        count_kernel<C><<<nb, E, 0, st>>>((const IT*) idx, R, cblk);
        scan_kernel<C><<<1, E, 0, st>>>(cblk, nb, GR, eoff, G, gexp, goff, gcnt);
        fill_kernel<C><<<nb, E, 0, st>>>((const IT*) idx, R, cblk, eoff, pairmap, prow);
        if (shared)
        {
            // one transform per token (the layer's experts share the gate/up input scales; expert 0's suh)
            half* xh1 = (half*) take((size_t) R * H * 2);
            const int warps1 = R * (H / 128);
            qexl3::had_in_kernel<XT><<<(warps1 * 32 + 255) / 256, 256, 0, st>>>((const XT*) x, ldx, (const half*) suh0, xh1, R, H);
            // GLM53_MOE_GU_STAGES (2..4, default 2): pipeline depth of the gate/up GEMM
            static const int stages = [] { const char* v = getenv("GLM53_MOE_GU_STAGES"); int n = v ? atoi(v) : 2; return n < 2 ? 2 : n > 4 ? 4 : n; }();
#define GU(n) do { static bool a = cudaFuncSetAttribute(gu_gemm_kernel<C, 4, n>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gu_smem_bytes<n>()) == cudaSuccess; (void) a; \
                   gu_gemm_kernel<C, 4, n><<<dim3(I / 128, Gmax), 512, gu_smem_bytes<n>(), st>>>(xh1, tab_, G, gexp, goff, gcnt, prow, xh2); } while (0)
            // GLM53_MOE_GU_PAIR=0: the 16-warp variant (gate and up tiles in different warps)
            static const bool pair = [] { const char* v = getenv("GLM53_MOE_GU_PAIR"); return !(v && *v == '0'); }();
#define GU2(n) do { static bool a = cudaFuncSetAttribute(gu2_gemm_kernel<C, 4, n>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gu_smem_bytes<n>()) == cudaSuccess; (void) a; \
                    gu2_gemm_kernel<C, 4, n><<<dim3(I / 128, Gmax), 256, gu_smem_bytes<n>(), st>>>(xh1, tab_, G, gexp, goff, gcnt, prow, xh2); } while (0)
            // GLM53_MOE_GU_PAIR=2 (default): paired, k-chunks of 32, two CTAs per SM
            static const int pmode = [] { const char* v = getenv("GLM53_MOE_GU_PAIR"); return v ? atoi(v) : 4; }();
#define GU3(n) do { static bool a = cudaFuncSetAttribute(gu2_gemm_kernel<C, 4, n, 32, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gu_smem_bytes<n, 32>()) == cudaSuccess; (void) a; \
                    gu2_gemm_kernel<C, 4, n, 32, 2><<<dim3(I / 128, Gmax), 256, gu_smem_bytes<n, 32>(), st>>>(xh1, tab_, G, gexp, goff, gcnt, prow, xh2); } while (0)
            // GLM53_MOE_GU_PAIR=4 (default): paired, 64 rows per CTA, k-chunks GLM53_MOE_GU_KC (32/64), two CTAs per SM
            static const int kcg = [] { const char* v = getenv("GLM53_MOE_GU_KC"); return v ? atoi(v) : 64; }();
#define GU4(n, kc) do { static bool a = cudaFuncSetAttribute(gu2_gemm_kernel<C, 4, n, kc, 2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gu_smem_bytes<n, kc, 64>()) == cudaSuccess; (void) a; \
                        gu2_gemm_kernel<C, 4, n, kc, 2, 4><<<dim3(I / 128, 2 * Gmax), 256, gu_smem_bytes<n, kc, 64>(), st>>>(xh1, tab_, G, gexp, goff, gcnt, prow, xh2); } while (0)
            if (pmode == 4)
            {
                if (kcg == 32) { if (stages == 2) GU4(2, 32); else if (stages == 3) GU4(3, 32); else GU4(4, 32); }
                else { if (stages == 2) GU4(2, 64); else GU4(3, 64); }
            }
            else if (pmode == 2) { if (stages == 2) GU3(2); else if (stages == 3) GU3(3); else GU3(4); }
            else if (pair) { if (stages == 2) GU2(2); else GU2(3); }
            else if (stages == 2) GU(2); else if (stages == 3) GU(3); else GU(4);
#undef GU4
#undef GU3
#undef GU2
#undef GU
        }
        else gux_kernel<C, 4><<<dim3(I / 128, 2 * Gmax), 256, 0, st>>>((const XT*) x, ldx, tab_, G, gexp, goff, gcnt, prow, xh2);
        // QWEN_MOE_YD16=1 (read per launch; off): the routed experts' outputs in fp16 (halves the down GEMM's writes and the
        // combine's reads). 2026-10-07: tail-512 KL vs the fp32 reference 0.00245 -> 0.00924, not adopted.
        const char* y16 = getenv("QWEN_MOE_YD16");
        if (!(C::YD16 || (y16 && *y16 == '1')))
        {
            gemm_grouped_kernel<C, 4, 8, float><<<dim3(H / 128, Gmax), 256, 0, st>>>(xh2, tab_, 2, G, gexp, goff, gcnt, Gmax, P,
                                                                                   I, H, (float*) yd, 8);
            combine_kernel<C, float><<<R, 256, 0, st>>>((const float*) yd, pairmap, (const WT*) w, (const float*) add, (float*) out, H);
        }
        else
        {
            gemm_grouped_kernel<C, 4, 8, half><<<dim3(H / 128, Gmax), 256, 0, st>>>(xh2, tab_, 2, G, gexp, goff, gcnt, Gmax, P,
                                                                                  I, H, (half*) yd, 8);
            combine_kernel<C, half><<<R, 256, 0, st>>>((const half*) yd, pairmap, (const WT*) w, (const float*) add, (float*) out, H);
        }
        return (int) cudaGetLastError();
    }
    half* xh1 = (half*) take(2 * (size_t) P * H * 2);
    float* part = (float*) take((size_t) max_slices<C>(S, R) * P * (2 * I > H ? 2 * I : H) * 4);
    float* y1 = (float*) take(2 * (size_t) P * I * 4);
    float* act = (float*) take((size_t) P * I * 4);
    half* xh2 = (half*) take((size_t) P * I * 2);
    float* yd = (float*) take((size_t) P * H * 4);
    float* part2 = C::PERSIST ? (float*) take((size_t) max_slices<C>(S, R) * P * H * 4) : nullptr;
    int* cnt1 = C::PERSIST ? (int*) take(8 * (size_t) Gmax) : nullptr;
    const GemvCfg gc0 = gemv_cfg<C>(S, R);
    const bool fuse_all = C::PERSIST && gc0.fused && nb == 1;

    // QWEN_MOE_UNFUSED=1 (read per launch, i.e. at graph capture): the separate grouping / finish / act / transform
    const char* uf = getenv("QWEN_MOE_UNFUSED");
    const bool fused = nb == 1 && !(uf && *uf == '1');
    if (fused) klaunch<C>(prep_small_kernel<C>, 1, E, st, (const IT*) idx, R, GR, eoff, G, gexp, goff, gcnt, pairmap, prow, pexp,
                          cnt1, fuse_all ? 2 * Gmax : 0);
    else
    {
        count_kernel<C><<<nb, E, 0, st>>>((const IT*) idx, R, cblk);
        scan_kernel<C><<<1, E, 0, st>>>(cblk, nb, GR, eoff, G, gexp, goff, gcnt);
        fill_kernel<C><<<nb, E, 0, st>>>((const IT*) idx, R, cblk, eoff, pairmap, prow);
        pexp_kernel<<<Gmax, 128, 0, st>>>(G, gexp, goff, gcnt, pexp);
    }
    int warps = 2 * P * (H / 128);
    klaunch<C>(had_grouped_kernel<C, XT>, (warps * 32 + 255) / 256, 256, st, (const XT*) x, ldx, 0, tab_, 3, 2, P, prow, pexp, H, xh1);
    // decode GEMV schedule (C::PERSIST): persistent grid, per-stage K slices; else the plain grids with S slices
    const GemvCfg gc = gemv_cfg<C>(S, R);
    int S1 = S, S2 = S;
    if (fuse_all)
    {
        S2 = gc.s2;
#define LF(a, b, c, d) launch_fused<C, 4, a, b, c, d>(gc.tpw, gc.pf, gc.warps, gc.apf, gc.ctas, xh1, xh2, tab_, G, gexp, goff, gcnt, Gmax, P, gc.s1, \
                                                       gc.s2, part, part2, cnt1, cnt1 + Gmax, st)
        if (!(LF(2, 4, 4, false) || LF(2, 4, 4, true) || LF(4, 4, 4, false) || LF(4, 4, 4, true) || LF(2, 8, 4, false) || LF(4, 8, 4, true)
              || LF(8, 4, 4, false) || LF(8, 4, 4, true) || LF(4, 4, 8, false) || LF(4, 4, 8, true)))
        { fprintf(stderr, "[moe] no fused instance\n"); abort(); }
#undef LF
        warps = R * (H / 128);
        klaunch<C>(finish_combine_kernel<C>, (warps * 32 + 255) / 256, 256, st, part2, S2, tab_, P, pairmap, pexp, (const WT*) w,
                   (const float*) add, (float*) out, R);
        return (int) cudaGetLastError();
    }
    if (C::PERSIST) { S1 = gc.s1; S2 = gc.s2; launch_decode_products<C, 4>(gc, xh1, tab_, 0, G, gexp, goff, gcnt, Gmax, 2, P, H, I, S1, part, st); }
    else launch_products<C, 4>(xh1, tab_, 0, G, gexp, goff, gcnt, Gmax, GR, 2, P, H, I, S, part, st);
    if (fused)
    {
        warps = P * (I / 128);
        klaunch<C>(mid_grouped_kernel<C>, (warps * 32 + 255) / 256, 256, st, part, S1, tab_, P, pexp, xh2);
    }
    else
    {
        warps = 2 * P * (I / 128);
        finish_grouped_kernel<C><<<(warps * 32 + 255) / 256, 256, 0, st>>>(part, S1, tab_, 6, 2, P, pexp, I, y1);
        act_kernel<C><<<(unsigned) (((int64_t) P * I + 255) / 256), 256, 0, st>>>(y1, P, I, act);
        warps = P * (I / 128);
        had_grouped_kernel<C, float><<<(warps * 32 + 255) / 256, 256, 0, st>>>(act, I, 1, tab_, 5, 1, P, prow, pexp, I, xh2);
    }
    if (C::PERSIST) launch_decode_products<C, 4>(gc, xh2, tab_, 2, G, gexp, goff, gcnt, Gmax, 1, P, I, H, S2, part, st);
    else launch_products<C, 4>(xh2, tab_, 2, G, gexp, goff, gcnt, Gmax, GR, 1, P, I, H, S, part, st);
    if (C::FINISH_COMBINE)
    {
        warps = R * (H / 128);
        klaunch<C>(finish_combine_kernel<C>, (warps * 32 + 255) / 256, 256, st, part, S2, tab_, P, pairmap, pexp, (const WT*) w,
                   (const float*) add, (float*) out, R);
        return (int) cudaGetLastError();
    }
    warps = P * (H / 128);
    finish_grouped_kernel<C><<<(warps * 32 + 255) / 256, 256, 0, st>>>(part, S2, tab_, 8, 1, P, pexp, H, yd);
    combine_kernel<C, float><<<R, 256, 0, st>>>(yd, pairmap, (const WT*) w, (const float*) add, (float*) out, H);
    return (int) cudaGetLastError();
}

}  // namespace moex
