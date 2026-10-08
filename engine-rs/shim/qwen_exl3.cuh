#pragma once
// EXL3 (mul1 codebook) matrix products for Qwen3.8-Flash-Next, small row counts (decode / verify).
//
// y = x @ W with W = ((H @ inner) * suh) @ H * svh (H = 128-blockwise Walsh-Hadamard / sqrt(128)), so
//   1. xh = had128(x * suh)                (exl3_had_in, fp16 out: tensor-core input)
//   2. part[s] = xh[:, ks] @ inner[ks, :]  (exl3_gemv: K split in S fixed slices, fp32 partials)
//   3. y = had128(sum_s part[s]) * svh     (exl3_finish: slices summed in order -> deterministic)
// inner is decoded straight into m16n8k16 B fragments: the quantizer stores each 16x16 tile in
// tensor-core order (lane t holds positions 8t..8t+7 = rows (t%4)*2+{0,1,8,9}, cols t/4 and t/4+8).
// Tile bitstream: 8*BITS uint32 (little-endian halves of the int16 words), read MSB-first; the state
// of position p is the 16 bits starting at ((p+1)*BITS - 16) mod (256*BITS) (tail-biting).
#include <cuda_fp16.h>
#include <cstdint>

namespace qexl3 {

__device__ __forceinline__ half mul1_decode(uint32_t state)
{
    uint32_t x = state * 0x83DCD12Du;
    uint32_t s = __dp4a(x, 0x01010101u, 0u);                    // sum of the 4 bytes, 0..1020
    half h = __ushort_as_half((unsigned short) (0x6400u + s));   // 1024 + s, exact in fp16
    return __hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

// mcg codebook (GLM-5.3-Flash experts): x = state * 0xCBAC1FED, (x & 0x8fff8fff) ^ 0x3b603b60 as two fp16, summed.
__device__ __forceinline__ half mcg_decode(uint32_t state)
{
    uint32_t x = state * 0xCBAC1FEDu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    const half2 h = *reinterpret_cast<const half2*>(&x);
    return __hadd(__low2half(h), __high2half(h));
}

__device__ __forceinline__ uint32_t pack2(half lo, half hi)
{
    return (uint32_t) __half_as_ushort(lo) | ((uint32_t) __half_as_ushort(hi) << 16);
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Words of one tile held across the warp: lane l has word l (r0) and word l + 32 (r1, BITS > 4).
// Decode this lane's 8 positions into two B fragments (n8 halves 0 and 1 of the 16-wide tile).
// Two codebook values packed as half2 (low = state a): the same fp16 operations per element as mul1_decode / mcg_decode.
template <int CBK>
__device__ __forceinline__ uint32_t decode_pair(uint32_t a, uint32_t b)
{
    if constexpr (CBK == 1)
    {
        uint32_t x0 = a * 0xCBAC1FEDu, x1 = b * 0xCBAC1FEDu;
        x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
        x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
        const uint32_t lo = __byte_perm(x0, x1, 0x5410), hi = __byte_perm(x0, x1, 0x7632);   // (x0.lo, x1.lo), (x0.hi, x1.hi)
        const half2 r = __hadd2(*reinterpret_cast<const half2*>(&lo), *reinterpret_cast<const half2*>(&hi));
        return *reinterpret_cast<const uint32_t*>(&r);
    }
    else
    {
        const uint32_t s0 = __dp4a(a * 0x83DCD12Du, 0x01010101u, 0x6400u), s1 = __dp4a(b * 0x83DCD12Du, 0x01010101u, 0x6400u);
        const uint32_t h = (s0 & 0xffffu) | (s1 << 16);
        const half2 r = __hfma2(*reinterpret_cast<const half2*>(&h), __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<const uint32_t*>(&r);
    }
}

// CBK: codebook, 2 = mul1 (Qwen), 1 = mcg (GLM experts).
template <int BITS, int CBK = 2>
__device__ __forceinline__ void decode_tile(uint32_t r0, uint32_t r1, int lane, uint32_t* b)
{
    if constexpr (BITS == 4)
    {
        // 4 bits: position 8*lane starts at bit 32*lane - 12 (mod 1024), so every lane's 8 states lie in words lane-1 and
        // lane (bit offsets 20 + 4j of that 64-bit pair): two shuffles, one funnel shift per state. Same states as below.
        const uint32_t w0 = __shfl_sync(0xffffffffu, r0, (lane + 31) & 31), w1 = __shfl_sync(0xffffffffu, r0, lane);
        uint32_t st[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) st[j] = __funnelshift_r(w1, w0, 28 - 4 * j) & 0xffffu;
        #pragma unroll
        for (int q = 0; q < 4; ++q) b[q] = decode_pair<CBK>(st[2 * q], st[2 * q + 1]);
        return;
    }
    constexpr int NW = 8 * BITS, L = 256 * BITS;
    const int s0 = ((8 * lane + 1) * BITS - 16 + L) % L;          // first state bit of position 8*lane
    const int i0 = s0 >> 5;
    uint32_t w[3];
    #pragma unroll
    for (int j = 0; j < 3; ++j)
    {
        int q = (i0 + j) % NW;
        uint32_t v0 = __shfl_sync(0xffffffffu, r0, q & 31);
        uint32_t v1 = __shfl_sync(0xffffffffu, r1, q & 31);
        w[j] = q < 32 ? v0 : v1;
    }
    const uint64_t hi = ((uint64_t) w[0] << 32) | w[1];
    const uint64_t lo = ((uint64_t) w[1] << 32) | w[2];
    half v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int rel = (s0 & 31) + j * BITS;                             // < 32 + 7*BITS + 16 <= 90
        uint32_t st = rel <= 48 ? (uint32_t) (hi >> (48 - rel)) & 0xffffu
                                : (uint32_t) (lo >> (80 - rel)) & 0xffffu;
        v[j] = CBK == 1 ? mcg_decode(st) : mul1_decode(st);
    }
    b[0] = pack2(v[0], v[1]); b[1] = pack2(v[2], v[3]);           // n8 half 0 (col t/4)
    b[2] = pack2(v[4], v[5]); b[3] = pack2(v[6], v[7]);           // n8 half 1 (col t/4 + 8)
}

// In-place normalized Walsh-Hadamard over 128 values held 4 per lane (lane l: elements 4l..4l+3).
__device__ __forceinline__ void fwht128(float* v, int lane)
{
    // stages within the lane (h = 1, 2)
    float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
    // stages across lanes (h = 4 .. 64 elements = 1 .. 16 lanes)
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1)
    {
        bool upper = lane & m;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = upper ? o - v[i] : v[i] + o;
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764831845f;
}

// fwht128 with 16 elements per lane: 8 lanes per row (part p = lane % 8 holds elements 16p..16p+15), 4 rows per warp.
// Stages h = 1..8 within the lane, 16..64 across the 8 lanes; the same butterflies, stage order and operand order as
// fwht128 (bitwise equal), 12 shuffles per row instead of 20.
__device__ __forceinline__ void fwht128_16(float* v, int part)
{
    #pragma unroll
    for (int h = 1; h < 16; h <<= 1)
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            if (!(i & h)) { const float a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
    #pragma unroll
    for (int m = 1; m < 8; m <<= 1)
    {
        const bool upper = part & m;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = upper ? o - v[i] : v[i] + o;
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) v[i] *= 0.08838834764831845f;
}

// xh[m, k] = had128(x[m, :] * suh)[k] as fp16; one warp per (row, 128-block). x fp32 or fp16.
template <typename T>
__global__ void had_in_kernel(const T* __restrict__ x, int64_t ldx, const half* __restrict__ suh,
                              half* __restrict__ xh, int M, int K)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int nb = K / 128;
    if (warp >= M * nb) return;
    int m = warp / nb, kb = warp % nb;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int k = kb * 128 + lane * 4 + i;
        v[i] = (float) x[m * ldx + k] * __half2float(suh[k]);
    }
    fwht128(v, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) xh[(int64_t) m * K + kb * 128 + lane * 4 + i] = __float2half_rn(v[i]);
}

// Several linears with the same input in one launch (decode rows): up to 4, each with its own suh / trellis / svh.
constexpr int MAXL = 4;
struct Multi
{
    int n;                       // linears
    const half* suh[MAXL];
    half* xh[MAXL];              // [M, K] fp16 each
    const uint32_t* tr[MAXL];
    float* part[MAXL];           // [S_i, M, N_i]
    int N[MAXL], S[MAXL], kts[MAXL], bits[MAXL], blk0[MAXL + 1];   // gemv blocks of linear i: [blk0[i], blk0[i+1]) = N_i / 128 x S_i
    const half* svh[MAXL];
    float* y[MAXL];
    int64_t ldy[MAXL];
};
// had_in for every linear from one read of x: warp per (row, 128-block); the same values as had_in_kernel per linear.
// mix (optional, decode HC): x is mixed[r][i] = (1/4) sum_s n[r][s][i] * sigmoid(g[r][s][i]) computed here (as
// hc mix2_kernel) and also written to xout.
static __global__ void had_in_multi_kernel(const float* __restrict__ x, int64_t ldx, const float* __restrict__ mg, const float* __restrict__ mn,
                                    float* __restrict__ xout, int M, int K, const Multi ml)
{
    const int64_t warp = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = K / 128;
    if (warp >= (int64_t) M * nb) return;
    const int m = (int) (warp / nb), kb = (int) (warp % nb);
    float xv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int k = kb * 128 + lane * 4 + i;
        if (mg)
        {
            const float* gr = mg + (int64_t) m * 4 * K;
            const float* nr = mn + (int64_t) m * 4 * K;
            float v0 = nr[k] / (1.0f + __expf(-gr[k])), v1 = nr[K + k] / (1.0f + __expf(-gr[K + k]));
            float v2 = nr[2 * K + k] / (1.0f + __expf(-gr[2 * K + k])), v3 = nr[3 * K + k] / (1.0f + __expf(-gr[3 * K + k]));
            xv[i] = ((v0 + v1) + (v2 + v3)) * 0.25f;
            xout[(int64_t) m * K + k] = xv[i];
        }
        else xv[i] = x[(int64_t) m * ldx + k];
    }
    for (int l = 0; l < ml.n; ++l)
    {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = xv[i] * __half2float(ml.suh[l][kb * 128 + lane * 4 + i]);
        fwht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) ml.xh[l][(int64_t) m * K + kb * 128 + lane * 4 + i] = __float2half_rn(v[i]);
    }
}
// finish for every linear: warp per (linear, row, 128-block), as finish_row
static __global__ void finish_multi_kernel(int M, const Multi ml)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    int l = 0;
    for (; l < ml.n; ++l) { const int w = M * (ml.N[l] / 128); if (warp < w) break; warp -= w; }
    if (l >= ml.n) return;
    const int N = ml.N[l], nb = N / 128, m = warp / nb, b = warp % nb, S = ml.S[l];
    const float* part = ml.part[l];
    float v[4] = {};
    for (int s = 0; s < S; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] += part[((int64_t) s * M + m) * N + b * 128 + lane * 4 + i];
    fwht128(v, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int n = b * 128 + lane * 4 + i;
        ml.y[l][m * ml.ldy[l] + n] = v[i] * __half2float(ml.svh[l][n]);
    }
}
// shared expert (decode): finish of gate (linear 0) and up (linear 1) and h = silu(gate) * up (as silu_mul_kernel)
static __global__ void finish_silu_kernel(int M, const Multi ml, float* __restrict__ h)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int N = ml.N[0], nb = N / 128;
    if (warp >= M * nb) return;
    const int m = warp / nb, b = warp % nb;
    float g[2][4];
    #pragma unroll
    for (int l = 0; l < 2; ++l)
    {
        float v[4] = {};
        for (int s = 0; s < ml.S[l]; ++s)
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] += ml.part[l][((int64_t) s * M + m) * N + b * 128 + lane * 4 + i];
        fwht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) g[l][i] = v[i] * __half2float(ml.svh[l][b * 128 + lane * 4 + i]);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const float x = g[0][i];
        const float sl = x / (1.0f + expf(-x));
        h[(int64_t) m * N + b * 128 + lane * 4 + i] = sl * g[1][i];
    }
}

// Block: WARPS warps x TPW n-tiles (16 cols each) over k-tiles [kt0, kt0 + kts); xh rows [M <= 16 MT, K] fp16.
// MT row tiles of 16 share each decoded B fragment (prefill: one weight decode per 16 MT rows).
// APF: the A fragments are loaded PF k-tiles ahead together with the weights (same values; latency off the MMA chain).
template <int BITS, int TPW = 2, int PF = 4, int WARPS = 4, int MT = 1, int CBK = 2, bool APF = false>
__device__ __forceinline__ void gemv_block(const half* __restrict__ xh, const uint32_t* __restrict__ tr,
                                           float* __restrict__ out, int M, int K, int N, int nblk, int kt0, int kts)
{
    constexpr int NW = 8 * BITS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int Nt = N / 16, Kt = K / 16;
    const int nt0 = nblk * (WARPS * TPW) + warp * TPW;
    const int kt1 = min(Kt, kt0 + kts);
    const int ar = lane >> 2, ak = (lane & 3) * 2;
    float c[MT][TPW][2][4] = {};
    uint32_t pf[PF][TPW][2];
    uint32_t apf[APF ? PF : 1][MT][4];
    auto load_a = [&](int kt, uint32_t (&a)[MT][4])
    {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
        {
            const int r0 = mt * 16 + ar;
            const half* x0 = xh + (int64_t) r0 * K + kt * 16 + ak;
            const half* x1 = x0 + (int64_t) 8 * K;
            a[mt][0] = r0 < M ? *(const uint32_t*) x0 : 0u;
            a[mt][1] = r0 + 8 < M ? *(const uint32_t*) x1 : 0u;
            a[mt][2] = r0 < M ? *(const uint32_t*) (x0 + 8) : 0u;
            a[mt][3] = r0 + 8 < M ? *(const uint32_t*) (x1 + 8) : 0u;
        }
    };
    auto load = [&](int kt, uint32_t (&d)[TPW][2])
    {
        #pragma unroll
        for (int t = 0; t < TPW; ++t)
        {
            const uint32_t* p = tr + ((int64_t) kt * Nt + nt0 + t) * NW;
            d[t][0] = __ldcs(p + lane);
            d[t][1] = NW > 32 && lane + 32 < NW ? __ldcs(p + 32 + lane) : 0u;
        }
    };
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (kt0 + d < kt1) { load(kt0 + d, pf[d]); if constexpr (APF) load_a(kt0 + d, apf[d]); }
    for (int kb = kt0; kb < kt1; kb += PF)
    {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            int kt = kb + d;
            if (kt >= kt1) break;
            uint32_t cur[TPW][2];
            #pragma unroll
            for (int t = 0; t < TPW; ++t) { cur[t][0] = pf[d][t][0]; cur[t][1] = pf[d][t][1]; }
            uint32_t a[MT][4];
            if constexpr (APF)
            {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) a[mt][q] = apf[d][mt][q];
                if (kt + PF < kt1) { load(kt + PF, pf[d]); load_a(kt + PF, apf[d]); }
            }
            else
            {
            if (kt + PF < kt1) load(kt + PF, pf[d]);
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
            {
                const int r0 = mt * 16 + ar;
                const half* x0 = xh + (int64_t) r0 * K + kt * 16 + ak;
                const half* x1 = x0 + (int64_t) 8 * K;
                a[mt][0] = r0 < M ? *(const uint32_t*) x0 : 0u;
                a[mt][1] = r0 + 8 < M ? *(const uint32_t*) x1 : 0u;
                a[mt][2] = r0 < M ? *(const uint32_t*) (x0 + 8) : 0u;
                a[mt][3] = r0 + 8 < M ? *(const uint32_t*) (x1 + 8) : 0u;
            }
            }
            #pragma unroll
            for (int t = 0; t < TPW; ++t)
            {
                uint32_t b[4];
                decode_tile<BITS, CBK>(cur[t][0], cur[t][1], lane, b);
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt)
                {
                    if (mt * 16 >= M) break;                        // warp-uniform
                    mma16816(c[mt][t][0], a[mt], b[0], b[1]);
                    mma16816(c[mt][t][1], a[mt], b[2], b[3]);
                }
            }
        }
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int t = 0; t < TPW; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int r0 = mt * 16 + ar;
                int n = (nt0 + t) * 16 + h * 8 + (lane & 3) * 2;
                if (r0 < M) { out[(int64_t) r0 * N + n] = c[mt][t][h][0]; out[(int64_t) r0 * N + n + 1] = c[mt][t][h][1]; }
                if (r0 + 8 < M) { out[(int64_t) (r0 + 8) * N + n] = c[mt][t][h][2]; out[(int64_t) (r0 + 8) * N + n + 1] = c[mt][t][h][3]; }
            }
}

// grid (N / (16 WARPS TPW), S); slice s covers k-tiles [s*kts, (s+1)*kts); part [S, M, N]
// MT row tiles of 16 (M <= 16 MT) share every decoded weight tile: one weight pass for up to 64 rows. Per element
// the same mma sequence for any MT (rows are independent), so results do not depend on the batch size.
template <int BITS, int TPW = 2, int PF = 4, int WARPS = 4, int MT = 1>
__global__ __launch_bounds__(WARPS * 32) void gemv_kernel(const half* __restrict__ xh, const uint32_t* __restrict__ tr,
                                                          float* __restrict__ part, int M, int K, int N, int kts)
{
    gemv_block<BITS, TPW, PF, WARPS, MT>(xh, tr, part + (int64_t) blockIdx.y * M * N, M, K, N, blockIdx.x, blockIdx.y * kts, kts);
}

// gemv of every linear in one grid (x: blocks of all linears, y: unused): the block finds its linear and runs
// gemv_block exactly as gemv_kernel for it (bitwise equal per linear).
template <int MT, int UBITS = 0>   // UBITS: all linears of that width (one code path); 0: per linear
__global__ __launch_bounds__(128) void gemv_multi_kernel(int M, int K, const Multi ml)
{
    int b = blockIdx.x, l = 0;
    while (l + 1 < ml.n && b >= ml.blk0[l + 1]) ++l;
    b -= ml.blk0[l];
    const int nbk = ml.N[l] / 128, nblk = b % nbk, s = b / nbk;
    float* out = ml.part[l] + (int64_t) s * M * ml.N[l];
    if (UBITS) { gemv_block<UBITS, 2, 4, 4, MT>(ml.xh[l], ml.tr[l], out, M, K, ml.N[l], nblk, s * ml.kts[l], ml.kts[l]); return; }
    switch (ml.bits[l])
    {
        case 4: gemv_block<4, 2, 4, 4, MT>(ml.xh[l], ml.tr[l], out, M, K, ml.N[l], nblk, s * ml.kts[l], ml.kts[l]); break;
        case 5: gemv_block<5, 2, 4, 4, MT>(ml.xh[l], ml.tr[l], out, M, K, ml.N[l], nblk, s * ml.kts[l], ml.kts[l]); break;
        default: gemv_block<6, 2, 4, 4, MT>(ml.xh[l], ml.tr[l], out, M, K, ml.N[l], nblk, s * ml.kts[l], ml.kts[l]); break;
    }
}

// Decode linear in one launch: had_in + gemv + finish of the same layout (bitwise equal). Block (n-block of 128, K
// slice s): the rows' 128-column blocks of x the slice touches are transformed (x * suh, fwht128, fp16) into shared
// memory, the slice's k-tiles run as in gemv_block reading A from there, the partials go to part [S, M, N]; the last
// slice block of the n-block to finish (counter cnt[nblk], reset by it) sums the S partials in order, applies fwht128
// and svh and writes y (fp32, row stride ldy). x fp32 [M <= 16 MT, K] (row stride ldx).
constexpr int GF_XLD = 128 + 8;
template <int BITS, int MT>
__global__ __launch_bounds__(128) void gemv_fused_kernel(const float* __restrict__ x, int64_t ldx, const half* __restrict__ suh,
                                                         const uint32_t* __restrict__ tr, float* __restrict__ part,
                                                         const half* __restrict__ svh, float* __restrict__ y, int64_t ldy,
                                                         int* __restrict__ cnt, int M, int K, int N, int kts)
{
    constexpr int TPW = 2, PF = 4, WARPS = 4, NW = 8 * BITS;
    // MT = 1 and a slice within GF_NB 128-blocks: all its blocks transformed up front (one barrier); otherwise one
    // 128-block at a time
    constexpr int GF_NB = MT == 1 ? 8 : 1;
    __shared__ __align__(16) half xs[GF_NB * 16 * MT * GF_XLD];
    __shared__ int is_last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int Nt = N / 16, Kt = K / 16, S = gridDim.y, nblk = blockIdx.x, s = blockIdx.y;
    const int nt0 = nblk * (WARPS * TPW) + warp * TPW;
    const int kt0 = s * kts, kt1 = min(Kt, kt0 + kts);
    const int ar = lane >> 2, ak = (lane & 3) * 2;
    float c[MT][TPW][2][4] = {};
    uint32_t pf[PF][TPW][2];
    auto load = [&](int kt, uint32_t (&d)[TPW][2])
    {
        #pragma unroll
        for (int t = 0; t < TPW; ++t)
        {
            const uint32_t* p = tr + ((int64_t) kt * Nt + nt0 + t) * NW;
            d[t][0] = __ldcs(p + lane);
            d[t][1] = NW > 32 && lane + 32 < NW ? __ldcs(p + 32 + lane) : 0u;
        }
    };
    const int cb0 = kt0 >> 3, ncb = kt1 > kt0 ? ((kt1 - 1) >> 3) - cb0 + 1 : 0;
    const bool upfront = ncb <= GF_NB && GF_NB > 1;
    auto transform = [&](int cb, half* xs)   // 128-block cb of every row -> xs (as had_in_kernel)
    {
        for (int m = warp; m < M; m += WARPS)
        {
            float v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int k = cb * 128 + lane * 4 + i;
                v[i] = x[m * ldx + k] * __half2float(suh[k]);
            }
            fwht128(v, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) xs[m * GF_XLD + lane * 4 + i] = __float2half_rn(v[i]);
        }
    };
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (kt0 + d < kt1) load(kt0 + d, pf[d]);
    if (upfront)
    {
        for (int j = 0; j < ncb; ++j) transform(cb0 + j, xs + j * 16 * MT * GF_XLD);
        __syncthreads();
    }
    for (int kb = kt0; kb < kt1; kb += PF)
    {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int kt = kb + d;
            if (kt >= kt1) break;
            if (!upfront && (kt == kt0 || (kt & 7) == 0))
            {
                __syncthreads();   // the previous 128-block is consumed
                transform(kt >> 3, xs);
                __syncthreads();
            }
            const half* xb = upfront ? xs + ((kt >> 3) - cb0) * 16 * MT * GF_XLD : xs;
            uint32_t cur[TPW][2];
            #pragma unroll
            for (int t = 0; t < TPW; ++t) { cur[t][0] = pf[d][t][0]; cur[t][1] = pf[d][t][1]; }
            if (kt + PF < kt1) load(kt + PF, pf[d]);
            const int kc = (kt & 7) * 16 + ak;
            uint32_t a[MT][4];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
            {
                const int r0 = mt * 16 + ar;
                const half* x0 = xb + r0 * GF_XLD + kc;
                const half* x1 = x0 + 8 * GF_XLD;
                a[mt][0] = r0 < M ? *(const uint32_t*) x0 : 0u;
                a[mt][1] = r0 + 8 < M ? *(const uint32_t*) x1 : 0u;
                a[mt][2] = r0 < M ? *(const uint32_t*) (x0 + 8) : 0u;
                a[mt][3] = r0 + 8 < M ? *(const uint32_t*) (x1 + 8) : 0u;
            }
            #pragma unroll
            for (int t = 0; t < TPW; ++t)
            {
                uint32_t b[4];
                decode_tile<BITS>(cur[t][0], cur[t][1], lane, b);
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt)
                {
                    if (mt * 16 >= M) break;
                    mma16816(c[mt][t][0], a[mt], b[0], b[1]);
                    mma16816(c[mt][t][1], a[mt], b[2], b[3]);
                }
            }
        }
    }
    float* out = part + (int64_t) s * M * N;
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int t = 0; t < TPW; ++t)
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int r0 = mt * 16 + ar;
                int n = (nt0 + t) * 16 + h * 8 + (lane & 3) * 2;
                if (r0 < M) { out[(int64_t) r0 * N + n] = c[mt][t][h][0]; out[(int64_t) r0 * N + n + 1] = c[mt][t][h][1]; }
                if (r0 + 8 < M) { out[(int64_t) (r0 + 8) * N + n] = c[mt][t][h][2]; out[(int64_t) (r0 + 8) * N + n + 1] = c[mt][t][h][3]; }
            }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) is_last = atomicAdd(&cnt[nblk], 1) == S - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    for (int m = warp; m < M; m += WARPS)
    {
        float v[4] = {};
        for (int ss = 0; ss < S; ++ss)
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] += __ldcg(part + ((int64_t) ss * M + m) * N + nblk * 128 + lane * 4 + i);
        fwht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int n = nblk * 128 + lane * 4 + i;
            y[m * ldy + n] = v[i] * __half2float(svh[n]);
        }
    }
    if (threadIdx.x == 0) cnt[nblk] = 0;
}

// y[m, n] = had128(sum_s part[s, m, :])[n] * svh[n]; one warp per (row, 128-block). y fp32 or fp16.
template <typename P, typename T>
__device__ __forceinline__ void finish_row(const P* __restrict__ part, int S, const half* __restrict__ svh,
                                           T* __restrict__ y, int64_t ldy, int M, int N)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int nb = N / 128;
    if (warp >= M * nb) return;
    int m = warp / nb, b = warp % nb;
    float v[4] = {};
    for (int s = 0; s < S; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] += (float) part[((int64_t) s * M + m) * N + b * 128 + lane * 4 + i];
    fwht128(v, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int n = b * 128 + lane * 4 + i;
        y[m * ldy + n] = (T) (v[i] * __half2float(svh[n]));
    }
}

template <typename T>
__global__ void finish_kernel(const float* __restrict__ part, int S, const half* __restrict__ svh,
                              T* __restrict__ y, int64_t ldy, int M, int N)
{
    finish_row<float, T>(part, S, svh, y, ldy, M, N);
}

// one fp16 part (a tensor-core GEMM output, S = 1): the same values as converting it to fp32 first
template <typename T>
__global__ void finish_h_kernel(const half* __restrict__ part, const half* __restrict__ svh, T* __restrict__ y, int64_t ldy, int M, int N)
{
    finish_row<half, T>(part, 1, svh, y, ldy, M, N);
}


// inner[k, n] fp16 for all tiles; one warp per tile (lane t: rows (t%4)*2+{0,1,8,9}, cols t/4 and t/4+8)
template <int BITS>
__global__ void reconstruct_kernel(const uint32_t* __restrict__ tr, half* __restrict__ w, int K, int N)
{
    constexpr int NW = 8 * BITS;
    int64_t tile = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int Nt = N / 16;
    if (tile >= (int64_t) (K / 16) * Nt) return;
    int kt = tile / Nt, nt = tile % Nt;
    const uint32_t* p = tr + tile * NW;
    uint32_t r0 = p[lane], r1 = NW > 32 && lane + 32 < NW ? p[32 + lane] : 0u;
    uint32_t b[4];
    decode_tile<BITS>(r0, r1, lane, b);
    int k0 = kt * 16 + (lane & 3) * 2, n0 = nt * 16 + (lane >> 2);
    auto lo = [](uint32_t v) { return __ushort_as_half((unsigned short) (v & 0xffff)); };
    auto hi = [](uint32_t v) { return __ushort_as_half((unsigned short) (v >> 16)); };
    w[(int64_t) (k0 + 0) * N + n0] = lo(b[0]); w[(int64_t) (k0 + 1) * N + n0] = hi(b[0]);
    w[(int64_t) (k0 + 8) * N + n0] = lo(b[1]); w[(int64_t) (k0 + 9) * N + n0] = hi(b[1]);
    w[(int64_t) (k0 + 0) * N + n0 + 8] = lo(b[2]); w[(int64_t) (k0 + 1) * N + n0 + 8] = hi(b[2]);
    w[(int64_t) (k0 + 8) * N + n0 + 8] = lo(b[3]); w[(int64_t) (k0 + 9) * N + n0 + 8] = hi(b[3]);
}


// Effective weight of an EXL3 linear, y = had(had(x * suh) @ inner) * svh = x @ W, W = diag(suh) H inner H diag(svh)
// (H: normalized Hadamard per 128-block): one 128 x 128 block per block of 256 threads, inner decoded into shared
// memory, both Hadamards and scales in fp32, W^T (Linear orientation [N, K], row stride ldo) written in fp16.
constexpr int FOLD_LD = 129;
template <int BITS>
__global__ __launch_bounds__(256) void fold_kernel(const uint32_t* __restrict__ tr, const half* __restrict__ suh, const half* __restrict__ svh,
                                                   int K, int N, half* __restrict__ out, int64_t ldo)
{
    constexpr int NW = 8 * BITS;
    extern __shared__ float fs[];   // [128][FOLD_LD], k-major
    const int nb = blockIdx.x, kb = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, Nt = N / 16;
    for (int t = warp; t < 64; t += 8)
    {
        const int kt = kb * 8 + t / 8, nt = nb * 8 + t % 8;
        const uint32_t* p = tr + ((int64_t) kt * Nt + nt) * NW;
        uint32_t b[4];
        decode_tile<BITS>(p[lane], NW > 32 && lane + 32 < NW ? p[32 + lane] : 0u, lane, b);
        const int k0 = (t / 8) * 16 + (lane & 3) * 2, n0 = (t % 8) * 16 + (lane >> 2);
        auto lo = [](uint32_t v) { return __half2float(__ushort_as_half((unsigned short) (v & 0xffff))); };
        auto hi = [](uint32_t v) { return __half2float(__ushort_as_half((unsigned short) (v >> 16))); };
        fs[(k0 + 0) * FOLD_LD + n0] = lo(b[0]); fs[(k0 + 1) * FOLD_LD + n0] = hi(b[0]);
        fs[(k0 + 8) * FOLD_LD + n0] = lo(b[1]); fs[(k0 + 9) * FOLD_LD + n0] = hi(b[1]);
        fs[(k0 + 0) * FOLD_LD + n0 + 8] = lo(b[2]); fs[(k0 + 1) * FOLD_LD + n0 + 8] = hi(b[2]);
        fs[(k0 + 8) * FOLD_LD + n0 + 8] = lo(b[3]); fs[(k0 + 9) * FOLD_LD + n0 + 8] = hi(b[3]);
    }
    __syncthreads();
    for (int c = warp; c < 128; c += 8)     // H along k, then * suh[k]
    {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = fs[(4 * lane + i) * FOLD_LD + c];
        fwht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) fs[(4 * lane + i) * FOLD_LD + c] = v[i] * __half2float(suh[kb * 128 + 4 * lane + i]);
    }
    __syncthreads();
    for (int r = warp; r < 128; r += 8)     // H along n, then * svh[n]
    {
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = fs[r * FOLD_LD + 4 * lane + i];
        fwht128(v, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) fs[r * FOLD_LD + 4 * lane + i] = v[i] * __half2float(svh[nb * 128 + 4 * lane + i]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 128 * 64; i += 256)
    {
        const int n = i / 64, k = (i % 64) * 2;
        *(__half2*) (out + (int64_t) (nb * 128 + n) * ldo + kb * 128 + k) =
            __floats2half2_rn(fs[k * FOLD_LD + n], fs[(k + 1) * FOLD_LD + n]);
    }
}

// v2: the block's k-slice of tiles (8 adjacent n-tiles = 8 * NW contiguous words per k-tile) is streamed into a
// STAGES-deep shared-memory ring with 16-byte cp.async; warps decode from shared memory (3 words per lane per tile).
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

template <int BITS>
__device__ __forceinline__ void decode_tile_smem(const uint32_t* t, int lane, uint32_t* b)
{
    constexpr int NW = 8 * BITS, L = 256 * BITS;
    const int s0 = ((8 * lane + 1) * BITS - 16 + L) % L;
    const int i0 = s0 >> 5;
    uint32_t w0 = t[i0], w1 = t[(i0 + 1) % NW], w2 = t[(i0 + 2) % NW];
    const uint64_t hi = ((uint64_t) w0 << 32) | w1;
    const uint64_t lo = ((uint64_t) w1 << 32) | w2;
    half v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int rel = (s0 & 31) + j * BITS;
        uint32_t st = rel <= 48 ? (uint32_t) (hi >> (48 - rel)) & 0xffffu : (uint32_t) (lo >> (80 - rel)) & 0xffffu;
        v[j] = mul1_decode(st);
    }
    b[0] = pack2(v[0], v[1]); b[1] = pack2(v[2], v[3]);
    b[2] = pack2(v[4], v[5]); b[3] = pack2(v[6], v[7]);
}

template <int BITS, int STAGES = 6>
__device__ __forceinline__ void gemv_block_v2(const half* __restrict__ xh, const uint32_t* __restrict__ tr,
                                              float* __restrict__ out, int M, int K, int N, int nblk, int kt0, int kts)
{
    constexpr int NW = 8 * BITS, ROWW = 8 * NW;            // words per k-step (8 n-tiles)
    constexpr int CHUNKS = ROWW / 4;                         // 16-byte chunks per k-step
    __shared__ __align__(16) uint32_t ring[STAGES][ROWW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int Nt = N / 16, Kt = K / 16;
    const int kt1 = min(Kt, kt0 + kts), nk = max(0, kt1 - kt0);
    const uint32_t* base = tr + (int64_t) nblk * ROWW;      // n-tile group start within a k-row
    const int64_t krow = (int64_t) Nt * NW;                 // words per k-row
    auto issue = [&](int i)
    {
        if (i < nk)
        {
            const uint32_t* src = base + (int64_t) (kt0 + i) * krow;
            for (int c = threadIdx.x; c < CHUNKS; c += blockDim.x) cp_async16(&ring[i % STAGES][c * 4], src + c * 4);
        }
        cp_async_commit();
    };
    #pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) issue(i);
    const int ar = lane >> 2, ak = (lane & 3) * 2;
    const bool row0 = ar < M, row1 = ar + 8 < M;
    float c[2][2][4] = {};
    for (int i = 0; i < nk; ++i)
    {
        issue(i + STAGES - 1);
        cp_async_wait<STAGES - 1>();
        __syncthreads();
        const int kt = kt0 + i;
        uint32_t a[4];
        const half* x0 = xh + (int64_t) ar * K + kt * 16 + ak;
        const half* x1 = x0 + (int64_t) 8 * K;
        a[0] = row0 ? *(const uint32_t*) x0 : 0u;
        a[1] = row1 ? *(const uint32_t*) x1 : 0u;
        a[2] = row0 ? *(const uint32_t*) (x0 + 8) : 0u;
        a[3] = row1 ? *(const uint32_t*) (x1 + 8) : 0u;
        const uint32_t* st = ring[i % STAGES];
        #pragma unroll
        for (int t = 0; t < 2; ++t)
        {
            uint32_t b[4];
            decode_tile_smem<BITS>(st + (warp * 2 + t) * NW, lane, b);
            mma16816(c[t][0], a, b[0], b[1]);
            mma16816(c[t][1], a, b[2], b[3]);
        }
        __syncthreads();
    }
    cp_async_wait<0>();
    const int nt0 = nblk * 8 + warp * 2;
    #pragma unroll
    for (int t = 0; t < 2; ++t)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int n = (nt0 + t) * 16 + h * 8 + (lane & 3) * 2;
            if (row0) { out[(int64_t) ar * N + n] = c[t][h][0]; out[(int64_t) ar * N + n + 1] = c[t][h][1]; }
            if (row1) { out[(int64_t) (ar + 8) * N + n] = c[t][h][2]; out[(int64_t) (ar + 8) * N + n + 1] = c[t][h][3]; }
        }
}

template <int BITS, int STAGES = 6>
__global__ __launch_bounds__(128) void gemv_v2_kernel(const half* __restrict__ xh, const uint32_t* __restrict__ tr,
                                                      float* __restrict__ part, int M, int K, int N, int kts)
{
    gemv_block_v2<BITS, STAGES>(xh, tr, part + (int64_t) blockIdx.y * M * N, M, K, N, blockIdx.x, blockIdx.y * kts, kts);
}

}  // namespace qexl3

