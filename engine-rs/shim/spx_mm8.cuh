// SPDX-License-Identifier: MIT
// SPX1 prefill on INT8 tensor cores (D7): y = W_rot . x_rot with the codes read straight from the SPX tiles (no Half
// dequant scratch) and x_rot quantized per token to symmetric INT8 (scale sx = max|x_rot| / 127, round half even).
//   W_rot[o, i] = (2 c - mid) s / 2, mid = 2^b - 1 (b in {4, 5, 6}: the odd integers 2c - mid fit s8 exactly),
//   y[t, o] = sx[t] sum_g s[o, g] / 2 sum_{i in g} (2 c - mid) x_q                         (g: 64-input groups)
// The integer part of every group is exact (int32 MMA accumulate); each group is scaled in FP32 as it completes.
// Kernels:
//   spx_quant8  x_rot Half [M, K] -> xq int8 [M, K], sx FP32 [M], sumx int32 [M, K/64] (one CTA per row)
//   spx_mm8     CTA tile 64 outputs (4 SPX row tiles) x 128 tokens, 8 warps (warp: one row tile x 64 tokens), K in quads of
//               128 through a 2-stage cp.async pipeline; mma.m16n8k32 s8 with the weights as A (rows) and the tokens as B.
// Operand order: lane (g, t) of the SPX tile holds k = u*32 + 8t + i (i < 8) of rows g and g + 8 in chunk u; mapped to the
// mma k slots 4t + i (a0/a1, b0) and 16 + 4t + i - 4 (a2/a3, b1) for A and B alike (a bijection of the 32 k of the chunk,
// so the dot products are unchanged): the B fragment of lane (g, t) is the 8 bytes x_q[token g][u*32 + 8t ..].

#include <algorithm>
#include <cstdlib>
namespace spx8 {
constexpr int TOK = 128, ROWS = 64, XS = 144;   // tokens and outputs per CTA, smem row stride of x_q (bytes)

// Codes 0..3 (lo) and 4..7 (hi) of a group as 4 bytes each: nibble word n (code i = 2p + h: low nibble at bit 16h + 4p),
// high word hw already shifted to this group's slot (code i: its b - 4 high bits at bit 16h + 4p).
template <int B>
__device__ __forceinline__ void codes8(uint32_t n, uint32_t hw, uint32_t& lo, uint32_t& hi)
{
    lo = __byte_perm(n & 0x000F000Fu, (n >> 4) & 0x000F000Fu, 0x6420);
    hi = __byte_perm((n >> 8) & 0x000F000Fu, (n >> 12) & 0x000F000Fu, 0x6420);
    if constexpr (B > 4)
    {
        constexpr uint32_t m = B == 5 ? 0x00010001u : 0x00030003u;
        lo |= __byte_perm(hw & m, (hw >> 4) & m, 0x6420) << 4;
        hi |= __byte_perm((hw >> 8) & m, (hw >> 12) & m, 0x6420) << 4;
    }
    // signed odd integers v = 2c - mid per byte: 2c + (128 - mid) < 256 (no carry between bytes), then - 128 = ^ 0x80
    constexpr uint32_t off = (uint32_t) (128 - ((1 << B) - 1)) * 0x01010101u;
    lo = ((lo << 1) + off) ^ 0x80808080u;
    hi = ((hi << 1) + off) ^ 0x80808080u;
}
__device__ __forceinline__ void mma8(int* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* s, const void* g)
{
    unsigned a = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(g));
}

__global__ void __launch_bounds__(256) quant8(const __half* __restrict__ X, int K, int8_t* __restrict__ Q, float* __restrict__ sx,
                                               int* __restrict__ sumx)
{
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const __half* x = X + (size_t) row * K;
    __shared__ float red[8];
    float m = 0.f;
    for (int i = tid * 8; i < K; i += 256 * 8)
    {
        const uint4 v = *reinterpret_cast<const uint4*>(x + i);
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) { const float2 f = __half22float2(h[j]); m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y))); }
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w) m = fmaxf(m, red[w]);
    const float s = m > 0.f ? m / 127.f : 1.f;
    if (tid == 0) sx[row] = s;
    // 8 threads per 64-input group (8 values each); group sums by shuffles within the 8 lanes
    for (int i = tid * 8; i < K; i += 256 * 8)
    {
        const uint4 v = *reinterpret_cast<const uint4*>(x + i);
        const __half2* h = reinterpret_cast<const __half2*>(&v);
        int q[8], sum = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float2 f = __half22float2(h[j]);
            q[2 * j] = (int) fminf(fmaxf(rintf(f.x / s), -127.f), 127.f);
            q[2 * j + 1] = (int) fminf(fmaxf(rintf(f.y / s), -127.f), 127.f);
        }
        uint2 o;
        o.x = (q[0] & 255) | (q[1] & 255) << 8 | (q[2] & 255) << 16 | (uint32_t) (q[3] & 255) << 24;
        o.y = (q[4] & 255) | (q[5] & 255) << 8 | (q[6] & 255) << 16 | (uint32_t) (q[7] & 255) << 24;
        *reinterpret_cast<uint2*>(Q + (size_t) row * K + i) = o;
        #pragma unroll
        for (int j = 0; j < 8; ++j) sum += q[j];
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        sum += __shfl_xor_sync(0xffffffffu, sum, 4);
        if ((lane & 7) == 0) sumx[(size_t) row * (K / 64) + i / 64] = sum;
    }
}

// x [M, K] (FP32 or Half, row stride ldx) -> x_rot = blockdiag(H_bk)(sign (.) x) quantized per row to INT8 in one pass:
// one CTA per row, K / 8 threads (8 values each), the same butterflies as spx_rot (in-thread, warp shuffles, shared
// memory), scale = max |x_rot| / 127 over the row from the FP32 values (no Half rounding before quantization).
template <typename TI>
__global__ void __launch_bounds__(1024) rot8(const TI* __restrict__ X, long long ldx, const int8_t* __restrict__ sign, int K, int bk,
                                             int8_t* __restrict__ Q, float* __restrict__ sx)
{
    extern __shared__ float rsm[];
    __shared__ float red[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nt = bk >> 3, m = blockIdx.x, base = tid * 8;
    const TI* xr = X + (long long) m * ldx + base;
    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = spx_ldf(xr + i) * (float) sign[base + i];
    #pragma unroll
    for (int h = 1; h < 8; h <<= 1)
        #pragma unroll
        for (int i = 0; i < 8; ++i) if (!(i & h)) { const float a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
    for (int hs = 1; hs < 32 && hs < nt; hs <<= 1)
        #pragma unroll
        for (int i = 0; i < 8; ++i) { const float o = __shfl_xor_sync(0xffffffffu, v[i], hs); v[i] = (tid & hs) ? (o - v[i]) : (v[i] + o); }
    for (int hs = 32; hs < nt; hs <<= 1)
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) rsm[tid * 8 + i] = v[i];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 8; ++i) { const float o = rsm[(tid ^ hs) * 8 + i]; v[i] = (tid & hs) ? (o - v[i]) : (v[i] + o); }
        __syncthreads();
    }
    const float sc = rsqrtf((float) bk);
    float a = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] *= sc; a = fmaxf(a, fabsf(v[i])); }
    #pragma unroll
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if (lane == 0) red[warp] = a;
    __syncthreads();
    a = 0.f;
    for (int w = 0; w < (K >> 8); ++w) a = fmaxf(a, red[w]);
    const float s = a > 0.f ? a / 127.f : 1.f;
    if (tid == 0) sx[m] = s;
    int q[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = (int) fminf(fmaxf(rintf(v[i] / s), -127.f), 127.f);
    uint2 o;
    o.x = (q[0] & 255) | (q[1] & 255) << 8 | (q[2] & 255) << 16 | (uint32_t) (q[3] & 255) << 24;
    o.y = (q[4] & 255) | (q[5] & 255) << 8 | (q[6] & 255) << 16 | (uint32_t) (q[7] & 255) << 24;
    *reinterpret_cast<uint2*>(Q + (long long) m * K + base) = o;
}

// Y [M, N] (row stride ldy): OUT 0 FP32, 1 FP32 of the Half-rounded value, 2 Half.
// Block rasterization (GLM53_SPX_MM8_GR, L0: per-CTA work unchanged, bitwise the same Y). gr == 0: the original 2-D grid
// (row blocks fastest: every token block streams all N weight rows from DRAM again). gr > 0: 1-D grid swizzled into groups
// of gr row blocks; inside a group the row blocks vary fastest and the token blocks slowly, so the group's weights (sized
// to stay in L2) are read from DRAM once and every 128-token x block is reused from L2 by the group's row blocks.
template <int B, int OUT>
__global__ void __launch_bounds__(256) mm8(const int8_t* __restrict__ Q, const float* __restrict__ sx, const int* __restrict__ sumx,
                                           const uint8_t* __restrict__ T, int M, int N, int K, void* __restrict__ Y, long long ldy, int gr)
{
    constexpr int TB = 256 * B + 64, WST = ROWS / 16 * TB, XST = TOK * XS;   // bytes per stage: weights, tokens
    extern __shared__ __align__(16) unsigned char sm[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp & 3, wn = warp >> 2;                 // row tile of the warp, token half
    int rb = blockIdx.x, tb = blockIdx.y;
    if (gr > 0)
    {
        const int nrb = (N + ROWS - 1) / ROWS, ntb = (M + TOK - 1) / TOK, id = blockIdx.x, grp = id / (gr * ntb);
        const int g0 = grp * gr, gn = min(gr, nrb - g0), r = id - g0 * ntb;
        tb = r / gn; rb = g0 + r % gn;
    }
    const int nq = K / 128, ntile = N / 16, tile0 = rb * 4, tok0 = tb * TOK;
    const bool live = tile0 + wm < ntile;
    auto load = [&](int q, int st)
    {
        unsigned char* w = sm + st * (WST + XST);
        unsigned char* x = w + WST;
        for (int i = tid; i < WST / 16; i += 256)
        {
            const int tl = i / (TB / 16), o = i % (TB / 16);
            const int tile = min(tile0 + tl, ntile - 1);
            cp16(w + tl * TB + o * 16, T + ((size_t) tile * nq + q) * TB + o * 16);
        }
        for (int i = tid; i < TOK * 8; i += 256)
        {
            const int r = i >> 3, o = i & 7;
            const int tok = min(tok0 + r, M - 1);
            cp16(x + r * XS + o * 16, Q + (size_t) tok * K + q * 128 + o * 16);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[8][4] = {};
    load(0, 0);
    for (int q = 0; q < nq; ++q)
    {
        if (q + 1 < nq) { load(q + 1, (q + 1) & 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const unsigned char* w = sm + (q & 1) * (WST + XST) + wm * TB;
        const unsigned char* x = sm + (q & 1) * (WST + XST) + WST + (wn * 64 + g) * XS + 8 * t;
        if (live)
        {
            const uint4 n0 = *reinterpret_cast<const uint4*>(w + lane * 16), n1 = *reinterpret_cast<const uint4*>(w + 512 + lane * 16);
            uint4 h = make_uint4(0, 0, 0, 0);
            if constexpr (B == 6) h = *reinterpret_cast<const uint4*>(w + 1024 + lane * 16);
            if constexpr (B == 5) { const uint2 h2 = *reinterpret_cast<const uint2*>(w + 1024 + lane * 8); h = make_uint4(h2.x, h2.y, 0, 0); }
            const uint32_t nw[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
            const uint32_t hw[4] = {h.x, h.y, h.z, h.w};
            const float2 sa = __half22float2(*reinterpret_cast<const __half2*>(w + 256 * B + 4 * g));       // row g: groups 0, 1
            const float2 sb = __half22float2(*reinterpret_cast<const __half2*>(w + 256 * B + 4 * (g + 8)));
            #pragma unroll
            for (int grp = 0; grp < 2; ++grp)
            {
                int c[8][4] = {};
                #pragma unroll
                for (int uu = 0; uu < 2; ++uu)
                {
                    const int u = grp * 2 + uu, w0 = 2 * u, w1 = 2 * u + 1;   // groups of rows g, g + 8
                    uint32_t h0 = 0, h1 = 0;
                    if constexpr (B == 6) { h0 = hw[w0 / 2] >> (2 * (w0 % 2)); h1 = hw[w1 / 2] >> (2 * (w1 % 2)); }
                    if constexpr (B == 5) { h0 = hw[w0 / 4] >> (w0 % 4); h1 = hw[w1 / 4] >> (w1 % 4); }
                    uint32_t a0, a2, a1, a3;
                    codes8<B>(nw[w0], h0, a0, a2);
                    codes8<B>(nw[w1], h1, a1, a3);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        const uint2 b = *reinterpret_cast<const uint2*>(x + j * 8 * XS + u * 32);
                        mma8(c[j], a0, a1, a2, a3, b.x, b.y);
                    }
                }
                // group done: acc += s / 2 * sum (2c - mid) x, rows g (c0, c1) and g + 8 (c2, c3)
                const float s0 = 0.5f * (grp ? sa.y : sa.x), s1 = 0.5f * (grp ? sb.y : sb.x);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    acc[j][0] += s0 * (float) c[j][0];
                    acc[j][1] += s0 * (float) c[j][1];
                    acc[j][2] += s1 * (float) c[j][2];
                    acc[j][3] += s1 * (float) c[j][3];
                }
            }
        }
        __syncthreads();
    }
    // epilogue: * sx[token], through shared memory [token][64 rows] fp32, then row-contiguous stores
    float* ys = reinterpret_cast<float*>(sm);
    constexpr int YS = ROWS + 4;
    if (live)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            const int r = wn * 64 + j * 8 + 2 * t;
            const float fa = sx[min(tok0 + r, M - 1)], fb = sx[min(tok0 + r + 1, M - 1)];
            ys[r * YS + wm * 16 + g] = acc[j][0] * fa;
            ys[(r + 1) * YS + wm * 16 + g] = acc[j][1] * fb;
            ys[r * YS + wm * 16 + g + 8] = acc[j][2] * fa;
            ys[(r + 1) * YS + wm * 16 + g + 8] = acc[j][3] * fb;
        }
    __syncthreads();
    const int ncol = min(ROWS, N - rb * ROWS);
    for (int i = tid; i < TOK * (ROWS / 4); i += 256)
    {
        const int r = i / (ROWS / 4), c4 = (i % (ROWS / 4)) * 4;
        const int tok = tok0 + r;
        if (tok >= M || c4 >= ncol) continue;
        const float4 v = *reinterpret_cast<const float4*>(ys + r * YS + c4);
        const size_t o = (size_t) tok * ldy + rb * ROWS + c4;
        if constexpr (OUT == 2) *reinterpret_cast<uint2*>(reinterpret_cast<__half*>(Y) + o) = make_uint2(f2h2(v.x, v.y), f2h2(v.z, v.w));
        else if constexpr (OUT == 1)
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(Y) + o) = make_float4(__half2float(__float2half_rn(v.x)), __half2float(__float2half_rn(v.y)),
                                                                                       __half2float(__float2half_rn(v.z)), __half2float(__float2half_rn(v.w)));
        else *reinterpret_cast<float4*>(reinterpret_cast<float*>(Y) + o) = v;
    }
}
template <int B> constexpr int smem_bytes()
{
    constexpr int st = 2 * ((ROWS / 16) * (256 * B + 64) + TOK * XS), ep = TOK * (ROWS + 4) * 4;
    return st > ep ? st : ep;
}
}  // namespace spx8

// mm8c: CTA 128 rows (8 SPX row tiles, one per warp) x 128 tokens. Each warp decodes its row tile's codes once per quad
// (A fragments reused by 16 token tiles) and loads them straight from global memory into registers one quad ahead (no
// shared-memory staging of weights); x_q goes through a 2-stage cp.async pipeline in XOR-swizzled shared memory (shared
// by all 8 warps). Per output the arithmetic is mm8's: exact int32 group sums, acc = fma(s / 2, float(c), acc) per 64-k
// group in k order, then * sx: bitwise the same Y.
namespace spx8c {
constexpr int WARPS = 4, NT = WARPS * 32, TOK = 128, ROWS = 16 * WARPS, XS = 128;
__device__ __forceinline__ int xsw(int r, int chunk) { return r * XS + ((chunk ^ (2 * (r & 3))) << 4); }
template <int B> struct W { uint4 n0, n1, h; uint32_t sa, sb; };
template <int B> __device__ __forceinline__ W<B> wload(const uint8_t* p, int lane, int g)
{
    W<B> d;
    d.n0 = skinny_ld_stream(p + lane * 16); d.n1 = skinny_ld_stream(p + 512 + lane * 16);
    if constexpr (B == 6) d.h = skinny_ld_stream(p + 1024 + lane * 16);
    else if constexpr (B == 5) { uint2 h; asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1},[%2];" : "=r"(h.x), "=r"(h.y) : "l"(p + 1024 + lane * 8)); d.h = make_uint4(h.x, h.y, 0, 0); }
    else d.h = make_uint4(0, 0, 0, 0);
    d.sa = __ldg(reinterpret_cast<const uint32_t*>(p + 256 * B) + g); d.sb = __ldg(reinterpret_cast<const uint32_t*>(p + 256 * B) + g + 8);
    return d;
}
template <int B, int OUT>
__global__ void __launch_bounds__(NT, 2) mm8c(const int8_t* __restrict__ Q, const float* __restrict__ sx, const uint8_t* __restrict__ T,
                                               int M, int N, int K, void* __restrict__ Y, long long ldy, int gr)
{
    constexpr int TB = 256 * B + 64, XST = TOK * XS, WST = WARPS * TB, ST = WST + XST;
    extern __shared__ __align__(16) unsigned char xs[];   // stage: [WARPS weight tiles][x_q swizzled]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int nq = K / 128, ntile = N / 16;
    int rb = blockIdx.x, tb = blockIdx.y;
    if (gr > 0)
    {
        const int nrb = (N + ROWS - 1) / ROWS, ntb = (M + TOK - 1) / TOK, id = blockIdx.x, grp = id / (gr * ntb);
        const int g0 = grp * gr, gn = min(gr, nrb - g0), r = id - g0 * ntb;
        tb = r / gn; rb = g0 + r % gn;
    }
    const int tile = rb * WARPS + warp, tok0 = tb * TOK;
    const bool live = tile < ntile;
    auto xload = [&](int q, int st)
    {
        unsigned char* w = xs + st * ST;
        for (int i = tid; i < WST / 16; i += NT)
        {
            const int tl = i / (TB / 16), o = i % (TB / 16), tile_ = min(rb * WARPS + tl, ntile - 1);
            spx8::cp16(w + tl * TB + o * 16, T + ((size_t) tile_ * nq + q) * TB + o * 16);
        }
        unsigned char* x = w + WST;
        #pragma unroll
        for (int i = tid; i < TOK * 8; i += NT)
        {
            const int r = i >> 3, o = i & 7, tok = min(tok0 + r, M - 1);
            spx8::cp16(x + xsw(r, o), Q + (size_t) tok * K + q * 128 + o * 16);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[16][4];
    #pragma unroll
    for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    xload(0, 0);
    for (int q = 0; q < nq; ++q)
    {
        if (q + 1 < nq) { xload(q + 1, (q + 1) & 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const unsigned char* wsm = xs + (q & 1) * ST + warp * TB;
        const unsigned char* x = xs + (q & 1) * ST + WST;
        if (live)
        {
            W<B> cur;
            cur.n0 = *reinterpret_cast<const uint4*>(wsm + lane * 16); cur.n1 = *reinterpret_cast<const uint4*>(wsm + 512 + lane * 16);
            if constexpr (B == 6) cur.h = *reinterpret_cast<const uint4*>(wsm + 1024 + lane * 16);
            else if constexpr (B == 5) { const uint2 h2 = *reinterpret_cast<const uint2*>(wsm + 1024 + lane * 8); cur.h = make_uint4(h2.x, h2.y, 0, 0); }
            else cur.h = make_uint4(0, 0, 0, 0);
            cur.sa = *reinterpret_cast<const uint32_t*>(wsm + 256 * B + 4 * g); cur.sb = *reinterpret_cast<const uint32_t*>(wsm + 256 * B + 4 * (g + 8));
            const uint32_t nw[8] = {cur.n0.x, cur.n0.y, cur.n0.z, cur.n0.w, cur.n1.x, cur.n1.y, cur.n1.z, cur.n1.w};
            const uint32_t hw[4] = {cur.h.x, cur.h.y, cur.h.z, cur.h.w};
            const float2 sa = __half22float2(*reinterpret_cast<const __half2*>(&cur.sa)), sb = __half22float2(*reinterpret_cast<const __half2*>(&cur.sb));
            #pragma unroll
            for (int grp = 0; grp < 2; ++grp)
            {
                uint32_t A[2][4];
                #pragma unroll
                for (int uu = 0; uu < 2; ++uu)
                {
                    const int u = grp * 2 + uu, w0 = 2 * u, w1 = 2 * u + 1;
                    uint32_t h0 = 0, h1 = 0;
                    if constexpr (B == 6) { h0 = hw[w0 / 2] >> (2 * (w0 % 2)); h1 = hw[w1 / 2] >> (2 * (w1 % 2)); }
                    if constexpr (B == 5) { h0 = hw[w0 / 4] >> (w0 % 4); h1 = hw[w1 / 4] >> (w1 % 4); }
                    spx8::codes8<B>(nw[w0], h0, A[uu][0], A[uu][2]);
                    spx8::codes8<B>(nw[w1], h1, A[uu][1], A[uu][3]);
                }
                const float s0 = 0.5f * (grp ? sa.y : sa.x), s1 = 0.5f * (grp ? sb.y : sb.x);
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    int c[4] = {0, 0, 0, 0};
                    #pragma unroll
                    for (int uu = 0; uu < 2; ++uu)
                    {
                        const int u = grp * 2 + uu;
                        const uint2 b = *reinterpret_cast<const uint2*>(x + xsw(j * 8 + g, 2 * u + (t >> 1)) + 8 * (t & 1));
                        spx8::mma8(c, A[uu][0], A[uu][1], A[uu][2], A[uu][3], b.x, b.y);
                    }
                    acc[j][0] += s0 * (float) c[0];
                    acc[j][1] += s0 * (float) c[1];
                    acc[j][2] += s1 * (float) c[2];
                    acc[j][3] += s1 * (float) c[3];
                }
            }
        }
        __syncthreads();
    }
    if (!live) return;
    // lane: rows rbase + g (c0, c1) and + 8 (c2, c3) of tokens j * 8 + 2t (c0, c2), + 1 (c1, c3)
    const int row = tile * 16 + g;
    #pragma unroll
    for (int j = 0; j < 16; ++j)
    {
        #pragma unroll
        for (int e = 0; e < 2; ++e)
        {
            const int tok = tok0 + j * 8 + 2 * t + e;
            if (tok >= M) continue;
            const float f = sx[tok];
            const float v0 = acc[j][e] * f, v1 = acc[j][2 + e] * f;
            const size_t o = (size_t) tok * ldy + row;
            if constexpr (OUT == 2) { reinterpret_cast<__half*>(Y)[o] = __float2half_rn(v0); reinterpret_cast<__half*>(Y)[o + 8] = __float2half_rn(v1); }
            else if constexpr (OUT == 1) { reinterpret_cast<float*>(Y)[o] = __half2float(__float2half_rn(v0)); reinterpret_cast<float*>(Y)[o + 8] = __half2float(__float2half_rn(v1)); }
            else { reinterpret_cast<float*>(Y)[o] = v0; reinterpret_cast<float*>(Y)[o + 8] = v1; }
        }
    }
}
}  // namespace spx8c
extern "C" int glm53_spx_quant8_cuda(const void* x, int m, int k, void* q, void* sx, void* sumx, cudaStream_t s)
{
    if (k % 256) return int(cudaErrorInvalidValue);   // whole warps per pass (group-sum shuffles)
    if (m == 0) return 0;
    spx8::quant8<<<m, 256, 0, s>>>((const __half*) x, k, (int8_t*) q, (float*) sx, (int*) sumx);
    return int(cudaGetLastError());
}
// Shared-expert activation for the SPX down projection (decode rows): y FP32 [M, 2n] (gate | up, the SPX gate/up group
// output) -> act FP32 [M, n] = silu(min(g, L)) * clamp(u, -L, L) (ATen's clamp / silu x / (1 + exp(-x)) / mul, FP32) and
// x_rot Half [M, n] = H_bk(sign (.) act) / sqrt(bk), the spx_rot<float, __half, true> butterflies in the same order. One CTA
// per row, n / 8 threads; replaces four elementwise launches and the rotation launch.
__global__ void __launch_bounds__(1024) swiglu_rot(const float* __restrict__ Y, int n, float L, const int8_t* __restrict__ sign, int bk,
                                                   float* __restrict__ act, __half* __restrict__ xr)
{
    extern __shared__ float ssm[];
    const int tid = threadIdx.x, nt = bk >> 3, m = blockIdx.x, base = tid * 8;
    const float* g = Y + (long long) m * 2 * n + base;
    const float* u = g + n;
    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const float gg = fminf(g[i], L), uu = fminf(fmaxf(u[i], -L), L);
        v[i] = gg / (1.0f + expf(-gg)) * uu;
    }
    *reinterpret_cast<float4*>(act + (long long) m * n + base) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(act + (long long) m * n + base + 4) = make_float4(v[4], v[5], v[6], v[7]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] *= (float) sign[base + i];
    #pragma unroll
    for (int h = 1; h < 8; h <<= 1)
        #pragma unroll
        for (int i = 0; i < 8; ++i) if (!(i & h)) { const float a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
    const unsigned mask = nt >= 32 ? 0xffffffffu : ((1u << nt) - 1u);
    for (int hs = 1; hs < 32 && hs < nt; hs <<= 1)
        #pragma unroll
        for (int i = 0; i < 8; ++i) { const float o = __shfl_xor_sync(mask, v[i], hs); v[i] = (tid & hs) ? (o - v[i]) : (v[i] + o); }
    for (int hs = 32; hs < nt; hs <<= 1)
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) ssm[tid * 8 + i] = v[i];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 8; ++i) { const float o = ssm[(tid ^ hs) * 8 + i]; v[i] = (tid & hs) ? (o - v[i]) : (v[i] + o); }
        __syncthreads();
    }
    const float sc = rsqrtf((float) bk);
    #pragma unroll
    for (int i = 0; i < 8; ++i) xr[(long long) m * n + base + i] = __float2half_rn(v[i] * sc);
}
// KDA output norm + gate for an SPX o_proj (decode rows): one CTA per row, one warp per head exactly as kda_onorm_gate
// (same per-lane sums and xor reduction: out bitwise equal), then x_rot = H_bk(sign (.) out) / sqrt(bk) in shared memory,
// stage by stage in ascending bit order with the same a + b / a - b pairs as spx_rot (bitwise equal x_rot).
__global__ void __launch_bounds__(1024) kda_onorm_rot(const float* __restrict__ o, const float* __restrict__ onorm, int onorm_len,
                                                     const float* __restrict__ sg2, float* __restrict__ out, int heads,
                                                     const int8_t* __restrict__ sign, int bk, __half* __restrict__ xr)
{
    extern __shared__ float ksm[];
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, W = heads * 128;
    {
        const long long g = (long long) row * heads + warp;
        const float* src = o + g * 128; float v[4]; float s = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) { v[j] = src[lane + 32 * j]; s += v[j] * v[j]; }
        for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        const float r = rsqrtf(s / 128.f + 1e-5f);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const long long i = g * 128 + lane + 32 * j; const int d = (int) (i % onorm_len);
            const float y = (onorm[d] * (v[j] * r)) * sg2[i];
            out[i] = y;
            const int c = warp * 128 + lane + 32 * j;
            ksm[c] = y * (float) sign[c];
        }
    }
    __syncthreads();
    // spx_rot's stage schedule on W / 8 threads (8 consecutive values each): bits 0-2 in registers, 3-7 by shuffles,
    // the rest through shared memory -- the same pairs in the same order (bitwise equal); the other threads only sync.
    const int tid = threadIdx.x, nt = bk >> 3; const bool act = tid < (W >> 3);
    float v[8];
    if (act)
    {
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = ksm[tid * 8 + i];
        #pragma unroll
        for (int h = 1; h < 8; h <<= 1)
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (!(i & h)) { const float a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
        for (int hs = 1; hs < 32 && hs < nt; hs <<= 1)
            #pragma unroll
            for (int i = 0; i < 8; ++i) { const float q = __shfl_xor_sync(0xffffffffu, v[i], hs); v[i] = (tid & hs) ? (q - v[i]) : (v[i] + q); }
    }
    for (int hs = 32; hs < nt; hs <<= 1)
    {
        __syncthreads();
        if (act)
            #pragma unroll
            for (int i = 0; i < 8; ++i) ksm[tid * 8 + i] = v[i];
        __syncthreads();
        if (act)
            #pragma unroll
            for (int i = 0; i < 8; ++i) { const float q = ksm[(tid ^ hs) * 8 + i]; v[i] = (tid & hs) ? (q - v[i]) : (v[i] + q); }
    }
    if (act)
    {
        const float sc = rsqrtf((float) bk);
        #pragma unroll
        for (int i = 0; i < 8; ++i) xr[(long long) row * W + tid * 8 + i] = __float2half_rn(v[i] * sc);
    }
}
extern "C" int glm53_spx_kda_onorm_rot_cuda(const void* o, const void* onorm, int onorm_len, const void* sg2, void* out, int rows, int heads,
                                           const void* sign, int bk, void* xr, cudaStream_t s)
{
    const int W = heads * 128;
    if (heads < 1 || heads > 32 || bk < 8 || bk > W || (bk & (bk - 1)) || W % bk) return int(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    kda_onorm_rot<<<rows, heads * 32, (size_t) W * sizeof(float), s>>>((const float*) o, (const float*) onorm, onorm_len, (const float*) sg2, (float*) out,
                                                                      heads, (const int8_t*) sign, bk, (__half*) xr);
    return int(cudaGetLastError());
}
extern "C" int glm53_spx_swiglu_rot_cuda(const void* y, int m, int n, float limit, const void* sign, int bk, void* act, void* xr, cudaStream_t s)
{
    if (n % 256 || n > 8192 || bk < 8 || bk > n || (bk & (bk - 1)) || n % bk) return int(cudaErrorInvalidValue);
    if (m == 0) return 0;
    swiglu_rot<<<m, n / 8, (size_t) n * sizeof(float), s>>>((const float*) y, n, limit, (const int8_t*) sign, bk, (float*) act, (__half*) xr);
    return int(cudaGetLastError());
}
extern "C" int glm53_spx_rot8_cuda(const void* x, long long ldx, int in_kind, const void* sign, int m, int k, int bk, void* q, void* sx,
                                   cudaStream_t s)
{
    if (k % 256 || k > 8192 || bk < 8 || bk > k || (bk & (bk - 1)) || k % bk) return int(cudaErrorInvalidValue);
    if (m == 0) return 0;
    const size_t smb = (size_t) k * sizeof(float);
    if (in_kind == 0) spx8::rot8<float><<<m, k / 8, smb, s>>>((const float*) x, ldx, (const int8_t*) sign, k, bk, (int8_t*) q, (float*) sx);
    else spx8::rot8<__half><<<m, k / 8, smb, s>>>((const __half*) x, ldx, (const int8_t*) sign, k, bk, (int8_t*) q, (float*) sx);
    return int(cudaGetLastError());
}
extern "C" int glm53_spx_mm8_cuda(const void* q, const void* sx, const void* sumx, const void* t, int m, int n, int k, int bits,
                                  void* y, long long ldy, int out, cudaStream_t s)
{
    if (n % 16 || k % 128 || ldy < n || ldy % 4 || (bits < 4 || bits > 6) || out < 0 || out > 2) return int(cudaErrorInvalidValue);
    if (m == 0) return 0;
    static const bool v1 = [] { const char* e = std::getenv("GLM53_SPX_MM8_V"); return e && e[0] == '1'; }();
    if (!v1)
    {
        const int nrb = (n + spx8c::ROWS - 1) / spx8c::ROWS, ntb = (m + spx8c::TOK - 1) / spx8c::TOK;
        static const int mb = [] { const char* e = std::getenv("GLM53_SPX_MM8_GR"); return e ? atoi(e) : 8; }();
        const long long wblk = (long long) spx8c::ROWS * k * (256 * bits + 64) / 2048;
        const int gr = mb <= 0 ? 0 : (int) std::max(1LL, std::min((long long) nrb, ((long long) mb << 20) / wblk));
        const dim3 grid = gr ? dim3(nrb * ntb) : dim3(nrb, ntb);
#define L(BB, OO) do { constexpr int sb = 2 * (spx8c::WARPS * (256 * BB + 64) + spx8c::TOK * spx8c::XS); static bool a = cudaFuncSetAttribute(spx8c::mm8c<BB, OO>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, sb) == cudaSuccess; (void) a;                                     \
                    spx8c::mm8c<BB, OO><<<grid, spx8c::NT, sb, s>>>((const int8_t*) q, (const float*) sx, (const uint8_t*) t, m, n, k, y, ldy, gr); } while (0)
#define LB(BB) do { if (out == 0) L(BB, 0); else if (out == 1) L(BB, 1); else L(BB, 2); } while (0)
        if (bits == 4) LB(4); else if (bits == 5) LB(5); else LB(6);
#undef LB
#undef L
        return int(cudaGetLastError());
    }
    const int nrb = (n + spx8::ROWS - 1) / spx8::ROWS, ntb = (m + spx8::TOK - 1) / spx8::TOK;
    // group size: the group's weight blocks take about GLM53_SPX_MM8_GR MB of L2 (default 8; 0 = original order)
    static const int mb = [] { const char* e = std::getenv("GLM53_SPX_MM8_GR"); return e ? atoi(e) : 8; }();
    const long long wblk = (long long) spx8::ROWS * k * (256 * bits + 64) / 2048;   // bytes of one 64-row weight block
    const int gr = mb <= 0 ? 0 : (int) std::max(1LL, std::min((long long) nrb, ((long long) mb << 20) / wblk));
    const dim3 grid = gr ? dim3(nrb * ntb) : dim3(nrb, ntb);
#define L1(BB, OO) do { constexpr int sb = spx8::smem_bytes<BB>(); static bool a = cudaFuncSetAttribute(spx8::mm8<BB, OO>,            \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, sb) == cudaSuccess; (void) a;                                     \
                    spx8::mm8<BB, OO><<<grid, 256, sb, s>>>((const int8_t*) q, (const float*) sx, (const int*) sumx,               \
                    (const uint8_t*) t, m, n, k, y, ldy, gr); } while (0)
#define L(BB, OO) L1(BB, OO)
#define LB(BB) do { if (out == 0) L(BB, 0); else if (out == 1) L(BB, 1); else L(BB, 2); } while (0)
    if (bits == 4) LB(4); else if (bits == 5) LB(5); else LB(6);
#undef LB
#undef L
#undef L1
    return int(cudaGetLastError());
}
