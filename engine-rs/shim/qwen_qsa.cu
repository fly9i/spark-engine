// QSA (Qwen sparse attention) for Qwen3.8-Flash-Next: 24 q heads / 2 kv heads of 256 (first 64 dims
// NeoX-rotated, theta 1e7), Gemma RMSNorm (1 + w) on q and k, sigmoid output gate interleaved per head
// in q_proj ([q 256 | gate 256] x 24). KV cache FP8 E4M3 with one fp32 scale per (token, kv head).
// Indexer: 4 heads x 128 queries (Gemma norm + rope), one raw key per token pooled per 4-token group
// (mean -> bf16 -> Gemma norm -> rope at the group's first position), scores sum_h relu(q_h . k_g) / sqrt(128).
// Row r of a chain sits at position pos0 + r; rows attend causally to the selected tokens.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cfloat>
#include <cstdlib>
#include <cmath>
#include "qwen_exl3.cuh"

namespace qqsa {

constexpr int HQ = 24, HK = 2, D = 256, ROT = 64, IH = 4, ID = 128, RING = 32;

// inv_freq[0][i] = theta^(-2i/64) in double like the reference, computed once on the device by the same expression (FP64 exp /
// log per element were most of qsa_prep's time on GB10). inv_freq[1]: YaRN (factor 4 over the native 262144 positions, as the
// model card recommends for up to 1M tokens; HF _compute_yarn_parameters: beta_fast 32, beta_slow 1, truncated correction
// range), filled from the host; its cos / sin are scaled by rope_scale[1] = 0.1 ln 4 + 1. A sequence uses one mode for
// all its positions (its K cache and pooled keys are written with it).
__device__ double inv_freq[2][ROT / 2];
__device__ float rope_scale[2];
__global__ void inv_freq_init_kernel()
{
    const int i = threadIdx.x;
    if (i < ROT / 2) inv_freq[0][i] = exp(-(2.0 * i / ROT) * log(1e7));
}
__device__ __forceinline__ void rope_pair(float& a, float& b, int pos, int i, int rot, int mode = 0)
{
    // angle in fp32 from the double product (rot is always ROT)
    double inv = inv_freq[mode][i];
    float ang = (float) ((double) pos * inv);
    float c, s;
    sincosf(ang, &s, &c);
    if (mode) { c *= rope_scale[1]; s *= rope_scale[1]; }
    float x1 = a, x2 = b;
    a = x1 * c - x2 * s;
    b = x2 * c + x1 * s;
}

// mRoPE position of token `tok` on axis a (0 t, 1 h, 2 w; frequency pair i uses axis i % 3, the reference's interleaved
// sections [11, 11, 10]): the sequence's table for its multimodal prompt (mctl[0] tokens), else tok - mctl[1] (text after
// the prompt). Text-only sequences: no table (mctl null or {0, 0}), so the position is the token index.
__device__ __forceinline__ int rope_pos(int tok, int a, const int* __restrict__ mtab, const int* __restrict__ mctl)
{
    if (!mctl) return tok;
    return tok < mctl[0] ? mtab[(int64_t) tok * 3 + a] : tok - mctl[1];
}

__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// One warp per (row, head-slot): slots 0..23 q heads, 24..25 k heads, 26..29 indexer q heads, 30 raw idx key.
// qp [R, 12288] fp32 (q_proj), kp/vp [R, 512], ip [R, 640] (index_qk_proj).
// Outputs: q [R, 24, 256] fp32 (normed + roped), gate [R, 24, 256] fp32 (sigmoid applied),
// K/V cache rows written at pos (fp8 + scale), qi [R, 4, 128] fp32, raw ring [RING, 128] fp32.
// qpart (prefill, may be null): q_proj's unfinished fp16 output [R, 12288] (row stride ldqp) with its svh: each q-head warp
// finishes its 4 Hadamard blocks (q 256 | gate 256) into shared memory (exl3 finish_h: bitwise equal), qp unused.
__global__ __launch_bounds__(256) void prep_kernel(const float* __restrict__ qp, const float* __restrict__ kp, const float* __restrict__ vp,
                            const float* __restrict__ ip, int R, const int* __restrict__ pos0p,
                            const __nv_bfloat16* __restrict__ qn, const __nv_bfloat16* __restrict__ kn,
                            const __nv_bfloat16* __restrict__ iqn,
                            float* __restrict__ q, float* __restrict__ gate,
                            uint8_t* __restrict__ kc, float* __restrict__ ksc, uint8_t* __restrict__ vc, float* __restrict__ vsc,
                            float* __restrict__ qi, float* __restrict__ ring,
                            const half* __restrict__ qpart = nullptr, int64_t ldqp = 0, const half* __restrict__ qsvh = nullptr, int yarn = 0,
                            const int* __restrict__ mtab = nullptr, const int* __restrict__ mctl = nullptr)
{
    __shared__ float qsm[8][512];
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= R * 31) return;
    const int pos0 = *pos0p;
    int r = warp / 31, slot = warp % 31, pos = pos0 + r;
    if (slot < HQ || slot < HQ + HK)
    {
        bool isq = slot < HQ;
        int h = isq ? slot : slot - HQ;
        const float* src = isq ? qp + (int64_t) r * 12288 + h * 512 : kp + (int64_t) r * 512 + h * D;
        if (isq && qpart)
        {
            float* sm = qsm[threadIdx.x >> 5];
            #pragma unroll
            for (int b = 0; b < 4; ++b)
            {
                const uint2 u = *(const uint2*) (qpart + (int64_t) r * ldqp + (h * 4 + b) * 128 + lane * 4);
                const float2 x0 = __half22float2(*(const __half2*) &u.x), x1 = __half22float2(*(const __half2*) &u.y);
                float v4[4] = {0.f, 0.f, 0.f, 0.f};
                v4[0] += x0.x; v4[1] += x0.y; v4[2] += x1.x; v4[3] += x1.y;
                qexl3::fwht128(v4, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) sm[b * 128 + lane * 4 + i] = v4[i] * __half2float(qsvh[(h * 4 + b) * 128 + lane * 4 + i]);
            }
            __syncwarp();
            src = sm;
        }
        float v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = src[lane + 32 * i];   // element e = lane + 32 i
        float ss = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
        float inv = rsqrtf(warp_sum(ss) / D + 1e-6f);
        const __nv_bfloat16* w = isq ? qn : kn;
        #pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = v[i] * inv * (1.0f + __bfloat162float(w[lane + 32 * i]));
        // rope on e < 64: pairs (e, e + 32) -> i = 0 (e = lane) with i = 1 (e = lane + 32)
        rope_pair(v[0], v[1], rope_pos(pos, lane % 3, mtab, mctl), lane, ROT, yarn);
        if (isq)
        {
            float* dq = q + ((int64_t) r * HQ + h) * D;
            float* dg = gate + ((int64_t) r * HQ + h) * D;
            const float* gs = src + D;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                dq[lane + 32 * i] = v[i];
                dg[lane + 32 * i] = 1.0f / (1.0f + __expf(-gs[lane + 32 * i]));
            }
        }
        else
        {
            // K (normed, roped) and V (raw) to fp8 with per-(token, head) scale amax / 448
            const float* vs = vp + (int64_t) r * 512 + h * D;
            float vv[8], ka = 0.f, va = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { vv[i] = vs[lane + 32 * i]; ka = fmaxf(ka, fabsf(v[i])); va = fmaxf(va, fabsf(vv[i])); }
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) { ka = fmaxf(ka, __shfl_xor_sync(0xffffffffu, ka, m)); va = fmaxf(va, __shfl_xor_sync(0xffffffffu, va, m)); }
            float kscale = ka > 0.f ? ka / 448.0f : 1.0f, vscale = va > 0.f ? va / 448.0f : 1.0f;
            int64_t row = ((int64_t) pos * HK + h);
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                kc[row * D + lane + 32 * i] = __nv_fp8_e4m3(v[i] / kscale).__x;
                vc[row * D + lane + 32 * i] = __nv_fp8_e4m3(vv[i] / vscale).__x;
            }
            if (lane == 0) { ksc[row] = kscale; vsc[row] = vscale; }
        }
    }
    else if (slot < HQ + HK + IH)
    {
        int h = slot - HQ - HK;
        const float* src = ip + (int64_t) r * 640 + h * ID;
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = src[lane + 32 * i];
        float ss = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) ss += v[i] * v[i];
        float inv = rsqrtf(warp_sum(ss) / ID + 1e-6f);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = v[i] * inv * (1.0f + __bfloat162float(iqn[lane + 32 * i]));
        rope_pair(v[0], v[1], rope_pos(pos, lane % 3, mtab, mctl), lane, ROT, yarn);
        float* d = qi + ((int64_t) r * IH + h) * ID;
        #pragma unroll
        for (int i = 0; i < 4; ++i) d[lane + 32 * i] = v[i];
    }
    else if (r >= R - RING)       // the ring keeps the last RING positions (earlier rows would collide)
    {
        const float* src = ip + (int64_t) r * 640 + IH * ID;
        float* d = ring + (pos % RING) * ID;
        #pragma unroll
        for (int i = 0; i < 4; ++i) d[lane + 32 * i] = src[lane + 32 * i];
    }
}

// For rows closing a 4-token group: pooled[g] = rope_{4g}(GemmaRMS(bf16(mean(raw 4g..4g+3)))) as bf16.
__global__ void pool_kernel(const float* __restrict__ ring, const float* __restrict__ ip, int R, const int* __restrict__ pos0p,
                            const __nv_bfloat16* __restrict__ ikn, __nv_bfloat16* __restrict__ pooled, int yarn = 0,
                            const int* __restrict__ mtab = nullptr, const int* __restrict__ mctl = nullptr)
{
    int r = blockIdx.x, lane = threadIdx.x;
    const int pos0 = *pos0p;
    int pos = pos0 + r;
    if ((pos + 1) % 4) return;
    int g = pos / 4;
    float v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int e = lane + 32 * i;
        float s = 0.f;
        for (int t = 0; t < 4; ++t)
        {
            int tok = 4 * g + t;      // rows of this chunk read the projection, earlier ones the ring
            s += tok >= pos0 ? ip[(int64_t) (tok - pos0) * 640 + IH * ID + e] : ring[(tok % RING) * ID + e];
        }
        v[i] = __bfloat162float(__float2bfloat16(s * 0.25f));
    }
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) ss += v[i] * v[i];
    float inv = rsqrtf(warp_sum(ss) / ID + 1e-6f);
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = v[i] * inv * (1.0f + __bfloat162float(ikn[lane + 32 * i]));
    rope_pair(v[0], v[1], rope_pos(4 * g, lane % 3, mtab, mctl), lane, ROT, yarn);
    #pragma unroll
    for (int i = 0; i < 4; ++i) pooled[(int64_t) g * ID + lane + 32 * i] = __float2bfloat16(v[i]);
}

// Sparse attention. Row r attends to tokens sel[r][0..cnt[r]) (sel null: all tokens 0..pos).
// grid (R, HK, SPLITS), block 384 threads = 12 warps = the 12 q heads of this kv head.
// Each warp keeps an online softmax over its split's tokens; partials (m, l, acc[256]) to scratch.
constexpr int SPLIT_TOK = 128;
__global__ __launch_bounds__(384) void attn_kernel(
    const float* __restrict__ q, int R, const int* __restrict__ pos0p, const int* __restrict__ sel, const int* __restrict__ cnt, int sel_ld,
    const uint8_t* __restrict__ kc, const float* __restrict__ ksc, const uint8_t* __restrict__ vc, const float* __restrict__ vsc,
    float* __restrict__ part_ml, float* __restrict__ part_acc, int splits)
{
    int r = blockIdx.x, kh = blockIdx.y, sp = blockIdx.z;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = kh * 12 + warp;
    int pos = *pos0p + r;
    int n = sel ? cnt[r] : pos + 1;
    int t0 = sp * SPLIT_TOK, t1 = min(n, t0 + SPLIT_TOK);
    float qv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) qv[i] = q[((int64_t) r * HQ + h) * D + lane * 8 + i] * 0.0625f;   // 1/sqrt(256)
    float m = -FLT_MAX, l = 0.f, acc[8] = {};
    for (int t = t0; t < t1; ++t)
    {
        int tok = sel ? sel[(int64_t) r * sel_ld + t] : t;
        int64_t row = (int64_t) tok * HK + kh;
        const uint8_t* kr = kc + row * D + lane * 8;
        uint2 kb = *(const uint2*) kr;
        const uint8_t* kbytes = (const uint8_t*) &kb;
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { __nv_fp8_e4m3 f; f.__x = kbytes[i]; s += qv[i] * (float) f; }
        s = warp_sum(s) * ksc[row];
        float mn = fmaxf(m, s), corr = __expf(m - mn), p = __expf(s - mn);
        l = l * corr + p;
        uint2 vb = *(const uint2*) (vc + row * D + lane * 8);
        const uint8_t* vbytes = (const uint8_t*) &vb;
        float vs = vsc[row];
        #pragma unroll
        for (int i = 0; i < 8; ++i) { __nv_fp8_e4m3 f; f.__x = vbytes[i]; acc[i] = acc[i] * corr + p * (float) f * vs; }
        m = mn;
    }
    int64_t pi = ((int64_t) (r * HQ + h) * splits + sp);
    if (lane == 0) { part_ml[pi * 2] = m; part_ml[pi * 2 + 1] = l; }
    #pragma unroll
    for (int i = 0; i < 8; ++i) part_acc[pi * D + lane * 8 + i] = acc[i];
}

// attn_kernel with the tokens of a split taken G at a time: the G selected indices, K / V rows and scales are loaded
// first (independent loads in flight together), the G scores computed, then the online-softmax updates run token by
// token in order with the same expressions as attn_kernel (bitwise equal; QWEN_QSA_ATTN_V1=1 per launch: attn_kernel).
template <int G>
__global__ __launch_bounds__(384) void attn2_kernel(
    const float* __restrict__ q, int R, const int* __restrict__ pos0p, const int* __restrict__ sel, const int* __restrict__ cnt, int sel_ld,
    const uint8_t* __restrict__ kc, const float* __restrict__ ksc, const uint8_t* __restrict__ vc, const float* __restrict__ vsc,
    float* __restrict__ part_ml, float* __restrict__ part_acc, int splits)
{
    int r = blockIdx.x, kh = blockIdx.y, sp = blockIdx.z;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = kh * 12 + warp;
    int pos = *pos0p + r;
    int n = sel ? cnt[r] : pos + 1;
    int t0 = sp * SPLIT_TOK, t1 = min(n, t0 + SPLIT_TOK);
    float qv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) qv[i] = q[((int64_t) r * HQ + h) * D + lane * 8 + i] * 0.0625f;   // 1/sqrt(256)
    float m = -FLT_MAX, l = 0.f, acc[8] = {};
    const int* srow = sel ? sel + (int64_t) r * sel_ld : nullptr;
    for (int tb = t0; tb < t1; tb += G)
    {
        int64_t row[G];
        #pragma unroll
        for (int g = 0; g < G; ++g)
        {
            const int t = min(tb + g, t1 - 1);   // past the end: a valid token, result unused
            row[g] = (int64_t) (srow ? srow[t] : t) * HK + kh;
        }
        uint2 kb[G], vb[G];
        float ks[G], vs[G];
        #pragma unroll
        for (int g = 0; g < G; ++g)
        {
            kb[g] = *(const uint2*) (kc + row[g] * D + lane * 8);
            vb[g] = *(const uint2*) (vc + row[g] * D + lane * 8);
            ks[g] = ksc[row[g]];
            vs[g] = vsc[row[g]];
        }
        float sc[G];
        #pragma unroll
        for (int g = 0; g < G; ++g)
        {
            const uint8_t* kbytes = (const uint8_t*) &kb[g];
            float s = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { __nv_fp8_e4m3 f; f.__x = kbytes[i]; s += qv[i] * (float) f; }
            sc[g] = warp_sum(s) * ks[g];
        }
        #pragma unroll
        for (int g = 0; g < G; ++g)
        {
            if (tb + g >= t1) break;
            const float s = sc[g];
            float mn = fmaxf(m, s), corr = __expf(m - mn), p = __expf(s - mn);
            l = l * corr + p;
            const uint8_t* vbytes = (const uint8_t*) &vb[g];
            #pragma unroll
            for (int i = 0; i < 8; ++i) { __nv_fp8_e4m3 f; f.__x = vbytes[i]; acc[i] = acc[i] * corr + p * (float) f * vs[g]; }
            m = mn;
        }
    }
    int64_t pi = ((int64_t) (r * HQ + h) * splits + sp);
    if (lane == 0) { part_ml[pi * 2] = m; part_ml[pi * 2 + 1] = l; }
    #pragma unroll
    for (int i = 0; i < 8; ++i) part_acc[pi * D + lane * 8 + i] = acc[i];
}

// attn2 with the split's selected indices loaded once per warp (lane l holds entries l, l + 32, l + 64, l + 96) and
// passed by shuffles, and the next G tokens' K / V loads issued before the current G are used (bitwise equal).
template <int G>
__global__ __launch_bounds__(384) void attn3_kernel(
    const float* __restrict__ q, int R, const int* __restrict__ pos0p, const int* __restrict__ sel, const int* __restrict__ cnt, int sel_ld,
    const uint8_t* __restrict__ kc, const float* __restrict__ ksc, const uint8_t* __restrict__ vc, const float* __restrict__ vsc,
    float* __restrict__ part_ml, float* __restrict__ part_acc, int splits)
{
    int r = blockIdx.x, kh = blockIdx.y, sp = blockIdx.z;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = kh * 12 + warp;
    int pos = *pos0p + r;
    int n = sel ? cnt[r] : pos + 1;
    int t0 = sp * SPLIT_TOK, t1 = min(n, t0 + SPLIT_TOK);
    float m = -FLT_MAX, l = 0.f, acc[8] = {};
    if (t0 < t1)
    {
        int idx[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int t = min(t0 + lane + 32 * j, t1 - 1);
            idx[j] = sel ? sel[(int64_t) r * sel_ld + t] : t;
        }
        float qv[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = q[((int64_t) r * HQ + h) * D + lane * 8 + i] * 0.0625f;
        auto tok_of = [&](int t) {   // token of split entry t - t0 (warp-uniform t, clamped)
            const int e = min(t, t1 - 1) - t0;
            const int j = e >> 5;
            const int v = j == 0 ? idx[0] : j == 1 ? idx[1] : j == 2 ? idx[2] : idx[3];
            return __shfl_sync(0xffffffffu, v, e & 31);
        };
        uint2 kb[2][G], vb[2][G];
        float ks[2][G], vs[2][G];
        auto issue = [&](int tb, int b) {
            #pragma unroll
            for (int g = 0; g < G; ++g)
            {
                const int64_t row = (int64_t) tok_of(tb + g) * HK + kh;
                kb[b][g] = *(const uint2*) (kc + row * D + lane * 8);
                vb[b][g] = *(const uint2*) (vc + row * D + lane * 8);
                ks[b][g] = ksc[row];
                vs[b][g] = vsc[row];
            }
        };
        issue(t0, 0);
        int b = 0;
        for (int tb = t0; tb < t1; tb += G, b ^= 1)
        {
            if (tb + G < t1) { if (b) issue(tb + G, 0); else issue(tb + G, 1); }
            float sc[G];
            #pragma unroll
            for (int g = 0; g < G; ++g)
            {
                const uint8_t* kbytes = (const uint8_t*) (b ? &kb[1][g] : &kb[0][g]);
                float s = 0.f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) { __nv_fp8_e4m3 f; f.__x = kbytes[i]; s += qv[i] * (float) f; }
                sc[g] = warp_sum(s) * (b ? ks[1][g] : ks[0][g]);
            }
            #pragma unroll
            for (int g = 0; g < G; ++g)
            {
                if (tb + g >= t1) break;
                const float s = sc[g];
                float mn = fmaxf(m, s), corr = __expf(m - mn), p = __expf(s - mn);
                l = l * corr + p;
                const uint8_t* vbytes = (const uint8_t*) (b ? &vb[1][g] : &vb[0][g]);
                const float vsg = b ? vs[1][g] : vs[0][g];
                #pragma unroll
                for (int i = 0; i < 8; ++i) { __nv_fp8_e4m3 f; f.__x = vbytes[i]; acc[i] = acc[i] * corr + p * (float) f * vsg; }
                m = mn;
            }
        }
    }
    int64_t pi = ((int64_t) (r * HQ + h) * splits + sp);
    if (lane == 0) { part_ml[pi * 2] = m; part_ml[pi * 2 + 1] = l; }
    #pragma unroll
    for (int i = 0; i < 8; ++i) part_acc[pi * D + lane * 8 + i] = acc[i];
}

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *(uint32_t*) &h; }
__device__ __forceinline__ __half2 fp8x2(const uint8_t* p0, const uint8_t* p1)
{
    __nv_fp8_e4m3 a, b; a.__x = *p0; b.__x = *p1;
    return __halves2half2(__half(a), __half(b));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Tensor-core variant of attn_kernel for prefill chunks (same grid, same partial layout, so attn_combine_kernel
// is shared). Block 128 threads = 4 warps for one (row, kv head, split of 128 tokens); the 12 q heads are the
// rows of m16n8k16 tiles (padded to 16). QK^T: warp w scores tokens [32w, 32w + 32) over the 256 dims (q in
// fp16 * 1/16, K fp8 -> fp16, the per-token K scale applied to the score columns); softmax over the split in
// shared memory; PV: P (fp16) as A, V (fp8 staged in shared memory, scaled per token into fp16) as B, warp w
// owns dims [64w, 64w + 64).
__global__ __launch_bounds__(128) void attn_tc_kernel(
    const float* __restrict__ q, int R, const int* __restrict__ pos0p, const int* __restrict__ sel, const int* __restrict__ cnt, int sel_ld,
    const uint8_t* __restrict__ kc, const float* __restrict__ ksc, const uint8_t* __restrict__ vc, const float* __restrict__ vsc,
    float* __restrict__ part_ml, float* __restrict__ part_acc, int splits)
{
    constexpr int TS = SPLIT_TOK, SP = TS + 4;
    __shared__ __align__(16) uint8_t sV[TS * D];
    __shared__ float sS[16 * SP];
    __shared__ float sVs[TS], sKs[TS], sM[16], sL[16];
    __shared__ int sTok[TS];
    const int r = blockIdx.x, kh = blockIdx.y, sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qr = lane >> 2, qc = (lane & 3) * 2;
    const int pos = *pos0p + r;
    const int n = sel ? cnt[r] : pos + 1;
    const int t0 = sp * TS, nt = min(n, t0 + TS) - t0;
    if (nt <= 0)   // empty split: no weight in the combine (l = 0)
    {
        if (tid < 12)
        {
            int64_t pi = (int64_t) (r * HQ + kh * 12 + tid) * splits + sp;
            part_ml[pi * 2] = -FLT_MAX;
            part_ml[pi * 2 + 1] = 0.f;
        }
        return;
    }
    for (int i = tid; i < TS; i += 128)
    {
        int tok = i < nt ? (sel ? sel[(int64_t) r * sel_ld + t0 + i] : t0 + i) : -1;
        sTok[i] = tok;
        sKs[i] = tok >= 0 ? ksc[(int64_t) tok * HK + kh] : 0.f;
        sVs[i] = tok >= 0 ? vsc[(int64_t) tok * HK + kh] : 0.f;
    }
    __syncthreads();
    // V rows of the split -> shared (zeros past the end)
    for (int i = tid; i < TS * (D / 16); i += 128)
    {
        int t = i / (D / 16), c = i % (D / 16), tok = sTok[t];
        uint4 v = tok >= 0 ? *(const uint4*) (vc + ((int64_t) tok * HK + kh) * D + c * 16) : make_uint4(0, 0, 0, 0);
        *(uint4*) (sV + t * D + c * 16) = v;
    }
    // Q fragments (rows = heads, 12 valid), 16 k-steps of 16 dims
    uint32_t qa[16][4];
    {
        const float* q0 = q + ((int64_t) r * HQ + kh * 12 + qr) * D;
        const float* q1 = q + ((int64_t) r * HQ + kh * 12 + qr + 8) * D;
        const bool v1 = qr + 8 < 12;
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks)
        {
            int k = ks * 16 + qc;
            qa[ks][0] = h2u(__floats2half2_rn(q0[k] * 0.0625f, q0[k + 1] * 0.0625f));
            qa[ks][1] = v1 ? h2u(__floats2half2_rn(q1[k] * 0.0625f, q1[k + 1] * 0.0625f)) : 0u;
            qa[ks][2] = h2u(__floats2half2_rn(q0[k + 8] * 0.0625f, q0[k + 9] * 0.0625f));
            qa[ks][3] = v1 ? h2u(__floats2half2_rn(q1[k + 8] * 0.0625f, q1[k + 9] * 0.0625f)) : 0u;
        }
    }
    // S = Q K^T for tokens [32 warp, 32 warp + 32)
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int tb = warp * 32 + j * 8;
        const int tok = sTok[tb + qr];
        const uint8_t* kr = kc + ((int64_t) (tok >= 0 ? tok : 0) * HK + kh) * D;
        float c[4] = {};
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks)
        {
            int k = ks * 16 + qc;
            uint32_t b0 = tok >= 0 ? h2u(fp8x2(kr + k, kr + k + 1)) : 0u;
            uint32_t b1 = tok >= 0 ? h2u(fp8x2(kr + k + 8, kr + k + 9)) : 0u;
            mma16816(c, qa[ks], b0, b1);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int row = qr + (i >= 2 ? 8 : 0), col = tb + qc + (i & 1);
            sS[row * SP + col] = col < nt ? c[i] * sKs[col] : -FLT_MAX;
        }
    }
    __syncthreads();
    // softmax per head row (warp w: rows 4w..4w+3); P * 1 kept in fp32, stats to sM / sL
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int row = warp * 4 + i;
        float mx = -FLT_MAX;
        for (int c2 = lane; c2 < TS; c2 += 32) mx = fmaxf(mx, sS[row * SP + c2]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float l = 0.f;
        for (int c2 = lane; c2 < TS; c2 += 32)
        {
            float pv = c2 < nt ? __expf(sS[row * SP + c2] - mx) : 0.f;
            sS[row * SP + c2] = pv;
            l += pv;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (lane == 0) { sM[row] = nt > 0 ? mx : -FLT_MAX; sL[row] = l; }
    }
    __syncthreads();
    // O = P V for dims [64 warp, 64 warp + 64)
    float o[8][4] = {};
    #pragma unroll
    for (int kt = 0; kt < TS / 16; ++kt)
    {
        const int tk = kt * 16 + qc;
        uint32_t a[4];
        a[0] = h2u(__floats2half2_rn(sS[qr * SP + tk], sS[qr * SP + tk + 1]));
        a[1] = h2u(__floats2half2_rn(sS[(qr + 8) * SP + tk], sS[(qr + 8) * SP + tk + 1]));
        a[2] = h2u(__floats2half2_rn(sS[qr * SP + tk + 8], sS[qr * SP + tk + 9]));
        a[3] = h2u(__floats2half2_rn(sS[(qr + 8) * SP + tk + 8], sS[(qr + 8) * SP + tk + 9]));
        const __half2 s01 = __floats2half2_rn(sVs[tk], sVs[tk + 1]), s89 = __floats2half2_rn(sVs[tk + 8], sVs[tk + 9]);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            const int d = warp * 64 + j * 8 + qr;
            uint32_t b0 = h2u(__hmul2(fp8x2(sV + tk * D + d, sV + (tk + 1) * D + d), s01));
            uint32_t b1 = h2u(__hmul2(fp8x2(sV + (tk + 8) * D + d, sV + (tk + 9) * D + d), s89));
            mma16816(o[j], a, b0, b1);
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int row = qr + (i >= 2 ? 8 : 0), d = warp * 64 + j * 8 + qc + (i & 1);
            if (row < 12) part_acc[((int64_t) (r * HQ + kh * 12 + row) * splits + sp) * D + d] = o[j][i];
        }
    if (tid < 12)
    {
        int64_t pi = (int64_t) (r * HQ + kh * 12 + tid) * splits + sp;
        part_ml[pi * 2] = sM[tid];
        part_ml[pi * 2 + 1] = sL[tid];
    }
}

// attn_tc_kernel at about fp32 accuracy (decode rows): q (* 1/16) and P * (per-token V scale) split into fp16 hi + lo (two MMAs),
// K / V raw fp8 (exact in fp16) with the K scale on the scores and the V scale in P, and every MMA started from zero with its
// result added in fp32 (the tensor cores' own accumulation truncates small terms against a large accumulator). Same grid and
// partials as attn_tc_kernel.
__device__ __forceinline__ void split_h2(float a, float b, uint32_t& hi, uint32_t& lo)
{
    const __half2 h = __floats2half2_rn(a, b);
    const float2 f = __half22float2(h);
    hi = h2u(h);
    lo = h2u(__floats2half2_rn(a - f.x, b - f.y));
}
__global__ __launch_bounds__(128) void attn_tcp_kernel(
    const float* __restrict__ q, int R, const int* __restrict__ pos0p, const int* __restrict__ sel, const int* __restrict__ cnt, int sel_ld,
    const uint8_t* __restrict__ kc, const float* __restrict__ ksc, const uint8_t* __restrict__ vc, const float* __restrict__ vsc,
    float* __restrict__ part_ml, float* __restrict__ part_acc, int splits)
{
    constexpr int TS = SPLIT_TOK, SP = TS + 4;
    __shared__ __align__(16) uint8_t sV[TS * D];
    __shared__ float sS[16 * SP];
    __shared__ float sVs[TS], sKs[TS], sM[16], sL[16];
    __shared__ int sTok[TS];
    const int r = blockIdx.x, kh = blockIdx.y, sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qr = lane >> 2, qc = (lane & 3) * 2;
    const int pos = *pos0p + r;
    const int n = sel ? cnt[r] : pos + 1;
    const int t0 = sp * TS, nt = min(n, t0 + TS) - t0;
    if (nt <= 0)
    {
        if (tid < 12)
        {
            int64_t pi = (int64_t) (r * HQ + kh * 12 + tid) * splits + sp;
            part_ml[pi * 2] = -FLT_MAX;
            part_ml[pi * 2 + 1] = 0.f;
        }
        return;
    }
    for (int i = tid; i < TS; i += 128)
    {
        int tok = i < nt ? (sel ? sel[(int64_t) r * sel_ld + t0 + i] : t0 + i) : -1;
        sTok[i] = tok;
        sKs[i] = tok >= 0 ? ksc[(int64_t) tok * HK + kh] : 0.f;
        sVs[i] = tok >= 0 ? vsc[(int64_t) tok * HK + kh] : 0.f;
    }
    __syncthreads();
    for (int i = tid; i < TS * (D / 16); i += 128)
    {
        int t = i / (D / 16), c = i % (D / 16), tok = sTok[t];
        uint4 v = tok >= 0 ? *(const uint4*) (vc + ((int64_t) tok * HK + kh) * D + c * 16) : make_uint4(0, 0, 0, 0);
        *(uint4*) (sV + t * D + c * 16) = v;
    }
    // S = Q K^T for tokens [32 warp, 32 warp + 32)
    {
        const float* q0 = q + ((int64_t) r * HQ + kh * 12 + qr) * D;
        const float* q1 = q + ((int64_t) r * HQ + kh * 12 + min(qr + 8, 11)) * D;
        const bool v1 = qr + 8 < 12;
        const uint8_t* kr[4];
        bool kv[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int tok = sTok[warp * 32 + j * 8 + qr];
            kv[j] = tok >= 0;
            kr[j] = kc + ((int64_t) (tok >= 0 ? tok : 0) * HK + kh) * D;
        }
        float c[4][4] = {};
        #pragma unroll 4
        for (int ks = 0; ks < 16; ++ks)
        {
            const int k = ks * 16 + qc;
            const float2 x0 = *(const float2*) (q0 + k), x2 = *(const float2*) (q0 + k + 8);
            float2 x1 = *(const float2*) (q1 + k), x3 = *(const float2*) (q1 + k + 8);
            if (!v1) { x1 = make_float2(0.f, 0.f); x3 = x1; }
            uint32_t ah[4], al[4];
            split_h2(x0.x * 0.0625f, x0.y * 0.0625f, ah[0], al[0]);
            split_h2(x1.x * 0.0625f, x1.y * 0.0625f, ah[1], al[1]);
            split_h2(x2.x * 0.0625f, x2.y * 0.0625f, ah[2], al[2]);
            split_h2(x3.x * 0.0625f, x3.y * 0.0625f, ah[3], al[3]);
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const uint32_t b0 = kv[j] ? h2u(fp8x2(kr[j] + k, kr[j] + k + 1)) : 0u;
                const uint32_t b1 = kv[j] ? h2u(fp8x2(kr[j] + k + 8, kr[j] + k + 9)) : 0u;
                float th[4] = {}, tl[4] = {};
                mma16816(th, ah, b0, b1);
                mma16816(tl, al, b0, b1);
                #pragma unroll
                for (int i = 0; i < 4; ++i) c[j][i] += th[i] + tl[i];
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int row = qr + (i >= 2 ? 8 : 0), col = warp * 32 + j * 8 + qc + (i & 1);
                sS[row * SP + col] = col < nt ? c[j][i] * sKs[col] : -FLT_MAX;
            }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int row = warp * 4 + i;
        float mx = -FLT_MAX;
        for (int c2 = lane; c2 < TS; c2 += 32) mx = fmaxf(mx, sS[row * SP + c2]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float l = 0.f;
        for (int c2 = lane; c2 < TS; c2 += 32)
        {
            float pv = c2 < nt ? __expf(sS[row * SP + c2] - mx) : 0.f;
            sS[row * SP + c2] = pv;
            l += pv;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (lane == 0) { sM[row] = mx; sL[row] = l; }
    }
    __syncthreads();
    // O = (P * vscale) V for dims [64 warp, 64 warp + 64)
    float o[8][4] = {};
    #pragma unroll 2
    for (int kt = 0; kt < TS / 16; ++kt)
    {
        const int tk = kt * 16 + qc;
        uint32_t ah[4], al[4];
        split_h2(sS[qr * SP + tk] * sVs[tk], sS[qr * SP + tk + 1] * sVs[tk + 1], ah[0], al[0]);
        split_h2(sS[(qr + 8) * SP + tk] * sVs[tk], sS[(qr + 8) * SP + tk + 1] * sVs[tk + 1], ah[1], al[1]);
        split_h2(sS[qr * SP + tk + 8] * sVs[tk + 8], sS[qr * SP + tk + 9] * sVs[tk + 9], ah[2], al[2]);
        split_h2(sS[(qr + 8) * SP + tk + 8] * sVs[tk + 8], sS[(qr + 8) * SP + tk + 9] * sVs[tk + 9], ah[3], al[3]);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            const int d = warp * 64 + j * 8 + qr;
            const uint32_t b0 = h2u(fp8x2(sV + tk * D + d, sV + (tk + 1) * D + d));
            const uint32_t b1 = h2u(fp8x2(sV + (tk + 8) * D + d, sV + (tk + 9) * D + d));
            float th[4] = {}, tl[4] = {};
            mma16816(th, ah, b0, b1);
            mma16816(tl, al, b0, b1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[j][i] += th[i] + tl[i];
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int row = qr + (i >= 2 ? 8 : 0), d = warp * 64 + j * 8 + qc + (i & 1);
            if (row < 12) part_acc[((int64_t) (r * HQ + kh * 12 + row) * splits + sp) * D + d] = o[j][i];
        }
    if (tid < 12)
    {
        int64_t pi = (int64_t) (r * HQ + kh * 12 + tid) * splits + sp;
        part_ml[pi * 2] = sM[tid];
        part_ml[pi * 2 + 1] = sL[tid];
    }
}

// Prefill attention, flash style (default for R > 16; QWEN_PREFILL_ATTN_SPLIT=1: attn_tc_kernel + combine). One
// block of 128 threads per (row, kv head) walks the row's tokens in tiles of 64: K and V rows go to shared
// memory with cp.async, S = Q K^T on tensor cores (12 q heads as the rows of m16 tiles, q fp16 * 1/16, the
// per-token K scale on the score columns), online softmax per head (running max / sum, accumulators rescaled per
// tile, tiles in order: deterministic), O += P V (P fp16, V fp8 -> fp16 * per-token scale). Output = O / l * gate.
constexpr int FT = 64;
__device__ __forceinline__ void cpa16(void* s, const void* g)
{
    unsigned a = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(g));
}
__global__ __launch_bounds__(128) void attn_fl_kernel(
    const float* __restrict__ q, int R, const int* __restrict__ pos0p, const int* __restrict__ sel, const int* __restrict__ cnt, int sel_ld,
    const uint8_t* __restrict__ kc, const float* __restrict__ ksc, const uint8_t* __restrict__ vc, const float* __restrict__ vsc,
    const float* __restrict__ gate, float* __restrict__ out)
{
    constexpr int SP = FT + 4;
    __shared__ __align__(16) uint8_t sK[FT * D], sV[FT * D];
    __shared__ float sS[16 * SP];
    __shared__ float sKs[FT], sVs[FT], sM[16], sL[16], sC[16];
    __shared__ int sTok[FT];
    const int r = blockIdx.x, kh = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qr = lane >> 2, qc = (lane & 3) * 2;
    const int pos = *pos0p + r;
    const int n = sel ? cnt[r] : pos + 1;
    uint32_t qa[16][4];
    {
        const float* q0 = q + ((int64_t) r * HQ + kh * 12 + qr) * D;
        const float* q1 = q + ((int64_t) r * HQ + kh * 12 + qr + 8) * D;
        const bool v1 = qr + 8 < 12;
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks)
        {
            int k = ks * 16 + qc;
            qa[ks][0] = h2u(__floats2half2_rn(q0[k] * 0.0625f, q0[k + 1] * 0.0625f));
            qa[ks][1] = v1 ? h2u(__floats2half2_rn(q1[k] * 0.0625f, q1[k + 1] * 0.0625f)) : 0u;
            qa[ks][2] = h2u(__floats2half2_rn(q0[k + 8] * 0.0625f, q0[k + 9] * 0.0625f));
            qa[ks][3] = v1 ? h2u(__floats2half2_rn(q1[k + 8] * 0.0625f, q1[k + 9] * 0.0625f)) : 0u;
        }
    }
    if (tid < 16) { sM[tid] = -FLT_MAX; sL[tid] = 0.f; }
    float o[8][4] = {};
    for (int t0 = 0; t0 < n; t0 += FT)
    {
        const int nt = min(FT, n - t0);
        __syncthreads();   // previous tile fully consumed
        if (tid < FT)
        {
            int tok = tid < nt ? (sel ? sel[(int64_t) r * sel_ld + t0 + tid] : t0 + tid) : -1;
            sTok[tid] = tok;
            sKs[tid] = tok >= 0 ? ksc[(int64_t) tok * HK + kh] : 0.f;
            sVs[tid] = tok >= 0 ? vsc[(int64_t) tok * HK + kh] : 0.f;
        }
        __syncthreads();
        for (int i = tid; i < FT * (D / 16); i += 128)
        {
            int t = i / (D / 16), c = i % (D / 16), tok = sTok[t];
            if (tok >= 0)
            {
                cpa16(sK + t * D + c * 16, kc + ((int64_t) tok * HK + kh) * D + c * 16);
                cpa16(sV + t * D + c * 16, vc + ((int64_t) tok * HK + kh) * D + c * 16);
            }
            else { *(uint4*) (sK + t * D + c * 16) = make_uint4(0, 0, 0, 0); *(uint4*) (sV + t * D + c * 16) = make_uint4(0, 0, 0, 0); }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        // S for tokens [16 warp, 16 warp + 16)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const int tb = warp * 16 + j * 8;
            const uint8_t* kr = sK + (tb + qr) * D;
            float c[4] = {};
            #pragma unroll
            for (int ks = 0; ks < 16; ++ks)
            {
                int k = ks * 16 + qc;
                mma16816(c, qa[ks], h2u(fp8x2(kr + k, kr + k + 1)), h2u(fp8x2(kr + k + 8, kr + k + 9)));
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int row = qr + (i >= 2 ? 8 : 0), col = tb + qc + (i & 1);
                sS[row * SP + col] = col < nt ? c[i] * sKs[col] : -FLT_MAX;
            }
        }
        __syncthreads();
        // online softmax: warp w owns head rows 4w..4w+3
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int row = warp * 4 + i;
            float mx = -FLT_MAX;
            for (int c2 = lane; c2 < FT; c2 += 32) mx = fmaxf(mx, sS[row * SP + c2]);
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
            const float mo = sM[row], mn = fmaxf(mo, mx);
            float l = 0.f;
            for (int c2 = lane; c2 < FT; c2 += 32)
            {
                float pv = c2 < nt ? __expf(sS[row * SP + c2] - mn) : 0.f;
                sS[row * SP + c2] = pv;
                l += pv;
            }
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) l += __shfl_xor_sync(0xffffffffu, l, m);
            if (lane == 0)
            {
                const float corr = mo == -FLT_MAX ? 0.f : __expf(mo - mn);
                sC[row] = corr; sM[row] = mn; sL[row] = sL[row] * corr + l;
            }
        }
        __syncthreads();
        // O = O * corr + P V for dims [64 warp, 64 warp + 64)
        const float c0 = sC[qr], c1 = sC[qr + 8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) { o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1; }
        #pragma unroll
        for (int kt = 0; kt < FT / 16; ++kt)
        {
            const int tk = kt * 16 + qc;
            uint32_t a[4];
            a[0] = h2u(__floats2half2_rn(sS[qr * SP + tk], sS[qr * SP + tk + 1]));
            a[1] = h2u(__floats2half2_rn(sS[(qr + 8) * SP + tk], sS[(qr + 8) * SP + tk + 1]));
            a[2] = h2u(__floats2half2_rn(sS[qr * SP + tk + 8], sS[qr * SP + tk + 9]));
            a[3] = h2u(__floats2half2_rn(sS[(qr + 8) * SP + tk + 8], sS[(qr + 8) * SP + tk + 9]));
            const __half2 s01 = __floats2half2_rn(sVs[tk], sVs[tk + 1]), s89 = __floats2half2_rn(sVs[tk + 8], sVs[tk + 9]);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                const int d = warp * 64 + j * 8 + qr;
                uint32_t b0 = h2u(__hmul2(fp8x2(sV + tk * D + d, sV + (tk + 1) * D + d), s01));
                uint32_t b1 = h2u(__hmul2(fp8x2(sV + (tk + 8) * D + d, sV + (tk + 9) * D + d), s89));
                mma16816(o[j], a, b0, b1);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int row = qr + (i >= 2 ? 8 : 0), d = warp * 64 + j * 8 + qc + (i & 1);
            if (row < 12)
            {
                int64_t oi = ((int64_t) r * HQ + kh * 12 + row) * D + d;
                out[oi] = o[j][i] / sL[row] * gate[oi];
            }
        }
}

// out[r, h*256 + d] = gate * sum_s exp(m_s - M) acc_s / sum_s exp(m_s - M) l_s, splits in order.
__global__ void attn_combine_kernel(const float* __restrict__ part_ml, const float* __restrict__ part_acc,
                                    const float* __restrict__ gate, float* __restrict__ out, int splits)
{
    int rh = blockIdx.x, d = threadIdx.x;   // rh = r * 24 + h, 256 threads
    float M = -FLT_MAX;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part_ml[((int64_t) rh * splits + s) * 2]);
    float L = 0.f, A = 0.f;
    for (int s = 0; s < splits; ++s)
    {
        float m = part_ml[((int64_t) rh * splits + s) * 2], l = part_ml[((int64_t) rh * splits + s) * 2 + 1];
        if (l == 0.f) continue;
        float w = __expf(m - M);
        L += w * l;
        A += w * part_acc[((int64_t) rh * splits + s) * D + d];
    }
    out[(int64_t) rh * D + d] = A / L * gate[(int64_t) rh * D + d];
}


// ---- selection: scores over the visible pooled groups, top-512 set, token list ----
// scores[r][g] = sum_h relu(qi[r][h] . pooled[g]) / sqrt(128) for g < vis_r = (pos0 + r + 1) / 4
__global__ void score_kernel(const float* __restrict__ qi, const __nv_bfloat16* __restrict__ pooled, int R, const int* __restrict__ pos0p,
                             int ld, float* __restrict__ scores)
{
    __shared__ float qs[IH * ID];
    int r = blockIdx.y;
    for (int i = threadIdx.x; i < IH * ID; i += blockDim.x) qs[i] = qi[(int64_t) r * IH * ID + i];
    __syncthreads();
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int vis = (*pos0p + r + 1) / 4;
    if (g >= ld) return;
    if (g >= vis) { scores[(int64_t) r * ld + g] = -FLT_MAX; return; }
    const __nv_bfloat16* k = pooled + (int64_t) g * ID;
    float d[IH] = {};
    for (int e = 0; e < ID; e += 2)
    {
        float2 kv = __bfloat1622float2(*(const __nv_bfloat162*) (k + e));
        #pragma unroll
        for (int h = 0; h < IH; ++h) d[h] += qs[h * ID + e] * kv.x + qs[h * ID + e + 1] * kv.y;
    }
    float s = 0.f;
    #pragma unroll
    for (int h = 0; h < IH; ++h) s += fmaxf(d[h], 0.f);
    scores[(int64_t) r * ld + g] = s * 0.08838834764831845f;
}

// score_kernel on tensor cores (L1): per block 16 query rows x 64 groups, warp w = groups 16w..16w+15 (two n8 tiles) for the
// 4 indexer heads (4 m16 tiles: rows = query rows); K = 128 in 8 steps. qi (fp32) split into fp16 hi + lo (2 MMAs), pooled
// bf16 -> fp16 (exact in fp16's normal range). Scores of groups >= vis_r are not written (select reads g < vis_r only).
__device__ __forceinline__ uint32_t bf2h2(const __nv_bfloat16* p)
{
    const __nv_bfloat162 v = *(const __nv_bfloat162*) p;
    return h2u(__floats2half2_rn(__bfloat162float(v.x), __bfloat162float(v.y)));
}
__global__ __launch_bounds__(128) void score_tc_kernel(const float* __restrict__ qi, const __nv_bfloat16* __restrict__ pooled, int R,
                                                       const int* __restrict__ pos0p, int ld, float* __restrict__ scores)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
    const int r0 = blockIdx.y * 16, g0 = blockIdx.x * 64 + warp * 16;
    const int pos0 = *pos0p;
    const int vis_max = (pos0 + min(R, r0 + 16) - 1 + 1) / 4;
    if (g0 >= min(ld, vis_max)) return;
    float c[IH][2][4];
    #pragma unroll
    for (int h = 0; h < IH; ++h)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) c[h][nt][0] = c[h][nt][1] = c[h][nt][2] = c[h][nt][3] = 0.f;
    const int ra = min(r0 + gq, R - 1), rb = min(r0 + gq + 8, R - 1);
    const __nv_bfloat16* pg[2];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) pg[nt] = pooled + (int64_t) min(g0 + nt * 8 + gq, ld - 1) * ID;
    #pragma unroll 2
    for (int ks = 0; ks < ID / 16; ++ks)
    {
        const int k = ks * 16 + 2 * t;
        uint32_t b[2][2];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) { b[nt][0] = bf2h2(pg[nt] + k); b[nt][1] = bf2h2(pg[nt] + k + 8); }
        #pragma unroll
        for (int h = 0; h < IH; ++h)
        {
            const float* qa = qi + ((int64_t) ra * IH + h) * ID + k;
            const float* qb = qi + ((int64_t) rb * IH + h) * ID + k;
            const float2 x0 = *(const float2*) qa, x1 = *(const float2*) qb, x2 = *(const float2*) (qa + 8), x3 = *(const float2*) (qb + 8);
            uint32_t hi[4], lo[4];
            auto split = [](float2 v, uint32_t& hh, uint32_t& ll) {
                const __half2 h2 = __floats2half2_rn(v.x, v.y);
                const float2 hf = __half22float2(h2);
                hh = h2u(h2); ll = h2u(__floats2half2_rn(v.x - hf.x, v.y - hf.y));
            };
            split(x0, hi[0], lo[0]); split(x1, hi[1], lo[1]); split(x2, hi[2], lo[2]); split(x3, hi[3], lo[3]);
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) { mma16816(c[h][nt], hi, b[nt][0], b[nt][1]); mma16816(c[h][nt], lo, b[nt][0], b[nt][1]); }
        }
    }
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int r = r0 + gq + (i >= 2 ? 8 : 0), g = g0 + nt * 8 + 2 * t + (i & 1);
            if (r >= R || g >= ld) continue;
            if (g >= (pos0 + r + 1) / 4) continue;
            float sc = 0.f;
            #pragma unroll
            for (int h = 0; h < IH; ++h) sc += fmaxf(c[h][nt][i], 0.f);
            scores[(int64_t) r * ld + g] = sc * 0.08838834764831845f;
        }
}

__device__ __forceinline__ uint32_t okey(float f)
{
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // order-preserving
}

// one block (1024 threads) per row: token list sel[r][0..cnt) (selected groups ascending, then the tail)
__global__ __launch_bounds__(1024) void select_kernel(const float* __restrict__ scores, int R, const int* __restrict__ pos0p, int ld,
                                                      int* __restrict__ sel, int* __restrict__ cnt, int sel_ld)
{
    constexpr int K = 512;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t prefix, need_s;
    __shared__ int scan[1024];
    __shared__ int base_s;
    int r = blockIdx.x, tid = threadIdx.x;
    int pos = *pos0p + r, vis = (pos + 1) / 4;
    const float* sc = scores + (int64_t) r * ld;
    int* out = sel + (int64_t) r * sel_ld;
    if (vis <= K)
    {
        for (int t = tid; t <= pos; t += 1024) out[t] = t;
        if (tid == 0) cnt[r] = pos + 1;
        return;
    }
    // radix select: find key T with count(> T) < K <= count(>= T)
    uint32_t pre = 0, mask = 0, need = K;
    for (int shift = 24; shift >= 0; shift -= 8)
    {
        for (int i = tid; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (int g = tid; g < vis; g += 1024)
        {
            uint32_t k = okey(sc[g]);
            if ((k & mask) == pre) atomicAdd(&hist[(k >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0)
        {
            uint32_t acc = 0;
            int b = 255;
            for (; b > 0; --b) { if (acc + hist[b] >= need) break; acc += hist[b]; }
            prefix = pre | ((uint32_t) b << shift);
            need_s = need - acc;
        }
        __syncthreads();
        pre = prefix; need = need_s; mask |= 255u << shift;
        __syncthreads();
    }
    // pre = threshold key T; take all keys > T and the first `need` keys == T in index order
    if (tid == 0) base_s = 0;
    int eq_taken_total = 0;
    __syncthreads();
    for (int g0 = 0; g0 < vis; g0 += 1024)
    {
        int g = g0 + tid;
        uint32_t k = g < vis ? okey(sc[g]) : 0;
        int gt = g < vis && k > pre, eq = g < vis && k == pre;
        // exclusive scan of eq to rank ties in index order
        scan[tid] = eq;
        __syncthreads();
        for (int o = 1; o < 1024; o <<= 1) { int v = tid >= o ? scan[tid - o] : 0; __syncthreads(); scan[tid] += v; __syncthreads(); }
        int eq_rank = eq_taken_total + scan[tid] - eq;
        int take = gt || (eq && eq_rank < (int) need);
        int eq_chunk = scan[1023];
        __syncthreads();
        scan[tid] = take;
        __syncthreads();
        for (int o = 1; o < 1024; o <<= 1) { int v = tid >= o ? scan[tid - o] : 0; __syncthreads(); scan[tid] += v; __syncthreads(); }
        int slot = base_s + scan[tid] - take;
        if (take) { out[slot * 4 + 0] = 4 * g; out[slot * 4 + 1] = 4 * g + 1; out[slot * 4 + 2] = 4 * g + 2; out[slot * 4 + 3] = 4 * g + 3; }
        __syncthreads();
        if (tid == 0) base_s += scan[1023];
        eq_taken_total += eq_chunk;
        __syncthreads();
    }
    int n = base_s * 4;
    for (int t = 4 * vis + tid; t <= pos; t += 1024) out[n + t - 4 * vis] = t;
    if (tid == 0) cnt[r] = n + (pos + 1 - 4 * vis);
}

// select_kernel with the take pass counted by warp ballots (per 1024-group chunk: two ballots, per-warp counts in shared
// memory, prefix over the warps) instead of two 1024-wide block scans (about 40 barriers per chunk -> 4): the same keys,
// threshold and ties in index order, so the same token list (bitwise equal). QWEN_QSA_SELECT_V1=1 per launch: select_kernel.
__global__ __launch_bounds__(1024) void select2_kernel(const float* __restrict__ scores, int R, const int* __restrict__ pos0p, int ld,
                                                       int* __restrict__ sel, int* __restrict__ cnt, int sel_ld)
{
    constexpr int K = 512;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t prefix, need_s;
    __shared__ int weq[32], wtk[32];
    int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int pos = *pos0p + r, vis = (pos + 1) / 4;
    const float* sc = scores + (int64_t) r * ld;
    int* out = sel + (int64_t) r * sel_ld;
    if (vis <= K)
    {
        for (int t = tid; t <= pos; t += 1024) out[t] = t;
        if (tid == 0) cnt[r] = pos + 1;
        return;
    }
    uint32_t pre = 0, mask = 0, need = K;
    for (int shift = 24; shift >= 0; shift -= 8)
    {
        for (int i = tid; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        // bin counts aggregated per warp (the keys of similar scores share their top bytes: one shared atomic per
        // distinct bin and warp instead of one per lane on the same address)
        for (int g0 = 0; g0 < vis; g0 += 1024)
        {
            const int g = g0 + tid;
            uint32_t k = g < vis ? okey(sc[g]) : 0;
            const bool in = g < vis && (k & mask) == pre;
            const int bin = in ? (int) ((k >> shift) & 255) : -1;
            const uint32_t peers = __match_any_sync(0xffffffffu, bin);
            if (in && (peers & ((1u << lane) - 1u)) == 0) atomicAdd(&hist[bin], (uint32_t) __popc(peers));
        }
        __syncthreads();
        if (tid == 0)
        {
            uint32_t acc = 0;
            int b = 255;
            for (; b > 0; --b) { if (acc + hist[b] >= need) break; acc += hist[b]; }
            prefix = pre | ((uint32_t) b << shift);
            need_s = need - acc;
        }
        __syncthreads();
        pre = prefix; need = need_s; mask |= 255u << shift;
        __syncthreads();
    }
    const uint32_t lt = (1u << lane) - 1u;
    int eq_before = 0, base = 0;   // ties seen in earlier chunks, groups taken in earlier chunks (same in every thread)
    for (int g0 = 0; g0 < vis; g0 += 1024)
    {
        int g = g0 + tid;
        uint32_t k = g < vis ? okey(sc[g]) : 0;
        const bool gt = g < vis && k > pre, eq = g < vis && k == pre;
        const uint32_t be = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) weq[warp] = __popc(be);
        __syncthreads();
        int eq_w = 0, eq_all = 0;
        for (int w = 0; w < 32; ++w) { const int c = weq[w]; eq_w += w < warp ? c : 0; eq_all += c; }
        const int eq_rank = eq_before + eq_w + __popc(be & lt);
        const bool take = gt || (eq && eq_rank < (int) need);
        const uint32_t bt = __ballot_sync(0xffffffffu, take);
        if (lane == 0) wtk[warp] = __popc(bt);
        __syncthreads();
        int tk_w = 0, tk_all = 0;
        for (int w = 0; w < 32; ++w) { const int c = wtk[w]; tk_w += w < warp ? c : 0; tk_all += c; }
        if (take)
        {
            const int slot = base + tk_w + __popc(bt & lt);
            out[slot * 4 + 0] = 4 * g; out[slot * 4 + 1] = 4 * g + 1; out[slot * 4 + 2] = 4 * g + 2; out[slot * 4 + 3] = 4 * g + 3;
        }
        base += tk_all;
        eq_before += eq_all;
        __syncthreads();   // weq / wtk reused by the next chunk
    }
    int n = base * 4;
    for (int t = 4 * vis + tid; t <= pos; t += 1024) out[n + t - 4 * vis] = t;
    if (tid == 0) cnt[r] = n + (pos + 1 - 4 * vis);
}

}  // namespace qqsa

extern "C" {

static void rope_init(cudaStream_t st)
{
    static bool done = false;
    if (done) return;
    qqsa::inv_freq_init_kernel<<<1, 32, 0, st>>>();
    // YaRN table (host, double): HF _compute_yarn_parameters with dim 64, base 1e7, factor 4, original 262144
    const double base = 1e7, factor = 4.0, orig = 262144.0, dim = qqsa::ROT;
    auto corr = [&](double rot) { return dim * log(orig / (rot * 2.0 * M_PI)) / (2.0 * log(base)); };
    double low = floor(corr(32.0)), high = ceil(corr(1.0));
    low = low < 0 ? 0 : low; high = high > dim - 1 ? dim - 1 : high;
    if (low == high) high += 0.001;
    double inv[qqsa::ROT / 2];
    for (int i = 0; i < qqsa::ROT / 2; ++i)
    {
        const double pf = pow(base, 2.0 * i / dim), ext = 1.0 / pf, itp = 1.0 / (factor * pf);
        double ramp = (i - low) / (high - low);
        ramp = ramp < 0 ? 0 : ramp > 1 ? 1 : ramp;
        const double ef = 1.0 - ramp;   // extrapolation weight
        inv[i] = itp * (1.0 - ef) + ext * ef;
    }
    const float sc[2] = {1.0f, (float) (0.1 * log(factor) + 1.0)};
    cudaMemcpyToSymbolAsync(qqsa::inv_freq, inv, sizeof(inv), sizeof(inv), cudaMemcpyHostToDevice, st);
    cudaMemcpyToSymbolAsync(qqsa::rope_scale, sc, sizeof(sc), 0, cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);
    done = true;
}

// yarn: the sequence's RoPE mode (0 native, 1 YaRN)
int qwen_qsa_prep_m(const void* qp, const void* kp, const void* vp, const void* ip, int R, const void* pos0,
                    const void* qn, const void* kn, const void* iqn, void* q, void* gate,
                    void* kc, void* ksc, void* vc, void* vsc, void* qi, void* ring, int yarn, const void* mtab, const void* mctl, cudaStream_t st)
{
    rope_init(st);
    int warps = R * 31, threads = 256;
    qqsa::prep_kernel<<<(warps * 32 + threads - 1) / threads, threads, 0, st>>>(
        (const float*) qp, (const float*) kp, (const float*) vp, (const float*) ip, R, (const int*) pos0,
        (const __nv_bfloat16*) qn, (const __nv_bfloat16*) kn, (const __nv_bfloat16*) iqn,
        (float*) q, (float*) gate, (uint8_t*) kc, (float*) ksc, (uint8_t*) vc, (float*) vsc, (float*) qi, (float*) ring,
        nullptr, 0, nullptr, yarn, (const int*) mtab, (const int*) mctl);
    return (int) cudaGetLastError();
}
int qwen_qsa_prep_hm(const void* qpart, int64_t ldqp, const void* qsvh, const void* kp, const void* vp, const void* ip, int R, const void* pos0,
                     const void* qn, const void* kn, const void* iqn, void* q, void* gate,
                     void* kc, void* ksc, void* vc, void* vsc, void* qi, void* ring, int yarn, const void* mtab, const void* mctl,
                     cudaStream_t st)
{
    rope_init(st);
    int warps = R * 31, threads = 256;
    qqsa::prep_kernel<<<(warps * 32 + threads - 1) / threads, threads, 0, st>>>(
        nullptr, (const float*) kp, (const float*) vp, (const float*) ip, R, (const int*) pos0,
        (const __nv_bfloat16*) qn, (const __nv_bfloat16*) kn, (const __nv_bfloat16*) iqn,
        (float*) q, (float*) gate, (uint8_t*) kc, (float*) ksc, (uint8_t*) vc, (float*) vsc, (float*) qi, (float*) ring,
        (const half*) qpart, ldqp, (const half*) qsvh, yarn, (const int*) mtab, (const int*) mctl);
    return (int) cudaGetLastError();
}
int qwen_qsa_pool_m(const void* ring, const void* ip, int R, const void* pos0, const void* ikn, void* pooled, int yarn,
                    const void* mtab, const void* mctl, cudaStream_t st)
{
    rope_init(st);
    qqsa::pool_kernel<<<R, 32, 0, st>>>((const float*) ring, (const float*) ip, R, (const int*) pos0, (const __nv_bfloat16*) ikn,
                                         (__nv_bfloat16*) pooled, yarn, (const int*) mtab, (const int*) mctl);
    return (int) cudaGetLastError();
}

int qwen_qsa_prep(const void* qp, const void* kp, const void* vp, const void* ip, int R, const void* pos0,
                  const void* qn, const void* kn, const void* iqn, void* q, void* gate,
                  void* kc, void* ksc, void* vc, void* vsc, void* qi, void* ring, cudaStream_t st)
{
    rope_init(st);
    int warps = R * 31, threads = 256;
    qqsa::prep_kernel<<<(warps * 32 + threads - 1) / threads, threads, 0, st>>>(
        (const float*) qp, (const float*) kp, (const float*) vp, (const float*) ip, R, (const int*) pos0,
        (const __nv_bfloat16*) qn, (const __nv_bfloat16*) kn, (const __nv_bfloat16*) iqn,
        (float*) q, (float*) gate, (uint8_t*) kc, (float*) ksc, (uint8_t*) vc, (float*) vsc, (float*) qi, (float*) ring);
    return (int) cudaGetLastError();
}

int qwen_qsa_prep_h(const void* qpart, int64_t ldqp, const void* qsvh, const void* kp, const void* vp, const void* ip, int R, const void* pos0,
                    const void* qn, const void* kn, const void* iqn, void* q, void* gate,
                    void* kc, void* ksc, void* vc, void* vsc, void* qi, void* ring, cudaStream_t st)
{
    rope_init(st);
    int warps = R * 31, threads = 256;
    qqsa::prep_kernel<<<(warps * 32 + threads - 1) / threads, threads, 0, st>>>(
        nullptr, (const float*) kp, (const float*) vp, (const float*) ip, R, (const int*) pos0,
        (const __nv_bfloat16*) qn, (const __nv_bfloat16*) kn, (const __nv_bfloat16*) iqn,
        (float*) q, (float*) gate, (uint8_t*) kc, (float*) ksc, (uint8_t*) vc, (float*) vsc, (float*) qi, (float*) ring,
        (const half*) qpart, ldqp, (const half*) qsvh);
    return (int) cudaGetLastError();
}

int qwen_qsa_pool(const void* ring, const void* ip, int R, const void* pos0, const void* ikn, void* pooled, cudaStream_t st)
{
    rope_init(st);
    qqsa::pool_kernel<<<R, 32, 0, st>>>((const float*) ring, (const float*) ip, R, (const int*) pos0, (const __nv_bfloat16*) ikn, (__nv_bfloat16*) pooled);
    return (int) cudaGetLastError();
}

// splits must cover the longest row: splits * 128 >= max token count
int qwen_qsa_attn(const void* q, int R, const void* pos0, const void* sel, const void* cnt, int sel_ld,
                  const void* kc, const void* ksc, const void* vc, const void* vsc, const void* gate,
                  void* part_ml, void* part_acc, int splits, void* out, cudaStream_t st)
{
    dim3 grid(R, qqsa::HK, splits);
    if (R > 16)
    {
        const char* sp = getenv("QWEN_PREFILL_ATTN_SPLIT");
        const char* sc0 = getenv("QWEN_PREFILL_ATTN_SCALAR");
        if (!(sp && *sp == '1') && !(sc0 && *sc0 == '1'))
        {
            qqsa::attn_fl_kernel<<<dim3(R, qqsa::HK), 128, 0, st>>>((const float*) q, R, (const int*) pos0, (const int*) sel,
                (const int*) cnt, sel_ld, (const uint8_t*) kc, (const float*) ksc, (const uint8_t*) vc, (const float*) vsc,
                (const float*) gate, (float*) out);
            return (int) cudaGetLastError();
        }
    }
    // prefill chunks: tensor-core kernel (L1: fp16 q / P operands; 8.5% faster prefill, same accuracy against the
    // reference); QWEN_PREFILL_ATTN_SCALAR=1 (read per launch): the scalar kernel
    const char* sc = R > 16 ? getenv("QWEN_PREFILL_ATTN_SCALAR") : nullptr;
    if (R > 16 && !(sc && *sc == '1'))
        qqsa::attn_tc_kernel<<<grid, 128, 0, st>>>((const float*) q, R, (const int*) pos0, (const int*) sel, (const int*) cnt, sel_ld,
            (const uint8_t*) kc, (const float*) ksc, (const uint8_t*) vc, (const float*) vsc,
            (float*) part_ml, (float*) part_acc, splits);
    else
    {
        const char* v1 = getenv("QWEN_QSA_ATTN_V1");
        if (v1 && *v1 == '1')
            qqsa::attn_kernel<<<grid, 384, 0, st>>>((const float*) q, R, (const int*) pos0, (const int*) sel, (const int*) cnt, sel_ld,
                (const uint8_t*) kc, (const float*) ksc, (const uint8_t*) vc, (const float*) vsc,
                (float*) part_ml, (float*) part_acc, splits);
        else
        {
            // decode rows: tensor cores at about fp32 accuracy (attn_tcp_kernel, L1); QWEN_QSA_ATTN_VER=t: attn_tc_kernel (fp16
            // operands), 3: attn3, 2: attn2 (both bitwise equal to attn_kernel)
            const char* av = getenv("QWEN_QSA_ATTN_VER");
            if (av && av[0] == 't')   // the fp16-operand tensor-core kernel (L1, less accurate in decode)
                qqsa::attn_tc_kernel<<<grid, 128, 0, st>>>((const float*) q, R, (const int*) pos0, (const int*) sel, (const int*) cnt, sel_ld,
                    (const uint8_t*) kc, (const float*) ksc, (const uint8_t*) vc, (const float*) vsc,
                    (float*) part_ml, (float*) part_acc, splits);
            else if (!av || (av[0] != '2' && av[0] != '3'))
                qqsa::attn_tcp_kernel<<<grid, 128, 0, st>>>((const float*) q, R, (const int*) pos0, (const int*) sel, (const int*) cnt, sel_ld,
                    (const uint8_t*) kc, (const float*) ksc, (const uint8_t*) vc, (const float*) vsc,
                    (float*) part_ml, (float*) part_acc, splits);
            else if (av[0] == '2')
                qqsa::attn2_kernel<8><<<grid, 384, 0, st>>>((const float*) q, R, (const int*) pos0, (const int*) sel, (const int*) cnt, sel_ld,
                    (const uint8_t*) kc, (const float*) ksc, (const uint8_t*) vc, (const float*) vsc,
                    (float*) part_ml, (float*) part_acc, splits);
            else
                qqsa::attn3_kernel<8><<<grid, 384, 0, st>>>((const float*) q, R, (const int*) pos0, (const int*) sel, (const int*) cnt, sel_ld,
                    (const uint8_t*) kc, (const float*) ksc, (const uint8_t*) vc, (const float*) vsc,
                    (float*) part_ml, (float*) part_acc, splits);
        }
    }
    qqsa::attn_combine_kernel<<<R * qqsa::HQ, qqsa::D, 0, st>>>((const float*) part_ml, (const float*) part_acc,
        (const float*) gate, (float*) out, splits);
    return (int) cudaGetLastError();
}


// qi [R, 4, 128] fp32 (from qwen_qsa_prep), pooled [cap/4, 128] bf16 -> sel [R, sel_ld >= 2051], cnt [R];
// scores scratch [R, ld] fp32 with ld >= max visible group count.
int qwen_qsa_select(const void* qi, const void* pooled, int R, const void* pos0, void* scores, int ld, void* sel, void* cnt, int sel_ld, cudaStream_t st)
{
    // tensor-core scores (L1; every pooled group read once per 16 rows instead of once per row); QWEN_QSA_SCORE_V1=1 (read
    // per launch): score_kernel
    const char* sv1 = getenv("QWEN_QSA_SCORE_V1");
    if (sv1 && *sv1 == '1')
    {
        dim3 g1((ld + 255) / 256, R);
        qqsa::score_kernel<<<g1, 256, 0, st>>>((const float*) qi, (const __nv_bfloat16*) pooled, R, (const int*) pos0, ld, (float*) scores);
    }
    else
        qqsa::score_tc_kernel<<<dim3((ld + 63) / 64, (R + 15) / 16), 128, 0, st>>>((const float*) qi, (const __nv_bfloat16*) pooled, R,
                                                                                (const int*) pos0, ld, (float*) scores);
    const char* s1 = getenv("QWEN_QSA_SELECT_V1");
    if (s1 && *s1 == '1')
        qqsa::select_kernel<<<R, 1024, 0, st>>>((const float*) scores, R, (const int*) pos0, ld, (int*) sel, (int*) cnt, sel_ld);
    else
        qqsa::select2_kernel<<<R, 1024, 0, st>>>((const float*) scores, R, (const int*) pos0, ld, (int*) sel, (int*) cnt, sel_ld);
    return (int) cudaGetLastError();
}

}
