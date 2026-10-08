// Gated DeltaNet (Qwen3.8-Flash-Next linear attention): causal conv + sequential delta-rule recurrence.
//
// Per layer: 16 q/k heads, 48 v heads (v head j reads q/k head j/3), head dim 128, conv kernel 4 over
// the 10240 q|k|v channels. State S[j] is [v=128][k=128] fp32; one block per v head, thread i owns row
// S[j][i][:] in registers. Rows are processed in order (decode, verify chains and prefill alike):
//   S = exp(g) S;  v' = beta (v - S k);  S += v' k^T;  o = S q
// g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b), q,k L2-normalized (eps 1e-6), q * 128^-0.5.
// Output: RMSNorm_128(o) * w * sigmoid(z) (plain w, shared by all heads).
// Speculative chains: verify runs with write_state = 0 (the base state is only read); commit replays the
// accepted rows on the base state with write_state = 1 (GLM's correction-replay scheme).
// Segments (blockIdx.y): up to 8 sequences batched in one launch, rows [r0, r0 + T) with their own state; a commit
// may read its row counts from the device (nd[seg], capped at T) so one captured graph serves any acceptance.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include "qwen_exl3.cuh"

namespace qgdn {

constexpr int C = 10240, NV = 48, D = 128, MAXSEG = 8;

struct Segs { int n; int r0[MAXSEG]; int T[MAXSEG]; float* S[MAXSEG]; };

// Segment fields by a runtime index through a select chain: indexing the by-value parameter array directly puts it
// in local memory (stack frame + spills in the recurrence kernels).
template <typename T> __device__ __forceinline__ T pick(const T (&a)[MAXSEG], int g)
{
    return g == 0 ? a[0] : g == 1 ? a[1] : g == 2 ? a[2] : g == 3 ? a[3] : g == 4 ? a[4] : g == 5 ? a[5] : g == 6 ? a[6] : a[7];
}
__device__ __forceinline__ int seg_rows(const Segs& sg, const int* nd, int g)
{
    const int T = pick(sg.T, g);
    return nd ? min(max(nd[g], 0), T) : T;
}

// y[t, c] = silu(sum_j w[c, j] * x[t - 3 + j, c]); rows before 0 come from state[c][0..3) (oldest first).
// grid (C / 256, segments, row blocks of CONV_RT): no recurrence (outputs only read inputs), so rows run in parallel;
// each thread walks CONV_RT rows of one channel starting from the 3 inputs before its block.
constexpr int CONV_RT = 16;
__global__ void conv_kernel(const float* __restrict__ x, int64_t ldx, const Segs sg,
                            const __nv_bfloat16* __restrict__ w, float* __restrict__ y)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int g = blockIdx.y, T = pick(sg.T, g), ta = blockIdx.z * CONV_RT;
    if (ta >= T) return;
    x += (int64_t) pick(sg.r0, g) * ldx;
    y += (int64_t) pick(sg.r0, g) * C;
    const float* state = pick(sg.S, g);
    float w0 = __bfloat162float(w[c * 4 + 0]), w1 = __bfloat162float(w[c * 4 + 1]);
    float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
    auto in = [&](int t) { return t >= 0 ? x[(int64_t) t * ldx + c] : state[c * 3 + 3 + t]; };   // t = -3..-1: state
    float h0 = in(ta - 3), h1 = in(ta - 2), h2 = in(ta - 1);
    for (int t = ta; t < min(T, ta + CONV_RT); ++t)
    {
        float cur = x[(int64_t) t * ldx + c];
        float s = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
        y[(int64_t) t * C + c] = s / (1.0f + __expf(-s));
        h0 = h1; h1 = h2; h2 = cur;
    }
}

// state <- last 3 of (state ++ x[0..n))
__global__ void conv_commit_kernel(const float* __restrict__ x, int64_t ldx, const Segs sg, const int* __restrict__ nd)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y, n = seg_rows(sg, nd, g);
    if (c >= C || n <= 0) return;
    x += (int64_t) pick(sg.r0, g) * ldx;
    float* state = pick(sg.S, g);
    float h[3] = {state[c * 3 + 0], state[c * 3 + 1], state[c * 3 + 2]};
    for (int t = 0; t < n; ++t) { h[0] = h[1]; h[1] = h[2]; h[2] = x[t * ldx + c]; }
    state[c * 3 + 0] = h[0]; state[c * 3 + 1] = h[1]; state[c * 3 + 2] = h[2];
}

__device__ __forceinline__ float block_sum128(float v, float* red)
{
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    return (red[0] + red[1]) + (red[2] + red[3]);
}

__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

// grid (NV, segments) x 128 threads. y [T, 10240] conv output (q 0..2047 | k 2048..4095 | v 4096..),
// ab [T, 96] (a 0..47 | b 48..95), z [T, 6144]; out [T, 6144] (fp32, may be null when only committing).
__global__ __launch_bounds__(128) void recur_kernel(
    const float* __restrict__ y, const float* __restrict__ ab, int64_t ldab, const float* __restrict__ z, int64_t ldz,
    const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w, const Segs sg, const int* __restrict__ nd, float* __restrict__ out, int write_state)
{
    const int j = blockIdx.x, i = threadIdx.x, kh = j / 3, g = blockIdx.y;
    const int T = seg_rows(sg, nd, g);
    if (T <= 0) return;
    const int64_t r0 = pick(sg.r0, g);
    y += r0 * C; ab += r0 * ldab; z += r0 * ldz;
    if (out) out += r0 * (NV * D);
    float* S = pick(sg.S, g);
    __shared__ float qs[D], ks[D], red[4];
    float s[D];
    float* Sj = S + ((int64_t) j * D + i) * D;
    #pragma unroll
    for (int k = 0; k < D; k += 4)
    {
        float4 v = *(const float4*) (Sj + k);
        s[k] = v.x; s[k + 1] = v.y; s[k + 2] = v.z; s[k + 3] = v.w;
    }
    const float alog = __bfloat162float(A_log[j]), dtb = __bfloat162float(dt_bias[j]);
    const float nw = __bfloat162float(norm_w[i]);
    for (int t = 0; t < T; ++t)
    {
        const float* yt = y + (int64_t) t * C;
        float q = yt[kh * D + i], k = yt[2048 + kh * D + i], v = yt[4096 + j * D + i];
        float qn = block_sum128(q * q, red);
        float kn = block_sum128(k * k, red);
        __syncthreads();
        qs[i] = q * rsqrtf(qn + 1e-6f) * 0.08838834764831845f;
        ks[i] = k * rsqrtf(kn + 1e-6f);
        __syncthreads();
        float g = -__expf(alog) * softplus(ab[t * ldab + j] + dtb);
        float beta = 1.0f / (1.0f + __expf(-ab[t * ldab + NV + j]));
        float decay = __expf(g);
        float kv = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D; ++kk) { s[kk] *= decay; kv += s[kk] * ks[kk]; }
        float vp = beta * (v - kv);
        float o = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D; ++kk) { s[kk] += vp * ks[kk]; o += s[kk] * qs[kk]; }
        if (out)
        {
            float ms = block_sum128(o * o, red) * (1.0f / D);
            float zz = z[t * ldz + j * D + i];
            out[(int64_t) t * (NV * D) + j * D + i] = o * rsqrtf(ms + 1e-6f) * nw / (1.0f + __expf(-zz));
        }
    }
    if (write_state)
        #pragma unroll
        for (int k = 0; k < D; k += 4)
            *(float4*) (Sj + k) = make_float4(s[k], s[k + 1], s[k + 2], s[k + 3]);
}

// recur_kernel with the state block of head j (128 rows x 128, 64 KB) moved between global and registers through
// shared memory: coalesced 16-byte loads / stores (each thread otherwise walks its own 512-byte row: 32 rows per warp
// instruction). Rows padded to 132 floats (conflict-free 16-byte row reads). Same arithmetic (bitwise equal);
// QWEN_GDN_RECUR_V1=1 per launch: recur_kernel.
constexpr int SLD = D + 4;
__global__ __launch_bounds__(128) void recur2_kernel(
    const float* __restrict__ y, const float* __restrict__ ab, int64_t ldab, const float* __restrict__ z, int64_t ldz,
    const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w, const Segs sg, const int* __restrict__ nd, float* __restrict__ out, int write_state)
{
    extern __shared__ __align__(16) float st_sm[];   // [128][SLD]
    const int j = blockIdx.x, i = threadIdx.x, kh = j / 3, g = blockIdx.y;
    const int T = seg_rows(sg, nd, g);
    if (T <= 0) return;
    const int64_t r0 = pick(sg.r0, g);
    y += r0 * C; ab += r0 * ldab; z += r0 * ldz;
    if (out) out += r0 * (NV * D);
    float* S = pick(sg.S, g) + (int64_t) j * D * D;
    __shared__ float qs[D], ks[D], red[4];
    #pragma unroll 8
    for (int q = i; q < D * D / 4; q += 128)
    {
        const float4 v = *(const float4*) (S + 4 * q);
        *(float4*) (st_sm + (q >> 5) * SLD + (q & 31) * 4) = v;
    }
    __syncthreads();
    float s[D];
    #pragma unroll
    for (int k = 0; k < D; k += 4)
    {
        float4 v = *(const float4*) (st_sm + i * SLD + k);
        s[k] = v.x; s[k + 1] = v.y; s[k + 2] = v.z; s[k + 3] = v.w;
    }
    const float alog = __bfloat162float(A_log[j]), dtb = __bfloat162float(dt_bias[j]);
    const float nw = __bfloat162float(norm_w[i]);
    for (int t = 0; t < T; ++t)
    {
        const float* yt = y + (int64_t) t * C;
        float q = yt[kh * D + i], k = yt[2048 + kh * D + i], v = yt[4096 + j * D + i];
        float qn = block_sum128(q * q, red);
        float kn = block_sum128(k * k, red);
        __syncthreads();
        qs[i] = q * rsqrtf(qn + 1e-6f) * 0.08838834764831845f;
        ks[i] = k * rsqrtf(kn + 1e-6f);
        __syncthreads();
        float gg = -__expf(alog) * softplus(ab[t * ldab + j] + dtb);
        float beta = 1.0f / (1.0f + __expf(-ab[t * ldab + NV + j]));
        float decay = __expf(gg);
        float kv = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D; ++kk) { s[kk] *= decay; kv += s[kk] * ks[kk]; }
        float vp = beta * (v - kv);
        float o = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D; ++kk) { s[kk] += vp * ks[kk]; o += s[kk] * qs[kk]; }
        if (out)
        {
            float ms = block_sum128(o * o, red) * (1.0f / D);
            float zz = z[t * ldz + j * D + i];
            out[(int64_t) t * (NV * D) + j * D + i] = o * rsqrtf(ms + 1e-6f) * nw / (1.0f + __expf(-zz));
        }
    }
    if (write_state)
    {
        #pragma unroll
        for (int k = 0; k < D; k += 4) *(float4*) (st_sm + i * SLD + k) = make_float4(s[k], s[k + 1], s[k + 2], s[k + 3]);
        __syncthreads();
        #pragma unroll 8
        for (int q = i; q < D * D / 4; q += 128) *(float4*) (S + 4 * q) = *(const float4*) (st_sm + (q >> 5) * SLD + (q & 31) * 4);
    }
}

// ---- prefill: the qkv / z projections' finish (had128 * svh, from the fp16 GEMM output) fused into their consumers ----
// The finished value of column n of row t: v = 0 + part[t][n] (fp32), fwht128 over the column's 128-block (lane l holds
// columns 4l..4l+3), * svh[n] -- exactly exl3 finish_h, so bitwise equal to finishing first.
__device__ __forceinline__ void finish_h4(const half* __restrict__ prow, const half* __restrict__ svh, int blk, int lane, float* v)
{
    const uint2 u = *(const uint2*) (prow + blk * 128 + lane * 4);
    const float2 a = __half22float2(*(const __half2*) &u.x), b = __half22float2(*(const __half2*) &u.y);
    v[0] = 0.f; v[1] = 0.f; v[2] = 0.f; v[3] = 0.f;
    v[0] += a.x; v[1] += a.y; v[2] += b.x; v[3] += b.y;
    qexl3::fwht128(v, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = v[i] * __half2float(svh[blk * 128 + lane * 4 + i]);
}
// conv_kernel on the unfinished qkv projection part [T, 10240] fp16 (row stride ldp): grid (C / 512, segments, row blocks
// of CONV_RT) x 128; warp = one 128-channel block, its rows finished in registers (3 rows before the block recomputed).
// block_sum128 (warp xor trees over elements 32w..32w+31, then (S0 + S1) + (S2 + S3)) for the layout of conv_h_kernel (lane l
// holds elements 4l..4l+3): the same additions in the same order (bitwise equal)
__device__ __forceinline__ float sum128_l4(const float* e)
{
    float v[4] = {e[0], e[1], e[2], e[3]};
    #pragma unroll
    for (int m = 4; m >= 1; m >>= 1)   // element xor 16, 8, 4 = lane xor 4, 2, 1
        #pragma unroll
        for (int u = 0; u < 4; ++u) v[u] += __shfl_xor_sync(0xffffffffu, v[u], m);
    float w2[4] = {v[0] + v[2], v[1] + v[3], v[2] + v[0], v[3] + v[1]};   // element xor 2
    const float s = w2[0] + w2[1];                                         // element xor 1 (all four equal)
    const float s0 = __shfl_sync(0xffffffffu, s, 0), s1 = __shfl_sync(0xffffffffu, s, 8);
    const float s2 = __shfl_sync(0xffffffffu, s, 16), s3 = __shfl_sync(0xffffffffu, s, 24);
    return (s0 + s1) + (s2 + s3);
}
// QK: also the recurrence's prep (prep_kernel, fused): the q / k blocks (channels < 4096) write normalized q * 128^-0.5 and k
// to qk [rows, 16, 256] instead of y, the k blocks also decay / beta of their 3 v heads to db (bitwise equal to prep_kernel).
template <bool QK>
__global__ __launch_bounds__(128) void conv_h_kernel(const half* __restrict__ part, int64_t ldp, const half* __restrict__ svh, const Segs sg,
                                                     const __nv_bfloat16* __restrict__ w, float* __restrict__ y,
                                                     const float* __restrict__ ab = nullptr, int64_t ldab = 0,
                                                     const __nv_bfloat16* __restrict__ A_log = nullptr,
                                                     const __nv_bfloat16* __restrict__ dt_bias = nullptr, float* __restrict__ qk = nullptr,
                                                     float* __restrict__ db = nullptr)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, blk = blockIdx.x * 4 + warp;
    const int g = blockIdx.y, T = pick(sg.T, g), ta = blockIdx.z * CONV_RT;
    if (ta >= T) return;
    part += (int64_t) pick(sg.r0, g) * ldp;
    y += (int64_t) pick(sg.r0, g) * C;
    const float* state = pick(sg.S, g);
    float w0[4], w1[4], w2[4], w3[4], h0[4], h1[4], h2[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int c = blk * 128 + lane * 4 + i;
        w0[i] = __bfloat162float(w[c * 4 + 0]); w1[i] = __bfloat162float(w[c * 4 + 1]);
        w2[i] = __bfloat162float(w[c * 4 + 2]); w3[i] = __bfloat162float(w[c * 4 + 3]);
    }
    auto in = [&](int t, float* v) {   // finished row t (t >= 0) or state (t = -3..-1)
        if (t >= 0) finish_h4(part + (int64_t) t * ldp, svh, blk, lane, v);
        else
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = state[(blk * 128 + lane * 4 + i) * 3 + 3 + t];
    };
    in(ta - 3, h0); in(ta - 2, h1); in(ta - 1, h2);
    const int64_t row0 = pick(sg.r0, g);
    for (int t = ta; t < min(T, ta + CONV_RT); ++t)
    {
        float cur[4], o[4];
        in(t, cur);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float s = w0[i] * h0[i] + w1[i] * h1[i] + w2[i] * h2[i] + w3[i] * cur[i];
            o[i] = s / (1.0f + __expf(-s));
            h0[i] = h1[i]; h1[i] = h2[i]; h2[i] = cur[i];
        }
        if (!QK || blk >= 32)
        {
            #pragma unroll
            for (int i = 0; i < 4; ++i) y[(int64_t) t * C + blk * 128 + lane * 4 + i] = o[i];
            continue;
        }
        const bool isk = blk >= 16;
        const int kh = blk & 15;
        const int64_t row = row0 + t;
        float sq[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) sq[i] = o[i] * o[i];
        const float nrm = sum128_l4(sq);
        float* dst = qk + (row * 16 + kh) * 2 * D + (isk ? D : 0) + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = isk ? o[i] * rsqrtf(nrm + 1e-6f) : o[i] * rsqrtf(nrm + 1e-6f) * 0.08838834764831845f;
        if (isk && lane < 3)
        {
            const int j = kh * 3 + lane;
            float gg = -__expf(__bfloat162float(A_log[j])) * softplus(ab[row * ldab + j] + __bfloat162float(dt_bias[j]));
            db[(row * NV + j) * 2] = __expf(gg);
            db[(row * NV + j) * 2 + 1] = 1.0f / (1.0f + __expf(-ab[row * ldab + NV + j]));
        }
    }
}
// conv_commit_kernel (all rows) on the unfinished part: state <- last 3 of (state ++ finished rows)
__global__ __launch_bounds__(128) void conv_commit_h_kernel(const half* __restrict__ part, int64_t ldp, const half* __restrict__ svh, const Segs sg)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, blk = blockIdx.x * 4 + warp, g = blockIdx.y;
    const int n = pick(sg.T, g);
    if (n <= 0) return;
    part += (int64_t) pick(sg.r0, g) * ldp;
    float* state = pick(sg.S, g);
    float h[3][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int q = 0; q < 3; ++q) h[q][i] = state[(blk * 128 + lane * 4 + i) * 3 + q];
    for (int t = max(0, n - 3); t < n; ++t)
    {
        float v[4];
        finish_h4(part + (int64_t) t * ldp, svh, blk, lane, v);
        #pragma unroll
        for (int i = 0; i < 4; ++i) { h[0][i] = h[1][i]; h[1][i] = h[2][i]; h[2][i] = v[i]; }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int q = 0; q < 3; ++q) state[(blk * 128 + lane * 4 + i) * 3 + q] = h[q][i];
}
// onorm_kernel with z from its unfinished part [R, 6144] fp16: v head j = z's 128-block j, finished by warp 0
__global__ __launch_bounds__(128) void onorm_h_kernel(float* __restrict__ out, const half* __restrict__ zp, int64_t ldz, const half* __restrict__ zsvh,
                                                      const __nv_bfloat16* __restrict__ norm_w)
{
    __shared__ float red[4], zs[D];
    const int t = blockIdx.x, j = blockIdx.y, i = threadIdx.x;
    if (i < 32)
    {
        float v[4];
        finish_h4(zp + (int64_t) t * ldz, zsvh, j, i, v);
        #pragma unroll
        for (int q = 0; q < 4; ++q) zs[i * 4 + q] = v[q];
    }
    float* o = out + (int64_t) t * (NV * D) + j * D;
    float ov = o[i];
    float ms = block_sum128(ov * ov, red) * (1.0f / D);   // (its barriers also publish zs)
    float zz = zs[i];
    o[i] = ov * rsqrtf(ms + 1e-6f) * __bfloat162float(norm_w[i]) / (1.0f + __expf(-zz));
}

// ---- row-parallel recurrence for prefill chunks (QWEN_GDN_V1=1 per launch: recur_kernel above) ----
// Rows of S never interact: row i only needs the shared k_t, q_t, decay and beta of its head. So:
//   prep_kernel   grid (R, 16 k heads) x 128: normalized q (* 128^-0.5), k; decay = exp(g), beta for the 3 v heads
//   rows_kernel   grid (48 v heads, 4 row groups, segments) x 32: lane owns one S row, tokens in order, raw o
//   onorm_kernel  grid (R, 48) x 128: out = RMSNorm_128(o) * w * sigmoid(z) in place
// Same operations and reduction orders as recur_kernel (bitwise equal results).
__global__ __launch_bounds__(128) void prep_kernel(const float* __restrict__ y, const float* __restrict__ ab, int64_t ldab,
                                                   const __nv_bfloat16* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
                                                   float* __restrict__ qk, float* __restrict__ db)
{
    __shared__ float red[4];
    const int t = blockIdx.x, kh = blockIdx.y, i = threadIdx.x;
    const float* yt = y + (int64_t) t * C;
    float q = yt[kh * D + i], k = yt[2048 + kh * D + i];
    float qn = block_sum128(q * q, red);
    float kn = block_sum128(k * k, red);
    qk[((int64_t) t * 16 + kh) * 2 * D + i] = q * rsqrtf(qn + 1e-6f) * 0.08838834764831845f;
    qk[((int64_t) t * 16 + kh) * 2 * D + D + i] = k * rsqrtf(kn + 1e-6f);
    if (i < 3)
    {
        int j = kh * 3 + i;
        float g = -__expf(__bfloat162float(A_log[j])) * softplus(ab[t * ldab + j] + __bfloat162float(dt_bias[j]));
        db[((int64_t) t * NV + j) * 2] = __expf(g);
        db[((int64_t) t * NV + j) * 2 + 1] = 1.0f / (1.0f + __expf(-ab[t * ldab + NV + j]));
    }
}

__global__ __launch_bounds__(32) void rows_kernel(const float* __restrict__ y, const float* __restrict__ qk, const float* __restrict__ db,
                                                  const Segs sg, const int* __restrict__ nd, float* __restrict__ out, int write_state)
{
    __shared__ __align__(16) float sq[D], sk[D];
    const int j = blockIdx.x, kh = j / 3, g = blockIdx.z, lane = threadIdx.x, i = blockIdx.y * 32 + lane;
    const int T = seg_rows(sg, nd, g);
    if (T <= 0) return;
    const int64_t r0 = pick(sg.r0, g);
    float* Sj = pick(sg.S, g) + ((int64_t) j * D + i) * D;
    float s[D];
    #pragma unroll
    for (int k = 0; k < D; k += 4)
    {
        float4 v = *(const float4*) (Sj + k);
        s[k] = v.x; s[k + 1] = v.y; s[k + 2] = v.z; s[k + 3] = v.w;
    }
    // token t + 1's inputs are loaded while token t computes (one warp per scheduler: nothing else hides latency)
    auto fetch = [&](int t, float4& q4, float4& k4, float2& d2, float& v)
    {
        const int64_t row = r0 + t;
        const float* src = qk + (row * 16 + kh) * 2 * D;
        q4 = *(const float4*) (src + lane * 4);
        k4 = *(const float4*) (src + D + lane * 4);
        d2 = *(const float2*) (db + (row * NV + j) * 2);
        v = y[row * C + 4096 + j * D + i];
    };
    float4 nq, nk;
    float2 nd2;
    float nv;
    fetch(0, nq, nk, nd2, nv);
    for (int t = 0; t < T; ++t)
    {
        const int64_t row = r0 + t;
        __syncwarp();
        *(float4*) (sq + lane * 4) = nq;
        *(float4*) (sk + lane * 4) = nk;
        __syncwarp();
        const float decay = nd2.x, beta = nd2.y, v = nv;
        if (t + 1 < T) fetch(t + 1, nq, nk, nd2, nv);
        // compiler barriers every 32 columns keep the shared k / q reads from being hoisted into registers
        // all at once (with the 128-float row that overflows 255 registers)
        float kv = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D; ++kk)
        {
            if (kk % 32 == 0) asm volatile("" ::: "memory");
            s[kk] *= decay; kv += s[kk] * sk[kk];
        }
        float vp = beta * (v - kv);
        float o = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D; ++kk)
        {
            if (kk % 32 == 0) asm volatile("" ::: "memory");
            s[kk] += vp * sk[kk]; o += s[kk] * sq[kk];
        }
        if (out) out[row * (NV * D) + j * D + i] = o;
    }
    if (write_state)
        #pragma unroll
        for (int k = 0; k < D; k += 4)
            *(float4*) (Sj + k) = make_float4(s[k], s[k + 1], s[k + 2], s[k + 3]);
}

// L1 variant of rows_kernel (default; QWEN_GDN_ROWS1=1 per launch: rows_kernel): LPR lanes share one S row, lane
// part p holding columns p, p + LPR, ... (interleaved: the parts read adjacent shared words, no bank conflicts); the row dot products k.S and q.S are summed per lane and then across the LPR lanes
// (xor shuffles). Shorter dependency chains, no register spills, LPR x more warps.
constexpr int LPR = 4, CPL = D / LPR;
__global__ __launch_bounds__(32) void rows4_kernel(const float* __restrict__ y, const float* __restrict__ qk, const float* __restrict__ db,
                                                   const Segs sg, const int* __restrict__ nd, float* __restrict__ out, int write_state)
{
    // q / k regrouped per part: part p's columns (p, p + LPR, ...) contiguous at p * PS (PS = CPL + 4: the parts'
    // 16-byte reads land in different banks), read 4 columns per 128-bit load
    constexpr int PS = CPL + 4;
    __shared__ __align__(16) float sq[LPR * PS], sk[LPR * PS];
    const int j = blockIdx.x, kh = j / 3, g = blockIdx.z, lane = threadIdx.x;
    const int i = blockIdx.y * (32 / LPR) + lane / LPR, part = lane % LPR;
    const int T = seg_rows(sg, nd, g);
    if (T <= 0) return;
    const int64_t r0 = pick(sg.r0, g);
    float* Sj = pick(sg.S, g) + ((int64_t) j * D + i) * D + part;
    float s[CPL];
    #pragma unroll
    for (int k = 0; k < CPL; ++k) s[k] = Sj[k * LPR];
    auto fetch = [&](int t, float4& q4, float4& k4, float2& d2, float& v)
    {
        const int64_t row = r0 + t;
        const float* src = qk + (row * 16 + kh) * 2 * D;
        q4 = *(const float4*) (src + lane * 4);
        k4 = *(const float4*) (src + D + lane * 4);
        d2 = *(const float2*) (db + (row * NV + j) * 2);
        v = y[row * C + 4096 + j * D + i];
    };
    float4 nq, nk;
    float2 nd2;
    float nv;
    fetch(0, nq, nk, nd2, nv);
    for (int t = 0; t < T; ++t)
    {
        const int64_t row = r0 + t;
        __syncwarp();
        {
            // lane holds columns 4 lane .. 4 lane + 3 = parts 0..3 of column group lane
            const float qv[4] = {nq.x, nq.y, nq.z, nq.w}, kv4[4] = {nk.x, nk.y, nk.z, nk.w};
            #pragma unroll
            for (int u = 0; u < 4; ++u) { sq[u * PS + lane] = qv[u]; sk[u * PS + lane] = kv4[u]; }
        }
        __syncwarp();
        const float decay = nd2.x, beta = nd2.y, v = nv;
        if (t + 1 < T) fetch(t + 1, nq, nk, nd2, nv);
        const float* skp = sk + part * PS;
        const float* sqp = sq + part * PS;
        float kv = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < CPL; kk += 4)
        {
            const float4 k4 = *(const float4*) (skp + kk);
            const float ka[4] = {k4.x, k4.y, k4.z, k4.w};
            #pragma unroll
            for (int u = 0; u < 4; ++u) { s[kk + u] *= decay; kv += s[kk + u] * ka[u]; }
        }
        #pragma unroll
        for (int m = 1; m < LPR; m <<= 1) kv += __shfl_xor_sync(0xffffffffu, kv, m);
        float vp = beta * (v - kv);
        float o = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < CPL; kk += 4)
        {
            const float4 k4 = *(const float4*) (skp + kk), q4 = *(const float4*) (sqp + kk);
            const float ka[4] = {k4.x, k4.y, k4.z, k4.w}, qa[4] = {q4.x, q4.y, q4.z, q4.w};
            #pragma unroll
            for (int u = 0; u < 4; ++u) { s[kk + u] += vp * ka[u]; o += s[kk + u] * qa[u]; }
        }
        #pragma unroll
        for (int m = 1; m < LPR; m <<= 1) o += __shfl_xor_sync(0xffffffffu, o, m);
        if (out && part == 0) out[row * (NV * D) + j * D + i] = o;
    }
    if (write_state)
        #pragma unroll
        for (int k = 0; k < CPL; ++k) Sj[k * LPR] = s[k];
}

__global__ __launch_bounds__(128) void onorm_kernel(float* __restrict__ out, const float* __restrict__ z, int64_t ldz,
                                                    const __nv_bfloat16* __restrict__ norm_w)
{
    __shared__ float red[4];
    const int t = blockIdx.x, j = blockIdx.y, i = threadIdx.x;
    float* o = out + (int64_t) t * (NV * D) + j * D;
    float ov = o[i];
    float ms = block_sum128(ov * ov, red) * (1.0f / D);
    float zz = z[t * ldz + j * D + i];
    o[i] = ov * rsqrtf(ms + 1e-6f) * __bfloat162float(norm_w[i]) / (1.0f + __expf(-zz));
}

}  // namespace qgdn

static qgdn::Segs one(void* S, int T) { qgdn::Segs sg{}; sg.n = 1; sg.r0[0] = 0; sg.T[0] = T; sg.S[0] = (float*) S; return sg; }
static int segs(qgdn::Segs& sg, int n, const int* r0, const int* T, void* const* S)
{
    if (n < 1 || n > qgdn::MAXSEG) return -1;
    sg = qgdn::Segs{}; sg.n = n;
    for (int g = 0; g < n; ++g) { sg.r0[g] = r0[g]; sg.T[g] = T[g]; sg.S[g] = (float*) S[g]; }
    return 0;
}

extern "C" {

int qwen_gdn_conv(const void* x, int64_t ldx, const void* state, const void* w, void* y, int T, cudaStream_t st)
{
    qgdn::conv_kernel<<<dim3(qgdn::C / 256, 1, (T + qgdn::CONV_RT - 1) / qgdn::CONV_RT), 256, 0, st>>>((const float*) x, ldx,
        one((void*) state, T), (const __nv_bfloat16*) w, (float*) y);
    return (int) cudaGetLastError();
}

int qwen_gdn_conv_commit(const void* x, int64_t ldx, void* state, int n, cudaStream_t st)
{
    qgdn::conv_commit_kernel<<<qgdn::C / 256, 256, 0, st>>>((const float*) x, ldx, one(state, n), nullptr);
    return (int) cudaGetLastError();
}

int qwen_gdn_recur(const void* y, const void* ab, int64_t ldab, const void* z, int64_t ldz, const void* A_log,
                   const void* dt_bias, const void* norm_w, void* S, void* out, int T, int write_state, cudaStream_t st)
{
    qgdn::recur_kernel<<<qgdn::NV, 128, 0, st>>>((const float*) y, (const float*) ab, ldab, (const float*) z, ldz,
        (const __nv_bfloat16*) A_log, (const __nv_bfloat16*) dt_bias, (const __nv_bfloat16*) norm_w,
        one(S, T), nullptr, (float*) out, write_state);
    return (int) cudaGetLastError();
}

// Segmented variants: nseg sequences, rows [r0[g], r0[g] + T[g]) of the batch with state S[g] (host arrays, baked
// into the launch); nd (device int[nseg], may be null) caps each segment's committed rows.
int qwen_gdn_conv_segs(const void* x, int64_t ldx, int nseg, const int* r0, const int* T, void* const* state, const void* w, void* y,
                       cudaStream_t st)
{
    qgdn::Segs sg;
    if (segs(sg, nseg, r0, T, state)) return -1;
    int tmax = 0;
    for (int g = 0; g < nseg; ++g) tmax = T[g] > tmax ? T[g] : tmax;
    qgdn::conv_kernel<<<dim3(qgdn::C / 256, nseg, (tmax + qgdn::CONV_RT - 1) / qgdn::CONV_RT), 256, 0, st>>>((const float*) x, ldx, sg,
        (const __nv_bfloat16*) w, (float*) y);
    return (int) cudaGetLastError();
}

int qwen_gdn_conv_commit_segs(const void* x, int64_t ldx, int nseg, const int* r0, const int* T, void* const* state, const int* nd,
                              cudaStream_t st)
{
    qgdn::Segs sg;
    if (segs(sg, nseg, r0, T, state)) return -1;
    qgdn::conv_commit_kernel<<<dim3(qgdn::C / 256, nseg), 256, 0, st>>>((const float*) x, ldx, sg, nd);
    return (int) cudaGetLastError();
}

// Prefill: conv / commit on the unfinished qkv projection (part fp16 [rows, 10240], row stride ldp, and its svh)
int qwen_gdn_conv_h(const void* part, int64_t ldp, const void* svh, int nseg, const int* r0, const int* T, void* const* state, const void* w,
                    void* y, cudaStream_t st)
{
    qgdn::Segs sg;
    if (segs(sg, nseg, r0, T, state)) return -1;
    int tmax = 0;
    for (int g = 0; g < nseg; ++g) tmax = T[g] > tmax ? T[g] : tmax;
    qgdn::conv_h_kernel<false><<<dim3(qgdn::C / 512, nseg, (tmax + qgdn::CONV_RT - 1) / qgdn::CONV_RT), 128, 0, st>>>((const half*) part, ldp,
        (const half*) svh, sg, (const __nv_bfloat16*) w, (float*) y);
    return (int) cudaGetLastError();
}
// conv_h with the recurrence's prep fused: qk / db go to ws (laid out as qwen_gdn_recur_segs uses it); y gets only v (channels
// >= 4096). Follow with qwen_gdn_recur_segs_z(..., prepped = 1) on the same ws.
int qwen_gdn_conv_hq(const void* part, int64_t ldp, const void* svh, int nseg, const int* r0, const int* T, void* const* state, const void* w,
                     void* y, const void* ab, int64_t ldab, const void* A_log, const void* dt_bias, void* ws, cudaStream_t st)
{
    qgdn::Segs sg;
    if (segs(sg, nseg, r0, T, state)) return -1;
    int tmax = 0, R = 0;
    for (int g = 0; g < nseg; ++g) { tmax = T[g] > tmax ? T[g] : tmax; R = r0[g] + T[g] > R ? r0[g] + T[g] : R; }
    float* qk = (float*) ws;
    float* db = qk + (size_t) R * 16 * 2 * qgdn::D;
    qgdn::conv_h_kernel<true><<<dim3(qgdn::C / 512, nseg, (tmax + qgdn::CONV_RT - 1) / qgdn::CONV_RT), 128, 0, st>>>((const half*) part, ldp,
        (const half*) svh, sg, (const __nv_bfloat16*) w, (float*) y, (const float*) ab, ldab, (const __nv_bfloat16*) A_log,
        (const __nv_bfloat16*) dt_bias, qk, db);
    return (int) cudaGetLastError();
}
int qwen_gdn_conv_commit_h(const void* part, int64_t ldp, const void* svh, int nseg, const int* r0, const int* T, void* const* state, cudaStream_t st)
{
    qgdn::Segs sg;
    if (segs(sg, nseg, r0, T, state)) return -1;
    qgdn::conv_commit_h_kernel<<<dim3(qgdn::C / 512, nseg), 128, 0, st>>>((const half*) part, ldp, (const half*) svh, sg);
    return (int) cudaGetLastError();
}

// ws: device scratch of qwen_gdn_ws_bytes(R) (R = rows of the batch)
size_t qwen_gdn_ws_bytes(int R) { return (size_t) R * (16 * 2 * qgdn::D + qgdn::NV * 2) * 4 + 256; }

int qwen_gdn_recur_segs_z(const void* y, const void* ab, int64_t ldab, const void* z, int64_t ldz, const void* z_svh, const void* A_log,
                          const void* dt_bias, const void* norm_w, int nseg, const int* r0, const int* T, void* const* S, const int* nd,
                          void* out, int write_state, void* ws, cudaStream_t st);
int qwen_gdn_recur_segs(const void* y, const void* ab, int64_t ldab, const void* z, int64_t ldz, const void* A_log,
                        const void* dt_bias, const void* norm_w, int nseg, const int* r0, const int* T, void* const* S, const int* nd,
                        void* out, int write_state, void* ws, cudaStream_t st)
{
    return qwen_gdn_recur_segs_z(y, ab, ldab, z, ldz, nullptr, A_log, dt_bias, norm_w, nseg, r0, T, S, nd, out, write_state, ws, st);
}
static int recur_segs_impl(const void* y, const void* ab, int64_t ldab, const void* z, int64_t ldz, const void* z_svh, const void* A_log,
                           const void* dt_bias, const void* norm_w, int nseg, const int* r0, const int* T, void* const* S, const int* nd,
                           void* out, int write_state, void* ws, int prepped, cudaStream_t st);
// z_svh non-null (prefill chunks only): z is the unfinished fp16 projection part, finished inside the output norm
int qwen_gdn_recur_segs_z(const void* y, const void* ab, int64_t ldab, const void* z, int64_t ldz, const void* z_svh, const void* A_log,
                          const void* dt_bias, const void* norm_w, int nseg, const int* r0, const int* T, void* const* S, const int* nd,
                          void* out, int write_state, void* ws, cudaStream_t st)
{
    return recur_segs_impl(y, ab, ldab, z, ldz, z_svh, A_log, dt_bias, norm_w, nseg, r0, T, S, nd, out, write_state, ws, 0, st);
}
// as qwen_gdn_recur_segs_z after qwen_gdn_conv_hq filled ws (prefill chunks only; no prep launch)
int qwen_gdn_recur_segs_zq(const void* y, const void* ab, int64_t ldab, const void* z, int64_t ldz, const void* z_svh, const void* A_log,
                           const void* dt_bias, const void* norm_w, int nseg, const int* r0, const int* T, void* const* S,
                           void* out, int write_state, void* ws, cudaStream_t st)
{
    return recur_segs_impl(y, ab, ldab, z, ldz, z_svh, A_log, dt_bias, norm_w, nseg, r0, T, S, nullptr, out, write_state, ws, 1, st);
}
static int recur_segs_impl(const void* y, const void* ab, int64_t ldab, const void* z, int64_t ldz, const void* z_svh, const void* A_log,
                           const void* dt_bias, const void* norm_w, int nseg, const int* r0, const int* T, void* const* S, const int* nd,
                           void* out, int write_state, void* ws, int prepped, cudaStream_t st)
{
    qgdn::Segs sg;
    if (segs(sg, nseg, r0, T, S)) return -1;
    // row-parallel kernels for prefill chunks only: for decode / verify chains (<= 16 rows a segment) their two
    // extra launches cost more than they save (same-process A/B: verify +0.6 ms at 1 sequence, +2.3 ms at 4)
    int tmax = 0;
    for (int g = 0; g < nseg; ++g) tmax = T[g] > tmax ? T[g] : tmax;
    const char* v1 = getenv("QWEN_GDN_V1");
    if ((z_svh || prepped) && (tmax <= 16 || (v1 && *v1 == '1'))) return -3;
    if (tmax <= 16 || (v1 && *v1 == '1'))
    {
        // staged kernel (bitwise equal) where it is faster: one shared-memory-bound block per SM, so not for many
        // segments x long chains (measured with DRAM-resident states: 1 sequence -20..30%, 8 x 11 rows +5..17%)
        const char* rv1 = getenv("QWEN_GDN_RECUR_V1");
        const int work = nseg * tmax;
        if ((v1 && *v1 == '1') || (rv1 && *rv1 == '1') || work > (write_state ? 44 : 16))
            qgdn::recur_kernel<<<dim3(qgdn::NV, nseg), 128, 0, st>>>((const float*) y, (const float*) ab, ldab, (const float*) z, ldz,
                (const __nv_bfloat16*) A_log, (const __nv_bfloat16*) dt_bias, (const __nv_bfloat16*) norm_w, sg, nd, (float*) out, write_state);
        else
        {
            constexpr int sm = qgdn::D * qgdn::SLD * 4;
            static bool attr = false;
            if (!attr) { cudaFuncSetAttribute(qgdn::recur2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); attr = true; }
            qgdn::recur2_kernel<<<dim3(qgdn::NV, nseg), 128, sm, st>>>((const float*) y, (const float*) ab, ldab, (const float*) z, ldz,
                (const __nv_bfloat16*) A_log, (const __nv_bfloat16*) dt_bias, (const __nv_bfloat16*) norm_w, sg, nd, (float*) out, write_state);
        }
        return (int) cudaGetLastError();
    }
    // rows of the whole batch [0, R): the last segment's end (segments are laid out in order)
    int R = 0;
    for (int g = 0; g < nseg; ++g) R = r0[g] + T[g] > R ? r0[g] + T[g] : R;
    float* qk = (float*) ws;
    float* db = qk + (size_t) R * 16 * 2 * qgdn::D;
    if (!prepped)
        qgdn::prep_kernel<<<dim3(R, 16), 128, 0, st>>>((const float*) y, (const float*) ab, ldab,
            (const __nv_bfloat16*) A_log, (const __nv_bfloat16*) dt_bias, qk, db);
    const char* r1 = getenv("QWEN_GDN_ROWS1");
    if (r1 && *r1 == '1')
        qgdn::rows_kernel<<<dim3(qgdn::NV, qgdn::D / 32, nseg), 32, 0, st>>>((const float*) y, qk, db, sg, nd, (float*) out, write_state);
    else
        qgdn::rows4_kernel<<<dim3(qgdn::NV, qgdn::D * qgdn::LPR / 32, nseg), 32, 0, st>>>((const float*) y, qk, db, sg, nd, (float*) out,
                                                                                         write_state);
    if (out && z_svh)
        qgdn::onorm_h_kernel<<<dim3(R, qgdn::NV), 128, 0, st>>>((float*) out, (const half*) z, ldz, (const half*) z_svh, (const __nv_bfloat16*) norm_w);
    else if (out)
        qgdn::onorm_kernel<<<dim3(R, qgdn::NV), 128, 0, st>>>((float*) out, (const float*) z, ldz, (const __nv_bfloat16*) norm_w);
    return (int) cudaGetLastError();
}

}
