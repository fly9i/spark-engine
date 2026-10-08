// Gated-residual hyper-connections (hc_count 4, hidden 2560, rank 320) for decode / verify rows (R <= 16).
//   n = RMSNorm_per_stream(X) * (1 + w)                         X [R, 4, 2560] fp32
//   d = n_flat @ [down | inject]^T   (324 outputs)             F16 weights [324, 10240] (down rows, then inject)
//   g = sigmoid(silu(d[:320] / 4) @ up^T)                       up [10240, 320] F16
//   mixed = mean_s(g[s] * n[s])  (block input, fp32)            post = 2 sigmoid(d[320:] / 4)
//   apply after the sublayer: X[s] += post[s] * y
// All sums run in fixed orders (K slices summed in order) -> deterministic.
#include <cuda_fp16.h>
#include <cstdint>

namespace qhc {

constexpr int H = 4, D = 2560, F = H * D, RK = 320, NO = 324;

__device__ __forceinline__ float warp_sum(float v)
{
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// scale[r][s] = rsqrt(mean(X[r,s,:]^2) + eps); one warp per (r, s)
__global__ void rms_kernel(const float* __restrict__ X, float* __restrict__ scale, int R)
{
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (w >= R * H) return;
    const float* x = X + (int64_t) w * D;
    float ss = 0.f;
    for (int i = lane; i < D; i += 32) ss += x[i] * x[i];
    ss = warp_sum(ss);
    if (lane == 0) scale[w] = rsqrtf(ss / D + 1e-6f);
}

// part[slice][r][o] = sum_{k in slice} n[r][k] * W[o][k]; grid (NO / 4, SL), block 128 = 4 warps = 4 outputs.
// Each warp reads its weight row slice once and serves all R rows.
constexpr int SL = 8, KS = F / SL;   // 8 K slices of 1280
__global__ __launch_bounds__(128) void down_kernel(const float* __restrict__ X, const float* __restrict__ scale,
                                                   const float* __restrict__ wnorm1, const half* __restrict__ W,
                                                   float* __restrict__ part, int R)
{
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * 4 + warp, sl = blockIdx.y, k0 = sl * KS;
    const half* wr = W + (int64_t) o * F + k0;
    float acc[16] = {};
    for (int k = lane * 2; k < KS; k += 64)
    {
        float2 wv = __half22float2(*(const half2*) (wr + k));
        float2 nw = *(const float2*) (wnorm1 + k0 + k);                  // (1 + w)
        int s = (k0 + k) / D;
        #pragma unroll
        for (int r = 0; r < 16; ++r)
        {
            if (r >= R) break;
            float2 xv = *(const float2*) (X + (int64_t) r * F + k0 + k);
            float sc = scale[r * H + s];
            acc[r] += xv.x * sc * nw.x * wv.x + xv.y * sc * nw.y * wv.y;
        }
    }
    #pragma unroll
    for (int r = 0; r < 16; ++r)
    {
        if (r >= R) break;
        float v = warp_sum(acc[r]);
        if (lane == 0) part[((int64_t) sl * R + r) * NO + o] = v;
    }
}

// t[r][j] = silu(d[r][j] / 4) for j < 320; post[r][s] = 2 sigmoid(d[r][320 + s] / 4); d = sum of slices in order.
__global__ void mid_kernel(const float* __restrict__ part, float* __restrict__ t, float* __restrict__ post, int R, int inject)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * NO) return;
    int r = i / NO, o = i % NO;
    float d = 0.f;
    for (int s = 0; s < SL; ++s) d += part[((int64_t) s * R + r) * NO + o];
    if (o < RK) { float a = d * 0.25f; t[r * RK + o] = a / (1.0f + __expf(-a)); }
    else if (inject) post[r * H + (o - RK)] = 2.0f / (1.0f + __expf(-d * 0.25f));
}

// mixed[r][i] = (1/4) sum_s sigmoid(up[s*D + i] . t[r]) * n[r][s][i]; block per 32 channels i (4 warps = 4 streams),
// each warp computes the 32 gate dots of its stream (warp-cooperative over k = 320).
__global__ __launch_bounds__(128) void up_kernel(const float* __restrict__ X, const float* __restrict__ scale,
                                                 const float* __restrict__ wnorm1, const half* __restrict__ up,
                                                 const float* __restrict__ t, float* __restrict__ mixed, int R)
{
    __shared__ float ts[16][RK];
    __shared__ float gsh[H][16][32];
    int s = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < R * RK; i += 128) ts[i / RK][i % RK] = t[i];
    __syncthreads();
    int i0 = blockIdx.x * 32;
    for (int c = 0; c < 32; ++c)
    {
        const half* w = up + (int64_t) (s * D + i0 + c) * RK;
        float wv[10];
        #pragma unroll
        for (int q = 0; q < 10; ++q) wv[q] = __half2float(w[lane + 32 * q]);
        for (int r = 0; r < R; ++r)
        {
            float acc = 0.f;
            #pragma unroll
            for (int q = 0; q < 10; ++q) acc += wv[q] * ts[r][lane + 32 * q];
            acc = warp_sum(acc);
            if (lane == 0) gsh[s][r][c] = 1.0f / (1.0f + __expf(-acc));
        }
    }
    __syncthreads();
    // thread (s = warp, lane = channel): accumulate over streams in order via shared memory
    for (int r = 0; r < R; ++r)
    {
        int i = i0 + lane;
        float nv = X[((int64_t) r * H + s) * D + i] * scale[r * H + s] * wnorm1[s * D + i];
        float v = gsh[s][r][lane] * nv;
        __syncthreads();
        gsh[s][r][lane] = v;
        __syncthreads();
        if (s == 0) mixed[(int64_t) r * D + i] = ((gsh[0][r][lane] + gsh[1][r][lane]) + (gsh[2][r][lane] + gsh[3][r][lane])) * 0.25f;
    }
}

// X[r][s][i] += post[r][s] * y[r][i]
__global__ void apply_kernel(float* __restrict__ X, const float* __restrict__ post, const float* __restrict__ y, int R)
{
    int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) R * F) return;
    int r = idx / F, s = (idx / D) % H, i = idx % D;
    X[idx] += post[r * H + s] * y[(int64_t) r * D + i];
}

// v2 pieces (the GEMVs run in qwen_f16.cu with 16-byte weight loads):
// n[r][s*2560 + i] = X[r][s][i] * rsqrt(mean_i X^2 + eps) * w1[s*2560 + i]; one block (256) per (r, s)
__global__ __launch_bounds__(256) void norm_kernel(const float* __restrict__ X, const float* __restrict__ w1, float* __restrict__ n)
{
    __shared__ float red[8];
    int rs = blockIdx.x;
    const float* x = X + (int64_t) rs * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) ss += x[i] * x[i];
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += red[i];
    float sc = rsqrtf(t / D + 1e-6f);
    int s = rs % H;
    for (int i = threadIdx.x; i < D; i += 256) n[(int64_t) rs * D + i] = x[i] * sc * w1[s * D + i];
}

// t[r][j] = silu(d[r][j] / 4) (j < 320); post[r][s] = 2 sigmoid(d[r][320 + s] / 4); d [R, ldd]
__global__ void mid2_kernel(const float* __restrict__ d, int ldd, float* __restrict__ t, float* __restrict__ post, int R, int inject)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * NO) return;
    int r = i / NO, o = i % NO;
    float v = d[(int64_t) r * ldd + o];
    if (o < RK) { float a = v * 0.25f; t[r * RK + o] = a / (1.0f + __expf(-a)); }
    else if (inject) post[r * H + (o - RK)] = 2.0f / (1.0f + __expf(-v * 0.25f));
}

// mixed[r][i] = (1/4) sum_s sigmoid(g[r][s*2560 + i]) * n[r][s*2560 + i]  (streams in order)
__global__ void mix2_kernel(const float* __restrict__ g, const float* __restrict__ n, float* __restrict__ mixed, int R)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * D) return;
    int r = idx / D, i = idx % D;
    const float* gr = g + (int64_t) r * F;
    const float* nr = n + (int64_t) r * F;
    float v0 = nr[i] / (1.0f + __expf(-gr[i])), v1 = nr[D + i] / (1.0f + __expf(-gr[D + i]));
    float v2 = nr[2 * D + i] / (1.0f + __expf(-gr[2 * D + i])), v3 = nr[3 * D + i] / (1.0f + __expf(-gr[3 * D + i]));
    mixed[idx] = ((v0 + v1) + (v2 + v3)) * 0.25f;
}

// ---- prefill rows (fp16 matmul operands, no fp32 n buffer); bitwise equal to norm / F16 matmul / mid2 / mix2
// with the torch dtype conversions in between ----
// n_h = fp16(x * sc * w1), sc[r * 4 + s] = rsqrt(mean x^2 + eps)
__global__ __launch_bounds__(256) void norm_h_kernel(const float* __restrict__ X, const float* __restrict__ w1, half* __restrict__ n,
                                                     float* __restrict__ scale)
{
    __shared__ float red[8];
    int rs = blockIdx.x;
    const float* x = X + (int64_t) rs * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) ss += x[i] * x[i];
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += red[i];
    float sc = rsqrtf(t / D + 1e-6f);
    int s = rs % H;
    if (threadIdx.x == 0) scale[rs] = sc;
    for (int i = threadIdx.x; i < D; i += 256) n[(int64_t) rs * D + i] = __float2half_rn(x[i] * sc * w1[s * D + i]);
}

// apply_kernel followed by norm_h_kernel in one pass: X += post * y (written back), then the row-stream rms and
// n_h from the updated values (the same per-thread element order and reductions: bitwise equal)
__global__ __launch_bounds__(256) void apply_norm_h_kernel(float* __restrict__ X, const float* __restrict__ post, const float* __restrict__ y,
                                                           const float* __restrict__ w1, half* __restrict__ n, float* __restrict__ scale)
{
    __shared__ float red[8];
    int rs = blockIdx.x, r = rs / H, s = rs % H;
    float* x = X + (int64_t) rs * D;
    const float* yr = y + (int64_t) r * D;
    const float p = post[r * H + s];
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) { float v = x[i] + p * yr[i]; x[i] = v; ss += v * v; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += red[i];
    float sc = rsqrtf(t / D + 1e-6f);
    if (threadIdx.x == 0) scale[rs] = sc;
    for (int i = threadIdx.x; i < D; i += 256) n[(int64_t) rs * D + i] = __float2half_rn(x[i] * sc * w1[s * D + i]);
}

// decode rows: apply_kernel followed by norm_kernel in one pass (fp32 n; bitwise equal to the two kernels)
__global__ __launch_bounds__(256) void apply_norm_kernel(float* __restrict__ X, const float* __restrict__ post, const float* __restrict__ y,
                                                         const float* __restrict__ w1, float* __restrict__ n)
{
    __shared__ float red[8];
    int rs = blockIdx.x, r = rs / H, s = rs % H;
    float* x = X + (int64_t) rs * D;
    const float* yr = y + (int64_t) r * D;
    const float p = post[r * H + s];
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) { float v = x[i] + p * yr[i]; x[i] = v; ss += v * v; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += red[i];
    float sc = rsqrtf(t / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += 256) n[(int64_t) rs * D + i] = x[i] * sc * w1[s * D + i];
}

// t = fp16(silu(d / 4)); post = 2 sigmoid(d / 4); d fp16 [R, ldd]
__global__ void mid2h_kernel(const half* __restrict__ d, int ldd, half* __restrict__ t, float* __restrict__ post, int R, int inject)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * NO) return;
    int r = i / NO, o = i % NO;
    float v = __half2float(d[(int64_t) r * ldd + o]);
    if (o < RK) { float a = v * 0.25f; t[r * RK + o] = __float2half_rn(a / (1.0f + __expf(-a))); }
    else if (inject) post[r * H + (o - RK)] = 2.0f / (1.0f + __expf(-v * 0.25f));
}

// mixed as mix2_kernel, n recomputed from x (same expression as norm) instead of read
__global__ void mix2h_kernel(const half* __restrict__ g, const float* __restrict__ X, const float* __restrict__ scale,
                             const float* __restrict__ w1, float* __restrict__ mixed, int R)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * D) return;
    int r = idx / D, i = idx % D;
    const half* gr = g + (int64_t) r * F;
    const float* xr = X + (int64_t) r * F;
    float v[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s)
    {
        float n = xr[s * D + i] * scale[r * 4 + s] * w1[s * D + i];
        v[s] = n / (1.0f + __expf(-__half2float(gr[s * D + i])));
    }
    mixed[idx] = ((v[0] + v[1]) + (v[2] + v[3])) * 0.25f;
}

// up GEMM + mix in one kernel (prefill rows): g = t @ up^T on tensor cores (fp16 operands, fp32 accumulation) and,
// in the epilogue, mixed[r][i] = mean_s n[r][s][i] * sigmoid(g[r][s*2560 + i]) with n = x * scale * w1 (fp32, as
// mix2h_kernel). g never goes to memory and stays fp32 (mix2h reads cuBLAS's fp16 g: L1).
// grid (2560 / 32, ceil(R / 64)) x 256 threads; block = 64 rows x 32 columns of each of the 4 streams (128 g
// columns); warp (wr, wc) = rows 32 wr.., g columns 32 wc.. (stream wc); K = 320 in chunks of 32, double buffered
// with cp.async; 34 KB of shared memory (2-3 blocks per SM).
constexpr int UM_BM = 64, UM_BN = 128, UM_BI = 32, UM_KC = 32, UM_LD = UM_KC + 8, UM_GLD = 4 * UM_BI + 4;
constexpr size_t UM_SMEM_MAIN = (size_t) 2 * (UM_BM + UM_BN) * UM_LD * 2, UM_SMEM_EPI = (size_t) UM_BM * UM_GLD * 4;
constexpr size_t UM_SMEM = UM_SMEM_MAIN > UM_SMEM_EPI ? UM_SMEM_MAIN : UM_SMEM_EPI;
__device__ __forceinline__ void um_cp16(void* sm, const void* g)
{
    unsigned a = (unsigned) __cvta_generic_to_shared(sm);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(g));
}
__device__ __forceinline__ void um_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ __launch_bounds__(256) void upmix_kernel(const half* __restrict__ t, const half* __restrict__ up, const float* __restrict__ X,
                                                    const float* __restrict__ scale, const float* __restrict__ w1, float* __restrict__ mixed, int R)
{
    extern __shared__ __align__(16) half um[];
    const int i0 = blockIdx.x * UM_BI, r0 = blockIdx.y * UM_BM;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wr = warp & 1, wc = warp >> 1;
    auto sa = [&](int b) { return um + (size_t) b * (UM_BM + UM_BN) * UM_LD; };
    auto sb = [&](int b) { return sa(b) + UM_BM * UM_LD; };
    auto load = [&](int kc, int b)
    {
        for (int q = tid; q < (UM_BM + UM_BN) * (UM_KC / 8); q += 256)
        {
            const int row = q / (UM_KC / 8), c = (q % (UM_KC / 8)) * 8;
            if (row < UM_BM)
            {
                const int ra = min(r0 + row, R - 1);                               // rows past R: computed, not stored
                um_cp16(sa(b) + row * UM_LD + c, t + (int64_t) ra * RK + kc * UM_KC + c);
            }
            else
            {
                const int br = row - UM_BM, gcol = (br / UM_BI) * D + i0 + br % UM_BI;   // B row = up row of (stream, i)
                um_cp16(sb(b) + br * UM_LD + c, up + (int64_t) gcol * RK + kc * UM_KC + c);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float c[2][4][4] = {};
    constexpr int NKC = RK / UM_KC;
    load(0, 0);
    for (int kc = 0; kc < NKC; ++kc)
    {
        if (kc + 1 < NKC) { load(kc + 1, (kc + 1) & 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const half* A = sa(kc & 1);
        const half* B = sb(kc & 1);
        #pragma unroll
        for (int kk = 0; kk < UM_KC / 16; ++kk)
        {
            uint32_t a[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
            {
                const half* ap = A + (wr * 32 + mt * 16 + (lane & 15)) * UM_LD + kk * 16 + (lane >> 4) * 8;
                unsigned sp = (unsigned) __cvta_generic_to_shared(ap);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[mt][0]), "=r"(a[mt][1]), "=r"(a[mt][2]), "=r"(a[mt][3]) : "r"(sp));
            }
            #pragma unroll
            for (int np = 0; np < 2; ++np)   // pairs of n8 tiles
            {
                uint32_t b[4];
                const half* bp = B + (wc * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * UM_LD + kk * 16 + ((lane >> 3) & 1) * 8;
                unsigned sp = (unsigned) __cvta_generic_to_shared(bp);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(sp));
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                {
                    um_mma(c[mt][np * 2], a[mt], b[0], b[1]);
                    um_mma(c[mt][np * 2 + 1], a[mt], b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    float* sg = (float*) um;   // [128][UM_GLD]: g of the block
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt)
        {
            const int row = wr * 32 + mt * 16 + (lane >> 2), col = wc * 32 + nt * 8 + (lane & 3) * 2;
            sg[row * UM_GLD + col] = c[mt][nt][0]; sg[row * UM_GLD + col + 1] = c[mt][nt][1];
            sg[(row + 8) * UM_GLD + col] = c[mt][nt][2]; sg[(row + 8) * UM_GLD + col + 1] = c[mt][nt][3];
        }
    __syncthreads();
    for (int q = tid; q < UM_BM * UM_BI; q += 256)
    {
        const int row = q / UM_BI, ii = q % UM_BI, r = r0 + row, i = i0 + ii;
        if (r >= R) break;
        const float* xr = X + (int64_t) r * F;
        float v[4];
        #pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            const float n = xr[s * D + i] * scale[r * 4 + s] * w1[s * D + i];
            v[s] = n / (1.0f + __expf(-sg[row * UM_GLD + s * UM_BI + ii]));
        }
        mixed[(int64_t) r * D + i] = ((v[0] + v[1]) + (v[2] + v[3])) * 0.25f;
    }
}

}  // namespace qhc

extern "C" {

// wnorm1: fp32 (1 + hc_norm) [10240]; W: F16 [324 (or 320 without inject), 10240]; up F16 [10240, 320].
// ws >= qwen_hc_ws_bytes(R). post may be null when inject == 0.
size_t qwen_hc_ws_bytes(int R) { return (size_t) R * 4 * 4 + (size_t) qhc::SL * R * qhc::NO * 4 + (size_t) R * qhc::RK * 4 + 1024; }

int qwen_hc_mix(const void* X, int R, const void* wnorm1, const void* W, int inject, const void* up,
                void* mixed, void* post, void* ws, cudaStream_t st)
{
    using namespace qhc;
    if (R > 16) return -1;
    float* scale = (float*) ws;
    float* part = scale + R * 4 + 64;
    float* t = part + SL * R * NO + 64;
    rms_kernel<<<(R * H * 32 + 127) / 128, 128, 0, st>>>((const float*) X, scale, R);
    int no = inject ? NO : RK;
    down_kernel<<<dim3(no / 4, SL), 128, 0, st>>>((const float*) X, scale, (const float*) wnorm1, (const half*) W, part, R);
    mid_kernel<<<(R * NO + 255) / 256, 256, 0, st>>>(part, t, (float*) post, R, inject);
    up_kernel<<<D / 32, 128, 0, st>>>((const float*) X, scale, (const float*) wnorm1, (const half*) up, t, (float*) mixed, R);
    return (int) cudaGetLastError();
}

int qwen_hc_apply(void* X, const void* post, const void* y, int R, cudaStream_t st)
{
    int64_t n = (int64_t) R * qhc::F;
    qhc::apply_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>((float*) X, (const float*) post, (const float*) y, R);
    return (int) cudaGetLastError();
}


int qwen_hc_norm(const void* X, int R, const void* w1, void* n, cudaStream_t st)
{
    qhc::norm_kernel<<<R * qhc::H, 256, 0, st>>>((const float*) X, (const float*) w1, (float*) n);
    return (int) cudaGetLastError();
}
int qwen_hc_mid2(const void* d, int ldd, void* t, void* post, int R, int inject, cudaStream_t st)
{
    qhc::mid2_kernel<<<(R * qhc::NO + 255) / 256, 256, 0, st>>>((const float*) d, ldd, (float*) t, (float*) post, R, inject);
    return (int) cudaGetLastError();
}
int qwen_hc_mix2(const void* g, const void* n, void* mixed, int R, cudaStream_t st)
{
    qhc::mix2_kernel<<<(R * qhc::D + 255) / 256, 256, 0, st>>>((const float*) g, (const float*) n, (float*) mixed, R);
    return (int) cudaGetLastError();
}


int qwen_hc_norm_h(const void* X, int R, const void* w1, void* n, void* scale, cudaStream_t st)
{
    qhc::norm_h_kernel<<<R * qhc::H, 256, 0, st>>>((const float*) X, (const float*) w1, (half*) n, (float*) scale);
    return (int) cudaGetLastError();
}
int qwen_hc_mid2h(const void* d, int ldd, void* t, void* post, int R, int inject, cudaStream_t st)
{
    qhc::mid2h_kernel<<<(R * qhc::NO + 255) / 256, 256, 0, st>>>((const half*) d, ldd, (half*) t, (float*) post, R, inject);
    return (int) cudaGetLastError();
}
// t [R, 320] fp16, up [10240, 320] fp16 (Linear), X [R, 4, 2560], scale [R * 4], w1 [10240] -> mixed [R, 2560]
int qwen_hc_upmix(const void* t, const void* up, const void* X, const void* scale, const void* w1, void* mixed, int R, cudaStream_t st)
{
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(qhc::upmix_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) qhc::UM_SMEM); attr = true; }
    qhc::upmix_kernel<<<dim3(qhc::D / qhc::UM_BI, (R + qhc::UM_BM - 1) / qhc::UM_BM), 256, qhc::UM_SMEM, st>>>((const half*) t, (const half*) up,
        (const float*) X, (const float*) scale, (const float*) w1, (float*) mixed, R);
    return (int) cudaGetLastError();
}

int qwen_hc_mix2h(const void* g, const void* X, const void* scale, const void* w1, void* mixed, int R, cudaStream_t st)
{
    qhc::mix2h_kernel<<<(R * qhc::D + 255) / 256, 256, 0, st>>>((const half*) g, (const float*) X, (const float*) scale,
                                                                (const float*) w1, (float*) mixed, R);
    return (int) cudaGetLastError();
}

int qwen_hc_apply_norm(void* X, const void* post, const void* y, const void* w1, void* n, int R, cudaStream_t st)
{
    qhc::apply_norm_kernel<<<R * qhc::H, 256, 0, st>>>((float*) X, (const float*) post, (const float*) y, (const float*) w1, (float*) n);
    return (int) cudaGetLastError();
}

int qwen_hc_apply_norm_h(void* X, const void* post, const void* y, const void* w1, void* n, void* scale, int R, cudaStream_t st)
{
    qhc::apply_norm_h_kernel<<<R * qhc::H, 256, 0, st>>>((float*) X, (const float*) post, (const float*) y, (const float*) w1, (half*) n,
                                                       (float*) scale);
    return (int) cudaGetLastError();
}

}
