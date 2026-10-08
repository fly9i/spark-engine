// Q8 copies of the F16 dense weights (HC down / up, router, PLE projections) for decode / verify rows (M <= 16):
// int8 W [N, K] with one fp32 scale per (row, group of 64 along K) (absmax / 127). y[m, n] = sum_k x[m, k] * w[n, k],
// fp32 x and accumulation, K split in fixed slices summed in order (deterministic). Lossy (L3): opt-in.
#include <cuda_fp16.h>
#include <cstdint>

namespace qq8 {

constexpr int G = 64;

// one warp per (row n, group g): scale = absmax / 127, q = round(w / scale)
__global__ void encode_kernel(const half* __restrict__ W, int N, int K, int8_t* __restrict__ q, float* __restrict__ s)
{
    int64_t warp = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int ng = K / G;
    if (warp >= (int64_t) N * ng) return;
    int n = (int) (warp / ng), g = (int) (warp % ng);
    const half* w = W + (int64_t) n * K + g * G;
    float a = __half2float(w[lane]), b = __half2float(w[lane + 32]);
    float m = fmaxf(fabsf(a), fabsf(b));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sc = m > 0.f ? m / 127.f : 1.f;
    int8_t* d = q + (int64_t) n * K + g * G;
    d[lane] = (int8_t) __float2int_rn(a / sc);
    d[lane + 32] = (int8_t) __float2int_rn(b / sc);
    if (lane == 0) s[(int64_t) n * ng + g] = sc;
}

// warp = 4 output rows, lane reads 16 int8 of a row per step (k step 512); block = 4 warps = 16 outputs;
// grid (N / 16, slices). part[s][m][n]
template <int MR>
__global__ __launch_bounds__(128) void gemv_kernel(const float* __restrict__ x, int64_t ldx, int M, const int8_t* __restrict__ q,
                                                   const float* __restrict__ sc, int K, int N, int kslice, float* __restrict__ part)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * 4 + warp) * 4;
    const int k0 = blockIdx.y * kslice, k1 = min(K, k0 + kslice);
    const int ng = K / G;
    float acc[4][MR];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int r = 0; r < MR; ++r) acc[j][r] = 0.f;
    for (int k = k0 + lane * 16; k < k1; k += 512)
    {
        float w[4][16];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            if (n0 + j < N)
            {
                int4 v = __ldcs((const int4*) (q + (int64_t) (n0 + j) * K + k));
                const int8_t* b = (const int8_t*) &v;
                const float s = sc[(int64_t) (n0 + j) * ng + k / G];
                #pragma unroll
                for (int i = 0; i < 16; ++i) w[j][i] = (float) b[i] * s;
            }
            else
                #pragma unroll
                for (int i = 0; i < 16; ++i) w[j][i] = 0.f;
        }
        #pragma unroll
        for (int r = 0; r < MR; ++r)
        {
            if (r >= M) break;
            const float* xr = x + r * ldx + k;
            float xv[16];
            #pragma unroll
            for (int i = 0; i < 16; i += 4) { float4 f = *(const float4*) (xr + i); xv[i] = f.x; xv[i + 1] = f.y; xv[i + 2] = f.z; xv[i + 3] = f.w; }
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int i = 0; i < 16; ++i) acc[j][r] += xv[i] * w[j][i];
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int r = 0; r < MR; ++r)
        {
            if (r >= M) break;
            float v = acc[j][r];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && n0 + j < N) part[((int64_t) blockIdx.y * M + r) * N + n0 + j] = v;
        }
}

__global__ void sum_kernel(const float* __restrict__ part, int S, int M, int N, float* __restrict__ y, int64_t ldy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    int m = i / N, n = i % N;
    float v = 0.f;
    for (int s = 0; s < S; ++s) v += part[(int64_t) s * M * N + i];
    y[(int64_t) m * ldy + n] = v;
}

}  // namespace qq8

extern "C" {

// W F16 [N, K] (K % 64 == 0) -> q int8 [N, K], s fp32 [N, K / 64]
int qwen_q8_encode(const void* W, int N, int K, void* q, void* s, cudaStream_t st)
{
    if (K % qq8::G) return -1;
    int64_t warps = (int64_t) N * (K / qq8::G);
    qq8::encode_kernel<<<(unsigned) ((warps * 32 + 255) / 256), 256, 0, st>>>((const half*) W, N, K, (int8_t*) q, (float*) s);
    return (int) cudaGetLastError();
}

static void q8_slices(int K, int& S, int& kslice)
{
    S = (K + 2047) / 2048;                     // slices of 2048 (4 lane-steps)
    kslice = (K + S - 1) / S;
    kslice = (kslice + 511) / 512 * 512;
    S = (K + kslice - 1) / kslice;
}
size_t qwen_q8_ws_bytes(int M, int N, int K) { int S, ks; q8_slices(K, S, ks); return (size_t) S * M * N * 4 + 256; }

// x [M <= 16, K] fp32 (row stride ldx, K % 16 == 0) -> y [M, N] fp32 (row stride ldy)
int qwen_q8_gemv(const void* x, int64_t ldx, int M, const void* q, const void* s, int K, int N, void* y, int64_t ldy, void* ws,
                 cudaStream_t st)
{
    if (M > 16 || K % 16) return -1;
    int S, kslice;
    q8_slices(K, S, kslice);
    dim3 grid((N + 15) / 16, S);
    float* part = (float*) ws;
    if (M <= 4) qq8::gemv_kernel<4><<<grid, 128, 0, st>>>((const float*) x, ldx, M, (const int8_t*) q, (const float*) s, K, N, kslice, part);
    else qq8::gemv_kernel<16><<<grid, 128, 0, st>>>((const float*) x, ldx, M, (const int8_t*) q, (const float*) s, K, N, kslice, part);
    qq8::sum_kernel<<<(M * N + 255) / 256, 256, 0, st>>>(part, S, M, N, (float*) y, ldy);
    return (int) cudaGetLastError();
}

}
