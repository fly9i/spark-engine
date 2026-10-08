// Skinny GEMV with F16 weights for Qwen3.8 decode / verify rows (M <= 64): y[m, n] = sum_k x[m, k] * W[n, k]
// (W row-major [N, K], the checkpoint's Linear orientation; x fp32). K split in fixed slices summed in order.
// Warp = NJ output rows of W (4 up to 16 x rows, 2 up to 32, 1 up to 64: one weight pass), lanes read 16 bytes
// (8 halves) per row per step; block = 4 warps. Per element the same sums for any NJ / MR (batch-independent).
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

namespace qf16 {

// cnt (may be null): per block column a counter; the last of the S slice blocks sums the partials of its outputs in
// slice order (as sum_kernel) into y and resets the counter.
template <int NJ, int MR>
__global__ __launch_bounds__(128) void gemv_kernel(const float* __restrict__ x, int64_t ldx, int M, const half* __restrict__ W,
                                                   int K, int N, int kslice, float* __restrict__ part,
                                                   int* __restrict__ cnt = nullptr, float* __restrict__ y = nullptr, int64_t ldy = 0)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * 4 + warp) * NJ;
    const int k0 = blockIdx.y * kslice, k1 = min(K, k0 + kslice);
    float acc[NJ][MR];
    #pragma unroll
    for (int j = 0; j < NJ; ++j)
        #pragma unroll
        for (int r = 0; r < MR; ++r) acc[j][r] = 0.f;
    for (int k = k0 + lane * 8; k < k1; k += 256)
    {
        float w[NJ][8];
        #pragma unroll
        for (int j = 0; j < NJ; ++j)
        {
            if (n0 + j < N)
            {
                uint4 v = __ldcs((const uint4*) (W + (int64_t) (n0 + j) * K + k));
                const half2* h = (const half2*) &v;
                #pragma unroll
                for (int q = 0; q < 4; ++q) { float2 f = __half22float2(h[q]); w[j][2 * q] = f.x; w[j][2 * q + 1] = f.y; }
            }
            else
                #pragma unroll
                for (int q = 0; q < 8; ++q) w[j][q] = 0.f;
        }
        #pragma unroll
        for (int r = 0; r < MR; ++r)
        {
            if (r >= M) break;
            const float4 xa = *(const float4*) (x + r * ldx + k), xb = *(const float4*) (x + r * ldx + k + 4);
            const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int j = 0; j < NJ; ++j)
                #pragma unroll
                for (int q = 0; q < 8; ++q) acc[j][r] += xv[q] * w[j][q];
        }
    }
    #pragma unroll
    for (int j = 0; j < NJ; ++j)
        #pragma unroll
        for (int r = 0; r < MR; ++r)
        {
            if (r >= M) break;
            float v = acc[j][r];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && n0 + j < N) part[((int64_t) blockIdx.y * M + r) * N + n0 + j] = v;
        }
    if (!cnt) return;
    __shared__ int is_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) is_last = atomicAdd(&cnt[blockIdx.x], 1) == (int) gridDim.y - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    const int nb0 = blockIdx.x * 4 * NJ;
    for (int i = threadIdx.x; i < M * 4 * NJ; i += 128)
    {
        const int r = i / (4 * NJ), n = nb0 + i % (4 * NJ);
        if (n >= N) continue;
        float v = 0.f;
        for (int s = 0; s < (int) gridDim.y; ++s) v += __ldcg(part + ((int64_t) s * M + r) * N + n);
        y[(int64_t) r * ldy + n] = v;
    }
    if (threadIdx.x == 0) cnt[blockIdx.x] = 0;
}

// v2 (QWEN_F16_V1=1 per launch: gemv_kernel): the same per-lane k mapping, products and sums as gemv_kernel (bitwise
// equal), with the row count M a compile-time constant (no per-row branches, registers sized to M) and the next k
// step's weights prefetched while the current one is multiplied.
// MR >= 0: exactly MR rows; MR < 0: up to -MR rows, M at run time (rows past M guarded)
template <int NJ, int MR>
__global__ __launch_bounds__(128) void gemv2_kernel(const float* __restrict__ x, int64_t ldx, const half* __restrict__ W,
                                                    int K, int N, int kslice, float* __restrict__ part, float* __restrict__ y, int64_t ldy, int Mrt)
{
    constexpr int M = MR > 0 ? MR : -MR;
    const int Mv = MR > 0 ? MR : Mrt;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * 4 + warp) * NJ;
    const int k0 = blockIdx.y * kslice, k1 = min(K, k0 + kslice);
    float acc[NJ][M];
    #pragma unroll
    for (int j = 0; j < NJ; ++j)
        #pragma unroll
        for (int r = 0; r < M; ++r) acc[j][r] = 0.f;
    uint4 nw[NJ];
    auto ldw = [&](int k)
    {
        #pragma unroll
        for (int j = 0; j < NJ; ++j) nw[j] = n0 + j < N ? __ldcs((const uint4*) (W + (int64_t) (n0 + j) * K + k)) : make_uint4(0, 0, 0, 0);
    };
    int k = k0 + lane * 8;
    if (k < k1) ldw(k);
    for (; k < k1; k += 256)
    {
        float w[NJ][8];
        #pragma unroll
        for (int j = 0; j < NJ; ++j)
        {
            const half2* h = (const half2*) &nw[j];
            #pragma unroll
            for (int q = 0; q < 4; ++q) { float2 f = __half22float2(h[q]); w[j][2 * q] = f.x; w[j][2 * q + 1] = f.y; }
        }
        if (k + 256 < k1) ldw(k + 256);
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            if (MR < 0 && r >= Mv) continue;
            const float4 xa = *(const float4*) (x + r * ldx + k), xb = *(const float4*) (x + r * ldx + k + 4);
            const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int j = 0; j < NJ; ++j)
                #pragma unroll
                for (int q = 0; q < 8; ++q) acc[j][r] += xv[q] * w[j][q];
        }
    }
    #pragma unroll
    for (int j = 0; j < NJ; ++j)
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            if (MR < 0 && r >= Mv) continue;
            float v = acc[j][r];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && n0 + j < N)
            {
                if (y) y[(int64_t) r * ldy + n0 + j] = v;   // one slice: the value itself (as the slice sum 0 + v)
                else part[((int64_t) blockIdx.y * Mv + r) * N + n0 + j] = v;
            }
        }
}

// Tensor-core variant (L1): x (fp32) split per element into fp16 hi + lo (x = hi + lo to ~2^-22 relative), W fp16 (exact);
// y^T = W x^T with A = W (two m16 tiles = 32 W rows per warp) and B = x^T (n8 tiles over the M <= 64 rows), fp32
// accumulation. k is permuted inside each 32-wide chunk in the same way for A and B (lane t holds columns 8t..8t+7 of its
// rows: one 16-byte W load per row and chunk), so every output sums the same products. Per output the work does not
// depend on M (the same chunks, k slices and MMA sequence): results are independent of the batch.
__device__ __forceinline__ void tc_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void split2(float v0, float v1, uint32_t& hi, uint32_t& lo)
{
    const half h0 = __float2half_rn(v0), h1 = __float2half_rn(v1);
    const half l0 = __float2half_rn(v0 - __half2float(h0)), l1 = __float2half_rn(v1 - __half2float(h1));
    hi = (uint32_t) __half_as_ushort(h0) | ((uint32_t) __half_as_ushort(h1) << 16);
    lo = (uint32_t) __half_as_ushort(l0) | ((uint32_t) __half_as_ushort(l1) << 16);
}
template <int MT8, int PF = (MT8 <= 2 ? 4 : MT8 <= 4 ? 3 : 2)>
__global__ __launch_bounds__(128) void gemv_tc_kernel(const float* __restrict__ x, int64_t ldx, int M, const half* __restrict__ W,
                                                      int K, int N, int kslice, float* __restrict__ part, float* __restrict__ y, int64_t ldy)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 32;
    if (n0 >= N) return;
    const int k0 = blockIdx.y * kslice, k1 = min(K, k0 + kslice);
    float c[2][MT8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < MT8; ++nt) c[mt][nt][0] = c[mt][nt][1] = c[mt][nt][2] = c[mt][nt][3] = 0.f;
    const half* wr[4];
    bool wv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int row = n0 + (i >> 1) * 16 + (i & 1) * 8 + g;
        wv[i] = row < N;
        wr[i] = W + (int64_t) (wv[i] ? row : 0) * K + 8 * t;
    }
    const float* xr[MT8];
    bool xv[MT8];
    #pragma unroll
    for (int nt = 0; nt < MT8; ++nt) { const int r = nt * 8 + g; xv[nt] = r < M; xr[nt] = x + (int64_t) (xv[nt] ? r : 0) * ldx + 8 * t; }
    // PF chunks of weights in flight (ring of registers, unrolled so the indices are static)
    uint4 wn[PF][4];
    auto ldw = [&](int kc, uint4 (&d)[4]) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) d[i] = wv[i] && kc < k1 ? __ldcs((const uint4*) (wr[i] + kc)) : make_uint4(0, 0, 0, 0);
    };
    #pragma unroll
    for (int d = 0; d < PF; ++d) ldw(k0 + 32 * d, wn[d]);
    for (int kb = k0; kb < k1; kb += 32 * PF)
    #pragma unroll
    for (int d = 0; d < PF; ++d)
    {
        const int kc = kb + 32 * d;
        if (kc >= k1) break;
        uint4 wa[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) wa[i] = wn[d][i];
        ldw(kc + 32 * PF, wn[d]);
        #pragma unroll
        for (int nt = 0; nt < MT8; ++nt)
        {
            if (nt * 8 >= M) break;   // warp-uniform
            float4 xa = make_float4(0.f, 0.f, 0.f, 0.f), xb = xa;
            if (xv[nt]) { xa = *(const float4*) (xr[nt] + kc); xb = *(const float4*) (xr[nt] + kc + 4); }
            uint32_t hi[4], lo[4];
            split2(xa.x, xa.y, hi[0], lo[0]); split2(xa.z, xa.w, hi[1], lo[1]);
            split2(xb.x, xb.y, hi[2], lo[2]); split2(xb.z, xb.w, hi[3], lo[3]);
            // each 32-wide chunk summed by the tensor cores from zero, then added in fp32 (round to nearest): the MMA's own
            // accumulation truncates small products against a large accumulator (relative error 8e-7 over K = 10240 when
            // accumulating there, vs 1e-7 for fp32 FMAs)
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
            {
                const uint32_t a0[4] = {wa[2 * mt].x, wa[2 * mt + 1].x, wa[2 * mt].y, wa[2 * mt + 1].y};
                const uint32_t a1[4] = {wa[2 * mt].z, wa[2 * mt + 1].z, wa[2 * mt].w, wa[2 * mt + 1].w};
                float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                tc_mma(tmp, a0, hi[0], hi[1]);
                tc_mma(tmp, a1, hi[2], hi[3]);
                float tl[4] = {0.f, 0.f, 0.f, 0.f};
                tc_mma(tl, a0, lo[0], lo[1]);
                tc_mma(tl, a1, lo[2], lo[3]);
                #pragma unroll
                for (int i = 0; i < 4; ++i) c[mt][nt][i] += tmp[i] + tl[i];
            }
        }
    }
    // c[mt][nt]: (W row n0 + 16 mt + g (+8), x row 8 nt + 2t (+1))
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < MT8; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = n0 + mt * 16 + g + (i >= 2 ? 8 : 0), m = nt * 8 + 2 * t + (i & 1);
                if (n < N && m < M)
                {
                    if (y) y[(int64_t) m * ldy + n] = c[mt][nt][i];
                    else part[((int64_t) blockIdx.y * M + m) * N + n] = c[mt][nt][i];
                }
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

}  // namespace qf16

template <int NJ, int MR>
static void gemv2_launch(const float* x, int64_t ldx, int M, const half* w, int K, int N, int kslice, int S, float* part, float* y, int64_t ldy, cudaStream_t st)
{
    qf16::gemv2_kernel<NJ, MR><<<dim3((N + 4 * NJ - 1) / (4 * NJ), S), 128, 0, st>>>(x, ldx, w, K, N, kslice, part, S == 1 ? y : nullptr, ldy, M);
}
template <int NJ>
static void gemv2_m(const float* x, int64_t ldx, int M, const half* w, int K, int N, int kslice, int S, float* part, float* y, int64_t ldy, cudaStream_t st)
{
    switch (M)
    {
#define G2(m) case m: gemv2_launch<NJ, m>(x, ldx, M, w, K, N, kslice, S, part, y, ldy, st); return;
        G2(1) G2(2) G2(3) G2(4) G2(5) G2(6) G2(7) G2(8) G2(9) G2(10) G2(11) G2(12) G2(13) G2(14) G2(15) G2(16)
#undef G2
        default:
            if (M <= 24) gemv2_launch<NJ, -24>(x, ldx, M, w, K, N, kslice, S, part, y, ldy, st);
            else if (M <= 32) gemv2_launch<(NJ > 2 ? 2 : NJ), -32>(x, ldx, M, w, K, N, kslice, S, part, y, ldy, st);
            else if (M <= 48) gemv2_launch<1, -48>(x, ldx, M, w, K, N, kslice, S, part, y, ldy, st);
            else gemv2_launch<1, -64>(x, ldx, M, w, K, N, kslice, S, part, y, ldy, st);
    }
}
// v2 dispatch: rows of W per warp NJ = 4, 2 or 1, the largest that still gives >= 192 blocks (QWEN_F16_NJ overrides)
static bool gemv2(const float* x, int64_t ldx, int M, const half* w, int K, int N, int kslice, int S, float* part, float* y, int64_t ldy, cudaStream_t st)
{
    if (M < 1 || M > 64) return false;
    const char* nje = getenv("QWEN_F16_NJ");
    int nj = nje ? atoi(nje) : ((N + 15) / 16 * S >= 192 ? 4 : (N + 7) / 8 * S >= 192 ? 2 : 1);
    if (nj >= 4) gemv2_m<4>(x, ldx, M, w, K, N, kslice, S, part, y, ldy, st);
    else if (nj == 2) gemv2_m<2>(x, ldx, M, w, K, N, kslice, S, part, y, ldy, st);
    else gemv2_m<1>(x, ldx, M, w, K, N, kslice, S, part, y, ldy, st);
    return true;
}

// Tensor-core kernel (L1) for all rows. Slices: one when the rows give >= 256 warps, else about 384 warps in all (k slices
// of a multiple of 32).
static size_t tc_slices(int N, int K) { int w = (N + 31) / 32, S = w >= 256 ? 1 : (384 + w - 1) / w; int ks = ((K + S - 1) / S + 31) / 32 * 32; return (K + ks - 1) / ks; }
static bool gemv_tc(const float* x, int64_t ldx, int M, const half* w, int K, int N, float* part, float* y, int64_t ldy, cudaStream_t st)
{
    if (M < 1 || M > 64 || K % 32) return false;
    const int S = (int) tc_slices(N, K);
    const int ks = (((K + S - 1) / S) + 31) / 32 * 32;
    dim3 grid((N + 127) / 128, S);
    float* yo = S == 1 ? y : nullptr;
    switch ((M + 7) / 8)
    {
#define TC(m8) case m8: qf16::gemv_tc_kernel<m8><<<grid, 128, 0, st>>>(x, ldx, M, w, K, N, ks, part, yo, ldy); break;
        TC(1) TC(2) TC(3) TC(4) TC(5) TC(6) TC(7) TC(8)
#undef TC
    }
    if (S > 1) qf16::sum_kernel<<<(M * N + 255) / 256, 256, 0, st>>>(part, S, M, N, y, ldy);
    return true;
}

extern "C" {

size_t qwen_f16_ws_bytes(int M, int N, int K) { int S = (K + 1023) / 1024, St = (int) tc_slices(N, K); return (size_t) (S > St ? S : St) * M * N * 4 + 256; }

// x [M <= 64, K] fp32 (K % 8 == 0, row stride ldx), W [N, K] f16 -> y [M, N] fp32 (row stride ldy); ws >= qwen_f16_ws_bytes
int qwen_f16_gemv_c(const void* x, int64_t ldx, int M, const void* W, int K, int N, void* y, int64_t ldy, void* ws, void* cnt, cudaStream_t st);
int qwen_f16_gemv(const void* x, int64_t ldx, int M, const void* W, int K, int N, void* y, int64_t ldy, void* ws, cudaStream_t st)
{
    return qwen_f16_gemv_c(x, ldx, M, W, K, N, y, ldy, ws, nullptr, st);
}
// cnt: [ceil(N / 4)] ints, zero (left zero): the slice sum runs in the gemv's last blocks (no sum_kernel)
int qwen_f16_gemv_c(const void* x, int64_t ldx, int M, const void* W, int K, int N, void* y, int64_t ldy, void* ws, void* cnt, cudaStream_t st)
{
    if (M > 64 || K % 8) return -1;
    int S = (K + 1023) / 1024;                 // slices of 1024 (4 lane-steps)
    int kslice = (K + S - 1) / S;
    kslice = (kslice + 255) / 256 * 256;
    S = (K + kslice - 1) / kslice;
    float* part = (float*) ws;
    const float* xf = (const float*) x;
    const half* w = (const half*) W;
    float* yf = (float*) y;
    int* c = (int*) cnt;   // null: separate sum_kernel
    // default: tensor-core kernel (L1, batch-independent); QWEN_F16_TC=0: v2 (bitwise equal to v1); QWEN_F16_V1=1: v1
    const char* tc = getenv("QWEN_F16_TC");
    if (!c && !(tc && *tc == '0') && gemv_tc(xf, ldx, M, w, K, N, part, yf, ldy, st)) return (int) cudaGetLastError();
    const char* v1 = getenv("QWEN_F16_V1");
    if (!c && !(v1 && *v1 == '1') && gemv2(xf, ldx, M, w, K, N, kslice, S, part, yf, ldy, st))
    {
        if (S > 1) qf16::sum_kernel<<<(M * N + 255) / 256, 256, 0, st>>>(part, S, M, N, yf, ldy);
        return (int) cudaGetLastError();
    }
    if (M <= 4) qf16::gemv_kernel<4, 4><<<dim3((N + 15) / 16, S), 128, 0, st>>>(xf, ldx, M, w, K, N, kslice, part, c, yf, ldy);
    else if (M <= 16) qf16::gemv_kernel<4, 16><<<dim3((N + 15) / 16, S), 128, 0, st>>>(xf, ldx, M, w, K, N, kslice, part, c, yf, ldy);
    else if (M <= 32) qf16::gemv_kernel<2, 32><<<dim3((N + 7) / 8, S), 128, 0, st>>>(xf, ldx, M, w, K, N, kslice, part, c, yf, ldy);
    else qf16::gemv_kernel<1, 64><<<dim3((N + 3) / 4, S), 128, 0, st>>>(xf, ldx, M, w, K, N, kslice, part, c, yf, ldy);
    if (!c) qf16::sum_kernel<<<(M * N + 255) / 256, 256, 0, st>>>(part, S, M, N, yf, ldy);
    return (int) cudaGetLastError();
}

}
