// SPDX-License-Identifier: MIT
// Proposal 2, prefill side (GLM53_C12_PREFILL=1): large-row GEMM reading the C12-coded weight (c12.cuh) directly,
// so the resident Half source can be released (GLM53_C12_FREE_SOURCE=1).
//   y[m, n] (row stride ldy) = half(x)[m, k] . W[n, k]^T, FP32 accumulate; rounded != 0: Half-rounded, widened.
// Structure of fp8_big2 (BM = BN = 128, BK = 32, 8 warps of 64 x 32, grouped rasterization, cp.async stages, the B
// tile converted once per block into a swizzled double-buffered Half tile, ldmatrix, mma.m16n8k16 in K order).
// B is decoded from m8/e4 (+ escapes) with c12_dec8, i.e. to exactly the Half weight the source held; the mma
// sequence along K is fp8_big's, which matched cuBLAS bitwise at the prefill shapes (bench/queue7). Gate end to end.
#pragma once
// Included by c12.cu (one translation unit: the c12.cuh kernels are defined once).
#include "c12.cuh"
#include <cuda_runtime.h>

namespace c12big {
constexpr int BM = 128, BN = 128, BK = 32, THREADS = 256, GROUP_M = 8;

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void cp8(void* dst, const void* src) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void cp_fence() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&a)[4], const void* p) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}
__device__ __forceinline__ void mma16816(const uint32_t (&a)[4], const uint32_t (&b)[2], float (&c)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ ((row >> 1) & 3); }

template <int STAGES>
__global__ __launch_bounds__(THREADS, 1)
void c12_big_kernel(const half* __restrict__ x, const C12View w, float* __restrict__ y, int m, int ldy, int rounded)
{
    static_assert(STAGES >= 3, "one coded stage in flight beyond the converted one");
    const int n = w.n, k = w.k;
    extern __shared__ __align__(16) unsigned char smem[];
    half* sa = reinterpret_cast<half*>(smem);                              // STAGES x [BM][BK] half (swizzled)
    uint8_t* sm8 = smem + STAGES * BM * BK * 2;                            // STAGES x [BN][4 lanes x 8 bytes]
    uint32_t* se4 = reinterpret_cast<uint32_t*>(sm8 + STAGES * BN * 32);   // STAGES x [BN][4 lanes] nibble words
    half* sh = reinterpret_cast<half*>(se4 + STAGES * BN * 4);             // 2 x [BN][BK] half (swizzled)
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int wm = warp >> 2, wn = warp & 3;

    const int tiles_m = (m + BM - 1) / BM, tiles_n = n / BN;
    const int pid = blockIdx.x, per_group = GROUP_M * tiles_n, group = pid / per_group, first_m = group * GROUP_M;
    const int gm = min(tiles_m - first_m, GROUP_M);
    const int tm = first_m + (pid % per_group) % gm, tn = (pid % per_group) / gm;
    const int m0 = tm * BM, n0 = tn * BN, k_tiles = k / BK;

    auto load = [&](int stage, int kt) {
        half* a = sa + stage * BM * BK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = i * THREADS + t, row = c >> 2, chunk = c & 3;
            int src = min(m0 + row, m - 1);
            cp16(a + row * BK + swz(row, chunk) * 8, x + (int64_t) src * k + kt * BK + chunk * 8);
        }
        // chunk kt of rows n0..n0+127: per row and lane t', 8 m8 bytes and one e4 word (c12.cuh storage order)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = i * THREADS + t, row = c >> 2, tp = c & 3;
            const int64_t r = n0 + row;
            cp8(sm8 + (stage * BN + row) * 32 + tp * 8, w.m8 + ((r * (k >> 6) + (kt >> 1)) * 4 + tp) * 16 + (kt & 1) * 8);
            cp4(se4 + (stage * BN + row) * 4 + tp, w.e4 + ((r * (k >> 7) + (kt >> 2)) * 4 + tp) * 16 + (kt & 3) * 4);
        }
    };
    // Thread t decodes 16 weights: row t>>1, lanes t' = 2*(t&1), 2*(t&1)+1 (K 16*(t&1) .. +15) -> half chunks 2*(t&1), +1.
    const int crow = t >> 1, cpart = t & 1;
    const uint32_t ceb = w.eb[n0 + crow];
    auto convert = [&](int stage, int buf, int kt) {
        const uint8_t* mp = sm8 + (stage * BN + crow) * 32 + cpart * 16;
        const uint2 w0 = *reinterpret_cast<const uint2*>(mp), w1 = *reinterpret_cast<const uint2*>(mp + 8);
        const uint32_t* ep = se4 + (stage * BN + crow) * 4 + cpart * 2;
        const int kb = kt * BK + cpart * 16;
#ifdef C12_BIG_NODECODE   // diagnostic (bench only): same data movement, no decode arithmetic
        const uint4 h0 = make_uint4(w0.x, w0.y, ep[0], ceb), h1 = make_uint4(w1.x, w1.y, ep[1], kb);
#else
        const uint4 h0 = c12_dec8(w0.x, w0.y, ep[0], ceb, w, n0 + crow, kb), h1 = c12_dec8(w1.x, w1.y, ep[1], ceb, w, n0 + crow, kb + 8);
#endif
        half* d = sh + buf * BN * BK + crow * BK;
        *reinterpret_cast<uint4*>(d + swz(crow, cpart * 2) * 8) = h0;
        *reinterpret_cast<uint4*>(d + swz(crow, cpart * 2 + 1) * 8) = h1;
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < k_tiles) load(s, s); cp_fence(); }
    cp_wait<STAGES - 2>();          // stage 0 landed
    __syncthreads();
    convert(0, 0, 0);

    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_hi = lane >> 4;
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_hi = (lane >> 3) & 1;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_wait<STAGES - 3>();      // stage kt+1 landed (A kt too)
        __syncthreads();            // half buffer kt&1 complete; everyone finished tile kt-1
        int nk = kt + STAGES - 1;
        if (nk < k_tiles) load(nk % STAGES, nk);
        cp_fence();
        if (kt + 1 < k_tiles) convert((kt + 1) % STAGES, (kt + 1) & 1, kt + 1);
        const half* a = sa + (kt % STAGES) * BM * BK;
        const half* b = sh + (kt & 1) * BN * BK;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t fb[4][2];
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                uint32_t r[4];
                int row = wn * 32 + p * 16 + b_row;
                ldsm4(r, b + row * BK + swz(row, j * 2 + b_hi) * 8);
                fb[p * 2][0] = r[0]; fb[p * 2][1] = r[1]; fb[p * 2 + 1][0] = r[2]; fb[p * 2 + 1][1] = r[3];
            }
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb) {
                uint32_t fa[4];
                int row = wm * 64 + mb * 16 + a_row;
                ldsm4(fa, a + row * BK + swz(row, j * 2 + a_hi) * 8);
                #pragma unroll
                for (int nb = 0; nb < 4; ++nb) mma16816(fa, fb[nb], acc[mb][nb]);
            }
        }
    }
    cp_wait<0>();

    #pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
        const int col = n0 + wn * 32 + nb * 8 + c2;
        #pragma unroll
        for (int mb = 0; mb < 4; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * 64 + mb * 16 + g + h * 8;
                if (row >= m) continue;
                float v0 = acc[mb][nb][h * 2], v1 = acc[mb][nb][h * 2 + 1];
                if (rounded) { v0 = __half2float(__float2half_rn(v0)); v1 = __half2float(__float2half_rn(v1)); }
                *reinterpret_cast<float2*>(y + (int64_t) row * ldy + col) = make_float2(v0, v1);
            }
    }
}

template <int STAGES>
int launch(const half* x, const C12View& w, float* y, int m, int ldy, int rounded, cudaStream_t st) {
    constexpr int bytes = STAGES * (BM * BK * 2 + BN * 32 + BN * 16) + 2 * BN * BK * 2;
    static bool init = false;
    if (!init) {
        cudaError_t e = cudaFuncSetAttribute(c12_big_kernel<STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e != cudaSuccess) return int(e);
        init = true;
    }
    int blocks = ((m + BM - 1) / BM) * (w.n / BN);
    c12_big_kernel<STAGES><<<blocks, THREADS, bytes, st>>>(x, w, y, m, ldy, rounded);
    return int(cudaGetLastError());
}
}  // namespace c12big

extern "C" int glm53_c12_big_cuda(const void* x, const void* m8, const void* e4, const void* eb, const int* ptr, const int* col, const void* val,
                                  float* y, int m, int n, int k, int ldy, int rounded, int stages, cudaStream_t st) {
    using namespace c12big;
    if (m < 1 || n % BN || k % 128 || ldy < n) return int(cudaErrorInvalidValue);
    const C12View w{(const uint8_t*)m8, (const uint8_t*)e4, (const uint8_t*)eb, ptr, col, (const uint16_t*)val, n, k};
    if (stages == 3) return launch<3>((const half*)x, w, y, m, ldy, rounded, st);
    return launch<4>((const half*)x, w, y, m, ldy, rounded, st);
}
