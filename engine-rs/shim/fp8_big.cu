// P1b: large-row weight-only FP8 GEMM for prefill (GLM53_FP8_BIG=1).
//   y[m, col0 + n] = scale[n] * sum_k x[m,k] * fp8(w[n,k])          (FP32 accumulate)
//   rounded != 0: y is rounded to Half and stored widened (same boundary as rs_fp8_epilogue).
// x: Half [m, k] row-major; w: FP8 e4m3 [n, k] row-major (the resident decode copy, no second
// layout); y: FP32 with row stride ldy (column slices of a wider buffer allowed).
// Replaces "expand FP8 -> Half weight tensor, cuBLAS, separate epilogue": no Half weight is
// materialized, the weight is read once per M-tile group as FP8, and the scale/rounding is fused.
// Measured bitwise equal to the expand path at every prefill shape (M 1024/2048; bench/queue7/probe.out):
// per-k16 mma accumulation in K order matches cuBLAS's result there. Gate end to end before relying on it.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int BM = 128, BN = 128, BK = 32, THREADS = 256;
constexpr int B_STRIDE = 48;   // bytes per B smem row (32 data + 16 pad): conflict-free fragment reads
constexpr int GROUP_M = 8;     // grouped rasterization: GROUP_M M-tiles sweep N together (A stays in L2)

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
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
__device__ __forceinline__ uint32_t fp8x2_to_half2(uint16_t v) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}
// 16-byte chunk swizzle inside a 64-byte A row (4 chunks): ldmatrix phases hit distinct banks.
__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ ((row >> 1) & 3); }

template <int STAGES>
__global__ __launch_bounds__(THREADS, 1)
void fp8_big_kernel(const half* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ scale,
                    float* __restrict__ y, int m, int n, int k, int ldy, int rounded)
{
    extern __shared__ __align__(16) unsigned char smem[];
    half* sa = reinterpret_cast<half*>(smem);
    uint8_t* sb = smem + STAGES * BM * BK * 2;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int wm = warp >> 2, wn = warp & 3;               // 2 x 4 warps, warp tile 64 x 32

    const int tiles_m = (m + BM - 1) / BM, tiles_n = n / BN;
    const int pid = blockIdx.x;
    const int per_group = GROUP_M * tiles_n;
    const int group = pid / per_group;
    const int first_m = group * GROUP_M;
    const int gm = min(tiles_m - first_m, GROUP_M);
    const int tm = first_m + (pid % per_group) % gm;
    const int tn = (pid % per_group) / gm;
    const int m0 = tm * BM, n0 = tn * BN;
    const int k_tiles = k / BK;

    auto load = [&](int stage, int kt) {
        half* a = sa + stage * BM * BK;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = i * THREADS + t, row = c >> 2, chunk = c & 3;
            int src = min(m0 + row, m - 1);
            cp16(a + row * BK + swz(row, chunk) * 8, x + (int64_t) src * k + kt * BK + chunk * 8);
        }
        uint8_t* b = sb + stage * BN * B_STRIDE;
        int row = t >> 1, chunk = t & 1;
        cp16(b + row * B_STRIDE + chunk * 16, w + (int64_t) (n0 + row) * k + kt * BK + chunk * 16);
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

    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_hi = lane >> 4;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        int nk = kt + STAGES - 1;
        if (nk < k_tiles) load(nk % STAGES, nk);
        cp_fence();
        const half* a = sa + (kt % STAGES) * BM * BK;
        const uint8_t* b = sb + (kt % STAGES) * BN * B_STRIDE;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t fb[4][2];
            #pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                const uint8_t* p = b + (wn * 32 + nb * 8 + g) * B_STRIDE + j * 16 + c2;
                fb[nb][0] = fp8x2_to_half2(*reinterpret_cast<const uint16_t*>(p));
                fb[nb][1] = fp8x2_to_half2(*reinterpret_cast<const uint16_t*>(p + 8));
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
        const float s0 = scale[col], s1 = scale[col + 1];
        #pragma unroll
        for (int mb = 0; mb < 4; ++mb) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * 64 + mb * 16 + g + h * 8;
                if (row >= m) continue;
                float v0 = acc[mb][nb][h * 2] * s0, v1 = acc[mb][nb][h * 2 + 1] * s1;
                if (rounded) { v0 = __half2float(__float2half_rn(v0)); v1 = __half2float(__float2half_rn(v1)); }
                *reinterpret_cast<float2*>(y + (int64_t) row * ldy + col) = make_float2(v0, v1);
            }
        }
    }
}

// v2 (GLM53_FP8_BIG_KERNEL=2): the FP8 B tile is converted to Half once per block into a swizzled smem
// buffer (double buffered) and read with ldmatrix, instead of every warp converting its fragments (each B
// element was converted by both M-warps). One __syncthreads per K tile: iteration kt computes tile kt from
// half buffer kt&1 while converting FP8 stage kt+1 into the other buffer. Same mma sequence and K order as v1.
template <int STAGES, int MINB>
__global__ __launch_bounds__(THREADS, MINB)
void fp8_big2_kernel(const half* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ scale,
                     float* __restrict__ y, int m, int n, int k, int ldy, int rounded)
{
    static_assert(STAGES >= 3, "v2 keeps one FP8 stage in flight beyond the converted one");
    extern __shared__ __align__(16) unsigned char smem[];
    half* sa = reinterpret_cast<half*>(smem);                       // STAGES x [BM][BK] half (swizzled)
    uint8_t* sb = smem + STAGES * BM * BK * 2;                      // STAGES x [BN][BK] fp8 (dense rows)
    half* sh = reinterpret_cast<half*>(sb + STAGES * BN * BK);      // 2 x [BN][BK] half (swizzled)
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
        int row = t >> 1, chunk = t & 1;
        cp16(sb + stage * BN * BK + row * BK + chunk * 16, w + (int64_t) (n0 + row) * k + kt * BK + chunk * 16);
    };
    // Thread t converts 16 FP8 values: row t>>1, K bytes (t&1)*16 .. +15 -> half chunks 2*(t&1), 2*(t&1)+1.
    auto convert = [&](int stage, int buf) {
        const int row = t >> 1, part = t & 1;
        uint4 raw = *reinterpret_cast<const uint4*>(sb + stage * BN * BK + row * BK + part * 16);
        const uint16_t* r16 = reinterpret_cast<const uint16_t*>(&raw);
        uint4 h0, h1;
        h0.x = fp8x2_to_half2(r16[0]); h0.y = fp8x2_to_half2(r16[1]); h0.z = fp8x2_to_half2(r16[2]); h0.w = fp8x2_to_half2(r16[3]);
        h1.x = fp8x2_to_half2(r16[4]); h1.y = fp8x2_to_half2(r16[5]); h1.z = fp8x2_to_half2(r16[6]); h1.w = fp8x2_to_half2(r16[7]);
        half* d = sh + buf * BN * BK + row * BK;
        *reinterpret_cast<uint4*>(d + swz(row, part * 2) * 8) = h0;
        *reinterpret_cast<uint4*>(d + swz(row, part * 2 + 1) * 8) = h1;
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
    convert(0, 0);

    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_hi = lane >> 4;
    const int b_row = (lane & 7) + ((lane >> 4) << 3), b_hi = (lane >> 3) & 1;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_wait<STAGES - 3>();      // stage kt+1 landed (A kt too)
        __syncthreads();            // half buffer kt&1 complete; everyone finished tile kt-1
        int nk = kt + STAGES - 1;
        if (nk < k_tiles) load(nk % STAGES, nk);
        cp_fence();
        if (kt + 1 < k_tiles) convert((kt + 1) % STAGES, (kt + 1) & 1);
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
        const float s0 = scale[col], s1 = scale[col + 1];
        #pragma unroll
        for (int mb = 0; mb < 4; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * 64 + mb * 16 + g + h * 8;
                if (row >= m) continue;
                float v0 = acc[mb][nb][h * 2] * s0, v1 = acc[mb][nb][h * 2 + 1] * s1;
                if (rounded) { v0 = __half2float(__float2half_rn(v0)); v1 = __half2float(__float2half_rn(v1)); }
                *reinterpret_cast<float2*>(y + (int64_t) row * ldy + col) = make_float2(v0, v1);
            }
    }
}

template <int STAGES, int MINB>
int launch2(const void* x, const void* w, const float* s, float* y, int m, int n, int k, int ldy, int rounded, cudaStream_t st) {
    constexpr int bytes = STAGES * (BM * BK * 2 + BN * BK) + 2 * BN * BK * 2;
    static bool init = false;
    if (!init) {
        cudaError_t e = cudaFuncSetAttribute(fp8_big2_kernel<STAGES, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e != cudaSuccess) return int(e);
        init = true;
    }
    int blocks = ((m + BM - 1) / BM) * (n / BN);
    fp8_big2_kernel<STAGES, MINB><<<blocks, THREADS, bytes, st>>>(static_cast<const half*>(x), static_cast<const uint8_t*>(w), s, y, m, n, k, ldy, rounded);
    return int(cudaGetLastError());
}

template <int STAGES>
int launch(const void* x, const void* w, const float* s, float* y, int m, int n, int k, int ldy, int rounded, cudaStream_t st) {
    constexpr int bytes = STAGES * (BM * BK * 2 + BN * B_STRIDE);
    static bool init = false;
    if (!init) {
        cudaError_t e = cudaFuncSetAttribute(fp8_big_kernel<STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e != cudaSuccess) return int(e);
        init = true;
    }
    int blocks = ((m + BM - 1) / BM) * (n / BN);
    fp8_big_kernel<STAGES><<<blocks, THREADS, bytes, st>>>(static_cast<const half*>(x), static_cast<const uint8_t*>(w), s, y, m, n, k, ldy, rounded);
    return int(cudaGetLastError());
}
// v3 (codes 31/32/33): v1's per-warp fragment conversion, generalized tile. BKT = K bytes/halves per stage
// (32 or 64: fewer barriers per K), BNT = block N width (128 or 256: warp tile 64 x BNT/4, so each A ldmatrix
// feeds twice the mma). Same mma order along K as v1.
template <int BKT, int BNT, int STAGES>
__global__ __launch_bounds__(THREADS, 1)
void fp8_big3_kernel(const half* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ scale,
                     float* __restrict__ y, int m, int n, int k, int ldy, int rounded)
{
    constexpr int CH = BKT / 8;                  // 16-byte A chunks per row
    constexpr int BS = BKT + 16;                 // B smem row stride (bytes), conflict-free fragment reads
    constexpr int NB = BNT / 32;                 // n8 blocks per warp (warp covers BNT/4 columns)
    constexpr int A_CHUNKS = BM * CH, B_CHUNKS = BNT * BKT / 16;
    extern __shared__ __align__(16) unsigned char smem[];
    half* sa = reinterpret_cast<half*>(smem);
    uint8_t* sb = smem + STAGES * BM * BKT * 2;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int wm = warp >> 2, wn = warp & 3;
    auto sw = [](int row, int chunk) { return CH == 4 ? (chunk ^ ((row >> 1) & 3)) : (chunk ^ (row & 7)); };

    const int tiles_m = (m + BM - 1) / BM, tiles_n = n / BNT;
    const int pid = blockIdx.x, per_group = GROUP_M * tiles_n, group = pid / per_group, first_m = group * GROUP_M;
    const int gm = min(tiles_m - first_m, GROUP_M);
    const int tm = first_m + (pid % per_group) % gm, tn = (pid % per_group) / gm;
    const int m0 = tm * BM, n0 = tn * BNT, k_tiles = k / BKT;

    auto load = [&](int stage, int kt) {
        half* a = sa + stage * BM * BKT;
        #pragma unroll
        for (int i = 0; i < (A_CHUNKS + THREADS - 1) / THREADS; ++i) {
            int c = i * THREADS + t; if (c >= A_CHUNKS) break;
            int row = c / CH, chunk = c % CH, src = min(m0 + row, m - 1);
            cp16(a + row * BKT + sw(row, chunk) * 8, x + (int64_t) src * k + kt * BKT + chunk * 8);
        }
        uint8_t* b = sb + stage * BNT * BS;
        #pragma unroll
        for (int i = 0; i < (B_CHUNKS + THREADS - 1) / THREADS; ++i) {
            int c = i * THREADS + t; if (c >= B_CHUNKS) break;
            int row = c / (BKT / 16), chunk = c % (BKT / 16);
            cp16(b + row * BS + chunk * 16, w + (int64_t) (n0 + row) * k + kt * BKT + chunk * 16);
        }
    };

    float acc[4][NB][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < NB; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < k_tiles) load(s, s); cp_fence(); }
    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_hi = lane >> 4;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        int nk = kt + STAGES - 1;
        if (nk < k_tiles) load(nk % STAGES, nk);
        cp_fence();
        const half* a = sa + (kt % STAGES) * BM * BKT;
        const uint8_t* b = sb + (kt % STAGES) * BNT * BS;
        #pragma unroll
        for (int j = 0; j < BKT / 16; ++j) {
            uint32_t fb[NB][2];
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                const uint8_t* p = b + (wn * (BNT / 4) + nb * 8 + g) * BS + j * 16 + c2;
                fb[nb][0] = fp8x2_to_half2(*reinterpret_cast<const uint16_t*>(p));
                fb[nb][1] = fp8x2_to_half2(*reinterpret_cast<const uint16_t*>(p + 8));
            }
            #pragma unroll
            for (int mb = 0; mb < 4; ++mb) {
                uint32_t fa[4];
                int row = wm * 64 + mb * 16 + a_row;
                ldsm4(fa, a + row * BKT + sw(row, j * 2 + a_hi) * 8);
                #pragma unroll
                for (int nb = 0; nb < NB; ++nb) mma16816(fa, fb[nb], acc[mb][nb]);
            }
        }
    }
    cp_wait<0>();
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        const int col = n0 + wn * (BNT / 4) + nb * 8 + c2;
        const float s0 = scale[col], s1 = scale[col + 1];
        #pragma unroll
        for (int mb = 0; mb < 4; ++mb)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * 64 + mb * 16 + g + h * 8;
                if (row >= m) continue;
                float v0 = acc[mb][nb][h * 2] * s0, v1 = acc[mb][nb][h * 2 + 1] * s1;
                if (rounded) { v0 = __half2float(__float2half_rn(v0)); v1 = __half2float(__float2half_rn(v1)); }
                *reinterpret_cast<float2*>(y + (int64_t) row * ldy + col) = make_float2(v0, v1);
            }
    }
}

template <int BKT, int BNT, int STAGES>
int launch3(const void* x, const void* w, const float* s, float* y, int m, int n, int k, int ldy, int rounded, cudaStream_t st) {
    if (n % BNT || k % BKT) return launch<3>(x, w, s, y, m, n, k, ldy, rounded, st);
    constexpr int bytes = STAGES * (BM * BKT * 2 + BNT * (BKT + 16));
    static bool init = false;
    if (!init) {
        cudaError_t e = cudaFuncSetAttribute(fp8_big3_kernel<BKT, BNT, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e != cudaSuccess) return int(e);
        init = true;
    }
    int blocks = ((m + BM - 1) / BM) * (n / BNT);
    fp8_big3_kernel<BKT, BNT, STAGES><<<blocks, THREADS, bytes, st>>>(static_cast<const half*>(x), static_cast<const uint8_t*>(w), s, y, m, n, k, ldy, rounded);
    return int(cudaGetLastError());
}

}  // namespace

// Requirements: n % 128 == 0, k % 32 == 0, ldy % 2 == 0, 16-byte aligned x/w rows, 8-byte aligned y.
extern "C" int glm53_fp8_big_cuda(const void* x, const void* w, const float* s, float* y, int m, int n, int k, int ldy,
                                  int rounded, int stages, cudaStream_t st) {
    if (m <= 0 || n % BN || k % BK || ldy < n || (ldy & 1) || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(w) & 15)
        || (reinterpret_cast<uintptr_t>(y) & 7) || (int64_t) ((m + BM - 1) / BM) * (n / BN) > 2147483647LL) return int(cudaErrorInvalidValue);
    switch (stages) {
        case 31: return launch3<64, 128, 3>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 32: return launch3<64, 128, 4>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 33: return launch3<32, 256, 3>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 34: return launch3<32, 256, 4>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 13: return launch2<3, 1>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 14: return launch2<4, 1>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 23: return launch2<3, 2>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 24: return launch2<4, 2>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 3: return launch<3>(x, w, s, y, m, n, k, ldy, rounded, st);
        case 5: return launch<5>(x, w, s, y, m, n, k, ldy, rounded, st);
        default: return launch<4>(x, w, s, y, m, n, k, ldy, rounded, st);
    }
}
