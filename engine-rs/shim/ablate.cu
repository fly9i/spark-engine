// Directional ablation shared by both models (crate::ablate): y[r] -= (y[r] . d) d for R rows of width D (fp32, row
// stride ld), d a unit vector. One block per row: block-wide dot product in a fixed order, then the update.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void ablate_rows_kernel(float* __restrict__ y, int64_t ld, const float* __restrict__ d, int D)
{
    __shared__ float red[32];
    float* row = y + (int64_t) blockIdx.x * ld;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += row[i] * d[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float c = red[0];
    for (int i = threadIdx.x; i < D; i += blockDim.x) row[i] -= c * d[i];
}

extern "C" int spark_ablate_rows(void* y, int64_t ld, int R, const void* d, int D, cudaStream_t st)
{
    if (R <= 0) return 0;
    ablate_rows_kernel<<<R, 256, 0, st>>>((float*) y, ld, (const float*) d, D);
    return (int) cudaGetLastError();
}

// Batched: remove K orthonormal directions in one launch. d is [K, D] row-major (unit rows). One block per row:
// read the row once, accumulate all K dot products, then subtract sum_k c_k d_k. For orthonormal D this equals the
// sequential per-direction projection (up to fp rounding). K must be <= 16.
__global__ void ablate_rows_k_kernel(float* __restrict__ y, int64_t ld, const float* __restrict__ d, int D, int K)
{
    extern __shared__ float sm[];            // [K*32] warp partials, then [K] combined dots at sm + K*32
    float* row = y + (int64_t) blockIdx.x * ld;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    float acc[16];
    #pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float yi = row[i];
        for (int k = 0; k < K; ++k) acc[k] += yi * d[(int64_t) k * D + i];
    }
    for (int k = 0; k < K; ++k) {
        float v = acc[k];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) sm[k * 32 + warp] = v;
    }
    __syncthreads();
    float* c = sm + K * 32;
    if (warp == 0) {
        for (int k = 0; k < K; ++k) {
            float t = (lane < nwarp) ? sm[k * 32 + lane] : 0.f;
            for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
            if (lane == 0) c[k] = t;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float upd = 0.f;
        for (int k = 0; k < K; ++k) upd += c[k] * d[(int64_t) k * D + i];
        row[i] -= upd;
    }
}

extern "C" int spark_ablate_rows_k(void* y, int64_t ld, int R, const void* d, int D, int K, cudaStream_t st)
{
    if (R <= 0 || K <= 0) return 0;
    if (K > 16) return (int) cudaErrorInvalidValue;
    size_t shmem = (size_t)(K * 32 + K) * sizeof(float);
    ablate_rows_k_kernel<<<R, 256, shmem, st>>>((float*) y, ld, (const float*) d, D, K);
    return (int) cudaGetLastError();
}
