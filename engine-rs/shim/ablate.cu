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
