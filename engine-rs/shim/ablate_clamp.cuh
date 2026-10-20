// SPARK_ABLATE_CLAMP (crate::ablate::Clamp) device side, shared by the standalone row kernel (ablate.cu) and the MHC pre
// finish epilogue (mhc.cu). A row z [D] has c_j = z . d_j for K orthonormal directions d [K, D]; the excess
//   e_j = relu(c_j - tau_j) - relu(lo_j - c_j)            (no target: clip into [lo, tau])
//   e_j = [c_j > tau_j or c_j < lo_j] (c_j - target_j)    (target: a triggered component moves to the target)
// is removed: z -= sum_j e_j d_j. tau / lo / target point at the K thresholds of one site.
// Count gate (count != null, SPARK_ABLATE_CLAMP_MIN): count[row] accumulates, over the sites of one forward pass, how many
// sites had any component outside [lo, tau] for that row (reset: this site starts the pass); a site edits the row only
// once the running count (this site included) reaches min.
#pragma once
#include <cuda_runtime.h>

#define SPARK_CLAMP_MAX_K 16
struct SparkClamp { const float* d; const float* tau; const float* lo; const float* target; int k; int* count; int min; int reset; };

__device__ __forceinline__ float spark_clamp_excess(const SparkClamp& cl, int j, float c)
{
    const float t = cl.tau[j], l = cl.lo[j];
    if (cl.target) return (c > t || c < l) ? c - cl.target[j] : 0.f;
    return fmaxf(c - t, 0.f) - fmaxf(l - c, 0.f);
}

// Block-wide: thread partial dots acc[0..k) of row `row` -> excess e[0..k) in shared memory (all threads see it after
// return; all zero when the count gate holds the row back). red must hold k*32 floats, e k floats; blockDim.x a
// multiple of 32 (<= 1024). Contains __syncthreads (CTA-uniform).
__device__ __forceinline__ void spark_clamp_reduce(const SparkClamp& cl, const float* acc, float* red, float* e, int row)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
#pragma unroll
    for (int j = 0; j < SPARK_CLAMP_MAX_K; ++j) if (j < cl.k) {
        float v = acc[j];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[j * 32 + warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        bool trig = false;
        for (int j = 0; j < cl.k; ++j) {
            float t = lane < nwarp ? red[j * 32 + lane] : 0.f;
            for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
            if (lane == 0) { e[j] = spark_clamp_excess(cl, j, t); trig |= t > cl.tau[j] || t < cl.lo[j]; }
        }
        if (cl.count && lane == 0) {
            const int n = (cl.reset ? 0 : cl.count[row]) + (int) trig;
            cl.count[row] = n;
            if (n < cl.min) for (int j = 0; j < cl.k; ++j) e[j] = 0.f;
        }
    }
    __syncthreads();
}

// sum_j e_j d_j at column i.
__device__ __forceinline__ float spark_clamp_update(const SparkClamp& cl, const float* e, int64_t D, int i)
{
    float u = 0.f;
    for (int j = 0; j < cl.k; ++j) u += e[j] * cl.d[j * D + i];
    return u;
}
