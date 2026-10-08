// Qwen3.8-Flash-Next MoE: softmax top-10 routing over 512 experts and the shared expert's small kernels; the routed
// experts are the shared EXL3 MoE (moe_exl3.cuh, config moex::QwenMoe).
#include "moe_exl3.cuh"

namespace qmoe {

constexpr int E = moex::QwenMoe::E, TOPK = moex::QwenMoe::TOPK;

// one block of 512 threads per row: softmax over logits, top-10 by repeated argmax (ties -> lower index)
__global__ __launch_bounds__(512) void route_kernel(const float* __restrict__ logits, int* __restrict__ idx,
                                                    float* __restrict__ w, int64_t ld = E)
{
    __shared__ float sv[E];
    __shared__ float red_v[16];
    __shared__ int red_i[16];
    __shared__ float top_v[TOPK];
    int r = blockIdx.x, e = threadIdx.x, lane = e & 31, wp = e >> 5;
    float x = logits[(int64_t) r * ld + e];
    // softmax
    float m = x;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red_v[wp] = m;
    __syncthreads();
    m = red_v[0];
    for (int i = 1; i < 16; ++i) m = fmaxf(m, red_v[i]);
    __syncthreads();
    float p = expf(x - m), s = p;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) red_v[wp] = s;
    __syncthreads();
    s = 0.f;
    for (int i = 0; i < 16; ++i) s += red_v[i];
    p = p / s;
    sv[e] = p;
    __syncthreads();
    for (int k = 0; k < TOPK; ++k)
    {
        float v = sv[e];
        int bi = e;
        for (int o = 16; o > 0; o >>= 1)
        {
            float ov = __shfl_xor_sync(0xffffffffu, v, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > v || (ov == v && oi < bi)) { v = ov; bi = oi; }
        }
        if (lane == 0) { red_v[wp] = v; red_i[wp] = bi; }
        __syncthreads();
        if (e == 0)
        {
            float bv = red_v[0];
            int bb = red_i[0];
            for (int i = 1; i < 16; ++i)
                if (red_v[i] > bv || (red_v[i] == bv && red_i[i] < bb)) { bv = red_v[i]; bb = red_i[i]; }
            idx[r * TOPK + k] = bb;
            top_v[k] = bv;
            sv[bb] = -1.0f;
        }
        __syncthreads();
    }
    if (e == 0)
    {
        float t = 0.f;
        for (int k = 0; k < TOPK; ++k) t += top_v[k];
        for (int k = 0; k < TOPK; ++k) w[r * TOPK + k] = top_v[k] / t;
    }
}

// shared expert (decode): h = silu(g) * u, then out = sigmoid(gl[r][col]) * d (the torch ops' expressions)
__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = g[i];
    const float sl = x / (1.0f + expf(-x));
    h[i] = sl * u[i];
}
__global__ void gate_scale_kernel(const float* __restrict__ gl, int64_t ldg, int col, const float* __restrict__ d, float* __restrict__ out, int R, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * N) return;
    const float a = gl[(int64_t) (i / N) * ldg + col];
    const float sg = 1.0f / (1.0f + expf(-a));
    out[i] = sg * d[i];
}

}  // namespace qmoe

extern "C" {

int qwen_moe_route(const void* logits, int R, void* idx, void* w, cudaStream_t st)
{
    qmoe::route_kernel<<<R, 512, 0, st>>>((const float*) logits, (int*) idx, (float*) w);
    return (int) cudaGetLastError();
}
// logits with row stride ld (the router GEMM output, no copy)
int qwen_moe_route_ld(const void* logits, int64_t ld, int R, void* idx, void* w, cudaStream_t st)
{
    qmoe::route_kernel<<<R, 512, 0, st>>>((const float*) logits, (int*) idx, (float*) w, ld);
    return (int) cudaGetLastError();
}
int qwen_moe_silu_mul(const void* g, const void* u, void* h, int n, cudaStream_t st)
{
    qmoe::silu_mul_kernel<<<(n + 255) / 256, 256, 0, st>>>((const float*) g, (const float*) u, (float*) h, n);
    return (int) cudaGetLastError();
}
int qwen_moe_gate_scale(const void* gl, int64_t ldg, int col, const void* d, void* out, int R, int N, cudaStream_t st)
{
    qmoe::gate_scale_kernel<<<(R * N + 255) / 256, 256, 0, st>>>((const float*) gl, ldg, col, (const float*) d, (float*) out, R, N);
    return (int) cudaGetLastError();
}
size_t qwen_moe_ws_bytes(int R, int S) { return moex::ws_bytes<moex::QwenMoe>(R, S); }
int qwen_moe_experts(const void* x, int64_t ldx, int R, const void* idx, const void* w, const void* tab,
                     const void* add, void* out, void* ws, int S, cudaStream_t st)
{
    return moex::experts<moex::QwenMoe>(x, ldx, R, idx, w, tab, add, out, ws, S, st);
}

}
