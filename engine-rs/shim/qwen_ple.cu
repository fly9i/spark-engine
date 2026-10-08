// PLE (hashed n-gram embedding injected into the 4-stream residual) for Qwen3.8-Flash-Next, decode / verify rows.
//   emb[r] = concat_h (codebook(ring states) * scale + head_bias[h])           16 heads x 160 (exl3 n-gram rings)
//   key = GroupRMS(emb @ key_proj^T)*(1+nk) [4,2560];  value = emb @ value_proj^T [2560];  query = GroupRMS(X)*(1+nq)
//   d_s = key_s . query_s / sqrt(2560);  gate_s = sigmoid(sign(d) sqrt(max(|d|, 1e-6)))
//   gated_s = gate_s * value;  nrm = GroupRMS(gated)*(1+nc) [10240]
//   X += gated + silu(w0 nrm[t-9] + w1 nrm[t-6] + w2 nrm[t-3] + w3 nrm[t])    (window: the 9 previous rows)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace qple {

constexpr int H = 4, D = 2560, F = H * D, NH = 16, RD = 160, WIN = 9;

__device__ __forceinline__ float mul1(uint32_t state)
{
    uint32_t x = state * 0x83DCD12Du;
    uint32_t s = __dp4a(x, 0x01010101u, 0u);
    half h = __ushort_as_half((unsigned short) (0x6400u + s));
    return __half2float(__hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
}

// packed [R*16, words] int16 rings (row r, head h at r*16 + h); emb [R, 2560] fp32. One thread per element.
__global__ void decode_kernel(const uint16_t* __restrict__ packed, int words, int bits, const half* __restrict__ head_bias,
                              float* __restrict__ emb, int R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * NH * RD) return;
    int rh = i / RD, e = i % RD, h = rh % NH;
    const uint16_t* row = packed + (int64_t) rh * words;
    float scale = __half2float(__ushort_as_half(row[0]));
    const uint16_t* stream = row + 1;
    int nw = words - 1;
    uint32_t state = 0;
    int nsym = (15 + bits) / bits;
    for (int j = 0; j < nsym; ++j)
    {
        int pos = (e - j + RD) % RD;
        int b = pos * bits;
        uint32_t win = (uint32_t) stream[b >> 4] | ((uint32_t) stream[((b >> 4) + 1) % nw] << 16);
        uint32_t sym = (win >> (b & 15)) & ((1u << bits) - 1);
        state |= sym << (j * bits);
    }
    state &= 0xffffu;
    emb[(int64_t) rh * RD + e] = mul1(state) * scale + __half2float(head_bias[h * RD + e]);
}

__device__ __forceinline__ float block_sum(float v, float* red)
{
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += red[i];
    return s;
}

// grid (R, 4 streams), 256 threads. gated [R, 4, 2560], nrm [R, 10240] (chain buffer).
__global__ __launch_bounds__(256) void gate_kernel(const float* __restrict__ X, const float* __restrict__ key,
                                                   const float* __restrict__ value, const float* __restrict__ nk1,
                                                   const float* __restrict__ nq1, const float* __restrict__ nc1,
                                                   float* __restrict__ gated, float* __restrict__ nrm)
{
    __shared__ float red[8];
    int r = blockIdx.x, s = blockIdx.y;
    const float* x = X + ((int64_t) r * H + s) * D;
    const float* k = key + ((int64_t) r * H + s) * D;
    const float* v = value + (int64_t) r * D;
    float sx = 0.f, sk = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) { sx += x[i] * x[i]; sk += k[i] * k[i]; }
    float ix = rsqrtf(block_sum(sx, red) / D + 1e-6f);
    float ik = rsqrtf(block_sum(sk, red) / D + 1e-6f);
    float dot = 0.f;
    for (int i = threadIdx.x; i < D; i += 256)
        dot += (k[i] * ik * nk1[s * D + i]) * (x[i] * ix * nq1[s * D + i]);
    float d = block_sum(dot, red) * 0.019764235376052372f;          // 1 / sqrt(2560)
    float sg = d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f);
    float gate = 1.0f / (1.0f + __expf(-sg * sqrtf(fmaxf(fabsf(d), 1e-6f))));
    float sv = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) { float g = gate * v[i]; gated[((int64_t) r * H + s) * D + i] = g; sv += g * g; }
    float iv = rsqrtf(block_sum(sv, red) / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += 256)
        nrm[(int64_t) r * F + s * D + i] = gated[((int64_t) r * H + s) * D + i] * iv * nc1[s * D + i];
}

// X[r][c] += gated[r][c] + silu(conv) over history = window (9 rows, oldest first) ++ nrm rows
__global__ void conv_add_kernel(float* __restrict__ X, const float* __restrict__ gated, const float* __restrict__ nrm,
                                const float* __restrict__ win, const half* __restrict__ w, int R)
{
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) R * F) return;
    int r = i / F, c = i % F;
    auto hist = [&](int t) -> float { return t < WIN ? win[(int64_t) t * F + c] : nrm[(int64_t) (t - WIN) * F + c]; };
    int t = r + WIN;   // current row in history coordinates
    float conv = __half2float(w[c * 4 + 0]) * hist(t - 9) + __half2float(w[c * 4 + 1]) * hist(t - 6)
               + __half2float(w[c * 4 + 2]) * hist(t - 3) + __half2float(w[c * 4 + 3]) * hist(t);
    X[i] += gated[i] + conv / (1.0f + __expf(-conv));
}

// window <- last 9 of (window ++ nrm[0..n))
__global__ void commit_kernel(float* __restrict__ win, const float* __restrict__ nrm, int n, const int* __restrict__ nd)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (nd) n = min(max(*nd, 0), n);
    if (c >= F || n <= 0) return;
    float h[WIN];
    #pragma unroll
    for (int t = 0; t < WIN; ++t) h[t] = win[(int64_t) t * F + c];
    for (int t = 0; t < n; ++t)
    {
        #pragma unroll
        for (int q = 0; q < WIN - 1; ++q) h[q] = h[q + 1];
        h[WIN - 1] = nrm[(int64_t) t * F + c];
    }
    #pragma unroll
    for (int t = 0; t < WIN; ++t) win[(int64_t) t * F + c] = h[t];
}

}  // namespace qple

extern "C" {

int qwen_ple_decode(const void* packed, int words, int bits, const void* head_bias, void* emb, int R, cudaStream_t st)
{
    int n = R * qple::NH * qple::RD;
    qple::decode_kernel<<<(n + 255) / 256, 256, 0, st>>>((const uint16_t*) packed, words, bits, (const half*) head_bias, (float*) emb, R);
    return (int) cudaGetLastError();
}

// X [R,4,2560] fp32 updated in place; nrm [R,10240] scratch kept for the commit; gated [R,10240] scratch.
int qwen_ple_apply(void* X, const void* key, const void* value, const void* nk1, const void* nq1, const void* nc1,
                   const void* conv_w, const void* win, void* gated, void* nrm, int R, cudaStream_t st)
{
    qple::gate_kernel<<<dim3(R, qple::H), 256, 0, st>>>((const float*) X, (const float*) key, (const float*) value,
        (const float*) nk1, (const float*) nq1, (const float*) nc1, (float*) gated, (float*) nrm);
    int64_t n = (int64_t) R * qple::F;
    qple::conv_add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>((float*) X, (const float*) gated, (const float*) nrm,
        (const float*) win, (const half*) conv_w, R);
    return (int) cudaGetLastError();
}

// nd (device int, may be null): commit min(*nd, n) rows (one captured graph for any acceptance)
int qwen_ple_commit(void* win, const void* nrm, int n, const int* nd, cudaStream_t st)
{
    qple::commit_kernel<<<qple::F / 256, 256, 0, st>>>((float*) win, (const float*) nrm, n, nd);
    return (int) cudaGetLastError();
}

}
