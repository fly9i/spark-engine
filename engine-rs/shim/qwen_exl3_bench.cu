// Variant sweep for the EXL3 GEMV (bench only, not linked into the engine).
#include "qwen_exl3.cuh"
template <int B, int TPW, int PF, int W>
static void go(const void* xh, const void* tr, void* part, int M, int K, int N, int S, cudaStream_t st)
{
    int kts = (K / 16 + S - 1) / S;
    dim3 grid(N / (16 * W * TPW), S);
    qexl3::gemv_kernel<B, TPW, PF, W><<<grid, W * 32, 0, st>>>((const half*) xh, (const uint32_t*) tr, (float*) part, M, K, N, kts);
}
template <int B, int ST>
static void go2(const void* xh, const void* tr, void* part, int M, int K, int N, int S, cudaStream_t st)
{
    int kts = (K / 16 + S - 1) / S;
    dim3 grid(N / 128, S);
    qexl3::gemv_v2_kernel<B, ST><<<grid, 128, 0, st>>>((const half*) xh, (const uint32_t*) tr, (float*) part, M, K, N, kts);
}
extern "C" int qwen_exl3_gemv_var(int var, const void* xh, const void* tr, void* part, int M, int K, int N, int bits, int S, cudaStream_t st)
{
    #define V(id, TPW, PF, W) case id: if (bits == 6) go<6, TPW, PF, W>(xh, tr, part, M, K, N, S, st); else go<4, TPW, PF, W>(xh, tr, part, M, K, N, S, st); break;
    switch (var)
    {
        V(0, 2, 4, 4) V(1, 1, 4, 4) V(2, 1, 8, 4) V(3, 2, 8, 4) V(4, 1, 8, 8) V(5, 1, 4, 8) V(6, 1, 16, 4) V(7, 2, 8, 2)
        case 10: if (bits == 6) go2<6, 4>(xh, tr, part, M, K, N, S, st); else go2<4, 4>(xh, tr, part, M, K, N, S, st); break;
        case 11: if (bits == 6) go2<6, 8>(xh, tr, part, M, K, N, S, st); else go2<4, 8>(xh, tr, part, M, K, N, S, st); break;
        case 12: if (bits == 6) go2<6, 12>(xh, tr, part, M, K, N, S, st); else go2<4, 12>(xh, tr, part, M, K, N, S, st); break;
        default: return -1;
    }
    return (int) cudaGetLastError();
}
