// extern "C" launchers for the EXL3 kernels (qwen_exl3.cuh)
#include "qwen_exl3.cuh"

template <int BITS>
static void gemv_mt(const void* xh, const void* trellis, void* part, int M, int K, int N, int kts, dim3 grid, cudaStream_t st)
{
    const half* x = (const half*) xh;
    const uint32_t* t = (const uint32_t*) trellis;
    float* p = (float*) part;
    switch ((M + 15) / 16)
    {
        case 1: qexl3::gemv_kernel<BITS, 2, 4, 4, 1><<<grid, 128, 0, st>>>(x, t, p, M, K, N, kts); break;
        case 2: qexl3::gemv_kernel<BITS, 2, 4, 4, 2><<<grid, 128, 0, st>>>(x, t, p, M, K, N, kts); break;
        case 3: qexl3::gemv_kernel<BITS, 2, 4, 4, 3><<<grid, 128, 0, st>>>(x, t, p, M, K, N, kts); break;
        default: qexl3::gemv_kernel<BITS, 2, 4, 4, 4><<<grid, 128, 0, st>>>(x, t, p, M, K, N, kts); break;
    }
}

template <int BITS>
static void gemv_fused_mt(const float* x, int64_t ldx, const half* suh, const uint32_t* t, float* part, const half* svh, float* y,
                          int64_t ldy, int* cnt, int M, int K, int N, int kts, dim3 grid, cudaStream_t st)
{
    switch ((M + 15) / 16)
    {
        case 1: qexl3::gemv_fused_kernel<BITS, 1><<<grid, 128, 0, st>>>(x, ldx, suh, t, part, svh, y, ldy, cnt, M, K, N, kts); break;
        case 2: qexl3::gemv_fused_kernel<BITS, 2><<<grid, 128, 0, st>>>(x, ldx, suh, t, part, svh, y, ldy, cnt, M, K, N, kts); break;
        case 3: qexl3::gemv_fused_kernel<BITS, 3><<<grid, 128, 0, st>>>(x, ldx, suh, t, part, svh, y, ldy, cnt, M, K, N, kts); break;
        default: qexl3::gemv_fused_kernel<BITS, 4><<<grid, 128, 0, st>>>(x, ldx, suh, t, part, svh, y, ldy, cnt, M, K, N, kts); break;
    }
}

template <int UB>
static void gemv_multi_ub(int M, int K, const qexl3::Multi& ml, cudaStream_t st)
{
    const dim3 grid(ml.blk0[ml.n]);
    switch ((M + 15) / 16)
    {
        case 1: qexl3::gemv_multi_kernel<1, UB><<<grid, 128, 0, st>>>(M, K, ml); break;
        case 2: qexl3::gemv_multi_kernel<2, UB><<<grid, 128, 0, st>>>(M, K, ml); break;
        case 3: qexl3::gemv_multi_kernel<3, UB><<<grid, 128, 0, st>>>(M, K, ml); break;
        default: qexl3::gemv_multi_kernel<4, UB><<<grid, 128, 0, st>>>(M, K, ml); break;
    }
}
static void gemv_multi_mt(int M, int K, const qexl3::Multi& ml, cudaStream_t st)
{
    bool uni = true;
    for (int i = 1; i < ml.n; ++i) uni &= ml.bits[i] == ml.bits[0];
    if (!uni) gemv_multi_ub<0>(M, K, ml, st);
    else if (ml.bits[0] == 6) gemv_multi_ub<6>(M, K, ml, st);
    else if (ml.bits[0] == 5) gemv_multi_ub<5>(M, K, ml, st);
    else gemv_multi_ub<4>(M, K, ml, st);
}

extern "C" {

// had_in of up to 4 linears from one read of x [M, K] fp32 (any M): xh[i] [M, K] fp16 = had128(x * suh[i]) (bitwise equal to
// qwen_exl3_had_in per linear)
int qwen_exl3_had_multi(const void* x, int64_t ldx, int M, int K, int n, const void* const* suh, void* const* xh, cudaStream_t st)
{
    if (n < 1 || n > qexl3::MAXL || K % 128) return -1;
    qexl3::Multi ml{};
    ml.n = n;
    for (int i = 0; i < n; ++i) { ml.suh[i] = (const half*) suh[i]; ml.xh[i] = (half*) xh[i]; }
    int64_t warps = (int64_t) M * (K / 128);
    qexl3::had_in_multi_kernel<<<(unsigned) ((warps * 32 + 255) / 256), 256, 0, st>>>((const float*) x, ldx, nullptr, nullptr, nullptr, M, K, ml);
    return (int) cudaGetLastError();
}

// Up to 4 decode linears (M <= 64 rows) on the same input x [M, K] fp32 (or, with mg / mn non-null, on the HC mix of
// g / n [M, 4K] fp32, also written to xout [M, K]): per linear i suh[i], trellis[i] (bits, all the same), svh[i], scratch
// xh[i] [M, K] fp16 and part[i] [S[i], M, N[i]] fp32, output y[i] [M, N[i]] fp32 (row stride ldy[i]). silu (n = 2): y[0] gets
// silu(linear 0) * linear 1 (y[1] unused). Three launches; per linear bitwise equal to had_in + gemv + finish. bits per linear.
// silu bit 1: one gemv launch per linear (only the transforms and finishes shared).
// silu bit 2 (4): fork a pool side stream (rs_stream_fork(1)) right after the input transform, so the caller can run
// latency-bound kernels on the mixed input xout / x next to the gemvs (QWEN_MOE_ROUTE_SIDE, QWEN_GDN_AB_SIDE).
extern "C" int rs_stream_fork(int n);
int qwen_exl3_multi(const void* x, int64_t ldx, const void* mg, const void* mn, void* xout, int M, int K, int n,
                    const void* const* suh, void* const* xh, const void* const* trellis, void* const* part, const int* N, const int* S,
                    const void* const* svh, void* const* y, const int64_t* ldy, const int* bits, int silu, cudaStream_t st)
{
    if (n < 1 || n > qexl3::MAXL || M > 64 || K % 128) return -1;
    qexl3::Multi ml{};
    ml.n = n;
    int blk = 0, fw = 0;
    for (int i = 0; i < n; ++i)
    {
        if (N[i] % 128) return -1;
        ml.suh[i] = (const half*) suh[i]; ml.xh[i] = (half*) xh[i]; ml.tr[i] = (const uint32_t*) trellis[i]; ml.part[i] = (float*) part[i];
        ml.N[i] = N[i]; ml.S[i] = S[i]; ml.kts[i] = (K / 16 + S[i] - 1) / S[i];
        ml.svh[i] = (const half*) svh[i]; ml.y[i] = (float*) y[i]; ml.ldy[i] = ldy[i];
        if (bits[i] < 4 || bits[i] > 6) return -2;
        ml.bits[i] = bits[i];
        ml.blk0[i] = blk; blk += N[i] / 128 * S[i];
        fw += M * (N[i] / 128);
    }
    ml.blk0[n] = blk;
    int warps = M * (K / 128);
    qexl3::had_in_multi_kernel<<<(warps * 32 + 255) / 256, 256, 0, st>>>((const float*) x, ldx, (const float*) mg, (const float*) mn,
                                                                         (float*) xout, M, K, ml);
    if ((silu & 4) && rs_stream_fork(1) != 0) return -4;
    if (silu & 2)   // separate gemv launches per linear (the transforms still shared)
        for (int i = 0; i < n; ++i)
        {
            const dim3 grid(N[i] / 128, S[i]);
            switch (bits[i])
            {
                case 4: gemv_mt<4>(ml.xh[i], ml.tr[i], ml.part[i], M, K, N[i], ml.kts[i], grid, st); break;
                case 5: gemv_mt<5>(ml.xh[i], ml.tr[i], ml.part[i], M, K, N[i], ml.kts[i], grid, st); break;
                default: gemv_mt<6>(ml.xh[i], ml.tr[i], ml.part[i], M, K, N[i], ml.kts[i], grid, st); break;
            }
        }
    else gemv_multi_mt(M, K, ml, st);
    silu &= 1;
    if (silu)
    {
        if (n != 2 || N[0] != N[1]) return -3;
        warps = M * (N[0] / 128);
        qexl3::finish_silu_kernel<<<(warps * 32 + 255) / 256, 256, 0, st>>>(M, ml, (float*) y[0]);
    }
    else qexl3::finish_multi_kernel<<<(fw * 32 + 255) / 256, 256, 0, st>>>(M, ml);
    return (int) cudaGetLastError();
}

// x [M, K] (fp32 if x_f32 else fp16, row stride ldx) -> xh [M, K] fp16
int qwen_exl3_had_in(const void* x, int x_f32, int64_t ldx, const void* suh, void* xh, int M, int K, cudaStream_t st)
{
    int warps = M * (K / 128), threads = 256;
    int blocks = (warps * 32 + threads - 1) / threads;
    if (x_f32)
        qexl3::had_in_kernel<float><<<blocks, threads, 0, st>>>((const float*) x, ldx, (const half*) suh, (half*) xh, M, K);
    else
        qexl3::had_in_kernel<half><<<blocks, threads, 0, st>>>((const half*) x, ldx, (const half*) suh, (half*) xh, M, K);
    return (int) cudaGetLastError();
}

// x [M <= 64, K] fp32 (row stride ldx) -> y [M, N] fp32 (row stride ldy) in one launch (had_in + gemv + finish);
// part: scratch [S, M, N] fp32; cnt: [N / 128] ints, zero (left zero)
int qwen_exl3_linear(const void* x, int64_t ldx, const void* suh, const void* trellis, const void* svh, void* part, void* y, int64_t ldy,
                     void* cnt, int M, int K, int N, int bits, int S, cudaStream_t st)
{
    if (M > 64 || K % 128 || N % 128) return -1;
    int Kt = K / 16, kts = (Kt + S - 1) / S;
    dim3 grid(N / 128, S);
    const float* xf = (const float*) x;
    const half* su = (const half*) suh;
    const half* sv = (const half*) svh;
    const uint32_t* t = (const uint32_t*) trellis;
    switch (bits)
    {
        case 4: gemv_fused_mt<4>(xf, ldx, su, t, (float*) part, sv, (float*) y, ldy, (int*) cnt, M, K, N, kts, grid, st); break;
        case 5: gemv_fused_mt<5>(xf, ldx, su, t, (float*) part, sv, (float*) y, ldy, (int*) cnt, M, K, N, kts, grid, st); break;
        case 6: gemv_fused_mt<6>(xf, ldx, su, t, (float*) part, sv, (float*) y, ldy, (int*) cnt, M, K, N, kts, grid, st); break;
        default: return -2;
    }
    return (int) cudaGetLastError();
}

// xh [M <= 64, K] fp16, trellis [K/16, N/16, 16*bits] int16 -> part [S, M, N] fp32 (one weight pass)
int qwen_exl3_gemv(const void* xh, const void* trellis, void* part, int M, int K, int N, int bits, int S, cudaStream_t st)
{
    if (M > 64 || K % 128 || N % 128) return -1;
    int Kt = K / 16, kts = (Kt + S - 1) / S;
    dim3 grid(N / 128, S);
    switch (bits)
    {
        case 4: gemv_mt<4>(xh, trellis, part, M, K, N, kts, grid, st); break;
        case 5: gemv_mt<5>(xh, trellis, part, M, K, N, kts, grid, st); break;
        case 6: gemv_mt<6>(xh, trellis, part, M, K, N, kts, grid, st); break;
        default: return -2;
    }
    return (int) cudaGetLastError();
}

// part [S, M, N] -> y [M, N] (fp32 if y_f32 else fp16, row stride ldy)
int qwen_exl3_finish(const void* part, int S, const void* svh, void* y, int y_f32, int64_t ldy, int M, int N, cudaStream_t st)
{
    int warps = M * (N / 128), threads = 256;
    int blocks = (warps * 32 + threads - 1) / threads;
    if (y_f32)
        qexl3::finish_kernel<float><<<blocks, threads, 0, st>>>((const float*) part, S, (const half*) svh, (float*) y, ldy, M, N);
    else
        qexl3::finish_kernel<half><<<blocks, threads, 0, st>>>((const float*) part, S, (const half*) svh, (half*) y, ldy, M, N);
    return (int) cudaGetLastError();
}

// part [M, N] fp16 (one slice) -> y [M, N] fp32 (row stride ldy)
int qwen_exl3_finish_h(const void* part, const void* svh, void* y, int64_t ldy, int M, int N, cudaStream_t st)
{
    int warps = M * (N / 128), threads = 256;
    int blocks = (warps * 32 + threads - 1) / threads;
    qexl3::finish_h_kernel<float><<<blocks, threads, 0, st>>>((const half*) part, (const half*) svh, (float*) y, ldy, M, N);
    return (int) cudaGetLastError();
}

// trellis, suh [K], svh [N] -> effective weight W^T [N, K] fp16 (row stride ldo): y = x @ W with no transforms
int qwen_exl3_fold(const void* trellis, const void* suh, const void* svh, int K, int N, int bits, void* out, int64_t ldo, cudaStream_t st)
{
    if (K % 128 || N % 128) return -1;
    const size_t sm = 128 * qexl3::FOLD_LD * 4;
    dim3 grid(N / 128, K / 128);
    #define FOLD(B) do { static bool a = false; if (!a) { cudaFuncSetAttribute(qexl3::fold_kernel<B>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm); a = true; } \
        qexl3::fold_kernel<B><<<grid, 256, sm, st>>>((const uint32_t*) trellis, (const half*) suh, (const half*) svh, K, N, (half*) out, ldo); } while (0)
    switch (bits) { case 4: FOLD(4); break; case 5: FOLD(5); break; case 6: FOLD(6); break; default: return -2; }
    #undef FOLD
    return (int) cudaGetLastError();
}

// trellis [K/16, N/16, 16*bits] -> inner weight w [K, N] fp16 (no Hadamard / scales)
int qwen_exl3_reconstruct(const void* trellis, void* w, int K, int N, int bits, cudaStream_t st)
{
    int64_t tiles = (int64_t) (K / 16) * (N / 16);
    unsigned blocks = (unsigned) ((tiles * 32 + 255) / 256);
    switch (bits)
    {
        case 4: qexl3::reconstruct_kernel<4><<<blocks, 256, 0, st>>>((const uint32_t*) trellis, (half*) w, K, N); break;
        case 5: qexl3::reconstruct_kernel<5><<<blocks, 256, 0, st>>>((const uint32_t*) trellis, (half*) w, K, N); break;
        case 6: qexl3::reconstruct_kernel<6><<<blocks, 256, 0, st>>>((const uint32_t*) trellis, (half*) w, K, N); break;
        default: return -2;
    }
    return (int) cudaGetLastError();
}

}
