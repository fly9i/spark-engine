// GLM-5.3-Flash routed experts (TP2: 288 local experts, top-8, I = 1024 per rank, mcg codebook): the shared EXL3 MoE
// (moe_exl3.cuh, config moex::GlmMoe). x [R, 4096] fp16, idx int64 / w fp16 [R, 8] (the engine's router outputs),
// out [R, 4096] fp32 = sum_k w[r][k] * expert(x[r]) in slot order (+ add [R, 4096] fp32 if given).
#include "moe_exl3.cuh"

extern "C" {

size_t glm_moe_ws_bytes(int R, int S) { return moex::ws_bytes<moex::GlmMoe>(R, S); }
// suh0: the layer's shared gate/up input scales (all experts equal, proven at load) or null
int glm_moe_experts(const void* x, int64_t ldx, int R, const void* idx, const void* w, const void* tab,
                    const void* add, void* out, void* ws, int S, const void* suh0, cudaStream_t st)
{
    return moex::experts<moex::GlmMoe>(x, ldx, R, idx, w, tab, add, out, ws, S, st, suh0);
}

}
