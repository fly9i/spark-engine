// SPDX-License-Identifier: MIT
// Programmatic dependent launch (PDL), shared by every model's kernels (GLM, DeepSeek-V4.1, Qwen). Each model keeps its
// own switch and launch helper (ds41_pdl.cuh: DS41_PDL, default on; fp8_skinny.cuh: GLM53_PDL=1; moe_exl3.cuh: the
// compile-time C::PDL) on top of these primitives.
//
// A kernel launched with the PDL attribute may start while the previous kernel of the stream still runs:
// - pdl_trigger(): lets this kernel's successor launch early (no memory semantics: the successor's pdl_wait() still
//   waits for this whole grid to complete and its writes to be visible, so nothing needs ordering before it);
// - pdl_wait(): waits until the previous grid has completed and its writes are visible. Everything that reads the
//   previous kernel's outputs or writes global memory comes after it; only data no earlier kernel writes (weights) may be
//   read before it. Use plain or volatile loads for activations: __ldg may be hoisted above the wait.
// Without the launch attribute both return at once.
#pragma once
#include <cuda_runtime.h>
#include <utility>

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
// first thing in a kernel that touches no memory before waiting: successor launches early, then wait
__device__ __forceinline__ void pdl_entry() { pdl_trigger(); pdl_wait(); }

// Launch with the PDL attribute when `pdl` (a plain launch otherwise); the arguments convert to the kernel's parameter
// types as in a <<<>>> launch.
template <typename... K, typename... A>
static inline cudaError_t pdl_launch_if(bool pdl, void (*kernel)(K...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, A&&... args) {
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<A>(args)...);
}
