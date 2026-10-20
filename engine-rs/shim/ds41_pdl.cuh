// SPDX-License-Identifier: MIT
// Programmatic dependent launch (docs/07): in the decode / verify CUDA graphs consecutive kernels are separated by the
// graph's launch gap (~1.5 us; ~3000 kernels per decode step). A kernel launched with pdl_launch may start while the
// previous kernel of the stream still runs; it calls pdl_entry() first, which lets its own successor launch early and
// then waits until the previous grid has completed and its writes are visible, so the results are the same as plain
// launches. Every kernel launched with pdl_launch must call pdl_entry() before touching memory. DS41_PDL=0: plain launches.
#pragma once
#include "pdl.cuh"
#include <cstdlib>

static inline bool pdl_on() {
  static int v = -1;
  if (v < 0) { const char* e = std::getenv("DS41_PDL"); v = !(e && e[0] == '0'); }
  return v;
}

template <typename... K, typename... A>
static inline cudaError_t pdl_launch(void (*kernel)(K...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, A&&... args) {
  return pdl_launch_if(pdl_on(), kernel, grid, block, smem, st, std::forward<A>(args)...);
}
