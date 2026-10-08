// Default-off native shared expert epilogue; no GEMM algorithm change.
// Match PyTorch cf30153c ActivationSiluKernel: float x/(1+::exp(-x)).
// Compile --fmad=false, default precise divide, no --use_fast_math or --ftz=true.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <stdint.h>

template<bool DEBUG>
__global__ void shared_gu(const half* gate,const half* up,half* out,float* stages,int count) {
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    float g=__half2float(gate[i]),u=__half2float(up[i]);
    // Equivalent scalar clamp with finite limits; retain NaN payload/sign and +/-0.
    g=g>10.f?10.f:g;u=u>10.f?10.f:u;u=u<-10.f?-10.f:u;
    const float denominator=__fadd_rn(1.f,::exp(-g));
    const float silu=__fdiv_rn(g,denominator);
    // This is a separate FP32 mul in the original graph, before down input Half.
    const float product=__fmul_rn(silu,u);
    out[i]=__float2half_rn(product);
    if constexpr(DEBUG){stages[i]=g;stages[count+i]=u;stages[2*count+i]=silu;stages[3*count+i]=product;}
}
// Half D7 (C12 group): gate and up from one row-concatenated launch, gu [rows, 2n] (gate cols 0..n, up n..2n).
// Same scalar operations as shared_gu, so the output equals it on the separate gate/up tensors.
template<typename O>
__global__ void shared_gu_packed(const half* gu,O* out,int rows,int n) {
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=rows*n)return;
    const int r=i/n,c=i-r*n;
    float g=__half2float(gu[(size_t)r*2*n+c]),u=__half2float(gu[(size_t)r*2*n+n+c]);
    g=g>10.f?10.f:g;u=u>10.f?10.f:u;u=u<-10.f?-10.f:u;
    const float denominator=__fadd_rn(1.f,::exp(-g));
    const float silu=__fdiv_rn(g,denominator);
    const half h=__float2half_rn(__fmul_rn(silu,u));
    if constexpr(sizeof(O)==4) out[i]=__half2float(h);   // GLM53_SHARED_GU_F32=1: the Half value, widened exactly
    else out[i]=h;
}
extern "C" int glm53_shared_gu_packed_cuda(const void* gu,void* out,int rows,int n,cudaStream_t stream) {
    if(!gu||!out||rows<=0||n<=0||(long long)rows*n>4194304)return int(cudaErrorInvalidValue);
    shared_gu_packed<half><<<(rows*n+255)/256,256,0,stream>>>((const half*)gu,(half*)out,rows,n);
    return int(cudaGetLastError());
}
extern "C" int glm53_shared_gu_cuda(const void* g,const void* u,void* out,float* stages,int count,cudaStream_t stream) {
    if(!g||!u||!out||count<=0||count>4194304)return int(cudaErrorInvalidValue);
    if(stages)shared_gu<true><<<(count+255)/256,256,0,stream>>>((const half*)g,(const half*)u,(half*)out,stages,count);
    else shared_gu<false><<<(count+255)/256,256,0,stream>>>((const half*)g,(const half*)u,(half*)out,nullptr,count);
    return int(cudaGetLastError());
}
extern "C" int glm53_shared_gu_packed_f32_cuda(const void* gu,float* out,int rows,int n,cudaStream_t stream) {
    if(!gu||!out||rows<=0||n<=0||(long long)rows*n>4194304)return int(cudaErrorInvalidValue);
    shared_gu_packed<float><<<(rows*n+255)/256,256,0,stream>>>((const half*)gu,out,rows,n);
    return int(cudaGetLastError());
}

// GLM53_PREFILL_SHARED_HALF=1 (L0): prefill shared-expert SwiGLU on the Half gate/up GEMM results with ATen's formulas
// (clamp keeps NaN; silu = x / (1 + expf(-x)); FP32 product), written as the Half down-projection input that
// mm16_partial's input conversion (__float2half_rn) would produce. Built with --fmad=false like the rest of the shim.
__global__ void shared_swiglu_hh(const __half* __restrict__ g,const __half* __restrict__ u,__half* __restrict__ out,long long n,float lim){
  const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  float a=__half2float(g[i]),b=__half2float(u[i]);
  a=a>lim?lim:a;b=b>lim?lim:b;b=b<-lim?-lim:b;          // shared_gu's clamp (NaN kept)
  const float s=__fdiv_rn(a,__fadd_rn(1.f,::exp(-a)));
  out[i]=__float2half_rn(__fmul_rn(s,b));
}
extern "C" int glm53_shared_swiglu_hh_cuda(const void* g,const void* u,void* out,long long n,float lim,cudaStream_t st){
  if(n<1)return int(cudaErrorInvalidValue);
  shared_swiglu_hh<<<(unsigned)((n+255)/256),256,0,st>>>((const __half*)g,(const __half*)u,(__half*)out,n,lim);return int(cudaGetLastError());
}
