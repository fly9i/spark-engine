// One warp consumes the already-computed Float edge scores. No head, edge
// GEMM, unary addition, candidate membership or FP arithmetic is rewritten.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <limits.h>

// Pinned ATen SharedReduceOps.h detail::GreaterOrNan: NaN beats numeric,
// and equal numbers or two NaNs choose the lowest candidate index.
__device__ __forceinline__ bool greater_or_nan(float a,float b,int ia,int ib) {
    if(isnan(a))return isnan(b)?ia<ib:true;
    return a==b?ia<ib:a>b;
}
__global__ void draft_selector_path(const float* edges,const int64_t* ids,int64_t* path,int steps) {
    const int lane=threadIdx.x;int previous=0;
    for(int t=0;t<steps;++t) {
        float value=lane<16?edges[(t*16+previous)*16+lane]:-CUDART_INF_F;
        int index=lane<16?lane:INT_MAX;
        #pragma unroll
        for(int offset=16;offset;offset>>=1) {
            const float other=__shfl_down_sync(0xffffffffu,value,offset);
            const int oi=__shfl_down_sync(0xffffffffu,index,offset);
            if(greater_or_nan(other,value,oi,index)){value=other;index=oi;}
        }
        previous=__shfl_sync(0xffffffffu,index,0);
        if(lane==0)path[t]=ids[t*16+previous];
    }
}
extern "C" int glm53_draft_selector_cuda(const float* edges,const int64_t* ids,
    int64_t* path,int steps,cudaStream_t stream) {
    if(!edges||!ids||!path||steps<1||steps>7)return static_cast<int>(cudaErrorInvalidValue);
    draft_selector_path<<<1,32,0,stream>>>(edges,ids,path,steps);
    return static_cast<int>(cudaGetLastError());
}
