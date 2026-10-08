// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Contiguous FP16 weights, FP32 activations rounded to FP16 as in mm16.
// A warp owns one output row; vector loads cover consecutive weight bytes.
template<bool RoundOutput>
__global__ void dense_gemv(const float* x,const half* w,float* y,int n,int k) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);
    if(row>=n)return;
    float sums[8]={};
    for(int col=lane*8;col<k;col+=256) {
        int4 packed=*reinterpret_cast<const int4*>(w+static_cast<long long>(row)*k+col);
        const half* values=reinterpret_cast<const half*>(&packed);
        #pragma unroll
        for(int j=0;j<8;++j) {
            float a=__half2float(__float2half_rn(x[col+j]));
            sums[j]=__fmaf_rn(a,__half2float(values[j]),sums[j]);
        }
    }
    float result=0.f;
    #pragma unroll
    for(int j=0;j<8;++j)result+=sums[j];
    for(int offset=16;offset;offset>>=1)result+=__shfl_down_sync(0xffffffff,result,offset);
    if(lane==0)y[row]=RoundOutput?__half2float(__float2half_rn(result)):result;
}

extern "C" int glm53_gemv_cuda(const float* x,const void* w,float* y,int n,int k,
                               int round_output,cudaStream_t stream) {
    if(round_output)dense_gemv<true><<<(n+3)/4,128,0,stream>>>(x,(const half*)w,y,n,k);
    else dense_gemv<false><<<(n+3)/4,128,0,stream>>>(x,(const half*)w,y,n,k);
    return static_cast<int>(cudaGetLastError());
}

// Small-N bandwidth candidate: each warp reuses a weight vector across up to
// eight query rows. It preserves input FP16 rounding and FP32 accumulation.
template<int Rows,bool RoundOutput>
__global__ void dense_small(const float* x,const half* w,float* y,int m,int n,int k) {
    const int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);
    if(row>=n)return;
    float sums[Rows][8]={};
    for(int col=lane*8;col<k;col+=256) {
        const int4 packed=*reinterpret_cast<const int4*>(w+static_cast<long long>(row)*k+col);
        const half* values=reinterpret_cast<const half*>(&packed);
        #pragma unroll
        for(int r=0;r<Rows;++r) {
            if(r<m) {
                #pragma unroll
                for(int j=0;j<8;++j) {
                    float a=__half2float(__float2half_rn(x[r*k+col+j]));
                    sums[r][j]=__fmaf_rn(a,__half2float(values[j]),sums[r][j]);
                }
            }
        }
    }
    #pragma unroll
    for(int r=0;r<Rows;++r) {
        float result=0.f;
        #pragma unroll
        for(int j=0;j<8;++j)result+=sums[r][j];
        for(int offset=16;offset;offset>>=1)result+=__shfl_down_sync(0xffffffff,result,offset);
        if(lane==0 && r<m)y[r*n+row]=RoundOutput?__half2float(__float2half_rn(result)):result;
    }
}
extern "C" int glm53_small_cuda(const float* x,const void* w,float* y,int m,int n,int k,
                                int round_output,cudaStream_t stream) {
    if(m<2||m>8||k%8||n<1)return int(cudaErrorInvalidValue);
    #define LAUNCH(R) if(round_output)dense_small<R,true><<<(n+3)/4,128,0,stream>>>(x,(const half*)w,y,m,n,k);else dense_small<R,false><<<(n+3)/4,128,0,stream>>>(x,(const half*)w,y,m,n,k)
    if(m<=2){LAUNCH(2);}else if(m<=4){LAUNCH(4);}else{LAUNCH(8);}
    #undef LAUNCH
    return static_cast<int>(cudaGetLastError());
}
