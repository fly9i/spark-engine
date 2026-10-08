// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdlib>

// A fixed chain's convolution history is immutable base[3,C] followed by
// projected[T,C]. Every token/channel owns distinct activation/state outputs;
// no block reads another block's writable state. The four products and three
// additions retain the eager tree's left-associated FP32 operation order.
template<bool write_states,bool silu=false>
__global__ void kda_conv_chain(const float* base,const float* projected,
    const float* weights,float* conv,float* states,int width) {
    int token=blockIdx.y,channel=blockIdx.x*blockDim.x+threadIdx.x;
    if(channel>=width)return;
    int first=token-3;
    float rows[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        int row=first+j;
        rows[j]=row<0?base[(row+3)*width+channel]:projected[row*width+channel];
    }
    float acc=__fmul_rn(weights[channel*4],rows[0]);
    #pragma unroll
    for(int j=1;j<4;++j)acc=__fadd_rn(acc,__fmul_rn(weights[channel*4+j],rows[j]));
    // GLM53_KDA_CONV_SILU=1: the caller's following SiLU (ATen x/(1+exp(-x)), exact expf and division, as
    // shared_gu.cu) applied here instead of a separate elementwise launch.
    if constexpr(silu) acc=__fdiv_rn(acc,__fadd_rn(1.f,::exp(-acc)));
    conv[token*width+channel]=acc;
    if constexpr(write_states) {
        #pragma unroll
        for(int j=0;j<3;++j)states[(token*3+j)*width+channel]=rows[j+1];
    }
}

extern "C" int glm53_kda_conv_chain_cuda(const float* base,const float* projected,
    const float* weights,float* conv,float* states,int tokens,int width,cudaStream_t stream) {
    if(!base||!projected||!weights||!conv||tokens<1||tokens>65535||(states&&tokens>16)||width<1)
        return static_cast<int>(cudaErrorInvalidValue);
    // Null explicitly requests activation-only output, without a placeholder.
    const char* sf=std::getenv("GLM53_KDA_CONV_SILU");const bool silu=sf&&sf[0]=='1'&&sf[1]==0;
    if(states){if(silu)kda_conv_chain<true,true><<<dim3((width+255)/256,tokens),256,0,stream>>>(base,projected,weights,conv,states,width);
        else kda_conv_chain<true><<<dim3((width+255)/256,tokens),256,0,stream>>>(base,projected,weights,conv,states,width);}
    else{if(silu)kda_conv_chain<false,true><<<dim3((width+255)/256,tokens),256,0,stream>>>(base,projected,weights,conv,nullptr,width);
        else kda_conv_chain<false><<<dim3((width+255)/256,tokens),256,0,stream>>>(base,projected,weights,conv,nullptr,width);}
    return static_cast<int>(cudaGetLastError());
}

// GLM53_KDA_MULTI_LAUNCH=1 (L0): kda_conv_chain<false,silu> for up to 8 sequences in one launch. Row r of the
// stacked projected/conv tensors belongs to the sequence whose [first, first+len) holds it; its window reads that
// sequence's base rows exactly as the per-sequence launch does.
struct KdaConvMulti {const float* base[8];int first[8];int len[8];int n;};
template<bool silu>
__global__ void kda_conv_chain_multi(const KdaConvMulti m,const float* projected,const float* weights,float* conv,int width){
    const int token=blockIdx.y,channel=blockIdx.x*blockDim.x+threadIdx.x;
    if(channel>=width)return;
    int s=0;
#pragma unroll 1
    for(;s<m.n-1;++s)if(token<m.first[s]+m.len[s])break;
    const int local=token-m.first[s];const float* base=m.base[s];const float* proj=projected+(size_t)m.first[s]*width;
    float rows[4];
#pragma unroll
    for(int j=0;j<4;++j){const int row=local-3+j;rows[j]=row<0?base[(row+3)*width+channel]:proj[row*width+channel];}
    float acc=__fmul_rn(weights[channel*4],rows[0]);
#pragma unroll
    for(int j=1;j<4;++j)acc=__fadd_rn(acc,__fmul_rn(weights[channel*4+j],rows[j]));
    if constexpr(silu) acc=__fdiv_rn(acc,__fadd_rn(1.f,::exp(-acc)));
    conv[(size_t)token*width+channel]=acc;
}
extern "C" int glm53_kda_conv_chain_multi_cuda(const void* table,const float* projected,const float* weights,float* conv,int tokens,int width,cudaStream_t stream){
    const KdaConvMulti m=*reinterpret_cast<const KdaConvMulti*>(table);
    if(m.n<1||m.n>8||tokens<1||tokens>65535||width<1)return static_cast<int>(cudaErrorInvalidValue);
    const char* sf=std::getenv("GLM53_KDA_CONV_SILU");const bool silu=sf&&sf[0]=='1'&&sf[1]==0;
    if(silu)kda_conv_chain_multi<true><<<dim3((width+255)/256,tokens),256,0,stream>>>(m,projected,weights,conv,width);
    else kda_conv_chain_multi<false><<<dim3((width+255)/256,tokens),256,0,stream>>>(m,projected,weights,conv,width);
    return static_cast<int>(cudaGetLastError());
}
// Same tree of additions and barriers as kda.cu::sum128. Keep this copy local
// to this translation unit so the existing recurrent kernel is untouched.
__device__ float chain_sum128(float v,float* shared) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int s=16;s;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
    if(lane==0)shared[warp]=v;
    __syncthreads();
    v=threadIdx.x<4?shared[threadIdx.x]:0.f;
    if(warp==0) {
        for(int s=16;s;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
        if(lane==0)shared[0]=v;
    }
    __syncthreads();
    return shared[0];
}

// Preserve every node's independently writable H, but read the immutable
// base only once and carry each state element in a register across the chain.
// No block reads another block's output. FMA stays disabled as in recurrent128.
__global__ void kda_recurrent_chain(const float* source,const float* q,const float* k,
    const float* v,const float* beta,const float* decay,float* states,float* out,
    int heads,int tokens) {
    int row=blockIdx.x,head=row/128,col=threadIdx.x;
    int state_index=row*128+col;
    float state=source[state_index];
    __shared__ float sums[4];
    for(int t=0;t<tokens;++t) {
        int i=(t*heads+head)*128+col;
        state=state*decay[i];
        float hk=chain_sum128(state*k[i],sums);
        float correction=beta[t*heads+head]*(v[t*heads*128+row]-hk);
        state=state+correction*k[i];
        states[t*heads*128*128+state_index]=state;
        __syncthreads();
        float result=chain_sum128(state*q[i],sums);
        if(col==0)out[t*heads*128+row]=result;
        // All lanes must finish consuming sums[0] before the next step reuses it.
        __syncthreads();
    }
}

extern "C" int glm53_kda_recurrent_chain_cuda(const float* base,const float* q,const float* k,
    const float* v,const float* beta,const float* decay,float* states,float* out,
    int heads,int tokens,cudaStream_t stream) {
    if(heads<1||tokens<1||tokens>16)return static_cast<int>(cudaErrorInvalidValue);
    kda_recurrent_chain<<<heads*128,128,0,stream>>>(base,q,k,v,beta,decay,states,out,heads,tokens);
    return static_cast<int>(cudaGetLastError());
}
