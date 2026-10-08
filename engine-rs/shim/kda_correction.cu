// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Match kda.cu::sum128 and kda_conv_chain.cu::chain_sum128 exactly. This
// translation unit must also be built with --fmad=false (no fast-math/FTZ).
__device__ float correction_sum128(float value, float* shared) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int s = 16; s; s >>= 1) value += __shfl_down_sync(0xffffffff, value, s);
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < 4 ? shared[threadIdx.x] : 0.f;
    if (warp == 0) {
        for (int s = 16; s; s >>= 1) value += __shfl_down_sync(0xffffffff, value, s);
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

// Original FP32 recurrent-chain order; retain H in registers and write its
// already-computed correction once per value row instead of all T full Hs.
__global__ void kda_correction_chain(const float* base, const float* q, const float* k,
    const float* v, const float* beta, const float* decay, float* correction,
    float* out, int heads, int tokens) {
    const int row = blockIdx.x, head = row / 128, col = threadIdx.x;
    const int state_index = row * 128 + col;
    float state = base[state_index];
    __shared__ float sums[4];
    for (int t = 0; t < tokens; ++t) {
        const int i = (t * heads + head) * 128 + col;
        state = __fmul_rn(state, decay[i]);
        const float hk = correction_sum128(__fmul_rn(state, k[i]), sums);
        const float u = __fmul_rn(beta[t * heads + head], __fsub_rn(v[t * heads * 128 + row], hk));
        if (col == 0) correction[t * heads * 128 + row] = u;
        state = __fadd_rn(state, __fmul_rn(u, k[i]));
        // Retain the original barrier between state update and output dot.
        __syncthreads();
        const float result = correction_sum128(__fmul_rn(state, q[i]), sums);
        if (col == 0) out[t * heads * 128 + row] = result;
        __syncthreads();
    }
}

// W05d (GLM53_KDA_CORR_WARP=1): one warp per value row, no barriers. Element j of each lane
// belongs to the original warp j (cols 32j..32j+31); each group reduces with the identical
// shfl_down tree, then groups combine as warp 0 did: (w0+w2)+(w1+w3). Bitwise identical.
__device__ __forceinline__ float warp_tree(float v){for(int s=16;s;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);return __shfl_sync(0xffffffff,v,0);}
__device__ __forceinline__ float sum128_warp(const float x[4]){
    // Warp 0 of the block path adds zero lanes before combining; +0.0f maps -0 to +0 identically.
    const float w0=__fadd_rn(warp_tree(x[0]),0.f),w1=__fadd_rn(warp_tree(x[1]),0.f),w2=__fadd_rn(warp_tree(x[2]),0.f),w3=__fadd_rn(warp_tree(x[3]),0.f);
    return __fadd_rn(__fadd_rn(w0,w2),__fadd_rn(w1,w3));
}
__global__ void kda_correction_chain_warp(const float* base,const float* q,const float* k,
    const float* v,const float* beta,const float* decay,float* correction,float* out,int heads,int tokens){
    const int row=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=heads*128)return;
    const int head=row/128;
    float state[4];
#pragma unroll
    for(int j=0;j<4;++j)state[j]=base[row*128+32*j+lane];
    for(int t=0;t<tokens;++t){
        const int b=(t*heads+head)*128;float prod[4];
#pragma unroll
        for(int j=0;j<4;++j){state[j]=__fmul_rn(state[j],decay[b+32*j+lane]);prod[j]=__fmul_rn(state[j],k[b+32*j+lane]);}
        const float hk=sum128_warp(prod);
        const float u=__fmul_rn(beta[t*heads+head],__fsub_rn(v[t*heads*128+row],hk));
        if(lane==0)correction[t*heads*128+row]=u;
#pragma unroll
        for(int j=0;j<4;++j){state[j]=__fadd_rn(state[j],__fmul_rn(u,k[b+32*j+lane]));prod[j]=__fmul_rn(state[j],q[b+32*j+lane]);}
        const float result=sum128_warp(prod);
        if(lane==0)out[t*heads*128+row]=result;
    }
}
// Chain fast path, step 1 (GLM53_KDA_CORR_WARP=2): kda_correction_chain_warp with the latency of the per-token loads taken off
// the dependency chain. A block's 8 warps are 8 value rows of one head (128 rows per head), which read the same k/q/decay
// vectors: they are staged once in shared memory for all tokens, and beta and the row's v are preloaded into registers.
// Every arithmetic operation and its order are unchanged, so outputs and corrections are bitwise those of the warp kernel.
__global__ void kda_correction_chain_warp_smem(const float* base,const float* q,const float* k,
    const float* v,const float* beta,const float* decay,float* correction,float* out,int heads,int tokens){
    __shared__ float sk[16][128],sq[16][128],sd[16][128];   // up to 16 chain tokens
    const int row0=blockIdx.x*8,head=row0/128,warp=threadIdx.x>>5,lane=threadIdx.x&31,row=row0+warp;
    float state[4];
#pragma unroll
    for(int j=0;j<4;++j)state[j]=base[row*128+32*j+lane];
    float vb[16],bb[16];
#pragma unroll
    for(int t=0;t<16;++t)if(t<tokens){vb[t]=v[t*heads*128+row];bb[t]=beta[t*heads+head];}
    for(int i=threadIdx.x;i<tokens*128;i+=blockDim.x){const int t=i>>7,c=i&127,b=(t*heads+head)*128+c;sk[t][c]=k[b];sq[t][c]=q[b];sd[t][c]=decay[b];}
    __syncthreads();
#pragma unroll
    for(int t=0;t<16;++t){
        if(t>=tokens)break;
        float prod[4];
#pragma unroll
        for(int j=0;j<4;++j){state[j]=__fmul_rn(state[j],sd[t][32*j+lane]);prod[j]=__fmul_rn(state[j],sk[t][32*j+lane]);}
        const float hk=sum128_warp(prod);
        const float u=__fmul_rn(bb[t],__fsub_rn(vb[t],hk));
        if(lane==0)correction[t*heads*128+row]=u;
#pragma unroll
        for(int j=0;j<4;++j){state[j]=__fadd_rn(state[j],__fmul_rn(u,sk[t][32*j+lane]));prod[j]=__fmul_rn(state[j],sq[t][32*j+lane]);}
        const float result=sum128_warp(prod);
        if(lane==0)out[t*heads*128+row]=result;
    }
}
// Chain fast path, step 1b (GLM53_KDA_CORR_WARP=3): one THREAD per value row, its 128 state values in registers. Each
// 128-term dot product is summed in exactly sum128_warp's order (per 32-column group the shfl_down tree
// ((x_i+x_{i+16})+(x_{i+8}+x_{i+24}))... down to lane 0, then +0.0f, then (w0+w2)+(w1+w3)), so outputs and corrections are
// bitwise those of the warp kernels, without the ~20 dependent shuffles per reduction. k/q/decay of all tokens are staged
// once per block in shared memory (every thread of the block is a row of the same head: broadcast reads).
__device__ __forceinline__ float tree32(const float* x){
    // shfl_down tree for lane 0: s = 16, 8, 4, 2, 1 (lane i adds lane i+s)
    float a[16];
#pragma unroll
    for(int i=0;i<16;++i)a[i]=__fadd_rn(x[i],x[i+16]);
#pragma unroll
    for(int i=0;i<8;++i)a[i]=__fadd_rn(a[i],a[i+8]);
#pragma unroll
    for(int i=0;i<4;++i)a[i]=__fadd_rn(a[i],a[i+4]);
#pragma unroll
    for(int i=0;i<2;++i)a[i]=__fadd_rn(a[i],a[i+2]);
    return __fadd_rn(a[0],a[1]);
}
__global__ void __launch_bounds__(128) kda_correction_chain_thread(const float* base,const float* q,const float* k,
    const float* v,const float* beta,const float* decay,float* correction,float* out,int heads,int tokens){
    __shared__ float sk[16][128],sq[16][128],sd[16][128];   // up to 16 chain tokens
    const int head=blockIdx.x,row=head*128+threadIdx.x;
    for(int i=threadIdx.x;i<tokens*128;i+=blockDim.x){const int t=i>>7,c=i&127,b=(t*heads+head)*128+c;sk[t][c]=k[b];sq[t][c]=q[b];sd[t][c]=decay[b];}
    float state[128];
    const float4* src=reinterpret_cast<const float4*>(base+(size_t)row*128);
#pragma unroll
    for(int j=0;j<32;++j){const float4 x=src[j];state[4*j]=x.x;state[4*j+1]=x.y;state[4*j+2]=x.z;state[4*j+3]=x.w;}
    __syncthreads();
    for(int t=0;t<tokens;++t){
        // products of one 32-column group at a time (state[128] + 32 temporaries stay in registers)
        float w[4];
#pragma unroll
        for(int j=0;j<4;++j){float x[32];
#pragma unroll
            for(int i=0;i<32;++i){const int c=32*j+i;state[c]=__fmul_rn(state[c],sd[t][c]);x[i]=__fmul_rn(state[c],sk[t][c]);}
            w[j]=__fadd_rn(tree32(x),0.f);}
        const float hk=__fadd_rn(__fadd_rn(w[0],w[2]),__fadd_rn(w[1],w[3]));
        const float u=__fmul_rn(beta[t*heads+head],__fsub_rn(v[t*heads*128+row],hk));
        correction[t*heads*128+row]=u;
#pragma unroll
        for(int j=0;j<4;++j){float x[32];
#pragma unroll
            for(int i=0;i<32;++i){const int c=32*j+i;state[c]=__fadd_rn(state[c],__fmul_rn(u,sk[t][c]));x[i]=__fmul_rn(state[c],sq[t][c]);}
            w[j]=__fadd_rn(tree32(x),0.f);}
        out[t*heads*128+row]=__fadd_rn(__fadd_rn(w[0],w[2]),__fadd_rn(w[1],w[3]));
    }
}
// GLM53_KDA_MULTI_LAUNCH=1 (L0): kda_correction_chain_thread for up to 8 sequences in one launch (blockIdx.y =
// sequence; its rows start at first in the stacked q/k/v/beta/decay/correction/out). Every output is bitwise the
// per-sequence thread (or warp, CORR_WARP=4) kernel's.
struct KdaCorrMulti {const float* base[8];int first[8];int len[8];int n;};
__global__ void __launch_bounds__(128) kda_correction_chain_thread_multi(const KdaCorrMulti m,const float* q,const float* k,
    const float* v,const float* beta,const float* decay,float* correction,float* out,int heads){
    const int sidx=blockIdx.y;if(sidx>=m.n)return;
    const int tokens=m.len[sidx];const size_t off=(size_t)m.first[sidx]*heads;
    q+=off*128;k+=off*128;v+=off*128;decay+=off*128;correction+=off*128;out+=off*128;beta+=off;const float* base=m.base[sidx];
    __shared__ float sk[16][128],sq[16][128],sd[16][128];
    const int head=blockIdx.x,row=head*128+threadIdx.x;
    for(int i=threadIdx.x;i<tokens*128;i+=blockDim.x){const int t=i>>7,c=i&127,b=(t*heads+head)*128+c;sk[t][c]=k[b];sq[t][c]=q[b];sd[t][c]=decay[b];}
    float state[128];
    const float4* src=reinterpret_cast<const float4*>(base+(size_t)row*128);
#pragma unroll
    for(int j=0;j<32;++j){const float4 x=src[j];state[4*j]=x.x;state[4*j+1]=x.y;state[4*j+2]=x.z;state[4*j+3]=x.w;}
    __syncthreads();
    for(int t=0;t<tokens;++t){
        float w[4];
#pragma unroll
        for(int j=0;j<4;++j){float x[32];
#pragma unroll
            for(int i=0;i<32;++i){const int c=32*j+i;state[c]=__fmul_rn(state[c],sd[t][c]);x[i]=__fmul_rn(state[c],sk[t][c]);}
            w[j]=__fadd_rn(tree32(x),0.f);}
        const float hk=__fadd_rn(__fadd_rn(w[0],w[2]),__fadd_rn(w[1],w[3]));
        const float u=__fmul_rn(beta[t*heads+head],__fsub_rn(v[t*heads*128+row],hk));
        correction[t*heads*128+row]=u;
#pragma unroll
        for(int j=0;j<4;++j){float x[32];
#pragma unroll
            for(int i=0;i<32;++i){const int c=32*j+i;state[c]=__fadd_rn(state[c],__fmul_rn(u,sk[t][c]));x[i]=__fmul_rn(state[c],sq[t][c]);}
            w[j]=__fadd_rn(tree32(x),0.f);}
        out[t*heads*128+row]=__fadd_rn(__fadd_rn(w[0],w[2]),__fadd_rn(w[1],w[3]));
    }
}
// The warp kernel's body for the sequences of a multi table (CORR_WARP=4 sends chains of 1-2 tokens there).
__global__ void kda_correction_chain_warp_multi(const KdaCorrMulti m,const float* q,const float* k,
    const float* v,const float* beta,const float* decay,float* correction,float* out,int heads){
    const int sidx=blockIdx.y;if(sidx>=m.n)return;
    const int tokens=m.len[sidx];const size_t off=(size_t)m.first[sidx]*heads;
    q+=off*128;k+=off*128;v+=off*128;decay+=off*128;correction+=off*128;out+=off*128;beta+=off;const float* base=m.base[sidx];
    const int row=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=heads*128)return;
    const int head=row/128;
    float state[4];
#pragma unroll
    for(int j=0;j<4;++j)state[j]=base[row*128+32*j+lane];
    for(int t=0;t<tokens;++t){
        const int b=(t*heads+head)*128;float prod[4];
#pragma unroll
        for(int j=0;j<4;++j){state[j]=__fmul_rn(state[j],decay[b+32*j+lane]);prod[j]=__fmul_rn(state[j],k[b+32*j+lane]);}
        const float hk=sum128_warp(prod);
        const float u=__fmul_rn(beta[t*heads+head],__fsub_rn(v[t*heads*128+row],hk));
        if(lane==0)correction[t*heads*128+row]=u;
#pragma unroll
        for(int j=0;j<4;++j){state[j]=__fadd_rn(state[j],__fmul_rn(u,k[b+32*j+lane]));prod[j]=__fmul_rn(state[j],q[b+32*j+lane]);}
        const float result=sum128_warp(prod);
        if(lane==0)out[t*heads*128+row]=result;
    }
}
extern "C" int glm53_kda_correction_chain_multi_cuda(const void* table,const float* q,const float* k,const float* v,const float* beta,
    const float* decay,float* correction,float* out,int heads,cudaStream_t stream){
    const KdaCorrMulti m=*reinterpret_cast<const KdaCorrMulti*>(table);
    if(m.n<1||m.n>8||heads<1||heads>1024)return static_cast<int>(cudaErrorInvalidValue);
    for(int i=0;i<m.n;i++)if(m.len[i]<1||m.len[i]>16)return static_cast<int>(cudaErrorInvalidValue);
    // CORR_WARP=4's per-chain choice: thread kernel for chains of 3+ tokens, warp kernel below (bitwise equal either way).
    KdaCorrMulti a{},b{};
    for(int i=0;i<m.n;i++){KdaCorrMulti& d=m.len[i]>=3?a:b;d.base[d.n]=m.base[i];d.first[d.n]=m.first[i];d.len[d.n]=m.len[i];d.n++;}
    if(a.n)kda_correction_chain_thread_multi<<<dim3(heads,a.n),128,0,stream>>>(a,q,k,v,beta,decay,correction,out,heads);
    if(b.n)kda_correction_chain_warp_multi<<<dim3((heads*128+7)/8,b.n),256,0,stream>>>(b,q,k,v,beta,decay,correction,out,heads);
    return static_cast<int>(cudaGetLastError());
}
// GLM53_KDA_CORR_WARP=5: four threads per value row, thread j owns the row's 32-column group j (32 state values in
// registers). Each group's 128-term dot-product part is summed with tree32 (+0.0f) exactly as the thread kernel does, the
// four parts are exchanged inside the row's lane quad and combined as (w0+w2)+(w1+w3): outputs and corrections are bitwise
// those of the thread/warp kernels. 4x the blocks of the thread kernel (all SMs busy at 32 heads) and a quarter of its
// registers per thread; k/q/decay of all tokens are staged per block as before.
__global__ void __launch_bounds__(128) kda_correction_chain_quad(const float* base,const float* q,const float* k,
    const float* v,const float* beta,const float* decay,float* correction,float* out,int heads,int tokens){
    __shared__ float sk[16][128],sq[16][128],sd[16][128];   // up to 16 chain tokens
    const int head=blockIdx.x>>2,j=threadIdx.x&3,row=head*128+(blockIdx.x&3)*32+(threadIdx.x>>2),lane=threadIdx.x&31,qb=lane&~3;
    for(int i=threadIdx.x;i<tokens*128;i+=blockDim.x){const int t=i>>7,c=i&127,b=(t*heads+head)*128+c;sk[t][c]=k[b];sq[t][c]=q[b];sd[t][c]=decay[b];}
    float state[32];
    const float4* src=reinterpret_cast<const float4*>(base+(size_t)row*128+32*j);
#pragma unroll
    for(int i=0;i<8;++i){const float4 x=src[i];state[4*i]=x.x;state[4*i+1]=x.y;state[4*i+2]=x.z;state[4*i+3]=x.w;}
    __syncthreads();
    for(int t=0;t<tokens;++t){
        float x[32];
#pragma unroll
        for(int i=0;i<32;++i){const int c=32*j+i;state[i]=__fmul_rn(state[i],sd[t][c]);x[i]=__fmul_rn(state[i],sk[t][c]);}
        float w=__fadd_rn(tree32(x),0.f);
        float w0=__shfl_sync(0xffffffffu,w,qb),w1=__shfl_sync(0xffffffffu,w,qb+1),w2=__shfl_sync(0xffffffffu,w,qb+2),w3=__shfl_sync(0xffffffffu,w,qb+3);
        const float hk=__fadd_rn(__fadd_rn(w0,w2),__fadd_rn(w1,w3));
        const float u=__fmul_rn(beta[t*heads+head],__fsub_rn(v[t*heads*128+row],hk));
        if(j==0)correction[t*heads*128+row]=u;
#pragma unroll
        for(int i=0;i<32;++i){const int c=32*j+i;state[i]=__fadd_rn(state[i],__fmul_rn(u,sk[t][c]));x[i]=__fmul_rn(state[i],sq[t][c]);}
        w=__fadd_rn(tree32(x),0.f);
        w0=__shfl_sync(0xffffffffu,w,qb);w1=__shfl_sync(0xffffffffu,w,qb+1);w2=__shfl_sync(0xffffffffu,w,qb+2);w3=__shfl_sync(0xffffffffu,w,qb+3);
        if(j==0)out[t*heads*128+row]=__fadd_rn(__fadd_rn(w0,w2),__fadd_rn(w1,w3));
    }
}
extern "C" int glm53_kda_correction_chain_cuda(const float* base, const float* q,
    const float* k, const float* v, const float* beta, const float* decay,
    float* correction, float* out, int heads, int tokens, cudaStream_t stream) {
    if (!base || !q || !k || !v || !beta || !decay || !correction || !out ||
        heads < 1 || heads > 1024 || tokens < 1 || tokens > 16)
        return static_cast<int>(cudaErrorInvalidValue);
    const char* warp_flag=std::getenv("GLM53_KDA_CORR_WARP");
    // GLM53_KDA_CORR_WARP=4: per call, the faster of two bitwise-identical kernels (bench/kda_fast/corr_test on GB10:
    // warp 12.0/17.3 us vs thread 16.3/17.4 us at 1/2 tokens; thread 1.48x/2.13x at 4/8 tokens).
    if(warp_flag&&warp_flag[0]=='5'&&warp_flag[1]==0)
        kda_correction_chain_quad<<<heads*4,128,0,stream>>>(base,q,k,v,beta,decay,correction,out,heads,tokens);
    else if(warp_flag&&warp_flag[0]=='4'&&warp_flag[1]==0){
        if(tokens>=3)kda_correction_chain_thread<<<heads,128,0,stream>>>(base,q,k,v,beta,decay,correction,out,heads,tokens);
        else kda_correction_chain_warp<<<(heads*128+7)/8,256,0,stream>>>(base,q,k,v,beta,decay,correction,out,heads,tokens);
    }
    else if(warp_flag&&warp_flag[0]=='3'&&warp_flag[1]==0)
        kda_correction_chain_thread<<<heads,128,0,stream>>>(base,q,k,v,beta,decay,correction,out,heads,tokens);
    else if(warp_flag&&warp_flag[0]=='2'&&warp_flag[1]==0)
        kda_correction_chain_warp_smem<<<heads*16,256,0,stream>>>(base,q,k,v,beta,decay,correction,out,heads,tokens);
    else if(warp_flag&&warp_flag[0]=='1'&&warp_flag[1]==0)
        kda_correction_chain_warp<<<(heads*128+7)/8,256,0,stream>>>(base,q,k,v,beta,decay,correction,out,heads,tokens);
    else
    kda_correction_chain<<<heads * 128, 128, 0, stream>>>(
        base, q, k, v, beta, decay, correction, out, heads, tokens);
    return static_cast<int>(cudaGetLastError());
}

// Host ABI matches kda_correction.rs::ReplayEntry. The host descriptor is copied
// into kernel launch parameters, not retained or uploaded as a pointer table.
struct KdaReplayEntry {
    const float* base;
    const float* k;
    const float* decay;
    const float* correction;
    int64_t dst_offset;
    int32_t heads;
    int32_t tokens;
};
static_assert(sizeof(KdaReplayEntry) == 48, "KDA replay entry ABI");
static_assert(offsetof(KdaReplayEntry, dst_offset) == 32, "KDA replay offset ABI");
static_assert(offsetof(KdaReplayEntry, heads) == 40, "KDA replay heads ABI");
static_assert(offsetof(KdaReplayEntry, tokens) == 44, "KDA replay tokens ABI");
constexpr int KDA_REPLAY_MAX_LAYERS = 64;
struct KdaReplayParams {
    KdaReplayEntry entries[KDA_REPLAY_MAX_LAYERS];
    float* dst;
    int32_t layers;
    int32_t steps;
};
static_assert(sizeof(KdaReplayParams) <= 4096, "KDA replay launch parameter limit");

// One CTA owns 16 value rows and all 128 key columns. Once u is known there
// are no reductions: each matrix element replays the original RN operations.
__global__ void kda_correction_commit(const KdaReplayParams params) {
    const int layer = blockIdx.z, head = blockIdx.y;
    const KdaReplayEntry entry = params.entries[layer];
    if (head >= entry.heads) return; // CTA-uniform for heterogeneous layers.
    const int col = threadIdx.x & 127, parity = threadIdx.x >> 7;
    const int first_row = blockIdx.x * 16;
    float state[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int row = first_row + parity + 2 * j;
        state[j] = entry.base[(head * 128 + row) * 128 + col];
    }
    __shared__ float keys[128], decays[128], updates[16];
    for (int t = 0; t < params.steps; ++t) {
        const int vector_offset = (t * entry.heads + head) * 128;
        if (threadIdx.x < 128) {
            keys[threadIdx.x] = entry.k[vector_offset + threadIdx.x];
            decays[threadIdx.x] = entry.decay[vector_offset + threadIdx.x];
        }
        if (threadIdx.x < 16)
            updates[threadIdx.x] = entry.correction[vector_offset + first_row + threadIdx.x];
        __syncthreads();
        const float key = keys[col], decay = decays[col];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float decayed = __fmul_rn(state[j], decay);
            const float delta = __fmul_rn(updates[parity + 2 * j], key);
            state[j] = __fadd_rn(decayed, delta);
        }
        // Every consumer finishes before any producer overwrites this tile.
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int row = first_row + parity + 2 * j;
        params.dst[entry.dst_offset + (head * 128 + row) * 128 + col] = state[j];
    }
}

__global__ void kda_correction_commit_inplace(const KdaReplayParams params) {
    // W07: same replay arithmetic; each thread overwrites exactly the base elements it read first.
    const int layer = blockIdx.z, head = blockIdx.y;
    const KdaReplayEntry entry = params.entries[layer];
    if (head >= entry.heads) return; // CTA-uniform for heterogeneous layers.
    const int col = threadIdx.x & 127, parity = threadIdx.x >> 7;
    const int first_row = blockIdx.x * 16;
    float state[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int row = first_row + parity + 2 * j;
        state[j] = entry.base[(head * 128 + row) * 128 + col];
    }
    __shared__ float keys[128], decays[128], updates[16];
    for (int t = 0; t < params.steps; ++t) {
        const int vector_offset = (t * entry.heads + head) * 128;
        if (threadIdx.x < 128) {
            keys[threadIdx.x] = entry.k[vector_offset + threadIdx.x];
            decays[threadIdx.x] = entry.decay[vector_offset + threadIdx.x];
        }
        if (threadIdx.x < 16)
            updates[threadIdx.x] = entry.correction[vector_offset + first_row + threadIdx.x];
        __syncthreads();
        const float key = keys[col], decay = decays[col];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float decayed = __fmul_rn(state[j], decay);
            const float delta = __fmul_rn(updates[parity + 2 * j], key);
            state[j] = __fadd_rn(decayed, delta);
        }
        // Every consumer finishes before any producer overwrites this tile.
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int row = first_row + parity + 2 * j;
        const_cast<float*>(entry.base)[(head * 128 + row) * 128 + col] = state[j];
    }
}

extern "C" int glm53_kda_correction_commit_inplace_cuda(const void* opaque_entries,int layers,int steps,cudaStream_t stream){
    const KdaReplayEntry* entries=static_cast<const KdaReplayEntry*>(opaque_entries);
    if(!entries||layers<1||layers>KDA_REPLAY_MAX_LAYERS||steps<0||steps>16)return static_cast<int>(cudaErrorInvalidValue);
    KdaReplayParams params{};params.dst=nullptr;params.layers=layers;params.steps=steps;int max_heads=0;
    for(int i=0;i<layers;++i){const KdaReplayEntry e=entries[i];
        if(!e.base||!e.k||!e.decay||!e.correction||e.heads<1||e.heads>1024||e.tokens<1||e.tokens>16||steps>e.tokens)return static_cast<int>(cudaErrorInvalidValue);
        params.entries[i]=e;if(e.heads>max_heads)max_heads=e.heads;}
    kda_correction_commit_inplace<<<dim3(8,max_heads,layers),256,0,stream>>>(params);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int glm53_kda_correction_commit_cuda(const void* opaque_entries,
    int layers, int steps, float* dst, int64_t elements, cudaStream_t stream) {
    const KdaReplayEntry* entries = static_cast<const KdaReplayEntry*>(opaque_entries);
    if (!entries || !dst || layers < 1 || layers > KDA_REPLAY_MAX_LAYERS ||
        steps < 0 || steps > 16 || elements < 1)
        return static_cast<int>(cudaErrorInvalidValue);
    KdaReplayParams params{};
    params.dst = dst; params.layers = layers; params.steps = steps;
    int64_t expected_offset = 0;
    int max_heads = 0;
    for (int i = 0; i < layers; ++i) {
        const KdaReplayEntry entry = entries[i];
        if (!entry.base || !entry.k || !entry.decay || !entry.correction ||
            entry.heads < 1 || entry.heads > 1024 || entry.tokens < 1 || entry.tokens > 16 ||
            steps > entry.tokens || entry.dst_offset != expected_offset)
            return static_cast<int>(cudaErrorInvalidValue);
        expected_offset += static_cast<int64_t>(entry.heads) * 128 * 128;
        if (expected_offset > elements) return static_cast<int>(cudaErrorInvalidValue);
        if (entry.heads > max_heads) max_heads = entry.heads;
        params.entries[i] = entry;
    }
    if (expected_offset != elements) return static_cast<int>(cudaErrorInvalidValue);
    kda_correction_commit<<<dim3(8, max_heads, layers), 256, 0, stream>>>(params);
    return static_cast<int>(cudaGetLastError());
}

// GLM53_KDA_CONV_COMMIT_ONE=1: the selected convolution windows of every KDA layer in one launch. Window of node n =
// rows n+1..n+3 of the virtual concat(base[3], projected[T]); dst may be base itself (in-place commit), so each thread
// owns one column of one layer and reads its three source values before writing any. Pure copies: bitwise the ATen
// materialize + copy_ sequence (3 launches per layer).
struct KdaConvCommitEntry {const float* base; const float* projected; float* dst;};
struct KdaConvCommitList {KdaConvCommitEntry e[64];};
__global__ void kda_conv_commit_many(const KdaConvCommitList list,int layers,int node,int width){
    const int l=blockIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(l>=layers||c>=width)return;
    const KdaConvCommitEntry e=list.e[l];float v[3];
#pragma unroll
    for(int r=0;r<3;++r){const int row=node+1+r;v[r]=row<3?e.base[(long long)row*width+c]:e.projected[(long long)(row-3)*width+c];}
#pragma unroll
    for(int r=0;r<3;++r)e.dst[(long long)r*width+c]=v[r];
}
extern "C" int glm53_kda_conv_commit_many_cuda(const void* entries,int layers,int node,int width,cudaStream_t s){
    if(!entries||layers<1||layers>64||node<0||node>15||width<1)return int(cudaErrorInvalidValue);
    KdaConvCommitList list;memcpy(list.e,entries,sizeof(KdaConvCommitEntry)*layers);
    kda_conv_commit_many<<<dim3((width+255)/256,layers),256,0,s>>>(list,layers,node,width);
    return int(cudaGetLastError());
}
