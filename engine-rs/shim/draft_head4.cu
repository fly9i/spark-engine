// SPDX-License-Identifier: MIT
// GLM53_DRAFT_HEAD_INT4=1 (draft side): coarse drafter head scores from an INT4 copy of the FP8 head (symmetric,
// one FP32 scale per 128 inputs), used only to preselect candidates; the candidates are rescored with the FP8 head.
// y[m,n] = sum_k half(x[m,k]) * ((q4[n,k]-8) * s[n,k/128]), FP32 accumulate. One warp per output row, each lane 32
// consecutive k per step (16 bytes of nibbles), X staged once per block as Half in shared memory.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
template<int M>
__global__ void __launch_bounds__(256) draft_head_int4(const __nv_bfloat16* __restrict__ x,const uint8_t* __restrict__ q4,const float* __restrict__ s,
    float* __restrict__ y,int m,int n,int k){
    extern __shared__ __half xs[];                                  // [m][k]
    for(int i=threadIdx.x;i<m*k;i+=blockDim.x)xs[i]=__float2half_rn(__bfloat162float(x[i]));
    __syncthreads();
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31,groups=k/128;
    for(int row=blockIdx.x*8+warp;row<n;row+=gridDim.x*8){
        const uint8_t* wr=q4+(long long)row*(k/2);const float* sr=s+(long long)row*groups;
        float acc[M];
#pragma unroll
        for(int r=0;r<M;++r)acc[r]=0.f;
        for(int it=0;it<k/1024;++it){
            const int k0=it*1024+lane*32;
            const uint4 w=*reinterpret_cast<const uint4*>(wr+k0/2);
            const float sc=sr[k0/128];
            const uint32_t ws[4]={w.x,w.y,w.z,w.w};
            float wf[32];
#pragma unroll
            for(int b=0;b<4;++b){
#pragma unroll
                for(int j=0;j<8;++j)wf[b*8+j]=(float)((int)((ws[b]>>(4*j))&15u)-8)*sc;
            }
#pragma unroll
            for(int r=0;r<M;++r){if(r<m){
                const uint4* xr=reinterpret_cast<const uint4*>(xs+r*k+k0);
                float a=acc[r];
#pragma unroll
                for(int v=0;v<4;++v){const uint4 u=xr[v];const __half2* h=reinterpret_cast<const __half2*>(&u);
#pragma unroll
                    for(int j=0;j<4;++j){const float2 f=__half22float2(h[j]);a=fmaf(f.x,wf[v*8+2*j],a);a=fmaf(f.y,wf[v*8+2*j+1],a);}}
                acc[r]=a;}}
        }
#pragma unroll
        for(int r=0;r<M;++r){float t=acc[r];for(int o=16;o;o>>=1)t+=__shfl_xor_sync(0xffffffff,t,o);acc[r]=t;}
        if(lane==0)for(int r=0;r<m;++r)y[(long long)r*n+row]=acc[r];
    }
}
// Tensor-core version (default when GLM53_DRAFT_HEAD_INT4_MMA != 0): one warp per 8 head rows (an n8 tile), K in
// 128-wide scale groups of 8 mma.m16n8k16 steps. Lane (g = lane/4, q = lane%4) loads 32 contiguous nibbles of head row
// n0+g at k = grp*128 + q*32 (16 bytes); step t uses k = grp*128 + q*32 + 4t + {0..3} as the fragment slots
// {2q, 2q+1, 2q+8, 2q+9}, and X (Half, shared memory) is read at the same k, so every product pairs the same k.
// Rows 8..15 of the A tile are zero (M <= 8). Per group: FP32 partial * scale[n, grp] added to the running total.
__device__ __forceinline__ uint32_t h2_from_nibbles(uint32_t two){   // two nibbles (bits 0-3, 4-7) -> half2(q-8)
    const uint32_t v=(two&0xFu)|((two&0xF0u)<<12);                   // low nibble -> low half, high -> high half
    const uint32_t h=v|0x64006400u;                                   // 1024 + q in each half
    const __half2 r=__hsub2(*reinterpret_cast<const __half2*>(&h),__floats2half2_rn(1032.f,1032.f));
    return *reinterpret_cast<const uint32_t*>(&r);
}
__global__ void __launch_bounds__(1024) draft_head_int4_mma(const __nv_bfloat16* __restrict__ x,const uint8_t* __restrict__ q4,const float* __restrict__ s,
    float* __restrict__ y,int m,int n,int k,bool tiled){
    extern __shared__ __half xs2[];                                  // [8][k+4] (padded rows: 2-way bank spread)
    const int ks=k+4;
    for(int i=threadIdx.x;i<8*k;i+=blockDim.x){const int r=i/k,c=i%k;xs2[r*ks+c]=r<m?__float2half_rn(__bfloat162float(x[i])):__float2half_rn(0.f);}
    __syncthreads();
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,q=lane&3,groups=k/128;
    const int wpb=blockDim.x>>5;
    for(int tile=blockIdx.x*wpb+warp;tile*8<n;tile+=gridDim.x*wpb){
        const int n0=tile*8;
        const uint8_t* wr=q4+(long long)(n0+g)*(k/2);
        // GLM53_DRAFT_HEAD4_TILED=1 (L0): q4 in draft_head4_tile's order, lane l's 16 bytes of (tile, group) at l*16.
        const uint8_t* tr=q4+(long long)tile*groups*512+lane*16;
        auto ld=[&](int gi)->uint4{return tiled?*reinterpret_cast<const uint4*>(tr+(size_t)gi*512):*reinterpret_cast<const uint4*>(wr+(gi*128+q*32)/2);};
        float tot0=0.f,tot1=0.f;
        // software pipeline: the next group's 16 weight bytes are in flight while this group's 8 MMAs run
        uint4 wnext=ld(0);
        uint4 wnext2=groups>1?ld(1):wnext;
        for(int grp=0;grp<groups;++grp){
            const int kb=grp*128+q*32;
            const uint4 w=wnext;wnext=wnext2;
            if(grp+2<groups)wnext2=ld(grp+2);
            const uint32_t ws[4]={w.x,w.y,w.z,w.w};
            float c[4]={0.f,0.f,0.f,0.f};
            const __half* xr=xs2+g*ks+kb;
#pragma unroll
            for(int t=0;t<8;++t){
                const uint32_t seg=(ws[t>>1]>>(16*(t&1)))&0xFFFFu;       // 4 nibbles: k = kb+4t .. kb+4t+3
                const uint32_t b0=h2_from_nibbles(seg&0xFFu),b1=h2_from_nibbles(seg>>8);
                const uint2 xv=*reinterpret_cast<const uint2*>(xr+4*t);   // x[g][kb+4t .. +3]
                const uint32_t a0=xv.x,a2=xv.y;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(0u),"r"(a2),"r"(0u),"r"(b0),"r"(b1));
            }
            const float* sc=s+(long long)(n0+2*q)*groups+grp;
            tot0=fmaf(c[0],sc[0],tot0);tot1=fmaf(c[1],sc[groups],tot1);
        }
        if(g<m){y[(long long)g*n+n0+2*q]=tot0;y[(long long)g*n+n0+2*q+1]=tot1;}
    }
}
static bool head4_tiled(){static const bool on=[]{const char* e=std::getenv("GLM53_DRAFT_HEAD4_TILED");return e&&e[0]=='1';}();return on;}
__global__ void draft_head4_tile(const uint8_t* __restrict__ src,uint8_t* __restrict__ dst,int n,int k){
    const int groups=k/128;const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x,total=(long long)(n/8)*groups*32;if(id>=total)return;
    const int lane=(int)(id&31);const long long tg=id>>5;const int grp=(int)(tg%groups),tile=(int)(tg/groups);const int g=lane>>2,q=lane&3;
    *reinterpret_cast<uint4*>(dst+(size_t)tg*512+lane*16)=*reinterpret_cast<const uint4*>(src+(size_t)(tile*8+g)*(k/2)+(grp*128+q*32)/2);
}
// GLM53_DRAFT_HEAD4_TILED=1: retile an INT4 head copy in place (load time; no-op when off). The tiled copy is read only by
// draft_head_int4_mma (the SIMT fallback refuses it).
extern "C" int glm53_draft_head4_tile_cuda(void* q4,int n,int k,cudaStream_t st){
    if(!head4_tiled())return 0;
    if(n%8||k%128)return int(cudaErrorInvalidValue);
    const size_t bytes=(size_t)n*k/2;void* tmp=nullptr;if(cudaMallocAsync(&tmp,bytes,st)!=cudaSuccess)return int(cudaErrorMemoryAllocation);
    cudaMemcpyAsync(tmp,q4,bytes,cudaMemcpyDeviceToDevice,st);
    const long long th=(long long)(n/8)*(k/128)*32;draft_head4_tile<<<(unsigned)((th+255)/256),256,0,st>>>((const uint8_t*)tmp,(uint8_t*)q4,n,k);
    cudaFreeAsync(tmp,st);return int(cudaGetLastError());
}
extern "C" int glm53_draft_head_int4_cuda(const void* x,const void* q4,const float* s,float* y,int m,int n,int k,cudaStream_t st){
    if(m<1||m>8||k%1024||n<1)return int(cudaErrorInvalidValue);
    const size_t sh=(size_t)m*k*sizeof(__half);
    static bool attr=false;if(!attr){cudaFuncSetAttribute(draft_head_int4<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,64*1024);attr=true;}
    if(sh>64*1024)return int(cudaErrorInvalidValue);
    int dev=0,sms=48;cudaGetDevice(&dev);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
    const char* mm=std::getenv("GLM53_DRAFT_HEAD_INT4_MMA");
    if(!(mm&&mm[0]=='0')&&n%8==0){
        static bool attr2=false;if(!attr2){cudaFuncSetAttribute(draft_head_int4_mma,cudaFuncAttributeMaxDynamicSharedMemorySize,64*1024+128);attr2=true;}
        const size_t sh8=(size_t)8*(k+4)*sizeof(__half);if(sh8>64*1024+128)return int(cudaErrorInvalidValue);
        // One 64 KiB X copy per SM (sm_121: ~100 KiB shared memory per SM), so the block carries the warps: 32.
        const int blocks=min((n/8+31)/32,sms);
        draft_head_int4_mma<<<blocks,1024,sh8,st>>>((const __nv_bfloat16*)x,(const uint8_t*)q4,s,y,m,n,k,head4_tiled());
        return int(cudaGetLastError());
    }
    if(head4_tiled())return int(cudaErrorInvalidValue);   // tiled copies are for the MMA kernel only
    const int blocks=min((n+7)/8,sms*3);
    draft_head_int4<8><<<blocks,256,sh,st>>>((const __nv_bfloat16*)x,(const uint8_t*)q4,s,y,m,n,k);
    return int(cudaGetLastError());
}
