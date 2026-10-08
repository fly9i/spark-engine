#include <cstdlib>
// Standalone candidate: preserve every original BF16 materialization rounding.
// Build with --fmad=false, without fast-math/FTZ; do not fuse add/multiply stages.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float bf16_rn_float(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
__global__ void draft_conv_bf16(const __nv_bfloat16* x,const __nv_bfloat16* delta,
    const __nv_bfloat16* base,__nv_bfloat16* out,int n,int side,
    int64_t ds0,int64_t ds1,int64_t ds2) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    const int t=blockIdx.y;
    if(c>=4096)return;
    const int group=c/16;
    const int64_t di=t*ds0+group*ds2;
    const int bi=side*8192+c;
    const float c0=bf16_rn_float(__fadd_rn(__bfloat162float(base[bi]),__bfloat162float(delta[di])));
    const float c1=bf16_rn_float(__fadd_rn(__bfloat162float(base[bi+4096]),__bfloat162float(delta[di+ds1])));
    const float current=__bfloat162float(x[t*4096+c]);
    // The original shifted tensor starts with +0. Do not skip c1*0: Inf/NaN
    // coefficients must still produce NaN, and signed-zero rules must remain.
    const float shifted=t==0?__uint_as_float(0u):__bfloat162float(x[(t-1)*4096+c]);
    const float p0=bf16_rn_float(__fmul_rn(c0,current));
    const float p1=bf16_rn_float(__fmul_rn(c1,shifted));
    out[t*4096+c]=__float2bfloat16_rn(__fadd_rn(p0,p1));
}
extern "C" int glm53_draft_conv_cuda(const void* x,const void* delta,const void* base,
    void* out,int n,int side,int64_t ds0,int64_t ds1,int64_t ds2,cudaStream_t stream) {
    if(!x||!delta||!base||!out||n<1||n>8||side<0||side>1||ds0<=0||ds1<=0||ds2<=0)
        return static_cast<int>(cudaErrorInvalidValue);
    draft_conv_bf16<<<dim3(16,n),256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(delta),
        static_cast<const __nv_bfloat16*>(base),static_cast<__nv_bfloat16*>(out),n,side,ds0,ds1,ds2);
    return static_cast<int>(cudaGetLastError());
}

// ---- W08 drafter attention for graph capture: device-side history range, fixed launch shape ----
// q [8,32,128] bf16, slab k/v [cap,8,128] bf16 (history rows [begin, begin+count)), new k/v [8,8,128] bf16.
// meta (int64): [0]=q_pos0 (context len), [1]=k_pos0 (context start), [2]=begin, [3]=count.
// Visible iff q_pos - k_pos < 2048 (history); new block keys are always visible (|i-j|<8).
// Chunks 0..63 cover history keys [c*32,(c+1)*32); chunk 64 covers the 8 new keys.
#include <cuda_bf16.h>
constexpr int DA_CHUNKS=65; constexpr int DA_HIST=64, DA_KEYS=32;
// C3: kvh KV heads on this rank (grid.y), 4*kvh query heads; per (query, head) arithmetic is unchanged.
__global__ void __launch_bounds__(256) draft_attn_split(const __nv_bfloat16* __restrict__ q,const __nv_bfloat16* __restrict__ sk,const __nv_bfloat16* __restrict__ sv,
    const __nv_bfloat16* __restrict__ nk,const __nv_bfloat16* __restrict__ nv,const long long* __restrict__ meta,
    float* __restrict__ part_o,float* __restrict__ part_ml,int kvh){
    const int chunk=blockIdx.x,g=blockIdx.y,warp=threadIdx.x>>5,lane=threadIdx.x&31;const int H=4*kvh,V=8*H;
    const long long qpos0=meta[0],kpos0=meta[1],begin=meta[2],count=meta[3];
    // 4 query vectors per warp: vec v in 0..31 of group g -> query i=v/4, head h=g*4+v%4
    float qv[4][4];int qi[4],qh[4];
#pragma unroll
    for(int a=0;a<4;++a){const int v=warp*4+a;qi[a]=v>>2;qh[a]=g*4+(v&3);
        const __nv_bfloat16* src=q+((long long)qi[a]*H+qh[a])*128+lane*4;
#pragma unroll
        for(int d=0;d<4;++d)qv[a][d]=__bfloat162float(src[d])*0.08838834764831845f;}
    float m[4],l[4],acc[4][4];
#pragma unroll
    for(int a=0;a<4;++a){m[a]=-INFINITY;l[a]=0.f;
#pragma unroll
        for(int d=0;d<4;++d)acc[a][d]=0.f;}
    int keys;const __nv_bfloat16 *kb,*vb;long long first=0;
    if(chunk<DA_HIST){first=(long long)chunk*DA_KEYS;keys=(int)max(0LL,min((long long)DA_KEYS,count-first));
        kb=sk+((begin+first)*kvh+g)*128;vb=sv+((begin+first)*kvh+g)*128;}
    else{keys=8;kb=nk+(long long)g*128;vb=nv+(long long)g*128;}
    const int stride=kvh*128;
    for(int j=0;j<keys;++j){
        const __nv_bfloat16* kr=kb+(long long)j*stride+lane*4;const __nv_bfloat16* vr=vb+(long long)j*stride+lane*4;
        float kf[4],vf[4];
#pragma unroll
        for(int d=0;d<4;++d){kf[d]=__bfloat162float(kr[d]);vf[d]=__bfloat162float(vr[d]);}
#pragma unroll
        for(int a=0;a<4;++a){
            float s=0.f;
#pragma unroll
            for(int d=0;d<4;++d)s=__fmaf_rn(qv[a][d],kf[d],s);
            for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
            bool vis=true;
            if(chunk<DA_HIST){const long long qp=qpos0+qi[a],kp=kpos0+first+j;vis=(qp-kp)<2048;}
            if(!vis)continue;
            const float mn=fmaxf(m[a],s);const float scale=expf(m[a]-mn);const float p=expf(s-mn);
            l[a]=l[a]*scale+p;
#pragma unroll
            for(int d=0;d<4;++d)acc[a][d]=acc[a][d]*scale+p*vf[d];
            m[a]=mn;
        }
    }
#pragma unroll
    for(int a=0;a<4;++a){const long long vec=(long long)qi[a]*H+qh[a];
        float* o=part_o+((long long)chunk*V+vec)*128+lane*4;
#pragma unroll
        for(int d=0;d<4;++d)o[d]=acc[a][d];
        if(lane==0){part_ml[((long long)chunk*V+vec)*2]=m[a];part_ml[((long long)chunk*V+vec)*2+1]=l[a];}}
}
// GLM53_DRAFT_ATTN_PREFETCH=1: draft_attn_split with each lane's K/V slices of 8 keys loaded ahead as one 8-byte load per
// key (the plain kernel issues 4 two-byte loads per key and waits on them key by key). Every arithmetic operation and its
// order are unchanged: bitwise the same partials.
__global__ void __launch_bounds__(256) draft_attn_split_pf(const __nv_bfloat16* __restrict__ q,const __nv_bfloat16* __restrict__ sk,const __nv_bfloat16* __restrict__ sv,
    const __nv_bfloat16* __restrict__ nk,const __nv_bfloat16* __restrict__ nv,const long long* __restrict__ meta,
    float* __restrict__ part_o,float* __restrict__ part_ml,int kvh){
    const int chunk=blockIdx.x,g=blockIdx.y,warp=threadIdx.x>>5,lane=threadIdx.x&31;const int H=4*kvh,V=8*H;
    const long long qpos0=meta[0],kpos0=meta[1],begin=meta[2],count=meta[3];
    float qv[4][4];int qi[4],qh[4];
#pragma unroll
    for(int a=0;a<4;++a){const int v=warp*4+a;qi[a]=v>>2;qh[a]=g*4+(v&3);
        const __nv_bfloat16* src=q+((long long)qi[a]*H+qh[a])*128+lane*4;
#pragma unroll
        for(int d=0;d<4;++d)qv[a][d]=__bfloat162float(src[d])*0.08838834764831845f;}
    float m[4],l[4],acc[4][4];
#pragma unroll
    for(int a=0;a<4;++a){m[a]=-INFINITY;l[a]=0.f;
#pragma unroll
        for(int d=0;d<4;++d)acc[a][d]=0.f;}
    int keys;const __nv_bfloat16 *kb,*vb;long long first=0;
    if(chunk<DA_HIST){first=(long long)chunk*DA_KEYS;keys=(int)max(0LL,min((long long)DA_KEYS,count-first));
        kb=sk+((begin+first)*kvh+g)*128;vb=sv+((begin+first)*kvh+g)*128;}
    else{keys=8;kb=nk+(long long)g*128;vb=nv+(long long)g*128;}
    const int stride=kvh*128;
    for(int j0=0;j0<keys;j0+=8){
        uint2 kr[8],vr[8];
#pragma unroll
        for(int u=0;u<8;++u)if(j0+u<keys){kr[u]=*reinterpret_cast<const uint2*>(kb+(long long)(j0+u)*stride+lane*4);
            vr[u]=*reinterpret_cast<const uint2*>(vb+(long long)(j0+u)*stride+lane*4);}
#pragma unroll
        for(int u=0;u<8;++u){
            const int j=j0+u;if(j>=keys)break;
            const __nv_bfloat16* kh=reinterpret_cast<const __nv_bfloat16*>(&kr[u]);const __nv_bfloat16* vh=reinterpret_cast<const __nv_bfloat16*>(&vr[u]);
            float kf[4],vf[4];
#pragma unroll
            for(int d=0;d<4;++d){kf[d]=__bfloat162float(kh[d]);vf[d]=__bfloat162float(vh[d]);}
#pragma unroll
            for(int a=0;a<4;++a){
                float s=0.f;
#pragma unroll
                for(int d=0;d<4;++d)s=__fmaf_rn(qv[a][d],kf[d],s);
                for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
                bool vis=true;
                if(chunk<DA_HIST){const long long qp=qpos0+qi[a],kp=kpos0+first+j;vis=(qp-kp)<2048;}
                if(!vis)continue;
                const float mn=fmaxf(m[a],s);const float scale=expf(m[a]-mn);const float p=expf(s-mn);
                l[a]=l[a]*scale+p;
#pragma unroll
                for(int d=0;d<4;++d)acc[a][d]=acc[a][d]*scale+p*vf[d];
                m[a]=mn;
            }
        }
    }
#pragma unroll
    for(int a=0;a<4;++a){const long long vec=(long long)qi[a]*H+qh[a];
        float* o=part_o+((long long)chunk*V+vec)*128+lane*4;
#pragma unroll
        for(int d=0;d<4;++d)o[d]=acc[a][d];
        if(lane==0){part_ml[((long long)chunk*V+vec)*2]=m[a];part_ml[((long long)chunk*V+vec)*2+1]=l[a];}}
}
__global__ void draft_attn_combine(const float* __restrict__ part_o,const float* __restrict__ part_ml,__nv_bfloat16* __restrict__ out,int V){
    const int vec=blockIdx.x,d=threadIdx.x;float M=-INFINITY;
    for(int c=0;c<DA_CHUNKS;++c)M=fmaxf(M,part_ml[((long long)c*V+vec)*2]);
    float L=0.f,o=0.f;
    for(int c=0;c<DA_CHUNKS;++c){const float mc=part_ml[((long long)c*V+vec)*2];if(mc==-INFINITY)continue;
        const float w=expf(mc-M);L+=part_ml[((long long)c*V+vec)*2+1]*w;o+=part_o[((long long)c*V+vec)*128+d]*w;}
    out[(long long)vec*128+d]=__float2bfloat16(o/L);
}
extern "C" int glm53_draft_attn_dev_cuda(const void* q,const void* sk,const void* sv,const void* nk,const void* nv,const long long* meta,
    float* part_o,float* part_ml,void* out,int kvh,cudaStream_t s){
    if(kvh!=8&&kvh!=4)return int(cudaErrorInvalidValue);
    const char* pf=std::getenv("GLM53_DRAFT_ATTN_PREFETCH");
    if(pf&&pf[0]=='1')draft_attn_split_pf<<<dim3(DA_CHUNKS,kvh),256,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)sk,(const __nv_bfloat16*)sv,(const __nv_bfloat16*)nk,(const __nv_bfloat16*)nv,meta,part_o,part_ml,kvh);
    else draft_attn_split<<<dim3(DA_CHUNKS,kvh),256,0,s>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)sk,(const __nv_bfloat16*)sv,(const __nv_bfloat16*)nk,(const __nv_bfloat16*)nv,meta,part_o,part_ml,kvh);
    draft_attn_combine<<<32*kvh,128,0,s>>>(part_o,part_ml,(__nv_bfloat16*)out,32*kvh);
    return int(cudaGetLastError());
}

// ---- W08b drafter fusions (GLM53_DRAFT_FUSED_NORM=1), L2 ----
// add_norm: sum = float(x) + float(r); z = bf16(sum * rsqrt(mean(sum^2)+eps) * w); r_out = bf16(sum). One block per row.
__global__ void draft_add_norm(const __nv_bfloat16* __restrict__ x,const __nv_bfloat16* __restrict__ r,const float* __restrict__ w,
    __nv_bfloat16* __restrict__ z,__nv_bfloat16* __restrict__ rout,int d,int has_r){
    const int row=blockIdx.x;__shared__ float red[32];float s=0.f;
    for(int i=threadIdx.x;i<d;i+=blockDim.x){float v=__bfloat162float(x[(long long)row*d+i]);if(has_r)v+=__bfloat162float(r[(long long)row*d+i]);s+=v*v;}
    for(int o=16;o;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
    if((threadIdx.x&31)==0)red[threadIdx.x>>5]=s;__syncthreads();
    if(threadIdx.x<32){float t=threadIdx.x<(blockDim.x>>5)?red[threadIdx.x]:0.f;for(int o=16;o;o>>=1)t+=__shfl_xor_sync(0xffffffff,t,o);if(threadIdx.x==0)red[0]=rsqrtf(t/(float)d+1e-5f);}
    __syncthreads();const float rs=red[0];
    for(int i=threadIdx.x;i<d;i+=blockDim.x){float v=__bfloat162float(x[(long long)row*d+i]);if(has_r)v+=__bfloat162float(r[(long long)row*d+i]);
        z[(long long)row*d+i]=__float2bfloat16(v*rs*w[i]);if(rout)rout[(long long)row*d+i]=__float2bfloat16(v);}
}
// head norm + rope: y [n,H,128] bf16 -> bf16(norm) -> rope with bf16-rounded cos/sin (angle = pos*inv) -> bf16. One warp per (row,head).
__global__ void draft_head_norm_rope(const __nv_bfloat16* __restrict__ y,const float* __restrict__ w,const long long* __restrict__ pos,const float* __restrict__ inv,
    __nv_bfloat16* __restrict__ out,int n,int heads,long long pos_base_is_tensor){
    const int g=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(g>=n*heads)return;
    const int row=g/heads;const __nv_bfloat16* src=y+(long long)g*128;
    float v[4];float s=0.f;
#pragma unroll
    for(int j=0;j<4;++j){v[j]=__bfloat162float(src[lane+32*j]);s+=v[j]*v[j];}
    for(int o=16;o;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);
    const float rs=rsqrtf(s/128.f+1e-5f);
    float nb[4];
#pragma unroll
    for(int j=0;j<4;++j)nb[j]=__bfloat162float(__float2bfloat16(v[j]*rs*w[lane+32*j]));
    // dims d and d+64 pair: lane handles d=lane (j=0 with j=2) and d=lane+32 (j=1 with j=3)
    const float p=(float)pos[row];
#pragma unroll
    for(int j=0;j<2;++j){const int d=lane+32*j;const float ang=p*inv[d];
        const float c=__bfloat162float(__float2bfloat16(cosf(ang))),sn=__bfloat162float(__float2bfloat16(sinf(ang)));
        const float a=nb[j],b=nb[j+2];
        out[(long long)g*128+d]=__float2bfloat16(a*c-b*sn);out[(long long)g*128+d+64]=__float2bfloat16(b*c+a*sn);}
}
// silu(float(g)) * float(u) -> bf16
__global__ void draft_silu_mul(const __nv_bfloat16* __restrict__ g,const __nv_bfloat16* __restrict__ u,__nv_bfloat16* __restrict__ out,long long n){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const float a=__bfloat162float(g[i]);out[i]=__float2bfloat16((a/(1.f+expf(-a)))*__bfloat162float(u[i]));
}
extern "C" int glm53_draft_add_norm_cuda(const void* x,const void* r,const float* w,void* z,void* rout,int rows,int d,cudaStream_t s){
    draft_add_norm<<<rows,512,0,s>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)r,w,(__nv_bfloat16*)z,(__nv_bfloat16*)rout,d,r!=nullptr);return int(cudaGetLastError());}
extern "C" int glm53_draft_head_norm_rope_cuda(const void* y,const float* w,const long long* pos,const float* inv,void* out,int n,int heads,cudaStream_t s){
    draft_head_norm_rope<<<(n*heads+7)/8,256,0,s>>>((const __nv_bfloat16*)y,w,pos,inv,(__nv_bfloat16*)out,n,heads,0);return int(cudaGetLastError());}
extern "C" int glm53_draft_silu_mul_cuda(const void* g,const void* u,void* out,long long n,cudaStream_t s){
    draft_silu_mul<<<(unsigned)((n+255)/256),256,0,s>>>((const __nv_bfloat16*)g,(const __nv_bfloat16*)u,(__nv_bfloat16*)out,n);return int(cudaGetLastError());}
