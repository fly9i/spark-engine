#include <cstdio>
#include <cstdlib>
// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ float sum128(float v, float* shared) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int s=16;s;s>>=1) v += __shfl_down_sync(0xffffffff,v,s);
    if (lane==0) shared[warp]=v;
    __syncthreads();
    v=threadIdx.x<4 ? shared[threadIdx.x] : 0.f;
    if (warp==0) {
        for (int s=16;s;s>>=1) v += __shfl_down_sync(0xffffffff,v,s);
        if (lane==0) shared[0]=v;
    }
    __syncthreads();
    return shared[0];
}

// Each block owns one value row. Read/write H once; no atomics or branch sharing.
__global__ void recurrent128(const float* source,float* h,const float* q,const float* k,const float* v,
                             const float* beta,const float* decay,float* out) {
    int row=blockIdx.x, head=row/128, col=threadIdx.x;
    int i=head*128+col;
    float state=source[row*128+col]*decay[i];
    __shared__ float sums[4];
    float hk=sum128(state*k[i],sums);
    float correction=beta[head]*(v[row]-hk);
    state=state+correction*k[i];
    h[row*128+col]=state;
    __syncthreads();
    float result=sum128(state*q[i],sums);
    if (col==0) out[row]=result;
}

extern "C" int glm53_kda_cuda(float* h,const float* q,const float* k,const float* v,
                               const float* beta,const float* decay,float* out,
                               int heads,cudaStream_t stream) {
    recurrent128<<<heads*128,128,0,stream>>>(h,h,q,k,v,beta,decay,out);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int glm53_kda_fork_cuda(const float* source,float* h,const float* q,const float* k,const float* v,
                                    const float* beta,const float* decay,float* out,
                                    int heads,cudaStream_t stream) {
    recurrent128<<<heads*128,128,0,stream>>>(source,h,q,k,v,beta,decay,out);
    return static_cast<int>(cudaGetLastError());
}

// Keep each state element in a register across the causal sequence. The
// reduction order matches recurrent128; only intermediate global H traffic
// and per-token launches disappear. Inputs are contiguous [T,H,128].
__global__ void recurrent_sequence(float* h,const float* q,const float* k,const float* v,
                                   const float* beta,const float* decay,float* out,int heads,int steps) {
    int row=blockIdx.x, head=row/128, col=threadIdx.x;
    float state=h[row*128+col];
    __shared__ float sums[4];
    for(int t=0;t<steps;++t) {
        int i=(t*heads+head)*128+col;
        state=state*decay[i];
        float hk=sum128(state*k[i],sums);
        float correction=beta[t*heads+head]*(v[t*heads*128+row]-hk);
        state=state+correction*k[i];
        __syncthreads();
        float result=sum128(state*q[i],sums);
        if(col==0) out[t*heads*128+row]=result;
        __syncthreads();
    }
    h[row*128+col]=state;
}
extern "C" int glm53_kda_sequence_cuda(float* h,const float* q,const float* k,const float* v,
    const float* beta,const float* decay,float* out,int heads,int steps,cudaStream_t stream) {
    recurrent_sequence<<<heads*128,128,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);
    return static_cast<int>(cudaGetLastError());
}

// Warp owns one value row, four state columns per lane. Removes CTA-wide
// barriers at the cost of a different FP32 reduction order (qualified apart).
__global__ void recurrent_warp_sequence(float* h,const float* q,const float* k,const float* v,
    const float* beta,const float* decay,float* out,int heads,int steps) {
    int row=blockIdx.x*4+(threadIdx.x>>5),head=row/128,lane=threadIdx.x&31;
    float state[4];
    #pragma unroll
    for(int j=0;j<4;++j)state[j]=h[row*128+lane+j*32];
    for(int t=0;t<steps;++t){float key[4],query[4],dot=0.f;
        #pragma unroll
        for(int j=0;j<4;++j){int i=(t*heads+head)*128+lane+j*32;key[j]=k[i];query[j]=q[i];state[j]*=decay[i];dot+=state[j]*key[j];}
        for(int s=16;s;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
        float correction=beta[t*heads+head]*(v[t*heads*128+row]-__shfl_sync(0xffffffff,dot,0));
        dot=0.f;
        #pragma unroll
        for(int j=0;j<4;++j){state[j]+=correction*key[j];dot+=state[j]*query[j];}
        for(int s=16;s;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
        if(lane==0)out[t*heads*128+row]=dot;
    }
    #pragma unroll
    for(int j=0;j<4;++j)h[row*128+lane+j*32]=state[j];
}
// P3a (GLM53_KDA_SEQUENCE=3): the same per-row arithmetic as recurrent_warp_sequence
// (bitwise), two rows of one head per warp sharing k/q/decay loads (one wave on GB10),
// and the next token's inputs prefetched while the current token's chain runs.
__global__ void __launch_bounds__(128) recurrent_warp_sequence2(float* h,const float* __restrict__ q,const float* __restrict__ k,const float* __restrict__ v,
    const float* __restrict__ beta,const float* __restrict__ decay,float* __restrict__ out,int heads,int steps) {
    const int pair=blockIdx.x*4+(threadIdx.x>>5),row0=pair*2,head=row0/128,lane=threadIdx.x&31;
    float st0[4],st1[4];
#pragma unroll
    for(int j=0;j<4;++j){st0[j]=h[row0*128+lane+j*32];st1[j]=h[(row0+1)*128+lane+j*32];}
    float nk[4],nq[4],nd[4],nv0=0.f,nv1=0.f,nb=0.f;
    auto load=[&](int t){
#pragma unroll
        for(int j=0;j<4;++j){const int i=(t*heads+head)*128+lane+j*32;nk[j]=k[i];nq[j]=q[i];nd[j]=decay[i];}
        nv0=v[t*heads*128+row0];nv1=v[t*heads*128+row0+1];nb=beta[t*heads+head];};
    if(steps>0)load(0);
    for(int t=0;t<steps;++t){
        float key[4],query[4],dec[4];
#pragma unroll
        for(int j=0;j<4;++j){key[j]=nk[j];query[j]=nq[j];dec[j]=nd[j];}
        const float vv0=nv0,vv1=nv1,bb=nb;
        if(t+1<steps)load(t+1);
        float d0=0.f,d1=0.f;
#pragma unroll
        for(int j=0;j<4;++j){st0[j]*=dec[j];d0+=st0[j]*key[j];st1[j]*=dec[j];d1+=st1[j]*key[j];}
        for(int s=16;s;s>>=1){d0+=__shfl_down_sync(0xffffffff,d0,s);d1+=__shfl_down_sync(0xffffffff,d1,s);}
        const float c0=bb*(vv0-__shfl_sync(0xffffffff,d0,0)),c1=bb*(vv1-__shfl_sync(0xffffffff,d1,0));
        d0=0.f;d1=0.f;
#pragma unroll
        for(int j=0;j<4;++j){st0[j]+=c0*key[j];d0+=st0[j]*query[j];st1[j]+=c1*key[j];d1+=st1[j]*query[j];}
        for(int s=16;s;s>>=1){d0+=__shfl_down_sync(0xffffffff,d0,s);d1+=__shfl_down_sync(0xffffffff,d1,s);}
        if(lane==0){out[t*heads*128+row0]=d0;out[t*heads*128+row0+1]=d1;}
    }
#pragma unroll
    for(int j=0;j<4;++j){h[row0*128+lane+j*32]=st0[j];h[(row0+1)*128+lane+j*32]=st1[j];}
}

// P3b (GLM53_KDA_SEQUENCE=4): recurrent_warp_sequence arithmetic per row (bitwise), 16 warps x 2
// rows = 32 rows of one head per block; k/q/decay/v/beta for 8-token tiles are staged in shared
// memory with cp.async double buffering so the serial per-token chain never waits on DRAM.
constexpr int KSEQ_T=8;
__device__ __forceinline__ void kseq_cp16(void* dst,const void* src){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src));}
__global__ void __launch_bounds__(512) recurrent_warp_sequence3(float* h,const float* __restrict__ q,const float* __restrict__ k,const float* __restrict__ v,
    const float* __restrict__ beta,const float* __restrict__ decay,float* __restrict__ out,int heads,int steps) {
  __shared__ __align__(16) float sk[2][KSEQ_T][128],sq[2][KSEQ_T][128],sd[2][KSEQ_T][128],sv[2][KSEQ_T][32],sb[2][KSEQ_T];
  const int head=blockIdx.x/4,rbase=(blockIdx.x%4)*32,warp=threadIdx.x>>5,lane=threadIdx.x&31,tid=threadIdx.x;
  const int row0=head*128+rbase+warp*2;           // global state row (head*128 + r)
  float st0[4],st1[4];
#pragma unroll
  for(int j=0;j<4;++j){st0[j]=h[row0*128+lane+j*32];st1[j]=h[(row0+1)*128+lane+j*32];}
  const int tiles=(steps+KSEQ_T-1)/KSEQ_T;
  auto stage=[&](int tile,int buf){
    const int t0=tile*KSEQ_T;
    // 3 x 8 x 128 floats = 768 x 16B for k/q/decay, 8 x 32 floats = 64 x 16B for v
    for(int i=tid;i<3*KSEQ_T*32+KSEQ_T*8;i+=512){
      if(i<3*KSEQ_T*32){const int which=i/(KSEQ_T*32),rem=i%(KSEQ_T*32),tt=rem/32,c=(rem%32)*4;const int t=min(t0+tt,steps-1);
        const float* src=(which==0?k:which==1?q:decay)+((long long)t*heads+head)*128+c;
        float* dst=(which==0?sk[buf][tt]:which==1?sq[buf][tt]:sd[buf][tt])+c;kseq_cp16(dst,src);}
      else{const int r=i-3*KSEQ_T*32,tt=r/8,c=(r%8)*4;const int t=min(t0+tt,steps-1);
        kseq_cp16(&sv[buf][tt][c],v+(long long)t*heads*128+head*128+rbase+c);}
    }
    if(tid<KSEQ_T){const int t=min(t0+tid,steps-1);sb[buf][tid]=beta[t*heads+head];}
    asm volatile("cp.async.commit_group;\n"::);
  };
  if(tiles>0)stage(0,0);
  for(int tile=0;tile<tiles;++tile){
    const int buf=tile&1;
    if(tile+1<tiles){stage(tile+1,buf^1);asm volatile("cp.async.wait_group 1;\n"::);}
    else asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    const int t0=tile*KSEQ_T,n=min(KSEQ_T,steps-t0);
    for(int tt=0;tt<n;++tt){const int t=t0+tt;
      float key[4],query[4],dec[4];
#pragma unroll
      for(int j=0;j<4;++j){key[j]=sk[buf][tt][lane+j*32];query[j]=sq[buf][tt][lane+j*32];dec[j]=sd[buf][tt][lane+j*32];}
      const float vv0=sv[buf][tt][warp*2],vv1=sv[buf][tt][warp*2+1],bb=sb[buf][tt];
      float d0=0.f,d1=0.f;
#pragma unroll
      for(int j=0;j<4;++j){st0[j]*=dec[j];d0+=st0[j]*key[j];st1[j]*=dec[j];d1+=st1[j]*key[j];}
      for(int s=16;s;s>>=1){d0+=__shfl_down_sync(0xffffffff,d0,s);d1+=__shfl_down_sync(0xffffffff,d1,s);}
      const float c0=bb*(vv0-__shfl_sync(0xffffffff,d0,0)),c1=bb*(vv1-__shfl_sync(0xffffffff,d1,0));
      d0=0.f;d1=0.f;
#pragma unroll
      for(int j=0;j<4;++j){st0[j]+=c0*key[j];d0+=st0[j]*query[j];st1[j]+=c1*key[j];d1+=st1[j]*query[j];}
      for(int s=16;s;s>>=1){d0+=__shfl_down_sync(0xffffffff,d0,s);d1+=__shfl_down_sync(0xffffffff,d1,s);}
      if(lane==0){out[(long long)t*heads*128+row0]=d0;out[(long long)t*heads*128+row0+1]=d1;}
    }
    __syncthreads();
  }
#pragma unroll
  for(int j=0;j<4;++j){h[row0*128+lane+j*32]=st0[j];h[(row0+1)*128+lane+j*32]=st1[j];}
}
// Mode 5 (GLM53_KDA_SEQUENCE=5, prefill): same per-row arithmetic as recurrent_warp_sequence3 (per step: decay,
// dot with k over 4x32 lanes, 5-level shuffle tree, delta update, dot with q) but R rows per warp and BPH blocks per
// head, so each block stages a head's k/q/decay tile for 128/BPH rows instead of 32 (sequence3 re-reads the same
// 1.5 KB per head-step 4x). Rows are independent; every row's operation order and reduction tree are unchanged.
template<int R,int BPH>
__global__ void __launch_bounds__(128/(R*BPH)*32) recurrent_warp_sequence5(float* h,const float* __restrict__ q,const float* __restrict__ k,const float* __restrict__ v,
    const float* __restrict__ beta,const float* __restrict__ decay,float* __restrict__ out,int heads,int steps) {
  constexpr int ROWS=128/BPH, WARPS=ROWS/R, THREADS=WARPS*32;
  __shared__ __align__(16) float sk[2][KSEQ_T][128],sq[2][KSEQ_T][128],sd[2][KSEQ_T][128],sv[2][KSEQ_T][ROWS],sb[2][KSEQ_T];
  const int head=blockIdx.x/BPH,rbase=(blockIdx.x%BPH)*ROWS,warp=threadIdx.x>>5,lane=threadIdx.x&31,tid=threadIdx.x;
  const int row0=head*128+rbase+warp*R;
  float st[R][4];
#pragma unroll
  for(int r=0;r<R;++r)
#pragma unroll
    for(int j=0;j<4;++j)st[r][j]=h[(row0+r)*128+lane+j*32];
  const int tiles=(steps+KSEQ_T-1)/KSEQ_T;
  auto stage=[&](int tile,int buf){
    const int t0=tile*KSEQ_T;
    for(int i=tid;i<3*KSEQ_T*32+KSEQ_T*(ROWS/4);i+=THREADS){
      if(i<3*KSEQ_T*32){const int which=i/(KSEQ_T*32),rem=i%(KSEQ_T*32),tt=rem/32,c=(rem%32)*4;const int t=min(t0+tt,steps-1);
        const float* src=(which==0?k:which==1?q:decay)+((long long)t*heads+head)*128+c;
        float* dst=(which==0?sk[buf][tt]:which==1?sq[buf][tt]:sd[buf][tt])+c;kseq_cp16(dst,src);}
      else{const int r=i-3*KSEQ_T*32,tt=r/(ROWS/4),c=(r%(ROWS/4))*4;const int t=min(t0+tt,steps-1);
        kseq_cp16(&sv[buf][tt][c],v+(long long)t*heads*128+head*128+rbase+c);}
    }
    if(tid<KSEQ_T){const int t=min(t0+tid,steps-1);sb[buf][tid]=beta[t*heads+head];}
    asm volatile("cp.async.commit_group;\n"::);
  };
  if(tiles>0)stage(0,0);
  for(int tile=0;tile<tiles;++tile){
    const int buf=tile&1;
    if(tile+1<tiles){stage(tile+1,buf^1);asm volatile("cp.async.wait_group 1;\n"::);}
    else asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    const int t0=tile*KSEQ_T,n=min(KSEQ_T,steps-t0);
    for(int tt=0;tt<n;++tt){const int t=t0+tt;
      float key[4],query[4],dec[4];
#pragma unroll
      for(int j=0;j<4;++j){key[j]=sk[buf][tt][lane+j*32];query[j]=sq[buf][tt][lane+j*32];dec[j]=sd[buf][tt][lane+j*32];}
      const float bb=sb[buf][tt];
      float d[R];
#pragma unroll
      for(int r=0;r<R;++r){d[r]=0.f;
#pragma unroll
        for(int j=0;j<4;++j){st[r][j]*=dec[j];d[r]+=st[r][j]*key[j];}}
#pragma unroll
      for(int s=16;s;s>>=1)
#pragma unroll
        for(int r=0;r<R;++r)d[r]+=__shfl_down_sync(0xffffffff,d[r],s);
      float c[R];
#pragma unroll
      for(int r=0;r<R;++r){c[r]=bb*(sv[buf][tt][warp*R+r]-__shfl_sync(0xffffffff,d[r],0));d[r]=0.f;
#pragma unroll
        for(int j=0;j<4;++j){st[r][j]+=c[r]*key[j];d[r]+=st[r][j]*query[j];}}
#pragma unroll
      for(int s=16;s;s>>=1)
#pragma unroll
        for(int r=0;r<R;++r)d[r]+=__shfl_down_sync(0xffffffff,d[r],s);
      if(lane==0){
#pragma unroll
        for(int r=0;r<R;++r)out[(long long)t*heads*128+row0+r]=d[r];}
    }
    __syncthreads();
  }
#pragma unroll
  for(int r=0;r<R;++r)
#pragma unroll
    for(int j=0;j<4;++j)h[(row0+r)*128+lane+j*32]=st[r][j];
}
// Mode 6 (GLM53_KDA_SEQUENCE=6, prefill, L1): each state row is held by L lanes (128/L contiguous columns per
// lane), so one warp carries 32/L rows and every warp-wide shuffle serves all of them (sequence3 spends 11 shuffles
// and 12 scalar LDS per 2 rows per step and is MIO-bound: ncu 18 cycles/issue, 33% MIO stalls). Per-lane partial
// sums run over contiguous columns, then an xor tree over L lanes: the summation order differs from mode 4 (L1).
template<int L,int BPH>
__global__ void __launch_bounds__(128/BPH/(32/L)*32) recurrent_lane_sequence6(float* h,const float* __restrict__ q,const float* __restrict__ k,const float* __restrict__ v,
    const float* __restrict__ beta,const float* __restrict__ decay,float* __restrict__ out,int heads,int steps) {
  constexpr int RPW=32/L, ROWS=128/BPH, WARPS=ROWS/RPW, THREADS=WARPS*32, CH=128/L;
  __shared__ __align__(16) float sk[2][KSEQ_T][128],sq[2][KSEQ_T][128],sd[2][KSEQ_T][128],sv[2][KSEQ_T][ROWS],sb[2][KSEQ_T];
  const int head=blockIdx.x/BPH,rbase=(blockIdx.x%BPH)*ROWS,warp=threadIdx.x>>5,lane=threadIdx.x&31,tid=threadIdx.x;
  const int grp=lane/L,lig=lane%L,lrow=warp*RPW+grp,row=head*128+rbase+lrow;
  // lane lig holds float4 chunks lig, lig+L, lig+2L, ...: a warp-wide float4 load touches adjacent 16 B chunks (no bank conflict)
  auto col=[&](int e){return ((e>>2)*L+lig)*4;};
  float st[CH];
#pragma unroll
  for(int e=0;e<CH;e+=4){float4 x=*reinterpret_cast<const float4*>(h+(long long)row*128+col(e));st[e]=x.x;st[e+1]=x.y;st[e+2]=x.z;st[e+3]=x.w;}
  const int tiles=(steps+KSEQ_T-1)/KSEQ_T;
  auto stage=[&](int tile,int buf){
    const int t0=tile*KSEQ_T;
    for(int i=tid;i<3*KSEQ_T*32+KSEQ_T*(ROWS/4);i+=THREADS){
      if(i<3*KSEQ_T*32){const int which=i/(KSEQ_T*32),rem=i%(KSEQ_T*32),tt=rem/32,c=(rem%32)*4;const int t=min(t0+tt,steps-1);
        const float* src=(which==0?k:which==1?q:decay)+((long long)t*heads+head)*128+c;
        float* dst=(which==0?sk[buf][tt]:which==1?sq[buf][tt]:sd[buf][tt])+c;kseq_cp16(dst,src);}
      else{const int r=i-3*KSEQ_T*32,tt=r/(ROWS/4),c=(r%(ROWS/4))*4;const int t=min(t0+tt,steps-1);
        kseq_cp16(&sv[buf][tt][c],v+(long long)t*heads*128+head*128+rbase+c);}
    }
    if(tid<KSEQ_T){const int t=min(t0+tid,steps-1);sb[buf][tid]=beta[t*heads+head];}
    asm volatile("cp.async.commit_group;\n"::);
  };
  if(tiles>0)stage(0,0);
  for(int tile=0;tile<tiles;++tile){
    const int buf=tile&1;
    if(tile+1<tiles){stage(tile+1,buf^1);asm volatile("cp.async.wait_group 1;\n"::);}
    else asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    const int t0=tile*KSEQ_T,n=min(KSEQ_T,steps-t0);
    for(int tt=0;tt<n;++tt){const int t=t0+tt;
      const float* kk=sk[buf][tt];const float* qq=sq[buf][tt];const float* dd=sd[buf][tt];
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f;   // four partial sums: dependent-add chain CH/4 long
#pragma unroll
      for(int e=0;e<CH;e+=4){float4 dv=*reinterpret_cast<const float4*>(dd+col(e)),kv=*reinterpret_cast<const float4*>(kk+col(e));
        st[e]*=dv.x;st[e+1]*=dv.y;st[e+2]*=dv.z;st[e+3]*=dv.w;
        a0+=st[e]*kv.x;a1+=st[e+1]*kv.y;a2+=st[e+2]*kv.z;a3+=st[e+3]*kv.w;}
      float d=(a0+a1)+(a2+a3);
#pragma unroll
      for(int s=L/2;s;s>>=1)d+=__shfl_xor_sync(0xffffffff,d,s);
      const float c=sb[buf][tt]*(sv[buf][tt][lrow]-d);
      float b0=0.f,b1=0.f,b2=0.f,b3=0.f;
#pragma unroll
      for(int e=0;e<CH;e+=4){float4 kv=*reinterpret_cast<const float4*>(kk+col(e)),qv=*reinterpret_cast<const float4*>(qq+col(e));
        st[e]+=c*kv.x;st[e+1]+=c*kv.y;st[e+2]+=c*kv.z;st[e+3]+=c*kv.w;
        b0+=st[e]*qv.x;b1+=st[e+1]*qv.y;b2+=st[e+2]*qv.z;b3+=st[e+3]*qv.w;}
      float o=(b0+b1)+(b2+b3);
#pragma unroll
      for(int s=L/2;s;s>>=1)o+=__shfl_xor_sync(0xffffffff,o,s);
      if(lig==0)out[(long long)t*heads*128+row]=o;
    }
    __syncthreads();
  }
#pragma unroll
  for(int e=0;e<CH;e+=4)*reinterpret_cast<float4*>(h+(long long)row*128+col(e))=make_float4(st[e],st[e+1],st[e+2],st[e+3]);
}
extern "C" int glm53_kda_warp_sequence_cuda(float* h,const float* q,const float* k,const float* v,
    const float* beta,const float* decay,float* out,int heads,int steps,cudaStream_t stream){
    const char* mode=std::getenv("GLM53_KDA_SEQUENCE");
    if(mode&&mode[0]=='6'){
      const char* var=std::getenv("GLM53_KDA_SEQ6");const int vv=var?atoi(var):44;   // L*10+BPH
      switch(vv){
        case 41: recurrent_lane_sequence6<4,1><<<heads,512,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 42: recurrent_lane_sequence6<4,2><<<heads*2,256,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 82: recurrent_lane_sequence6<8,2><<<heads*2,512,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 84: recurrent_lane_sequence6<8,4><<<heads*4,256,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 48: recurrent_lane_sequence6<4,8><<<heads*8,64,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 24: recurrent_lane_sequence6<2,4><<<heads*4,64,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 28: recurrent_lane_sequence6<2,8><<<heads*8,32,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 88: recurrent_lane_sequence6<8,8><<<heads*8,128,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 164: recurrent_lane_sequence6<16,4><<<heads*4,512,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        default: recurrent_lane_sequence6<4,4><<<heads*4,128,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
      }
      return static_cast<int>(cudaGetLastError());}
    if(mode&&mode[0]=='5'){
      const char* var=std::getenv("GLM53_KDA_SEQ5");const int vv=var?atoi(var):41;   // R*10+BPH
      switch(vv){
        case 42: recurrent_warp_sequence5<4,2><<<heads*2,512,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 81: recurrent_warp_sequence5<8,1><<<heads,512,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 22: recurrent_warp_sequence5<2,2><<<heads*2,1024,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        case 24: recurrent_warp_sequence5<2,4><<<heads*4,512,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
        default: recurrent_warp_sequence5<4,1><<<heads,1024,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);break;
      }
      return static_cast<int>(cudaGetLastError());}
    if(mode&&mode[0]=='4'){recurrent_warp_sequence3<<<heads*4,512,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);return static_cast<int>(cudaGetLastError());}
    if(mode&&mode[0]=='3'){recurrent_warp_sequence2<<<heads*16,128,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);return static_cast<int>(cudaGetLastError());}
    recurrent_warp_sequence<<<heads*32,128,0,stream>>>(h,q,k,v,beta,decay,out,heads,steps);
    return static_cast<int>(cudaGetLastError());
}

// ---- W05 KDA gate fusion (GLM53_KDA_GATE_FUSED=1), rows <= 8 ----
// K1: one warp per output row of [fa;ga] (Half weights, Half-rounded X) and wb (FP32 weights and X).
//     fa/ga outputs are rounded to Half (mm16 boundary); wb outputs pass through sigmoid (beta).
__device__ __forceinline__ float kda_sig(float x){return 1.f/(1.f+expf(-x));}
template<int Rows>
__global__ void kda_gate_first(const float* __restrict__ x,const __half* __restrict__ fa,const __half* __restrict__ ga,
    const float* __restrict__ wb,float* __restrict__ part,int rows,int k,int rank_a,int nb){
    // warp = output*4 + quarter; each quarter covers k/4 columns. part[(quarter*rows+r)*total+output].
    const int gw=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
    const int total=2*rank_a+nb;if(gw>=total*4)return;
    const int out=gw>>2,quarter=gw&3,span=k/4,base=quarter*span;
    float acc[Rows]={};
    if(out<2*rank_a){
        const __half* w=(out<rank_a?fa+(long long)out*k:ga+(long long)(out-rank_a)*k)+base;
        for(int c=lane*8;c<span;c+=256){
            const uint4 raw=*reinterpret_cast<const uint4*>(w+c);const __half* hv=reinterpret_cast<const __half*>(&raw);
            float wf[8];
#pragma unroll
            for(int j=0;j<8;++j)wf[j]=__half2float(hv[j]);
#pragma unroll
            for(int r=0;r<Rows;++r){if(r<rows){const float* xr=x+(long long)r*k+base+c;const float4 a=*reinterpret_cast<const float4*>(xr),b=*reinterpret_cast<const float4*>(xr+4);
                const float xs[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
                for(int j=0;j<8;++j)acc[r]=__fmaf_rn(__half2float(__float2half_rn(xs[j])),wf[j],acc[r]);}}
        }
    }else{
        const float* w=wb+(long long)(out-2*rank_a)*k+base;
        for(int c=lane*4;c<span;c+=128){
            const float4 wv=*reinterpret_cast<const float4*>(w+c);
#pragma unroll
            for(int r=0;r<Rows;++r){if(r<rows){const float4 a=*reinterpret_cast<const float4*>(x+(long long)r*k+base+c);
                acc[r]=__fmaf_rn(a.x,wv.x,acc[r]);acc[r]=__fmaf_rn(a.y,wv.y,acc[r]);acc[r]=__fmaf_rn(a.z,wv.z,acc[r]);acc[r]=__fmaf_rn(a.w,wv.w,acc[r]);}}
        }
    }
#pragma unroll
    for(int r=0;r<Rows;++r){float s=acc[r];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);acc[r]=s;}
    if(lane==0){for(int r=0;r<rows;++r)part[((long long)quarter*rows+r)*total+out]=acc[r];}
}
// K1b: combine quarters in fixed order; Half-round fa/ga outputs, sigmoid for wb outputs.
__global__ void kda_gate_combine(const float* __restrict__ part,float* __restrict__ mid,float* __restrict__ beta,int rows,int rank_a,int nb){
    const int total=2*rank_a+nb;const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*total)return;
    const int r=i/total,o=i%total;const long long st=(long long)rows*total;
    const float s=((part[i]+part[st+i])+part[2*st+i])+part[3*st+i];
    if(o<2*rank_a)mid[(long long)r*2*rank_a+o]=__half2float(__float2half_rn(s));else beta[(long long)r*nb+o-2*rank_a]=kda_sig(s);
}
// K2: one warp per output row n of fb and gb (Half, K=rank_a), Half-rounded result, then
//     decay = exp(-5/(exp(-(A*(g1+dt_bias)))+1)) and sg2 = sigmoid(g2).
template<int Rows>
__global__ void kda_gate_second(const float* __restrict__ mid,const __half* __restrict__ fb,const __half* __restrict__ gb,
    const float* __restrict__ a_log,const float* __restrict__ dt_bias,float* __restrict__ decay,float* __restrict__ sg2,
    int rows,int n,int rank_a){
    // Each half-warp owns one output row; lane j of 16 loads 8 halves (16 B) at columns 8j..8j+7 (rank_a==128).
    const int hw=blockIdx.x*16+(threadIdx.x>>4),sub=threadIdx.x&15;if(hw>=2*n)return;
    const bool second=hw>=n;const int row=second?hw-n:hw;
    const __half* w=(second?gb:fb)+(long long)row*rank_a;
    float acc[Rows]={};
    for(int c=sub*8;c<rank_a;c+=128){
        const uint4 raw=*reinterpret_cast<const uint4*>(w+c);const __half* hv=reinterpret_cast<const __half*>(&raw);
#pragma unroll
        for(int j=0;j<8;++j){const float wf=__half2float(hv[j]);
#pragma unroll
            for(int r=0;r<Rows;++r)if(r<rows)acc[r]=__fmaf_rn(mid[(long long)r*2*rank_a+(second?rank_a:0)+c+j],wf,acc[r]);}
    }
#pragma unroll
    for(int r=0;r<Rows;++r){float s=acc[r];for(int off=8;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off,16);acc[r]=s;}
    if(sub==0){
        const int head=row/128;const float a=expf(a_log[head]);const float bias=dt_bias[row];
        for(int r=0;r<rows;++r){
            const float g=__half2float(__float2half_rn(acc[r]));
            if(second)sg2[(long long)r*n+row]=kda_sig(g);
            else{const float t=a*(g+bias);const float e=expf(-t)+1.f;decay[(long long)r*n+row]=expf(-5.f/e);}
        }
    }
}
// K3: hidden = o_norm * (o * rsqrt(mean(o^2)+1e-5)) * sg2, one warp per (row, head), 128 per head.
__global__ void kda_onorm_gate(const float* __restrict__ o,const float* __restrict__ onorm,int onorm_len,const float* __restrict__ sg2,float* __restrict__ out,int groups){
    const int g=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(g>=groups)return;
    const float* src=o+(long long)g*128;float v[4];float s=0.f;
#pragma unroll
    for(int j=0;j<4;++j){v[j]=src[lane+32*j];s+=v[j]*v[j];}
    for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
    const float r=rsqrtf(s/128.f+1e-5f);
#pragma unroll
    for(int j=0;j<4;++j){const long long i=(long long)g*128+lane+32*j;const int d=(int)(i%onorm_len);out[i]=(onorm[d]*(v[j]*r))*sg2[i];}
}
// P10 (GLM53_KDA_ONORM_SIGMOID=1): kda_onorm_gate reading the raw g2 and applying ATen's float sigmoid in place
// (1/(1+expf(-x)), IEEE division; this file builds without fast math), so sigmoid(g2) is never written to memory.
__global__ void kda_onorm_gate_sig(const float* __restrict__ o,const float* __restrict__ onorm,int onorm_len,const float* __restrict__ g2,float* __restrict__ out,int groups){
    const int g=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(g>=groups)return;
    const float* src=o+(long long)g*128;float v[4];float s=0.f;
#pragma unroll
    for(int j=0;j<4;++j){v[j]=src[lane+32*j];s+=v[j]*v[j];}
    for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
    const float r=rsqrtf(s/128.f+1e-5f);
#pragma unroll
    for(int j=0;j<4;++j){const long long i=(long long)g*128+lane+32*j;const int d=(int)(i%onorm_len);
        const float sg=1.0f/(1.0f+expf(-g2[i]));out[i]=(onorm[d]*(v[j]*r))*sg;}
}
// GLM53_KDA_GATE_SMEM=1: the same two GEMV stages with the shared operand staged once per block (L0: every output keeps
// its warp's lane-to-column mapping, per-lane accumulation order and shuffle tree; only where X / mid are read from
// changes). K1s: block = (16 outputs, one quarter of K); X[rows][quarter] staged once, Half-rounded for fa/ga blocks
// and raw FP32 for wb blocks (as K1 reads them). K2s: block = 64 output rows of fb or gb; mid staged once.
template<int Rows>
__global__ void __launch_bounds__(256) kda_gate_first_smem(const float* __restrict__ x,const __half* __restrict__ fa,const __half* __restrict__ ga,
    const float* __restrict__ wb,float* __restrict__ part,int rows,int k,int rank_a,int nb){
    extern __shared__ float xs[];                       // [Rows][k/4]
    const int total=2*rank_a+nb,span=k/4;
    const int groups_ab=(2*rank_a)/16,quarter=blockIdx.x&3,grp=blockIdx.x>>2;
    const bool half_blk=grp<groups_ab;const int out0=half_blk?grp*16:2*rank_a+(grp-groups_ab)*16;
    const int base=quarter*span;
    for(int i=threadIdx.x;i<rows*span;i+=256){const int r=i/span,c=i%span;const float v=x[(long long)r*k+base+c];
        xs[r*span+c]=half_blk?__half2float(__float2half_rn(v)):v;}
    __syncthreads();
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int o=warp;o<16;o+=8){
        const int out=out0+o;if(out>=total||(half_blk&&out>=2*rank_a))break;
        float acc[Rows]={};
        if(half_blk){
            const __half* w=(out<rank_a?fa+(long long)out*k:ga+(long long)(out-rank_a)*k)+base;
            for(int c=lane*8;c<span;c+=256){
                const uint4 raw=*reinterpret_cast<const uint4*>(w+c);const __half* hv=reinterpret_cast<const __half*>(&raw);
                float wf[8];
#pragma unroll
                for(int j=0;j<8;++j)wf[j]=__half2float(hv[j]);
#pragma unroll
                for(int r=0;r<Rows;++r){if(r<rows){const float* xr=xs+r*span+c;
#pragma unroll
                    for(int j=0;j<8;++j)acc[r]=__fmaf_rn(xr[j],wf[j],acc[r]);}}
            }
        }else{
            const float* w=wb+(long long)(out-2*rank_a)*k+base;
            for(int c=lane*4;c<span;c+=128){
                const float4 wv=*reinterpret_cast<const float4*>(w+c);
#pragma unroll
                for(int r=0;r<Rows;++r){if(r<rows){const float* a=xs+r*span+c;
                    acc[r]=__fmaf_rn(a[0],wv.x,acc[r]);acc[r]=__fmaf_rn(a[1],wv.y,acc[r]);acc[r]=__fmaf_rn(a[2],wv.z,acc[r]);acc[r]=__fmaf_rn(a[3],wv.w,acc[r]);}}
            }
        }
#pragma unroll
        for(int r=0;r<Rows;++r){float s=acc[r];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);acc[r]=s;}
        if(lane==0){for(int r=0;r<rows;++r)part[((long long)quarter*rows+r)*total+out]=acc[r];}
    }
}
template<int Rows>
__global__ void __launch_bounds__(256) kda_gate_second_smem(const float* __restrict__ mid,const __half* __restrict__ fb,const __half* __restrict__ gb,
    const float* __restrict__ a_log,const float* __restrict__ dt_bias,float* __restrict__ decay,float* __restrict__ sg2,
    int rows,int n,int rank_a){
    extern __shared__ float ms[];                       // [rows][2*rank_a]
    for(int i=threadIdx.x;i<rows*2*rank_a;i+=256)ms[i]=mid[i];
    __syncthreads();
    const int sub=threadIdx.x&15,hwl=threadIdx.x>>4;
    for(int q=0;q<4;++q){
        const int hw=blockIdx.x*64+q*16+hwl;if(hw>=2*n)return;
        const bool second=hw>=n;const int row=second?hw-n:hw;
        const __half* w=(second?gb:fb)+(long long)row*rank_a;
        float acc[Rows]={};
        for(int c=sub*8;c<rank_a;c+=128){
            const uint4 raw=*reinterpret_cast<const uint4*>(w+c);const __half* hv=reinterpret_cast<const __half*>(&raw);
#pragma unroll
            for(int j=0;j<8;++j){const float wf=__half2float(hv[j]);
#pragma unroll
                for(int r=0;r<Rows;++r)if(r<rows)acc[r]=__fmaf_rn(ms[r*2*rank_a+(second?rank_a:0)+c+j],wf,acc[r]);}
        }
#pragma unroll
        for(int r=0;r<Rows;++r){float s=acc[r];for(int off=8;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off,16);acc[r]=s;}
        if(sub==0){
            const int head=row/128;const float a=expf(a_log[head]);const float bias=dt_bias[row];
            for(int r=0;r<rows;++r){
                const float g=__half2float(__float2half_rn(acc[r]));
                if(second)sg2[(long long)r*n+row]=kda_sig(g);
                else{const float t=a*(g+bias);const float e=expf(-t)+1.f;decay[(long long)r*n+row]=expf(-5.f/e);}
            }
        }
    }
}
// GLM53_KDA_GATE_TILE=1 (wide 9..32-row calls; L0 against kda_gate_first<Rows> / kda_gate_second<Rows>): K1t block =
// 8 outputs of one kind (fa/ga or wb) and one K quarter; X is staged 256 columns at a time (Half-rounded for fa/ga
// blocks, raw FP32 for wb) so 8 warps share one L2 read of X instead of one each. Every lane keeps its column
// mapping (fa/ga: lane*8 + 256*i, wb: lane*4 + 128*i) and per-row accumulation order. K2t: mid staged once per block
// of 64 output rows, read as float4 pairs in the same j order.
template<int Rows>
__global__ void __launch_bounds__(256) kda_gate_first_tile(const float* __restrict__ x,const __half* __restrict__ fa,const __half* __restrict__ ga,
    const float* __restrict__ wb,float* __restrict__ part,int rows,int k,int rank_a,int nb){
    __shared__ __align__(16) float xs[Rows*256];
    const int total=2*rank_a+nb,span=k/4;
    const int groups_ab=(2*rank_a)/8,quarter=blockIdx.x&3,grp=blockIdx.x>>2;
    const bool half_blk=grp<groups_ab;
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    const int out=half_blk?grp*8+warp:2*rank_a+(grp-groups_ab)*8+warp;
    const bool live=out<total;
    const int base=quarter*span;
    float acc[Rows]={};
    for(int c0=0;c0<span;c0+=256){
        __syncthreads();
        for(int i=threadIdx.x;i<rows*64;i+=256){const int r=i>>6,c=(i&63)*4;
            float4 v=*reinterpret_cast<const float4*>(x+(long long)r*k+base+c0+c);
            if(half_blk){v.x=__half2float(__float2half_rn(v.x));v.y=__half2float(__float2half_rn(v.y));
                v.z=__half2float(__float2half_rn(v.z));v.w=__half2float(__float2half_rn(v.w));}
            *reinterpret_cast<float4*>(xs+r*256+c)=v;}
        __syncthreads();
        if(!live)continue;
        if(half_blk){
            const __half* w=(out<rank_a?fa+(long long)out*k:ga+(long long)(out-rank_a)*k)+base+c0;
            const uint4 raw=*reinterpret_cast<const uint4*>(w+lane*8);const __half* hv=reinterpret_cast<const __half*>(&raw);
            float wf[8];
#pragma unroll
            for(int j=0;j<8;++j)wf[j]=__half2float(hv[j]);
#pragma unroll
            for(int r=0;r<Rows;++r){if(r<rows){const float4 a=*reinterpret_cast<const float4*>(xs+r*256+lane*8),b=*reinterpret_cast<const float4*>(xs+r*256+lane*8+4);
                const float xv[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
                for(int j=0;j<8;++j)acc[r]=__fmaf_rn(xv[j],wf[j],acc[r]);}}
        }else{
            const float* w=wb+(long long)(out-2*rank_a)*k+base+c0;
#pragma unroll
            for(int h=0;h<2;++h){
                const float4 wv=*reinterpret_cast<const float4*>(w+h*128+lane*4);
#pragma unroll
                for(int r=0;r<Rows;++r){if(r<rows){const float4 a=*reinterpret_cast<const float4*>(xs+r*256+h*128+lane*4);
                    acc[r]=__fmaf_rn(a.x,wv.x,acc[r]);acc[r]=__fmaf_rn(a.y,wv.y,acc[r]);acc[r]=__fmaf_rn(a.z,wv.z,acc[r]);acc[r]=__fmaf_rn(a.w,wv.w,acc[r]);}}
            }
        }
    }
    if(!live)return;
#pragma unroll
    for(int r=0;r<Rows;++r){float s=acc[r];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);acc[r]=s;}
    if(lane==0){for(int r=0;r<rows;++r)part[((long long)quarter*rows+r)*total+out]=acc[r];}
}
template<int Rows>
__global__ void __launch_bounds__(256) kda_gate_second_tile(const float* __restrict__ mid,const __half* __restrict__ fb,const __half* __restrict__ gb,
    const float* __restrict__ a_log,const float* __restrict__ dt_bias,float* __restrict__ decay,float* __restrict__ sg2,int rows,int n){
    __shared__ __align__(16) float ms[Rows*256];        // [rows][2*rank_a], rank_a == 128
    for(int i=threadIdx.x;i<rows*64;i+=256)*reinterpret_cast<float4*>(ms+i*4)=*reinterpret_cast<const float4*>(mid+i*4);
    __syncthreads();
    const int sub=threadIdx.x&15,hwl=threadIdx.x>>4;
    for(int q=0;q<4;++q){
        const int hw=blockIdx.x*64+q*16+hwl;if(hw>=2*n)return;
        const bool second=hw>=n;const int row=second?hw-n:hw;
        const __half* w=(second?gb:fb)+(long long)row*128;
        float acc[Rows]={};
        const uint4 raw=*reinterpret_cast<const uint4*>(w+sub*8);const __half* hv=reinterpret_cast<const __half*>(&raw);
        float wf[8];
#pragma unroll
        for(int j=0;j<8;++j)wf[j]=__half2float(hv[j]);
#pragma unroll
        for(int r=0;r<Rows;++r)if(r<rows){const float* m=ms+r*256+(second?128:0)+sub*8;
            const float4 a=*reinterpret_cast<const float4*>(m),b=*reinterpret_cast<const float4*>(m+4);const float mv[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
            for(int j=0;j<8;++j)acc[r]=__fmaf_rn(mv[j],wf[j],acc[r]);}
#pragma unroll
        for(int r=0;r<Rows;++r){float s=acc[r];for(int off=8;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off,16);acc[r]=s;}
        if(sub==0){
            const int head=row/128;const float a=expf(a_log[head]);const float bias=dt_bias[row];
            for(int r=0;r<rows;++r){
                const float g=__half2float(__float2half_rn(acc[r]));
                if(second)sg2[(long long)r*n+row]=kda_sig(g);
                else{const float t=a*(g+bias);const float e=expf(-t)+1.f;decay[(long long)r*n+row]=expf(-5.f/e);}
            }
        }
    }
}
// GLM53_KDA_GATE_PT=1 (1..32 rows; L0 against kda_gate_first<R> / kda_gate_second<R>, i.e. also against the tile
// kernels): one thread per (row, output). Each thread forms the original lanes' partial sums itself, in the original
// lane's column order (fa/ga: columns l*8+256*i+j; wb: l*4+128*i'; fb/gb: sub*8+j) and adds them in the original
// shuffle tree's shape (xor 16/8/4/2/1, resp. 8/4/2/1; float addition is commutative, so lane-local operand order is
// irrelevant). Weights are converted to float once per block into shared memory and shared by its 8 rows (warps); lanes
// are consecutive outputs, so partial / decay / sg2 writes are coalesced and the epilogue runs per (row, output).
__device__ __forceinline__ float kda_tree32(const float* p){float a[16],b[8],c[4],d[2];
#pragma unroll
    for(int i=0;i<16;++i)a[i]=p[i]+p[i+16];
#pragma unroll
    for(int i=0;i<8;++i)b[i]=a[i]+a[i+8];
#pragma unroll
    for(int i=0;i<4;++i)c[i]=b[i]+b[i+4];
#pragma unroll
    for(int i=0;i<2;++i)d[i]=c[i]+c[i+2];
    return d[0]+d[1];}
__device__ __forceinline__ float kda_tree16(const float* p){float a[8],b[4],c[2];
#pragma unroll
    for(int i=0;i<8;++i)a[i]=p[i]+p[i+8];
#pragma unroll
    for(int i=0;i<4;++i)b[i]=a[i]+a[i+4];
#pragma unroll
    for(int i=0;i<2;++i)c[i]=b[i]+b[i+2];
    return c[0]+c[1];}
// grid (ceil(total/32), 4 quarters, ceil(rows/8)); total = 2*rank_a + nb with 2*rank_a % 32 == 0 (blocks never mix kinds).
__global__ void __launch_bounds__(256) kda_gate_first_pt(const float* __restrict__ x,const __half* __restrict__ fa,const __half* __restrict__ ga,
    const float* __restrict__ wb,float* __restrict__ part,int rows,int k,int rank_a,int nb){
    __shared__ __align__(16) float ws[32][260];          // 32 outputs x 256 columns of this chunk (+4 pad: conflict-free float4)
    __shared__ __align__(16) float xs[8][256];           // 8 rows x 256 columns (Half-rounded for fa/ga blocks)
    const int total=2*rank_a+nb,span=k/4,quarter=blockIdx.y,base=quarter*span;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int out0=blockIdx.x*32,out=out0+lane,r=blockIdx.z*8+warp;
    const bool half_blk=out0<2*rank_a;
    float p[32];
#pragma unroll
    for(int l=0;l<32;++l)p[l]=0.f;
    for(int c0=0;c0<span;c0+=256){
        __syncthreads();
        for(int i=threadIdx.x;i<32*64;i+=256){const int o=i>>6,c=(i&63)*4;const int oo=out0+o;float4 v=make_float4(0.f,0.f,0.f,0.f);
            if(oo<total){
                if(half_blk){const __half* w=(oo<rank_a?fa+(long long)oo*k:ga+(long long)(oo-rank_a)*k)+base+c0+c;
                    const uint2 raw=*reinterpret_cast<const uint2*>(w);const __half* h=reinterpret_cast<const __half*>(&raw);
                    v=make_float4(__half2float(h[0]),__half2float(h[1]),__half2float(h[2]),__half2float(h[3]));}
                else v=*reinterpret_cast<const float4*>(wb+(long long)(oo-2*rank_a)*k+base+c0+c);}
            *reinterpret_cast<float4*>(&ws[o][c])=v;}
        for(int i=threadIdx.x;i<8*64;i+=256){const int rr=i>>6,c=(i&63)*4;const int gr=blockIdx.z*8+rr;float4 v=make_float4(0.f,0.f,0.f,0.f);
            if(gr<rows){v=*reinterpret_cast<const float4*>(x+(long long)gr*k+base+c0+c);
                if(half_blk){v.x=__half2float(__float2half_rn(v.x));v.y=__half2float(__float2half_rn(v.y));v.z=__half2float(__float2half_rn(v.z));v.w=__half2float(__float2half_rn(v.w));}}
            *reinterpret_cast<float4*>(&xs[rr][c])=v;}
        __syncthreads();
        if(r>=rows||out>=total)continue;
        const float* wr=ws[lane];const float* xr=xs[warp];
        if(half_blk){
#pragma unroll
            for(int l=0;l<32;++l){const float4 w0=*reinterpret_cast<const float4*>(wr+l*8),w1=*reinterpret_cast<const float4*>(wr+l*8+4);
                const float4 a0=*reinterpret_cast<const float4*>(xr+l*8),a1=*reinterpret_cast<const float4*>(xr+l*8+4);float q=p[l];
                q=__fmaf_rn(a0.x,w0.x,q);q=__fmaf_rn(a0.y,w0.y,q);q=__fmaf_rn(a0.z,w0.z,q);q=__fmaf_rn(a0.w,w0.w,q);
                q=__fmaf_rn(a1.x,w1.x,q);q=__fmaf_rn(a1.y,w1.y,q);q=__fmaf_rn(a1.z,w1.z,q);q=__fmaf_rn(a1.w,w1.w,q);p[l]=q;}
        }else{
#pragma unroll
            for(int h=0;h<2;++h){
#pragma unroll
                for(int l=0;l<32;++l){const float4 w=*reinterpret_cast<const float4*>(wr+h*128+l*4),a=*reinterpret_cast<const float4*>(xr+h*128+l*4);float q=p[l];
                    q=__fmaf_rn(a.x,w.x,q);q=__fmaf_rn(a.y,w.y,q);q=__fmaf_rn(a.z,w.z,q);q=__fmaf_rn(a.w,w.w,q);p[l]=q;}}
        }
    }
    if(r<rows&&out<total)part[((long long)quarter*rows+r)*total+out]=kda_tree32(p);
}
// grid (2n/32, ceil(rows/8)); rank_a == 128; n % 32 == 0 (a block is all fb or all gb rows).
__global__ void __launch_bounds__(256) kda_gate_second_pt(const float* __restrict__ mid,const __half* __restrict__ fb,const __half* __restrict__ gb,
    const float* __restrict__ a_log,const float* __restrict__ dt_bias,float* __restrict__ decay,float* __restrict__ sg2,int rows,int n){
    __shared__ __align__(16) float ws[32][132];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int hw0=blockIdx.x*32;const bool second=hw0>=n;
    for(int i=threadIdx.x;i<32*32;i+=256){const int o=i>>5,c=(i&31)*4;const int row=(second?hw0-n:hw0)+o;
        const uint2 raw=*reinterpret_cast<const uint2*>((second?gb:fb)+(long long)row*128+c);const __half* h=reinterpret_cast<const __half*>(&raw);
        *reinterpret_cast<float4*>(&ws[o][c])=make_float4(__half2float(h[0]),__half2float(h[1]),__half2float(h[2]),__half2float(h[3]));}
    __syncthreads();
    const int r=blockIdx.y*8+warp;if(r>=rows)return;
    const int row=(second?hw0-n:hw0)+lane;
    const float* m=mid+(long long)r*256+(second?128:0);const float* wr=ws[lane];
    float p[16];
#pragma unroll
    for(int sub=0;sub<16;++sub){const float4 m0=*reinterpret_cast<const float4*>(m+sub*8),m1=*reinterpret_cast<const float4*>(m+sub*8+4);
        const float4 w0=*reinterpret_cast<const float4*>(wr+sub*8),w1=*reinterpret_cast<const float4*>(wr+sub*8+4);float q=0.f;
        q=__fmaf_rn(m0.x,w0.x,q);q=__fmaf_rn(m0.y,w0.y,q);q=__fmaf_rn(m0.z,w0.z,q);q=__fmaf_rn(m0.w,w0.w,q);
        q=__fmaf_rn(m1.x,w1.x,q);q=__fmaf_rn(m1.y,w1.y,q);q=__fmaf_rn(m1.z,w1.z,q);q=__fmaf_rn(m1.w,w1.w,q);p[sub]=q;}
    const float g=__half2float(__float2half_rn(kda_tree16(p)));
    const int head=row/128;
    if(second)sg2[(long long)r*n+row]=kda_sig(g);
    else{const float a=expf(a_log[head]);const float t=a*(g+dt_bias[row]);const float e=expf(-t)+1.f;decay[(long long)r*n+row]=expf(-5.f/e);}
}
// GLM53_KDA_GATE_ONE=1: K1, K1b and K2 in one persistent kernel with one grid barrier (L0: every output keeps its
// warp/half-warp lane-to-column mapping, per-lane accumulation order, shuffle trees and the fixed quarter-sum order of
// K1b; only scheduling changes). The fb/gb rows a block will need after the barrier do not depend on mid, so their loads
// are issued at kernel start and land while stage 1 runs; mid is rebuilt per block in shared memory straight from the
// quarter partials (never written to global). Grid <= co-resident blocks, so the spin barrier cannot deadlock.
__device__ unsigned int kda_gate_bar_count=0;
__device__ volatile unsigned int kda_gate_bar_gen=0;
template<int Rows,int MaxPass>
__global__ void __launch_bounds__(256) kda_gate_one(const float* __restrict__ x,const __half* __restrict__ fa,const __half* __restrict__ ga,
    const float* __restrict__ wb,const __half* __restrict__ fb,const __half* __restrict__ gb,const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,float* part,float* __restrict__ beta,float* __restrict__ decay,float* __restrict__ sg2,
    int rows,int k,int rank_a,int nb,int n){
    extern __shared__ float ms[];                       // [rows][2*rank_a]
    const int warp=threadIdx.x>>5,lane=threadIdx.x&31,sub=threadIdx.x&15,hwl=threadIdx.x>>4;
    const int total=2*rank_a+nb,span=k/4;
    // Stage-2 weights for this block's half-warp rows (one 16-byte slice per pass; rank_a==128).
    uint4 w2[MaxPass];
#pragma unroll
    for(int p=0;p<MaxPass;++p){const int hw=(p*gridDim.x+blockIdx.x)*16+hwl;
        if(hw<2*n){const bool second=hw>=n;const int row=second?hw-n:hw;
            w2[p]=*reinterpret_cast<const uint4*>((second?gb:fb)+(long long)row*rank_a+sub*8);}}
    // Stage 1 (K1): warp task = output*4 + quarter.
    for(int gw=blockIdx.x*8+warp;gw<total*4;gw+=gridDim.x*8){
        const int out=gw>>2,quarter=gw&3,base=quarter*span;
        float acc[Rows]={};
        if(out<2*rank_a){
            const __half* w=(out<rank_a?fa+(long long)out*k:ga+(long long)(out-rank_a)*k)+base;
            uint4 raw[4];
#pragma unroll
            for(int i=0;i<4;++i)raw[i]=*reinterpret_cast<const uint4*>(w+lane*8+i*256);
#pragma unroll
            for(int i=0;i<4;++i){const int c=lane*8+i*256;const __half* hv=reinterpret_cast<const __half*>(&raw[i]);
                float wf[8];
#pragma unroll
                for(int j=0;j<8;++j)wf[j]=__half2float(hv[j]);
#pragma unroll
                for(int r=0;r<Rows;++r){if(r<rows){const float* xr=x+(long long)r*k+base+c;const float4 a=*reinterpret_cast<const float4*>(xr),b=*reinterpret_cast<const float4*>(xr+4);
                    const float xs[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
                    for(int j=0;j<8;++j)acc[r]=__fmaf_rn(__half2float(__float2half_rn(xs[j])),wf[j],acc[r]);}}
            }
        }else{
            const float* w=wb+(long long)(out-2*rank_a)*k+base;
            float4 wv[8];
#pragma unroll
            for(int i=0;i<8;++i)wv[i]=*reinterpret_cast<const float4*>(w+lane*4+i*128);
#pragma unroll
            for(int i=0;i<8;++i){const int c=lane*4+i*128;
#pragma unroll
                for(int r=0;r<Rows;++r){if(r<rows){const float4 a=*reinterpret_cast<const float4*>(x+(long long)r*k+base+c);
                    acc[r]=__fmaf_rn(a.x,wv[i].x,acc[r]);acc[r]=__fmaf_rn(a.y,wv[i].y,acc[r]);acc[r]=__fmaf_rn(a.z,wv[i].z,acc[r]);acc[r]=__fmaf_rn(a.w,wv[i].w,acc[r]);}}
            }
        }
#pragma unroll
        for(int r=0;r<Rows;++r){float s=acc[r];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);acc[r]=s;}
        if(lane==0){for(int r=0;r<rows;++r)part[((long long)quarter*rows+r)*total+out]=acc[r];}
    }
    // Grid barrier (generation counter; the last arriver resets the count before releasing).
    __threadfence();__syncthreads();
    if(threadIdx.x==0){
        const unsigned int gen=kda_gate_bar_gen;__threadfence();
        if(atomicAdd(&kda_gate_bar_count,1u)==gridDim.x-1){kda_gate_bar_count=0;__threadfence();kda_gate_bar_gen=gen+1;}
        else{while(kda_gate_bar_gen==gen){}}
        __threadfence();
    }
    __syncthreads();
    // K1b: mid (Half-rounded fixed-order quarter sum) into shared memory; beta spread over the grid.
    const long long st=(long long)rows*total;
    for(int i=threadIdx.x;i<rows*2*rank_a;i+=256){const int r=i/(2*rank_a),o=i%(2*rank_a);const long long j=(long long)r*total+o;
        const float s=((__ldcg(part+j)+__ldcg(part+st+j))+__ldcg(part+2*st+j))+__ldcg(part+3*st+j);ms[i]=__half2float(__float2half_rn(s));}
    for(int i=blockIdx.x*256+threadIdx.x;i<rows*nb;i+=gridDim.x*256){const int r=i/nb,o=2*rank_a+i%nb;const long long j=(long long)r*total+o;
        const float s=((__ldcg(part+j)+__ldcg(part+st+j))+__ldcg(part+2*st+j))+__ldcg(part+3*st+j);beta[i]=kda_sig(s);}
    __syncthreads();
    // Stage 2 (K2) from the prefetched registers.
#pragma unroll
    for(int p=0;p<MaxPass;++p){
        const int hw=(p*gridDim.x+blockIdx.x)*16+hwl;if(hw>=2*n)break;
        const bool second=hw>=n;const int row=second?hw-n:hw;const int c=sub*8;
        const __half* hv=reinterpret_cast<const __half*>(&w2[p]);
        float acc[Rows]={};
#pragma unroll
        for(int j=0;j<8;++j){const float wf=__half2float(hv[j]);
#pragma unroll
            for(int r=0;r<Rows;++r)if(r<rows)acc[r]=__fmaf_rn(ms[r*2*rank_a+(second?rank_a:0)+c+j],wf,acc[r]);}
#pragma unroll
        for(int r=0;r<Rows;++r){float s=acc[r];for(int off=8;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off,16);acc[r]=s;}
        if(sub==0){
            const int head=row/128;const float a=expf(a_log[head]);const float bias=dt_bias[row];
            for(int r=0;r<rows;++r){
                const float g=__half2float(__float2half_rn(acc[r]));
                if(second)sg2[(long long)r*n+row]=kda_sig(g);
                else{const float t=a*(g+bias);const float e=expf(-t)+1.f;decay[(long long)r*n+row]=expf(-5.f/e);}
            }
        }
    }
}
// GLM53_KDA_GATE_V2=1 (L0 against kda_gate_first<8> / kda_gate_second<8>, k == 4096, rank_a == 128): the same warp and
// half-warp to output mapping, lane columns, per-lane FMA order and shuffle trees, with the row count a template parameter
// and every lane's weight vectors loaded before the first FMA (stage 1: 4 x 16 B Half or 8 x 16 B FP32; stage 2: each
// half-warp owns P consecutive output rows, its P weight vectors loaded together).
template<int R>
__global__ void __launch_bounds__(256) kda_gate_first_pf(const float* __restrict__ x,const __half* __restrict__ fa,const __half* __restrict__ ga,
    const float* __restrict__ wb,float* __restrict__ part,int rank_a,int nb){
    constexpr int K=4096,SPAN=K/4;
    const int gw=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
    const int total=2*rank_a+nb;if(gw>=total*4)return;
    const int out=gw>>2,quarter=gw&3,base=quarter*SPAN;
    float acc[R];
#pragma unroll
    for(int r=0;r<R;++r)acc[r]=0.f;
    if(out<2*rank_a){
        const __half* w=(out<rank_a?fa+(long long)out*K:ga+(long long)(out-rank_a)*K)+base;
        uint4 raw[SPAN/256];
#pragma unroll
        for(int it=0;it<SPAN/256;++it)raw[it]=*reinterpret_cast<const uint4*>(w+lane*8+it*256);
#pragma unroll
        for(int it=0;it<SPAN/256;++it){
            const int c=lane*8+it*256;const __half* hv=reinterpret_cast<const __half*>(&raw[it]);
            float wf[8];
#pragma unroll
            for(int j=0;j<8;++j)wf[j]=__half2float(hv[j]);
#pragma unroll
            for(int r=0;r<R;++r){const float* xr=x+(long long)r*K+base+c;const float4 a=*reinterpret_cast<const float4*>(xr),b=*reinterpret_cast<const float4*>(xr+4);
                const float xs[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
                for(int j=0;j<8;++j)acc[r]=__fmaf_rn(__half2float(__float2half_rn(xs[j])),wf[j],acc[r]);}
        }
    }else{
        const float* w=wb+(long long)(out-2*rank_a)*K+base;
        float4 wv[SPAN/128];
#pragma unroll
        for(int it=0;it<SPAN/128;++it)wv[it]=*reinterpret_cast<const float4*>(w+lane*4+it*128);
#pragma unroll
        for(int it=0;it<SPAN/128;++it){
            const int c=lane*4+it*128;
#pragma unroll
            for(int r=0;r<R;++r){const float4 a=*reinterpret_cast<const float4*>(x+(long long)r*K+base+c);
                acc[r]=__fmaf_rn(a.x,wv[it].x,acc[r]);acc[r]=__fmaf_rn(a.y,wv[it].y,acc[r]);acc[r]=__fmaf_rn(a.z,wv[it].z,acc[r]);acc[r]=__fmaf_rn(a.w,wv[it].w,acc[r]);}
        }
    }
#pragma unroll
    for(int r=0;r<R;++r){float s=acc[r];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);acc[r]=s;}
    if(lane==0){
#pragma unroll
        for(int r=0;r<R;++r)part[((long long)quarter*R+r)*total+out]=acc[r];}
}
template<int R,int P>
__global__ void __launch_bounds__(256) kda_gate_second_pf(const float* __restrict__ mid,const __half* __restrict__ fb,const __half* __restrict__ gb,
    const float* __restrict__ a_log,const float* __restrict__ dt_bias,float* __restrict__ decay,float* __restrict__ sg2,int n){
    constexpr int RA=128;
    const int hw0=(blockIdx.x*16+(threadIdx.x>>4))*P,sub=threadIdx.x&15;if(hw0>=2*n)return;
    uint4 raw[P];
#pragma unroll
    for(int p=0;p<P;++p){const int hw=hw0+p;const bool second=hw>=n;const int row=second?hw-n:hw;
        raw[p]=*reinterpret_cast<const uint4*>((second?gb:fb)+(long long)row*RA+sub*8);}
#pragma unroll
    for(int p=0;p<P;++p){
        const int hw=hw0+p;const bool second=hw>=n;const int row=second?hw-n:hw;const int c=sub*8;
        const __half* hv=reinterpret_cast<const __half*>(&raw[p]);
        float acc[R];
#pragma unroll
        for(int r=0;r<R;++r)acc[r]=0.f;
#pragma unroll
        for(int j=0;j<8;++j){const float wf=__half2float(hv[j]);
#pragma unroll
            for(int r=0;r<R;++r)acc[r]=__fmaf_rn(mid[(long long)r*2*RA+(second?RA:0)+c+j],wf,acc[r]);}
#pragma unroll
        for(int r=0;r<R;++r){float s=acc[r];for(int off=8;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off,16);acc[r]=s;}
        if(sub==0){
            const int head=row/128;const float a=expf(a_log[head]);const float bias=dt_bias[row];
#pragma unroll
            for(int r=0;r<R;++r){
                const float g=__half2float(__float2half_rn(acc[r]));
                if(second)sg2[(long long)r*n+row]=kda_sig(g);
                else{const float t=a*(g+bias);const float e=expf(-t)+1.f;decay[(long long)r*n+row]=expf(-5.f/e);}
            }
        }
    }
}
template<int R>
static void kda_gate_v2(const float* x,const void* fa,const void* ga,const float* wb,const void* fb,const void* gb,const float* a_log,const float* dt_bias,
    float* mid,float* beta,float* decay,float* sg2,float* part,int rank_a,int nb,int n,cudaStream_t s){
    const int t1=2*rank_a+nb;
    kda_gate_first_pf<R><<<(t1*4+7)/8,256,0,s>>>(x,(const __half*)fa,(const __half*)ga,wb,part,rank_a,nb);
    kda_gate_combine<<<(R*t1+255)/256,256,0,s>>>(part,mid,beta,R,rank_a,nb);
    kda_gate_second_pf<R,2><<<(2*n+31)/32,256,0,s>>>(mid,(const __half*)fb,(const __half*)gb,a_log,dt_bias,decay,sg2,n);
}
extern "C" int glm53_kda_gate_cuda(const float* x,const void* fa,const void* ga,const float* wb,const void* fb,const void* gb,
    const float* a_log,const float* dt_bias,float* mid,float* beta,float* decay,float* sg2,int rows,int k,int rank_a,int nb,int n,cudaStream_t s){
    // GLM53_KDA_GATE_WIDE=1: up to 32 rows in one call (L0: every row's arithmetic is the <8> kernels'), so batched
    // verify windows read the gate weights once instead of once per 8-row piece.
    static const bool wide=[]{const char* e=std::getenv("GLM53_KDA_GATE_WIDE");return e&&e[0]=='1';}();
    if(rows<1||rows>(wide?32:8)||k%256||rank_a%32||n%128)return int(cudaErrorInvalidValue);
    {static const bool pt=[]{const char* e=std::getenv("GLM53_KDA_GATE_PT");return e&&e[0]=='1';}();
    if(pt&&rank_a==128&&k%1024==0&&n%32==0){
        const int t1=2*rank_a+nb;float* part=mid+(long long)rows*2*rank_a;const int rb=(rows+7)/8;
        kda_gate_first_pt<<<dim3((t1+31)/32,4,rb),256,0,s>>>(x,(const __half*)fa,(const __half*)ga,wb,part,rows,k,rank_a,nb);
        kda_gate_combine<<<(rows*t1+255)/256,256,0,s>>>(part,mid,beta,rows,rank_a,nb);
        kda_gate_second_pt<<<dim3(2*n/32,rb),256,0,s>>>(mid,(const __half*)fb,(const __half*)gb,a_log,dt_bias,decay,sg2,rows,n);
        return int(cudaGetLastError());}}
    if(rows>8){
        if(k%1024)return int(cudaErrorInvalidValue);
        const int t1=2*rank_a+nb;float* part=mid+(long long)rows*2*rank_a;
        static const bool tile=[]{const char* e=std::getenv("GLM53_KDA_GATE_TILE");return e&&e[0]=='1';}();
        if(tile&&rank_a==128){
            const int blocks1=((2*rank_a)/8+(nb+7)/8)*4;
            if(rows<=16)kda_gate_first_tile<16><<<blocks1,256,0,s>>>(x,(const __half*)fa,(const __half*)ga,wb,part,rows,k,rank_a,nb);
            else kda_gate_first_tile<32><<<blocks1,256,0,s>>>(x,(const __half*)fa,(const __half*)ga,wb,part,rows,k,rank_a,nb);
            kda_gate_combine<<<(rows*t1+255)/256,256,0,s>>>(part,mid,beta,rows,rank_a,nb);
            if(rows<=16)kda_gate_second_tile<16><<<(2*n+63)/64,256,0,s>>>(mid,(const __half*)fb,(const __half*)gb,a_log,dt_bias,decay,sg2,rows,n);
            else kda_gate_second_tile<32><<<(2*n+63)/64,256,0,s>>>(mid,(const __half*)fb,(const __half*)gb,a_log,dt_bias,decay,sg2,rows,n);
            return int(cudaGetLastError());
        }
        if(rows<=16){kda_gate_first<16><<<(t1*4+7)/8,256,0,s>>>(x,(const __half*)fa,(const __half*)ga,wb,part,rows,k,rank_a,nb);}
        else{kda_gate_first<32><<<(t1*4+7)/8,256,0,s>>>(x,(const __half*)fa,(const __half*)ga,wb,part,rows,k,rank_a,nb);}
        kda_gate_combine<<<(rows*t1+255)/256,256,0,s>>>(part,mid,beta,rows,rank_a,nb);
        if(rows<=16)kda_gate_second<16><<<(2*n+15)/16,256,0,s>>>(mid,(const __half*)fb,(const __half*)gb,a_log,dt_bias,decay,sg2,rows,n,rank_a);
        else kda_gate_second<32><<<(2*n+15)/16,256,0,s>>>(mid,(const __half*)fb,(const __half*)gb,a_log,dt_bias,decay,sg2,rows,n,rank_a);
        return int(cudaGetLastError());
    }
    if(k%1024)return int(cudaErrorInvalidValue);
    const int t1=2*rank_a+nb;
    // part lives after mid in the caller's scratch: mid holds rows*2*rank_a, part 4*rows*t1 floats.
    float* part=mid+(long long)rows*2*rank_a;
    const char* sm=std::getenv("GLM53_KDA_GATE_SMEM");
    if(sm&&sm[0]=='1'&&(2*rank_a)%16==0&&nb%16==0&&n%64==0){
        const int blocks1=((2*rank_a)/16+nb/16)*4;const size_t sh1=(size_t)rows*(k/4)*sizeof(float);
        if(sh1>48*1024)cudaFuncSetAttribute(kda_gate_first_smem<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh1);
        kda_gate_first_smem<8><<<blocks1,256,sh1,s>>>(x,(const __half*)fa,(const __half*)ga,wb,part,rows,k,rank_a,nb);
        kda_gate_combine<<<(rows*t1+255)/256,256,0,s>>>(part,mid,beta,rows,rank_a,nb);
        kda_gate_second_smem<8><<<(2*n+63)/64,256,(size_t)rows*2*rank_a*sizeof(float),s>>>(mid,(const __half*)fb,(const __half*)gb,a_log,dt_bias,decay,sg2,rows,n,rank_a);
        return int(cudaGetLastError());
    }
    {static const bool v2=[]{const char* e=std::getenv("GLM53_KDA_GATE_V2");return e&&e[0]=='1';}();
    if(v2&&k==4096&&rank_a==128&&n%16==0){
        switch(rows){
#define C(R) case R: kda_gate_v2<R>(x,fa,ga,wb,fb,gb,a_log,dt_bias,mid,beta,decay,sg2,part,rank_a,nb,n,s);break;
            C(1)C(2)C(3)C(4)C(5)C(6)C(7)C(8)
#undef C
        }
        return int(cudaGetLastError());}}
    const char* one=std::getenv("GLM53_KDA_GATE_ONE");
    if(one&&one[0]=='1'&&k==4096&&rank_a==128){
        // Persistent grid = co-resident blocks (capped at the stage-1 warp tasks); each thread holds MaxPass stage-2 slices,
        // so grid*16*MaxPass must cover the 2n stage-2 rows: 4 slices when the grid allows, else 8.
        static int grid4=0,grid8=0;
        if(!grid4){int dev=0,sms=0,o4=0,o8=0;cudaGetDevice(&dev);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o4,kda_gate_one<8,4>,256,8*2*128*sizeof(float));
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o8,kda_gate_one<8,8>,256,8*2*128*sizeof(float));
            const int want=(t1*4+7)/8;grid4=sms*o4<want?sms*o4:want;grid8=sms*o8<want?sms*o8:want;
            if(std::getenv("GLM53_KDA_GATE_ONE_LOG"))fprintf(stderr,"[kda-gate-one] sms %d occ %d/%d grid %d/%d\n",sms,o4,o8,grid4,grid8);}
        const size_t sh=(size_t)rows*2*rank_a*sizeof(float);
        if(grid4*16*4>=2*n){
            kda_gate_one<8,4><<<grid4,256,sh,s>>>(x,(const __half*)fa,(const __half*)ga,wb,(const __half*)fb,(const __half*)gb,
                a_log,dt_bias,part,beta,decay,sg2,rows,k,rank_a,nb,n);
            return int(cudaGetLastError());
        }
        if(grid8*16*8>=2*n){
            kda_gate_one<8,8><<<grid8,256,sh,s>>>(x,(const __half*)fa,(const __half*)ga,wb,(const __half*)fb,(const __half*)gb,
                a_log,dt_bias,part,beta,decay,sg2,rows,k,rank_a,nb,n);
            return int(cudaGetLastError());
        }
    }
    kda_gate_first<8><<<(t1*4+7)/8,256,0,s>>>(x,(const __half*)fa,(const __half*)ga,wb,part,rows,k,rank_a,nb);
    kda_gate_combine<<<(rows*t1+255)/256,256,0,s>>>(part,mid,beta,rows,rank_a,nb);
    kda_gate_second<8><<<(2*n+15)/16,256,0,s>>>(mid,(const __half*)fb,(const __half*)gb,a_log,dt_bias,decay,sg2,rows,n,rank_a);
    return int(cudaGetLastError());
}
extern "C" int glm53_kda_onorm_gate_sig_cuda(const float* o,const float* onorm,int onorm_len,const float* g2,float* out,int groups,cudaStream_t s){
    kda_onorm_gate_sig<<<(groups+7)/8,256,0,s>>>(o,onorm,onorm_len,g2,out,groups);return int(cudaGetLastError());
}
extern "C" int glm53_kda_onorm_gate_cuda(const float* o,const float* onorm,int onorm_len,const float* sg2,float* out,int groups,cudaStream_t s){
    kda_onorm_gate<<<(groups+7)/8,256,0,s>>>(o,onorm,onorm_len,sg2,out,groups);return int(cudaGetLastError());
}

// ---- W05c (GLM53_NORM_FUSED=1) ----
// act [T, 3*H*128]: q = l2(act_q)*scale, k = l2(act_k), v = act_v (contiguous copies). One warp per (t,h,part).
__global__ void kda_chain_l2(const float* __restrict__ act,float* __restrict__ q,float* __restrict__ k,float* __restrict__ v,int t,int h,float scale){
    const int w=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(w>=t*h*3)return;
    const int part=w%3,th=w/3,row=th/h,head=th%h;
    const float* src=act+(long long)row*3*h*128+(long long)part*h*128+head*128;
    float x[4];float s=0.f;
#pragma unroll
    for(int j=0;j<4;++j){x[j]=src[lane+32*j];s+=x[j]*x[j];}
    for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
    float* dst=(part==0?q:part==1?k:v)+(long long)th*128;
    const float d=sqrtf(s+1e-6f);
#pragma unroll
    for(int j=0;j<4;++j){float y=x[j];if(part<2)y=y/d;if(part==0)y=y*scale;dst[lane+32*j]=y;}
}
// GLM53_KDA_CONV_L2=1 (with GLM53_KDA_CONV_SILU=1): kda_conv_chain<false,true> and kda_chain_l2 in one launch. Each lane
// computes its 4 channels' convolution + SiLU with kda_conv_chain's exact operations (same intrinsics, same order; this
// file is also built with --fmad=false), then the warp normalizes them exactly as kda_chain_l2 does. The activation
// [T, 3*H*128] is never written.
__global__ void kda_conv_l2(const float* __restrict__ base,const float* __restrict__ projected,const float* __restrict__ weights,
    float* __restrict__ q,float* __restrict__ k,float* __restrict__ v,int t,int h,float scale){
    const int w=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(w>=t*h*3)return;
    const int part=w%3,th=w/3,row=th/h,head=th%h,width=3*h*128,first=row-3;
    float x[4];float s=0.f;
#pragma unroll
    for(int j=0;j<4;++j){
        const int channel=part*h*128+head*128+lane+32*j;
        float rows[4];
#pragma unroll
        for(int i=0;i<4;++i){const int r=first+i;rows[i]=r<0?base[(r+3)*width+channel]:projected[r*width+channel];}
        float acc=__fmul_rn(weights[channel*4],rows[0]);
#pragma unroll
        for(int i=1;i<4;++i)acc=__fadd_rn(acc,__fmul_rn(weights[channel*4+i],rows[i]));
        acc=__fdiv_rn(acc,__fadd_rn(1.f,::exp(-acc)));
        x[j]=acc;s+=x[j]*x[j];
    }
    for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
    float* dst=(part==0?q:part==1?k:v)+(long long)th*128;
    const float d=sqrtf(s+1e-6f);
#pragma unroll
    for(int j=0;j<4;++j){float y=x[j];if(part<2)y=y/d;if(part==0)y=y*scale;dst[lane+32*j]=y;}
}
extern "C" int glm53_kda_conv_l2_cuda(const float* base,const float* projected,const float* weights,float* q,float* k,float* v,int t,int h,float scale,cudaStream_t s){
    if(t<1||t>16||h<1)return int(cudaErrorInvalidValue);
    kda_conv_l2<<<(t*h*3+7)/8,256,0,s>>>(base,projected,weights,q,k,v,t,h,scale);return int(cudaGetLastError());
}
extern "C" int glm53_kda_chain_l2_cuda(const float* act,float* q,float* k,float* v,int t,int h,float scale,cudaStream_t s){
    kda_chain_l2<<<(t*h*3+7)/8,256,0,s>>>(act,q,k,v,t,h,scale);return int(cudaGetLastError());
}
// RMSNorm rows: y = x * rsqrt(mean(x^2)+eps) * w, one block per row.
__global__ void rms_rows(const float* __restrict__ x,const float* __restrict__ w,float* __restrict__ y,int d,float eps,int ldx){
    const int r=blockIdx.x;const float* xr=x+(long long)r*ldx;__shared__ float ws[32];float s=0.f;
    for(int i=threadIdx.x;i<d;i+=blockDim.x){const float v=xr[i];s+=v*v;}
    for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
    if((threadIdx.x&31)==0)ws[threadIdx.x>>5]=s;__syncthreads();
    if(threadIdx.x<32){float t=threadIdx.x<(blockDim.x>>5)?ws[threadIdx.x]:0.f;for(int off=16;off;off>>=1)t+=__shfl_xor_sync(0xffffffff,t,off);if(threadIdx.x==0)ws[0]=rsqrtf(t/(float)d+eps);}
    __syncthreads();const float r2=ws[0];
    for(int i=threadIdx.x;i<d;i+=blockDim.x)y[(long long)r*d+i]=xr[i]*r2*w[i];
}
extern "C" int glm53_rms_rows_cuda(const float* x,const float* w,float* y,int rows,int d,float eps,cudaStream_t s){
    rms_rows<<<rows,256,0,s>>>(x,w,y,d,eps,d);return int(cudaGetLastError());
}
// Same per-row arithmetic on a row-strided input (a column slice of a wider buffer, e.g. a C12 group output).
extern "C" int glm53_rms_rows_ld_cuda(const float* x,int ldx,const float* w,float* y,int rows,int d,float eps,cudaStream_t s){
    if(ldx<d)return int(cudaErrorInvalidValue);
    rms_rows<<<rows,256,0,s>>>(x,w,y,d,eps,ldx);return int(cudaGetLastError());
}
__global__ void round_half_inplace(float* __restrict__ y,long long n){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__half2float(__float2half_rn(y[i]));
}
extern "C" int glm53_round_half_inplace_cuda(float* y,long long n,cudaStream_t s){
    round_half_inplace<<<(unsigned)((n+255)/256),256,0,s>>>(y,n);return int(cudaGetLastError());
}

// ---- P1 (GLM53_KDA_PREFILL_FUSED=1): decay = exp(-5 * reciprocal(exp(-(a*(g1+dt)))+1)) ----
// Replays the ATen op sequence of kda_chunk one rounding at a time (no FMA contraction).
__global__ void kda_decay_rows(const float* __restrict__ g1,const float* __restrict__ a,const float* __restrict__ dt,
    float* __restrict__ out,long long total,int heads){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
    const int hd=(int)(i%((long long)heads*128)),head=hd/128;
    const float t2=__fmul_rn(a[head],__fadd_rn(g1[i],dt[hd]));
    const float t5=__fadd_rn(expf(-t2),1.f);
    out[i]=expf(__fmul_rn(__fdiv_rn(1.f,t5),-5.f));
}
extern "C" int glm53_kda_decay_rows_cuda(const float* g1,const float* a,const float* dt,float* out,long long total,int heads,cudaStream_t s){
    if(total<1||heads<1)return int(cudaErrorInvalidValue);
    kda_decay_rows<<<(unsigned)((total+255)/256),256,0,s>>>(g1,a,dt,out,total,heads);return int(cudaGetLastError());
}

// ---- P1b (GLM53_KDA_PREFILL_FUSED=1): conv(4 taps) -> SiLU -> l2 per (row, head, part) ----
// Same rounding sequence as kda_conv_chain, ATen SiLU (x / (1 + expf(-x))) and kda_chain_l2.
// One warp owns 128 channels of one (part, head) for 16 consecutive rows; the 3-row
// convolution window stays in registers, so each input row is read once per warp.
__global__ void kda_conv_silu_l2(const float* __restrict__ base,const float* __restrict__ projected,const float* __restrict__ wall,
    float* __restrict__ q,float* __restrict__ k,float* __restrict__ v,int t,int h,float scale){
    const int wid=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;const int groups=3*h,tblocks=(t+15)/16;
    if(wid>=groups*tblocks)return;
    const int g=wid%groups,tb=wid/groups,part=g/h,head=g%h,width=3*h*128,c0=part*h*128+head*128,t0=tb*16;
    float wv[4][4],win[4][3];
#pragma unroll
    for(int j=0;j<4;++j){const int ch=c0+lane+32*j;
#pragma unroll
        for(int p=0;p<4;++p)wv[j][p]=wall[ch*4+p];
#pragma unroll
        for(int p=0;p<3;++p){const int row=t0-3+p;win[j][p]=row<0?base[(row+3)*width+ch]:projected[(long long)row*width+ch];}}
    float* dst_base=(part==0?q:part==1?k:v);
    for(int tt=0;tt<16;++tt){const int row=t0+tt;if(row>=t)break;
        float x[4];float s=0.f;
#pragma unroll
        for(int j=0;j<4;++j){const int ch=c0+lane+32*j;const float cur=projected[(long long)row*width+ch];
            float acc=__fmul_rn(wv[j][0],win[j][0]);acc=__fadd_rn(acc,__fmul_rn(wv[j][1],win[j][1]));
            acc=__fadd_rn(acc,__fmul_rn(wv[j][2],win[j][2]));acc=__fadd_rn(acc,__fmul_rn(wv[j][3],cur));
            win[j][0]=win[j][1];win[j][1]=win[j][2];win[j][2]=cur;
            x[j]=__fdiv_rn(acc,__fadd_rn(1.f,expf(-acc)));s+=x[j]*x[j];}
        for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
        const float d=sqrtf(s+1e-6f);float* dst=dst_base+((long long)row*h+head)*128;
#pragma unroll
        for(int j=0;j<4;++j){float y=x[j];if(part<2)y=y/d;if(part==0)y=y*scale;dst[lane+32*j]=y;}
    }
}
extern "C" int glm53_kda_conv_silu_l2_cuda(const float* base,const float* projected,const float* wall,float* q,float* k,float* v,int t,int h,float scale,cudaStream_t s){
    if(t<1||h<1)return int(cudaErrorInvalidValue);
    const int warps=3*h*((t+15)/16);
    kda_conv_silu_l2<<<(warps+7)/8,256,0,s>>>(base,projected,wall,q,k,v,t,h,scale);return int(cudaGetLastError());
}

// ---- GLM53_PREFILL_HALF_GLUE=1 (L0): the same kernels reading the Half GEMM results the FP32 tensors widened ----
__device__ __forceinline__ float kda_ld(const float* p,long long i){return p[i];}
__device__ __forceinline__ float kda_ld(const __half* p,long long i){return __half2float(p[i]);}
// kda_conv_silu_l2 with the q/k/v projections as three Half [t, h*128] results instead of one FP32 [t, 3*h*128] cat.
__global__ void kda_conv_silu_l2_h(const float* __restrict__ base,const __half* __restrict__ p0,const __half* __restrict__ p1,
    const __half* __restrict__ p2,const float* __restrict__ wall,float* __restrict__ q,float* __restrict__ k,float* __restrict__ v,int t,int h,float scale){
    const int wid=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;const int groups=3*h,tblocks=(t+15)/16;
    if(wid>=groups*tblocks)return;
    const int g=wid%groups,tb=wid/groups,part=g/h,head=g%h,width=3*h*128,pw=h*128,c0=part*h*128+head*128,t0=tb*16;
    const __half* src=part==0?p0:part==1?p1:p2;
    float wv[4][4],win[4][3];
#pragma unroll
    for(int j=0;j<4;++j){const int ch=c0+lane+32*j,pc=head*128+lane+32*j;
#pragma unroll
        for(int p=0;p<4;++p)wv[j][p]=wall[ch*4+p];
#pragma unroll
        for(int p=0;p<3;++p){const int row=t0-3+p;win[j][p]=row<0?base[(row+3)*width+ch]:__half2float(src[(long long)row*pw+pc]);}}
    float* dst_base=(part==0?q:part==1?k:v);
    for(int tt=0;tt<16;++tt){const int row=t0+tt;if(row>=t)break;
        float x[4];float s=0.f;
#pragma unroll
        for(int j=0;j<4;++j){const int pc=head*128+lane+32*j;const float cur=__half2float(src[(long long)row*pw+pc]);
            float acc=__fmul_rn(wv[j][0],win[j][0]);acc=__fadd_rn(acc,__fmul_rn(wv[j][1],win[j][1]));
            acc=__fadd_rn(acc,__fmul_rn(wv[j][2],win[j][2]));acc=__fadd_rn(acc,__fmul_rn(wv[j][3],cur));
            win[j][0]=win[j][1];win[j][1]=win[j][2];win[j][2]=cur;
            x[j]=__fdiv_rn(acc,__fadd_rn(1.f,expf(-acc)));s+=x[j]*x[j];}
        for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
        const float d=sqrtf(s+1e-6f);float* dst=dst_base+((long long)row*h+head)*128;
#pragma unroll
        for(int j=0;j<4;++j){float y=x[j];if(part<2)y=y/d;if(part==0)y=y*scale;dst[lane+32*j]=y;}
    }
}
extern "C" int glm53_kda_conv_silu_l2_h_cuda(const float* base,const void* p0,const void* p1,const void* p2,const float* wall,float* q,float* k,float* v,int t,int h,float scale,cudaStream_t s){
    if(t<1||h<1)return int(cudaErrorInvalidValue);
    const int warps=3*h*((t+15)/16);
    kda_conv_silu_l2_h<<<(warps+7)/8,256,0,s>>>(base,(const __half*)p0,(const __half*)p1,(const __half*)p2,wall,q,k,v,t,h,scale);return int(cudaGetLastError());
}
__global__ void kda_decay_rows_h(const __half* __restrict__ g1,const float* __restrict__ a,const float* __restrict__ dt,
    float* __restrict__ out,long long total,int heads){
    const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
    const int hd=(int)(i%((long long)heads*128)),head=hd/128;
    const float t2=__fmul_rn(a[head],__fadd_rn(__half2float(g1[i]),dt[hd]));
    const float t5=__fadd_rn(expf(-t2),1.f);
    out[i]=expf(__fmul_rn(__fdiv_rn(1.f,t5),-5.f));
}
extern "C" int glm53_kda_decay_rows_h_cuda(const void* g1,const float* a,const float* dt,float* out,long long total,int heads,cudaStream_t s){
    if(total<1||heads<1)return int(cudaErrorInvalidValue);
    kda_decay_rows_h<<<(unsigned)((total+255)/256),256,0,s>>>((const __half*)g1,a,dt,out,total,heads);return int(cudaGetLastError());
}
// kda_onorm_gate_sig with g2 as the Half GEMM result and the hidden written as Half (what the wo GEMM's Half input
// conversion, __float2half_rn, makes of the FP32 hidden).
__global__ void kda_onorm_gate_sig_hh(const float* __restrict__ o,const float* __restrict__ onorm,int onorm_len,const __half* __restrict__ g2,__half* __restrict__ out,int groups){
    const int g=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(g>=groups)return;
    const float* src=o+(long long)g*128;float v[4];float s=0.f;
#pragma unroll
    for(int j=0;j<4;++j){v[j]=src[lane+32*j];s+=v[j]*v[j];}
    for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
    const float r=rsqrtf(s/128.f+1e-5f);
#pragma unroll
    for(int j=0;j<4;++j){const long long i=(long long)g*128+lane+32*j;const int d=(int)(i%onorm_len);
        const float sg=1.0f/(1.0f+expf(-__half2float(g2[i])));out[i]=__float2half_rn((onorm[d]*(v[j]*r))*sg);}
}
extern "C" int glm53_kda_onorm_gate_sig_hh_cuda(const float* o,const float* onorm,int onorm_len,const void* g2,void* out,int groups,cudaStream_t s){
    kda_onorm_gate_sig_hh<<<(groups+7)/8,256,0,s>>>(o,onorm,onorm_len,(const __half*)g2,(__half*)out,groups);return int(cudaGetLastError());
}
