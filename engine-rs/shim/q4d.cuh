// SPDX-License-Identifier: MIT
// Dense format "Q4" (GLM53_DENSE_Q4_SET): affine 4-bit per weight plus (scale, minimum) as half2 per row and 64 k,
// 4.5 bits per weight (g64 affine; ranges MSE-searched). Lossy (L3): operand = Half_RN(q*scale + minimum).
//   q   per (row, quad of 128 k, lane_t): one uint32 per chunk u (nibble i = k (4*quad+u)*32 + lane_t*8 + i),
//       16 bytes at ((row*(K/128) + quad)*4 + lane_t)*16
//   sm  [N, K/64] half2 (scale, minimum); chunk u of a quad uses group 2*quad + u/2
// skinny_q4d follows skinny_c12/skinny_q8 (lane k-permutation, chunk order, split-K ranges, MMA sequence, reduction).
#pragma once
#include "fp8_skinny.cuh"

struct Q4View {const uint8_t* q; const half2* sm; int n,k;};

__device__ __forceinline__ uint4 q4d_dec8(uint32_t w,half2 sm){
  const float s=__low2float(sm),m=__high2float(sm);float f[8];
#pragma unroll
  for(int i=0;i<8;i++)f[i]=fmaf((float)((w>>(4*i))&15u),s,m);
  return make_uint4(f2h2(f[0],f[1]),f2h2(f[2],f[3]),f2h2(f[4],f[5]),f2h2(f[6],f[7]));
}
template<int KS,int NT,int OUT,int QU=1>
__global__ void __launch_bounds__(256) skinny_q4d(const float* __restrict__ X,const Q4View w,void* __restrict__ Y,int M){
  constexpr int RT=8/KS;
  __shared__ float red[KS][RT][NT][4][32];
  const int N=w.n,K=w.k;
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS,rbase=(blockIdx.x*RT+rt)*16;
  float c[NT][4];
#pragma unroll
  for(int j=0;j<NT;j++){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
  const int chunks=K>>5,c0=chunks*ks/KS,c1=chunks*(ks+1)/KS;
  const bool live=rbase<N;const int r0=min(rbase+g,N-1),r1=min(rbase+g+8,N-1);
  const uint8_t* m0=w.q+((size_t)r0*(K>>7)*4+t)*16;const uint8_t* m1=w.q+((size_t)r1*(K>>7)*4+t)*16;
  const uint2* s0=reinterpret_cast<const uint2*>(w.sm+(size_t)r0*(K>>6));const uint2* s1=reinterpret_cast<const uint2*>(w.sm+(size_t)r1*(K>>6));
  bool xv[NT];const float* xr[NT];
#pragma unroll
  for(int j=0;j<NT;j++){const int tok=j*8+g;xv[j]=tok<M;xr[j]=X+(size_t)(xv[j]?tok:0)*K+t*8;}
  if(live){
    for(int q0=c0>>2;q0<(c1>>2);q0+=QU){
      uint4 A[QU],B[QU];uint2 SA[QU],SB[QU];
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;A[v]=skinny_ld_stream(m0+(size_t)q*64);B[v]=skinny_ld_stream(m1+(size_t)q*64);SA[v]=__ldg(s0+q);SB[v]=__ldg(s1+q);}
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;
        const uint32_t W0[4]={A[v].x,A[v].y,A[v].z,A[v].w},W1[4]={B[v].x,B[v].y,B[v].z,B[v].w};
        const uint32_t sa[2]={SA[v].x,SA[v].y},sb[2]={SB[v].x,SB[v].y};
#pragma unroll
        for(int u=0;u<4;u++){
          const int cc=4*q+u;
          const uint4 a=q4d_dec8(W0[u],*reinterpret_cast<const half2*>(&sa[u>>1])),b=q4d_dec8(W1[u],*reinterpret_cast<const half2*>(&sb[u>>1]));
#pragma unroll
          for(int j=0;j<NT;j++){
            uint32_t xb[4]={0,0,0,0};
            if(xv[j]){const float4 p=__ldg(reinterpret_cast<const float4*>(xr[j]+(size_t)cc*32)),s=__ldg(reinterpret_cast<const float4*>(xr[j]+(size_t)cc*32+4));
              xb[0]=f2h2(p.x,p.y);xb[1]=f2h2(p.z,p.w);xb[2]=f2h2(s.x,s.y);xb[3]=f2h2(s.z,s.w);}
            skinny_mma16816(c[j],a.x,b.x,a.y,b.y,xb[0],xb[1]);
            skinny_mma16816(c[j],a.z,b.z,a.w,b.w,xb[2],xb[3]);
          }
        }
      }
    }
  }
#pragma unroll
  for(int j=0;j<NT;j++)
#pragma unroll
    for(int i=0;i<4;i++)red[ks][rt][j][i][lane]=c[j][i];
  __syncthreads();
  if(ks==0&&live){
#pragma unroll
    for(int j=0;j<NT;j++)
#pragma unroll
    for(int i=0;i<4;i++){float s=red[0][rt][j][i][lane];
#pragma unroll
      for(int q=1;q<KS;q++)s+=red[q][rt][j][i][lane];
      const int row=rbase+g+(i>>1)*8,tok=j*8+2*t+(i&1);
      if(row<N&&tok<M){
        if constexpr(OUT==2)reinterpret_cast<__half*>(Y)[(size_t)tok*N+row]=__float2half_rn(s);
        else reinterpret_cast<float*>(Y)[(size_t)tok*N+row]=OUT==1?__half2float(__float2half_rn(s)):s;}}
  }
}
inline int skinny_q4d_launch(const float* X,const Q4View& w,void* Y,int M,int out,int ks,cudaStream_t s){
  const int chunks=w.k>>5;
  if(M<1||M>32||w.n%16||w.k%128||(chunks/ks)*ks!=chunks||(chunks/ks)%4)return int(cudaErrorInvalidValue);
  auto grid=[&](int KS){int rt=8/KS;return (w.n+16*rt-1)/(16*rt);};
  static const int qu_env=[]{const char* e=std::getenv("GLM53_Q4_QU");return e?std::atoi(e):2;}();
  const bool qu2=qu_env==2&&(chunks/ks/4)%2==0;
#define L(KS,NT,O) do{if(qu2)skinny_q4d<KS,NT,O,2><<<grid(KS),256,0,s>>>(X,w,Y,M);else skinny_q4d<KS,NT,O,1><<<grid(KS),256,0,s>>>(X,w,Y,M);}while(0)
#define LO(KS,NT) do{if(out==0)L(KS,NT,0);else if(out==1)L(KS,NT,1);else L(KS,NT,2);}while(0)
#define LK(NT) do{if(ks==2)LO(2,NT);else if(ks==4)LO(4,NT);else LO(8,NT);}while(0)
  if(M<=8)LK(1);else if(M<=16)LK(2);else LK(4);
#undef LK
#undef LO
#undef L
  return int(cudaGetLastError());
}
// ---- Encoder. q4d_ranges: one warp per (row, group of 64): (scale, minimum) as the MSE-best of 8 shrinks of [min, max]
// (in the Half values the GEMM decodes). q4d_pack: one thread per (row, quad, lane_t); squared error / source into err.
__device__ __forceinline__ float q4bf(uint16_t v){return __uint_as_float((uint32_t)v<<16);}
__device__ __forceinline__ float q4dq(float q,half2 sm){return __half2float(__float2half_rn(fmaf(q,__low2float(sm),__high2float(sm))));}
__global__ void q4d_ranges(const uint16_t* __restrict__ W,int N,int K,int mse,half2* __restrict__ SM){
  const long long gid=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;const int lane=threadIdx.x&31;
  if(gid>=(long long)N*(K>>6))return;
  const int r=(int)(gid/(K>>6)),gq=(int)(gid%(K>>6));const uint16_t* w=W+(size_t)r*K+gq*64;
  const float v0=q4bf(w[lane]),v1=q4bf(w[lane+32]);float lo=fminf(v0,v1),hi=fmaxf(v0,v1);
#pragma unroll
  for(int o=16;o;o>>=1){lo=fminf(lo,__shfl_xor_sync(0xffffffffu,lo,o));hi=fmaxf(hi,__shfl_xor_sync(0xffffffffu,hi,o));}
  half2 best=__floats2half2_rn(hi>lo?(hi-lo)/15.f:1.f,lo);float be=INFINITY;
  const int tries=mse?8:1;
  for(int c=0;c<tries;c++){
    const float sh=0.03f*c,a=lo+(hi-lo)*sh*0.5f,b=hi-(hi-lo)*sh*0.5f;
    const half2 sm=__floats2half2_rn(b>a?(b-a)/15.f:1.f,a);const float s=__low2float(sm),m=__high2float(sm);
    if(!(s>0.f))continue;
    const float q0=fminf(fmaxf(rintf((v0-m)/s),0.f),15.f),q1=fminf(fmaxf(rintf((v1-m)/s),0.f),15.f);
    float e=(q4dq(q0,sm)-v0)*(q4dq(q0,sm)-v0)+(q4dq(q1,sm)-v1)*(q4dq(q1,sm)-v1);
#pragma unroll
    for(int o=16;o;o>>=1)e+=__shfl_xor_sync(0xffffffffu,e,o);
    if(e<be){be=e;best=sm;}
  }
  if(lane==0)SM[gid]=best;
}
__global__ void q4d_pack(const uint16_t* __restrict__ W,int N,int K,const half2* __restrict__ SM,uint8_t* __restrict__ Q,double* err){
  const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x,per=(long long)(K>>7)*4;
  float e2=0.f,r2=0.f;
  if(id<(long long)N*per){
    const int r=int(id/per),q=int((id%per)>>2),t=int(id&3);const uint16_t* w=W+(size_t)r*K;
    uint32_t o[4];
    for(int u=0;u<4;u++){const half2 sm=SM[(size_t)r*(K>>6)+2*q+(u>>1)];const float s=__low2float(sm),m=__high2float(sm);uint32_t x=0;
      for(int i=0;i<8;i++){const float v=q4bf(w[(4*q+u)*32+t*8+i]);const float qv=fminf(fmaxf(rintf((v-m)/s),0.f),15.f);
        x|=(uint32_t)qv<<(4*i);const float d=q4dq(qv,sm)-v;e2+=d*d;r2+=v*v;}
      o[u]=x;}
    *reinterpret_cast<uint4*>(Q+((size_t)r*(K>>7)*4+(size_t)q*4+t)*16)=make_uint4(o[0],o[1],o[2],o[3]);
  }
#pragma unroll
  for(int o=16;o;o>>=1){e2+=__shfl_xor_sync(0xffffffffu,e2,o);r2+=__shfl_xor_sync(0xffffffffu,r2,o);}
  if((threadIdx.x&31)==0&&(e2>0.f||r2>0.f)){atomicAdd(err,(double)e2);atomicAdd(err+1,(double)r2);}
}
