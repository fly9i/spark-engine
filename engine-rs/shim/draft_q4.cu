// SPDX-License-Identifier: MIT
// GLM53_DRAFT_Q4=1: DFlash2 drafter weights as affine 4-bit (one FP32 scale and minimum per row and 128 k), read by
// a skinny BF16 GEMM. Draft side only (L2): drafts may change, verified output cannot.
//   q  per (row, quad of 128 k, lane_t): one uint32 per chunk u (k = (4*quad+u)*32 + lane_t*8 + i at nibble i),
//      16 bytes at ((row*(K/128) + quad)*4 + lane_t)*16
//   sm [N, K/128] float2 (scale, minimum): w = q*scale + minimum
// Lane k-permutation, chunk order, split-K and reduction follow skinny_c12 (c12.cuh); X is BF16 [M,K].
// Tiled copy (glm53_draft_q4t_*; the Qwen MTP draft head's Q8-style tiling): per (16-row tile T, quad) 1024 contiguous
// bytes, rows 0-7 of the tile then rows 8-15, lane (g, t) at lane*16, i.e. q at (((T*(K/128)+quad)*2+half)*32+lane)*16;
// sm per (T, quad) 16 float2 in row order. A warp load is one contiguous 512 B (the row layout: 8 rows x 64 B); the decoded
// operands, MMA order and reduction are unchanged (bitwise equal outputs).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include "fp8_skinny.cuh"

__device__ __forceinline__ uint32_t f2bf2(float a,float b){__nv_bfloat162 h=__floats2bfloat162_rn(a,b);return *reinterpret_cast<uint32_t*>(&h);}
__device__ __forceinline__ uint4 q4_dec8(uint32_t w,float2 sm){
  float f[8];
#pragma unroll
  for(int i=0;i<8;i++)f[i]=fmaf((float)((w>>(4*i))&15u),sm.x,sm.y);
  return make_uint4(f2bf2(f[0],f[1]),f2bf2(f[2],f[3]),f2bf2(f[4],f[5]),f2bf2(f[6],f[7]));
}
// XF: X is FP32 (rounded to BF16 RN in-kernel, as x.to_kind(BF16) does); YB: Y written as BF16 RN (y.to_kind(BF16)).
template<int KS,int NT,bool XF=false,bool YB=false,bool TL=false>
__global__ void __launch_bounds__(256) skinny_q4_bf(const void* __restrict__ Xv,const uint8_t* __restrict__ Q,const float2* __restrict__ SM,
                                                     void* __restrict__ Yv,int M,int N,int K){
  constexpr int RT=8/KS;
  __shared__ float red[KS][RT][NT][4][32];
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS,rbase=(blockIdx.x*RT+rt)*16;
  float c[NT][4];
#pragma unroll
  for(int j=0;j<NT;j++){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
  const int chunks=K>>5,c0=chunks*ks/KS,c1=chunks*(ks+1)/KS;
  const bool live=rbase<N;const int r0=min(rbase+g,N-1),r1=min(rbase+g+8,N-1);
  const size_t tq=(size_t)(rbase>>4)*(K>>7);   // TL: first (tile, quad) of this row tile
  const uint8_t* q0p=TL?Q+tq*1024+lane*16:Q+((size_t)r0*(K>>7)*4+t)*16;const uint8_t* q1p=TL?q0p+512:Q+((size_t)r1*(K>>7)*4+t)*16;
  const float2* s0=TL?SM+tq*16+g:SM+(size_t)r0*(K>>7);const float2* s1=TL?s0+8:SM+(size_t)r1*(K>>7);
  constexpr int QS=TL?1024:64,SS=TL?16:1;
  bool xv[NT];size_t xo[NT];
#pragma unroll
  for(int j=0;j<NT;j++){const int tok=j*8+g;xv[j]=tok<M;xo[j]=(size_t)(xv[j]?tok:0)*K+t*8;}
  if(live){
    for(int q=c0>>2;q<(c1>>2);q+=2){
      uint4 A[2],B[2];float2 SA[2],SB[2];
#pragma unroll
      for(int v=0;v<2;v++){const int qq=min(q+v,(c1>>2)-1);A[v]=skinny_ld_stream(q0p+(size_t)qq*QS);B[v]=skinny_ld_stream(q1p+(size_t)qq*QS);SA[v]=__ldg(s0+qq*SS);SB[v]=__ldg(s1+qq*SS);}
#pragma unroll
      for(int v=0;v<2;v++){
        if(q+v>=(c1>>2))break;
        const uint32_t W0[4]={A[v].x,A[v].y,A[v].z,A[v].w},W1[4]={B[v].x,B[v].y,B[v].z,B[v].w};
#pragma unroll
        for(int u=0;u<4;u++){
          const int cc=4*(q+v)+u;
          const uint4 a=q4_dec8(W0[u],SA[v]),b=q4_dec8(W1[u],SB[v]);
#pragma unroll
          for(int j=0;j<NT;j++){
            uint4 xb=make_uint4(0,0,0,0);
            if(xv[j]){
              if constexpr(XF){const float* xp=reinterpret_cast<const float*>(Xv)+xo[j]+(size_t)cc*32;const float4 p=*reinterpret_cast<const float4*>(xp),r=*reinterpret_cast<const float4*>(xp+4);
                xb=make_uint4(f2bf2(p.x,p.y),f2bf2(p.z,p.w),f2bf2(r.x,r.y),f2bf2(r.z,r.w));}
              else xb=*reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(Xv)+xo[j]+(size_t)cc*32);}
            skinny_mma16816_bf(c[j],a.x,b.x,a.y,b.y,xb.x,xb.y);
            skinny_mma16816_bf(c[j],a.z,b.z,a.w,b.w,xb.z,xb.w);
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
      for(int qq=1;qq<KS;qq++)s+=red[qq][rt][j][i][lane];
      const int row=rbase+g+(i>>1)*8,tok=j*8+2*t+(i&1);
      if(row<N&&tok<M){if constexpr(YB)reinterpret_cast<__nv_bfloat16*>(Yv)[(size_t)tok*N+row]=__float2bfloat16_rn(s);else reinterpret_cast<float*>(Yv)[(size_t)tok*N+row]=s;}}
  }
}
static int q4_ks_ok(int k,int ks){const int chunks=k>>5;return chunks%ks==0&&(chunks/ks)%4==0;}
// Split-K per launch: GLM53_DRAFT_Q4_KS (2/4/8) when it divides, else the widest valid split (KS 8 measured fastest or
// within 3% on every drafter shape).
static int q4_ks(int k){
  static const int env=[]{const char* e=std::getenv("GLM53_DRAFT_Q4_KS");return e?std::atoi(e):0;}();
  if(env&&q4_ks_ok(k,env))return env;
  for(int ks:{8,4,2})if(q4_ks_ok(k,ks))return ks;
  return 0;}
extern "C" int glm53_draft_q4_ok(int n,int k){return n%16==0&&k%128==0&&q4_ks(k)>0;}
// xf: X is FP32 (else BF16); yb: Y is BF16 (else FP32).
template<bool TL>
static int q4_gemm2(const void* x,int xf,const void* q,const void* sm,void* y,int yb,int m,int n,int k,cudaStream_t s){
  const int ks=q4_ks(k);if(m<1||m>32||n%16||k%128||!ks)return int(cudaErrorInvalidValue);
  const int rt=8/ks,grid=(n+16*rt-1)/(16*rt);
  const uint8_t* Q=(const uint8_t*)q;const float2* SM=(const float2*)sm;
#define L4(KS,NT,XF,YB) skinny_q4_bf<KS,NT,XF,YB,TL><<<grid,256,0,s>>>(x,Q,SM,y,m,n,k)
#define LXY(KS,NT) do{if(xf){if(yb)L4(KS,NT,true,true);else L4(KS,NT,true,false);}else{if(yb)L4(KS,NT,false,true);else L4(KS,NT,false,false);}}while(0)
#define LK(NT) do{if(ks==2)LXY(2,NT);else if(ks==4)LXY(4,NT);else LXY(8,NT);}while(0)
  if(m<=8)LK(1);else if(m<=16)LK(2);else LK(4);
#undef LK
#undef LXY
#undef L4
  return int(cudaGetLastError());
}
extern "C" int glm53_draft_q4_gemm2_cuda(const void* x,int xf,const void* q,const void* sm,void* y,int yb,int m,int n,int k,cudaStream_t s){
  return q4_gemm2<false>(x,xf,q,sm,y,yb,m,n,k,s);
}
// The same GEMM on a tiled copy (see the top of this file).
extern "C" int glm53_draft_q4t_gemm2_cuda(const void* x,int xf,const void* q,const void* sm,void* y,int yb,int m,int n,int k,cudaStream_t s){
  return q4_gemm2<true>(x,xf,q,sm,y,yb,m,n,k,s);
}
__global__ void q4_tile(const uint8_t* __restrict__ q,const float2* __restrict__ sm,int n,int k,uint8_t* __restrict__ qt,float2* __restrict__ smt){
  const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x,tqs=(long long)(n/16)*(k>>7);if(id>=tqs*64)return;
  const long long tq=id>>6;const int j=(int)(id&63),half=j>>5,lane=j&31;
  const int tile=(int)(tq/(k>>7)),quad=(int)(tq%(k>>7)),row=tile*16+half*8+(lane>>2),t=lane&3;
  *reinterpret_cast<uint4*>(qt+(size_t)((tq*2+half)*32+lane)*16)=*reinterpret_cast<const uint4*>(q+((size_t)row*(k>>7)*4+(size_t)quad*4+t)*16);
  if(j<16)smt[tq*16+j]=sm[(size_t)(tile*16+j)*(k>>7)+quad];
}
// Tiled copy (qt [n*k/2] bytes, smt [n*k/128] float2) of a row-layout q / sm.
extern "C" int glm53_draft_q4t_tile_cuda(const void* q,const void* sm,int n,int k,void* qt,void* smt,cudaStream_t s){
  if(n%16||k%128)return int(cudaErrorInvalidValue);
  const long long th=(long long)(n/16)*(k>>7)*64;
  q4_tile<<<(unsigned)((th+255)/256),256,0,s>>>((const uint8_t*)q,(const float2*)sm,n,k,(uint8_t*)qt,(float2*)smt);
  return int(cudaGetLastError());
}
extern "C" int glm53_draft_q4_gemm_cuda(const void* x,const void* q,const void* sm,float* y,int m,int n,int k,cudaStream_t s){
  return glm53_draft_q4_gemm2_cuda(x,0,q,sm,y,0,m,n,k,s);
}
// Encoder: one warp per (row, group of 128 k) computes (scale, minimum); then one thread per (row, quad, lane_t) packs.
// Ranges: the MSE-best of 6 symmetric shrinks of [min, max] (GLM53_DRAFT_Q4_MSE=0: plain min/max).
__device__ __forceinline__ float bf16v(uint16_t v){return __uint_as_float((uint32_t)v<<16);}
__global__ void q4_ranges(const uint16_t* __restrict__ W,int N,int K,int mse,float2* __restrict__ SM){
  const long long gid=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;const int lane=threadIdx.x&31;
  if(gid>=(long long)N*(K>>7))return;
  const int r=(int)(gid/(K>>7)),q=(int)(gid%(K>>7));const uint16_t* w=W+(size_t)r*K+q*128;
  float v[4],lo=INFINITY,hi=-INFINITY;
#pragma unroll
  for(int i=0;i<4;i++){v[i]=bf16v(w[lane+32*i]);lo=fminf(lo,v[i]);hi=fmaxf(hi,v[i]);}
#pragma unroll
  for(int o=16;o;o>>=1){lo=fminf(lo,__shfl_xor_sync(0xffffffffu,lo,o));hi=fmaxf(hi,__shfl_xor_sync(0xffffffffu,hi,o));}
  float2 best=make_float2(hi>lo?(hi-lo)/15.f:1.f,lo);float be=INFINITY;
  const int tries=mse?6:1;
  for(int c=0;c<tries;c++){
    const float sh=0.04f*c,a=lo+(hi-lo)*sh*0.5f,b=hi-(hi-lo)*sh*0.5f;const float s=b>a?(b-a)/15.f:1.f;float e=0.f;
#pragma unroll
    for(int i=0;i<4;i++){const float qv=fminf(fmaxf(rintf((v[i]-a)/s),0.f),15.f);const float d=fmaf(qv,s,a)-v[i];e+=d*d;}
#pragma unroll
    for(int o=16;o;o>>=1)e+=__shfl_xor_sync(0xffffffffu,e,o);
    if(e<be){be=e;best=make_float2(s,a);}
  }
  if(lane==0)SM[gid]=best;
}
__global__ void q4_pack(const uint16_t* __restrict__ W,int N,int K,const float2* __restrict__ SM,uint32_t* __restrict__ Q){
  const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x,per=(long long)(K>>7)*4;if(id>=(long long)N*per)return;
  const int r=int(id/per),q=int((id%per)>>2),t=int(id&3);const float2 sm=SM[(size_t)r*(K>>7)+q];const uint16_t* w=W+(size_t)r*K;
  uint32_t o[4];
  for(int u=0;u<4;u++){uint32_t x=0;for(int i=0;i<8;i++){const float v=bf16v(w[(4*q+u)*32+t*8+i]);
    const uint32_t qv=(uint32_t)fminf(fmaxf(rintf((v-sm.y)/sm.x),0.f),15.f);x|=qv<<(4*i);}o[u]=x;}
  *reinterpret_cast<uint4*>(Q+((size_t)r*(K>>7)*4+(size_t)q*4+t)*4)=make_uint4(o[0],o[1],o[2],o[3]);
}
extern "C" int glm53_draft_q4_encode_cuda(const void* w,int n,int k,int mse,void* sm,void* q,cudaStream_t s){
  if(n%16||k%128)return int(cudaErrorInvalidValue);
  const long long groups=(long long)n*(k/128);
  q4_ranges<<<(unsigned)((groups*32+255)/256),256,0,s>>>((const uint16_t*)w,n,k,mse,(float2*)sm);
  const long long th=groups*4;
  q4_pack<<<(unsigned)((th+255)/256),256,0,s>>>((const uint16_t*)w,n,k,(const float2*)sm,(uint32_t*)q);
  return int(cudaGetLastError());
}
