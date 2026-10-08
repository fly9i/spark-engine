// SPDX-License-Identifier: MIT
// Near-lossless dense format "Q8" (GLM53_Q8=1): symmetric int8 per weight plus one FP32 scale per (row, 128 k),
// 8.25 bits per weight instead of C12's ~12. Lossy (requantization of the BF16 checkpoint values): the decoded
// operand is Half_RN(q * scale) with |q| <= 127 and scale chosen per group (absmax, or the MSE-best clip).
//   q8 [N,K] int8 in C12's m8 storage order (each lane issues 16-byte loads):
//      ((row*(K/64) + pair)*4 + lane_t)*16 + (chunk&1)*8 + i   holds k = chunk*32 + lane_t*8 + i, pair = chunk/2
//   s  [N,K/128] FP32, group = 4 chunks (one "quad" of skinny_c12)
// skinny_q8 is skinny_c12 with the weight decode swapped: same lane k-permutation, chunk order, split-K ranges,
// MMA sequence and reduction, so only the operand values differ.
#pragma once
#include "fp8_skinny.cuh"

struct Q8View {const uint8_t* q; const float* s; int n,k;};

// 8 weights (k = kbase..kbase+7) of one row -> 4 half2 registers in skinny_half's operand order.
__device__ __forceinline__ uint4 q8_dec8(uint32_t lo,uint32_t hi,float s){
  float f[8];
#pragma unroll
  for(int i=0;i<8;i++){const int v=(int)(int8_t)(((i<4?lo:hi)>>(8*(i&3)))&0xffu);f[i]=(float)v*s;}
  return make_uint4(f2h2(f[0],f[1]),f2h2(f[2],f[3]),f2h2(f[4],f[5]),f2h2(f[6],f[7]));
}

// Float X [M,K] (Half RN in-kernel), FP32 accumulate. OUT: 0 FP32, 1 FP32 of the Half-rounded value, 2 Half.
// XH (GLM53_Q8_XHALF): X arrives already Half-rounded (same values as the in-kernel f2h2), one 16-byte load per 8 k.
template<int KS,int NT,int OUT,int QU=1,bool XH=false>
__global__ void __launch_bounds__(256) skinny_q8(const float* __restrict__ X,const Q8View w,void* __restrict__ Y,int M){
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
  const uint8_t* m0=w.q+((size_t)r0*(K>>6)*4+t)*16;const uint8_t* m1=w.q+((size_t)r1*(K>>6)*4+t)*16;
  const float* s0=w.s+(size_t)r0*(K>>7);const float* s1=w.s+(size_t)r1*(K>>7);
  bool xv[NT];const float* xr[NT];
#pragma unroll
  for(int j=0;j<NT;j++){const int tok=j*8+g;xv[j]=tok<M;xr[j]=X+(size_t)(xv[j]?tok:0)*K+t*8;}
  if(live){
    for(int q0=c0>>2;q0<(c1>>2);q0+=QU){
      uint4 PA0[QU],PB0[QU],PA1[QU],PB1[QU];float SA[QU],SB[QU];
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;
        PA0[v]=skinny_ld_stream(m0+(size_t)(2*q)*64);PB0[v]=skinny_ld_stream(m0+(size_t)(2*q+1)*64);
        PA1[v]=skinny_ld_stream(m1+(size_t)(2*q)*64);PB1[v]=skinny_ld_stream(m1+(size_t)(2*q+1)*64);
        SA[v]=__ldg(s0+q);SB[v]=__ldg(s1+q);}
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;
      const uint4 pa0=PA0[v],pb0=PB0[v],pa1=PA1[v],pb1=PB1[v];
      const uint32_t M0[8]={pa0.x,pa0.y,pa0.z,pa0.w,pb0.x,pb0.y,pb0.z,pb0.w},M1[8]={pa1.x,pa1.y,pa1.z,pa1.w,pb1.x,pb1.y,pb1.z,pb1.w};
#pragma unroll
      for(int u=0;u<4;u++){
        const int cc=4*q+u;
        const uint4 a=q8_dec8(M0[2*u],M0[2*u+1],SA[v]),b=q8_dec8(M1[2*u],M1[2*u+1],SB[v]);
#pragma unroll
        for(int j=0;j<NT;j++){
          uint32_t xb[4]={0,0,0,0};
          if(xv[j]){
            if constexpr(XH){const uint4 h=__ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(X)+(size_t)(j*8+g)*K+t*8+(size_t)cc*32));
              xb[0]=h.x;xb[1]=h.y;xb[2]=h.z;xb[3]=h.w;}
            else{const float4 p=__ldg(reinterpret_cast<const float4*>(xr[j]+(size_t)cc*32)),s=__ldg(reinterpret_cast<const float4*>(xr[j]+(size_t)cc*32+4));
            xb[0]=f2h2(p.x,p.y);xb[1]=f2h2(p.z,p.w);xb[2]=f2h2(s.x,s.y);xb[3]=f2h2(s.z,s.w);}}
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
inline int skinny_q8_launch(const float* X,const Q8View& w,void* Y,int M,int out,int ks,cudaStream_t s,bool xhalf=false){
  const int chunks=w.k>>5;
  if(M<1||M>32||w.n%16||w.k%128||(chunks/ks)*ks!=chunks||(chunks/ks)%4)return int(cudaErrorInvalidValue);
  auto grid=[&](int KS){int rt=8/KS;return (w.n+16*rt-1)/(16*rt);};
  // GLM53_Q8_QU (read once, default 2): quads per iteration; 2 only when every split holds an even quad count.
  static const int qu_env=[]{const char* e=std::getenv("GLM53_Q8_QU");return e?std::atoi(e):2;}();
  const bool qu2=qu_env==2&&(chunks/ks/4)%2==0;
#define L(KS,NT,O) do{if(xhalf){if(qu2)skinny_q8<KS,NT,O,2,true><<<grid(KS),256,0,s>>>(X,w,Y,M);else skinny_q8<KS,NT,O,1,true><<<grid(KS),256,0,s>>>(X,w,Y,M);}\
  else if(qu2)skinny_q8<KS,NT,O,2><<<grid(KS),256,0,s>>>(X,w,Y,M);else skinny_q8<KS,NT,O,1><<<grid(KS),256,0,s>>>(X,w,Y,M);}while(0)
#define LO(KS,NT) do{if(out==0)L(KS,NT,0);else if(out==1)L(KS,NT,1);else L(KS,NT,2);}while(0)
#define LK(NT) do{if(ks==2)LO(2,NT);else if(ks==4)LO(4,NT);else LO(8,NT);}while(0)
  if(M<=8)LK(1);else if(M<=16)LK(2);else LK(4);
#undef LK
#undef LO
#undef L
  return int(cudaGetLastError());
}

// GLM53_Q8_TILED=1 (L0 against skinny_q8): the same int8 values and scales in a tiled layout. Per 16-row tile and quad
// (128 k) 2048 B of int8 arranged so that load j (PA0, PB0, PA1, PB1 of skinny_q8) of lane l is bytes [j*512 + l*16, +16),
// then the tile's 16 FP32 scales of that quad (64 B); tile stride 2112 B, tiles ordered (row tile, quad). Every warp load
// is one contiguous 512 B instead of 16 rows x 64 B, and a warp's split-K range is one contiguous run of tiles. Same
// decoded operands, lane k-permutation, chunk order, split-K ranges, MMA sequence and reduction: bitwise skinny_q8.
constexpr int Q8T_TILE=2112;
struct Q8T {const uint8_t* t; int n,k;};
template<int KS,int NT,int OUT,int QU=1,bool XH=false>
__global__ void __launch_bounds__(256) skinny_q8t(const float* __restrict__ X,const Q8T w,void* __restrict__ Y,int M){
  constexpr int RT=8/KS;
  __shared__ float red[KS][RT][NT][4][32];
  const int N=w.n,K=w.k,Q=K>>7;
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS,tile=blockIdx.x*RT+rt,rbase=tile*16;
  float c[NT][4];
#pragma unroll
  for(int j=0;j<NT;j++){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
  const int chunks=K>>5,c0=chunks*ks/KS,c1=chunks*(ks+1)/KS;
  const bool live=rbase<N;
  const uint8_t* tb=w.t+(size_t)(live?tile:0)*Q*Q8T_TILE;
  bool xv[NT];const float* xr[NT];
#pragma unroll
  for(int j=0;j<NT;j++){const int tok=j*8+g;xv[j]=tok<M;xr[j]=X+(size_t)(xv[j]?tok:0)*K+t*8;}
  if(live){
    for(int q0=c0>>2;q0<(c1>>2);q0+=QU){
      uint4 PA0[QU],PB0[QU],PA1[QU],PB1[QU];float SA[QU],SB[QU];
#pragma unroll
      for(int v=0;v<QU;v++){const uint8_t* p=tb+(size_t)(q0+v)*Q8T_TILE;
        PA0[v]=skinny_ld_stream(p+lane*16);PB0[v]=skinny_ld_stream(p+512+lane*16);
        PA1[v]=skinny_ld_stream(p+1024+lane*16);PB1[v]=skinny_ld_stream(p+1536+lane*16);
        const float* sc=reinterpret_cast<const float*>(p+2048);SA[v]=__ldg(sc+g);SB[v]=__ldg(sc+g+8);}
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;
      const uint4 pa0=PA0[v],pb0=PB0[v],pa1=PA1[v],pb1=PB1[v];
      const uint32_t M0[8]={pa0.x,pa0.y,pa0.z,pa0.w,pb0.x,pb0.y,pb0.z,pb0.w},M1[8]={pa1.x,pa1.y,pa1.z,pa1.w,pb1.x,pb1.y,pb1.z,pb1.w};
#pragma unroll
      for(int u=0;u<4;u++){
        const int cc=4*q+u;
        const uint4 a=q8_dec8(M0[2*u],M0[2*u+1],SA[v]),b=q8_dec8(M1[2*u],M1[2*u+1],SB[v]);
#pragma unroll
        for(int j=0;j<NT;j++){
          uint32_t xb[4]={0,0,0,0};
          if(xv[j]){
            if constexpr(XH){const uint4 h=__ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(X)+(size_t)(j*8+g)*K+t*8+(size_t)cc*32));
              xb[0]=h.x;xb[1]=h.y;xb[2]=h.z;xb[3]=h.w;}
            else{const float4 pp=__ldg(reinterpret_cast<const float4*>(xr[j]+(size_t)cc*32)),ss=__ldg(reinterpret_cast<const float4*>(xr[j]+(size_t)cc*32+4));
            xb[0]=f2h2(pp.x,pp.y);xb[1]=f2h2(pp.z,pp.w);xb[2]=f2h2(ss.x,ss.y);xb[3]=f2h2(ss.z,ss.w);}}
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
inline int skinny_q8t_launch(const float* X,const Q8T& w,void* Y,int M,int out,int ks,cudaStream_t s,bool xhalf=false){
  const int chunks=w.k>>5;
  if(M<1||M>32||w.n%16||w.k%128||(chunks/ks)*ks!=chunks||(chunks/ks)%4)return int(cudaErrorInvalidValue);
  auto grid=[&](int KS){int rt=8/KS;return (w.n+16*rt-1)/(16*rt);};
  static const int qu_env=[]{const char* e=std::getenv("GLM53_Q8_QU");return e?std::atoi(e):2;}();
  const bool qu2=qu_env==2&&(chunks/ks/4)%2==0;
#define L(KS,NT,O) do{if(xhalf){if(qu2)skinny_q8t<KS,NT,O,2,true><<<grid(KS),256,0,s>>>(X,w,Y,M);else skinny_q8t<KS,NT,O,1,true><<<grid(KS),256,0,s>>>(X,w,Y,M);}\
  else if(qu2)skinny_q8t<KS,NT,O,2><<<grid(KS),256,0,s>>>(X,w,Y,M);else skinny_q8t<KS,NT,O,1><<<grid(KS),256,0,s>>>(X,w,Y,M);}while(0)
#define LO(KS,NT) do{if(out==0)L(KS,NT,0);else if(out==1)L(KS,NT,1);else L(KS,NT,2);}while(0)
#define LK(NT) do{if(ks==2)LO(2,NT);else if(ks==4)LO(4,NT);else LO(8,NT);}while(0)
  if(M<=8)LK(1);else if(M<=16)LK(2);else LK(4);
#undef LK
#undef LO
#undef L
  return int(cudaGetLastError());
}
// Repack skinny_q8's layout (q [N,K] in m8 order, s [N,K/128]) into the tiled layout: one thread per (tile, quad, lane).
__global__ void q8t_pack(const uint8_t* __restrict__ q,const float* __restrict__ s,int N,int K,uint8_t* __restrict__ out){
  const int Q=K>>7;const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  const long long total=(long long)(N/16)*Q*32;if(id>=total)return;
  const int lane=(int)(id&31);const long long tq=id>>5;const int quad=(int)(tq%Q),tile=(int)(tq/Q);
  const int g=lane>>2,t=lane&3;const int r0=tile*16+g,r1=r0+8;
  uint8_t* o=out+(size_t)tq*Q8T_TILE+lane*16;
  const uint8_t* m0=q+((size_t)r0*(K>>6)*4+t)*16;const uint8_t* m1=q+((size_t)r1*(K>>6)*4+t)*16;
  *reinterpret_cast<uint4*>(o)=*reinterpret_cast<const uint4*>(m0+(size_t)(2*quad)*64);
  *reinterpret_cast<uint4*>(o+512)=*reinterpret_cast<const uint4*>(m0+(size_t)(2*quad+1)*64);
  *reinterpret_cast<uint4*>(o+1024)=*reinterpret_cast<const uint4*>(m1+(size_t)(2*quad)*64);
  *reinterpret_cast<uint4*>(o+1536)=*reinterpret_cast<const uint4*>(m1+(size_t)(2*quad+1)*64);
  if(lane<16){float* so=reinterpret_cast<float*>(out+(size_t)tq*Q8T_TILE+2048);so[lane]=s[(size_t)(tile*16+lane)*Q+quad];}
}

// ---- Encoder (load time). BF16 source W [N,K] on device.
// q8_scales: one warp per (row, group of 128 k). Clip ratio r in {1, 0.97, ..., 0.76} (GLM53_Q8_MSE=1, default) or
// r = 1 (absmax): scale = r*absmax/127, picking the r with the least squared error over the group.
__device__ __forceinline__ float bf16f(uint16_t v){return __uint_as_float((uint32_t)v<<16);}
__global__ void q8_scales(const uint16_t* __restrict__ W,int N,int K,int mse,float* __restrict__ S){
  const long long gid=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;const int lane=threadIdx.x&31;
  const long long groups=(long long)N*(K>>7);if(gid>=groups)return;
  const int r=(int)(gid/(K>>7)),q=(int)(gid%(K>>7));
  const uint16_t* w=W+(size_t)r*K+q*128;
  float v[4];float am=0.f;
#pragma unroll
  for(int i=0;i<4;i++){v[i]=bf16f(w[lane+32*i]);am=fmaxf(am,fabsf(v[i]));}
#pragma unroll
  for(int o=16;o;o>>=1)am=fmaxf(am,__shfl_xor_sync(0xffffffffu,am,o));
  if(am==0.f){if(lane==0)S[gid]=1.f;return;}
  float best_s=am/127.f,best_e=INFINITY;
  const int tries=mse?9:1;
  for(int c=0;c<tries;c++){
    const float s=(1.f-0.03f*c)*am/127.f;float e=0.f;
#pragma unroll
    for(int i=0;i<4;i++){const float qv=fminf(fmaxf(rintf(v[i]/s),-127.f),127.f);const float d=__half2float(__float2half_rn(qv*s))-v[i];e+=d*d;}
#pragma unroll
    for(int o=16;o;o>>=1)e+=__shfl_xor_sync(0xffffffffu,e,o);
    if(e<best_e){best_e=e;best_s=s;}
  }
  if(lane==0)S[gid]=best_s;
}
// q8_pack: one thread per (row, chunk, lane_t) -> 8 int8 bytes; accumulates the squared error and the squared
// source (Half_RN of the decoded value against the BF16 source) into err[0], err[1].
__global__ void q8_pack(const uint16_t* __restrict__ W,int N,int K,const float* __restrict__ S,uint8_t* __restrict__ Q,double* err){
  const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x,per=(long long)(K>>5)*4;
  float e2=0.f,r2=0.f;
  if(id<(long long)N*per){
    const int r=int(id/per),c=int((id%per)>>2),t=int(id&3);
    const uint16_t* w=W+(size_t)r*K+c*32+t*8;const float s=S[(size_t)r*(K>>7)+(c>>2)];
    uint32_t lo=0,hi=0;
#pragma unroll
    for(int i=0;i<8;i++){
      const float v=bf16f(w[i]);const float qv=fminf(fmaxf(rintf(v/s),-127.f),127.f);
      const uint32_t b=(uint32_t)(uint8_t)(int8_t)(int)qv;
      if(i<4)lo|=b<<(8*i);else hi|=b<<(8*(i-4));
      const float d=__half2float(__float2half_rn(qv*s))-v;e2+=d*d;r2+=v*v;
    }
    *reinterpret_cast<uint2*>(Q+(((size_t)r*(K>>6)+(c>>1))*4+t)*16+(c&1)*8)=make_uint2(lo,hi);
  }
#pragma unroll
  for(int o=16;o;o>>=1){e2+=__shfl_xor_sync(0xffffffffu,e2,o);r2+=__shfl_xor_sync(0xffffffffu,r2,o);}
  if((threadIdx.x&31)==0&&(e2>0.f||r2>0.f)){atomicAdd(err,(double)e2);atomicAdd(err+1,(double)r2);}
}
