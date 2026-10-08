// SPDX-License-Identifier: MIT
// W04 candidate (default off, GLM53_FP8_SKINNY=1): Y[M,N] = scale[n] * (half(X[M,K]) . e4m3(W[N,K])), FP32 accumulate, M<=16.
// Same operand semantics as fp8_small_*: X Float->Half RN, e4m3->Half exact, mma f16/f32,
// then *scale, optional Half rounding. Summation order differs (L1).
// Each lane loads 16 contiguous FP8 bytes per weight row per k64 chunk; the identical K
// permutation is applied to X, so MMA fragments are built in registers (no smem staging).
#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
__device__ __forceinline__ void skinny_mma16816(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
// Programmatic dependent launch (GLM53_PDL=1): weight loads (independent of the previous kernel) are issued before
// griddepcontrol.wait; activations are read and outputs written only after it. No-ops when not PDL-launched.
__device__ __forceinline__ void pdl_wait(){asm volatile("griddepcontrol.wait;" ::: "memory");}
__device__ __forceinline__ void pdl_trigger(){asm volatile("griddepcontrol.launch_dependents;" :::);}
__device__ __forceinline__ uint4 skinny_ld_stream(const void* p){
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p)); return v;
}

// Activation loads: plain coherent ld.global in volatile asm. __ldg (ld.global.nc, invariant) may be hoisted above
// griddepcontrol.wait, reading X before the producing kernel finished under PDL.
__device__ __forceinline__ uint4 skinny_ld_x(const void* p){
  uint4 v; asm volatile("ld.global.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p)); return v;
}
__device__ __forceinline__ uint32_t cvt2(uint16_t two){ __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two,__NV_E4M3); return *reinterpret_cast<uint32_t*>(&h);}
__device__ __forceinline__ uint32_t cvt2_bf(uint16_t two){ __half2_raw r=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two,__NV_E4M3); float2 f=__half22float2(*reinterpret_cast<half2*>(&r)); __nv_bfloat162 b=__floats2bfloat162_rn(f.x,f.y); return *reinterpret_cast<uint32_t*>(&b);}
__device__ __forceinline__ void skinny_mma16816_bf(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint32_t f2h2(float a,float b){ half2 h=__floats2half2_rn(a,b); return *reinterpret_cast<uint32_t*>(&h);}

// IN: 0 X Float (Half RN in-kernel), 1 X Half, 2 X BF16 with BF16 MMA (weights e4m3->BF16 exact).
template<int KS,int NT,int U,int IN>
__global__ void __launch_bounds__(256) skinny_fp8(const void* __restrict__ Xv,const uint8_t* __restrict__ W,const float* __restrict__ scale,float* __restrict__ Y,int M,int N,int K,bool round){
  constexpr int RT=8/KS;
  __shared__ float red[KS][RT][NT][4][32];
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS;
  const int rbase=(blockIdx.x*RT+rt)*16;
  float c[NT][4]={};
  const int chunks=K>>6;
  const int c0=chunks*ks/KS,c1=chunks*(ks+1)/KS;
  const bool live=rbase<N;
  const int r0=min(rbase+g,N-1),r1=min(rbase+g+8,N-1);
  const uint8_t* w0=W+(size_t)r0*K+t*16; const uint8_t* w1=W+(size_t)r1*K+t*16;
  bool xv[NT]; size_t xo[NT];
#pragma unroll
  for(int j=0;j<NT;j++){int tok=j*8+g;xv[j]=tok<M;xo[j]=(size_t)(xv[j]?tok:0)*K+t*16;}
  if(live){
    for(int cc=c0;cc<c1;cc+=U){
      uint4 a[U],b[U];
#pragma unroll
      for(int u=0;u<U;u++){ const int ci=min(cc+u,c1-1);a[u]=skinny_ld_stream(w0+(size_t)ci*64);b[u]=skinny_ld_stream(w1+(size_t)ci*64); }   // clamped, unpredicated: the tail duplicates are never consumed
      if(cc==c0) pdl_wait();   // first weight batch in flight before depending on the previous kernel
#pragma unroll
      for(int u=0;u<U;u++){ if(cc+u>=c1) break;
#pragma unroll
        for(int j=0;j<NT;j++){
          uint32_t xh[8];
          if(!xv[j]){
#pragma unroll
            for(int q=0;q<8;q++)xh[q]=0;
          }else if constexpr(IN==0){
            const float4* xp=reinterpret_cast<const float4*>(reinterpret_cast<const float*>(Xv)+xo[j]+(size_t)(cc+u)*64);
#pragma unroll
            for(int q=0;q<4;q++){uint4 u4=skinny_ld_x(xp+q);float4 f=make_float4(__uint_as_float(u4.x),__uint_as_float(u4.y),__uint_as_float(u4.z),__uint_as_float(u4.w));xh[2*q]=f2h2(f.x,f.y);xh[2*q+1]=f2h2(f.z,f.w);}
          }else{
            const uint4* xp=reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(Xv)+xo[j]+(size_t)(cc+u)*64);
            uint4 p=skinny_ld_x(xp),q2=skinny_ld_x(xp+1);xh[0]=p.x;xh[1]=p.y;xh[2]=p.z;xh[3]=p.w;xh[4]=q2.x;xh[5]=q2.y;xh[6]=q2.z;xh[7]=q2.w;
          }
          const uint32_t aw[4]={a[u].x,a[u].y,a[u].z,a[u].w},bw[4]={b[u].x,b[u].y,b[u].z,b[u].w};
#pragma unroll
          for(int s=0;s<4;s++){
            // bytes 4s..4s+3 of this lane's 16: pairs (4s,4s+1)->logical {2t,2t+1}, (4s+2,4s+3)->{2t+8,2t+9}
            if constexpr(IN==2){
              uint32_t A0=cvt2_bf(aw[s]&0xffff),A2=cvt2_bf(aw[s]>>16),A1=cvt2_bf(bw[s]&0xffff),A3=cvt2_bf(bw[s]>>16);
              skinny_mma16816_bf(c[j],A0,A1,A2,A3,xh[2*s],xh[2*s+1]);
            }else{
              uint32_t A0=cvt2(aw[s]&0xffff),A2=cvt2(aw[s]>>16),A1=cvt2(bw[s]&0xffff),A3=cvt2(bw[s]>>16);
              skinny_mma16816(c[j],A0,A1,A2,A3,xh[2*s],xh[2*s+1]);
            }
          }
        }
      }
    }
  }
  pdl_wait();      // every block (including ones without K work) waits before writing outputs
  pdl_trigger();   // main loop done: dependents may start their own weight prefetch
#pragma unroll
  for(int j=0;j<NT;j++)
#pragma unroll
    for(int i=0;i<4;i++) red[ks][rt][j][i][lane]=c[j][i];
  __syncthreads();
  if(ks==0&&live){
#pragma unroll
    for(int j=0;j<NT;j++){
      float s[4];
#pragma unroll
      for(int i=0;i<4;i++){s[i]=red[0][rt][j][i][lane];
#pragma unroll
        for(int q=1;q<KS;q++)s[i]+=red[q][rt][j][i][lane];}
#pragma unroll
      for(int i=0;i<4;i++){
        int row=rbase+g+(i>>1)*8,tok=j*8+2*t+(i&1);
        if(row<N&&tok<M){float v=s[i]*scale[row];if(round)v=__half2float(__float2half_rn(v));Y[(size_t)tok*N+row]=v;}
      }
    }
  }
}
inline bool skinny_pdl_enabled(){const char* e=std::getenv("GLM53_PDL");return e&&e[0]=='1';}
template<typename... A>
inline void skinny_launch_pdl(void(*kernel)(A...),int grid,cudaStream_t s,A... args){
  if(!skinny_pdl_enabled()){kernel<<<grid,256,0,s>>>(args...);return;}
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=0;cfg.stream=s;
  cudaLaunchAttribute at[1];at[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;at[0].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=at;cfg.numAttrs=1;cudaLaunchKernelEx(&cfg,kernel,args...);
}
template<int U,int IN>
inline void skinny_fp8_launch(int ks,const void* X,const uint8_t* W,const float* sc,float* Y,int M,int N,int K,bool round,cudaStream_t s){
  auto grid=[&](int KS){int rt=8/KS;return (N+16*rt-1)/(16*rt);};
#define L(KS,NT) skinny_launch_pdl(skinny_fp8<KS,NT,U,IN>,grid(KS),s,X,W,sc,Y,M,N,K,round)
  if(M<=8){ if(ks==1)L(1,1);else if(ks==2)L(2,1);else if(ks==4)L(4,1);else L(8,1);}
  else if(M<=16){ if(ks==1)L(1,2);else if(ks==2)L(2,2);else if(ks==4)L(4,2);else L(8,2);}
  else    { if(ks==1)L(1,4);else if(ks==2)L(2,4);else if(ks==4)L(4,4);else L(8,4);}   // C2: 17..32 rows (batched drafter)
#undef L
}

// ---- W02b (GLM53_HALF_SKINNY=1): Half weights [N,K], Float X [M,K] (Half RN in-kernel), FP32 accumulate.
// OUT: 0 FP32, 1 FP32 of the Half-rounded value (mm16 boundary), 2 Half.
template<int KS,int U,int OUT>
__global__ void __launch_bounds__(256) skinny_half(const float* __restrict__ X,const __half* __restrict__ W,void* __restrict__ Y,int M,int N,int K){
  constexpr int RT=8/KS;
  __shared__ float red[KS][RT][4][32];
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS,rbase=(blockIdx.x*RT+rt)*16;
  float c[4]={0.f,0.f,0.f,0.f};
  const int chunks=K>>5,c0=chunks*ks/KS,c1=chunks*(ks+1)/KS;
  const bool live=rbase<N;const int r0=min(rbase+g,N-1),r1=min(rbase+g+8,N-1);
  const __half* w0=W+(size_t)r0*K+t*8;const __half* w1=W+(size_t)r1*K+t*8;
  const bool xv=g<M;const float* xr=X+(size_t)(xv?g:0)*K+t*8;
  if(live){
    for(int cc=c0;cc<c1;cc+=U){
      uint4 a[U],b[U];
#pragma unroll
      for(int u=0;u<U;u++)if(cc+u<c1){a[u]=skinny_ld_stream(w0+(size_t)(cc+u)*32);b[u]=skinny_ld_stream(w1+(size_t)(cc+u)*32);}
#pragma unroll
      for(int u=0;u<U;u++){ if(cc+u>=c1)break;
        uint32_t xb[4]={0,0,0,0};
        if(xv){const float4 p=__ldg(reinterpret_cast<const float4*>(xr+(size_t)(cc+u)*32)),q=__ldg(reinterpret_cast<const float4*>(xr+(size_t)(cc+u)*32+4));
          xb[0]=f2h2(p.x,p.y);xb[1]=f2h2(p.z,p.w);xb[2]=f2h2(q.x,q.y);xb[3]=f2h2(q.z,q.w);}
        skinny_mma16816(c,a[u].x,b[u].x,a[u].y,b[u].y,xb[0],xb[1]);
        skinny_mma16816(c,a[u].z,b[u].z,a[u].w,b[u].w,xb[2],xb[3]);
      }
    }
  }
#pragma unroll
  for(int i=0;i<4;i++)red[ks][rt][i][lane]=c[i];
  __syncthreads();
  if(ks==0&&live){
#pragma unroll
    for(int i=0;i<4;i++){float s=red[0][rt][i][lane];
#pragma unroll
      for(int q=1;q<KS;q++)s+=red[q][rt][i][lane];
      const int row=rbase+g+(i>>1)*8,tok=2*t+(i&1);
      if(row<N&&tok<M){
        if constexpr(OUT==2)reinterpret_cast<__half*>(Y)[(size_t)tok*N+row]=__float2half_rn(s);
        else reinterpret_cast<float*>(Y)[(size_t)tok*N+row]=OUT==1?__half2float(__float2half_rn(s)):s;}}
  }
}
inline int skinny_half_launch(const float* X,const void* W,void* Y,int M,int N,int K,int out,int ks,cudaStream_t s){
  if(M<1||M>8||N%16||K%32)return int(cudaErrorInvalidValue);
  auto grid=[&](int KS){int rt=8/KS;return (N+16*rt-1)/(16*rt);};
#define L(KS,O) skinny_half<KS,4,O><<<grid(KS),256,0,s>>>(X,(const __half*)W,Y,M,N,K)
#define LO(KS) do{if(out==0)L(KS,0);else if(out==1)L(KS,1);else L(KS,2);}while(0)
  if(ks==2)LO(2);else if(ks==4)LO(4);else LO(8);
#undef LO
#undef L
  return int(cudaGetLastError());
}
