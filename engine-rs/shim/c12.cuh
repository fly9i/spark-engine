// SPDX-License-Identifier: MIT
// Proposal 2 (lossless entropy coding of BF16 weights), format "C12": 12 bits per weight instead of 16.
//   m8  [N,K] bytes   sign<<7 | 7-bit BF16 mantissa
//   e4  [N,K/2] bytes 4-bit exponent index idx = eb[row] - e (0..14); 15 = escape
//   eb  [N]           per-row maximum BF16 exponent
//   escapes (CSR, ascending k per row): weights with idx >= 15 or a zero exponent (zero / BF16 subnormal),
//   stored as their exact BF16 bits. Measured escape rate on GLM-5.3 non-expert weights ~7.7e-5.
// Decoding rebuilds the exact BF16 value, widens it to FP32 and rounds to Half with the same RN conversion as
// the engine's resident BF16->Half copy, so every MMA operand equals the Half path's.
// skinny_c12 mirrors skinny_half (lane k-permutation, chunk order, split-K ranges, reduction order): for the
// same KS its output is bitwise identical to skinny_half for 1..8 rows, and each row's value does not depend on
// the row count (NT tiles reuse the same A fragments), so 1..32 rows are row-count invariant.
//
// Storage order (chosen so each lane issues 16-byte loads):
//   m8: ((row*(K/64) + pair)*4 + lane_t)*16 + (chunk&1)*8 + i   holds k = chunk*32 + lane_t*8 + i, pair = chunk/2
//   e4: ((row*(K/128) + quad)*4 + lane_t)*16 + (chunk&3)*4      uint32 of 8 nibbles (nibble i = weight i), quad = chunk/4
#pragma once
#include "fp8_skinny.cuh"

struct C12View {const uint8_t* m8; const uint8_t* e4; const uint8_t* eb; const int* esc_ptr; const int* esc_col; const uint16_t* esc_val; int n,k;};

__device__ __noinline__ float c12_escape(const C12View w,int row,int col){
  for(int i=w.esc_ptr[row],e=w.esc_ptr[row+1];i<e;i++) if(w.esc_col[i]==col) return __uint_as_float((uint32_t)w.esc_val[i]<<16);
  return __int_as_float(0x7fc00000);   // unreachable for a valid encoding: NaN makes a broken table visible
}
// 8 weights (k = kbase..kbase+7) of one row -> 4 half2 registers in skinny_half's operand order.
__device__ __forceinline__ uint4 c12_dec8(uint32_t mlo,uint32_t mhi,uint32_t e,uint32_t eb,const C12View& w,int row,int kbase){
  float f[8];
#pragma unroll
  for(int i=0;i<8;i++){
    const uint32_t m=((i<4?mlo:mhi)>>(8*(i&3)))&0xffu, idx=(e>>(4*i))&0xfu;
    f[i]=__uint_as_float(((m&0x80u)<<24)|((eb-idx)<<23)|((m&0x7fu)<<16));
  }
  if(__builtin_expect((e&(e>>1)&(e>>2)&(e>>3)&0x11111111u)!=0,0)){
#pragma unroll
    for(int i=0;i<8;i++) if(((e>>(4*i))&0xfu)==15u) f[i]=c12_escape(w,row,kbase+i);
  }
  return make_uint4(f2h2(f[0],f[1]),f2h2(f[2],f[3]),f2h2(f[4],f[5]),f2h2(f[6],f[7]));
}

// Float X [M,K] (Half RN in-kernel), FP32 accumulate. OUT: 0 FP32, 1 FP32 of the Half-rounded value, 2 Half.
// NT token tiles of 8 (M <= 8*NT). Split-K ranges must be multiples of 4 chunks (host-checked).
// XH (GLM53_C12_XHALF): X arrives as the Half [M,K] copy of the FP32 input (the same RN values f2h2 makes), read
// with one 16-byte load per lane and token tile instead of two float4 loads and four conversions (L0).
template<int KS,int NT,int OUT,int QU=1,bool XH=false>
__global__ void __launch_bounds__(256) skinny_c12(const float* __restrict__ X,const C12View w,void* __restrict__ Y,int M){
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
  const uint8_t* m0=w.m8+((size_t)r0*(K>>6)*4+t)*16;const uint8_t* m1=w.m8+((size_t)r1*(K>>6)*4+t)*16;
  const uint8_t* e0=w.e4+((size_t)r0*(K>>7)*4+t)*16;const uint8_t* e1=w.e4+((size_t)r1*(K>>7)*4+t)*16;
  const uint32_t b0=w.eb[r0],b1=w.eb[r1];
  bool xv[NT];const float* xr[NT];
#pragma unroll
  for(int j=0;j<NT;j++){const int tok=j*8+g;xv[j]=tok<M;xr[j]=X+(size_t)(xv[j]?tok:0)*K+t*8;}
  if(live){
    // QU quads (4 chunks each) are loaded per iteration (bytes in flight); chunks are still consumed in order, so the
    // MMA sequence and the result do not depend on QU. The host picks QU only when the split's quad count divides.
    for(int q0=c0>>2;q0<(c1>>2);q0+=QU){
      uint4 PA0[QU],PB0[QU],PA1[QU],PB1[QU],EA[QU],EB[QU];
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;
        PA0[v]=skinny_ld_stream(m0+(size_t)(2*q)*64);PB0[v]=skinny_ld_stream(m0+(size_t)(2*q+1)*64);
        PA1[v]=skinny_ld_stream(m1+(size_t)(2*q)*64);PB1[v]=skinny_ld_stream(m1+(size_t)(2*q+1)*64);
        EA[v]=skinny_ld_stream(e0+(size_t)q*64);EB[v]=skinny_ld_stream(e1+(size_t)q*64);}
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;
      const uint4 pa0=PA0[v],pb0=PB0[v],pa1=PA1[v],pb1=PB1[v],ea=EA[v],eb=EB[v];
      const uint32_t M0[8]={pa0.x,pa0.y,pa0.z,pa0.w,pb0.x,pb0.y,pb0.z,pb0.w},M1[8]={pa1.x,pa1.y,pa1.z,pa1.w,pb1.x,pb1.y,pb1.z,pb1.w};
      const uint32_t E0[4]={ea.x,ea.y,ea.z,ea.w},E1[4]={eb.x,eb.y,eb.z,eb.w};
#pragma unroll
      for(int u=0;u<4;u++){
        const int cc=4*q+u,kb=cc*32+t*8;
        const uint4 a=c12_dec8(M0[2*u],M0[2*u+1],E0[u],b0,w,r0,kb),b=c12_dec8(M1[2*u],M1[2*u+1],E1[u],b1,w,r1,kb);
#pragma unroll
        for(int j=0;j<NT;j++){
          uint32_t xb[4]={0,0,0,0};
          if constexpr(XH){if(xv[j]){const uint4 v=__ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(X)+(size_t)(j*8+g)*K+t*8+(size_t)cc*32));
            xb[0]=v.x;xb[1]=v.y;xb[2]=v.z;xb[3]=v.w;}}
          else if(xv[j]){const float4 p=__ldg(reinterpret_cast<const float4*>(xr[j]+(size_t)cc*32)),s=__ldg(reinterpret_cast<const float4*>(xr[j]+(size_t)cc*32+4));
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
inline bool c12_shape_ok(int n,int k,int ks){const int chunks=k>>5;return n%16==0&&k%128==0&&(chunks/ks)*ks==chunks&&(chunks/ks)%4==0;}
inline int skinny_c12_launch(const float* X,const C12View& w,void* Y,int M,int out,int ks,cudaStream_t s,bool xhalf=false){
  if(M<1||M>32||!c12_shape_ok(w.n,w.k,ks))return int(cudaErrorInvalidValue);
  auto grid=[&](int KS){int rt=8/KS;return (w.n+16*rt-1)/(16*rt);};
  // GLM53_C12_QU (read once): quads per iteration, 1 or 2; 2 only when every split holds an even quad count.
  static const int qu_env=[]{const char* e=std::getenv("GLM53_C12_QU");return e?std::atoi(e):1;}();
  const bool qu2=qu_env==2&&((w.k>>5)/ks/4)%2==0;
#define L(KS,NT,O) do{if(xhalf){if(qu2)skinny_c12<KS,NT,O,2,true><<<grid(KS),256,0,s>>>(X,w,Y,M);else skinny_c12<KS,NT,O,1,true><<<grid(KS),256,0,s>>>(X,w,Y,M);}\
  else if(qu2)skinny_c12<KS,NT,O,2><<<grid(KS),256,0,s>>>(X,w,Y,M);else skinny_c12<KS,NT,O,1><<<grid(KS),256,0,s>>>(X,w,Y,M);}while(0)
#define LO(KS,NT) do{if(out==0)L(KS,NT,0);else if(out==1)L(KS,NT,1);else L(KS,NT,2);}while(0)
#define LK(NT) do{if(ks==2)LO(2,NT);else if(ks==4)LO(4,NT);else LO(8,NT);}while(0)
  if(M<=8)LK(1);else if(M<=16)LK(2);else LK(4);
#undef LK
#undef LO
#undef L
  return int(cudaGetLastError());
}

// ---- Encoder (load time). BF16 source W [N,K] on device.
// 1) c12_stats: per row maximum exponent and escape count (one block of 256 per row).
__global__ void c12_stats(const uint16_t* __restrict__ W,int K,uint8_t* eb,int* esc_cnt){
  const int r=blockIdx.x;const uint16_t* w=W+(size_t)r*K;
  __shared__ int mx,cnt;if(threadIdx.x==0){mx=0;cnt=0;}__syncthreads();
  int m=0;for(int k=threadIdx.x;k<K;k+=blockDim.x)m=max(m,(w[k]>>7)&0xff);
  atomicMax(&mx,m);__syncthreads();
  const int e_max=mx;int n=0;
  for(int k=threadIdx.x;k<K;k+=blockDim.x){const int e=(w[k]>>7)&0xff;n+=(e==0||e_max-e>=15);}
  atomicAdd(&cnt,n);__syncthreads();
  if(threadIdx.x==0){eb[r]=(uint8_t)e_max;esc_cnt[r]=cnt;}
}
// 2) c12_pack: one thread per (row, chunk, lane_t) -> 8 bytes of m8, one uint32 of e4.
__global__ void c12_pack(const uint16_t* __restrict__ W,int N,int K,const uint8_t* eb,uint8_t* m8,uint8_t* e4){
  const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x,per=(long long)(K>>5)*4;
  if(id>=(long long)N*per)return;
  const int r=int(id/per),c=int((id%per)>>2),t=int(id&3);
  const uint16_t* w=W+(size_t)r*K+c*32+t*8;const int e_max=eb[r];
  uint32_t lo=0,hi=0,e=0;
#pragma unroll
  for(int i=0;i<8;i++){
    const uint32_t v=w[i],ex=(v>>7)&0xff,idx=(ex==0||e_max-(int)ex>=15)?15u:(uint32_t)(e_max-(int)ex);
    const uint32_t b=((v>>8)&0x80u)|(v&0x7fu);
    if(i<4)lo|=b<<(8*i);else hi|=b<<(8*(i-4));
    e|=idx<<(4*i);
  }
  uint2* mp=reinterpret_cast<uint2*>(m8+(((size_t)r*(K>>6)+(c>>1))*4+t)*16+(c&1)*8);*mp=make_uint2(lo,hi);
  *reinterpret_cast<uint32_t*>(e4+(((size_t)r*(K>>7)+(c>>2))*4+t)*16+(c&3)*4)=e;
}
// 3) c12_escapes: one warp per row, ascending k (ballot compaction), after the host exclusive scan of esc_cnt.
__global__ void c12_escapes(const uint16_t* __restrict__ W,int N,int K,const uint8_t* eb,const int* esc_ptr,int* esc_col,uint16_t* esc_val){
  const int r=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5),lane=threadIdx.x&31;if(r>=N)return;
  const uint16_t* w=W+(size_t)r*K;const int e_max=eb[r];int base=esc_ptr[r];
  for(int k0=0;k0<K;k0+=32){
    const uint16_t v=w[k0+lane];const int ex=(v>>7)&0xff;const bool esc=ex==0||e_max-ex>=15;
    const unsigned mask=__ballot_sync(0xffffffffu,esc);
    if(esc){const int p=base+__popc(mask&((1u<<lane)-1));esc_col[p]=k0+lane;esc_val[p]=v;}
    base+=__popc(mask);
  }
}
