// SPDX-License-Identifier: MIT
// SPX1 non-expert weight format (src/spx.rs): b-bit codes (b in {4,5,6,8}) with one FP16 scale per 64 inputs, in a
// randomly signed block-Hadamard rotated input space. W_rot[o,i] = (c[o,i] - (2^b-1)/2) * s[o,i/64];
// y = W_rot . x_rot, x_rot = blockdiag(H_B)(sign (.) x) with H_B the orthonormal Sylvester Hadamard.
//
// Kernels:
//   spx_rot      x [M,K] (FP32 or Half) -> x_rot Half [M,K] (FP32 FWHT, one CTA per (row, block)); the same kernel with
//                the sign applied after the transform gives W = W_rot R^T for weights only used in reconstructed form.
//   spx_gemv     decode-sized GEMM (1..32 rows) over the tiled codes: Half MMA on the exact odd integers 2c - (2^b-1) and
//                x_rot, FP32 accumulate per 64-k group, scaled by s/2 in FP32 (weights applied exactly).
//   spx_dequant  tiled codes -> Half_RN(W_rot) [N,K] (prefill: a cuBLAS Half GEMM follows) or exact FP32 (reconstruction).
//   spx_pack     load-time repack of the on-disk row bitstream (any code offset k0 into the row: TP row split) into tiles.
//
// Tiled layout (like q8.cuh's Q8T): tiles ordered (16-row tile, quad of 128 k), TB(b) = 256*b + 64 bytes each.
// Lane l (g = l>>2, t = l&3) of a warp owns rows g and g+8 and, in chunk u (32 k) of the quad, k = u*32 + t*8 + i, i<8:
// the skinny_q8t operand order (same lane k-permutation for the A and B fragments of mma.m16n8k16).
// Per lane 8 groups w = u*2 + r (r 0: row g, 1: row g+8) of 8 codes; code i of a group is in half2 pair p = i>>1, lane
// half h = i&1 of the MMA operand. Bits are pre-placed so that one shift + LOP3 per half2 builds Half(1024 + 2c):
//   nibble word of group w (u32): low nibble of code i at bit 16h + 4p;   [0,1024): uint4 j of lane l at j*512 + l*16
//     holds the words of groups 4j..4j+3.
//   high words (b>4, (b-4) bits per code, G = 4/(b-4) groups per u32): bit j of code i of group w at 16h + 4p +
//     (b-4)*(w%G) + j in word w/G; [1024, 256b): b=5 uint2 at 1024 + l*8; b=6 uint4 at 1024 + l*16; b=8 uint4 j at
//     1024 + j*512 + l*16.
//   [256b,256b+64)  scales: half2 (group 0: k 0..63, group 1: k 64..127) of tile row r at 256b + 4r.
// Decode: Half(1024 + 2c) - Half(1024 + 2^b - 1) = 2c - (2^b - 1), an exact odd integer (|.| <= 255) as the MMA A operand;
// each 64-k group accumulates separately and is scaled by s/2 in FP32 (spx_gemv). The weights are thus applied exactly
// (no Half rounding of (c - mid) * s); the prefill dequant writes Half_RN((c - mid) * s).
#include "fp8_skinny.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__host__ __device__ constexpr int spx_tb(int b){return 256*b+64;}
template<int B> __host__ __device__ constexpr int spx_gpw(){return B==5?4:B==6?2:1;}   // groups per high word

// Shift right by s (left when s < 0); s is a constant after unrolling.
__device__ __forceinline__ uint32_t spx_shr(uint32_t x,int s){return s>=0?(x>>s):(x<<(-s));}
// Half2 (bit pattern) of Half(1024 + 2c) for pair p of a group with nibble word n, high word h, group slot q in h.
template<int B>
__device__ __forceinline__ uint32_t spx_pair(uint32_t n,uint32_t h,int p,int q){
  uint32_t v=(spx_shr(n,4*p-1)&0x001E001Eu)|0x64006400u;
  if constexpr(B>4){constexpr uint32_t m=((1u<<(B-4))-1u)<<5;v|=spx_shr(h,4*p+(B-4)*q-5)&(m|(m<<16));}
  return v;
}
__device__ __forceinline__ uint32_t spx_hsub(uint32_t a,uint32_t b){
  __half2 r=__hsub2(*reinterpret_cast<__half2*>(&a),*reinterpret_cast<__half2*>(&b));return *reinterpret_cast<uint32_t*>(&r);
}
// Four half2 of group (nibble n, high h, slot q): odd integers 2c - (2^b-1), codes 0..7 in mma operand order.
template<int B>
__device__ __forceinline__ uint4 spx_int8(uint32_t n,uint32_t h,int q){
  constexpr uint32_t bias=0x6400u+(uint32_t)((1<<B)-1);   // Half(1024 + 2^b - 1): exact (< 2048)
  constexpr uint32_t bb=bias|(bias<<16);
  return make_uint4(spx_hsub(spx_pair<B>(n,h,0,q),bb),spx_hsub(spx_pair<B>(n,h,1,q),bb),spx_hsub(spx_pair<B>(n,h,2,q),bb),spx_hsub(spx_pair<B>(n,h,3,q),bb));
}
// Code i (0..7) of a group, for the dequant kernel.
template<int B>
__device__ __forceinline__ uint32_t spx_code(uint32_t n,uint32_t h,int q,int i){
  const int p=i>>1,hh=i&1;uint32_t c=(n>>(16*hh+4*p))&15u;
  if constexpr(B>4)c|=((h>>(16*hh+4*p+(B-4)*q))&((1u<<(B-4))-1u))<<4;
  return c;
}

// One quad's lane data: 8 nibble words, up to 8 high words, scales (half2 per row) of rows g and g+8.
template<int B> struct SpxLane {uint4 n0,n1,h0,h1;uint32_t sa,sb;};
template<int B>
__device__ __forceinline__ SpxLane<B> spx_load(const uint8_t* p,int lane,int g){
  SpxLane<B> d;
  d.n0=skinny_ld_stream(p+lane*16);d.n1=skinny_ld_stream(p+512+lane*16);
  if constexpr(B==5){uint2 h;asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1},[%2];":"=r"(h.x),"=r"(h.y):"l"(p+1024+lane*8));d.h0=make_uint4(h.x,h.y,0,0);d.h1=make_uint4(0,0,0,0);}
  else if constexpr(B==6){d.h0=skinny_ld_stream(p+1024+lane*16);d.h1=make_uint4(0,0,0,0);}
  else if constexpr(B==8){d.h0=skinny_ld_stream(p+1024+lane*16);d.h1=skinny_ld_stream(p+1536+lane*16);}
  else {d.h0=d.h1=make_uint4(0,0,0,0);}
  d.sa=__ldg(reinterpret_cast<const uint32_t*>(p+256*B)+g);d.sb=__ldg(reinterpret_cast<const uint32_t*>(p+256*B)+g+8);
  return d;
}
template<int B> __device__ __forceinline__ uint32_t spx_nib(const SpxLane<B>& d,int w){
  const uint32_t a[8]={d.n0.x,d.n0.y,d.n0.z,d.n0.w,d.n1.x,d.n1.y,d.n1.z,d.n1.w};return a[w];
}
template<int B> __device__ __forceinline__ uint32_t spx_hiw(const SpxLane<B>& d,int w){   // high word holding group w
  const uint32_t a[8]={d.h0.x,d.h0.y,d.h0.z,d.h0.w,d.h1.x,d.h1.y,d.h1.z,d.h1.w};return a[w/spx_gpw<B>()];
}
__device__ __forceinline__ float2 spx_half_scales(uint32_t s){
  __half2 h=*reinterpret_cast<__half2*>(&s);float2 f=__half22float2(h);return make_float2(0.5f*f.x,0.5f*f.y);
}

// X: Half [M,K] (the rotated activations), T tiles, Y [M,N]: OUT 0 FP32, 1 FP32 of the Half-rounded value, 2 Half.
template<int B,int KS,int NT,int OUT>
__global__ void __launch_bounds__(256) spx_gemv(const __half* __restrict__ X,const uint8_t* __restrict__ T,int N,int K,void* __restrict__ Y,int M){
  constexpr int RT=8/KS;constexpr int TB=spx_tb(B);constexpr int G=spx_gpw<B>();
  __shared__ float red[KS][RT][NT][4][32];
  const int Q=K>>7;
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS,tile=blockIdx.x*RT+rt,rbase=tile*16;
  float c[NT][4];
#pragma unroll
  for(int j=0;j<NT;j++){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
  const int q0=Q*ks/KS,q1=Q*(ks+1)/KS;
  const bool live=rbase<N;
  const uint8_t* tb=T+(size_t)(live?tile:0)*Q*TB;
  bool xv[NT];const __half* xr[NT];
#pragma unroll
  for(int j=0;j<NT;j++){const int tok=j*8+g;xv[j]=tok<M;xr[j]=X+(size_t)(xv[j]?tok:0)*K+t*8;}
  if(live&&q0<q1){
    SpxLane<B> cur=spx_load<B>(tb+(size_t)q0*TB,lane,g);
    for(int q=q0;q<q1;q++){
      SpxLane<B> nxt=cur;
      if(q+1<q1)nxt=spx_load<B>(tb+(size_t)(q+1)*TB,lane,g);
      const float2 fa=spx_half_scales(cur.sa),fb=spx_half_scales(cur.sb);
#pragma unroll
      for(int gr=0;gr<2;gr++){            // 64-k scale group: chunks 2gr, 2gr+1
        float a4[NT][4];
#pragma unroll
        for(int j=0;j<NT;j++){a4[j][0]=a4[j][1]=a4[j][2]=a4[j][3]=0.f;}
#pragma unroll
        for(int uu=0;uu<2;uu++){
          const int u=2*gr+uu,cc=4*q+u,wa=2*u,wb=2*u+1;
          const uint4 a=spx_int8<B>(spx_nib<B>(cur,wa),spx_hiw<B>(cur,wa),wa%G);
          const uint4 b=spx_int8<B>(spx_nib<B>(cur,wb),spx_hiw<B>(cur,wb),wb%G);
#pragma unroll
          for(int j=0;j<NT;j++){
            uint4 h=make_uint4(0,0,0,0);
            if(xv[j])h=skinny_ld_x(xr[j]+(size_t)cc*32);
            skinny_mma16816(a4[j],a.x,b.x,a.y,b.y,h.x,h.y);
            skinny_mma16816(a4[j],a.z,b.z,a.w,b.w,h.z,h.w);
          }
        }
        const float sA=gr?fa.y:fa.x,sB=gr?fb.y:fb.x;
#pragma unroll
        for(int j=0;j<NT;j++){c[j][0]=__fmaf_rn(a4[j][0],sA,c[j][0]);c[j][1]=__fmaf_rn(a4[j][1],sA,c[j][1]);
          c[j][2]=__fmaf_rn(a4[j][2],sB,c[j][2]);c[j][3]=__fmaf_rn(a4[j][3],sB,c[j][3]);}
      }
      cur=nxt;
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

template<int B,int KS,int NT>
static void spx_gemv_out(const __half* X,const uint8_t* T,int N,int K,void* Y,int M,int out,cudaStream_t s){
  const int grid=(N/16+(8/KS)-1)/(8/KS);
  if(out==0)spx_gemv<B,KS,NT,0><<<grid,256,0,s>>>(X,T,N,K,Y,M);
  else if(out==1)spx_gemv<B,KS,NT,1><<<grid,256,0,s>>>(X,T,N,K,Y,M);
  else spx_gemv<B,KS,NT,2><<<grid,256,0,s>>>(X,T,N,K,Y,M);
}
template<int B,int KS>
static void spx_gemv_nt(const __half* X,const uint8_t* T,int N,int K,void* Y,int M,int out,cudaStream_t s){
  if(M<=8)spx_gemv_out<B,KS,1>(X,T,N,K,Y,M,out,s);else if(M<=16)spx_gemv_out<B,KS,2>(X,T,N,K,Y,M,out,s);else spx_gemv_out<B,KS,4>(X,T,N,K,Y,M,out,s);
}
template<int B>
static void spx_gemv_ks(const __half* X,const uint8_t* T,int N,int K,void* Y,int M,int out,int ks,cudaStream_t s){
  if(ks==1)spx_gemv_nt<B,1>(X,T,N,K,Y,M,out,s);else if(ks==2)spx_gemv_nt<B,2>(X,T,N,K,Y,M,out,s);
  else if(ks==4)spx_gemv_nt<B,4>(X,T,N,K,Y,M,out,s);else spx_gemv_nt<B,8>(X,T,N,K,Y,M,out,s);
}
extern "C" int glm53_spx_gemv_cuda(const void* x,const void* t,int n,int k,int bits,void* y,int m,int out,int ks,cudaStream_t s){
  if(m<1||m>32||n%16||k%128||!(ks==1||ks==2||ks==4||ks==8)||out<0||out>2)return int(cudaErrorInvalidValue);
  const __half* X=(const __half*)x;const uint8_t* T=(const uint8_t*)t;
  switch(bits){case 4:spx_gemv_ks<4>(X,T,n,k,y,m,out,ks,s);break;case 5:spx_gemv_ks<5>(X,T,n,k,y,m,out,ks,s);break;
    case 6:spx_gemv_ks<6>(X,T,n,k,y,m,out,ks,s);break;case 8:spx_gemv_ks<8>(X,T,n,k,y,m,out,ks,s);break;default:return int(cudaErrorInvalidValue);}
  return int(cudaGetLastError());
}

// ---- Dequantization (one thread per (tile, quad, lane)): W [N, K] (row stride ldw elements), Half_RN((c-mid)*s) or FP32 (exact).
template<int B,bool F32>
__global__ void spx_dequant(const uint8_t* __restrict__ T,int N,int K,void* __restrict__ W,long long ldw){
  const int Q=K>>7;const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=(long long)(N/16)*Q*32)return;
  const int lane=(int)(id&31);const long long tq=id>>5;const int quad=(int)(tq%Q),tile=(int)(tq/Q);const int g=lane>>2,t=lane&3;
  const SpxLane<B> d=spx_load<B>(T+(size_t)tq*spx_tb(B),lane,g);
  const float2 fa=spx_half_scales(d.sa),fb=spx_half_scales(d.sb);
#pragma unroll
  for(int w=0;w<8;w++){
    const int u=w>>1,r=w&1;const float hs=r?(u<2?fb.x:fb.y):(u<2?fa.x:fa.y);
    const size_t row=(size_t)tile*16+g+8*r,k=(size_t)quad*128+u*32+t*8;
    const uint32_t nib=spx_nib<B>(d,w),hi=spx_hiw<B>(d,w);const int qs=w%spx_gpw<B>();
    float f[8];
#pragma unroll
    for(int i=0;i<8;i++){const uint32_t c=spx_code<B>(nib,hi,qs,i);f[i]=(float)(int)(2*c-((1u<<B)-1u))*hs;}
    if constexpr(F32){float* o=reinterpret_cast<float*>(W)+row*ldw+k;
#pragma unroll
      for(int i=0;i<8;i++)o[i]=f[i];}
    else *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(W)+row*ldw+k)=make_uint4(f2h2(f[0],f[1]),f2h2(f[2],f[3]),f2h2(f[4],f[5]),f2h2(f[6],f[7]));
  }
}
extern "C" int glm53_spx_dequant_cuda(const void* t,int n,int k,int bits,void* w,long long ldw,int f32,cudaStream_t s){
  if(n%16||k%128||ldw<k||(!f32&&ldw%8))return int(cudaErrorInvalidValue);
  const long long th=(long long)(n/16)*(k/128)*32;const unsigned grid=(unsigned)((th+255)/256);const uint8_t* T=(const uint8_t*)t;
#define D(BB) do{if(f32)spx_dequant<BB,true><<<grid,256,0,s>>>(T,n,k,w,ldw);else spx_dequant<BB,false><<<grid,256,0,s>>>(T,n,k,w,ldw);}while(0)
  switch(bits){case 4:D(4);break;case 5:D(5);break;case 6:D(6);break;case 8:D(8);break;default:return int(cudaErrorInvalidValue);}
#undef D
  return int(cudaGetLastError());
}

// ---- Load-time repack. src: on-disk codes, row r of the tensor at src + r*rs (bytes), little-endian bitstream (code i at
// bits [i*b, (i+1)*b)); this tensor's columns are codes k0 .. k0+K of each row. sc: Half scales [N, K/64] contiguous.
template<int B>
__global__ void spx_pack(const uint8_t* __restrict__ src,long long rs,long long k0,const __half* __restrict__ sc,int N,int K,uint8_t* __restrict__ out){
  constexpr int G=spx_gpw<B>();
  const int Q=K>>7;const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=(long long)(N/16)*Q*32)return;
  const int lane=(int)(id&31);const long long tq=id>>5;const int quad=(int)(tq%Q),tile=(int)(tq/Q);const int g=lane>>2,t=lane&3;
  uint32_t nib[8],hi[8];
#pragma unroll
  for(int w=0;w<8;w++){nib[w]=0;hi[w]=0;}
#pragma unroll
  for(int w=0;w<8;w++){
    const int u=w>>1,r=w&1;const long long row=(long long)tile*16+g+8*r;const long long kb=k0+(long long)quad*128+u*32+t*8;
    const uint8_t* rp=src+row*rs;
#pragma unroll
    for(int i=0;i<8;i++){
      const long long bit=(kb+i)*B;const long long by=bit>>3;const int sh=(int)(bit&7);
      uint32_t v=rp[by];if(sh+B>8)v|=(uint32_t)rp[by+1]<<8;
      const uint32_t c=(v>>sh)&((1u<<B)-1u);
      const int pos=16*(i&1)+4*(i>>1);
      nib[w]|=(c&15u)<<pos;
      if constexpr(B>4)hi[w/G]|=(c>>4)<<(pos+(B-4)*(w%G));
    }
  }
  uint8_t* o=out+(size_t)tq*spx_tb(B);
  *reinterpret_cast<uint4*>(o+lane*16)=make_uint4(nib[0],nib[1],nib[2],nib[3]);
  *reinterpret_cast<uint4*>(o+512+lane*16)=make_uint4(nib[4],nib[5],nib[6],nib[7]);
  if constexpr(B==5){*reinterpret_cast<uint2*>(o+1024+lane*8)=make_uint2(hi[0],hi[1]);}
  else if constexpr(B==6){*reinterpret_cast<uint4*>(o+1024+lane*16)=make_uint4(hi[0],hi[1],hi[2],hi[3]);}
  else if constexpr(B==8){*reinterpret_cast<uint4*>(o+1024+lane*16)=make_uint4(hi[0],hi[1],hi[2],hi[3]);*reinterpret_cast<uint4*>(o+1536+lane*16)=make_uint4(hi[4],hi[5],hi[6],hi[7]);}
  if(lane<16){
    const long long row=(long long)tile*16+lane;const __half* sr=sc+row*(K/64)+quad*2;
    __half2 h=__halves2half2(sr[0],sr[1]);*reinterpret_cast<__half2*>(o+256*B+lane*4)=h;
  }
}
extern "C" int glm53_spx_pack_cuda(const void* src,long long rs,long long k0,const void* sc,int n,int k,int bits,void* out,cudaStream_t s){
  if(n%16||k%128||k0<0||rs*8<(k0+k)*(long long)bits)return int(cudaErrorInvalidValue);
  const long long th=(long long)(n/16)*(k/128)*32;const unsigned grid=(unsigned)((th+255)/256);
  const uint8_t* S=(const uint8_t*)src;const __half* C=(const __half*)sc;uint8_t* O=(uint8_t*)out;
  switch(bits){case 4:spx_pack<4><<<grid,256,0,s>>>(S,rs,k0,C,n,k,O);break;case 5:spx_pack<5><<<grid,256,0,s>>>(S,rs,k0,C,n,k,O);break;
    case 6:spx_pack<6><<<grid,256,0,s>>>(S,rs,k0,C,n,k,O);break;case 8:spx_pack<8><<<grid,256,0,s>>>(S,rs,k0,C,n,k,O);break;default:return int(cudaErrorInvalidValue);}
  return int(cudaGetLastError());
}

// ---- Rotation. One CTA (BK/8 threads, 8 elements each) per (row, block of BK). FP32 Walsh-Hadamard: strides 1..4 in
// registers, 8..128 by warp shuffles, larger through shared memory; scaled by 1/sqrt(BK) at the end.
// SIGN_FIRST: y = H (sign (.) x) (activations); else y = sign (.) (H x) (W_rot rows -> W = W_rot R^T).
__device__ __forceinline__ float spx_ldf(const float* p){return *p;}
__device__ __forceinline__ float spx_ldf(const __half* p){return __half2float(*p);}
__device__ __forceinline__ void spx_st(float* p,float v){*p=v;}
__device__ __forceinline__ void spx_st(__half* p,float v){*p=__float2half_rn(v);}
template<typename TI,typename TO,bool SIGN_FIRST>
__global__ void spx_rot(const TI* __restrict__ X,long long ldx,const int8_t* __restrict__ sign,int bk,TO* __restrict__ Y,long long ldy){
  extern __shared__ float sm[];
  const int nt=bk>>3,tid=threadIdx.x,m=blockIdx.y;const long long base=(long long)blockIdx.x*bk+tid*8;
  const TI* xr=X+m*ldx+base;
  float v[8];
#pragma unroll
  for(int i=0;i<8;i++){v[i]=spx_ldf(xr+i);if(SIGN_FIRST)v[i]*=(float)sign[base+i];}
#pragma unroll
  for(int h=1;h<8;h<<=1)
#pragma unroll
    for(int i=0;i<8;i++)if(!(i&h)){const float a=v[i],b=v[i+h];v[i]=a+b;v[i+h]=a-b;}
  const unsigned mask=nt>=32?0xffffffffu:((1u<<nt)-1u);
  for(int hs=1;hs<32&&hs<nt;hs<<=1){
#pragma unroll
    for(int i=0;i<8;i++){const float o=__shfl_xor_sync(mask,v[i],hs);v[i]=(tid&hs)?(o-v[i]):(v[i]+o);}
  }
  for(int hs=32;hs<nt;hs<<=1){
#pragma unroll
    for(int i=0;i<8;i++)sm[tid*8+i]=v[i];
    __syncthreads();
#pragma unroll
    for(int i=0;i<8;i++){const float o=sm[(tid^hs)*8+i];v[i]=(tid&hs)?(o-v[i]):(v[i]+o);}
    __syncthreads();
  }
  const float sc=rsqrtf((float)bk);
  TO* yr=Y+m*ldy+base;
#pragma unroll
  for(int i=0;i<8;i++){float r=v[i]*sc;if(!SIGN_FIRST)r*=(float)sign[base+i];spx_st(yr+i,r);}
}
// in_kind/out_kind: 0 FP32, 1 Half. sign_first: 1 activations, 0 weight reconstruction.
extern "C" int glm53_spx_rot_cuda(const void* x,long long ldx,int in_kind,const void* sign,int m,int k,int bk,void* y,long long ldy,int out_kind,int sign_first,cudaStream_t s){
  if(m<1||bk<8||bk>8192||(bk&(bk-1))||k%bk)return int(cudaErrorInvalidValue);
  const dim3 grid(k/bk,m);const int th=bk/8;const size_t sm=(size_t)bk*sizeof(float);
  const int8_t* S=(const int8_t*)sign;
#define R(TI,TO,SF) spx_rot<TI,TO,SF><<<grid,th,sm,s>>>((const TI*)x,ldx,S,bk,(TO*)y,ldy)
  if(sign_first){
    if(in_kind==0&&out_kind==1)R(float,__half,true);else if(in_kind==1&&out_kind==1)R(__half,__half,true);
    else if(in_kind==0&&out_kind==0)R(float,float,true);else R(__half,float,true);
  } else {
    if(in_kind==0&&out_kind==0)R(float,float,false);else if(in_kind==0&&out_kind==1)R(float,__half,false);
    else if(in_kind==1&&out_kind==0)R(__half,float,false);else R(__half,__half,false);
  }
#undef R
  return int(cudaGetLastError());
}

#include "spx_mm8.cuh"
