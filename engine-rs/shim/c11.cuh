// SPDX-License-Identifier: MIT
// Lossless format "C11" (GLM53_C11=1): C12 with the 4-bit exponent index replaced by a 3-bit code plus a rare escape.
// GLM-5.3's non-expert BF16 exponents, relative to the row maximum (C12's idx = eb - e), are 97% within 7 levels;
// a per-tensor window [base, base+6] gets codes 1..7, code 0 means "read C12's nibble from the escape stream".
// About 8 + 3 + 0.11 + offsets (0.125-0.25) ~= 11.3 bits per weight instead of 12. (A Huffman exponent code reaches
// 10.9 bits, but decoding it inside the GEMV was 10-50% slower than C12 on GB10: see bench/lossless/c11_bench.cu.)
//   m8   [N,K] bytes, exactly C12's (sign<<7 | 7-bit mantissa, C12 storage order)
//   e3   per (row, quad of 128 k, lane_t): 3 words = 32 codes; chunk u's 8 codes at bits 24u.., weight i at +3i
//        (k = (4*quad+u)*32 + lane_t*8 + i), at byte ((row*(K/128) + quad)*4 + lane_t)*12
//   eb   [N] row maximum exponent; escapes of C12 (nibble 15: zero, subnormal, idx >= 15) keep C12's CSR
//   esc  C12 nibbles of the code-0 weights, one nibble packet per (row, split, lane_t) in consumption order,
//        packet p = (row*KS + split)*4 + lane_t starting at nibble off[p] (KS fixed at encode time)
// skinny_c11 rebuilds C12's nibble word for each 8 weights and runs c12_dec8 unchanged: bitwise skinny_c12.
#pragma once
#include "c12.cuh"

struct C11View {const uint8_t* m8; const uint32_t* e3; const uint8_t* esc; const uint32_t* off; const uint8_t* eb;
                const int* esc_ptr; const int* esc_col; const uint16_t* esc_val; int n,k,ks,base;};

// Escape nibbles of one packet, in order: a 64-bit window hi:lo (nb valid bits, >= 32 between reads) plus the next
// stream word loaded ahead, so an escape never waits on memory.
struct C11Esc {
  uint32_t lo,hi,w1;int nb;const uint32_t* p;
  __device__ __forceinline__ void init(const uint8_t* esc,uint32_t nib){   // packets start on byte boundaries
    const uint8_t* b=esc+(nib>>1);p=reinterpret_cast<const uint32_t*>(reinterpret_cast<uintptr_t>(b)&~uintptr_t(3));
    const int sh=int(reinterpret_cast<uintptr_t>(b)&3)*8;
    const uint32_t a=__ldg(p),c=__ldg(p+1);w1=__ldg(p+2);p+=3;lo=__funnelshift_r(a,c,sh);hi=c>>sh;nb=64-sh;
  }
  __device__ __forceinline__ uint32_t next(){
    const uint32_t n=lo&15u;lo=__funnelshift_r(lo,hi,4);hi>>=4;nb-=4;
    if(nb<32){lo|=w1<<nb;hi=w1>>(32-nb);nb+=32;w1=__ldg(p++);}
    return n;
  }
};
// 8 codes (24 bits) -> C12 nibble word; code c in 1..7 -> idx base+c-1; code 0 -> the next escape nibble.
__device__ __forceinline__ uint32_t c11_nibbles(uint32_t f,int base,C11Esc& E){
  uint32_t e=0;
#pragma unroll
  for(int i=0;i<8;i++){const uint32_t c=(f>>(3*i))&7u;e|=((c+(uint32_t)base-1u)&15u)<<(4*i);}
#ifndef C11_ABL_NOESC
  uint32_t z=~(f|(f>>1)|(f>>2))&0x249249u;   // the LSB of every zero 3-bit field
  while(z){const int i=(__ffs(z)-1)/3;e=(e&~(15u<<(4*i)))|(E.next()<<(4*i));z&=z-1u;}
#endif
  return e;
}

// Float X [M,K] (Half RN in-kernel), FP32 accumulate. OUT: 0 FP32, 1 FP32 of the Half-rounded value, 2 Half.
template<int KS,int NT,int OUT,int QU=1>
__global__ void __launch_bounds__(256) skinny_c11(const float* __restrict__ X,const C11View w,void* __restrict__ Y,int M){
  constexpr int RT=8/KS;
  __shared__ float red[KS][RT][NT][4][32];
  const int N=w.n,K=w.k;
  const C12View v12{w.m8,nullptr,w.eb,w.esc_ptr,w.esc_col,w.esc_val,N,K};
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS,rbase=(blockIdx.x*RT+rt)*16;
  float c[NT][4];
#pragma unroll
  for(int j=0;j<NT;j++){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
  const int chunks=K>>5,c0=chunks*ks/KS,c1=chunks*(ks+1)/KS;
  const bool live=rbase<N;const int r0=min(rbase+g,N-1),r1=min(rbase+g+8,N-1);
  const uint8_t* m0=w.m8+((size_t)r0*(K>>6)*4+t)*16;const uint8_t* m1=w.m8+((size_t)r1*(K>>6)*4+t)*16;
  const uint32_t* x0=w.e3+((size_t)r0*(K>>7)*4+t)*3;const uint32_t* x1=w.e3+((size_t)r1*(K>>7)*4+t)*3;
  const uint32_t b0=w.eb[r0],b1=w.eb[r1];
  bool xv[NT];const float* xr[NT];
#pragma unroll
  for(int j=0;j<NT;j++){const int tok=j*8+g;xv[j]=tok<M;xr[j]=X+(size_t)(xv[j]?tok:0)*K+t*8;}
  if(live){
    C11Esc E0,E1;E0.init(w.esc,__ldg(w.off+((size_t)r0*KS+ks)*4+t));E1.init(w.esc,__ldg(w.off+((size_t)r1*KS+ks)*4+t));
    for(int q0=c0>>2;q0<(c1>>2);q0+=QU){
      uint4 PA0[QU],PB0[QU],PA1[QU],PB1[QU];uint32_t A[QU][3],B[QU][3];
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;
        PA0[v]=skinny_ld_stream(m0+(size_t)(2*q)*64);PB0[v]=skinny_ld_stream(m0+(size_t)(2*q+1)*64);
        PA1[v]=skinny_ld_stream(m1+(size_t)(2*q)*64);PB1[v]=skinny_ld_stream(m1+(size_t)(2*q+1)*64);
        const uint32_t* y0=x0+(size_t)q*12;const uint32_t* y1=x1+(size_t)q*12;
        A[v][0]=__ldg(y0);A[v][1]=__ldg(y0+1);A[v][2]=__ldg(y0+2);B[v][0]=__ldg(y1);B[v][1]=__ldg(y1+1);B[v][2]=__ldg(y1+2);}
#pragma unroll
      for(int v=0;v<QU;v++){const int q=q0+v;
      const uint4 pa0=PA0[v],pb0=PB0[v],pa1=PA1[v],pb1=PB1[v];
      const uint32_t M0[8]={pa0.x,pa0.y,pa0.z,pa0.w,pb0.x,pb0.y,pb0.z,pb0.w},M1[8]={pa1.x,pa1.y,pa1.z,pa1.w,pb1.x,pb1.y,pb1.z,pb1.w};
      const uint32_t F0[4]={A[v][0]&0xffffffu,__funnelshift_r(A[v][0],A[v][1],24)&0xffffffu,__funnelshift_r(A[v][1],A[v][2],16)&0xffffffu,A[v][2]>>8};
      const uint32_t F1[4]={B[v][0]&0xffffffu,__funnelshift_r(B[v][0],B[v][1],24)&0xffffffu,__funnelshift_r(B[v][1],B[v][2],16)&0xffffffu,B[v][2]>>8};
#pragma unroll
      for(int u=0;u<4;u++){
        const int cc=4*q+u,kb=cc*32+t*8;
        const uint32_t e0=c11_nibbles(F0[u],w.base,E0),e1=c11_nibbles(F1[u],w.base,E1);
        const uint4 a=c12_dec8(M0[2*u],M0[2*u+1],e0,b0,v12,r0,kb),b=c12_dec8(M1[2*u],M1[2*u+1],e1,b1,v12,r1,kb);
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
inline int skinny_c11_launch(const float* X,const C11View& w,void* Y,int M,int out,cudaStream_t s){
  const int ks=w.ks;
  if(M<1||M>32||!c12_shape_ok(w.n,w.k,ks))return int(cudaErrorInvalidValue);
  const int rt=8/ks,grid=(w.n+16*rt-1)/(16*rt);
  // GLM53_C11_QU (read once, default 2): quads per iteration; 2 only when every split holds an even quad count.
  static const int qu_env=[]{const char* e=std::getenv("GLM53_C11_QU");return e?std::atoi(e):2;}();
  const bool qu2=qu_env==2&&((w.k>>5)/ks/4)%2==0;
#define L(KS,NT,O) do{if(qu2)skinny_c11<KS,NT,O,2><<<grid,256,0,s>>>(X,w,Y,M);else skinny_c11<KS,NT,O,1><<<grid,256,0,s>>>(X,w,Y,M);}while(0)
#define LO(KS,NT) do{if(out==0)L(KS,NT,0);else if(out==1)L(KS,NT,1);else L(KS,NT,2);}while(0)
#define LK(NT) do{if(ks==2)LO(2,NT);else if(ks==4)LO(4,NT);else LO(8,NT);}while(0)
  if(M<=8)LK(1);else if(M<=16)LK(2);else LK(4);
#undef LK
#undef LO
#undef L
  return int(cudaGetLastError());
}

// ---- Encoder (load time). BF16 source W [N,K] on device, eb from c12_stats.
__device__ __forceinline__ uint32_t c11_idx(uint32_t v,int e_max){const int ex=(v>>7)&0xff;return (ex==0||e_max-ex>=15)?15u:(uint32_t)(e_max-ex);}
__device__ __forceinline__ uint32_t c11_code(uint32_t idx,int base){return (idx>=(uint32_t)base&&idx<=(uint32_t)base+6u)?idx-(uint32_t)base+1u:0u;}
// C12 nibble histogram (16 bins) over the whole tensor: the host picks base (the 7-level window with most weights).
__global__ void c11_hist(const uint16_t* __restrict__ W,int N,int K,const uint8_t* __restrict__ eb,unsigned long long* hist){
  __shared__ unsigned int h[16];if(threadIdx.x<16)h[threadIdx.x]=0;__syncthreads();
  for(int r=blockIdx.x;r<N;r+=gridDim.x){const int e_max=eb[r];for(int k=threadIdx.x;k<K;k+=blockDim.x)atomicAdd(&h[c11_idx(W[(size_t)r*K+k],e_max)],1u);}
  __syncthreads();if(threadIdx.x<16&&h[threadIdx.x])atomicAdd(hist+threadIdx.x,(unsigned long long)h[threadIdx.x]);
}
// e3: one thread per (row, quad, lane_t).
__global__ void c11_pack_e3(const uint16_t* __restrict__ W,int N,int K,const uint8_t* __restrict__ eb,int base,uint32_t* e3){
  const long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x,per=(long long)(K>>7)*4;if(id>=(long long)N*per)return;
  const int r=int(id/per),q=int((id%per)>>2),t=int(id&3);const int e_max=eb[r];const uint16_t* w=W+(size_t)r*K;
  uint32_t o[3]={0,0,0};
  for(int u=0;u<4;u++)for(int i=0;i<8;i++){const int bit=24*u+3*i;const uint32_t c=c11_code(c11_idx(w[(4*q+u)*32+t*8+i],e_max),base);
    o[bit>>5]|=c<<(bit&31);if((bit&31)>29)o[(bit>>5)+1]|=c>>(32-(bit&31));}
  uint32_t* d=e3+((size_t)r*(K>>7)*4+(size_t)q*4+t)*3;d[0]=o[0];d[1]=o[1];d[2]=o[2];
}
template<class F> __device__ __forceinline__ void c11_walk(const uint16_t* W,int K,int KS,const uint8_t* eb,long long p,F&& fn){
  const int t=int(p&3),s=int((p>>2)%KS);const long long r=(p>>2)/KS;const int chunks=K>>5,c0=chunks*s/KS,c1=chunks*(s+1)/KS;
  const uint16_t* w=W+(size_t)r*K;const int e_max=eb[r];
  for(int cc=c0;cc<c1;cc++)for(int i=0;i<8;i++)fn(c11_idx(w[cc*32+t*8+i],e_max));
}
// escape counts per packet (nibbles)
__global__ void c11_esc_count(const uint16_t* __restrict__ W,int N,int K,int KS,const uint8_t* __restrict__ eb,int base,uint32_t* cnt){
  const long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=(long long)N*KS*4)return;
  uint32_t n=0;c11_walk(W,K,KS,eb,p,[&](uint32_t idx){n+=c11_code(idx,base)==0u;});cnt[p]=n;
}
// escape nibbles at nibble offset off[p] (packets start on even nibbles: off counts nibbles rounded up to bytes)
__global__ void c11_esc_pack(const uint16_t* __restrict__ W,int N,int K,int KS,const uint8_t* __restrict__ eb,int base,const uint32_t* __restrict__ off,uint8_t* esc){
  const long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=(long long)N*KS*4)return;
  uint32_t pos=off[p];c11_walk(W,K,KS,eb,p,[&](uint32_t idx){if(c11_code(idx,base)==0u){uint8_t* b=esc+(pos>>1);
    if(pos&1)*b=(uint8_t)((*b&15u)|(idx<<4));else *b=(uint8_t)idx;pos++;}});
}
// Load-time proof (always on): rebuild every 8-weight nibble word the GEMM way and compare with C12's.
__global__ void c11_verify(const uint16_t* __restrict__ W,int N,int K,int KS,const uint8_t* __restrict__ eb,const C11View v,unsigned long long* bad){
  const long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(p>=(long long)N*KS*4)return;
  const int t=int(p&3),s=int((p>>2)%KS);const int r=int((p>>2)/KS);const int chunks=K>>5,c0=chunks*s/KS,c1=chunks*(s+1)/KS;
  C11Esc E;E.init(v.esc,v.off[p]);const int e_max=eb[r];unsigned n=0;
  for(int cc=c0;cc<c1;cc++){
    const uint32_t* y=v.e3+((size_t)r*(K>>7)*4+(size_t)(cc>>2)*4+t)*3;const int u=cc&3;
    const uint32_t F[4]={y[0]&0xffffffu,__funnelshift_r(y[0],y[1],24)&0xffffffu,__funnelshift_r(y[1],y[2],16)&0xffffffu,y[2]>>8};
    const uint32_t got=c11_nibbles(F[u],v.base,E);uint32_t want=0;
    for(int i=0;i<8;i++)want|=c11_idx(W[(size_t)r*K+cc*32+t*8+i],e_max)<<(4*i);
    n+=got!=want;
  }
  if(n)atomicAdd(bad,(unsigned long long)n);
}
// Host: the window base (0..9) holding the most weights in [base, base+6] (nibble 15 always escapes).
inline int c11_pick_base(const unsigned long long hist[16]){
  int best=0;unsigned long long bw=0;
  for(int b=0;b<=8;b++){unsigned long long s=0;for(int i=b;i<=b+6&&i<15;i++)s+=hist[i];if(s>bw){bw=s;best=b;}}
  return best;
}
