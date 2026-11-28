// SPDX-License-Identifier: MIT
// Proposal 2 (GLM53_C12=1): entry points of the lossless 12-bit weight format (c12.cuh).
#include "c12.cuh"
#include "c12_big.cuh"
#include "gumbel.cuh"
#include "q8.cuh"
#include "q4d.cuh"

extern "C" int glm53_c12_stats_cuda(const void* w,int n,int k,void* eb,int* cnt,cudaStream_t s){
  if(n<1||k<1)return int(cudaErrorInvalidValue);
  c12_stats<<<n,256,0,s>>>((const uint16_t*)w,k,(uint8_t*)eb,cnt);return int(cudaGetLastError());
}
extern "C" int glm53_c12_pack_cuda(const void* w,int n,int k,const void* eb,const int* ptr,void* m8,void* e4,int* col,void* val,cudaStream_t s){
  if(n%16||k%128)return int(cudaErrorInvalidValue);
  const long long th=(long long)n*(k/32)*4;
  c12_pack<<<(unsigned)((th+255)/256),256,0,s>>>((const uint16_t*)w,n,k,(const uint8_t*)eb,(uint8_t*)m8,(uint8_t*)e4);
  c12_escapes<<<(n+7)/8,256,0,s>>>((const uint16_t*)w,n,k,(const uint8_t*)eb,ptr,col,(uint16_t*)val);
  return int(cudaGetLastError());
}
// Load-time proof (always on): decode every weight with the GEMM's own c12_dec8 and compare with the Half RN of the
// BF16 source value (the operand the Half path feeds its MMA). One thread per 8 consecutive k of one row; mismatches
// are counted into *bad.
__global__ void c12_verify(const uint16_t* __restrict__ W,const C12View w,unsigned long long* bad){
  const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;const int per_row=w.k/8;
  if(t>=(long long)w.n*per_row)return;
  const int row=(int)(t/per_row),kbase=(int)(t%per_row)*8,chunk=kbase>>5,lt=(kbase>>3)&3;
  const uint2 mm=*reinterpret_cast<const uint2*>(w.m8+(((size_t)row*(w.k>>6)+(chunk>>1))*4+lt)*16+(chunk&1)*8);
  const uint32_t e=*reinterpret_cast<const uint32_t*>(w.e4+(((size_t)row*(w.k>>7)+(chunk>>2))*4+lt)*16+(chunk&3)*4);
  const uint4 d=c12_dec8(mm.x,mm.y,e,w.eb[row],w,row,kbase);
  const uint32_t got[4]={d.x,d.y,d.z,d.w};unsigned n=0;
  for(int i=0;i<8;i++){
    const float v=__uint_as_float((uint32_t)W[(size_t)row*w.k+kbase+i]<<16);
    const unsigned short want=__half_as_ushort(__float2half_rn(v));
    const unsigned short have=(unsigned short)(got[i>>1]>>(16*(i&1)));
    n+=want!=have;
  }
  if(n)atomicAdd(bad,(unsigned long long)n);
}
extern "C" int glm53_c12_verify_cuda(const void* w,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,
                                     int n,int k,unsigned long long* bad,cudaStream_t s){
  const C12View v{(const uint8_t*)m8,(const uint8_t*)e4,(const uint8_t*)eb,ptr,col,(const uint16_t*)val,n,k};
  const long long th=(long long)n*(k/8);
  c12_verify<<<(unsigned)((th+255)/256),256,0,s>>>((const uint16_t*)w,v,bad);
  return int(cudaGetLastError());
}
extern "C" int glm53_c12_gemm_cuda(const float* x,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,
                                   int n,int k,void* y,int m,int out,int ks,cudaStream_t s){
  const C12View w{(const uint8_t*)m8,(const uint8_t*)e4,(const uint8_t*)eb,ptr,col,(const uint16_t*)val,n,k};
  return skinny_c12_launch(x,w,y,m,out,ks,s);
}
// GLM53_C12_XHALF: x is the Half [m,k] copy of the FP32 input (same RN values); bitwise the FP32-input launch.
extern "C" int glm53_c12_gemm_xh_cuda(const void* xh,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,
                                      int n,int k,void* y,int m,int out,int ks,cudaStream_t s){
  const C12View w{(const uint8_t*)m8,(const uint8_t*)e4,(const uint8_t*)eb,ptr,col,(const uint16_t*)val,n,k};
  return skinny_c12_launch((const float*)xh,w,y,m,out,ks,s,true);
}

// GLM53_MLA_BMM_C12=1: latent_half_bmm (MLA absorb/expand, per-head W [N,K] times x_h [M,K]) reading the per-head
// weights from one C12 coding of [heads*N, K]. Same lane k-permutation, chunk order, split-K ranges (KS 4 for K>=512,
// else 2), MMA sequence and reduction as latent_half_bmm, and c12_dec8 yields the same Half operands: bitwise equal.
template<int KS,int NT=1>
__global__ void __launch_bounds__(256) latent_c12_bmm(const float* __restrict__ X,long long xh,long long xm,const C12View w,int N,
    float* __restrict__ Y,int M){
  constexpr int RT=8/KS;
  __shared__ float red[KS][RT][NT][4][32];
  const int head=blockIdx.y,K=w.k;
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS,rbase=(blockIdx.x*RT+rt)*16;
  float c[NT][4];
#pragma unroll
  for(int j=0;j<NT;j++){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
  const int chunks=K>>5,c0=chunks*ks/KS,c1=chunks*(ks+1)/KS;
  const bool live=rbase<N;
  const int r0=head*N+min(rbase+g,N-1),r1=head*N+min(rbase+g+8,N-1);
  const uint8_t* m0=w.m8+((size_t)r0*(K>>6)*4+t)*16;const uint8_t* m1=w.m8+((size_t)r1*(K>>6)*4+t)*16;
  const uint8_t* e0=w.e4+((size_t)r0*(K>>7)*4+t)*16;const uint8_t* e1=w.e4+((size_t)r1*(K>>7)*4+t)*16;
  const uint32_t b0=w.eb[r0],b1=w.eb[r1];
  bool xv[NT];const float* xr[NT];
#pragma unroll
  for(int j=0;j<NT;j++){const int tok=j*8+g;xv[j]=tok<M;xr[j]=X+head*xh+(xv[j]?tok:0)*xm+t*8;}
  if(live){
    for(int q=c0>>2;q<(c1>>2);++q){
      const uint4 pa0=skinny_ld_stream(m0+(size_t)(2*q)*64),pb0=skinny_ld_stream(m0+(size_t)(2*q+1)*64);
      const uint4 pa1=skinny_ld_stream(m1+(size_t)(2*q)*64),pb1=skinny_ld_stream(m1+(size_t)(2*q+1)*64);
      const uint4 ea=skinny_ld_stream(e0+(size_t)q*64),eb=skinny_ld_stream(e1+(size_t)q*64);
      const uint32_t M0[8]={pa0.x,pa0.y,pa0.z,pa0.w,pb0.x,pb0.y,pb0.z,pb0.w},M1[8]={pa1.x,pa1.y,pa1.z,pa1.w,pb1.x,pb1.y,pb1.z,pb1.w};
      const uint32_t E0[4]={ea.x,ea.y,ea.z,ea.w},E1[4]={eb.x,eb.y,eb.z,eb.w};
#pragma unroll
      for(int u=0;u<4;u++){
        const int cc=4*q+u,kb=cc*32+t*8;
        const uint4 a=c12_dec8(M0[2*u],M0[2*u+1],E0[u],b0,w,r0,kb),b=c12_dec8(M1[2*u],M1[2*u+1],E1[u],b1,w,r1,kb);
#pragma unroll
        for(int j=0;j<NT;j++){
          uint32_t xb[4]={0,0,0,0};
          if(xv[j]){const float4 p=*reinterpret_cast<const float4*>(xr[j]+cc*32),s=*reinterpret_cast<const float4*>(xr[j]+cc*32+4);
            xb[0]=f2h2(p.x,p.y);xb[1]=f2h2(p.z,p.w);xb[2]=f2h2(s.x,s.y);xb[3]=f2h2(s.z,s.w);}
          skinny_mma16816(c[j],a.x,b.x,a.y,b.y,xb[0],xb[1]);
          skinny_mma16816(c[j],a.z,b.z,a.w,b.w,xb[2],xb[3]);
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
      if(row<N&&tok<M)Y[((long long)head*M+tok)*N+row]=s;}
  }
}
extern "C" int glm53_latent_c12_bmm_cuda(const float* x,long long xh,long long xm,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,
    const void* val,int heads,int n,int k,float* y,int m,cudaStream_t s){
  // m 1..8 as before; 9..32 (GLM53_MLA_BMM_WIDE) take 2 or 4 token tiles per block (each row's MMA sequence unchanged).
  if(m<1||m>32||n%16||k%128||heads<1)return int(cudaErrorInvalidValue);
  const int ks=(k>=512)?4:2;const int chunks=k>>5;if((chunks/ks)*ks!=chunks||(chunks/ks)%4)return int(cudaErrorInvalidValue);
  const C12View w{(const uint8_t*)m8,(const uint8_t*)e4,(const uint8_t*)eb,ptr,col,(const uint16_t*)val,heads*n,k};
  const int rt=8/ks;dim3 grid((n+16*rt-1)/(16*rt),heads);
#define LB(KS,NT) latent_c12_bmm<KS,NT><<<grid,256,0,s>>>(x,xh,xm,w,n,y,m)
  if(ks==4){if(m<=8)LB(4,1);else if(m<=16)LB(4,2);else LB(4,4);}
  else{if(m<=8)LB(2,1);else if(m<=16)LB(2,2);else LB(2,4);}
#undef LB
  return int(cudaGetLastError());
}

// GLM53_Q8=1 (q8.cuh): near-lossless int8 dense decode format.
extern "C" int glm53_q8_encode_cuda(const void* w,int n,int k,int mse,float* s,void* q,double* err,cudaStream_t st){
  if(n%16||k%128)return int(cudaErrorInvalidValue);
  const long long groups=(long long)n*(k/128);
  q8_scales<<<(unsigned)((groups*32+255)/256),256,0,st>>>((const uint16_t*)w,n,k,mse,s);
  const long long th=(long long)n*(k/32)*4;
  q8_pack<<<(unsigned)((th+255)/256),256,0,st>>>((const uint16_t*)w,n,k,s,(uint8_t*)q,err);
  return int(cudaGetLastError());
}
extern "C" int glm53_q8_gemm_cuda(const float* x,const void* q,const float* s,int n,int k,void* y,int m,int out,int ks,cudaStream_t st){
  const Q8View w{(const uint8_t*)q,s,n,k};
  return skinny_q8_launch(x,w,y,m,out,ks,st);
}
// GLM53_Q8_XHALF: x is the Half (RN) copy of X.
extern "C" int glm53_q8_gemm_xh_cuda(const void* x,const void* q,const float* s,int n,int k,void* y,int m,int out,int ks,cudaStream_t st){
  const Q8View w{(const uint8_t*)q,s,n,k};
  return skinny_q8_launch((const float*)x,w,y,m,out,ks,st,true);
}

// GLM53_Q8_TILED=1: tiled Q8 layout (q8.cuh skinny_q8t), bitwise skinny_q8.
extern "C" int glm53_q8t_pack_cuda(const void* q,const float* s,int n,int k,void* t,cudaStream_t st){
  if(n%16||k%128)return int(cudaErrorInvalidValue);
  const long long th=(long long)(n/16)*(k/128)*32;
  q8t_pack<<<(unsigned)((th+255)/256),256,0,st>>>((const uint8_t*)q,s,n,k,(uint8_t*)t);
  return int(cudaGetLastError());
}
extern "C" int glm53_q8t_gemm_cuda(const void* x,const void* t,int n,int k,void* y,int m,int out,int ks,int xhalf,cudaStream_t st){
  const Q8T w{(const uint8_t*)t,n,k};
  return skinny_q8t_launch((const float*)x,w,y,m,out,ks,st,xhalf!=0);
}

// GLM53_DENSE_Q4_SET (q4d.cuh): affine 4-bit dense decode format.
extern "C" int glm53_q4d_encode_cuda(const void* w,int n,int k,int mse,void* sm,void* q,double* err,cudaStream_t st){
  if(n%16||k%128)return int(cudaErrorInvalidValue);
  const long long groups=(long long)n*(k/64);
  q4d_ranges<<<(unsigned)((groups*32+255)/256),256,0,st>>>((const uint16_t*)w,n,k,mse,(half2*)sm);
  const long long th=(long long)n*(k/128)*4;
  q4d_pack<<<(unsigned)((th+255)/256),256,0,st>>>((const uint16_t*)w,n,k,(const half2*)sm,(uint8_t*)q,err);
  return int(cudaGetLastError());
}
extern "C" int glm53_q4d_gemm_cuda(const float* x,const void* q,const void* sm,int n,int k,void* y,int m,int out,int ks,cudaStream_t st){
  const Q4View w{(const uint8_t*)q,(const half2*)sm,n,k};
  return skinny_q4d_launch(x,w,y,m,out,ks,st);
}

// ---- FP32-arithmetic warp-per-row GEMV on the C12 coding (first user: DS41's BF16 weights, ds41.cu bf16_gemv).
// The decoded weight is the exact BF16 value widened to FP32 (no Half rounding). Each output row is one warp; lane l
// takes k = i*256 + l*8 .. +7 for i = 0, 1, ... in order, FMAs x * w into one FP32 accumulator per input row and the
// warp sums with shfl_down 16..1: the same operands, FMA sequence and reduction as ds41.cu bf16_gemv, hence bitwise
// equal to it (and, like it, each row's value does not depend on m). The GLM m8 / e4 layout puts those 8 weights of a
// lane in 8 contiguous m8 bytes and one e4 word (warp reads 256 + 128 contiguous bytes per i). Needs k % 256 == 0.
__device__ __forceinline__ void c12_dec8_f32(uint2 mm,uint32_t e,uint32_t eb,const C12View& w,int row,int kbase,float f[8]){
#pragma unroll
  for(int i=0;i<8;i++){
    const uint32_t m=((i<4?mm.x:mm.y)>>(8*(i&3)))&0xffu, idx=(e>>(4*i))&0xfu;
    f[i]=__uint_as_float(((m&0x80u)<<24)|((eb-idx)<<23)|((m&0x7fu)<<16));
  }
  if(__builtin_expect((e&(e>>1)&(e>>2)&(e>>3)&0x11111111u)!=0,0)){
#pragma unroll
    for(int i=0;i<8;i++) if(((e>>(4*i))&0xfu)==15u) f[i]=c12_escape(w,row,kbase+i);
  }
}
template<int M,int U>
__global__ void __launch_bounds__(256) c12_gemv_f32(const __nv_bfloat16* __restrict__ x,const C12View w,float* __restrict__ y,int m){
  pdl_trigger();
  const int lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
  if(row>=w.n)return;
  const int K=w.k,t=lane&3,iters=K>>8;
  const uint8_t* mr=w.m8+(size_t)row*K;const uint8_t* er=w.e4+(size_t)row*(K>>1);
  const uint32_t eb=w.eb[row];
  float acc[M];
#pragma unroll
  for(int r=0;r<M;++r)acc[r]=0.f;
  // the weights (and the escape tables) do not depend on earlier kernels: the first U steps load before the PDL wait
  bool waited=false;
  for(int i0=0;i0<iters;i0+=U){
    // U steps' weights in flight; consumed in step order (the FMA sequence does not depend on U)
    uint2 mm[U];uint32_t ee[U];
#pragma unroll
    for(int v=0;v<U;v++){
      const int c=(i0+v)*256+lane*8;
      if(i0+v<iters){mm[v]=__ldg(reinterpret_cast<const uint2*>(mr+((c>>6)*4+t)*16+((c>>5)&1)*8));
                     ee[v]=__ldg(reinterpret_cast<const uint32_t*>(er+((c>>7)*4+t)*16+((c>>5)&3)*4));}
    }
    if(!waited){pdl_wait();waited=true;}
#pragma unroll
    for(int v=0;v<U;v++){
      if(i0+v>=iters)break;
      const int c=(i0+v)*256+lane*8;
      float wv[8];c12_dec8_f32(mm[v],ee[v],eb,w,row,c,wv);
#pragma unroll
      for(int r=0;r<M;++r){
        if(r<m){
          const uint4 xp=*reinterpret_cast<const uint4*>(x+(long long)r*K+c);
          const __nv_bfloat162* xb=reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
          for(int j=0;j<4;++j){const float2 f=__bfloat1622float2(xb[j]);acc[r]=__fmaf_rn(f.x,wv[2*j],acc[r]);acc[r]=__fmaf_rn(f.y,wv[2*j+1],acc[r]);}
        }
      }
    }
  }
#pragma unroll
  for(int r=0;r<M;++r){
    float v=acc[r];
    for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o);
    if(lane==0&&r<m)y[(long long)r*w.n+row]=v;
  }
}
extern "C" int glm53_c12_gemv_f32_cuda(const void* x,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,
                                       float* y,int m,int n,int k,int pdl,cudaStream_t st){
  if(m<1||m>32||k%256!=0||n%16!=0)return int(cudaErrorInvalidValue);
  const C12View w{(const uint8_t*)m8,(const uint8_t*)e4,(const uint8_t*)eb,ptr,col,(const uint16_t*)val,n,k};
  const __nv_bfloat16* xx=(const __nv_bfloat16*)x;const dim3 g((n+7)/8),b(256);const bool p=pdl!=0;
  // 17..32 rows (DS41_GEMV32 verify rounds; DS41 is the only caller of this entry with m > 16): one read of the weight,
  // each row's FMA sequence and reduction its own (bitwise the <= 16-row calls). The M <= 16 instantiations are unchanged.
  if(m>16){
    if(n<=(std::getenv("DS41_C12_DEEP_ROWS")?std::atoi(std::getenv("DS41_C12_DEEP_ROWS")):4096)&&k==5120)pdl_launch_if(p,c12_gemv_f32<32,10>,g,b,0,st,xx,w,y,m);
    else pdl_launch_if(p,c12_gemv_f32<32,4>,g,b,0,st,xx,w,y,m);
    return int(cudaGetLastError());
  }
  // 9..16 rows (multi-sequence verify): one read of the weight for all of them; each row's FMA sequence and reduction
  // are M's and m's own (bitwise the 8-row calls)
  if(m>8){
    if(n<=(std::getenv("DS41_C12_DEEP_ROWS")?std::atoi(std::getenv("DS41_C12_DEEP_ROWS")):4096)&&k==5120)pdl_launch_if(p,c12_gemv_f32<16,10>,g,b,0,st,xx,w,y,m);
    else pdl_launch_if(p,c12_gemv_f32<16,4>,g,b,0,st,xx,w,y,m);
    return int(cudaGetLastError());
  }
  // few rows (routers [384 or 128, 5120]): one warp per row leaves too few loads in flight (3.9 MB in 33 us), so
  // every lane loads all 20 of its steps up front (U = 20; the FMA sequence is the same)
  static const int deep_rows=std::getenv("DS41_C12_DEEP_ROWS")?std::atoi(std::getenv("DS41_C12_DEEP_ROWS")):4096;
  if(n<=deep_rows&&k==5120){
    switch(m){case 1:pdl_launch_if(p,c12_gemv_f32<1,20>,g,b,0,st,xx,w,y,m);break;case 2:pdl_launch_if(p,c12_gemv_f32<2,20>,g,b,0,st,xx,w,y,m);break;
      case 3:case 4:pdl_launch_if(p,c12_gemv_f32<4,20>,g,b,0,st,xx,w,y,m);break;default:pdl_launch_if(p,c12_gemv_f32<8,20>,g,b,0,st,xx,w,y,m);}
    return int(cudaGetLastError());
  }
  switch(m){case 1:pdl_launch_if(p,c12_gemv_f32<1,4>,g,b,0,st,xx,w,y,m);break;case 2:pdl_launch_if(p,c12_gemv_f32<2,4>,g,b,0,st,xx,w,y,m);break;
    case 3:case 4:pdl_launch_if(p,c12_gemv_f32<4,4>,g,b,0,st,xx,w,y,m);break;default:pdl_launch_if(p,c12_gemv_f32<8,4>,g,b,0,st,xx,w,y,m);}
  return int(cudaGetLastError());
}
// Load-time proof for the FP32 users: every decoded weight equals its BF16 source bits exactly (mismatches -> *bad).
__global__ void c12_verify_exact(const uint16_t* __restrict__ W,const C12View w,unsigned long long* bad){
  const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;const int per_row=w.k/8;
  if(t>=(long long)w.n*per_row)return;
  const int row=(int)(t/per_row),kbase=(int)(t%per_row)*8,chunk=kbase>>5,lt=(kbase>>3)&3;
  const uint2 mm=*reinterpret_cast<const uint2*>(w.m8+(((size_t)row*(w.k>>6)+(chunk>>1))*4+lt)*16+(chunk&1)*8);
  const uint32_t e=*reinterpret_cast<const uint32_t*>(w.e4+(((size_t)row*(w.k>>7)+(chunk>>2))*4+lt)*16+(chunk&3)*4);
  float f[8];c12_dec8_f32(mm,e,w.eb[row],w,row,kbase,f);unsigned n=0;
  for(int i=0;i<8;i++) n+=__float_as_uint(f[i])!=((uint32_t)W[(size_t)row*w.k+kbase+i]<<16);
  if(n)atomicAdd(bad,(unsigned long long)n);
}
extern "C" int glm53_c12_verify_exact_cuda(const void* src,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,
                                           int n,int k,unsigned long long* bad,cudaStream_t st){
  const C12View v{(const uint8_t*)m8,(const uint8_t*)e4,(const uint8_t*)eb,ptr,col,(const uint16_t*)val,n,k};
  const long long th=(long long)n*(k/8);
  c12_verify_exact<<<(unsigned)((th+255)/256),256,0,st>>>((const uint16_t*)src,v,bad);
  return int(cudaGetLastError());
}

// ---- Fused Markov head C12 GEMV (K = 256) + head logits add + Gumbel top-1 argmax (DS41_MARKOV_FUSED, L0).
// Exact same c12_dec8_f32 weights, FMA order and warp reduction as c12_gemv_f32, same FP32 logits + bias addition, and
// same Gumbel estimate/exact pruning and tie-break as gumbel_argmax_part + gumbel_argmax_packet: bitwise identical
// without materializing bias [m, n] or lg [m, n] in global memory.
template<int M>
__global__ void __launch_bounds__(256) c12_markov_argmax_part(
    const __nv_bfloat16* __restrict__ x,const C12View w,const float* __restrict__ logits,int blk,int step_j,
    long long col_offset,const float* __restrict__ temps,const unsigned long long* __restrict__ keys,
    int m,float* __restrict__ part_v,int* __restrict__ part_i){
  pdl_trigger();
  constexpr int WARPS=8;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,t=lane&3,c=lane*8;
  const int stride=gridDim.x*WARPS;
  const int row0=blockIdx.x*WARPS+warp;
  const size_t m_off=((c>>6)*4+t)*16+((c>>5)&1)*8;
  const size_t e_off=((c>>7)*4+t)*16+((c>>5)&3)*4;
  // Prefetch the first row's C12 weights before PDL wait (weights do not depend on earlier kernels).
  uint2 mm_cur{};uint32_t ee_cur=0;uint32_t eb_cur=0;
  if(row0<w.n){
    mm_cur=__ldg(reinterpret_cast<const uint2*>(w.m8+(size_t)row0*256+m_off));
    ee_cur=__ldg(reinterpret_cast<const uint32_t*>(w.e4+(size_t)row0*128+e_off));
    eb_cur=__ldg(w.eb+row0);
  }
  pdl_wait();
  __shared__ __nv_bfloat16 sx[M][256];
  for(int i=tid;i<m*32;i+=256){
    const int r=i>>5,col=(i&31)*8;
    *reinterpret_cast<uint4*>(&sx[r][col])=*reinterpret_cast<const uint4*>(x+(long long)r*256+col);
  }
  __syncthreads();
  float tm[M];unsigned long long ky[M];
  float bv[M];int bi[M];
#pragma unroll
  for(int r=0;r<M;++r){
    tm[r]=r<m?temps[r]:0.f;
    ky[r]=r<m?keys[r]:0ULL;
    bv[r]=-INFINITY;bi[r]=0x7fffffff;
  }
  float xf[M<=4?M:1][8];
  if constexpr(M<=4){
#pragma unroll
    for(int r=0;r<M;++r){
      if(r<m){
        const uint4 xp=*reinterpret_cast<const uint4*>(&sx[r][c]);
        const __nv_bfloat162* xb=reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
        for(int j=0;j<4;++j){const float2 f=__bfloat1622float2(xb[j]);xf[r][2*j]=f.x;xf[r][2*j+1]=f.y;}
      }
    }
  }
  for(int row=row0;row<w.n;row+=stride){
    const int next_row=row+stride;
    uint2 mm_nxt{};uint32_t ee_nxt=0;uint32_t eb_nxt=0;
    if(next_row<w.n){
      mm_nxt=__ldg(reinterpret_cast<const uint2*>(w.m8+(size_t)next_row*256+m_off));
      ee_nxt=__ldg(reinterpret_cast<const uint32_t*>(w.e4+(size_t)next_row*128+e_off));
      eb_nxt=__ldg(w.eb+next_row);
    }
    float wv[8];c12_dec8_f32(mm_cur,ee_cur,eb_cur,w,row,c,wv);
#pragma unroll
    for(int r=0;r<M;++r){
      if(r<m){
        float acc=0.f;
        if constexpr(M<=4){
#pragma unroll
          for(int j=0;j<4;++j){acc=__fmaf_rn(xf[r][2*j],wv[2*j],acc);acc=__fmaf_rn(xf[r][2*j+1],wv[2*j+1],acc);}
        }else{
          const uint4 xp=*reinterpret_cast<const uint4*>(&sx[r][c]);
          const __nv_bfloat162* xb=reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
          for(int j=0;j<4;++j){const float2 f=__bfloat1622float2(xb[j]);acc=__fmaf_rn(f.x,wv[2*j],acc);acc=__fmaf_rn(f.y,wv[2*j+1],acc);}
        }
        for(int o=16;o;o>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,o);
        if(lane==0){
          const float l=logits[((long long)r*blk+step_j)*w.n+row]+acc;
          const float tr=tm[r];
          if(!(tr>0.f)){
            if(bi[r]==0x7fffffff||gumbel_better(l,row,bv[r],bi[r])){bv[r]=l;bi[r]=row;}
          }else{
            const long long gid=(long long)row+col_offset;
            const float s=l+gumbel_estimate(ky[r],gid,tr);
            const float eps=isfinite(s)?tr*1e-4f+1e-5f*fabsf(s):0.f;
            if(bi[r]==0x7fffffff||!isfinite(s)||(bv[r]==bv[r]&&s+eps>=bv[r])){
              const float v=l+gumbel_exact(ky[r],gid,tr);
              if(bi[r]==0x7fffffff||gumbel_better(v,row,bv[r],bi[r])){bv[r]=v;bi[r]=row;}
            }
          }
        }
      }
    }
    mm_cur=mm_nxt;ee_cur=ee_nxt;eb_cur=eb_nxt;
  }
  __shared__ float sv[M][WARPS];__shared__ int si[M][WARPS];
  if(lane==0){
#pragma unroll
    for(int r=0;r<M;++r){sv[r][warp]=bv[r];si[r][warp]=bi[r];}
  }
  __syncthreads();
  if(tid<m){
    const int r=tid;
    float rv=-INFINITY;int ri=0x7fffffff;
#pragma unroll
    for(int w0=0;w0<WARPS;++w0){
      const int ci=si[r][w0];
      if(ci!=0x7fffffff&&(ri==0x7fffffff||gumbel_better(sv[r][w0],ci,rv,ri))){rv=sv[r][w0];ri=ci;}
    }
    part_v[r*gridDim.x+blockIdx.x]=rv;part_i[r*gridDim.x+blockIdx.x]=ri;
  }
}
__global__ void c12_markov_argmax_fin(const float* __restrict__ part_v,const int* __restrict__ part_i,int nb,int rows,
                                      int world,int rank,long long start,float* __restrict__ packet){
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31;
  for(int j=tid;j<world*rows*2;j+=blockDim.x)packet[j]=0.f;
  __syncthreads();
  if(warp<rows){
    float bv=-INFINITY;int bi=0x7fffffff;
    for(int b=lane;b<nb;b+=32){
      const int ci=part_i[warp*nb+b];
      if(ci!=0x7fffffff&&(bi==0x7fffffff||gumbel_better(part_v[warp*nb+b],ci,bv,bi))){bv=part_v[warp*nb+b];bi=ci;}
    }
    for(int o=16;o;o>>=1){
      const float ov=__shfl_xor_sync(0xffffffffu,bv,o);const int oi=__shfl_xor_sync(0xffffffffu,bi,o);
      if(oi!=0x7fffffff&&(bi==0x7fffffff||gumbel_better(ov,oi,bv,bi))){bv=ov;bi=oi;}
    }
    if(lane==0){packet[((long long)rank*rows+warp)*2]=bv;packet[((long long)rank*rows+warp)*2+1]=(float)(bi+start);}
  }
}
extern "C" int glm53_c12_markov_argmax_cuda(
    const void* x,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,
    const float* logits,int blk,int step_j,int m,int n,int k,long long start,
    const float* temps,const unsigned long long* keys,float* part_v,int* part_i,float* packet,
    int world,int rank,int nb,int pdl,cudaStream_t st){
  if(m<1||m>8||k!=256||n<1||n%16!=0||blk<1||step_j<0||step_j>=blk||world<1||rank<0||rank>=world||nb<1)return int(cudaErrorInvalidValue);
  const C12View w{(const uint8_t*)m8,(const uint8_t*)e4,(const uint8_t*)eb,ptr,col,(const uint16_t*)val,n,k};
  const __nv_bfloat16* xx=(const __nv_bfloat16*)x;
  const dim3 g((unsigned)nb),b(256);const bool p=pdl!=0;
  switch(m){
    case 1:pdl_launch_if(p,c12_markov_argmax_part<1>,g,b,0,st,xx,w,logits,blk,step_j,start,temps,keys,m,part_v,part_i);break;
    case 2:pdl_launch_if(p,c12_markov_argmax_part<2>,g,b,0,st,xx,w,logits,blk,step_j,start,temps,keys,m,part_v,part_i);break;
    case 3:case 4:pdl_launch_if(p,c12_markov_argmax_part<4>,g,b,0,st,xx,w,logits,blk,step_j,start,temps,keys,m,part_v,part_i);break;
    default:pdl_launch_if(p,c12_markov_argmax_part<8>,g,b,0,st,xx,w,logits,blk,step_j,start,temps,keys,m,part_v,part_i);break;
  }
  c12_markov_argmax_fin<<<1,256,0,st>>>(part_v,part_i,nb,m,world,rank,start,packet);
  return int(cudaGetLastError());
}

// ---- Top-2 Markov argmax + branch continuation in one pass (DS41_SPEC_TREE, L0).
// Supports b_mod <= m (e.g. m = 2*b: rows 0..b-1 primary chain, rows b..2b-1 alt0 continuation sharing the same
// logits/temps/keys of sequence r % b_mod). Returns packet2 [world, m, 4] = (v1, id1, v2, id2); top-1 is bitwise
// identical to c12_markov_argmax_part.
__device__ __forceinline__ void top2_push(float v,int i,float& v1,int& i1,float& v2,int& i2){
  if(i==0x7fffffff)return;
  if(i1==0x7fffffff||gumbel_better(v,i,v1,i1)){v2=v1;i2=i1;v1=v;i1=i;}
  else if(i2==0x7fffffff||gumbel_better(v,i,v2,i2)){v2=v;i2=i;}
}
template<int M>
__global__ void __launch_bounds__(256) c12_markov_top2_part(
    const __nv_bfloat16* __restrict__ x,const C12View w,const float* __restrict__ logits,int blk,int step_j,
    int b_mod,long long col_offset,const float* __restrict__ temps,const unsigned long long* __restrict__ keys,
    int m,float* __restrict__ pv1,int* __restrict__ pi1,float* __restrict__ pv2,int* __restrict__ pi2){
  pdl_trigger();
  constexpr int WARPS=8;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,t=lane&3,c=lane*8;
  const int stride=gridDim.x*WARPS;
  const int row0=blockIdx.x*WARPS+warp;
  const size_t m_off=((c>>6)*4+t)*16+((c>>5)&1)*8;
  const size_t e_off=((c>>7)*4+t)*16+((c>>5)&3)*4;
  uint2 mm_cur{};uint32_t ee_cur=0;uint32_t eb_cur=0;
  if(row0<w.n){
    mm_cur=__ldg(reinterpret_cast<const uint2*>(w.m8+(size_t)row0*256+m_off));
    ee_cur=__ldg(reinterpret_cast<const uint32_t*>(w.e4+(size_t)row0*128+e_off));
    eb_cur=__ldg(w.eb+row0);
  }
  pdl_wait();
  __shared__ __nv_bfloat16 sx[M][256];
  for(int i=tid;i<m*32;i+=256){
    const int r=i>>5,col=(i&31)*8;
    *reinterpret_cast<uint4*>(&sx[r][col])=*reinterpret_cast<const uint4*>(x+(long long)r*256+col);
  }
  __syncthreads();
  float tm[M];unsigned long long ky[M];
  float bv1[M],bv2[M];int bi1[M],bi2[M];int rseq[M];
#pragma unroll
  for(int r=0;r<M;++r){
    const int rb=r<m?(r<b_mod?r:r-b_mod):0;
    rseq[r]=rb;
    tm[r]=r<m?temps[rb]:0.f;
    ky[r]=r<m?keys[rb]:0ULL;
    bv1[r]=-INFINITY;bi1[r]=0x7fffffff;bv2[r]=-INFINITY;bi2[r]=0x7fffffff;
  }
  float xf[M<=4?M:1][8];
  if constexpr(M<=4){
#pragma unroll
    for(int r=0;r<M;++r){
      if(r<m){
        const uint4 xp=*reinterpret_cast<const uint4*>(&sx[r][c]);
        const __nv_bfloat162* xb=reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
        for(int j=0;j<4;++j){const float2 f=__bfloat1622float2(xb[j]);xf[r][2*j]=f.x;xf[r][2*j+1]=f.y;}
      }
    }
  }
  for(int row=row0;row<w.n;row+=stride){
    const int next_row=row+stride;
    uint2 mm_nxt{};uint32_t ee_nxt=0;uint32_t eb_nxt=0;
    if(next_row<w.n){
      mm_nxt=__ldg(reinterpret_cast<const uint2*>(w.m8+(size_t)next_row*256+m_off));
      ee_nxt=__ldg(reinterpret_cast<const uint32_t*>(w.e4+(size_t)next_row*128+e_off));
      eb_nxt=__ldg(w.eb+next_row);
    }
    float wv[8];c12_dec8_f32(mm_cur,ee_cur,eb_cur,w,row,c,wv);
#pragma unroll
    for(int r=0;r<M;++r){
      if(r<m){
        float acc=0.f;
        if constexpr(M<=4){
#pragma unroll
          for(int j=0;j<4;++j){acc=__fmaf_rn(xf[r][2*j],wv[2*j],acc);acc=__fmaf_rn(xf[r][2*j+1],wv[2*j+1],acc);}
        }else{
          const uint4 xp=*reinterpret_cast<const uint4*>(&sx[r][c]);
          const __nv_bfloat162* xb=reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
          for(int j=0;j<4;++j){const float2 f=__bfloat1622float2(xb[j]);acc=__fmaf_rn(f.x,wv[2*j],acc);acc=__fmaf_rn(f.y,wv[2*j+1],acc);}
        }
        for(int o=16;o;o>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,o);
        if(lane==0){
          const float l=logits[((long long)rseq[r]*blk+step_j)*w.n+row]+acc;
          const float tr=tm[r];
          if(!(tr>0.f)){
            top2_push(l,row,bv1[r],bi1[r],bv2[r],bi2[r]);
          }else{
            const long long gid=(long long)row+col_offset;
            const float s=l+gumbel_estimate(ky[r],gid,tr);
            const float eps=isfinite(s)?tr*1e-4f+1e-5f*fabsf(s):0.f;
            // Prune only against bv2[r] so both top-1 and top-2 are exact.
            if(bi2[r]==0x7fffffff||!isfinite(s)||(bv2[r]==bv2[r]&&s+eps>=bv2[r])){
              const float v=l+gumbel_exact(ky[r],gid,tr);
              top2_push(v,row,bv1[r],bi1[r],bv2[r],bi2[r]);
            }
          }
        }
      }
    }
    mm_cur=mm_nxt;ee_cur=ee_nxt;eb_cur=eb_nxt;
  }
  __shared__ float sv1[M][WARPS],sv2[M][WARPS];__shared__ int si1[M][WARPS],si2[M][WARPS];
  if(lane==0){
#pragma unroll
    for(int r=0;r<M;++r){sv1[r][warp]=bv1[r];si1[r][warp]=bi1[r];sv2[r][warp]=bv2[r];si2[r][warp]=bi2[r];}
  }
  __syncthreads();
  if(tid<m){
    const int r=tid;
    float r1=-INFINITY,r2=-INFINITY;int j1=0x7fffffff,j2=0x7fffffff;
#pragma unroll
    for(int w0=0;w0<WARPS;++w0){
      top2_push(sv1[r][w0],si1[r][w0],r1,j1,r2,j2);
      top2_push(sv2[r][w0],si2[r][w0],r1,j1,r2,j2);
    }
    const int idx=r*gridDim.x+blockIdx.x;
    pv1[idx]=r1;pi1[idx]=j1;pv2[idx]=r2;pi2[idx]=j2;
  }
}
__global__ void c12_markov_top2_fin(const float* __restrict__ pv1,const int* __restrict__ pi1,
                                    const float* __restrict__ pv2,const int* __restrict__ pi2,
                                    int nb,int rows,int world,int rank,long long start,float* __restrict__ packet2){
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31;
  for(int j=tid;j<world*rows*4;j+=blockDim.x)packet2[j]=0.f;
  __syncthreads();
  if(warp<rows){
    float v1=-INFINITY,v2=-INFINITY;int i1=0x7fffffff,i2=0x7fffffff;
    for(int b=lane;b<nb;b+=32){
      const int idx=warp*nb+b;
      top2_push(pv1[idx],pi1[idx],v1,i1,v2,i2);
      top2_push(pv2[idx],pi2[idx],v1,i1,v2,i2);
    }
    for(int o=16;o;o>>=1){
      const float ov1=__shfl_xor_sync(0xffffffffu,v1,o),ov2=__shfl_xor_sync(0xffffffffu,v2,o);
      const int oi1=__shfl_xor_sync(0xffffffffu,i1,o),oi2=__shfl_xor_sync(0xffffffffu,i2,o);
      top2_push(ov1,oi1,v1,i1,v2,i2);
      top2_push(ov2,oi2,v1,i1,v2,i2);
    }
    if(lane==0){
      float* dst=packet2+((long long)rank*rows+warp)*4;
      dst[0]=v1;dst[1]=i1!=0x7fffffff?(float)(i1+start):-1.f;
      dst[2]=v2;dst[3]=i2!=0x7fffffff?(float)(i2+start):-1.f;
    }
  }
}
__global__ void choose_top2_kernel(const float* __restrict__ packet2,int world,int rows,
                                   long long* __restrict__ tok1,long long* __restrict__ tok2,float* __restrict__ margin){
  const int r=threadIdx.x;
  if(r>=rows)return;
  float v1=-INFINITY,v2=-INFINITY;int i1=0x7fffffff,i2=0x7fffffff;
  for(int k=0;k<world;++k){
    const float* p=packet2+((long long)k*rows+r)*4;
    const int id1=(int)p[1],id2=(int)p[3];
    if(id1>=0)top2_push(p[0],id1,v1,i1,v2,i2);
    if(id2>=0)top2_push(p[2],id2,v1,i1,v2,i2);
  }
  tok1[r]=i1!=0x7fffffff?(long long)i1:0LL;
  tok2[r]=i2!=0x7fffffff?(long long)i2:-1LL;
  margin[r]=(i1!=0x7fffffff&&i2!=0x7fffffff&&isfinite(v1)&&isfinite(v2))?(v1-v2):1e30f;
}
extern "C" int glm53_c12_markov_top2_cuda(
    const void* x,const void* m8,const void* e4,const void* eb,const int* ptr,const int* col,const void* val,
    const float* logits,int blk,int step_j,int b_mod,int m,int n,int k,long long start,
    const float* temps,const unsigned long long* keys,float* pv1,int* pi1,float* pv2,int* pi2,float* packet2,
    int world,int rank,int nb,int pdl,cudaStream_t st){
  if(m<1||m>8||b_mod<1||b_mod>m||k!=256||n<1||n%16!=0||blk<1||step_j<0||step_j>=blk||world<1||rank<0||rank>=world||nb<1)return int(cudaErrorInvalidValue);
  const C12View w{(const uint8_t*)m8,(const uint8_t*)e4,(const uint8_t*)eb,ptr,col,(const uint16_t*)val,n,k};
  const __nv_bfloat16* xx=(const __nv_bfloat16*)x;
  const dim3 g((unsigned)nb),b(256);const bool p=pdl!=0;
  switch(m){
    case 1:pdl_launch_if(p,c12_markov_top2_part<1>,g,b,0,st,xx,w,logits,blk,step_j,b_mod,start,temps,keys,m,pv1,pi1,pv2,pi2);break;
    case 2:pdl_launch_if(p,c12_markov_top2_part<2>,g,b,0,st,xx,w,logits,blk,step_j,b_mod,start,temps,keys,m,pv1,pi1,pv2,pi2);break;
    case 3:case 4:pdl_launch_if(p,c12_markov_top2_part<4>,g,b,0,st,xx,w,logits,blk,step_j,b_mod,start,temps,keys,m,pv1,pi1,pv2,pi2);break;
    default:pdl_launch_if(p,c12_markov_top2_part<8>,g,b,0,st,xx,w,logits,blk,step_j,b_mod,start,temps,keys,m,pv1,pi1,pv2,pi2);break;
  }
  c12_markov_top2_fin<<<1,256,0,st>>>(pv1,pi1,pv2,pi2,nb,m,world,rank,start,packet2);
  return int(cudaGetLastError());
}
extern "C" int glm53_choose_top2_cuda(const float* packet2,int world,int rows,long long* tok1,long long* tok2,float* margin,cudaStream_t st){
  if(rows<1||rows>64||world<1)return int(cudaErrorInvalidValue);
  choose_top2_kernel<<<1,64,0,st>>>(packet2,world,rows,tok1,tok2,margin);
  return int(cudaGetLastError());
}
