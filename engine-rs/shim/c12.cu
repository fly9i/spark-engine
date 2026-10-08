// SPDX-License-Identifier: MIT
// Proposal 2 (GLM53_C12=1): entry points of the lossless 12-bit weight format (c12.cuh).
#include "c12.cuh"
#include "c12_big.cuh"
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
