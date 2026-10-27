// SPDX-License-Identifier: MIT
// DeepSeek-V4.1-Flash decode kernels (docs/07 D4): weights in the checkpoint formats, read once per call.
//
// The official GEMMs (inference/kernel.py fp8_gemm / fp4_gemm) multiply FP8 (or FP4) operands, sum each 32-wide K block
// in FP32 and apply the two power-of-two block scales. A power-of-two factor scales exactly, so dequantizing both
// operands on the fly and accumulating products in FP32 gives the same values except the order in which the K blocks
// are summed (L1). The activation is quantized beforehand (ds41_act_quant, UE8M0 per 32) like act_quant.
//
// ds41_fp8_gemv: W E4M3 [n, k] row-major, scales E8M0 [ceil(n/32), k/32]; y[m, n] = x[m, :] . W[n, :]^T (m <= 8).
// ds41_mxfp4_gemv: W E2M1 pairs [n, k/2] (low nibble first), scales E8M0 [n, k/32]; same contract.
// One warp per output row; every lane reads 16 bytes per step (16 FP8 or 32 FP4 values: always inside one scale block).
#include "ds41_pdl.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

#ifndef FP8_U
#define FP8_U 4
#endif
namespace {
__device__ __forceinline__ float e8m0(uint8_t b){return __uint_as_float((unsigned)b<<23);}   // 2^(b-127)
__device__ __constant__ float FP4[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};

// xs: elements between input rows; rpg > 0: grouped (block-diagonal) weight, row `row` reads input group row / rpg
// (the group's k elements at offset (row / rpg) * k of the input row): wo_a's groups in one launch
template<int M>
__global__ void fp8_gemv(const __nv_bfloat16* __restrict__ x,const uint8_t* __restrict__ w,const uint8_t* __restrict__ ws,
                         float* __restrict__ y,int m,int n,int k,int xs,int rpg){
  pdl_entry();
  const int lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
  if(row>=n)return;
  const uint8_t* wr=w+(long long)row*k;const uint8_t* sr=ws+(long long)(row>>5)*(k>>5);
  float acc[M];
  #pragma unroll
  for(int r=0;r<M;++r)acc[r]=0.f;
  // FP8_U independent 16-byte loads per lane in flight (columns c, c + 512, ...)
  constexpr int U=FP8_U;
  for(int c0=lane*16;c0<k;c0+=512*U){
    uint4 p[U];float s[U];bool ok[U];
    #pragma unroll
    for(int u=0;u<U;++u){const int c=c0+u*512;ok[u]=c<k;if(ok[u]){p[u]=__ldg(reinterpret_cast<const uint4*>(wr+c));s[u]=e8m0(__ldg(sr+(c>>5)));}}
    #pragma unroll
    for(int u=0;u<U;++u){
      if(!ok[u])continue;
      const int c=c0+u*512;
      const uint8_t* b=reinterpret_cast<const uint8_t*>(&p[u]);
      float wv[16];
      #pragma unroll
      for(int j=0;j<16;j+=2){
        __nv_fp8x2_storage_t two=(__nv_fp8x2_storage_t)(b[j]|(b[j+1]<<8));
        const __half2_raw h=__nv_cvt_fp8x2_to_halfraw2(two,__NV_E4M3);
        const float2 f=__half22float2(*reinterpret_cast<const __half2*>(&h));
        wv[j]=f.x*s[u];wv[j+1]=f.y*s[u];
      }
      #pragma unroll
      for(int r=0;r<M;++r){
        if(r<m){
          const __nv_bfloat16* xr=x+(long long)r*xs+(rpg?(long long)(row/rpg)*k:0)+c;
          #pragma unroll
          for(int j=0;j<16;++j)acc[r]=__fmaf_rn(__bfloat162float(xr[j]),wv[j],acc[r]);
        }
      }
    }
  }
  #pragma unroll
  for(int r=0;r<M;++r){
    float v=acc[r];
    for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o);
    if(lane==0&&r<m)y[(long long)r*n+row]=v;
  }
}

template<int M>
__global__ void mxfp4_gemv(const __nv_bfloat16* __restrict__ x,const uint8_t* __restrict__ w,const uint8_t* __restrict__ ws,
                           float* __restrict__ y,int m,int n,int k){
  const int lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
  if(row>=n)return;
  const uint8_t* wr=w+(long long)row*(k>>1);const uint8_t* sr=ws+(long long)row*(k>>5);
  float acc[M];
  #pragma unroll
  for(int r=0;r<M;++r)acc[r]=0.f;
  for(int c=lane*32;c<k;c+=1024){
    const uint4 p=*reinterpret_cast<const uint4*>(wr+(c>>1));
    const float s=e8m0(sr[c>>5]);
    const uint8_t* b=reinterpret_cast<const uint8_t*>(&p);
    float wv[32];
    #pragma unroll
    for(int j=0;j<16;++j){   // hardware E2M1x2 -> FP16x2 (low nibble first), exact
      const __half2_raw h=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b[j],__NV_E2M1);
      const float2 f=__half22float2(*reinterpret_cast<const __half2*>(&h));
      wv[2*j]=f.x*s;wv[2*j+1]=f.y*s;
    }
    #pragma unroll
    for(int r=0;r<M;++r){
      if(r<m){
        const __nv_bfloat16* xr=x+(long long)r*k+c;
        #pragma unroll
        for(int j=0;j<32;++j)acc[r]=__fmaf_rn(__bfloat162float(xr[j]),wv[j],acc[r]);
      }
    }
  }
  #pragma unroll
  for(int r=0;r<M;++r){
    float v=acc[r];
    for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o);
    if(lane==0&&r<m)y[(long long)r*n+row]=v;
  }
}

// bf16_gemv: W BF16 [n, k], y[m, n] FP32 = x[m, :] . W[n, :] (x BF16, m <= 8): exact products, FP32 sums, i.e. the
// official FP32 head (x.float() @ W.float()) up to summation order, without materializing W in FP32.
template<int M>
__global__ void bf16_gemv(const __nv_bfloat16* __restrict__ x,const __nv_bfloat16* __restrict__ w,float* __restrict__ y,int m,int n,int k){
  pdl_entry();
  const int lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
  if(row>=n)return;
  const __nv_bfloat16* wr=w+(long long)row*k;
  float acc[M];
  #pragma unroll
  for(int r=0;r<M;++r)acc[r]=0.f;
  for(int c=lane*8;c<k;c+=256){
    const uint4 p=*reinterpret_cast<const uint4*>(wr+c);
    const __nv_bfloat162* b=reinterpret_cast<const __nv_bfloat162*>(&p);
    float wv[8];
    #pragma unroll
    for(int j=0;j<4;++j){const float2 f=__bfloat1622float2(b[j]);wv[2*j]=f.x;wv[2*j+1]=f.y;}
    #pragma unroll
    for(int r=0;r<M;++r){
      if(r<m){
        const uint4 xp=*reinterpret_cast<const uint4*>(x+(long long)r*k+c);
        const __nv_bfloat162* xb=reinterpret_cast<const __nv_bfloat162*>(&xp);
        #pragma unroll
        for(int j=0;j<4;++j){const float2 f=__bfloat1622float2(xb[j]);acc[r]=__fmaf_rn(f.x,wv[2*j],acc[r]);acc[r]=__fmaf_rn(f.y,wv[2*j+1],acc[r]);}
      }
    }
  }
  #pragma unroll
  for(int r=0;r<M;++r){
    float v=acc[r];
    for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o);
    if(lane==0&&r<m)y[(long long)r*n+row]=v;
  }
}

// act_quant inplace (kernel.py, UE8M0 scales per 32): x BF16 [rows, k] -> quantized-dequantized BF16 out.
__global__ void act_quant32(const __nv_bfloat16* __restrict__ x,__nv_bfloat16* __restrict__ out,long long blocks){
  pdl_entry();
  const long long b=(long long)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
  const int lane=threadIdx.x&31;
  if(b>=blocks)return;
  const float v=__bfloat162float(x[b*32+lane]);
  float a=fabsf(v);
  for(int o=16;o;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffffu,a,o));
  a=fmaxf(a,1e-4f);
  const float t=a*(1.0f/448.0f);
  const unsigned bits=__float_as_uint(t);
  const int e=(int)((bits>>23)&0xFF)-127+((bits&0x7FFFFF)!=0);
  const float s=__uint_as_float((unsigned)(e+127)<<23);
  const float q=fminf(fmaxf(v/s,-448.f),448.f);
  const __nv_fp8_e4m3 f8(q);
  out[b*32+lane]=__float2bfloat16(float(f8)*s);
}

// ds41_fp8_mm (prefill, m > 8): same contract as ds41_fp8_gemv with BF16 MMA (m16n8k16, FP32 accumulation). E4M3 and
// the act-quantized activations are exact in BF16 and the scales are powers of two, so every product is exact (L1 vs
// the dequantize + FP32 reference). CTA: 64 rows x 128 columns, 4 warps of 32 columns (4 n8 tiles), 4 m16 tiles. The
// lane's fragment k mapping: physical k = 64 s + 16 q + 4 i + {0..3} (q = lane % 4, substep i) for A and B alike, so a
// lane reads 16 contiguous weight bytes (inside one 32-wide scale block) per 64-k step. Grid x = row tiles (fastest:
// CTAs sharing a weight tile run together), y = column tiles.
__device__ __forceinline__ void mma16816(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
// two E4M3 bytes (low byte = lower k) times a power of two -> BF16x2
__device__ __forceinline__ uint32_t e4m3x2_bf16(uint16_t two,float s){
  const __half2_raw h=__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two,__NV_E4M3);
  const float2 f=__half22float2(*reinterpret_cast<const __half2*>(&h));
  __nv_bfloat162 r=__floats2bfloat162_rn(f.x*s,f.y*s);
  return *reinterpret_cast<uint32_t*>(&r);
}
// B16: BF16 output (the FP32 sum rounded to nearest even, as converting it); xs / ys: input / output row strides
// (elements); grid z: groups of a grouped weight (group z: input elements [z k, +k) of each row, weight rows [z n, +n),
// output columns [z n, +n))
template<bool B16>
__global__ __launch_bounds__(128) void fp8_mm(const __nv_bfloat16* __restrict__ x,const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ ws,void* __restrict__ yv,int m,int n,int k,int xs,int ys){
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,q=lane&3;
  x+=(long long)blockIdx.z*k;w+=(long long)blockIdx.z*n*k;ws+=(long long)blockIdx.z*(n>>5)*(k>>5);
  const long long y0=(long long)blockIdx.z*n;
  const int r0=blockIdx.x*64,n0=blockIdx.y*128+warp*32;
  const uint8_t* wr[4];const uint8_t* sr[4];bool nok[4];
  for(int t=0;t<4;++t){
    const int row=n0+8*t+g;nok[t]=row<n;const int rr=nok[t]?row:0;
    wr[t]=w+(long long)rr*k+q*16;sr[t]=ws+(long long)(rr>>5)*(k>>5);
  }
  const __nv_bfloat16* xr[4][2];
  for(int mt=0;mt<4;++mt)for(int hh=0;hh<2;++hh){const int r=r0+mt*16+hh*8+g;xr[mt][hh]=x+(long long)(r<m?r:0)*xs+q*16;}
  float acc[4][4][4];
  for(int a=0;a<4;++a)for(int b=0;b<4;++b)for(int c=0;c<4;++c)acc[a][b][c]=0.f;
  for(int s=0;s<k/64;++s){
    uint4 bw[4];float sc[4];
    for(int t=0;t<4;++t){bw[t]=__ldg(reinterpret_cast<const uint4*>(wr[t]+s*64));sc[t]=e8m0(__ldg(sr[t]+((s*64+q*16)>>5)));}
#pragma unroll
    for(int i=0;i<4;++i){
      uint2 a[4][2];
      for(int mt=0;mt<4;++mt)for(int hh=0;hh<2;++hh)a[mt][hh]=__ldg(reinterpret_cast<const uint2*>(xr[mt][hh]+s*64+4*i));
#pragma unroll
      for(int t=0;t<4;++t){
        const uint8_t* b=reinterpret_cast<const uint8_t*>(&bw[t]);
        const uint32_t b0=e4m3x2_bf16((uint16_t)(b[4*i]|(b[4*i+1]<<8)),sc[t]),b1=e4m3x2_bf16((uint16_t)(b[4*i+2]|(b[4*i+3]<<8)),sc[t]);
        for(int mt=0;mt<4;++mt)mma16816(acc[mt][t],a[mt][0].x,a[mt][1].x,a[mt][0].y,a[mt][1].y,b0,b1);
      }
    }
  }
  for(int mt=0;mt<4;++mt)for(int hh=0;hh<2;++hh){
    const int r=r0+mt*16+hh*8+g;if(r>=m)continue;
    for(int t=0;t<4;++t){
      const int c=n0+8*t+2*q;if(!nok[t])continue;
      if(B16)*reinterpret_cast<__nv_bfloat162*>((__nv_bfloat16*)yv+y0+(long long)r*ys+c)=__floats2bfloat162_rn(acc[mt][t][hh*2],acc[mt][t][hh*2+1]);
      else *reinterpret_cast<float2*>((float*)yv+y0+(long long)r*ys+c)=make_float2(acc[mt][t][hh*2],acc[mt][t][hh*2+1]);
    }
  }
}
} // namespace

#define DS41_GEMV(NAME,KERNEL,KALIGN) \
extern "C" int NAME(const void* x,const void* w,const void* ws,float* y,int m,int n,int k,cudaStream_t st){ \
  if(m<1||m>8||k%(KALIGN)!=0)return 1; \
  const int warps=8;dim3 g((n+warps-1)/warps),b(32*warps); \
  const __nv_bfloat16* xx=(const __nv_bfloat16*)x;const uint8_t* ww=(const uint8_t*)w;const uint8_t* ss=(const uint8_t*)ws; \
  switch(m){case 1:KERNEL<1><<<g,b,0,st>>>(xx,ww,ss,y,m,n,k);break;case 2:KERNEL<2><<<g,b,0,st>>>(xx,ww,ss,y,m,n,k);break; \
    case 3:case 4:KERNEL<4><<<g,b,0,st>>>(xx,ww,ss,y,m,n,k);break;default:KERNEL<8><<<g,b,0,st>>>(xx,ww,ss,y,m,n,k);} \
  return (int)cudaGetLastError();}
DS41_GEMV(ds41_mxfp4_gemv,mxfp4_gemv,32)

// y[m, n] = x . W^T for FP8 W [n, k]; xs = input row stride (elements), rpg = rows per group of a grouped weight (0: no
// groups; row reads input elements [(row / rpg) * k, +k) of its input row)
extern "C" int ds41_fp8_gemv_ex(const void* x,const void* w,const void* ws,float* y,int m,int n,int k,int xs,int rpg,cudaStream_t st){
  if(m<1||m>8||k%32!=0||(rpg&&rpg%32))return 1;
  const int warps=8;dim3 g((n+warps-1)/warps),b(32*warps);
  const __nv_bfloat16* xx=(const __nv_bfloat16*)x;const uint8_t* ww=(const uint8_t*)w;const uint8_t* ss=(const uint8_t*)ws;
  switch(m){case 1:pdl_launch(fp8_gemv<1>,g,b,0,st,xx,ww,ss,y,m,n,k,xs,rpg);break;case 2:pdl_launch(fp8_gemv<2>,g,b,0,st,xx,ww,ss,y,m,n,k,xs,rpg);break;
    case 3:case 4:pdl_launch(fp8_gemv<4>,g,b,0,st,xx,ww,ss,y,m,n,k,xs,rpg);break;default:pdl_launch(fp8_gemv<8>,g,b,0,st,xx,ww,ss,y,m,n,k,xs,rpg);}
  return (int)cudaGetLastError();
}
extern "C" int ds41_fp8_gemv(const void* x,const void* w,const void* ws,float* y,int m,int n,int k,cudaStream_t st){
  return ds41_fp8_gemv_ex(x,w,ws,y,m,n,k,k,0,st);
}

extern "C" int ds41_bf16_gemv(const void* x,const void* w,float* y,int m,int n,int k,cudaStream_t st){
  if(m<1||m>8||k%8!=0)return 1;
  const int warps=8;dim3 g((n+warps-1)/warps),b(32*warps);
  const __nv_bfloat16* xx=(const __nv_bfloat16*)x;const __nv_bfloat16* ww=(const __nv_bfloat16*)w;
  switch(m){case 1:pdl_launch(bf16_gemv<1>,g,b,0,st,xx,ww,y,m,n,k);break;case 2:pdl_launch(bf16_gemv<2>,g,b,0,st,xx,ww,y,m,n,k);break;
    case 3:case 4:pdl_launch(bf16_gemv<4>,g,b,0,st,xx,ww,y,m,n,k);break;default:pdl_launch(bf16_gemv<8>,g,b,0,st,xx,ww,y,m,n,k);}
  return (int)cudaGetLastError();
}

extern "C" int ds41_fp8_mm(const void* x,const void* w,const void* ws,float* y,int m,int n,int k,cudaStream_t st){
  if(m<1||k%64!=0||n%8!=0)return 1;
  fp8_mm<false><<<dim3((m+63)/64,(n+127)/128),128,0,st>>>((const __nv_bfloat16*)x,(const uint8_t*)w,(const uint8_t*)ws,y,m,n,k,k,n);
  return (int)cudaGetLastError();
}
// grouped (wo_a): y BF16 [m, groups n] = per group g: x[:, g k .. +k] (row stride xs) . W[g n .. +n]^T, rounded to BF16
// (each output the same MMA sequence as ds41_fp8_mm on that group's contiguous input, then the BF16 conversion)
extern "C" int ds41_fp8_mm_grouped_bf16(const void* x,const void* w,const void* ws,void* y,int m,int n,int k,int groups,int xs,cudaStream_t st){
  if(m<1||k%64!=0||n%32!=0||groups<1||xs<groups*k)return 1;
  fp8_mm<true><<<dim3((m+63)/64,(n+127)/128,groups),128,0,st>>>((const __nv_bfloat16*)x,(const uint8_t*)w,(const uint8_t*)ws,y,m,n,k,xs,groups*n);
  return (int)cudaGetLastError();
}

extern "C" int ds41_act_quant32(const void* x,void* out,long long numel,cudaStream_t st){
  if(numel%32)return 1;const long long blocks=numel/32;
  pdl_launch(act_quant32,dim3((unsigned)((blocks+7)/8)),dim3(256),0,st,(const __nv_bfloat16*)x,(__nv_bfloat16*)out,blocks);
  return (int)cudaGetLastError();
}
extern "C" int ds41_pdl_enabled(){return pdl_on()?1:0;}
