// SPDX-License-Identifier: MIT
// TP2 candidate: FP8 per-output-channel weights, FP16 activations, FP32 sums.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>
#include <cstdlib>
#include <cstring>
using namespace nvcuda;
template<bool BF16=false>
__global__ void fp8_gemv(const float* x,const unsigned char* w,const float* scale,float* y,int n,int k,bool rounded){
    int row=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=n)return;
    float sums[8]={};
    for(int col=lane*8;col<k;col+=256){unsigned long long packed=*reinterpret_cast<const unsigned long long*>(w+(long long)row*k+col);
        #pragma unroll
        for(int j=0;j<8;++j){half b=__nv_cvt_fp8_to_halfraw((packed>>(j*8))&255,__NV_E4M3);float a=BF16?__bfloat162float(__float2bfloat16_rn(x[col+j])):__half2float(__float2half_rn(x[col+j]));sums[j]=__fmaf_rn(a,__half2float(b),sums[j]);}
    }
    float sum=0.f;for(int j=0;j<8;++j)sum+=sums[j];for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);
    if(lane==0){sum*=scale[row];y[row]=rounded?__half2float(__float2half_rn(sum)):sum;}
}
template<bool Packed,bool BF16=false>
__global__ void fp8_small(const float* x,const unsigned char* w,const float* scale,float* y,int m,int n,int k,bool rounded,int splits){
    using Tile=typename std::conditional<BF16,__nv_bfloat16,half>::type;
    int part=blockIdx.z;
    int start=blockIdx.x*64,warp=threadIdx.x/32;
    __shared__ __align__(32) Tile a[16*64],b[64*64];
    __shared__ __align__(32) float c[16*64];
    wmma::fragment<wmma::accumulator,16,16,16,float> accum;wmma::fill_fragment(accum,0.f);
    int span=((k+splits-1)/splits+63)/64*64;
    for(int base=part*span;base<min(k,(part+1)*span);base+=64){
        if constexpr(Packed){
            if constexpr(BF16){
                for(int i=threadIdx.x;i<16*32;i+=128){int r=i/32,col=(i%32)*2;reinterpret_cast<__nv_bfloat162*>(a)[i]=r<m?__floats2bfloat162_rn(x[r*k+base+col],x[r*k+base+col+1]):__float2bfloat162_rn(0.f);}
                for(int i=threadIdx.x;i<64*32;i+=128){int r=i/32,col=(i%32)*2;auto raw=*reinterpret_cast<const __nv_fp8x2_storage_t*>(w+(long long)(start+r)*k+base+col);const __half2 decoded=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);const float2 f=__half22float2(decoded);reinterpret_cast<__nv_bfloat162*>(b)[i]=__floats2bfloat162_rn(f.x,f.y);}
            }else{
            for(int i=threadIdx.x;i<16*32;i+=128){int r=i/32,col=(i%32)*2;reinterpret_cast<half2*>(a)[i]=r<m?__floats2half2_rn(x[r*k+base+col],x[r*k+base+col+1]):__float2half2_rn(0.f);}
            for(int i=threadIdx.x;i<64*32;i+=128){int r=i/32,col=(i%32)*2;auto raw=*reinterpret_cast<const __nv_fp8x2_storage_t*>(w+(long long)(start+r)*k+base+col);reinterpret_cast<__half2_raw*>(b)[i]=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);}
            }
        }else{
            for(int i=threadIdx.x;i<16*64;i+=128){int r=i/64,col=i%64;a[i]=r<m?__float2half_rn(x[r*k+base+col]):__float2half_rn(0.f);}
            for(int i=threadIdx.x;i<64*64;i+=128){int r=i/64,col=i%64;b[i]=__nv_cvt_fp8_to_halfraw(w[(long long)(start+r)*k+base+col],__NV_E4M3);}
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<64;kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,Tile,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,16,16,16,Tile,wmma::col_major> fb;
            wmma::load_matrix_sync(fa,a+kk,64);wmma::load_matrix_sync(fb,b+warp*16*64+kk,64);
            wmma::mma_sync(accum,fa,fb,accum);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c+warp*16,accum,64,wmma::mem_row_major);__syncthreads();
    for(int i=threadIdx.x;i<m*64;i+=128){int r=i/64,col=i%64;float value=c[i];if(splits==1)value*=scale[start+col];y[(part*m+r)*n+start+col]=(rounded && splits==1)?__half2float(__float2half_rn(value)):value;}
}
#include "fp8_small_padded.cuh"
#include "fp8_skinny.cuh"
#include "fp8_small_transpose.cuh"

static bool fp8_small_transpose_enabled(int m){
    const char* flag=std::getenv("GLM53_FP8_SMALL_TRANSPOSE");
    return m>=2 && m<=8 && flag && std::strcmp(flag,"1")==0;
}

static bool fp8_skinny_enabled(int m){
    const char* flag=std::getenv("GLM53_FP8_SKINNY");
    return m>=2 && m<=16 && flag && std::strcmp(flag,"1")==0;
}
static int fp8_skinny_ks(){
    const char* v=std::getenv("GLM53_FP8_SKINNY_KS");int ks=v?std::atoi(v):8;
    return (ks==1||ks==2||ks==4||ks==8)?ks:8;
}

static bool fp8_small_pad_enabled(int m){
    const char* flag=std::getenv("GLM53_FP8_SMALL_PAD");
    return m>=2 && m<=8 && flag && std::strcmp(flag,"1")==0;
}

template<bool Packed,bool BF16=false>
static void fp8_small_dispatch(const float* x,const unsigned char* w,const float* scale,float* y,
                              int m,int n,int k,bool rounded,int splits,cudaStream_t stream){
    const dim3 grid(n/64,1,splits);
    const bool transpose=fp8_small_transpose_enabled(m),pad=fp8_small_pad_enabled(m);
    if(transpose && pad)fp8_small_transpose<Packed,BF16,true><<<grid,128,0,stream>>>(x,w,scale,y,m,n,k,rounded,splits);
    else if(transpose)fp8_small_transpose<Packed,BF16,false><<<grid,128,0,stream>>>(x,w,scale,y,m,n,k,rounded,splits);
    else if(pad)fp8_small_padded<Packed,BF16><<<grid,128,0,stream>>>(x,w,scale,y,m,n,k,rounded,splits);
    else fp8_small<Packed,BF16><<<grid,128,0,stream>>>(x,w,scale,y,m,n,k,rounded,splits);
}

template<int Rows>
__global__ void fp8_simt(const float* x,const unsigned char* w,const float* scale,float* y,int m,int n,int k,bool rounded){
    int row=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;if(row>=n)return;
    float sums[Rows][4]={};
    for(int col=lane*4;col<k;col+=128){unsigned int packed=*reinterpret_cast<const unsigned int*>(w+(long long)row*k+col);
        #pragma unroll
        for(int j=0;j<4;++j){half b=__nv_cvt_fp8_to_halfraw((packed>>(j*8))&255,__NV_E4M3);float weight=__half2float(b);
            #pragma unroll
            for(int r=0;r<Rows;++r){if(r<m){float a=__half2float(__float2half_rn(x[r*k+col+j]));sums[r][j]=__fmaf_rn(a,weight,sums[r][j]);}}
        }
    }
    #pragma unroll
    for(int r=0;r<Rows;++r){float sum=0.f;for(int j=0;j<4;++j)sum+=sums[r][j];for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);
        if(lane==0 && r<m){sum*=scale[row];y[r*n+row]=rounded?__half2float(__float2half_rn(sum)):sum;}}
}
__global__ void fp8_merge(const float* parts,const float* scale,float* y,int m,int n,int splits,bool rounded){
    int i=blockIdx.x*256+threadIdx.x;if(i>=m*n)return;float sum=0.f;for(int s=0;s<splits;++s)sum+=parts[s*m*n+i];sum*=scale[i%n];y[i]=rounded?__half2float(__float2half_rn(sum)):sum;
}
extern "C" int glm53_fp8_dense_cuda(const float* x,const void* w,const float* scales,float* y,float* scratch,int m,int n,int k,int rounded,int mode,int split_override,cudaStream_t stream){
    if(m<1||m>16||n%64||k%64)return int(cudaErrorInvalidValue);
    if(mode>0&&fp8_skinny_enabled(m)){
        // Registers-only MMA fragments, CTA split-K with fixed-order smem reduction; no merge launch.
        skinny_fp8_launch<4,0>(fp8_skinny_ks(),x,(const uint8_t*)w,scales,y,m,n,k,rounded!=0,stream);
        return int(cudaGetLastError());
    }
    if(m==1)fp8_gemv<false><<<(n+3)/4,128,0,stream>>>(x,(const unsigned char*)w,scales,y,n,k,rounded);
    else if(mode==0){
        if(m<=2)fp8_simt<2><<<(n+3)/4,128,0,stream>>>(x,(const unsigned char*)w,scales,y,m,n,k,rounded);
        else if(m<=4)fp8_simt<4><<<(n+3)/4,128,0,stream>>>(x,(const unsigned char*)w,scales,y,m,n,k,rounded);
        else if(m<=8)fp8_simt<8><<<(n+3)/4,128,0,stream>>>(x,(const unsigned char*)w,scales,y,m,n,k,rounded);
        else fp8_simt<16><<<(n+3)/4,128,0,stream>>>(x,(const unsigned char*)w,scales,y,m,n,k,rounded);
    }
    else {int splits=split_override?split_override:(mode>=2?8:1);
        if(mode>=3)fp8_small_dispatch<true>(x,(const unsigned char*)w,scales,splits>1?scratch:y,m,n,k,rounded,splits,stream);
        else fp8_small_dispatch<false>(x,(const unsigned char*)w,scales,splits>1?scratch:y,m,n,k,rounded,splits,stream);
        if(splits>1)fp8_merge<<<(m*n+255)/256,256,0,stream>>>(scratch,scales,y,m,n,splits,rounded);
    }
    return int(cudaGetLastError());
}

extern "C" int glm53_half_skinny_cuda(const float* x,const void* w,void* y,int m,int n,int k,int out,int ks,cudaStream_t stream){
    return skinny_half_launch(x,w,y,m,n,k,out,ks,stream);
}
// Drafter BF16 boundary for 2..16 rows: BF16 activations read directly, FP32 unrounded output.
extern "C" int glm53_fp8_bf16_skinny_cuda(const void* x_bf16,const void* w,const float* scales,float* y,int m,int n,int k,cudaStream_t stream){
    // Each row's fragments/split-K reduction do not depend on the row-tile count: rows 17..32 (NT=4)
    // produce, per row, the same value as a <=16-row call (batched drafter proposals, C2).
    if(m<2||m>32||n%16||k%64)return int(cudaErrorInvalidValue);
    skinny_fp8_launch<4,2>(fp8_skinny_ks(),x_bf16,(const uint8_t*)w,scales,y,m,n,k,false,stream);
    return int(cudaGetLastError());
}
extern "C" int glm53_fp8_bf16_cuda(const float* x,const void* w,const float* scales,float* y,float* scratch,int m,int n,int k,int splits,cudaStream_t stream){
    if(m<1||m>16||n%64||k%64||!(splits==1||splits==2||splits==4||splits==8||splits==16))return int(cudaErrorInvalidValue);
    if(m==1)fp8_gemv<true><<<(n+3)/4,128,0,stream>>>(x,(const unsigned char*)w,scales,y,n,k,false);
    else {
        fp8_small_dispatch<true,true>(x,(const unsigned char*)w,scales,splits>1?scratch:y,m,n,k,false,splits,stream);
        if(splits>1)fp8_merge<<<(m*n+255)/256,256,0,stream>>>(scratch,scales,y,m,n,splits,false);
    }
    return int(cudaGetLastError());
}

// Large-M candidate: reconstruct only the shared tile consumed immediately by
// Tensor Cores. 64 query rows share each FP8 weight load, no full FP16 matrix.
__global__ void fp8_large(const float* x,const unsigned char* w,const float* scale,float* y,int m,int n,int k,bool rounded){
    int warp=threadIdx.x/32,m0=blockIdx.y*64,n0=blockIdx.x*64;
    __shared__ __align__(32) half a[64*64],b[64*64];
    __shared__ __align__(32) float c[64*64];
    wmma::fragment<wmma::accumulator,16,16,16,float> accum;wmma::fill_fragment(accum,0.f);
    for(int base=0;base<k;base+=64){
        for(int i=threadIdx.x;i<64*32;i+=512){int row=i/32,col=(i%32)*2;
            reinterpret_cast<half2*>(a)[i]=m0+row<m?__floats2half2_rn(x[(m0+row)*k+base+col],x[(m0+row)*k+base+col+1]):__float2half2_rn(0.f);
            auto raw=*reinterpret_cast<const __nv_fp8x2_storage_t*>(w+(long long)(n0+row)*k+base+col);
            reinterpret_cast<__half2_raw*>(b)[i]=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<64;kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> fb;
            wmma::load_matrix_sync(fa,a+(warp/4)*16*64+kk,64);wmma::load_matrix_sync(fb,b+(warp%4)*16*64+kk,64);
            wmma::mma_sync(accum,fa,fb,accum);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c+(warp/4)*16*64+(warp%4)*16,accum,64,wmma::mem_row_major);__syncthreads();
    for(int i=threadIdx.x;i<64*64;i+=512){int row=m0+i/64,col=n0+i%64;if(row<m){float value=c[i]*scale[col];y[row*n+col]=rounded?__half2float(__float2half_rn(value)):value;}}
}
extern "C" int glm53_fp8_large_cuda(const float* x,const void* w,const float* scale,float* y,int m,int n,int k,int rounded,cudaStream_t stream){
    if(m<1||n%64||k%64)return int(cudaErrorInvalidValue);
    fp8_large<<<dim3(n/64,(m+63)/64),512,0,stream>>>(x,(const unsigned char*)w,scale,y,m,n,k,rounded);
    return int(cudaGetLastError());
}

// Rework the first large-M attempt: four accumulator fragments per warp,
// 128 output columns and a union that reuses staging memory for the epilogue.
// K=128 also halves CTA barriers; benchmark occupancy against K=64.
template<int KTile>
__global__ void fp8_large_reuse(const float* x,const unsigned char* w,const float* scale,float* y,int m,int n,int k,bool rounded){
    int warp=threadIdx.x/32,m0=blockIdx.y*64,n0=blockIdx.x*128;
    __shared__ __align__(32) union {half staging[(64+128)*KTile];float output[64*128];} mem;
    half* a=mem.staging;half* b=mem.staging+64*KTile;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][2];
    #pragma unroll
    for(int r=0;r<2;++r)for(int c=0;c<2;++c)wmma::fill_fragment(acc[r][c],0.f);
    for(int base=0;base<k;base+=KTile){
        for(int i=threadIdx.x;i<64*KTile/2;i+=256){int row=i/(KTile/2),col=(i%(KTile/2))*2;
            reinterpret_cast<half2*>(a)[i]=m0+row<m?__floats2half2_rn(x[(m0+row)*k+base+col],x[(m0+row)*k+base+col+1]):__float2half2_rn(0.f);}
        for(int i=threadIdx.x;i<128*KTile/2;i+=256){int row=i/(KTile/2),col=(i%(KTile/2))*2;
            auto raw=*reinterpret_cast<const __nv_fp8x2_storage_t*>(w+(long long)(n0+row)*k+base+col);
            reinterpret_cast<__half2_raw*>(b)[i]=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);}
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<KTile;kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> fb[2];
            #pragma unroll
            for(int r=0;r<2;++r)wmma::load_matrix_sync(fa[r],a+((warp/4)*32+r*16)*KTile+kk,KTile);
            #pragma unroll
            for(int c=0;c<2;++c)wmma::load_matrix_sync(fb[c],b+((warp%4)*32+c*16)*KTile+kk,KTile);
            #pragma unroll
            for(int r=0;r<2;++r)for(int c=0;c<2;++c)wmma::mma_sync(acc[r][c],fa[r],fb[c],acc[r][c]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<2;++r)for(int c=0;c<2;++c)wmma::store_matrix_sync(mem.output+((warp/4)*32+r*16)*128+(warp%4)*32+c*16,acc[r][c],128,wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<64*128;i+=256){int row=m0+i/128,col=n0+i%128;if(row<m){float value=mem.output[i]*scale[col];y[row*n+col]=rounded?__half2float(__float2half_rn(value)):value;}}
}
extern "C" int glm53_fp8_large_reuse_cuda(const float* x,const void* w,const float* scale,float* y,int m,int n,int k,int rounded,int mode,cudaStream_t stream){
    if(m<1||n%128||k%128)return int(cudaErrorInvalidValue);
    if(mode==3)fp8_large_reuse<64><<<dim3(n/128,(m+63)/64),256,0,stream>>>(x,(const unsigned char*)w,scale,y,m,n,k,rounded);
    else fp8_large_reuse<128><<<dim3(n/128,(m+63)/64),256,0,stream>>>(x,(const unsigned char*)w,scale,y,m,n,k,rounded);
    return int(cudaGetLastError());
}

