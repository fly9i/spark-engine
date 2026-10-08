// SPDX-License-Identifier: MIT
// Isolated shared-stride-only arm; original WMMA math retained.
#pragma once
template<bool Packed,bool BF16=false>
__global__ void fp8_small_padded(const float* x,const unsigned char* w,const float* scale,float* y,int m,int n,int k,bool rounded,int splits){
    using Tile=typename std::conditional<BF16,__nv_bfloat16,half>::type;
    constexpr int AS=72,CS=68;
    int part=blockIdx.z;
    int start=blockIdx.x*64,warp=threadIdx.x/32;
    __shared__ __align__(32) Tile a[16*AS],b[64*AS];
    __shared__ __align__(32) float c[16*CS];
    wmma::fragment<wmma::accumulator,16,16,16,float> accum;wmma::fill_fragment(accum,0.f);
    int span=((k+splits-1)/splits+63)/64*64;
    for(int base=part*span;base<min(k,(part+1)*span);base+=64){
        if constexpr(Packed){
            if constexpr(BF16){
                for(int i=threadIdx.x;i<16*32;i+=128){int r=i/32,col=(i%32)*2;reinterpret_cast<__nv_bfloat162*>(a)[r*(AS/2)+col/2]=r<m?__floats2bfloat162_rn(x[r*k+base+col],x[r*k+base+col+1]):__float2bfloat162_rn(0.f);}
                for(int i=threadIdx.x;i<64*32;i+=128){int r=i/32,col=(i%32)*2;auto raw=*reinterpret_cast<const __nv_fp8x2_storage_t*>(w+(long long)(start+r)*k+base+col);const __half2 decoded=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);const float2 f=__half22float2(decoded);reinterpret_cast<__nv_bfloat162*>(b)[r*(AS/2)+col/2]=__floats2bfloat162_rn(f.x,f.y);}
            }else{
            for(int i=threadIdx.x;i<16*32;i+=128){int r=i/32,col=(i%32)*2;reinterpret_cast<half2*>(a)[r*(AS/2)+col/2]=r<m?__floats2half2_rn(x[r*k+base+col],x[r*k+base+col+1]):__float2half2_rn(0.f);}
            for(int i=threadIdx.x;i<64*32;i+=128){int r=i/32,col=(i%32)*2;auto raw=*reinterpret_cast<const __nv_fp8x2_storage_t*>(w+(long long)(start+r)*k+base+col);reinterpret_cast<__half2_raw*>(b)[r*(AS/2)+col/2]=__nv_cvt_fp8x2_to_halfraw2(raw,__NV_E4M3);}
            }
        }else{
            for(int i=threadIdx.x;i<16*64;i+=128){int r=i/64,col=i%64;a[r*AS+col]=r<m?__float2half_rn(x[r*k+base+col]):__float2half_rn(0.f);}
            for(int i=threadIdx.x;i<64*64;i+=128){int r=i/64,col=i%64;b[r*AS+col]=__nv_cvt_fp8_to_halfraw(w[(long long)(start+r)*k+base+col],__NV_E4M3);}
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<64;kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,Tile,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,16,16,16,Tile,wmma::col_major> fb;
            wmma::load_matrix_sync(fa,a+kk,AS);wmma::load_matrix_sync(fb,b+warp*16*AS+kk,AS);
            wmma::mma_sync(accum,fa,fb,accum);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c+warp*16,accum,CS,wmma::mem_row_major);__syncthreads();
    for(int i=threadIdx.x;i<m*64;i+=128){int r=i/64,col=i%64;float value=c[r*CS+col];if(splits==1)value*=scale[start+col];y[(part*m+r)*n+start+col]=(rounded && splits==1)?__half2float(__float2half_rn(value)):value;}
}
