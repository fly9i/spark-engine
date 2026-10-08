// SPDX-License-Identifier: MIT
// Isolated draft: no new activation quantization. Not numerically qualified.
#pragma once
template<bool Packed,bool BF16=false,bool Pad=false>
__global__ void fp8_small_transpose(const float* x,const unsigned char* w,const float* scale,float* y,int m,int n,int k,bool rounded,int splits){
    using Tile=typename std::conditional<BF16,__nv_bfloat16,half>::type;
    constexpr int AS=Pad?72:64,CS=Pad?68:64;
    int part=blockIdx.z;
    int start=blockIdx.x*64,warp=threadIdx.x/32;
    int lane=threadIdx.x&31,group=lane>>2,tid=lane&3;
    __shared__ __align__(32) Tile a[16*AS],b[64*AS];
    __shared__ __align__(32) float c[16*CS];
    float accum[4]={0.f,0.f,0.f,0.f};
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
            // A is weight[feature,k], B is activation[token,k]^T.
            // Keep the old shared loads, casts, kk order, and split span.
            // PTX m16n8k16 A registers: (row0,k0),(row8,k0),
            // (row0,k8),(row8,k8), each containing two adjacent K values.
            const int feature=warp*16+group,col=kk+2*tid;
            const unsigned a0=*reinterpret_cast<const unsigned*>(b+feature*AS+col);
            const unsigned a1=*reinterpret_cast<const unsigned*>(b+(feature+8)*AS+col);
            const unsigned a2=*reinterpret_cast<const unsigned*>(b+feature*AS+col+8);
            const unsigned a3=*reinterpret_cast<const unsigned*>(b+(feature+8)*AS+col+8);
            const unsigned b0=*reinterpret_cast<const unsigned*>(a+group*AS+col);
            const unsigned b1=*reinterpret_cast<const unsigned*>(a+group*AS+col+8);
            if constexpr(BF16){
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(accum[0]),"+f"(accum[1]),"+f"(accum[2]),"+f"(accum[3])
                             : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }else{
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(accum[0]),"+f"(accum[1]),"+f"(accum[2]),"+f"(accum[3])
                             : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        __syncthreads();
    }
    // Restore the existing token-major c layout; 4 warps write 8*64
    // unique positions. Rows m..7 are padding and never reach y.
    const int token=2*tid,feature=warp*16+group;
    c[token*CS+feature]=accum[0];
    c[(token+1)*CS+feature]=accum[1];
    c[token*CS+feature+8]=accum[2];
    c[(token+1)*CS+feature+8]=accum[3];
    __syncthreads();
    for(int i=threadIdx.x;i<m*64;i+=128){int r=i/64,col=i%64;float value=c[r*CS+col];if(splits==1)value*=scale[start+col];y[(part*m+r)*n+start+col]=(rounded && splits==1)?__half2float(__float2half_rn(value)):value;}
}
