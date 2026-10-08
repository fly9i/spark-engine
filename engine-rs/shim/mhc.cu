// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdint>

// One 16-lane group per 4x4 matrix. Keep every lane active, including padding.
// Input is the existing FP32 softmax + epsilon; only the 39 normalizations fuse.
__global__ void sinkhorn4(const float* input, float* output, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < rows * 16 ? input[i] : 0.0f;
    for (int step = 0; step < 39; ++step) {
        int stride = (step % 2 == 0) ? 4 : 1; // column, row, column, ...
        float sum = v + __shfl_xor_sync(0xffffffff, v, stride, 16);
        sum += __shfl_xor_sync(0xffffffff, sum, 2 * stride, 16);
        v = v / (sum + 1.0e-6f);
    }
    if (i < rows * 16) output[i] = v;
}

extern "C" int glm53_sinkhorn_cuda(const float* input, float* output, int rows,
                                    cudaStream_t stream) {
    if (rows <= 0) return 0;
    sinkhorn4<<<(rows + 7) / 8, 128, 0, stream>>>(input, output, rows);
    return static_cast<int>(cudaGetLastError());
}

// Contract the four residual streams and add the new branch in one pass.
// Explicit strides support the initial broadcast embedding without materializing it.
__global__ void mhc_post_kernel(const float* x,const float* residual,const float* comb,
                                const float* post,float* out,int rows,int hidden,
                                long long stride0,long long stride1,long long stride2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*4*hidden)return;
    int col=i%hidden,stream=(i/hidden)%4,row=i/(hidden*4);
    float mixed=0.f;
    #pragma unroll
    for(int k=0;k<4;++k) {
        float r=residual[row*stride0+k*stride1+col*stride2];
        mixed=__fmaf_rn(comb[row*16+k*4+stream],r,mixed);
    }
    out[i]=mixed+post[row*4+stream]*x[row*hidden+col];
}
extern "C" int glm53_mhc_post_cuda(const float* x,const float* residual,const float* comb,const float* post,
                                   float* out,int rows,int hidden,long long s0,long long s1,long long s2,cudaStream_t stream) {
    mhc_post_kernel<<<(rows*4*hidden+255)/256,256,0,stream>>>(x,residual,comb,post,out,rows,hidden,s0,s1,s2);
    return static_cast<int>(cudaGetLastError());
}

// PACKED is two independent, already globally reduced FP32 lanes. The shared
// lane's Half round belongs AFTER TP SUM and BEFORE addition to the routed lane.
template<bool PACKED, bool ROUND_SHARED>
__device__ __forceinline__ float post_branch(const float* x,int i,int count) {
    if constexpr (!PACKED) return x[i];
    float shared=x[count+i];
    if constexpr (ROUND_SHARED) shared=__half2float(__float2half_rn(shared));
    return __fadd_rn(x[i],shared);
}

// One thread owns all four output streams for (row,col). Reuse each residual
// value across four independent accumulators, preserving the old k=0..3 order.
// Output must not alias residual: every output stream consumes all four inputs.
template<bool PACKED, bool ROUND_SHARED>
__global__ void mhc_post_four_kernel(const float* x,const float* residual,const float* comb,
                                    const float* post,float* out,int rows,int hidden,
                                    long long stride0,long long stride1,long long stride2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*hidden)return;
    int row=i/hidden,col=i%hidden;
    float mixed[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int k=0;k<4;++k) {
        float r=residual[row*stride0+k*stride1+col*stride2];
        #pragma unroll
        for(int stream=0;stream<4;++stream)
            mixed[stream]=__fmaf_rn(comb[row*16+k*4+stream],r,mixed[stream]);
    }
    float branch=post_branch<PACKED,ROUND_SHARED>(x,i,rows*hidden);
    #pragma unroll
    for(int stream=0;stream<4;++stream)
        out[(row*4+stream)*hidden+col]=__fadd_rn(mixed[stream],__fmul_rn(post[row*4+stream],branch));
}

// Independent packed-consumer arm with the old thread-to-output mapping.
// This separates removing the branch tensor from reusing four-stream inputs.
template<bool ROUND_SHARED>
__global__ void mhc_post_packed_kernel(const float* packed,const float* residual,const float* comb,
                                      const float* post,float* out,int rows,int hidden,
                                      long long stride0,long long stride1,long long stride2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*4*hidden)return;
    int col=i%hidden,stream=(i/hidden)%4,row=i/(hidden*4);
    float mixed=0.f;
    #pragma unroll
    for(int k=0;k<4;++k) {
        float r=residual[row*stride0+k*stride1+col*stride2];
        mixed=__fmaf_rn(comb[row*16+k*4+stream],r,mixed);
    }
    float branch=post_branch<true,ROUND_SHARED>(packed,row*hidden+col,rows*hidden);
    out[i]=__fadd_rn(mixed,__fmul_rn(post[row*4+stream],branch));
}

extern "C" int glm53_mhc_post_four_cuda(const float* x,const float* residual,const float* comb,const float* post,
                                        float* out,int rows,int hidden,long long s0,long long s1,long long s2,cudaStream_t stream) {
    if(rows<=0 || hidden<=0)return int(cudaErrorInvalidValue);
    mhc_post_four_kernel<false,false><<<(rows*hidden+255)/256,256,0,stream>>>(x,residual,comb,post,out,rows,hidden,s0,s1,s2);
    return static_cast<int>(cudaGetLastError());
}

template<bool ROUND_SHARED>
int launch_post_packed(const float* packed,const float* residual,const float* comb,const float* post,
                       float* out,int rows,int hidden,long long s0,long long s1,long long s2,bool four,cudaStream_t stream) {
    if(four)
        mhc_post_four_kernel<true,ROUND_SHARED><<<(rows*hidden+255)/256,256,0,stream>>>(packed,residual,comb,post,out,rows,hidden,s0,s1,s2);
    else
        mhc_post_packed_kernel<ROUND_SHARED><<<(rows*4*hidden+255)/256,256,0,stream>>>(packed,residual,comb,post,out,rows,hidden,s0,s1,s2);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int glm53_mhc_post_packed_cuda(const float* packed,const float* residual,const float* comb,const float* post,
                                          float* out,int rows,int hidden,long long s0,long long s1,long long s2,
                                          int round_shared,int four_streams,cudaStream_t stream) {
    if(rows<=0 || hidden<=0 || (round_shared!=0 && round_shared!=1) || (four_streams!=0 && four_streams!=1))return int(cudaErrorInvalidValue);
    if(round_shared)return launch_post_packed<true>(packed,residual,comb,post,out,rows,hidden,s0,s1,s2,four_streams!=0,stream);
    return launch_post_packed<false>(packed,residual,comb,post,out,rows,hidden,s0,s1,s2,four_streams!=0,stream);
}


// I3 step 2 (GLM53_AR_FUSED=1): the four-stream post as the consumer of a send-only allreduce (rdma_ar.cu). x is this
// rank's partial; the peer's partial is read from its recv slot (seq % nslot, seq read on device so graph replays follow
// the ring). The sum, its rounding and the post arithmetic are the separate path's, in its order: rank0+rank1 (IEEE add
// is commutative, so either rank's mine+peer equals the reduction kernel's), then for PACKED the shared lane's optional
// Half round and routed+shared, else the o-projection's optional Half round (round_row_output), then the post.
template<bool PACKED, bool ROUND>
__global__ void mhc_post_four_ar_kernel(const float* x,const float* __restrict__ recv_base,const unsigned long long* seq,int slot_floats,int nslot,
                                       const float* residual,const float* comb,const float* post,float* out,int rows,int hidden,
                                       long long stride0,long long stride1,long long stride2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*hidden)return;
    int row=i/hidden,col=i%hidden;
    const float* peer=recv_base+(long long)(*(volatile const unsigned long long*)seq%(unsigned long long)nslot)*slot_floats;
    float mixed[4]={0.f,0.f,0.f,0.f};
    #pragma unroll
    for(int k=0;k<4;++k) {
        float r=residual[row*stride0+k*stride1+col*stride2];
        #pragma unroll
        for(int stream=0;stream<4;++stream)
            mixed[stream]=__fmaf_rn(comb[row*16+k*4+stream],r,mixed[stream]);
    }
    float branch=__fadd_rn(x[i],__ldcv(peer+i));
    if constexpr (PACKED) {
        const int c=rows*hidden;float shared=__fadd_rn(x[c+i],__ldcv(peer+c+i));
        if constexpr (ROUND) shared=__half2float(__float2half_rn(shared));
        branch=__fadd_rn(branch,shared);
    } else if constexpr (ROUND) branch=__half2float(__float2half_rn(branch));
    #pragma unroll
    for(int stream=0;stream<4;++stream)
        out[(row*4+stream)*hidden+col]=__fadd_rn(mixed[stream],__fmul_rn(post[row*4+stream],branch));
}
extern "C" int glm53_rdma_ar_peer(const float**,const unsigned long long**,int*,int*);
extern "C" int glm53_mhc_post_four_ar_cuda(const float* x,const float* residual,const float* comb,const float* post,float* out,
                                           int rows,int hidden,long long s0,long long s1,long long s2,int packed,int round,cudaStream_t stream) {
    if(rows<=0 || hidden<=0 || (packed!=0 && packed!=1) || (round!=0 && round!=1))return int(cudaErrorInvalidValue);
    const float* recv=nullptr;const unsigned long long* seq=nullptr;int slot_floats=0,nslot=0;
    if(glm53_rdma_ar_peer(&recv,&seq,&slot_floats,&nslot))return int(cudaErrorNotReady);
    if((long long)rows*hidden*(packed?2:1)>slot_floats)return int(cudaErrorInvalidValue);
    const unsigned blocks=(rows*hidden+255)/256;
#define POST_AR(P,R) mhc_post_four_ar_kernel<P,R><<<blocks,256,0,stream>>>(x,recv,seq,slot_floats,nslot,residual,comb,post,out,rows,hidden,s0,s1,s2)
    if(packed){if(round)POST_AR(true,true);else POST_AR(true,false);}else{if(round)POST_AR(false,true);else POST_AR(false,false);}
#undef POST_AR
    return static_cast<int>(cudaGetLastError());
}

// M4: weight loads from FP32 or BF16 storage. A BF16 value widened to FP32 is exact (low 16 bits
// zero), so every kernel sees the very same FP32 operand as with FP32-resident weights (L0).
#include <cuda_bf16.h>
__device__ __forceinline__ float4 mw4(const float* p){return *reinterpret_cast<const float4*>(p);}
__device__ __forceinline__ float4 mw4(const __nv_bfloat16* p){const uint2 u=*reinterpret_cast<const uint2*>(p);
  return make_float4(__uint_as_float(u.x<<16),__uint_as_float(u.x&0xffff0000u),__uint_as_float(u.y<<16),__uint_as_float(u.y&0xffff0000u));}
__device__ __forceinline__ float mw1(const float* p){return *p;}
__device__ __forceinline__ float mw1(const __nv_bfloat16* p){return __uint_as_float(((unsigned)*reinterpret_cast<const unsigned short*>(p))<<16);}
// ---- W05 fused mhc_pre (GLM53_MHC_PRE_FUSED=1), rows <= 16, contiguous FP32 inputs ----
// Kernel 1: K-sliced partial sums of the 24 mixes and the square sum for every row.
// partial layout [slice][row][25]; slice-order reduction happens in kernel 2 (fixed order).
constexpr int MHC_PRE_SLICE=128;
template<typename W>
__global__ void mhc_pre_partial(const float* __restrict__ x,const W* __restrict__ fn,float* __restrict__ partial,int rows,int k){
    const int slice=blockIdx.x,r=blockIdx.y,tid=threadIdx.x,col=slice*MHC_PRE_SLICE+tid;
    __shared__ float red[4][25];
    const float v=x[(long long)r*k+col];
    float acc[25];
#pragma unroll
    for(int o=0;o<24;++o)acc[o]=v*mw1(fn+(long long)o*k+col);
    acc[24]=v*v;
#pragma unroll
    for(int o=0;o<25;++o){float s=acc[o];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);acc[o]=s;}
    if((tid&31)==0){
#pragma unroll
        for(int o=0;o<25;++o)red[tid>>5][o]=acc[o];
    }
    __syncthreads();
    if(tid<25)partial[((long long)slice*rows+r)*25+tid]=(red[0][tid]+red[1][tid])+(red[2][tid]+red[3][tid]);
}
__device__ __forceinline__ float mhc_sigmoid(float x){return 1.f/(1.f+expf(-x));}
// Kernel 2: one CTA per row. Partials are staged to shared memory cooperatively, then summed in
// fixed slice order. Gates, softmax and Sinkhorn run on 16 lanes with sinkhorn4's pairing order.
static bool mhc_finish_reg(int h){const char* e=std::getenv("GLM53_MHC_FINISH_REG");return e&&e[0]=='1'&&h<=8*1024;}
__global__ void mhc_pre_finish(const float* __restrict__ partial,int slices,const float* __restrict__ residual,
    const float* __restrict__ scale,const float* __restrict__ base,const float* __restrict__ ln,
    float* __restrict__ z,float* __restrict__ post,float* __restrict__ comb,int rows,int h,bool reg_z){
    const int r=blockIdx.x,tid=threadIdx.x,lane=tid&31;
    extern __shared__ float stage[];
    __shared__ float mix[25];__shared__ float pre[4];__shared__ float wsum[32];__shared__ float rinv;
    for(int i=tid;i<slices*25;i+=blockDim.x){const int sl=i/25,c=i%25;stage[i]=partial[((long long)sl*rows+r)*25+c];}
    __syncthreads();
    if(tid<25){float s=0.f;for(int i=0;i<slices;++i)s+=stage[i*25+tid];mix[tid]=s;}
    __syncthreads();
    if(tid<32){
        const float rstd=rsqrtf(mix[24]/(float)(4*h)+1e-5f);
        if(lane<4){pre[lane]=mhc_sigmoid(mix[lane]*rstd*scale[0]+base[lane])+1e-6f;}
        else if(lane<8){const int s=lane-4;post[r*4+s]=mhc_sigmoid(mix[4+s]*rstd*scale[1]+base[4+s])*2.f;}
        const int e=lane&15;
        float x=mix[8+e]*rstd*scale[2]+base[8+e];
        float mx=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1,16));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2,16));
        const float ex=expf(x-mx);
        float sum=ex+__shfl_xor_sync(0xffffffff,ex,1,16);sum+=__shfl_xor_sync(0xffffffff,sum,2,16);
        float v=ex/sum+1e-6f;
        for(int step=0;step<39;++step){
            const int stride=(step%2==0)?4:1;
            float t=v+__shfl_xor_sync(0xffffffff,v,stride,16);
            t+=__shfl_xor_sync(0xffffffff,t,2*stride,16);
            v=v/(t+1.0e-6f);
        }
        if(lane<16)comb[r*16+lane]=v;
    }
    __syncthreads();
    const float* res=residual+(long long)r*4*h;
    float sq=0.f;
    // GLM53_MHC_FINISH_REG (reg_z): the collapsed row stays in registers (h <= 8*blockDim) instead of a global
    // write + re-read; the same values in the same order feed sq and the final scaling.
    float keep[8];
    if(reg_z){
#pragma unroll
        for(int j=0;j<8;++j){const int c=tid+j*blockDim.x;if(c<h){
            const float col=(((pre[0]*res[c])+pre[1]*res[h+c])+pre[2]*res[2*h+c])+pre[3]*res[3*h+c];keep[j]=col;sq+=col*col;}}
    } else
    for(int c=tid;c<h;c+=blockDim.x){
        const float col=(((pre[0]*res[c])+pre[1]*res[h+c])+pre[2]*res[2*h+c])+pre[3]*res[3*h+c];
        z[(long long)r*h+c]=col;sq+=col*col;
    }
    for(int off=16;off;off>>=1)sq+=__shfl_xor_sync(0xffffffff,sq,off);
    if(lane==0)wsum[tid>>5]=sq;
    __syncthreads();
    if(tid==0){float s=0.f;for(int i=0;i<(int)(blockDim.x>>5);++i)s+=wsum[i];rinv=rsqrtf(s/(float)h+1e-5f);}
    __syncthreads();
    if(reg_z){
#pragma unroll
        for(int j=0;j<8;++j){const int c=tid+j*blockDim.x;if(c<h)z[(long long)r*h+c]=ln[c]*(keep[j]*rinv);}
    } else
    for(int c=tid;c<h;c+=blockDim.x){const long long i=(long long)r*h+c;z[i]=ln[c]*(z[i]*rinv);}
}
template<bool SPLIT,typename W> __global__ void mhc_pre_partial_tc(const float* __restrict__ x,const W* __restrict__ fn,float* __restrict__ partial,int rows,int k);
template<typename W> static void mhc_pre_partial_launch(const float* residual,const W* fn,float* partial,int rows,int k,int slices,cudaStream_t stream){
    const char* tc=std::getenv("GLM53_MHC_PRE_TC");
    // Proposal 3 (GLM53_VERIFY_INVARIANT=1): the plain TC partials for up to 32 rows (8-row tiles on blockIdx.y), so
    // verify batches of any size use the per-row arithmetic of the <=8-row path. Unchanged for <=8 rows.
    const char* inv=std::getenv("GLM53_VERIFY_INVARIANT");const int tc_rows=(inv&&inv[0]=='1')?32:8;
    if(tc&&tc[0]=='1'&&rows<=tc_rows)mhc_pre_partial_tc<false,W><<<dim3(slices,(rows+7)/8),128,0,stream>>>(residual,fn,partial,rows,k);
    // Large rows (prefill): split-TF32 (hi*hi+hi*lo+lo*hi) keeps FP32-class products (L1).
    else if(rows>16)mhc_pre_partial_tc<true,W><<<dim3(slices,(rows+7)/8),128,0,stream>>>(residual,fn,partial,rows,k);
    else mhc_pre_partial<W><<<dim3(slices,rows),MHC_PRE_SLICE,0,stream>>>(residual,fn,partial,rows,k);
}
extern "C" int glm53_mhc_pre_fused_cuda(const float* residual,const void* fn,int fn_bf16,const float* scale,const float* base,const float* ln,
    float* partial,float* z,float* post,float* comb,int rows,int h,cudaStream_t stream){
    const int k=4*h;
    if(rows<1||k%MHC_PRE_SLICE)return int(cudaErrorInvalidValue);
    const int slices=k/MHC_PRE_SLICE;
    if(fn_bf16)mhc_pre_partial_launch(residual,(const __nv_bfloat16*)fn,partial,rows,k,slices,stream);
    else mhc_pre_partial_launch(residual,(const float*)fn,partial,rows,k,slices,stream);
    mhc_pre_finish<<<rows,1024,slices*25*sizeof(float),stream>>>(partial,slices,residual,scale,base,ln,z,post,comb,rows,h,mhc_finish_reg(h));
    return int(cudaGetLastError());
}

// ---- W05e (GLM53_MHC_PRE_TC=1): TF32 tensor-core partials, same [slice][row][25] layout ----
__device__ __forceinline__ void mhc_mma_tf32(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
// Block = one 128-column slice, 4 warps x 2 chunks of 16 columns. Rows 0..23 of fn (24..31 zero).
__device__ __forceinline__ uint32_t mhc_tf32(float x){uint32_t r;asm("cvt.rna.tf32.f32 %0,%1;":"=r"(r):"f"(x));return r;}
template<bool SPLIT>
__device__ __forceinline__ void mhc_mma_pair(float* c,float a0,float a1,float a2,float a3,float b0,float b1){
  if constexpr(!SPLIT){
    mhc_mma_tf32(c,__float_as_uint(a0),__float_as_uint(a1),__float_as_uint(a2),__float_as_uint(a3),__float_as_uint(b0),__float_as_uint(b1));
  }else{
    const uint32_t ah0=mhc_tf32(a0),ah1=mhc_tf32(a1),ah2=mhc_tf32(a2),ah3=mhc_tf32(a3),bh0=mhc_tf32(b0),bh1=mhc_tf32(b1);
    const uint32_t al0=mhc_tf32(a0-__uint_as_float(ah0)),al1=mhc_tf32(a1-__uint_as_float(ah1)),al2=mhc_tf32(a2-__uint_as_float(ah2)),al3=mhc_tf32(a3-__uint_as_float(ah3));
    const uint32_t bl0=mhc_tf32(b0-__uint_as_float(bh0)),bl1=mhc_tf32(b1-__uint_as_float(bh1));
    mhc_mma_tf32(c,al0,al1,al2,al3,bh0,bh1);mhc_mma_tf32(c,ah0,ah1,ah2,ah3,bl0,bl1);mhc_mma_tf32(c,ah0,ah1,ah2,ah3,bh0,bh1);
  }
}
template<bool SPLIT,typename W>
__global__ void mhc_pre_partial_tc(const float* __restrict__ x,const W* __restrict__ fn,float* __restrict__ partial,int rows,int k){
  const int slice=blockIdx.x,warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3,row0=blockIdx.y*8;
  float c[2][4]={{0,0,0,0},{0,0,0,0}};float sq=0.f;
  const bool xv=row0+g<rows;
  for(int ch=0;ch<2;++ch){
    const int col=slice*MHC_PRE_SLICE+(warp*2+ch)*16+t*4;
    float4 xb=make_float4(0,0,0,0);
    if(xv){xb=*reinterpret_cast<const float4*>(x+(long long)(row0+g)*k+col);sq+=xb.x*xb.x+xb.y*xb.y+xb.z*xb.z+xb.w*xb.w;}
#pragma unroll
    for(int tile=0;tile<2;++tile){
      const int r0=tile*16+g,r1=tile*16+g+8;
      const float4 a=r0<24?mw4(fn+(long long)r0*k+col):make_float4(0,0,0,0);
      const float4 b=r1<24?mw4(fn+(long long)r1*k+col):make_float4(0,0,0,0);
      mhc_mma_pair<SPLIT>(c[tile],a.x,b.x,a.y,b.y,xb.x,xb.y);
      mhc_mma_pair<SPLIT>(c[tile],a.z,b.z,a.w,b.w,xb.z,xb.w);
    }
  }
  sq+=__shfl_xor_sync(0xffffffff,sq,1);sq+=__shfl_xor_sync(0xffffffff,sq,2);
  __shared__ float red[4][8][25];
  // C fragment: c[i]: row g+(i>>1)*8 (+16*tile), token 2t+(i&1)
#pragma unroll
  for(int tile=0;tile<2;++tile)
#pragma unroll
    for(int i=0;i<4;++i){const int o=tile*16+g+(i>>1)*8,tok=2*t+(i&1);if(o<24)red[warp][tok][o]=c[tile][i];}
  if(t==0)red[warp][g][24]=sq;
  __syncthreads();
  const int nt=min(8,rows-row0);
  for(int i=threadIdx.x;i<nt*25;i+=blockDim.x){const int tok=i/25,o=i%25;
    partial[((long long)slice*rows+row0+tok)*25+o]=(red[0][tok][o]+red[1][tok][o])+(red[2][tok][o]+red[3][tok][o]);}
}

// ---- P1d (GLM53_MHC_POST_PRE_FUSED=1): post (four streams, non-packed) + next pre partials ----
// The new residual is produced with exactly mhc_post_four_kernel<false,false>'s arithmetic,
// stored, and fed from registers into mhc_pre_partial_tc<true>'s split-TF32 MMA with the same
// slice/chunk layout, so residual and partials are bitwise those of the two separate kernels;
// one full read of the new residual is removed. Contiguous residual/x, hidden % 128 == 0.
template<typename W,bool SPLIT=true>
__global__ void mhc_post_pre_partial(const float* __restrict__ x,const float* __restrict__ residual,const float* __restrict__ comb,
    const float* __restrict__ post,float* __restrict__ out,const W* __restrict__ fn,float* __restrict__ partial,int rows,int hidden){
  const int k=4*hidden,slice=blockIdx.x,warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3,row0=blockIdx.y*8,row=row0+g;
  float c[2][4]={{0,0,0,0},{0,0,0,0}};float sq=0.f;
  const bool xv=row<rows;
  for(int ch=0;ch<2;++ch){
    const int col=slice*MHC_PRE_SLICE+(warp*2+ch)*16+t*4;const int stream=col/hidden,hc=col%hidden;
    float4 xb=make_float4(0,0,0,0);
    if(xv){
      float4 r[4];
#pragma unroll
      for(int kk=0;kk<4;++kk)r[kk]=*reinterpret_cast<const float4*>(residual+(long long)row*k+(long long)kk*hidden+hc);
      const float4 br=*reinterpret_cast<const float4*>(x+(long long)row*hidden+hc);
      float m0=0.f,m1=0.f,m2=0.f,m3=0.f;
#pragma unroll
      for(int kk=0;kk<4;++kk){const float cw=comb[row*16+kk*4+stream];
        m0=__fmaf_rn(cw,r[kk].x,m0);m1=__fmaf_rn(cw,r[kk].y,m1);m2=__fmaf_rn(cw,r[kk].z,m2);m3=__fmaf_rn(cw,r[kk].w,m3);}
      const float p=post[row*4+stream];
      xb=make_float4(__fadd_rn(m0,__fmul_rn(p,br.x)),__fadd_rn(m1,__fmul_rn(p,br.y)),__fadd_rn(m2,__fmul_rn(p,br.z)),__fadd_rn(m3,__fmul_rn(p,br.w)));
      *reinterpret_cast<float4*>(out+(long long)row*k+col)=xb;
      sq+=xb.x*xb.x+xb.y*xb.y+xb.z*xb.z+xb.w*xb.w;
    }
#pragma unroll
    for(int tile=0;tile<2;++tile){
      const int r0=tile*16+g,r1=tile*16+g+8;
      const float4 a=r0<24?mw4(fn+(long long)r0*k+col):make_float4(0,0,0,0);
      const float4 b=r1<24?mw4(fn+(long long)r1*k+col):make_float4(0,0,0,0);
      mhc_mma_pair<SPLIT>(c[tile],a.x,b.x,a.y,b.y,xb.x,xb.y);
      mhc_mma_pair<SPLIT>(c[tile],a.z,b.z,a.w,b.w,xb.z,xb.w);
    }
  }
  sq+=__shfl_xor_sync(0xffffffff,sq,1);sq+=__shfl_xor_sync(0xffffffff,sq,2);
  __shared__ float red[4][8][25];
#pragma unroll
  for(int tile=0;tile<2;++tile)
#pragma unroll
    for(int i=0;i<4;++i){const int o=tile*16+g+(i>>1)*8,tok=2*t+(i&1);if(o<24)red[warp][tok][o]=c[tile][i];}
  if(t==0)red[warp][g][24]=sq;
  __syncthreads();
  const int nt=min(8,rows-row0);
  for(int i=threadIdx.x;i<nt*25;i+=blockDim.x){const int tok=i/25,o=i%25;
    partial[((long long)slice*rows+row0+tok)*25+o]=(red[0][tok][o]+red[1][tok][o])+(red[2][tok][o]+red[3][tok][o]);}
}
static int glm53_mhc_post_pre_fused_cuda_rows(const float*,const float*,const float*,const float*,float*,const void*,int,const float*,const float*,const float*,float*,float*,float*,float*,int,int,cudaStream_t);
extern "C" int glm53_mhc_post_pre_fused_cuda(const float* x,const float* residual,const float* comb,const float* post,float* out,
    const void* fn,int fn_bf16,const float* scale,const float* base,const float* ln,float* partial,float* z,float* post_next,float* comb_next,int rows,int h,cudaStream_t stream){
  const int k=4*h;
  if(rows<1||h%128||k%MHC_PRE_SLICE)return int(cudaErrorInvalidValue);
  const int slices=k/MHC_PRE_SLICE;
  // GLM53_MHC_GROUP_ROWS=G (multiple of 8): prefill rows in groups of G, partial then finish per group, so the finish
  // re-reads the new residual from L2 instead of DRAM. Every row's arithmetic (its 8-row MMA tile, slice partials,
  // finish) is unchanged: L0.
  static const int grp=[]{const char* e=std::getenv("GLM53_MHC_GROUP_ROWS");int g=e?atoi(e):0;return g>=8&&g%8==0?g:0;}();
  if(grp&&rows>grp){
    for(int r0=0;r0<rows;r0+=grp){const int n=rows-r0<grp?rows-r0:grp;
      const int rc=glm53_mhc_post_pre_fused_cuda_rows(x+(long long)r0*h,residual+(long long)r0*k,comb+(long long)r0*16,post+(long long)r0*4,out+(long long)r0*k,
        fn,fn_bf16,scale,base,ln,partial,z+(long long)r0*h,post_next+(long long)r0*4,comb_next+(long long)r0*16,n,h,stream);
      if(rc)return rc;}
    return 0;
  }
  return glm53_mhc_post_pre_fused_cuda_rows(x,residual,comb,post,out,fn,fn_bf16,scale,base,ln,partial,z,post_next,comb_next,rows,h,stream);
}
static int glm53_mhc_post_pre_fused_cuda_rows(const float* x,const float* residual,const float* comb,const float* post,float* out,
    const void* fn,int fn_bf16,const float* scale,const float* base,const float* ln,float* partial,float* z,float* post_next,float* comb_next,int rows,int h,cudaStream_t stream){
  const int k=4*h,slices=k/MHC_PRE_SLICE;
  // Decode rows <= 8 with GLM53_MHC_PRE_TC=1 use the plain-TF32 MMA of mhc_pre_partial_tc<false> (what the
  // separate decode pre uses); prefill keeps split-TF32. Either way bitwise the separate post + pre partial.
  const char* tc=std::getenv("GLM53_MHC_PRE_TC");const bool plain=rows<=8&&tc&&tc[0]=='1';
  if(fn_bf16){if(plain)mhc_post_pre_partial<__nv_bfloat16,false><<<dim3(slices,(rows+7)/8),128,0,stream>>>(x,residual,comb,post,out,(const __nv_bfloat16*)fn,partial,rows,h);
              else mhc_post_pre_partial<__nv_bfloat16,true><<<dim3(slices,(rows+7)/8),128,0,stream>>>(x,residual,comb,post,out,(const __nv_bfloat16*)fn,partial,rows,h);}
  else{if(plain)mhc_post_pre_partial<float,false><<<dim3(slices,(rows+7)/8),128,0,stream>>>(x,residual,comb,post,out,(const float*)fn,partial,rows,h);
       else mhc_post_pre_partial<float,true><<<dim3(slices,(rows+7)/8),128,0,stream>>>(x,residual,comb,post,out,(const float*)fn,partial,rows,h);}
  mhc_pre_finish<<<rows,1024,slices*25*sizeof(float),stream>>>(partial,slices,out,scale,base,ln,z,post_next,comb_next,rows,h,mhc_finish_reg(h));
  return int(cudaGetLastError());
}
