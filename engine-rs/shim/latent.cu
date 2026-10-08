// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <math_constants.h>

// Length is a device scalar, so capture does not freeze the copied extent.
// A bounded persistent grid avoids launching work proportional to capacity.
__global__ void active_cache_copy(const uint4* src,uint4* dst,const long long* len,
    int capacity,int row_vectors,int divisor) {
    long long rows=(*len+divisor-1)/divisor;
    rows=rows<0?0:(rows>capacity?capacity:rows);
    long long count=rows*row_vectors;
    for(long long i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)dst[i]=src[i];
}
extern "C" int glm53_active_cache_copy_cuda(const void* src,void* dst,const long long* len,
    int capacity,int row_vectors,int divisor,cudaStream_t stream) {
    if(capacity<=0 || row_vectors<=0 || divisor<=0)return int(cudaErrorInvalidValue);
    active_cache_copy<<<128,256,0,stream>>>((const uint4*)src,(uint4*)dst,len,capacity,row_vectors,divisor);
    return int(cudaGetLastError());
}


// ---- KV precision (GLM53_KV_FP8=1): latent row = 512 x FP8 e4m3 + 4 FP32 scales (one per 128) = 528 B ----
constexpr int KV8_ROW=528;
struct LatentHalfReader {const half* p;
  __device__ __forceinline__ float get(long long pos,int e) const {return __half2float(p[pos*512+e]);}};
struct LatentFp8Reader {const uint8_t* p;
  __device__ __forceinline__ float get(long long pos,int e) const {
    const uint8_t* row=p+pos*KV8_ROW;const float s=reinterpret_cast<const float*>(row+512)[e>>7];
    __nv_fp8_e4m3 v;v.__x=row[e];return float(v)*s;}};
// n rows of FP16 latent -> FP8 rows at positions (pos_dev ? *pos_dev : pos_host) + r. One warp per 128-tile.
__global__ void latent_fp8_store(const half* __restrict__ rows,int n,const long long* pos_dev,long long pos_host,uint8_t* __restrict__ latent){
  const int r=blockIdx.x,tile=threadIdx.x>>5,lane=threadIdx.x&31;if(r>=n)return;
  const long long pos=(pos_dev?*pos_dev:pos_host)+r;
  float v[4],amax=0.f;
#pragma unroll
  for(int j=0;j<4;++j){v[j]=__half2float(rows[(long long)r*512+tile*128+lane+32*j]);amax=fmaxf(amax,fabsf(v[j]));}
  for(int o=16;o;o>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,o));
  const float scale=amax>0.f?amax/448.f:1.f;
  uint8_t* row=latent+pos*KV8_ROW;
#pragma unroll
  for(int j=0;j<4;++j)row[tile*128+lane+32*j]=__nv_cvt_float_to_fp8(v[j]/scale,__NV_SATFINITE,__NV_E4M3);
  if(lane==0)reinterpret_cast<float*>(row+512)[tile]=scale;
}
// GLM53_MLA_MULTI_LAUNCH (L0): latent_fp8_store for up to 8 sequences in one launch (blockIdx.y = sequence; its rows start
// at first in the stacked Half latent rows and land in its own FP8 cache at *pos + r).
struct LatentStoreMulti {uint8_t* latent[8];const long long* pos[8];int first[8];int n[8];int nseq;};
__global__ void latent_fp8_store_multi(const half* __restrict__ rows_all,const LatentStoreMulti m){
  const int sidx=blockIdx.y,r=blockIdx.x,tile=threadIdx.x>>5,lane=threadIdx.x&31;if(sidx>=m.nseq||r>=m.n[sidx])return;
  const half* rows=rows_all+(long long)m.first[sidx]*512;
  const long long pos=*m.pos[sidx]+r;
  float v[4],amax=0.f;
#pragma unroll
  for(int j=0;j<4;++j){v[j]=__half2float(rows[(long long)r*512+tile*128+lane+32*j]);amax=fmaxf(amax,fabsf(v[j]));}
  for(int o=16;o;o>>=1)amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,o));
  const float scale=amax>0.f?amax/448.f:1.f;
  uint8_t* row=m.latent[sidx]+pos*KV8_ROW;
#pragma unroll
  for(int j=0;j<4;++j)row[tile*128+lane+32*j]=__nv_cvt_float_to_fp8(v[j]/scale,__NV_SATFINITE,__NV_E4M3);
  if(lane==0)reinterpret_cast<float*>(row+512)[tile]=scale;
}
extern "C" int glm53_latent_fp8_store_multi_cuda(const void* rows,const void* table,cudaStream_t s){
  const LatentStoreMulti m=*reinterpret_cast<const LatentStoreMulti*>(table);
  if(m.nseq<1||m.nseq>8)return int(cudaErrorInvalidValue);int mx=0;for(int g=0;g<m.nseq;++g){if(m.n[g]<1)return int(cudaErrorInvalidValue);mx=m.n[g]>mx?m.n[g]:mx;}
  latent_fp8_store_multi<<<dim3(mx,m.nseq),128,0,s>>>((const half*)rows,m);return int(cudaGetLastError());
}
extern "C" int glm53_latent_fp8_store_cuda(const void* rows,int n,const long long* pos_dev,long long pos_host,void* latent,cudaStream_t s){
  if(n<1)return int(cudaErrorInvalidValue);
  latent_fp8_store<<<n,128,0,s>>>((const half*)rows,n,pos_dev,pos_host,(uint8_t*)latent);return int(cudaGetLastError());
}

// The all-visible verifier retains the exact padded-pool + three-tail slot
// order of the explicit index tensor. Derive indices in the consumer instead
// of generating, storing and reading that tensor for every branch.
template<bool Visible>
__device__ __forceinline__ long long latent_position(const long long* selected,
    int query,int slot,int slots) {
    if constexpr(Visible) {
        const long long len=selected[query]+1,full=(len/4)*4;
        if(slot<slots-3)return slot<full?slot:-1;
        const int tail=slot-(slots-3);
        return tail<len%4?full+tail:-1;
    } else return selected[query*slots+slot];
}

// One CTA per (query,head), four independent online-softmax streams. Invalid
// DSA slots never read latent. FP32 Q/accumulators, FP16 resident latent.
template<bool Visible=false,class R=LatentHalfReader>
__global__ void sparse_latent(const float* q,R latent,const long long* selected,
    float* out,int heads,int queries,int slots,int query_stride,int splits) {
    int query=blockIdx.x,head=blockIdx.y,part=blockIdx.z,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    float query_values[16],acc[16];
    #pragma unroll
    for(int j=0;j<16;++j){query_values[j]=q[head*query_stride+query*512+lane+j*32];acc[j]=0.f;}
    float maximum=-CUDART_INF_F,total=0.f;
    for(int slot=part*4+warp;slot<slots;slot+=4*splits) {
        long long pos=latent_position<Visible>(selected,query,slot,slots);
        // DSA emits all valid full pools before padding, then three tail
        // slots. Jump over the entire invalid suffix instead of scanning it.
        if(pos<0 && slot<slots-3){
            // Keep this worker's residue modulo 4*splits. Merely adding its
            // residue to full_slots duplicates tail rows at small capacities
            // when full_slots is not divisible by 32 (splits is 1 or 8).
            const int full_slots=slots-3,stride=4*splits;
            slot=full_slots+((part*4+warp-full_slots)&(stride-1));
            if(slot>=slots)break;pos=latent_position<Visible>(selected,query,slot,slots);
        }
        if(pos<0)continue;
        float values[16],dot=0.f;
        #pragma unroll
        for(int j=0;j<16;++j){values[j]=latent.get(pos,lane+j*32);dot+=query_values[j]*values[j];}
        for(int s=16;s;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
        float score=__shfl_sync(0xffffffff,dot,0)*(1.f/16.f);
        float next=fmaxf(maximum,score),old_scale=expf(maximum-next),weight=expf(score-next);
        total=total*old_scale+weight;
        #pragma unroll
        for(int j=0;j<16;++j)acc[j]=acc[j]*old_scale+weight*values[j];
        maximum=next;
    }
    __shared__ float partial[4*512],maxima[4],totals[4];
    #pragma unroll
    for(int j=0;j<16;++j)partial[warp*512+lane+j*32]=acc[j];
    if(lane==0){maxima[warp]=maximum;totals[warp]=total;}
    __syncthreads();
    float global_max=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
    float scales[4],denom=0.f;
    #pragma unroll
    for(int w=0;w<4;++w){scales[w]=totals[w]>0.f?expf(maxima[w]-global_max):0.f;denom+=totals[w]*scales[w];}
    #pragma unroll
    for(int j=0;j<4;++j){int dim=threadIdx.x+j*128;float value=0.f;
        #pragma unroll
        for(int w=0;w<4;++w)value+=partial[w*512+dim]*scales[w];
        if(splits==1)out[(head*queries+query)*512+dim]=value/denom;
        else out[((part*heads+head)*queries+query)*514+dim]=value;
    }
    if(splits>1 && threadIdx.x==0){int base=((part*heads+head)*queries+query)*514;out[base+512]=global_max;out[base+513]=denom;}
}
__global__ void merge_latent(const float* partial,float* out,int heads,int queries,int splits){
    int row=blockIdx.x;float maximum=-CUDART_INF_F;
    for(int s=0;s<splits;++s)maximum=fmaxf(maximum,partial[(s*heads*queries+row)*514+512]);
    float scale[8],denom=0.f;
    for(int s=0;s<splits;++s){int base=(s*heads*queries+row)*514;scale[s]=partial[base+513]>0.f?expf(partial[base+512]-maximum):0.f;denom+=partial[base+513]*scale[s];}
    for(int j=0;j<4;++j){int dim=threadIdx.x+j*128;float value=0.f;for(int s=0;s<splits;++s)value+=partial[(s*heads*queries+row)*514+dim]*scale[s];out[row*512+dim]=value/denom;}
}
// GLM53_MLA_MULTI_LAUNCH (L0): sparse_latent<false,LatentFp8Reader> over the stacked queries of up to 8 sequences, each
// query reading its own sequence's FP8 latent cache; per-(query, head, part) arithmetic, scratch layout by global query and
// merge_latent unchanged, so out[head, query] equals the per-sequence call's.
struct LatentSeqs {const uint8_t* latent[8];int first[8];int n;};
__global__ void sparse_latent_fp8_multi(const float* q,const LatentSeqs L,const long long* selected,
    float* out,int heads,int queries,int slots,int query_stride,int splits) {
    int query=blockIdx.x,head=blockIdx.y,part=blockIdx.z,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int sidx=0;
#pragma unroll 1
    for(;sidx<L.n-1;++sidx)if(query<L.first[sidx+1])break;
    const LatentFp8Reader latent{L.latent[sidx]};
    float query_values[16],acc[16];
    #pragma unroll
    for(int j=0;j<16;++j){query_values[j]=q[head*query_stride+query*512+lane+j*32];acc[j]=0.f;}
    float maximum=-CUDART_INF_F,total=0.f;
    for(int slot=part*4+warp;slot<slots;slot+=4*splits) {
        long long pos=latent_position<false>(selected,query,slot,slots);
        if(pos<0 && slot<slots-3){
            const int full_slots=slots-3,stride=4*splits;
            slot=full_slots+((part*4+warp-full_slots)&(stride-1));
            if(slot>=slots)break;pos=latent_position<false>(selected,query,slot,slots);
        }
        if(pos<0)continue;
        float values[16],dot=0.f;
        #pragma unroll
        for(int j=0;j<16;++j){values[j]=latent.get(pos,lane+j*32);dot+=query_values[j]*values[j];}
        for(int s=16;s;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
        float score=__shfl_sync(0xffffffff,dot,0)*(1.f/16.f);
        float next=fmaxf(maximum,score),old_scale=expf(maximum-next),weight=expf(score-next);
        total=total*old_scale+weight;
        #pragma unroll
        for(int j=0;j<16;++j)acc[j]=acc[j]*old_scale+weight*values[j];
        maximum=next;
    }
    __shared__ float partial[4*512],maxima[4],totals[4];
    #pragma unroll
    for(int j=0;j<16;++j)partial[warp*512+lane+j*32]=acc[j];
    if(lane==0){maxima[warp]=maximum;totals[warp]=total;}
    __syncthreads();
    float global_max=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
    float scales[4],denom=0.f;
    #pragma unroll
    for(int w=0;w<4;++w){scales[w]=totals[w]>0.f?expf(maxima[w]-global_max):0.f;denom+=totals[w]*scales[w];}
    #pragma unroll
    for(int j=0;j<4;++j){int dim=threadIdx.x+j*128;float value=0.f;
        #pragma unroll
        for(int w=0;w<4;++w)value+=partial[w*512+dim]*scales[w];
        if(splits==1)out[(head*queries+query)*512+dim]=value/denom;
        else out[((part*heads+head)*queries+query)*514+dim]=value;
    }
    if(splits>1 && threadIdx.x==0){int base=((part*heads+head)*queries+query)*514;out[base+512]=global_max;out[base+513]=denom;}
}
extern "C" int glm53_latent_attention_fp8_multi_cuda(const float* q,const void* table,const long long* selected,
    float* out,float* scratch,int heads,int queries,int slots,int query_stride,int splits,cudaStream_t stream) {
    const LatentSeqs L=*reinterpret_cast<const LatentSeqs*>(table);
    if(L.n<1||L.n>8||queries<1)return int(cudaErrorInvalidValue);
    sparse_latent_fp8_multi<<<dim3(queries,heads,splits),128,0,stream>>>(q,L,selected,splits>1?scratch:out,heads,queries,slots,query_stride,splits);
    if(splits>1)merge_latent<<<heads*queries,128,0,stream>>>(scratch,out,heads,queries,splits);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int glm53_latent_attention_fp8_cuda(const float* q,const void* latent,const long long* selected,
    float* out,float* scratch,int heads,int queries,int slots,int query_stride,int splits,cudaStream_t stream) {
    sparse_latent<false,LatentFp8Reader><<<dim3(queries,heads,splits),128,0,stream>>>(q,LatentFp8Reader{(const uint8_t*)latent},selected,splits>1?scratch:out,heads,queries,slots,query_stride,splits);
    if(splits>1)merge_latent<<<heads*queries,128,0,stream>>>(scratch,out,heads,queries,splits);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int glm53_latent_attention_cuda(const float* q,const void* latent,const long long* selected,
    float* out,float* scratch,int heads,int queries,int slots,int query_stride,int splits,cudaStream_t stream) {
    sparse_latent<false><<<dim3(queries,heads,splits),128,0,stream>>>(q,LatentHalfReader{(const half*)latent},selected,splits>1?scratch:out,heads,queries,slots,query_stride,splits);
    if(splits>1)merge_latent<<<heads*queries,128,0,stream>>>(scratch,out,heads,queries,splits);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int glm53_visible_latent_cuda(const float* q,const void* latent,const long long* positions,
    float* out,float* scratch,int heads,int queries,int slots,int query_stride,int splits,cudaStream_t stream) {
    if(slots<7 || slots>2051 || (slots-3)%4 || heads<1 || queries<1 || (splits!=1 && splits!=8))
        return int(cudaErrorInvalidValue);
    sparse_latent<true><<<dim3(queries,heads,splits),128,0,stream>>>(q,LatentHalfReader{(const half*)latent},positions,splits>1?scratch:out,heads,queries,slots,query_stride,splits);
    if(splits>1)merge_latent<<<heads*queries,128,0,stream>>>(scratch,out,heads,queries,splits);
    return int(cudaGetLastError());
}

// Gather into shared memory, directly from the resident FP16 latent cache.
// Each warp owns one head; 4/8 heads share every fetched key/value tile. No
// query x selected-key tensor is ever written to global memory. Full FP32
// query, score, softmax and value accumulation are retained.
template<int HeadGroup,int Tile,bool Vectorized=false,bool Blockwise=false>
__global__ void shared_latent(const float* q,const half* latent,const long long* selected,
    float* out,int heads,int queries,int slots,int query_stride) {
    int query=blockIdx.x,head=blockIdx.y*HeadGroup+(threadIdx.x>>5),lane=threadIdx.x&31;
    __shared__ __align__(16) half keys[Tile*512];
    __shared__ long long positions[Tile];
    __shared__ int full_count;
    if(threadIdx.x==0){
        int lo=0,hi=slots-3;
        while(lo<hi){int mid=(lo+hi)/2;if(selected[query*slots+mid]>=0)lo=mid+1;else hi=mid;}
        full_count=lo;
    }
    float query_values[16],acc[16];
    #pragma unroll
    for(int j=0;j<16;++j){query_values[j]=head<heads?q[head*query_stride+query*512+lane+j*32]:0.f;acc[j]=0.f;}
    float maximum=-CUDART_INF_F,total=0.f;
    __syncthreads();
    for(int base=0;base<full_count+3;base+=Tile){
        if(threadIdx.x<Tile){int key=base+threadIdx.x;int slot=key<full_count?key:slots-3+key-full_count;
            positions[threadIdx.x]=key<full_count+3?selected[query*slots+slot]:-1;}
        if constexpr(Vectorized){
            // Each loading lane obtains its own small cached index. This
            // removes the barrier waiting for the shared position table and
            // moves eight FP16 elements per memory instruction.
            for(int i=threadIdx.x;i<Tile*64;i+=HeadGroup*32){int key=base+i/64,col=(i%64)*8;
                int slot=key<full_count?key:slots-3+key-full_count;
                long long pos=key<full_count+3?selected[query*slots+slot]:-1;
                reinterpret_cast<uint4*>(keys)[i]=pos>=0?*reinterpret_cast<const uint4*>(latent+pos*512+col):make_uint4(0,0,0,0);}
        }else{
            __syncthreads();
            for(int i=threadIdx.x;i<Tile*256;i+=HeadGroup*32){int key=i/256,col=(i%256)*2;long long pos=positions[key];
                reinterpret_cast<half2*>(keys)[i]=pos>=0?*reinterpret_cast<const half2*>(latent+pos*512+col):__float2half2_rn(0.f);}
        }
        __syncthreads();
        if constexpr(Blockwise){
            float scores[Tile],tile_max=-CUDART_INF_F;
            #pragma unroll
            for(int key=0;key<Tile;++key){
                float dot=0.f;
                if(positions[key]>=0){
                    #pragma unroll
                    for(int j=0;j<16;++j)dot+=query_values[j]*__half2float(keys[key*512+lane+j*32]);
                    for(int s=16;s;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
                    scores[key]=__shfl_sync(0xffffffff,dot,0)*(1.f/16.f);
                }else scores[key]=-CUDART_INF_F;
                tile_max=fmaxf(tile_max,scores[key]);
            }
            float next=fmaxf(maximum,tile_max),old_scale=total>0.f?expf(maximum-next):0.f;
            total*=old_scale;
            #pragma unroll
            for(int j=0;j<16;++j)acc[j]*=old_scale;
            // Rescale the running output once per tile, not once per key.
            #pragma unroll
            for(int key=0;key<Tile;++key){
                if(positions[key]<0)continue;
                float weight=expf(scores[key]-next);total+=weight;
                #pragma unroll
                for(int j=0;j<16;++j)acc[j]+=weight*__half2float(keys[key*512+lane+j*32]);
            }
            maximum=next;
        }else{for(int key=0;key<Tile;++key){
            if(positions[key]<0)continue;
            float values[16],dot=0.f;
            #pragma unroll
            for(int j=0;j<16;++j){values[j]=__half2float(keys[key*512+lane+j*32]);dot+=query_values[j]*values[j];}
            for(int s=16;s;s>>=1)dot+=__shfl_down_sync(0xffffffff,dot,s);
            float score=__shfl_sync(0xffffffff,dot,0)*(1.f/16.f);
            float next=fmaxf(maximum,score),old_scale=expf(maximum-next),weight=expf(score-next);
            total=total*old_scale+weight;
            #pragma unroll
            for(int j=0;j<16;++j)acc[j]=acc[j]*old_scale+weight*values[j];
            maximum=next;
        }}
        __syncthreads();
    }
    if(head<heads){
        #pragma unroll
        for(int j=0;j<16;++j)out[(head*queries+query)*512+lane+j*32]=acc[j]/total;
    }
}

// ---- P-MLA (GLM53_MLA_PREFILL_DENSE=8): tensor-core form of shared_latent ----
// CTA = one query x 16 heads, 4 warps. Keys follow exactly the same DSA selected list
// (prefix + 3 tail slots, -1 masked). FP32 query and FP32 probabilities enter
// mma.m16n8k8.tf32 as hi+lo TF32 pairs; FP16 latent values are exact in TF32, so each
// product keeps ~22-bit operands with FP32 accumulation (L1: blocking/summation order).
// Warp w owns latent dims [128w,128w+128) for both QK (partial scores) and PV (output).
__device__ __forceinline__ uint32_t tcl_tf32(float x){uint32_t r;asm("cvt.rna.tf32.f32 %0,%1;":"=r"(r):"f"(x));return r;}
__device__ __forceinline__ void tcl_split(float x,uint32_t& hi,uint32_t& lo){hi=tcl_tf32(x);lo=tcl_tf32(x-__uint_as_float(hi));}
__device__ __forceinline__ void tcl_mma(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
constexpr int TCL_KEYS=32,TCL_LD=520;
template<bool FP8=false>
__global__ void __launch_bounds__(128) latent_tc(const float* __restrict__ q,const half* __restrict__ latent,
    const long long* __restrict__ selected,float* __restrict__ out,int heads,int queries,int slots,int query_stride){
  const int query=blockIdx.x,h0=blockIdx.y*16,warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3,tid=threadIdx.x;
  __shared__ __align__(16) half keys[TCL_KEYS*TCL_LD];
  __shared__ float spart[4][16][TCL_KEYS+1];
  __shared__ float prob[16][TCL_KEYS+4];
  __shared__ long long pos[TCL_KEYS];
  __shared__ float rowscale[16];
  __shared__ int full_count;
  if(tid==0){int lo=0,hi=slots-3;
    while(lo<hi){int mid=(lo+hi)/2;if(selected[(long long)query*slots+mid]>=0)lo=mid+1;else hi=mid;}
    full_count=lo;}
  const int dbase=warp*128;
  float qv[16][4];
  {const float* q0=q+(long long)(h0+g)*query_stride+(long long)query*512+dbase;
   const float* q1=q0+8LL*query_stride;
#pragma unroll
   for(int s=0;s<16;++s){qv[s][0]=q0[s*8+t];qv[s][1]=q1[s*8+t];qv[s][2]=q0[s*8+t+4];qv[s][3]=q1[s*8+t+4];}}
  float o[16][4];
#pragma unroll
  for(int n=0;n<16;++n){o[n][0]=o[n][1]=o[n][2]=o[n][3]=0.f;}
  float maximum=-CUDART_INF_F,total=0.f;   // per softmax row (tid>>3), replicated on 8 threads
  __syncthreads();
  const int nkeys=full_count+3;
  for(int base=0;base<nkeys;base+=TCL_KEYS){
    if(tid<TCL_KEYS){const int key=base+tid;const int slot=key<full_count?key:slots-3+key-full_count;
      pos[tid]=key<nkeys?selected[(long long)query*slots+slot]:-1;}
    __syncthreads();
    if constexpr(FP8){
      // 8 FP8 bytes -> 8 halves (value * tile scale rounded to FP16, exact in TF32 afterwards).
      const uint8_t* lat8=reinterpret_cast<const uint8_t*>(latent);
      for(int i=tid;i<TCL_KEYS*64;i+=128){const int key=i>>6,col=(i&63)*8;const long long p=pos[key];
        half hv[8];
        if(p>=0){const uint8_t* row=lat8+p*KV8_ROW;const float sc=reinterpret_cast<const float*>(row+512)[col>>7];
          const uint2 raw=*reinterpret_cast<const uint2*>(row+col);const uint8_t* b=reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
          for(int e=0;e<8;++e){__nv_fp8_e4m3 v;v.__x=b[e];hv[e]=__float2half_rn(float(v)*sc);}}
        else{
#pragma unroll
          for(int e=0;e<8;++e)hv[e]=__float2half_rn(0.f);}
        *reinterpret_cast<uint4*>(keys+key*TCL_LD+col)=*reinterpret_cast<const uint4*>(hv);}
    }else{
    for(int i=tid;i<TCL_KEYS*64;i+=128){const int key=i>>6,col=(i&63)*8;const long long p=pos[key];
      *reinterpret_cast<uint4*>(keys+key*TCL_LD+col)=p>=0?*reinterpret_cast<const uint4*>(latent+p*512+col):make_uint4(0,0,0,0);}
    }
    __syncthreads();
    // partial scores over this warp's 128 dims: 16 heads x 32 keys
    float c[4][4];
#pragma unroll
    for(int j=0;j<4;++j){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
#pragma unroll
    for(int s=0;s<16;++s){
      uint32_t ah[4],al[4];
#pragma unroll
      for(int r=0;r<4;++r)tcl_split(qv[s][r],ah[r],al[r]);
#pragma unroll
      for(int j=0;j<4;++j){const half* kr=keys+(j*8+g)*TCL_LD+dbase+s*8+t;
        const uint32_t b0=__float_as_uint(__half2float(kr[0])),b1=__float_as_uint(__half2float(kr[4]));
        tcl_mma(c[j],ah[0],ah[1],ah[2],ah[3],b0,b1);tcl_mma(c[j],al[0],al[1],al[2],al[3],b0,b1);}
    }
#pragma unroll
    for(int j=0;j<4;++j){spart[warp][g][j*8+2*t]=c[j][0];spart[warp][g][j*8+2*t+1]=c[j][1];
      spart[warp][g+8][j*8+2*t]=c[j][2];spart[warp][g+8][j*8+2*t+1]=c[j][3];}
    __syncthreads();
    {const int r=tid>>3,k0=(tid&7)*4;float sc[4],tmax=-CUDART_INF_F;
#pragma unroll
     for(int i=0;i<4;++i){const int k=k0+i;
       sc[i]=pos[k]>=0?((spart[0][r][k]+spart[1][r][k])+(spart[2][r][k]+spart[3][r][k]))*(1.f/16.f):-CUDART_INF_F;
       tmax=fmaxf(tmax,sc[i]);}
     tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,1));tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,2));
     tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,4));
     const float next=fmaxf(maximum,tmax),old_scale=total>0.f?expf(maximum-next):0.f;
     float psum=0.f;
#pragma unroll
     for(int i=0;i<4;++i){const float p=pos[k0+i]>=0?expf(sc[i]-next):0.f;prob[r][k0+i]=p;psum+=p;}
     psum+=__shfl_xor_sync(0xffffffff,psum,1);psum+=__shfl_xor_sync(0xffffffff,psum,2);psum+=__shfl_xor_sync(0xffffffff,psum,4);
     total=total*old_scale+psum;if(next>-CUDART_INF_F)maximum=next;
     if((tid&7)==0)rowscale[r]=old_scale;}
    __syncthreads();
    {const float s0=rowscale[g],s1=rowscale[g+8];
#pragma unroll
     for(int n=0;n<16;++n){o[n][0]*=s0;o[n][1]*=s0;o[n][2]*=s1;o[n][3]*=s1;}}
#pragma unroll
    for(int ks=0;ks<4;++ks){
      uint32_t ah[4],al[4];
      tcl_split(prob[g][ks*8+t],ah[0],al[0]);tcl_split(prob[g+8][ks*8+t],ah[1],al[1]);
      tcl_split(prob[g][ks*8+t+4],ah[2],al[2]);tcl_split(prob[g+8][ks*8+t+4],ah[3],al[3]);
      const half* v0=keys+(ks*8+t)*TCL_LD+dbase+g;const half* v1=v0+4*TCL_LD;
#pragma unroll
      for(int n=0;n<16;++n){const uint32_t b0=__float_as_uint(__half2float(v0[n*8])),b1=__float_as_uint(__half2float(v1[n*8]));
        tcl_mma(o[n],ah[0],ah[1],ah[2],ah[3],b0,b1);tcl_mma(o[n],al[0],al[1],al[2],al[3],b0,b1);}
    }
    __syncthreads();
  }
  // row statistics live on thread (row*8); fetch them for rows g and g+8
  __shared__ float rtotal[16];
  if((tid&7)==0)rtotal[tid>>3]=total;
  __syncthreads();
  const float i0=1.f/rtotal[g],i1=1.f/rtotal[g+8];
  float* out0=out+((long long)(h0+g)*queries+query)*512+dbase+2*t;
  float* out1=out+((long long)(h0+g+8)*queries+query)*512+dbase+2*t;
#pragma unroll
  for(int n=0;n<16;++n){*reinterpret_cast<float2*>(out0+n*8)=make_float2(o[n][0]*i0,o[n][1]*i0);
    *reinterpret_cast<float2*>(out1+n*8)=make_float2(o[n][2]*i1,o[n][3]*i1);}
}

// ---- P7 (GLM53_MLA_PREFILL_F16=1): FP16 tensor-core form of latent_tc (mode 10 half / 11 FP8 latent) ----
// Same CTA layout, key enumeration and online softmax as latent_tc; Q and probabilities enter
// mma.m16n8k16 as FP16 (FP32 accumulation), latent keys/values are FP16 (exact) or dequantized FP8.
__device__ __forceinline__ void tch_mma16(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint32_t tch_pack(float lo,float hi){half2 h=__floats2half2_rn(lo,hi);return *reinterpret_cast<uint32_t*>(&h);}
__device__ __forceinline__ uint32_t tch_pack_h(half lo,half hi){half2 h=__halves2half2(lo,hi);return *reinterpret_cast<uint32_t*>(&h);}
template<bool FP8=false>
__global__ void __launch_bounds__(128) latent_tc16(const float* __restrict__ q,const half* __restrict__ latent,
    const long long* __restrict__ selected,float* __restrict__ out,int heads,int queries,int slots,int query_stride){
  const int query=blockIdx.x,h0=blockIdx.y*16,warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3,tid=threadIdx.x;
  __shared__ __align__(16) half keys[TCL_KEYS*TCL_LD];
  __shared__ float spart[4][16][TCL_KEYS+1];
  __shared__ float prob[16][TCL_KEYS+4];
  __shared__ long long pos[TCL_KEYS];
  __shared__ float rowscale[16];
  __shared__ int full_count;
  if(tid==0){int lo=0,hi=slots-3;
    while(lo<hi){int mid=(lo+hi)/2;if(selected[(long long)query*slots+mid]>=0)lo=mid+1;else hi=mid;}
    full_count=lo;}
  const int dbase=warp*128;
  uint32_t qa[8][4];   // A fragments over this warp's 128 dims: 8 k16-steps
  {const float* q0=q+(long long)(h0+g)*query_stride+(long long)query*512+dbase;const float* q1=q0+8LL*query_stride;
#pragma unroll
   for(int s=0;s<8;++s){const int k=s*16+2*t;
     qa[s][0]=tch_pack(q0[k],q0[k+1]);qa[s][1]=tch_pack(q1[k],q1[k+1]);qa[s][2]=tch_pack(q0[k+8],q0[k+9]);qa[s][3]=tch_pack(q1[k+8],q1[k+9]);}}
  float o[16][4];
#pragma unroll
  for(int n=0;n<16;++n){o[n][0]=o[n][1]=o[n][2]=o[n][3]=0.f;}
  float maximum=-CUDART_INF_F,total=0.f;
  __syncthreads();
  const int nkeys=full_count+3;
  for(int base=0;base<nkeys;base+=TCL_KEYS){
    if(tid<TCL_KEYS){const int key=base+tid;const int slot=key<full_count?key:slots-3+key-full_count;
      pos[tid]=key<nkeys?selected[(long long)query*slots+slot]:-1;}
    __syncthreads();
    if constexpr(FP8){
      const uint8_t* lat8=reinterpret_cast<const uint8_t*>(latent);
      for(int i=tid;i<TCL_KEYS*64;i+=128){const int key=i>>6,col=(i&63)*8;const long long p=pos[key];
        half hv[8];
        if(p>=0){const uint8_t* row=lat8+p*KV8_ROW;const float sc=reinterpret_cast<const float*>(row+512)[col>>7];
          const uint2 raw=*reinterpret_cast<const uint2*>(row+col);const uint8_t* b=reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
          for(int e=0;e<8;++e){__nv_fp8_e4m3 v;v.__x=b[e];hv[e]=__float2half_rn(float(v)*sc);}}
        else{
#pragma unroll
          for(int e=0;e<8;++e)hv[e]=__float2half_rn(0.f);}
        *reinterpret_cast<uint4*>(keys+key*TCL_LD+col)=*reinterpret_cast<const uint4*>(hv);}
    }else{
      for(int i=tid;i<TCL_KEYS*64;i+=128){const int key=i>>6,col=(i&63)*8;const long long p=pos[key];
        *reinterpret_cast<uint4*>(keys+key*TCL_LD+col)=p>=0?*reinterpret_cast<const uint4*>(latent+p*512+col):make_uint4(0,0,0,0);}
    }
    __syncthreads();
    float c[4][4];
#pragma unroll
    for(int j=0;j<4;++j){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
#pragma unroll
    for(int s=0;s<8;++s){
#pragma unroll
      for(int j=0;j<4;++j){const half* kr=keys+(j*8+g)*TCL_LD+dbase+s*16+2*t;
        const uint32_t b0=*reinterpret_cast<const uint32_t*>(kr),b1=*reinterpret_cast<const uint32_t*>(kr+8);
        tch_mma16(c[j],qa[s][0],qa[s][1],qa[s][2],qa[s][3],b0,b1);}
    }
#pragma unroll
    for(int j=0;j<4;++j){spart[warp][g][j*8+2*t]=c[j][0];spart[warp][g][j*8+2*t+1]=c[j][1];
      spart[warp][g+8][j*8+2*t]=c[j][2];spart[warp][g+8][j*8+2*t+1]=c[j][3];}
    __syncthreads();
    {const int r=tid>>3,k0=(tid&7)*4;float sc[4],tmax=-CUDART_INF_F;
#pragma unroll
     for(int i=0;i<4;++i){const int k=k0+i;
       sc[i]=pos[k]>=0?((spart[0][r][k]+spart[1][r][k])+(spart[2][r][k]+spart[3][r][k]))*(1.f/16.f):-CUDART_INF_F;
       tmax=fmaxf(tmax,sc[i]);}
     tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,1));tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,2));
     tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,4));
     const float next=fmaxf(maximum,tmax),old_scale=total>0.f?expf(maximum-next):0.f;
     float psum=0.f;
#pragma unroll
     for(int i=0;i<4;++i){const float p=pos[k0+i]>=0?expf(sc[i]-next):0.f;prob[r][k0+i]=p;psum+=p;}
     psum+=__shfl_xor_sync(0xffffffff,psum,1);psum+=__shfl_xor_sync(0xffffffff,psum,2);psum+=__shfl_xor_sync(0xffffffff,psum,4);
     total=total*old_scale+psum;if(next>-CUDART_INF_F)maximum=next;
     if((tid&7)==0)rowscale[r]=old_scale;}
    __syncthreads();
    {const float s0=rowscale[g],s1=rowscale[g+8];
#pragma unroll
     for(int n=0;n<16;++n){o[n][0]*=s0;o[n][1]*=s0;o[n][2]*=s1;o[n][3]*=s1;}}
#pragma unroll
    for(int ks=0;ks<TCL_KEYS;ks+=16){
      const uint32_t a0=tch_pack(prob[g][ks+2*t],prob[g][ks+2*t+1]),a1=tch_pack(prob[g+8][ks+2*t],prob[g+8][ks+2*t+1]);
      const uint32_t a2=tch_pack(prob[g][ks+2*t+8],prob[g][ks+2*t+9]),a3=tch_pack(prob[g+8][ks+2*t+8],prob[g+8][ks+2*t+9]);
      const half* v0=keys+(ks+2*t)*TCL_LD+dbase+g;
#pragma unroll
      for(int n=0;n<16;++n){const half* v=v0+n*8;
        const uint32_t b0=tch_pack_h(v[0],v[TCL_LD]),b1=tch_pack_h(v[8*TCL_LD],v[9*TCL_LD]);
        tch_mma16(o[n],a0,a1,a2,a3,b0,b1);}
    }
    __syncthreads();
  }
  __shared__ float rtotal[16];
  if((tid&7)==0)rtotal[tid>>3]=total;
  __syncthreads();
  const float i0=1.f/rtotal[g],i1=1.f/rtotal[g+8];
  float* out0=out+((long long)(h0+g)*queries+query)*512+dbase+2*t;
  float* out1=out+((long long)(h0+g+8)*queries+query)*512+dbase+2*t;
#pragma unroll
  for(int n=0;n<16;++n){*reinterpret_cast<float2*>(out0+n*8)=make_float2(o[n][0]*i0,o[n][1]*i0);
    *reinterpret_cast<float2*>(out1+n*8)=make_float2(o[n][2]*i1,o[n][3]*i1);}
}
template<bool FP8,int HG>
__global__ void __launch_bounds__(128*HG) latent_tc16_hg(const float* __restrict__ q,const half* __restrict__ latent,
    const long long* __restrict__ selected,float* __restrict__ out,int heads,int queries,int slots,int query_stride){
  // GLM53_MLA_TC16_HG=2: HG groups of 16 heads per CTA share each gathered key tile (one gather per query instead of
  // one per 16 heads); group hg's four warps run latent_tc16's exact per-head arithmetic (L0).
  const int hg=threadIdx.x>>7,query=blockIdx.x,h0=(blockIdx.y*HG+hg)*16,warp=(threadIdx.x>>5)&3,lane=threadIdx.x&31,g=lane>>2,t=lane&3,tid=threadIdx.x&127,ctid=threadIdx.x;
  extern __shared__ __align__(16) unsigned char tchg_smem[];
  half* keys=reinterpret_cast<half*>(tchg_smem);
  float (*spart)[TCL_KEYS+1]=reinterpret_cast<float(*)[TCL_KEYS+1]>(tchg_smem+TCL_KEYS*TCL_LD*2+(size_t)hg*4*16*(TCL_KEYS+1)*4);
  float (*prob)[TCL_KEYS+4]=reinterpret_cast<float(*)[TCL_KEYS+4]>(tchg_smem+TCL_KEYS*TCL_LD*2+(size_t)HG*4*16*(TCL_KEYS+1)*4+(size_t)hg*16*(TCL_KEYS+4)*4);
  __shared__ long long pos[TCL_KEYS];
  __shared__ float rowscale_all[HG][16];float* rowscale=rowscale_all[hg];
  __shared__ int full_count;
  if(ctid==0){int lo=0,hi=slots-3;
    while(lo<hi){int mid=(lo+hi)/2;if(selected[(long long)query*slots+mid]>=0)lo=mid+1;else hi=mid;}
    full_count=lo;}
  const int dbase=warp*128;
  uint32_t qa[8][4];   // A fragments over this warp's 128 dims: 8 k16-steps
  {const float* q0=q+(long long)(h0+g)*query_stride+(long long)query*512+dbase;const float* q1=q0+8LL*query_stride;
#pragma unroll
   for(int s=0;s<8;++s){const int k=s*16+2*t;
     qa[s][0]=tch_pack(q0[k],q0[k+1]);qa[s][1]=tch_pack(q1[k],q1[k+1]);qa[s][2]=tch_pack(q0[k+8],q0[k+9]);qa[s][3]=tch_pack(q1[k+8],q1[k+9]);}}
  float o[16][4];
#pragma unroll
  for(int n=0;n<16;++n){o[n][0]=o[n][1]=o[n][2]=o[n][3]=0.f;}
  float maximum=-CUDART_INF_F,total=0.f;
  __syncthreads();
  const int nkeys=full_count+3;
  for(int base=0;base<nkeys;base+=TCL_KEYS){
    if(ctid<TCL_KEYS){const int key=base+ctid;const int slot=key<full_count?key:slots-3+key-full_count;
      pos[tid]=key<nkeys?selected[(long long)query*slots+slot]:-1;}
    __syncthreads();
    if constexpr(FP8){
      const uint8_t* lat8=reinterpret_cast<const uint8_t*>(latent);
      for(int i=ctid;i<TCL_KEYS*64;i+=128*HG){const int key=i>>6,col=(i&63)*8;const long long p=pos[key];
        half hv[8];
        if(p>=0){const uint8_t* row=lat8+p*KV8_ROW;const float sc=reinterpret_cast<const float*>(row+512)[col>>7];
          const uint2 raw=*reinterpret_cast<const uint2*>(row+col);const uint8_t* b=reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
          for(int e=0;e<8;++e){__nv_fp8_e4m3 v;v.__x=b[e];hv[e]=__float2half_rn(float(v)*sc);}}
        else{
#pragma unroll
          for(int e=0;e<8;++e)hv[e]=__float2half_rn(0.f);}
        *reinterpret_cast<uint4*>(keys+key*TCL_LD+col)=*reinterpret_cast<const uint4*>(hv);}
    }else{
      for(int i=ctid;i<TCL_KEYS*64;i+=128*HG){const int key=i>>6,col=(i&63)*8;const long long p=pos[key];
        *reinterpret_cast<uint4*>(keys+key*TCL_LD+col)=p>=0?*reinterpret_cast<const uint4*>(latent+p*512+col):make_uint4(0,0,0,0);}
    }
    __syncthreads();
    float c[4][4];
#pragma unroll
    for(int j=0;j<4;++j){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
#pragma unroll
    for(int s=0;s<8;++s){
#pragma unroll
      for(int j=0;j<4;++j){const half* kr=keys+(j*8+g)*TCL_LD+dbase+s*16+2*t;
        const uint32_t b0=*reinterpret_cast<const uint32_t*>(kr),b1=*reinterpret_cast<const uint32_t*>(kr+8);
        tch_mma16(c[j],qa[s][0],qa[s][1],qa[s][2],qa[s][3],b0,b1);}
    }
#pragma unroll
    for(int j=0;j<4;++j){spart[warp*16+g][j*8+2*t]=c[j][0];spart[warp*16+g][j*8+2*t+1]=c[j][1];
      spart[warp*16+g+8][j*8+2*t]=c[j][2];spart[warp*16+g+8][j*8+2*t+1]=c[j][3];}
    __syncthreads();
    {const int r=tid>>3,k0=(tid&7)*4;float sc[4],tmax=-CUDART_INF_F;
#pragma unroll
     for(int i=0;i<4;++i){const int k=k0+i;
       sc[i]=pos[k]>=0?((spart[r][k]+spart[16+r][k])+(spart[32+r][k]+spart[48+r][k]))*(1.f/16.f):-CUDART_INF_F;
       tmax=fmaxf(tmax,sc[i]);}
     tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,1));tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,2));
     tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,4));
     const float next=fmaxf(maximum,tmax),old_scale=total>0.f?expf(maximum-next):0.f;
     float psum=0.f;
#pragma unroll
     for(int i=0;i<4;++i){const float p=pos[k0+i]>=0?expf(sc[i]-next):0.f;prob[r][k0+i]=p;psum+=p;}
     psum+=__shfl_xor_sync(0xffffffff,psum,1);psum+=__shfl_xor_sync(0xffffffff,psum,2);psum+=__shfl_xor_sync(0xffffffff,psum,4);
     total=total*old_scale+psum;if(next>-CUDART_INF_F)maximum=next;
     if((tid&7)==0)rowscale[r]=old_scale;}
    __syncthreads();
    {const float s0=rowscale[g],s1=rowscale[g+8];
#pragma unroll
     for(int n=0;n<16;++n){o[n][0]*=s0;o[n][1]*=s0;o[n][2]*=s1;o[n][3]*=s1;}}
#pragma unroll
    for(int ks=0;ks<TCL_KEYS;ks+=16){
      const uint32_t a0=tch_pack(prob[g][ks+2*t],prob[g][ks+2*t+1]),a1=tch_pack(prob[g+8][ks+2*t],prob[g+8][ks+2*t+1]);
      const uint32_t a2=tch_pack(prob[g][ks+2*t+8],prob[g][ks+2*t+9]),a3=tch_pack(prob[g+8][ks+2*t+8],prob[g+8][ks+2*t+9]);
      const half* v0=keys+(ks+2*t)*TCL_LD+dbase+g;
#pragma unroll
      for(int n=0;n<16;++n){const half* v=v0+n*8;
        const uint32_t b0=tch_pack_h(v[0],v[TCL_LD]),b1=tch_pack_h(v[8*TCL_LD],v[9*TCL_LD]);
        tch_mma16(o[n],a0,a1,a2,a3,b0,b1);}
    }
    __syncthreads();
  }
  __shared__ float rtotal_all[HG][16];float* rtotal=rtotal_all[hg];
  if((tid&7)==0)rtotal[tid>>3]=total;
  __syncthreads();
  const float i0=1.f/rtotal[g],i1=1.f/rtotal[g+8];
  float* out0=out+((long long)(h0+g)*queries+query)*512+dbase+2*t;
  float* out1=out+((long long)(h0+g+8)*queries+query)*512+dbase+2*t;
#pragma unroll
  for(int n=0;n<16;++n){*reinterpret_cast<float2*>(out0+n*8)=make_float2(o[n][0]*i0,o[n][1]*i0);
    *reinterpret_cast<float2*>(out1+n*8)=make_float2(o[n][2]*i1,o[n][3]*i1);}
}
template<int HG>
__global__ void __launch_bounds__(128*HG) latent_tc16_pf(const float* __restrict__ q,const half* __restrict__ latent,
    const long long* __restrict__ selected,float* __restrict__ out,int heads,int queries,int slots,int query_stride){
  // GLM53_MLA_TC16_HG=2: HG groups of 16 heads per CTA share each gathered key tile (one gather per query instead of
  // one per 16 heads); group hg's four warps run latent_tc16's exact per-head arithmetic (L0).
  const int hg=threadIdx.x>>7,query=blockIdx.x,h0=(blockIdx.y*HG+hg)*16,warp=(threadIdx.x>>5)&3,lane=threadIdx.x&31,g=lane>>2,t=lane&3,tid=threadIdx.x&127,ctid=threadIdx.x;
  extern __shared__ __align__(16) unsigned char tchg_smem[];
  // GLM53_MLA_TC16_PF=1: two key buffers; tile b+1's gather (cp.async) is in flight during tile b's math. Same
  // arithmetic as latent_tc16_hg (L0); zero rows for missing keys as before (cp.async zero-fill).
  half* kbuf=reinterpret_cast<half*>(tchg_smem);
  float (*spart)[TCL_KEYS+1]=reinterpret_cast<float(*)[TCL_KEYS+1]>(tchg_smem+2*TCL_KEYS*TCL_LD*2+(size_t)hg*4*16*(TCL_KEYS+1)*4);
  float (*prob)[TCL_KEYS+4]=reinterpret_cast<float(*)[TCL_KEYS+4]>(tchg_smem+2*TCL_KEYS*TCL_LD*2+(size_t)HG*4*16*(TCL_KEYS+1)*4+(size_t)hg*16*(TCL_KEYS+4)*4);
  __shared__ long long posb[2][TCL_KEYS];
  __shared__ float rowscale_all[HG][16];float* rowscale=rowscale_all[hg];
  __shared__ int full_count;
  if(ctid==0){int lo=0,hi=slots-3;
    while(lo<hi){int mid=(lo+hi)/2;if(selected[(long long)query*slots+mid]>=0)lo=mid+1;else hi=mid;}
    full_count=lo;}
  const int dbase=warp*128;
  uint32_t qa[8][4];   // A fragments over this warp's 128 dims: 8 k16-steps
  {const float* q0=q+(long long)(h0+g)*query_stride+(long long)query*512+dbase;const float* q1=q0+8LL*query_stride;
#pragma unroll
   for(int s=0;s<8;++s){const int k=s*16+2*t;
     qa[s][0]=tch_pack(q0[k],q0[k+1]);qa[s][1]=tch_pack(q1[k],q1[k+1]);qa[s][2]=tch_pack(q0[k+8],q0[k+9]);qa[s][3]=tch_pack(q1[k+8],q1[k+9]);}}
  float o[16][4];
#pragma unroll
  for(int n=0;n<16;++n){o[n][0]=o[n][1]=o[n][2]=o[n][3]=0.f;}
  float maximum=-CUDART_INF_F,total=0.f;
  __syncthreads();
  const int nkeys=full_count+3,ntiles=(nkeys+TCL_KEYS-1)/TCL_KEYS;
  auto load_pos=[&](int tile,int b){if(ctid<TCL_KEYS){const int key=tile*TCL_KEYS+ctid;const int slot=key<full_count?key:slots-3+key-full_count;
      posb[b][ctid]=key<nkeys?selected[(long long)query*slots+slot]:-1;}};
  auto issue=[&](int b){half* kb=kbuf+(size_t)b*TCL_KEYS*TCL_LD;
    for(int i=ctid;i<TCL_KEYS*64;i+=128*HG){const int key=i>>6,col=(i&63)*8;const long long p=posb[b][key];
      const unsigned d=(unsigned)__cvta_generic_to_shared(kb+key*TCL_LD+col);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(d),"l"(latent+(p>=0?p:0)*512+col),"r"(p>=0?16:0));}
    asm volatile("cp.async.commit_group;\n"::);};
  load_pos(0,0);__syncthreads();issue(0);
  for(int tile=0;tile<ntiles;++tile){
    const int b=tile&1;const half* keys=kbuf+(size_t)b*TCL_KEYS*TCL_LD;const long long* pos=posb[b];
    if(tile+1<ntiles){load_pos(tile+1,b^1);__syncthreads();issue(b^1);asm volatile("cp.async.wait_group 1;\n"::);}
    else asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    float c[4][4];
#pragma unroll
    for(int j=0;j<4;++j){c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f;}
#pragma unroll
    for(int s=0;s<8;++s){
#pragma unroll
      for(int j=0;j<4;++j){const half* kr=keys+(j*8+g)*TCL_LD+dbase+s*16+2*t;
        const uint32_t b0=*reinterpret_cast<const uint32_t*>(kr),b1=*reinterpret_cast<const uint32_t*>(kr+8);
        tch_mma16(c[j],qa[s][0],qa[s][1],qa[s][2],qa[s][3],b0,b1);}
    }
#pragma unroll
    for(int j=0;j<4;++j){spart[warp*16+g][j*8+2*t]=c[j][0];spart[warp*16+g][j*8+2*t+1]=c[j][1];
      spart[warp*16+g+8][j*8+2*t]=c[j][2];spart[warp*16+g+8][j*8+2*t+1]=c[j][3];}
    __syncthreads();
    {const int r=tid>>3,k0=(tid&7)*4;float sc[4],tmax=-CUDART_INF_F;
#pragma unroll
     for(int i=0;i<4;++i){const int k=k0+i;
       sc[i]=pos[k]>=0?((spart[r][k]+spart[16+r][k])+(spart[32+r][k]+spart[48+r][k]))*(1.f/16.f):-CUDART_INF_F;
       tmax=fmaxf(tmax,sc[i]);}
     tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,1));tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,2));
     tmax=fmaxf(tmax,__shfl_xor_sync(0xffffffff,tmax,4));
     const float next=fmaxf(maximum,tmax),old_scale=total>0.f?expf(maximum-next):0.f;
     float psum=0.f;
#pragma unroll
     for(int i=0;i<4;++i){const float p=pos[k0+i]>=0?expf(sc[i]-next):0.f;prob[r][k0+i]=p;psum+=p;}
     psum+=__shfl_xor_sync(0xffffffff,psum,1);psum+=__shfl_xor_sync(0xffffffff,psum,2);psum+=__shfl_xor_sync(0xffffffff,psum,4);
     total=total*old_scale+psum;if(next>-CUDART_INF_F)maximum=next;
     if((tid&7)==0)rowscale[r]=old_scale;}
    __syncthreads();
    {const float s0=rowscale[g],s1=rowscale[g+8];
#pragma unroll
     for(int n=0;n<16;++n){o[n][0]*=s0;o[n][1]*=s0;o[n][2]*=s1;o[n][3]*=s1;}}
#pragma unroll
    for(int ks=0;ks<TCL_KEYS;ks+=16){
      const uint32_t a0=tch_pack(prob[g][ks+2*t],prob[g][ks+2*t+1]),a1=tch_pack(prob[g+8][ks+2*t],prob[g+8][ks+2*t+1]);
      const uint32_t a2=tch_pack(prob[g][ks+2*t+8],prob[g][ks+2*t+9]),a3=tch_pack(prob[g+8][ks+2*t+8],prob[g+8][ks+2*t+9]);
      const half* v0=keys+(ks+2*t)*TCL_LD+dbase+g;
#pragma unroll
      for(int n=0;n<16;++n){const half* v=v0+n*8;
        const uint32_t b0=tch_pack_h(v[0],v[TCL_LD]),b1=tch_pack_h(v[8*TCL_LD],v[9*TCL_LD]);
        tch_mma16(o[n],a0,a1,a2,a3,b0,b1);}
    }
    __syncthreads();
  }
  __shared__ float rtotal_all[HG][16];float* rtotal=rtotal_all[hg];
  if((tid&7)==0)rtotal[tid>>3]=total;
  __syncthreads();
  const float i0=1.f/rtotal[g],i1=1.f/rtotal[g+8];
  float* out0=out+((long long)(h0+g)*queries+query)*512+dbase+2*t;
  float* out1=out+((long long)(h0+g+8)*queries+query)*512+dbase+2*t;
#pragma unroll
  for(int n=0;n<16;++n){*reinterpret_cast<float2*>(out0+n*8)=make_float2(o[n][0]*i0,o[n][1]*i0);
    *reinterpret_cast<float2*>(out1+n*8)=make_float2(o[n][2]*i1,o[n][3]*i1);}
}
extern "C" int glm53_shared_latent_cuda(const float* q,const void* latent,const long long* selected,
    float* out,int heads,int queries,int slots,int query_stride,int mode,cudaStream_t stream){
    if(slots<3)return int(cudaErrorInvalidValue);
    if(mode==3)shared_latent<4,16><<<dim3(queries,(heads+3)/4),128,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);
    else if(mode==4)shared_latent<8,16><<<dim3(queries,(heads+7)/8),256,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);
    else if(mode==5)shared_latent<8,32><<<dim3(queries,(heads+7)/8),256,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);
    else if(mode==6)shared_latent<8,16,true><<<dim3(queries,(heads+7)/8),256,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);
    else if(mode==8){if(heads%16)return int(cudaErrorInvalidValue);latent_tc<false><<<dim3(queries,heads/16),128,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);}
    else if((mode==10||mode==11)&&heads%32==0&&[]{const char* e=std::getenv("GLM53_MLA_TC16_HG");return e&&e[0]=='2';}()){
      constexpr size_t sm=(size_t)TCL_KEYS*TCL_LD*2+2*4*16*(TCL_KEYS+1)*4+2*16*(TCL_KEYS+4)*4;
      static bool set=false;if(!set){cudaFuncSetAttribute(latent_tc16_hg<false,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
        cudaFuncSetAttribute(latent_tc16_hg<true,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);set=true;}
      static const bool pf=[]{const char* e=std::getenv("GLM53_MLA_TC16_PF");return e&&e[0]=='1';}();
      if(mode==10&&pf){constexpr size_t sm2=sm+(size_t)TCL_KEYS*TCL_LD*2;static bool set2=false;
        if(!set2){cudaFuncSetAttribute(latent_tc16_pf<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm2);set2=true;}
        latent_tc16_pf<2><<<dim3(queries,heads/32),256,sm2,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);}
      else if(mode==10)latent_tc16_hg<false,2><<<dim3(queries,heads/32),256,sm,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);
      else latent_tc16_hg<true,2><<<dim3(queries,heads/32),256,sm,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);}
    else if(mode==10||mode==11){if(heads%16)return int(cudaErrorInvalidValue);
      if(mode==10)latent_tc16<false><<<dim3(queries,heads/16),128,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);
      else latent_tc16<true><<<dim3(queries,heads/16),128,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);}
    else if(mode==9){if(heads%16)return int(cudaErrorInvalidValue);latent_tc<true><<<dim3(queries,heads/16),128,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);}
    else if(mode==7)shared_latent<8,16,true,true><<<dim3(queries,(heads+7)/8),256,0,stream>>>(q,(const half*)latent,selected,out,heads,queries,slots,query_stride);
    else return int(cudaErrorInvalidValue);
    return int(cudaGetLastError());
}

// ---- W03 (GLM53_MLA_HALF_BMM=1): batched skinny MMA, Y[h][m][n] = sum_k half(X[h][m][k]) * W[h][n][k]
// X Float with (head,row) strides, W Half [heads][N][K] with head stride, FP32 accumulate, FP32 out.
// Same register-fragment scheme as the FP8 skinny kernel (lane loads 8 contiguous K per row).
__device__ __forceinline__ void lat_mma(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint32_t lat_h2(float a,float b){half2 h=__floats2half2_rn(a,b);return *reinterpret_cast<uint32_t*>(&h);}
template<int KS>
__global__ void __launch_bounds__(256) latent_half_bmm(const float* __restrict__ X,long long xh,long long xm,const __half* __restrict__ W,long long wh,
    float* __restrict__ Y,int M,int N,int K){
  constexpr int RT=8/KS;
  __shared__ float red[KS][RT][4][32];
  const int head=blockIdx.y,warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  const int rt=warp/KS,ks=warp%KS,rbase=(blockIdx.x*RT+rt)*16;
  const float* x=X+head*xh;const __half* w=W+head*wh;
  float c[4]={0.f,0.f,0.f,0.f};
  const int chunks=K>>5,c0=chunks*ks/KS,c1=chunks*(ks+1)/KS;
  const bool live=rbase<N;const int r0=min(rbase+g,N-1),r1=min(rbase+g+8,N-1);
  const bool xv=g<M;const float* xr=x+(xv?g:0)*xm+t*8;
  if(live){
    for(int cc=c0;cc<c1;++cc){
      const uint4 a=*reinterpret_cast<const uint4*>(w+(long long)r0*K+cc*32+t*8);
      const uint4 b=*reinterpret_cast<const uint4*>(w+(long long)r1*K+cc*32+t*8);
      uint32_t xb[4]={0,0,0,0};
      if(xv){const float4 p=*reinterpret_cast<const float4*>(xr+cc*32),q=*reinterpret_cast<const float4*>(xr+cc*32+4);
        xb[0]=lat_h2(p.x,p.y);xb[1]=lat_h2(p.z,p.w);xb[2]=lat_h2(q.x,q.y);xb[3]=lat_h2(q.z,q.w);}
      lat_mma(c,a.x,b.x,a.y,b.y,xb[0],xb[1]);
      lat_mma(c,a.z,b.z,a.w,b.w,xb[2],xb[3]);
    }
  }
#pragma unroll
  for(int i=0;i<4;i++)red[ks][rt][i][lane]=c[i];
  __syncthreads();
  if(ks==0&&live){
#pragma unroll
    for(int i=0;i<4;i++){float s=red[0][rt][i][lane];
#pragma unroll
      for(int q=1;q<KS;q++)s+=red[q][rt][i][lane];
      const int row=rbase+g+(i>>1)*8,tok=2*t+(i&1);
      if(row<N&&tok<M)Y[((long long)head*M+tok)*N+row]=s;}
  }
}
extern "C" int glm53_latent_half_bmm_cuda(const float* x,long long xh,long long xm,const void* w,long long wh,float* y,int heads,int m,int n,int k,cudaStream_t s){
  if(m<1||m>8||n%16||k%32||heads<1)return int(cudaErrorInvalidValue);
  const int ks=(k>=512)?4:2;const int rt=8/ks;
  dim3 grid((n+16*rt-1)/(16*rt),heads);
  if(ks==4)latent_half_bmm<4><<<grid,256,0,s>>>(x,xh,xm,(const __half*)w,wh,y,m,n,k);
  else latent_half_bmm<2><<<grid,256,0,s>>>(x,xh,xm,(const __half*)w,wh,y,m,n,k);
  return int(cudaGetLastError());
}

// ---- W07 (GLM53_STATE_INPLACE=1): copy rows [floor(*lo/div), ceil(*hi/div)) of a row-major cache.
// Grid.x bounds the row count (host knows node+1 <= 8 new tokens); rows outside the range are skipped.
__global__ void range_cache_copy(const uint4* __restrict__ src,uint4* __restrict__ dst,const long long* lo,const long long* hi,int row_vectors,int divisor,int capacity){
  const long long first=*lo/divisor,end=(*hi+divisor-1)/divisor;
  const long long row=first+blockIdx.x;if(row>=end||row>=capacity)return;
  for(int i=threadIdx.x;i<row_vectors;i+=blockDim.x)dst[row*row_vectors+i]=src[row*row_vectors+i];
}
extern "C" int glm53_range_cache_copy_cuda(const void* src,void* dst,const long long* lo,const long long* hi,int capacity,int row_vectors,int divisor,int max_rows,cudaStream_t s){
  if(max_rows<1||row_vectors<1||divisor<1)return int(cudaErrorInvalidValue);
  range_cache_copy<<<max_rows,128,0,s>>>((const uint4*)src,(uint4*)dst,lo,hi,row_vectors,divisor,capacity);return int(cudaGetLastError());
}

// ---- Temperature sampling noise (Gumbel-max). noise[r,c] = T_r * G(seed_r, pos_r, c_global), a pure
// function of (seed, output position, global vocabulary id): TP shards and replicated heads agree,
// and the sampled token at an output position does not depend on the speculative structure.
struct GumbelRows {float temp[64];unsigned long long key[64];};
__device__ __forceinline__ unsigned long long glm53_mix64(unsigned long long x){
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;return x;}
__global__ void gumbel_noise(float* __restrict__ out,long long stride,int cols,long long col_offset,GumbelRows p){
  int r=blockIdx.y;float t=p.temp[r];unsigned long long key=p.key[r];
  for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<cols;c+=gridDim.x*blockDim.x){
    float v=0.f;
    if(t>0.f){unsigned long long h=glm53_mix64(key^glm53_mix64((unsigned long long)(c+col_offset)+0x9e3779b97f4a7c15ULL));
      // G=-log(E), E=-log(1-r) ~ Exp(1), r in (0,1) with 53-bit resolution in double. A float u near 1
      // rounds to 1.0 and turns __logf chains into +inf noise (always-sampled columns), so no float here.
      double r=((double)(h>>11)+0.5)*(1.0/9007199254740992.0);double e=-log1p(-r);v=(float)(-(double)t*log(e));}
    out[r*stride+c]=v;}
}
// Coupled drafting (GLM53_DRAFT_COUPLED=1): the same T*G(key_t, id) the verifier row for output position t adds to
// the target logits, evaluated at the drafter's 16 candidate ids per position (same hash, same double math).
__global__ void gumbel_candidates(const long long* __restrict__ ids,const float* __restrict__ temps,const long long* __restrict__ keys,float* __restrict__ out,int positions){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=positions*16)return;
  const int t=i/16;const float temp=temps[t];float v=0.f;
  if(temp>0.f){unsigned long long h=glm53_mix64((unsigned long long)keys[t]^glm53_mix64((unsigned long long)ids[i]+0x9e3779b97f4a7c15ULL));
    double r=((double)(h>>11)+0.5)*(1.0/9007199254740992.0);double e=-log1p(-r);v=(float)(-(double)temp*log(e));}
  out[i]=v;
}
extern "C" int glm53_gumbel_candidates_cuda(const long long* ids,const float* temps,const long long* keys,float* out,int positions,cudaStream_t s){
  if(positions<1||positions>64)return int(cudaErrorInvalidValue);
  gumbel_candidates<<<(positions*16+127)/128,128,0,s>>>(ids,temps,keys,out,positions);return int(cudaGetLastError());
}
extern "C" int glm53_gumbel_noise_cuda(float* out,long long stride,int rows,int cols,long long col_offset,const float* temps,const unsigned long long* keys,cudaStream_t s){
  if(rows<=0)return 0;if(rows>64)return 1;GumbelRows p;
  for(int i=0;i<rows;i++){p.temp[i]=temps[i];p.key[i]=keys[i];}
  dim3 grid((cols+255)/256<64?(cols+255)/256:64,rows);gumbel_noise<<<grid,256,0,s>>>(out,stride,cols,col_offset,p);return int(cudaGetLastError());
}

// ---- GLM53_GUMBEL_FUSED=1: verifier top-1 packet straight from the local head values, the Gumbel noise evaluated on
// the fly (no [rows, width] noise buffer, no FP64 pass over the whole slice). The result is bitwise the old
// perturb (values + noise) -> max_dim -> packet: every block first scores its columns with an FP32 estimate of the noise
// whose error is far below eps, then evaluates the exact FP64 noise (same expression as gumbel_noise) only for columns
// whose estimate is within eps of the block maximum, and keeps ATen's max rule (NaN wins, larger wins, lowest index on
// ties). Per-row temperature/key live in a small device buffer written by gumbel_params (graph replays read it).
__device__ __forceinline__ float gumbel_exact(unsigned long long key,long long id,float t){
  if(!(t>0.f))return 0.f;
  unsigned long long h=glm53_mix64(key^glm53_mix64((unsigned long long)id+0x9e3779b97f4a7c15ULL));
  double r=((double)(h>>11)+0.5)*(1.0/9007199254740992.0);double e=-log1p(-r);return (float)(-(double)t*log(e));
}
__device__ __forceinline__ float gumbel_estimate(unsigned long long key,long long id,float t){
  if(!(t>0.f))return 0.f;
  unsigned long long h=glm53_mix64(key^glm53_mix64((unsigned long long)id+0x9e3779b97f4a7c15ULL));
  const unsigned long long m=h>>11;float e;
  if(m<(1ULL<<52)){const float r=((float)m+0.5f)*1.1102230246251565e-16f;e=-log1pf(-r);}
  else{const float u=((float)((1ULL<<53)-m)-0.5f)*1.1102230246251565e-16f;e=-logf(u);}
  return -t*logf(e);
}
__device__ __forceinline__ bool gumbel_better(float v,int i,float bv,int bi){
  const bool vn=v!=v,bn=bv!=bv;
  if(vn||bn)return vn&&(!bn||i<bi);
  return v>bv||(v==bv&&i<bi);
}
__global__ void gumbel_params(float* __restrict__ temps,unsigned long long* __restrict__ keys,int rows,GumbelRows p){
  const int i=threadIdx.x;if(i<64){temps[i]=i<rows?p.temp[i]:0.f;keys[i]=i<rows?p.key[i]:0ULL;}
}
extern "C" int glm53_gumbel_params_cuda(float* temps,unsigned long long* keys,int rows,const float* t,const unsigned long long* k,cudaStream_t s){
  if(rows<0||rows>64)return int(cudaErrorInvalidValue);GumbelRows p;
  for(int i=0;i<64;i++){p.temp[i]=i<rows?t[i]:0.f;p.key[i]=i<rows?k[i]:0ULL;}
  gumbel_params<<<1,64,0,s>>>(temps,keys,rows,p);return int(cudaGetLastError());
}
__global__ void __launch_bounds__(256) gumbel_argmax_part(const float* __restrict__ values,int width,long long col_offset,
    const float* __restrict__ temps,const unsigned long long* __restrict__ keys,float* __restrict__ part_v,int* __restrict__ part_i){
  const int r=blockIdx.y,nb=gridDim.x,chunk=(width+nb-1)/nb,c0=blockIdx.x*chunk,c1=min(width,c0+chunk);
  const float t=temps[r];const unsigned long long key=keys[r];const float* row=values+(long long)r*width;
  __shared__ float sm[8];__shared__ float sv[8];__shared__ int si[8];
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  // 1: lower bound of the block maximum from the estimates
  float lo=-INFINITY;
  for(int c=c0+threadIdx.x;c<c1;c+=256){const float s=row[c]+gumbel_estimate(key,c+col_offset,t);
    const float eps=(t>0.f&&isfinite(s))?t*1e-4f+1e-5f*fabsf(s):0.f;if(s==s)lo=fmaxf(lo,s-eps);}
  for(int o=16;o;o>>=1)lo=fmaxf(lo,__shfl_xor_sync(0xffffffffu,lo,o));
  if(lane==0)sm[warp]=lo;__syncthreads();
  lo=sm[0];for(int w=1;w<8;++w)lo=fmaxf(lo,sm[w]);
  // 2: exact values for the columns that can reach it (and every NaN)
  float bv=-INFINITY;int bi=0x7fffffff;
  for(int c=c0+threadIdx.x;c<c1;c+=256){const float l=row[c];const float s=l+gumbel_estimate(key,c+col_offset,t);
    const float eps=(t>0.f&&isfinite(s))?t*1e-4f+1e-5f*fabsf(s):0.f;
    if(s!=s||s+eps>=lo){const float x=l+gumbel_exact(key,c+col_offset,t);if(bi==0x7fffffff||gumbel_better(x,c,bv,bi)){bv=x;bi=c;}}}
  for(int o=16;o;o>>=1){const float ov=__shfl_xor_sync(0xffffffffu,bv,o);const int oi=__shfl_xor_sync(0xffffffffu,bi,o);
    if(oi!=0x7fffffff&&(bi==0x7fffffff||gumbel_better(ov,oi,bv,bi))){bv=ov;bi=oi;}}
  if(lane==0){sv[warp]=bv;si[warp]=bi;}__syncthreads();
  if(threadIdx.x==0){for(int w=1;w<8;++w)if(si[w]!=0x7fffffff&&(bi==0x7fffffff||gumbel_better(sv[w],si[w],bv,bi))){bv=sv[w];bi=si[w];}
    part_v[r*nb+blockIdx.x]=bv;part_i[r*nb+blockIdx.x]=bi;}
}
// packet [world, rows, 2]: zeros except this rank's lane = (best value, float(global id)).
__global__ void gumbel_argmax_packet(const float* __restrict__ part_v,const int* __restrict__ part_i,int nb,int rows,int world,int rank,
    long long start,float* __restrict__ packet){
  const int i=threadIdx.x;
  for(int j=i;j<world*rows*2;j+=blockDim.x)packet[j]=0.f;
  __syncthreads();
  if(i<rows){float bv=-INFINITY;int bi=0x7fffffff;
    for(int b=0;b<nb;++b){const int ci=part_i[i*nb+b];if(ci!=0x7fffffff&&(bi==0x7fffffff||gumbel_better(part_v[i*nb+b],ci,bv,bi))){bv=part_v[i*nb+b];bi=ci;}}
    packet[((long long)rank*rows+i)*2]=bv;packet[((long long)rank*rows+i)*2+1]=(float)(bi+start);}
}
extern "C" int glm53_gumbel_argmax_packet_cuda(const float* values,int rows,int width,long long start,const float* temps,const unsigned long long* keys,
    float* part_v,int* part_i,float* packet,int world,int rank,cudaStream_t s){
  if(rows<1||rows>64||width<1||world<1||rank<0||rank>=world)return int(cudaErrorInvalidValue);
  const int nb=32;
  gumbel_argmax_part<<<dim3(nb,rows),256,0,s>>>(values,width,start,temps,keys,part_v,part_i);
  gumbel_argmax_packet<<<1,256,0,s>>>(part_v,part_i,nb,rows,world,rank,start,packet);
  return int(cudaGetLastError());
}
