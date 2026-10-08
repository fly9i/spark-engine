#include <cstring>
// SPDX-License-Identifier: MIT
// Ranked index bookkeeping only. Score arithmetic and sorted ATen topk stay
// outside these kernels. Device pos remains dynamic on every graph replay.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__global__ void dsa_index_mask(const float* scores,const int64_t* pos,float* masked,int pools) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pools)return;
    // The caller's checked sequence contract proves pos >= 0 and no overflow.
    const int64_t complete=(pos[0]+1)/4;
    // Bit-copy active NaNs, signed zeros, subnormals and infinities; inactive
    // values must be -FLT_MAX, not -Inf, to preserve the original topk ties.
    reinterpret_cast<uint32_t*>(masked)[i]=static_cast<int64_t>(i)<complete
        ?reinterpret_cast<const uint32_t*>(scores)[i]:0xff7fffffu;
}

__global__ void dsa_index_expand(const int64_t* selected,const int64_t* pos,int64_t* out,int k) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=4*k+3)return;
    const int64_t len=pos[0]+1,complete=len/4;
    if(i<4*k) {
        const int64_t pool=selected[i/4];
        // Keep sorted topk's original pool order, including padding choices.
        out[i]=pool<complete?pool*4+i%4:-1;
    } else {
        const int tail=i-4*k;
        out[i]=tail<len%4?complete*4+tail:-1;
    }
}

// Row-batched dsa_index_mask_capture (GLM53_MLA_NODE_BATCH): row r reads its own pre-append length pointer (a chain
// node's parent len, final once every append is enqueued), same per-element formula and sidecar write.
struct PosPtrs {const int64_t* p[8];};
__global__ void dsa_index_mask_capture_rows(const float* scores,const PosPtrs pos,float* masked,int64_t* captured_pos,int pools) {
    const int r=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pools)return;
    const int64_t oldpos=pos.p[r][0];
    if(i==0)captured_pos[r]=oldpos;
    const int64_t complete=(oldpos+1)/4;
    reinterpret_cast<uint32_t*>(masked+(int64_t)r*pools)[i]=static_cast<int64_t>(i)<complete
        ?reinterpret_cast<const uint32_t*>(scores+(int64_t)r*pools)[i]:0xff7fffffu;
}
extern "C" int glm53_dsa_index_mask_capture_rows_cuda(const float* scores,const int64_t* const* pos,float* masked,int64_t* captured,int pools,int rows,cudaStream_t stream) {
    if(!scores||!pos||!masked||!captured||pools<1||pools>0x7fffff00||rows<1||rows>8)return static_cast<int>(cudaErrorInvalidValue);
    PosPtrs pp{};for(int r=0;r<rows;++r){if(!pos[r])return static_cast<int>(cudaErrorInvalidValue);pp.p[r]=pos[r];}
    dsa_index_mask_capture_rows<<<dim3((pools+255)/256,rows),256,0,stream>>>(scores,pp,masked,captured,pools);
    return static_cast<int>(cudaGetLastError());
}
// Row-batched dsa_index_expand (GLM53_MLA_NODE_BATCH): row r of selected [rows,k] with pos[r] into out[r] [4k+3],
// the same per-element formula.
__global__ void dsa_index_expand_rows(const int64_t* selected,const int64_t* pos,int64_t* out,int k) {
    const int r=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=4*k+3)return;
    const int64_t len=pos[r]+1,complete=len/4;
    int64_t* o=out+(int64_t)r*(4*k+3);
    if(i<4*k) {const int64_t pool=selected[(int64_t)r*k+i/4];o[i]=pool<complete?pool*4+i%4:-1;}
    else {const int tail=i-4*k;o[i]=tail<len%4?complete*4+tail:-1;}
}
extern "C" int glm53_dsa_index_expand_rows_cuda(const int64_t* selected,const int64_t* pos,int64_t* out,int k,int rows,cudaStream_t stream) {
    if(!selected||!pos||!out||k<1||k>512||rows<1||rows>64)return static_cast<int>(cudaErrorInvalidValue);
    dsa_index_expand_rows<<<dim3((4*k+3+255)/256,rows),256,0,stream>>>(selected,pos,out,k);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int glm53_dsa_index_mask_cuda(const float* scores,const int64_t* pos,
    float* masked,int pools,cudaStream_t stream) {
    if(!scores||!pos||!masked||pools<1||pools>0x7fffff00)
        return static_cast<int>(cudaErrorInvalidValue);
    dsa_index_mask<<<(pools+255)/256,256,0,stream>>>(scores,pos,masked,pools);
    return static_cast<int>(cudaGetLastError());
}
extern "C" int glm53_dsa_index_expand_cuda(const int64_t* selected,const int64_t* pos,
    int64_t* out,int k,cudaStream_t stream) {
    if(!selected||!pos||!out||k<1||k>512)
        return static_cast<int>(cudaErrorInvalidValue);
    dsa_index_expand<<<(4*k+3+255)/256,256,0,stream>>>(selected,pos,out,k);
    return static_cast<int>(cudaGetLastError());
}

// New ABI only. The original mask and expand kernels/entry points are retained.
// All blocks read pre-append pos on one stream; the following len update cannot
// run until this kernel completes. Only thread zero owns the sidecar write.
__global__ void dsa_index_mask_capture(const float* scores,const int64_t* pos,
    float* masked,int64_t* captured_pos,int pools) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pools)return;
    const int64_t oldpos=pos[0];
    if(i==0)captured_pos[0]=oldpos;
    const int64_t complete=(oldpos+1)/4;
    reinterpret_cast<uint32_t*>(masked)[i]=static_cast<int64_t>(i)<complete
        ?reinterpret_cast<const uint32_t*>(scores)[i]:0xff7fffffu;
}
extern "C" int glm53_dsa_index_mask_capture_cuda(const float* scores,const int64_t* pos,
    float* masked,int64_t* captured_pos,int pools,cudaStream_t stream) {
    if(!scores||!pos||!masked||!captured_pos||captured_pos==pos||pools<1||pools>0x7fffff00)
        return static_cast<int>(cudaErrorInvalidValue);
    dsa_index_mask_capture<<<(pools+255)/256,256,0,stream>>>(scores,pos,masked,captured_pos,pools);
    return static_cast<int>(cudaGetLastError());
}

// ---- W06-lite (GLM53_DSA_NODE_FUSED=1): one launch per verifier node replaces tail copies,
// tail slot writes, provisional pool (softmax over four slots) + pool write, latent row write
// and len+1. Child latent/pools already hold the parent's active rows (active copies).
__global__ void dsa_node_append(const long long* __restrict__ parent_len,const float* __restrict__ ptk,const float* __restrict__ ptg,
    float* __restrict__ ctk,float* __restrict__ ctg,float* __restrict__ cpools,const float* __restrict__ ape,
    const float* __restrict__ krow,const float* __restrict__ grow,__half* __restrict__ clatent,const __half* __restrict__ lrow,
    long long* __restrict__ clen,int dim,int lat,float* __restrict__ crow){
    const long long pos=*parent_len;const int off=(int)(pos%4);const int tid=threadIdx.x;
    if(tid<dim){
        const int d=tid;float kk[4],gg[4];
#pragma unroll
        for(int j=0;j<4;++j){kk[j]=j==off?krow[d]:ptk[j*dim+d];gg[j]=j==off?grow[d]:ptg[j*dim+d];ctk[j*dim+d]=kk[j];ctg[j*dim+d]=gg[j];}
        float x[4],m=-INFINITY;
#pragma unroll
        for(int j=0;j<4;++j){x[j]=gg[j]+ape[j*dim+d];m=fmaxf(m,x[j]);}
        float e[4],s=0.f;
#pragma unroll
        for(int j=0;j<4;++j){e[j]=expf(x[j]-m);s+=e[j];}
        float pool=0.f;
#pragma unroll
        for(int j=0;j<4;++j)pool+=(e[j]/s)*kk[j];
        cpools[(pos/4)*dim+d]=pool;
        if(crow)crow[d]=pool;              // chain-shared nodes keep their own copy of this row
    }
    for(int i=tid;i<lat;i+=blockDim.x)clatent[pos*lat+i]=lrow[i];
    if(tid==0)*clen=pos+1;
}
// GLM53_MLA_NODE_BATCH: a chain's node appends in one launch. One block walks the nodes in order with the per-node body
// above (lat == 0: FP8 latent rows are stored separately); node i's parent pointers are node i-1's outputs, visible after
// the barrier. Same per-thread arithmetic, same writes.
struct NodeAppendPtrs {const long long* parent_len;const float* ptk;const float* ptg;float* ctk;float* ctg;float* cpools;
    const float* krow;const float* grow;long long* clen;float* crow;};
struct NodeAppendChain {NodeAppendPtrs n[8];};
__global__ void dsa_node_append_chain(const NodeAppendChain c,int nodes,const float* __restrict__ ape,int dim){
    const int tid=threadIdx.x;
    for(int q=0;q<nodes;++q){
        const NodeAppendPtrs& a=c.n[q];
        const long long pos=*a.parent_len;const int off=(int)(pos%4);
        if(tid<dim){
            const int d=tid;float kk[4],gg[4];
#pragma unroll
            for(int j=0;j<4;++j){kk[j]=j==off?a.krow[d]:a.ptk[j*dim+d];gg[j]=j==off?a.grow[d]:a.ptg[j*dim+d];a.ctk[j*dim+d]=kk[j];a.ctg[j*dim+d]=gg[j];}
            float x[4],m=-INFINITY;
#pragma unroll
            for(int j=0;j<4;++j){x[j]=gg[j]+ape[j*dim+d];m=fmaxf(m,x[j]);}
            float e[4],s=0.f;
#pragma unroll
            for(int j=0;j<4;++j){e[j]=expf(x[j]-m);s+=e[j];}
            float pool=0.f;
#pragma unroll
            for(int j=0;j<4;++j)pool+=(e[j]/s)*kk[j];
            a.cpools[(pos/4)*dim+d]=pool;
            if(a.crow)a.crow[d]=pool;
        }
        if(tid==0)*a.clen=pos+1;
        __syncthreads();
    }
}
extern "C" int glm53_dsa_node_append_chain_cuda(const void* ptrs,int nodes,const float* ape,int dim,cudaStream_t s){
    if(!ptrs||nodes<1||nodes>8||dim<1||dim>1024)return int(cudaErrorInvalidValue);
    NodeAppendChain c{};memcpy(c.n,ptrs,sizeof(NodeAppendPtrs)*nodes);
    dsa_node_append_chain<<<1,dim>256?dim:256,0,s>>>(c,nodes,ape,dim);
    return int(cudaGetLastError());
}
// GLM53_MLA_MULTI_LAUNCH (L0): the chain node appends of up to 8 sequences in one launch, one block per sequence running
// dsa_node_append_chain's per-node body in node order (each sequence's chain only reads its own nodes' outputs).
struct NodeAppendMulti {NodeAppendPtrs n[8][8];int nodes[8];};
__global__ void dsa_node_append_chain_multi(const NodeAppendMulti c,const float* __restrict__ ape,int dim){
    const int sidx=blockIdx.x,tid=threadIdx.x;
    for(int q=0;q<c.nodes[sidx];++q){
        const NodeAppendPtrs& a=c.n[sidx][q];
        const long long pos=*a.parent_len;const int off=(int)(pos%4);
        if(tid<dim){
            const int d=tid;float kk[4],gg[4];
#pragma unroll
            for(int j=0;j<4;++j){kk[j]=j==off?a.krow[d]:a.ptk[j*dim+d];gg[j]=j==off?a.grow[d]:a.ptg[j*dim+d];a.ctk[j*dim+d]=kk[j];a.ctg[j*dim+d]=gg[j];}
            float x[4],m=-INFINITY;
#pragma unroll
            for(int j=0;j<4;++j){x[j]=gg[j]+ape[j*dim+d];m=fmaxf(m,x[j]);}
            float e[4],s=0.f;
#pragma unroll
            for(int j=0;j<4;++j){e[j]=expf(x[j]-m);s+=e[j];}
            float pool=0.f;
#pragma unroll
            for(int j=0;j<4;++j)pool+=(e[j]/s)*kk[j];
            a.cpools[(pos/4)*dim+d]=pool;
            if(a.crow)a.crow[d]=pool;
        }
        if(tid==0)*a.clen=pos+1;
        __syncthreads();
    }
}
extern "C" int glm53_dsa_node_append_chain_multi_cuda(const void* ptrs,const int* nodes,int nseq,const float* ape,int dim,cudaStream_t s){
    if(!ptrs||!nodes||nseq<1||nseq>8||dim<1||dim>1024)return int(cudaErrorInvalidValue);
    NodeAppendMulti c{};
    for(int g=0;g<nseq;++g){if(nodes[g]<1||nodes[g]>8)return int(cudaErrorInvalidValue);c.nodes[g]=nodes[g];
        memcpy(c.n[g],(const char*)ptrs+(size_t)g*8*sizeof(NodeAppendPtrs),sizeof(NodeAppendPtrs)*nodes[g]);}
    dsa_node_append_chain_multi<<<nseq,dim>256?dim:256,0,s>>>(c,ape,dim);
    return int(cudaGetLastError());
}
extern "C" int glm53_dsa_node_append_cuda(const long long* parent_len,const float* ptk,const float* ptg,float* ctk,float* ctg,float* cpools,
    const float* ape,const float* krow,const float* grow,void* clatent,const void* lrow,long long* clen,int dim,int lat,float* crow,cudaStream_t s){
    if(dim<1||dim>1024||lat<0)return int(cudaErrorInvalidValue);
    dsa_node_append<<<1,dim>256?dim:256,0,s>>>(parent_len,ptk,ptg,ctk,ctg,cpools,ape,krow,grow,(__half*)clatent,(const __half*)lrow,clen,dim,lat,crow);
    return int(cudaGetLastError());
}
// GLM53_DSA_TOPK_FAST=1: sorted top-k (k<=512) of each masked row [rows, pools], one block per row, with the exact
// result of ATen's sorted largest topk: order = value descending, ties by pool index ascending (NaN largest). Only the
// valid prefix [0, complete) is read; the -FLT_MAX tail the mask kernel wrote beyond it is folded in as one bucket of
// identical keys at indices complete.. (it is never scanned). Radix select (4 x 8-bit digits) finds the k-th key, keys
// above it are gathered, equal keys are taken in index order, then one bitonic sort of (key, ~index) composites.
__device__ __forceinline__ unsigned dsa_ordkey(unsigned b){
    if((b&0x7f800000u)==0x7f800000u&&(b&0x007fffffu))return 0xffffffffu;
    return (b&0x80000000u)?~b:(b|0x80000000u);
}
// ATen's order (measured, dsa-topk-fast-probe, widths >= 1024): selection by converted bits (-0 below +0), then a stable
// sort by value (-0 == +0) of the gathered list, which holds the keys above the k-th in index order followed by the keys
// equal to it in index order. Composite: value key with -0 folded onto +0, then the gather group, then ~index.
__device__ __forceinline__ unsigned long long dsa_sortkey(unsigned key,unsigned idx,bool above){
    return ((unsigned long long)(key==0x7fffffffu?0x80000000u:key)<<32)|(above?0x80000000u:0u)|((~idx)&0x7fffffffu);
}
__global__ void __launch_bounds__(1024) dsa_topk_rows(const float* __restrict__ masked,const int64_t* __restrict__ pos,
    int64_t* __restrict__ selected,int pools,int k){
    __shared__ unsigned hist[256];__shared__ unsigned long long cand[512];__shared__ unsigned wsum[32];
    __shared__ unsigned s_prefix,s_mask;__shared__ int s_need,s_gt,s_taken;
    const int r=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const unsigned* row=reinterpret_cast<const unsigned*>(masked+(int64_t)r*pools);
    const int64_t complete=(pos[r]+1)/4;
    const int n=(int)(complete<0?0:(complete>pools?pools:complete)),pad=pools-n;
    const unsigned kp=dsa_ordkey(0xff7fffffu);
    if(tid==0){s_prefix=0;s_mask=0;s_need=k;s_gt=0;s_taken=0;}
    for(int d=3;d>=0;--d){
        if(tid<256)hist[tid]=0;
        __syncthreads();
        const unsigned prefix=s_prefix,mask=s_mask;const int sh=8*d;
        for(int i=tid;i<n;i+=1024){const unsigned key=dsa_ordkey(__ldg(row+i));if((key&mask)==prefix)atomicAdd(&hist[(key>>sh)&255u],1u);}
        if(tid==0&&pad>0&&(kp&mask)==prefix)atomicAdd(&hist[(kp>>sh)&255u],(unsigned)pad);
        __syncthreads();
        if(warp==0){
            // lane l owns digits 255-8l .. 248-8l (descending); find the digit where the running count reaches need.
            unsigned c[8];unsigned s=0;
#pragma unroll
            for(int j=0;j<8;++j){c[j]=hist[255-8*lane-j];s+=c[j];}
            unsigned inc=s;
#pragma unroll
            for(int off=1;off<32;off<<=1){const unsigned v=__shfl_up_sync(0xffffffffu,inc,off);if(lane>=off)inc+=v;}
            const unsigned exc=inc-s;const unsigned need=(unsigned)s_need;
            const unsigned hit=__ballot_sync(0xffffffffu,inc>=need);const int first=__ffs(hit)-1;
            if(lane==first){unsigned run=exc;int j=0;for(;j<8;++j){if(run+c[j]>=need)break;run+=c[j];}
                const unsigned digit=255u-8u*lane-j;s_prefix=prefix|(digit<<sh);s_mask=mask|(255u<<sh);s_need=(int)(need-run);}
        }
        __syncthreads();
    }
    const unsigned T=s_prefix;const int need=s_need;const int gt=k-need;
    // keys above T (order irrelevant: the sort below is total)
    for(int i=tid;i<n;i+=1024){const unsigned key=dsa_ordkey(__ldg(row+i));
        if(key>T){const int at=atomicAdd(&s_gt,1);cand[at]=dsa_sortkey(key,(unsigned)i,true);}}
    if(kp>T){for(int j=tid;j<pad;j+=1024){const int at=atomicAdd(&s_gt,1);cand[at]=dsa_sortkey(kp,(unsigned)(n+j),true);}}
    __syncthreads();
    // keys equal to T: the first `need` in index order
    for(int base=0;base<n&&s_taken<need;base+=1024){
        const int i=base+tid;const bool f=i<n&&dsa_ordkey(__ldg(row+i))==T;
        const unsigned b=__ballot_sync(0xffffffffu,f);if(lane==0)wsum[warp]=__popc(b);
        __syncthreads();
        if(warp==0){unsigned v=wsum[lane],inc=v;
            for(int off=1;off<32;off<<=1){const unsigned u=__shfl_up_sync(0xffffffffu,inc,off);if(lane>=off)inc+=u;}
            wsum[lane]=inc-v;}
        __syncthreads();
        const int taken=s_taken;
        if(f){const int at=taken+(int)wsum[warp]+__popc(b&((1u<<lane)-1u));if(at<need)cand[gt+at]=dsa_sortkey(T,(unsigned)i,false);}
        __syncthreads();
        if(tid==1023)s_taken=taken+(int)wsum[31]+__popc(b);
        __syncthreads();
    }
    if(kp==T){const int taken=s_taken;for(int j=tid;j<need-taken;j+=1024)cand[gt+taken+j]=dsa_sortkey(kp,(unsigned)(n+j),false);}
    if(tid>=k&&tid<512)cand[tid]=0ull;
    __syncthreads();
    // bitonic sort, descending
    for(int size=2;size<=512;size<<=1){
        for(int stride=size>>1;stride>0;stride>>=1){
            if(tid<512){const int p=tid^stride;
                if(p>tid){const unsigned long long a=cand[tid],b=cand[p];const bool desc=((tid&size)==0);
                    if(desc?(a<b):(a>b)){cand[tid]=b;cand[p]=a;}}}
            __syncthreads();
        }
    }
    for(int j=tid;j<k;j+=1024)selected[(int64_t)r*k+j]=(int64_t)((~(unsigned)cand[j])&0x7fffffffu);
}
extern "C" int glm53_dsa_topk_rows_cuda(const float* masked,const int64_t* pos,int64_t* selected,int pools,int k,int rows,cudaStream_t s){
    if(!masked||!pos||!selected||pools<1024||pools>0x7fffff00||k<1||k>512||rows<1||rows>64)return int(cudaErrorInvalidValue);
    dsa_topk_rows<<<rows,1024,0,s>>>(masked,pos,selected,pools,k);
    return int(cudaGetLastError());
}
// GLM53_DSA_PREFILL_TOPK_FAST=1: the same two kernels for prefill query blocks (rows up to 65535): dsa_topk_rows reads only
// each row's visible prefix (the masked_fill + ATen topk of DSA::append_chunk, bitwise for widths >= 1024), then
// dsa_index_expand_rows writes the [rows, 4k+3] token rows (the ids*4+slot / tail / cat of append_chunk).
extern "C" int glm53_dsa_prefill_topk_expand_cuda(const float* scores,const int64_t* pos,int64_t* selected,int64_t* out,int pools,int k,int rows,cudaStream_t s){
    if(!scores||!pos||!selected||!out||pools<1024||pools>0x7fffff00||k<1||k>512||rows<1||rows>65535)return int(cudaErrorInvalidValue);
    dsa_topk_rows<<<rows,1024,0,s>>>(scores,pos,selected,pools,k);
    dsa_index_expand_rows<<<dim3((4*k+3+255)/256,rows),256,0,s>>>(selected,pos,out,k);
    return int(cudaGetLastError());
}
// GLM53_DSA_TOPK_FAST: when dsa_score_multi wrote the scores straight into the masked rows (it writes -FLT_MAX at and
// beyond each row's complete count, exactly what dsa_index_mask_capture_rows would copy), only the positions are captured.
struct PosPtrs8 {const int64_t* p[8];};
__global__ void dsa_capture_rows(const PosPtrs8 pos,int64_t* captured,int rows){const int r=threadIdx.x;if(r<rows)captured[r]=pos.p[r][0];}
struct PosPtrs64 {const int64_t* p[64];};
__global__ void dsa_capture_rows64(const PosPtrs64 pos,int64_t* captured,int rows){const int r=threadIdx.x;if(r<rows)captured[r]=pos.p[r][0];}
extern "C" int glm53_dsa_capture_rows64_cuda(const int64_t* const* pos,int64_t* captured,int rows,cudaStream_t s){
    if(!pos||!captured||rows<1||rows>64)return int(cudaErrorInvalidValue);
    PosPtrs64 pp{};for(int r=0;r<rows;++r){if(!pos[r])return int(cudaErrorInvalidValue);pp.p[r]=pos[r];}
    dsa_capture_rows64<<<1,64,0,s>>>(pp,captured,rows);return int(cudaGetLastError());
}
extern "C" int glm53_dsa_capture_rows_cuda(const int64_t* const* pos,int64_t* captured,int rows,cudaStream_t s){
    if(!pos||!captured||rows<1||rows>8)return int(cudaErrorInvalidValue);
    PosPtrs8 pp{};for(int r=0;r<rows;++r){if(!pos[r])return int(cudaErrorInvalidValue);pp.p[r]=pos[r];}
    dsa_capture_rows<<<1,32,0,s>>>(pp,captured,rows);return int(cudaGetLastError());
}
