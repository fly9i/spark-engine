#include <cstdlib>
// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
// TensorIterator sum along a short, noncontiguous dimension uses four
// independent accumulators then combines them in order. No FMA (--fmad=false).
__global__ void grouped_reduce(const half* x,const half* w,float* y,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*4096)return;
    int row=i/4096,col=i%4096;float a[4]={0,0,0,0};
    #pragma unroll
    for(int slot=0;slot<8;++slot)
        a[slot%4]+=__half2float(x[(row*8+slot)*4096+col])*__half2float(w[row*8+slot]);
    y[i]=((a[0]+a[1])+a[2])+a[3];
}
extern "C" int glm53_grouped_reduce_cuda(const void* x,const void* w,float* y,int rows,cudaStream_t stream) {
    if(rows<=0)return int(cudaErrorInvalidValue);
    grouped_reduce<<<(rows*4096+255)/256,256,0,stream>>>((const half*)x,(const half*)w,y,rows);
    return int(cudaGetLastError());
}
// P3b: the same fixed-order 8-slot sum, then one FP32 add of the shared-expert partial
// (fl(routed+shared), identical to the separate ATen add), written straight to `y`.
__global__ void grouped_reduce_add(const half* x,const half* w,const float* add,float* y,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*4096)return;
    int row=i/4096,col=i%4096;float a[4]={0,0,0,0};
    #pragma unroll
    for(int slot=0;slot<8;++slot)
        a[slot%4]+=__half2float(x[(row*8+slot)*4096+col])*__half2float(w[row*8+slot]);
    float routed=((a[0]+a[1])+a[2])+a[3];
    y[i]=routed+add[i];
}
extern "C" int glm53_grouped_reduce_add_cuda(const void* x,const void* w,const float* add,float* y,int rows,cudaStream_t stream) {
    if(rows<=0)return int(cudaErrorInvalidValue);
    grouped_reduce_add<<<(rows*4096+255)/256,256,0,stream>>>((const half*)x,(const half*)w,add,y,rows);
    return int(cudaGetLastError());
}

// P2: device-side route grouping for the fat MoE (replaces the host BTreeMap). Stable counting
// sort of the R = rows*8 assignments by expert: dest order is expert ascending, assignment index
// ascending within an expert, exactly the host order. Segments of <= tile rows per expert, in
// expert order. Layout (rs_fatmoe): rows64 = dest[R], token[R]; rows32 = row_expert[R],
// seg_expert[S], seg_row0[S], seg_rows[S], num_rows, num_segs (S = capacity >= used segments).
// One CTA of 32 warps; warp w owns the contiguous slice [w*R/32, (w+1)*R/32).
#define RG_WARPS 32
#define RG_MAX_EXP 320
__global__ void __launch_bounds__(RG_WARPS*32) route_group(const long long* ids,int R,int n_exp,int tile,int S,
    long long* rows64,int* rows32,int* error) {
    __shared__ int cnt[RG_WARPS][RG_MAX_EXP];
    __shared__ int base[RG_MAX_EXP+1],segbase[RG_MAX_EXP+1];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int i=threadIdx.x;i<RG_WARPS*RG_MAX_EXP;i+=blockDim.x)(&cnt[0][0])[i]=0;
    __syncthreads();
    const int per=(R+RG_WARPS-1)/RG_WARPS,lo=min(R,warp*per),hi=min(R,lo+per);
    // 1. per-warp histogram (warp-aggregated: one smem add per distinct expert per step)
    for(int j=lo;j<hi;j+=32){
        int a=j+lane;bool valid=a<hi;int e=valid?(int)ids[a]:-1;
        if(valid&&(e<0||e>=n_exp))atomicExch(error,1);
        if(valid&&(e<0||e>=n_exp))e=-1;
        unsigned active=__ballot_sync(0xffffffffu,e>=0);
        unsigned peers=__match_any_sync(0xffffffffu,e)&active;
        if(e>=0&&lane==__ffs(peers)-1)cnt[warp][e]+=__popc(peers);
        __syncwarp();
    }
    __syncthreads();
    // 2. per-expert totals and exclusive scans (experts in order; segments per expert)
    for(int e=threadIdx.x;e<n_exp;e+=blockDim.x){   // column scan over warps (relative offsets)
        int c=0;for(int w=0;w<RG_WARPS;++w){int v=cnt[w][e];cnt[w][e]=c;c+=v;}
        base[e]=c;segbase[e]=(c+tile-1)/tile;
    }
    __syncthreads();
    if(threadIdx.x==0){   // exclusive scans over experts (in expert order)
        int acc=0,sacc=0;
        for(int e=0;e<n_exp;++e){int c=base[e],n=segbase[e];base[e]=acc;segbase[e]=sacc;acc+=c;sacc+=n;}
        base[n_exp]=acc;segbase[n_exp]=sacc;
        if(sacc>S)atomicExch(error,2);
        rows32[R+3*S]=R;rows32[R+3*S+1]=min(sacc,S);
    }
    __syncthreads();
    for(int i=threadIdx.x;i<RG_WARPS*n_exp;i+=blockDim.x){int w=i/n_exp,e=i%n_exp;cnt[w][e]+=base[e];}
    __syncthreads();
    // 3. stable scatter: warp slices in order, and in-order within a warp (rank among equal lanes)
    for(int j=lo;j<hi;j+=32){
        int a=j+lane;bool valid=a<hi;int e=valid?(int)ids[a]:-1;
        if(e<0||e>=n_exp)e=-1;
        unsigned active=__ballot_sync(0xffffffffu,e>=0);
        unsigned peers=__match_any_sync(0xffffffffu,e)&active;
        if(e>=0){
            int pos=cnt[warp][e]+__popc(peers&((1u<<lane)-1u));
            rows64[pos]=a;rows64[R+pos]=a/8;rows32[pos]=e;
        }
        __syncwarp();
        if(e>=0&&lane==__ffs(peers)-1)cnt[warp][e]+=__popc(peers);
        __syncwarp();
    }
    // 4. segment table
    for(int e=threadIdx.x;e<n_exp;e+=blockDim.x){
        int c=base[e+1]-base[e],s=segbase[e];
        for(int off=0;off<c&&s<S;off+=tile,++s){
            rows32[R+s]=e;rows32[R+S+s]=base[e]+off;rows32[R+2*S+s]=min(tile,c-off);
        }
    }
}
extern "C" int glm53_route_group_cuda(const long long* ids,int R,int n_exp,int tile,int S,long long* rows64,int* rows32,int* error,cudaStream_t stream) {
    if(R<=0||n_exp<=0||n_exp>RG_MAX_EXP||tile<=0||S<=0)return int(cudaErrorInvalidValue);
    route_group<<<1,RG_WARPS*32,0,stream>>>(ids,R,n_exp,tile,S,rows64,rows32,error);
    return int(cudaGetLastError());
}

#include <cuda_bf16.h>
#include <math_constants.h>
__device__ float warp_sum(float x){for(int s=16;s;s>>=1)x+=__shfl_down_sync(0xffffffff,x,s);return x;}
__device__ float warp_max(float x){for(int s=16;s;s>>=1)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,s));return x;}
// Short-window tier. Read two segments directly, no cat, FP32 KV expansion,
// score/probability global buffers, or repeated KV heads.
// C3: kvh = KV heads held by this rank (8 full, 4 head-parallel TP2); query heads = 4*kvh.
__global__ void draft_attention(const __nv_bfloat16* q,const __nv_bfloat16* ck,const __nv_bfloat16* cv,
    const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int history,int n,int kvh){
    const int H=4*kvh;int query=blockIdx.x/H,head=blockIdx.x%H,kvhead=head/4;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,total=history+n;
    extern __shared__ float prob[];
    __shared__ float denom,maximum;
    float qq[4];
    #pragma unroll
    for(int j=0;j<4;++j)qq[j]=__bfloat162float(q[(query*H+head)*128+lane+32*j]);
    for(int token=warp;token<total;token+=4){
        bool valid=abs(history+query-token)<2048;float score=0.f;
        if(valid){const __nv_bfloat16* keys=token<history?ck+(token*kvh+kvhead)*128:k+((token-history)*kvh+kvhead)*128;
            #pragma unroll
            for(int j=0;j<4;++j)score=__fmaf_rn(qq[j],__bfloat162float(keys[lane+32*j]),score);
        }
        score=warp_sum(score);
        if(lane==0)prob[token]=valid?score*0.08838834764831844f:-CUDART_INF_F;
    }
    __syncthreads();
    if(warp==0){float m=-CUDART_INF_F;for(int token=lane;token<total;token+=32)m=fmaxf(m,prob[token]);m=warp_max(m);m=__shfl_sync(0xffffffff,m,0);
        if(lane==0)maximum=m;
        float sum=0.f;for(int token=lane;token<total;token+=32){float p=expf(prob[token]-m);prob[token]=p;sum+=p;}
        sum=warp_sum(sum);if(lane==0)denom=sum;
    }
    __syncthreads();float acc=0.f;
    for(int token=0;token<total;++token){
        if(abs(history+query-token)>=2048)continue;
        const __nv_bfloat16* values=token<history?cv+(token*kvh+kvhead)*128:v+((token-history)*kvh+kvhead)*128;
        acc=__fmaf_rn(prob[token]/denom,__bfloat162float(values[threadIdx.x]),acc);
    }
    out[(query*H+head)*128+threadIdx.x]=__float2bfloat16_rn(acc);
}
extern "C" int glm53_draft_attention_cuda(const void* q,const void* ck,const void* cv,const void* k,const void* v,void* out,int history,int n,int kvh,cudaStream_t stream){
    if(history<0 || history>2048 || n<1 || n>8 || (kvh!=8&&kvh!=4))return int(cudaErrorInvalidValue);
    draft_attention<<<n*4*kvh,128,(history+n)*sizeof(float),stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)ck,(const __nv_bfloat16*)cv,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,history,n,kvh);
    return int(cudaGetLastError());
}

// Fixed launch extent, device-read logical length: graph replay can shorten or
// extend the cache without a host synchronization. Only complete pools are read.
__global__ void dsa_score(const float* q,const float* pools,const float* mixing,const long long* pos,float* out,int capacity){
    int pool=blockIdx.x,complete=(*pos+1)/4;
    if(pool>=complete){if(threadIdx.x==0)out[pool]=-3.4028234663852886e38F;return;}
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    float key[4];for(int j=0;j<4;++j)key[j]=pools[pool*128+lane+32*j];
    float sum=0.f;
    for(int h=warp;h<32;h+=4){float dot=0.f;
        #pragma unroll
        for(int j=0;j<4;++j)dot=__fmaf_rn(q[h*128+lane+32*j],key[j],dot);
        dot=warp_sum(dot);
        if(lane==0)sum+=fmaxf(dot*0.08838834764831844f,0.f)*mixing[h];
    }
    __shared__ float partial[4];if(lane==0)partial[warp]=sum;__syncthreads();
    if(threadIdx.x==0)out[pool]=((partial[0]+partial[1])+partial[2])+partial[3];
}
extern "C" int glm53_dsa_score_cuda(const float* q,const float* pools,const float* mixing,const long long* pos,float* out,int capacity,cudaStream_t stream){
    if(capacity<=0)return int(cudaErrorInvalidValue);
    dsa_score<<<capacity,128,0,stream>>>(q,pools,mixing,pos,out,capacity);return int(cudaGetLastError());
}

// Four neighboring pools reuse each query load. Same per-dot and per-head
// accumulation order as dsa_score; no TF32 conversion or host length read.
template<int Tile>
__global__ void dsa_score_tile(const float* q,const float* pools,const float* mixing,const long long* pos,float* out,int capacity){
    int first=blockIdx.x*Tile,complete=(*pos+1)/4;
    if(first>=complete){if(threadIdx.x<Tile && first+threadIdx.x<capacity)out[first+threadIdx.x]=-3.4028234663852886e38F;return;}
    int lane=threadIdx.x%32,warp=threadIdx.x/32;float key[Tile][4],sum[Tile]={};
    #pragma unroll
    for(int p=0;p<Tile;++p)for(int j=0;j<4;++j)key[p][j]=first+p<complete?pools[(first+p)*128+lane+32*j]:0.f;
    for(int h=warp;h<32;h+=4){float dot[Tile]={};
        #pragma unroll
        for(int j=0;j<4;++j){float a=q[h*128+lane+32*j];
            #pragma unroll
            for(int p=0;p<Tile;++p)dot[p]=__fmaf_rn(a,key[p][j],dot[p]);}
        #pragma unroll
        for(int p=0;p<Tile;++p){dot[p]=warp_sum(dot[p]);if(lane==0)sum[p]+=fmaxf(dot[p]*0.08838834764831844f,0.f)*mixing[h];}
    }
    __shared__ float partial[Tile][4];
    if(lane==0){for(int p=0;p<Tile;++p)partial[p][warp]=sum[p];}__syncthreads();
    if(threadIdx.x<Tile && first+threadIdx.x<capacity){int p=threadIdx.x;
        out[first+p]=first+p<complete?((partial[p][0]+partial[p][1])+partial[p][2])+partial[p][3]:-3.4028234663852886e38F;}
}
// C1 (GLM53_DSA_SCORE_MULTI=1): the T chain nodes' score rows in one pass over the pools. Each block loads
// its pool tile once (rows < max complete) and runs dsa_score_tile's exact per-query arithmetic for every
// node (same FMA/warp_sum/head/partial order), writing -FLT_MAX at and beyond that node's complete count.
struct DsaPosList {const long long* p[8];};
template<int Tile>
__global__ void dsa_score_multi(const float* q,const float* pools,const float* mixing,DsaPosList pos,int T,float* out,int capacity){
    const int first=blockIdx.x*Tile;int comp[8];int maxc=0;
    #pragma unroll
    for(int t=0;t<8;++t){comp[t]=t<T?(int)((*pos.p[t]+1)/4):0;maxc=max(maxc,comp[t]);}
    if(first>=maxc){for(int t=0;t<T;++t)if(threadIdx.x<Tile && first+threadIdx.x<capacity)out[(long long)t*capacity+first+threadIdx.x]=-3.4028234663852886e38F;return;}
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;float key[Tile][4];
    #pragma unroll
    for(int p=0;p<Tile;++p)for(int j=0;j<4;++j)key[p][j]=first+p<maxc?pools[(first+p)*128+lane+32*j]:0.f;
    __shared__ float partial[Tile][4];
    for(int t=0;t<T;++t){
        const float* qt=q+(long long)t*32*128;const float* mt=mixing+t*32;float sum[Tile]={};
        if(first<comp[t]){
            for(int h=warp;h<32;h+=4){float dot[Tile]={};
                #pragma unroll
                for(int j=0;j<4;++j){float a=qt[h*128+lane+32*j];
                    #pragma unroll
                    for(int p=0;p<Tile;++p)dot[p]=__fmaf_rn(a,key[p][j],dot[p]);}
                #pragma unroll
                for(int p=0;p<Tile;++p){dot[p]=warp_sum(dot[p]);if(lane==0)sum[p]+=fmaxf(dot[p]*0.08838834764831844f,0.f)*mt[h];}
            }
            if(lane==0){for(int p=0;p<Tile;++p)partial[p][warp]=sum[p];}
        }
        __syncthreads();
        if(threadIdx.x<Tile && first+threadIdx.x<capacity){const int p=threadIdx.x;
            out[(long long)t*capacity+first+p]=first+p<comp[t]?((partial[p][0]+partial[p][1])+partial[p][2])+partial[p][3]:-3.4028234663852886e38F;}
        __syncthreads();
    }
}
// GLM53_MLA_MULTI_LAUNCH (L0): dsa_score_multi for up to 8 sequences in one launch (blockIdx.y = sequence, its own pools,
// node position pointers and first row of the stacked q/mixing/out); per-sequence body unchanged.
struct DsaSeqScore {const float* pools[8];const long long* p[8][8];int T[8];int first[8];int n;};
template<int Tile>
__global__ void dsa_score_multi_seq(const float* q,const float* mixing,const DsaSeqScore S,float* out,int capacity){
    const int sidx=blockIdx.y;if(sidx>=S.n)return;
    const int T=S.T[sidx];const float* pools=S.pools[sidx];
    q+=(long long)S.first[sidx]*32*128;mixing+=(long long)S.first[sidx]*32;out+=(long long)S.first[sidx]*capacity;
    const int first=blockIdx.x*Tile;int comp[8];int maxc=0;
    #pragma unroll
    for(int t=0;t<8;++t){comp[t]=t<T?(int)((*S.p[sidx][t]+1)/4):0;maxc=max(maxc,comp[t]);}
    if(first>=maxc){for(int t=0;t<T;++t)if(threadIdx.x<Tile && first+threadIdx.x<capacity)out[(long long)t*capacity+first+threadIdx.x]=-3.4028234663852886e38F;return;}
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;float key[Tile][4];
    #pragma unroll
    for(int p=0;p<Tile;++p)for(int j=0;j<4;++j)key[p][j]=first+p<maxc?pools[(first+p)*128+lane+32*j]:0.f;
    __shared__ float partial[Tile][4];
    for(int t=0;t<T;++t){
        const float* qt=q+(long long)t*32*128;const float* mt=mixing+t*32;float sum[Tile]={};
        if(first<comp[t]){
            for(int h=warp;h<32;h+=4){float dot[Tile]={};
                #pragma unroll
                for(int j=0;j<4;++j){float a=qt[h*128+lane+32*j];
                    #pragma unroll
                    for(int p=0;p<Tile;++p)dot[p]=__fmaf_rn(a,key[p][j],dot[p]);}
                #pragma unroll
                for(int p=0;p<Tile;++p){dot[p]=warp_sum(dot[p]);if(lane==0)sum[p]+=fmaxf(dot[p]*0.08838834764831844f,0.f)*mt[h];}
            }
            if(lane==0){for(int p=0;p<Tile;++p)partial[p][warp]=sum[p];}
        }
        __syncthreads();
        if(threadIdx.x<Tile && first+threadIdx.x<capacity){const int p=threadIdx.x;
            out[(long long)t*capacity+first+p]=first+p<comp[t]?((partial[p][0]+partial[p][1])+partial[p][2])+partial[p][3]:-3.4028234663852886e38F;}
        __syncthreads();
    }
}
extern "C" int glm53_dsa_score_multi_seq_cuda(const float* q,const float* mixing,const void* table,float* out,int capacity,int mode,cudaStream_t stream){
    const DsaSeqScore S=*reinterpret_cast<const DsaSeqScore*>(table);
    if(capacity<=0||S.n<1||S.n>8||mode<2||mode>4)return int(cudaErrorInvalidValue);
    for(int g=0;g<S.n;++g)if(S.T[g]<1||S.T[g]>8)return int(cudaErrorInvalidValue);
    if(mode==2)dsa_score_multi_seq<4><<<dim3((capacity+3)/4,S.n),128,0,stream>>>(q,mixing,S,out,capacity);
    else if(mode==3)dsa_score_multi_seq<8><<<dim3((capacity+7)/8,S.n),128,0,stream>>>(q,mixing,S,out,capacity);
    else dsa_score_multi_seq<16><<<dim3((capacity+15)/16,S.n),128,0,stream>>>(q,mixing,S,out,capacity);
    return int(cudaGetLastError());
}
extern "C" int glm53_dsa_score_multi_cuda(const float* q,const float* pools,const float* mixing,const long long* const* pos,int T,float* out,int capacity,int mode,cudaStream_t stream){
    if(capacity<=0||T<1||T>8||mode<2||mode>4)return int(cudaErrorInvalidValue);
    DsaPosList l{};for(int t=0;t<T;++t)l.p[t]=pos[t];
    if(mode==2)dsa_score_multi<4><<<(capacity+3)/4,128,0,stream>>>(q,pools,mixing,l,T,out,capacity);
    else if(mode==3)dsa_score_multi<8><<<(capacity+7)/8,128,0,stream>>>(q,pools,mixing,l,T,out,capacity);
    else dsa_score_multi<16><<<(capacity+15)/16,128,0,stream>>>(q,pools,mixing,l,T,out,capacity);
    return int(cudaGetLastError());
}
extern "C" int glm53_dsa_score_tile4_cuda(const float* q,const float* pools,const float* mixing,const long long* pos,float* out,int capacity,int mode,cudaStream_t stream){
    if(capacity<=0)return int(cudaErrorInvalidValue);
    if(mode==2)dsa_score_tile<4><<<(capacity+3)/4,128,0,stream>>>(q,pools,mixing,pos,out,capacity);
    else if(mode==3)dsa_score_tile<8><<<(capacity+7)/8,128,0,stream>>>(q,pools,mixing,pos,out,capacity);
    else dsa_score_tile<16><<<(capacity+15)/16,128,0,stream>>>(q,pools,mixing,pos,out,capacity);
    return int(cudaGetLastError());
}

#include <mma.h>
// Two TF32 components retain float input precision. Four Tensor Core products
// reconstruct (hi+lo)*(hi+lo); small terms accumulate separately. This trades
// arithmetic for less redundant query traffic and higher matrix throughput.
__global__ void dsa_score_tensor(const float* q,const float* pools,const float* mixing,const long long* pos,float* out,int capacity){
    using namespace nvcuda;
    int first=blockIdx.x*32,complete=(*pos+1)/4;
    if(first>=complete){if(threadIdx.x<32 && first+threadIdx.x<capacity)out[first+threadIdx.x]=-3.4028234663852886e38F;return;}
    int warp=threadIdx.x/32;
    __shared__ __align__(32) union {float stage[8192];float result[32*32];} mem;
    __shared__ float partial[128];
    float* qh=mem.stage;float* ql=qh+2048;float* ph=ql+2048;float* pl=ph+2048;
    wmma::fragment<wmma::accumulator,16,16,8,float> main,small;
    wmma::fill_fragment(main,0.f);wmma::fill_fragment(small,0.f);
    for(int base=0;base<128;base+=64){
        for(int i=threadIdx.x;i<2048;i+=128){int row=i/64,col=i%64;
            float a=q[row*128+base+col],b=first+row<complete?pools[(first+row)*128+base+col]:0.f;
            float ah=wmma::__float_to_tf32(a),bh=wmma::__float_to_tf32(b);
            qh[i]=ah;ql[i]=wmma::__float_to_tf32(a-ah);ph[i]=bh;pl[i]=wmma::__float_to_tf32(b-bh);
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<64;kk+=8){
            wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
            wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> bh,bl;
            wmma::load_matrix_sync(ah,qh+(warp/2)*16*64+kk,64);wmma::load_matrix_sync(al,ql+(warp/2)*16*64+kk,64);
            wmma::load_matrix_sync(bh,ph+(warp%2)*16*64+kk,64);wmma::load_matrix_sync(bl,pl+(warp%2)*16*64+kk,64);
            wmma::mma_sync(main,ah,bh,main);
            wmma::mma_sync(small,ah,bl,small);wmma::mma_sync(small,al,bh,small);wmma::mma_sync(small,al,bl,small);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<main.num_elements;++i)main.x[i]+=small.x[i];
    wmma::store_matrix_sync(mem.result+(warp/2)*16*32+(warp%2)*16,main,32,wmma::mem_row_major);__syncthreads();
    int pool=threadIdx.x%32;float sum=0.f;
    for(int h=warp;h<32;h+=4)sum+=fmaxf(mem.result[h*32+pool]*0.08838834764831844f,0.f)*mixing[h];
    partial[threadIdx.x]=sum;__syncthreads();
    if(threadIdx.x<32 && first+pool<capacity)out[first+pool]=first+pool<complete?((partial[pool]+partial[32+pool])+partial[64+pool])+partial[96+pool]:-3.4028234663852886e38F;
}
extern "C" int glm53_dsa_score_tensor_cuda(const float* q,const float* pools,const float* mixing,const long long* pos,float* out,int capacity,cudaStream_t stream){
    if(capacity<=0)return int(cudaErrorInvalidValue);
    dsa_score_tensor<<<(capacity+31)/32,128,0,stream>>>(q,pools,mixing,pos,out,capacity);return int(cudaGetLastError());
}

__global__ void grouped_swiglu(const half* gate,const half* up,half* out,int count,float limit){
    int i=blockIdx.x*256+threadIdx.x;if(i>=count)return;
    float g=__half2float(gate[i]),u=__half2float(up[i]);
    // Comparisons preserve NaNs, unlike fminf/fmaxf with a finite argument.
    g=g>limit?limit:g;u=u>limit?limit:u;u=u<-limit?-limit:u;
    out[i]=__float2half_rn((g/(1.f+expf(-g)))*u);
}
extern "C" int glm53_grouped_swiglu_cuda(const void* gate,const void* up,void* out,int count,float limit,cudaStream_t stream){
    if(count<=0)return int(cudaErrorInvalidValue);
    grouped_swiglu<<<(count+255)/256,256,0,stream>>>((const half*)gate,(const half*)up,(half*)out,count,limit);return int(cudaGetLastError());
}

__global__ void fp8_epilogue(float* y,const float* scale,int count,int cols,bool rounded){
    int i=blockIdx.x*256+threadIdx.x;if(i>=count)return;
    float value=y[i]*scale[i%cols];
    y[i]=rounded?__half2float(__float2half_rn(value)):value;
}
extern "C" int glm53_fp8_epilogue_cuda(float* y,const float* scale,int rows,int cols,int rounded,cudaStream_t stream){
    if(rows<=0 || cols<=0 || (long long)rows*cols>2147483647LL)return int(cudaErrorInvalidValue);
    int count=rows*cols;fp8_epilogue<<<(count+255)/256,256,0,stream>>>(y,scale,count,cols,rounded!=0);return int(cudaGetLastError());
}

// ---- W05 fused router (GLM53_ROUTER_FUSED=1): rows<=16, experts<=320, top-8 sigmoid + bias ----
// M4: FP32 or BF16 router weights (BF16 widened to FP32 is exact: same operands, L0).
__device__ __forceinline__ float4 rw4(const float* p){return *reinterpret_cast<const float4*>(p);}
__device__ __forceinline__ float4 rw4(const __nv_bfloat16* p){const uint2 u=*reinterpret_cast<const uint2*>(p);
  return make_float4(__uint_as_float(u.x<<16),__uint_as_float(u.x&0xffff0000u),__uint_as_float(u.y<<16),__uint_as_float(u.y&0xffff0000u));}
// S: K slices per expert (2 = original). GLM53_ROUTER_SPLIT=8 (L1: the slice sums are added in a fixed tree) gives four
// times the warps and a quarter of each warp's dependent load chain.
// MR: accumulator rows (GLM53_ROUTER_WIDE=1 launches MR=32 for 17..32 rows; each row's FMA order is unchanged, L0 per row).
template<typename W,int S=2,int MR=16>
__global__ void router_logits(const float* __restrict__ h,const W* __restrict__ w,float* __restrict__ partial,int rows,int experts,int k){
    const int warp=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(warp>=experts*S)return;
    const int e=warp/S,half=warp%S,span=k/S,start=half*span;
    const W* wr=w+(long long)e*k+start;float acc[MR]={};
    for(int c=lane*4;c<span;c+=128){const float4 wv=rw4(wr+c);
        for(int r=0;r<rows;++r){const float4 x=*reinterpret_cast<const float4*>(h+(long long)r*k+start+c);
            float a=acc[r];a=__fmaf_rn(x.x,wv.x,a);a=__fmaf_rn(x.y,wv.y,a);a=__fmaf_rn(x.z,wv.z,a);a=__fmaf_rn(x.w,wv.w,a);acc[r]=a;}}
    for(int r=0;r<rows;++r){float s=acc[r];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
        if(lane==0)partial[((long long)r*experts+e)*S+half]=s;}
}
template<int S>
__device__ __forceinline__ float router_logit(const float* __restrict__ p){
    if constexpr (S==2) return p[0]+p[1];
    else return ((p[0]+p[1])+(p[2]+p[3]))+((p[4]+p[5])+(p[6]+p[7]));
}
template<bool CG>
__device__ __forceinline__ void router_select_row(int r,int lane,const float* __restrict__ partial,const float* __restrict__ bias,long long* __restrict__ ids,float* __restrict__ weights,int experts,float scaling,__half* __restrict__ wh){
    float score[10],sel[10];
    for(int j=0;j<10;++j){const int e=lane+32*j;
        if(e<experts){const float p0=CG?__ldcg(partial+((long long)r*experts+e)*2):partial[((long long)r*experts+e)*2];
            const float p1=CG?__ldcg(partial+((long long)r*experts+e)*2+1):partial[((long long)r*experts+e)*2+1];const float l=p0+p1;
            score[j]=1.f/(1.f+expf(-l));sel[j]=score[j]+bias[e];}else{score[j]=0.f;sel[j]=-INFINITY;}}
    float chosen[8];int chosen_id[8];
    for(int n=0;n<8;++n){
        float best=-INFINITY;int bid=0x7fffffff;float bscore=0.f;
        for(int j=0;j<10;++j){const int e=lane+32*j;if(e<experts&&(sel[j]>best||(sel[j]==best&&e<bid))){best=sel[j];bid=e;bscore=score[j];}}
        for(int off=16;off;off>>=1){const float ob=__shfl_xor_sync(0xffffffff,best,off);const int oi=__shfl_xor_sync(0xffffffff,bid,off);const float os=__shfl_xor_sync(0xffffffff,bscore,off);
            if(ob>best||(ob==best&&oi<bid)){best=ob;bid=oi;bscore=os;}}
        chosen[n]=bscore;chosen_id[n]=bid;
        for(int j=0;j<10;++j)if(lane+32*j==bid)sel[j]=-INFINITY;
    }
    if(lane==0){float s=0.f;for(int n=0;n<8;++n)s+=chosen[n];s=fmaxf(s,1e-20f);
        for(int n=0;n<8;++n){ids[r*8+n]=chosen_id[n];const float v=(chosen[n]/s)*scaling;weights[r*8+n]=v;if(wh)wh[r*8+n]=__float2half_rn(v);}}
}
// GLM53_ROUTER_ONE=1: router_logits with each lane's 16 weight vectors loaded ahead (same per-lane FMA order), and the
// last block to finish runs router_select for every row (one warp per row, the same code). L0 against the two launches.
__device__ unsigned int g_router_done;
template<typename W>
__global__ void __launch_bounds__(256) router_one(const float* __restrict__ h,const W* __restrict__ w,float* __restrict__ partial,int rows,int experts,int k,
    const float* __restrict__ bias,long long* __restrict__ ids,float* __restrict__ weights,float scaling,__half* __restrict__ wh){
    const int warp=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
    if(warp<experts*2){
        const int e=warp>>1,half=warp&1,span=k/2,start=half*span;
        const W* wr=w+(long long)e*k+start;float acc[16]={};
        float4 wv[16];
#pragma unroll
        for(int it=0;it<16;++it){const int c=lane*4+it*128;wv[it]=c<span?rw4(wr+c):make_float4(0.f,0.f,0.f,0.f);}
#pragma unroll
        for(int it=0;it<16;++it){const int c=lane*4+it*128;if(c>=span)break;
            for(int r=0;r<rows;++r){const float4 x=*reinterpret_cast<const float4*>(h+(long long)r*k+start+c);
                float a=acc[r];a=__fmaf_rn(x.x,wv[it].x,a);a=__fmaf_rn(x.y,wv[it].y,a);a=__fmaf_rn(x.z,wv[it].z,a);a=__fmaf_rn(x.w,wv[it].w,a);acc[r]=a;}}
        for(int r=0;r<rows;++r){float s=acc[r];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
            if(lane==0)partial[((long long)r*experts+e)*2+half]=s;}
    }
    __threadfence();__syncthreads();
    __shared__ bool last;
    if(threadIdx.x==0){last=atomicAdd(&g_router_done,1u)==gridDim.x-1;if(last)g_router_done=0;}
    __syncthreads();
    if(!last)return;
    __threadfence();
    for(int r=threadIdx.x>>5;r<rows;r+=8)router_select_row<true>(r,lane,partial,bias,ids,weights,experts,scaling,wh);
}
template<int S=2>
__global__ void router_select(const float* __restrict__ partial,const float* __restrict__ bias,long long* __restrict__ ids,float* __restrict__ weights,int experts,float scaling,__half* __restrict__ wh=nullptr){
    const int r=blockIdx.x,lane=threadIdx.x;float score[10],sel[10];
    for(int j=0;j<10;++j){const int e=lane+32*j;
        if(e<experts){const float l=router_logit<S>(partial+((long long)r*experts+e)*S);
            score[j]=1.f/(1.f+expf(-l));sel[j]=score[j]+bias[e];}else{score[j]=0.f;sel[j]=-INFINITY;}}
    float chosen[8];int chosen_id[8];
    for(int n=0;n<8;++n){
        float best=-INFINITY;int bid=0x7fffffff;float bscore=0.f;
        for(int j=0;j<10;++j){const int e=lane+32*j;if(e<experts&&(sel[j]>best||(sel[j]==best&&e<bid))){best=sel[j];bid=e;bscore=score[j];}}
        for(int off=16;off;off>>=1){const float ob=__shfl_xor_sync(0xffffffff,best,off);const int oi=__shfl_xor_sync(0xffffffff,bid,off);const float os=__shfl_xor_sync(0xffffffff,bscore,off);
            if(ob>best||(ob==best&&oi<bid)){best=ob;bid=oi;bscore=os;}}
        chosen[n]=bscore;chosen_id[n]=bid;
        for(int j=0;j<10;++j)if(lane+32*j==bid)sel[j]=-INFINITY;
    }
    if(lane==0){float s=0.f;for(int n=0;n<8;++n)s+=chosen[n];s=fmaxf(s,1e-20f);
        for(int n=0;n<8;++n){ids[r*8+n]=chosen_id[n];const float v=(chosen[n]/s)*scaling;weights[r*8+n]=v;if(wh)wh[r*8+n]=__float2half_rn(v);}}
}
// GLM53_ROUTER_V2=1 (L0 against router_logits<W,2>): the same warp-per-(expert, K half) mapping, lane columns, per-lane
// FMA order (column step ascending, rows ascending, x.x..x.w) and shuffle tree, with the row count a template parameter
// (rows fully unrolled, so a step's activation loads issue together) and the lane's 16 weight vectors loaded up front.
template<typename W,int R>
__global__ void __launch_bounds__(256) router_logits_r(const float* __restrict__ h,const W* __restrict__ w,float* __restrict__ partial,int experts,int k){
    const int warp=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(warp>=experts*2)return;
    const int e=warp>>1,half=warp&1,span=k/2,start=half*span;   // span == 2048 (host-checked)
    const W* wr=w+(long long)e*k+start;
    float4 wv[16];
#pragma unroll
    for(int it=0;it<16;++it)wv[it]=rw4(wr+lane*4+it*128);
    float acc[R];
#pragma unroll
    for(int r=0;r<R;++r)acc[r]=0.f;
    const float* hb=h+start+lane*4;
#pragma unroll
    for(int it=0;it<16;++it){
#pragma unroll
        for(int r=0;r<R;++r){const float4 x=*reinterpret_cast<const float4*>(hb+(long long)r*k+it*128);
            float a=acc[r];a=__fmaf_rn(x.x,wv[it].x,a);a=__fmaf_rn(x.y,wv[it].y,a);a=__fmaf_rn(x.z,wv[it].z,a);a=__fmaf_rn(x.w,wv[it].w,a);acc[r]=a;}}
#pragma unroll
    for(int r=0;r<R;++r){float s=acc[r];for(int off=16;off;off>>=1)s+=__shfl_xor_sync(0xffffffff,s,off);
        if(lane==0)partial[((long long)r*experts+e)*2+half]=s;}
}
template<typename W>
static void router_logits_v2(const float* h,const W* w,float* partial,int rows,int experts,int k,cudaStream_t s){
    const int g=(experts*2+7)/8;
    switch(rows){
#define C(R) case R: router_logits_r<W,R><<<g,256,0,s>>>(h,w,partial,experts,k);break;
        C(1)C(2)C(3)C(4)C(5)C(6)C(7)C(8)C(9)C(10)C(11)C(12)C(13)C(14)C(15)C(16)
#undef C
    }
}
static bool router_v2(int k){static const bool on=[]{const char* e=std::getenv("GLM53_ROUTER_V2");return e&&e[0]=='1';}();return on&&k==4096;}
// GLM53_ROUTER_V2: router_select with each round's warp argmax as two redux instructions (max of the order-preserving
// key of sel, then min expert id among the lanes holding that key) instead of the 15-shuffle tree. Same winner (largest
// sel, ties to the lower expert; -0 is canonicalized to +0, as float == treats them), same selection order: L0.
__device__ __forceinline__ unsigned router_key(float v){unsigned u=__float_as_uint(v+0.0f);return (u&0x80000000u)?~u:(u|0x80000000u);}
template<int S=2>
__global__ void router_select_fast(const float* __restrict__ partial,const float* __restrict__ bias,long long* __restrict__ ids,float* __restrict__ weights,int experts,float scaling,__half* __restrict__ wh=nullptr){
    const int r=blockIdx.x,lane=threadIdx.x;float score[10];unsigned key[10];
#pragma unroll
    for(int j=0;j<10;++j){const int e=lane+32*j;
        if(e<experts){const float l=router_logit<S>(partial+((long long)r*experts+e)*S);
            score[j]=1.f/(1.f+expf(-l));key[j]=router_key(score[j]+bias[e]);}else{score[j]=0.f;key[j]=0u;}}
    float chosen[8];int chosen_id[8];
#pragma unroll
    for(int n=0;n<8;++n){
        unsigned bk=0u;int bj=0;
#pragma unroll
        for(int j=0;j<10;++j)if(key[j]>bk){bk=key[j];bj=j;}            // strict >: the lowest j (lowest expert) on ties
        const unsigned wk=__reduce_max_sync(0xffffffffu,bk);
        const unsigned cand=(bk==wk)?(unsigned)(lane+32*bj):0xffffffffu;
        const unsigned we=__reduce_min_sync(0xffffffffu,cand);
        float bs=0.f;
#pragma unroll
        for(int j=0;j<10;++j)if(lane+32*j==(int)we){bs=score[j];key[j]=0u;}
        chosen[n]=__shfl_sync(0xffffffffu,bs,we&31u);chosen_id[n]=(int)we;
    }
    // Unrolled (registers, not local memory); lanes 0..7 write one pick each, the sum in pick order as before.
    float sum=0.f;
#pragma unroll
    for(int n=0;n<8;++n)sum+=chosen[n];
    sum=fmaxf(sum,1e-20f);
#pragma unroll
    for(int n=0;n<8;++n)if(lane==n){ids[r*8+n]=chosen_id[n];const float v=(chosen[n]/sum)*scaling;weights[r*8+n]=v;if(wh)wh[r*8+n]=__float2half_rn(v);}
}
static bool router_wide(){static const bool on=[]{const char* e=std::getenv("GLM53_ROUTER_WIDE");return e&&e[0]=='1';}();return on;}
static int router_split(){static int v=[]{const char* e=std::getenv("GLM53_ROUTER_SPLIT");return (e&&e[0]=='8'&&e[1]==0)?8:2;}();return v;}
extern "C" int glm53_router_split(){return router_split();}
extern "C" int glm53_router_fused_cuda(const float* h,const void* w,int w_bf16,const float* bias,float* partial,long long* ids,float* weights,int rows,int experts,int k,float scaling,cudaStream_t s){
    if(rows<1||rows>(router_wide()?32:16)||experts<8||experts>320||k%256)return int(cudaErrorInvalidValue);
    if(rows>16){   // GLM53_ROUTER_WIDE (production split 2, no V2/ONE): one logits launch for up to 32 rows
        if(router_split()!=2)return int(cudaErrorInvalidValue);
        if(w_bf16)router_logits<__nv_bfloat16,2,32><<<(experts*2+7)/8,256,0,s>>>(h,(const __nv_bfloat16*)w,partial,rows,experts,k);
        else router_logits<float,2,32><<<(experts*2+7)/8,256,0,s>>>(h,(const float*)w,partial,rows,experts,k);
        router_select<<<rows,32,0,s>>>(partial,bias,ids,weights,experts,scaling,nullptr);
        return int(cudaGetLastError());
    }
    {const char* o=std::getenv("GLM53_ROUTER_ONE");if(o&&o[0]=='1'&&k/2<=16*128&&router_split()==2){
        __half* whp=nullptr;
        if(w_bf16)router_one<<<(experts*2+7)/8,256,0,s>>>(h,(const __nv_bfloat16*)w,partial,rows,experts,k,bias,ids,weights,scaling,whp);
        else router_one<<<(experts*2+7)/8,256,0,s>>>(h,(const float*)w,partial,rows,experts,k,bias,ids,weights,scaling,whp);
        return int(cudaGetLastError());}}
    if(router_split()==8){
        if(w_bf16)router_logits<__nv_bfloat16,8><<<(experts*8+7)/8,256,0,s>>>(h,(const __nv_bfloat16*)w,partial,rows,experts,k);
        else router_logits<float,8><<<(experts*8+7)/8,256,0,s>>>(h,(const float*)w,partial,rows,experts,k);
    }else{
    if(router_v2(k)){if(w_bf16)router_logits_v2(h,(const __nv_bfloat16*)w,partial,rows,experts,k,s);else router_logits_v2(h,(const float*)w,partial,rows,experts,k,s);}
    else if(w_bf16)router_logits<<<(experts*2+7)/8,256,0,s>>>(h,(const __nv_bfloat16*)w,partial,rows,experts,k);
    else router_logits<<<(experts*2+7)/8,256,0,s>>>(h,(const float*)w,partial,rows,experts,k);}
    if(router_split()==8)router_select<8><<<rows,32,0,s>>>(partial,bias,ids,weights,experts,scaling);
    else if(router_v2(k))router_select_fast<<<rows,32,0,s>>>(partial,bias,ids,weights,experts,scaling);
    else router_select<<<rows,32,0,s>>>(partial,bias,ids,weights,experts,scaling);
    return int(cudaGetLastError());
}

// GLM53_ROUTER_HALF_W=1: router_select also writes the Half RN copy of the weights that the cooperative expert kernel
// otherwise converts in a separate launch (same conversion, same value).
extern "C" int glm53_router_fused_h_cuda(const float* h,const void* w,int w_bf16,const float* bias,float* partial,long long* ids,float* weights,void* wh,int rows,int experts,int k,float scaling,cudaStream_t s){
    if(rows<1||rows>(router_wide()?32:16)||experts<8||experts>320||k%256||!wh)return int(cudaErrorInvalidValue);
    if(rows>16){   // GLM53_ROUTER_WIDE (production split 2, no V2/ONE): one logits launch for up to 32 rows
        if(router_split()!=2)return int(cudaErrorInvalidValue);
        if(w_bf16)router_logits<__nv_bfloat16,2,32><<<(experts*2+7)/8,256,0,s>>>(h,(const __nv_bfloat16*)w,partial,rows,experts,k);
        else router_logits<float,2,32><<<(experts*2+7)/8,256,0,s>>>(h,(const float*)w,partial,rows,experts,k);
        router_select<<<rows,32,0,s>>>(partial,bias,ids,weights,experts,scaling,(__half*)wh);
        return int(cudaGetLastError());
    }
    {const char* o=std::getenv("GLM53_ROUTER_ONE");if(o&&o[0]=='1'&&k/2<=16*128&&router_split()==2){
        __half* whp=(__half*)wh;
        if(w_bf16)router_one<<<(experts*2+7)/8,256,0,s>>>(h,(const __nv_bfloat16*)w,partial,rows,experts,k,bias,ids,weights,scaling,whp);
        else router_one<<<(experts*2+7)/8,256,0,s>>>(h,(const float*)w,partial,rows,experts,k,bias,ids,weights,scaling,whp);
        return int(cudaGetLastError());}}
    if(router_split()==8){
        if(w_bf16)router_logits<__nv_bfloat16,8><<<(experts*8+7)/8,256,0,s>>>(h,(const __nv_bfloat16*)w,partial,rows,experts,k);
        else router_logits<float,8><<<(experts*8+7)/8,256,0,s>>>(h,(const float*)w,partial,rows,experts,k);
    }else{
    if(router_v2(k)){if(w_bf16)router_logits_v2(h,(const __nv_bfloat16*)w,partial,rows,experts,k,s);else router_logits_v2(h,(const float*)w,partial,rows,experts,k,s);}
    else if(w_bf16)router_logits<<<(experts*2+7)/8,256,0,s>>>(h,(const __nv_bfloat16*)w,partial,rows,experts,k);
    else router_logits<<<(experts*2+7)/8,256,0,s>>>(h,(const float*)w,partial,rows,experts,k);}
    if(router_split()==8)router_select<8><<<rows,32,0,s>>>(partial,bias,ids,weights,experts,scaling,(__half*)wh);
    else if(router_v2(k))router_select_fast<<<rows,32,0,s>>>(partial,bias,ids,weights,experts,scaling,(__half*)wh);
    else router_select<<<rows,32,0,s>>>(partial,bias,ids,weights,experts,scaling,(__half*)wh);
    return int(cudaGetLastError());
}

// ---- P5 (GLM53_DSA_PREFILL_SCORE_FUSED=1): prefill DSA index scores without the [n,heads,pools] tensor ----
// score[q,p] = sum_h mixing[q,h] * relu(dot(q[q,h,:], pools[p,:]) * dim^-0.5), dim=128, heads=32.
// TF32 tensor cores (the same input precision as the TF32 matmul it replaces; summation order differs, L1).
// Block = 2 queries (64 rows = query x head) x 64 pools; 8 warps = 4 m-tiles x 2 halves of 32 pools.
__device__ __forceinline__ uint32_t pdsa_tf32(float x){uint32_t r;asm("cvt.rna.tf32.f32 %0,%1;":"=r"(r):"f"(x));return r;}
__device__ __forceinline__ void pdsa_mma(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
constexpr int PDSA_LD=132;
__global__ void __launch_bounds__(256) dsa_prefill_scores(const float* __restrict__ q,const float* __restrict__ mixing,
    const float* __restrict__ pools,float* __restrict__ out,int n,int active){
  extern __shared__ __align__(16) float pd_smem[];
  float* sq=pd_smem;                    // [64][132]
  float* sp=sq+64*PDSA_LD;              // [64][132]
  float* red=sp+64*PDSA_LD;             // [8][32]
  const int q0=blockIdx.y*2,p0=blockIdx.x*64,tid=threadIdx.x,warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
  for(int i=tid;i<64*32;i+=256){const int r=i>>5,c=(i&31)*4;const int qq=q0+(r>>5);
    float4 v=qq<n?*reinterpret_cast<const float4*>(q+((long long)qq*32+(r&31))*128+c):make_float4(0,0,0,0);
    *reinterpret_cast<float4*>(sq+r*PDSA_LD+c)=v;
    const int pp=p0+r;float4 w=pp<active?*reinterpret_cast<const float4*>(pools+(long long)pp*128+c):make_float4(0,0,0,0);
    *reinterpret_cast<float4*>(sp+r*PDSA_LD+c)=w;}
  __syncthreads();
  const int mt=warp&3,nh=warp>>2;       // m-tile (16 rows), pool half (32 pools)
  float acc[4][4];
#pragma unroll
  for(int j=0;j<4;++j){acc[j][0]=acc[j][1]=acc[j][2]=acc[j][3]=0.f;}
#pragma unroll 4
  for(int k=0;k<128;k+=8){
    const float* a=sq+(mt*16)*PDSA_LD+k;
    const uint32_t a0=pdsa_tf32(a[g*PDSA_LD+t]),a1=pdsa_tf32(a[(g+8)*PDSA_LD+t]),a2=pdsa_tf32(a[g*PDSA_LD+t+4]),a3=pdsa_tf32(a[(g+8)*PDSA_LD+t+4]);
#pragma unroll
    for(int j=0;j<4;++j){const float* b=sp+(nh*32+j*8+g)*PDSA_LD+k;
      pdsa_mma(acc[j],a0,a1,a2,a3,pdsa_tf32(b[t]),pdsa_tf32(b[t+4]));}
  }
  // rows g and g+8 of this m-tile -> query (mt>>1), heads (mt&1)*16 + g / +8
  const int qq=q0+(mt>>1);const int h0=(mt&1)*16+g,h1=h0+8;
  const float m0=qq<n?mixing[qq*32+h0]:0.f,m1=qq<n?mixing[qq*32+h1]:0.f;const float sc=0.08838834764831845f;
  float part[4][2];
#pragma unroll
  for(int j=0;j<4;++j){
    part[j][0]=fmaxf(acc[j][0]*sc,0.f)*m0+fmaxf(acc[j][2]*sc,0.f)*m1;
    part[j][1]=fmaxf(acc[j][1]*sc,0.f)*m0+fmaxf(acc[j][3]*sc,0.f)*m1;
#pragma unroll
    for(int o=4;o<32;o<<=1){part[j][0]+=__shfl_xor_sync(0xffffffff,part[j][0],o);part[j][1]+=__shfl_xor_sync(0xffffffff,part[j][1],o);}
  }
  if(g==0){
#pragma unroll
    for(int j=0;j<4;++j){red[warp*32+j*8+2*t]=part[j][0];red[warp*32+j*8+2*t+1]=part[j][1];}}
  __syncthreads();
  // warps (mt=0,1) -> query 0, (mt=2,3) -> query 1; nh selects the pool half
  if(tid<128){const int qi=tid>>6,col=tid&63,half=col>>5,c=col&31;
    const int w0=half*4+qi*2,w1=w0+1;const int qx=q0+qi,px=p0+col;
    if(qx<n&&px<active)out[(long long)qx*active+px]=red[w0*32+c]+red[w1*32+c];}
}
extern "C" int glm53_dsa_prefill_scores_cuda(const float* q,const float* mixing,const float* pools,float* out,int n,int active,cudaStream_t s){
  if(n<1||active<1)return int(cudaErrorInvalidValue);
  const int smem=(2*64*PDSA_LD+8*32)*4;
  static bool set=false;if(!set){cudaFuncSetAttribute(dsa_prefill_scores,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);set=true;}
  dsa_prefill_scores<<<dim3((active+63)/64,(n+1)/2),256,smem,s>>>(q,mixing,pools,out,n,active);return int(cudaGetLastError());
}

// ---- P5b: tiled TF32 GEMM form of dsa_prefill_scores (same math, better reuse) ----
// Block tile = 4 queries (128 rows = query x 32 heads) x 128 pools, K = 128 streamed in chunks of 32 with
// cp.async double buffering. Warp w: query (w & 3), pools half (w >> 2) -> 32 rows x 64 pools; the head
// reduction stays inside the warp (lanes over g, then the two m16 tiles).
constexpr int PD2_KC=16,PD2_LD=PD2_KC+4;
__device__ __forceinline__ void pd2_cp16(void* dst,const void* src,bool valid){
  const unsigned d=(unsigned)__cvta_generic_to_shared(dst);const int sz=valid?16:0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(d),"l"(src),"r"(sz));}
__device__ __forceinline__ uint32_t pd2_h2(float lo,float hi){half2 h=__floats2half2_rn(lo,hi);return *reinterpret_cast<uint32_t*>(&h);}
__device__ __forceinline__ void pd2_mma16(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
template<bool F16>
__global__ void __launch_bounds__(256,2) dsa_prefill_scores2(const float* __restrict__ q,const float* __restrict__ mixing,
    const float* __restrict__ pools,float* __restrict__ out,int n,int active,long long first_pos){
  extern __shared__ __align__(16) float pd2_smem[];
  float* sa=pd2_smem;                          // [2][128][36]
  float* sb=sa+2*128*PD2_LD;                   // [2][128][36]
  const int p0=blockIdx.x*128,tid=threadIdx.x,warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
  // Pools tile is fixed per block (read once from DRAM, then L2); 32 queries walk it in chunks of 4.
  for(int qc=0;qc<8;++qc){
  const int q0=blockIdx.y*32+qc*4;if(q0>=n)break;
  const int qi=warp&3,nh=warp>>2;
  auto load=[&](int kc,int buf){
    // 128 rows x PD2_KC floats (16B chunks) for A and for B
    for(int i=tid;i<128*PD2_KC/4;i+=256){const int r=i/(PD2_KC/4),c=(i%(PD2_KC/4))*4;
      const int qq=q0+(r>>5);pd2_cp16(sa+(buf*128+r)*PD2_LD+c,q+((long long)(qq<n?qq:0)*32+(r&31))*128+kc+c,qq<n);
      const int pp=p0+r;pd2_cp16(sb+(buf*128+r)*PD2_LD+c,pools+(long long)(pp<active?pp:0)*128+kc+c,pp<active);}
    asm volatile("cp.async.commit_group;\n"::);
  };
  float acc[2][8][4];
#pragma unroll
  for(int m=0;m<2;++m)
#pragma unroll
    for(int j=0;j<8;++j){acc[m][j][0]=acc[m][j][1]=acc[m][j][2]=acc[m][j][3]=0.f;}
  load(0,0);
  for(int kc=0;kc<128/PD2_KC;++kc){
    const int buf=kc&1;
    if(kc+1<128/PD2_KC){load((kc+1)*PD2_KC,buf^1);asm volatile("cp.async.wait_group 1;\n"::);}else asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    const float* A=sa+(buf*128+qi*32)*PD2_LD;const float* B=sb+(buf*128+nh*64)*PD2_LD;
    if constexpr(F16){
#pragma unroll
    for(int k=0;k<PD2_KC;k+=16){
      uint32_t a[2][4];
#pragma unroll
      for(int m=0;m<2;++m){const float* am=A+(m*16)*PD2_LD+k+2*t;
        a[m][0]=pd2_h2(am[g*PD2_LD],am[g*PD2_LD+1]);a[m][1]=pd2_h2(am[(g+8)*PD2_LD],am[(g+8)*PD2_LD+1]);
        a[m][2]=pd2_h2(am[g*PD2_LD+8],am[g*PD2_LD+9]);a[m][3]=pd2_h2(am[(g+8)*PD2_LD+8],am[(g+8)*PD2_LD+9]);}
#pragma unroll
      for(int j=0;j<8;++j){const float* bj=B+(j*8+g)*PD2_LD+k+2*t;const uint32_t b0=pd2_h2(bj[0],bj[1]),b1=pd2_h2(bj[8],bj[9]);
#pragma unroll
        for(int m=0;m<2;++m)pd2_mma16(acc[m][j],a[m][0],a[m][1],a[m][2],a[m][3],b0,b1);}
    }
    }else{
#pragma unroll
    for(int k=0;k<PD2_KC;k+=8){
      uint32_t a[2][4];
#pragma unroll
      for(int m=0;m<2;++m){const float* am=A+(m*16)*PD2_LD+k;
        a[m][0]=pdsa_tf32(am[g*PD2_LD+t]);a[m][1]=pdsa_tf32(am[(g+8)*PD2_LD+t]);a[m][2]=pdsa_tf32(am[g*PD2_LD+t+4]);a[m][3]=pdsa_tf32(am[(g+8)*PD2_LD+t+4]);}
#pragma unroll
      for(int j=0;j<8;++j){const float* bj=B+(j*8+g)*PD2_LD+k;const uint32_t b0=pdsa_tf32(bj[t]),b1=pdsa_tf32(bj[t+4]);
#pragma unroll
        for(int m=0;m<2;++m)pdsa_mma(acc[m][j],a[m][0],a[m][1],a[m][2],a[m][3],b0,b1);}
    }
    }
    __syncthreads();
  }
  const int qq=q0+qi;
  if(qq<n){
  const float sc=0.08838834764831845f;
  float mx[2][2];
#pragma unroll
  for(int m=0;m<2;++m){mx[m][0]=mixing[qq*32+m*16+g];mx[m][1]=mixing[qq*32+m*16+g+8];}
#pragma unroll
  for(int j=0;j<8;++j){
    float s0=0.f,s1=0.f;
#pragma unroll
    for(int m=0;m<2;++m){
      s0+=fmaxf(acc[m][j][0]*sc,0.f)*mx[m][0]+fmaxf(acc[m][j][2]*sc,0.f)*mx[m][1];
      s1+=fmaxf(acc[m][j][1]*sc,0.f)*mx[m][0]+fmaxf(acc[m][j][3]*sc,0.f)*mx[m][1];}
#pragma unroll
    for(int o=4;o<32;o<<=1){s0+=__shfl_xor_sync(0xffffffff,s0,o);s1+=__shfl_xor_sync(0xffffffff,s1,o);}
    if(g==0){const int p=p0+nh*64+j*8+2*t;
      // first_pos >= 0: pools at or beyond this query's completed count are masked (fused visibility).
      const long long complete=first_pos>=0?(first_pos+qq+1)/4:(long long)active;
      if(p<active)out[(long long)qq*active+p]=p<complete?s0:-3.4028234663852886e38F;
      if(p+1<active)out[(long long)qq*active+p+1]=p+1<complete?s1:-3.4028234663852886e38F;}
  }
  }
  __syncthreads();
  }
}
extern "C" int glm53_dsa_prefill_scores2_cuda(const float* q,const float* mixing,const float* pools,float* out,int n,int active,long long first_pos,cudaStream_t s){
  if(n<1||active<1)return int(cudaErrorInvalidValue);
  const int smem=4*128*PD2_LD*4;
  const char* f=std::getenv("GLM53_DSA_PREFILL_SCORE_F16");const bool f16=f&&f[0]=='1';
  static bool set=false;if(!set){cudaFuncSetAttribute(dsa_prefill_scores2<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
    cudaFuncSetAttribute(dsa_prefill_scores2<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);set=true;}
  if(f16)dsa_prefill_scores2<true><<<dim3((active+127)/128,(n+31)/32),256,smem,s>>>(q,mixing,pools,out,n,active,first_pos);
  else dsa_prefill_scores2<false><<<dim3((active+127)/128,(n+31)/32),256,smem,s>>>(q,mixing,pools,out,n,active,first_pos);
  return int(cudaGetLastError());
}

// D8: chain-shared MLA commit for all latent layers in one launch (GLM53_MLA_COMMIT_FUSED=1). Per layer:
// pools[max((node_len-1)/4,0)] <- node pool_row; tails <- node tails; len <- node len. Same data moves as
// patch_pool_row + three copy_ calls (pure copies, bitwise identical), without ~7 host launches per layer.
struct MlaCommitEntry {const long long* node_len;float* dst_pools;const float* pool_row;int row_floats;
  const float* src_tk;float* dst_tk;int tk_floats;const float* src_tg;float* dst_tg;int tg_floats;long long* dst_len;int has_row;};
struct MlaCommitBatch {MlaCommitEntry e[16];int n;};
__global__ void mla_commit_many(MlaCommitBatch b){
  const MlaCommitEntry& e=b.e[blockIdx.x];
  const long long len=*e.node_len;
  if(e.has_row){long long row=(len-1)/4; if(len-1<0) row=0; if(row<0)row=0;
    float* dst=e.dst_pools+row*(long long)e.row_floats;
    for(int i=threadIdx.x;i<e.row_floats;i+=blockDim.x)dst[i]=e.pool_row[i];}
  for(int i=threadIdx.x;i<e.tk_floats;i+=blockDim.x)e.dst_tk[i]=e.src_tk[i];
  for(int i=threadIdx.x;i<e.tg_floats;i+=blockDim.x)e.dst_tg[i]=e.src_tg[i];
  if(threadIdx.x==0)*e.dst_len=len;
}
extern "C" int glm53_mla_commit_many_cuda(const void* entries,int n,cudaStream_t stream){
  if(n<1||n>16)return int(cudaErrorInvalidValue);
  MlaCommitBatch b;memcpy(b.e,entries,sizeof(MlaCommitEntry)*n);b.n=n;
  mla_commit_many<<<n,128,0,stream>>>(b);
  return int(cudaGetLastError());
}

// L1-a (GLM53_DSA_INDEX_BF16=1): decode-sized DSA indexer projections, Y[m,n] = sum_k X[m,k]*W[n,k] with
// FP32 activations and BF16-resident weights (widened exactly), FP32 FMA in a fixed order (lane-strided
// partials, then a xor-shuffle tree). Replaces TF32 cuBLAS (more precise, different order: L1).
// One warp per output column n; 8 warps per block; M <= 16 rows kept in registers.
template<int MR>
__global__ void __launch_bounds__(256) bf16w_rows(const float* __restrict__ x,int ldx,const __nv_bfloat16* __restrict__ w,float* __restrict__ y,int m,int n,int k){
  const int warp=threadIdx.x>>5,lane=threadIdx.x&31,col=blockIdx.x*8+warp;
  if(col>=n)return;
  const __nv_bfloat16* wr=w+(long long)col*k;
  float acc[MR];
#pragma unroll
  for(int r=0;r<MR;++r)acc[r]=0.f;
  for(int c=lane*8;c<k;c+=256){
    const uint4 u=*reinterpret_cast<const uint4*>(wr+c);
    const float wv[8]={__uint_as_float(u.x<<16),__uint_as_float(u.x&0xffff0000u),__uint_as_float(u.y<<16),__uint_as_float(u.y&0xffff0000u),
                       __uint_as_float(u.z<<16),__uint_as_float(u.z&0xffff0000u),__uint_as_float(u.w<<16),__uint_as_float(u.w&0xffff0000u)};
#pragma unroll
    for(int r=0;r<MR;++r){if(r<m){
      const float4 a=*reinterpret_cast<const float4*>(x+(long long)r*ldx+c),b=*reinterpret_cast<const float4*>(x+(long long)r*ldx+c+4);
      float s=acc[r];s=__fmaf_rn(a.x,wv[0],s);s=__fmaf_rn(a.y,wv[1],s);s=__fmaf_rn(a.z,wv[2],s);s=__fmaf_rn(a.w,wv[3],s);
      s=__fmaf_rn(b.x,wv[4],s);s=__fmaf_rn(b.y,wv[5],s);s=__fmaf_rn(b.z,wv[6],s);s=__fmaf_rn(b.w,wv[7],s);acc[r]=s;}}
  }
#pragma unroll
  for(int r=0;r<MR;++r){if(r<m){float s=acc[r];for(int o=16;o;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o);if(lane==0)y[(long long)r*n+col]=s;}}
}
extern "C" int glm53_bf16w_rows_cuda(const float* x,int ldx,const void* w,float* y,int m,int n,int k,cudaStream_t stream){
  if(m<1||m>16||k%256||ldx%4)return int(cudaErrorInvalidValue);
  const int grid=(n+7)/8;const __nv_bfloat16* wb=(const __nv_bfloat16*)w;
  if(m<=4)bf16w_rows<4><<<grid,256,0,stream>>>(x,ldx,wb,y,m,n,k);
  else if(m<=8)bf16w_rows<8><<<grid,256,0,stream>>>(x,ldx,wb,y,m,n,k);
  else bf16w_rows<16><<<grid,256,0,stream>>>(x,ldx,wb,y,m,n,k);
  return int(cudaGetLastError());
}

// GLM53_DSA_KEY_FUSED=1 (L1): DSA indexer key LayerNorm + gate copy in one launch. Per row y = [raw keys (dim) | gate
// (gcols)] of the C12 group output (row stride ldy). Elementwise ops in ATen's order with FP32 rounding each
// ((raw-mean), *rsqrt, *norm_w, +norm_b); only the two mean reductions' summation order differs from ATen's mean_dim.
__global__ void dsa_key_ln(const float* __restrict__ y,int ldy,int dim,int gcols,const float* __restrict__ w,const float* __restrict__ b,
                           float* __restrict__ kout,float* __restrict__ gout,int rows){
    const int r=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;if(r>=rows)return;
    const float* yr=y+(long long)r*ldy;
    float s=0.f;for(int c=lane;c<dim;c+=32)s=__fadd_rn(s,yr[c]);
    for(int o=16;o;o>>=1)s=__fadd_rn(s,__shfl_xor_sync(0xffffffffu,s,o));
    const float mean=__fdiv_rn(s,(float)dim);
    float v=0.f;for(int c=lane;c<dim;c+=32){const float d=__fsub_rn(yr[c],mean);v=__fadd_rn(v,__fmul_rn(d,d));}
    for(int o=16;o;o>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffffu,v,o));
    const float rs=rsqrtf(__fadd_rn(__fdiv_rn(v,(float)dim),1e-6f));
    for(int c=lane;c<dim;c+=32){const float d=__fsub_rn(yr[c],mean);kout[(long long)r*dim+c]=__fadd_rn(__fmul_rn(__fmul_rn(d,rs),w[c]),b[c]);}
    for(int c=lane;c<gcols;c+=32)gout[(long long)r*gcols+c]=yr[dim+c];
}
extern "C" int glm53_dsa_key_ln_cuda(const float* y,int ldy,int dim,int gcols,const float* w,const float* b,float* k,float* g,int rows,cudaStream_t s){
    if(rows<1||dim<1||gcols<0)return int(cudaErrorInvalidValue);
    dsa_key_ln<<<(rows+7)/8,256,0,s>>>(y,ldy,dim,gcols,w,b,k,g,rows);return int(cudaGetLastError());
}
