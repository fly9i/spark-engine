#include <type_traits>
// SPDX-License-Identifier: MIT
// N-way lean FP32 allreduce over RoCE RC queue pairs (GLM53_RDMA_ARN_INIT=1; TP3, docs/06-TP3.md §5.2).
//
// One-shot full mesh: every rank writes its whole buffer into each peer's recv area (indexed by the sender's rank),
// then sums the world contributions in fixed rank order rank0, rank1, ... with explicit round-to-nearest adds, so all
// ranks get bitwise the same result and world=2 equals the TP2 path (x0+x1). The TP2 file rdma_ar.cu is untouched.
//
// Per call (all ranks, same stream order):
//   GPU kernel: copy local data -> pinned send slot (host-mapped), last block publishes {bytes, seq} in ready; every
//   block waits for the flags of all peers for this seq, then sums in rank order (or returns: send-only form).
//   One CPU proxy thread per peer watches ready and posts RDMA WRITEs of the payload, then an 8-byte flag, on the same
//   QP (same-QP writes land in order). Messages of at least GLM53_RDMA_ARN_SPLIT_KB (default 32) KiB are split over the
//   peer's two links (the two PCIe functions of one CX7 port), each part followed by its own flag.
// Slot reuse is safe per pair exactly as in rdma_ar.cu: this rank rewrites send slot s only in call s+NSLOT, which runs
// after it received every peer's s+NSLOT-1, which each peer sent only after it had received this rank's s+NSLOT-2.
//
// Links are found by subnet: each rank publishes its RoCE devices with their netdev IPv4/prefix through the process
// group's TCPStore; a peer link is a (local device, peer device) pair on the same subnet (one /30 per link, 06-TP3 §2.2),
// ordered by the lower rank's device name so both sides agree. The GID used is the IPv4-mapped RoCE v2 entry of that
// address (GLM53_RDMA_ARN_GID overrides the index).
#include <pthread.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "host_pin.cuh"
#include "rdma_gid.h"
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <dirent.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {
constexpr int MAXW=4;            // ranks
constexpr int MAXL=2;            // links per peer
constexpr int NSLOT=32;
constexpr size_t SLOT_BYTES=1024*1024;
struct Host {                    // pinned, host-mapped, registered with every device in use
  alignas(64) volatile unsigned long long ready_seq[NSLOT];
  alignas(64) volatile unsigned long long ready_bytes[NSLOT];
  alignas(64) volatile unsigned long long recv_flag[MAXW][MAXL][NSLOT];   // [sender][link][slot], written by the sender's NIC
  alignas(64) unsigned long long flag_src[MAXW][MAXL][NSLOT];             // [peer][link][slot], local source of flag writes
  alignas(4096) unsigned char send[NSLOT][SLOT_BYTES];
  alignas(4096) unsigned char recv[MAXW][NSLOT][SLOT_BYTES];               // [sender][slot]
};
// seq: last call sent by this rank; fseq: last call summed (equal except inside a pipelined large allreduce)
// ready_bytes of a slot: payload bytes, bits 62..63 the mode: 0 the whole buffer to every peer, 1 reduce-scatter (peer
// j gets part j), 2 all-gather (every peer gets this rank's part). Parts: the buffer of n floats in `world` pieces at
// float offsets floor(n k / world) rounded down to a multiple of 4 (16-byte aligned), the last ending at n.
// bit 61: the payload is BF16 (parts are counted in 2-byte elements)
constexpr unsigned long long MODE_SHIFT=62,MODE_RS=1ull<<MODE_SHIFT,MODE_AG=2ull<<MODE_SHIFT,MODE_BF16=1ull<<61,BYTES_MASK=(1ull<<61)-1;
__host__ __device__ inline long long part_lo(long long n,int k,int world){return k>=world?n:((n*k/world)/4)*4;}
struct DevState {unsigned long long seq;unsigned int arrive;unsigned int done;unsigned long long fseq;unsigned int fdone;};

struct DevInfo {char name[32];uint32_t ip;uint32_t mask;};                   // ip/mask in network order
struct LinkInfo {uint32_t qpn,psn,rkey;uint8_t gid[16];};
struct PeerInfo {uint64_t addr;uint32_t nlink;LinkInfo link[MAXL];};        // what rank r tells rank j about the j<->r QPs

struct Dev {std::string name;ibv_context* ctx=nullptr;ibv_pd* pd=nullptr;ibv_mr* mr=nullptr;int gid_index=-1;uint32_t ip=0;};
struct Link {Dev* dev=nullptr;ibv_cq* cq=nullptr;ibv_qp* qp=nullptr;LinkInfo remote{};};
struct Peer {int rank=-1;int nlink=0;Link link[MAXL];uint64_t addr=0;};

Host* g_host=nullptr;Host* g_host_dev=nullptr;DevState* g_state=nullptr;
int g_rank=-1,g_world=0;unsigned long long g_split=0;
std::vector<Dev*> g_devs;Peer g_peers[MAXW];std::atomic<bool> g_stop{false};

using KvSet=int(*)(const char*,const void*,size_t);
using KvGet=int(*)(const char*,void*,size_t);

template<bool SUM>
__global__ void arn_kernel(float* data,long long n,int rank,int world,Host* h,DevState* st,unsigned long long split){
  const unsigned long long s=st->seq+1;const int slot=(int)(s%NSLOT);
  float* snd=reinterpret_cast<float*>(h->send[slot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  // 16-byte accesses (the send / recv slots are host memory: every access is a slow round trip over the C2C link)
  const long long n4=((size_t)data&15)?0:n/4;
  for(long long i=first;i<n4;i+=stride)reinterpret_cast<float4*>(snd)[i]=reinterpret_cast<const float4*>(data)[i];
  for(long long i=n4*4+first;i<n;i+=stride)snd[i]=data[i];
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[slot]=(unsigned long long)n*4;__threadfence_system();h->ready_seq[slot]=s;__threadfence_system();}
    const bool two=split&&(unsigned long long)n*4>=split;
    for(int r=0;r<world;++r){
      if(r==rank)continue;
      long long spins=0;
      while(h->recv_flag[r][0][slot]!=s||(two&&h->recv_flag[r][1][slot]!=s)){if(++spins>(1ll<<34))__trap();}
    }
  }
  __syncthreads();
  if constexpr(SUM){
    // every rank's part loaded first (independent loads in flight together), then summed in rank order
    for(long long i=first;i<n4;i+=stride){
      float4 x[MAXW];
#pragma unroll
      for(int r=0;r<MAXW;++r)if(r<world)x[r]=r==rank?reinterpret_cast<const float4*>(data)[i]:__ldcv(reinterpret_cast<const float4*>(h->recv[r][slot])+i);
      float4 acc=x[0];
#pragma unroll
      for(int r=1;r<MAXW;++r)if(r<world){acc.x=__fadd_rn(acc.x,x[r].x);acc.y=__fadd_rn(acc.y,x[r].y);acc.z=__fadd_rn(acc.z,x[r].z);acc.w=__fadd_rn(acc.w,x[r].w);}
      reinterpret_cast<float4*>(data)[i]=acc;
    }
    for(long long i=n4*4+first;i<n;i+=stride){
      float acc=0.f;
      for(int r=0;r<world;++r){
        const float x=r==rank?data[i]:__ldcv(reinterpret_cast<const float*>(h->recv[r][slot])+i);
        acc=r==0?x:__fadd_rn(acc,x);
      }
      data[i]=acc;
    }
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0&&atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;st->fseq=s;}
}

// Pipelined large allreduce (glm53_rdma_arn_big_cuda): the send half (copy into the next slot, publish it) and the
// finish half (wait for every peer's part of the oldest unfinished call, sum in rank order) as separate kernels, so
// several slot-sized calls are on the network at once. Same per-element order as arn_kernel.
__global__ void arn_send_kernel(const float* data,long long n,Host* h,DevState* st){
  const unsigned long long s=st->seq+1;const int slot=(int)(s%NSLOT);
  float* snd=reinterpret_cast<float*>(h->send[slot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  // float4 copies (data and slots are 16-byte aligned: every chunk starts at a multiple of the slot size), tail by element
  const long long n4=n/4;
  for(long long i=first;i<n4;i+=stride)reinterpret_cast<float4*>(snd)[i]=reinterpret_cast<const float4*>(data)[i];
  for(long long i=n4*4+first;i<n;i+=stride)snd[i]=data[i];
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[slot]=(unsigned long long)n*4;__threadfence_system();h->ready_seq[slot]=s;__threadfence_system();}
    if(atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;}
  }
}
__global__ void arn_finish_kernel(float* data,long long n,int rank,int world,const Host* h,DevState* st,unsigned long long split){
  const unsigned long long s=st->fseq+1;const int slot=(int)(s%NSLOT);
  if(threadIdx.x==0){
    const bool two=split&&(unsigned long long)n*4>=split;
    for(int r=0;r<world;++r){
      if(r==rank)continue;
      long long spins=0;
      while(h->recv_flag[r][0][slot]!=s||(two&&h->recv_flag[r][1][slot]!=s)){if(++spins>(1ll<<34))__trap();}
    }
  }
  __syncthreads();
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  const long long n4=n/4;
  for(long long i=first;i<n4;i+=stride){
    float4 acc=make_float4(0.f,0.f,0.f,0.f);
    for(int r=0;r<world;++r){
      const float4 x=r==rank?reinterpret_cast<const float4*>(data)[i]:__ldcv(reinterpret_cast<const float4*>(h->recv[r][slot])+i);
      if(r==0)acc=x;else{acc.x=__fadd_rn(acc.x,x.x);acc.y=__fadd_rn(acc.y,x.y);acc.z=__fadd_rn(acc.z,x.z);acc.w=__fadd_rn(acc.w,x.w);}
    }
    reinterpret_cast<float4*>(data)[i]=acc;
  }
  for(long long i=n4*4+first;i<n;i+=stride){
    float acc=0.f;
    for(int r=0;r<world;++r){
      const float x=r==rank?data[i]:__ldcv(reinterpret_cast<const float*>(h->recv[r][slot])+i);
      acc=r==0?x:__fadd_rn(acc,x);
    }
    data[i]=acc;
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0&&atomicAdd(&st->fdone,1u)==gridDim.x-1){st->fdone=0;st->fseq=s;}
}

// Reduce-scatter + all-gather large allreduce (glm53_rdma_arn_big_cuda): per slot-sized chunk
//   rs_send: copy the chunk into the next slot, publish it as reduce-scatter (peer j receives part j);
//   rs_finish: wait for every peer's copy of this rank's part, sum it in rank order (the same per-element order as the
//     one-shot path, so the result is bitwise the same), write it to the data and to the next slot, publish that as
//     all-gather (every peer receives this rank's summed part);
//   ag_finish: wait for every peer's summed part and copy it into the data.
// Each rank sends 4/3 of the buffer instead of twice the buffer.
__device__ inline void wait_peers(const Host* h,int rank,int world,int slot,unsigned long long s,unsigned long long split,long long n,bool own){
  for(int r=0;r<world;++r){
    if(r==rank)continue;
    const int k=own?rank:r;   // the part the sender r wrote: ours (reduce-scatter) or its own (all-gather)
    const bool two=split&&(unsigned long long)(part_lo(n,k+1,world)-part_lo(n,k,world))*4>=split;
    long long spins=0;
    while(h->recv_flag[r][0][slot]!=s||(two&&h->recv_flag[r][1][slot]!=s)){if(++spins>(1ll<<34))__trap();}
  }
}
__global__ void arn_rs_send_kernel(const float* data,long long n,Host* h,DevState* st){
  const unsigned long long s=st->seq+1;const int slot=(int)(s%NSLOT);
  float* snd=reinterpret_cast<float*>(h->send[slot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  const long long n4=n/4;
  for(long long i=first;i<n4;i+=stride)reinterpret_cast<float4*>(snd)[i]=reinterpret_cast<const float4*>(data)[i];
  for(long long i=n4*4+first;i<n;i+=stride)snd[i]=data[i];
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[slot]=((unsigned long long)n*4)|MODE_RS;__threadfence_system();h->ready_seq[slot]=s;__threadfence_system();}
    if(atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;}
  }
}
__global__ void arn_rs_finish_kernel(float* data,long long n,int rank,int world,Host* h,DevState* st,unsigned long long split){
  const unsigned long long sf=st->fseq+1,sp=st->seq+1;const int slot=(int)(sf%NSLOT),pslot=(int)(sp%NSLOT);
  if(threadIdx.x==0)wait_peers(h,rank,world,slot,sf,split,n,true);
  __syncthreads();
  const long long lo=part_lo(n,rank,world),hi=part_lo(n,rank+1,world);
  float* snd=reinterpret_cast<float*>(h->send[pslot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  for(long long i=lo/4+first;i<hi/4;i+=stride){
    float4 acc=make_float4(0.f,0.f,0.f,0.f);
    for(int r=0;r<world;++r){
      const float4 x=r==rank?reinterpret_cast<const float4*>(data)[i]:__ldcv(reinterpret_cast<const float4*>(h->recv[r][slot])+i);
      if(r==0)acc=x;else{acc.x=__fadd_rn(acc.x,x.x);acc.y=__fadd_rn(acc.y,x.y);acc.z=__fadd_rn(acc.z,x.z);acc.w=__fadd_rn(acc.w,x.w);}
    }
    reinterpret_cast<float4*>(data)[i]=acc;reinterpret_cast<float4*>(snd)[i]=acc;
  }
  for(long long i=(hi/4)*4+first;i<hi;i+=stride){   // the last part's tail (n not a multiple of 4)
    float acc=0.f;
    for(int r=0;r<world;++r){const float x=r==rank?data[i]:__ldcv(reinterpret_cast<const float*>(h->recv[r][slot])+i);acc=r==0?x:__fadd_rn(acc,x);}
    data[i]=acc;snd[i]=acc;
  }
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[pslot]=((unsigned long long)n*4)|MODE_AG;__threadfence_system();h->ready_seq[pslot]=sp;__threadfence_system();}
    if(atomicAdd(&st->fdone,1u)==gridDim.x-1){st->fdone=0;st->fseq=sf;st->seq=sp;}
  }
}
__global__ void arn_ag_finish_kernel(float* data,long long n,int rank,int world,const Host* h,DevState* st,unsigned long long split){
  const unsigned long long s=st->fseq+1;const int slot=(int)(s%NSLOT);
  if(threadIdx.x==0)wait_peers(h,rank,world,slot,s,split,n,false);
  __syncthreads();
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  for(int r=0;r<world;++r){
    if(r==rank)continue;
    const long long lo=part_lo(n,r,world),hi=part_lo(n,r+1,world);
    for(long long i=lo/4+first;i<hi/4;i+=stride)reinterpret_cast<float4*>(data)[i]=__ldcv(reinterpret_cast<const float4*>(h->recv[r][slot])+i);
    for(long long i=(hi/4)*4+first;i<hi;i+=stride)data[i]=__ldcv(reinterpret_cast<const float*>(h->recv[r][slot])+i);
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0&&atomicAdd(&st->fdone,1u)==gridDim.x-1){st->fdone=0;st->fseq=s;}
}

// BF16 transport (glm53_rdma_arn_big_bf16_cuda): the same reduce-scatter + all-gather with every rank's partial rounded
// to BF16 on the wire; the owner sums the BF16 values of all ranks (its own included) in rank order in FP32 and rounds
// the sum to BF16, which every rank then holds (as FP32 in the data). Half the bytes of the FP32 transport.
__global__ void arn_rs_send_bf16_kernel(const float* data,long long n,Host* h,DevState* st){
  const unsigned long long s=st->seq+1;const int slot=(int)(s%NSLOT);
  __nv_bfloat16* snd=reinterpret_cast<__nv_bfloat16*>(h->send[slot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  for(long long i=first;i<n/2;i+=stride){const float2 v=reinterpret_cast<const float2*>(data)[i];reinterpret_cast<__nv_bfloat162*>(snd)[i]=__floats2bfloat162_rn(v.x,v.y);}
  if(first==0&&(n&1))snd[n-1]=__float2bfloat16(data[n-1]);
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[slot]=((unsigned long long)n*2)|MODE_RS|MODE_BF16;__threadfence_system();h->ready_seq[slot]=s;__threadfence_system();}
    if(atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;}
  }
}
__device__ inline void wait_peers_e(const Host* h,int rank,int world,int slot,unsigned long long s,unsigned long long split,long long n,bool own,int esz){
  for(int r=0;r<world;++r){
    if(r==rank)continue;
    const int k=own?rank:r;
    const bool two=split&&(unsigned long long)(part_lo(n,k+1,world)-part_lo(n,k,world))*esz>=split;
    long long spins=0;
    while(h->recv_flag[r][0][slot]!=s||(two&&h->recv_flag[r][1][slot]!=s)){if(++spins>(1ll<<34))__trap();}
  }
}
__global__ void arn_rs_finish_bf16_kernel(float* data,long long n,int rank,int world,Host* h,DevState* st,unsigned long long split){
  const unsigned long long sf=st->fseq+1,sp=st->seq+1;const int slot=(int)(sf%NSLOT),pslot=(int)(sp%NSLOT);
  if(threadIdx.x==0)wait_peers_e(h,rank,world,slot,sf,split,n,true,2);
  __syncthreads();
  const long long lo=part_lo(n,rank,world),hi=part_lo(n,rank+1,world);
  const __nv_bfloat16* own=reinterpret_cast<const __nv_bfloat16*>(h->send[slot]);   // this rank's rounded partial
  __nv_bfloat16* snd=reinterpret_cast<__nv_bfloat16*>(h->send[pslot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  for(long long i=lo+first;i<hi;i+=stride){
    float acc=0.f;
    for(int r=0;r<world;++r){
      const float x=__bfloat162float(r==rank?own[i]:reinterpret_cast<const volatile __nv_bfloat16*>(h->recv[r][slot])[i]);
      acc=r==0?x:__fadd_rn(acc,x);
    }
    const __nv_bfloat16 b=__float2bfloat16(acc);
    data[i]=__bfloat162float(b);snd[i]=b;
  }
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[pslot]=((unsigned long long)n*2)|MODE_AG|MODE_BF16;__threadfence_system();h->ready_seq[pslot]=sp;__threadfence_system();}
    if(atomicAdd(&st->fdone,1u)==gridDim.x-1){st->fdone=0;st->fseq=sf;st->seq=sp;}
  }
}
__global__ void arn_ag_finish_bf16_kernel(float* data,long long n,int rank,int world,const Host* h,DevState* st,unsigned long long split){
  const unsigned long long s=st->fseq+1;const int slot=(int)(s%NSLOT);
  if(threadIdx.x==0)wait_peers_e(h,rank,world,slot,s,split,n,false,2);
  __syncthreads();
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  for(int r=0;r<world;++r){
    if(r==rank)continue;
    const volatile __nv_bfloat16* src=reinterpret_cast<const volatile __nv_bfloat16*>(h->recv[r][slot]);
    for(long long i=part_lo(n,r,world)+first;i<part_lo(n,r+1,world);i+=stride)data[i]=__bfloat162float(const_cast<const __nv_bfloat16&>(src[i]));
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0&&atomicAdd(&st->fdone,1u)==gridDim.x-1){st->fdone=0;st->fseq=s;}
}

// Row-block collectives (TP3 sequence-parallel prefill: glm53_rdma_arn_rows_cuda). The buffer is world row blocks of
// blk bytes each (block r = rank r's rows); a call moves chunk [off, off + m) of every block through one slot laid out as
// world regions of m bytes, region j = block j's chunk, so the proxy's existing modes route it: reduce-scatter sends region
// j to peer j, all-gather sends this rank's region to every peer (m: a multiple of 16 bytes, i.e. of 4 elements, so the
// proxy's parts are exactly the regions).
__device__ inline void copy16(unsigned char* d,const unsigned char* s,long long bytes,long long first,long long stride){
  for(long long i=first;i<bytes/16;i+=stride)reinterpret_cast<uint4*>(d)[i]=reinterpret_cast<const uint4*>(s)[i];
}
__device__ inline void copy16_cv(unsigned char* d,const unsigned char* s,long long bytes,long long first,long long stride){
  for(long long i=first;i<bytes/16;i+=stride)reinterpret_cast<uint4*>(d)[i]=__ldcv(reinterpret_cast<const uint4*>(s)+i);
}
// ag: `data` is this rank's block; else `data` is the whole buffer (world blocks)
__global__ void arn_rows_send_kernel(const unsigned char* data,long long blk,long long off,long long m,int rank,int world,int ag,
                                     unsigned long long flags,Host* h,DevState* st){
  const unsigned long long s=st->seq+1;const int slot=(int)(s%NSLOT);
  unsigned char* snd=h->send[slot];
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(ag)copy16(snd+(long long)rank*m,data+off,m,first,stride);
  else for(int j=0;j<world;++j)copy16(snd+(long long)j*m,data+(long long)j*blk+off,m,first,stride);
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[slot]=((unsigned long long)world*m)|flags;__threadfence_system();h->ready_seq[slot]=s;__threadfence_system();}
    if(atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;}
  }
}
// BF16 wire (DS41's transport, GLM53_RDMA_ARN_RS_BF16): reduce-scatter regions of m bytes carry m / 2 bytes of BF16 rounded
// FP32 elements (region j = block j's chunk of m FP32 bytes from `data`)
__global__ void arn_rows_send_bf16_kernel(const float* data,long long blk_f,long long off_f,long long m_f,int world,
                                          unsigned long long flags,Host* h,DevState* st){
  const unsigned long long s=st->seq+1;const int slot=(int)(s%NSLOT);
  __nv_bfloat16* snd=reinterpret_cast<__nv_bfloat16*>(h->send[slot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  for(int j=0;j<world;++j){const float* src=data+(long long)j*blk_f+off_f;__nv_bfloat16* d=snd+(long long)j*m_f;
    for(long long i=first;i<m_f/2;i+=stride){const float2 v=reinterpret_cast<const float2*>(src)[i];reinterpret_cast<__nv_bfloat162*>(d)[i]=__floats2bfloat162_rn(v.x,v.y);}}
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[slot]=((unsigned long long)world*m_f*2)|flags;__threadfence_system();h->ready_seq[slot]=s;__threadfence_system();}
    if(atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;}
  }
}
// out chunk (FP32) = sum in rank order of this rank's own FP32 partial and the peers' BF16 partials of block `rank`
__global__ void arn_rows_finish_bf16_kernel(float* out,const float* own,long long blk_f,long long off_f,long long m_f,int rank,int world,
                                            const Host* h,DevState* st,unsigned long long split){
  const unsigned long long s=st->fseq+1;const int slot=(int)(s%NSLOT);
  if(threadIdx.x==0){
    const bool two=split&&(unsigned long long)m_f*2>=split;
    for(int r=0;r<world;++r){
      if(r==rank)continue;
      long long spins=0;
      while(h->recv_flag[r][0][slot]!=s||(two&&h->recv_flag[r][1][slot]!=s)){if(++spins>(1ll<<34))__trap();}
    }
  }
  __syncthreads();
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  const float* mine=own+(long long)rank*blk_f+off_f;
  for(long long i=first;i<m_f/2;i+=stride){
    float2 acc=make_float2(0.f,0.f);
    for(int r=0;r<world;++r){
      float2 x;
      if(r==rank)x=reinterpret_cast<const float2*>(mine)[i];
      else{const __nv_bfloat162 b=__ldcv(reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const __nv_bfloat16*>(h->recv[r][slot])+(long long)rank*m_f)+i);x=__bfloat1622float2(b);}
      if(r==0)acc=x;else{acc.x=__fadd_rn(acc.x,x.x);acc.y=__fadd_rn(acc.y,x.y);}
    }
    reinterpret_cast<float2*>(out+off_f)[i]=acc;
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0&&atomicAdd(&st->fdone,1u)==gridDim.x-1){st->fdone=0;st->fseq=s;}
}
// The allreduce's BF16 sum as a row-block reduce-scatter (DS41 sequence-parallel prefill, glm53_rdma_arn_rows_bf16r_cuda):
// every rank's FP32 partial rounded to BF16 (its own too), summed in rank order in FP32 and rounded to BF16 -- per
// element exactly glm53_rdma_arn_big_bf16_cuda's value. `data` holds n_valid FP32 elements of world blocks of blk_f
// (the rows past them, the last block's padding, read as zeros: no padded copy of the input).
// T = Comb: the partial is a routed-expert call's combine computed here (y = add + sum_k out[pair (r, k)], each element
// with combine_kernel's adds in its order, ds41_mx.cu): no FP32 y written and read back, and a chunk's combine runs
// while the earlier chunks are on the wire
struct Comb { const float* out; const int* pairmap; const float* add; int topk; int hcols; };
__device__ __forceinline__ float comb1(const Comb& cb,long long g){
  const long long r=g/cb.hcols;const int c=(int)(g%cb.hcols);
  float acc=cb.add?cb.add[g]:0.f,s=0.f;
  for(int k=0;k<cb.topk;++k)s+=cb.out[(long long)cb.pairmap[r*cb.topk+k]*cb.hcols+c];
  return acc+s;
}
// T = __nv_bfloat16: the partial is already BF16 (its FP32 rounding is the value itself: the same bytes on the wire)
template<class T>
__global__ void arn_rows_send_bf16v_kernel(typename std::conditional<std::is_same<T,Comb>::value,Comb,const T*>::type data,long long blk_f,long long off_f,long long m_f,long long n_valid,int world,
                                           unsigned long long flags,Host* h,DevState* st){
  const unsigned long long s=st->seq+1;const int slot=(int)(s%NSLOT);
  __nv_bfloat16* snd=reinterpret_cast<__nv_bfloat16*>(h->send[slot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  for(int j=0;j<world;++j){const long long base=(long long)j*blk_f+off_f;__nv_bfloat16* d=snd+(long long)j*m_f;
    for(long long i=first;i<m_f/2;i+=stride){
      if constexpr(std::is_same<T,Comb>::value){const long long g=base+2*i;
        const float v0=g<n_valid?comb1(data,g):0.f,v1=g+1<n_valid?comb1(data,g+1):0.f;
        reinterpret_cast<__nv_bfloat162*>(d)[i]=__floats2bfloat162_rn(v0,v1);}
      else if constexpr(std::is_same<T,float>::value){const T* src=data+base;const float2 v=base+2*i+1<n_valid?reinterpret_cast<const float2*>(src)[i]:make_float2(base+2*i<n_valid?src[2*i]:0.f,0.f);
        reinterpret_cast<__nv_bfloat162*>(d)[i]=__floats2bfloat162_rn(v.x,v.y);}
      else{const T* src=data+base;const __nv_bfloat16 z=__float2bfloat16_rn(0.f);
        reinterpret_cast<__nv_bfloat162*>(d)[i]=base+2*i+1<n_valid?reinterpret_cast<const __nv_bfloat162*>(src)[i]:__halves2bfloat162(base+2*i<n_valid?src[2*i]:z,z);}}}
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[slot]=((unsigned long long)world*m_f*2)|flags;__threadfence_system();h->ready_seq[slot]=s;__threadfence_system();}
    if(atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;}
  }
}
// out (BF16, this rank's block) chunk = BF16(sum in rank order of the BF16 partials of block `rank`); this rank's own is
// read back from its send slot (the rounded values it sent)
__global__ void arn_rows_finish_bf16r_kernel(__nv_bfloat16* out,long long m_f,long long off_f,int rank,int world,
                                             const Host* h,DevState* st,unsigned long long split){
  const unsigned long long s=st->fseq+1;const int slot=(int)(s%NSLOT);
  if(threadIdx.x==0){
    const bool two=split&&(unsigned long long)m_f*2>=split;
    for(int r=0;r<world;++r){
      if(r==rank)continue;
      long long spins=0;
      while(h->recv_flag[r][0][slot]!=s||(two&&h->recv_flag[r][1][slot]!=s)){if(++spins>(1ll<<34))__trap();}
    }
  }
  __syncthreads();
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  const __nv_bfloat162* mine=reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const __nv_bfloat16*>(h->send[slot])+(long long)rank*m_f);
  for(long long i=first;i<m_f/2;i+=stride){
    float2 acc=make_float2(0.f,0.f);
    for(int r=0;r<world;++r){
      const __nv_bfloat162 b=r==rank?mine[i]:__ldcv(reinterpret_cast<const __nv_bfloat162*>(reinterpret_cast<const __nv_bfloat16*>(h->recv[r][slot])+(long long)rank*m_f)+i);
      const float2 x=__bfloat1622float2(b);
      if(r==0)acc=x;else{acc.x=__fadd_rn(acc.x,x.x);acc.y=__fadd_rn(acc.y,x.y);}
    }
    reinterpret_cast<__nv_bfloat162*>(out+off_f)[i]=__floats2bfloat162_rn(acc.x,acc.y);
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0&&atomicAdd(&st->fdone,1u)==gridDim.x-1){st->fdone=0;st->fseq=s;}
}
// ag: `out` is the whole buffer, `own` this rank's block (copied into its place); else (FP32) `out` is this rank's block,
// `own` the whole partial buffer: out chunk = sum over r in rank order of block `rank` of rank r's partial
__global__ void arn_rows_finish_kernel(unsigned char* out,const unsigned char* own,long long blk,long long off,long long m,int rank,int world,
                                       int ag,const Host* h,DevState* st,unsigned long long split){
  const unsigned long long s=st->fseq+1;const int slot=(int)(s%NSLOT);
  if(threadIdx.x==0){
    const bool two=split&&(unsigned long long)m>=split;
    for(int r=0;r<world;++r){
      if(r==rank)continue;
      long long spins=0;
      while(h->recv_flag[r][0][slot]!=s||(two&&h->recv_flag[r][1][slot]!=s)){if(++spins>(1ll<<34))__trap();}
    }
  }
  __syncthreads();
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(ag){
    for(int r=0;r<world;++r){
      if(r==rank)copy16(out+(long long)r*blk+off,own+off,m,first,stride);
      else copy16_cv(out+(long long)r*blk+off,h->recv[r][slot]+(long long)r*m,m,first,stride);
    }
  } else {
    const float* mine=reinterpret_cast<const float*>(own+(long long)rank*blk+off);float* o=reinterpret_cast<float*>(out+off);
    for(long long i=first;i<m/16;i+=stride){
      float4 acc=make_float4(0.f,0.f,0.f,0.f);
      for(int r=0;r<world;++r){
        const float4 x=r==rank?reinterpret_cast<const float4*>(mine)[i]:__ldcv(reinterpret_cast<const float4*>(h->recv[r][slot]+(long long)rank*m)+i);
        if(r==0)acc=x;else{acc.x=__fadd_rn(acc.x,x.x);acc.y=__fadd_rn(acc.y,x.y);acc.z=__fadd_rn(acc.z,x.z);acc.w=__fadd_rn(acc.w,x.w);}
      }
      reinterpret_cast<float4*>(o)[i]=acc;
    }
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0&&atomicAdd(&st->fdone,1u)==gridDim.x-1){st->fdone=0;st->fseq=s;}
}

__global__ void arn_materialize_kernel(float* data,long long n,int rank,int world,const Host* h,const DevState* st,int round){
  const int slot=(int)(st->seq%NSLOT);
  for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
    float acc=0.f;
    for(int r=0;r<world;++r){
      const float x=r==rank?data[i]:__ldcv(reinterpret_cast<const float*>(h->recv[r][slot])+i);
      acc=r==0?x:__fadd_rn(acc,x);
    }
    if(round)acc=__half2float(__float2half_rn(acc));
    data[i]=acc;
  }
}

// RoCE devices with the IPv4 address and netmask of their netdev.
std::vector<DevInfo> local_devices(){
  std::vector<DevInfo> out;int n=0;ibv_device** list=ibv_get_device_list(&n);
  ifaddrs* ifa=nullptr;getifaddrs(&ifa);
  for(int i=0;i<n;++i){
    const char* name=ibv_get_device_name(list[i]);
    std::string dir=std::string("/sys/class/infiniband/")+name+"/device/net";
    DIR* d=opendir(dir.c_str());if(!d)continue;
    std::string netdev;while(dirent* e=readdir(d)){if(e->d_name[0]!='.'){netdev=e->d_name;break;}}closedir(d);
    for(ifaddrs* a=ifa;a;a=a->ifa_next){
      if(!a->ifa_addr||a->ifa_addr->sa_family!=AF_INET||netdev!=a->ifa_name)continue;
      DevInfo di{};snprintf(di.name,sizeof di.name,"%s",name);
      di.ip=((sockaddr_in*)a->ifa_addr)->sin_addr.s_addr;di.mask=((sockaddr_in*)a->ifa_netmask)->sin_addr.s_addr;
      out.push_back(di);break;
    }
  }
  freeifaddrs(ifa);ibv_free_device_list(list);
  std::sort(out.begin(),out.end(),[](const DevInfo& a,const DevInfo& b){return strcmp(a.name,b.name)<0;});
  return out;
}

// The GID index holding the IPv4-mapped RoCE v2 address `ip`.
int find_gid(ibv_context* ctx,const char* name,uint32_t ip){return rdma_ipv4_v2_gid(ctx,name,ip,"GLM53_RDMA_ARN_GID");}

Dev* open_dev(const DevInfo& di){
  for(Dev* d:g_devs)if(d->name==di.name)return d;
  int n=0;ibv_device** list=ibv_get_device_list(&n);ibv_device* dev=nullptr;
  for(int i=0;i<n;++i)if(!strcmp(ibv_get_device_name(list[i]),di.name))dev=list[i];
  Dev* d=new Dev;d->name=di.name;d->ip=di.ip;
  if(dev)d->ctx=ibv_open_device(dev);
  ibv_free_device_list(list);
  if(!d->ctx){fprintf(stderr,"[rdma_arn] cannot open %s\n",di.name);return nullptr;}
  d->pd=ibv_alloc_pd(d->ctx);
  d->mr=d->pd?ibv_reg_mr(d->pd,g_host,sizeof(Host),IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE):nullptr;
  d->gid_index=find_gid(d->ctx,di.name,di.ip);
  if(!d->mr||d->gid_index<0){fprintf(stderr,"[rdma_arn] %s: pd/mr/gid setup failed (gid %d)\n",di.name,d->gid_index);return nullptr;}
  g_devs.push_back(d);return d;
}

bool qp_to_rts(Link& L){
  ibv_qp_attr a{};a.qp_state=IBV_QPS_INIT;a.pkey_index=0;a.port_num=1;a.qp_access_flags=IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE;
  if(ibv_modify_qp(L.qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))return false;
  ibv_qp_attr r{};r.qp_state=IBV_QPS_RTR;r.path_mtu=IBV_MTU_4096;r.dest_qp_num=L.remote.qpn;r.rq_psn=L.remote.psn;
  r.max_dest_rd_atomic=1;r.min_rnr_timer=12;r.ah_attr.is_global=1;r.ah_attr.port_num=1;
  memcpy(r.ah_attr.grh.dgid.raw,L.remote.gid,16);r.ah_attr.grh.sgid_index=L.dev->gid_index;r.ah_attr.grh.hop_limit=1;
  if(ibv_modify_qp(L.qp,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER))return false;
  ibv_qp_attr t{};t.qp_state=IBV_QPS_RTS;t.timeout=14;t.retry_cnt=7;t.rnr_retry=7;t.sq_psn=0;t.max_rd_atomic=1;
  return ibv_modify_qp(L.qp,&t,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC)==0;
}

void post_write(Link& L,const void* local,size_t bytes,uint64_t remote,bool signaled){
  ibv_sge sge{};sge.addr=(uintptr_t)local;sge.length=(uint32_t)bytes;sge.lkey=L.dev->mr->lkey;
  ibv_send_wr wr{},*bad=nullptr;wr.opcode=IBV_WR_RDMA_WRITE;wr.sg_list=&sge;wr.num_sge=1;
  wr.send_flags=(signaled?IBV_SEND_SIGNALED:0)|(bytes<=64?IBV_SEND_INLINE:0);
  wr.wr.rdma.remote_addr=remote;wr.wr.rdma.rkey=L.remote.rkey;
  if(ibv_post_send(L.qp,&wr,&bad)){fprintf(stderr,"[rdma_arn] post_send failed\n");std::abort();}
}

void proxy_loop(int j){
  Peer& P=g_peers[j];unsigned long long s=1;
  const uint64_t recv_dst=P.addr+offsetof(Host,recv)+(uint64_t)g_rank*NSLOT*SLOT_BYTES;
  // signaled writes (one flag per call and link) not yet reaped: up to MAXOUT calls stay on the wire per link, so a
  // pipelined large allreduce keeps the links busy (the old loop waited for every call's completion before the next)
  constexpr int MAXOUT=8;int out[MAXL]={0,0};
  auto reap=[&](int l,bool block){
    ibv_wc wc[8];int got;
    do{got=ibv_poll_cq(P.link[l].cq,8,wc);}while(block&&got==0);
    if(got<0){fprintf(stderr,"[rdma_arn] poll error peer %d link %d\n",j,l);std::abort();}
    for(int i=0;i<got;++i)if(wc[i].status!=IBV_WC_SUCCESS){fprintf(stderr,"[rdma_arn] completion error peer %d link %d: %d\n",j,l,(int)wc[i].status);std::abort();}
    out[l]-=got;
  };
  // idle back-off (GLM53_RDMA_ARN_IDLE_US, default 5000; 0: always spin): with no call for that long (no request
  // being served) the loop polls every ~100 us instead of spinning a core; while serving, calls come far more often
  // and it never sleeps (the first call after an idle spell waits at most one sleep)
  static const long long idle_us=[]{const char* e=std::getenv("GLM53_RDMA_ARN_IDLE_US");return e?std::atoll(e):5000ll;}();
  auto last=std::chrono::steady_clock::now();unsigned spins=0;bool idle=false;
  while(!g_stop.load(std::memory_order_relaxed)){
    const int slot=(int)(s%NSLOT);
    if(g_host->ready_seq[slot]!=s){
      for(int l=0;l<P.nlink;++l)if(out[l])reap(l,false);
      if(idle){timespec ts{0,100000};nanosleep(&ts,nullptr);continue;}
      if(idle_us>0&&++spins>=4096){
        spins=0;
        idle=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-last).count()>idle_us;
      }
      asm volatile("yield");continue;
    }
    if(idle_us>0){last=std::chrono::steady_clock::now();spins=0;idle=false;}
    std::atomic_thread_fence(std::memory_order_acquire);
    const unsigned long long rb=g_host->ready_bytes[slot];
    const size_t all=(size_t)(rb&BYTES_MASK);const unsigned long long mode=rb&(MODE_RS|MODE_AG);const size_t esz=(rb&MODE_BF16)?2:4;
    // the byte range of the slot this peer gets
    size_t r0=0,bytes=all;
    if(mode){const long long n=(long long)(all/esz);const int k=mode==MODE_RS?j:g_rank;r0=(size_t)part_lo(n,k,g_world)*esz;bytes=(size_t)part_lo(n,k+1,g_world)*esz-r0;}
    const uint64_t dst=recv_dst+(uint64_t)slot*SLOT_BYTES+r0;
    const bool two=P.nlink==2&&g_split&&bytes>=g_split;
    const size_t part[2]={two?((bytes/2+63)&~(size_t)63):bytes,0};
    const int nl=two?2:1;
    for(int l=0;l<nl;++l){
      while(out[l]>=MAXOUT)reap(l,true);
      const size_t off=l==0?0:part[0],len=l==0?part[0]:bytes-part[0];
      post_write(P.link[l],g_host->send[slot]+r0+off,len,dst+off,false);
      g_host->flag_src[j][l][slot]=s;
      post_write(P.link[l],&g_host->flag_src[j][l][slot],8,P.addr+offsetof(Host,recv_flag)+(((uint64_t)g_rank*MAXL+l)*NSLOT+slot)*8,true);
      ++out[l];
    }
    for(int l=0;l<nl;++l)if(out[l])reap(l,false);
    ++s;
  }
}

bool kv_put(KvSet set,const std::string& k,const void* v,size_t n){return set(k.c_str(),v,n)==0;}
bool kv_take(KvGet get,const std::string& k,void* v,size_t n){return get(k.c_str(),v,n)==0;}
} // namespace

// rank/world from the process group; kv_set/kv_get wrap its TCPStore (blocking get). 0 on success.
extern "C" int glm53_rdma_arn_init(int rank,int world,KvSet kv_set,KvGet kv_get){
  if(g_host)return 0;
  if(world<2||world>MAXW||rank<0||rank>=world)return 20;
  g_rank=rank;g_world=world;
  if(glm53_host_alloc_mapped((void**)&g_host,sizeof(Host))!=cudaSuccess)return 1;
  memset((void*)g_host,0,sizeof(Host));
  if(cudaHostGetDevicePointer((void**)&g_host_dev,g_host,0)!=cudaSuccess)return 2;
  if(cudaMalloc(&g_state,sizeof(DevState))!=cudaSuccess)return 3;cudaMemset(g_state,0,sizeof(DevState));
  {const char* k=std::getenv("GLM53_RDMA_ARN_SPLIT_KB");g_split=(unsigned long long)(k?std::atoi(k):32)*1024;}
  // 1. publish this rank's devices, read everyone's
  std::vector<DevInfo> mine=local_devices();
  uint32_t nd=(uint32_t)mine.size();
  std::vector<uint8_t> blob(4+nd*sizeof(DevInfo));memcpy(blob.data(),&nd,4);memcpy(blob.data()+4,mine.data(),nd*sizeof(DevInfo));
  if(!kv_put(kv_set,"arn/dev/"+std::to_string(rank),blob.data(),blob.size()))return 4;
  std::vector<std::vector<DevInfo>> all(world);
  for(int r=0;r<world;++r){
    std::vector<uint8_t> b(4+16*sizeof(DevInfo));
    if(!kv_take(kv_get,"arn/dev/"+std::to_string(r),b.data(),b.size()))return 5;
    uint32_t c;memcpy(&c,b.data(),4);if(c>16)return 6;all[r].resize(c);memcpy(all[r].data(),b.data()+4,c*sizeof(DevInfo));
  }
  // 2. per peer: links = same-subnet device pairs, ordered by the lower rank's device name; create the QPs
  for(int j=0;j<world;++j){
    if(j==rank)continue;Peer& P=g_peers[j];P.rank=j;
    const int lo=std::min(rank,j);
    std::vector<std::pair<const DevInfo*,const DevInfo*>> pairs;   // (lower rank's device, higher rank's device)
    for(const DevInfo& a:all[lo])for(const DevInfo& b:all[lo==rank?j:rank])
      if((a.ip&a.mask)==(b.ip&b.mask)&&a.mask==b.mask)pairs.push_back({&a,&b});
    if(pairs.empty()){fprintf(stderr,"[rdma_arn] rank%d: no RoCE subnet shared with rank%d\n",rank,j);return 7;}
    P.nlink=(int)std::min<size_t>(pairs.size(),MAXL);
    const char* single=std::getenv("GLM53_RDMA_ARN_SINGLE");if(single&&!strcmp(single,"1"))P.nlink=1;
    for(int l=0;l<P.nlink;++l){
      const DevInfo& local=rank==lo?*pairs[l].first:*pairs[l].second;
      Link& L=P.link[l];L.dev=open_dev(local);if(!L.dev)return 8;
      L.cq=ibv_create_cq(L.dev->ctx,64,nullptr,nullptr,0);if(!L.cq)return 9;
      ibv_qp_init_attr qa{};qa.send_cq=L.cq;qa.recv_cq=L.cq;qa.qp_type=IBV_QPT_RC;qa.cap.max_send_wr=64;qa.cap.max_recv_wr=1;
      qa.cap.max_send_sge=1;qa.cap.max_recv_sge=1;qa.cap.max_inline_data=64;
      L.qp=ibv_create_qp(L.dev->pd,&qa);if(!L.qp)return 10;
    }
    PeerInfo pi{};pi.addr=(uint64_t)(uintptr_t)g_host;pi.nlink=(uint32_t)P.nlink;
    for(int l=0;l<P.nlink;++l){
      Link& L=P.link[l];ibv_gid gid;if(ibv_query_gid(L.dev->ctx,1,L.dev->gid_index,&gid))return 11;
      pi.link[l].qpn=L.qp->qp_num;pi.link[l].psn=0;pi.link[l].rkey=L.dev->mr->rkey;memcpy(pi.link[l].gid,gid.raw,16);
    }
    if(!kv_put(kv_set,"arn/qp/"+std::to_string(rank)+"/"+std::to_string(j),&pi,sizeof pi))return 12;
  }
  // 3. connect
  for(int j=0;j<world;++j){
    if(j==rank)continue;Peer& P=g_peers[j];PeerInfo pi{};
    if(!kv_take(kv_get,"arn/qp/"+std::to_string(j)+"/"+std::to_string(rank),&pi,sizeof pi))return 13;
    if((int)pi.nlink!=P.nlink){fprintf(stderr,"[rdma_arn] link count with rank%d differs (%d vs %u)\n",j,P.nlink,pi.nlink);return 14;}
    P.addr=pi.addr;
    for(int l=0;l<P.nlink;++l){P.link[l].remote=pi.link[l];if(!qp_to_rts(P.link[l])){fprintf(stderr,"[rdma_arn] QP to rank%d link %d failed\n",j,l);return 15;}}
  }
  // 4. barrier: every QP of every rank is RTS before anyone writes
  {const char ok='1';if(!kv_put(kv_set,"arn/ready/"+std::to_string(rank),&ok,1))return 16;
   for(int r=0;r<world;++r){char c;if(!kv_take(kv_get,"arn/ready/"+std::to_string(r),&c,1))return 17;}}
  // 5. one proxy per peer, pinned to the big cores (GLM53_RDMA_ARN_CPUS, default "19,18,17")
  std::vector<int> cpus;{const char* c=std::getenv("GLM53_RDMA_ARN_CPUS");std::string s=c?c:"19,18,17";size_t p=0;
    while(p<s.size()){size_t q=s.find(',',p);cpus.push_back(std::atoi(s.substr(p,q-p).c_str()));if(q==std::string::npos)break;p=q+1;}}
  int k=0;
  for(int j=0;j<world;++j){
    if(j==rank)continue;std::thread t(proxy_loop,j);
    char nm[16];snprintf(nm,sizeof nm,"glm53-arn%d",j);pthread_setname_np(t.native_handle(),nm);
    const int cpu=k<(int)cpus.size()?cpus[k]:-1;++k;
    if(cpu>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);
      if(pthread_setaffinity_np(t.native_handle(),sizeof(set),&set))fprintf(stderr,"[rdma_arn] affinity cpu%d failed\n",cpu);}
    t.detach();
  }
  for(int j=0;j<world;++j){
    if(j==rank)continue;const Peer& P=g_peers[j];
    fprintf(stderr,"[rdma_arn] rank%d<->rank%d:",rank,j);
    for(int l=0;l<P.nlink;++l)fprintf(stderr," %s(gid %d)",P.link[l].dev->name.c_str(),P.link[l].dev->gid_index);
    fprintf(stderr,"\n");
  }
  fprintf(stderr,"[rdma_arn] rank%d/%d ready: slots=%d x %zu KiB split>=%llu B\n",rank,world,NSLOT,SLOT_BYTES/1024,g_split);
  return 0;
}
extern "C" size_t glm53_rdma_arn_max_bytes(){return SLOT_BYTES;}
static int arn_blocks(long long n){int b=(int)((n+1023)/1024);return b>32?32:(b<1?1:b);}
extern "C" int glm53_rdma_arn_cuda(float* data,long long n,cudaStream_t s){
  if(!g_host||n<1||(size_t)n*4>SLOT_BYTES)return 1;
  arn_kernel<true><<<arn_blocks(n),256,0,s>>>(data,n,g_rank,g_world,g_host_dev,g_state,g_split);
  return (int)cudaGetLastError();
}
extern "C" int glm53_rdma_arn_send_cuda(float* data,long long n,cudaStream_t s){
  if(!g_host||n<1||(size_t)n*4>SLOT_BYTES)return 1;
  arn_kernel<false><<<arn_blocks(n),256,0,s>>>(data,n,g_rank,g_world,g_host_dev,g_state,g_split);
  return (int)cudaGetLastError();
}
// Large allreduce: slot-sized chunks, each a reduce-scatter call and an all-gather call. Iteration t publishes RS(t)
// and (inside rs_finish) AG(t - K), finishes RS(t - K) and AG(t - 2K): the finishes of iteration t are exactly the
// publications of iteration t - K in the same order, so the device counters stay monotonic. Slot reuse: a slot is
// rewritten 2 NSLOT / 2 calls later; at most ~4K + skew calls are unfinished (K = GLM53_RDMA_ARN_PIPE, 3; NSLOT 32).
// GLM53_RDMA_ARN_RSAG=0: one-shot calls (every peer gets the whole chunk), K in flight.
extern "C" int glm53_rdma_arn_big_cuda(float* data,long long n,cudaStream_t s){
  if(!g_host||n<1)return 1;
  static const bool rsag=[]{const char* e=std::getenv("GLM53_RDMA_ARN_RSAG");return !(e&&e[0]=='0');}();
  static const int K=[]{const char* e=std::getenv("GLM53_RDMA_ARN_PIPE");int k=e?std::atoi(e):3;return k<1?1:(k>6?6:k);}();
  const long long chunk=(long long)(SLOT_BYTES/4),nch=(n+chunk-1)/chunk;
  auto len=[&](long long i){return std::min(chunk,n-i*chunk);};
  // more blocks than the small path (GLM53_RDMA_ARN_BIG_BLOCKS, 96): a 1 MiB chunk per kernel
  static const int nb=[]{const char* e=std::getenv("GLM53_RDMA_ARN_BIG_BLOCKS");int b=e?std::atoi(e):96;return b<1?1:b;}();
  auto blocks=[&](long long m){long long b=(m/4+255)/256;return (int)std::max(1ll,std::min<long long>(nb,b));};
  if(!rsag){
    for(long long i=0;i<nch;++i){
      arn_send_kernel<<<blocks(len(i)),256,0,s>>>(data+i*chunk,len(i),g_host_dev,g_state);
      if(i>=K)arn_finish_kernel<<<blocks(len(i-K)),256,0,s>>>(data+(i-K)*chunk,len(i-K),g_rank,g_world,g_host_dev,g_state,g_split);
    }
    for(long long i=std::max(0ll,nch-K);i<nch;++i)
      arn_finish_kernel<<<blocks(len(i)),256,0,s>>>(data+i*chunk,len(i),g_rank,g_world,g_host_dev,g_state,g_split);
    return (int)cudaGetLastError();
  }
  for(long long t=0;t<nch+2*K;++t){
    if(t<nch)arn_rs_send_kernel<<<blocks(len(t)),256,0,s>>>(data+t*chunk,len(t),g_host_dev,g_state);
    const long long a=t-K,b=t-2*K;
    if(a>=0&&a<nch)arn_rs_finish_kernel<<<blocks(len(a)),256,0,s>>>(data+a*chunk,len(a),g_rank,g_world,g_host_dev,g_state,g_split);
    if(b>=0&&b<nch)arn_ag_finish_kernel<<<blocks(len(b)),256,0,s>>>(data+b*chunk,len(b),g_rank,g_world,g_host_dev,g_state,g_split);
  }
  return (int)cudaGetLastError();
}
// BF16-transport large allreduce (prefill layer outputs): chunks of 2 SLOT_BYTES / 2 elements, the schedule of the
// FP32 path. data ends up as FP32 holding BF16-rounded sums, identical on every rank.
extern "C" int glm53_rdma_arn_big_bf16_cuda(float* data,long long n,cudaStream_t s){
  if(!g_host||n<1)return 1;
  static const int K=[]{const char* e=std::getenv("GLM53_RDMA_ARN_PIPE");int k=e?std::atoi(e):3;return k<1?1:(k>6?6:k);}();
  const long long chunk=(long long)(SLOT_BYTES/2),nch=(n+chunk-1)/chunk;
  auto len=[&](long long i){return std::min(chunk,n-i*chunk);};
  static const int nb=[]{const char* e=std::getenv("GLM53_RDMA_ARN_BIG_BLOCKS");int b=e?std::atoi(e):96;return b<1?1:b;}();
  auto blocks=[&](long long m){long long b=(m/2+255)/256;return (int)std::max(1ll,std::min<long long>(nb,b));};
  for(long long t=0;t<nch+2*K;++t){
    if(t<nch)arn_rs_send_bf16_kernel<<<blocks(len(t)),256,0,s>>>(data+t*chunk,len(t),g_host_dev,g_state);
    const long long a=t-K,b=t-2*K;
    if(a>=0&&a<nch)arn_rs_finish_bf16_kernel<<<blocks(len(a)),256,0,s>>>(data+a*chunk,len(a),g_rank,g_world,g_host_dev,g_state,g_split);
    if(b>=0&&b<nch)arn_ag_finish_bf16_kernel<<<blocks(len(b)),256,0,s>>>(data+b*chunk,len(b),g_rank,g_world,g_host_dev,g_state,g_split);
  }
  return (int)cudaGetLastError();
}
// Calls in flight of the row-block reduce-scatter / all-gather (one call per chunk, so at most about K unfinished, against
// NSLOT 32 slots): GLM53_RDMA_ARN_ROWS_PIPE (3, at most 12; the large allreduce keeps GLM53_RDMA_ARN_PIPE <= 6: it has
// ~4K unfinished). Every chunk's round trip (send kernel, proxy, RDMA write, flag, finish kernel) is a latency the deeper
// pipeline hides: DS41 TP3 32k prefill (two 2048-row parts per pass) 3 -> 7018 ms, 6 6875, 8 6823, 12 6761 (10-26;
// start-ds41.sh / bench/ds41/tp3-run.sh set 12); GLM TP2 prefill level within noise (8k 6390 / 6420 ms, 32k 25640 /
// 25820 ms), so the default stays 3. The sums are per element in rank order whatever the chunking: the same bits.
static int arn_rows_pipe(){
  const char* e=std::getenv("GLM53_RDMA_ARN_ROWS_PIPE");
  const int k=e?std::atoi(e):3;return k<1?1:(k>12?12:k);
}
// Fallback consumer of a send-only call: data = rank-ordered sum in place (optionally Half-rounded).
// Row-block reduce-scatter (ag = 0: FP32; `in` world blocks of blk bytes, `out` this rank's block) or all-gather (ag = 1:
// any element size; `in` this rank's block, `out` world blocks). blk: a multiple of 16 bytes. Chunks of up to a slot's
// share per block, K (GLM53_RDMA_ARN_PIPE) calls in flight as the large allreduce. Each rank sends (world - 1) / world of
// the buffer, as NCCL's ring does; the reduce-scatter sums in rank order (every rank's block bitwise as the allreduce's).
extern "C" int glm53_rdma_arn_rows_bf16_cuda(const float* in,float* out,long long blk,cudaStream_t s);
extern "C" int glm53_rdma_arn_rows_cuda(const void* in,void* out,long long blk,int ag,cudaStream_t s){
  if(!g_host||blk<16||blk%16)return 1;
  static const bool rs_bf16=[]{const char* e=std::getenv("GLM53_RDMA_ARN_RS_BF16");return e&&e[0]=='1';}();
  if(!ag&&rs_bf16)return glm53_rdma_arn_rows_bf16_cuda((const float*)in,(float*)out,blk,s);
  static const int K=arn_rows_pipe();
  static const int nb=[]{const char* e=std::getenv("GLM53_RDMA_ARN_BIG_BLOCKS");int b=e?std::atoi(e):96;return b<1?1:b;}();
  const long long mmax=(long long)(SLOT_BYTES/g_world)/64*64,nch=(blk+mmax-1)/mmax;
  auto len=[&](long long i){return std::min(mmax,blk-i*mmax);};
  auto blocks=[&](long long m){long long b=(m/16+255)/256;return (int)std::max(1ll,std::min<long long>(nb,b));};
  // the proxy reads the element size from the flags only to split the slot into parts: 2-byte units keep every region
  // boundary exact for any m (a multiple of 16)
  const unsigned long long flags=(ag?MODE_AG:MODE_RS)|MODE_BF16;
  const unsigned char* i8=(const unsigned char*)in;unsigned char* o8=(unsigned char*)out;
  for(long long t=0;t<nch+K;++t){
    if(t<nch)arn_rows_send_kernel<<<blocks(len(t)),256,0,s>>>(i8,blk,t*mmax,len(t),g_rank,g_world,ag,flags,g_host_dev,g_state);
    const long long a=t-K;
    if(a>=0)arn_rows_finish_kernel<<<blocks(len(a)),256,0,s>>>(o8,i8,blk,a*mmax,len(a),g_rank,g_world,ag,g_host_dev,g_state,g_split);
  }
  return (int)cudaGetLastError();
}
// Row-block reduce-scatter with BF16 on the wire (FP32 blocks of blk bytes): half the bytes; the peers' partials are rounded
// to BF16, this rank's own stays FP32, the rank-ordered sum is FP32 (L1: GLM53_RDMA_ARN_RS_BF16=1).
extern "C" int glm53_rdma_arn_rows_bf16_cuda(const float* in,float* out,long long blk,cudaStream_t s){
  if(!g_host||blk<16||blk%16)return 1;
  static const int K=arn_rows_pipe();
  static const int nb=[]{const char* e=std::getenv("GLM53_RDMA_ARN_BIG_BLOCKS");int b=e?std::atoi(e):96;return b<1?1:b;}();
  const long long blk_f=blk/4,mmax=(long long)(SLOT_BYTES/2/g_world)/32*32,nch=(blk_f+mmax-1)/mmax;   // FP32 elements per region
  auto len=[&](long long i){return std::min(mmax,blk_f-i*mmax);};
  auto blocks=[&](long long m){long long b=(m/2+255)/256;return (int)std::max(1ll,std::min<long long>(nb,b));};
  const unsigned long long flags=MODE_RS|MODE_BF16;
  for(long long t=0;t<nch+K;++t){
    if(t<nch)arn_rows_send_bf16_kernel<<<blocks(len(t)),256,0,s>>>(in,blk_f,t*mmax,len(t),g_world,flags,g_host_dev,g_state);
    const long long a=t-K;
    if(a>=0)arn_rows_finish_bf16_kernel<<<blocks(len(a)),256,0,s>>>(out,in,blk_f,a*mmax,len(a),g_rank,g_world,g_host_dev,g_state,g_split);
  }
  return (int)cudaGetLastError();
}
// The BF16 allreduce's sum as a row-block reduce-scatter (arn_rows_finish_bf16r_kernel): `in` FP32, world blocks of blk_f
// elements of which the first n_valid exist; `out` this rank's block as BF16. blk_f: a multiple of 8.
template<class T> static int rows_bf16r(typename std::conditional<std::is_same<T,Comb>::value,Comb,const T*>::type in,void* out,long long blk_f,long long n_valid,cudaStream_t s){
  if(!g_host||blk_f<8||blk_f%8)return 1;
  static const int K=arn_rows_pipe();
  static const int nb=[]{const char* e=std::getenv("GLM53_RDMA_ARN_BIG_BLOCKS");int b=e?std::atoi(e):96;return b<1?1:b;}();
  const long long mmax=(long long)(SLOT_BYTES/2/g_world)/32*32,nch=(blk_f+mmax-1)/mmax;
  auto len=[&](long long i){return std::min(mmax,blk_f-i*mmax);};
  auto blocks=[&](long long m){long long b=(m/2+255)/256;return (int)std::max(1ll,std::min<long long>(nb,b));};
  const unsigned long long flags=MODE_RS|MODE_BF16;
  for(long long t=0;t<nch+K;++t){
    if(t<nch)arn_rows_send_bf16v_kernel<T><<<blocks(len(t)),256,0,s>>>(in,blk_f,t*mmax,len(t),n_valid,g_world,flags,g_host_dev,g_state);
    const long long a=t-K;
    if(a>=0)arn_rows_finish_bf16r_kernel<<<blocks(len(a)),256,0,s>>>((__nv_bfloat16*)out,len(a),a*mmax,g_rank,g_world,g_host_dev,g_state,g_split);
  }
  return (int)cudaGetLastError();
}
extern "C" int glm53_rdma_arn_rows_bf16r_cuda(const float* in,void* out,long long blk_f,long long n_valid,cudaStream_t s){ return rows_bf16r<float>(in,out,blk_f,n_valid,s); }
// the same with a BF16 partial (`in` BF16 values: bitwise the FP32 call on their FP32 copy)
extern "C" int glm53_rdma_arn_rows_bf16r_b16_cuda(const void* in,void* out,long long blk_f,long long n_valid,cudaStream_t s){ return rows_bf16r<__nv_bfloat16>((const __nv_bfloat16*)in,out,blk_f,n_valid,s); }
// the same of a routed-expert combine (Comb: `pairmap` / `add` at the part's first row, out [P, hcols] FP32)
extern "C" int glm53_rdma_arn_rows_bf16r_comb_cuda(const float* pout,const int* pairmap,const float* add,int topk,int hcols,void* out,long long blk_f,long long n_valid,cudaStream_t s){
  if(hcols%2)return 1;
  return rows_bf16r<Comb>(Comb{pout,pairmap,add,topk,hcols},out,blk_f,n_valid,s);
}
extern "C" int glm53_rdma_arn_materialize_cuda(float* data,long long n,int round,cudaStream_t s){
  if(!g_host||n<1||(size_t)n*4>SLOT_BYTES)return 1;
  arn_materialize_kernel<<<arn_blocks(n),256,0,s>>>(data,n,g_rank,g_world,g_host_dev,g_state,round);
  return (int)cudaGetLastError();
}
// Device view for fused consumers: recv area of sender 0 (sender r at + r*rank_stride floats), slot = seq % nslot.
extern "C" int glm53_rdma_arn_peer(const float** recv_base,const unsigned long long** seq,int* slot_floats,int* nslot,
                                   long long* rank_stride,int* rank,int* world){
  if(!g_host)return 1;
  *recv_base=reinterpret_cast<const float*>(g_host_dev->recv[0][0]);*seq=&g_state->seq;*slot_floats=(int)(SLOT_BYTES/4);
  *nslot=NSLOT;*rank_stride=(long long)NSLOT*SLOT_BYTES/4;*rank=g_rank;*world=g_world;return 0;
}
