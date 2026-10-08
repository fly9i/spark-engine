#include <string>
#include <chrono>
// SPDX-License-Identifier: MIT
// C2: lean TP2 FP32 allreduce over one RoCE RC queue pair (GLM53_RDMA_AR=1).
//
// Data path per call (both ranks, same stream order):
//   GPU kernel: copy local data -> pinned send slot (host-mapped), last block publishes
//   {bytes, seq} in local_ready; every block spins on recv_flag[slot]==seq (written by
//   the peer's NIC after the payload), then out = x_rank0 + x_rank1 (commutative: bitwise
//   equal to a 2-rank NCCL sum). A CPU proxy thread watches local_ready and posts two RDMA
//   WRITEs on the same QP: payload, then the 8-byte flag (same-QP writes are placed in
//   order on host memory). GB10 GPU accesses pinned memory coherently (ATS).
// Slots form a ring so the NIC never reads a slot the GPU is rewriting.
// GLM53_RDMA_AR_DUAL=1 (default off): a second RC queue pair on the other RoCE function (GLM53_RDMA_AR_DEV1, default
// roceP2p1s0f0). A message of at least GLM53_RDMA_AR_SPLIT_KB (default 32) KiB is split at a 64-byte boundary: the
// first part and flag go on link 0, the second part and a second flag on link 1, in parallel (measured ib_write_bw
// 109 Gb/s per function alone, 98+98 together); the GPU waits for both flags. Same bytes land in the same recv slot,
// so every consumer and the sum are unchanged (L0).
#include <pthread.h>
#include <cuda_runtime.h>
#include "host_pin.cuh"
#include <cuda_fp16.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {
constexpr int NSLOT=16;
constexpr size_t SLOT_BYTES=1024*1024;   // 1 MiB: batched verify windows up to 32 rows (MoE packed sum 2 x 32 x 4096 FP32); was 256 KiB (16 rows)
struct Host {                         // pinned, host-mapped, NIC-registered
  alignas(64) volatile unsigned long long ready_seq[NSLOT];
  alignas(64) volatile unsigned long long ready_bytes[NSLOT];
  alignas(64) volatile unsigned long long recv_flag[NSLOT];
  alignas(64) unsigned long long flag_src[NSLOT];
  alignas(64) volatile unsigned long long recv_flag2[NSLOT];     // link 1 part of a split message
  alignas(64) unsigned long long flag_src2[NSLOT];
  alignas(4096) unsigned char send[NSLOT][SLOT_BYTES];
  alignas(4096) unsigned char recv[NSLOT][SLOT_BYTES];
};
struct DevState {unsigned long long seq;unsigned int arrive;unsigned int done;};
// GLM53_AR_STATS=<path> (diagnostic, default off): block 0 of every allreduce stamps %globaltimer at entry, after its local
// copy, when the peer flag is seen and at exit, into a pinned ring keyed by the sequence number (identical on both ranks);
// a host thread dumps the ring to <path>.rank<r> every 5 s. Offline alignment by seq separates copy, wait (transfer + peer
// lateness) and add; the per-seq difference of the two ranks' waits is the load skew.
constexpr int STAT_N=65536;
struct Stamp {unsigned long long seq,t0,t1,t2,t3,n;};   // n: elements of the call
constexpr int NLINK=2;
struct Peer {uint32_t nlink,qpn[NLINK],psn[NLINK],rkey[NLINK];uint64_t addr;uint8_t gid[NLINK][16];};

Host* g_host=nullptr;Host* g_host_dev=nullptr;DevState* g_state=nullptr;int g_rank=-1;
Stamp* g_stats=nullptr;Stamp* g_stats_dev=nullptr;
ibv_context* g_ctx[NLINK]={};ibv_pd* g_pd[NLINK]={};ibv_cq* g_cq[NLINK]={};ibv_qp* g_qp[NLINK]={};ibv_mr* g_mr[NLINK]={};
int g_nlink=1;unsigned long long g_split=0;   // bytes at or above which a message is split over both links (0: never)
Peer g_peer{};std::thread g_proxy;std::atomic<bool> g_stop{false};

__device__ __forceinline__ unsigned long long gtimer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
// SUM=false (GLM53_AR_FUSED=1, I3 step 2): send + wait only. `data` keeps this rank's partial; the single consumer reads
// it and the peer's recv slot (glm53_rdma_ar_peer) and forms rank0+rank1 itself. The slot stays valid until this rank
// launches its next allreduce: the peer can refill it only for seq+NSLOT, which needs this rank's seq+NSLOT-1 publish.
template<bool SUM>
__global__ void ar_kernel(float* data,long long n,int rank,Host* h,DevState* st,Stamp* stats,unsigned long long split){
  const unsigned long long s=st->seq+1;const int slot=(int)(s%NSLOT);
  const bool stamp=stats&&blockIdx.x==0&&threadIdx.x==0;unsigned long long t0=0,t1=0,t2=0;if(stamp)t0=gtimer();
  float* snd=reinterpret_cast<float*>(h->send[slot]);const float* rcv=reinterpret_cast<const float*>(h->recv[slot]);
  const long long stride=(long long)gridDim.x*blockDim.x,first=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  for(long long i=first;i<n;i+=stride)snd[i]=data[i];
  __threadfence_system();__syncthreads();
  if(stamp)t1=gtimer();
  if(threadIdx.x==0){
    if(atomicAdd(&st->arrive,1u)==gridDim.x-1){st->arrive=0;h->ready_bytes[slot]=(unsigned long long)n*4;__threadfence_system();h->ready_seq[slot]=s;__threadfence_system();}
    long long spins=0;
    const bool two=split&&(unsigned long long)n*4>=split;
    while(h->recv_flag[slot]!=s||(two&&h->recv_flag2[slot]!=s)){if(++spins>(1ll<<34))__trap();}
    if(stamp)t2=gtimer();
  }
  __syncthreads();
  if constexpr(SUM) for(long long i=first;i<n;i+=stride){
    const float mine=data[i];const float peer=__ldcv(rcv+i);
    data[i]=rank==0?mine+peer:peer+mine;
  }
  __threadfence();__syncthreads();
  if(stamp){Stamp* e=stats+(s%STAT_N);e->t0=t0;e->t1=t1;e->t2=t2;e->t3=gtimer();e->n=(unsigned long long)n;__threadfence_system();e->seq=s;}
  if(threadIdx.x==0&&atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;}
}

bool sock_exchange(const Peer& mine,Peer& theirs){
  const char* addr=std::getenv("GLM53_MASTER_ADDR");const char* port_s=std::getenv("GLM53_MASTER_PORT");
  if(!addr||!port_s)return false;int port=std::atoi(port_s)+17;
  int fd=-1;
  if(g_rank==0){
    int ls=socket(AF_INET,SOCK_STREAM,0);int one=1;setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(port);a.sin_addr.s_addr=INADDR_ANY;
    if(bind(ls,(sockaddr*)&a,sizeof a)||listen(ls,1)){perror("rdma_ar bind");return false;}
    fd=accept(ls,nullptr,nullptr);close(ls);
  }else{
    for(int t=0;t<600&&fd<0;++t){
      int s=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(port);inet_pton(AF_INET,addr,&a.sin_addr);
      if(connect(s,(sockaddr*)&a,sizeof a)==0){fd=s;break;}close(s);usleep(100000);
    }
  }
  if(fd<0)return false;
  bool ok=write(fd,&mine,sizeof mine)==(ssize_t)sizeof mine && read(fd,&theirs,sizeof theirs)==(ssize_t)sizeof theirs;
  char b=0;ok=ok&&write(fd,&b,1)==1&&read(fd,&b,1)==1;close(fd);return ok;
}

void post_write(int l,const void* local,size_t bytes,uint64_t remote,bool signaled){
  ibv_sge sge{};sge.addr=(uintptr_t)local;sge.length=(uint32_t)bytes;sge.lkey=g_mr[l]->lkey;
  ibv_send_wr wr{},*bad=nullptr;wr.opcode=IBV_WR_RDMA_WRITE;wr.sg_list=&sge;wr.num_sge=1;
  wr.send_flags=(signaled?IBV_SEND_SIGNALED:0)|(bytes<=64?IBV_SEND_INLINE:0);
  wr.wr.rdma.remote_addr=remote;wr.wr.rdma.rkey=g_peer.rkey[l];
  if(ibv_post_send(g_qp[l],&wr,&bad)){fprintf(stderr,"[rdma_ar] post_send failed\n");std::abort();}
}

void proxy_loop(){
  unsigned long long s=1;const uint64_t base=g_peer.addr;
  while(!g_stop.load(std::memory_order_relaxed)){
    const int slot=(int)(s%NSLOT);
    if(g_host->ready_seq[slot]!=s){asm volatile("yield");continue;}
    std::atomic_thread_fence(std::memory_order_acquire);
    const size_t bytes=(size_t)g_host->ready_bytes[slot];
    const uint64_t dst=base+offsetof(Host,recv)+slot*SLOT_BYTES;
    const bool two=g_split&&bytes>=g_split;const size_t first=two?((bytes/2+63)&~(size_t)63):bytes;
    post_write(0,g_host->send[slot],first,dst,false);
    g_host->flag_src[slot]=s;
    post_write(0,&g_host->flag_src[slot],8,base+offsetof(Host,recv_flag)+slot*8,true);
    if(two){
      post_write(1,g_host->send[slot]+first,bytes-first,dst+first,false);
      g_host->flag_src2[slot]=s;
      post_write(1,&g_host->flag_src2[slot],8,base+offsetof(Host,recv_flag2)+slot*8,true);
    }
    for(int l=0;l<(two?2:1);++l){
      ibv_wc wc;int got;do{got=ibv_poll_cq(g_cq[l],1,&wc);}while(got==0);
      if(got<0||wc.status!=IBV_WC_SUCCESS){fprintf(stderr,"[rdma_ar] completion error link %d: %d\n",l,got<0?-1:(int)wc.status);std::abort();}
    }
    ++s;
  }
}

bool qp_to_rts(int l){
  const char* gid_s=std::getenv("GLM53_RDMA_AR_GID");int gid_index=gid_s?std::atoi(gid_s):3;
  ibv_qp* g_qp_l=g_qp[l];
  ibv_qp_attr a{};a.qp_state=IBV_QPS_INIT;a.pkey_index=0;a.port_num=1;a.qp_access_flags=IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE;
  if(ibv_modify_qp(g_qp_l,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))return false;
  ibv_qp_attr r{};r.qp_state=IBV_QPS_RTR;r.path_mtu=IBV_MTU_4096;r.dest_qp_num=g_peer.qpn[l];r.rq_psn=g_peer.psn[l];
  r.max_dest_rd_atomic=1;r.min_rnr_timer=12;r.ah_attr.is_global=1;r.ah_attr.port_num=1;
  memcpy(r.ah_attr.grh.dgid.raw,g_peer.gid[l],16);r.ah_attr.grh.sgid_index=gid_index;r.ah_attr.grh.hop_limit=1;
  if(ibv_modify_qp(g_qp_l,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER))return false;
  ibv_qp_attr t{};t.qp_state=IBV_QPS_RTS;t.timeout=14;t.retry_cnt=7;t.rnr_retry=7;t.sq_psn=0;t.max_rd_atomic=1;
  return ibv_modify_qp(g_qp_l,&t,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC)==0;
}
} // namespace

extern "C" int glm53_rdma_ar_init(int rank){
  if(g_host)return 0;g_rank=rank;
  if(glm53_host_alloc_mapped((void**)&g_host,sizeof(Host))!=cudaSuccess)return 1;
  memset((void*)g_host,0,sizeof(Host));
  if(cudaHostGetDevicePointer((void**)&g_host_dev,g_host,0)!=cudaSuccess)return 2;
  if(cudaMalloc(&g_state,sizeof(DevState))!=cudaSuccess)return 3;cudaMemset(g_state,0,sizeof(DevState));
  const char* d0=std::getenv("GLM53_RDMA_AR_DEV");const char* d1=std::getenv("GLM53_RDMA_AR_DEV1");
  const char* wants[NLINK]={d0?d0:"rocep1s0f0",d1?d1:"roceP2p1s0f0"};const char* want=wants[0];
  g_nlink=std::getenv("GLM53_RDMA_AR_DUAL")&&!strcmp(std::getenv("GLM53_RDMA_AR_DUAL"),"1")?2:1;
  {const char* k=std::getenv("GLM53_RDMA_AR_SPLIT_KB");g_split=g_nlink==2?(unsigned long long)(k?std::atoi(k):32)*1024:0;}
  const char* gid_s=std::getenv("GLM53_RDMA_AR_GID");int gid_index=gid_s?std::atoi(gid_s):3;
  int n=0;ibv_device** list=ibv_get_device_list(&n);
  Peer mine{};mine.nlink=(uint32_t)g_nlink;mine.addr=(uint64_t)(uintptr_t)g_host;
  for(int l=0;l<g_nlink;++l){
    ibv_device* dev=nullptr;for(int i=0;i<n;++i)if(!strcmp(ibv_get_device_name(list[i]),wants[l]))dev=list[i];
    if(!dev){fprintf(stderr,"[rdma_ar] device %s not found\n",wants[l]);return 4;}
    g_ctx[l]=ibv_open_device(dev);if(!g_ctx[l])return 5;
    g_pd[l]=ibv_alloc_pd(g_ctx[l]);g_cq[l]=ibv_create_cq(g_ctx[l],64,nullptr,nullptr,0);if(!g_pd[l]||!g_cq[l])return 6;
    g_mr[l]=ibv_reg_mr(g_pd[l],g_host,sizeof(Host),IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);if(!g_mr[l]){perror("rdma_ar reg_mr");return 7;}
    ibv_qp_init_attr qa{};qa.send_cq=g_cq[l];qa.recv_cq=g_cq[l];qa.qp_type=IBV_QPT_RC;qa.cap.max_send_wr=64;qa.cap.max_recv_wr=1;qa.cap.max_send_sge=1;qa.cap.max_recv_sge=1;qa.cap.max_inline_data=64;
    g_qp[l]=ibv_create_qp(g_pd[l],&qa);if(!g_qp[l])return 8;
    ibv_gid gid;if(ibv_query_gid(g_ctx[l],1,gid_index,&gid))return 9;
    mine.qpn[l]=g_qp[l]->qp_num;mine.psn[l]=0;mine.rkey[l]=g_mr[l]->rkey;memcpy(mine.gid[l],gid.raw,16);
  }
  ibv_free_device_list(list);
  if(!sock_exchange(mine,g_peer)){fprintf(stderr,"[rdma_ar] exchange failed\n");return 10;}
  if(g_peer.nlink!=mine.nlink){fprintf(stderr,"[rdma_ar] link count differs between ranks (%u vs %u): set GLM53_RDMA_AR_DUAL alike\n",mine.nlink,g_peer.nlink);return 12;}
  for(int l=0;l<g_nlink;++l)if(!qp_to_rts(l)){fprintf(stderr,"[rdma_ar] QP %d transition failed\n",l);return 11;}
  g_proxy=std::thread(proxy_loop);
  {// Pin the spinning proxy to one big core (default cpu19); -1 leaves it unpinned.
   const char* c=std::getenv("GLM53_RDMA_AR_CPU");const int cpu=c?std::atoi(c):19;
   pthread_setname_np(g_proxy.native_handle(),"glm53-rdma-ar");
   if(cpu>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);
     if(pthread_setaffinity_np(g_proxy.native_handle(),sizeof(set),&set))fprintf(stderr,"[rdma_ar] affinity cpu%d failed\n",cpu);}}
  g_proxy.detach();
  if(const char* sp=std::getenv("GLM53_AR_STATS")){
    if(glm53_host_alloc_mapped((void**)&g_stats,sizeof(Stamp)*STAT_N)==cudaSuccess&&
       cudaHostGetDevicePointer((void**)&g_stats_dev,g_stats,0)==cudaSuccess){
      memset(g_stats,0,sizeof(Stamp)*STAT_N);
      std::string path=std::string(sp)+".rank"+std::to_string(rank);
      std::thread([path]{for(;;){std::this_thread::sleep_for(std::chrono::seconds(5));
        if(FILE* f=fopen((path+".tmp").c_str(),"wb")){fwrite(g_stats,sizeof(Stamp),STAT_N,f);fclose(f);rename((path+".tmp").c_str(),path.c_str());}}}).detach();
      fprintf(stderr,"[rdma_ar] rank%d allreduce stamps -> %s\n",rank,path.c_str());
    }else{g_stats=nullptr;g_stats_dev=nullptr;}
  }
  fprintf(stderr,"[rdma_ar] rank%d ready: dev=%s%s%s gid_index=%d slots=%d x %zu KiB split>=%llu B\n",rank,want,g_nlink==2?" + ":"",g_nlink==2?wants[1]:"",gid_index,NSLOT,SLOT_BYTES/1024,g_split);
  return 0;
}
extern "C" size_t glm53_rdma_ar_max_bytes(){return SLOT_BYTES;}
extern "C" int glm53_rdma_ar_cuda(float* data,long long n,cudaStream_t s){
  if(!g_host||n<1||(size_t)n*4>SLOT_BYTES)return 1;
  int blocks=(int)((n+1023)/1024);if(blocks>32)blocks=32;if(blocks<1)blocks=1;
  ar_kernel<true><<<blocks,256,0,s>>>(data,n,g_rank,g_host_dev,g_state,g_stats_dev,g_split);
  return (int)cudaGetLastError();
}
extern "C" int glm53_rdma_ar_send_cuda(float* data,long long n,cudaStream_t s){
  if(!g_host||n<1||(size_t)n*4>SLOT_BYTES)return 1;
  int blocks=(int)((n+1023)/1024);if(blocks>32)blocks=32;if(blocks<1)blocks=1;
  ar_kernel<false><<<blocks,256,0,s>>>(data,n,g_rank,g_host_dev,g_state,g_stats_dev,g_split);
  return (int)cudaGetLastError();
}
// Fallback consumer of a send-only allreduce: data = rank0+rank1 in place (optionally Half-rounded, round_row_output).
__global__ void ar_materialize_kernel(float* data,long long n,const Host* h,const DevState* st,int round){
  const int slot=(int)(st->seq%NSLOT);const float* rcv=reinterpret_cast<const float*>(h->recv[slot]);
  for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
    float v=__fadd_rn(data[i],__ldcv(rcv+i));if(round)v=__half2float(__float2half_rn(v));data[i]=v;
  }
}
extern "C" int glm53_rdma_ar_materialize_cuda(float* data,long long n,int round,cudaStream_t s){
  if(!g_host||n<1||(size_t)n*4>SLOT_BYTES)return 1;
  int blocks=(int)((n+1023)/1024);if(blocks>32)blocks=32;
  ar_materialize_kernel<<<blocks,256,0,s>>>(data,n,g_host_dev,g_state,round);
  return (int)cudaGetLastError();
}
// Device view of the last completed allreduce's peer data: recv slot (seq % NSLOT) of recv_base, seq read on device
// (graph replays advance it). Consumers must run before this rank's next allreduce.
extern "C" int glm53_rdma_ar_peer(const float** recv_base,const unsigned long long** seq,int* slot_floats,int* nslot){
  if(!g_host)return 1;
  *recv_base=reinterpret_cast<const float*>(g_host_dev->recv[0]);*seq=&g_state->seq;*slot_floats=(int)(SLOT_BYTES/4);*nslot=NSLOT;return 0;
}
