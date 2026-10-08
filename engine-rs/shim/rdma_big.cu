// SPDX-License-Identifier: MIT
// P5: TP2 large-message all_gather / reduce_scatter over BOTH RoCE ports (GLM53_RDMA_BIG=1).
//
// Replaces NCCL's host-proxied 16-64 MiB SP-prefill collectives. Data path per op (both ranks, same
// stream order): one GPU kernel copies this rank's outgoing rows chunk by chunk (1 MiB) into a pinned,
// NIC-registered send area and publishes ready_seq[chunk]; a CPU proxy thread posts, per chunk, an
// RDMA WRITE of the payload followed by an 8-byte flag WRITE on the queue pair of port (chunk % 2)
// (same-QP writes are placed in order). The same kernel then spins on each incoming chunk's flag and
// consumes it straight from the pinned receive area: all_gather copies it to the output (pure copy),
// reduce_scatter adds it to the local half as x_rank0 + x_rank1 (commutative: bitwise equal to the
// 2-rank NCCL sum).
//
// Buffer reuse without credits: op k uses area (k % 2). A rank reaches op k+2 only after consuming the
// peer's op k+1 data, which the peer sends only after its op k kernel completed (so it consumed our
// op k data, and our NIC finished reading our op k send area). Hence both the peer's receive area and
// our send area of parity k%2 are free again when op k+2 starts.
#include <pthread.h>
#include <cuda_runtime.h>
#include "host_pin.cuh"
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {
constexpr size_t CHUNK=1u<<20;
constexpr int MAX_CHUNKS=64;                    // 64 MiB per op and direction (8192-row FP32 halves)
constexpr size_t AREA=CHUNK*MAX_CHUNKS;
constexpr int NPORT=2;
struct Ctl {                                   // pinned, host-mapped, registered on both HCAs
  alignas(64) volatile unsigned long long op_seq[2];      // GPU -> proxy: op seq published per parity
  alignas(64) volatile unsigned long long op_chunks[2];
  alignas(64) volatile unsigned long long op_last_bytes[2];
  alignas(64) volatile unsigned long long ready_seq[2][MAX_CHUNKS];  // GPU -> proxy per chunk
  alignas(64) volatile unsigned long long recv_flag[2][MAX_CHUNKS];  // written by the peer's NIC
  alignas(64) unsigned long long flag_src[2][MAX_CHUNKS];
};
struct Bufs {unsigned char send[2][AREA];unsigned char recv[2][AREA];};
constexpr int PARTS=8;                          // blocks per 1 MiB chunk (one block alone copies only ~4 GB/s)
struct DevState {unsigned long long seq;unsigned int arrive;unsigned int done;unsigned int parts[2][MAX_CHUNKS];};
struct Peer {uint32_t qpn[NPORT],psn[NPORT],rkey_ctl[NPORT],rkey_buf[NPORT];uint64_t ctl,buf;uint8_t gid[NPORT][16];};

Ctl* g_ctl=nullptr;Ctl* g_ctl_dev=nullptr;Bufs* g_buf=nullptr;Bufs* g_buf_dev=nullptr;DevState* g_state=nullptr;int g_rank=-1;
ibv_context* g_ctx[NPORT]{};ibv_pd* g_pd[NPORT]{};ibv_cq* g_cq[NPORT]{};ibv_qp* g_qp[NPORT]{};ibv_mr* g_mr_ctl[NPORT]{};ibv_mr* g_mr_buf[NPORT]{};
Peer g_peer{};std::atomic<bool> g_stop{false};unsigned long long g_host_seq=0;

// mode 0: all_gather (out[own]=local, out[peer]=recv); mode 1: reduce_scatter FP32 (out=own+peer).
// src_send: rows to send (all_gather: local; reduce_scatter: the peer-owned half of the input).
// src_own:  all_gather: local (copied to out_own); reduce_scatter: this rank's half of the input.
__global__ void big_kernel(int mode,const unsigned char* src_send,const unsigned char* src_own,unsigned char* out_own,unsigned char* out_peer,
    unsigned long long bytes,int rank,Ctl* c,Bufs* b,DevState* st){
  const unsigned long long s=st->seq+1;const int par=(int)(s&1);
  const int chunks=(int)((bytes+CHUNK-1)/CHUNK);
  if(blockIdx.x==0&&threadIdx.x==0){c->op_chunks[par]=chunks;c->op_last_bytes[par]=bytes-(unsigned long long)(chunks-1)*CHUNK;__threadfence_system();c->op_seq[par]=s;__threadfence_system();}
  // Work item w = (chunk, part): PARTS blocks share each 1 MiB chunk; the last part to finish publishes it.
  const int items=chunks*PARTS;
  // Phase 1: outgoing parts -> send area (+ all_gather's own rows to the output).
  for(int w=blockIdx.x;w<items;w+=gridDim.x){
    const int ch=w/PARTS,part=w%PARTS;
    const unsigned long long off=(unsigned long long)ch*CHUNK,n=min((unsigned long long)CHUNK,bytes-off);
    const unsigned long long n16=n/16,lo=n16*part/PARTS,hi=n16*(part+1)/PARTS;
    const uint4* in=reinterpret_cast<const uint4*>(src_send+off);uint4* snd=reinterpret_cast<uint4*>(b->send[par]+off);
    for(unsigned long long i=lo+threadIdx.x;i<hi;i+=blockDim.x){uint4 v=in[i];snd[i]=v;}
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0 && atomicAdd(&st->parts[par][ch],1u)==PARTS-1){st->parts[par][ch]=0;__threadfence_system();c->ready_seq[par][ch]=s;}
    if(mode==0){const uint4* own=reinterpret_cast<const uint4*>(src_own+off);uint4* o=reinterpret_cast<uint4*>(out_own+off);
      for(unsigned long long i=lo+threadIdx.x;i<hi;i+=blockDim.x)o[i]=own[i];}
  }
  // Phase 2: incoming parts, consumed as their chunk's flag arrives.
  for(int w=blockIdx.x;w<items;w+=gridDim.x){
    const int ch=w/PARTS,part=w%PARTS;
    const unsigned long long off=(unsigned long long)ch*CHUNK,n=min((unsigned long long)CHUNK,bytes-off);
    const unsigned long long n16=n/16,lo=n16*part/PARTS,hi=n16*(part+1)/PARTS;
    if(threadIdx.x==0){long long spins=0;while(c->recv_flag[par][ch]!=s){if(++spins>(1ll<<34))__trap();}}
    __syncthreads();
    if(mode==0){
      const uint4* r=reinterpret_cast<const uint4*>(b->recv[par]+off);uint4* o=reinterpret_cast<uint4*>(out_peer+off);
      for(unsigned long long i=lo+threadIdx.x;i<hi;i+=blockDim.x)o[i]=__ldcv(r+i);
    }else{
      const float4* r=reinterpret_cast<const float4*>(b->recv[par]+off);const float4* own=reinterpret_cast<const float4*>(src_own+off);
      float4* o=reinterpret_cast<float4*>(out_own+off);
      for(unsigned long long i=lo+threadIdx.x;i<hi;i+=blockDim.x){
        const float4 m=own[i],p=__ldcv(r+i);
        o[i]=rank==0?make_float4(m.x+p.x,m.y+p.y,m.z+p.z,m.w+p.w):make_float4(p.x+m.x,p.y+m.y,p.z+m.z,p.w+m.w);
      }
    }
  }
  __threadfence();__syncthreads();
  if(threadIdx.x==0&&atomicAdd(&st->done,1u)==gridDim.x-1){st->done=0;st->seq=s;}
}

bool sock_exchange(const Peer& mine,Peer& theirs){
  const char* addr=std::getenv("GLM53_MASTER_ADDR");const char* port_s=std::getenv("GLM53_MASTER_PORT");
  if(!addr||!port_s)return false;int port=std::atoi(port_s)+19;int fd=-1;
  if(g_rank==0){
    int ls=socket(AF_INET,SOCK_STREAM,0);int one=1;setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one);
    sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(port);a.sin_addr.s_addr=INADDR_ANY;
    if(bind(ls,(sockaddr*)&a,sizeof a)||listen(ls,1)){perror("rdma_big bind");return false;}
    fd=accept(ls,nullptr,nullptr);close(ls);
  }else{
    for(int t=0;t<600&&fd<0;++t){int s=socket(AF_INET,SOCK_STREAM,0);sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(port);inet_pton(AF_INET,addr,&a.sin_addr);
      if(connect(s,(sockaddr*)&a,sizeof a)==0){fd=s;break;}close(s);usleep(100000);}
  }
  if(fd<0)return false;
  bool ok=write(fd,&mine,sizeof mine)==(ssize_t)sizeof mine&&read(fd,&theirs,sizeof theirs)==(ssize_t)sizeof theirs;
  char b=0;ok=ok&&write(fd,&b,1)==1&&read(fd,&b,1)==1;close(fd);return ok;
}

void post(int p,const void* local,uint32_t lkey,size_t bytes,uint64_t remote,uint32_t rkey,bool signaled){
  ibv_sge sge{};sge.addr=(uintptr_t)local;sge.length=(uint32_t)bytes;sge.lkey=lkey;
  ibv_send_wr wr{},*bad=nullptr;wr.opcode=IBV_WR_RDMA_WRITE;wr.sg_list=&sge;wr.num_sge=1;
  wr.send_flags=(signaled?IBV_SEND_SIGNALED:0)|(bytes<=64?IBV_SEND_INLINE:0);
  wr.wr.rdma.remote_addr=remote;wr.wr.rdma.rkey=rkey;
  if(ibv_post_send(g_qp[p],&wr,&bad)){fprintf(stderr,"[rdma_big] post_send failed\n");std::abort();}
}

void proxy_loop(){
  unsigned long long s=1;int outstanding[NPORT]={0,0};
  auto reap=[&](int p,int keep){while(outstanding[p]>keep){ibv_wc wc[16];int got=ibv_poll_cq(g_cq[p],16,wc);
      if(got<0){fprintf(stderr,"[rdma_big] poll error\n");std::abort();}
      for(int i=0;i<got;++i)if(wc[i].status!=IBV_WC_SUCCESS){fprintf(stderr,"[rdma_big] completion error %d\n",(int)wc[i].status);std::abort();}
      outstanding[p]-=got;}};
  while(!g_stop.load(std::memory_order_relaxed)){
    const int par=(int)(s&1);
    if(g_ctl->op_seq[par]!=s){asm volatile("yield");continue;}
    std::atomic_thread_fence(std::memory_order_acquire);
    const int chunks=(int)g_ctl->op_chunks[par];const size_t last=(size_t)g_ctl->op_last_bytes[par];
    for(int ch=0;ch<chunks;++ch){
      while(g_ctl->ready_seq[par][ch]!=s){asm volatile("yield");}
      std::atomic_thread_fence(std::memory_order_acquire);
      const int p=ch%NPORT;const size_t n=ch==chunks-1?last:CHUNK;const size_t off=(size_t)ch*CHUNK;
      post(p,g_buf->send[par]+off,g_mr_buf[p]->lkey,n,g_peer.buf+offsetof(Bufs,recv)+(size_t)par*AREA+off,g_peer.rkey_buf[p],false);
      g_ctl->flag_src[par][ch]=s;
      post(p,&g_ctl->flag_src[par][ch],g_mr_ctl[p]->lkey,8,g_peer.ctl+offsetof(Ctl,recv_flag)+((size_t)par*MAX_CHUNKS+ch)*8,g_peer.rkey_ctl[p],true);
      ++outstanding[p];reap(p,24);
    }
    reap(0,0);reap(1,0);
    ++s;
  }
}

bool qp_rts(int p,int gid_index){
  ibv_qp_attr a{};a.qp_state=IBV_QPS_INIT;a.pkey_index=0;a.port_num=1;a.qp_access_flags=IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE;
  if(ibv_modify_qp(g_qp[p],&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))return false;
  ibv_qp_attr r{};r.qp_state=IBV_QPS_RTR;r.path_mtu=IBV_MTU_4096;r.dest_qp_num=g_peer.qpn[p];r.rq_psn=g_peer.psn[p];
  r.max_dest_rd_atomic=1;r.min_rnr_timer=12;r.ah_attr.is_global=1;r.ah_attr.port_num=1;
  memcpy(r.ah_attr.grh.dgid.raw,g_peer.gid[p],16);r.ah_attr.grh.sgid_index=gid_index;r.ah_attr.grh.hop_limit=1;
  if(ibv_modify_qp(g_qp[p],&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER))return false;
  ibv_qp_attr t{};t.qp_state=IBV_QPS_RTS;t.timeout=14;t.retry_cnt=7;t.rnr_retry=7;t.sq_psn=0;t.max_rd_atomic=1;
  return ibv_modify_qp(g_qp[p],&t,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC)==0;
}
} // namespace

extern "C" int glm53_rdma_big_init(int rank){
  if(g_ctl)return 0;g_rank=rank;
  if(glm53_host_alloc_mapped((void**)&g_ctl,sizeof(Ctl))!=cudaSuccess)return 1;
  memset((void*)g_ctl,0,sizeof(Ctl));
  if(glm53_host_alloc_mapped((void**)&g_buf,sizeof(Bufs))!=cudaSuccess)return 1;
  if(cudaHostGetDevicePointer((void**)&g_ctl_dev,g_ctl,0)!=cudaSuccess||cudaHostGetDevicePointer((void**)&g_buf_dev,g_buf,0)!=cudaSuccess)return 2;
  if(cudaMalloc(&g_state,sizeof(DevState))!=cudaSuccess)return 3;cudaMemset(g_state,0,sizeof(DevState));
  const char* d0=std::getenv("GLM53_RDMA_BIG_DEV0");const char* d1=std::getenv("GLM53_RDMA_BIG_DEV1");
  const char* want[NPORT]={d0?d0:"rocep1s0f0",d1?d1:"roceP2p1s0f0"};
  const char* gid_s=std::getenv("GLM53_RDMA_AR_GID");int gid_index=gid_s?std::atoi(gid_s):3;
  int n=0;ibv_device** list=ibv_get_device_list(&n);
  Peer mine{};mine.ctl=(uint64_t)(uintptr_t)g_ctl;mine.buf=(uint64_t)(uintptr_t)g_buf;
  for(int p=0;p<NPORT;++p){
    ibv_device* dev=nullptr;for(int i=0;i<n;++i)if(!strcmp(ibv_get_device_name(list[i]),want[p]))dev=list[i];
    if(!dev){fprintf(stderr,"[rdma_big] device %s not found\n",want[p]);return 4;}
    g_ctx[p]=ibv_open_device(dev);if(!g_ctx[p])return 5;
    g_pd[p]=ibv_alloc_pd(g_ctx[p]);g_cq[p]=ibv_create_cq(g_ctx[p],256,nullptr,nullptr,0);if(!g_pd[p]||!g_cq[p])return 6;
    g_mr_ctl[p]=ibv_reg_mr(g_pd[p],g_ctl,sizeof(Ctl),IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);
    g_mr_buf[p]=ibv_reg_mr(g_pd[p],g_buf,sizeof(Bufs),IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE);
    if(!g_mr_ctl[p]||!g_mr_buf[p]){perror("rdma_big reg_mr");return 7;}
    ibv_qp_init_attr qa{};qa.send_cq=g_cq[p];qa.recv_cq=g_cq[p];qa.qp_type=IBV_QPT_RC;qa.cap.max_send_wr=128;qa.cap.max_recv_wr=1;
    qa.cap.max_send_sge=1;qa.cap.max_recv_sge=1;qa.cap.max_inline_data=64;
    g_qp[p]=ibv_create_qp(g_pd[p],&qa);if(!g_qp[p])return 8;
    ibv_gid gid;if(ibv_query_gid(g_ctx[p],1,gid_index,&gid))return 9;
    mine.qpn[p]=g_qp[p]->qp_num;mine.psn[p]=0;mine.rkey_ctl[p]=g_mr_ctl[p]->rkey;mine.rkey_buf[p]=g_mr_buf[p]->rkey;memcpy(mine.gid[p],gid.raw,16);
  }
  ibv_free_device_list(list);
  if(!sock_exchange(mine,g_peer)){fprintf(stderr,"[rdma_big] exchange failed\n");return 10;}
  for(int p=0;p<NPORT;++p)if(!qp_rts(p,gid_index)){fprintf(stderr,"[rdma_big] QP %d transition failed\n",p);return 11;}
  std::thread t(proxy_loop);
  {const char* c=std::getenv("GLM53_RDMA_BIG_CPU");const int cpu=c?std::atoi(c):18;
   pthread_setname_np(t.native_handle(),"glm53-rdma-big");
   if(cpu>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);if(pthread_setaffinity_np(t.native_handle(),sizeof(set),&set))fprintf(stderr,"[rdma_big] affinity cpu%d failed\n",cpu);}}
  t.detach();
  fprintf(stderr,"[rdma_big] rank%d ready: %s + %s, gid %d, %d x %zu KiB per op/direction, pinned %zu MiB\n",rank,want[0],want[1],gid_index,MAX_CHUNKS,CHUNK/1024,(sizeof(Bufs)+sizeof(Ctl))>>20);
  return 0;
}
extern "C" unsigned long long glm53_rdma_big_max_bytes(){return AREA;}
// all_gather: in [bytes] (this rank's rows) -> out [2*bytes] (rank 0 rows first).
extern "C" int glm53_rdma_big_all_gather(const void* in,void* out,unsigned long long bytes,cudaStream_t s){
  if(!g_ctl||bytes==0||bytes>AREA||bytes%16)return 1;
  unsigned char* o=(unsigned char*)out;unsigned char* own=o+(g_rank==0?0:bytes);unsigned char* peer=o+(g_rank==0?bytes:0);
  const int chunks=(int)((bytes+CHUNK-1)/CHUNK);
  big_kernel<<<chunks*PARTS<128?chunks*PARTS:128,256,0,s>>>(0,(const unsigned char*)in,(const unsigned char*)in,own,peer,bytes,g_rank,g_ctl_dev,g_buf_dev,g_state);
  return (int)cudaGetLastError();
}
// reduce_scatter FP32: in [2*bytes] -> out [bytes] = rank's half of (x_rank0 + x_rank1).
extern "C" int glm53_rdma_big_reduce_scatter(const void* in,void* out,unsigned long long bytes,cudaStream_t s){
  if(!g_ctl||bytes==0||bytes>AREA||bytes%16)return 1;
  const unsigned char* i=(const unsigned char*)in;const unsigned char* own=i+(g_rank==0?0:bytes);const unsigned char* send=i+(g_rank==0?bytes:0);
  const int chunks=(int)((bytes+CHUNK-1)/CHUNK);
  big_kernel<<<chunks*PARTS<128?chunks*PARTS:128,256,0,s>>>(1,send,own,(unsigned char*)out,nullptr,bytes,g_rank,g_ctl_dev,g_buf_dev,g_state);
  return (int)cudaGetLastError();
}

// Diagnostic: bandwidth of the staging copies big_kernel performs (same block/thread shape, uint4).
__global__ void bw_copy(const uint4* __restrict__ src,uint4* __restrict__ dst,unsigned long long n16,int volatile_src){
  for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<n16;i+=(unsigned long long)gridDim.x*blockDim.x)
    dst[i]=volatile_src?__ldcv(src+i):src[i];
}
// Pinned (mapped) host buffer -> device with SM loads: on GB10 the host buffer is the same DRAM, read at
// device-memory speed (see the probe below), and the copy engine stays free for other transfers.
extern "C" int rs_pinned_to_device(void* dst,const void* host_src,unsigned long long bytes,void* stream){
  if(bytes%16)return 1;
  void* src=nullptr;
  if(cudaHostGetDevicePointer(&src,const_cast<void*>(host_src),0)!=cudaSuccess)return 2;
  bw_copy<<<192,256,0,(cudaStream_t)stream>>>((const uint4*)src,(uint4*)dst,bytes/16,0);
  return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" int glm53_pinned_bw_probe(){
  const size_t bytes=64u<<20;unsigned char *pin=nullptr,*pin_dev=nullptr,*d0=nullptr,*d1=nullptr;
  if(cudaHostAlloc((void**)&pin,bytes,cudaHostAllocMapped|cudaHostAllocPortable)!=cudaSuccess)return 1;
  cudaHostGetDevicePointer((void**)&pin_dev,pin,0);cudaMalloc(&d0,bytes);cudaMalloc(&d1,bytes);
  cudaMemset(d0,1,bytes);memset(pin,2,bytes);
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
  struct Case{const char* name;const void* s;void* d;int vol;int blocks;};
  Case cases[]={{"dev->dev",d0,d1,0,32},{"dev->pinned",d0,pin_dev,0,32},{"pinned->dev",pin_dev,d0,0,32},{"pinned->dev (ldcv)",pin_dev,d0,1,32},
                {"dev->pinned 192 blk",d0,pin_dev,0,192},{"pinned->dev ldcv 192 blk",pin_dev,d0,1,192},{"dev->dev 192 blk",d0,d1,0,192}};
  for(auto& c:cases){
    for(int w=0;w<3;++w)bw_copy<<<c.blocks,256>>>((const uint4*)c.s,(uint4*)c.d,bytes/16,c.vol);
    cudaEventRecord(a);for(int r=0;r<10;++r)bw_copy<<<c.blocks,256>>>((const uint4*)c.s,(uint4*)c.d,bytes/16,c.vol);cudaEventRecord(b);cudaEventSynchronize(b);
    float ms=0;cudaEventElapsedTime(&ms,a,b);
    printf("pinned-bw %-26s %7.1f GB/s (read+write %zu MiB x10 in %.2f ms)\n",c.name,2.0*bytes*10/(ms*1e-3)/1e9,bytes>>20,ms);
  }
  cudaFree(d0);cudaFree(d1);cudaFreeHost(pin);return (int)cudaGetLastError();
}
