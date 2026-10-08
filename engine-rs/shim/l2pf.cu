// Item 5 feasibility: L2 prefetch of the next operator's weights while the stream waits in an allreduce.
// spin_us: one block spins for `ns` nanoseconds (stands in for the RDMA allreduce peer wait).
// l2_prefetch: cp.async.bulk.prefetch.L2 of [ptr, ptr+bytes) in 64 KiB pieces, one thread per piece.
#include <cuda_runtime.h>
#include <cstdint>
__global__ void spin_ns(long long ns){
  long long start; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  while(true){long long now; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); if(now-start>=ns)break;}
}
__global__ void l2_prefetch(const char* p,long long bytes){
  const long long piece=65536; long long off=((long long)blockIdx.x*blockDim.x+threadIdx.x)*piece;
  if(off>=bytes)return; long long n=bytes-off<piece?bytes-off:piece; n&=~15LL; if(n<=0)return;
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p+off), "r"((unsigned)n) : "memory");
}
extern "C" int glm53_spin_ns_cuda(long long ns,cudaStream_t s){spin_ns<<<1,32,0,s>>>(ns);return int(cudaGetLastError());}
extern "C" int glm53_l2_prefetch_cuda(const void* p,long long bytes,cudaStream_t s){
  if(bytes<=0)return 0; long long pieces=(bytes+65535)/65536; int threads=128; int blocks=(int)((pieces+threads-1)/threads);
  l2_prefetch<<<blocks,threads,0,s>>>((const char*)p,bytes);return int(cudaGetLastError());
}
