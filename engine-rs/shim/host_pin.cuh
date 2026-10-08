// Pinned, GPU-mapped host memory that kernel compaction leaves alone.
// cudaHostAlloc memory is a shared /dev/zero (shmem) mapping. When the node is fragmented, proactive compaction
// (kcompactd, vm.compaction_proactiveness) isolates those pages, replaces their PTEs with migration entries (one MMU
// notifier invalidation per page), fails to migrate them because they are pinned, and restores them: every ~34 s on
// rank0, ~2.3 s bursts over ~1 GB of the engine's pinned pages, during which decode rounds slow down (bench/w78b).
// Anonymous private pages pinned with cudaHostRegister fail compaction's early check for pinned anonymous pages
// (reference count above the map count) and are never isolated or unmapped.
// GLM53_HOST_REGISTER=1 selects mmap + cudaHostRegister; otherwise cudaHostAlloc as before. Same flags semantics
// (mapped + portable), page-aligned, zero-filled.
#pragma once
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

static inline bool glm53_host_register_enabled(){
  const char* e=std::getenv("GLM53_HOST_REGISTER");return e&&!std::strcmp(e,"1");
}
static inline cudaError_t glm53_host_alloc_mapped(void** p,size_t bytes){
  if(!glm53_host_register_enabled())return cudaHostAlloc(p,bytes,cudaHostAllocMapped|cudaHostAllocPortable);
  const size_t page=(size_t)sysconf(_SC_PAGESIZE);const size_t len=(bytes+page-1)/page*page;
  void* m=mmap(nullptr,len,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE,-1,0);
  if(m==MAP_FAILED)return cudaErrorMemoryAllocation;
  madvise(m,len,MADV_DONTFORK);   // RDMA-registered: a fork must not make these pages copy-on-write
  const cudaError_t e=cudaHostRegister(m,len,cudaHostRegisterMapped|cudaHostRegisterPortable);
  if(e!=cudaSuccess){munmap(m,len);return e;}
  *p=m;return cudaSuccess;
}
