// extern "C" wrappers so Rust links the pinned-host allocator in host_pin.cuh.
#include "host_pin.cuh"
extern "C" int  glm53_host_pin_alloc(void** p, size_t bytes){ return (int)glm53_host_alloc_mapped(p, bytes); }
extern "C" void glm53_host_pin_free (void* p, size_t bytes){ glm53_host_free_mapped(p, bytes); }
