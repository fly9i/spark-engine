// SPDX-License-Identifier: MIT
// Growable KV pool backing (GLM53_KV_VMM=1, crate::kv_pool): CUDA virtual memory management. Every region (one per
// store slot and per-token tensor) reserves the virtual range of its largest size once, so its address never changes
// and captured graphs keep pointing at it; physical chunks from one shared pool are mapped at the end of a region as it
// grows and unmapped (back to the pool) as it shrinks. Physical chunks are created once at init (cuMemCreate is the
// slow part, ~30 ms/GiB on GB10); map + access cost ~8 ms/GiB. Contents of a newly mapped chunk are undefined: the
// caller zero-fills rows it hands out.
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
struct Region { CUdeviceptr base = 0; size_t reserved = 0; std::vector<CUmemGenericAllocationHandle> mapped; };
std::vector<CUmemGenericAllocationHandle> g_free;
std::vector<Region> g_regions;
size_t g_chunk = 0;
CUmemAccessDesc g_access{};
int fail(CUresult r, const char* what) {
    const char* s = nullptr; cuGetErrorString(r, &s);
    fprintf(stderr, "[vmm] %s: %s\n", what, s ? s : "?");
    return (int) r;
}
}

// Set up for device `dev`; `*chunk_bytes` is the chunk size (the allocation granularity rounded up to
// `min_chunk_bytes`). Chunks are created by rs_vmm_create.
CUmemAllocationProp g_prop{};
extern "C" int rs_vmm_init(int dev, unsigned long long min_chunk_bytes, unsigned long long* chunk_bytes) {
    if (g_chunk) return 1;
    CUresult r = cuInit(0); if (r) return fail(r, "cuInit");
    g_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    g_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; g_prop.location.id = dev;
    size_t gran = 0; r = cuMemGetAllocationGranularity(&gran, &g_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    if (r) return fail(r, "granularity");
    size_t ch = (min_chunk_bytes + gran - 1) / gran * gran; if (ch == 0) ch = gran;
    g_access.location = g_prop.location; g_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    g_chunk = ch; *chunk_bytes = ch;
    return 0;
}
// Add `chunks` physical chunks to the free pool.
extern "C" int rs_vmm_create(long long chunks) {
    if (!g_chunk) return 1;
    for (long long i = 0; i < chunks; ++i) {
        CUmemGenericAllocationHandle h; CUresult r = cuMemCreate(&h, g_chunk, &g_prop, 0);
        if (r) return fail(r, "cuMemCreate");
        g_free.push_back(h);
    }
    return 0;
}

// Reserve a region of at most `bytes` (rounded up to whole chunks). Returns its id (>= 0) or -1; `*base` its address.
extern "C" long long rs_vmm_region(unsigned long long bytes, unsigned long long* base) {
    if (!g_chunk) return -1;
    Region rg; rg.reserved = (bytes + g_chunk - 1) / g_chunk * g_chunk;
    CUresult r = cuMemAddressReserve(&rg.base, rg.reserved, 0, 0, 0);
    if (r) { fail(r, "cuMemAddressReserve"); return -1; }
    *base = rg.base; g_regions.push_back(rg);
    return (long long) g_regions.size() - 1;
}

// Map or unmap chunks at the end of region `id` so that exactly ceil(bytes / chunk) are mapped. Returns 0, or 2 when
// the pool has too few free chunks (nothing changed).
extern "C" int rs_vmm_resize(long long id, unsigned long long bytes) {
    if (id < 0 || id >= (long long) g_regions.size()) return 1;
    Region& rg = g_regions[id];
    size_t want = (bytes + g_chunk - 1) / g_chunk;
    if (want * g_chunk > rg.reserved) return 1;
    if (want > rg.mapped.size() && want - rg.mapped.size() > g_free.size()) return 2;
    while (rg.mapped.size() < want) {
        CUdeviceptr at = rg.base + rg.mapped.size() * g_chunk;
        CUmemGenericAllocationHandle h = g_free.back();
        CUresult r = cuMemMap(at, g_chunk, 0, h, 0); if (r) return fail(r, "cuMemMap");
        r = cuMemSetAccess(at, g_chunk, &g_access, 1); if (r) { cuMemUnmap(at, g_chunk); return fail(r, "cuMemSetAccess"); }
        g_free.pop_back(); rg.mapped.push_back(h);
    }
    while (rg.mapped.size() > want) {
        CUdeviceptr at = rg.base + (rg.mapped.size() - 1) * g_chunk;
        CUresult r = cuMemUnmap(at, g_chunk); if (r) return fail(r, "cuMemUnmap");
        g_free.push_back(rg.mapped.back()); rg.mapped.pop_back();
    }
    return 0;
}

extern "C" long long rs_vmm_free_chunks() { return (long long) g_free.size(); }
extern "C" long long rs_vmm_mapped_chunks(long long id) { return id >= 0 && id < (long long) g_regions.size() ? (long long) g_regions[id].mapped.size() : -1; }
