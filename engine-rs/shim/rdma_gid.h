// SPDX-License-Identifier: MIT
// The GID index of a RoCE device's port 1 that holds its IPv4-mapped RoCE v2 address, shared by the RDMA transports
// (rdma_ar, rdma_big, rdma_arn). The index is not the same on every node (one node 3, another 4 -- that node's
// index 3 is empty), so it is looked up instead of fixed.
#pragma once
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// `ip` (network order) the address to match, 0: the first IPv4-mapped RoCE v2 entry; `env` names an override variable
// (its value is the index). Returns -1 when there is none.
inline int rdma_ipv4_v2_gid(ibv_context* ctx, const char* dev_name, uint32_t ip, const char* env) {
  if (env) if (const char* g = std::getenv(env)) return std::atoi(g);
  static const uint8_t v4[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
  for (int i = 0; i < 64; ++i) {
    ibv_gid gid;
    if (ibv_query_gid(ctx, 1, i, &gid)) break;
    if (memcmp(gid.raw, v4, 12)) continue;
    if (ip ? memcmp(gid.raw + 12, &ip, 4) != 0 : (gid.raw[12] | gid.raw[13] | gid.raw[14] | gid.raw[15]) == 0) continue;
    char path[256], type[32] = {};
    snprintf(path, sizeof path, "/sys/class/infiniband/%s/ports/1/gid_attrs/types/%d", dev_name, i);
    if (FILE* f = fopen(path, "r")) { if (fgets(type, sizeof type, f)) {} fclose(f); }
    if (strstr(type, "v2")) return i;
  }
  return -1;
}
