// SPDX-License-Identifier: MIT
// Shared by the RDMA transports (rdma_ar, rdma_big, rdma_arn):
//  - the GID index of a RoCE device's port 1 that holds its IPv4-mapped RoCE v2 address. The index is not the same
//    on every node (one node 3, another 4 -- that node's index 3 is empty), so it is looked up instead of fixed.
//  - the RC path MTU. RoCE reports the port active MTU from the netdev MTU (1500 -> IBV_MTU_1024, jumbo -> 4096);
//    mlx5 rejects ibv_modify_qp(RTR) with EINVAL when path_mtu is larger, so it is queried and clamped to 4096.
#pragma once
#include <infiniband/verbs.h>
#include <cstdint>
#include <cerrno>
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

static_assert(IBV_MTU_256 < IBV_MTU_512 && IBV_MTU_512 < IBV_MTU_1024 && IBV_MTU_1024 < IBV_MTU_2048 &&
              IBV_MTU_2048 < IBV_MTU_4096, "ibv_mtu enumerators increase with the payload size");

// Payload bytes of an ibv_mtu enumerator. 0 when `m` is not one of them.
inline int rdma_mtu_bytes(int m) {
  switch (m) {
    case IBV_MTU_256: return 256;
    case IBV_MTU_512: return 512;
    case IBV_MTU_1024: return 1024;
    case IBV_MTU_2048: return 2048;
    case IBV_MTU_4096: return 4096;
    default: return 0;
  }
}

// `active` is a port's active_mtu enumerator. Outside 256..4096 it stays at 4096, which is what these QPs used
// before the port was queried. A jumbo-frame link reports IBV_MTU_4096 and is unchanged.
inline ibv_mtu rdma_clamp_mtu(int active) {
  if (active < IBV_MTU_256 || active > IBV_MTU_4096) return IBV_MTU_4096;
  return (ibv_mtu)active;
}

// Path MTU for an RC QP on `port` (the shims use port 1): that port's active MTU, never above 4096.
// Both ends of a direct cable share the netdev MTU, so each side clamping locally picks the same value and the
// peer exchange is left unchanged. `who` is the shim tag used if the query itself fails.
inline ibv_mtu rdma_path_mtu(ibv_context* ctx, const char* who = "rdma", uint8_t port = 1) {
  ibv_port_attr pa{};
  if (!ctx) {
    fprintf(stderr, "[%s] ibv_query_port(%u) failed: no device context; using path_mtu 4096\n", who, (unsigned)port);
    return IBV_MTU_4096;
  }
  // Same return convention as ibv_modify_qp: 0, or a positive errno (older builds return -1 and set errno).
  const int rc = ibv_query_port(ctx, port, &pa);
  if (rc) {
    const int e = rc < 0 ? (errno > 0 ? errno : -rc) : rc;
    fprintf(stderr, "[%s] ibv_query_port(%u) failed: errno %d (%s); using path_mtu 4096\n",
            who, (unsigned)port, e, strerror(e));
    return IBV_MTU_4096;
  }
  const ibv_mtu m = rdma_clamp_mtu((int)pa.active_mtu);
  if (m == IBV_MTU_4096 && (int)pa.active_mtu != IBV_MTU_4096)
    fprintf(stderr, "[%s] port %u active_mtu %d out of range; using path_mtu 4096\n",
            who, (unsigned)port, (int)pa.active_mtu);
  return m;
}

// ibv_modify_qp returns 0 or a positive errno (older builds return -1 and set errno). On failure, prints the
// transition (`step` is INIT, RTR or RTS), the path MTU in bytes and as the enumerator, and errno.
// `peer` < 0: "[who] QP <idx> ..."; `peer` >= 0: "[who] QP to rank<peer> link <idx> ...".
inline bool rdma_modify_qp(ibv_qp* qp, ibv_qp_attr* attr, int mask, const char* who, int idx, const char* step,
                          ibv_mtu mtu, int peer = -1) {
  const int rc = ibv_modify_qp(qp, attr, mask);
  if (rc == 0) return true;
  const int e = rc < 0 ? (errno > 0 ? errno : -rc) : rc;
  if (peer < 0)
    fprintf(stderr, "[%s] QP %d transition to %s failed: mtu %d (%d) errno %d (%s)\n",
            who, idx, step, rdma_mtu_bytes(mtu), (int)mtu, e, strerror(e));
  else
    fprintf(stderr, "[%s] QP to rank%d link %d transition to %s failed: mtu %d (%d) errno %d (%s)\n",
            who, peer, idx, step, rdma_mtu_bytes(mtu), (int)mtu, e, strerror(e));
  return false;
}
