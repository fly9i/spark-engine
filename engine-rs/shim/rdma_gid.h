// SPDX-License-Identifier: MIT
// Shared by the RDMA transports (rdma_ar, rdma_big, rdma_arn):
//  - the GID index of a RoCE device's port 1 that holds its IPv4-mapped RoCE v2 address. The index is not the same on
//    every node (one node 3, another 4 -- that node's index 3 is empty), so it is looked up instead of fixed;
//  - the RC path MTU: RoCE derives the port's active MTU from the netdev MTU (1500 -> IBV_MTU_1024, 9000 -> 4096) and
//    with some driver / firmware versions mlx5 rejects the RTR transition (EINVAL) when path_mtu exceeds it (issue #2;
//    others accept it). Each side sends its port's active MTU with its QP info and both program the smaller of the two,
//    so ends with different netdev MTUs agree. A 1500-byte netdev MTU means 1024-byte RDMA packets: set the direct
//    links to MTU 9000 for 4096 (640 KiB all-reduce over one 1024 link: ~80 us vs ~71 us).
#pragma once
#include <infiniband/verbs.h>
#include <algorithm>
#include <cerrno>
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

// Payload bytes of an ibv_mtu enumerator (0 for anything else).
inline int rdma_mtu_bytes(int m) {
  switch (m) {
    case IBV_MTU_256: return 256; case IBV_MTU_512: return 512; case IBV_MTU_1024: return 1024;
    case IBV_MTU_2048: return 2048; case IBV_MTU_4096: return 4096; default: return 0;
  }
}
// This side's offer: port 1's active MTU, at most IBV_MTU_4096 (what these QPs always used); 4096 when the query fails
// or reports something outside 256..4096 (with a message naming `who`).
inline uint32_t rdma_port_mtu(ibv_context* ctx, const char* who) {
  ibv_port_attr pa{};
  const int rc = ctx ? ibv_query_port(ctx, 1, &pa) : -1;
  if (rc) {
    const int e = rc < 0 ? (errno > 0 ? errno : EINVAL) : rc;
    fprintf(stderr, "[%s] ibv_query_port(1) failed: errno %d (%s); offering path_mtu 4096\n", who, e, strerror(e));
    return IBV_MTU_4096;
  }
  const int m = (int)pa.active_mtu;
  if (m < IBV_MTU_256 || m > IBV_MTU_4096) {
    fprintf(stderr, "[%s] port 1 active_mtu %d out of range; offering path_mtu 4096\n", who, m);
    return IBV_MTU_4096;
  }
  return (uint32_t)m;
}
// The path MTU both ends program: the smaller offer (an offer outside 256..4096, e.g. from a zeroed field, counts as 4096).
inline ibv_mtu rdma_path_mtu(uint32_t mine, uint32_t peer) {
  auto ok = [](uint32_t m) { return m >= IBV_MTU_256 && m <= IBV_MTU_4096 ? m : (uint32_t)IBV_MTU_4096; };
  return (ibv_mtu)std::min(ok(mine), ok(peer));
}
// ibv_modify_qp with the failure named: the transition (`step` INIT / RTR / RTS), the path MTU and errno (ibv_modify_qp
// returns 0 or a positive errno; older builds -1 with errno set). `peer` >= 0: "QP to rank<peer> link <idx>".
inline bool rdma_modify_qp(ibv_qp* qp, ibv_qp_attr* attr, int mask, const char* who, int idx, const char* step,
                           ibv_mtu mtu, int peer = -1) {
  const int rc = ibv_modify_qp(qp, attr, mask);
  if (rc == 0) return true;
  const int e = rc < 0 ? (errno > 0 ? errno : EINVAL) : rc;
  if (peer < 0) fprintf(stderr, "[%s] QP %d transition to %s failed: path_mtu %d errno %d (%s)\n", who, idx, step, rdma_mtu_bytes(mtu), e, strerror(e));
  else fprintf(stderr, "[%s] QP to rank%d link %d transition to %s failed: path_mtu %d errno %d (%s)\n", who, peer, idx, step, rdma_mtu_bytes(mtu), e, strerror(e));
  return false;
}
