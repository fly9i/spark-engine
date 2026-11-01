spark-engine v0.3.11 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Fixes since v0.3.10: a hang on DeepSeek-V4.1-Flash image requests, RDMA start-up on links at the default MTU 1500, and
three-node runs no longer need a network shared by all three nodes. Text outputs are bitwise identical to v0.3.10.

**DeepSeek-V4.1-Flash: image requests could hang the server.** Only rank 0 encodes images, and two prefill decisions
were taken from rank 0's image rows, so ranks 1 and 2 took a different path and the collectives no longer matched. The
three ranks then waited on each other for ever (or rank 0 failed with `rs_sp_reduce_scatter async`). It happened
whenever a prefill chunk that contained image rows had at least 1,024 rows: a large image (about 930 image tokens or
more) with a short question, or any image after about 1,000 tokens of text. The decisions now come from the image
placeholders in the token ids, which every rank holds; rank 0's choice is unchanged and text-only chunks take the same
path as before. The image checks of `bench/ds41/vision_e2e.py ... boundary` (an image after 4.5K / 5.6K tokens) pass.

**RDMA path MTU follows the ports (#2, #3).** The RDMA transports set the RC path MTU to 4096 regardless of the port.
On a link left at the default netdev MTU 1500 the port reports `active_mtu` 1024, and with some driver / firmware
versions mlx5 then rejects the queue pair's RTR transition (`QP 0 transition failed`). Each side now sends its port's
active MTU with its queue-pair information and both program the smaller value, so links whose two ends differ also
agree. Transition failures name the step, path MTU and errno, and the ready lines print the MTU. On three DGX Sparks
with one link at MTU 1500 on both ends, and with 1500 on one end and 9000 on the other, both ends programmed 1024 on that
link and 4096 on the others, and the sums were correct. 1024-byte packets cost some speed on large transfers (a 640 KiB
all-reduce over that link: about 80 µs against 71 µs), so MTU 9000 on the direct links is still recommended. Thanks to
@erdholion for the report and the analysis.

**Three nodes: no NCCL group.** Three DGX Sparks are a mesh of point-to-point links, so NCCL's socket bootstrap (one
address that every rank reaches) needed a LAN shared by all three (`SPARK_TP3_IFNAME`). Every collective of GLM-5.3 TP3
and DeepSeek-V4.1 already went over the engine's N-way RDMA; NCCL was only initialized. `GLM53_TP_NCCL=0` (set by
`profiles/tp3-rdma.env` and `serve/ds41-rank.sh`) skips it: the ranks meet at rank 0's TCP store over the direct links
(`SPARK_MASTER_ADDRS`) and exchange their RDMA queue pairs there. `SPARK_TP3_IFNAME` is now optional (only for a
`GLM53_TP_NCCL=1` debugging run), and a collective without an RDMA path stops with an error naming it instead of
silently falling back to NCCL sockets. Two nodes are unchanged.

Checks against v0.3.10 on our three DGX Sparks (MTU 9000): DeepSeek release goldens and paired-prefill checks identical,
image and serving tests pass; GLM-5.3 TP3 greedy replies byte-identical and teacher-forced KL 0; GLM TP2 and
Qwen3.8-Flash-Next greedy replies identical (20 / 20 each); speed unchanged.
