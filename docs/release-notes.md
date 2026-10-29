spark-engine v0.3.8 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.7: Qwen3.8-Flash-Next engine changes, a DeepSeek start-script fix and re-measured numbers. GLM-5.3-Flash
and DeepSeek-V4.1-Flash engine code is unchanged; their outputs were checked identical (DeepSeek release goldens and
paired-prefill checks bitwise identical, GLM teacher-forced logits identical to an earlier export: KL 0.00000, top-1
1.0000 over 167,943 positions). Every Qwen change keeps the verified output unchanged: greedy replies bitwise identical,
sampled replies identical per seed.

**Fix — DeepSeek start script**: `engine-rs/serve/start-ds41.sh` waited for every background job, including the rank 0
engine, which runs until stopped, so the front end was never started (since v0.3.3). It now waits only for the two ssh
launches of ranks 1 and 2.

**Qwen prefill** (all bitwise identical): 12K-token prompt 6.155 → 5.83 s (−5.3%), 64K 33.29 → 31.42 s (−5.6%).
- QSA attention: fp8 K/V shared-memory row stride 256 → 272 bytes (the 8 rows of a fragment load hit one bank, an
  8-way conflict), in the prefill flash kernel and the decode / verify kernels.
- QSA prefill attention: K and V share one tile buffer (V's `cp.async` issued after scoring, overlapping the softmax),
  shared memory 40 → 22.7 KB, 4 blocks per SM.
- HC prefill: the residual-stream reads of the upmix tail are issued before the GEMM main loop, hiding DRAM latency.

**Qwen decode scheduling**:
- `QWEN_SPEC_COST_ONLINE` (default on): the rate rule's round-time model is fitted online (recursive least squares per
  batch size and round type, the static `QWEN_SPEC_COST` as prior, slopes bounded to 0.5–2.5× of it). It only changes
  which verify width is chosen; sampled 4-stream decode +0.7–2.9%, greedy unchanged.
- `QWEN_SPEC_RAGGED` (default off): per-sequence verify widths inside a batch; output unchanged but slower in the
  service, kept for experiments.

**Test system and numbers**: the measuring nodes were updated to DGX OS 7.6.0 (kernel 7.0.0-1019, driver 580.178.04,
new EC / UEFI / ConnectX-7 firmware), and a leftover service that had locked the GPU clock at 2,200 MHz was removed;
earlier numbers were measured with that lock. All DeepSeek and Qwen tables in the README, `docs/benchmarks.md` and
`docs/ds41.md` were re-measured on the updated system at default clocks (sustained runs, clock sampled every second, no
throttling recorded). `bench/decode_bench.py`, greedy, thinking off, 1 / 2 / 3 / 4 streams, median of 5:

| | Prose | Structured |
| --- | --- | --- |
| DeepSeek-V4.1-Flash (3× DGX Spark) | 67.3 / 91.7 / 108.4 / 122.7 tok/s | 126.0 / 177.2 / 190.5 / 222.3 tok/s |
| Qwen3.8-Flash-Next (1× DGX Spark) | 68.4 / 98.0 / 122.4 / 139.2 tok/s | 157.8 / 230.7 / 280.1 / 310.3 tok/s |

Prefill: DeepSeek ~5,550 tok/s for 8K–64K prompts of new text (4K ~4,750), Qwen 2,056 / 2,084 tok/s at 13K and
1,994 / 1,976 tok/s at 77K tokens.

**Docs**: README sections "Test system and GPU clock" (component versions, default clocks vs a 2,200 MHz lock),
"Measured by other users" and an FAQ (context below 1M, slow prefill and RDMA link checks, startup time, decode
differences, node hangs); new switch rows in `docs/qwen-switches.md`.
