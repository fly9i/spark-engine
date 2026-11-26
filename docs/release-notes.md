spark-engine v0.3.14 for NVIDIA DGX Spark (GB10, aarch64, CUDA 13.0, sm_121).

Changes since v0.3.13, all for Qwen3.8-Flash-Next and all contributed by @kk1987 (#4, #6, #7, #8; background in #5): the
smaller EXL3 revisions of turboderp/Qwen3.8-Flash-Next-exl3 now load, and a crash with images in the history is fixed.
GLM-5.3-Flash and DeepSeek-V4.1-Flash are unchanged.

**More EXL3 revisions of Qwen3.8-Flash-Next.** Until now only `4.05bpw_h6_ng6` loaded. Three things were in the way:

- *n-gram table* (#6). Other revisions store the table as exllamav3 row shards and at other code widths (NG4, NG5, NG8).
  The loader reads either the single tensor or the shards (which sit back to back in one file, so the table stays one
  memory-mapped span) and takes the width from the row size. It also opens `mtp_hyper_connection_mixer_patch.safetensors`,
  which some revisions ship outside their index.
- *Expert and dense widths* (#7). The dense kernels get K = 2 and 3; routed experts run at any K from 2 to 6 through the
  mixed-K path GLM-5.3 already uses. A trellis width that is not a whole K is now rejected at load instead of being
  rounded down.
- *Half-integer K* (#8). The self-calibrated revisions use exllamav3's K = 2.5 for most experts; the expert kernels decode
  it.

On one DGX Spark, from the unmodified downloads:

| Revision | n-gram table | Load | Decode, prose prompt |
|---|---|---|---|
| `2.05bpw_h4_ng4` | NG4, 26.2 GB | 12 s | 1 / 4 streams 66.7 / 141 tok/s |
| `2.68bpw_h6_ng8` | NG8, 51.8 GB | 13 s | 1 / 3 streams 67.5 / 111.5 tok/s |
| `4.05bpw_h6_ng6` (as before) | NG6, 39.0 GB | 14 s | 3 / 4 streams 108.7 / 123.5 tok/s |

Both new revisions give sensible replies to the 20 test prompts. `3.05bpw_h5_ng5` was tested by the contributor (on an
N1X laptop), not on our Sparks. Known gaps: one 4-stream run of 2.68 bpw was slow (46 tok/s aggregate) and has not been
investigated; and only 4.05 bpw ships the vision tower the engine reads (`vision_k6.safetensors`), so the other
revisions are text-only for now. `start-qwen.sh` starts them text-only on its own and says so.

**Images in the history: prompt-lookup drafts stop at media** (#4). When the output repeated a prompt line that an image
follows (a caption, a file name), the lookup draft ran on into the image's placeholder ids, the verify pass embedded
them as token ids, and the engine died with a CUDA device-side assert. The draft now ends before the first placeholder,
as GLM's already did. A prompt of eight captions each followed by an image, with the model asked to copy the captions
at temperature 0.3, completes 10 / 10 requests.

Checks against v0.3.13 on our DGX Sparks: Qwen3.8-Flash-Next 4.05 bpw greedy replies identical (20 / 20) at unchanged
speed; GLM-5.3 on two and on three Sparks, which shares the expert kernels, bitwise identical outputs and expert-kernel
timings within 0.6% (8 to 32 rows) or 2-4% faster (1 to 4 rows); DeepSeek-V4.1 release goldens and paired-prefill checks
identical, serving and image tests pass.
