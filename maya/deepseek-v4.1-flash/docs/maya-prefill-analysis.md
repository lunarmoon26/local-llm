# Maya 1017 prefill: findings and next experiments

Inspected 2026-11-05. The source investigation below is followed by a controlled
GPU experiment. The earlier Strata comparison is historical, not Maya's design.

## Controlled experiment: initial chunk sweep

The existing 1.1.34 binary was tested serially with unchanged EXL3 weights,
524288-token capacity, 4096 MiB reserve, two layer-split GPUs, DSpark on, full
expert execution, no expert-profile file, and frozen source-code prompt IDs.
A short warm-up precedes two fresh 8192-token reads (zero prefix reuse), a cached
continuation, and one fresh 32768-token read. A different initial token resets
the conversation between fresh reads; it does not flush OS/Engram/expert caches.

| Chunk | 8K run 1 / run 2 (s) | 32K (s) | 32K tokens/s | Cached continuation (s) |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 19.13 / 18.55 | 66.18 | 495 | 3.39 |
| 1024 | 10.82 / 9.96 | 33.68 | 973 | 1.71 |
| 2048 | 7.48 / 5.93 | 19.67 | 1666 | 1.81 |
| 4096 | 6.66 / 3.96 | 12.69 | 2582 | 1.86 |

The continuation has 8736 prompt tokens, 8223 reused and 513 read (the final
generated token is not yet cached). Larger chunks win strongly on this fresh
long-prompt workload; 1024 is fastest for this one small cached extension.
The 32K speedup for 4096 versus 512 is 5.21x, but each 32K shape has only one
measurement. Cache warming and routing vary: the first/second 8K difference is
large at 4096. These are native-engine prompt times, not an HTTP throughput or
quality benchmark. Greedy text is coherent but not bit-identical, even across
repeated same-chunk runs. Decode after 32K remains 66–74 tokens/s across chunks.

Larger scratch reduces expert residency: the startup reports 21.5 GB outside
VRAM at chunk 512 and 23.1 GB at 4096. The requested reserve remains available.
No weights were rewritten. Raw receipts, prompt hashes, logs, exact commands and
failed attempts are retained in the ignored experiment directory referenced by
`logs/maya-prefill-active.json`. An initial harness attempt used the wrong
tokenizer format and failed before stopping the service; the corrected harness
uses Maya's pack tokenizer. The original 1.1.34/512 service was restored after
the sweep.

### Version candidate and initial attribution

The clean 1.1.55 candidate builds successfully with the same Release/SM120 and
native-expert-disabled settings. Its 144 server CPU tests pass (with upstream
resource warnings and expected simulated-error output). Both native chunk 512
and 4096 runs pass arithmetic (`323`) and prefix reuse. The deployment's chunk
validation and launcher tests pass (21 tests).

| Version / chunk | 8K run 1 / run 2 (s) | 32K (s) | Cached extension (s) |
| --- | ---: | ---: | ---: |
| 1.1.55 / 512 | 14.75 / 14.23 | 42.53 | 2.11 |
| 1.1.55 / 4096 | 4.61 / 4.52 | 14.04 | 1.51 |

1.1.55 improves the small-chunk case, but the initial best-chunk 32K result
does not beat 1.1.34. A repeated best-chunk comparison is necessary before
selecting a version; neither arithmetic nor coherent continuation proves broad
quality equivalence.

The follow-up repeats three fresh 32K reads per version at chunk 4096:

| Version | 32K repeats (s) | Median (s) | Cached extension (s) |
| --- | --- | ---: | ---: |
| 1.1.34 | 13.073 / 13.161 / 13.048 | 13.073 | 1.961 |
| 1.1.55 | 12.680 / 12.777 / 12.835 | 12.777 | 1.405 |

This supports selecting **1.1.55 / chunk 4096**, with the caveat that the version
gain at 32K is only about 2.3% versus the same chunk on 1.1.34. The large overall
gain comes from chunk sizing. The cached-extension comparison is shape-matched,
not byte-identical: each follows that run's generated 32-token continuation.
The source prompt IDs are identical, and every extension reuses 8223 tokens.
The original source checkout and its validation patch remain untouched; selected
serving uses the clean pinned worktree. Old binaries/configuration are retained.

Separate profiled 8K trials report the following GPU-event category totals:

| Version / chunk | Wall prefill (s) | Attention (ms) | Routing/shared (ms) | Experts including waits (ms) | Between layers (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.1.34 / 512 | 20.23 | 1129 | 609 | 25560 | 10613 |
| 1.1.34 / 4096 | 6.77 | 903 | 197 | 2928 | 1505 |
| 1.1.55 / 4096 | 4.67 | 829 | 200 | 2939 | 1398 |

These totals are summed across devices and include dependency waits. They are
not additive wall-clock costs or a physical-SSD profile. Experts and their waits
dominate the original configuration; attention-only kernel work cannot explain
the large chunk-size gain. At 4096, the newer version's compression/indexer
category falls from 129 to 77 ms in this profiled 8K run.

`DS4_FLOW_STATS` produces no stream lines on the active non-flow batch-streaming
path, and `DS4_EG_PROFILE` does not print its window breakdown here. Thus absence
of those lines is not zero I/O or zero Engram cost. A targeted standalone run
uses `DS4_PREFILL_STATS=1` for actual streamed-weight counters and the executable's
prompt host-time report for Engram waits, without changing model math. Its first
standalone attempt accidentally used the executable's default 16-token window;
it was stopped and excluded, with its partial logs retained. The corrected run
sets `--window 8192` so the engine selects the configured internal chunk size.

The corrected standalone attribution runs (`DS4_BENCH_REPEAT=2`) report:

| Version / chunk | Streamed weights per 8K, both GPUs (GB) | First / warm prompt (s) | First / warm Engram waits (ms) |
| --- | ---: | ---: | ---: |
| 1.1.34 / 512 | 252.74 / 252.74 | 19.015 / 18.782 | 390 / 3 |
| 1.1.34 / 4096 | 36.33 / 36.33 | 6.816 / 4.008 | 2820 / 10 |
| 1.1.55 / 4096 | 36.44 / 36.44 | 4.471 / 3.905 | 528 / 9 |

Byte totals sum the engine's `DS4_PREFILL streamed_weight_bytes` counters, not
physical SSD reads. OS page cache/read-ahead can supply these expert bytes; this
experiment does not independently measure PCIe link throughput or physical SSD
traffic. Nevertheless, the approximately sevenfold reduction in streamed-weight
volume directly supports the small-chunk repeated-transfer diagnosis. Engram
cache warming removes most of its wait on repeats. On the first 4096 read,
1.1.55 substantially reduces the exposed Engram wait, consistent with its changed
reader policy, but multiple version changes prevent attributing all of that gain
to thread count alone. No additional Engram tuning is selected from these data.

**Outcome:** choose the existing weights with 1.1.55 and chunk 4096. Keep full
expert computation, conservative YOCO and 524288 capacity. Further kernel work
is not needed to obtain this gain. Broad quality, actual 512K inputs and diverse
prompt workloads remain separate qualification work.

### Installed-service verification

The selected detached service reports **1.1.55**, **524288 capacity**, chunk
**4096**, and loopback binding. `bash check.sh` passes with the reconciled pins;
`bash smoke-test.sh` returns `323`. All ten real HTTP checks pass: loaded health
and model identity, nonstreaming and streaming generation, verbatim copy,
Responses, Messages, tool calling, concurrent-session isolation, disconnect
cancellation/recovery, and context-overflow refusal. The 21 offline deployment
and launcher tests pass; Compose configuration validates. Docker image rebuild
and container runtime were not exercised by this native experiment.

The original binary, config, private environment backup and matching source
worktree remain available in the local experiment receipts. No weights or shared
Engram links were changed, and no boot service, commit or push was created.

## Initial investigation: source and runtime identities

| Item | Identity |
| --- | --- |
| Initial live engine | Reports Maya 1.1.34; deployment selects the build pinned to `249f9a87d1a9d72bec2b684d704b0aa9296b1c9c` |
| Maya checkout inspected | `c4effee67cb95fb4303720919b075fddf04b4e69`, version 1.1.55, with local pack-validation edits and their untracked test |
| vLLM checkout inspected | `cb95d994f4ec066b75672035ca3ae7e1c9745e84`, clean working tree |
| Checkpoint | Existing LibertAIDAI 1017 EXL3 2.9-bpw release `c178a01284fc5c0e859d8668cffaac81ca8f4042`; shared original Engram shards preserved |

Initial configuration: two GPUs in layer-split groups (`--tp 1 --threads on`),
524288 capacity, forced prefill chunk 512, 4096 MiB reserve, mmap experts on,
DSpark on, vision off, no expert profile file. The runtime learns expert-cache
placement; `--expert-profile none` does not mean every decode misses.

**Initial operational blocker:** `bash check.sh` failed with a source-pin mismatch:
the checkout advanced but the deployment still expected 1.1.34. The trial driver
restored the old binary using a clean matching server checkout after each phase.
The selected deployment reconciles the source, binary and launcher pins at
1.1.55 rather than pairing new Python sources with the old engine.

## Observed slowdown

The latest live session's recorded requests:

| New prompt tokens read | Prefix tokens reused | Prompt time | Tokens/s |
| ---: | ---: | ---: | ---: |
| 56328 | 0 | 104748 ms | 537.7 |
| 660 | 56383 | 3417 ms | 193.2 |
| 4395 | 57099 | 12194 ms | 360.4 |
| 833 | 61568 | 3366 ms | 247.5 |

These are existing log observations, not matched A/B trials. Prefix reuse works.
The startup reports approximately 21.5 GB / 1670 experts outside VRAM and read
from model files; prefetch/page cache can serve those reads, so they are not all
physical SSD reads. The per-request CPU-lane percentage and decode cache-hit
percentage do not establish prefill transfer volume. Use flow statistics for that.

## Priority 1: stop forcing tiny chunks without measuring

The launcher fixes `--prefill-chunk 512`. Maya uses that override before automatic
policy (`src/ds4/ds4_model.cu:3830-3868` in 1.1.55; `:3133-3136` in 1.1.34).
The 1.1.55 auto policy ordinarily starts higher for this class of multi-GPU/file
setup; that is not proof its larger setting wins on our host.

At 56328 tokens, nominal chunk counts before YOCO/checkpoint cuts are 111 at 512,
56 at 1024, 28 at 2048 and 14 at 4096. Uniform routing gives 8, 16, 32 and 64
rows/expert respectively (384 experts, top-6); actual routing is not uniform.
Larger chunks amortize dense reconstruction, host routing, streamed expert reads
and pipeline startup, and give tensor-core expert jobs more rows.

**Experiment:** same binary first, chunk 512 -> 1024 -> 2048 -> 4096, one change
at a time. Preserve capacity, source bytes, reserve, speculative settings and
full expert execution. Record selected shapes, expert residency, streamed bytes,
TTFT and decode rate. Larger scratch may evict useful experts or fail to fit;
do not reduce reserve or enable borrowing at the same time to hide that tradeoff.
Retain a tested fallback and expect possible numerical changes from reblocking.

The deployment now accepts `MAYA_PREFILL_CHUNK=512|1024|2048|4096` and validates
its agreement with generated engine arguments. Legacy configurations retain
512. Editing only the argument in the generated JSON is rejected.

## Priority 2: adopt or backport existing Maya fixes before a broad rewrite

### Actual-prompt-sized indexer slabs

1.1.34 sizes indexer query slabs by maximum configured capacity
(`ds4_model.cu:4218`). At 524288 this is only 32 rows even for short prompts.
1.1.55's `DS4_AI_RUN` reserves a bounded 128-row capacity and selects the run's
pitch/row count from actual keys (`:5215-5218,17636-17648`). At 2K, 16K and 56K
prefix endpoints, the inspected formula permits 128 rows rather than 32 for
ratios 1 and 2. Candidate users also bound against their candidate extent.

This is up to four times fewer slabs, **not** a fourfold end-to-end speed claim.
The slab loop also calls EXL3 projection GEMM, so fewer slabs avoid repeated
weight reconstructions. Test full-logit/continuation behavior and memory at long
contexts. It becomes default in 1.1.55; setting the variable on 1.1.34 alone does
not add this implementation.

### Engram queue depth and turn preparation

1.1.34 uses half the hardware thread count, clamped to 4..16, unless
`DS4_ENGRAM_THREADS` is set (`:3703-3714`). 1.1.55 uses twice the thread count,
clamped to 16..64 (`:4532-4547`). Benchmark 8/16/32 readers with cold and warm
Engram distinguished; do not assume more threads always help.

Newer Maya also keeps near-end prompt restore points during one read and adds
incremental prompt preparation (`feed_marks`, `MAYA_PROMPT_MEMO`). Those target
turn latency and repeated setup; tokenizer savings are separate from the logged
GPU prompt-read time. The changes between the deployed pin and HEAD are broad:
qualification is required, not just relinking an old binary to new Python files.

## What to learn from vLLM

### A. Device-side expert planning — adapt after profiling

Maya already sorts routes on GPU, but reads bounds back, synchronizes, constructs
host vectors and plans expert sets each layer/chunk
(`ds4_model.cu:17814-17838,6498-6585`). A device-resident expert subset could have
GPU-generated work queues and grouped dispatch without waiting for CPU bounds.
Keep the host-controlled file/streaming lane for misses, and preserve publication
and reduction order.

vLLM's `model_executor/layers/fused_moe/moe_align_block_size.py:128-165` keeps
sorted IDs, expert IDs and padded counts on device; its fused path consumes them
(`fused_moe.py:1664-1714`). It explicitly avoids per-token weight reads for
host-resident weights. This is an architectural reference, not a drop-in EXL3
implementation: no EXL3 integration was found in this checkout's model executor.

### B. Avoid full dense-weight reconstruction per small GEMM — experiment

Maya `gemm()` reconstructs EXL3 weights into bounded FP16 scratch for every call
before cuBLAS (`ds4_model.cu:5469-5488`). Indexer and attention slabs invoke it
repeatedly. Prototype direct packed dense GEMM or bounded per-layer reconstruction
reuse only if phase profiling identifies it as costly. Preserve exact source
trellis/scales/signs, including different projection bit rates.

Do not confuse this with an absent packed expert kernel: Maya already has
tensor-core EXL3 routed experts (`src/kernels/cuda/exl3.cu:3185-3257`), using
32-row tiles (`:1448-1450`). Existing pack metadata shows 35 uniform 3-bit routed
layers and five uniform 2-bit layers, not mixed per-expert classes. Any new grouped
kernel must support both. vLLM's native FP8/FP4 expert kernels cannot just read
these EXL3 trellises.

### C. Fuse HC boundaries — later, measured kernel work

Maya already batches HC dots through `cublasSgemm`
(`ds4_model.cu:5582-5592`), followed by separate scale/split/norm/FP16-conversion
operations (`:17544-17547,17802-17805`). vLLM fuses prenorm epilogues and post/pre
boundaries (`model_executor/kernels/mhc/tilelang.py:150-222,256-430`). Adapt a
bounded native equivalent if profiler results justify it; respect FP32/BF16/FP16
rounding boundaries. The old Strata eight-token K15-loop diagnosis does not apply.

### D. Bounded decoder replay — preserve semantics, not a free toggle

vLLM's eager prefill writes every row's last-source KV, then runs the remaining
decoder work on the trailing 128 rows (`models/deepseek_v41/nvidia/model.py:876-916`,
`decoder_replay_layers.py:75-83,113-145`). Maya already has YOCO dependency pruning;
its conservative default keeps `(40-20)*(128-1)+128 = 2668` trailing rows, and
requires compatible source/layer placement (1.1.34 `ds4_model.cu:15965-16053`).
The observed layer split appears compatible with the suffix condition, but no
per-request skip trace was collected. Do not assume the skip is off just because
the experts stream, or blindly switch to a 128-row approximation. Verify actual
selected plan first. A smaller replay tail is a separately quality-gated option.

### E. Reuse what Maya already does

Maya already pipelines chunks over layer groups, has compact candidate block
lists, indexed quantized compressed-cache reads, an Engram row cache and early
read overlap. vLLM's direct two-pool sparse attention
(`nvidia/flashinfer_sparse.py:548-571`) is a useful kernel reference, not proof
that Maya copies a full KV prefix like the earlier Strata engine did. Do not
spend time reimplementing capabilities already present.

## Local patch / rollback decision

The only dirty engine-checkout file is `tools/ds4/pack.py`, plus its untracked
test. It changes K-map validation, not runtime math or dispatch.

- Seven local CPU-only K-map tests pass.
- Existing installed pack metadata contains 192503 tensor entries. Unmodified
  packers from both 1.1.34 and current HEAD accept it, checking 47897 EXL3 matrices.
- The local patch also accepts it and checks 47900 matrices (three additional
  mixed shared-expert projections). These tests read metadata, not tensors.

Thus the patch is not required to accept this already-built pack on the inspected
upstreams, and it is not a prefill bottleneck. It was left intact: removing it
would not change performance, and its broader metadata validation can be decided
separately during an upgrade. No user edits or tests were discarded.

## Original experiment plan (completed above)

1. Keep the live 512K service available until a scheduled restart; resolve the
   checkout/binary/server pin mismatch in a separate candidate, without changing
   weights or shared Engram links.
2. Establish one frozen source-code prompt plus cached-turn extensions. Keep
   token IDs, prefix reuse, capacity and DSpark identical across comparisons.
3. On the deployed engine, enable `DS4_PROF=1`, `DS4_FLOW_STATS=1`,
   `DS4_EG_PROFILE=1` for an attribution run. GPU-event category totals include
   waits; sums across overlapped devices are not elapsed wall time. Record
   complete TTFT separately.
4. Sweep chunk 512/1024/2048/4096 serially, then test the newer Maya candidate
   with matched shapes. Separate version effects from chunk effects.
5. Compare first and warm prefills, logits/greedy continuation, prefix restore,
   cancellation, memory and decode speed. Only then choose device-side routing,
   packed dense GEMM or HC fusion based on measured remaining cost.

The source investigation itself launched no model-sized processes; the serial
controlled trials above are subsequent, explicitly authorized GPU work.
