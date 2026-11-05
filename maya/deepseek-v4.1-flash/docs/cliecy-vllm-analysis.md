# Why cliecy's 1017 stack is faster than our current 2.9 path

Inspected 2026-10-25. **Source analysis, not a new runtime profile or benchmark.**
No model ran, no quality evaluation resumed, and no engine code or serving
configuration changed in this investigation.

## Source identity and coverage

- vLLM: `5555a2d8f7116b2b4877f66516f4128176cf722a`.
  The local `vllm-cliecy-5555a2d8` reference initially had no valid HEAD or
  checked-out files. It is now a clean, detached, shallow **sparse checkout** at
  this exact commit, covering the model, attention, mHC, scheduling, speculative
  decoding, workers, collectives and relevant layer implementations.
- The adjacent `vllm-cliecy-5555a2d8.tar.gz` is **truncated**: 20792657 bytes,
  gzip EOF before end-of-stream. It is preserved, not used as complete source.
- Cliecy launcher/results: `c3a37665c93b49a3dbc63d0f0c92d036475b2fc6`.
- Diffbot overlay: `6062be5f0f43c50ba562ef4e7087d4fc83eea36c`, including the patch
  against vllm-exl3 `d3cfd394`, its native kernels and `sitecustomize.py`.
- Strata: local base `3cda8ac27f40840716c9051b2aa382bdfe3325d0` plus the current
  uncommitted 2.9 compatibility, prefill and decode-cache work. The local frozen
  cache receipt records its exact source and binary identity. Strata line anchors
  below refer to this working tree, not a claim those changes are published.

The base vLLM checkout alone is **not** cliecy's full runtime. Its launcher enables
the EXL3 plugin, native xmoe/xmoe_do modules and monkey patches. The inspected
defaults and code establish dispatch behavior; they do not independently verify
every module in cliecy's deployed runtime archive or assign milliseconds to it.

## Main conclusion

### "Full model" does not mean all weights reside in VRAM

Cliecy executes the full model, but does **not** fit every weight on the two
cards. Its detailed architecture places approximately 71 GB of hot expert
weights per GPU, about 60 GB of cold expert weights in host RAM, and 103 GiB
of int4 Engram tables in host RAM. Total reported host use is 180–186 GiB on a
245-GiB machine. GPU execution reads cold mapped host weights over PCIe. Our host
has only about 45 GiB of physical RAM, so the same all-memory-resident layout
does not transfer unchanged.

The bpw labels also hide a large difference: our verified 2.9 source uses
**203.073 GB of original FP8 Engram shards**, whereas cliecy's re-encoded int4
Engram is about **111 GB**. Our source receipt totals approximately 413.892 GB
including the reused Engram files; their advertised ready-to-run bundle is
approximately 335 GB. These are different package contents, not an exact
tensor-by-tensor size comparison, but they show why "3 bpw must be larger than
2.9 bpw" is not a valid total-memory assumption. Our lossless expert/dense pack
has 196.105/6.906 GB respectively, with Engram retained separately.

The earlier Strata allocation failure was the **all-backbone-resident layout**
and its scratch/reserve policy. Our current 2.9 path executes every layer and
all selected experts on GPU, streaming missing bytes from storage; it does not
prune layers or drop cold experts. Thus it already runs the full logical model,
but not with cliecy's memory hierarchy or throughput. Whole-weight residency,
model completeness, configured context capacity and actual long-input
qualification are four different claims.

### Why a 1M context can fit

The reference sets `max_model_len=1048576` and an **FP8 KV pool of 4.25 GiB total**,
about 2.1 GiB/device. DeepSeek's sparse/sliding-window, compressed and shared-
source cache topology is not a dense full-length KV cache duplicated at every
layer (`attention.py:303-348`). The reference documents about 1.07M tokens of
shared pool capacity: one 1M request uses almost the whole pool, not eight
independent 1M conversations (`docs/ARCHITECTURE.md:44-46,58-75`). Host offload and
single-copy dense storage leave GPU room for that pool and workspaces.

Our current 524288 setting is a configured/tested limit, **not demonstrated
proof that the hardware cannot support 1M**. Native cache allocation follows
`max_seq`; inspection found no 524288 hard maximum in the engine. But our BF16
cache representation, mirrored state and full-capacity prefill scratch differ:
`layout()` allocates a `[chunk,max_seq]` byte candidate mask and a full-context
attention staging arena (`engine.cu:1705,1727`). Simply doubling the requested
context changes VRAM pressure and may reduce resident experts or prefill shapes.
It does not qualify actual 1M execution. A proper 1M target needs an explicit
cache/scratch budget and real long-input/retrieval/continuation tests. No such
2.9 test was performed in this analysis.

The comparison is not "one smaller quantization versus one larger quantization."
It is our **eager, non-speculative, small-chunk compatibility path** versus a
**TP2, graph-captured, speculative, packed-kernel serving stack**, with very
different storage hierarchies and dense-weight formats.

Our new GPU cache raised matched short-fixture native decode from 9.765 to
29.641 tok/s, retaining exact logits, but it did not implement the other paths.
Cliecy's 89–112 output tok/s is not a measurement of one ordinary target-token
forward pass. Its approximately 5K prefill tok/s is also not explained by the
decode cache we just added.

## Decode: concrete mechanisms

### 1. DSpark emits several verified tokens per target iteration

**Direct evidence:** cliecy `scripts/serve-bwrap.sh:34-37,72-80` selects DSpark,
five speculative tokens and FULL_DECODE_ONLY CUDA graphs. Its reported accepted
output is 3.97 tokens/step on quality traffic and 3.27 on synthetic code traffic
(`results/README.md:39-41`). These acceptance numbers are workload-specific.

In vLLM, `config/speculative.py:1468-1469` enables parallel drafting for DSpark.
`v1/spec_decode/llm_base_proposer.py:590-637` runs the draft model once and returns
the whole parallel block instead of entering the serial drafting loop.
`models/deepseek_v41/nvidia/dspark.py:72-155,168-199` builds three MTP layers,
shares embeddings with the target, projects target auxiliary hidden states and
precomputes draft context KV. `v1/sample/rejection_sampler.py:45-65,138-190`
verifies drafts and handles the target's recovered/bonus token.

Our 2.9 gate explicitly rejects DSpark/verification, so our 29.641 tok/s measures
96 ordinary target steps, not speculative output throughput
(`src/ds41/engine.cu:586-594`). Multiplying it by their acceptance figure is **not
a prediction**: drafting, multi-row verification and cache costs must be measured.
But comparing output tokens to target iterations obscures a major mechanism.

### 2. Their cold experts are already in mapped, pinned RAM

**Direct evidence:** `exl3-thor.patch:642-647,712-755` holds cold trellis arrays in
CPU tensors, registers them with `cudaHostRegister(Portable|Mapped)`, and exposes
CUDA-addressable UVA aliases. Sign/scale metadata stays on GPU. No file read is
required to service a selected cold expert after loading.

The same patch `:437-558` splits hot/cold counts, starts a cold kernel on a
separate stream, assigns it a bounded SM share, runs the hot kernel concurrently
and joins the streams. The launcher chooses 32 cold SMs for decode and 64 for
prefill (`serve-bwrap.sh:51-54`). **This is GPU execution, not CPU MoE.**

Our cache avoids many file uploads, but every remaining miss still takes the
bounded file-to-host-to-GPU path. More importantly, even cache hits enter
`VramExperts::prepare_decode`: routes are copied to CPU, the consumer stream is
synchronized, selection is prepared on CPU, and publication is fenced again
(`src/ds41/vram_experts.cu:282-353`). Across 40 layers this introduces up to
80 host stream-wait calls per ordinary token before other synchronization.
The calls are structural evidence, not a measured fraction of token latency.

### 3. Their hot set follows routing frequency, ours is largely a layer prefix

**Direct evidence:** the reference plugin consumes per-layer cold-ID plans
(`exl3-thor.patch:660-673`). `recipe/plans/README.md:9-17,37-44` explains that
30% of expert *count* can represent only a few percent of routed uses on the
profiling workloads. Those plan shares are from the recipe's legacy-model
traffic, not fresh 1017 measurements or guaranteed shares on our inputs.

Our experimental static tier enters the `require_all` branch and ranks experts
by `(layer, expert)` order until bytes run out, rather than consulting a
frequency profile (`src/ds41/vram_experts.cu:143-168`). A high resident expert
percentage is therefore not necessarily a high routed-use hit rate. The new
LRU helps after warm-up but does not replace static placement with a calibrated
2.9 hot set. Profiling and byte-budgeted ranking can transfer without importing
their huge host cache; do not blindly reuse their expert IDs.

### 4. Graphs and TP operate on a different execution boundary

Cliecy's launcher enables FULL_DECODE_ONLY graph capture at sizes reaching
8 sequences × 6 rows, and TP2 (`serve-bwrap.sh:72-80,104-109`). Its pointer-table
cold/hot fork/join is graph-capturable (`exl3-thor.patch:456-471`). Our 2.9 gate
requires eager execution; its CPU route/file work prevents simply switching
`DS41_GRAPH=1` on safely.

vLLM shards attention heads/groups and uses column-parallel expansion followed
by row-parallel output projection
(`models/deepseek_v41/attention.py:286-301,364-403`). The plugin patch likewise
uses column sharding for expert w1/w3 and row sharding for w2
(`exl3-thor.patch:699-707,825-832`). Our selected layer split instead completes
stage 0, transfers state, then completes stage 1 (`engine.cu:2933-2947`). The
two GPUs do not concurrently compute a single layer in this mode.

Our experimental TP is not an interchangeable implementation: all three expert
projections are output-channel sharded (`include/strata/ds41/tensor_shards.hpp:30-44`),
with intermediate hidden and output gathers (`engine.cu:1542-1549`). The
reference's complementary sharding/collectives are a concrete design difference
to investigate, not proof selecting our existing TP flag makes it faster.

## Prefill: concrete mechanisms

### 5. 16K effective chunks amortize expert work; our active chunk is 512

The launcher sets `max_num_batched_tokens=16384`
(`serve-bwrap.sh:25-28,107`). vLLM schedules token deficits against that budget
and lets a lone long request use it, rather than artificially applying a
multi-request fairness cap (`v1/core/sched/scheduler.py:600-614,640-655,703-720`).
The 16K budget is not a promise that every actual call has exactly 16384 rows.

Our partitioned prefill is hard-capped at 1024 and the validated 2.9 configuration
requests 512/256 (`engine.cu:1790-1809`). Each chunk reconstructs resident
experts and streams its missing active experts again (`:2135-2214`). At uniform
routing, 512 tokens × 6 / 384 is only eight rows/expert; 16384 tokens give 256.
That arithmetic illustrates reuse opportunity, not a measured route histogram.

### 6. Their enabled packed grouped MoE is disabled on our 2.9 path

The plugin chooses hot-expert decode-once for eligible eager calls from 257 rows
(`exl3-thor.patch:206-217,370-405`). `kernels/xmoe_do/do_gemm.cu:122-135,188-216`
dispatches grouped tensor-core jobs of up to 128 rows, decoding packed tiles
inside the multiply rather than reconstructing a full matrix for each expert.
The path batches expert jobs into common launches.

Our grouped packed kernel exists for the qualified resident 2.7 path, but the
2.9 gate prohibits grouped prefill. Its active K12 loops experts on the host;
for each expert it reconstructs gate, up and down matrices, then launches the
Hadamard/GEMM/activation/scatter pipeline
(`src/ds41/kernels/k12_exl3_moe_prefill.cu:45-75`). The latest cache does not alter
that path. **A 2.7 optimization already present in the tree is not evidence our
2.9 model is using it.**

### 7. Dense weights, Engram and sparse attention have different costs

- Cliecy retains native FP8 dense layers, adds split-K GEMV for small row counts,
  and switches eligible larger dense prefill calls to FlashInfer MXFP8 on one
  retained weight copy (`sitecustomize.py:380-454`; `thor_dense2.py:1-11,185-200`).
  Some small/medium calls still dequantize and use cuBLAS; it is incorrect to
  describe the whole stack as having no dequantization.
- Our dense tensors are EXL3; `exl3_linear_rows` reconstructs the full FP16 matrix
  on each call before cuBLAS (`src/ds41/exl3_linear.cu:149-178`). A smaller routed-
  expert bpw label does not determine this work or encode dense-kernel efficiency.
- Their int4 Engram patch gathers mapped pinned rows directly
  (`sm120-thor-fixes.diff:100-215`); upstream already has stable staged buffers
  and event-ordered side-stream lookups (`common/engram.py:1591-1647`). Our
  preserved original FP8 table rows come from bounded disk reads. Full pinned
  tables cannot fit our host, and int4 conversion is not an allowed shortcut.
- SM120 dispatch explicitly selects FlashInfer sparse MLA
  (`nvidia/model.py:197-243`). The attention kernel receives separate sliding-
  window and compressed pools plus storage-offset indices
  (`nvidia/flashinfer_sparse.py:558-599`), rather than a full-prefix staging copy.
  Candidate buffers store compact block IDs (`nvidia/model.py:703-721`). Our
  dense candidate mask and copied attention arena limit larger stable chunks.

## Optimizations we must not mistakenly credit

- **Host DMA staging is off by default** (`exl3-thor.patch:18-25`). Its README
  records illegal-address failures in two of three boots with decode-once.
  The inspected launch recipe does not enable it. It is not a demonstrated
  explanation for their published rate and is not a safe drop-in recipe.
- SM100/GB200-only mHC coefficient overlap and TP4 multicast all-reduce do not
  automatically run on RTX PRO 6000 TP2 (`nvidia/ops/mhc.py:44-72`). Shared
  TileLang batched/fused post/pre paths remain relevant
  (`model_executor/kernels/mhc/tilelang.py:162-210,386-432`).
- Cliecy disables NCCL P2P on its test host and uses host shared-memory transport.
  Lack of NVLink/P2P alone does not explain our gap or establish a hardware ceiling.
- Their layer/dense formats and loader transformations differ, including an
  enabled FP8 LM-head conversion hook (`sitecustomize.py:457-490`). We cannot
  claim byte-identical model arithmetic from the shared "1017 EXL3" labels.
- vllm-exl3 and its overlay patch are **AGPL-3.0-only**; native xmoe_do is separately
  MIT-licensed. Borrow design ideas with attribution and preserve component
  licensing; do not silently paste plugin Python into the native engine.

## Recommended next work for our 2.9 target

1. **Profile-guided static expert placement**, calibrated on representative 2.9
   traffic and checked on held-out prompts. Keep existing byte budgets, complete
   execution and no model-sized host cache. Count routed uses, not just stored IDs.
2. **Remove per-layer CPU waits from the cache-hit path**, then overlap bounded
   misses with hot-expert computation using explicit publication/lifetime events.
   Reuse the lessons of the earlier descriptor-ordering bug; do not erase fences
   without replacing the ownership dependency. This is the foundation for graphs.
3. **Hybrid 2.9 packed prefill**: grouped hot experts plus bounded cold waves,
   then larger effective chunks after stage-aware scratch planning/compact
   candidates. Retain deterministic reduction and numerical/E2E gates; the
   reference's FP16 intermediates/atomic scatter are not our exact arithmetic.
4. **2.9 DSpark and efficient multi-row verification** after correctness and
   memory qualification, so output-token throughput can be compared at the same
   speculative boundary. Do not just enable the currently rejected flag.
5. **Revisit TP layout/collectives** and direct two-pool attention, separately.
   Keep graph kernels applicable to SM120 distinct from GB200-only mechanisms.

The priority between these candidates needs measured stage latency and routing
histograms. Source inspection establishes missing mechanisms, not their additive
speedups, a guaranteed 5K prefill result, or quality equivalence. A matched
reference would hold prompts/token counts, speculation, precision, warm-up,
cache state, concurrency, capacity and timing boundaries explicit.
