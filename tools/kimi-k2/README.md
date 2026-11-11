# Kimi K2 bring-up

**Current result: full-model short inference completed on CUDA + HIP.**
The 27-token chat prompt traversed all61 layers and generated `Hi there`
(two tokens). Both remote expert directions executed with zero fallbacks;
allfour native processes exited0. See [the milestone report](../../docs/KIMI_K2.md).
The sections below retain the development history; earlier NOT RUN statements
describe their original checkpoints, not the current vertical milestone.

This directory is separate from the K3 reference in `tools/kimi`.
It reads the pinned K2 UD-IQ3_XXS GGUF by byte ranges for CPU reference work.
It does not add a pager, engine cache, placement system, or transport.

The source/quant revisions, exact nine-shard identities, dependency map and
storage estimate are in `kimi-k2-compatibility.json` at the repository root.
Gate receipts are in `evidence/kimi-k2/`. Raw captures are outside Git at
`/mnt/work/kimi-k2/`. Full shard hashes are published Hugging Face LFS
identities; full shards have not been downloaded or locally hash-verified.

K0 passed the identity, source compatibility and storage inventory checks.
K1 passed narrow real-weight reading: all layer-0 MLA/dense tensors and
expert 0's gate/up/down in layer 1, totaling 289,568,768 bytes. CPU format
checks used gguf's dequantizer and compared bounded float32 dot products
against float64 accumulation. This is not K2 operator-reference parity or
CUDA/HIP execution evidence.

Commands used (from this worktree):

```sh
uv run python tools/kimi-k2/index_gguf.py /mnt/work/kimi-k2/k1/gguf-header.bin --output /mnt/work/kimi-k2/k1/gguf-index.json
OPENBLAS_NUM_THREADS=4 uv run --with gguf python tools/kimi-k2/read_slices.py /mnt/work/kimi-k2/k1/gguf-index.json /mnt/work/kimi-k2/k1/slices
```

K2 comparison thresholds were recorded before running any layer/sublayer
comparison: relative RMS at most 1e-4 and maximum absolute error at most 1e-3.
K2 subsequently passed as recorded below. K3 and later gates have not passed. Full model staging is not enabled.

## K2 CPU operator reference

K2 passed on three distinct scaled Gaussian inputs per computation. Layer 0
includes standard MLA, K2 YaRN at positions 0..2, residual/RMSNorm, and dense
SwiGLU. Its final maximum absolute difference against the actual pinned upstream
attention and MLP definitions was `2.3841858e-7` (relative RMS `3.4183383e-8`).

Layer 1's MoE sublayer selected matching top-8 expert sets on all inputs,
including nonuniform routing. Routing-weight maximum difference was `8.9406967e-8`
(the weights are numerically agreeing, not bitwise identical). Reduction preserves
the upstream router's native unsorted order. Combined routed/shared output maximum
difference was `2.2053719e-6` (relative RMS `1.1085997e-6`). Every input separately
passed the frozen relative RMS <=1e-4 and absolute <=1e-3 limits. Inputs, IDs,
weights and per-expert boundaries are in `/mnt/work/kimi-k2/k2/moe.npz`.

The independent codec check compiled the existing IQ3_S/IQ4_XS dequantizers
from ggml commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, using Strata's bundled
`ggml-common.h`. All 44,040,192 decoded expert0 values matched the Python decoder
bit-for-bit. This does not qualify native GPU expert execution.

Reproduction commands:

```sh
OPENBLAS_NUM_THREADS=4 uv run --with gguf python tools/kimi-k2/cpu_reference.py /mnt/work/kimi-k2/k1/slices /mnt/work/kimi-k2/k0/config.json --output /mnt/work/kimi-k2/k2/layer0.npz
OPENBLAS_NUM_THREADS=4 uv run --with gguf --with torch python tools/kimi-k2/compare_upstream.py /mnt/work/kimi-k2/k1/slices /mnt/work/kimi-k2/k0/config.json /mnt/work/kimi-k2/k0/modeling_deepseek.py /mnt/work/kimi-k2/k2/layer0.npz --output /mnt/work/kimi-k2/k2/layer0-comparison.json
OPENBLAS_NUM_THREADS=4 uv run --with gguf --with torch python tools/kimi-k2/moe_reference.py /mnt/work/kimi-k2/k1/gguf-index.json /mnt/work/kimi-k2/k0/config.json /mnt/work/kimi-k2/k0/modeling_deepseek.py /mnt/work/kimi-k2/k2
OPENBLAS_NUM_THREADS=4 uv run --with gguf python tools/kimi-k2/check_quant_decode.py /mnt/work/kimi-k2/k2/ggml/ggml-quants.c third_party/ggml /mnt/work/kimi-k2/k1/slices /mnt/work/kimi-k2/k2/quant-oracle
```

The upstream oracle reconstructs the original `kv_b_proj` by undoing GGUF's
key-block transpose and concatenating key/value blocks. It runs the upstream
mathematical class definitions without importing the server or full model.
All compared weights belong to the pinned GGUF quant track, not original FP8.
Full-model output, text quality, native GPU support and compressed-cache
incremental decoding have not been qualified by K2.

## K3 sequence reference (in progress)

`prepare_sequence.py` fetches 24 embedding rows from the pinned token sequence
and layer 1's attention/normalization tensors. `sequence_reference.py` evaluates
both actual K2 layers using absorbed MLA with compressed state `[positions,512]`
and rotary keys `[positions,64]` on each layer owner. The comparison runs:

1. One 24-position causal prefill.
2. A 16-position prefill followed by eight single-token, teacher-forced positions.
3. The pinned upstream expanded-MLA implementation on the same 24 IDs.

It compares both layer boundaries, compressed KV state, and expert selection.
The input IDs come from the pinned tokenizer/chat template; the eight later
positions are fixed input IDs, not generated model tokens. No output-head or
text-generation claim follows from this two-layer check.

```sh
OPENBLAS_NUM_THREADS=4 uv run --with gguf --with torch python tools/kimi-k2/sequence_reference.py /mnt/work/kimi-k2
```

K0/K1/K2 captures are read-only. Reproducing those comparisons requires a fresh
output destination instead of overwriting the captured receipts.

K3 passed on 11 November 2026. Both layers executed all 24 real token positions.
The final layer's incremental/prefill maximum difference was `7.1525574e-7`
(relative RMS `4.3803686e-7`); upstream-expanded/prefill maximum difference was
`1.7881393e-7` (relative RMS `3.2608304e-7`). Both caches retained exactly
24 latent rows of 512 values and 24 rotary-key rows of 64 values. Cache comparisons
and expert-set checks passed; all captured arrays were finite. Exact IDs,
layer outputs, expert IDs and KV states are preserved read-only in
`/mnt/work/kimi-k2/k3/two-layer.npz`. This is two-layer CPU qualification only.

## K4 native GPU bring-up (partial)

`src/kimi2/layer0.cu` uses Strata's existing F32 GEMV, `finish` RMSNorm/RoPE,
and SwiGLU kernels, with a K2 standard causal-attention layout. The opt-in
`strata-kimi2-layer` CMake target is excluded from default builds.

The real-weight layer0 fixture executed on Lucy's RTX3080/CUDA and Astor's
RX6800/HIP gfx1030. Both final outputs had maximum absolute difference
`2.3841858e-7` against the frozen CPU reference. Normalization, attention output
and MLP output boundaries also passed the same 1e-4 relative / 1e-3 absolute
comparison bounds. Build commands and per-boundary metrics are in the K4 receipt.

This first native run uses decoded F32 fixture matrices and three synthetic
hidden-state vectors. It does not yet bind the original GGUF blocks to GPU
kernels, route/execute K2 experts, or use the existing expert cache/CPU fallback.
K4 therefore remains unpassed. It does not register a serving model family.

```sh
OPENBLAS_NUM_THREADS=4 uv run --with gguf python tools/kimi-k2/export_native.py /mnt/work/kimi-k2
/mnt/work/kimi-k2/k4/strata-kimi2-layer-cuda /mnt/work/kimi-k2/k4/fixture /mnt/work/kimi-k2/k4/cuda
OPENBLAS_NUM_THREADS=4 uv run --with numpy python tools/kimi-k2/compare_native.py /mnt/work/kimi-k2/k2/layer0.npz /mnt/work/kimi-k2/k4/cuda --output /mnt/work/kimi-k2/k4/cuda-comparison.json
```

K4 packed-dense update: `--packed-dense` consumes the original Q4_K/IQ4_XS
files (`*.quant` plus type descriptors) and decodes them on the device. An
initial F16 scratch path failed the attention relative-error bound (2.7564e-4).
That negative result is preserved. The existing GPU block decoder now also
supports explicit F32 output for Q4_K, IQ3_S and IQ4_XS. CUDA and HIP runs using
F32 scratch both passed, with final maximum absolute error 2.3841858e-7.

The split Q8 key/value projection still uses the decoded F32 fixture. This is
not yet a fully native quantized layer or expert-tier integration. The adapter
allocates temporary projection buffers only; no expert cache or pager is added.

K4 native quant update: all original layer0 matrices, including split Q8 key/value,
now decode on the GPU. The Q8 path uses the existing `dequant_q8_0` and a K2
layout conversion to reconstruct the original combined projection. CUDA and
HIP again passed with final maximum absolute difference 2.3841858e-7.

Standalone expert0 now executes from original IQ3_S gate/up and IQ4_XS down
blocks with `--expert`. CUDA maximum difference was 4.2915344e-6; HIP maximum
difference was 4.5299530e-6. Both relative RMS errors were below 6.3e-7.
The three inputs are the saved K2 operator inputs. `export_native.py` produces
this expert fixture and its CPU reference as well as the layer0 fixture.

K4 remains partial: this operator executable has not yet connected GPU routing,
expert-cache residency, the CPU pool/file-backed miss path, or a native real-token
layer0/1 chain. No cache, pager, placement system or transport was introduced.

K4 GPU routing update: the existing native top-k template now has an explicit
K2 sigmoid/bias policy. `strata-kimi2-router` executes the real 384x7168 gate
projection followed by top8 selection. CUDA and HIP selected exactly the CPU
reference expert sets on all three inputs; aligned weights differed by at most
5.9604645e-8. The native ordering is descending selection score, with lower IDs
winning exact ties; comparisons align equivalent unordered expert sets.

A focused pre-change/updated binary comparison covered the existing 256/8 and
512/10 router cases, three tokens each, on both GPUs. IDs and weights were
byte-identical. Source and captures are under `k4/default-router` outside Git
and the shared K4 router fixture directory. This is a bounded default-path check,
not a full regression-suite claim.

```sh
/mnt/work/kimi-k2/k4/strata-kimi2-router-cuda /mnt/work/kimi-k2/k4/router /mnt/work/kimi-k2/k4/router-cuda
OPENBLAS_NUM_THREADS=4 uv run --with numpy python tools/kimi-k2/compare_router.py /mnt/work/kimi-k2/k2/moe.npz /mnt/work/kimi-k2/k4/router-cuda --output /mnt/work/kimi-k2/k4/router-cuda-comparison.json
```

K4 is still partial until these operators run together through the existing
expert cache and CPU/file-backed tier. The existing pool's fixed activation
scratch limits need K2 geometry support; this is adapter/pool plumbing, not a
reason to introduce another expert executor.

K4 CPU-tier update: the old format entry point rejected K2's 8176-byte input
and 2336-byte down activations. An explicit pool-sized overload now permits the
existing `ExpertPool` to resize its own native intermediate scratch for K2.
Legacy format calls keep their old limits and fixed buffers.

The mapped expert ran on both Lucy and Astor through the existing two-worker
pool. Native ggml activation-quantized arithmetic matched serial native rows
exactly, but differed from the decoded-F32 reference by 1.7913% relative RMS
(maximum absolute 0.06989193). This arithmetic difference is preserved in the
receipt; it is not a pass against the F32 reference bounds.

The explicit `NativeArithmetic::DequantizedF32` mode uses ggml's existing
weight decoder with original F32 activations, F64 dot accumulation and F32
outputs inside the same pool. It passed on both hosts: maximum absolute
2.3841858e-6, relative RMS 5.2238994e-7. All three inputs individually passed the
1e-3 absolute / 1e-4 relative limits. Pool-vs-serial parity was exact. Legacy
native outputs were byte-identical before/after on each host, and legacy
format rejection remained unchanged.

`strata-kimi2-cpu-expert FIXTURE PREFIX --f32` selects this arithmetic explicitly;
omitting `--f32` retains the native activation-quantized path. The current
executable maps the three immutable expert tensors directly, using the same
split-base offset contract as the existing mapped-GGUF `ExpertSource` path.
No new pool, expert cache, pager or queue is introduced. K4 is still partial:
GPU-cache admission and combined routed/shared MoE execution remain unqualified.

K4 combined MoE update: `strata-kimi2-moe` now uses the existing `ExpertCache`
with a fixed one-slot plan (layer1, expert0) and the existing `ExpertPool` for
mapped misses. The real GPU router remains unchanged by residency. Both hosts
executed one GPU-cache assignment and 23 mapped CPU assignments across the three
inputs. The native shared expert was added after routed weighted reduction.

All saved boundaries and all three final outputs passed the frozen comparison
bounds against the K2 upstream reference. Final maximum absolute differences:
CUDA 2.0265579e-6; HIP 1.9669533e-6. Per-expert outputs, routing weights, reduction,
shared output and final output are captured outside Git under `k4/moe-*` and
in the shared K4 fixture directory. The cache held 20,414,464 original packed
bytes. This is a bounded tiered MoE sublayer run, not cluster capacity proof.

```sh
/mnt/work/kimi-k2/k4/strata-kimi2-moe-cuda /mnt/work/kimi-k2/k4/moe-fixture /mnt/work/kimi-k2/k4/moe-cuda
OPENBLAS_NUM_THREADS=4 uv run --with numpy python tools/kimi-k2/compare_moe_native.py /mnt/work/kimi-k2/k2/moe.npz /mnt/work/kimi-k2/k4/moe-cuda --output /mnt/work/kimi-k2/k4/moe-cuda-comparison.json
```

The first HIP combined build interpreted CPU object files as source; the driver
command was corrected with `-x none` before those objects. The failed log is
retained separately from the successful retry. CUDA and HIP executed separately;
no distributed K2 pass is claimed. K4 still needs the native layer1 attention/
residual path composed with this MoE path, followed by the real-token native chain.

The native K2 functions now live in the opt-in `strata_kimi2` library, with a
small model-specific API in `include/strata/kimi2/engine.hpp`. The current profile
is explicit: layers0/1, three hidden rows, fresh MLA state. `strata-kimi2-chain`
passes actual token embeddings through layer0 and then through layer1's MLA,
residual/RMSNorm and the same tiered routed/shared MoE implementation. It does
not shell out to the operator executables or substitute CPU attention.

The worktree's isolated CUDA CMake build completed for `strata-kimi2-moe` and
`strata-kimi2-chain`; the CMake-built MoE also passed the frozen reference check.
The first chain build command needed regeneration after adding the new CMake
target; the initial 'No rule to make target' log is retained.

```sh
uv run --with gguf python tools/kimi-k2/prepare_native_chain.py
cmake -S . -B /mnt/work/kimi-k2/build-cuda
cmake --build /mnt/work/kimi-k2/build-cuda --target strata-kimi2-chain -j 4
/mnt/work/kimi-k2/build-cuda/strata-kimi2-chain /mnt/work/kimi-k2/k4/fixture /mnt/work/kimi-k2/k4/native-chain/layer1 /mnt/work/kimi-k2/k4/native-chain/experts /mnt/work/kimi-k2/k4/native-chain/embeddings.f32 /mnt/work/kimi-k2/k4/native-chain/cuda
OPENBLAS_NUM_THREADS=4 uv run --with numpy python tools/kimi-k2/compare_native_chain.py /mnt/work/kimi-k2/k3/two-layer.npz /mnt/work/kimi-k2/k4/native-chain/cuda --output /mnt/work/kimi-k2/k4/native-chain/cuda-comparison.json
```

K4 passed for the two-layer bring-up profile on 11 November 2026. Both hosts
executed actual embeddings for token IDs `[163594, 14062, 163601]` through full
native layers0/1, including GPU MLA, residual/RMSNorm and tiered routed/shared
MoE. Expert sets agreed exactly with the frozen upstream reference. Final layer1
maximum absolute differences were 1.4901161e-7 on CUDA and 1.1920929e-7 on HIP
(relative RMS 5.0421045e-7 and 5.1008647e-7). Both layer boundaries passed.

Astor's completed layer1 capture initially was not visible from Lucy's NFS
view. It was collected directly from Astor, without restarting the model; the
remote and collected layer1 SHA-256 matched. The failed observation/comparison
logs and successful collected comparison are retained. Shared model bytes were
not changed. This is a two-layer native operator/tier qualification, not a
native incremental-cache, distributed or full-model result. K5 remains next.

K5 now passes on Lucy/CUDA for four consecutive real K2 layers (0–3), on
four actual tokenizer embedding rows. The independent CPU reference executes
the pinned upstream attention, router and MLP arithmetic on the same decoded
pack tensors. Layer 2 introduces Q5_K shared-down; layer 3 introduces Q5_K
attention and IQ3_S expert-down. These use existing Strata decoders.

The native run compares `[4]` prefill with `[2,1,1]` prefill/incremental windows.
All layer outputs pass the unchanged maximum absolute `1e-3` and relative RMS
`1e-4` bounds against the independent reference. The largest measured output
error is `1.0728836059570312e-6`, relative RMS `6.207860678841826e-7`.
Expert sets match exactly. Final compressed MLA latent `[4,512]` and rotary
`[4,64]` states are byte-identical across both native window modes in all four
layers. The prefill executes one existing GPU-cache expert assignment and 95
mapped CPU assignments; it does not force individual tokens to hit expert 0.

Commands and captures:

```sh
uv run --project tools/kimi-k2 --extra compare python tools/kimi-k2/four_layer_reference.py
uv run --project tools/kimi-k2 --extra compare python tools/kimi-k2/prepare_four_layer_native.py
cmake --build /mnt/work/kimi-k2/build-cuda --target strata-kimi2-sequence -j 4
/mnt/work/kimi-k2/build-cuda/strata-kimi2-sequence /mnt/work/kimi-k2/k5/native /mnt/work/kimi-k2/k5/cuda-prefill prefill
/mnt/work/kimi-k2/build-cuda/strata-kimi2-sequence /mnt/work/kimi-k2/k5/native /mnt/work/kimi-k2/k5/cuda-incremental incremental
uv run --project tools/kimi-k2 python tools/kimi-k2/compare_four_layer_native.py /mnt/work/kimi-k2/k5/cuda-prefill prefill
uv run --project tools/kimi-k2 python tools/kimi-k2/compare_four_layer_native.py /mnt/work/kimi-k2/k5/cuda-incremental incremental
```

Raw reference and native outputs are under `/mnt/work/kimi-k2/k5`; the gate
receipt is `evidence/kimi-k2/K5/receipt.json`. The initial reference failed by
attempting to write the read-only K3 manifest; K5 uses its own manifest and links
to preserved original bytes. That failure log remains. This proves a four-layer
native chain, not logits, generation, distributed ownership or full-model fit.
K6 remains unrun and complete-pack staging remains prohibited.

K5 also passed on Astor/HIP for the same four real embedding rows using `[2,1,1]`
windows. Expert sets are exact; maximum output error is `1.0728836059570312e-6`
and relative RMS is `6.394565161826904e-7`. Captures were collected from Astor's
local `/var/tmp/kimi2-k5-hip*` into `/mnt/work/kimi-k2/k5/hip-capture` after exit 0.

K6 is in progress. The opt-in `strata-kimi2-expert-worker` builds the existing
`src/program/expert_worker.cu` with K2 fixture binding and the already verified
F32 decoded-weight expert operator. It uses the existing worker protocol, plan
installation, VRAM slots, byte verification and weighted result reduction.
Default worker builds do not compile the K2 binding. `strata-kimi2-expert-probe`
uses the existing `NetworkExpertExecutor` without a new transport.

One real layer1 expert0 assignment, with original router weight `0.3508636057`,
executed from Astor's client on Lucy's CUDA worker over the existing link. Maximum
absolute weighted-output error is `7.548806024715304e-11`, relative RMS
`1.0847962812476908e-6`, with one executed assignment and zero fallbacks.
The input is the explicitly labeled three-row K2 operator fixture, not a token
sequence. The worker identifies this as a partial slice fixture, not a complete
checkpoint load. `/mnt/work/kimi-k2/k6` retains the raw client/worker logs,
independent expected output, comparison and fixture manifest. The first worker
startup failure (legacy grouped-kernel capability predicate) is retained too.
Neither the reverse direction nor two-host native layer ownership has passed.

K6 now passes for the four-layer partial K2 model. The pinned DeepSeek layer
session/TCP implementation was imported into this worktree from source commit
`c9f535fe9d73c784aa3ba518199bc4264d150ce9`; exact original-file hashes are in
`/mnt/work/kimi-k2/k6/layer-fabric-import.json`. The opt-in K2 build supplies
cut 2, top-8 decoded-F32 arithmetic and a declared 7168-wide hidden result.
The legacy result container's `logits` member carries hidden rows in this
partial-model adapter; it does not represent full-model vocabulary logits.
No Q3 checkout source or running process was changed.

The healthy run used shared NFS raw slices, one immutable epoch-1 placement,
Lucy/CUDA layers0–1 and Astor/HIP layers2–3. It executed windows `[2,1,1]`, with
one routed expert assignment Lucy→Astor and two Astor→Lucy. Both directions
reported zero fallbacks. All four layer outputs passed the previously frozen
`1e-3` max absolute / `1e-4` relative RMS bounds against the real-weight CPU
reference; largest output error was `1.0728836059570312e-6`. Router expert sets
matched exactly, and returned suffix bytes equal Astor's captured bytes.
The coordinator, suffix and both healthy expert workers all exited 0.
Raw captures and comparisons are under `/mnt/work/kimi-k2/k6`.

A separate failure run deliberately terminated only its owned expert worker
after READY. The existing executor returned `expert worker disconnected` in
0 measured milliseconds, became unhealthy and recorded one failure. That
expected failure is separate from the healthy run's zero-fallback receipt.
No local model recomputation or success was substituted for the failed request.

K0–K6 PASS now authorizes K7 staging of the exact nine-shard full pack into
`/mnt/storage/models/Kimi-K2-Instruct-UD-IQ3_XXS-23bf90d8196d`, using the existing
Hugging Face CLI with two workers and the pinned quant revision. K7 itself
remains unrun until all61 layers, final logits and autoregressive decode execute.
The F, C and P scores are still unqualified.

K7 implementation now builds on CUDA and HIP. `kimi2::Model` composes the existing
`GgufModel` sharded reader and requires the original tensors for all61 layers,
all384 routed experts per MoE table, embeddings, final norm and output head.
Native layer/MoE calls borrow packed bytes from those mappings; they introduce
no new weight cache or pager. IQ3_XXS uses the existing GPU decoder's F32 output.
The opt-in worker's `--kimi2-model FIRST_SHARD --kimi2-identity PROFILE/fixture.identity`
mode binds the full pack to its existing slot/plan protocol. The split executable's
`--model FIRST_SHARD` mode runs layers0–1 on the coordinator, layers2–60 on the
suffix owner, applies final norm/output head and performs greedy token selection.
These full-model paths are compiled but have not executed yet.

`prepare_full_profile.py` produced a complete256-token chat prompt with the
actual pinned tokenizer and froze an eight-row prefill window, at most16 greedy
decode tokens, the original memory ceilings and the full shard-manifest identity.
The generic tokenizer `[EOS]` is163585; the Instruct generation config and pack
both designate `<|im_end|>`163586 as the generation stop. The full decode path
uses that pack stop ID. Earlier tokenizer/numerical raw captures are unchanged.
The staged files must finish downloading and pass the published shard SHA256
checks before full-model execution. K7 and F/C/P remain unqualified.

## Shared hardware reservation for the next K7 attempt

The nine shards have now passed their published SHA256 checks. K7 attempts
returned finite full-model logits for eight real prompt positions, but none
completed the required256-token prefill and16-token decode. The captures and
failed/stopped attempts remain in the K7 receipt. K7 and F/C/P are unqualified.

Every model run, operator probe and heavy CPU/RAM/NFS fixture must use the
canonical shared hardware lease. Both host keys also cover CPU-only model work.
Ordinary source editing and builds may proceed without a model reservation.

`full-job.sh` requires a reservation owned by its canonical parent process,
starts the existing coordinator, suffix and two expert workers, and waits for
their local and remote cleanup before returning. It makes no service changes.
The next profile uses the corrected CPU-pool watchdog clock, existing-protocol
keepalives, and `GPU_MAX_HW_QUEUES=1` in the K2 HIP processes. The one-queue
layer12 replay was byte-identical to the default-queue HIP replay. This is not
full-model or performance qualification.

Run from Lucy only after coordinated hardware availability, using a fresh
capture directory:

```sh
uv run --no-project --with pydantic python /mnt/code/projects/Strata/tools/hardware_lock.py run \
  --resource lucy.gpu0 --resource astor.gpu0 --owner kimi-k2 \
  --thread d7f95b1d-9358-49ce-8e05-b44115b911b6 -- \
  bash /mnt/code/worktrees/Strata/t3-9c8cb6db/tools/kimi-k2/full-job.sh \
  /mnt/storage/models/strata-kimi-k2-evidence/k7-isolated-run1
```

The launcher has passed shell syntax validation only. It has not been executed
under the new lease while Q3 owns the hosts. A busy reservation must not be
released or stolen. Native execution awaits that job's completion and cleanup.
