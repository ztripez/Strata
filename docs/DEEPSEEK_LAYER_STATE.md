# DeepSeek layer state and ownership

Implementation contract for E2 of the
[network inference plan](NETWORK_SHARDED_DEEPSEEK_PLAN.md), 9 November 2026.
The inventory below comes from reading `src/dsv41/ds4_model.cu` and the pinned
2.00-bpw pack. It is static analysis, not a completed replay or network test.

The first implementation uses one sequence, windows of at most 16 tokens,
six routed experts, no DSpark, no vision and no chunk pipeline. Each process
executes the existing DeepSeek layer and expert paths. Transport adapts their
inputs and outputs; it does not implement attention or quantization again.

## Ownership inventory

An owner below is the sole authority for mutation. A replica may be read only
after its source update is committed for the identified window. A session ends
on reset, cancellation that has mutated state, or layer-worker loss. State from
an aborted session cannot be reused by a later request.

| Object in the current engine | Authoritative owner and lifetime | Mutation and visibility rule |
| --- | --- | --- |
| `pos`, `toks`, `ctoks` | Coordinator; one session from initialization through reset. Layer owners keep the canonical token prefix needed by their existing Engram path. | Submit the same window token IDs and starting position to every layer owner. Publish the new committed position only after the final range succeeds. Validate token-prefix identity before using a continued session. |
| Sampling state and output token history | Coordinator; one request/session. | Sample once from acknowledged final logits. A repeated response must not sample or append a token again. Greedy decoding is the initial policy. |
| `Layer::win` | Owner of that layer; one session. | `attn_core()` updates the ring on the owning stream. Earlier positions are replaced in ring order. Do not rerun a duplicate window; abort the session after a partially executed window fails. |
| `Layer::c_kv`, `c_score` | Owner of a KV source layer; one session. | `compress_win()` accumulates an incomplete compression group. Preserve partial groups across windows, including odd starting positions for ratio two. Only the source writes them. |
| `Src::ckv` | Owner of `Src::layer`; append-only completed rows within a session. | Publish exactly rows `[floor(p0/R), floor((p0+T)/R))` after source computation completes. Readers require these rows before their attention. A duplicate update never appends rows again; offset and count are explicit. |
| `Src::ik` | The corresponding KV/index-key source owner; same lifetime and completed-row interval as compressed KV. | The source's `i_wk`/`finish()` path writes quantized index keys. Later indexer layers read the source keys; they do not become owners by recalculating selection scores. Replicas are updated before the destination indexer runs. |
| `Dev::h`, `pre` | Current layer-range owner while executing a window; transferred output becomes the next range's input. | Capture after the last owned layer's stream completes. `h` contains all residual streams; `pre` has the engine's eight-value padded row stride. Preserve bits in same-backend replay. |
| `Dev::topk`, `cand` | Current range's most recent indexer/candidate producer for each row; valid for this window. | Carry both arrays at the cut. They can be consumed by later layers that do not replace them. Carry their valid-count metadata too; never infer validity from uninitialized padding. |
| `cur_topk_n`, `cur_cand_n`, `WinPlan::topk/cand` | Deterministic host planning state for the current session/window. | Reproduce the authoritative `plan_window()` decisions from identical metadata, positions and window rows, or transfer their resolved values. Test equality against the uninterrupted executor. A tensor payload without its valid counts is incomplete. |
| Engram token history and pending `EgBatch` reads | Layer owner executes Engram using the canonical token prefix; history lasts for the session. Pending reads last only through their window. | Supply canonical window token IDs before Engram issues reads. Finish owned pending reads before releasing/reusing staging buffers. Engram tables are immutable backing weights, not mutable KV state. |
| Checkpoints `ck`, `mk`, `slot_held` and parked `SeqState` | Existing local session owner. Network v1 starts fresh and does not expose checkpoint/slot migration. | Reject network requests to restore or bind an unsupported checkpoint/session. Do not reuse a local checkpoint as a cross-process state descriptor. A future restore protocol must restore all owners at one committed position. |
| Expert cache placement, heat, RAM store and staging | Each node's existing Strata local tier manager; process/model lifetime, subject to its existing policy. | Keep local. Do not serialize pointers, eviction state or a new replacement policy into the layer protocol. Logical expert assignment is acknowledged separately from physical residency. |
| `Req/Resp`, `seq_issued`, `layer_of`, CUDA events, mapped handoff flags and scratch | Local process; transient service/window lifetime. | Complete the local service and device dependencies before publishing a boundary. Allocate fresh local objects on the destination. None is a portable state representation. |
| Network session/epoch/window sequencing and retained result | Coordinator owns session/epoch; each range owner records its accepted window sequence and result until acknowledgement. | Validate identity and expected position before mutation. Keep one in-flight window initially. Return a retained result on an identical duplicate; reject a conflicting duplicate or stale epoch. Release retained results after acknowledgement; reject late duplicates after release. |

`reset()` in the existing engine resets the position, tokens, rings and partial
groups. Compressed rows past the logical position become inaccessible rather
than necessarily being erased. A new network session therefore fences the old
identity and valid row ranges; zeroing residual buffers alone is not a reset.

## Dependencies in the pinned pack

The pack declares KV source layers `2, 8, 14, 20`, indexer layers
`2, 8, 14, 20, 24, 28, 32, 36`, and Engram layers `1, 14`. Layers 0–1 use
window attention alone. The source ranges are:

| Source layer | Compression ratio | Layers that read its compressed KV | Layers that select using its index keys |
| --- | --- | --- | --- |
| 2 | 2 | 2–7 | 2 |
| 8 | 2 | 8–13 | 8 |
| 14 | 2 | 14–19 | 14 |
| 20 | 1 | 20–39 | 20, 24, 28, 32, 36 |

These relationships follow the loader's `last_src`, `l.src`, `l.ik_src`,
`l.kv_src`, `l.ix_src` and `l.owns_k` assignments. Preserve them when assigning
layers to hosts. Do not assign a later indexer its own independent copy of the
source computation.

The initial cut to investigate is **before layer 20**: Astor `[0,20)`, Lucy
`[20,40)`. Neither compressed-KV source spans that cut, and both Engram layers
remain on Astor. This reduces the first handoff's state dependencies. It is not
yet a measured memory fit or a valid-cut pass.

A cut **before layer 24** deliberately exercises the harder case. Astor owns
source 20, while Lucy's layers 24–39 need its committed compressed rows and keys
for every window. Before this cut is admitted, source-20 cache updates must be
transferred and acknowledged. The existing local `replicate()` and `run_dev()`
paths show the required semantics; their CUDA events and host-mapped flags cannot
be reused across hosts. Reject this cut until replay tests establish the update
contract. Other cuts need the same dependency audit; storage-shard boundaries
have no bearing on it.

## Boundary capture and restore

Use one separately versioned layer message containing model and pack identity,
plan identity/epoch, session ID, window sequence, token IDs, starting position,
row count, producer range, expected next layer and resolved field descriptors.
Derive all lengths from validated geometry, with a 16-row bound for v1.

Capture `h`, `pre`, `topk`, `cand` and valid counts using the same semantic layout
as `hand_xfer()`. Include every newly completed cross-cut source-cache interval.
The pinned pack has four residual streams of 5,120 FP32 values per token; this
explains the 80-KiB residual field but is not a hardcoded wire layout. Cache row
sizes and formats must come from `ckv_rb()` and `ik_rb()`, not a host struct's
size. Explicitly encode scalar fields and byte order.

`include/strata/core/deepseek_layer_boundary.hpp` supplies the initial host-only
capture codec, with a separate `DS4B` magic/version and a 4-MiB record limit.
It checks complete identity, trusted field sizes, finite residuals and audited
source-cache intervals before returning a new host record. The caller must also
check valid-selection counts against its authoritative window plan. The codec
does not yet capture device buffers, restore another executor, sequence sessions
or perform TCP layer RPC. Its synthetic tests are serialization evidence only.

`deepseek_boundary_io.hpp` adds bounded stream delivery of this record, sharing
the expert protocol's underlying deadline-aware socket transfer routine.
Synthetic one- and sixteen-row records pass local TCP and fragmented-stream
roundtrips. Short/disconnected, oversized, wrong-version and stale-identity
records fail without changing the destination host capture. Trickle-data and
blocked-send tests enforce one deadline for the complete frame. This transports
captures only; plan installation, engine mutation, result acknowledgement and
layer-worker failure semantics are not supplied by this helper.

Before restore, validate the entire message, expected ownership, tensor sizes,
cache source and exact row intervals. No device or session mutation occurs while
validating an incomplete frame. Copy data to local buffers, order copies before
the destination's existing layer execution, then publish completion. A device
failure after mutation aborts the session; it cannot be retried as stateless
expert work.

## E2 evidence required

1. Same-backend uninterrupted execution versus two executors exchanging captured
   boundaries: exact logits, token IDs and persistent-state bytes at two cuts.
2. Prefill windows of 1 and 16 rows, plus windows that straddle a ratio-two group;
   at least 128 decode steps. Include ring wrap at 128 positions.
3. Source-cache replicas match their owner for all committed rows. Valid counts,
   token history and logical positions match at each comparison point.
4. Reset, conflicting/identical duplicates, stale sessions/epochs, missing or
   reordered cache updates and cancellation cannot cause a second KV append or
   expose state from another session.
5. Golden encoding, fragmented delivery, truncated/oversized frames and invalid
   descriptors fail before mutation. A lost layer executor returns a bounded
   request error; no transparent continuation is asserted.

The codec and stream-framing parts of item 5 have CPU test receipts. Native
executor replay and the remaining session/fault requirements are **NOT RUN**.
This document alone does not admit either candidate cut for network execution.

## Native executor probe

The opt-in `Model` probe and `ds4_boundary_adapter.inc` now connect this record
to the existing `forward_serial()`, `layer()` and head paths. The adapter can
resume a second `Model` instance at cut 20 or 24. It validates complete captures
before mutation, refuses duplicates/stale sessions, and poisons a partially
mutated session until reset with a new session ID. Index/candidate sizing reuses
the same host planner as ordinary execution.

`deepseek_layer_replay_test` builds for CUDA and HIP. It is not run by CTest,
because it loads real weights and two full dense executors on one GPU. It uses
six experts, synchronous windows, mapped CPU experts, no GPU expert residency
and no RAM store, so both instances use the same deterministic local paths.
Persistent-state comparison includes token histories, valid-count carry state,
window rings, partial groups and valid source-cache rows. On mismatch it retains
paired logits/state files in the reported temporary evidence directory.

A bounded pilot is:

```sh
env DS4_IQ_Q8K=1 /path/to/deepseek_layer_replay_test /path/to/pack /path/to/first.gguf 1 20
```

The last arguments are decode steps and cut (`0` means both cuts). The full
long-replay command uses `128 0`. Each run also checks malformed captures,
duplicate delivery, pre-mutation cancellation, cancellation after a layer mutates
state, reset and stale-session rejection. Both constructors currently load all
dense layers; this probe cannot establish E3's assigned-only loading, process
isolation, network transfer or capacity claims. Runtime replay acceptance remains
**NOT RUN** until these commands have raw receipts.

## Assigned-only loading audit for E3

Reading `ds4_model.cu` identifies several allocation paths that a layer range
must constrain. Skipping layers in `forward_serial()` alone does not save their
memory. The E2 suffix instance deliberately still uses the complete loader.

- Keep complete geometry and source-layer metadata, but restrict `load_dense()`
  to owned layers. Its allocations include dense matrices, routers, window rings
  and partial compressors. Load the output head only on the final range owner.
- Restrict the dense-fit calculation and `place_experts()` budgets to those same
  layers. `load_experts_mx()` must neither load nor warm unowned experts. Preserve
  the existing native expert layout, CPU pool and store for owned layers.
- Load the pinned embedding table only on the first range owner. Engram readers,
  batches and table metadata are needed only where an owned layer uses them;
  token history still follows the session contract on every owner.
- Allocate compressed KV and index keys for locally produced or consumed
  sources. At cut 24, the suffix needs a replica of source 20 without loading
  source 20's dense weights or running its compressor. At cut 20, earlier source
  caches are unnecessary on the suffix.
- Audit reset, checkpoints, lookahead and destruction for unallocated layers.
  Initially reject unsupported checkpoint, multi-sequence, chunk-pipeline,
  DSpark and vision paths rather than allowing them to access absent buffers.

Before admitting a range, retain per-layer allocation receipts and verify that
an unowned layer has no dense, window or expert-arena allocation. Also retain
the before/after device-memory readings, process RSS/PSS and backing-file I/O.
Read-only mappings and metadata access are distinct from resident weight pages;
report them separately. These are implementation and measurement requirements,
not evidence that assigned-only loading already works.

The initial experimental implementation now exposes `Options::layer_begin`
and `layer_end` when network support is built. Their default retains the complete
loader. The range probe accepts `[0,20/24)` or `[20/24,40)` with one synchronous
GPU, six experts, mapped CPU experts and disabled expert cache/store, DSpark,
vision and chunk prefill. It refuses a boundary role/cut that differs from the
loaded range. A prefix must request zero local logits and hand its capture to
its observer; a suffix accepts `replay_boundary()` only. Native all-layer slots,
checkpoints and checkpointed feed are unsupported and refused.

The opt-in native harness also accepts a fixed GPU expert budget through the
existing allocator. This extends the original zero-residency probe profile;
its live loading/tier/arithmetic qualification is separate from the CPU-only
E2/E3 receipts below. See [the fixed-budget option](DEEPSEEK_LAYER_RPC.md#optional-fixed-local-gpu-expert-residency).

`deepseek_layer_range_load_test PACK BEGIN END` checks the loader's per-layer
allocation invariant and these API rejections. An optional final `EXPERT_VRAM_MIB`
argument exercises fixed-budget loading or refusal; this adds allocation evidence,
not native inference or fresh loaded-artifact hash verification.
Cut-20 load-only probes exited 0
on Astor `[0,20)` and Lucy `[20,40)`. Their unowned layers reported zero weights,
window rings and expert arenas. Full raw receipts and the measurement limits are
in the [historical fabric checkpoint](DISTRIBUTED_FABRIC_STATUS.md#historical-e3-assigned-only-allocation-probe).
This is not replay or connected-inference acceptance. The cut-24 range loader has
build evidence only; its real allocation and cross-cut state tests remain open.

## Assigned-range replay oracle

`deepseek_layer_partition_test PACK FIRST_GGUF CUT [DECODE_STEPS=128]` loads a
full reference plus independent prefix and suffix instances on one backend.
The prefix emits its own boundary; the test compares that capture with the
reference before passing it to the suffix. It compares finite, exact logits,
token histories and both owners' persistent rings, compressors and source-cache
rows after every window. Prefill uses one then sixteen rows; the default 128
decode steps cross the window-ring wrap. A partial suffix cancellation requires
all three instances to reset into a fresh session before continuing.

The diagnostic `boundary_state(..., prefix=true)` selects the prefix region of
the full oracle or an assigned prefix. The default selects the suffix. Assigned
executors refuse a diagnostic view containing unowned layers, and failed views
leave the caller's buffer unchanged. These bytes are not a checkpoint format.

Captures zero inactive selection slots and the padded mixing coefficients beyond
the model's residual-stream count. The native planner declares the live counts;
`hc_pre_norm_kernel()` reads only the stream count. This removes stale scratch
padding from the portable record without changing live fields or device buffers.
No numerical tolerance is relaxed. The assigned replay probe compares all emitted
bytes, including this deterministic padding.

Allocation bounds alone do not prove a selected cache row is visible to the
current query. The adapter also checks live top-k indices against the last
producer's completed compression groups, and candidate block IDs against their
producer's completed blocks. Both checks occur before applying an incoming
capture. The replay fixtures include indices inside the allocated context but
beyond those committed intervals and require unchanged state after rejection.
Their runtime receipts remain open.

CUDA/HIP builds pass. Runtime assigned replay remains **NOT RUN**. The test uses
three native instances on one GPU; it does not establish separate-process TCP
inference, mixed-backend quality, cluster memory fit or throughput. Its source and
binary identities are separate from the already-queued two-executor E2 pilot.

## Layer delivery state

`DeepSeekLayerSession` owns delivery and acknowledgement state; the native
`Model` continues to own all attention/KV state. Only one window may await ACK.
An exact duplicate before ACK returns the retained result without another
executor call. A conflicting window, next window before ACK, duplicate after ACK
or invalid acknowledgement aborts the session. Executor failure also aborts,
including failure after partial native mutation. Creating another delivery object
does not reset native state or fence reused session IDs; the worker integration
must do both before accepting a new session.

The `DS4R` result record carries the original request identity, actual committed
position, explicit logit-row count and finite FP32 logits. Its separate version
and 16-MiB limit allow up to sixteen rows without changing the expert protocol.
Shapes come from the trusted model layout. The result is retained only until an
exact ACK; no local layer fallback or hidden retry is provided.

GPU-free delivery/result tests and Lucy's address/undefined-behavior sanitizer
checks pass. They are mock-executor evidence. These helpers do not yet implement
plan installation, a TCP layer service or a connected native inference run.
