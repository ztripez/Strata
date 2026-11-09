# Network-sharded DeepSeek inference with Strata

Execution plan, 9 November 2026. Status: proposed acceptance contract; no new
runtime pass is asserted by this plan. Audience: implementers and reviewers of
`ztripez/Strata`, branch `feat/distributed-experts`.

This plan extends [the fabric specification](DISTRIBUTED_FABRIC_SPEC.md) with
cross-machine **whole-layer execution**, alongside distributed MoE experts. It
replaces the assumption that Lucy must host every dense layer. The existing
[checkpoint](DISTRIBUTED_FABRIC_STATUS.md) remains the record of measured results.

## 1. Deliverable and scope

Run the complete pinned DeepSeek V4.1-Flash ISTA-DAS 2.00-bpw model across Lucy
(CUDA RTX 3080) and Astor (HIP RX 6800), backed by immutable GGUF files on NFS.
Each node runs its assigned layers and uses Strata's existing local expert
VRAM/RAM/file-backed execution. A layer may send selected expert computation to
the other node. An offline orchestrator discovers resources, produces a valid
plan, installs it, starts inference and records the result.

The final demonstration must combine all of these in **one actual inference
run**: cross-node layer handoff, nonzero remote expert execution, bounded local
expert residency, correct model output, and a replayable orchestrator manifest.
Separate layer-only and expert-only demonstrations are prerequisites, not the
combined milestone.

Initial scope is two nodes, text only, one sequence, greedy decoding, FP32 network
activations/results, six routed experts per token, DSpark disabled, and unchanged
weights/router/quantization. Start with windows; chunked prefill is a later gate.
Existing defaults remain unchanged. No new local cache manager, quantization
scheme, learned router, transport replacement or dynamic migration is included.

The first target is distributed 2.00-bpw residency, qualified as capacity grade
C1 below. Model-size scaling is a separate C2 qualification. A complete 2.25-bpw
pack is a possible follow-up, subject to compatibility and measured fit; this
follow-up sits outside the initial 2.00-bpw qualification.

## 2. Starting evidence and gaps

- The seven pinned files are on
  `/mnt/storage/strata-models/deepseek41-ist-das-2.00bpw`; source revisions, sizes
  and hashes are in `download-manifest.json`. Lucy verified every file through
  NFS. Astor's independent complete-file verification is still required.
- `maya-ds4` was made with the plain official pack recipe. Astor's stock HIP
  engine returned “The capital of France is Paris.” at EOS. This one short smoke
  does not establish broad output quality or long decoding.
- Lucy's stock full-model loads failed with its then-available VRAM. That blocks
  that solo configuration, not a placement containing only some dense layers.
- HIP IQ expert, router and focused Q5_K/Q8_0 dense tests passed. The general
  GGUF fixture reproducibly failed IQ4_XS/Q4_K token/window bit identity. That
  expert pair is absent from the selected pack. The full dense suite timed out;
  it must not be reported as passing.
- Qwen has bounded expert offload and a tiered worker proof. Full-model numerical
  quality remains open. See [the Qwen report](DISTRIBUTED_EXPERTS.md).
- The working tree contains a **draft, unvalidated** DeepSeek expert adapter,
  geometry guard and parity test. CUDA and HIP builds completed; real expert
  parity has not run to acceptance. These changes are not an accepted
  DeepSeek network implementation.
- The untracked `tools/fabric/orchestrator.py` is a draft. Its predictions and
  schemas are not acceptance evidence and must be audited before reuse.

The numerical experiment was deferred when this plan was requested. Its owned
CUDA worker was stopped. Preserve the patch and logs in
`/var/tmp/strata-deepseek-network-20261109-98KacO/`. Already-started build receipts
may be collected, but do not start new implementation campaigns while reviewing
this plan.

## 3. Execution architecture

### Responsibilities

**Coordinator:** initially Astor. Owns API/tokenization, sequence identity,
positions, sampling policy and the active immutable plan. It does not dispatch
every layer through a separate central scheduling hop.

**Layer owner:** executes its layer range's dense/attention/shared-expert/router
operations, maintains the declared attention state, and combines local and remote
routed-expert contributions. The model router stays unchanged on this node.

**Expert executor:** evaluates the acknowledged `(layer, expert)` assignments,
using the selected model adapter and existing node-local execution/storage paths.
Expert ownership and actual residency are separate concepts.

**Orchestrator:** control plane only. Discovers capabilities and available
resources, chooses layer ranges and expert destinations, verifies artifacts,
installs a plan and supervises processes. It is absent from the per-layer hot path.

**NFS server:** supplies immutable weights on startup and cold reads. Normal hot
inference between compute nodes sends activations/results, not expert weights.
Cold backing-file reads are expected and must be measured separately.

### Build the architecture in three configurations

1. **Expert-only:** Astor owns all layers/dense state; Lucy evaluates a small
   fixed expert set. This is the immediate adapter/parity milestone.
2. **Layer-only:** give each node one contiguous layer range and keep expert work
   local to each range. Choose the cut from measured memory feasibility. A cut
   such as Astor `[0, 24)` and Lucy `[24, 40)` is a candidate, not a promised fit.
3. **Hybrid:** keep a functionally tested layer split and allow expert RPC from each layer
   owner to the other node. Use one expert delegation hop; recursive delegation
   and cyclic dependency chains are forbidden.

The coordinator role need not coincide with the final-layer/output-head owner.
Initially, return final logits to the coordinator for deterministic sampling.
A distributed sampler is unnecessary for the first implementation.

### GGUF shards are storage, not execution partitions

Use existing GGUF tensor offsets to read assigned weights. Do not interpret
`00001-of-00004` as a layer range. Do not repack or requantize just to choose an
execution cut. Metadata maps every layer/tensor to immutable source bytes;
execution ownership maps those layers and experts to nodes.

## 4. Reuse and implementation boundaries

- `src/dsv41/ds4_model.cu`: existing layer execution, `forward_serial()` handoffs,
  CPU service lane, attention/KV dependencies and expert-store hooks.
- `src/dsv41/ds4_model.hpp`, `ds4_main.cpp`: opt-in configuration and CLI.
- `src/dsv41/ds4_gguf*`, `ds4_iq*`, `ds4_tq*`, `ds4_pool.hpp`, `ds4_store.hpp`:
  authoritative DeepSeek arithmetic and local staging. Do not replace them.
- `src/core/network_experts.cpp`, expert wire/identity helpers and
  `src/program/expert_worker.cu`: existing expert RPC and lifecycle. Keep Qwen's
  supported path intact and independently tested.
- A small DeepSeek network-adapter module should own pack/source validation,
  network request assembly and layer-state serialization. Keep scheduling and
  protocol policy out of the already-large model implementation where practical.
- `strata_control/` already owns hardware/model-fit functions. Reuse their
  discovery results instead of inventing another GPU detector. Audit the draft
  fabric planner before deciding its final location/schema.

Document the missing capability before adding any storage or caching component.
The network worker must call the existing DeepSeek expert and dense execution
paths through a thin adapter. During E1/E2 review, map each adapter call to its
authoritative local path. Reject duplicated quantization arithmetic, replacement
policy, paging or layer-execution logic introduced solely for remote execution.
Adapter code may validate, frame, copy and dispatch; it must not become another
execution engine. Apply any required local execution fix to the authoritative
path and qualify both local and network callers.

In particular, DeepSeek's `--ssd-cache` budgets a **RAM expert store**; it is not
proof of a local SSD file cache. A local GGUF mirror or shard copy needs verified
content and an explicit source mapping. Do not silently add sparse-file caching.

## 5. Contracts that must be specified before implementation

### Artifact and placement manifest

Record model family, pinned revisions, every source file's byte size/SHA-256,
pack/tensor-map digest, tokenizer, six-expert setting, arithmetic policy, protocol
versions, source and binary hashes, and a deterministic plan ID/epoch.

For each node record operations/quant types it actually passed, layer range,
embedding/head owner, Engram placement, expert destinations and replicas,
VRAM/RAM/pinned-memory limits, backing files and fallback policy. Paths locate
artifacts; equal paths do not establish equal bytes. Reject overridden pack views
unless their exact source bytes and arithmetic have been qualified.

Require explicit `--experts 6` on the solo reference and every layer-owning
engine. At startup, every participant must emit its role, effective routed-expert
count, model/plan identity and binary identity. An expert-only worker must verify
the six-expert model contract even when a request contains only a subset of the
six assignments. The coordinator must compare these receipts before READY and
reject a missing receipt or count other than six. A CLI argument or manifest
value alone does not prove the effective runtime setting; capture the resolved
engine value after configuration/default handling. Keep these receipts with
every quality and performance run.

Budgets must include dense weights, attention/KV state, expert arenas, staging,
service buffers, temporary peak allocations and a measured safety reserve.
Report OS page cache and process RSS/PSS separately; do not double-count them.

### Layer boundary

Reuse the semantic state already transferred by DeepSeek's local multi-GPU
`forward_serial()` path: `h`, `pre`, `topk` and `cand`. For this pack, `h` is four
residual streams of 5,120 FP32 values: **80 KiB per token**, before other fields.
Derive sizes from the model and validate them; do not hardcode an 80-KiB wire ABI.

Audit shared/compressed KV and indexer source dependencies across every proposed
cut. Transfer or place the required cache updates explicitly. “KV stays with its
layer” is not sufficient for DeepSeek's cross-layer dependencies. Initially
reject cuts whose dependency transfer is not implemented.

Before accepting a cut, enumerate every persistent session state object in a
state-ownership table: semantic identity, shape/dtype, authoritative owner,
lifetime (creation through reset/destruction), writers/readers, mutation order,
and transfer/acknowledgement rules. Include shared/compressed KV, indexer state,
cross-layer caches and sequencing/deduplication records. If a replicated view is
needed, declare its source owner and visibility rule explicitly. No state may
depend on implicit shared memory between the CUDA and HIP processes.

Every request carries protocol/model/plan identity, session and sequence ID,
request ID, token positions, prefill/decode mode, expected next layer, bounded
shapes/dtypes/lengths, and the required state tensors/cache-update descriptors.
No raw process structs, pointers, CUDA events or mapped-memory flags cross nodes.
Use existing framing/I/O helpers; define a separately versioned layer protocol
rather than reinterpreting existing v1 expert messages or reserved fields.

### Completion and failure

- Expert RPC is stateless: validate exact assignment/token coverage, finite sums
  and request identity before merge. On failure, recompute only the missing work
  through a declared local path, or fail visibly if that path is unavailable.
- Layer execution changes session/KV state. Duplicate delivery must not append
  the same token twice. Keep bounded per-session sequencing and completed-result
  records until acknowledgement. A duplicate returns the same result or an
  explicit error; it never executes the state mutation again.
- First release: a lost layer worker aborts the request within the declared
  coordinator deadline. Do not claim transparent continuation. Rebuilding state
  by replaying the prompt and committed tokens is a later, separately tested
  recovery feature.
- Plan installation is transactional: validate files and budgets, load/identify
  resources, receive READY from every participant, then publish the plan. Partial
  installation never receives traffic. Switch plans only between requests after
  sessions drain; fence old epochs and stale responses.
- Enforce backpressure with bounded in-flight work. Cancellation stops publication
  and drains/fences late responses. NFS hard-mount stalls may outlive a process
  deadline: the coordinator must still return an error and quarantine the node;
  do not promise immediate cancellation of blocked kernel I/O.

## 6. Success gates and dependency order

Use `NOT RUN`, `PASS`, `FAIL` or `BLOCKED` for each gate. Each pass requires raw
receipts, not a progress message. Gates E0–E8 below refine, rather than silently
claim completion of, the original fabric gates G0–G7.

### E0 — Reproducible model, machines and budgets (G0/G3)

**Work:** verify actual model/pack mappings on both nodes, stable before/after
hashes, backend/build identities, hardware, available memory and link paths.
Inventory dense and KV dependencies per layer. Preserve the corruption history.

**Pass:** a replayable two-node inventory and tensor-byte manifest; feasible
per-node budgets for at least one candidate layer split. No new integrity error.
Solo and participant startup receipts confirm the effective six-expert contract;
configuration mismatch fails before plan activation.
Unfinished physical DIMM/multi-pass memory checks remain explicit final-hardware
qualification limits; the user's earlier decision permits bounded development.
Lucy need not fit all dense layers. Astor's stock smoke is the starting solo
reference; refresh it for the pinned campaign and capture actual output/logits.

**Stop:** changed bytes, unsupported required kernel, or no feasible candidate
under the declared budgets. Do not silently change the model or quantization.

### E1 — Expert adapter and Qwen non-regression (G1 / part of G4)

**Work:** finish the opt-in DeepSeek adapter against the existing expert protocol.
Test actual CUDA/HIP kernels and CPU references across every expert-format pair
in the pinned pack: singleton, weighted sum, repeated expert, sparse rows,
maximum supported token count, invalid ownership, nonfinite/stale/short results.

**Pass:** all fixtures finite and coverage-complete; preserve the existing GPU
relative-L2 <= 1e-3/cosine >= .99999 and CPU max-absolute <= 2e-3/cosine >= .9995
criteria. Reject unsupported geometry rather than broadening the implementation
without tests. Qwen default/tiered parity and failure tests still pass; no Qwen
model/runtime default changes. Its paired repeated regression must stay within
the specification's 5% performance tolerance on the matched workload. Reproduce
and track the unrelated HIP bit-identity
failure; do not claim universal backend compatibility.
Review confirms that network expert computation calls authoritative DeepSeek
execution paths, with no second arithmetic or local tier manager in the adapter.

### E2 — Layer-boundary contract and local replay

**Work:** capture the existing local boundary state at a chosen cut; restore it
into a second executor using the same backend first. Establish explicit ownership
of every persistent state object using the section 5 ownership table, and an
oracle for every serialized field. Audit dense execution reuse as well as expert
execution reuse.
The [state inventory](DEEPSEEK_LAYER_STATE.md) records the initial source audit
and candidate cuts; its static findings are not a replay pass.

**Pass:** same-backend capture/restore is byte-identical for logits/tokens and
state under deterministic settings, for prefill windows and >=128 decode steps.
At least two valid cuts pass dependency checks. Invalid shapes, sequence/epoch
mixups, duplicate tokens and missing cache updates are rejected. Wire golden-byte,
fragmentation, size-limit and cancellation tests pass before a network layer run.
Every persistent state object has an owner, lifetime and mutation order; tests
cover reset, cancellation, duplicate delivery and stale-session access to those
objects. Any unresolved state dependency blocks E3, regardless of transfer speed.

### E3 — Real network layer-only inference

**Work:** Astor and Lucy own disjoint contiguous ranges covering all 40 layers,
with one layer cut, local MoE execution and persistent TCP boundary exchange.
Start with windows, not speculative decoding or interleaved layer ownership.

**Pass:** no node loads unassigned dense layers; measured VRAM peaks fit each
budget. Real nonzero boundary traffic occurs. Two distinct prompts each produce
at least 128 decode tokens without errors or unbounded queues, plus a separate
normal-EOS smoke. Collect same-prefix logits for E5; this gate establishes
functional execution, and the quality-qualified label waits for E5. A short EOS response alone
cannot substitute for long-decode testing. Restart and stale-session rejection
work; layer-worker loss produces a bounded clean error.

### E4 — Hybrid network layers plus remote MoE (G4)

**Work:** enable expert RPC on a functionally proven E3 layer-only plan. Every layer owner
retains unchanged routing and delegates only to READY destinations. Exercise both
CUDA-to-HIP and HIP-to-CUDA expert execution across the accepted fixtures.

**Pass:** the same two-node inference requests demonstrate a real layer handoff
and nonzero expert RPC, with no missing/double contributions. Require a controlled
fixture where one token's picks span local GPU, local CPU/file-backed and remote
execution. Zero-expert-ownership controls preserve logits/tokens for their matched
layer configuration. Run the two long-decode prompts and normal-EOS tests again,
with no fallback in the healthy case. Local expert fallback and layer clean-abort
behaviors pass their separate failure contracts.

### E5 — Numerical and task-quality acceptance

**Freeze before fresh qualification data:** unchanged pinned weights, tokenizer,
six-expert routing, deterministic sampling, all tolerances and test prompts.
Expert tolerances in E1 are not substitutes for full-model quality.

**Proposed full-model acceptance:** >=1,000 teacher-forced target positions across
>=5 held-out prompts and contexts spanning short inputs through 2K; all logits
finite; mean KL(reference || candidate) <= .01 nat and >=99% next-token top-1
agreement. These are acceptance targets, not existing measurements. Establish the
solo reference repeatability floor first. No post-result threshold relaxation.
Compare same token prefixes; free-running divergence is recorded separately.

**Pass:** both layer-only and hybrid plans meet the frozen thresholds, the
same-backend serialization controls are exact, and a frozen set of eight
exact-answer/tool-independent tasks retains every baseline pass. Failure blocks
quality and performance acceptance until explained and fixed. Capture intermediate
boundary/expert errors to distinguish transport defects from arithmetic drift.

### E6 — Offline resource-aware orchestrator (G2/G7 control plane)

**Work:** validated inventory -> feasible layer cuts -> expert/tier assignments ->
manifest -> transactional install -> execute -> observe -> saved results.
Use measured CPU/GPU/host/file-backed costs and per-layer boundary latency/bytes.
Account for two expert network crossings per delegated contribution, layer-cut
traffic, final logits, queues and cold reads. Avoid adding measured totals and
subtracting unmeasured “overlap” as a fictional speedup.

Generate `speed`, `capacity` and `balanced` candidates. Fit constraints are hard;
rankings use an explicitly labeled cost model. Validate on held-out traces, not
the trace used to rank the plan. The draft planner's constants are not calibrated
measurements. No per-token central scheduling is allowed.

The [resource collector and draft audit](NETWORK_FABRIC_INVENTORY.md) record the
read-only inventory boundary. Its uncalibrated observations do not admit plans.

**Pass:** an identical inventory/model/workload snapshot reproduces the same plan
ID; infeasible or capability-incompatible plans are rejected before READY. The
orchestrator reproduces the E4 static hybrid plan or a better admissible plan,
shows measured predicted-versus-observed costs, and safely refuses/drains/restarts
when node availability changes. No in-request plan mutation is required for v1.

### E7 — Capacity, speed and fault campaign (G5/G6)

**Cases:** best admissible solo Astor baseline; rebuilt controls; expert-only;
layer-only; static hybrid; orchestrated hybrid. All retain the same quantization,
expert count, context and numerical acceptance. A configuration that cannot start
is recorded as infeasible, never assigned zero throughput for a speedup ratio.

**Capacity-functional pass:** complete 2K-context, >=128-token tests within declared
per-node GPU/host/pinned budgets, without blanket copies of unassigned dense
weights or permanent copies of every remote-owned expert. Count file cache,
replicas and staging separately. Hashing can touch the entire artifact; that is
not permanent model residency.

**C1 — Distributed residency:** the complete 2.00-bpw model passes the functional
and E5 quality gates with materially lower per-node resource requirements than
the admissible solo reference. Before measurement, freeze the constraining
resource (peak VRAM or host memory), the numerical reduction target and all
per-node limits. Report every node's peaks and aggregate costs so moving pressure
to the peer or page cache is visible. Reduced backing I/O alone is an efficiency
result, not proof of reduced residency or larger-model capability.

**C1 usable pass:** C1 plus the specification's target >=5 warm decode tokens/s
at the stated 2K-context workload. If throughput fails, retain functional C1
evidence without claiming usable capacity acceptance. Measure whether cold-read
bytes/bandwidth prevent the target before optimizing scheduling.

**C2 — Model-size scaling (separate follow-up):** a complete larger or
higher-fidelity model passes its pinned quality and functional gates across the
cluster within fixed per-node limits, while no individual node can run that same
configuration acceptably under those limits. A compatible 2.25-bpw DeepSeek pack
is a candidate after 2.00-bpw qualification, not a promised outcome. Freeze its
artifact identity, context, expert count, budgets, quality reference, startup
deadline and acceptable throughput before any comparison. Test solo execution
on both Lucy and Astor with supported local offloading enabled; a GPU-only load
failure does not establish solo infeasibility. Preserve whether each solo case
fails fit, quality, deadline or throughput. A cluster pass requires the same
quality contract and >=5 warm decode tokens/s at the stated 2K workload. Report
C1 and C2 independently; a C1 pass never implies C2.

**Speed pass:** >=30 requests per case, >=5 held-out prompts, >=3 interleaved
rounds; warm median decode >=10% faster than the best admissible same-quality solo
baseline, bootstrap 95% interval excluding zero improvement, and <=10% p95 TTFT
regression. Report request-level pairing, contexts, warm/first-use state, p95/p99
waits, all-node costs and total wall time. Fixed-token replay measures comparable
work; free-generation task runs remain a separate quality check. First-use is not
proof of physical cold storage. Do not drop system-wide caches or time against a
concurrent download/build.

**Failure pass:** kill/disconnect/stall each owned test worker, corrupt/truncate/
replay a response through a test relay, restart with a wrong model or stale plan,
exercise cancellation and backing-read failure. No stale merges, double KV append,
false successful completion or unbounded coordinator waits. Expert fallback must
use declared accessible weights; layer failure must follow its clean-abort policy.
Do not disrupt the household NFS service to inject faults.

### E8 — Reproducible two-node release checkpoint (G7)

**Pass:** E0–E7 have explicit receipts; numerical, C1-usable and speed gates
pass independently; one command can discover/plan/install/run the qualified
configuration; opt-in flags, stop/restart, rollback and failure behavior are
documented. Evidence is mirrored with a verified checksum manifest. Original
Qwen and solo engine paths still pass their controls. No arbitrary N-node or Kimi
execution claim follows from this two-node result.
Report C2 as `NOT RUN`, `PASS`, `FAIL` or `BLOCKED` separately. It is not required
for the initial 2.00-bpw checkpoint; a model-size-scaling claim requires its own
complete C2 receipts.

Complete the agreed outstanding hardware checks before claiming the original
G0/G7 release gates. While those are pending, label the result an experimental
two-node checkpoint and retain the hardware-stability limits explicitly.

## 7. Experiment bounds and evidence

Record commands, source diff/commit, binaries, full artifact manifest, placement,
backend/driver, seed, thread count, contexts, generated/forced token IDs, logits,
per-layer traffic/timings, remote coverage/fallbacks, VRAM peaks, RSS/PSS, pinned
arenas, file-cache residency and attributable backing I/O. Store large logs outside
Git and a short accurate checkpoint in Git.

Before each campaign declare startup, per-request and wall-clock deadlines. The
initial functional budget is 30 minutes per cold startup/request; fixture compute
is capped at five minutes after identity verification. Estimate the full 30-request
campaign from a pilot and cap it at eight hours. Report TIMEOUT with the unfinished
phase; do not silently increase budgets or call a timeout a numerical failure.
One retry is allowed only after recording a concrete cause and changed condition.

For the new E4 static controls, declare 45 minutes per expert-enabled primary
startup before collecting data. These primaries hash the four GGUF expert shards
for the expert contract, then the six mapped files for the layer contract.
Layer-only primaries and expert workers retain the 30-minute startup limit;
each request retains 30 minutes, and the full fixture retains eight hours.
This declaration does not change a running or completed E1/E3 trial's limits.
The worker's opt-in two-hour first-byte idle allowance covers the declared
worst-case sequential startup wait of 75 minutes. Active frame/RPC limits stay
30 seconds. The earlier one-hour fixture was superseded before any native launch;
its interruption exit and sealed inputs are retained.

Any unexplained byte corruption stops runtime qualification. A quality failure
stops performance claims. A failed speed gate does not erase a passing capacity
result, but E8 remains incomplete. Fix transport/ownership/arithmetic defects before
trying compression, alternative Ethernet transports or dynamic migrations.

## 8. Immediate execution order

1. Review/freeze this contract and snapshot the unvalidated adapter working tree.
2. Complete E0 and E1: real DeepSeek expert parity and Qwen regression receipts.
3. Implement E2's local boundary capture/replay and state sequencing contract.
4. Prove E3 layer-only execution using a feasible measured cut.
5. Combine layer handoffs and expert RPC in E4; run the fresh E5 quality suite.
6. Calibrate E6's orchestrator from those accepted paths and held-out traces.
7. Run E7's independent capacity/speed/fault campaigns and publish E8 receipts.

Dense CPU offload, compressed wire formats, automatic placement migration, more
than two nodes and Kimi are later work. They require measured bottlenecks and new
acceptance gates; they are not hidden prerequisites for this deliverable.
