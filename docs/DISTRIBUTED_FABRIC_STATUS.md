# Distributed fabric checkpoint

Updated 10 November 2026. Source branch `feat/distributed-experts`. The stock
smoke below used `3c4f7ff8`; later campaigns retain their own source snapshots
and binary hashes. Strata's existing node-local staging remains authoritative.

Latest control checkpoint, 9 November at 23:29 CET: the native HIP TCP control
passed two distinct 128-step probes and a separate EOS request with exact tokens
and all 270 saved logit rows byte-identical to the ordinary HIP reference.
The real native suffix retained identical duplicate results and rejected
conflicting/late duplicates and stale incarnations. Healthy sessions consumed
IDs 4, 5 and 6 after the negative probes. Killing the owned suffix after the
fourth request's acknowledged prompt made the caller exit with a bounded
disconnect error, without publishing an unacknowledged continuation.
The accepted driver exited 0. The loss request exited 1 and its remote worker's
SSH command exited 255 by design; raw exits and logs are retained. An exact
private-executable check found no remaining native control processes.
Receipts are in `/var/tmp/strata-deepseek-e3-batch-control-20261109/`.
This is a same-backend serialization/lifecycle control, not full E3 or E5 acceptance.

Earlier mixed-host checkpoint at 21:33 CET: E2 passed both cuts
with 128 decode steps; the assigned-only HIP cut20 oracle also passed 128 steps.
The first real mixed-backend layer-only run passed: Astor owned `[0,20)`, Lucy
owned `[20,40)`, and all 130 windows were acknowledged over persistent TCP.
It generated 128 tokens with finite saved logits and zero unassigned dense
allocations. The first response was “The capital of France is Paris.” followed
by EOS; the functional loop explicitly continued after EOS to reach 128 steps.
Both native processes and the acceptance driver exited 0.

The run used native mapped CPU experts, six routed experts and the qualified
stock Q8_K activation policy. All six mapped backing files, parsed pack identity
and pinned executables matched the independent oracle. IPv4 routing and negotiated
link checks used the 10Gb interfaces in both directions. This is one-prompt
functional evidence, not E3 completion, hybrid expert execution, broad quality,
capacity or performance acceptance. Raw receipts are in
`/var/tmp/strata-deepseek-e3-tcp-20261109/`; its `summary.json`, `first-response.json`
and source/binary snapshots are the acceptance record.

A frozen mixed-host two-prompt/EOS/restart campaign began at 23:29 CET in
`/var/tmp/strata-deepseek-e3-mixed-batch-20261109/`. Its result is pending.
Fixed-budget allocation probes and the bidirectional tiered-worker format matrix
are queued behind its successful acceptance. Earlier sections below retain the
preceding checkpoints; separate frozen executables identify each campaign.

## Prepared combined test and resource collector

On 10 November, the new E4 static fixture was sealed in
`/var/tmp/strata-deepseek-e4-static-20261110-idle2h/` and queued behind the allocation
probes and bidirectional tiered-worker matrix. It compares layer-only, empty
expert ownership and real remote ownership with the same 384 MiB native GPU
expert budget on each layer owner. The real placement delegates 128 identities
at prefix layer 0 and suffix layer 20, in opposite directions. Each expert
worker has 20 MiB GPU and 20 MiB copied RAM budgets; remaining ownership uses
the existing native mapped-file CPU lane. Separate expert-process losses must
recompute locally while the stateful layer sessions finish. This is a frozen
experiment, not a completed hybrid gate or an orchestrator-installed plan.

The verifier accepted the three completed mixed-host trajectories and rejected
eight deliberately damaged copies, including wrong identity, stale sequence,
nonfinite/truncated logits and contradictory metadata. These verifier checks do
not constitute new native inference. Held-out E5 inputs remain separate.

Source inspection found that the worker's frame receive timeout also covered
idle startup hashing. `f4ece324` adds an opt-in first-byte idle allowance while
retaining active frame deadlines and existing defaults. CUDA/HIP builds,
protocol tests on both hosts, ASan/UBSan on Lucy and eight pre-load admission
cases per host passed. The E4 fixture uses a two-hour idle allowance and a
30-second active RPC limit. Native startup integration remains to be qualified;
receipts are in `/var/tmp/strata-deepseek-worker-idle-20261110/`.
The earlier one-hour fixture at `/var/tmp/strata-deepseek-e4-static-20261110/`
was interrupted before any native launch: its idle allowance was shorter than
the declared 75-minute sequential startup bound. Its exit 1, interruption reason
and sealed inputs remain intact. This was pre-launch correction, not a model
quality or numerical failure, and did not change an E1/E3 trial's deadlines.

The [resource collector and draft audit](NETWORK_FABRIC_INVENTORY.md) use Control's
existing probes. Four tests, live read-only snapshots on Lucy/Astor and refusal
to overwrite an existing receipt passed. Astor's first command exited 127 because
`uv` was absent from its SSH PATH; the existing absolute installation succeeded.
Receipts are in `/var/tmp/strata-network-inventory-20261110/`. Observations are
explicitly uncalibrated. No E6 ranking, reservation or READY admission is claimed.

## Qwen: bounded tiered-worker milestone completed

The opt-in worker can acknowledge more experts than fit in GPU memory and use
the native CPU pool for RAM and mapped GGUF experts. All 10,240 Qwen experts
were assigned with 8 MiB expert-VRAM and 8 MiB host-RAM budgets. All three paths
passed the bounded expert fixtures. Full-model functional and failure tests
completed; output-quality qualification remains open. See
[DISTRIBUTED_EXPERTS.md](DISTRIBUTED_EXPERTS.md).

## DeepSeek: stock HIP smoke passed; CUDA primary remains blocked

The pinned ISTA-DAS 2.00-bpw seven-file set (256,816,878,701 bytes) is downloaded
in `/mnt/storage/strata-models/deepseek41-ist-das-2.00bpw`. All seven files
passed their source SHA-256 manifest through Lucy's NFS mount. Astor can read
the store. An independent full-file pass on Astor has now also verified all seven
files against the pinned source SHA-256 manifest and exited 0.
The earlier HTTP downloader failure and the successful Xet retry are retained.

Strata's plain official pack recipe created `maya-ds4` beside those files,
without rewriting weights or changing quantization. It includes Engram,
DSpark metadata and tokenizer. `strata-dsv41 --pack-info` reports all six
referenced weight files present and long enough. Pack construction and
structural checks alone are not inference qualification.

On Astor's RX 6800 (`gfx1030`), the rebuilt stock HIP engine returned:

> The capital of France is Paris.

It exited 0 at EOS after eight tokens, using six routed experts, context 1024,
DSpark disabled, text only, a read-only shipped profile, mapped experts and a
4 GB existing RAM-store budget. It reported 962 experts / 8.60 GB in VRAM,
14,398 experts on the CPU lane and zero failed store reads. This establishes
one narrow stock-model end-to-end smoke, not broad quality or speed acceptance.
The cold NFS-backed prompt took 119.8 seconds and decode 61.8 seconds; compiler
and fixture work overlapped portions of the run, so these are diagnostic only.

Lucy failed before generation in two stock CUDA configurations: context 1024,
and context 256 with windowed prefill and GPU expert caching disabled. Both
reported a failed 433 MiB allocation. The engine estimated 5.96 GB of dense
weights versus about 6.3-6.4 GB available. The earlier 16.8 GB BF16-layout
figure is not the dense GPU footprint of this pinned quantized pack. No desktop
applications were stopped. This is a current-free-VRAM blocker, not proof that
an otherwise idle RTX 3080 cannot run it.

HIP IQ expert fixtures and router guards passed. The general GGUF fixture
failed its IQ4_XS/Q4_K token-versus-window bit-identity contract; both outputs
individually passed the fixture's numerical-reference check. That expert-format
pair is absent from this pinned pack. The full dense fixture exceeded its
120-second budget; it is not a pass. Focused dense checks and an idle GGUF
repeat are recorded separately in the evidence directory. The focused
`--quick --type 13,8 --kernel default` dense run passed with `SUMMARY fails=0`.
The idle general-GGUF repeat reproduced the same single bit-identity failure.

Evidence: `/var/tmp/strata-deepseek-feasibility-20261109-BA9x8e/` on both hosts.
It retains the invalid missing-prompt launch, both CUDA allocation failures,
pack output, commands/settings, binary identities and raw HIP test results.

G3 is partial: a stock HIP smoke exists, while the intended CUDA-primary stock
baseline remains blocked and backend test limits remain explicit. G4 distributed
DeepSeek inference is not qualified. The DeepSeek adapter below has one bounded
cross-vendor expert pass using the existing kernels. Full-model acceptance is
separate. No new local cache or quantization implementation was added.

## E0/E1 implementation checkpoint

The [execution plan](NETWORK_SHARDED_DEEPSEEK_PLAN.md) is being implemented.
The [layer-state inventory](DEEPSEEK_LAYER_STATE.md) records source ownership,
lifetimes, update ordering and the candidate cuts before layers 20 and 24.
Neither candidate has passed executor replay or runtime memory-fit checks.

The host-only `DeepSeekWorkerGeometry` guard passes twelve synthetic metadata
cases. The new `DS4B` layer-capture codec passes roundtrip and malformed-record
checks for one and sixteen rows on Lucy and Astor. Address/undefined-behavior
sanitizer checks also pass. Five GPU-free CTest targets (geometry, capture
codec, bounded capture delivery, delivery-session/result contracts and the existing
expert protocol) pass. These results do not establish
device-state capture, a second executor's replay or network layer inference.

The new capture-delivery tests pass on Lucy and Astor; address/undefined-behavior
sanitizers pass on Lucy. One- and sixteen-row synthetic captures preserve bits
over loopback TCP and fragmented stream writes. Malformed/truncated frames leave
the destination host record unchanged; trickle receive and blocked send respect
one finite frame deadline. The expert and capture protocols share the underlying
socket I/O routine, and expert-v1 golden-wire tests still pass. These CPU tests
do not install a layer plan or exercise a GPU session across the network.

The new delivery-session tests enforce one unacknowledged window, retained
responses for identical duplicates, terminal errors for conflicts/late duplicates,
and no implicit retry after an executor error. The separate `DS4R` result codec
checks exact identity, actual committed position, finite logit rows and a bounded
byte count. These CPU tests and Lucy's address/undefined-behavior sanitizer run
pass. They use mock callbacks; native KV mutation and TCP layer-RPC integration
remain separate gates. Native capture/replay now also checks selections against
committed source rows, in addition to allocation bounds; its new runtime rejection
cases have not run yet.

The rebuilt HIP worker passes the existing eleven-case Qwen expert comparison
in default and tiered modes. The tiered probe reports five GPU, five RAM and
fifteen mapped assignments. Lucy's post-test Qwen hash matches the pinned stock
hash. Full-model Qwen regression/fault repetition and paired performance remain
open; concurrent integrity hashing prevents performance conclusions here.

The DeepSeek draft includes an effective six-expert identity receipt and a guard
against any layer resolved to a different count before network READY. These are
built but not yet live-qualified on a primary. Its expanded real
fixture covers all eight format pairs with six cases each, including sixteen
rows and ninety-six assignments. Default-policy numerical acceptance failed as
recorded below; the Q8_K forward campaign passed. The first launch was invalidated by a
supervisor change that terminated the worker during hashing, before arithmetic.
The retained retry separates each node's thirty-minute identity budget from the
five-minute fixture budget; tolerances and inputs remain unchanged.

Evidence and source snapshots:
`/var/tmp/strata-deepseek-e1-20261109/`, with the active attempt in `retry1/`.
The worker and real parity fixture now have the bounded forward result below;
the primary-network adapter has not run on a DeepSeek primary. Full E0 acceptance
and peak-allocation measurements during layer execution remain incomplete.
No complete E0/E1/E2 gate pass is claimed by this checkpoint.

The first real E1 default-policy DeepSeek fixture failed its CPU comparison:
GPU relative L2 `3.67e-8` passed, CPU maximum absolute error `0.0565` exceeded
the frozen `0.002` limit. The exact mapping digest agreed on both nodes:
`19ed8c0c20c5801bfba13d00788dae853d1120fd15c30a2bd916e2cfaa54dc6e`.
The remaining numerical cases were not run after that failure. Expert v2 now
negotiates the resolved activation arithmetic; the next campaign uses the stock
`DS4_IQ_Q8K=1` path on both GPU processes. Thresholds and original input cases
are retained. Metadata audit found eight format pairs, so four additional layers
were appended before the new oracle started: 48 cases over 64 owned experts.
The original four layers' seeded cases remain unchanged.

With that existing Q8_K option, Astor's HIP/CPU oracle and Lucy's CUDA worker
passed all 48 cases across all eight format pairs: 64 assigned experts, 960
remote assignments and zero fallbacks. Worst GPU relative L2 was `0.000662`
(limit `0.001`); worst CPU maximum absolute error was `0.000447` (limit `0.002`).
Both cosine limits passed. Eight invalid local requests were refused, and an
opposite arithmetic policy was rejected before READY. Both nodes reported the
same exact mapping digest above; worker, oracle and driver exited 0.

Evidence: `/var/tmp/strata-deepseek-e1-q8k-20261109/`, including raw logs,
`forward-summary.json`, frozen source/binary hashes and the coverage correction.
The reverse direction (HIP worker, CUDA/CPU oracle) also passed all 48 cases,
960 remote assignments and zero fallbacks. Its worst GPU relative L2 was
`0.000662` and worst CPU maximum absolute error was `0.00165`; the same frozen
limits and cosine checks passed. Arithmetic mismatch was rejected before READY
in both directions. Reverse worker and oracle exited 0; raw evidence and
`reverse/summary.json` are retained beside the forward run. These are bounded
expert results, not full-model output-quality, failure-recovery or speed passes.
Every later Q8_K baseline and participant must use and report the same arithmetic.

All seven pinned files passed full checksums on Lucy through an isolated,
read-only NFSv3-over-SSH path. It uses pve01's reachable 10-Gb IPv6 bridge;
the normal NFSv4.2 address still uses its 1-Gb bridge. Server interfaces,
exports and existing mounts were not reconfigured. Temporary mounts, forwarding
commands, byte-equivalence probe and pack-path relocation are recorded in
`/var/tmp/strata-deepseek-e2-20261109/`. These are correctness-experiment paths,
not an inference speed result. Astor's full seven-file check also passed, with
seven `OK` lines and exit 0 in `verify-astor.log`/`verify-astor.exit`.

The native E2 capture/replay probe builds for CUDA and HIP. It uses two `Model`
instances and the authoritative local layers/head, serializes the audited state
between them, and compares exact logits, token histories, carry counts and owned
attention state. Its frozen cut-20 pilot passed one decode step, exact logits
and persistent state, plus cancellation/reset/stale-session checks (exit 0).
Its initial campaign reached position eighteen over three windows. This does not meet
the two-cut, 128-decode-step E2 gate. Both instances still load full
dense weights; assigned-only loading and separate-process network execution
remain E3 work. See [the probe contract](DEEPSEEK_LAYER_STATE.md#native-executor-probe).

## E3 assigned-only allocation probe

The first opt-in range loader has load-only receipts for the proposed cut 20.
It uses the existing native mapped CPU experts, with six experts per token,
synchronous windows, no expert cache/RAM store, no DSpark and no vision.

- Astor loaded `[0,20)`, including the host embedding table and both Engram
  layers, with no output head. Measured dense-weight allocations were
  2,814,286,336 bytes (2.62 GiB); loaded process RSS was 1,802,092 KiB.
- Lucy loaded `[20,40)`, including the output head, without the embedding table.
  Measured dense-weight allocations were 3,047,921,152 bytes (2.84 GiB); loaded
  process RSS was 716,904 KiB. The device reported 3,145,859,072 bytes (2.93 GiB)
  free after loading. This is global device free memory with desktop apps present.
- Each process emitted 40 per-layer receipts. Its 20 unowned layers had zero
  dense-weight, window-ring and expert-arena bytes. Owned expert arenas were
  also zero: each half's 7,680 experts stayed on the native mapped CPU path.
- Unsupported ranges/configurations, execution before session configuration,
  full-model checkpoint/slot calls and checkpointed feed were refused. Reset and
  destruction completed; both load probes exited 0.

Evidence: `/var/tmp/strata-deepseek-e3-range-20261109/`, including exact source
and binary hashes, commands, raw per-layer receipts and machine summaries.
Expert qualification overlapped the loads; loading durations and free-memory
deltas are not paired performance measurements. OS file cache is separate from
process RSS/PSS. The loader warms owned experts through the existing policy;
this is not a new SSD cache or another paging implementation.

These receipts establish that both assigned halves loaded independently.
They do not establish connected inference, exact state replay, numerical
quality, five-token/second usability, C1/C2 capacity acceptance or speed.
Astor's full seven-file checksum pass completed successfully before the E2 pilot
started. The pilot's exact source and executable are retained separately from
the newer assigned-range implementation. The frozen HIP executable is copied to
`deepseek_layer_replay_frozen-pilot` in the E2 evidence directory before rebuilding.
The frozen full campaign in `/var/tmp/strata-deepseek-e2-full-20261109/` passed
on 9 November at 20:24 CET. Both cuts (20 and 24) completed 128 decode steps
through position 145, with exact finite logits, token histories and persistent
state. Three future-selection rejection checks passed before native mutation,
as did partial cancellation/reset and stale-session checks. Native and driver
exits were both 0. This qualifies the same-backend two-cut E2 replay; assigned
loading, TCP execution and mixed-backend quality remain separate gates.

The new `deepseek_layer_partition_test` builds for CUDA and HIP. It compares a
full reference against independently loaded prefix/suffix executors, including
exact handoff fields, logits and both owners' persistent state. Its HIP cut20
campaign passed at 20:50 CET: 128 decode steps through position 145, plus native
cancellation/reset checks. This is an in-process oracle; assigned cut24 and
separate-process TCP execution remain unqualified. Native exit was 0 and the
controller wrote its successful acceptance ledger. The outer zsh exit-recording
command then failed because `status` is read-only; that wrapper failure is
retained separately in `wrapper.exit`/`wrapper-failure.txt`.

## Other unfinished work

`tools/fabric/orchestrator.py` and `docs/KIMI_K3_FEASIBILITY.md` are untracked
work inherited from an earlier continuation. Their presence does not establish
G2 or G8 acceptance. Kimi implementation remains gated by DeepSeek qualification.

## E3 layer-delivery protocol checkpoint

The separate DS4L v1 client/service wraps DS4B captures, DS4R results and the
existing delivery session. It pins the worker executable and complete layer
contract before BEGIN, consumes fresh session IDs before native preparation,
and publishes logits only after exact acknowledgement. A single deadline spans
the complete EXECUTE/RESULT/ACK/ACKED transaction. Failed transactions close the
channel without automatic retry or local layer fallback.

Six CPU protocol/geometry tests passed on both Lucy and Astor. Real loopback TCP
with synthetic callbacks covers duplicates/conflicts, stale sessions, preparation
failure fencing, changed capability/READY receipts, malformed frames, disconnects,
lost acknowledgements and deadlines. Lucy's RPC test passed ASan/UBSan; the
network library built in the CUDA and HIP build trees. See
[the delivery contract](DEEPSEEK_LAYER_RPC.md). Native TCP adapters and connected
inference are NOT RUN; these tests do not advance the E3 inference gate.

## Native TCP layer harness startup

The opt-in `deepseek_layer_tcp_test` builds on CUDA and HIP. Its suffix invokes
native `replay_boundary()` and returns the actual native position/logits. Its
prefix invokes the persistent DS4L client from the existing native boundary
observer. It requires explicit `--experts 6`; native configuration and loaded
artifact receipts verify that effective count on every layer. Six invalid CLI
admission cases fail before loading on both backends. The shared SHA helper
passes the independent SHA-256 `abc` vector and split-span/null-input tests.

The loaded model supplies DS4M1/DS4P1 identities from actual mapped weights and
parsed pack data, excluding only the backing-directory locator. An independent
Python oracle reconstructed expected identities from the previously verified
source SHA manifest and raw pack metadata. This remains an expected-value
oracle until actual native receipts match it; equal peer claims alone do not
qualify source integrity.

A frozen one-prompt, 128-token cross-host campaign is running in
`/var/tmp/strata-deepseek-e3-tcp-20261109/`. Lucy's assigned CUDA suffix loaded
and is hashing its actual backing mappings. The assigned-only cut20 prerequisite
has passed; Astor's HIP prefix will start after the suffix identity/listener
checks pass. Full-file verification, bounded startup/decode deadlines, exact source/
binary snapshots, token/window logs and acknowledged logits are retained.
The passed assigned-only oracle is retained in
`/var/tmp/strata-deepseek-e3-partition-20261109/`. The cross-host campaign has not
passed its final checks.

One long prompt would still be a partial E3 result. A second distinct long
prompt, normal EOS, native restart/loss behavior, same-backend TCP control and
measured memory bounds remain required. E4 hybrid experts, E5 quality and the
orchestrator/capacity/performance gates remain open.

## Bounded multiple-request harness

The newer harness can retain one loaded model for up to sixteen explicitly
listed requests/connections. Each request resets native state, negotiates a fresh
session and writes a separate new evidence directory. Worker session floors
persist across connections; coordinator IDs never decrease within a coordinator
process. A failed connection stays failed in the final worker status, even when
a later fresh request succeeds. No failed inference request is retried.

The new admission checks refuse duplicate output directories, excessive request
counts and invalid deadlines/connection counts before loading. CPU real-TCP
tests cover two successful fresh connections, reset/floor preservation and an
unexplained preparation failure without reusing an old error string. CUDA/HIP
builds and CPU/sanitizer receipts are separate from native multi-request runtime,
which is NOT RUN. This change does not replace the frozen one-prompt campaign.

Two distinct long functional prompts and an EOS smoke were rendered with Strata's
pinned DeepSeek tokenizer and frozen before cross-host output was observed, in
`/var/tmp/strata-deepseek-e3-functional-fixtures-20261109/`. Their token IDs roundtrip
exactly. They are E3 fixtures and are excluded from the held-out E5 quality set.

## Synchronous native expert dispatch preparation

Expert RPC previously covered only the asynchronous native CPU-miss lane.
The synchronous host-routed window path now uses the same small dispatch helper.
It delegates eligible CPU misses, preserves resident GPU picks and sends their
unchanged router weights. Successful remote sums are combined once with native
local results. A runtime delivery failure retains all original CPU picks for
local recomputation, reports the failure and disables that channel. Configuration
failures continue to reject startup.

The actual helper passed real TCP tests against the existing independent mock
arithmetic oracle on Lucy and Astor. These cover contiguous/padded row strides,
nonleading route slots, mixed local/remote picks, sparse token coverage, no-remote
work and retention of fallback work after disconnect, NaN or truncated results.
ASan/UBSan passed on Lucy. CUDA and HIP engine builds passed. This is dispatch-
contract evidence, not live native MoE arithmetic/fallback or an E4 hybrid pass.
Assigned-range admission now permits the existing expert-worker configuration.
The harness requires a shared explicit plan ID for that path and rejects
unqualified per32 arithmetic before native loading. CUDA and HIP engine/harness
builds and twenty CLI admission cases on each backend passed. Numeric-address
validation now precedes loading; placements cannot delegate unowned layers.
Receipts are in `/var/tmp/strata-deepseek-hybrid-admission-20261109/`.
Live combined layer/expert execution has NOT RUN; admission and
build receipts are separate from E4 acceptance. The currently running E3
campaigns retain their older, separately frozen binaries.

## Teacher-forced capture preparation

The native layer harness now accepts a fixed target-ID file for each request.
It records the logit window before each target and consumes that target rather
than its own argmax. This keeps reference/candidate prefixes identical while
retaining free-running generation as a separate mode. It does not select a new
continuation or relax the frozen E5 thresholds.

CUDA/HIP harness builds and twenty-eight CLI admission cases per backend passed.
Live teacher-forced native capture has NOT RUN; no E5 quality pass is asserted.
Receipts are in `/var/tmp/strata-deepseek-teacher-forcing-20261109/`.
The active E3 control and queued mixed-host campaign use their earlier frozen
executables and do not include this change.

## Fixed native GPU expert budget preparation

The assigned-layer profile now permits GPU expert residency through Strata's
existing allocator. The harness accepts an explicit `--expert-vram-mib` limit;
an infeasible fixed budget is refused instead of silently reduced. Nonzero
budgets require an explicit plan ID. Local replacement caching and the RAM store
remain disabled in this bounded profile. Verbose expert-network runs record
per-layer/token local GPU, local CPU and acknowledged remote pick counts.

CUDA/HIP engine and harness builds and thirty-seven CLI admission cases per
backend passed. Live fixed-budget loading/refusal, the same-token three-tier
fixture and combined hybrid arithmetic have NOT RUN. Receipts are in
`/var/tmp/strata-deepseek-local-gpu-budget-20261109/`. No E4, E5, capacity or
performance pass is asserted by these build/admission checks.

## DeepSeek expert-worker tier preparation

The DeepSeek worker now connects the existing bounded GPU/RAM/mapped placement
to its native CPU row functions and CPU pool. CUDA/HIP worker and fixture builds
passed. The actual CPU dispatch helper passed eight exact serial/threaded and
separate/packed-view comparisons plus twenty-eight nonmutating refusals on both
hosts. ASan/UBSan passed on Lucy, with prebuilt backend libraries unsanitized.

Live tiered RPC, all-eight-format arithmetic, worker fallback and combined hybrid
inference remain NOT RUN to acceptance. The new opt-in fixture uses fresh
connections for each immutable layer plan; its seeded cases and tolerances stay
unchanged. See [the tier contract and commands](DEEPSEEK_EXPERT_TIERS.md).

## Same-prefix numerical scorer preparation

The comparison tool now validates complete native teacher-forced trajectories
and scores fixed-prefix logits. An explicit greedy-reference mode preserves and
checks the baseline's own selected continuation. Its analytic tests cover KL
direction, normalization, underflow and first-maximum selection. Artifact tests
reject mismatched consumed tokens, target/window misalignment, incorrect argmax
metadata, nonfinite/truncated rows and incomplete/extra captures.

Eight synthetic test methods passed. Live teacher-forced capture, the frozen
held-out corpus, baseline repeatability and E5 numerical/task acceptance remain
open. See [the comparison commands](DEEPSEEK_LAYER_RPC.md#compare-completed-captures).

The harness's endpoint admission now matches the current IPv4-only TCP transport.
IPv6 literals and hostnames fail before loading. CUDA/HIP harness builds and
thirty-eight admission cases per backend passed; active campaigns retain their
earlier frozen binaries and use IPv4. Receipts are in
`/var/tmp/strata-deepseek-ipv4-admission-20261109/`.
