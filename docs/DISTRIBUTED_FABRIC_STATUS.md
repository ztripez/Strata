# Distributed fabric checkpoint

Updated 9 November 2026. Source branch `feat/distributed-experts`, HEAD
`3c4f7ff8` before this status update. Continue the approved fabric architecture;
Strata's existing node-local staging remains authoritative.

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
the store, but an independent full-file checksum pass on Astor is not claimed.
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
DeepSeek inference is not qualified. A DeepSeek adapter using the existing
kernels is present as an unvalidated working-tree draft; Qwen remains the last
qualified worker family. No new cache or transport is required by that adapter.

## E0/E1 implementation checkpoint

The [execution plan](NETWORK_SHARDED_DEEPSEEK_PLAN.md) is being implemented.
The [layer-state inventory](DEEPSEEK_LAYER_STATE.md) records source ownership,
lifetimes, update ordering and the candidate cuts before layers 20 and 24.
Neither candidate has passed executor replay or runtime memory-fit checks.

The host-only `DeepSeekWorkerGeometry` guard passes twelve synthetic metadata
cases. The new `DS4B` layer-capture codec passes roundtrip and malformed-record
checks for one and sixteen rows on Lucy and Astor. Address/undefined-behavior
sanitizer checks also pass. The three GPU-free CTest targets (geometry, capture
codec and the existing expert protocol) pass. These results do not establish
device-state capture, a second executor's replay or network layer inference.

The rebuilt HIP worker passes the existing eleven-case Qwen expert comparison
in default and tiered modes. The tiered probe reports five GPU, five RAM and
fifteen mapped assignments. Lucy's post-test Qwen hash matches the pinned stock
hash. Full-model Qwen regression/fault repetition and paired performance remain
open; concurrent integrity hashing prevents performance conclusions here.

The DeepSeek draft includes an effective six-expert identity receipt and a guard
against any layer resolved to a different count before network READY. These are
built but not yet live-qualified on a primary. Its expanded real
fixture is prepared for four format-covering layers, six cases each, including
sixteen rows and ninety-six assignments. Numerical acceptance is **NOT RUN**
until its raw results are collected. The first launch was invalidated by a
supervisor change that terminated the worker during hashing, before arithmetic.
The retained retry separates each node's thirty-minute identity budget from the
five-minute fixture budget; tolerances and inputs remain unchanged.

Evidence and source snapshots:
`/var/tmp/strata-deepseek-e1-20261109/`, with the active attempt in `retry1/`.
The worker, real parity fixture and primary-network draft remain uncommitted
pending numerical validation. E0's independent complete-artifact checks on
Astor and the measured layer split remain incomplete. No E0/E1/E2 gate pass is
claimed by this checkpoint.

## Other unfinished work

`tools/fabric/orchestrator.py` and `docs/KIMI_K3_FEASIBILITY.md` are untracked
work inherited from an earlier continuation. Their presence does not establish
G2 or G8 acceptance. Kimi implementation remains gated by DeepSeek qualification.
