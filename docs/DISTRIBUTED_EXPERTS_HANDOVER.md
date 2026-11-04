Working branch: feat/distributed-experts (created locally; verify whether pushed)
Source directory on BOTH machines: /mnt/code/projects/Strata
Goal: opt-in, network-distributed expert execution, not a general distributed inference rewrite.

1. Environment and existing setup
The user has two separate Linux computers:

Role	Hardware	Compute backend	Responsibility
Primary	NVIDIA RTX 3080, 10 GB VRAM	CUDA, SM 86	Full Strata inference process: attention, dense layers, router, KV, hot experts and CPU fallback
Worker	AMD Radeon RX 6800 XT, 16 GB VRAM (gfx1030)	HIP/ROCm	Expert-only service: cache resident expert weights, execute selected experts, return contributions
Do not assume there is a local RTX 4090. The 4090 used in earlier work was rented on RunPod.

The same source checkout is mounted at /mnt/code/projects/Strata on both computers. Passwordless SSH already works between them. Do not waste effort cloning another copy or setting up SSH. However, a shared source tree is not shared model data, VRAM, running processes, or a shared build environment. Verify model-pack visibility and paths separately. Verify the actual interconnect speed and latency; a 10GbE-or-better link is the proposed target, not a measured fact in this handover.

Shared-tree hazard: Git state and source modifications may be shared. Only one machine/agent should edit or switch branches. Use separate, node-local out-of-tree build directories and separate ROCm/CUDA environments. Do not have the two nodes run setup.sh concurrently against a shared .venv or build directory.

2. Current state (verified in fork's main source, not yet implemented)
Fork ztripez/Strata exists. The user successfully cloned it and created local branch feat/distributed-experts under /mnt/code/projects/Strata. The transcript does not confirm a remote push or subsequent code changes.
include/strata/core/remote_experts.hpp / src/core/remote_experts.cpp: existing same-process, other-GPU helper tier. Despite the class name it is not network RPC. RemoteExperts::begin() takes float32 activation rows, selected expert IDs and planner metadata; finish() waits and copies results. open() explicitly rejects using the primary device as a helper. It cannot run unchanged as a standalone worker on its own HIP device 0.
src/core/remote_expert_opt.cu and include/strata/core/remote_expert_opt.hpp: grouped GPU expert output is reduced to a weighted partial result per token. This is a useful semantic pattern for network responses. Floating-point accumulation order changes, so exact bit identity is not expected.
src/core/peer_experts.cpp: a distinct local peer-GPU path; also assumes same-process GPU access. Not a network transport.
src/core/verify.cpp: verifier host/pool per-layer handshake and remote reduction/combine integration. Preserve its plan, mask, lifetime, timeout and publication ordering. A remote worker must never leave the GPU waiting forever for an expert result.
CMakeLists.txt: engine sources are built into strata_engine, and CUDA/HIP are alternative backends, not enabled together. HIP uses a CUDA-compatibility shim. Building an independent HIP worker executable is compatible with the two-process approach but requires implementation.
docs/SECOND_GPU.md: documents existing local helper GPU behavior and --remote-expert-opt, not a network feature; the latter performs local weighted reduction.
docs/AMD_HIP.md: gfx1030 (RX 6800 XT/RDNA2) is buildable with community results, but remains incompletely validated upstream. It has no WMMA/hipBLASLt gfx1030 kernels. First prove the unmodified HIP engine works on this actual AMD card.
AGENTS.md, CONTRIBUTING.md and docs/SOURCE_INSTALL.md: follow these before coding. All new behavior must be explicitly opt-in and the existing single-node path unchanged by default.
Important model integrity constraint: Previous TierMoE adapted/corrected GGUF experiments encountered unresolved buffered/mmap vs. direct-I/O read inconsistencies. Do not use those artifacts as golden models or silently repair/refresh them. Establish a trustworthy stock GGUF/pack baseline first, and record model identifiers and content hashes.

3. Target architecture
  PRIMARY NODE — CUDA / RTX 3080                      WORKER NODE — HIP / RX 6800 XT
  ┌─────────────────────────────────────┐              ┌─────────────────────────────────┐
  │ Strata inference process            │              │ strata-expert-worker (new)      │
  │ dense / attention / KV / router     │  persistent  │ isolated HIP runtime, GPU 0    │
  │ primary VRAM cache + CPU fallback   │◄────TCP─────►│ pinned static expert cache     │
  │ NetworkExpertExecutor (new)         │              │ grouped expert kernels         │
  │ weighted combination + accounting  │              │ weighted partial-sum reduction │
  └─────────────────────────────────────┘              └─────────────────────────────────┘
Data plane, per MoE layer: The primary computes its usual router choices, identifies selected experts owned by the AMD worker, and sends only required token activation rows plus (layer, token index, expert ID, routing weight) metadata. The AMD worker executes already resident experts, computes weighted partial sums on the GPU, then returns at most one output vector per token, along with explicit row ownership/completion information. The primary combines this with local GPU/CPU contributions. Expert weights do not traverse the network per token.

Use float32 activation/output payloads at first, because Strata's current RemoteExperts host-facing interface uses float* rows. Do not quietly substitute BF16/FP16 or rely on earlier 16-bit bandwidth assumptions. The exact hidden dimension and expert packing must be derived from the loaded model, not hardcoded from one model family. V1 may initially inherit the existing helper path's k == 10 restriction only if it validates and rejects unsupported configurations; do not silently assume all MoE models have top-10 routing.

The orchestrator is control plane only: it chooses static ownership at startup and publishes an immutable placement table. It is not a third per-layer network hop. The primary and worker talk directly on the inference path.

4. Execution phases and acceptance gates
Phase 0 — Inspect machines and freeze a reproducible baseline
From the primary, identify the worker's working SSH alias and the actual fast-path IP/interface; do not invent hostnames. Capture:

cd /mnt/code/projects/Strata
git status --short --branch
git remote -v
git rev-parse HEAD
hostname -f
nvidia-smi -L
ip -br addr
# Substitute the real SSH alias, already configured on both nodes:
ssh <worker-ssh-alias> 'hostname -f; ip -br addr; command -v rocminfo; command -v hipcc; rocminfo | grep -m 3 gfx'
Measure network latency and one-stream throughput on the direct 10G+ interface (e.g. ping and iperf3), reporting MTU, link rate, RTT p50/p99, and actual TCP throughput. SSH is for process control; don't tunnel the performance data plane through SSH for final benchmarks. Bind the worker listener to the dedicated/private interface, never 0.0.0.0 by default.

Confirm stock model/pack, tokenizer, expert profile and formats. Determine whether model bytes are visible to both machines. Record local paths, sizes, hash manifests and engine commit. A matching source mount does not imply matching model artifacts. Do not modify any trained/adapted TierMoE files.

Phase 1 — Unmodified primary and AMD worker baselines
Build the current engine from the same source commit into local build folders, not the shared checkout. Primary (RTX 3080 / CUDA):

SRC=/mnt/code/projects/Strata
cmake -S "$SRC" -B "/var/tmp/strata-cuda-$USER" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "/var/tmp/strata-cuda-$USER" --target strata -j8
Worker (RX 6800 XT / HIP):

SRC=/mnt/code/projects/Strata
cmake -S "$SRC" -B "/var/tmp/strata-hip-$USER" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030
cmake --build "/var/tmp/strata-hip-$USER" --target strata -j2
Adjust compiler/toolkit paths only after inspecting docs/AMD_HIP.md and installed ROCm; use matching ROCm compiler and libraries. /var/tmp should be confirmed node-local. The worker may need ROCm installation or source toolchain fixes before anything else; stop at that gate rather than implementing network code around a nonfunctional HIP backend.

Run a known-good supported stock model and existing model-appropriate parity checks on each GPU. Capture baseline decode/prefill throughput, output/checksum/tolerance, cache sizes and log. If the model cannot run on AMD, prove a standalone expert kernel parity fixture before integrating inference.

Phase 2 — Protocol and GPU-free loopback proof
Implement a narrow, versioned binary protocol over persistent TCP (no per-layer JSON or reconnect). Proposed messages:

Message	Payload / behavior
HELLO / CAPS	Protocol version, model+format fingerprint, hidden size, top-k, layer/expert counts, dtype, endianness, backend and worker limits
INSTALL_PLAN / READY	Immutable worker-owned (layer,expert) set; load and verify all selected expert blobs before acknowledging
EXECUTE	Request ID, layer, token count, owned token/expert/weight mappings, float32 activation rows
RESULT	Matching request ID, float32 per-token weighted partial sums, explicit ownership/coverage mask, actual worker timings
ERROR / PING	Bounded errors, health, teardown; no indefinite blocked reads
Use explicit lengths, validated bounds, consistent byte order and checksums/fingerprints where useful; never send raw C++ structures/pointers or device addresses. Match responses by request ID. V1: one active inference request, static placement, no speculation, no batch concurrency, no dynamic adaptation, no compression, no RDMA.

Before touching verifier integration, run a fake CPU executor over loopback and between machines. Test framing, float32 round trips, zero remote experts, sparse ownership, multiple tokens, malformed packet handling, reconnect behavior and timeout. Measure idle TCP RTT and payload size. Use TCP_NODELAY if benchmarked beneficial.

Phase 3 — Standalone HIP expert worker
Add a worker executable (proposed name: strata-expert-worker, does not exist yet) linked with the required HIP runtime, expert layout/source/cache, grouped native expert kernels and reduction. Run on the AMD host's GPU device 0. Do not directly instantiate existing RemoteExperts::open() and call it done: its same-process primary/helper assumptions reject device 0 and its current ownership APIs depend on the primary's cache.

Extract/reuse the minimal kernel/cache execution layer needed for a worker-local device, without refactoring the whole verifier or interfering with the local helper path. V1 preloads a small, deterministic set of (layer,expert) pairs into RX 6800 XT VRAM and refuses execution for any nonresident pair. Preserve source layout/quantization and routing weights. Show worker ready state only after actual cache load/verification. Return a reduced float32 vector per token and timing counters.

Prove single expert, several experts, duplicate tokens, and partial sums against independently computed reference outputs. Respect realistic floating-point tolerances across CUDA/HIP; report absolute and relative errors, NaNs, token deviations and any failed layer. Failure is explicit and request-scoped.

Phase 4 — Opt-in primary integration
Implement a network executor next to (not in place of) the existing helper tier. Proposed switches, not existing Strata flags:

--expert-worker <worker-private-ip>:<port>
--expert-worker-placement <profile-or-plan-path>
Do not add CLI flags until finding the correct parser/config path and following its conventions. Primary owns the authoritative placement map; only entries loaded and acknowledged on the worker may be marked remote. All other experts remain on the current primary/CPU path. No double execution; no missing rows.

Keep existing planner/pool semantics, ownership masks, zero-copy lifetimes and GPU doorbells intact. Dispatch remote work early enough to overlap with local CPU/GPU work, but never publish a completed layer until its remote partial sum has been accounted for. For worker failure after ownership is assigned: either recompute affected rows locally before releasing the layer, or fail the request cleanly and drain/unblock GPU waits. Add a finite deadline and prevent use-after-free/stale responses. Network features must be off by default with unchanged single-node behavior.

Start with decode only and no remote prefill. No network cache refill or expert migration yet. Do not train or modify routers, LoRA adapters, model weights or the existing cache policies as part of the first feature.

Phase 5 — Qualification and decision
Run a paired, same-model, same-commit, same-prompts comparison:

RTX 3080 baseline, no network worker.
RTX 3080 with distributed feature enabled but zero remote ownership (should behave like baseline).
RTX 3080 + RX 6800 XT, fixed expert set, real remote executions.
Remote worker deliberately unavailable / disconnected; assert local recovery or clear bounded failure, never deadlock.
For each, report prompt/prefill tok/s, steady-state decode tok/s, time to first token, total generated tokens, output divergence, resident remote expert count, expert hit/dispatch count, CPU fallback count and transferred bytes. Instrument per-layer send/stage/kernel/return/combine times, wait-on-remote time, median/p95/p99, and CUDA/HIP CPU/GPU utilization. Warm/cold results must be identified separately. Verify worker executions actually occurred; reported combined VRAM is not evidence of useful remote compute.

Success gate: correct output within declared tolerances, robust failure behavior, and a measured end-to-end throughput or model-capacity benefit over RTX 3080 alone under at least one realistic workload. If it works but is slower, report that clearly; do not obscure the finding by changing the router or introducing speculative scheduling.

5. Scope boundaries / risk register
Cross-vendor GPUs are intentionally isolated in separate processes. No mixed CUDA/HIP linker, no unified VRAM address space, no reliance on GPU P2P.
Local helper code is not network-safe: direct CUDA/HIP calls, mapped host memory and device pointers must not leak into protocol objects. Keep local code path intact.
Critical path is layer-by-layer. At 10GbE, FP32 activation/result payloads are plausibly manageable, but dozens of per-layer round trips can still cost more than CPU fallback. Measure, don't assume a speedup.
RDNA2 support is a hard dependency. gfx1030 compiler/runtime and exact expert quantization kernels must work locally before remote scheduling can work.
Shared checkout can cause races. One writer; separate, node-local CMake trees, caches, venvs and logs. Pin both binaries to the same Git commit/protocol version.
Model integrity was previously unresolved for adapted GGUF exports. No qualification against suspect checkpoints or comparisons until a stock model baseline is clean.
V1 does not include dynamic expert placement, distributed training, multi-worker orchestration, SSD-tier remote page fetches, cross-machine layer splitting, speculative-token batching, encrypted WAN operation, generic distributed RPC or upstream PR submission.
6. Deliverables from the local agent
A concise inventory document showing actual host aliases, network interface, GPU/backend versions, model files/hashes and source commit.
Working CUDA and HIP unmodified baselines, with logs.
Versioned on-wire protocol and CPU-only round-trip test harness.
HIP expert worker executable plus per-expert numerical parity evidence.
Opt-in primary network executor with fixed placement and clear fallback/failure semantics.
Repeatable benchmark script and JSON/CSV evidence; notes saying measured, failed, not run, or code inspection only.
Documentation at docs/DISTRIBUTED_EXPERTS.md covering setup, known limits and reproduction.
One or more small, well-scoped commits on feat/distributed-experts; do not commit downloaded models, build products, logs, temporary caches or secrets.
7. Recommended local-agent starting instruction
Work in /mnt/code/projects/Strata on branch feat/distributed-experts. Both the CUDA RTX 3080 primary and HIP RX 6800 XT worker mount this same checkout at the same path; passwordless SSH is already established. Read AGENTS.md, docs/AMD_HIP.md, docs/SECOND_GPU.md and this handover. Inspect current code and gather host/network/toolchain information without changing model artifacts. First demonstrate unmodified engine/kernel execution on both GPUs using separate local build directories and a verified stock model. Then implement the smallest opt-in network expert executor described here, with static expert ownership and remote GPU reduction. Keep the current single-node path untouched. Do not add training, dynamic cache scheduling, model modifications or unrelated features. At each gate, record evidence and stop if the underlying HIP/backend or model-integrity prerequisite fails. You may run remote commands over the existing passwordless SSH. Make no performance claims without paired measurements.

First practical action: identify the worker's SSH alias and 10GbE IP, verify the branch/working tree and actual ROCm gfx1030 support; then build unmodified Strata independently on both hosts.

⋮⋮⋮⋮
