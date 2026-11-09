# Distributed Strata MoE Fabric

**Project specification · v0.1 · 2026-11-06**  
**Repository:** `ztripez/Strata` (fork of `Niko1221/Strata`)  
**Working directory on both nodes:** `/mnt/code/projects/Strata`  
**Model ladder:** Qwen3.6-35B-A3B → DeepSeek V4.1-Flash → Kimi K3  
**Status:** Specification. No new benchmark results are asserted in this document.

**Execution update, 9 November:**
[Network-sharded DeepSeek execution plan](NETWORK_SHARDED_DEEPSEEK_PLAN.md) is the
next implementation plan and proposed acceptance contract. It extends this v0.1 design
to whole-layer placement plus MoE offload. The fixed Lucy-primary/Astor-worker
diagram below describes the initial expert-only configuration, not a constraint
on layer ownership. A layer owner executes its router; the coordinator need not
own every dense layer. A node need only fit its assigned range, not a solo copy
of the entire dense model. Historical measurements below remain unchanged.

> [!IMPORTANT]
> ## NON-NEGOTIABLE: STRATA ALREADY HAS STAGED MoE ALLOCATION
>
> **Do not invent a new VRAM → RAM → SSD expert storage/caching system.**
>
> Strata already provides adaptive GPU expert caching, CPU fallback, expert placement and profiling, prefetch/staging, pinned-RAM residency, RAM budgets, and file-backed/SSD expert execution. The relevant Qwen engine already exposes `--resident-experts`, `--resident-budget-gib N`, and `--mmap-experts`. Strata also has existing multi-GPU and DeepSeek-specific placement logic.
>
> **This project extends existing staging ACROSS MACHINES.** Our work is cross-node resource discovery, cost-aware placement, remote execution, ownership, accounting, model adapters, and fault handling. Existing Strata node-local storage/staging/cache policies remain authoritative. Agents must inspect them before coding and must justify any feature that appears to duplicate them. A proposal to rewrite Strata's expert cache, LRU/LFU, RAM arena, SSD pager, GGUF reader, quantization kernels, or model router is **out of scope by default**.

## 1. Mission

Coordinate dissimilar consumer computers as a **distributed expert memory-and-execution fabric** with two independently measurable objectives:

1. **Speed:** Improve real inference latency and/or throughput at the same model quality and resource budget.
2. **Capacity:** Execute larger or higher-quality complete models, or reduce the resource footprint at the same usable quality/speed, by making VRAM/RAM/SSD across nodes useful without requiring every node to hold every expert.

The mental model is **fastest VRAM → slower VRAM → fastest RAM → slower RAM → fastest SSD → slower SSD**. The actual orchestrator must **not** assume that fixed order: a network round trip can make remote VRAM slower than local CPU execution, and SSD read latency matters differently for a cold expert than for an already resident remote expert. Use a cost graph, not a global tier list.

**Routing ≠ placement ≠ execution:** the unchanged model router selects experts; the orchestrator chooses which node/tier owns their weights and which executor should compute them. The orchestrator is a **control plane** and must not sit in a per-layer network hot path.

### Success is multidimensional

Report the Pareto frontier: **quality/model size × warm/cold decode × prefill/TTFT × RAM/VRAM/SSD residency × node count**. More VRAM occupied or higher worker GPU utilization is not, by itself, a success. Lower RSS is not proof that a larger model is usable. The final project must demonstrate **both** a speed win and a capacity win, but may qualify them independently along the way.

## 2. Existing capabilities: preserve, do not rebuild

### Upstream Strata

- **GPU VRAM**: adaptive resident expert cache; model-specific dense/attention/KV paths.
- **Host RAM**: resident expert arena / pinned execution and CPU expert fallback.
- **SSD/file-backed**: mapped experts, staging/prefetch, RAM-budget-dependent read paths; separate DeepSeek Engram/n-gram lookup.
- **Low-RAM modes** in the applicable engine: `--resident-experts`, `--resident-budget-gib N`, `--mmap-experts`.
- **Multi-GPU/model placement**: existing local multi-GPU strategies; `strata-dsv41` has separate DeepSeek placement logic.
- **Model-specific kernels and formats**: CUDA/HIP, existing host expert kernels; re-use rather than abstract away critical implementations.

**Do not assume the Qwen flags or helper interfaces have identical names inside DeepSeek.** Inspect `src/dsv41` and its engine controls before extending them.

References: [Strata low-RAM details](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md), [DeepSeek engine](https://github.com/Niko1221/Strata/blob/main/docs/DEEPSEEK41.md), [multi-GPU](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md), [AGENTS.md](https://github.com/Niko1221/Strata/blob/main/AGENTS.md).

### Existing experimental fork (supplied project checkpoint)

The Qwen3.6 path already includes:

- `NetworkExpertExecutor`: NVIDIA RTX 3080/CUDA primary → TCP → AMD RX 6800/HIP worker; versioned wire format and GGUF content-hash handshake.
- Full-model decode, zero-ownership identity, numerical parity, and bounded disconnect/timeout fallback.
- Trace-ranked worker placement; **whole-layer concentration** outperformed spreading small expert sets across every layer because it avoids network sync on unowned layers.
- Opt-in integration with **existing** Strata low-RAM flags; omits remote-owned weights from permanent local pinned copies, supports mapped-file fallback and remote prefill, and avoids unnecessary `h_ewts` copies for unowned layers.
- Reported `layer-24L-full`: **6,144 experts on 24 layers** on Astor. Earlier sweep showed **+5.1% warm decode** versus its paired local baseline. Later low-RAM sweep reported **42.05 → 9.09 GiB** primary process RSS with `--resident-experts` at roughly **50 tok/s** warm decode. These are **Qwen experimental results, not DeepSeek forecasts**.
- Experimental code is off at build time and run time by default. Keep it so.

Inspect the **local branch**, not only public GitHub: `src/qwen35/q36_engine.cu`, `include/strata/core/expert_worker_proto.hpp`, `expert_worker_wire.hpp`, `include/strata/kernels/cpu/pool.hpp`, `tools/qwen35/generate_placement_from_trace.py`, `bench/distributed_experts.mjs`, and `docs/DISTRIBUTED_EXPERTS.md`.

### Known starting hardware

| Node | Confirmed baseline | Must still be measured/checked |
|---|---|---|
| `lucy` | RTX 3080 10 GiB CUDA; Kingston-only 64 GiB RAM | Physical DIMM slots and longer memory stability; local SSD capacity/I/O |
| `astor` | Radeon RX 6800 reported by HIP (`gfx1030`), ~16 GiB; working remote Qwen experts | Actual worker RAM/SSD, DeepSeek-format HIP support, available VRAM |
| Link | Dedicated 10GbE previously measured near 9 Gb/s | Verify route to `192.168.63.177` actually uses `enp4s0`, not 1GbE `enp7s0` |

Passwordless SSH and shared source mount already work. **One writer** on the shared source tree; **separate node-local build trees, runtime environments, and evidence**. Neither a shared source path nor an SSH link means shared GPU memory.

## 3. Model ladder and benchmark definitions

### M0 — Qwen3.6-35B-A3B (permanent regression)

Use the verified stock model and earlier comparison harness as the non-regression test for remote expert execution, low-RAM integration, failure semantics, and measured placement costs. Preserve the existing code and evidence.

### M1 — DeepSeek V4.1-Flash-1017 (primary benchmark)

Strata already runs this family through a **separate engine**, `strata-dsv41` (`src/dsv41`). The documented model has **40 layers, 384 routed experts/layer, top-6 routed experts**, plus dense/attention components and a large **Engram** lookup table that is *not* the MoE expert cache. A full-model implementation must account for **all three**: dense path, routed experts, and Engram I/O.

**Pinned starting benchmark:** ISTA-DAS **2.00 bpw** pack (`strata launch --family deepseek41 --model ist-das-2.00bpw --plan`), approximately **257 GB download** and **265 GB required free disk** per Strata's documentation. Record the exact revision, each shard hash, tokenizer, quantization, Engram and drafter. Do the fit/disk preflight **before downloading**. Do not switch to a cheaper pack and compare throughput as though quality were unchanged.

**Known blocker to investigate:** documented dense weights are about **16.8 GB** in one published layout, exceeding Lucy's 10 GiB VRAM on their own; the existing DeepSeek engine has its own GPU-placement path. Documentation of the Radeon R9700 DeepSeek HIP backend **does not establish** DeepSeek-format support on Astor's `gfx1030`. Verify dense placement and expert kernel compatibility **before** spending time on remote expert performance.

### M2 — Kimi K3 (north star)

Kimi K3 has **2.8T total / 104B active parameters**, **93 layers**, **896 routed experts/layer**, top-16 routing, latent MoE, hybrid KDA/MLA attention and native MXFP4 weights. Treat this as a **separate backend and hardware feasibility project**, not a promise that the two current GPUs can run the full model. An idealized 4-bit copy of 2.8T weights already requires approximately **1.4 TB decimal before scale metadata and other tensors**. SSD capacity, random I/O, dense/attention execution and model-specific kernels will matter as much as remote experts.

**No Kimi implementation before DeepSeek gates.** Kimi K3 is an explicit north-star interpretation of “Kimi”; pin its precise model revision and license during its feasibility gate.

Reference: [Kimi K3 official model card](https://huggingface.co/moonshotai/Kimi-K3).

## 4. Architecture contract

```text
                    GLOBAL EXPERT ORCHESTRATOR
              discovery | cost model | placement plans
                 resource budgets | policy versions
                              |
                    plan / READY / ACK
             +----------------+----------------+
             |                                 |
        LUCY (CUDA)                        ASTOR (HIP)
   existing Strata cache/tiering      existing Strata cache/tiering
   attention/dense/router             GPU/CPU expert execution
   local VRAM/RAM/SSD                  worker VRAM/RAM/SSD
             |                                 |
             +---- direct persistent link -----+
                activation / weighted output
```

### Global resource inventory

Per node/device: backend + supported operations/quant formats, true available VRAM, host RAM/NUMA/pinned allocation budget, SSD free space and measured latency/bandwidth, actual cache residency, local expert compute cost, link RTT and bandwidth, queues/load and failure status.

### Plan manifest

Versioned, model-hash-bound ownership map for `(layer, expert)` or full layers. Each ownership entry identifies node, execution device, backing tier, replicas, local Strata cache settings and exact fallback source. Include dense/attention and Engram placement constraints separately. Worker reports READY only after validating the bytes and actually loading/identifying resources it promises.

### Execution

- Primary executes router **unchanged** and dispatches only to an owned, READY remote destination.
- Reuse existing `NetworkExpertExecutor` and wire protocol. Send activations and routing metadata; **do not ship expert weights per token**.
- Preserve no-extra-work fast path on unowned layers. Avoid double execution and missing contributions.
- Local Strata stays in charge of **node-local** VRAM↔RAM↔SSD staging. Remote CPU/RAM/SSD are node-owned execution/staging options; they are not a global pointer-accessible memory pool.
- Use explicit timeouts, model hash checks, coverage accounting and rollback/fallback policies. Missing worker-held bytes cannot magically be reconstructed from a removed host copy; file-backed fallback must be real and predeclared.
- Initial global planner is **offline**: trace → plan → qualification, with static plan installation. No agent should add per-token central scheduling or dynamic migration until the current gates pass.

### Cost model

Consider per-layer **critical path**, not just fastest memory:

`cost(remote) = dispatch + network + staging + queue + expert compute + return + combine − overlap`

Include local CPU alternatives, layer synchronization, prefill/decode distributions, SSD misses, residency/replication costs and backend-specific numerical effects. Select plans on a measured **Pareto frontier**, not a single hardcoded hierarchy order.

## 5. Success gates

**Status vocabulary:** `PASS`, `FAIL`, `BLOCKED`, `NOT RUN`. Every gate must have exact model ID, full hashes, source/binary revision, commands, tests, raw metrics and an evidence manifest. *Build success is not numerical success; loading is not decode success; fewer bytes in primary RSS is not automatically model-capacity success.*

### G0 — Machine and byte-integrity baseline

**Entry:** current hardware and fork. **Required:** repeat complete Memtest86+ passes on Kingston-only Lucy; confirm physical DIMM slots and clocking; pin both code/binary versions and verified model bytes; inventory memory/disk/device support on both nodes; log `ip route get` and 10GbE bandwidth/RTT. Keep the historical NFS/local-page-cache corruption evidence.

**PASS:** zero reported memory errors in the agreed repeated test, stable full-file hashes before/after runs, reliable route/backend inventory. **BLOCK:** any new memory error or unexplained content mismatch.

### G1 — Qwen regression safety net

**Required:** bit-identical instrumented baseline and zero-ownership logits/tokens; original expert parity tolerances; full-model finite outputs; remote execution and safe disconnect/timeout; existing low-RAM sweep and zero per-layer overhead on unowned layers.

**PASS:** no correctness/failure regressions; performance within **5%** of the previous known-good matched Qwen fork on a paired repeated run; remote-owned experts are not silently repinned on Lucy. **FAIL:** identity/regression/fallback violation.

### G2 — Discovery and offline cost-aware orchestrator

**Required:** machine-readable resource inventory; trace-informed plan generator; three candidate plans (`speed`, `capacity`, `balanced`); actual per-layer dispatch cost; capacity constraints including files and dense memory; held-out Qwen sweep.

**PASS:** reproduces a viable `layer-24L-full`-class Qwen plan or better on held-out prompts without bypassing cache semantics, obeys all hard budgets and produces replayable manifests. **FAIL:** invented rank that ignores synchronization or resource budgets; duplicate Strata cache system.

### G3 — DeepSeek stock-engine feasibility

**Required:** `strata-dsv41` official preflight and storage plan before downloading; pinned 2.00-bpw weights/Engram/quantization; establish best available **unmodified** local baseline if it actually runs; identify/resolve or explicitly block dense placement and `gfx1030` HIP format compatibility. Test relevant standalone expert formats.

**PASS:** accurate feasibility report and verified stock model/baseline *or separately qualified prerequisite port*; code/protocol contract for DeepSeek. **BLOCK:** cannot place the dense spine; missing HIP kernels; insufficient disk; model identity mismatch. **Do not label a planning command a working inference run.**

### G4 — DeepSeek cross-vendor functional inference

**Required:** exact same DeepSeek model bytes on both ends; real nonzero remote expert assignments in `strata-dsv41`; dense path and Engram work; numerical tolerance/teacher-forced comparison; at least **two distinct prompts each producing ≥128 decode tokens** (legitimate EOS is allowed if documented); zero-ownership identity where expected; deterministic bounded worker failure behavior.

**PASS:** real full-model end-to-end decode with remote HIP worker, bounded numerical drift and no integrity violations. **FAIL:** mock-only success, data corruption, silently changed model quantization or unhandled worker loss.

### G5 — DeepSeek capacity gate (independent of speed)

**Required:** fixed **per-node** VRAM/RAM budgets, one complete pinned model, measurements of both nodes' RSS/PSS, pinned/unpinned memory, OS page cache, VRAM, SSD I/O and startup. Demonstrate no blanket copy of remote-owned experts on Lucy. Test low-RAM steady state and failure recovery. Engram and dense weights are separately accounted.

**C-functional PASS:** full model runs under documented budgets and supports **128+ tokens per test** with bounded resource usage. **C-usable PASS:** additionally, at identical model quality, distributed mode demonstrably increases admissible memory headroom/usable model fidelity **or** reduces SSD pressure/host residency relative to a feasible single-node execution under the same limits; target **≥5 warm decode tok/s** at a specified 2K-context workload (project acceptance target, not a forecast). **Do not claim larger-model enablement merely because RSS falls if the same model can already be used equally well locally.**

### G6 — DeepSeek performance gate (independent of capacity)

**Required:** same pinned quantization, tokens, expert count, settings, prompt contexts and hardware constraints. Test unchanged solo Strata, enabled-zero-owner, best static layer plan and orchestrated plan. At least **30 requests per case**, ≥5 held-out prompts, ≥3 interleaved test rounds; warm/cold latency, prefill, TTFT, decode, p95, per-layer waits and all-node resource costs. Report uncertainty and numerical equivalence, not just a best run.

**P PASS:** warm median decode **≥10% faster** than best admissible same-quality solo baseline, bootstrap 95% confidence interval excluding zero improvement, and no >10% p95 TTFT regression on the defined workload. If no solo baseline can start, mark speed **UNCOMPARED** rather than fabricate a speedup. A capacity result may pass while this remains pending.

### G7 — Orchestrator v1 release gate

**Required:** offline inventory→candidate plan→install→execute→measure loop; safe worker setup/restart/loss handling; checksums, bounded timeouts, no exposed unauthenticated listener outside trusted LAN, reversible opt-in, unchanged original paths, saved manifests/scripts and documented measured frontier.

**PASS:** G0–G6 satisfied, **both G5 C-usable and G6 P** demonstrated on a pinned DeepSeek pack, no unresolved integrity issue. This is **one primary + one worker** only; do not infer arbitrary N-node scaling.

### G8 — Kimi K3 feasibility gate (north-star entry, not full inference)

**Required:** exact official Kimi K3 revision and license; format/ops compatibility matrix (KDA, gated MLA, latent MoE, SiTU-GLU, MXFP4), sharded weight/dense storage footprint, GPU/RAM/SSD capacity and IO requirements; estimate tokens/sec and whether a two-node or expanded cluster is viable; isolated operator and expert parity test plan.

**PASS:** reproducible costed feasibility report and scoped missing backend work. **NOT a PASS:** running Kimi K3 full inference—this requires later, separately budgeted gates.

## 6. Agent constraints and review checklist

Before proposing **any new subsystem**, the agent must answer:

1. **Which Strata module already implements the local function?** Cite its code/path.
2. **Which observed limitation** cannot be solved through current flags, placement, ownership, or a narrow adapter?
3. Can we extend `NetworkExpertExecutor`/existing APIs without introducing a second parallel cache manager or RPC stack?
4. Is the change **Qwen-specific, DeepSeek-specific, or genuinely generic**?
5. Does `STRATA_NETWORK_EXPERTS=OFF` retain stock behavior and dependencies?
6. Which gate provides objective evidence of value?

**Immediate rejection criteria:** new VRAM/RAM/SSD cache hierarchy; router retraining/replacement; a second SSD paging system; moving weights per token; hardcoded global tier order; unpinned model revision; capacity claims from virtual allocations alone; speed claims without paired tests; bypassed hash/READY/fallback; claiming Kimi support based on DeepSeek success.

**Development sequence:** read local `AGENTS.md`/DeepSeek docs and capture working-tree state → G0/G1 → inventory and offline planner → DeepSeek feasibility and required scoped backend bridges → end-to-end parity → independent capacity/speed qualification → safe orchestrator v1 → Kimi feasibility. Do not overwrite a shared checkout or previous evidence.

## 7. Evidence and deliverables

Each gate must generate a `qualification.json` (or equivalent), status, machine/backend identity, source/binary hashes, model shard checksums, plan ID, exact commands, raw logs, measured metrics, rejection reasons, and `evidence-manifest.sha256`. Preserve failures and collect from **both** nodes. Keep large model files and raw evidence outside Git.

Mandatory deliverables:

- `docs/DISTRIBUTED_FABRIC.md` or this specification in the repository; architecture and protocol version contract.
- Resource inventory and cost/placement manifest schema; offline planner, **not a new cache manager**.
- DeepSeek adapter to existing network execution, with dense/Engram compatibility documented separately.
- Reproducible model-byte/geometry and numerical-parity suites, failure tests and performance/capacity harness.
- A dashboard/report with separate **correctness**, **capacity**, **speed**, **resource accounting**, **failure safety** and **Kimi feasibility** statuses.

## 8. Agent starting prompt

> You are working on `ztripez/Strata`, with a shared source checkout at `/mnt/code/projects/Strata` on `lucy` and `astor`. Read this specification and current `AGENTS.md`, `docs/DETAILS.md`, `docs/DEEPSEEK41.md`, `docs/MULTI_GPU.md` and local `docs/DISTRIBUTED_EXPERTS.md` before any implementation. **Strata ALREADY stages and caches experts across VRAM, RAM and SSD. DO NOT REIMPLEMENT THIS.** Preserve its cache, RAM budgets, SSD staging, router and quantized expert kernels. Build only the missing cross-node inventory, cost-aware placement/orchestration, and the smallest model-specific DeepSeek integration on top of the existing `NetworkExpertExecutor`. Follow success gates G0–G8 in order; benchmark capacity and speed independently. Use evidence, not assumptions, to declare a gate passed. First capture the actual local source revision, verify hardware/model integrity and run the Qwen regression suite.

---

**Project definition of success:** Two (or ultimately more) heterogeneous ordinary computers can run a **larger useful MoE model** and **infer faster** through coordinated expert placement, *without replacing Strata's existing staged MoE allocation*. DeepSeek V4.1-Flash is the proving ground; Kimi K3 is the north star.
