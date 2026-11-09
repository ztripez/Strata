# Experimental DeepSeek expert-worker tiers

Source-checkout qualification path, 9 November 2026. This extends the existing
expert worker behind `STRATA_NETWORK_EXPERTS`; it is separate from the app and
Terminal launch paths. Live tiered RPC and full hybrid inference remain unqualified.

## Placement and native execution

Without `--expert-vram-mib`, the DeepSeek worker retains its GPU-resident profile.
Supplying that option enables the worker's existing static placement across GPU,
bounded copied RAM and mapped GGUF views. `--resident-budget-mib` limits the copied
expert arena. These limits exclude process scratch, CPU tables and OS file cache.
READY acknowledges logical ownership, including experts outside GPU residency.

The DeepSeek tiered path requires the qualified `DS4_IQ_Q8K=1` arithmetic.
It uses the existing `ds4::Pool`, bounded here to one worker plus its caller,
and `iq_cpu_quant`, `iq_cpu_gate_up` and `iq_cpu_down` for host experts. The small
adapter borrows source views and dispatches row bands. Routing weights and
assignment order stay unchanged; each unweighted expert result is reduced once.
GPU experts continue through the existing native token kernels.

The same worker placement code supplies packed RAM blobs or the original
read-only gate/up/down mappings. DeepSeek uses its native CPU row layout;
Qwen keeps its existing native CPU pool and formats. Layer owners and workers
keep their existing memory and arithmetic implementations.

`strata-worker-tiers` reports actual placement counts/bytes. With `--verbose`,
`strata-worker-tier-request` reports executed GPU/RAM/mapped assignments for each
request. Mapped assignments are a logical storage path, not proof of physical
SSD/NFS reads. The DeepSeek `kernel_us` field currently includes service time,
transfers and host work; it must not be treated as pure GPU kernel time.

## Bounded all-format fixture

Build `strata-expert-worker`, `deepseek_worker_parity_test` and
`deepseek_worker_cpu_test` in both CUDA and HIP build trees. The CPU-only test
requires neither model files nor a GPU operation.

The opt-in `--per-layer-plan` fixture uses one fresh expert connection per layer.
Each installs an immutable eight-expert plan. With 20 MiB GPU and 20 MiB copied
RAM budgets, the pinned pack's eight format pairs statically predict two GPU,
two RAM and four mapped experts per plan. Check actual placement and request
counters; the byte calculation alone is not runtime evidence.

```sh
# Worker: nine connections = one deliberate arithmetic mismatch, eight valid plans.
DS4_IQ_Q8K=1 BUILD/strata-expert-worker --model FIRST_GGUF --bind WORKER_IP:19175 --expert-vram-mib 20 --resident-budget-mib 20 --timeout-ms 30000 --max-sessions 9 --verbose

# Oracle on the other host; run both CUDA-to-HIP and HIP-to-CUDA directions.
DS4_IQ_Q8K=1 BUILD/deepseek_worker_parity_test FIRST_GGUF WORKER_IP:19175 --per-layer-plan
```

The existing seeded 48 cases and independent native GPU/CPU evaluations remain
unchanged. Fresh connections change placement, not samples or tolerances.
Require all eight format pairs, all three actual tiers, exact coverage, 960 total
assignments per direction, no fallback and successful supervised completion.

## Current evidence

CUDA/HIP worker and fixture builds passed. On both hosts, the actual CPU dispatch
helper passed eight exact serial/threaded and separate/packed-view comparisons
against native full-row execution, including a partial final row band. Twenty-eight
invalid-input cases preserved output and scratch. ASan/UBSan passed on Lucy;
the linked prebuilt CPU backend libraries were not rebuilt with sanitizers.

These are synthetic dispatch controls. Live tiered RPC across the eight model
format pairs, model arithmetic, worker-loss fallback, E4 hybrid inference, E5
quality and performance have NOT RUN to acceptance for this change. Receipts,
including the initial unsupported-fixture refusal and build-invocation failure,
are in `/var/tmp/strata-deepseek-tiered-worker-20261109/`.
