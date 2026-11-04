# Network expert execution: experimental checkpoint

This fork adds a Linux-only expert worker for **Qwen3.6 (`qwen35moe`) text decode**.
It is off at build time and at run time. It is **not qualified for inference or a speedup**.

## Current gate: model integrity failed

On 4 November 2026, the new content-hash handshake refused the primary and worker pair.
The source revision was `ffa80d73133ea5c548aefe57889a16e8085a938f`, plus this feature's local changes.

The stock file is `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, 22,853,663,008 bytes.
Its SHA-256 in `setup.py` is:

```text
55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1
```

The worker's local file and its Strata mapping both matched that hash. A later read
of the same NFS path from the worker also matched. The primary's
first `sha256sum` also matched. Later, the primary's Strata mapping, an independent
Python mapping, `pread`, and a repeated `sha256sum` all returned:

```text
226dc880081fce5443606bc8f3ab2081d60a6434753d84fc1e0b23805439e494
```

A chunk-by-chunk comparison of the primary's mapping and `pread` found no differences
between those two views during that probe. Thus this is an observed change between reads,
not proof of a simultaneous mmap-versus-read discrepancy. Reported size, mtime and ctime
were unchanged. The cause is unknown. The primary file is on NFS; that fact alone does not
establish a storage or kernel cause.

The handover requires stopping on an integrity failure. At that initial checkpoint, no
model was repaired, replaced, redownloaded, or written. Do not bypass the handshake or treat the
earlier geometry-only parity run as qualification of these model bytes.

Evidence is outside Git, on each machine in:

```text
/var/tmp/strata-distributed-20261104-resume/
```

The primary directory contains the initial and repeated hashes, the mapping probe and
its source, protocol logs, both build receipts, preserved pre-integration binaries,
and the rejected real-worker parity log. `evidence-manifest.sha256` identifies the
saved files. This directory is local evidence, not a release artifact.

### User-supplied follow-up, 4 November 2026

The user supplied a `sha256sum` result run on `pve01` for the same model path. It
matched the full stock hash above. This is a user-supplied observation, not a new
measurement by this continuation. The transcript does not include the server's
mount information or a simultaneous server/client capture.

The user then supplied alternating NFS and SSH-stream results from Lucy. All three
NFS passes returned the full `226dc880...` hash above. SSH passes 1 and 2 returned the
full stock `55983c5a...` hash. SSH pass 3 is truncated to `55983c` in the supplied
text, so it is not counted as a complete hash result. The duplicated paste is one
reported sequence, not two independent runs. These checks were not repeated here.

This supports a failure associated with Lucy's NFS read path during those passes.
It does not identify its cause or establish that Lucy's mixed RAM is faulty or
healthy. A correct SSH-stream hash does not explain how the differing NFS bytes
arose. No differing byte offsets or bit counts have been established for this file.

The selected workaround is a separate local download from
`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, using the exact filename, byte count and SHA-256
pinned in `setup.py`. Two successful full-file hash checks are required before
retrying the content-hash handshake and numerical parity. This permits qualification
of the new local artifact; it does not resolve or clear the original NFS incident.
The NFS model and existing evidence must remain untouched.

At the local-storage preflight, neither disk had room: `/` had about 3.6 GiB free
and `/mnt/work` about 6.9 GiB, while the model alone needs 21.3 GiB. The Hugging Face
CLI dry run found the exact 22.9 GB file. The user then authorized clearing Lucy's
`uv` and `sccache` caches to make room for a separate local download.

### Local ext4 retry: integrity gate failed again

The cache cleanup completed. The exact file was downloaded directly from Hugging
Face revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` to:

```text
/home/ztripez/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

This path is on Lucy's local `/dev/nvme0n1p2` ext4 filesystem. Its size was exactly
22,853,663,008 bytes. Two consecutive full-file `sha256sum` checks matched the stock
hash. Both binary sets also matched the previously saved build manifests.

The next real CUDA-to-HIP parity test rejected the model during the content-hash
handshake. Astor reported the stock hash, but Lucy's mapping of the new local file
reported:

```text
eca1cd3dbea39929b314baea37f35efadbd2dbe8535552e258548c352a438c79
```

An independent Python probe then obtained that same alternate hash through both
`mmap` and `pread`. It compared corresponding 4 MiB chunks and found no differences
between those two views. Reported device, inode, size, mtime and ctime remained
unchanged across the recorded checks. Chunk hashes are saved for later byte-level
investigation; they do not establish the number of bytes differing from stock.
A fresh `sha256sum` after the probe also returned `eca1cd3d...`; its comparison
against the pinned stock digest failed. Thus the later mismatch was reproduced by
the same command that had passed both initial checks, as well as by the mapping
and `pread` probes.

The local-download workaround therefore did **not** restore the integrity prerequisite.
This reproduces a read-integrity discrepancy on local ext4, so the investigation
cannot be limited to the NFS path. It does not establish faulty RAM, a disk failure,
or an on-disk modification. Qualification stopped before numerical expert execution;
no full-model or performance test was started. The worker exited after its one test
session. The original NFS model, mounts and cache settings were not changed.

New evidence is separate from the earlier failure record:

```text
/var/tmp/strata-local-model-20261104-HYw3mB/
```

It contains the cleanup logs, pinned HF metadata, download receipt, both successful
hash checks, the rejected handshake logs, file metadata and independent read probe.
`qualification.json` records the stopped gate; `evidence-manifest.sha256` identifies
the saved evidence files. No thresholds were changed to continue execution.

## Observed machines

| Role | Host and data address | GPU reported by runtime | Toolchain | Active link |
| --- | --- | --- | --- | --- |
| Primary, initial route | `lucy`, `192.168.63.230`, `enp7s0` | NVIDIA GeForce RTX 3080, 10 GiB | CUDA 13.4.92; driver 615.78.08 | 1,000 Mb/s, MTU 1500 |
| Primary, later link test | `lucy`, `192.168.60.155`, `enp4s0` | same GPU | same toolchain | 10,000 Mb/s, MTU 1500 |
| Worker | SSH alias `astor`, `192.168.63.177`, `enp37s0f0` | AMD Radeon RX 6800, about 16 GiB, gfx1030 | HIP 7.2.53211; ROCm packages 7.2.4 | 10,000 Mb/s, MTU 1500 |

The requested hardware was described as an RX 6800 XT; the runtime reports RX 6800.
No board identity beyond that was established. Initially Linux selected the primary's
1 Gb/s interface. After the user enabled its 10 Gb/s link, a temporary route to the
worker alone selected `enp4s0`. A 512 MiB single-stream transfer in each direction measured
8.93 Gb/s primary-to-worker and 9.03 Gb/s worker-to-primary, including the probe's host
copy/check overhead. A 1,000-request CPU-mock protocol test passed on this path, with
RTT p50/p95/p99 of 239/344/399 microseconds. This is transport evidence, not inference speed.

The 32-byte echo probe with TCP_NODELAY had p99 69.7 us versus 89.4 us without it;
8 KiB echo p50 was about 145-147 us for either setting, with no useful payload-speed
claim. These are single bounded runs. Raw results and the probe source are in `tcp-10g.json`
and `tcp_probe.py`; the earlier 1 Gb/s protocol log is retained separately.

The temporary host route was removed after testing. The default routes were unchanged;
Linux again selects `enp7s0` for worker traffic. To reproduce the isolated 10 Gb/s test
without changing a default route, add and later remove this host route:

```sh
sudo ip route add 192.168.63.177/32 dev enp4s0 src 192.168.60.155
# Run the test, then restore the previous route selection:
sudo ip route del 192.168.63.177/32 dev enp4s0 src 192.168.60.155
```

Primary model path:

```text
/mnt/storage/strata-models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

Worker model path:

```text
/home/ztripez/.local/share/strata/Strata-data/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

## What is implemented

- A versioned TCP protocol with explicit little-endian integers and IEEE-754 FP32 rows.
  C++ object layout, padding, pointers and device addresses are never transmitted.
- A SHA-256 over the exact mapped GGUF bytes, plus geometry and top-k checks. Hashes are
  computed at startup and are not taken from a cached sidecar. Files must remain unchanged
  while either process is running.
- Primary-owned, immutable placement. The worker loads and reads back **every** requested
  expert before READY. The primary checks the acknowledged count before publishing its map.
  An empty plan owns zero experts, including after reconnect.
- A worker that uses device 0, native expert formats, grouped kernels and weighted GPU
  reduction. Results carry token coverage, assignment counts and GPU-event timings.
- A background I/O thread on the primary. Only selected experts absent from the primary
  GPU cache are sent to the worker. The existing CPU pool handles all other misses.
  Primary GPU work and CPU work can overlap the worker request.
- Finite frame and request deadlines, matching response IDs, coverage and finite-value
  checks. A failed runtime request is reported and its assigned rows are recomputed locally
  before the layer continues. Further network dispatch stays disabled for that process.
  A setup failure exits before serving. There is no automatic reconnect.

The existing cache replacement policy stays in use. A worker-resident expert promoted
into the primary cache executes on the primary. Its worker copy remains resident but is
not executed twice. Every expert is still retained in primary host RAM for fallback;
this prototype does not reduce that RAM requirement.

Prefill remains local, even for a one-token prompt. MTP, multiple sessions, batch decoding
and vision are refused when the feature is selected. The same-process helper and peer
implementations are unchanged. No weights, router, LoRA or kernel arithmetic were changed.

The protocol schema lives in `include/strata/core/expert_worker_proto.hpp` and its explicit
codec in `expert_worker_wire.hpp`. The frame header is 24 bytes; payloads are capped at
64 MiB. Timings are microseconds. RTT and wait include host overhead; worker stage, kernel
and return-copy times use GPU events. Reported percentiles retain the last 4,096 successful
requests. Traffic counters cover successful EXECUTE/RESULT exchanges, excluding setup
and failed exchanges. The protocol has no authentication or encryption and is intended
for the specified private, trusted interface only.

## Build

Use separate local build trees. The shared checkout must have one writer. OpenSSL's
Crypto development library is required only when this option is enabled.

```sh
cmake -S . -B /var/tmp/strata-cuda-ztripez \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_NETWORK_EXPERTS=ON
cmake --build /var/tmp/strata-cuda-ztripez \
  --target strata strata-expert-worker expert_worker_proto_test expert_worker_parity_test -j4
```

On `astor`, retain the configured ROCm and ggml paths described in `docs/AMD_HIP.md`:

```sh
ROCM_PATH=/opt/rocm cmake -S . -B /var/tmp/strata-hip-ztripez \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_NETWORK_EXPERTS=ON
ROCM_PATH=/opt/rocm cmake --build /var/tmp/strata-hip-ztripez \
  --target strata strata-expert-worker expert_worker_proto_test expert_worker_parity_test -j4
```

Run the GPU-free checks directly on each host:

```sh
/var/tmp/strata-cuda-ztripez/expert_worker_proto_test
# On astor:
/var/tmp/strata-hip-ztripez/expert_worker_proto_test
```

The qualification build trees have `STRATA_BUILD_TESTS=OFF`, so CTest lists no tests
there. The pass receipts come from executing the test binaries directly. Set
`STRATA_BUILD_TESTS=ON` when CTest registration is wanted.

`STRATA_NETWORK_EXPERTS=OFF` is the default. It adds no network library, OpenSSL dependency,
worker target or network integration to the engine. Windows, Metal and SYCL are not
supported by this experiment.

## Run after the integrity gate is resolved

These are **engine flags**, not `strata launch` flags. The primary must use hybrid experts.
First verify stable model hashes against the stock catalog on both machines, including
mapped reads. Then rerun the real-worker parity test before inference.

Start the worker on its private interface:

```sh
/var/tmp/strata-hip-ztripez/strata-expert-worker \
  --model /path/to/verified/model.gguf --bind 192.168.63.177:19415
```

Without `--bind`, it listens on `127.0.0.1:19450`. It accepts sequential sessions.
`--max-sessions N` bounds a test run. The primary is the sole placement authority.

A plan is a text file containing layer and expert numbers, one or more per line:

```text
# A small deterministic set for a first test.
0 0-7
1 0-7
20 0-7
39 0-7
```

Comments and empty plans are accepted. Invalid lines and out-of-range entries fail.
Duplicate entries in a text plan are normalized before the immutable wire plan is sent.

On the primary:

```sh
STRATA_Q8_PACKED=0 STRATA_Q36_NO_MMQ=1 \
/var/tmp/strata-cuda-ztripez/strata \
  --model /path/to/verified/model.gguf --max-context 2048 --chunk 256 \
  --mtp 0 --expert-slots 12 --serve \
  --expert-worker 192.168.63.177:19415 \
  --expert-worker-placement /path/to/placement.txt \
  --expert-worker-timeout-ms 5000
```

The two environment settings and small context/chunk above fitted the observed free VRAM
on `lucy`; they are not new engine defaults or general recommendations. The unmodified
baseline with its packed defaults ran out of memory on this occupied desktop.
The timeout applies to setup frames as well as execution. Increase it explicitly if
loading a larger plan exceeds the deadline; the maximum is 600,000 ms.

## Validation and remaining work

**Passed in this continuation:** CUDA and HIP builds; GPU-free protocol/executor tests
on both hosts; address/undefined-behavior sanitizer tests; a 1,000-request CPU-mock run
over both the initial 1 Gb/s route and the later 10 Gb/s path. Tests cover golden wire bytes, sparse and repeated activation
rows, zero ownership, invalid placement, concurrent request rejection, reconnect, bad
model identity, stale IDs, incomplete READY, incorrect result coverage, nonfinite results,
disconnect and whole-frame/request deadlines. Failed results leave caller buffers unchanged.

**Failed:** the real CUDA-primary/HIP-worker parity handshake rejected different model
content hashes. The default-settings pre-integration CUDA run also ran out of memory.
Both failures are retained. The pre-integration binary completed a narrower run after
using the explicit fit settings, but that is not a valid paired performance comparison.

**Historical evidence only:** the earlier continuation recorded expert parity against
CUDA and CPU for four layers. Its handshake checked geometry rather than model content.
Those measurements cannot supersede the new integrity failure.

**Not run after the gate failed:** real-worker numerical parity, full-model remote inference,
zero-ownership token/logit identity, real GPU disconnect/timeout recovery, same-day paired
throughput and capacity qualification, and GPU/CPU utilization analysis. The benchmark harness below has only passed a syntax check. Default-path token
identity remains unverified. No SYCL, Windows or Metal build/runtime claim is made.

The repeatable harness records raw engine output, token IDs, per-token arrival times,
prefill/decode rates, time to first token, startup time, worker counters, JSON and CSV:

```sh
node bench/distributed_experts.mjs \
  --engine /var/tmp/strata-cuda-ztripez/strata \
  --model /path/to/verified/model.gguf --worker 192.168.63.177:19415 \
  --placement /path/to/placement.txt --out /var/tmp/strata-qualified-new-run
```

It runs baseline, empty-plan, real-worker, disconnection, withheld-response and unavailable
cases sequentially, with three repetitions of three repository prompts. A local test relay
injects the two runtime failures; it does not stop a user service. Keep the real worker
running for those sessions. The harness refuses to overwrite an evidence directory.
Only the direct remote case is suitable for performance comparison; fault relays are
correctness probes. The unavailable case must fail setup. Zero ownership must preserve
tokens. Remote token deviations are recorded and need separate numerical/quality review.

The success gate is still open: stable stock model identity, declared numerical tolerances,
correct complete inference, bounded real-worker failure handling, and a measured useful
throughput or capacity benefit. No threshold was relaxed to pass the failed integrity gate.
