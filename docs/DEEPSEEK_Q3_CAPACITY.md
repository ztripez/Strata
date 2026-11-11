# DeepSeek Q3 capacity milestone

The completed vertical capacity target is the complete Unsloth
[DeepSeek V4.1-Flash-1017 UD-IQ3_XXS set](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF).
The existing 2.00-bpw result is frozen as the performance reference. The 2.25-bpw
step and cut16/18 placement sweep are parked. Q4 is outside this milestone.

## Current status

Updated 11 November 2026, 10:25 CET. Retry5 completed both layer-only and hybrid
requests at cut28: each consumed 1,900 prompt tokens and generated 148 tokens,
reaching an actual 2,048-token sequence. The complete published Q3 set ran within
the unchanged 14/9 GiB VRAM and 48 GiB per-process RSS ceilings. Hybrid execution
used remote CPU experts in both directions, with zero fallbacks.

**Vertical inference is complete. Quality refinement and performance are deferred
under the user's subsequent direction to prioritize getting models running.**
The separate frozen quality check failed: the layer-only task check retained
three of four baseline passes. The prime task returned `{"answer":true}` instead
of the frozen exact answer `true`; the scorer and retention rule are unchanged.
The pipeline stopped before hybrid quality and numerical comparison. This is an
observed output-format regression, not evidence of an incorrect primality result.
No strict quality pass or automatic numerical waiver has been applied. This
recorded failure does not reopen the completed vertical inference milestone.

Astor's generation service was restored, embedding remained active, and both
hardware reservations released after cleanup. Earlier failed attempts below are
historical evidence and do not override these completed execution results.

| User gate | Status | Evidence and remaining work |
| --- | --- | --- |
| Q3-0 — compatibility | EXECUTION DEMONSTRATED | Mixed-format checks passed both backends; actual cut28 requests completed. Remote GPU expert quality remains unqualified. |
| Q3-1 — loading | COMPLETE SHARED BACKING LOADED | All seven published files, including the original Q6_K Engram, supplied both owners. No full local Q3 copies were required. |
| Q3-2 — execution and bounded quality | VERTICAL EXECUTION COMPLETE; QUALITY DEFERRED | Both actual 2K requests completed. Hybrid dispatched 3,409 remote assignments to Lucy and 3,551 to Astor with zero fallbacks. Layer-only task check scored 3/8, losing the baseline prime exact-answer pass. Hybrid quality and numerical comparison were not run. |
| Q3-3 — memory | OBSERVED FIT, CUT28 | No budget-exceeded samples or telemetry gaps in either 2K request. Hybrid device peaks: Astor 6,942,351,360 bytes; Lucy 8,329,887,744 bytes. Layer-owner RSS HWM: Astor 45,703,952 KiB; Lucy 32,404,764 KiB. Native and remote expert VRAM budgets were zero. |
| Q3-4 — throughput | MEASURED; 5 TOK/S NOT MET | Hybrid prefill 1.618 and decode 0.512 tokens/s. Layer-only prefill 0.911 and decode 0.115 tokens/s. One sequential request per mode, inherited caches and different background allocations: no controlled speedup or isolated quantization-cost claim. |
| Q3-5 — higher-fidelity capacity | VERTICAL Q3 CAPACITY COMPLETE | Complete 394.74 GB Q3 inference finished within the fixed memory limits. Hybrid ended with 78,344,953,856 unique model-backed cache bytes across both hosts and 151,552 duplicated bytes. This establishes capacity; improved output quality remains unqualified. |

Retry5's `driver.exit` is 0. Its `pipeline.exit` is 1 because task retention
failed; `service-restore.exit` is 0. The execution `summary.json` covers only the
successful 2K runs. It does not supersede the later quality failure.

The historical E7 target remains **5 decode tokens/s**, with its failed result
preserved. Performance is deferred and is not a prerequisite for the current
vertical objective of getting models running.
Original E5 thresholds and the accepted 2.00-bpw experimental checkpoint remain
unchanged. Synthetic kernel checks do not qualify full-model quality or remote
GPU expert execution.

## Artifacts and execution path

Hugging Face revision: `2688c0c3c2c3b052ec4f6d10a8417b0ef7ee4eba`.
The seven files total **394,736,596,867 bytes**. The first is metadata-only;
the last contains the **161,284,798,500-byte Q6_K Engram tensor**. The complete
published Engram stays in this experiment; it is not replaced with the smaller
IQ4_NL Engram used by the 2.00-bpw reference.

The header inspection found these GGUF types: F32, Q8_0, BF16, MXFP4, Q3_K,
Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S. The routed expert tensors total
224,298,270,720 bytes. These are header-derived sizes, not measured residency.
The target name does not mean every tensor is encoded as IQ3_XXS.

Shared download location:
`/mnt/storage/strata-models/deepseek41-unsloth-ud-iq3-xxs/UD-IQ3_XXS/`.
The existing private read-only NFS mapping supplies the native processes. Strata's
existing packer produces tensor offsets and tokenizer/Engram metadata without
copying or requantizing weights.

The first execution uses the qualified cut20 boundary: Astor owns layers 0–19;
Lucy owns layers 20–39. The existing native layer harness first runs layer-only,
then hybrid with bidirectional remote experts. Each request uses the original
1,900-token input and generates 148 tokens, reaching an actual 2,048-token
sequence. The initial native expert VRAM budgets were 209 MiB on Astor and 172 MiB on
Lucy. After the observed device-wide ceiling stop, retry2 sets both to 0 MiB.
This residency change must be included when comparing throughput with 2.00-bpw. Remote workers use 0 MiB expert VRAM and 20 MiB RAM staging each; other
assigned experts use mapped backing. These allocations are experimental inputs,
not a claim that Q3 fits.

No engine rewrite, cache manager or transport change was introduced. The initial
Q3 correction was in the existing expert test's random-block generator:
`block_q3_K` stores its F16 scale at byte 108, not byte 0. The initial test left
the real scale random and reported invalid failures. With the scale written to
the correct field, token, prompt-set and CPU-row comparisons passed on both
backends under their original tolerances. Initial failed logs are preserved.

## Evidence and limits

Working evidence: `/var/tmp/strata-deepseek-q3-20261111/`.

- `tensor-header-inspection.json`: metadata and tensor directories fetched with
  HTTP byte ranges. Header files are isolated and never used for inference.
- `iq3-cuda.log`, `iq3-hip.log`: initial IQ3_XXS checks.
- `q3-additional-formats-cuda.log` and the corresponding Astor log: initial
  invalid Q3_K test inputs, preserved as failures rather than engine evidence.
- `q3-formats-cuda.log`, `q3-formats-hip.log`: corrected Q3_K/IQ3_XXS/IQ3_S
  checks, both exit 0.
- `driver.log`, `conditions.json`, native command/log/telemetry files: queued
  actual execution. A queued command is not a completed inference result.

The existing handshake identities will be generated from the complete Q3 set.
The 2.00-bpw model and pack identities must not be reused for Q3. Repeated client
artifact scans are unnecessary after those identities have been established.

For the completed 2.00-bpw measurements and limitations, see
[actual 2K capacity and storage measurements](DEEPSEEK_CAPACITY_PERFORMANCE.md).

The first load failure is preserved in `driver.log` and
`logs/layer-only-cut20-suffix.log`. The required worker model identity was
calculated once on pve01 and retained in `worker-model-identity.json`; the retry
reuses it. The mixed-format change adds MXFP4 to the existing IQ GPU dispatch
and generic ggml CPU row allowlist. It does not requantize weights. Uniform
MXFP4 execution retains its existing path. The focused expert test now accepts
`--down-type 39` so it checks the actual failing format pairs rather than only
uniform expert formats.

The complete Q4 download also finished successfully at 02:33 CET under
`/mnt/storage/strata-models/deepseek41-unsloth-ud-q4-k-xl/`. This is download
completion only; Q4 execution is not part of the current retry.

Retry evidence: `/var/tmp/strata-deepseek-q3-20261111/retry1/`.
`mixed-all-pairs-cuda.log` and `mixed-all-pairs-hip.log` in the parent directory
record all five actual mixed pairs passing at unchanged tolerances. The original
Q3 tokenizer vocabulary, token types, special IDs and merges match the frozen
task inputs. The 2.00-bpw baseline, strict E5 thresholds and experimental
acceptance scope have not changed.

Retry2 evidence: `/var/tmp/strata-deepseek-q3-20261111/retry2/`. The executable,
model, pack, cut20 layer split, six experts and request inputs are unchanged from
retry1. Only the two native expert VRAM budgets change to zero. Both the first
format rejection and the later ceiling failure remain preserved in their
original directories. The model is not redownloaded, and the previously computed
worker model identity is reused.

Retry3 uses the earlier E2-style raw identity convention through the existing
trusted-identity path, with explicit log labels rather than DS4M1/DS4P1 claims.
The complete seven-GGUF concatenation was hashed once locally on pve01:
`a1fa850b77bdf33420ebeafd5cf67607b0077e9f3a958ab15666fe2d970a7d5f`.
The SHA-256 of `ds4.txt + tensors.txt + engram.bin` bytes was calculated on both
hosts and agreed: `9e3919c45430c4a03c17a827b5036da173405db8c5c3386f5b77d5409b2875d5`.
This avoids further per-client whole-model scans. It changes neither model math
nor wire framing, and the original E5 artifact domains remain unchanged.

The approved, running service wrapper is
`/var/tmp/strata-deepseek-q3-20261111/retry4/run-with-service-park.sh`.
It stops `llama-server.service`, runs actual Q3 inference and bounded quality,
then restores that service on ordinary success or failure. It leaves
`llama-embedding.service` running. The user approved this temporary service interruption. Retry4 started at
06:46 CET, with the embedding service confirmed active and generation confirmed
inactive. A shell EXIT handler and service ExecStopPost restore generation on
completion or failure.

Retry4 is owned by `strata-q3-r4-20261111.service`. Evidence is under
`/var/tmp/strata-deepseek-q3-20261111/retry4/`; `pipeline.log`,
`service-restore.exit` and `pipeline.exit` record lifecycle completion. The existing
private NFS tunnel lifetime was extended to 24 hours without restarting the live
SSH processes, replacing its earlier 08:04 CET expiry during the active run.

Retry4 partial output and Astor telemetry were collected after the failed request:
`retry4/results/layer-only-cut24/2k/windows.jsonl` ends at window 115, committed
position 1840. The suffix monitor's VRAM stop explains the broken pipe; private
NFS tunnels remained active. `service-restore.exit` is 0 and both Astor services
were confirmed active after cleanup. Retry5 evidence is under
`/var/tmp/strata-deepseek-q3-20261111/retry5/`, owned by
`strata-q3-r5-20261111.service`. Cut28 is an experimental native placement, not
a qualification of public orchestrator installation or a strict E5 pass.

Retry5's shared hardware reservations were released after pipeline cleanup and
generation-service restoration. Read live [hardware ownership](HARDWARE_LOCK.md)
before another run; this completed job does not reserve either host.

Completed execution receipts: `retry5/layer-only-cut28-measurements.json`,
`retry5/hybrid-cut28-measurements.json`, `retry5/summary.json` and native logs.
Quality failure receipts: `retry5/quality-layer-only-tasks.json`,
`retry5/quality-layer-only-retention.json` and `retry5/quality.log`.
All raw captures remain in the original retry5 directory. The 2.00-bpw baseline,
its strict E5 failure and experimental acceptance remain frozen.
