# Experimental network resource observations

The E6 resource collector records facts without selecting or installing a plan.
It is separate from the app and released Terminal paths. Run from a source
checkout with an explicit binary and storage paths:

```sh
uv run --no-project --with pydantic==2.14.2 python -m tools.network_fabric_inventory \
  --engine /absolute/path/to/engine \
  --storage /absolute/path/to/models --storage /absolute/path/to/local/storage \
  --output /absolute/path/to/new-inventory.json
```

Existing output files are refused. The collector reads the binary to record its
SHA-256; it does not execute it or load model weights. RAM, CPU and GPU facts use
`strata_control.tune_machine`; backend detection uses `strata_control.updater`.
An unreadable probe remains unknown. Zero available memory remains zero.
Contradictory readings fail validation. Physical cores are not guessed from the
logical thread count. Paths on one local filesystem share one capacity receipt.
Do not add the same NFS export's capacity across nodes.

These readings are observations, not reservations. A snapshot collected during
another workload does not describe idle resources. Backend detection and a binary
hash do not establish model or quantization compatibility. Native capabilities,
artifact identities, state ownership and a fit check remain required before READY.

## Native window observations

The experimental `strata-deepseek-layer` accepts `--trace-windows` on both layer
owners. It adds `INFO window_trace=1` and writes `FABRIC_WINDOW` JSON lines to
stderr. The default is off. This adds no tensor, quantization, cache or paging
implementation and changes no DS4L wire fields.

Prefix records split one synchronous `Model::feed` around its boundary RPC:
`before_rpc_us`, `rpc_us` and `after_rpc_us`. `feed_us` is the enclosing span;
separate microsecond rounding can leave a difference of at most two microseconds.
The optional transaction record measures EXECUTE/RESULT and ACK/ACKED separately
and counts their actual frame headers and payload bytes. It excludes connection
setup, BEGIN/CLOSE and TCP/IP overhead. Both result and timing are published only
after ACKED, under the same cancellation fence. Failed/interrupted transactions
leave both caller outputs untouched.

Suffix records measure native replay plus host-logit readback. **That work is
already inside the prefix RPC time.** Do not add it to RPC as a separate cost.
The paired difference also includes serialization, validation, protocol, ACK,
remote trace writing and possible clock-rate error. It is not pure network latency.
No absolute timestamps from different machines are subtracted. Local prefix trace
writing occurs after `feed_us`; remote trace writing remains inside RPC.

Every record binds actual model/pack/plan and executable hashes, the suffix's
incarnation, epoch, session, sequence, position, rows, cut and configured CPU/expert
arena limits. Two writers cannot interleave parts of a record on the same stream.
Trace write/flush errors fail the native caller. Suffix traces can describe work
whose acknowledgement never completed; the reader refuses unpaired records.

Join complete logs with independently pinned inventories and workload bytes:

```sh
uv run --no-project --with pydantic==2.14.2 python -m tools.network_fabric_calibration \
  --binding /absolute/path/to/window-binding.json \
  --prefix-log /absolute/path/to/prefix.log --suffix-log /absolute/path/to/suffix.log \
  --prefix-inventory /absolute/path/to/prefix-inventory.json \
  --suffix-inventory /absolute/path/to/suffix-inventory.json \
  --workload /absolute/path/to/workload.json \
  --output /absolute/path/to/new-window-observations.json
```

The strict binding has `schema_version: strata-window-calibration-binding/1`,
`model_sha256`, `pack_sha256`, `plan_sha256`, `epoch`, `incarnation`, `cut`,
`workload_sha256`, `expected_windows`, and `prefix`/`suffix` objects. Each participant
specifies `node`, `inventory_sha256`, `binary_sha256`, `cpu_threads_limit` and
`expert_vram_mib`. Hashes are SHA-256 of exact bytes. Obtain the suffix incarnation
from its independent readiness receipt. Obtain window counts from the declared
workload/capture receipt; counting the same trace cannot prove its completeness.
Verify consumed token prefixes separately: this reader binds declared workload
bytes but does not attest their consumption. Verify expert coverage, fallback
counts and backing/cache state with separate runtime receipts before using any
of these observations for calibration.

The reader verifies and hashes the same input buffers. It rejects changed profiles,
mixed incarnations, duplicate/stale/gapped sessions, context overflow, missing
peers and impossible spans. Grouped summaries are recomputed from paired raw
records when validating a saved receipt. Residual percentiles come from paired
samples; subtracting two marginal percentiles gives a different quantity. A
one-row window can be prompt work, so groups are labeled by row count, not decode.
Existing output files are preserved.

These are window observations, not complete placement calibration. Per-layer
dense/expert costs, worker GPU/RAM/mapped isolation, backing-I/O state, held-out
prediction error and transactional installation remain open. Reusing a receipt
also requires checking its source hashes against the retained input files.
No native model has yet produced an accepted calibration dataset with this mode.

## Observation-tool verification

On 10 November 2026, CUDA/HIP builds and CPU transport/input controls passed on
Lucy and Astor. Six actual executable admission cases per host checked the
default/trace syntax and pre-device arithmetic guard; none loaded a model or
selected a device. AddressSanitizer/UndefinedBehaviorSanitizer and ThreadSanitizer
passed the CPU transaction tests on Lucy; ThreadSanitizer also passed the process
input/trace tests. Fifteen Python inventory/calibration tests passed. The actual
C++ JSON writer was parsed by Python using explicitly synthetic scalar values.
These checks establish build/schema/control behavior, not measured model costs.

Receipts and original failed preparations are retained in
`/var/tmp/strata-deepseek-window-cost-20261110/`. The first HIP link missed the
ROCm library search path; the command-local `ROCM_PATH=/opt/rocm` and
`LIBRARY_PATH=/opt/rocm/lib` build passed. No system library or driver was changed.

Every receipt says `UNCALIBRATED; no plan ranking or READY admission`. The tool
does not report compute, backing-I/O or network costs. E6 still needs measured
cost receipts, held-out placement validation, deterministic plan generation and
transactional installation. Strata remains responsible for local expert staging.

## Draft audit

The following are source findings from 10 November 2026, not runtime measurements.
The untracked `tools/fabric/orchestrator.py` was inspected and left unchanged.
It cannot be used as the accepted E6 planner:

- `discover_local_node` labels constant CPU, GPU and cold-page costs as measured.
  `discover_link` supplies constant TCP throughput and fallback latency values.
  Probe failures can become plausible resource/cost values.
- `evaluate_candidate_plan` subtracts a constant overlap allowance and assumes
  a constant dense spine, per-expert size and memory overhead. These are not
  calibrated costs for the complete DeepSeek pack or accepted E4 configuration.
- `build_manifest` hard-codes Qwen geometry, eight picks and one dense owner.
  It does not express the independent DeepSeek layer and expert ownership contract.
- The emitted `qwen36-PROFILE-NL` plan label is not a content hash binding the
  inventory, model, workload, binaries, placement and execution settings.

Retain that draft as historical material. The accepted planner must use explicit
unknowns and reject missing calibration rather than importing these constants.
The new collector addresses observations only; it is not an E6 pass.
