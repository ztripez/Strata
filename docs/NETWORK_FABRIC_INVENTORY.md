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
