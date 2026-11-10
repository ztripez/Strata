# Experimental executable fabric plans

`strata_control.fabric_plan` defines the logical static placement used to prepare
native DeepSeek launches. `strata_control.fabric_bundle` binds that placement to
an explicit installation epoch, endpoints and paths. Both are opt-in source
tools. They do not rank placements, start processes or publish HTTP.

Consult the [current gate table](DISTRIBUTED_FABRIC_STATUS.md#current-gate-status).
Passing these structural controls does not pass E6. Measured placement costs,
accepted evidence and transactional installation remain required.

## Logical identity

The strict `Plan` records the following:

- DeepSeek family/revision, DS4M1, DS4P1 and expert-model identities; six routed
  experts, Q8_K arithmetic and the 2K serving profile.
- Ordered size/SHA-256 pins for all six mapped backing files, the separate
  tokenizer GGUF, pack metadata and all five exported tokenizer assets.
- Two node identities/backends, source/binary pins, inventory and capability
  evidence pins, node budgets and evidence-bound whole-node requirements.
- Prefix/suffix ownership, a supported cut (20 or 24), local expert arenas,
  optional opposite-node expert ownership, native RAM budgets and explicit
  local recomputation on expert failure.
- Workload identity, protocol versions and declared deadlines.

The prefix owns `[0,cut)` and embedding; the suffix owns `[cut,40)` and the
head. Engram uses the existing assigned-range native path. GGUF storage shards
remain backing files rather than layer partitions. The current native worker
uses its existing two-thread DeepSeek CPU path; this schema adds no CPU pool,
cache, quantizer or paging mechanism.

`Plan.identity()` hashes canonical JSON with the domain `strata-fabric-plan/1`.
Node, worker and pack-source enumeration is normalized. Expert-pair order remains
significant: the native worker fills GPU residency and then copied RAM in that
order. The planner must retain its intended tier priority. Mapped-file order is
also significant because it is part of the native model contract.
`PlanDocument` rejects a claimed hash that differs from its actual plan.
The schema admits static placements only; it cannot label unmeasured constants
as a calibrated speed/capacity ranking.

Capability/resource evidence is pinned, not automatically accepted. A requirement
receipt must account for dense weights, attention/state, all expert arenas,
staging, service buffers, temporary peaks and a safety reserve. RAM requirements
include pinned allocations. Report mapped RSS/PSS and OS page cache separately;
do not add shared mapped pages twice. The installer still needs to validate the
receipt's provenance, scope and acceptance before treating it as fit evidence.

`check_inventory(node, path)` verifies and parses the same pinned bytes, reusing
Strata Control's existing inventory schema. It rejects unknown/exhausted capacity
and contradictory node/binary/backend observations. This is a point-in-time
snapshot check, not a reservation, freshness check or actual runtime fit proof.
The native process remains authoritative for parsed pack/model identity and
effective configuration. No second Python implementation of DS4P1 is introduced.

## Prepare an installation bundle

An `Installation` contains a `PlanDocument`, a positive 64-bit `epoch`, exactly
one `Location` per node, the suffix's numeric IPv4 layer endpoint and a loopback
HTTP port. Each location supplies absolute model, pack, tokenizer, binary and
runtime paths. Expert endpoints/binaries are required only for active workers.
An existing runtime directory is not made safe merely by preparing a bundle.

Paths, endpoints and the epoch are outside the logical plan identity. Reinstalling
the same logical snapshot therefore retains its plan ID while its epoch can
change. Selecting/persisting a newer epoch, draining old sessions and verifying
the actual suffix incarnation remain installer responsibilities.

Locators do not rewrite a pinned pack. The layer engine resolves backing files
from the pack itself; `Location.model_dir` locates the expert-model GGUF passed
to the existing worker/client interfaces. This initial schema pins the same
pack source bytes on both hosts. Rewriting pack metadata, including an embedded
`model_dir`, changes the pinned source snapshot and requires a newly qualified
plan. Equal DS4P1 strings alone do not authorize an unverified pack override.

Prepare a new local review directory from the strict JSON binding:

Supply its existing pinned evidence under
`EVIDENCE_DIR/NODE_ID/inventory/FILE_NAME`,
`EVIDENCE_DIR/NODE_ID/capability/FILE_NAME` and
`EVIDENCE_DIR/NODE_ID/resource/FILE_NAME`. Preparation verifies these exact bytes,
checks the inventory snapshot and copies the receipts into the review bundle.
It refuses changed or missing evidence before creating the output directory.
Checking an evidence file's hash does not establish that its claim is accepted.

```sh
uv run --no-project --with pydantic==2.14.2 --with regex==2026.9.10 --with jinja2==3.1.6 \
  python -m strata_control.fabric_bundle \
  --installation /absolute/path/to/installation.json \
  --evidence-dir /absolute/path/to/pinned-evidence \
  --output /absolute/path/to/new-review-bundle
```

The output contains the canonical binding, per-node native argv/environment
records, native expert placement text and the existing
[prefix HTTP launch binding](DEEPSEEK_LAYER_PROCESS.md#opt-in-http-launch).
The prefix layer command is inspectable; the eventual installer must launch
that process through the HTTP adapter, not start a second prefix beside it.
Worker ownership is installed by the native layer owner's existing expert RPC
handshake, not by a new text-placement interpreter.

`bundle.json` is written last and hashes all preceding files. Its status is
`PREPARED ONLY; no process, reservation or READY`. Existing output directories
are refused. A partially written directory is retained for diagnosis; it is
not an installed or traffic-admitted plan. The installer must revalidate the
binding, file hashes, artifacts, resources and every participant's actual READY
before exposing the endpoint. It must never execute an arbitrary changed argv
file just because a preparation directory exists.

## Controls

Run the source-only controls with an optional built native syntax checker:

```sh
STRATA_TEST_LAYER_BINARY=/absolute/path/to/strata-deepseek-layer \
uv run --no-project --with pydantic==2.14.2 --with regex==2026.9.10 --with jinja2==3.1.6 \
  python -m unittest tools.test_network_fabric_plan
```

The tests use synthetic artifacts/inventories. Native executable checks use
`--check-config` and do not select a device or load a model. They cannot establish
artifact integrity, real fit, model arithmetic, quality or performance.

On 10 November 2026, all 12 manifest/bundle controls and 17 inventory/HTTP
integration tests passed on both Lucy and Astor. Generated prefix/suffix arguments
passed actual CUDA/HIP syntax-only checks. Five CLI controls reproduced all
15 bundle files exactly and refused an existing destination, a stale plan hash
and changed evidence without replacing prior output. Receipts and the synthetic
inputs are retained in `/var/tmp/strata-fabric-plan-controls-20261110/`.
No native model ran as part of these plan controls.
