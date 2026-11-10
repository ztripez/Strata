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
[prefix HTTP launch binding](DEEPSEEK_LAYER_PROCESS.md#opt-in-http-launcher).
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

## Node agent and installation ownership

`strata_control.fabric_node` provides the node side of installation. It verifies
the declared node's seven backing files, pack files, binaries, exported tokenizer
and pinned evidence. It then takes a fresh reading through the existing Control
inventory and checks the same resource constraints. It streams large file hashes;
it creates no model-weight cache or mirror.

The agent takes a per-node advisory lock in a separate persistent state directory.
An epoch must be newer than the stored epoch. The epoch record is atomically
replaced and fsynced before native processes can start. Failed/partial starts do
not reset it. A corrupt record, another active owner or an existing runtime
directory causes refusal. Keep this state directory outside the fresh runtime
directory; it must survive installation cleanup.

The internal entry point is:

```sh
uv run --no-project --with pydantic==2.14.2 --with regex==2026.9.10 --with jinja2==3.1.6 \
  python -m strata_control.fabric_node \
  --installation /absolute/path/to/bundle/installation.json \
  --node astor --evidence-dir /absolute/path/to/bundle/evidence \
  --state-dir /absolute/path/to/persistent-node-state
```

The controller keeps stdin open and sends one JSON command at a time, awaiting
its reply. Commands have increasing positive `id` values and an `action` of
`verify`, `start-worker`, `start-suffix`, `ownership`, `status` or `stop`.
Commands carry no arbitrary argv or per-token work. The agent derives native
arguments from the canonical plan. Only the suffix can start here; the coordinator
must start the prefix through the existing HTTP adapter after node preparation.

Worker listening and ownership READY are separate states. Admission must check
both: actual model identity/six-expert arithmetic and exact installed ownership,
tier counts and GPU/copied-RAM bytes. Suffix admission checks its actual INFO,
epoch, process incarnation and all six native post-load file hashes. A PID alone
never identifies a process: startup records include argv, birth token and the
running executable's hash. Linux pidfds bind subsequent signals to that process
instance. The existing lifeline and owner guard cover parent loss.

`stop` closes owned children before replying and retains raw per-process exits.
EOF, malformed/pipelined commands and failed operations terminate the agent and
unwind its owned resources. Atomic, exclusive record publication preserves prior
evidence. A failed expert process remains visible in status; it may be handled by
the plan's existing native local-recomputation policy. A lost stateful layer must
abort the request. No process is automatically restarted.

This agent never opens HTTP or publishes a whole plan. The coordinator below
owns the global startup barrier and HTTP lifetime. Actual model installation,
safe reload/quarantine qualification, accepted capability/resource evidence
and measured placement ranking remain separate requirements. A pinned evidence
file is not automatically an accepted qualification claim. Kernel-blocked I/O
can outlive a stop deadline; it needs an explicit failure/quarantine outcome.

The node controls use a synthetic C++ executable with no model parsing, GPU or
inference. They test real file verification, process/epoch ownership, bad receipt
refusal, raw exits and parent EOF during startup. They do not qualify DeepSeek
execution through the agent.

On 10 November 2026, 13 node/lease/process controls and the 29 focused
manifest/inventory/HTTP controls passed on each host, with no skips. The argument
checks used the actual CUDA/HIP layer executables; model-free lifecycle tests used
the explicitly synthetic ELF fixture. Receipts are retained in
`/var/tmp/strata-fabric-node-controls-20261110/`. Native DeepSeek execution and
actual model activation through this agent have not run.

## Two-node coordinator and HTTP publication

`serve.fabric_install` connects the node agents to the existing native prefix
adapter. Run it in a dedicated Linux process on the prefix owner, using Python
3.11 or newer. This is an opt-in experimental entry point; the ordinary server
path and model arithmetic are unchanged.

A strict `CoordinatorSpec` embeds the complete `Installation`, two `AgentRoute`
records, a fresh absolute `journal_dir`, health polling/deadline bounds and a
graceful drain bound. Each route specifies its `node`, absolute `python`,
`repository`, staged `installation`, `evidence_dir` and persistent `state_dir`.
The prefix route is local. The suffix route additionally requires a numeric
`ssh_address` and explicit `ssh_host_alias`; `ssh_user` is optional. SSH uses
batch mode, strict host-key checking and keepalives. It does not fall back to
another interface, request credentials or restart a failed process.

Stage the same installation binding, source, dependencies and pinned evidence
on both hosts before starting. The agent verifies a canonical SHA-256 of the
complete installation object supplied by the coordinator. Equal logical plan
IDs and epochs do not authorize different path/endpoint bindings. This assumes
trusted operator-controlled source and routes; it is not a code-distribution
or remote-attestation system.

Use a persistent interpreter for `AgentRoute.python`, for example a runtime
project's `.venv/bin/python`. An interpreter printed by `uv run --no-project`
can disappear when that command exits. The coordinator neither creates that
runtime nor installs packages on a host. Existing `uv` projects can pin
`pydantic==2.14.2`, `regex==2026.9.10` and `jinja2==3.1.6` as in the controls.

Construct the strict configuration from the existing binding, then run it:

```python
from pathlib import Path
from serve.fabric_install import CoordinatorSpec
from strata_control.fabric_bundle import Installation
from strata_control.fabric_client import AgentRoute

binding = Path("/absolute/path/to/installation.json")
spec = Installation.model_validate_json(binding.read_bytes())
routes = tuple(
    AgentRoute(
        node=node,
        python=Path("/absolute/path/to/runtime/.venv/bin/python"),
        repository=Path("/mnt/code/projects/Strata"),
        installation=binding,
        evidence_dir=Path("/absolute/path/to/pinned-evidence"),
        state_dir=Path("/absolute/path/to/persistent-node-state"),
        # Example: Astor prefix, Lucy suffix. Use the actual suffix's address.
        ssh_address=None if node == spec.document.plan.prefix.node else "192.168.60.155",
        ssh_host_alias=None if node == spec.document.plan.prefix.node else "lucy",
    )
    for node in (spec.document.plan.prefix.node, spec.document.plan.suffix.node)
)
config = CoordinatorSpec(
    installation=spec, routes=routes,
    journal_dir=Path("/absolute/path/to/new-installation-journal"),
)
with Path("/absolute/path/to/coordinator.json").open("x") as output:
    output.write(config.model_dump_json(indent=2) + "\n")
```

```sh
uv run --no-project --with pydantic==2.14.2 --with regex==2026.9.10 --with jinja2==3.1.6 \
  python -m serve.fabric_install /absolute/path/to/coordinator.json
```

Startup follows this order:

1. Both agents verify files/evidence, recheck resources and stage fresh inputs.
2. Both declared expert workers report actual model identities and listeners.
3. The suffix reports its actual loaded artifacts, epoch, profile and incarnation.
4. The existing HTTP adapter loads the prefix and verifies its native READY.
5. Both workers report installed ownership and bounded native tier allocations.
6. The coordinator checks all actual process identities/liveness and writes
   `startup-barrier.json`. Only then does it bind loopback HTTP. Inference stays
   drained until `published.json` is written.

Startup failure unwinds owned processes without leaving a published endpoint.
The journal retains replies, stderr and raw process outcomes. Publication-record
failure closes the bound socket while inference admission is still drained.
Periodic control health checks do not carry token or expert computation. After
publication, an expert-process exit remains visible and retains the native
local-recomputation policy. Agent or layer loss ends this installation; there is
no automatic layer replay, placement change or replacement process.

Shutdown drains admission, cancels remaining owned calls, closes the prefix,
then stops both agents. Failures are retained in `coordinator-exit.json` and
raised after attempting the other cleanups. Its `unconfirmed_nodes` list marks
agents without an acknowledged stop. An exited SSH process alone does not prove
remote quiescence. Unreachable hosts or kernel-blocked native I/O still require
an explicit quarantine/reconciliation policy before safe reload can be qualified.
There is no automatic reinstall path in this entry point.

The coordinator controls use the synthetic ELF fixture with real agent pipes,
the existing prefix adapter and loopback HTTP. Synthetic probes replace hardware
and tokenizer admission; SSH is mapped to local exec in those unit tests.
Separate real local/10Gb SSH controls check agent status, clean stop and changed
binding rejection without staging or running a model. These checks do not prove
DeepSeek execution through the coordinator, accepted resource fit, E5 quality,
measured placement ranking or full E6 acceptance. Receipts are retained in
`/var/tmp/strata-fabric-install-controls-20261110/`.
