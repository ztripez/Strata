# Experimental persistent DeepSeek layer process

`strata-deepseek-layer` is a source-checkout adapter for the existing native
DeepSeek Model and [DS4L layer protocol](DEEPSEEK_LAYER_RPC.md). It keeps assigned
layers loaded across requests. It adds no quantization, expert cache or paging
implementation. The existing released executables and launch defaults are unchanged.

This is serving preparation, not a model-qualified network endpoint or installed
E6 plan. Consult the [current gate table](DISTRIBUTED_FABRIC_STATUS.md#current-gate-status)
for acceptance. HTTP/stdio integration has synthetic-process checks; real model
execution through this new process and orchestrator installation remain open.

## Build and configuration

Use a configured CUDA or HIP source build with `STRATA_NETWORK_EXPERTS=ON` and
`STRATA_BUILD_DSV41=ON`:

```sh
cmake --build BUILD_DIRECTORY --target strata-deepseek-layer deepseek_layer_process_test -j2
BUILD_DIRECTORY/deepseek_layer_process_test
```

The network backend is Linux-only and does not support SYCL. The process is an
explicit build target, excluded from the normal build. It requires
`DS4_IQ_Q8K=1` before model execution. Its fixed initial profile is text, one
sequence, greedy decoding, a 2,048-token context, synchronous windows, six routed
experts, no vision/DSpark and no primary expert cache/RAM store. File-backed CPU
experts and fixed native GPU expert placement use the existing Model paths.

The launch options are:

| Option | Meaning |
| --- | --- |
| `--serve` | Explicit persistent-process mode. `serve.server.StrataEngine` already prepends this flag. |
| `--role prefix` or `--role suffix` | Assigned native layer owner. |
| `--pack PATH` | Native execution pack; GGUF storage shards are not layer partitions. |
| `--cut 20` or `--cut 24` | Prefix owns `[0,cut)`; suffix owns `[cut,40)`. Allowed geometry is not runtime qualification of both cuts. |
| `--endpoint IPV4:PORT` | Prefix's destination or suffix's listening endpoint. |
| `--model-sha256 HASH` | Expected DS4M1 identity of all six actual mapped files. |
| `--pack-sha256 HASH` | Expected canonical DS4P1 execution-pack identity. |
| `--plan-id HASH` | Shared nonzero plan identity supplied by the sealed experiment or installer. |
| `--plan-epoch N` | Installer-supplied positive 64-bit installation epoch, identical on both layer owners. Existing static callers that omit it retain epoch 1. |
| `--experts 6` | Explicit routing setting; native boundary setup checks every layer's effective count. |
| `--expert-vram-mib N` | Fixed expert arena budget, `0..65536` MiB. This is not a cap on dense weights or total device use. Infeasible allocation fails at load. |
| `--cpu-threads N` | Native CPU pool limit, `1..64`, not a reservation. Actual pool size is reported by Model startup. |
| `--rpc-timeout-ms N` | One layer transaction's bound, `1..600000` ms. |
| `--trace-windows` | Optional paired window/transaction timing on stderr; default off. Does not select a placement or qualify its cost. |

Prefix also requires `--peer-binary-sha256 HASH` and
`--request-timeout-ms N`. The request bound must cover at least one RPC timeout
and cannot exceed 7,200,000 ms. Suffix instead requires
`--idle-timeout-ms N`, `1..7200000` ms, for accepting a connection. Prefix-only
options are refused on a suffix and vice versa.

Expert RPC is optional. Supply all four expert options together:
`--expert-worker IPV4:PORT`, `--expert-worker-model GGUF`,
`--expert-worker-placement FILE` and `--expert-worker-timeout-ms N`.
The Model retains routing, placement, execution and visible expert fallback.

`--check-config` checks argument syntax only. It reports no READY and does not
select a device or load a model. It does not verify files, fit, compatibility or
plan installation. Nonzero matching plan strings alone do not install a plan.

Before READY, the process allocates dense weights/state and expert arenas only
for its assigned range. It still maps and hashes all six backing files for
DS4M1 verification, checks DS4P1 and checks the serving geometry. It hashes its own
running executable. Prefix additionally discovers and pins the suffix's actual
contract and executable. INFO reports actual artifact/binary identities and
resolved limits. A suffix publishes `LAYER_READY`; a prefix publishes
`READY 2048 stop` for the existing server's stdio engine interface.

An explicit `--plan-epoch` sets the native DS4L contract and every boundary's
epoch. Discovery rejects a peer with a different epoch before native execution.
INFO reports `plan_epoch=N` when explicitly supplied. An installer must supply
and verify it; missing INFO is unknown, never evidence of epoch 1. Plan content
identity, installation epoch and the suffix's process incarnation are separate
fences. This option alone does not provide transactional installation or choose
the next epoch. The existing sealed static fixtures keep their original binaries
and epoch-1 contracts.

## Request and failure behavior

The prefix accepts the existing token-stream protocol:

```text
GEN 128 temperature=0 ckpt=0 42,43,44
STOP
QUIT
```

Only one request may be active or pending. A prompt has at most 1,900 tokens;
the output limit is `1..1024`, and prompt plus output must fit the 2K context.
EOS is token 1. Non-greedy sampling, nonneutral penalties, checkpoints, images,
parallel requests and logprobs are refused before inference. Every request resets
native state and begins a strictly newer fenced layer session.

`PP`, `T` and `DONE` lines use the existing Strata server interface. The coordinator
exposes logits/tokens only after DS4L acknowledgement and position checks.
The normal path matches the bounded window executor, including its final extra
post-token row. A malformed request returns ERR without invoking Model execution.

STOP cancels local work and interrupts an established layer channel. The reader
joins on shutdown and unregisters callbacks before their clients are destroyed.
Queued STOP state cannot leak into a later fresh request. A known cancellation
reports `DONE ... cancel`; native computation and unexpected generation failures
exit nonzero. A request deadline interrupts an established channel and fails the
request rather than reporting a successful completion.

The suffix logs cancelled/disconnected sessions, resets native state and accepts
only a new fenced session. Native preparation or execution failure ends its
process. There is no automatic replay of a partially executed layer request.
Expert RPC keeps its separate, recomputable fallback policy.

Socket interruption cannot kill a GPU kernel or a blocked filesystem operation.
TCP setup retains its declared connect bound. Native cancellation polls and the
request deadline prevent further publication once expiry is observed; a process
supervisor must still enforce hard limits on stuck computation/I/O. This process
does not provide that external supervisor. The opt-in HTTP launcher below owns
its prefix process, but does not supervise remote workers or blocked kernel I/O.

## Opt-in HTTP launcher

`serve.deepseek_layer` uses the existing Strata HTTP server, chat template,
tokenizer and stdio engine transport. It serves one explicitly specified prefix
on `127.0.0.1`. Start the pinned suffix and any expert workers separately first.
It does not discover machines, calibrate placement or install a cluster plan.

Run it in a dedicated process from the source checkout:

```sh
uv run --no-project --with pydantic==2.14.2 --with regex==2026.9.10 --with jinja2==3.1.6 \
  python -m serve.deepseek_layer /absolute/path/to/launch.json
```

The strict JSON launch binding has these fields. Generate it from the sealed
experiment or installer; do not substitute a hash of a storage-shard filename
for a model or pack identity.

| Field | Required value |
| --- | --- |
| `schema_version` | `strata-layer-http-launch/1`. |
| `executable` | Absolute path to the pinned `strata-deepseek-layer` prefix executable. |
| `native_args` | Array of the native options above, beginning with `--role`, `prefix`. Omit `--serve` and `--check-config`; the launcher supplies them. |
| `expected` | Expected native INFO receipt: `arch=deepseek41`, `fabric=1`, `role=prefix`, `context=2048`, `effective_experts=6`, `arithmetic=2`; the four `model_sha256`, `pack_sha256`, `plan_sha256`, `binary_sha256` hashes; and the explicit `cpu_threads_limit`, `expert_vram_mib`, `request_timeout_ms`, `rpc_timeout_ms`, `idle_timeout_ms=0`. Use JSON numbers for numeric fields. |
| `tokenizer_dir` | Absolute directory containing exactly the five exported assets below. |
| `tokenizer` | Object with `assets_sha256` mapping each of `vocab.json`, `merges.txt`, `tokenizer.json`, `token_type.json`, `chat_template.jinja` to its SHA-256; and `source_sha256` pinning `tools/strata_tokenizer.py`. |
| `engine_log` | Absolute path to a new log. An existing file is refused and preserved. |
| `engine_ready_s` | Explicit startup allowance, greater than zero and at most 7,200 seconds. Model hashing is part of startup. |
| `port` | Optional loopback HTTP port, default `8098`. |
| `max_queue`, `queue_timeout_s` | Optional bounded queue, default zero waiting requests and 30 seconds. |
| `model_name` | Optional API model name, default `deepseek-v4.1-flash-fabric`. |

All hashes are 64 lowercase hexadecimal characters. Before HTTP opens, the
launcher verifies the executable and tokenizer, runs the native syntax preflight,
then compares actual READY/INFO with the expected identity and resource limits.
It prevents the ordinary server's leftover-process cleanup and automatic engine
retry/reload for this dedicated fabric process. A failed prefix needs an explicit
fresh start; a partial generation is not replayed.

For an explicit epoch, include both `--plan-epoch`, `N` in `native_args` and
`"plan_epoch": N` in `expected`. The launcher rejects missing, duplicate or
conflicting bindings before starting the native process. It then requires the
actual INFO epoch to match before HTTP can open. Static bindings may omit both;
they do not attest an explicit installation epoch.

The existing `/v1/chat/completions` endpoint supports ordinary and streaming text
responses here. Defaults are greedy. Unsupported sampling, logprobs, checkpoint,
image and tuning controls are refused before a native GEN write. Prompt plus
requested output must fit 2K, with the same native prompt/output bounds above.
SIGINT/SIGTERM during startup or serving closes the owned prefix through the
existing bounded STOP/QUIT, terminate and kill sequence. Remote layer/expert
processes remain the external controller's responsibility.

## Verification scope

The opt-in [window-observation reader](NETWORK_FABRIC_INVENTORY.md#native-window-observations)
pairs native timing records by actual identity, incarnation and window. These
records are preparation for measured placement costs. Transaction time already
includes remote computation. No native calibration dataset has been accepted yet.

On 10 November 2026, CUDA and HIP builds passed. CPU configuration, request,
STOP/reset, callback-lifetime, input-bound and deadline checks passed on Lucy
and Astor. The CPU suite passed AddressSanitizer, UndefinedBehaviorSanitizer and
ThreadSanitizer on Lucy. Eight actual executable admission cases per host checked
valid syntax, the explicit pre-device arithmetic guard and refused configurations.
None selected a device, loaded a model or published READY.

Receipts, source snapshots and pinned binaries are retained in
`/var/tmp/strata-deepseek-layer-process-20261110/`. The earlier prototype and its
checks remain separate from the version with an explicit request deadline.
These are build/control checks, not native inference, numerical quality, HTTP
cancellation, orchestration, capacity or performance acceptance.

HTTP adapter receipts are in
`/var/tmp/strata-deepseek-layer-http-20261110/`. Twenty-one focused tests passed,
including synthetic stdio admission, actual loopback HTTP/SSE, greedy defaults,
strict request limits, preserved logs, refusal of automatic startup retries and
child cleanup on startup/serving termination. The existing server suite passed
68 tests with one skipped; that
tokenizer test passed separately with the pinned real assets. The common loader
also decoded an existing excluded native EOS capture correctly, and all 79 E5
pre-data files retained their sealed hashes. These checks do not run a native
model through HTTP or qualify native cancellation, E5 quality or E6 installation.

Explicit installation-epoch controls are retained separately in
`/var/tmp/strata-deepseek-plan-epoch-20261110/`. CUDA and HIP builds passed;
13 HTTP/stdio tests passed on each host, including rejection of missing and
conflicting actual epochs. Sixteen native syntax/pre-device cases passed per
host, alongside the process-input and TCP contract suites. These checks do not
execute a model with a new epoch or implement transactional plan installation.

## Native endpoint qualification

The frozen continuation in `/var/tmp/strata-deepseek-layer-api-20261110-r2/` is
queued behind successful fresh E5 acceptance. All 254 pinned input/control files
match on both hosts. The original 281-file trial correctly refused to launch
after an upstream observer failure; its old preparation records remain there.
The continuation retains the original binaries, frozen serving bundle, request
and EOS-oracle bytes, resource limits and timeouts. No model has run through
either HTTP fixture yet.
Five fixture controls passed on each host, including failure capture, cancellation
drain ordering, observing layer loss without disconnecting the client, refusal of
an incorrect process birth time and refusal to launch after a failed prerequisite.
Three native syntax/pre-device admission controls passed per host. These are
preparation checks, not endpoint inference results.

The static profile assigns `[0,20)` to HIP on Astor and `[20,40)` to CUDA on Lucy.
Both layer owners request eight CPU threads, six experts and a 384 MiB native
expert arena. Opposite-host expert workers each have 20 MiB GPU and 20 MiB copied
RAM budgets, with remaining assigned identities using the existing mapped lane.
Model/pack identities, binaries, the serving Python bundle, resource limits and
the excluded mixed-host EOS token oracle were pinned before API output.

The controller requires ordinary and streaming EOS responses to reproduce the
fixed IDs. It then disconnects a long stream after visible text, requires a native
cancellation receipt and checks exact EOS replay on the next fresh request. Both
expert directions must execute mapped work without fallback. Finally it stops
only the owned suffix, verified by argv, process birth time and a Linux pidfd.
The failed request must abort; subsequent traffic must receive HTTP 503 without
automatic replay. That loss client stays connected while observing the failure.

Startup bounds are 30 minutes per expert worker and 45 minutes per layer owner.
Requests have a 30-minute limit; cancellation/loss observation has 60 seconds.
The active wall limit is eight hours, separate from the 72-hour prerequisite wait.
Before loading, the controller checks that prior native processes are gone, its
private storage lease covers the active wall plus 30 minutes, and each evidence
filesystem has at least 2 GiB free. This fixture saves token records and telemetry,
without E5's full-vocabulary logit dumps. Raw memory/CPU/I/O and private NFS
statistics remain separate from performance qualification.

Acceptance requires `driver.exit` equal to `0` and `summary.json` status
`native_hybrid_http_functional_cancel_and_layer_loss_pass`. No accepted receipt
exists yet. This is a static endpoint/lifecycle check; it does not replace E5
quality, install an E6 plan or qualify E7 capacity/speed or E8 release readiness.
