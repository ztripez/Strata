# Experimental persistent DeepSeek layer process

`strata-deepseek-layer` is a source-checkout adapter for the existing native
DeepSeek Model and [DS4L layer protocol](DEEPSEEK_LAYER_RPC.md). It keeps assigned
layers loaded across requests. It adds no quantization, expert cache or paging
implementation. The existing released executables and launch defaults are unchanged.

This is serving preparation, not a qualified network endpoint or installed E6
plan. Consult the [current gate table](DISTRIBUTED_FABRIC_STATUS.md#current-gate-status)
for acceptance. Real model execution through this new process, HTTP integration
and orchestrator installation still require separate checks.

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

Every launch must supply these options:

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
| `--experts 6` | Explicit routing setting; native boundary setup checks every layer's effective count. |
| `--expert-vram-mib N` | Fixed expert arena budget, `0..65536` MiB. This is not a cap on dense weights or total device use. Infeasible allocation fails at load. |
| `--cpu-threads N` | Native CPU pool limit, `1..64`, not a reservation. Actual pool size is reported by Model startup. |
| `--rpc-timeout-ms N` | One layer transaction's bound, `1..600000` ms. |

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
does not provide that supervisor or an HTTP launcher yet.

## Verification scope

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
