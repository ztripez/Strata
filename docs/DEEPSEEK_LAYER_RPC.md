# DeepSeek layer RPC

Experimental Linux TCP delivery contract, behind `STRATA_NETWORK_EXPERTS`.
This implements delivery around the [native layer state contract](DEEPSEEK_LAYER_STATE.md).
The opt-in native harness connects the executors. Its first one-prompt
mixed-backend TCP run passed; the complete E3 gate remains open.

`deepseek_layer_rpc.hpp` supplies a serial suffix service and a persistent client.
The service delegates preparation and computation to callbacks. Native `Model`
remains the owner of attention, KV state and expert staging. The delivery service
owns only session fencing and one retained, unacknowledged result.

## Wire contract

DS4L version 1 is separate from expert RPC versions 1 and 2. Each frame has a
24-byte little-endian header: magic `DS4L` (4), version (2), type (2), strictly
increasing request ID (8), payload length (4), and zero flags (4). Replies echo
their request ID. Unknown types/versions, nonzero flags and invalid lengths fail
before allocating the payload. Capture and result bodies keep their separate
DS4B and DS4R encodings; no native struct or pointer is sent.

The initial contract requires cut 20 or 24, six effective routed experts and the
qualified stock `DS4_IQ_Q8K=1` arithmetic. Model/pack hashes, plan ID, epoch, cut,
trusted boundary layout and source descriptors are immutable within a service.
Session/window fields in this discovery contract are zero.

| Request / response | Body and bound |
| --- | --- |
| DISCOVER / CAPABILITIES | Empty request. At most 4 KiB of actual contract, worker executable SHA-256, CUDA/HIP backend, nonzero process incarnation and last consumed session ID. |
| BEGIN / READY | Incarnation plus fresh session identity (144 bytes). READY repeats the actual capability receipt with the consumed session ID. |
| EXECUTE / RESULT | Requested logit-row count plus a bounded DS4B capture (4 MiB plus 4 bytes). DS4R result is at most 16 MiB. |
| ACK / ACKED | Exact window identity (136 bytes). |
| CLOSE / CLOSED | Empty. CLOSE refuses an unacknowledged result. |
| ERROR | At most 1 KiB of visible failure text, then the caller abandons/closes the connection. |

Contract layout scalars are hidden width, stream count, top-k/candidate strides,
vocabulary and context limit. An ordered source list carries layer, compression
ratio and KV/key row byte sizes. Array lengths come from this pinned layout,
never from peer-supplied tensor dimensions. Capability decoding rejects trailing
bytes, invalid backends, missing executable identity and unqualified arithmetic.

## Commit and failure rules

Discovery validates the pinned executable and full contract before native
preparation. BEGIN must use the discovered incarnation and a session ID larger
than the service's floor. The ID is consumed **before** calling native preparation:
a partial preparation failure cannot make it reusable. The adapter must reset
and configure the native model and verify effective settings before READY.

For each window, the service validates the capture and invokes its executor
once. An identical duplicate EXECUTE before ACK returns the retained result.
A different duplicate, a later window before ACK, a late duplicate after ACK,
or any native failure ends the session. Position/sequence checks precede native
execution. The coordinator publishes logits only after validating RESULT and
receiving the exact ACKED identity. Loss of ACKED therefore aborts the request
even if the suffix has already computed it.

The client never reconnects or retries automatically. Disconnects, wrong reply
IDs/types, changed identities, malformed or nonfinite results and deadlines close
the channel and leave its output argument unchanged. Layer-worker loss aborts
inference; the expert worker's recomputable fallback policy does not apply.

One absolute deadline covers EXECUTE, RESULT, ACK and ACKED, including partial
socket progress. Socket deadlines cannot interrupt a GPU kernel or an NFS read.
The native adapter and its supervisor must bound computation separately, and
must abandon native state after a failed service call.

## Validation and limits

On 9 November 2026, six CPU protocol/geometry tests passed on Lucy and Astor.
The new RPC test uses real loopback TCP, with synthetic native callbacks. It
checks successful 3- and 16-token windows, identity/policy refusals, preparation
fencing across connections, duplicate/conflicting delivery, stale incarnation
and session IDs, malformed framing, lost replies/ACKs and one transaction
deadline. The RPC test also passed AddressSanitizer and UndefinedBehaviorSanitizer
on Lucy. The shared network library built in both CUDA and HIP build trees.

These are delivery-contract tests, not GPU arithmetic, native TCP inference,
mixed-backend parity, quality, recovery, capacity or performance evidence.

## Native layer harness (qualification pending)

`deepseek_layer_tcp_test` is a separately built native harness, not a server/app
integration. It is opt-in and excluded from the default build. It supports a
suffix service, an assigned prefix coordinator and a complete native reference:

```sh
# Build by name in a configured CUDA or HIP build directory.
cmake --build BUILD --target deepseek_layer_tcp_test

# No weights or GPU execution: check invalid configurations on the built binary.
uv run --no-project tools/test_ds4_layer_admission.py BUILD/deepseek_layer_tcp_test

# New output directories are required; existing evidence is never overwritten.
DS4_IQ_Q8K=1 BUILD/deepseek_layer_tcp_test --experts 6 suffix PACK 20 0.0.0.0:19170 600000 3600000 1 SUFFIX_OUT
DS4_IQ_Q8K=1 BUILD/deepseek_layer_tcp_test --experts 6 prefix PACK 20 WORKER_IP:19170 WORKER_EXE_SHA256 600000 PROMPT_IDS 128 1 PREFIX_OUT
DS4_IQ_Q8K=1 BUILD/deepseek_layer_tcp_test --experts 6 reference PACK PROMPT_IDS 128 1 REFERENCE_OUT
```

`PROMPT_IDS` contains whitespace-separated token IDs from the pinned tokenizer.
By default the harness uses one sequence, context 2048, native synchronous windows,
six experts, mapped CPU experts, no expert GPU cache/RAM store, no DSpark and no
vision. Prefix and suffix load only their assigned ranges. A 16-row prompt
window returns its final logit row; decode uses one row. The coordinator's
initial greedy policy chooses the first finite maximum; the reference mode
checks this choice against the existing native sampler for each window.

The final `1` explicitly continues after EOS for long functional probes. `0`
stops at the pinned pack's EOS token 1; logs distinguish EOS from the token limit.
Prefix/reference commands can repeat the `PROMPT_IDS DECODE IGNORE_EOS OUT`
group, up to sixteen requests. The model is loaded and hashed once; each group
uses a fresh native session and a separate new output directory. Match the
suffix connection count to the planned requests. Prefix RPC timeout is explicit.
The coordinator consumes increasing session IDs, including across a worker
incarnation change within that coordinator process. All group paths, bounds and
token IDs are checked before loading.

Every committed window records input IDs, position, next token and raw logits.
The prefix also retains acknowledged DS4R records. A failed layer transaction
aborts; it never generates a continuation from unacknowledged output.

The suffix's discovery receipt is derived after native configuration. BEGIN
resets/configures the native suffix again. The callback calls `replay_boundary()`
and reads `Model::logits()`/`pos()`; the prefix callback calls the persistent TCP
client during the existing `feed()` boundary observer. Native cancellation polls
a computation deadline; a separate process supervisor is still required to
bound stalled device/filesystem calls. Socket and accept timeouts are separate.
The suffix takes an explicit connection count (1..16), resets native state after
each connection and preserves its consumed session floor between connections.
An aborted connection is logged and leaves a failed final status, even if a later
fresh session succeeds. The coordinator never retries the failed request.

### Loaded artifact identity

The native model's `boundary_artifacts()` hashes its actual read-only backing
mappings, including Engram, rather than reopening a purported model path. It
refuses an unloaded pack or any layer whose effective expert count is not six.
For this pack there are six backing files: four main shards, Engram and DSpark.
The tokenizer/config metadata source is separately pinned in the seven-file
source manifest; this token-ID harness does not run a tokenizer.

The new identity domains are explicit and have no compatibility fallback:

- **DS4M1 model identity:** SHA-256 of literal `DS4M1` plus its NUL byte, file
  count (u32), then file-index-ordered byte size (u64) and SHA-256 (32 bytes)
  of each actual mapped file. Integers are little endian. This can be checked
  independently against the previously verified full-file source manifest.
- **DS4P1 pack identity:** SHA-256 of a length-prefixed `DS4P1` string; sorted
  parsed key/value metadata, excluding only `model_dir`; indexed backing names
  and byte sizes; sorted parsed tensor names/types/file indices/offsets/sizes/
  shapes; and parsed Engram header/map/multiplier/prime/offset arrays. Strings
  have u32 byte lengths, collection counts are u32 (Engram arrays u64), shapes
  and Engram integer arrays use their native signed 32/64-bit widths, and all
  scalar bytes are little endian. File index is signed i32; offsets/sizes u64.

These differ from E1's raw four-shard digest and E2's raw pack-file digest.
Excluding the `model_dir` locator permits verified local and NFS backing paths
to share an execution identity without allowing different parsed tensor views.
Without an explicit plan ID, the harness derives a bounded layer-only test
identity. Installation of a complete resource/expert/orchestrator manifest
remains E6 work.

### Optional expert delegation

The native harness accepts these global options after `--experts 6` and before
the role:

```text
--expert-worker NUMERIC_IPV4:PORT
--expert-worker-model FIRST_SOURCE_GGUF
--expert-worker-placement PLACEMENT_FILE
--expert-worker-timeout MILLISECONDS
--plan-id NONZERO_SHA256
```

Endpoint, source GGUF and placement must be supplied together. Layer and expert
TCP currently accept numeric IPv4 endpoints; the harness refuses IPv6 and
hostnames before loading. Expert delegation also requires an explicit shared
plan ID on both layer owners. The timeout must
be positive. Duplicate options, missing files and zero identities fail before
loading. Every harness role requires `DS4_IQ_Q8K=1`, including layer-only runs;
the native model also refuses expert networking with unqualified per32 arithmetic.
The no-worker native model path keeps its existing arithmetic policy.

For this experimental harness, the trusted driver supplies the shared static
plan ID. It must pin both expert placements and all process settings in its
experiment manifest. The harness does not derive or transactionally install an
orchestrator plan. Actual loaded model/pack and peer executable identities are
still independently checked. Supplying a plan ID is not E6 qualification.

The assigned native model uses its existing synchronous CPU-miss dispatch helper
for eligible remote experts. The local replacement cache and RAM store remain
disabled in this bounded profile. Expert arithmetic, fallback during native
execution and the combined hybrid path still require live qualification.

### Optional fixed local GPU expert residency

`--expert-vram-mib MIB` requests a fixed budget for the existing native expert
allocator. It accepts 0..65536 MiB and refuses a budget that does not fit the
allocator's available room; it never silently reduces that budget. Expert units
can leave unused space within the limit. Zero residency remains the harness
default. A nonzero budget requires an explicit `--plan-id`; the trusted experiment
manifest must pin each process's budget. The native allocator places only owned layers'
experts and leaves other picks on the existing CPU/file-backed path. It does not
introduce another cache, allocator or transfer scheme.

When expert networking and verbose output are enabled, `ds4-network-tiers`
records each applicable layer/token's local GPU, local CPU and acknowledged
remote pick counts. A live E4 fixture must show all three counts positive for
one token/layer, no fallback and correct combined output. A local CPU count
does not measure cold SSD/NFS reads; OS page-cache residency is separate.
Nonzero-GPU assigned loading, hard-budget refusal, native tier accounting and
hybrid arithmetic still require live qualification. Previous E2/E3 CPU-only
receipts retain their original zero-residency profile.

### Teacher-forced logit capture

The qualification harness also accepts `--cpu-threads N` before the role, using
the existing native CPU pool limit (1..64). Its default stays two. A different
limit requires an explicit shared `--plan-id`, whose experiment manifest must
bind each role's CPU budget. The native startup description records the effective
pool size; the requested limit is not a promise that every thread will be used.
This does not change the expert worker's separate two-thread CPU profile.
CPU-budget changes need their own capture/repeatability qualification before
fresh held-out data. Previously frozen campaigns retain their original limits.

For same-prefix numerical comparisons, put `--decode-mode teacher-forced` before
the prefix/reference role. Each request's `DECODE` argument then names a token-ID
file containing the unchanged reference continuation:

```sh
DS4_IQ_Q8K=1 BUILD/deepseek_layer_tcp_test --experts 6 --decode-mode teacher-forced reference PACK PROMPT_IDS TARGET_IDS 1 REFERENCE_OUT
DS4_IQ_Q8K=1 BUILD/deepseek_layer_tcp_test --experts 6 --decode-mode teacher-forced prefix PACK 20 WORKER_IP:19170 WORKER_EXE_SHA256 600000 PROMPT_IDS TARGET_IDS 1 PREFIX_OUT
```

Both runs must receive the same frozen prompt and target IDs. The harness does
not generate or repair the target file. Each request accepts 1..1024 targets;
prompt plus targets must fit the declared 2048-token context. `IGNORE_EOS` must
be `1`, so a target EOS is consumed in the same position on both runs.
The suffix is unchanged and does not take a decode-mode option.

To reuse one loaded prefix/reference across both capture modes, opt into
`--decode-mode per-request`. Every request's `DECODE` field must then be explicit:
`teacher-forced:TARGET_IDS` or `greedy:COUNT`. Teacher-forced requests still require
`IGNORE_EOS=1`; greedy task checks can stop at natural EOS. The same model is
reset between requests. Existing global greedy/teacher-forced modes retain their
argument format. This option changes only the opt-in qualification harness;
mixed-mode native capture still requires live validation.

Before consuming each target, `targets.jsonl` records its index, current position,
raw-logit window number, target ID and this executor's predicted ID. The model
then consumes the target rather than its prediction. The last target produces
one extra saved logit row, which is not a scored target position. Native
reference mode still checks its greedy choice against the existing sampler.
Prefix rows retain the same acknowledged DS4R provenance as greedy runs.

The final `E5_TCP_TEACHER_FORCING` line reports captured positions; it makes no
quality claim. Comparing these rows, freezing held-out prompts, proving baseline
repeatability and meeting the full E5 numerical/task thresholds remain required.
CUDA/HIP build and CLI-admission checks are separate from live capture validation.

### Compare completed captures

The numerical tool checks the exact prompt and target IDs, window positions,
pre-consumption target records, complete float32 rows, finite logits and recorded
argmax IDs. It requires the final post-target row and rejects missing or extra
row files. All captured rows are checked; only fixed target positions contribute
to mean KL and top-1 agreement.

```sh
uv run --no-project --with pydantic==2.14.2 tools/compare_ds4_layer_logits.py REFERENCE_OUT CANDIDATE_OUT PROMPT_IDS TARGET_IDS --output NEW_REPORT.json
uv run --no-project --with pydantic==2.14.2 tools/test_compare_ds4_layer_logits.py
```

Both captures normally use teacher forcing. To compare an ordinary greedy
reference with a teacher-forced candidate, explicitly add
`--reference-mode greedy`. The tool then requires every frozen target to be the
reference's actual selected token, and verifies that the recorded feeds consume
those targets. It does not silently substitute a different continuation or infer
the reference mode from missing files. Keep the baseline's complete recorded
continuation and successful native exit.

Reports retain per-position KL(reference || candidate) in nats, first-maximum
IDs, aggregate metrics and hashes of inputs/ordered raw rows. Float64 centered
log probabilities avoid overflow and loss of normalization at large common
offsets; tiny signed roundoff is retained. The report makes no E5 pass claim.
The campaign still verifies model/pack/executable identities and acknowledged
delivery, baseline repeatability, the frozen five-prompt/1,000-position corpus,
the numerical thresholds and the separate eight-task gate.

### Freeze the actual greedy reference

The baseline must save its complete greedy continuation with `IGNORE_EOS=1`
for the requested fixed count. Freeze those selected IDs before running a
teacher-forced candidate:

```sh
uv run --no-project --with pydantic==2.14.2 python -m tools.freeze_ds4_reference freeze BASELINE_OUT PROMPT_IDS 200 NEW_FROZEN_CASE
uv run --no-project --with pydantic==2.14.2 python -m tools.freeze_ds4_reference verify SECOND_BASELINE_OUT NEW_FROZEN_CASE NEW_REPEAT_REPORT.json
```

The helper validates the entire prompt/continuation history, every finite raw
logit row and its argmax, including the final post-target row. It copies every
selected ID in order, including EOS, and records input/history/logit hashes.
Existing destinations are refused. Token text must use unsigned decimal IDs;
floating-point text such as `1.0` is not silently converted to a token.

The second ordinary baseline must use the same pinned binary, weights, arithmetic
and resource settings. Its exact history/raw-row comparison establishes the
repeatability floor only after the campaign verifies those runtime identities.
A failed numerical repeat writes a failed receipt and exits nonzero. Neither a
frozen file nor this helper's success establishes an E5 pass or replaces the
held-out corpus and task gates.

The scorer passed analytic divergence/direction/normalization controls and
malformed/incomplete capture tests. This synthetic evidence does not qualify
native teacher-forced execution or full-model output quality.

### Measured native scope

On 9 November 2026, the first native TCP pilot completed 128 generated steps and
130 acknowledged windows with Astor HIP `[0,20)` and Lucy CUDA `[20,40)`. Actual
loaded model/pack identities matched the independent source-manifest oracle.
The natural response was “The capital of France is Paris.” followed by EOS;
the functional loop explicitly continued after EOS to reach 128 steps.
Both native processes and the driver exited successfully. This is one-prompt
functional evidence, not broad numerical quality, performance or complete E3
acceptance.

The later HIP same-backend campaign passed at 23:29 CET: two 128-step probes
and a natural EOS request had exact tokens and 270 byte-identical saved logit
rows against the ordinary reference. Real-native duplicate/incarnation tests and
the planned fourth-request suffix loss also passed their delivery/clean-error
contracts. The driver exited 0; intentional loss-request/remote-command failures
remain recorded separately. These controls precede full mixed-host long-decode,
actual restart, E5 quality and memory/performance qualification.
