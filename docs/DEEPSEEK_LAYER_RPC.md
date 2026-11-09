# DeepSeek layer RPC

Experimental Linux TCP delivery contract, behind `STRATA_NETWORK_EXPERTS`.
This implements delivery around the [native layer state contract](DEEPSEEK_LAYER_STATE.md).
It does not yet connect the native executors or qualify E3 inference.

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
