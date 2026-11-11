# DeepSeek: actual 2K capacity and storage measurements

Completed 10 November 2026 at 23:40 CET. All four native requests consumed
1,900 prompt tokens and generated 148 tokens, reaching actual position 2,048.
The driver and all 16 native processes exited 0. Both remote expert directions
executed, with zero fallbacks in every configuration. No monitored memory
ceiling was exceeded, and telemetry recorded no gaps.

E7 remains PARTIAL. Actual 2K execution within the stated budgets is demonstrated.
The best decode result is below the unchanged 5-token/s usable-capacity target.
No larger or higher-fidelity model was run. The
[experimental E5 checkpoint](DEEPSEEK_EXPERIMENTAL_CHECKPOINT.md) remains frozen:
strict E5 FAIL, with remote GPU expert quality unqualified. Cut24 is the frozen
numerical checkpoint; cut20 remains a performance candidate with the previously
recorded bounded quality evidence.

## Completed 2K results

| Backing / split | Prefill (tokens/s) | Decode (tokens/s) | NFS server reads (GiB) |
| --- | --- | --- | --- |
| NFS / cut20 | 2.816 | 1.332 | 181.56 |
| NFS / cut24 | 3.573 | 1.398 | 139.50 |
| Staged backbone / cut20 | 5.032 | 2.338 | 3.16 |
| Staged backbone / cut24 | 5.225 | 1.416 | 3.91 |

Cut20's observed staged/NFS ratios were 1.79 for prefill and 1.75 for decode,
with 98.26% fewer NFS server-read bytes. Cut24's ratios were 1.46 and 1.013,
with 97.20% fewer NFS bytes. Cut24 therefore gained prefill speed with little
change in decode. Removing NFS reads alone did not determine the better split.

For each cut, all 148 emitted token IDs matched between its NFS and staged
runs, and both layer owners performed the same number of CPU row jobs. This is
bounded storage-path evidence, not raw-logit parity or a new E5 qualification.

Each condition ran once, in order: NFS cut20, NFS cut24, file copies, staged
cut20, staged cut24. Cache states were inherited; caches were not flushed.
These are observed ratios, not replicated controlled speedup claims. The
native helper feeds prompts in 16-token windows and includes a final forward
after the last emitted token plus session close in its decode timer. These are
native helper measurements, not production HTTP throughput.

NFS counters cover startup and the request. They measure client NFS server-read
bytes, not physical NAS SSD reads. The existing private NFS path uses SSH
forwarding, so its costs also include that path. Copying occurred after both
NFS requests and before either staged request; copy traffic is outside those
request measurements. Raw interface counters include unrelated traffic.

## Memory fit

The monitor aborts on sampled device VRAM above 14 GiB on Astor or 9 GiB on
Lucy, or an individual process RSS high-water mark above 48 GiB.

| Backing / split | Device VRAM peak, Astor / Lucy (GiB) | Largest individual process RSS HWM, Astor / Lucy (GiB) | Sum of process PSS peaks, Astor / Lucy (GiB) |
| --- | --- | --- | --- |
| NFS / cut20 | 12.63 / 7.69 | 30.51 / 26.94 | 31.12 / 27.62 |
| NFS / cut24 | 13.21 / 7.07 | 31.40 / 28.11 | 31.98 / 28.84 |
| Staged / cut20 | 12.66 / 7.72 | 29.76 / 29.05 | 30.32 / 29.77 |
| Staged / cut24 | 13.24 / 7.06 | 29.82 / 28.58 | 30.42 / 29.37 |

Device values include unrelated allocations. PSS apportions shared resident
pages. Individual process peaks can occur at different times; their sum is an
upper bound, not a simultaneous physical peak. RSS, PSS and file-cache bytes
must not be added as independent memory categories. Raw one-second samples
retain host available RAM, RSS/PSS, locked memory and process I/O. Mapped-file
PSS is sampled every five seconds. File-cache snapshots are endpoints, not peaks.

The final tracked active/local and original NFS views contained about 55.29 GiB
of distinct model backing: 27.23 GiB on Astor and 28.07 GiB on Lucy. The recorded
views had no overlapping backing pages across nodes or duplicated local/NFS
pages within a node. This does not mean all caches on the machines were surveyed:
other historical mount views are outside the census. Active and original-NFS
snapshots were about 100 seconds apart, with inference stopped, rather than a
simultaneous peak. Exact times and page vectors are preserved in
`post-run-cache-accounting.json` and `cache/`.

## CPU execution versus waits

Linux profiling uses `STRATA_DS4_CPU_PROFILE=1`. It records per-thread user and
system CPU time, elapsed time while a pool participant has row jobs, faults and
context switches, then prints a summary when the pool is destroyed. Idle sleep
and the caller's `wait()` spin are excluded. These are the layer owners' local
CPU expert-row pools; separate remote expert services retain their existing RPC
and tier counters.

| Backing / split | Off-CPU share of active row-worker spans, Astor / Lucy | Major faults in row workers, Astor / Lucy |
| --- | --- | --- |
| NFS / cut20 | 67.0% / 83.6% | 249,633 / 663,699 |
| NFS / cut24 | 68.8% / 76.4% | 343,556 / 302,509 |
| Staged / cut20 | 52.5% / 40.4% | 5,838 / 793,733 |
| Staged / cut24 | 53.6% / 29.8% | 6,909 / 375,711 |

Off-CPU includes blocking and descheduling; it is not exact NFS-wait time or
page-fault latency. On-CPU time also includes row-loop bookkeeping and memory
stalls, not just arithmetic instructions. Worker spans overlap across threads
and are not request wall time. NFS READ queue/round-trip/execution durations
are cumulative across concurrent operations and must not be added to request
time either.

For cut20, Astor's measured user CPU time changed from 637.57 to 630.94 seconds;
its off-CPU worker time fell from 1,336.87 to 709.13 seconds. Lucy's user CPU time
changed from 519.13 to 473.38 seconds; off-CPU time fell from 2,881.64 to 348.59
seconds. Row job counts and generated tokens matched within each storage pair.
These observations, together with page-wait samples and the fall in NFS reads,
identify weight-access waits as a substantial cost. They do not isolate every
stall to NFS rather than scheduling or other causes.

Fault counts alone are insufficient: Lucy's cut20 major-fault count increased
on local backing while its off-CPU time fell sharply. The staged Astor samples
also contain `folio_wait_bit_common`, `blk_mq_get_tag` and `ext4_read_bh` waits.
Local backing reduced network reads but did not eliminate storage waits.

## Decode critical path

The following values sum the 148 forward windows. Suffix native time is already
inside layer RPC wall time; it must not be added to that wall time again.
Remaining RPC work includes encoding, transfer, validation and acknowledgements,
not pure network latency. It excludes separate expert RPC costs inside owners.

| Backing / split | Astor outside layer RPC (s) | Lucy suffix native (s) | Other layer RPC (s) |
| --- | --- | --- | --- |
| NFS / cut20 | 51.65 | 59.16 | 0.28 |
| NFS / cut24 | 76.16 | 29.39 | 0.27 |
| Staged / cut20 | 46.19 | 16.82 | 0.26 |
| Staged / cut24 | 90.25 | 13.98 | 0.27 |

For staged cut20, about 73% of forward-window time was outside the layer RPC on
Astor; for staged cut24 it was about 86%. Cut24's faster suffix was offset by
more elapsed time on Astor's side. That side includes native computation,
weight access, expert RPCs and other owner work; this measurement does not assign
all of it to one cause. Layer transport replacement is not supported as the next
priority by these measurements.

## Local disk placement and cost

Astor uses a Kingston A400 SATA SSD; Lucy uses NVMe. Local copies are at
`/mnt/work/strata-models/deepseek41-e7-backbone-20261110/` on each host.

| Host | Local backbone shards | Copied bytes (decimal GB) | Remaining free after copy (decimal GB) |
| --- | --- | --- | --- |
| Astor | 1 and 2 | 78.46 | 7.63 |
| Lucy | All four | 138.24 | 101.64 |

The two inference hosts use 216.70 GB of local disk copies, including 78.46 GB
of duplicated backbone data between their SSDs. This coarse whole-file staging
is a storage experiment, not exclusive assigned-expert storage or a final
placement policy. Disk duplication is separate from the largely disjoint
resident RAM working sets.

The 110 GB Engram file stays on NFS. Astor's uncopied backbone shards also remain
NFS symlinks, including some layer24 remote-expert data. Lucy's local copies
cover all backbone expert data but not Engram. The source GGUF files remain the
shared immutable backing. The experiment changes only the pack's `model_dir`;
the existing pack identity explicitly excludes that location. No model format,
quantizer, transport or cache policy was added or changed. Previous verification
was trusted; local copies were not subjected to a new cryptographic hash scan.

## Model and execution settings

The target is the same DeepSeek V4.1-Flash 2.00-bpw model on Astor (RX 6800,
Ryzen 3900X, 64 GiB RAM) and Lucy (RTX 3080 10 GB, Ryzen 5900X, 64 GiB RAM),
using the existing 10 Gb network. Every process uses six routed experts per
token. Each layer owner uses eight CPU threads.

Cut20 assigns layers 0–19 to Astor and 20–39 to Lucy; native expert VRAM budgets
are 209/172 MiB. Cut24 assigns 0–23 to Astor and 24–39 to Lucy, with 242/139 MiB.
Native GPU expert identities match the original solo placement. Lucy serves
experts 128–255 in layer0 to Astor; Astor serves that set in the first suffix
layer to Lucy. Remote workers use 0 MiB expert VRAM and a 20 MiB RAM arena, with
other assigned experts accessed through mappings. Dense layers execute on GPUs.

The 1,900-token input is the frozen archive-ledger prompt plus 52 tokens from
its prior solo continuation. There is no new numerical campaign. Keep cut24 as
the accepted numerical checkpoint and staged cut20 as the observed faster decode
candidate. No default placement was changed. The draft expert-only optimizer
uses assumed costs and has no accepted DeepSeek whole-layer prediction accuracy.

## Gate state

- Actual 2K full-model execution and monitored budget fit: demonstrated for all
  four requests, under the existing experimental quality scope.
- Storage-path improvement: observed for cut20 decode and both prefills;
  cut24 decode changed little. Reproducible general speedup remains open.
- The 5-token/s decode target: not met; prefill above five does not satisfy it.
- C2 larger/higher-fidelity capacity: not run.
- Strict E5: remains FAIL. No thresholds or numerical acceptance were changed.
- Remote GPU expert quality and measured optimizer ranking: remain unqualified.

## Evidence and execution correction

Working evidence: `/var/tmp/strata-deepseek-e7-2k-storage-20261110/`.
Persistent evidence: `/mnt/storage/strata-evidence/deepseek-e7-2k-storage-20261110/`.
The archive preserves logs, binaries, declared/actual commands, inputs,
one-second telemetry, page vectors, staging receipts, both build logs and the
scoped CPU profiling patch. Weights themselves stay at their NAS/local paths.

Read `comparison.json`, `summary.json`, `post-run-cache-accounting.json` and the
four `*-measurements.json` files for measurements and their scopes.
`stage-astor.json` and `stage-lucy.json` record file copies and remaining links.
All 16 native exit receipts are included in `comparison.json`.

The first suffix retained the old two-request connection count. Its inference
completed, but shutdown awaited an unused second connection. The existing DS4L
client closed that slot without BEGIN or EXECUTE; the suffix then exited 0.
Future launches used one connection. The original running controller source,
first-request monitor source, declared/actual commands and correction receipts
are preserved. No completed inference was repeated. Request timing excludes the
unused connection wait; lifetime resource sampling includes that interval.

## Historical: earlier short workload

The earlier 256-input/128-output first/repeat slice is preserved separately at
`/mnt/storage/strata-evidence/deepseek-e7-20261110/`. Its repeat decode values were
0.621 tokens/s solo Astor, 2.030 cut20 and 1.469 cut24. Its 3.27 ratio was an
observation with inherited caches and different greedy continuations. Do not
mix it with the actual 2K results above. The frozen E5 archive remains separate
and unchanged.
