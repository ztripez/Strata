# DeepSeek distributed inference: experimental checkpoint

Experimentally accepted, 10 November 2026, after full-corpus confirmation.
This is not a strict E5 pass. The checkpoint is frozen at
`/mnt/storage/strata-evidence/deepseek-e5-experimental-20261110/`.
`evidence.tar` preserves complete captures, failed trials, native binaries,
commands, inputs, tokenizer, analysis code and the scoped engine patch. The
acceptance record and original strict numerical reports are also directly
readable beside the archive. Model weights remain at their original NFS paths.

## Completed result

Both layer-only and CPU-backed hybrid execution completed all five frozen
prompts and 1,000 teacher-forced positions, including the 2K context case.
Both had 989 matching top-1 predictions (98.9% agreement), 11 disagreements,
and mean KL 0.0015021134. All four baseline-passing tasks were retained out
of eight tasks tested. The complete ordered raw-logit digests match between
layer-only and hybrid for every prompt.

Per-prompt agreement was 98.5%, 100%, 99%, 98.5% and 98.5%. The remaining
results therefore confirm comparable behavior under the user's experimental
decision, without a materially worse case or a lost baseline task pass.
The original strict E5 numerical predicate remains false: 989/1,000 is below
the unchanged 99% threshold. Remote GPU expert quality remains unqualified.

`experimental-corpus-complete.json` contains the measurements;
`experimental-acceptance.json` records the explicit experimental decision.

## Acceptance decision

For the current experimental milestone, the user accepted the observed 98.5%
next-token agreement subject to the remaining frozen corpus showing comparable
behavior and retaining every baseline task pass. That review is now complete.
This decision did not automatically waive materially worse results.

The original E5 thresholds remain unchanged: at least 1,000 teacher-forced
positions across five prompts through 2K, finite logits, mean KL at most 0.01
nat, at least 99% top-1 agreement, and retention of every baseline task pass.
Keep strict comparator results and failed runs. Do not label this milestone
`strict E5 PASS`.

The final experimental acceptance record must state the measured agreement,
disagreement count, KL, per-prompt results, task retention, execution settings,
and any incomplete scope. It must distinguish experimental acceptance from
the original strict criteria.

## Evidence and current scope

The cut20 CPU-worker configuration completed 800 positions across four prompts.
It had 12 disagreements: 98.5% top-1 agreement and mean KL 0.00262657. CPU-backed
remote MoE raw logit rows matched layer-only exactly across those four captures.
All four baseline-passing tasks were retained; the solo baseline passed four
of the eight tasks.

The native trace found exact state entering CUDA layer20. The first observed
difference was attention normalization, at maximum absolute error 7.45e-9,
followed by growth through the layer. Further numerical refinement is deferred.
Do not start another numerical investigation without new evidence of a
functional regression.

The accepted configuration uses the already-supported cut24: Astor owns layers
`[0,24)` and Lucy owns `[24,40)`. Native expert VRAM budgets are 242 MiB on Astor
and 139 MiB on Lucy, preserving the original solo GPU expert identities.
Remote workers use 0 MiB of expert VRAM and 20 MiB of RAM, with remaining
assigned experts read from the model mappings. Lucy serves assigned layer0
experts to Astor; Astor serves assigned layer24 experts to Lucy. Dense layers
still execute on both GPUs. Remote GPU expert quality remains unqualified.

Evidence is retained under
`/var/tmp/strata-deepseek-e5-vertical-20261110/`. In particular:

- `experimental-decision-20261110.txt`: the user decision.
- `ordered-cpu-partial-summary.json`: the completed cut20 measurements.
- `diagnostic-dense-trace/report.json`: the observed layer20 differences.
- `continue-cut24.py`: the original strict runner, left running unchanged.
- `finish-experimental-corpus.py`: completes missing cases after that runner
  finishes, without the strict top-1 early stop. Runtime failures and lost task
  passes remain blockers.
- `experimental-corpus-complete.json`: the full measurements for review, written
  only after the remaining captures and original comparisons complete. Its
  existence does not itself assert experimental acceptance.

Previously verified model identities are trusted in these runs; full model
files are not rehashed at each process start. The original verification and
failed numerical results are preserved.

The original cut24 controller received SIGTERM during the final reference
repeat. That native process continued on Astor. `adopt-archive-repeat.py` waits
for its actual exit, collects and verifies the existing capture, then releases
the continuation. Its recovery receipt preserves controller exit 143; it does
not invent a successful controller exit. Transient user services
`strata-e5-adopt-20261110`, `strata-e5-complete-20261110` and
`strata-e5-archive-20261110` own recovery, remaining execution and evidence
preservation independently of the command session. No completed inference case
is restarted by this recovery.

## Next milestone

The first E7 slice and actual-2K storage comparison are complete:
[capacity and performance measurements](DEEPSEEK_CAPACITY_PERFORMANCE.md).
All four 2K requests completed within monitored memory ceilings. Staged cut20
reached 2.338 decode tokens/s versus 1.332 on NFS, with the same generated tokens.
This does not complete full E7, meet its 5-token/s decode target or demonstrate
a larger/higher-fidelity model. The frozen numerical checkpoint is unchanged.

Full measurements confirmed comparable behavior and task retention, and evidence
preservation is complete. Advance to E7 capacity and performance work in these
three areas:

1. Capacity: measure physical RAM, GPU VRAM, mapped file residency and storage
   traffic on each node. Separate distributed model capacity from duplicated
   weights, shared pages and staging buffers.
2. Performance: measure full-model prefill and decode throughput. Use the existing
   native window/transaction timings and expert-service counters to identify
   dense execution, expert staging and network synchronization costs. A layer
   RPC already includes suffix execution; do not add both as independent costs.
3. Placement: compare the solo Astor baseline, cut20 and cut24 under the same
   model and workload. Then compare optimizer predictions with the observed
   winner, keeping assumed costs distinct from measured inputs.

The runtime already exposes request prefill/decode times and `--trace-windows`;
see [native layer execution](DEEPSEEK_LAYER_PROCESS.md) and the existing
`tools/network_fabric_calibration.py` observation reader. The draft
`tools/fabric/orchestrator.py` contains hardcoded cost assumptions. Its output
does not establish measured prediction accuracy or accepted whole-layer plan
ranking.

Prioritize a working distributed endpoint, resource allocation, larger-model or
higher-fidelity capacity, and measured inference speed. Keep the existing local
tiers and transport. Numerical refinement and unrelated orchestration hardening
remain deferred. Experimental resource and performance results must cite this
acceptance decision; they do not become strict E5 or release acceptance claims.
