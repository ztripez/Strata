# Experimental DeepSeek quality checks

The [network plan's E5 gate](NETWORK_SHARDED_DEEPSEEK_PLAN.md#e5--numerical-and-task-quality-acceptance)
requires a repeatable native reference, same-prefix numerical comparisons and
eight frozen exact-answer tasks. A successful worker call or capture is not this
quality gate. These source-checkout analysis tools do not change inference.

The corpus input file records the prompt hashes, target counts, exact expected
answers, normalization, tokenizer assets and preparation sources. It also pins
the approved numerical thresholds. `tools/ds4_quality_inputs.py` validates that
contract. A separate runtime campaign must seal its hash and verify the actual
model, pack, binary, six-expert arithmetic, resources and delivery receipts.

## Complete numerical corpus

Freeze each first reference's selected continuation and verify the second ordinary
reference before collecting candidate data, using
[the reference helper](DEEPSEEK_LAYER_RPC.md#freeze-the-actual-greedy-reference).
Then score each candidate configuration separately:

```sh
uv run --no-project --with pydantic==2.14.2 python -m tools.compare_ds4_corpus CORPUS.json INPUT_ROOT FROZEN_ROOT FIRST_REFERENCE_OUT SECOND_REFERENCE_OUT CANDIDATE_OUT NEW_NUMERICAL_REPORT.json
```

Folders under each root use the frozen logit case ID. The corpus checker rechecks
every original prompt hash, frozen target count and both references' exact raw
rows/history. Candidate comparisons use the unchanged baseline targets. All
recorded rows, including prefill and the final extra row, must be finite and have
the recorded argmax. Only the declared target positions contribute to KL and
top-1 acceptance.

Means are weighted by positions, not by prompts. The top-1 predicate compares
integer counts against the exact decimal threshold. Duplicate prompts cannot
satisfy the corpus size requirement. A failed numerical threshold writes a failed
report and exits nonzero; a broken input/reference stops before candidate scoring.
The report is numerical capture evidence only. Run it for both layer-only and
hybrid outputs, and retain the separate task and runtime gates.

## Free-running divergence

Run separate fixed-length greedy diagnostics with native ignore-EOS enabled.
Compare them against the unchanged first reference and its frozen continuation:

```sh
uv run --no-project --with pydantic==2.14.2 python -m tools.compare_ds4_generation FIRST_REFERENCE_CASE FROZEN_CASE CANDIDATE_GREEDY_CASE NEW_GENERATION_REPORT.json
```

The report retains both token sequences, every selected EOS, the first differing
token's zero-based index and the shared-prefix length. Positional matches after
divergence have different prefixes; they are not same-prefix top-1 agreement.
Teacher-forced captures are refused, even when their feeds happen to match greedy
choices. Both greedy captures must include the complete prompt, every selected
token and the final extra row. All raw rows must be finite and match their
recorded argmax. The reference must still match its frozen raw bytes.

A valid divergent run writes a diagnostic and exits zero. This command has no
quality pass predicate and cannot replace the numerical corpus or task gates.
Invalid captures fail without publishing a report. Existing reports are refused.

## Exact-answer tasks

```sh
uv run --no-project --with pydantic==2.14.2 --with regex==2026.9.10 python -m tools.grade_ds4_tasks grade CORPUS.json INPUT_ROOT BASELINE_OUT TOKENIZER NEW_BASELINE_TASKS.json
uv run --no-project --with pydantic==2.14.2 --with regex==2026.9.10 python -m tools.grade_ds4_tasks grade CORPUS.json INPUT_ROOT CANDIDATE_OUT TOKENIZER NEW_CANDIDATE_TASKS.json
uv run --no-project --with pydantic==2.14.2 --with regex==2026.9.10 python -m tools.grade_ds4_tasks retain CORPUS.json BASELINE_TASKS.json CANDIDATE_TASKS.json NEW_RETENTION.json
```

Input and output task folders are named `task-ID`. The input folder contains
`prompt.ids`; the output folder contains the native window history and raw rows.
The default decode limit is 128 and must match the native command. Both reports
must use the same decode limit, corpus and tokenizer identities. Existing report
files are refused.

The grader validates every prompt/feed, raw-row inventory, finite logit and
argmax. Teacher-forced task captures are refused. The authoritative Strata
tokenizer decodes the emitted IDs after checking every pinned asset and its own
source hash. The native EOS stop marker is recorded and excluded from answer
text. Other special tokens, fences, prefixes, case and number formatting remain
untouched. Only outer whitespace is trimmed for comparison.

A token predicted by the final extra row after a decode limit was not emitted.
The report distinguishes that case from natural EOS and retains the decoded
text and emitted IDs. Invalid UTF-8 is not repaired. Wrong answers remain visible
as failed answers in a task report; they are not capture-protocol failures.

Retention refuses changed task definitions or budgets. A lost baseline pass
writes a failed receipt and exits nonzero. The report lists the actual baseline
pass identities, including an empty set; it never claims all eight tasks passed
merely because no baseline pass was lost. Numerical acceptance and runtime
qualification remain separate.

## Larger CPU-budget control

The sealed fixture in `/var/tmp/strata-deepseek-e5-cpu8-control-20261110/`
requires exact short-capture agreement with the earlier two-thread control,
using the existing native CPU pool with an eight-thread limit. A separate
excluded marker prompt has 1,848 input tokens and 200 fixed targets, reaching
position 2,048. Those targets repeat the old E3 continuation; they are not a
new reference answer. Natural EOS before and after both captures must remain
exact after reset. The held-out corpus is separate.

Startup and each request are bounded to 30 minutes; the whole control has a
three-hour bound. It records raw process CPU/fault/I/O counters and private NFS
mount totals alongside memory samples. These diagnostics are not calibrated
per-expert costs or a speed result. The fixture follows the mixed-host loss
control to avoid overlapping native model campaigns. See the
[current gate table](DISTRIBUTED_FABRIC_STATUS.md#current-gate-status) for run status.

## Analysis-tool evidence

Five synthetic capture/retention tests passed. They cover native EOS versus the
unconsumed final row, strict answer normalization, frozen prompt/budget identity,
teacher-forcing refusal and damaged raw rows. The pinned tokenizer also decoded
the existing ordinary HIP E3 Paris EOS capture correctly. This is a decoder
control on an excluded fixture. No fresh held-out task run or E5 pass is claimed.
Receipts are in `/var/tmp/strata-deepseek-task-grader-20261110/`.

Three analytical corpus tests passed. They cover the exact KL/top-1 boundaries,
unequal prompt lengths, insufficient/duplicate cases and failed references.
Receipts are in `/var/tmp/strata-deepseek-corpus-scorer-20261110/`. These unit checks
do not establish fresh numerical acceptance; use the current gate table for
runtime status.

Four free-running fixture tests passed, including valid divergence, preserved
EOS, teacher-forced refusal, changed baseline and damaged prefill/final rows.
On the existing excluded E3 engineering captures, the same-backend TCP control
matched all 128 baseline tokens. The mixed-host capture first differed at index
17 and matched 18 of 128 token positions overall. This reports the earlier
functional run; it is not fresh E5 qualification. Receipts are in
`/var/tmp/strata-deepseek-generation-diagnostic-20261110/`.
