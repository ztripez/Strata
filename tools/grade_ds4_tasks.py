"""Grade complete native greedy captures using the frozen exact-answer contract.

Run with uv, pydantic==2.14.2 and regex==2026.9.10. This is capture/task analysis;
runtime identity, delivery, repeatability and numerical gates remain separate.
"""

import argparse
import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from serve.tokenizer_assets import TokenizerPins, load_deepseek
from tools.compare_ds4_layer_logits import (
    Record,
    Token,
    Window,
    read_records,
    read_tokens,
)
from tools.ds4_quality_inputs import CaseId, QualityInputs, TaskCase
from tools.freeze_ds4_reference import Digest, capture_hash
from tools.strata_tokenizer import Tokenizer


class TaskResult(Record):
    """A decoded task result tied to its exact prompt and raw capture rows."""

    task: CaseId
    expected: str
    decoded: str
    matches_expected: bool
    emitted_token_ids: list[Token]
    stopped_at_eos: bool
    captured_windows: int
    prompt_sha256: Digest
    ordered_logits_sha256: Digest

    @model_validator(mode="after")
    def exact_answer(self):
        """A serialized pass cannot claim a different normalization rule."""
        if self.matches_expected != (self.decoded.strip() == self.expected):
            raise ValueError(
                "task match differs from exact whitespace-only answer rule"
            )
        return self


class TaskReport(Record):
    """Eight task observations; this report alone never qualifies a runtime."""

    corpus_sha256: Digest
    output_root: str
    decode_limit: Annotated[int, Field(ge=1, le=1024)]
    tokenizer_assets_sha256: dict[str, Digest]
    results: Annotated[list[TaskResult], Field(min_length=8, max_length=8)]
    scope: Literal["exact-answer captures only; no full E5 acceptance"] = (
        "exact-answer captures only; no full E5 acceptance"
    )

    @model_validator(mode="after")
    def unique_tasks(self):
        """One task cannot stand in for another in the eight-task gate."""
        if len({result.task for result in self.results}) != 8:
            raise ValueError("task report repeats or omits task identities")
        return self


class TaskRetention(Record):
    """Baseline pass identities retained or lost under an identical task contract."""

    corpus_sha256: Digest
    baseline_passes: list[CaseId]
    candidate_passes: list[CaseId]
    lost_baseline_passes: list[CaseId]
    scope: Literal["task retention only; no full E5 acceptance"] = (
        "task retention only; no full E5 acceptance"
    )


def retain(
    corpus_path: Path, baseline: TaskReport, candidate: TaskReport
) -> TaskRetention:
    """Compare passes only after binding both reports to the frozen task definitions."""
    raw = corpus_path.read_bytes()
    corpus = QualityInputs.model_validate_json(raw)
    digest = hashlib.sha256(raw).hexdigest()
    cases = {case.id: case for case in corpus.task_cases}
    if baseline.decode_limit != candidate.decode_limit:
        raise ValueError("baseline/candidate task decode budgets differ")
    for report in (baseline, candidate):
        if (
            report.corpus_sha256 != digest
            or report.tokenizer_assets_sha256 != corpus.tokenizer_assets_sha256
        ):
            raise ValueError("task report corpus/tokenizer identity differs")
        if {result.task for result in report.results} != set(cases):
            raise ValueError("task report does not cover the frozen task identities")
        for result in report.results:
            case = cases[result.task]
            if (result.expected, result.prompt_sha256) != (
                case.expected,
                case.prompt_ids_sha256,
            ):
                raise ValueError("task report answer/prompt differs from frozen task")
    before = sorted(
        result.task for result in baseline.results if result.matches_expected
    )
    after = sorted(
        result.task for result in candidate.results if result.matches_expected
    )
    return TaskRetention(
        corpus_sha256=digest,
        baseline_passes=before,
        candidate_passes=after,
        lost_baseline_passes=sorted(set(before) - set(after)),
    )


def load_tokenizer(assets: Path, corpus: QualityInputs) -> Tokenizer:
    """Load the authoritative tokenizer after matching every pinned asset hash."""
    return load_deepseek(
        assets,
        TokenizerPins(
            assets_sha256=corpus.tokenizer_assets_sha256,
            source_sha256=corpus.source_code_sha256["tools/strata_tokenizer.py"],
        ),
    )


def greedy_capture(
    root: Path, prompt: list[int], limit: int
) -> tuple[list[int], bool, list[Window], str]:
    """Recover emitted greedy IDs and validate every feed, finite row and argmax."""
    if (
        not prompt
        or len(prompt) > 1900
        or not 1 <= limit <= 1024
        or len(prompt) + limit > 2048
    ):
        raise ValueError("task prompt/decode exceeds native bounds")
    if (root / "targets.jsonl").exists():
        raise ValueError("task capture must be free-running greedy, not teacher-forced")
    windows = read_records(root / "windows.jsonl", Window)
    expected_prefill = [
        prompt[offset : offset + 16] for offset in range(0, len(prompt), 16)
    ]
    prefill = len(expected_prefill)
    if (
        len(windows) < prefill
        or [row.tokens for row in windows[:prefill]] != expected_prefill
    ):
        raise ValueError("task capture does not consume the frozen prompt")
    position = 0
    for index, row in enumerate(windows, 1):
        position += len(row.tokens)
        if (row.window, row.position) != (index, position):
            raise ValueError("task window position or sequence differs")
        if index > prefill and row.tokens != [windows[index - 2].next]:
            raise ValueError("task capture is not a connected greedy continuation")
    if len(windows) - prefill < limit and windows[-1].next == 1:
        emitted = [row.next for row in windows[prefill - 1 :]]
        stopped = True
        if 1 in emitted[:-1]:
            raise ValueError("task continued after an earlier natural EOS")
    elif len(windows) == prefill + limit:
        emitted = [row.next for row in windows[prefill - 1 : -1]]
        stopped = False
        if 1 in emitted:
            raise ValueError("task ignored EOS instead of stopping")
    else:
        raise ValueError("task capture lacks a natural EOS or complete decode limit")
    if {path.name for path in root.glob("window-*.f32")} != {
        f"window-{row.window}.f32" for row in windows
    }:
        raise ValueError("task raw-row inventory differs from recorded windows")
    return emitted, stopped, windows, capture_hash(root, windows)


def grade_task(
    capture: Path, prompt_path: Path, case: TaskCase, tokenizer: Tokenizer, limit: int
) -> TaskResult:
    """Decode the exact emitted tokens; trim only outer whitespace for comparison."""
    prompt_raw = prompt_path.read_bytes()
    if hashlib.sha256(prompt_raw).hexdigest() != case.prompt_ids_sha256:
        raise ValueError("task prompt differs from frozen input")
    prompt = read_tokens(prompt_path)
    if len(prompt) != case.prompt_tokens:
        raise ValueError("task prompt count differs from frozen input")
    emitted, stopped, windows, digest = greedy_capture(capture, prompt, limit)
    # EOS is the native stop marker, not answer text. Other special tokens,
    # fences, prefixes, capitalization and formatting remain untouched.
    visible = emitted[:-1] if stopped else emitted
    text = tokenizer.decode(visible, errors="strict")
    return TaskResult(
        task=case.id,
        expected=case.expected,
        decoded=text,
        matches_expected=text.strip() == case.expected,
        emitted_token_ids=emitted,
        stopped_at_eos=stopped,
        captured_windows=len(windows),
        prompt_sha256=case.prompt_ids_sha256,
        ordered_logits_sha256=digest,
    )


def grade(
    corpus_path: Path, inputs: Path, outputs: Path, assets: Path, limit: int
) -> TaskReport:
    """Grade all eight frozen tasks without changing baseline pass criteria."""
    corpus_raw = corpus_path.read_bytes()
    corpus = QualityInputs.model_validate_json(corpus_raw)
    tokenizer = load_tokenizer(assets, corpus)
    results = [
        grade_task(
            outputs / ("task-" + case.id),
            inputs / ("task-" + case.id) / "prompt.ids",
            case,
            tokenizer,
            limit,
        )
        for case in corpus.task_cases
    ]
    return TaskReport(
        corpus_sha256=hashlib.sha256(corpus_raw).hexdigest(),
        output_root=str(outputs.resolve()),
        decode_limit=limit,
        tokenizer_assets_sha256=corpus.tokenizer_assets_sha256,
        results=results,
    )


def main() -> None:
    """Save a new task report; failed answers remain visible as failed answers."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    collect = commands.add_parser("grade")
    collect.add_argument("corpus", type=Path)
    collect.add_argument("inputs", type=Path)
    collect.add_argument("outputs", type=Path)
    collect.add_argument("tokenizer", type=Path)
    collect.add_argument("report", type=Path)
    collect.add_argument("--decode-limit", type=int, default=128)
    compare = commands.add_parser("retain")
    compare.add_argument("corpus", type=Path)
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("report", type=Path)
    arguments = parser.parse_args()
    if arguments.report.exists():
        raise FileExistsError(arguments.report)
    if arguments.action == "grade":
        report = grade(
            arguments.corpus,
            arguments.inputs,
            arguments.outputs,
            arguments.tokenizer,
            arguments.decode_limit,
        )
    else:
        report = retain(
            arguments.corpus,
            TaskReport.model_validate_json(arguments.baseline.read_bytes()),
            TaskReport.model_validate_json(arguments.candidate.read_bytes()),
        )
    with arguments.report.open("x") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    if arguments.action == "grade":
        print(
            str(sum(result.matches_expected for result in report.results))
            + "/8 exact answers; "
            + report.scope
        )
    else:
        print(
            str(len(report.baseline_passes))
            + " baseline task passes, "
            + str(len(report.lost_baseline_passes))
            + " lost; "
            + report.scope
        )
        if report.lost_baseline_passes:
            raise SystemExit(
                "baseline task pass lost; failed retention receipt preserved"
            )


if __name__ == "__main__":
    main()
