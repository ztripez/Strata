"""Apply the frozen numerical capture gate to every held-out DeepSeek case.

This verifies reference repeatability and same-prefix capture metrics. Runtime
identity, acknowledged delivery, task retention and full E5 acceptance remain
separate campaign gates. Run with uv and pydantic==2.14.2.
"""

import argparse
import hashlib
import math
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, computed_field, model_validator

from tools.compare_ds4_layer_logits import Comparison, Record, compare_case, read_tokens
from tools.ds4_quality_inputs import CaseId, QualityInputs, Thresholds
from tools.freeze_ds4_reference import Digest, Reference, Repeatability, verify


class CaseComparison(Record):
    """One declared case, both exact-reference checks and its canonical comparison."""

    id: CaseId
    prompt_tokens: int
    target_positions: int
    first_reference: Repeatability
    repeated_reference: Repeatability
    comparison: Comparison

    @model_validator(mode="after")
    def matching_counts(self):
        """Reference rows and numerical positions must describe the same whole case."""
        if self.comparison.scored_positions != self.target_positions:
            raise ValueError("comparison omits declared target positions")
        for reference in (self.first_reference, self.repeated_reference):
            if (reference.target_positions, reference.captured_windows) != (
                self.target_positions,
                self.comparison.captured_windows,
            ):
                raise ValueError("reference and candidate capture sizes differ")
        return self


class CorpusComparison(Record):
    """Output-only numerical capture evidence, derived from all canonical positions."""

    corpus_sha256: Digest
    thresholds: Thresholds
    cases: Annotated[list[CaseComparison], Field(min_length=5)]
    scope: Literal["numerical captures only; no runtime/task/full E5 acceptance"] = (
        "numerical captures only; no runtime/task/full E5 acceptance"
    )

    @model_validator(mode="after")
    def distinct_cases(self):
        """Duplicating a case cannot satisfy the five-prompt requirement."""
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate numerical case identity")
        return self

    @computed_field
    @property
    def scored_positions(self) -> int:
        """Total positions, without giving shorter prompts disproportionate weight."""
        return sum(case.comparison.scored_positions for case in self.cases)

    @computed_field
    @property
    def mean_kl_reference_candidate(self) -> float:
        """Position-weighted KL(reference || candidate), retaining signed roundoff."""
        return (
            math.fsum(
                position.kl_reference_candidate
                for case in self.cases
                for position in case.comparison.positions
            )
            / self.scored_positions
        )

    @computed_field
    @property
    def matching_top1_positions(self) -> int:
        """Count first-maximum agreements across the complete fixed corpus."""
        return sum(
            position.reference_argmax == position.candidate_argmax
            for case in self.cases
            for position in case.comparison.positions
        )

    @computed_field
    @property
    def top1_agreement(self) -> float:
        """Observed agreement fraction; the pass predicate uses exact integer counts."""
        return self.matching_top1_positions / self.scored_positions

    @computed_field
    @property
    def references_exact(self) -> bool:
        """Every first and repeated reference matches its frozen raw rows/history."""
        return all(
            reference.windows_exact and reference.logits_exact
            for case in self.cases
            for reference in (case.first_reference, case.repeated_reference)
        )

    @computed_field
    @property
    def numerical_capture_gate_passes(self) -> bool:
        """Apply the fixed corpus thresholds without asserting runtime/task gates."""
        agreement = Fraction(str(self.thresholds.top1_agreement_min))
        return (
            self.references_exact
            and len(self.cases) >= self.thresholds.minimum_prompts
            and self.scored_positions >= self.thresholds.minimum_positions
            and self.mean_kl_reference_candidate
            <= self.thresholds.mean_kl_reference_candidate_max_nat
            and self.matching_top1_positions * agreement.denominator
            >= self.scored_positions * agreement.numerator
        )


def compare(
    corpus_path: Path,
    inputs: Path,
    frozen: Path,
    reference: Path,
    repeated: Path,
    candidate: Path,
) -> CorpusComparison:
    """Validate all frozen inputs/references, then compare every candidate position."""
    raw = corpus_path.read_bytes()
    corpus = QualityInputs.model_validate_json(raw)
    results = []
    for case in corpus.logit_cases:
        original = inputs / case.id / "prompt.ids"
        if hashlib.sha256(original.read_bytes()).hexdigest() != case.prompt_ids_sha256:
            raise ValueError(
                "quality prompt hash differs from frozen corpus: " + case.id
            )
        folder = frozen / case.id
        prompt = read_tokens(folder / "prompt.ids")
        targets = read_tokens(folder / "targets.ids")
        if prompt != read_tokens(original) or (len(prompt), len(targets)) != (
            case.prompt_tokens,
            case.target_positions,
        ):
            raise ValueError(
                "frozen continuation does not cover the declared case: " + case.id
            )
        frozen_case = Reference.model_validate_json(
            (folder / "reference.json").read_bytes()
        )
        if (frozen_case.prompt_tokens, frozen_case.target_positions) != (
            case.prompt_tokens,
            case.target_positions,
        ):
            raise ValueError("reference manifest counts differ from frozen corpus")
        first = verify(reference / case.id, folder)
        repeat = verify(repeated / case.id, folder)
        if not all(
            item.windows_exact and item.logits_exact for item in (first, repeat)
        ):
            raise ValueError("ordinary reference repeatability failed: " + case.id)
        comparison = compare_case(
            reference / case.id,
            candidate / case.id,
            folder / "prompt.ids",
            folder / "targets.ids",
            reference_mode="greedy",
        )
        results.append(
            CaseComparison(
                id=case.id,
                prompt_tokens=case.prompt_tokens,
                target_positions=case.target_positions,
                first_reference=first,
                repeated_reference=repeat,
                comparison=comparison,
            )
        )
    return CorpusComparison(
        corpus_sha256=hashlib.sha256(raw).hexdigest(),
        thresholds=corpus.thresholds_from_approved_plan,
        cases=results,
    )


def main() -> None:
    """Preserve a new report and exit nonzero when the numerical capture gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "corpus",
        "inputs",
        "frozen",
        "reference",
        "repeated",
        "candidate",
        "report",
    ):
        parser.add_argument(name, type=Path)
    arguments = parser.parse_args()
    if arguments.report.exists():
        raise FileExistsError(arguments.report)
    report = compare(
        arguments.corpus,
        arguments.inputs,
        arguments.frozen,
        arguments.reference,
        arguments.repeated,
        arguments.candidate,
    )
    with arguments.report.open("x") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    print(
        f"{report.scored_positions} positions; mean KL={report.mean_kl_reference_candidate:.9g}; top1={report.top1_agreement:.6f}; "
        + report.scope
    )
    if not report.numerical_capture_gate_passes:
        raise SystemExit("numerical capture gate failed; report retained")


if __name__ == "__main__":
    main()
