"""Report fixed-length native greedy divergence without asserting E5 quality.

The baseline continuation remains frozen. Both captures must be complete and
finite, including prefill and the final extra row. EOS is retained because this
diagnostic requires native ignore-EOS execution. Run with uv and pydantic==2.14.2.
"""

import argparse
import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, computed_field, model_validator

from tools.compare_ds4_layer_logits import (
    Record,
    Token,
    Window,
    read_records,
    read_tokens,
    trajectory,
)
from tools.freeze_ds4_reference import Digest, Reference, capture_hash, verify


class GenerationComparison(Record):
    """Output-only token diagnostics; post-divergence positions have different prefixes."""

    scope: Literal["free-running diagnostic only; no numerical/task/E5 acceptance"] = (
        "free-running diagnostic only; no numerical/task/E5 acceptance"
    )
    reference_manifest_sha256: Digest
    candidate_capture: str
    candidate_ordered_logits_sha256: Digest
    baseline_tokens: Annotated[list[Token], Field(min_length=1, max_length=1024)]
    candidate_tokens: Annotated[list[Token], Field(min_length=1, max_length=1024)]

    @model_validator(mode="after")
    def matching_lengths(self):
        """A fixed-length diagnostic compares every selected token in both runs."""
        if len(self.baseline_tokens) != len(self.candidate_tokens):
            raise ValueError("free-running continuation lengths differ")
        return self

    @computed_field
    @property
    def first_divergence_index(self) -> int | None:
        """Zero-based first unequal selected ID, or None for an exact continuation."""
        return next(
            (
                index
                for index, (baseline, candidate) in enumerate(
                    zip(self.baseline_tokens, self.candidate_tokens, strict=True)
                )
                if baseline != candidate
            ),
            None,
        )

    @computed_field
    @property
    def matching_prefix_tokens(self) -> int:
        """Length of the shared continuation before each run follows its own tokens."""
        divergence = self.first_divergence_index
        return len(self.baseline_tokens) if divergence is None else divergence

    @computed_field
    @property
    def matching_token_positions(self) -> int:
        """Positional matches, including positions after prefixes may have diverged."""
        return sum(
            baseline == candidate
            for baseline, candidate in zip(
                self.baseline_tokens, self.candidate_tokens, strict=True
            )
        )

    @computed_field
    @property
    def baseline_eos_indices(self) -> list[int]:
        """All selected native EOS IDs, including those followed by further tokens."""
        return [index for index, token in enumerate(self.baseline_tokens) if token == 1]

    @computed_field
    @property
    def candidate_eos_indices(self) -> list[int]:
        """EOS is reported without filtering or truncating the candidate continuation."""
        return [
            index for index, token in enumerate(self.candidate_tokens) if token == 1
        ]


def compare(reference: Path, frozen: Path, candidate: Path) -> GenerationComparison:
    """Validate both complete greedy captures and retain the original baseline IDs."""
    if any((root / "targets.jsonl").exists() for root in (reference, candidate)):
        raise ValueError("free-running diagnostic refuses teacher-forced captures")
    repeat = verify(reference, frozen)
    if not repeat.windows_exact or not repeat.logits_exact:
        raise ValueError("baseline no longer matches the frozen reference")
    manifest = (frozen / "reference.json").read_bytes()
    record = Reference.model_validate_json(manifest)
    prompt = read_tokens(frozen / "prompt.ids")
    baseline = read_tokens(frozen / "targets.ids")
    windows = read_records(candidate / "windows.jsonl", Window)
    prefill = (len(prompt) + 15) // 16
    if len(windows) != prefill + record.target_positions:
        raise ValueError("candidate lacks the complete fixed-length greedy capture")
    selected = [
        window.next
        for window in windows[prefill - 1 : prefill - 1 + record.target_positions]
    ]
    validated, _ = trajectory(candidate, prompt, selected, "greedy")
    return GenerationComparison(
        reference_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        candidate_capture=str(candidate.resolve()),
        candidate_ordered_logits_sha256=capture_hash(candidate, validated),
        baseline_tokens=baseline,
        candidate_tokens=selected,
    )


def main() -> None:
    """Write a new diagnostic; valid divergence is reported without a pass/fail gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = compare(args.reference, args.frozen, args.candidate)
    with args.report.open("x") as output:
        output.write(report.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
