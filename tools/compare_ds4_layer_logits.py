"""Compare complete native DeepSeek teacher-forced dumps on identical inputs.

Run with uv and pydantic, as documented in docs/DEEPSEEK_LAYER_RPC.md.
This scores one case; corpus, artifact identity and task gates remain separate.
"""

import argparse
import hashlib
import math
import sys
from array import array
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, computed_field

VOCABULARY = 129280
Token = Annotated[int, Field(ge=0, lt=VOCABULARY)]
WindowIndex = Annotated[int, Field(ge=1, le=2048)]
TokenList = TypeAdapter(list[Token])
TokenTextList = TypeAdapter(list[Annotated[str, Field(pattern=r"^[0-9]+$")]])
ReferenceMode = Literal["teacher-forced", "greedy"]


class Record(BaseModel):
    """Strict boundary schema for the native harness's JSON records."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


RecordT = TypeVar("RecordT", bound=Record)


class Window(Record):
    """One completed feed in native windows.jsonl; next is that row's argmax."""

    window: WindowIndex
    position: Annotated[int, Field(ge=1, le=2048)]
    tokens: Annotated[list[Token], Field(min_length=1, max_length=16)]
    next: Token


class Target(Record):
    """The current row scored before a fixed target is consumed."""

    index: Annotated[int, Field(ge=0, lt=1024)]
    window: WindowIndex
    position: Annotated[int, Field(ge=1, lt=2048)]
    target: Token
    predicted: Token


class Position(Record):
    """Natural-log divergence and argmax evidence for one fixed-prefix target."""

    index: int
    window: int
    position: int
    target: int
    reference_argmax: int
    candidate_argmax: int
    kl_reference_candidate: float


class Comparison(Record):
    """One case's numerical evidence; this is not an E5 acceptance receipt."""

    scope: Literal["same-prefix logit comparison only"] = (
        "same-prefix logit comparison only"
    )
    reference: str
    candidate: str
    reference_mode: ReferenceMode = "teacher-forced"
    captured_windows: int
    exact_float_rows: int
    input_sha256: dict[str, str]
    ordered_logit_bytes_sha256: dict[str, str]
    positions: Annotated[list[Position], Field(min_length=1, max_length=1024)]

    @computed_field
    @property
    def scored_positions(self) -> int:
        """Number of fixed target positions, excluding prefill and final extra row."""
        return len(self.positions)

    @computed_field
    @property
    def mean_kl_reference_candidate(self) -> float:
        """Position-weighted mean KL(reference || candidate), in nats."""
        return math.fsum(row.kl_reference_candidate for row in self.positions) / len(
            self.positions
        )

    @computed_field
    @property
    def top1_agreement(self) -> float:
        """Fraction of target positions with the same first-maximum token."""
        return sum(
            row.reference_argmax == row.candidate_argmax for row in self.positions
        ) / len(self.positions)


def read_records(path: Path, schema: type[RecordT]) -> list[RecordT]:
    """Read native JSON lines through the corresponding strict boundary schema."""
    return [schema.model_validate_json(line) for line in path.read_bytes().splitlines()]


def read_tokens(path: Path) -> list[int]:
    """Validate the native harness's whitespace-separated decimal token inputs."""
    return TokenList.validate_python(
        TokenTextList.validate_python(path.read_text().split())
    )


def trajectory(
    root: Path,
    prompt: list[int],
    targets: list[int],
    mode: ReferenceMode = "teacher-forced",
) -> tuple[list[Window], list[Target]]:
    """Require a complete prompt/target trajectory with its final post-target row."""
    windows = read_records(root / "windows.jsonl", Window)
    if mode not in ("teacher-forced", "greedy"):
        raise ValueError("unknown reference capture mode")
    scored = (
        read_records(root / "targets.jsonl", Target) if mode == "teacher-forced" else []
    )
    prefill = (len(prompt) + 15) // 16
    if (
        not prompt
        or len(prompt) > 1900
        or not targets
        or len(targets) > 1024
        or len(prompt) + len(targets) > 2048
    ):
        raise ValueError("prompt/target input exceeds the native case bounds")
    expected_inputs = [
        prompt[offset : offset + 16] for offset in range(0, len(prompt), 16)
    ]
    expected_inputs += [[target] for target in targets]
    if len(windows) != len(expected_inputs) or (
        mode == "teacher-forced" and len(scored) != len(targets)
    ):
        raise ValueError(
            "incomplete teacher-forced trajectory or missing final post-target row"
        )
    position = 0
    for number, (window, consumed) in enumerate(
        zip(windows, expected_inputs, strict=True), 1
    ):
        position += len(consumed)
        if (window.window, window.position, window.tokens) != (
            number,
            position,
            consumed,
        ):
            raise ValueError(
                "recorded window does not consume the fixed prompt/targets"
            )
    if mode == "greedy":
        for index, target in enumerate(targets):
            window = windows[prefill - 1 + index]
            if window.next != target:
                raise ValueError(
                    "greedy reference target differs from its selected token"
                )
            scored.append(
                Target(
                    index=index,
                    window=window.window,
                    position=window.position,
                    target=target,
                    predicted=window.next,
                )
            )
    for index, (row, target) in enumerate(zip(scored, targets, strict=True)):
        window = windows[prefill - 1 + index]
        if (row.index, row.window, row.position, row.target, row.predicted) != (
            index,
            window.window,
            window.position,
            target,
            window.next,
        ):
            raise ValueError(
                "target metadata does not identify its pre-consumption logit row"
            )
    expected_files = {f"window-{window.window}.f32" for window in windows}
    if {path.name for path in root.glob("window-*.f32")} != expected_files:
        raise ValueError(
            "logit-file inventory differs from the completed window history"
        )
    return windows, scored


def row_metrics(reference, candidate) -> tuple[float, int, int]:
    """Stable float64 KL and first-maximum IDs for finite equal-size logit rows."""
    if not reference or len(reference) != len(candidate):
        raise ValueError("logit row sizes differ or are empty")
    if not all(math.isfinite(value) for value in reference) or not all(
        math.isfinite(value) for value in candidate
    ):
        raise ValueError("nonfinite logit")
    rmax, cmax = max(reference), max(candidate)
    rz = math.log(math.fsum(math.exp(value - rmax) for value in reference))
    cz = math.log(math.fsum(math.exp(value - cmax) for value in candidate))
    # Keep centered log probabilities: adding log(sum(exp)) to a very large
    # raw maximum would lose the normalization term through float rounding.
    kl = math.fsum(
        math.exp(r - rmax - rz) * ((r - rmax - rz) - (c - cmax - cz))
        for r, c in zip(reference, candidate, strict=True)
    )
    if not math.isfinite(kl):
        raise ValueError("nonfinite logit divergence")
    return kl, reference.index(rmax), candidate.index(cmax)


def logits(path: Path) -> tuple[bytes, array]:
    """Read one complete little-endian float32 row of the pinned vocabulary."""
    raw = path.read_bytes()
    if len(raw) != VOCABULARY * 4:
        raise ValueError("logit file is not one complete DeepSeek float32 row")
    values = array("f")
    if values.itemsize != 4:
        raise RuntimeError("analysis platform must support 32-bit C float arrays")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return raw, values


def compare_case(
    reference: Path,
    candidate: Path,
    prompt_path: Path,
    target_path: Path,
    reference_mode: ReferenceMode = "teacher-forced",
) -> Comparison:
    """Validate both complete trajectories and stream their saved rows for scoring."""
    prompt, targets = read_tokens(prompt_path), read_tokens(target_path)
    reference_windows, reference_targets = trajectory(
        reference, prompt, targets, reference_mode
    )
    candidate_windows, _ = trajectory(candidate, prompt, targets)
    scored = {row.window: row for row in reference_targets}
    digest_r, digest_c = hashlib.sha256(), hashlib.sha256()
    positions, exact = [], 0
    for rw, cw in zip(reference_windows, candidate_windows, strict=True):
        raw_r, r = logits(reference / f"window-{rw.window}.f32")
        raw_c, c = logits(candidate / f"window-{cw.window}.f32")
        digest_r.update(raw_r)
        digest_c.update(raw_c)
        kl, rbest, cbest = row_metrics(r, c)
        if (rbest, cbest) != (rw.next, cw.next):
            raise ValueError("predicted ID differs from recorded logits")
        exact += raw_r == raw_c
        if rw.window in scored:
            row = scored[rw.window]
            positions.append(
                Position(
                    index=row.index,
                    window=row.window,
                    position=row.position,
                    target=row.target,
                    reference_argmax=rbest,
                    candidate_argmax=cbest,
                    kl_reference_candidate=kl,
                )
            )
    inputs = {
        "prompt_ids": prompt_path,
        "target_ids": target_path,
        "reference_windows": reference / "windows.jsonl",
        "candidate_windows": candidate / "windows.jsonl",
        "candidate_targets": candidate / "targets.jsonl",
    }
    if reference_mode == "teacher-forced":
        inputs["reference_targets"] = reference / "targets.jsonl"
    return Comparison(
        reference=str(reference.resolve()),
        candidate=str(candidate.resolve()),
        reference_mode=reference_mode,
        captured_windows=len(reference_windows),
        exact_float_rows=exact,
        input_sha256={
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in inputs.items()
        },
        ordered_logit_bytes_sha256={
            "reference": digest_r.hexdigest(),
            "candidate": digest_c.hexdigest(),
        },
        positions=positions,
    )


def main() -> None:
    """Write a new, nonoverwriting numerical report for one teacher-forced case."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "prompt_ids", "target_ids"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference-mode",
        choices=("teacher-forced", "greedy"),
        default="teacher-forced",
    )
    args = parser.parse_args()
    report = compare_case(
        args.reference,
        args.candidate,
        args.prompt_ids,
        args.target_ids,
        args.reference_mode,
    )
    with args.output.open("x") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    print(
        f"Compared {report.scored_positions} target positions: mean KL={report.mean_kl_reference_candidate:.9g} nat; top-1={report.top1_agreement:.6%}. E5 acceptance is separate."
    )


if __name__ == "__main__":
    main()
