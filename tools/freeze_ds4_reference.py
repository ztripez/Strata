"""Freeze an exact native greedy continuation and check reference repeatability.

This validates capture files, not runtime identity or full E5 acceptance. Run as
``uv run --no-project --with pydantic==2.14.2 python -m tools.freeze_ds4_reference``.
"""

import argparse
import hashlib
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from tools.compare_ds4_layer_logits import (
    Record,
    Window,
    logits,
    read_records,
    read_tokens,
    trajectory,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Reference(Record):
    """A continuation derived from complete, finite native greedy capture rows."""

    schema_version: Literal["strata-ds4-reference/1"] = "strata-ds4-reference/1"
    native_capture: str
    prompt_tokens: Annotated[int, Field(ge=1, le=1900)]
    target_positions: Annotated[int, Field(ge=1, le=1024)]
    captured_windows: Annotated[int, Field(ge=1, le=2048)]
    prompt_sha256: Digest
    targets_sha256: Digest
    windows_sha256: Digest
    ordered_logits_sha256: Digest
    scope: Literal["capture validation only; no runtime identity or E5 pass"] = (
        "capture validation only; no runtime identity or E5 pass"
    )


class Repeatability(Record):
    """An exact repeatability result; source identity remains a separate gate."""

    reference_manifest_sha256: Digest
    repeated_capture: str
    captured_windows: int
    target_positions: int
    windows_exact: bool
    logits_exact: bool
    observed_windows_sha256: Digest
    observed_logits_sha256: Digest


def capture_hash(root: Path, windows: list[Window]) -> str:
    """Validate every raw row's finite argmax and hash all rows in feed order."""
    digest = hashlib.sha256()
    for window in windows:
        raw, values = logits(root / f"window-{window.window}.f32")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite reference logit")
        if values.index(max(values)) != window.next:
            raise ValueError("capture argmax differs from native window metadata")
        digest.update(raw)
    return digest.hexdigest()


def freeze(root: Path, prompt_path: Path, count: int, destination: Path) -> Reference:
    """Create new immutable inputs from the actual baseline, without filtering EOS."""
    if destination.exists():
        raise FileExistsError(destination)
    prompt = read_tokens(prompt_path)
    if not 1 <= count <= 1024:
        raise ValueError("reference target count must be 1..1024")
    windows = read_records(root / "windows.jsonl", Window)
    prefill = (len(prompt) + 15) // 16
    if not prefill or len(windows) != prefill + count:
        raise ValueError(
            "reference lacks the complete greedy continuation and final row"
        )
    targets = [window.next for window in windows[prefill - 1 : prefill - 1 + count]]
    validated, _ = trajectory(root, prompt, targets, "greedy")
    logits_sha = capture_hash(root, validated)
    # Canonical whitespace changes no token; every selected ID, including EOS,
    # is preserved in its original order.
    prompt_bytes = (" ".join(map(str, prompt)) + "\n").encode()
    target_bytes = (" ".join(map(str, targets)) + "\n").encode()
    record = Reference(
        native_capture=str(root.resolve()),
        prompt_tokens=len(prompt),
        target_positions=count,
        captured_windows=len(windows),
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        targets_sha256=hashlib.sha256(target_bytes).hexdigest(),
        windows_sha256=hashlib.sha256(
            (root / "windows.jsonl").read_bytes()
        ).hexdigest(),
        ordered_logits_sha256=logits_sha,
    )
    destination.mkdir()
    (destination / "prompt.ids").write_bytes(prompt_bytes)
    (destination / "targets.ids").write_bytes(target_bytes)
    (destination / "reference.json").write_text(record.model_dump_json(indent=2) + "\n")
    for name in ("prompt.ids", "targets.ids", "reference.json"):
        (destination / name).chmod(0o444)
    return record


def verify(root: Path, frozen: Path) -> Repeatability:
    """Validate a second greedy run against all frozen inputs and raw rows."""
    manifest = (frozen / "reference.json").read_bytes()
    reference = Reference.model_validate_json(manifest)
    prompt_path, target_path = frozen / "prompt.ids", frozen / "targets.ids"
    if (
        hashlib.sha256(prompt_path.read_bytes()).hexdigest() != reference.prompt_sha256
        or hashlib.sha256(target_path.read_bytes()).hexdigest()
        != reference.targets_sha256
    ):
        raise ValueError("frozen reference token inputs changed")
    prompt, targets = read_tokens(prompt_path), read_tokens(target_path)
    if (len(prompt), len(targets)) != (
        reference.prompt_tokens,
        reference.target_positions,
    ):
        raise ValueError("frozen reference token counts differ")
    windows, _ = trajectory(root, prompt, targets, "greedy")
    if len(windows) != reference.captured_windows:
        raise ValueError("reference repeat window count differs")
    observed_logits = capture_hash(root, windows)
    observed_windows = hashlib.sha256((root / "windows.jsonl").read_bytes()).hexdigest()
    return Repeatability(
        reference_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        repeated_capture=str(root.resolve()),
        captured_windows=len(windows),
        target_positions=len(targets),
        windows_exact=observed_windows == reference.windows_sha256,
        logits_exact=observed_logits == reference.ordered_logits_sha256,
        observed_windows_sha256=observed_windows,
        observed_logits_sha256=observed_logits,
    )


def main() -> None:
    """Write a new freeze or repeat receipt, preserving failures and existing files."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("freeze")
    create.add_argument("capture", type=Path)
    create.add_argument("prompt_ids", type=Path)
    create.add_argument("target_count", type=int)
    create.add_argument("destination", type=Path)
    repeat = commands.add_parser("verify")
    repeat.add_argument("capture", type=Path)
    repeat.add_argument("frozen", type=Path)
    repeat.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "freeze":
        record = freeze(
            arguments.capture,
            arguments.prompt_ids,
            arguments.target_count,
            arguments.destination,
        )
        print(record.scope)
    else:
        if arguments.output.exists():
            raise FileExistsError(arguments.output)
        record = verify(arguments.capture, arguments.frozen)
        with arguments.output.open("x") as stream:
            stream.write(record.model_dump_json(indent=2) + "\n")
        if not (record.windows_exact and record.logits_exact):
            raise SystemExit("reference repeatability failed; receipt retained")
        print(
            "reference capture repeat is exact; runtime identity and E5 remain separate"
        )


if __name__ == "__main__":
    main()
