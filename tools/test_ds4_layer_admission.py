"""Check native layer CLI refusals before model loading or device selection.

Run with ``uv run --no-project tools/test_ds4_layer_admission.py EXECUTABLE``.
An already-built CUDA or HIP ``deepseek_layer_tcp_test`` is required.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Check the actual native executable and print exact refusal receipts."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_ds4_layer_admission.py NATIVE_LAYER_EXECUTABLE")
    executable = Path(sys.argv[1]).resolve(strict=True)
    receipts = []
    with tempfile.TemporaryDirectory(prefix="strata-hybrid-admission-") as temporary:
        root = Path(temporary)
        source, placement, output = (
            root / "source.gguf",
            root / "placement",
            root / "out",
        )
        source.touch()
        placement.touch()
        prompt, targets = root / "prompt.ids", root / "targets.ids"
        empty, invalid = root / "empty.ids", root / "invalid.ids"
        excessive, wide = root / "excessive.ids", root / "wide.ids"
        prompt.write_text("42\n")
        targets.write_text("1 6 7\n")
        empty.touch()
        invalid.write_text("129280\n")
        excessive.write_text("0 " * 1025)
        wide.write_text("0 " * 1900)
        continuation = root / "continuation.ids"
        continuation.write_text("0 " * 200)
        worker = ["--expert-worker", "127.0.0.1:19171"]
        model = ["--expert-worker-model", str(source)]
        plan = ["--expert-worker-placement", str(placement)]
        identity = ["--plan-id", "11" * 32]
        role = [
            "suffix",
            "PACK",
            "20",
            "127.0.0.1:19170",
            "600000",
            "3600000",
            "1",
            str(output),
        ]
        qualified = worker + model + plan + identity
        gpu_budget = ["--expert-vram-mib", "384"]
        teacher = ["--decode-mode", "teacher-forced"]
        reference = ["reference", "PACK", str(prompt), str(targets), "1", str(output)]
        cases = [
            ("partial-worker", worker + role, "required together", "1"),
            ("partial-source", model + role, "required together", "1"),
            ("partial-placement", plan + role, "required together", "1"),
            (
                "missing-plan-id",
                worker + model + plan + role,
                "explicit shared --plan-id",
                "1",
            ),
            (
                "zero-plan-id",
                worker + model + plan + ["--plan-id", "00" * 32] + role,
                "missing nonzero SHA-256",
                "1",
            ),
            (
                "invalid-plan-id",
                ["--plan-id", "zz" * 32] + role,
                "invalid pinned executable SHA-256",
                "1",
            ),
            (
                "short-plan-id",
                ["--plan-id", "11"] + role,
                "64 hexadecimal characters",
                "1",
            ),
            (
                "duplicate-option",
                qualified + worker + role,
                "repeated layer adapter option",
                "1",
            ),
            (
                "unknown-option",
                ["--unknown", "value"] + role,
                "unknown layer adapter option",
                "1",
            ),
            ("missing-option-value", ["--plan-id"], "option lacks a value", "1"),
            (
                "orphan-timeout",
                ["--expert-worker-timeout", "5"] + role,
                "timeout requires an expert worker",
                "1",
            ),
            (
                "zero-timeout",
                qualified + ["--expert-worker-timeout", "0"] + role,
                "timeout must be positive",
                "1",
            ),
            (
                "invalid-endpoint",
                ["--expert-worker", "hostname:19171"] + model + plan + identity + role,
                "invalid numeric expert-worker endpoint",
                "1",
            ),
            (
                "absent-source",
                worker
                + ["--expert-worker-model", str(root / "absent")]
                + plan
                + identity
                + role,
                "source GGUF or placement is absent",
                "1",
            ),
            (
                "source-directory",
                worker + ["--expert-worker-model", str(root)] + plan + identity + role,
                "source GGUF or placement is absent",
                "1",
            ),
            (
                "absent-placement",
                worker
                + model
                + ["--expert-worker-placement", str(root / "absent")]
                + identity
                + role,
                "source GGUF or placement is absent",
                "1",
            ),
            (
                "hybrid-per32-refused",
                qualified + role,
                "requires explicit DS4_IQ_Q8K=1",
                "0",
            ),
            ("layer-only-per32-refused", role, "requires explicit DS4_IQ_Q8K=1", "0"),
            (
                "ipv6-hybrid-per32-refused",
                ["--expert-worker", "[::1]:19171"] + model + plan + identity + role,
                "requires explicit DS4_IQ_Q8K=1",
                "0",
            ),
            (
                "invalid-layer-endpoint",
                [
                    "suffix",
                    "PACK",
                    "20",
                    "hostname:19170",
                    "600000",
                    "3600000",
                    "1",
                    str(output),
                ],
                "invalid numeric layer endpoint",
                "1",
            ),
            (
                "unknown-decode-mode",
                ["--decode-mode", "sampling"] + role,
                "decode mode must be greedy or teacher-forced",
                "1",
            ),
            (
                "suffix-decode-mode",
                teacher + role,
                "decode mode belongs to prefix/reference",
                "1",
            ),
            (
                "teacher-EOS-policy",
                teacher + reference[:-2] + ["0", str(output)],
                "teacher forcing requires 1..1024 targets and IGNORE_EOS=1",
                "1",
            ),
            (
                "teacher-empty-targets",
                teacher
                + ["reference", "PACK", str(prompt), str(empty), "1", str(output)],
                "token file must contain 1..1900",
                "1",
            ),
            (
                "teacher-invalid-target",
                teacher
                + ["reference", "PACK", str(prompt), str(invalid), "1", str(output)],
                "invalid probe argument",
                "1",
            ),
            (
                "teacher-too-many-targets",
                teacher
                + ["reference", "PACK", str(prompt), str(excessive), "1", str(output)],
                "teacher forcing requires 1..1024 targets",
                "1",
            ),
            (
                "teacher-context-overflow",
                teacher
                + ["reference", "PACK", str(wide), str(continuation), "1", str(output)],
                "generation exceeds declared context",
                "1",
            ),
            (
                "teacher-per32-refused",
                teacher + reference,
                "requires explicit DS4_IQ_Q8K=1",
                "0",
            ),
            (
                "gpu-budget-over-budget",
                ["--expert-vram-mib", "65537"] + role,
                "invalid probe argument",
                "1",
            ),
            (
                "gpu-budget-negative",
                ["--expert-vram-mib", "-1"] + role,
                "invalid probe argument",
                "1",
            ),
            (
                "gpu-budget-nan",
                ["--expert-vram-mib", "nan"] + role,
                "invalid probe argument",
                "1",
            ),
            (
                "gpu-budget-infinity",
                ["--expert-vram-mib", "inf"] + role,
                "invalid probe argument",
                "1",
            ),
            (
                "gpu-budget-invalid-text",
                ["--expert-vram-mib", "384suffix"] + role,
                "invalid probe argument",
                "1",
            ),
            (
                "gpu-budget-missing-plan",
                gpu_budget + role,
                "nonzero expert VRAM budget requires an explicit --plan-id",
                "1",
            ),
            (
                "gpu-budget-per32-refused",
                gpu_budget + identity + role,
                "requires explicit DS4_IQ_Q8K=1",
                "0",
            ),
            (
                "gpu-budget-zero-per32-refused",
                ["--expert-vram-mib", "0"] + role,
                "requires explicit DS4_IQ_Q8K=1",
                "0",
            ),
            (
                "gpu-budget-duplicate-option",
                gpu_budget + gpu_budget + identity + role,
                "repeated layer adapter option",
                "1",
            ),
        ]
        for name, arguments, expected, arithmetic in cases:
            environment = dict(os.environ, DS4_IQ_Q8K=arithmetic)
            result = subprocess.run(
                [str(executable), "--experts", "6", *arguments],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 1, (
                name,
                result.returncode,
                result.stdout,
                result.stderr,
            )
            assert expected in result.stderr, (name, result.stderr)
            assert not result.stdout, (
                name,
                "unexpected native execution",
                result.stdout,
            )
            assert not output.exists(), (name, "refused configuration created output")
            receipts.append(
                {"case": name, "exit": result.returncode, "stderr": result.stderr}
            )
    with executable.open("rb") as binary:
        binary_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    print(
        json.dumps(
            {
                "binary_sha256": binary_sha256,
                "cases": receipts,
                "scope": "CLI admission only; no model load or GPU execution",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
