"""Independent free-running fixtures with EOS, divergence and invalid raw rows."""

import json
import math
import tempfile
import unittest
from array import array
from pathlib import Path

from tools.compare_ds4_generation import compare
from tools.compare_ds4_layer_logits import VOCABULARY
from tools.freeze_ds4_reference import freeze


class GenerationTests(unittest.TestCase):
    """A valid divergent continuation stays diagnostic; damaged captures fail loudly."""

    def capture(self, root: Path, name: str, selected: list[int]) -> Path:
        """Write three greedy selected IDs and their final extra argmax row."""
        destination = root / name
        destination.mkdir()
        consumed = [[42, 43], *[[token] for token in selected]]
        predicted = [*selected, 7]
        windows = []
        position = 0
        for index, (tokens, token) in enumerate(
            zip(consumed, predicted, strict=True), 1
        ):
            position += len(tokens)
            windows.append(
                {"window": index, "position": position, "tokens": tokens, "next": token}
            )
            values = array("f", [0]) * VOCABULARY
            values[token] = 1
            (destination / f"window-{index}.f32").write_bytes(values.tobytes())
        (destination / "windows.jsonl").write_text(
            "".join(json.dumps(window) + "\n" for window in windows)
        )
        return destination

    def setup_captures(self, root: Path, candidate_tokens: list[int]):
        """Freeze an independent baseline whose second selected token is EOS."""
        prompt = root / "prompt.ids"
        prompt.write_text("42 43\n")
        reference = self.capture(root, "reference", [4, 1, 6])
        frozen = root / "frozen"
        freeze(reference, prompt, 3, frozen)
        candidate = self.capture(root, "candidate", candidate_tokens)
        return reference, frozen, candidate

    def test_divergence_retains_eos_and_records_shared_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = compare(*self.setup_captures(Path(temporary), [4, 5, 6]))
            self.assertEqual(report.first_divergence_index, 1)
            self.assertEqual(report.matching_prefix_tokens, 1)
            self.assertEqual(report.matching_token_positions, 2)
            self.assertEqual(report.baseline_tokens, [4, 1, 6])
            self.assertEqual(report.candidate_tokens, [4, 5, 6])
            self.assertEqual(report.baseline_eos_indices, [1])
            self.assertEqual(report.candidate_eos_indices, [])
            self.assertNotIn("passes", report.model_dump())

    def test_exact_tokens_do_not_require_identical_raw_candidate_logits(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.setup_captures(Path(temporary), [4, 1, 6])
            path = paths[2] / "window-4.f32"
            values = array("f")
            values.frombytes(path.read_bytes())
            values[8] = 0.1
            path.write_bytes(values.tobytes())
            report = compare(*paths)
            self.assertIsNone(report.first_divergence_index)
            self.assertEqual(report.matching_prefix_tokens, 3)
            self.assertEqual(report.candidate_eos_indices, [1])

    def test_teacher_forced_even_matching_feeds_and_missing_final_refused(self):
        for fault in ("teacher-forced", "missing-final", "extra-row", "wrong-feed"):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(fault=fault):
                paths = self.setup_captures(Path(temporary), [4, 1, 6])
                candidate = paths[2]
                if fault == "teacher-forced":
                    (candidate / "targets.jsonl").write_text("")
                elif fault == "missing-final":
                    (candidate / "window-4.f32").unlink()
                elif fault == "extra-row":
                    (candidate / "window-99.f32").write_bytes(b"")
                else:
                    path = candidate / "windows.jsonl"
                    rows = [json.loads(row) for row in path.read_text().splitlines()]
                    rows[2]["tokens"] = [5]
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    compare(*paths)

    def test_all_rows_checked_and_changed_baseline_rejected(self):
        for owner, row, value in (
            (2, 1, math.nan),
            (2, 4, math.inf),
            (2, 4, 2.0),
            (0, 2, 0.1),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                paths = self.setup_captures(Path(temporary), [4, 1, 6])
                path = paths[owner] / f"window-{row}.f32"
                values = array("f")
                values.frombytes(path.read_bytes())
                values[8] = value
                path.write_bytes(values.tobytes())
                with self.subTest(owner=owner, row=row), self.assertRaises(ValueError):
                    compare(*paths)


if __name__ == "__main__":
    unittest.main()
