"""Independent analytic and malformed-artifact checks for the DeepSeek scorer."""

import json
import math
import tempfile
import unittest
from array import array
from pathlib import Path

from compare_ds4_layer_logits import VOCABULARY, compare_case, row_metrics


class LogitComparisonTests(unittest.TestCase):
    """Check divergence direction, numeric stability and actual fixed-input alignment."""

    def test_known_asymmetric_divergence(self):
        reference = [math.log(0.9), math.log(0.1)]
        candidate = [math.log(0.5), math.log(0.5)]
        kl, rb, cb = row_metrics(reference, candidate)
        self.assertAlmostEqual(kl, 0.9 * math.log(1.8) + 0.1 * math.log(0.2), places=14)
        self.assertEqual((rb, cb), (0, 0))
        reverse, _, _ = row_metrics(candidate, reference)
        self.assertAlmostEqual(
            reverse, 0.5 * math.log(5 / 9) + 0.5 * math.log(5), places=14
        )
        self.assertNotAlmostEqual(kl, reverse, places=3)

    def test_symmetric_swap_and_first_maximum(self):
        kl, rb, cb = row_metrics(
            [math.log(0.75), math.log(0.25)], [math.log(0.25), math.log(0.75)]
        )
        self.assertAlmostEqual(kl, 0.5 * math.log(3), places=14)
        self.assertEqual((rb, cb), (0, 1))
        self.assertEqual(row_metrics([3.0, 3.0, 2.0], [5.0, 5.0, 4.0]), (0.0, 0, 0))

    def test_large_common_offset_keeps_normalization(self):
        self.assertEqual(row_metrics([1e30, 1e30], [0.0, 0.0]), (0.0, 0, 0))
        kl, _, _ = row_metrics([0.0, -10000.0], [-10000.0, 0.0])
        self.assertEqual(kl, 10000.0)

    def test_invalid_rows(self):
        for reference, candidate in [
            ([], []),
            ([0.0], [0.0, 1.0]),
            ([math.nan], [0.0]),
            ([0.0], [math.inf]),
        ]:
            with (
                self.subTest(reference=reference, candidate=candidate),
                self.assertRaises(ValueError),
            ):
                row_metrics(reference, candidate)

    def fixture(self, root: Path):
        """Write independent native-format records with a target EOS and divergent prediction."""
        prompt, targets = root / "prompt.ids", root / "targets.ids"
        prompt.write_text("42 43\n")
        targets.write_text("1 6 7\n")
        for name in ("reference", "candidate"):
            folder = root / name
            folder.mkdir()
            predictions = [9, 10 if name == "reference" else 21, 11, 12]
            windows = [
                {
                    "window": index + 1,
                    "position": index + 2,
                    "tokens": [42, 43] if index == 0 else [[1], [6], [7]][index - 1],
                    "next": prediction,
                }
                for index, prediction in enumerate(predictions)
            ]
            target_rows = [
                {
                    "index": index,
                    "window": index + 1,
                    "position": index + 2,
                    "target": target,
                    "predicted": predictions[index],
                }
                for index, target in enumerate((1, 6, 7))
            ]
            (folder / "windows.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in windows)
            )
            (folder / "targets.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in target_rows)
            )
            for index, prediction in enumerate(predictions, 1):
                row = array("f", [0.0]) * VOCABULARY
                row[prediction] = 1.0
                (folder / f"window-{index}.f32").write_bytes(row.tobytes())
        return root / "reference", root / "candidate", prompt, targets

    def test_complete_aligned_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            report = compare_case(*paths)
            self.assertEqual(report.scored_positions, 3)
            self.assertEqual(report.captured_windows, 4)
            self.assertEqual(report.exact_float_rows, 3)
            self.assertEqual(report.top1_agreement, 2 / 3)
            self.assertEqual([row.window for row in report.positions], [1, 2, 3])
            self.assertEqual([row.target for row in report.positions], [1, 6, 7])
            self.assertGreater(report.mean_kl_reference_candidate, 0)
            self.assertNotEqual(
                report.ordered_logit_bytes_sha256["reference"],
                report.ordered_logit_bytes_sha256["candidate"],
            )
            exact = compare_case(paths[0], paths[0], paths[2], paths[3])
            self.assertEqual(exact.mean_kl_reference_candidate, 0)
            self.assertEqual(exact.top1_agreement, 1)
            self.assertEqual(exact.exact_float_rows, 4)
            self.assertEqual(
                json.loads(report.model_dump_json())["scored_positions"], 3
            )

    def test_invalid_capture_is_refused(self):
        for mutation in (
            "consumed_target",
            "target_window",
            "prediction",
            "missing_final",
            "nonfinite",
            "truncated",
            "extra_file",
            "boolean_index",
            "duplicate_index",
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                paths = self.fixture(Path(temporary))
                candidate = paths[1]
                windows = [
                    json.loads(line)
                    for line in (candidate / "windows.jsonl").read_text().splitlines()
                ]
                targets = [
                    json.loads(line)
                    for line in (candidate / "targets.jsonl").read_text().splitlines()
                ]
                if mutation == "consumed_target":
                    windows[2]["tokens"] = [8]
                elif mutation == "target_window":
                    targets[1]["window"] = 3
                elif mutation == "prediction":
                    windows[1]["next"] = targets[1]["predicted"] = 22
                elif mutation == "missing_final":
                    windows.pop()
                    (candidate / "window-4.f32").unlink()
                elif mutation == "nonfinite":
                    row = array("f", [0.0]) * VOCABULARY
                    row[0] = math.nan
                    (candidate / "window-2.f32").write_bytes(row.tobytes())
                elif mutation == "truncated":
                    (candidate / "window-2.f32").write_bytes(b"\0" * 4)
                elif mutation == "extra_file":
                    (candidate / "window-99.f32").touch()
                elif mutation == "boolean_index":
                    targets[0]["index"] = False
                elif mutation == "duplicate_index":
                    targets[1]["index"] = 0
                (candidate / "windows.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in windows)
                )
                (candidate / "targets.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in targets)
                )
                with self.assertRaises(ValueError):
                    compare_case(*paths)

    def test_explicit_greedy_reference_requires_its_actual_selected_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            selected = [9, 10, 11]
            paths[3].write_text("9 10 11\n")
            for folder in paths[:2]:
                windows = [
                    json.loads(line)
                    for line in (folder / "windows.jsonl").read_text().splitlines()
                ]
                for index, target in enumerate(selected, 1):
                    windows[index]["tokens"] = [target]
                (folder / "windows.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in windows)
                )
                if folder == paths[0]:
                    (folder / "targets.jsonl").unlink()
                else:
                    rows = [
                        json.loads(line)
                        for line in (folder / "targets.jsonl").read_text().splitlines()
                    ]
                    for row, target in zip(rows, selected, strict=True):
                        row["target"] = target
                    (folder / "targets.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
            report = compare_case(*paths, reference_mode="greedy")
            self.assertEqual(report.reference_mode, "greedy")
            self.assertEqual(report.top1_agreement, 2 / 3)
            self.assertNotIn("reference_targets", report.input_sha256)
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "greedy reference target differs"):
                compare_case(*paths, reference_mode="greedy")

    def test_native_case_input_bounds(self):
        for prompt_count, target_count in (
            (1901, 3),
            (1, 1025),
            (1848, 201),
            (0, 3),
            (2, 0),
        ):
            with (
                self.subTest(prompt=prompt_count, targets=target_count),
                tempfile.TemporaryDirectory() as temporary,
            ):
                paths = self.fixture(Path(temporary))
                paths[2].write_text("0 " * prompt_count)
                paths[3].write_text("1 " * target_count)
                with self.assertRaisesRegex(
                    ValueError, "input exceeds the native case bounds"
                ):
                    compare_case(*paths)


if __name__ == "__main__":
    unittest.main()
