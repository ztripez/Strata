"""Independent capture fixtures for exact continuation freezing and repeat checks."""

import json
import math
import shutil
import tempfile
import unittest
from array import array
from pathlib import Path

from pydantic import ValidationError

from tools.compare_ds4_layer_logits import VOCABULARY, read_tokens
from tools.freeze_ds4_reference import freeze, verify


class ReferenceTests(unittest.TestCase):
    """Bad captures must not become trusted continuations or exact repeat receipts."""

    def capture(self, root: Path):
        """Write a greedy two-target trajectory that includes a selected EOS."""
        capture = root / "capture"
        capture.mkdir()
        prompt = root / "prompt.ids"
        prompt.write_text("42 43\n")
        windows = [
            {"window": 1, "position": 2, "tokens": [42, 43], "next": 1},
            {"window": 2, "position": 3, "tokens": [1], "next": 4},
            {"window": 3, "position": 4, "tokens": [4], "next": 5},
        ]
        (capture / "windows.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in windows)
        )
        for window in windows:
            values = array("f", [0]) * VOCABULARY
            values[window["next"]] = 1
            (capture / f"window-{window['window']}.f32").write_bytes(values.tobytes())
        return capture, prompt

    def test_exact_continuation_retains_eos_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, prompt = self.capture(root)
            frozen = root / "frozen"
            record = freeze(capture, prompt, 2, frozen)
            self.assertEqual(read_tokens(frozen / "targets.ids"), [1, 4])
            self.assertEqual(record.target_positions, 2)
            repeat = root / "repeat"
            shutil.copytree(capture, repeat)
            report = verify(repeat, frozen)
            self.assertTrue(report.windows_exact and report.logits_exact)
            with self.assertRaises(FileExistsError):
                freeze(capture, prompt, 2, frozen)

    def test_finite_logit_change_is_not_an_exact_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, prompt = self.capture(root)
            frozen = root / "frozen"
            freeze(capture, prompt, 2, frozen)
            path = capture / "window-3.f32"
            values = array("f")
            values.frombytes(path.read_bytes())
            values[7] = 0.1  # same argmax, different raw numerical output
            path.write_bytes(values.tobytes())
            report = verify(capture, frozen)
            self.assertTrue(report.windows_exact)
            self.assertFalse(report.logits_exact)

    def test_nonfinite_wrong_argmax_and_incomplete_capture_refused(self):
        for fault in ("nan", "prediction", "missing-final", "extra-row"):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(fault=fault):
                root = Path(temporary)
                capture, prompt = self.capture(root)
                if fault == "missing-final":
                    (capture / "window-3.f32").unlink()
                elif fault == "extra-row":
                    (capture / "window-99.f32").write_bytes(b"")
                else:
                    path = capture / "window-1.f32"
                    values = array("f")
                    values.frombytes(path.read_bytes())
                    values[7] = math.nan if fault == "nan" else 2
                    path.write_bytes(values.tobytes())
                with self.assertRaises((ValueError, FileNotFoundError)):
                    freeze(capture, prompt, 2, root / "frozen")
                self.assertFalse((root / "frozen").exists())

    def test_native_decimal_token_syntax_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokens.ids"
            for text in ("1.0", "1e0", "+1", "-1", "129280"):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValidationError):
                    read_tokens(path)
            path.write_text("0 001 129279\n")
            self.assertEqual(read_tokens(path), [0, 1, 129279])


if __name__ == "__main__":
    unittest.main()
