"""Check greedy/EOS interpretation and exact task grading against independent rows."""

import hashlib
import json
import tempfile
import unittest
from array import array
from pathlib import Path

from pydantic import ValidationError

from tools.compare_ds4_layer_logits import VOCABULARY
from tools.ds4_quality_inputs import NORMALIZATION, QualityInputs, TaskCase
from tools.grade_ds4_tasks import (
    TaskReport,
    TaskResult,
    grade_task,
    greedy_capture,
    retain,
)


class Decoder:
    """Test decoder records exactly which IDs the capture checker presents."""

    def __init__(self, text):
        self.text = text
        self.seen = None

    def decode(self, ids, errors):
        self.seen = ids
        if errors != "strict":
            raise AssertionError("decoder silently allowed damaged UTF-8")
        return self.text


class TaskTests(unittest.TestCase):
    """Capture interpretation must agree with the native generation loop."""

    def fixture(self, root: Path):
        """One generated content token followed by a predicted but unfed EOS."""
        prompt = root / "prompt.ids"
        prompt.write_text("42\n")
        folder = root / "capture"
        folder.mkdir()
        windows = [
            {"window": 1, "position": 1, "tokens": [42], "next": 7},
            {"window": 2, "position": 2, "tokens": [7], "next": 1},
        ]
        (folder / "windows.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in windows)
        )
        for row in windows:
            values = array("f", [0]) * VOCABULARY
            values[row["next"]] = 1
            (folder / f"window-{row['window']}.f32").write_bytes(values.tobytes())
        case = TaskCase(
            id="multiply",
            prompt_tokens=1,
            expected="703",
            normalization=NORMALIZATION,
            prompt_ids_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
        )
        return folder, prompt, case

    def test_eos_and_limit_have_different_emitted_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, prompt, case = self.fixture(Path(temporary))
            emitted, stopped, _, _ = greedy_capture(folder, [42], 128)
            self.assertEqual(emitted, [7, 1])
            self.assertTrue(stopped)
            decoder = Decoder("703")
            result = grade_task(folder, prompt, case, decoder, 128)
            self.assertEqual(decoder.seen, [7])
            self.assertTrue(result.matches_expected)
            emitted, stopped, _, _ = greedy_capture(folder, [42], 1)
            self.assertEqual(emitted, [7])
            self.assertFalse(stopped)  # final extra row's EOS was never emitted

    def test_no_answer_extraction_numeric_coercion_or_fence_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, prompt, case = self.fixture(Path(temporary))
            for text, passed in [
                ("703", True),
                (" 703\n", True),
                ("Answer: 703", False),
                ("`703`", False),
                ("703.0", False),
                ("703 extra", False),
            ]:
                with self.subTest(text=text):
                    self.assertEqual(
                        grade_task(
                            folder, prompt, case, Decoder(text), 128
                        ).matches_expected,
                        passed,
                    )

    def test_wrong_prompt_or_teacher_forcing_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, prompt, case = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "frozen prompt"):
                greedy_capture(folder, [43], 128)
            (folder / "targets.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "free-running greedy"):
                grade_task(folder, prompt, case, Decoder("703"), 128)
            (folder / "targets.jsonl").unlink()
            prompt.write_text("43\n")
            with self.assertRaisesRegex(ValueError, "frozen input"):
                grade_task(folder, prompt, case, Decoder("703"), 128)

    def test_extra_or_wrong_raw_rows_are_not_graded(self):
        for fault in ("extra", "nonfinite", "argmax"):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(fault=fault):
                folder, prompt, case = self.fixture(Path(temporary))
                if fault == "extra":
                    (folder / "window-99.f32").write_bytes(b"")
                else:
                    file = folder / "window-1.f32"
                    values = array("f")
                    values.frombytes(file.read_bytes())
                    values[8] = float("nan") if fault == "nonfinite" else 2
                    file.write_bytes(values.tobytes())
                with self.assertRaises(ValueError):
                    grade_task(folder, prompt, case, Decoder("703"), 128)

    def test_retention_requires_same_frozen_tasks_and_decode_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.json"
            assets = {
                name: "a" * 64
                for name in [
                    "vocab.json",
                    "merges.txt",
                    "tokenizer.json",
                    "token_type.json",
                    "chat_template.jinja",
                ]
            }
            corpus = QualityInputs.model_validate(
                {
                    "status": "synthetic test inputs",
                    "logit_cases": [
                        {
                            "id": f"logit-{i}",
                            "prompt_tokens": 50,
                            "target_positions": 200,
                            "context_at_last_consumed_target": 250,
                            "prompt_ids_sha256": "b" * 64,
                        }
                        for i in range(5)
                    ],
                    "task_cases": [
                        {
                            "id": f"case-{i}",
                            "prompt_tokens": 1,
                            "expected": "703",
                            "normalization": NORMALIZATION,
                            "prompt_ids_sha256": "b" * 64,
                        }
                        for i in range(8)
                    ],
                    "thresholds_from_approved_plan": {
                        "minimum_positions": 1000,
                        "minimum_prompts": 5,
                        "mean_kl_reference_candidate_max_nat": 0.01,
                        "top1_agreement_min": 0.99,
                        "task_gate": "retain every baseline pass across the eight fixed tasks",
                    },
                    "tokenizer_assets_sha256": assets,
                    "template_settings": {"thinking": False, "enable_thinking": False},
                    "source_code_sha256": {
                        "tools/strata_tokenizer.py": "c" * 64,
                        "serve/frontend.py": "c" * 64,
                    },
                    "limits": "synthetic fixture, no runtime claims",
                }
            )
            path.write_text(corpus.model_dump_json())
            results = [
                TaskResult(
                    task=case.id,
                    expected=case.expected,
                    decoded="703",
                    matches_expected=True,
                    emitted_token_ids=[7, 1],
                    stopped_at_eos=True,
                    captured_windows=2,
                    prompt_sha256=case.prompt_ids_sha256,
                    ordered_logits_sha256="d" * 64,
                )
                for case in corpus.task_cases
            ]
            baseline = TaskReport(
                corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                output_root="synthetic",
                decode_limit=128,
                tokenizer_assets_sha256=assets,
                results=results,
            )
            candidate = TaskReport.model_validate(baseline.model_dump())
            self.assertEqual(retain(path, baseline, candidate).lost_baseline_passes, [])
            data = candidate.model_dump()
            data["results"][3].update(decoded="Answer: 703", matches_expected=False)
            changed = TaskReport.model_validate(data)
            self.assertEqual(
                retain(path, baseline, changed).lost_baseline_passes, ["case-3"]
            )
            for key, value in [("decode_limit", 256), ("corpus_sha256", "e" * 64)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    retain(
                        path,
                        baseline,
                        TaskReport.model_validate(
                            candidate.model_dump() | {key: value}
                        ),
                    )
            data = candidate.model_dump()
            data["results"][3].update(expected="709", decoded="709")
            with self.assertRaisesRegex(ValueError, "answer/prompt"):
                retain(path, baseline, TaskReport.model_validate(data))
            data = candidate.model_dump()
            data["results"][3]["decoded"] = "wrong"
            with self.assertRaises(ValidationError):
                TaskReport.model_validate(data)


if __name__ == "__main__":
    unittest.main()
