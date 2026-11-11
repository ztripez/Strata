"""Independent threshold and weighting checks for the complete numerical corpus."""

import unittest

from pydantic import ValidationError

from tools.compare_ds4_corpus import CaseComparison, CorpusComparison
from tools.compare_ds4_layer_logits import Comparison, Position
from tools.ds4_quality_inputs import Thresholds
from tools.freeze_ds4_reference import Repeatability


class CorpusTests(unittest.TestCase):
    """Metrics must weight positions and preserve every frozen acceptance boundary."""

    def report(
        self, counts=(200, 200, 200, 200, 200), kl=None, mismatches=None, exact=True
    ):
        kl = [0.0] * len(counts) if kl is None else kl
        mismatches = [0] * len(counts) if mismatches is None else mismatches
        cases = []
        for index, (count, divergence, errors) in enumerate(
            zip(counts, kl, mismatches, strict=True)
        ):
            repeat = Repeatability(
                reference_manifest_sha256="a" * 64,
                repeated_capture="synthetic",
                captured_windows=count + 4,
                target_positions=count,
                windows_exact=exact,
                logits_exact=exact,
                observed_windows_sha256="b" * 64,
                observed_logits_sha256="c" * 64,
            )
            comparison = Comparison(
                reference="synthetic-reference",
                candidate="synthetic-candidate",
                reference_mode="greedy",
                captured_windows=count + 4,
                exact_float_rows=0,
                input_sha256={},
                ordered_logit_bytes_sha256={},
                positions=[
                    Position(
                        index=j,
                        window=j + 4,
                        position=j + 50,
                        target=0,
                        reference_argmax=0,
                        candidate_argmax=int(j < errors),
                        kl_reference_candidate=divergence,
                    )
                    for j in range(count)
                ],
            )
            cases.append(
                CaseComparison(
                    id=f"case-{index}",
                    prompt_tokens=50,
                    target_positions=count,
                    first_reference=repeat,
                    repeated_reference=repeat,
                    comparison=comparison,
                )
            )
        thresholds = Thresholds(
            minimum_positions=1000,
            minimum_prompts=5,
            mean_kl_reference_candidate_max_nat=0.01,
            top1_agreement_min=0.99,
            task_gate="retain every baseline pass across the eight fixed tasks",
        )
        return CorpusComparison(
            corpus_sha256="d" * 64, thresholds=thresholds, cases=cases
        )

    def test_exact_agreement_and_frozen_kl_boundary(self):
        report = self.report(kl=[0.01] * 5, mismatches=[10, 0, 0, 0, 0])
        self.assertEqual(report.scored_positions, 1000)
        self.assertEqual(report.matching_top1_positions, 990)
        self.assertTrue(report.numerical_capture_gate_passes)
        self.assertFalse(self.report(kl=[0.010001] * 5).numerical_capture_gate_passes)
        self.assertFalse(
            self.report(mismatches=[11, 0, 0, 0, 0]).numerical_capture_gate_passes
        )

    def test_unequal_prompt_lengths_weight_positions_not_prompt_means(self):
        report = self.report(counts=[100, 100, 100, 100, 600], kl=[0, 0, 0, 0, 0.02])
        self.assertAlmostEqual(report.mean_kl_reference_candidate, 0.012)
        self.assertFalse(report.numerical_capture_gate_passes)
        report = self.report(
            counts=[100, 100, 100, 100, 600], mismatches=[2, 2, 2, 2, 0]
        )
        self.assertAlmostEqual(report.top1_agreement, 0.992)
        self.assertTrue(report.numerical_capture_gate_passes)

    def test_missing_or_duplicate_cases_and_failed_reference_do_not_pass(self):
        self.assertFalse(self.report(counts=[100] * 5).numerical_capture_gate_passes)
        self.assertFalse(self.report(exact=False).numerical_capture_gate_passes)
        with self.assertRaises(ValidationError):
            self.report(counts=[250] * 4)
        report = self.report()
        with self.assertRaisesRegex(ValidationError, "duplicate numerical case"):
            CorpusComparison(
                corpus_sha256=report.corpus_sha256,
                thresholds=report.thresholds,
                cases=[report.cases[0]] * 5,
            )


if __name__ == "__main__":
    unittest.main()
