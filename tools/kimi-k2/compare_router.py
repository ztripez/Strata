"""Compare native K2 GPU expert sets and aligned weights to frozen upstream results."""

import argparse
import json
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("prefix")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    r = np.load(args.reference)
    ids = np.fromfile(args.prefix + ".ids.i32", dtype="<i4").reshape(3, 8)
    weights = np.fromfile(args.prefix + ".weights.f32", dtype="<f4").reshape(3, 8)
    logits = np.fromfile(args.prefix + ".logits.f32", dtype="<f4").reshape(3, 384)
    expected_ids = r["expert_ids"]
    expected_weights = r["routing_weights_upstream"]
    np.testing.assert_array_equal(np.sort(ids, axis=1), np.sort(expected_ids, axis=1))
    actual_order = np.argsort(ids, axis=1)
    expected_order = np.argsort(expected_ids, axis=1)
    aligned = np.take_along_axis(weights, actual_order, axis=1)
    expected = np.take_along_axis(expected_weights, expected_order, axis=1)
    difference = aligned - expected
    logit_error = logits - r["router_logits_upstream"]
    metrics = {
        "expert_ids": ids.tolist(),
        "expert_sets_exact": True,
        "max_weight_abs": float(abs(difference).max()),
        "weight_relative_rms": float(
            np.linalg.norm(difference) / np.linalg.norm(expected)
        ),
        "max_logit_abs": float(abs(logit_error).max()),
        "weights": weights.tolist(),
    }
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(logits).all()
        or metrics["max_weight_abs"] > 1e-3
        or metrics["weight_relative_rms"] > 1e-4
    ):
        raise ValueError(f"routing outside bounds: {metrics}")
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))
