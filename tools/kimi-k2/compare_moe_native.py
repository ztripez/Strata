"""Compare the combined native K2 GPU-cache/CPU-pool MoE with the frozen K2 reference."""

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
    parts = np.fromfile(args.prefix + ".parts.f32", dtype="<f4").reshape(3, 8, 7168)
    np.testing.assert_array_equal(
        np.sort(ids, axis=1), np.sort(r["expert_ids"], axis=1)
    )
    expected_parts = np.empty_like(parts)
    expected_weights = np.empty_like(weights)
    for t in range(3):
        for k in range(8):
            slot = int(np.flatnonzero(r["expert_ids"][t] == ids[t, k])[0])
            expected_parts[t, k] = r["per_expert_outputs_upstream"][t, slot]
            expected_weights[t, k] = r["routing_weights_upstream"][t, slot]
    metrics = {}

    def check(name, a, b):
        delta = a - b
        absolute = float(abs(delta).max())
        relative = float(np.linalg.norm(delta) / np.linalg.norm(b))
        metrics[name] = {"max_abs": absolute, "relative_rms": relative}
        if not np.isfinite(a).all() or absolute > 1e-3 or relative > 1e-4:
            raise ValueError(f"{name}: {metrics[name]}")

    check("weights", weights, expected_weights)
    check("parts", parts, expected_parts)
    for name, key in [
        ("reduced", "weighted_reduction_upstream"),
        ("shared", "shared_output_upstream"),
        ("output", "output_upstream"),
    ]:
        actual = np.fromfile(args.prefix + "." + name + ".f32", dtype="<f4").reshape(
            3, 7168
        )
        check(name, actual, r[key])
        if name == "output":
            for t in range(3):
                check("output_input_" + str(t), actual[t], r[key][t])
    Path(args.output).write_text(
        json.dumps(
            {
                "expert_ids": ids.tolist(),
                "expert_sets_exact": True,
                "boundaries": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(metrics))
