"""Compare native K2 layers0/1 on actual embedding rows to the K3 upstream reference."""

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
    metrics = {}
    for layer in (0, 1):
        a = np.fromfile(args.prefix + f".layer{layer}.output.f32", dtype="<f4").reshape(
            3, 7168
        )
        b = r[f"upstream_layer{layer}"][:3]
        delta = a - b
        metrics[f"layer{layer}"] = {
            "max_abs": float(abs(delta).max()),
            "relative_rms": float(np.linalg.norm(delta) / np.linalg.norm(b)),
        }
        for i in range(3):
            d = a[i] - b[i]
            absolute = float(abs(d).max())
            relative = float(np.linalg.norm(d) / np.linalg.norm(b[i]))
            if not np.isfinite(a[i]).all() or absolute > 1e-3 or relative > 1e-4:
                raise ValueError(
                    f"layer{layer} input{i}: abs={absolute} relative={relative}"
                )
    ids = np.fromfile(args.prefix + ".layer1.moe.ids.i32", dtype="<i4").reshape(3, 8)
    np.testing.assert_array_equal(
        np.sort(ids, axis=1), np.sort(r["upstream_expert_ids"][:3], axis=1)
    )
    result = {
        "token_ids": r["token_ids"][:3].tolist(),
        "expert_sets_exact": True,
        "expert_ids": ids.tolist(),
        "boundaries": metrics,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
