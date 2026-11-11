"""Compare K5 native layer outputs and router sets against frozen K2 arithmetic."""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("prefix")
parser.add_argument("mode", choices=("prefill", "incremental"))
args = parser.parse_args()
r = np.load("/mnt/work/kimi-k2/k5/four-layer-reference.npz")
metrics = {}
for layer in range(4):
    a = np.fromfile(f"{args.prefix}.layer{layer}.output.f32", dtype="<f4").reshape(
        4, 7168
    )
    b = r[f"layer{layer}"]
    d = a - b
    metrics[f"layer{layer}"] = {
        "max_abs": float(abs(d).max()),
        "relative_rms": float(np.linalg.norm(d) / np.linalg.norm(b)),
    }
    for t in range(4):
        if (
            not np.isfinite(a[t]).all()
            or abs(d[t]).max() > 1e-3
            or np.linalg.norm(d[t]) / np.linalg.norm(b[t]) > 1e-4
        ):
            raise ValueError(f"layer{layer} position{t} exceeds unchanged K2 bounds")
    if layer:
        starts = (0,) if args.mode == "prefill" else (0, 2, 3)
        ids = np.concatenate(
            [
                np.fromfile(
                    f"{args.prefix}.position{start}.layer{layer}.moe.ids.i32",
                    dtype="<i4",
                ).reshape(-1, 8)
                for start in starts
            ]
        )
        np.testing.assert_array_equal(
            np.sort(ids, axis=1), np.sort(r[f"ids{layer}"], axis=1)
        )
result = {
    "boundaries": metrics,
    "expert_sets_exact": True,
    "mode": args.mode,
    "positions": 4,
    "layers": 4,
}
Path(args.prefix + ".comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
