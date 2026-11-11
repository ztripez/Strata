"""Compare K6's actual two-owner layer boundaries and native routed expert IDs."""

import json
from pathlib import Path

import numpy as np

root = Path("/mnt/work/kimi-k2")
prefix = root / "k6/split-prefix"
suffix = root / "k6/split-suffix-capture/kimi2-split-suffix"
reference = np.load(root / "k5/four-layer-reference.npz")
plan = json.loads((root / "k6/split-plan.json").read_text())
metrics = {}
for layer in range(4):
    owner = prefix if layer < 2 else suffix
    a = np.fromfile(str(owner) + f".layer{layer}.output.f32", dtype="<f4").reshape(
        4, 7168
    )
    b = reference[f"layer{layer}"]
    d = a - b
    metrics[f"layer{layer}"] = {
        "owner": "lucy/CUDA" if layer < 2 else "astor/HIP",
        "max_abs": float(abs(d).max()),
        "relative_rms": float(np.linalg.norm(d) / np.linalg.norm(b)),
    }
    for position in range(4):
        if (
            not np.isfinite(a[position]).all()
            or abs(d[position]).max() > plan["max_abs"]
            or np.linalg.norm(d[position]) / np.linalg.norm(b[position])
            > plan["relative_rms"]
        ):
            raise ValueError(f"layer{layer} position{position} exceeds frozen bounds")
    if layer:
        ids = np.concatenate(
            [
                np.fromfile(
                    str(owner) + f".position{p}.layer{layer}.moe.ids.i32", dtype="<i4"
                ).reshape(-1, 8)
                for p in (0, 2, 3)
            ]
        )
        np.testing.assert_array_equal(
            np.sort(ids, axis=1), np.sort(reference[f"ids{layer}"], axis=1)
        )
returned = np.fromfile(str(prefix) + ".layer3.output.f32", dtype="<f4")
actual = np.fromfile(str(suffix) + ".layer3.output.f32", dtype="<f4")
np.testing.assert_array_equal(returned, actual)
result = {
    "status": "PASS",
    "scope": "K6 real partial model: four layers, four actual token embeddings, windows [2,1,1]",
    "plan": plan,
    "expert_sets_exact": True,
    "returned_suffix_bytes_equal": True,
    "boundaries": metrics,
}
(root / "k6/split-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
