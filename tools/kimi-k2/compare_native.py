"""Compare captured native K2 layer0 boundaries with the frozen CPU reference."""

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
    reference = np.load(args.reference)
    metrics = {}
    for name in ("input_norm", "attention_output", "mlp_output", "output"):
        actual = np.fromfile(args.prefix + "." + name + ".f32", dtype="<f4").reshape(
            reference[name].shape
        )
        diff = actual - reference[name]
        metrics[name] = {
            "max_abs": float(abs(diff).max()),
            "relative_rms": float(
                np.linalg.norm(diff) / np.linalg.norm(reference[name])
            ),
        }
        if (
            not np.isfinite(actual).all()
            or metrics[name]["max_abs"] > 1e-3
            or metrics[name]["relative_rms"] > 1e-4
        ):
            raise ValueError(f"{name} differs: {metrics[name]}")
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))
