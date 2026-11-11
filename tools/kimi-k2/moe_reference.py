"""K2 MoE comparison on real selected GGUF experts using actual upstream modules."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from compare_upstream import upstream
from read_slices import fetch


class Fixture:
    """Fetch named reference slices for selected experts only; preserve raw bytes."""

    def __init__(self, index, destination, layer=1):
        self.layer = layer
        self.index = {
            t["name"]: t for t in json.loads(Path(index).read_text())["tensors"]
        }
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=True)
        manifest_path = self.destination / "manifest.json"
        self.manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        )

    def __call__(self, name, expert=None):
        t = self.index[f"blk.{self.layer}." + name]
        kind = gguf.GGMLQuantizationType(t["type"])
        block, byte = gguf.GGML_QUANT_SIZES[kind]
        shape = t["shape"][:2] if expert is not None else t["shape"]
        size = int(np.prod(shape)) // block * byte
        offset = t["absolute_offset"] + (expert * size if expert is not None else 0)
        path = self.destination / (
            name + (f".expert{expert}" if expert is not None else "") + ".bin"
        )
        if not path.exists():
            path.write_bytes(fetch(offset, size))
        raw = path.read_bytes()
        if len(raw) != size:
            raise ValueError("wrong slice length")
        self.manifest = [
            item
            for item in self.manifest
            if (item["tensor"], item["expert"]) != (name, expert)
        ]
        self.manifest.append(
            {
                "tensor": name,
                "expert": expert,
                "shape": shape,
                "type": kind.name,
                "offset": offset,
                "bytes": size,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        (self.destination / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2) + "\n"
        )
        out = gguf.dequantize(np.frombuffer(raw, dtype=np.uint8), kind).reshape(
            tuple(reversed(shape))
        )
        if not np.isfinite(out).all():
            raise ValueError("nonfinite decoded weight")
        return out


def run(fixture, config, source, destination):
    """Compare native router order, expert boundaries, shared MLP and weighted sum."""
    cfg = SimpleNamespace(**json.loads(Path(config).read_text()))
    ns = upstream(source)
    arrays = {}
    metrics = {}

    def check(name, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(name + " nonfinite")
        error = a - b
        rel = float(np.linalg.norm(error) / max(float(np.linalg.norm(b)), 1e-30))
        absolute = float(abs(error).max())
        metrics[name] = {"max_abs": absolute, "relative_rms": rel}
        if rel > 1e-4 or absolute > 1e-3:
            raise ValueError(f"{name} violates frozen thresholds: {metrics[name]}")
        arrays[name] = a
        arrays[name + "_upstream"] = b

    x = np.random.default_rng(193).normal(size=(3, 7168)).astype(np.float32)
    x *= np.array([0.003, 0.03, 0.3], dtype=np.float32)[:, None]
    arrays["input"] = x
    gatew = fixture("ffn_gate_inp.weight")
    bias = fixture("exp_probs_b.bias")
    gate = ns["MoEGate"](cfg).eval()
    gate.load_state_dict(
        {
            "weight": torch.from_numpy(gatew),
            "e_score_correction_bias": torch.from_numpy(bias),
        },
        strict=True,
        assign=True,
    )
    ids_t, weights_t = gate(torch.from_numpy(x)[None])
    ids = ids_t.numpy()
    arrays["expert_ids"] = ids
    logits = x @ gatew.T
    scores = 1 / (1 + np.exp(-logits))
    choice = scores + bias
    independent_ids = np.argsort(-choice, axis=-1, kind="stable")[:, :8]
    np.testing.assert_array_equal(
        np.sort(ids, axis=-1), np.sort(independent_ids, axis=-1)
    )
    sorted_scores = np.sort(choice, axis=-1)
    margins = sorted_scores[:, -8] - sorted_scores[:, -9]
    if np.any(margins <= 0):
        raise ValueError("selection tie at top8 boundary requires explicit handling")
    arrays["top8_boundary_margin"] = margins
    # Preserve upstream's unsorted native order; weights still independently recomputed.
    weights = np.take_along_axis(scores, ids, axis=-1)
    weights = (
        weights / (weights.sum(-1, keepdims=True) + 1e-20) * cfg.routed_scaling_factor
    )
    check(
        "router_logits",
        logits,
        F.linear(torch.from_numpy(x), torch.from_numpy(gatew)).numpy(),
    )
    check("routing_weights", weights, weights_t.numpy())
    if not np.any(np.ptp(weights, axis=-1) > 0.001):
        raise ValueError("no deliberately nonuniform routing sample")
    outputs = np.zeros((3, 8, 7168), dtype=np.float32)
    ref = np.zeros_like(outputs)
    for expert in np.unique(ids):
        print("K2 expert", expert, flush=True)
        rows, slots = np.where(ids == expert)
        z = x[rows]
        matrices = {
            name: fixture("ffn_" + name + "_exps.weight", int(expert))
            for name in ("gate", "up", "down")
        }
        g = z @ matrices["gate"].T
        u = z @ matrices["up"].T
        h = g / (1 + np.exp(-g)) * u
        out = h @ matrices["down"].T
        module = ns["DeepseekV3MLP"](cfg, intermediate_size=2048).eval()
        module.load_state_dict(
            {
                name + "_proj.weight": torch.from_numpy(a)
                for name, a in matrices.items()
            },
            strict=True,
            assign=True,
        )
        tx = torch.from_numpy(z)
        tg = module.gate_proj(tx)
        tu = module.up_proj(tx)
        check(f"expert_{expert}_gate", g, tg.numpy())
        check(f"expert_{expert}_up", u, tu.numpy())
        check(f"expert_{expert}_swiglu", h, (F.silu(tg) * tu).numpy())
        oracle = module(tx).numpy()
        check(f"expert_{expert}_output", out, oracle)
        outputs[rows, slots] = out
        ref[rows, slots] = oracle
    arrays["per_expert_outputs"] = outputs
    arrays["per_expert_outputs_upstream"] = ref
    reduced = np.sum(outputs * weights[..., None], axis=1)
    ref_reduced = (torch.from_numpy(ref) * weights_t[..., None]).sum(1).numpy()
    check("weighted_reduction", reduced, ref_reduced)
    matrices = {
        name: fixture("ffn_" + name + "_shexp.weight")
        for name in ("gate", "up", "down")
    }
    g = x @ matrices["gate"].T
    u = x @ matrices["up"].T
    shared = (g / (1 + np.exp(-g)) * u) @ matrices["down"].T
    module = ns["DeepseekV3MLP"](cfg, intermediate_size=2048).eval()
    module.load_state_dict(
        {name + "_proj.weight": torch.from_numpy(a) for name, a in matrices.items()},
        strict=True,
        assign=True,
    )
    oracle_shared = module(torch.from_numpy(x)).numpy()
    check("shared_output", shared, oracle_shared)
    check("output", reduced + shared, ref_reduced + oracle_shared)
    np.savez(destination / "moe.npz", **arrays)
    (destination / "moe-comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "ids": ids.tolist(),
                "final": metrics["output"],
                "routing_weights": weights.tolist(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("index", "config", "source", "destination"):
        parser.add_argument(arg)
    args = parser.parse_args()
    dest = Path(args.destination)
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with torch.no_grad():
        run(Fixture(args.index, dest / "slices"), args.config, args.source, dest)
