"""Real-weight first MoE block reference; fixture downloads only, no runtime cache."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from inspect_weights import read_header
from layer0 import norm
from operators import situ

URL = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/model-00002-of-000096.safetensors"
PREFIX = "language_model.model.layers.1.block_sparse_moe."


class Fixture:
    """Fetch named fixture tensors with HTTP ranges; preserve original packed bytes."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        header_path = self.directory / "header.json"
        if not header_path.exists():
            header_path.write_text(json.dumps(read_header(URL)))
        self.header = json.loads(header_path.read_text())
        # Published safetensors header length is read separately, including padding.
        request = urllib.request.Request(
            URL + "?length=moe", headers={"Range": "bytes=0-7"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise ValueError("server ignored range")
            self.offset = 8 + int.from_bytes(response.read(8), "little")

    def __call__(self, name):
        item = self.header[PREFIX + name]
        path = self.directory / (name + ".bin")
        start, end = item["data_offsets"]
        if not path.exists():
            request = urllib.request.Request(
                URL + f"?tensor={name}",
                headers={
                    "Range": f"bytes={self.offset + start}-{self.offset + end - 1}"
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise ValueError("server ignored range")
                data = response.read(end - start + 1)
            if len(data) != end - start:
                raise ValueError("incomplete tensor response")
            path.write_bytes(data)
        raw = np.fromfile(
            path, dtype={"BF16": "<u2", "F32": "<f4", "U8": "u1"}[item["dtype"]]
        ).reshape(item["shape"])
        if item["dtype"] == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return raw


def decode(packed, scales):
    """Decode native E2M1 low/even nibbles and E8M0 scales in groups of 32."""
    table = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=np.float32,
    )
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    return table[codes] * np.repeat(
        np.exp2(scales.astype(np.float32) - 127), 32, axis=-1
    )


def forward(x, w):
    """Execute sigmoid routing, selected native experts, latent norm and shared MLP."""
    logits = x @ w("gate.weight").T
    scores = 1 / (1 + np.exp(-logits))
    choice = scores + w("gate.e_score_correction_bias")
    ids = np.argsort(-choice, axis=-1, kind="stable")[:, :16]
    weights = np.take_along_axis(scores, ids, axis=-1)
    weights /= weights.sum(-1, keepdims=True)
    latent = x @ w("routed_expert_down_proj.weight").T
    expert_outputs = np.zeros((len(x), 16, 3584), dtype=np.float32)
    for expert in np.unique(ids):
        print(f"expert {expert}", flush=True)

        def matrix(name, expert=expert):
            base = f"experts.{expert}.{name}."
            return decode(w(base + "weight_packed"), w(base + "weight_scale"))

        rows, slots = np.where(ids == expert)
        z = latent[rows]
        gate, up = z @ matrix("w1").T, z @ matrix("w3").T
        expert_outputs[rows, slots] = situ(gate, up) @ matrix("w2").T
    reduced = np.sum(expert_outputs * weights[..., None], axis=1)
    normalized = norm(reduced, w("routed_expert_norm.weight"))
    routed = normalized @ w("routed_expert_up_proj.weight").T
    shared_gate = x @ w("shared_experts.gate_proj.weight").T
    shared_up = x @ w("shared_experts.up_proj.weight").T
    shared = situ(shared_gate, shared_up) @ w("shared_experts.down_proj.weight").T
    return {
        "input": x,
        "logits": logits,
        "ids": ids,
        "weights": weights,
        "latent": latent,
        "expert_outputs": expert_outputs,
        "reduced": reduced,
        "normalized": normalized,
        "routed": routed,
        "shared": shared,
        "output": routed + shared,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    x = np.random.default_rng(42).normal(size=(1, 7168)).astype(np.float32)
    result = forward(x, Fixture(args.fixtures))
    if not np.isfinite(result["output"]).all():
        raise ValueError("nonfinite output")
    np.savez(args.output, **result)
    print(
        json.dumps(
            {
                "ids": result["ids"].tolist(),
                "output_rms": float(np.sqrt(np.mean(result["output"] ** 2))),
            }
        )
    )
