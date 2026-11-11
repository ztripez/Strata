"""Fetch and decode pinned K2 GGUF tensor slices for K1, without staging the pack."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import gguf
import numpy as np

REPO = "unsloth/Kimi-K2-Instruct-GGUF"
REVISION = "23bf90d8196ddfcb3328a6f60c99a9bdb5c1dadf"
SHARD = "UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00001-of-00009.gguf"


def fetch(start, size):
    """Read an exact pinned HTTP byte range, refusing ignored or short responses."""
    request = urllib.request.Request(
        f"https://huggingface.co/{REPO}/resolve/{REVISION}/{SHARD}?slice={start}-{size}",
        headers={"Range": f"bytes={start}-{start + size - 1}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise ValueError("server ignored byte range")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("incomplete range response")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    index = json.loads(args.index.read_text())
    selected = [
        t
        for t in index["tensors"]
        if t["name"].startswith("blk.0.")
        or t["name"]
        in (
            "blk.1.ffn_gate_exps.weight",
            "blk.1.ffn_up_exps.weight",
            "blk.1.ffn_down_exps.weight",
        )
    ]
    manifest = []
    for tensor in selected:
        kind = gguf.GGMLQuantizationType(tensor["type"])
        block, byte = gguf.GGML_QUANT_SIZES[kind]
        # Read the complete dense/MLA layer and exactly expert 0 of each MoE tensor.
        shape = tensor["shape"][:2] if "_exps." in tensor["name"] else tensor["shape"]
        elements = int(np.prod(shape))
        size = elements // block * byte
        path = args.destination / (tensor["name"] + ".bin")
        if not path.exists():
            path.write_bytes(fetch(tensor["absolute_offset"], size))
        data = path.read_bytes()
        if len(data) != size:
            raise ValueError("wrong saved slice length")
        packed = np.frombuffer(data, dtype=np.uint8)
        decoded = gguf.dequantize(packed, kind).reshape(tuple(reversed(shape)))
        if not np.isfinite(decoded).all():
            raise ValueError("nonfinite decoded weights")
        # Dot against two independently accumulated activation vectors, bounded rows.
        rows = decoded.reshape(-1, shape[0])[:8]
        x = np.random.default_rng(42).normal(size=shape[0]).astype(np.float32)
        a = rows @ x
        b = np.array(
            [np.sum(row.astype(np.float64) * x.astype(np.float64)) for row in rows]
        )
        if not np.allclose(a, b, rtol=1e-4, atol=1e-3):
            raise ValueError("CPU accumulation disagreement")
        entry = {
            "tensor": tensor["name"],
            "type": kind.name,
            "shape": shape,
            "offset": tensor["absolute_offset"],
            "bytes": size,
            "sha256": hashlib.sha256(data).hexdigest(),
            "finite": True,
            "dot_max_abs": float(np.max(abs(a - b))),
        }
        manifest.append(entry)
        print(json.dumps(entry), flush=True)
        (args.destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    # A fresh network read proves byte stability for an actual MLA tensor slice.
    first = manifest[0]
    if (
        hashlib.sha256(fetch(first["offset"], first["bytes"])).hexdigest()
        != first["sha256"]
    ):
        raise ValueError("repeated pinned read changed")
    print(
        "PASS: complete layer-0 tensors and one layer-1 expert decoded; repeated MLA read matched"
    )
