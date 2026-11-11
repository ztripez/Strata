"""K5 real four-layer reference and narrow layer2/3 fixture acquisition."""

import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from compare_upstream import upstream
from cpu_reference import SliceWeights
from moe_reference import Fixture
from read_slices import fetch
from sequence_reference import Layer

root = Path("/mnt/work/kimi-k2")
out = root / "k5"
index = json.loads((root / "k1/gguf-index.json").read_text())
config = json.loads((root / "k0/config.json").read_text())
ns = upstream(root / "k0/modeling_deepseek.py")
torch.set_num_threads(4)
with torch.no_grad():
    x = torch.from_numpy(np.load(root / "k3/token-input.npz")["embeddings"][:4])
    saved = {"embeddings": x.numpy().copy()}
    for layer in range(4):
        if layer < 2:
            weights = SliceWeights(root / ("k1/slices" if layer == 0 else "k3"))
            experts = None
            if layer == 1:
                dest = out / "experts1"
                dest.mkdir(exist_ok=True)
                for src in (root / "k3/moe-slices").glob("*.bin"):
                    target = dest / src.name
                    if not target.exists():
                        target.symlink_to(src)
                experts = Fixture(root / "k1/gguf-index.json", dest)
        else:
            dest = out / f"layer{layer}"
            dest.mkdir(exist_ok=True)
            manifest = []
            for t in index["tensors"]:
                if not (
                    t["name"].startswith(f"blk.{layer}.attn_")
                    or t["name"] == f"blk.{layer}.ffn_norm.weight"
                ):
                    continue
                kind = gguf.GGMLQuantizationType(t["type"])
                block, byte = gguf.GGML_QUANT_SIZES[kind]
                size = t["elements"] // block * byte
                path = dest / (t["name"] + ".bin")
                print("tensor", t["name"], size, flush=True)
                if not path.exists():
                    path.write_bytes(fetch(t["absolute_offset"], size))
                raw = path.read_bytes()
                if len(raw) != size:
                    raise ValueError("short tensor")
                manifest.append(
                    {
                        "tensor": t["name"],
                        "shape": t["shape"],
                        "type": kind.name,
                        "offset": t["absolute_offset"],
                        "bytes": size,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
            weights = SliceWeights(dest)
            experts = Fixture(
                root / "k1/gguf-index.json", out / f"experts{layer}", layer=layer
            )
        model = Layer(layer, weights, config, ns, experts)
        print("upstream layer", layer, flush=True)
        x = model(x, official=True)
        saved[f"layer{layer}"] = x.numpy().copy()
        if layer:
            saved[f"ids{layer}"] = model.last_ids.numpy().copy()
        np.savez(out / "four-layer-reference.npz", **saved)
        if experts is not None:
            for role in ("gate", "up", "down"):
                experts(f"ffn_{role}_exps.weight", 0)
        del model
print("Four complete K2 layers finished on four actual token embeddings", flush=True)
