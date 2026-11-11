"""Fetch real K2 token embedding rows and layer1 attention slices for K3."""

import hashlib
import json
import pathlib

import gguf
import numpy as np
from read_slices import fetch

p = pathlib.Path("/mnt/work/kimi-k2")
out = p / "k3"
index = json.loads((p / "k1/gguf-index.json").read_text())
tok = json.loads((p / "k2/tokenizer-bringup.json").read_text())
ids = tok["ids"][:24]
t = next(t for t in index["tensors"] if t["name"] == "token_embd.weight")
kind = gguf.GGMLQuantizationType(t["type"])
block, byte = gguf.GGML_QUANT_SIZES[kind]
size = 7168 // block * byte
embeddings = []
manifest = []
for token in ids:
    path = out / f"embedding-{token}.bin"
    if not path.exists():
        path.write_bytes(fetch(t["absolute_offset"] + token * size, size))
    raw = path.read_bytes()
    embeddings.append(gguf.dequantize(np.frombuffer(raw, dtype=np.uint8), kind))
    manifest.append(
        {
            "token_id": token,
            "bytes": size,
            "offset": t["absolute_offset"] + token * size,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
np.savez(out / "token-input.npz", ids=np.array(ids), embeddings=np.stack(embeddings))
(out / "embedding-manifest.json").write_text(json.dumps(manifest, indent=2))
# Layer1 attention and norms only. Routed expert fixture weights remain separate.
entries = []
for t in index["tensors"]:
    if not t["name"].startswith("blk.1.attn_") and t["name"] != "blk.1.ffn_norm.weight":
        continue
    kind = gguf.GGMLQuantizationType(t["type"])
    block, byte = gguf.GGML_QUANT_SIZES[kind]
    size = t["elements"] // block * byte
    path = out / (t["name"] + ".bin")
    if not path.exists():
        path.write_bytes(fetch(t["absolute_offset"], size))
    entries.append(
        {
            "tensor": t["name"],
            "type": kind.name,
            "shape": t["shape"],
            "offset": t["absolute_offset"],
            "bytes": size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
(out / "manifest.json").write_text(json.dumps(entries, indent=2))
print("fetched 24 real token embedding rows and layer1 attention/norms")
