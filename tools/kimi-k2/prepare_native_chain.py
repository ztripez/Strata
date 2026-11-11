"""Bind original K2 layer1 weights and three actual embedding rows for the native chain."""

import json
import shutil
from pathlib import Path

import gguf
import numpy as np

root = Path("/mnt/work/kimi-k2")
p = root / "k4/native-chain"
p.mkdir(exist_ok=True)
(p / "layer1").mkdir(exist_ok=True)
(p / "experts").mkdir(exist_ok=True)
r = np.load(root / "k3/two-layer.npz")
r["embeddings"][:3].astype("<f4").tofile(p / "embeddings.f32")
(p / "input-manifest.json").write_text(
    json.dumps(
        {
            "token_ids": r["token_ids"][:3].tolist(),
            "shape": [3, 7168],
            "source": "pinned K2 tokenizer/chat-template and original quantized embedding rows",
        },
        indent=2,
    )
)
for t in json.loads((root / "k3/manifest.json").read_text()):
    name = t["tensor"].removeprefix("blk.1.").removesuffix(".weight")
    src = root / "k3" / (t["tensor"] + ".bin")
    if t["type"] == "F32":
        shutil.copyfile(src, p / "layer1" / (name + ".f32"))
    else:
        shutil.copyfile(src, p / "layer1" / (name + ".quant"))
        (p / "layer1" / (name + ".type")).write_text(
            str(int(gguf.GGMLQuantizationType[t["type"]]))
        )
shutil.copyfile(root / "k4/fixture/rope_freqs.f32", p / "layer1/rope_freqs.f32")
for expert in np.unique(r["prefill_expert_ids"][:3]):
    for role in ("gate", "up", "down"):
        name = f"ffn_{role}_exps.weight.expert{expert}.bin"
        src = root / "k3/moe-slices" / name
        dest = p / "experts" / name
        if not dest.exists():
            dest.symlink_to(src)
for name in (
    "ffn_gate_shexp.weight.bin",
    "ffn_up_shexp.weight.bin",
    "ffn_down_shexp.weight.bin",
):
    src = root / "k3/moe-slices" / name
    dest = p / "experts" / name
    if not dest.exists():
        dest.symlink_to(src)
for name in ("gate.f32", "bias.f32"):
    src = root / "k4/router" / name
    dest = p / "experts" / name
    if not dest.exists():
        dest.symlink_to(src)
print(p)
