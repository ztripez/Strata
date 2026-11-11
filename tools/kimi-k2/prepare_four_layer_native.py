"""Bind only captured real four-layer K2 reference tensors for native execution."""

import json
from pathlib import Path

import gguf
import numpy as np
from moe_reference import Fixture

root = Path("/mnt/work/kimi-k2")
p = root / "k5/native"
p.mkdir(exist_ok=True)
r = np.load(root / "k5/four-layer-reference.npz")
r["embeddings"].astype("<f4").tofile(p / "embeddings.f32")
index = {
    t["name"]: t
    for t in json.loads((root / "k1/gguf-index.json").read_text())["tensors"]
}
for layer in range(4):
    dest = p / f"layer{layer}"
    if layer == 0:
        if not dest.exists():
            dest.symlink_to(root / "k4/fixture", target_is_directory=True)
        continue
    src = root / ("k3" if layer == 1 else f"k5/layer{layer}")
    dest.mkdir(exist_ok=True)
    for item in json.loads((src / "manifest.json").read_text()):
        name = item["tensor"].removeprefix(f"blk.{layer}.").removesuffix(".weight")
        suffix = ".f32" if item["type"] == "F32" else ".quant"
        target = dest / (name + suffix)
        if not target.exists():
            target.symlink_to(src / (item["tensor"] + ".bin"))
        (dest / (name + ".type")).write_text(
            str(int(gguf.GGMLQuantizationType[item["type"]]))
        )
    rope = dest / "rope_freqs.f32"
    if not rope.exists():
        rope.symlink_to(root / "k4/fixture/rope_freqs.f32")
    experts = root / f"k5/experts{layer}"
    reference = Fixture(root / "k1/gguf-index.json", experts, layer=layer)
    for role in ("gate", "up", "down"):
        reference(f"ffn_{role}_exps.weight", 0)
    fixture = p / f"experts{layer}"
    fixture.mkdir(exist_ok=True)
    selected = {int(x) for x in r[f"ids{layer}"].flat} | {0}
    for source in experts.glob("*.bin"):
        if (
            ".expert" in source.name
            and int(source.name.split(".expert")[1].split(".")[0]) not in selected
        ):
            continue
        target = fixture / source.name
        if not target.exists():
            target.symlink_to(source)
    # Expert zero is the fixed existing GPU-cache resident; acquire it explicitly below if unselected.
    for role in (
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
    ):
        (fixture / (role + ".type")).write_text(
            str(index[f"blk.{layer}.{role}.weight"]["type"])
        )
    for original, name in [
        ("ffn_gate_inp.weight.bin", "gate.f32"),
        ("exp_probs_b.bias.bin", "bias.f32"),
    ]:
        target = fixture / name
        if not target.exists():
            target.symlink_to(experts / original)
print(p)
