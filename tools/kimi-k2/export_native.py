"""Export K2 layer0 decoded fixture boundaries for the first native GPU run."""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from cpu_reference import SliceWeights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "k4/fixture"
    out.mkdir(parents=True, exist_ok=True)
    w = SliceWeights(root / "k1/slices")
    index = []
    for name in (
        "attn_norm",
        "attn_q_a",
        "attn_q_a_norm",
        "attn_q_b",
        "attn_kv_a_mqa",
        "attn_kv_a_norm",
        "attn_output",
        "ffn_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ):
        data = w("blk.0." + name + ".weight")
        entry = w.manifest["blk.0." + name + ".weight"]
        if entry["type"] in ("Q4_K", "IQ4_XS"):
            import gguf

            shutil.copyfile(
                root / "k1/slices" / ("blk.0." + name + ".weight.bin"),
                out / (name + ".quant"),
            )
            (out / (name + ".type")).write_text(
                str(int(gguf.GGMLQuantizationType[entry["type"]]))
            )
        data.astype("<f4").tofile(out / (name + ".f32"))
        index.append({"name": name, "shape": list(data.shape)})
    for name in ("attn_k_b", "attn_v_b"):
        shutil.copyfile(
            root / "k1/slices" / ("blk.0." + name + ".weight.bin"),
            out / (name + ".quant"),
        )
    kb = w("blk.0.attn_k_b.weight").transpose(0, 2, 1)
    vb = w("blk.0.attn_v_b.weight")
    np.concatenate([kb, vb], axis=1).reshape(16384, 512).astype("<f4").tofile(
        out / "kv_b.f32"
    )
    receipt = np.load(root / "k2/layer0.npz")
    receipt["input"].astype("<f4").tofile(out / "input.f32")
    cfg = json.loads((root / "k0/config.json").read_text())
    d = 64
    sc = cfg["rope_scaling"]
    correction = (
        d
        * math.log(
            sc["original_max_position_embeddings"] / (sc["beta_fast"] * 2 * math.pi)
        )
        / (2 * math.log(cfg["rope_theta"]))
    )
    lo = max(math.floor(correction), 0)
    hi = min(math.ceil(correction), d - 1)
    if lo == hi:
        hi += 0.001
    ramp = np.clip((np.arange(d // 2, dtype=np.float32) - lo) / (hi - lo), 0, 1)
    freq = 1 / np.power(
        np.float32(cfg["rope_theta"]), np.arange(0, d, 2, dtype=np.float32) / d
    )
    (freq * ((1 - ramp) + ramp / sc["factor"])).astype("<f4").tofile(
        out / "rope_freqs.f32"
    )
    (out / "manifest.json").write_text(json.dumps(index, indent=2))
    expert = root / "k4/expert0"
    expert.mkdir(exist_ok=True)
    for name, kind in (("gate", 21), ("up", 21), ("down", 23)):
        shutil.copyfile(
            root / "k1/slices" / ("blk.1.ffn_" + name + "_exps.weight.bin"),
            expert / (name + ".quant"),
        )
        (expert / (name + ".type")).write_text(str(kind))
    x = np.load(root / "k2/moe.npz")["input"]
    x.astype("<f4").tofile(expert / "input.f32")
    gate = x @ w("blk.1.ffn_gate_exps.weight").T
    up = x @ w("blk.1.ffn_up_exps.weight").T
    expected = (gate / (1 + np.exp(-gate)) * up) @ w("blk.1.ffn_down_exps.weight").T
    np.save(root / "k4/expert0-reference.npy", expected)
    router = root / "k4/router"
    router.mkdir(exist_ok=True)
    shutil.copyfile(root / "k2/slices/ffn_gate_inp.weight.bin", router / "gate.f32")
    shutil.copyfile(root / "k2/slices/exp_probs_b.bias.bin", router / "bias.f32")
    x.astype("<f4").tofile(router / "input.f32")
    moe = root / "k4/moe-fixture"
    moe.mkdir(exist_ok=True)
    for source in (root / "k2/slices").glob("*.bin"):
        target = moe / source.name
        if not target.exists():
            target.symlink_to(source)
    for source in router.glob("*.f32"):
        target = moe / source.name
        if not target.exists():
            target.symlink_to(source)
    print(out)
