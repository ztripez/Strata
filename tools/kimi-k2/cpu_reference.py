"""K2-only float32 reference using pinned GGUF slices and standard MLA/SwiGLU."""

import argparse
import json
import math
from pathlib import Path

import gguf
import numpy as np


class SliceWeights:
    """Decode existing K1 slices; no model staging or execution cache."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = {
            x["tensor"]: x
            for x in json.loads((self.directory / "manifest.json").read_text())
        }

    def __call__(self, name):
        item = self.manifest[name]
        raw = np.fromfile(self.directory / (name + ".bin"), dtype=np.uint8)
        return gguf.dequantize(raw, gguf.GGMLQuantizationType[item["type"]]).reshape(
            tuple(reversed(item["shape"]))
        )


def rms(x, w):
    """K2 RMSNorm, epsilon 1e-6."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w


def rotary(x, positions, config):
    """K2 YaRN on interleaved query/key rotary channels."""
    d = x.shape[-1]
    sc = config["rope_scaling"]
    base = config["rope_theta"]

    def correction(rot):
        return (
            d
            * math.log(sc["original_max_position_embeddings"] / (rot * 2 * math.pi))
            / (2 * math.log(base))
        )

    lo = max(math.floor(correction(sc["beta_fast"])), 0)
    hi = min(math.ceil(correction(sc["beta_slow"])), d - 1)
    if lo == hi:
        hi += 0.001
    ramp = np.clip((np.arange(d // 2, dtype=np.float32) - lo) / (hi - lo), 0, 1)
    freq = 1 / np.power(np.float32(base), np.arange(0, d, 2, dtype=np.float32) / d)
    inv = freq * ((1 - ramp) + ramp / sc["factor"])
    angles = positions[:, None] * inv[None, :]
    cos = np.repeat(np.cos(angles)[:, None, :], x.shape[1], axis=1)
    sin = np.repeat(np.sin(angles)[:, None, :], x.shape[1], axis=1)
    even, odd = x[..., 0::2], x[..., 1::2]
    # Return the same split-half layout as upstream apply_rotary_pos_emb.
    return np.concatenate([even * cos - odd * sin, odd * cos + even * sin], axis=-1)


def layer0(x, w, config):
    """Return all MLA and dense MLP boundaries for a causal K2 layer-0 sequence."""
    result = {"input": x.copy()}

    def keep(name, value):
        result[name] = value
        return value

    def weight(name):
        return w("blk.0." + name + ".weight")

    n = keep("input_norm", rms(x, weight("attn_norm")))
    qa = keep("q_a", n @ weight("attn_q_a").T)
    qan = keep("q_a_norm", rms(qa, weight("attn_q_a_norm")))
    q = keep("q", (qan @ weight("attn_q_b").T).reshape(len(x), 64, 192))
    kv = keep("kv_a", n @ weight("attn_kv_a_mqa").T)
    latent = keep("kv_norm", rms(kv[:, :512], weight("attn_kv_a_norm")))
    # GGUF absorption layout: k_b [heads, latent, nope], v_b [heads, value, latent].
    keys = np.einsum("tl,hlk->thk", latent, weight("attn_k_b"))
    values = keep("values", np.einsum("tl,hvl->thv", latent, weight("attn_v_b")))
    positions = np.arange(len(x), dtype=np.float32)
    qp = rotary(q[:, :, 128:], positions, config)
    kp = rotary(kv[:, None, 512:], positions, config)
    queries = keep("queries", np.concatenate([q[:, :, :128], qp], axis=-1))
    keys = keep(
        "keys", np.concatenate([keys, np.broadcast_to(kp, (len(x), 64, 64))], axis=-1)
    )
    scale = 192**-0.5 * (1 + 0.1 * math.log(config["rope_scaling"]["factor"])) ** 2
    scores = np.einsum("thd,shd->hts", queries, keys) * scale
    scores = np.where(
        np.triu(np.ones((len(x), len(x)), dtype=bool), 1)[None], -np.inf, scores
    )
    prob = np.exp(scores - scores.max(axis=-1, keepdims=True))
    prob /= prob.sum(axis=-1, keepdims=True)
    keep("attention_probs", prob)
    attended = keep("attention_values", np.einsum("hts,shv->thv", prob, values))
    attention = keep(
        "attention_output", attended.reshape(len(x), -1) @ weight("attn_output").T
    )
    residual = keep("attention_residual", x + attention)
    post = keep("post_norm", rms(residual, weight("ffn_norm")))
    gate = keep("gate", post @ weight("ffn_gate").T)
    up = keep("up", post @ weight("ffn_up").T)
    activated = keep("swiglu", (gate / (1 + np.exp(-gate))) * up)
    mlp = keep("mlp_output", activated @ weight("ffn_down").T)
    keep("output", residual + mlp)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slices")
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    x = np.random.default_rng(137).normal(size=(3, 7168)).astype(np.float32)
    x *= np.array([0.01, 0.1, 1], dtype=np.float32)[:, None]
    result = layer0(x, SliceWeights(args.slices), config)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("nonfinite boundary")
    np.savez(args.output, **result)
    print("finite K2 layer-0 output", result["output"].shape, flush=True)
