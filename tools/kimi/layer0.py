"""Float32 CPU forward for Kimi K3's real first (dense, KDA) layer."""

import argparse
import json
import struct

import numpy as np
from operators import kda, situ


class Weights:
    """Read individual F32/BF16 tensors from a local safetensors shard."""

    def __init__(self, path):
        with open(path, "rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            self.header = json.loads(stream.read(size))
        self.path, self.offset = path, 8 + size

    def __call__(self, name):
        item = self.header["language_model.model.layers.0." + name]
        dtype = {"BF16": "<u2", "F32": "<f4"}[item["dtype"]]
        raw = np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=self.offset + item["data_offsets"][0],
            shape=tuple(item["shape"]),
        )
        if item["dtype"] == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return np.array(raw)


def norm(x, weight, eps=1e-5):
    """Apply weighted RMS normalization in float32."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def forward(x, w):
    """Run the first layer, returning its prefix sum and residual block."""
    original = x.copy()
    x = norm(x, w("input_layernorm.weight"))

    def linear(value, name):
        return value @ w(name + ".weight").T

    def att(name):
        return "self_attn." + name

    heads = w(att("b_proj.weight")).shape[0]
    dim = w(att("o_norm.weight")).size
    values = []
    for name in ("q", "k", "v"):
        projected = linear(x, att(name + "_proj"))
        kernel = w(att(name + "_conv1d.weight"))[:, 0, :]
        padded = np.pad(projected, ((3, 0), (0, 0)))
        conv = sum(padded[i : i + len(x)] * kernel[:, i] for i in range(4))
        conv = conv / (1 + np.exp(-conv))
        values.append(conv.reshape(len(x), heads, dim))
    q, k, v = values
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    a = w(att("A_log"))
    if a.size < heads or np.any(a[heads:] != 0):
        raise ValueError("A_log has missing heads or nonzero padding")
    g = linear(linear(x, att("f_a_proj")), att("f_b_proj")).reshape(len(x), heads, dim)
    g = (g + w(att("dt_bias")).reshape(heads, dim)) * np.exp(a[:heads])[None, :, None]
    g = -5 / (1 + np.exp(-g))
    beta = 1 / (1 + np.exp(-linear(x, att("b_proj"))))
    out, _ = kda(q, k, v, g, beta)
    out = norm(out, w(att("o_norm.weight")))
    out *= (1 / (1 + np.exp(-linear(x, att("g_proj"))))).reshape(out.shape)
    prefix = linear(out.reshape(len(x), -1), att("o_proj"))
    residuals = np.stack([original, prefix], axis=1)
    scores = np.sum(
        norm(residuals, w("mlp_res_norm.weight")) * w("mlp_res_proj.weight"), axis=-1
    )
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    x = np.sum(residuals * probs[..., None], axis=1)
    x = norm(x, w("post_attention_layernorm.weight"))
    x = situ(linear(x, "mlp.gate_proj"), linear(x, "mlp.up_proj"))
    return prefix + linear(x, "mlp.down_proj"), original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    w = Weights(args.weights)
    width = w("input_layernorm.weight").size
    x = np.random.default_rng(42).normal(size=(3, width)).astype(np.float32)
    out, block = forward(x, w)
    if not np.isfinite(out).all():
        raise ValueError("nonfinite layer output")
    np.savez(args.output, input=x, output=out, block=block)
    print(
        json.dumps(
            {
                "shape": list(out.shape),
                "rms": float(np.sqrt(np.mean(out * out))),
                "output": args.output,
            }
        )
    )
