"""Compare the NumPy layer against a separate PyTorch operator formulation."""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from layer0 import Weights


def reference(x, weights):
    """Evaluate layer zero using PyTorch dense, grouped convolution and matmul ops."""

    def w(name):
        return torch.from_numpy(weights(name))

    def linear(x, name):
        return F.linear(x, w(name + ".weight"))

    def norm(x, name):
        return F.rms_norm(x, (x.shape[-1],), w(name + ".weight"), 1e-5)

    original = x.clone()
    x = norm(x, "input_layernorm")
    h, d = 96, 128
    qkv = []
    for name in ("q", "k", "v"):
        z = linear(x, "self_attn." + name + "_proj").T.unsqueeze(0)
        z = F.conv1d(
            F.pad(z, (3, 0)), w("self_attn." + name + "_conv1d.weight"), groups=h * d
        )
        qkv.append(F.silu(z).squeeze(0).T.reshape(-1, h, d))
    q, k, v = qkv
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    g = linear(linear(x, "self_attn.f_a_proj"), "self_attn.f_b_proj").reshape(-1, h, d)
    g = -5 * torch.sigmoid(
        (g + w("self_attn.dt_bias").reshape(h, d))
        * w("self_attn.A_log")[:h].exp()[None, :, None]
    )
    beta = linear(x, "self_attn.b_proj").sigmoid()
    state = torch.zeros(h, d, d)
    outputs = []
    for i in range(len(x)):
        state = torch.diag_embed(g[i].exp()) @ state
        error = v[i] - (k[i].unsqueeze(-2) @ state).squeeze(-2)
        state = state + beta[i, :, None, None] * k[i, :, :, None] @ error[:, None, :]
        outputs.append(((q[i, :, None, :] @ state) / d**0.5).squeeze(-2))
    out = norm(torch.stack(outputs), "self_attn.o_norm")
    out *= linear(x, "self_attn.g_proj").reshape_as(out).sigmoid()
    prefix = linear(out.flatten(1), "self_attn.o_proj")
    residuals = torch.stack([original, prefix], 1)
    scores = linear(norm(residuals, "mlp_res_norm"), "mlp_res_proj")
    x = (scores.softmax(1) * residuals).sum(1)
    x = norm(x, "post_attention_layernorm")
    gate, up = linear(x, "mlp.gate_proj"), linear(x, "mlp.up_proj")
    x = 4 * (gate / 4).tanh() * gate.sigmoid() * 25 * (up / 25).tanh()
    return prefix + linear(x, "mlp.down_proj")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights")
    parser.add_argument("result")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = np.load(args.result)
    with torch.no_grad():
        actual = reference(
            torch.from_numpy(result["input"]), Weights(args.weights)
        ).numpy()
    expected = result["output"]
    delta = actual - expected
    relative_rms = float(np.linalg.norm(delta) / np.linalg.norm(expected))
    print(
        json.dumps({"max_abs": float(abs(delta).max()), "relative_rms": relative_rms})
    )
    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0.002)
