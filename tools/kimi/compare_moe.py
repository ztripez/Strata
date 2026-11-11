"""Compare preserved first-MoE boundaries with an independent PyTorch formulation."""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from moe import Fixture


def compare(fixture, receipt):
    """Recompute routing and all selected experts; assert each preserved boundary."""

    def w(name):
        return torch.from_numpy(fixture(name))

    def linear(x, name):
        return F.linear(x, w(name + ".weight"))

    def activation(g, u):
        return 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25)

    def expert_matrix(base):
        packed = w(base + "weight_packed").to(torch.int64)
        # Build each column from its byte position rather than interleave arrays.
        col = torch.arange(packed.shape[1] * 2)
        code = (packed[:, col // 2] >> ((col % 2) * 4)) & 15
        magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])[code & 7]
        signed = torch.where((code & 8) != 0, -magnitude, magnitude)
        scale = w(base + "weight_scale").float()[:, col // 32]
        return torch.ldexp(signed, (scale - 127).int())

    metrics = {}

    def check(name, actual):
        expected = receipt[name]
        actual = actual.numpy()
        metrics[name] = {
            "max_abs": float(np.max(np.abs(actual - expected))),
            "relative_rms": float(
                np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)
            ),
        }
        np.testing.assert_allclose(
            actual, expected, rtol=0.002, atol=0.002, err_msg=name
        )

    x = torch.from_numpy(receipt["input"])
    logits = linear(x, "gate")
    check("logits", logits)
    scores = logits.sigmoid()
    ids = (scores + w("gate.e_score_correction_bias")).topk(16, dim=-1).indices
    np.testing.assert_array_equal(
        np.sort(ids.numpy(), axis=-1), np.sort(receipt["ids"], axis=-1)
    )
    # Align equivalent unordered top-k results for per-expert boundary comparison.
    ids = torch.from_numpy(receipt["ids"])
    weights = scores.gather(1, ids)
    weights /= weights.sum(-1, keepdim=True)
    check("weights", weights)
    latent = linear(x, "routed_expert_down_proj")
    check("latent", latent)
    outputs = torch.zeros(len(x), 16, 3584)
    for expert in ids.unique():
        rows, slots = torch.where(ids == expert)
        base = f"experts.{expert.item()}."
        gate = F.linear(latent[rows], expert_matrix(base + "w1."))
        up = F.linear(latent[rows], expert_matrix(base + "w3."))
        outputs[rows, slots] = F.linear(
            activation(gate, up), expert_matrix(base + "w2.")
        )
    check("expert_outputs", outputs)
    reduced = (outputs * weights[..., None]).sum(1)
    check("reduced", reduced)
    normalized = F.rms_norm(reduced, (3584,), w("routed_expert_norm.weight"), 1e-5)
    check("normalized", normalized)
    routed = linear(normalized, "routed_expert_up_proj")
    check("routed", routed)
    gate = linear(x, "shared_experts.gate_proj")
    up = linear(x, "shared_experts.up_proj")
    shared = linear(activation(gate, up), "shared_experts.down_proj")
    check("shared", shared)
    check("output", routed + shared)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures")
    parser.add_argument("receipt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.no_grad():
        metrics = compare(Fixture(args.fixtures), np.load(args.receipt))
    with open(args.output, "w") as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics))
