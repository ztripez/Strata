"""K3: real K2 tokens through two layers with compressed MLA state continuity."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from compare_upstream import upstream
from cpu_reference import SliceWeights, rotary
from moe_reference import Fixture


class Layer:
    """Two-layer CPU reference adapter, with actual K2 compressed KV tensors."""

    def __init__(self, index, weights, config, namespace, experts=None):
        self.index = index
        self.cfg = config
        self.experts = experts
        self.w = {
            name: torch.from_numpy(weights(f"blk.{index}.{name}.weight"))
            for name in (
                "attn_norm",
                "attn_q_a",
                "attn_q_a_norm",
                "attn_q_b",
                "attn_kv_a_mqa",
                "attn_kv_a_norm",
                "attn_k_b",
                "attn_v_b",
                "attn_output",
                "ffn_norm",
            )
        }
        self.namespace = namespace
        self.official = namespace["DeepseekV3Attention"](
            SimpleNamespace(**config), layer_idx=index
        ).eval()
        self.official.load_state_dict(
            {
                "q_a_proj.weight": self.w["attn_q_a"],
                "q_a_layernorm.weight": self.w["attn_q_a_norm"],
                "q_b_proj.weight": self.w["attn_q_b"],
                "kv_a_proj_with_mqa.weight": self.w["attn_kv_a_mqa"],
                "kv_a_layernorm.weight": self.w["attn_kv_a_norm"],
                "kv_b_proj.weight": torch.cat(
                    [self.w["attn_k_b"].transpose(1, 2), self.w["attn_v_b"]], 1
                ).reshape(64 * 256, 512),
                "o_proj.weight": self.w["attn_output"],
            },
            strict=True,
            assign=True,
        )
        self.shared = namespace["DeepseekV3MLP"](
            SimpleNamespace(**config), intermediate_size=18432 if index == 0 else 2048
        ).eval()
        if index == 0:
            matrices = {
                name: torch.from_numpy(weights(f"blk.0.ffn_{name}.weight"))
                for name in ("gate", "up", "down")
            }
        else:
            matrices = {
                name: torch.from_numpy(experts(f"ffn_{name}_shexp.weight"))
                for name in ("gate", "up", "down")
            }
            self.router = namespace["MoEGate"](SimpleNamespace(**config)).eval()
            self.router.load_state_dict(
                {
                    "weight": torch.from_numpy(experts("ffn_gate_inp.weight")),
                    "e_score_correction_bias": torch.from_numpy(
                        experts("exp_probs_b.bias")
                    ),
                },
                strict=True,
                assign=True,
            )
        self.shared.load_state_dict(
            {name + "_proj.weight": w for name, w in matrices.items()},
            strict=True,
            assign=True,
        )
        self.state = None
        self.last_ids = None

    def norm(self, x, name):
        return F.rms_norm(x, (x.shape[-1],), self.w[name], 1e-6)

    def compressed_attention(self, x):
        """Absorbed MLA; retain [positions,512] latent and [positions,64] rotary key."""
        start = 0 if self.state is None else len(self.state[0])
        q = F.linear(
            self.norm(F.linear(x, self.w["attn_q_a"]), "attn_q_a_norm"),
            self.w["attn_q_b"],
        ).reshape(-1, 64, 192)
        kv = F.linear(x, self.w["attn_kv_a_mqa"])
        latent = self.norm(kv[:, :512], "attn_kv_a_norm")
        positions = np.arange(start, start + len(x), dtype=np.float32)
        qp = torch.from_numpy(rotary(q[:, :, 128:].numpy(), positions, self.cfg))
        kp = torch.from_numpy(rotary(kv[:, None, 512:].numpy(), positions, self.cfg))[
            :, 0, :
        ]
        if self.state is not None:
            latent = torch.cat([self.state[0], latent])
            kp = torch.cat([self.state[1], kp])
        self.state = (latent, kp)
        absorbed = torch.einsum("thk,hlk->thl", q[:, :, :128], self.w["attn_k_b"])
        scores = torch.einsum("thl,sl->hts", absorbed, latent) + torch.einsum(
            "thd,sd->hts", qp, kp
        )
        scale = (
            192**-0.5 * (1 + 0.1 * math.log(self.cfg["rope_scaling"]["factor"])) ** 2
        )
        mask = (
            torch.arange(len(latent))[None, :]
            > torch.arange(start, start + len(x))[:, None]
        )
        probabilities = (
            (scores * scale).masked_fill(mask[None], float("-inf")).softmax(-1)
        )
        context = torch.einsum("hts,sl->thl", probabilities, latent)
        values = torch.einsum("thl,hvl->thv", context, self.w["attn_v_b"])
        return F.linear(values.reshape(len(x), -1), self.w["attn_output"])

    def feed_forward(self, x):
        """Use native K2 router IDs and actual selected expert tensors."""
        shared = self.shared(x)
        if self.index == 0:
            return shared
        ids, weights = self.router(x[None])
        self.last_ids = ids.clone()
        outputs = torch.zeros(len(x), 8, 7168)
        for expert in ids.unique():
            rows, slots = torch.where(ids == expert)
            print("expert", expert.item(), flush=True)
            matrices = {
                name: torch.from_numpy(
                    self.experts(f"ffn_{name}_exps.weight", expert.item())
                )
                for name in ("gate", "up", "down")
            }
            gate = F.linear(x[rows], matrices["gate"])
            up = F.linear(x[rows], matrices["up"])
            outputs[rows, slots] = F.linear(F.silu(gate) * up, matrices["down"])
        return shared + (outputs * weights[..., None]).sum(1)

    def __call__(self, x, official=False):
        normalized = self.norm(x, "attn_norm")
        if official:
            mask = torch.full((1, 1, len(x), len(x)), float("-inf")).triu(1)
            attention = self.official(
                normalized[None],
                attention_mask=mask,
                position_ids=torch.arange(len(x))[None],
            )[0][0]
        else:
            attention = self.compressed_attention(normalized)
        x = x + attention
        return x + self.feed_forward(self.norm(x, "ffn_norm"))


def run(root):
    """Compare 24-position prefill, 16+8 incremental positions and upstream expansion."""
    root = Path(root)
    dest = root / "k3"
    config = json.loads((root / "k0/config.json").read_text())
    ns = upstream(root / "k0/modeling_deepseek.py")
    fixture = Fixture(root / "k1/gguf-index.json", dest / "moe-slices")
    layers = [
        Layer(0, SliceWeights(root / "k1/slices"), config, ns),
        Layer(1, SliceWeights(dest), config, ns, fixture),
    ]
    data = np.load(dest / "token-input.npz")
    input = torch.from_numpy(data["embeddings"])
    saved = {"token_ids": data["ids"], "embeddings": data["embeddings"]}
    # Bounds are unchanged from K2 for final hidden outputs.
    metrics = {}

    def check(name, a, b):
        a = a.numpy()
        b = b.numpy()
        diff = a - b
        absolute = float(abs(diff).max())
        relative = float(np.linalg.norm(diff) / np.linalg.norm(b))
        metrics[name] = {"max_abs": absolute, "relative_rms": relative}
        if not np.isfinite(a).all() or absolute > 1e-3 or relative > 1e-4:
            raise ValueError(f"{name}: {metrics[name]}")

    for mode in ("prefill", "incremental", "upstream"):
        for layer in layers:
            layer.state = None
        chunks = (
            [(0, 24)]
            if mode != "incremental"
            else [(0, 16)] + [(i, i + 1) for i in range(16, 24)]
        )
        outputs = [[], []]
        ids = []
        for begin, end in chunks:
            x = input[begin:end]
            for i, layer in enumerate(layers):
                print(mode, "positions", begin, end, "layer", i, flush=True)
                x = layer(x, official=mode == "upstream")
                outputs[i].append(x.clone())
                if i == 1:
                    ids.append(layer.last_ids.clone())
        for i in range(2):
            saved[f"{mode}_layer{i}"] = torch.cat(outputs[i]).numpy()
        saved[f"{mode}_expert_ids"] = torch.cat(ids).numpy()
        if mode != "upstream":
            for i, layer in enumerate(layers):
                assert layer.state[0].shape == (24, 512) and layer.state[1].shape == (
                    24,
                    64,
                )
                saved[f"{mode}_cache{i}_latent"] = layer.state[0].numpy()
                saved[f"{mode}_cache{i}_rope"] = layer.state[1].numpy()
        # Keep partial evidence while later modes execute.
        np.savez(dest / "two-layer.npz", **saved)
    for layer in (0, 1):
        for mode in ("incremental", "upstream"):
            check(
                f"{mode}_layer{layer}",
                torch.from_numpy(saved[f"{mode}_layer{layer}"]),
                torch.from_numpy(saved[f"prefill_layer{layer}"]),
            )
        for field in ("latent", "rope"):
            check(
                f"cache{layer}_{field}",
                torch.from_numpy(saved[f"incremental_cache{layer}_{field}"]),
                torch.from_numpy(saved[f"prefill_cache{layer}_{field}"]),
            )
    for mode in ("incremental", "upstream"):
        np.testing.assert_array_equal(
            np.sort(saved[f"{mode}_expert_ids"], axis=1),
            np.sort(saved["prefill_expert_ids"], axis=1),
        )
    (dest / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.no_grad():
        run(args.root)
