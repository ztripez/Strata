"""Compare GGUF CPU reference to actual pinned K2 upstream attention/MLP code."""

import argparse
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from cpu_reference import SliceWeights


def upstream(path):
    """Load only upstream mathematical definitions, without transformer server imports."""
    names = {
        "DeepseekV3RMSNorm",
        "DeepseekV3RotaryEmbedding",
        "DeepseekV3LinearScalingRotaryEmbedding",
        "DeepseekV3DynamicNTKScalingRotaryEmbedding",
        "DeepseekV3YarnRotaryEmbedding",
        "yarn_find_correction_dim",
        "yarn_find_correction_range",
        "yarn_get_mscale",
        "yarn_linear_ramp_mask",
        "rotate_half",
        "apply_rotary_pos_emb",
        "DeepseekV3Attention",
        "DeepseekV3MLP",
        "MoEGate",
    }
    tree = ast.parse(Path(path).read_text())
    selected = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    if {n.name for n in selected} != names:
        raise ValueError("missing upstream mathematical definitions")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": F,
        "math": math,
        "ACT2FN": {"silu": F.silu},
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102 - executes selected pinned upstream math definitions
    return namespace


def compare(slices, config_path, source_path, receipt_path):
    """Run pinned upstream code and assert all available matching boundaries."""
    cfg = SimpleNamespace(**json.loads(Path(config_path).read_text()))
    namespace = upstream(source_path)
    attention = namespace["DeepseekV3Attention"](cfg, layer_idx=0).eval()
    mlp = namespace["DeepseekV3MLP"](cfg).eval()
    w = SliceWeights(slices)

    def weight(name):
        return torch.from_numpy(w("blk.0." + name + ".weight"))

    kb = weight("attn_k_b").transpose(1, 2)
    vb = weight("attn_v_b")
    state = {
        "q_a_proj.weight": weight("attn_q_a"),
        "q_a_layernorm.weight": weight("attn_q_a_norm"),
        "q_b_proj.weight": weight("attn_q_b"),
        "kv_a_proj_with_mqa.weight": weight("attn_kv_a_mqa"),
        "kv_a_layernorm.weight": weight("attn_kv_a_norm"),
        "kv_b_proj.weight": torch.cat([kb, vb], dim=1).reshape(64 * 256, 512),
        "o_proj.weight": weight("attn_output"),
    }
    attention.load_state_dict(state, strict=True, assign=True)
    mlp.load_state_dict(
        {
            name + "_proj.weight": weight("ffn_" + name)
            for name in ("gate", "up", "down")
        },
        strict=True,
        assign=True,
    )
    receipt = np.load(receipt_path)
    metrics = {}
    captured = {}

    def check(name, actual):
        a = actual.detach().numpy().reshape(receipt[name].shape)
        b = receipt[name]
        captured[name] = a.copy()
        diff = a - b
        rel = float(np.linalg.norm(diff) / max(float(np.linalg.norm(b)), 1e-30))
        absolute = float(abs(diff).max())
        metrics[name] = {"max_abs": absolute, "relative_rms": rel}
        if not np.isfinite(a).all() or rel > 1e-4 or absolute > 1e-3:
            raise ValueError(f"{name} outside frozen tolerances: {metrics[name]}")

    hooks = []
    for attr, name in [
        ("q_a_proj", "q_a"),
        ("q_a_layernorm", "q_a_norm"),
        ("q_b_proj", "q"),
        ("kv_a_proj_with_mqa", "kv_a"),
        ("kv_a_layernorm", "kv_norm"),
    ]:
        hooks.append(
            getattr(attention, attr).register_forward_hook(
                lambda m, i, o, name=name: check(name, o)
            )
        )
    for attr, name in [("gate_proj", "gate"), ("up_proj", "up")]:
        hooks.append(
            getattr(mlp, attr).register_forward_hook(
                lambda m, i, o, name=name: check(name, o)
            )
        )
    x = torch.from_numpy(receipt["input"]).unsqueeze(0)
    n = F.rms_norm(x, (7168,), weight("attn_norm"), 1e-6)
    check("input_norm", n)
    mask = torch.full((1, 1, len(x[0]), len(x[0])), float("-inf")).triu(1)
    att, prob, _ = attention(
        n,
        attention_mask=mask,
        position_ids=torch.arange(len(x[0]))[None],
        output_attentions=True,
    )
    check("attention_probs", prob)
    check("attention_output", att)
    residual = x + att
    check("attention_residual", residual)
    post = F.rms_norm(residual, (7168,), weight("ffn_norm"), 1e-6)
    check("post_norm", post)
    out = mlp(post)
    check("mlp_output", out)
    check("output", residual + out)
    for hook in hooks:
        hook.remove()
    np.savez(Path(receipt_path).with_name("layer0-upstream.npz"), **captured)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("slices", "config", "upstream", "receipt"):
        parser.add_argument(arg)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with torch.no_grad():
        result = compare(args.slices, args.config, args.upstream, args.receipt)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
