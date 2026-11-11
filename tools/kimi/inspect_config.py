"""Inspect the geometry needed by the experimental Kimi K3 adapter."""

import argparse
import json
from pathlib import Path


def inspect(path: Path) -> dict:
    """Read the published config and return its execution geometry."""
    config = json.loads(path.read_text())
    if config["model_type"] != "kimi_k3":
        raise ValueError("expected a kimi_k3 configuration")
    text = config["text_config"]
    attention = text["linear_attn_config"]
    kda = set(attention["kda_layers"])
    mla = set(attention["full_attn_layers"])
    count = text["num_hidden_layers"]
    if kda & mla or kda | mla != set(range(1, count + 1)):
        raise ValueError("attention layers must partition the model")
    return {
        "hidden_size": text["hidden_size"],
        "experts": text["num_experts"],
        "selected_experts": text["num_experts_per_token"],
        "expert_hidden_size": text["routed_expert_hidden_size"],
        "weight_format": text["quantization_config"]["format"],
        "layers": [
            {
                "index": i,
                "attention": "kda" if i + 1 in kda else "gated_mla",
                "feed_forward": "dense" if i < text["first_k_dense_replace"] else "moe",
            }
            for i in range(count)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.config), indent=2))
