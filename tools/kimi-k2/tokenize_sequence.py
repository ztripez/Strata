"""Run the pinned K2 tokenizer and chat template for the K3 reference sequence."""

import argparse
import importlib.util
import json
from pathlib import Path


def load_tokenizer(root):
    """Load the actual pinned K2 tokenizer from local source bytes."""
    root = Path(root)
    source = root / "k0"
    spec = importlib.util.spec_from_file_location(
        "pinned_kimi_k2_tokenizer", source / "tokenization_kimi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.TikTokenTokenizer.from_pretrained(
        str(source), local_files_only=True
    )
    return tokenizer


def tokenize(root):
    """Capture exact IDs, vocabulary/EOS identity and the rendered chat prompt."""
    root = Path(root)
    tokenizer = load_tokenizer(root)
    messages = [
        {
            "role": "user",
            "content": "Explain why the sky appears blue in one short sentence.",
        }
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if tokenizer.eos_token_id != 163585 or len(tokenizer) != 163840:
        raise ValueError("tokenizer identity differs from pinned K2 config")
    previous = json.loads((root / "k2/tokenizer-bringup.json").read_text())
    if ids != previous["ids"]:
        raise ValueError("token IDs changed from the K3 input fixture")
    result = {
        "messages": messages,
        "all_template_ids": ids,
        "tested_ids": ids[:24],
        "prefill_ids": ids[:16],
        "incremental_teacher_forced_ids": ids[16:24],
        "eos": tokenizer.eos_token_id,
        "vocab": len(tokenizer),
        "rendered": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        "tested_decoded": tokenizer.decode(ids[:24]),
    }
    (root / "k3/tokenizer.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    tokenize(args.root)
