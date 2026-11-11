"""Freeze K7's pinned full-model placement and exact 256-token chat input."""

import hashlib
import json
from pathlib import Path

from tokenize_sequence import load_tokenizer

root = Path("/mnt/work/kimi-k2")
pack = Path("/mnt/storage/models/Kimi-K2-Instruct-UD-IQ3_XXS-23bf90d8196d")
profile = pack / "profile"
profile.mkdir(exist_ok=True)
manifest = (root / "k0/shard-manifest.json").read_bytes()
pack_sha = hashlib.sha256(manifest).hexdigest()
tokenizer = load_tokenizer(root)
messages = None
for padding in range(400):
    content = (
        "Write a detailed explanation of why the daytime sky appears blue. "
        "Explain how sunlight interacts with molecules in the atmosphere, "
        "how the effect depends on wavelength, and why sunsets look different. "
        "Use at least five substantial paragraphs and avoid a short summary. "
        "Consider each additional detail carefully: " + "detail " * padding
    )
    candidate = [{"role": "user", "content": content}]
    ids = tokenizer.apply_chat_template(
        candidate, tokenize=True, add_generation_prompt=True
    )
    if len(ids) == 256:
        messages = candidate
        break
if messages is None:
    raise ValueError("could not construct a complete 256-token chat prompt")
generation_eos = json.loads((root / "k0/generation_config.json").read_text())[
    "eos_token_id"
]
if generation_eos != 163586 or len(tokenizer) != 163840:
    raise ValueError("pinned generation/tokenizer identity mismatch")
source_identity = hashlib.sha256((root / "k0/config.json").read_bytes()).hexdigest()
plan = {
    "epoch": 1,
    "scope": "K7 full 61-layer K2 Instruct UD-IQ3_XXS",
    "model_revision": "fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
    "quant_revision": "23bf90d8196ddfcb3328a6f60c99a9bdb5c1dadf",
    "model_manifest_sha256": pack_sha,
    "cut": 2,
    "layer_owners": {"lucy": [0, 1], "astor": list(range(2, 61))},
    "remote_experts": {"lucy_to_astor": [[1, 0]], "astor_to_lucy": [[2, 2], [3, 2]]},
    "arithmetic": "decoded F32 GPU / DequantizedF32 existing CPU pool",
    "prompt_tokens": 256,
    "prefill_window": 8,
    "maximum_decode_tokens": 16,
    "sampling": "greedy",
    "generation_eos": generation_eos,
    "layer_transaction_timeout_ms": 600000,
    "checkpoint_has_every_routed_expert": 384,
    "memory_profile": {
        "lucy_vram_gib": 9,
        "astor_vram_gib": 14,
        "process_rss_hwm_gib_exclusive": 48,
    },
}
raw = json.dumps(plan, sort_keys=True).encode()
(profile / "model.identity").write_text(source_identity + "\n")
(profile / "fixture.identity").write_text(pack_sha + "\n")
(profile / "split-plan.identity").write_text(hashlib.sha256(raw).hexdigest() + "\n")
(profile / "token-ids.txt").write_text(" ".join(str(x) for x in ids) + "\n")
(profile / "split-plan.json").write_bytes(raw)
(profile / "shard-manifest.json").write_bytes(manifest)
(root / "k7/checksums.sha256").write_text(
    "".join(f"{s['sha256']}  {s['shard']}\n" for s in json.loads(manifest))
)
capture = {
    "messages": messages,
    "ids": ids,
    "rendered": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ),
    "tokenizer_generic_eos": tokenizer.eos_token_id,
    "generation_eos": generation_eos,
}
(root / "k7/prompt.json").write_text(json.dumps(capture, indent=2) + "\n")
(root / "k7/split-plan.json").write_bytes(raw)
print(
    json.dumps(
        {
            "profile": str(profile),
            "prompt_tokens": len(ids),
            "generation_eos": generation_eos,
            "pack_sha256": pack_sha,
        }
    )
)
