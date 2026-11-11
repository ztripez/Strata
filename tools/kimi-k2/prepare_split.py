"""Bind K6's fixed four-layer split plan to the original K2 fixture identity."""

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path("/mnt/work/kimi-k2")
shared = Path("/mnt/storage/models/strata-kimi-k2-reference/k5-native")
model = hashlib.sha256((root / "k0/config.json").read_bytes()).hexdigest()
plan = {
    "epoch": 1,
    "scope": "K6 real layers0-3 only",
    "cut": 2,
    "layer_owners": {"lucy": [0, 1], "astor": [2, 3]},
    "remote_experts": {"lucy_to_astor": [[1, 0]], "astor_to_lucy": [[2, 2], [3, 2]]},
    "fixture_identity": (root / "k6/fixture.identity").read_text().strip(),
    "arithmetic": "decoded F32 GPU / DequantizedF32 existing CPU pool",
    "windows": [2, 1, 1],
    "max_abs": 1e-3,
    "relative_rms": 1e-4,
}
raw = json.dumps(plan, sort_keys=True).encode()
(root / "k6/split-plan.json").write_bytes(raw)
ids = np.load(root / "k3/token-input.npz")["ids"][:4]
for dest in (shared, root / "k5/native"):
    (dest / "model.identity").write_text(model + "\n")
    (dest / "split-plan.identity").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    (dest / "token-ids.txt").write_text(" ".join(str(int(x)) for x in ids) + "\n")
print(json.dumps(plan))
