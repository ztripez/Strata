#!/usr/bin/env python3
"""Generate network expert placement plans from decode CPU-miss traces.

Reads a TSV trace emitted by `STRATA_Q36_TRACE_MISSES=<path>` with columns:
  layer, expert, decode_hits, decode_misses, worker_slot_bytes

Supports two network-aware ranking modes:
  - `expert` (default): Ranks individual `(layer, expert)` pairs globally by
    expected CPU work saved per byte of worker VRAM residency
    (`(decode_misses - dispatch_penalty_misses) / worker_slot_bytes`), with
    deterministic tie-breaking by `(decode_misses desc, decode_hits desc, layer asc, expert asc)`.
  - `layer`: Ranks whole layers by net CPU misses saved per byte of worker VRAM
    residency after accounting for the per-layer remote synchronization cost
    (`(layer_decode_misses - layer_sync_penalty_misses) / layer_worker_bytes`),
    then selects experts within the top-ranked layers in descending order of
    `decode_misses / worker_slot_bytes`. This concentrates remote residency onto
    fewer layers so unowned layers pay zero network round-trip or `h_ewts`
    transfer overhead while owned layers eliminate nearly all local CPU pool work.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def format_ranges(experts: list[int]) -> str:
    if not experts:
        return ""
    s = sorted(set(experts))
    parts: list[str] = []
    start = prev = s[0]
    for x in s[1:]:
        if x == prev + 1:
            prev = x
        else:
            parts.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = x
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, help="Input TSV from STRATA_Q36_TRACE_MISSES")
    ap.add_argument("--out", required=True, help="Output placement .txt path")
    ap.add_argument("--summary-json", default="", help="Optional JSON summary output path")
    ap.add_argument("--mode", choices=["expert", "layer"], default="expert", help="Ranking mode")
    ap.add_argument("--max-experts", type=int, default=0, help="Maximum number of (layer, expert) pairs (0 = no count cap)")
    ap.add_argument("--max-layers", type=int, default=0, help="Maximum number of layers to populate (0 = all layers)")
    ap.add_argument("--max-gib", type=float, default=14.8, help="Maximum worker VRAM budget in GiB")
    ap.add_argument("--include-zero-miss", action="store_true", help="Include 0-miss experts after positive-miss ones (e.g. for full-layer coverage)")
    ap.add_argument(
        "--layer-sync-penalty-misses",
        type=float,
        default=0.0,
        help="Equivalent CPU-miss penalty per active remote layer (models per-layer network synchronization overhead)",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    by_layer: dict[int, list[dict]] = defaultdict(list)
    with open(args.trace, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            item = {
                "layer": int(r["layer"]),
                "expert": int(r["expert"]),
                "hits": int(r["decode_hits"]),
                "misses": int(r["decode_misses"]),
                "bytes": int(r["worker_slot_bytes"]),
            }
            item["misses_per_mib"] = item["misses"] / (item["bytes"] / (1024.0 * 1024.0))
            rows.append(item)
            by_layer[item["layer"]].append(item)

    total_misses = sum(r["misses"] for r in rows)
    total_hits = sum(r["hits"] for r in rows)
    max_bytes = int(args.max_gib * (1024**3))

    for l in by_layer:
        by_layer[l].sort(key=lambda x: (-x["misses"] / x["bytes"], -x["misses"], -x["hits"], x["expert"]))

    # Layer-level ranking by net misses saved per byte after layer sync penalty
    layer_Order = sorted(
        by_layer.keys(),
        key=lambda l: (
            -(sum(x["misses"] for x in by_layer[l]) - args.layer_sync_penalty_misses)
            / sum(x["bytes"] for x in by_layer[l]),
            -sum(x["misses"] for x in by_layer[l]),
            l,
        ),
    )
    allowed_layers = set(layer_Order[: args.max_layers] if args.max_layers > 0 else layer_Order)

    candidates: list[dict] = []
    if args.mode == "expert":
        pool = [r for r in rows if r["layer"] in allowed_layers and (r["misses"] > 0 or args.include_zero_miss)]
        pool.sort(key=lambda x: (-x["misses"] / x["bytes"], -x["misses"], -x["hits"], x["layer"], x["expert"]))
        candidates = pool
    else:
        for l in layer_Order:
            if l not in allowed_layers:
                continue
            for r in by_layer[l]:
                if r["misses"] > 0 or args.include_zero_miss:
                    candidates.append(r)

    selected: list[dict] = []
    used_bytes = 0
    for r in candidates:
        if args.max_experts > 0 and len(selected) >= args.max_experts:
            break
        if used_bytes + r["bytes"] > max_bytes:
            continue
        selected.append(r)
        used_bytes += r["bytes"]

    sel_by_layer: dict[int, list[int]] = defaultdict(list)
    for r in selected:
        sel_by_layer[r["layer"]].append(r["expert"])

    covered_misses = sum(r["misses"] for r in selected)
    covered_hits = sum(r["hits"] for r in selected)
    active_layers = sorted(sel_by_layer.keys())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(
            f"# Generated by tools/qwen35/generate_placement_from_trace.py\n"
            f"# mode={args.mode} experts={len(selected)} layers={len(active_layers)} "
            f"vram_bytes={used_bytes} ({used_bytes / (1024**3):.3f} GiB) "
            f"miss_coverage={covered_misses}/{total_misses} ({(100.0 * covered_misses / max(1, total_misses)):.2f}%)\n"
        )
        for l in active_layers:
            out.write(f"{l} {format_ranges(sel_by_layer[l])}\n")

    summary = {
        "mode": args.mode,
        "max_experts": args.max_experts,
        "max_layers": args.max_layers,
        "max_gib": args.max_gib,
        "selected_experts": len(selected),
        "selected_layers": len(active_layers),
        "active_layers": active_layers,
        "worker_vram_bytes": used_bytes,
        "worker_vram_gib": used_bytes / (1024**3),
        "trace_total_decode_hits": total_hits,
        "trace_total_decode_misses": total_misses,
        "covered_decode_misses": covered_misses,
        "covered_decode_miss_fraction": covered_misses / max(1, total_misses),
        "covered_decode_hits": covered_hits,
    }
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
            jf.write("\n")

    print(
        f"Wrote {out_path}: {len(selected)} experts across {len(active_layers)} layers, "
        f"{used_bytes / (1024**3):.2f} GiB worker VRAM, "
        f"covers {covered_misses}/{total_misses} decode misses ({100.0 * covered_misses / max(1, total_misses):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
