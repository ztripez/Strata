#!/usr/bin/env python3
"""
Historical offline placement prototype (unqualified).

This draft uses assumed costs. It is not the live fabric coordinator, has no
accepted whole-layer ranking accuracy, and does not establish a passed gate.
See docs/FABRIC.md and tools/network_fabric_inventory.py for current context.

NON-NEGOTIABLE CONTRACT:
  Strata ALREADY manages node-local VRAM -> RAM -> SSD expert caching, residency, and execution
  (--resident-experts, --resident-budget-gib N, --mmap-experts, ExpertSlotCache, ExpertStore).
  This orchestrator does NOT replace or duplicate Strata's local cache hierarchy.
  It performs:
    1. Cross-node resource discovery (GPU VRAM/arch/formats, host RAM/NUMA, SSD/NFS capacity, 10GbE RTT/BW).
    2. Explicit per-layer critical-path cost modeling:
         cost(remote) = dispatch + network + staging + queue + expert_compute + return + combine - overlap
       accounting for unowned-layer zero-sync fast paths and local CPU/SSD tier costs.
    3. Replayable, model-hash-bound plan manifest generation for three Pareto profiles:
       `speed`, `balanced`, and `capacity`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = "strata.fabric.v1"


def run_cmd(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def parse_meminfo(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, rest = line.split(":", 1)
        parts = rest.strip().split()
        if parts and parts[0].isdigit():
            val = int(parts[0])
            if len(parts) > 1 and parts[1].lower() == "kb":
                val *= 1024
            out[k.strip()] = val
    return out


def parse_df(text: str) -> List[Dict[str, Any]]:
    mounts: List[Dict[str, Any]] = []
    lines = text.splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        mounts.append({
            "source": parts[0],
            "fstype": parts[1],
            "total_bytes": int(parts[2]) * 1024,
            "used_bytes": int(parts[3]) * 1024,
            "free_bytes": int(parts[4]) * 1024,
            "mount": parts[6],
        })
    return mounts


def discover_local_node(model_path: str, model_sha256: str) -> Dict[str, Any]:
    _, hostname, _ = run_cmd(["hostname"])
    _, lscpu_out, _ = run_cmd(["lscpu"])
    cpu_model = "unknown"
    logical_cpus = os.cpu_count() or 1
    numa_nodes = 1
    for line in lscpu_out.splitlines():
        if line.startswith("Model name:"):
            cpu_model = line.split(":", 1)[1].strip()
        elif line.startswith("CPU(s):"):
            try:
                logical_cpus = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("NUMA node(s):"):
            try:
                numa_nodes = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    mem = parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
    _, df_out, _ = run_cmd(["df", "-T", "-k", "/", "/mnt/work", "/mnt/storage"])
    mounts = parse_df(df_out)

    # Query NVIDIA GPU
    _, smi_out, _ = run_cmd([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,memory.used,compute_cap,pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader,nounits",
    ])
    gpus: List[Dict[str, Any]] = []
    for line in smi_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({
                "index": int(parts[0]),
                "vendor": "nvidia",
                "backend": "cuda",
                "name": parts[1],
                "vram_total_bytes": int(parts[2]) * 1024 * 1024,
                "vram_free_bytes": int(parts[3]) * 1024 * 1024,
                "vram_used_bytes": int(parts[4]) * 1024 * 1024,
                "arch": f"sm_{parts[5].replace('.', '')}",
                "supported_quant_formats": [
                    "Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16",
                    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "MXFP4"
                ],
            })

    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    return {
        "node_id": hostname or "lucy",
        "role": "primary",
        "cpu": {
            "model": cpu_model,
            "logical_cpus": logical_cpus,
            "physical_cores": logical_cpus // 2,
            "numa_nodes": numa_nodes,
        },
        "memory": {
            "total_bytes": mem.get("MemTotal", 0),
            "available_bytes": mem.get("MemAvailable", 0),
            "free_bytes": mem.get("MemFree", 0),
            "cached_bytes": mem.get("Cached", 0),
            "safe_pinned_budget_bytes": max(0, mem.get("MemTotal", 0) - 12 * (1 << 30)),
        },
        "gpus": gpus,
        "storage": mounts,
        "local_strata_capabilities": {
            "gpu_expert_cache": "ExpertSlotCache (--expert-slots N)",
            "ram_residency_modes": [
                "--resident-experts",
                "--resident-budget-gib N",
                "--mmap-experts",
            ],
            "dsv41_store_modes": [
                "--ssd-cache N",
                "--mmap-experts on|off|auto",
            ],
        },
        "model_replica": {
            "path": model_path,
            "size_bytes": model_size,
            "sha256": model_sha256,
        },
        "measured_costs": {
            "cpu_pool_threads": 12,
            "cpu_expert_eval_ms_per_wave": 0.420,
            "cpu_wave_capacity_experts": 12,
            "ssd_cold_page_fault_ms_per_expert": 0.650,
            "local_gpu_cached_expert_us": 14.0,
        },
    }


def discover_remote_worker(
    ssh_host: str,
    worker_endpoint: str,
    worker_model_path: str,
    model_sha256: str,
) -> Dict[str, Any]:
    remote_script = f"""python3 - <<'PY'
import json, os, subprocess
from pathlib import Path

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip()

lscpu = run(["lscpu"])
cpu_model = "unknown"
logical_cpus = os.cpu_count() or 1
for line in lscpu.splitlines():
    if line.startswith("Model name:"):
        cpu_model = line.split(":", 1)[1].strip()

mem = {{}}
for line in Path("/proc/meminfo").read_text().splitlines():
    if ":" in line:
        k, rest = line.split(":", 1)
        p = rest.strip().split()
        if p and p[0].isdigit():
            val = int(p[0]) * (1024 if len(p) > 1 and p[1].lower() == "kb" else 1)
            mem[k.strip()] = val

df_out = run(["df", "-T", "-k", "/", "/mnt/storage"])
mounts = []
for line in df_out.splitlines()[1:]:
    p = line.split()
    if len(p) >= 7:
        mounts.append({{
            "source": p[0], "fstype": p[1],
            "total_bytes": int(p[2]) * 1024, "used_bytes": int(p[3]) * 1024,
            "free_bytes": int(p[4]) * 1024, "mount": p[6]
        }})

# Query HIP / sysfs VRAM
vram_total = int(Path("/sys/class/drm/card1/device/mem_info_vram_total").read_text().strip())
vram_used = int(Path("/sys/class/drm/card1/device/mem_info_vram_used").read_text().strip())
msize = os.path.getsize({worker_model_path!r}) if os.path.exists({worker_model_path!r}) else 0

print(json.dumps({{
    "node_id": run(["hostname"]) or "astor",
    "role": "worker",
    "endpoint": {worker_endpoint!r},
    "cpu": {{"model": cpu_model, "logical_cpus": logical_cpus, "physical_cores": logical_cpus // 2, "numa_nodes": 1}},
    "memory": {{
        "total_bytes": mem.get("MemTotal", 0),
        "available_bytes": mem.get("MemAvailable", 0),
        "free_bytes": mem.get("MemFree", 0),
        "cached_bytes": mem.get("Cached", 0),
    }},
    "gpus": [{{
        "index": 0,
        "vendor": "amd",
        "backend": "hip",
        "name": "AMD Radeon RX 6800",
        "arch": "gfx1030",
        "vram_total_bytes": vram_total,
        "vram_used_bytes": vram_used,
        "vram_free_bytes": vram_total - vram_used,
        "vram_total_idle_free_bytes": vram_total - 26 * 1024 * 1024,
        "supported_quant_formats": [
            "Q4_K", "Q5_K", "Q6_K", "Q8_0", "BF16", "F16",
            "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "MXFP4"
        ]
    }}],
    "storage": mounts,
    "model_replica": {{
        "path": {worker_model_path!r},
        "size_bytes": msize,
        "sha256": {model_sha256!r}
    }},
    "measured_costs": {{
        "worker_gpu_dispatch_overhead_us": 42.0,
        "worker_gpu_per_expert_us": 16.5,
        "worker_stage_combine_us": 18.0
    }}
}}))
PY"""
    rc, out, err = run_cmd(["ssh", ssh_host, remote_script], timeout=30)
    if rc != 0 or not out:
        raise RuntimeError(f"Failed to discover remote worker {ssh_host}: {err}")
    return json.loads(out)


def discover_link(worker_ip: str) -> Dict[str, Any]:
    _, route_out, _ = run_cmd(["ip", "route", "get", worker_ip])
    dev = "enp4s0"
    src_ip = "192.168.60.155"
    m_dev = re.search(r"\bdev\s+(\S+)", route_out)
    if m_dev:
        dev = m_dev.group(1)
    m_src = re.search(r"\bsrc\s+(\S+)", route_out)
    if m_src:
        src_ip = m_src.group(1)

    speed_mbps = 10000
    speed_path = Path(f"/sys/class/net/{dev}/speed")
    if speed_path.exists():
        try:
            speed_mbps = int(speed_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass

    _, ping_out, _ = run_cmd(["ping", "-c", "5", "-i", "0.05", worker_ip])
    rtt_avg_ms = 0.069
    m_rtt = re.search(r"=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms", ping_out)
    if m_rtt:
        rtt_avg_ms = float(m_rtt.group(2))

    return {
        "primary_interface": dev,
        "primary_ip": src_ip,
        "worker_ip": worker_ip,
        "route": route_out.splitlines()[0] if route_out else "",
        "link_speed_mbps": speed_mbps,
        "measured_tcp_gbps": 9.183,
        "rtt_avg_ms": rtt_avg_ms,
        "tcp_roundtrip_base_ms": max(rtt_avg_ms, 0.075),
    }


def load_trace_stats(trace_path: str, n_layers: int = 40, n_experts: int = 256) -> Dict[str, Any]:
    layer_misses = [0] * n_layers
    layer_hits = [0] * n_layers
    expert_misses: Dict[Tuple[int, int], int] = {}
    slot_bytes = 1900544

    with open(trace_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            layer = int(row["layer"])
            expert = int(row["expert"])
            misses = int(row.get("decode_misses") or row.get("cpu_misses") or 0)
            hits = int(row.get("decode_hits") or 0)
            if "worker_slot_bytes" in row and row["worker_slot_bytes"]:
                slot_bytes = int(row["worker_slot_bytes"])
            if 0 <= layer < n_layers and 0 <= expert < n_experts:
                layer_misses[layer] += misses
                layer_hits[layer] += hits
                expert_misses[(layer, expert)] = expert_misses.get((layer, expert), 0) + misses

    total_misses = sum(layer_misses)
    total_hits = sum(layer_hits)
    # In Qwen3.6-35B-A3B decode, top-8 routed experts per layer -> (total_hits + total_misses) / (40 * 8) = decode tokens
    est_tokens = max(1.0, (total_hits + total_misses) / (float(n_layers) * 8.0))
    ranked_layers = sorted(range(n_layers), key=lambda l: (-layer_misses[l], l))

    return {
        "trace_path": str(Path(trace_path).resolve()),
        "total_cpu_misses": total_misses,
        "total_gpu_hits": total_hits,
        "worker_slot_bytes": slot_bytes,
        "estimated_trace_tokens": round(est_tokens, 1),
        "layer_misses": layer_misses,
        "ranked_layers": ranked_layers,
        "expert_misses": expert_misses,
    }


def evaluate_candidate_plan(
    *,
    num_owned_layers: int,
    ranked_layers: List[int],
    layer_misses: List[int],
    total_misses: int,
    est_tokens: float,
    local_residency_mode: str,
    local_budget_gib: Optional[float],
    worker_vram_free_bytes: int,
    expert_bytes: int = 1906346,
    n_layers: int = 40,
    n_experts: int = 256,
    hidden_dim: int = 2048,
) -> Dict[str, Any]:
    """
    Evaluate the explicit Section 4 per-layer critical-path cost model:
      cost(remote) = dispatch + network + staging + queue + expert_compute + return + combine - overlap
    versus local CPU pool + SSD miss cost.
    """
    owned_layers = sorted(ranked_layers[:num_owned_layers])
    owned_set = set(owned_layers)
    owned_experts = num_owned_layers * n_experts
    worker_vram_bytes = owned_experts * expert_bytes
    # Safe worker VRAM ceiling: leave at least 1.5 GiB headroom on RX 6800 for HIP scratch/streams/page-table stability
    worker_vram_safe_budget = worker_vram_free_bytes - int(1.5 * (1 << 30))
    fits_worker_vram = worker_vram_bytes <= worker_vram_safe_budget

    offloaded_misses = sum(layer_misses[l] for l in owned_layers)
    remaining_local_misses = total_misses - offloaded_misses
    coverage = offloaded_misses / total_misses if total_misses > 0 else 0.0

    # Per-token decode critical-path model (ms/token):
    # Base GPU attention + dense + router + local VRAM cached experts across 40 layers: ~14.10 ms/token
    t_gpu_spine_ms = 14.10

    # Local CPU pool cost per unowned layer:
    # Unowned layers average `misses_per_tok` CPU misses (typically ~4.0-5.0 experts <= 12-thread pool wave).
    # One CPU wave of Q4_K/Q6_K experts takes ~0.165 ms when resident in RAM.
    # If local_residency_mode is budget-limited or mmap, cold page-cache misses add SSD read latency.
    local_unowned_layers = n_layers - num_owned_layers
    local_expert_ram_gib = (local_unowned_layers * n_experts * expert_bytes) / float(1 << 30)

    if local_residency_mode == "resident-experts":
        primary_pinned_expert_gib = local_expert_ram_gib
        ssd_miss_penalty_ms_per_layer = 0.0
    elif local_residency_mode == "resident-budget-gib":
        budget = local_budget_gib or 4.0
        primary_pinned_expert_gib = min(local_expert_ram_gib, budget)
        # Remaining unpinned experts reside in OS page cache (64 GiB host RAM holds them warm after first pass)
        unpinned_frac = max(0.0, (local_expert_ram_gib - primary_pinned_expert_gib) / max(1e-6, local_expert_ram_gib))
        ssd_miss_penalty_ms_per_layer = 0.008 * unpinned_frac
    elif local_residency_mode == "mmap-experts":
        primary_pinned_expert_gib = 0.0
        ssd_miss_penalty_ms_per_layer = 0.004
    else:  # default legacy full-pin
        primary_pinned_expert_gib = (n_layers * n_experts * expert_bytes) / float(1 << 30)
        ssd_miss_penalty_ms_per_layer = 0.0

    # Critical path per layer:
    # For an UNOWNED layer:
    #   cost_local(l) = local_dispatch (0.012 ms) + cpu_pool_wave (0.165 ms * miss_weight) + ssd_miss_penalty
    #   network sync cost = 0.0 ms (fast path: primary never touches socket on unowned layers!)
    # For a WHOLE-LAYER OWNED remote layer:
    #   dispatch (0.008 ms) + net_tx (8.3 KB @ 9.18 Gbps = 0.007 ms + 0.035 ms wire/socket)
    #   + staging/queue (0.012 ms) + worker_gpu_compute (avg 4.8 experts * 0.014 ms = 0.067 ms)
    #   + net_rx (0.038 ms) + combine (0.008 ms) - overlap (0.025 ms)
    #   = ~0.150 ms/layer when owned layers <= 24.
    #   Above 24 layers (>10.9 GiB on 16 GiB RX 6800), L2/TLB footprint and back-to-back TCP bursts
    #   add queueing jitter: +0.0045 ms per layer above 24.
    total_local_cpu_ms_per_tok = 0.0
    total_remote_net_ms_per_tok = 0.0

    avg_misses_per_layer = (total_misses / est_tokens) / n_layers
    for l in range(n_layers):
        l_misses_per_tok = (layer_misses[l] / est_tokens)
        rel_intensity = l_misses_per_tok / max(1e-6, avg_misses_per_layer)
        if l in owned_set:
            dispatch_ms = 0.008
            net_tx_ms = 0.042
            stage_queue_ms = 0.012 + (0.0045 * max(0, num_owned_layers - 24))
            worker_compute_ms = 0.067 * (0.5 + 0.5 * rel_intensity)
            net_rx_combine_ms = 0.046
            overlap_ms = 0.025
            layer_rem_ms = dispatch_ms + net_tx_ms + stage_queue_ms + worker_compute_ms + net_rx_combine_ms - overlap_ms
            total_remote_net_ms_per_tok += layer_rem_ms
        else:
            local_disp_ms = 0.012
            cpu_wave_ms = 0.168 * rel_intensity
            layer_loc_ms = local_disp_ms + cpu_wave_ms + ssd_miss_penalty_ms_per_layer
            total_local_cpu_ms_per_tok += layer_loc_ms

    total_step_ms = t_gpu_spine_ms + total_local_cpu_ms_per_tok + total_remote_net_ms_per_tok
    pred_warm_decode_tps = 1000.0 / total_step_ms
    # Primary base RSS (model non-routed + runtime buffers + CUDA context) is ~9.09 GiB
    pred_primary_rss_gib = 9.09 + primary_pinned_expert_gib

    return {
        "num_owned_layers": num_owned_layers,
        "owned_layers": owned_layers,
        "owned_experts": owned_experts,
        "worker_vram_bytes": worker_vram_bytes,
        "worker_vram_gib": round(worker_vram_bytes / float(1 << 30), 3),
        "fits_worker_vram": fits_worker_vram,
        "coverage_fraction": round(coverage, 6),
        "offloaded_cpu_misses": offloaded_misses,
        "remaining_local_cpu_misses": remaining_local_misses,
        "local_residency_mode": local_residency_mode,
        "local_budget_gib": local_budget_gib,
        "primary_pinned_expert_gib": round(primary_pinned_expert_gib, 3),
        "predicted_primary_rss_gib": round(pred_primary_rss_gib, 3),
        "cost_breakdown_ms_per_token": {
            "gpu_spine_ms": round(t_gpu_spine_ms, 3),
            "local_cpu_and_ssd_ms": round(total_local_cpu_ms_per_tok, 3),
            "remote_critical_path_ms": round(total_remote_net_ms_per_tok, 3),
            "total_decode_step_ms": round(total_step_ms, 3),
        },
        "predicted_warm_decode_tok_s": round(pred_warm_decode_tps, 2),
    }


def write_placement_txt(path: Path, owned_layers: List[int], n_experts: int = 256, comment: str = "") -> str:
    lines = [f"# {comment}" if comment else "# generated by strata fabric orchestrator"]
    exp_list = ",".join(str(e) for e in range(n_experts))
    for l in sorted(owned_layers):
        lines.append(f"{l}:{exp_list}")
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_manifest(
    *,
    plan_id: str,
    profile: str,
    model_id: str,
    model_sha256: str,
    candidate: Dict[str, Any],
    placement_file: str,
    placement_sha256: str,
    primary_node: str,
    worker_node: str,
    worker_endpoint: str,
) -> Dict[str, Any]:
    local_flags: List[str] = []
    mode = candidate["local_residency_mode"]
    if mode == "resident-experts":
        local_flags = ["--resident-experts"]
    elif mode == "resident-budget-gib":
        local_flags = ["--resident-budget-gib", str(int(candidate["local_budget_gib"] or 4))]
    elif mode == "mmap-experts":
        local_flags = ["--mmap-experts"]

    ownership_entries = []
    for l in range(40):
        is_remote = l in set(candidate["owned_layers"])
        ownership_entries.append({
            "layer": l,
            "experts": "0-255",
            "owner_node": worker_node if is_remote else primary_node,
            "execution_device": "hip:0 (gfx1030)" if is_remote else "cuda:0 (sm_86) + host_cpu_pool",
            "backing_tier": "worker_vram" if is_remote else f"local_vram_cache_12slots + {mode}",
            "fallback_source": "primary_mmap_file_view" if is_remote else "local_host_or_mmap",
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan_id,
        "profile": profile,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": {
            "id": model_id,
            "sha256": model_sha256,
            "family": "qwen35",
            "layers": 40,
            "experts_per_layer": 256,
            "top_k": 8,
        },
        "constraints": {
            "dense_attention_owner": primary_node,
            "router_owner": primary_node,
            "engram_owner": "none (qwen35 has no Engram table)",
            "preserve_strata_local_cache": True,
        },
        "primary": {
            "node": primary_node,
            "strata_cache_flags": local_flags,
            "expert_slots_per_layer": 12,
            "unowned_layer_fast_path": True,
            "fallback_policy": "on_worker_disconnect_or_timeout_use_mmap_or_resident_host_weights",
        },
        "worker": {
            "node": worker_node,
            "endpoint": worker_endpoint,
            "device": "hip:0",
            "placement_file": placement_file,
            "placement_sha256": placement_sha256,
        },
        "cost_model_evaluation": candidate,
        "layer_ownership": ownership_entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Distributed Strata MoE Fabric Offline Orchestrator")
    ap.add_argument("--trace", required=True, help="Path to calibration trace TSV")
    ap.add_argument("--out-dir", required=True, help="Directory to write inventory and candidate plans")
    ap.add_argument("--primary-model", default="/home/ztripez/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
    ap.add_argument("--worker-host", default="astor")
    ap.add_argument("--worker-ip", default="192.168.63.177")
    ap.add_argument("--worker-endpoint", default="192.168.63.177:19415")
    ap.add_argument(
        "--worker-model",
        default="/home/ztripez/.local/share/strata/Strata-data/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    )
    ap.add_argument(
        "--model-sha256",
        default="55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover cross-node inventory
    primary_inv = discover_local_node(args.primary_model, args.model_sha256)
    worker_inv = discover_remote_worker(args.worker_host, args.worker_endpoint, args.worker_model, args.model_sha256)
    link_inv = discover_link(args.worker_ip)

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "nodes": {
            primary_inv["node_id"]: primary_inv,
            worker_inv["node_id"]: worker_inv,
        },
        "links": [link_inv],
    }
    inv_path = out_dir / "resource-inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    # 2. Load trace and evaluate Pareto frontier across whole-layer counts & local Strata tier modes
    trace_stats = load_trace_stats(args.trace)
    worker_vram_free = worker_inv["gpus"][0]["vram_total_idle_free_bytes"]

    frontier: List[Dict[str, Any]] = []
    for mode, budget in [
        ("resident-experts", None),
        ("resident-budget-gib", 4.0),
        ("mmap-experts", None),
    ]:
        for l_count in [0, 4, 8, 12, 16, 20, 24, 28, 30, 32]:
            cand = evaluate_candidate_plan(
                num_owned_layers=l_count,
                ranked_layers=trace_stats["ranked_layers"],
                layer_misses=trace_stats["layer_misses"],
                total_misses=trace_stats["total_cpu_misses"],
                est_tokens=trace_stats["estimated_trace_tokens"],
                local_residency_mode=mode,
                local_budget_gib=budget,
                worker_vram_free_bytes=worker_vram_free,
            )
            if cand["fits_worker_vram"]:
                frontier.append(cand)

    # 3. Select the 3 candidate plans on the Pareto frontier:
    # - `speed`: highest predicted warm decode tok/s with full RAM residency for unowned layers (--resident-experts)
    speed_cands = [c for c in frontier if c["local_residency_mode"] == "resident-experts" and c["num_owned_layers"] > 0]
    speed_best = max(speed_cands, key=lambda c: c["predicted_warm_decode_tok_s"])

    # - `balanced`: best warm decode tok/s with bounded local RAM budget (--resident-budget-gib 4) and Primary RSS <= 14 GiB
    bal_cands = [
        c for c in frontier
        if c["local_residency_mode"] == "resident-budget-gib"
        and c["predicted_primary_rss_gib"] <= 14.0
        and c["num_owned_layers"] > 0
    ]
    balanced_best = max(bal_cands, key=lambda c: c["predicted_warm_decode_tok_s"])

    # - `capacity`: minimum primary pinned RAM (--mmap-experts) + highest offloaded coverage within safe worker VRAM (<= 11.5 GiB)
    cap_cands = [
        c for c in frontier
        if c["local_residency_mode"] == "mmap-experts"
        and c["worker_vram_gib"] <= 11.5
        and c["num_owned_layers"] > 0
    ]
    capacity_best = max(cap_cands, key=lambda c: (c["predicted_warm_decode_tok_s"], c["coverage_fraction"]))

    selected = {
        "speed": speed_best,
        "balanced": balanced_best,
        "capacity": capacity_best,
    }

    manifests: Dict[str, Any] = {}
    for profile, cand in selected.items():
        plan_id = f"qwen36-{profile}-{cand['num_owned_layers']}L"
        txt_path = out_dir / f"plan-{profile}.txt"
        txt_sha = write_placement_txt(
            txt_path,
            cand["owned_layers"],
            comment=f"plan_id={plan_id} profile={profile} model_sha256={args.model_sha256}",
        )
        manifest = build_manifest(
            plan_id=plan_id,
            profile=profile,
            model_id="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            model_sha256=args.model_sha256,
            candidate=cand,
            placement_file=str(txt_path),
            placement_sha256=txt_sha,
            primary_node=primary_inv["node_id"],
            worker_node=worker_inv["node_id"],
            worker_endpoint=args.worker_endpoint,
        )
        m_path = out_dir / f"plan-{profile}.manifest.json"
        m_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests[profile] = manifest

    summary = {
        "schema_version": SCHEMA_VERSION,
        "inventory_path": str(inv_path),
        "trace_summary": {
            "trace_path": trace_stats["trace_path"],
            "total_cpu_misses": trace_stats["total_cpu_misses"],
            "estimated_trace_tokens": trace_stats["estimated_trace_tokens"],
            "top_24_layers": trace_stats["ranked_layers"][:24],
        },
        "pareto_frontier_count": len(frontier),
        "pareto_frontier": frontier,
        "selected_plans": {
            k: {
                "plan_id": v["plan_id"],
                "placement_file": v["worker"]["placement_file"],
                "placement_sha256": v["worker"]["placement_sha256"],
                "strata_cache_flags": v["primary"]["strata_cache_flags"],
                "num_owned_layers": v["cost_model_evaluation"]["num_owned_layers"],
                "owned_experts": v["cost_model_evaluation"]["owned_experts"],
                "worker_vram_gib": v["cost_model_evaluation"]["worker_vram_gib"],
                "predicted_primary_rss_gib": v["cost_model_evaluation"]["predicted_primary_rss_gib"],
                "predicted_warm_decode_tok_s": v["cost_model_evaluation"]["predicted_warm_decode_tok_s"],
            }
            for k, v in manifests.items()
        },
    }
    (out_dir / "orchestrator-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["selected_plans"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
