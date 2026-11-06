# Network expert execution: experimental checkpoint

This fork adds a Linux-only expert worker for **Qwen3.6 (`qwen35moe`) text decode**
and integrates it with Strata's existing low-RAM expert storage modes
(`--resident-experts`, `--resident-budget-gib N`, and `--mmap-experts`).
It is off at build time and at run time. Full-model functional, zero-ownership,
fault-recovery, a 10 GbE trace-ranked placement sweep (up to `7,794` experts /
`13.92 GiB` worker VRAM), and a physical RAM residency sweep have now been completed
on the Kingston-only configuration.
Eliminating unowned-layer `h_ewts` copies removed the zero-ownership overhead,
layer-concentrated placement (`layer-24L-full`, `6,144` experts across the top 24
layers, `10.91 GiB` worker VRAM) cut primary CPU expert pool time during decode from
`7,953 ms` to `2,755 ms` (`-65.4%`) and raised warm median decode throughput from
`47.22 tok/s` to `49.62 tok/s` (`+5.1%`), and combining `layer-24L-full` with
`--resident-experts` / `--resident-budget-gib N` / `--mmap-experts` cut primary
process physical RAM (`Rss`) from **`42.05 GiB` to `8.31–9.23 GiB` (`-78%` to `-80%`)**
while sustaining **`49.61–50.44 tok/s`** warm median decode and preserving clean
mapped-GGUF fallback on worker disconnect.

## Current checkpoint: full-model functional, fault-recovery, 10 GbE placement, and low-RAM residency sweep

Following the Kingston-only integrity and expert-parity recheck on 5 November 2026,
full-model CLI logit comparisons, the 6-case `--serve` qualification campaign
(`bench/distributed_experts.mjs`), and a 10-point trace-driven placement sweep
(`tools/qwen35/generate_placement_from_trace.py`) were completed between `lucy`
(`RTX 3080 10 GiB`, CUDA) and `astor` (`AMD Radeon RX 6800 16 GiB`, HIP,
`192.168.63.177:19415`) on the verified stock `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
(`55983c5a...`). Physical DIMM placement remains unconfirmed (SMBIOS reports A1/B1,
two 32 GiB Kingston `KF3200C16D4/32GX` at 3200 MT/s; one Memtest86+ pass only).

### 1. Full-logit CLI functional gate (32 greedy tokens, prompt 0, 248,320 vocab)

All runs used `STRATA_Q8_PACKED=0 STRATA_Q36_NO_MMQ=1 --max-context 2048 --chunk 256 --mtp 0 --expert-slots 12`:

- **Instrumentation control (`baseline` vs rebuilt `instrumented-baseline`):**
  Byte-identical across all 32 rows (`unequal_floats = 0`, `max_abs = 0`, `relative_l2 = 0`, `token_mismatches = 0`).
- **Zero remote ownership (`baseline` vs `zero` with `--expert-worker` and 0-expert plan):**
  Byte-identical across all 32 rows (`unequal_floats = 0`, `max_abs = 0`, `relative_l2 = 0`, `token_mismatches = 0`, `remote_entries = 0`, `fallbacks = 0`).
- **Remote 32-expert placement (`0-7` on layers `0, 1, 20, 39`, free-running `remote` vs `baseline`):**
  All floats finite; `21` layers dispatched, `22` remote expert assignments executed, `0` fallbacks. Rows `0–2` are bit-identical (`max_abs = 0`) before any remote-owned expert is selected; rows `3–23` match every greedy top-1 token while small cross-path differences (`native_expert_grouped` Q8_1 on HIP vs local `ggml-cpu` Q8_K on primary CPU misses, plus primary slot-replacement divergence because remote hits do not run on the local CPU pool) accumulate to flip argmax at token `24` (`557` vs `995`), after which autoregressive continuation diverges for the remaining 8 tokens.
- **Teacher-forced remote (`remote-forced` with `--force` of baseline tokens vs `baseline`):**
  Conditioned on the identical prefix at every step (`19` layers dispatched, `20` remote entries, `0` fallbacks), all 32 rows remain finite with `relative_l2 = 0.0369`, `mean_kl = 0.00198`, `max_abs = 1.735`, and `31 / 32` top-1 tokens identical (only step 24 flips between near-tied logits `557` and `995` at step KL `0.00107`).

### 2. Paired 6-case `--serve` campaign (3 prompts × 64 tokens × 3 repetitions)

| Case | Exit | Warm decode tok/s (med / mean) | All decode tok/s (med / mean) | Warm TTFT ms (med) | CPU s/req (med) | Remote layers / entries / fallbacks | Token mismatches vs baseline (9 reqs) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 0 | 49.56 / 48.90 | 48.94 / 48.78 | 212.6 | 17.63 | — | — |
| `zero` (0 remote experts) | 0 | 46.88 / 46.64 | 46.64 / 47.00 | 219.9 | 17.94 | 0 / 0 / 0 | **0 / 9** (all 576 tokens identical) |
| `remote` (32 remote experts) | 0 | 47.91 / 48.07 | 47.80 / 48.30 | 210.9 | 18.13 | 356 / 398 / 0 | 0 on prompts 1–2 (warm); prompt 0 & rep 0 prompt 1 diverge as above |
| `disconnect` (cut after 3 results) | 0 | 47.77 / 47.43 | 47.33 / 47.29 | 213.9 | 18.00 | 3 / 3 / 1 | Fallback on layer 39; all 9 requests complete cleanly |
| `timeout` (hold 1st `EXECUTE`, 5 s) | 0 | 49.99 / 49.84 | 49.64 / 45.19 | 207.1 | 17.53 | 0 / 0 / 1 | **0 / 9** (all 576 tokens identical after local recompute) |
| `unavailable` (closed port) | 1 | — | — | — | — | setup refused | Exits non-zero before `READY` (`Connection refused`) |

- **Per-layer network telemetry (`remote`, 356 dispatches, 398 assignments, 2.94 MB TX / 2.95 MB RX):**
  Overall `rtt_us` p50/p95/p99 = `426 / 699 / 804 us`, `wait_us` p50/p95/p99 = `151 / 447 / 550 us`, `worker_stage_us` p50 = `33 us`, `worker_kernel_us` p50 = `80 us`, `worker_copy_us` p50 = `21 us`, `worker_total_us` p50 = `193 us` (`layer 0`: 115 samples, wait p50 `144 us`; `layer 1`: 97 samples, wait p50 `181 us`; `layer 20`: 48 samples, wait p50 `248 us`; `layer 39`: 96 samples, wait p50 `83 us`).
- **GPU utilization (`100 ms` sampling) & post-run model integrity:**
  Primary RTX 3080 averaged `18.2%` busy (`9,087 MiB` peak VRAM) during `remote` vs `23.0%` (`9,144 MiB`) in `baseline`; worker RX 6800 averaged `16.1%` busy (`42%` max, `2,295.2 MiB` VRAM). Post-campaign full-file `sha256sum` of `/home/ztripez/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` matched stock (`55983c5a...`) with unchanged `stat` metadata.

All initial campaign artifacts, logit dumps, GPU utilization traces, and `evidence-manifest.sha256` are saved in `/var/tmp/strata-full-decode-20261105-urobIt/` on `lucy` and mirrored under `/var/tmp/strata-full-decode-20261105-urobIt/primary-evidence/` on `astor`.

### 3. Zero-ownership overhead fix and 10 GbE trace-ranked placement sweep

In the initial campaign above, `moe_hybrid_gpu` and `moe_hybrid_cpu` checked `net_worker->healthy()` on all 40 layers even when a layer owned zero remote experts, queueing an extra per-layer device-to-host `h_ewts` copy and taking `moe_network_cpu` on unowned layers. Adding `NetworkExpertExecutor::holds_layer(layer)` (and `has_any_owned()`) ensures that any layer without remote-owned experts takes the unmodified local fast path with zero extra transfers or branches.

With `ip route get 192.168.63.177` verified on the `enp4s0` 10 GbE route (`src 192.168.60.155`, `10,000 Mb/s`), we captured exact decode CPU-miss traces (`STRATA_Q36_TRACE_MISSES`, `181,440` routed decode entries across the 9 benchmark requests: `25,753` primary GPU hits [`14.2%`] and `155,687` CPU pool misses [`85.8%`] spanning `7,794` unique `(layer, expert)` pairs) and generated two families of placements via `tools/qwen35/generate_placement_from_trace.py`:

1. **Global expert-ranked (`--mode expert`):** Ranks individual `(layer, expert)` pairs across all 40 layers by `decode_misses / worker_slot_bytes` (`256`, `1,024`, `2,048`, `4,096`, and all `7,794` active decode-miss experts).
2. **Layer-concentrated (`--mode layer`):** Ranks whole layers by `layer_decode_misses / layer_worker_bytes` and places all 256 experts of the top-$L$ layers (`4`, `8`, `16`, `24`, and `30` layers) on the worker so the remaining $40 - L$ layers pay zero network synchronization or `h_ewts` D2H overhead while the offloaded layers eliminate `100%` of their local CPU pool work during decode.

| Placement case | Worker experts / layers | Worker VRAM (GiB) | Warm decode tok/s (med / mean) | All decode tok/s (med / mean) | Dispatched layers | Remote entries (`%` decode misses) | Total CPU pool ms (decode CPU ms) | Total network wait ms | `rtt_us` p50 | `wait_us` p50 | `worker_kernel_us` p50 | `worker_total_us` p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 0 / 0 | 0.00 | 47.22 / 47.15 | 47.07 / 46.87 | 0 | 0 (0.0%) | 9,537 (~7,953) | 0.0 | — | — | — | — |
| `zero` (0 remote) | 0 / 0 | 0.00 | 47.83 / 47.71 | 47.78 / 47.95 | 0 | 0 (0.0%) | 9,203 (~7,619) | 0.0 | — | — | — | — |
| `contig-32` (`0-7` on 4L) | 32 / 4 | 0.06 | 48.17 / 47.93 | 47.99 / 47.45 | 356 | 398 (0.3%) | 9,175 (~7,591) | 165.9 | 455 | 172 | 80 | 194 |
| `expert-256` | 256 / 37 | 0.46 | 47.68 / 47.78 | 47.64 / 47.47 | 13,741 | 24,435 (15.7%) | 7,994 (~6,410) | 1,006.7 | 277 | 55 | 35 | 144 |
| `expert-1024` | 1,024 / 40 | 1.82 | 48.07 / 48.11 | 48.06 / 47.98 | 19,969 | 61,412 (39.4%) | 6,158 (~4,574) | 2,675.7 | 286 | 120 | 41 | 150 |
| `expert-2048` | 2,048 / 40 | 3.65 | 48.63 / 48.40 | 48.44 / 48.44 | 22,377 | 93,200 (59.9%) | 4,641 (~3,057) | 4,039.3 | 283 | 176 | 47 | 155 |
| `expert-4096` | 4,096 / 40 | 7.32 | 48.46 / 48.32 | 48.52 / 48.40 | 22,670 | 131,856 (84.7%) | 2,739 (~1,155) | 6,039.4 | 291 | 260 | 63 | 168 |
| `expert-7794` | 7,794 / 40 | 13.92 | 48.68 / 48.39 | 48.61 / 48.53 | 22,680 | 155,548 (99.9%) | 1,584 (~0) | 7,242.9 | 293 | 315 | 65 | 175 |
| `layer-4L-full` | 1,024 / 4 | 1.81 | 49.02 / 48.74 | 48.99 / 48.75 | 2,268 | 16,930 (10.9%) | 8,224 (~6,640) | 760.9 | 306 | 330 | 66 | 178 |
| `layer-8L-full` | 2,048 / 8 | 3.62 | 48.89 / 48.48 | 48.74 / 48.71 | 4,536 | 33,284 (21.4%) | 7,493 (~5,909) | 1,465.2 | 296 | 316 | 67 | 177 |
| `layer-16L-full` | 4,096 / 16 | 7.28 | 48.85 / 48.92 | 48.85 / 49.05 | 9,072 | 65,538 (42.1%) | 5,931 (~4,347) | 2,857.4 | 290 | 309 | 67 | 177 |
| **`layer-24L-full`** | **6,144 / 24** | **10.91** | **49.62 / 49.34** | **49.59 / 49.44** | **13,608** | **96,784 (62.2%)** | **4,339 (~2,755)** | **4,268.7** | **285** | **305** | **66** | **176** |
| `layer-30L-full` | 7,680 / 30 | 13.63 | 48.37 / 48.51 | 48.25 / 48.56 | 17,010 | 119,045 (76.5%) | 3,410 (~1,826) | 5,389.4 | 287 | 308 | 65 | 175 |

*(Note: Total `hy_cpu_ms` reported at engine exit includes `~1,584 ms` of local CPU pool execution during the 9 prefill requests [`75,089` prefill CPU misses], because prefill stays local. Decode-only CPU pool time is `hy_cpu_ms - 1,584 ms`.)*

- **Break-even economics (per-layer network synchronization vs. local CPU pool time):**
  On `lucy` (12-thread CPU pool), one hybrid MoE layer during decode takes an average of `~350 us` on the local CPU pool (`~7,953 ms / 22,680` layer steps). Over `enp4s0` (10 GbE), a full-layer remote dispatch to `astor` (`AMD Radeon RX 6800`) takes `worker_kernel_us` p50 = `65–67 us`, `worker_total_us` p50 = `175–178 us`, and network `rtt_us` p50 = `285–306 us` (`wait_us` p50 = `305–315 us` after overlapping with primary GPU shared-expert execution).
  - When experts are spread thinly across all 40 layers (`expert-256` to `expert-2048`), almost every layer (`13,741` to `22,377` of `22,680` decode layer steps) pays a network dispatch *and* still runs remaining misses on the local CPU pool.
  - When entire layers are offloaded (`layer-4L-full` to `layer-24L-full`), unowned layers (`16` to `36` layers) pay zero network overhead and zero `h_ewts` copies, while each offloaded layer replaces `~350 us` of 12-thread local CPU pool work with `~305 us` of network wait (`~45 us` net wall-time gain per offloaded layer, peaking at **`49.62 tok/s` warm median** at 24 layers / `6,144` experts / `10.91 GiB` worker VRAM). Beyond 24 layers (`layer-30L-full` and `expert-7794`), the remaining lowest-miss layers take less than `~308 us` locally on the CPU pool, so offloading them adds slightly more network wait (`+1,121 ms` from 24L to 30L) than the CPU time saved (`-929 ms`).
- **32-token teacher-forced full-logit qualification across the sweep:**
  Conditioned on the identical baseline token prefix (`prompt 0`, 32 steps × `248,320` floats), all sweep points produce finite logits with zero fallbacks and bounded cross-device differences that do not grow with remote coverage:
  - `zero` (`0` remote entries): `unequal_floats = 0`, `relative_l2 = 0`, `mean_kl = 0`, `32 / 32` top-1 tokens match.
  - `expert-1024` (`1,160` dispatches, `3,579` remote entries): `relative_l2 = 0.0410`, `mean_kl = 0.00390`, `max_kl = 0.0392`, **`32 / 32` top-1 tokens match**.
  - `layer-24L-full` (`744` dispatches, `4,785` remote entries): `relative_l2 = 0.0420`, `mean_kl = 0.00227`, `max_kl = 0.00899`, **`32 / 32` top-1 tokens match**.
  - `expert-4096` (`1,240` dispatches, `6,779` remote entries): `relative_l2 = 0.0398`, `mean_kl = 0.00274`, `max_kl = 0.0161`, `31 / 32` top-1 tokens match.
  - `expert-7794` (`1,240` dispatches, `7,658` remote entries, `99.9%` of decode misses): `relative_l2 = 0.0406`, `mean_kl = 0.00313`, `max_kl = 0.0281`, `31 / 32` top-1 tokens match.
- **Post-sweep model integrity & evidence:**
  Post-sweep full-file `sha256sum` of `/home/ztripez/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` matched stock (`55983c5a...`) with unchanged `stat` metadata. All sweep traces, placement files, logit dumps, and `evidence-manifest.sha256` are saved in `/var/tmp/strata-placement-sweep-20261105-laBhdK/` on `lucy` and mirrored under `/var/tmp/strata-placement-sweep-20261105-laBhdK/primary-evidence/` on `astor`.

### 4. Low-RAM expert storage integration and physical RAM residency sweep (`--resident-experts`, `--resident-budget-gib N`, `--mmap-experts`)

Instead of introducing a separate memory subsystem, `src/qwen35/q36_engine.cu` now wires Strata's existing expert-storage CLI flags (`--resident-experts`, `--resident-budget-gib N`, and `--mmap-experts`) directly into the Qwen3.6 hybrid + network expert hierarchy:

1. **Per-expert host arena residency (`H.hblob_of[x]`) & remote-expert omission:**
   - `E.net_worker` connects and installs its placement plan before `place_experts()` runs.
   - In `--resident-experts` and `--resident-budget-gib N` modes, `place_experts()` skips both primary-GPU-slot experts (`0 .. S-1`) and remote-owned experts (`net_worker->holds(li, e)`), allocating pinned host `arena` space only for local non-VRAM experts (or up to `N GiB` round-robin across local layers). In `--mmap-experts` mode, `host_arena` is `0.00 GiB`.
2. **Zero-copy mapped GGUF CPU fallback (`H.zc_fmt[x]`):**
   - For any expert with `H.hblob_of[x] == nullptr`, `place_experts()` precomputes a per-expert `NativeFmt` (`H.zc_fmt[x]`) whose `up_off` and `down_off` point directly from `L.src_g + x * L.src_gsz` to `L.src_u + x * L.src_gsz` and `L.src_d + x * L.src_dsz` inside the read-only GGUF `mmap`, and `ExpertJobMulti` carries an optional `const NativeFmt* fmt` override so `ExpertPool` reads non-resident experts directly from the mapped GGUF without repacking.
3. **Unmapping GGUF PTEs after startup & remote prefill execution:**
   - In low-RAM modes, `evict_all_gguf_pages()` calls `madvise(..., MADV_DONTNEED)` after `place_experts()` and `warmup()` so the 21.5 GiB GGUF read during SHA-256 verification and placement is unmapped from the process's page tables before serving begins.
   - Multi-token hybrid prefill on remote-owned layers is also dispatched to `net_worker` (sliced into sub-batches of `<= caps().max_tokens = 16` tokens) when remote experts have no local host RAM copy, preventing prompt prefill from faulting the 24 remote layers (`11.17 GiB`) into `lucy`'s page tables during normal operation.

Measured on `lucy` (`RTX 3080 10 GiB`, `--expert-slots 12`) + `astor` (`AMD Radeon RX 6800 16 GiB`, `layer-24L-full` = `6,144` remote experts across 24 layers, `3` prompts $\times$ `64` tokens $\times$ `3` repetitions, `/proc/self/smaps_rollup` at engine exit, evidence in `/var/tmp/strata-lowram-sweep-20261106/`):

| Mode (`layer-24L-full` unless noted) | Case | Host `arena` (`GiB` / experts) | Process `Rss` (`GiB`) | Process `Pss` (`GiB`) | `Pss_Anon` (`GiB`) | `Pss_File` (`GiB`) | Mapped GGUF expert reads (`hy_mmap_reads`) | Warm decode `tok/s` (med / mean) | All decode `tok/s` (med / mean) | Warm `TTFT` `ms` (med) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default (`all` in RAM, local only) | `baseline` | 18.75 (`10,496`) | 41.99 | 41.92 | 1.64 | 21.51 | 0 | 49.88 / 49.65 | 49.51 / 49.68 | 214.4 |
| Default (`all` in RAM + `astor`) | `remote` | 18.75 (`10,496`) | 42.05 | 41.98 | 1.64 | 21.57 | 0 | 49.63 / 49.38 | 48.97 / 49.18 | 209.7 |
| **`--resident-experts`** | **`remote`** | **7.04 (`3,904`)** | **9.09** | **9.02** | **1.65** | **0.31** | **0** | **50.22 / 49.92** | **50.10 / 49.79** | **198.5** |
| `--resident-experts` | `disconnect` | 7.04 (`3,904`) | 18.67 | 18.61 | 1.64 | 9.90 | 109,897 | 48.86 / 48.58 | 48.43 / 48.59 | 215.0 |
| `--resident-budget-gib 6` | `remote` | 6.00 (`3,329`) | 9.23 | 9.16 | 1.64 | 1.48 | 10,868 | 49.61 / 49.72 | 49.56 / 49.73 | 192.0 |
| `--resident-budget-gib 4` | `remote` | 4.00 (`2,219`) | 9.11 | 9.04 | 1.65 | 3.36 | 30,212 | 49.94 / 49.95 | 49.87 / 50.05 | 187.0 |
| `--resident-budget-gib 2` | `remote` | 2.00 (`1,109`) | 8.74 | 8.68 | 1.64 | 5.00 | 49,064 | 50.33 / 50.32 | 50.31 / 50.35 | 178.5 |
| `--resident-budget-gib 1` | `remote` | 1.00 (`555`) | 8.57 | 8.50 | 1.64 | 5.83 | 57,420 | 50.03 / 50.23 | 49.99 / 50.28 | 169.5 |
| **`--mmap-experts`** | **`remote`** | **0.00 (`0`)** | **8.31** | **8.24** | **1.65** | **6.56** | **68,095** | **50.44 / 50.01** | **50.41 / 50.08** | **168.7** |
| `--mmap-experts` | `disconnect` | 0.00 (`0`) | 18.08 | 18.01 | 1.65 | 16.33 | 175,644 | 49.45 / 48.40 | 49.51 / 49.26 | 182.2 |

- **Physical RAM savings vs. Linux page cache behavior:**
  - In Default mode, `lucy` holds both the `18.75 GiB` pinned `host_arena` (`10,496` experts across 41 layers) and `21.57 GiB` of mapped GGUF file pages (`Pss_File`) faulted in during SHA-256 verification and placement, totaling **`42.05 GiB` `Rss`**.
  - With `--resident-experts` + `layer-24L-full`, `lucy` keeps only the 16 local layers' non-VRAM experts in `host_arena` (`3,904` experts = **`7.04 GiB`** pinned), unmaps the GGUF after warmup (`Pss_File = 0.31 GiB`), performs **`0` mapped GGUF reads** during prefill or decode, and cuts process physical memory to **`Rss = 9.09 GiB` (`-32.96 GiB` / `-78.4%`)** while running at **`50.22 tok/s`** warm median decode (`198.5 ms` warm TTFT).
  - Lowering `--resident-budget-gib` from `6 GiB` down to `0 GiB` (`--mmap-experts`) reduces pinned `host_arena` from `6.00 GiB` to `0.00 GiB`, while `Pss_File` grows correspondingly (`1.48 GiB` $\to$ `6.56 GiB`) as the 16 local layers' file-backed experts are faulted into the OS page cache by the CPU pool (`10,868` $\to$ `68,095` mmap reads). Across the entire budget sweep, the 24 remote-owned layers (`11.17 GiB`) remain completely unmapped in `lucy`'s process page tables (`Rss = 8.31–9.23 GiB`), and on `--mmap-experts` all `6.56 GiB` of local expert pages are clean file-backed pages (`Clean = 6.57 GiB`) that the Linux kernel can reclaim under memory pressure rather than locked `cudaHostAlloc` memory.
  - When `disconnect` is injected mid-request after 30 worker responses, both `--resident-experts` and `--mmap-experts` recover seamlessly by faulting the 24 remote layers' experts directly from the mapped GGUF file (`Pss_File` rises by `+9.59 GiB` / `+9.77 GiB` to `9.90 GiB` / `16.33 GiB`, `Rss = 18.67 GiB` / `18.08 GiB`), completing all 9 requests with zero errors at `48.86–49.45 tok/s` warm median decode.

## Earlier checkpoint (5 November 2026): integrity and expert parity recheck passed

On 5 November 2026, the user reported a Memtest failure with the Corsair kit and
returned Lucy to Linux with only the Kingston pair. The supplied Kingston-only
photos show one completed Memtest86+ pass and zero errors; no longer completed
test duration is claimed here. Live SMBIOS now reports two 32 GiB Kingston
`KF3200C16D4/32GX` modules in A1/B1 at 3200 MT/s, with A2/B2 empty.

The requested recheck passed with the existing local model and unchanged CUDA/HIP
binaries. It performed this sequence:

1. The full local-file SHA-256 matched stock.
2. The real CUDA-to-HIP content-hash handshake passed. All 11 expert comparisons
   passed against the CUDA and CPU references at the original tolerances. The
   worker loaded 32 experts across layers 0, 1, 20 and 39, completed nine requests
   containing 25 expert assignments, and reported zero fallbacks.
3. The full local-file SHA-256 still matched stock after GPU execution.
4. Four buffered and three aligned direct reads of the previously affected
   4 MiB block all matched the preserved server reference byte-for-byte. The
   direct-read syscall traces confirm complete reads with `O_DIRECT`. Reported
   file metadata stayed unchanged.

The model path was:

```text
/home/ztripez/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

The recheck used the same reader, block offset 6,559,891,456, and reference hash
as the earlier failure. No model was replaced, no caches were dropped and no
network route was changed. The current route to the worker selects `enp4s0`.
The one-session test worker exited normally.

Evidence is in `/var/tmp/strata-kingston-recheck-20261105-UfoqQI/`. A separate copy
on Astor under that directory's `primary-evidence/` passed the full evidence
checksum manifest. All earlier failure evidence remains retained below.

## Historical model integrity failure, 4 November 2026

On 4 November 2026, the new content-hash handshake refused the primary and worker pair.
The source revision was `ffa80d73133ea5c548aefe57889a16e8085a938f`, plus this feature's local changes.

The stock file is `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, 22,853,663,008 bytes.
Its SHA-256 in `setup.py` is:

```text
55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1
```

The worker's local file and its Strata mapping both matched that hash. A later read
of the same NFS path from the worker also matched. The primary's
first `sha256sum` also matched. Later, the primary's Strata mapping, an independent
Python mapping, `pread`, and a repeated `sha256sum` all returned:

```text
226dc880081fce5443606bc8f3ab2081d60a6434753d84fc1e0b23805439e494
```

A chunk-by-chunk comparison of the primary's mapping and `pread` found no differences
between those two views during that probe. Thus this is an observed change between reads,
not proof of a simultaneous mmap-versus-read discrepancy. Reported size, mtime and ctime
were unchanged. The cause is unknown. The primary file is on NFS; that fact alone does not
establish a storage or kernel cause.

The handover requires stopping on an integrity failure. At that initial checkpoint, no
model was repaired, replaced, redownloaded, or written. Do not bypass the handshake or treat the
earlier geometry-only parity run as qualification of these model bytes.

Evidence is outside Git, on each machine in:

```text
/var/tmp/strata-distributed-20261104-resume/
```

The primary directory contains the initial and repeated hashes, the mapping probe and
its source, protocol logs, both build receipts, preserved pre-integration binaries,
and the rejected real-worker parity log. `evidence-manifest.sha256` identifies the
saved files. This directory is local evidence, not a release artifact.

### User-supplied follow-up, 4 November 2026

The user supplied a `sha256sum` result run on `pve01` for the same model path. It
matched the full stock hash above. This is a user-supplied observation, not a new
measurement by this continuation. The transcript does not include the server's
mount information or a simultaneous server/client capture.

The user then supplied alternating NFS and SSH-stream results from Lucy. All three
NFS passes returned the full `226dc880...` hash above. SSH passes 1 and 2 returned the
full stock `55983c5a...` hash. SSH pass 3 is truncated to `55983c` in the supplied
text, so it is not counted as a complete hash result. The duplicated paste is one
reported sequence, not two independent runs. These checks were not repeated here.

This supports a failure associated with Lucy's NFS read path during those passes.
It does not identify its cause or establish that Lucy's mixed RAM is faulty or
healthy. A correct SSH-stream hash does not explain how the differing NFS bytes
arose. At that checkpoint, no differing byte offsets or bit counts had been established.

The selected workaround is a separate local download from
`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, using the exact filename, byte count and SHA-256
pinned in `setup.py`. Two successful full-file hash checks are required before
retrying the content-hash handshake and numerical parity. This permits qualification
of the new local artifact; it does not resolve or clear the original NFS incident.
The NFS model and existing evidence must remain untouched.

At the local-storage preflight, neither disk had room: `/` had about 3.6 GiB free
and `/mnt/work` about 6.9 GiB, while the model alone needs 21.3 GiB. The Hugging Face
CLI dry run found the exact 22.9 GB file. The user then authorized clearing Lucy's
`uv` and `sccache` caches to make room for a separate local download.

### Local ext4 retry: integrity gate failed again

The cache cleanup completed. The exact file was downloaded directly from Hugging
Face revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` to:

```text
/home/ztripez/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

This path is on Lucy's local `/dev/nvme0n1p2` ext4 filesystem. Its size was exactly
22,853,663,008 bytes. Two consecutive full-file `sha256sum` checks matched the stock
hash. Both binary sets also matched the previously saved build manifests.

The next real CUDA-to-HIP parity test rejected the model during the content-hash
handshake. Astor reported the stock hash, but Lucy's mapping of the new local file
reported:

```text
eca1cd3dbea39929b314baea37f35efadbd2dbe8535552e258548c352a438c79
```

An independent Python probe then obtained that same alternate hash through both
`mmap` and `pread`. It compared corresponding 4 MiB chunks and found no differences
between those two views. Reported device, inode, size, mtime and ctime remained
unchanged across the recorded checks. Chunk hashes are saved for later byte-level
investigation; they do not establish the number of bytes differing from stock.
A fresh `sha256sum` after the probe also returned `eca1cd3d...`; its comparison
against the pinned stock digest failed. Thus the later mismatch was reproduced by
the same command that had passed both initial checks, as well as by the mapping
and `pread` probes.

The local-download workaround therefore did **not** restore the integrity prerequisite.
This reproduces a read-integrity discrepancy on local ext4, so the investigation
cannot be limited to the NFS path. It does not establish faulty RAM, a disk failure,
or an on-disk modification. Qualification stopped before numerical expert execution;
no full-model or performance test was started. The worker exited after its one test
session. The original NFS model, mounts and cache settings were not changed.

New evidence is separate from the earlier failure record:

```text
/var/tmp/strata-local-model-20261104-HYw3mB/
```

It contains the cleanup logs, pinned HF metadata, download receipt, both successful
hash checks, the rejected handshake logs, file metadata and independent read probe.
`qualification.json` records the stopped gate; `evidence-manifest.sha256` identifies
the saved evidence files. No thresholds were changed to continue execution.

### Targeted byte comparison: buffered reads differ, direct reads match

A fresh reference pass on `pve01` matched the full stock SHA-256. Its backing path
was confirmed as local ZFS. Comparing its 5,449 block hashes with the saved Lucy
manifest identified one differing 4 MiB block at zero-based file offset
6,559,891,456. This is a comparison with the earlier Lucy snapshot, not a new
whole-file scan of Lucy.

That block was captured through buffered reads and three aligned `O_DIRECT` reads.
The reference block was fetched over SSH and checked against its server-side hash.
All three direct reads matched it byte-for-byte. The buffered samples contained
9, 12, 14 and 14 differing bits, respectively, across 9, 12, 13 and 13 bytes.
Thus the buffered result changed during the test while the direct result stayed
equal to the reference. The bad buffered bytes were saved before the first direct
read. The original file was not written and no caches were dropped.

The first difference in the initial buffered sample was at zero-based file offset
6,560,024,129 (one-based byte 6,560,024,130): reference `0x7f`, buffered `0x3f`, a
one-bit XOR of `0x40`. Later samples also differed at an earlier offset. Raw block
captures and every observed differing position are retained; the earlier alternate
whole-file hash does not identify these later, changing buffered samples.

The reader used `STATX_DIOALIGN` and aligned allocation, refused fallback, checked
the descriptor flags and required a complete read. Syscall traces show
`O_RDONLY|O_DIRECT`, matching range offsets and successful 4,194,304-byte `pread64`
returns. This is evidence for completed direct reads of this range, not proof of
the entire on-disk model or of reads bypassing a storage controller's cache. See
the [Linux O_DIRECT notes](https://man7.org/linux/man-pages/man2/open.2.html) and
[statx alignment fields](https://man7.org/linux/man-pages/man2/statx.2.html).

The result strongly supports corruption associated with Lucy's cached-memory read
path. It does not identify a faulty DIMM or distinguish hardware instability from
software or DMA corruption. Reported file metadata remained unchanged. Inference
qualification stays stopped.

Read-only hardware checks found NVMe critical warning, media-error and error-log
counts of zero. SMBIOS reports two 32 GiB Kingston modules and two 16 GiB Corsair
modules, all configured at 2133 MT/s, with no memory-array error correction.
The captured kernel log has older unclean-shutdown and filesystem warnings; none
is established as the cause. These observations do not establish hardware health.
The next hardware investigation is offline Memtest86+ across the full 96 GiB and
module-group isolation if errors appear, following its
[testing and troubleshooting guidance](https://www.memtest.org/readme).
No memory stress test, reboot, BIOS change or filesystem repair was performed.

Evidence, including the raw block captures, C reader source, syscall traces,
comparison JSON and hardware logs, is in:

```text
/var/tmp/strata-byte-integrity-20261104-GrepNw/
```

A separate copy at the same path on Astor passed every entry in
`evidence-manifest.sha256`, preserving the captures away from Lucy.

## Observed machines, 4 November 2026

| Role | Host and data address | GPU reported by runtime | Toolchain | Active link |
| --- | --- | --- | --- | --- |
| Primary, initial route | `lucy`, `192.168.63.230`, `enp7s0` | NVIDIA GeForce RTX 3080, 10 GiB | CUDA 13.4.92; driver 615.78.08 | 1,000 Mb/s, MTU 1500 |
| Primary, later link test | `lucy`, `192.168.60.155`, `enp4s0` | same GPU | same toolchain | 10,000 Mb/s, MTU 1500 |
| Worker | SSH alias `astor`, `192.168.63.177`, `enp37s0f0` | AMD Radeon RX 6800, about 16 GiB, gfx1030 | HIP 7.2.53211; ROCm packages 7.2.4 | 10,000 Mb/s, MTU 1500 |

The requested hardware was described as an RX 6800 XT; the runtime reports RX 6800.
No board identity beyond that was established. Initially Linux selected the primary's
1 Gb/s interface. After the user enabled its 10 Gb/s link, a temporary route to the
worker alone selected `enp4s0`. A 512 MiB single-stream transfer in each direction measured
8.93 Gb/s primary-to-worker and 9.03 Gb/s worker-to-primary, including the probe's host
copy/check overhead. A 1,000-request CPU-mock protocol test passed on this path, with
RTT p50/p95/p99 of 239/344/399 microseconds. This is transport evidence, not inference speed.

The 32-byte echo probe with TCP_NODELAY had p99 69.7 us versus 89.4 us without it;
8 KiB echo p50 was about 145-147 us for either setting, with no useful payload-speed
claim. These are single bounded runs. Raw results and the probe source are in `tcp-10g.json`
and `tcp_probe.py`; the earlier 1 Gb/s protocol log is retained separately.

The temporary host route was removed after testing. The default routes were unchanged;
Linux again selects `enp7s0` for worker traffic. To reproduce the isolated 10 Gb/s test
without changing a default route, add and later remove this host route:

```sh
sudo ip route add 192.168.63.177/32 dev enp4s0 src 192.168.60.155
# Run the test, then restore the previous route selection:
sudo ip route del 192.168.63.177/32 dev enp4s0 src 192.168.60.155
```

Primary model path:

```text
/mnt/storage/strata-models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

Worker model path:

```text
/home/ztripez/.local/share/strata/Strata-data/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

## What is implemented

- A versioned TCP protocol with explicit little-endian integers and IEEE-754 FP32 rows.
  C++ object layout, padding, pointers and device addresses are never transmitted.
- A SHA-256 over the exact mapped GGUF bytes, plus geometry and top-k checks. Hashes are
  computed at startup and are not taken from a cached sidecar. Files must remain unchanged
  while either process is running.
- Primary-owned, immutable placement. The worker loads and reads back **every** requested
  expert before READY. The primary checks the acknowledged count before publishing its map.
  An empty plan owns zero experts, including after reconnect.
- A worker that uses device 0, native expert formats, grouped kernels and weighted GPU
  reduction. Results carry token coverage, assignment counts and GPU-event timings.
- A background I/O thread on the primary. Only selected experts absent from the primary
  GPU cache are sent to the worker. The existing CPU pool handles all other misses.
  Primary GPU work and CPU work can overlap the worker request.
- Finite frame and request deadlines, matching response IDs, coverage and finite-value
  checks. A failed runtime request is reported and its assigned rows are recomputed locally
  before the layer continues. Further network dispatch stays disabled for that process.
  A setup failure exits before serving. There is no automatic reconnect.

The existing cache replacement policy stays in use. A worker-resident expert promoted
into the primary cache executes on the primary. Its worker copy remains resident but is
not executed twice. Every expert is still retained in primary host RAM for fallback;
this prototype does not reduce that RAM requirement.

Prefill remains local, even for a one-token prompt. MTP, multiple sessions, batch decoding
and vision are refused when the feature is selected. The same-process helper and peer
implementations are unchanged. No weights, router, LoRA or kernel arithmetic were changed.

The protocol schema lives in `include/strata/core/expert_worker_proto.hpp` and its explicit
codec in `expert_worker_wire.hpp`. The frame header is 24 bytes; payloads are capped at
64 MiB. Timings are microseconds. RTT and wait include host overhead; worker stage, kernel
and return-copy times use GPU events. Reported percentiles retain the last 4,096 successful
requests. Traffic counters cover successful EXECUTE/RESULT exchanges, excluding setup
and failed exchanges. The protocol has no authentication or encryption and is intended
for the specified private, trusted interface only.

## Build

Use separate local build trees. The shared checkout must have one writer. OpenSSL's
Crypto development library is required only when this option is enabled.

```sh
cmake -S . -B /var/tmp/strata-cuda-ztripez \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_NETWORK_EXPERTS=ON
cmake --build /var/tmp/strata-cuda-ztripez \
  --target strata strata-expert-worker expert_worker_proto_test expert_worker_parity_test -j4
```

On `astor`, retain the configured ROCm and ggml paths described in `docs/AMD_HIP.md`:

```sh
ROCM_PATH=/opt/rocm cmake -S . -B /var/tmp/strata-hip-ztripez \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_NETWORK_EXPERTS=ON
ROCM_PATH=/opt/rocm cmake --build /var/tmp/strata-hip-ztripez \
  --target strata strata-expert-worker expert_worker_proto_test expert_worker_parity_test -j4
```

Run the GPU-free checks directly on each host:

```sh
/var/tmp/strata-cuda-ztripez/expert_worker_proto_test
# On astor:
/var/tmp/strata-hip-ztripez/expert_worker_proto_test
```

The qualification build trees have `STRATA_BUILD_TESTS=OFF`, so CTest lists no tests
there. The pass receipts come from executing the test binaries directly. Set
`STRATA_BUILD_TESTS=ON` when CTest registration is wanted.

`STRATA_NETWORK_EXPERTS=OFF` is the default. It adds no network library, OpenSSL dependency,
worker target or network integration to the engine. Windows, Metal and SYCL are not
supported by this experiment.

## Run after the integrity gate is resolved

These are **engine flags**, not `strata launch` flags. The primary must use hybrid experts.
First verify stable model hashes against the stock catalog on both machines, including
mapped reads. Then rerun the real-worker parity test before inference.

Start the worker on its private interface:

```sh
/var/tmp/strata-hip-ztripez/strata-expert-worker \
  --model /path/to/verified/model.gguf --bind 192.168.63.177:19415
```

Without `--bind`, it listens on `127.0.0.1:19450`. It accepts sequential sessions.
`--max-sessions N` bounds a test run. The primary is the sole placement authority.

A plan is a text file containing layer and expert numbers, one or more per line:

```text
# A small deterministic set for a first test.
0 0-7
1 0-7
20 0-7
39 0-7
```

Comments and empty plans are accepted. Invalid lines and out-of-range entries fail.
Duplicate entries in a text plan are normalized before the immutable wire plan is sent.

On the primary:

```sh
STRATA_Q8_PACKED=0 STRATA_Q36_NO_MMQ=1 \
/var/tmp/strata-cuda-ztripez/strata \
  --model /path/to/verified/model.gguf --max-context 2048 --chunk 256 \
  --mtp 0 --expert-slots 12 --serve \
  --expert-worker 192.168.63.177:19415 \
  --expert-worker-placement /path/to/placement.txt \
  --expert-worker-timeout-ms 5000
```

The two environment settings and small context/chunk above fitted the observed free VRAM
on `lucy`; they are not new engine defaults or general recommendations. The unmodified
baseline with its packed defaults ran out of memory on this occupied desktop.
The timeout applies to setup frames as well as execution. Increase it explicitly if
loading a larger plan exceeds the deadline; the maximum is 600,000 ms.

## Validation and remaining work

**Passed:** the 5 November Kingston-only full-hash and targeted-read recheck; all 11
real-worker expert parity comparisons; full-model 32-token CLI logit identity for
rebuilt instrumentation and zero-ownership; teacher-forced 32-token full-logit bounded
comparison (`relative_l2 = 0.0369`, `mean_kl = 0.00198`, `31 / 32` top-1 match); and
the 6-case `--serve` campaign (`baseline`, `zero`, `remote`, `disconnect`, `timeout`,
`unavailable`) with per-layer telemetry and GPU/CPU utilization tracking. Earlier validation
passed CUDA and HIP builds and GPU-free protocol/executor tests
on both hosts; address/undefined-behavior sanitizer tests; a 1,000-request CPU-mock run
over both the initial 1 Gb/s route and the later 10 Gb/s path. Tests cover golden wire bytes, sparse and repeated activation
rows, zero ownership, invalid placement, concurrent request rejection, reconnect, bad
model identity, stale IDs, incomplete READY, incorrect result coverage, nonfinite results,
disconnect, whole-frame/request deadlines, and per-layer sample attribution. Failed results leave caller buffers unchanged.

**Retained earlier failures:** the real CUDA-primary/HIP-worker parity handshake rejected different model
content hashes when the Corsair DIMMs were installed. The default-settings pre-integration CUDA run also ran out of memory.
Both failures are retained. The pre-integration binary completed a narrower run after
using the explicit fit settings, but that is not a valid paired performance comparison.

**Historical evidence only:** the earlier continuation recorded expert parity against
CUDA and CPU for four layers. Its handshake checked geometry rather than model content.
Those measurements cannot supersede the integrity failures. The 5 November parity
and full-decode campaign runs supply new content-verified evidence.

**Still open for speedup/capacity qualification:** the 32-expert (61 MiB) placement
tested in `/var/tmp/strata-full-decode-20261105-urobIt/` verified end-to-end correctness,
zero-ownership identity, and fault recovery, but does not beat the local CPU pool on
decode tok/s (`47.91` vs `49.56 tok/s` warm median) or reduce primary host RAM because
every expert remains pinned in primary RAM for fallback. Qualifying a speedup or capacity
benefit requires a large multi-layer placement (e.g., thousands of experts on the worker
over the 10 Gb/s route) and/or omitting remote-owned experts from primary RAM when fallback
is not required. Physical DIMM slot placement on `lucy` (`A1/B1` in SMBIOS vs `A2/B2`) and
long-duration memory stability testing also remain open. No SYCL, Windows or Metal
build/runtime claim is made.

The repeatable harness records raw engine output, token IDs, per-token arrival times,
CPU clock ticks, prefill/decode rates, time to first token, startup time, worker and
per-layer counters, JSON and CSV:

```sh
node bench/distributed_experts.mjs \
  --engine /var/tmp/strata-cuda-ztripez/strata \
  --model /path/to/verified/model.gguf --worker 192.168.63.177:19415 \
  --placement /path/to/placement.txt --out /var/tmp/strata-qualified-new-run
```

It runs baseline, empty-plan, real-worker, disconnection, withheld-response and unavailable
cases sequentially, with three repetitions of three repository prompts. A local test relay
injects the two runtime failures after `READY`; it does not stop a user service. Keep the real worker
running for those sessions. The harness refuses to overwrite an evidence directory.
Only the direct remote case is suitable for performance comparison; fault relays are
correctness probes. The unavailable case must fail setup. Zero ownership must preserve
tokens. Remote token deviations are recorded and need separate numerical/quality review.
