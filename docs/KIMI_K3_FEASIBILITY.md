# Historical Kimi K3 feasibility draft

**Model ID:** `moonshotai/Kimi-K3`  
**Pinned Hugging Face Revision (`sha`):** `f831ab66814297da540d832a5235f8e904f29d06` (`2026-09-02T02:22:41Z`)  
**License:** `other` (`license_name: kimi-k3`, official `LICENSE` file in repository)  
**Status:** Research draft only. No Kimi K3 model execution is established by this report. Hardware rates, throughput estimates and viability verdicts below are unvalidated planning assumptions, not benchmark results or current acceptance criteria. Performance is deferred under the current vertical-first objective. Kimi K2 execution is documented separately in [KIMI_K2.md](KIMI_K2.md).

---

## 1. Official Model Geometry & Sharded Storage Footprint

Verified directly against `https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json` and the Hugging Face model index (`96` safetensors shards, `model-00001-of-000096.safetensors` through `model-00096-of-000096.safetensors`):

| Parameter | Official Value (`config.json` / Safetensors Index) | Notes |
|---|---|---|
| **Architecture class** | `KimiK3ForConditionalGeneration` / `KimiLinearForCausalLM` | Hybrid KDA + Gated MLA + Latent MoE + Vision Tower |
| **Total parameters** | `2,779,931,837,184` (~2.78T / 2.8T class) | ~104B active parameters per token |
| **Total repository storage (`usedStorage`)** | `1,561,018,243,668` bytes (**1.561 TB** decimal / **1.420 TiB** binary) | Split across 96 `.safetensors` shards (~16.26 GB/shard) |
| **Text hidden dimension (`hidden_size`)** | `7,168` | Vocabulary size `163,840` (`bos=163584`, `eos=163586`) |
| **Layers (`num_hidden_layers`)** | `93` (`first_k_dense_replace = 1`) | Layer 0 is dense (`intermediate_size = 33,792`); Layers 1..92 (`92` layers) are MoE |
| **Attention hybrid schedule** | `69` KDA layers + `24` Full Gated-MLA layers | Full MLA on layers `4, 8, 12, ..., 92, 93`; KDA on remaining 69 layers |
| **Block Attention Residuals** | `attn_res_block_size = 12` | Cross-layer block attention residual connections |
| **Routed experts per MoE layer (`num_experts`)** | `896` (`82,432` routed experts across 92 MoE layers) | `num_experts_per_token = 16` (`top-16` routing, `sigmoid`, `noaux_tc`, `moe_renormalize = true`) |
| **Shared experts per MoE layer** | `2` (`num_shared_experts = 2`) | Unquantized BF16 in official checkpoint |
| **Latent MoE dimensions** | `routed_expert_hidden_size = 3,584`, `moe_intermediate_size = 3,072` | `latent_moe_use_norm = true` (projects `7168 -> 3584` before routed experts, `3584 -> 7168` after) |
| **FFN activation (`hidden_act`)** | `situ` (SiTU-GLU) | `activation_situ_beta = 4.0`, `activation_situ_linear_beta = 25.0` |
| **Context length (`max_position_embeddings`)** | `1,048,576` (1M tokens) | MLA KV LoRA rank `512`, Q LoRA rank `1536`, `96` heads |

### Exact Tensor Precision & Byte Breakdown

From the official safetensors parameter index:
- **`F32` tensors:** `11,122,432` parameters = **`44.49 MB`** (norms / scalars).
- **`BF16` tensors (unquantized spine):** `57,179,884,544` parameters = **`114.36 GB`** (`106.51 GiB`).
  - Per `quantization_config.ignore`, all `self_attn.*` (KDA + Gated MLA), `shared_experts.*`, dense `mlp.*` (Layer 0), latent MoE projections/norms, `lm_head`, `embed_tokens`, `vision_tower.*` (27 layers), and `mm_projector.*` remain in **BF16**.
- **`U8` (`mxfp4-pack-quantized` routed experts):** `2,722,740,830,208` packed bytes/elements = **`1,446.61 GB`** (`1,347.26 GiB`) including group-32 E8M0 `uint8` scales.
  - Each single routed expert (`3584 -> 3072` gate/up, `3072 -> 3584` down) has $3 \times 3584 \times 3072 = 33,030,144$ weights. At MXFP4 (`17` bytes per `32` weights = `0.53125` bytes/weight), one routed expert is **`17,547,264` bytes (`17.55 MB` / `16.73 MiB`)**.
  - Per MoE layer (`896` experts): **`15.72 GB`** (`14.64 GiB`).
  - Per token active routed expert weights (`top-16` $\times$ `92` MoE layers = `1,472` expert evaluations/token): **`25.83 GB`** of routed expert weights touched per decode token!

---

## 2. Format & Operator Compatibility Matrix Against Current Strata (`strata-dsv41`)

| Subsystem / Operator | Kimi K3 Requirement | Current Strata Status (`src/dsv41`) | Gap / Scoped Backend Work Required |
|---|---|---|---|
| **MXFP4 expert weights** | `mxfp4-pack-quantized`, group size 32, symmetric E8M0 `uint8` scales | **Implemented** (`gg::kFmtMXFP4` in `src/dsv41/ds4_gguf.cu` and `ds4_model.cu` GPU + AVX2/AVX-512 CPU lane) | **Adapter needed:** Map `compressed-tensors` safetensors packing layout to `strata-dsv41` `kFmtMXFP4` trellis/block layout and replace SiLU with SiTU-GLU. |
| **SiTU-GLU activation** | `hidden_act = "situ"`, $\beta = 4.0$, $\beta_{\text{lin}} = 25.0$ | **Missing** (Strata currently implements clipped SiLU-GLU in `ds4_model.cu` and `q36_engine.cu`) | **Small kernel change (~60 LOC):** Add `situ_glu(g, u, beta=4.0, lin_beta=25.0)` device and CPU-lane fused activation functor. |
| **Latent MoE wrapper** | Down-project `7168 -> 3584` + RMSNorm (`latent_moe_use_norm`) before routed experts; up-project `3584 -> 7168` after | **Missing** (Current MoE dispatches `hidden_size` directly into experts) | **Medium engine change (~250 LOC):** Add per-layer latent projection GEMMs + RMSNorm around routed MoE dispatch; reduces `NetworkExpertExecutor` wire payload by **2×** (`3584` vs `7168` per token)! |
| **Gated MLA (24 layers)** | `mla_use_nope = true`, `mla_use_output_gate = true`, `kv_lora_rank = 512`, `q_lora_rank = 1536`, `96` heads | **Partial** (`strata-dsv41` implements MLA with `kv_lora_rank=512`, `q_lora_rank=1536`, `qk_nope=128`, `qk_rope=64`, `v=128`) | **Medium kernel change (~300 LOC):** Add output gating (`mla_use_output_gate`) and `96`-head geometry support without Indexer when `ix_topk` is absent. |
| **KDA Linear Attention (69 layers)** | Kernel Delta Attention (`num_heads = 96`, `head_dim = 128`, `short_conv_kernel_size = 4`, `use_full_rank_gate = true`, `gate_lower_bound = -5.0`) | **Missing** (`q36_engine.cu` has Gated DeltaNet for Qwen3.5/3.6, but `strata-dsv41` only has MLA) | **Major backend feature (~1,200 LOC):** Port/implement KDA recurrent state + shortconv4 + full-rank gate kernels in CUDA/HIP and recurrent state checkpoints. |
| **Block Attention Residuals** | `attn_res_block_size = 12` | **Missing** (`strata-dsv41` uses DeepSeek `mHC` 4-stream hyper-connections) | **Medium engine change (~250 LOC):** Replace `mHC` stream mixing with Kimi K3's 12-layer block attention residual accumulator. |
| **Router (`noaux_tc` sigmoid top-16)** | `896` experts, `top-16`, sigmoid + bias (`noaux_tc`), `moe_renormalize = true` | **Mostly implemented** (`strata-dsv41` supports sigmoid + bias `noaux_tc` routing up to `1024` experts; `kPicks` is currently `8`) | **Small constant/template update (~80 LOC):** Raise `kPicks` from `8` to `16` and verify `n_exp = 896` in router kernel. |

---

## 3. Hardware & Cluster Feasibility Analysis (`lucy` + `astor` vs Expanded Cluster)

### 3.1 Storage Capacity & I/O Feasibility

1. **Disk Footprint (`1.561 TB` raw safetensors / `~1.50 TB` quantized GGUF):**
   - `lucy` local NVMe (`/` `4.7 GiB` free, `/mnt/work` `6.9 GiB` free) and `astor` local SATA SSD (`/` `23 GiB` free): **Cannot store Kimi K3 locally.**
   - Shared NFS `/mnt/storage` (`4.9 TiB` free): **Can store the `1.56 TB` checkpoint**, *however* cold-reading `25.83 GB` of active routed experts per token over 10GbE NFS (`1.15 GB/s`) would cap cold decode at **$\le 0.044\text{ tok/s}$** (`22.5 seconds/token`).
   -Even with a dedicated local PCIe 5.0 NVMe SSD (`12 GB/s` random/streaming read) on each node, reading `25.8 GB` of uncached experts per token from SSD would limit SSD-bound decode to **$\sim 0.46\text{ tok/s}$**.

2. **Dense & Shared-Expert Spine Footprint (`114.36 GB` BF16):**
   - Unlike DeepSeek V4.1-Flash (`~16.8 GB` BF16 dense, `6.16 GB` in `Q8_0/Q5_K`), Kimi K3 has **`57.18B` active non-routed parameters** (`92` layers $\times$ `2` shared experts + KDA/MLA projections + latent MoE projections + Layer 0 dense FFN):
     - In `BF16`: **`114.36 GB`**
     - Quantized to `Q8_0`: **`~60.8 GB`**
     - Quantized to `Q4_K / MXFP4`: **`~32.5 GB`**
   - **Two-node (`lucy` 10 GiB RTX 3080 + `astor` 16 GiB RX 6800 = `26 GiB` total VRAM) verdict:**
     - Even if the entire dense/shared spine is quantized to 4-bit (`~32.5 GB`), **the dense/shared spine alone exceeds the combined `26 GiB` VRAM of `lucy` + `astor`** before allocating a single KV/KDA state slot or a single routed expert in VRAM!
     - Furthermore, `lucy` + `astor` have `128 GiB` combined DDR4 RAM (`~104 GiB` usable), which holds only **`~7.2%`** (`~104 GB` / `1,446 GB`) of Kimi K3's routed experts.
     - Therefore, on `lucy` + `astor` (`26 GiB` total VRAM, `128 GiB` total DDR4 RAM), full Kimi K3 inference would be **100% SSD/NFS I/O-bound** at **$\approx 0.04\text{ tok/s}$ (over 10GbE NFS)** or **$\approx 0.35\text{ tok/s}$ (if equipped with two local 2 TB PCIe 4.0 NVMe drives at 6 GB/s aggregate)**.

### 3.2 Throughput Estimates Across Hardware Tiers

Using the Section 4 critical-path model (`92` MoE layers, `16` routed experts/layer, `17.55 MB`/expert, `25.83 GB` active routed expert weights/token + `32.5 GB` Q4_K dense/shared spine):

| Cluster Configuration | Total VRAM | Total Host RAM | Storage Tier | Estimated Warm Decode (`tok/s`) | Bottleneck & Viability Verdict |
|---|---|---|---|---|---|
| **Current 2-node (`lucy` RTX 3080 10G + `astor` RX 6800 16G)** | `26 GiB` | `128 GiB` DDR4 (`~90 GB/s` aggregate) | Shared 10GbE NFS (`/mnt/storage`, `4.9 TiB` free) | **`0.04 tok/s`** (~25 s/tok) | **NOT VIABLE for interactive use.** Dense spine (`32.5 GB` at Q4_K) spills out of VRAM; 93% of routed experts miss RAM and read over 10GbE NFS. |
| **Current 2-node + 2× Local 2 TB PCIe 4.0 NVMe SSDs** | `26 GiB` | `128 GiB` DDR4 | 2× Local NVMe (`~10 GB/s` aggregate read) | **`0.35 – 0.45 tok/s`** (~2.5 s/tok) | **Functional batch/offline only.** Eliminates NFS bottleneck, but 93% of `25.8 GB/tok` routed experts still page from NVMe. |
| **Expanded 4-node Consumer Cluster (e.g., 4× 24–32 GiB GPUs + 4× 384 GB DDR5 RAM + 10/25GbE)** | `96–128 GiB` | `1.5 TB` DDR5 (`~600 GB/s` aggregate) | Local NVMe + `ExpertStore` RAM residency (`100%` of `1.45 TB` experts in RAM) | **`8.5 – 12.0 tok/s`** | **VIABLE C-usable & P-usable.** `32.5 GB` Q4_K dense/shared spine + KV/KDA states fit entirely in VRAM; all `82,432` MXFP4 routed experts reside in distributed DDR5 `ExpertStore` across the 4 nodes (~23 layers/node, zero SSD misses in warm decode). |

---

## 4. Isolated Operator & Expert Parity Test Plan (Prerequisite Before Any Full-Model Gate)

Before downloading any `16.26 GB` shard of `moonshotai/Kimi-K3`, backend development must pass four standalone unit/parity gates requiring **zero full-model downloads**:

1. **Test K1 — `kimi_situ_mx_test` (Standalone Latent MoE + SiTU-GLU + MXFP4 Parity):**
   - Synthetic single-expert and 16-expert batch with Kimi K3 dimensions (`hidden = 7168`, `latent = 3584`, `ffn = 3072`, `situ` $\beta=4.0, \beta_{\text{lin}}=25.0$, `MXFP4` weights).
   - Verify CUDA (`sm_86`), HIP (`gfx1030`), and CPU-lane AVX2 outputs against FP32 reference (`max_abs_diff < 5e-3`, `cosine > 0.9995`).
2. **Test K2 — `kimi_router_top16_test` (`noaux_tc` Sigmoid Top-16 over 896 Experts):**
   - Verify exact top-16 expert ID and renormalized sigmoid weight parity between CPU reference, CUDA (`sm_86`), and HIP (`gfx1030`) for `n_exp = 896, k_exp = 16`.
3. **Test K3 — `kimi_kda_mla_test` (Isolated KDA Layer + Gated MLA Layer Parity):**
   - Verify 1 KDA layer (`96` heads, `head_dim = 128`, `short_conv = 4`, full-rank gate with `-5.0` lower bound) and 1 Gated MLA layer (`kv_lora_rank = 512`, `q_lora_rank = 1536`, output gate) across prefill (`T=64`) and single-step decode (`T=1`) against PyTorch `modeling_kimi_linear.py` reference tensors.
4. **Test K4 — `kimi_network_latent_test` (Cross-Node Latent Activation Dispatch):**
   - Verify `NetworkExpertExecutor` sending latent-projected activations (`dim = 3584`, `top_k = 16`) from `lucy` (`sm_86`) to `astor` (`gfx1030`) over 10GbE and combining `3584 -> 7168` outputs with zero numerical drift.
