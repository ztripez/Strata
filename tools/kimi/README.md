> Historical Kimi K3 reference research only. These scripts are not a Strata
> model adapter and do not establish full K3 inference. See [current fabric
> scope](../../docs/FABRIC.md). Local virtual environments and model data are
> deliberately excluded from Git.

# Kimi K3 first-layer reference

This experimental CPU path loads the published first-layer safetensors shard and
runs KDA attention, attention residual mixing, and the dense SiTU MLP in float32.
It does not register Kimi as a supported Strata engine family. It does not yet
execute MLA or routed experts, or use Strata's GPU and distributed executor.

From the repository root, with a local layer-0 shard:

```sh
OPENBLAS_NUM_THREADS=4 uv run --with numpy python tools/kimi/layer0.py /path/to/model-00001-of-000096.safetensors --output /tmp/kimi-layer0.npz
uv run --with numpy --with torch python tools/kimi/compare_layer0.py /path/to/model-00001-of-000096.safetensors /tmp/kimi-layer0.npz
```

The input is three deterministic random hidden-state vectors (seed 42), not
text embeddings. The second command compares the output against a separate
PyTorch formulation using grouped convolution and matrix multiplication. Both
paths use float32 operators and the same real BF16/F32 checkpoint tensors;
this is not a comparison with the full upstream GPU model or BF16 arithmetic.

`A_log` in the released first-layer checkpoint has 96 active head entries and
32 zero padding entries. The loader checks the padding before selecting the
active entries. This follows the loading behavior discussed in
[the model repository](https://huggingface.co/moonshotai/Kimi-K3/discussions/150)
and the architecture documented by
[NVIDIA](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/kimi/kimi-k3.html).

The model config and operator definitions come from
[Moonshot's model repository](https://huggingface.co/moonshotai/Kimi-K3).

Observed on 11 November 2026 on the local CPU, using the published layer-0
shard and the deterministic three-vector input above:

- NumPy forward output: shape `[3, 7168]`, RMS `0.0039089031`.
- PyTorch comparison: maximum absolute difference `4.2928877e-8`;
  relative RMS difference `1.5041524e-6`.
- Comparison assertion passed (`atol=0.002`, `rtol=0.002`).

These observations establish a real-weight first-layer float32 reference path.
They do not establish text generation, GPU kernel parity, model quality, MLA,
MoE execution, or integration with the Strata inference engine.

## First MoE block

`moe.py` evaluates layer 1's MoE sublayer for one deterministic random
7168-wide hidden-state vector. It preserves packed E2M1/E8M0 source tensors
and downloads only the selected experts plus the router, latent projections,
normalization and shared experts. The fixture directory is an offline reference
artifact; this does not add an engine cache, scheduler or placement mechanism.

```sh
OPENBLAS_NUM_THREADS=4 uv run --with numpy python tools/kimi/moe.py /mnt/work/kimi-k3-moe --output /mnt/work/kimi-k3-moe/result.npz
OPENBLAS_NUM_THREADS=4 uv run --with numpy --with torch python tools/kimi/compare_moe.py /mnt/work/kimi-k3-moe /mnt/work/kimi-k3-moe/result.npz --output /mnt/work/kimi-k3-moe/comparison.json
```

The NPZ preserves the input, logits, selected IDs, normalized routing weights,
latent projection, each selected expert's output, weighted reduction, latent
normalization, routed output, shared output and combined output. The comparison
uses independent PyTorch routing, nibble unpacking, scale reconstruction and
matrix operations. It checks the selected expert set before aligning the
unordered top-k results for intermediate comparison.

The packing follows the
[compressed-tensors MXFP4 format](https://docs.vllm.ai/projects/compressed-tensors/en/stable/api/compressed_tensors/compressors/mxfp4/):
two E2M1 values per byte, even column in the low nibble, and an E8M0
bias-127 scale per 32 input columns.

This sublayer run does not include layer 1's KDA attention or its residual
mixing. Its input is synthetic hidden state, not a text-derived activation.
Activation arithmetic is float32; it does not emulate MXFP8 activation rounding.

Observed first-MoE run on 11 November 2026: the selected expert sets matched
exactly, and all nine preserved numerical boundaries passed the comparison.
Combined output maximum absolute difference was `1.3232231e-5`; relative RMS
difference was `1.3733833e-6`. Per-expert output maximum absolute difference was
`5.1021576e-5`. Exact IDs and boundary errors are in `moe-evidence.json`.
Selected routing weights ranged from `0.06239361` to `0.06253541` and
matched the PyTorch values exactly for this input. This is a synthetic hidden
state; routing on text-derived activations remains untested.
