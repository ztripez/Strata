# Experimental Kimi K2 inference in this fork

This fork completed a short full-model inference with Kimi K2 Instruct across
an NVIDIA RTX 3080 10 GB and AMD RX 6800 16 GB. The native CUDA and HIP owners
executed all 61 layers using the same pinned GGUF checkpoint on shared NFS.

The actual chat prompt contained 27 tokens. Two autoregressive tokens decoded
to **Hi there**. Both remote expert directions executed, with zero fallbacks.
The coordinator, suffix and both expert workers exited 0, and their shared
hardware reservation was released. This was slow experimental bring-up, not
quality, throughput, 2K-context or production qualification.

## Scope

The adapter reuses Strata's sharded GGUF reader, GPU expert cache, CPU expert
pool, network expert executor and whole-layer TCP session implementation.
Dense/MLA operations execute on the layer owner's GPU. CPU/file-backed experts
handle most routed assignments in the current configuration. No new expert
cache, weight pager or model RPC protocol was added.

The source checkpoint is `moonshotai/Kimi-K2-Instruct` at
`fd1984e2b7a3350dbf7305fe73a4ede25c14de50`. The tested quantized pack is
`unsloth/Kimi-K2-Instruct-GGUF`, UD-IQ3_XXS, at
`23bf90d8196ddfcb3328a6f60c99a9bdb5c1dadf`: nine shards totaling
416,622,218,560 bytes. All nine published SHA-256 identities were verified.

## Build and run

The experimental targets are excluded from the default build. Configure the
existing CUDA or HIP backend with `STRATA_NATIVE_EXPERTS=ON` and
`STRATA_NETWORK_EXPERTS=ON`, then build `strata-kimi2-split` and
`strata-kimi2-expert-worker`. The measured run used CUDA on Lucy and a manual
HIP build for gfx1030 on Astor; portable packaging is not implemented.

The split executable accepts:

```text
strata-kimi2-split worker|coordinator PROFILE LAYER_ENDPOINT EXPERT_ENDPOINT CAPTURE_PREFIX --model FIRST_SHARD
```

`PROFILE/token-ids.txt` supplies real token IDs. `max-new-tokens.txt` selects
1–16 generated tokens; absent that file, the original limit is 16. Model and
placement identities are required in the profile. The output token IDs are
written to `CAPTURE_PREFIX.generated-ids.txt`; use the pinned tokenizer to decode.

[The bring-up tools](../tools/kimi-k2/README.md) describe profile preparation,
references and the two-host launcher. That launcher contains the measured
homelab's absolute paths and addresses and requires its canonical hardware
lease. It is a development recipe, not a portable installer or serving endpoint.
Do not launch competing CPU/GPU/model-storage workloads on reserved hosts.

## Evidence and limits

[The short-inference receipt](../evidence/kimi-k2/vertical-inference.json)
records the completed milestone. [Gate status](../evidence/kimi-k2/status.json)
retains the original qualification gates. The short request does not pass the
earlier 256-token/16-token K7 requirement or K8–K10. Raw arrays, binaries,
weights and logs remain outside Git at the paths recorded in receipts.

Longer attempts and their failures are preserved. One earlier attempt exceeded
the prescribed Astor VRAM ceiling. No capacity or performance pass is claimed.
SYCL, Metal, Windows, ordinary CLI/HTTP serving and upstream default-path
regression qualification have not been established for this adapter.

Kimi K2 is distinct from the separate Kimi K3 reference work. This change does
not make K3 an executable Strata model family.
