# Distributed fabric checkpoint

Updated 9 November 2026. Source branch `feat/distributed-experts`, HEAD
`3c4f7ff8` before this status update. Continue the approved fabric architecture;
Strata's existing node-local staging remains authoritative.

## Qwen: bounded tiered-worker milestone completed

The opt-in worker can acknowledge more experts than fit in GPU memory and use
the native CPU pool for RAM and mapped GGUF experts. All 10,240 Qwen experts
were assigned with 8 MiB expert-VRAM and 8 MiB host-RAM budgets. All three paths
passed the bounded expert fixtures. Full-model functional and failure tests
completed; output-quality qualification remains open. See
[DISTRIBUTED_EXPERTS.md](DISTRIBUTED_EXPERTS.md).

## DeepSeek: stock HIP smoke passed; CUDA primary remains blocked

The pinned ISTA-DAS 2.00-bpw seven-file set (256,816,878,701 bytes) is downloaded
in `/mnt/storage/strata-models/deepseek41-ist-das-2.00bpw`. All seven files
passed their source SHA-256 manifest through Lucy's NFS mount. Astor can read
the store, but an independent full-file checksum pass on Astor is not claimed.
The earlier HTTP downloader failure and the successful Xet retry are retained.

Strata's plain official pack recipe created `maya-ds4` beside those files,
without rewriting weights or changing quantization. It includes Engram,
DSpark metadata and tokenizer. `strata-dsv41 --pack-info` reports all six
referenced weight files present and long enough. Pack construction and
structural checks alone are not inference qualification.

On Astor's RX 6800 (`gfx1030`), the rebuilt stock HIP engine returned:

> The capital of France is Paris.

It exited 0 at EOS after eight tokens, using six routed experts, context 1024,
DSpark disabled, text only, a read-only shipped profile, mapped experts and a
4 GB existing RAM-store budget. It reported 962 experts / 8.60 GB in VRAM,
14,398 experts on the CPU lane and zero failed store reads. This establishes
one narrow stock-model end-to-end smoke, not broad quality or speed acceptance.
The cold NFS-backed prompt took 119.8 seconds and decode 61.8 seconds; compiler
and fixture work overlapped portions of the run, so these are diagnostic only.

Lucy failed before generation in two stock CUDA configurations: context 1024,
and context 256 with windowed prefill and GPU expert caching disabled. Both
reported a failed 433 MiB allocation. The engine estimated 5.96 GB of dense
weights versus about 6.3-6.4 GB available. The earlier 16.8 GB BF16-layout
figure is not the dense GPU footprint of this pinned quantized pack. No desktop
applications were stopped. This is a current-free-VRAM blocker, not proof that
an otherwise idle RTX 3080 cannot run it.

HIP IQ expert fixtures and router guards passed. The general GGUF fixture
failed its IQ4_XS/Q4_K token-versus-window bit-identity contract; both outputs
individually passed the fixture's numerical-reference check. That expert-format
pair is absent from this pinned pack. The full dense fixture exceeded its
120-second budget; it is not a pass. Focused dense checks and an idle GGUF
repeat are recorded separately in the evidence directory. The focused
`--quick --type 13,8 --kernel default` dense run passed with `SUMMARY fails=0`.
The idle general-GGUF repeat reproduced the same single bit-identity failure.

Evidence: `/var/tmp/strata-deepseek-feasibility-20261109-BA9x8e/` on both hosts.
It retains the invalid missing-prompt launch, both CUDA allocation failures,
pack output, commands/settings, binary identities and raw HIP test results.

G3 is partial: a stock HIP smoke exists, while the intended CUDA-primary stock
baseline remains blocked and backend test limits remain explicit. G4 distributed
DeepSeek inference has not been implemented or qualified. The current network
worker remains Qwen-specific; DeepSeek needs its own adapter reusing the
existing DeepSeek kernels and store, not a new cache or transport.

## Other unfinished work

`tools/fabric/orchestrator.py` and `docs/KIMI_K3_FEASIBILITY.md` are untracked
work inherited from an earlier continuation. Their presence does not establish
G2 or G8 acceptance. Kimi implementation remains gated by DeepSeek qualification.
