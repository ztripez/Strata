# Draft Reddit post

**Title:** I got a full Kimi K2 checkpoint generating across an RTX 3080 and RX 6800

I've been working on a distributed inference fork of Strata using two gaming PCs:
an RTX 3080 10 GB with a Ryzen 5900X, and an RX 6800 16 GB with a Ryzen 3900X.
Both have 64 GB RAM. They're connected over 10GbE, with model weights backed by
shared NFS storage.

The goal is to run models larger than either machine can keep resident. Some
layers run on CUDA, others on HIP. Expert work can also go in either direction,
using GPU, CPU/RAM and mapped weights rather than loading the whole model into VRAM.

The latest result is the complete Kimi K2 Instruct UD-IQ3_XXS checkpoint, about
417 GB. It executed all 61 layers, consumed a real 27-token chat prompt and
generated two autoregressive tokens: **Hi there**. The generated first token was
fed back through the model to generate the second. Both remote expert directions
executed, with zero fallbacks, and all four native processes exited cleanly.

That's a very small output. It's also the complete model path running on mixed
consumer NVIDIA/AMD hardware. It isn't a claim that the whole checkpoint fits in
26 GB VRAM: most weights stay in file-backed storage, with only a working set
resident. The run was slow; the process lifetimes were roughly two hours including
setup, prefill, generation and cleanup. This is bring-up, not an interactive-speed demo.

The same fabric has also run DeepSeek V4.1 Flash at 2.00-bpw and the complete
~395 GB Q3 set through actual 2K sequences, plus earlier Qwen3.6 remote-expert and
decode checks. DeepSeek Q3 stayed within the measured memory ceilings and used
remote CPU experts in both directions.

Limitations: placement is static during a session, launchers still need manual
setup, and model quality/performance work is deferred. DeepSeek's strict numerical
gate failed even though we accepted a bounded experimental checkpoint. Q3 also
has a recorded exact-answer formatting failure. Kimi's result is two tokens,
not a general quality evaluation or a long-context test. Pro and Kimi K3 are not
proven fabric models.

Code and model-by-model evidence:
https://github.com/ztripez/Strata/blob/main/docs/FABRIC.md

This builds on Niko1221's Strata project:
https://github.com/Niko1221/Strata

Next objective: get more complete models running. Speed and polish can follow.
