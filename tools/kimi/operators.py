"""CPU reference operators for the experimental Kimi adapter (not engine dispatch)."""

import numpy as np


def situ(gate, up, beta=4.0, linear_beta=25.0):
    """Apply Kimi's bounded SiTU gate and bounded up activation."""
    return (
        beta
        * np.tanh(gate / beta)
        / (1 + np.exp(-gate))
        * (linear_beta * np.tanh(up / linear_beta))
    )


def kda(q, k, v, log_decay, beta, state=None):
    """Execute normalized KDA inputs [time, heads, channels]; return output/state.

    Gates are already transformed to log decay and beta to sigmoid values.
    State layout is [heads, key channels, value channels].
    """
    if state is None:
        state = np.zeros((k.shape[1], k.shape[2], v.shape[2]), dtype=np.float32)
    else:
        state = state.copy()
    output = []
    for qt, kt, vt, gt, bt in zip(q, k, v, log_decay, beta, strict=True):
        state *= np.exp(gt)[..., None]
        error = vt - np.einsum("hk,hkv->hv", kt, state)
        state += kt[..., None] * (bt[..., None] * error)[:, None, :]
        output.append(np.einsum("hk,hkv->hv", qt, state) / np.sqrt(q.shape[-1]))
    return np.stack(output), state
