"""Optional SPP-driven residual-noise post-filter (off by default; parameters tuned on validation data only).

g(t,f) = beta + (1 - beta) * s(t,f)^gamma, with s = max(spp_bin, spp_frame), optionally smoothed over time
(fast attack / slower release, coefficient `smooth`) so the gain does not flutter between frames.
spp_frame = mean SPP over bins 6..70 (300-3500 Hz).  beta = 1 -> identity.
Streaming-friendly: causal, per-frame, no look-ahead (state = previous smoothed s).
"""
from __future__ import annotations

import numpy as np


def spp_postfilter(Y: np.ndarray, spp: np.ndarray, beta: float = 1.0, gamma: float = 1.0, smooth: float = 0.0,
                   band=(6, 70)) -> np.ndarray:
    """Y: (T,F) complex enhanced spectrum, spp: (T,F) in [0,1]."""
    if beta >= 1.0:
        return Y
    s_frame = spp[:, band[0]:band[1]].mean(1, keepdims=True)
    s = np.maximum(spp, s_frame)
    if smooth > 0:
        out = np.empty_like(s)
        prev = s[0]
        for t in range(s.shape[0]):
            prev = np.where(s[t] > prev, s[t], smooth * prev + (1 - smooth) * s[t])   # instant attack, smooth release
            out[t] = prev
        s = out
    return Y * (beta + (1.0 - beta) * np.clip(s, 0, 1) ** gamma)


class StreamingPostFilter:
    def __init__(self, beta=1.0, gamma=1.0, smooth=0.0, band=(6, 70)):
        self.beta, self.gamma, self.smooth, self.band = beta, gamma, smooth, band
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, Y: np.ndarray, spp: np.ndarray) -> np.ndarray:
        if self.beta >= 1.0:
            return Y
        s = np.maximum(spp, spp[self.band[0]:self.band[1]].mean())
        if self.smooth > 0:
            self.prev = s if self.prev is None else np.where(s > self.prev, s, self.smooth * self.prev + (1 - self.smooth) * s)
            s = self.prev
        return Y * (self.beta + (1.0 - self.beta) * np.clip(s, 0, 1) ** self.gamma)
