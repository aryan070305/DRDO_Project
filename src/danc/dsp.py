"""Shared signal-processing conventions for the whole D-ANC stack.

One STFT definition is used everywhere (training, streaming inference, the
sub-band adaptive filter and the classical baselines) so that the adaptive
filter and the network can share a single analysis/synthesis pair and add
zero extra latency.

    fs = 16 kHz, window = 320 samples (20 ms), hop = 160 (10 ms), n_fft = 320
    analysis/synthesis window = sqrt(periodic Hann)  ->  w_a * w_s = Hann,
    and a periodic Hann at 50 % overlap sums to exactly 1 (perfect reconstruction).

Algorithmic latency of the analysis/synthesis pair = window length = 20 ms.

Frame indexing (offline == streaming): the signal is left-padded with
(win - hop) zeros, so frame k covers x[k*hop - (win-hop) : k*hop + hop].  In
streaming mode frame k is available as soon as input block k (samples
[k*hop, (k+1)*hop)) has arrived, and synthesis then emits samples
[k*hop - (win-hop), k*hop).  Output-stream index delay = win - hop; adding the
one-hop block buffer of any block-based audio I/O gives a constant wall-clock
algorithmic latency of win = 20 ms (processing time must fit inside one hop).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is optional for the numpy-only parts (baselines, adaptive filter)
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class STFTConfig:
    fs: int = 16000
    win: int = 320
    hop: int = 160
    n_fft: int = 320

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def latency_samples(self) -> int:
        return self.win

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.win / self.fs


DEFAULT = STFTConfig()


def sqrt_hann(win: int) -> np.ndarray:
    n = np.arange(win)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / win)  # periodic Hann
    return np.sqrt(hann).astype(np.float64)


# ----------------------------------------------------------------------------
# numpy offline
# ----------------------------------------------------------------------------

def n_frames(n_samples: int, cfg: STFTConfig = DEFAULT) -> int:
    return int(np.ceil(n_samples / cfg.hop)) + 1


def stft(x: np.ndarray, cfg: STFTConfig = DEFAULT) -> np.ndarray:
    """x: (..., N) real -> (..., T, F) complex, T = ceil(N/hop) + 1."""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    T = n_frames(N, cfg)
    pad_l = cfg.win - cfg.hop
    total = (T - 1) * cfg.hop + cfg.win
    xp = np.zeros(x.shape[:-1] + (total,), dtype=np.float64)
    xp[..., pad_l:pad_l + N] = x
    w = sqrt_hann(cfg.win)
    idx = np.arange(cfg.win)[None, :] + cfg.hop * np.arange(T)[:, None]
    frames = xp[..., idx] * w
    return np.fft.rfft(frames, n=cfg.n_fft, axis=-1)


def istft(X: np.ndarray, length: int, cfg: STFTConfig = DEFAULT) -> np.ndarray:
    """(..., T, F) complex -> (..., length) real, aligned with the stft() input."""
    T = X.shape[-2]
    frames = np.fft.irfft(X, n=cfg.n_fft, axis=-1)[..., : cfg.win] * sqrt_hann(cfg.win)
    total = (T - 1) * cfg.hop + cfg.win
    y = np.zeros(X.shape[:-2] + (total,), dtype=np.float64)
    for t in range(T):
        y[..., t * cfg.hop: t * cfg.hop + cfg.win] += frames[..., t, :]
    pad_l = cfg.win - cfg.hop
    return y[..., pad_l: pad_l + length]


# ----------------------------------------------------------------------------
# numpy streaming (exactly matches the offline functions)
# ----------------------------------------------------------------------------

class StreamingSTFT:
    """Hop-by-hop analysis/synthesis. Output is delayed by cfg.win samples."""

    def __init__(self, cfg: STFTConfig = DEFAULT, channels: int = 1):
        self.cfg = cfg
        self.w = sqrt_hann(cfg.win)
        self.inbuf = np.zeros((channels, cfg.win), dtype=np.float64)
        self.olabuf = np.zeros(cfg.win, dtype=np.float64)

    def reset(self):
        self.inbuf[:] = 0
        self.olabuf[:] = 0

    def analyze(self, block: np.ndarray) -> np.ndarray:
        """block: (channels, hop) or (hop,) -> (channels, F) complex."""
        block = np.atleast_2d(np.asarray(block, dtype=np.float64))
        h = self.cfg.hop
        self.inbuf[:, :-h] = self.inbuf[:, h:]
        self.inbuf[:, -h:] = block
        return np.fft.rfft(self.inbuf * self.w, n=self.cfg.n_fft, axis=-1)

    def synthesize(self, frame: np.ndarray) -> np.ndarray:
        """frame: (F,) complex -> hop output samples."""
        h, W = self.cfg.hop, self.cfg.win
        y = np.fft.irfft(frame, n=self.cfg.n_fft)[:W] * self.w
        self.olabuf += y
        out = self.olabuf[:h].copy()
        self.olabuf[:-h] = self.olabuf[h:]
        self.olabuf[-h:] = 0
        return out


# ----------------------------------------------------------------------------
# torch (training) - identical framing
# ----------------------------------------------------------------------------

if torch is not None:

    class TorchSTFT(torch.nn.Module):
        def __init__(self, cfg: STFTConfig = DEFAULT):
            super().__init__()
            self.cfg = cfg
            self.register_buffer("window", torch.from_numpy(sqrt_hann(cfg.win)).float(), persistent=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (B, N) -> (B, T, F) complex64."""
            cfg = self.cfg
            N = x.shape[-1]
            T = n_frames(N, cfg)
            total = (T - 1) * cfg.hop + cfg.win
            xp = F.pad(x, (cfg.win - cfg.hop, total - N - (cfg.win - cfg.hop)))
            frames = xp.unfold(-1, cfg.win, cfg.hop) * self.window  # (B, T, win)
            return torch.fft.rfft(frames, n=cfg.n_fft, dim=-1)

        def inverse(self, X: "torch.Tensor", length: int) -> "torch.Tensor":
            """X: (B, T, F) complex -> (B, length)."""
            cfg = self.cfg
            B, T, _ = X.shape
            frames = torch.fft.irfft(X, n=cfg.n_fft, dim=-1)[..., : cfg.win] * self.window  # (B,T,win)
            total = (T - 1) * cfg.hop + cfg.win
            y = F.fold(frames.transpose(1, 2), output_size=(1, total), kernel_size=(1, cfg.win),
                       stride=(1, cfg.hop))
            y = y.reshape(B, total)
            pad_l = cfg.win - cfg.hop
            return y[:, pad_l: pad_l + length]
