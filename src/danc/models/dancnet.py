"""DANCNet - causal sub-band/full-band complex-domain speech enhancement network.

Design goals (see reports/REPORT.md for the evidence behind each choice)
-------------------------------------------------------------------------
* 16 kHz, 20 ms window / 10 ms hop (danc.dsp), zero look-ahead -> 20 ms
  algorithmic latency; one frame of compute must fit in 10 ms on the edge SoC.
* Features in the complex domain (phase is preserved and enhanced), with
  power-law compression (c = 0.3) of the magnitude.
* Impulse-robust input normalisation: the frame log-energy is normalised by its
  running *median* over the last `norm_win` frames.  A gunshot lasting a few
  frames cannot move a median, so speech after an impulse is not
  under-scaled (an exponential-mean normaliser would be pulled up by the blast
  for hundreds of milliseconds).
* Sub-band resolution where it matters: bins 0..63 (0-3.15 kHz, speech
  harmonics and rotor/engine lines) are kept at full 50 Hz resolution, bins
  64..160 are merged into 32 ERB-spaced bands (cf. DeepFilterNet / GTCRN ERB
  compression), plus sub-band feature unfolding (neighbouring units).
* Dual-path recurrent modelling: a bidirectional GRU across frequency within a
  frame (full-band context - legal, it does not look into the future) and a
  causal GRU across time per frequency unit (sub-band temporal context),
  cf. DPCRN / GTCRN.
* Output stage: low band -> complex *deep filter* of order N over the current
  and N-1 past frames (DeepFilterNet-style; models periodic/harmonic noise such
  as rotor blade-passing lines with sub-bin precision), high band -> real
  ERB-band gains (phase is perceptually less important there).
* An auxiliary speech-presence probability (SPP) head; at run time the SPP
  gates the reference-microphone adaptive filter (freeze adaptation while the
  talker is active, like double-talk control in echo cancellers).

Streaming
---------
`forward(spec, states)` accepts any number of frames T together with an
explicit list of state tensors (convolution caches, GRU hidden states,
normaliser history, deep-filter history).  Training calls it with T = 400 and
zero states; the streaming engine / ONNX export call it with T = 1 and carry
the states.  Because both use the *same* code path, offline and streaming
outputs are identical (verified in tests/test_model.py).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# The normaliser history stores log10(energy) + HIST_OFFSET.  Real frames are always
# > HIST_OFFSET - 10, so a value < HIST_OFFSET / 2 means "no history yet".  This makes an
# all-ZERO initial state valid, so any runtime (ONNX Runtime, TensorRT) can zero-init every state.
HIST_OFFSET = 100.0


@dataclass
class DANCNetConfig:
    n_bins: int = 161          # n_fft 320 -> 161
    n_low: int = 64            # full-resolution low band (0 .. 3150 Hz)
    n_erb: int = 32            # ERB bands for bins n_low .. 160
    fs: int = 16000
    n_fft: int = 320
    c1: int = 16
    c2: int = 32
    c: int = 64                # dual-path width
    n_dp: int = 2              # dual-path blocks
    df_order: int = 3          # deep-filter taps (current + 2 past frames)
    norm_win: int = 64         # frames in the running-median normaliser (0.64 s)
    norm_type: str = "median"  # "median" (impulse-robust, default) | "ema" (ablation: DeepFilterNet-style exponential
                               # mean of the log energy, alpha = norm_alpha per frame, over the same 64-frame history)
    norm_alpha: float = 0.95
    compress: float = 0.3
    feat_clamp: float = 10.0
    df_bound: float = 1.5      # |Re|,|Im| of deep-filter taps bounded by tanh * df_bound
    in_ch: int = 1             # 1 = primary mic only; 2 = primary + reference mic (learned spatial cues, see forward)
    lookahead: int = 0         # L frames of look-ahead (0 = 20 ms latency mode; 2 = 40 ms "HQ" mode).
                               # The DF window covers frames t-df_order+1 .. t and the output is frame t-L,
                               # i.e. L of the df_order taps lie in the future (as DeepFilterNet's df_lookahead).

    def to_dict(self):
        return asdict(self)


# -----------------------------------------------------------------------------
# ERB filterbank for the high band
# -----------------------------------------------------------------------------

def _hz_to_erb(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def _erb_to_hz(e):
    return (10 ** (e / 21.4) - 1) / 0.00437


def erb_matrices(n_bins: int, n_low: int, n_erb: int, fs: int, n_fft: int):
    """Triangular ERB filterbank over bins n_low..n_bins-1.

    Returns (fb, ifb):
      fb  [n_high, n_erb] - columns sum to 1 (band value = weighted mean of its bins)
      ifb [n_erb, n_high] - columns sum to 1 (bin value = interpolation of band values)
    """
    freqs = np.arange(n_bins) * fs / n_fft
    hf = freqs[n_low:]
    n_high = len(hf)
    e_lo, e_hi = _hz_to_erb(hf[0]), _hz_to_erb(hf[-1])
    centers = _erb_to_hz(np.linspace(e_lo, e_hi, n_erb))
    w = np.zeros((n_high, n_erb))
    for b in range(n_erb):
        left = centers[b - 1] if b > 0 else centers[0] - (centers[1] - centers[0])
        right = centers[b + 1] if b < n_erb - 1 else centers[-1] + (centers[-1] - centers[-2])
        c = centers[b]
        up = (hf - left) / max(c - left, 1e-9)
        down = (right - hf) / max(right - c, 1e-9)
        w[:, b] = np.clip(np.minimum(up, down), 0, None)
    # make sure every bin belongs to at least one band
    w[w.sum(1) == 0, np.argmin(np.abs(hf[w.sum(1) == 0, None] - centers[None]), axis=1)] = 1.0
    fb = w / w.sum(0, keepdims=True)
    ifb = (w / w.sum(1, keepdims=True)).T
    return fb.astype(np.float32), ifb.astype(np.float32)


# -----------------------------------------------------------------------------
# Building blocks (all accept explicit caches so T may be 1 or many)
# -----------------------------------------------------------------------------

class CausalConv2d(nn.Module):
    """Conv over (time, freq); causal in time via an explicit left cache."""

    def __init__(self, cin, cout, kt, kf, sf=1, groups=1):
        super().__init__()
        self.kt = kt
        self.conv = nn.Conv2d(cin, cout, (kt, kf), stride=(1, sf), padding=(0, kf // 2), groups=groups)

    def forward(self, x, cache):
        # x [B, C, T, F], cache [B, C, kt-1, F]
        if self.kt > 1:
            x = torch.cat([cache, x], dim=2)
            new_cache = x[:, :, -(self.kt - 1):]
        else:
            new_cache = cache
        return self.conv(x), new_cache


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kt, kf, sf):
        super().__init__()
        self.conv = CausalConv2d(cin, cout, kt, kf, sf)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x, cache):
        y, cache = self.conv(x, cache)
        return self.act(self.bn(y)), cache


class UpBlock(nn.Module):
    """Frequency up-sampling x2 (no temporal context -> stateless)."""

    def __init__(self, cin, cout, last=False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, (1, 4), stride=(1, 2), padding=(0, 1))
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DPBlock(nn.Module):
    """Dual-path block: full-band BiGRU over frequency, causal GRU over time."""

    def __init__(self, c, n_freq):
        super().__init__()
        self.intra = nn.GRU(c, c // 2, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(c, c)
        self.intra_ln = nn.LayerNorm([n_freq, c])
        self.inter = nn.GRU(c, c, batch_first=True)
        self.inter_fc = nn.Linear(c, c)
        self.inter_ln = nn.LayerNorm([n_freq, c])

    def forward(self, x, h):
        # x [B, C, T, Fq]; h [1, B*Fq, C]
        B, C, T, Fq = x.shape
        z = x.permute(0, 2, 3, 1)                       # B T F C
        a, _ = self.intra(z.reshape(B * T, Fq, C))      # (B*T) F C
        a = self.intra_ln(self.intra_fc(a).reshape(B, T, Fq, C))
        z = z + a
        u = z.permute(0, 2, 1, 3).reshape(B * Fq, T, C)  # (B*F) T C
        v, h = self.inter(u, h)
        v = self.inter_fc(v).reshape(B, Fq, T, C).permute(0, 2, 1, 3)
        z = z + self.inter_ln(v)
        return z.permute(0, 3, 1, 2), h


# -----------------------------------------------------------------------------
# DANCNet
# -----------------------------------------------------------------------------

class DANCNet(nn.Module):
    def __init__(self, cfg: DANCNetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or DANCNetConfig()
        n_high = cfg.n_bins - cfg.n_low
        fb, ifb = erb_matrices(cfg.n_bins, cfg.n_low, cfg.n_erb, cfg.fs, cfg.n_fft)
        self.register_buffer("erb_fb", torch.from_numpy(fb), persistent=False)    # [n_high, n_erb]
        self.register_buffer("erb_ifb", torch.from_numpy(ifb), persistent=False)  # [n_erb, n_high]
        self.n_units = cfg.n_low + cfg.n_erb                                       # 96
        assert self.n_units % 4 == 0
        f1, f2 = self.n_units // 2, self.n_units // 4                             # 48, 24
        self.f2 = f2
        # encoder
        self.enc1 = ConvBlock(9 * cfg.in_ch, cfg.c1, 1, 5, 2)   # 96 -> 48 (stateless in time)
        self.enc2 = ConvBlock(cfg.c1, cfg.c2, 2, 3, 2)   # 48 -> 24
        self.enc3 = ConvBlock(cfg.c2, cfg.c, 2, 3, 1)    # 24
        self.dp = nn.ModuleList([DPBlock(cfg.c, f2) for _ in range(cfg.n_dp)])
        # decoder (skip connections by concatenation)
        self.dec3 = ConvBlock(2 * cfg.c, cfg.c2, 2, 3, 1)
        self.dec2 = UpBlock(2 * cfg.c2, cfg.c1)           # 24 -> 48
        self.dec1 = UpBlock(2 * cfg.c1, cfg.c1)           # 48 -> 96
        self.n_out = 2 * cfg.df_order + 2                 # DF taps (re,im) | gain | spp
        self.head = nn.Conv2d(cfg.c1, self.n_out, 1)
        self._n_high = n_high
        assert 0 <= cfg.lookahead <= cfg.df_order - 1, "lookahead must be < df_order"
        # L = 0 keeps only the low band in the history (unchanged legacy state shape);
        # L > 0 also needs the high band of the L future frames.
        self._hist_bins = cfg.n_low if cfg.lookahead == 0 else cfg.n_bins

    # ------------------------------------------------------------------ states
    def initial_states(self, batch: int = 1, device=None, dtype=torch.float32):
        c = self.cfg
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)  # noqa: E731
        st = [
            z(batch, c.norm_win - 1),                   # log-energy history (+HIST_OFFSET, 0 = empty)
            z(batch, c.c1, 1, self.n_units // 2),       # enc2 cache
            z(batch, c.c2, 1, self.f2),                 # enc3 cache
            z(batch, 2 * c.c, 1, self.f2),              # dec3 cache
            z(batch, c.df_order - 1, self._hist_bins, 2),  # deep-filter history (raw spectrum)
        ]
        st += [z(1, batch * self.f2, c.c) for _ in range(c.n_dp)]   # GRU hidden
        return st

    # ---------------------------------------------------------------- features
    def _normalise(self, spec, hist):
        """Running-median log-energy normaliser. spec [B,T,F,2]; hist [B,W-1] (0 = no history yet)."""
        e = torch.log10((spec ** 2).sum(-1).mean(-1) + 1e-10) + HIST_OFFSET   # [B,T]
        seq = torch.cat([hist, e], dim=1)                              # [B, W-1+T]
        W = self.cfg.norm_win
        T = e.shape[1]
        win = seq.unsqueeze(1) if T == 1 else seq.unfold(1, W, 1)      # [B, T, W]
        # empty slots (start-up) are replaced by the current frame's value
        cur = e.unsqueeze(-1).expand_as(win)
        win = torch.where(win < HIST_OFFSET / 2, cur, win)
        if self.cfg.norm_type == "ema":
            # exponentially weighted mean of log10 energy over the window (newest frame weight largest)
            wts = self.cfg.norm_alpha ** torch.arange(W - 1, -1, -1, device=win.device, dtype=win.dtype)
            med = (win * wts).sum(-1) / wts.sum() - HIST_OFFSET
        else:
            med = torch.sort(win, dim=-1)[0][..., W // 2] - HIST_OFFSET    # [B,T] median log10 energy
        new_hist = seq[:, -(W - 1):]
        scale = torch.pow(10.0, -0.5 * med)                            # 1/sqrt(energy)
        self._last_scale = scale
        return spec * scale[..., None, None], new_hist

    def _features(self, specn):
        """specn [B,T,F,2] normalised -> unit features [B,9,T,96]."""
        c = self.cfg
        mag = torch.sqrt((specn ** 2).sum(-1) + 1e-12)                 # [B,T,F]
        magc = torch.clamp(mag ** c.compress, max=c.feat_clamp)
        unit = specn / mag.unsqueeze(-1)                               # phase (unit complex)
        feat = torch.stack([magc, magc * unit[..., 0], magc * unit[..., 1]], dim=1)  # [B,3,T,F]
        low = feat[..., : c.n_low]
        high = feat[..., c.n_low:] @ self.erb_fb                       # [B,3,T,n_erb]
        x = torch.cat([low, high], dim=-1)                             # [B,3,T,96]
        # sub-band feature unfolding: each unit sees its two neighbours
        xp = F.pad(x, (1, 1))
        x = torch.cat([xp[..., :-2], xp[..., 1:-1], xp[..., 2:]], dim=1)  # [B,9,T,96]
        return x

    # ----------------------------------------------------------------- forward
    def forward(self, spec, states, ref=None):
        """spec [B,T,161,2] (raw STFT re/im) -> (spec_out [B,T,161,2], spp [B,T,161], new_states)."""
        c = self.cfg
        hist, c_e2, c_e3, c_d3, df_hist = states[:5]
        hs = states[5:]
        specn, hist = self._normalise(spec, hist)
        x = self._features(specn)
        if c.in_ch == 2:
            # reference mic normalised with the PRIMARY's scale, so inter-mic level/phase differences survive
            if ref is None:
                raise ValueError("in_ch=2 model needs the reference-mic spectrum (ref)")
            x = torch.cat([x, self._features(ref * self._last_scale[..., None, None])], dim=1)
        e1, _ = self.enc1(x, x[:, :, :0])
        e2, c_e2 = self.enc2(e1, c_e2)
        e3, c_e3 = self.enc3(e2, c_e3)
        z = e3
        new_h = []
        for blk, h in zip(self.dp, hs):
            z, h = blk(z, h)
            new_h.append(h)
        d3, c_d3 = self.dec3(torch.cat([z, e3], 1), c_d3)
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        out = self.head(d1)                                            # [B, n_out, T, 96]
        N, L = c.df_order, c.lookahead
        # --- low band: deep filter over frames t-N+1 .. t of the RAW spectrum; output = frame t-L
        coef = torch.tanh(out[:, : 2 * N, :, : c.n_low]) * c.df_bound   # [B,2N,T,n_low]
        coef = coef.reshape(coef.shape[0], N, 2, coef.shape[2], c.n_low).permute(0, 3, 4, 1, 2)  # B T F N 2
        seq_full = torch.cat([df_hist, spec[..., : self._hist_bins, :]], dim=1)  # B (N-1+T) F 2
        seq = seq_full[..., : c.n_low, :]
        T = spec.shape[1]
        # frames t, t-1, ..., t-N+1 stacked on a tap axis
        taps = torch.stack([seq[:, N - 1 - k: N - 1 - k + T] for k in range(N)], dim=3)  # B T F N 2
        yr = (coef[..., 0] * taps[..., 0] - coef[..., 1] * taps[..., 1]).sum(-1)
        yi = (coef[..., 0] * taps[..., 1] + coef[..., 1] * taps[..., 0]).sum(-1)
        y_low = torch.stack([yr, yi], dim=-1)
        new_df_hist = seq_full[:, -(N - 1):] if N > 1 else df_hist
        # --- high band: real ERB-band gains applied to frame t-L
        g_band = torch.sigmoid(out[:, 2 * N, :, c.n_low:])              # [B,T,n_erb]
        g_bins = g_band @ self.erb_ifb                                  # [B,T,n_high]
        x_high = spec[..., c.n_low:, :] if L == 0 else seq_full[:, N - 1 - L: N - 1 - L + T, c.n_low:]
        y_high = x_high * g_bins.unsqueeze(-1)
        spec_out = torch.cat([y_low, y_high], dim=2)
        # --- speech presence
        spp_u = torch.sigmoid(out[:, 2 * N + 1])                        # [B,T,96]
        spp = torch.cat([spp_u[..., : c.n_low], spp_u[..., c.n_low:] @ self.erb_ifb], dim=-1)
        new_states = [hist, c_e2, c_e3, c_d3, new_df_hist] + new_h
        return spec_out, spp, new_states

    # ------------------------------------------------------------- utilities
    def enhance_spec(self, spec, ref=None):
        """Offline convenience: spec [B,T,F,2] with zero initial state; output aligned with input
        (for look-ahead L > 0 the L-frame output delay is removed by flushing L zero frames)."""
        st = self.initial_states(spec.shape[0], spec.device, spec.dtype)
        L = self.cfg.lookahead
        if L > 0:
            spec = torch.cat([spec, spec.new_zeros(spec.shape[0], L, *spec.shape[2:])], dim=1)
            if ref is not None:
                ref = torch.cat([ref, ref.new_zeros(ref.shape[0], L, *ref.shape[2:])], dim=1)
        y, spp, _ = self.forward(spec, st, ref) if ref is not None else self.forward(spec, st)
        return (y[:, L:], spp[:, L:]) if L > 0 else (y, spp)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def macs_per_frame(cfg: DANCNetConfig | None = None) -> dict:
    """Analytic multiply-accumulate count for ONE 10 ms frame (batch 1)."""
    cfg = cfg or DANCNetConfig()
    U = cfg.n_low + cfg.n_erb
    f1, f2 = U // 2, U // 4
    m = {}
    k = int(getattr(cfg, "in_ch", 1))                     # microphones fed to the network
    m["features_erb"] = 3 * (cfg.n_bins - cfg.n_low) * cfg.n_erb * k
    m["enc1"] = f1 * cfg.c1 * 9 * k * 5
    m["enc2"] = f2 * cfg.c2 * cfg.c1 * 2 * 3
    m["enc3"] = f2 * cfg.c * cfg.c2 * 2 * 3
    gru = lambda i, h: 3 * (i * h + h * h)  # noqa: E731
    dp = 0
    for _ in range(cfg.n_dp):
        dp += f2 * 2 * gru(cfg.c, cfg.c // 2) + f2 * cfg.c * cfg.c   # intra BiGRU + fc
        dp += f2 * gru(cfg.c, cfg.c) + f2 * cfg.c * cfg.c            # inter GRU + fc
    m["dual_path"] = dp
    m["dec3"] = f2 * cfg.c2 * 2 * cfg.c * 2 * 3
    m["dec2"] = f1 * cfg.c1 * 2 * cfg.c2 * 4 // 2
    m["dec1"] = U * cfg.c1 * 2 * cfg.c1 * 4 // 2
    m["head"] = U * cfg.c1 * (2 * cfg.df_order + 2)
    m["deep_filter"] = cfg.n_low * cfg.df_order * 4
    m["total"] = sum(m.values())
    return m
