"""Dual-microphone headset / helmet scene simulator for the hybrid D-ANC system.

The *primary* mic sits at the mouth (boom or in-helmet) and picks up speech
plus noise.  The *reference* mic sits ``d`` = 8-15 cm away, facing outward,
and picks up mostly noise plus attenuated speech leakage.  The simulator
produces training and evaluation material for the reference-mic adaptive
canceller (:mod:`danc.adaptive.nlms`) and for the DNN that follows it.

Noise field model
-----------------
* **Directional sources (1-3).**  Point sources at 1-30 m are propagated
  to both mics with spherical spreading (``1/r``, relative to the primary)
  and exact geometric delays.  Fractional delays are applied with a
  Kaiser-windowed sinc interpolator.  Each source also has 1-3 early
  reflections (ground / body / vehicle surfaces).  A reflection is modelled
  as an image source from a random direction with a random extra delay and
  attenuation, so it carries its own inter-mic delay.  A mild random short
  FIR plus a +/-3 dB gain on the reference path mimics head / helmet
  shadowing.  This is a simplified free-field + image model, not a measured
  head-related transfer function.
* **Diffuse component.**  The spherically isotropic noise field has
  inter-mic spatial coherence

      Gamma(f) = sin(2 pi f d / c) / (2 pi f d / c) = np.sinc(2 f d / c)

  (R. K. Cook, R. V. Waterhouse, R. D. Berendt, S. Edelman and
  M. C. Thompson Jr., "Measurement of correlation coefficients in
  reverberant sound fields," J. Acoust. Soc. Am., vol. 27, no. 6,
  pp. 1072-1077, 1955).  Sensor signals with this coherence come from
  mixing two mutually independent noise realisations per STFT bin through
  the Cholesky factor of the 2x2 coherence matrix::

      X_p = N_1
      X_r = Gamma(f) N_1 + sqrt(1 - Gamma(f)^2) N_2

  following E. A. P. Habets and S. Gannot, "Generating sensor signals in
  isotropic noise fields," J. Acoust. Soc. Am., vol. 122, no. 6,
  pp. 3464-3470, 2007, and its non-stationary extension E. A. P. Habets,
  I. Cohen and S. Gannot, "Generating nonstationary multisensor signals
  under a spatial coherence constraint," J. Acoust. Soc. Am., vol. 124,
  no. 5, pp. 2911-2917, 2008.  NOTE: as far as we recall, the
  STFT-domain mixing-matrix formulation is spelled out in the 2008 paper;
  the 2007 paper's own generator may differ in detail.  Check against the
  originals before citing this file in a publication.  Free-field coherence
  ignores the head / helmet, which in practice lowers the coherence further
  at high frequencies.
* **Speech leakage.**  The reference picks up speech attenuated by
  ``speech_leak_db`` (relative to the primary), delayed by ``d / c`` and
  coloured by a mild random FIR.
* **Mic self-noise.**  Independent white Gaussian noise in each mic at
  ``self_noise_db`` dBFS RMS, where 0 dBFS means an RMS of 1.0.

SNR convention: ``snr_db`` is set at the PRIMARY mic over the whole segment,
as ``10 log10(sum clean^2 / sum noise_primary^2)``.  Here ``noise_primary``
is everything at the primary that is not the clean speech, including
self-noise.

All randomness comes from the passed ``numpy.random.Generator``, so the same
seed gives the same scene bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import oaconvolve

from danc.dsp import DEFAULT, STFTConfig, istft, stft

__all__ = ["DualMicConfig", "simulate_scene", "diffuse_coherence", "fractional_delay_ir"]

_SINC_HALF = 16      # half-length of the windowed-sinc fractional-delay kernel
_KAISER_BETA = 8.0
_EPS = 1e-20


@dataclass
class DualMicConfig:
    """Sampling ranges of the dual-mic scene.  Each range is ``(low, high)``,
    sampled uniformly; set ``low == high`` to fix a parameter.

    Attributes
    ----------
    mic_spacing_m : primary-to-reference distance d [m].
    speech_leak_db : speech level at the reference relative to the primary [dB].
    n_directional : number of directional noise sources (integer range, inclusive).
    source_distance_m : distance of each directional source from the primary [m].
    diffuse_ratio : fraction of the ambient (non-self) noise energy at the
        primary that is diffuse.
    self_noise_db : mic self-noise RMS level [dBFS, 0 dBFS = RMS 1.0].
    c : speed of sound [m/s].
    elevation_deg : elevation range of the direct paths [deg].
    n_reflections : early reflections per directional source (integer range, inclusive).
    reflection_delay_ms : extra delay of a reflection after its direct path [ms].
    reflection_gain_db : reflection level relative to its direct path [dB].
    shadow_gain_db : broadband reference-path gain per source (head/helmet shadow) [dB].
    shadow_fir_std : std of the non-leading taps of the 4-tap relative shadowing FIR.
    leak_fir_std : std of the non-leading taps of the 3-tap speech-leakage FIR.
    level_spread_db : relative level spread between directional sources [dB].
    max_peak : if not None, all outputs are scaled down together when the
        peak of primary/reference exceeds it (SNR and relations preserved).
    """

    mic_spacing_m: tuple[float, float] = (0.08, 0.15)
    speech_leak_db: tuple[float, float] = (-25.0, -12.0)
    n_directional: tuple[int, int] = (1, 3)
    source_distance_m: tuple[float, float] = (1.0, 30.0)
    diffuse_ratio: tuple[float, float] = (0.0, 0.5)
    self_noise_db: tuple[float, float] = (-70.0, -60.0)
    c: float = 343.0
    elevation_deg: tuple[float, float] = (-20.0, 45.0)
    n_reflections: tuple[int, int] = (1, 3)
    reflection_delay_ms: tuple[float, float] = (0.3, 10.0)
    reflection_gain_db: tuple[float, float] = (-12.0, -3.0)
    shadow_gain_db: tuple[float, float] = (-3.0, 3.0)
    shadow_fir_std: float = 0.1
    leak_fir_std: float = 0.1
    level_spread_db: float = 6.0
    max_peak: float | None = 0.99


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def diffuse_coherence(f: np.ndarray, d: float, c: float = 343.0) -> np.ndarray:
    """Spherically isotropic coherence ``sin(2 pi f d/c) / (2 pi f d/c)`` (Cook et al. 1955)."""
    return np.sinc(2.0 * np.asarray(f, dtype=np.float64) * d / c)


def _uniform(rng: np.random.Generator, rg) -> float:
    lo, hi = float(rg[0]), float(rg[1])
    return lo if hi <= lo else float(rng.uniform(lo, hi))


def _randint(rng: np.random.Generator, rg) -> int:
    lo, hi = int(rg[0]), int(rg[1])
    return lo if hi <= lo else int(rng.integers(lo, hi + 1))


def _unit(az: float, el: float) -> np.ndarray:
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _kernel(frac: float, half: int = _SINC_HALF) -> np.ndarray:
    """Kaiser-windowed sinc taps for offsets m = -half+1 .. half, approximating x(n - frac)."""
    m = np.arange(-half + 1, half + 1, dtype=np.float64)
    t = m - frac
    win = np.i0(_KAISER_BETA * np.sqrt(np.clip(1.0 - (t / half) ** 2, 0.0, 1.0))) / np.i0(_KAISER_BETA)
    return np.sinc(t) * win


def fractional_delay_ir(delays: np.ndarray, gains: np.ndarray, half: int = _SINC_HALF) -> np.ndarray:
    """Multi-path IR ``sum_p g_p delta(n - tau_p)`` with fractional delays (windowed sinc).

    ``delays`` are non-negative delays in samples.  The returned IR is
    offset by ``half`` samples (tap ``half + i`` is integer delay ``i``),
    so ``y[n] = conv(x, ir)[n + half]``.
    """
    delays = np.asarray(delays, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    if np.any(delays < 0):
        raise ValueError("delays must be non-negative")
    L = int(np.floor(delays.max())) + 2 * half + 1
    ir = np.zeros(L)
    for tau, g in zip(delays, gains):
        i = int(np.floor(tau))
        k = _kernel(tau - i, half)
        start = half + i - half + 1  # position of offset m = -half+1
        ir[start:start + 2 * half] += g * k
    return ir


def _segment(x: np.ndarray, start: int, n: int, pre: int, post: int) -> np.ndarray:
    """x[start-pre : start+n+post], zero-padded where it runs past the signal."""
    out = np.zeros(pre + n + post)
    lo, hi = start - pre, start + n + post
    a, b = max(lo, 0), min(hi, len(x))
    if b > a:
        out[a - lo:b - lo] = x[a:b]
    return out


def _apply_ir(x: np.ndarray, start: int, n: int, ir: np.ndarray, half: int = _SINC_HALF) -> np.ndarray:
    """Filter noise ``x`` starting at ``start`` with an IR from :func:`fractional_delay_ir`.

    Uses the true signal history before ``start`` where it exists, so there
    is no onset transient.
    """
    pre = len(ir)
    ext = _segment(x, start, n, pre, half)
    y = oaconvolve(ext, ir)
    return y[pre + half: pre + half + n]


def _energy(x: np.ndarray) -> float:
    return float(np.dot(x, x))


# ----------------------------------------------------------------------------
# components
# ----------------------------------------------------------------------------

def _directional(noise: np.ndarray, start: int, n: int, d: float, cfg: DualMicConfig,
                 rng: np.random.Generator, fs: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """One directional source with early reflections -> (primary, reference, meta).

    Mic geometry: primary at the origin, reference at (d, 0, 0).  The array
    orientation does not matter because source directions are uniform in
    azimuth.
    """
    mics = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    az = rng.uniform(0.0, 2.0 * np.pi)
    el = np.deg2rad(_uniform(rng, cfg.elevation_deg))
    r = _uniform(rng, cfg.source_distance_m)
    src = r * _unit(az, el)
    # paths: (position of the (image) source, gain relative to its own direct path)
    paths = [(src, 1.0)]
    refl_meta = []
    for _ in range(_randint(rng, cfg.n_reflections)):
        extra_ms = _uniform(rng, cfg.reflection_delay_ms)
        gain_db = _uniform(rng, cfg.reflection_gain_db)
        raz = rng.uniform(0.0, 2.0 * np.pi)
        rel = np.arcsin(rng.uniform(-1.0, 1.0))  # uniform on the sphere
        rr = r + extra_ms * 1e-3 * cfg.c
        paths.append((rr * _unit(raz, rel), 10.0 ** (gain_db / 20.0)))
        refl_meta.append({"extra_delay_ms": extra_ms, "gain_db": gain_db,
                          "azimuth_deg": float(np.rad2deg(raz)), "elevation_deg": float(np.rad2deg(rel))})
    dist = np.array([[np.linalg.norm(pos - m) for pos, _ in paths] for m in mics])  # (2, n_paths)
    g = np.array([gg for _, gg in paths])
    # spherical spreading relative to the direct path at the primary (direct-path gain 1 there);
    # the reflection gain already includes its absorption, the extra spreading is applied on top
    amp = g[None, :] * dist[0, 0] / dist
    tau = dist / cfg.c * fs
    tau -= tau.min()
    ir_p = fractional_delay_ir(tau[0], amp[0])
    ir_r = fractional_delay_ir(tau[1], amp[1])
    shadow = np.concatenate([[1.0], cfg.shadow_fir_std * rng.standard_normal(3)])
    shadow_db = _uniform(rng, cfg.shadow_gain_db)
    ir_r = np.convolve(ir_r, shadow * 10.0 ** (shadow_db / 20.0))
    L = max(len(ir_p), len(ir_r))
    ir_p = np.pad(ir_p, (0, L - len(ir_p)))
    ir_r = np.pad(ir_r, (0, L - len(ir_r)))
    yp = _apply_ir(noise, start, n, ir_p)
    yr = _apply_ir(noise, start, n, ir_r)
    meta = {"azimuth_deg": float(np.rad2deg(az)), "elevation_deg": float(np.rad2deg(el)), "distance_m": r,
            "inter_mic_delay_samples": float((dist[1, 0] - dist[0, 0]) / cfg.c * fs),
            "reflections": refl_meta, "shadow_gain_db": shadow_db, "shadow_fir": shadow.tolist()}
    return yp, yr, meta


def _diffuse(noise: np.ndarray, n: int, d: float, cfg: DualMicConfig, rng: np.random.Generator,
             fs: int, stft_cfg: STFTConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Two-channel spherically isotropic noise from one mono noise signal.

    Two mutually independent realisations come from two disjoint segments
    when the signal is long enough.  Otherwise the second is a random
    circular shift of the first by at least n/3, which is approximately
    independent for noise-like signals.  The two are mixed per STFT bin
    (Habets & Gannot 2007; Habets, Cohen & Gannot 2008).
    """
    L = len(noise)
    if L >= 2 * n:
        s1 = int(rng.integers(0, L - 2 * n + 1))
        gap = int(rng.integers(0, L - 2 * n - s1 + 1))
        s2 = s1 + n + gap
        n1, n2 = noise[s1:s1 + n], noise[s2:s2 + n]
        how = "disjoint"
    else:
        s1 = int(rng.integers(0, L - n + 1))
        n1 = noise[s1:s1 + n]
        shift = int(rng.integers(max(1, n // 3), max(2, n - n // 3)))
        n2 = np.roll(n1, shift)
        s2 = shift
        how = "circular_shift"
    X1, X2 = stft(n1, stft_cfg), stft(n2, stft_cfg)
    f = np.fft.rfftfreq(stft_cfg.n_fft, 1.0 / fs)
    gam = diffuse_coherence(f, d, cfg.c)
    Xr = gam * X1 + np.sqrt(np.clip(1.0 - gam ** 2, 0.0, 1.0)) * X2
    yp = np.asarray(n1, dtype=np.float64).copy()  # == istft(stft(n1)) (perfect reconstruction)
    yr = istft(Xr, n, stft_cfg)
    return yp, yr, {"segment_starts": [s1, s2], "independence": how}


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

def simulate_scene(speech: np.ndarray, noises: list[np.ndarray], snr_db: float, rng: np.random.Generator,
                   cfg: DualMicConfig = DualMicConfig(), fs: int = 16000) -> dict:
    """Simulate one dual-mic (primary + reference) noisy scene.

    Parameters
    ----------
    speech : (N,) float
        Dry close-talk speech.  It becomes the clean target at the primary
        mic, level unchanged unless peak limiting applies.
    noises : list of (>= N,) float
        Mono noise recordings.  Each directional source and the diffuse
        field draws one of them, a random permutation that is reused
        cyclically when there are fewer noises than components.  Every use
        takes a random segment.
    snr_db : float
        Speech-to-total-noise ratio at the primary mic over the whole segment.
    rng : numpy.random.Generator
        The only source of randomness.
    cfg : DualMicConfig
        Parameter ranges.
    fs : int
        Sampling rate.  The STFT used for diffuse-noise mixing takes its
        hop and window from ``danc.dsp.DEFAULT`` (assumes 16 kHz).

    Returns
    -------
    dict
        ``primary``, ``reference``: (N,) mic signals.
        ``clean``: speech component at the primary (the training target).
        ``noise_primary``, ``noise_reference``: everything except speech at
        each mic.
        ``speech_reference``: speech leakage at the reference.
        ``meta``: dict of all sampled parameters.
        By construction ``primary == clean + noise_primary`` and
        ``reference == speech_reference + noise_reference``.
    """
    s = np.asarray(speech, dtype=np.float64).reshape(-1)
    N = s.shape[0]
    if N == 0:
        raise ValueError("speech is empty")
    e_s = _energy(s)
    if e_s <= _EPS:
        raise ValueError("speech has zero energy; SNR is undefined")
    if len(noises) == 0:
        raise ValueError("at least one noise signal is required")
    noises = [np.asarray(x, dtype=np.float64).reshape(-1) for x in noises]
    for i, x in enumerate(noises):
        if len(x) < N:
            raise ValueError(f"noise {i} has {len(x)} samples < len(speech) = {N}")
    stft_cfg = DEFAULT if fs == DEFAULT.fs else STFTConfig(fs=fs, win=int(0.02 * fs), hop=int(0.01 * fs),
                                                           n_fft=int(0.02 * fs))

    # ---- scene parameters (fixed draw order => reproducible)
    d = _uniform(rng, cfg.mic_spacing_m)
    leak_db = _uniform(rng, cfg.speech_leak_db)
    n_dir = _randint(rng, cfg.n_directional)
    rho = float(np.clip(_uniform(rng, cfg.diffuse_ratio), 0.0, 1.0))
    self_db = _uniform(rng, cfg.self_noise_db)
    if rho >= 1.0:
        n_dir = 0
    if n_dir == 0 and rho <= 0.0:
        rho = 1.0  # no directional source requested: all ambient noise is diffuse
    perm = rng.permutation(len(noises))
    pick = lambda j: int(perm[j % len(noises)])  # noqa: E731

    amb_p = np.zeros(N)
    amb_r = np.zeros(N)
    src_meta = []
    # ---- directional sources
    if n_dir > 0:
        lev_db = rng.uniform(-cfg.level_spread_db, 0.0, size=n_dir)
        frac = 10.0 ** (lev_db / 10.0)
        frac = (1.0 - rho) * frac / frac.sum()
        for j in range(n_dir):
            idx = pick(j)
            x = noises[idx]
            start = int(rng.integers(0, len(x) - N + 1))
            yp, yr, m = _directional(x, start, N, d, cfg, rng, fs)
            ep = _energy(yp)
            if ep <= _EPS:
                continue
            g = np.sqrt(frac[j] / (ep / N))  # unit mean power at the primary, times its share
            amb_p += g * yp
            amb_r += g * yr
            m.update({"noise_index": idx, "segment_start": start, "energy_fraction": float(frac[j])})
            src_meta.append(m)
    # ---- diffuse field
    diff_meta = None
    if rho > 0.0:
        idx = pick(n_dir)
        yp, yr, diff_meta = _diffuse(noises[idx], N, d, cfg, rng, fs, stft_cfg)
        ep = _energy(yp)
        if ep > _EPS:
            g = np.sqrt(rho / (ep / N))
            amb_p += g * yp
            amb_r += g * yr
        diff_meta.update({"noise_index": idx, "energy_fraction": rho})

    # ---- speech leakage into the reference: attenuated, delayed by d/c, mildly coloured
    leak_fir = np.concatenate([[1.0], cfg.leak_fir_std * rng.standard_normal(2)])
    ir = np.convolve(fractional_delay_ir(np.array([d / cfg.c * fs]), np.array([1.0])), leak_fir)
    s_ref = _apply_ir(s, 0, N, ir)
    e_l = _energy(s_ref)
    s_ref *= np.sqrt(e_s * 10.0 ** (leak_db / 10.0) / max(e_l, _EPS))

    # ---- mic self-noise (independent per mic)
    sn_p = rng.standard_normal(N) * 10.0 ** (self_db / 20.0)
    sn_r = rng.standard_normal(N) * 10.0 ** (self_db / 20.0)

    # ---- scale ambient noise to hit the SNR exactly at the primary
    target = e_s / 10.0 ** (snr_db / 10.0)
    e_amb, e_self = _energy(amb_p), _energy(sn_p)
    self_scale = 1.0
    if e_self > 0.5 * target:  # self-noise alone would eat the noise budget: turn it down
        self_scale = np.sqrt(0.5 * target / e_self)
        sn_p *= self_scale
        sn_r *= self_scale
        e_self = _energy(sn_p)
    if e_amb <= _EPS:
        raise ValueError("all noise signals have zero energy")
    # solve ||g*amb + sn||^2 = target for g >= 0 (cross term kept => exact)
    a, b, c = e_amb, 2.0 * float(np.dot(amb_p, sn_p)), e_self - target
    g_amb = (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    n_p = g_amb * amb_p + sn_p
    n_r = g_amb * amb_r + sn_r

    clean = s.copy()
    out_gain = 1.0
    if cfg.max_peak is not None:
        peak = max(np.max(np.abs(clean + n_p)), np.max(np.abs(s_ref + n_r)))
        if peak > cfg.max_peak:
            out_gain = cfg.max_peak / peak
    clean *= out_gain
    n_p *= out_gain
    n_r *= out_gain
    s_ref *= out_gain

    meta = {
        "fs": fs, "n_samples": N, "snr_db": float(snr_db),
        "snr_db_achieved": float(10.0 * np.log10(_energy(clean) / max(_energy(n_p), _EPS))),
        "mic_spacing_m": d, "spatial_aliasing_hz": cfg.c / (2.0 * d),
        "speech_leak_db": leak_db, "leak_fir": leak_fir.tolist(),
        "n_directional": len(src_meta), "diffuse_ratio": rho,
        "self_noise_db": self_db, "self_noise_db_effective": self_db + 20.0 * np.log10(self_scale * out_gain),
        "sources": src_meta, "diffuse": diff_meta, "noise_permutation": perm.tolist(),
        "ambient_gain": float(g_amb), "output_gain": float(out_gain),
    }
    return {"primary": clean + n_p, "reference": s_ref + n_r, "clean": clean,
            "noise_primary": n_p, "noise_reference": n_r, "speech_reference": s_ref, "meta": meta}
