"""Classical single-channel speech-enhancement baselines (causal, STFT domain).

These are the reference points the D-ANC network is compared against.  All of
them run strictly frame-by-frame on the shared ``danc.dsp`` STFT (20 ms
sqrt-Hann window, 10 ms hop at 16 kHz), so their algorithmic latency is the
same 20 ms as the DNN:

* every per-frame quantity (noise PSD, a-priori SNR, gain) at frame ``k`` is a
  function of frames ``0..k`` only - recursions run forward in time and there
  is no look-ahead of any kind (Boll's three-frame residual-noise reduction,
  which needs the *next* frame, is deliberately not used);
* the offline functions (:func:`spectral_subtraction`, :func:`wiener_dd`,
  :func:`logmmse`, :func:`enhance`) use the aligned :func:`danc.dsp.stft` /
  :func:`danc.dsp.istft` pair, so the output is time-aligned with the input
  (convenient for metrics).  Output sample ``n`` then depends only on input
  samples ``x[: n + win]``, the look-ahead being exactly the analysis window,
  i.e. the 20 ms algorithmic latency shared with the DNN;
* :class:`StreamingEnhancer` runs the identical per-frame code on
  :class:`danc.dsp.StreamingSTFT` (hop in, hop out) and reproduces the offline
  output delayed by ``win - hop`` samples, as in ``danc.dsp``.

Methods
-------
``"specsub"``  power spectral subtraction with SNR-dependent over-subtraction
               and a spectral floor (Berouti, Schwartz & Makhoul 1979; after
               Boll 1979).
``"wiener"``   Wiener gain driven by the decision-directed a-priori SNR
               (Scalart & Vieira Filho 1996; DD rule of Ephraim & Malah 1984).
``"logmmse"``  MMSE log-spectral amplitude estimator (Ephraim & Malah 1985)
               with decision-directed a-priori SNR and a gain floor.

All three share one causal noise-PSD tracker, :class:`NoiseTracker`
(MCRA-2, Rangachari & Loizou 2006), initialised from the first ~120 ms which
are assumed noise-only, and tracking continuously afterwards.

Robustness conventions: gains are limited to ``[g_min, 1]`` (no bin is ever
amplified), every PSD is floored at a tiny positive constant, and the output
is guaranteed finite float32 for digital silence and for very loud impulses.

References
----------
S. F. Boll, "Suppression of acoustic noise in speech using spectral
    subtraction", IEEE Trans. ASSP 27(2):113-120, 1979.
M. Berouti, R. Schwartz, J. Makhoul, "Enhancement of speech corrupted by
    acoustic noise", Proc. IEEE ICASSP 1979, pp. 208-211.
Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error
    short-time spectral amplitude estimator", IEEE Trans. ASSP
    32(6):1109-1121, 1984.
Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error
    log-spectral amplitude estimator", IEEE Trans. ASSP 33(2):443-445, 1985.
O. Cappe, "Elimination of the musical noise phenomenon with the Ephraim and
    Malah noise suppressor", IEEE Trans. Speech Audio Proc. 2(2):345-349, 1994.
P. Scalart, J. Vieira Filho, "Speech enhancement based on a priori signal to
    noise estimation", Proc. IEEE ICASSP 1996, vol. 2, pp. 629-632.
G. Doblinger, "Computationally efficient speech enhancement by spectral minima
    tracking in subbands", Proc. EUROSPEECH 1995, pp. 1513-1516.
I. Cohen, B. Berdugo, "Noise estimation by minima controlled recursive
    averaging for robust speech enhancement", IEEE Signal Processing Letters
    9(1):12-15, 2002.
S. Rangachari, P. C. Loizou, "A noise-estimation algorithm for highly
    non-stationary environments", Speech Communication 48(2):220-231, 2006.
P. C. Loizou, "Speech Enhancement: Theory and Practice", 2nd ed., CRC Press,
    2013 (reference MATLAB implementations of the above).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.special import exp1

from danc.dsp import DEFAULT, STFTConfig, StreamingSTFT, istft, stft

__all__ = [
    "METHODS",
    "NoiseTracker",
    "FrameEnhancer",
    "StreamingEnhancer",
    "spectral_subtraction",
    "wiener_dd",
    "logmmse",
    "enhance",
    "stft_config_for",
]

METHODS: tuple[str, ...] = ("specsub", "wiener", "logmmse")

_ALIASES = {
    "specsub": "specsub",
    "spectral_subtraction": "specsub",
    "berouti": "specsub",
    "wiener": "wiener",
    "wiener_dd": "wiener",
    "logmmse": "logmmse",
    "lsa": "logmmse",
    "mmse_lsa": "logmmse",
}

# Absolute floor for every power / PSD value.  The algorithms are scale
# invariant; this only guards divisions for digital silence.
_PSD_FLOOR = 1e-30
# Cap on the a-posteriori SNR: far beyond the point where every gain is 1,
# keeps all intermediate values bounded for absurdly loud inputs.
_GAMMA_MAX = 1e15
# Lower bound on the LSA integral argument v (E1(v) -> inf as v -> 0).
_V_MIN = 1e-12

# Per-method parameters accepted by FrameEnhancer (and the public functions)
# with their defaults.  Unknown keyword arguments raise TypeError.
_COMMON_DEFAULTS: dict[str, Any] = {"init_ms": 120.0, "tracker_kw": None}
_METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "specsub": {"alpha": None, "beta": 0.02, "alpha0": 4.0},
    "wiener": {"alpha_dd": 0.98, "xi_min_db": -25.0, "g_min_db": -20.0},
    "logmmse": {"alpha_dd": 0.98, "xi_min_db": -25.0, "g_min_db": -20.0},
}


def stft_config_for(fs: int) -> STFTConfig:
    """STFT configuration with the project's 20 ms / 10 ms framing at rate ``fs``.

    Returns :data:`danc.dsp.DEFAULT` for 16 kHz.  For other rates the window is
    the even number of samples closest to 20 ms (hop = half of it, n_fft = win),
    so the latency in milliseconds stays comparable to the 16 kHz system.
    """
    fs = int(fs)
    if fs == DEFAULT.fs:
        return DEFAULT
    if fs < 400:
        raise ValueError(f"unsupported sample rate fs={fs}")
    hop = int(round(0.010 * fs))
    return STFTConfig(fs=fs, win=2 * hop, hop=hop, n_fft=2 * hop)


def _canonical_method(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_")
    if key not in _ALIASES:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    return _ALIASES[key]


# ----------------------------------------------------------------------------
# Noise PSD tracking: MCRA-2
# ----------------------------------------------------------------------------

class NoiseTracker:
    r"""Causal frame-by-frame noise-PSD estimator (MCRA-2).

    Implements the minima-controlled recursive averaging variant of
    Rangachari & Loizou (2006), which combines MCRA (Cohen & Berdugo 2002) with
    Doblinger's (1995) continuous spectral-minimum tracking.  For every frame
    ``lam`` and bin ``k``, with noisy periodogram :math:`|Y|^2`:

    1. smoothed noisy power   :math:`P = \eta P_{-1} + (1-\eta)|Y|^2`
    2. continuous minimum     if :math:`P_{min,-1} < P`:
       :math:`P_{min} = \gamma P_{min,-1} + \frac{1-\gamma}{1-\beta}(P - \beta P_{-1})`,
       else :math:`P_{min} = P`
    3. speech-presence decision :math:`I = [P / P_{min} > \delta(k)]` with a
       frequency-dependent threshold (``delta`` below 1 kHz, 1-3 kHz, above 3 kHz)
    4. speech-presence probability :math:`p = \alpha_p p_{-1} + (1-\alpha_p) I`
    5. time-frequency smoothing factor :math:`\alpha_s = \alpha_d + (1-\alpha_d) p`
    6. noise PSD :math:`\hat\lambda_d = \alpha_s \hat\lambda_{d,-1} + (1-\alpha_s)|Y|^2`

    The defaults (eta=0.7, gamma=0.998, beta=0.8, alpha_p=0.2, alpha_d=0.85,
    delta=(2, 2, 5), band edges 1/3 kHz) are the values of Rangachari & Loizou
    (2006) as best recalled; they were designed for a 10 ms frame advance,
    which matches the D-ANC STFT.  Some descriptions of Doblinger's tracker use
    beta=0.96 - the value is not critical (it only shapes how fast the minimum
    rises during increasing power).

    Initialisation: the first ``init_ms`` of input are assumed to be noise only;
    during that period the estimate is the running mean of the periodograms and
    the MCRA-2 state is seeded from it.  Tracking then continues for the whole
    signal (the estimate is never frozen on account of speech).

    Robustness guards added on top of the published algorithm (they do not
    change its behaviour on ordinary speech-in-noise):

    * digital silence (a frame whose periodogram is entirely <= ``floor``, e.g.
      exact zeros from a muted channel or zero padding) carries no information
      about the acoustic noise: it neither counts towards the initialisation
      period nor updates any state, and the current estimate is returned.
      Without this, a stream that starts with zeros would initialise the noise
      PSD to ~0 and need seconds to recover;
    * if the step-2 recursion would make :math:`P_{min}` non-positive (possible
      when :math:`P` falls quickly from more than ~30 dB above the minimum, e.g.
      at the end of a loud burst) the update falls back to
      :math:`\gamma P_{min,-1}`; otherwise the published update is used;
    * outlier clamps, so that a single huge impulse (gunshot, key click) can
      neither inflate the noise PSD nor reset the minimum tracker to a high
      level (unguarded, a +80 dB click keeps :math:`P` - and hence
      :math:`P_{min}` - elevated for ~0.5 s, during which speech leaks into
      the noise estimate and is then over-suppressed).  The periodogram entering
      steps 1-3 is limited to ``power_clamp`` (1000, +30 dB) times the previous
      noise estimate, and the one entering step 6 to ``update_clamp`` (10).
      For a noise-only bin the periodogram is ~exponentially distributed, so
      ``update_clamp=10`` clips it with probability ``exp(-10) ~ 5e-5``
      (negligible bias) while the estimate can still grow by up to
      ``alpha_d + (1 - alpha_d) * update_clamp`` (= 2.35) per frame; the
      +30 dB ``power_clamp`` only slows the tracking of noise-level jumps of
      more than ~20 dB by a fraction of a second.  ``None`` disables a clamp
      (both ``None`` = published MCRA-2).

    Measured on stationary white noise the estimate is ~1 dB low on average
    (~3 dB below 3 kHz): with delta=2, high-power noise frames are often
    flagged as speech and excluded from the average.  This bias is inherent to
    the decision rule (MCRA-type estimators are known to underestimate without
    a bias correction) and errs on the side of less suppression.

    Parameters
    ----------
    n_bins : number of STFT bins (``n_fft // 2 + 1``).
    fs, hop : sample rate and frame advance, used for the band thresholds and to
        convert ``init_ms`` into frames.
    init_ms : length of the initial noise-only period (default 120 ms).
    floor : absolute lower bound of every PSD value.
    """

    def __init__(
        self,
        n_bins: int = DEFAULT.n_bins,
        fs: int = DEFAULT.fs,
        hop: int = DEFAULT.hop,
        init_ms: float = 120.0,
        *,
        eta: float = 0.7,
        gamma: float = 0.998,
        beta: float = 0.8,
        alpha_p: float = 0.2,
        alpha_d: float = 0.85,
        delta: tuple[float, float, float] = (2.0, 2.0, 5.0),
        band_edges_hz: tuple[float, float] = (1000.0, 3000.0),
        power_clamp: float | None = 1000.0,
        update_clamp: float | None = 10.0,
        floor: float = _PSD_FLOOR,
    ):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        for name, val in (("eta", eta), ("gamma", gamma), ("alpha_p", alpha_p), ("alpha_d", alpha_d)):
            if not 0.0 <= val < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {val}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if init_ms < 0:
            raise ValueError("init_ms must be >= 0")
        for name, val in (("power_clamp", power_clamp), ("update_clamp", update_clamp)):
            if val is not None and not val > 1.0:
                raise ValueError(f"{name} must be > 1 or None, got {val}")
        if floor <= 0:
            raise ValueError("floor must be > 0")

        self.n_bins = int(n_bins)
        self.fs = int(fs)
        self.hop = int(hop)
        self.init_frames = max(1, int(round(init_ms * 1e-3 * self.fs / self.hop)))
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.alpha_p = float(alpha_p)
        self.alpha_d = float(alpha_d)
        self.power_clamp = None if power_clamp is None else float(power_clamp)
        self.update_clamp = None if update_clamp is None else float(update_clamp)
        self.floor = float(floor)

        freqs = np.linspace(0.0, self.fs / 2.0, self.n_bins)
        f_low, f_mid = band_edges_hz
        d_low, d_mid, d_high = delta
        self._delta = np.where(freqs <= f_low, d_low, np.where(freqs <= f_mid, d_mid, d_high))
        self._rise = (1.0 - self.gamma) / (1.0 - self.beta)
        self.reset()

    def reset(self) -> None:
        """Forget all state (the next frame starts a new initialisation period)."""
        n = self.n_bins
        self._frame = 0
        self._init_sum = np.zeros(n)
        self._P = np.zeros(n)
        self._P_min = np.zeros(n)
        self._spp = np.zeros(n)
        self._noise = np.full(n, self.floor)

    @property
    def noise_psd(self) -> np.ndarray:
        """Current noise PSD estimate, shape ``(n_bins,)`` (copy)."""
        return self._noise.copy()

    @property
    def speech_presence(self) -> np.ndarray:
        """Current MCRA-2 speech-presence probability per bin (copy)."""
        return self._spp.copy()

    @property
    def initialised(self) -> bool:
        """True once the noise-only initialisation period is over."""
        return self._frame >= self.init_frames

    def update(self, power_frame: np.ndarray) -> np.ndarray:
        """Consume one noisy periodogram ``|Y(k)|^2`` and return the noise PSD.

        Parameters
        ----------
        power_frame : non-negative array of shape ``(n_bins,)`` for the current
            frame (it is included in the returned estimate - still causal).

        Returns
        -------
        Noise PSD estimate for this frame, shape ``(n_bins,)``, float64, >= floor.
        """
        p = np.asarray(power_frame, dtype=np.float64)
        if p.shape != (self.n_bins,):
            raise ValueError(f"power_frame must have shape ({self.n_bins},), got {p.shape}")
        if not np.all(np.isfinite(p)):
            raise ValueError("power_frame contains non-finite values")
        p = np.maximum(p, 0.0)
        if p.max() <= self.floor:
            # digital silence: no information about the noise, keep all state
            return self._noise.copy()

        if self._frame < self.init_frames:
            # Noise-only initialisation: running mean, seeds the MCRA-2 state.
            self._frame += 1
            self._init_sum += p
            mean = np.maximum(self._init_sum / self._frame, self.floor)
            self._noise = mean
            self._P = mean.copy()
            self._P_min = mean.copy()
            self._spp[:] = 0.0
            return self._noise.copy()
        self._frame += 1

        # 1) recursively smoothed noisy power (impulse-limited input)
        p_s = p if self.power_clamp is None else np.minimum(p, self.power_clamp * self._noise)
        P_prev = self._P
        P = self.eta * P_prev + (1.0 - self.eta) * p_s
        # 2) continuous minimum tracking (Doblinger 1995); positivity guard
        rise = self.gamma * self._P_min + self._rise * (P - self.beta * P_prev)
        rise = np.where(rise > 0.0, rise, self.gamma * self._P_min)
        P_min = np.where(self._P_min < P, rise, P)
        P_min = np.maximum(P_min, self.floor)
        # 3) + 4) speech-presence decision and its recursive probability
        present = (P > self._delta * P_min).astype(np.float64)
        spp = self.alpha_p * self._spp + (1.0 - self.alpha_p) * present
        # 5) + 6) time-frequency dependent recursive averaging of the noise PSD
        alpha_s = self.alpha_d + (1.0 - self.alpha_d) * spp
        p_u = p if self.update_clamp is None else np.minimum(p, self.update_clamp * self._noise)
        noise = alpha_s * self._noise + (1.0 - alpha_s) * p_u

        self._P, self._P_min, self._spp = P, P_min, spp
        self._noise = np.maximum(noise, self.floor)
        return self._noise.copy()


# ----------------------------------------------------------------------------
# Per-frame gain rules
# ----------------------------------------------------------------------------

def _berouti_gain(power: np.ndarray, noise: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Power spectral subtraction with over-subtraction and spectral floor.

    |S|^2 = |Y|^2 - alpha*N   if that exceeds beta*N, else beta*N   (Berouti et al. 1979),
    expressed as an amplitude gain on Y and limited to <= 1.
    """
    ratio = noise / np.maximum(power, _PSD_FLOOR)
    g2 = np.maximum(1.0 - alpha * ratio, beta * ratio)
    return np.sqrt(np.minimum(g2, 1.0))


def _lsa_gain(xi: np.ndarray, gamma_post: np.ndarray) -> np.ndarray:
    """Ephraim & Malah (1985) log-spectral amplitude gain.

    G = xi / (1 + xi) * exp(0.5 * E1(v)),  v = xi * gamma / (1 + xi),
    with E1 the exponential integral.  Not clipped here.
    """
    r = xi / (1.0 + xi)
    v = np.maximum(r * gamma_post, _V_MIN)
    return r * np.exp(0.5 * exp1(v))


# ----------------------------------------------------------------------------
# Causal frame processor (shared by offline and streaming paths)
# ----------------------------------------------------------------------------

class FrameEnhancer:
    """Causal per-frame STFT-domain enhancer for one of :data:`METHODS`.

    ``process_frame`` maps one complex STFT frame ``Y_k`` to ``G_k * Y_k``; the
    gain depends only on frames ``0..k``.  Both the offline functions and
    :class:`StreamingEnhancer` are thin loops around this class, so they are
    numerically identical.

    Keyword parameters (all optional)
    ---------------------------------
    Common:
        init_ms (120.0)       noise-only initialisation period of the tracker.
        tracker_kw (None)     extra keyword arguments for :class:`NoiseTracker`.
    ``"specsub"`` (Berouti, Schwartz & Makhoul 1979):
        alpha (None)          over-subtraction factor; ``None`` -> SNR dependent
                              ``alpha = alpha0 - 3/20 * SNR_dB``, SNR clipped to
                              [-5, 20] dB (i.e. 4.75 ... 1).  The frame SNR is the
                              a-posteriori ratio sum|Y|^2 / sum N, as in Loizou's
                              (2013) reference implementation.
        beta (0.02)           spectral floor (fraction of the noise power).
        alpha0 (4.0)          over-subtraction at 0 dB SNR.
    ``"wiener"`` / ``"logmmse"``:
        alpha_dd (0.98)       decision-directed smoothing (Ephraim & Malah 1984).
        xi_min_db (-25.0)     a-priori SNR floor (Cappe 1994 discusses ~-25 dB
                              as a musical-noise / distortion trade-off).
        g_min_db (-20.0)      gain floor; limits the residual-noise attenuation
                              and keeps the background natural.
    """

    def __init__(self, method: str = "logmmse", fs: int = DEFAULT.fs, **kw: Any):
        self.method = _canonical_method(method)
        self.cfg = stft_config_for(fs)
        params = {**_COMMON_DEFAULTS, **_METHOD_DEFAULTS[self.method]}
        unknown = set(kw) - set(params)
        if unknown:
            raise TypeError(
                f"unexpected parameter(s) {sorted(unknown)} for method {self.method!r}; "
                f"accepted: {sorted(params)}"
            )
        params.update(kw)
        self.params: Mapping[str, Any] = params

        tracker_kw = dict(params["tracker_kw"] or {})
        tracker_kw.setdefault("init_ms", params["init_ms"])
        self.tracker = NoiseTracker(self.cfg.n_bins, self.cfg.fs, self.cfg.hop, **tracker_kw)

        if self.method == "specsub":
            alpha = params["alpha"]
            if alpha is not None and alpha <= 0:
                raise ValueError("alpha must be > 0 or None")
            if not params["beta"] >= 0:
                raise ValueError("beta must be >= 0")
        else:
            if not 0.0 <= params["alpha_dd"] < 1.0:
                raise ValueError("alpha_dd must be in [0, 1)")
            self._xi_min = 10.0 ** (params["xi_min_db"] / 10.0)
            self._g_min = 10.0 ** (params["g_min_db"] / 20.0)
            if not 0.0 <= self._g_min <= 1.0:
                raise ValueError("g_min_db must be <= 0")
        self.reset()

    def reset(self) -> None:
        """Reset the noise tracker and the decision-directed memory."""
        self.tracker.reset()
        # |A_{k-1}|^2 / lambda_d(k-1) = G_{k-1}^2 * gamma_{k-1}; zero before the first
        # frame (no prior speech evidence).
        self._dd_prev = np.zeros(self.cfg.n_bins)
        self.last_gain = np.ones(self.cfg.n_bins)

    def process_frame(self, Y: np.ndarray) -> np.ndarray:
        """Enhance one complex STFT frame of shape ``(n_bins,)``; returns ``G * Y``."""
        Y = np.asarray(Y)
        power = Y.real * Y.real + Y.imag * Y.imag
        noise = self.tracker.update(power)
        if self.method == "specsub":
            gain = self._gain_specsub(power, noise)
        else:
            gain = self._gain_dd(power, noise)
        self.last_gain = gain
        return gain * Y

    # -- gain rules -----------------------------------------------------------

    def _gain_specsub(self, power: np.ndarray, noise: np.ndarray) -> np.ndarray:
        p = self.params
        alpha = p["alpha"]
        if alpha is None:
            snr_db = 10.0 * np.log10(max(power.sum(), _PSD_FLOOR) / max(noise.sum(), _PSD_FLOOR))
            alpha = p["alpha0"] - (3.0 / 20.0) * float(np.clip(snr_db, -5.0, 20.0))
        return _berouti_gain(power, noise, alpha, p["beta"])

    def _gain_dd(self, power: np.ndarray, noise: np.ndarray) -> np.ndarray:
        a = self.params["alpha_dd"]
        gamma_post = np.minimum(power / noise, _GAMMA_MAX)
        # decision-directed a-priori SNR (Ephraim & Malah 1984, eq. 51)
        xi = a * self._dd_prev + (1.0 - a) * np.maximum(gamma_post - 1.0, 0.0)
        xi = np.maximum(xi, self._xi_min)
        if self.method == "wiener":
            gain = xi / (1.0 + xi)
        else:
            gain = _lsa_gain(xi, gamma_post)
        gain = np.clip(gain, self._g_min, 1.0)
        self._dd_prev = gain * gain * gamma_post
        return gain


class StreamingEnhancer:
    """Real-time wrapper: ``hop`` input samples in, ``hop`` output samples out.

    Uses :class:`danc.dsp.StreamingSTFT` around a :class:`FrameEnhancer`.  The
    output stream equals the offline result delayed by ``win - hop`` samples
    (160 at 16 kHz); including one hop of block buffering the wall-clock
    algorithmic latency is ``win`` (20 ms), the same as the DNN.

    Example::

        se = StreamingEnhancer("logmmse")
        for block in blocks:          # each block: 160 samples at 16 kHz
            out = se.process(block)
    """

    def __init__(self, method: str = "logmmse", fs: int = DEFAULT.fs, **kw: Any):
        self.enhancer = FrameEnhancer(method, fs, **kw)
        self.cfg = self.enhancer.cfg
        self._stft = StreamingSTFT(self.cfg)

    @property
    def hop(self) -> int:
        return self.cfg.hop

    @property
    def delay_samples(self) -> int:
        """Index delay of the output stream relative to the input stream."""
        return self.cfg.win - self.cfg.hop

    def reset(self) -> None:
        self.enhancer.reset()
        self._stft.reset()

    def process(self, block: np.ndarray) -> np.ndarray:
        """Process exactly ``hop`` finite samples; returns ``hop`` float32 samples."""
        block = np.asarray(block, dtype=np.float64).reshape(-1)
        if block.shape[0] != self.cfg.hop:
            raise ValueError(f"block must have {self.cfg.hop} samples, got {block.shape[0]}")
        Y = self._stft.analyze(block)[0]
        out = self._stft.synthesize(self.enhancer.process_frame(Y))
        return _to_float32(out)


# ----------------------------------------------------------------------------
# Offline API
# ----------------------------------------------------------------------------

def _as_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {x.shape}")
    x = x.astype(np.float64, copy=False)
    if not np.all(np.isfinite(x)):
        raise ValueError("input contains NaN or Inf")
    return x


def _to_float32(y: np.ndarray) -> np.ndarray:
    big = float(np.finfo(np.float32).max)
    return np.clip(y, -big, big).astype(np.float32)


def _run(x: np.ndarray, method: str, fs: int, **kw: Any) -> np.ndarray:
    x = _as_signal(x)
    enh = FrameEnhancer(method, fs, **kw)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    X = stft(x, enh.cfg)
    Y = np.empty_like(X)
    for k in range(X.shape[0]):  # strictly forward in time
        Y[k] = enh.process_frame(X[k])
    return _to_float32(istft(Y, x.shape[0], enh.cfg))


def spectral_subtraction(
    x: np.ndarray,
    fs: int = 16000,
    alpha: float | None = None,
    beta: float = 0.02,
    **kw: Any,
) -> np.ndarray:
    """Causal power spectral subtraction (Berouti, Schwartz & Makhoul 1979).

    Boll's (1979) spectral subtraction with Berouti's over-subtraction factor
    ``alpha`` (SNR-dependent per frame when ``None``) and spectral floor
    ``beta``; noise PSD from :class:`NoiseTracker`; noisy phase is kept.

    Parameters
    ----------
    x : 1-D finite signal.
    fs : sample rate (20 ms / 10 ms framing at any rate, see :func:`stft_config_for`).
    alpha : fixed over-subtraction factor, or ``None`` for
        ``alpha0 - 3/20 * SNR_dB`` with SNR clipped to [-5, 20] dB.
    beta : spectral floor relative to the noise power.
    **kw : ``alpha0``, ``init_ms``, ``tracker_kw`` (see :class:`FrameEnhancer`).

    Returns
    -------
    float32 array with the same length as ``x``, time-aligned with it.
    """
    return _run(x, "specsub", fs, alpha=alpha, beta=beta, **kw)


def wiener_dd(x: np.ndarray, fs: int = 16000, **kw: Any) -> np.ndarray:
    """Causal Wiener filter with decision-directed a-priori SNR.

    ``G = xi / (1 + xi)`` with ``xi`` from the decision-directed rule of
    Ephraim & Malah (1984), as proposed by Scalart & Vieira Filho (1996).

    ``**kw``: ``alpha_dd`` (0.98), ``xi_min_db`` (-25), ``g_min_db`` (-20),
    ``init_ms`` (120), ``tracker_kw`` - see :class:`FrameEnhancer`.

    Returns a float32 array with the same length as ``x``, time-aligned with it.
    """
    return _run(x, "wiener", fs, **kw)


def logmmse(x: np.ndarray, fs: int = 16000, **kw: Any) -> np.ndarray:
    """Causal MMSE log-spectral amplitude estimator (Ephraim & Malah 1985).

    ``G = xi/(1+xi) * exp(0.5 * E1(xi*gamma/(1+xi)))`` with decision-directed
    ``xi`` (Ephraim & Malah 1984), clipped to ``[g_min, 1]``.

    ``**kw``: ``alpha_dd`` (0.98), ``xi_min_db`` (-25), ``g_min_db`` (-20),
    ``init_ms`` (120), ``tracker_kw`` - see :class:`FrameEnhancer`.

    Returns a float32 array with the same length as ``x``, time-aligned with it.
    """
    return _run(x, "logmmse", fs, **kw)


def enhance(x: np.ndarray, method: str, fs: int = 16000, **kw: Any) -> np.ndarray:
    """Dispatch to a baseline by name: ``"specsub"``, ``"wiener"`` or ``"logmmse"``.

    A few aliases are accepted (``"spectral_subtraction"``, ``"wiener_dd"``,
    ``"lsa"``, ...); ``**kw`` is forwarded to the selected method.
    """
    return _run(x, _canonical_method(method), fs, **kw)
