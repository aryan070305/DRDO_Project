"""Reference-microphone adaptive noise cancellers (time domain and STFT domain).

Two-microphone adaptive noise cancelling (ANC) after

    B. Widrow, J. R. Glover Jr., J. M. McCool, J. Kaunitz, C. S. Williams,
    R. H. Hearn, J. R. Zeidler, E. Dong Jr. and R. C. Goodlin,
    "Adaptive noise cancelling: Principles and applications,"
    Proc. IEEE, vol. 63, no. 12, pp. 1692-1716, Dec. 1975.

The *primary* microphone picks up speech plus noise, the *reference*
microphone (outward facing, away from the mouth) picks up mostly noise.  An
adaptive filter maps the reference onto the noise in the primary and the
filter output is subtracted; the error signal is the enhanced speech.

Both cancellers use the normalised LMS (NLMS) update

    J. Nagumo and A. Noda, "A learning method for system identification,"
    IEEE Trans. Automatic Control, vol. AC-12, no. 3, pp. 282-287, 1967.

What the canceller can and cannot remove follows from Widrow et al. (1975):
the best achievable noise reduction in a frequency bin is
``-10 log10(1 - MSC(f))``, where MSC is the magnitude-squared coherence
between the primary-mic noise and the reference-mic noise.  *Coherent*
noise (one or a few directional sources) is cancelled well.  *Diffuse* noise
has coherence ``sinc(2 pi f d / c)`` for mic spacing d, which drops to zero
at ``c / (2 d)`` (about 1.1-2.1 kHz for d = 8-15 cm), so it cannot be
cancelled above roughly that frequency.  The DNN has to handle that
residual.  Speech leaking into the reference makes the canceller remove
speech too (Widrow et al. 1975, Sec. VIII), which is why
adaptation must be frozen while speech is present.  :class:`SubbandNLMS` takes
a speech-presence probability from the DNN as a per-bin step-size scale, in
the same way acoustic echo cancellers gate adaptation during double talk.

:class:`TimeNLMS`
    Sample-by-sample (leaky) NLMS FIR canceller.  Simple and exact.  A causal
    FIR can only model the reference-to-primary path when the noise reaches
    the reference first.  Optionally the primary can be delayed (``delay``),
    which costs that much extra latency.

:class:`SubbandNLMS`
    Multi-frame complex NLMS working per bin on the shared :mod:`danc.dsp`
    STFT frames (20 ms sqrt-Hann, 10 ms hop).  When it shares the analysis
    and synthesis with the DNN it adds **zero** extra algorithmic latency.
    Small negative inter-mic delays (noise reaching the primary first, up to
    about 0.5 ms for a 15 cm spacing) fit inside the analysis window, so the
    STFT-domain filter handles them without a bulk delay.  The per-bin
    multi-frame filter is the "convolutive transfer function" / band-to-band
    approximation of an LTI system in the STFT domain, with cross-band terms
    ignored.  See

        Y. Avargel and I. Cohen, "System identification in the short-time
        Fourier transform domain with crossband filtering," IEEE Trans.
        Audio, Speech, Lang. Process., vol. 15, no. 4, pp. 1305-1319, 2007,

    and for sub-band adaptive filtering in general

        A. Gilloire and M. Vetterli, "Adaptive filtering in subbands with
        critical sampling: analysis, experiments, and application to acoustic
        echo cancellation," IEEE Trans. Signal Process., vol. 40, no. 8,
        pp. 1862-1875, 1992.

    The robustness features are engineering heuristics, not taken from a
    specific paper: impulse gating of the update, a misadjustment-based
    divergence detector that resets a bin, weight-magnitude clipping, and an
    output limiter that never boosts a bin by more than ``out_limit_db``
    over the primary.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from danc.dsp import DEFAULT, STFTConfig, istft, n_frames, stft

__all__ = ["TimeNLMS", "SubbandNLMS", "run_time_nlms", "run_subband_nlms"]

_TINY = 1e-30


def _as_scale(mu_scale, n: int, name: str = "mu_scale") -> np.ndarray:
    """Broadcast a scalar or length-``n`` step-size scale to (n,), clipped to [0, 1]."""
    ms = np.asarray(mu_scale, dtype=np.float64)
    if ms.ndim == 0:
        ms = np.full(n, float(ms))
    elif ms.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have shape ({n},), got {ms.shape}")
    return np.clip(np.nan_to_num(ms, nan=0.0), 0.0, 1.0)


# ----------------------------------------------------------------------------
# time-domain NLMS
# ----------------------------------------------------------------------------

class TimeNLMS:
    """Streaming time-domain (leaky) NLMS adaptive noise canceller.

    Per sample ``n`` (reference tap vector ``x(n) = [r(n), r(n-1), ...,
    r(n-L+1)]``, primary ``d(n)``, possibly delayed by ``delay`` samples)::

        y(n)   = w(n)^T x(n)
        e(n)   = d(n) - y(n)                                   (output)
        w(n+1) = (1 - g * leak) w(n) + g * e(n) x(n) / (eps + ||x(n)||^2),
        g      = mu * mu_scale(n)

    NLMS: Nagumo & Noda (1967).  ANC configuration: Widrow et al. (1975).
    The leakage term is the "tap-leakage" algorithm of R. D. Gitlin,
    H. C. Meadors Jr. and S. B. Weinstein, Bell Syst. Tech. J., vol. 61,
    no. 8, pp. 1817-1839, 1982.  It biases the weights toward zero and
    bounds them when the reference is not persistently exciting.

    Stable for ``0 < mu < 2``.  Smaller ``mu`` gives slower convergence and
    lower misadjustment.

    Parameters
    ----------
    n_taps : int
        FIR length L.  64 taps = 4 ms at 16 kHz.
    mu : float
        NLMS step size.
    eps : float
        Absolute regularisation of the tap-input power.
    leak : float
        Leakage factor (0 = plain NLMS).
    delay : int
        Bulk delay applied to the primary, so the filter can model noise
        that reaches the primary up to ``delay`` samples before the
        reference.  Adds ``delay`` samples of latency.  The output is
        aligned with the *delayed* primary.  Default 0.
    """

    def __init__(self, n_taps: int = 64, mu: float = 0.1, eps: float = 1e-6, leak: float = 0.0,
                 delay: int = 0):
        if n_taps < 1:
            raise ValueError("n_taps must be >= 1")
        if not 0.0 < mu < 2.0:
            raise ValueError("NLMS is only stable for 0 < mu < 2")
        if leak < 0.0 or eps < 0.0 or delay < 0:
            raise ValueError("leak, eps and delay must be non-negative")
        self.n_taps = int(n_taps)
        self.mu = float(mu)
        self.eps = float(eps)
        self.leak = float(leak)
        self.delay = int(delay)
        self.reset()

    def reset(self) -> None:
        """Zero the weights and the reference/primary delay lines."""
        self.w = np.zeros(self.n_taps, dtype=np.float64)
        self._xhist = np.zeros(self.n_taps - 1, dtype=np.float64)  # r(n-L+1) .. r(n-1)
        self._dhist = np.zeros(self.delay, dtype=np.float64)  # last `delay` primary samples

    def process(self, primary_block: np.ndarray, reference_block: np.ndarray, mu_scale=1.0) -> np.ndarray:
        """Filter one block.  State is kept across calls (any block length).

        Parameters
        ----------
        primary_block, reference_block : (N,) float
            Synchronous blocks from the primary and reference microphones.
        mu_scale : float or (N,) array in [0, 1]
            Per-sample step-size scale.  For example ``1 - spp`` freezes
            adaptation during speech.  Filtering always runs.

        Returns
        -------
        (N,) float
            Error signal (enhanced output), aligned with the primary delayed
            by ``self.delay`` samples.
        """
        d = np.asarray(primary_block, dtype=np.float64).reshape(-1)
        x = np.asarray(reference_block, dtype=np.float64).reshape(-1)
        N = d.shape[0]
        if x.shape[0] != N:
            raise ValueError("primary_block and reference_block must have the same length")
        if N == 0:
            return np.zeros(0)
        ms = _as_scale(mu_scale, N)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.delay:
            dext = np.concatenate([self._dhist, d])
            self._dhist = dext[N:].copy()
            d = dext[:N]

        L = self.n_taps
        xe = np.concatenate([self._xhist, x])  # (L-1+N,)
        # X[n] = [x(n), x(n-1), ..., x(n-L+1)]  (reversed sliding windows, a view)
        X = sliding_window_view(xe, L)[:, ::-1]
        c = np.concatenate([[0.0], np.cumsum(xe * xe)])
        pw = np.maximum(c[L:] - c[:-L], 0.0)  # ||x(n)||^2, (N,)
        self._xhist = xe[N:].copy() if L > 1 else self._xhist

        w = self.w
        e = np.empty(N, dtype=np.float64)
        g = self.mu * ms
        leak_f = 1.0 - g * self.leak
        use_leak = self.leak > 0.0
        for n in range(N):
            xn = X[n]
            en = d[n] - w.dot(xn)
            e[n] = en
            gn = g[n]
            if gn > 0.0:
                if use_leak:
                    w *= leak_f[n]
                w += (gn * en / (self.eps + pw[n])) * xn
        if not np.all(np.isfinite(w)):  # divergence / bad input guard
            w[:] = 0.0
        self.w = w
        return e


def run_time_nlms(primary: np.ndarray, reference: np.ndarray, mu_scale=1.0, **kw) -> np.ndarray:
    """Offline helper: run :class:`TimeNLMS` over whole signals.

    ``kw`` is passed to :class:`TimeNLMS`.  When ``delay > 0`` the input is
    flushed with zeros and the output is advanced by ``delay`` samples, so it
    is sample-aligned with ``primary``.  This only works offline; a real-time
    system pays the delay as latency.
    """
    p = np.asarray(primary, dtype=np.float64).reshape(-1)
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    if p.shape != r.shape:
        raise ValueError("primary and reference must have the same length")
    nl = TimeNLMS(**kw)
    D = nl.delay
    if D == 0:
        return nl.process(p, r, mu_scale)
    ms = np.asarray(mu_scale, dtype=np.float64)
    if ms.ndim == 1:
        ms = np.concatenate([ms, np.zeros(D)])
    e = nl.process(np.concatenate([p, np.zeros(D)]), np.concatenate([r, np.zeros(D)]), ms)
    return e[D:]


# ----------------------------------------------------------------------------
# STFT-domain multi-frame complex NLMS
# ----------------------------------------------------------------------------

class SubbandNLMS:
    """Per-bin multi-frame complex NLMS on the shared STFT frames (zero extra latency).

    For every bin f and frame t (``R_hist[f, k] = R[t-k, f]``)::

        Y[f]     = sum_k W[f, k] R[t-k, f]
        E[f]     = P[f] - Y[f]                                   (a-priori error = output)
        W[f, k] += mu * s[f] * E[f] * conj(R[t-k, f]) / D[f]
        D[f]     = max(sum_k |R[t-k, f]|^2, K * Pr[f]) + delta * mean_f(Pr) + eps

    ``Pr`` is the recursively smoothed (``beta``) reference power per bin and
    ``s = mu_scale`` (for example ``1 - SPP`` from the DNN).  The update is the
    complex NLMS gradient step: the Wirtinger derivative of ``|E|^2`` with
    respect to ``conj(W)`` is ``-E conj(R)``.  Taking the max of the
    instantaneous tap-vector norm and the smoothed norm keeps the effective
    step at or below ``mu``, so the usual NLMS stability range
    ``0 < mu < 2`` holds.  The smoothed term lowers gradient noise when the
    instantaneous power dips.  ``delta`` regularises relative to the
    broadband reference power, so near-empty bins do not get huge steps.

    Robustness (heuristics, see module docstring):

    * **Impulses.**  If the broadband reference frame power exceeds
      ``impulse_kappa`` times its slow running average (time constant set by
      ``slow_beta``), the frame still gets filtered.  The weight update, the
      power smoothers and the divergence statistics are frozen for that frame
      and for the next ``n_taps - 1`` frames, while the impulse is still in
      the tap-delay line.  A "loud" period lasting more than
      ``impulse_max_frames`` frames counts as a genuine level change: the
      slow average snaps to the new level and adaptation resumes.
    * **Divergence.**  Per bin, if the smoothed error power exceeds
      ``max_misadjust`` times the smoothed primary power (the canceller is
      adding noise), or anything becomes non-finite, the bin's weights reset
      to zero and that bin's output falls back to the primary.
    * **Weight clipping.**  ``|W[f, k]| <= max_weight`` (phase preserved).
    * **Output limiter.**  ``|E[f]| <= 10^(out_limit_db/20) |P[f]|``, so a
      reference-only transient (a tap on the reference mic, wind) can never
      be injected into the user's ear at more than ``out_limit_db`` above
      the primary.  Pass ``None`` to disable (exactly linear filtering).

    Parameters
    ----------
    n_bins : int
        Number of STFT bins F (161 for the shared 320-point FFT).
    n_taps : int
        Number of frames K in each bin's filter.  4 frames span 20 ms of
        window plus 30 ms of hops, about 50 ms of reference-to-primary
        impulse response.
    mu : float
        NLMS step size, 0 < mu < 2.
    beta : float
        Smoothing constant of the per-bin power estimates (per frame).
    delta : float
        Regularisation relative to the mean (over bins) reference power.
    impulse_kappa : float
        Impulse threshold (power ratio).  ``np.inf`` disables impulse gating.
    max_misadjust : float
        Divergence threshold on smoothed error power / primary power
        (4 = +6 dB).
    max_weight : float
        Magnitude clip of every complex weight.
    out_limit_db : float or None
        Per-bin output limiter (dB above the primary), or None.
    slow_beta : float
        Smoothing of the slow broadband reference power used for impulse
        detection (0.995 per 10 ms frame, about 2 s).
    impulse_max_frames : int
        Consecutive loud frames after which a level change is accepted.
    eps : float
        Absolute regularisation floor.
    """

    def __init__(self, n_bins: int = 161, n_taps: int = 4, mu: float = 0.3, beta: float = 0.9,
                 delta: float = 1e-4, impulse_kappa: float = 20.0, max_misadjust: float = 4.0,
                 max_weight: float = 10.0, out_limit_db: float | None = 6.0, slow_beta: float = 0.995,
                 impulse_max_frames: int = 8, eps: float = 1e-12):
        if not 0.0 < mu < 2.0:
            raise ValueError("NLMS is only stable for 0 < mu < 2")
        if not 0.0 <= beta < 1.0 or not 0.0 <= slow_beta < 1.0:
            raise ValueError("beta and slow_beta must be in [0, 1)")
        if n_bins < 1 or n_taps < 1:
            raise ValueError("n_bins and n_taps must be >= 1")
        self.n_bins = int(n_bins)
        self.n_taps = int(n_taps)
        self.mu = float(mu)
        self.beta = float(beta)
        self.delta = float(delta)
        self.impulse_kappa = float(impulse_kappa)
        self.max_misadjust = float(max_misadjust)
        self.max_weight = float(max_weight)
        self.out_limit = None if out_limit_db is None else float(10.0 ** (out_limit_db / 20.0))
        self.slow_beta = float(slow_beta)
        self.impulse_max_frames = int(impulse_max_frames)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        """Clear weights, delay line and all running statistics."""
        F, K = self.n_bins, self.n_taps
        self.W = np.zeros((F, K), dtype=np.complex128)        # weights, W[f, k] multiplies R[t-k, f]
        self.R_hist = np.zeros((F, K), dtype=np.complex128)   # R_hist[f, k] = R[t-k, f]
        self._pr = np.zeros(F)        # smoothed |R|^2 per bin
        self._pp = np.zeros(F)        # smoothed |P|^2 per bin (divergence detector)
        self._pe = np.zeros(F)        # smoothed |E|^2 per bin (divergence detector)
        self._slow = 0.0              # slow broadband reference power (impulse detector)
        self._loud_run = 0            # consecutive frames above the impulse threshold
        self._hold = 0                # frames left with adaptation frozen after an impulse
        self._initialised = False
        # diagnostics
        self.impulse_flag = False
        self.n_frames = 0
        self.n_impulse_frames = 0
        self.n_bin_resets = 0

    # -- impulse detector ---------------------------------------------------
    def _detect_impulse(self, frame_pow: float) -> bool:
        if not self._initialised or self._slow <= _TINY:
            self._slow = frame_pow
            self._initialised = frame_pow > _TINY
            return False
        if frame_pow > self.impulse_kappa * self._slow:
            self._loud_run += 1
            if self._loud_run > self.impulse_max_frames:  # sustained: a level change, not an impulse
                self._slow = frame_pow
                self._loud_run = 0
                self._hold = 0
                return False
            self._hold = self.n_taps  # current frame + (K-1) frames while it sits in the delay line
            return True
        self._loud_run = 0
        self._slow = self.slow_beta * self._slow + (1.0 - self.slow_beta) * frame_pow
        return False

    # -- one frame ------------------------------------------------------------
    def step(self, P: np.ndarray, R: np.ndarray, mu_scale=1.0) -> np.ndarray:
        """Process one STFT frame.

        Parameters
        ----------
        P, R : (F,) complex
            Primary and reference STFT frames (same analysis, same frame index).
        mu_scale : float or (F,) array in [0, 1]
            Step-size scale.  0 freezes adaptation, as during speech (use
            ``1 - spp``).

        Returns
        -------
        (F,) complex
            Enhanced primary frame E.
        """
        P = np.asarray(P, dtype=np.complex128).reshape(-1)
        R = np.asarray(R, dtype=np.complex128).reshape(-1)
        F = self.n_bins
        if P.shape != (F,) or R.shape != (F,):
            raise ValueError(f"P and R must have shape ({F},)")
        s = _as_scale(mu_scale, F)
        R = np.where(np.isfinite(R), R, 0.0)

        # shift the reference delay line (numpy handles the overlapping copy)
        self.R_hist[:, 1:] = self.R_hist[:, :-1]
        self.R_hist[:, 0] = R
        r2 = R.real ** 2 + R.imag ** 2

        # filter (always) and a-priori error
        Y = np.einsum("fk,fk->f", self.W, self.R_hist)
        E = P - Y

        impulse = self._detect_impulse(float(np.mean(r2)))
        self.impulse_flag = impulse
        self.n_frames += 1
        if impulse:
            self.n_impulse_frames += 1
        frozen = self._hold > 0
        if frozen:
            self._hold -= 1

        bad = ~np.isfinite(E)
        if not frozen:
            b = self.beta
            self._pr = b * self._pr + (1.0 - b) * r2
            p2 = np.where(bad, 0.0, P.real ** 2 + P.imag ** 2)
            e2 = np.where(bad, 0.0, E.real ** 2 + E.imag ** 2)
            self._pp = b * self._pp + (1.0 - b) * p2
            self._pe = b * self._pe + (1.0 - b) * e2

            g = self.mu * s
            if np.any(g > 0.0):
                h2 = self.R_hist.real ** 2 + self.R_hist.imag ** 2
                norm = np.maximum(h2.sum(axis=1), self.n_taps * self._pr)
                denom = norm + self.delta * float(np.mean(self._pr)) + self.eps
                upd = (g * np.where(bad, 0.0, E) / denom)[:, None] * np.conj(self.R_hist)
                self.W += upd

            # weight magnitude clip (phase preserved)
            mag = np.abs(self.W)
            over = mag > self.max_weight
            if np.any(over):
                self.W[over] *= self.max_weight / mag[over]

            # misadjustment-based divergence detection
            diverged = self._pe > self.max_misadjust * self._pp + self.eps
        else:
            diverged = np.zeros(F, dtype=bool)

        reset = diverged | bad | ~np.all(np.isfinite(self.W), axis=1)
        if np.any(reset):
            self.W[reset] = 0.0
            self._pe[reset] = self._pp[reset]
            E = np.where(reset, np.where(np.isfinite(P), P, 0.0), E)
            self.n_bin_resets += int(np.count_nonzero(reset))

        if self.out_limit is not None:
            ae = np.abs(E)
            lim = self.out_limit * np.abs(P)
            scale = np.where(ae > lim, lim / np.maximum(ae, _TINY), 1.0)
            E = E * scale
        return E


def run_subband_nlms(primary: np.ndarray, reference: np.ndarray, spp: np.ndarray | None = None,
                     cfg: STFTConfig = DEFAULT, **kw) -> np.ndarray:
    """Offline helper: STFT -> :class:`SubbandNLMS` frame loop -> iSTFT.

    Parameters
    ----------
    primary, reference : (N,) float
        Microphone signals.
    spp : None, (T,) or (T, F) array in [0, 1]
        Speech-presence probability per frame (or per frame and bin), with
        ``T = danc.dsp.n_frames(N)``.  The step size is scaled by
        ``1 - spp``.  None adapts everywhere.
    cfg : STFTConfig
        The shared STFT configuration.
    **kw
        Passed to :class:`SubbandNLMS` (``n_bins`` comes from ``cfg``).

    Returns
    -------
    (N,) float
        Enhanced primary, sample-aligned with ``primary``.
    """
    p = np.asarray(primary, dtype=np.float64).reshape(-1)
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    if p.shape != r.shape:
        raise ValueError("primary and reference must have the same length")
    Pf = stft(p, cfg)
    Rf = stft(r, cfg)
    T, F = Pf.shape
    if spp is None:
        mus = np.ones((T, 1))
    else:
        spp = np.asarray(spp, dtype=np.float64)
        if spp.shape == (T,):
            mus = (1.0 - spp)[:, None]
        elif spp.shape == (T, F):
            mus = 1.0 - spp
        else:
            raise ValueError(f"spp must have shape ({T},) or ({T}, {F}) (T = n_frames(len(primary)) "
                             f"= {n_frames(len(p), cfg)}), got {spp.shape}")
    mus = np.clip(mus, 0.0, 1.0)
    kw.setdefault("n_bins", F)
    nl = SubbandNLMS(**kw)
    E = np.empty_like(Pf)
    scalar = mus.shape[1] == 1
    for t in range(T):
        E[t] = nl.step(Pf[t], Rf[t], float(mus[t, 0]) if scalar else mus[t])
    return istft(E, len(p), cfg)
