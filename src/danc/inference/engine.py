"""Real-time hybrid noise-cancellation engine (reference-mic NLMS -> DANCNet).

Signal flow per 10 ms hop (160 samples @ 16 kHz)

    primary mic ─┐                    ┌───────────────┐
                 ├─ shared STFT ─► P ─►│ sub-band NLMS │─► E ─► DANCNet ─► Y ─► iSTFT ─► limiter ─► out
    reference ───┘               ► R ─►│  (per bin)    │         │  spp
                                       └──────▲────────┘         │
                                              └── mu = 1 - SPP ◄─┘  (adaptation frozen while talking)

Why this order (adaptive filter FIRST, network SECOND)
  * The linear canceller removes the part of the noise that is *coherent* between the
    two microphones (low-frequency engine/rotor/vehicle noise from directional sources)
    without any speech distortion, raising the SNR seen by the network.
  * The network then removes what a linear filter cannot: diffuse noise (coherence
    sinc(2*pi*f*d/c) collapses above ~c/(2d) ~ 1.1-2 kHz for d = 8-15 cm), non-linear
    and impulsive components, and residual noise.
  * The network's speech-presence output gates the canceller's step size, the analogue of
    double-talk detection in echo cancellers: speech leaking into the reference mic would
    otherwise be cancelled from the primary.
  * Both share ONE analysis/synthesis STFT, so the canceller adds zero algorithmic latency.
  * If the reference mic fails or is absent, the engine degrades gracefully to DNN-only.  With a two-microphone
    network, an optional single-mic fallback_runner is kept in hot standby and cross-faded in when the
    reference goes dead (reference-health monitor, EngineConfig.ref_*).

Total algorithmic latency = 20 ms (window), compute must fit in one 10 ms hop.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from danc.adaptive.nlms import SubbandNLMS
from danc.dsp import DEFAULT, StreamingSTFT


@dataclass
class EngineConfig:
    use_lms: bool = True
    spp_gate: bool = True
    # NLMS <-> DNN coupling: tuned on the dual-mic VALIDATION scenes (data/testsets/dualmic_val,
    # scripts/tune_hybrid.py; first version used mu=0.3, taps=4, gate_power=2 - see report section 8.4)
    lms_mu: float = 0.1
    lms_taps: int = 2
    spp_band: tuple = (6, 70)          # bins (300 .. 3500 Hz) used for the global speech-presence
    gate_power: float = 4.0             # mu *= (1 - spp_bin) * (1 - spp_global) ** gate_power
    gate_threshold: float | None = None # if set: adaptation fully frozen while spp_global >= threshold
    gain_floor_db: float | None = None  # optional residual "comfort" floor: y = Y + g*E
    limiter_dbfs: float = -1.0          # output peak limiter (hearing protection / DAC headroom)
    limiter_release: float = 0.999      # per-sample release coefficient
    timing_window: int = 1_000_000      # per-hop timings kept (bounded deque; ~2.8 h of hops at 10 ms)
    lms_two_mic: bool = False           # two-mic network: run the gated NLMS FIRST and feed (E, R) instead of (P, R)
                                        # (experimental cascade; default: the two-mic network bypasses the NLMS)
    # reference-mic health monitor - active only for a two-microphone network WITH a single-mic fallback_runner:
    # Levels are DC-removed and smoothed over ~100 ms.  Evidence per hop:
    #   dead : reference below ref_floor_dbfs (digital silence), or ref_rel_dead_db below an ACTIVE primary
    #   alive: reference within ref_rel_alive_db of an active primary
    #   none : both channels quiet (quiet room, speech pause) -> the state is kept, so nothing flaps
    ref_floor_dbfs: float = -100.0      # absolute floor: an unplugged USB channel / no bias reads at or below this
    ref_active_dbfs: float = -60.0      # the relative test needs the primary above this level
    ref_rel_dead_db: float = 40.0       # reference this far below an active primary => dead evidence
    ref_rel_alive_db: float = 30.0      # reference within this of an active primary => alive evidence
    ref_dead_s: float = 0.5             # consecutive evidence needed to switch (either direction)
    ref_xfade_hops: int = 10            # cross-fade between the two networks (100 ms, no click)


class HybridEngine:
    def __init__(self, runner=None, cfg: EngineConfig | None = None, fallback_runner=None):
        self.cfg = cfg or EngineConfig()
        self.runner = runner                 # None -> NLMS-only (no DNN)
        # optional single-mic network kept in hot standby next to a two-mic network: if the reference microphone
        # goes dead (unplugged, broken cable) the engine cross-fades to it instead of feeding the two-mic network
        # an input it never saw in training
        self.fallback = fallback_runner
        if fallback_runner is not None:
            if getattr(runner, "n_mics", 1) != 2 or getattr(fallback_runner, "n_mics", 1) != 1:
                raise ValueError("fallback_runner needs a two-microphone main runner and a single-microphone fallback")
            if int(getattr(runner, "lookahead", 0)) != int(getattr(fallback_runner, "lookahead", 0)):
                raise ValueError("fallback_runner must have the same look-ahead as the main runner (output alignment)")
        self.stft = StreamingSTFT(DEFAULT, channels=2)
        self.lms = SubbandNLMS(DEFAULT.n_bins, n_taps=self.cfg.lms_taps, mu=self.cfg.lms_mu) if self.cfg.use_lms else None
        self.reset()

    @property
    def lookahead(self) -> int:
        """Look-ahead frames of the network (adds lookahead * 10 ms of latency)."""
        return int(getattr(self.runner, "lookahead", 0)) if self.runner is not None else 0

    @property
    def latency_ms(self) -> float:
        return DEFAULT.latency_ms + 1000.0 * DEFAULT.hop / DEFAULT.fs * self.lookahead

    def reset(self):
        self.stft.reset()
        if self.lms is not None:
            self.lms.reset()
        if self.runner is not None:
            self.runner.reset()
        if self.fallback is not None:
            self.fallback.reset()
        self.ref_dead, self._ref_low, self._ref_high, self._mix, self.ref_switches = False, 0, 0, 0.0, 0
        self._lp, self._lr = None, None      # smoothed primary / reference level [dB] (DC removed)
        self.spp = np.zeros(DEFAULT.n_bins)
        self.lim_gain = 1.0
        self.timing = deque(maxlen=self.cfg.timing_window)   # bounded: constant memory in long live runs

    # ------------------------------------------------------------------ core
    def _mu_scale(self):
        if not self.cfg.spp_gate:
            return 1.0
        a, b = self.cfg.spp_band
        s_glob = float(np.mean(self.spp[a:b]))
        if self.cfg.gate_threshold is not None and s_glob >= self.cfg.gate_threshold:
            return 0.0
        return np.clip((1.0 - self.spp) * (1.0 - s_glob) ** self.cfg.gate_power, 0.0, 1.0)

    def _limit(self, y):
        thr = 10 ** (self.cfg.limiter_dbfs / 20)
        out = np.empty_like(y)
        g = self.lim_gain
        rel = self.cfg.limiter_release
        for i, v in enumerate(y):
            g = 1.0 - (1.0 - g) * rel                  # smooth release towards unity gain
            a = abs(v)
            if a * g > thr:                            # instantaneous attack: never exceed thr
                g = thr / a
            out[i] = v * g
        self.lim_gain = g
        return out

    def _ref_health(self, prim, ref):
        """Track whether the reference microphone is alive (two-mic network with a fallback only)."""
        c = self.cfg
        hold = max(1, int(round(c.ref_dead_s * DEFAULT.fs / DEFAULT.hop)))
        if ref is None:                                       # no reference channel at all: switch at once
            self._ref_low, self._ref_high = hold, 0
        else:
            pw = lambda x: float(np.mean(np.square(x - np.mean(x))))      # noqa: E731  (DC removed)
            p = 10.0 * np.log10(pw(np.asarray(prim, dtype=np.float64)) + 1e-20)
            r = 10.0 * np.log10(pw(np.asarray(ref, dtype=np.float64)) + 1e-20)
            a = 0.9                                           # ~100 ms smoothing (in dB: a drop shows within ~10 hops)
            self._lp = p if self._lp is None else a * self._lp + (1 - a) * p
            self._lr = r if self._lr is None else a * self._lr + (1 - a) * r
            lp, lr = self._lp, self._lr
            active = lp > c.ref_active_dbfs
            if lr < c.ref_floor_dbfs or (active and lr < lp - c.ref_rel_dead_db):
                self._ref_low, self._ref_high = self._ref_low + 1, 0
            elif active and lr >= lp - c.ref_rel_alive_db:
                self._ref_low, self._ref_high = 0, self._ref_high + 1
            else:                                             # no evidence (quiet) or in between: keep the state
                self._ref_low, self._ref_high = 0, 0
        if not self.ref_dead and self._ref_low >= hold:
            self.ref_dead, self.ref_switches = True, self.ref_switches + 1
        elif self.ref_dead and self._ref_high >= hold:
            self.ref_dead, self.ref_switches = False, self.ref_switches + 1

    def process_block(self, prim: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        if self.fallback is not None:
            self._ref_health(prim, ref)
        ref = np.zeros_like(prim) if ref is None else ref
        P, R = self.stft.analyze(np.stack([prim, ref]))
        two_mic_net = self.runner is not None and getattr(self.runner, "n_mics", 1) == 2
        if two_mic_net and not (self.cfg.lms_two_mic and self.lms is not None):
            E = P   # the two-microphone network does its own (learned) spatial processing; NLMS is bypassed
        else:
            E = self.lms.step(P, R, self._mu_scale()) if (self.lms is not None and np.any(ref)) else P
        if self.runner is not None:
            spec = np.stack([E.real, E.imag], -1)[None, None].astype(np.float32)
            if two_mic_net:
                y, spp = self.runner(spec, np.stack([R.real, R.imag], -1)[None, None].astype(np.float32))
                if self.fallback is not None:                 # hot standby: its recurrent state stays current
                    y1, spp1 = self.fallback(spec)
                    step = 1.0 / max(1, self.cfg.ref_xfade_hops)
                    self._mix = min(1.0, self._mix + step) if self.ref_dead else max(0.0, self._mix - step)
                    if self._mix >= 1.0:                      # fully switched: never pass on the two-mic output
                        y, spp = y1, spp1
                    elif self._mix > 0.0:
                        y, spp = (1.0 - self._mix) * y + self._mix * y1, (1.0 - self._mix) * spp + self._mix * spp1
            else:
                y, spp = self.runner(spec)
            self.spp = spp.astype(np.float64)
            Y = y[0, 0, :, 0] + 1j * y[0, 0, :, 1]
            if self.cfg.gain_floor_db is not None:
                Y = Y + 10 ** (self.cfg.gain_floor_db / 20) * E
        else:
            Y = E
        out = self.stft.synthesize(Y)
        if self.cfg.limiter_dbfs is not None:
            out = self._limit(out)
        self.timing.append(time.perf_counter() - t0)
        return out

    # --------------------------------------------------------------- offline
    def process_signal(self, prim: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
        """Run the streaming engine over a whole signal; output is delay-compensated."""
        self.reset()
        hop = DEFAULT.hop
        d = DEFAULT.win - DEFAULT.hop + hop * self.lookahead   # output-stream index delay
        n = len(prim)
        nb = int(np.ceil((n + d) / hop))
        P = np.zeros(nb * hop)
        P[:n] = prim
        Rr = None
        if ref is not None:
            Rr = np.zeros(nb * hop)
            Rr[:n] = ref
        out = np.empty(nb * hop)
        for k in range(nb):
            sl = slice(k * hop, (k + 1) * hop)
            out[sl] = self.process_block(P[sl], None if Rr is None else Rr[sl])
        return out[d: d + n].astype(np.float32)

    def timing_stats(self) -> dict:
        t = np.array(list(self.timing)[10:]) * 1000
        if len(t) == 0:
            return {}
        return {"blocks": int(len(t)), "mean_ms": float(t.mean()), "p50_ms": float(np.median(t)),
                "p99_ms": float(np.percentile(t, 99)), "max_ms": float(t.max()),
                "rtf": float(t.mean() / (1000 * DEFAULT.hop / DEFAULT.fs)),
                "deadline_misses": int(np.sum(t > 1000 * DEFAULT.hop / DEFAULT.fs))}
