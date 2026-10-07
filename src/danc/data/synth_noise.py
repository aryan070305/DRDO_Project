"""Physically-motivated synthetic DEFENCE noise generators.

Why synthetic?  No openly licensed, curated corpus of gunfire / artillery / rotor /
drone noise recorded at a headset microphone was found (see docs/research/T2_datasets.md).
Recorded material we do have (NOISEX-92 machine gun, ESC-50 helicopter/fireworks/siren,
the drone dataset) is limited, so these generators add *parametric variety* - every call
draws new physical parameters.  They are an augmentation, NOT a substitute for field
recordings: the report lists "validate on recorded range/vehicle noise" as a required step.

Physical models (sources; parameter ranges are engineering choices unless stated):
  * Muzzle blast - Friedlander waveform p(t) = Ps (1 - t/T+) exp(-b t/T+), the classic
    free-field blast-wave model (Friedlander 1946; used for gunshots e.g. in R.C. Maher,
    "Acoustical characterization of gunshots", IEEE SAFE Workshop 2007).  Positive-phase
    duration of small arms is of the order of a millisecond (Maher 2007).
  * Supersonic projectile -> ballistic shock "N-wave" of a few hundred microseconds that
    reaches a down-range observer before the muzzle blast (Maher 2007).
  * Ground reflection: delayed, attenuated copy; air absorption: distance-dependent
    low-pass; environment: exponentially decaying reverberant / echo tail.
  * Artillery / explosions: same blast model with a much longer positive phase and strong
    low-frequency rumble; optional incoming-round whistle (descending tone).
  * Helicopter: main-rotor blade-passing frequency (BPF) 10-30 Hz with harmonics and
    impulsive blade-vortex-interaction (BVI) "slap" pulses once per blade passage, tail
    rotor BPF 50-120 Hz, turbine whine tones, rotor-modulated broadband noise.
  * Multirotor drone: 4-8 rotors with BPF 80-400 Hz, RPM jitter/drift, many harmonics,
    motor whine, broadband turbulence.
  * Sirens: wail (slow sweep), yelp (fast sweep), hi-lo (two-tone); square-ish harmonic
    spectrum; optional Doppler for a passing vehicle.
  * Wind on a microphone: low-frequency turbulence (< ~500 Hz) with a gust envelope, in the
    spirit of the excitation/gain model of Nelke & Vary (IWAENC 2014) - simplified.
  * Tracked vehicle: diesel firing-frequency harmonics, track-slap impulses, gearbox whine.
  * Jet fly-by: broadband jet-mixing noise + fan tones with Doppler and a level envelope.

API:  KINDS, generate(kind, duration_s, fs=16000, rng=None, **params) -> float32 [n]
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

C_SOUND = 343.0

KINDS: dict[str, dict] = {
    "gunshot_single": {"category": "impulsive", "description": "single small-arms shot, 5-300 m"},
    "gunfire_burst": {"category": "impulsive", "description": "automatic fire, 600-1000 rpm bursts"},
    "gunfire_sporadic": {"category": "impulsive", "description": "several shooters at different ranges"},
    "artillery": {"category": "impulsive", "description": "artillery / distant explosions with rumble"},
    "explosion": {"category": "impulsive", "description": "close blast with debris crackle"},
    "helicopter": {"category": "nonstationary", "description": "main + tail rotor, BVI slap, turbine"},
    "drone_multirotor": {"category": "nonstationary", "description": "4-8 rotor UAV with RPM jitter"},
    "siren_wail": {"category": "nonstationary", "description": "slow-sweep siren"},
    "siren_yelp": {"category": "nonstationary", "description": "fast-sweep siren"},
    "siren_hilo": {"category": "nonstationary", "description": "two-tone siren"},
    "wind": {"category": "nonstationary", "description": "gusty wind turbulence on a mic"},
    "jet_flyby": {"category": "nonstationary", "description": "jet aircraft pass-by"},
    "tracked_vehicle": {"category": "stationary", "description": "tank/APC engine + track slap"},
}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _lp(x, fc, fs, order=2):
    fc = float(np.clip(fc, 20, fs / 2 * 0.95))
    return sosfilt(butter(order, fc, "low", fs=fs, output="sos"), x)


def _hp(x, fc, fs, order=2):
    fc = float(np.clip(fc, 10, fs / 2 * 0.95))
    return sosfilt(butter(order, fc, "high", fs=fs, output="sos"), x)


def _bp(x, lo, hi, fs, order=2):
    lo, hi = float(max(lo, 10)), float(min(hi, fs / 2 * 0.95))
    return sosfilt(butter(order, [lo, hi], "band", fs=fs, output="sos"), x)


def _shaped_noise(n, fs, rng, slope_db_oct=-3.0, f_ref=500.0):
    """Gaussian noise with a power-law spectrum (slope in dB/octave)."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    f[0] = f[1]
    X *= (f / f_ref) ** (slope_db_oct / (20 * np.log10(2)))
    return np.fft.irfft(X, n)


def _smooth_random(n, fs, rng, tau_s):
    """Slowly varying zero-mean unit-ish random process (low-passed noise)."""
    fc = 1.0 / (2 * np.pi * tau_s)
    m = max(8, int(n * fc * 8 / fs) + 8)                  # coarse grid then interpolate
    coarse = rng.standard_normal(m)
    coarse = np.convolve(coarse, np.hanning(5) / np.hanning(5).sum(), "same")
    y = np.interp(np.linspace(0, m - 1, n), np.arange(m), coarse)
    return y / (np.std(y) + 1e-9)


def _phase(freq, fs):
    return 2 * np.pi * np.cumsum(freq) / fs


def _tail(n, fs, rng, rt60, fc):
    """Exponentially decaying low-passed noise (reverberant / echo tail)."""
    t = np.arange(n) / fs
    return _lp(rng.standard_normal(n), fc, fs) * np.exp(-6.91 * t / rt60)


def _norm(y, rng=None):
    y = np.nan_to_num(np.asarray(y, dtype=np.float64))
    pk = np.max(np.abs(y))
    if pk < 1e-12:
        return np.zeros(len(y), np.float32)
    return (0.99 * y / pk).astype(np.float32)


def _friedlander(fs, T_pos, b, dur):
    t = np.arange(int(dur * fs)) / fs
    return (1 - t / T_pos) * np.exp(-b * t / T_pos)


def _air_absorption_fc(dist_m):
    # engineering approximation: high frequencies lost with distance (not an ISO 9613 model)
    return 7500.0 / (1.0 + dist_m / 80.0)


# -----------------------------------------------------------------------------
# impulsive events
# -----------------------------------------------------------------------------

def _gunshot_event(fs, rng, dist=None, weapon=None):
    """One shot as heard at distance `dist` (m). Returns a waveform (peak ~1/dist)."""
    dist = rng.uniform(5, 300) if dist is None else dist
    weapon = weapon or {}
    T_pos = weapon.get("T_pos", rng.uniform(0.5e-3, 3e-3)) * (1 + dist / 400)
    b = weapon.get("b", rng.uniform(1.0, 3.0))
    blast = _friedlander(fs, T_pos, b, 0.06)
    L = int(fs * rng.uniform(0.4, 1.6))
    ev = np.zeros(L)
    t0 = int(fs * 0.06)                                    # leave room for an earlier N-wave
    ev[t0: t0 + len(blast)] += blast
    if weapon.get("supersonic", rng.random() < 0.5):       # ballistic shock N-wave, arrives first
        dn = max(2, int(fs * rng.uniform(0.2e-3, 0.5e-3)))
        nwave = np.linspace(1, -1, dn) * rng.uniform(0.3, 1.0)
        lead = int(fs * rng.uniform(0.003, 0.05))
        ev[t0 - lead: t0 - lead + dn] += nwave
    gd = int(fs * rng.uniform(0.0005, 0.01))               # ground reflection
    ev[gd:] += rng.uniform(0.3, 0.9) * ev[:-gd].copy() if gd > 0 else 0
    ev = _lp(ev, _air_absorption_fc(dist), fs, 2)
    env = rng.choice(["open", "urban", "forest", "indoor"])
    rt = {"open": (0.15, 0.5), "urban": (0.4, 1.5), "forest": (0.3, 1.0), "indoor": (0.3, 1.2)}[env]
    tail = _tail(L - t0, fs, rng, rng.uniform(*rt), rng.uniform(800, 4000))
    ev[t0:] += tail * rng.uniform(0.05, 0.3) * np.max(np.abs(ev))
    if env == "urban":                                     # discrete building echoes
        for _ in range(rng.integers(1, 4)):
            d = int(fs * rng.uniform(0.05, 0.4))
            if d < L - t0:
                ev[t0 + d:] += rng.uniform(0.1, 0.4) * ev[t0: L - d]
    return ev / (np.max(np.abs(ev)) + 1e-12) * (30.0 / (30.0 + dist))


def _artillery_event(fs, rng, dist=None):
    dist = rng.uniform(200, 5000) if dist is None else dist
    T_pos = rng.uniform(5e-3, 50e-3)
    blast = _friedlander(fs, T_pos, rng.uniform(0.8, 2.0), 0.4)
    L = int(fs * rng.uniform(2.0, 5.0))
    ev = np.zeros(L)
    ev[: len(blast)] += blast
    ev = _lp(ev, max(150.0, 3000.0 / (1 + dist / 500)), fs, 2)
    rumble = _lp(rng.standard_normal(L), rng.uniform(60, 250), fs, 4)
    rumble *= np.exp(-6.91 * np.arange(L) / fs / rng.uniform(1.0, 4.0))
    ev += rumble / (np.max(np.abs(rumble)) + 1e-12) * rng.uniform(0.2, 0.7)
    return ev / (np.max(np.abs(ev)) + 1e-12) * (500.0 / (500.0 + dist))


def _whistle(fs, rng, dur):
    n = int(dur * fs)
    f = np.linspace(rng.uniform(1500, 3000), rng.uniform(300, 800), n)
    amp = np.linspace(0.05, 1.0, n) ** 2
    return amp * np.sin(_phase(f, fs))


def _place(n, fs, rng, events, times):
    y = np.zeros(n)
    for ev, t in zip(events, times):
        i = int(t * fs)
        if i >= n:
            continue
        m = min(len(ev), n - i)
        y[i: i + m] += ev[:m]
    return y


def gunshot_single(n, fs, rng, **p):
    t = rng.uniform(0, max(0.01, n / fs - 0.3))
    y = _place(n, fs, rng, [_gunshot_event(fs, rng)], [t])
    return y


def gunfire_burst(n, fs, rng, **p):
    rpm = p.get("rpm", rng.uniform(600, 1000))
    dist = rng.uniform(5, 300)
    weapon = {"T_pos": rng.uniform(0.5e-3, 3e-3), "b": rng.uniform(1, 3), "supersonic": rng.random() < 0.5}
    dur = n / fs
    evs, times = [], []
    n_bursts = rng.integers(1, max(2, int(dur / 1.5)) + 1)
    for _ in range(n_bursts):
        t = rng.uniform(0, max(0.01, dur - 0.2))
        for _ in range(rng.integers(3, 16)):
            evs.append(_gunshot_event(fs, rng, dist * rng.uniform(0.95, 1.05), weapon))
            times.append(t)
            t += 60.0 / rpm * rng.uniform(0.95, 1.05)
    return _place(n, fs, rng, evs, times)


def gunfire_sporadic(n, fs, rng, **p):
    dur = n / fs
    y = np.zeros(n)
    for _ in range(rng.integers(2, 6)):
        dist = rng.uniform(10, 800)
        weapon = {"T_pos": rng.uniform(0.5e-3, 3e-3), "b": rng.uniform(1, 3), "supersonic": rng.random() < 0.5}
        rate = rng.uniform(0.3, 2.0)
        t = rng.exponential(1 / rate)
        evs, times = [], []
        while t < dur:
            k = 1 if rng.random() < 0.7 else rng.integers(2, 5)
            for j in range(k):
                evs.append(_gunshot_event(fs, rng, dist, weapon))
                times.append(t + j * rng.uniform(0.06, 0.12))
            t += rng.exponential(1 / rate)
        y += _place(n, fs, rng, evs, times)
    return y


def artillery(n, fs, rng, **p):
    dur = n / fs
    evs, times = [], []
    for _ in range(rng.integers(1, max(2, int(dur / 2.5)) + 1)):
        t = rng.uniform(0, max(0.01, dur - 0.5))
        ev = _artillery_event(fs, rng)
        if rng.random() < 0.3:                            # incoming round
            w = _whistle(fs, rng, rng.uniform(0.8, 2.5)) * 0.3
            ev = np.concatenate([w, ev])
            t = max(0.0, t - len(w) / fs)
        evs.append(ev)
        times.append(t)
    return _place(n, fs, rng, evs, times)


def explosion(n, fs, rng, **p):
    dur = n / fs
    t0 = rng.uniform(0, max(0.01, dur * 0.6))
    blast = _friedlander(fs, rng.uniform(2e-3, 10e-3), rng.uniform(1, 2.5), 0.2)
    blast = _lp(np.pad(blast, (0, int(fs * 0.3))), rng.uniform(2000, 6000), fs)
    L = int(fs * rng.uniform(1.0, 3.0))
    crackle = np.zeros(L)
    t = np.arange(L) / fs
    dens = rng.uniform(50, 400) * np.exp(-t / rng.uniform(0.3, 1.2))     # clicks per second
    clicks = rng.random(L) < dens / fs
    crackle[clicks] = rng.standard_normal(clicks.sum())
    crackle = _hp(fftconvolve(crackle, np.hanning(9))[:L], 500, fs) * 0.4
    rumble = _lp(rng.standard_normal(L), 120, fs, 4) * np.exp(-t / rng.uniform(0.8, 2.5))
    ev = np.zeros(max(L, len(blast)))
    ev[: len(blast)] += blast
    ev[:L] += crackle + rumble / (np.abs(rumble).max() + 1e-9) * 0.5
    return _place(n, fs, rng, [ev], [t0])


# -----------------------------------------------------------------------------
# rotors, vehicles, sirens, wind
# -----------------------------------------------------------------------------

def _harmonic_stack(phase, fs, f_inst_max, n_harm, rng, decay=1.0):
    y = np.zeros(len(phase))
    for k in range(1, n_harm + 1):
        if k * f_inst_max >= fs / 2 * 0.95:
            break
        y += (k ** -decay) * rng.uniform(0.5, 1.0) * np.cos(k * phase + rng.uniform(0, 2 * np.pi))
    return y


def _doppler_track(n, fs, rng, speed=None):
    """Return (frequency factor, amplitude factor) for a straight pass-by."""
    v = speed if speed is not None else rng.uniform(10, 60)
    d0 = rng.uniform(20, 200)                 # closest-approach distance
    t = np.arange(n) / fs
    tc = rng.uniform(0.2, 0.8) * n / fs
    x = v * (t - tc)
    r = np.sqrt(x ** 2 + d0 ** 2)
    vr = v * x / r                            # radial velocity (+ = receding)
    return 1.0 / (1.0 + vr / C_SOUND), d0 / r


def helicopter(n, fs, rng, **p):
    t = np.arange(n) / fs
    drift = 1 + 0.02 * _smooth_random(n, fs, rng, 2.0)
    dop, amp = _doppler_track(n, fs, rng, rng.uniform(5, 70)) if rng.random() < 0.4 else (np.ones(n), np.ones(n))
    f_m = rng.uniform(10, 30) * drift * dop
    ph_m = _phase(f_m, fs)
    main = _harmonic_stack(ph_m, fs, f_m.max(), 25, rng, decay=0.9)
    # BVI slap: one sharp pulse per blade passage
    cyc = np.floor(ph_m / (2 * np.pi))
    idx = np.where(np.diff(cyc) > 0)[0]
    slap = np.zeros(n)
    slap[idx] = 1.0
    w = int(fs * rng.uniform(0.002, 0.006))
    kern = -np.diff(np.exp(-0.5 * ((np.arange(w) - w / 2) / (w / 6)) ** 2))
    slap = fftconvolve(slap, kern / np.abs(kern).max())[:n] * rng.uniform(0.0, 1.5)
    f_t = rng.uniform(50, 120) * drift * dop
    tail = _harmonic_stack(_phase(f_t, fs), fs, f_t.max(), 12, rng, decay=1.2) * rng.uniform(0.2, 0.6)
    turb = np.zeros(n)
    for _ in range(rng.integers(1, 4)):
        ft = rng.uniform(1000, 6000) * (1 + 0.003 * _smooth_random(n, fs, rng, 1.0)) * dop
        turb += np.cos(_phase(ft, fs)) * rng.uniform(0.02, 0.1)
    bb = _lp(_shaped_noise(n, fs, rng, -4.0), 3000, fs) * (1 + 0.5 * np.cos(ph_m))
    bb *= rng.uniform(0.3, 0.8) / (np.std(bb) + 1e-9) * np.std(main)
    y = main + slap * np.std(main) * 3 + tail * np.std(main) + turb * np.std(main) + bb
    return y * amp * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t))


def drone_multirotor(n, fs, rng, **p):
    n_rot = int(rng.integers(4, 9))
    base = rng.uniform(80, 400)
    y = np.zeros(n)
    manoeuvre = 0.08 * _smooth_random(n, fs, rng, rng.uniform(0.5, 3.0))
    for _ in range(n_rot):
        jitter = 0.02 * _smooth_random(n, fs, rng, rng.uniform(0.05, 0.5))
        f = base * rng.uniform(0.9, 1.1) * (1 + manoeuvre + jitter)
        ph = _phase(f, fs)
        y += _harmonic_stack(ph, fs, f.max(), int(rng.integers(10, 31)), rng, decay=rng.uniform(0.8, 1.5))
        if rng.random() < 0.6:   # motor electrical whine (pole-pair multiple)
            y += 0.1 * np.cos(ph * rng.integers(6, 15))
    turb = _bp(rng.standard_normal(n), 200, 4000, fs)
    turb *= (1 + 0.3 * _smooth_random(n, fs, rng, 0.2))
    y += turb / (np.std(turb) + 1e-9) * np.std(y) * rng.uniform(0.2, 0.6)
    if rng.random() < 0.4:
        dop, amp = _doppler_track(n, fs, rng, rng.uniform(5, 20))
        y = y * amp
    return y


def _siren(n, fs, rng, f_inst):
    ph = _phase(f_inst, fs)
    y = np.zeros(n)
    for k in range(1, 16, 2):                  # odd harmonics -> square-ish timbre
        if k * f_inst.max() >= fs / 2 * 0.95:
            break
        y += np.sin(k * ph) / k
    if rng.random() < 0.5:
        dop, amp = _doppler_track(n, fs, rng, rng.uniform(8, 30))
        ph = _phase(f_inst * dop, fs)
        y = sum(np.sin(k * ph) / k for k in range(1, 16, 2) if k * f_inst.max() < fs / 2 * 0.9) * amp
    tail = _tail(int(0.5 * fs), fs, rng, rng.uniform(0.2, 0.8), 3000)
    return fftconvolve(y, np.concatenate([[1.0], tail[1:] * 0.05]))[:n]


def siren_wail(n, fs, rng, **p):
    t = np.arange(n) / fs
    lo, hi = rng.uniform(500, 800), rng.uniform(1200, 1700)
    sweep = 0.5 - 0.5 * np.cos(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.uniform(0, 2 * np.pi))
    return _siren(n, fs, rng, lo + (hi - lo) * sweep)


def siren_yelp(n, fs, rng, **p):
    t = np.arange(n) / fs
    lo, hi = rng.uniform(500, 800), rng.uniform(1200, 1700)
    r = rng.uniform(2, 5)
    saw = (t * r + rng.uniform(0, 1)) % 1.0
    return _siren(n, fs, rng, lo + (hi - lo) * saw)


def siren_hilo(n, fs, rng, **p):
    t = np.arange(n) / fs
    f1, f2 = rng.uniform(550, 800), rng.uniform(900, 1300)
    rate = rng.uniform(0.5, 2.0)
    sq = ((t * rate + rng.uniform(0, 1)) % 1.0) < 0.5
    f = np.where(sq, f1, f2).astype(float)
    f = np.convolve(f, np.ones(32) / 32, "same")            # short glide between tones
    return _siren(n, fs, rng, f)


def wind(n, fs, rng, **p):
    base = _lp(_shaped_noise(n, fs, rng, -6.0, 100.0), rng.uniform(150, 500), fs, 2)
    base = _hp(base, 15, fs)
    gust = np.exp(rng.uniform(0.5, 1.2) * _smooth_random(n, fs, rng, rng.uniform(0.2, 2.0)))
    slow = np.exp(0.5 * _smooth_random(n, fs, rng, rng.uniform(2.0, 5.0)))
    y = base / (np.std(base) + 1e-9) * gust * slow
    for _ in range(rng.integers(0, 4)):                       # pops / mic-rumble bursts
        i = rng.integers(0, max(1, n - fs // 4))
        L = int(fs * rng.uniform(0.02, 0.2))
        burst = _lp(rng.standard_normal(L), 300, fs) * np.hanning(L)
        y[i: i + L] += burst / (np.std(burst) + 1e-9) * rng.uniform(2, 6)
    return y


def tracked_vehicle(n, fs, rng, **p):
    rpm = 1 + 0.05 * _smooth_random(n, fs, rng, rng.uniform(1, 4))
    f_e = rng.uniform(20, 80) * rpm
    eng = _harmonic_stack(_phase(f_e, fs), fs, f_e.max(), 40, rng, decay=0.7)
    v = rng.uniform(3, 15)
    f_tr = v / rng.uniform(0.15, 0.2)                          # track-link passing frequency
    cyc = np.floor(_phase(np.full(n, f_tr) * (1 + 0.02 * _smooth_random(n, fs, rng, 1.0)), fs) / (2 * np.pi))
    clank = np.zeros(n)
    idx = np.where(np.diff(cyc) > 0)[0]
    clank[idx] = rng.uniform(0.5, 1.0, len(idx))
    L = int(fs * 0.01)
    kern = _bp(rng.standard_normal(L), 500, 3000, fs) * np.exp(-np.arange(L) / (L / 4))
    clank = fftconvolve(clank, kern)[:n]
    whine = np.cos(_phase(rng.uniform(300, 1500) * rpm, fs)) * rng.uniform(0.05, 0.2)
    bb = _lp(_shaped_noise(n, fs, rng, -5.0), 2500, fs)
    s = np.std(eng)
    return eng + clank / (np.std(clank) + 1e-9) * s * rng.uniform(0.3, 1.0) + whine * s * 3 + bb / (np.std(bb) + 1e-9) * s * 0.5


def jet_flyby(n, fs, rng, **p):
    dop, amp = _doppler_track(n, fs, rng, rng.uniform(80, 250))
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    fp = rng.uniform(200, 1500)                                # jet-mixing noise hump
    X *= 1.0 / (1 + (f / fp) ** 2) ** 0.75 * (f / fp + 0.05) ** 0.5
    bb = np.fft.irfft(X, n)
    tones = np.zeros(n)
    for _ in range(rng.integers(1, 3)):
        tones += np.cos(_phase(rng.uniform(2000, 5000) * dop, fs)) * rng.uniform(0.05, 0.2)
    y = bb / (np.std(bb) + 1e-9) + tones / 0.1 * 0.3
    return y * amp ** 2


_GEN = {
    "gunshot_single": gunshot_single, "gunfire_burst": gunfire_burst, "gunfire_sporadic": gunfire_sporadic,
    "artillery": artillery, "explosion": explosion, "helicopter": helicopter, "drone_multirotor": drone_multirotor,
    "siren_wail": siren_wail, "siren_yelp": siren_yelp, "siren_hilo": siren_hilo, "wind": wind,
    "jet_flyby": jet_flyby, "tracked_vehicle": tracked_vehicle,
}


def generate(kind: str, duration_s: float, fs: int = 16000, rng: np.random.Generator | None = None,
             **params) -> np.ndarray:
    if kind not in _GEN:
        raise KeyError(f"unknown kind {kind!r}; choose from {sorted(_GEN)}")
    rng = rng if rng is not None else np.random.default_rng()
    n = int(round(duration_s * fs))
    y = _GEN[kind](n, fs, rng, **params)
    y = np.asarray(y, dtype=np.float64)[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    y = y + rng.standard_normal(n) * 1e-5 * (np.max(np.abs(y)) + 1e-9)   # avoid digital silence
    return _norm(y)
