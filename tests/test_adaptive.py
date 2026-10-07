"""Tests for the reference-mic adaptive cancellers and the dual-mic scene simulator."""
import numpy as np
import pytest
from scipy.signal import csd, lfilter, welch

from danc.adaptive.nlms import SubbandNLMS, TimeNLMS, run_subband_nlms, run_time_nlms
from danc.data.dual_mic import (DualMicConfig, diffuse_coherence, fractional_delay_ir, simulate_scene)
from danc.data.dual_mic import _apply_ir  # noqa: PLC2701  (internal helper, used for leakage paths)
from danc.dsp import DEFAULT, StreamingSTFT, istft, n_frames, stft

FS = 16000
# a short room-like coherent path: 3-sample bulk delay plus two echoes
H_NOISE = np.zeros(24)
H_NOISE[[3, 4, 12]] = [1.0, 0.4, -0.25]


def _db(num, den):
    return 10.0 * np.log10(np.sum(np.square(num)) / np.sum(np.square(den)))


def _coherent_pair(rng, n, lead=0, scale=0.1):
    """Reference = white noise; primary = H_NOISE-filtered noise, optionally *leading* the reference."""
    s = rng.standard_normal(n + 200) * scale
    prim = lfilter(H_NOISE, 1.0, s)[200:] if lead == 0 else s[200:]
    ref = s[200:] if lead == 0 else s[200 - lead:n + 200 - lead]
    return prim, ref


def _speech_like(n, on=0.6, off=0.4, start=1.0):
    """Voiced, syllable-modulated harmonic signal with on/off bursts and its sample-level activity."""
    t = np.arange(n) / FS
    ph = 2 * np.pi * np.cumsum(130 + 20 * np.sin(2 * np.pi * 0.7 * t)) / FS
    x = sum(np.sin(k * ph) / k for k in range(1, 25))
    env = 0.5 - 0.5 * np.cos(2 * np.pi * 4 * t)
    gate = (t >= start) & (((t - start) % (on + off)) < on)
    return x * env * gate, gate


# ----------------------------------------------------------------------------
# (a) coherent noise cancellation
# ----------------------------------------------------------------------------

def test_time_nlms_cancels_coherent_noise():
    rng = np.random.default_rng(0)
    p, r = _coherent_pair(rng, 3 * FS)
    e = run_time_nlms(p, r, n_taps=64, mu=0.1)
    assert np.all(np.isfinite(e))
    assert _db(p[2 * FS:], e[2 * FS:]) > 15.0


def test_time_nlms_streaming_equals_one_shot_and_reset():
    rng = np.random.default_rng(1)
    p, r = _coherent_pair(rng, FS // 2)
    ref = TimeNLMS(n_taps=32, leak=1e-4).process(p, r)
    nl = TimeNLMS(n_taps=32, leak=1e-4)
    cuts = np.sort(rng.choice(np.arange(1, len(p)), size=20, replace=False))
    out = [nl.process(pb, rb) for pb, rb in zip(np.split(p, cuts), np.split(r, cuts))]
    assert np.allclose(np.concatenate(out), ref, atol=1e-12)
    nl.reset()
    assert np.allclose(nl.process(p, r), ref, atol=1e-12)


def test_time_nlms_bulk_delay_handles_acausal_noise():
    """Noise reaching the primary first needs a primary bulk delay in the time domain."""
    rng = np.random.default_rng(2)
    p, r = _coherent_pair(rng, 2 * FS, lead=4)
    e0 = run_time_nlms(p, r)               # causal FIR: cannot model a negative delay
    e8 = run_time_nlms(p, r, delay=8)      # +0.5 ms latency, output re-aligned offline
    sl = slice(FS, 2 * FS - 16)            # (last samples need reference beyond the end)
    assert _db(p[sl], e0[sl]) < 3.0
    assert _db(p[sl], e8[sl]) > 15.0


@pytest.mark.parametrize("lead", [0, 4, 7])
def test_subband_nlms_cancels_coherent_noise(lead):
    """Zero extra latency, including noise that reaches the primary up to 7 samples (15 cm) earlier."""
    rng = np.random.default_rng(3 + lead)
    p, r = _coherent_pair(rng, 3 * FS, lead=lead)
    e = run_subband_nlms(p, r)
    assert np.all(np.isfinite(e))
    assert _db(p[2 * FS:], e[2 * FS:]) > 15.0


def test_subband_nlms_streaming_matches_offline_with_stft_latency_only():
    rng = np.random.default_rng(4)
    hop = DEFAULT.hop
    p, r = _coherent_pair(rng, hop * 120)
    y_off = run_subband_nlms(p, r)
    sp, sr, nl = StreamingSTFT(), StreamingSTFT(), SubbandNLMS()
    out = []
    for k in range(len(p) // hop):
        P = sp.analyze(p[k * hop:(k + 1) * hop])[0]
        R = sr.analyze(r[k * hop:(k + 1) * hop])[0]
        out.append(sp.synthesize(nl.step(P, R)))
    y = np.concatenate(out)
    d = DEFAULT.win - DEFAULT.hop  # the shared STFT's own index delay: the canceller adds nothing
    assert np.max(np.abs(y[d:] - y_off[:len(y) - d])) < 1e-9


def test_run_subband_nlms_spp_shapes():
    rng = np.random.default_rng(5)
    p, r = _coherent_pair(rng, FS // 2)
    T = n_frames(len(p))
    spp_t = rng.uniform(0, 1, T)
    y1 = run_subband_nlms(p, r, spp=spp_t)
    y2 = run_subband_nlms(p, r, spp=np.repeat(spp_t[:, None], DEFAULT.n_bins, axis=1))
    assert np.allclose(y1, y2, atol=1e-12)
    # spp == 1 everywhere -> no adaptation -> W stays 0 -> output == primary
    assert np.allclose(run_subband_nlms(p, r, spp=np.ones(T)), p, atol=1e-9)
    with pytest.raises(ValueError):
        run_subband_nlms(p, r, spp=np.ones(T + 1))


# ----------------------------------------------------------------------------
# (b) speech-presence controlled adaptation (signal cancellation via leakage)
# ----------------------------------------------------------------------------

def _speech_trial(leak_db, freeze, seed=10, dur=6.0):
    """Return (speech-component change in dB, noise reduction in noise-only parts in dB).

    The speech component of the output is computed exactly by passing the
    clean primary speech and the leaked reference speech through the same
    (time-varying) weights the canceller used for that frame.
    """
    rng = np.random.default_rng(seed)
    N = int(dur * FS)
    s, gate = _speech_like(N)
    noise = rng.standard_normal(N + 200) * 0.1
    n_p, n_r = lfilter(H_NOISE, 1.0, noise)[200:], noise[200:]
    s *= np.sqrt(np.sum(n_p ** 2) / np.sum(s ** 2))  # 0 dB SNR at the primary
    leak = _apply_ir(s, 0, N, fractional_delay_ir(np.array([4.2]), np.array([10 ** (leak_db / 20)])))
    P, R, S, Lk = stft(s + n_p), stft(n_r + leak), stft(s), stft(leak)
    T, F = P.shape
    hop = DEFAULT.hop
    active = np.array([gate[max(0, t * hop - hop):t * hop + hop].any() for t in range(T)])  # oracle SPP
    nl = SubbandNLMS(n_bins=F, out_limit_db=None)  # limiter off -> exactly linear per frame
    L_hist = np.zeros((F, nl.n_taps), dtype=complex)
    E, Es = np.empty_like(P), np.empty_like(P)
    for t in range(T):
        L_hist[:, 1:] = L_hist[:, :-1]
        L_hist[:, 0] = Lk[t]
        Es[t] = S[t] - np.sum(nl.W * L_hist, axis=1)      # speech component with the weights in use
        E[t] = nl.step(P[t], R[t], 0.0 if (freeze and active[t]) else 1.0)
    es, e = istft(Es, N), istft(E, N)
    a = 2 * FS
    noise_only = ~gate
    noise_only[:a] = False
    return _db(es[a:], s[a:]), _db(n_p[noise_only], (e - s)[noise_only])


def test_spp_freeze_preserves_speech_and_free_adaptation_cancels_it():
    d_freeze, nr_freeze = _speech_trial(-20.0, freeze=True)
    d_adapt, nr_adapt = _speech_trial(-20.0, freeze=False)
    d_adapt_strong, _ = _speech_trial(-12.0, freeze=False)
    # SPP-controlled: speech passes essentially unchanged and the noise solution is kept
    assert abs(d_freeze) < 1.0
    assert nr_freeze > 20.0
    # always adapting: the filter learns to cancel speech through the leakage path
    assert d_adapt < -2.0
    assert d_adapt_strong < -3.0
    assert d_adapt_strong < d_adapt  # stronger leakage -> more speech cancellation
    assert nr_freeze > nr_adapt + 6.0  # and the noise model is disturbed too


# ----------------------------------------------------------------------------
# (c) impulse robustness / divergence guards
# ----------------------------------------------------------------------------

def _impulse_pair(seed=6, n=4 * FS, at=2 * FS, amp=100.0):
    rng = np.random.default_rng(seed)
    p, r = _coherent_pair(rng, n)
    r_imp = r.copy()
    r_imp[at:at + 32] += amp * np.hanning(32)  # ~1000x the noise std, reference only
    return p, r_imp


def _sec(x, a, b):
    return x[int(a * FS):int(b * FS)]


def test_subband_nlms_impulse_robust():
    p, r = _impulse_pair()
    P, R = stft(p), stft(r)
    nl = SubbandNLMS()
    E = np.empty_like(P)
    W_before = None
    t_imp = 2 * FS // DEFAULT.hop
    for t in range(P.shape[0]):
        if t == t_imp - 2:
            W_before = nl.W.copy()
        E[t] = nl.step(P[t], R[t])
    e = istft(E, len(p))
    assert np.all(np.isfinite(e)) and np.all(np.isfinite(nl.W))
    assert np.max(np.abs(nl.W)) <= nl.max_weight + 1e-9
    assert 1 <= nl.n_impulse_frames <= 4 and nl.n_bin_resets == 0
    # the impulse did not disturb the converged weights ...
    assert np.linalg.norm(nl.W - W_before) / np.linalg.norm(W_before) < 0.1
    # ... cancellation continues right after it ...
    assert _db(_sec(p, 2.05, 2.5), _sec(e, 2.05, 2.5)) > 15.0
    # ... and the output limiter kept the reference-only click out of the output
    assert np.max(np.abs(e)) < 3.0 * np.max(np.abs(p))


def test_subband_nlms_without_impulse_gating_is_disturbed():
    """Reference behaviour: the same impulse with gating disabled costs a long re-convergence."""
    p, r = _impulse_pair()
    e_on = run_subband_nlms(p, r)
    e_off = run_subband_nlms(p, r, impulse_kappa=np.inf)
    assert np.all(np.isfinite(e_off))  # still no divergence (normalised step + guards)
    assert _db(_sec(p, 2.05, 2.5), _sec(e_on, 2.05, 2.5)) > _db(_sec(p, 2.05, 2.5), _sec(e_off, 2.05, 2.5)) + 10


def test_subband_nlms_level_step_is_not_an_impulse():
    """A sustained +30 dB noise onset must be accepted as a level change and adapted to."""
    rng = np.random.default_rng(7)
    p, r = _coherent_pair(rng, 3 * FS)
    p[:FS] *= 0.03
    r[:FS] *= 0.03
    nl = SubbandNLMS()
    e = run_subband_nlms(p, r)
    assert _db(_sec(p, 1.5, 3.0), _sec(e, 1.5, 3.0)) > 15.0
    P, R = stft(p), stft(r)
    for t in range(P.shape[0]):
        nl.step(P[t], R[t])
    assert nl.n_impulse_frames <= nl.impulse_max_frames + 1


def test_subband_nlms_divergence_and_nonfinite_guards():
    rng = np.random.default_rng(8)
    p, r = _coherent_pair(rng, 2 * FS)
    P, R = stft(p), stft(r)
    T = P.shape[0]
    nl = SubbandNLMS()
    for t in range(T // 2):
        nl.step(P[t], R[t])
    nl.W[:] = 8.0 * np.exp(2j * np.pi * rng.uniform(size=nl.W.shape))  # corrupt every bin
    nl.W[10, 0] = np.nan
    E = np.array([nl.step(P[t], R[t]) for t in range(T // 2, T)])
    assert np.all(np.isfinite(E)) and np.all(np.isfinite(nl.W))
    assert nl.n_bin_resets > 0
    tail = slice(len(E) - 30, len(E))
    assert 10 * np.log10(np.sum(np.abs(P[T // 2:][tail]) ** 2) / np.sum(np.abs(E[tail]) ** 2)) > 15.0
    # non-finite reference input is ignored
    out = nl.step(P[0], np.full(DEFAULT.n_bins, np.nan + 0j))
    assert np.all(np.isfinite(out)) and np.all(np.isfinite(nl.W))


def test_time_nlms_impulse_does_not_diverge():
    p, r = _impulse_pair()
    nl = TimeNLMS()
    e = nl.process(p, r)
    assert np.all(np.isfinite(e)) and np.all(np.isfinite(nl.w))
    assert np.max(np.abs(nl.w)) < 10.0
    assert _db(_sec(p, 2.3, 3.0), _sec(e, 2.3, 3.0)) > 15.0


# ----------------------------------------------------------------------------
# (d) dual-mic scene simulator
# ----------------------------------------------------------------------------

def _scene_inputs(seed=0, dur=3.0, n_noises=2):
    rng = np.random.default_rng(seed)
    N = int(dur * FS)
    speech, _ = _speech_like(N, start=0.2)
    speech *= 0.05
    w = rng.standard_normal((n_noises, N + 2 * FS))
    pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                   [1, -2.494956002, 2.017265875, -0.522189400], w, axis=-1)
    return speech, list(0.1 * pink / pink.std(axis=-1, keepdims=True))


@pytest.mark.parametrize("snr_db,seed", [(-5.0, 0), (5.0, 1), (20.0, 2)])
def test_simulate_scene_shapes_and_snr(snr_db, seed):
    speech, noises = _scene_inputs(seed)
    out = simulate_scene(speech, noises, snr_db, np.random.default_rng(seed))
    N = len(speech)
    for k in ("primary", "reference", "clean", "noise_primary", "noise_reference", "speech_reference"):
        assert out[k].shape == (N,) and np.all(np.isfinite(out[k]))
    m = out["meta"]
    assert abs(_db(out["clean"], out["noise_primary"]) - snr_db) < 0.5
    assert np.allclose(out["primary"], out["clean"] + out["noise_primary"])
    assert np.allclose(out["reference"], out["speech_reference"] + out["noise_reference"])
    assert abs(_db(out["speech_reference"], out["clean"]) - m["speech_leak_db"]) < 0.5
    cfg = DualMicConfig()
    assert cfg.mic_spacing_m[0] <= m["mic_spacing_m"] <= cfg.mic_spacing_m[1]
    assert cfg.speech_leak_db[0] <= m["speech_leak_db"] <= cfg.speech_leak_db[1]
    assert 0.0 <= m["diffuse_ratio"] <= 0.5 and m["n_directional"] <= 3
    max_delay = m["mic_spacing_m"] / cfg.c * FS
    for s in m["sources"]:
        assert 1.0 <= s["distance_m"] <= 30.0 and 1 <= len(s["reflections"]) <= 3
        assert abs(s["inter_mic_delay_samples"]) <= max_delay + 1e-9


def test_simulate_scene_peak_limit_keeps_snr():
    speech, noises = _scene_inputs(3)
    out = simulate_scene(30.0 * speech, noises, 0.0, np.random.default_rng(3))
    assert max(np.abs(out["primary"]).max(), np.abs(out["reference"]).max()) <= 0.99 + 1e-12
    assert out["meta"]["output_gain"] < 1.0
    assert abs(_db(out["clean"], out["noise_primary"])) < 0.5


def test_simulate_scene_diffuse_coherence_follows_sinc():
    speech, noises = _scene_inputs(4, dur=4.0)
    d = 0.10
    cfg = DualMicConfig(mic_spacing_m=(d, d), diffuse_ratio=(1.0, 1.0), self_noise_db=(-150.0, -150.0))
    out = simulate_scene(speech, noises, 0.0, np.random.default_rng(4), cfg)
    assert out["meta"]["n_directional"] == 0
    x, y = out["noise_primary"], out["noise_reference"]
    f, pxy = csd(x, y, fs=FS, nperseg=512)
    _, pxx = welch(x, fs=FS, nperseg=512)
    _, pyy = welch(y, fs=FS, nperseg=512)
    coh = np.real(pxy) / np.sqrt(pxx * pyy)
    for fc in (250.0, 500.0, 1000.0, 2500.0, 5000.0):
        band = (f > fc - 100) & (f < fc + 100)
        assert abs(coh[band].mean() - diffuse_coherence(np.array([fc]), d)[0]) < 0.12, fc
    first_zero = (f > 343 / (2 * d) - 100) & (f < 343 / (2 * d) + 100)
    assert abs(coh[first_zero].mean()) < 0.12
    assert abs(_db(y, x)) < 1.0  # equal power at both mics


def test_simulate_scene_directional_noise_is_coherent():
    speech, noises = _scene_inputs(5)
    cfg = DualMicConfig(n_directional=(1, 1), diffuse_ratio=(0.0, 0.0), self_noise_db=(-150.0, -150.0))
    out = simulate_scene(speech, noises, 0.0, np.random.default_rng(5), cfg)
    f, pxy = csd(out["noise_primary"], out["noise_reference"], fs=FS, nperseg=512)
    _, pxx = welch(out["noise_primary"], fs=FS, nperseg=512)
    _, pyy = welch(out["noise_reference"], fs=FS, nperseg=512)
    msc = np.abs(pxy) ** 2 / (pxx * pyy)
    assert msc[(f > 100) & (f < 4000)].mean() > 0.9


def test_simulate_scene_deterministic():
    speech, noises = _scene_inputs(6, n_noises=1)  # one noise reused for directional + diffuse
    a = simulate_scene(speech, noises, 3.0, np.random.default_rng(42))
    b = simulate_scene(speech, noises, 3.0, np.random.default_rng(42))
    c = simulate_scene(speech, noises, 3.0, np.random.default_rng(43))
    for k in ("primary", "reference", "clean", "noise_primary", "noise_reference"):
        assert np.array_equal(a[k], b[k])
    assert a["meta"] == b["meta"]
    assert not np.array_equal(a["reference"], c["reference"])


def test_simulate_scene_input_validation():
    speech, noises = _scene_inputs(7)
    with pytest.raises(ValueError):
        simulate_scene(speech, [noises[0][: len(speech) - 1]], 0.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        simulate_scene(np.zeros_like(speech), noises, 0.0, np.random.default_rng(0))
