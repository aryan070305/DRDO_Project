import numpy as np
import torch

from danc.dsp import istft, stft
from danc.inference.engine import EngineConfig, HybridEngine
from danc.inference.export_onnx import export
from danc.inference.runners import ORTStepRunner, TorchStepRunner
from danc.models.dancnet import DANCNet, DANCNetConfig


def _model(L=0):
    torch.manual_seed(3)
    m = DANCNet(DANCNetConfig(in_ch=2, df_order=5 if L else 3, lookahead=L))
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m.enhance_spec(torch.randn(2, 40, 161, 2), torch.randn(2, 40, 161, 2))
    return m.eval()


def test_two_mic_streaming_equals_offline():
    for L in (0, 2):
        m = _model(L)
        x, r = torch.randn(1, 60, 161, 2) * 0.05, torch.randn(1, 60, 161, 2) * 0.05
        with torch.no_grad():
            y_off, _ = m.enhance_spec(x, r)
            st, ys = m.initial_states(1), []
            for t in range(60):
                y, _, st = m(x[:, t:t + 1], st, r[:, t:t + 1])
                ys.append(y)
        y_str = torch.cat(ys, 1)
        assert torch.allclose(y_str[:, L:], y_off[:, : 60 - L], atol=1e-4, rtol=1e-4)


def test_reference_input_is_used():
    m = _model()
    x = torch.randn(1, 30, 161, 2) * 0.05
    with torch.no_grad():
        a, _ = m.enhance_spec(x, torch.zeros_like(x))
        b, _ = m.enhance_spec(x, torch.randn_like(x))
    assert not torch.allclose(a, b)


def test_two_mic_engine_onnx_matches_offline(tmp_path):
    m = _model(2)
    rng = np.random.default_rng(0)
    p, r = (rng.standard_normal(8000) * 0.05).astype(np.float32), (rng.standard_normal(8000) * 0.05).astype(np.float32)
    P = torch.view_as_real(torch.from_numpy(stft(p)).to(torch.complex64))[None].float()
    R = torch.view_as_real(torch.from_numpy(stft(r)).to(torch.complex64))[None].float()
    with torch.no_grad():
        Y, _ = m.enhance_spec(P, R)
    y_off = istft((Y[0, ..., 0] + 1j * Y[0, ..., 1]).numpy(), len(p))
    f = tmp_path / "two.onnx"
    export(m, str(f))
    for runner in (TorchStepRunner(m), ORTStepRunner(str(f))):
        assert runner.n_mics == 2
        e = HybridEngine(runner, EngineConfig(limiter_dbfs=None))
        y = e.process_signal(p, r)
        assert np.max(np.abs(y - y_off)) < 1e-4


def test_dual_mixer_redraws_silent_noise(monkeypatch):
    from danc.data.dual_mixer import DualDefenceMixer
    m = DualDefenceMixer("train", 1.0, speech_manifest="speech_train_v2.json")
    calls = {"n": 0}
    orig = m._noises

    def silent_first(category, rng, meta):
        calls["n"] += 1
        z = orig(category, rng, meta)
        return [np.zeros_like(x) for x in z] if calls["n"] == 1 else z

    monkeypatch.setattr(m, "_noises", silent_first)
    ex = m.sample(np.random.default_rng(0), category="impulsive", snr_db=0.0)
    assert np.isfinite(ex["primary"]).all() and calls["n"] >= 2


def test_dual_mixer_last_resort_floor(monkeypatch):
    from danc.data.dual_mixer import DualDefenceMixer
    m = DualDefenceMixer("train", 1.0, speech_manifest="speech_train_v2.json")
    orig = m._noises
    monkeypatch.setattr(m, "_noises", lambda c, r, meta: [np.zeros_like(x) for x in orig(c, r, meta)])
    ex = m.sample(np.random.default_rng(1), category="stationary", snr_db=5.0)
    assert np.isfinite(ex["primary"]).all() and np.isfinite(ex["reference"]).all()


def _single(L=2):
    torch.manual_seed(5)
    m = DANCNet(DANCNetConfig(df_order=5 if L else 3, lookahead=L))
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m.enhance_spec(torch.randn(2, 40, 161, 2))
    return m.eval()


def test_reference_health_fallback():
    """Two-mic network + single-mic fallback: bit-exact while the reference is alive; after the reference dies the
    output becomes the single-mic network's; when it comes back the two-mic network takes over again."""
    two, one = _model(2), _single(2)
    cfg = EngineConfig(limiter_dbfs=None)
    rng = np.random.default_rng(4)
    n = 16000 * 3
    p = (rng.standard_normal(n) * 0.05).astype(np.float32)
    r = (rng.standard_normal(n) * 0.05).astype(np.float32)
    plain = HybridEngine(TorchStepRunner(two), cfg).process_signal(p, r)
    e = HybridEngine(TorchStepRunner(two), cfg, fallback_runner=TorchStepRunner(one))
    assert np.array_equal(e.process_signal(p, r), plain)          # reference alive: no change at all
    assert not e.ref_dead and e.ref_switches == 0
    r_dead = r.copy()
    r_dead[16000:] = 0.0                                         # reference unplugged at t = 1 s
    y = e.process_signal(p, r_dead)
    assert e.ref_dead and e.ref_switches == 1
    y1 = HybridEngine(TorchStepRunner(one), EngineConfig(use_lms=False, limiter_dbfs=None)).process_signal(p)
    tail = slice(int(1.8 * 16000), n)                            # after hold (0.5 s) + cross-fade (0.1 s) + margin
    assert np.max(np.abs(y[tail] - y1[tail])) < 1e-5
    assert np.array_equal(y[:15000], plain[:15000])              # before the fault: identical to two-mic output
    r_back = r_dead.copy()
    r_back[int(1.5 * 16000):] = r[int(1.5 * 16000):]            # reference plugged back in at t = 1.5 s
    e.process_signal(p, r_back)
    assert not e.ref_dead and e.ref_switches == 2


def test_reference_health_fallback_rejects_bad_configs():
    import pytest
    with pytest.raises(ValueError):                              # main runner must be two-mic
        HybridEngine(TorchStepRunner(_single(2)), EngineConfig(), fallback_runner=TorchStepRunner(_single(2)))
    with pytest.raises(ValueError):                              # look-ahead must match (output alignment)
        HybridEngine(TorchStepRunner(_model(2)), EngineConfig(), fallback_runner=TorchStepRunner(_single(0)))


class _Stub:
    """Minimal runner: identity output, for testing the reference-health monitor only."""
    def __init__(self, n_mics):
        self.n_mics, self.lookahead = n_mics, 0

    def reset(self):
        pass

    def __call__(self, spec, ref=None):
        return spec, np.zeros(161, np.float32)


def _run(e, prim, ref):
    for k in range(len(prim) // 160):
        e.process_block(prim[k * 160:(k + 1) * 160], ref[k * 160:(k + 1) * 160])


def test_reference_health_quiet_room_does_not_flap():
    """A live reference in a quiet room (both mics near the noise floor, speech pauses) must not be declared dead;
    regression: the first, absolute-threshold monitor switched 6 times in 6 s."""
    rng = np.random.default_rng(0)
    n = 16000 * 6
    prim = rng.standard_normal(n) * 10 ** (-80 / 20)             # quiet room at the primary
    ref = rng.standard_normal(n) * 10 ** (-85 / 20)              # live reference, also near its noise floor
    talk = (np.arange(n) // 16000) % 2 == 1                      # wearer talks every other second
    prim[talk] += rng.standard_normal(talk.sum()) * 10 ** (-30 / 20)
    ref[talk] += rng.standard_normal(talk.sum()) * 10 ** (-48 / 20)   # leakage 18 dB below the primary
    e = HybridEngine(_Stub(2), EngineConfig(limiter_dbfs=None), fallback_runner=_Stub(1))
    _run(e, prim, ref)
    assert e.ref_switches == 0 and not e.ref_dead


def test_reference_health_detects_dead_mic_with_dc_or_hiss():
    rng = np.random.default_rng(1)
    n = 16000 * 3
    prim = rng.standard_normal(n) * 10 ** (-30 / 20)             # wearer talking / noise at the primary
    for dead in (np.full(n, 0.005),                              # broken cable: 5 mV DC offset only
                 rng.standard_normal(n) * 10 ** (-80 / 20)):     # dead capsule: preamp hiss only
        e = HybridEngine(_Stub(2), EngineConfig(limiter_dbfs=None), fallback_runner=_Stub(1))
        _run(e, prim, dead)
        assert e.ref_dead and e.ref_switches == 1


def test_two_mic_cascade_runs_nlms_first():
    """EngineConfig(lms_two_mic=True): the gated NLMS removes reference-coherent noise BEFORE the two-mic network;
    the default (False) keeps the round-2 two-mic behaviour (network sees the raw primary)."""
    rng = np.random.default_rng(2)
    n = 16000 * 4
    noise = rng.standard_normal(n) * 0.05
    prim, ref = noise.copy(), np.concatenate([np.zeros(3), noise[:-3]])   # coherent noise, 3-sample delay
    out = {}
    for casc in (False, True):
        e = HybridEngine(_Stub(2), EngineConfig(limiter_dbfs=None, lms_two_mic=casc))   # stub: output = its input
        out[casc] = e.process_signal(prim, ref)
    tail = slice(2 * 16000, n)
    assert np.mean(out[False][tail] ** 2) > 0.5 * np.mean(prim[tail] ** 2)           # bypass: noise untouched
    assert np.mean(out[True][tail] ** 2) < 0.1 * np.mean(prim[tail] ** 2)            # cascade: >10 dB cancelled
