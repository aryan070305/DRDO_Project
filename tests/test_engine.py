import numpy as np
import torch

from danc.dsp import istft, stft
from danc.inference.engine import EngineConfig, HybridEngine
from danc.inference.export_onnx import export
from danc.inference.runners import ORTStepRunner, TorchStepRunner
from danc.models.dancnet import DANCNet


def _model():
    torch.manual_seed(0)
    return DANCNet().eval()


def test_engine_matches_offline_model(tmp_path):
    m = _model()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000) * 0.05).astype(np.float32)
    X = torch.view_as_real(torch.from_numpy(stft(x)).to(torch.complex64))[None].float()
    with torch.no_grad():
        Y, _ = m.enhance_spec(X)
    y_off = istft((Y[0, ..., 0] + 1j * Y[0, ..., 1]).numpy(), len(x))
    onnx = tmp_path / "m.onnx"
    export(m, str(onnx))
    for r in (TorchStepRunner(m), ORTStepRunner(str(onnx))):
        e = HybridEngine(r, EngineConfig(use_lms=False, limiter_dbfs=None))
        y = e.process_signal(x)
        assert np.max(np.abs(y - y_off)) < 1e-4


def test_engine_bypass_is_identity_with_delay_compensation():
    x = np.random.default_rng(1).standard_normal(8000).astype(np.float32) * 0.1
    e = HybridEngine(None, EngineConfig(use_lms=False, limiter_dbfs=None))
    assert np.max(np.abs(e.process_signal(x) - x)) < 1e-5


def test_limiter_caps_output():
    x = np.ones(4800, np.float32) * 3.0
    e = HybridEngine(None, EngineConfig(use_lms=False, limiter_dbfs=-1.0))
    y = e.process_signal(x)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6


def test_engine_lookahead_model_matches_offline(tmp_path):
    from danc.models.dancnet import DANCNetConfig

    torch.manual_seed(1)
    m = DANCNet(DANCNetConfig(df_order=5, lookahead=2)).eval()
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(16000) * 0.05).astype(np.float32)
    X = torch.view_as_real(torch.from_numpy(stft(x)).to(torch.complex64))[None].float()
    with torch.no_grad():
        Y, _ = m.enhance_spec(X)
    y_off = istft((Y[0, ..., 0] + 1j * Y[0, ..., 1]).numpy(), len(x))
    onnx = tmp_path / "la.onnx"
    export(m, str(onnx))
    r = ORTStepRunner(str(onnx))
    assert r.lookahead == 2
    e = HybridEngine(r, EngineConfig(use_lms=False, limiter_dbfs=None))
    assert abs(e.latency_ms - 40.0) < 1e-9
    y = e.process_signal(x)
    assert np.max(np.abs(y - y_off)) < 1e-4
