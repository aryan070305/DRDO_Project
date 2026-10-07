import numpy as np
import torch

from danc.models.dancnet import DANCNet, DANCNetConfig, count_params, macs_per_frame


def _model():
    torch.manual_seed(0)
    m = DANCNet(DANCNetConfig())
    # give BatchNorm non-trivial running stats so eval mode is exercised
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m.enhance_spec(torch.randn(2, 50, 161, 2))
    return m.eval()


def test_shapes_and_finiteness():
    m = _model()
    x = torch.randn(2, 37, 161, 2) * 0.1
    y, spp = m.enhance_spec(x)
    assert y.shape == x.shape and spp.shape == (2, 37, 161)
    assert torch.isfinite(y).all() and (spp >= 0).all() and (spp <= 1).all()


def test_streaming_equals_offline():
    m = _model()
    x = torch.randn(1, 120, 161, 2) * 0.05
    x[:, 60] *= 300.0  # an impulse frame
    with torch.no_grad():
        y_off, spp_off = m.enhance_spec(x)
        st = m.initial_states(1)
        ys, ps = [], []
        for t in range(x.shape[1]):
            y, p, st = m(x[:, t:t + 1], st)
            ys.append(y)
            ps.append(p)
    y_str = torch.cat(ys, 1)
    p_str = torch.cat(ps, 1)
    assert torch.allclose(y_off, y_str, atol=1e-4, rtol=1e-4), (y_off - y_str).abs().max()
    assert torch.allclose(spp_off, p_str, atol=1e-5)


def test_causality():
    m = _model()
    x = torch.randn(1, 80, 161, 2)
    x2 = x.clone()
    x2[:, 50:] = torch.randn(1, 30, 161, 2) * 10
    with torch.no_grad():
        y1, _ = m.enhance_spec(x)
        y2, _ = m.enhance_spec(x2)
    assert torch.allclose(y1[:, :50], y2[:, :50], atol=1e-5)


def test_size_budget():
    m = DANCNet()
    n = count_params(m)
    macs = macs_per_frame()["total"]
    assert n < 1_000_000 and macs < 20e6


def _model_la(L=2, N=5):
    torch.manual_seed(0)
    m = DANCNet(DANCNetConfig(df_order=N, lookahead=L))
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m.enhance_spec(torch.randn(2, 50, 161, 2))
    return m.eval()


def test_lookahead_streaming_equals_offline_with_L_frame_delay():
    m = _model_la()
    L = m.cfg.lookahead
    x = torch.randn(1, 90, 161, 2) * 0.05
    x[:, 40] *= 300.0
    with torch.no_grad():
        y_off, p_off = m.enhance_spec(x)                     # aligned with input
        st = m.initial_states(1)
        ys = []
        for t in range(x.shape[1]):
            y, _, st = m(x[:, t:t + 1], st)
            ys.append(y)
    y_str = torch.cat(ys, 1)                                 # step t -> frame t-L
    assert torch.allclose(y_str[:, L:], y_off[:, : x.shape[1] - L], atol=1e-4, rtol=1e-4)


def test_lookahead_bounded_future_dependency():
    m = _model_la()
    L = m.cfg.lookahead
    x = torch.randn(1, 80, 161, 2)
    x2 = x.clone()
    x2[:, 50:] = torch.randn(1, 30, 161, 2) * 10
    with torch.no_grad():
        y1, _ = m.enhance_spec(x)
        y2, _ = m.enhance_spec(x2)
    # frames < 50 - L may not see the change; frame 50 - L must (look-ahead is real)
    assert torch.allclose(y1[:, : 50 - L], y2[:, : 50 - L], atol=1e-5)
    assert not torch.allclose(y1[:, 50 - L], y2[:, 50 - L], atol=1e-5)


def test_ema_normaliser_ablation_streaming_equals_offline():
    torch.manual_seed(5)
    m = DANCNet(DANCNetConfig(norm_type="ema")).eval()
    x = torch.randn(1, 70, 161, 2) * 0.05
    x[:, 35] *= 300
    with torch.no_grad():
        y_off, _ = m.enhance_spec(x)
        st, ys = m.initial_states(1), []
        for t in range(70):
            y, _, st = m(x[:, t:t + 1], st)
            ys.append(y)
    assert torch.allclose(torch.cat(ys, 1), y_off, atol=1e-4, rtol=1e-4)
