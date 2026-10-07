import numpy as np
import torch

from danc.dsp import DEFAULT, StreamingSTFT, TorchSTFT, istft, stft


def test_offline_perfect_reconstruction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000 + 37)
    y = istft(stft(x), len(x))
    assert np.max(np.abs(x - y)) < 1e-10


def test_streaming_matches_offline_with_fixed_delay():
    rng = np.random.default_rng(1)
    hop = DEFAULT.hop
    x = rng.standard_normal(hop * 50)
    X = stft(x)
    st = StreamingSTFT()
    frames, out = [], []
    for k in range(len(x) // hop):
        f = st.analyze(x[k * hop:(k + 1) * hop])[0]
        frames.append(f)
        out.append(st.synthesize(f))
    frames = np.stack(frames)
    # streaming frame k (available once input block k has arrived) == offline frame k
    assert np.allclose(frames, X[: len(frames)], atol=1e-9)
    y = np.concatenate(out)
    # index delay of the output stream = win - hop; + one hop of block buffering = win (20 ms) wall-clock
    d = DEFAULT.win - DEFAULT.hop
    assert np.max(np.abs(y[d:] - x[: len(x) - d])) < 1e-9


def test_torch_matches_numpy():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 8000)).astype(np.float32)
    st = TorchSTFT()
    Xt = st(torch.from_numpy(x))
    Xn = stft(x)
    assert np.allclose(Xt.numpy(), Xn, atol=1e-4)
    y = st.inverse(Xt, 8000).numpy()
    assert np.max(np.abs(y - x)) < 1e-5
