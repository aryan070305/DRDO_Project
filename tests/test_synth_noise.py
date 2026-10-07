import numpy as np
import pytest

from danc.data.synth_noise import KINDS, generate

FS = 16000


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_basic_properties(kind):
    y = generate(kind, 3.0, FS, np.random.default_rng(0))
    assert y.dtype == np.float32 and len(y) == 3 * FS
    assert np.isfinite(y).all() and np.abs(y).max() <= 0.99 + 1e-6
    assert np.array_equal(y, generate(kind, 3.0, FS, np.random.default_rng(0)))
    assert not np.allclose(y, generate(kind, 3.0, FS, np.random.default_rng(1)))
    assert KINDS[kind]["category"] in ("impulsive", "nonstationary", "stationary")


def _crest_db(y):
    return 20 * np.log10(np.abs(y).max() / np.sqrt(np.mean(y ** 2)))


@pytest.mark.parametrize("kind", [k for k, v in KINDS.items() if v["category"] == "impulsive"])
def test_impulsive_high_crest(kind):
    assert _crest_db(generate(kind, 6.0, FS, np.random.default_rng(3))) > 15


@pytest.mark.parametrize("kind", ["siren_wail", "siren_yelp", "siren_hilo"])
def test_siren_band(kind):
    y = generate(kind, 4.0, FS, np.random.default_rng(4))
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / FS)
    assert P[(f > 450) & (f < 3500)].sum() / P.sum() > 0.8


def _tonal_prominence_db(y):
    """Median over 0.25 s frames of the strongest spectral peak above the local (median-filtered)
    spectral envelope, 40-1500 Hz.  Robust to RPM drift and to steep broadband slopes."""
    from scipy.ndimage import median_filter
    from scipy.signal import spectrogram

    fr, _, S = spectrogram(y, FS, nperseg=4000, noverlap=2000)
    band = (fr > 40) & (fr < 1500)
    D = 10 * np.log10(S[band] + 1e-12)
    return float(np.median((D - median_filter(D, size=(31, 1), mode="nearest")).max(0)))


@pytest.mark.parametrize("kind", ["helicopter", "drone_multirotor", "tracked_vehicle"])
@pytest.mark.parametrize("seed", [0, 1, 5])
def test_rotor_harmonic_peaks(kind, seed):
    assert _tonal_prominence_db(generate(kind, 4.0, FS, np.random.default_rng(seed))) > 12.0


def test_wind_is_not_tonal():
    assert _tonal_prominence_db(generate("wind", 4.0, FS, np.random.default_rng(0))) < 12.0
