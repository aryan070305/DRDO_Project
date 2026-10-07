"""Objective metrics (all intrusive, i.e. need the clean reference).

  pesq_wb   ITU-T P.862.2 wideband PESQ (MOS-LQO, ~1.04 .. 4.64), via the `pesq` package
            (ludlows/python-pesq).  P.862 was not validated for impulsive, non-speech-like
            interference or for many DNN artefacts - treat as indicative.
  pesq_nb   ITU-T P.862 narrowband (resampled to 8 kHz) - closer to tactical-radio bandwidth.
  stoi      short-time objective intelligibility, Taal et al. (IEEE TASLP 2011), `pystoi`
  estoi     extended STOI, Jensen & Taal (IEEE/ACM TASLP 2016) - more sensitive to
            fluctuating/modulated maskers (relevant for rotor and impulsive noise)
  sisdr     scale-invariant SDR, Le Roux et al. (ICASSP 2019)
  snr       output SNR = 10 log10( sum s^2 / sum (s - y)^2 ) - the brief's "SNR > 15 dB" is
            interpreted as this quantity on the enhanced signal (it penalises both residual
            noise and speech distortion; it is the plain, non-scale-invariant SDR)
  segsnr    segmental SNR (20 ms frames, clamped to [-10, 35] dB, Quackenbush et al. 1988 style)
"""
from __future__ import annotations

import numpy as np
import soxr
from pesq import pesq as _pesq
from pystoi import stoi as _stoi

FS = 16000


def snr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-12) + 1e-12))


def segsnr(ref, est, frame=320, lo=-10.0, hi=35.0):
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    n = len(ref) // frame
    r = ref[: n * frame].reshape(n, frame)
    e = est[: n * frame].reshape(n, frame)
    num = np.sum(r ** 2, 1)
    den = np.sum((r - e) ** 2, 1) + 1e-12
    act = num > 1e-10 * frame                         # skip digital silence
    if not np.any(act):
        return float("nan")
    v = np.clip(10 * np.log10(num[act] / den[act] + 1e-12), lo, hi)
    return float(np.mean(v))


def sisdr(ref, est):
    r = ref.astype(np.float64) - np.mean(ref)
    e = est.astype(np.float64) - np.mean(est)
    a = np.dot(e, r) / (np.dot(r, r) + 1e-12)
    t = a * r
    return float(10 * np.log10(np.sum(t ** 2) / (np.sum((e - t) ** 2) + 1e-12) + 1e-12))


def pesq_wb(ref, est):
    try:
        return float(_pesq(FS, ref.astype(np.float64), est.astype(np.float64), "wb"))
    except Exception:
        return float("nan")


def pesq_nb(ref, est):
    try:
        r8 = soxr.resample(ref.astype(np.float64), FS, 8000)
        e8 = soxr.resample(est.astype(np.float64), FS, 8000)
        return float(_pesq(8000, r8, e8, "nb"))
    except Exception:
        return float("nan")


def all_metrics(ref: np.ndarray, est: np.ndarray, nb: bool = True) -> dict:
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    out = {
        "pesq_wb": pesq_wb(ref, est),
        "stoi": float(_stoi(ref, est, FS, extended=False)),
        "estoi": float(_stoi(ref, est, FS, extended=True)),
        "sisdr": sisdr(ref, est),
        "snr": snr(ref, est),
        "segsnr": segsnr(ref, est),
    }
    if nb:
        out["pesq_nb"] = pesq_nb(ref, est)
    return out


def _worker(args):
    ref, est = args
    return all_metrics(ref, est)


def batch_metrics(pairs, workers: int = 8) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, pairs, chunksize=4))
