#!/usr/bin/env python3
"""Electrical loop-back latency measurement (works on Jetson and on a Mac/PC).

Wire the interface's OUTPUT to its INPUT (line level, use an attenuator for mic inputs),
then:   python3 measure_latency.py --device USB --rate 16000 --block 160

The script plays a log-sweep, records it, cross-correlates and reports the round-trip
latency of the audio I/O path.  Run it (a) as a plain loop-back -> I/O latency, and
(b) with the D-ANC engine in the path (realtime.py --bypass) -> I/O + 20 ms
algorithmic + compute.  NOT VERIFIED ON JETSON HARDWARE IN THIS PROJECT.
"""
from __future__ import annotations

import argparse

import numpy as np


def sweep(fs, dur=1.0, f0=100.0, f1=None):
    f1 = f1 or fs * 0.45
    t = np.arange(int(dur * fs)) / fs
    k = np.log(f1 / f0)
    x = np.sin(2 * np.pi * f0 * dur / k * (np.exp(t * k / dur) - 1))
    return (0.5 * x * np.hanning(len(x)) ** 0.1).astype(np.float32)


def measure(device=None, fs=16000, block=160, repeats=5, channel=0):
    import sounddevice as sd

    x = sweep(fs)
    pad = np.zeros(int(0.5 * fs), np.float32)
    sig = np.concatenate([pad, x, pad])
    lat = []
    for _ in range(repeats):
        rec = sd.playrec(np.stack([sig, sig], 1), samplerate=fs, channels=max(2, channel + 1), dtype="float32",
                         device=device, blocksize=block, latency="low")
        sd.wait()
        r = rec[:, channel]
        n = len(r) + len(sig)
        c = np.fft.irfft(np.fft.rfft(r, n) * np.conj(np.fft.rfft(sig, n)), n)
        lag = int(np.argmax(np.abs(c[: len(r)])))
        lat.append(1000.0 * lag / fs)
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--block", type=int, default=160)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--channel", type=int, default=0)
    a = ap.parse_args()
    lat = measure(a.device, a.rate, a.block, a.repeats, a.channel)
    print("round-trip latency per run (ms):", ", ".join(f"{v:.2f}" for v in lat))
    print(f"median {np.median(lat):.2f} ms  (includes DAC+ADC converter group delay and driver buffering)")


if __name__ == "__main__":
    main()
