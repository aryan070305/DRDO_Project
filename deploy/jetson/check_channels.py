#!/usr/bin/env python3
"""Check microphone channel mapping and input levels before going live.

    NOT VERIFIED ON JETSON HARDWARE IN THIS PROJECT (the --file analysis is tested on macOS).

The hybrid canceller assumes input ch0 = PRIMARY boom mic (at the mouth) and ch1 = outward-facing
REFERENCE mic.  If the two are swapped, the NLMS uses the talker's own voice as the "noise" reference and
cancels speech.  Run this in a quiet room while speaking normally into the boom mic:

    python3 deploy/jetson/check_channels.py --device USB              # records 6 s from the interface
    python3 deploy/jetson/check_channels.py --device USB --rate 48000 # if the interface refuses 16 kHz
    python3 deploy/jetson/check_channels.py --file two_channel.wav    # analyse an existing 2-ch recording

Nothing is written to disk.  Reports per-channel peak / RMS (dBFS), clipping, and the level difference
primary - reference over the loudest 20 % of 100 ms frames (= while speaking).
PASS: primary >= 6 dB louder than the reference while speaking, no clipping, speech not too quiet.
Exit code 0 = PASS, 1 = FAIL (mapping or levels wrong), 2 = could not record / read.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-9))


def analyse(x: np.ndarray, fs: int, primary_ch: int = 0, reference_ch: int = 1, min_diff_db: float = 6.0) -> dict:
    """x: [n, channels] float in [-1, 1].  Returns the measurements and a verdict."""
    if x.ndim != 2 or x.shape[1] <= max(primary_ch, reference_ch):
        raise ValueError(f"need at least {max(primary_ch, reference_ch) + 1} channels, got shape {x.shape}")
    p, r = x[:, primary_ch].astype(np.float64), x[:, reference_ch].astype(np.float64)
    fl = int(0.1 * fs)
    n = len(p) // fl
    if n < 10:
        raise ValueError("recording too short (need >= 1 s)")
    ep = np.sqrt(np.mean(p[: n * fl].reshape(n, fl) ** 2, 1))
    er = np.sqrt(np.mean(r[: n * fl].reshape(n, fl) ** 2, 1))
    k = max(1, int(round(0.2 * n)))
    act = np.argsort(ep)[-k:]                               # loudest 20 % of primary frames = talking
    quiet = np.argsort(ep)[: max(1, int(round(0.2 * n)))]   # quietest 20 % = background / noise floor
    diff = float(np.mean(db(ep[act])) - np.mean(db(er[act])))
    res = {
        "fs": fs, "seconds": len(p) / fs,
        "primary": {"ch": primary_ch, "peak_dbfs": float(db(np.max(np.abs(p)))), "speech_rms_dbfs": float(np.mean(db(ep[act]))),
                    "floor_rms_dbfs": float(np.mean(db(ep[quiet]))), "clipped_fraction": float(np.mean(np.abs(p) >= 0.999))},
        "reference": {"ch": reference_ch, "peak_dbfs": float(db(np.max(np.abs(r)))), "speech_rms_dbfs": float(np.mean(db(er[act]))),
                      "floor_rms_dbfs": float(np.mean(db(er[quiet]))), "clipped_fraction": float(np.mean(np.abs(r) >= 0.999))},
        "primary_minus_reference_db": diff,
    }
    problems = []
    if diff < 0:
        problems.append(f"reference is LOUDER than primary while speaking ({diff:+.1f} dB): channels probably SWAPPED "
                        f"- swap the cables or run with --primary-ch {reference_ch} --reference-ch {primary_ch}")
    elif diff < min_diff_db:
        problems.append(f"primary only {diff:.1f} dB above the reference while speaking (need >= {min_diff_db:g} dB): "
                        "move the reference mic away from the mouth / outward, or check the mapping")
    for name in ("primary", "reference"):
        c = res[name]
        if c["clipped_fraction"] > 1e-4 or c["peak_dbfs"] > -0.1:
            problems.append(f"{name} clips (peak {c['peak_dbfs']:.1f} dBFS): reduce the pre-amp gain")
    if res["primary"]["speech_rms_dbfs"] < -45:
        problems.append(f"primary speech is very quiet ({res['primary']['speech_rms_dbfs']:.1f} dBFS RMS): raise the gain "
                        "(aim for speech peaks around -12 dBFS)")
    res["problems"] = problems
    res["verdict"] = "PASS" if not problems else "FAIL"
    return res


def record(device, fs: int, seconds: float, channels: int) -> np.ndarray:
    import sounddevice as sd

    dev = int(device) if device is not None and str(device).isdigit() else device
    print(f"Recording {seconds:g} s from {device!r} ({channels} ch @ {fs} Hz): speak normally into the BOOM mic now...",
          flush=True)
    x = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, dtype="float32", device=dev)
    sd.wait()
    return np.asarray(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None, help="input device name substring or index (see danc_live.py --list-devices)")
    ap.add_argument("--file", default=None, help="analyse a 2-channel audio file instead of recording")
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--primary-ch", type=int, default=0)
    ap.add_argument("--reference-ch", type=int, default=1)
    ap.add_argument("--min-diff-db", type=float, default=6.0)
    a = ap.parse_args(argv)
    try:
        if a.file:
            import soundfile as sf
            x, fs = sf.read(a.file, dtype="float32", always_2d=True)
        else:
            x, fs = record(a.device, a.rate, a.seconds, a.channels), a.rate
        res = analyse(x, fs, a.primary_ch, a.reference_ch, a.min_diff_db)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for name in ("primary", "reference"):
        c = res[name]
        print(f"{name:9s} ch{c['ch']}: peak {c['peak_dbfs']:6.1f} dBFS | speech RMS {c['speech_rms_dbfs']:6.1f} dBFS | "
              f"floor RMS {c['floor_rms_dbfs']:6.1f} dBFS | clipped {100 * c['clipped_fraction']:.3f} %")
    print(f"primary - reference while speaking: {res['primary_minus_reference_db']:+.1f} dB (need >= {a.min_diff_db:g} dB)")
    for pr in res["problems"]:
        print("  - " + pr)
    print(res["verdict"])
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
