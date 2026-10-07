"""Edge-case evaluation set for defence voice: data/testsets/defence_edge_v1 (single primary mic).

Test-split speakers (LibriSpeech test-clean) and test-split noise only; fixed seeds.  Cases (30 files each,
6 s, unless noted):
  clip_heavy    impulsive / mixed noise at 0-5 dB, then ADC hard clipping at 2-6x full scale (target unclipped)
  level_low     talker at -45 dBFS active level (very quiet mic gain), noise at 0/5/10 dB
  level_high    mixture peak-normalised to 0.98 FS (hot mic gain, no clipping)
  reverb_cabin  talker 1-6 m from the mic in a reverberant room/cabin (far RIR); target = direct + 50 ms
  radio_band    mixture and target band-limited to 300-3400 Hz (narrow-band radio receive chain)
  noise_only    no talker at all (metric: noise attenuation in dB; any output is residual or artefact)
  post_blast    explosion 25-35 dB above the talker at t = 1 s; speech starts 50-300 ms after the blast
  noise_switch  three 2-s noise segments of different categories joined abruptly (vehicle / gunfire / siren...)
  hum400        400 Hz aircraft power hum (+harmonics) and 50 Hz mains hum over a weak engine bed
  crew_babble   2-3 competing talkers (other test speakers) at TIR 0/5/10 dB + vehicle bed (target = main talker)
  long_stream   4 files of 60 s: several utterances, noise type changes every 10 s, SNR 5 dB

    python -m danc.data.build_edgeset
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

from danc.data import synth_noise
from danc.data.mixer import DefenceMixer, active_power, power, scale_to_snr
from danc.data.rir import early_part

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "testsets" / "defence_edge_v1"
FS = 16000
N = 6 * FS


def _w(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(x, -1, 1), FS, subtype="PCM_16", format="FLAC")


def _norm_pair(noisy, clean, peak=0.99):
    pk = max(np.abs(noisy).max(), np.abs(clean).max())
    if pk > peak:
        noisy, clean = noisy * peak / pk, clean * peak / pk
    return noisy, clean


def main(seed: int = 9090):
    if (OUT / "meta.csv").exists():
        print("[edge] exists - skip")
        return
    m = DefenceMixer("test", 6.0, use_synth=True, augment=False)
    rng = np.random.default_rng(seed)
    utts = [u for u in m.utts if u["dur"] >= 3.0]
    rows = []
    k = 0

    def speech(level_db=-26.0, n=N):
        s, u = m._speech(rng, n=n, utt=utts[rng.integers(len(utts))])
        return s * np.sqrt(10 ** (level_db / 10) / active_power(s)), u

    def add(case, noisy, clean, u, snr, extra=""):
        nonlocal k
        uid = f"{k:04d}_{case}"
        _w(OUT / "noisy" / f"{uid}.flac", noisy)
        _w(OUT / "clean" / f"{uid}.flac", clean)
        rows.append({"id": uid, "case": case, "snr_db": snr, "utt": u["path"] if u else "", "info": extra})
        k += 1

    for i in range(30):   # clip_heavy
        s, u = speech()
        snr = [0, 5][i % 2]
        nz = scale_to_snr(s, m._noise(["impulsive", "mixed"][i % 2], N, rng, {}), snr)
        x = s + nz
        drive = rng.uniform(2, 6)
        g = drive / (np.abs(x).max() + 1e-9)
        x = np.clip(g * x, -1, 1) / g
        x, c = _norm_pair(x, s)
        add("clip_heavy", x, c, u, snr, f"drive={drive:.2f}")
    for i in range(30):   # level_low
        s, u = speech(-45.0)
        snr = [0, 5, 10][i % 3]
        cat = ["stationary", "nonstationary", "impulsive", "mixed"][i % 4]
        nz = scale_to_snr(s, m._noise(cat, N, rng, {}), snr)
        add("level_low", s + nz, s, u, snr, cat)
    for i in range(30):   # level_high
        s, u = speech()
        snr = [5, 10][i % 2]
        cat = ["stationary", "nonstationary", "impulsive", "mixed"][i % 4]
        x = s + scale_to_snr(s, m._noise(cat, N, rng, {}), snr)
        g = 0.98 / (np.abs(x).max() + 1e-9)
        add("level_high", x * g, s * g, u, snr, cat)
    for i in range(30):   # reverb_cabin
        s, u = speech()
        h = m.rirs["far"][rng.integers(len(m.rirs["far"]))]
        h = h / (np.abs(h).max() + 1e-9)
        rev, tgt = fftconvolve(s, h)[:N], fftconvolve(s, early_part(h))[:N]
        g = np.sqrt(active_power(s) / active_power(rev))
        rev, tgt = rev * g, tgt * g
        snr = [5, 10][i % 2]
        nz = scale_to_snr(rev, m._noise(["stationary", "nonstationary"][i % 2], N, rng, {}), snr)
        x, c = _norm_pair(rev + nz, tgt)
        add("reverb_cabin", x, c, u, snr)
    sos = butter(6, [300, 3400], "band", fs=FS, output="sos")
    for i in range(30):   # radio_band
        s, u = speech()
        snr = [5, 10][i % 2]
        cat = ["stationary", "nonstationary", "impulsive", "mixed"][i % 4]
        x = s + scale_to_snr(s, m._noise(cat, N, rng, {}), snr)
        x, c = _norm_pair(sosfilt(sos, x), sosfilt(sos, s))
        add("radio_band", x, c, u, snr, cat)
    for i in range(30):   # noise_only
        cat = ["stationary", "nonstationary", "impulsive", "mixed"][i % 4]
        nz = m._noise(cat, N, rng, {})
        nz = nz * np.sqrt(10 ** (-26 / 10) / power(nz))
        nz, _ = _norm_pair(nz, nz)
        add("noise_only", nz, np.zeros(N), None, float("nan"), cat)
    for i in range(30):   # post_blast
        s, u = speech(n=int(4.5 * FS))
        delay = int(FS * (1.0 + rng.uniform(0.05, 0.3)))
        clean = np.zeros(N)
        clean[delay: delay + len(s)] = s[: N - delay]
        blast = synth_noise.generate("explosion", 5.0, FS, rng)
        b = np.zeros(N)
        on = int(FS * 1.0) - int(np.argmax(np.abs(blast) > 0.5 * np.abs(blast).max()))
        b[max(0, on): max(0, on) + len(blast)] = blast[: N - max(0, on)]
        b *= np.sqrt(10 ** (rng.uniform(25, 35) / 10) * active_power(s) / (np.abs(b).max() ** 2 + 1e-12))
        bed = m._component("stationary", N, rng, {}, "bed")
        bed = bed * np.sqrt(active_power(s) / power(bed) * 10 ** (-15 / 10))
        x, c = _norm_pair(clean + b + bed, clean)
        add("post_blast", x, c, u, 15.0, "blast +25..35 dB peak re speech, bed 15 dB")
    for i in range(30):   # noise_switch
        s, u = speech()
        cats = rng.permutation(["stationary", "nonstationary", "impulsive"])
        segs = [m._noise(str(c), 2 * FS, rng, {}) for c in cats]
        nz = np.concatenate([z / np.sqrt(power(z) + 1e-12) for z in segs])   # equal level per segment, abrupt joins
        snr = [0, 5][i % 2]
        x, c = _norm_pair(s + scale_to_snr(s, nz, snr), s)
        add("noise_switch", x, c, u, snr, "->".join(cats))
    t = np.arange(N) / FS
    for i in range(30):   # hum400
        s, u = speech()
        f0 = rng.uniform(395, 405)
        hum = sum((0.6 ** h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3)) for h in range(1, 6))
        hum += 0.5 * sum((0.5 ** h) * np.sin(2 * np.pi * 50 * h * t) for h in range(1, 4))
        bed = synth_noise.generate("tracked_vehicle", 6.0, FS, rng)
        nz = hum / np.sqrt(power(hum)) + 0.3 * bed / np.sqrt(power(bed))
        snr = [0, 5][i % 2]
        x, c = _norm_pair(s + scale_to_snr(s, nz, snr), s)
        add("hum400", x, c, u, snr, f"f0={f0:.1f}")
    for i in range(30):   # crew_babble
        s, u = speech()
        tir = [0, 5, 10][i % 3]
        bab = np.zeros(N)
        for _ in range(rng.integers(2, 4)):
            o, uo = m._speech(rng, n=N, utt=utts[rng.integers(len(utts))])
            if uo["spk"] != u["spk"]:
                bab += o / np.sqrt(active_power(o) + 1e-12)
        bab = scale_to_snr(s, bab, tir)
        bed = scale_to_snr(s, m._component("stationary", N, rng, {}, "bed"), 10)
        x, c = _norm_pair(s + bab + bed, s)
        add("crew_babble", x, c, u, float(tir), "TIR (talker/interferer)")
    for i in range(4):    # long_stream
        L = 60 * FS
        clean = np.zeros(L)
        pos = int(0.5 * FS)
        while pos < L - 3 * FS:
            sp, _ = m._speech(rng, utt=utts[rng.integers(len(utts))], n=None)
            sp = sp * np.sqrt(10 ** (-26 / 10) / active_power(sp))
            seg = sp[: L - pos]
            clean[pos: pos + len(seg)] = seg
            pos += len(seg) + int(rng.uniform(0.3, 1.5) * FS)
        nz = np.concatenate([(lambda z: z / np.sqrt(power(z) + 1e-12))(
            m._noise(str(rng.choice(["stationary", "nonstationary", "impulsive", "mixed"])), 10 * FS, rng, {}))
            for _ in range(6)])
        x, c = _norm_pair(clean + scale_to_snr(clean, nz, 5), clean)
        add("long_stream", x, c, None, 5.0, "60 s, noise changes every 10 s")
    with open(OUT / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "README.txt").write_text(__doc__)
    print(f"[edge] wrote {len(rows)} files to {OUT}")


if __name__ == "__main__":
    main()
