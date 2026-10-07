"""Spectrogram gallery + example wavs of every synthetic defence-noise generator (fixed seed)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from danc.data.synth_noise import KINDS, generate

ROOT = Path(__file__).resolve().parents[1]
FS = 16000


def main():
    wav_dir = ROOT / "data" / "synthetic_examples"
    fig_dir = ROOT / "reports" / "figures"
    wav_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    kinds = sorted(KINDS, key=lambda k: (KINDS[k]["category"], k))
    cols = 3
    rows = int(np.ceil(len(kinds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.6 * rows), constrained_layout=True)
    for ax, k in zip(axes.flat, kinds):
        y = generate(k, 6.0, FS, np.random.default_rng(2026))
        sf.write(wav_dir / f"{k}.wav", y, FS)
        f, t, S = spectrogram(y, FS, nperseg=512, noverlap=384)
        ax.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma",
                      vmin=10 * np.log10(S.max() + 1e-12) - 70)
        ax.set_title(f"{k}  [{KINDS[k]['category']}]", fontsize=9)
        ax.set_ylabel("kHz", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(kinds):]:
        ax.axis("off")
    fig.suptitle("Synthetic defence-noise generators (6 s, seed 2026)", fontsize=11)
    fig.savefig(fig_dir / "synth_noise_grid.png", dpi=110)
    print("saved", fig_dir / "synth_noise_grid.png")


if __name__ == "__main__":
    main()
