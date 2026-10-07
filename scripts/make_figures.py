"""Report figures from reports/results/*/per_file.csv and logs/train_*.jsonl.

    python scripts/make_figures.py --systems noisy,logmmse,file:dfn3,dancnet_hq --labels ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
CATS = ["stationary", "nonstationary", "impulsive", "mixed"]
COLORS = ["#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
TARGETS = {"pesq_wb": 2.5, "stoi": 0.85, "snr": 15.0}
YLAB = {"pesq_wb": "PESQ-WB (P.862.2)", "pesq_nb": "PESQ-NB (P.862)", "stoi": "STOI", "snr": "output SNR [dB]",
        "estoi": "ESTOI", "sisdr": "SI-SDR [dB]"}


def metric_vs_snr(df, systems, labels, set_name, metrics=("pesq_wb", "stoi", "snr")):
    fig, axes = plt.subplots(len(metrics), len(CATS), figsize=(16, 3.2 * len(metrics)), sharex=True,
                             constrained_layout=True)
    for i, m in enumerate(metrics):
        for j, c in enumerate(CATS):
            ax = axes[i, j]
            for k, (s, lab) in enumerate(zip(systems, labels)):
                d = df[(df.system == s) & (df.category == c)].groupby("snr_db")[m].mean()
                if len(d):
                    ax.plot(d.index, d.values, "o-", color=COLORS[k % len(COLORS)], label=lab, ms=4, lw=1.6)
            if m in TARGETS:
                ax.axhline(TARGETS[m], color="k", ls="--", lw=1, alpha=0.6)
            if i == 0:
                ax.set_title(c)
            if j == 0:
                ax.set_ylabel(YLAB[m])
            if i == len(metrics) - 1:
                ax.set_xlabel("input SNR [dB]")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{set_name}: metrics vs input SNR (dashed = project target)")
    out = FIG / f"{set_name}_metrics_vs_snr.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def pass_rates(df, systems, labels, set_name):
    d = df[df.snr_db >= -5]
    rows = []
    for s, lab in zip(systems, labels):
        x = d[d.system == s]
        if len(x):
            rows.append([lab] + [float((x[m] > t).mean()) for m, t in TARGETS.items()]
                        + [float(((x.snr > 15) & (x.stoi > 0.85) & (x.pesq_wb > 2.5)).mean())])
    cols = ["SNR>15 dB", "STOI>0.85", "PESQ>2.5", "all three"]
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    w = 0.8 / max(1, len(rows))
    for k, r in enumerate(rows):
        ax.bar(np.arange(4) + k * w, r[1:], w, label=r[0], color=COLORS[k % len(COLORS)])
    ax.set_xticks(np.arange(4) + w * (len(rows) - 1) / 2)
    ax.set_xticklabels(cols)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of test files")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{set_name}: share of files meeting each target (input SNR -5..15 dB)")
    out = FIG / f"{set_name}_pass_rates.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def training_curves(runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
    for k, r in enumerate(runs):
        p = ROOT / "logs" / f"train_{r}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(line) for line in open(p)]
        tr = [x for x in recs if x["type"] == "train"]
        va = [x for x in recs if x["type"] == "val" and "pesq" in x]
        if tr:
            axes[0].plot([x["step"] for x in tr], [x["loss"] for x in tr if "loss" in x][: len(tr)], color=COLORS[k + 4],
                         label=r, lw=1)
        if va:
            axes[1].plot([x["step"] for x in va], [x["pesq"] for x in va], "o-", color=COLORS[k + 4], label=r)
            axes[2].plot([x["step"] for x in va], [x["stoi"] for x in va], "o-", color=COLORS[k + 4], label=r)
    for ax, t in zip(axes, ["training loss", "validation PESQ-WB", "validation STOI"]):
        ax.set_title(t)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    out = FIG / "training_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def example_spectrograms(set_name, ids, systems, labels):
    import soundfile as sf
    from scipy.signal import spectrogram

    d = ROOT / "data" / "testsets" / set_name
    rows = len(ids)
    systems_with_audio = [s for s in systems if (ROOT / "reports" / "audio" /
                          f"enhanced_{set_name}_{s.replace('file:', '').replace(':', '_')}").exists()]
    cols = 2 + len(systems_with_audio)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, uid in enumerate(ids):
        sigs = [("noisy", d / "noisy" / f"{uid}.flac"), ("clean", d / "clean" / f"{uid}.flac")]
        sigs += [(lab, ROOT / "reports" / "audio" / f"enhanced_{set_name}_{s.replace('file:', '').replace(':', '_')}"
                  / f"{uid}.flac") for s, lab in zip(systems, labels)]
        sigs = [x for x in sigs if Path(x[1]).exists()]
        for j, (lab, path) in enumerate(sigs):
            ax = axes[i, j]
            if not Path(path).exists():
                ax.axis("off")
                continue
            x, fs = sf.read(path)
            f, t, S = spectrogram(x, fs, nperseg=320, noverlap=240)
            ax.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-30)
            ax.set_title(f"{lab}" + (f"\n{uid}" if j == 0 else ""), fontsize=8)
            ax.tick_params(labelsize=7)
    out = FIG / f"{set_name}_example_spectrograms.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="defence_v1")
    ap.add_argument("--systems", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--runs", default="v2_hq")
    ap.add_argument("--examples", default="")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    systems = a.systems.split(",")
    labels = a.labels.split(",") if a.labels else systems
    df = pd.read_csv(ROOT / "reports" / "results" / a.set / "per_file.csv")
    print(metric_vs_snr(df, systems, labels, a.set))
    print(pass_rates(df, systems, labels, a.set))
    print(training_curves(a.runs.split(",")))
    if a.examples:
        enh = [s for s in systems if s not in ("noisy", "primary")]
        print(example_spectrograms(a.set, a.examples.split(","), enh, [labels[systems.index(s)] for s in enh]))


if __name__ == "__main__":
    main()
