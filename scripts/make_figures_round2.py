"""Round-2 report figures (the round-1 figures stay in scripts/make_figures.py):

  reports/figures/round2_pesq_attainment.png  mean PESQ-WB / PESQ-NB per input SNR vs the 3.25 / 3.5 request
  reports/figures/round2_edge_cases.png       STOI and PESQ-WB per edge case (defence_edge_v1)
  reports/figures/round2_ablations.png        ablation arms vs the reference arm (defence_v1, -5..15 dB)
  reports/figures/round2_training_curves.png  validation PESQ-WB of the round-2 runs (logs/train_*.jsonl)

    python scripts/make_figures_round2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_tables import LABEL, deployment  # noqa: E402

R = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"
COLORS = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]


def lab(s):
    return LABEL.get(s, s).replace("**", "")


def present(df, systems):
    have = set(df.system)
    return [s for s in dict.fromkeys(systems) if s in have]


def pesq_attainment(dep):
    de = pd.read_csv(R / "defence_v1" / "per_file.csv")
    hs = R / "defence_highsnr_v1" / "per_file.csv"          # same construction at 20 / 25 dB: extends the SNR axis
    if hs.exists():
        de = pd.concat([de, pd.read_csv(hs)], ignore_index=True)
    dm = pd.read_csv(R / "dualmic_v1" / "per_file.csv")
    hq_run, hq_tag = dep["HQ"]
    ll_run, _ = dep["LL"]
    s_de = present(de, ["noisy", "file:dfn3", f"dancnet:{ll_run}", f"dancnet:{hq_run}", "dancnet:v3ll", "dancnet:v3hq",
                        "dancnet:v3hq_cont"])
    s_dm = present(dm, ["primary", f"hybrid:{hq_tag}", "hybrid:v3hq", "hybrid:v3hq_cont", "dnn2:v3_2mic", "hybrid2:v3_2mic"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True, sharey=True)
    for i, (df, systems, name) in enumerate([(de, s_de, "defence_v1 (+ 20/25 dB extension), single mic"),
                                             (dm, s_dm, "dualmic_v1, primary + reference microphone")]):
        for j, (m, mname) in enumerate([("pesq_wb", "PESQ-WB (P.862.2)"), ("pesq_nb", "PESQ-NB (P.862)")]):
            ax = axes[i, j]
            for k, s in enumerate(systems):
                g = df[df.system == s].groupby("snr_db")[m].mean()
                ax.plot(g.index, g.values, "o-", color=COLORS[k % len(COLORS)], label=lab(s), ms=4, lw=1.6)
            for y, ls in ((3.25, "--"), (3.5, ":")):
                ax.axhline(y, color="k", ls=ls, lw=1, alpha=0.7)
                ax.text(0.99, y + 0.03, f"requested {y}", fontsize=8, ha="right", transform=ax.get_yaxis_transform())
            ax.set_title(f"{name}: {mname}", fontsize=10)
            ax.set_xlabel("input SNR [dB]")
            ax.set_ylabel(f"mean {mname}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Mean PESQ per input SNR vs the requested 3.25-3.5 (dashed / dotted)", fontsize=12)
    out = FIG / "round2_pesq_attainment.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def edge_cases(dep):
    p = R / "defence_edge_v1" / "per_file.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    hq_run = dep["HQ"][0]
    systems = present(d, ["noisy", "logmmse", f"dancnet:{hq_run}", "dancnet:v3hq", "dancnet:v3hq_cont"])
    if not systems:
        return None
    cases = ["clip_heavy", "level_low", "level_high", "reverb_cabin", "radio_band", "post_blast", "noise_switch",
             "hum400", "crew_babble", "long_stream"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), constrained_layout=True, sharex=True)
    w = 0.8 / len(systems)
    for i, (m, mname) in enumerate([("stoi", "STOI"), ("pesq_wb", "PESQ-WB")]):
        ax = axes[i]
        for k, s in enumerate(systems):
            v = [d[(d.case == c) & (d.system == s)][m].mean() for c in cases]
            ax.bar(np.arange(len(cases)) + k * w, v, w, color=COLORS[k % len(COLORS)], label=lab(s))
        ax.set_ylabel(mname)
        ax.grid(axis="y", alpha=0.3)
        if m == "stoi":
            ax.axhline(0.85, color="k", ls="--", lw=1)
            ax.set_ylim(0.6, 1.0)
        else:
            ax.axhline(2.5, color="k", ls="--", lw=1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=len(systems), fontsize=8)
    axes[1].set_xticks(np.arange(len(cases)) + w * (len(systems) - 1) / 2)
    axes[1].set_xticklabels(cases, rotation=20)
    axes[0].set_title("Edge-case set defence_edge_v1 (30 files per case, 4 x 60 s long streams)", fontsize=11)
    out = FIG / "round2_edge_cases.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def ablations():
    de = pd.read_csv(R / "defence_v1" / "per_file.csv")
    d = de[de.snr_db >= -5]
    arms = [("abl_base", "reference\n(L=0, DF5, median)"), ("abl_ema", "EMA\nnormaliser"), ("abl_crm", "complex mask\n(no deep filter)"),
            ("abl_la2", "look-ahead 2\n(40 ms)"), ("abl_la4", "look-ahead 4\n(60 ms)"), ("abl_nopmsqe", "no PMSQE\nloss")]
    have = set(d.system)
    arms = [(a, t) for a, t in arms if f"dancnet:{a}" in have]
    if len(arms) < 2 or arms[0][0] != "abl_base":
        return None
    base = d[d.system == "dancnet:abl_base"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (m, mname) in zip(axes, [("pesq_wb", "Δ PESQ-WB"), ("stoi", "Δ STOI"), ("sisdr", "Δ SI-SDR [dB]")]):
        v = [d[d.system == f"dancnet:{a}"][m].mean() - base[m].mean() for a, _ in arms[1:]]
        ax.bar(range(len(v)), v, color=["#2ca02c" if x > 0 else "#d62728" for x in v])
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels([t for _, t in arms[1:]], fontsize=8)
        ax.set_title(f"{mname} vs reference arm", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablations: v2 architecture, 2 500 steps each, identical data order; defence_v1 (-5..15 dB, 800 files)",
                 fontsize=11)
    out = FIG / "round2_ablations.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def training_curves():
    runs = [("v3hq", 0, "v3 HQ"), ("v3hq_cont", 16000, "v3 HQ continued (warm restart at 16k)"), ("v3ll", 0, "v3 LL (warm start)"),
            ("v3_2mic", 0, "two-mic v3 (warm start; dual-mic validation set)")]
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    for k, (r, off, name) in enumerate(runs):
        p = ROOT / "logs" / f"train_{r}.jsonl"
        if not p.exists():
            continue
        pts = {}
        for line in p.read_text().splitlines():
            e = json.loads(line)
            if e.get("type") == "val" and "step" in e:
                pts[int(e["step"]) + off] = e["pesq"]
        if pts:
            x = sorted(pts)
            ax.plot(x, [pts[i] for i in x], "o-", color=COLORS[(k + 1) % len(COLORS)], label=name, ms=4)
    ax.set_xlabel("training step (batch 16 x 3 s)")
    ax.set_ylabel("validation PESQ-WB (EMA weights)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Round-2 training runs on the Apple M4 GPU (MPS)")
    out = FIG / "round2_training_curves.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    dep = deployment()
    for f in (pesq_attainment(dep), edge_cases(dep), ablations(), training_curves()):
        if f:
            print(f)


if __name__ == "__main__":
    main()
