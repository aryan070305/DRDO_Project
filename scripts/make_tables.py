"""Generate every results table of the report directly from the per-file CSVs (no hand transcription).

    python scripts/make_tables.py > reports/results/tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports" / "results"
LABEL = {
    "noisy": "Noisy input", "primary": "Primary mic (unprocessed)", "specsub": "Spectral subtraction",
    "wiener": "Wiener (DD)", "logmmse": "Log-MMSE (LSA)", "file:dfn3": "DeepFilterNet3 (public, 40 ms)",
    "dancnet:hq": "D-ANC v2 HQ base (40 ms)", "dancnet:hq_metric": "D-ANC v2 HQ + fine-tune (40 ms)",
    "dancnet:ll": "D-ANC v2 LL (20 ms)", "dancnet:ll_metric": "D-ANC v2 LL fine-tuned (not adopted)",
    "dancnet:v3hq": "D-ANC v3 HQ (40 ms)", "dancnet:v3ll": "D-ANC v3 LL (20 ms)",
    "dancnet:v3hq_cont": "D-ANC v3 HQ continued (40 ms)",
    "nlms": "NLMS only (ref. mic)", "nlms_mu0.3": "NLMS only, mu=0.3 (untuned)",
    "dnn:hq": "DNN only (v2 HQ base)", "dnn:hqft": "DNN only (v2 HQ + fine-tune)", "dnn:ll": "DNN only (v2 LL)",
    "dnn:v3hq": "DNN only (v3 HQ)", "dnn:v3ll": "DNN only (v3 LL)", "dnn:v3hq_cont": "DNN only (v3 HQ continued)",
    "hybrid:hq": "Hybrid NLMS + v2 HQ base", "hybrid:hqft": "Hybrid NLMS + v2 HQ fine-tuned", "hybrid:ll": "Hybrid NLMS + v2 LL",
    "hybrid:v3hq": "Hybrid NLMS + v3 HQ", "hybrid:v3ll": "Hybrid NLMS + v3 LL", "hybrid:v3hq_cont": "Hybrid NLMS + v3 HQ continued",
    "dnn2:v3_2mic": "Two-mic network v3 (40 ms)", "dnn2_refdead:v3_2mic": "Two-mic network v3, reference mic dead",
    "hybrid2:v3_2mic": "**Cascade: NLMS → two-mic network v3 (40 ms)**",
    "hybrid2_refdead:v3_2mic": "Cascade, reference mic dead",
    "hybrid_mu0.3:hq": "Hybrid, untuned coupling (mu=0.3, 4 taps, gate exp. 2)", "hybrid_nogate:hq": "Hybrid, NO SPP gate",
    "hybrid_nogate_mu0.3:hq": "Hybrid, NO SPP gate, untuned (mu=0.3, 4 taps)",
}
ARCH = {"hq_metric": "v2 + fine-tune", "ll": "v2", "v3hq": "v3", "v3ll": "v3", "v3hq_cont": "v3 continued"}


def deployment():
    """The deployed models come from exports/deployment_selection.json (the single place where they are chosen);
    their table labels are marked "deployed" from there, so the tables can never disagree with the deployment."""
    sel = json.loads((ROOT / "exports" / "deployment_selection.json").read_text())
    dep = {}
    for mode, v in sel.items():
        run, tag = Path(v["checkpoint"]).parent.name, Path(v["onnx"]).stem.replace("dancnet_", "")
        rd = ROOT / "checkpoints" / run          # evaluate.py scores dancnet:<run> with best.pt if present, else last.pt
        scored = "best.pt" if (rd / "best.pt").exists() else "last.pt"
        if rd.is_dir() and Path(v["checkpoint"]).name != scored:     # (skipped when checkpoints/ is not shipped)
            raise SystemExit(f"deployment_selection.json: {mode} uses {v['checkpoint']} but the dancnet:{run} rows of "
                             f"the result tables come from {run}/{scored} - the 'deployed' label would be wrong")
        try:     # algorithmic latency from the exported model's look-ahead (ONNX metadata), not from the mode name
            import onnxruntime as ort
            la = int(ort.InferenceSession(str(ROOT / v["onnx"]), providers=["CPUExecutionProvider"])
                     .get_modelmeta().custom_metadata_map.get("lookahead", 0))
        except Exception:   # noqa: BLE001
            la = 2 if mode.upper() != "LL" else 0
        lat = f"{20 + 10 * la} ms"
        dep[mode.upper()] = (run, tag)
        if (v.get("engine") or {}).get("lms_two_mic"):     # two-microphone cascade entry
            LABEL[f"hybrid2:{tag}"] = f"**Cascade NLMS → two-mic network ({mode.upper()} deployed, {lat})**"
            continue
        LABEL[f"dancnet:{run}"] = f"**D-ANC {mode.upper()} deployed ({lat}, {ARCH.get(run, run)})**"
        LABEL[f"dnn:{tag}"] = f"DNN only ({mode.upper()} deployed)"
        LABEL[f"hybrid:{tag}"] = f"**Hybrid NLMS+DNN ({mode.upper()} deployed)**"
    return dep
MET = ["pesq_wb", "pesq_nb", "stoi", "estoi", "snr", "delta_snr", "sisdr"]
HDR = ["PESQ-WB", "PESQ-NB", "STOI", "ESTOI", "SNR out [dB]", "ΔSNR [dB]", "SI-SDR [dB]"]


def fmt(v, m):
    return f"{v:.3f}" if m in ("stoi", "estoi", "pesq_wb", "pesq_nb") else f"{v:.2f}"


def overall(df, systems, title, snr_min=-5):
    d = df if df["snr_db"].isna().all() else df[df["snr_db"] >= snr_min]
    out = [f"\n#### {title}\n", "| System | " + " | ".join(HDR) + " | n |", "|---" * (len(HDR) + 2) + "|"]
    for s in systems:
        x = d[d.system == s]
        if not len(x):
            continue
        out.append(f"| {LABEL.get(s, s)} | " + " | ".join(fmt(x[m].mean(), m) for m in MET) + f" | {len(x)} |")
    return "\n".join(out)


def per_snr(df, systems, metrics, title):
    snrs = sorted(df["snr_db"].dropna().unique())
    out = [f"\n#### {title}\n"]
    for m, h in metrics:
        out += [f"\n*{h}*\n", "| System | " + " | ".join(f"{int(s)} dB" for s in snrs) + " |", "|---" * (len(snrs) + 1) + "|"]
        for s in systems:
            x = df[df.system == s]
            if not len(x):
                continue
            g = x.groupby("snr_db")[m].mean()
            out.append(f"| {LABEL.get(s, s)} | " + " | ".join(fmt(g.get(v, float('nan')), m) for v in snrs) + " |")
    return "\n".join(out)


def per_cat(df, systems, title):
    d = df[df["snr_db"] >= -5]
    cats = ["stationary", "nonstationary", "impulsive", "mixed"]
    out = [f"\n#### {title}\n", "| System | " + " | ".join(f"{c} PESQ-WB / STOI / SNR" for c in cats) + " |",
           "|---" * (len(cats) + 1) + "|"]
    for s in systems:
        x = d[d.system == s]
        if not len(x):
            continue
        cells = []
        for c in cats:
            y = x[x.category == c]
            cells.append(f"{y.pesq_wb.mean():.2f} / {y.stoi.mean():.3f} / {y.snr.mean():.1f}")
        out.append(f"| {LABEL.get(s, s)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def targets(df, systems, title):
    d = df[df["snr_db"] >= -5]
    out = [f"\n#### {title}\n",
           "| System | SNR>15 dB | STOI>0.85 | PESQ-WB>2.5 | all three | PESQ-NB>4 | PESQ-WB>4 |", "|---" * 7 + "|"]
    for s in systems:
        x = d[d.system == s]
        if not len(x):
            continue
        allt = ((x.snr > 15) & (x.stoi > 0.85) & (x.pesq_wb > 2.5)).mean()
        out.append(f"| {LABEL.get(s, s)} | {(x.snr > 15).mean():.1%} | {(x.stoi > 0.85).mean():.1%} | "
                   f"{(x.pesq_wb > 2.5).mean():.1%} | {allt:.1%} | {(x.pesq_nb > 4).mean():.1%} | {(x.pesq_wb > 4).mean():.1%} |")
    return "\n".join(out)


def target_by_snr(df, s, title):
    out = [f"\n#### {title}\n", "| input SNR | PESQ-WB | PESQ-NB | STOI | SNR out | meets SNR>15 | STOI>0.85 | PESQ>2.5 |",
           "|---" * 8 + "|"]
    x = df[df.system == s]
    for v, y in x.groupby("snr_db"):
        pw, pn, st, sn = y.pesq_wb.mean(), y.pesq_nb.mean(), y.stoi.mean(), y.snr.mean()
        out.append(f"| {int(v)} dB | {pw:.2f} | {pn:.2f} | {st:.3f} | {sn:.1f} dB | {'yes' if sn > 15 else 'no'} | "
                   f"{'yes' if st > 0.85 else 'no'} | {'yes' if pw > 2.5 else 'no'} |")
    return "\n".join(out)


def pesq_attainment(df, systems, title):
    """Mean PESQ-WB / PESQ-NB per input SNR and the share of files at or above 3.25 and 3.5."""
    out = [f"\n#### {title}\n", "| System | input SNR | PESQ-WB mean | PESQ-NB mean | WB ≥ 3.25 | WB ≥ 3.5 | NB ≥ 3.25 | NB ≥ 3.5 |",
           "|---" * 8 + "|"]
    for s in systems:
        x = df[df.system == s]
        if not len(x):
            continue
        for v, y in x.groupby("snr_db"):
            out.append(f"| {LABEL.get(s, s)} | {int(v)} dB | {y.pesq_wb.mean():.2f} | {y.pesq_nb.mean():.2f} | "
                       f"{(y.pesq_wb >= 3.25).mean():.0%} | {(y.pesq_wb >= 3.5).mean():.0%} | "
                       f"{(y.pesq_nb >= 3.25).mean():.0%} | {(y.pesq_nb >= 3.5).mean():.0%} |")
    return "\n".join(out)


def snr_needed(df, systems, title):
    """Input SNR at which the MEAN PESQ reaches 3.25 / 3.5, by linear interpolation between the measured SNR points
    (no extrapolation: '> top' means not reached within the tested range)."""
    out = [f"\n#### {title}\n", "| System | PESQ-WB ≥ 3.25 | PESQ-WB ≥ 3.5 | PESQ-NB ≥ 3.25 | PESQ-NB ≥ 3.5 |", "|---" * 5 + "|"]
    for s in systems:
        x = df[df.system == s]
        if not len(x):
            continue
        cells = []
        for m in ("pesq_wb", "pesq_nb"):
            g = x.groupby("snr_db")[m].mean().sort_index()
            for thr in (3.25, 3.5):
                v, prev = None, None
                for snr_v, val in g.items():
                    if val >= thr:
                        v = snr_v if prev is None else prev[0] + (thr - prev[1]) / (val - prev[1]) * (snr_v - prev[0])
                        break
                    prev = (snr_v, val)
                if v is not None and prev is None:      # met already at the lowest tested SNR: crossing unknown
                    cells.append(f"≤ {v:.0f} dB")
                else:
                    cells.append(f"{v:.1f} dB" if v is not None else f"> {int(g.index.max())} dB (not reached)")
        out.append(f"| {LABEL.get(s, s)} | " + " | ".join(cells) + " |")
    out.append("\nLinear interpolation of the per-SNR means in Tables R12a/R12b; values are input SNRs at the primary microphone.")
    return "\n".join(out)


def edge_table(systems, title):
    p = R / "defence_edge_v1" / "per_file.csv"
    if not p.exists():
        return ""
    d = pd.read_csv(p)
    cases = ["clip_heavy", "level_low", "level_high", "reverb_cabin", "radio_band", "post_blast", "noise_switch",
             "hum400", "crew_babble", "long_stream"]
    syss = [s for s in systems if s in set(d.system)]
    out = [f"\n#### {title}\n", "| Case | " + " | ".join(LABEL.get(s, s) for s in syss) + " |", "|---" * (len(syss) + 1) + "|"]
    for c in cases:
        cells = []
        for s in syss:
            y = d[(d.case == c) & (d.system == s)]
            cells.append(f"{y.pesq_wb.mean():.2f} / {y.pesq_nb.mean():.2f} / {y.stoi.mean():.3f} / {y.snr.mean():.1f}" if len(y) else "—")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    cells = []
    for s in syss:
        y = d[(d.case == "noise_only") & (d.system == s)]
        cells.append(f"attenuation {y.attenuation_db.mean():.1f} dB" if len(y) else "—")
    out.append("| noise_only | " + " | ".join(cells) + " |")
    cells = []
    for s in syss:
        y = d[(d.case == "post_blast") & (d.system == s)]
        cells.append(f"recovery STOI {y.recovery_stoi.mean():.3f}" if len(y) else "—")
    out.append("| post_blast (2 s after the blast) | " + " | ".join(cells) + " |")
    out.append("\nCells: PESQ-WB / PESQ-NB / STOI / output SNR [dB], mean over 30 files per case (4 × 60 s for long_stream).")
    return "\n".join(out)


def ablation_table(df, title):
    arms = [("dancnet:abl_base", "reference arm (deep filter 5 taps, no look-ahead, median normaliser)"),
            ("dancnet:abl_ema", "exponential-mean normaliser instead of median"),
            ("dancnet:abl_crm", "plain complex mask (1 tap) instead of deep filter"),
            ("dancnet:abl_la2", "look-ahead 2 frames (40 ms)"),
            ("dancnet:abl_la4", "look-ahead 4 frames (60 ms)"),
            ("dancnet:abl_nopmsqe", "no PMSQE perceptual loss")]
    d = df[df.snr_db >= -5]
    present = set(d.system)
    # ablation runs may carry a batch suffix (e.g. dancnet:abl_base_b32, see scripts/choose_batch.py)
    resolve = {a: next((s for s in sorted(present) if s == a or s.startswith(a + "_b")), None) for a, _ in arms}
    arms = [(resolve[a], desc) for a, desc in arms if resolve[a]]
    if not arms:
        return ""
    base = d[d.system == arms[0][0]] if arms[0][0].startswith("dancnet:abl_base") else d.iloc[0:0]
    out = [f"\n#### {title}\n", "| Arm | PESQ-WB | PESQ-NB | STOI | output SNR | SI-SDR | Δ PESQ-WB vs ref | Δ STOI vs ref | impulsive STOI |",
           "|---" * 9 + "|"]
    for a, desc in arms:
        x = d[d.system == a]
        if not len(x):
            continue
        dp = x.pesq_wb.mean() - base.pesq_wb.mean() if len(base) else float("nan")
        ds = x.stoi.mean() - base.stoi.mean() if len(base) else float("nan")
        out.append(f"| {desc} | {x.pesq_wb.mean():.3f} | {x.pesq_nb.mean():.3f} | {x.stoi.mean():.3f} | {x.snr.mean():.2f} | "
                   f"{x.sisdr.mean():.2f} | {dp:+.3f} | {ds:+.4f} | {x[x.category == 'impulsive'].stoi.mean():.3f} |")
    return "\n".join(out)


def main():
    de = pd.read_csv(R / "defence_v1" / "per_file.csv")
    dm = pd.read_csv(R / "dualmic_v1" / "per_file.csv")
    vb = pd.read_csv(R / "vbdemand" / "per_file.csv")
    dep = deployment()
    hq_run, hq_tag = dep["HQ"]
    ll_run, ll_tag = dep["LL"]
    v3 = ["dancnet:v3ll", "dancnet:v3hq", "dancnet:v3hq_cont"]
    uniq = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    sys_de = uniq(["noisy", "specsub", "wiener", "logmmse", "file:dfn3", "dancnet:ll", "dancnet:ll_metric", "dancnet:hq",
                   "dancnet:hq_metric"] + v3)
    sys_dm = uniq(["primary", "logmmse", "nlms_mu0.3", "nlms", "hybrid_nogate_mu0.3:hq", "hybrid_nogate:hq", "dnn:hq",
                   "hybrid_mu0.3:hq", "hybrid:hq", "dnn:hqft", "hybrid:hqft", "dnn:ll", "hybrid:ll", "dnn:v3ll",
                   "hybrid:v3ll", "dnn:v3hq", "hybrid:v3hq", "dnn:v3hq_cont", "hybrid:v3hq_cont", "dnn2:v3_2mic",
                   "dnn2_refdead:v3_2mic", "hybrid2:v3_2mic", "hybrid2_refdead:v3_2mic"])
    sys_vb = uniq(["noisy", "logmmse", "file:dfn3", "dancnet:hq", "dancnet:hq_metric", "dancnet:ll", "dancnet:ll_metric"] + v3)
    deployed = [s for s in [f"dancnet:{hq_run}", f"dancnet:{ll_run}"] if s in set(de.system)]
    key_de = uniq(["noisy", "logmmse", "file:dfn3", "dancnet:ll", "dancnet:hq_metric", f"dancnet:{ll_run}",
                   f"dancnet:{hq_run}"] + v3)
    key_dm = uniq(["primary", "nlms", "dnn:hqft", "hybrid:hqft", "dnn:ll", "hybrid:ll", f"hybrid:{ll_tag}", f"hybrid:{hq_tag}",
                   "hybrid:v3hq", "hybrid:v3hq_cont", "dnn2:v3_2mic", "hybrid2:v3_2mic"])
    print("# Results tables (auto-generated by scripts/make_tables.py from reports/results/*/per_file.csv)")
    print(overall(de, sys_de, "Table R1: defence_v1, single microphone, mean over input SNR −5…15 dB (800 files per system)"))
    print(per_snr(de, key_de,
                  [("pesq_wb", "PESQ-WB"), ("pesq_nb", "PESQ-NB"), ("stoi", "STOI"), ("snr", "output SNR [dB]")],
                  "Table R2: defence_v1 by input SNR (−10 dB = stress condition, 100 files; others 160 files)"))
    print(per_cat(de, key_de,
                  "Table R3: defence_v1 by noise category (−5…15 dB)"))
    print(targets(de, uniq(["noisy", "file:dfn3"] + deployed + ["dancnet:hq_metric"] + v3), "Table R4: share of defence_v1 files meeting each target (−5…15 dB)"))
    for s in deployed:
        print(target_by_snr(de, s, f"Table R5: target check by input SNR, {LABEL[s].strip('*')}"))
    print(overall(dm, sys_dm, "Table R6: dualmic_v1, primary + reference microphone (240 scenes, −5…10 dB)"))
    print(per_snr(dm, key_dm,
                  [("pesq_wb", "PESQ-WB"), ("stoi", "STOI"), ("snr", "output SNR [dB]")], "Table R7: dualmic_v1 by input SNR"))
    print(targets(dm, key_dm,
                  "Table R8: share of dualmic_v1 scenes meeting each target"))
    print(overall(vb, sys_vb, "Table R9: VoiceBank+DEMAND test (824 files; civilian benchmark, not defence noise)"))
    dep_de = [s for s in uniq(deployed + ["dancnet:hq_metric", "dancnet:ll"] + v3) if s in set(de.system)]
    # defence_highsnr_v1 (20 / 25 dB, built exactly like defence_v1) extends the SNR axis of the attainment tables
    hs = R / "defence_highsnr_v1" / "per_file.csv"
    de_ext, ext = (pd.concat([de, pd.read_csv(hs)], ignore_index=True), " + defence_highsnr_v1 (20, 25 dB)") if hs.exists() else (de, "")
    print(pesq_attainment(de_ext, dep_de, f"Table R12a: PESQ attainment by input SNR, defence_v1 (single mic){ext}"))
    dep_dm = [s for s in uniq([f"dnn:{hq_tag}", f"hybrid:{hq_tag}", f"hybrid:{ll_tag}", "hybrid:hqft", "hybrid:ll", "hybrid:v3hq",
                               "hybrid:v3hq_cont", "dnn2:v3_2mic", "hybrid2:v3_2mic"]) if s in set(dm.system)]
    print(pesq_attainment(dm, dep_dm, "Table R12b: PESQ attainment by input SNR, dualmic_v1 (primary + reference mic)"))
    print(snr_needed(de_ext, uniq(["file:dfn3"] + dep_de), f"Table R12c: input SNR needed for a mean PESQ of 3.25 / 3.5, defence_v1 (single mic){ext}"))
    print(snr_needed(dm, dep_dm, "Table R12d: input SNR needed for a mean PESQ of 3.25 / 3.5, dualmic_v1 (primary + reference mic)"))
    print(edge_table(uniq(["noisy", "logmmse", "dancnet:ll", "dancnet:hq_metric"] + v3),
                     "Table R11: edge-case set defence_edge_v1"))
    print(ablation_table(de, "Table R13: ablations (v2 architecture, 2 500 steps each, same data order; defence_v1 −5…15 dB)"))
    tp = R / "transparency_clean_input.json"
    if tp.exists():
        t = json.loads(tp.read_text())
        print("\n#### Table R10: transparency (clean speech in → model → compared with the input)\n")
        print("| Model | VoiceBank clean PESQ-WB / STOI / SI-SDR | LibriSpeech clean PESQ-WB / STOI / SI-SDR |\n|---|---|---|")
        for run in sorted({k.split('/')[0] for k in t}):
            a, b = t.get(f"{run}/vbd_clean"), t.get(f"{run}/libri_clean")
            if a and b:
                print(f"| {run} | {a['pesq_wb']:.2f} / {a['stoi']:.3f} / {a['sisdr']:.1f} | "
                      f"{b['pesq_wb']:.2f} / {b['stoi']:.3f} / {b['sisdr']:.1f} |")

    lat = next((R / f for f in ("latency_mac_m4_round2_final.json", "latency_mac_m4_round2.json") if (R / f).exists()), None)
    if lat is not None:
        L = json.loads(lat.read_text())
        print(f"\n#### Table R14: per-hop compute, Apple M4, ONNX Runtime CPU, 1 thread, {L['frames']} hops, unpaced\n")
        print("| Model (ONNX) | mics | algorithmic latency | network step mean / p99.9 [ms] | full engine block mean / p99.9 / max [ms] | RTF | misses > 10 ms |")
        print("|---|---|---|---|---|---|---|")
        for k, v in L["models"].items():
            n, f = v["network_step"], v["full_engine_block"]
            nm = (("NLMS → " if v.get("cascade") else "") + v["onnx"].split("/")[-1]
                  + (f" + {v['fallback_onnx'].split('/')[-1]} (hot standby)" if v.get("fallback_onnx") else ""))
            ncell = f"{n['mean_ms']:.2f} / {n['p99_9_ms']:.2f}" if n else "—"
            print(f"| {nm} | {v.get('n_mics', 1)} | {v['algorithmic_latency_ms']:.0f} ms | {ncell} | "
                  f"{f['mean_ms']:.2f} / {f['p99_9_ms']:.2f} / {f['max_ms']:.2f} | {f['rtf']:.3f} | {f['deadline_misses_gt_10ms']} |")


if __name__ == "__main__":
    main()
