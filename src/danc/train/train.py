"""DANCNet training script.

    python -m danc.train.train --config configs/train_main.yaml

Writes checkpoints/<name>/{last.pt,best.pt,config.yaml}, logs/train_<name>.jsonl.
Validation (fixed, deterministic set from held-out speakers + training-split noise
audio rendered with a disjoint seed) reports PESQ-WB / STOI / SI-SDR every
`val_every` steps; best.pt is selected on PESQ + 2*STOI.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from danc.data.mixer import EVAL_CATS, DefenceMixer, MixDataset
from danc.dsp import TorchSTFT
from danc.models.dancnet import DANCNet, DANCNetConfig, count_params
from danc.train.losses import DANCLoss

ROOT = Path(__file__).resolve().parents[3]


def _metrics_one(args):
    from pesq import pesq
    from pystoi import stoi

    est, ref = args
    out = {}
    try:
        out["pesq"] = float(pesq(16000, ref, est, "wb"))
    except Exception:
        out["pesq"] = float("nan")
    out["stoi"] = float(stoi(ref, est, 16000, extended=False))
    r = ref - ref.mean()
    e = est - est.mean()
    a = np.dot(e, r) / (np.dot(r, r) + 1e-8)
    out["sisdr"] = float(10 * np.log10(np.sum((a * r) ** 2) / (np.sum((e - a * r) ** 2) + 1e-8) + 1e-8))
    return out


def build_val_set(n: int, seed: int = 4242, segment_s: float = 4.0, use_synth: bool = True):
    mixer = DefenceMixer("val", segment_s, use_synth=use_synth, augment=False)
    rng = np.random.default_rng(seed)
    noisy, clean, cats = [], [], []
    for i in range(n):
        cat = EVAL_CATS[i % len(EVAL_CATS)]
        snr = [-5, 0, 5, 10, 15][(i // len(EVAL_CATS)) % 5]
        ex = mixer.sample(rng, category=cat, snr_db=snr)
        noisy.append(ex["noisy"])
        clean.append(ex["clean"])
        cats.append(cat)
    return np.stack(noisy), np.stack(clean), cats


@torch.no_grad()
def validate(model, stft, noisy, clean, cats, device, pool, bs=16):
    model.eval()
    outs = []
    for i in range(0, len(noisy), bs):
        x = torch.from_numpy(noisy[i:i + bs]).to(device)
        X = torch.view_as_real(stft(x))
        Y, _ = model.enhance_spec(X)
        y = stft.inverse(torch.view_as_complex(Y.contiguous()), x.shape[-1])
        outs.append(y.cpu().numpy())
    est = np.concatenate(outs)
    res = list(pool.map(_metrics_one, [(est[i].astype(np.float64), clean[i].astype(np.float64)) for i in range(len(est))]))
    model.train()
    agg = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
    for c in EVAL_CATS:
        idx = [i for i, cc in enumerate(cats) if cc == c]
        agg[f"pesq_{c}"] = float(np.nanmean([res[i]["pesq"] for i in idx]))
        agg[f"stoi_{c}"] = float(np.nanmean([res[i]["stoi"] for i in idx]))
    return agg


def lr_at(step, cfg):
    w = cfg["warmup_steps"]
    if step < w:
        return cfg["lr"] * (step + 1) / w
    p = min(1.0, (step - w) / max(1, cfg["max_steps"] - w))
    return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * p))


def restore_ema(ema, state_dict, step: int):
    """Load EMA weights so that the next update_parameters() AVERAGES into them.

    torch.optim.swa_utils.AveragedModel copies the raw weights on its first update while n_averaged == 0, which
    would silently discard EMA weights restored from a checkpoint."""
    ema.module.load_state_dict(state_dict)
    ema.n_averaged.fill_(max(1, int(step)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    ckdir = ROOT / "checkpoints" / name
    ckdir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logf = open(ROOT / "logs" / f"train_{name}.jsonl", "a")
    yaml.safe_dump(cfg, open(ckdir / "config.yaml", "w"))

    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "mps" if torch.backends.mps.is_available() else "cpu"))
    mcfg = DANCNetConfig(**cfg.get("model", {}))
    model = DANCNet(mcfg).to(device)
    stft = TorchSTFT().to(device)
    lossf = DANCLoss(**cfg.get("loss", {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg.get("weight_decay", 1e-4))
    if cfg.get("init_ckpt"):
        # warm start (weights only, e.g. the 20 ms model initialised from the 40 ms look-ahead model;
        # both have identical weight shapes - only the output alignment / state shapes differ)
        src = torch.load(ROOT / cfg["init_ckpt"], map_location="cpu")
        missing, unexpected = model.load_state_dict(src["model"], strict=False)
        print(f"[train] warm start from {cfg['init_ckpt']} (missing={missing}, unexpected={unexpected})", flush=True)
    print(f"[train] {name}: params={count_params(model)} device={device}", flush=True)
    # optional exponential moving average of the weights (incl. BatchNorm buffers); validation,
    # best.pt and the exported model then use the EMA weights
    ema = None
    if cfg.get("ema_decay"):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]), use_buffers=True)
    stop_step = cfg.get("stop_step", cfg["max_steps"])

    step, best = 0, -1e9
    if args.resume and (ckdir / "last.pt").exists():
        ck = torch.load(ckdir / "last.pt", map_location=device)
        model.load_state_dict(ck.get("model_raw", ck["model"]))
        if ema is not None and "model" in ck:
            restore_ema(ema, ck["model"], ck["step"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", -1e9)
        print(f"[train] resumed at step {step}", flush=True)

    ds = MixDataset("train", cfg["segment_s"], size=(cfg["max_steps"] - step) * cfg["batch_size"],
                    seed=cfg.get("seed", 0) * 1000 + step, use_synth=cfg.get("use_synth", True),
                    speech_manifest=cfg.get("speech_manifest"), mix_opts=cfg.get("mix_opts"))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg.get("workers", 6), drop_last=True,
                    persistent_workers=False, prefetch_factor=4)
    vn, vc, vcat = build_val_set(cfg.get("val_n", 200), use_synth=cfg.get("use_synth", True))
    pool = ProcessPoolExecutor(max_workers=cfg.get("val_workers", 6))

    t_start = time.time()
    max_s = cfg.get("max_hours", 1e9) * 3600
    model.train()
    agg, n_agg = {}, 0
    t_log = time.time()
    for noisy, clean, cat in dl:
        if step >= min(cfg["max_steps"], stop_step) or time.time() - t_start > max_s:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        noisy, clean = noisy.to(device), clean.to(device)
        X = torch.view_as_real(stft(noisy))
        S = torch.view_as_real(stft(clean))
        Y, spp = model.enhance_spec(X)
        y = stft.inverse(torch.view_as_complex(Y.contiguous()), noisy.shape[-1])
        loss, parts = lossf(Y, y, spp, S, clean, X)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.0))
        if not torch.isfinite(loss):
            print("[train] non-finite loss, skipping step", flush=True)
            continue
        opt.step()
        if ema is not None:
            ema.update_parameters(model)
        step += 1
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
        agg["gnorm"] = agg.get("gnorm", 0.0) + float(gn)
        n_agg += 1
        if step % cfg.get("log_every", 100) == 0:
            rec = {"step": step, "lr": lr_at(step, cfg), "elapsed_h": (time.time() - t_start) / 3600,
                   "s_per_step": (time.time() - t_log) / n_agg, **{k: v / n_agg for k, v in agg.items()}}
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            logf.write(json.dumps({"type": "train", **rec}) + "\n")
            logf.flush()
            agg, n_agg, t_log = {}, 0, time.time()
        if step % cfg.get("val_every", 1000) == 0 or step == cfg["max_steps"]:
            v = validate(ema.module if ema is not None else model, stft, vn, vc, vcat, device, pool)
            score = v["pesq"] + 2 * v["stoi"]
            v.update({"step": step, "score": score})
            print("[val]", json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()}), flush=True)
            logf.write(json.dumps({"type": "val", **v}) + "\n")
            logf.flush()
            ck = {"model": (ema.module if ema is not None else model).state_dict(), "opt": opt.state_dict(),
                  "step": step, "best": max(best, score), "model_cfg": mcfg.to_dict(), "train_cfg": cfg, "val": v}
            if ema is not None:
                ck["model_raw"] = model.state_dict()
            torch.save(ck, ckdir / "last.pt")
            if score > best:
                best = score
                torch.save(ck, ckdir / "best.pt")
                print(f"[val] new best {best:.4f}", flush=True)
    # final validation/checkpoint
    final_model = ema.module if ema is not None else model
    v = validate(final_model, stft, vn, vc, vcat, device, pool)
    score = v["pesq"] + 2 * v["stoi"]
    v.update({"step": step, "score": score, "final": True})
    logf.write(json.dumps({"type": "val", **v}) + "\n")
    ck = {"model": final_model.state_dict(), "opt": opt.state_dict(), "step": step, "best": max(best, score),
          "model_cfg": mcfg.to_dict(), "train_cfg": cfg, "val": v}
    if ema is not None:
        ck["model_raw"] = model.state_dict()
    torch.save(ck, ckdir / "last.pt")
    if score > best:
        torch.save(ck, ckdir / "best.pt")
    print("[train] done", json.dumps(v), flush=True)
    pool.shutdown()


if __name__ == "__main__":
    main()
