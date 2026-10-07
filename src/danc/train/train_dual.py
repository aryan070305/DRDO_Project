"""Training of the two-microphone DANCNet (primary + reference mic as network inputs).

    python -m danc.train.train_dual --config configs/train_v3_2mic.yaml

Warm start (`init_ckpt`): every weight is copied from a single-mic model; the first encoder conv gets the
single-mic filters for the primary-mic features and ZERO filters for the reference-mic features, so training
starts from exactly the single-mic model and only has to learn what the reference adds.
Validation: the dual-mic validation scenes data/testsets/dualmic_val (validation speakers, training-split noise).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader

from danc.data.dual_mixer import DualMixDataset
from danc.dsp import TorchSTFT
from danc.models.dancnet import DANCNet, DANCNetConfig, count_params
from danc.train.losses import DANCLoss
from danc.train.train import _metrics_one, lr_at, restore_ema

ROOT = Path(__file__).resolve().parents[3]


def load_dual_val(name="dualmic_val"):
    d = ROOT / "data" / "testsets" / name
    rows = list(csv.DictReader(open(d / "meta.csv")))
    rd = lambda p: sf.read(p, dtype="float32")[0]  # noqa: E731
    return ([rd(d / "primary" / f"{r['id']}.flac") for r in rows], [rd(d / "reference" / f"{r['id']}.flac") for r in rows],
            [rd(d / "clean" / f"{r['id']}.flac") for r in rows], [r["category"] for r in rows])


@torch.no_grad()
def validate_dual(model, stft, prim, ref, clean, device, pool):
    model.eval()
    ests = []
    for p, r in zip(prim, ref):
        P = torch.view_as_real(stft(torch.from_numpy(p)[None].to(device)))
        R = torch.view_as_real(stft(torch.from_numpy(r)[None].to(device)))
        Y, _ = model.enhance_spec(P, R)
        ests.append(stft.inverse(torch.view_as_complex(Y.contiguous()), len(p))[0].cpu().numpy())
    res = list(pool.map(_metrics_one, [(e.astype(np.float64), c.astype(np.float64)) for e, c in zip(ests, clean)]))
    model.train()
    return {k: float(np.nanmean([x[k] for x in res])) for k in res[0]}


def warm_start(model: DANCNet, ckpt: str):
    src = torch.load(ROOT / ckpt, map_location="cpu")["model"]
    own = model.state_dict()
    for k, v in src.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif k == "enc1.conv.conv.weight" and own[k].shape[1] == 2 * v.shape[1]:
            w = torch.zeros_like(own[k])
            w[:, : v.shape[1]] = v                       # primary features: single-mic filters
            own[k] = w                                   # reference features: start at zero
        else:
            raise ValueError(f"cannot warm-start {k}: {tuple(v.shape)} -> {tuple(own[k].shape)}")
    model.load_state_dict(own)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>/last.pt (weights, EMA, step)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    name = cfg["name"]
    ckdir = ROOT / "checkpoints" / name
    ckdir.mkdir(parents=True, exist_ok=True)
    logf = open(ROOT / "logs" / f"train_{name}.jsonl", "a")
    yaml.safe_dump(cfg, open(ckdir / "config.yaml", "w"))
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "mps" if torch.backends.mps.is_available() else "cpu"))
    mcfg = DANCNetConfig(**{**cfg["model"], "in_ch": 2})
    model = DANCNet(mcfg)
    start_step, resume_ck = 0, None
    if a.resume and not (ckdir / "last.pt").exists():
        print(f"[dual] WARNING: --resume but {ckdir / 'last.pt'} does not exist - starting a new run", flush=True)
    if a.resume and (ckdir / "last.pt").exists():
        resume_ck = torch.load(ckdir / "last.pt", map_location="cpu")
        model.load_state_dict(resume_ck["model_raw"])
        start_step = int(resume_ck["step"])
        has_opt = "opt" in resume_ck
        print(f"[dual] resumed from {ckdir / 'last.pt'} at step {start_step} "
              f"({'optimizer state restored' if has_opt else 'optimizer state re-initialised, LR re-warm-up'})", flush=True)
    elif cfg.get("init_ckpt"):
        warm_start(model, cfg["init_ckpt"])
        print(f"[dual] warm start from {cfg['init_ckpt']}", flush=True)
    model.to(device).train()
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.get("ema_decay", 0.998)), use_buffers=True)
    if resume_ck is not None:
        restore_ema(ema, resume_ck["model"], start_step)   # keeps the EMA (AveragedModel would copy on 1st update)
    stft = TorchSTFT().to(device)
    lossf = DANCLoss(**cfg.get("loss", {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    rewarm = 0          # fresh AdamW state on a resume from a checkpoint without it: short linear LR re-warm-up
    if resume_ck is not None:
        if "opt" in resume_ck:
            opt.load_state_dict(resume_ck["opt"])
        else:
            rewarm = int(cfg.get("resume_warmup_steps", 100))
    print(f"[dual] {name}: params={count_params(model)} device={device}", flush=True)
    ds = DualMixDataset(cfg["segment_s"], size=(cfg["max_steps"] - start_step) * cfg["batch_size"],
                        seed=cfg.get("seed", 0) * 1000 + start_step,
                        speech_manifest=cfg.get("speech_manifest"), mix_opts=cfg.get("mix_opts"))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg.get("workers", 4), drop_last=True, prefetch_factor=4)
    vp, vr, vc, _ = load_dual_val()
    pool = ProcessPoolExecutor(max_workers=cfg.get("val_workers", 3))
    v0 = validate_dual(ema.module, stft, vp, vr, vc, device, pool)
    best = v0["pesq"] + 2 * v0["stoi"]
    if resume_ck is not None and (ckdir / "best.pt").exists():
        b = torch.load(ckdir / "best.pt", map_location="cpu")["val"]
        best = max(best, b["pesq"] + 2 * b["stoi"])
    print("[val] start", json.dumps({k: round(x, 4) for k, x in v0.items()}), flush=True)
    logf.write(json.dumps({"type": "val", "step": start_step, **v0}) + "\n")
    step, t0, agg, n = start_step, time.time(), {}, 0
    max_s = cfg.get("max_hours", 1e9) * 3600   # wall-clock budget of THIS invocation (restarts on --resume)
    last_val = start_step

    def validate_and_save():
        nonlocal best, last_val
        v = validate_dual(ema.module, stft, vp, vr, vc, device, pool)
        score = v["pesq"] + 2 * v["stoi"]
        v.update({"step": step, "score": score})
        print("[val]", json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()}), flush=True)
        logf.write(json.dumps({"type": "val", **v}) + "\n")
        logf.flush()
        ck = {"model": ema.module.state_dict(), "model_raw": model.state_dict(), "opt": opt.state_dict(), "step": step,
              "model_cfg": mcfg.to_dict(), "train_cfg": cfg, "val": v, "init_val": v0}
        torch.save(ck, ckdir / "last.pt")
        if score > best:
            best = score
            torch.save(ck, ckdir / "best.pt")
            print(f"[val] new best {best:.4f}", flush=True)
        last_val = step
    for prim, ref, clean, _ in dl:
        if step >= cfg["max_steps"] or time.time() - t0 > max_s:
            break
        lr = lr_at(step, cfg) * (min(1.0, (step - start_step + 1) / rewarm) if rewarm else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        prim, ref, clean = prim.to(device), ref.to(device), clean.to(device)
        P, R, S = (torch.view_as_real(stft(v)) for v in (prim, ref, clean))
        Y, spp = model.enhance_spec(P, R)
        y = stft.inverse(torch.view_as_complex(Y.contiguous()), prim.shape[-1])
        loss, parts = lossf(Y, y, spp, S, clean, P)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.0))
        if not torch.isfinite(loss):
            continue
        opt.step()
        ema.update_parameters(model)
        step += 1
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
        n += 1
        if step % cfg.get("log_every", 100) == 0:
            rec = {"step": step, "lr": lr, "elapsed_h": (time.time() - t0) / 3600,
                   "s_per_step": (time.time() - t0) / max(1, step - start_step), **{k: v / n for k, v in agg.items()}}
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            logf.write(json.dumps({"type": "train", **rec}) + "\n")
            logf.flush()
            agg, n = {}, 0
        if step % cfg.get("val_every", 1000) == 0 or step == cfg["max_steps"]:
            validate_and_save()
    if step > last_val:      # stopped by max_hours, or the data ran out after skipped non-finite batches
        validate_and_save()
    pool.shutdown()
    print("[dual] done", flush=True)


if __name__ == "__main__":
    main()
