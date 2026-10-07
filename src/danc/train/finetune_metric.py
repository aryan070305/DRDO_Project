"""Perceptual (metric-guided) fine-tuning, MetricGAN+-style.

    python -m danc.train.finetune_metric --config configs/finetune_v2_metric.yaml

Idea (Fu et al., "MetricGAN: Generative adversarial networks based black-box metric scores optimization
for speech enhancement", ICML 2019; "MetricGAN+", Interspeech 2021): a discriminator D learns to predict
the *true* (non-differentiable) PESQ of an enhanced/clean pair; the enhancer is then pushed towards
D(enhanced, clean) = 1 (= maximum PESQ).  Unlike pure MetricGAN training we keep the full D-ANC loss
(compressed spectral + MR-STFT + SI-SDR + PMSQE + over-suppression penalty) as an anchor, so the model
cannot trade intelligibility (STOI) or SNR for PESQ.  The fine-tuned model is only adopted if it
improves the validation score PESQ + 2 STOI without lowering SI-SDR (decided in the report).

Labels: PESQ-WB normalised to [0, 1] as (pesq - 1.04) / (4.64 - 1.04), computed on CPU in a process pool
for (enhanced, clean) and (noisy, clean); (clean, clean) is labelled 1.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader

from danc.data.mixer import MixDataset
from danc.dsp import TorchSTFT
from danc.models.dancnet import DANCNet, DANCNetConfig
from danc.train.losses import DANCLoss
from danc.train.train import build_val_set, validate

ROOT = Path(__file__).resolve().parents[3]
P_MIN, P_MAX = 1.04, 4.64


def _pesq_norm(args):
    from pesq import pesq

    ref, est = args
    try:
        v = pesq(16000, ref.astype(np.float64), est.astype(np.float64), "wb")
    except Exception:
        v = P_MIN
    return float(np.clip((v - P_MIN) / (P_MAX - P_MIN), 0, 1))


class MetricDiscriminator(nn.Module):
    """MetricGAN+-like CNN on compressed magnitude spectrograms of (estimate, reference)."""

    def __init__(self, ch=16):
        super().__init__()
        layers, cin = [], 2
        for _ in range(4):
            layers += [spectral_norm(nn.Conv2d(cin, ch, 5, padding=2)), nn.LeakyReLU(0.3)]
            cin = ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(spectral_norm(nn.Linear(ch, 50)), nn.LeakyReLU(0.3),
                                spectral_norm(nn.Linear(50, 10)), nn.LeakyReLU(0.3), spectral_norm(nn.Linear(10, 1)))

    def forward(self, est_mag, ref_mag):
        x = torch.stack([est_mag, ref_mag], 1) ** 0.3          # [B,2,T,F]
        h = self.conv(x).mean(dim=(2, 3))
        return torch.sigmoid(self.fc(h)).squeeze(-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    name = cfg["name"]
    ckdir = ROOT / "checkpoints" / name
    ckdir.mkdir(parents=True, exist_ok=True)
    logf = open(ROOT / "logs" / f"train_{name}.jsonl", "a")
    yaml.safe_dump(cfg, open(ckdir / "config.yaml", "w"))
    device = torch.device(cfg.get("device", "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(cfg.get("seed", 0))

    src = torch.load(ROOT / cfg["init_ckpt"], map_location="cpu")
    mcfg = DANCNetConfig(**src["model_cfg"])
    G = DANCNet(mcfg)
    G.load_state_dict(src["model"])                  # EMA weights of the base run
    G.to(device).train()
    D = MetricDiscriminator().to(device)
    stft = TorchSTFT().to(device)
    lossf = DANCLoss(**cfg.get("loss", {})).to(device)
    optG = torch.optim.AdamW(G.parameters(), lr=cfg["lr_g"], weight_decay=1e-4)
    optD = torch.optim.AdamW(D.parameters(), lr=cfg["lr_d"])
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema = AveragedModel(G, multi_avg_fn=get_ema_multi_avg_fn(cfg.get("ema_decay", 0.999)), use_buffers=True)

    ds = MixDataset("train", cfg["segment_s"], size=cfg["steps"] * cfg["batch_size"], seed=cfg.get("seed", 0),
                    use_synth=True, speech_manifest=cfg.get("speech_manifest"), mix_opts=cfg.get("mix_opts"))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg.get("workers", 4), drop_last=True,
                    prefetch_factor=4)
    vn, vc, vcat = build_val_set(cfg.get("val_n", 200))
    pool = ProcessPoolExecutor(max_workers=cfg.get("pesq_workers", 4))
    v0 = validate(ema.module, stft, vn, vc, vcat, device, pool)
    best = v0["pesq"] + 2 * v0["stoi"]
    print("[ft] start", json.dumps({k: round(x, 4) for k, x in v0.items()}), flush=True)
    logf.write(json.dumps({"type": "val", "step": 0, **v0}) + "\n")
    step, t0, agg, n = 0, time.time(), {}, 0
    for noisy, clean, _ in dl:
        if step >= cfg["steps"]:
            break
        noisy, clean = noisy.to(device), clean.to(device)
        X = torch.view_as_real(stft(noisy))
        S = torch.view_as_real(stft(clean))
        Y, spp = G.enhance_spec(X)
        y = stft.inverse(torch.view_as_complex(Y.contiguous()), noisy.shape[-1])
        # --- true PESQ labels (CPU)
        y_np, s_np, x_np = y.detach().cpu().numpy(), clean.cpu().numpy(), noisy.cpu().numpy()
        valid = [i for i in range(len(s_np)) if np.mean(s_np[i] ** 2) > 1e-7]
        q_est = list(pool.map(_pesq_norm, [(s_np[i], y_np[i]) for i in valid]))
        q_noi = list(pool.map(_pesq_norm, [(s_np[i], x_np[i]) for i in valid]))
        vi = torch.tensor(valid, device=device)
        mag = lambda z: torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + 1e-8)  # noqa: E731
        Ym, Sm, Xm = mag(Y)[vi], mag(S)[vi], mag(X)[vi]
        # --- discriminator step
        dl_ = ((D(Ym.detach(), Sm) - torch.tensor(q_est, device=device)) ** 2).mean() \
            + ((D(Xm, Sm) - torch.tensor(q_noi, device=device)) ** 2).mean() \
            + ((D(Sm, Sm) - 1.0) ** 2).mean()
        optD.zero_grad(set_to_none=True)
        dl_.backward()
        optD.step()
        # --- generator step: anchor loss + metric adversarial term
        base, parts = lossf(Y, y, spp, S, clean, X)
        for p in D.parameters():
            p.requires_grad_(False)
        adv = ((D(Ym, Sm) - 1.0) ** 2).mean()
        for p in D.parameters():
            p.requires_grad_(True)
        loss = base + cfg["w_adv"] * adv
        optG.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 3.0)
        optG.step()
        ema.update_parameters(G)
        step += 1
        for k, v in {**parts, "adv": float(adv), "d_loss": float(dl_), "pesq_batch": float(np.mean(q_est)) * (P_MAX - P_MIN) + P_MIN}.items():
            agg[k] = agg.get(k, 0) + v
        n += 1
        if step % cfg.get("log_every", 50) == 0:
            rec = {"step": step, "s_per_step": (time.time() - t0) / step, **{k: v / n for k, v in agg.items()}}
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            logf.write(json.dumps({"type": "train", **rec}) + "\n")
            logf.flush()
            agg, n = {}, 0
        if step % cfg.get("val_every", 500) == 0 or step == cfg["steps"]:
            v = validate(ema.module, stft, vn, vc, vcat, device, pool)
            score = v["pesq"] + 2 * v["stoi"]
            v.update({"step": step, "score": score})
            print("[val]", json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()}), flush=True)
            logf.write(json.dumps({"type": "val", **v}) + "\n")
            logf.flush()
            ck = {"model": ema.module.state_dict(), "model_raw": G.state_dict(), "disc": D.state_dict(),
                  "step": step, "model_cfg": mcfg.to_dict(), "train_cfg": cfg, "val": v, "init_val": v0}
            torch.save(ck, ckdir / "last.pt")
            if score > best:
                best = score
                torch.save(ck, ckdir / "best.pt")
                print(f"[val] new best {best:.4f}", flush=True)
    pool.shutdown()
    print("[ft] done", flush=True)


if __name__ == "__main__":
    main()
