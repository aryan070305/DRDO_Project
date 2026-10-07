"""Training losses for D-ANC.

Total loss (weights in configs/train_*.yaml):

  L = w_spec * L_cspec        power-law compressed complex + magnitude MSE (c = 0.3),
                              "30 * (re/im) + 70 * mag" weighting as in GTCRN's HybridLoss
                              (Rong et al., ICASSP 2024; compression from Ephrat/Wilson et al. 2018, DPCRN)
    + w_mr   * L_mrstft       multi-resolution STFT loss on compressed magnitudes
                              (n_fft 256/512/1024) - temporal/spectral detail at several resolutions;
                              short windows matter for impulses (Yamamoto et al. 2020 PWG-style MR loss)
    + w_sisdr* L_sisdr        negative SI-SDR (Le Roux et al., ICASSP 2019), in Bels (dB/10)
    + w_pmsqe* L_pmsqe        PMSQE perceptual loss (Martin-Donas et al., IEEE SPL 2018): a
                              differentiable approximation of PESQ's symmetric + asymmetric
                              loudness disturbances (vendored, MIT)
    + w_asym * L_asym         asymmetric over-suppression penalty: only penalises target
                              compressed magnitude *above* the estimate (speech removed),
                              protecting intelligibility/STOI under very low SNR
    + w_spp  * L_spp          BCE between the SPP head and an IRM-style speech-presence target
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from danc.dsp import sqrt_hann
from danc.train.third_party.pmsqe import SingleSrcPMSQE

EPS = 1e-8


def _mag(z):  # z [..., 2]
    return torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + EPS)


def cspec_loss(est, ref, c=0.3):
    """est/ref: [B,T,F,2] spectra."""
    me, mr = _mag(est), _mag(ref)
    ce = est * (me ** (c - 1)).unsqueeze(-1)
    cr = ref * (mr ** (c - 1)).unsqueeze(-1)
    l_ri = F.mse_loss(ce, cr)
    l_mag = F.mse_loss(me ** c, mr ** c)
    return 30 * l_ri + 70 * l_mag


def asym_loss(est, ref, c=0.3):
    me, mr = _mag(est) ** c, _mag(ref) ** c
    return 100 * torch.mean(F.relu(mr - me) ** 2)


class MRSTFTLoss(torch.nn.Module):
    def __init__(self, ffts=(256, 512, 1024), c=0.3):
        super().__init__()
        self.ffts, self.c = ffts, c
        for n in ffts:
            self.register_buffer(f"w{n}", torch.hann_window(n), persistent=False)

    def forward(self, y, s):
        loss = 0.0
        for n in self.ffts:
            w = getattr(self, f"w{n}")
            Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
            S = torch.stft(s, n, n // 4, window=w, return_complex=True).abs()
            Yc, Sc = (Y + EPS) ** self.c, (S + EPS) ** self.c
            sc = torch.norm(Sc - Yc, p="fro") / (torch.norm(Sc, p="fro") + EPS)
            loss = loss + sc + F.l1_loss(Yc, Sc)
        return loss / len(self.ffts)


def si_sdr(est, ref, eps=EPS):
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / ((ref ** 2).sum(-1, keepdim=True) + eps)
    t = a * ref
    return 10 * torch.log10((t ** 2).sum(-1) / (((est - t) ** 2).sum(-1) + eps) + eps)


def sisdr_loss(est, ref):
    """-SI-SDR in Bels, ignoring (near-)silent targets (noise-only examples)."""
    valid = (ref ** 2).mean(-1) > 1e-7
    if valid.sum() == 0:
        return est.new_zeros(())
    return -(si_sdr(est[valid], ref[valid]) / 10).mean()


class PMSQELoss(torch.nn.Module):
    """PMSQE on 32 ms / 16 ms sqrt-Hann frames (the configuration it was designed for)."""

    def __init__(self):
        super().__init__()
        self.pmsqe = SingleSrcPMSQE(window_name="sqrt_hann", sample_rate=16000)
        self.register_buffer("win", torch.from_numpy(sqrt_hann(512)).float(), persistent=False)

    def forward(self, y, s):
        valid = (s ** 2).mean(-1) > 1e-7
        if valid.sum() == 0:
            return y.new_zeros(())
        y, s = y[valid], s[valid]
        Y = torch.stft(y, 512, 256, window=self.win, center=False, return_complex=True)
        S = torch.stft(s, 512, 256, window=self.win, center=False, return_complex=True)
        Yp = (Y.real ** 2 + Y.imag ** 2).transpose(1, 2)   # B T F
        Sp = (S.real ** 2 + S.imag ** 2).transpose(1, 2)
        # pad mask: frames where the reference is not digital silence
        pad = ((Sp.sum(-1, keepdim=True) > 1e-10).float())
        pad[:, :1] = 1.0
        return self.pmsqe(Yp, Sp, pad_mask=pad).mean()


def spp_target(clean_spec, noisy_spec):
    """IRM-style speech presence: |S|^2 / (|S|^2 + |N|^2), N = noisy - clean."""
    ps = (clean_spec ** 2).sum(-1)
    pn = ((noisy_spec - clean_spec) ** 2).sum(-1)
    return ps / (ps + pn + EPS)


class DANCLoss(torch.nn.Module):
    def __init__(self, w_spec=1.0, w_mr=1.0, w_sisdr=1.0, w_pmsqe=0.0, w_asym=0.5, w_spp=0.1, c=0.3):
        super().__init__()
        self.w = dict(spec=w_spec, mr=w_mr, sisdr=w_sisdr, pmsqe=w_pmsqe, asym=w_asym, spp=w_spp)
        self.c = c
        self.mr = MRSTFTLoss()
        self.pmsqe = PMSQELoss() if w_pmsqe > 0 else None

    def forward(self, est_spec, est_wav, spp, clean_spec, clean_wav, noisy_spec):
        parts = {}
        parts["spec"] = cspec_loss(est_spec, clean_spec, self.c)
        parts["asym"] = asym_loss(est_spec, clean_spec, self.c)
        parts["mr"] = self.mr(est_wav, clean_wav)
        parts["sisdr"] = sisdr_loss(est_wav, clean_wav)
        tgt = spp_target(clean_spec, noisy_spec).clamp(0, 1)
        parts["spp"] = F.binary_cross_entropy(spp.clamp(1e-5, 1 - 1e-5), tgt)
        if self.pmsqe is not None:
            parts["pmsqe"] = self.pmsqe(est_wav, clean_wav)
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}
