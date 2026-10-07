"""Export the single-frame streaming DANCNet to ONNX and verify it against PyTorch.

    python -m danc.inference.export_onnx --ckpt checkpoints/main/best.pt --out exports/dancnet_stream.onnx

ONNX contract (consumed by danc.inference.runners and deploy/jetson/trt_runner.py):
  inputs : spec [1,1,161,2] float32 (re/im of the current STFT frame, 20 ms sqrt-Hann, 10 ms hop)
           state_in_0 .. state_in_{K-1} float32 (static shapes, ALL-ZERO initialisation is valid)
  outputs: spec_out [1,1,161,2], spp [1,1,161], state_out_0 .. state_out_{K-1}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from danc.models.dancnet import DANCNet, DANCNetConfig


class StepWrapper(torch.nn.Module):
    def __init__(self, model: DANCNet):
        super().__init__()
        self.m = model

    def forward(self, spec, *states):
        y, spp, new = self.m(spec, list(states))
        return (y, spp, *new)


class StepWrapper2(torch.nn.Module):
    """Two-microphone model: inputs (spec, spec_ref, *states)."""

    def __init__(self, model: DANCNet):
        super().__init__()
        self.m = model

    def forward(self, spec, spec_ref, *states):
        y, spp, new = self.m(spec, list(states), spec_ref)
        return (y, spp, *new)


def load_model(ckpt: str | None, device="cpu") -> DANCNet:
    if ckpt is None:
        m = DANCNet()
    else:
        ck = torch.load(ckpt, map_location=device)
        m = DANCNet(DANCNetConfig(**ck["model_cfg"]))
        m.load_state_dict(ck["model"])
    return m.eval().to(device)


def export(model: DANCNet, out: str, opset: int = 17) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    two = model.cfg.in_ch == 2
    w = (StepWrapper2(model) if two else StepWrapper(model)).eval()
    states = model.initial_states(1)
    spec = torch.zeros(1, 1, model.cfg.n_bins, 2)
    lead = [spec, torch.zeros_like(spec)] if two else [spec]
    in_names = (["spec", "spec_ref"] if two else ["spec"]) + [f"state_in_{i}" for i in range(len(states))]
    out_names = ["spec_out", "spp"] + [f"state_out_{i}" for i in range(len(states))]
    with torch.no_grad():
        torch.onnx.export(w, (*lead, *states), str(out), input_names=in_names, output_names=out_names,
                          opset_version=opset, dynamo=False, do_constant_folding=True)
    _add_metadata(out, model)
    return {"inputs": {n: list(t.shape) for n, t in zip(in_names, [*lead, *states])}, "outputs": out_names}


def _add_metadata(path: Path, model: DANCNet) -> None:
    """Store the streaming contract (fs, window, hop, look-ahead, model config) in the ONNX file."""
    import onnx

    from danc.dsp import DEFAULT

    m = onnx.load(str(path))
    meta = {"fs": DEFAULT.fs, "win": DEFAULT.win, "hop": DEFAULT.hop, "n_fft": DEFAULT.n_fft,
            "lookahead": model.cfg.lookahead, "in_ch": model.cfg.in_ch,
            "algorithmic_latency_ms": DEFAULT.latency_ms + 1000.0 * DEFAULT.hop / DEFAULT.fs * model.cfg.lookahead,
            "model_cfg": json.dumps(model.cfg.to_dict())}
    del m.metadata_props[:]
    for k, v in meta.items():
        e = m.metadata_props.add()
        e.key, e.value = k, str(v)
    onnx.save(m, str(path))


def verify(model: DANCNet, onnx_path: str, n_frames: int = 300, seed: int = 0) -> dict:
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n_frames, 1, 1, model.cfg.n_bins, 2)) * 0.05).astype(np.float32)
    x[n_frames // 2] *= 200  # impulse
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    two = model.cfg.in_ch == 2
    st_np = [np.zeros(i.shape, np.float32) for i in sess.get_inputs()[(2 if two else 1):]]
    xr = (rng.standard_normal(x.shape) * 0.05).astype(np.float32)
    st_t = model.initial_states(1)
    max_err, max_ref = 0.0, 0.0
    t_ort = []
    with torch.no_grad():
        for t in range(n_frames):
            feeds = {"spec": x[t], **({"spec_ref": xr[t]} if two else {}), **{f"state_in_{i}": s for i, s in enumerate(st_np)}}
            t0 = time.perf_counter()
            outs = sess.run(None, feeds)
            t_ort.append(time.perf_counter() - t0)
            y_o, st_np = outs[0], outs[2:]
            y_t, _, st_t = (model(torch.from_numpy(x[t]), st_t, torch.from_numpy(xr[t])) if two
                            else model(torch.from_numpy(x[t]), st_t))
            max_err = max(max_err, float(np.abs(y_o - y_t.numpy()).max()))
            max_ref = max(max_ref, float(np.abs(y_t.numpy()).max()))
    t_ort = np.array(t_ort[10:]) * 1000
    return {"max_abs_err": max_err, "max_abs_ref": max_ref, "ort_ms_mean": float(t_ort.mean()),
            "ort_ms_p99": float(np.percentile(t_ort, 99))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="exports/dancnet_stream.onnx")
    a = ap.parse_args(argv)
    model = load_model(a.ckpt)
    info = export(model, a.out)
    v = verify(model, a.out)
    info.update(v)
    print(json.dumps(info, indent=1))
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
