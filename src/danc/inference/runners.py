"""Stateful single-frame back-ends with one common interface:

    runner.reset()
    spec_out, spp = runner(spec)     # spec [1,1,161,2] float32 -> ([1,1,161,2], [161])

TorchStepRunner - PyTorch eager (reference implementation)
ORTStepRunner   - ONNX Runtime (CPU EP by default; CUDA/TensorRT EPs on Jetson)
The TensorRT runner for Jetson lives in deploy/jetson/trt_runner.py (same interface).
"""
from __future__ import annotations

import numpy as np


class TorchStepRunner:
    """PyTorch reference runner (development machine only; torch is imported lazily so that the ONNX Runtime
    path below works on a torch-free Jetson install)."""

    def __init__(self, model, threads: int = 1):
        import torch
        torch.set_num_threads(threads)
        self._torch = torch
        self.model = model.eval()
        self.lookahead = int(model.cfg.lookahead)
        self.n_mics = int(getattr(model.cfg, "in_ch", 1))
        self.reset()

    def reset(self):
        self.states = self.model.initial_states(1)

    def __call__(self, spec: np.ndarray, ref: np.ndarray | None = None):
        torch = self._torch
        with torch.no_grad():
            if self.n_mics == 2:
                y, spp, self.states = self.model(torch.from_numpy(spec), self.states, torch.from_numpy(ref))
            else:
                y, spp, self.states = self.model(torch.from_numpy(spec), self.states)
        return y.numpy(), spp.numpy().reshape(-1)


class ORTStepRunner:
    def __init__(self, onnx_path: str, threads: int = 1, providers=None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so, providers=providers or ["CPUExecutionProvider"])
        self.state_inputs = [i for i in self.sess.get_inputs() if i.name.startswith("state_in_")]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.lookahead = int(meta.get("lookahead", 0))   # written by danc.inference.export_onnx
        self.n_mics = 2 if any(i.name == "spec_ref" for i in self.sess.get_inputs()) else 1
        self.reset()

    def reset(self):
        self.states = {i.name: np.zeros(i.shape, np.float32) for i in self.state_inputs}

    def __call__(self, spec: np.ndarray, ref: np.ndarray | None = None):
        feeds = {"spec": spec.astype(np.float32, copy=False), **self.states}
        if self.n_mics == 2:
            feeds["spec_ref"] = ref.astype(np.float32, copy=False)
        outs = self.sess.run(self.out_names, feeds)
        d = dict(zip(self.out_names, outs))
        for k in self.states:
            self.states[k] = d[k.replace("state_in_", "state_out_")]
        return d["spec_out"], d["spp"].reshape(-1)
