#!/usr/bin/env python3
"""Single-frame streaming runners for the D-ANC network: TensorRT 10.x and ONNX Runtime.

    NOT VERIFIED ON HARDWARE IN THIS PROJECT
    ----------------------------------------
    The TensorRT / CUDA code paths in this file (``TRTStepRunner``, the
    cuda-python and torch memory back-ends, CUDA-graph capture) were written
    against the TensorRT 10.x and cuda-python 12.x Python APIs but have NOT been
    executed on a Jetson AGX Orin by this project (development happened on a
    Mac, where TensorRT is unavailable).  The pure-Python parts (I/O-contract
    validation, ``ORTStepRunner``, latency statistics, numerical comparison)
    are unit-tested on the development machine (tests/test_jetson_scripts.py).

Model contract (deployed single-mic models: exports/dancnet_v3hq_cont.onnx = HQ 40 ms, exports/dancnet_ll.onnx = LL 20 ms; the two-mic model is ORT-only)
---------------------------------------------------------------------------
    inputs   spec           float32 [1, 1, 161, 2]   real/imag of the current STFT frame
             state_in_k     float32, static shape    k = 0 .. K-1 (recurrent state)
    outputs  spec_out       float32 [1, 1, 161, 2]   enhanced frame
             spp            float32 [1, 1, 161]      speech-presence probability in [0, 1]
             state_out_k    same shape/dtype as state_in_k

The STFT framing is the one defined in ``danc.dsp`` (fs = 16 kHz, 20 ms
sqrt-Hann window, 10 ms hop, n_fft = 320 -> 161 bins).  One call of a runner
processes one 10 ms hop, so the per-call latency has to stay well below 10 ms
at the 99th (better 99.9th) percentile, not just on average.  Initial
recurrent states are assumed to be all-zero (``reset()``); if the exported
network needs a different initial state, that is a contract change.

Design notes
------------
* **State ping-pong.**  Each ``state_in_k``/``state_out_k`` pair gets two device
  buffers A and B.  On even frames the engine reads A and writes B, on odd
  frames it reads B and writes A, so the recurrent state never leaves the GPU
  and no device-to-device copy is needed.  It also guarantees that an input
  and an output binding never alias the same memory, which TensorRT does not
  allow in general.
* **Device-memory dependency.**  TensorRT's Python API does not allocate
  memory itself, so a CUDA binding is needed.  Two interchangeable back-ends
  are provided:

    - ``cudart`` (default): NVIDIA ``cuda-python``.  Small, no torch needed, maps
      1:1 to the CUDA runtime API (``cudaMallocHost``, ``cudaMemcpyAsync``,
      stream capture).  Install the version matching JetPack's CUDA, e.g.
      12.6.x for JetPack 6.2 (the Jetson AI Lab index ``jp6/cu126`` ships
      ``cuda-python 12.6.2.post1`` - checked 2026-10-03).
    - ``torch``: CUDA tensors / pinned CPU tensors from a Jetson build of PyTorch.
      Convenient when torch is installed anyway, but it is a ~GB dependency and
      its caching allocator / stream semantics add another layer.

* **Pinned host buffers.**  Host staging buffers are page-locked so that the
  ``cudaMemcpyAsync`` calls are truly asynchronous.  On Jetson the CPU and GPU
  share the same physical DRAM, so mapped zero-copy memory
  (``cudaHostAllocMapped``) could avoid the copies altogether; with 1.3 KB per
  frame the copies are not the bottleneck, so the portable pinned+copy path is
  used here.
* **CUDA graphs (optional).**  A batch-1, single-frame recurrent network is
  dominated by per-kernel launch overhead rather than arithmetic.  Capturing
  ``execute_async_v3`` into a CUDA graph replaces many launches by one.
  Because tensor addresses are baked into a graph, one graph is captured per
  ping-pong parity (two graphs total).  Capture follows the pattern in the
  TensorRT developer guide: call ``execute_async_v3`` once before capturing.
  If capture fails (unsupported layer/plugin), the runner falls back to plain
  enqueue and reports ``cuda_graph_active = False``.

Usage
-----
    # TensorRT engine micro-benchmark (on the Jetson)
    python trt_runner.py --engine exports/dancnet_v3hq_cont_fp16.engine --frames 3000 --cuda-graph
    # ONNX Runtime CPU baseline (works on any machine with onnxruntime)
    python trt_runner.py --onnx exports/dancnet_v3hq_cont.onnx --threads 1
    # FP16 engine vs FP32 ONNX numerical check on real audio (16 kHz mono wav)
    python trt_runner.py --engine ..._fp16.engine --onnx exports/dancnet_v3hq_cont.onnx --compare --wav noisy.wav
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------------
# I/O contract
# ----------------------------------------------------------------------------

SPEC_IN = "spec"
SPEC_OUT = "spec_out"
SPP_OUT = "spp"
STATE_IN_PREFIX = "state_in_"
STATE_OUT_PREFIX = "state_out_"

#: == danc.dsp.DEFAULT.n_bins (n_fft = 320).  Duplicated so this file also runs
#: standalone on the target without the danc package installed.
DEFAULT_N_BINS = 161
#: == 1000 * danc.dsp.DEFAULT.hop / danc.dsp.DEFAULT.fs: the real-time deadline per call.
HOP_MS = 10.0
#: Marker for a dimension whose size is only known at run time (outputs only).
UNKNOWN_DIM = -1


class ContractError(ValueError):
    """The network's I/O tensors do not match the D-ANC streaming contract."""


@dataclass(frozen=True)
class TensorSpec:
    """Name, shape, dtype and direction of one network I/O tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    is_input: bool

    @property
    def nbytes(self) -> int:
        if any(d < 0 for d in self.shape):
            raise ContractError(f"tensor '{self.name}' has no static shape: {self.shape}")
        return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class StreamContract:
    """Validated view of the streaming network's I/O (see module docstring)."""

    spec_in: TensorSpec
    spec_out: TensorSpec
    spp: TensorSpec
    states: tuple[tuple[TensorSpec, TensorSpec], ...]  # (state_in_k, state_out_k), k = 0..K-1

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_bins(self) -> int:
        return self.spec_in.shape[2]

    @property
    def state_bytes(self) -> int:
        return sum(s_in.nbytes for s_in, _ in self.states)

    def describe(self) -> str:
        lines = [f"  {t.name:<14s} {'in ' if t.is_input else 'out'} {np.dtype(t.dtype).name:<8s} {list(t.shape)}"
                 for t in (self.spec_in, self.spec_out, self.spp)]
        for s_in, s_out in self.states:
            lines.append(f"  {s_in.name:<14s} in  {np.dtype(s_in.dtype).name:<8s} {list(s_in.shape)}"
                         f"  <->  {s_out.name}")
        lines.append(f"  K = {self.n_states} state tensors, {self.state_bytes} bytes of recurrent state")
        return "\n".join(lines)


def resolve_shape(dims: Iterable[object], name: str, *, allow_unknown: bool = False) -> tuple[int, ...]:
    """Convert a framework shape (ints; -1 / None / str for dynamic dims) to a static tuple.

    The streaming contract is static.  A dynamic *leading* (batch) dimension is
    accepted and pinned to 1, because exporters often leave it symbolic.  Any
    other dynamic dimension is an error, unless ``allow_unknown`` is set (used
    for ONNX Runtime outputs whose shapes are only inferred at run time), in
    which case it is kept as ``UNKNOWN_DIM``.
    """
    out: list[int] = []
    for axis, d in enumerate(dims):
        if isinstance(d, (int, np.integer)) and not isinstance(d, bool) and d >= 0:
            out.append(int(d))
        elif axis == 0:
            out.append(1)
        elif allow_unknown:
            out.append(UNKNOWN_DIM)
        else:
            raise ContractError(f"tensor '{name}' has a dynamic dimension at axis {axis} ({d!r}); "
                                "the streaming contract requires static shapes")
    return tuple(out)


def _compatible(a: Sequence[int], b: Sequence[int]) -> bool:
    """Equal rank and equal sizes wherever both sizes are known."""
    return len(a) == len(b) and all(x == y or UNKNOWN_DIM in (x, y) for x, y in zip(a, b))


def parse_contract(tensors: Sequence[TensorSpec], expected_bins: int | None = DEFAULT_N_BINS) -> StreamContract:
    """Validate a list of I/O tensors against the D-ANC streaming contract.

    State tensors are discovered by name prefix (``state_in_<k>`` /
    ``state_out_<k>``); indices must be contiguous ``0..K-1`` and each pair must
    agree in shape and dtype.  Any tensor outside the contract is rejected so
    that no engine binding is ever left unset.
    """
    by_name: dict[str, TensorSpec] = {}
    for t in tensors:
        if t.name in by_name:
            raise ContractError(f"duplicate tensor name '{t.name}'")
        by_name[t.name] = t

    def get(name: str, is_input: bool) -> TensorSpec:
        t = by_name.get(name)
        kind = "input" if is_input else "output"
        if t is None:
            raise ContractError(f"missing {kind} tensor '{name}'; network has {sorted(by_name)}")
        if t.is_input != is_input:
            raise ContractError(f"tensor '{name}' must be an {kind}")
        return t

    spec_in, spec_out, spp = get(SPEC_IN, True), get(SPEC_OUT, False), get(SPP_OUT, False)
    s = spec_in.shape
    if len(s) != 4 or s[0] != 1 or s[1] != 1 or s[3] != 2:
        raise ContractError(f"'{SPEC_IN}' must have shape [1, 1, F, 2], got {list(s)}")
    n_bins = s[2]
    if expected_bins is not None and n_bins != expected_bins:
        raise ContractError(f"'{SPEC_IN}' has {n_bins} bins, expected {expected_bins} (danc.dsp STFT)")
    if not _compatible(spec_out.shape, spec_in.shape):
        raise ContractError(f"'{SPEC_OUT}' shape {list(spec_out.shape)} != '{SPEC_IN}' shape {list(s)}")
    if not _compatible(spp.shape, (1, 1, n_bins)):
        raise ContractError(f"'{SPP_OUT}' must have shape [1, 1, {n_bins}], got {list(spp.shape)}")

    state_in: dict[int, TensorSpec] = {}
    state_out: dict[int, TensorSpec] = {}
    for name, t in by_name.items():
        for prefix, store, want_input in ((STATE_IN_PREFIX, state_in, True), (STATE_OUT_PREFIX, state_out, False)):
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if not suffix.isdigit():
                    raise ContractError(f"state tensor '{name}' must end in an integer index")
                if t.is_input != want_input:
                    raise ContractError(f"state tensor '{name}' has the wrong direction")
                store[int(suffix)] = t
    known = {SPEC_IN, SPEC_OUT, SPP_OUT} | {t.name for t in (*state_in.values(), *state_out.values())}
    extra = sorted(set(by_name) - known)
    if extra:
        raise ContractError(f"tensors outside the streaming contract: {extra}")
    if set(state_in) != set(state_out):
        raise ContractError(f"unpaired state tensors: in={sorted(state_in)} out={sorted(state_out)}")
    k_total = len(state_in)
    if sorted(state_in) != list(range(k_total)):
        raise ContractError(f"state indices must be contiguous 0..{k_total - 1}, got {sorted(state_in)}")

    pairs = []
    for k in range(k_total):
        s_in, s_out = state_in[k], state_out[k]
        if not _compatible(s_in.shape, s_out.shape):
            raise ContractError(f"'{s_in.name}' {list(s_in.shape)} and '{s_out.name}' {list(s_out.shape)} differ")
        if np.dtype(s_in.dtype) != np.dtype(s_out.dtype):
            raise ContractError(f"'{s_in.name}' and '{s_out.name}' have different dtypes")
        pairs.append((s_in, s_out))
    return StreamContract(spec_in, spec_out, spp, tuple(pairs))


def complex_to_spec(frame: np.ndarray) -> np.ndarray:
    """(F,) complex STFT frame (e.g. ``StreamingSTFT.analyze(x)[0]``) -> float32 [1, 1, F, 2]."""
    frame = np.asarray(frame).reshape(-1)
    return np.stack([frame.real, frame.imag], axis=-1).astype(np.float32)[None, None]


def spec_to_complex(spec: np.ndarray) -> np.ndarray:
    """float [1, 1, F, 2] -> (F,) complex128, ready for ``StreamingSTFT.synthesize``."""
    spec = np.asarray(spec, dtype=np.float64)
    return spec[0, 0, :, 0] + 1j * spec[0, 0, :, 1]


def _check_frame(spec: np.ndarray, expected: TensorSpec) -> np.ndarray:
    arr = np.asarray(spec)
    if arr.shape != expected.shape:
        raise ValueError(f"'{expected.name}' must have shape {expected.shape}, got {arr.shape}")
    return arr


# ----------------------------------------------------------------------------
# CUDA memory back-ends (used by TRTStepRunner only)
# ----------------------------------------------------------------------------

@dataclass
class DeviceBuffer:
    """A device allocation; ``owner`` keeps the backing object alive (torch back-end)."""

    ptr: int
    nbytes: int
    owner: object = None


@dataclass
class HostBuffer:
    """A page-locked host buffer exposed as a numpy array."""

    array: np.ndarray
    owner: object = None

    @property
    def ptr(self) -> int:
        return int(self.array.ctypes.data)

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)


def _import_cudart():
    """Import the CUDA runtime bindings from cuda-python (module path changed in 12.8)."""
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.8
    except ImportError:
        from cuda import cudart  # cuda-python <= 12.7, e.g. 12.6.x for JetPack 6.2
    return cudart


def _as_handle(obj: object) -> int:
    """Raw integer handle of a cuda-python object (cudaStream_t, ...)."""
    try:
        return int(obj)  # type: ignore[arg-type]
    except TypeError:
        return int(obj.getPtr())  # type: ignore[attr-defined]


class CudartMemory:
    """Device/pinned-host memory, one CUDA stream and CUDA graphs via ``cuda-python``.

    All functions of the cuda-python runtime bindings return a tuple whose first
    element is a ``cudaError_t``; ``_check`` unwraps it.  NOT VERIFIED ON
    HARDWARE IN THIS PROJECT.
    """

    name = "cudart"

    def __init__(self) -> None:
        self.rt = _import_cudart()
        flags = getattr(self.rt, "cudaStreamNonBlocking", 1)  # no implicit sync with the legacy stream
        self._stream = self._check(self.rt.cudaStreamCreateWithFlags(flags), "cudaStreamCreateWithFlags")
        self._device_ptrs: list[int] = []
        self._host_ptrs: list[int] = []
        self._graph_execs: list[object] = []
        self._closed = False

    def _check(self, result: object, what: str):
        res = result if isinstance(result, tuple) else (result,)
        err, rest = res[0], res[1:]
        if err != self.rt.cudaError_t.cudaSuccess:
            try:
                msg = self.rt.cudaGetErrorString(err)[1].decode()
            except Exception:  # pragma: no cover - best-effort message only
                msg = repr(err)
            raise RuntimeError(f"{what} failed: {msg}")
        if not rest:
            return None
        return rest[0] if len(rest) == 1 else rest

    @property
    def stream_handle(self) -> int:
        return _as_handle(self._stream)

    def device_alloc(self, nbytes: int) -> DeviceBuffer:
        ptr = int(self._check(self.rt.cudaMalloc(nbytes), "cudaMalloc"))
        self._device_ptrs.append(ptr)
        return DeviceBuffer(ptr, nbytes)

    def host_alloc(self, shape: tuple[int, ...], dtype: np.dtype) -> HostBuffer:
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        ptr = int(self._check(self.rt.cudaMallocHost(nbytes), "cudaMallocHost"))
        self._host_ptrs.append(ptr)
        arr = np.frombuffer((ctypes.c_byte * nbytes).from_address(ptr), dtype=dtype).reshape(shape)
        arr[...] = 0
        return HostBuffer(arr, owner=ptr)

    def h2d(self, dst: DeviceBuffer, src: HostBuffer) -> None:
        kind = self.rt.cudaMemcpyKind.cudaMemcpyHostToDevice
        self._check(self.rt.cudaMemcpyAsync(dst.ptr, src.ptr, src.nbytes, kind, self._stream), "cudaMemcpyAsync(H2D)")

    def d2h(self, dst: HostBuffer, src: DeviceBuffer) -> None:
        kind = self.rt.cudaMemcpyKind.cudaMemcpyDeviceToHost
        self._check(self.rt.cudaMemcpyAsync(dst.ptr, src.ptr, dst.nbytes, kind, self._stream), "cudaMemcpyAsync(D2H)")

    def zero(self, buf: DeviceBuffer) -> None:
        self._check(self.rt.cudaMemsetAsync(buf.ptr, 0, buf.nbytes, self._stream), "cudaMemsetAsync")

    def synchronize(self) -> None:
        self._check(self.rt.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")

    def capture(self, enqueue: Callable[[], None]) -> object:
        """Record ``enqueue()`` (work on this stream) into an instantiated CUDA graph."""
        mode = self.rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
        self._check(self.rt.cudaStreamBeginCapture(self._stream, mode), "cudaStreamBeginCapture")
        try:
            enqueue()
        except BaseException:
            self.rt.cudaStreamEndCapture(self._stream)  # leave capture mode, discard the partial graph
            raise
        graph = self._check(self.rt.cudaStreamEndCapture(self._stream), "cudaStreamEndCapture")
        graph_exec = self._check(self.rt.cudaGraphInstantiate(graph, 0), "cudaGraphInstantiate")  # CUDA 12 signature
        self._check(self.rt.cudaGraphDestroy(graph), "cudaGraphDestroy")  # the executable graph is independent
        self._graph_execs.append(graph_exec)
        return graph_exec

    def launch(self, graph_exec: object) -> None:
        self._check(self.rt.cudaGraphLaunch(graph_exec, self._stream), "cudaGraphLaunch")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        rt = self.rt
        rt.cudaStreamSynchronize(self._stream)
        for g in self._graph_execs:
            rt.cudaGraphExecDestroy(g)
        for p in self._device_ptrs:
            rt.cudaFree(p)
        for p in self._host_ptrs:
            rt.cudaFreeHost(p)
        rt.cudaStreamDestroy(self._stream)
        self._graph_execs.clear()
        self._device_ptrs.clear()
        self._host_ptrs.clear()


class TorchMemory:
    """Same interface as :class:`CudartMemory`, backed by torch CUDA / pinned tensors.

    NOT VERIFIED ON HARDWARE IN THIS PROJECT.  Requires a CUDA-enabled PyTorch
    build for Jetson (e.g. from the Jetson AI Lab index); the generic PyPI
    aarch64 wheels are CPU-only.
    """

    name = "torch"

    def __init__(self, device: int = 0) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False (CPU-only torch build?)")
        self.torch = torch
        self.device = torch.device("cuda", device)
        self._stream = torch.cuda.Stream(device=self.device)
        self._owners: list[object] = []

    @property
    def stream_handle(self) -> int:
        return int(self._stream.cuda_stream)

    def device_alloc(self, nbytes: int) -> DeviceBuffer:
        t = self.torch.zeros(nbytes, dtype=self.torch.uint8, device=self.device)
        self._owners.append(t)
        return DeviceBuffer(int(t.data_ptr()), nbytes, owner=t)

    def host_alloc(self, shape: tuple[int, ...], dtype: np.dtype) -> HostBuffer:
        tdtype = self.torch.from_numpy(np.zeros(0, dtype=dtype)).dtype
        t = self.torch.zeros(shape, dtype=tdtype, pin_memory=True)
        self._owners.append(t)
        return HostBuffer(t.numpy(), owner=t)

    def _bytes(self, host: HostBuffer):
        return host.owner.reshape(-1).view(self.torch.uint8)  # type: ignore[union-attr]

    def h2d(self, dst: DeviceBuffer, src: HostBuffer) -> None:
        with self.torch.cuda.stream(self._stream):
            dst.owner.copy_(self._bytes(src), non_blocking=True)  # type: ignore[union-attr]

    def d2h(self, dst: HostBuffer, src: DeviceBuffer) -> None:
        with self.torch.cuda.stream(self._stream):
            self._bytes(dst).copy_(src.owner, non_blocking=True)

    def zero(self, buf: DeviceBuffer) -> None:
        with self.torch.cuda.stream(self._stream):
            buf.owner.zero_()  # type: ignore[union-attr]

    def synchronize(self) -> None:
        self._stream.synchronize()

    def capture(self, enqueue: Callable[[], None]) -> object:
        graph = self.torch.cuda.CUDAGraph()
        with self.torch.cuda.graph(graph, stream=self._stream):
            enqueue()
        self._owners.append(graph)
        return graph

    def launch(self, graph: object) -> None:
        with self.torch.cuda.stream(self._stream):
            graph.replay()  # type: ignore[attr-defined]

    def close(self) -> None:
        self._stream.synchronize()
        self._owners.clear()


def make_memory_backend(kind: str = "auto"):
    """``'cudart'`` | ``'torch'`` | ``'auto'`` (cuda-python first, then torch)."""
    builders = {"cudart": CudartMemory, "torch": TorchMemory}
    if kind != "auto":
        if kind not in builders:
            raise ValueError(f"unknown memory back-end '{kind}' (use auto, cudart or torch)")
        return builders[kind]()
    errors = []
    for name, build in builders.items():
        try:
            return build()
        except Exception as exc:  # ImportError, no CUDA device, ...
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no CUDA memory back-end available -> install cuda-python (recommended) "
                       "or a CUDA build of torch.\n  " + "\n  ".join(errors))


# ----------------------------------------------------------------------------
# TensorRT runner
# ----------------------------------------------------------------------------

class TRTStepRunner:
    """Run one 10 ms STFT frame through a serialized TensorRT 10.x engine.

    NOT VERIFIED ON HARDWARE IN THIS PROJECT.

    Args:
        engine_path: ``.engine`` / ``.plan`` file built *on the target* by
            build_trt_engine.sh (engines are specific to TensorRT version and GPU).
        memory: ``'cudart'`` (cuda-python), ``'torch'`` or ``'auto'``.
        use_cuda_graph: capture the enqueue into two CUDA graphs (one per state
            ping-pong parity).  Falls back to plain enqueue if capture fails.
        copy_outputs: return copies (default).  ``False`` returns views of the
            pinned output buffers, which are overwritten by the next call -
            allocation-free, for a carefully written real-time loop.
        expected_bins: frequency bins required in ``spec`` (``None`` = any).
        log_level: TensorRT logger severity name (``'WARNING'``, ``'INFO'``, ...).

    Example::

        runner = TRTStepRunner("exports/dancnet_v3hq_cont_fp16.engine", use_cuda_graph=True)
        runner.reset()
        spec_out, spp = runner(spec)          # spec: float32 [1, 1, 161, 2]
    """

    def __init__(self, engine_path: str, *, memory: str = "auto", use_cuda_graph: bool = False,
                 copy_outputs: bool = True, expected_bins: int | None = DEFAULT_N_BINS,
                 log_level: str = "WARNING") -> None:
        import tensorrt as trt  # JetPack ships the bindings (python3-libnvinfer)

        self._trt = trt
        self._logger = trt.Logger(getattr(trt.Logger, log_level))
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self.engine = self._runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize '{engine_path}' - an engine must be built with the same "
                               "TensorRT version on the same GPU type; rebuild it with build_trt_engine.sh")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context() failed (out of device memory?)")

        # -- discover I/O tensors; pin a dynamic batch dim to 1 ------------------------------
        raw = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            dims = tuple(self.engine.get_tensor_shape(name))
            if is_input and any(d < 0 for d in dims):
                if not self.context.set_input_shape(name, resolve_shape(dims, name)):
                    raise ContractError(f"engine rejected static shape for '{name}'")
            raw.append((name, is_input, dtype))
        tensors = [TensorSpec(n, resolve_shape(tuple(self.context.get_tensor_shape(n)), n), d, inp)
                   for n, inp, d in raw]
        self.contract = parse_contract(tensors, expected_bins)

        # -- buffers ---------------------------------------------------------------------------
        self._mem = make_memory_backend(memory)
        c = self.contract
        self._h_spec = self._mem.host_alloc(c.spec_in.shape, c.spec_in.dtype)
        self._h_spec_out = self._mem.host_alloc(c.spec_out.shape, c.spec_out.dtype)
        self._h_spp = self._mem.host_alloc(c.spp.shape, c.spp.dtype)
        self._d_spec = self._mem.device_alloc(c.spec_in.nbytes)
        self._d_spec_out = self._mem.device_alloc(c.spec_out.nbytes)
        self._d_spp = self._mem.device_alloc(c.spp.nbytes)
        self._d_states = [(self._mem.device_alloc(s_in.nbytes), self._mem.device_alloc(s_in.nbytes))
                          for s_in, _ in c.states]
        for name, buf in ((c.spec_in.name, self._d_spec), (c.spec_out.name, self._d_spec_out),
                          (c.spp.name, self._d_spp)):
            self._set_address(name, buf.ptr)
        self.copy_outputs = copy_outputs
        self._parity = 0
        self._graphs: list[object] = []
        self.cuda_graph_active = False
        self.cuda_graph_error: str | None = None
        self.reset()
        if use_cuda_graph:
            self._capture_graphs()

    # -- internals ------------------------------------------------------------------------------
    def _set_address(self, name: str, ptr: int) -> None:
        if not self.context.set_tensor_address(name, ptr):
            raise RuntimeError(f"set_tensor_address('{name}') failed")

    def _bind_states(self, parity: int) -> None:
        """parity 0: read A / write B; parity 1: read B / write A."""
        for (s_in, s_out), (buf_a, buf_b) in zip(self.contract.states, self._d_states):
            src, dst = (buf_a, buf_b) if parity == 0 else (buf_b, buf_a)
            self._set_address(s_in.name, src.ptr)
            self._set_address(s_out.name, dst.ptr)

    def _enqueue(self) -> None:
        if not self.context.execute_async_v3(self._mem.stream_handle):
            raise RuntimeError("execute_async_v3 returned False (see TensorRT log)")

    def _capture_graphs(self) -> None:
        try:
            graphs = []
            for parity in (0, 1):
                self._bind_states(parity)
                self._enqueue()  # TensorRT: run once before capture so internal state is up to date
                self._mem.synchronize()
                graphs.append(self._mem.capture(self._enqueue))
            self._graphs = graphs
            self.cuda_graph_active = True
        except Exception as exc:  # unsupported layer / plugin / driver
            self._graphs = []
            self.cuda_graph_active = False
            self.cuda_graph_error = f"{type(exc).__name__}: {exc}"
        self.reset()  # the warm-up enqueues advanced the recurrent state

    # -- public API -----------------------------------------------------------------------------
    @property
    def memory_backend(self) -> str:
        return self._mem.name

    def reset(self) -> None:
        """Zero every recurrent state buffer (start of a new stream)."""
        for buf_a, buf_b in self._d_states:
            self._mem.zero(buf_a)
            self._mem.zero(buf_b)
        self._mem.synchronize()
        self._parity = 0

    def __call__(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """spec float [1, 1, F, 2] -> (spec_out [1, 1, F, 2], spp [1, 1, F])."""
        np.copyto(self._h_spec.array, _check_frame(spec, self.contract.spec_in), casting="same_kind")
        mem = self._mem
        mem.h2d(self._d_spec, self._h_spec)
        if self.cuda_graph_active:
            mem.launch(self._graphs[self._parity])
        else:
            self._bind_states(self._parity)
            self._enqueue()
        mem.d2h(self._h_spec_out, self._d_spec_out)
        mem.d2h(self._h_spp, self._d_spp)
        mem.synchronize()
        self._parity ^= 1
        if self.copy_outputs:
            return self._h_spec_out.array.copy(), self._h_spp.array.copy()
        return self._h_spec_out.array, self._h_spp.array

    def close(self) -> None:
        """Release device memory, graphs and the stream (also done at interpreter exit)."""
        mem = getattr(self, "_mem", None)
        if mem is not None:
            mem.close()
            self._mem = None  # type: ignore[assignment]

    def __enter__(self) -> "TRTStepRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ----------------------------------------------------------------------------
# ONNX Runtime runner (CPU baseline / fallback, same interface)
# ----------------------------------------------------------------------------

_ORT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}


class ORTStepRunner:
    """Same interface as :class:`TRTStepRunner`, backed by ONNX Runtime.

    On the Jetson this is the ARM-CPU path (``CPUExecutionProvider``, one
    intra-op thread pinned to an isolated core), which for a tiny batch-1
    recurrent model can match or beat the GPU on tail latency and power - this
    must be measured, see README.md.  States live in host numpy arrays and are
    threaded from ``state_out_k`` to ``state_in_k`` on every call.

    Args:
        onnx_path: the streaming ONNX model.
        providers: ONNX Runtime execution providers (default CPU only).
        intra_op_num_threads: 1 is recommended for a real-time audio thread.
        allow_spinning: let idle intra-op threads spin (only matters with >1 thread).
        expected_bins / copy_outputs: as for :class:`TRTStepRunner`.
    """

    def __init__(self, onnx_path: str, *, providers: Sequence[str] | None = None,
                 intra_op_num_threads: int = 1, allow_spinning: bool = False,
                 expected_bins: int | None = DEFAULT_N_BINS, copy_outputs: bool = False) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra_op_num_threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spinning else "0")
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so,
                                            providers=list(providers or ["CPUExecutionProvider"]))
        tensors = []
        for meta, is_input in [(m, True) for m in self.session.get_inputs()] + \
                              [(m, False) for m in self.session.get_outputs()]:
            if meta.type not in _ORT_DTYPES:
                raise ContractError(f"tensor '{meta.name}' has unsupported type {meta.type}")
            shape = resolve_shape(meta.shape, meta.name, allow_unknown=not is_input)
            tensors.append(TensorSpec(meta.name, shape, np.dtype(_ORT_DTYPES[meta.type]), is_input))
        self.contract = parse_contract(tensors, expected_bins)
        self.providers = self.session.get_providers()
        self.copy_outputs = copy_outputs  # ORT already returns fresh arrays; kept for API symmetry
        self._output_names = [SPEC_OUT, SPP_OUT] + [s_out.name for _, s_out in self.contract.states]
        self._feeds: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        """Zero every recurrent state (start of a new stream)."""
        for s_in, _ in self.contract.states:
            self._feeds[s_in.name] = np.zeros(s_in.shape, dtype=s_in.dtype)

    def __call__(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """spec float [1, 1, F, 2] -> (spec_out [1, 1, F, 2], spp [1, 1, F])."""
        c = self.contract
        self._feeds[SPEC_IN] = _check_frame(spec, c.spec_in).astype(c.spec_in.dtype, copy=False)
        outs = self.session.run(self._output_names, self._feeds)
        for k, (s_in, _) in enumerate(c.states):
            self._feeds[s_in.name] = outs[2 + k]
        if self.copy_outputs:
            return outs[0].copy(), outs[1].copy()
        return outs[0], outs[1]

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]

    def __enter__(self) -> "ORTStepRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ----------------------------------------------------------------------------
# Benchmark / comparison helpers
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class LatencyStats:
    """Per-call wall-clock latency summary (milliseconds)."""

    n: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    p999_ms: float
    max_ms: float
    budget_ms: float
    over_budget: int  # calls slower than budget_ms (each one is a dropped/late hop in real time)

    def format(self, label: str = "") -> str:
        head = f"{label}: " if label else ""
        return (f"{head}n={self.n} mean={self.mean_ms:.3f} std={self.std_ms:.3f} p50={self.p50_ms:.3f} "
                f"p90={self.p90_ms:.3f} p99={self.p99_ms:.3f} p99.9={self.p999_ms:.3f} max={self.max_ms:.3f} ms | "
                f"real-time factor (mean/hop)={self.mean_ms / self.budget_ms:.3f} | "
                f"calls > {self.budget_ms:g} ms: {self.over_budget}")


def summarize_latencies(samples_ms: Sequence[float], budget_ms: float = HOP_MS) -> LatencyStats:
    """Summarize per-call latencies; percentiles use numpy's default (linear) interpolation."""
    x = np.asarray(samples_ms, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no latency samples")
    p50, p90, p99, p999 = np.percentile(x, [50, 90, 99, 99.9])
    return LatencyStats(int(x.size), float(x.mean()), float(x.std()), float(p50), float(p90), float(p99),
                        float(p999), float(x.max()), float(budget_ms), int(np.sum(x > budget_ms)))


def random_frames(n_frames: int, rng: np.random.Generator, n_bins: int = DEFAULT_N_BINS,
                  scale: float = 0.1) -> np.ndarray:
    """(n_frames, 1, 1, F, 2) float32 Gaussian frames for timing / equivalence checks."""
    return (scale * rng.standard_normal((n_frames, 1, 1, n_bins, 2))).astype(np.float32)


def frames_from_wav(path: str, max_frames: int | None = None) -> np.ndarray:
    """STFT frames of a 16 kHz wav using the project's shared framing (needs danc + soundfile)."""
    import soundfile as sf

    from danc.dsp import DEFAULT, stft

    x, fs = sf.read(path, dtype="float64", always_2d=True)
    if fs != DEFAULT.fs:
        raise ValueError(f"{path}: fs={fs}, expected {DEFAULT.fs} Hz (resample first)")
    X = stft(x[:, 0])  # first channel = primary mic
    if max_frames is not None:
        X = X[:max_frames]
    return np.stack([X.real, X.imag], axis=-1).astype(np.float32)[:, None, None]


def benchmark(runner: Callable[[np.ndarray], object], frames: np.ndarray, warmup: int = 100,
              budget_ms: float = HOP_MS, disable_gc: bool = True) -> LatencyStats:
    """Time ``runner(frame)`` per frame (host wall clock, includes H2D/D2H + sync).

    This is the number that matters for the audio callback; trtexec's "GPU
    Compute Time" excludes Python and transfer overhead.  Python's cyclic GC is
    disabled during timing (as a real-time loop should do), then re-enabled.
    """
    if hasattr(runner, "reset"):
        runner.reset()  # type: ignore[attr-defined]
    for i in range(min(warmup, len(frames))):
        runner(frames[i])
    gc_was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    times = np.empty(len(frames), dtype=np.float64)
    try:
        for i in range(len(frames)):
            t0 = time.perf_counter_ns()
            runner(frames[i])
            times[i] = (time.perf_counter_ns() - t0) * 1e-6
    finally:
        if gc_was_enabled:
            gc.enable()
    return summarize_latencies(times, budget_ms)


def compare_runners(reference, candidate, frames: np.ndarray) -> dict[str, float]:
    """Run two runners over the same frame sequence (states threaded) and report the error.

    Typical use: FP32 ONNX (reference) vs FP16 TensorRT engine (candidate).  The
    late/early error ratio exposes recurrent-state drift that a single-frame
    test would miss.
    """
    reference.reset()
    candidate.reset()
    ref_spec, cand_spec, ref_spp, cand_spp = [], [], [], []
    for f in frames:
        rs, rp = reference(f)
        cs, cp = candidate(f)
        ref_spec.append(np.array(rs, dtype=np.float64))
        cand_spec.append(np.array(cs, dtype=np.float64))
        ref_spp.append(np.array(rp, dtype=np.float64))
        cand_spp.append(np.array(cp, dtype=np.float64))
    rs, cs = np.stack(ref_spec), np.stack(cand_spec)
    rp, cp = np.stack(ref_spp), np.stack(cand_spp)
    err = cs - rs
    per_frame = np.sqrt(np.mean(err.reshape(len(err), -1) ** 2, axis=1))
    q = max(1, len(per_frame) // 10)
    tiny = np.finfo(np.float64).tiny
    return {
        "n_frames": float(len(frames)),
        "spec_out_max_abs_err": float(np.max(np.abs(err))),
        "spec_out_snr_db": float(10 * np.log10((np.sum(rs ** 2) + tiny) / (np.sum(err ** 2) + tiny))),
        "spp_max_abs_err": float(np.max(np.abs(cp - rp))),
        "spp_mean_abs_err": float(np.mean(np.abs(cp - rp))),
        "late_over_early_rms_err": float((per_frame[-q:].mean() + tiny) / (per_frame[:q].mean() + tiny)),
    }


# ----------------------------------------------------------------------------
# CLI micro-benchmark
# ----------------------------------------------------------------------------

def _apply_rt_settings(rt_priority: int | None, cpus: str | None) -> None:
    """Optionally pin to CPUs and switch to SCHED_FIFO (Linux; needs root or rtprio limits)."""
    if cpus:
        os.sched_setaffinity(0, {int(c) for c in cpus.split(",") if c.strip()})
    if rt_priority:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(rt_priority))


def _env_report() -> str:
    parts = [f"python {platform.python_version()}", platform.platform()]
    if hasattr(os, "sched_getaffinity"):
        parts.append(f"cpus={sorted(os.sched_getaffinity(0))}")
    if hasattr(os, "sched_getscheduler"):
        pol = os.sched_getscheduler(0)
        parts.append("sched=" + {getattr(os, "SCHED_FIFO", -1): "FIFO", getattr(os, "SCHED_RR", -2): "RR"}
                     .get(pol, "OTHER"))
    return " | ".join(parts)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D-ANC streaming-network micro-benchmark "
                                            "(TensorRT engine and/or ONNX Runtime).")
    p.add_argument("--engine", help="TensorRT engine (.engine/.plan) built on this device")
    p.add_argument("--onnx", help="ONNX model for the ONNX Runtime runner")
    p.add_argument("--memory", default="auto", choices=["auto", "cudart", "torch"],
                   help="CUDA memory back-end for the TensorRT runner")
    p.add_argument("--cuda-graph", action="store_true", help="capture the TensorRT enqueue into CUDA graphs")
    p.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads")
    p.add_argument("--providers", default="CPUExecutionProvider",
                   help="comma-separated ONNX Runtime providers, e.g. CUDAExecutionProvider,CPUExecutionProvider")
    p.add_argument("--frames", type=int, default=3000, help="timed frames (3000 = 30 s of audio)")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--budget-ms", type=float, default=HOP_MS, help="deadline per call (hop length)")
    p.add_argument("--seed", type=int, default=0, help="seed of the numpy Generator for random frames")
    p.add_argument("--wav", help="16 kHz wav whose STFT frames are used instead of random frames")
    p.add_argument("--compare", action="store_true",
                   help="with --engine and --onnx: report engine-vs-ONNX numerical error first")
    p.add_argument("--rt-priority", type=int, default=None, help="SCHED_FIFO priority for this process (Linux)")
    p.add_argument("--cpus", default=None, help="CPU affinity, e.g. 10,11 (Linux)")
    p.add_argument("--json", default=None, help="write results as JSON to this path")
    args = p.parse_args(argv)
    if not args.engine and not args.onnx:
        p.error("give --engine and/or --onnx")
    if args.compare and not (args.engine and args.onnx):
        p.error("--compare needs both --engine and --onnx")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _apply_rt_settings(args.rt_priority, args.cpus)
    print(_env_report())
    rng = np.random.default_rng(args.seed)
    n_total = args.frames + args.warmup
    frames = frames_from_wav(args.wav, n_total) if args.wav else random_frames(n_total, rng)
    if len(frames) <= args.warmup:
        raise SystemExit(f"only {len(frames)} frames available, need more than --warmup={args.warmup}")
    results: dict[str, object] = {"env": _env_report(), "frames": int(len(frames) - args.warmup)}

    runners: dict[str, object] = {}
    if args.onnx:
        runners["ort"] = ORTStepRunner(args.onnx, providers=args.providers.split(","),
                                       intra_op_num_threads=args.threads)
        print(f"[ort] providers={runners['ort'].providers}\n{runners['ort'].contract.describe()}")
    if args.engine:
        trt_runner = TRTStepRunner(args.engine, memory=args.memory, use_cuda_graph=args.cuda_graph)
        runners["trt"] = trt_runner
        print(f"[trt] memory={trt_runner.memory_backend} cuda_graph={trt_runner.cuda_graph_active}"
              + (f" (capture failed: {trt_runner.cuda_graph_error})" if trt_runner.cuda_graph_error else "")
              + f"\n{trt_runner.contract.describe()}")

    if args.compare:
        cmp = compare_runners(runners["ort"], runners["trt"], frames)
        results["compare_trt_vs_ort"] = cmp
        print("[compare] " + " ".join(f"{k}={v:.4g}" for k, v in cmp.items()))

    timed = frames[args.warmup:]
    for name, runner in runners.items():
        runner.reset()  # type: ignore[attr-defined]
        for i in range(args.warmup):
            runner(frames[i])  # type: ignore[operator]
        stats = benchmark(runner, timed, warmup=0, budget_ms=args.budget_ms)  # type: ignore[arg-type]
        results[name] = asdict(stats)
        print(stats.format(f"[{name}]"))
    for runner in runners.values():
        runner.close()  # type: ignore[attr-defined]
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
