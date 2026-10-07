"""Run the public DeepFilterNet3 model (Schroeter et al., Interspeech 2023; MIT/Apache-2.0) on a
D-ANC test set as an external state-of-the-art reference.

DFN3 operates at 48 kHz with 2-frame look-ahead (40 ms algorithmic latency).  Our 16 kHz test audio is
upsampled to 48 kHz (soxr VHQ), enhanced, and downsampled back - so DFN3 sees band-limited input it was
not trained on; it is NOT re-trained on defence noise.  Outputs go to
reports/audio/enhanced_<set>_dfn3/ and are scored by:
    python -m danc.eval.evaluate --set <set> --systems file:dfn3

Run with the isolated DFN dependencies (numpy 1.26 etc.):
    PYTHONPATH=third_party/dfn_pkgs python scripts/run_deepfilternet3.py --set defence_v1
"""
import argparse
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# torchaudio >= 2.9 removed torchaudio.backend; DFN only imports AudioMetaData for type hints.
_m = types.ModuleType("torchaudio.backend")
_c = types.ModuleType("torchaudio.backend.common")
_c.AudioMetaData = object
_m.common = _c
sys.modules.setdefault("torchaudio.backend", _m)
sys.modules.setdefault("torchaudio.backend.common", _c)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402
import torch  # noqa: E402
from df.enhance import enhance, init_df  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from danc.eval.evaluate import load_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="defence_v1")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, df_state, _ = init_df(str(ROOT / "third_party" / "dfn_models" / "DeepFilterNet3"), log_file=None,
                                 log_level="WARNING")
    out = ROOT / "reports" / "audio" / f"enhanced_{a.set}_dfn3"
    out.mkdir(parents=True, exist_ok=True)
    items = load_set(a.set)
    t0 = time.time()
    for i, it in enumerate(items):
        dst = out / f"{it['id']}.flac"
        if dst.exists():
            continue
        x, fs = sf.read(it["noisy"], dtype="float32")
        x48 = soxr.resample(x, fs, df_state.sr(), quality="VHQ").astype(np.float32)
        with torch.no_grad():
            y48 = enhance(model, df_state, torch.from_numpy(x48)[None]).squeeze(0).numpy()
        y = soxr.resample(y48, df_state.sr(), fs, quality="VHQ")[: len(x)]
        sf.write(dst, np.clip(y, -1, 1), fs, subtype="PCM_16", format="FLAC")
        if i % 100 == 0:
            print(f"[dfn3] {i}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    print("[dfn3] done", time.time() - t0)


if __name__ == "__main__":
    main()
