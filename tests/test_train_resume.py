import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from danc.train.train import restore_ema


def test_restored_ema_is_averaged_not_overwritten():
    """AveragedModel copies the raw weights on its first update while n_averaged == 0; restore_ema() must make the
    first update after a resume AVERAGE into the restored EMA weights (regression: two-mic --resume lost its EMA)."""
    torch.manual_seed(0)
    raw = torch.nn.Linear(4, 3)
    ema = AveragedModel(raw, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    saved = {k: torch.randn_like(v) for k, v in ema.module.state_dict().items()}   # "EMA from the checkpoint"
    restore_ema(ema, saved, step=2000)
    ema.update_parameters(raw)
    for k, v in ema.module.state_dict().items():
        expect = 0.9 * saved[k] + 0.1 * raw.state_dict()[k]
        assert torch.allclose(v, expect, atol=1e-6), k
