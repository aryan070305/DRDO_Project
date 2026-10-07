import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
torch.manual_seed(0)
m = torch.nn.Linear(4, 4)
saved_ema = {k: torch.full_like(v, 7.0) for k, v in m.state_dict().items()}   # pretend restored EMA
ema = AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(0.998), use_buffers=True)
ema.module.load_state_dict(saved_ema)
print("n_averaged after resume:", int(ema.n_averaged))
with torch.no_grad():
    m.weight.add_(0.01)
ema.update_parameters(m)
print("EMA weight after 1 update (should be ~7*0.998+w*0.002 = ~6.99):", ema.module.weight[0, :2].tolist())
print("raw model weight:", m.weight[0, :2].tolist())
