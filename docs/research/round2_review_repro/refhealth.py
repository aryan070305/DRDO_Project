import sys, numpy as np
sys.path.insert(0, "/Users/nsivaprasadnandyala/Desktop/DRDO_Project/src")
from danc.inference.engine import HybridEngine, EngineConfig

class Fake:
    """minimal runner pair: two-mic returns zeros, single-mic returns ones (mix is then directly visible)."""
    def __init__(self, n, L=0): self.n_mics, self.lookahead = n, L
    def reset(self): pass
    def __call__(self, spec, ref=None):
        v = 0.0 if self.n_mics == 2 else 1.0
        return np.full_like(spec, v), np.full(161, v, np.float32)

def run(ref_blocks, cfg=None):
    e = HybridEngine(Fake(2), cfg or EngineConfig(limiter_dbfs=None), fallback_runner=Fake(1))
    trace = []
    for r in ref_blocks:
        e.process_block(np.zeros(160), r)
        trace.append((e.ref_dead, round(e._mix, 2)))
    return e, trace

rng = np.random.default_rng(0)
lvl = lambda db: rng.standard_normal(160) * 10 ** (db / 20)
# 1) alive but quiet reference (quiet room, -75 dBFS self-noise) for 1 s, then normal noise (-40 dBFS) 1 s, repeated
blocks = ([lvl(-75) for _ in range(100)] + [lvl(-40) for _ in range(100)]) * 3
e, tr = run(blocks)
print("quiet-room alternation: switches =", e.ref_switches, " ref_dead at t=0.9s:", tr[90][0], " t=1.2s:", tr[120][0])
# 2) dead mic with DC offset (unplugged input biased at 5 mV ~ -46 dBFS DC) + tiny noise
blocks = [0.005 + lvl(-100) for _ in range(200)]
e, tr = run(blocks)
print("dead mic with DC offset: ref_dead =", e.ref_dead)
# 3) dead mic whose preamp hiss sits at -67 dBFS (inside the 6 dB hysteresis band) from the start
blocks = [lvl(-67) for _ in range(300)]
e, tr = run(blocks)
print("dead mic at -67 dBFS hiss: ref_dead =", e.ref_dead)
# 4) dead mic hiss at -66 dBFS fluctuating +-3 dB per hop around the threshold band
blocks = [lvl(-70 + 3 * np.sin(k)) for k in range(300)]
e, tr = run(blocks)
print("ref fluctuating -73..-67 dBFS: ref_dead =", e.ref_dead, "switches", e.ref_switches)
# 5) ref None from start
e, tr = run([None] * 12)
print("ref None: ", tr[:3], tr[-1])
# 6) int16 reference block
blocks = [(rng.standard_normal(160) * 3000).astype(np.int16) for _ in range(60)]
try:
    e, tr = run(blocks)
    print("int16 ref: ref_dead =", e.ref_dead)
except Exception as ex:
    print("int16 ref:", type(ex).__name__, ex)
