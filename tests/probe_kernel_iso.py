"""Isolate: ONE adamw_step_fused call on identical values, fp32-acc vs fp64-acc.
If outputs differ ~6e-4, the ref-vs-fp64 gap is intrinsic math divergence (fp32
reference is simply a different trajectory), not an FSDP sharding bug."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def _fake_compile(fn=None, **_kwargs):
    if fn is None:
        return lambda f: f
    return fn


torch.compile = _fake_compile

from FoxSay.optim import adamw_step_fused
from tests.test_fsdp import build_tiny_model, build_batches

# --- capture the real step-2 state of the ve group from a stock run ---
model = build_tiny_model(seed=42)
opt = model.setup_optimizer()
batches = build_batches()

captured = {}
orig = adamw_step_fused


def wrap(p, grad, m, v, st, lr, b1, b2, eps, wd):
    if abs(float(lr.item()) - 0.3464101552963257) < 1e-6:
        captured.setdefault("calls", []).append((p.clone(), grad.clone(), m.clone(), v.clone(), [t.clone() for t in (st, lr, b1, b2, eps, wd)]))
    return orig(p, grad, m, v, st, lr, b1, b2, eps, wd)


import FoxSay.optim as om

om.adamw_step_fused = wrap

for i, (x, y) in enumerate(batches[:2]):
    loss = model(x, y)
    loss.backward()
    opt.step()
    model.zero_grad(set_to_none=True)

calls = captured["calls"]
print(f"captured {len(calls)} ve calls across 2 steps")
p0, g0, m0, v0, h0 = calls[1]  # second optimizer step (step_t = 2)

st, lr, b1, b2, eps, wd = [t.clone() for t in h0]

# --- run the SAME values through fp32-acc and fp64-acc ---
p32, m32, v32, g32 = p0.clone(), m0.clone(), v0.clone(), g0.clone()
adamw_step_fused(p32, g32, m32, v32, st, lr, b1, b2, eps, wd)

p64, m64, v64, g64 = p0.double(), g0.double(), m0.double(), v0.double()
h64 = [t for t in (st, lr, b1, b2, eps, wd)]  # hypers stay fp32 0-D, as in prod
adamw_step_fused(p64, g64, m64, v64, st, lr, b1, b2, eps, wd)

d = (p32.double() - p64).abs()
print(f"ONE call, identical input values, fp32-acc vs fp64-acc:")
print(f"  max|diff| = {d.max().item():.3e}  mean = {d.mean().item():.3e}")

# where does the divergence live? inspect the worst element
i_worst = d.argmax().item()
print(f"  worst element idx {i_worst}: p_in={p0.flatten()[i_worst].item():.6e}, "
      f"grad={g0.flatten()[i_worst].item():.6e}, m_in={m0.flatten()[i_worst].item():.6e}, "
      f"v_in={v0.flatten()[i_worst].item():.6e}")
step = float(st)
denom32 = (v32 / (1 - float(b2) ** step)).sqrt() + float(eps)
ratio32 = (m32 / (1 - float(b1) ** step)) / denom32
denom64 = (v64 / (1 - float(b2) ** step)).sqrt() + float(eps)
ratio64 = (m64 / (1 - float(b1) ** step)) / denom64
print(f"  update ratio m_hat/denom at worst elem: fp32={ratio32.flatten()[i_worst].item():.6e} fp64={ratio64.flatten()[i_worst].item():.6e}")
v_sorted = v0.flatten().abs()
print(f"  v stats: median={v_sorted.median().item():.3e}, p1={v_sorted.quantile(0.01).item():.3e}, min={v_sorted.min().item():.3e}")
