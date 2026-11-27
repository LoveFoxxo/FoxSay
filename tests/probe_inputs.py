"""Instrumented: capture adamw_step_fused inputs at every step for ref vs FSDP-fp64
(single process), then diff them element-wise for the value_embeds group."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def _fake_compile(fn=None, **_kwargs):
    if fn is None:
        return lambda f: f
    return fn


torch.compile = _fake_compile

import FoxSay.optim as optim_mod
import FoxSay.fsdp.core as core_mod
from tests.test_fsdp import build_tiny_model, build_batches

_RECORD = {"calls": []}


def wrap(original):
    def wrapper(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, b1_t, b2_t, eps_t, wd_t):
        rec = dict(
            p=p.detach().clone(),
            grad=grad.detach().clone(),
            m=exp_avg.detach().clone(),
            v=exp_avg_sq.detach().clone(),
            step=float(step_t.item()),
            lr=float(lr_t.item()),
            b1=float(b1_t.item()),
            b2=float(b2_t.item()),
            eps=float(eps_t.item()),
            wd=float(wd_t.item()),
        )
        ret = original(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, b1_t, b2_t, eps_t, wd_t)
        rec["p_out"] = p.detach().clone()
        rec["m_out"] = exp_avg.detach().clone()
        rec["v_out"] = exp_avg_sq.detach().clone()
        _RECORD["calls"].append(rec)
        return ret

    return wrapper


optim_mod.adamw_step_fused = wrap(optim_mod.adamw_step_fused)
core_mod.adamw_step_fused = wrap(core_mod.adamw_step_fused)


def run(kind, master_dtype=None):
    _RECORD["calls"] = []
    model = build_tiny_model(seed=42)
    opt = model.setup_optimizer()
    fsdp = None
    if kind == "fsdp":
        from FoxSay.fsdp.core import FullySharded

        fsdp = FullySharded(model, opt, master_dtype=master_dtype)
        fsdp.shard()
        runner = fsdp.optimizer
    else:
        runner = opt
    calls = []
    for x, y in build_batches():
        loss = model(x, y)
        loss.backward()
        _RECORD["calls"] = []
        runner.step()
        model.zero_grad(set_to_none=True)
        calls.append(_RECORD["calls"])
    return calls


# ve flat-group calls: total numel 24576 (3 x 8192); ref calls are per-param (8192)
ref_steps = run("ref")
f64_steps = run("fsdp", master_dtype=torch.float64)

print("groups seen (numel, lr, wd, betas, eps) per step, ref:")
for s, calls in enumerate(ref_steps):
    print(f"  step {s}:", sorted({(c["p"].numel(), round(c["lr"], 6), round(c["wd"], 4), (c["b1"], c["b2"]), c["eps"]) for c in calls}))
print("fsdp fp64:")
for s, calls in enumerate(f64_steps):
    print(f"  step {s}:", sorted({(c["p"].numel(), round(c["lr"], 6), round(c["wd"], 4), (c["b1"], c["b2"]), c["eps"]) for c in calls}))


def ve_call(calls, step):
    for c in calls[step]:
        if c["p"].numel() == 24576 and abs(c["lr"] - 0.3464101615137755) < 1e-9:
            return c
    for c in calls[step]:
        if c["p"].numel() == 8192 and abs(c["lr"] - 0.3464101615137755) < 1e-9:
            return c
    return None


def ve_calls(calls, step, lr_target=0.3464101615137755):
    out = [c for c in calls[step] if abs(c["lr"] - lr_target) < 1e-6]
    assert out, f"no ve call at step {step}"
    return out


for run_name, steps in (("fp32", run("fsdp", master_dtype=torch.float32)), ("fp64", run("fsdp", master_dtype=torch.float64))):
    print(f"\n=== {run_name}: input+output diffs (concat of the 3 ve params) ===")
    for step in (0, 1):
        ref_cs = ve_calls(ref_steps, step)
        fs_cs = ve_calls(steps, step)
        print(f"  step {step} hypers: ref(step={ref_cs[0]['step']}, lr={ref_cs[0]['lr']!r}, b1={ref_cs[0]['b1']!r}, b2={ref_cs[0]['b2']!r}, eps={ref_cs[0]['eps']!r}, wd={ref_cs[0]['wd']!r})")
        print(f"           fsdp(step={fs_cs[0]['step']}, lr={fs_cs[0]['lr']!r}, b1={fs_cs[0]['b1']!r}, b2={fs_cs[0]['b2']!r}, eps={fs_cs[0]['eps']!r}, wd={fs_cs[0]['wd']!r})")
        ref_cat = {k: torch.cat([c[k].float().reshape(-1) for c in ref_cs]) for k in ("p", "grad", "m", "v", "p_out")}
        fs_cat = {k: torch.cat([c[k].float().reshape(-1) for c in fs_cs]) for k in ("p", "grad", "m", "v", "p_out")}
        n = min(ref_cat["p"].numel(), fs_cat["p"].numel())
        for name in ("p", "grad", "m", "v", "p_out"):
            a, b = ref_cat[name][:n], fs_cat[name][:n]
            d = (a - b).abs()
            print(f"    {name:6s}: max|diff| = {d.max().item():.3e}  mean = {d.mean().item():.3e}")
