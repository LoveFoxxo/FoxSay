"""Probe: where does the fp64-master divergence start and what does it look like?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def _fake_compile(fn=None, **_kwargs):
    if fn is None:
        return lambda f: f
    return fn


torch.compile = _fake_compile

from tests.test_fsdp import build_tiny_model, build_batches


def run_steps(model, optimizer, batches, collect):
    for i, (x, y) in enumerate(batches):
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
        collect(i, loss.item())


def ref_states():
    torch.manual_seed(42)
    model = build_tiny_model(seed=42)
    opt = model.setup_optimizer()
    ve_group = [g for g in opt.param_groups if g["kind"] == "adamw" and "value_embeds" in str(g.get("name", ""))]
    print("adamw groups:", [(g.get("name"), g["lr"], g["weight_decay"], g["betas"], g["eps"]) for g in opt.param_groups if g["kind"] == "adamw"])
    states = {}
    batches = build_batches()

    def collect(i, loss):
        states[i] = model.value_embeds["0"].weight.detach().clone()

    run_steps(model, opt, batches, collect)
    return states


def fsdp_states(master_dtype):
    from FoxSay.fsdp.core import FullySharded

    torch.manual_seed(42)
    model = build_tiny_model(seed=42)
    opt = model.setup_optimizer()
    fsdp = FullySharded(model, opt, master_dtype=master_dtype)
    fsdp.shard()
    states = {}
    batches = build_batches()

    def collect(i, loss):
        fsdp.free_all()
        sd = fsdp.consolidated_state_dict()
        states[i] = sd["value_embeds.0.weight"].detach().clone()

    run_steps(model, fsdp.optimizer, batches, collect)
    return states


ref = ref_states()
f32 = fsdp_states(torch.float32)
f64 = fsdp_states(torch.float64)
print("\nper-step max|ref - fsdp| for value_embeds.0.weight:")
for i in ref:
    d32 = (ref[i].float() - f32[i].float()).abs().max().item()
    d64 = (ref[i].float() - f64[i].float()).abs().max().item()
    print(f"  step {i}: fp32-master diff = {d32:.3e} | fp64-master diff = {d64:.3e}")
w = ref[0]
print("\nref weight stats: mean|w| =", w.abs().mean().item(), "max|w| =", w.abs().max().item())
