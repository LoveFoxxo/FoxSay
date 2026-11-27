"""
test_bias_precision.py -- bias insertion, fp64 upgrade, and the loss->0
convergence theorems.

  * Linear casts bias to the activation dtype (no bf16/fp32 mismatch)
  * lm_head / ve_gate carry bias; init_weights zeroes every bias
  * setup_optimizer routes 1D params (bias) to AdamW, Muon only sees 2D
  * upgrade_to_fp64 lifts model + optimizer state to fp64 in place
  * Chinchilla bound is finite and above the irreducible entropy
  * interpolation feasibility: capacity vs data information (B1)
  * PL bound is linear; linear_convergence_check detects geometric decay
  * NTK lambda_min > 0 for an over-parametrized linear net (B3)
  * convergence_path decides regime and names the certifying theorem
"""

import math

import torch
import torch.nn as nn

from FoxSay.gpt import GPT, GPTConfig, Linear
from FoxSay.optim import upgrade_to_fp64
from FoxSay.mathlib.convergence import (
    chinchilla_lower_bound, data_information_bits, model_capacity_bits,
    interpolation_feasible, pl_convergence_bound, linear_convergence_check,
    ntk_min_eigenvalue, convergence_path,
)


def _small_gpt():
    cfg = GPTConfig(n_layer=2, n_embd=32, n_head=4, n_kv_head=4,
                    vocab_size=64, sequence_len=64)
    m = GPT(cfg, pad_vocab_size_to=64)
    m.init_weights()
    return m, cfg


# ---------------------------------------------------------------------------
# bias
# ---------------------------------------------------------------------------
def test_linear_casts_bias_to_activation_dtype():
    lin = Linear(8, 8, bias=True)
    x = torch.randn(2, 8, dtype=torch.bfloat16)   # activation dtype bf16
    y = lin(x)                                    # must not raise dtype error
    assert y.dtype == torch.bfloat16


def test_bias_present_at_necessary_positions_and_zero_init():
    m, cfg = _small_gpt()
    assert m.lm_head.bias is not None
    assert torch.all(m.lm_head.bias == 0)
    # ve_gate exists on layer 1 (n_layer=2 -> has_ve(1,2) True)
    assert m.transformer.h[1].attn.ve_gate.bias is not None
    assert torch.all(m.transformer.h[1].attn.ve_gate.bias == 0)
    # every bias in the model is zero-initialized
    for mod in m.modules():
        if isinstance(mod, nn.Linear) and mod.bias is not None:
            assert torch.all(mod.bias == 0)
    # forward still works end-to-end
    x = torch.randint(0, 64, (2, 8))
    logits = m(x)
    assert logits.shape == (2, 8, 64)


def test_optimizer_routes_bias_to_adamw_muon_2d_only():
    m, cfg = _small_gpt()
    opt = m.setup_optimizer()
    for group in opt.param_groups:
        if group["kind"] == "muon":
            for p in group["params"]:
                assert p.dim() >= 2, "Muon must never see 1D params (bias)"
        else:
            assert group["kind"] == "adamw"
    # lm_head bias and ve_gate bias are covered by SOME group
    covered = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(m.lm_head.bias) in covered
    assert id(m.transformer.h[1].attn.ve_gate.bias) in covered
    # the bias group is AdamW with no weight decay
    bias_ids = {id(m.lm_head.bias), id(m.transformer.h[1].attn.ve_gate.bias)}
    found = 0
    for group in opt.param_groups:
        if any(id(p) in bias_ids for p in group["params"]):
            assert group["kind"] == "adamw"
            assert group["weight_decay"] == 0.0
            found += 1
    assert found >= 1


# ---------------------------------------------------------------------------
# fp64 upgrade
# ---------------------------------------------------------------------------
def test_upgrade_to_fp64_model_and_optimizer_state():
    m, cfg = _small_gpt()
    opt = m.setup_optimizer()
    m, opt = upgrade_to_fp64(m, opt)
    assert m.lm_head.weight.dtype == torch.float64
    assert m.lm_head.bias.dtype == torch.float64
    assert m.transformer.wte.weight.dtype == torch.float64
    for group in opt.param_groups:
        for p in group["params"]:
            assert p.dtype == torch.float64
    # fp64 forward runs end-to-end; logits come out fp32 by design (gpt
    # softcaps logits in fp32), the fp64 masters live in the weights
    x = torch.randint(0, 64, (2, 8))
    logits = m(x)
    assert logits.dtype == torch.float32


# ---------------------------------------------------------------------------
# loss -> 0 mathematics
# ---------------------------------------------------------------------------
def test_chinchilla_bound_finite_above_entropy():
    c = chinchilla_lower_bound(N=1e9, D=1e10)
    assert c["L_pred"] > c["irreducible"]          # finite, above entropy
    c2 = chinchilla_lower_bound(N=1e12, D=1e13)
    assert c2["L_pred"] < c["L_pred"]              # more scale -> lower bound


def test_interpolation_feasibility_boundary():
    # 1M params, 2 bits/param -> 2M bits; corpus of 500k tokens, vocab 64
    # info = 500k * 6 bits = 3M bits > 2M -> NOT feasible
    feas = interpolation_feasible(P=1_000_000, N_tokens=500_000, vocab_size=64)
    assert not feas["feasible"]
    # 10M params -> 20M bits >= 3M bits -> feasible
    feas2 = interpolation_feasible(P=10_000_000, N_tokens=500_000, vocab_size=64)
    assert feas2["feasible"]
    assert feas["params_critical"] > 1_000_000


def test_pl_bound_and_linear_check():
    bound = pl_convergence_bound(mu=0.2, L_smooth=1.0, t=10, L0=2.0)
    assert abs(bound - 2.0 * (0.8 ** 10)) < 1e-9
    geom = [2.0 * (0.9 ** t) for t in range(20)]
    check = linear_convergence_check(geom)
    assert check["converging"]
    flat = [2.0] * 10
    assert not linear_convergence_check(flat)["converging"]


def test_ntk_lambda_min_positive_for_overparametrized_net():
    torch.manual_seed(0)
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x.float())

    net = TinyNet()
    x = torch.randint(0, 2, (4, 4))               # 4 distinct binary vectors
    lam, K = ntk_min_eigenvalue(net, x)
    assert K.shape == (4, 4)
    assert lam > 0                                  # positive-definite NTK


def test_convergence_path_regime_decision():
    # memorizable corpus -> interpolation regime with the B3/B2 path
    out = convergence_path(P=10_000_000, N_tokens=500_000, vocab_size=64)
    assert out["regime"] == "interpolation"
    assert "B3" in out["loss_to_zero_path"]
    # Internet-scale corpus: scaling-law estimate is not a zero-loss proof.
    out2 = convergence_path(P=1_000_000, N_tokens=10_000_000_000, vocab_size=50257)
    assert out2["regime"] == "compression"
    assert math.isfinite(out2["empirical_loss_estimate"])
    assert not out2["zero_loss_certified"]
    assert not out2["feasible"]
    # with a loss history in the feasible regime, the PL check attaches
    geom = [2.0 * (0.9 ** t) for t in range(20)]
    out3 = convergence_path(P=10_000_000, N_tokens=500_000, vocab_size=64,
                            loss_history=geom)
    assert out3["linear_check"]["converging"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(fns)} tests passed")
