"""
test_mathlib.py -- numerical verification of FoxSay.mathlib against the
reference semantics of the original FoxSay/gan/reward.py.

Run:  python tests/test_mathlib.py        (no torch required; numpy backend)
      pytest -q tests/test_mathlib.py -k "torch"   (with torch installed)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass

from FoxSay.mathlib import (
    Mat, mat,
    openblas_kernels as blas,
    reward_math as rm,
    penalty_math as pm,
)


@dataclass
class Cfg:
    w_correct: float = 1.0
    w_process: float = 0.25
    w_adversarial: float = 0.5
    w_efficiency: float = 0.15
    w_truncation: float = -0.25
    max_cycles_budget: int = 4
    target_cycles_fast: int = 1
    truncation_limit: int = 64
    answer_marker: str = None


def ref_grpo(rewards, groups=None):
    rewards = rewards.astype(np.float64)
    if groups is None:
        return rewards - rewards.mean()
    adv = np.zeros_like(rewards)
    for g in np.unique(groups):
        m = groups == g
        adv[m] = rewards[m] - rewards[m].mean()
    return adv


def test_kernels():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    B = rng.standard_normal((4, 3))
    assert np.allclose(blas.gemm(A, B), A @ B)
    assert np.allclose(blas.potrf(A @ A.T + 4 * np.eye(4)) @ blas.potrf(A @ A.T + 4 * np.eye(4)).T,
                       A @ A.T + 4 * np.eye(4), atol=1e-8)
    assert np.allclose(blas.getrs(blas.getrf(A), B), np.linalg.solve(A, B))
    x = rng.standard_normal(4)
    assert np.isclose(blas.dot(x, x), x @ x)
    assert np.isclose(blas.nrm2(x), np.linalg.norm(x))
    m = Mat(A)
    assert np.allclose(m.t()._a, A.T)
    assert np.allclose(m.i()._a, np.linalg.inv(A))
    assert np.allclose(m.solve(B)._a, np.linalg.solve(A, B))


def test_efficiency():
    cfg = Cfg()
    c = np.array([1, 2, 4, 9], dtype=np.float64)
    r = rm.compute_efficiency_reward(c, cfg)
    expect = -np.clip((c - 1) / 3.0, 0, 1)
    assert np.allclose(r, expect), (r, expect)


def test_shape_reward():
    cfg = Cfg()
    rng = np.random.default_rng(1)
    B = 8
    r_c = rng.integers(0, 2, B).astype(np.float64)
    r_p = rng.uniform(-2, 2, B)
    p_p = rng.uniform(0, 1, B)
    cyc = rng.integers(0, 5, B).astype(np.float64)
    trunc = rng.integers(0, 2, B).astype(bool)
    total = rm.shape_reward(r_c, r_p, p_p, cyc, trunc, cfg)
    r_p_c = np.clip(r_p, -1, 1)
    r_adv = np.clip(p_p - 0.5, -0.5, 0.5)
    r_eff = rm.compute_efficiency_reward(cyc, cfg) * r_c
    r_trunc = trunc.astype(float) * cfg.w_truncation
    expect = (cfg.w_correct * r_c + cfg.w_process * r_p_c
              + cfg.w_adversarial * r_adv + cfg.w_efficiency * r_eff + r_trunc)
    assert np.allclose(total, expect)


def test_grpo_vectorized():
    rng = np.random.default_rng(2)
    rewards = rng.standard_normal(64)
    groups = rng.integers(0, 7, 64)
    assert np.allclose(rm.grpo_advantages(rewards, groups), ref_grpo(rewards, groups))
    assert np.allclose(rm.grpo_advantages(rewards), rewards - rewards.mean())


def test_token_level():
    rng = np.random.default_rng(3)
    B, T = 4, 16
    adv = rng.standard_normal(B)
    mask = rng.integers(0, 2, (B, T)).astype(np.float64)
    got = rm.token_level_advantages(adv, mask)
    expect = adv[:, None] * mask / np.clip(mask.sum(1), 1, None)[:, None]
    assert np.allclose(got, expect)


def test_truncation():
    cfg = Cfg()
    toks = [[1, 2, 3], [2] * 70, [3] * 64, [4] * 64, []]
    out = rm.compute_truncation_status(toks, cfg, eos_ids={3})
    # row0 short; row1 long no eos -> trunc; row2 len64 ends eos 3 -> no;
    # row3 len64 no eos -> trunc; row4 short
    assert list(out) == [False, True, False, True, False]
    # limit None -> all False
    cfg2 = Cfg(truncation_limit=None)
    assert not rm.compute_truncation_status(toks, cfg2, eos_ids={3}).any()


def test_penalty():
    cfg = Cfg()
    toks = [[2] * 70]
    assert pm.truncation_penalty(toks, cfg)[0] == cfg.w_truncation
    pos = np.array([1.0, 0.5, 0.9]); neg = np.array([0.2, 0.1, 0.7])
    expect = np.maximum(0.0, 0.5 - (pos - neg)).mean()
    assert np.isclose(pm.verifier_rank_loss(pos, neg), expect)
    # combined penalty == shape_reward minus outcome term
    rng = np.random.default_rng(4)
    B = 6
    r_c = rng.integers(0, 2, B).astype(np.float64)
    r_p = rng.uniform(-1, 1, B)
    p_p = rng.uniform(0, 1, B)
    cyc = rng.integers(0, 5, B).astype(np.float64)
    trunc = rng.integers(0, 2, B).astype(bool)
    total = rm.shape_reward(r_c, r_p, p_p, cyc, trunc, cfg)
    comb = pm.combined_penalty(r_c, r_p, p_p, cyc, trunc, cfg)
    assert np.allclose(comb, total - cfg.w_correct * r_c)


def test_torch_backend():
    try:
        import torch
    except ImportError:
        print("torch not installed; skipping torch backend checks")
        return
    rng = np.random.default_rng(5)
    rewards = torch.tensor(rng.standard_normal(32))
    groups = torch.tensor(rng.integers(0, 4, 32))
    got = rm._torch_grpo_advantages(rewards, groups)
    assert torch.allclose(got.double(), torch.tensor(ref_grpo(rewards.numpy(), groups.numpy())))
    cfg = Cfg()
    c = torch.tensor([1.0, 2.0, 4.0])
    got = rm._torch_compute_efficiency_reward(c, cfg)
    assert torch.allclose(got.double(), torch.tensor(rm.compute_efficiency_reward(c.numpy(), cfg)))
    print("torch backend OK")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"{fn.__name__}: PASS")
    print("ALL MATHLIB TESTS PASSED")
