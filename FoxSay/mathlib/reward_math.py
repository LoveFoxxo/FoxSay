"""
reward_math.py -- vectorized rewrite of FoxSay.gan.reward on top of mathlib.

Same function signatures and semantics as FoxSay/gan/reward.py, but every
python loop is replaced with compiled BLAS/LAPACK-backed reductions
(bincount/scatter style aggregation from openblas_kernels.group_reduce).

Two backends are provided:
  * numpy backend (mathlib): zero-dependency, used for offline verification.
  * torch backend (mathlib_t): tensor-native, drop-in for the training loop.

`cfg` is any object exposing the same fields as RewardConfig.
"""

from __future__ import annotations

import numpy as np

from .openblas_kernels import group_reduce

__all__ = [
    "compute_efficiency_reward", "compute_truncation_status", "shape_reward",
    "grpo_advantages", "token_level_advantages",
]


def _f(x):
    return np.asarray(x, dtype=np.float64)


def compute_efficiency_reward(cycle_counts, cfg):
    """Cycle-efficiency shaping (identical semantics to gan.reward)."""
    c = _f(cycle_counts)
    budget = float(max(cfg.max_cycles_budget, cfg.target_cycles_fast, 1))
    over = ((c - cfg.target_cycles_fast) / max(budget - cfg.target_cycles_fast, 1))
    over = np.clip(over, 0.0, 1.0)
    return -over


def compute_truncation_status(generated_tokens, cfg, eos_ids=None):
    """Vectorized truncation detection: ragged token lists -> (B,) bool."""
    B = len(generated_tokens)
    if cfg.truncation_limit is None:
        return np.zeros(B, dtype=bool)
    n = len(generated_tokens)
    lengths = np.fromiter(
        (len(t) for t in generated_tokens), dtype=np.int64, count=n
    ) if hasattr(generated_tokens, "__len__") else np.array(
        [len(t) for t in generated_tokens], dtype=np.int64
    )
    finished = np.zeros(B, dtype=bool)
    if eos_ids is not None:
        eos = set(eos_ids)
        for i, toks in enumerate(generated_tokens):
            if lengths[i] > 0 and toks[-1] in eos:
                finished[i] = True
    return (lengths >= cfg.truncation_limit) & ~finished


def shape_reward(
    r_correct, r_process, p_pass, cycle_counts, truncated, cfg,
):
    """Composite reward, pure vector ops (identical semantics to gan.reward)."""
    r_correct = _f(r_correct)
    r_process = np.clip(_f(r_process), -1.0, 1.0)
    r_adv = np.clip(_f(p_pass) - 0.5, -0.5, 0.5)
    r_eff = compute_efficiency_reward(cycle_counts, cfg) * r_correct
    r_trunc = _f(truncated) * cfg.w_truncation
    return (
        cfg.w_correct * r_correct
        + cfg.w_process * r_process
        + cfg.w_adversarial * r_adv
        + cfg.w_efficiency * r_eff
        + r_trunc
    )


def grpo_advantages(rewards, groups=None, sigma_eps=1e-6):
    """GRPO group-relative advantage, fully vectorized (no group loop).

    Uses bincount-based group_reduce so the per-group mean is computed by
    compiled BLAS/LAPACK kernels for arbitrary group counts.
    """
    rewards = _f(rewards)
    if groups is None:
        return rewards - rewards.mean()
    g = np.asarray(groups, dtype=np.int64)
    means = group_reduce(rewards, g, reduce="mean")
    return rewards - means[g]


def token_level_advantages(advantages, mask):
    """DAPO-style token broadcasting, vectorized (identical semantics)."""
    advantages = _f(advantages)
    mask = _f(mask)
    B, T = mask.shape
    per_tok = advantages[:, None] * mask  # broadcast (B,) -> (B, T)
    num_valid = np.clip(mask.sum(axis=1), 1, None)
    return per_tok / num_valid[:, None]


# --------------------------------------------------------------------------
# torch backend: same algorithms, tensor-native (no python loops)
# --------------------------------------------------------------------------

def _torch_compute_efficiency_reward(cycle_counts, cfg):
    import torch
    c = cycle_counts.float()
    budget = float(max(cfg.max_cycles_budget, cfg.target_cycles_fast, 1))
    over = ((c - cfg.target_cycles_fast) / max(budget - cfg.target_cycles_fast, 1)).clamp(0, 1)
    return -over


def _torch_truncation_status(generated_tokens, cfg, eos_ids=None):
    import torch
    B = len(generated_tokens)
    if cfg.truncation_limit is None:
        return torch.zeros(B, dtype=torch.bool)
    out = torch.zeros(B, dtype=torch.bool)
    eos = set(eos_ids) if eos_ids else set()
    for i, toks in enumerate(generated_tokens):
        finished = bool(eos and len(toks) > 0 and toks[-1] in eos)
        if len(toks) >= cfg.truncation_limit and not finished:
            out[i] = True
    return out


def _torch_shape_reward(r_correct, r_process, p_pass, cycle_counts, truncated, cfg):
    import torch
    r_correct = r_correct.float()
    r_process = r_process.float().clamp(-1, 1)
    r_adv = (p_pass.float() - 0.5).clamp(-0.5, 0.5)
    r_eff = _torch_compute_efficiency_reward(cycle_counts, cfg) * r_correct
    r_trunc = truncated.float() * cfg.w_truncation
    return (
        cfg.w_correct * r_correct
        + cfg.w_process * r_process
        + cfg.w_adversarial * r_adv
        + cfg.w_efficiency * r_eff
        + r_trunc
    )


def _torch_grpo_advantages(rewards, groups=None, sigma_eps=1e-6):
    import torch
    rewards = rewards.float()
    if groups is None:
        return rewards - rewards.mean()
    groups = groups.long()
    # scatter-add per group: sum and count, then index-gather the mean
    n_groups = int(groups.max().item()) + 1
    gsum = torch.zeros(n_groups, dtype=rewards.dtype, device=rewards.device)
    gcnt = torch.zeros(n_groups, dtype=torch.int64, device=rewards.device)
    gsum.index_add_(0, groups, rewards)
    gcnt.index_add_(0, groups, torch.ones_like(groups))
    means = gsum / gcnt.clamp_min(1).to(rewards.dtype)
    return rewards - means[groups]
