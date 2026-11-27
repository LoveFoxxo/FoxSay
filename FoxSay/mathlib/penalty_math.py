"""
penalty_math.py -- vectorized penalty terms on top of mathlib.

Collects every penalty in the GAN verification loop into one place and
rewrites each with BLAS/LAPACK-backed reductions instead of python loops:

  * truncation penalty      (token cap hit without finishing)
  * overthinking penalty    (cycle overuse slope from reward_math)
  * verifier rank penalty   (margin ranking loss, matrix form)
  * token-validity scaling  (DAPO normalization helper)

Signatures mirror the pieces they replace in FoxSay/gan/reward.py and
FoxSay/gan/verifier.py so the gan package can delegate directly.
"""

from __future__ import annotations

import numpy as np

from .reward_math import compute_efficiency_reward, compute_truncation_status

__all__ = [
    "truncation_penalty", "overthinking_penalty", "verifier_rank_loss",
    "token_validity_scale",
]


def truncation_penalty(generated_tokens, cfg, eos_ids=None):
    """(B,) penalty = w_truncation * truncated, fully vectorized."""
    truncated = compute_truncation_status(generated_tokens, cfg, eos_ids)
    return truncated.astype(np.float64) * cfg.w_truncation


def overthinking_penalty(cycle_counts, cfg):
    """(B,) non-positive slope for cycling beyond the fast target.

    This is exactly the negative part of compute_efficiency_reward; keeping
    it separate lets callers combine it with the correctness multiplier.
    """
    return compute_efficiency_reward(cycle_counts, cfg)


def verifier_rank_loss(pos_logit, neg_logit, rank_margin=0.5):
    """Margin ranking loss in matrix form (batch of scalar logits).

    loss = mean(max(0, margin - (pos - neg)))  -- replaces VerifierLoss.
    pos_logit/neg_logit: (B,) or (B, 1).
    """
    p = np.asarray(pos_logit, dtype=np.float64).reshape(-1)
    n = np.asarray(neg_logit, dtype=np.float64).reshape(-1)
    diff = p - n
    hinge = np.maximum(0.0, rank_margin - diff)
    return float(hinge.mean())


def token_validity_scale(mask):
    """1 / valid_token_count per sample (DAPO scaling), vectorized."""
    mask = np.asarray(mask, dtype=np.float64)
    return 1.0 / np.clip(mask.sum(axis=1), 1, None)


def combined_penalty(
    r_correct, r_process, p_pass, cycle_counts, truncated, cfg,
):
    """All non-outcome penalty components of shape_reward summed.

    Returns the (B,) tensor of w_process*r_process + w_adversarial*r_adv
    + w_efficiency*r_eff + r_trunc -- the terms that can go negative.
    """
    r_process = np.clip(np.asarray(r_process, dtype=np.float64), -1, 1)
    r_adv = np.clip(np.asarray(p_pass, dtype=np.float64) - 0.5, -0.5, 0.5)
    r_eff = compute_efficiency_reward(cycle_counts, cfg) * np.asarray(
        r_correct, dtype=np.float64
    )
    r_trunc = np.asarray(truncated, dtype=np.float64) * cfg.w_truncation
    return (
        cfg.w_process * r_process
        + cfg.w_adversarial * r_adv
        + cfg.w_efficiency * r_eff
        + r_trunc
    )
