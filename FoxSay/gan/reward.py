"""
Reward shaping for the internal verification loop - replaces the old
"binary GSM8K reward + 256-token truncation = 0 reward" regime.

The old objective punished thinking: any rollout clipped at 256 tokens got
reward 0, and nothing rewarded correct intermediate reasoning, so the policy
collapsed to "answer fast, think little". The new reward is a mainstream
composite (GRPO-style group-relative advantage + PRM process reward +
GAN-adversarial score + efficiency shaping):

    r_total = w_correct   * r_correct        (task outcome, binary)
            + w_process   * r_process         (verifier PRM chunk scores, tanh)
            + w_adv       * r_adversarial     (sigmoid(turn_logit) - 0.5)
            + w_efficiency* r_efficiency      (cycle-efficiency bonus: reward
                                                thinking only when needed)
            + w_truncation* r_truncation      (penalty ONLY if truly truncated
                                                by max_tokens with no answer,
                                                never zeroing the process part)

r_correct is computed by the caller (the task's `reward()`); everything else
comes from the gan package. The same signal is used at inference by the
OutputGate (accept iff P(pass) >= threshold), so the reward the policy is
trained to maximize is exactly the criterion its outputs are gated on -
the definition of an aligned train/inference objective.

Since the mathlib rewrite (see FoxSay/mathlib), every function below is a
thin torch-native delegation to the vectorized kernels of reward_math /
penalty_math, whose reductions run on the compiled OpenBLAS/LAPACK shipped
with NumPy.  No python loop remains on the training hot path.

All functions are pure & batched; safe to call under torch.no_grad().
"""

from dataclasses import dataclass

import torch

from FoxSay.mathlib.reward_math import (
    _torch_grpo_advantages,
    _torch_compute_efficiency_reward,
    _torch_shape_reward,
    _torch_truncation_status,
)


@dataclass
class RewardConfig:
    # component weights (tuned to keep |r| <= ~1.5 so GRPO advantages stay sane)
    w_correct: float = 1.0
    w_process: float = 0.25
    w_adversarial: float = 0.5
    w_efficiency: float = 0.15
    w_truncation: float = -0.25
    # efficiency shaping parameters
    max_cycles_budget: int = 4      # latent cycles the loop may use
    target_cycles_fast: int = 1     # cycles considered "fast/cheap"
    # truncation: max NEW tokens before "truncated" (None = no cap)
    truncation_limit: int = None
    # answer format requirement (e.g. "#### N" for GSM8K) - None disables
    answer_marker: str = None


def compute_efficiency_reward(cycle_counts, cfg: RewardConfig):
    """Cycle-efficiency shaping, delegated to mathlib (vectorized)."""
    return _torch_compute_efficiency_reward(cycle_counts, cfg)


def compute_truncation_status(generated_tokens, cfg: RewardConfig, eos_ids=None):
    """(B,) bool: truly truncated (hit the token cap without finishing).

    Delegated to mathlib: lengths/finish flags are computed vectorized,
    only the ragged per-row EOS check stays scalar.
    """
    return _torch_truncation_status(generated_tokens, cfg, eos_ids)


def shape_reward(
    r_correct: torch.Tensor,
    r_process: torch.Tensor,
    p_pass: torch.Tensor,
    cycle_counts: torch.Tensor,
    truncated: torch.Tensor,
    cfg: RewardConfig,
):
    """Composite reward. All inputs (B,); r_correct in {0,1} (or [-1,1] for
    graded tasks), r_process/p_pass from the verifier, cycle_counts int,
    truncated bool. Returns (B,) shaped reward. Delegated to mathlib."""
    return _torch_shape_reward(
        r_correct, r_process, p_pass, cycle_counts, truncated, cfg
    )


def grpo_advantages(rewards: torch.Tensor, groups: torch.Tensor = None, sigma_eps=1e-6):
    """GRPO-style group-relative advantage, now loop-free: group means are
    scatter-added with index_add_ in one pass (mathlib kernel)."""
    return _torch_grpo_advantages(rewards, groups, sigma_eps=sigma_eps)


def token_level_advantages(advantages: torch.Tensor, mask: torch.Tensor):
    """DAPO-style: broadcast sequence-level advantage to tokens, normalized by
    the number of VALID (sampled) tokens per sample. advantages (B,), mask
    (B, T) with 1 = sampled (trainable) token. Returns (B, T)."""
    B, T = mask.shape
    per_tok = advantages.unsqueeze(1).expand(B, T).to(mask.dtype) * mask
    num_valid = mask.sum(dim=1).clamp(min=1)
    return per_tok / num_valid.unsqueeze(1)
