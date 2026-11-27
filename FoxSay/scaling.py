"""Architecture-only scale planning; no large tensors are allocated."""
from __future__ import annotations

import torch
from FoxSay.gpt import GPT, GPTConfig
from FoxSay.moe import SparseMoE
from FoxSay.right_brain import RightBrain, mirror_left

__all__ = ["trillion_dual_configs", "dual_parameter_report"]


def trillion_dual_configs(sequence_len: int = 2048, vocab_size: int = 32768):
    """Approximately 1T total parameters with Top-1 experts in both brains.

    This is a distributed construction plan, not a claim that a single GPU can
    initialize or train it. Production use still requires expert-parallel
    dispatch and enough FSDP ranks; this repository's FSDP shards storage but
    currently broadcasts every expert selected by a block's pre-hook.
    """
    left = GPTConfig(
        sequence_len=sequence_len, vocab_size=vocab_size,
        n_layer=64, n_embd=8192, n_head=64, n_kv_head=8,
        moe_num_experts=14, moe_expert_ratio=4.0, moe_top_k=1,
        moe_every=1, loss_chunk_size=128,
    )
    right = mirror_left(left, ssm_state=64, conv_kernel=7, dt=0.1)
    return left, right


def _report(model):
    total = sum(p.numel() for p in model.parameters())
    inactive = sum(m.parameter_report()["total"] - m.parameter_report()["active"]
                   for m in model.modules() if isinstance(m, SparseMoE))
    return {"total": total, "active_upper_bound": total - inactive}


def dual_parameter_report(left_config=None, right_config=None) -> dict:
    """Count parameters on the meta device; safe for trillion-scale configs."""
    if left_config is None or right_config is None:
        left_config, right_config = trillion_dual_configs()
    with torch.device("meta"):
        left, right = GPT(left_config), RightBrain(right_config)
    l, r = _report(left), _report(right)
    return {
        "left": l, "right": r,
        "total": l["total"] + r["total"],
        "active_upper_bound": l["active_upper_bound"] + r["active_upper_bound"],
    }
