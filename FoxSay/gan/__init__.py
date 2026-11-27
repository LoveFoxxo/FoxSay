"""Adversarial verification loop package (internal latent recurrence + GAN gate).

Components
==========
- gate.LoopGate        ACT halting gate + adapter + mixture scalar (lives on the GPT)
- verifier.VerifierModel  discriminator / process-reward model
- verifier.OutputGate     inference-time final-output threshold gate
- verifier.VerifierTrainer LSGAN-style adversarial trainer for the verifier
- reward.RewardConfig     composite reward configuration
- reward.shape_reward     outcome + process + adversarial + efficiency + truncation
- reward.grpo_advantages  group-relative advantage (GRPO/DAPO flavored)
"""

from FoxSay.gan.gate import LoopGate, neutral_state, GATE_HALT_BIAS_FIRST, GATE_HALT_BIAS_LATER
from FoxSay.gan.verifier import (
    GPTConfigLite,
    VerifierModel,
    VerifierLoss,
    OutputGate,
    VerifierTrainer,
    collate_turns,
)
from FoxSay.gan.reward import (
    RewardConfig,
    shape_reward,
    grpo_advantages,
    token_level_advantages,
    compute_efficiency_reward,
    compute_truncation_status,
)

__all__ = [
    "LoopGate", "neutral_state", "GATE_HALT_BIAS_FIRST", "GATE_HALT_BIAS_LATER",
    "GPTConfigLite", "VerifierModel", "VerifierLoss", "OutputGate", "VerifierTrainer",
    "collate_turns",
    "RewardConfig", "shape_reward", "grpo_advantages", "token_level_advantages",
    "compute_efficiency_reward", "compute_truncation_status",
]
