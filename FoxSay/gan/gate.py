"""
ACT-style halting gate + latent loop machinery for the internal verification loop.

Design (Universal Transformer / Adaptive Computation Time lineage, adapted):
- The transformer trunk runs up to `loop_max_cycles` times over the same
  position (weight sharing across cycles = recurrence depth).
- Cycle 1 is the "commit" pass: it writes K/V projections (training in-tensor,
  decode into the KV cache). Cycles >= 2 are "refinement" passes: attention is
  ALWAYS computed against the frozen cycle-1 K/V, while the MLP/residual stream
  keeps being refined. No new tokens are generated: this is thinking in latent
  space, not test-time token spam.
- A small gate (per-cycle scalar biases + a linear readout on the cycle state)
  produces p_stop per cycle. Training uses the ACT soft mixture (fully
  differentiable). Inference uses hard halting (stop refining when p_stop >=
  threshold), so untrained gates halt at cycle 1 = exactly the old model.
- `loop_mix_lambda` (zero-init) scales how much the recurrent mixture may
  deviate from the cycle-1 state. At zero the forward pass is bit-identical to
  the original model, so old checkpoints load and behave identically until
  training moves the lambda.

All parameters live on the GPT model (checkpointed with it). The neutral/
initial values are centralized here in NEUTRAL() so that both GPT.init_weights()
and checkpoint_manager._patch_missing_keys() write identical tensors when
grafting the loop onto an old checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Gate biases: cycle 1 strongly biased to halt (p_stop ~ 0.95), later cycles
# biased to continue only if the readout learns to say so.
GATE_HALT_BIAS_FIRST = 3.0
GATE_HALT_BIAS_LATER = -3.0


class _CastingLinear(nn.Linear):
    """nn.Linear that casts its weight to the input dtype in forward.

    Same semantics as gpt.Linear: under FSDP fp64 masters the materialized
    master copy may be fp64 while activations run in the compute dtype (fp32/
    bf16); casting at matmul time keeps the loop gate usable in every dtype
    regime without duplicating weights."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype))


def neutral_state(n_embd: int, max_cycles: int):
    """The exact tensor values that make the loop a no-op (bit-identical to
    the original model). Shared source of truth for init and checkpoint patching."""
    return {
        "readout.weight": torch.zeros(1, n_embd),
        "readout.bias": torch.zeros(1),
        "cycle_biases": torch.cat([
            torch.full((1,), GATE_HALT_BIAS_FIRST),
            torch.full((max(0, max_cycles - 1),), GATE_HALT_BIAS_LATER),
        ]),
        "adapter.weight": torch.zeros(n_embd, n_embd),
        "adapter.bias": torch.zeros(n_embd),
        "mix_lambda": torch.zeros(1),
    }


class LoopGate(nn.Module):
    """Halting policy + cycle adapter + mixture scalar for the internal loop.

    Parameters (all checkpointed with the GPT model):
    - readout: Linear(n_embd -> 1) on mean-pooled norm(x_cycle)
    - cycle_biases: per-cycle learnable halting bias (max_cycles,)
    - adapter: Linear(n_embd -> n_embd), zero-init; injected between cycles so
      later cycles can learn a recurrence-specific transform
    - mix_lambda: zero-init scalar; gates how far the refined mixture may move
      away from the committed cycle-1 state
    """

    def __init__(self, n_embd: int, max_cycles: int):
        super().__init__()
        assert max_cycles >= 1
        self.n_embd = n_embd
        self.max_cycles = max_cycles
        self.readout = _CastingLinear(n_embd, 1, bias=True)
        self.cycle_biases = nn.Parameter(torch.empty(max_cycles))  # real init in init_weights / patch
        self.adapter = _CastingLinear(n_embd, n_embd, bias=True)
        self.mix_lambda = nn.Parameter(torch.empty(1))

    @torch.no_grad()
    def init_weights(self):
        """Real initialization (runs outside meta-device context)."""
        st = neutral_state(self.n_embd, self.max_cycles)
        self.readout.weight.copy_(st["readout.weight"])
        self.readout.bias.copy_(st["readout.bias"])
        self.cycle_biases.copy_(st["cycle_biases"])
        self.adapter.weight.copy_(st["adapter.weight"])
        self.adapter.bias.copy_(st["adapter.bias"])
        self.mix_lambda.copy_(st["mix_lambda"])

    def adapt(self, x):
        """Transform applied to the stream before entering cycle >= 2."""
        return self.adapter(x)

    def stop_probs(self, x, cycle_idx):
        """p_stop for one cycle. x: (B, T, C) stream state. Returns (B,) in [0,1]."""
        pooled = F.rms_norm(x, (x.size(-1),)).mean(dim=1)  # (B, C)
        logit = self.readout(pooled).squeeze(-1) + self.cycle_biases[cycle_idx].to(dtype=pooled.dtype)
        return torch.sigmoid(logit)

    @staticmethod
    def act_weights(stop_probs):
        """ACT cumulative halting weights from a list of per-cycle stop probs.

        stop_probs: list of (B,) tensors, one per executed cycle.
        Returns list of (B,) mixture weights that sum to 1 per row
        (remaining probability mass goes to the last cycle).
        """
        device = stop_probs[0].device
        remaining = torch.ones_like(stop_probs[0])
        weights = []
        for i, p in enumerate(stop_probs):
            if i == len(stop_probs) - 1:
                w = remaining  # clamp the remainder onto the last executed cycle
            else:
                w = p * remaining
                remaining = remaining - w
            weights.append(w)
        return weights

    def mixture(self, cycle_states, stop_probs):
        """Soft mixture used in training: x = sum_c w_c * x_c (ACT relaxation)."""
        ws = self.act_weights(stop_probs)
        acc = None
        for w, x in zip(ws, cycle_states):
            acc = w.unsqueeze(-1).unsqueeze(-1) * x if acc is None else acc + w.unsqueeze(-1).unsqueeze(-1) * x
        return acc

    def blend(self, x_commit, x_refined):
        """Apply the zero-init mix: out = x_commit + lambda * (x_refined - x_commit).
        At init (lambda=0) this is EXACTLY x_commit -> bit-identical old model."""
        lam = self.mix_lambda.to(x_commit.dtype)
        return x_commit + lam * (x_refined - x_commit)
