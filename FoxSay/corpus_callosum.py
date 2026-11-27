"""
corpus_callosum.py -- the bridge that connects the two hemispheres at the
inference-engine level.

Left brain (GPT) produces logits; the right brain (HiPPO temporal hemisphere)
re-reads the left brain's final hidden state and contributes a logit offset
through a tiny linear readout. The bridge is the ENGINE's fusion point: it
owns neither brain, just couples them.

Robustness contract (same zero-init philosophy as LoopGate / sub-GANs):
both projections are ZERO-initialized, so an untrained bridge returns
logits bit-identical to the legacy left-brain-only engine. The right brain
can only ever change the output after the bridge has been trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CorpusCallosum"]


class CorpusCallosum(nn.Module):
    """Fusion of left-brain logits with a right-brain logit offset.

    forward flow:
        r        = right_brain( to_right(left_hidden) )     (B, T, C)
        logits   = left_logits + readout(r)                 (B, T, V)

    init_weights() zeroes both projections: before training, the right brain
    is mathematically invisible (legacy bit-exactness).
    """

    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.to_right = nn.Linear(n_embd, n_embd, bias=False)
        self.readout = nn.Linear(n_embd, vocab_size, bias=False)

    def init_weights(self):
        nn.init.zeros_(self.to_right.weight)
        nn.init.zeros_(self.readout.weight)

    def fuse(self, left_logits: torch.Tensor, history_hidden: torch.Tensor,
             right_brain) -> torch.Tensor:
        """history_hidden: (B, Th, C) full causal history, Th >= T of logits.

        The right brain convolves the WHOLE history (its temporal channels see
        context), then only the last T outputs (aligned with the current token
        window) contribute the logit offset -- causal by construction of the
        right brain's impulse-response convolution.
        """
        if left_logits.dim() not in (2, 3) or history_hidden.dim() not in (2, 3):
            raise ValueError("expected (B,V)/(B,T,V) logits and (B,C)/(B,T,C) history")
        squeezed = left_logits.dim() == 2
        h = history_hidden.unsqueeze(1) if history_hidden.dim() == 2 else history_hidden
        if h.size(-1) != self.to_right.in_features:
            raise ValueError("left/right embedding widths do not match")
        T = left_logits.size(1) if left_logits.dim() == 3 else 1
        if h.size(1) < T:
            raise ValueError("right-brain history cannot be shorter than the logit window")
        r = right_brain(F.linear(h, self.to_right.weight.to(h.dtype)))
        offset = F.linear(r, self.readout.weight.to(r.dtype))
        offset = offset[:, -T:, :]
        if squeezed:
            offset = offset.squeeze(1)
        return left_logits + offset.to(left_logits.dtype)
