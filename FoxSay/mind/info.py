"""
info.py -- cross-modal mutual information objective (InfoNCE).

The modality router currently classifies text/audio/image/video by a linear
head; mutual information gives the missing contrastive shadow: maximize
I(z_text ; z_other) via the InfoNCE lower bound

    L = - E[ log  exp(sim(z_i, z_i^+)) /
                sum_j exp(sim(z_i, z_j)) ]

so the four modality streams converge to a shared information geometry
instead of just being separable.  This plugs directly into the
SuperWindow router features (FoxSay/modality_window.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["info_nce", "ModalityContrast"]


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1):
    """Symmetric InfoNCE between two modality feature sets (B, C) each."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature        # (B, B)
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss


class ModalityContrast:
    """Contrastive coupling for the modality router's per-modality features.

    collect(modality, z) buffers the pooled feature; contrast() returns the
    sum of InfoNCE losses across all modality pairs seen in the batch --
    the router learns a shared information geometry.
    """

    def __init__(self, modalities=("text", "audio", "image", "video"),
                 temperature: float = 0.1):
        self.modalities = list(modalities)
        self.temperature = temperature
        self._buf = {}

    def collect(self, modality: str, z: torch.Tensor):
        self._buf[modality] = z.detach()

    def contrast(self):
        keys = [m for m in self.modalities if m in self._buf]
        if len(keys) < 2:
            return None
        total = None
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                l = info_nce(self._buf[keys[i]], self._buf[keys[j]], self.temperature)
                total = l if total is None else total + l
        self._buf.clear()
        return total
