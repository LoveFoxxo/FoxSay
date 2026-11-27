"""
uncertainty.py -- Bayesian uncertainty shadows.

  * SnapshotEnsemble: SWA-style parameter snapshots; the predictive variance
    across snapshots is the deep-ensemble uncertainty proxy.  The variance
    is exactly what "hesitation / anxiety" should modulate.
  * TemperatureCalibration: proper-scoring-rule temperature (single scalar)
    fitted to minimize NLL on observed targets -- the calibrated model knows
    how sure it is, not just what it says.
  * MCDropout shadow: a Bernoulli gating layer that samples a different
    subnet per forward pass (dropout already lives in the model; here it is
    exposed as an explicit uncertainty interface).
"""

from __future__ import annotations

import copy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SnapshotEnsemble", "TemperatureCalibration", "MCShadow"]


class SnapshotEnsemble:
    """Cyclic parameter snapshots; variance across snapshots = uncertainty."""

    def __init__(self, model, max_snapshots: int = 4):
        self.model = model
        self.max_snapshots = max_snapshots
        self.snapshots = deque(maxlen=max_snapshots)

    def snapshot(self):
        self.snapshots.append({k: v.detach().clone()
                               for k, v in self.model.state_dict().items()})

    def predict_with_variance(self, fn, *args, **kwargs):
        """fn(model, *args) -> (B, ...) logits; returns (mean, variance)."""
        if not self.snapshots:
            out = fn(self.model, *args, **kwargs)
            return out, torch.zeros_like(out)
        outs = []
        for sd in self.snapshots:
            cur = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(sd)
            outs.append(fn(self.model, *args, **kwargs))
            self.model.load_state_dict(cur)
        stack = torch.stack(outs)
        return stack.mean(0), stack.var(0)

    def calibrate_variance_scale(self, targets, predictions):
        """Match ensemble std to observed error (simple moment match)."""
        err = ((predictions - targets) ** 2).mean().sqrt()
        std = self.predict_with_variance(
            lambda m: predictions)[1].mean().sqrt()
        return (err / (std + 1e-8)).item()


class TemperatureCalibration(nn.Module):
    """Single-parameter Platt/temperature scaling (proper scoring rule)."""

    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, logits):
        return logits / self.log_temp.exp()

    def fit(self, logits, targets, steps: int = 200, lr: float = 0.05):
        """Minimize NLL (the proper scoring rule) over the temperature."""
        opt = torch.optim.Adam([self.log_temp], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits.detach()), targets)
            loss.backward()
            opt.step()
        return float(self.log_temp.detach().exp())


class MCShadow(nn.Module):
    """Explicit MC-dropout shadow: Bernoulli gate with trainable rate."""

    def __init__(self, n_embd: int, p: float = 0.1):
        super().__init__()
        self.raw_p = nn.Parameter(torch.tensor(p).logit())
        self.scale = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        if self.training:
            p = torch.sigmoid(self.raw_p).detach()
            mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
            x = x * mask / (1 - p + 1e-8)     # inverted dropout (unbiased)
        return x * self.scale
