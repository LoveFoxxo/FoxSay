"""
memory.py -- modern Hopfield associative memory: remembering as gradient
descent into an attractor of an energy landscape.

Modern Hopfield energy (Ramsauer et al. 2020):
    E(x) = - lse(beta X^T x) + 1/2 x^T x + beta^{-1} log N
whose stationary points are attractors; one update step
    x <- X softmax(beta X^T x)
is exactly a transformer attention over stored patterns X -- so the
attention head the model already has IS the retrieval operator.  This
module makes the attractor structure explicit: an energy, a retrieval
step, and an attractor projection used inside the thinking loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["HopfieldMemory"]


class HopfieldMemory(nn.Module):
    """Store key patterns; retrieve by iterating the Hopfield update."""

    def __init__(self, n_embd: int, n_slots: int = 64, beta: float = 8.0):
        super().__init__()
        self.beta = beta
        self.n_slots = n_slots
        self.register_buffer("slots", torch.randn(n_slots, n_embd) * 0.02)
        self.register_buffer("age", torch.zeros(n_slots))
        self.used = 0

    def init_weights(self):
        """Re-initialize buffers on the real device (meta-safe)."""
        self.slots.normal_(0, 0.02)
        self.age.zero_()

    def energy(self, x: torch.Tensor):
        """E(x) = -lse(beta X^T x) + 0.5||x||^2 + beta^-1 log N."""
        sim = self.beta * (self.slots @ x.unsqueeze(-1)).squeeze(-1)   # (N,)
        lse = torch.logsumexp(sim, dim=0)
        return -lse + 0.5 * (x ** 2).sum() + torch.log(
            torch.tensor(self.slots.shape[0], dtype=x.dtype, device=x.device)) / self.beta

    def retrieve(self, x: torch.Tensor, iters: int = 1):
        """x: (B, C) -> (B, C) after `iters` Hopfield update steps."""
        for _ in range(iters):
            sim = self.beta * (x @ self.slots.T)                  # (B, N)
            att = torch.softmax(sim, dim=-1)                      # (B, N)
            x = att @ self.slots                                  # (B, C) update
        return x

    def store(self, x: torch.Tensor):
        """Write new patterns into the slots (cyclic overwrite by age)."""
        with torch.no_grad():
            for v in x.detach():
                idx = self.used % self.slots.shape[0]
                self.slots[idx] = v / (v.norm() + 1e-8)
                self.age[idx] = 0.0
                self.used += 1
            self.age += 1.0

    def energy_gradient_step(self, x: torch.Tensor, lr: float = 0.1):
        """One explicit gradient-descent step on E -- guaranteed to lower E
        for small lr (the attractor dynamics in exact form)."""
        with torch.enable_grad():
            xx = x.detach().clone().requires_grad_(True)
            E = self.energy(xx.mean(0)) if xx.dim() > 1 else self.energy(xx)
            E.backward()
            return x - lr * xx.grad.detach()

    def attractor_loss(self, x: torch.Tensor):
        """Pull loop states toward stored attractors (regularizer)."""
        return self.energy(x.detach()).mean()
