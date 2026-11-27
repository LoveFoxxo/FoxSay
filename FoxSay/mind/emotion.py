"""
emotion.py -- the affective dynamics of the mind.

Emotion is modelled as the integrated temporal derivative of reward -- a
signal, not a static scalar.  Four coupled mathematical objects:

  * Critic (TD-lambda):  V(s) value network with eligibility traces.
        delta_t = r_t + gamma V(s_{t+1}) - V(s_t)          (TD error,
        the computational neuroscience model of dopamine)
        E_{t+1} = gamma lambda E_t + grad V(s_t)           (trace)
        V <- V + alpha * delta_t * E_t

  * EmotionState: valence/arousal as a complex variable z = rho e^{i phi}.
        arousal  rho <- (1-l) rho + l |delta| + sigma sqrt(1-l^2) eps
                 (Ornstein-Uhlenbeck: mean-reverting, noisy)
        valence  phi <- phi + k * sign(delta) * step       (sign of surprise)
        Outputs: valence = cos(phi), arousal = rho.

  * FreeEnergy (Friston): variational free energy as surprise,
        F = ||x - x_hat||^2 / (2 sigma^2) + 0.5 (||z||^2 - log sigma^2)
        one gradient step on F drives the emotion state (emotion as the
        gradient of expected surprise).

  * Homeostat (PID): homeostasis loop with differentiable gains.
        e = setpoint - measurement
        u = Kp e + Ki integral(e) + Kd de/dt
        w_eff = w0 (1 + tanh(u))     -- modulates RewardConfig weights live,
        so the reward/penalty weights become part of the mind's dynamics.

The emotion state modulates generation temperature, the LoopGate stop
threshold, and an attention gain -- exactly the interfaces the loop already
exposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = ["Critic", "EmotionState", "FreeEnergy", "Homeostat"]


class Critic(nn.Module):
    """TD-lambda value critic over pooled hidden states."""

    def __init__(self, n_embd: int, hidden: int = 128, gamma: float = 0.9,
                 lam: float = 0.7, lr: float = 1e-3):
        super().__init__()
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.net = nn.Sequential(nn.Linear(n_embd, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        self.trace = None  # eligibility trace

    def value(self, s):
        return self.net(s.detach()).squeeze(-1)

    def td_error(self, s, r, s_next):
        v = self.value(s)
        with torch.no_grad():
            v_next = self.value(s_next) if s_next is not None else 0.0
        return r + self.gamma * v_next - v

    def update(self, s, r, s_next):
        """One TD-lambda step. Returns delta (the dopamine signal)."""
        delta = self.td_error(s, r, s_next)
        v = self.value(s)
        if self.trace is None or self.trace.shape != s.shape:
            self.trace = torch.zeros_like(s)
        self.trace = self.gamma * self.lam * self.trace + s.detach()
        # eligibility-weighted value-gradient update (linearized)
        target = (v + delta).detach()
        loss = 0.5 * ((v - target) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for p in self.net.parameters():
                if p.grad is not None:
                    p -= self.lr * p.grad
        self.net.zero_grad()
        return delta


@dataclass
class EmotionState:
    """Valence/arousal as a complex variable on the unit circle x ray."""

    n_embd: int
    decay: float = 0.9          # mean-reversion rate (OU lambda)
    noise: float = 0.05         # OU diffusion sigma
    phi_step: float = 0.15      # valence phase step per surprise sign

    def __post_init__(self):
        self.rho = 0.0        # arousal magnitude (python float: meta-safe)
        self.phi = 0.0        # valence phase

    @property
    def valence(self):
        return math.cos(self.phi)

    @property
    def arousal(self):
        return self.rho

    def update(self, delta: torch.Tensor):
        """OU step driven by the TD error (delta = dopamine signal)."""
        d = float(delta.detach().float().mean())
        l = self.decay
        # discrete OU: rho <- (1-l) rho + l |d| + sigma sqrt(1-l^2) eps
        eps = float(torch.randn(()))
        self.rho = (1 - l) * self.rho + l * abs(d) + self.noise * math.sqrt(1 - l * l) * eps
        self.phi = self.phi + self.phi_step * (1 if d > 0 else -1) * (0.3 + self.rho)
        return self

    # -- modulators (the interfaces the loop exposes) ----------------------
    def temperature_scale(self):
        """High arousal -> lower temperature (deliberate), calm -> explore."""
        return 1.0 / (1.0 + 2.0 * self.arousal)

    def stop_threshold_shift(self):
        """Positive valence -> willingness to keep thinking (lower stop bar)."""
        return -0.1 * self.valence

    def attention_gain(self):
        """Valence gates how strongly the stream attends."""
        return 1.0 + 0.2 * self.valence


class FreeEnergy:
    """Variational free energy; one gradient step = emotion update."""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.log_sigma = math.log(sigma)  # python float: meta-safe

    def __call__(self, x, x_hat, z):
        """F = prediction error term + Gaussian prior KL (z = emotion state)."""
        pe = ((x - x_hat) ** 2).sum(dim=-1) / (2 * self.sigma ** 2)
        kl = 0.5 * ((z ** 2).sum(dim=-1) - self.log_sigma * z.shape[-1])
        return (pe + kl).mean()

    def descend(self, x, x_hat, z, lr: float = 0.01):
        """z <- z - lr * dF/dz  (emotion follows the free-energy gradient)."""
        with torch.enable_grad():
            zz = z.detach().clone().requires_grad_(True)
            F = self.__call__(x.detach(), x_hat.detach(), zz)
            F.backward()
            z = z - lr * zz.grad
        return z


@dataclass
class Homeostat:
    """PID homeostasis: makes reward/penalty weights differentiable."""

    setpoint: float = 0.0
    Kp: float = 0.5
    Ki: float = 0.05
    Kd: float = 0.1

    def __post_init__(self):
        self.integral = 0.0
        self.prev_e = 0.0

    def __call__(self, measurement: float):
        e = self.setpoint - measurement
        self.integral = 0.9 * self.integral + e        # leaky integral
        de = e - self.prev_e
        self.prev_e = e
        u = self.Kp * e + self.Ki * self.integral + self.Kd * de
        return math.tanh(u)                            # bounded gain in (-1, 1)

    def modulate(self, w0: float, measurement: float):
        """w_eff = w0 (1 + u): the homeostat scales the reward weight."""
        return w0 * (1.0 + self(measurement))
