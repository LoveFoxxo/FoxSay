"""
spectral.py -- random-matrix-theory / spectral regularization framework.

Every subsystem of the mind (trunk, each sub-GAN, each verifier) gets its OWN
spectral framework -- that is the "multiple spectral frameworks" design:
training injects one spectral loss per subsystem, each with its own target
spectral radius, so no single subsystem can dominate the joint dynamics.

Mathematical toolbox implemented here:

  * power iteration for the spectral norm  ||W||_2 = sigma_max(W)
  * sigma-reparametrization:  W = W_raw * (sigma_target / ||W_raw||_2)
    (the standard stable-training trick, in exact form)
  * RMT-informed spectral init:
      - circular law: an n x n matrix with iid entries of std s has
        spectral radius ~ s * sqrt(n)
      - so given a target rho, entries are initialized with
        std = rho / sqrt(n), then the matrix is rescaled exactly.
  * residual-Lipschitz regularization:
      a residual stream f(x) = x + F(x) is a contraction iff
      ||dF/dx||_2 <= rho < 1.  We bound ||dF/dx||_2 by the product of the
      spectral norms of the layer weights (submultiplicativity), and add the
      hinge penalty  sum_i max(0, ||W_i||_2^2 - rho_i^2)^2.
  * per-subsystem ensemble: each named system carries its own (rho, sigma)
    budget; `spectral_loss()` sums the hinge penalties and `project()` snaps
    every weight to its budget (cheap, deterministic, no JVP needed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "power_iteration", "spectral_norm", "spectral_init", "sigma_reparam",
    "SpectralSystem", "SpectralEnsemble", "SpectralHub",
]


# --------------------------------------------------------------------------
# core spectral primitives
# --------------------------------------------------------------------------

def power_iteration(W: torch.Tensor, u: torch.Tensor = None, iters: int = 10):
    """Estimate sigma_max(W) via the power method (Hutchinson-free).

    Returns (sigma, u, v) with ||W v|| = sigma, u = W v / sigma.
    """
    work_dtype = (W.dtype if W.dtype in (torch.float32, torch.float64) else torch.float32)
    Wd = W.detach().to(work_dtype)
    m, n = Wd.shape[-2], Wd.shape[-1]
    if u is None:
        u = torch.randn(m, device=Wd.device, dtype=Wd.dtype)
    u = u / (u.norm() + 1e-12)
    for _ in range(iters):
        v = torch.einsum("ij,i->j", Wd, u)          # W^T u  (dim n)
        v = v / (v.norm() + 1e-12)
        u = torch.einsum("ij,j->i", Wd, v)          # W v    (dim m)
        sigma = u.norm()
        u = u / (sigma + 1e-12)
    return sigma, u, v


def spectral_norm(W: torch.Tensor, iters: int = 10):
    return power_iteration(W, iters=iters)[0]


def spectral_init(W: torch.Tensor, rho: float):
    """RMT-informed init: scale a weight to spectral radius rho exactly.

    Circular law: for iid entries with std s, sigma_max ~ s sqrt(n).  We do
    the exact version: rescale W to ||W||_2 = rho (not just the asymptotic
    law), which is what the law says it should concentrate around.
    """
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and non-negative")
    with torch.no_grad():
        # Exact SVD removes small-matrix initialization flake; large matrices
        # retain the memory-efficient power method with more iterations.
        if W.numel() <= 16384:
            work = W.detach().to(W.dtype if W.dtype in (torch.float32, torch.float64) else torch.float32)
            s = torch.linalg.svdvals(work)[0]
        else:
            s = spectral_norm(W, iters=50)
        if s > 0:
            W.mul_((rho / s).to(W.dtype))
    return W


def sigma_reparam(W_raw: torch.Tensor, sigma_target: float):
    """sigma-reparametrization: returns W with ||W||_2 = sigma_target.

    The raw parameter keeps its direction; the magnitude is snapped each
    step so gradient updates cannot inflate the spectrum.  A 0.1% margin
    absorbs the power-iteration estimation error.
    """
    if not math.isfinite(sigma_target) or sigma_target < 0:
        raise ValueError("sigma_target must be finite and non-negative")
    with torch.no_grad():
        if W_raw.numel() <= 16384:
            work = W_raw.detach().to(W_raw.dtype if W_raw.dtype in (torch.float32, torch.float64) else torch.float32)
            s = torch.linalg.svdvals(work)[0]
        else:
            s = spectral_norm(W_raw, iters=50)
        if s > sigma_target and s > 0:
            W_raw.mul_((sigma_target / s * 0.999).to(W_raw.dtype))
    return W_raw


# --------------------------------------------------------------------------
# per-subsystem spectral framework
# --------------------------------------------------------------------------

@dataclass
class SpectralSystem:
    """One spectral framework for one named subsystem (e.g. 'trunk', 'sg0').

    weights: list of (name, param) 2D+ tensors to control
    rho:     target spectral radius per weight (residual contraction budget)
    sigma:   hard cap ||W||_2 <= sigma (sigma-reparam after each step)
    lam:     penalty weight of the hinge spectral loss for THIS system
    """

    name: str
    weights: list = field(default_factory=list)
    rho: float = 1.0
    sigma: float = 1.2
    lam: float = 1e-3

    def register(self, module):
        """Register all Linear/Embedding weights of a module."""
        import torch.nn as nn
        for n, m in module.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                self.weights.append((f"{self.name}/{n}", m.weight))
        return self

    def hinge(self):
        """sum_i max(0, ||W_i||_2^2 - rho_i^2)^2  (contraction budget)."""
        loss = torch.zeros((), device=self.weights[0][1].device) if self.weights else None
        if loss is None:
            return None
        for _, w in self.weights:
            if w.dim() < 2:
                continue
            s = spectral_norm(w, iters=5)
            excess = s * s - self.rho * self.rho
            if excess > 0:
                loss = loss + excess * excess
        return self.lam * loss

    def project(self):
        """Snap every weight to its hard spectral cap (sigma-reparam)."""
        for _, w in self.weights:
            if w.dim() >= 2:
                sigma_reparam(w, self.sigma)


class SpectralEnsemble:
    """Multiple spectral frameworks: one per subsystem of the mind.

    Usage:
        ens = SpectralEnsemble()
        ens.add(SpectralSystem("trunk", rho=0.95, sigma=1.1).register(model))
        for i, sg in enumerate(cluster.subgans.values()):
            ens.add(SpectralSystem(f"sg{i}", rho=0.5, sigma=0.6).register(sg))
        # in training loop:
        loss += ens.spectral_loss()
        ens.project()
    """

    def __init__(self):
        self.systems: dict = {}

    def add(self, system: SpectralSystem):
        self.systems[system.name] = system
        return self

    def spectral_loss(self):
        total = None
        for sys in self.systems.values():
            h = sys.hinge()
            if h is not None:
                total = h if total is None else total + h
        return total

    def project(self):
        for sys in self.systems.values():
            sys.project()

    def report(self):
        return {name: {n: float(spectral_norm(w, iters=5))
                       for n, w in sys.weights if w.dim() >= 2}
                for name, sys in self.systems.items()}


# --------------------------------------------------------------------------
# SpectralHub -- the dura mater: ONE shared spectral center that couples the
# two hemispheres during training so their spectra cannot drift apart
# (split-brain prevention at the level of the weight spectrum).
# --------------------------------------------------------------------------

class SpectralHub:
    """Central spectral node for the left and right brains.

    Each hemisphere keeps its own SpectralEnsemble (the "multiple spectral
    frameworks" design), but BOTH report into one hub that imposes:

      * paired spectral gap loss: for weights paired by name across brains,
            L_couple = sum_{(l,r)} || sigma_max(W_l) - sigma_max(W_r) ||^2
        so the two spectra co-move (mutual influence, like the dura mater
        mechanically linking the hemispheres);
      * a shared global spectral budget: the total spectral mass
            S = sum_i sigma_max(W_i)^2
        across BOTH brains is capped by a hinge, so neither brain can hoard
        spectral capacity.

    hub_loss() = sum of the individual ensemble hinge losses
               + lam_couple * L_couple + lam_budget * hinge(S - budget).
    """

    def __init__(self, lam_couple: float = 1e-2, lam_budget: float = 1e-3,
                 budget: float = None, couple_pairs: list = None):
        self.ensembles: dict = {}                    # 'left'/'right' -> SpectralEnsemble
        self.lam_couple = lam_couple
        self.lam_budget = lam_budget
        self.budget = budget                         # None -> auto from registrations
        # couple_pairs: list of (left_weight_name, right_weight_name) tuples;
        # when None, pairs are auto-matched by identical registered names.
        self.couple_pairs = couple_pairs

    def couple(self, side: str, ens):
        """Attach one hemisphere's spectral ensemble to the hub."""
        self.ensembles[side] = ens
        return self

    def _snap(self, weights):
        return {n: float(spectral_norm(w, iters=5)) for n, w in weights if w.dim() >= 2}

    def _pairs(self):
        if self.couple_pairs is not None:
            return self.couple_pairs
        # auto-match: identical names registered on both sides
        common = set()
        sides = list(self.ensembles.keys())
        if len(sides) >= 2:
            left, right = sides[0], sides[1]
            names_l = {n for ens in [self.ensembles[left]] for sys in ens.systems.values()
                       for n, _ in sys.weights}
            names_r = {n for ens in [self.ensembles[right]] for sys in ens.systems.values()
                       for n, _ in sys.weights}
            common = names_l & names_r
        return [(n, n) for n in sorted(common)]

    def coupling_loss(self):
        """Paired spectral gap loss across the two brains."""
        if len(self.ensembles) < 2:
            return None
        sides = list(self.ensembles.keys())
        snap_l, snap_r = {}, {}
        for sys in self.ensembles[sides[0]].systems.values():
            for n, w in sys.weights:
                if w.dim() >= 2:
                    snap_l[n] = float(spectral_norm(w, iters=5))
        for sys in self.ensembles[sides[1]].systems.values():
            for n, w in sys.weights:
                if w.dim() >= 2:
                    snap_r[n] = float(spectral_norm(w, iters=5))
        loss = torch.zeros(())
        count = 0
        for nl, nr in self._pairs():
            if nl in snap_l and nr in snap_r:
                gap = snap_l[nl] - snap_r[nr]
                loss = loss + gap * gap
                count += 1
        return self.lam_couple * loss if count else None

    def budget_loss(self):
        """Hinge on the total spectral mass across both brains."""
        total = 0.0
        n = 0
        for ens in self.ensembles.values():
            for sys in ens.systems.values():
                for _, w in sys.weights:
                    if w.dim() >= 2:
                        total += float(spectral_norm(w, iters=5)) ** 2
                        n += 1
        if n == 0:
            return None
        budget = self.budget if self.budget is not None else 2.0 * n
        excess = total - budget
        if excess <= 0:
            return None
        return self.lam_budget * excess * excess

    def hub_loss(self):
        """Total dura-mater loss = individual hinges + coupling + budget."""
        total = None
        for ens in self.ensembles.values():
            h = ens.spectral_loss()
            if h is not None:
                total = h if total is None else total + h
        for extra in (self.coupling_loss(), self.budget_loss()):
            if extra is not None:
                total = extra if total is None else total + extra
        return total

    def project(self):
        """Snap every brain's weights to their own hard spectral caps."""
        for ens in self.ensembles.values():
            ens.project()

    def report(self):
        out = {side: ens.report() for side, ens in self.ensembles.items()}
        coupl = self.coupling_loss()
        out["coupling_loss"] = None if coupl is None else float(coupl.detach())
        return out
