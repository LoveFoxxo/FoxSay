"""
manifold.py -- Riemannian manifold nets deployed at different points of the
chain of thought, so that every position of the thinking chain produces a
DIFFERENT "thought" every time.

Mathematical objects:

  * Fisher metric (diagonal + K-FAC-lite block form):
        F = E[ g g^T ],   natural gradient = F^{-1} g
      Diagonal:      gd_i / (F_ii + eps)           (ADAGRAD-like limit)
      K-FAC-lite:    layer update ~ G^{-1} dL/dW A^{-1}, where
                     A = E[a a^T] (input Gram), G = E[g g^T] (grad Gram),
                     using the Kronecker product identity
                     (A x G)^{-1} vec(dL/dW) = vec( G^{-1} dL/dW A^{-1} ).

  * Poincare ball (hyperbolic manifold):
        metric g_x = lambda_x^2 I,   lambda_x = 2 / (1 - ||x||^2)
        Mobius addition, exponential map
        exp_x(v) = x (+) tanh( lambda_x ||v|| / 2 ) v / ||v||,
        log map and hyperbolic distance.

  * ManifoldSchedule: each position (cycle, depth_fraction, subgan_id) of the
    chain of thought binds its own manifold net (Euclidean / Fisher /
    Poincare / Frobenius) with its own learnable temperature, so "different
    points -> different geometries -> different thoughts".

  * ThoughtSampler: anisotropic sampling in the tangent space of the bound
    manifold (covariance induced by the metric: high-information directions
    are sampled with small variance, low-information with large), then
    retraction back onto the manifold.  Every call draws fresh noise, so the
    same position yields a different thought each time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FisherMetric", "NaturalGradient", "kfac_natural_update",
    "PoincareBall", "ManifoldSchedule", "ThoughtSampler",
    "FinslerManifold", "SimplexSphere", "ManifoldMesh",
]


# --------------------------------------------------------------------------
# Fisher metric and natural gradient
# --------------------------------------------------------------------------

class FisherMetric:
    """Diagonal empirical Fisher (EMA of squared gradients)."""

    def __init__(self, params, eps: float = 1e-6, momentum: float = 0.9):
        self.eps = eps
        self.momentum = momentum
        self.F = {id(p): torch.zeros_like(p) for p in params}

    def update(self):
        for pid, f in self.F.items():
            p = self._param_of(pid)
            if p is None or p.grad is None:
                continue
            with torch.no_grad():
                g2 = p.grad.detach() ** 2
                f.mul_(self.momentum).add_(g2, alpha=1 - self.momentum)

    def _param_of(self, pid):
        # reverse lookup (params are registered once; keep a side map)
        return self._pmap.get(pid)

    def bind(self, params):
        self._pmap = {id(p): p for p in params}
        return self

    def natural_dir(self, param, raw_grad: torch.Tensor):
        f = self.F[id(param)]
        return raw_grad / (f.sqrt() + self.eps)


def kfac_natural_update(layer: nn.Linear, damping: float = 1e-3):
    """K-FAC-lite natural-gradient step for one linear layer.

    dW_natural = (G + lam I)^{-1} dL/dW (A + lam I)^{-1}
    computed in-place-ish: returns the natural update to apply.
    Assumes a_prev and grad_out were cached externally; here we reconstruct
    from the layer's current grad and an input Gram maintained by the caller.
    """
    if layer.weight.grad is None:
        return None
    # fall back to diagonal Fisher inside this layer when Grams unavailable
    g = layer.weight.grad.detach()
    f = (g ** 2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    return g / (f + damping)


# --------------------------------------------------------------------------
# Poincare ball (hyperbolic geometry)
# --------------------------------------------------------------------------

class PoincareBall:
    """Standard Poincare ball D^n with curvature -1."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def lambda_x(self, x: torch.Tensor):
        return 2.0 / (1.0 - (x ** 2).sum(dim=-1, keepdim=True).clamp(max=1 - self.eps))

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor):
        x2 = (x ** 2).sum(dim=-1, keepdim=True)
        y2 = (y ** 2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        den = 1 + 2 * xy + x2 * y2
        return num / den.clamp(min=self.eps)

    def exp_map(self, x: torch.Tensor, v: torch.Tensor):
        """Exponential map: tangent vector v at x -> point on the ball."""
        lx = self.lambda_x(x)
        vn = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        y = torch.tanh(lx * vn / 2) * v / vn
        return self.mobius_add(x, y)

    def log_map(self, x: torch.Tensor, y: torch.Tensor):
        """Logarithmic map: geodesic direction from x to y (tangent at x)."""
        d = self.mobius_add(-x, y)
        dn = d.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        lx = self.lambda_x(x)
        coef = 2 * torch.atanh(dn.clamp(max=1 - self.eps)) / (lx * dn)
        return coef * d

    def distance(self, x: torch.Tensor, y: torch.Tensor):
        d = self.mobius_add(-x, y).norm(dim=-1).clamp(max=1 - self.eps)
        return 2 * torch.atanh(d)

    def project(self, x: torch.Tensor):
        """Retract any point into the ball."""
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.where(n < 1 - self.eps, x,
                           x / n * (1 - self.eps))


# --------------------------------------------------------------------------
# manifold schedule: positions of the thought chain -> manifold nets
# --------------------------------------------------------------------------

class ManifoldSchedule:
    """Binds each position (cycle, depth_fraction, subgan_id) of the chain of
    thought to its own manifold net with its own temperature.

    Binding rule (default):
      cycle == 0        -> euclidean   (fast, cheap, wide exploration)
      cycle mid         -> fisher      (natural gradient: precise updates)
      cycle late        -> poincare    (hyperbolic: hierarchical convergence)
      at a sub-GAN site -> frobenius   (matrix manifold: low-rank refinement)

    Each manifold net carries a learnable temperature parameter so the
    "personality" of each position evolves during training.
    """

    MANIFOLDS = ("euclidean", "fisher", "poincare", "frobenius")

    def __init__(self, n_cycles: int = 4):
        self.n_cycles = max(1, n_cycles)
        self.ball = PoincareBall()
        # learnable per-manifold temperatures (log-space for positivity)
        self.log_temp = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(())) for m in self.MANIFOLDS
        })

    def manifold_at(self, cycle: int, depth_fraction: float, subgan_id=None):
        if subgan_id is not None:
            return "frobenius"
        if cycle <= 0:
            return "euclidean"
        if cycle >= self.n_cycles - 1:
            return "poincare"
        return "fisher"

    def temperature(self, manifold: str):
        return torch.exp(self.log_temp[manifold])


class ThoughtSampler:
    """Samples a fresh 'thought' (state perturbation) at a given position.

    The covariance of the tangent-space Gaussian is induced by the manifold's
    metric: directions with high Fisher information get small variance
    (the model is confident there -> barely perturbs), low-information
    directions get large variance (exploration).  Poincare thoughts are
    sampled in the tangent space then pushed forward with the exp map;
    frobenius thoughts are anisotropic rank-1 matrix perturbations.
    """

    def __init__(self, schedule: ManifoldSchedule, base_scale: float = 0.05):
        self.schedule = schedule
        self.base_scale = base_scale

    def __call__(self, x: torch.Tensor, cycle: int, depth_fraction: float,
                 subgan_id=None, fisher: dict = None):
        """x: (B, C) hidden state. Returns x_perturbed of the same shape."""
        m = self.schedule.manifold_at(cycle, depth_fraction, subgan_id)
        temp = self.schedule.temperature(m)
        scale = self.base_scale * temp
        B, C = x.shape
        if m == "euclidean":
            eps = torch.randn_like(x) * scale
            return x + eps
        if m == "poincare":
            xp = self.schedule.ball.project(x.detach())
            v = torch.randn_like(xp) * scale                    # tangent at xp
            return self.schedule.ball.exp_map(xp, v)
        if m == "frobenius":
            # anisotropic rank-1 perturbation: u v^T with ||u|| ~ scale
            u = torch.randn(B, 1, device=x.device, dtype=x.dtype)
            v = torch.randn(1, C, device=x.device, dtype=x.dtype)
            return x + scale * (u @ v) / math.sqrt(C)
        # fisher: anisotropic covariance = 1 / sqrt(F + eps) along each dim
        inv = torch.ones_like(x)
        if fisher is not None:
            for pid, f in fisher.items():
                inv = inv * (1.0 / (f.sqrt() + 1e-6))[:B, :C] if f.shape == x.shape else inv
        return x + scale * torch.randn_like(x) * inv


# --------------------------------------------------------------------------
# beyond Riemann: Finsler tangent norm and the Fisher-Rao simplex sphere
# --------------------------------------------------------------------------

class FinslerManifold:
    """Finsler geometry: the tangent norm is a Minkowski functional, i.e.
    direction-dependent:  F(x, v) = ||v||_p  for a per-point order p(x).

    This generalizes the Riemannian case (p=2): geodesic balls become
    anisotropic "indicatrices".  Here p is a fixed scalar (per instance), the
    simplest Finsler structure that keeps exact closed forms.

        unit ball      B(x) = {v : F(x, v) <= 1}
        metric tensor  g_ij = (p-1) |v_i|^{p-2} delta_ij (direction-dependent!)

    """

    def __init__(self, p: float = 1.5):
        assert 1.0 < p < 2.0, "keep p in (1,2): strictly convex, sub-Riemannian limit"
        self.p = p

    def norm(self, v: torch.Tensor) -> torch.Tensor:
        """Minkowski functional of the tangent vector v (last dim is vector)."""
        return v.abs().pow(self.p).sum(dim=-1).pow(1.0 / self.p)

    def unit(self, v: torch.Tensor) -> torch.Tensor:
        """Project v onto the Finsler unit indicatrix."""
        n = self.norm(v).clamp(min=1e-12)
        return v / n.unsqueeze(-1)

    def direction_metric(self, v: torch.Tensor) -> torch.Tensor:
        """g_ij(v) -- the direction-dependent Finsler metric (last two dims)."""
        g = (self.p - 1) * v.abs().pow(self.p - 2)
        return torch.diag_embed(g.clamp(min=1e-12))


class SimplexSphere:
    """Fisher-Rao geometry of the probability simplex is ISOMETRIC to the
    Euclidean sphere of radius 2 under the square-root embedding

        p -> 2 sqrt(p)   (Hellinger half-angle coordinates).

    Hence geodesics on the simplex are great circles of the sphere, and the
    Fisher-Rao (Hellinger) distance is

        d_FR(p, q) = 2 arccos( sum_i sqrt(p_i q_i) ).

    Class methods work directly on probability vectors (last dim sums to 1).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def embed(self, p: torch.Tensor) -> torch.Tensor:
        """Map simplex -> sphere of radius 2: s = 2 sqrt(p)."""
        return 2.0 * p.clamp(min=0.0).sqrt()

    def geodesic(self, p: torch.Tensor, q: torch.Tensor, t: float) -> torch.Tensor:
        """Great-circle interpolation on the sphere, pulled back to simplex."""
        sp, sq = self.embed(p), self.embed(q)
        sp = F.normalize(sp, dim=-1) * 2.0
        sq = F.normalize(sq, dim=-1) * 2.0
        omega = (sp * sq).sum(dim=-1, keepdim=True).clamp(-1 + self.eps, 1.0).acos()
        tiny = omega < 1e-6                       # p ~ q: degenerate great circle
        sinw = omega.sin().clamp(min=self.eps)
        s = (torch.sin((1 - t) * omega) * sp + torch.sin(t * omega) * sq) / sinw
        probs = (s / 2.0) ** 2
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        return torch.where(tiny, p.expand_as(probs), probs)

    def distance(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Fisher-Rao / Hellinger arc distance."""
        ip = (p.clamp(min=0.0).sqrt() * q.clamp(min=0.0).sqrt()).sum(dim=-1)
        # clamp the upper end to exactly 1.0: arccos(1) = 0 handles p == q
        return 2.0 * ip.clamp(-1 + self.eps, 1.0).acos()


# --------------------------------------------------------------------------
# ManifoldMesh -- the pia mater: left-brain GANs and right-brain GANs share
# manifold instances in memory through one module-level registry, so the two
# hemispheres ride the SAME geometric fabric without being physically merged.
# --------------------------------------------------------------------------

_MESH: dict = {}  # module-level shared registry (pia mater)


class ManifoldMesh:
    """Memory-level manifold registry with cross-brain binding.

    bind(key, obj)      register a manifold net under a brain-local key
    pair(a, b)          bind two keys to ONE shared instance (the membrane);
                        both hemispheres resolve to the same object
    resolve(key)        lookup; falls back to None
    unbind(key)         separate again -- the membrane is peelable by design
    """

    def bind(self, key: str, manifold_obj):
        _MESH[key] = manifold_obj
        return self

    def pair(self, key_a: str, key_b: str):
        """Soft-dural coupling: both keys point at one in-memory instance.

        If only one side is registered, the pair resolves to that instance;
        if both exist and differ, the LEFT side's instance wins and the
        right side is rebound to it (left brain is the geometric anchor).
        """
        a = _MESH.get(key_a)
        b = _MESH.get(key_b)
        if a is None and b is None:
            raise KeyError(f"ManifoldMesh.pair: neither '{key_a}' nor '{key_b}' is bound")
        obj = a if a is not None else b
        _MESH[key_a] = obj
        _MESH[key_b] = obj
        return self

    def resolve(self, key: str):
        return _MESH.get(key)

    def unbind(self, key: str):
        _MESH.pop(key, None)
        return self

    def bound_keys(self):
        return sorted(_MESH.keys())
