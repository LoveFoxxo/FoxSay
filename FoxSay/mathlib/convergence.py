"""Checkable convergence bounds, with empirical estimates kept distinct.

No finite parameter-count rule or scaling-law fit can certify zero test loss.
For cross-entropy the true floor is the target distribution's conditional
entropy; deterministic fixed datasets can approach zero, stochastic language
generally cannot. PL/NTK statements below are conditional training results,
not promises about generalization.
"""
from __future__ import annotations

import math
import torch

__all__ = [
    "chinchilla_lower_bound", "data_information_bits", "model_capacity_bits",
    "interpolation_feasible", "pl_convergence_bound", "linear_convergence_check",
    "ntk_min_eigenvalue", "convergence_path",
]


def _positive_finite(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def chinchilla_lower_bound(N: float, D: float) -> dict:
    """Return the original Chinchilla *empirical fit*, not a universal bound.

    E=1.69 is dataset/tokenizer-specific and must not be interpreted as the
    irreducible entropy of every corpus. The historical function name is kept
    for checkpoint tooling compatibility.
    """
    N, D = _positive_finite("N", N), _positive_finite("D", D)
    capacity_term = 406.4 / N ** 0.34
    data_term = 410.7 / D ** 0.28
    return {"L_pred": 1.69 + capacity_term + data_term, "irreducible": 1.69,
            "capacity_term": capacity_term, "data_term": data_term,
            "estimate_only": True}


def data_information_bits(N_tokens: int, vocab_size: int) -> float:
    """Raw corpus-size upper bound; not its conditional entropy or Kolmogorov complexity."""
    if N_tokens < 0 or vocab_size < 2:
        raise ValueError("N_tokens must be >= 0 and vocab_size >= 2")
    return float(N_tokens) * math.log2(vocab_size)


def model_capacity_bits(P: int, bits_per_param: float = 2.0) -> float:
    """Storage heuristic only; bits/parameter is not an interpolation theorem."""
    if P < 0 or not math.isfinite(bits_per_param) or bits_per_param <= 0:
        raise ValueError("P must be >= 0 and bits_per_param must be positive")
    return float(P) * bits_per_param


def interpolation_feasible(P: int, N_tokens: int, vocab_size: int,
                           bits_per_param: float = 2.0) -> dict:
    """A coarse capacity heuristic; neither necessary nor sufficient for fitting."""
    cap = model_capacity_bits(P, bits_per_param)
    info = data_information_bits(N_tokens, vocab_size)
    return {"feasible": cap >= info, "capacity_bits": cap, "data_bits": info,
            "params_critical": math.ceil(info / bits_per_param),
            "heuristic_only": True}


def pl_convergence_bound(mu: float, L_smooth: float, t: int, L0: float,
                         L_star: float = 0.0) -> float:
    """GD bound for a mu-PL, L-smooth objective with step size 1/L_smooth."""
    mu, L_smooth = _positive_finite("mu", mu), _positive_finite("L_smooth", L_smooth)
    if not 0 < mu <= L_smooth or t < 0 or L0 < L_star:
        raise ValueError("require 0 < mu <= L_smooth, t >= 0 and L0 >= L_star")
    return L_star + (L0 - L_star) * (1.0 - mu / L_smooth) ** t


def linear_convergence_check(losses, tol: float = 0.05, tail: int = 5,
                             loss_floor: float = 0.0) -> dict:
    """Detect geometric decay of *excess* loss; never infers a zero floor."""
    if tail < 1 or not 0 <= tol < 1 or not math.isfinite(loss_floor):
        raise ValueError("invalid tail/tolerance/loss_floor")
    values = [float(x) for x in losses]
    if len(values) < tail + 1 or any(not math.isfinite(x) for x in values):
        return {"converging": False, "rate": None, "ratio_tail": None}
    excess = [x - loss_floor for x in values]
    if min(excess) <= 0:
        return {"converging": False, "rate": None, "ratio_tail": None}
    ratios = [excess[i + 1] / excess[i]
              for i in range(len(values) - tail - 1, len(values) - 1)]
    rate = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    converging = rate < 1.0 - tol and all(0 < r < 1 for r in ratios)
    return {"converging": converging, "rate": rate, "ratio_tail": ratios,
            "assumed_floor": loss_floor}


def ntk_min_eigenvalue(model, x_batch: torch.Tensor) -> tuple:
    """Empirical NTK at the current weights, computed without polluting .grad.

    A positive eigenvalue here alone is not a global-convergence certificate:
    the usual lazy-training bounds also require spectral stability along the
    optimization trajectory and sufficient over-parameterization.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    grads = []
    for i in range(x_batch.size(0)):
        out = model(x_batch[i:i + 1]).mean()
        row = torch.autograd.grad(out, params, retain_graph=False, allow_unused=True)
        grads.append(torch.cat([
            torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
            for p, g in zip(params, row)
        ]).double())
    K = torch.stack(grads)
    K = K @ K.t()
    K = (K + K.t()) * 0.5  # remove round-off asymmetry before eigvalsh
    lam = torch.linalg.eigvalsh(K)
    return float(lam[0].item()), K


def convergence_path(P: int, N_tokens: int, vocab_size: int,
                     loss_history=None, N_capacity_extra: float = 1.0,
                     **ntk_kwargs) -> dict:
    """Report heuristics and conditional checks without claiming test loss -> 0."""
    _positive_finite("N_capacity_extra", N_capacity_extra)
    feas = interpolation_feasible(P, N_tokens, vocab_size)
    chin = chinchilla_lower_bound(max(P, 1), max(N_tokens * N_capacity_extra, 1.0))
    out = {
        "regime": "interpolation" if feas["feasible"] else "compression",
        "feasible": feas["feasible"],
        "data_bits": feas["data_bits"], "capacity_bits": feas["capacity_bits"],
        "params_critical": feas["params_critical"],
        "empirical_loss_estimate": chin["L_pred"],
        "zero_loss_certified": False,
        "loss_to_zero_path": (
            "B1 is only a capacity heuristic. B3 requires NTK stability along "
            "the whole trajectory; B2 requires measured PL/smoothness constants "
            "and a stated loss floor. Neither certifies zero generalization error."
        ),
    }
    if loss_history is not None:
        out["linear_check"] = linear_convergence_check(loss_history)
    return out
