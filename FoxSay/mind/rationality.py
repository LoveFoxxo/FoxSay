"""
rationality.py -- the mathematics of STRONG RATIONAL INTUITION: System 1
fast intuition and System 2 formal verification mutually calibrated.

Neuroscience analogy:
  * FastWeights         = synaptic short-term plasticity (STP): intuition is
                          not "computed", it is temporarily GROWN on top of
                          the slow weights, then decays away.
  * TypeChecker         = anterior cingulate error monitoring (ERN): a
                          differentiable compiler type-check over the chain
                          of thought (Curry-Howard: propositions are types,
                          proofs are programs).
  * MartingaleMonitor   = metacognitive confidence (prefrontal-parietal
                          network): a rational confidence sequence must be a
                          martingale, hence by Doob's theorem it MUST converge.
  * SubmodularBound     = proof-carrying pruning: diminishing marginal value
                          of reasoning steps gives UCT a provable upper bound,
                          not a heuristic c.
  * SavageDecision      = the formal DEFINITION of rationality: Savage's
                          axioms imply a separation representation
                          EU(a) = sum_s p(s) U(o(a,s)) -- subjective
                          probability and utility split apart.

Mathematics implemented:

  * Fast weights (Hebbian / STP):
        w_fast <- rho * w_fast + eta * v_out v_in^T      (Hebbian outer product)
    applied as an additive low-rank perturbation on the forward pass.

  * Type consistency (Curry-Howard shadow):
        score(t_1..t_T) = -sum_i || t_{i+1} - compose(t_i) ||^2
    in a learnable type-embedding space; a well-typed chain scores 0 penalty.

  * Martingale calibration:
        E[conf_{t+1} | F_t] = conf_t
    penalty = sum_t (conf_{t+1} - conf_t)^2 (deviations from the martingale
    property); Doob's upcrossing inequality bounds how often a bounded
    confidence process can oscillate:  E[U(a,b)] <= E[(X_n - a)+] / (b - a).

  * Submodularity:
        f(A u {e}) - f(A) >= f(B u {e}) - f(B)   for A subseteq B
    The greedy bound (1 - 1/e) is the mathematical guarantee that lets UCT
    prune with confidence; marginal_gain() supplies the exact quantity.

  * Savage separation:
        EU(a) = sum_s p(s) U(o(a, s))
    with p a subjective probability (sums to 1) and U a utility -- the
    two objects live in SEPARATE structures, which is the formal meaning of
    "rational".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FastWeights", "TypeChecker", "MartingaleMonitor",
    "SubmodularBound", "SavageDecision",
]


# ---------------------------------------------------------------------------
# 1. fast weights (Hebbian / short-term plasticity)
# ---------------------------------------------------------------------------

class FastWeights(nn.Module):
    """w_fast <- rho * w_fast + eta * v_out v_in^T (Hebbian learning).

    The fast matrix is a STATE (like synaptic facilitation), not a
    parameter: it lives only inside the current thought cycle and decays at
    rate rho. reset() wipes it (intuition is context-local).
    """

    def __init__(self, dim: int, eta: float = 0.05, rho: float = 0.9):
        super().__init__()
        self.dim = dim
        self.eta = eta
        self.rho = rho
        self.register_buffer("w_fast", torch.zeros(dim, dim))

    @torch.no_grad()
    def hebbian_update(self, x_in: torch.Tensor, x_out: torch.Tensor):
        """x_in: (B, C), x_out: (B, C). One Hebbian step on the mean outer
        product: v_out v_in^T, with decay rho (forgetting)."""
        v_in = x_in.detach().mean(dim=0)      # (C,)
        v_out = x_out.detach().mean(dim=0)    # (C,)
        self.w_fast.mul_(self.rho).add_(self.eta * torch.outer(v_out, v_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fast matrix as an additive perturbation (B, C) -> (B, C)."""
        return x + F.linear(x, self.w_fast)

    @torch.no_grad()
    def reset(self):
        self.w_fast.zero_()


# ---------------------------------------------------------------------------
# 2. type consistency (Curry-Howard: propositions are types, proofs programs)
# ---------------------------------------------------------------------------

class TypeChecker(nn.Module):
    """A differentiable compiler for the chain of thought.

    Each reasoning step has a type embedding; a well-typed chain satisfies
    t_{i+1} = compose(t_i) (function application / modus ponens). The score
    is the negative squared residual, so a type-correct chain has zero
    penalty and any ill-typed step is backpropagated as a loss.
    """

    def __init__(self, n_type: int = 16, type_dim: int = 32):
        super().__init__()
        self.type_emb = nn.Parameter(torch.randn(n_type, type_dim) * 0.1)
        self.compose = nn.Linear(type_dim, type_dim, bias=False)   # f: type -> type
        # init compose near-identity: identity functions are the canonical
        # well-typed programs
        nn.init.eye_(self.compose.weight)

    def consistency(self, steps: torch.Tensor) -> torch.Tensor:
        """steps: (T,) integer type ids of the chain. Returns (T-1,) residuals."""
        t = self.type_emb[steps]                      # (T, D)
        t_next = self.compose(t[:-1])                 # (T-1, D)
        return -((t[1:] - t_next) ** 2).sum(dim=-1)   # (T-1,)

    def type_loss(self, steps: torch.Tensor) -> torch.Tensor:
        """Training loss: negative total consistency (0 for a perfect chain)."""
        return -self.consistency(steps).mean()


# ---------------------------------------------------------------------------
# 3. martingale calibration (Doob)
# ---------------------------------------------------------------------------

class MartingaleMonitor:
    """A rational confidence sequence {conf_t} is a martingale:
    E[conf_{t+1} | history] = conf_t.

    martingale_penalty() measures squared deviations; upcrossings() counts
    oscillations through an interval [a, b] and checks them against Doob's
    bound:  E[U(a,b)] <= E[(X_n - a)+] / (b - a).
    """

    def martingale_penalty(self, conf_seq: torch.Tensor) -> torch.Tensor:
        """conf_seq: (T,). Squared step deviations from the martingale property."""
        return ((conf_seq[1:] - conf_seq[:-1]) ** 2).mean()

    @staticmethod
    def upcrossings(conf_seq: torch.Tensor, a: float, b: float) -> int:
        """Number of complete upward crossings of [a, b] (Doob's U(a,b))."""
        U, below = 0, True
        for x in conf_seq.tolist():
            if below and x <= a:
                below = False
            elif (not below) and x >= b:
                U += 1
                below = True
        return U

    def doob_bound(self, conf_seq: torch.Tensor, a: float, b: float) -> float:
        """Doob's upcrossing inequality upper bound for E[U(a,b)]."""
        Xn = conf_seq[-1]
        return float((Xn - a).clamp_min(0) / (b - a)) if b > a else float("inf")

    def check(self, conf_seq: torch.Tensor, a: float, b: float) -> bool:
        """True iff the observed upcrossings respect Doob's bound."""
        return self.upcrossings(conf_seq, a, b) <= self.doob_bound(conf_seq, a, b) + 1e-6


# ---------------------------------------------------------------------------
# 4. submodularity (provable pruning bounds)
# ---------------------------------------------------------------------------

class SubmodularBound:
    """Submodular set functions have diminishing marginal returns:
        f(A u {e}) - f(A) >= f(B u {e}) - f(B)  for all A subseteq B.

    For a monotone submodular f, the greedy algorithm is (1 - 1/e)-optimal.
    UCT's heuristic exploration term c gets replaced by a PROVABLE bound:
    prune a node when its marginal gain upper bound cannot beat the current
    best (diminishing returns guarantee it never will).
    """

    def marginal(self, f, S: set, e) -> float:
        """Marginal gain of adding e to S: f(S u {e}) - f(S)."""
        S_plus = S | {e}
        return f(S_plus) - f(S)

    def is_submodular(self, f, ground_set: set, n_checks: int = 200) -> bool:
        """Statistical test of the submodular inequality on random pairs."""
        import random
        items = list(ground_set)
        for _ in range(n_checks):
            k = random.randint(0, len(items))
            A = set(random.sample(items, k))
            B = set(random.sample(items, k))
            B |= A
            e = random.choice([x for x in items if x not in B] or items)
            if self.marginal(f, A, e) + 1e-9 < self.marginal(f, B, e):
                return False
        return True

    def greedy_guarantee(self, f, ground_set: set, k: int, opt=None):
        """Greedy maximization; returns (chosen_set, f_value, guarantee).

        guarantee = f(greedy) / OPT >= 1 - 1/e when OPT is provided."""
        S = set()
        items = list(ground_set)
        for _ in range(k):
            e = max(items, key=lambda x: self.marginal(f, S, x) if x not in S else -1)
            S.add(e)
        val = f(S)
        if opt is None:
            return S, val, None
        return S, val, val / max(opt, 1e-12)

    def prune_decision(self, f, chosen: set, candidate, best_value: float) -> bool:
        """True = prune candidate: its optimistic marginal upper bound
        (first-order from submodularity: gain <= marginal over empty set)
        cannot reach best_value."""
        gain_bound = self.marginal(f, set(), candidate)   # maximal possible gain
        return f(chosen) + gain_bound <= best_value + 1e-9


# ---------------------------------------------------------------------------
# 5. Savage decision theory (utility/probability separation)
# ---------------------------------------------------------------------------

class SavageDecision:
    """Savage's axioms => EU(a) = sum_s p(s) U(o(a, s)).

    The class keeps subjective probability p and utility U as SEPARATE
    structures (the separation representation is the formal definition of
    rationality); expected_utility() combines them only at decision time.
    """

    def expected_utility(self, subj_prob: torch.Tensor,
                         utility: torch.Tensor) -> torch.Tensor:
        """subj_prob: (S,) sums to 1; utility: (S,) or (A, S). Returns EU."""
        assert abs(float(subj_prob.sum() - 1.0)) < 1e-5, "subjective probability must sum to 1"
        return (subj_prob * utility).sum(dim=-1)

    def certainty_equivalent(self, subj_prob, utility, risk_aversion: float = 0.0):
        """CE = E[U] - (ra/2) Var[U] (Arrow-Pratt quadratic approximation)."""
        eu = self.expected_utility(subj_prob, utility)
        var = (subj_prob * (utility - eu.unsqueeze(-1)) ** 2).sum(dim=-1)
        return eu - 0.5 * risk_aversion * var
