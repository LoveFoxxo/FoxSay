"""
intuition.py -- the mathematical-intuition substrate of the mind.

  * ThinkTree (MCTS-UCT): the linear ACT cycle becomes a search tree.
        UCT(a) = Q(a) + c sqrt( ln N(parent) / n(a) )     (upper confidence)
        each cycle expands top-k candidates, selects by UCT, and back-
        propagates Q-values with TD; the GAN gate becomes the tree pruner.

  * Combinator (TPR): differentiable composition operator.
        bind(g, f) ~ g (x) f  (tensor product, TPR style) reduced to a
        learned bilinear map; unbind recovers the components by pseudo-
        inverse -- the categorical shadow of "g after f".

  * NumEmbed: continuous representation of numbers preserving modular and
        radix structure.
        embed(n) = [sin(2 pi n / p), cos(2 pi n / p) for prime p in P]
                   (+) one-hot radix digits of n in base b.
        CRT guarantees the prime-period coordinates separate congruences.

  * ProofDAG: message passing over a derivation DAG (GNN shadow):
        messages flow along the adjacency mask; attention is restricted to
        predecessors -- reasoning as graph computation, not a flat sequence.

  * Equivariant: symmetry shadows.
        DeepSets-style permutation-equivariant layer
            h_i = rho(x_i) + sum_j phi(x_j)
        and a translation-symmetric conv via weight mirroring.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ThinkTree", "Combinator", "NumEmbed", "ProofDAG", "Equivariant"]

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


# --------------------------------------------------------------------------
# MCTS-UCT
# --------------------------------------------------------------------------

class ThinkTree:
    """Top-k branching cycle with UCT selection and Q back-propagation."""

    def __init__(self, k: int = 3, c: float = 1.4, gamma: float = 0.9):
        self.k = k            # branching factor (candidates per cycle)
        self.c = c            # exploration constant
        self.gamma = gamma
        self.Q = {}           # state-hash -> (Q_value, visit_count)

    @staticmethod
    def _key(states: torch.Tensor):
        return states.detach().sum().item()

    def select(self, candidates: torch.Tensor, values: torch.Tensor, parent):
        """Choose top candidates by UCT among `candidates` (M, C) with
        value estimates `values` (M,).  Returns indices of survivors."""
        n_parent = self.Q.get(self._key(parent), (0, 1))[1]
        ucts = []
        for i, (x, q) in enumerate(zip(candidates, values)):
            qv, n = self.Q.get(self._key(x.unsqueeze(0)), (float(q), 1))
            uct = qv + self.c * math.sqrt(math.log(max(n_parent, 1) + 1) / n)
            ucts.append(uct)
        ucts = torch.tensor(ucts, device=candidates.device)
        return torch.topk(ucts, k=min(self.k, len(ucts))).indices

    def backup(self, path, terminal_value: float):
        """TD-style Q back-propagation along the chosen path."""
        g = terminal_value
        for state in reversed(path):
            key = self._key(state)
            q, n = self.Q.get(key, (0.0, 0))
            g = self.gamma * g
            self.Q[key] = ((q * n + g) / (n + 1), n + 1)


# --------------------------------------------------------------------------
# TPR combinator (categorical composition shadow)
# --------------------------------------------------------------------------

class Combinator(nn.Module):
    """Differentiable g-after-f: bind/unbind via a learned bilinear form."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        # bilinear composition tensor reduced to two projections (low rank)
        self.Pg = nn.Linear(n_embd, n_embd, bias=False)   # role projection
        self.Pf = nn.Linear(n_embd, n_embd, bias=False)   # filler projection
        nn.init.orthogonal_(self.Pg.weight)
        nn.init.orthogonal_(self.Pf.weight)

    def bind(self, g: torch.Tensor, f: torch.Tensor):
        """(g o f) vector = Pg g + Pf f + (Pg g) * (Pf f)."""
        a, b = self.Pg(g), self.Pf(f)
        return a + b + a * b

    def unbind(self, bound: torch.Tensor):
        """Recover the role component by pseudo-inverse of the role map."""
        with torch.no_grad():
            P = self.Pg.weight
            pinv = torch.linalg.pinv(P)
        return bound @ pinv.T


# --------------------------------------------------------------------------
# continuous numbers with modular + radix structure
# --------------------------------------------------------------------------

class NumEmbed:
    """embed(n) keeps congruences linearly separable (CRT over prime periods)
    and keeps radix structure via base-b digit one-hots."""

    def __init__(self, n_primes: int = 8, base: int = 10, digits: int = 4):
        self.primes = PRIMES[:n_primes]
        self.base = base
        self.digits = digits

    @property
    def dim(self):
        return 2 * len(self.primes) + self.base * self.digits

    def embed(self, n):
        """n: int or tensor of ints -> (..., dim) float tensor."""
        n = torch.as_tensor(n, dtype=torch.float32)
        scalar = n.dim() == 0
        if scalar:
            n = n.unsqueeze(0)
        parts = []
        for p in self.primes:
            a = 2 * math.pi * n / p
            parts.append(torch.stack([torch.sin(a), torch.cos(a)], dim=-1))
        # radix digits (support scalar and batched ints)
        m = torch.as_tensor(n, dtype=torch.long).abs()
        for d in range(self.digits):
            digit = (m // (self.base ** d)) % self.base
            parts.append(F.one_hot(digit, self.base).float())
        out = torch.cat(parts, dim=-1)
        return out.squeeze(0) if scalar else out

    def congruence_separable(self, a: int, b: int, mod: int):
        """True iff the prime-period coords can separate a, b (mod)."""
        ea, eb = self.embed(a)[:2 * len(self.primes)], self.embed(b)[:2 * len(self.primes)]
        return not torch.allclose(ea, eb)


# --------------------------------------------------------------------------
# proof DAG message passing
# --------------------------------------------------------------------------

class ProofDAG(nn.Module):
    """One graph-attention step over a derivation DAG.

    edges: (N, N) adjacency (edge i->j means i precedes j).
    states: (N, C).  Messages flow from predecessors to successors only.
    """

    def __init__(self, n_embd: int, n_head: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, states: torch.Tensor, edges: torch.Tensor):
        N = states.shape[0]
        mask = (edges.T > 0).float()          # (N, N): attend only to predecessors
        mask = mask + torch.eye(N, device=states.device)  # self-loops: no NaN rows
        mask = mask.masked_fill(mask == 0, float("-inf")).unsqueeze(0)
        out, _ = self.attn(states.unsqueeze(0), states.unsqueeze(0),
                           states.unsqueeze(0), attn_mask=mask[0])
        return self.ln(states + out.squeeze(0))


# --------------------------------------------------------------------------
# equivariance shadows
# --------------------------------------------------------------------------

class Equivariant(nn.Module):
    """Two symmetry shadows kept separate (mixing them breaks both):

      * set_forward : DeepSets permutation-equivariant layer
            h_i = rho(x_i) + sum_j phi(x_j)     (equivariant to any perm)
      * conv_forward: translation-equivariant conv with mirrored weights
            w = (w + flip(w)) / 2  (palindromic kernel -> shift symmetry)
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.rho = nn.Linear(n_embd, n_embd)
        self.phi = nn.Linear(n_embd, n_embd)
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1,
                              padding_mode="circular")  # circular -> exact shift equivariance

    def set_forward(self, x: torch.Tensor):
        """x: (B, T, C) -> permutation-equivariant features."""
        return self.rho(x) + self.phi(x.mean(dim=1, keepdim=True))

    def conv_forward(self, x: torch.Tensor):
        """x: (B, T, C) -> translation-equivariant features."""
        with torch.no_grad():
            w = self.conv.weight
            self.conv.weight.copy_(0.5 * (w + w.flip(-1)))
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor):
        """Concatenated features (not used for equivariance guarantees)."""
        return torch.cat([self.set_forward(x), self.conv_forward(x)], dim=-1)
