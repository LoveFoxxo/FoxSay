"""
causal.py -- structural-causal-model shadows.

Pearl's three-step counterfactual recipe, in differentiable form:

  1. Abduction   : infer the noise/exogenous state u from the observed x
                   (encode x -> u, the "why did this happen" variable)
  2. Action      : intervention -- set the cause variable to a chosen value
                   (do-operator: replace the variable, cut incoming edges)
  3. Prediction  : run the forward model on the intervened variable.

Plus a causal-chain reward: the model is rewarded when its own chain
prediction  event -> my-reaction -> consequence  is consistent with the
observed consequence -- the "emotion is causal" objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CausalChain"]


class CausalChain(nn.Module):
    """A differentiable SCM for the chain  event -> reaction -> outcome.

    x_event (E, C), x_reaction (E, C) -> predicts outcome; supports do()
    interventions on the reaction node and counterfactual queries.
    """

    def __init__(self, n_embd: int, hidden: int = 64):
        super().__init__()
        self.n_embd = n_embd
        self.hidden = hidden
        self.abduct = nn.Linear(2 * n_embd, hidden)   # (event, outcome) -> u
        self.forward_model = nn.Sequential(
            nn.Linear(2 * n_embd + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_embd),
        )

    def predict(self, event, reaction):
        """P(outcome | event, reaction) -- exogenous u marginalized to 0."""
        u = torch.zeros(event.shape[0], self.hidden,
                        device=event.device, dtype=event.dtype)
        return self.forward_model(torch.cat([event, reaction, u], dim=-1))

    def do(self, event, reaction, outcome=None):
        """do(reaction = r): cut the edge into reaction, predict outcome."""
        if outcome is None:
            return self.predict(event, reaction)
        # abduction: infer u consistent with the observed outcome
        u = self.abduct(torch.cat([event, outcome], dim=-1))
        return self.forward_model(torch.cat([event, reaction, u], dim=-1))

    def counterfactual(self, event, reaction_observed, outcome_observed,
                       reaction_counterfactual):
        """Pearl's recipe: abduct u from observed, then predict under do(r')."""
        u = self.abduct(torch.cat([event, outcome_observed], dim=-1)).detach()
        return self.forward_model(
            torch.cat([event, reaction_counterfactual, u], dim=-1))

    def chain_reward(self, event, reaction, outcome):
        """Consistency of the model's own causal chain (higher = better)."""
        pred = self.predict(event, reaction)
        return -((pred - outcome) ** 2).sum(dim=-1).mean()
