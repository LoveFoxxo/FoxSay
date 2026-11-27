"""Sparse, lossless-token Top-K mixture of experts for both hemispheres."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SparseMoE"]


class SparseMoE(nn.Module):
    """ReLU^2 experts with fp32+ routing and no token dropping.

    Expert matrices are separate 2-D parameters: Muon and the existing FSDP
    owner-sharding logic can therefore handle them without a special optimizer.
    ``last_aux_loss`` is collected by the containing brain. This deliberately
    favors correctness over fused-kernel complexity; distributed expert-parallel
    dispatch is still required for production trillion-parameter training.
    """

    def __init__(self, n_embd: int, n_experts: int, expert_ratio: float = 4.0,
                 top_k: int = 1, aux_loss_scale: float = 0.01,
                 router_z_loss_scale: float = 0.001):
        super().__init__()
        if n_embd <= 0 or n_experts < 2:
            raise ValueError("n_embd must be positive and n_experts must be >= 2")
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must lie in [1, n_experts]")
        if not math.isfinite(expert_ratio) or expert_ratio <= 0:
            raise ValueError("expert_ratio must be finite and positive")
        if aux_loss_scale < 0 or router_z_loss_scale < 0:
            raise ValueError("auxiliary loss scales must be non-negative")
        self.n_embd = int(n_embd)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.expert_hidden = max(1, int(round(n_embd * expert_ratio)))
        self.aux_loss_scale = float(aux_loss_scale)
        self.router_z_loss_scale = float(router_z_loss_scale)
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        self.fc = nn.ParameterList([
            nn.Parameter(torch.empty(self.expert_hidden, n_embd))
            for _ in range(n_experts)
        ])
        self.proj = nn.ParameterList([
            nn.Parameter(torch.empty(n_embd, self.expert_hidden))
            for _ in range(n_experts)
        ])
        self.last_aux_loss = None

    def init_weights(self):
        nn.init.normal_(self.router.weight, mean=0.0, std=self.n_embd ** -0.5)
        bound = math.sqrt(3.0) * 0.4 * self.n_embd ** -0.5
        for w in self.fc:
            nn.init.uniform_(w, -bound, bound)
        # Neutral-at-birth output projections match the stock dense MLP.
        for w in self.proj:
            nn.init.zeros_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if shape[-1] != self.n_embd:
            raise ValueError(f"expected width {self.n_embd}, got {shape[-1]}")
        flat = x.reshape(-1, self.n_embd)
        route_dtype = (torch.float64 if x.dtype == torch.float64 or
                       self.router.weight.dtype == torch.float64 else torch.float32)
        logits = F.linear(flat.to(route_dtype), self.router.weight.to(route_dtype))
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(route_dtype).tiny)

        # Switch-style load balancing, normalized for top-k > 1. The z term
        # penalizes oversized router logits without changing selected experts.
        density = F.one_hot(indices, self.n_experts).to(route_dtype).mean(dim=(0, 1))
        balance = self.n_experts * (density * probs.mean(dim=0)).sum()
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        self.last_aux_loss = (self.aux_loss_scale * balance +
                              self.router_z_loss_scale * z_loss)

        out = torch.zeros_like(flat)
        weights = weights.to(x.dtype)
        for expert_id in range(self.n_experts):
            rows, slots = (indices == expert_id).nonzero(as_tuple=True)
            # Empty expert calls are intentional: every expert receives a real
            # (zero) gradient, keeping distributed collectives rank-aligned.
            selected = flat.index_select(0, rows)
            h = F.linear(selected, self.fc[expert_id].to(x.dtype))
            h = F.relu(h).square()
            update = F.linear(h, self.proj[expert_id].to(x.dtype))
            update = update * weights[rows, slots].unsqueeze(-1)
            out = out.index_add(0, rows, update)
        return out.view(shape)

    def parameter_report(self) -> dict:
        expert_total = sum(p.numel() for p in self.fc) + sum(p.numel() for p in self.proj)
        expert_active = self.top_k * 2 * self.n_embd * self.expert_hidden
        return {
            "total": self.router.weight.numel() + expert_total,
            "active": self.router.weight.numel() + expert_active,
            "experts": self.n_experts,
            "top_k": self.top_k,
        }
