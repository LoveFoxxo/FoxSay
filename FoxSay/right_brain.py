"""Heterogeneous temporal right hemisphere with causal HiPPO and sparse MoE."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from FoxSay.moe import SparseMoE
from FoxSay.numerics import rms_norm as _norm

__all__ = ["hippo_legs_matrix", "HiPPOChannel", "ConvChannel",
           "RightBlock", "RightBrainConfig", "RightBrain", "mirror_left"]


def mirror_left(left_config, ssm_state: int = 16, conv_kernel: int = 7,
                dt: float = 0.1) -> "RightBrainConfig":
    """Mirror left width/depth and, when enabled, its sparse-expert budget."""
    return RightBrainConfig(
        n_embd=left_config.n_embd,
        n_layer=left_config.n_layer,
        ssm_state=ssm_state,
        conv_kernel=conv_kernel,
        dt=dt,
        moe_num_experts=getattr(left_config, "moe_num_experts", 0),
        moe_expert_ratio=getattr(left_config, "moe_expert_ratio", 4.0),
        moe_top_k=getattr(left_config, "moe_top_k", 1),
        moe_every=getattr(left_config, "moe_every", 1),
        moe_aux_loss_scale=getattr(left_config, "moe_aux_loss_scale", 0.01),
        moe_router_z_loss_scale=getattr(left_config, "moe_router_z_loss_scale", 0.001),
    )


def hippo_legs_matrix(N: int) -> torch.Tensor:
    """Scaled Legendre (LegS) HiPPO matrix in fp64 for stable discretization."""
    if N <= 0:
        raise ValueError("HiPPO state size must be positive")
    A = torch.zeros(N, N, dtype=torch.float64)
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
            elif n == k:
                A[n, k] = -(n + 1.0)
    return A


def _work_dtype(x, weight):
    if x.dtype == torch.float64 or weight.dtype == torch.float64:
        return torch.float64
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


class HiPPOChannel(nn.Module):
    """Fixed HiPPO-LegS dynamics with exact Tustin-discretized input.

    state_t = state_(t-1) A_bar^T + u_t B_bar^T
    y_t     = state_t C^T
    """

    def __init__(self, cfg):
        super().__init__()
        C, N = cfg.n_embd, cfg.ssm_state
        self.dt = float(getattr(cfg, "dt", 0.1))
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("HiPPO dt must be finite and positive")
        self.N = N
        self.input_proj = nn.Linear(C, N, bias=False)
        self.readout = nn.Linear(N, C, bias=False)
        self.register_buffer("A", torch.empty(N, N))
        self.register_buffer("A_bar", torch.empty(N, N))
        self.register_buffer("B_bar", torch.empty(N, N))

    def init_weights(self):
        with torch.no_grad():
            device = self.A.device
            A = hippo_legs_matrix(self.N).to(device)
            I = torch.eye(self.N, device=device, dtype=torch.float64)
            L = I - 0.5 * self.dt * A
            A_bar = torch.linalg.solve(L, I + 0.5 * self.dt * A)
            B_bar = torch.linalg.solve(L, self.dt * I)
            self.A.copy_(A.to(self.A.dtype))
            self.A_bar.copy_(A_bar.to(self.A_bar.dtype))
            self.B_bar.copy_(B_bar.to(self.B_bar.dtype))
        nn.init.normal_(self.readout.weight, std=0.02)
        nn.init.zeros_(self.input_proj.weight)

    def _kernel(self, T: int) -> torch.Tensor:
        """K[c,n,t] = (C A_bar^t B_bar)[c,n], t = 0..T-1."""
        if T <= 0:
            raise ValueError("HiPPO kernel length must be positive")
        Cm = self.readout.weight.to(self.A_bar.dtype)
        h = []
        p = torch.eye(self.N, device=self.A_bar.device, dtype=self.A_bar.dtype)
        for _ in range(T):
            h.append(Cm @ p @ self.B_bar)
            p = p @ self.A_bar
        return torch.stack(h, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_proj.in_features or x.size(1) <= 0:
            raise ValueError("expected non-empty (batch, time, n_embd) input")
        dtype = _work_dtype(x, self.input_proj.weight)
        work = x.to(dtype)
        u = F.linear(work, self.input_proj.weight.to(dtype)).transpose(1, 2)
        # conv1d computes cross-correlation, so the causal impulse is flipped.
        K = self._kernel(x.size(1)).to(dtype).flip(-1)
        u_pad = F.pad(u, (x.size(1) - 1, 0))
        return F.conv1d(u_pad, K).transpose(1, 2).to(x.dtype)

    def init_state(self, batch: int, device, dtype):
        return torch.zeros(batch, self.N, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor):
        if state.shape != (x_t.size(0), self.N):
            raise ValueError("invalid HiPPO recurrent state shape")
        dtype = _work_dtype(x_t, self.input_proj.weight)
        work, state_work = x_t.to(dtype), state.to(dtype)
        u = F.linear(work, self.input_proj.weight.to(dtype))
        state = (state_work @ self.A_bar.to(dtype).t() +
                 u @ self.B_bar.to(dtype).t())
        y = F.linear(state, self.readout.weight.to(dtype))
        return y.to(x_t.dtype), state


class ConvChannel(nn.Module):
    """Depthwise causal convolution; circular mode is opt-in for non-LM data."""

    def __init__(self, cfg):
        super().__init__()
        C = cfg.n_embd
        self.kernel_size = int(getattr(cfg, "conv_kernel", 7))
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("conv_kernel must be a positive odd integer")
        self.causal = bool(getattr(cfg, "causal_conv", True))
        self.conv = nn.Conv1d(C, C, self.kernel_size, groups=C, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = _work_dtype(x, self.conv.weight)
        work = x.to(dtype).transpose(1, 2)
        weight = self.conv.weight.to(dtype)
        if self.causal:
            work = F.pad(work, (self.kernel_size - 1, 0))
            y = F.conv1d(work, weight, groups=x.size(-1))
        else:
            # Explicit symmetric circular padding; no implicit zero padding.
            pad = self.kernel_size // 2
            y = F.conv1d(F.pad(work, (pad, pad), mode="circular"), weight,
                         groups=x.size(-1))
        return y.transpose(1, 2).to(x.dtype)

    def init_state(self, batch: int, device, dtype):
        return torch.zeros(batch, self.conv.in_channels, self.kernel_size - 1,
                           device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, state: torch.Tensor):
        if not self.causal:
            raise RuntimeError("circular convolution has no causal streaming form")
        dtype = _work_dtype(x_t, self.conv.weight)
        window = torch.cat([state.to(dtype), x_t.to(dtype).unsqueeze(-1)], dim=-1)
        y = F.conv1d(window, self.conv.weight.to(dtype), groups=x_t.size(-1)).squeeze(-1)
        next_state = window[..., 1:] if self.kernel_size > 1 else window[..., :0]
        return y.to(x_t.dtype), next_state


class RightBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.ssm = HiPPOChannel(cfg)
        self.conv = ConvChannel(cfg)
        self.gate = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        use_moe = (cfg.moe_num_experts > 0 and
                   ((layer_idx + 1) % cfg.moe_every == 0 or layer_idx == cfg.n_layer - 1))
        self.moe = SparseMoE(
            cfg.n_embd, cfg.moe_num_experts, cfg.moe_expert_ratio, cfg.moe_top_k,
            cfg.moe_aux_loss_scale, cfg.moe_router_z_loss_scale,
        ) if use_moe else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nx = _norm(x)
        dtype = _work_dtype(x, self.gate.weight)
        g = torch.sigmoid(F.linear(nx.to(dtype), self.gate.weight.to(dtype))).to(x.dtype)
        x = x + g * self.ssm(nx) + (1 - g) * self.conv(nx)
        if self.moe is not None:
            x = x + self.moe(_norm(x))
        return x

    def step(self, x_t: torch.Tensor, state):
        ssm_state, conv_state = state
        nx = _norm(x_t)
        dtype = _work_dtype(x_t, self.gate.weight)
        g = torch.sigmoid(F.linear(nx.to(dtype), self.gate.weight.to(dtype))).to(x_t.dtype)
        y_ssm, ssm_state = self.ssm.step(nx, ssm_state)
        y_conv, conv_state = self.conv.step(nx, conv_state)
        x_t = x_t + g * y_ssm + (1 - g) * y_conv
        if self.moe is not None:
            x_t = x_t + self.moe(_norm(x_t))
        return x_t, (ssm_state, conv_state)


@dataclass
class RightBrainConfig:
    n_embd: int = 768
    n_layer: int = 12
    ssm_state: int = 16
    conv_kernel: int = 7
    dt: float = 0.1
    causal_conv: bool = True
    moe_num_experts: int = 0
    moe_expert_ratio: float = 4.0
    moe_top_k: int = 1
    moe_every: int = 1
    moe_aux_loss_scale: float = 0.01
    moe_router_z_loss_scale: float = 0.001


class RightBrain(nn.Module):
    def __init__(self, config: RightBrainConfig):
        super().__init__()
        if config.n_layer <= 0 or config.moe_every <= 0:
            raise ValueError("n_layer and moe_every must be positive")
        self.config = config
        self.blocks = nn.ModuleList([RightBlock(config, i) for i in range(config.n_layer)])
        self._last_aux_loss = None

    def init_weights(self):
        for blk in self.blocks:
            blk.ssm.init_weights()
            nn.init.normal_(blk.conv.conv.weight, std=0.02 / math.sqrt(blk.conv.kernel_size))
            nn.init.zeros_(blk.gate.weight)
            if blk.moe is not None:
                blk.moe.init_weights()

    def _finish(self, x):
        terms = [blk.moe.last_aux_loss for blk in self.blocks
                 if blk.moe is not None and blk.moe.last_aux_loss is not None]
        self._last_aux_loss = torch.stack(terms).sum() if terms else None
        return _norm(x)

    def auxiliary_loss(self):
        """Add this scalar to the LM loss when training the right brain."""
        return self._last_aux_loss

    def setup_optimizer(self, matrix_lr: float = 0.02, weight_decay: float = 0.0):
        """Mirror the left brain's shape-grouped Muon routing for FSDP."""
        from FoxSay.optim import MuonAdamW
        params = list(self.parameters())
        assert all(p.ndim >= 2 for p in params), "unexpected scalar right-brain parameter"
        groups = []
        for shape in sorted({p.shape for p in params}):
            groups.append(dict(
                kind="muon", params=[p for p in params if p.shape == shape],
                lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.9,
                weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for blk in self.blocks:
            if blk.moe is not None:
                report = blk.moe.parameter_report()
                inactive += report["total"] - report["active"]
        return {"total": total, "active_upper_bound": total - inactive}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self._finish(x)

    def step(self, x_t: torch.Tensor, states: list) -> tuple:
        if len(states) != len(self.blocks):
            raise ValueError("one recurrent state pair is required per block")
        for i, blk in enumerate(self.blocks):
            x_t, states[i] = blk.step(x_t, states[i])
        return self._finish(x_t), states

    def init_states(self, batch: int, device, dtype):
        return [(blk.ssm.init_state(batch, device, dtype),
                 blk.conv.init_state(batch, device, dtype)) for blk in self.blocks]
