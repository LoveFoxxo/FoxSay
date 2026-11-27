"""
modality_window.py -- hierarchical sliding window (main window + sub-windows).

The standard sliding-window pattern ("SSSL" etc.) is upgraded to a
hierarchical scheme:

    SuperWindow (main window, one per model)
      `-- SubWindow x4  (text / audio / image / video)
            each judges what modality the current attention focus is on
            and proposes its own (left, right) window;
      `-- the main window aggregates all proposals into one unified
          (left, right) format and hands it to flash attention.

Design notes:
  * The router is a single Linear(n_embd -> 4) over mean-pooled hidden
    states -- deliberately tiny so it adds negligible FLOPs and can be
    trained by the same optimizer as the trunk.
  * Aggregation is a softmax-weighted average of the sub-window proposals,
    EMA-smoothed, then snapped to the FA3 tile size (128). The main window
    always clamps to [min sub proposal, base left] so it can never widen the
    window beyond the layer's static pattern (the pattern stays the hard cap;
    the sub-windows only specialize inside it).
  * `last_stats` exposes the most recent routing weights for monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

# canonical modality registry: index must stay fixed for checkpoint compat
MODALITIES = ("text", "audio", "image", "video")

# default sub-window proposals as a multiplicative factor on the base left
# window: audio benefits from longer temporal context, image/video even more.
DEFAULT_SUB_FACTORS = {"text": 1.0, "audio": 1.5, "image": 2.0, "video": 2.5}

_TILE = 128  # FA3 window tile alignment


def _snap(v: float, base: int) -> int:
    """Snap a window length to the FA3 tile grid; NEVER exceed base.

    The static layer window (base) is the hard cap -- sub-windows may only
    specialize *inside* it, never widen attention beyond it.
    """
    v = max(1.0, min(float(v), float(base)))
    snapped = int(v / _TILE + 0.5) * _TILE  # round-half-up, not banker's
    return max(1, min(snapped, base))


@dataclass
class SubWindow:
    """One modality's sliding window. Judges focus and proposes a window."""

    modality: str
    left_factor: float = 1.0      # proposal = base_left * left_factor
    right: int = 0                # causal default; can allow lookahead later

    def propose(self, base_left: int) -> tuple:
        return (self.left_factor * base_left, self.right)


@dataclass
class SuperWindow:
    """Main sliding window ruling a set of sub-windows; aggregates proposals.

    Dual-brain routing: when brain_gate=True, a second tiny head maps each
    modality's focus onto a (left, right) share pair, i.e. every modality is
    free to lean into the left brain (attention) or the right brain (HiPPO
    temporal). A load-balance auxiliary loss keeps one brain from starving.
    """

    router: nn.Module
    sub_windows: dict = field(default_factory=dict)
    smooth: float = 0.9           # EMA factor for proposal smoothing
    brain_router: nn.Module = None  # (4, 2) per-modality left/right shares
    # runtime state (not checkpointed semantics, only diagnostics)
    last_stats: dict = field(default_factory=dict)

    @classmethod
    def build(cls, n_embd: int, sub_factors=None, smooth: float = 0.9,
              brain_gate: bool = False,
              dtype=None, device=None, init_params: bool = True):
        """Factory: router + one sub-window per modality (+ optional brain gate).

        init_params=False defers weight init to the caller's init_weights()
        (required under meta-device construction in GPT.__init__).
        """
        factors = dict(DEFAULT_SUB_FACTORS)
        if sub_factors:
            factors.update(sub_factors)
        sub_windows = {
            m: SubWindow(modality=m, left_factor=factors.get(m, 1.0))
            for m in MODALITIES
        }
        router = nn.Linear(n_embd, len(MODALITIES), bias=True)
        brain_router = None
        if brain_gate:
            brain_router = nn.Linear(n_embd, 2 * len(MODALITIES), bias=True)
            if init_params:
                # untrained => uniform 50/50 split per modality
                nn.init.zeros_(brain_router.weight)
                nn.init.zeros_(brain_router.bias)
            if dtype is not None:
                brain_router = brain_router.to(dtype)
            if device is not None:
                brain_router = brain_router.to(device)
        if init_params:
            # start roughly uniform so an untrained model keeps the static window
            nn.init.zeros_(router.weight)
            nn.init.zeros_(router.bias)
        if dtype is not None:
            router = router.to(dtype)
        if device is not None:
            router = router.to(device)
        return cls(router=router, sub_windows=sub_windows, smooth=smooth,
                   brain_router=brain_router)

    def route(self, x: torch.Tensor) -> torch.Tensor:
        """Modality focus weights from hidden states.

        x: (B, T, C) -> mean-pool over tokens -> (B, C) -> softmax (B, 4).
        Returns the mean routing vector across the batch, (4,), for a single
        unified window per layer (a per-token window would break flash attn).
        """
        z = x.detach().float().mean(dim=(0, 1))  # (C,) pooled across batch+tokens
        logits = self.router(z)                   # (4,)
        return torch.softmax(logits, dim=-1)      # (4,)

    def route_brain(self, x: torch.Tensor) -> torch.Tensor:
        """Per-modality (left, right) brain shares, shape (4, 2).

        Route the SAME pooled hidden state through the brain gate; softmax
        over the brain axis for each modality. Untrained (zero init) this is
        exactly 0.5/0.5 -- the hemispheres split evenly by default.
        """
        if self.brain_router is None:
            dev = x.device
            return torch.full((len(MODALITIES), 2), 0.5, device=dev, dtype=torch.float32)
        z = x.detach().float().mean(dim=(0, 1))
        logits = self.brain_router(z).view(len(MODALITIES), 2)
        return torch.softmax(logits, dim=-1)

    def load_balance_loss(self, shares: torch.Tensor = None, x: torch.Tensor = None):
        """Starve-prevention: penalize deviation from a 50/50 total brain load.

        L = (mean_modality left_share - 0.5)^2 + (max share - min share)^2/16
        The second term is an anti-collapse regularizer on the (4,2) matrix.
        """
        if shares is None:
            shares = self.route_brain(x)
        left = shares[:, 0]
        balance = (left.mean() - 0.5) ** 2
        anti_collapse = ((shares.max() - shares.min()) / 4.0) ** 2
        return balance + anti_collapse

    def aggregate(self, weights: torch.Tensor, base_left: int) -> tuple:
        """Main window: unify sub-window proposals into one (left, right)."""
        proposals = []
        for i, m in enumerate(MODALITIES):
            left, right = self.sub_windows[m].propose(base_left)
            proposals.append(left)
        p = torch.tensor(proposals, dtype=weights.dtype, device=weights.device)
        unified = float((weights * p).sum().item())
        # EMA smoothing against sudden modality flips
        prev = self.last_stats.get("left")
        if prev is not None:
            unified = self.smooth * prev + (1.0 - self.smooth) * unified
        left = _snap(unified, base_left)
        right = self.sub_windows["text"].right
        self.last_stats = {
            "left": left,
            "weights": {m: round(float(w), 4) for m, w in zip(MODALITIES, weights.tolist())},
        }
        return (left, right)

    def __call__(self, x: torch.Tensor, base_window: tuple) -> tuple:
        """Full path: route modality focus, aggregate, return unified window."""
        base_left = base_window[0]
        if base_left <= 0:  # (-1, 0) = full context: nothing to specialize
            return base_window
        weights = self.route(x)
        if self.brain_router is not None:
            shares = self.route_brain(x)
            self.last_stats["brain_shares"] = {
                m: [round(float(s0), 4), round(float(s1), 4)]
                for m, (s0, s1) in zip(MODALITIES, shares.tolist())
            }
        return self.aggregate(weights, base_left)

    def parameters(self):
        if self.brain_router is not None:
            return list(self.router.parameters()) + list(self.brain_router.parameters())
        return self.router.parameters()
