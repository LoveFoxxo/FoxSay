"""
subgans.py -- 2/3/4 sub-GANs mounted at different depths of the trunk.

The internal verification loop (ACT-style latent cycles in gpt.py) uses a
single gate (gan/gate.py) around the whole trunk.  That concentrates all
"recurrent thinking" pressure in one place: a single verifier sees only the
final stream, and its gradient path crosses every layer on every cycle.

This module splits the loop's thinking into a cluster of sub-GANs attached at
2/3/4 chosen depths.  Each SubGAN = tiny verifier (score of the stream at
that depth) + gate that blends a refined stream back in.  Mounting them at
different depths means different "brain regions" reason about the stream
where their information is freshest.

Robustness rules (the "will not hurt" contract):
  * LONG-THINK ONLY: a sub-GAN activates only on cycles >= 2 (cycle 1 is the
    plain forward pass -> zero interference with the reward/penalty baseline).
  * NO REVERB: each sub-GAN has a per-cycle activation budget (default 1);
    once exhausted it becomes a pass-through.  A global budget caps total
    activations per cycle, so 2/3/4 sub-GANs can never ping-pong forever.
  * GRADIENT ISOLATION: the refined stream is injected as a detached VALUE
    (x += gate.detach() * refined.detach()), so neither sub-GAN parameters
    nor the trunk receive any gradient through the injection path. Sub-GANs
    are trained ONLY by their own verifier rank loss (see gan/verifier.py),
    exactly like the existing GAN gate -- reward/penalty backprop on the
    trunk is never contaminated.
  * DEFAULT OFF: GPTConfig.subgan_layers = None keeps legacy behavior
    bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SubGANConfig:
    n_embd: int = 256
    hidden: int = 128          # verifier MLP hidden size
    gate_lr_scale: float = 0.0  # blend init = pass-through (safe at birth)
    max_activations_per_cycle: int = 1
    max_total_per_cycle: int = 4  # global anti-reverb budget


class SubGAN(nn.Module):
    """One sub-GAN at one depth: verifier score + gate-refined stream."""

    def __init__(self, cfg: SubGANConfig, init_params: bool = True):
        super().__init__()
        n = cfg.n_embd
        self.verifier = nn.Sequential(
            nn.Linear(n, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 1)
        )
        # gate logit: how much refined stream to admit (0 at init -> pass-through)
        self.gate = nn.Linear(n, 1, bias=True)
        # refinement head keeps dimensionality; zero-init -> identity-ish start
        self.refine = nn.Sequential(
            nn.Linear(n, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, n)
        )
        if init_params:
            self.reset_parameters()
        self.max_activations_per_cycle = cfg.max_activations_per_cycle
        self.activations_this_cycle = 0

    def reset_parameters(self):
        """Zero-init gate/refine-out -> pass-through at birth (meta-safe when
        called from GPT.init_weights on real tensors)."""
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.refine[2].weight)
        nn.init.zeros_(self.refine[2].bias)
        for lin in self.verifier:
            if isinstance(lin, nn.Linear):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)

    def begin_cycle(self):
        self.activations_this_cycle = 0

    def process(self, x: torch.Tensor):
        """x: (B, T, C). Returns (blended_x, score, gate_val)."""
        pooled = x.detach().mean(dim=1)          # verifier sees depth-local stream
        score = self.verifier(pooled)            # (B,1) process score
        gate_val = torch.sigmoid(self.gate(pooled))   # (B,1) admit ratio
        refined = self.refine(x)                 # (B,T,C) depth-local refinement
        # value-only injection: no gradient crosses this line in either
        # direction, so the trunk's reward/penalty backprop stays untouched
        x = x + (gate_val.detach().unsqueeze(1) * refined.detach())
        return x, score, gate_val


class SubGANCluster(nn.Module):
    """2/3/4 sub-GANs mounted at given depth fractions of the trunk."""

    def __init__(self, n_embd: int, layers=None, n_subgans: int = 2, **cfg_kw):
        super().__init__()
        init_params = cfg_kw.pop("init_params", True)
        self.n_embd = n_embd
        # attach fractions: 2 -> [0.5, 0.75]; 3 -> [1/3, 2/3, 0.875];
        # 4 -> [0.25, 0.5, 0.75, 0.875]; explicit list overrides
        if layers is None:
            if n_subgans == 2:
                layers = [0.5, 0.75]
            elif n_subgans == 3:
                layers = [1 / 3, 2 / 3, 0.875]
            else:
                layers = [0.25, 0.5, 0.75, 0.875]
        assert len(layers) in (2, 3, 4), "need 2, 3 or 4 sub-GAN attach points"
        self.attach_fractions = sorted(set(float(f) for f in layers))
        self.subgans = nn.ModuleDict()
        for i, frac in enumerate(self.attach_fractions):
            cfg = SubGANConfig(n_embd=n_embd, **cfg_kw)
            self.subgans[f"sg{i}"] = SubGAN(cfg, init_params=init_params)
        self.cycle = 0                     # 0-based; activates only when >= 1
        self.max_total = max(cfg_kw.get("max_total_per_cycle", 4),
                            len(self.subgans))
        self._attach_cache: dict = {}
        # per-forward bookkeeping (not checkpointed semantics)
        self.last_scores: dict = {}

    def reset_parameters(self):
        for sg in self.subgans.values():
            sg.reset_parameters()

    def begin_cycle(self, cycle: int):
        self.cycle = cycle
        self.last_scores = {}
        for sg in self.subgans.values():
            sg.begin_cycle()

    def attach_map(self, n_layers: int) -> dict:
        """layer_idx -> subgan name (first layer at each fraction)."""
        if n_layers in self._attach_cache:
            return self._attach_cache[n_layers]
        m = {max(0, int(n_layers * f) - 1): name
             for name, f in zip(self.subgans.keys(), self.attach_fractions)}
        self._attach_cache[n_layers] = m
        return m

    def maybe_think(self, x: torch.Tensor, layer_idx: int, n_layers: int):
        """Called after every trunk layer. Pass-through unless:
        (1) we are in a long-think cycle (cycle >= 1, i.e. any refine pass),
        (2) a sub-GAN is attached exactly here,
        (3) it still has activation budget and the global budget is not spent.
        """
        if self.cycle < 1:
            return x
        amap = self.attach_map(n_layers)
        if layer_idx not in amap:
            return x
        total_used = sum(sg.activations_this_cycle for sg in self.subgans.values())
        if total_used >= self.max_total:
            return x
        name = amap[layer_idx]
        sg = self.subgans[name]
        if sg.activations_this_cycle >= sg.max_activations_per_cycle:
            return x
        sg.activations_this_cycle += 1
        x, score, gate = sg.process(x)
        self.last_scores[name] = float(score.detach().mean().item())
        return x

    def verifier_scores(self):
        """Dict of sub-GAN process scores this cycle (for PRM reward)."""
        return dict(self.last_scores)

    def parameters(self, recurse: bool = True):
        # only sub-GAN params; the caller decides how (if at all) to train them
        return self.subgans.parameters(recurse=recurse)
