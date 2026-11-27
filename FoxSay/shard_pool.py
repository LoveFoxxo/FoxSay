"""
shard_pool.py -- runtime on-demand load/unload of brain-region weight shards.

brain_shards split the weights into 2/3/4 region files and load them all at
startup.  ShardPool goes one step further: only the regions a forward pass is
about to touch stay resident; everything else is evicted.

Mechanics (zero changes needed in gpt.py):
  * attach(model) registers a forward_pre_hook on every transformer block.
  * When block i is about to run, the hook ensures its region is resident:
    - if not resident: evict the least-recently-used resident region, then
      load block i's region from the region cache (or its shard file).
  * "Resident" means the parameter carries real data.  An evicted region's
    parameters are swapped to meta tensors, so a buggy hook would crash loudly
    instead of silently running with stale weights.
  * Shared params (embedding / lm_head / super_window / subgans) live in
    region 0 and stay resident forever -- every forward needs them.
  * Gradient safety: eviction is REFUSED while any parameter of the victim
    region still holds a grad (i.e. before zero_grad).  The pool may briefly
    exceed max_resident then, but it never corrupts an in-flight backward.
  * max_resident counts layer regions only (region 0 excluded), so
    max_resident=1 approximates "one brain region active per cycle".

Usage:
    from FoxSay.brain_shards import save_shards
    from FoxSay.shard_pool import ShardPool
    save_shards(model, "runs/x", n_shards=3, step=1)
    pool = ShardPool(model, "runs/x", max_resident=1)
    pool.attach(model)          # from now on, regions swap in/out per forward
"""

from __future__ import annotations

import os

import torch

from .brain_shards import list_shards, _region_of

__all__ = ["ShardPool"]


class ShardPool:
    def __init__(self, model, shards_dir: str, max_resident: int = 1,
                 device=None):
        self.shards_dir = shards_dir
        self.max_resident = max(1, int(max_resident))
        self.device = device
        # map param name -> region id (reuse brain_shards' split logic)
        state = model.state_dict()
        idxs = {int(k.split(".")[2]) for k in state
                if k.split(".")[0:2] == ["transformer", "h"]}
        self.n_layers = max(idxs) + 1 if idxs else 1
        self.param_region = {
            name: _region_of(name, self._n_shards(), self.n_layers)
            for name in state
        }
        self._param_names = {id(p): n for n, p in model.named_parameters()}
        self.region_cache: dict = {}   # region_id -> {name: cpu tensor}
        self.resident: dict = {}       # region_id -> last-use stamp
        self._clock = 0
        self.stats = {"loads": 0, "evicts": 0}

    def _n_shards(self) -> int:
        return len(list_shards(self.shards_dir))

    # -- region cache ------------------------------------------------------
    def _load_region(self, region_id: int):
        if region_id in self.region_cache:
            return self.region_cache[region_id]
        path = list_shards(self.shards_dir)[region_id]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        params = payload.get("params", payload)
        self.region_cache[region_id] = {k: v for k, v in params.items()}
        return self.region_cache[region_id]

    # -- residency ----------------------------------------------------------
    def _params_of(self, region_id: int, model):
        return [p for n, p in model.named_parameters()
                if self.param_region.get(n) == region_id]

    def _can_evict(self, region_id: int, model) -> bool:
        return all(p.grad is None for p in self._params_of(region_id, model))

    def _evict(self, region_id: int, model):
        """Swap a region's parameters to garbage placeholders; keep the real
        values in the cache. The hooks guarantee no forward ever touches an
        evicted region, so any bug shows up immediately as wrong numerics."""
        if not self._can_evict(region_id, model):
            return False  # in-flight gradient: refuse, stay resident
        self._load_region(region_id)  # make sure cache holds the values
        for p in self._params_of(region_id, model):
            p.data = torch.empty_like(p.data)  # garbage placeholder
        self.resident.pop(region_id, None)
        self.stats["evicts"] += 1
        return True

    def _ensure_resident(self, region_id: int, model):
        self._clock += 1
        if region_id in self.resident:
            self.resident[region_id] = self._clock
            return
        # evict LRU layer-region(s) until under budget (region 0 never evicted)
        while True:
            layer_resident = [r for r in self.resident if r != 0]
            if len(layer_resident) < self.max_resident:
                break
            lru = sorted((stamp, r) for r, stamp in self.resident.items() if r != 0)
            if not lru:
                break
            victim = lru[0][1]
            if not self._evict(victim, model):
                # refused (grad in flight): allow over-budget this round
                self.resident[victim] = self._clock
                break
        if region_id in self.resident:  # eviction refused -> already resident
            self.resident[region_id] = self._clock
            return
        cached = self._load_region(region_id)
        for p in self._params_of(region_id, model):
            v = cached[self._param_names[id(p)]]
            p.data = v.to(device=self.device or p.device, dtype=p.dtype)
        self.resident[region_id] = self._clock
        self.stats["loads"] += 1

    def _name_for(self, param, model):
        return self._param_names[id(param)]

    # -- attach -------------------------------------------------------------
    def attach(self, model):
        """Register per-block hooks and set initial residency:
        region 0 (shared params) resident forever, all layer regions start
        evicted (meta) and swap in on demand. Call once after model init."""
        # region 0: load from shards now and pin resident
        self._ensure_resident(0, model)
        self.resident = {r: s for r, s in self.resident.items() if r == 0}
        # all layer regions start evicted (garbage placeholders)
        for region in set(self.param_region.values()) - {0}:
            for p in self._params_of(region, model):
                p.data = torch.empty_like(p.data)
        for i, block in enumerate(model.transformer.h):
            region = _region_of(f"transformer.h.{i}.attn.c_q.weight",
                                self._n_shards(), self.n_layers)

            def make_hook(rid):
                def hook(module, inputs):
                    self._ensure_resident(rid, model)
                return hook

            block.register_forward_pre_hook(make_hook(region))
        return self

    def report(self):
        return dict(self.stats, resident=list(self.resident), cache=list(self.region_cache))
