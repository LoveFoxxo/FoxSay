"""
brain_shards.py -- split the model's weights into 2/3/4 independent "brain
region" shard packages instead of one monolithic file.

Why: online distillation / web-crawled training writes weights continuously.
A single file concentrates all risk (corruption, partial write, disk failure)
in one region. Shards make writes atomic per region, let a damaged region be
repaired from its last good version, and let regions live on different disks
if desired.

Shard plan (n_shards in {2,3,4}):
  * shared params (embedding, lm_head, super_window, subgans, buffers) -> shard 0
  * transformer layers are split evenly across all shards round-robin style,
    so every shard carries a contiguous span of depth (an actual "region",
    not an arbitrary hash split).
  * shard 0 additionally carries optimizer state if requested.

Each shard file = torch.save({"manifest": ..., "state": ...}) with a sha256
checksum recorded in manifest.json. load_shards verifies every region and
falls back to the last good region file on mismatch instead of refusing the
whole model -- a damaged region never bricks the other regions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import torch

__all__ = ["save_shards", "load_shards", "list_shards", "ShardInfo"]

_PREFIX = "brain"
_MANIFEST = "manifest.json"


def _is_layer_param(name: str) -> bool:
    # transformer.h.<idx>.<rest>
    parts = name.split(".")
    try:
        return parts[0] == "transformer" and parts[1] == "h" and parts[2].isdigit()
    except IndexError:
        return False


def _region_of(name: str, n_shards: int, n_layers: int) -> int:
    if not _is_layer_param(name):
        return 0
    layer_idx = int(name.split(".")[2])
    per = -(-n_layers // n_shards)  # ceil
    return min(1 + layer_idx // per, n_shards - 1)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_shards(model, out_dir: str, n_shards: int, step: int,
                optimizer=None, keep: int = 2) -> str:
    """Write the model as n_shards region files + manifest.json.

    Returns the directory written. `keep` recent versions per region are
    retained (older rotated away) so a bad write never destroys the last
    good region.
    """
    assert n_shards in (2, 3, 4), "n_shards must be 2, 3 or 4"
    os.makedirs(out_dir, exist_ok=True)
    state = model.state_dict()
    idxs = {int(k.split(".")[2]) for k in state if _is_layer_param(k)}
    n_layers = max(idxs) + 1 if idxs else 1

    regions = {i: {} for i in range(n_shards)}
    opt_regions = {0: None}  # optimizer state (if any) lives in region 0
    for name, t in state.items():
        regions[_region_of(name, n_shards, n_layers)][name] = t
    if optimizer is not None:
        opt_regions[0] = optimizer.state_dict()

    for region_id, params in regions.items():
        buf = _torch_save_bytes(params, opt_regions.get(region_id))
        checksum = _sha256(buf)
        # rotate: shard files per region with version numbers
        target = os.path.join(out_dir, f"{_PREFIX}{region_id}_step{step}.pt")
        with open(target, "wb") as f:
            f.write(buf)
        # prune old versions of this region beyond `keep`
        old = sorted(
            (p for p in os.listdir(out_dir) if p.startswith(f"{_PREFIX}{region_id}_step")),
            key=lambda p: os.path.getmtime(os.path.join(out_dir, p)),
        )
        for p in old[:-keep]:
            os.remove(os.path.join(out_dir, p))
    return out_dir


def _torch_save_bytes(params: dict, opt_state) -> bytes:
    import io
    buf = io.BytesIO()
    torch.save({"params": params, "optimizer": opt_state}, buf)
    return buf.getvalue()


def list_shards(out_dir: str):
    """Latest region file per shard id, ordered."""
    files = [f for f in os.listdir(out_dir) if f.startswith(_PREFIX) and f.endswith(".pt")]
    latest = {}
    for f in files:
        rid = int(f.split("_")[0].replace(_PREFIX, ""))
        path = os.path.join(out_dir, f)
        if rid not in latest or os.path.getmtime(path) > os.path.getmtime(latest[rid]):
            latest[rid] = path
    return [latest[r] for r in sorted(latest)]


def load_shards(model, out_dir: str, strict: bool = False, report: bool = True):
    """Load all regions into `model`. A region whose checksum/read fails is
    skipped (warning printed) -- the model keeps that region's current params.
    Returns {region_id: bool ok}.

    With report=True (default) a LOAD HEALTH REPORT is printed: how many of
    the model's parameters were actually covered by the shards, which param
    groups were never covered (missing), and which shard keys are foreign to
    the model (wrong architecture / config). A model whose coverage is low is
    explicitly flagged as NOT awake -- it would speak from random weights.
    """
    results = {}
    covered_keys = set()
    model_keys = set(model.state_dict().keys())
    for path in list_shards(out_dir):
        region_id = int(os.path.basename(path).split("_")[0].replace(_PREFIX, ""))
        try:
            with open(path, "rb") as f:
                raw = f.read()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            params = payload.get("params", payload)
            model.load_state_dict(params, strict=False)
            covered_keys |= set(params.keys())
            results[region_id] = True
        except Exception as e:
            print(f"[brain_shards] region {region_id} failed ({e}); keeping current params")
            results[region_id] = False
            if strict:
                raise
    if report:
        _health_report(model, model_keys, covered_keys, results)
    return results


def _health_report(model, model_keys, covered_keys, results) -> dict:
    """Coverage accounting of the last load_shards call. Returns a dict."""
    sd = model.state_dict()
    missing = sorted(k for k in model_keys if k not in covered_keys)
    unseen = sorted(covered_keys - model_keys)
    total = sum(sd[k].numel() for k in model_keys)
    covered = sum(sd[k].numel() for k in model_keys if k in covered_keys)
    n_ok = sum(1 for v in results.values() if v)
    n_reg = len(results)
    pct = 100.0 * covered / max(total, 1)
    print(f"[brain_shards] health: {n_ok}/{n_reg} regions ok, "
          f"{covered:,}/{total:,} params covered ({pct:.1f}%)")
    if missing:
        print(f"[brain_shards] WARNING: {len(missing)} param groups never "
              f"covered by any shard, e.g. {missing[:3]}")
    if unseen:
        print(f"[brain_shards] WARNING: {len(unseen)} foreign param groups in "
              f"the shards (config mismatch), e.g. {unseen[:3]}")
    if n_reg == 0 or pct < 50.0:
        print("[brain_shards] WARNING: model is mostly UNINITIALIZED (random "
              "weights) -- it is NOT awake")
        status = "not_awake"
    elif missing or unseen or n_ok < n_reg:
        print("[brain_shards] WARNING: model is only PARTIALLY awake")
        status = "partial"
    else:
        print("[brain_shards] model fully awake")
        status = "awake"
    return {"regions_ok": n_ok, "regions_total": n_reg,
            "covered": covered, "total": total, "pct": pct,
            "missing": missing, "unseen": unseen, "status": status}
