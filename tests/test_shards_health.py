"""
test_shards_health.py -- load health accounting for brain shards.

Asserts the WAKE-UP contract of load_shards:
  * full shard set  -> 100% coverage, status 'awake'
  * one region gone -> coverage < 100%, missing groups reported, 'partial'
  * empty dir        -> 0 regions, 'not_awake' (never silently random)
  * foreign keys     -> 'unseen' reported (config mismatch detection)
"""

import os
import shutil
import tempfile

import torch
import torch.nn as nn

from FoxSay import brain_shards


class LayeredModel(nn.Module):
    """A model whose parameter names follow FoxSay's transformer.h.<idx>.*
    layout, so save_shards really splits layers across regions."""

    def __init__(self, n_layers: int = 3, dim: int = 8):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "h": nn.ModuleList([nn.Linear(dim, dim, bias=False)
                                for _ in range(n_layers)])
        })
        self.head = nn.Linear(dim, dim, bias=False)


def _save(tmp, n_layers=3, n_shards=2):
    m = LayeredModel(n_layers)
    brain_shards.save_shards(m, tmp, n_shards=n_shards, step=1)
    return m


def test_full_shard_set_is_awake():
    with tempfile.TemporaryDirectory() as tmp:
        _save(tmp)
        m = LayeredModel()
        health = brain_shards._health_report(
            m, set(m.state_dict().keys()), set(m.state_dict().keys()),
            {0: True, 1: True})
        assert health["status"] == "awake"
        assert health["pct"] == 100.0
        assert not health["missing"] and not health["unseen"]


def test_missing_region_is_partial():
    with tempfile.TemporaryDirectory() as tmp:
        _save(tmp)
        os.remove(os.path.join(tmp, "brain1_step1.pt"))
        m = LayeredModel()
        res = brain_shards.load_shards(m, tmp, report=False)
        assert len(res) == 1 and res[0] is True
        model_keys = set(m.state_dict().keys())
        covered = {k for k in model_keys if k.startswith("transformer.h.0")
                   or k.startswith("transformer.h.1") or k.startswith("head")}
        health = brain_shards._health_report(m, model_keys, covered, res)
        assert health["status"] == "partial"
        assert health["pct"] < 100.0
        assert any("transformer.h.2" in k for k in health["missing"])


def test_empty_dir_is_not_awake():
    with tempfile.TemporaryDirectory() as tmp:
        m = LayeredModel()
        res = brain_shards.load_shards(m, tmp, report=False)
        assert res == {}
        health = brain_shards._health_report(m, set(m.state_dict().keys()),
                                             set(), res)
        assert health["status"] == "not_awake"
        assert health["covered"] == 0


def test_foreign_keys_detected():
    with tempfile.TemporaryDirectory() as tmp:
        _save(tmp, n_layers=1, n_shards=2)
        # overwrite brain0 with a payload carrying a foreign param name
        foreign = {"params": {"alien.synapse": torch.zeros(4, 4)},
                   "optimizer": None}
        torch.save(foreign, os.path.join(tmp, "brain0_step1.pt"))
        m = LayeredModel(1)
        res = brain_shards.load_shards(m, tmp, report=False)
        assert len(res) == 2 and res[0] is True and res[1] is True
        # covered after real load: the foreign key (tolerated) + brain1's layer
        covered = {"alien.synapse", "transformer.h.0.weight"}
        health = brain_shards._health_report(m, set(m.state_dict().keys()),
                                             covered, res)
        assert health["unseen"] == ["alien.synapse"]
        assert health["missing"] == ["head.weight"]
        assert health["status"] == "partial"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(fns)} tests passed")
