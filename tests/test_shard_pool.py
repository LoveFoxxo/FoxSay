"""
test_shard_pool.py -- runtime on-demand region loading: numerical identity
with full load, residency budget, LRU eviction, and a training step.
Run:  python tests/test_shard_pool.py
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_model():
    from FoxSay.gpt import GPT, GPTConfig
    m = GPT(GPTConfig(n_layer=6, n_embd=64, n_head=4, n_kv_head=4,
                      vocab_size=512, sequence_len=128))
    m.init_weights()
    return m


def test_numerical_identity():
    from FoxSay import brain_shards
    from FoxSay.shard_pool import ShardPool
    m = make_model()
    with tempfile.TemporaryDirectory() as d:
        brain_shards.save_shards(m, d, n_shards=3, step=1)

        # reference: full load, normal forward
        ref = make_model()
        brain_shards.load_shards(ref, d)
        ref.eval()
        idx = torch.randint(0, 512, (2, 16))
        with torch.no_grad():
            y_ref = ref(idx, targets=None)

        # pool: regions swap in on demand
        p = make_model()
        pool = ShardPool(p, d, max_resident=1)
        pool.attach(p)
        p.eval()
        with torch.no_grad():
            y_pool = p(idx, targets=None)

        assert torch.allclose(y_ref, y_pool, atol=1e-5), (y_ref - y_pool).abs().max()
        # budget: region0 + at most max_resident layer regions
        layer_res = [r for r in pool.resident if r != 0]
        assert len(layer_res) <= 1, pool.resident
        assert pool.stats["loads"] >= 2, pool.stats
        assert pool.stats["evicts"] >= 1, pool.stats
    print("shard-pool numerical identity + budget: PASS")


def test_training_step():
    from FoxSay import brain_shards
    from FoxSay.shard_pool import ShardPool
    m = make_model()
    with tempfile.TemporaryDirectory() as d:
        brain_shards.save_shards(m, d, n_shards=2, step=1)
        p = make_model()
        pool = ShardPool(p, d, max_resident=1)
        pool.attach(p)
        p.train()
        opt = torch.optim.AdamW(p.parameters(), lr=1e-4)
        idx = torch.randint(0, 512, (1, 32))
        y = torch.cat([idx[:, 1:], torch.full((1, 1), -1, dtype=idx.dtype)], dim=1)
        loss = p(idx, targets=y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        # second forward after zero_grad: eviction is now legal again
        loss2 = p(idx, targets=y)
        loss2.backward()
        opt.step()
        assert torch.isfinite(loss) and torch.isfinite(loss2)
    print("shard-pool training step (backward/opt/evict): PASS")


if __name__ == "__main__":
    test_numerical_identity()
    test_training_step()
    print("ALL SHARD POOL TESTS PASSED")
