"""
test_brain_subgans.py -- sharded weights (2/3/4 regions) round-trip and the
multi-sub-GAN cluster (activation budget, gradient isolation, pass-through at
birth).  Run:  python tests/test_brain_subgans.py
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_model(**kw):
    from FoxSay.gpt import GPT, GPTConfig
    cfg = GPTConfig(n_layer=8, n_embd=64, n_head=4, n_kv_head=4,
                    vocab_size=512, sequence_len=128,
                    loop_max_cycles=3, **kw)
    m = GPT(cfg)
    m.init_weights()
    return m


def test_shard_roundtrip():
    from FoxSay import brain_shards
    m = make_model()
    for n in (2, 3, 4):
        with tempfile.TemporaryDirectory() as d:
            brain_shards.save_shards(m, d, n, step=1)
            files = os.listdir(d)
            assert len([f for f in files if f.endswith(".pt")]) == n, files
            m2 = make_model()
            res = brain_shards.load_shards(m2, d)
            assert all(res.values()), res
            for k, v in m.state_dict().items():
                assert torch.equal(m2.state_dict()[k], v), k
            # rotation: saving again keeps newest versions only
            brain_shards.save_shards(m, d, n, step=2, keep=2)
            brain_shards.save_shards(m, d, n, step=3, keep=2)
            pt_files = [f for f in os.listdir(d) if f.endswith(".pt")]
            assert len(pt_files) <= 2 * n, pt_files
    print("shard roundtrip (2/3/4): PASS")


def test_subgan_cluster():
    from FoxSay.gan.subgans import SubGANCluster
    cl = SubGANCluster(64, layers=[0.5, 0.75], init_params=False)
    cl.reset_parameters()
    x = torch.randn(2, 8, 64, requires_grad=True)
    n_layers = 8
    amap = cl.attach_map(n_layers)
    assert len(amap) == 2 and len(cl.subgans) == 2

    # cycle 0 (commit pass): pass-through, no activation
    cl.begin_cycle(0)
    y0 = cl.maybe_think(x, list(amap)[0], n_layers)
    assert y0 is x and not cl.last_scores

    # cycle 1+ (refine pass): activates at attach layers, zero gate -> identity value
    cl.begin_cycle(1)
    y = cl.maybe_think(x, list(amap)[0], n_layers)
    assert y is not x and cl.last_scores
    # zero-init gate: y == x numerically (pass-through at birth)
    assert torch.allclose(y, x, atol=1e-6)

    # gradient isolation: NO gradient crosses the injection line at all
    loss = y.sum()
    loss.backward()
    assert cl.subgans["sg0"].gate.weight.grad is None, "gate must be detached"
    assert cl.subgans["sg0"].refine[2].weight.grad is None, "refine must be detached"
    assert x.grad is not None  # main path still flows through the addition

    # anti-reverb budget: after max_total activations, pass-through again
    cl.begin_cycle(2)
    budget = cl.max_total
    attach_layers = sorted(amap)
    out = x
    n_activated = 0
    for li in attach_layers * 3:  # keep feeding the same attach points
        out = cl.maybe_think(out, li, n_layers)
        n_activated += 1
    total_used = sum(sg.activations_this_cycle for sg in cl.subgans.values())
    assert total_used <= budget, total_used
    print("sub-GAN cluster (activation/budget/grad-isolation): PASS")


def test_gpt_with_subgans_forward():
    m = make_model(subgan_layers=[0.5, 0.75])
    m.eval()
    idx = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        y = m(idx, targets=None)
    assert tuple(y.shape) == (2, 16, 512)
    assert m.subgan_cluster is not None
    assert len(m.subgan_cluster.subgans) == 2
    print("GPT forward with 2 sub-GANs mounted: PASS")


if __name__ == "__main__":
    test_shard_roundtrip()
    test_subgan_cluster()
    test_gpt_with_subgans_forward()
    print("ALL BRAIN/SUBGANS TESTS PASSED")
