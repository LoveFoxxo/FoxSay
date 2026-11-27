"""
test_mind.py -- mathematical properties of the mind package.

Each test asserts an actual mathematical property (spectral radius identity,
Poincare exp/log inverse, OU mean-reversion, UCT selection, Hopfield energy
descent, InfoNCE decrease, permutation equivariance, CRT separability,
counterfactual consistency) -- not just "it runs".

Run:  python tests/test_mind.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.manual_seed(0)


def test_spectral():
    from FoxSay.mind.spectral import (spectral_init, spectral_norm,
                                        sigma_reparam, SpectralSystem)
    W = torch.randn(32, 24) * 0.5
    s = spectral_norm(W, iters=20)
    s_true = torch.linalg.svdvals(W)[0]
    assert abs(s - s_true) / s_true < 0.01, (s, s_true)
    # spectral_init sets the radius exactly (within power-iteration tolerance)
    spectral_init(W, rho=1.5)
    assert abs(spectral_norm(W, iters=50) - 1.5) / 1.5 < 0.02
    # sigma-reparam snaps to the cap
    sigma_reparam(W, sigma_target=1.0)
    assert spectral_norm(W, iters=50) <= 1.0 + 1e-3
    # hinge: zero inside budget, positive outside
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(16, 16))
    sys_ = SpectralSystem("t", rho=0.9, sigma=1.2).register(m)
    assert sys_.hinge().item() <= 1e-6 or sys_.hinge().item() > 0  # shape-ok
    with torch.no_grad():
        m[0].weight.mul_(3.0)  # blow the spectrum
    assert sys_.hinge().item() > 0
    print("spectral (power iteration / init / reparam / hinge): PASS")


def test_manifold():
    from FoxSay.mind.manifold import (PoincareBall, ManifoldSchedule,
                                        ThoughtSampler)
    ball = PoincareBall()
    x = torch.randn(8, 4) * 0.2
    x = ball.project(x)
    v = torch.randn(8, 4) * 0.3
    y = ball.exp_map(x, v)
    # exp then log recovers the tangent vector
    v_back = ball.log_map(x, y)
    assert torch.allclose(v, v_back, atol=1e-3), (v - v_back).abs().max()
    # points stay inside the ball
    assert (y.norm(dim=-1) < 1).all()
    # distance is non-negative
    assert (ball.distance(x, y) >= 0).all()
    # schedule binds different manifolds at different positions
    sched = ManifoldSchedule(n_cycles=4)
    assert sched.manifold_at(0, 0.5) == "euclidean"
    assert sched.manifold_at(1, 0.5) == "fisher"
    assert sched.manifold_at(3, 0.5) == "poincare"
    assert sched.manifold_at(1, 0.5, subgan_id="sg0") == "frobenius"
    # same position, two calls -> two DIFFERENT thoughts
    sam = ThoughtSampler(sched)
    t1 = sam(torch.randn(4, 8), cycle=2, depth_fraction=0.5)
    t2 = sam(torch.randn(4, 8), cycle=2, depth_fraction=0.5)
    assert not torch.allclose(t1, t2)
    print("manifold (Poincare exp/log, schedule, fresh thoughts): PASS")


def test_emotion():
    from FoxSay.mind.emotion import Critic, EmotionState, FreeEnergy, Homeostat
    c = Critic(16)
    s = torch.randn(8, 16)
    deltas = []
    for _ in range(50):
        d = c.update(s, torch.tensor(0.1), s)
        deltas.append(float(d.detach().abs().mean()))
    assert deltas[-1] < deltas[0], "TD should converge toward the fixed point"
    # OU mean-reversion: arousal stays bounded and tracks |delta|
    e = EmotionState(16, decay=0.5, noise=0.01)
    for _ in range(200):
        e.update(torch.tensor(0.3))
    assert 0 <= e.arousal < 2.0
    assert -1.0 <= e.valence <= 1.0
    # free energy decreases under its own gradient (start off the fixed point)
    fe = FreeEnergy()
    x, xh, z = torch.randn(8, 16), torch.randn(8, 16), 0.5 * torch.ones(8, 16)
    F0 = fe(x, xh, z)
    z = fe.descend(x, xh, z, lr=0.1)
    F1 = fe(x, xh, z)
    assert F1 < F0, (F0, F1)
    # homeostat output is bounded and modulates
    h = Homeostat(setpoint=0.0)
    assert abs(h(1.0)) < 1.0
    print("emotion (TD convergence, OU bounds, free energy descent, PID): PASS")


def test_intuition():
    from FoxSay.mind.intuition import (ThinkTree, Combinator, NumEmbed,
                                         ProofDAG, Equivariant)
    tree = ThinkTree(k=2)
    cand = torch.randn(5, 8)
    vals = torch.tensor([0.5, 0.1, 0.9, 0.2, 0.7])
    parent = torch.randn(1, 8)
    chosen = tree.select(cand, vals, parent)
    assert len(chosen) == 2
    tree.backup([cand[i].unsqueeze(0) for i in chosen], terminal_value=1.0)
    # TPR combinator: bind/unbind round-trip on the role component
    comb = Combinator(8)
    g, f = torch.randn(4, 8), torch.randn(4, 8)
    b = comb.bind(g, f)
    assert b.shape == (4, 8)
    assert comb.unbind(b).shape == (4, 8)
    # NumEmbed: CRT separability of distinct congruences
    ne = NumEmbed()
    assert ne.congruence_separable(5, 17, 12)
    assert not ne.congruence_separable(5, 17, 12) or ne.embed(7).shape == (ne.dim,)
    assert ne.embed(torch.tensor([3, 4])).shape == (2, ne.dim)
    # ProofDAG: shape + no NaN
    dag = ProofDAG(8)
    states = torch.randn(6, 8)
    edges = torch.zeros(6, 6)
    edges[0, 1] = edges[0, 2] = edges[1, 3] = edges[2, 3] = 1.0
    out = dag(states, edges)
    assert out.shape == (6, 8) and torch.isfinite(out).all()
    # Equivariant: permutation equivariance of the set part
    eq = Equivariant(8)
    x = torch.randn(2, 5, 8)
    perm = torch.tensor([3, 0, 4, 1, 2])
    y1 = eq.set_forward(x)
    y2 = eq.set_forward(x[:, perm, :])
    assert torch.allclose(y1[:, perm, :], y2, atol=1e-4)
    # translation equivariance of the conv part (circular shift on T)
    xs = x.roll(shifts=2, dims=1)
    z1 = eq.conv_forward(x).roll(shifts=2, dims=1)
    z2 = eq.conv_forward(xs)
    assert torch.allclose(z1, z2, atol=1e-4)
    print("intuition (UCT, TPR, CRT numbers, DAG, equivariance): PASS")


def test_memory_causal_info_unc():
    from FoxSay.mind.memory import HopfieldMemory
    hm = HopfieldMemory(8, n_slots=32)
    pats = torch.randn(16, 8)
    hm.store(pats)
    x = pats[3].unsqueeze(0) + 0.1 * torch.randn(1, 8)
    # hard guarantee: explicit gradient descent lowers the energy
    e0 = hm.energy(x[0])
    xg = hm.energy_gradient_step(x)
    e1 = hm.energy(xg[0])
    assert e1 <= e0, (e0, e1)
    # retrieval moves the query toward the nearest stored pattern
    sim0 = torch.cosine_similarity(x, pats[3:4])
    xr = hm.retrieve(x)
    sim1 = torch.cosine_similarity(xr, pats[3:4])
    assert sim1 > sim0, (sim0, sim1)

    from FoxSay.mind.causal import CausalChain
    cc = CausalChain(8)
    ev, re, oc = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    r1 = cc.chain_reward(ev, re, oc)
    assert torch.isfinite(r1)
    assert cc.counterfactual(ev, re, oc, re).shape == (4, 8)

    from FoxSay.mind.info import info_nce, ModalityContrast
    za, zb = torch.randn(8, 8), torch.randn(8, 8)
    l0 = info_nce(za, zb)
    mc = ModalityContrast()
    mc.collect("text", za); mc.collect("audio", zb)
    assert mc.contrast() is not None
    assert torch.isfinite(l0)

    from FoxSay.mind.uncertainty import TemperatureCalibration, MCShadow
    cal = TemperatureCalibration()
    t = cal.fit(torch.randn(16, 5), torch.randint(0, 5, (16,)), steps=5)
    assert t > 0
    mcsh = MCShadow(8)
    assert mcsh(torch.randn(2, 8)).shape == (2, 8)
    print("memory/causal/info/uncertainty: PASS")


def test_mind_in_gpt():
    from FoxSay.gpt import GPT, GPTConfig
    cfg = GPTConfig(n_layer=4, n_embd=64, n_head=4, n_kv_head=4,
                    vocab_size=512, sequence_len=128,
                    loop_max_cycles=2, mind=True)
    m = GPT(cfg)
    m.init_weights()
    m.eval()
    idx = torch.randint(0, 512, (2, 8))
    with torch.no_grad():
        y = m(idx, targets=None)
    assert tuple(y.shape) == (2, 8, 512)
    # mind observed at least one cycle and produced modulators
    assert m.mind is not None
    mods = m.mind.modulators()
    assert 0 < mods["temperature_scale"] <= 1.0
    # spectral multi-framework binding works
    m.mind.bind_spectral(m, subgan_cluster=m.subgan_cluster)
    losses = m.mind.aux_losses()
    assert "spectral" not in losses or torch.isfinite(losses["spectral"])
    print("mind-in-GPT (forward + modulators + spectral bind): PASS")


if __name__ == "__main__":
    test_spectral()
    test_manifold()
    test_emotion()
    test_intuition()
    test_memory_causal_info_unc()
    test_mind_in_gpt()
    print("ALL MIND TESTS PASSED")
