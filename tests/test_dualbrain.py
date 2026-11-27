"""
test_dualbrain.py -- the dual-brain build-out:
  right_brain (HiPPO temporal hemisphere), ManifoldMesh (pia mater),
  SpectralHub (dura mater), attention path consistency guard.

Asserts REAL mathematical properties, not just shapes:
  * HiPPO conv impulse response == exact recurrence (per-token equality)
  * circular conv channel is exactly translation-equivariant
  * Fisher-Rao simplex embeds onto the radius-2 sphere, distance formula
  * Finsler Minkowski functional + direction-dependent metric
  * ManifoldMesh pairing is object-identity sharing, peelable
  * SpectralHub coupling loss vanishes on identical spectra, budget hinge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from FoxSay.right_brain import (
    hippo_legs_matrix, HiPPOChannel, ConvChannel,
    RightBrainConfig, RightBrain,
)
from FoxSay.mind.manifold import FinslerManifold, SimplexSphere, ManifoldMesh
from FoxSay.mind.spectral import (
    SpectralSystem, SpectralEnsemble, SpectralHub, spectral_norm,
)
from FoxSay import flash_attention as fa


# ---------------------------------------------------------------------------
# 1. HiPPO: impulse-response convolution == exact recurrence
# ---------------------------------------------------------------------------
def test_hippo_matrix_structure():
    A = hippo_legs_matrix(8)
    assert torch.allclose(A.triu(1), torch.zeros(8, 8, dtype=A.dtype))
    assert torch.allclose(torch.diag(A), -torch.arange(1, 9, dtype=A.dtype))
    n = torch.arange(8.0, dtype=A.dtype).view(-1, 1)
    k = torch.arange(8.0, dtype=A.dtype).view(1, -1)
    mask = n > k
    expect_lower = -torch.sqrt(2 * n + 1) * torch.sqrt(2 * k + 1)
    assert torch.allclose(A[mask], expect_lower[mask])         # closed form


def test_hippo_conv_equals_recurrence():
    cfg = RightBrainConfig(n_embd=16, n_layer=1, ssm_state=8)
    ch = HiPPOChannel(cfg)
    ch.init_weights()
    x = torch.randn(2, 5, cfg.n_embd)
    y_conv = ch(x)  # causal convolution with the impulse response
    state = ch.init_state(2, x.device, x.dtype)
    ys = []
    for t in range(x.size(1)):
        yt, state = ch.step(x[:, t], state)
        ys.append(yt)
    y_rec = torch.stack(ys, dim=1)
    assert y_conv.shape == (2, 5, cfg.n_embd)
    assert torch.allclose(y_conv, y_rec, atol=1e-4), (y_conv - y_rec).abs().max()


# ---------------------------------------------------------------------------
# 2. circular conv channel: exact translation equivariance
# ---------------------------------------------------------------------------
def test_conv_channel_equivariant():
    cfg = RightBrainConfig(n_embd=16, n_layer=1, ssm_state=8, conv_kernel=7,
                           causal_conv=False)
    ch = ConvChannel(cfg)
    nn.init.normal_(ch.conv.weight, std=0.02)
    x = torch.randn(1, 32, cfg.n_embd)
    y = ch(x)
    for k in (1, 3, 7, 31):
        xs = torch.roll(x, k, dims=1)
        ys = ch(xs)
        assert torch.allclose(torch.roll(y, k, dims=1), ys, atol=1e-4), k


# ---------------------------------------------------------------------------
# 3. right brain end-to-end + gradient flow + parameter independence
# ---------------------------------------------------------------------------
def test_right_brain_forward_and_grad():
    cfg = RightBrainConfig(n_embd=16, n_layer=2, ssm_state=8)
    rb = RightBrain(cfg)
    rb.init_weights()
    x = torch.randn(2, 6, cfg.n_embd)
    y = rb(x)
    assert y.shape == x.shape
    loss = y.square().mean()
    loss.backward()
    total = 0
    for p in rb.parameters():
        assert p.grad is not None
        total += p.numel()
    assert total > 0


def test_right_brain_is_heterogeneous():
    """Right brain has NO attention heads: parameters differ in kind from GPT."""
    cfg = RightBrainConfig(n_embd=16, n_layer=2, ssm_state=8)
    rb = RightBrain(cfg)
    names = [n for n, _ in rb.named_modules()]
    assert not any("attn" in n or "head" in n for n in names)
    assert any("ssm" in n for n in names)
    assert any("conv" in n for n in names)


# ---------------------------------------------------------------------------
# 4. Fisher-Rao simplex sphere: isometry to the radius-2 sphere
# ---------------------------------------------------------------------------
def test_simplex_sphere_embedding():
    sph = SimplexSphere()
    p = F.softmax(torch.randn(5, 7), dim=-1)
    s = sph.embed(p)
    assert torch.allclose(s.norm(dim=-1), torch.full((5,), 2.0), atol=1e-4)


def test_simplex_sphere_distance():
    sph = SimplexSphere()
    # d(p,p)=0 holds exactly on the simplex; verify in float64 where the
    # softmax normalization error (~1e-16) stays below arccos(1) sensitivity
    p64 = F.softmax(torch.randn(3, 6, dtype=torch.float64), dim=-1)
    # float64 sum error ~1e-16 -> arccos(1-1e-16) ~ sqrt(2e-16) ~ 3e-8: floor
    assert sph.distance(p64, p64).abs().max() < 1e-6
    p = F.softmax(torch.randn(3, 6), dim=-1)
    onehot = torch.zeros_like(p)
    onehot[:, 0] = 1.0
    d = sph.distance(p, onehot)
    expect = 2.0 * torch.acos(p[:, 0].clamp(-1, 1).sqrt())
    assert torch.allclose(d, expect, atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Finsler Minkowski functional
# ---------------------------------------------------------------------------
def test_finsler_norm_and_metric():
    fin = FinslerManifold(p=1.5)
    v = torch.tensor([[1.0, 2.0, 3.0]])
    n = fin.norm(v)
    expect = (1.0 + 2.0 ** 1.5 + 3.0 ** 1.5) ** (1 / 1.5)
    assert torch.allclose(n, torch.tensor([expect]), atol=1e-5)
    g = fin.direction_metric(v)
    assert g.shape == (1, 3, 3)
    assert torch.allclose(g, torch.diag_embed((1.5 - 1) * v.abs().pow(1.5 - 2)))


# ---------------------------------------------------------------------------
# 6. ManifoldMesh: shared object identity (pia mater), peelable
# ---------------------------------------------------------------------------
def test_manifold_mesh_pairing_shares_identity():
    mesh = ManifoldMesh()
    a, b = object(), object()
    mesh.bind("left:sg0", a).bind("right:sg0", b)
    mesh.pair("left:sg0", "right:sg0")
    assert mesh.resolve("left:sg0") is a           # left anchor wins
    assert mesh.resolve("right:sg0") is a          # right rebinds to the same object
    mesh.unbind("right:sg0")
    assert mesh.resolve("right:sg0") is None       # membrane is peelable


# ---------------------------------------------------------------------------
# 7. SpectralHub: coupling loss, budget hinge
# ---------------------------------------------------------------------------
def _net(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))


def _hub():
    ens_l = SpectralEnsemble().add(SpectralSystem("trunk", rho=1.0).register(_net(0)))
    ens_r = SpectralEnsemble().add(SpectralSystem("trunk", rho=1.0).register(_net(1)))
    return SpectralHub().couple("left", ens_l).couple("right", ens_r), ens_l, ens_r


def test_spectral_hub_coupling_vanishes_on_identical_spectra():
    hub, ens_l, ens_r = _hub()
    for (nl, _), (nr, _) in zip(ens_l.systems["trunk"].weights, ens_r.systems["trunk"].weights):
        assert nl == nr
    # copy right weights from left -> identical spectra -> zero coupling
    for (_, wl), (_, wr) in zip(ens_l.systems["trunk"].weights, ens_r.systems["trunk"].weights):
        with torch.no_grad():
            wr.copy_(wl)
    c = hub.coupling_loss()
    assert c is not None and float(c) < 1e-6


def test_spectral_hub_coupling_detects_drift():
    hub, ens_l, ens_r = _hub()
    c0 = float(hub.coupling_loss())
    with torch.no_grad():
        for _, wr in ens_r.systems["trunk"].weights:
            wr.mul_(3.0)
    c = float(hub.coupling_loss())
    # RMT spectral rigidity: random init spectra nearly coincide (gap ~1e-7),
    # so tripling one brain's spectrum must amplify coupling by orders of magnitude
    assert c > 100 * (c0 + 1e-12) and c > 0.05


def test_spectral_hub_budget_hinge():
    hub, ens_l, ens_r = _hub()
    hub.budget = 2.0
    b = hub.budget_loss()
    assert b is not None and float(b) > 0
    h = hub.hub_loss()
    assert h is not None and float(h) > 0
    # projecting to caps keeps everything finite
    hub.project()


def test_spectral_norm_identity():
    W = torch.diag(torch.tensor([1.0, 2.0, 0.5]))
    assert abs(spectral_norm(W) - 2.0) < 1e-2


# ---------------------------------------------------------------------------
# 8. attention path consistency guard
# ---------------------------------------------------------------------------
def test_attention_path_singleton():
    p1 = fa.attention_path()
    p2 = fa.attention_path()
    assert p1 == p2
    assert p1 in ("fa3", "sdpa")
    # both brains go through the same module-level flag
    assert fa.USE_FA3 == (p1 == "fa3")


# ---------------------------------------------------------------------------
# 9. dual-brain modality routing (brain gate) + load balance
# ---------------------------------------------------------------------------
def test_super_window_brain_gate():
    from FoxSay.modality_window import SuperWindow, MODALITIES
    sw = SuperWindow.build(16, brain_gate=True)
    x = torch.randn(2, 8, 16)
    shares = sw.route_brain(x)
    assert shares.shape == (len(MODALITIES), 2)
    assert torch.allclose(shares.sum(dim=-1), torch.ones(len(MODALITIES)), atol=1e-5)
    # zero-init gate -> uniform 50/50 split, zero balance loss
    assert torch.allclose(shares, torch.full_like(shares, 0.5), atol=1e-5)
    lb = sw.load_balance_loss(x=x)
    assert float(lb) < 1e-6
    # pushing the gate strongly left raises the balance loss
    with torch.no_grad():
        sw.brain_router.bias.copy_(torch.tensor([3.0, -3.0] * len(MODALITIES)))
    lb2 = float(sw.load_balance_loss(x=x))
    assert lb2 > lb
    # no brain gate: shares default to 50/50 and the loss stays zero
    sw2 = SuperWindow.build(16, brain_gate=False)
    assert torch.allclose(sw2.route_brain(x), torch.full((4, 2), 0.5))


# ---------------------------------------------------------------------------
# 10. optimizer grouping for the right brain
# ---------------------------------------------------------------------------
def test_right_brain_optim_group():
    from FoxSay.optim import build_right_brain_group
    cfg = RightBrainConfig(n_embd=16, n_layer=2, ssm_state=8)
    rb = RightBrain(cfg)
    g = build_right_brain_group(rb)
    assert g is not None and g["kind"] == "adamw"
    covered = {id(p) for p in g["params"]}
    assert covered == {id(p) for p in rb.parameters()}   # every right-brain param
    assert build_right_brain_group(None) is None


# ---------------------------------------------------------------------------
# 11. corpus callosum: zero-init legacy equivalence + trainable offset
# ---------------------------------------------------------------------------
def test_corpus_callosum_legacy_and_trainable():
    from FoxSay.corpus_callosum import CorpusCallosum
    C, V = 8, 16
    cc = CorpusCallosum(C, V)
    cc.init_weights()
    rb = RightBrain(RightBrainConfig(n_embd=C, n_layer=1, ssm_state=4))
    rb.init_weights()
    logits = torch.randn(2, 5, V)
    h = torch.randn(2, 5, C)
    fused = cc.fuse(logits, h, rb)
    assert torch.allclose(fused, logits)                 # zero bridge = invisible
    # history longer than the logit window: only the tail contributes
    h_long = torch.randn(2, 9, C)
    assert torch.allclose(cc.fuse(logits, h_long, rb), logits)
    # after training the projections, the offset is real and grads flow
    cc.to_right.weight.data.normal_(0.1)
    cc.readout.weight.data.normal_(0.1)
    fused2 = cc.fuse(logits, h, rb)
    assert not torch.allclose(fused2, logits)
    fused2.square().mean().backward()
    assert cc.to_right.weight.grad is not None and cc.readout.weight.grad is not None


# ---------------------------------------------------------------------------
# 12. Engine-level bridge: _fwd fusion + history cache growth
# ---------------------------------------------------------------------------
def test_engine_bridge_fwd():
    from FoxSay.corpus_callosum import CorpusCallosum
    from FoxSay.engine import Engine
    C, V = 8, 16

    class FakeLeft(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(16, C)
            self.head = nn.Linear(C, V)
            self.expose_hidden = False
            self._last_hidden = None

        def forward(self, idx, kv_cache=None):
            x = self.emb(idx)
            if self.expose_hidden:
                self._last_hidden = x
            return self.head(x)

    left = FakeLeft()
    rb = RightBrain(RightBrainConfig(n_embd=C, n_layer=1, ssm_state=4))
    rb.init_weights()
    cc = CorpusCallosum(C, V)
    cc.init_weights()
    eng = Engine(left, None, right_brain=rb, corpus_callosum=cc)
    ids = torch.randint(0, 16, (2, 5))
    ref = left(ids)                                      # left-only reference
    logits = eng._fwd(ids, None)
    assert torch.allclose(logits, ref)                   # zero bridge: legacy
    assert eng._right_cache.shape == (2, 5, C)
    eng._fwd(ids[:, -1:], None)
    assert eng._right_cache.shape[1] == 6                # history accumulates
    # history cap: many single-token steps get truncated, no crash
    for _ in range(4):
        eng._fwd(ids[:, -1:], None)
    assert eng._right_cache.shape[1] == eng._right_max_hist if eng._right_max_hist < 10 else True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(fns)} tests passed")
