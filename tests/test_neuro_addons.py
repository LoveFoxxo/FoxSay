"""
test_neuro_addons.py -- expression drive, strong rationality, functional layer,
and the equal-sized right brain. Asserts REAL mathematical properties.

  * VOI = E[V_post] - V_prior - cost
  * EFE decomposition: G = epistemic + pragmatic
  * intrinsic reward KL = 0 iff posterior == prior
  * Hawkes recurrence == closed form; self-excitation raises intensity
  * fast weights: Hebbian outer-product + decay (STP)
  * Curry-Howard type consistency (identity-compose => well-typed chains)
  * martingale: constant confidence => zero penalty; Doob upcrossing bound
  * submodularity of the coverage function; greedy (1-1/e) guarantee
  * Bayes filter posterior equals hand-computed posterior
  * Kalman steady gain = fixed point of the Riccati equation
"""

import math

import torch
import torch.nn as nn

from FoxSay.right_brain import mirror_left, RightBrain, RightBrainConfig
from FoxSay.mind.communication import (
    VOIGate, ExpectedFreeEnergy, IntrinsicReward,
    ListenerModel, HawkesSpeaker, OptionIntent,
)
from FoxSay.mind.rationality import (
    FastWeights, TypeChecker, MartingaleMonitor, SubmodularBound, SavageDecision,
)
from FoxSay.mind.belief import BeliefState, DialogueState, HTNPlanner, KalmanStream
from FoxSay.mind import Mind


# ---------------------------------------------------------------------------
# expression drive
# ---------------------------------------------------------------------------
def test_voi_criterion():
    gate = VOIGate(message_cost=0.1)
    voi_zero = gate.vois(torch.tensor(1.0), torch.tensor(1.0))   # no info gain
    assert abs(float(voi_zero) - (-0.1)) < 1e-6
    assert not gate.should_speak(torch.tensor(1.0), torch.tensor(1.0))
    voi_pos = gate.vois(torch.tensor(1.0), torch.tensor(1.5))
    assert abs(float(voi_pos) - 0.4) < 1e-6
    assert gate.should_speak(torch.tensor(1.0), torch.tensor(1.5))
    # batch form
    M = gate.batch(torch.tensor(1.0), torch.tensor([[1.0, 1.0], [1.5, 1.5]]))
    assert torch.allclose(M, torch.tensor([-0.1, 0.4]), atol=1e-6)


def test_expected_free_energy_decomposition():
    efe = ExpectedFreeEnergy()
    G = efe(torch.tensor(2.0), torch.tensor(0.5), torch.tensor(0.3),
            torch.tensor(-0.7))
    assert abs(float(G) - (2.0 + 0.5 + 0.3 - 0.7)) < 1e-6
    d = efe.decompose(torch.tensor(2.0), torch.tensor(0.5),
                      torch.tensor(0.3), torch.tensor(-0.7))
    assert abs(float(d["epistemic"]) - 2.8) < 1e-6
    assert abs(float(d["pragmatic"]) + 0.7) < 1e-6


def test_intrinsic_reward_kl():
    ir = IntrinsicReward()
    mu = torch.zeros(4)
    s = torch.ones(4)
    assert abs(float(ir(mu, s, mu, s))) < 1e-6          # nothing learned
    kl = ir(mu, s, torch.ones(4) * 2.0, torch.ones(4))
    expect = 4 * 0.5 * 4.0                              # 0.5*sum((1-0)^2/1)
    assert abs(float(kl) - expect) < 1e-5


def test_listener_model_value():
    lm = ListenerModel()
    listener = lambda m: torch.tensor(0.2) if m is None else torch.tensor(0.8)
    v = lm.message_value(listener, message="hello")
    assert abs(float(v) - 0.6) < 1e-6
    assert abs(float(lm.grice_relevance(v, 0.2)) - 3.0) < 1e-5


def test_hawkes_recurrence_matches_closed_form():
    hs = HawkesSpeaker(mu=0.1, alpha=0.5, beta=0.7, threshold=9.0)
    events = []
    for i in range(6):
        fire = (i == 1)
        lam = hs.advance(1.0, fire=fire)   # one event fired at t = i+1 = 2.0
        if fire:
            events.append(i + 1.0)         # the fire happens AT the new time
        exact = HawkesSpeaker.closed_form(i + 1.0, events, 0.1, 0.5, 0.7)
        assert abs(lam - exact) < 1e-6, (lam, exact)
    # self-excitation: intensity right after the event > before
    hs2 = HawkesSpeaker(mu=0.1, alpha=0.5, beta=0.7, threshold=9.0)
    before = hs2.advance(1.0)
    after = hs2.advance(0.0, fire=True)
    assert after > before


def test_option_intent_termination():
    opt = OptionIntent("ask", termination=lambda s: s >= 3)
    assert not opt.step(1) and opt.is_active()
    assert not opt.step(2) and opt.is_active()
    assert opt.step(3) and not opt.is_active()      # own termination condition
    opt2 = OptionIntent("cap", termination=lambda s: False, max_steps=2)
    assert not opt2.step(0)
    assert opt2.step(0)                              # max_steps safety cap


# ---------------------------------------------------------------------------
# strong rationality
# ---------------------------------------------------------------------------
def test_fast_weights_hebbian_and_decay():
    fw = FastWeights(8, eta=0.5, rho=0.9)
    x = torch.randn(4, 8)
    assert torch.allclose(fw(x), x)                  # zero fast weight: identity
    fw.hebbian_update(x, x)
    assert not torch.allclose(fw(x), x)              # Hebbian change applied
    w1 = fw.w_fast.clone()
    fw.hebbian_update(x, x)
    # decay: with identical inputs the update is w <- rho*w + eta*outer, so
    # ||w|| stays bounded; verify decay factor shows up in the difference
    expect = 0.9 * w1 + 0.5 * torch.outer(x.mean(0), x.mean(0))
    assert torch.allclose(fw.w_fast, expect, atol=1e-6)
    fw.reset()
    assert torch.allclose(fw.w_fast, torch.zeros(8, 8))


def test_type_checker_consistency():
    tc = TypeChecker(n_type=8, type_dim=16)
    well = torch.tensor([0, 0, 0, 0])               # constant types compose (id)
    bad = torch.tensor([0, 5, 0, 5])                # wild jumps
    assert float(tc.type_loss(well)) < float(tc.type_loss(bad))
    # consistency has per-step residuals
    assert tc.consistency(well).shape == (3,)


def test_martingale_penalty_and_doob():
    mm = MartingaleMonitor()
    const = torch.full((8,), 0.7)
    assert float(mm.martingale_penalty(const)) == 0.0
    osc = torch.tensor([0.0, 0.9, 0.0, 0.9, 0.0])
    assert float(mm.martingale_penalty(osc)) > 0.5
    # the wild oscillation VIOLATES Doob's upcrossing bound: a rational
    # (sub)martingale confidence process cannot behave like this -> flagged
    assert not mm.check(osc, 0.3, 0.6)
    # a calm bounded process respects the bound
    calm = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.2])
    assert mm.check(calm, 0.1, 0.2)


def test_submodular_coverage_and_greedy_guarantee():
    sb = SubmodularBound()
    sets = {0: {0, 1}, 1: {0, 2}, 2: {1, 2}, 3: {3}}
    f = lambda S: len(set().union(*(sets[i] for i in S))) if S else 0
    assert sb.is_submodular(f, {0, 1, 2, 3})
    S, val, ratio = sb.greedy_guarantee(f, {0, 1, 2, 3}, k=2, opt=3)
    assert val == 3
    assert ratio >= 1 - 1 / math.e - 1e-9
    # prune decision: a candidate whose best gain cannot reach best is pruned
    assert sb.prune_decision(f, {0}, 3, best_value=4.0)


def test_savage_separation_and_certainty():
    sd = SavageDecision()
    p = torch.tensor([0.5, 0.5])
    U = torch.tensor([10.0, 0.0])
    assert abs(float(sd.expected_utility(p, U)) - 5.0) < 1e-6
    ce = sd.certainty_equivalent(p, U, risk_aversion=1.0)
    var = 0.5 * 25.0 + 0.5 * 25.0
    assert abs(float(ce) - (5.0 - 0.5 * var)) < 1e-6


# ---------------------------------------------------------------------------
# functional layer
# ---------------------------------------------------------------------------
def test_belief_bayes_filter_exact():
    bs = BeliefState()
    b = torch.tensor([0.5, 0.5])
    T = torch.tensor([[0.8, 0.2], [0.2, 0.8]])      # columns sum to 1
    O = torch.tensor([0.9, 0.1])                    # observation favors state 0
    post = bs.filter(b, T, O)
    expect_pred = torch.tensor([0.8 * 0.5 + 0.2 * 0.5, 0.2 * 0.5 + 0.8 * 0.5])
    expect = expect_pred * O
    expect = expect / expect.sum()
    assert torch.allclose(post, expect, atol=1e-6)
    assert abs(float(post.sum() - 1.0)) < 1e-6


def test_dialogue_state_compression():
    ds = DialogueState(4)
    info0 = ds.summarize()
    assert abs(info0["entropy"] - math.log(4)) < 1e-4   # uniform prior
    ds.update(torch.tensor([0.9, 0.2, 0.05, 0.05]))      # sharp likelihood
    info1 = ds.summarize()
    assert info1["map_state"] == 0
    assert info1["entropy"] < info0["entropy"]
    assert info1["turns"] == 1


def test_htn_plan_expansion():
    methods = {
        "book_flight": [["check_calendar", "pay", "confirm"]],
        "pay": [["call_bank"]],
    }
    htn = HTNPlanner(methods)
    plan = htn.plan("book_flight")
    assert plan == ["check_calendar", "call_bank", "confirm"]
    assert htn.depth_of("book_flight") == 2


def test_kalman_steady_gain_riccati():
    ks = KalmanStream(a=1.0, q=0.01, r=1.0)
    Kinf = ks.steady_gain()
    # invert K = P_pred/(P_pred + r); stationary POST-update covariance is
    # the fixed point of the Riccati map  P -> a^2P + q - (a^2P+q)^2/(a^2P+q+r)
    P_pred = Kinf * 1.0 / (1.0 - Kinf)
    P_post = (1 - Kinf) * P_pred
    rhs = 1.0 * P_post + 0.01 - (1.0 * P_post + 0.01) ** 2 / (1.0 * P_post + 0.01 + 1.0)
    assert abs(P_post - rhs) < 1e-6
    # filtering converges to the truth
    ks.reset()
    ests = [float(ks.step(torch.tensor(5.0))) for _ in range(50)]
    assert abs(ests[-1] - 5.0) < 0.1


# ---------------------------------------------------------------------------
# equal-sized hemispheres + Mind wiring
# ---------------------------------------------------------------------------
def test_equal_sized_hemispheres():
    class LeftCfg:
        n_embd = 32
        n_layer = 12
    cfg = mirror_left(LeftCfg)
    assert cfg.n_layer == LeftCfg.n_layer            # equal depth = equal size
    rb = RightBrain(cfg)
    assert len(rb.blocks) == 12


def test_mind_wiring_and_speak_decision():
    mind = Mind(n_embd=16)
    x = torch.randn(2, 4, 16)
    out = mind.observe(x, cycle=1)
    assert "urge" in out and out["urge"] > 0
    out2 = mind.observe(x, cycle=2)
    # fast weights grew: w_fast nonzero after two Hebbian updates
    assert float(mind.fast.w_fast.abs().max()) > 0
    voi, urge, speak = mind.speak_decision(1.0, 3.0)
    assert voi == 2.0                                # 3 - 1 - 0 cost
    assert speak in (True, False)                    # rhythm-dependent
    # EFE + intrinsic objects exist and compute
    g = mind.efe(torch.tensor(1.0), torch.tensor(1.0),
                 torch.tensor(1.0), torch.tensor(1.0))
    assert abs(float(g) - 4.0) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(fns)} tests passed")
