"""
Tests for the internal verification loop + adversarial GAN gate package.

Run:  PYTHONPATH=. python3 tests/test_gan.py [case_numbers...]
e.g.  PYTHONPATH=. python3 tests/test_gan.py 1 2 3
      (no args = run all; cases are independent, small numbers keep runtime low)
"""

import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from FoxSay.gan import (
    LoopGate, neutral_state, VerifierModel, VerifierTrainer, OutputGate,
    GPTConfigLite, RewardConfig, shape_reward, grpo_advantages,
    token_level_advantages, compute_efficiency_reward, collate_turns,
)
from FoxSay.gpt import GPT, GPTConfig
from FoxSay.engine import Engine, KVCache
from FoxSay.checkpoint_manager import _patch_missing_keys


def tiny_cfg(loop=1, seq=128):
    return GPTConfig(sequence_len=seq, vocab_size=512, n_layer=4, n_head=4,
                     n_kv_head=4, n_embd=64, window_pattern="L", loop_max_cycles=loop)


class FakeTok:
    def encode_special(self, s):
        return {"<|python_start|>": 501, "<|python_end|>": 502, "<|output_start|>": 503,
                "<|output_end|>": 504, "<|assistant_end|>": 505}[s]
    def get_bos_token_id(self): return 1
    def decode(self, ids): return "".join(map(str, ids))
    def encode(self, s): return [2, 3]


def make_models(loop):
    torch.manual_seed(42)
    m_loop = GPT(tiny_cfg(loop)); m_loop.init_weights()
    m_leg = GPT(tiny_cfg(1)); m_leg.init_weights()
    missing, unexpected = m_loop.load_state_dict(m_leg.state_dict(), strict=False)
    assert all(k.startswith("loop_gate.") for k in missing) and len(unexpected) == 0
    m_loop.eval(); m_leg.eval()
    return m_loop, m_leg


# ---------------------------------------------------------------- gate tests
def test_1_gate_neutral_is_identity():
    gate = LoopGate(64, 4); gate.init_weights()
    xc, xr = torch.randn(2, 10, 64), torch.randn(2, 10, 64)
    assert torch.equal(gate.blend(xc, xr), xc), "blend must be identity at zero init"
    p = gate.stop_probs(xc, 0)
    assert p.shape == (2,) and (p > 0.9).all(), f"cycle-1 stop prob ~0.95, got {p}"
    ws = LoopGate.act_weights([torch.tensor([0.9]), torch.tensor([0.9]), torch.tensor([0.9])])
    assert abs(sum(w.item() for w in ws) - 1.0) < 1e-6, "ACT weights sum to 1"
    print("[ok] 1 LoopGate neutral init: identity blend, ~0.95 first-cycle halt, ACT weights sum to 1")


def test_2_loop_bit_identical_untrained():
    m_loop, m_leg = make_models(4)
    idx = torch.randint(1, 500, (2, 32))
    with torch.no_grad():
        assert torch.equal(m_loop(idx), m_leg(idx)), "untrained loop must be bit-identical"
        assert (m_loop._last_loop_cycles == 1).all(), "neutral gate halts at cycle 1"
    # forced deeper cycles stay identical while params are neutral
    m_loop._loop_depth = 4
    with torch.no_grad():
        assert torch.equal(m_loop(idx), m_leg(idx))
    print("[ok] 2 loop-enabled-but-untrained == legacy model bit-for-bit (incl. forced 4 cycles)")


def test_3_kv_cache_loop_paths():
    m_loop, m_leg = make_models(3)
    prompt = [1] + torch.randint(2, 500, (12,)).tolist()
    kv1 = KVCache(1, 4, 128, 16, 4, torch.device("cpu"), torch.float32)
    kv2 = KVCache(1, 4, 128, 16, 4, torch.device("cpu"), torch.float32)
    ids = torch.tensor([prompt])
    with torch.no_grad():
        l1 = m_loop(ids, kv_cache=kv1); l2 = m_leg(ids, kv_cache=kv2)
        assert torch.equal(l1, l2) and kv1.get_pos() == kv2.get_pos() == len(prompt)
        tok = int(torch.argmax(l1[:, -1, :]))
        t1 = torch.tensor([[tok]])
        l1 = m_loop(t1, kv_cache=kv1); l2 = m_leg(t1, kv_cache=kv2)
        assert torch.equal(l1, l2) and kv1.get_pos() == kv2.get_pos()
        # activate the gate: cache still advances exactly +1 per token
        m_loop.loop_gate.mix_lambda.fill_(0.3); m_loop.loop_gate.readout.bias.fill_(-10.0)
        pos = kv1.get_pos()
        la = m_loop(t1, kv_cache=kv1)
        assert kv1.get_pos() == pos + 1 and torch.isfinite(la).all() and not torch.equal(la, l1)
    print("[ok] 3 KV-cache: neutral loop bit-identical (prefill+decode); active gate advances cache exactly +1/token")


def test_4_ignition_gradient_and_optimizer():
    torch.manual_seed(42)
    m = GPT(tiny_cfg(4)); m.init_weights()
    idx = torch.randint(1, 500, (2, 32))
    opt = m.setup_optimizer()
    m.train()
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in m.parameters()), "optimizer must cover every param exactly once"
    loss = m(idx, targets=idx); loss.backward()
    assert m.loop_gate.mix_lambda.grad.abs().sum() > 0, "ignition gradient must reach mix_lambda"
    opt.step(); m.zero_grad()
    loss = m(idx, targets=idx); loss.backward()
    assert m.loop_gate.adapter.weight.grad.abs().sum() > 0
    assert m.transformer.h[0].attn.c_q.weight.grad.abs().sum() > 0
    print("[ok] 4 ignition gradient reaches mix_lambda at step 1; full flow (gate+trunk) at step 2; optimizer coverage exact")


# --------------------------------------------------------- verifier/gate tests
def test_5_verifier_adversarial_separation():
    v = VerifierModel(GPTConfigLite(n_embd=32, n_layer=2, n_head=4, vocab_size=40000)); v.init_weights()
    pos = torch.randint(0, 100, (16, 20)) + 20000
    neg = torch.randint(0, 100, (16, 20)) + 30000
    tr = VerifierTrainer(v, lr=3e-3)
    p0, n0 = tr.evaluate(pos), tr.evaluate(neg)
    for _ in range(60):
        tr.train_batch(pos, neg)
    p1, n1 = tr.evaluate(pos), tr.evaluate(neg)
    assert p1 > p0 and p1 > n1 + 0.3, f"discriminator must separate: {p0:.3f}->{p1:.3f} vs {n0:.3f}->{n1:.3f}"
    og = OutputGate(v, gate_threshold=0.5, max_rejects=3)
    acc, p = og.decide(pos)
    assert acc.shape == (16,)
    print(f"[ok] 5 adversarial separation converges (pos {p0:.2f}->{p1:.2f}, neg {n0:.2f}->{n1:.2f}); OutputGate decides")


def test_6_engine_output_gate():
    m_loop, m_leg = make_models(4)
    orig_forward = m_loop.forward
    def biased_forward(idx, targets=None, kv_cache=None, loss_reduction='mean'):
        logits = orig_forward(idx, targets=targets, kv_cache=kv_cache, loss_reduction=loss_reduction)
        if targets is None:
            logits = logits.clone(); logits[..., 505] += 10.0  # always sample <|assistant_end|>
        return logits
    m_loop.forward = biased_forward
    tok = FakeTok()
    base = Engine(m_loop, tok)
    r0, _ = base.generate_batch([1, 10, 11], num_samples=2, max_tokens=20, temperature=0.0, seed=5)
    assert all(len(r) == 3 for r in r0)
    # untrained gate accepts everything (p=0.5 >= 0.5): identical to no gate
    v = VerifierModel(GPTConfigLite(n_embd=32, n_layer=2, n_head=4, vocab_size=512)); v.init_weights()
    og = OutputGate(v, gate_threshold=0.5, max_rejects=3)
    e2 = Engine(m_loop, tok, output_gate=og)
    r2, _ = e2.generate_batch([1, 10, 11], num_samples=2, max_tokens=20, temperature=0.0, seed=5)
    assert e2.gate_stats["rejected"] == 0 and r2 == r0
    # reject-all: 3 rejects/row then release; loop budget bumps to max_cycles
    with torch.no_grad():
        v.turn_head.bias.fill_(-50.0)
    e3 = Engine(m_loop, tok, output_gate=og)
    r3, _ = e3.generate_batch([1, 10, 11], num_samples=2, max_tokens=30, temperature=0.0, seed=5)
    assert e3.gate_stats["rejected"] == 6 and all(len(x) == 6 for x in r3)
    assert m_loop._loop_depth == 4, "budget capped at max_cycles"
    # accept-all: identical to no gate
    with torch.no_grad():
        v.turn_head.bias.fill_(50.0)
    m_loop._loop_depth = None
    e4 = Engine(m_loop, tok, output_gate=og)
    r4, _ = e4.generate_batch([1, 10, 11], num_samples=2, max_tokens=20, temperature=0.0, seed=5)
    assert e4.gate_stats["rejected"] == 0 and r4 == r0
    print("[ok] 6 engine gate: untrained==no-gate; reject-all masks end, releases after cap, budget bounded; accept-all==no-gate")


# ------------------------------------------------------------- reward tests
def test_7_reward_shaping_ordering():
    cfg = RewardConfig(max_cycles_budget=4, target_cycles_fast=1)
    r_fast  = shape_reward(torch.tensor([1.]), torch.tensor([0.4]), torch.tensor([0.9]), torch.tensor([1]), torch.tensor([False]), cfg)
    r_slow  = shape_reward(torch.tensor([1.]), torch.tensor([0.4]), torch.tensor([0.9]), torch.tensor([4]), torch.tensor([False]), cfg)
    r_wrong = shape_reward(torch.tensor([0.]), torch.tensor([0.4]), torch.tensor([0.9]), torch.tensor([1]), torch.tensor([False]), cfg)
    r_trunc = shape_reward(torch.tensor([0.]), torch.tensor([0.4]), torch.tensor([0.9]), torch.tensor([1]), torch.tensor([True]), cfg)
    assert r_fast > r_slow > r_wrong > r_trunc, (r_fast, r_slow, r_wrong, r_trunc)
    # truncation is a small penalty, never a zeroing
    r_trunc_ok = shape_reward(torch.tensor([1.]), torch.tensor([0.4]), torch.tensor([0.9]), torch.tensor([1]), torch.tensor([True]), cfg)
    assert r_trunc_ok > 0.5, f"correct-but-truncated keeps most of its reward: {r_trunc_ok}"
    adv = grpo_advantages(torch.tensor([1.0, 0.0, 0.2, 0.2]), torch.tensor([0, 0, 1, 1]))
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 0.0, 0.0]))
    ta = token_level_advantages(torch.tensor([0.5, -0.5]), torch.tensor([[1., 1., 0.], [1., 0., 0.]]))
    assert torch.allclose(ta.sum(dim=1), torch.tensor([0.5, -0.5]))
    print("[ok] 7 reward ordering correct+fast > correct+slow > wrong > wrong+trunc; truncation never zeroes; GRPO/token-adv math exact")


# --------------------------------------------------- checkpoint grafting test
def test_8_checkpoint_patch_neutral():
    # old checkpoint state dict (no loop keys) + loop-enabled config
    m_leg = GPT(tiny_cfg(1)); m_leg.init_weights()
    old_data = {k: v.clone() for k, v in m_leg.state_dict().items()}
    cfg_loop = tiny_cfg(4)
    _patch_missing_keys(old_data, cfg_loop)
    for k, v in neutral_state(cfg_loop.n_embd, 4).items():
        assert torch.equal(old_data[f"loop_gate.{k}"], v), f"patched key {k} must be neutral"
    # loading the patched dict into a loop-enabled model is strict and no-op
    m_loop = GPT(cfg_loop)
    m_loop.load_state_dict(old_data, strict=True)
    idx = torch.randint(1, 500, (2, 32))
    with torch.no_grad():
        assert torch.equal(m_loop(idx), m_leg(idx)), "grafted checkpoint must behave bit-identically"
    print("[ok] 8 _patch_missing_keys grafts neutral loop params; strict load + bit-identical forward")


# ----------------------------------------------------- online RL-shaped reward
def test_9_rl_pipeline_stub():
    v = VerifierModel(GPTConfigLite(n_embd=32, n_layer=2, n_head=4, vocab_size=512)); v.init_weights()
    cfg = RewardConfig(w_process=0.25, w_adversarial=0.5, w_efficiency=0.15, w_truncation=-0.25,
                       max_cycles_budget=4, truncation_limit=32)
    answers = ["#### 42"] * 4 + ["wrong"] * 4
    turns = [[1, 10] + [ord(c) % 200 + 2 for c in a] for a in answers]
    r_correct = torch.tensor([1.0] * 4 + [0.0] * 4)
    turn_t = collate_turns(turns, pad_id=1)
    r_process = v.process_reward(turn_t)
    _, tl = v(turn_t)
    p_pass = torch.sigmoid(tl)
    cycles = torch.tensor([1, 2, 4, 1, 1, 4, 2, 1])
    truncated = torch.tensor([False] * 7 + [True])
    rewards = shape_reward(r_correct, r_process, p_pass, cycles, truncated, cfg)
    adv = grpo_advantages(rewards)
    assert (rewards[:4] > rewards[4:]).all() and adv.mean().abs() < 1e-6
    tr = VerifierTrainer(v, lr=1e-3)
    pos_t, neg_t = collate_turns(turns[:4], pad_id=1), collate_turns(turns[4:], pad_id=1)
    for _ in range(30):
        tr.train_batch(pos_t, neg_t)
    assert tr.evaluate(pos_t) - tr.evaluate(neg_t) > 0.4
    print("[ok] 9 full RL stub: shaped rewards rank correctly, zero-mean advantages, online verifier separation")


TESTS = [test_1_gate_neutral_is_identity, test_2_loop_bit_identical_untrained,
         test_3_kv_cache_loop_paths, test_4_ignition_gradient_and_optimizer,
         test_5_verifier_adversarial_separation, test_6_engine_output_gate,
         test_7_reward_shaping_ordering, test_8_checkpoint_patch_neutral,
         test_9_rl_pipeline_stub]


def main():
    wanted = [int(a) for a in sys.argv[1:]] or list(range(1, len(TESTS) + 1))
    for i, fn in enumerate(TESTS, start=1):
        if i in wanted:
            fn()
    print("All GAN/loop tests passed.")


if __name__ == "__main__":
    main()
