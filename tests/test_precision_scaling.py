import torch

from FoxSay.gpt import GPT, GPTConfig
from FoxSay.moe import SparseMoE
from FoxSay.optim import prepare_gradients
from FoxSay.right_brain import RightBrain, RightBrainConfig
from FoxSay.scaling import dual_parameter_report


def test_hippo_fp64_convolution_matches_recurrence():
    torch.manual_seed(1)
    cfg = RightBrainConfig(n_embd=8, n_layer=1, ssm_state=4)
    ch = RightBrain(cfg).blocks[0].ssm.double()
    ch.init_weights()
    with torch.no_grad():
        ch.input_proj.weight.normal_(0, 0.1)
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    full = ch(x)
    state = ch.init_state(2, x.device, x.dtype)
    steps = []
    for t in range(x.size(1)):
        y, state = ch.step(x[:, t], state)
        steps.append(y)
    assert torch.allclose(full, torch.stack(steps, 1), atol=1e-10, rtol=1e-10)


def test_right_brain_is_causal_and_streaming_exact():
    torch.manual_seed(2)
    cfg = RightBrainConfig(n_embd=8, n_layer=2, ssm_state=4, conv_kernel=3)
    rb = RightBrain(cfg)
    rb.init_weights()
    with torch.no_grad():
        for blk in rb.blocks:
            blk.ssm.input_proj.weight.normal_(0, 0.1)
            blk.gate.weight.normal_(0, 0.02)
    x = torch.randn(2, 5, 8)
    full = rb(x)
    changed = x.clone()
    changed[:, 4] = 1000
    prefix = rb(changed)
    assert torch.allclose(full[:, :4], prefix[:, :4], atol=1e-5, rtol=1e-5)
    states = rb.init_states(2, x.device, x.dtype)
    streamed = []
    for t in range(x.size(1)):
        y, states = rb.step(x[:, t], states)
        streamed.append(y)
    assert torch.allclose(full, torch.stack(streamed, 1), atol=2e-4, rtol=2e-4)


def test_moe_has_sparse_active_params_and_complete_grads():
    torch.manual_seed(3)
    moe = SparseMoE(8, n_experts=4, expert_ratio=2, top_k=2)
    moe.init_weights()
    report = moe.parameter_report()
    assert report["active"] < report["total"]
    x = torch.randn(3, 5, 8)
    target = torch.randn_like(x)
    loss = (moe(x) - target).square().mean() + moe.last_aux_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in moe.parameters())
    assert all(torch.isfinite(p.grad).all() for p in moe.parameters())


def test_gpt_moe_chunked_loss_backward_and_empty_targets():
    torch.manual_seed(4)
    cfg = GPTConfig(n_layer=2, n_embd=32, n_head=4, n_kv_head=4,
                    vocab_size=48, sequence_len=16, moe_num_experts=4,
                    moe_top_k=2, loss_chunk_size=3)
    model = GPT(cfg)
    model.init_weights()
    x = torch.randint(0, 48, (2, 6))
    y = torch.randint(0, 48, (2, 6))
    loss = model(x, y)
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    optimizer = model.setup_optimizer()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    empty = model(x, torch.full_like(y, -1))
    assert empty.item() == 0  # fully ignored batches stay differentiable zero
    empty.backward()
    base_loss = model(x, y, loss_reduction="none")
    assert base_loss.shape == y.reshape(-1).shape


def test_chunked_and_plain_loss_agree_without_moe():
    torch.manual_seed(5)
    common = dict(n_layer=2, n_embd=32, n_head=4, n_kv_head=4,
                  vocab_size=48, sequence_len=16)
    a, b = GPT(GPTConfig(**common)), GPT(GPTConfig(**common, loss_chunk_size=3))
    a.init_weights()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 48, (2, 6))
    y = torch.randint(0, 48, (2, 6))
    assert torch.allclose(a(x, y), b(x, y), atol=1e-5, rtol=1e-5)
    ignored = torch.full_like(y, -1)
    assert a(x, ignored).item() == 0 and b(x, ignored).item() == 0


def test_gradient_guard_skips_nonfinite_and_clips():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p.grad = torch.tensor([float("nan"), 1.0])
    assert not prepare_gradients([p])
    p.grad = torch.tensor([3.0, 4.0])
    assert prepare_gradients([p], max_norm=1.0)
    assert torch.allclose(p.grad, torch.tensor([0.6, 0.8]), atol=1e-6)


def test_right_moe_bf16_and_fsdp_step():
    from FoxSay.fsdp import FullySharded
    torch.manual_seed(6)
    cfg = RightBrainConfig(n_embd=8, n_layer=2, ssm_state=4,
                           moe_num_experts=2, moe_top_k=1)
    rb = RightBrain(cfg)
    rb.init_weights()
    y = rb(torch.randn(2, 4, 8, dtype=torch.bfloat16))
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    optimizer = rb.setup_optimizer()
    fsdp = FullySharded(rb, optimizer, master_dtype=torch.float64)
    fsdp.shard()
    out = rb(torch.randn(2, 4, 8))
    loss = out.square().mean() + rb.auxiliary_loss()
    loss.backward()
    fsdp.optimizer.step()
    state = fsdp.consolidated_state_dict()
    assert set(state) == set(rb.state_dict())
    assert all(torch.isfinite(v).all() for v in state.values())


def test_trillion_plan_is_meta_countable_and_sparse():
    report = dual_parameter_report()
    assert 0.9e12 <= report["total"] <= 1.1e12, report
    assert report["active_upper_bound"] < 0.25 * report["total"], report
    print(report)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
