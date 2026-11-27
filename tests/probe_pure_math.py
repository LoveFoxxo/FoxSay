"""Pure-math check: identical inputs, AdamW two steps, fp32 vs fp64 accumulation.
If this reproduces ~6e-4, the divergence is inherent to the dtype change (chaotic
amplification through the m/sqrt(v) normalization at large lr), not an FSDP bug."""

import torch

torch.manual_seed(0)
N = 128 * 64  # value_embeds-like element count
lr, beta1, beta2, eps, wd = 0.3464101615137755, 0.8, 0.995, 1e-10, 0.01

w0 = torch.randn(N).clamp(-0.3, 0.3)  # ~ve init scale
g0 = torch.randn(N) * 0.05
g1 = torch.randn(N) * 0.05  # sign flips vs g0 on many elements -> cancellation in m


def adamw2(acc_dtype):
    p = w0.to(acc_dtype).clone()
    m = torch.zeros(N, dtype=acc_dtype)
    v = torch.zeros(N, dtype=acc_dtype)
    for step, g in ((1, g0.to(acc_dtype)), (2, g1.to(acc_dtype))):
        p.mul_(1 - lr * wd)
        m.lerp_(g, 1 - beta1)
        v.lerp_(g.square(), 1 - beta2)
        bias1 = 1 - beta1**step
        bias2 = 1 - beta2**step
        denom = (v / bias2).sqrt() + eps
        p.add_(m / bias1 / denom, alpha=-lr)
    return p


p32 = adamw2(torch.float32)
p64 = adamw2(torch.float64)
d = (p32.double() - p64).abs()
print(f"max|fp32 - fp64| after 2 steps: {d.max().item():.3e}")
print(f"mean|fp32 - fp64|:             {d.mean().item():.3e}")
print(f"max|w| (fp64 path):           {p64.abs().max().item():.3f}")
