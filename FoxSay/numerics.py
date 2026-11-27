"""Numerical primitives shared by both hemispheres."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """RMS-normalize in fp32/fp64, then restore the activation dtype.

    Low-precision accumulators are particularly harmful for long, deep residual
    streams. The output cast keeps existing attention/MLP kernels compatible.
    """
    work = x if x.dtype in (torch.float32, torch.float64) else x.float()
    if eps is None:
        eps = 1e-12 if work.dtype == torch.float64 else 1e-6
    return F.rms_norm(work, (work.size(-1),), eps=eps).to(x.dtype)
