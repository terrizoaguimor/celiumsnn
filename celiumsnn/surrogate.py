# SPDX-License-Identifier: AGPL-3.0-or-later
"""Surrogate spike functions for the threshold comparator (P2, decides D5).

Forward is the exact golden fire condition (v >= theta, i.e. Heaviside with
H(0) = 1). Backward substitutes a kernel g((v - theta) / width) / width:

  atan         g(u) = 1 / (1 + (pi/2 * u)^2)          (Fang et al.)
  fast_sigmoid g(u) = 1 / (1 + |u|)^2                 (Zenke & Ganguli)
  triangular   g(u) = max(0, 1 - |u|)                 (Esser / Bellec)

All kernels peak at 1 at u = 0, so gradients are comparable across shapes
and `width` has the same meaning (membrane units from threshold) everywhere.
"""

import torch

SHAPES = ("atan", "fast_sigmoid", "triangular")


def surrogate_kernel(u: torch.Tensor, shape: str) -> torch.Tensor:
    if shape == "atan":
        return 1.0 / (1.0 + (torch.pi / 2 * u) ** 2)
    if shape == "fast_sigmoid":
        return 1.0 / (1.0 + u.abs()) ** 2
    if shape == "triangular":
        return torch.clamp(1.0 - u.abs(), min=0.0)
    raise ValueError(f"unknown surrogate shape {shape!r}; pick from {SHAPES}")


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width, shape):
        ctx.save_for_backward(x)
        ctx.width, ctx.shape = width, shape
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        g = surrogate_kernel(x / ctx.width, ctx.shape) / ctx.width
        return grad_out * g, None, None


def spike(v_minus_theta: torch.Tensor, width: float, shape: str) -> torch.Tensor:
    """Heaviside(v - theta) forward, surrogate gradient backward."""
    if width <= 0:
        raise ValueError("surrogate width must be positive")
    if shape not in SHAPES:
        raise ValueError(f"unknown surrogate shape {shape!r}; pick from {SHAPES}")
    return _SpikeFn.apply(v_minus_theta, float(width), shape)
