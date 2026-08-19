# SPDX-License-Identifier: Apache-2.0
"""Straight-through estimators and weight quantizers (P2/P3, decides D3).

Weight quantizers return INTEGER-VALUED float tensors (exact on the chip
grid) so that spike (0/1) x integer-weight matmuls produce integer currents
and the DiffLIF integer-exact dynamics stay bit-faithful. Scales are not
applied to the output: on CeliumNeUR a scale can only live in the learnable
per-neuron threshold, never in the weight word.
"""

import torch


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def ceil_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (x.ceil() - x).detach()


def quantize_int8(w: torch.Tensor) -> torch.Tensor:
    """Nearest int in [-128, 127]; STE through round, hard-clamp gradient
    at the rails (saturating, invariant I6 discipline)."""
    return torch.clamp(round_ste(w), -128, 127)


def quantize_ternary(w: torch.Tensor) -> torch.Tensor:
    """BitNet-b1.58 absmean ternarization to exact {-1, 0, +1} (unscaled:
    ternary is a strict subset of the chip's int8, D3)."""
    gamma = w.abs().mean().clamp(min=1e-8)
    return torch.clamp(round_ste(w / gamma), -1, 1)


def quantize_ternary_over(w: torch.Tensor, dim) -> torch.Tensor:
    """Ternarize with an independent absmean scale per slice (e.g. per
    block), so masked-out or unrelated regions cannot bias the scale."""
    gamma = w.abs().mean(dim=dim, keepdim=True).clamp(min=1e-8)
    return torch.clamp(round_ste(w / gamma), -1, 1)


QUANTIZERS = {
    "float": lambda w: w,
    "int8": quantize_int8,
    "ternary": quantize_ternary,
}
