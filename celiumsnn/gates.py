# SPDX-License-Identifier: Apache-2.0
"""Topology learning over block-sparse connectivity (P4).

Hard-Concrete L0 gates (Louizos, Welling & Kingma, ICLR 2018) over the
block grid of a BlockSparseSynapse. Two-phase discipline from the handoff:

  learn:  continuous stochastic gates scale each block's contribution;
          an L0 penalty prices every open block; gates can hit exactly 0.
  freeze: binarize (deterministic gate > 0) into a plain BlockSparseSynapse
          — a static, hashable routing artifact — then fine-tune weights
          and thresholds with the graph fixed.

Event semantics during learning follow the sampled gate: a block delivers
events iff its gate is nonzero this pass (a partially-open block is still
a set of valid dendrite entries; the gate scales currents, not validity).
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from celiumsnn.synapse import BlockSparseSynapse

BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1


class HardConcreteGate(nn.Module):
    """Per-element hard-concrete gates z in [0, 1] with P(z = 0) > 0."""

    def __init__(self, shape, init_log_alpha: float = 1.5) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full(shape, float(init_log_alpha)))

    def sample(self) -> torch.Tensor:
        u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / BETA)
        return torch.clamp(s * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)

    def deterministic(self) -> torch.Tensor:
        return torch.clamp(
            torch.sigmoid(self.log_alpha) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)

    def l0_penalty(self) -> torch.Tensor:
        """Sum over gates of P(z != 0) — the expected number of open gates."""
        return torch.sigmoid(
            self.log_alpha - BETA * torch.log(torch.tensor(-GAMMA / ZETA))).sum()

    def open_mask(self) -> torch.Tensor:
        return self.deterministic() > 0


class GatedBlockSparseSynapse(BlockSparseSynapse):
    """BlockSparseSynapse whose block mask is being LEARNED.

    The parent's static block_mask is all-True support; gates decide the
    topology. train(): stochastic gates (reparameterized), eval():
    deterministic gates. freeze() binarizes into the parent class.
    """

    def __init__(self, n_pre: int, n_post: int, block_size: int,
                 precision: str = "ternary", weight_sigma: float = 1.0,
                 seed: int | None = None, init_log_alpha: float = 1.5) -> None:
        p_blocks, q_blocks = n_pre // block_size, n_post // block_size
        super().__init__(n_pre, n_post, block_size,
                         torch.ones(p_blocks, q_blocks, dtype=torch.bool),
                         precision, weight_sigma, seed)
        self.gates = HardConcreteGate((p_blocks, q_blocks), init_log_alpha)

    def l0_penalty(self) -> torch.Tensor:
        return self.gates.l0_penalty()

    def forward(self, spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = spikes.shape[0]
        z = self.gates.sample() if self.training else self.gates.deterministic()
        s = spikes.reshape(batch, self.p_blocks, self.block_size)
        w_eff = self.quantized_weight() * z[..., None, None]
        currents = torch.einsum("bpi,pqij->bqj", s, w_eff).reshape(batch, self.n_post)
        block_any = (s.detach() > 0.5).any(-1).float()
        connected = (z.detach() > 0).float()
        post_events = (block_any @ connected) > 0
        events = post_events[:, :, None].expand(
            batch, self.q_blocks, self.block_size).reshape(batch, self.n_post)
        return currents, events

    def freeze(self) -> BlockSparseSynapse:
        """Binarize gates and return the static artifact. Weights are copied
        as-is; fine-tune afterwards so they re-accommodate to gate scale 1."""
        frozen = BlockSparseSynapse(self.n_pre, self.n_post, self.block_size,
                                    self.gates.open_mask(), self.precision)
        with torch.no_grad():
            frozen.weight.copy_(self.weight)
        return frozen


def topology_hash(*masks: torch.Tensor) -> str:
    """SHA-256 of the frozen routing masks — the signable artifact of the
    verifiability thesis (handoff §5)."""
    h = hashlib.sha256()
    for m in masks:
        h.update(str(list(m.shape)).encode())
        h.update(m.to(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
