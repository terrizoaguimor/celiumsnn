# SPDX-License-Identifier: Apache-2.0
"""Sparse connectivity primitives (P3).

Two modules with ONE semantic contract — the chip's dendrite table:

  EdgeListSynapse   Exact edge list: (pre, post, weight) entries, duplicate
                    entries with real multiplicity (SPEC §6.2), scatter_add
                    delivery. Any topology, chip-faithful at any grain.

  BlockSparseSynapse The GPU-viable form (handoff §6, R4): neurons grouped
                    in blocks, a static boolean block mask, DENSE weights
                    inside every active block. An active block means all
                    Bp x Bq (pre, post) pairs are valid dendrite entries —
                    including zero-weight ones, which still deliver events
                    (P0-SEMANTICS.md §4: E is event presence, not I != 0).
                    That definition makes the event mask exact and cheap:
                    a post-block has events iff any connected pre-block has
                    any spike. Expanding an active block to Bp x Bq edge
                    list entries reproduces I and E exactly (tested).

Both return `(currents, events)` ready for IntLIF/DiffLIF.step(I, E).

Chip-mapping note: one active 64x64 block = 4096 dendrite entries, which
already exceeds the v1 silicon budget (1024). Block size is a GPU-economics
parameter (>= 64 to clear the sparsity floor), not a chip parameter; the
chip-faithful reference instance uses EdgeListSynapse (or blocks <= 16).
"""

from __future__ import annotations

import torch
from torch import nn

from celiumsnn.quant import quantize_int8, quantize_ternary_over


def _quantize(w: torch.Tensor, precision: str, block_dims=None) -> torch.Tensor:
    if precision == "float":
        return w
    if precision == "int8":
        return quantize_int8(w)
    if precision == "ternary":
        dim = block_dims if block_dims is not None else tuple(range(w.dim()))
        return quantize_ternary_over(w, dim)
    raise ValueError(f"unknown precision {precision!r}")


class EdgeListSynapse(nn.Module):
    """Chip-exact dendrite table as differentiable edge list."""

    def __init__(self, n_pre: int, n_post: int, edges_pre, edges_post,
                 weights_init, precision: str = "ternary") -> None:
        super().__init__()
        edges_pre = torch.as_tensor(edges_pre, dtype=torch.long)
        edges_post = torch.as_tensor(edges_post, dtype=torch.long)
        if edges_pre.shape != edges_post.shape or edges_pre.dim() != 1:
            raise ValueError("edges_pre/edges_post must be equal-length 1-D")
        if edges_pre.numel() and not (0 <= edges_pre.min() and edges_pre.max() < n_pre
                                      and 0 <= edges_post.min() and edges_post.max() < n_post):
            raise ValueError("edge endpoints out of range")
        self.n_pre, self.n_post = n_pre, n_post
        self.precision = precision
        self.register_buffer("edges_pre", edges_pre)
        self.register_buffer("edges_post", edges_post)
        self.weight = nn.Parameter(
            torch.as_tensor(weights_init, dtype=torch.float32).reshape(-1))
        if self.weight.numel() != edges_pre.numel():
            raise ValueError("one weight per edge required")

    def quantized_weight(self) -> torch.Tensor:
        return _quantize(self.weight, self.precision)

    def forward(self, spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """spikes: (batch, n_pre) 0/1 float. Returns (currents, events)."""
        batch = spikes.shape[0]
        w_q = self.quantized_weight()
        contrib = spikes[:, self.edges_pre] * w_q  # (batch, n_edges)
        currents = torch.zeros(batch, self.n_post, dtype=contrib.dtype)
        currents = currents.index_add(1, self.edges_post, contrib)
        # A delivered entry is an event regardless of its weight value.
        active = (spikes.detach() > 0.5).float()[:, self.edges_pre]
        counts = torch.zeros(batch, self.n_post).index_add(1, self.edges_post, active)
        return currents, counts > 0


class BlockSparseSynapse(nn.Module):
    """Static block-sparse connectivity: dense weights inside active blocks.

    block_mask: (n_pre_blocks, n_post_blocks) bool buffer — the frozen,
    hashable routing artifact of the thesis (handoff §5). P4 learns it;
    here it is given. Weights are stored dense and masked (reference
    implementation — a gathered layout is a later GPU optimization).
    """

    def __init__(self, n_pre: int, n_post: int, block_size: int,
                 block_mask, precision: str = "ternary",
                 weight_sigma: float = 1.0, seed: int | None = None) -> None:
        super().__init__()
        if n_pre % block_size or n_post % block_size:
            raise ValueError("n_pre and n_post must be multiples of block_size")
        self.n_pre, self.n_post, self.block_size = n_pre, n_post, block_size
        self.p_blocks = n_pre // block_size
        self.q_blocks = n_post // block_size
        mask = torch.as_tensor(block_mask, dtype=torch.bool)
        if mask.shape != (self.p_blocks, self.q_blocks):
            raise ValueError(f"block_mask must be {(self.p_blocks, self.q_blocks)}")
        self.register_buffer("block_mask", mask)
        self.precision = precision
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.weight = nn.Parameter(
            torch.randn(self.p_blocks, self.q_blocks, block_size, block_size,
                        generator=gen) * weight_sigma)

    def quantized_weight(self) -> torch.Tensor:
        """Per-block quantization (independent absmean scale per block),
        masked to the active topology."""
        w_q = _quantize(self.weight, self.precision, block_dims=(-1, -2))
        return w_q * self.block_mask[..., None, None]

    def flop_fraction(self) -> float:
        """Fraction of the dense matmul actually computed — the honest
        efficiency metric available without GPU kernels (handoff §6)."""
        return float(self.block_mask.float().mean())

    def forward(self, spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = spikes.shape[0]
        s = spikes.reshape(batch, self.p_blocks, self.block_size)
        w_q = self.quantized_weight()
        currents = torch.einsum("bpi,pqij->bqj", s, w_q).reshape(batch, self.n_post)
        block_any = (s.detach() > 0.5).any(-1).float()          # (batch, P)
        post_events = (block_any @ self.block_mask.float()) > 0  # (batch, Q)
        events = post_events[:, :, None].expand(
            batch, self.q_blocks, self.block_size).reshape(batch, self.n_post)
        return currents, events

    def to_edge_list(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand active blocks to chip dendrite entries (pre, post, weight).
        Zero-weight entries inside active blocks are kept: they are valid
        synapses and deliver events."""
        pres, posts, weights = [], [], []
        w_q = self.quantized_weight().detach()
        for p in range(self.p_blocks):
            for q in range(self.q_blocks):
                if not bool(self.block_mask[p, q]):
                    continue
                i = torch.arange(self.block_size)
                pre_idx = (p * self.block_size + i).repeat_interleave(self.block_size)
                post_idx = (q * self.block_size + i).repeat(self.block_size)
                pres.append(pre_idx)
                posts.append(post_idx)
                weights.append(w_q[p, q].reshape(-1))
        cat = lambda xs: torch.cat(xs) if xs else torch.empty(0)
        return cat(pres).long(), cat(posts).long(), cat(weights)
