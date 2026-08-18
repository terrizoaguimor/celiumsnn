# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mycelium — the D2 macro-architecture (P5).

A stack of recurrent modules over a STATIC block graph: each layer has a
feed-in block-sparse synapse and a recurrent block-sparse synapse (learned-
then-frozen via P4 gates, or fixed), DiffLIF neurons with learnable
per-neuron thresholds and heterogeneous per-neuron leak (chip invariant I7),
and a float linear head over the top layer's spikes, averaged across ticks.

No attention, no data-dependent routing: the compute graph is input-
independent and hashable (handoff §5). Recurrence supplies temporal mixing
the way the chip does — spikes staged at tick t integrate at t+1.
"""

from __future__ import annotations

import torch
from torch import nn

from celiumsnn.gates import GatedBlockSparseSynapse, topology_hash
from celiumsnn.lif_diff import DiffLIF
from celiumsnn.synapse import BlockSparseSynapse


def _make_syn(n_pre, n_post, block, gated, precision, seed):
    if gated:
        return GatedBlockSparseSynapse(n_pre, n_post, block, precision, seed=seed)
    mask = torch.ones(n_pre // block, n_post // block, dtype=torch.bool)
    return BlockSparseSynapse(n_pre, n_post, block, mask, precision, seed=seed)


class Mycelium(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_classes: int, block: int = 64,
                 n_layers: int = 1, gated: bool = False, precision: str = "ternary",
                 theta0: float = 8.0, leak_shift="diverse",
                 surrogate_width: float | None = None, dropout: float = 0.0,
                 seed: int = 0) -> None:
        super().__init__()
        self.hidden, self.n_layers = hidden, n_layers
        if surrogate_width is None:
            surrogate_width = 0.25 * theta0  # P2's D5 default
        self.syn_feed = nn.ModuleList()
        self.syn_rec = nn.ModuleList()
        self.lifs = nn.ModuleList()
        for layer in range(n_layers):
            pre = n_in if layer == 0 else hidden
            self.syn_feed.append(_make_syn(pre, hidden, block, gated, precision, seed + 2 * layer))
            self.syn_rec.append(_make_syn(hidden, hidden, block, gated, precision, seed + 2 * layer + 1))
            if leak_shift == "diverse":
                # Heterogeneous per-neuron time constants over the 4-bit range.
                g = torch.Generator().manual_seed(seed + 7 + layer)
                k = torch.randint(1, 16, (hidden,), generator=g)
            else:
                k = leak_shift
            self.lifs.append(DiffLIF(theta=theta0, leak_shift=k, refractory_ticks=1,
                                     learnable_theta=True, surrogate_width=surrogate_width,
                                     n_neurons=hidden))
        self.readout = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(dropout)  # spikes only; identity at deploy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, n_in) spike counts (float carrying small ints)."""
        batch, T, _ = x.shape
        for lif in self.lifs:
            lif.reset_state(batch)
        states = [torch.zeros(batch, self.hidden) for _ in range(self.n_layers)]
        logits = 0.0
        rate = 0.0
        for t in range(T):
            feed = x[:, t]
            for layer in range(self.n_layers):
                c_f, e_f = self.syn_feed[layer](feed)
                c_r, e_r = self.syn_rec[layer](states[layer])
                s = self.lifs[layer].step(c_f + c_r, e_f | e_r)
                rate = rate + float(s.detach().mean())
                states[layer] = self.dropout(s)
                feed = states[layer]
            logits = logits + self.readout(feed)
        self.last_rate = rate / (T * self.n_layers)
        return logits / T

    def _all_syns(self):
        return list(self.syn_feed) + list(self.syn_rec)

    def l0_penalty(self) -> torch.Tensor:
        return sum(s.l0_penalty() for s in self._all_syns())

    def freeze(self) -> "Mycelium":
        """Binarize gates -> static Mycelium; copies neuron and readout state."""
        frozen = Mycelium.__new__(Mycelium)
        nn.Module.__init__(frozen)
        frozen.hidden, frozen.n_layers = self.hidden, self.n_layers
        frozen.syn_feed = nn.ModuleList(s.freeze() for s in self.syn_feed)
        frozen.syn_rec = nn.ModuleList(s.freeze() for s in self.syn_rec)
        frozen.lifs = nn.ModuleList()
        for lif in self.lifs:
            twin = DiffLIF(theta=1.0, leak_shift=15, refractory_ticks=1,
                           learnable_theta=True, surrogate_width=lif.surrogate_width,
                           n_neurons=self.hidden)
            twin.load_state_dict(lif.state_dict())
            frozen.lifs.append(twin)
        frozen.readout = nn.Linear(self.readout.in_features, self.readout.out_features)
        frozen.readout.load_state_dict(self.readout.state_dict())
        frozen.dropout = self.dropout
        return frozen

    def hash(self) -> str:
        masks = [s.gates.open_mask() if isinstance(s, GatedBlockSparseSynapse)
                 else s.block_mask for s in self._all_syns()]
        return topology_hash(*masks)

    # --- accounting (the P5 curves) ------------------------------------

    @staticmethod
    def _frac(syn) -> float:
        if isinstance(syn, GatedBlockSparseSynapse):
            return float(syn.gates.open_mask().float().mean())
        return syn.flop_fraction()

    def _fracs(self) -> tuple[float, float]:
        feed = sum(self._frac(s) for s in self.syn_feed) / self.n_layers
        rec = sum(self._frac(s) for s in self.syn_rec) / self.n_layers
        return feed, rec

    def param_bytes(self) -> float:
        """Deployment bytes: 2-bit ternary weights in active blocks, block
        masks, per-neuron soma params (theta 16b + leak 4b + refractory 8b),
        fp16 readout."""
        bits = 0.0
        for syn in self._all_syns():
            n_blocks = syn.p_blocks * syn.q_blocks
            bits += self._frac(syn) * n_blocks * syn.block_size ** 2 * 2
            bits += n_blocks
        bits += self.n_layers * self.hidden * (16 + 4 + 8)
        bits += (self.readout.in_features * self.readout.out_features
                 + self.readout.out_features) * 16
        return bits / 8

    def macs_per_step(self) -> float:
        total = self.hidden * self.readout.out_features
        for syn in self._all_syns():
            total += self._frac(syn) * syn.n_pre * syn.n_post
        return total
