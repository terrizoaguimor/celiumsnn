# SPDX-License-Identifier: Apache-2.0
"""snnTorch-style adapter for the CeliumNeUR neuron (P6).

`CeliumLeaky` exposes DiffLIF through the calling convention snnTorch users
expect from `snn.Leaky` / `snn.Synaptic`:

    lif = CeliumLeaky(theta=48, leak_shift=15, n_neurons=128)
    state = None
    for t in range(T):
        spk, state = lif(x[:, t], state)

so celiumsnn dynamics drop into snnTorch training loops, dataset tooling
(`snntorch.spikegen`, Tonic) and utilities without forking snnTorch — the
handoff's "mental fork" discipline. State is the pair (v, cd); pass None
(or call `init_celium`) to start a fresh batch. The forward pass is DiffLIF
verbatim: integer-exact, golden-equivalent, surrogate-gradient trainable.
"""

from __future__ import annotations

import torch
from torch import nn

from celiumsnn.lif_diff import DiffLIF


class CeliumLeaky(nn.Module):
    def __init__(self, **diff_lif_kwargs) -> None:
        super().__init__()
        self.core = DiffLIF(**diff_lif_kwargs)

    def init_celium(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        self.core.reset_state(batch_size)
        return self.core.v, self.core.cd

    def forward(self, input_: torch.Tensor, state=None):
        """Returns (spk, state), snnTorch-style. `state` is (v, cd) or None."""
        input_ = torch.atleast_2d(torch.as_tensor(input_, dtype=torch.float32))
        if state is None:
            self.init_celium(input_.shape[0])
        else:
            self.core.v, self.core.cd = state
        spk = self.core.step(input_)
        return spk, (self.core.v, self.core.cd)
