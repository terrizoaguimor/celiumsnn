# SPDX-License-Identifier: AGPL-3.0-or-later
"""DiffLIF — differentiable twin of IntLIF (P2).

Forward pass is the SAME tick-synchronous contract as IntLIF
(P0-SEMANTICS.md §4): with integer inputs and integer_exact=True it is
bit-identical to IntLIF (tested), so the gradient diagnostic measures the
real dynamics, not an approximation. Differentiability comes from:

  - surrogate gradient at the two fire comparators (celiumsnn.surrogate);
  - STE through the ceiling of the leak (forward exact ceil, backward the
    smooth v * 2^-k slope);
  - hard-clamp saturation (gradient 1 inside the int16 rails, 0 outside —
    true saturation, not gradient clipping);
  - refractory countdown kept as integer state, entering the graph only as
    a detached 0/1 gate (standard SNN practice for hard refractoriness).

integer_exact=False gives the float LIF control for R1: identical pass
structure, but exponential leak v*(1 - 2^-k), no ceil, no rounding, no
saturation — isolating what the integer/ceil constraints cost.

All membrane values stay well inside float32's exact-integer range
(|v| <= 32768 << 2^24), so integer arithmetic in float tensors is exact.
"""

from __future__ import annotations

import torch
from torch import nn

from celiumsnn.lif import VMEM_MAX, VMEM_MIN, _per_neuron
from celiumsnn.quant import ceil_ste, round_ste
from celiumsnn.surrogate import spike


class DiffLIF(nn.Module):
    def __init__(self, theta, leak_shift, refractory_ticks=1,
                 surrogate_shape: str = "atan", surrogate_width: float = 25.0,
                 integer_exact: bool = True, detach_reset: bool = True,
                 learnable_theta: bool = False, n_neurons: int | None = None) -> None:
        super().__init__()
        if n_neurons is None:
            n_neurons = max(
                torch.as_tensor(x).reshape(-1).numel()
                for x in (theta, leak_shift, refractory_ticks)
            )
        self.n_neurons = n_neurons
        theta_t = _per_neuron(theta, n_neurons, torch.float32, "theta")
        if learnable_theta:
            self.theta = nn.Parameter(theta_t)  # float master; see effective_theta
        else:
            self.register_buffer("theta", theta_t)
        self.learnable_theta = learnable_theta
        self.register_buffer("leak_shift", _per_neuron(leak_shift, n_neurons, torch.float32, "leak_shift"))
        self.register_buffer("two_pow_k", 2.0 ** self.leak_shift)
        self.register_buffer("refractory", _per_neuron(refractory_ticks, n_neurons, torch.int32, "refractory_ticks"))
        if not bool((self.refractory >= 1).all()):
            raise ValueError("refractory_ticks must be >= 1 (C2)")
        self.surrogate_shape = surrogate_shape
        self.surrogate_width = float(surrogate_width)
        self.integer_exact = integer_exact
        self.detach_reset = detach_reset
        self.v: torch.Tensor | None = None
        self.cd: torch.Tensor | None = None
        self.reset_state()

    def reset_state(self, batch_size: int = 1, v0=None) -> None:
        shape = (batch_size, self.n_neurons)
        if v0 is None:
            v = torch.zeros(shape)
        else:
            v = torch.as_tensor(v0, dtype=torch.float32).expand(shape).clone()
            if self.integer_exact:
                v = torch.clamp(v, VMEM_MIN, VMEM_MAX)
        self.v = v
        self.cd = torch.zeros(shape, dtype=torch.int32)

    def detach_state(self) -> None:
        """Cut the BPTT graph at the current tick (for truncated training)."""
        self.v = self.v.detach()

    def _saturate(self, v: torch.Tensor) -> torch.Tensor:
        return torch.clamp(v, VMEM_MIN, VMEM_MAX) if self.integer_exact else v

    def _leak(self, v: torch.Tensor) -> torch.Tensor:
        if self.integer_exact:
            share = ceil_ste(v.abs() / self.two_pow_k)
            return self._saturate(v - share * torch.sign(v).detach())
        return v * (1.0 - 1.0 / self.two_pow_k)

    def effective_theta(self) -> torch.Tensor:
        """Threshold actually used in the pass. Learnable mode keeps a float
        master and STE-rounds onto the chip grid [1, VMEM_MAX] when
        integer_exact (deployment maps the rounded value into IntLIF)."""
        if not self.learnable_theta:
            return self.theta
        if self.integer_exact:
            return torch.clamp(round_ste(self.theta), 1, VMEM_MAX)
        return torch.clamp(self.theta, min=1e-6)

    def _fire(self, v: torch.Tensor, gate: torch.Tensor,
              theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (spike, v_after_reset). gate is a detached 0/1 mask."""
        s = spike(v - theta, self.surrogate_width, self.surrogate_shape) * gate
        s_reset = s.detach() if self.detach_reset else s
        v = v - s_reset * theta  # subtractive only (C1)
        return s, v

    def step(self, input_current: torch.Tensor, has_event=None) -> torch.Tensor:
        I = torch.as_tensor(input_current, dtype=torch.float32)
        I = torch.broadcast_to(I, self.v.shape)
        if has_event is None:
            has_event = (I != 0).detach()
        else:
            has_event = torch.broadcast_to(
                torch.as_tensor(has_event, dtype=torch.bool), self.v.shape)

        v = self._saturate(self.v + I)
        self.last_v_evt = v.detach()   # comparator input, event path (telemetry)
        theta = self.effective_theta()
        gate_evt = (has_event & (self.cd == 0)).to(v.dtype)
        f_evt, v = self._fire(v, gate_evt, theta)
        f_evt_bool = (f_evt > 0.5).detach()
        cd = torch.where(f_evt_bool, self.refractory.expand_as(self.cd), self.cd)

        v = self._leak(v)
        self.last_v_tick = v.detach()  # comparator input, tick path (telemetry)
        gate_tick = (cd == 0).to(v.dtype)
        f_tick, v = self._fire(v, gate_tick, theta)
        cd = torch.where((f_tick > 0.5).detach(), self.refractory.expand_as(cd), cd)

        cd = torch.clamp(cd - 1, min=0)
        self.v, self.cd = v, cd
        return f_evt + f_tick  # exclusive under C2, so this is the 0/1 spike
