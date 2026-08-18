# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integer LIF neuron with exact CeliumNeUR golden-model semantics (P1).

Implements the reference forward pass of P0-SEMANTICS.md §4, which is
tick-synchronous and batched. Equivalence with golden/soma.py holds under
the C1-C5 contract:

  C1  subtractive reset for any neuron receiving >1 event per phase
      (reset-to-zero is supported but order-invariant only at <=1 event/phase)
  C2  refractory_ticks >= 1 (enforced here; golden allows 0, where a neuron's
      per-phase output is a count, not a bit)
  C3  no intermediate int16 saturation inside a phase
  C4  no mixed-sign threshold crossing inside a phase
  C5  external stimulation folded into `input_current` must be subthreshold
      at injection time in the golden reference

All state and arithmetic is integer (int32 carrying int16-range values:
the widened accumulator of invariant I6). No autograd — the differentiable
surrogate twin is P2's deliverable, not P1's.
"""

from __future__ import annotations

import torch
from torch import nn

VMEM_BITS = 16
VMEM_MAX = (1 << (VMEM_BITS - 1)) - 1   # +32767
VMEM_MIN = -(1 << (VMEM_BITS - 1))      # -32768

WEIGHT_BITS = 8
WEIGHT_MAX = (1 << (WEIGHT_BITS - 1)) - 1  # +127
WEIGHT_MIN = -(1 << (WEIGHT_BITS - 1))     # -128

THETA_MIN, THETA_MAX = 1, VMEM_MAX          # golden/soma.py:69-70
LEAK_SHIFT_MIN, LEAK_SHIFT_MAX = 0, 15      # golden/soma.py:42-43
REFRACTORY_MIN, REFRACTORY_MAX = 1, 255     # word field 8 bits; >=1 is C2


def saturate_vmem(raw: torch.Tensor) -> torch.Tensor:
    """Clamp into signed 16-bit membrane range (I6: clamps, never wraps)."""
    return torch.clamp(raw, VMEM_MIN, VMEM_MAX)


def ceiling_leak_amount(v: torch.Tensor, leak_shift: torch.Tensor) -> torch.Tensor:
    """Signed magnitude ceil(|v| / 2**k) toward zero, mirror of golden/soma.py:35-48."""
    two_pow_k = torch.ones_like(leak_shift) << leak_shift
    share = (v.abs() + two_pow_k - 1) // two_pow_k
    return torch.where(v >= 0, share, -share)


def _per_neuron(value, n: int, dtype: torch.dtype, name: str) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=dtype).reshape(-1)
    if t.numel() == 1:
        t = t.expand(n).contiguous()
    if t.numel() != n:
        raise ValueError(f"{name} must have 1 or {n} entries, got {t.numel()}")
    return t


class IntLIF(nn.Module):
    """Vectorized CeliumNeUR neuron layer; one step() = one global tick.

    State shape is (batch, n_neurons); parameters are per-neuron (I7).
    step(input_current, has_event) returns the boolean spike tensor for the
    phase; spikes are meant to be delivered to targets at the NEXT step
    (uniform 1-phase delay of the phase contract).
    """

    def __init__(self, theta, leak_shift, refractory_ticks,
                 subtractive_reset=True, n_neurons: int | None = None) -> None:
        super().__init__()
        if n_neurons is None:
            n_neurons = max(
                torch.as_tensor(x).reshape(-1).numel()
                for x in (theta, leak_shift, refractory_ticks)
            )
        theta = _per_neuron(theta, n_neurons, torch.int32, "theta")
        leak_shift = _per_neuron(leak_shift, n_neurons, torch.int32, "leak_shift")
        refractory = _per_neuron(refractory_ticks, n_neurons, torch.int32, "refractory_ticks")
        subtractive = _per_neuron(subtractive_reset, n_neurons, torch.bool, "subtractive_reset")

        if not bool(((THETA_MIN <= theta) & (theta <= THETA_MAX)).all()):
            raise ValueError(f"theta must be in [{THETA_MIN}, {THETA_MAX}]")
        if not bool(((LEAK_SHIFT_MIN <= leak_shift) & (leak_shift <= LEAK_SHIFT_MAX)).all()):
            raise ValueError(f"leak_shift must be in [{LEAK_SHIFT_MIN}, {LEAK_SHIFT_MAX}]")
        if not bool(((REFRACTORY_MIN <= refractory) & (refractory <= REFRACTORY_MAX)).all()):
            raise ValueError(
                f"refractory_ticks must be in [{REFRACTORY_MIN}, {REFRACTORY_MAX}] "
                "(C2: the tick-synchronous contract needs >=1; R=0 multi-fire "
                "phases exist only in the golden/RTL event world)")

        self.n_neurons = n_neurons
        self.register_buffer("theta", theta)
        self.register_buffer("leak_shift", leak_shift)
        self.register_buffer("refractory", refractory)
        self.register_buffer("subtractive", subtractive)
        self.v: torch.Tensor | None = None
        self.cd: torch.Tensor | None = None
        self.reset_state()

    def reset_state(self, batch_size: int = 1, v0=None) -> None:
        shape = (batch_size, self.n_neurons)
        if v0 is None:
            v = torch.zeros(shape, dtype=torch.int32)
        else:
            v = saturate_vmem(
                torch.as_tensor(v0, dtype=torch.int32).expand(shape).contiguous())
        self.v = v
        self.cd = torch.zeros(shape, dtype=torch.int32)

    def _reset_potential(self, v: torch.Tensor, fired: torch.Tensor) -> torch.Tensor:
        after = torch.where(self.subtractive, v - self.theta, torch.zeros_like(v))
        return torch.where(fired, after, v)

    def step(self, input_current, has_event=None) -> torch.Tensor:
        """One global tick. `input_current` is the phase sum of synaptic
        weights (+ external stim) per neuron, int, shape broadcastable to
        (batch, n_neurons). `has_event` marks event PRESENCE per neuron —
        required for exact golden equivalence, since a zero-sum or
        zero-weight event still triggers a pre-leak evaluation in golden.
        Defaults to `input_current != 0` (correct whenever no zero-sum
        phase occurs)."""
        I = torch.as_tensor(input_current, dtype=torch.int32)
        I = torch.broadcast_to(I, self.v.shape)
        if has_event is None:
            has_event = I != 0
        else:
            has_event = torch.broadcast_to(
                torch.as_tensor(has_event, dtype=torch.bool), self.v.shape)

        # Event path: integrate the phase sum, evaluate pre-leak (gated on E).
        v = saturate_vmem(self.v + I)
        f_evt = has_event & (self.cd == 0) & (v >= self.theta)
        v = self._reset_potential(v, f_evt)
        cd = torch.where(f_evt, self.refractory.expand_as(self.cd), self.cd)

        # Tick path: leak toward zero, refractory-exit evaluation post-leak.
        v = saturate_vmem(v - ceiling_leak_amount(v, self.leak_shift))
        f_tick = (cd == 0) & (v >= self.theta)
        v = self._reset_potential(v, f_tick)
        cd = torch.where(f_tick, self.refractory.expand_as(cd), cd)

        # Decrement AFTER evaluation: preserves the golden asymmetry
        # (event fire blocks R ticks, tick fire blocks R-1). soma.py:103-107.
        cd = torch.clamp(cd - 1, min=0)

        self.v, self.cd = v, cd
        return f_evt | f_tick
