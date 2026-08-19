# SPDX-License-Identifier: Apache-2.0
"""DiffLIF (P2): forward bit-equivalence with IntLIF, and gradient sanity.

The forward test chains the verification: golden ≡ IntLIF (P1 suite) and
IntLIF ≡ DiffLIF (here), so the trainable object carries the RTL semantics.
Both are tick-synchronous batched models, so NO C-conditions are needed —
equality must hold for arbitrary mixed-sign integer currents.
"""

import random

import pytest
import torch

from celiumsnn import DiffLIF, IntLIF, quantize_int8, quantize_ternary, spike, surrogate_kernel

N, T = 16, 100


@pytest.mark.parametrize("seed", range(5))
def test_forward_bit_equivalence_with_intlif(seed):
    rng = random.Random(seed)
    theta = [rng.randint(1, 300) for _ in range(N)]
    k = [rng.randint(0, 15) for _ in range(N)]
    r = [rng.randint(1, 5) for _ in range(N)]
    ilif = IntLIF(theta=theta, leak_shift=k, refractory_ticks=r)
    dlif = DiffLIF(theta=theta, leak_shift=k, refractory_ticks=r)
    for t in range(T):
        I = torch.tensor([[rng.randint(-300, 300) for _ in range(N)]])
        E = torch.tensor([[rng.random() < 0.7 for _ in range(N)]])
        si = ilif.step(I, E)
        sd = dlif.step(I.float(), E)
        assert torch.equal(sd, si.float()), f"spikes diverged at t={t}"
        assert torch.equal(dlif.v, ilif.v.float()), f"v diverged at t={t}"
        assert torch.equal(dlif.cd, ilif.cd), f"cd diverged at t={t}"


def test_forward_equivalence_near_the_rails():
    # Saturation and the ceil leak at extreme magnitudes.
    ilif = IntLIF(theta=32767, leak_shift=15, refractory_ticks=1)
    dlif = DiffLIF(theta=32767, leak_shift=15, refractory_ticks=1)
    for I in (32000, 5000, -70000, -5000, 100000):
        cur = torch.tensor([[I]])
        ilif.step(cur)
        dlif.step(cur.float())
        assert torch.equal(dlif.v, ilif.v.float())


def test_surrogate_gradient_flows_to_input_weight():
    dlif = DiffLIF(theta=100, leak_shift=15, refractory_ticks=1,
                   surrogate_shape="triangular", surrogate_width=50.0)
    w = torch.tensor([[80.0]], requires_grad=True)
    s = dlif.step(w)
    s.sum().backward()
    # Both comparators contribute when neither fires (they both evaluate in
    # golden too): event path at v=80 (u=-20/50) and tick path post-leak at
    # v=79 (u=-21/50), the latter through the leak slope (1 - 2^-15).
    expected = (1 - 20 / 50) / 50 + (1 - 21 / 50) / 50 * (1 - 2**-15)
    assert w.grad is not None
    assert abs(float(w.grad) - expected) < 1e-6


def test_gradient_is_zero_outside_triangular_support():
    dlif = DiffLIF(theta=100, leak_shift=15, refractory_ticks=1,
                   surrogate_shape="triangular", surrogate_width=10.0)
    w = torch.tensor([[50.0]], requires_grad=True)  # 50 units below theta
    dlif.step(w).sum().backward()
    assert float(w.grad) == 0.0


def test_bptt_carries_gradient_across_ticks():
    # Input at t=0 influences the spike at t=2 through the membrane state.
    dlif = DiffLIF(theta=100, leak_shift=15, refractory_ticks=1,
                   surrogate_shape="atan", surrogate_width=50.0)
    w = torch.tensor([[60.0]], requires_grad=True)
    dlif.step(w)                      # v = 59
    dlif.step(torch.zeros(1, 1))      # v = 58
    s = dlif.step(torch.zeros(1, 1))  # v = 57, still subthreshold
    s.sum().backward()
    assert w.grad is not None and float(w.grad) != 0.0


def test_detach_reset_blocks_reset_path_only():
    kwargs = dict(theta=50, leak_shift=15, refractory_ticks=1,
                  surrogate_shape="atan", surrogate_width=25.0)
    grads = {}
    for detach in (True, False):
        dlif = DiffLIF(detach_reset=detach, **kwargs)
        w = torch.tensor([[70.0]], requires_grad=True)
        dlif.step(w)                      # fires, subtractive reset
        dlif.step(torch.zeros(1, 1))
        loss = dlif.v.sum()               # membrane after reset+leaks
        loss.backward()
        grads[detach] = float(w.grad)
    assert grads[True] != grads[False]  # reset path contributes iff not detached


def test_surrogate_kernels_peak_at_one_and_decay():
    u = torch.tensor([0.0, 1.0, 5.0])
    for shape in ("atan", "fast_sigmoid", "triangular"):
        g = surrogate_kernel(u, shape)
        assert float(g[0]) == 1.0
        assert float(g[1]) < 1.0 and float(g[2]) <= float(g[1])


def test_spike_forward_is_golden_ge_condition():
    x = torch.tensor([-1.0, 0.0, 1.0])
    assert spike(x, 10.0, "atan").tolist() == [0.0, 1.0, 1.0]  # >= : H(0)=1


def test_quantizers_land_on_the_chip_grid():
    w = torch.randn(64, 64) * 10
    q8 = quantize_int8(w)
    assert torch.equal(q8, q8.round()) and q8.min() >= -128 and q8.max() <= 127
    qt = quantize_ternary(w)
    assert set(qt.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_quantizer_ste_passes_gradient():
    w = torch.randn(8, 8, requires_grad=True)
    quantize_ternary(w).sum().backward()
    assert w.grad is not None and float(w.grad.abs().sum()) > 0
