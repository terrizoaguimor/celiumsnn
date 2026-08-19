# SPDX-License-Identifier: AGPL-3.0-or-later
"""P6 adapter tests: CeliumLeaky behaves like an snnTorch neuron and is
DiffLIF verbatim. The snnTorch integration test runs only when snntorch is
installed (pytest.importorskip)."""

import pytest
import torch

from celiumsnn import DiffLIF
from celiumsnn.snntorch_adapter import CeliumLeaky


def test_adapter_matches_difflif_exactly():
    kwargs = dict(theta=48.0, leak_shift=15, refractory_ticks=1,
                  n_neurons=8, surrogate_width=12.0)
    adapter = CeliumLeaky(**kwargs)
    direct = DiffLIF(**kwargs)
    direct.reset_state(4)
    gen = torch.Generator().manual_seed(0)
    state = None
    for _ in range(50):
        x = torch.randint(-30, 60, (4, 8), generator=gen).float()
        spk_a, state = adapter(x, state)
        spk_d = direct.step(x)
        assert torch.equal(spk_a, spk_d)
        assert torch.equal(state[0], direct.v)
        assert torch.equal(state[1], direct.cd)


def test_adapter_state_none_resets_batch():
    lif = CeliumLeaky(theta=10.0, leak_shift=15, refractory_ticks=1, n_neurons=4)
    spk, state = lif(torch.full((2, 4), 20.0), None)
    assert spk.shape == (2, 4) and bool(spk.all())
    spk2, _ = lif(torch.full((7, 4), 20.0), None)  # new batch size, fresh state
    assert spk2.shape == (7, 4)


def test_adapter_gradient_flows_snntorch_style_loop():
    lif = CeliumLeaky(theta=100.0, leak_shift=15, refractory_ticks=1,
                      n_neurons=1, surrogate_width=50.0)
    w = torch.tensor([[30.0]], requires_grad=True)
    state, out = None, 0.0
    for _ in range(3):
        spk, state = lif(w, state)
        out = out + spk
    out.sum().backward()
    assert w.grad is not None and float(w.grad.abs()) > 0


def test_adapter_composes_with_snntorch_network():
    snn = pytest.importorskip("snntorch")
    torch.manual_seed(0)
    # Mixed net: snnTorch Leaky layer feeding a CeliumLeaky layer.
    fc1, fc2 = torch.nn.Linear(10, 16), torch.nn.Linear(16, 4)
    leaky = snn.Leaky(beta=0.9)
    celium = CeliumLeaky(theta=2.0, leak_shift=15, refractory_ticks=1,
                         n_neurons=4, surrogate_width=0.5)
    mem, cstate = leaky.init_leaky(), None
    x = (torch.rand(6, 5, 10) < 0.3).float()
    out = 0.0
    for t in range(5):
        s1, mem = leaky(fc1(x[:, t]), mem)
        s2, cstate = celium(fc2(s1), cstate)
        out = out + s2
    loss = out.sum()
    loss.backward()
    assert fc1.weight.grad is not None and float(fc1.weight.grad.abs().sum()) > 0
