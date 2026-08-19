# SPDX-License-Identifier: Apache-2.0
"""P4 gate mechanics: hard-concrete behavior, L0 penalty, freeze semantics."""

import torch

from celiumsnn import (
    BlockSparseSynapse,
    GatedBlockSparseSynapse,
    HardConcreteGate,
    topology_hash,
)


def test_samples_live_in_unit_interval_and_hit_exact_zero_and_one():
    torch.manual_seed(0)
    gate = HardConcreteGate((1000,), init_log_alpha=0.0)
    z = gate.sample()
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0
    assert (z == 0.0).any() and (z == 1.0).any()  # the point of hard concrete


def test_deterministic_gate_saturates_at_extremes():
    gate = HardConcreteGate((2,))
    with torch.no_grad():
        gate.log_alpha.copy_(torch.tensor([10.0, -10.0]))
    assert gate.deterministic().tolist() == [1.0, 0.0]
    assert gate.open_mask().tolist() == [True, False]


def test_l0_penalty_is_monotone_in_log_alpha():
    lo, mid, hi = (HardConcreteGate((1,), a).l0_penalty() for a in (-4.0, 0.0, 4.0))
    assert float(lo) < float(mid) < float(hi)
    assert 0.0 < float(lo) and float(hi) < 1.0


def test_gate_gradient_flows_through_sampled_path():
    torch.manual_seed(1)
    gate = HardConcreteGate((8,), init_log_alpha=0.5)
    (gate.sample() ** 2).sum().backward()
    assert gate.log_alpha.grad is not None
    assert float(gate.log_alpha.grad.abs().sum()) > 0


def test_saturated_gated_forward_equals_frozen_forward():
    torch.manual_seed(2)
    gated = GatedBlockSparseSynapse(16, 24, 8, precision="ternary", seed=3)
    with torch.no_grad():
        gated.gates.log_alpha.copy_(
            torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]]))
    gated.eval()
    frozen = gated.freeze()
    assert frozen.block_mask.tolist() == [[True, False, True], [False, True, False]]
    spikes = (torch.rand(4, 16) < 0.3).float()
    gi, ge = gated(spikes)
    fi, fe = frozen(spikes)
    assert torch.equal(gi, fi)  # saturated gates are exactly 0/1
    assert torch.equal(ge, fe)


def test_partial_gate_scales_currents_but_events_follow_connectivity():
    gated = GatedBlockSparseSynapse(8, 8, 8, precision="float", seed=4)
    with torch.no_grad():
        gated.weight.fill_(1.0)
        gated.gates.log_alpha.fill_(0.0)  # deterministic gate = 0.45
    gated.eval()
    z = float(gated.gates.deterministic())
    currents, events = gated(torch.ones(1, 8))
    assert abs(float(currents[0, 0]) - 8.0 * z) < 1e-5
    assert bool(events.all())  # connected while z > 0, regardless of scale


def test_frozen_artifact_is_hashable_and_order_sensitive():
    m1 = torch.tensor([[True, False]])
    m2 = torch.tensor([[False, True]])
    assert topology_hash(m1, m2) == topology_hash(m1, m2)
    assert topology_hash(m1, m2) != topology_hash(m2, m1)
    assert len(topology_hash(m1)) == 64


def test_freeze_copies_weights_and_stops_gate_training():
    gated = GatedBlockSparseSynapse(16, 16, 8, precision="ternary", seed=5)
    frozen = gated.freeze()
    assert isinstance(frozen, BlockSparseSynapse)
    assert not isinstance(frozen, GatedBlockSparseSynapse)
    assert torch.equal(frozen.weight, gated.weight)
    assert all(name != "gates.log_alpha" for name, _ in frozen.named_parameters())
