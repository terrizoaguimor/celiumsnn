# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differential equivalence: IntLIF vs the golden Soma, phase by phase.

Golden is driven event-by-event (apply_synaptic_input per weight, then
advance_time); the model is driven with the per-phase sum and event mask.
End-of-phase state (v, cd, fired) must be EXACTLY equal under C1-C4 of
P0-SEMANTICS.md §4. Event generation is same-sign per (neuron, phase),
which satisfies C3/C4 structurally; C1/C2 are satisfied by configuration.

The final test documents the contract boundary: a C4 violation where the
golden model and the tick-synchronous model legitimately diverge.
"""

import random

import pytest
import torch
from soma import NeuronParams, Soma  # golden, via pyproject pythonpath

from celiumsnn import IntLIF

N_NEURONS = 32
N_PHASES = 200


def build_pair(rng: random.Random, subtractive: bool):
    params = [
        NeuronParams(
            theta=rng.randint(1, 300),
            leak_shift=rng.randint(0, 15),
            refractory_ticks=rng.randint(1, 5),
            subtractive_reset=subtractive,
        )
        for _ in range(N_NEURONS)
    ]
    goldens = [Soma(p) for p in params]
    lif = IntLIF(
        theta=[p.theta for p in params],
        leak_shift=[p.leak_shift for p in params],
        refractory_ticks=[p.refractory_ticks for p in params],
        subtractive_reset=[p.subtractive_reset for p in params],
    )
    return goldens, lif


def run_phase(rng, goldens, lif, max_events: int):
    currents = [0] * N_NEURONS
    events = [False] * N_NEURONS
    golden_fired = [False] * N_NEURONS
    for i, soma in enumerate(goldens):
        n_events = rng.randint(0, max_events)
        sign = rng.choice([1, -1])
        if abs(soma.v) >= 30000:
            # C3: near the int16 rails a mid-phase fire changes where the
            # clamp engages, so batched and per-event clamping diverge.
            # Quiet phases only until leak pulls the neuron back.
            n_events = 0
        elif soma.refractory_countdown == 0 and soma.v >= soma.params.theta:
            # C4: a neuron entering the phase superthreshold with cd==0 fires
            # golden on its first event regardless of sign; inhibition here is
            # order-dependent (see P0-SEMANTICS.md §4). Excitation only.
            sign = 1
        for _ in range(n_events):
            w = sign * rng.randint(1, 127)  # same-sign per (neuron, phase): C3/C4
            currents[i] += w
            events[i] = True
            if soma.apply_synaptic_input(w):
                golden_fired[i] = True
        if soma.advance_time():
            golden_fired[i] = True
    model_fired = lif.step(torch.tensor([currents]), torch.tensor([events]))
    return golden_fired, model_fired


def assert_state_equal(goldens, lif, phase: int):
    assert lif.v[0].tolist() == [s.v for s in goldens], f"v diverged at phase {phase}"
    assert lif.cd[0].tolist() == [s.refractory_countdown for s in goldens], \
        f"cd diverged at phase {phase}"


@pytest.mark.parametrize("seed", range(5))
def test_subtractive_reset_bit_exact_over_random_stimulus(seed):
    rng = random.Random(seed)
    goldens, lif = build_pair(rng, subtractive=True)
    total_fires = 0
    for phase in range(N_PHASES):
        golden_fired, model_fired = run_phase(rng, goldens, lif, max_events=3)
        assert model_fired[0].tolist() == golden_fired, f"spikes diverged at phase {phase}"
        assert_state_equal(goldens, lif, phase)
        total_fires += sum(golden_fired)
    assert total_fires > 50, "vacuous run: I8 — tests must fire neurons"


@pytest.mark.parametrize("seed", range(3))
def test_reset_to_zero_bit_exact_at_one_event_per_phase(seed):
    # Reset-to-zero does not commute with later same-phase additions (C1),
    # so its equivalence contract is <=1 event per phase.
    rng = random.Random(1000 + seed)
    goldens, lif = build_pair(rng, subtractive=False)
    total_fires = 0
    for phase in range(N_PHASES):
        golden_fired, model_fired = run_phase(rng, goldens, lif, max_events=1)
        assert model_fired[0].tolist() == golden_fired, f"spikes diverged at phase {phase}"
        assert_state_equal(goldens, lif, phase)
        total_fires += sum(golden_fired)
    assert total_fires > 20, "vacuous run: I8 — tests must fire neurons"


def test_c4_violation_diverges_as_documented():
    # P0-SEMANTICS.md §4 C4: excitation crossing theta followed by same-phase
    # inhibition. Golden fires on the +100 prefix; the phase total (50) never
    # reaches theta, so the tick-synchronous model must not fire. This is the
    # contract boundary, asserted so it stays documented behavior.
    p = NeuronParams(theta=100, leak_shift=15, refractory_ticks=1)
    golden = Soma(p)
    fired_golden = golden.apply_synaptic_input(100)
    fired_golden |= golden.apply_synaptic_input(-50)
    fired_golden |= golden.advance_time()

    lif = IntLIF(theta=100, leak_shift=15, refractory_ticks=1)
    fired_model = bool(lif.step(torch.tensor([[50]]), torch.tensor([[True]]))[0, 0])

    assert fired_golden is True and fired_model is False
    assert golden.v == -49   # (100-100) - 50, leak toward zero: -50 -> -49
    assert int(lif.v[0, 0]) == 49  # 50 unfired, leak -> 49


def test_c3_violation_diverges_as_documented():
    # P0-SEMANTICS.md §4 C3: same-sign inputs near the positive rail. Golden
    # fires mid-phase (62 first: 32762 -> subtract 20 -> 32742), so its second
    # add clamps (32742+91 -> 32767); the batched model clamps the raw sum
    # first (32700+153 -> 32767) and subtracts theta after. Both fire; the
    # clamp engages at different points. Contract boundary, kept documented.
    p = NeuronParams(theta=20, leak_shift=15, refractory_ticks=1)
    golden = Soma(p, v0=32700)
    fired_golden = golden.apply_synaptic_input(62)
    golden.apply_synaptic_input(91)
    golden.advance_time()

    lif = IntLIF(theta=20, leak_shift=15, refractory_ticks=1)
    lif.reset_state(v0=32700)
    fired_model = bool(lif.step(torch.tensor([[153]]))[0, 0])

    assert fired_golden is True and fired_model is True
    assert golden.v == 32766
    assert int(lif.v[0, 0]) == 32746
