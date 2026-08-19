# SPDX-License-Identifier: Apache-2.0
"""Ported golden soma tests (celiumneur/golden/test_soma.py) adapted to the
tick-synchronous IntLIF contract (P0-SEMANTICS.md §4).

Adaptations, uniform across the file:
  - golden's event-granular sequences become per-phase input sums;
  - refractory_ticks >= 1 everywhere (C2) — golden's R=0 multi-fire tests
    have no tick-synchronous counterpart and stay golden-only;
  - every step() applies the tick leak, so expected potentials shift by the
    leak amount relative to the golden event-path assertions. NO_LEAK (k=15)
    decays by exactly 1 per tick.
Each test names the golden ancestor it ports.
"""

import pytest
import torch

from celiumsnn import (
    IntLIF,
    VMEM_MAX,
    VMEM_MIN,
    WEIGHT_MAX,
    WEIGHT_MIN,
    ceiling_leak_amount,
    saturate_vmem,
)

FAST_LEAK = 1
NO_LEAK = 15


def make_lif(**overrides) -> IntLIF:
    config = {"theta": 100, "leak_shift": FAST_LEAK, "refractory_ticks": 1}
    config.update(overrides)
    return IntLIF(**config)


def v_of(lif: IntLIF) -> int:
    return int(lif.v[0, 0])


def cd_of(lif: IntLIF) -> int:
    return int(lif.cd[0, 0])


def fired(lif: IntLIF, current, has_event=None) -> bool:
    return bool(lif.step(current, has_event)[0, 0])


# --- Integration & saturation (I6) ------------------------------------------

def test_excitatory_input_integrates_exactly():
    # golden test_soma.py:32 — adapted: one tick of leak (k=15 -> -1) follows.
    lif = make_lif(leak_shift=NO_LEAK)
    assert fired(lif, 30) is False
    assert v_of(lif) == 29


def test_input_saturates_at_positive_rail_without_wrapping():
    # golden test_soma.py:38 — same trick: refractory blocks resets so the
    # pure accumulation path reaches the rail. Regression: lif-tt-asic wrap.
    lif = make_lif(refractory_ticks=5, leak_shift=NO_LEAK)
    assert fired(lif, 100) is True          # v := 0, cd := 5-1 = 4
    assert fired(lif, 10 * VMEM_MAX) is False  # refractory: clamps, no fire
    assert v_of(lif) == VMEM_MAX - 1        # rail, then one tick of -1 leak


def test_input_saturates_at_negative_rail_without_wrapping():
    # golden test_soma.py:49
    lif = make_lif(leak_shift=NO_LEAK)
    lif.reset_state(v0=-32000)
    assert fired(lif, 300 * WEIGHT_MIN) is False
    assert v_of(lif) == VMEM_MIN + 1        # clamped at -32768, leak +1


def test_saturate_vmem_clamps_both_ends():
    # golden test_soma.py:58
    raw = torch.tensor([VMEM_MAX + 10_000, VMEM_MIN - 10_000, 0])
    assert saturate_vmem(raw).tolist() == [VMEM_MAX, VMEM_MIN, 0]


# --- Leak convergence (kills sticky residue) --------------------------------

def test_leak_converges_to_zero_from_small_positive():
    # golden test_soma.py:66 — regression: truncating >>> leaves v in {1..7}.
    lif = make_lif(leak_shift=3, theta=VMEM_MAX)
    lif.reset_state(v0=5)
    ticks = 0
    while v_of(lif) != 0:
        lif.step(0)
        ticks += 1
        assert ticks < 100, "leak must reach exactly zero, not asymptote"
    assert v_of(lif) == 0


def test_leak_converges_to_zero_from_small_negative():
    # golden test_soma.py:77
    lif = make_lif(leak_shift=3, theta=VMEM_MAX)
    lif.reset_state(v0=-5)
    for _ in range(100):
        lif.step(0)
    assert v_of(lif) == 0


def test_leak_is_monotone_decay_toward_zero():
    # golden test_soma.py:84
    lif = make_lif(theta=VMEM_MAX)
    lif.reset_state(v0=200)
    deltas = []
    while v_of(lif) != 0:
        previous = abs(v_of(lif))
        lif.step(0)
        deltas.append(previous - abs(v_of(lif)))
    assert all(delta > 0 for delta in deltas)
    assert deltas == sorted(deltas, reverse=True)  # geometric, not linear


def test_max_leak_shift_decays_by_one_per_tick():
    # golden test_soma.py:95
    k = torch.tensor([NO_LEAK, NO_LEAK, FAST_LEAK])
    v = torch.tensor([10_000, -10_000, 0])
    assert ceiling_leak_amount(v, k).tolist() == [1, -1, 0]


def test_leak_shift_zero_discharges_fully_in_one_tick():
    # P0-SEMANTICS.md §2.2: k=0 is "maximum leak", intended (soma.py:57).
    lif = make_lif(leak_shift=0, theta=VMEM_MAX)
    lif.reset_state(v0=1234)
    lif.step(0)
    assert v_of(lif) == 0


# --- Firing & reset ----------------------------------------------------------

def test_threshold_crossing_fires_at_exact_theta():
    # golden test_soma.py:103 + :123 — pins the >= condition (soma.py:113).
    lif = make_lif()
    assert fired(lif, 99) is False
    lif.reset_state()
    assert fired(lif, 100) is True


def test_subtractive_reset_keeps_residue():
    # golden test_soma.py:109 — residue 20, then one tick of -1 leak.
    lif = make_lif(leak_shift=NO_LEAK)
    assert fired(lif, 120) is True
    assert v_of(lif) == 19


def test_reset_to_zero_mode_drops_residue():
    # golden test_soma.py:117
    lif = make_lif(subtractive_reset=False)
    assert fired(lif, 120) is True
    assert v_of(lif) == 0


# --- Refractory --------------------------------------------------------------

def test_refractory_blocks_spiking_but_still_integrates():
    # golden test_soma.py:131 — adapted trace: fire -> v=20, leak -> 19;
    # blocked phase integrates 127 -> 146, leak -> 145.
    lif = make_lif(refractory_ticks=3, leak_shift=NO_LEAK)
    assert fired(lif, 120) is True
    assert fired(lif, 127) is False
    assert v_of(lif) == 145


def test_refractory_enforces_minimum_interspike_interval():
    # golden test_soma.py:138
    lif = make_lif(refractory_ticks=3, leak_shift=NO_LEAK)
    fire_ticks = [t for t in range(200) if fired(lif, 100)]
    intervals = [b - a for a, b in zip(fire_ticks, fire_ticks[1:])]
    assert len(fire_ticks) >= 4
    assert all(interval >= 3 for interval in intervals)


def test_superthreshold_survives_refractory_then_fires_on_tick_path():
    # golden test_soma.py:152 — the charge survives masking and fires
    # POST-LEAK on refractory exit with no new input (the E-gate case).
    lif = make_lif(refractory_ticks=2, leak_shift=NO_LEAK, theta=50)
    assert fired(lif, 127) is True                     # v := 77-1 = 76
    assert fired(lif, 0, has_event=False) is False     # cd 1 -> 0, v = 75
    assert fired(lif, 0, has_event=False) is True      # leak 75->74, fires
    assert v_of(lif) == 24
    assert cd_of(lif) == 1  # tick-path fire: cd := R then decremented (§2.4)


def test_event_gate_distinguishes_pre_and_post_leak_evaluation():
    # P0-SEMANTICS.md §4: a zero-weight event evaluates PRE-leak in golden
    # (apply_synaptic_input(0)); a quiet phase evaluates POST-leak only.
    quiet = make_lif(theta=100, leak_shift=FAST_LEAK)
    quiet.reset_state(v0=150)
    assert fired(quiet, 0, has_event=False) is False   # post-leak 75 < 100
    assert v_of(quiet) == 75

    evented = make_lif(theta=100, leak_shift=FAST_LEAK)
    evented.reset_state(v0=150)
    assert fired(evented, 0, has_event=True) is True   # pre-leak 150 >= 100
    assert v_of(evented) == 25                         # residue 50, leak -> 25


def test_batched_state_evolves_per_row_independently():
    lif = make_lif(leak_shift=NO_LEAK)
    lif.reset_state(batch_size=2)
    out = lif.step(torch.tensor([[100], [30]]))
    assert out.tolist() == [[True], [False]]
    assert lif.v.squeeze(1).tolist() == [0, 29]


# --- Per-neuron independence (I7) --------------------------------------------

def test_minimum_signed_weight_integrates_exactly():
    # golden test_soma.py:164
    lif = make_lif(leak_shift=NO_LEAK)
    assert fired(lif, WEIGHT_MIN) is False
    assert v_of(lif) == WEIGHT_MIN + 1  # leak moves toward zero


def test_neurons_with_distinct_params_evolve_independently():
    # golden test_soma.py:170 — vectorized: two neurons in one layer.
    lif = IntLIF(theta=[80, 90], leak_shift=[1, NO_LEAK], refractory_ticks=[1, 1])
    out = lif.step(torch.tensor([127, 127]))
    assert out.tolist() == [[True, True]]
    # subtractive residues 47 and 37, then each neuron's own leak
    assert lif.v[0].tolist() == [47 - 24, 37 - 1]

    def ticks_until_silent(idx: int, cap: int = 100_000) -> int:
        ticks = 0
        while int(lif.v[0, idx]) != 0:
            lif.step(0)
            ticks += 1
            assert ticks < cap
        return ticks

    assert ticks_until_silent(0) < ticks_until_silent(1)


# --- Boundary sanity ----------------------------------------------------------

def test_resting_neuron_is_silent_without_input():
    # golden test_soma.py:191
    lif = make_lif()
    assert all(fired(lif, 0, has_event=False) is False for _ in range(50))
    assert v_of(lif) == 0


def test_repeated_stimulation_produces_periodic_firing():
    # golden test_soma.py:197 — 25/phase against -1/tick leak: net +24,
    # crossing every ~5 phases.
    lif = make_lif(leak_shift=NO_LEAK)
    fires = sum(fired(lif, 25) for _ in range(20))
    assert fires >= 3


# --- Config validation mirrors golden ----------------------------------------

@pytest.mark.parametrize("bad", [
    {"theta": 0},                # golden soma.py:69 rejects
    {"theta": VMEM_MAX + 1},     # P0 §1: above verified space
    {"leak_shift": 16},          # golden soma.py:42
    {"leak_shift": -1},
    {"refractory_ticks": 0},     # C2: model contract requires >= 1
    {"refractory_ticks": 256},   # word field is 8 bits
])
def test_invalid_params_are_rejected(bad):
    with pytest.raises(ValueError):
        make_lif(**bad)
