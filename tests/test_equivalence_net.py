# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differential equivalence: IntLIF + edge-list delivery vs the golden
NeuroSandbox (mesh + dendrite + somas), phase by phase.

The fabric is semantically transparent under the phase contract
(P0-SEMANTICS.md §3): credits/FIFOs affect only intra-phase timing, spikes
staged at phase t deliver at t+1 uniformly, and duplicate table entries
have real multiplicity. So a scatter_add over an edge list plus the 1-step
delay must reproduce the whole chip bit-exactly under C1-C5.

Harness constraints mapping to the contract:
  - all weights into a given post neuron share one polarity (C4/C3);
  - subtractive reset everywhere, refractory >= 1 (C1/C2);
  - external stimulation is clamped subthreshold at injection (C5);
  - activity is seeded by superthreshold initial potentials, which fire
    through the TICK path at phase 0 — exercising the §2.4 asymmetry.

This inherits the mesh/routing golden tests transitively: if routing or
multiplicity broke, per-phase state equality would fail.
"""

import random

import pytest
import torch
from golden_net import NEURONS_PER_CORE, GLOBAL_NEURONS, NeuroSandbox
from soma import NeuronParams

from celiumsnn import EdgeListSynapse, IntLIF, saturate_vmem

ACTIVE_PER_CORE = 16
N_PHASES = 80
N_EDGES = 160

DEAD = NeuronParams(theta=32767, leak_shift=0, refractory_ticks=1)


def build_world(seed: int):
    rng = random.Random(seed)
    cores = GLOBAL_NEURONS // NEURONS_PER_CORE
    gids = [c * NEURONS_PER_CORE + j for c in range(cores) for j in range(ACTIVE_PER_CORE)]
    local = {g: i for i, g in enumerate(gids)}
    n = len(gids)

    params = {}
    for g in gids:
        params[g] = NeuronParams(
            theta=rng.randint(20, 200),
            leak_shift=rng.randint(0, 15),
            refractory_ticks=rng.randint(1, 4),
            subtractive_reset=True,
        )

    # One polarity per POST neuron: every weight it ever receives (synaptic
    # or stim) shares that sign, satisfying C3/C4 structurally. Seeds must be
    # positive-polarity: a seed can enter a phase superthreshold with cd==0,
    # where inhibitory input is order-dependent (P0-SEMANTICS.md §4, C4).
    polarity = {g: rng.choice([1, -1]) for g in gids}

    seeds = rng.sample([g for g in gids if polarity[g] > 0], 8)
    # Superthreshold v0 that survives the phase-0 leak
    # (k >= 1 guarantees post-leak >= v0/2 >= 2*theta).
    for g in seeds:
        p = params[g]
        if p.leak_shift == 0:
            params[g] = NeuronParams(theta=p.theta, leak_shift=rng.randint(1, 15),
                                     refractory_ticks=p.refractory_ticks,
                                     subtractive_reset=True)
        assert 4 * params[g].theta <= 32767
    v0 = {g: 4 * params[g].theta for g in seeds}

    edges = []
    for _ in range(N_EDGES):
        pre, post = rng.choice(gids), rng.choice(gids)
        edges.append((pre, post, polarity[post] * rng.randint(1, 127)))
    for _ in range(10):  # duplicate (pre, post) pairs: real multiplicity
        pre, post, _w = rng.choice(edges)
        edges.append((pre, post, polarity[post] * rng.randint(1, 127)))

    box = NeuroSandbox([params.get(g, DEAD) for g in range(GLOBAL_NEURONS)])
    for pre, post, w in edges:
        box.wire(pre, post, w)
    for g, value in v0.items():
        box.somas[g].v = value  # test-harness poke; sandbox has no v0 hook

    lif = IntLIF(
        theta=[params[g].theta for g in gids],
        leak_shift=[params[g].leak_shift for g in gids],
        refractory_ticks=[params[g].refractory_ticks for g in gids],
    )
    v0_tensor = torch.zeros(1, n, dtype=torch.int32)
    for g, value in v0.items():
        v0_tensor[0, local[g]] = value
    lif.reset_state(v0=v0_tensor)

    # Delivery goes through the P3 primitive, certifying it against golden:
    # int8 quantization is the identity on the already-integer weights.
    synapse = EdgeListSynapse(
        n_pre=n, n_post=n,
        edges_pre=[local[p] for p, _, _ in edges],
        edges_post=[local[q] for _, q, _ in edges],
        weights_init=[w for _, _, w in edges],
        precision="int8",
    )
    return rng, box, lif, gids, local, polarity, synapse


@pytest.mark.parametrize("seed", range(3))
def test_whole_network_bit_exact_against_neurosandbox(seed):
    rng, box, lif, gids, local, polarity, synapse = build_world(seed)
    n = len(gids)
    prev_spikes = torch.zeros(n, dtype=torch.bool)
    total_fires = 0

    for phase in range(N_PHASES):
        # External stimulation, C5: clamped subthreshold at injection.
        stim_currents = torch.zeros(n, dtype=torch.int32)
        stim_events = torch.zeros(n, dtype=torch.bool)
        for g in gids:
            if rng.random() >= 0.5:
                continue
            soma = box.somas[g]
            if polarity[g] > 0:
                headroom = soma.params.theta - 1 - soma.v
                w = min(headroom, 127)
                if w < 1:
                    continue
            else:
                w = -rng.randint(1, 64)
            if soma.refractory_countdown == 0 and soma.v + w >= soma.params.theta:
                continue  # C5: stimulation must never fire at injection
            box.stimulate(g, w)
            stim_currents[local[g]] += w
            stim_events[local[g]] = True

        box.tick()
        golden_fired = {g for t, g in box.fire_log if t == phase}
        assert golden_fired <= set(gids), "a dead neuron fired"
        assert all(abs(box.somas[g].v) < 30000 for g in gids), \
            "harness drifted into the C3 rail region — reshape generation"

        with torch.no_grad():
            syn_currents, syn_events = synapse(prev_spikes.float().unsqueeze(0))
        spikes = lif.step(syn_currents + stim_currents.unsqueeze(0),
                          syn_events | stim_events.unsqueeze(0))
        prev_spikes = spikes[0].bool()

        model_fired = {gids[i] for i in range(n) if bool(prev_spikes[i])}
        assert model_fired == golden_fired, f"spike sets diverged at phase {phase}"
        assert lif.v[0].tolist() == [box.somas[g].v for g in gids], \
            f"v diverged at phase {phase}"
        assert lif.cd[0].tolist() == [box.somas[g].refractory_countdown for g in gids], \
            f"cd diverged at phase {phase}"
        total_fires += len(golden_fired)

    assert total_fires > 20, "vacuous run: I8 — tests must fire neurons"
