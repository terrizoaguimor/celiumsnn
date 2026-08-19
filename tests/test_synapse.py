# SPDX-License-Identifier: Apache-2.0
"""P3 primitives: block-sparse <-> edge-list semantic equivalence, chip-grid
quantization, learnable per-neuron theta, and an end-to-end functional check.

The load-bearing test is block->edge equivalence: an active block IS its
Bp x Bq dendrite entries (zero-weight entries included — they deliver
events), so BlockSparseSynapse inherits EdgeListSynapse's golden
certification (tests/test_equivalence_net.py) by expansion.
"""

import pytest
import torch

from celiumsnn import BlockSparseSynapse, DiffLIF, EdgeListSynapse, IntLIF


def test_edge_list_duplicate_entries_have_real_multiplicity():
    # SPEC §6.2: the same (pre, post) pair twice = two independent events.
    syn = EdgeListSynapse(n_pre=2, n_post=1, edges_pre=[0, 0], edges_post=[0, 0],
                          weights_init=[3.0, 3.0], precision="int8")
    spikes = torch.tensor([[1.0, 0.0]])
    currents, events = syn(spikes)
    assert float(currents.detach()[0, 0]) == 6.0
    assert bool(events[0, 0])


def test_edge_list_zero_weight_edge_still_delivers_event():
    # P0-SEMANTICS.md §4: E is event presence, not I != 0.
    syn = EdgeListSynapse(n_pre=1, n_post=1, edges_pre=[0], edges_post=[0],
                          weights_init=[0.0], precision="int8")
    currents, events = syn(torch.tensor([[1.0]]))
    assert float(currents[0, 0]) == 0.0
    assert bool(events[0, 0])


@pytest.mark.parametrize("seed", range(3))
def test_block_sparse_equals_its_edge_list_expansion(seed):
    gen = torch.Generator().manual_seed(seed)
    p_blocks, q_blocks, bs = 2, 3, 8
    mask = torch.rand(p_blocks, q_blocks, generator=gen) < 0.5
    mask[0, 0] = True  # at least one active block
    block = BlockSparseSynapse(p_blocks * bs, q_blocks * bs, bs, mask,
                               precision="ternary", seed=seed)
    pre, post, w = block.to_edge_list()
    edges = EdgeListSynapse(block.n_pre, block.n_post, pre, post, w,
                            precision="int8")  # weights already on grid
    for _ in range(20):
        spikes = (torch.rand(4, block.n_pre, generator=gen) < 0.15).float()
        bi, be = block(spikes)
        ei, ee = edges(spikes)
        assert torch.equal(bi, ei)
        assert torch.equal(be, ee)


def test_block_quantization_is_per_block_and_on_grid():
    mask = torch.tensor([[True, False]])
    block = BlockSparseSynapse(4, 8, 4, mask, precision="ternary", seed=0)
    with torch.no_grad():
        block.weight[0, 0] *= 100.0  # huge scale in the active block only
    w_q = block.quantized_weight()
    assert set(w_q[0, 0].unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.equal(w_q[0, 1], torch.zeros(4, 4))  # masked block silent
    assert block.flop_fraction() == 0.5


def test_gradients_flow_to_active_block_weights_only():
    mask = torch.tensor([[True, False]])
    block = BlockSparseSynapse(4, 8, 4, mask, precision="ternary", seed=0)
    spikes = torch.ones(2, 4)
    currents, _ = block(spikes)
    currents.sum().backward()
    assert float(block.weight.grad[0, 0].abs().sum()) > 0
    assert float(block.weight.grad[0, 1].abs().sum()) == 0


def test_edge_weight_gradients_flow_through_spike_path():
    syn = EdgeListSynapse(1, 1, [0], [0], [2.0], precision="float")
    lif = DiffLIF(theta=100, leak_shift=15, refractory_ticks=1,
                  surrogate_width=50.0)
    currents, events = syn(torch.tensor([[1.0]]))
    s = lif.step(currents, events)
    s.sum().backward()
    assert syn.weight.grad is not None and float(syn.weight.grad.abs().sum()) > 0


# --- Learnable per-neuron theta (P2's calibration-floor fix) -----------------

def test_learnable_theta_forward_matches_intlif_on_rounded_grid():
    dlif = DiffLIF(theta=[10.4, 199.6], leak_shift=[3, 15],
                   refractory_ticks=[1, 2], learnable_theta=True)
    ilif = IntLIF(theta=[10, 200], leak_shift=[3, 15], refractory_ticks=[1, 2])
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        I = torch.randint(-50, 250, (1, 2), generator=gen)
        E = torch.rand(1, 2, generator=gen) < 0.8
        sd = dlif.step(I.float(), E)
        si = ilif.step(I, E)
        assert torch.equal(sd.detach(), si.float())
        assert torch.equal(dlif.v.detach(), ilif.v.float())


def test_learnable_theta_receives_gradient_and_stays_on_grid():
    dlif = DiffLIF(theta=50.0, leak_shift=15, refractory_ticks=1,
                   learnable_theta=True, surrogate_width=25.0)
    s = dlif.step(torch.tensor([[40.0]]))
    assert float(dlif.effective_theta().detach()) == 50.0  # integer grid
    s.sum().backward()
    assert dlif.theta.grad is not None and float(dlif.theta.grad) != 0.0
    # raising v toward theta should push d(spike)/d(theta) negative
    assert float(dlif.theta.grad) < 0


def test_learnable_theta_clamps_to_chip_range():
    dlif = DiffLIF(theta=0.2, leak_shift=15, refractory_ticks=1,
                   learnable_theta=True)
    assert float(dlif.effective_theta().detach()) == 1.0  # floor of the grid


# --- Functional: block-sparse ternary net trains the toy task ---------------

def train_block_sparse_ternary(steps: int = 200) -> tuple[float, float]:
    """Returns (accuracy, layer2 flop fraction). Shared by the test and the
    P3 report script."""
    from experiments.p2_diagnostic import N_CLASSES, make_batch

    torch.manual_seed(0)
    bs, hidden = 64, 256
    layer1 = BlockSparseSynapse(64, hidden, bs, torch.ones(1, hidden // bs, dtype=torch.bool),
                                precision="ternary", seed=1)
    mask2 = torch.rand(hidden // bs, hidden // bs) < 0.5  # actually sparse
    mask2[0, 0] = True
    layer2 = BlockSparseSynapse(hidden, hidden, bs, mask2, precision="ternary", seed=2)
    lif1 = DiffLIF(theta=5.0, leak_shift=15, refractory_ticks=1, n_neurons=hidden,
                   learnable_theta=True, surrogate_width=1.25)
    lif2 = DiffLIF(theta=5.0, leak_shift=15, refractory_ticks=1, n_neurons=hidden,
                   learnable_theta=True, surrogate_width=1.25)
    readout = torch.nn.Linear(hidden, N_CLASSES)
    params = (list(layer1.parameters()) + list(layer2.parameters()) +
              list(lif1.parameters()) + list(lif2.parameters()) +
              list(readout.parameters()))
    opt = torch.optim.Adam(params, lr=5e-3)
    gen = torch.Generator().manual_seed(3)

    def forward(x):
        batch, T, _ = x.shape
        lif1.reset_state(batch)
        lif2.reset_state(batch)
        logits = 0.0
        for t in range(T):
            s1 = lif1.step(*layer1(x[:, t]))
            s2 = lif2.step(*layer2(s1))
            logits = logits + readout(s2)
        return logits / T

    for _ in range(steps):
        x, y = make_batch(gen, T=8)
        loss = torch.nn.functional.cross_entropy(forward(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    correct = total = 0
    with torch.no_grad():
        for _ in range(5):
            x, y = make_batch(gen, T=8)
            correct += int((forward(x).argmax(-1) == y).sum())
            total += len(y)
    return correct / total, layer2.flop_fraction()


def test_block_sparse_ternary_network_learns_rate_task():
    accuracy, _ = train_block_sparse_ternary()
    assert accuracy > 0.4, f"block-sparse ternary net failed to learn: {accuracy=}"
