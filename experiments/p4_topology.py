# SPDX-License-Identifier: Apache-2.0
"""P4 — Topology learning and freeze (handoff §9 P4).

Planted-structure task: 128 inputs in two 64-blocks. Block 0 carries the
signal (4 groups of 16; the class's group fires at RATE_HI, others at
RATE_LO). Block 1 is pure noise at RATE_NOISE — informative about nothing.
If L0 gating works, layer 1's gates from the noise block must close: the
learned topology recovers the planted relevance structure, making "this is
the topology the task asked for" a checkable finding, not a slogan.

Pipeline per L0 weight lambda:
  A. train gated net (CE + lambda * sum of per-layer L0), 400 steps;
  B. binarize -> frozen BlockSparseSynapse pair + SHA-256 topology hash;
  C. fine-tune the frozen net 200 steps (weights + thetas re-accommodate).

Baselines at equal total budget (600 steps): dense (all blocks), and
random masks matched to the learned per-layer density (2 seeds).

Usage: .venv/bin/python experiments/p4_topology.py
Writes experiments/results/p4_results.json.
"""

import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celiumsnn import (  # noqa: E402
    BlockSparseSynapse,
    DiffLIF,
    GatedBlockSparseSynapse,
    topology_hash,
)

N_IN, HIDDEN, N_CLASSES, BLOCK = 128, 256, 4, 64
GROUP = 16
RATE_HI, RATE_LO, RATE_NOISE = 0.35, 0.10, 0.20
BATCH, T = 32, 8
STEPS_A, STEPS_B, STEPS_BASE = 800, 200, 1000
LAMBDAS = (0.02, 0.1, 0.3)
GATE_LR, LR = 0.1, 5e-3  # gates need a much hotter lr (L0 literature standard)
LIF_KW = dict(theta=5.0, leak_shift=15, refractory_ticks=1, n_neurons=HIDDEN,
              learnable_theta=True, surrogate_width=1.25)


def make_batch(gen, batch=BATCH):
    y = torch.randint(0, N_CLASSES, (batch,), generator=gen)
    rates = torch.full((batch, T, N_IN), RATE_LO)
    rates[:, :, BLOCK:] = RATE_NOISE
    for b in range(batch):
        rates[b, :, y[b] * GROUP:(y[b] + 1) * GROUP] = RATE_HI
    x = (torch.rand(batch, T, N_IN, generator=gen) < rates).float()
    return x, y


class Net(nn.Module):
    def __init__(self, layer1, layer2):
        super().__init__()
        self.layer1, self.layer2 = layer1, layer2
        self.lif1 = DiffLIF(**LIF_KW)
        self.lif2 = DiffLIF(**LIF_KW)
        self.readout = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x):
        batch = x.shape[0]
        self.lif1.reset_state(batch)
        self.lif2.reset_state(batch)
        logits = 0.0
        for t in range(x.shape[1]):
            s1 = self.lif1.step(*self.layer1(x[:, t]))
            s2 = self.lif2.step(*self.layer2(s1))
            logits = logits + self.readout(s2)
        return logits / x.shape[1]


def run(net, steps, gen, lam=0.0):
    gate_params = [p for n, p in net.named_parameters() if "log_alpha" in n]
    other_params = [p for n, p in net.named_parameters() if "log_alpha" not in n]
    groups = [{"params": other_params, "lr": LR}]
    if gate_params:
        groups.append({"params": gate_params, "lr": GATE_LR})
    opt = torch.optim.Adam(groups)
    net.train()
    for _ in range(steps):
        x, y = make_batch(gen)
        loss = nn.functional.cross_entropy(net(x), y)
        if lam:
            loss = loss + lam * (net.layer1.l0_penalty() + net.layer2.l0_penalty())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def accuracy(net, gen, batches=10):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(gen)
            correct += int((net(x).argmax(-1) == y).sum())
            total += len(y)
    return correct / total


def random_mask(shape, n_open, gen):
    flat = torch.zeros(shape.numel() if hasattr(shape, "numel") else shape[0] * shape[1])
    idx = torch.randperm(flat.numel(), generator=gen)[:n_open]
    flat[idx] = 1
    return flat.reshape(shape).bool()


def run_lambda(lam: float) -> dict:
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(100)
    net = Net(GatedBlockSparseSynapse(N_IN, HIDDEN, BLOCK, seed=1),
              GatedBlockSparseSynapse(HIDDEN, HIDDEN, BLOCK, seed=2))
    run(net, STEPS_A, gen, lam=lam)
    acc_gated = accuracy(net, gen)

    g1 = net.layer1.gates.deterministic().detach()
    g2 = net.layer2.gates.deterministic().detach()
    frozen = Net(net.layer1.freeze(), net.layer2.freeze())
    frozen.lif1.load_state_dict(net.lif1.state_dict())
    frozen.lif2.load_state_dict(net.lif2.state_dict())
    frozen.readout.load_state_dict(net.readout.state_dict())
    m1, m2 = frozen.layer1.block_mask, frozen.layer2.block_mask
    run(frozen, STEPS_B, gen)
    acc_frozen = accuracy(frozen, gen)

    entry = {
        "acc_gated": acc_gated, "acc_frozen_finetuned": acc_frozen,
        "gates_layer1": g1.tolist(), "gates_layer2": g2.tolist(),
        "mask_layer1": m1.tolist(), "mask_layer2": m2.tolist(),
        "flop_fraction": {"layer1": frozen.layer1.flop_fraction(),
                          "layer2": frozen.layer2.flop_fraction()},
        "noise_block_fully_pruned": bool((~m1[1]).all()),
        "signal_block_open_count": int(m1[0].sum()),
        "topology_sha256": topology_hash(m1, m2),
    }
    print(f"lambda={lam}: gated {acc_gated:.3f} -> frozen+ft {acc_frozen:.3f}, "
          f"flops L1 {entry['flop_fraction']['layer1']:.2f} "
          f"L2 {entry['flop_fraction']['layer2']:.2f}, "
          f"noise pruned={entry['noise_block_fully_pruned']}")
    print(f"  mask1={m1.tolist()} mask2={m2.tolist()}")
    print(f"  hash={entry['topology_sha256'][:16]}…")
    return entry


def run_dense() -> float:
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(100)
    dense = Net(BlockSparseSynapse(N_IN, HIDDEN, BLOCK, torch.ones(2, 4, dtype=torch.bool), seed=1),
                BlockSparseSynapse(HIDDEN, HIDDEN, BLOCK, torch.ones(4, 4, dtype=torch.bool), seed=2))
    run(dense, STEPS_BASE, gen)
    acc = accuracy(dense, gen)
    print(f"dense baseline: {acc:.3f}")
    return acc


def run_random(n1: int, n2: int) -> list:
    rand_accs = []
    for s in (7, 8, 9, 10):
        mgen = torch.Generator().manual_seed(s)
        torch.manual_seed(42)
        gen = torch.Generator().manual_seed(100)
        rnet = Net(BlockSparseSynapse(N_IN, HIDDEN, BLOCK, random_mask((2, 4), n1, mgen), seed=1),
                   BlockSparseSynapse(HIDDEN, HIDDEN, BLOCK, random_mask((4, 4), n2, mgen), seed=2))
        run(rnet, STEPS_BASE, gen)
        rand_accs.append(accuracy(rnet, gen))
    print(f"random-mask baselines (n1={n1}, n2={n2}): {rand_accs}")
    return rand_accs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=None,
                        help="lam-<value> | dense | random (default: full sequential run)")
    parser.add_argument("--n1", type=int, default=None)
    parser.add_argument("--n2", type=int, default=None)
    args = parser.parse_args()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    if args.part:  # one component, own JSON — parallelize across processes
        if args.part.startswith("lam-"):
            payload = run_lambda(float(args.part[4:]))
        elif args.part == "dense":
            payload = run_dense()
        elif args.part == "random":
            payload = run_random(args.n1, args.n2)
        else:
            raise SystemExit(f"unknown part {args.part}")
        path = out / f"p4_part_{args.part}.json"
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"written: {path}")
        return

    results = {"lambdas": {}, "baselines": {}}
    for lam in LAMBDAS:
        results["lambdas"][str(lam)] = run_lambda(lam)
    results["baselines"]["dense"] = run_dense()
    ref = results["lambdas"][str(LAMBDAS[1])]
    n1 = int(sum(sum(r) for r in ref["mask_layer1"]))
    n2 = int(sum(sum(r) for r in ref["mask_layer2"]))
    results["baselines"]["random_same_density"] = run_random(n1, n2)
    with open(out / "p4_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"written: {out / 'p4_results.json'}")


if __name__ == "__main__":
    main()
