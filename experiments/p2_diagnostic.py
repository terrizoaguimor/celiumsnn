# SPDX-License-Identifier: AGPL-3.0-or-later
"""P2 — Gradient diagnostic (handoff §9 P2, risks R1/R2; decides D3/D5).

Measures, over a sweep of surrogate shape x width x weight precision x
temporal depth T x leak k x threshold placement:

  - where the membrane lives relative to threshold (R1's direct question);
  - fraction of comparator evaluations inside the surrogate support;
  - input-layer gradient magnitude, absolute and relative to a float-LIF
    control with the same seed, shape, width and T (isolates the cost of
    the integer/ceil dynamics);
  - per-tick gradient profile of BPTT (R2: temporal credit assignment).

Then trains the best config per precision on a 4-class rate-discrimination
task for a functional check.

KILL CRITERIA — pre-registered before any run (see P2-REPORT.md):
  K1: best support fraction across the whole sweep < 0.05  -> STOP.
  K2: for a quantized precision, best relative-gradient ratio vs matched
      float control < 1e-3                                  -> STOP for it.
  K3: best (t=0 / t=T-1) per-tick gradient ratio at T=8 < 1e-4
      -> no temporal credit assignment (bounds D4; kill if K1/K2 marginal).
  F:  best config must beat chance (25%) by wide margin (>40% accuracy)
      within 300 Adam steps on the toy task.

Usage: .venv/bin/python experiments/p2_diagnostic.py [--quick]
Writes experiments/results/p2_results.json.
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celiumsnn import DiffLIF, QUANTIZERS, surrogate_kernel  # noqa: E402

N_IN, HIDDEN, N_CLASSES = 64, 128, 4
GROUP = N_IN // N_CLASSES
RATE_HI, RATE_LO = 0.35, 0.10
BATCH = 32
CALIB_BATCHES = 4
SIGMA = {"float": 1.0, "int8": 6.0, "ternary": 1.0}

SHAPES = ("atan", "fast_sigmoid", "triangular")
WIDTH_RELS = (0.05, 0.25, 1.0)
PRECISIONS = ("float", "int8", "ternary")
TS = (8, 32)
KS = (2, 15)
QUANTILES = (0.5, 0.8, 0.95)


def make_batch(gen, batch=BATCH, T=8):
    y = torch.randint(0, N_CLASSES, (batch,), generator=gen)
    rates = torch.full((batch, T, N_IN), RATE_LO)
    for b in range(batch):
        rates[b, :, y[b] * GROUP:(y[b] + 1) * GROUP] = RATE_HI
    x = (torch.rand(batch, T, N_IN, generator=gen) < rates).float()
    return x, y


class SpikeNet(nn.Module):
    """input spikes -> W1 -> DiffLIF -> W2 -> DiffLIF -> mean-over-T readout."""

    def __init__(self, precision, shape, k, thetas=(32767.0, 32767.0),
                 width_rel=0.25, seed=0):
        super().__init__()
        self.precision = precision
        self.quant = QUANTIZERS[precision]
        gen = torch.Generator().manual_seed(seed)
        s = SIGMA[precision]
        self.w1 = nn.Parameter(torch.randn(N_IN, HIDDEN, generator=gen) * s)
        self.w2 = nn.Parameter(torch.randn(HIDDEN, HIDDEN, generator=gen) * s)
        self.readout = nn.Linear(HIDDEN, N_CLASSES)
        integer_exact = precision != "float"
        self.widths = [max(width_rel * t, 1e-3) for t in thetas]
        self.lifs = nn.ModuleList([
            DiffLIF(theta=thetas[i], leak_shift=k, refractory_ticks=1,
                    surrogate_shape=shape, surrogate_width=self.widths[i],
                    integer_exact=integer_exact, n_neurons=HIDDEN)
            for i in range(2)
        ])

    def forward(self, x, record=None):
        batch, T, _ = x.shape
        for lif in self.lifs:
            lif.reset_state(batch)
        logits = 0.0
        for t in range(T):
            s = x[:, t]
            tick = {"t": t}
            for li, lif in enumerate(self.lifs):
                w_q = self.quant(self.w1 if li == 0 else self.w2)
                events = (s.detach() > 0.5).any(-1, keepdim=True)
                s = lif.step(s @ w_q, events)
                if record is not None:
                    u = (lif.last_v_evt - lif.theta) / lif.surrogate_width
                    tick[f"u{li}"] = u
                    tick[f"rate{li}"] = float(s.detach().mean())
                    if li == 0 and s.requires_grad:
                        s.retain_grad()
                        tick["s0"] = s
            logits = logits + self.readout(s)
            if record is not None:
                record.append(tick)
        return logits / T


def calibrate_thetas(precision, k, q, T, seed):
    """Layer-by-layer threshold placement from the measured membrane
    distribution (theta = q-quantile of the event-comparator input)."""
    thetas = [32767.0, 32767.0]
    for layer in range(2):
        net = SpikeNet(precision, "atan", k, tuple(thetas), seed=seed)
        gen = torch.Generator().manual_seed(9000 + seed)
        samples = []
        with torch.no_grad():
            for _ in range(CALIB_BATCHES):
                x, _ = make_batch(gen, T=T)
                rec = []
                net(x, record=rec)
                samples.append(torch.cat(
                    [tk[f"u{layer}"].flatten() for tk in rec]))
        v = torch.cat(samples) * net.widths[layer] + thetas[layer]  # undo u-scaling
        theta = float(torch.quantile(v, q))
        if precision != "float":
            theta = max(1.0, round(theta))
        else:
            theta = max(1e-3, theta)
        thetas[layer] = theta
    return tuple(thetas)


def measure(config, thetas, seed=0, n_batches=2):
    precision, shape, width_rel, T, k, q = config
    net = SpikeNet(precision, shape, k, thetas, width_rel, seed=seed)
    gen = torch.Generator().manual_seed(4000 + seed)
    agg = {"support": [], "kernel": [], "rate0": [], "rate1": [],
           "u_mean": [], "u_std": [], "grad_w1": [], "grad_w1_rel": [],
           "tick_ratio": []}
    for _ in range(n_batches):
        net.zero_grad()
        x, y = make_batch(gen, T=T)
        rec = []
        logits = net(x, record=rec)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()

        u_all = torch.cat([tk["u0"].flatten() for tk in rec] +
                          [tk["u1"].flatten() for tk in rec])
        agg["support"].append(float((u_all.abs() <= 1).float().mean()))
        agg["kernel"].append(float(surrogate_kernel(u_all, shape).mean()))
        agg["u_mean"].append(float(u_all.mean()))
        agg["u_std"].append(float(u_all.std()))
        agg["rate0"].append(sum(tk["rate0"] for tk in rec) / len(rec))
        agg["rate1"].append(sum(tk["rate1"] for tk in rec) / len(rec))

        g = net.w1.grad
        w_eff = net.quant(net.w1).detach()
        gnorm = float(g.norm()) if g is not None else 0.0
        agg["grad_w1"].append(gnorm)
        agg["grad_w1_rel"].append(gnorm / max(float(w_eff.norm()), 1e-12))

        tick_grads = [float(tk["s0"].grad.norm()) if tk["s0"].grad is not None
                      else 0.0 for tk in rec]
        first, last = tick_grads[0], tick_grads[-1]
        agg["tick_ratio"].append(first / last if last > 0 else 0.0)
        agg["tick_grads"] = tick_grads  # keep last batch's profile

    out = {key: (sum(vals) / len(vals) if key != "tick_grads" else vals)
           for key, vals in agg.items()}
    out["thetas"] = list(thetas)
    return out


def train_check(config, thetas, steps=300, seed=0):
    precision, shape, width_rel, T, k, q = config
    net = SpikeNet(precision, shape, k, thetas, width_rel, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(7000 + seed)
    losses = []
    for step in range(steps):
        x, y = make_batch(gen, T=T)
        loss = nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    correct = total = 0
    with torch.no_grad():
        for _ in range(10):
            x, y = make_batch(gen, T=T)
            correct += int((net(x).argmax(-1) == y).sum())
            total += len(y)
    return {"loss_first20": sum(losses[:20]) / 20,
            "loss_last20": sum(losses[-20:]) / 20,
            "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)

    shapes = SHAPES[:1] if args.quick else SHAPES
    width_rels = WIDTH_RELS[1:2] if args.quick else WIDTH_RELS
    ts = TS[:1] if args.quick else TS
    ks = KS if not args.quick else KS[:1]
    qs = QUANTILES[1:2] if args.quick else QUANTILES

    theta_cache = {}
    results = []
    combos = list(itertools.product(PRECISIONS, shapes, width_rels, ts, ks, qs))
    print(f"sweep: {len(combos)} configs")
    for i, config in enumerate(combos):
        precision, shape, width_rel, T, k, q = config
        key = (precision, k, q, T)
        if key not in theta_cache:
            theta_cache[key] = calibrate_thetas(precision, k, q, T, seed=0)
        met = measure(config, theta_cache[key])
        met["config"] = {"precision": precision, "shape": shape,
                         "width_rel": width_rel, "T": T, "k": k, "q": q}
        results.append(met)
        if (i + 1) % 27 == 0:
            print(f"  {i+1}/{len(combos)}")

    # Relative-to-control ratios: match each quantized config to the float
    # config with the same (shape, width_rel, T, k, q).
    by_key = {}
    for r in results:
        c = r["config"]
        by_key[(c["precision"], c["shape"], c["width_rel"], c["T"], c["k"], c["q"])] = r
    for r in results:
        c = r["config"]
        if c["precision"] == "float":
            r["ratio_vs_float"] = 1.0
            continue
        ctrl = by_key.get(("float", c["shape"], c["width_rel"], c["T"], c["k"], c["q"]))
        r["ratio_vs_float"] = (r["grad_w1_rel"] / ctrl["grad_w1_rel"]
                               if ctrl and ctrl["grad_w1_rel"] > 0 else 0.0)

    # Best config per precision by (support x relative gradient), then train.
    def score(r):
        return r["grad_w1_rel"] * (r["support"] + 1e-6)

    best, training = {}, {}
    for precision in PRECISIONS:
        cands = [r for r in results if r["config"]["precision"] == precision
                 and r["config"]["T"] == min(ts)]
        best[precision] = max(cands, key=score)
        c = best[precision]["config"]
        config = (c["precision"], c["shape"], c["width_rel"], c["T"], c["k"], c["q"])
        print(f"training best {precision}: {c}")
        training[precision] = train_check(config, tuple(best[precision]["thetas"]))
        print(f"  -> {training[precision]}")

    # Kill-criteria evaluation.
    k1_best = max(r["support"] for r in results)
    k2 = {p: max((r["ratio_vs_float"] for r in results
                  if r["config"]["precision"] == p), default=0.0)
          for p in ("int8", "ternary")}
    k3_best = max((r["tick_ratio"] for r in results if r["config"]["T"] == min(ts)),
                  default=0.0)
    verdict = {
        "K1_best_support": k1_best, "K1_fail": k1_best < 0.05,
        "K2_best_ratio": k2, "K2_fail": {p: v < 1e-3 for p, v in k2.items()},
        "K3_best_tick_ratio_T8": k3_best, "K3_fail": k3_best < 1e-4,
        "F_accuracy": {p: t["accuracy"] for p, t in training.items()},
        "F_fail": {p: t["accuracy"] <= 0.40 for p, t in training.items()},
    }
    print("\nVERDICT:", json.dumps(verdict, indent=2))

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "p2_results.json", "w") as fh:
        json.dump({"results": results,
                   "best": {p: r["config"] for p, r in best.items()},
                   "best_metrics": {p: {k2_: v for k2_, v in r.items() if k2_ != "config"}
                                    for p, r in best.items()},
                   "training": training, "verdict": verdict}, fh, indent=1)
    print(f"written: {out / 'p2_results.json'}")


if __name__ == "__main__":
    main()
