# P4 — Topology Learning and Freeze Report

**Date:** 2026-08-18 · **Code:** `experiments/p4_topology.py` · **Data:** `experiments/results/p4_results.json`
**Compute:** DigitalOcean `c-60-intel` droplet `celiumsnn-p4` (60 vCPU, provisioned mid-run after the local sweep saturated the workstation; parts run as parallel processes).

## Setup — a task with planted structure

128 inputs in two 64-blocks: block 0 carries the 4-class rate signal, block 1 is
pure noise (rate 0.20, informative about nothing). Network: gated block-sparse
128→256→256 (ternary weights, learnable per-neuron θ, atan surrogate), dense
readout. Pipeline per L0 weight λ: **(A)** 800 steps with stochastic
hard-concrete gates + L0 penalty (gate lr 0.1, weight lr 5e-3), **(B)**
binarize → frozen `BlockSparseSynapse` + SHA-256, **(C)** 200 steps fine-tune
with the graph fixed. Baselines at equal 1000-step budget.

Planting the structure converts "the model got sparser" into a checkable
claim: the gates from the noise block must close.

## Results

| Config | Acc (gated) | Acc (frozen+FT) | FLOPs L1 | FLOPs L2 | Noise block pruned |
|---|---|---|---|---|---|
| λ = 0.02 | 1.000 | 1.000 | 0.88 | 0.62 | no (partial) |
| **λ = 0.1** | **1.000** | **1.000** | **0.50** | **0.125** | **yes — exactly** |
| λ = 0.3 | 1.000 | 1.000 | 0.38 | 0.06 | yes |
| dense baseline | — | 1.000 | 1.00 | 1.00 | — |
| random masks @ λ=0.1 density (4 seeds) | — | 1.0 / 0.225 / 0.244 / 1.0 | 0.50 | 0.125 | — |

**λ = 0.1 is the headline row.** The learned layer-1 mask is
`[[T,T,T,T],[F,F,F,F]]` — the four signal-block routes kept, the four
noise-block routes all closed. The planted relevance structure was recovered
exactly, at zero accuracy cost, with layer 2 at **12.5% of dense FLOPs**
(2 of 16 blocks). Frozen artifact:
`sha256: 029f8db749fab0deca236e534e3e14c7d77ffeea416ef7c2250485ca626d7fb7`
— the signable, input-independent routing graph of the verifiability thesis
(handoff §5).

**Random topology at the same density is a coin flip:** 2 of 4 random masks
train to 1.0, the other 2 collapse to chance (~0.23) because their routing
severs the signal path. The learned topology is not just sparse — it is
*reliably placed*. That is the minimal form of "topology as a result":
at equal density and budget, learned ≫ random in reliability, and the
learned mask is interpretable against the planted ground truth.

**Freeze costs nothing here:** accuracy is identical before binarization and
after freeze+fine-tune at every λ (the fine-tune stage exists to let weights
re-accommodate to gate scale 1 and did its job — no residual gap).

## The failed first attempt (kept for the record)

The initial run (gate lr = weight lr = 5e-3, 400 steps, λ ≤ 0.08) moved gates
by < 0.1 and pruned nothing — not because L0 gating fails, but because Adam
at lr 5e-3 cannot move `log_alpha` the ~4 units needed to close a gate in 400
steps. Direction was already correct (noise gates uniformly lower). Fix:
separate optimizer group for gates at lr 0.1 — the standard L0-literature
setting. Diagnosis took one gate-value inspection; the planted structure made
the failure visible immediately.

## Honest limits

Toy task where dense also reaches 1.0 — this experiment demonstrates the
**mechanism** (learn → binarize → freeze → no quality loss, structure
recovered, artifact hashable), not a quality-per-FLOP win; that curve is P5's
job on a real task (D6). λ was swept, not tuned; 4 random seeds; single run
per λ. Layer-2's surviving blocks are not unique — different λ found different
valid routings (hashes differ), so "the" topology is per-run, and the
signable claim is about a *trained artifact*, not a task invariant.

## Next

P5 — task, baseline, curve: quality-per-byte and quality-per-operation
against a dense fp16 equivalent under identical budget (D6 decides the task;
SHD-class keyword spotting was the standing recommendation).
