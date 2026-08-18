# P5 — Task, Baseline, Curve: Final Report

**Date:** 2026-08-18 · **Code:** `experiments/p5_shd.py`, `celiumsnn/model.py` · **Data:** `experiments/results/p5_*.json`
**Compute:** DO `c-60-intel` droplet, runs as parallel processes.
**Task (D6):** Spiking Heidelberg Digits (SHD), 20 spoken digits, 700 spike channels → counts binned to **T=32** (D7: integer count injection = the chip's event-multiplicity semantics; D4: T=32, inside P2's validated range).
**Model (D2):** `Mycelium` — recurrent module over a static block graph: block-sparse ternary feed + recurrent synapses (P4 L0-learned, frozen), DiffLIF with learnable per-neuron θ and heterogeneous per-neuron leak (I7), spike-average readout. No attention, no data-dependent routing.
**Baseline discipline:** GRU (fp16-deployed) trained with the same budget: 60 epochs cosine, lr swept {1e-3, 3e-3} per size; Mycelium's config was locked from its own sweep (θ₀=48, dropout 0.25, lr 1e-2, cosine 60; ternary; single layer).

## Headline result

**The quality-per-byte curve crosses.** In the small-model regime the block-sparse ternary SNN dominates the dense GRU at equal deployment size; the GRU takes over above ~300 KB:

| Deployment bytes | Mycelium (gated→frozen) | GRU fp16 | Winner |
|---|---|---|---|
| ~70 KB | **62.3%** (gated-256, 70 KB) | 53.7% (GRU-16, 68 KB) | **Mycelium +8.6 pts** |
| ~140–170 KB | 67.9% (gated-512, 167 KB) | 68.2% (GRU-32, 140 KB) | ≈ tie |
| ~290–440 KB | 71.3% (gated-1024, 441 KB) | 73.2% (GRU-64, 291 KB) | GRU |
| ~630 KB | — | 79.9% (GRU-128) | GRU |
| ~1.45 MB | — | 82.9% (GRU-256) | GRU |

Per the handoff's own framing this is the composite outcome: the byte curve **crosses in the ≤150 KB regime**, and above it the result is the second publishable kind — within ~2–5 points of dense **with a frozen, signable, integer-only compute graph** (topology SHA-256 per run, e.g. gated-1024: see `p5_myc-gated-1024-final.json`).

**Quality-per-operation does not cross:** the GRU reaches equal accuracy with fewer MACs at every scale (e.g. 73% at 4.8 MMAC/sample vs Mycelium's 71% at 53 MMAC). Two caveats recorded, not claimed as wins: ternary "MACs" are additions (no multiplies), and event-driven sparsity (hidden rate ~0.4) is left on the table by dense-block execution — the §6 GPU floor, as predicted.

## Findings along the way

1. **Quantization is not the cost — the float ablation LOST to ternary** (54.5% float vs 61.2% ternary, same dynamics, same budget). Consistent with P2: the constraint set is gradient-friendly; the gap to GRU is architectural (binary spikes, rate readout), not precision.
2. **Topology learning beats dense blocks at every scale** — gated→freeze→fine-tune outperformed all-blocks-active: 62.3 vs 61.5 (256), 67.9 vs 64.6 (512), **71.3 vs 62.2 (1024)**. At λ=0.02 it pruned little (fractions 0.90–0.98), so the win comes mostly from the two-phase optimization itself. (Confound noted: the gated pipeline's flat-then-cosine schedule differs from the full run's; not isolated.)
3. **Diagnosis history:** the first Mycelium runs sat at ~50% from *underfitting* (train ≈ test) — fixed by lr 1e-2 (+10 pts) and mild spike dropout, not by θ placement (rates were healthy 0.35–0.48), not by leak diversity alone, not by int8, not by two layers (2-layer stacking hurt: 38.6%).
4. **Context:** literature recurrent-LIF baselines on SHD sit ≈71% (Cramer et al.), typically at T≈100+; Mycelium's 71.3% at T=32 with ternary weights and a frozen graph is in family, while our tuned GRU (82.9%) matches LSTM-class numbers.

## The chip-faithful certificate (D1-c)

2-class SHD (channels pooled to 64), network of **34 neurons and 832 dendrite entries** — inside the v1 silicon budget (≤1,024 / ≤1,024) — trained with the full pipeline, readout = output-neuron spike counts (chip-native):

- Test accuracy: **74.9%** (chance 50%), DiffLIF deploy weights.
- **IntLIF replay: 74.9%, 0 logit mismatches over the full test set** — the trained model, run through the integer model that is bit-exact against the chip's golden model (P1), produces identical spike counts everywhere.
- Artifact hash (edges + int8 weights + integer thresholds): `4710 0390 8edb 3f1e…` (full value in `p5_reference.json`).

This is the loop the handoff said no SNN paper closes: **task-trained model ≡ integer model ≡ golden model ≡ verified RTL**, under the C1–C5 contract, with a hashable artifact at the end.

## Honest limits

Single seed per point; T=32 only (T=64+ untried — likely helps both families); no data augmentation (SHD SOTA uses it heavily); GRU got a 2-value lr sweep vs Mycelium's multi-round tuning (asymmetry favors Mycelium in tuning effort — though GRU's numbers landed at literature level, suggesting it wasn't left undercooked); L0 λ untuned on SHD (λ=0.02 barely prunes; the per-op curve would need real pruning pressure plus event-driven execution to move); bytes counted at 2 bits/ternary weight (1.58 information-theoretic would shift Mycelium's curve ~20% left); training cost is higher than dense (BPTT stores T steps) — efficiency claims are inference-side only, per handoff §6.

## Verdict

- **D6/D7/D4 closed:** SHD, integer count injection, T=32.
- **The thesis held where the handoff predicted it would:** the memory/bytes axis pays today (curve crosses in the small regime), the compute axis does not without hardware or real event-driven execution.
- **The verifiability claim is now demonstrated end-to-end**, not just argued: a task-trained instance inside silicon budget, bit-exact to the verified RTL's referee, with a signed artifact.

**Next candidates (not started):** T=64 + augmentation to close the accuracy gap; λ sweep on SHD for real FLOP reduction; the snnTorch data-tooling adapter; a second task (DVS-Gesture) for generality; multi-seed error bars before any writeup.
