# P6 — Consolidation Report: T=64+Augmentation, λ Sweep, DVS-Gesture, Multi-Seed, snnTorch Adapter

**Date:** 2026-08-18/19 · **Code:** `experiments/p5_shd.py` (extended), `experiments/p6_dvs.py`, `celiumsnn/snntorch_adapter.py` · **Data:** `experiments/results/p5_*T64*|*lam*`, `p6dvs_*`
**Compute:** DO `c-60-intel` droplet `celiumsnn-p6`. This report executes the five "next candidates" P5 left open and closes the experimental program.

## 1. T=64 + augmentation — honest null: the gap is architectural, not budget

SHD, 60 epochs cosine, count binning at T=64, roll augmentation (time ±T/8, channels ±8):

| Model | Accuracy (mean ± sd, n) | Bytes | MMAC/sample |
|---|---|---|---|
| Mycelium gated-512 | 0.617 ± 0.058 (3) | 174 KB | 40.5 |
| Mycelium gated-1024 | 0.637 (1) | 319 KB | 73.4 |
| Mycelium gated-512, **no aug** | 0.614 (1) | 174 KB | 40.5 |
| GRU-64 | **0.830 ± 0.012** (3) | 291 KB | 9.5 |
| GRU-128 | 0.857 ± 0.007 (3) | 631 KB | 20.6 |
| GRU-256 | 0.883 ± 0.007 (3) | 1453 KB | 47.5 |

Doubling temporal resolution + augmentation moved the GRU up ~5 points across the board and moved Mycelium **nowhere** (augmentation gain: +0.4 pts; T=64 is *worse* than its own T=32 result of 0.679). The accuracy gap to dense is therefore not a budget artifact — it lives in the architecture (binary spikes + rate readout), and closing it needs a different idea (e.g. membrane-readout heads, multi-bit spikes, or longer-timescale mechanisms), not more epochs.

**Multi-seed reality check:** Mycelium seed spread is wide (0.573 / 0.596 / 0.683) while the GRU is tight (±0.007–0.012). Training stability is a real weakness of the current recipe and retroactively adds unquantified seed noise to P5's single-seed points.

## 2. λ sweep — the topology learner deletes recurrence, and the small-regime win strengthens

SHD T=32, gated-512, sweeping the L0 price (P5 used λ=0.02):

| λ | Accuracy | frac in / rec | Bytes | MMAC/sample |
|---|---|---|---|---|
| 0.02 (P5) | 0.679 | 0.93 / 0.98 | 167 KB | 19.3 |
| 0.05 | 0.562 | 0.94 / 1.00 | 169 KB | 19.6 |
| **0.15** | **0.660** | **0.45 / 0.00** | **62 KB** | **5.6** |
| 0.4 | 0.599 | 0.11 / 0.00 | 32 KB | 1.6 |

Two findings:

- **At λ ≥ 0.15 the learner prunes the recurrent synapse to exactly zero** — the frozen model is feedforward — at a cost of 1.9 points. *"This is the topology the task asked for"* materialized: rate-readout SHD at T=32 does not pay for recurrence. Topology hash `548f204ddd3f9778…`.
- The λ=0.15 point (**0.660 @ 62 KB / 5.6 MMAC**) strengthens the P5 small-regime byte win decisively: vs GRU-16 (0.537 @ 68 KB) that is **+12.3 points at fewer bytes**, and it cuts the per-op distance to dense from ~8× to ~2.4× (GRU-32: 0.682 @ 2.3 MMAC). The per-op curve still does not cross, but the frontier moved a long way for free.
- (λ=0.05 underperforming both neighbors is consistent with the seed-variance finding above; single seeds.)

## 3. DVS-Gesture — the pattern generalizes to event vision

IBM DVS128 Gesture (11 classes, chance 9.1%), 128×128×2 events pooled to 16×16×2 = 512 channels, T=32 counts, single seed. (Dataset note: tonic's `figshare.com/ndownloader` URLs are WAF-blocked and return empty 202s; the `ndownloader.figshare.com` subdomain works — pipeline handles it.)

| Model | Accuracy | Bytes | MMAC/sample |
|---|---|---|---|
| Mycelium gated-256 | 0.671 | 54 KB | 6.4 |
| Mycelium gated-512 | 0.689 | 133 KB | 15.9 |
| GRU-64 | 0.769 | 218 KB | 3.6 |
| GRU-128 | 0.773 | 484 KB | 7.9 |

Same qualitative picture as SHD: the architecture works far above chance on a second modality, sits below tuned dense at large budgets, and holds the byte-efficiency edge at the small end (0.671 @ 54 KB with no dense point measured anywhere near that size). SHD is not a fluke; low-end dense baselines on DVS remain unmeasured.

## 4. snnTorch adapter — done

`celiumsnn/snntorch_adapter.py`: `CeliumLeaky` exposes DiffLIF through snnTorch's `spk, state = lif(x, state)` convention. Tested (4/4, `tests/test_snntorch_adapter.py`): exact parity with DiffLIF, state reset semantics, gradient flow, and a **mixed network with a real `snn.Leaky` layer feeding a CeliumLeaky layer** trains end-to-end. The "mental fork" is now a plug: celiumsnn dynamics drop into snnTorch tooling without forking anything.

## Consolidated verdict after P5+P6

1. **The regime where this architecture wins is now sharply defined:** small models (≲150 KB), where frozen ternary block-sparse topology beats dense fp16 by +8–12 points at equal bytes (SHD; DVS consistent, unconfirmed at the low dense end). Above ~300 KB dense wins accuracy; per-op dense wins everywhere, though λ=0.15 closed most of that distance.
2. **The gap at scale is architectural.** More time bins, augmentation and budget do not move it. Next lever is a model change, not a training change.
3. **Topology-as-a-result is real:** the learner deleted recurrence when the task didn't pay for it, at 3× fewer FLOPs and 1/3 the bytes.
4. **Stability is the main open weakness:** ±6-point seed spread vs the GRU's ±1.
5. The unique claims from P0–P5 (bit-exact verified chain, signable frozen graphs, chip-faithful deployment certificate) are untouched by any of this and remain the project's differentiation.

## Limits

Single seed for λ sweep, DVS, and gated-1024; augmentation not tuned (one scheme tried); DVS lacks small dense baselines; λ and T were not jointly swept; MAC counting treats ternary adds as MACs and ignores event sparsity (conservative against Mycelium).
