# P7 — Stability, Membrane Readout, Writeup

**Date:** 2026-08-19 · **Code:** `celiumsnn/model.py` (readout_mode), `experiments/p5_shd.py` (--readout/--clip) · **Data:** `experiments/results/p5_*p7*-*.json` · **Compute:** DO `c-60-intel` `celiumsnn-p7` (destroyed after).
All results: SHD, T=32, gated→freeze→fine-tune, 60 epochs cosine, **3 seeds everywhere** (both families — the first fully multi-seed table of the project).

## 1. Membrane readout — better AND more stable

New head option: logits read the top layer's membrane potential in threshold units (`v/θ`) instead of spikes. Chip-honest as a host-side layer: the membrane is readable through the chip's non-invasive `rb_*` sideband (invariant I5). Gradient reaches every tick through the membrane recursion, no surrogate on the readout path (3.6× more gradient at init).

| gated-512, T=32 | mean ± sd (3 seeds) |
|---|---|
| spike readout, lr 1e-2 | 0.648 ± 0.029 |
| spike readout, lr 5e-3 | 0.629 ± 0.031 |
| **membrane readout, lr 1e-2** | **0.673 ± 0.019** |
| membrane readout, lr 5e-3 | 0.643 ± 0.038 |
| spike + grad-clip 5.0 | 0.544 ± 0.025 |
| spike + grad-clip 1.0 (first attempt) | 0.417 ± 0.049 |

Findings: **(a)** membrane beats spike by +2.6 points at the same budget and halves the seed spread; **(b)** gradient clipping *hurts* at every tested strength — this model's healthy gradient norms are large, and clipping them is the wrong stabilizer; **(c)** P5's single-seed 0.679 for spike was a lucky seed (honest multi-seed value: 0.648 ± 0.029).

## 2. Stability, reframed by data

P6's alarm (±5.8 at T=64) shrinks to ±2.9 at T=32 for spike and **±1.9 for membrane**. Meanwhile the multi-seeded *small dense baselines* turn out equally noisy (GRU-16 ± 4.0, GRU-32 ± 3.3) — the tight ±0.7–1.2 GRUs of P6 were the large ones. Conclusion: at small scale, seed noise is a property of the regime, not a Mycelium defect; with the membrane head, Mycelium is the *most* stable model in its size class in this table.

## 3. The consolidated frontier (all points 3 seeds, SHD T=32)

| Model | Accuracy | Bytes | MMAC/sample |
|---|---|---|---|
| **Mycelium λ=0.15, membrane** | **0.623 ± 0.030** | **31 KB** | **1.5** |
| GRU-16 fp16 | 0.554 ± 0.040 | 68 KB | 1.1 |
| GRU-32 fp16 | 0.669 ± 0.033 | 140 KB | 2.3 |
| Mycelium λ=0.02, membrane | 0.673 ± 0.019 | 167 KB | 19.3 |
| Mycelium 1024, membrane | 0.650 ± 0.021 | 284 KB | 32.1 |

- **Bytes:** the 31 KB point beats GRU-16 by **+6.9 points with less than half the bytes** — the membrane head let λ=0.15 prune even harder than in P6 (input fraction 0.10, recurrence 0.00, 62 KB → 31 KB) at slightly lower accuracy. At ~150 KB the families tie; larger Mycelium (284 KB) does not beat its own 167 KB point.
- **Operations — a first, modest crossing:** at 1.5 MMAC Mycelium scores 0.623; the dense frontier interpolated between GRU-16 (1.1 MMAC, 0.554) and GRU-32 (2.3 MMAC, 0.669) passes ≈0.59 there. In the extreme-edge corner the per-op frontier now also favors Mycelium by ~3 points. Above that corner, dense still wins per-op decisively.
- Note 1024 ± seeds (0.650 ± 0.021) confirms P5's 0.713 single-seed was optimistic; scale does not currently pay past 512 with this recipe.

## 4. Writeup

`WRITEUP.md` (paper draft v0.2): abstract, thesis, constraint set, C1–C5 contract, Mycelium, all results P2→P7 with the multi-seed table above, related work, limitations, and the recommended lead claim — *first task-trained SNN with a bit-exact verification chain to silicon-verified RTL, plus the memory-frontier win in the edge regime.*

## Limits

3 seeds per point (enough for honest ±, not for significance theater — the 31 KB gap is ~2σ); GRU baselines not retuned per size beyond the lr pair; λ and readout not jointly swept beyond {0.02, 0.15}; DVS not rerun with the membrane head; wall-clock GPU kernels still unmeasured (FLOP fractions stand in).

## Recipe going forward (locked)

Membrane readout, lr 1e-2 cosine, no gradient clipping, dropout 0.25, θ₀=48, diverse per-neuron leak, ternary, T=32; λ = 0.02 for max accuracy, 0.15 for the edge point.
