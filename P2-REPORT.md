# P2 — Gradient Diagnostic Report

**Date:** 2026-08-18 · **Code:** `experiments/p2_diagnostic.py` · **Data:** `experiments/results/p2_results.json`
**Sweep:** 324 configs = 3 surrogate shapes × 3 widths × 3 weight precisions × T ∈ {8, 32} × leak k ∈ {2, 15} × 3 threshold placements (θ calibrated per layer from the measured membrane distribution at quantiles {0.5, 0.8, 0.95}).
**Network:** 64 → 128 → 128 → 4-class readout, `DiffLIF` dynamics (forward bit-identical to `IntLIF`, hence to golden/RTL), dense weights, R = 1, subtractive reset, B = 32.
**Task (functional check):** 4-class rate discrimination, 300 Adam steps.

## Verdict — GATE: PROCEED

Pre-registered kill criteria (in the script header before any run):

| Criterion | Threshold | Measured | Result |
|---|---|---|---|
| K1: best surrogate-support fraction | < 0.05 → stop | **0.65** | pass |
| K2: best relative-gradient ratio vs float control, int8 | < 10⁻³ → stop | **11.4** (median 0.028) | pass |
| K2: same, ternary | < 10⁻³ → stop | **419** (median **2.46**) | pass |
| K3: per-tick gradient ratio t₀/t₇ at T=8 | < 10⁻⁴ → stop | min **0.91** across all quantized configs | pass |
| F: toy-task accuracy (chance 25%) | ≤ 40% → stop | float **100%**, int8 **97.2%**, ternary **100%** | pass |

## R1 resolved: the NULL was placement, not ternarization

The ChannelBitLinear NULL (gate gradient ~2.4e-5) is **reproduced inside this sweep** — as its worst corner, not its typical behavior. The three worst configs are all int8 + **fast leak (k=2) + high threshold (q=0.95) + triangular surrogate**, with input-layer relative gradients of 1.4e-5–2.4e-5 — the same order as the NULL. One of them has 52% of membranes inside the surrogate support and still passes ~2e-5: with a compact-support kernel and a fast leak, the membrane distribution clusters at the *edge* of the support, where the triangular kernel is ~0. Fat-tailed kernels (atan, fast sigmoid) have no such cliff.

So the diagnosis the handoff hypothesized is confirmed and sharpened: **the pathology is threshold placement × leak speed × kernel shape.** With θ calibrated to the membrane distribution, gradient flows in essentially every config, and all three precisions train the toy task to ≥97%.

**Ternarization is not the problem — it is the best-conditioned precision.** Median relative input-layer gradient: ternary **3.1e-2**, float 6.8e-3, int8 2.4e-4. Median ratio vs the matched float control: ternary **2.46** (passes *more* relative gradient than float), int8 0.028. Mechanism: ternary weights force small integer currents, calibration then places θ within a few units, so membranes live close to threshold on a coarse grid — dense comparator traffic. Int8's wider dynamic range (θ ~ tens–hundreds) spreads membranes far from θ (median u ≈ −8.5 widths).

## R2 resolved: no BPTT collapse at T ≤ 32

Per-tick gradient ratios (earliest/latest tick) for quantized precisions: median 3.0–3.4 at T=8 and 4.7–8.6 at T=32, minimum 0.91 — early ticks receive *more* credit than late ones (they influence every later readout through the membrane), never vanishingly less. Temporal credit assignment is healthy through the integer dynamics at both depths on this task.

## Secondary findings (feed D3/D4/D5)

- **Leak speed dominates trainability:** k=15 (slowest, −1/tick) beats k=2 by ~6× in median ternary gradient (9.9e-2 vs 1.7e-2). Consistent with R3: fast leak destroys the temporal pathway. Default k high; k is per-neuron learnable on the chip anyway.
- **Lower thresholds are better:** median ternary gradient at θ-quantile 0.5 / 0.8 / 0.95 = 1.05e-1 / 2.7e-2 / 1.5e-2.
- **Shape:** atan ≈ fast_sigmoid > triangular at every width (ternary medians at w=0.25: 4.2e-2 / 4.1e-2 / 2.1e-2), and triangular owns the entire dead-corner zone. D5 → **atan** (or fast sigmoid), never triangular.
- **Width caveat:** raw gradient magnitude scales as 1/width, so "narrow width wins" in gradient norm is partly arithmetic inflation, not signal quality. The functional check trains fine at width 0.05·θ and 0.25·θ both. Recommended default: **atan, width 0.25·θ**, tune with θ jointly.
- **Calibration floor:** at q=0.5 the calibrated θ clamps to 1 (median membrane ≤ 0) for quantized nets — the winning configs are effectively "fire on any positive potential". This says θ and weight scale must be co-designed; per-neuron *learnable* θ (a chip feature, invariant I7) is the natural fix and should be a P3+ deliverable.

## Decisions taken

- **D3: ternary `{−1, 0, +1}` as primary weight precision** (chip-losslessly mappable, BitNet memory win, and now: best gradient conditioning). Int8 stays as the fallback superset.
- **D5: atan surrogate, width 0.25·θ default**, triangular excluded.
- **D4 input:** no structural obstacle up to T=32; choose T from the task + leak analysis (P0 §6) once D6 fixes the task.

## Honest limits of this diagnostic

Toy task (rate discrimination — no long temporal dependencies), dense connectivity (P3's sparse edge lists not yet in the loop), fixed uniform θ per layer, gradient statistics at init only, single seed for the sweep (5 seeds would tighten the medians but the margins are orders of magnitude). The 419× and 9553× *maxima* in the verdict come from near-dead float controls in bad corners; the medians quoted above are the honest summary.

## Next

P3 — sparse connectivity primitive (edge-list `scatter_add`, block structure ≥ 64) with ternary weights and learnable per-neuron θ, then P4 topology learn-and-freeze.
