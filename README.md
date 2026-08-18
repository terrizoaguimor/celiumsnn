# celiumsnn — SNN model architecture from the CeliumNeUR constraint set

Working repo for the model project described in `celiumneur-snn-handoff.md`
(chip repo vendored read-only under `celiumneur/`, AGPL-3.0-or-later).

## Status

- **P0 done** — `P0-SEMANTICS.md`: extracted semantics, supersedes handoff §3.
  Source of truth is `celiumneur/golden/` (55/55 pytest green).
- **P1 done** — `celiumsnn/lif.py`: `IntLIF`, integer tick-synchronous LIF,
  bit-exact against the golden model under the C1–C5 contract
  (P0-SEMANTICS.md §4). 40 tests in `tests/`:
  - `test_lif_unit.py` — golden `test_soma.py` ported to the tick contract;
  - `test_equivalence_soma.py` — randomized differential vs golden `Soma`,
    plus explicit C3/C4 contract-boundary divergence tests;
  - `test_equivalence_net.py` — whole-network differential vs golden
    `NeuroSandbox` (mesh + dendrite + somas), phase-by-phase state equality.
  Extended sweeps run clean: 50 seeds × 500 phases × 32 neurons (soma) and
  10 seeds × 80 phases × 1024-neuron sandbox (network).
- **P2 done — GATE: PROCEED** (`P2-REPORT.md`): all pre-registered kill
  criteria pass. The ChannelBitLinear NULL reproduced only as the sweep's
  worst corner (fast leak + high θ + triangular surrogate) — a placement
  pathology, not ternarization. Ternary is the best-conditioned precision.
  Decisions: D3 = ternary primary, D5 = atan width 0.25·θ, k high.
  New code: `celiumsnn/lif_diff.py` (DiffLIF, forward bit-identical to
  IntLIF — tested), `celiumsnn/surrogate.py`, `celiumsnn/quant.py`,
  `experiments/p2_diagnostic.py` (sweep + verdict, results under
  `experiments/results/`).
- **P3 done** — `celiumsnn/synapse.py`: `EdgeListSynapse` (chip-exact
  dendrite table, certified against golden — the P1 network equivalence
  test now routes delivery through it) and `BlockSparseSynapse` (static
  block mask, dense ternary/int8 blocks, exact event semantics: an active
  block = B² valid dendrite entries; equivalence to its edge-list expansion
  tested). `DiffLIF` gained learnable per-neuron θ on the chip grid
  (STE-rounded, IntLIF-equivalent forward). Functional check: block-sparse
  ternary net (hidden 256, layer-2 mask at 69% of dense FLOPs, learnable θ)
  reaches 100% on the rate task in 200 steps.
- **P4 done** — `celiumsnn/gates.py`: hard-concrete L0 gates over the block
  grid (`GatedBlockSparseSynapse`), `freeze()` → static mask + SHA-256
  topology hash. On a planted-structure task (signal block + noise block),
  λ=0.1 recovers the planted relevance exactly — noise routes all closed,
  layer 2 at 12.5% of dense FLOPs, accuracy 1.0 before and after freeze.
  Random masks at the same density are a coin flip (2/4 collapse to
  chance). See `P4-REPORT.md`. Experiments now run on the DO droplet
  `celiumsnn-p4` (c-60-intel, 142.93.187.63, project at
  `/root/snnceliumsneur`) — the local workstation is too small.
- **Next: P5** — task, baseline, curve (D6): quality-per-byte and
  quality-per-op vs a dense fp16 equivalent under identical budget.

## Run

```bash
.venv/bin/python -m pytest            # model suite (tests/)
cd celiumneur/golden && ../../.venv/bin/python -m pytest   # golden referee
```

Requires the venv at `.venv/` (python ≥ 3.10, torch CPU, pytest).

## Contract in one line

`IntLIF.step(I, has_event)` = one global tick of the chip: integrate the
phase sum, evaluate pre-leak iff events arrived, leak (ceil, toward zero),
evaluate post-leak on refractory exit, decrement the countdown last.
Bit-exactness vs golden holds under C1–C5; violations are order-dependent
by nature and are pinned by the two divergence tests, not hidden.
