# Decision Log — CeliumNeUR → SNN

Ratified 2026-08-17, before first line of model code, per handoff §11-Q5 / R5.
Context documents: `celiumneur-snn-handoff.md` (Downloads), `P0-SEMANTICS.md` (this repo).

## D1 — Scope: (c) Both

The architecture scales freely; a chip-faithful configuration (≤1,024 neurons,
≤1,024 synapse entries, int8 weights, semantics per P0) is held as the
verifiable reference instance.

Post-P0 qualification: the reference instance operates in the C1–C2 subspace
(subtractive reset only, `refractory_ticks ≥ 1`) and its equivalence claim is
**end-of-phase state equality**, with C3/C4 violation rates measured and
reported per run. This matches the granularity the RTL bench already contracts
(multiset equality). Claims at intra-phase granularity are out of scope.

## Phase order — P1 → P2 gate before any architecture decision

P1 (integer LIF + equivalence harness) runs first because P2 cannot measure
surrogate gradients without a differentiable forward. D2 (macro-architecture),
D3 (weight precision), D4 (temporal depth), D5 (surrogate shape/width),
D6 (task/baseline), D7 (input encoding) are all deferred until the P2 gate
passes. Only negative decision taken now: no data-dependent routing
(attention) in the signable path — it is what the fabric cannot express.

## Kill criteria (ratified; adjustable only toward stricter)

### Gate P1 — equivalence
- Exact end-of-phase state equality (v, refractory countdown, spike sets,
  weights if plasticity off) against `golden/` on:
  (i) scenarios derived from the 55 golden tests,
  (ii) ≥10⁵ fuzzed neuron-phases within C1–C2 configurations.
- Every divergence must be classified as a measured C3 (intermediate
  saturation) or C4 (mixed-sign threshold crossing) violation.
- Any unclassified divergence = blocking bug. Do not proceed to P2.

### Gate P2 — gradient viability (the real kill)
Fixed diagnostic task, fixed budget, sweep of surrogate widths expressed in
units of per-neuron θ. **Stop the project** if ALL three hold:
1. Median |∂L/∂w| on hidden weights < 10⁻³ × a dense float control with
   identical init and task;
2. Dead-gradient neuron fraction > 90% across the entire width sweep;
3. Fails to beat logistic regression on the same input within budget.
Any single condition alone = yellow flag, report and continue with caution.

### R5 — time box
Owner-defined (Mario). To be stated before P3 begins; P1+P2 are bounded by
their gates.

## Standing corrections adopted from P0 (supersede handoff wording)
- Ternary-CWR "tractability" argument is inverted (P0 §5); D3 will be decided
  on R1 diagnostics + BitNet memory economics only.
- Temporal window T is signal-scale-dependent (P0 §6), not a constant; D4
  derives from P2 measurements, not SNN convention.
- Golden evaluates fire per event; the model's batched contract is P0 §4's
  reference forward pass under C1–C2.
