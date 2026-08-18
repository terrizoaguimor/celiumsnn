# P0 — Extracted Semantics of the CeliumNeUR Neuron & Network

**Source of truth:** `celiumneur/golden/` at clone of 2026-08-17 (v0.0.2, 55/55 pytest green in this environment).
**Authority order:** golden model > SPEC.md prose > handoff §3. This document supersedes handoff §3.
**Status:** P0 complete. Every claim below cites the golden source line that pins it.

---

## 1. Pinned parameter spaces

| Parameter | Range | Evidence |
|---|---|---|
| membrane `v` | signed 16-bit, [−32768, +32767], saturating (I6) | `soma.py:18-20,27-32` |
| `theta` | **[1, 32767]** — golden rejects 0 and anything above `VMEM_MAX` | `soma.py:69-70` |
| `leak_shift k` | [0, 15]; validated at leak time | `soma.py:42-43` |
| `refractory_ticks` | ≥ 0 validated; word field is 8 bits → [0, 255] | `soma.py:71-72`, SPEC §6.1 |
| weight | signed 8-bit [−128, +127]; validated on the apply path; CWR clamps | `soma.py:22-24,89-90`, `plasticity.py:20-26` |
| reset mode | per-neuron: subtractive (default) or to-zero | `soma.py:66` |

**Checklist resolution — threshold width mismatch:** SPEC §6.1 declares the threshold field unsigned 16-bit (up to 65535), but the golden model refuses `theta > 32767` at construction. Values 32768–65535 are expressible in the soma word but **outside the verified semantic space** — no golden test, no RTL equivalence claim covers them. The model pins `theta_max = 32767`. (Whether RTL silently accepts a dead threshold ≥ 32768 is an open item for the chip repo, not for this model — see §10.)

---

## 2. Single-neuron operational semantics

The neuron has two entry points mirroring the hardware split (`soma.py:75-117`). State: `v` (int16), `refractory_countdown` (`cd`), constructed with `v0` saturated (`soma.py:84`).

### 2.1 Event path — `apply_synaptic_input(weight)` (`soma.py:87-92`)

```
v  ← sat16(v + weight)
fired ← evaluate()
```

- **Fire is evaluated on every synaptic event**, immediately, not at the tick boundary.
- The apply path **never** touches `cd` (refractory ages on ticks only — SPEC §6.1, pinned by `test_soma.py:38-46`).
- Input during refractory integrates normally but cannot fire (`test_soma.py:131-135`).

### 2.2 Tick path — `advance_time()` (`soma.py:94-107`)

```
v  ← sat16(v − ceil_leak(v, k))      # leak toward zero
fired ← evaluate()                    # refractory-exit fire lives here
if cd > 0: cd ← cd − 1               # decrement AFTER evaluation
```

- **Leak:** magnitude `ceil(|v| / 2^k)` toward zero, exact ceiling division (`soma.py:35-48`). Reaches exactly 0, never asymptotes (`test_soma.py:66-81`).
- **Checklist resolution — `k = 0`:** `ceil(|v|/1) = |v|` → full discharge to zero in one tick. **Intended**, documented as "maximum leak" (`soma.py:57`).
- `k = 15` decays by exactly 1/tick for any |v| ≤ 32768 (`test_soma.py:95-98`) — the slowest possible leak, aliased `NO_LEAK` in the tests.
- Leak alone never causes a threshold crossing (it only shrinks |v|); the tick-path evaluation exists for a superthreshold membrane **leaving refractory with no new input** (`soma.py:96-99`, pinned by `test_soma.py:152-159`).

### 2.3 Fire evaluation — `_evaluate_spike()` (`soma.py:109-117`)

```
if cd > 0:        no fire (v untouched — superthreshold charge survives refractory)
if v < theta:     no fire
else:             fire; v ← (v − theta) if subtractive else 0; cd ← refractory_ticks
```

- **Checklist resolution — `>=` vs `>`:** the guard is `v < theta → no fire`, therefore the fire condition is **`v >= theta`** (`soma.py:113`).
- Subtractive reset keeps the residue `v − theta` (`test_soma.py:109-114`); reset-to-zero drops it.

### 2.4 Refractory timing — including an asymmetry the handoff did not know about

Because `advance_time` decrements `cd` *after* evaluating, the two fire paths block differently:

- **Event-path fire** (during integration): `cd ← R`; the next R ticks evaluate with `cd > 0` → blocked. **Blocks exactly R ticks** (the comment at `soma.py:103-104` and `test_soma.py:138-149`).
- **Tick-path fire** (refractory exit): `cd ← R` inside evaluation, then the same call decrements it to `R−1`. **Blocks only R−1 subsequent ticks.** With `R = 1`, a tick-path fire permits firing again on the very next tick.

This asymmetry is golden-pinned behavior (implied by the trace of `test_soma.py:152-159`). **The PyTorch model must replicate it, not rationalize it.** If it is judged a golden-model bug, that is a chip-repo issue to file; until the golden model changes, the model copies it.

### 2.5 Multi-fire per tick

With `refractory_ticks = 0`, nothing prevents multiple fires per phase: each event can fire (`test_soma.py:197-203`), and the tick path can fire again on the residue. Each fire stages an **independent spike packet** (`golden_net.py:95-115`) — a neuron's per-tick output is a **count, not a bit**.

With `refractory_ticks ≥ 1`, at most one fire per phase is possible (the first fire sets `cd ≥ 1`, blocking every later evaluation in that phase).

**Design consequence:** the model should enforce `refractory_ticks ≥ 1` in its configuration space to get binary spike tensors. `R = 0` configurations remain expressible only in the golden/RTL world.

---

## 3. Network phase contract (`golden_net.py:140-164`)

Per `tick()`, in this exact order:

1. **Drain & integrate** (`golden_net.py:117-133`): the mesh drains **fully** (`run_until_idle`); every delivered packet expands through the dendrite table; each matching entry applies one `apply_synaptic_input` in order (core 0→3, packet delivery order, table row order). Fires stage packets that stay queued **until the next phase** — no intra-phase cascade (deliberate, documented ReckOn-class boundary, `golden_net.py:9-16`).
2. **Advance time**: every soma takes one `advance_time`; tick-path fires also stage for the next phase.
3. **Trace**: `v_trace` records `v` *after* the tick path (`golden_net.py:148-149`).
4. **CWR LTD pass** (if plasticity enabled), then `tick_index += 1`.

Other pinned facts:

- **Checklist resolution — order of leak vs accumulation:** synaptic integration first, leak second, within every tick.
- **Uniform 1-phase delay:** fires at phase t deliver at phase t+1 regardless of hop count, because the mesh drains to quiescence inside each phase. The SPEC's phase-parity mechanism (§4.3) implements the same contract in RTL.
- **Fabric is semantically transparent** to the model: credit flow control and FIFO depths are liveness/backpressure mechanisms with **no drop path** (I1, `hyphae.py:18-21`); they do not alter which weights arrive, only when within the phase.
- **Duplicate `(pre, post)` entries have real multiplicity**: the dendrite table is a list of rows, each delivering an independent soma event (`golden_net.py:39-44,121-126`; SPEC §6.2).
- **External stimulus** (`stimulate`, `golden_net.py:135-138`) is an immediate event-path injection into one neuron, weight constrained to int8 range; a resulting fire stages for the next phase. This is the input-encoding primitive the D7 decision must build on.

---

## 4. The finding that changes P1: per-event evaluation vs batched `scatter_add`

The handoff assumed the P1 forward pass is "scatter_add, then evaluate." The golden model evaluates fire **per event**, in delivery order, with saturation per event. These are **not equivalent in general**. A neuron can fire mid-phase and integrate the rest of the phase's input on the post-reset potential; delivery order is deterministic but arbitration-dependent (`hyphae.py:143-168`).

Sum-then-evaluate-once matches the golden model for a given neuron in a given phase **iff all of**:

- **C1 — subtractive reset.** Subtracting θ commutes with later additions, so *when* the fire happened inside the phase doesn't change the final `v`. Reset-to-zero does not commute: order-dependence is unfixable there.
- **C2 — `refractory_ticks ≥ 1`.** Otherwise per-event semantics can fire multiple times per phase while the batched form fires once.
- **C3 — no intermediate saturation.** No prefix of the delivery sequence leaves int16 range when the phase total is in range (weights are ±127, so this needs |v| near the rails). Same-signed inputs do **not** guarantee this: a mid-phase fire subtracts θ before the remaining inputs land, so golden can clamp at a different point than the batched sum does (the P1 harness hit this with a low-θ neuron drifting to |v| ≳ 32600). Bit-exact operation requires staying clear of the rails.
- **C4 — no order-dependent threshold crossing.** Golden evaluates fire at every event, so with phase events `w_1..w_n` it fires iff **any prefix** `S_j = v_start + w_1 + … + w_j` reaches θ (with `cd == 0`); the batched form fires iff `S_n ≥ θ`. Equivalence needs `(∃j: S_j ≥ θ) ⟺ (S_n ≥ θ)`. Same-signed inputs per (neuron, phase) guarantee it **except in one reachable state found by the P1 differential harness:** a neuron can *enter* a phase already superthreshold with `cd == 0` (it accumulated past θ during refractory and the countdown expired), and then even a purely inhibitory phase fires golden on its first event unless that event alone drops v below θ. Inhibitory input onto a superthreshold-entering neuron is therefore also order-dependent; harnesses must exclude it (excitation-only into such neurons), and network designs wanting bit-exactness should keep per-target input single-signed.

**Recommendation for P1:** define the model's contract as the batched tick-synchronous form below, and make the bit-exactness harness *measure* C1–C4: drive golden and model on identical stimulus, assert exact state equality, and classify any divergence by which condition failed. C1 and C2 are enforceable by configuration fiat (model config space: subtractive reset only, R ≥ 1). C3–C4 are data-dependent; the chip-faithful reference instance must either operate in a regime where they hold or downgrade its claim from bit-exact to divergence-bounded on those phases. This is now a *measured* property, not an assumption.

- **C5 — external stimulation must not fire at injection time.** Golden `stimulate` is a pre-phase event whose fire would deliver **within the same phase** (`golden_net.py:135-138` injects before `tick()` drains), breaking the model's uniform 1-step delay. The model folds external stim into the phase input `I`; equivalence holds only when every golden stim event is subthreshold at injection (`v + w < θ`), so integration-path fires are the only fires. This is a harness/encoding constraint, and D7 (input encoding) should choose an encoding that satisfies it by construction.

### Reference forward pass (candidate P1 contract, all ops int, per tick)

**The event gate `E` is load-bearing.** Golden evaluates fire only when `apply_synaptic_input` is actually called; a neuron receiving *no* events this phase gets its only evaluation **post-leak** (tick path). Without the gate, a neuron exiting refractory with superthreshold charge would be evaluated pre-leak on quiet phases — firing when golden does not (e.g. `k=1, v=90, θ=50`: golden leaks to 45 → no fire; ungated `f_evt` fires). Conversely a zero-weight or zero-sum event *does* evaluate in golden, so `E` is event **presence**, not `I ≠ 0`.

```
I      = scatter_add(weights of spikes staged at t−1) + external stim
E      = any event arrived this phase (synaptic or stim), per neuron
v      = sat16(v + I)
f_evt  = E & (cd == 0) & (v >= θ)                # event-path fire (≤1 by C2)
v      = f_evt ? (v − θ) : v                     # C1: subtractive only
cd     = f_evt ? R : cd
v      = sat16(v − ceil_leak(v, k))              # leak after integration
f_tick = (cd == 0) & (v >= θ)                    # refractory-exit fire, post-leak
v      = f_tick ? (v − θ) : v
cd     = f_tick ? R : cd
cd     = max(cd − 1, 0)                          # decrement last → §2.4 asymmetry preserved
spike_out[t] = f_evt | f_tick                    # delivered to targets at t+1
v_trace[t]   = v
```

Note `f_evt` and `f_tick` are exclusive under C2 (an event-path fire sets `cd = R ≥ 1` before the tick-path evaluation).

---

## 5. CWR — pinned learning semantics (`golden_net.py:95-133,150-163`, `plasticity.py`)

Not used for GPU training; pinned here because D3 (ternary) argued from it.

- **Ledger:** one slot per *physical synapse entry*, keyed `(core, pre_gid, entry_index)`; a newer arrival **overwrites** the slot (`test_golden_net.py:9-23`). Arrivals are stamped with the current `tick_index`.
- **LTP** at fire time (any fire path, including `stimulate`): every ledger entry targeting the firing neuron with `0 ≤ t_fire − t_arr ≤ W` gets `w ← clamp(w+1)` and is **consumed** — each arrival pays at most one fire (`golden_net.py:99-114`).
- **LTD** at end of each tick: every surviving entry with `t − t_arr > W` gets `w ← clamp(w−1)` and is removed (`golden_net.py:150-163`). Expiry is the **only** LTD path (the v1.0 order-artifact is explicitly dead, `plasticity.py:14-17`).
- Default `W = 3` ticks. Steps are ±1 LSB, saturating at ±127/−128.
- No weight math at arrival time (pair rule v1.2) — arrival only writes the ledger.

**Correction to handoff §3:** the claim "ternarization is what makes on-chip learning tractable" is inverted. Over int8, ±1 steps make CWR an integrator with inertia (128 causal pairs to traverse the range). Over a ternary range, one or two events flip a weight's sign — a hair-trigger, not tractability. If D3 chooses ternary *and* on-chip CWR matters, that interaction needs its own analysis; do not carry the handoff's framing forward.

---

## 6. Leak → memory horizon (feeds D4)

The leak subtracts `ceil(|v|/2^k) ≥ 1` per tick, so:

- **Lifetime bound:** any potential dies in at most |v| ticks; at `k = 15` in exactly |v| ticks (linear −1/tick). Between ticks nothing decays — memory is per-phase, not per-event.
- **Geometric regime:** for `|v| > 2^k` the decay is ≈ factor `(1 − 2^−k)` per tick; below `2^k` it degenerates to −1/tick steps until zero.
- **Hold cost:** to sustain a potential v against leak requires net input ≥ `ceil(v/2^k)` per tick. With ternary weights and fan-in f, sustainable v is bounded by `f·2^k`; thresholds above that are unreachable in steady state. This is the quantitative form of the D3 counter-consideration, and the P2 diagnostic should report measured `v`-distribution against both θ and `f·2^k`.
- The correct framing (supersedes handoff §3 wording): the neuron cannot hold **weak** evidence; strong evidence persists for a time proportional to its magnitude. The usable temporal window T is a function of signal scale relative to θ, not a constant.

---

## 7. Open items (none block P1)

1. **RTL behavior for `theta ∈ [32768, 65535]`** — expressible in the soma word, unverifiable via golden. Chip-repo question; the model excludes the range (§1).
2. **Refractory asymmetry (§2.4)** — replicated as-is; worth filing upstream as a "confirm intended" issue against `celiumneur`, since SPEC prose ("refractory duration in ticks") doesn't distinguish the two fire paths.
3. **Delivery-order canonicalization** — the golden order (core, delivery, row) is deterministic, but the RTL's intra-phase order comes from router arbitration. Bit-exactness *across all three* (PyTorch ≡ golden ≡ RTL) at intra-phase granularity is only meaningful under C1–C4, where order is irrelevant. The equivalence harness should compare **end-of-phase state**, which is what the RTL bench's multiset-equality contract already does.
4. **SPEC section numbering** — handoff cites soma as SPEC §6.1 (correct for SPEC.md v0.0.2); golden docstrings cite §3 (stale). Cosmetic; golden numbering should not be trusted for SPEC cross-references.
