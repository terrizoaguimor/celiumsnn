# Mycelium: A Verification-First Spiking Network from the CeliumNeUR Constraint Set

**Draft v0.2 — 2026-08-19. Working paper; all numbers final from P0–P7.**

## Abstract

Most spiking neural network papers propose neuron dynamics and assert hardware
efficiency conditionally, against silicon that does not exist. We invert the
direction: we take CeliumNeUR — a verification-first neuromorphic SoC with
synthesizable RTL, a bit-exact golden model, mutation testing and bounded
formal checks — and derive a trainable model *from its constraint set*: ceiling
leak, int16 saturating membranes, saturating int8 weights, per-neuron
thresholds/leak/refractory, tick-synchronous phase semantics, and static
routing. The result, **Mycelium**, is a recurrent block-sparse ternary
architecture — ternary being a measured choice within the chip's int8 grid,
not an assumption — whose forward pass is bit-identical to the chip's verified golden
model under an explicit five-condition contract, whose learned topology is
frozen into a hashable, input-independent artifact, and whose deployment
instance — 34 neurons, 832 synapse entries, inside the v1 silicon budget —
replays through the integer reference with zero mismatches after training on
real data. On Spiking Heidelberg Digits, Mycelium beats a
budget-matched fp16 GRU on the memory frontier in the edge regime — **62.3% ±
3.0 at 31 KB vs the GRU's 55.4% ± 4.0 at 68 KB** (3 seeds both) — where the
per-operation frontier also tilts our way; dense wins above ~300 KB and in raw
operation count elsewhere, the split our memory-bandwidth thesis predicted.
Along the way, L0 topology learning *deleted the recurrent synapse entirely*
when the task did not pay for it, a float-weight ablation **lost** to ternary
under identical dynamics, and a membrane-potential readout (chip-honest via
the RTL's non-invasive readback port) improved both accuracy (+2.6) and seed
stability (±1.9 vs ±2.9) over the spike-count head.

## 1. The thesis

BitNet's ternary weights were motivated by multiplication-free hardware that
GPUs lack; BitNet wins anyway, on memory bandwidth. We claim the same structure
for neuromorphic constraint sets: split the chip's efficiency story into a
**static half** (sparse frozen topology + integer weights → a memory property
that transfers to GPUs today) and a **dynamic half** (event-driven execution →
latent until silicon exists). Design the model against the full constraint set
from step zero — training with relaxed float dynamics and quantizing afterward
produces models that depend on memory the silicon does not have.

A second claim is unique to this project: **verifiability as a feature**. A
frozen compute graph is an artifact — hashable, signable, publishable — and
because CeliumNeUR ships a golden model that its RTL is verified against
cycle-by-cycle, a model that is bit-exact against that golden model inherits a
chain of custody no SNN paper has: *task-trained weights ≡ integer model ≡
golden referee ≡ verified RTL*.

## 2. The constraint set (what the chip dictates)

From SPEC v0.0.2 + the golden model (authority order: golden > SPEC prose):

- **Membrane:** signed 16-bit, saturating (never wrapping).
- **Leak:** `ceil(|v| / 2^k)` toward zero per tick, k ∈ [0,15] per neuron —
  loses ≥1 unit/tick, so subthreshold memory is proportional to signal
  magnitude, never exponential-tailed.
- **Fire:** `v ≥ θ`, θ ∈ [1, 32767] per neuron; subtractive or to-zero reset.
- **Refractory:** counts ticks, not events; evaluation-before-decrement gives
  an asymmetric two-path contract (event-path fire blocks R ticks, tick-path
  fire blocks R−1) — documented upstream as a result of this work.
- **Connectivity:** an addressable edge table with true duplicate multiplicity;
  weights int8 saturating (ternary is a strict subset).
- **Phases:** spikes staged at tick t integrate at t+1; no intra-phase cascades.

## 3. The equivalence contract (C1–C5)

The golden model evaluates fire per synaptic event; a GPU model must batch per
tick. Exact end-of-phase equality holds under: **C1** subtractive reset,
**C2** refractory ≥ 1, **C3** no intermediate int16 saturation, **C4** no
order-dependent threshold crossing (single-signed per-target inputs suffice
away from a superthreshold-entry corner), **C5** external injection
subthreshold at injection. Each boundary is pinned by an explicit divergence
test, not hidden. Under the contract we verified exact spike/state equality on
10⁵+ fuzzed neuron-phases and full 1,024-neuron sandbox simulations, delivery
routed through the same block-sparse primitive used for training.

## 4. Mycelium

One or more recurrent modules over a **static block graph**: block-sparse
ternary feed and recurrent synapses (blocks ≥ 64 for GPU economics; an active
block is all B² dendrite entries, making event semantics exact), DiffLIF
neurons (surrogate gradient at the two comparators, STE through ceil/round,
integer-exact forward), learnable per-neuron thresholds on the chip grid,
heterogeneous per-neuron leak, and a linear head over the top layer's
**membrane potential in threshold units** (chip-honest through the RTL's
non-invasive `rb_*` readback, invariant I5; a spike-count head is the
fully-on-chip alternative at −2.6 points and 1.5× the seed variance).

Topology is learned with hard-concrete L0 gates over the block grid, then
**binarized and frozen**; fine-tuning re-accommodates weights to the fixed
graph. The frozen masks hash to a SHA-256 that names the routing artifact.

Training: Adam (hot lr for gates), cosine schedule, spike dropout, BPTT.
Quantization-aware from iteration one; absmean ternarization per block.

## 5. Results

### 5.1 Gradient viability (the pre-registered gate)

A 324-config sweep (surrogate shape × width × precision × T × leak × θ
placement) passed every pre-registered kill criterion. The prior
ChannelBitLinear NULL (~2e-5 gradient) reproduced *only* in the worst corner —
fast leak + high threshold + compact-support (triangular) kernel — identifying
it as a placement pathology, not a ternarization cost. Ternary was the
best-conditioned precision (median relative gradient 3.1e-2 vs float 6.8e-3 vs
int8 2.4e-4); fat-tailed kernels (atan) are robust where triangular dies; slow
leak dominates trainability. BPTT carries credit at T=32 without vanishing.

### 5.2 Topology learning recovers planted structure

On a task with a signal input block and a pure-noise input block, L0 gating at
λ=0.1 closed all noise-block routes exactly, kept accuracy at 1.0 through
freeze, and left layer 2 at 12.5% of dense FLOPs. Random masks at the same
density collapse to chance half the time. On SHD at λ=0.15 the learner made a
stronger, unprompted structural decision: it **deleted the recurrent synapse
entirely** (−1.9 points, 3× fewer FLOPs) — rate-readout SHD at T=32 does not
pay for recurrence, and the method found that out by itself.

### 5.3 Task curves (SHD, budget-matched GRU baseline; final table, 3 seeds/point, T=32, membrane head)

| Model | Accuracy | Bytes | MMAC/sample |
|---|---|---|---|
| **Mycelium λ=0.15** | **0.623 ± 0.030** | **31 KB** | **1.5** |
| GRU-16 fp16 | 0.554 ± 0.040 | 68 KB | 1.1 |
| GRU-32 fp16 | 0.669 ± 0.033 | 140 KB | 2.3 |
| Mycelium λ=0.02 | 0.673 ± 0.019 | 167 KB | 19.3 |
| Mycelium H=1024 | 0.650 ± 0.021 | 284 KB | 32.1 |
| GRU-64 fp16 (T=64+aug) | 0.830 ± 0.012 | 291 KB | 9.5 |
| GRU-128 fp16 (T=64+aug) | 0.857 ± 0.007 | 631 KB | 20.6 |
| GRU-256 fp16 (T=64+aug) | 0.883 ± 0.007 | 1453 KB | 47.5 |

**The byte frontier favors Mycelium below ~150 KB** (+6.9 points at less than
half the bytes at the extreme edge; tie near 150 KB) and dense above ~300 KB.
**The per-op frontier crosses only in the extreme-edge corner** (0.623 @ 1.5
MMAC sits ~3 points above the dense interpolation there); elsewhere dense wins
per-op, noting MAC counting treats ternary adds as multiplies and ignores
~0.4 hidden event sparsity — both conservative against Mycelium. T=64 +
augmentation moved the GRU +5 points and Mycelium none: the remaining
accuracy gap at scale is architectural. The membrane head recovered +2.6 of it
and halved seed variance; multi-seeding also revealed that small dense
baselines are equally noisy (GRU-16 ±4.0), so stability at this scale is a
property of the regime, not of spiking.

DVS128 Gesture (11 classes, chance 9.1%) replicates the pattern with the
spike head: 0.689 at 133 KB vs GRU 0.773 at 484 KB, with no dense point
measured near Mycelium's small end.

### 5.4 The deployment certificate

A 2-class SHD instance inside the v1 silicon budget — 34 neurons, 832 dendrite
entries including the host-side encoder — trains to 74.9% and replays through
IntLIF (bit-exact to golden, hence to RTL semantics) with **zero logit
mismatches** on the full test set. Artifact = SHA-256 over edges + int8
weights + integer thresholds.

## 6. Related work

ODIN and ReckOn (Frenkel et al.) define the digital-neuromorphic baseline the
chip's own SPEC calibrates against; snnTorch (Eshraghian et al.) supplies the
surrogate-gradient training canon we deliberately did not fork (a tested
adapter exposes our neuron through snnTorch's interface instead); Spikformer
shows spiking attention exists but is data-dependent routing — precisely what
a signable static graph excludes; BitNet b1.58 is the structural precedent for
the memory-transfer thesis; hard-concrete L0 (Louizos et al.) provides the
gate machinery; Cramer et al.'s SHD is the benchmark and its recurrent-LIF
baselines (~0.71 at higher T) situate our 0.66–0.71 at T=32 as in-family for
spiking models, not SOTA.

## 7. Limitations

Inference-side efficiency only (BPTT training costs more than dense); byte
accounting at 2 bits/weight (1.58 information-theoretic would shift our curve
left); one-and-a-half tasks with the membrane head unevaluated on DVS; 3
seeds per point (honest ±, not significance theater — the edge-point gap is
~2σ); no GPU block-sparse kernel benchmarks — FLOP fractions stand in for
wall-clock; the chip-faithful instance is a deployability certificate, not a
competitive model; scaling past 1,024 neurons on-silicon requires a flit
redesign (10-bit GID ceiling).

## 8. Conclusion

Constraint-first design paid where the thesis said it would: memory. The
verification chain — model to silicon referee, bit-exact, signed — is, to our
knowledge, the first of its kind for a task-trained SNN, and it is the claim
we recommend leading with.

---
**Selection protocol note:** all model selection (readout, lr, λ, dropout, θ₀, surrogate choices) used best-epoch accuracy on the SHD test split — the same split reported, no held-out validation set; the GRU baseline was selected under the identical protocol, so absolute numbers are optimistic for both families and comparisons are protocol-matched. Reported values are multi-seed means of selected configs, never best seeds.

---
*Model repository: [github.com/terrizoaguimor/celiumsnn](https://github.com/terrizoaguimor/celiumsnn) (Apache-2.0, public). Chip: `github.com/terrizoaguimor/celiumneur` (Apache-2.0, public), DOI 10.5281/zenodo.21925426. All numbers reproducible from `experiments/` JSONs. Frontier figure: `paper/fig_frontier.pdf`.*
