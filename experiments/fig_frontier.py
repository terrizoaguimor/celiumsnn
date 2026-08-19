# SPDX-License-Identifier: Apache-2.0
"""Frontier figure for the paper: accuracy vs deployment bytes, SHD.

Every point is 3 seeds (mean ± sd). GRU points use each size's best
measured condition — T=32 for GRU-16/32, T=64+augmentation for
GRU-64/128/256 — i.e. the baseline is shown at its best, which is
conservative against Mycelium. Colors are the CVD-safe Okabe-Ito pair
(validated: worst-pair ΔE 21.9 protan / 31.2 normal on white).

Usage: .venv/bin/python experiments/fig_frontier.py
Writes paper/fig_frontier.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MYCELIUM = "#0072B2"
GRU = "#D55E00"
INK, MUTED = "#333333", "#767676"

# (label, KB, mean, sd, label dx multiplier, label dy)
myc = [
    (r"$\lambda$=0.15", 30.8, 0.623, 0.030, 1.0, 0.042),
    (r"$\lambda$=0.02", 166.8, 0.673, 0.019, 1.28, -0.012),
    ("H=1024", 283.6, 0.650, 0.021, 1.0, -0.042),
]
gru = [
    ("GRU-16", 68.4, 0.554, 0.040, 1.0, -0.058),
    ("GRU-32", 139.7, 0.669, 0.033, 0.80, 0.030),
    ("GRU-64", 291.3, 0.830, 0.012, 0.90, 0.028),
    ("GRU-128", 630.5, 0.857, 0.007, 1.0, 0.024),
    ("GRU-256", 1453.0, 0.883, 0.007, 1.0, 0.024),
]

fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)

for pts, color, name in ((myc, MYCELIUM, "Mycelium (ternary, frozen topology)"),
                         (gru, GRU, "GRU (dense fp16)")):
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    es = [p[3] for p in pts]
    ax.errorbar(xs, ys, yerr=es, color=color, linewidth=1.8, marker="o",
                markersize=5.5, capsize=2.5, elinewidth=1.1, label=name, zorder=3)
    for label, x, y, _, dxm, dy in pts:
        ax.annotate(label, (x * dxm, y + dy), ha="center", fontsize=7.2, color=INK)

ax.set_xscale("log")
ax.set_xticks([32, 64, 128, 256, 512, 1024])
ax.set_xticklabels(["32", "64", "128", "256", "512", "1024"])
ax.set_xlim(24, 2100)
ax.set_ylim(0.48, 0.94)
ax.set_xlabel("Deployment size (KB, log scale)", fontsize=9, color=INK)
ax.set_ylabel("SHD test accuracy", fontsize=9, color=INK)
ax.tick_params(labelsize=8, colors=INK)
ax.grid(True, which="major", color="#dddddd", linewidth=0.6, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(MUTED)
ax.legend(fontsize=8, frameon=False, loc="lower right")

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "paper" / "fig_frontier.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"written: {out}")
