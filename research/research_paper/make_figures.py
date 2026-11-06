"""Regenerates the research paper's figures. Values come from research/paper/tables/ and research/paper/logs/ (E1-E7).

    python research/research_paper/make_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#1f5fae", "#eb6834", "#1baf7a", "#8f8e88"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def finish(ax, fig, name):
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)


# Fig. 1: gain by drowsy share so far (research/paper/tables/e3_statemix.md)
labels = ["0–10%", "10–25%", "25–50%", "50–75%", "75–100%"]
prior, cum, n = [1.22, 1.24, 7.70, 12.47, 1.18], [2.06, -0.71, 5.42, 9.88, 2.45], [22, 16, 16, 11, 6]
fig, ax = plt.subplots(figsize=(3.45, 2.2))
xs = range(5)
ax.bar([i - 0.2 for i in xs], prior, 0.36, color=BLUE, label="Prior alignment (κ = 30)")
ax.bar([i + 0.2 for i in xs], cum, 0.36, color=AQUA, label="Cumulative (α = 0.1)")
ax.set_xticks(list(xs))
ax.set_xticklabels([f"{a}\nn = {b}" for a, b in zip(labels, n)], color=INK, fontsize=6.6)
ax.set_xlabel("Drowsy share of labelled trials so far in the session")
ax.set_ylabel("Gain over no alignment (points)")
ax.legend(frameon=False, fontsize=6.4, loc="upper left", borderaxespad=0, handlelength=1)
finish(ax, fig, "fig1_statemix")

# Fig. 2: gain by trial position (research/paper/tables/e5_time_final.md; prior30 from research/paper/logs/person2.md)
bins = ["31–100", "101–200", "201–300", "301+"]
series = [("Whole session (not live)", [3.02, 5.81, 6.84, 0.14], GREY, "--"), ("Prior alignment (κ = 30)", [0.18, 5.11, 5.96, 1.92], BLUE, "-"), ("Frozen first 30 (α = 0.1)", [-0.28, 3.25, 3.23, -6.40], ORANGE, "-")]
fig, ax = plt.subplots(figsize=(3.45, 2.1))
for name, v, c, ls in series:
    ax.plot(range(4), v, color=c, ls=ls, lw=1.7, marker="o", ms=3.5, label=name)
ax.set_xticks(range(4))
ax.set_xticklabels(bins, color=INK)
ax.set_xlabel("Trial position in the session")
ax.set_ylabel("Gain over no alignment (points)")
ax.legend(frameon=False, fontsize=6.4, loc="lower left", borderaxespad=0)
finish(ax, fig, "fig2_time")
