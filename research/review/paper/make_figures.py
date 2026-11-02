"""Regenerates the review's figures in figures/ (vector PDF, plus PNG for checking).

Fig. 2 counts are computed from ../appraisal_matrix.csv.
Fig. 3 values are the headline results in the team's summaries, checked against the PDFs (see ../EVIDENCE_CHECK.md).

    python research/review/paper/make_figures.py
"""
import csv
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_MID, BLUE_LIGHT, ORANGE = "#1f5fae", "#7fa9dc", "#d9d8d3", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)


def box(ax, x, y, w, h, text, face="#ffffff", edge=MUTED, color=INK, weight="normal", size=5.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=face, ec=edge, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size, fontweight=weight, linespacing=1.25)


def link(ax, x0, y0, x1, y1):
    xm = (x0 + x1) / 2
    ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=MUTED, lw=0.8, solid_capstyle="round")


def taxonomy():
    """Fig. 1: how an explanation is produced (branches) and what it explains (bottom band)."""
    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.24, 0.90, 0.52, 0.09, "Interpretable and explainable drowsiness detection", BLUE, BLUE, "#ffffff", "bold", 7)
    branches = [
        (0.015, 0.215, "Intrinsic (by design)", ["Rules, fuzzy logic,\nthresholds", "Transparent features\n(EAR, MAR, band power)", "Model internals\n(LSTM states, attention)"]),
        (0.255, 0.215, "Post-hoc, model-specific", ["CAM, Grad-CAM,\nGrad-CAM++", "LRP, DeepLIFT,\nintegrated gradients", "Saliency, guided\nbackpropagation"]),
        (0.495, 0.215, "Post-hoc, model-agnostic", ["SHAP", "LIME, occlusion,\nfeature masking", "Partial dependence,\nICE"]),
        (0.735, 0.25, "Beyond attribution", ["Example-based\ninspection", "Counterfactuals", "Uncertainty\nestimates"]),
    ]
    for x, w, title, items in branches:
        box(ax, x, 0.72, w, 0.09, title, "#e8f0fa", BLUE, INK, "bold", 6.4)
        ax.plot([0.5, 0.5], [0.90, 0.855], color=MUTED, lw=0.8)
        ax.plot([0.5, x + w / 2, x + w / 2], [0.855, 0.855, 0.81], color=MUTED, lw=0.8)
        for i, item in enumerate(items):
            box(ax, x, 0.575 - i * 0.135, w, 0.115, item)
        ax.plot([x + w / 2, x + w / 2], [0.72, 0.69], color=MUTED, lw=0.8)
    ax.text(0.015, 0.215, "What the explanation refers to", color=MUTED, fontsize=7.2, fontweight="bold", va="center")
    targets = ["Spatial\n(channels, regions)", "Spectral\n(frequency bands)", "Temporal\n(time segments)", "Facial region\n(eyes, mouth, nose)", "Features\n(HRV, blink, steering)", "Decision logic\n(rules)"]
    w = 0.97 / 6 - 0.008
    for i, t in enumerate(targets):
        box(ax, 0.015 + i * (w + 0.0095), 0.045, w, 0.135, t, "#f4f3f0", GRID, INK, "normal", 5.5)
    save(fig, "fig1_taxonomy")


def appraisal():
    """Fig. 2: appraisal checks over the 27 counted studies."""
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "..", "appraisal_matrix.csv"))) if r["counted"] == "1"]
    n = len(rows)
    def count(field, full, part=()):
        f = sum(r[field] in full for r in rows)
        p = sum(r[field] in part for r in rows)
        return f, p, n - f - p
    data = [("Agrees with known physiology", count("physiology", {"yes"})),
            ("Tested on unseen drivers", count("subject_independent", {"yes"}, {"stated"})),
            ("Explanation tested for faithfulness", count("evaluated", {"faithfulness"}, {"partial"})),
            ("Code public", count("code", {"yes"}))]
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    colors = [BLUE, BLUE_MID, BLUE_LIGHT]
    text_colors = ["#ffffff", INK, INK]
    for y, (name, vals) in enumerate(reversed(data)):
        left = 0
        for v, c, tc in zip(vals, colors, text_colors):
            if v:
                ax.barh(y, v - 0.25, left=left + 0.125, color=c, height=0.62)
                ax.text(left + v / 2, y, str(v), ha="center", va="center", color=tc, fontsize=5.5 if v == 1 else 7.5, fontweight="bold")
            left += v
    ax.set_yticks(range(len(data))); ax.set_yticklabels([d[0] for d in reversed(data)], color=INK)
    ax.set_xlim(0, n); ax.set_xticks([0, 9, 18, 27]); ax.set_xlabel(f"Studies (n = {n})")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, ["Yes", "Partial or stated, not verifiable", "No, not shown or unclear"], ncol=1, frameon=False, fontsize=6.6,
              loc="lower center", bbox_to_anchor=(0.2, 1.0), handlelength=1, handleheight=1, borderaxespad=0.2)
    save(fig, "fig2_appraisal")
    return data


def validation():
    """Fig. 3: reported accuracy by test scheme; lines join results from the same study."""
    mixed = {"A5": 96.9, "A8": 97.47, "A10": 97.06, "B4": 86.05, "B6": 79.0, "C2": 97.2, "C5": 99.71, "C6": 99.88, "C7": 99.15, "C8": 99.0, "C9": 99.1}
    unseen = {"A1": 78.35, "A2": 73.22, "A3": 72.97, "A4": 77.16, "A5": 50.8, "A7": 74.54, "A8": 85.43, "B1": 80.1, "B3": 77.0, "C2": 92.4, "C4": 90.7, "C10": 65.2}
    jitter = {"A5": 0, "A8": 0, "C2": 0, "A10": -0.13, "C5": -0.12, "C6": 0.12, "C7": 0.13, "C8": -0.07, "C9": 0.07, "B4": 0, "B6": 0,
              "A1": 0.1, "A2": -0.1, "A3": 0.11, "A4": -0.11, "A7": -0.05, "B1": -0.08, "B3": 0.04, "C4": 0.06, "C10": 0}
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for k in set(mixed) & set(unseen):
        ax.plot([0, 1], [mixed[k], unseen[k]], color=ORANGE, lw=1.5, zorder=2, solid_capstyle="round")
    for x, d in ((0, mixed), (1, unseen)):
        for k, v in d.items():
            paired = k in mixed and k in unseen
            ax.scatter(x + jitter[k], v, s=26, color=ORANGE if paired else BLUE, edgecolor="#ffffff", linewidth=0.8, zorder=3)
        med = statistics.median(d.values())
        ax.plot([x - 0.25, x + 0.25], [med, med], color=INK, lw=1.2, zorder=1)
        ax.text(x + (-0.28 if x == 0 else 0.28), med, f"median\n{med:.1f}%", ha="right" if x == 0 else "left", va="center", fontsize=6.6, color=INK)
    for k, name, dy in (("A5", "Seo", -0.5), ("A8", "Zhou", -0.2), ("C2", "Rafy", 1.0)):
        ax.text(1.05, unseen[k] + dy, name, fontsize=6.6, color=MUTED, va="center", ha="left")
    ax.set_xlim(-0.72, 1.72); ax.set_ylim(45, 102)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Random or\nwithin-subject split", "Tested on\nunseen drivers"], color=INK)
    ax.set_ylabel("Reported accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_validation")
    return statistics.median(mixed.values()), statistics.median(unseen.values()), len(mixed), len(unseen)


REPO = os.path.join(HERE, "..", "..", "..")
XAI_TABLES = os.path.join(REPO, "research", "paper", "tables")
XAI_RUN = os.path.join(REPO, "research", "results", "eeg-xai-case", "results")
AQUA, VIOLET, GREY = "#1baf7a", "#6250d6", "#8f8e88"
METHOD_STYLE = [("gradinput", "Gradient × input", BLUE, "-"), ("ig", "Integrated gradients", VIOLET, "--"), ("cam_fix", "CAM + tracing", ORANGE, "-"),
                ("saliency", "Saliency", AQUA, "-"), ("random", "Random", GREY, ":")]
# Approximate 10-20 positions (nose up, ears at x = +/-1), for drawing only.
POS = dict(FP1=(-.31, .95), FP2=(.31, .95), F7=(-.81, .59), F3=(-.39, .51), FZ=(0, .5), F4=(.39, .51), F8=(.81, .59), FT7=(-.95, .31), FC3=(-.46, .26),
           FCZ=(0, .25), FC4=(.46, .26), FT8=(.95, .31), T3=(-1, 0), C3=(-.5, 0), CZ=(0, 0), C4=(.5, 0), T4=(1, 0), TP7=(-.95, -.31), CP3=(-.46, -.26),
           CPZ=(0, -.25), CP4=(.46, -.26), TP8=(.95, -.31), T5=(-.81, -.59), P3=(-.39, -.51), PZ=(0, -.5), P4=(.39, -.51), T6=(.81, -.59), O1=(-.31, -.95),
           OZ=(0, -1), O2=(.31, -.95))


def xai_tests():
    """Fig. 4: deletion curves and perturbation by class, whole-subject model (mean over subjects of the seed means)."""
    import pandas as pd

    curves = pd.read_csv(os.path.join(XAI_TABLES, "xai_case_deletion_curves.csv"))
    curves = curves[curves.model == "icnn_batch"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.45], wspace=0.42))
    for ax, unit, xlabel, title in ((axes[0], "point", "Share of points zeroed", "(a) Point deletion"), (axes[1], "channel", "Channels zeroed", "(b) Channel deletion")):
        for key, label, color, ls in METHOD_STYLE:
            g = curves[(curves.unit == unit) & (curves.method == key)].sort_values("level")
            ax.plot(g.level, g.p, color=color, ls=ls, lw=1.6, label=label, solid_capstyle="round")
        ax.set_xlabel(xlabel); ax.set_title(title, fontsize=8, color=INK, loc="left"); ax.set_ylim(0.1, 0.8)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("P(predicted class)")
    axes[0].legend(frameon=False, fontsize=5.4, loc="center right", bbox_to_anchor=(1.03, 0.36), handlelength=1.8, borderaxespad=0, labelspacing=0.25)

    q = pd.read_csv(os.path.join(XAI_RUN, "perturb.csv"))
    q = q[q.model == "icnn_batch"].groupby(["part", "cls", "subject"]).dp.mean().reset_index()
    parts = ["delta", "theta", "alpha", "beta", "gamma", "peripheral", "central"]
    TICKS = [0, 1, 2, 3, 4, 5.7, 7.0]
    ax = axes[2]
    for cls, color, label, off in ((0, BLUE, "Alert windows", -0.2), (1, ORANGE, "Drowsy windows", 0.2)):
        g = q[q.cls == cls].groupby("part").dp
        mean, sem = g.mean().reindex(parts), g.sem().reindex(parts)
        xs = [TICKS[i] + off for i in range(len(parts))]
        ax.bar(xs, mean, width=0.36, color=color, label=label, yerr=sem, error_kw=dict(ecolor=MUTED, lw=0.7))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(TICKS)
    ax.set_xticklabels(["δ", "θ", "α", "β", "γ", "Outer", "Centre"], color=INK, fontsize=6.4)
    ax.set_ylabel("Change in P(true class)"); ax.set_title("(c) Removing a band or channel group", fontsize=8, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=5.8, loc="lower left", handlelength=1, borderaxespad=0)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig4_xai_tests")


def scalp_maps():
    """Fig. 5: where two methods place relevance, per class (every window, mean over subjects)."""
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    from scipy.interpolate import griddata

    sp = pd.read_csv(os.path.join(XAI_TABLES, "xai_case_channel_maps.csv"))
    sp = sp[sp.model == "icnn_batch"]
    chs = list(POS)
    pts = np.array([POS[c] for c in chs])
    gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, 220), np.linspace(-1.1, 1.1, 220))
    inside = gx ** 2 + gy ** 2 <= 1.1 ** 2
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f3f0", "#a9c6ea", "#1f5fae", "#0d2f5c"])
    rows = [("cam_fix", "CAM + tracing"), ("gradinput", "Gradient × input")]
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.35), gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for r, (key, label) in enumerate(rows):
        vals = {cls: sp[(sp.method == key) & (sp.cls == cls)][chs].mean().values * 100 for cls in (0, 1)}
        vmin, vmax = min(v.min() for v in vals.values()), max(v.max() for v in vals.values())
        for c, (cls, name) in enumerate(((0, "Alert"), (1, "Drowsy"))):
            ax = axes[r, c]
            z = griddata(pts, vals[cls], (gx, gy), method="cubic")
            near = griddata(pts, vals[cls], (gx, gy), method="nearest")
            z = np.where(np.isnan(z), near, z)
            z = np.ma.array(z, mask=~inside)
            im = ax.imshow(z, extent=(-1.1, 1.1, -1.1, 1.1), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.add_patch(Circle((0, 0), 1.1, fill=False, ec=MUTED, lw=0.8))
            ax.plot([-0.1, 0, 0.1], [1.09, 1.2, 1.09], color=MUTED, lw=0.8)
            ax.scatter(pts[:, 0], pts[:, 1], s=2.5, color=INK, linewidths=0)
            ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.3); ax.axis("off")
            ax.set_title(f"{label}: {name.lower()}", fontsize=6.8, color=INK, pad=1)
        cb = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=5.5, length=2); cb.set_label("Relevance share (%)", fontsize=5.8)
    save(fig, "fig5_scalp")


if __name__ == "__main__":
    taxonomy()
    print(appraisal())
    print(validation())
    xai_tests()
    scalp_maps()
