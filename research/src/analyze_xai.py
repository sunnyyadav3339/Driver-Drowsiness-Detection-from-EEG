"""Tables for the XAI case study (run_xai.py).

Every statistic is computed per subject (mean over seeds), then compared across subjects with two-sided Wilcoxon
signed-rank tests. With 11 subjects the smallest possible two-sided p is about 0.001.

    python -m research.src.analyze_xai research/results/eeg-xai-case/results --out research/paper/tables/xai_case
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METHODS = ["cam_fix", "saliency", "gradinput", "ig"]
NAMES = {"cam_fix": "CAM + tracing", "saliency": "Saliency", "gradinput": "Gradient x input", "ig": "Integrated gradients", "random": "Random"}
VARIANTS = {"icnn_batch": "whole-subject", "icnn_running": "running"}
LABELS = {0: "alert", 1: "drowsy"}


def wx(a, b=None):
    a = np.asarray(a, float) if b is None else np.asarray(a, float) - np.asarray(b, float)
    a = a[~np.isnan(a)]
    return wilcoxon(a).pvalue if len(a) > 1 and (a != 0).any() else float("nan")


def auc(g):
    """Area under the deletion curve, in units of the level range (lower = probability falls faster)."""
    g = g.sort_values("level")
    return np.trapezoid(g.p, g.level) / (g.level.max() - g.level.min())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    read = lambda n: pd.read_csv(os.path.join(a.run, n + ".csv"))
    L = []

    ps = read("per_subject")
    L += ["## Accuracy check against Phase 0", "", "| variant | acc (%) | AUC | seeds | subjects |", "|---|---|---|---|---|"]
    for v, g in ps.groupby("model", sort=False):
        m = g.groupby("subject")[["acc", "auc"]].mean()
        L.append(f"| {VARIANTS[v]} | {100*m.acc.mean():.2f} | {m.auc.mean():.3f} | {g.seed.nunique()} | {g.subject.nunique()} |")

    d = read("deletion")
    per = d.groupby(["model", "unit", "method", "seed", "subject"]).apply(auc, include_groups=False).groupby(["model", "unit", "method", "subject"]).mean().rename("auc").reset_index()
    per.to_csv(a.out + "_deletion_auc_per_subject.csv", index=False)
    L += ["", "## Deletion test: area under the P(predicted class) curve; lower is more faithful", "",
          "W/L: subjects where the method beats random / loses. p: Wilcoxon against random across subjects.", "",
          "| variant | unit | method | AUC | random AUC | W/L vs random | p |", "|---|---|---|---|---|---|---|"]
    for (v, u), g in per.groupby(["model", "unit"], sort=False):
        r = g[g.method == "random"].set_index("subject").auc
        for k in METHODS:
            m = g[g.method == k].set_index("subject").auc
            diff = m - r
            L.append(f"| {VARIANTS[v]} | {u} | {NAMES[k]} | {m.mean():.3f} | {r.mean():.3f} | {(diff<0).sum()}/{(diff>0).sum()} | {wx(m.values, r.loc[m.index].values):.4f} |")
    L += ["", "## Deletion curves (mean P(predicted class) over subjects)", ""]
    curves = d.groupby(["model", "unit", "method", "level", "subject"]).p.mean().groupby(["model", "unit", "method", "level"]).mean().reset_index()
    curves.to_csv(a.out + "_deletion_curves.csv", index=False)
    for (v, u), g in curves.groupby(["model", "unit"], sort=False):
        t = g.pivot(index="method", columns="level", values="p").round(3)
        L += [f"{VARIANTS[v]}, {u}:", "", t.to_markdown(), ""]

    s = read("sanity").groupby(["model", "method", "subject"]).spearman.mean().reset_index()
    L += ["## Sanity check: Spearman correlation with a randomly re-initialised model; lower is better", "", "| variant | method | mean rho | min | max |", "|---|---|---|---|---|"]
    for (v, k), g in s.groupby(["model", "method"], sort=False):
        L.append(f"| {VARIANTS[v]} | {NAMES[k]} | {g.spearman.mean():.3f} | {g.spearman.min():.3f} | {g.spearman.max():.3f} |")

    q = read("perturb")
    pq = q.groupby(["model", "kind", "part", "cls", "subject"]).dp.mean().reset_index()
    L += ["", "## Removing a band or channel group: change in P(true class), per true class", "",
          "Negative means the class relied on it. p: Wilcoxon against zero across subjects. p(alert vs drowsy): paired across subjects.", "",
          "| variant | removed | alert dP | p | drowsy dP | p | p(alert vs drowsy) |", "|---|---|---|---|---|---|---|"]
    for (v, kind, part), g in pq.groupby(["model", "kind", "part"], sort=False):
        al = g[g.cls == 0].set_index("subject").dp
        dr = g[g.cls == 1].set_index("subject").dp
        both = al.index.intersection(dr.index)
        L.append(f"| {VARIANTS[v]} | {part} | {al.mean():+.3f} | {wx(al.values):.4f} | {dr.mean():+.3f} | {wx(dr.values):.4f} | {wx(al[both].values, dr[both].values):.4f} |")

    c = read("context")
    pc = c.groupby(["context", "subject"])[["acc", "flips", "pred_drowsy", "map_corr"]].mean().reset_index()
    L += ["", "## Same windows, different normalisation context (whole-subject model)", "",
          "| statistics from | acc (%) | predictions changed (%) | predicted drowsy (%) | map similarity to baseline |", "|---|---|---|---|---|"]
    for cname in ["all", "random30", "alert_only", "drowsy_only"]:
        g = pc[pc.context == cname]
        if len(g):
            L.append(f"| {cname} | {100*g.acc.mean():.2f} | {100*g.flips.mean():.2f} | {100*g.pred_drowsy.mean():.2f} | {g.map_corr.mean():.3f} |")

    # Spatial results and stability, from the per-window channel maps.
    rows, stab, agree, between = [], [], [], []
    channels = None
    from .data import CHANNELS as channels
    from .data import find_dataset, load_sadt
    from .run_xai import PERIPHERAL_IDX
    _, y, subj = load_sadt(find_dataset(None))
    means = {}
    for f in sorted(glob.glob(os.path.join(a.run, "maps_*_seed*.npz"))):
        v, seed = os.path.basename(f)[5:-4].rsplit("_seed", 1)
        z = np.load(f)
        idx = z["index"]
        for k in METHODS:
            cm = z["channel_" + k]
            for s_ in np.unique(subj[idx]):
                for cls in (0, 1):
                    mask = (subj[idx] == s_) & (y[idx] == cls)
                    if mask.any():
                        means.setdefault((v, k, int(s_), cls), []).append(cm[mask].mean(0))
    for (v, k, s_, cls), arr in means.items():
        arr = np.stack(arr)
        rows.append(dict(model=v, method=k, subject=s_, cls=cls, peripheral=arr.mean(0)[PERIPHERAL_IDX].sum(), **{c: arr.mean(0)[i] for i, c in enumerate(channels)}))
        if len(arr) > 1:
            cc = np.corrcoef(arr)
            stab.append(dict(model=v, method=k, subject=s_, cls=cls, seed_corr=cc[np.triu_indices_from(cc, 1)].mean()))
    sp = pd.DataFrame(rows)
    sp.to_csv(a.out + "_channel_maps.csv", index=False)
    L += ["", "## Share of relevance on the 15 peripheral channels, every window (chance 0.50)", "",
          "p: Wilcoxon alert vs drowsy across subjects.", "", "| variant | method | alert | drowsy | p |", "|---|---|---|---|---|"]
    for (v, k), g in sp.groupby(["model", "method"], sort=False):
        al, dr = g[g.cls == 0].set_index("subject").peripheral, g[g.cls == 1].set_index("subject").peripheral
        both = al.index.intersection(dr.index)
        L.append(f"| {VARIANTS[v]} | {NAMES[k]} | {al.mean():.3f} | {dr.mean():.3f} | {wx(al[both].values, dr[both].values):.4f} |")
    if stab:
        st = pd.DataFrame(stab)
        L += ["", "## Stability: correlation of subject-class channel maps between seeds", "", "| variant | method | mean r | min |", "|---|---|---|---|"]
        for (v, k), g in st.groupby(["model", "method"], sort=False):
            L.append(f"| {VARIANTS[v]} | {NAMES[k]} | {g.seed_corr.mean():.3f} | {g.seed_corr.min():.3f} |")
    L += ["", "## Agreement between methods: correlation of subject-class channel maps", ""]
    for v, g in sp.groupby("model", sort=False):
        piv = g.set_index(["method", "subject", "cls"])[channels]
        M = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
        for m1 in METHODS:
            for m2 in METHODS:
                both = piv.loc[m1].index.intersection(piv.loc[m2].index)
                M.loc[m1, m2] = np.mean([np.corrcoef(piv.loc[m1].loc[i], piv.loc[m2].loc[i])[0, 1] for i in both])
        L += [f"{VARIANTS[v]}:", "", M.round(3).to_markdown(), ""]

    text = "\n".join(L) + "\n"
    open(a.out + ".md", "w", encoding="utf-8").write(text)
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
