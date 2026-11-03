"""E3: does the gain from online alignment depend on the state mix of the driver's windows so far, on real drives?

For each scored trial, share = drowsy / (alert + drowsy) among labelled trials of that session up to and including it. Labels are used
for this analysis only; the methods never see them. Per subject and bin: balanced accuracy of the mode minus no alignment, over seeds,
counting a subject only if both classes occur in the bin. Share rises with time on task, so this is association, not an isolated cause.

    python -m research.src.analyze_statemix --out research/paper/tables/e3_statemix
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score

EDGES = [0, 0.1, 0.25, 0.5, 0.75, 1.0001]
LABELS = ["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"]
SOURCES = {"prior30": ("research/results/eeg-e2-prior/results/predictions.csv", "prior30"),
           "cum (a=0.1)": ("research/results/eeg-e1-final/results/a0.1/predictions.csv", "cum"),
           "none": ("research/results/eeg-e6-competitors/results/predictions.csv", "none")}
TRIALS = "research/results/eeg-phase1-convert/results/trials.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t = pd.read_csv(TRIALS).sort_values(["session", "trial"])
    t["pos"] = t.groupby("session").cumcount()
    t["seen_d"], t["seen_l"] = (t.label == 1).groupby(t.session).cumsum(), (t.label >= 0).groupby(t.session).cumsum()
    t["share"] = t.seen_d / t.seen_l.clip(lower=1)
    t["bin"] = pd.cut(t.share, EDGES, right=False, labels=LABELS)
    frames = {}
    for name, (path, mode) in SOURCES.items():
        d = pd.read_csv(path)
        d = d[d["mode"] == mode].merge(t[["session", "trial", "pos", "bin", "share"]], on=["session", "trial"])
        d = d[d.pos >= 30]  # same trials as every other comparison
        frames[name] = (d.groupby(["seed", "subject", "bin"], observed=True)[["y", "prob"]]
                        .apply(lambda g: balanced_accuracy_score(g.y, g.prob >= 0.5) if g.y.nunique() == 2 else np.nan)
                        .groupby(["subject", "bin"], observed=True).mean())
    base = frames["none"]
    n_trials = d.groupby("bin", observed=True).size() / d.seed.nunique()
    L = ["E3: balanced accuracy (%) by drowsy share of the labelled trials so far in the session, from the 31st trial; paired against no alignment.", "",
         "| mode | drowsy share so far | n subjects | none | mode | Delta | W/L | p |", "|---|---|---|---|---|---|---|---|"]
    for name in ("prior30", "cum (a=0.1)"):
        for b in LABELS:
            x = pd.concat([frames[name].xs(b, level="bin"), base.xs(b, level="bin")], axis=1, join="inner", keys=["m", "n"]).dropna()
            if len(x) < 2:
                continue
            q = x.m - x.n
            pv = wilcoxon(q).pvalue if (q != 0).any() else float("nan")
            L.append(f"| {name} | {b} | {len(x)} | {100*x.n.mean():.2f} | {100*x.m.mean():.2f} | {100*q.mean():+.2f} | {(q>0).sum()}/{(q<0).sum()} | {pv:.4f} |")
    L += ["", "Scored trials per bin (one seed): " + ", ".join(f"{k}: {int(v)}" for k, v in n_trials.items())]
    text = "\n".join(L) + "\n"
    open(a.out + ".md", "w", encoding="utf-8").write(text)
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
