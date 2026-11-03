"""E5, time to benefit: balanced accuracy by trial position in the session, for each alignment mode against no alignment.

Per subject: pool scored trials in a bin, over sessions and seeds' mean of the per-seed score; a subject counts in a bin only if both
classes occur there. Bins compare different subject sets, so paired tests (mode vs none, same subjects) are the evidence, not the raw curve.

    python -m research.src.analyze_time research/results/eeg-e1-shrink/results --alpha 0.1 --out research/paper/tables/e5_time
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score

BINS = [30, 100, 200, 300, 10_000]
LABELS = ["31-100", "101-200", "201-300", "301+"]
MODES = ["first30", "cum", "slide100", "all"]


def binned(df):
    df = df.assign(bin=pd.cut(df.trial, BINS, right=False, labels=LABELS))
    def score(g):
        return balanced_accuracy_score(g.y, g.prob >= 0.5) if g.y.nunique() == 2 else np.nan
    per = df.groupby(["mode", "seed", "subject", "bin"], observed=True)[["y", "prob"]].apply(score)
    return per.groupby(["mode", "subject", "bin"], observed=True).mean().rename("bacc").reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--alpha", default="0.1")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d = binned(pd.concat([pd.read_csv(os.path.join(a.run, "none", "predictions.csv")), pd.read_csv(os.path.join(a.run, f"a{a.alpha}", "predictions.csv"))]))
    d.to_csv(a.out + "_per_subject.csv", index=False)
    base = d[d["mode"] == "none"].set_index(["subject", "bin"]).bacc
    L = [f"E5 time to benefit, alpha = {a.alpha}. Balanced accuracy (%) by trial position; Delta and p are paired against none on the subjects with both classes in that bin.", "",
         "| mode | trials | n subjects | none | mode | Delta | W/L | p |", "|---|---|---|---|---|---|---|---|"]
    for m in MODES:
        for b in LABELS:
            g = d[(d["mode"] == m) & (d.bin == b)].set_index(["subject", "bin"]).bacc
            both = pd.concat([g, base], axis=1, join="inner", keys=["m", "n"]).dropna()
            if len(both) < 2:
                continue
            diff = both.m - both.n
            pv = wilcoxon(diff).pvalue if (diff != 0).any() else float("nan")
            L.append(f"| {m} | {b} | {len(both)} | {100*both.n.mean():.2f} | {100*both.m.mean():.2f} | {100*diff.mean():+.2f} | {(diff>0).sum()}/{(diff<0).sum()} | {pv:.4f} |")
    text = "\n".join(L) + "\n"
    open(a.out + ".md", "w", encoding="utf-8").write(text)
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
