"""Phase 3 tables: online label-free alignment on the full Cao et al. dataset (27 subjects, LOSO).

Per subject, metrics are averaged over seeds (ICNN: 5 seeds; classical models: 1), then compared across the 27 subjects
with two-sided Wilcoxon signed-rank tests (pairs with an undefined AUC, i.e. a single-class subject, are dropped).
    subject AUC   AUC over all scored trials of the subject (sessions pooled)
    session AUC   AUC within each session, averaged over the subject's sessions (ranking within one drive only)
Time course: balanced accuracy by position of the trial in its session, mean over subjects with both classes in the bin.

    python -m research.src.analyze_phase3 research/results/eeg-phase3-online/results --out research/paper/tables/phase3_online
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

MODES = ["none", "first30", "cum", "slide100", "all"]
PAIRS = [("first30", "none"), ("cum", "none"), ("slide100", "none"), ("all", "none"), ("cum", "first30"), ("cum", "all")]
BINS = [30, 100, 200, 10_000]
warnings.filterwarnings("ignore", category=UserWarning)


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def pval(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return wilcoxon(a[ok], b[ok]).pvalue if ok.sum() > 1 and (a[ok] != b[ok]).any() else np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ps = pd.read_csv(f"{a.run}/per_subject.csv")
    pred = pd.read_csv(f"{a.run}/predictions.csv")
    pred["pos"] = pred.groupby(["model", "mode", "seed", "session"]).cumcount() + 30  # scoring starts at the 31st trial

    sess_auc = (pred.groupby(["model", "mode", "seed", "subject", "session"]).apply(lambda g: auc(g.y, g.prob), include_groups=False)
                .groupby(["model", "mode", "seed", "subject"]).mean().rename("sauc"))
    m = ps.set_index(["model", "mode", "seed", "subject"])[["bacc", "auc"]].join(sess_auc)
    m = m.groupby(["model", "mode", "subject"]).mean()
    m.reset_index().to_csv(a.out + "_per_subject.csv", index=False)
    models = list(dict.fromkeys(ps.model))
    n_subj = ps.subject.nunique()

    lines = [f"Phase 3 (`results/eeg-phase3-online`): full Cao et al. 2019 dataset, {n_subj} subjects, LOSO. Held-out driver "
             "aligned per session with a label-free reference: `first30` = first 30 trials, `cum` = all trials so far, "
             "`slide100` = last 100 trials, `all` = whole session (transductive). Scored on labelled trials from the 31st "
             "trial of each session on (same trials for every variant). Mean over subjects (ICNN: per-subject mean over 5 "
             "seeds).", "",
             "| model | mode | bal-acc (%) | subject AUC | session AUC |", "|---|---|---|---|---|"]
    for model in models:
        for mode in MODES:
            r = m.loc[(model, mode)]
            lines.append(f"| {model} | {mode} | {100*r.bacc.mean():.2f} ± {100*r.bacc.std():.2f} | {r.auc.mean():.3f} | "
                         f"{r.sauc.mean():.3f} |")
    lines += ["", "Paired comparisons across subjects (Δ = A − B; W/L = subjects where A is better/worse in bal-acc; "
              "Wilcoxon two-sided p).", "",
              "| model | A vs B | Δ bal-acc | W/L | p | Δ subject AUC | p | Δ session AUC | p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        for x, z in PAIRS:
            A, B = m.loc[(model, x)], m.loc[(model, z)]
            d = A.bacc - B.bacc
            lines.append(f"| {model} | {x} vs {z} | {100*d.mean():+.2f} | {(d>0).sum()}/{(d<0).sum()} | "
                         f"{pval(A.bacc.values, B.bacc.values):.4f} | {(A.auc-B.auc).mean():+.3f} | "
                         f"{pval(A.auc.values, B.auc.values):.4f} | {(A.sauc-B.sauc).mean():+.3f} | "
                         f"{pval(A.sauc.values, B.sauc.values):.4f} |")

    pred["bin"] = pd.cut(pred.pos, BINS, right=False, labels=[f"{lo+1}-{hi}" if hi < BINS[-1] else f"{lo+1}+" for lo, hi in zip(BINS, BINS[1:])])
    tb = (pred.groupby(["model", "mode", "seed", "subject", "bin"], observed=True)
          .apply(lambda g: balanced_accuracy_score(g.y, g.prob >= 0.5) if g.y.nunique() == 2 else np.nan, include_groups=False)
          .groupby(["model", "mode", "subject", "bin"], observed=True).mean()
          .groupby(["model", "mode", "bin"], observed=True).agg(["mean", "count"]))
    tb.reset_index().to_csv(a.out + "_time.csv", index=False)
    labels = list(pred["bin"].cat.categories)
    lines += ["", "Time course: balanced accuracy (%) by trial position within the session (mean over subjects with both "
              "classes in that bin; n subjects in the header).", "",
              "| model | mode | " + " | ".join(f"trials {b} (n={int(tb.loc[(models[0], 'none', b), 'count'])})" for b in labels) + " |",
              "|---|---|" + "---|" * len(labels)]
    for model in models:
        for mode in MODES:
            lines.append(f"| {model} | {mode} | " + " | ".join(f"{100*tb.loc[(model, mode, b), 'mean']:.2f}" for b in labels) + " |")

    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
