"""Phase 2b main table: every aligned variant vs no alignment, scored on the SAME non-calibration epochs.

The calibration window (alignment reference of the held-out subject) is re-drawn exactly as in run_align.py and
removed from scoring, for the aligned model and for the unaligned baseline alike (same seed, same epochs).
Balanced accuracy is reported because removing a one-class window unbalances the remaining epochs.

    python -m research.src.analyze_mix research/results/eeg-phase2b-mix/results --out research/paper/tables/phase2b_mix
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .data import find_dataset, load_sadt
from .run_align import subject_refs

ORDER = ["all>mix0", "all>mix25", "all>mix50", "all>mix75", "all>mix100", "all>rand30", "rand30"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    _, y, subj = load_sadt(find_dataset(None))
    pred = pd.read_csv(f"{a.run}/predictions.csv")
    rows = []
    for model in pred.model.unique():
        pm = pred[pred.model == model]
        base = pm[pm["mode"] == "none"]
        for mode in [m for m in ORDER if m in set(pm["mode"])]:
            for seed in sorted(pm[pm["mode"] == mode].seed.unique()):
                refs = subject_refs(mode.split(">")[-1], y, subj, seed)
                g = pm[(pm["mode"] == mode) & (pm.seed == seed)]
                b = base[base.seed == (seed if seed in set(base.seed) else base.seed.min())]
                for s, gs in g.groupby("subject"):
                    bs = b[b.subject == s].set_index("index").loc[gs["index"]]
                    ev = ~np.isin(gs["index"].values, refs[s])
                    yy = gs.y.values[ev]
                    for name, pr in (("aligned", gs.prob.values[ev]), ("none", bs.prob.values[ev])):
                        rows.append(dict(model=model, mode=mode, seed=seed, subject=s, method=name,
                                         bacc=balanced_accuracy_score(yy, pr >= 0.5), auc=roc_auc_score(yy, pr)))
    df = pd.DataFrame(rows)
    m = df.groupby(["model", "mode", "method", "subject"])[["bacc", "auc"]].mean()
    lines = ["Phase 2b (`results/eeg-phase2b-mix`): balanced accuracy (%) and AUC on the held-out subject's NON-calibration "
             "epochs; the unaligned baseline is scored on exactly the same epochs. Mean over seeds per subject, then over 11 "
             "subjects. Training subjects aligned with their whole sessions (`all>`), except `rand30` (train and test both "
             "30 random epochs). Wilcoxon two-sided, aligned vs none, across subjects.", "",
             "| model | test calibration window (30 epochs) | none bal-acc | aligned bal-acc | Δ | W/L | p | none AUC | aligned AUC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model in df.model.unique():
        for mode in [x for x in ORDER if (model, x) in {(k[0], k[1]) for k in m.index}]:
            al, no = m.loc[(model, mode, "aligned")], m.loc[(model, mode, "none")]
            d = al.bacc - no.bacc
            lines.append(f"| {model} | {mode} | {100*no.bacc.mean():.2f} | {100*al.bacc.mean():.2f} | {100*d.mean():+.2f} | "
                         f"{(d>0).sum()}/{(d<0).sum()} | {wilcoxon(al.bacc, no.bacc).pvalue:.4f} | {no.auc.mean():.3f} | {al.auc.mean():.3f} |")
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
