"""Post-hoc analysis of Phase 2b: alert-anchored decision threshold (no retraining).

For each held-out subject the calibration window (the 30-epoch alignment reference) is re-drawn exactly as in
run_align.py. Assuming that window is alert, the decision threshold is set to the q-quantile of the window's
predicted drowsiness, so at most (1-q) of the calibration epochs would be flagged drowsy. Accuracy is then
measured on the remaining (non-calibration) epochs only, for both the default 0.5 threshold and the anchored one.

    python -m research.src.analyze_threshold research/results/eeg-phase2b-mix/results --out research/paper/tables/phase2b_threshold
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .data import find_dataset, load_sadt
from .run_align import subject_refs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--out", required=True)
    p.add_argument("--quantiles", default="0.8,0.9,0.95")
    a = p.parse_args()
    _, y, subj = load_sadt(find_dataset(None))
    pred = pd.read_csv(f"{a.run}/predictions.csv")
    qs = [float(q) for q in a.quantiles.split(",")]
    rows = []
    for (model, mode, seed), g in pred.groupby(["model", "mode", "seed"]):
        if ">" not in mode:
            continue
        refs = subject_refs(mode.split(">")[1], y, subj, seed)
        for s, gs in g.groupby("subject"):
            calib = np.isin(gs["index"].values, refs[s])
            pr, yy = gs.prob.values, gs.y.values
            ev = ~calib
            r = dict(model=model, mode=mode, seed=seed, subject=s, acc_default=((pr[ev] >= 0.5) == yy[ev]).mean())
            for q in qs:
                t = np.quantile(pr[calib], q)
                r[f"acc_q{q}"] = ((pr[ev] > t) == yy[ev]).mean()
            rows.append(r)
    df = pd.DataFrame(rows)
    cols = ["acc_default"] + [f"acc_q{q}" for q in qs]
    subj_mean = df.groupby(["model", "mode", "subject"])[cols].mean()
    lines = ["Alert-anchored threshold (post hoc, no retraining). Accuracy (%) on non-calibration epochs; mean over 5 seeds per "
             "subject, then over 11 subjects. p: Wilcoxon (two-sided) anchored vs default threshold across subjects.", "",
             "| model | train>test reference | default 0.5 | " + " | ".join(f"anchored q={q} (p)" for q in qs) + " |",
             "|---|---|---|" + "---|" * len(qs)]
    for (model, mode), g in subj_mean.groupby(level=[0, 1]):
        cells = [f"{100*g[c].mean():.2f} ({wilcoxon(g[c], g.acc_default).pvalue:.3f})" for c in cols[1:]]
        lines.append(f"| {model} | {mode} | {100*g.acc_default.mean():.2f} | " + " | ".join(cells) + " |")
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    (100 * subj_mean).round(2).to_csv(a.out + "_per_subject.csv")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
