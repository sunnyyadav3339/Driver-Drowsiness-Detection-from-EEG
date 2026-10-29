"""Combine run folders into paper tables: per-subject accuracy and Wilcoxon tests against a reference.

    python -m research.src.report research/results/eeg-phase0-fast/results [more run dirs] \
        --ref icnn_running:streaming --out research/paper/tables/phase0

Writes <out>_per_subject.csv and <out>.md.
"""
import argparse
import os

import pandas as pd
from scipy.stats import wilcoxon


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="folders containing per_subject.csv")
    p.add_argument("--ref", default="icnn_running:streaming", help="model:mode used as the reference in the tests")
    p.add_argument("--out", required=True, help="output path prefix")
    a = p.parse_args()

    df = pd.concat([pd.read_csv(os.path.join(r, "per_subject.csv")) for r in a.runs], ignore_index=True)
    df["method"] = df["model"] + ":" + df["mode"]
    # Average over seeds first, so each subject contributes one value per method.
    subj = df.groupby(["method", "subject"])[["acc", "f1_macro", "auc"]].mean().reset_index()
    wide = subj.pivot(index="subject", columns="method", values="acc")
    order = wide.mean().sort_values(ascending=False).index
    wide = wide[order]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    (100 * wide).round(2).to_csv(a.out + "_per_subject.csv")

    lines = [f"Accuracy per held-out subject (%), averaged over seeds. Wilcoxon signed-rank vs `{a.ref}` across the "
             f"{len(wide)} subjects (two-sided).", "",
             "| method | mean acc | macro-F1 | AUC | seeds | wins/ties/losses vs ref | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
    seeds = df.groupby("method").seed.nunique()
    for m in order:
        s = subj[subj.method == m]
        if m == a.ref:
            cmp, pval = "reference", ""
        else:
            diff = wide[m] - wide[a.ref]
            cmp = f"{(diff > 0).sum()}/{(diff == 0).sum()}/{(diff < 0).sum()}"
            pval = f"{wilcoxon(wide[m], wide[a.ref]).pvalue:.4f}" if (diff != 0).any() else "n/a"
        lines.append(f"| {m} | {100*s.acc.mean():.2f} | {100*s.f1_macro.mean():.2f} | {s.auc.mean():.3f} | {seeds[m]} | {cmp} | {pval} |")
    lines += ["", "Per subject:", "", (100 * wide).round(1).to_markdown()]
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
