Phase 3 (`results/eeg-phase3-online`): full Cao et al. 2019 dataset, 27 subjects, LOSO. Held-out driver aligned per session with a label-free reference: `first30` = first 30 trials, `cum` = all trials so far, `slide100` = last 100 trials, `all` = whole session (transductive). Scored on labelled trials from the 31st trial of each session on (same trials for every variant). Mean over subjects (ICNN: per-subject mean over 5 seeds).

| model | mode | bal-acc (%) | subject AUC | session AUC |
|---|---|---|---|---|
| icnn_running | none | 73.59 ± 13.30 | 0.828 | 0.822 |
| icnn_running | first30 | 52.55 ± 14.53 | 0.533 | 0.520 |
| icnn_running | cum | 75.26 ± 11.52 | 0.802 | 0.795 |
| icnn_running | slide100 | 71.76 ± 12.00 | 0.765 | 0.751 |
| icnn_running | all | 77.85 ± 11.89 | 0.834 | 0.826 |
| riemann_lr | none | 64.92 ± 16.60 | 0.739 | 0.754 |
| riemann_lr | first30 | 57.27 ± 14.91 | 0.615 | 0.631 |
| riemann_lr | cum | 64.95 ± 14.22 | 0.651 | 0.652 |
| riemann_lr | slide100 | 61.16 ± 13.88 | 0.612 | 0.626 |
| riemann_lr | all | 68.51 ± 15.24 | 0.728 | 0.733 |
| bandpower_lr | none | 68.30 ± 14.46 | 0.757 | 0.743 |
| bandpower_lr | first30 | 71.23 ± 10.92 | 0.763 | 0.751 |
| bandpower_lr | cum | 67.17 ± 10.71 | 0.725 | 0.711 |
| bandpower_lr | slide100 | 64.45 ± 10.63 | 0.688 | 0.681 |
| bandpower_lr | all | 69.61 ± 10.77 | 0.756 | 0.751 |

Paired comparisons across subjects (Δ = A − B; W/L = subjects where A is better/worse in bal-acc; Wilcoxon two-sided p).

| model | A vs B | Δ bal-acc | W/L | p | Δ subject AUC | p | Δ session AUC | p |
|---|---|---|---|---|---|---|---|---|
| icnn_running | first30 vs none | -21.04 | 3/24 | 0.0000 | -0.295 | 0.0000 | -0.302 | 0.0000 |
| icnn_running | cum vs none | +1.67 | 13/14 | 0.6617 | -0.026 | 0.1563 | -0.027 | 0.1199 |
| icnn_running | slide100 vs none | -1.83 | 8/19 | 0.1698 | -0.064 | 0.0031 | -0.071 | 0.0028 |
| icnn_running | all vs none | +4.26 | 19/8 | 0.0187 | +0.006 | 0.8532 | +0.003 | 0.5782 |
| icnn_running | cum vs first30 | +22.71 | 25/2 | 0.0000 | +0.269 | 0.0000 | +0.275 | 0.0000 |
| icnn_running | cum vs all | -2.59 | 5/22 | 0.0065 | -0.032 | 0.0061 | -0.031 | 0.0038 |
| riemann_lr | first30 vs none | -7.65 | 7/20 | 0.0200 | -0.125 | 0.0001 | -0.122 | 0.0001 |
| riemann_lr | cum vs none | +0.02 | 13/14 | 0.9529 | -0.089 | 0.0755 | -0.101 | 0.2099 |
| riemann_lr | slide100 vs none | -3.76 | 12/15 | 0.4410 | -0.128 | 0.0081 | -0.128 | 0.0275 |
| riemann_lr | all vs none | +3.58 | 17/10 | 0.0772 | -0.012 | 0.2776 | -0.021 | 0.2776 |
| riemann_lr | cum vs first30 | +7.68 | 17/10 | 0.0410 | +0.036 | 0.3957 | +0.021 | 0.3388 |
| riemann_lr | cum vs all | -3.56 | 13/14 | 0.1698 | -0.077 | 0.0626 | -0.080 | 0.1266 |
| bandpower_lr | first30 vs none | +2.93 | 14/13 | 0.3012 | +0.006 | 0.6915 | +0.008 | 0.5965 |
| bandpower_lr | cum vs none | -1.13 | 11/16 | 0.6277 | -0.032 | 0.2635 | -0.032 | 0.2752 |
| bandpower_lr | slide100 vs none | -3.85 | 10/17 | 0.1004 | -0.069 | 0.0393 | -0.062 | 0.0516 |
| bandpower_lr | all vs none | +1.31 | 15/12 | 0.4410 | -0.000 | 0.8532 | +0.008 | 0.5965 |
| bandpower_lr | cum vs first30 | -4.05 | 10/17 | 0.0140 | -0.038 | 0.1817 | -0.040 | 0.0125 |
| bandpower_lr | cum vs all | -2.43 | 5/22 | 0.0014 | -0.032 | 0.0160 | -0.040 | 0.0125 |

Time course: balanced accuracy (%) by trial position within the session (mean over subjects with both classes in that bin; n subjects in the header).

| model | mode | trials 31-100 (n=24) | trials 101-200 (n=18) | trials 201+ (n=11) |
|---|---|---|---|---|
| icnn_running | none | 72.55 | 70.33 | 69.92 |
| icnn_running | first30 | 56.60 | 52.16 | 54.26 |
| icnn_running | cum | 72.09 | 72.41 | 70.02 |
| icnn_running | slide100 | 70.64 | 66.82 | 70.67 |
| icnn_running | all | 76.44 | 73.18 | 70.17 |
| riemann_lr | none | 62.51 | 62.81 | 66.02 |
| riemann_lr | first30 | 52.18 | 58.94 | 56.38 |
| riemann_lr | cum | 55.95 | 63.34 | 66.52 |
| riemann_lr | slide100 | 54.18 | 58.76 | 63.17 |
| riemann_lr | all | 64.30 | 63.85 | 66.95 |
| bandpower_lr | none | 67.09 | 62.96 | 64.03 |
| bandpower_lr | first30 | 69.18 | 67.07 | 69.04 |
| bandpower_lr | cum | 63.35 | 65.35 | 69.07 |
| bandpower_lr | slide100 | 60.93 | 64.95 | 65.63 |
| bandpower_lr | all | 67.61 | 66.24 | 69.74 |
