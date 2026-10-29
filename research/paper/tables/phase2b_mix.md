Phase 2b (`results/eeg-phase2b-mix`): balanced accuracy (%) and AUC on the held-out subject's NON-calibration epochs; the unaligned baseline is scored on exactly the same epochs. Mean over seeds per subject, then over 11 subjects. Training subjects aligned with their whole sessions (`all>`), except `rand30` (train and test both 30 random epochs). Wilcoxon two-sided, aligned vs none, across subjects.

| model | test calibration window (30 epochs) | none bal-acc | aligned bal-acc | Δ | W/L | p | none AUC | aligned AUC |
|---|---|---|---|---|---|---|---|---|
| icnn_running | all>mix0 | 73.83 | 70.62 | -3.21 | 4/7 | 0.3652 | 0.820 | 0.836 |
| icnn_running | all>mix25 | 73.49 | 76.89 | +3.40 | 7/4 | 0.1475 | 0.813 | 0.834 |
| icnn_running | all>mix50 | 73.30 | 77.91 | +4.61 | 8/3 | 0.0244 | 0.818 | 0.831 |
| icnn_running | all>mix75 | 73.72 | 76.82 | +3.10 | 7/4 | 0.0674 | 0.820 | 0.822 |
| icnn_running | all>mix100 | 73.55 | 73.73 | +0.18 | 5/6 | 0.8984 | 0.815 | 0.808 |
| icnn_running | all>rand30 | 73.47 | 77.61 | +4.14 | 8/3 | 0.0537 | 0.814 | 0.828 |
| icnn_running | rand30 | 73.47 | 78.09 | +4.63 | 9/2 | 0.0186 | 0.814 | 0.833 |
| riemann_lr | all>mix0 | 65.57 | 67.18 | +1.61 | 7/4 | 0.5195 | 0.748 | 0.757 |
| riemann_lr | all>mix25 | 65.36 | 71.03 | +5.67 | 9/2 | 0.0537 | 0.742 | 0.758 |
| riemann_lr | all>mix50 | 65.36 | 72.30 | +6.94 | 8/2 | 0.0371 | 0.746 | 0.763 |
| riemann_lr | all>mix75 | 65.03 | 69.93 | +4.89 | 7/4 | 0.2402 | 0.743 | 0.757 |
| riemann_lr | all>mix100 | 65.32 | 65.80 | +0.48 | 6/5 | 0.7646 | 0.743 | 0.748 |
| riemann_lr | all>rand30 | 65.11 | 71.73 | +6.62 | 8/3 | 0.0537 | 0.739 | 0.756 |
| riemann_lr | rand30 | 65.11 | 71.37 | +6.26 | 7/4 | 0.0830 | 0.739 | 0.752 |
