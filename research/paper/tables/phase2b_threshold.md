Alert-anchored threshold (post hoc, no retraining). Accuracy (%) on non-calibration epochs; mean over 5 seeds per subject, then over 11 subjects. p: Wilcoxon (two-sided) anchored vs default threshold across subjects.

| model | train>test reference | default 0.5 | anchored q=0.8 (p) | anchored q=0.9 (p) | anchored q=0.95 (p) |
|---|---|---|---|---|---|
| icnn_running | all>mix0 | 73.90 | 76.97 (0.240) | 73.77 (0.966) | 70.14 (0.465) |
| icnn_running | all>mix100 | 76.20 | 67.91 (0.024) | 65.23 (0.019) | 63.51 (0.010) |
| icnn_running | all>mix25 | 77.17 | 71.81 (0.001) | 61.76 (0.001) | 56.43 (0.001) |
| icnn_running | all>mix50 | 77.91 | 67.04 (0.001) | 59.70 (0.001) | 56.13 (0.001) |
| icnn_running | all>mix75 | 77.37 | 66.47 (0.004) | 61.92 (0.001) | 59.64 (0.001) |
| icnn_running | all>rand30 | 77.58 | 66.88 (0.001) | 59.45 (0.001) | 56.08 (0.001) |
| riemann_lr | all>mix0 | 70.49 | 66.76 (0.465) | 60.94 (0.067) | 57.73 (0.032) |
| riemann_lr | all>mix100 | 68.70 | 65.51 (0.240) | 63.25 (0.123) | 62.72 (0.102) |
| riemann_lr | all>mix25 | 71.81 | 65.30 (0.010) | 55.02 (0.001) | 50.23 (0.001) |
| riemann_lr | all>mix50 | 72.30 | 62.61 (0.003) | 55.05 (0.002) | 53.02 (0.002) |
| riemann_lr | all>mix75 | 70.54 | 63.60 (0.067) | 59.40 (0.007) | 57.94 (0.007) |
| riemann_lr | all>rand30 | 71.80 | 61.69 (0.005) | 54.52 (0.002) | 52.60 (0.002) |
