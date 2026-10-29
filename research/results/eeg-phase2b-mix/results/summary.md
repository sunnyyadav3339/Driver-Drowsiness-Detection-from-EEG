Mean over seeds of the subject-averaged LOSO metric; ± is the std across seeds.

| model | mode | seeds | acc | macro-F1 | drowsy-F1 | AUC | acc std (subjects) |
|---|---|---|---|---|---|---|---|
| icnn_running | none | 5 | 73.68 ± 0.45 | 72.57 | 72.36 | 0.818 | 9.81 |
| icnn_running | all | 5 | 78.46 ± 0.55 | 78.32 | 77.90 | 0.837 | 6.90 |
| icnn_running | rand30 | 5 | 78.29 ± 1.21 | 78.08 | 77.43 | 0.837 | 7.26 |
| icnn_running | all>rand30 | 5 | 77.76 ± 0.99 | 77.52 | 77.23 | 0.831 | 6.91 |
| icnn_running | all>mix0 | 5 | 70.70 ± 0.71 | 69.23 | 74.87 | 0.837 | 5.11 |
| icnn_running | all>mix25 | 5 | 77.15 ± 0.61 | 76.87 | 77.92 | 0.838 | 6.50 |
| icnn_running | all>mix50 | 5 | 77.79 ± 0.89 | 77.59 | 77.48 | 0.831 | 6.95 |
| icnn_running | all>mix75 | 5 | 76.65 ± 0.47 | 76.36 | 75.27 | 0.823 | 6.78 |
| icnn_running | all>mix100 | 5 | 73.76 ± 1.33 | 72.95 | 69.96 | 0.808 | 7.87 |
| riemann_lr | none | 1 | 65.34 ± 0.00 | 62.02 | 61.97 | 0.748 | 15.96 |
| riemann_lr | all | 1 | 72.34 ± 0.00 | 72.21 | 73.04 | 0.759 | 13.76 |
| riemann_lr | rand30 | 5 | 71.60 ± 0.71 | 71.39 | 71.43 | 0.754 | 14.06 |
| riemann_lr | all>rand30 | 5 | 71.92 ± 0.48 | 71.73 | 71.79 | 0.759 | 13.65 |
| riemann_lr | all>mix0 | 5 | 67.02 ± 0.65 | 66.05 | 70.86 | 0.756 | 8.78 |
| riemann_lr | all>mix25 | 5 | 71.18 ± 0.69 | 70.88 | 73.04 | 0.757 | 12.66 |
| riemann_lr | all>mix50 | 5 | 72.07 ± 0.48 | 71.90 | 72.41 | 0.759 | 13.89 |
| riemann_lr | all>mix75 | 5 | 70.18 ± 0.14 | 69.94 | 69.02 | 0.757 | 12.95 |
| riemann_lr | all>mix100 | 5 | 65.80 ± 0.29 | 64.73 | 61.10 | 0.752 | 10.76 |
