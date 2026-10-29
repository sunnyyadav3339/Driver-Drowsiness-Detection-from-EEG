Mean over seeds of the subject-averaged LOSO metric; ± is the std across seeds.

| model | mode | seeds | acc | macro-F1 | drowsy-F1 | AUC | acc std (subjects) |
|---|---|---|---|---|---|---|---|
| bandpower_lr | none | 1 | 67.14 ± 0.00 | 65.50 | 67.70 | 0.743 | 10.55 |
| bandpower_lr | all | 1 | 71.01 ± 0.00 | 70.88 | 71.49 | 0.762 | 11.65 |
| bandpower_lr | rand10 | 3 | 66.52 ± 1.90 | 65.49 | 64.60 | 0.742 | 10.25 |
| bandpower_lr | rand30 | 3 | 69.62 ± 1.57 | 69.41 | 70.13 | 0.756 | 10.74 |
| bandpower_lr | alert10 | 1 | 65.29 ± 0.00 | 63.24 | 58.37 | 0.692 | 13.04 |
| bandpower_lr | alert30 | 1 | 64.83 ± 0.00 | 62.86 | 57.03 | 0.685 | 14.06 |
| riemann_lr | none | 1 | 65.34 ± 0.00 | 62.02 | 61.97 | 0.748 | 15.96 |
| riemann_lr | all | 1 | 72.34 ± 0.00 | 72.21 | 73.04 | 0.759 | 13.76 |
| riemann_lr | rand10 | 3 | 69.79 ± 0.58 | 69.27 | 69.10 | 0.750 | 12.34 |
| riemann_lr | rand30 | 3 | 71.13 ± 0.49 | 70.90 | 70.87 | 0.751 | 14.18 |
| riemann_lr | alert10 | 1 | 65.41 ± 0.00 | 61.23 | 50.04 | 0.733 | 13.30 |
| riemann_lr | alert30 | 1 | 68.40 ± 0.00 | 64.75 | 55.43 | 0.758 | 13.23 |
| icnn_running | none | 3 | 73.95 ± 0.29 | 72.81 | 72.65 | 0.824 | 10.46 |
| icnn_running | all | 3 | 78.54 ± 0.69 | 78.37 | 77.94 | 0.839 | 6.92 |
| icnn_running | rand10 | 3 | 76.43 ± 1.09 | 75.76 | 74.26 | 0.830 | 8.86 |
| icnn_running | rand30 | 3 | 78.73 ± 1.37 | 78.51 | 77.78 | 0.843 | 7.44 |
| icnn_running | alert10 | 3 | 60.04 ± 0.52 | 49.49 | 31.42 | 0.745 | 12.56 |
| icnn_running | alert30 | 3 | 57.65 ± 0.10 | 45.35 | 22.89 | 0.625 | 12.37 |
