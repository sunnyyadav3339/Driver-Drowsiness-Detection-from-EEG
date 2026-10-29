Phase 2 alignment probe (`results/eeg-phase2-align`). Accuracy (%) per method: mean over seeds per subject, then over the 11 LOSO subjects. Wilcoxon signed-rank (two-sided) across subjects.

| model | reference set | seeds | acc | macro-F1 | AUC | W/T/L vs none | p vs none |
|---|---|---|---|---|---|---|---|
| icnn_running | none | 3 | 73.95 | 72.81 | 0.824 | — |  |
| icnn_running | rand10 | 3 | 76.43 | 75.76 | 0.830 | 6/0/5 | 0.3652 |
| icnn_running | rand30 | 3 | 78.73 | 78.51 | 0.843 | 8/0/3 | 0.0322 |
| icnn_running | all | 3 | 78.54 | 78.37 | 0.839 | 8/1/2 | 0.0371 |
| icnn_running | alert10 | 3 | 60.04 | 49.49 | 0.745 | 4/0/7 | 0.0830 |
| icnn_running | alert30 | 3 | 57.65 | 45.35 | 0.625 | 3/0/8 | 0.0244 |
| riemann_lr | none | 1 | 65.34 | 62.02 | 0.748 | — |  |
| riemann_lr | rand10 | 3 | 69.79 | 69.27 | 0.750 | 7/0/4 | 0.1475 |
| riemann_lr | rand30 | 3 | 71.13 | 70.90 | 0.751 | 8/0/3 | 0.0674 |
| riemann_lr | all | 1 | 72.34 | 72.21 | 0.759 | 9/0/2 | 0.0420 |
| riemann_lr | alert10 | 1 | 65.41 | 61.23 | 0.733 | 6/0/5 | 0.8984 |
| riemann_lr | alert30 | 1 | 68.40 | 64.75 | 0.758 | 7/0/4 | 0.2061 |
| bandpower_lr | none | 1 | 67.14 | 65.50 | 0.743 | — |  |
| bandpower_lr | rand10 | 3 | 66.52 | 65.49 | 0.742 | 6/0/5 | 0.7002 |
| bandpower_lr | rand30 | 3 | 69.62 | 69.41 | 0.756 | 5/0/6 | 0.6377 |
| bandpower_lr | all | 1 | 71.01 | 70.88 | 0.762 | 6/0/5 | 0.1748 |
| bandpower_lr | alert10 | 1 | 65.29 | 63.24 | 0.692 | 5/0/6 | 0.4648 |
| bandpower_lr | alert30 | 1 | 64.83 | 62.86 | 0.685 | 4/0/7 | 0.4131 |

ICNN + Euclidean alignment vs the Phase 0 references (5-seed Phase 0 runs):

| method | acc | vs icnn_batch transductive (W/L, p) | vs icnn_running streaming (W/L, p) |
|---|---|---|---|
| icnn_running + EA (rand10) | 76.43 | 5/6, p=0.4648 | 6/5, p=0.1748 |
| icnn_running + EA (rand30) | 78.73 | 6/5, p=0.7002 | 9/2, p=0.0098 |
| icnn_running + EA (all) | 78.54 | 4/7, p=0.7646 | 9/2, p=0.0068 |

ICNN per subject (%):

|   subject |   none |   rand10 |   rand30 |   all |   alert10 |   alert30 |
|----------:|-------:|---------:|---------:|------:|----------:|----------:|
|         1 |   87.2 |     84   |     84.2 |  82.8 |      50   |      50   |
|         2 |   53   |     66.7 |     71.2 |  71.5 |      50   |      50   |
|         3 |   64.9 |     67.8 |     75.6 |  75.3 |      76.7 |      76   |
|         4 |   73.9 |     79.5 |     73.4 |  77.5 |      50   |      50   |
|         5 |   75.4 |     88.1 |     88.1 |  86.9 |      50   |      50   |
|         6 |   82.7 |     86.3 |     85.7 |  85.3 |      83.5 |      83.7 |
|         7 |   63.7 |     65.4 |     65.7 |  65   |      68.6 |      50   |
|         8 |   77.8 |     74   |     79.4 |  77.8 |      50   |      50   |
|         9 |   89.6 |     87.7 |     89.1 |  88.3 |      50   |      50   |
|        10 |   78.1 |     76.5 |     82.4 |  81.5 |      56.8 |      51.9 |
|        11 |   67.1 |     64.7 |     71.2 |  72   |      74.8 |      72.6 |
