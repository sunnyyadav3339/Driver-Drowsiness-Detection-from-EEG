Mean over seeds of the subject-averaged LOSO metric; ± is the std across seeds.

| model | mode | seeds | acc | macro-F1 | drowsy-F1 | AUC | acc std (subjects) |
|---|---|---|---|---|---|---|---|
| icnn_batch | streaming | 5 | 50.00 ± 0.00 | 33.33 | 66.67 | 0.498 | 0.00 |
| icnn_batch | transductive | 5 | 77.93 ± 1.34 | 77.50 | 76.22 | 0.835 | 7.60 |
| icnn_running | streaming | 5 | 73.68 ± 0.45 | 72.57 | 72.36 | 0.818 | 9.81 |
| eegnet | streaming | 5 | 63.85 ± 1.25 | 59.76 | 50.25 | 0.752 | 7.93 |
| bandpower_lr | streaming | 1 | 67.14 ± 0.00 | 65.50 | 67.70 | 0.743 | 10.55 |
| riemann_lr | streaming | 1 | 65.34 ± 0.00 | 62.02 | 61.97 | 0.748 | 15.96 |
