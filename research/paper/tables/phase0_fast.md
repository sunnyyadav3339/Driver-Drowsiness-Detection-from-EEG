Accuracy per held-out subject (%), averaged over seeds. Wilcoxon signed-rank vs `icnn_running:streaming` across the 11 subjects (two-sided).

| method | mean acc | macro-F1 | AUC | seeds | wins/ties/losses vs ref | Wilcoxon p |
|---|---|---|---|---|---|---|
| icnn_batch:transductive | 77.93 | 77.50 | 0.835 | 5 | 9/0/2 | 0.0420 |
| icnn_running:streaming | 73.68 | 72.57 | 0.818 | 5 | reference |  |
| bandpower_lr:streaming | 67.14 | 65.50 | 0.743 | 1 | 3/0/8 | 0.0674 |
| riemann_lr:streaming | 65.34 | 62.02 | 0.748 | 1 | 3/0/8 | 0.1748 |
| eegnet:streaming | 63.85 | 59.76 | 0.752 | 5 | 1/0/10 | 0.0049 |
| icnn_batch:streaming | 50.00 | 33.33 | 0.498 | 5 | 0/0/11 | 0.0010 |

Per subject:

|   subject |   icnn_batch:transductive |   icnn_running:streaming |   bandpower_lr:streaming |   riemann_lr:streaming |   eegnet:streaming |   icnn_batch:streaming |
|----------:|--------------------------:|-------------------------:|-------------------------:|-----------------------:|-------------------:|-----------------------:|
|         1 |                      83.8 |                     85.3 |                     48.4 |                   60.6 |               71.9 |                     50 |
|         2 |                      69.4 |                     55.3 |                     61.4 |                   92.4 |               54.7 |                     50 |
|         3 |                      80.3 |                     65.7 |                     73.3 |                   58.7 |               49.9 |                     50 |
|         4 |                      77.6 |                     73.6 |                     63.5 |                   71.6 |               63.1 |                     50 |
|         5 |                      87.7 |                     76.2 |                     75.9 |                   86.2 |               64.8 |                     50 |
|         6 |                      84   |                     82.2 |                     82.5 |                   87.3 |               67.7 |                     50 |
|         7 |                      67.1 |                     61.4 |                     52   |                   46.1 |               58.6 |                     50 |
|         8 |                      78.6 |                     77   |                     68.2 |                   56.8 |               60.5 |                     50 |
|         9 |                      89.4 |                     89   |                     82.5 |                   60.5 |               69.6 |                     50 |
|        10 |                      68.1 |                     76.9 |                     66.7 |                   54.6 |               79.8 |                     50 |
|        11 |                      71.3 |                     67.8 |                     64.2 |                   43.8 |               61.8 |                     50 |
