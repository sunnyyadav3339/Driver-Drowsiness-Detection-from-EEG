## Accuracy check against Phase 0

| variant | acc (%) | AUC | seeds | subjects |
|---|---|---|---|---|
| whole-subject | 77.93 | 0.835 | 5 | 11 |
| running | 73.68 | 0.818 | 5 | 11 |

## Deletion test: area under the P(predicted class) curve; lower is more faithful

W/L: subjects where the method beats random / loses. p: Wilcoxon against random across subjects.

| variant | unit | method | AUC | random AUC | W/L vs random | p |
|---|---|---|---|---|---|---|
| whole-subject | channel | CAM + tracing | 0.561 | 0.707 | 11/0 | 0.0010 |
| whole-subject | channel | Saliency | 0.641 | 0.707 | 10/1 | 0.0020 |
| whole-subject | channel | Gradient x input | 0.501 | 0.707 | 11/0 | 0.0010 |
| whole-subject | channel | Integrated gradients | 0.500 | 0.707 | 11/0 | 0.0010 |
| whole-subject | point | CAM + tracing | 0.542 | 0.667 | 11/0 | 0.0010 |
| whole-subject | point | Saliency | 0.580 | 0.667 | 10/1 | 0.0020 |
| whole-subject | point | Gradient x input | 0.185 | 0.667 | 11/0 | 0.0010 |
| whole-subject | point | Integrated gradients | 0.178 | 0.667 | 11/0 | 0.0010 |
| running | channel | CAM + tracing | 0.584 | 0.737 | 11/0 | 0.0010 |
| running | channel | Saliency | 0.659 | 0.737 | 10/1 | 0.0020 |
| running | channel | Gradient x input | 0.526 | 0.737 | 11/0 | 0.0010 |
| running | channel | Integrated gradients | 0.525 | 0.737 | 11/0 | 0.0010 |
| running | point | CAM + tracing | 0.563 | 0.695 | 11/0 | 0.0010 |
| running | point | Saliency | 0.583 | 0.695 | 10/1 | 0.0020 |
| running | point | Gradient x input | 0.152 | 0.695 | 11/0 | 0.0010 |
| running | point | Integrated gradients | 0.161 | 0.695 | 11/0 | 0.0010 |

## Deletion curves (mean P(predicted class) over subjects)

whole-subject, channel:

| method    |   0.0 |   1.0 |   2.0 |   3.0 |   5.0 |   10.0 |
|:----------|------:|------:|------:|------:|------:|-------:|
| cam_fix   | 0.752 | 0.662 | 0.606 | 0.567 | 0.528 |  0.506 |
| gradinput | 0.752 | 0.626 | 0.547 | 0.504 | 0.455 |  0.445 |
| ig        | 0.752 | 0.624 | 0.545 | 0.502 | 0.455 |  0.444 |
| random    | 0.752 | 0.745 | 0.737 | 0.728 | 0.707 |  0.657 |
| saliency  | 0.752 | 0.728 | 0.701 | 0.669 | 0.618 |  0.575 |

whole-subject, point:

| method    |   0.0 |   0.01 |   0.02 |   0.05 |   0.1 |   0.2 |   0.3 |   0.5 |
|:----------|------:|-------:|-------:|-------:|------:|------:|------:|------:|
| cam_fix   | 0.752 |  0.702 |  0.668 |  0.599 | 0.545 | 0.513 | 0.519 | 0.528 |
| gradinput | 0.752 |  0.448 |  0.331 |  0.223 | 0.181 | 0.161 | 0.156 | 0.156 |
| ig        | 0.752 |  0.447 |  0.327 |  0.214 | 0.171 | 0.153 | 0.15  | 0.15  |
| random    | 0.752 |  0.751 |  0.749 |  0.743 | 0.727 | 0.686 | 0.644 | 0.58  |
| saliency  | 0.752 |  0.725 |  0.696 |  0.639 | 0.596 | 0.567 | 0.556 | 0.546 |

running, channel:

| method    |   0.0 |   1.0 |   2.0 |   3.0 |   5.0 |   10.0 |
|:----------|------:|------:|------:|------:|------:|-------:|
| cam_fix   |  0.78 | 0.694 | 0.637 | 0.598 | 0.551 |  0.517 |
| gradinput |  0.78 | 0.661 | 0.582 | 0.537 | 0.481 |  0.454 |
| ig        |  0.78 | 0.66  | 0.581 | 0.535 | 0.48  |  0.456 |
| random    |  0.78 | 0.774 | 0.766 | 0.758 | 0.738 |  0.686 |
| saliency  |  0.78 | 0.759 | 0.735 | 0.697 | 0.634 |  0.575 |

running, point:

| method    |   0.0 |   0.01 |   0.02 |   0.05 |   0.1 |   0.2 |   0.3 |   0.5 |
|:----------|------:|-------:|-------:|-------:|------:|------:|------:|------:|
| cam_fix   |  0.78 |  0.732 |  0.699 |  0.633 | 0.575 | 0.534 | 0.535 | 0.54  |
| gradinput |  0.78 |  0.484 |  0.352 |  0.21  | 0.149 | 0.121 | 0.115 | 0.113 |
| ig        |  0.78 |  0.485 |  0.354 |  0.21  | 0.152 | 0.13  | 0.127 | 0.127 |
| random    |  0.78 |  0.779 |  0.777 |  0.771 | 0.757 | 0.719 | 0.675 | 0.593 |
| saliency  |  0.78 |  0.754 |  0.725 |  0.661 | 0.61  | 0.57  | 0.552 | 0.536 |

## Sanity check: Spearman correlation with a randomly re-initialised model; lower is better

| variant | method | mean rho | min | max |
|---|---|---|---|---|
| whole-subject | CAM + tracing | 0.174 | 0.065 | 0.242 |
| whole-subject | Gradient x input | 0.513 | 0.496 | 0.531 |
| whole-subject | Integrated gradients | 0.515 | 0.498 | 0.532 |
| whole-subject | Saliency | 0.128 | 0.108 | 0.145 |
| running | CAM + tracing | 0.167 | 0.074 | 0.254 |
| running | Gradient x input | 0.510 | 0.493 | 0.526 |
| running | Integrated gradients | 0.511 | 0.493 | 0.526 |
| running | Saliency | 0.126 | 0.098 | 0.142 |

## Removing a band or channel group: change in P(true class), per true class

Negative means the class relied on it. p: Wilcoxon against zero across subjects. p(alert vs drowsy): paired across subjects.

| variant | removed | alert dP | p | drowsy dP | p | p(alert vs drowsy) |
|---|---|---|---|---|---|---|
| whole-subject | alpha | +0.091 | 0.0010 | -0.167 | 0.0010 | 0.0010 |
| whole-subject | beta | +0.008 | 0.2783 | -0.005 | 0.5195 | 0.4131 |
| whole-subject | delta | +0.066 | 0.0010 | -0.089 | 0.0010 | 0.0010 |
| whole-subject | gamma | -0.036 | 0.0010 | +0.027 | 0.0010 | 0.0010 |
| whole-subject | theta | +0.156 | 0.0010 | -0.240 | 0.0010 | 0.0010 |
| whole-subject | central | +0.195 | 0.0010 | -0.405 | 0.0010 | 0.0010 |
| whole-subject | peripheral | -0.007 | 0.5771 | -0.102 | 0.0029 | 0.0537 |
| running | alpha | +0.100 | 0.0010 | -0.172 | 0.0010 | 0.0010 |
| running | beta | +0.013 | 0.0322 | -0.006 | 0.2783 | 0.0830 |
| running | delta | +0.069 | 0.0010 | -0.081 | 0.0020 | 0.0010 |
| running | gamma | -0.029 | 0.0010 | +0.026 | 0.0010 | 0.0010 |
| running | theta | +0.152 | 0.0010 | -0.225 | 0.0010 | 0.0010 |
| running | central | +0.208 | 0.0010 | -0.383 | 0.0010 | 0.0010 |
| running | peripheral | -0.004 | 0.4648 | -0.071 | 0.0244 | 0.2402 |

## Same windows, different normalisation context (whole-subject model)

| statistics from | acc (%) | predictions changed (%) | predicted drowsy (%) | map similarity to baseline |
|---|---|---|---|---|
| all | 77.93 | 0.00 | 45.59 | 1.000 |
| random30 | 77.55 | 4.80 | 45.39 | 0.890 |
| alert_only | 68.65 | 24.55 | 68.21 | 0.502 |
| drowsy_only | 72.12 | 14.59 | 31.02 | 0.694 |

## Share of relevance on the 15 peripheral channels, every window (chance 0.50)

p: Wilcoxon alert vs drowsy across subjects.

| variant | method | alert | drowsy | p |
|---|---|---|---|---|
| whole-subject | CAM + tracing | 0.640 | 0.456 | 0.0068 |
| whole-subject | Saliency | 0.517 | 0.517 | 0.3652 |
| whole-subject | Gradient x input | 0.519 | 0.462 | 0.0098 |
| whole-subject | Integrated gradients | 0.521 | 0.460 | 0.0098 |
| running | CAM + tracing | 0.629 | 0.475 | 0.0186 |
| running | Saliency | 0.518 | 0.518 | 0.2402 |
| running | Gradient x input | 0.513 | 0.467 | 0.0537 |
| running | Integrated gradients | 0.514 | 0.465 | 0.0537 |

## Stability: correlation of subject-class channel maps between seeds

| variant | method | mean r | min |
|---|---|---|---|
| whole-subject | CAM + tracing | 0.563 | 0.077 |
| whole-subject | Saliency | 0.754 | 0.630 |
| whole-subject | Gradient x input | 0.719 | 0.495 |
| whole-subject | Integrated gradients | 0.730 | 0.513 |
| running | CAM + tracing | 0.603 | 0.461 |
| running | Saliency | 0.800 | 0.725 |
| running | Gradient x input | 0.770 | 0.582 |
| running | Integrated gradients | 0.775 | 0.565 |

## Agreement between methods: correlation of subject-class channel maps

whole-subject:

|           |   cam_fix |   saliency |   gradinput |    ig |
|:----------|----------:|-----------:|------------:|------:|
| cam_fix   |     1     |      0.54  |       0.814 | 0.81  |
| saliency  |     0.54  |      1     |       0.757 | 0.746 |
| gradinput |     0.814 |      0.757 |       1     | 0.997 |
| ig        |     0.81  |      0.746 |       0.997 | 1     |

running:

|           |   cam_fix |   saliency |   gradinput |    ig |
|:----------|----------:|-----------:|------------:|------:|
| cam_fix   |     1     |      0.538 |       0.809 | 0.803 |
| saliency  |     0.538 |      1     |       0.77  | 0.765 |
| gradinput |     0.809 |      0.77  |       1     | 0.997 |
| ig        |     0.803 |      0.765 |       0.997 | 1     |

