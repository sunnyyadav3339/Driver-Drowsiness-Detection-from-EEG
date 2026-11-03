# Person 2 log (research paper: main method track)

## 2026-11-03 — E2 launched (`eeg-e2-prior`)
- *Method:* R_t = (n R_hat_t + kappa R_pop) / (n + kappa); R_pop is the training subjects' mean covariance, recomputed per fold, label-free. No shrinkage.
- *Code:* `research/src/run_prior.py`; self-test covers kappa limits, window 1 and causality (no future windows). InterpretableCNN, seeds 0-4, Kaggle T4.
- *Fixed before seeing results:* primary = prior30 vs none on balanced accuracy and AUC, from the 31st trial, 27 subjects. kappa 10 and 100 are sensitivity checks; `pop` is a control with no driver data. From-first-trial scores are secondary.

## 2026-11-03 — E2 results (`eeg-e2-prior`, 5 seeds, T4, 2,551 s) -> `research/paper/tables/e2_prior.csv`
- *Primary, passed:* prior30 77.37% vs none 73.59%, +3.78 points, 18/9 subjects, p = 0.039. AUC 0.820 vs 0.828, -0.009, p = 0.46 (no ranking gain; the gain is threshold calibration, as with the whole-session reference).
- *Sensitivity:* prior10 +3.42 (p = 0.065), prior100 +3.49 (p = 0.032). Stable across kappa.
- *Control:* pop (no driver data) +0.53 (13/14, p = 0.75), AUC -0.024 (p = 0.027). So the gain comes from the driver's own data, not from the prior alone.
- *From the first trial (secondary):* prior30 79.08% vs 75.18%, +3.90, 17/10, p = 0.034.
- *Time course (exploratory):* trials 1-30 -3.72 (2/7, n = 10, p = 0.13), 31-100 +0.18, 101-200 +5.11, 201-300 +5.96 (13/5, p = 0.0096), 301+ +1.92. **The prior does not help early on**, against the original idea; the gain arrives after about 100 trials.
- *Robustness, unplanned:* median gain +2.41. Without subject 52 (+34.3): +2.60, 17/9, p = 0.067; also without 49: +2.04, p = 0.11. The primary result leans on a few drivers.
- *Against other references (uncorrected):* vs Phase 3 cum, same training: +2.10, p = 0.032. vs E1 cum at a = 0.1: +1.11, p = 0.20, not significantly better. vs Phase 3 whole session: -0.48, 13/14, p = 0.90, so no detectable loss from going live.
- *Still needed:* competitors (E6, especially online BN adaptation) and the novelty search.
