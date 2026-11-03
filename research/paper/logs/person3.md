# Person 3 log (research paper: competitors track)

## 2026-11-03 — E6 launched (`eeg-e6-competitors`)
- *Competitors:* online BN adaptation with weight 10, 30, 100 and an EMA (momentum 0.1); Riemannian LR, unaligned and with prior30 re-centring. These are our own implementations of the general ideas, not reproductions of any published method.
- *Code:* `research/src/run_competitors.py`; self-test covers huge-weight equivalence with plain inference and causality. InterpretableCNN, seeds 0-4, Kaggle T4.
- *Fixed before seeing results:* primary = E2 prior30 vs bnadapt30, balanced accuracy and AUC, from the 31st trial, 27 subjects, same seeds. `none` must match E2.

## 2026-11-03 — E6 results (`eeg-e6-competitors`, 5 seeds, T4, 1,912 s) -> `research/paper/tables/e6_competitors.csv`
- *Check:* `none` matches E2 exactly (73.59%, AUC 0.828; largest per-subject difference 0).
- *Primary, passed:* prior30 77.37% vs bnadapt30 70.56%, +6.81 points, 23/4 subjects, p = 0.0008; without subject 52 +6.63, p = 0.0013. AUC 0.820 vs 0.802, +0.017, p = 0.19, not significant.
- *BN adaptation is worse than doing nothing:* bnadapt30 -3.03 (8/19, p = 0.025), AUC -0.026 (p = 0.045); bnadapt10 -3.50; bnadapt100 -2.54; EMA 59.58%, -14.01 (3/24, p < 0.001), AUC 0.651. The longer the memory, the less harm. Hypothesis, untested: updating statistics during the drive absorbs the slow state change that defines drowsiness.
- *Riemannian LR:* prior30 64.75% vs none 64.92%, -0.17, p = 0.99; AUC 0.701 vs 0.739. The prior gives no gain with this model, so the benefit is not model-independent.
- *CPU cost, median per window (Kaggle CPU, one window at a time):* whitening update 0.85 ms, BN update 0.64 ms, forward pass 0.59 ms.
- *Limits:* our own implementations, not reproductions of published test-time adaptation (for example Jang 2025). The BN model was not trained with per-session statistics while the aligned model was trained on aligned sessions, so the comparison is not of adaptation alone.
