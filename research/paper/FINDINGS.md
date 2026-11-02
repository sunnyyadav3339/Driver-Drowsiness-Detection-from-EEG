# Findings (paper-ready)

Curated claims we expect to state in the paper. Each one gives its evidence (run folder or notebook entry) and status. Only promote a claim to **confirmed** once it comes from a full 5-seed run under the protocol in [../RESEARCH_PLAN.md](../RESEARCH_PLAN.md#4-experiment-protocol-fixed-before-running-anything).

| # | Claim | Evidence | Status |
|---|---|---|---|
| F1 | InterpretableCNN's published LOSO accuracy (78.35%) relies on BatchNorm statistics computed over the whole held-out subject; scored one epoch at a time, the model outputs a constant (50.00%, AUC 0.498). | `results/eeg-phase0-fast`, 5 seeds; LAB_NOTEBOOK 2026-10-29 | **confirmed** |
| F2 | With running-stats BN (streaming-valid), InterpretableCNN reaches 73.68 ± 0.45% LOSO accuracy, 4.25 pts below its transductive score (9/11 subjects, Wilcoxon p = 0.042). | `results/eeg-phase0-fast`, 5 seeds | **confirmed** |
| F3 | Validation-subject early stopping selects nearly untrained models in cross-subject drowsiness decoding (EEGNet: median best epoch 2/100), which lowers accuracy (63.85%). | `results/eeg-phase0-fast` | preliminary: check with DeltaGateNet and a fixed-budget rule |
| F5 | Under LOSO, DeltaGateNet (published 83.21% on 5-fold subject splits) reaches 64.74%, below streaming InterpretableCNN on 10/11 subjects (p = 0.032); it is near-perfectly inverted on subject 2 (AUC 0.15) and badly miscalibrated on subject 10 (AUC 0.98, acc 65.7%). | `results/eeg-phase0-dgn-s0`, seed 0 | preliminary (1 seed) |
| F6 | Euclidean alignment with 30 unlabeled epochs (90 s) per subject raises streaming InterpretableCNN from 73.68% to 78.73% LOSO (9/11 subjects, p = 0.0098), matching the transductive per-batch-BN score (77.93%, p = 0.70) while scoring each later epoch on its own. | `results/eeg-phase2-align`, 3 seeds; re-checked on non-calibration epochs with 5 seeds in `eeg-phase2b-mix` (rand30: +4.63 bal-acc, 9/2, p = 0.019) | **confirmed for mixed-state windows** (see F8) |
| F7 | Alignment to an alert-only reference collapses the CNN to near-constant output (57.65%). Naive 'deviation from alert baseline' does not transfer across subjects. | `results/eeg-phase2-align` | preliminary |
| F8 | The gain from alignment depends on the calibration window's state mix (inverted U): 0% drowsy −3.2, 25% +3.4, 50% +4.6 (p = 0.024), 75% +3.1, 100% +0.2 bal-acc points for the CNN; the same shape for Riemannian LR (peak +6.9, p = 0.037). An alert-only 'first 90 s' window does not work. | `results/eeg-phase2b-mix`, 5 seeds, [tables/phase2b_mix.md](tables/phase2b_mix.md) | **confirmed** (on SADT-2022) |
| F9 | The 11 SADT-2022 subjects correspond to 11 specific sessions of Cao et al. 2019 (table in LAB_NOTEBOOK 2026-10-30); re-deriving labels from RT reproduces SADT's labels for 98.7% of matched epochs. The full release has 27,192 trials from 27 subjects, with natural (unbalanced) class proportions. | `results/eeg-phase1-convert` | confirmed (dataset fact) |
| F10 | On the full Cao et al. dataset (27 subjects, LOSO), whole-session label-free EA raises streaming InterpretableCNN from 73.59% to 77.85% balanced accuracy (19/8 subjects, p = 0.019), replicating F6, but subject AUC is unchanged (0.828 → 0.834): the gain is threshold calibration for the new driver, not better ranking. Online references (cumulative, last-100, first-30) did not help, but this is confounded: the pre-processed EEG is rank-deficient and short references are near-singular (cond up to 1e15), so the unregularised whitener blows up the signal. | `results/eeg-phase3-online`, ICNN 5 seeds, [tables/phase3_online.md](tables/phase3_online.md) | transductive part confirmed; online part **pending regularised rerun** |
| F11 | On every held-out window, InterpretableCNN's decisions depend on the other windows used for normalisation: statistics from alert windows alone change 24.55% of predictions with the same weights; 30 random windows change 4.80%. | `results/eeg-xai-case`, 5 seeds | confirmed (one model, one dataset) |
| F12 | All four attribution methods beat random deletion (at least 10/11 subjects), including the published CAM + tracing method, but the strongest on deletion (gradient x input, IG) are weakest on a randomisation sanity check (rho 0.51 vs 0.13-0.17). | `results/eeg-xai-case`, [tables/xai_case.md](tables/xai_case.md) | confirmed; point deletion for gradient x input is partly tautological |
| F13 | The published reading that alert decisions rest on peripheral artefacts is supported by map location (64.0% vs 45.6% peripheral, p = 0.007) but not by perturbation: zeroing peripheral channels (-0.007, p = 0.58) and removing beta (+0.008, p = 0.28) barely change P(alert). Removing theta, alpha or delta strongly lowers P(drowsy). | `results/eeg-xai-case` | descriptive; see caveats in LAB_NOTEBOOK |
| F4 | Riemannian tangent-space accuracy varies enormously between subjects (43.8–92.4%), pointing to subject-specific covariance shift. | `results/eeg-phase0-fast` | observation: motivates direction B |

## Paper outline (working)

1. **Introduction:** drowsy driving; EEG; the cross-subject problem; the streaming constraint.
2. **Related work:** see RESEARCH_PLAN §2 and `references.bib`.
3. **Data:** Cao et al. 2019 (full) and Cui et al. 2022 (11-subject subset); labeling via local/global RT.
4. **Method:** candidate: streaming CNN + label-free Euclidean alignment from a short calibration window (F6); robustness to the calibration window's state mix is open (Phase 2b).
5. **Experimental protocol:** LOSO, streaming inference, 5 seeds, Wilcoxon tests (RESEARCH_PLAN §4).
6. **Results:** baselines (Phase 0); ablations; full-data benchmark.
7. **Analysis:** spatial and spectral interpretability; latency.
8. **Limitations.**

## Figures and tables to produce

- [ ] Table: baselines under LOSO (streaming vs transductive), mean ± std over 5 seeds (Phase 0). Now [tables/phase0.md](tables/phase0.md); DeltaGateNet has 1 seed
- [ ] Figure: per-subject accuracy, baselines vs ours
- [ ] Figure: RT distribution and the labeling thresholds (Phase 1)
- [ ] Table: ablations A/B/C
- [ ] Figure: interpretability maps
