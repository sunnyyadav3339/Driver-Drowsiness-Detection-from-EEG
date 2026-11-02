# Lab Notebook

A dated, append-only record of what was tried, why, what happened and what was decided. It's the raw material for the paper's Methods, Results and Limitations sections. Paper-ready numbers go in [FINDINGS.md](FINDINGS.md); every number here links to a run folder in `research/results/`.

Entry format: **date — title**, then *Goal*, *Setup* (run id, commit, config), *Result*, *Decision / next*.

---

## 2026-09-16 — Reproduce InterpretableCNN (Cui et al. 2022)
- *Goal:* reproduce the paper's LOSO result on the 11-subject / 2022-sample SADT release.
- *Setup:* `model.ipynb` (Kaggle GPU); Adam 1e-3, batch 50, 11 epochs, torch seed 0; per-batch BatchNorm (`track_running_stats=False`), as in the paper.
- *Result:* mean LOSO accuracy **78.03%** (paper: 78.35%). Per subject: 85.6, 69.7, 78.7, 78.4, 88.0, 84.3, 66.7, 78.8, 89.2, 73.2, 65.9.
- *Decision:* reproduction accepted.

## 2026-10-15 — Finding: the paper's BatchNorm is transductive
- *Goal:* explain why the live demo always showed 54.1%.
- *Setup:* scored each held-out subject one epoch at a time vs as a whole-subject batch (LOSO, seed 0, CPU).
- *Result:*

  | Mode | Mean LOSO acc |
  |---|---|
  | Per-batch BN, whole held-out subject as one batch | 78.05% |
  | Per-batch BN, one epoch at a time | 50.00% (output constant) |
  | Per-batch BN, rolling buffer of last 30 epochs | 52.66% |
  | Running-stats BN, one epoch at a time | 73.79% |

  With batch size 1, BN zero-centres every channel, and global average pooling then outputs the BN bias, so the prediction is constant. **The published 78.35% therefore depends on normalizing with statistics from the whole test subject (a form of transductive test-time adaptation).** The streaming-valid number is about 73–74%.
- *Decision:* the demo uses running-stats BN (`training/train.py`, LOSO 73.39%, F1 0.729, AUC 0.828, seed 0). For the paper: report streaming and transductive numbers separately, and use the ~4.6-point gap as motivation for label-free subject alignment (direction B).

## 2026-10-29 — Literature review and research plan
- 2024–2026 literature reviewed; plan in [../RESEARCH_PLAN.md](../RESEARCH_PLAN.md). Open-access PDFs are in `papers/`; BibTeX is in [references.bib](references.bib).
- Key comparison points: DeltaGateNet (2026) reports 83.21% inter-subject on SADT-2022, but with 5-fold 60/20/20 subject splits rather than LOSO. Online TTA (Jang 2025) reports F1 81.73 ± 20.2 under LOSO.
- DeltaGateNet code reviewed (MIT): AdamW 1e-4, batch 32, 200 epochs, checkpoint chosen by best validation-subject loss, single seed 2026, no input normalization.
- *Decision:* experiments run on Kaggle (GPU, user already has the dataset there as `sunnyydv602/eeg-dataset`). Start with Phase 0: all baselines under one LOSO protocol.

## 2026-10-29 — Phase 0 launched: baselines under one LOSO protocol
- *Goal:* a fair baseline table on SADT-2022 (11 subjects, 2022 epochs), with streaming and transductive inference reported separately.
- *Code:* `research/src/run_phase0.py` (models in `research/src/models.py`, classical in `research/src/classical.py`). Launched with `research/kaggle/launch.py`, which embeds the code in a private Kaggle kernel; the code's sha256 is recorded in each run's `env.json`.
- *Protocol:* LOSO over 11 subjects; seeds 0–4 for deep models (classical ones are deterministic, seed 0 only). Models that select a checkpoint (EEGNet, DeltaGateNet) hold out 2 training subjects for validation, drawn with `default_rng(1000*seed + test_subject)`; the InterpretableCNN variants use the paper's fixed 11 epochs with no validation. No tuning on test subjects. Inputs are raw µV with no normalization (all original papers do the same).
- *Recipes:* InterpretableCNN Adam 1e-3/b50/11 ep; EEGNet-8,2 Adam 1e-3/b32/100 ep; DeltaGateNet AdamW 1e-4 (wd 0.01)/b32/200 ep (official code); band-power: Welch log power in 5 bands × 30 channels + standardization + LR; Riemannian: OAS covariance + tangent space + standardization + LR.
- *Runs:* Kaggle kernels `eeg-phase0-fast` (all models except DeltaGateNet, 5 seeds) and `eeg-phase0-dgn-s0` (DeltaGateNet, seed 0; also a timing run). GPU T4 ×2. Code sha256 `351ed7813800…`, base commit 326cefb (research code not yet committed).
- *Local smoke test* (2 subjects, ~2% of epochs, CPU) passed for all six models. Accuracy at that budget means nothing.

## 2026-10-29 — Phase 0 results: fast baselines (`eeg-phase0-fast`)
- *Run:* `research/results/eeg-phase0-fast/` on a Tesla T4 (CUDA confirmed in `env.json`), torch 2.11.0, pyriemann 0.12. 232/232 folds, 24 min. Tables: [tables/phase0_fast.md](tables/phase0_fast.md).
- *Result* (mean over seeds of subject-averaged LOSO accuracy; Wilcoxon across 11 subjects vs icnn_running):

  | Method | Acc | Macro-F1 | AUC | Seeds | vs ref (W/L), p |
  |---|---|---|---|---|---|
  | InterpretableCNN, per-batch BN, **transductive** | 77.93 ± 1.34 | 77.50 | 0.835 | 5 | 9/2, p = 0.042 |
  | InterpretableCNN, running BN, **streaming** (ref) | 73.68 ± 0.45 | 72.57 | 0.818 | 5 | — |
  | Band power + LR | 67.14 | 65.50 | 0.743 | 1 | 3/8, p = 0.067 |
  | Riemannian tangent space + LR | 65.34 | 62.02 | 0.748 | 1 | 3/8, p = 0.175 |
  | EEGNet-8,2 | 63.85 ± 1.25 | 59.76 | 0.752 | 5 | 1/10, p = 0.005 |
  | InterpretableCNN, per-batch BN, streaming | 50.00 | 33.33 | 0.498 | 5 | 0/11, p = 0.001 |

- *Observations:*
  1. **Transductive vs streaming gap confirmed over 5 seeds:** 77.93 vs 73.68 (+4.25 pts, better on 9 of 11 subjects, p = 0.042). The published 78.35% reproduces only in transductive mode; per-batch BN collapses to chance when streaming.
  2. **Validation-based checkpointing hurts EEGNet:** the best validation-subject loss came at a median of epoch 2 (range 1–92), so the selected models are barely trained. Cross-subject validation loss starts rising almost immediately, and training on 8 instead of 10 subjects costs data. Our EEGNet (63.85%) is below Jang et al.'s source-only F1 of 67.23%. **DeltaGateNet uses the same selection rule**, so expect the same effect.
  3. **Caveat on the reference's own protocol:** InterpretableCNN's 11 epochs were taken from the paper's Fig. 2a, which plotted accuracy on the held-out subjects, so that choice also saw test data. Every published number on this benchmark has some selection issue; we need a single rule for every model (see decision).
  4. **Huge between-subject variance for feature models:** Riemannian accuracy ranges 43.8–92.4% across subjects, and on subject 2 it gets 92.4% where InterpretableCNN gets 55.3%. This is strong evidence of subject-specific covariance shift, which is what Riemannian re-centring / Euclidean alignment (direction B) corrects. The models are also complementary per subject, so an ensemble may help.
- *Decision / next:*
  - Wait for DeltaGateNet (seed 0) timing before launching its seeds 1–4.
  - Add a second, uniform selection rule for every deep model: a **fixed epoch budget chosen by nested LOSO on training subjects only**, so no model's training length is chosen on test data. Report both rules.
  - Quick Phase 2 probe: Riemannian / Euclidean alignment of each subject using an unlabeled reference set (first *k* epochs; for the 11-subject release epoch order within the session is unknown, so first check whether indices follow time).

## 2026-10-29 — Phase 2 probe prepared (not launched): label-free subject alignment
- *Data check:* in SADT-2022 each subject's epochs are stored **sorted by label** (all alert, then all drowsy; 2 label runs per subject). Within each class block, adjacent epochs are only weakly more similar than random pairs (ratio 0.66–1.04), so the original time order is mostly lost. Any "first k epochs" reference is therefore guaranteed alert, which makes an alert-baseline calibration look optimistic on this release (the Phase 1 full dataset, with real timestamps, is needed to test it properly).
- *Code:* `research/src/run_align.py`. Reference sets: none / all epochs / k random epochs / first k alert epochs (k = 10, 30). Alignment: Euclidean alignment for icnn_running, Riemannian re-centring for riemann_lr, baseline-subtracted log band power for bandpower_lr. `launch.py` gained `--accelerator none` (CPU kernel, no GPU quota).
- *Smoke test* (subject 2 only, seed 0, local CPU; one subject means nothing statistically): icnn none 60.6 → all 70.5 / rand10 72.0; bandpower none 61.4 → all 83.3 / rand10 84.1; riemann none 92.4 → all 86.4. alert10 gave a strongly shifted decision threshold (icnn: acc 50.0 but AUC 0.94).
- *Status:* launch blocked because the local Kaggle login expired. Planned slug `eeg-phase2-align`, launched after DeltaGateNet seed 0 finishes.

## 2026-10-29 — Phase 0 results: DeltaGateNet seed 0 (`eeg-phase0-dgn-s0`)
- *Run:* `research/results/eeg-phase0-dgn-s0/`, Tesla T4, official recipe (AdamW 1e-4, b32, 200 epochs, best validation-subject loss). 11 folds, ~46 min each (2,624–2,957 s), about 8.5 h in total. Combined table: [tables/phase0.md](tables/phase0.md).
- *Result:* LOSO accuracy **64.74%** (macro-F1 63.04, AUC 0.766), seed 0 only; worse than icnn_running on 10/11 subjects (Wilcoxon p = 0.032). Per subject: 84.0, **18.2**, 81.3, 61.5, 74.1, 71.1, 54.9, 73.9, 72.9, 65.7, 54.4. Best epochs: 40, 72, 124, 27, 33, 47, 46, 91, 66, 31, **2** (max 124 out of 200).
- *Observations:*
  1. Under LOSO, DeltaGateNet is far below its published 83.21% (5-fold subject splits). It is in line with the other baselines and below the simple streaming InterpretableCNN.
  2. **Subject 2 is anti-correlated** (acc 18.2%, AUC 0.15): what the model learned from other drivers is inverted for this driver, whereas Riemannian LR gets 92.4% on the same subject. Strong evidence of subject-specific shift.
  3. **Subject 10: AUC 0.98 but accuracy 65.7%.** Good ranking, wrong decision threshold for this driver. Label-free alignment or recalibration should help.
  4. Validation-based stopping did not collapse here as it did for EEGNet, except on subject 11 (best epoch 2, 54.4%).
- *Decision / next:* seeds 1–4 are not worth ~34 GPU-hours as the official recipe stands. If they are run, cap training at 130 epochs: seed 0 never selected a later epoch, so its result would not change. Launch the Phase 2 alignment probe next (CPU kernel).

## 2026-10-29 — Phase 2 probe results: label-free subject alignment (`eeg-phase2-align`)
- *Run:* `research/results/eeg-phase2-align/`, Kaggle T4 (version 2; version 1 was the same code on CPU and was superseded), about 11 min. Code sha256 `df82ee263602…`. ICNN uses 3 seeds; classical models use 3 seeds only for the random references. Table: [tables/phase2_align.md](tables/phase2_align.md).
- *Result (accuracy, Wilcoxon vs no alignment, 11 subjects):*

  | Model | none | rand10 | **rand30** | all | alert10 | alert30 |
  |---|---|---|---|---|---|---|
  | InterpretableCNN (running BN) + Euclidean alignment | 73.95 | 76.43 (p=0.37) | **78.73** (8/3, p=0.032) | 78.54 (p=0.037) | 60.04 | 57.65 (p=0.024, worse) |
  | Riemannian LR + re-centring | 65.34 | 69.79 | 71.13 (p=0.067) | 72.34 (9/2, p=0.042) | 65.41 | 68.40 |
  | Band power LR + baseline subtraction | 67.14 | 66.52 | 69.62 | 71.01 (p=0.17) | 65.29 | 64.83 |

- *Observations:*
  1. **Headline:** Euclidean alignment from just **30 unlabeled epochs (90 s of EEG)** brings streaming InterpretableCNN from 73.68 to **78.73%**, better on 9/11 subjects than the 5-seed streaming reference (p = 0.0098). That matches the transductive per-batch-BN number (77.93; 6/5 subjects, p = 0.70), so a short unlabeled calibration recovers the whole transductive gain while every later epoch is still scored on its own. Subject 2 (the anti-correlated subject) goes from 53.0 to 71.2.
  2. Ten epochs (30 s) is not enough (76.43, n.s.); 30 epochs is about as good as the whole session (78.54).
  3. **The alert-only reference fails badly** for the CNN (57.65–60.04; the model outputs a near-constant "alert" on 8/11 subjects). Whitening by an alert-only covariance changes the input distribution differently for each subject, depending on how far that subject's drowsy state is from their alert state, so the decision threshold no longer transfers. This is the naive version of the "alert-baseline" idea; it would need a different normalisation (e.g. alignment with the mixed reference plus an alert-anchored offset) or a recalibrated threshold.
  4. Every reference used here comes from the subject's own session with its balanced 50/50 class mix (SADT-2022 is balanced by design). **Major caveat:** a real drive's first 90 s is mostly alert, and we just saw that an alert-only reference breaks the method. How sensitive the result is to the reference's class mix is the key open question.
- *Decision / next:*
  - Phase 2b: sweep the **class mix of the 30-epoch reference** (0/25/50/75/100% drowsy) and test a running/online EA (reference = every epoch seen so far, updated continuously). Report EA with 5 seeds.
  - Online EA needs the true time order, which the 11-subject release has lost (see the earlier entry). That makes Phase 1 (the original Cao et al. dataset, with timestamps and RT) necessary for a deployment-realistic evaluation.

## 2026-10-29 — Phase 2b launched: class-mix sensitivity of the alignment reference (`eeg-phase2b-mix`)
- *Goal:* find out whether the gain in F6 survives a calibration window that is not 50/50 alert/drowsy (a real drive starts mostly alert).
- *Change to `run_align.py`:* new rule `mix<p>` (30 random epochs, p% drowsy; labels only build the mix). New syntax `A>B` aligns training subjects with rule A and the held-out subject with rule B. Training subjects can use their whole sessions offline, so `all>mix<p>` is the deployment-realistic setting.
- *Configs:* none, all, rand30, all>rand30, all>mix{0,25,50,75,100}; InterpretableCNN (running BN) + EA and Riemannian LR + re-centring; seeds 0–4; Kaggle T4. Code sha256 `74cc4913b886…`.

## 2026-10-29 — Phase 2b results: alignment needs a mixed-state calibration window
- *Run:* `research/results/eeg-phase2b-mix/` (T4, 5 seeds). Raw summary: `results/summary.md`. **Fair analysis** (`research/src/analyze_mix.py`): the held-out subject's 30 calibration epochs are re-drawn exactly and removed from scoring, the unaligned model is scored on the same epochs, and balanced accuracy is used because removing a one-class window unbalances the rest. Table: [tables/phase2b_mix.md](tables/phase2b_mix.md).
- *Result (InterpretableCNN + EA, balanced accuracy on non-calibration epochs, Δ vs none on the same epochs):*

  | Calibration window drowsy share | 0% | 25% | 50% | 75% | 100% | random |
  |---|---|---|---|---|---|---|
  | Δ bal-acc (W/L, p) | −3.21 (4/7, 0.37) | +3.40 (7/4, 0.15) | **+4.61** (8/3, 0.024) | +3.10 (7/4, 0.067) | +0.18 (5/6, 0.90) | +4.14 (0.054); train+test rand30: **+4.63** (9/2, 0.019) |

  Riemannian LR shows the same inverted U: +1.6, +5.7, **+6.9** (p = 0.037), +4.9, +0.5.
- *Observations:*
  1. **The gain from label-free alignment depends on the calibration window containing both states.** A single-state window (all alert or all drowsy) gives no gain; 25–75% drowsy gives +3 to +7 points. The re-centring point has to sit *between* the two states for a decision boundary learned on other subjects to transfer.
  2. So the realistic "first 90 s of the drive, assumed alert" calibration **does not work** (−3.2, n.s.). The F6 headline holds only for mixed windows.
  3. With an alert-only window, AUC rises slightly (0.820 → 0.836) while balanced accuracy falls: ranking is kept and the threshold moves. A post-hoc alert-anchored threshold (quantile of the window's scores; `analyze_threshold.py`, [tables/phase2b_threshold.md](tables/phase2b_threshold.md)) did **not** fix it (best +3.1 at q = 0.8, p = 0.24).
  4. Methodological note: the Phase 2 numbers (F6) included the calibration epochs in scoring. On non-calibration epochs, train+test rand30 still gives +4.63 (p = 0.019), so F6 stands.
- *Implication / decision:* the deployable version is **online (cumulative) alignment**: re-estimate the reference from every epoch seen so far. Over a real drive with drowsiness episodes, it gradually acquires a mixed reference. Testing that needs true time order and natural state proportions, which the 11-subject release lacks. → **Phase 1 (original Cao et al. dataset with timestamps and RT) is now the critical path.** The paper's angle: when does unsupervised alignment help streaming drowsiness detection, and what does the calibration window need to contain?

## 2026-10-30 — Phase 1: converting the original Cao et al. 2019 dataset
- *User choices:* the pre-processed full release (figshare 7666055, CC BY 4.0, 16.6 GB, 62 sessions / 27 subjects), downloaded by Kaggle directly; keep per-trial RT, trial timing, Cui-style labels, 30 ch × 128 Hz × 3 s windows; long RTs kept and flagged.
- *Format (probe `eeg-phase1-probe`):* EEGLAB .set + .fdt, 30 channels in the same order as SADT-2022, 500 Hz, ~60 min per session (s01_051017m: 3,623 s, 199 trials). Events: 251/252 deviation onset (left/right), 253 response onset, 254 response offset.
- *Converter:* `research/src/convert_cao.py`. Window = 3 s before deviation onset, resampled 500 → 128 Hz (resample_poly 32/125). RT = 253 − 251/252. global RT = mean of valid RTs in the preceding 90 s (inclusive). alert RT = 5th percentile of the session's valid RTs. Labels per Wei et al. 2018 / Cui et al. 2022 (1.5× / 2.5×).
- *Issue found in test (`eeg-phase1-convert-test`):* 45/199 trials in s01 had RT = 0.016 s, which made alert RT 0.016 s and labelled 155 trials drowsy. The dataset tutorial (figshare file "Tutorial Data Analysis…", §III.6) says: "Reject trials with extremely short RT (RT < 0.3 sec)". Applied: such trials get `valid=False` and label −1, and are excluded from alert/global RT. The tutorial's trajectory-based "zigzag" rejection cannot be reproduced (no trajectory in the release); this is a limitation. Trials with RT > 10 s get `long_rt=True` and are kept.
- *Test caveat:* s01 is not one of Cui's 11 subjects (best correlation with any SADT epoch 0.59), so the full run's SADT match check is what validates the converter.
- *Full run:* `eeg-phase1-convert` launched on a T4 at the user's request (code sha256 `c086e2c20102…`).

## 2026-10-30 — Phase 1 results: full Cao et al. dataset converted (`eeg-phase1-convert`)
- *Run:* Kaggle T4, 746 s for all 62 sessions. Metadata in `research/results/eeg-phase1-convert/results/` (`sessions.csv`, `trials.csv`, `sadt_match.csv`). The windows themselves (`sessions/*.npz`, about 1.2 GB) stay on Kaggle as the output of kernel `sunnyydv602/eeg-phase1-convert`, which later kernels read as an input.
- *Totals:* 62 sessions, 27 subjects, sessions 41–118 min; **27,192 trials**: 8,941 alert, 3,947 drowsy, about 13.5k transitional, 800 invalid (RT < 0.3 s), 711 with RT > 10 s. Per-session alert RT 0.45–0.97 s. Some sessions have almost no drowsy trials (s49_080522n: 0, s55_090930n: 0), so the class balance is far from SADT's 50/50.
- *Validation against SADT-2022:* every SADT subject's epochs map almost entirely onto **one** converted session (share 94–100%):

  | SADT subject | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Cao session | s01_061102n | s05_061101n | s22_090825n | s31_061103n | s35_070322n | s41_080520m | s42_070105n | s43_070205n | s44_070325n | s45_070307n | s53_090918n |

  Our labels agree with SADT's on **98.7%** of matched epochs (correlation > 0.8, n = 1,130). The waveforms are not identical, though: median best correlation is 0.82 and only 3 epochs exceed 0.99. Random cross-trial pairs give about 0.2–0.6 (s01 test), so these are the same trials with a processing difference. Likely causes: a small window offset or a different resampling filter in Cui's pipeline.
- *Conclusion:* the converter identifies the same sessions and reproduces the labels, but not Cui's exact samples. For the paper, SADT-2022 results and full-dataset results are reported as separate benchmarks.
- *Open item:* estimate the time lag between matched windows (cross-correlation) to explain the 0.82.

## 2026-10-30 — Phase 3 launched: full dataset with online alignment (`eeg-phase3-online`)
- *Goal:* the deployable version of F6/F8. Align the held-out driver online, from the EEG seen so far, on real drive timelines.
- *Code:* `research/src/run_phase3.py` (loader `load_cao` in `data.py`; the launcher gained `--kernel-sources`, so the kernel reads the `eeg-phase1-convert` output directly). Code sha256 `c020676f47a7…`, Kaggle T4.
- *Protocol:* LOSO over 27 subjects (all of a subject's sessions held out). Training uses alert/drowsy trials of the other subjects, each session aligned with its whole session. Held-out sessions are aligned per session with: none / first30 (fixed calibration) / **cum** (all windows so far) / **slide100** (last 100) / all (transductive). References are label-free and include transitional trials. Scoring covers labelled trials from the 31st trial of each session on, identical for every variant. Primary metric: balanced accuracy (classes are unbalanced); also AUC.
- *Models:* InterpretableCNN (running BN) + EA, 5 seeds; Riemannian LR + re-centring and band-power LR + baseline subtraction (deterministic, seed 0). The reference is the arithmetic mean of sample covariances for every family.
- *Checks:* the online reference computations were unit-tested (cumulative, sliding and first-30 means equal the explicit means). A local smoke test on 4 sessions ran end to end.
- *Estimated cost:* 45–60 min on the T4.

## 2026-10-30 — Phase 3 results: online alignment on the full dataset (`eeg-phase3-online`)
- *Run:* Kaggle T4, 3,708 s (62 min; estimate was 45–60). Outputs in `research/results/eeg-phase3-online/` (per_subject, predictions, summary, config, env, log). Analysis: `research/src/analyze_phase3.py` → [tables/phase3_online.md](tables/phase3_online.md) (+ `_per_subject.csv`, `_time.csv`). 27 subjects, 12,888 labelled trials; scored trials start at the 31st trial of each session.
- *Main numbers (balanced accuracy %, subject AUC; Δ vs none, W/L over 27 subjects, Wilcoxon p):*

  | model | none | first30 | cum | slide100 | all (transductive) |
  |---|---|---|---|---|---|
  | ICNN (5 seeds) | 73.59 / 0.828 | 52.55 / 0.533 (−21.0, 3/24, p<1e-4) | 75.26 / 0.802 (+1.7, 13/14, p = 0.66) | 71.76 / 0.765 (−1.8, p = 0.17) | **77.85** / 0.834 (**+4.3, 19/8, p = 0.019**) |
  | Riemann LR | 64.92 / 0.739 | 57.27 (−7.7, p = 0.020) | 64.95 (+0.0) | 61.16 (−3.8) | 68.51 (+3.6, p = 0.077) |
  | Band-power LR | 68.30 / 0.757 | **71.23** (+2.9, p = 0.30) | 67.17 (−1.1) | 64.45 (−3.9, p = 0.10) | 69.61 (+1.3) |

- *Observations:*
  1. **Whole-session alignment reproduces F6 on the full dataset:** ICNN +4.3 bal-acc points (p = 0.019), same size as on SADT-2022. But AUC does not change (0.828 → 0.834, p = 0.85): the gain is calibration (where the threshold falls for a new driver), not ranking.
  2. **Online alignment (`cum`, `slide100`) did not help** in this run: `cum` +1.7 n.s., `slide100` −1.8 and lower AUC for every model. The time course table shows `cum` is weakest early in the session (Riemann: 56.0 vs 62.5 for none in trials 31–100) and catches up later.
  3. **`first30` collapses ICNN to chance** (AUC 0.53; 95% of outputs saturated at < 0.02 or > 0.98, vs 19% for none).
- *Diagnosis of 2–3 (4 sessions checked locally: s05, s35, s44, s45):* the pre-processed Cao EEG is rank-deficient (per-trial covariances have near-zero eigenvalues, consistent with ICA component removal), and the null directions change over the session. The whole-session mean covariance is well conditioned (cond 2e3–6e5), but the mean of the first 30 trials is not (cond 1e13–3e15; 1–3 eigenvalues < 1e-7 of the largest). `inv_sqrtm` only floors eigenvalues at 1e-10 × max, so R^{-1/2} amplifies those directions by ~1e5, and later trials, which carry real power there, blow up (aligned std 2.8e3–3.5e4 vs 1.0 with the whole-session reference). `cum` at trial 31 is exactly this first-30 mean, and `slide100` keeps re-estimating from 100 trials, so both inherit the problem early on. Band-power baseline subtraction inverts no matrix and is unaffected; there `first30` is the *best* variant (the first 30 trials are 69% alert, 2% drowsy, 28% unlabelled vs 30/15/54% for the rest), i.e. a clean alert baseline does help band power.
- *Is F7 (SADT, alert-only reference collapses ICNN) the same artefact?* No. On SADT the first-30-alert reference is singular for subjects 3, 6, 11 only (cond 1e12–1e15), and those are exactly the subjects that did *not* collapse (0.76, 0.84, 0.73); the 8 well-conditioned subjects fell to a constant 0.50. F7 stands as a separate effect (there, training subjects were also aligned with alert-only windows).
- *Conclusion:* the Phase 3 online result is **confounded by an unregularised whitener** and is not yet a test of online alignment. The transductive result (+4.3) is valid. Next: rerun with a shrinkage-regularised reference (R ← (1−α)R + α·tr(R)/C·I, α fixed in advance, applied to every reference including training), and log each reference's condition number.

## 2026-11-02 — XAI case study launched for the review (`eeg-xai-case`)
- *Goal:* quantitative evidence for the review's case study: are explanations of InterpretableCNN faithful, and does a window's decision depend on the rest of the session? Asked for by the user on a T4.
- *Code:* `research/src/run_xai.py`. Same LOSO, seeds and recipe as Phase 0 on the 2,022-window subset, so accuracy should reproduce 77.93% (`icnn_batch`, whole-subject statistics) and 73.68% (`icnn_running`).
- *Tests:* point and channel deletion against a random baseline for saliency, gradient x input, integrated gradients and the published CAM + CNN-Fixation tracing; model-randomisation sanity check; band and channel-group perturbation per true class; normalisation context (all, 30 random, alert only, drowsy only).
- *Decided before seeing results:* peripheral channels = FP1, FP2, F7, F8, FT7, FT8, T3, T4, TP7, TP8, T5, T6, O1, OZ, O2; the other 15 are central. Bands: delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-50 Hz.
- *Local smoke test:* 2 subjects, reduced epochs, ran end to end; frozen statistics reproduce whole-subject batch norm (assertion in the code).
- *Estimated cost:* 20-40 min on the T4.
- *Version 1 failed at once* (subject ids passed as numpy integers to a torch generator; only the `--subjects all` path, which the smoke test had not used). Fixed in `run_xai.py`, retested locally on all 11 subjects at reduced settings, and pushed as version 2. One fold had finished (subject 1, accuracy 83.0%, in line with Phase 0).
- *Version 1 failed at once* (subject ids passed as numpy integers to a torch generator; only the `--subjects all` path, which the smoke test had not used). Fixed in `run_xai.py`, retested locally on all 11 subjects at reduced settings, and pushed as version 2. One fold had finished (subject 1, accuracy 83.0%, in line with Phase 0).

## 2026-11-02 — XAI case study results (`eeg-xai-case`, version 2)
- *Run:* Kaggle T4, 351 s for all 110 folds (estimate was 20-40 min; attribution is much cheaper than expected). Outputs in `research/results/eeg-xai-case/results/`; analysis `research/src/analyze_xai.py` -> [tables/xai_case.md](tables/xai_case.md).
- *Check:* accuracy reproduces Phase 0 exactly: 77.93% / AUC 0.835 (whole-subject) and 73.68% / 0.818 (running).
- *Context:* same windows and weights, statistics from alert windows only change 24.55% of predictions, predicted drowsy 45.59 -> 68.21%, map similarity 0.502. Drowsy only: 14.59%, 31.02%, 0.694. 30 random windows: 4.80%, acc 77.55%.
- *Deletion (whole-subject, AUC; random 0.667 points, 0.707 channels):* tracing 0.542 / 0.561, saliency 0.580 / 0.641, gradient x input 0.185 / 0.501, IG 0.178 / 0.500. All beat random on at least 10 of 11 subjects (p <= 0.002). Point deletion for gradient x input is close to tautological in a nearly piecewise-linear network.
- *Sanity (Spearman with randomised model):* saliency 0.13, tracing 0.17, gradient x input and IG 0.51. Methods strongest on deletion are weakest here.
- *Stability across seeds:* tracing 0.56 (min 0.08), others 0.72-0.75.
- *All-window use (whole-subject):* removing theta / alpha / delta lowers P(drowsy) by 0.24 / 0.17 / 0.09 (all 11 subjects, p = 0.001). Tracing puts 64.0% of alert relevance on peripheral channels vs 45.6% drowsy (p = 0.007), peaks F7 (alert), PZ and CPZ (drowsy).
- *Did not go as expected:* zeroing peripheral channels barely changes P(alert) (-0.007, p = 0.58); removing beta does nothing significant (+0.008, p = 0.28); gamma removal lowers P(alert) slightly (-0.036, p = 0.001). So the published "alert = artefacts" reading is supported spatially by the maps but not confirmed by perturbation. Caveats: removal lowers energy, which may favour alert; EMG is broadband; averages can hide individual windows; no multiple-comparison correction. The review draft was reworded accordingly.
