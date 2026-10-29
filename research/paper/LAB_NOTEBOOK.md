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
