# Research Plan: Finding Novelty for an EEG Driver-Drowsiness Paper

_Drafted 2026-10-29. The demo app (`backend/`, `frontend/`, `training/`) stays frozen for presentation; all research work lives under `research/`._

## 1. Where we stand

| Item | Value |
|---|---|
| Original dataset | Cao et al. 2019, *Sci. Data*: 27 subjects, 62 sessions, 90-min VR lane-keeping task, 32-ch EEG at 500 Hz, lane-departure events with **reaction times (RT)** |
| Derived dataset (what we use now) | Cui et al.: 11 subjects, 2022 balanced 3-s epochs, 30 ch at 128 Hz, binary alert/drowsy from RT thresholds |
| Our baseline | InterpretableCNN, LOSO: **78.0%** when each held-out subject is scored as one batch (paper: 78.35%), **73.4%** when scored one epoch at a time (streaming-realistic) |

How the derived dataset was labeled (Cui et al.; restated in Jang et al. 2025): "alert RT" is a subject's baseline. Local RT is for one event; global RT is the mean over the previous 90 s. **Alert** means both are < 1.5× alert RT, **drowsy** means both are > 2.5× alert RT, and **everything in between is discarded**. Only 11 of 27 subjects survived (≥ 50 samples per class), using one session each.

**Takeaway:** the benchmark everyone uses throws away most of the original data: 16 subjects, extra sessions, all transitional samples, and the continuous RT signal.

## 2. Recent literature (2024–2026) and what it leaves open

| Paper | What it does | Result on this data | Gap it leaves |
|---|---|---|---|
| **DeltaGateNet**, Po et al., arXiv 2602.14071 (Feb 2026, v3 Aug 2026), [code](https://github.com/yip-tin-po-alex/DeltaGateNet) | Bidirectional first-order temporal differences + gated depthwise temporal convolutions | **83.21%** inter-subject on SADT-2022, 84.49% on SADT-2952 | Uses 5-fold, 60/20/20 subject splits, not LOSO, so it is not directly comparable to the 78.35% LOSO figure. No domain adaptation (listed as future work). Single epoch, no cross-epoch context. |
| **Calibration-free online TTA**, Jang et al., arXiv 2511.22030 (Nov 2025) | Online test-time adaptation (BN affine updates, energy-based memory bank, prototypes) on EEGNet | Average F1 **81.73%** (std 20.2) LOSO on the 11 subjects; InterpretableCNN source-only F1 66.55% | Binary only; label-free but needs a stream of test data; very high per-subject variance; no code found |
| **Entropy Optimization Network**, Yuan et al., IEEE JBHI 2025 | Cross-dataset unsupervised DA with self-training | 89.2% / 77.6% on two public datasets | Cross-*dataset* rather than cross-subject streaming |
| **ID3RSNet**, 2025 (PMC11751225) | Interpretable residual shrinkage net, single channel | 74.72% on the 11-subject setup | Single channel only |
| **Pre-stimulus EEG → RT regression**, Rahman et al., *Biomed. Phys. Eng. Express* 2025 | 1D-CNN regresses RT from a 2-s pre-stimulus window on the Cao dataset (25 drivers) | MAE 0.51 → 0.36 s (alpha band) | Regression on the full data exists, but with no ordinal/distribution modeling, no alignment, and its subject split needs checking |
| **Behavioral latency as weak event-time supervision**, Aimoldin et al., arXiv 2608.29428 (Aug 2026, accepted *Neural Computation*) | Predicts a *distribution* over event times instead of regressing RT; beats scalar regression across 4 backbones | Healthy Brain Network task, **not driving** | Never applied to drowsiness or driving data, so it is directly transferable |
| **EEG foundation models**: CBraMod (ICLR 2025, includes SEED-VIG vigilance), LaBraM, REVE (2025; code released Mar 2026) | Large pretrained EEG encoders | LaBraM reaches only **70.58%** on SADT-2022 in DeltaGateNet's table; EEGMamba 61.36% | Generic pretraining doesn't transfer well yet; parameter-efficient adaptation to this task is untested |
| **Euclidean alignment revisited**, arXiv 2502.09203 (2025); online EA, arXiv 2606.16462 (2026) | Whiten each subject by its mean covariance; online version uses the first N trials | Motor imagery | **Not applied to drowsiness.** In driving, the natural reference is the driver's alert period at the start of the drive |

## 3. Candidate novelty directions

Ranked by novelty × feasibility for our setup (CPU locally, Kaggle GPU available).

### A. Use the full dataset: continuous and ordinal drowsiness estimation (strongest novelty)
- Rebuild epochs from the **original** Cao dataset for all 27 subjects and all sessions, **keeping transitional samples** and the continuous RT.
- Train with **distributional / ordinal RT supervision**: predict a distribution over RT bins, or ordinal levels (alert < transitional < drowsy), following Aimoldin et al. 2026, which hasn't been applied to driving.
- The binary alert/drowsy decision becomes a threshold on the predicted RT. We report both the classic 11-subject binary benchmark (for comparability) and the new full-data continuous benchmark (a contribution in itself).
- **Why novel:** existing deep models (DeltaGateNet, TTA, InterpretableCNN) all train on the binary 2022 subset. RT-regression work (Rahman 2025) uses scalar regression only.

### B. Alert-baseline subject alignment (calibration without labels)
- Every drive starts alert. Use the **first few minutes** of each held-out driver's session as an unlabeled reference to whiten that subject's EEG (Euclidean alignment) and/or set BN statistics.
- This works in streaming mode, unlike the transductive whole-subject BN that inflates the 78% figure. It is cheaper than full TTA (Jang 2025) and has a clear physiological rationale.
- **Our own evidence:** whole-subject batch normalization gives 78.0% vs 73.4% per-epoch, so roughly 4.6 points come from subject-level statistics. B tries to recover that honestly.
- **Needs the original data**, because epoch timing within the drive is required.

### C. Cross-epoch temporal context
- The labels depend on **global RT**, the average over the past 90 s, yet every model scores one 3-s window in isolation. Add a lightweight sequence head (GRU or temporal convolution over the last *k* epoch embeddings) on top of a per-epoch encoder (InterpretableCNN or DeltaGateNet).
- Also cheap to test, and complementary to A and B.

### D. (Optional) Parameter-efficient fine-tuning of a foundation model
- LoRA/adapter-tune CBraMod or REVE with alignment from B. This is high risk: LaBraM did poorly here, and it needs a GPU. Use it as a comparison row, not the main claim.

**Recommended paper story:** A + B + C combined into one model: "Alert-anchored, context-aware ordinal estimation of driver drowsiness from EEG on the full sustained-attention driving dataset". Each component gets its own ablation. B and C can each be shown on the classic 11-subject benchmark too, so we have a comparable number against DeltaGateNet and TTA even if A's benchmark is new.

## 4. Experiment protocol (fixed before running anything)

- **Split:** leave-one-subject-out (LOSO). Hyperparameters are chosen on validation subjects drawn from the training folds (nested). **Never tune on the test subject.**
- **Inference mode:** report **streaming** (one epoch at a time, causal) as the primary number; report batch/transductive results separately and label them.
- **Metrics:** binary accuracy, macro-F1, AUROC; for continuous targets, RT MAE and Pearson/Spearman correlation. Report per-subject values and mean ± std.
- **Seeds:** 5 seeds per configuration.
- **Statistics:** Wilcoxon signed-rank test across subjects against the strongest baseline.
- **Baselines rerun under our protocol:** InterpretableCNN, EEGNet, DeltaGateNet (official code), band-power + logistic regression, Riemannian tangent-space + logistic regression. DeltaGateNet's 83.21% used a different split, so we rerun it under LOSO.

## 5. Phased work plan

| Phase | Work | Output | Needs |
|---|---|---|---|
| **0. Baselines** | LOSO + 5 seeds for the five baselines on the 2022 subset; streaming vs batch BN | Baseline table, results JSON | Data we already have; Kaggle GPU for DeltaGateNet |
| **1. Original data pipeline** | Download Cao et al. (figshare 6427334, preprocessed version); extract 3-s pre-deviation epochs with local/global/alert RT for every event; **reproduce the 2022-sample subset as a sanity check** | `research/data/` loader, dataset stats (samples per subject, RT distribution) | Download (size not yet checked; likely several GB) |
| **2. Quick wins** | (B) alert-baseline EA/BN; (C) temporal-context head, each on the 11-subject benchmark first | Ablation rows | — |
| **3. Full-data task** | (A) ordinal/distributional RT supervision on 27 subjects; compare against scalar RT regression and binary training | New benchmark + main results | Kaggle GPU |
| **4. Combine + analyze** | A+B+C model; ablations; interpretability (spatial/spectral maps: which bands and regions predict drowsiness); latency per epoch | Final tables and figures | — |
| **5. (Optional) Foundation model** | LoRA on CBraMod/REVE | One comparison row | GPU |
| **6. Writing** | Related work (section 2), method, results, limitations | Paper draft | — |

## 6. Risks and open questions

- **Label noise:** RT is a noisy proxy for drowsiness (a 2022 LSTM-CNN study notes high false negatives). Ordinal/distributional training is partly meant to absorb this.
- **The "alert start" assumption** in B must be checked per session from the RT time course in Phase 1.
- **Compute:** the local machine has no GPU. Phases 0, 3 and 5 should run on Kaggle.
- **Comparability:** new benchmark numbers won't be directly comparable to prior work, which is why the 11-subject results under LOSO stay in the paper.
- **Not yet verified:** the exact download size and file format of the original dataset; whether Rahman et al. 2025 used a subject-independent split; details of the Yuan et al. JBHI 2025 protocol.
- **Questions for the TA:** target venue and page limit; whether a new benchmark on the full dataset is acceptable as a contribution; whether a GPU is available beyond Kaggle.

## 7. Sources

- Cao et al. 2019, Multi-channel EEG recordings during a sustained-attention driving task: https://pmc.ncbi.nlm.nih.gov/articles/PMC6472414 (raw data: https://figshare.com/articles/Multi-channel_EEG_recordings_during_a_sustained-attention_driving_task/6427334/5)
- Cui et al., derived 11-subject dataset: https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687
- DeltaGateNet (2026): https://arxiv.org/abs/2602.14071
- Calibration-free online TTA (2025): https://arxiv.org/abs/2511.22030
- Entropy Optimization Network, JBHI 2025: https://doi.org/10.1109/jbhi.2024.3519730
- ID3RSNet (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11751225/
- Pre-stimulus EEG for driver RT prediction (2025): https://pubmed.ncbi.nlm.nih.gov/40068221
- Behavioral latency as weak event-time supervision (2026): https://arxiv.org/abs/2608.29428
- CBraMod (ICLR 2025): https://arxiv.org/abs/2412.07236; REVE: https://brain-bzh.github.io/reve/
- Revisiting Euclidean alignment (2025): https://arxiv.org/abs/2502.09203; aligned subject-specific encoders (2026): https://arxiv.org/abs/2606.16462
- Review of EEG driver-fatigue detection, WIREs 2026: https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70099

