# Person 4 log (research paper: literature and analysis track)

## 2026-11-03 — E3 state mix on real drives (`analyze_statemix.py`) -> `research/paper/tables/e3_statemix.md`
- *Method:* bin scored trials (from the 31st) by drowsy share of the labelled trials so far in the session; balanced accuracy vs none, paired per subject, 5 seeds. Labels for analysis only.
- *prior30:* 0-10% +1.22 (11/11, p = 0.97); 10-25% +1.24 (p = 0.56); 25-50% +7.70 (12/4, p = 0.0076); 50-75% +12.47 (8/3, p = 0.032); 75-100% +1.18 (2/4, n = 6, p = 1.0).
- *cum, a = 0.1:* +2.06, -0.71, +5.42 (p = 0.015), +9.88 (p = 0.067), +2.45. Same shape.
- *Reading:* reproduces the inverted U of Phase 2b (SADT, constructed mixes) on real timelines: alignment helps most when the driver's windows so far contain both states.
- *Caveats:* exploratory; uncorrected; the subject set shrinks with share (22 down to 6); share rises with time on task, so state mix and elapsed time are entangled, and this is association. The peak bin (50-75%) is near but not the same as Phase 2b's 50%.

## 2026-11-03 — E4 launched (`eeg-e4-xai-aligned`)
- *Code:* `research/src/run_xai_aligned.py`. none vs prior30, 27 subjects, 5 seeds, Kaggle T4. Attributions with respect to raw input through the whitener. Saliency and gradient x input only: CAM tracing does not apply through a whitener, and integrated gradients nearly duplicated gradient x input (r = 0.997).
- *Fixed before seeing results:* (1) does removing peripheral channels, beta or gamma change P(alert) differently? (2) does removing theta, alpha or central channels change P(drowsy) differently? Paired across subjects, Holm-corrected over these 6.

## 2026-11-03 — E4 results (`eeg-e4-xai-aligned`, 5 seeds, T4, 2,430 s) -> `research/paper/tables/e4_perturb.csv`
- *Check:* trials from the 31st: AUC none 0.828, prior30 0.820, as in E2. Plain accuracy 78.55% vs 81.33%.
- *Primary, prior30 vs none, change in P(true class), Holm over 6:* alert: peripheral +0.207 vs -0.110 (Holm < 0.001); beta +0.074 vs +0.054 (0.003); gamma -0.195 vs -0.074 (< 0.001). Drowsy: theta -0.300 vs -0.244 (< 0.001); alpha -0.061 vs -0.070 (0.57); central -0.402 vs -0.361 (0.57).
- *Reading:* with alignment the model leans more on gamma for alert and more on theta for drowsy. Gamma overlaps muscle activity, so this may be a shortcut; not established.
- *Do not over-read the peripheral result:* zeroing 15 raw channels with the whitener fixed disturbs the whitener's channel mixing, so peripheral results may reflect that rather than reliance (drowsy -0.583 vs -0.060). Relevance maps disagree: peripheral share falls from 0.54 to 0.36 (gradient x input) under alignment.
- *Faithfulness weakens under alignment:* gradient x input point deletion AUC 0.635 vs random 0.671 (none: 0.511 vs 0.789), still 27/0 subjects. Random deletion also hurts more (0.671 vs 0.789), so the aligned model is more sensitive to input zeroing.
- *Sanity worsens:* correlation with a randomised model rises, saliency 0.117 -> 0.366, gradient x input 0.503 -> 0.599, because a fixed whitener shapes the maps whatever the weights.
- *Review paper impact:* on all 27 subjects unaligned, zeroing peripheral channels lowers P(alert) by 0.110 (p < 0.001) and gamma removal by 0.074 (p < 0.001). The review's 11-subject case study found -0.007 (p = 0.58) and -0.036. The artefact reading has more support on the full dataset, in our own conversion with natural class balance.

## 2026-11-03 — Novelty search outcome
- *Search:* team sheet of 28 papers (`novelty_search.csv`, Downloads). Read directly: Duan et al. 2025 (arXiv:2509.19403) full text, Gao & Lv 2026 (Frontiers) page, arXiv metadata for six others. The attached PDFs could not be read here (no longer in Downloads).
- *Point 1 is not new:* Duan et al. Eq. 8: R_0 = R_train; R_i = (n_(i-1) R_(i-1) + w cov_i) / (n_(i-1) + w), n counting training plus processed trials. Same form as our prior. It is an arXiv preprint on MI and SSVEP, no drowsiness data, no XAI, no state-mix analysis.
- *Decision:* do not claim prior-regularised alignment as a new method. Per the plan's decision table, the paper reproduces it under strict LOSO on driving data and leads with what is new: state mix on real drives, BN adaptation harming here, short-reference singularity, explanations under alignment.

## 2026-11-03 — Team PDFs read (`papers ieee` folder; abstract and keyword-level, plus closer reading of one)
- *Closest:* Li, ..., Wu 2026 (IEEE JBHI, backpropagation-free TTA): integrates online EA (online-updated target reference) and BN-adapt into all TTA methods, on driver-drowsiness regression (Driving, 15 subjects; SEED-VIG, 23). So "first online EA on drowsiness" cannot be claimed. It treats EA and BN-adapt as tools; a keyword search of the PDF finds no state-mix or short-reference conditioning analysis. It calls BN-adapt effective, against our E6 on classification with an unaligned model.
- *Drowsiness, no alignment found:* domain generalisation via style mixing (77.26%, 11 subjects); manifold-level augmentation (best baseline F1 64.67%); SPARK black-box adaptation (SADT); EDJAN cross-dataset (83.3%, SADT to SEED-VIG). Useful for related work.
- *Motor imagery (Euclidean-alignment style transfer, continual decoding, ReCIL):* background only.
- *Not readable:* Hasan et al. comparative paper; its PDF text layer is garbled.
- *Decision unchanged:* no new-method claim. State mix on real drives stays the main contribution. No paper here analyses it for drowsiness, but this is not an exhaustive search. E7 now matters more, since EA plus BN is the published combination.
