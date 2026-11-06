# Evidence check before drafting (2026-11-02)

Source: team Google Doc "AIML" (tabs Sunny = A, Akshay = B, Aryan = C; 30 summaries, all read in full).
Data behind the counts: [appraisal_matrix.csv](appraisal_matrix.csv).

## What was checked
- **A1-A10:** key numbers checked against the 10 attached PDFs (text extraction). All matched except one value that could not be read: A1's per-subject range (56.64 to 89.00) sits in Table III, which does not extract as text.
- **B1-B10 and C1-C10:** key numbers and the strongest critical claims checked against the 20 PDFs in `papers.zip` (text extraction, plus viewing C3's figures and B3's SHAP plot). This covers key claims, not every sentence. Not readable as text, so not confirmed: B6's result tables (AUC 0.70, accuracy 0.79 vs 0.55, PPV 0.80 vs 0.88) and C5's InceptionV3 99.27%.
- **Citations flagged "from memory":** checked against Crossref (results below).

## Corrections from the B and C check
1. **B4 labelling rule is incomplete in the summary.** Algorithm 2 labels a window drowsy only if the percentage change in heart rate is within -16.7% to -1.9% AND heart rate is within 72.3-90.7 bpm. The circularity point still holds, and more strongly: the label is computed from heart rate, the selected features include MeanRR, and the paper explicitly states "no circular reasoning", listing SDNN, RMSSD and LF/HF as its features without mentioning MeanRR there.
2. **B3 line 7 misreads the SHAP plot.** In Figure 6, high MeanRR (slow heart) has strongly negative SHAP and low MeanRR positive, so the top feature pushes towards fatigue when the heart is fast. MeanHR, its mathematical inverse, points the other way (low HR towards fatigue). High RMSSD also has positive SHAP, against the text's "reduced HRV". The authors discuss only the MeanHR direction. This is a useful finding: SHAP on redundant features can give physiologically conflicting signs. Reading assumes SHAP is for the fatigued class (label 1), which is the usual default.
3. **B3 "ROF as a feature" concern is resolved.** ROF appears only in Figure 5's abbreviation list. Figure 6 shows 8 HRV features and no ROF, so it looks like a leftover abbreviation. The two validation descriptions (70/30 and Group k-Fold) are both in the paper, as the summary says.
4. **B1 wording:** the authors do not "recommend LOPO as most appropriate" in those words. They say it "gives the best generalised results" for participant-independent evaluation and addresses autocorrelation. Paraphrase accordingly.
5. **C4:** the abstract says Grad-CAM revealed regions "around the nasal dorsum"; the body says activation was primarily periocular. The summary follows the body, which is right. The abstract-vs-body gap is itself worth noting.
6. **C6:** the ethics statement says "first-degree relative", not "father" (C5 is the one that says father). Do not put this detail in the paper.

Confirmed as the summaries state: C3 has no heatmap figure, a 10-sample confusion matrix (9 correct), ROC AUC 1.00 and validation accuracy near 95.5% against a claimed 99.27%, PERCLOS thresholds of both 0.4 and 0.7, and an e-waste passage. C1 says explanations were "evaluated qualitatively only" while reporting Table 14, and its baseline rows do shift between tables. C2 says "four" datasets and lists three, splits by stratified sampling, and states UTA-RLDD has fewer than 25 subjects (C10: 60). C7 reports Swin 100% (+/-0.0). C8 states an 18/5 subject split but also a 70/30 split. C9's abstract claims 2-3%. B10 names no dataset. The appraisal totals below are unchanged.

## Corrections to make in the doc or when writing
1. **A patterns, point 2:** 6 of 10 papers use the 11-subject Cui subset, not 5. A1 builds it (A1, A2, A3, A4, A8, A9).
2. **A patterns, point 4:** "72-79%" mixes balanced and unbalanced results, and multi-channel with single-channel input. Like for like:
   - balanced: A1 78.35 (30 channels), A2 73.22, A3 72.97, A4 77.16 (all three Oz only), A8 79;
   - unbalanced: A1 77.70, A4 74.72.
3. **CAM classification conflict:** tab C's background notes call CAM "intrinsic to GAP networks", while tabs A (A2, A4) call it post-hoc, model-specific. Decision used here: post-hoc, model-specific, because the map is computed after training.
4. **B5 journal version exists:** Zhou et al., Int. J. Human-Computer Interaction 38, 719-729, DOI 10.1080/10447318.2021.1965774. Different model (GPBoost + SHAP). Nobody has read it, so B5's arXiv summary cannot simply be re-cited to it.
5. **B10 citation:** Crossref gives pages 1820-1827 and "SHAP-Based" in the title (the doc has 1820-1825).
6. **Assignment sheet errors, caught by the team:** C4 note ("not the eyes") was wrong, because the eyes are the dominant region. C2 and C7 use Grad-CAM and CAM, not attention maps. Follow the summaries, not the sheet.

## Appraisal denominators must be unified
The three tabs tally differently:
- A: all 10 papers.
- B: 2024-2026 peer-reviewed only (n = 5).
- C: 2024-2026 including a preprint (n = 7).

The user removed the year limit on 2026-11-01, so one rule now applies: all detection studies that make an interpretability claim, n = 27. Excluded from the counts: A9 (methods study), B9 (classifies manoeuvres, not drowsiness), C10 (dataset paper, no XAI).

| Check (n = 27) | Result | Studies |
|---|---|---|
| Faithfulness tested quantitatively | 1 (3.7%) | A4 |
| Partial check only (consistency, stability, sufficiency, plausibility) | 7 (25.9%) | A7, B1, B2, B3, B5, C1, C2 |
| No evaluation of the explanation | 19 (70.4%) | the rest |
| Subject-independent testing, clear | 11 (40.7%) | A1-A4, A6-A8, B1, B3, B8, C4 |
| Subject-independent stated or partial, not verifiable | 3 (11.1%) | C1, C2, C3 |
| Not subject-independent, or unclear | 13 (48.1%) | A5, A10, B2, B4, B5, B6, B7, B10, C5-C9 |
| Code public | 3 (11.1%) | A1, A2, A3 (one research group) |
| Explanation agrees, at least partly, with known physiology | 21 (77.8%) | see the matrix |
| Faithfulness tested AND subject-independent | 1 (3.7%) | A4 |
| Mechanism | post-hoc 17, intrinsic 6, hybrid 3, claimed but none shown 1 | see the matrix |

Restricted to 2024-2026 (n = 19): subject-independent 8, faithfulness 1, partial 5, code 0.

The "matches physiology" count is generous: for eye-state tasks (C7) a heatmap on the eyelid is true by label definition, and C1's claim is weakly supported by its own figures.

## Limits the paper must state
- The 30 papers were chosen by a targeted search, not exhaustive screening, and no found-screened-included counts were recorded. So the numbers describe this set, not the prevalence across the field. Write "in the studies we examined", and do not call it PRISMA or systematic.
- The "first dedicated review" claim still needs the IEEE Xplore + Scopus check assigned for Nov 1. There is no record of it in the doc.
- Weak evidence to label as such: B10 (no data or results), C3 (explanations never shown; figures contradict the headline numbers), B9 and C9 (preprints), B5 (preprint).
- Case study: DeltaGateNet's 64.74% under LOSO is one seed, and its published 83.21% uses subject-level 5-fold, not LOSO. Report it as preliminary. The ICNN transductive vs streaming result (77.93 vs 73.68, 5 seeds on SADT-2022) is stronger and directly supports the batch-norm caveat noted on A2.

## Citations verified on Crossref
| Item | Verified details |
|---|---|
| B2 Vitorino 2023 | LNCS, pp. 108-113, DOI correct |
| C10 Ghoddoosian 2019 | CVPRW, pp. 178-187, DOI correct |
| A3 Cui 2021 | Cyberworlds, pp. 201-208, DOI correct |
| C8 Albadawi 2023 | J. Imaging 9, 91, DOI correct |
| Sujatha Ravindran and Contreras-Vidal 2023 | Sci. Rep. 13, DOI 10.1038/s41598-023-43871-8 |
| Friedman 2001 (PDP) | Ann. Stat. 29, 1189-1232, DOI 10.1214/aos/1013203451 |
| Goldstein et al. 2015 (ICE) | JCGS 24, 44-65, DOI 10.1080/10618600.2014.907095 |
| Wachter et al. (counterfactuals) | SSRN version, DOI 10.2139/ssrn.3063289 (2017; journal volume still to check) |
| Chattopadhay et al. 2018 (Grad-CAM++) | WACV, pp. 839-847, DOI 10.1109/WACV.2018.00097 |
| Selvaraju et al. 2017 (Grad-CAM) | ICCV, pp. 618-626, DOI 10.1109/ICCV.2017.74 |
| Massoz et al. 2016 (DROZY) | WACV, pp. 1-7, DOI 10.1109/WACV.2016.7477715 |
| Weng et al. 2017 (NTHU-DDD) | LNCS, pp. 117-133, DOI 10.1007/978-3-319-54526-4_9 |
| Fusek 2018 (MRL Eye) | LNCS, pp. 433-444, DOI 10.1007/978-3-030-03801-4_38 |
| Abtahi et al. 2014 (YawDD) | ACM MMSys, pp. 24-28, DOI 10.1145/2557642.2563678 |
| Song et al. 2014 (CEW) | Pattern Recognition 47, 2825-2838, DOI 10.1016/j.patcog.2014.03.024 |
| Tao et al. 2024 (MPDB) | Sci. Data 11, DOI 10.1038/s41597-024-03222-2 |

Zeiler and Fergus 2014 was later confirmed (DOI 10.1007/978-3-319-10590-1_53). Not verified: NITYMED's original reference, so the paper cites Florez et al. for it.

## Added while drafting (2026-11-02)
- **C9 quotable overstates:** personalising narrowed "part of", not "much of", the gap to the CNN: eye state 91.70 to 93.23 vs CNN 99.10 (1.53 of 7.40 points); yawning 95.90 to 97.23 vs 98.80 (1.33 of 2.90). The paper uses "part of".
- **IEEE Xplore check, done 2026-11-02:** title and metadata queries for drowsiness or driver fatigue with explainable, interpretable or XAI found no dedicated review. Scopus is not checked.
- **New related work:** Ajayi et al., IEEE Access 2025 (78 studies, explainability in 19.2%). Abstract and contents read; full text not read.
- **Read only from search summaries, not full text:** Fadhel et al. 2026 and Hassan et al. 2026 (the two multimodal surveys cited in the introduction). Someone should confirm how they treat explainability.
- **Case-study attribution tests added** (`eeg-xai-case`): the draft's wording that EEG models "recognise alertness from artefacts" was changed to the authors' reading, because all-window perturbation only partly supports it.
- **Before submitting:** the paper now states that code and per-subject results are on GitHub. That is only true after `research/` is pushed.

# Second check: IEEE additions (2026-11-05)

Source: team Google Doc tabs A, B, C (A11-A15, B11-B14, C11-C16, R1), against the 16 PDFs in `papers (1).zip`. Crossref confirmed every new DOI, title and page range, including the IEEE Access pages that the doc left open (B12 83815-83827, B14 70486-70509). Counts are in [appraisal_matrix.csv](appraisal_matrix.csv): 41 counted studies.

| Check (n = 41) | Yes | Partial or stated | No, unclear or n/a |
|---|---|---|---|
| Agrees with physiology | 32 | - | 9 |
| Tested on unseen drivers | 15 | 5 | 21 (3 are rule systems with no detection test) |
| Explanation evaluated | 2 (A4, C11) | 11 | 28 |
| Code public | 5 | - | 36 |

Types: 24 post-hoc, 11 intrinsic, 5 combined, 1 claimed only. Fig. 3 medians: 97.06% (n = 17) and 77.16% (n = 15). B13 and C13/C14 are left out of Fig. 3 because their splits cannot be verified. A11 is plotted at 70.5% (all bands), not the 72.6% theta-only result, which was chosen after seeing the bands.

## Confirmed in the PDFs
- A13's tables do not extract as text, so the page was rendered and read: 83.84 and 80.06 confirmed. Precision, recall and F1 are equal within every row of both tables.
- A14: band results, "does not significantly improve" for attention, delta conflict stated by the authors, LOSO 0.853 with the 21 vs 23 subject inconsistency.
- A15: 0.891 is reported under the within-subjects experiment; LOSO is described, and we found no separate LOSO result.
- A12 90/10 split, B11 5-fold plus 7:3 split, B12 timed sessions and tablet movie, C11 18/2 split with explanations on correct predictions only, C15 three drivers.

## Corrections to the team doc
1. **C13/C14 "same reference ordering" is wrong.** The reference lists differ (ratio 0.29, different first reference). What is verified: identical results tables at all three ratios, shared runs of 100+ words including equations, word-sequence similarity 0.59, no named dataset, different authors. The paper states only these facts and makes no allegation.
2. **B13 segment-count "impossible" is not used.** The paper's windows overlap, so the arithmetic is not decisive. Stronger facts used instead: skin conductance is listed as a top feature, but none of the three datasets as the paper describes them records it; SIM-DROWSY is cited as [26], while the printed reference list ends at [22]. Also not used: "NTHU-DDD has no PERCLOS annotations", because PERCLOS can be derived from its eye labels.
3. **A14 and EMG:** the paper does not discuss EMG. The review's muscle-artefact remark is ours and is worded as a possibility.
4. **B11 leakage:** the paper describes no nesting, which is what the review says. It does not explicitly state that selection used test folds.
5. **A15 and SEED-VIG:** the review says "temporal and posterior" sites, following the dataset description.
