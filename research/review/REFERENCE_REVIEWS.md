# Model review papers shared by the TA (read 2026-11-01)

The TA's note: "this is how a review paper is written". Notes on their structure, so the team can copy the form.

## R1. A Survey on Drowsiness Detection: Modern Applications and Methods
Fu, Boutros, Lin, Damer. *IEEE Trans. Intelligent Vehicles* 9(11), 2024, pp. 7279–7300. DOI 10.1109/TIV.2024.3395889. 190 refs.
- **Type:** narrative survey (no PRISMA). Broad: EEG + ECG + vision, all application areas (driving, workplace, healthcare, aviation, rail).
- **Outline:**
  1. Introduction (motivation, existing surveys and their gap, Fig. 2 taxonomy)
  2. Areas of application
  3. Measuring technology: how EEG, ECG and vision each work for drowsiness
  4. Modern methods: EEG-, ECG- and vision-based (one summary table each)
  5. Widely used databases (table)
  6. Performance and evaluation metrics
  7. Technical and practical limitations (vision; physiological)
  8. Potential research directions
  9. Conclusion
- **What to borrow:**
  - a taxonomy figure early (Fig. 2) that the whole paper follows;
  - one "recent works" table per category;
  - a datasets table;
  - a metrics section;
  - "limitations" kept separate from "future directions";
  - an explicit claim of how this survey differs from earlier ones.
- **Gap for us:** no section on interpretability or XAI.

## R2. Intelligent Driver Drowsiness Detection Using Brain Imaging: A Systematic Review for Enhanced Road Safety in ITS
Ali, Zafar, Kim, Kim, Lee. *IEEE Trans. Intelligent Transportation Systems*, July 2026. 234 refs.
- **Type:** **systematic review following PRISMA**, with a macro (bibliometric) and a micro (study-by-study, 2020–2025) analysis. Four research questions are stated in the introduction.
- **Outline:**
  1. Introduction
  2. Materials and Methods (PRISMA, search strategy, inclusion/exclusion)
  3. Macro-level analysis:
     - publication trend;
     - fields (Web of Science categories);
     - bibliographic coupling of sources;
     - keyword co-occurrence map (VOSviewer-style).
  4. Micro-level analysis:
     - what drowsiness is;
     - its stages;
     - the ML pipeline;
     - neural indicators per modality (EEG, fNIRS, fMRI, MEG).
  5. Discussion
  6. Ethics, privacy, consent and bias
  7. Conclusion
- **Search protocol (Web of Science):** four OR-groups ANDed together:
  1. fatigue / sleepiness / drowsiness / tiredness;
  2. EEG / fNIRS / fMRI / MEG (full names and acronyms);
  3. deep learning / machine learning / CNN / classifier / classification / AI;
  4. "driving".
  - The general dataset feeds the macro analysis.
  - Inclusion: peer-reviewed journal articles, real or simulated driving, brain imaging + ML. Some studies were added by hand.
- **Tables:**
  - methods comparison;
  - symptoms, causes and prevention;
  - sleep-stage indicators;
  - EEG+ML binary studies;
  - EEG+ML multiclass studies;
  - EEG + other physiological signals;
  - fNIRS, fMRI and MEG studies;
  - physiological-method comparison;
  - neuroimaging comparison;
  - ML algorithm comparison.
- **What to borrow:** the PRISMA flowchart, the keyword-group search string, explicit research questions, bibliometric figures, and an ethics section.
- **Gap for us:** XAI is not a focus.

## R3. A Comprehensive Review of Explainable AI in Deep Learning Algorithms for EEG Analysis
*ACM Trans. Computing for Healthcare*, published 17 March 2026. DOI 10.1145/3785409.
- **Type:** systematic-style review of 55 studies (2018–2023), about XAI on EEG across all application areas. **Closest to our topic.**
- **Outline:**
  1. Introduction
  2. Preliminaries and methodology:
     - EEG fundamentals: anatomy, electrode placement, bands, preprocessing;
     - DL models: RNN, CNN, Transformer;
     - XAI methods: LRP, CAM, LIME, DeepLIFT, SHAP, activation maximisation, attention;
     - search strategy.
  3. Results:
     - application areas (epilepsy, **drowsiness**, sleep, emotion, ADHD, PD, AD, MI, eye motion, other);
     - matching XAI methods to EEG tasks;
     - EEG insights gained through XAI;
     - impact of XAI on model performance;
     - **evaluation of XAI techniques**.
  4. Discussion: uncertainty, limitations and future perspectives, conclusions.
- **Search:** (explain OR explanation OR visualization OR visualize OR interpret) AND (EEG OR electroencephalography OR BCI OR brain) AND (deep learning OR CNN).
- **Note on terms:** they use interpretability and explainability interchangeably, and note the distinction:
  - explainability = user engagement;
  - interpretability = intrinsic transparency of the model.
- **Tables:**
  - technical overview: Application | Ref | Year | XAI method | Black-box | Accuracy | Dataset | Code;
  - EEG findings per application;
  - task → suitable XAI methods, with caveats.
- **Their drowsiness coverage is one paragraph:** Cui et al.'s InterpretableCNN with modified CAM (78%), the finding that EMG in peripheral channels (usually treated as artefact) separates drowsy from alert, and two earlier single-channel papers by the same group.
- **Their §3.5:** very few studies evaluate whether XAI explanations are faithful (ROAR, comparisons of LRP / IG / DeepLIFT), and explanations generalise poorly across sessions and subjects.

## The gap our review can fill
- R1 and R2 survey drowsiness detection but do not cover interpretability.
- R3 surveys EEG XAI but gives drowsiness one paragraph (Cui et al. only) and stops at 2023.
- **A dedicated, systematic review of interpretable and explainable drowsiness detection, 2024–2026,** is therefore not yet covered by these three. It should include:
  - interpretable-by-design models vs post-hoc XAI;
  - what each explanation reveals (bands, channels, time, EMG/EOG artefacts);
  - whether explanations are evaluated for faithfulness;
  - whether they hold under cross-subject (LOSO) testing.
- Our own LOSO results (`research/paper/FINDINGS.md`) can support a small reproducibility or benchmark section. Example: DeltaGateNet scores 83% in its paper but 65% under LOSO.
- To confirm with the professor: EEG only or all modalities; PRISMA systematic review or narrative survey; target venue; year range.

## Novelty check (2026-11-01, quick web search; to re-check in IEEE Xplore / Scopus before claiming "first")
- No review was found that is dedicated to **XAI for drowsiness detection across modalities**.
- Closest existing reviews, which our related-work paragraph must cite and set ourselves apart from:
  - *AI-Based Non-intrusive Driver Drowsiness Detection: A Multimodal Review* (J. Intelligent Transportation Systems Research, 2026; 2020–2025). Mentions XAI only as a future direction. https://link.springer.com/article/10.1007/s13177-026-00682-1
  - *Intelligent driver monitoring systems: a survey of drowsiness detection technologies* (Artificial Intelligence Review, 2026; PRISMA 2020, Jan 2020–Sep 2025, vision/physiological/vehicular/multimodal). https://link.springer.com/article/10.1007/s10462-026-11505-w
  - *A review on AI-driven drowsiness detection systems using deep learning* (EPJ Web of Conf., 2026). https://www.epj-conferences.org/articles/epjconf/abs/2026/04/epjconf_icmsi2026_04001/epjconf_icmsi2026_04001.html
  - R1, R2 and R3 above.
- Primary XAI drowsiness studies seen so far (to include):
  - low-light detection with dual attention + XAI (Sci. Rep. 2026), https://www.nature.com/articles/s41598-026-44442-3
  - del Pino et al., "Drowsiness Detection in Drivers with Explainable AI" (SSRN 4597353)
  - Cui et al. InterpretableCNN (EEG)
  - SHAP in federated drowsiness detection
  - Grad-CAM / LRP on facial frames (evaluated qualitatively only)

## Planned differentiators
1. **Topic:** the first review dedicated to interpretability and explainability in drowsiness detection, across EEG, ECG, vision, vehicle and multimodal systems. The others treat XAI as a one-line future direction.
2. **A cross-modal taxonomy:** *how* the explanation is produced (intrinsic vs post-hoc) × *what* it explains (spatial/channel, spectral, temporal, facial region, feature).
3. **A critical-appraisal table for every study:**
   - Is the explanation checked against known physiology (alpha/theta rise, PERCLOS, HRV LF/HF)?
   - Is its faithfulness evaluated?
   - Is validation subject-independent?
   - Are code and data available?

   Then report percentages, e.g. "only X% evaluate their explanations".
4. **A small empirical case study** from our own experiments: published vs LOSO accuracy, and XAI maps on InterpretableCNN.
5. **A reporting checklist** for future XAI drowsiness papers.
