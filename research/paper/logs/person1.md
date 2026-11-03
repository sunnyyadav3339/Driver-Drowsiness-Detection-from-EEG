# Person 1 log (research paper: bug fix track)

## 2026-11-03 — E1 launched (`eeg-e1-shrink`)
- *Goal:* a fair test of online alignment, by fixing Phase 3's near-singular references. Every reference, training included, becomes (1-a)R + a tr(R)/C I.
- *Code:* `research/src/run_shrink.py`. Same protocol and recipe as `run_phase3.py`. InterpretableCNN, seed 0, a = 0.01, 0.05, 0.1, 0.2, Kaggle T4.
- *Check:* unit test (a = 0 leaves R unchanged; whitener equals `inv_sqrtm`). Local test on 4 sessions: condition number at trial 31 falls from about 4e14 to about 150 at a = 0.1.
- *Comparison:* seed 0 of `none` should match Phase 3's seed 0.
- *Selection rule:* the final a will be fixed from these seed-0 results and stated in the paper, with all values reported.

## 2026-11-03 — E1 results (`eeg-e1-shrink`, seed 0)
- *Run:* Kaggle T4, 1,590 s. `none` matches Phase 3 seed 0 exactly (bacc 73.89%, AUC 0.836; largest per-subject difference 0).
- *Diagnosis confirmed:* the first-30 reference has condition number above 1e10 in all 62 sessions (median 3.4e14); after shrinkage it is 69-1,686.
- *first30 no longer collapses:* 52.5% (Phase 3 seed 0) -> 68.6 / 74.0 / 75.3 / 76.3% at a = 0.01 / 0.05 / 0.1 / 0.2.
- *Online gain stays small and not significant on this seed:* cum vs none +1.45 to +2.12 points (best a = 0.1: 76.02%, 16/11 subjects, p = 0.28), AUC -0.015 (p = 0.94).
- *Whole-session reference:* +2.88 to +3.70 points, AUC unchanged. Shrinkage does not raise it above Phase 3 seed 0 (77.79%).
- *Choice:* a = 0.1 as primary. Picked by looking at test subjects on one seed, so the paper must say so and report all four. cum and all differ by under 1 point across a, so this choice adds little tuning gain.
- *Caveat:* p-values here are single-seed and exploratory; the 5-seed run is the evidence.

## 2026-11-03 — E5 time to benefit (`analyze_time.py`, a = 0.1, seed 0) -> `research/paper/tables/e5_time.md`
- *Trials 31-100:* cum gives no benefit (-1.91, p = 0.83), the gap Person 2's population prior targets.
- *Trials 201-300:* cum +5.40 (11/5 subjects, p = 0.020), close to the whole-session +6.38.
- *Trials 301+:* no mode beats none; first30 is worse (-6.30, 1/14, p = 0.0034), consistent with a fixed early reference going stale.
- *Caveat:* exploratory: 16 uncorrected tests, one seed, different subject sets per bin. Confirm on 5 seeds before reporting.

## 2026-11-03 — E1 final launched (`eeg-e1-final`)
- a = 0.1, seeds 0-4, InterpretableCNN, same code as `eeg-e1-shrink`. Estimated about 50 min on the T4.
- *Primary comparison, stated before seeing results:* cum vs none on balanced accuracy and AUC across 27 subjects.

## 2026-11-03 — E1 final results (`eeg-e1-final`, a = 0.1, 5 seeds) -> `research/paper/tables/e1_final.csv`, `e5_time_final.md`
- *Run:* Kaggle T4, 2,355 s. `none` reproduces Phase 3: 73.59%, AUC 0.828.
- *Primary comparison, cum vs none:* 76.25% vs 73.59%, +2.66 points, 16/11 subjects, p = 0.155 (Holm 0.465); AUC 0.818 vs 0.828, -0.010, p = 0.38. **Not significant.**
- *first30:* fixed, 52.55% -> 74.79% (AUC 0.533 -> 0.801); vs none +1.20, p = 0.55.
- *slide100:* 74.77%, +1.18, p = 0.86.
- *all (whole session, not available live):* 77.27%, +3.68, 18/9, p = 0.032 (Holm 0.128); AUC +0.004. Phase 3 without shrinkage: 77.85%.
- *Conclusion:* the bug fix works, and cum improves on Phase 3 (75.26% / 0.802), but plain online alignment does not significantly beat no alignment across a whole session.
- *Time course (exploratory; 16 tests, different subject sets per bin):* cum -0.38 in trials 31-100 (p = 0.90), +3.71 in 101-200, +6.54 in 201-300 (15/3, p = 0.0021), -0.34 at 301+. first30 at 301+ -6.40 (2/13, p = 0.0015), so a fixed early reference goes stale. The whole-session reference gives no gain at 301+ either (+0.14).
- *Selection note:* a = 0.1 was chosen from seed 0 of this same data, so these results are not fully independent of the choice.
