# C4: Full-Data Results (REAL RUN, CPU)

- **Data:** ASSISTments 2012–13 with affect. All students were requested; **26,901** students and **2,514,851** predictions remained after filtering (main problems, at least 3 prior problems per student).
- **Split:** 70/30 by student, seed 42. Rule thresholds were tuned on a validation split of the *training* students only.
- **Scripts:** `code/c4_fusion/c4_refine.py` (~9 min) and `c4_dkt.py` (~2.5 min, 3 epochs, CPU).
- **Raw output:** `c4_full_metrics.json`, `c4_dkt_metrics.json`.

## Predicting next-answer correctness
| Features | Model | AUC | Acc | F1 (incorrect) |
|---|---|---|---|---|
| Majority baseline | – | 0.500 | 0.698 | 0.000 |
| Affect only | XGBoost | 0.609 | 0.702 | 0.103 |
| Cognitive only | LogReg | 0.689 | 0.720 | 0.307 |
| Cognitive only | XGBoost | 0.707 | 0.728 | 0.343 |
| Fused | XGBoost | 0.708 | 0.728 | 0.346 |
| Cognitive (skill sequence) | **DKT (LSTM)** | 0.721 | 0.733 | – |
| Fused | **DKT + affect** | **0.722** | **0.734** | – |

## Rule engine (tuned on validation, reported on test; base rate of an incorrect next answer 30.2%)
| Rule | Params | Coverage | Precision | Recall | Lift |
|---|---|---|---|---|---|
| Performance only | acc5 < 0.6 & att5 > 1.25 | 14.9% | 45.2% | 22.3% | 1.50 |
| **Emotion-aware** | perf rule OR (acc5 < 0.7 & frustration/confusion > 60th pct) | 18.7% | 44.3% | **27.5%** | 1.47 |

## Findings
1. **Ranking:** DKT + affect > DKT > fused XGBoost > cognitive XGBoost >> affect-only. Sequence modelling adds about +0.014 AUC over hand-crafted features.
2. **Affect fusion gives a consistent but small gain:** +0.0013 AUC for XGBoost and +0.0008 for DKT. These detector-based affect values are derived from the same logs, so they are mostly redundant with the performance features. This motivates independent facial affect (C1–C3).
3. **The rule engine is where affect matters most.** The emotion-aware rule catches **23% more upcoming failures** (recall 22.3% → 27.5%) for about a 1-point precision cost. In practice the tutor would intervene for about 4% more problems and reach many more struggling students.
4. **Pilot vs full data:** the results are stable. Cognitive XGBoost AUC was 0.704 on the pilot and 0.707 on the full data, so the pilot was representative.

## Robustness: 5 seeds + student-clustered bootstrap (REAL RUN)
`code/c4_fusion/c4_stats.py`, seeds 0–4 (different student splits), 200 bootstrap resamples of test students per seed, about 69 min on CPU. Raw output: `c4_stats_metrics.json`.

| Metric | Mean ± SD over 5 seeds |
|---|---|
| XGBoost cognitive AUC | 0.7039 ± 0.0009 |
| XGBoost fused AUC | 0.7052 ± 0.0010 |
| DKT AUC | 0.7189 ± 0.0015 |
| DKT + affect AUC | 0.7195 ± 0.0014 |
| Rule recall: perf-only → emotion-aware | 21.8% → 27.0% |
| Rule precision: perf-only → emotion-aware | 44.7% → 43.6% |

| Gain | Mean over seeds | Seeds > 0 | Per-seed bootstrap 95% CI excludes 0 |
|---|---|---|---|
| XGBoost fused − cognitive (AUC) | **+0.0013** (SD 0.00007) | 5/5 | 5/5 (e.g., seed 0: [+0.0011, +0.0015]) |
| DKT+affect − DKT (AUC) | +0.0006 (SD 0.0005) | 5/5 | 4/5 (seed 4: [−0.0002, +0.0004]) |
| Emotion-aware rule recall | **+5.3 pts** (SD 0.07) | 5/5 | 5/5 (≈ [+5.0, +5.5]) |
| Emotion-aware rule precision | −1.1 pts | 0/5 | 5/5 (a significant small cost) |

**Verdict:**
- The affect gain for XGBoost is **statistically reliable but practically tiny** (about +0.1 AUC point).
- For DKT the gain is **marginal and not robust**: one seed's CI includes 0.
- The rule-engine recall gain (+5.3 pts, a 24% relative increase) is **robust and practically meaningful**, at a significant precision cost of −1.1 pts.
- The same thresholds were chosen on validation in all 5 seeds, so the rule is stable.

## Caveats
- DKT ran for only 3 epochs, and the training loss was still noisy. More epochs or a validation-based early stop might add about 0.005–0.01.
- The affect values are detector estimates, not observed emotions.
