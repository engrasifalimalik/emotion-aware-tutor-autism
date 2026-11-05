# C4: Actual Pilot Results (REAL RUN, CPU)

- **Data:** ASSISTments 2012–13 with affect predictions. 4,000 students were sampled at random; 2,297 remained after filtering, giving **214,581** next-problem predictions.
- **Filtering:** only main problems are kept, and a student needs at least 3 prior problems before predictions start.
- **Split:** 70/30 by student, so no student is in both train and test. Seed 42. Runtime 119 s.
- **Raw output:** `results/c4/c4_pilot_metrics.json`.

## Predicting next-answer correctness
| Features | Model | AUC | Acc | F1 (incorrect) |
|---|---|---|---|---|
| Majority baseline | – | 0.500 | 0.693 | 0.000 |
| Cognitive only | LogReg | 0.690 | 0.712 | 0.320 |
| Cognitive only | XGBoost | 0.704 | 0.718 | 0.352 |
| Affect only | LogReg | 0.521 | 0.693 | 0.000 |
| Affect only | XGBoost | 0.603 | 0.697 | 0.129 |
| **Fused** | LogReg | 0.690 | 0.713 | 0.323 |
| **Fused** | **XGBoost** | **0.705** | **0.720** | **0.357** |

## Offline rule engine (flags for the next answer being incorrect; base rate 30.7%)
| Rule | Coverage | Precision | Recall | Lift |
|---|---|---|---|---|
| Performance only | 13.1% | 45.5% | 19.4% | 1.48 |
| Affect only | 13.0% | 33.8% | 14.3% | 1.10 |
| **Emotion-aware (perf OR low-acc + frustrated/confused)** | 14.2% | **46.3%** | **21.5%** | **1.51** |

## Interpretation
1. **Cognitive features carry most of the signal.** Affect alone is weak (AUC 0.60).
2. **Fusion adds only a little.** AUC rises by about 0.001 and incorrect-answer F1 by about 0.005. The likely reason is that these affect values are *sensor-free detector outputs*, which were themselves derived from the same log features. They are largely redundant with the cognitive features. This is an important finding for the thesis: it supports using real facial affect (C1–C3) as an independent modality.
3. **The emotion-aware rule catches about 11% more of the upcoming failures** (recall 19.4% → 21.5%) at equal or better precision. That is a small but consistent improvement.
4. Fixed thresholds for frustrated or confused (top 20%) were not tuned. A proper threshold sweep is the next step.
