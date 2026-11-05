# C1: Cross-Domain Benchmark, Adult → Child → Autistic (REAL RUN, CPU)

**Setup**
- Frozen ImageNet backbone (224 px) + class-balanced logistic regression.
- Trained on 6,000 FER2013 training faces, with **child and teen faces removed** (age-model exclusion).
- Run time about 70 min on CPU.

**Code and output**
- Script: `code/c1_benchmark/c1_pilot.py`. Output: `results/c1_cross/c1_pilot_metrics.json`.

**Test sets**
| Domain | Source | n |
|---|---|---|
| Adult | FER2013 privateTest, P(age 20+) ≥ 0.9 | 2,327 |
| Child | FER2013 (all splits), P(age 0–9) ≥ 0.9 (`nateraw/vit-age-classifier`); visually verified on 48 random samples (`child_sample_sheet.png`) | 4,582 |
| Autistic | Kaggle "Autistic Children Emotions – Dr. Fatma M. Talaat" (6 classes, no disgust), train folder | 1,199 |
| Autistic (small) | Same dataset, test folder (heavily augmented) | 226 |

## Results (accuracy / macro-F1 over 7 classes)
| Backbone | Adult | Child | Autistic (1,199) | Autistic test (226) |
|---|---|---|---|---|
| MobileNetV3-L | 0.446 / 0.397 | 0.354 / 0.298 | 0.205 / 0.137 | 0.173 / 0.117 |
| EfficientNet-B0 | 0.444 / 0.412 | 0.316 / 0.263 | 0.163 / 0.129 | 0.186 / 0.156 |
| ResNet-50 | 0.428 / 0.400 | 0.318 / 0.258 | 0.151 / 0.117 | 0.204 / 0.178 |
| **ViT-S/16** | **0.560 / 0.504** | **0.408 / 0.326** | **0.239** / 0.124 | 0.204 / 0.105 |

**Reference points:** chance accuracy is 14.3% for 7 classes and 16.7% for the 6 classes in the autism set. Macro-F1 here averages over 7 labels; because the autism set has no disgust class, its macro-F1 reads slightly lower than a 6-class F1 would.

## Accuracy drop vs adult
| Backbone | Adult → Child | Adult → Autistic |
|---|---|---|
| MobileNetV3 | −9.2 pts | −24.1 pts |
| EfficientNet-B0 | −12.8 | −28.1 |
| ResNet-50 | −11.0 | −27.7 |
| ViT-S | −15.2 | −32.1 |

## Per-class recall, ViT-S
| Domain | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|
| Adult | .45 | .43 | .33 | .80 | .43 | .70 | .54 |
| Child | .33 | .05 | .12 | .50 | .19 | .58 | .63 |
| Autistic | .50 | – | .05 | **.84** | **.01** | **.02** | **.02** |

## Findings
1. **Performance degrades step by step: adult > child >> autistic.** This holds for all 4 backbones and confirms the C1 hypothesis.
2. **On autistic children the models are near chance.** Accuracy is 15–24% against 16.7% chance, and macro-F1 is about 0.12–0.18. Models trained on adult faces are not usable for autistic learners without adaptation, which motivates C2.
3. **The predictions collapse onto a few classes.** For autistic faces, ViT predicts almost everything as happy or angry: sad, surprise, neutral and fear recall is ≤ 5%. Neutral faces of autistic children were almost never recognised (2%), even though neutral was the *easiest* class for typically developing children (63%).
4. **Children also show the expected pattern.** Fear (12%), sad (19%) and disgust (5%) are hard. Neutral and surprise transfer best.
5. **ViT-S is best on adults and children but loses the most in absolute terms.** On autistic faces, all backbones are equally poor.

## Threats to validity
- **The child labels come from an age classifier, not true ages.** The visual check was clean, but some teenagers may be included.
- **The autism dataset is a web-collected set of uncertain provenance.** Its images differ from FER2013 in resolution, colour and cropping, so part of the drop is a dataset/source shift, not purely a difference in autistic expression. It also contains near-duplicates (94 within train, 4 across train and test), and the test folder is mostly augmented copies.
- These are frozen-feature probes on 6k training images. Absolute numbers will rise with fine-tuning; the relative ranking is what C1 claims.
