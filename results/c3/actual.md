# C3: Facial Emotion → Learning-Centred States on DAiSEE (REAL RUN, CPU)

**Data:** DAiSEE (Kaggle mirror `olgaparfenova/daisee`), official subject-disjoint splits.
- Train: 5,358 clips. Validation: 1,429. Test: 1,784.

**Features**
- 4 frames per 10-second clip; face cropped with an OpenCV Haar cascade. A face was found in 97% / 91% / 94% of frames for train / validation / test.
- Each frame goes through a frozen ViT-S/16, giving two feature sets:
  1. **emo:** 7 emotion probabilities from the C1 adult FER2013 probe (children excluded), averaged over frames.
  2. **emb:** ViT-S features (mean + SD over frames) → PCA to 128 dimensions.
- About 85 min of extraction on CPU.

**Classifier and evaluation**
- Class-balanced logistic regression. C and the decision threshold are tuned on Validation; scores are reported on Test.
- 95% CIs come from 1,000 bootstrap resamples of test clips.

**Scripts and output:** `code/c3_learning_states/c3_extract.py`, `c3_classify.py`, `results/c3/c3_metrics.json`.

## Binary learning states (Test)
Columns give AUC [95% CI] and macro-F1 for each feature set.

| State (positive rate) | Majority-class macro-F1 | emo (7-d): AUC | emo: F1 | emb (128-d): AUC | emb: F1 | emo+emb: AUC | emo+emb: F1 |
|---|---|---|---|---|---|---|---|
| Bored ≥1 (54%) | 0.350 | **0.610** [.583–.636] | **0.580** | 0.562 [.535–.590] | 0.544 | 0.583 | 0.565 |
| Confused ≥1 (33%) | 0.402 | 0.641 [.614–.668] | 0.592 | **0.646** [.620–.673] | 0.598 | 0.640 | **0.602** |
| Frustrated ≥1 (22%) | 0.438 | **0.612** [.582–.645] | **0.579** | 0.586 [.552–.618] | 0.553 | 0.586 | 0.542 |
| Low engagement ≤1 (5%) | 0.487 | **0.637** [.585–.687] | 0.519 | 0.610 [.546–.674] | **0.554** | 0.624 | 0.540 |

## 4-level engagement (Test)
| Features | Accuracy | Balanced accuracy | Majority-class accuracy |
|---|---|---|---|
| emo | 0.507 | 0.265 | 0.494 |
| emb | 0.522 | 0.274 | 0.494 |
| emo+emb | 0.520 | 0.273 | 0.494 |

## Emotion → state mapping (standardised logistic coefficients, emo model)
| State | Strongest positive | Strongest negative |
|---|---|---|
| Bored | fear-like +0.17, **neutral +0.16**, sad +0.14 | **happy −0.25**, angry −0.15 |
| Confused | **sad +0.20**, fear +0.14 | **happy −0.22** |
| Frustrated | fear +0.12, sad +0.07 | **happy −0.21** |
| Low engagement | **fear +0.38**, neutral +0.15 | surprise −0.19, angry −0.11 |

## Findings
1. **Basic emotions carry a real but modest signal for learning states.** All 4 AUCs are 0.61–0.64, and every 95% CI excludes 0.5. Macro-F1 beats the majority-class baseline by +0.14–0.23 for boredom, confusion and frustration.
2. **A 7-number emotion vector is as good as the full 128-d face features, or better.** It wins for boredom (+0.05 AUC), frustration (+0.03) and low engagement (+0.03), and ties for confusion. So the emotion layer is a **compact, interpretable bridge** to learning states. This supports the C3 design.
3. **The mapping makes sense.** A lack of "happy" predicts every negative learning state. Sad and fear-like expressions go with confusion and frustration. Neutral and fear-like expressions go with boredom and disengagement; the "fear" output on webcam faces likely captures a wide-eyed, staring look.
4. **4-level engagement is barely above the majority baseline** (0.52 vs 0.49; balanced accuracy 0.27). This is DAiSEE's well-known imbalance problem: very few clips have levels 0–1. Published video models report about 0.5–0.6 accuracy (verify), so frame-averaged static features are not enough here.

## Limitations
- Only 4 frames per clip and static features; there is no temporal modelling (e.g., an LSTM over frames, or head-pose and gaze dynamics).
- The FER probe is a frozen adult FER2013 model with about 0.53 accuracy in-domain (C1), so errors in the emotion signal pass through to C3.
- DAiSEE participants are Indian college-age adults, not autistic primary-school children. C3 validates the *mapping*, not its transfer to autistic learners.
- The DAiSEE labels are crowd-annotated and noisy, and inter-rater agreement is known to be moderate.
