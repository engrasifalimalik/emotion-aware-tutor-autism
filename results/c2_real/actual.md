# C2: Adaptation to Child and Autistic Faces (REAL RUNS, CPU)

**Source:** FER2013 adult faces (6,000; child and teen faces excluded).

**Targets**
- **Child:** FER2013 child subset, 4,582 images (4,188 distinct).
- **Autism:** Talaat autism-emotions train folder, 1,199 images (1,105 distinct), 6 classes.

**Split:** each target is split 70% adaptation pool / 30% test, **grouped by perceptual hash**, so near-duplicate images never cross the split.

## Part 1: Frozen embeddings (C1 features → PCA-256 → logistic regression), 3 seeds
Script `code/c2_adaptation/c2_embed.py`, about 25 min. Raw output: `c2_embed_results.csv`, `c2_embed_summary.csv`. The maximum seed SD of any cell is 0.036.

### Autism target: macro-F1 (6 classes)
| Method @ target-label fraction | MobileNetV3 | EfficientNet-B0 | ResNet-50 | ViT-S |
|---|---|---|---|---|
| Source only (no adaptation) | 0.137 | 0.180 | 0.146 | 0.111 |
| CORAL (unsupervised) | 0.157 | 0.163 | 0.129 | 0.113 |
| Combined @ 5% | 0.215 | 0.243 | 0.230 | 0.296 |
| Combined @ 10% | 0.245 | 0.279 | 0.244 | 0.343 |
| Combined @ 25% | 0.351 | 0.357 | 0.296 | 0.442 |
| Combined @ 50% | 0.438 | 0.430 | 0.358 | 0.536 |
| Combined @ 100% | 0.513 | 0.503 | 0.433 | 0.621 |
| Target only @ 10% | 0.305 | 0.293 | 0.310 | 0.393 |
| Target only @ 50% | 0.494 | 0.471 | 0.426 | 0.560 |
| Target only @ 100% (~844 imgs) | 0.562 | 0.525 | 0.479 | **0.658** |

### Child target: macro-F1 (7 classes)
| Method | MobileNetV3 | EfficientNet-B0 | ResNet-50 | ViT-S |
|---|---|---|---|---|
| Source only | 0.320 | 0.295 | 0.296 | 0.326 |
| CORAL | 0.302 | 0.294 | 0.268 | 0.335 |
| Combined @ 10% | 0.363 | 0.337 | 0.325 | 0.395 |
| Combined @ 50% | 0.424 | 0.395 | 0.373 | 0.454 |
| Target only @ 100% | 0.486 | 0.446 | 0.407 | **0.503** |

## Part 2: End-to-end fine-tuning on the autism target (MobileNetV3, 112 px, 3 epochs, seed 42)
Script `code/c2_adaptation/c2_adapt.py`, about 93 min in total. Raw output: `finetune/*.json`.

| Strategy | Autism labels | Autism acc | Autism macro-F1 | Adult (FER test) acc | Train time |
|---|---|---|---|---|---|
| Full FT, source only (3 seeds) | 0% | 0.170 ± 0.026 | 0.134 ± 0.028 | 0.510 ± 0.007 | 17 min |
| **DANN** (unlabelled autism, 3 seeds) | 0% | **0.304 ± 0.017** | **0.228 ± 0.016** | 0.478 ± 0.006 | 42 min |
| Full FT + autism labels (3 seeds) | 25% (211) | 0.460 ± 0.029 | 0.460 ± 0.036 | 0.504 ± 0.013 | 13 min |
| Full FT + autism labels (3 seeds) | 50% (422) | **0.577 ± 0.010** | **0.581 ± 0.002** | 0.499 ± 0.013 | 21 min |

Frozen MobileNetV3 at the same label fractions scored 0.351 (25%) and 0.438 (50%). **Fine-tuning adds +0.14–0.15 macro-F1.**

## Findings
1. **Without adaptation, adult-trained FER fails on autistic children** (macro-F1 0.10–0.18, about chance). This holds both for frozen probes and after full fine-tuning on adult data.
2. **Unsupervised adaptation:** CORAL on frozen features does **not** help (Δ ≈ 0). End-to-end **DANN improves macro-F1 by +0.094 ± 0.038** (0.134 → 0.228, mean ± SD over seeds 42/1/2; gain positive on every seed: +0.131, +0.056, +0.095) without any autism labels, at a cost of −3.2 ± 0.6 pts adult accuracy. The single-seed "doubling" was optimistic because seed 42 had the weakest baseline. Only 3 seeds: treat as indicative.
3. **Few labels help a lot.** With ViT-S, 10% of the autism labels (~84 images) gives 0.34 macro-F1, three times the baseline. Gains continue roughly log-linearly up to 100%.
4. **Fine-tuning beats frozen features** at equal label budgets (+0.15 macro-F1 for MobileNetV3), while adult performance drops only about 2–3 pts. Little is forgotten.
5. **Adult data helps only in the low-label regime.** At ≥ 50% target labels, *target-only* beats *combined*, which suggests a real domain conflict between adult and autistic expression. Mixing in adult data then hurts.
6. **ViT-S is the best frozen backbone after adaptation** (0.658 vs ≤ 0.562), even though it was the worst before adaptation. Its features are rich but miscalibrated for this domain.

## Limitations
- The autism dataset comes from web sources of uncertain provenance. It has 6 classes, and its images differ in style from FER2013.
- The fine-tuning results are a single seed at 112 px with 3 epochs, all on CPU. GPU runs (see `KAGGLE_GUIDE.md`) would raise the absolute numbers.
- The child labels come from an age classifier.
