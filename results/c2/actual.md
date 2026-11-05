# C2: CPU Pilot Results (REAL RUN, pipeline check)

- **Setup:** MobileNetV3-L, 112 px, 3 epochs, batch size 64, lr 1e-3 (OneCycle), class-weighted cross-entropy, seed 42.
- **Training data:** 6,000 FER2013 training images, plus 10% of the stand-in target's labels (251 images).
- **Stand-in target:** FER2013 publicTest, used because no child data is available yet. It comes from the **same domain** as the source, so this pilot measures fine-tuning gains, **not** domain shift.
- **Target test:** 1,080 images. **Source test:** FER2013 privateTest (3,589 images).
- **Script and output:** `code/c2_adaptation/c2_adapt.py`, `results/c2/*.json`.

| Strategy | Trainable params | Train time (CPU) | Target acc | Target macro-F1 | Source acc | Source macro-F1 |
|---|---|---|---|---|---|---|
| Probe (frozen) | 0.01 M | 4 min | 0.317 | 0.278 | 0.319 | 0.280 |
| **Full fine-tune** | 4.21 M | 10 min | **0.495** | **0.446** | **0.499** | **0.440** |
| DANN | 4.54 M | 24 min | 0.483 | 0.434 | 0.493 | 0.431 |

## Findings
1. **Full fine-tuning beats the frozen probe by about 18 points of accuracy** after only 3 CPU epochs on 6k images. Adapting the backbone matters far more than the classifier on top.
2. **DANN gives no gain here (−1.2 pts), as expected.** The source and stand-in target come from the same distribution, so there is no domain gap to close. DANN should only help on real adult → child or autistic data. This run confirms that the DANN code runs, nothing more.
3. **The probe in this run (0.32) is below the C1 probe (0.44).** C1 used 224 px, no augmentation and a fully converged solver (LBFGS). This run used 112 px, augmentation and only 3 SGD epochs. Use the C1 numbers as the frozen baseline.
4. Source and target scores are nearly identical, which confirms that the stand-in target has no shift.

## Next (GPU, real target)
Run the grid in `code/c2_adaptation/KAGGLE_GUIDE.md` on ViT-S at 224 px with CAFE or DE-ENIGMA as `--target`, and produce learning curves over `--target_frac`.
