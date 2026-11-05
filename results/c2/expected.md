# C2: Expected Results on a Real Target

> **ILLUSTRATIVE – literature-based estimate, not an actual run.** These numbers are extrapolated from the CPU pilot plus typical domain-adaptation gains reported in the FER and transfer-learning literature. Verify before relying on them.

Setting: ViT-S, 224 px, 8 epochs on a GPU. Source = FER2013 (or RAF-DB). Target = a child or autistic dataset.

| Strategy | Target macro-F1, 0% labels | 10% labels | 50% labels |
|---|---|---|---|
| Source-only full FT (no adaptation) | 0.30–0.45 (child) / 0.20–0.35 (autistic) | – | – |
| Probe on target labels | – | +3–6 pts | +6–10 pts |
| LoRA | – | +6–10 pts | +10–15 pts |
| Full FT (source + target) | – | +7–12 pts | +12–18 pts |
| DANN | +2–6 pts (unsupervised) | +8–13 pts | ≈ full FT |

## Expected patterns
- **LoRA should come close to full FT (within about 2 pts) while training only about 1–2% of the parameters.** This is a good "data-efficient" claim for the thesis.
- **DANN helps most when there are no target labels.** Its benefit shrinks as more target labels are added.
- **The learning curve should flatten after about 25% of the target labels.**
- **Minority classes (fear, disgust) gain most from focal loss**, which typically adds about +1–3 pts macro-F1.
