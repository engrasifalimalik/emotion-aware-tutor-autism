# C4: Expected Full-Scale Results

> **ILLUSTRATIVE – literature-based estimate, not an actual run.** These numbers are extrapolated from the real pilot (2,297 students) plus published knowledge-tracing results on ASSISTments 2012. Verify the cited ranges before relying on them.

| Setting | Expected AUC | Basis |
|---|---|---|
| XGBoost, cognitive, all ~46k students | 0.70–0.72 | Pilot 0.704; more data usually adds about +0.005–0.015 for tree models |
| XGBoost, fused | 0.71–0.72 (Δ ≤ +0.005 over cognitive) | Pilot Δ = +0.001; the affect values are log-derived, so they are redundant |
| DKT (LSTM knowledge tracing), cognitive | 0.72–0.74 | Published DKT on ASSIST2012 ≈ 0.72–0.73 (verify) |
| AKT / SAKT (attention-based), cognitive | 0.74–0.78 | Published AKT ≈ 0.77 on ASSIST2012 (verify) |
| AKT + affect features | +0.003–0.01 over AKT | Prior affect-aware KT papers report small gains (verify) |
| Rule engine, emotion-aware vs perf-only recall | +1–4 pts | Pilot +2.1 pts |

**Main point to compare with your run:** if fusion gives more than +0.02 AUC, check for leakage. The current-row affect values must never be used; only past ones. Gains from affect should be small on this dataset.
