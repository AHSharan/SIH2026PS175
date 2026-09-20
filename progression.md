# DepthWizard - evaluation progression

Every row is a real measured run on real data.
nDSM (height above ground), metres. GAMUS test split.


## 2026-09-21 - B0 floor: predict 0 everywhere

Not a model. RMS of the true heights - the number any model must beat.

| bucket | tiles | px | RMSE (m) | MAE (m) | bias (m) | r | <1m | <2.5m | <5m |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 28 | 29.3M | 9.02 | 5.04 | -4.99 | nan | 50.8% | 56.6% | 64.3% |
| **urban** | 14 | 14.7M | 8.65 | 4.82 | -4.82 | nan | 46.5% | 53.3% | 62.3% |
| **sparse** | 7 | 7.3M | 5.63 | 2.49 | -2.29 | nan | 71.0% | 76.3% | 81.9% |
| **forest** | 7 | 7.3M | 12.03 | 8.09 | -8.08 | nan | 38.9% | 43.4% | 50.6% |
