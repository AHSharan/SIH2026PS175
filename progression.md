# DepthWizard - evaluation progression

Every row is a real measured run on real data.
nDSM (height above ground), metres. GAMUS test split.


## 2026-09-20 - A raw (no GSD norm, no TTA)

RS3DAda vitl DPT height | 40 GAMUS test tiles | input_gsd=0.3 model_gsd=0.5

| bucket | tiles | px | RMSE (m) | MAE (m) | bias (m) | r | <1m | <2.5m | <5m |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 40 | 41.9M | 6.82 | 3.46 | -2.11 | 0.565 | 50.2% | 66.9% | 79.8% |
| **urban** | 24 | 25.1M | 5.13 | 2.61 | -1.06 | 0.738 | 50.4% | 68.8% | 83.7% |
| **sparse** | 8 | 8.4M | 2.94 | 1.47 | -0.16 | 0.655 | 68.4% | 82.9% | 91.7% |
| **forest** | 8 | 8.4M | 12.04 | 7.97 | -7.20 | 0.250 | 31.6% | 44.9% | 56.4% |

## 2026-09-20 - B + GSD normalisation

RS3DAda vitl DPT height | 40 GAMUS test tiles | input_gsd=0.3 model_gsd=0.5

| bucket | tiles | px | RMSE (m) | MAE (m) | bias (m) | r | <1m | <2.5m | <5m |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 40 | 41.9M | 6.82 | 3.52 | -2.29 | 0.580 | 48.4% | 65.3% | 79.5% |
| **urban** | 24 | 25.1M | 5.23 | 2.74 | -1.35 | 0.738 | 47.5% | 66.5% | 83.2% |
| **sparse** | 8 | 8.4M | 2.87 | 1.45 | -0.32 | 0.670 | 69.6% | 82.9% | 91.5% |
| **forest** | 8 | 8.4M | 11.93 | 7.92 | -7.10 | 0.285 | 30.0% | 44.1% | 56.3% |

## 2026-09-20 - C + GSD + tiling/TTA

RS3DAda vitl DPT height | 40 GAMUS test tiles | input_gsd=0.3 model_gsd=0.5

| bucket | tiles | px | RMSE (m) | MAE (m) | bias (m) | r | <1m | <2.5m | <5m |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 40 | 41.9M | 6.74 | 3.47 | -2.27 | 0.597 | 48.6% | 65.7% | 79.9% |
| **urban** | 24 | 25.1M | 5.11 | 2.68 | -1.33 | 0.757 | 47.4% | 66.9% | 83.8% |
| **sparse** | 8 | 8.4M | 2.83 | 1.43 | -0.32 | 0.684 | 70.6% | 83.1% | 91.5% |
| **forest** | 8 | 8.4M | 11.87 | 7.87 | -7.07 | 0.306 | 30.1% | 44.5% | 56.7% |
