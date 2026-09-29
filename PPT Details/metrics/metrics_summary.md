# Metrics summary (every number measured; source in each table)

Metric definitions (per pixel, e = prediction − reference, metres):

- **RMSE** = sqrt(mean(e²)); **MAE** = mean(|e|); **ME (bias)** = mean(e): negative = too low
- **r** = Pearson correlation of prediction and reference (the PS's 'correlation')
- **within x m** = share of pixels with |e| < x
- The PS asks for RMSE, MAE and correlation. 'RME' is not defined in the PS; if it was meant as mean error, that is the ME (bias) column. No other 'RME' number is reported here.

## 1. GAMUS held-out test: 40 tiles, airborne-LiDAR height above ground (nDSM)

Same 40 tiles for every row. Model chosen on the validation split, never on test. Source: `RESULTS.md`; reproduced locally by `tests/png_jpg_accuracy.py`.

| method | RMSE | MAE | ME (bias) | r | within 1 m | within 2.5 m | within 5 m |
|---|---|---|---|---|---|---|---|
| Predict 0 m (floor) | 9.35 m | 5.10 m | -5.05 m | - | 49.8% | 55.4% | 63.6% |
| RS3DAda public weights (run by us) | 6.74 m | 3.47 m | -2.27 m | 0.597 | 48.6% | 65.7% | 79.9% |
| **Ours** (DINOv3-SAT + head) | 4.96 m | 2.44 m | -0.52 m | 0.780 | 52.5% | 70.5% | 85.4% |

By landscape (RMSE / MAE / r):

| landscape | tiles | predict 0 | RS3DAda | ours |
|---|---|---|---|---|
| urban | 24 | 8.81 / 4.74 / - | 5.11 / 2.68 / 0.76 | **5.28 / 2.40 / 0.73** |
| sparse | 8 | 4.21 / 1.94 / - | 2.83 / 1.43 / 0.68 | **2.15 / 1.12 / 0.83** |
| forest | 8 | 13.66 / 9.35 / - | 11.87 / 7.87 / 0.31 | **5.91 / 3.87 / 0.81** |

## 2. Independent check: USGS 3DEP airborne LiDAR, 8 US sites, full pipeline

NAIP 0.6 m GeoTIFF → `dsm.run` (our model + Copernicus GLO-30) → DSM, scored on a 2 m grid against truth rebuilt from the 3DEP point cloud. Source: `results/lidar_benchmark.json`.

| landscape | sites | full DSM RMSE | full DSM MAE | DSM without model (DEM alone) RMSE | height above ground RMSE |
|---|---|---|---|---|---|
| urban | 2 | 25.71 m | 14.43 m | 26.76 m | 27.22 m |
| sparse | 3 | 3.57 m | 2.31 m | 4.99 m | 6.30 m |
| hilly | 2 | 3.84 m | 2.85 m | 5.88 m | 4.79 m |
| forest | 1 | 7.95 m | 5.92 m | 8.43 m | 20.49 m |

Per site:

| site | landscape | full DSM RMSE | DEM alone RMSE | nDSM RMSE | nDSM r |
|---|---|---|---|---|---|
| pittsburgh_downtown | urban | 35.95 | 37.20 | 37.81 | 0.14 |
| pittsburgh_residential | urban | 5.40 | 6.95 | 7.24 | 0.48 |
| pittsburgh_mt_washington | hilly | 4.26 | 6.62 | 5.76 | 0.64 |
| park_city_ut | hilly | 3.37 | 5.04 | 3.55 | 0.67 |
| pennsylvania_woods | forest | 7.95 | 8.43 | 20.49 | 0.19 |
| heber_ut_farms | sparse | 1.17 | 1.36 | 0.37 | 0.86 |
| butler_pa_rural | sparse | 3.58 | 4.48 | 6.11 | 0.88 |
| washington_pa_rural | sparse | 4.91 | 7.27 | 9.02 | 0.77 |

Pooled urban includes Pittsburgh downtown (towers to 150 m, 35.95 m DSM RMSE); residential alone is 5.40 m. NAIP is not a true orthophoto, so tall buildings lean away from their LiDAR footprint.

## 3. Upload path (same 40 GAMUS tiles, `results/png_jpg_accuracy.md`)

| input | RMSE | MAE | ME | r |
|---|---|---|---|---|
| original data (direct) | 4.96 | 2.44 | -0.52 | 0.780 |
| PNG via upload path | 4.96 | 2.44 | -0.52 | 0.780 |
| JPG q90 via upload path | 4.99 | 2.45 | -0.63 | 0.780 |
| JPG q75 via upload path | 4.96 | 2.46 | -0.56 | 0.782 |

## Not verifiable here

- **4.69 m** (1,500-training-tile model, `best_1500.pt`): quoted in DSM.md and dsm.py, but no results file for that run is in the repo and the checkpoint is not on this machine. Do not use it on the slide until its RESULTS file is added.
