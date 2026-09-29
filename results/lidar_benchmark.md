# Benchmark against airborne LiDAR (USGS 3DEP) on georeferenced imagery

Model `best.pt`, run through the real GeoTIFF pipeline (`dsm.run`: our model + Copernicus GLO-30 terrain). Imagery: USDA NAIP. Truth: USGS 3DEP LiDAR point cloud, rebuilt on a 2 m grid (highest non-noise return minus ground returns, 3 × 3 median against isolated spikes). Compared on a common 2 m grid (our 0.6 m output averaged to 2 m). 600 × 600 m per site; photo and LiDAR 0–2 years apart. All numbers measured by `tests/lidar_benchmark.py`.

## By landscape (pooled pixels)

| landscape | sites | heights above ground: RMSE / MAE / bias / r | predict-zero RMSE | full DSM: RMSE / MAE / bias / r | terrain only (Copernicus vs LiDAR): RMSE |
|---|---|---|---|---|---|
| **urban** | 2 | 27.22 | 13.76 | -9.98 | 0.324 | 31.96 | 25.71 | 14.43 | +0.23 | 0.874 | 14.03 |
| **sparse** | 3 | 6.30 | 3.22 | -3.12 | 0.849 | 9.07 | 3.57 | 2.31 | -0.80 | 1.000 | 5.53 |
| **hilly** | 2 | 4.79 | 3.14 | -2.27 | 0.624 | 7.77 | 3.84 | 2.85 | -0.26 | 1.000 | 3.62 |
| **forest** | 1 | 20.49 | 18.78 | -18.53 | 0.194 | 24.22 | 7.95 | 5.92 | -0.41 | 0.918 | 19.40 |
| **all** | 8 | 16.07 | 7.78 | -6.55 | 0.454 | 19.35 | 13.47 | 5.92 | -0.36 | 1.000 | 10.54 |

## By site

| site | landscape | LiDAR / photo year | relief (m) | median height LiDAR / ours (m) | heights: RMSE / MAE / bias / r | full DSM: RMSE / MAE / bias / r | terrain RMSE | run (s) |
|---|---|---|---|---|---|---|---|---|
| pittsburgh_downtown | urban | 2019 / 2019 | 14 | 16.4 / 11.0 | 37.81 | 22.34 | -15.23 | 0.137 | 35.95 | 24.75 | +1.53 | 0.162 | 19.43 | 6.7 |
| pittsburgh_residential | urban | 2019 / 2017 | 27 | 7.0 / 1.7 | 7.24 | 5.18 | -4.73 | 0.479 | 5.40 | 4.11 | -1.06 | 0.781 | 4.06 | 3.4 |
| pittsburgh_mt_washington | hilly | 2019 / 2019 | 132 | 5.3 / 2.0 | 5.76 | 3.67 | -3.16 | 0.638 | 4.26 | 3.01 | -0.72 | 0.980 | 4.38 | 5.0 |
| park_city_ut | hilly | 2019 / 2021 | 162 | 3.7 / 2.5 | 3.55 | 2.61 | -1.37 | 0.669 | 3.37 | 2.69 | +0.20 | 0.996 | 2.64 | 6.0 |
| pennsylvania_woods | forest | 2019 / 2019 | 79 | 24.7 / 3.3 | 20.49 | 18.78 | -18.53 | 0.194 | 7.95 | 5.92 | -0.41 | 0.918 | 19.40 | 3.6 |
| heber_ut_farms | sparse | 2019 / 2018 | 7 | 0.0 / 0.0 | 0.37 | 0.07 | -0.03 | 0.857 | 1.17 | 1.10 | -1.10 | 0.980 | 1.08 | 5.0 |
| butler_pa_rural | sparse | 2019 / 2019 | 48 | 0.1 / 0.1 | 6.11 | 2.90 | -2.73 | 0.875 | 3.58 | 2.08 | -0.02 | 0.844 | 6.61 | 5.2 |
| washington_pa_rural | sparse | 2019 / 2019 | 60 | 8.6 / 1.7 | 9.02 | 6.71 | -6.60 | 0.772 | 4.91 | 3.75 | -1.29 | 0.967 | 6.85 | 5.0 |

Notes: LiDAR heights use NAVD88 and Copernicus uses EGM2008, so part of the DSM bias is a datum offset. NAIP is an ordinary orthophoto, not a true orthophoto: tall buildings lean in the picture, so their roofs sit metres away from their LiDAR footprints. This inflates downtown error for any image-based method. The ready-made 3DEP 2 m DSM/HAG products were not used as truth: checked against the point cloud, their DSM puts forest canopy at 8.9 m (points: 24.4 m) and their HAG has spikes up to 150 m on flat farmland. Copernicus GLO-30 is a radar surface model, so it already holds part of the forest canopy and some buildings: that is why its 'terrain' error is large in forest and downtown while our full DSM there stays much closer.
