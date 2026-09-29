# Comparison with other implementations (research notes, 29 Sep 2026)

Purpose: technical validation of our choices, not copying. Everything below
was read from the projects' own public READMEs / files on 29 Sep 2026.
**Their numbers are as reported by them and are NOT comparable to ours**
unless the row says so: different datasets, different truth, different units
(per building vs per pixel), and different alignment rules.

## What the problem statement fixes

- Official reference repo (IMG-PROCESS-SAC/SIH2026 and SIH-DepthWizard-2026):
  documentation only, no data or baseline numbers. It recommends **GAMUS**
  (Hugging Face `earthflow/GAMUS`) for training/testing and **SRTM 30 m** as a
  low-resolution DEM. Metrics: RMSE, MAE, correlation vs LiDAR/reference,
  across urban, sparse, hilly and forested landscapes.
- Our choices against that: GAMUS for training and the held-out test ✔;
  Copernicus GLO-30 (30 m) instead of SRTM by default, and any DEM GeoTIFF
  (SRTM, CartoDEM) via `dsm.py --dem` ✔; RMSE, MAE, r reported per landscape,
  hilly measured on USGS LiDAR because GAMUS has no terrain ✔.

## Projects compared

| project | height model | how metres are obtained | truth used for their numbers | their reported numbers | comparable to ours? |
|---|---|---|---|---|---|
| **Ours** | DINOv3-SAT ViT-L/16 (frozen) + DPT head trained on GAMUS | head outputs metres (trained on LiDAR heights); DEM added for absolute DSM | GAMUS AGL (airborne-LiDAR nDSM), 40 held-out tiles; USGS 3DEP point cloud, 8 sites | GAMUS RMSE 4.96 m, MAE 2.44 m, r 0.78 (per pixel); 3DEP full-DSM RMSE hilly 3.84 m, sparse 3.57 m | - |
| [ParmarManthanrajsinh/DepthWizard](https://github.com/ParmarManthanrajsinh/DepthWizard) | Depth Anything V2-Small (frozen) + DPT head, scale-shift-invariant + gradient loss | one global affine (a = 9.81, b = 33.73 m) fitted on a validation split | ISPRS Potsdam DSM, 150 test crops from 6 tiles | RMSE 4.63 m, MAE 4.07 m, r 0.647 | No: different dataset (Potsdam, 5 cm), DSM-based truth |
| [ArnabTechiee/depthwizard](https://github.com/ArnabTechiee/depthwizard) | shadow length ÷ tan(sun elevation) + Depth Anything V2-Base | shadow geometry | USGS 3DEP point cloud, **per building** ridge height, Fort Myers (140 buildings) | RMSE 3.88 m, MAE 3.10 m, bias −0.48 m, **r 0.067** | No: per-building, one suburb. Their own committed `data/work/fm/validation.json` shows per-pixel building RMSE 5.24 m, **r −0.27**, R² −1.43 |
| [amogh-hub/depthwizard](https://github.com/amogh-hub/depthwizard) | Depth Anything 3 mono (DA3MONO-LARGE) | Huber/IRLS fit to a DEM and/or ≥6 GCPs, leave-one-out gates | not stated in README | no numbers in README | Can't compare |
| [Vishalchandravanshiai/…TENSOR-TITANS](https://github.com/Vishalchandravanshiai/DEPTHWIZARD-SIH2026-TENSOR-TITANS) | Depth Anything V2-Small | prototype outputs relative DSM only; affine to DEM/GCP planned | none yet | none | Can't compare |
| [Lucifer7636/DEPTHWIZARD](https://github.com/Lucifer7636/DEPTHWIZARD) | MiDaS / DPT | reference building height, altitude, focal length or GSD | none | none (inference speed only) | Can't compare |
| [sancharimouri/depthwizard-studio](https://github.com/sancharimouri/depthwizard-studio) | planned: DAv2 prior + nDSM fusion | planned: CartoDEM | none | none ("demo mode must not masquerade as validated ML") | Can't compare |

## Papers / public models

| work | what it is | relevance |
|---|---|---|
| GAMUS, Xiong et al., arXiv 2305.14914 | aerial RGB + LiDAR height (AGL) + classes, US cities (DFC2019/US3D origin) | the PS's recommended dataset; our training and test data |
| SynRS3D / RS3DAda, Song et al., NeurIPS 2024, arXiv 2406.18151 | synthetic RS dataset + a height model that outputs metres | our baseline: public weights **run by us** on the same 40 GAMUS tiles (6.74 m); GAMUS is not in their paper, so no published number exists to quote |
| DINOv3 (Meta, 2025), SAT-493M weights | self-supervised ViT pre-trained on 493M satellite images | our frozen encoder |
| Depth Anything V2, Yang et al., arXiv 2406.09414 | relative monocular depth, natural images | what most other teams use; the PS itself warns such models predict relative depth |

## Did we run any of them?

- **ArnabTechiee/depthwizard**: cloned (188 MB with data). Their LiDAR
  validator needs four USGS LAZ tiles (~231 MB, National Map) plus their
  Maxar imagery pipeline, so we did **not** rerun it. We checked that their
  README numbers equal their committed `data/work/fm/validation.json`
  (per building RMSE 3.883 m, r 0.0668) and read the same file's per-pixel
  results (above). Their truth construction (highest return − ground
  returns, nDSM vs nDSM) is the **same** as our LiDAR benchmark, an
  independent confirmation of our ground-truth definition.
- The others were compared at source/README level only: Potsdam needs an
  ISPRS registration; amogh-hub ships a macOS DMG; the rest publish no
  evaluation to reproduce.

## Differences that explain different numbers

- **Per building vs per pixel**: per-building errors average away the
  hard pixels (edges, trees). Ours are per pixel over whole tiles.
- **Alignment before scoring**: a scale/offset fitted on the test data
  itself (oracle) can hide a wrong scale. Parmar's oracle row (2.20 m) vs
  their frozen row (4.63 m) shows the effect. Our GAMUS numbers use no
  fitting at all; our reference-validation rows fit on half the area and
  score on the other half.
- **Truth definition**: nDSM (height above ground) vs DSM (elevation).
  DSM errors look small on hills because the terrain dominates (r ≈ 0.99);
  that is why we always also report the "DEM alone" baseline.
- **Resolution**: Potsdam 5 cm, GAMUS 0.3 m, NAIP 0.6 m, Sentinel-2 10 m.
  At 10–30 m buildings are sub-pixel, so a "DSM" there is mostly the DEM.
- **Image/LiDAR alignment**: GAMUS images are co-registered with the LiDAR
  grid. NAIP is an ordinary orthophoto, so tall buildings lean: this is
  why our downtown Pittsburgh error (35.95 m) is far above residential.
