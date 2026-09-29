# Final audit: problem statement ↔ code ↔ tests ↔ README ↔ results (29 Sep 2026)

Method: every row was checked against the code and, where marked **run**,
executed on this machine (Windows 11, RTX 3060 laptop, Python 3.13, Node 24).
"Read" means checked in source only.

## 1. PS requirements

| PS requirement | implementation | evidence | status |
|---|---|---|---|
| Non-georeferenced PNG/JPG → rDSM | `dsm.run`: `ndsm.tif` + `rdsm_0to1.png/.npy`; `--gsd`, `--gsd-unknown` | `tests/test_dsm_cpu.py` (run, 47/47); live upload of JPG (earlier session) | ✔ |
| Georeferenced GeoTIFF → absolute metric DSM | `dsm.tif` = DEM + nDSM, float32, input CRS/grid | Park City run through `/api/run` (run); 3DEP benchmark 8 sites (run) | ✔ |
| Pre-trained backbone | frozen DINOv3-SAT ViT-L/16 | `train_head.py`; encoder self-check in RESULTS.md | ✔ (a satellite foundation model rather than a *depth* model; deliberate, explained in README) |
| "initial relative-depth maps" | not used: the head predicts metric height directly | README "Key design decision" | ⚠ different from the PS's suggested route, by design; the PS's goal (metric elevation) is met and measured |
| Low-res DEM or GCPs → absolute | Copernicus GLO-30 (default), `--dem` any DEM (SRTM, CartoDEM), `--gcp` ×3+ for PNG/JPG | `dsm.py`; test_dsm_cpu GCP cases (run) | ✔ |
| Scale calibration module | learned metric scale + GSD normalisation + DEM; `validate.py` robust affine (`calibrate.fit_affine`) with held-out scoring, writes `dsm_calibrated.tif` | `tests/test_validate_cpu.py` (run, 7/7); Park City: raw 3.26 → affine 2.99 m held-out (run) | ✔ (was: `calibrate.py` not connected; fixed in this pass) |
| Image projected onto a 3D mesh, Three.js | identity UVs on the image grid | `web/index.html` | ✔ |
| First-person navigation; heights and slopes | Fly (W A S D, Q/E), Orbit, probe, slope mode, profile | browser checks (run) | ✔ |
| Upload imagery, visualise, validate against reference | live mode `/api/run`, `/api/validate`, accuracy panel, error map | `tests/test_live_cpu.py` (run, 21/21); browser (run) | ✔ |
| RMSE, MAE, correlation vs LiDAR across urban/sparse/hilly/forest | GAMUS 40 tiles (urban/sparse/forest); 3DEP LiDAR 8 sites incl. hilly | RESULTS.md; `results/lidar_benchmark.*` (rerun today, identical) | ✔ |
| DSM in a standard geospatial format | GeoTIFF (LZW, nodata −9999, CRS kept) | `dsm.write_tif` | ✔ |
| Standalone deployment | local server + browser; `start_demo.bat`; three.js bundled; DINOv3 offline once cached | offline model load (run) | ⚠ needs Python + packages; not a single installer |

## 2. Issues found and fixed in this pass

| # | issue | fix | commit |
|---|---|---|---|
| 1 | Headline **4.69 m** (1,500-tile run) quoted in TALKING_POINTS, DSM.md, RUN_DSM_SIMPLE.md, dsm.py and the slide diagram, but **no results file for it exists** in the repo; the demo laptop runs the 480-tile model (4.96 m) | all user-facing places now use the verified 4.96 m (26 % lower than RS3DAda); 4.69 m labelled "reported, not verifiable" | this pass |
| 2 | Slide diagram mixed two models: 4.69 m (1,500 tiles) next to correlation 0.78 vs 0.60 (480 tiles) | diagram regenerated with one model's numbers | this pass |
| 3 | Viewer **sun azimuth** and **compass** mirrored north–south (azimuth 0 lit from the south; needle pointed south when looking north) | maths moved to `web/lib/geo.js`, fixed, 7 tests | b5cec25 |
| 4 | Measure-distance profile panel stayed open after turning Measure off | closes with Measure | b5cec25 |
| 5 | `calibrate.fit_affine` only used for scoring; no calibrated product | `validate.py` writes `dsm_calibrated.tif` + stores s, t | this pass |
| 6 | `tests/test_live_cpu.py` left a **fake-height scene** in the real viewer list after every run | the test now deletes what it created (verified: 0 files left) | this pass |
| 7 | `dsm.py` docstring said `--gsd` is required for PNG/JPG (outdated: `--gsd-unknown`, `--gcp` exist) | docstring corrected | this pass |
| 8 | README described an older state (e.g. "40/40 CPU tests", "hilly is never measured") | rewritten against the code | this pass |
| 9 | 3DEP 2 m DSM/HAG rasters used as truth in the first benchmark draft (canopy 8.9 m vs 24.4 m in the points; 150 m spikes) | truth rebuilt from the point cloud | ea84076 |

## 3. Tests (all run on 29 Sep 2026)

| suite | result |
|---|---|
| `python tests/test_dsm_cpu.py` | **47/47 passed** |
| `python tests/test_live_cpu.py` | **21/21 passed** |
| `python tests/test_validate_cpu.py` | **7/7 passed** |
| `python tests/test_train_head_cpu.py` | **19/19 passed** |
| `node --test tests/test_sun_geo.mjs` | **7/7 passed** |

## 4. Results re-run today

| result | re-run | outcome |
|---|---|---|
| 3DEP LiDAR benchmark (`tests/lidar_benchmark.py`) | full rerun | every per-site number identical to the committed run; new "DEM alone" baseline added |
| GAMUS 40 tiles through the upload path (`tests/png_jpg_accuracy.py`) | full rerun | see §6 |
| LiDAR demo tiles (`export_viewer.py --lidar-demo`) | run | per-tile metrics computed at full resolution |
| metric pooling in `docs/ppt_materials.py` | run | pooled per-landscape values equal the benchmark's own pixel-pooled table (independent check) |

## 5. Documented commands executed

`python dsm.py --help` · `python serve.py --help` · `python serve.py --live` (via the
preview server) · `python serve.py --demo-lidar` (URL printed; missing-assets
message checked) · `python validate.py <run> <reference>` · `python export_viewer.py
--lidar-demo ...` · `python docs/ppt_materials.py` · `python docs/lidar_sensing_figure.py`
· `python docs/architecture_v2.py` · all test commands above. Not executed today:
`colab_train.py` (Colab GPU, hours) and `kaggle_run.py` (Kaggle); their outputs are
RESULTS.md and progression.md.

## 6. GAMUS reproduction (run 29 Sep 2026, local RTX 3060, `best.pt`)

`tests/png_jpg_accuracy.py` on the same 40 held-out tiles: **RMSE 4.96 m, MAE 2.44 m,
bias −0.52 m, r 0.780; urban 5.28, sparse 2.15, forest 5.91 m**. Every accuracy
number is identical to `RESULTS.md` (Colab run, 26 Sep) and to the committed
`results/png_jpg_accuracy.md`; only the seconds-per-tile column differs (faster
today). PNG upload identical to the original; JPG q90 4.99 m.

## 7. Could not be verified

- **4.69 m** for the 1,500-tile model: no results file, checkpoint not on this machine.
- Accuracy on **ISRO / Indian imagery**: no LiDAR truth for India available to us; the
  Chungthang run is visual only (its report states "no reference, so no RMSE").
- Other teams' numbers: read from their READMEs; one project's committed output file
  was checked (see `PPT Details/comparisons/comparison_summary.md`), none were rerun.
