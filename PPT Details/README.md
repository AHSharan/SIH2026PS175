# PPT Details: material for the technical slide

Everything here is generated from committed result files by scripts in `docs/`
(no hand-typed numbers). Regenerate with `python docs/ppt_materials.py` and
`python docs/lidar_sensing_figure.py`.

## Use on the one technical slide

1. **`charts/performance_comparison.png`**: the result chart.
   Left: GAMUS held-out test (40 tiles), ours 4.96 m vs RS3DAda 6.74 m vs predict-0
   9.35 m, by landscape. Right: whole pipeline vs USGS LiDAR, ours vs "DEM alone",
   per landscape including **hilly**.
2. **`lidar/lidar_sensing_overview.png`**: how LiDAR truth is made and how we are
   scored (photo → laser returns → truth grid → our model → error → profile). Real
   data from Park City, Utah.
3. Optional: **`charts/system_architecture.png`** (same as `docs/architecture_v2.png`).

## The numbers to say (all measured)

| claim | value | source |
|---|---|---|
| height error, GAMUS held-out test | **RMSE 4.96 m, MAE 2.44 m, r 0.78** | RESULTS.md |
| vs RS3DAda on the same 40 tiles | 6.74 m, r 0.60 → **26 % lower** | RESULTS.md |
| forest | 5.91 m vs 11.87 m (RS3DAda) | RESULTS.md |
| hilly terrain, full DSM vs airborne LiDAR | **3.84 m** (DEM alone 5.88 m) | results/lidar_benchmark.md |
| sparse / rural | 3.57 m (DEM alone 4.99 m) | results/lidar_benchmark.md |
| PNG upload = original | identical 4.96 m; JPG q90 4.99 m | results/png_jpg_accuracy.md |

**Do not use 4.69 m**: it is the 1,500-tile run and its results file is not in the
repo. "RME" is not a PS metric; if it means mean error, use ME (bias) = −0.52 m.

## Folder contents

```
PPT Details/
├── README.md                          this file
├── metrics/
│   ├── metrics_summary.csv            every metric row, with status and source
│   └── metrics_summary.md             tables + metric definitions
├── charts/
│   ├── performance_comparison.png/.svg
│   └── system_architecture.png
├── lidar/
│   ├── lidar_sensing_overview.png/.svg   photo → LiDAR → truth → model → error
│   └── lidar_truth_demo_examples.png     real image vs LiDAR truth vs model (3 tiles)
├── comparisons/
│   └── comparison_summary.md          other teams' methods, data, numbers; what is comparable
├── tests/
│   └── sun_elevation_tests.md         sun / compass tests: inputs, expected, results
├── audit/
│   └── requirements_audit.md          PS ↔ code ↔ tests ↔ README ↔ results
└── references/
    └── references.md
```

## Live demo commands

```bash
python serve.py --live          # upload, run, validate, 3D (or double-click start_demo.bat)
python serve.py --demo-lidar    # real image vs LiDAR truth (GAMUS test tiles)
```

## Limitations to put last on the slide

Trained on US aerial imagery (accuracy on ISRO imagery not yet measured) · tall
towers and dense 25 m canopy under-predicted (downtown 35.95 m DSM RMSE) · GAMUS
urban 5.28 m vs RS3DAda 5.11 m · Copernicus DEM already contains some canopy.
