# DepthWizard — SIH 2026 PS 26175 (ISRO / SAC)

Single-view RGB remote-sensing image → **metric height map (nDSM)** → DSM,
with an interactive 3D flythrough and an honest evaluation harness.

## Core design decision

We predict **nDSM (height above ground, in metres)**, *not* perspective depth.

Nadir/ortho imagery has no depth geometry — there is no perspective, no focal
cue, and every pixel is effectively the same distance from the sensor. Metric
*depth* models (Depth Pro, UniDepth, Metric3D) rely on focal-length priors that
are meaningless here. The PS itself names this failure mode:

> "foundational monocular depth models are trained largely on natural egocentric
> imagery and predict relative depth... Converting relative depth into metric
> elevation remains a critical challenge."

Final product: `DSM = DEM(terrain) + nDSM(objects)`.

## Status

| Component | State |
|---|---|
| `data.py` — GAMUS loader + deterministic tile selection | done, verified against real files and the Kaggle tile set |
| `eval.py` — metrics harness | done, self-tested (0.13 m noise floor) |
| `predict.py` — tiled inference, GSD normalisation, TTA | done, 5/5 CPU tests, **ran on Kaggle T4** |
| `calibrate.py` — ground shift + robust affine | done, 5/5 tests on real tiles |
| `train_head.py` — frozen DINOv3-SAT + trained DPT head | done, 19/19 CPU tests (`tests/`) — **real run pending on Colab** |
| `colab_train.py` — cache -> train -> eval, resumable | done, full dry run + disconnect/resume verified |
| `web/` — Three.js flythrough | working; prediction mode (error overlay, live RMSE) verified |

**Measured so far** (RS3DAda zero-shot, 40 GAMUS test tiles, see `progression.md`):
overall RMSE 6.74 m / MAE 3.47 m / r 0.60; urban 5.11 m; sparse 2.83 m; forest 11.87 m.

## Where to start

- **Training the model** (no coding needed): [`COLAB.md`](COLAB.md)
- **Submission form text + slide content**: [`SUBMISSION.md`](SUBMISSION.md)
- **Every measured number**: [`progression.md`](progression.md), and `RESULTS.md` from the Colab run

## Verified dataset facts (measured, not assumed)

GAMUS (`earthflow/GAMUS`, CC-BY-4.0) — read from the actual files:

- Files are **HDF5**, not GeoTIFF. One dataset per file, key `"image"`.
- Layout `{images,heights,classes}/{train,val,test}/{STEM}_{RGB|IMG,AGL,CLS}.h5`
- **The RGB suffix is not uniform**: DC/PHL use `_RGB`, NYC uses `_IMG`.
  Pairing on `_RGB` alone silently drops all 1000 NYC test tiles and 1167 train
  tiles — i.e. exactly the city with the tall buildings.
- Modality dirs have equal counts but are **not stem-aligned**; intersect them.
- RGB `(1024,1024,3)` uint8 · AGL `(1024,1024)` float32 metres · CLS 7 classes
- **No geotransform, no CRS, no nodata attribute exists.**
- Height range observed: −5.0 to 146.4 m (PHL has 100 m+ towers).
- No declared nodata. DC tiles carry an exact `−5.0` sentinel; NYC's small
  negatives (to −3.1) are genuine LiDAR noise. These are handled differently.
- Class codes identified by correlating against height:
  `3=building`, `6=tree`, `4=water` → buckets are derived automatically.

**GSD = 0.30 m is asserted, not read** — the files carry no transform. It comes
from the DFC2019 US3D spec, cross-checked by measuring parked cars (~15 px at a
4.5 m car). Confirm against the GAMUS paper before trusting downstream metres.

## Evaluation harness

`eval.py` reports RMSE, MAE, Pearson r, mean bias and threshold accuracy
(<1 m, <2.5 m, <5 m), overall and per landscape bucket, matching the PS's
"urban, sparse, hilly, forested" wording.

**Harness self-test**: feeding ground truth in as its own prediction gives
RMSE 0.13 m, not 0.00. That residual is fully explained — 6.5% of GT pixels are
negative and predictions are clamped at ≥0 (verified: clamping alone produces
0.1332 m). So **0.13 m is the noise floor** of this evaluation; no model can
score below it here.

`hilly` is never auto-assigned: nDSM has terrain removed by construction, so a
hilly bucket cannot be derived from it. It requires a DEM.

## Running

```bash
python data.py --fetch 12 --split test --n 5      # fetch + describe real tiles
python make_baseline.py --mode zero --out preds_zero
python eval.py --pred preds_zero --gt gamus/test/heights --gt-glob "*_AGL.h5" \
               --auto-buckets-cls gamus/test/classes --tag "B0 floor"
python export_viewer.py --n 3                     # build viewer assets
python serve.py                                   # http://localhost:8777
```

GPU inference runs on Kaggle — see `kaggle_run.py`. Training runs on Colab — see `COLAB.md`.

```bash
python tests/test_train_head_cpu.py                # 19 CPU tests, needs gamus/test tiles
```

Viewer with real predictions: copy the Colab `viewer_assets/` into `web/assets_pred/`, then open `http://localhost:8777/?assets=assets_pred`.

## Model backend

RS3DAda (`JTRNEO/SynRS3D`, MIT) — ViT-L + DPT, outputs metres, trained on
0.05–1.0 m GSD synthetic remote-sensing data. Interface read from their
`infer_height.py`, not guessed:

- `DPT_DINOv2(encoder='vitl', head_configs=[{'name':'regression','nclass':1},
  {'name':'segmentation','nclass':8}], pretrained=False)`
- normalisation mean `(123.675,116.28,103.53)` std `(58.395,57.12,57.375)` with
  `max_pixel_value=1` — **the image is NOT divided by 255 first**
- patch size 1022 (must be divisible by 14), output key `'regression'`

## Licences

RS3DAda / SynRS3D: MIT · GAMUS: CC-BY-4.0 · DINOv3: Meta custom licence
(gated) · Three.js: MIT
