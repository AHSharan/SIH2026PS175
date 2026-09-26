# SIH 2026 — PS 26175 submission kit (DepthWizard)

Everything the submission form asks for, ready to paste. **Every number here was
measured** (sources: `progression.md` for the Kaggle runs, and `RESULTS.md` once
the Colab training finishes).

Anything in `[[double brackets]]` must be filled in from `RESULTS.md` — or
**deleted** if the DINOv3 head does not beat the baseline (see the rule at the
bottom). Never submit a bracket, and never type in a number that did not come
from a run.

---

## 1. Idea Title (max 100 characters)

```
DepthWizard: Metric Height from One Satellite Image, DEM-Calibrated, with a 3D Flythrough
```

## 2. Technology Bucket

**Disaster Management** — this is the bucket the PS itself is listed under.

## 3. Abstract / Summary (max 10,000 characters)

```
DepthWizard turns a single optical RGB satellite or aerial image into a metric Digital Surface Model (DSM) and an interactive 3D flythrough.

Core idea - predict height, not depth. Monocular depth models learn perspective cues (focal length, vanishing lines, objects shrinking with distance) from ground-level photos. A nadir satellite image has no perspective: every pixel is roughly the same distance from the sensor, so "metric depth" is the wrong quantity to estimate. The problem statement names this gap itself. We instead regress the normalised DSM (nDSM, height above ground in metres) directly, and compose DSM = DEM (terrain from SRTM/CartoDEM) + nDSM (buildings and trees). Non-georeferenced PNG/JPG inputs produce a relative DSM.

Pipeline:
(1) Elevation extraction with pre-trained backbones: RS3DAda (ViT-L + DPT, trained on synthetic remote-sensing data) as a zero-shot baseline, and a frozen DINOv3 ViT-L encoder pre-trained on 493M satellite images, with a DPT-style height head that we train. Inference uses ground-sampling-distance (GSD) normalisation, 25%-overlap tiling with Gaussian-feathered blending, and flip test-time augmentation.
(2) Scale calibration: a robust affine fit to low-resolution DEM samples or a few Ground Control Points (RANSAC + Huber, bounded scale), a per-block ground-shift field, and class-wise offsets taken from the model's own segmentation head and fitted on a validation split, never on test data.
(3) Visualisation: a Three.js viewer that ortho-projects the optical image onto a 1.05-million-vertex terrain mesh, with first-person fly and orbit navigation, click-to-probe height, slope and error, distance/height-difference measurement, height/slope/error overlays and a live validation panel against reference data.

Measured results (GAMUS dataset, US cities, LiDAR ground truth, 40 held-out test tiles at 0.3 m):
- RS3DAda zero-shot: RMSE 6.74 m, MAE 3.47 m, correlation 0.60 overall; urban 5.11 / 2.68 m (r 0.76); sparse 2.83 / 1.43 m (r 0.68); forest 11.87 / 7.87 m (r 0.31).
- Forest is the documented failure mode: canopy is under-predicted by about 7 m on average, which is why we trained the satellite-pretrained DINOv3 head on real LiDAR heights.
- DINOv3-SAT head (ours), same 40 tiles: RMSE 4.96 m, MAE 2.44 m, correlation 0.78 overall - 26% lower RMSE than RS3DAda. Forest 5.91 m (was 11.87 m, and the canopy bias shrank from -7.07 m to -0.20 m); sparse 2.15 m (was 2.83 m). Urban is the one bucket that did not improve: 5.28 m vs RS3DAda's 5.11 m.

Rigour: every figure comes from a real run. The evaluation harness was self-tested (ground truth scored against itself gives a fully explained 0.13 m noise floor). Dataset traps were found and fixed before any number was produced - for example, one city's files use a different naming suffix, and naive pairing silently drops exactly the tiles with the tallest buildings. Negative results are reported too: GSD normalisation gave no gain for a backbone already trained across 0.05-1 m GSD. Honest limits: all accuracy is measured on US aerial data, so ISRO Cartosat imagery will differ, and DSM accuracy in hilly terrain is bounded by the 30 m DEM, not by the model.
```

## 4. Idea Description (max 50,000 characters)

```
1. PROBLEM UNDERSTANDING
The task is to recover metric elevation from ONE optical image and make it explorable in 3D. Two things make this hard. First, a single image has no stereo parallax, so height must be inferred from learned cues: shadows, roof texture, object size and context. Second, most "monocular depth" models are trained on ground-level photos and output relative, perspective depth, which has no meaning for a nadir satellite view. The evaluation scores DSM accuracy (RMSE, MAE, correlation, stability across urban, sparse, hilly and forested land) and the visualisation layer equally.

2. KEY DESIGN DECISION - HEIGHT ABOVE GROUND, NOT DEPTH
We predict the normalised DSM (nDSM): height above ground in metres. It is well defined from above, it is exactly what LiDAR measures once terrain is removed, and it separates what a model can learn from an image (buildings, trees) from what it cannot (absolute terrain elevation). The final product is DSM = DEM + nDSM. The DEM (SRTM 30 m or ISRO CartoDEM) supplies terrain, and the model supplies objects. Focal-length-based metric depth models (Depth Pro, UniDepth, Metric3D) are rejected because their core assumption - perspective geometry - does not hold for ortho imagery.

3. ARCHITECTURE
ingest -> GSD normalisation -> tiled inference (+TTA) -> scale calibration -> DSM composition -> validation -> 3D mesh + viewer
- Ingest: GeoTIFF (CRS and transform preserved, GSD read from the geotransform) or PNG/JPG (relative DSM path).
- GSD normalisation: resample the input so one pixel matches the ground distance the model was trained at, then resample the output back to the original grid.
- Tiled inference: overlapping tiles (25%) blended with Gaussian feathering. We verified numerically that the blend introduces zero seam artefacts.
- Test-time augmentation: horizontal and vertical flips, averaged.
- Output: DSM in a standard geospatial format (GeoTIFF, float32, CRS and grid of the input image, metres above the EGM2008 geoid), plus nDSM, the resampled DEM and a provenance/sanity report. Built and run end to end on a real Indian scene: Chungthang, North Sikkim (WorldView-2, 0.305 m) + Copernicus GLO-30 terrain. Chungthang run (RS3DAda, RTX 5060 laptop GPU, 18 s end to end for 625 x 625 m): predicted height above ground median 1.32 m, 95th percentile 6.42 m, 99.9th percentile 12.7 m - plausible for a small hill town of low-rise buildings and trees; terrain 1549-1770 m above the EGM2008 geoid; the 30 m DEM already contains ~2.4 m of smoothed buildings/trees on average (the double-count we report). No LiDAR exists there, so these are plausibility checks, not an RMSE.

4. MODELS
(a) Baseline - RS3DAda (SynRS3D, MIT licence): ViT-L encoder + DPT head that outputs metres, trained on 69,667 synthetic remote-sensing images at 0.05-1 m GSD. Used zero-shot. We read its inference code rather than guessing the interface, and caught a silent trap: it expects raw 0-255 pixel values, and passing [0,1] input produces wrong heights with no error.
(b) Main model - frozen DINOv3 ViT-L/16 pre-trained on SAT-493M (493 million satellite images) plus a DPT-style head we train. Four intermediate encoder layers are linearly projected, reassembled at four scales, progressively upsampled and fused into a one-channel height map (12.7M trainable parameters). Because the encoder is frozen, its features are computed once and cached, so training fits a free T4 GPU. The satellite model needs its own normalisation statistics (not ImageNet's), taken from the official DINOv3 release and checked at run time.

5. SCALE CALIBRATION (the PS's "relative -> absolute" milestone)
- Ground Control Points / DEM samples: robust affine h = s * prediction + t, fitted with RANSAC then Huber-weighted least squares, with s bounded to [0.6, 1.6]. In testing, an unbounded fit on bad points produced s = -1922, which the bound prevents.
- Ground-shift field: per 64x64-pixel block, estimate the local ground level from the lowest predicted values, smooth it, and subtract it. This removes per-tile bias and slow drift (83-90% of a synthetic tilt removed in tests).
- Class-wise offsets: the baseline under-predicts (measured bias -2.27 m overall, -7.07 m on forest). We fit one offset per class of the model's own segmentation output on a VALIDATION split and apply it unchanged to test data, so no ground-truth labels are used at test time.

6. VALIDATION METHOD
Metrics: RMSE, MAE, mean bias, Pearson correlation, and the share of pixels within 1 m / 2.5 m / 5 m, overall and per landscape. Buckets are derived automatically from land-cover labels (urban, sparse, forest). "Hilly" is a DEM question - nDSM has terrain removed by construction - so it is evaluated on the DSM once the DEM is added, not on the model output.
Controls: a predict-zero floor on the same tiles (how much does the model add over doing nothing?), a self-test of the harness, fixed held-out tiles shared by every run, and model selection on the validation split only.

7. MEASURED RESULTS (GAMUS test split, 40 tiles, 0.3 m, LiDAR nDSM)
RS3DAda zero-shot, GSD normalisation + tiling + flip TTA:
  overall  RMSE 6.74 m  MAE 3.47 m  bias -2.27 m  r 0.597
  urban    RMSE 5.11 m  MAE 2.68 m                r 0.757
  sparse   RMSE 2.83 m  MAE 1.43 m                r 0.684
  forest   RMSE 11.87 m MAE 7.87 m                r 0.306
Ablation: raw inference 6.82 m -> + GSD normalisation 6.82 m (no gain; urban slightly worse) -> + tiling/TTA 6.74 m.
DINOv3-SAT head (ours), frozen encoder + trained DPT head, same 40 tiles:
  overall  RMSE 4.96 m  MAE 2.44 m  bias -0.52 m  r 0.780
  urban    RMSE 5.28 m  MAE 2.40 m                r 0.726
  sparse   RMSE 2.15 m  MAE 1.12 m                r 0.829
  forest   RMSE 5.91 m  MAE 3.87 m                r 0.807
Predict-zero floor on the same tiles: overall 9.35 m, urban 8.81 m, sparse 4.21 m, forest 13.66 m.
Model selected on the validation split (best val RMSE 3.765 m, epoch 22 of 30), never on test.
GSD robustness (test images degraded to each GSD, scored against ground truth resampled to the same grid):
  input GSD      0.30   0.50   0.75   1.00 m
  DINOv3 head    4.85   4.60   4.68   5.46 m RMSE
  RS3DAda        6.82   6.70   6.83   6.92 m RMSE
  The head stays ahead at every resolution but degrades more at 1.0 m (it trains at a fixed 0.5 m). Coarser ground truth is also smoother, so part of each change is the target, not only the model.

8. WHAT THE NUMBERS TELL US
- Urban and sparse areas work zero-shot (correlation 0.76 and 0.68). Versus the predict-zero floor on the same 40 tiles: 42% lower RMSE on urban (5.11 vs 8.81 m) and 33% on sparse (2.83 vs 4.21 m).
- Forest is the failure mode: correlation only 0.31, and canopy is under-predicted by 7.07 m on average. RS3DAda forest RMSE was 11.87 m against a predict-zero floor of 13.66 m - barely better than guessing zero. This is why we trained a satellite-pretrained encoder on real LiDAR heights.
- It worked: the DINOv3-SAT head halves forest RMSE (11.87 -> 5.91 m), lifts forest correlation from 0.31 to 0.81, and removes the canopy bias (-7.07 -> -0.20 m). Overall RMSE drops 26% (6.74 -> 4.96 m).
- Honest limit: urban did not improve (5.28 m vs 5.11 m; r 0.73 vs 0.76). RS3DAda remains slightly better on buildings, so urban is our next target (e.g. unfreezing the last encoder blocks, or using RS3DAda in urban areas).
- GSD normalisation - often assumed essential - gave no gain on a backbone trained across many resolutions. We report that rather than drop it. It does matter for our DINOv3 head, which trains at one fixed 0.5 m GSD.

9. VISUALISATION LAYER (built and running)
Three.js, runs in any modern browser:
- The optical image is ortho-projected (identity UV) onto a 1.05M-vertex displaced terrain mesh with shadows and an adjustable sun.
- First-person fly (W/A/S/D, Q/E altitude, mouse look, terrain collision) and orbit mode, with a drag-to-look fallback where pointer lock is blocked.
- Analysis tools: click-probe for height, slope and error; two-click distance and height-difference measurement; height, slope and signed-error overlays; a vertical-exaggeration slider; a scale bar and compass.
- Validation panel: live RMSE, MAE and bias against the reference, and a signed error map (red too high, blue too low).
- Building walls: a single-view height grid would otherwise smear the roof texture down vertical walls. We detect near-vertical faces and render them as facades in a shader.
- Honesty built in: every scene is badged "GROUND TRUTH" or "MODEL PREDICTION".

10. DATA AND LICENCES
GAMUS (CC-BY-4.0; DC, New York, Philadelphia; aerial RGB + LiDAR nDSM + land cover). RS3DAda / SynRS3D (MIT). DINOv3 (Meta DINOv3 licence, gated). Three.js (MIT). Terrain: Copernicus DEM GLO-30 (© DLR 2010-2014 / Airbus 2014-2018, provided under COPERNICUS by the EU and ESA); CartoDEM supported via --dem. Indian demo image: Maxar Open Data Program, WorldView-2 2022-03-07, CC BY-NC 4.0.

11. RISKS AND MITIGATIONS
- Domain gap to ISRO imagery -> satellite-pretrained encoder, GSD normalisation, a GSD robustness sweep, and DEM/GCP calibration at deployment.
- Off-nadir Cartosat views -> documented. We evaluate nadir-most tiles and treat building lean as future work.
- Hilly terrain -> bounded by DEM resolution, and reported separately from model error.
- Tall towers -> the training clip is 150 m, because the data holds 146 m towers and an 80 m clip would erase them.

12. ROADMAP TO THE FINALE
Accuracy evaluation on Indian imagery against surveyed heights (the Chungthang DSM is built, but India has no open LiDAR truth, so it has sanity checks only); removing the buildings/trees that the 30 m surface DEM already contains (CartoDEM or a ground-filtered DEM); a canopy-specific correction; one-click standalone packaging of the viewer; recorded flythrough demos for all four landscapes.
```

---

## 5. The PDF (use the official template from the form's "Download Template" link)

Match each block below to the matching heading in the template. Screenshots are
in the repo's `shots/` folder. Keep the file **under 10 MB** (export images as JPG).

### Slide 1 — Title
- PS ID **26175** · *DepthWizard - Single-View Height Estimation and 3D Flythrough*
- Theme **Disaster Management** · Category **Software** · team name and ID
- One line: *"Height, not depth: metric elevation from one satellite image."*

### Slide 2 — Proposed solution / idea
- The core insight, in one diagram: **depth models assume perspective; satellite views have none → we predict height above ground (nDSM) → DSM = DEM + nDSM**
- Three boxes: *Elevation extraction · Scale calibration · 3D flythrough* (the PS's own three milestones)
- What is unique: measured per-landscape accuracy, validation built into the viewer, and honest limits
- Image: `shots/02_height_colormap.jpg`

### Slide 3 — Technical approach
- Flow: `image → GSD normalise → tiled ViT inference + TTA → calibration (DEM/GCP, class offsets) → DSM GeoTIFF → Three.js mesh`
- Models: RS3DAda (baseline) · DINOv3-SAT ViT-L frozen + trained DPT head (ours)
- Stack: PyTorch, HuggingFace, rasterio/h5py, NumPy/SciPy, Three.js
- **Results table** — copy it from `progression.md`, plus the DINOv3 row from `RESULTS.md`:

| model (40 held-out GAMUS tiles) | RMSE | MAE | r | forest RMSE |
|---|---|---|---|---|
| predict zero (floor) | 9.35 m | 5.10 m | — | 13.66 m |
| RS3DAda zero-shot | 6.74 m | 3.47 m | 0.60 | 11.87 m |
| DINOv3-SAT head (ours) | **4.96 m** | **2.44 m** | **0.78** | **5.91 m** |

Footnote on the slide: urban RMSE 5.28 m (ours) vs 5.11 m (RS3DAda) — the one bucket we did not improve.

- Images: `shots/08_walls_gradient.jpg` (textured 3D) and the **error-overlay screenshot** from the real predictions (see "Viewer screenshots with real predictions" below)

### Slide 4 — Feasibility and viability
- Runs on a free T4 GPU: the encoder is frozen and its features cached; the head is 12.7M parameters
- Everything open-source, with named weights and licences
- Risks → mitigations: domain gap, off-nadir views, hilly terrain, forest canopy (from section 11 above)
- Rigour: harness self-test (0.13 m floor), dataset traps caught, negative results reported

### Slide 5 — Impact and benefits
- Disaster management: flood-depth estimation from the DSM, landslide slope maps, rapid damage assessment. It needs one image, not a stereo pair, LiDAR or InSAR.
- Cost and speed: it works on imagery ISRO already acquires (Cartosat) plus free DEMs (CartoDEM)
- The navigable 3D view lets responders and planners check heights and slopes without GIS expertise

### Slide 6 — Research and references
- SynRS3D / RS3DAda - Song et al., arXiv 2406.18151
- DINOv3 (Meta), SAT-493M satellite weights - github.com/facebookresearch/dinov3
- Depth Anything V2 - Yang et al., arXiv 2406.09414 (NeurIPS 2024)
- GAMUS - Xiong et al., arXiv 2305.14914
- Depth Any Canopy - Rege Cambrin et al., arXiv 2408.04523
- DFC2019 US3D (source imagery of GAMUS) · Three.js

(arXiv IDs, titles and first authors checked against arxiv.org on 2026-09-26.)

---

## Viewer screenshots with real predictions

Once the Colab run finishes, it writes `viewer_assets/` to Drive. Then, on any laptop:

```bash
git clone https://github.com/AHSharan/SIH2026PS175
```

1. Copy all files from the downloaded `viewer_assets` folder into `SIH2026PS175/web/assets_pred/`.
2. In the `SIH2026PS175` folder, run `python serve.py`.
3. Open **http://localhost:8777/?assets=assets_pred**. The badge now reads **MODEL PREDICTION**.
4. Click **Error** for the signed error map; the right panel shows live RMSE / MAE / bias. Screenshot with `Win + Shift + S`.

Without `?assets=...` the viewer shows the ground-truth demo scenes, badged GROUND TRUTH.

---

## Rules before you submit

1. **If the DINOv3 head does NOT beat RS3DAda** on the same tiles, say so. Keep the
   RS3DAda numbers as the result and describe the head as "trained; did not yet beat
   the baseline on held-out data". Judges trust honest numbers; a caught
   exaggeration costs far more than a modest result.
2. The GAMUS numbers are **US aerial imagery**. Never present them as Cartosat accuracy.
3. Every `[[...]]` is filled from `RESULTS.md` or deleted.
4. PDF under 10 MB. Title under 100 characters (the one above is checked).

## Likely judge questions

- **"Why not Depth Anything / Depth Pro?"** They estimate perspective depth from
  ground-level cues. A nadir image has no perspective, so we predict height above
  ground directly and use the DEM for terrain.
- **"How do you get absolute metres?"** The model outputs metres, trained on LiDAR.
  Calibration against DEM samples or GCPs (robust affine with bounded scale) and
  class-wise offsets correct the residual bias.
- **"What about hilly terrain?"** Terrain comes from the DEM, so hilly-area DSM error is
  bounded by the DEM's 30 m resolution. We report it separately from the model's error.
- **"How do we know the numbers are real?"** A fixed held-out test set, model
  selection on validation only, a self-tested harness with a known 0.13 m floor, a
  predict-zero control, and code in a public repo.
