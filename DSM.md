# Make the Indian DSM (one command, on a GPU PC)

This turns one real Indian satellite image into an **absolute DSM GeoTIFF**
(metres above sea level, opens in QGIS/ArcGIS) plus a 3D viewer scene.

**Scene:** Chungthang town, North Sikkim, 625 m × 625 m. It sits at the
Lachen/Lachung confluence and is the town the Oct 2023 Teesta flood hit.
The image is WorldView-2 from 7 Mar 2022, before the flood.

Everything the command needs is already in the repo (`samples/`), except the
model weights, which download automatically (1.5 GB, once).

## Before you start

Do `RUN_LOCAL.md` steps 1–3 first (clone, GPU PyTorch, `pip install -r
requirements.txt`). You don't need the HuggingFace login for this.

## Run it

In the `SIH2026PS175` folder:

```bash
git pull
```

```bash
python dsm.py
```

After the first download it's quick: 18 s end to end on an RTX 5060 laptop GPU
(measured). The last lines look like this:

```
[done] DSM -> out_dsm/chungthang  (… s)
       nDSM p50/p95/p99.9 = …/…/… m | DEM [1549.4, 1769.7] m | 30 m double-count ~… m
```

If it says `CUDA out of memory`, run `python dsm.py --tile 518` instead.

## Our model (for the demo)

```bash
python dsm.py --model ours
```

This runs our DINOv3-SAT head trained on 1,500 GAMUS tiles. That run was
reported at RMSE 4.69 m on the 40 held-out test tiles, but its results file is not
in this repo, so the verified figure is the 480-tile model's **4.96 m** (RESULTS.md;
RS3DAda 6.74 m on the same tiles). It needs:

- `dw_run/ckpt/best_1500.pt`: the `best.pt` from the 1,500-tile run, as ONE
  file. A `.pt` is a zip inside, so never open or unzip it. `dsm.py` says
  so explicitly if it finds a folder there.
- A DINOv3 HuggingFace login: `export HF_TOKEN=hf_...` in the same terminal.

Output goes to `out_dsm/chungthang_ours/` and the viewer scene `chungthang_ours`,
so it never overwrites the RS3DAda run. Plain `python dsm.py` still runs
RS3DAda, which needs no login.

## Optional viewer tools (for the demo)

All three are off until clicked, in the *Tools (optional)* panel:

| tool | what it does | honest limit |
|---|---|---|
| **Flood** | water rises from the lowest ground (the river) as you drag the level; shows flooded area and buildings reached | "bathtub" screening: everything below the level is wet, with no river flow and no connectivity; heights carry the model's few-metre error |
| **Buildings** | pins on candidate buildings with estimated floors; click a roof for details | heuristic: model height ≥ 2.5 m, trees removed by colour + texture, floors = height ÷ 3 m; neighbouring roofs can merge or split |
| **Tour** | 24 s scripted fly-through; *Record video* saves MP4 (or WebM) of the 3D view | needs Chrome with a GPU; checked here only with a slow software renderer |

The buildings come from `dsm.py` (`<scene>_bld.json`, `<scene>_bldid.bin`, and a
count in `SUMMARY.txt`). Skip them with `--no-buildings`.

## Live mode: upload an image in the browser

`python serve.py --live`, then open http://localhost:8777/?assets=assets_live.
The *Run a model* panel uploads a GeoTIFF/PNG/JPG, runs a chosen model on this
machine (models are loaded once and kept in memory) and opens the result in 3D.
Models: every `.pt` in `dw_run/ckpt/` and `models/` (extra folders with
`--models-dir`) that is one of our DINOv3 height heads or an RS3DAda checkpoint,
plus the public RS3DAda weights and a Demo model. Images larger than 4096 px are
centre-cropped to 4096 px (the log says so). Details in `live.py`; tests in
`tests/test_live_cpu.py` (20/20).

## Where the results are saved

- `out_dsm/chungthang_results.zip`: **everything in one file** (the GeoTIFFs,
  `report.json`, `SUMMARY.txt` and the viewer scene). This is the one to share.
- `out_dsm/chungthang/SUMMARY.txt`: the key numbers in plain words.

A step-by-step version for non-technical users is in
[`RUN_DSM_SIMPLE.md`](RUN_DSM_SIMPLE.md).

## Look at it in 3D

```bash
python serve.py
```

Then open http://localhost:8777/?assets=assets_dsm. The badge reads
**DSM = DEM + MODEL nDSM**. Clicking the terrain shows real elevation in metres.
DSM scenes open at true scale (1.0×). The probe also shows *Above ground*,
the model's height of the clicked building or tree. Building walls are
detected from that height above ground, not from the terrain slope, so steep
hillsides keep their real image colours instead of being painted as walls.

---

## What the files are (for the write-up)

| file | what it is |
|---|---|
| `dsm.tif` | **DSM = DEM + nDSM**, metres above the EGM2008 geoid. float32, nodata −9999, LZW, UTM 45N (EPSG:32645), 0.305 m pixels, same grid as the image |
| `ndsm.tif` | model output: height above ground (m), RS3DAda, GSD-normalised to 0.5 m, with TTA |
| `dem.tif` | Copernicus GLO-30 terrain, bilinearly resampled to the image grid |
| `report.json` | where every input came from, plus the sanity checks below |
| `SUMMARY.txt` | the same key numbers in plain words |

**Other images.** Pass any GeoTIFF (a local path or an `https://` URL; add
`--window col,row,width,height` for a crop):
`python dsm.py my_image.tif`. The GSD comes from the file itself, degrees are
converted to metres, and the Copernicus DEM tiles for the footprint download
automatically. To use CartoDEM or any other DEM instead:
`--dem cartodem.tif`. PNG/JPG has no location, so there is no DEM to add: pass
`--gsd 0.5` and you get the nDSM plus a 0–1 relative DSM (`rdsm_0to1.png`).

## Be honest about this scene

- **There is no RMSE for India.** No LiDAR or survey truth exists for
  Chungthang, so there is no accuracy number. Every accuracy figure we quote
  comes from GAMUS (US LiDAR). `report.json` has plausibility checks only:
  nDSM percentiles (Chungthang is a small hill town, so building heights far
  above ~30 m would be suspect), DEM coverage and relief, and the double-count size.
- **Double-counting.** GLO-30 is a 30 m *surface* model (radar, 2011–2015), so
  in built-up and forested cells it already contains a smoothed share of the
  buildings and trees. DEM + nDSM counts part of that twice.
  `double_count_30m_mean_m` in `report.json` estimates how much (our nDSM
  averaged over 30 m cells). The caveat is also written into the GeoTIFF tags.
- **The DEM is older than the image,** and the valley floor has changed (the dam
  reservoir, and the 2023 flood came after both).
- **Sample vs original.** `samples/chungthang_wv2.tif` is a JPEG-compressed
  crop of the original Maxar file. On a test run the DEM was bit-identical to
  reading the full GLO-30 tile live, and the fake test model's nDSM differed by
  0.05 m on average (0.67 m at most) from running on the original. For the
  exact original pixels:
  `python dsm.py https://maxar-opendata.s3.amazonaws.com/events/India-Floods-Oct-2023/ard/45/120220122202/2022-03-07/10300100CE8D0400-visual.tif --window 7676,1276,2048,2048 --name chungthang`
- **Pixel size vs sharpness.** The file's pixels are 0.305 m, and that is what
  the model's scale normalisation uses. The WorldView-2 sensor's native
  resolution here is about 0.49 m, so the image is somewhat softer than a true
  0.3 m image.

## Credits (must appear on any slide showing this scene)

- Image: **Maxar Open Data Program**, WorldView-2, 2022-03-07, catalog
  10300100CE8D0400, **CC BY-NC 4.0** (non-commercial use only; a hackathon
  submission is fine).
- Terrain: **Copernicus DEM GLO-30** © DLR e.V. 2010–2014 and © Airbus Defence
  and Space GmbH 2014–2018, provided under COPERNICUS by the European Union
  and ESA.

## Tested here (no GPU)

`python tests/test_dsm_cpu.py` → **40/40 pass**. It checks CRS/transform
preservation, DSM = DEM + nDSM exactly, reprojection from a DEM in a different
CRS (exact to 0.01 m), degree→metre GSD, scene-fill masking, window crops,
the PNG path, GLO-30 tile naming, the summary file and the results zip,
the viewer's above-ground layer and 1.0× default, the `--model ours` preset and
its error messages, the building finder (known roofs found, dark-green canopy rejected), and the Chungthang sample end to end. The
whole pipeline also ran on the sample with a fake model, and the viewer
rendered the scene. What we could **not** run here is the real RS3DAda model on
this image: this cloud machine has no GPU and can't reach HuggingFace. That's
the one command above.
