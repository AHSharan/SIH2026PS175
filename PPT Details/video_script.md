# DepthWizard: demo video script (pre-recorded, about 4 minutes)

The video opens straight on the product, with no title slides. Each part shows a
different input and ends on proof, ideally a check against real LiDAR. Every number
spoken below is measured (sources in `PPT Details/README.md`).

---

## Why this order

| # | part | input | proves (PS wording) | needed? |
|---|---|---|---|---|
| 0 | cold open | Chungthang 3D tour | "interactive 3D assets that can be navigated in real time" | hook, 10 s |
| 1 | GeoTIFF + LiDAR check | Park City GeoTIFF (US aerial) | "Georeferenced → Absolute DSM with metric heights"; "validate estimated height values against reference datasets"; "RMSE, MAE, correlation against LiDAR"; "hilly" | **yes, core** |
| 2 | PNG | Chungthang as plain PNG | "Non-georeferenced (PNG or JPG) → rDSM"; "upload imagery" | **yes, core** |
| 3 | LiDAR truth vs real image | GAMUS test tiles | accuracy across "urban, sparse … forested"; the dataset the PS recommends | **yes, 50 % of marks are accuracy** |
| 4 | fly and analyse | Chungthang GeoTIFF (India, hills) | "first-person navigation and analysis of structural heights and slopes"; "project the optical image onto a 3D terrain mesh" | **yes, core** |
| 5 | tools | same scene | disaster theme (flood), building heights | optional: cut first if too long |
| 6 | export + result | files + chart | "DSM in a standard geospatial format"; "standalone" | **yes** |

Each part is recorded as a separate clip, then joined (Clipchamp comes with Windows).

---

## Before recording (once)

1. Stop any old server (Ctrl + C in its window), then `git pull`.
2. Double-click **`start_demo.bat`**. The black window must say
   **"DINOv3 on disk (offline OK)"**. The browser opens by itself.
3. Browser full screen (**F11**), zoom 100 %, window 1920 × 1080. Close other tabs.
4. Warm-up (so nothing waits on loading in the video): in **Run a model**, run
   `demo/4_geotiff_india/chungthang_wv2.tif` once with model **best**.
   Every file used below is in the **`demo/`** folder (see `demo/README.md`).
5. Second terminal for part 3: `python serve.py --demo-lidar --port 8778` (opens its own tab).
6. Recorder: **Win + Alt + R** starts and stops (Game Bar, saves MP4), or OBS.
7. Record the voice-over separately afterwards and lay it on top: cleaner than
   talking while clicking. Move the mouse slowly; drag the orbit gently.

---

## Part 0: cold open (0:00–0:10)

**Screen:** Chungthang scene → **Tools → Tour → Record video** (a clean 24 s fly-over
of the 3D view, saved to Downloads). Use its first 10 seconds.

**Say:** "This is a town in the Himalayas, rebuilt in 3D from a single satellite
photo. Every point you see has a real height in metres."

## Part 1: GeoTIFF, checked against real LiDAR (0:10–1:10)

**Screen:**
1. **Run a model** → choose `demo/1_geotiff_with_lidar/park_city_naip2021.tif` → model **best** → **Run**.
   Show the progress bar, then the pipeline bar at the top lighting up:
   Image › Model › Terrain › DSM.
2. The 3D scene opens. Drag to orbit once. **Surface → Height.**
3. Point the mouse at **Scene info**: map projection, pixel size 0.6 m, terrain
   Copernicus GLO-30, elevation about 2,125 to 2,316 m.
4. **Check against a reference** → choose `demo/1_geotiff_with_lidar/park_city_lidar_reference.tif` →
   **Score this run**. The **Accuracy** panel appears; the **Validated** chip turns green.
5. **Compare side by side** (in the Accuracy panel) → **Swipe** on → move the mouse
   slowly: photo on the left, LiDAR reference on the right. Click **Our model**, then
   **Error**. Hover a roof: the line underneath shows reference, model and error.
6. **Back to 3D** → **Surface → Error.** Orbit slowly over the red/blue map.

**Say:** "We start with a georeferenced GeoTIFF: an aerial photo of a hillside in
Utah. Our model predicts the height of every building and tree from the photo
alone, and adds the ground from a 30-metre terrain map, giving an absolute
surface model in metres above sea level. Now we check it against airborne LiDAR
for the same place. The error is 3.3 metres. Using the terrain map alone would
give 4.9, so the model clearly adds the buildings and trees. The adjusted rows
are fitted on one half of the area and tested on the other half, so they cannot
flatter themselves. Side by side, you can see the photo, the laser truth and our
prediction on the same pixels. Blue means we are too low, red too high."

## Part 2: PNG, JPG and plain TIFF (1:10–1:50)

**Screen:** (files in `demo/2_png_jpg_tiff/`, the same picture three ways)
1. **Run a model** → `chungthang_plain.png` → tick **"I don't know the pixel size"**
   → **Run** → badge says **RELATIVE SHAPE**. Orbit once.
2. **Run a model** → `chungthang_plain.jpg` → pixel size **0.3** → **Run** →
   heights in metres (Scene info: output "Heights above ground").
3. *(Optional, 5 s)* the same with `chungthang_plain.tif`: a TIFF without map
   information is handled like PNG/JPG.

**Say:** "Not every image has map information. From a plain PNG we still get the
shape of the terrain, a relative surface model. If we know the pixel size, the
heights are in real metres, and it works the same for JPG and for TIFFs without
coordinates. With three map points the image is placed on the map and gets a
full elevation model."

*(Optional, only if time allows: show the "Map points" box.)*
*(Optional extra: `demo/3_gamus_png_with_lidar/`: upload `urban_DC_27_47_image.png`
with pixel size 0.3, then check it against `urban_DC_27_47_lidar_heights.tif`: a PNG
scored against LiDAR, RMSE 3.34 m.)*

## Part 3: real image vs LiDAR truth (1:45–2:25)

**Screen:** switch to the `--demo-lidar` tab.
1. Starts on **Real image**. Press **T** a few times: photo ↔ **LiDAR truth**.
2. Click **Our model**. Then **Swipe** and move the mouse slowly across the picture.
3. Hover over a house: the line under the picture shows LiDAR, model and error.
4. Change the tile in the drop-down: urban → sparse → forest.

**Say:** "This is the GAMUS dataset the problem statement recommends: real aerial
photos with laser-measured heights. On the left, the photo; on the right, the
LiDAR truth. Our model never sees the LiDAR. It predicts these heights from the
photo alone. These are typical tiles, the median error of each landscape, not
picked examples. Over 40 held-out test tiles the average error is 4.96 metres,
26 percent lower than RS3DAda, a published remote-sensing height model, on the
same tiles."

## Part 4: fly through and analyse (2:25–3:10)

**Screen:** back to the main tab, scene `chungthang_wv2_best`.
1. **Photo**, then **Orbit** gently around the town.
2. **Fly** → click the view → W A S D along the river for 5 seconds → **Esc**.
3. **Orbit** → click a roof: the **Probe** shows elevation, height above ground, slope.
4. **Surface → Slope**, then **Contours** on.
5. **Measure distance** → click two points across the town → the height profile
   appears → click **Measure** again to close it.

**Say:** "This is Chungthang in Sikkim, from a WorldView-2 satellite image. You can
fly through it in first person, click anywhere to read its elevation, height above
ground and slope, colour the terrain by slope, draw contour lines, and cut a
height profile across the town. There is no LiDAR for India we could use, so here
we show the product itself; this is where ISRO imagery plugs in."

## Part 5: tools (3:10–3:35), optional

**Screen:** **Tools → Buildings** (pins coloured by floors), then **Flood**, drag the
level to about +36 m.

**Say:** "From the heights we can pick out candidate buildings and estimate their
floors, and run a simple flood check: at a 36-metre rise, 12 of the 204 candidate
buildings are reached. These are estimates for screening, not survey data."

## Part 6: export and result (3:35–4:00)

**Screen:**
1. **Export** panel: point at **DSM GeoTIFF**, then click **3D model (.glb)**.
2. Cut to `PPT Details/charts/performance_comparison.png` (full screen).

**Say:** "Everything exports as standard GeoTIFFs that open in QGIS or ArcGIS, plus a
textured 3D model. The whole app runs offline on a laptop. Measured against
airborne LiDAR: 4.96 metres on 40 held-out test tiles, and 3.8 metres on hilly
terrain, compared with 5.9 for the terrain map alone. DepthWizard."

---

## Numbers used (all measured)

| said | value | source |
|---|---|---|
| Park City vs LiDAR | 3.26 m raw; terrain alone 4.94 m | live check, `validate.py` |
| GAMUS 40 tiles | RMSE 4.96 m, r 0.78; RS3DAda 6.74 m → 26 % lower | RESULTS.md (reproduced 29 Sep) |
| hilly, full pipeline | 3.84 m; DEM alone 5.88 m | results/lidar_benchmark.md |
| Chungthang flood | 12 of 204 candidate buildings at +36 m (9 ha) | checked in the viewer, 29 Sep |

If a number on screen differs from the script on the day (e.g. a re-run gives
3.27), say the number on screen.
