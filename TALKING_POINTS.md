# DepthWizard: talking points and demo script

Every number below is **measured** on our own runs unless it says otherwise.

---

## The 30-second opening

> "From a single satellite photo, DepthWizard estimates how tall every building and
> tree is, in metres, and builds a 3D model you can fly through. Existing depth
> models measure distance from the camera, which doesn't work for images taken
> from directly above. We measure height above the ground instead. On 40 test
> areas with laser-measured ground truth, our model's error is **4.96 metres,
> 26% lower than RS3DAda (6.74 m)**, a published remote-sensing height model we
> ran on the same data. We also checked the whole pipeline against airborne
> LiDAR on hills: 3.8 m, against 5.9 m for the terrain map alone. It runs
> on Indian imagery, and one 625 × 625 m scene takes about 13 seconds on a
> laptop GPU."

## The one number

**4.96 m average error vs 6.74 m for RS3DAda: 26 % lower**
*(measured; RMSE against airborne LiDAR on the same 40 held-out GAMUS test tiles,
RESULTS.md, model trained on 480 GAMUS tiles)*

If you only say one thing, say that, and say "same test tiles".

Do **not** quote 4.69 m (the 1,500-tile run): its results file is not in the repo,
so we cannot show where it comes from. If that file is added, it can replace 4.96.

---

## Talking points, in order of strength

1. **Height, measured properly.**
   "Depth models like Depth Anything learn from ground-level photos, where things
   far away look smaller. A satellite looks straight down, so that cue doesn't
   exist. We predict height above ground directly, in metres, and add the terrain
   from a free elevation map to get the full surface model ISRO asks for."

2. **Forests, where others fail.**
   "RS3DAda, the published model we tested, under-predicted tree canopy by about
   7 metres. Our model **halved the forest error, from 11.87 m to 5.91 m**
   *(measured)*, because we trained a satellite-pretrained encoder on real laser
   heights."

3. **It works at Cartosat-like resolution.**
   "We degraded the test images from 0.3 m to 1 m pixels. Our model stayed between
   **4.60 and 5.46 m**, while the baseline stayed between 6.70 and 6.92 m
   *(measured)*: better at every resolution we tried."

4. **An Indian location, end to end.**
   "This is Chungthang in Sikkim: a WorldView-2 image in, a georeferenced
   elevation GeoTIFF out, 1,549 to 1,776 m above sea level, on a 220 m hillside.
   It opens directly in QGIS or ArcGIS."

5. **Any image, honestly labelled.**
   "Upload a GeoTIFF and you get absolute heights above sea level. Upload a plain
   PNG or JPG and you still get the 3D shape, clearly labelled as relative. Click
   three or more known map points and it becomes a full elevation map."

6. **Checks built into the viewer.**
   "Every scene is badged as model output or ground truth. You can switch to the
   laser truth, see an error map, and draw a line to compare heights along it."

7. **Disaster use.**
   "The flood tool raises the water level and counts which buildings get wet. On
   Chungthang, the lowest building sits 28.6 m above the river; at a 36 m rise,
   12 of 204 buildings are reached *(computed from our model's elevation map)*."

---

## Live demo script (about 3 minutes)

**Before the judges arrive, on the demo laptop:** stop any server that is already
running (Ctrl + C in its window): a server started before `git pull` keeps
serving the old code. Then:

```bash
git pull
```

Then double-click **`start_demo.bat`** in the project folder (or run
`python serve.py --live`). It opens **http://localhost:8777/?assets=assets_live** by
itself and needs no internet once DINOv3 has been downloaded (the window says
"DINOv3 on disk (offline OK)"). Load the Chungthang scene once, so the model is
already warm. Keep the finished screenshots in `shots/` open in another
window as a backup.

**On stage:**

1. **Open Chungthang (textured 3D).**
   Say: "This is a real satellite image of Chungthang, Sikkim, turned into 3D by our model."
   Drag to orbit slowly.

2. **Click "Height".**
   Say: "Colour is elevation above sea level: the town sits on a 220 m slope."

3. **Click "Measure distance", then two points across the town.**
   Say: "The profile shows the height along this line: each bump is a building."

4. **Tools → "Buildings".**
   Say: "It finds 204 buildings and estimates their floors from height."

5. **Tools → "Flood", drag the level to about +36 m.**
   Say: "At a 36 metre rise, 9 hectares flood and 12 buildings are reached."

6. **Check it against LiDAR, live.** Run panel → choose `samples/park_city_naip2021.tif`
   → Run. When it opens, "Check against a reference" → choose
   `samples/park_city_lidar_dsm_2m.tif` → "Score this run", then Surface → "Error".
   Say: "This is a US aerial photo of a hillside with 160 metres of relief. We score our
   DSM against airborne LiDAR right here: about 3.3 metres error, against 4.9 for the
   terrain map alone. The last two rows fit a datum offset and a scale on half the area
   and score on the other half, so they can't cheat."

7. **Close on the number.**
   Say: "On laser-measured test data: 4.96 metres error, 26 % lower than RS3DAda on the same tiles."

**If anything fails live:** switch to the screenshots and say "here is the same run
we recorded this morning". Don't debug in front of judges.

---

## Likely questions

- **"Why not Depth Anything or Depth Pro?"**
  "They estimate distance from the camera using perspective cues from ground-level
  photos. A top-down satellite view has no perspective, so we estimate height above
  ground directly."

- **"How do you get real metres?"**
  "The model is trained on laser-measured heights, so it outputs metres. Terrain comes
  from the Copernicus 30 m elevation map. For plain images, three or more map points
  fix the position and scale."

- **"Is 6.74 m RS3DAda's own published number?"**
  "No, we ran RS3DAda ourselves on the same 40 test tiles, so the comparison is fair.
  GAMUS isn't in their paper."

- **"How do you know the numbers are real?"**
  "Fixed held-out test tiles, the model chosen on a separate validation set, a
  do-nothing baseline on the same tiles (9.35 m), and all code is public on GitHub."

- **"Did you check it against LiDAR?"**
  "Yes, on 8 US sites with airborne LiDAR, running the full pipeline. Full DSM error:
  hilly 3.8 m, sparse 3.6 m, residential 5.4 m. Tall downtown towers and dense forest
  are under-predicted; those are in our limitations. Details in
  results/lidar_benchmark.md."

- **"Does it work on Cartosat?"**
  "We simulated Cartosat-like 0.3–1 m resolution and our model stayed ahead at every
  step. We'd welcome Cartosat data with ground truth to measure it directly."

---

## Limitations: say these last, and only if asked

- Accuracy is measured on US aerial data (GAMUS); Indian imagery has no laser truth
  available to us, so there we show plausibility checks, not an error number.
- In hilly areas the terrain part is only as sharp as the 30 m elevation map.
- Very tall towers and dense forest remain the hardest cases.
