# How to make the 3D height map of Chungthang (Sikkim)

You'll copy and paste a few commands. The first time takes about 20–30
minutes, mostly waiting for downloads. You need a **Windows PC with an NVIDIA
graphics card** and an internet connection.

---

## Part A — One-time setup

Skip any step you've already done.

**1. Install Python**
- Go to https://www.python.org/downloads/ and download **Python 3.11**.
- When you run the installer, **tick "Add Python to PATH"** at the bottom of
  the first screen, then click *Install Now*.

**2. Install Git**
- Go to https://git-scm.com/download/win, download it and install it.
  Just click *Next* on every screen.

**3. Open PowerShell**
- Press the **Windows key**, type **PowerShell** and open it.
- A blue or black window appears. Paste each command below into it and press
  **Enter** after each one. (To paste in PowerShell, right-click.)

**4. Download the project**
```
cd $HOME\Desktop
```
```
git clone https://github.com/AHSharan/SIH2026PS175
```
```
cd SIH2026PS175
```
A folder called `SIH2026PS175` now appears on your Desktop.

**5. Install the software it needs**
```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
```
pip install -r requirements.txt
```
Each can take 5–10 minutes. Lots of text scrolling past is normal.

**6. Check that the graphics card works**
```
python -c "import torch; print(torch.cuda.is_available())"
```
- **True** → go to Part B.
- **False** → update your NVIDIA driver (https://www.nvidia.com/Download/index.aspx),
  restart the PC, and try this step again.

---

## Part B — Make the height map

If you closed PowerShell, open it again and run:
```
cd $HOME\Desktop\SIH2026PS175
```
Then:
```
git pull
```
```
python dsm.py
```
- The first time, it downloads the AI model (about 1.5 GB), so give it a while.
- It's finished when you see these lines at the end:
  ```
  [saved] plain-English numbers: ...\out_dsm\chungthang\SUMMARY.txt
  [saved] everything in ONE file: ...\out_dsm\chungthang_results.zip
  ```
- If you see **`CUDA out of memory`**, run this instead:
  ```
  python dsm.py --tile 518
  ```

---

## Part C — Your results are saved

Open the `SIH2026PS175` folder on your Desktop, then the **`out_dsm`** folder.

| file | what it is |
|---|---|
| **`chungthang_results.zip`** | **everything in one file.** This is the one to share. |
| `chungthang\SUMMARY.txt` | the key numbers in plain words. It opens in Notepad. |
| `chungthang\dsm.tif` | the height map itself (opens in QGIS / ArcGIS) |

To share it, send `chungthang_results.zip` by WhatsApp, email or Google Drive.
(If email says it's too big, Gmail offers Google Drive automatically.)

Running `python dsm.py` again replaces these files with fresh ones. To keep an
old result, copy the zip somewhere else first.

---

## Part C2 — Run OUR model (use this for the demo)

Our own model is more accurate than the default one (4.69 m vs 6.74 m error on
the US test tiles), especially on trees. It needs two extra things, once:

**1. The model file.** Get `best.pt` from the 1,500-tile training run
(Google Drive: `depthwizard_1500/ckpt/best.pt`).
- Download it, then **don't double-click or unzip it**.
- Rename it to **`best_1500.pt`**.
- Put it in the `SIH2026PS175\dw_run\ckpt\` folder (create the folders if needed).
- Check: it must be **one file** (a few MB or more), not a yellow folder.

**2. A HuggingFace token** (Read type) from the account that accepted the
DINOv3 licence. In Git Bash, in the `SIH2026PS175` folder:
```
export HF_TOKEN=hf_YOUR_TOKEN
```
Never share the token or show it in a screenshot.

**Run it** (same window):
```
python dsm.py --model ours
```
It's done when you see `[saved] everything in ONE file: ...chungthang_ours_results.zip`.
Your results go to `out_dsm\chungthang_ours\`. They don't overwrite the other runs.

---

## Part D (optional) — See it in 3D

```
python serve.py
```
Open this link in Chrome: **http://localhost:8777/?assets=assets_dsm**

- Drag with the mouse to look around, and click the ground to see its height
  above sea level.
- Pick the scene in the *Scene* list: `chungthang_ours` is our model,
  `chungthang` the default one. The newest run opens first.
- Scenes open at true scale (1.0×). Raise *Vertical exaggeration* only if you want
  hills to look steeper.
- *Above ground* in the probe panel is the model's height of the building or
  tree you clicked.

**Optional tools** (bottom right, all off until you click them):
- **Flood**: drag *Water level* to raise water through the valley. It shows the
  flooded area and how many buildings the water reaches. This is a simple
  "bathtub" screening, not a river-flow simulation.
- **Buildings**: pins on candidate buildings, coloured by estimated floors
  (green 1 → red 5+). With Flood on, pins turn **red** when the water reaches
  them. Switch to *Orbit* and click a roof to see its height and floors.
- **Tour → Play tour**: a 24-second automatic flight. **Record video** does the
  same flight and saves a video file to your Downloads folder (use Chrome, and
  keep the tab visible while it records). Press **Esc** to stop early.
- When you're done, go back to PowerShell and press **Ctrl + C**.

---

## Part E — Live mode: upload any image and run a model from the browser

Use this on the **GPU laptop** for the demo.

**1. Put your model files** (`.pt`) in `SIH2026PS175\dw_run\ckpt\` or in a new
folder `SIH2026PS175\models\`. Any number of them. Don't unzip them.

**2. Start the viewer in live mode** (Git Bash, in `SIH2026PS175`):
```
export HF_TOKEN=hf_YOUR_TOKEN
```
```
python serve.py --live
```
It prints which GPU it found and every model file it sees.

**3. Open** http://localhost:8777/?assets=assets_live in Chrome.

**4. In the "Run a model" panel** (top right):
- **Choose File**: a GeoTIFF (gives a real elevation map) or a PNG/JPG
  (then also type its pixel size in metres).
- **Model**: pick one of your `.pt` files, *RS3DAda*, or *Demo* (fake heights, for testing).
- Press **Run**. When it says *Done*, the result opens in 3D by itself, with
  Flood, Buildings and Tour available.

The first run with a DINOv3 model takes longer (it loads the model); after
that each run takes seconds. Results are also saved in `out_dsm\` as usual.

**Which `.pt` files work:** our DINOv3 height models (any `best.pt` or
`last.pt` from our training) and RS3DAda checkpoints. A `.pt` of a different
kind of model shows a clear "not a checkpoint type this app can build" message.

---

## If anything goes wrong

Take a screenshot of the PowerShell window showing the **last lines of text**.
That screenshot is what's needed to fix it. Don't worry, nothing on your
computer can break.
