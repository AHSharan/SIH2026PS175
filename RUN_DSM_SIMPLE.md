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

## Part D (optional) — See it in 3D

```
python serve.py
```
Open this link in Chrome: **http://localhost:8777/?assets=assets_dsm**

- Drag with the mouse to look around, and click the ground to see its height
  above sea level.
- Set *Vertical exaggeration* to 1.0× to see the true shape of the valley.
- When you're done, go back to PowerShell and press **Ctrl + C**.

---

## If anything goes wrong

Take a screenshot of the PowerShell window showing the **last lines of text**.
That screenshot is what's needed to fix it. Don't worry, nothing on your
computer can break.
