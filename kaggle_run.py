"""
kaggle_run.py - run this on Kaggle (T4 GPU ON) to produce the real numbers.

It runs the SAME experiment three times so you get a progression table:
    A  raw            : no GSD normalisation, no TTA
    B  + GSD norm     : resample to the model's GSD first
    C  + GSD + tiling/TTA
and appends all three to progression.md.

HOW TO USE (5 minutes):
 1. New Kaggle notebook -> Settings -> Accelerator: GPU T4 x1, Internet: ON
 2. Paste this whole file into one cell, or upload it and `%run kaggle_run.py`
 3. Set REPO below to your GitHub repo (see note), or upload the .py files
    as a Kaggle Dataset and set CODE_DIR to that path.
 4. Run. Takes ~10-20 min for 40 tiles.
 5. Send me the printed tables + any traceback.

NOTE ON GETTING THE CODE THERE: Kaggle can't see your laptop. Either
  (a) push data.py / eval.py / predict.py to a GitHub repo and git clone it
      (easiest, and you wanted a repo anyway), or
  (b) Kaggle -> Datasets -> New Dataset -> upload the 3 .py files, then
      CODE_DIR = '/kaggle/input/<your-dataset-name>'
"""
import os, sys, subprocess, shutil

# ----------------------------------------------------------------- config
REPO      = ""            # e.g. "https://github.com/<you>/depthwizard"
CODE_DIR  = ""            # e.g. "/kaggle/input/depthwizard-code"
N_TILES   = 40            # per-run tile count; 40 is enough for a stable RMSE
SPLIT     = "test"
INPUT_GSD = 0.3           # GAMUS; asserted from DFC2019 spec, see data.py
MODEL_GSD = 0.5           # RS3DAda trained 0.05-1.0 m; 0.5 = midpoint
WORK      = "/kaggle/working"


def sh(cmd, check=True):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout: print(r.stdout[-4000:])
    if r.returncode and check:
        print(r.stderr[-4000:])
        raise SystemExit(f"command failed: {cmd}")
    return r


# ------------------------------------------------------------- 1. our code
os.chdir(WORK)
if REPO:
    if not os.path.isdir("code"):
        sh(f"git clone --depth 1 {REPO} code")
    sys.path.insert(0, os.path.join(WORK, "code"))
elif CODE_DIR:
    for f in ("data.py", "eval.py", "predict.py"):
        shutil.copy(os.path.join(CODE_DIR, f), WORK)
    sys.path.insert(0, WORK)
else:
    raise SystemExit("Set REPO or CODE_DIR first - see the docstring.")

# --------------------------------------------------------- 2. SynRS3D repo
if not os.path.isdir("SynRS3D"):
    sh("git clone --depth 1 https://github.com/JTRNEO/SynRS3D")

# SynRS3D's DPT_DINOv2 builds a DINOv2 backbone. With pretrained=False it
# should not hit torch.hub, but if it does, this makes the failure obvious
# instead of silently downloading a DIFFERENT set of weights.
os.environ.setdefault("TORCH_HOME", os.path.join(WORK, "torchhub"))

sh("pip -q install h5py rasterio 2>&1 | tail -2", check=False)

import numpy as np
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if not torch.cuda.is_available():
    print("!! GPU IS OFF. Settings -> Accelerator -> GPU T4 x1. Stopping.")
    raise SystemExit(1)

import data as D
import predict as P
import eval as E

# --------------------------------------------------------------- 3. data
rgb_d, h_d, c_d = D.fetch_gamus_subset(SPLIT, N_TILES, os.path.join(WORK, "gamus"))
recs = D.pair_dirs(rgb_d, h_d, c_d, "*.h5", "*_AGL.h5", "*_CLS.h5")
print(f"paired {len(recs)} tiles")
D.describe(recs, n=3)

# --------------------------------------------------------- 4. checkpoint
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("JTRNEO/RS3DAda", "RS3DAda_vitl_DPT_height.pth")
print("checkpoint:", ckpt, f"{os.path.getsize(ckpt)/1e9:.2f} GB")

backend = P.RS3DAda(ckpt, synrs3d_dir=os.path.join(WORK, "SynRS3D"))
fn = backend.predict_fn

# ------------------------------------------------------- 5. three variants
buckets = E.auto_buckets_from_cls(c_d)

RUNS = [
    ("A raw (no GSD norm, no TTA)",  dict(gsd_normalise=False, tta=False)),
    ("B + GSD normalisation",        dict(gsd_normalise=True,  tta=False)),
    ("C + GSD + tiling/TTA",         dict(gsd_normalise=True,  tta=True)),
]

for tag, kw in RUNS:
    outdir = os.path.join(WORK, "preds_" + tag.split()[0])
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    import time; t0 = time.time()
    for i, rec in enumerate(recs):
        arr, gsd, _ = D.read_array(rec["rgb"])
        arr = arr[..., :3].astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0          # RS3DAda wants 0-255, NOT [0,1]
        h = P.predict_height(arr, fn, input_gsd=INPUT_GSD, model_gsd=MODEL_GSD,
                             tile=P.RS3DADA_PATCH, overlap=0.25,
                             verbose=(i == 0), **kw)
        np.save(os.path.join(outdir, f"{rec['tile_id']}_pred.npy"), h)
        if i % 10 == 0:
            print(f"  {i+1}/{len(recs)} {rec['tile_id']} "
                  f"[{h.min():.1f},{h.max():.1f}] m  {time.time()-t0:.0f}s")

    res = E.evaluate(outdir, h_d, buckets, "*", "*_AGL.h5")
    print()
    print(E.markdown_table(res["summary"], tag))
    E.append_progression(os.path.join(WORK, "progression.md"), tag,
                         res["summary"],
                         notes=f"RS3DAda vitl DPT height | {len(recs)} GAMUS "
                               f"{SPLIT} tiles | input_gsd={INPUT_GSD} "
                               f"model_gsd={MODEL_GSD}")

print("\n\n########## FINAL progression.md ##########\n")
print(open(os.path.join(WORK, "progression.md"), encoding="utf-8").read())
print("\nDownload /kaggle/working/progression.md and send it to me.")
