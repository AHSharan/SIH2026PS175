"""
Cell 2 - class-wise bias calibration, fitted on VAL, applied to TEST.

WHY NOT ground_shift:
Measured bias on the test run is NEGATIVE everywhere (-2.27 m overall,
-7.07 m on forest): the model UNDER-predicts. ground_shift subtracts an
estimated ground level, which would push it further down. Wrong tool here.

WHAT THIS DOES INSTEAD:
  1. runs inference on the VALIDATION split (never used for the test numbers)
  2. asks the model for its OWN 8-class segmentation (free, same forward pass)
  3. fits a robust per-class height offset on val:  offset_c = median(gt - pred)
  4. applies those offsets to the TEST predictions, using the model's predicted
     segmentation on test - so no ground-truth labels are used at test time
  5. reports before/after, plus a global-offset-only variant for comparison

The class ids are never interpreted semantically. We do not need to know which
id means "tree"; we only need the offset that id implies. That avoids guessing
SynRS3D's class mapping.

METHODOLOGY NOTE for the deck: the offsets come from val and are applied
unchanged to test. Fitting them on test would be tuning on the test set and
the resulting number would be meaningless.
"""
import os
import sys
import glob
import json
import time

import numpy as np

WORK = "/kaggle/working"
sys.path.insert(0, os.path.join(WORK, "code"))

import data as D        # noqa: E402
import predict as P     # noqa: E402
import eval as E        # noqa: E402

INPUT_GSD, MODEL_GSD = 0.3, 0.5
N_VAL = 24
TEST_GT = os.path.join(WORK, "gamus", "test", "heights")
TEST_CLS = os.path.join(WORK, "gamus", "test", "classes")
PRED_C = os.path.join(WORK, "preds_C")
OUT_GLOBAL = os.path.join(WORK, "preds_D_global")
OUT_CLASS = os.path.join(WORK, "preds_E_classwise")
MIN_PX = 20000          # per-class pixel floor before we trust an offset

for d in (OUT_GLOBAL, OUT_CLASS):
    os.makedirs(d, exist_ok=True)
if not glob.glob(os.path.join(PRED_C, "*.npy")):
    raise SystemExit(f"no predictions in {PRED_C} - run cell 1 first")

# ---------------------------------------------------------------- model
from huggingface_hub import hf_hub_download   # noqa: E402
ckpt = hf_hub_download("JTRNEO/RS3DAda", "RS3DAda_vitl_DPT_height.pth")
backend = P.RS3DAda(ckpt, synrs3d_dir=os.path.join(WORK, "SynRS3D"))


def seg_for(rgb_1024):
    """Model's own class map at native res, resampled back to 1024 (nearest)."""
    crop = rgb_1024[:1022, :1022]
    chw = np.transpose(crop, (2, 0, 1)).astype(np.float32)
    s = backend.predict_seg(chw)
    out = np.zeros(rgb_1024.shape[:2], np.int16)
    out[:1022, :1022] = s
    out[1022:, :] = out[1021, :][None, :]
    out[:, 1022:] = out[:, 1021][:, None]
    return out


def load_rgb(path):
    a, _, _ = D.read_array(path)
    a = a[..., :3].astype(np.float32)
    return a * 255.0 if a.max() <= 1.5 else a


# ------------------------------------------------- 1. VAL: fit the offsets
print("=" * 66)
print("FITTING on the validation split (not the test split)")
print("=" * 66)
v_rgb, v_h, v_cls = D.fetch_gamus_subset("val", N_VAL, os.path.join(WORK, "gamus"))
vrecs = D.pair_dirs(v_rgb, v_h, v_cls, "*.h5", "*_AGL.h5", "*_CLS.h5")
print(f"val tiles: {len(vrecs)}")

sums, counts = {}, {}          # class -> list of residuals (subsampled)
all_resid = []
t0 = time.time()
for i, rec in enumerate(vrecs):
    rgb = load_rgb(rec["rgb"])
    gt, _, _ = D.read_array(rec["height"])
    gt = np.asarray(gt, np.float32)
    m = np.isfinite(gt) & (gt != -5.0)

    pred = P.predict_height(rgb, backend.predict_fn, input_gsd=INPUT_GSD,
                            model_gsd=MODEL_GSD, tile=P.RS3DADA_PATCH,
                            overlap=0.25, tta=True, gsd_normalise=True)
    seg = seg_for(rgb)
    resid = (gt - pred)[m]                       # positive => we under-predict
    all_resid.append(resid[::7])
    sc = seg[m]
    for c in np.unique(sc):
        sel = resid[sc == c]
        sums.setdefault(int(c), []).append(sel[::7])
        counts[int(c)] = counts.get(int(c), 0) + int(sel.size)
    print(f"  val {i+1}/{len(vrecs)} {rec['tile_id']}  {time.time()-t0:.0f}s",
          flush=True)

global_off = float(np.median(np.concatenate(all_resid)))
offsets = {}
for c, chunks in sums.items():
    v = np.concatenate(chunks)
    offsets[c] = float(np.median(v)) if counts[c] >= MIN_PX else global_off

print("\nfitted offsets (metres to ADD to the prediction):")
print(f"  global            {global_off:+.3f}")
for c in sorted(offsets):
    flag = "" if counts[c] >= MIN_PX else "  (too few px -> global)"
    print(f"  predicted class {c}  {offsets[c]:+.3f}   "
          f"n={counts[c]/1e6:.2f}M{flag}")
with open(os.path.join(WORK, "calib_offsets.json"), "w") as f:
    json.dump({"global": global_off, "per_class": offsets,
               "counts": counts, "n_val_tiles": len(vrecs)}, f, indent=2)

# ------------------------------------------------- 2. TEST: apply them
print("\n" + "=" * 66)
print("APPLYING to the test split")
print("=" * 66)
trecs = D.pair_dirs(os.path.join(WORK, "gamus", "test", "images"), TEST_GT,
                    TEST_CLS, "*.h5", "*_AGL.h5", "*_CLS.h5")
t0 = time.time()
for i, rec in enumerate(trecs):
    p = os.path.join(PRED_C, f"{rec['tile_id']}_pred.npy")
    if not os.path.exists(p):
        continue
    pred = np.load(p).astype(np.float32)
    np.save(os.path.join(OUT_GLOBAL, os.path.basename(p)),
            np.maximum(pred + global_off, 0.0))

    seg = seg_for(load_rgb(rec["rgb"]))
    corr = np.full(pred.shape, global_off, np.float32)
    for c, off in offsets.items():
        corr[seg == c] = off
    np.save(os.path.join(OUT_CLASS, os.path.basename(p)),
            np.maximum(pred + corr, 0.0))
    if i % 10 == 0:
        print(f"  test {i+1}/{len(trecs)}  {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------- 3. compare
buckets = E.auto_buckets_from_cls(TEST_CLS)
rows = []
for tag, d in (("C  GSD + tiling/TTA (before)", PRED_C),
               ("D  + global offset", OUT_GLOBAL),
               ("E  + class-wise offset", OUT_CLASS)):
    r = E.evaluate(d, TEST_GT, buckets, "*", "*_AGL.h5")
    rows.append((tag, r))
    print("\n" + E.markdown_table(r["summary"], tag))

for tag, r in rows[1:]:
    E.append_progression(os.path.join(WORK, "progression.md"), tag,
                         r["summary"],
                         notes=f"offsets fitted on {len(vrecs)} VAL tiles, "
                               f"applied unchanged to test; segmentation from "
                               f"the model's own head (no GT labels at test)")

print("\n---------------- overall ----------------")
for tag, r in rows:
    o = r["summary"]["overall"]
    print(f"{tag:32} RMSE {o['rmse_m']:5.2f}  MAE {o['mae_m']:5.2f}  "
          f"bias {o['bias_m']:+5.2f}  r {o['pearson_r']:.3f}")
print("\n---------------- forest ----------------")
for tag, r in rows:
    o = r["summary"].get("forest")
    if o:
        print(f"{tag:32} RMSE {o['rmse_m']:5.2f}  MAE {o['mae_m']:5.2f}  "
              f"bias {o['bias_m']:+5.2f}")
print("\nIf a row got WORSE, report it as-is. A calibration that fails on one "
      "bucket is a real finding, not a bug to hide.")
