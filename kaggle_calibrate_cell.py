"""
Paste this as a SECOND Kaggle cell, AFTER kaggle_run.py has finished.

It reuses the predictions already on disk (preds_C) - no GPU, no re-inference,
runs in well under a minute - and gives you the before/after row for
ground-shift calibration. That is the fourth line of your progression table.

If you edited the repo since the first cell ran, restart and re-clone, or just
`!cd /kaggle/working/code && git pull`.
"""
import os
import sys
import glob
import numpy as np

WORK = "/kaggle/working"
sys.path.insert(0, os.path.join(WORK, "code"))

import eval as E          # noqa: E402
import calibrate as C     # noqa: E402

PRED_IN = os.path.join(WORK, "preds_C")        # best uncalibrated run
PRED_OUT = os.path.join(WORK, "preds_D_calib")
GT = os.path.join(WORK, "gamus", "test", "heights")
CLS = os.path.join(WORK, "gamus", "test", "classes")

os.makedirs(PRED_OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(PRED_IN, "*.npy")))
if not files:
    raise SystemExit(f"no predictions in {PRED_IN} - did cell 1 finish?")

shifts = []
for p in files:
    arr = np.load(p).astype(np.float32)
    cal, rep = C.ground_shift(arr, block=64, pct=20.0, sigma=2.0)
    np.save(os.path.join(PRED_OUT, os.path.basename(p)), cal)
    if rep.get("applied"):
        shifts.append(rep["shift_mean_m"])

print(f"[calib] {len(files)} tiles")
if shifts:
    print(f"[calib] mean ground shift removed: {np.mean(shifts):+.3f} m "
          f"(range {np.min(shifts):+.2f} .. {np.max(shifts):+.2f})")

buckets = E.auto_buckets_from_cls(CLS)

print("\n================ BEFORE (C: GSD + tiling/TTA) ================")
before = E.evaluate(PRED_IN, GT, buckets, "*", "*_AGL.h5")
print(E.markdown_table(before["summary"], "C  GSD + tiling/TTA"))

print("\n================ AFTER (D: + ground_shift) ===================")
after = E.evaluate(PRED_OUT, GT, buckets, "*", "*_AGL.h5")
print(E.markdown_table(after["summary"], "D  + ground_shift calibration"))

E.append_progression(os.path.join(WORK, "progression.md"),
                     "D + ground_shift calibration", after["summary"],
                     notes="calibrate.ground_shift(block=64, pct=20, sigma=2) "
                           "applied to run C predictions")

b, a = before["summary"]["overall"], after["summary"]["overall"]
print("\n---------------- overall delta ----------------")
print(f"RMSE {b['rmse_m']:.2f} -> {a['rmse_m']:.2f} m   "
      f"({a['rmse_m']-b['rmse_m']:+.2f})")
print(f"MAE  {b['mae_m']:.2f} -> {a['mae_m']:.2f} m   "
      f"({a['mae_m']-b['mae_m']:+.2f})")
print(f"bias {b['bias_m']:+.2f} -> {a['bias_m']:+.2f} m")
print(f"r    {b['pearson_r']:.3f} -> {a['pearson_r']:.3f}")
print("\nIf RMSE got WORSE, say so - that is a real result. ground_shift "
      "over-subtracts by ~0.5 m on GAMUS (measured), so if the model already "
      "underestimates, the two errors compound instead of cancelling.")
