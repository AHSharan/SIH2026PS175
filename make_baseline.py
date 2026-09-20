"""
Trivial baselines - the accuracy FLOOR any real model must beat.

These are not models. They are honest reference points:
  zero   : "everything is ground"      -> RMSE = RMS of the true heights
  const  : "everything is the mean"    -> the best possible constant guess
  oracle_ground_truth_smoothed : GT blurred - an UPPER bound sanity check,
           proving the eval harness reports ~0 error when pred==GT.

The third one matters: if eval.py does NOT report ~0 for pred==GT, the harness
is broken and every other number is worthless.
"""
import argparse
import glob
import os

import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="gamus/test/heights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["zero", "const", "copy_gt"], required=True)
    ap.add_argument("--const", type=float, default=None,
                    help="constant metres; default = median of valid GT")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.gt, "*.h5")))
    if not files:
        raise SystemExit(f"no GT in {a.gt}")

    # global stats over valid pixels (exclude the -5.0 sentinel)
    if a.mode == "const" and a.const is None:
        vals = []
        for p in files:
            with h5py.File(p, "r") as f:
                v = f["image"][:]
            v = v[(v != -5.0) & np.isfinite(v)]
            vals.append(v[::37])          # subsample, plenty for a median
        allv = np.concatenate(vals)
        a.const = float(np.median(allv))
        print(f"[baseline] global median height = {a.const:.3f} m "
              f"(mean {allv.mean():.3f}, p90 {np.percentile(allv,90):.2f})")

    for p in files:
        stem = os.path.basename(p).replace("_AGL.h5", "")
        with h5py.File(p, "r") as f:
            gt = f["image"][:].astype(np.float32)
        if a.mode == "zero":
            pred = np.zeros_like(gt)
        elif a.mode == "const":
            pred = np.full_like(gt, a.const)
        else:                              # copy_gt - harness self-test
            pred = gt.copy()
            pred[pred == -5.0] = np.nan
        np.save(os.path.join(a.out, f"{stem}_pred.npy"), pred)

    print(f"[baseline] wrote {len(files)} '{a.mode}' predictions -> {a.out}")


if __name__ == "__main__":
    main()
