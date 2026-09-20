"""
eval.py - height-map evaluation for DepthWizard (SIH PS 26175).

Metrics are the ones the PS names verbatim:
  "Evaluate RMSE, MAE, and correlation against LiDAR or reference data,
   including performance stability across urban, sparse, hilly, and forested
   landscapes."   -> RMSE, MAE, Pearson r, + bias and threshold accuracy.

DESIGN NOTE - this file must survive meeting a dataset it has never seen.
Final judging uses ISRO RGB optical satellite imagery (GeoTIFF), not GAMUS
(.h5). So:
  - reading is format-agnostic (GeoTIFF via rasterio, .h5 via h5py, .npy),
  - nothing here imports a model or assumes GAMUS conventions,
  - bucket names are the PS's four: urban / sparse / hilly / forest.

CAVEAT on 'hilly': a *hilly* bucket cannot be derived from nDSM, because nDSM
has terrain removed by construction. It is accepted from buckets.csv if you
supply one, but it is never auto-assigned. Reporting 'hilly' from nDSM alone
would be a fabricated number.

Usage:
  python eval.py --pred preds/ --gt gamus/test/heights/ --buckets buckets.csv \
                 --out metrics.json --error-maps errmaps/ --tag "rs3dada base"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import glob as _glob
import datetime as _dt

import numpy as np

PS_BUCKETS = ("urban", "sparse", "hilly", "forest")
THRESHOLDS_M = (1.0, 2.5, 5.0)


# --------------------------------------------------------------------------
# IO - deliberately not tied to any one dataset
# --------------------------------------------------------------------------
def read_height(path: str):
    """Read a single-band height raster. Returns (array float32, nodata|None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            key = "image" if "image" in f else list(f.keys())[0]
            return np.asarray(f[key][:], dtype=np.float32), None
    if ext == ".npy":
        return np.load(path).astype(np.float32), None
    if ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32), src.nodata
    raise ValueError(f"unsupported height format: {path}")


def stem_of(path: str, strip=("_RGB", "_IMG", "_AGL", "_CLS", "_pred", "_PRED", "_height")) -> str:
    s = os.path.splitext(os.path.basename(path))[0]
    for suf in strip:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def index_dir(d: str, pattern: str = "*") -> dict:
    return {stem_of(p): p for p in sorted(_glob.glob(os.path.join(d, pattern)))
            if os.path.isfile(p)}


# --------------------------------------------------------------------------
# resampling (pred -> GT grid)
# --------------------------------------------------------------------------
def resample_to(arr: np.ndarray, shape) -> np.ndarray:
    """Bilinear resample to `shape`. NaN-safe: NaNs are filled by nearest valid
    before interpolation so they do not bleed into good pixels."""
    from scipy.ndimage import map_coordinates, distance_transform_edt
    h, w = arr.shape
    H, W = shape
    a = arr.astype(np.float32).copy()
    bad = ~np.isfinite(a)
    if bad.any():
        if bad.all():
            return np.full(shape, np.nan, np.float32)
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        a = a[tuple(idx)]
    # pixel-centre aligned
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    xs = (np.arange(W) + 0.5) * w / W - 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    out = map_coordinates(a, [gy, gx], order=1, mode="nearest").astype(np.float32)
    if bad.any():
        bad_r = map_coordinates(bad.astype(np.float32), [gy, gx], order=1,
                                mode="nearest")
        out[bad_r > 0.5] = np.nan
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def accumulate(pred: np.ndarray, gt: np.ndarray, gt_nodata=None,
               nodata_sentinel: float | None = -5.0):
    """Return (err, gt_valid) for valid pixels only, or (None, None) if empty."""
    m = np.isfinite(pred) & np.isfinite(gt)
    if gt_nodata is not None:
        m &= (gt != np.float32(gt_nodata))
    if nodata_sentinel is not None:
        m &= (gt != np.float32(nodata_sentinel))
    if not m.any():
        return None, None
    return (pred[m] - gt[m]).astype(np.float64), gt[m].astype(np.float64)


def metrics_from(err: np.ndarray, gt: np.ndarray) -> dict:
    n = err.size
    pred = gt + err
    if n >= 2 and np.std(pred) > 1e-9 and np.std(gt) > 1e-9:
        r = float(np.corrcoef(pred, gt)[0, 1])
    else:
        r = float("nan")
    out = {
        "n_px": int(n),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "mae_m": float(np.mean(np.abs(err))),
        "bias_m": float(np.mean(err)),
        "pearson_r": r,
        "gt_mean_m": float(np.mean(gt)),
        "gt_p99_m": float(np.percentile(gt, 99)),
    }
    for t in THRESHOLDS_M:
        out[f"acc_lt_{t}m"] = float(np.mean(np.abs(err) < t))
    return out


# --------------------------------------------------------------------------
# error maps
# --------------------------------------------------------------------------
def save_error_map(pred, gt, path, vmax=10.0, title=""):
    """Diverging colormap, FIXED symmetric scale so tiles are comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    err = np.where(np.isfinite(pred) & np.isfinite(gt), pred - gt, np.nan)
    fig, ax = plt.subplots(figsize=(6, 5.4), dpi=110)
    im = ax.imshow(err, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{title}\npred - GT  (red = too tall, blue = too short)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.85, label="error (m)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------
# buckets
# --------------------------------------------------------------------------
def load_buckets(path: str | None) -> dict:
    if not path:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): (v.strip() if v else v) for k, v in row.items()}
            tid, b = row.get("tile_id"), (row.get("bucket") or "").lower()
            if not tid:
                continue
            if b not in PS_BUCKETS:
                print(f"[eval] WARNING bucket '{b}' for {tid} is not one of {PS_BUCKETS}")
            out[stem_of(tid)] = b
    print(f"[eval] loaded {len(out)} bucket assignments from {path}")
    return out


def auto_buckets_from_cls(cls_dir: str, pattern="*_CLS.h5") -> dict:
    """Derive urban/sparse/forest from GAMUS CLS rasters. Never returns 'hilly'
    (see module docstring)."""
    import h5py
    out = {}
    for p in sorted(_glob.glob(os.path.join(cls_dir, pattern))):
        with h5py.File(p, "r") as f:
            c = f["image"][:]
        n = c.size
        b = float((c == 3).sum()) / n      # building
        t = float((c == 6).sum()) / n      # tree
        out[stem_of(p)] = "urban" if b >= 0.20 else ("forest" if t >= 0.35 else "sparse")
    print(f"[eval] auto-bucketed {len(out)} tiles from CLS "
          f"(hilly not derivable from nDSM - supply buckets.csv for that)")
    return out


# --------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------
def evaluate(pred_dir, gt_dir, buckets=None, pred_glob="*", gt_glob="*",
             error_maps=None, err_vmax=10.0, nodata_sentinel=-5.0,
             clamp_negative_pred=True, limit=None):
    preds, gts = index_dir(pred_dir, pred_glob), index_dir(gt_dir, gt_glob)
    common = sorted(set(preds) & set(gts))
    if limit:
        common = common[:limit]
    missing = sorted(set(gts) - set(preds))
    if missing:
        print(f"[eval] WARNING {len(missing)} GT tiles have no prediction "
              f"(e.g. {missing[:3]}) - they are EXCLUDED, not counted as error")
    if not common:
        raise SystemExit(f"[eval] no matching tiles between {pred_dir} and {gt_dir}")
    print(f"[eval] evaluating {len(common)} tiles")

    buckets = buckets or {}
    per_tile, pooled = {}, {}
    n_resampled = 0

    if error_maps:
        os.makedirs(error_maps, exist_ok=True)

    for tid in common:
        pred, _ = read_height(preds[tid])
        gt, gt_nd = read_height(gts[tid])
        if pred.ndim == 3:
            pred = pred[..., 0] if pred.shape[-1] <= 4 else pred[0]

        if pred.shape != gt.shape:
            n_resampled += 1
            print(f"[eval] !! WARNING shape mismatch {tid}: pred{pred.shape} "
                  f"vs GT{gt.shape} -> RESAMPLING prediction to GT grid. "
                  f"This is a real accuracy risk; check your inference grid.")
            pred = resample_to(pred, gt.shape)

        if clamp_negative_pred:
            pred = np.where(np.isfinite(pred), np.maximum(pred, 0.0), pred)

        err, gtv = accumulate(pred, gt, gt_nd, nodata_sentinel)
        if err is None:
            print(f"[eval] WARNING {tid}: no valid overlapping pixels, skipped")
            continue

        per_tile[tid] = metrics_from(err, gtv)
        per_tile[tid]["bucket"] = buckets.get(tid, "unknown")

        for key in ("overall", buckets.get(tid, "unknown")):
            a, b = pooled.setdefault(key, ([], []))
            a.append(err); b.append(gtv)

        if error_maps:
            save_error_map(pred, gt, os.path.join(error_maps, f"{tid}_err.png"),
                           vmax=err_vmax, title=tid)

    summary = {}
    for key, (errs, gtvs) in pooled.items():
        m = metrics_from(np.concatenate(errs), np.concatenate(gtvs))
        m["n_tiles"] = sum(1 for t in per_tile.values()
                           if key == "overall" or t["bucket"] == key)
        summary[key] = m

    return {"summary": summary, "per_tile": per_tile,
            "n_tiles": len(per_tile), "n_resampled": n_resampled}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def markdown_table(summary: dict, tag: str = "") -> str:
    order = ["overall"] + [b for b in PS_BUCKETS if b in summary] + \
            [k for k in summary if k not in PS_BUCKETS and k != "overall"]
    hdr = ("| bucket | tiles | px | RMSE (m) | MAE (m) | bias (m) | r | "
           "<1m | <2.5m | <5m |")
    sep = "|" + "---|" * 10
    rows = [hdr, sep]
    for k in order:
        if k not in summary:
            continue
        m = summary[k]
        rows.append(
            f"| **{k}** | {m['n_tiles']} | {m['n_px']/1e6:.1f}M | "
            f"{m['rmse_m']:.2f} | {m['mae_m']:.2f} | {m['bias_m']:+.2f} | "
            f"{m['pearson_r']:.3f} | {m['acc_lt_1.0m']*100:.1f}% | "
            f"{m['acc_lt_2.5m']*100:.1f}% | {m['acc_lt_5.0m']*100:.1f}% |")
    title = f"### {tag}\n\n" if tag else ""
    return title + "\n".join(rows)


def append_progression(path, tag, summary, notes=""):
    """Append one eval run to progression.md - the table that goes in the deck."""
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("# DepthWizard - evaluation progression\n\n"
                    "Every row is a real measured run on real data.\n"
                    "nDSM (height above ground), metres. GAMUS test split.\n\n")
        f.write(f"\n## {_dt.date.today().isoformat()} - {tag}\n\n")
        if notes:
            f.write(f"{notes}\n\n")
        f.write(markdown_table(summary) + "\n")
    print(f"[eval] appended to {path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate predicted height rasters.")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred-glob", default="*")
    ap.add_argument("--gt-glob", default="*")
    ap.add_argument("--buckets", default=None, help="CSV tile_id,bucket")
    ap.add_argument("--auto-buckets-cls", default=None,
                    help="dir of GAMUS *_CLS.h5 to derive buckets from")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--error-maps", default=None)
    ap.add_argument("--err-vmax", type=float, default=10.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--progression", default="progression.md")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-progression", action="store_true")
    a = ap.parse_args()

    buckets = load_buckets(a.buckets)
    if a.auto_buckets_cls:
        auto = auto_buckets_from_cls(a.auto_buckets_cls)
        auto.update(buckets)      # explicit csv wins
        buckets = auto

    res = evaluate(a.pred, a.gt, buckets, a.pred_glob, a.gt_glob,
                   error_maps=a.error_maps, err_vmax=a.err_vmax, limit=a.limit)

    res["tag"] = a.tag
    res["date"] = _dt.date.today().isoformat()
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print()
    print(markdown_table(res["summary"], a.tag))
    print()
    if res["n_resampled"]:
        print(f"[eval] !! {res['n_resampled']}/{res['n_tiles']} tiles needed resampling")
    print(f"[eval] wrote {a.out}")

    if not a.no_progression:
        append_progression(a.progression, a.tag or "(untagged)", res["summary"], a.notes)


if __name__ == "__main__":
    main()
