"""
calibrate.py - pure numpy. No model, no torch, no rasterio.

Turns a scale/shift-ambiguous height prediction into something metric.
This is the PS's mandated "Scale Calibration" milestone:

    "Develop a module that converts relative depth to absolute height using
     scene-level statistics, low-resolution DEMs, semantic priors, or minimal
     Ground Control Points for georeferenced inputs."

Two methods:

  ground_shift(pred)      - scene-level statistics, no reference needed.
                            Estimates the local ground level per block and
                            subtracts it. Removes per-tile bias and slow
                            terrain drift. This is the one that works when
                            you have nothing else.

  fit_affine(pred, ref)   - the GCP / DEM-sample case. Robust h = s*pred + t
                            via IRLS-Huber with optional RANSAC seeding.

Both return (calibrated_array, report_dict) so the report can go in calib.json
and, more importantly, so you can SEE what the calibration decided instead of
it silently rescaling your numbers.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, NaN-aware (normalised by a blurred mask)."""
    if sigma <= 0:
        return a
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()

    valid = np.isfinite(a).astype(np.float64)
    filled = np.where(np.isfinite(a), a, 0.0).astype(np.float64)

    def conv(arr):
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, arr)
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)

    num, den = conv(filled), conv(valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-8, num / den, np.nan)


def _block_reduce_percentile(a: np.ndarray, block: int, pct: float) -> np.ndarray:
    """Per-block median of values below the `pct`-th percentile of that block.

    Using the median OF THE LOW TAIL (not the percentile itself) is more
    robust: a single dark outlier pixel cannot drag the estimate.
    """
    H, W = a.shape
    nby, nbx = int(np.ceil(H / block)), int(np.ceil(W / block))
    out = np.full((nby, nbx), np.nan)
    for by in range(nby):
        for bx in range(nbx):
            b = a[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            v = b[np.isfinite(b)]
            if v.size < 8:
                continue
            thr = np.percentile(v, pct)
            low = v[v <= thr]
            if low.size:
                out[by, bx] = np.median(low)
    return out


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    if not np.isnan(a).any():
        return a
    if np.isnan(a).all():
        return np.zeros_like(a)
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(np.isnan(a), return_distances=False,
                                 return_indices=True)
    return a[tuple(idx)]


def _resize_bilinear(a: np.ndarray, out_hw) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    H, W = out_hw
    h, w = a.shape
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    xs = (np.arange(W) + 0.5) * w / W - 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return map_coordinates(a, [gy, gx], order=1, mode="nearest")


# --------------------------------------------------------------------------
# (a) ground shift
# --------------------------------------------------------------------------
def ground_shift(pred: np.ndarray, block: int = 64, pct: float = 20.0,
                 sigma: float = 2.0, clamp_min: float | None = 0.0,
                 ground_mask: np.ndarray | None = None):
    """Estimate a smooth per-pixel ground level and subtract it.

    For each `block`x`block` cell take the median of the lowest `pct`-percentile
    of predicted values -> that is "the ground here". Gaussian-smooth the field
    across blocks (sigma in BLOCK units), upsample, subtract.

    Removes constant per-tile bias and slow drift. It cannot fix a wrong scale
    - that is what fit_affine is for.

    ground_mask: optional bool array of known-ground pixels (e.g. road/ground
    class from a segmentation head). When given, only those pixels feed the
    estimate, which is much sharper than the percentile fallback.
    """
    a = np.asarray(pred, np.float64)
    src = a if ground_mask is None else np.where(ground_mask, a, np.nan)

    coarse = _block_reduce_percentile(src, block, pct)
    n_est = int(np.isfinite(coarse).sum())
    if n_est == 0:
        return np.asarray(pred, np.float32), {
            "method": "ground_shift", "applied": False,
            "reason": "no block had enough valid pixels"}

    coarse_f = _fill_nan_nearest(coarse)
    smooth = _gaussian_blur(coarse_f, sigma)
    smooth = _fill_nan_nearest(smooth)
    field = _resize_bilinear(smooth, a.shape)

    out = a - field
    if clamp_min is not None:
        out = np.maximum(out, clamp_min)
    out = np.where(np.isfinite(a), out, np.nan)

    return out.astype(np.float32), {
        "method": "ground_shift", "applied": True,
        "block": block, "pct": pct, "sigma_blocks": sigma,
        "blocks_estimated": n_est,
        "shift_mean_m": float(np.nanmean(field)),
        "shift_min_m": float(np.nanmin(field)),
        "shift_max_m": float(np.nanmax(field)),
        "used_ground_mask": ground_mask is not None,
    }


# --------------------------------------------------------------------------
# (b) robust affine
# --------------------------------------------------------------------------
def _huber_irls(x, y, s0=1.0, t0=0.0, delta=1.0, iters=25):
    s, t = float(s0), float(t0)
    for _ in range(iters):
        r = y - (s * x + t)
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (delta * scale)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
        sw = w.sum()
        if sw < 1e-9:
            break
        mx, my = (w * x).sum() / sw, (w * y).sum() / sw
        vxx = (w * (x - mx) ** 2).sum()
        if vxx < 1e-12:
            break
        s_new = (w * (x - mx) * (y - my)).sum() / vxx
        t_new = my - s_new * mx
        if abs(s_new - s) < 1e-9 and abs(t_new - t) < 1e-9:
            s, t = s_new, t_new
            break
        s, t = s_new, t_new
    return s, t


def fit_affine(pred: np.ndarray, reference_samples, s_clip=(0.6, 1.6),
               ransac_iters: int = 200, ransac_thresh: float = 2.0,
               min_samples: int = 3, seed: int = 0):
    """Robust h = s*pred + t against sparse reference heights (GCPs / DEM).

    reference_samples: either
        (rows, cols, heights) arrays, or
        an (N,3) array of [row, col, height], or
        a full-shape array with NaN where unknown.

    s is clipped to `s_clip` - an unconstrained fit on a handful of noisy GCPs
    can produce a wild scale that destroys an otherwise decent prediction.
    """
    a = np.asarray(pred, np.float64)

    if isinstance(reference_samples, np.ndarray) and reference_samples.shape == a.shape:
        m = np.isfinite(reference_samples) & np.isfinite(a)
        x, y = a[m], np.asarray(reference_samples, np.float64)[m]
    elif isinstance(reference_samples, (tuple, list)) and len(reference_samples) == 3:
        r, c, h = (np.asarray(v) for v in reference_samples)
        r = np.clip(r.astype(int), 0, a.shape[0] - 1)
        c = np.clip(c.astype(int), 0, a.shape[1] - 1)
        x, y = a[r, c], h.astype(np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
    else:
        arr = np.asarray(reference_samples, np.float64)
        r = np.clip(arr[:, 0].astype(int), 0, a.shape[0] - 1)
        c = np.clip(arr[:, 1].astype(int), 0, a.shape[1] - 1)
        x, y = a[r, c], arr[:, 2]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]

    n = x.size
    if n < min_samples:
        return np.asarray(pred, np.float32), {
            "method": "fit_affine", "applied": False,
            "reason": f"only {n} usable samples, need {min_samples}"}

    # RANSAC seed
    rng = np.random.default_rng(seed)
    best, best_inl = (1.0, 0.0), -1
    if n >= 2:
        for _ in range(ransac_iters):
            i, j = rng.choice(n, 2, replace=False) if n >= 2 else (0, 0)
            if abs(x[i] - x[j]) < 1e-6:
                continue
            s = (y[i] - y[j]) / (x[i] - x[j])
            t = y[i] - s * x[i]
            inl = int((np.abs(y - (s * x + t)) < ransac_thresh).sum())
            if inl > best_inl:
                best_inl, best = inl, (s, t)

    s, t = _huber_irls(x, y, *best)
    s_raw = float(s)
    s = float(np.clip(s, *s_clip))
    if s != s_raw:                    # refit t with s pinned
        t = float(np.median(y - s * x))

    out = (s * a + t).astype(np.float32)
    resid = y - (s * x + t)
    return out, {
        "method": "fit_affine", "applied": True,
        "s": float(s), "t": float(t), "s_before_clip": s_raw,
        "s_was_clipped": bool(s != s_raw),
        "n_samples": int(n), "ransac_inliers": int(max(best_inl, 0)),
        "resid_rmse_m": float(np.sqrt(np.mean(resid ** 2))),
        "resid_mae_m": float(np.mean(np.abs(resid))),
    }


# --------------------------------------------------------------------------
# CLI: calibrate a directory of predictions
# --------------------------------------------------------------------------
def main():
    import argparse
    import glob
    import json
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="dir of *_pred.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["ground_shift"], default="ground_shift")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--pct", type=float, default=20.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.pred, "*.npy")))
    if not files:
        raise SystemExit(f"no *.npy in {a.pred}")

    reports = {}
    for p in files:
        arr = np.load(p).astype(np.float32)
        cal, rep = ground_shift(arr, block=a.block, pct=a.pct, sigma=a.sigma)
        np.save(os.path.join(a.out, os.path.basename(p)), cal)
        reports[os.path.basename(p)] = rep

    with open(os.path.join(a.out, "calib.json"), "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    shifts = [r["shift_mean_m"] for r in reports.values() if r.get("applied")]
    print(f"[calibrate] {len(files)} tiles -> {a.out}")
    if shifts:
        print(f"[calibrate] mean ground shift removed: {np.mean(shifts):+.3f} m "
              f"(range {np.min(shifts):+.2f} to {np.max(shifts):+.2f})")


if __name__ == "__main__":
    main()
