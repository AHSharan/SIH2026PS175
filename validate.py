"""
validate.py - score a finished run against a reference raster (LiDAR DSM,
CartoDEM, survey heights ...) that the user supplies.

    python validate.py out_dsm/<run> reference.tif

The reference is reprojected onto the run's own grid (GeoTIFF runs) or resized
to it (plain PNG/JPG runs, same picture only). It may be either
  * a surface (elevation above sea level)  -> scored against dsm.tif, or
  * heights above ground                   -> scored against ndsm.tif;
which one is decided from the data (whichever it is closer to) and reported.

Three rows, the same as the calibration options in calibrate.py:
  raw            nothing removed - the honest number
  datum shift    one constant offset removed (different vertical datums,
                 e.g. EGM2008 vs a local geoid, are a constant)
  robust affine  offset + scale of the model's heights (calibrate.fit_affine)
Rows 2 and 3 are FITTED ON HALF THE AREA AND SCORED ON THE OTHER HALF
(64-px checkerboard, both ways), so they cannot flatter themselves by fitting
the pixels they are scored on. Baseline row: the terrain alone (no model),
or "height 0 everywhere" for a heights-above-ground reference - it shows what
the model adds.

Writes validation.json, reference_on_grid.tif and residual.tif into the run
folder. With viewer_dir it also feeds the 3D viewer (Error colouring, the
Reference button and the validation table).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate as C   # noqa: E402
import eval as E        # noqa: E402

BLOCK = 64
MAX_FIT_SAMPLES = 20000


def _read(path):
    import rasterio
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        return a, s.crs, s.transform


def reference_on_grid(ref_path, crs, tr, shape):
    """Reference resampled onto the run's grid, NaN where it has no data."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    with rasterio.open(ref_path) as s:
        if crs is not None and s.crs is not None:
            coarser = abs(s.transform.a) >= abs(tr.a) * 0.99
            out = np.full(shape, np.nan, np.float32)
            reproject(rasterio.band(s, 1), out, src_nodata=s.nodata, dst_nodata=np.nan,
                      dst_transform=tr, dst_crs=crs,
                      resampling=Resampling.bilinear if coarser else Resampling.average)
            how = f"reprojected from {s.crs.to_string()} at {abs(s.transform.a):.3g} units/px"
        elif crs is None:
            a = s.read(1).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            H, W = shape
            if abs(a.shape[1] / a.shape[0] - W / H) > 0.02 * (W / H):
                raise ValueError(f"reference is {a.shape[1]}x{a.shape[0]} px but the run is "
                                 f"{W}x{H}: without map coordinates it must be the same picture")
            import predict as P
            ok = np.isfinite(a)
            out = P.resample_hw(np.where(ok, a, 0), shape, order=1)
            out[P.resample_hw(ok.astype(np.float32), shape, order=1) < 0.999] = np.nan
            how = f"resized from {a.shape[1]}x{a.shape[0]} px (no map coordinates)"
        else:
            raise ValueError("the reference has no map coordinates but this run does: "
                             "give a georeferenced GeoTIFF")
    out = out.astype(np.float64)
    out[~np.isfinite(out) | (np.abs(out) > 1e5)] = np.nan
    return out, how


def _row(label, note, pred, ref, m, extra=None):
    k = m & np.isfinite(pred)
    if k.sum() < 50:
        return None
    r = E.metrics_from(pred[k] - ref[k], ref[k])
    out = {"label": label, "note": note, "rmse_m": r["rmse_m"], "mae_m": r["mae_m"],
           "bias_m": r["bias_m"], "r": r["pearson_r"], "n_px": r["n_px"],
           "acc_lt_1m": r["acc_lt_1.0m"] if "acc_lt_1.0m" in r else None}
    out.update(extra or {})
    return out


def score(base, obj, ref, scale_known=True, seed=0):
    """base + obj is the run's surface (base = terrain or 0). Returns the rows."""
    pred = base + obj
    valid = np.isfinite(ref) & np.isfinite(pred)
    if valid.sum() < 100:
        raise ValueError(f"the reference overlaps only {int(valid.sum())} valid pixels of this run")
    H, W = ref.shape
    rr, cc = np.mgrid[0:H, 0:W]
    fold = ((rr // BLOCK) + (cc // BLOCK)) % 2
    rng = np.random.default_rng(seed)

    shifted, affine = np.full_like(pred, np.nan), np.full_like(pred, np.nan)
    ts, ss = [], []
    for f in (0, 1):
        fit, test = valid & (fold == f), valid & (fold != f)
        if fit.sum() < 50 or test.sum() < 50:
            fit = test = valid                     # tiny overlap: no split possible
        t = float(np.median(ref[fit] - pred[fit]))
        shifted[test] = pred[test] + t
        ts.append(t)
        idx = np.flatnonzero(fit.ravel())
        if idx.size > MAX_FIT_SAMPLES:
            idx = rng.choice(idx, MAX_FIT_SAMPLES, replace=False)
        r_, c_ = np.unravel_index(idx, fit.shape)
        y = ref[r_, c_] - base[r_, c_]
        _, info = C.fit_affine(obj, (r_, c_, y), s_clip=(0.6, 1.6) if scale_known else (1e-3, 1e3))
        s, t2 = (info["s"], info["t"]) if info.get("applied") else (1.0, 0.0)
        affine[test] = base[test] + s * obj[test] + t2
        ss.append((s, t2))

    rows = [
        _row("Raw", "nothing removed - the honest number", pred, ref, valid),
        _row("Datum shift", "one constant offset removed (fitted on the other half)",
             shifted, ref, valid, {"offset_m": float(np.mean(ts))}),
        _row("Robust affine", "offset + scale of model heights (fitted on the other half)",
             affine, ref, valid, {"scale": float(np.mean([s for s, _ in ss])),
                                  "offset_m": float(np.mean([t for _, t in ss]))}),
    ]
    return [r for r in rows if r], valid


def validate(run_dir, ref_path, viewer_dir=None, scene=None, ref_name=None):
    import dsm as M
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as f:
        rep = json.load(f)
    kind = rep.get("kind")
    ndsm, crs, tr = _read(os.path.join(run_dir, "ndsm.tif"))
    if kind == "dsm":
        dem, _, _ = _read(os.path.join(run_dir, "dem.tif"))
    else:
        crs = None                                   # plain PNG/JPG: pixel grid only
        dem = np.zeros_like(ndsm)
    ref, how = reference_on_grid(ref_path, crs, tr, ndsm.shape)
    cover = float(np.mean(np.isfinite(ref)))

    # surface or heights above ground? whichever the run is closer to
    ref_type = "height"
    if kind == "dsm":
        m = np.isfinite(ref) & np.isfinite(dem)
        if m.sum() and np.median(np.abs(ref[m] - (dem[m] + ndsm[m]))) < np.median(np.abs(ref[m] - ndsm[m])):
            ref_type = "surface"
    base = dem if ref_type == "surface" else np.zeros_like(ndsm)
    obj = np.where(np.isfinite(ndsm), ndsm, np.nan)
    rows, valid = score(base, obj, ref, scale_known=rep.get("scale_known", True) is not False)

    if ref_type == "surface":
        b = _row("Terrain only (no model)", "the DEM alone - what the model adds",
                 dem, ref, valid)
    else:
        b = _row("Height 0 everywhere (no model)", "flat ground - what the model adds",
                 np.zeros_like(ref), ref, valid)
    if b:
        rows.append(b)

    res = {"reference": ref_name or os.path.basename(ref_path), "reference_type": ref_type,
           "compared_with": "dsm.tif" if ref_type == "surface" else "ndsm.tif",
           "resampling": how, "coverage": round(cover, 4), "rows": rows,
           "split": f"{BLOCK}-px checkerboard, fitted on one half, scored on the other, both ways"}

    surf_ref = ref if ref_type == "surface" or kind != "dsm" else dem + ref
    run_surf = dem + obj if kind == "dsm" else obj
    if crs is not None:
        tags = {"PRODUCT": f"reference {res['reference']} on this grid"}
        M.write_tif(os.path.join(run_dir, "reference_on_grid.tif"), surf_ref, crs, tr, tags)
        M.write_tif(os.path.join(run_dir, "residual.tif"), run_surf - surf_ref, crs, tr,
                    {"PRODUCT": "run minus reference (m), raw"})
    with open(os.path.join(run_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    if viewer_dir and scene:
        _to_viewer(viewer_dir, scene, surf_ref, run_surf, res)
    return res


def _to_viewer(viewer_dir, scene, surf_ref, run_surf, res):
    """<scene>_ref.bin on the viewer grid + metrics in the scene meta. Cells the
    reference does not cover take the run's own value (error 0 there)."""
    import predict as P
    mp = os.path.join(viewer_dir, f"{scene}_meta.json")
    with open(mp, encoding="utf-8") as f:
        meta = json.load(f)
    filled = np.where(np.isfinite(surf_ref), surf_ref, run_surf)
    filled = np.where(np.isfinite(filled), filled, np.nanmin(run_surf))
    small = P.resample_hw(filled.astype(np.float32), (meta["height"], meta["width"]), order=1)
    small.astype(np.float32).tofile(os.path.join(viewer_dir, f"{scene}_ref.bin"))
    raw = res["rows"][0]
    meta.update(has_reference=True, reference_name=res["reference"],
                metrics={"rmse_m": raw["rmse_m"], "mae_m": raw["mae_m"], "bias_m": raw["bias_m"]},
                validation=res)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("reference")
    a = ap.parse_args()
    r = validate(a.run_dir, a.reference)
    print(f"reference: {r['reference']} ({r['reference_type']}, {r['resampling']}, "
          f"covers {r['coverage']:.0%} of the run)")
    print(f"{'':32} {'RMSE':>7} {'MAE':>7} {'bias':>7} {'r':>6}")
    for x in r["rows"]:
        print(f"{x['label']:32} {x['rmse_m']:7.2f} {x['mae_m']:7.2f} {x['bias_m']:+7.2f} {x['r']:6.3f}")
    print(f"({r['split']})")
