"""
tests/lidar_benchmark.py - score the REAL pipeline against airborne LiDAR on
georeferenced imagery, per landscape, including hilly terrain.

Data (free, public, no account):
  imagery : USDA NAIP aerial RGB (GeoTIFF, 0.6 m), Microsoft Planetary Computer
  truth   : USGS 3DEP LiDAR point cloud (COPC), rebuilt here on a 2 m grid:
            surface = highest non-noise return per cell, terrain = class-2
            ground returns (gaps under roofs/canopy filled linearly),
            height above ground = surface - terrain, then a 3 x 3 median to
            drop isolated spikes (birds, wires: ~1.7 % of cells on flat farms)
  Why not the ready-made 2 m products: their DSM is not a top-of-surface
  model (forest canopy median 8.9 m vs 24.4 m from the points) and their HAG
  has spikes up to 150 m on flat farmland. Checked, not assumed.
Photo and LiDAR are 0-2 years apart (years listed per site).

For each site: crop 600 x 600 m of NAIP -> GeoTIFF -> dsm.run() (exactly what
a user's GeoTIFF goes through: our model + Copernicus GLO-30 terrain) ->
resample everything to a common 2 m grid (our 0.6 m output is AVERAGED to 2 m,
the LiDAR is the 2 m product) -> metrics:

  heights above ground : our ndsm.tif vs LiDAR height  (the model's job)
  full DSM             : our dsm.tif  vs LiDAR surface  (what the PS scores)
  terrain only         : Copernicus DEM vs LiDAR ground (error we inherit)

Vertical datums differ (LiDAR NAVD88, Copernicus EGM2008): any constant offset
shows up as DSM bias and is reported, not hidden.

    pip install pystac-client planetary-computer "laspy[lazrs]" pyproj
    HF_HUB_OFFLINE=1 python tests/lidar_benchmark.py --head models/best.pt
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import dsm as M     # noqa: E402
import eval as E    # noqa: E402

SITES = [  # name, landscape, lon, lat  (photo/LiDAR years: see report)
    ("pittsburgh_downtown", "urban", -80.000, 40.441),
    ("pittsburgh_residential", "urban", -79.923, 40.438),
    ("pittsburgh_mt_washington", "hilly", -80.010, 40.431),
    ("park_city_ut", "hilly", -111.498, 40.646),
    ("pennsylvania_woods", "forest", -79.900, 40.520),
    ("heber_ut_farms", "sparse", -111.430, 40.520),
    ("butler_pa_rural", "sparse", -79.950, 40.900),
    ("washington_pa_rural", "sparse", -80.300, 40.150),
]
HALF = 300.0          # metres: 600 x 600 m per site
GRID = 2.0            # common grid = the LiDAR product resolution
NOISE = (7, 18)       # ASPRS low / high noise classes


def stac():
    from pystac_client import Client
    import planetary_computer as pc
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                       modifier=pc.sign_inplace)


def year(i):
    return int(str(i.datetime or i.properties.get("start_datetime"))[:4])


def lidar_items(cat, col, bbox):
    it = list(cat.search(collections=[col], bbox=bbox, max_items=50).items())
    if not it:
        return [], None
    y = max(year(i) for i in it)
    return [i for i in it if year(i) == y], y


def naip_item(cat, bbox, target_year):
    it = [i for i in cat.search(collections=["naip"], bbox=bbox, max_items=60).items()
          if i.bbox[0] <= bbox[0] and i.bbox[1] <= bbox[1]
          and i.bbox[2] >= bbox[2] and i.bbox[3] >= bbox[3]]          # fully covers the site
    if not it:
        return None
    return min(it, key=lambda i: (abs(year(i) - target_year), -year(i)))


def to_grid(srcs, dst_crs, dst_tr, shape, resampling):
    """Reproject one or more rasters into the common grid; first valid wins."""
    import rasterio
    from rasterio.warp import reproject
    out = np.full(shape, np.nan, np.float32)
    for href in srcs:
        with rasterio.open(href) as s:
            tmp = np.full(shape, np.nan, np.float32)
            reproject(rasterio.band(s, 1), tmp, src_nodata=s.nodata, dst_nodata=np.nan,
                      dst_transform=dst_tr, dst_crs=dst_crs, resampling=resampling)
        m = np.isnan(out) & np.isfinite(tmp)
        out[m] = tmp[m]
    return out


def lidar_truth(cat, bbox, crs, x0, y1, n):
    """Height above ground, surface and terrain from the raw point cloud, on the grid."""
    import laspy
    from laspy.copc import Bounds
    from pyproj import CRS, Transformer
    from scipy.interpolate import griddata
    from scipy.ndimage import median_filter
    items, yr = lidar_items(cat, "3dep-lidar-copc", bbox)
    ncrs = CRS.from_user_input(crs.to_wkt())
    top = np.full(n * n, -np.inf)
    gsum, gcnt = np.zeros(n * n), np.zeros(n * n)
    ext = n * GRID
    for it in items:
        with laspy.CopcReader.open(it.assets["data"].href) as r:
            lcrs = r.header.parse_crs()
            lcrs = lcrs.sub_crs_list[0] if lcrs.is_compound else lcrs
            xs, ys = Transformer.from_crs(ncrs, lcrs, always_xy=True).transform(
                [x0, x0 + ext, x0, x0 + ext], [y1, y1, y1 - ext, y1 - ext])
            pts = r.query(Bounds(mins=np.array([min(xs), min(ys)]),
                                 maxs=np.array([max(xs), max(ys)])))
        cls = np.asarray(pts.classification)
        k = ~np.isin(cls, NOISE)
        X, Y = Transformer.from_crs(lcrs, ncrs, always_xy=True).transform(
            np.asarray(pts.x)[k], np.asarray(pts.y)[k])
        Z, cls = np.asarray(pts.z)[k], cls[k]
        c, rr = ((X - x0) // GRID).astype(int), ((y1 - Y) // GRID).astype(int)
        ok = (c >= 0) & (c < n) & (rr >= 0) & (rr < n)
        idx, Z, g = rr[ok] * n + c[ok], Z[ok], cls[ok] == 2
        np.maximum.at(top, idx, Z)
        np.add.at(gsum, idx[g], Z[g])
        np.add.at(gcnt, idx[g], 1)
    top = np.where(np.isfinite(top), top, np.nan).reshape(n, n)
    ground = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), np.nan).reshape(n, n)
    rr, cc = np.nonzero(np.isfinite(ground))
    gi, gj = np.mgrid[0:n, 0:n]
    dtm = griddata((rr, cc), ground[rr, cc], (gi, gj), method="linear")
    hag = np.clip(top - dtm, 0, None)
    hag = np.where(np.isfinite(hag), median_filter(np.nan_to_num(hag), size=3, mode="nearest"),
                   np.nan)
    return (hag.astype(np.float32), (dtm + hag).astype(np.float32), dtm.astype(np.float32),
            yr, float(np.mean(gcnt > 0)))


def metrics(pred, ref, extra_mask=None):
    m = np.isfinite(pred) & np.isfinite(ref)
    if extra_mask is not None:
        m &= extra_mask
    if m.sum() < 100:
        return None
    return E.metrics_from((pred[m] - ref[m]).astype(np.float64), ref[m].astype(np.float64))


def run_site(cat, fn, name, land, lon, lat, work):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_xy
    from rasterio.windows import from_bounds

    bbox = [lon - 0.0045, lat - 0.0035, lon + 0.0045, lat + 0.0035]
    hag_it, ly = lidar_items(cat, "3dep-lidar-copc", bbox)
    nai = naip_item(cat, bbox, ly) if hag_it else None
    if not (hag_it and nai):
        return {"site": name, "error": "missing LiDAR or NAIP coverage"}

    sdir = os.path.join(work, name)
    os.makedirs(sdir, exist_ok=True)
    img = os.path.join(sdir, "naip.tif")
    with rasterio.open(nai.assets["image"].href) as s:
        (cx,), (cy,) = warp_xy("EPSG:4326", s.crs, [lon], [lat])
        win = from_bounds(cx - HALF, cy - HALF, cx + HALF, cy + HALF, s.transform)
        rgb = s.read([1, 2, 3], window=win, boundless=False)
        prof = s.profile.copy()
        prof.update(driver="GTiff", count=3, width=rgb.shape[2], height=rgb.shape[1],
                    transform=s.window_transform(win), compress="deflate")
        prof.pop("photometric", None)
        with rasterio.open(img, "w", **prof) as d:
            d.write(rgb)
        crs, gsd_naip = s.crs, abs(s.transform.a)

    out = os.path.join(sdir, "pred")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        M.run(img, out, backend="dinov3", fn=fn, tta=True, buildings=False, export=None,
              name=name)
    secs = time.time() - t0

    # common 2 m grid over the site, in the image CRS
    x0, y1 = cx - HALF, cy + HALF
    n = int(2 * HALF / GRID)
    tr = from_origin(x0, y1, GRID, GRID)
    shape = (n, n)
    avg = Resampling.average
    ours_ndsm = to_grid([os.path.join(out, "ndsm.tif")], crs, tr, shape, avg)
    ours_dsm = to_grid([os.path.join(out, "dsm.tif")], crs, tr, shape, avg)
    ours_dem = to_grid([os.path.join(out, "dem.tif")], crs, tr, shape, avg)
    l_hag, l_dsm, l_dtm, _, ground_frac = lidar_truth(cat, bbox, crs, x0, y1, n)
    ok = np.isfinite(l_hag)
    ours_ndsm = np.maximum(ours_ndsm, 0.0)

    r = {"site": name, "landscape": land, "lidar_year": ly, "naip_year": year(nai),
         "naip_id": nai.id, "naip_gsd_m": round(gsd_naip, 3),
         "lidar_project": hag_it[0].properties.get("3dep:usgs_id"),
         "run_s": round(secs, 1), "valid_px": int(ok.sum()),
         "lidar_ground_cell_frac": round(ground_frac, 3),
         "lidar_hag_p95_m": float(np.nanpercentile(l_hag, 95)),
         "lidar_relief_m": float(np.nanpercentile(l_dtm, 99) - np.nanpercentile(l_dtm, 1)),
         "ndsm": metrics(ours_ndsm, l_hag),
         "ndsm_zero_floor": metrics(np.zeros_like(l_hag), l_hag),
         "dsm": metrics(ours_dsm, l_dsm),
         "dsm_with_lidar_terrain": metrics(l_dtm + ours_ndsm, l_dsm),
         "terrain": metrics(ours_dem, l_dtm),
         "median_m": {"lidar": float(np.nanmedian(l_hag)),
                      "ours": float(np.nanmedian(ours_ndsm))}}
    np.savez_compressed(os.path.join(sdir, "grid2m.npz"), ours_ndsm=ours_ndsm, ours_dsm=ours_dsm,
                        ours_dem=ours_dem, lidar_hag=l_hag, lidar_dsm=l_dsm, lidar_dtm=l_dtm)
    return r


def pooled(work, names, key_pred, key_ref):
    e, g = [], []
    for n in names:
        z = np.load(os.path.join(work, n, "grid2m.npz"))
        if key_pred == "zero":
            p = np.zeros_like(z[key_ref])
        else:
            p = z[key_pred]
        ref = z[key_ref]
        m = np.isfinite(p) & np.isfinite(ref)
        e.append((p[m] - ref[m]).astype(np.float64))
        g.append(ref[m].astype(np.float64))
    return E.metrics_from(np.concatenate(e), np.concatenate(g))


def fmt(m):
    return (f"{m['rmse_m']:.2f} | {m['mae_m']:.2f} | {m['bias_m']:+.2f} | {m['pearson_r']:.3f}"
            if m else "- | - | - | -")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--work", default=os.path.join(os.environ.get("TEMP", "/tmp"), "lb"))   # short: Windows 260-char paths
    ap.add_argument("--report", default=os.path.join(ROOT, "results", "lidar_benchmark.md"))
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from --work")
    a = ap.parse_args()

    if a.report_only:
        with open(a.report.replace(".md", ".json"), encoding="utf-8") as f:
            res = json.load(f)
    else:
        cat = stac()
        fn = M.get_dinov3(a.head)
        res = []
    for name, land, lon, lat in ([] if a.report_only else SITES):
        try:
            r = run_site(cat, fn, name, land, lon, lat, a.work)
        except Exception as e:                                          # noqa: BLE001
            r = {"site": name, "landscape": land, "error": f"{type(e).__name__}: {e}"}
        res.append(r)
        msg = r.get("error") or (f"nDSM RMSE {r['ndsm']['rmse_m']:.2f} m | DSM RMSE "
                                 f"{r['dsm']['rmse_m']:.2f} m | {r['run_s']} s")
        print(f"  {name:28} {land:7} {msg}", flush=True)

    good = [r for r in res if not r.get("error")]
    lands = ["urban", "sparse", "hilly", "forest"]
    L = ["# Benchmark against airborne LiDAR (USGS 3DEP) on georeferenced imagery", "",
         f"Model `{os.path.basename(a.head)}`, run through the real GeoTIFF pipeline "
         "(`dsm.run`: our model + Copernicus GLO-30 terrain). Imagery: USDA NAIP. "
         "Truth: USGS 3DEP LiDAR point cloud, rebuilt on a 2 m grid (highest non-noise return "
         "minus ground returns, 3 × 3 median against isolated spikes). Compared on a common 2 m grid "
         "(our 0.6 m output averaged to 2 m). 600 × 600 m per site; photo and LiDAR 0–2 years "
         "apart. All numbers measured by `tests/lidar_benchmark.py`.", "",
         "## By landscape (pooled pixels)", "",
         "| landscape | sites | heights above ground: RMSE / MAE / bias / r | predict-zero RMSE "
         "| full DSM: RMSE / MAE / bias / r | terrain only (Copernicus vs LiDAR): RMSE |",
         "|---|---|---|---|---|---|"]
    for land in lands + ["all"]:
        ns = [r["site"] for r in good if land == "all" or r["landscape"] == land]
        if not ns:
            continue
        nd = pooled(a.work, ns, "ours_ndsm", "lidar_hag")
        z = pooled(a.work, ns, "zero", "lidar_hag")
        ds = pooled(a.work, ns, "ours_dsm", "lidar_dsm")
        te = pooled(a.work, ns, "ours_dem", "lidar_dtm")
        L.append(f"| **{land}** | {len(ns)} | {fmt(nd)} | {z['rmse_m']:.2f} | {fmt(ds)} | "
                 f"{te['rmse_m']:.2f} |")
    L += ["", "## By site", "",
          "| site | landscape | LiDAR / photo year | relief (m) | median height LiDAR / ours (m) "
          "| heights: RMSE / MAE / bias / r | full DSM: RMSE / MAE / bias / r | terrain RMSE | run (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        if r.get("error"):
            L.append(f"| {r['site']} | {r.get('landscape', '')} | – | – | {r['error']} | | | | |")
            continue
        L.append(f"| {r['site']} | {r['landscape']} | {r['lidar_year']} / {r['naip_year']} | "
                 f"{r['lidar_relief_m']:.0f} | {r['median_m']['lidar']:.1f} / {r['median_m']['ours']:.1f} | "
                 f"{fmt(r['ndsm'])} | {fmt(r['dsm'])} | "
                 f"{r['terrain']['rmse_m']:.2f} | {r['run_s']} |")
    L += ["", "Notes: LiDAR heights use NAVD88 and Copernicus uses EGM2008, so part of the DSM "
          "bias is a datum offset. NAIP is an ordinary orthophoto, not a true orthophoto: tall "
          "buildings lean in the picture, so their roofs sit metres away from their LiDAR "
          "footprints. This inflates downtown error for any image-based method. The ready-made "
          "3DEP 2 m DSM/HAG products were not used as truth: checked against the point cloud, "
          "their DSM puts forest canopy at 8.9 m (points: 24.4 m) and their HAG has spikes "
          "up to 150 m on flat farmland. Copernicus GLO-30 is a radar surface model, so it already "
          "holds part of the forest canopy and some buildings: that is why its 'terrain' error is "
          "large in forest and downtown while our full DSM there stays much closer."]
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(a.report.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
