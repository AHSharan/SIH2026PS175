"""
CPU tests for dsm.py. No network, no GPU, no model weights: synthetic rasters
with known values plus a constant-height backend, so every expected number is
exact. Also runs the shipped Chungthang sample with the dummy backend.

Run from the repo root:  python tests/test_dsm_cpu.py
What this cannot test: RS3DAda's real heights (GPU run, see DSM.md) and the
live Copernicus download (dsm.py falls back to --dem / the shipped sample).
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import dsm as M             # noqa: E402
import predict as P         # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def const_backend(h):
    return lambda chw: np.full(chw.shape[1:], h, np.float32)


def write_rgb(path, rgb, crs, tr):
    with rasterio.open(path, "w", driver="GTiff", width=rgb.shape[1],
                       height=rgb.shape[0], count=3, dtype="uint8",
                       crs=crs, transform=tr) as d:
        d.write(np.transpose(rgb, (2, 0, 1)))


def main():
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    try:
        # ---- 1. GSD from a geographic CRS (degrees -> metres)
        tr = from_origin(88.6, 27.61, 1e-5, 1e-5)
        gx, gy = M.gsd_from_transform(tr, rasterio.crs.CRS.from_epsg(4326), 27.6)
        # WGS84 at 27.6 deg: 1 deg lon = 98.76 km, 1 deg lat = 110.85 km
        check("geographic GSD x", abs(gx - 0.9876) < 0.002, f"{gx:.4f} m")
        check("geographic GSD y", abs(gy - 1.1085) < 0.002, f"{gy:.4f} m")

        # ---- 2. UTM image + a DEM in a DIFFERENT CRS (EPSG:4326)
        H = W = 300
        utm = rasterio.crs.CRS.from_epsg(32645)
        itr = from_origin(662000.0, 3054800.0, 0.5, 0.5)
        rgb = rng.integers(40, 220, (H, W, 3), dtype=np.uint8)
        rgb[:60, :80] = 0                      # scene fill touching the edge
        rgb[150, 150] = 0                      # one black shadow pixel inside
        img = os.path.join(tmp, "img.tif")
        write_rgb(img, rgb, utm, itr)

        # planar DEM z = 1000 + 0.01*easting_offset: bilinear keeps a plane exact
        from rasterio.warp import transform_bounds
        w, s, e, n = transform_bounds(utm, "EPSG:4326", 662000, 3054650, 662150, 3054800)
        dres = 1 / 3600
        dtr = from_origin(w - 0.005, n + 0.005, dres, dres)
        dh = int((n - s + 0.01) / dres) + 2
        dw = int((e - w + 0.01) / dres) + 2
        lon = dtr.c + (np.arange(dw) + 0.5) * dres
        dem_arr = np.tile(1000 + 5000 * (lon - lon[0]), (dh, 1)).astype(np.float32)
        dem_p = os.path.join(tmp, "dem.tif")
        with rasterio.open(dem_p, "w", driver="GTiff", width=dw, height=dh, count=1,
                           dtype="float32", crs="EPSG:4326", transform=dtr) as d:
            d.write(dem_arr, 1)

        out = os.path.join(tmp, "out")
        exp = os.path.join(tmp, "viewer")
        rep = M.run(img, out, fn=const_backend(7.5), dem=[dem_p], export=exp,
                    name="syn", tta=False, backend="const")
        with rasterio.open(os.path.join(out, "dsm.tif")) as d:
            dsm = d.read(1)
            check("CRS preserved", d.crs == utm, str(d.crs))
            check("transform preserved", d.transform == itr)
            check("float32 + nodata -9999 + LZW",
                  d.dtypes[0] == "float32" and d.nodata == -9999
                  and d.profile.get("compress") == "lzw")
            check("vertical datum tag", "EGM2008" not in d.tags().get("VERTICAL_DATUM", "")
                  and d.tags().get("VERTICAL_DATUM") == "as the supplied DEM")
        dem_g = rasterio.open(os.path.join(out, "dem.tif")).read(1)
        ndsm = rasterio.open(os.path.join(out, "ndsm.tif")).read(1)
        v = dsm != -9999
        check("DSM == DEM + nDSM exactly", np.array_equal(dsm[v], dem_g[v] + ndsm[v]))
        check("nDSM = backend constant", np.allclose(ndsm[v], 7.5), f"{ndsm[v].min()}..{ndsm[v].max()}")
        check("DEM reprojected from EPSG:4326 in range",
              900 < dem_g[v].min() and dem_g[v].max() < 1100,
              f"{dem_g[v].min():.1f}..{dem_g[v].max():.1f}")
        # exact value: the DEM is a plane in lon, and bilinear keeps a plane exact
        from rasterio.warp import transform as wt
        (lo,), _ = wt(utm, "EPSG:4326", [662000 + 200.5 * 0.5], [3054800 - 200.5 * 0.5])
        want = 1000 + 5000 * (lo - lon[0])
        check("DEM value exact at a pixel", abs(dem_g[200, 200] - want) < 0.01,
              f"{dem_g[200, 200]:.3f} vs {want:.3f}")
        check("edge fill -> nodata", (dsm[:60, :80] == -9999).all())
        check("interior black pixel kept", dsm[150, 150] != -9999)
        check("only the fill is nodata", (~v).sum() == 60 * 80, f"{(~v).sum()} px")
        s = rep["sanity"]
        check("sanity p50 = 7.5", s["ndsm_percentiles_m"]["50"] == 7.5)
        check("double-count = mean nDSM at 30 m", abs(s["double_count_30m_mean_m"] - 7.5) < 1e-6)
        check("report says no RMSE", "no RMSE" in s["note"] and "rmse" not in json.dumps(rep).lower().replace("no rmse", ""))
        meta = json.load(open(os.path.join(exp, "syn_meta.json")))
        hb = np.fromfile(os.path.join(exp, "syn_h.bin"), np.float32)
        check("viewer kind=dsm, h_base set",
              meta["kind"] == "dsm" and abs(meta["h_base"] - np.percentile(hb, 1)) < 1e-3)
        check("viewer bin matches dims", hb.size == meta["width"] * meta["height"])
        check("viewer scenes.json lists scene",
              json.load(open(os.path.join(exp, "scenes.json")))[0] == "syn")

        # ---- 3. window read keeps the georeference right
        out2 = os.path.join(tmp, "out2")
        M.run(img, out2, fn=const_backend(1.0), dem=[dem_p], window=(100, 120, 64, 50),
              tta=False, backend="const")
        with rasterio.open(os.path.join(out2, "dsm.tif")) as d:
            check("window transform offset",
                  d.transform.c == 662000 + 100 * 0.5 and d.transform.f == 3054800 - 120 * 0.5
                  and d.shape == (50, 64))

        # ---- 4. PNG (no georeference) -> nDSM + relative 0-1 DSM
        png = os.path.join(tmp, "a.png")
        Image.fromarray(rgb[60:, 80:]).save(png)
        try:
            M.run(png, os.path.join(tmp, "o3"), fn=const_backend(1.0), backend="const")
            check("PNG without --gsd refused", False)
        except SystemExit:
            check("PNG without --gsd refused", True)
        rep3 = M.run(png, os.path.join(tmp, "o3"), fn=P.dummy_backend(), gsd=0.3,
                     tta=False, backend="dummy", export=os.path.join(tmp, "v3"), name="p")
        rd = np.load(os.path.join(tmp, "o3", "rdsm_0to1.npy"))
        check("PNG -> rdsm in [0,1]", rep3["kind"] == "rdsm" and rd.min() >= 0 and rd.max() <= 1
              and rd.max() > 0.9)
        check("PNG viewer kind=rdsm",
              json.load(open(os.path.join(tmp, "v3", "p_meta.json")))["kind"] == "rdsm")

        # ---- 5. GLO-30 tile naming
        u = M.glo30_tiles((88.58, 27.55, 88.69, 27.65))
        check("GLO-30 tile name", len(u) == 1 and u[0].endswith(
            "Copernicus_DSM_COG_10_N27_00_E088_00_DEM/Copernicus_DSM_COG_10_N27_00_E088_00_DEM.tif"))
        u = M.glo30_tiles((-0.5, -0.5, 0.5, 0.5))
        check("GLO-30 S/W naming", len(u) == 4 and any("S01_00_W001" in x for x in u))

        # ---- 6. the shipped Indian sample, end to end (dummy model)
        if os.path.exists(M.SAMPLE["rgb"]):
            r = M.run(M.SAMPLE["rgb"], os.path.join(tmp, "ct"), backend="dummy",
                      dem=[M.SAMPLE["dem"]], tta=False, source_note=M.SAMPLE["credit"])
            d = rasterio.open(os.path.join(tmp, "ct", "dem.tif")).read(1)
            check("sample: CRS EPSG:32645, GSD 0.3052",
                  r["crs"] == "EPSG:32645" and r["gsd_m"] == 0.3052)
            check("sample: DEM fully covered, 1500-1900 m",
                  (d != -9999).all() and 1500 < d.min() and d.max() < 1900,
                  f"{d.min():.1f}..{d.max():.1f}")
        else:
            check("sample present", False, "samples/ missing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
