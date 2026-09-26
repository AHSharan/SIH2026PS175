"""
dsm.py - georeferenced RGB image in -> absolute DSM GeoTIFF out.

    DSM (m above EGM2008 geoid) = DEM (Copernicus GLO-30) + nDSM (model)

Inputs
  GeoTIFF (any CRS): GSD comes from the file's transform. Outputs keep its
      CRS and grid, so they overlay the image exactly in QGIS/ArcGIS.
  PNG / JPG (no georeference): there is no DEM to add, so we write the nDSM
      (metres above local ground) plus a 0-1 relative DSM. --gsd is required.

Outputs (in --out)
  dsm.tif   float32, nodata -9999, LZW, same CRS/transform as the image
  ndsm.tif  model height above ground (m)
  dem.tif   Copernicus GLO-30 bilinearly resampled onto the image grid
  report.json  provenance + sanity checks (there is NO LiDAR truth, so NO RMSE)
  viewer assets (--export) with meta kind='dsm' and h_base for the 3D viewer

Caveat, written into every output: GLO-30 is itself a 30 m SURFACE model.
In built-up or forested blocks it already contains a smoothed share of the
buildings/trees, so DEM + nDSM partly double-counts there. report.json gives
the size of that effect for the scene (mean nDSM at the 30 m cell scale).

Run (the Chungthang sample shipped in samples/):
    python dsm.py samples/chungthang_wv2.tif --out out_dsm/chungthang
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

import predict as P

NODATA = -9999.0
GLO30_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
             "{n}/{n}.tif")
GLO30_NAME = "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM"
GLO30_CREDIT = ("Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus "
                "Defence and Space GmbH 2014-2018, provided under COPERNICUS "
                "by the European Union and ESA")
DOUBLE_COUNT_NOTE = ("GLO-30 is a 30 m surface model (X-band radar, 2011-2015). "
                     "In dense built-up or forested cells it already contains a "
                     "smoothed part of the buildings/trees, so DEM + nDSM partly "
                     "double-counts there. The DEM also predates the image.")

# reading a URL through GDAL: never list the S3 'directory' (slow, often 403)
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")


# ==========================================================================
# reading
# ==========================================================================
def _gdal_path(p: str) -> str:
    return "/vsicurl/" + p if p.startswith(("http://", "https://")) else p


def gsd_from_transform(transform, crs, center_lat: float | None = None):
    """Pixel size in metres (x, y). Handles geographic CRSs (degrees)."""
    rx, ry = abs(transform.a), abs(transform.e)
    if crs is None:
        raise ValueError("no CRS")
    if crs.is_geographic:
        if center_lat is None:
            raise ValueError("geographic CRS needs the centre latitude")
        lat = math.radians(center_lat)
        # WGS84 metres per degree at this latitude
        m_lat = 111132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
        m_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
        return rx * m_lon, ry * m_lat
    f = crs.linear_units_factor[1] if crs.is_projected else 1.0
    return rx * f, ry * f


def _edge_black(rgb):
    """Large pure-black regions touching the image edge = scene fill. Black
    pixels inside the image are real (deep shadow) and stay valid."""
    from scipy.ndimage import label
    z = np.all(rgb == 0, axis=2)
    if not z.any():
        return z
    lab, _ = label(z)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    sizes = np.bincount(lab.ravel())
    fill = edge[sizes[edge] >= max(1000, z.size // 1000)]
    return np.isin(lab, fill)


def read_image(path: str, window=None):
    """-> dict(rgb HxWx3 float32 0-255, valid HxW bool, crs, transform, gsd,
    georef bool). window = (col, row, width, height) in pixels."""
    ext = os.path.splitext(path.split("?")[0])[1].lower()
    if ext in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        a = np.asarray(Image.open(path).convert("RGB"), np.float32)
        if window:
            c, r, w, h = window
            a = a[r:r + h, c:c + w]
        return {"rgb": a, "valid": np.ones(a.shape[:2], bool), "crs": None,
                "transform": None, "gsd": None, "georef": False}

    import rasterio
    from rasterio.windows import Window
    with rasterio.open(_gdal_path(path)) as s:
        win = Window(*window) if window else None
        n = min(3, s.count)
        a = s.read(list(range(1, n + 1)), window=win).astype(np.float32)
        if n == 1:
            a = np.repeat(a, 3, 0)
        tr = s.window_transform(win) if win else s.transform
        crs = s.crs
        # valid pixels: dataset mask (nodata/alpha) where the file has one
        try:
            m = s.dataset_mask(window=win) > 0
        except Exception:
            m = np.ones(a.shape[1:], bool)
        dtype = s.dtypes[0]
    rgb = np.transpose(a, (1, 2, 0))
    if dtype != "uint8":
        # 16-bit / float imagery -> 0-255 by a robust 2-98 % stretch per band.
        # RS3DAda was trained on 8-bit RGB; this is a display-style stretch.
        for b in range(3):
            v = rgb[..., b][m]
            lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
            rgb[..., b] = np.clip((rgb[..., b] - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    m &= ~_edge_black(rgb)                  # Maxar/OAM black fill
    out = {"rgb": rgb, "valid": m, "crs": crs, "transform": tr, "gsd": None,
           "georef": crs is not None}
    if crs is not None:
        H, W = rgb.shape[:2]
        lat = None
        if crs.is_geographic:
            lat = tr.f + tr.e * H / 2
        gx, gy = gsd_from_transform(tr, crs, lat)
        out["gsd"] = (gx + gy) / 2
        out["gsd_xy"] = (gx, gy)
    return out


# ==========================================================================
# DEM
# ==========================================================================
def glo30_tiles(bounds_ll):
    """GLO-30 tile URLs covering (west, south, east, north) in degrees."""
    w, s, e, n = bounds_ll
    urls = []
    for lat in range(math.floor(s), math.floor(n) + 1):
        for lon in range(math.floor(w), math.floor(e) + 1):
            la = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lo = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            urls.append(GLO30_URL.format(n=GLO30_NAME.format(lat=la, lon=lo)))
    return urls


def dem_on_grid(shape, crs, transform, dem_paths=None, verbose=True):
    """Bilinear-resample DEM(s) onto the image grid. Returns (dem, sources).
    dem_paths=None -> Copernicus GLO-30 tiles fetched for the footprint.
    Cells no DEM covers stay NaN (e.g. GLO-30 ocean tiles do not exist)."""
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds

    H, W = shape
    if not dem_paths:
        ll = transform_bounds(crs, "EPSG:4326", *_wsen(transform, H, W))
        dem_paths = glo30_tiles(ll)
    dem = np.full((H, W), np.nan, np.float32)
    used = []
    for p in dem_paths:
        try:
            with rasterio.open(_gdal_path(p)) as s:
                tmp = np.full((H, W), np.nan, np.float32)
                reproject(rasterio.band(s, 1), tmp,
                          src_nodata=s.nodata, dst_nodata=np.nan,
                          dst_transform=transform, dst_crs=crs,
                          resampling=Resampling.bilinear)
                vdat = s.tags().get("VERTICAL_DATUM") or ""
        except Exception as ex:                    # missing tile = ocean/void
            if verbose:
                print(f"[dem] skip {os.path.basename(p)}: {type(ex).__name__}")
            continue
        fill = np.isnan(dem) & np.isfinite(tmp)
        dem[fill] = tmp[fill]
        used.append(p)
        if verbose:
            print(f"[dem] {os.path.basename(p)}: {fill.mean()*100:.1f} % of grid"
                  + (f" ({vdat})" if vdat else ""))
    return dem, used


def _wsen(transform, H, W):
    xs = [transform.c, transform.c + transform.a * W]
    ys = [transform.f, transform.f + transform.e * H]
    return min(xs), min(ys), max(xs), max(ys)


# ==========================================================================
# writing
# ==========================================================================
def write_tif(path, arr, crs, transform, tags=None, desc=None):
    import rasterio
    a = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)
    H, W = a.shape
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
                crs=crs, transform=transform, nodata=NODATA, compress="lzw",
                predictor=3, tiled=True, blockxsize=256, blockysize=256)
    if H < 256 or W < 256:
        prof.update(tiled=False)
        prof.pop("blockxsize"); prof.pop("blockysize")
    with rasterio.open(path, "w", **prof) as d:
        d.write(a, 1)
        if tags:
            d.update_tags(**{k: str(v) for k, v in tags.items()})
        if desc:
            d.set_band_description(1, desc)
            d.update_tags(1, UNITS="metre")


def block_mean(a, k):
    """Mean over k x k blocks, ignoring NaN (for the 30 m double-count check)."""
    H, W = a.shape
    h, w = H // k, W // k
    if h == 0 or w == 0:
        return np.array([np.nanmean(a)])
    b = a[:h * k, :w * k].reshape(h, k, w, k)
    import warnings
    with warnings.catch_warnings():                # all-nodata blocks -> NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3))


def sanity(ndsm, dem, dsm, valid, gsd):
    """Checks that need no ground truth. Nothing here is an accuracy number."""
    v = valid & np.isfinite(ndsm)
    r = {}
    nv = ndsm[v]
    r["ndsm_percentiles_m"] = {str(p): round(float(np.percentile(nv, p)), 2)
                               for p in (50, 90, 95, 99, 99.9)} if nv.size else {}
    r["ndsm_frac_above_2m"] = round(float((nv > 2).mean()), 4) if nv.size else None
    r["ndsm_frac_above_60m"] = round(float((nv > 60).mean()), 5) if nv.size else None
    r["ndsm_plausible"] = bool(nv.size and np.percentile(nv, 99.9) < 80)
    if dem is not None:
        dv = dem[v & np.isfinite(dem)]
        r["dem_coverage"] = round(float(np.isfinite(dem[valid]).mean()), 4)
        if dv.size:
            r["dem_min_max_m"] = [round(float(dv.min()), 1), round(float(dv.max()), 1)]
            r["dem_relief_m"] = round(float(dv.max() - dv.min()), 1)
            gy, gx = np.gradient(np.where(np.isfinite(dem), dem, np.nanmean(dv)), gsd)
            r["dem_slope_deg_median"] = round(float(np.degrees(np.arctan(
                np.median(np.hypot(gx, gy)[v])))), 1)
        # size of the GLO-30 double-count: our nDSM averaged over 30 m cells is
        # roughly what a 30 m surface model has already absorbed
        k = max(1, int(round(30.0 / gsd)))
        bm = block_mean(np.where(v, ndsm, np.nan), k)
        bm = bm[np.isfinite(bm)]
        if bm.size:
            r["double_count_30m_mean_m"] = round(float(bm.mean()), 2)
            r["double_count_30m_p95_m"] = round(float(np.percentile(bm, 95)), 2)
        dsv = dsm[v & np.isfinite(dsm)]
        if dsv.size:
            r["dsm_min_max_m"] = [round(float(dsv.min()), 1), round(float(dsv.max()), 1)]
    r["note"] = ("No LiDAR/survey truth exists for this scene, so there is no "
                 "RMSE. These are plausibility checks only.")
    return r


def export_viewer(out_dir, name, rgb, surf, gsd, kind, meta_extra, max_px=1024):
    """Viewer assets. Heights are downsampled to <= max_px per side (mesh is
    1024 verts anyway); the texture keeps up to 2048 px for sharpness."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    H, W = surf.shape
    s = min(1.0, max_px / max(H, W))
    h2, w2 = max(2, int(round(H * s))), max(2, int(round(W * s)))
    fill = np.nanmin(surf) if np.isfinite(surf).any() else 0.0
    hs = P.resample_hw(np.where(np.isfinite(surf), surf, fill).astype(np.float32),
                       (h2, w2), order=1)
    g2 = gsd * W / w2
    t = min(1.0, 2048 / max(H, W))
    tex = np.clip(rgb, 0, 255).astype(np.uint8)
    if t < 1.0:
        tex = np.asarray(Image.fromarray(tex).resize((int(W * t), int(H * t)),
                                                     Image.BILINEAR))
    Image.fromarray(tex).save(os.path.join(out_dir, f"{name}_tex.jpg"),
                              quality=92, subsampling=0)
    hs.astype(np.float32).tofile(os.path.join(out_dir, f"{name}_h.bin"))
    meta = {"name": name, "width": int(w2), "height": int(h2),
            "gsd_m": round(float(g2), 4), "gsd_asserted": False,
            "extent_m": [float(w2 * g2), float(h2 * g2)],
            "h_min": float(hs.min()), "h_max": float(hs.max()),
            "h_p99": float(np.percentile(hs, 99)),
            "h_base": float(np.percentile(hs, 1)) if kind == "dsm" else 0.0,
            "source": "prediction", "kind": kind, "has_reference": False,
            "tile_id": name}
    meta.update(meta_extra)
    with open(os.path.join(out_dir, f"{name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    sp = os.path.join(out_dir, "scenes.json")
    scenes = []
    if os.path.exists(sp):
        try:
            scenes = json.load(open(sp, encoding="utf-8"))
        except Exception:
            scenes = []
    if name not in scenes:
        scenes.insert(0, name)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(scenes, f)
    print(f"[viewer] {name}: {w2}x{h2} @ {g2:.3f} m, kind={kind} -> {out_dir}")
    return meta


# ==========================================================================
# pipeline
# ==========================================================================
def get_backend(name, ckpt=None, synrs3d_dir="SynRS3D"):
    if name == "dummy":
        print("[model] DUMMY backend (darkness -> fake height). "
              "Plumbing test only: these heights mean nothing.")
        return P.dummy_backend()
    if not os.path.isdir(synrs3d_dir):
        import subprocess
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/JTRNEO/SynRS3D", synrs3d_dir], check=True)
    if not ckpt:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("JTRNEO/RS3DAda", "RS3DAda_vitl_DPT_height.pth")
    return P.RS3DAda(ckpt, synrs3d_dir=synrs3d_dir).predict_fn


def run(image, out, *, backend="rs3dada", fn=None, window=None, gsd=None,
        model_gsd=P.RS3DADA_TRAIN_GSD, tta=True, tile=P.RS3DADA_PATCH,
        dem=None, export=None, name=None, ckpt=None, synrs3d_dir="SynRS3D",
        source_note=None):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    img = read_image(image, window)
    rgb, valid = img["rgb"], img["valid"]
    H, W = rgb.shape[:2]
    if img["georef"]:
        in_gsd = img["gsd"]
        if gsd and abs(gsd - in_gsd) > 1e-3:
            print(f"[image] NOTE --gsd {gsd} overrides file GSD {in_gsd:.4f}")
            in_gsd = gsd
        print(f"[image] {W}x{H} px, CRS {img['crs'].to_string()}, "
              f"GSD {in_gsd:.4f} m (from the file's transform)")
    else:
        if not gsd:
            raise SystemExit("PNG/JPG has no georeference: pass --gsd <metres per pixel>")
        in_gsd = gsd
        print(f"[image] {W}x{H} px, NO georeference, GSD {in_gsd} m (from --gsd)")

    fn = fn or get_backend(backend, ckpt, synrs3d_dir)
    t1 = time.time()
    ndsm = P.predict_height(rgb, fn, input_gsd=in_gsd, model_gsd=model_gsd,
                            tile=tile, tta=tta, verbose=True)
    ndsm[~valid] = np.nan
    print(f"[model] nDSM done in {time.time()-t1:.0f} s")

    base_tags = {"PRODUCT": "", "MODEL": "RS3DAda (SynRS3D)" if backend != "dummy"
                 else "DUMMY - plumbing test, not a real prediction",
                 "INPUT_IMAGE": image, "INPUT_GSD_M": round(in_gsd, 4),
                 "MODEL_GSD_M": model_gsd, "TTA": tta}
    if source_note:
        base_tags["IMAGE_CREDIT"] = source_note
    name = name or os.path.splitext(os.path.basename(image.split("?")[0]))[0]
    report = {"image": image, "window": window, "size_px": [W, H],
              "gsd_m": round(in_gsd, 4), "model": base_tags["MODEL"],
              "model_gsd_m": model_gsd, "tta": tta, "image_credit": source_note}

    if img["georef"]:
        crs, tr = img["crs"], img["transform"]
        demg, used = dem_on_grid((H, W), crs, tr, dem)
        if not used:
            raise SystemExit("no DEM could be read for this footprint "
                             "(offline? pass --dem <local DEM GeoTIFF>)")
        demg[~valid] = np.nan
        dsm = demg + ndsm
        is_glo = all("copernicus" in u.lower() or "glo30" in u.lower() for u in used)
        dem_credit = GLO30_CREDIT if is_glo else ", ".join(used)
        common = dict(base_tags, VERTICAL_DATUM="EGM2008 geoid (orthometric)"
                      if is_glo else "as the supplied DEM", DEM=dem_credit)
        if is_glo:
            common["CAVEAT"] = DOUBLE_COUNT_NOTE
        write_tif(os.path.join(out, "dsm.tif"), dsm, crs, tr,
                  dict(common, PRODUCT="DSM = DEM + nDSM (m)"), "DSM elevation (m)")
        write_tif(os.path.join(out, "ndsm.tif"), ndsm, crs, tr,
                  dict(base_tags, PRODUCT="nDSM: height above ground (m)"),
                  "height above ground (m)")
        write_tif(os.path.join(out, "dem.tif"), demg, crs, tr,
                  dict(common, PRODUCT="DEM resampled to image grid (bilinear)"),
                  "DEM elevation (m)")
        report.update(kind="dsm", crs=crs.to_string(), transform=list(tr)[:6],
                      dem_sources=used, dem_credit=dem_credit,
                      vertical_datum=common["VERTICAL_DATUM"],
                      caveat=DOUBLE_COUNT_NOTE if is_glo else None,
                      sanity=sanity(ndsm, demg, dsm, valid, in_gsd),
                      files=["dsm.tif", "ndsm.tif", "dem.tif"])
        surf, kind = dsm, "dsm"
    else:
        from PIL import Image
        write_tif(os.path.join(out, "ndsm.tif"), ndsm, None,
                  _identity(in_gsd), dict(base_tags, PRODUCT="nDSM (m), no georeference"))
        v = ndsm[np.isfinite(ndsm)]
        lo, hi = (np.percentile(v, [0.5, 99.5]) if v.size else (0, 1))
        rd = np.clip((np.nan_to_num(ndsm, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
        np.save(os.path.join(out, "rdsm_0to1.npy"), rd.astype(np.float32))
        Image.fromarray((rd * 255).astype(np.uint8)).save(os.path.join(out, "rdsm_0to1.png"))
        report.update(kind="rdsm", note=("No georeference, so no DEM: ndsm.tif is "
                      "height above local ground in metres (valid only if --gsd is "
                      "right); rdsm_0to1 is the same surface scaled to 0-1."),
                      sanity=sanity(ndsm, None, None, valid, in_gsd),
                      files=["ndsm.tif", "rdsm_0to1.png", "rdsm_0to1.npy"])
        surf, kind = ndsm, "rdsm"

    if export:
        extra = {"image_credit": source_note, "model": base_tags["MODEL"]}
        if kind == "dsm":
            extra.update(dem_credit=report["dem_credit"], caveat=report["caveat"],
                         vertical_datum=report["vertical_datum"])
        report["viewer"] = export_viewer(export, name, rgb, surf, in_gsd, kind, extra)

    report["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    s = report["sanity"]
    print(f"[done] {kind.upper()} -> {out}  ({report['seconds']} s)")
    print(f"       nDSM p50/p95/p99.9 = {s['ndsm_percentiles_m'].get('50')}/"
          f"{s['ndsm_percentiles_m'].get('95')}/{s['ndsm_percentiles_m'].get('99.9')} m"
          + (f" | DEM {s['dem_min_max_m']} m | 30 m double-count ~{s.get('double_count_30m_mean_m')} m"
             if kind == "dsm" and "dem_min_max_m" in s else ""))
    write_summary(out, report)
    return report


def write_summary(out, r):
    """SUMMARY.txt: the key numbers in plain words, for people who won't open JSON."""
    s = r["sanity"]
    pc = s.get("ndsm_percentiles_m", {})
    L = [f"DepthWizard DSM run - {time.strftime('%Y-%m-%d %H:%M')}",
         f"Image      : {r['image']}",
         f"Credit     : {r.get('image_credit') or '-'}",
         f"Size       : {r['size_px'][0]} x {r['size_px'][1]} px at {r['gsd_m']} m/px "
         f"({r['size_px'][0]*r['gsd_m']:.0f} x {r['size_px'][1]*r['gsd_m']:.0f} m)",
         f"Model      : {r['model']}",
         f"Output     : {r['kind'].upper()}  ({', '.join(r['files'])})",
         "",
         "Heights above ground predicted by the model (nDSM):",
         f"  half of the area is below {pc.get('50')} m, 95 % below {pc.get('95')} m, "
         f"highest points ~{pc.get('99.9')} m",
         f"  share of area taller than 2 m : {s.get('ndsm_frac_above_2m')}",
         f"  looks plausible (99.9 % below 80 m): {'yes' if s.get('ndsm_plausible') else 'NO - check'}"]
    if r["kind"] == "dsm":
        L += ["",
              "Terrain and final DSM (metres above sea level, EGM2008):",
              f"  terrain (DEM) : {s.get('dem_min_max_m')}  relief {s.get('dem_relief_m')} m, "
              f"median slope {s.get('dem_slope_deg_median')} deg",
              f"  final DSM     : {s.get('dsm_min_max_m')}",
              f"  DEM double-count estimate: ~{s.get('double_count_30m_mean_m')} m on average "
              f"({s.get('double_count_30m_p95_m')} m in the busiest 5 %)",
              f"  terrain source: {r.get('dem_credit')}"]
    L += ["", s["note"], f"Run time: {r['seconds']} s"]
    with open(os.path.join(out, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def zip_results(zip_path, out, export, name):
    """One file to share: the GeoTIFFs, report, summary and this scene's
    viewer files."""
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out)):
            z.write(os.path.join(out, fn), os.path.join("results", fn))
        if export and os.path.isdir(export):
            for fn in sorted(os.listdir(export)):
                if fn.startswith(name + "_"):
                    z.write(os.path.join(export, fn), os.path.join("viewer", fn))
    mb = os.path.getsize(zip_path) / 1e6
    print(f"[saved] everything in ONE file: {os.path.abspath(zip_path)}  ({mb:.1f} MB)")


def _identity(gsd):
    from rasterio.transform import Affine
    return Affine(gsd, 0, 0, 0, -gsd, 0)


# ==========================================================================
# the shipped Indian sample
# ==========================================================================
SAMPLE = {
    # Maxar Open Data, event "India-Floods-Oct-2023", WorldView-2, 2022-03-07,
    # catalog id 10300100CE8D0400, ARD quadkey 120220122202. Licence CC BY-NC 4.0.
    "url": ("https://maxar-opendata.s3.amazonaws.com/events/India-Floods-Oct-2023/"
            "ard/45/120220122202/2022-03-07/10300100CE8D0400-visual.tif"),
    "window": (7676, 1276, 2048, 2048),       # Chungthang town, North Sikkim
    "rgb": "samples/chungthang_wv2.tif",
    "dem": "samples/chungthang_glo30.tif",
    "credit": ("Maxar Open Data Program, WorldView-2 2022-03-07 "
               "(catalog 10300100CE8D0400), CC BY-NC 4.0"),
}


def fetch_sample():
    """Cut the Chungthang window from Maxar's COG and the matching GLO-30
    patch, and save both under samples/ (run once; the files are committed)."""
    import rasterio
    from rasterio.windows import Window
    os.makedirs("samples", exist_ok=True)
    with rasterio.open(_gdal_path(SAMPLE["url"])) as s:
        w = Window(*SAMPLE["window"])
        a = s.read(window=w)
        prof = dict(driver="GTiff", width=a.shape[2], height=a.shape[1], count=3,
                    dtype="uint8", crs=s.crs, transform=s.window_transform(w),
                    compress="jpeg", jpeg_quality=92, photometric="ycbcr",
                    tiled=True, blockxsize=256, blockysize=256)
        crs, tr = s.crs, s.window_transform(w)
    with rasterio.open(SAMPLE["rgb"], "w", **prof) as d:
        d.write(a)
        d.update_tags(SOURCE=SAMPLE["url"], WINDOW=str(SAMPLE["window"]),
                      CREDIT=SAMPLE["credit"], LICENSE="CC BY-NC 4.0")
    # DEM: a plain crop of the native GLO-30 tile (no resampling) with a
    # margin, so dsm.py gives the same answer as when it reads the full tile
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    w_, s_, e_, n_ = transform_bounds(crs, "EPSG:4326", *_wsen(tr, a.shape[1], a.shape[2]))
    m = 0.003                                     # ~300 m margin
    (url,) = glo30_tiles((w_, s_, e_, n_))
    with rasterio.open(_gdal_path(url)) as s:
        dw = from_bounds(w_ - m, s_ - m, e_ + m, n_ + m, s.transform).round_offsets().round_lengths()
        dem = s.read(1, window=dw).astype(np.float32)
        write_tif(SAMPLE["dem"], dem, s.crs, s.window_transform(dw),
                  {"SOURCE": url, "WINDOW": str(dw), "CREDIT": GLO30_CREDIT,
                   "VERTICAL_DATUM": "EGM2008 geoid (orthometric)",
                   "NOTE": "unresampled crop of the GLO-30 tile"})
    print(f"[sample] wrote {SAMPLE['rgb']} and {SAMPLE['dem']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default=None,
                    help="GeoTIFF / PNG / JPG path or http(s) URL. "
                         "Omit to run the Chungthang sample.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", default=None, help="col,row,width,height in pixels")
    ap.add_argument("--gsd", type=float, default=None,
                    help="metres/pixel; REQUIRED for PNG/JPG, ignored-with-warning "
                         "for GeoTIFFs unless it differs")
    ap.add_argument("--dem", nargs="*", default=None,
                    help="local DEM GeoTIFF(s) instead of fetching Copernicus GLO-30")
    ap.add_argument("--backend", choices=["rs3dada", "dummy"], default="rs3dada")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--synrs3d-dir", default="SynRS3D")
    ap.add_argument("--model-gsd", type=float, default=P.RS3DADA_TRAIN_GSD)
    ap.add_argument("--tile", type=int, default=P.RS3DADA_PATCH,
                    help="model tile (multiple of 14); 518 if the GPU runs out of memory")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--export", default="web/assets_dsm",
                    help="viewer asset folder ('' to skip)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--credit", default=None, help="image credit written into outputs")
    ap.add_argument("--no-zip", action="store_true",
                    help="don't pack the results into <out>_results.zip")
    ap.add_argument("--fetch-sample", action="store_true",
                    help="(maintainers) re-cut samples/ from Maxar + GLO-30")
    a = ap.parse_args()

    if a.fetch_sample:
        return fetch_sample()
    image, dem, credit, name = a.image, a.dem, a.credit, a.name
    if image is None:
        image, dem = SAMPLE["rgb"], dem or [SAMPLE["dem"]]
        credit, name = credit or SAMPLE["credit"], name or "chungthang"
    window = tuple(int(v) for v in a.window.split(",")) if a.window else None
    out = a.out or os.path.join("out_dsm", name or
                                os.path.splitext(os.path.basename(image))[0])
    rep = run(image, out, backend=a.backend, window=window, gsd=a.gsd,
              model_gsd=a.model_gsd, tta=not a.no_tta, tile=a.tile, dem=dem,
              export=a.export or None, name=name, ckpt=a.ckpt,
              synrs3d_dir=a.synrs3d_dir, source_note=credit)
    print(f"[saved] plain-English numbers: {os.path.abspath(os.path.join(out, 'SUMMARY.txt'))}")
    if not a.no_zip:
        zip_results(out.rstrip("/\\") + "_results.zip", out, a.export,
                    rep.get("viewer", {}).get("name") or name or "")


if __name__ == "__main__":
    main()
