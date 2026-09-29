"""
docs/make_demo_folder.py - build demo/ : every file the demo video uploads or
shows, one sub-folder per part of PPT Details/video_script.md.

  1_geotiff_with_lidar/   Park City NAIP GeoTIFF + its USGS 3DEP LiDAR reference
  2_png_jpg_tiff/         Chungthang as plain PNG, JPG and TIFF (no map information)
  3_gamus_png_with_lidar/ GAMUS test tiles as PNG + their LiDAR heights as TIFF
                          (the same tiles as the --demo-lidar scenes)
  4_geotiff_india/        Chungthang WorldView-2 GeoTIFF (no LiDAR exists for it)
  5_results/              the charts for the end of the video

    python docs/make_demo_folder.py --gamus <GAMUS test folder>
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "demo")


def cp(src, dst_dir, name=None):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name or os.path.basename(src))
    shutil.copyfile(src, dst)
    return dst


def plain_versions(src_png, d):
    """The same pixels as JPG and as a TIFF WITHOUT map information."""
    import rasterio
    from PIL import Image
    os.makedirs(d, exist_ok=True)
    rgb = np.asarray(Image.open(src_png).convert("RGB"))
    Image.fromarray(rgb).save(os.path.join(d, "chungthang_plain.jpg"), quality=92)
    prof = dict(driver="GTiff", width=rgb.shape[1], height=rgb.shape[0], count=3,
                dtype="uint8", compress="deflate")          # no crs, no transform
    with rasterio.open(os.path.join(d, "chungthang_plain.tif"), "w", **prof) as s:
        s.write(np.transpose(rgb, (2, 0, 1)))


def gamus_pairs(gamus_root, d):
    """PNG image + float32 LiDAR height TIFF (no map info, nodata -9999) per tile."""
    import rasterio
    from PIL import Image
    import data as D
    os.makedirs(d, exist_ok=True)
    scenes = json.load(open(os.path.join(ROOT, "web", "assets_lidar", "scenes.json")))
    recs = {r["tile_id"]: r for r in D.pair_dirs(
        os.path.join(gamus_root, "images"), os.path.join(gamus_root, "heights"),
        os.path.join(gamus_root, "classes"), "*.h5", "*_AGL.h5", "*_CLS.h5")}
    out = []
    for sc in scenes:
        land, tid = sc.split("_", 1)
        t = D.load_tile(recs[tid])
        Image.fromarray((np.clip(t.rgb, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(d, f"{land}_{tid}_image.png"))
        h = np.where(t.mask & np.isfinite(t.height), t.height, -9999).astype(np.float32)
        prof = dict(driver="GTiff", width=h.shape[1], height=h.shape[0], count=1,
                    dtype="float32", nodata=-9999, compress="deflate", predictor=3)
        with rasterio.open(os.path.join(d, f"{land}_{tid}_lidar_heights.tif"), "w", **prof) as s:
            s.write(h, 1)
            s.update_tags(PRODUCT="GAMUS AGL: airborne-LiDAR height above ground (m), 0.3 m pixels",
                          SOURCE="GAMUS test split (CC BY 4.0), tile " + tid)
        out.append((land, tid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamus", required=True, help="GAMUS test folder (images/ heights/ classes/)")
    a = ap.parse_args()
    s = os.path.join(ROOT, "samples")
    cp(os.path.join(s, "park_city_naip2021.tif"), os.path.join(OUT, "1_geotiff_with_lidar"))
    cp(os.path.join(s, "park_city_lidar_dsm_2m.tif"), os.path.join(OUT, "1_geotiff_with_lidar"),
       "park_city_lidar_reference.tif")
    d2 = os.path.join(OUT, "2_png_jpg_tiff")
    cp(os.path.join(s, "chungthang_plain.png"), d2)
    plain_versions(os.path.join(s, "chungthang_plain.png"), d2)
    tiles = gamus_pairs(a.gamus, os.path.join(OUT, "3_gamus_png_with_lidar"))
    cp(os.path.join(s, "chungthang_wv2.tif"), os.path.join(OUT, "4_geotiff_india"))
    p = os.path.join(ROOT, "PPT Details")
    for f in ["charts/performance_comparison.png", "lidar/lidar_sensing_overview.png",
              "lidar/lidar_truth_demo_examples.png"]:
        cp(os.path.join(p, f), os.path.join(OUT, "5_results"))
    print("demo/ ready:", ", ".join(f"{l} {t}" for l, t in tiles))


if __name__ == "__main__":
    main()
