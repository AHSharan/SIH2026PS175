"""
docs/arch_thumbs.py - make the real-image thumbnails for docs/architecture_panels.svg.

Every thumbnail is cut from a real file - nothing is drawn by hand:
  GAMUS aerial RGB + LiDAR heights  web/assets/DC_40_24_*   (GAMUS test tile)
  Indian satellite image            samples/chungthang_wv2.tif
  terrain / predicted heights / DSM out_dsm/<run>/dem.tif, ndsm.tif, dsm.tif
                                    (a real dsm.py run with our DINOv3 head)

    python docs/arch_thumbs.py --run out_dsm/chungthang_wv2_best
"""
import argparse
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm                       # noqa: E402
from matplotlib.colors import LightSource       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "thumbs")
PX = 420


def save(rgb, name):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((PX, PX), Image.LANCZOS)
    img.save(os.path.join(OUT, name), quality=90)
    print("  wrote", name)


def square(a):
    h, w = a.shape[:2]
    s = min(h, w)
    return a[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def read_band(path):
    import rasterio
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dsm.py output folder (dem/ndsm/dsm.tif)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # (a) GAMUS training data: aerial photo + its LiDAR height label
    tex = np.asarray(Image.open(os.path.join(ROOT, "web", "assets", "DC_40_24_tex.jpg"))) / 255.0
    save(square(tex), "gamus_rgb.jpg")
    h = np.fromfile(os.path.join(ROOT, "web", "assets", "DC_40_24_h.bin"), np.float32).reshape(1024, 1024)
    save(cm.turbo(np.clip(h / 45.0, 0, 1))[..., :3], "gamus_lidar.jpg")

    # (a) Indian input: WorldView-2 over Chungthang, Sikkim
    import rasterio
    with rasterio.open(os.path.join(ROOT, "samples", "chungthang_wv2.tif")) as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.float32)
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99)       # display stretch only
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    save(square(rgb), "india_rgb.jpg")

    ls = LightSource(azdeg=315, altdeg=40)
    dem = read_band(os.path.join(a.run, "dem.tif"))
    ndsm = read_band(os.path.join(a.run, "ndsm.tif"))
    dsm = read_band(os.path.join(a.run, "dsm.tif"))
    print(f"  dem {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m | ndsm max {np.nanmax(ndsm):.1f} m | "
          f"dsm {np.nanmin(dsm):.0f}-{np.nanmax(dsm):.0f} m")

    # (b) terrain: Copernicus GLO-30 resampled to the image grid, hillshaded
    d = np.nan_to_num(dem, nan=np.nanmin(dem))
    save(ls.shade(d, cmap=cm.gist_earth, vert_exag=2, blend_mode="soft", dx=0.3, dy=0.3,
                  vmin=d.min() - 150, vmax=d.max() + 60)[..., :3], "india_dem.jpg")

    # (c) model output: predicted height above ground
    n = np.nan_to_num(ndsm, nan=0.0)
    save(cm.turbo(np.clip(n / 15.0, 0, 1))[..., :3], "india_ndsm.jpg")

    # (c) final DSM = DEM + nDSM, hillshaded so buildings show on the slope
    s = np.nan_to_num(dsm, nan=np.nanmin(dsm))
    save(ls.shade(s, cmap=cm.gist_earth, vert_exag=1.5, blend_mode="soft", dx=0.3, dy=0.3,
                  vmin=s.min() - 150, vmax=s.max() + 60)[..., :3], "india_dsm.jpg")


if __name__ == "__main__":
    main()
