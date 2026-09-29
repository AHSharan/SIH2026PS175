"""
docs/lidar_sensing_figure.py - one slide explaining how LiDAR truth is made
and how our output is scored, drawn from REAL data of one benchmark site.

  1 photo (USDA NAIP)  ->  2 laser returns (USGS 3DEP point cloud, a strip
  along one transect)  ->  3 LiDAR truth grid (height above ground, 2 m)
  ->  4 our model's height from the photo alone  ->  5 error map,
  plus the elevation profile along the same transect.

Needs the work folder of tests/lidar_benchmark.py (naip.tif, grid2m.npz) and
internet for the point-cloud strip (Microsoft Planetary Computer, no login).

    python docs/lidar_sensing_figure.py --work %TEMP%/lb --site park_city_ut
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

SITES = {"park_city_ut": (-111.498, 40.646, "Park City, Utah (hillside, houses, trees)")}
HALF, GRID, STRIP = 300.0, 2.0, 3.0       # site half-width, grid, strip half-width (m)


def strip_points(lon, lat, crs, x0, y1, row):
    """3DEP points within +-STRIP m of grid row `row` (a west-east transect)."""
    import laspy
    from laspy.copc import Bounds
    from pyproj import CRS, Transformer
    import lidar_benchmark as LB
    cat = LB.stac()
    bbox = [lon - 0.0045, lat - 0.0035, lon + 0.0045, lat + 0.0035]
    items, yr = LB.lidar_items(cat, "3dep-lidar-copc", bbox)
    ncrs = CRS.from_user_input(crs.to_wkt())
    yc = y1 - (row + 0.5) * GRID
    X, Y, Z, C = [], [], [], []
    for it in items:
        with laspy.CopcReader.open(it.assets["data"].href) as r:
            lcrs = r.header.parse_crs()
            lcrs = lcrs.sub_crs_list[0] if lcrs.is_compound else lcrs
            xs, ys = Transformer.from_crs(ncrs, lcrs, always_xy=True).transform(
                [x0, x0 + 2 * HALF], [yc - STRIP, yc + STRIP])
            pts = r.query(Bounds(mins=np.array([min(xs), min(ys)]),
                                 maxs=np.array([max(xs), max(ys)])))
        if not len(pts):
            continue
        px, py = Transformer.from_crs(lcrs, ncrs, always_xy=True).transform(
            np.asarray(pts.x), np.asarray(pts.y))
        k = (np.abs(py - yc) <= STRIP) & (px >= x0) & (px <= x0 + 2 * HALF)
        cls = np.asarray(pts.classification)[k]
        keep = ~np.isin(cls, (7, 18))                       # drop noise, as in the benchmark
        X.append((px[k] - x0)[keep]); Z.append(np.asarray(pts.z)[k][keep]); C.append(cls[keep])
    return np.concatenate(X), np.concatenate(Z), np.concatenate(C), yr, items[0].id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(os.environ.get("TEMP", "/tmp"), "lb"))
    ap.add_argument("--site", default="park_city_ut")
    ap.add_argument("--out", default=os.path.join(ROOT, "PPT Details", "lidar",
                                                  "lidar_sensing_overview.png"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import TwoSlopeNorm
    from pyproj import CRS, Transformer

    lon, lat, title = SITES[a.site]
    import json
    bench = json.load(open(os.path.join(ROOT, "results", "lidar_benchmark.json"), encoding="utf-8"))
    naip_year = next(b["naip_year"] for b in bench if b["site"] == a.site)
    d = os.path.join(a.work, a.site)
    z = np.load(os.path.join(d, "grid2m.npz"))
    with rasterio.open(os.path.join(d, "naip.tif")) as s:
        rgb = np.transpose(s.read([1, 2, 3]), (1, 2, 0))
        crs = s.crs
    cx, cy = Transformer.from_crs(4326, CRS.from_user_input(crs.to_wkt()),
                                  always_xy=True).transform(lon, lat)
    x0, y1 = cx - HALF, cy + HALF
    n = z["lidar_hag"].shape[0]

    # transect: the middle-half row with the most above-ground structure
    rows = np.arange(n // 4, 3 * n // 4)
    row = int(rows[np.argmax([np.nanmean(z["lidar_hag"][r]) for r in rows])])
    px, pz, pc, yr, item = strip_points(lon, lat, crs, x0, y1, row)
    print(f"[fig] transect row {row}: {px.size} laser returns from {item} ({yr})")

    hag, ours, err = z["lidar_hag"], z["ours_ndsm"], z["ours_ndsm"] - z["lidar_hag"]
    m = np.isfinite(err)
    rmse, mae = float(np.sqrt(np.mean(err[m] ** 2))), float(np.mean(np.abs(err[m])))
    r = float(np.corrcoef(ours[m], hag[m])[0, 1])
    dsm_err = z["ours_dsm"] - z["lidar_dsm"]
    md = np.isfinite(dsm_err)
    dsm_rmse = float(np.sqrt(np.mean(dsm_err[md] ** 2)))
    vmax = float(np.nanpercentile(hag, 99))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.34,
                          left=0.045, right=0.985, top=0.855, bottom=0.075)
    fig.text(0.03, 0.955, "How the LiDAR truth is made, and how our output is scored",
             fontsize=24, fontweight="bold", color="#14213d")
    fig.text(0.03, 0.915, f"{title}  ·  600 × 600 m  ·  photo: USDA NAIP {naip_year}, 0.6 m  ·  "
             f"truth: USGS 3DEP airborne LiDAR {yr}  ·  every panel is real data",
             fontsize=13, color="#4a5568")
    ext = [0, 2 * HALF, 0, 2 * HALF]
    yline = 2 * HALF - (row + 0.5) * GRID
    H1 = dict(loc="left", fontsize=15, fontweight="bold", color="#14213d", pad=22)

    def sub(ax, text):
        ax.text(0, 1.012, text, transform=ax.transAxes, fontsize=10.5, color="#4a5568", va="bottom")

    def mapax(cell, img, head, subtext, cbar=None, **kw):
        ax = fig.add_subplot(cell)
        im = ax.imshow(img, extent=ext, **kw)
        ax.axhline(yline, color="#ff2d55", lw=2, ls="--")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(head, **H1); sub(ax, subtext)
        if cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(cbar, fontsize=10)
        return ax

    # 1 photo
    mapax(gs[0, 0], rgb, "1  Real image", "the model's only input · red line = transect")

    # 2 sensing: laser returns along the transect, rays from the scanner
    ax = fig.add_subplot(gs[0, 1:3])
    g = pc == 2
    top = np.nanmax(pz) + 22
    ac = 2 * HALF * 0.5
    edges = np.linspace(0, 2 * HALF, 28)
    for lo_, hi_ in zip(edges[:-1], edges[1:]):          # first return per bin = highest point
        k = (px >= lo_) & (px < hi_)
        if k.any():
            j = np.argmax(np.where(k, pz, -np.inf))
            ax.plot([ac, px[j]], [top, pz[j]], color="#f4a261", lw=0.8, alpha=0.8)
    ax.plot(ac, top, marker="v", ms=16, color="#14213d")
    ax.text(ac + 8, top + 2, "airborne laser scanner", fontsize=10.5, color="#14213d")
    ax.scatter(px[~g], pz[~g], s=1.2, c="#2a9d8f", label="returns from roofs and trees")
    ax.scatter(px[g], pz[g], s=1.2, c="#8d6e63", label="ground returns (class 2)")
    ax.set_xlim(0, 2 * HALF); ax.set_ylim(np.nanmin(pz) - 5, top + 12)
    ax.set_title("2  LiDAR sensing: every laser pulse returns a 3D point", **H1)
    sub(ax, f"{px.size:,} real returns in a {2 * STRIP:.0f} m strip along the red line · "
            "per 2 m cell: surface = highest return, ground = ground returns")
    ax.set_xlabel("distance west to east (m)"); ax.set_ylabel("elevation (m)")
    ax.legend(loc="upper left", markerscale=8, fontsize=10, frameon=False)

    # 3 truth, 4 ours, 5 error
    mapax(gs[0, 3], hag, "3  LiDAR truth", "height above ground = surface − ground",
          "m above ground", cmap="turbo", vmin=0, vmax=vmax)
    mapax(gs[1, 0], ours, "4  Our model", "height above ground, from the photo alone",
          "m above ground", cmap="turbo", vmin=0, vmax=vmax)
    mapax(gs[1, 1], err, "5  Error = model − LiDAR",
          f"RMSE {rmse:.2f} m · MAE {mae:.2f} m · r {r:.2f} (whole site)",
          "m", cmap="RdBu_r", norm=TwoSlopeNorm(0, -8, 8))

    # 6 profile along the same line
    ax = fig.add_subplot(gs[1, 2:])
    xs = (np.arange(n) + 0.5) * GRID
    ax.fill_between(xs, z["lidar_dtm"][row], z["lidar_dsm"][row], color="#2a9d8f", alpha=0.25,
                    label="LiDAR: objects above ground")
    ax.plot(xs, z["lidar_dtm"][row], color="#8d6e63", lw=2, label="LiDAR ground")
    ax.plot(xs, z["lidar_dsm"][row], color="#1d3557", lw=2, label="LiDAR surface = truth DSM")
    ax.plot(xs, z["ours_dsm"][row], color="#e63946", lw=2, label="our DSM = Copernicus DEM + model")
    ax.set_xlim(0, 2 * HALF)
    ax.set_title("6  Full DSM along the red line", **H1)
    sub(ax, f"whole-site full-DSM RMSE {dsm_rmse:.2f} m against the LiDAR surface")
    ax.set_xlabel("distance west to east (m)"); ax.set_ylabel("elevation (m)")
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    fig.text(0.03, 0.018, "Same idea as our training data: GAMUS heights are airborne-LiDAR "
             "height above ground on a grid co-registered with the orthophoto. Metrics: "
             "tests/lidar_benchmark.py (all pixels of this site, 2 m grid). LiDAR datum NAVD88, "
             "Copernicus EGM2008.", fontsize=10, color="#4a5568")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=100)
    fig.savefig(a.out.replace(".png", ".svg"))
    print(f"[fig] wrote {a.out}  (nDSM RMSE {rmse:.2f}, MAE {mae:.2f}, r {r:.2f}; DSM RMSE {dsm_rmse:.2f})")


if __name__ == "__main__":
    main()
