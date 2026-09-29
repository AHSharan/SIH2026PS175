"""
export_viewer.py - turn a tile into web-viewer assets.

Writes, per scene:
  <name>_tex.jpg    RGB texture (identity UV -> ortho-exact projection)
  <name>_h.bin      float32 heights, row-major, H*W*4 bytes
  <name>_ref.bin    float32 reference heights (optional, for the error overlay)
  <name>_meta.json  dims, gsd, height range, metrics if a reference exists

Float32 binary rather than an encoded PNG: exact metres, no quantisation, and
the viewer can report true heights when the user clicks.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

import data as D


def pick_interesting(recs, n=1, prefer="urban"):
    """Rank tiles by how much structure they show - a flat field is a bad demo."""
    scored = []
    for rec in recs:
        t = D.load_tile(rec)
        hv = t.height[t.mask]
        if hv.size == 0:
            continue
        b = float((t.cls == D.CLS_BUILDING).mean()) if t.cls is not None else 0.0
        score = float(np.percentile(hv, 99)) * (1.0 + 2.0 * b)
        if prefer == "urban":
            score *= (1.0 + 3.0 * b)
        scored.append((score, rec, t))
    scored.sort(key=lambda x: -x[0])
    return [(r, t) for _, r, t in scored[:n]]


def export(tile, out_dir, name, pred=None, gsd=None):
    os.makedirs(out_dir, exist_ok=True)
    H, W = tile.height.shape
    gsd = gsd or tile.gsd

    # texture
    tex = (np.clip(tile.rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(tex).save(os.path.join(out_dir, f"{name}_tex.jpg"),
                              quality=92, subsampling=0)

    # heights: the surface the viewer displaces
    surf = pred if pred is not None else tile.height
    surf = np.where(np.isfinite(surf), surf, 0.0).astype(np.float32)
    surf.tofile(os.path.join(out_dir, f"{name}_h.bin"))

    meta = {
        "name": name, "width": int(W), "height": int(H),
        "gsd_m": float(gsd), "gsd_asserted": bool(tile.meta.get("gsd_asserted")),
        "extent_m": [float(W * gsd), float(H * gsd)],
        "h_min": float(np.nanmin(surf)), "h_max": float(np.nanmax(surf)),
        "h_p99": float(np.nanpercentile(surf, 99)),
        "source": "prediction" if pred is not None else "ground_truth",
        "tile_id": tile.tile_id,
    }

    # reference + honest metrics, only when a real prediction exists
    if pred is not None:
        ref = np.where(np.isfinite(tile.height), tile.height, 0.0).astype(np.float32)
        ref.tofile(os.path.join(out_dir, f"{name}_ref.bin"))
        m = tile.mask & np.isfinite(pred) & np.isfinite(tile.height)
        if m.any():
            e = (pred[m] - tile.height[m]).astype(np.float64)
            meta["metrics"] = {
                "rmse_m": float(np.sqrt((e ** 2).mean())),
                "mae_m": float(np.abs(e).mean()),
                "bias_m": float(e.mean()),
                "n_px": int(m.sum()),
            }
        meta["has_reference"] = True
    else:
        meta["has_reference"] = False

    with open(os.path.join(out_dir, f"{name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] {name}: {W}x{H} @ {gsd} m  "
          f"({meta['extent_m'][0]:.0f}x{meta['extent_m'][1]:.0f} m)  "
          f"h=[{meta['h_min']:.1f},{meta['h_max']:.1f}] source={meta['source']}")
    return meta


def _half(a):
    """2 x 2 block mean (NaN-aware): 1024 px GAMUS tile -> 512 px for the demo."""
    H, W = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    b = a[:H, :W].reshape(H // 2, 2, W // 2, 2)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def build_lidar_demo(root, pred_dir, out, model_label):
    """Assets for the "real image vs LiDAR truth" demo (serve.py --demo-lidar).

    One urban, one sparse and one forest GAMUS test tile (the landscape comes
    from the tile's class map, data.bucket_of). In each landscape the tile
    with the MEDIAN per-tile RMSE is used: a typical case, not a picked one. Each scene holds the real
    image, the LiDAR truth (GAMUS AGL = airborne-LiDAR height above ground)
    and our model's prediction for the same pixels. Metrics are computed at
    FULL resolution before the 2 x downsample that keeps the demo small.
    """
    import eval as E
    recs = D.pair_dirs(os.path.join(root, "images"), os.path.join(root, "heights"),
                       os.path.join(root, "classes"), "*.h5", "*_AGL.h5", "*_CLS.h5")
    have = [r for r in recs if os.path.exists(os.path.join(pred_dir, f"{r['tile_id']}_pred.npy"))]
    print(f"[demo] {len(have)} test tiles have a prediction in {pred_dir}")
    by_land = {}
    for r in have:
        t = D.load_tile(r)
        land = D.bucket_of(t)
        if land not in ("urban", "sparse", "forest"):
            continue
        pred = np.load(os.path.join(pred_dir, f"{t.tile_id}_pred.npy"))
        m = t.mask & np.isfinite(pred)
        rmse = float(np.sqrt(np.mean((pred[m] - t.height[m]) ** 2)))
        by_land.setdefault(land, []).append((rmse, r))
    best = {}
    for land, lst in by_land.items():
        lst.sort(key=lambda x: x[0])
        best[land] = lst[(len(lst) - 1) // 2]                         # median tile
        print(f"[demo] {land}: {len(lst)} tiles, per-tile RMSE {lst[0][0]:.2f}-{lst[-1][0]:.2f} m, "
              f"median tile {best[land][1]['tile_id']} ({best[land][0]:.2f} m)")
    os.makedirs(out, exist_ok=True)
    names = []
    for land in ("urban", "sparse", "forest"):
        if land not in best:
            continue
        t = D.load_tile(best[land][1])
        pred = np.load(os.path.join(pred_dir, f"{t.tile_id}_pred.npy")).astype(np.float32)
        m = t.mask & np.isfinite(pred) & np.isfinite(t.height)
        met = E.metrics_from((pred[m] - t.height[m]).astype(np.float64),
                             t.height[m].astype(np.float64))
        name = f"{land}_{t.tile_id}"
        tex = (np.clip(t.rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(tex).save(os.path.join(out, f"{name}_tex.jpg"), quality=90)
        ph = np.where(np.isfinite(_half(pred)), _half(pred), 0).astype(np.float32)
        rh = _half(np.where(t.mask, t.height, np.nan))
        rh = np.where(np.isfinite(rh), rh, ph).astype(np.float32)   # no-data: show the model
        ph.tofile(os.path.join(out, f"{name}_h.bin"))
        rh.tofile(os.path.join(out, f"{name}_ref.bin"))
        h2, w2 = ph.shape
        g2 = t.gsd * t.height.shape[1] / w2
        meta = {"name": name, "width": w2, "height": h2, "gsd_m": g2, "gsd_asserted": True,
                "extent_m": [w2 * g2, h2 * g2], "h_min": float(min(ph.min(), rh.min())),
                "h_max": float(max(ph.max(), rh.max())),
                "h_p99": float(np.percentile(np.concatenate([ph.ravel(), rh.ravel()]), 99)),
                "source": "prediction", "kind": "ndsm", "has_reference": True,
                "tile_id": t.tile_id, "vex_default": 2.0, "reference_name": "LiDAR truth",
                "image_credit": f"GAMUS test tile {t.tile_id} (CC BY 4.0)", "model": model_label,
                "metrics": {"rmse_m": met["rmse_m"], "mae_m": met["mae_m"],
                            "bias_m": met["bias_m"], "r": met["pearson_r"], "n_px": met["n_px"]},
                "demo": {"landscape": land,
                         "truth": "GAMUS AGL: airborne-LiDAR height above ground, 0.3 m "
                                  "(DFC2019 / US3D), shown at 0.6 m",
                         "metrics_at": "full 0.3 m resolution, all valid LiDAR pixels",
                         "chosen": f"median per-tile RMSE of the {len(by_land[land])} {land} "
                                   "test tiles (typical case, not cherry-picked)"}}
        with open(os.path.join(out, f"{name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        names.append(name)
        print(f"[demo] {name}: {land}, RMSE {met['rmse_m']:.2f} m, MAE {met['mae_m']:.2f} m, "
              f"r {met['pearson_r']:.3f}")
    with open(os.path.join(out, "scenes.json"), "w", encoding="utf-8") as f:
        json.dump(names, f)
    print(f"[demo] wrote {len(names)} scenes -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="gamus/test")
    ap.add_argument("--out", default="web/assets")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pred-dir", default=None,
                    help="dir of *_pred.npy; when given, the mesh shows the "
                         "PREDICTION and the reference enables the error overlay")
    ap.add_argument("--lidar-demo", action="store_true",
                    help="build the real-image vs LiDAR-truth demo scenes (needs --pred-dir); "
                         "default --out becomes web/assets_lidar")
    ap.add_argument("--model-label", default="DINOv3-SAT head (best.pt, 480 GAMUS train tiles)")
    a = ap.parse_args()
    if a.lidar_demo:
        if not a.pred_dir:
            ap.error("--lidar-demo needs --pred-dir (predictions as <tile>_pred.npy)")
        out = a.out if a.out != "web/assets" else "web/assets_lidar"
        return build_lidar_demo(a.root, a.pred_dir, out, a.model_label)

    recs = D.pair_dirs(os.path.join(a.root, "images"),
                       os.path.join(a.root, "heights"),
                       os.path.join(a.root, "classes"),
                       "*.h5", "*_AGL.h5", "*_CLS.h5")
    print(f"[export] {len(recs)} paired tiles")

    scenes = []
    for rec, tile in pick_interesting(recs, a.n):
        pred = None
        if a.pred_dir:
            p = os.path.join(a.pred_dir, f"{tile.tile_id}_pred.npy")
            if os.path.exists(p):
                pred = np.load(p).astype(np.float32)
            else:
                print(f"[export] no prediction for {tile.tile_id}, using GT")
        scenes.append(export(tile, a.out, tile.tile_id, pred=pred))

    with open(os.path.join(a.out, "scenes.json"), "w", encoding="utf-8") as f:
        json.dump([s["name"] for s in scenes], f)
    print(f"[export] wrote {len(scenes)} scenes -> {a.out}")


if __name__ == "__main__":
    main()
