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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="gamus/test")
    ap.add_argument("--out", default="web/assets")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pred-dir", default=None,
                    help="dir of *_pred.npy; when given, the mesh shows the "
                         "PREDICTION and the reference enables the error overlay")
    a = ap.parse_args()

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
