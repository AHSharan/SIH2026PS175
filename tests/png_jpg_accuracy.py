"""
tests/png_jpg_accuracy.py - is the PNG/JPG upload path as accurate as the
original data? Measured, not assumed.

Takes GAMUS test tiles WITH airborne-LiDAR ground truth and feeds the same
pixels to the same trained model four ways:
  direct  : the image array straight into predict_height (how RESULTS.md was scored)
  png     : saved as PNG (lossless), run through dsm.run - the upload path
  jpg90   : saved as JPG quality 90 (a typical export), through dsm.run
  jpg75   : saved as JPG quality 75 (heavier compression), through dsm.run
then scores every variant against the LiDAR heights with eval.py.

    python tests/png_jpg_accuracy.py --tiles <gamus test dir> --head models/best.pt
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import data as D            # noqa: E402
import dsm as M             # noqa: E402
import eval as E            # noqa: E402
import predict as P         # noqa: E402
import train_head as TH     # noqa: E402

GSD = 0.3                   # GAMUS pixel size (DFC2019 spec; see data.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="folder with images/ heights/ classes/")
    ap.add_argument("--head", required=True, help="trained head checkpoint (best.pt)")
    ap.add_argument("--work", default=os.path.join(ROOT, "dw_run", "png_jpg_test"))
    ap.add_argument("--report", default=os.path.join(ROOT, "results", "png_jpg_accuracy.md"))
    a = ap.parse_args()

    recs = D.pair_dirs(os.path.join(a.tiles, "images"), os.path.join(a.tiles, "heights"),
                       os.path.join(a.tiles, "classes"), "*.h5", "*_AGL.h5", "*_CLS.h5")
    print(f"{len(recs)} test tiles with LiDAR ground truth")
    fn = M.get_dinov3(a.head)
    variants = ["direct", "png", "jpg90", "jpg75"]
    dirs = {v: os.path.join(a.work, "pred_" + v) for v in variants}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    tmp = os.path.join(a.work, "files")
    os.makedirs(tmp, exist_ok=True)
    secs = {v: [] for v in variants}

    for i, r in enumerate(recs, 1):
        tid = r["tile_id"]
        rgb = TH.load_rgb_255(r["rgb"])                     # float 0-255, as in training eval
        u8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)

        t0 = time.time()
        pred = P.predict_height(rgb, fn, input_gsd=GSD, model_gsd=0.5, tile=608,
                                overlap=0.25, tta=True, gsd_normalise=True, patch_multiple=16)
        secs["direct"].append(time.time() - t0)
        np.save(os.path.join(dirs["direct"], f"{tid}_pred.npy"), pred)

        for v, ext, kw in (("png", "png", {}), ("jpg90", "jpg", {"quality": 90}),
                           ("jpg75", "jpg", {"quality": 75})):
            path = os.path.join(tmp, f"{tid}_{v}.{ext}")
            Image.fromarray(u8).save(path, **kw)
            out = os.path.join(a.work, "runs", f"{tid}_{v}")
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                M.run(path, out, backend="dinov3", fn=fn, gsd=GSD, tta=True,
                      buildings=False, export=None, name=f"{tid}_{v}")
            secs[v].append(time.time() - t0)
            import rasterio
            with rasterio.open(os.path.join(out, "ndsm.tif")) as src:
                nd = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    nd[nd == src.nodata] = np.nan
            if nd.shape != pred.shape:
                raise SystemExit(f"{v}: ndsm.tif is {nd.shape}, expected {pred.shape}")
            np.save(os.path.join(dirs[v], f"{tid}_pred.npy"), nd)
        print(f"  {i}/{len(recs)} {tid}", flush=True)

    buckets = E.auto_buckets_from_cls(os.path.join(a.tiles, "classes"))
    gt = os.path.join(a.tiles, "heights")
    res = {}
    for v in variants:
        with contextlib.redirect_stdout(io.StringIO()):
            res[v] = E.evaluate(dirs[v], gt, buckets, "*", "*_AGL.h5")["summary"]

    # how far each file-based variant moves from the direct prediction, per pixel
    diffs = {}
    for v in variants[1:]:
        d = []
        for r in recs:
            x = np.load(os.path.join(dirs["direct"], f"{r['tile_id']}_pred.npy"))
            y = np.load(os.path.join(dirs[v], f"{r['tile_id']}_pred.npy"))
            m = np.isfinite(x) & np.isfinite(y)
            d.append(np.abs(x[m] - y[m]))
        d = np.concatenate(d)
        diffs[v] = (float(d.mean()), float(np.percentile(d, 99)))

    names = {"direct": "original data (direct)", "png": "PNG via upload path",
             "jpg90": "JPG q90 via upload path", "jpg75": "JPG q75 via upload path"}
    L = ["# PNG / JPG accuracy check", "",
         f"{len(recs)} GAMUS test tiles with airborne-LiDAR ground truth, model "
         f"`{os.path.basename(a.head)}`, pixel size {GSD} m given. Same pixels, four ways "
         "in. All numbers measured by `tests/png_jpg_accuracy.py`.", "",
         "| input | RMSE (m) | MAE (m) | bias (m) | r | forest RMSE (m) | "
         "mean change vs original (m) | 99th pct change (m) | s / tile |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v in variants:
        o, fo = res[v]["overall"], res[v].get("forest", {})
        ch = ("-", "-") if v == "direct" else (f"{diffs[v][0]:.3f}", f"{diffs[v][1]:.2f}")
        L.append(f"| {names[v]} | {o['rmse_m']:.2f} | {o['mae_m']:.2f} | {o['bias_m']:+.2f} | "
                 f"{o['pearson_r']:.3f} | {fo.get('rmse_m', float('nan')):.2f} | {ch[0]} | "
                 f"{ch[1]} | {np.mean(secs[v]):.1f} |")
    L += ["", "Per landscape (RMSE, m):", "",
          "| input | urban | sparse | forest |", "|---|---|---|---|"]
    for v in variants:
        g = lambda b: res[v].get(b, {}).get("rmse_m", float("nan"))   # noqa: E731
        L.append(f"| {names[v]} | {g('urban'):.2f} | {g('sparse'):.2f} | {g('forest'):.2f} |")
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwritten -> {a.report}")


if __name__ == "__main__":
    main()
