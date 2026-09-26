"""
CPU tests for train_head.py. Uses FakeEncoder (same interface as the real
DINOv3 encoder) and REAL GAMUS tiles from gamus/test (fetch with data.py first).

Run from the repo root:  python tests/test_train_head_cpu.py
What this cannot test: the real DINOv3 download/forward (gated, GPU-sized)
and Colab specifics. colab_train.py logs self-checks for those at runtime.
"""
import glob
import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import data as D            # noqa: E402
import predict as P         # noqa: E402
import train_head as TH     # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def main():
    gt_dir = os.path.join(ROOT, "gamus", "test")
    recs = D.pair_dirs(os.path.join(gt_dir, "images"), os.path.join(gt_dir, "heights"),
                       os.path.join(gt_dir, "classes"), "*.h5", "*_AGL.h5", "*_CLS.h5")
    if len(recs) < 10:
        raise SystemExit("need >=10 local GAMUS tiles: python data.py --fetch 12")
    tmp = tempfile.mkdtemp(prefix="dw_test_")
    print(f"tmp dir: {tmp}")

    print("\n1. grid arithmetic matches predict.predict_height")
    hw = TH.work_size((1024, 1024))
    buf = io.StringIO()
    with redirect_stdout(buf):
        P.predict_height(np.zeros((1024, 1024, 3), np.float32),
                         lambda c: np.zeros(c.shape[1:], np.float32),
                         input_gsd=0.3, model_gsd=0.5, tile=608, tta=False,
                         patch_multiple=16, verbose=True)
    check("work_size(1024) == (608, 608)", hw == (608, 608), str(hw))
    check("predict_height resamples to the same 608x608 grid", "608x608" in buf.getvalue(),
          buf.getvalue().strip().splitlines()[0])

    print("\n2. real-size head (1024-dim DINOv3 features)")
    head = TH.HeightHead(1024, 256)
    n = sum(p.numel() for p in head.parameters())
    with torch.no_grad():
        out = head(torch.randn(1, 4, 1024, 38, 38, dtype=torch.float16), (608, 608))
    check("output shape (1, 608, 608)", tuple(out.shape) == (1, 608, 608), str(tuple(out.shape)))
    check("head is small (< 15M params)", n < 15e6, f"{n / 1e6:.2f}M")

    print("\n3. rot90/flip keep pixel<->patch correspondence (the augmentation claim)")
    h = w = 38
    f = torch.zeros(4, 2, h, w, dtype=torch.float16)
    ii, jj = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    f[:, 0], f[:, 1] = ii.half(), jj.half()
    py, px = torch.meshgrid(torch.arange(608), torch.arange(608), indexing="ij")
    t = ((py // 16) * 1000 + (px // 16)).float()        # pixel encodes its patch
    ok_all = True
    for k in range(4):
        for flip in (False, True):
            ff, tt = torch.rot90(f, k, dims=(-2, -1)), torch.rot90(t, k, dims=(-2, -1))
            if flip:
                ff, tt = torch.flip(ff, dims=(-1,)), torch.flip(tt, dims=(-1,))
            # the patch that now sits at grid (a,b) must own every pixel of block (a,b)
            enc = (ff[0, 0].float() * 1000 + ff[0, 1].float())
            blk = tt.reshape(38, 16, 38, 16).permute(0, 2, 1, 3).reshape(38, 38, 256)
            ok_all &= bool((blk == enc[..., None]).all())
    check("all 8 rot/flip combos keep alignment", ok_all)

    print("\n4. feature cache: build, resume, corruption recovery")
    enc = TH.FakeEncoder(hidden=32)
    tr_dir, va_dir = os.path.join(tmp, "cache_train"), os.path.join(tmp, "cache_val")
    tr_recs, va_recs, te_recs = recs[:6], recs[6:9], recs[9:12]
    made = [TH.cache_tile(r, enc, tr_dir) for r in tr_recs]
    made += [TH.cache_tile(r, enc, va_dir) for r in va_recs]
    again = [TH.cache_tile(r, enc, tr_dir) for r in tr_recs]
    check("first pass caches every tile", all(made), f"{sum(made)} new")
    check("second pass skips everything (resume-safe)", not any(again))
    fz = np.load(TH.cache_paths(tr_dir, tr_recs[0]["tile_id"])[0])
    tz = np.load(TH.cache_paths(tr_dir, tr_recs[0]["tile_id"])[1])
    check("feature shape (4, C, 38, 38) fp16", fz.shape == (4, 32, 38, 38) and fz.dtype == np.float16,
          f"{fz.shape} {fz.dtype}")
    check("target 608x608 with real heights", tz.shape == (608, 608) and np.nanmax(tz) > 5,
          f"max {np.nanmax(tz):.1f} m")
    bad_fp = TH.cache_paths(tr_dir, tr_recs[1]["tile_id"])[0]
    with open(bad_fp, "wb") as fh:
        fh.write(b"not a numpy file")
    open(os.path.join(tr_dir, "junk_f.npy.tmp"), "wb").close()
    n_good = TH.verify_cache(tr_dir, log=lambda *a: None)
    check("corrupt entry + stray .tmp removed", n_good == 5 and not os.path.exists(bad_fp),
          f"{n_good} good")
    TH.cache_tile(tr_recs[1], enc, tr_dir)                  # heal it
    check("re-cache heals the removed tile", TH.verify_cache(tr_dir, log=lambda *a: None) == 6)

    print("\n5. training: loss drops, checkpoints written, resume works")
    ck = os.path.join(tmp, "ckpt")
    logs = []
    best_p, hist = TH.train(tr_dir, va_dir, ck, epochs=3, batch=2, lr=1e-3, head_dim=32,
                            num_workers=0, device="cpu", log=logs.append)
    check("3 epochs recorded", len(hist) == 3)
    check("training loss decreased", hist[-1]["train_loss"] < hist[0]["train_loss"],
          f"{hist[0]['train_loss']:.3f} -> {hist[-1]['train_loss']:.3f}")
    check("best.pt / last.pt / history.json exist",
          all(os.path.exists(os.path.join(ck, f)) for f in ("best.pt", "last.pt", "history.json")))
    logs2 = []
    _, hist2 = TH.train(tr_dir, va_dir, ck, epochs=5, batch=2, lr=1e-3, head_dim=32,
                        num_workers=0, device="cpu", log=logs2.append)
    resumed = any("RESUMED at epoch 4/5" in m for m in logs2)
    check("second call RESUMES at epoch 4 (not from scratch)", resumed and len(hist2) == 5,
          next((m for m in logs2 if "RESUMED" in m), "no resume line"))
    logs3 = []
    TH.train(tr_dir, va_dir, ck, epochs=5, batch=2, head_dim=32, num_workers=0,
             device="cpu", log=logs3.append)
    check("finished run is a no-op", any("already finished" in m for m in logs3))
    logs4 = []
    TH.train(tr_dir, va_dir, ck, epochs=1, batch=2, head_dim=48, num_workers=0,
             device="cpu", log=logs4.append)
    check("dimension-mismatched checkpoint is set aside, not crashed on",
          any("starting fresh" in m for m in logs4) and os.path.exists(os.path.join(ck, "last.pt.stale")))

    print("\n6. inference through predict.predict_height on held-out tiles")
    head, meta = TH.load_head(best_p, device="cpu")
    fn = TH.make_predict_fn(enc, head)
    import eval as E
    pred_dir = os.path.join(tmp, "preds")
    os.makedirs(pred_dir)
    for r in te_recs:
        rgb = TH.load_rgb_255(r["rgb"])
        pred = P.predict_height(rgb, fn, input_gsd=0.3, model_gsd=0.5, tile=608,
                                overlap=0.25, tta=True, gsd_normalise=True, patch_multiple=16)
        np.save(os.path.join(pred_dir, f"{r['tile_id']}_pred.npy"), pred)
    check("prediction on the original 1024 grid, finite, >= 0",
          pred.shape == (1024, 1024) and np.isfinite(pred).all() and pred.min() >= 0,
          f"{pred.shape} range [{pred.min():.2f}, {pred.max():.2f}]")
    with redirect_stdout(io.StringIO()):
        res = E.evaluate(pred_dir, os.path.join(gt_dir, "heights"), {}, "*", "*_AGL.h5")
    check("eval.py scores the head's predictions", res["n_tiles"] == 3,
          f"RMSE {res['summary']['overall']['rmse_m']:.2f} m (fake encoder - plumbing only)")

    shutil.rmtree(tmp, ignore_errors=True)
    n_pass = sum(ok for _, ok in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} passed")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
