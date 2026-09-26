"""
colab_train.py - DINOv3-SAT height head: cache -> train -> evaluate. Resumable.

Step-by-step instructions for whoever runs this are in COLAB.md. In short, in
Google Colab with a T4 GPU:
    cell 1 (every session):  mount Drive, pip install, git clone
    cell 2:                  %run /content/code/colab_train.py
If Colab disconnects: run cell 1, then cell 2 again. It RESUMES, it does not
start over.

Where things go
    Google Drive  MyDrive/depthwizard/     checkpoints, predictions, RESULTS.md,
                                           viewer_assets/, run_log.txt
                                           -> survive disconnects
    /content/dw_local/                     raw tiles + feature cache
                                           -> fast; rebuilt automatically (~10 min)

What it produces (all measured, nothing estimated)
    - the DINOv3-SAT head, trained on GAMUS train, early-stopped on GAMUS val
    - test metrics on the SAME 40 tiles as the Kaggle RS3DAda run (verified)
    - the predict-zero floor and RS3DAda zero-shot on those same tiles
    - a GSD robustness sweep (0.3 / 0.5 / 0.75 / 1.0 m)
    - viewer assets with real predictions for the 3D flythrough screenshots

Environment overrides - for testing only, the person running this never needs them:
    DW_N_TRAIN DW_N_VAL DW_N_TEST DW_EPOCHS DW_BATCH DW_LR DW_WORKERS
    DW_ROOT DW_LOCAL DW_FAKE_ENCODER=1 DW_SKIP_RS3DADA=1 DW_CACHE_ON_DRIVE=1
"""
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# %run re-executes this file in a live kernel; drop cached copies of our modules
# so a fresh `git clone` in cell 1 is actually the code that runs
for _m in ("data", "eval", "predict", "calibrate", "train_head", "export_viewer"):
    sys.modules.pop(_m, None)

import numpy as np          # noqa: E402

import data as D            # noqa: E402
import eval as E            # noqa: E402
import predict as P         # noqa: E402
import train_head as TH     # noqa: E402


def _env(k, default, cast=str):
    v = os.environ.get(k)
    return default if v in (None, "") else cast(v)


CFG = {
    "N_TRAIN": _env("DW_N_TRAIN", 480, int),
    "N_VAL": _env("DW_N_VAL", 48, int),
    "N_TEST": _env("DW_N_TEST", 40, int),    # 40 + seed 0 == the Kaggle RS3DAda tiles
    "SEED": 0,
    "EPOCHS": _env("DW_EPOCHS", 30, int),
    "BATCH": _env("DW_BATCH", 4, int),
    "LR": _env("DW_LR", 1e-4, float),
    "WORKERS": _env("DW_WORKERS", 2, int),
    "SRC_GSD": TH.SRC_GSD,
    "TRAIN_GSD": TH.TRAIN_GSD,
    "GSD_SWEEP": [0.3, 0.5, 0.75, 1.0],
    "FAKE_ENCODER": _env("DW_FAKE_ENCODER", "0") == "1",
    "RUN_RS3DADA": _env("DW_SKIP_RS3DADA", "0") != "1",
    "CACHE_ON_DRIVE": _env("DW_CACHE_ON_DRIVE", "0") == "1",
}

# measured on Kaggle, same 40 tiles (progression.md, run "C + GSD + tiling/TTA")
KAGGLE_RS3DADA_C = {"overall": (6.74, 3.47, 0.597), "urban": (5.11, 2.68, 0.757),
                    "sparse": (2.83, 1.43, 0.684), "forest": (11.87, 7.87, 0.306)}
KAGGLE_FIRST_IDS = ["DC_09_29", "DC_19_37", "DC_21_10"]
KAGGLE_ID_AT = {10: "NYC_06437", 20: "PHL_3649"}

_LOG = None
_ENC = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _LOG:
        try:
            with open(_LOG, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass


def now():
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def resolve_paths():
    root = os.environ.get("DW_ROOT")
    persistent = True
    if not root:
        drive = "/content/drive/MyDrive"
        if os.path.isdir(drive):
            root = os.path.join(drive, "depthwizard")
        else:
            root, persistent = "/content/depthwizard_NOT_ON_DRIVE", False
    local = os.environ.get("DW_LOCAL") or (
        "/content/dw_local" if os.path.isdir("/content") else os.path.join(root, "_local"))
    cache_root = os.path.join(root if CFG["CACHE_ON_DRIVE"] else local, "cache")
    p = {"root": root, "local": local, "persistent": persistent,
         "cache_train": os.path.join(cache_root, "train"),
         "cache_val": os.path.join(cache_root, "val"),
         "train_local": os.path.join(local, "cache_train_local"),
         "val_local": os.path.join(local, "cache_val_local"),
         "ckpt": os.path.join(root, "ckpt"),
         "preds": os.path.join(root, "preds"),
         "viewer": os.path.join(root, "viewer_assets"),
         "results": os.path.join(root, "RESULTS.md"),
         "log": os.path.join(root, "run_log.txt"),
         "progression": os.path.join(root, "progression_colab.md"),
         "gamus": os.path.join(local, "gamus")}
    for k in ("root", "local", "ckpt", "preds"):
        os.makedirs(p[k], exist_ok=True)
    return p


def check_env():
    import torch
    gpu = torch.cuda.is_available()
    log(f"[setup] torch {torch.__version__} | GPU: "
        f"{torch.cuda.get_device_name(0) if gpu else 'NONE'}")
    try:
        import transformers
        log(f"[setup] transformers {transformers.__version__}")
    except Exception as e:                                      # noqa: BLE001
        log(f"[setup] transformers not importable: {e}")
    if not gpu and not CFG["FAKE_ENCODER"]:
        raise SystemExit("!! No GPU. Runtime -> Change runtime type -> T4 GPU -> Save. "
                         "Then run cell 1 and cell 2 again.")


def hf_token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception as e:                                      # noqa: BLE001
        log(f"[setup] could not read the Colab secret HF_TOKEN ({type(e).__name__}). "
            "DINOv3 is gated, so loading it will fail - see COLAB.md step 3.")
        return None


def repo_files():
    from huggingface_hub import HfApi
    for attempt in range(4):
        try:
            return HfApi().list_repo_files("earthflow/GAMUS", repo_type="dataset")
        except Exception as e:                                  # noqa: BLE001
            log(f"[setup] listing GAMUS failed ({type(e).__name__}), retrying")
            time.sleep(5 * (attempt + 1))
    raise SystemExit("could not list the GAMUS dataset on HuggingFace - check Internet")


def get_encoder(token):
    global _ENC
    if _ENC is None:
        if CFG["FAKE_ENCODER"]:
            log("[setup] !! FAKE ENCODER (test mode) - numbers are NOT DINOv3 results")
            _ENC = TH.FakeEncoder(hidden=32)
        else:
            log(f"[setup] loading {TH.DINOV3_REPO} (first time ~1.2 GB) ...")
            _ENC = TH.DinoV3Encoder(token=token, log=log)
    return _ENC


def download(repo_paths, token, workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download

    def get(rp):
        last = None
        for attempt in range(4):
            try:
                return rp, hf_hub_download("earthflow/GAMUS", rp, repo_type="dataset",
                                           token=token)
            except Exception as e:                              # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"download failed 4 times: {rp}: {last}")

    out, t0 = {}, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(get, rp) for rp in repo_paths]
        for i, fu in enumerate(as_completed(futs), 1):
            rp, lp = fu.result()
            out[rp] = lp
            if i % 200 == 0 or i == len(futs):
                log(f"  downloaded {i}/{len(futs)} files ({time.time() - t0:.0f}s)")
    return out


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage_cache(p, files, token):
    enc = get_encoder(token)
    for split, n, cdir in (("train", CFG["N_TRAIN"], p["cache_train"]),
                           ("val", CFG["N_VAL"], p["cache_val"])):
        os.makedirs(cdir, exist_ok=True)
        TH.verify_cache(cdir, log=log)
        recs = D.gamus_index(split, n, seed=CFG["SEED"], with_cls=False, files=files,
                             verbose=False)
        todo = [r for r in recs if not TH.is_cached(cdir, r["tile_id"])]
        log(f"[cache:{split}] {len(recs)} tiles selected, {len(recs) - len(todo)} "
            f"already cached, {len(todo)} to encode")
        if not todo:
            continue
        local = download([x for r in todo for x in (r["rgb"], r["height"])], token)
        t0 = time.time()
        for i, r in enumerate(todo, 1):
            TH.cache_tile({"tile_id": r["tile_id"], "rgb": local[r["rgb"]],
                           "height": local[r["height"]]}, enc, cdir,
                          CFG["SRC_GSD"], CFG["TRAIN_GSD"])
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"  encoded {i}/{len(todo)}  {el:.0f}s "
                    f"(~{el / i * (len(todo) - i) / 60:.1f} min left)")
        log(f"[cache:{split}] {TH.verify_cache(cdir, log=log)} good tiles")
    if enc.n_fp32_fallback or enc.n_clamped:
        log(f"[cache] note: {enc.n_fp32_fallback} fp32 fallbacks, "
            f"{enc.n_clamped} clamped activations")


def stage_train(p):
    tr, va = p["cache_train"], p["cache_val"]
    if CFG["CACHE_ON_DRIVE"]:
        n = TH.sync_dir(tr, p["train_local"]) + TH.sync_dir(va, p["val_local"])
        log(f"[train] copied {n} cache files Drive -> local disk")
        tr, va = p["train_local"], p["val_local"]
    _, hist = TH.train(tr, va, p["ckpt"], epochs=CFG["EPOCHS"], batch=CFG["BATCH"],
                       lr=CFG["LR"], num_workers=CFG["WORKERS"], log=log)
    best = min(hist, key=lambda h: h["val_rmse"])
    TH._atomic_json({"best_val_rmse": best["val_rmse"], "best_epoch": best["epoch"],
                     "epochs": len(hist), "finished": now()},
                    os.path.join(p["ckpt"], "DONE.json"))


def get_rs3dada(p, token):
    sdir = os.path.join(p["local"], "SynRS3D")
    if not os.path.isdir(sdir):
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/JTRNEO/SynRS3D", sdir], check=True)
    os.environ.setdefault("TORCH_HOME", os.path.join(p["local"], "torchhub"))
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download("JTRNEO/RS3DAda", "RS3DAda_vitl_DPT_height.pth", token=token)
    return P.RS3DAda(ckpt, synrs3d_dir=sdir).predict_fn


def gsd_sweep(recs, fn, tile, patch_multiple, model_gsd, label):
    """Simulate coarser imagery (Cartosat is 0.3-1 m): degrade each test image
    to GSD g, predict, score against GT on the same degraded grid."""
    rows = []
    for g in CFG["GSD_SWEEP"]:
        errs, gts = [], []
        for r in recs:
            rgb, gt = TH.load_rgb_255(r["rgb"]), TH.load_height(r["height"])
            H0, W0 = gt.shape
            s = CFG["SRC_GSD"] / g
            hw = (max(16, int(round(H0 * s))), max(16, int(round(W0 * s))))
            rgb_g = P.resample_hw(rgb, hw) if hw != (H0, W0) else rgb
            gt_g = TH.resample_height(gt, hw)
            pred = P.predict_height(rgb_g, fn, input_gsd=g, model_gsd=model_gsd,
                                    tile=tile, overlap=0.25, tta=False,
                                    gsd_normalise=True, patch_multiple=patch_multiple)
            m = np.isfinite(gt_g) & np.isfinite(pred)
            errs.append((pred[m] - gt_g[m]).astype(np.float64))
            gts.append(gt_g[m].astype(np.float64))
        met = E.metrics_from(np.concatenate(errs), np.concatenate(gts))
        rows.append((g, met))
        log(f"  [{label}] GSD {g:.2f} m: RMSE {met['rmse_m']:.2f}  MAE {met['mae_m']:.2f}  "
            f"r {met['pearson_r']:.3f}")
    return rows


def predict_dir(recs, fn, out_dir, **kw):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for i, r in enumerate(recs, 1):
        pred = P.predict_height(TH.load_rgb_255(r["rgb"]), fn, verbose=(i == 1), **kw)
        np.save(os.path.join(out_dir, f"{r['tile_id']}_pred.npy"), pred)
        if i % 10 == 0 or i == len(recs):
            log(f"  {i}/{len(recs)}  {time.time() - t0:.0f}s")


def stage_eval(p, token):
    enc = get_encoder(token)
    info = {"tables": {}, "sweeps": {}, "notes": [], "enc_report": enc.report}

    # the SAME tiles as the Kaggle run (gamus_index is verified against it)
    rgb_d, h_d, c_d = D.fetch_gamus_subset("test", CFG["N_TEST"], p["gamus"], seed=CFG["SEED"])
    recs = D.pair_dirs(rgb_d, h_d, c_d, "*.h5", "*_AGL.h5", "*_CLS.h5")
    ids = [r["tile_id"] for r in recs]
    same = (len(ids) == 40 and ids[:3] == KAGGLE_FIRST_IDS
            and all(ids[k] == v for k, v in KAGGLE_ID_AT.items()))
    info["same_tiles"] = same
    info["test_ids"] = ids
    log(f"[eval] {len(ids)} test tiles, first {ids[:3]} -> "
        f"{'IDENTICAL to the Kaggle RS3DAda tiles' if same else 'NOT the Kaggle tile set'}")
    buckets = E.auto_buckets_from_cls(c_d)

    try:
        # 1. our model
        head, ck = TH.load_head(os.path.join(p["ckpt"], "best.pt"), device=enc.device)
        info["best_epoch"] = ck.get("epoch")
        tile = TH.work_size((1024, 1024), CFG["SRC_GSD"], CFG["TRAIN_GSD"], enc.patch)[0]
        fn = TH.make_predict_fn(enc, head)
        pdir = os.path.join(p["preds"], "dinov3_head")
        log("[eval] predicting with the trained DINOv3-SAT head (GSD-normalised, TTA)")
        predict_dir(recs, fn, pdir, input_gsd=CFG["SRC_GSD"], model_gsd=CFG["TRAIN_GSD"],
                    tile=tile, overlap=0.25, tta=True, gsd_normalise=True,
                    patch_multiple=enc.patch)

        # 2. the do-nothing floor on the same tiles
        zdir = os.path.join(p["local"], "preds_zero")
        os.makedirs(zdir, exist_ok=True)
        for r in recs:
            np.save(os.path.join(zdir, f"{r['tile_id']}_pred.npy"),
                    np.zeros(TH.load_height(r["height"]).shape, np.float32))

        fake = " [FAKE ENCODER - TEST ONLY]" if CFG["FAKE_ENCODER"] else ""
        order = [("B0  predict zero (floor)", zdir),
                 ("H1  DINOv3-SAT head (ours)" + fake, pdir)]

        # 3. RS3DAda zero-shot, same notebook, same tiles (optional)
        rs_fn = None
        if CFG["RUN_RS3DADA"]:
            try:
                log("[eval] RS3DAda baseline on the same tiles (config C: GSD norm + TTA)")
                rs_fn = get_rs3dada(p, token)
                rdir = os.path.join(p["preds"], "rs3dada")
                predict_dir(recs, rs_fn, rdir, input_gsd=CFG["SRC_GSD"], model_gsd=0.5,
                            tile=P.RS3DADA_PATCH, overlap=0.25, tta=True, gsd_normalise=True)
                order.insert(1, ("C   RS3DAda zero-shot (baseline)", rdir))
            except Exception as e:                              # noqa: BLE001
                rs_fn = None
                info["notes"].append(f"RS3DAda re-run failed ({type(e).__name__}: {e}); "
                                     "use the Kaggle numbers as the baseline")
                log(f"[eval] RS3DAda skipped: {type(e).__name__}: {e}")

        for tag, d in order:
            res = E.evaluate(d, h_d, buckets, "*", "*_AGL.h5")
            info["tables"][tag] = res["summary"]
            log("\n" + E.markdown_table(res["summary"], tag))
            if not tag.startswith("B0"):
                E.append_progression(p["progression"], tag, res["summary"],
                                     notes=f"{len(recs)} GAMUS test tiles "
                                           f"(Kaggle set: {same}); colab_train.py")

        # 4. GSD robustness
        try:
            log("\n[eval] GSD robustness sweep (simulating coarser, Cartosat-like input)")
            info["sweeps"]["DINOv3-SAT head (ours)" + fake] = gsd_sweep(
                recs, fn, tile, enc.patch, CFG["TRAIN_GSD"], "ours")
            if rs_fn is not None:
                info["sweeps"]["RS3DAda zero-shot"] = gsd_sweep(
                    recs, rs_fn, P.RS3DADA_PATCH, 14, 0.5, "RS3DAda")
        except Exception as e:                                  # noqa: BLE001
            info["notes"].append(f"GSD sweep failed: {type(e).__name__}: {e}")
            log(traceback.format_exc())

        # 5. viewer assets with REAL predictions
        try:
            import export_viewer as X
            os.makedirs(p["viewer"], exist_ok=True)
            names = []
            for _, t in X.pick_interesting(recs, 3):
                pred = np.load(os.path.join(pdir, f"{t.tile_id}_pred.npy"))
                X.export(t, p["viewer"], t.tile_id, pred=pred)
                names.append(t.tile_id)
            with open(os.path.join(p["viewer"], "scenes.json"), "w", encoding="utf-8") as fh:
                json.dump(names, fh)
            info["viewer_scenes"] = names
        except Exception as e:                                  # noqa: BLE001
            info["notes"].append(f"viewer export failed: {type(e).__name__}: {e}")
            log(traceback.format_exc())
    finally:
        write_results(p, info)


# --------------------------------------------------------------------------
# RESULTS.md
# --------------------------------------------------------------------------
def write_results(p, info):
    hist = []
    hp = os.path.join(p["ckpt"], "history.json")
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as fh:
            hist = json.load(fh)
    L = ["# DepthWizard - DINOv3-SAT height head: results", "",
         f"Generated {now()} by `colab_train.py`. Every number here was measured in this run.", ""]
    if CFG["FAKE_ENCODER"]:
        L += ["> **TEST MODE - FAKE ENCODER. These numbers are NOT DINOv3 results.**", ""]

    L += ["## Setup", "",
          f"- Training tiles: {CFG['N_TRAIN']} (GAMUS train split), validation: "
          f"{CFG['N_VAL']} (GAMUS val split) - model selected on val, never on test",
          f"- Test tiles: {len(info.get('test_ids', []))} (GAMUS test split) - "
          f"**{'identical to the Kaggle RS3DAda run' if info.get('same_tiles') else 'NOT the Kaggle tile set'}**",
          f"- Encoder: `{TH.DINOV3_REPO}`, frozen; features cached in fp16",
          f"- Head: DPT-style, trained {len(hist)} epochs, batch {CFG['BATCH']}, "
          f"AdamW lr {CFG['LR']}, cosine schedule, masked Huber loss on metres, "
          f"target clipped to [0, {TH.HCLIP:.0f}] m",
          f"- Training GSD {CFG['TRAIN_GSD']} m (GAMUS 0.3 m resampled 1024 -> 608 px); "
          "inputs are GSD-normalised to it at inference", "",
          "Encoder self-checks (read at runtime, not assumed):", "",
          "```json", json.dumps(info.get("enc_report", {}), indent=2, default=str), "```", ""]

    if hist:
        best = min(hist, key=lambda h: h["val_rmse"])
        L += ["## Training", "",
              f"Best validation RMSE **{best['val_rmse']:.3f} m** at epoch {best['epoch']} "
              f"(checkpoint `ckpt/best.pt`).", "",
              "| epoch | train loss | val RMSE (m) | val MAE (m) | val bias (m) |",
              "|---|---|---|---|---|"]
        step = max(1, len(hist) // 10)
        for h in hist[::step] + ([hist[-1]] if (len(hist) - 1) % step else []):
            L.append(f"| {h['epoch']} | {h['train_loss']:.3f} | {h['val_rmse']:.3f} | "
                     f"{h['val_mae']:.3f} | {h['val_bias']:+.3f} |")
        L.append("")

    if info.get("tables"):
        L += ["## Test results", ""]
        for tag, summ in info["tables"].items():
            L += [E.markdown_table(summ, tag), ""]
        L += ["### Overall, side by side", "",
              "| run | RMSE (m) | MAE (m) | bias (m) | r | forest RMSE (m) |",
              "|---|---|---|---|---|---|"]
        for tag, summ in info["tables"].items():
            o, fo = summ["overall"], summ.get("forest")
            L.append(f"| {tag} | {o['rmse_m']:.2f} | {o['mae_m']:.2f} | {o['bias_m']:+.2f} | "
                     f"{o['pearson_r']:.3f} | {fo['rmse_m']:.2f} |" if fo else
                     f"| {tag} | {o['rmse_m']:.2f} | {o['mae_m']:.2f} | {o['bias_m']:+.2f} | "
                     f"{o['pearson_r']:.3f} | - |")
        k = KAGGLE_RS3DADA_C
        L += [f"| *reference: Kaggle RS3DAda run C* | {k['overall'][0]:.2f} | "
              f"{k['overall'][1]:.2f} | -2.27 | {k['overall'][2]:.3f} | {k['forest'][0]:.2f} |", ""]

    if info.get("sweeps"):
        L += ["## GSD robustness sweep", "",
              "Each test image degraded to the given GSD, predicted, and scored against "
              "ground truth resampled to the same grid. Coarser GT is also smoother, so "
              "part of any change is the target changing, not only the model. The 0.30 m "
              "row reaches the model through one resample and the others through two "
              "(slightly smoother), so read the 0.50 -> 1.00 m trend, not the first step.", ""]
        names = list(info["sweeps"])
        L.append("| input GSD (m) | " + " | ".join(f"{n} RMSE / r" for n in names) + " |")
        L.append("|---|" + "---|" * len(names))
        for i, g in enumerate(CFG["GSD_SWEEP"]):
            cells = []
            for n in names:
                rows = info["sweeps"][n]
                m = rows[i][1] if i < len(rows) else None
                cells.append(f"{m['rmse_m']:.2f} / {m['pearson_r']:.3f}" if m else "-")
            L.append(f"| {g:.2f} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## Honest caveats (keep these in the deck)", "",
          "- Trained and tested on GAMUS (US cities, aerial). The test split is held out, "
          "but ISRO evaluation imagery is a different sensor and region, so these numbers "
          "will overstate performance there.",
          "- nDSM only: terrain is removed, so there is no 'hilly' bucket without a DEM.",
          "- Fixed 0.5 m training GSD with flip/rotation augmentation only; colour "
          "augmentation was not possible on cached features.",
          "- GAMUS GSD (0.3 m) is asserted from the DFC2019 spec and a car-length check, "
          "not read from the files.", ""]
    if info.get("notes"):
        L += ["## Run notes", ""] + [f"- {n}" for n in info["notes"]] + [""]
    if info.get("viewer_scenes"):
        L += ["## Viewer assets", "",
              f"`viewer_assets/` holds {len(info['viewer_scenes'])} scenes with REAL "
              f"predictions ({', '.join(info['viewer_scenes'])}). Copy the folder's files "
              "into the repo's `web/assets/`, run `python serve.py`, open "
              "http://localhost:8777 - the badge reads MODEL PREDICTION and the Error "
              "overlay + live RMSE panel work.", ""]
    with open(p["results"], "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    log(f"\n[results] written -> {p['results']}")


# --------------------------------------------------------------------------
def main():
    global _LOG
    stage = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    p = resolve_paths()
    _LOG = p["log"]
    log("=" * 70)
    log(f"DepthWizard colab_train  stage={stage}  {now()}")
    log(f"[setup] results -> {p['root']}" + ("" if p["persistent"] else
        "   !! NOT ON GOOGLE DRIVE - run cell 1 (drive.mount) or results are lost"))
    log(f"[setup] config {json.dumps({k: v for k, v in CFG.items() if k != 'GSD_SWEEP'})}")
    check_env()
    token = None if CFG["FAKE_ENCODER"] else hf_token()

    done = os.path.exists(os.path.join(p["ckpt"], "DONE.json"))
    if stage in ("all", "cache", "train") and not done:
        files = repo_files()
        if stage in ("all", "cache"):
            log("\n########## 1/3  CACHE ENCODER FEATURES ##########")
            stage_cache(p, files, token)
        if stage in ("all", "train"):
            log("\n########## 2/3  TRAIN HEAD ##########")
            stage_train(p)
    elif done:
        log("[setup] training already finished (ckpt/DONE.json) - skipping to evaluation")

    if stage in ("all", "eval"):
        log("\n########## 3/3  EVALUATE ##########")
        stage_eval(p, token)
    log(f"\nALL DONE {now()}. Send {p['results']} to the team lead.")


if __name__ == "__main__":
    main()
