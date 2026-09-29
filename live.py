"""
live.py - "live mode" for the viewer: upload an image in the browser, run a
real model on this machine, and open the result in 3D.

Started by `python serve.py --live`. Plain `python serve.py` does not import
this file, so the static viewer keeps working with no extra dependencies.

Models the app can run (listed at GET /api/models):
  * every .pt file in dw_run/ckpt/ and models/ that is a checkpoint type the
    code knows how to build:
      - DINOv3 height head (our training: keys head / in_dim / head_dim)
      - RS3DAda / SynRS3D DPT_DINOv2 state_dict (keys start with "pretrained.")
  * "rs3dada"  - the public RS3DAda weights (downloaded once, no login)
  * "demo"     - fake heights from image darkness, for testing the plumbing
A .pt of any other architecture is rejected with a clear message: loading it
would need code for that architecture.

Models are loaded once and kept in memory, so repeat runs are fast. The
DINOv3 encoder is shared by all DINOv3 heads. One job runs at a time.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [os.path.join(HERE, "dw_run", "ckpt"), os.path.join(HERE, "models")]
UPLOADS = os.path.join(HERE, "uploads")
OUT_ROOT = os.path.join(HERE, "out_dsm")
LIVE_ASSETS = os.path.join(HERE, "web", "assets_live")
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
# run outputs the viewer may offer for download (GET /api/files/<scene>)
DOWNLOADS = [("dsm.tif", "DSM GeoTIFF", "elevation above sea level (m)"),
             ("ndsm.tif", "Heights GeoTIFF", "height above ground (m)"),
             ("dem.tif", "Terrain GeoTIFF", "the DEM on this grid (m)"),
             ("dsm_calibrated.tif", "Calibrated DSM GeoTIFF",
              "DSM after offset + scale fitted to the reference"),
             ("residual.tif", "Error GeoTIFF", "run minus reference (m)"),
             ("reference_on_grid.tif", "Reference GeoTIFF", "reference on this grid (m)"),
             ("rdsm_0to1.png", "Relative shape PNG", "0-1 scaled heights"),
             ("report.json", "Run report", "every setting and number"),
             ("validation.json", "Validation", "scores against the reference"),
             ("SUMMARY.txt", "Summary", "key numbers in plain words")]
MAX_UPLOAD = 1024 ** 3          # 1 GB
MAX_SIDE = 4096                 # larger images are centre-cropped to this (reported)


# ==========================================================================
# checkpoint types
# ==========================================================================
def classify_state(obj) -> str:
    """'dinov3_head' | 'rs3dada' | 'unknown' from a loaded checkpoint object."""
    if isinstance(obj, dict):
        if {"head", "in_dim", "head_dim"} <= set(obj.keys()):
            return "dinov3_head"
        sd = obj.get("state_dict", obj)
        if isinstance(sd, dict) and sd:
            keys = [k for k in sd.keys() if isinstance(k, str)]
            if keys and sum(k.startswith("pretrained.") for k in keys) > len(keys) * 0.3:
                return "rs3dada"
    return "unknown"


def _label(stem):
    return re.sub(r"[_-]+", " ", stem)


def list_models(model_dirs=None):
    """Cheap listing: files only, nothing loaded. The type is confirmed when a
    model is first used (loading a .pt needs PyTorch and can take seconds)."""
    out = [{"id": "demo", "label": "Demo (fake heights, no model)", "kind": "demo",
            "path": None},
           {"id": "rs3dada", "label": "RS3DAda (public weights)", "kind": "rs3dada",
            "path": None}]
    seen = set()
    for d in (model_dirs or MODEL_DIRS):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if not fn.lower().endswith(".pt") or fn in seen:
                continue
            seen.add(fn)
            stem = fn[:-3]
            out.append({"id": "file:" + fn, "label": f"{_label(stem)}  ({fn})",
                        "kind": "folder" if os.path.isdir(p) else "checkpoint",
                        "path": p})
    return out


DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-sat493m"      # same as train_head.py


def dinov3_cached() -> bool:
    """True when the DINOv3 weights are already downloaded on this machine,
    so no HF_TOKEN and no internet are needed to load them."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(DINOV3_REPO, f), str)
                   for f in ("config.json", "model.safetensors"))
    except Exception:                               # noqa: BLE001
        return False


def environment():
    info = {"torch": False, "cuda": False, "device": "cpu", "hf_token": bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "dinov3_cached": dinov3_cached(), "offline": os.environ.get("HF_HUB_OFFLINE") == "1"}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
    except Exception:                               # noqa: BLE001
        pass
    return info


# ==========================================================================
# model cache
# ==========================================================================
class Models:
    def __init__(self, model_dirs=None):
        self.model_dirs = model_dirs
        self.cache = {}             # id -> (predict_fn, backend_settings_name, label)
        self.encoder = None

    def get(self, model_id, log=print):
        if model_id in self.cache:
            return self.cache[model_id]
        import dsm as M
        entry = next((m for m in list_models(self.model_dirs) if m["id"] == model_id), None)
        if entry is None:
            raise ValueError(f"unknown model '{model_id}'")
        if entry["kind"] == "demo":
            import predict as P
            res = (P.dummy_backend(), "dummy", "DEMO - fake heights, not a real model")
        elif entry["kind"] == "rs3dada":
            res = (M.get_backend("rs3dada"), "rs3dada", M.BACKENDS["rs3dada"]["label"])
        elif entry["kind"] == "folder":
            raise ValueError(f"{os.path.basename(entry['path'])} is a FOLDER, not a file: it "
                             "was unzipped. Copy the original .pt file again without opening it.")
        else:
            res = self._load_file(entry["path"], log)
        self.cache[model_id] = res
        return res

    def _load_file(self, path, log):
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("PyTorch is not installed on this machine") from e
        import dsm as M
        log(f"[live] reading {os.path.basename(path)} ...")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        kind = classify_state(obj)
        name = os.path.basename(path)
        if kind == "dinov3_head":
            import train_head as TH
            if self.encoder is None:
                log("[live] loading the DINOv3 encoder (first time only) ...")
                try:
                    self.encoder = TH.DinoV3Encoder(log=log)
                except RuntimeError as e:
                    raise RuntimeError(f"{e} - DINOv3 is not downloaded yet: set HF_TOKEN once "
                                       "(or run `hf auth login`) and start serve.py again") from e
            head, ck = TH.load_head(path, device=self.encoder.device)
            log(f"[live] {name}: DINOv3 head, epoch {ck.get('epoch')}, on {self.encoder.device}")
            return (TH.make_predict_fn(self.encoder, head), "dinov3",
                    f"DINOv3-SAT head ({name})")
        if kind == "rs3dada":
            import predict as P
            if not os.path.isdir(os.path.join(HERE, "SynRS3D")):
                M.get_backend("rs3dada")        # clones SynRS3D once
            fn = P.RS3DAda(path, synrs3d_dir=os.path.join(HERE, "SynRS3D")).predict_fn
            log(f"[live] {name}: RS3DAda-type checkpoint")
            return fn, "rs3dada", f"RS3DAda ({name})"
        keys = list(obj.keys())[:6] if isinstance(obj, dict) else type(obj).__name__
        raise ValueError(f"{name} is not a checkpoint type this app can build "
                         f"(first keys: {keys}). Supported: our DINOv3 height heads "
                         "and RS3DAda checkpoints.")


# ==========================================================================
# jobs
# ==========================================================================
class _Tee(io.TextIOBase):
    def __init__(self, sink, echo):
        self.sink, self.echo = sink, echo

    def write(self, s):
        if s.strip():
            for ln in s.rstrip().splitlines():
                self.sink.append(ln)
        self.echo.write(s)
        return len(s)

    def flush(self):
        self.echo.flush()


STAGES = [("[live]", "loading model"), ("[image]", "reading image"),
          ("[predict]", "predicting heights"), ("[model] nDSM done", "adding terrain"),
          ("[dem]", "adding terrain"), ("[viewer]", "building 3D scene"),
          ("[done]", "finishing")]


class Jobs:
    def __init__(self, models: Models):
        self.models = models
        self.jobs = {}
        self.lock = threading.Lock()        # one model run at a time (GPU memory)

    def submit(self, filename, data: bytes, model_id, gsd=None, gsd_unknown=False, gcps=None):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXT:
            raise ValueError(f"unsupported file type '{ext}': use GeoTIFF, PNG or JPG")
        # "plain" = no map information: PNG/JPG, or a TIFF without a CRS. A GeoTIFF
        # carries its own position and pixel size, so pixel size / map points
        # from the form are dropped for it (they would override the file).
        plain = ext in (".png", ".jpg", ".jpeg")
        if ext in (".tif", ".tiff"):
            try:
                from rasterio.io import MemoryFile
                with MemoryFile(data) as mf, mf.open() as s:
                    plain = s.crs is None
            except Exception as e:                  # noqa: BLE001
                raise ValueError(f"could not read the TIFF: {e}") from e
        if not plain:
            gsd, gsd_unknown, gcps = None, False, None
        pts = None
        if gcps and plain:
            import dsm as M
            pts = M.parse_gcps(gcps)
            if len(pts) < 3:
                raise ValueError(f"{len(pts)} ground control point(s) given: need at least 3")
        if plain and not (gsd or gsd_unknown or pts):
            raise ValueError("this image has no map information (PNG/JPG or a plain TIFF): "
                             "enter the pixel size, tick 'pixel size unknown', or give 3+ "
                             "ground control points")
        if not any(m["id"] == model_id for m in list_models(self.models.model_dirs)):
            raise ValueError(f"unknown model '{model_id}'")
        jid = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4]
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(os.path.basename(filename))[0])[:40]
        d = os.path.join(UPLOADS, jid)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, stem + ext)
        with open(path, "wb") as f:
            f.write(data)
        mtag = re.sub(r"[^A-Za-z0-9]+", "", model_id.replace("file:", "").replace(".pt", ""))[:20]
        job = {"id": jid, "state": "queued", "stage": "queued", "log": [], "scene": f"{stem}_{mtag}",
               "model": model_id, "file": filename, "error": None, "started": time.time(),
               "seconds": None, "summary": None}
        self.jobs[jid] = job
        threading.Thread(target=self._run, args=(job, path, gsd, gsd_unknown, pts),
                         daemon=True).start()
        return job

    def _run(self, job, path, gsd, gsd_unknown=False, gcps=None):
        with self.lock:
            job["state"] = "running"
            lines = job["log"]
            tee = _Tee(lines, sys.__stdout__)

            def log(msg):
                print(msg)

            try:
                with contextlib.redirect_stdout(tee):
                    stager = threading.Thread(target=self._stage_watch, args=(job,), daemon=True)
                    stager.start()
                    fn, bname, label = self.models.get(job["model"], log)
                    import dsm as M
                    window = self._window(path)
                    rep = M.run(path, os.path.join(OUT_ROOT, job["scene"]), fn=fn, backend=bname,
                                gsd=gsd, window=window, export=LIVE_ASSETS, name=job["scene"],
                                label=label, source_note=f"uploaded: {job['file']}",
                                gsd_unknown=gsd_unknown, gcps=gcps)
                job["summary"] = {k: rep.get(k) for k in ("kind", "gsd_m", "size_px", "model", "seconds",
                                                          "scale_known", "georeferenced_from_gcps")}
                job["summary"]["sanity"] = rep.get("sanity", {}).get("ndsm_percentiles_m")
                if rep.get("buildings"):
                    job["summary"]["buildings"] = rep["buildings"]["count"]
                job["state"], job["stage"] = "done", "done"
            except SystemExit as e:
                job["state"], job["error"] = "error", str(e)
            except Exception as e:                  # noqa: BLE001
                job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"
                lines.append(traceback.format_exc().splitlines()[-1])
            job["seconds"] = round(time.time() - job["started"], 1)

    @staticmethod
    def _stage_watch(job):
        while job["state"] in ("queued", "running"):
            for ln in reversed(job["log"][-20:]):
                hit = next((s for tag, s in STAGES if ln.startswith(tag)), None)
                if hit:
                    job["stage"] = hit
                    break
            time.sleep(0.3)

    @staticmethod
    def _window(path):
        """Centre crop for very large images: the 3D view uses <= 1024 px of
        height anyway, and memory/time grow with area. Reported in the log."""
        if not path.lower().endswith((".tif", ".tiff")):
            return None
        try:
            import rasterio
            with rasterio.open(path) as s:
                W, H = s.width, s.height
        except Exception:                           # noqa: BLE001
            return None
        if max(W, H) <= MAX_SIDE:
            return None
        w, h = min(W, MAX_SIDE), min(H, MAX_SIDE)
        print(f"[image] {W}x{H} px is large: using the centre {w}x{h} px")
        return ((W - w) // 2, (H - h) // 2, w, h)

    def public(self, jid):
        j = self.jobs.get(jid)
        if not j:
            return None
        return {k: (v[-12:] if k == "log" else v) for k, v in j.items()}


# ==========================================================================
# HTTP glue (called from serve.py)
# ==========================================================================
class Live:
    def __init__(self, model_dirs=None):
        self.models = Models(model_dirs)
        self.jobs = Jobs(self.models)
        os.makedirs(LIVE_ASSETS, exist_ok=True)
        sp = os.path.join(LIVE_ASSETS, "scenes.json")
        if not os.path.exists(sp):
            with open(sp, "w") as f:
                json.dump([], f)

    def handle_get(self, path):
        if path == "/api/health":
            return 200, {"live": True, "env": environment()}
        if path == "/api/models":
            return 200, {"models": [{k: m[k] for k in ("id", "label", "kind")}
                                    for m in list_models(self.models.model_dirs)],
                         "loaded": list(self.models.cache.keys())}
        if path.startswith("/api/files/"):
            scene = path.rsplit("/", 1)[1]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", scene):
                return 400, {"error": "bad scene name"}
            d = os.path.join(OUT_ROOT, scene)
            return 200, {"files": [{"name": n, "label": lab, "what": w,
                                    "bytes": os.path.getsize(os.path.join(d, n))}
                                   for n, lab, w in DOWNLOADS
                                   if os.path.isfile(os.path.join(d, n))]}
        if path.startswith("/api/job/"):
            j = self.jobs.public(path.rsplit("/", 1)[1])
            return (200, j) if j else (404, {"error": "no such job"})
        return None

    def handle_post(self, path, query, body):
        if path == "/api/validate":
            return self.validate(query, body)
        if path != "/api/run":
            return None
        if len(body) > MAX_UPLOAD:
            return 413, {"error": "file larger than 1 GB"}
        try:
            gsd = float(query.get("gsd")) if query.get("gsd") else None
            job = self.jobs.submit(query.get("name", "upload.tif"), body,
                                   query.get("model", "demo"), gsd,
                                   gsd_unknown=query.get("gsd_unknown") == "1",
                                   gcps=query.get("gcps") or None)
            return 200, {"job": job["id"], "scene": job["scene"]}
        except Exception as e:                      # noqa: BLE001
            return 400, {"error": str(e)}

    def file_path(self, path):
        """/api/file/<scene>/<name> -> absolute path of an allowed run output, or None."""
        parts = path.split("/")
        if len(parts) != 5 or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", parts[3]):
            return None
        if parts[4] not in {n for n, _, _ in DOWNLOADS}:
            return None
        p = os.path.join(OUT_ROOT, parts[3], parts[4])
        return p if os.path.isfile(p) else None

    def validate(self, query, body):
        """Score a finished run against a reference GeoTIFF (validate.py)."""
        scene = query.get("scene", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", scene):
            return 400, {"error": "bad scene name"}
        run_dir = os.path.join(OUT_ROOT, scene)
        if not os.path.exists(os.path.join(run_dir, "report.json")):
            return 404, {"error": f"no run output for '{scene}' on this machine: "
                                  "run the image first, then add the reference"}
        name = query.get("name", "reference.tif")
        if not name.lower().endswith((".tif", ".tiff")):
            return 400, {"error": "the reference must be a GeoTIFF (.tif)"}
        if len(body) > MAX_UPLOAD:
            return 413, {"error": "file larger than 1 GB"}
        d = os.path.join(UPLOADS, "ref_" + time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(name))[:60])
        with open(path, "wb") as f:
            f.write(body)
        try:
            import validate as V
            with self.jobs.lock:
                res = V.validate(run_dir, path, LIVE_ASSETS, scene, ref_name=os.path.basename(name))
            return 200, res
        except Exception as e:                      # noqa: BLE001
            return 400, {"error": f"{type(e).__name__}: {e}"}
