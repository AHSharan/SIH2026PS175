"""
CPU tests for live.py (serve.py --live). No GPU, no PyTorch, no weights:
checkpoint-type detection on plain dicts, model listing, upload validation,
and a full Demo job on the shipped Chungthang sample through the real HTTP
server.

Run from the repo root:  python tests/test_live_cpu.py
What this cannot test: loading real .pt files (needs PyTorch) and the
DINOv3/RS3DAda models themselves - that runs on the GPU laptop.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import live as L            # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def http(method, url, data=None):
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    # ---- checkpoint type detection (what torch.load returns, as plain dicts)
    check("our head checkpoint -> dinov3_head",
          L.classify_state({"head": {}, "in_dim": 1024, "head_dim": 256, "epoch": 22}) == "dinov3_head")
    check("training last.pt (with optimiser) -> dinov3_head",
          L.classify_state({"head": {}, "opt": {}, "in_dim": 1024, "head_dim": 256}) == "dinov3_head")
    sd = {f"pretrained.blocks.{i}.w": 0 for i in range(20)}
    sd.update({"depth_head.x": 0, "seg_head.y": 0})
    check("RS3DAda state_dict -> rs3dada", L.classify_state(sd) == "rs3dada")
    check("wrapped state_dict -> rs3dada", L.classify_state({"state_dict": sd}) == "rs3dada")
    check("other architecture -> unknown",
          L.classify_state({"backbone.conv1.weight": 0, "fc.bias": 0}) == "unknown")
    check("non-dict -> unknown", L.classify_state([1, 2, 3]) == "unknown")

    tmp = tempfile.mkdtemp()
    try:
        # ---- model listing
        md = os.path.join(tmp, "models")
        os.makedirs(os.path.join(md, "unzipped.pt"))
        open(os.path.join(md, "best_1500.pt"), "wb").write(b"x")
        open(os.path.join(md, "notes.txt"), "w").write("x")
        ms = {m["id"]: m for m in L.list_models([md])}
        check("lists demo + rs3dada + .pt files only",
              set(ms) == {"demo", "rs3dada", "file:best_1500.pt", "file:unzipped.pt"}, str(sorted(ms)))
        check("unzipped .pt folder flagged", ms["file:unzipped.pt"]["kind"] == "folder")
        mdl = L.Models([md])
        try:
            mdl.get("file:unzipped.pt")
            check("folder model refused with a clear message", False)
        except ValueError as e:
            check("folder model refused with a clear message", "FOLDER" in str(e))

        # ---- the real server, live mode, on a free port
        port = 8791
        env = dict(os.environ)
        srv = subprocess.Popen([sys.executable, "serve.py", "--live", "--port", str(port)],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(40):
                try:
                    code, h = http("GET", base + "/api/health")
                    break
                except Exception:                   # noqa: BLE001
                    time.sleep(0.25)
            check("GET /api/health says live", code == 200 and h.get("live") is True)
            code, m = http("GET", base + "/api/models")
            check("GET /api/models lists demo", code == 200 and any(x["id"] == "demo" for x in m["models"]))
            code, r = http("POST", base + "/api/run?name=x.md&model=demo", b"hi")
            check("wrong file type rejected", code == 400 and "unsupported" in r["error"])
            code, r = http("POST", base + "/api/run?name=x.png&model=demo", b"hi")
            check("PNG without pixel size rejected", code == 400 and "pixel size" in r["error"])
            code, r = http("POST", base + "/api/run?name=x.png&model=nope&gsd=0.3", b"hi")
            check("unknown model rejected up front", code == 400 and "unknown model" in r["error"])

            data = open("samples/chungthang_wv2.tif", "rb").read()
            code, r = http("POST", base + "/api/run?name=chungthang_wv2.tif&model=demo", data)
            check("GeoTIFF upload accepted", code == 200 and "job" in r, str(r))
            j = {}
            for _ in range(240):
                code, j = http("GET", base + "/api/job/" + r["job"])
                if j.get("state") in ("done", "error"):
                    break
                time.sleep(0.5)
            check("demo job finishes", j.get("state") == "done", j.get("error") or j.get("stage"))
            sc = r["scene"]
            meta = json.load(open(os.path.join(ROOT, "web", "assets_live", sc + "_meta.json")))
            check("scene written for the viewer (DSM, buildings)",
                  meta["kind"] == "dsm" and meta.get("has_ndsm") and meta.get("has_buildings"))
            scenes = json.load(open(os.path.join(ROOT, "web", "assets_live", "scenes.json")))
            check("new scene listed first", scenes and scenes[0] == sc)
            check("GeoTIFF outputs saved",
                  all(os.path.exists(os.path.join(ROOT, "out_dsm", sc, f))
                      for f in ("dsm.tif", "ndsm.tif", "SUMMARY.txt")))
        finally:
            srv.terminate()
            srv.wait(5)
        # plain serve.py (no --live) keeps the API off
        srv = subprocess.Popen([sys.executable, "serve.py", "--port", str(port + 1)], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(40):
                try:
                    code, h = http("GET", f"http://127.0.0.1:{port + 1}/api/health")
                    break
                except Exception:                   # noqa: BLE001
                    time.sleep(0.25)
            check("plain serve.py: live mode off", code == 404 and h.get("live") is False)
        finally:
            srv.terminate()
            srv.wait(5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
