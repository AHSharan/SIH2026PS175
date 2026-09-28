"""Static server for web/ plus a POST /save endpoint so the viewer can write
high-resolution captures straight to disk (shots/). Local only.

    python serve.py          viewer only (no extra dependencies)
    python serve.py --live   also lets the viewer upload an image and run a
                             model on this machine (see live.py)
"""
import argparse
import base64
import json
import os
import re
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
os.makedirs(SHOTS, exist_ok=True)
LIVE = None                         # live.Live instance when started with --live


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # live results are rewritten in place: never serve a stale scene
        if self.path.startswith("/assets_live/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith("/api/"):
            res = LIVE.handle_get(u.path) if LIVE else (404, {"live": False})
            if res is None:
                res = (404, {"error": "unknown endpoint"})
            return self._json(*res)
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        if u.path.startswith("/api/"):
            if not LIVE:
                self.rfile.read(n)
                return self._json(404, {"error": "start the server with: python serve.py --live"})
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            res = LIVE.handle_post(u.path, q, self.rfile.read(n))
            return self._json(*(res or (404, {"error": "unknown endpoint"})))
        if u.path != "/save":
            self.send_error(404)
            return
        try:
            payload = json.loads(self.rfile.read(n))
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", payload["name"])[:80]
            url = payload["dataurl"]
            raw = base64.b64decode(url.split(",", 1)[1])
            ext = ".jpg" if "jpeg" in url[:30] else ".png"
            path = os.path.join(SHOTS, name + ext)
            with open(path, "wb") as f:
                f.write(raw)
            body = {"ok": True, "path": path, "bytes": len(raw)}
            print(f"[save] {name}{ext}  {len(raw)/1024:.0f} KB")
        except Exception as e:                      # noqa: BLE001
            body = {"ok": False, "error": str(e)}
            print(f"[save] FAILED: {e}")
        self._json(200, body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="enable uploading an image and running a model from the viewer")
    ap.add_argument("--models-dir", action="append", default=None,
                    help="extra folder with .pt checkpoints (default: dw_run/ckpt and models/)")
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    if a.live:
        import live
        dirs = live.MODEL_DIRS + [os.path.abspath(d) for d in (a.models_dir or [])]
        LIVE = live.Live(dirs)
        env = live.environment()
        print(f"[live] ON  - PyTorch {env['torch'] or 'NOT installed'} · "
              f"GPU: {env['device'] if env['cuda'] else 'none (CPU, slow)'} · "
              f"HF_TOKEN {'set' if env['hf_token'] else 'not set (needed for DINOv3)'}")
        for m in live.list_models(dirs):
            print(f"[live] model: {m['label']}")
        print(f"[live] open http://localhost:{a.port}/?assets=assets_live")
    print(f"serving {ROOT} on http://localhost:{a.port}  (POST /save -> {SHOTS})")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
