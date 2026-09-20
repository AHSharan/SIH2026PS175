"""Static server for web/ plus a POST /save endpoint so the viewer can write
high-resolution captures straight to disk (shots/). Local only."""
import base64
import json
import os
import re
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
os.makedirs(SHOTS, exist_ok=True)


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n))
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", payload["name"])[:80]
            url = payload["dataurl"]
            raw = base64.b64decode(url.split(",", 1)[1])
            ext = ".jpg" if "jpeg" in url[:30] else ".png"
            path = os.path.join(SHOTS, name + ext)
            with open(path, "wb") as f:
                f.write(raw)
            body = json.dumps({"ok": True, "path": path, "bytes": len(raw)}).encode()
            print(f"[save] {name}{ext}  {len(raw)/1024:.0f} KB")
        except Exception as e:                      # noqa: BLE001
            body = json.dumps({"ok": False, "error": str(e)}).encode()
            print(f"[save] FAILED: {e}")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving {ROOT} on http://localhost:8777  (POST /save -> {SHOTS})")
    ThreadingHTTPServer(("127.0.0.1", 8777), H).serve_forever()
