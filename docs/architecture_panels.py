"""
docs/architecture_panels.py - the slide architecture diagram in the grouped-panel
style: (a) data, (b) terrain + DSM assembly, (c) model, evaluation and 3D viewer.

Every picture is a real file (docs/thumbs/ from docs/arch_thumbs.py, plus 3D
viewer captures in shots/). Every number is measured: RESULTS.md (480-tile
run), the 1,500-tile run (4.69 m, DSM.md), train_head.py (head size).

    python docs/arch_thumbs.py --run <dsm.py output folder>
    python docs/architecture_panels.py          -> docs/architecture_panels.svg
"""
import base64
import io
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TH = os.path.join(HERE, "thumbs")
SHOTS = os.path.join(ROOT, "shots")
OUT = os.path.join(HERE, "architecture_panels.svg")

W, H = 1860, 1150
INK, DIM = "#1f1f1f", "#3d3d3d"
BLUE, BEIGE, GREY = "#dce8f6", "#f2e8cf", "#efefef"
FONT = "Arial, Helvetica, sans-serif"
s = []
add = s.append


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def img_uri(path, crop=None, max_w=900, quality=86):
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def content_bbox(path, pad=18):
    """Crop a viewer capture to the terrain: drop the flat dark background."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    mask = a.max(-1) > 85            # textured terrain; the dark slab + sky stay out
    ys, xs = np.where(mask)
    y0, y1 = np.percentile(ys, 0.5), np.percentile(ys, 99.5)
    x0, x1 = np.percentile(xs, 0.5), np.percentile(xs, 99.5)
    return (max(0, int(x0) - pad), max(0, int(y0) - pad),
            min(a.shape[1], int(x1) + pad), min(a.shape[0], int(y1) + pad))


def fit_crop(bbox, aspect, limit):
    """Grow bbox to the target aspect ratio, clamped to the image."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if w / h > aspect:
        nh = w / aspect
        cy = (y0 + y1) / 2
        y0, y1 = cy - nh / 2, cy + nh / 2
    else:
        nw = h * aspect
        cx = (x0 + x1) / 2
        x0, x1 = cx - nw / 2, cx + nw / 2
    W0, H0 = limit
    dx = max(0, -x0) - max(0, x1 - W0)
    dy = max(0, -y0) - max(0, y1 - H0)
    return (int(max(0, x0 + dx)), int(max(0, y0 + dy)),
            int(min(W0, x1 + dx)), int(min(H0, y1 + dy)))


def image(x, y, w, h, uri, stroke=True):
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}" '
        f'preserveAspectRatio="xMidYMid slice"/>')
    if stroke:
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" '
            f'stroke="#9a9a9a" stroke-width="1"/>')


def panel(x, y, w, h, caption):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="none" '
        f'stroke="#3a3a3a" stroke-width="2" stroke-dasharray="10 7"/>')
    add(f'<text x="{x + w / 2}" y="{y + h - 16}" text-anchor="middle" font-size="19" '
        f'font-weight="700" fill="{INK}">{esc(caption)}</text>')


def card(x, y, w, h, caption=None, left=False):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" '
        f'stroke="#8a8a8a" stroke-width="1.3" stroke-dasharray="2 4"/>')
    if caption:
        cx, anchor = (x + 18, "start") if left else (x + w / 2, "middle")
        add(f'<text x="{cx}" y="{y + h - 12}" text-anchor="{anchor}" font-size="15.5" '
            f'font-weight="700" fill="{INK}">{esc(caption)}</text>')


def box(x, y, w, h, title, bullets, fill=BLUE, fs=14.2, lead=21):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
        f'stroke="#2b2b2b" stroke-width="1.3"/>')
    add(f'<text x="{x + w / 2}" y="{y + 26}" text-anchor="middle" font-size="16.5" '
        f'font-weight="700" fill="{INK}">{esc(title)}</text>')
    for i, b in enumerate(bullets):
        add(f'<text x="{x + 14}" y="{y + 52 + i * lead}" font-size="{fs}" fill="{DIM}">'
            f'&#8226;  {esc(b)}</text>')


def arrow(pts, label=None, lx=0, ly=0, anchor="start"):
    d = "M " + " L ".join(f"{px} {py}" for px, py in pts)
    add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="2.2" '
        f'marker-end="url(#ah)"/>')
    if label:
        add(f'<text x="{lx}" y="{ly}" font-size="13.5" font-style="italic" '
            f'text-anchor="{anchor}" fill="{DIM}">{esc(label)}</text>')


def data_card(x, y, thumb, title, bullets, caption, w=420):
    card(x, y, w, 250, caption)
    image(x + 14, y + 14, 166, 166, thumb)
    box(x + 194, y + 14, w - 208, 166, title, bullets)


def main():
    t = {n: img_uri(os.path.join(TH, n + ".jpg"), max_w=420)
         for n in ("gamus_rgb", "gamus_lidar", "india_rgb", "india_dem", "india_ndsm", "india_dsm")}
    shots = {}
    for n in ("india_view_default", "india_03_buildings_floors", "india_04_flood"):
        p = os.path.join(SHOTS, n + ".jpg")
        im = Image.open(p)
        bb = fit_crop(content_bbox(p), 16 / 9, im.size)
        shots[n] = img_uri(p, crop=bb, max_w=1200 if n == "india_view_default" else 520)

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="{FONT}">')
    add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" '
        f'fill="{INK}"/></marker></defs>')
    add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    add(f'<text x="{W / 2}" y="40" text-anchor="middle" font-size="27" font-weight="700" '
        f'fill="{INK}">DepthWizard: Proposed System Architecture</text>')

    # ------------------------------------------------------------ (a) data
    panel(40, 60, 1330, 322, "(a) Data Acquisition")
    data_card(58, 76, t["india_rgb"], "Indian input",
              ["WorldView-2, Sikkim", "GeoTIFF / PNG / JPG", "0.31 m, 625 × 625 m",
               "single image only"], "Satellite image (inference)")
    data_card(498, 76, t["gamus_rgb"], "GAMUS aerial",
              ["1,500 training tiles", "0.3 m RGB", "DC · NYC · PHL", "CC-BY-4.0"],
              "Aerial images (training)")
    data_card(938, 76, t["gamus_lidar"], "LiDAR heights",
              ["metres above ground", "0 – 146 m", "airborne laser", "ground truth"],
              "Height labels (training)")

    # ----------------------------------------------- (c) model, eval, viewer
    panel(40, 400, 1330, 732, "(c) Height Model, Evaluation and 3D Visualisation")
    card(58, 418, 1294, 238, "Deep-learning height estimator  ·  12.7 s per 625 × 625 m scene on a laptop GPU", left=True)
    box(76, 436, 236, 170, "Scale normalisation",
        ["any pixel size → 0.5 m", "608-px tiles, 25 % overlap", "flip averaging (TTA)",
         "Gaussian-blended seams"])
    box(372, 436, 262, 170, "DINOv3 ViT-L/16",
        ["satellite-pretrained", "(493 M images, SAT-493M)", "frozen encoder",
         "4 feature levels"], fill=GREY)
    box(694, 436, 262, 170, "DPT height head (ours)",
        ["12.7 M trained parameters", "masked Huber loss, metres", "trained on LiDAR heights",
         "outputs metres directly"], fill=BEIGE)
    image(1016, 436, 170, 170, t["india_ndsm"])
    add(f'<text x="1101" y="628" text-anchor="middle" font-size="13.5" fill="{DIM}">'
        f'predicted height above ground</text>')
    add(f'<text x="1101" y="646" text-anchor="middle" font-size="13.5" fill="{DIM}">'
        f'(nDSM, Chungthang)</text>')
    arrow([(312, 521), (368, 521)])
    arrow([(634, 521), (690, 521)])
    arrow([(956, 521), (1012, 521)])

    # data -> model
    arrow([(194, 326), (194, 432)], "input image", 204, 356)
    arrow([(566, 326), (566, 432)], "training images", 556, 356, "end")
    arrow([(1006, 326), (1006, 372), (824, 372), (824, 432)], "LiDAR height labels", 834, 366)

    # evaluation library
    card(58, 676, 620, 406, "Evaluation library (measured, 40 held-out LiDAR tiles)")
    box(76, 694, 584, 108, "Accuracy vs RS3DAda",
        ["RMSE 4.69 m (ours)  vs  6.74 m (RS3DAda), same tiles: 30 % lower",
         "correlation with LiDAR 0.78 (ours)  vs  0.60"])
    box(76, 812, 584, 108, "Per landscape",
        ["forest RMSE 11.87 → 5.91 m (halved)",
         "sparse RMSE 2.83 → 2.15 m;  bias −2.27 → −0.52 m"])
    box(76, 930, 584, 108, "Robustness to image resolution",
        ["input 0.3 → 1.0 m pixel size (Cartosat-like)",
         "ours 4.60 – 5.46 m  vs  RS3DAda 6.70 – 6.92 m"])
    arrow([(825, 610), (825, 664), (368, 664), (368, 690)], "evaluate", 380, 684)

    # viewer
    card(698, 676, 654, 406, "Interactive 3D fly-through (Three.js, runs in a browser)")
    image(716, 694, 460, 259, shots["india_view_default"])
    image(1190, 694, 146, 82, shots["india_03_buildings_floors"])
    add(f'<text x="1263" y="792" text-anchor="middle" font-size="12.5" fill="{DIM}">'
        f'204 buildings, floors</text>')
    image(1190, 806, 146, 82, shots["india_04_flood"])
    add(f'<text x="1263" y="904" text-anchor="middle" font-size="12.5" fill="{DIM}">'
        f'flood what-if</text>')
    box(716, 962, 620, 84, "Explore and check",
        ["probe · measure · height profile · model vs LiDAR truth · error map",
         "upload PNG / JPG / GeoTIFF and run the model live"], fill=BEIGE, fs=13.6)

    # ---------------------------------------------- (b) terrain + DSM
    panel(1400, 60, 420, 1072, "(b) Terrain and DSM Assembly")
    card(1418, 76, 384, 250, "Terrain (DEM)")
    image(1432, 90, 150, 150, t["india_dem"])
    box(1596, 90, 192, 150, "Copernicus GLO-30",
        ["30 m terrain", "EGM2008 datum", "free, global"], fs=13.6)
    card(1418, 416, 384, 250, "DSM = DEM + nDSM")
    image(1432, 430, 150, 150, t["india_dsm"])
    box(1596, 430, 192, 150, "Absolute DSM",
        ["metres above sea level", "GeoTIFF, CRS kept", "1,549 – 1,776 m here"], fs=13.6)
    card(1418, 756, 384, 250, "Map points (optional)")
    image(1432, 770, 150, 150, t["india_rgb"])
    for px, py in ((1462, 800), (1552, 812), (1470, 893), (1548, 900)):
        add(f'<circle cx="{px}" cy="{py}" r="7" fill="#d9480f" stroke="#fff" stroke-width="2"/>')
    box(1596, 770, 192, 150, "Place PNG / JPG",
        ["3+ map points", "robust fit (RANSAC)", "→ full DSM"], fs=13.6)
    arrow([(1610, 326), (1610, 412)], "terrain", 1620, 374)
    arrow([(1610, 756), (1610, 670)], "georeference", 1620, 718)

    # model -> DSM -> viewer
    arrow([(1186, 521), (1414, 521)], "nDSM", 1300, 512, "middle")
    arrow([(1418, 600), (1384, 600), (1384, 824), (1356, 824)])

    add(f'<text x="40" y="{H - 4}" font-size="12" fill="#6b6b6b">All accuracy figures are measured '
        'against airborne LiDAR on 40 held-out GAMUS test tiles (headline 4.69 m: 1,500-tile model; '
        'per-landscape and resolution: 480-tile model). Imagery: Maxar Open Data (CC BY-NC 4.0). '
        'Terrain: Copernicus DEM GLO-30.</text>')
    add("</svg>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(s))
    print("wrote", OUT, f"({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
