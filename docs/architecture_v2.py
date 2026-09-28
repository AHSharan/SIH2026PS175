"""
docs/architecture_v2.py - slide diagram, version 2: fewer words, more pictures.

  (a) the data                  photo cards with label: value pairs
  (c) how an image becomes 3D   a filmstrip of REAL pictures, one per step
  (d) results                   one big number + a bar chart + checks
  (b) terrain, placement, files the geographic side

Every picture is a real file (docs/thumbs/ via arch_thumbs.py + arch_features.py,
viewer captures in shots/). Every number is measured - sources in the footer.

    python docs/architecture_v2.py      -> docs/architecture_v2.svg
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from architecture_panels import img_uri, content_bbox, fit_crop, esc   # noqa: E402

from PIL import Image                                                  # noqa: E402

TH = os.path.join(HERE, "thumbs")
SHOTS = os.path.join(os.path.dirname(HERE), "shots")
OUT = os.path.join(HERE, "architecture_v2.svg")

W, H = 1860, 1180
INK, DIM, MUTE = "#1c1c1c", "#3a3a3a", "#6b6b6b"
C_IN, C_TR, C_OUT = "#dbe8f7", "#fde5cc", "#d8efdf"      # input / training / output
A_IN, A_TR, A_OUT = "#1f4e8c", "#c0661a", "#2f7d4f"      # arrow colours
FONT = "Arial, Helvetica, sans-serif"
s = []
add = s.append


def image(x, y, w, h, uri):
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}" '
        f'preserveAspectRatio="xMidYMid slice"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#8f8f8f" '
        f'stroke-width="1"/>')


def panel(x, y, w, h, caption):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="none" '
        f'stroke="#3a3a3a" stroke-width="2" stroke-dasharray="10 7"/>')
    add(f'<text x="{x + w / 2}" y="{y + h - 14}" text-anchor="middle" font-size="19" '
        f'font-weight="700" fill="{INK}">{esc(caption)}</text>')


def card(x, y, w, h):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" '
        f'stroke="#9a9a9a" stroke-width="1.3" stroke-dasharray="2 4"/>')


def kv(x, y, rows, fs=14, lead=22):
    """label: value pairs - the label bold, the value plain, one per line."""
    for i, (k, v) in enumerate(rows):
        add(f'<text x="{x}" y="{y + i * lead}" font-size="{fs}" fill="{DIM}">'
            f'<tspan font-weight="700" fill="{INK}">{esc(k)}:</tspan> {esc(v)}</text>')


def kvbox(x, y, w, h, title, rows, fill, fs=14, lead=22):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
        f'stroke="#2b2b2b" stroke-width="1.2"/>')
    add(f'<text x="{x + 14}" y="{y + 27}" font-size="16.5" font-weight="700" '
        f'fill="{INK}">{esc(title)}</text>')
    kv(x + 14, y + 54, rows, fs, lead)


def badge(cx, cy, n, colour=A_IN):
    add(f'<circle cx="{cx}" cy="{cy}" r="16" fill="{colour}" stroke="#fff" stroke-width="2.5"/>')
    add(f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" font-size="17" font-weight="800" '
        f'fill="#fff">{n}</text>')


def arrow(pts, colour, width=3.2, dash=None, label=None, lx=0, ly=0, anchor="start"):
    d = "M " + " L ".join(f"{px} {py}" for px, py in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    mid = {A_IN: "ai", A_TR: "at", A_OUT: "ao"}[colour]
    add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{da} '
        f'marker-end="url(#{mid})"/>')
    if label:
        add(f'<text x="{lx}" y="{ly}" font-size="13.5" font-style="italic" font-weight="700" '
            f'text-anchor="{anchor}" fill="{colour}">{esc(label)}</text>')


def data_card(x, y, thumb, title, rows, fill, n=None):
    card(x, y, 420, 214)
    image(x + 14, y + 14, 180, 180, thumb)
    if n:
        badge(x + 30, y + 30, n)
    kvbox(x + 206, y + 14, 200, 180, title, rows, fill, fs=13.2, lead=34)


def chart(x, y, w, h):
    """RMSE by landscape, ours vs RS3DAda (480-tile model, same 40 tiles)."""
    data = [("Urban", 5.28, 5.11), ("Sparse", 2.15, 2.83), ("Forest", 5.91, 11.87)]
    add(f'<text x="{x}" y="{y + 18}" font-size="16" font-weight="700" fill="{INK}">'
        f'Error by landscape</text>')
    add(f'<text x="{x}" y="{y + 36}" font-size="12.5" fill="{MUTE}">metres, lower is better</text>')
    px0, py0, pw, ph = x + 34, y + 58, w - 44, h - 98
    ymax = 12.0
    for g in (0, 4, 8, 12):
        gy = py0 + ph - g / ymax * ph
        add(f'<line x1="{px0}" y1="{gy}" x2="{px0 + pw}" y2="{gy}" stroke="#e1e1e1"/>')
        add(f'<text x="{px0 - 8}" y="{gy + 4}" text-anchor="end" font-size="12" '
            f'fill="{MUTE}">{g}</text>')
    gw = pw / len(data)
    bw = 42
    for i, (name, ours, base) in enumerate(data):
        gx = px0 + i * gw + gw / 2
        for j, (val, col) in enumerate(((ours, A_OUT), (base, "#a4acb6"))):
            bx = gx - bw - 3 + j * (bw + 6)
            bh = val / ymax * ph
            add(f'<rect x="{bx}" y="{py0 + ph - bh}" width="{bw}" height="{bh}" rx="3" '
                f'fill="{col}"/>')
            add(f'<text x="{bx + bw / 2}" y="{py0 + ph - bh - 6}" text-anchor="middle" '
                f'font-size="12.5" font-weight="700" fill="{INK}">{val:.2f}</text>')
        add(f'<text x="{gx}" y="{py0 + ph + 20}" text-anchor="middle" font-size="13.5" '
            f'font-weight="700" fill="{INK}">{name}</text>')
    lx = x + w - 188
    add(f'<rect x="{lx}" y="{y + 6}" width="14" height="14" rx="2" fill="{A_OUT}"/>')
    add(f'<text x="{lx + 20}" y="{y + 18}" font-size="13" fill="{INK}">Ours</text>')
    add(f'<rect x="{lx + 72}" y="{y + 6}" width="14" height="14" rx="2" fill="#a4acb6"/>')
    add(f'<text x="{lx + 92}" y="{y + 18}" font-size="13" fill="{INK}">RS3DAda</text>')


def main():
    t = {n: img_uri(os.path.join(TH, n + ".jpg"), max_w=420)
         for n in ("gamus_rgb", "gamus_lidar", "india_rgb", "india_dem", "india_ndsm",
                   "india_dsm", "india_tiles", "india_features")}
    shot = {}
    for n, asp, mw in (("india_view_default", 1.0, 520), ("india_03_buildings_floors", 16 / 9, 360),
                       ("india_04_flood", 16 / 9, 360)):
        p = os.path.join(SHOTS, n + ".jpg")
        shot[n] = img_uri(p, crop=fit_crop(content_bbox(p), asp, Image.open(p).size), max_w=mw)

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="{FONT}">')
    add("<defs>")
    for mid, col in (("ai", A_IN), ("at", A_TR), ("ao", A_OUT)):
        add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="5.5" '
            f'markerHeight="5.5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" '
            f'fill="{col}"/></marker>')
    add("</defs>")
    add(f'<rect width="{W}" height="{H}" fill="#fff"/>')
    add(f'<text x="40" y="42" font-size="28" font-weight="700" fill="{INK}">'
        f'DepthWizard: System Architecture</text>')
    # legend
    lx, ly = 1020, 22
    for i, (col, name) in enumerate(((C_IN, "input"), (C_TR, "training"), (C_OUT, "output"))):
        add(f'<rect x="{lx + i * 112}" y="{ly}" width="18" height="18" rx="3" fill="{col}" '
            f'stroke="#2b2b2b" stroke-width="1"/>')
        add(f'<text x="{lx + i * 112 + 26}" y="{ly + 14}" font-size="14" fill="{INK}">{name}</text>')
    add(f'<line x1="{lx + 350}" y1="{ly + 9}" x2="{lx + 396}" y2="{ly + 9}" stroke="{A_IN}" '
        f'stroke-width="3.2" marker-end="url(#ai)"/>')
    add(f'<text x="{lx + 404}" y="{ly + 14}" font-size="14" fill="{INK}">image → 3D</text>')
    add(f'<line x1="{lx + 506}" y1="{ly + 9}" x2="{lx + 552}" y2="{ly + 9}" stroke="{A_TR}" '
        f'stroke-width="2.6" stroke-dasharray="7 5" marker-end="url(#at)"/>')
    add(f'<text x="{lx + 560}" y="{ly + 14}" font-size="14" fill="{INK}">training</text>')
    add(f'<line x1="{lx + 642}" y1="{ly + 9}" x2="{lx + 688}" y2="{ly + 9}" stroke="{A_OUT}" '
        f'stroke-width="3" marker-end="url(#ao)"/>')
    add(f'<text x="{lx + 696}" y="{ly + 14}" font-size="14" fill="{INK}">geo</text>')

    # ---------------------------------------------------------------- (a)
    panel(40, 62, 1330, 288, "(a) Data")
    data_card(58, 76, t["india_rgb"], "Indian input",
              [("Source", "WorldView-2"), ("Place", "Chungthang, Sikkim"),
               ("Pixel size", "0.31 m"), ("Formats", "GeoTIFF, PNG, JPG")], C_IN, n=1)
    data_card(498, 76, t["gamus_rgb"], "Training photos",
              [("Dataset", "GAMUS"), ("Tiles", "1,500"), ("Pixel size", "0.3 m"),
               ("Cities", "DC, NYC, PHL")], C_TR)
    data_card(938, 76, t["gamus_lidar"], "Laser heights",
              [("Source", "airborne LiDAR"), ("Measures", "true heights"),
               ("Range", "0 – 146 m"), ("Role", "ground truth")], C_TR)

    # ---------------------------------------------------------------- (c)
    panel(40, 368, 1330, 420, "(c) How an Image Becomes 3D")
    frames = [
        (t["india_tiles"], 2, "Split into tiles", [("Tile", "608 px"), ("Overlap", "25 %")]),
        (t["india_features"], 3, "What the AI sees",
         [("Encoder", "DINOv3, satellite"), ("Colours", "river · town · forest")]),
        (t["india_ndsm"], 4, "Height above ground",
         [("Model", "our trained head"), ("Output", "metres")]),
        (t["india_dsm"], 5, "Full surface model",
         [("Adds", "ground elevation"), ("File", "DSM GeoTIFF")]),
        (shot["india_view_default"], 6, "3D fly-through",
         [("Runs in", "any browser"), ("Time", "12.7 s per scene")]),
    ]
    fx0, fy, fs, step = 68, 400, 214, 262
    for i, (uri, n, title, rows) in enumerate(frames):
        fx = fx0 + i * step
        fill = C_OUT if n >= 4 else C_IN
        image(fx, fy, fs, fs, uri)
        badge(fx + 18, fy + 18, n, A_OUT if n >= 4 else A_IN)
        add(f'<rect x="{fx}" y="{fy + fs + 10}" width="{fs}" height="100" rx="8" fill="{fill}" '
            f'stroke="#2b2b2b" stroke-width="1.1"/>')
        add(f'<text x="{fx + 12}" y="{fy + fs + 36}" font-size="15.5" font-weight="700" '
            f'fill="{INK}">{esc(title)}</text>')
        kv(fx + 12, fy + fs + 62, rows, fs=13.2, lead=22)
        if i < len(frames) - 1:
            arrow([(fx + fs + 6, fy + fs / 2), (fx + step - 8, fy + fs / 2)], A_IN, width=3.6)

    # (a) -> (c): the image goes in, the training data trains the height model
    arrow([(149, 290), (149, 396)], A_IN, width=3.6, label="image in", lx=160, ly=332)
    # both training sources feed step 4 - the height model we train
    arrow([(650, 290), (650, 396)], A_TR, width=2.6, dash="7 5")
    arrow([(1100, 290), (1100, 316), (760, 316), (760, 396)], A_TR, width=2.6, dash="7 5",
          label="trains the height model", lx=1112, ly=309)

    # ---------------------------------------------------------------- (d)
    panel(40, 806, 1330, 344, "(d) Results  —  measured on 40 held-out laser-measured tiles")
    card(58, 822, 360, 288)
    add(f'<text x="80" y="890" font-size="60" font-weight="800" fill="{A_OUT}">4.69 m</text>')
    add(f'<text x="80" y="918" font-size="15" fill="{DIM}">average height error (best model)</text>')
    kv(80, 962, [("Our error", "4.69 m"), ("RS3DAda error", "6.74 m"),
                 ("Improvement", "30 % lower"), ("Correlation", "0.78 vs 0.60")], fs=15, lead=30)
    card(436, 822, 430, 288)
    chart(452, 836, 400, 262)
    card(884, 822, 470, 288)
    image(900, 838, 212, 119, shot["india_03_buildings_floors"])
    image(1126, 838, 212, 119, shot["india_04_flood"])
    add(f'<text x="1006" y="976" text-anchor="middle" font-size="13" fill="{MUTE}">buildings + floors</text>')
    add(f'<text x="1232" y="976" text-anchor="middle" font-size="13" fill="{MUTE}">flood what-if</text>')
    kv(900, 1008, [("Buildings found", "204"), ("Flood at +36 m", "12 buildings reached"),
                   ("PNG vs original", "identical"), ("JPG vs original", "+0.03 m")],
       fs=14.2, lead=24)

    # ---------------------------------------------------------------- (b)
    panel(1400, 62, 420, 1088, "(b) Terrain, Placement, Files")
    card(1418, 76, 384, 300)
    image(1432, 90, 170, 170, t["india_dem"])
    add(f'<text x="1616" y="114" font-size="15" font-weight="700" fill="{INK}">Terrain</text>')
    kv(1616, 140, [("Grid", "30 m"), ("Datum", "EGM2008"), ("Cost", "free, global")],
       fs=13.6, lead=24)
    kvbox(1432, 270, 356, 92, "Ground elevation", [("Source", "Copernicus GLO-30")],
          C_IN, fs=13.6)
    card(1418, 432, 384, 318)
    image(1432, 446, 170, 170, t["india_dsm"])
    add(f'<text x="1616" y="470" font-size="15" font-weight="700" fill="{INK}">Output files</text>')
    kv(1616, 496, [("Surface", "dsm.tif"), ("Heights", "ndsm.tif"), ("Terrain", "dem.tif")],
       fs=13.6, lead=24)
    kvbox(1432, 628, 356, 108, "Absolute DSM",
          [("Here", "1,549 – 1,776 m"), ("Opens in", "QGIS, ArcGIS")], C_OUT, fs=13.6)
    card(1418, 806, 384, 300)
    image(1432, 820, 170, 170, t["india_rgb"])
    for px, py in ((1462, 850), (1552, 862), (1470, 943), (1548, 950)):
        add(f'<circle cx="{px}" cy="{py}" r="7" fill="#d9480f" stroke="#fff" stroke-width="2"/>')
    add(f'<text x="1616" y="844" font-size="15" font-weight="700" fill="{INK}">Map points</text>')
    kv(1616, 870, [("For", "PNG / JPG"), ("Needs", "3+ points"), ("Fit", "robust")],
       fs=13.6, lead=24)
    kvbox(1432, 1000, 356, 92, "Placed on the map",
          [("Result", "full DSM, not only shape")], C_OUT, fs=13.6)

    # geo arrows: terrain into step 5, step 5 out to files, map points up to files
    arrow([(1418, 175), (1386, 175), (1386, 382), (1000, 382), (1000, 396)], A_OUT, width=3)
    arrow([(961, 724), (961, 752), (1386, 752), (1386, 600), (1414, 600)], A_OUT, width=3,
          label="saves", lx=1290, ly=745, anchor="middle")
    arrow([(1610, 806), (1610, 754)], A_OUT, width=3, label="places", lx=1620, ly=786)

    add(f'<text x="40" y="{H - 10}" font-size="12" fill="{MUTE}">All figures measured against '
        'airborne LiDAR on the same 40 held-out GAMUS test tiles. Best model 4.69 m: 1,500 training '
        'tiles. Landscape chart and correlation: 480-tile model. PNG/JPG: tests/png_jpg_accuracy.py. '
        'Imagery: Maxar Open Data (CC BY-NC 4.0). Terrain: Copernicus DEM GLO-30.</text>')
    add("</svg>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(s))
    print("wrote", OUT, f"({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
