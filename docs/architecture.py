"""
docs/architecture.py - draws docs/architecture.svg, the slide diagram.
The full technical version is docs/architecture_detailed.py.

Numbers are from the code and real runs: RMSE on the 40 held-out GAMUS test
tiles (RESULTS.md / the 1,500-tile Colab run), head size counted from
train_head.py, run time from results/chungthang_dinov3_SUMMARY.txt.
Re-run: python docs/architecture.py
"""
import os

W, H = 1600, 790
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.svg")
# restrained palette: navy structure, neutral greys, one copper accent
INK, DIM, LINE = "#1f2933", "#52606d", "#9aa5b1"
NAVY, COPPER, SOFT, MODEL = "#1f3a5f", "#b5651d", "#f5f4f0", "#e6ecf3"
s = []
add = s.append


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, n, title, lines, fill="#fff"):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
        f'stroke="{NAVY}" stroke-width="1.6"/>')
    add(f'<circle cx="{x + 30}" cy="{y + 34}" r="16" fill="{NAVY}"/>')
    add(f'<text x="{x + 30}" y="{y + 40.5}" text-anchor="middle" font-size="18" '
        f'font-weight="800" fill="#fff">{n}</text>')
    add(f'<text x="{x + 58}" y="{y + 42}" font-size="22" font-weight="800" fill="{INK}">{esc(title)}</text>')
    for i, ln in enumerate(lines):
        add(f'<text x="{x + 24}" y="{y + 78 + i * 23}" font-size="15.5" fill="{DIM}">{esc(ln)}</text>')


def chip(x, y, w, title, sub, fill="#fff"):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="62" rx="8" fill="{fill}" stroke="{LINE}" stroke-width="1.2"/>')
    add(f'<text x="{x + 16}" y="{y + 26}" font-size="15.5" font-weight="700" fill="{INK}">{esc(title)}</text>')
    add(f'<text x="{x + 16}" y="{y + 48}" font-size="13.5" fill="{DIM}">{esc(sub)}</text>')


def arrow(pts, width=2.6):
    d = "M" + " L".join(f"{a},{b}" for a, b in pts)
    add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="{width}" marker-end="url(#ah)"/>')


def line(pts):
    d = "M" + " L".join(f"{a},{b}" for a, b in pts)
    add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="2.6"/>')


def stat(x, y, w, big, small):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="92" rx="8" fill="{SOFT}" stroke="{LINE}" stroke-width="1"/>')
    add(f'<rect x="{x}" y="{y}" width="6" height="92" rx="3" fill="{COPPER}"/>')
    add(f'<text x="{x + 24}" y="{y + 44}" font-size="30" font-weight="800" fill="{NAVY}">{esc(big)}</text>')
    add(f'<text x="{x + 24}" y="{y + 72}" font-size="15.5" fill="{INK}">{esc(small)}</text>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#fff"/>')
add(f'<text x="40" y="52" font-size="30" font-weight="800" fill="{INK}">DepthWizard: one satellite '
    'image → 3D elevation map</text>')

# ---- training strip (offline, once) -> feeds the Height AI
TY = 88
add(f'<rect x="40" y="{TY}" width="1520" height="96" rx="10" fill="{SOFT}" stroke="{LINE}" stroke-width="1"/>')
add(f'<text x="60" y="{TY + 26}" font-size="13" font-weight="700" fill="{DIM}" '
    'letter-spacing="1">TRAINING · OFFLINE, ONCE</text>')
cy, cw = TY + 30, 300
cx = [300, 630, 960, 1250]
chip(cx[0], cy - 14, cw, "GAMUS: 1,500 tiles", "US aerial RGB + LiDAR heights, 0.3 m")
chip(cx[1], cy - 14, cw, "DINOv3 ViT-L (frozen)", "pre-trained on 493 M satellite images", fill=MODEL)
chip(cx[2], cy - 14, 260, "Height head (trained)", "12.7 M parameters, 30 epochs", fill=MODEL)
chip(cx[3], cy - 14, 290, "best.pt", "picked on validation, never on test")
for a, wa in [(0, cw), (1, cw), (2, 260)]:
    arrow([(cx[a] + wa, cy + 17), (cx[a + 1] - 4, cy + 17)], width=2)

# ---- pipeline
Y1, Y2, BH = 230, 395, 140
MID = (Y1 + Y2 + BH) / 2
box(40, MID - 80, 300, 160, 1, "Satellite image", ["any RGB GeoTIFF", "keeps its map position",
                                                   "demo: Sikkim, 0.3 m pixels"])
box(410, Y1, 360, BH, 2, "Height AI", ["scale to 0.5 m · 608 px tiles, 25% overlap",
                                       "flip averaging for stable results",
                                       "→ height of every pixel, metres"], fill=MODEL)
box(410, Y2, 360, BH, 3, "Terrain", ["Copernicus 30 m elevation map (auto)",
                                     "or ISRO CartoDEM",
                                     "reprojected onto the image grid"])
box(850, MID - 80, 320, 160, 4, "Elevation map", ["DSM = terrain + heights",
                                                   "dsm · ndsm · dem GeoTIFFs",
                                                   "opens in QGIS / ArcGIS"])
box(1250, MID - 80, 310, 160, 5, "3D fly-through", ["probe heights · measure",
                                                     "height profiles · flood levels",
                                                     "building floors · video tour"])
arrow([(340, MID), (375, MID), (375, Y1 + BH / 2), (406, Y1 + BH / 2)])
arrow([(375, MID), (375, Y2 + BH / 2), (406, Y2 + BH / 2)])
arrow([(770, Y1 + BH / 2), (810, Y1 + BH / 2), (810, MID), (846, MID)])
line([(770, Y2 + BH / 2), (810, Y2 + BH / 2), (810, MID)])        # joins the arrow above
arrow([(1170, MID), (1246, MID)])
# trained weights -> Height AI
arrow([(cx[3] + 145, cy + 48), (cx[3] + 145, TY + 118), (590, TY + 118), (590, Y1 - 4)], width=2)
add(f'<text x="600" y="{TY + 112}" font-size="13" fill="{DIM}" font-style="italic">trained weights</text>')

# ---- key numbers
SY = 610
stat(40, SY, 500, "4.69 m error", "vs 6.74 m for RS3DAda, measured on the same tiles: 30% lower")
stat(560, SY, 480, "Forest error halved", "11.9 m → 6.0 m; better on urban & sparse too")
stat(1060, SY, 500, "14 s per scene", "625 × 625 m, end to end, laptop GPU")
add(f'<text x="40" y="{H - 22}" font-size="13" fill="{DIM}">Error = RMSE against LiDAR on 40 '
    'held-out test tiles.</text>')
add("</svg>")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote", OUT)
