"""
docs/architecture.py - draws docs/architecture.svg, the one-glance slide
diagram. The full technical version is docs/architecture_detailed.py.

Numbers are from real runs: RMSE on the 40 held-out GAMUS test tiles
(RESULTS.md / the 1,500-tile Colab run), run time from
results/chungthang_dinov3_SUMMARY.txt. Re-run: python docs/architecture.py
"""
import os

W, H = 1600, 600
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
        add(f'<text x="{x + 24}" y="{y + 80 + i * 24}" font-size="16.5" fill="{DIM}">{esc(ln)}</text>')


def arrow(pts):
    d = "M" + " L".join(f"{a},{b}" for a, b in pts)
    add(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="2.6" marker-end="url(#ah)"/>')


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

# ---- pipeline
Y1, Y2, BH = 90, 250, 140
box(40, 150, 290, BH, 1, "Satellite image", ["any RGB GeoTIFF", "demo: Sikkim, 0.3 m pixels"])
box(410, Y1, 350, BH, 2, "Height AI", ["DINOv3 (frozen) + our trained head",
                                            "→ height of every pixel, metres"], fill=MODEL)
box(410, Y2, 350, BH, 3, "Terrain", ["Copernicus 30 m elevation map",
                                           "(or ISRO CartoDEM)"])
box(840, 150, 330, BH, 4, "Elevation map", ["terrain + heights = DSM",
                                                  "standard GeoTIFF (QGIS/ArcGIS)"])
box(1250, 150, 310, BH, 5, "3D fly-through", ["probe heights · measure",
                                                     "flood levels · video tour"])
arrow([(330, 220), (370, 220), (370, Y1 + BH / 2), (406, Y1 + BH / 2)])
arrow([(370, 220), (370, Y2 + BH / 2), (406, Y2 + BH / 2)])
arrow([(760, Y1 + BH / 2), (800, Y1 + BH / 2), (800, 220), (836, 220)])
add(f'<path d="M760,{Y2 + BH / 2} L800,{Y2 + BH / 2} L800,220" fill="none" stroke="{INK}" '
    'stroke-width="2.6"/>')                      # joins the arrow above: no head
arrow([(1170, 220), (1246, 220)])
add(f'<text x="585" y="{Y1 - 12}" text-anchor="middle" font-size="14.5" fill="{DIM}" '
    'font-style="italic">trained once on 1,500 US tiles with LiDAR heights</text>')

# ---- key numbers
SY = 440
stat(40, SY, 500, "4.69 m error", "vs 6.74 m for RS3DAda (NeurIPS 2024): 30% lower")
stat(560, SY, 480, "Forest error halved", "11.9 m → 6.0 m; better on urban & sparse too")
stat(1060, SY, 500, "14 s per scene", "625 × 625 m, end to end, laptop GPU")
add(f'<text x="40" y="{H - 18}" font-size="13" fill="{DIM}">Error = RMSE against LiDAR on 40 '
    'held-out test tiles.</text>')
add("</svg>")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote", OUT)
