"""
docs/limitations.py - draws docs/limitations.svg: known limitations and how we
address them. Kept separate from the architecture diagrams on purpose.
Numbers are from real runs (RESULTS.md, the 1,500-tile Colab run,
results/*_SUMMARY.txt). Re-run: python docs/limitations.py
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limitations.svg")
INK, DIM, LINE = "#1f2933", "#52606d", "#cbd2d9"
NAVY, COPPER, SOFT = "#1f3a5f", "#b5651d", "#f7f6f2"

ROWS = [
    ("Validated on US imagery only",
     ["Accuracy is measured on GAMUS (US aerial, LiDAR truth). India has no open",
      "LiDAR, so the Chungthang output is checked for plausibility, not accuracy."],
     ["Check against NASA GEDI spaceborne LiDAR (it covers India) and",
      "ISRO / NRSC reference DSMs where available."]),
    ("30 m terrain map double-counts",
     ["Copernicus GLO-30 is a surface model: it already holds smoothed buildings",
      "and trees (≈2.4–3.0 m on average at Chungthang). It is also from 2011–15."],
     ["Use a bare-earth DTM (CartoDEM or ground-filtered) via --dem;",
      "the double-count estimate is reported in every run."]),
    ("Coarser images lose accuracy",
     ["RMSE rises from 4.40 m at 0.5 m pixels to 5.63 m at 1.0 m pixels",
      "(the model trains at a fixed 0.5 m)."],
     ["Train across 0.3–1.0 m pixel sizes to match Cartosat products."]),
    ("Forest is the hardest land type",
     ["6.03 m RMSE on forest vs 4.79 m urban and 2.15 m sparse;",
      "the tallest trees are likely under-estimated."],
     ["More training tiles (5,004 available) and unfreezing the last",
      "encoder layers."]),
    ("Building list is an estimate",
     ["Candidates come from predicted heights with a colour/texture tree",
      "filter; floors = height ÷ 3 m; neighbouring roofs can merge or split."],
     ["Add a building-segmentation head or OpenStreetMap footprints."]),
    ("Flood tool is a screening view",
     ["\"Bathtub\" model: everything below the level is wet; no river flow,",
      "connectivity or timing."],
     ["Feed the DSM into a hydraulic model (e.g. HEC-RAS) for planning use."]),
    ("Off-nadir views not modelled",
     ["Tilted satellite views make buildings lean; the model assumes a",
      "near-vertical view."],
     ["Prefer near-nadir scenes; lean correction is future work."]),
    ("Evaluation scope",
     ["40 held-out test tiles and one training run (single seed)."],
     ["Full GAMUS test split (2,861 tiles) and 3 seeds with error bars."]),
    ("Access and hardware",
     ["Our model needs the gated DINOv3 licence and a login; a GPU is",
      "recommended (CPU works but is slow)."],
     ["RS3DAda fallback runs with public weights and no login."]),
]

W = 1600
X0, C1, C2, C3 = 40, 40, 380, 1000          # columns: limitation | detail | what we do
RH, HDR = 64, 150
H = HDR + len(ROWS) * RH + 70
s = []
add = s.append


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
add(f'<rect width="{W}" height="{H}" fill="#fff"/>')
add(f'<text x="{X0}" y="52" font-size="30" font-weight="800" fill="{INK}">Known limitations '
    'and how we address them</text>')
add(f'<text x="{X0}" y="82" font-size="15" fill="{DIM}">What DepthWizard does not do yet, '
    'stated plainly, with the next step for each.</text>')
# header
hy = HDR - 26
add(f'<rect x="{X0}" y="{hy - 22}" width="{W - 2 * X0}" height="34" rx="6" fill="{NAVY}"/>')
for x, t in [(C1 + 16, "Limitation"), (C2, "Detail (measured where possible)"), (C3, "What we do about it")]:
    add(f'<text x="{x}" y="{hy}" font-size="14.5" font-weight="700" fill="#fff">{t}</text>')
y = HDR
for i, (title, detail, fix) in enumerate(ROWS):
    if i % 2 == 0:
        add(f'<rect x="{X0}" y="{y - 6}" width="{W - 2 * X0}" height="{RH}" fill="{SOFT}"/>')
    add(f'<rect x="{X0}" y="{y - 6}" width="5" height="{RH}" fill="{COPPER}"/>')
    add(f'<text x="{C1 + 16}" y="{y + 20}" font-size="15" font-weight="700" fill="{INK}">'
        f'{i + 1}. {esc(title)}</text>')
    for k, ln in enumerate(detail):
        add(f'<text x="{C2}" y="{y + 20 + k * 21}" font-size="14" fill="{DIM}">{esc(ln)}</text>')
    for k, ln in enumerate(fix):
        add(f'<text x="{C3}" y="{y + 20 + k * 21}" font-size="14" fill="{INK}">{esc(ln)}</text>')
    y += RH
add(f'<line x1="{X0}" y1="{y - 6}" x2="{W - X0}" y2="{y - 6}" stroke="{LINE}"/>')
add(f'<text x="{X0}" y="{H - 24}" font-size="13" fill="{DIM}">RMSE = error against LiDAR on 40 '
    'held-out GAMUS test tiles. Double-count values from the Chungthang runs (RS3DAda 2.36 m, '
    'ours, 480-tile model 2.95 m).</text>')
add("</svg>")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote", OUT, H)
