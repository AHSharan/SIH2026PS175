"""
docs/architecture_detailed.py - draws docs/architecture_detailed.svg (full technical diagram; the slide version is docs/architecture.py).

Every number here comes from the code or a real run (see RESULTS.md,
results/*.txt, train_head.py, dsm.py). Edit the text below and re-run:
    python docs/architecture_detailed.py
"""
import os

W, H = 1800, 1215
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_detailed.svg")

# palette: one tint per stage + neutral ink
# restrained palette: navy structure, warm-neutral panels, one copper accent
INK, DIM, LINE = "#1f2933", "#52606d", "#9aa5b1"
NAVY, COPPER = "#1f3a5f", "#b5651d"
TRAIN = INFER = VIEW = RES = ("#f7f6f2", NAVY)      # (fill, accent)
OURS = "#e6ecf3"

svg = []
add = svg.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, title, lines, fill="#ffffff", stroke=LINE, num=None,
        accent=None, dashed=False, tsize=15, bsize=12.5, lh=17):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.4"{dash}/>')
    tx = x + 14
    if num is not None:
        add(f'<circle cx="{x + 18}" cy="{y + 20}" r="12" fill="{accent or INK}"/>')
        add(f'<text x="{x + 18}" y="{y + 24.5}" text-anchor="middle" font-size="13" '
            f'font-weight="700" fill="#fff">{num}</text>')
        tx = x + 38
    add(f'<text x="{tx}" y="{y + 25}" font-size="{tsize}" font-weight="700" '
        f'fill="{INK}">{esc(title)}</text>')
    for i, ln in enumerate(lines):
        weight, col, txt = "400", DIM, ln
        if ln.startswith("**"):
            weight, col, txt = "700", INK, ln[2:]
        add(f'<text x="{x + 14}" y="{y + 48 + i * lh}" font-size="{bsize}" '
            f'font-weight="{weight}" fill="{col}">{esc(txt)}</text>')


def arrow(pts, label=None, lx=None, ly=None, dashed=False, color=INK):
    d = "M" + " L".join(f"{a},{b}" for a, b in pts)
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dash} '
        f'marker-end="url(#ah)"/>')
    if label:
        add(f'<text x="{lx}" y="{ly}" font-size="11.5" fill="{DIM}" '
            f'font-style="italic">{esc(label)}</text>')


def band(x, y, w, h, tint, title, sub):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{tint[0]}" '
        f'stroke="{tint[1]}" stroke-width="1.6"/>')
    add(f'<rect x="{x}" y="{y}" width="7" height="{h}" rx="3" fill="{tint[1]}"/>')
    add(f'<text x="{x + 22}" y="{y + 30}" font-size="18" font-weight="800" '
        f'fill="{tint[1]}">{esc(title)}</text>')
    add(f'<text x="{x + 22}" y="{y + 50}" font-size="12.5" fill="{DIM}">{esc(sub)}</text>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/>'
    '</marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ---------------------------------------------------------------- title
add(f'<text x="30" y="44" font-size="26" font-weight="800" fill="{INK}">DepthWizard — '
    'system architecture</text>')
add(f'<text x="30" y="70" font-size="14.5" fill="{DIM}">One optical RGB image → height above '
    'ground (nDSM, metres) → absolute DSM GeoTIFF (DSM = DEM + nDSM) → interactive 3D. '
    'Numbered circles = processing order.</text>')

# ================================================================ A. TRAINING
AX, AY, AW, AH = 30, 90, 1350, 260
band(AX, AY, AW, AH, TRAIN, "A · Training (offline, once)",
     "Frozen satellite foundation model + small trained height head. Model selection on validation only.")
bw, bh, by, gap = 200, 170, AY + 70, 25
bx = [AX + 22 + k * (bw + gap) for k in range(6)]
box(bx[0], by, bw, bh, "GAMUS dataset", [
    "US aerial RGB + LiDAR nDSM", "+ land-cover labels", "1024 × 1024 px @ 0.3 m",
    "**Used: 1,500 train", "**48 val · 40 test (held out)", "CC BY 4.0"],
    num="A1", accent=TRAIN[1], fill="#fff")
box(bx[1], by, bw, bh, "Scale normalise", [
    "0.3 m → 0.5 m per pixel", "1024 → 608 px", "= 38 × 38 patches of 16 px",
    "same resampling as", "inference (checked)"],
    num="A2", accent=TRAIN[1], fill="#fff")
box(bx[2], by, bw, bh, "DINOv3 ViT-L/16", [
    "pre-trained on SAT-493M", "(493 M satellite images)", "**FROZEN (not trained)",
    "features from layers", "6 · 12 · 18 · 24", "SAT mean/std normalisation"],
    num="A3", accent=TRAIN[1], fill=OURS)
box(bx[3], by, bw, bh, "Feature cache", [
    "computed once per tile", "4 × 1024 × 38 × 38", "fp16 ≈ 12 MB / tile",
    "→ training fits a", "free Colab T4 GPU"],
    num="A4", accent=TRAIN[1], fill="#fff")
box(bx[4], by, bw, bh, "DPT height head", [
    "**TRAINED: 12.7 M params", "4-level reassemble + fuse", "masked Huber loss (m),",
    "target clip 0–150 m", "AdamW 1e-4, cosine, 30 ep", "aug: flips + 90° rotations"],
    num="A5", accent=TRAIN[1], fill=OURS)
box(bx[5], by, bw, bh, "best.pt", [
    "checkpoint with lowest", "**validation RMSE", "(test tiles never used", "for selection)",
    "→ best_1500.pt"],
    num="A6", accent=TRAIN[1], fill="#fff")
for k in range(5):
    arrow([(bx[k] + bw, by + bh / 2), (bx[k + 1] - 3, by + bh / 2)])

# ================================================================ B. INFERENCE
BX, BY, BW, BH = 30, 370, 1350, 520
band(BX, BY, BW, BH, INFER, "B · Inference — dsm.py (one command: python dsm.py --model ours)",
     "Any georeferenced image in, standard GeoTIFF DSM out. Runs on a laptop GPU.")
r1y, r1h = BY + 70, 190
c1 = (BX + 22, 230)
c2 = (c1[0] + c1[1] + 25, 230)
c3 = (c2[0] + c2[1] + 25, 210)
c4 = (c3[0] + c3[1] + 25, BX + BW - 22 - (c3[0] + c3[1] + 25))
box(c1[0], r1y, c1[1], r1h, "Input image", [
    "GeoTIFF (any CRS) or", "PNG / JPG (+ --gsd)", "", "**Demo: Chungthang, Sikkim",
    "WorldView-2, 0.305 m px", "2048 × 2048 px = 625 × 625 m", "Maxar Open Data, CC BY-NC"],
    num=1, accent=INFER[1], fill="#fff")
box(c2[0], r1y, c2[1], r1h, "Georeference", [
    "CRS + transform preserved", "pixel size (GSD) read from", "the file; degrees → metres",
    "scene-fill mask (edge black)", "", "no CRS (PNG/JPG) →", "relative DSM 0–1 path"],
    num=2, accent=INFER[1], fill="#fff")
box(c3[0], r1y, c3[1], r1h, "Scale normalise", [
    "resample to the model's", "training scale: 0.5 m", "", "Chungthang:",
    "2048 → 1246 px", "(0.305 → 0.5 m)"],
    num=3, accent=INFER[1], fill="#fff")
# step 4: model box with two backends
box(c4[0], r1y, c4[1], r1h, "Height model (nDSM, metres)", [
    "tiles + 25% overlap · Gaussian-feathered blend · 3-way flip TTA ·",
    "resample back to input grid · clamp ≥ 0 m"],
    num=4, accent=INFER[1], fill="#fff", bsize=12)
sx, sw_ = c4[0] + 14, (c4[1] - 42) // 2
box(sx, r1y + 88, sw_, 90, "OURS  (--model ours)", [
    "DINOv3-SAT + trained head", "tile 608 px · patch 16", "weights: best_1500.pt"],
    fill=OURS, stroke=TRAIN[1], tsize=13, bsize=11.5, lh=15)
box(sx + sw_ + 14, r1y + 88, sw_, 90, "Baseline (default)", [
    "RS3DAda (SynRS3D, MIT)", "ViT-L + DPT · tile 1022 · patch 14", "public weights, no login"],
    fill="#f4f6f9", tsize=13, bsize=11.5, lh=15)
for (a, b_) in [(c1, c2), (c2, c3), (c3, c4)]:
    arrow([(a[0] + a[1], r1y + r1h / 2), (b_[0] - 3, r1y + r1h / 2)])
# best.pt -> OURS
arrow([(bx[5] + bw / 2, by + bh), (bx[5] + bw / 2, r1y + 70), (sx + sw_ / 2, r1y + 70),
       (sx + sw_ / 2, r1y + 85)], label="trained weights", lx=bx[5] + bw / 2 + 8, ly=by + bh + 22,
      color=TRAIN[1])

r2y, r2h = r1y + r1h + 45, 170
d1 = (BX + 22, 300)
d2 = (d1[0] + d1[1] + 25, 205)
d3 = (d2[0] + d2[1] + 25, 330)
d4 = (d3[0] + d3[1] + 25, BX + BW - 22 - (d3[0] + d3[1] + 25))
box(d1[0], r2y, d1[1], r2h, "Terrain (DEM)", [
    "Copernicus GLO-30 (auto-fetched", "for the image footprint)", "or CartoDEM via --dem",
    "bilinear reprojection to the", "image grid · heights: EGM2008", "Chungthang: 1549–1770 m"],
    num=5, accent=INFER[1], fill="#fff")
box(d2[0], r2y, d2[1], r2h, "Compose", [
    "**DSM = DEM + nDSM", "metres above sea level", "(EGM2008 geoid)", "",
    "one value per image pixel", "at the image resolution"],
    num=6, accent=INFER[1], fill="#fff")
box(d3[0], r2y, d3[1], r2h, "Outputs (standard formats)", [
    "**dsm.tif · ndsm.tif · dem.tif", "GeoTIFF float32 · LZW · nodata −9999",
    "same CRS & grid as the input image", "report.json (provenance + checks)",
    "SUMMARY.txt · one shareable .zip", "viewer assets → C"],
    num=7, accent=INFER[1], fill="#fff")
box(d4[0], r2y, d4[1], r2h, "Checks + buildings", [
    "no ground truth needed:", "height percentiles, > 80 m flag,", "DEM coverage, 30 m double-count",
    "building finder: nDSM ≥ 2.5 m,", "trees removed (colour + texture),", "20–5000 m², floors = h ÷ 3 m"],
    num=8, accent=INFER[1], fill="#fff")
# 2 -> 5 (footprint)
arrow([(c2[0] + 22, r1y + r1h), (c2[0] + 22, r2y - 3)], label="footprint + grid",
      lx=c2[0] + 30, ly=r1y + r1h + 26)
# 4 -> 6 (nDSM)
m4 = c4[0] + 60
arrow([(m4, r1y + r1h), (m4, r1y + r1h + 22), (d2[0] + d2[1] / 2, r1y + r1h + 22),
       (d2[0] + d2[1] / 2, r2y - 3)], label="nDSM", lx=d2[0] + d2[1] / 2 + 8, ly=r1y + r1h + 38)
arrow([(d1[0] + d1[1], r2y + r2h / 2), (d2[0] - 3, r2y + r2h / 2)])
arrow([(d2[0] + d2[1], r2y + r2h / 2), (d3[0] - 3, r2y + r2h / 2)])
arrow([(d3[0] + d3[1], r2y + r2h / 2), (d4[0] - 3, r2y + r2h / 2)])

# ================================================================ C. VIEWER
CX, CY, CW, CH = 30, 910, 1350, 235
band(CX, CY, CW, CH, VIEW, "C · Visualisation — Three.js viewer (any browser, python serve.py)",
     "Loads step 7's assets: 1024² float32 height grid, nDSM layer, ≤ 2048 px texture, building list.")
ew, eh, ey = 250, 145, CY + 72
ex = [CX + 22 + k * (ew + 17) for k in range(5)]
box(ex[0], ey, ew, eh, "3D terrain", [
    "1.05 M-vertex mesh (1024²)", "image draped exactly (ortho)", "walls detected from nDSM",
    "(hillsides stay photo-real)", "true scale 1.0× for DSM", "adjustable sun + shadows"],
    num="C1", accent=VIEW[1], fill="#fff")
box(ex[1], ey, ew, eh, "Navigate", [
    "fly (W A S D, Q E) with", "terrain collision", "orbit + zoom",
    "scripted 24 s tour", "→ record MP4/WebM video"],
    num="C2", accent=VIEW[1], fill="#fff")
box(ex[2], ey, ew, eh, "Analyse", [
    "probe: elevation, height", "above ground, slope, floors", "measure: distance, Δheight",
    "+ height-profile chart", "overlays: height · slope · error"],
    num="C3", accent=VIEW[1], fill="#fff")
box(ex[3], ey, ew, eh, "Validate (GAMUS tiles)", [
    "switch model ↔ LiDAR truth", "signed error map", "(red too high, blue too low)",
    "live RMSE · MAE · bias", "on the loaded tile"],
    num="C4", accent=VIEW[1], fill="#fff")
box(ex[4], ey, ew, eh, "Tools (optional)", [
    "Flood: water-level slider,", "flooded area + buildings hit", "Buildings: pins by floors,",
    "red when flooded"],
    num="C5", accent=VIEW[1], fill="#fff", dashed=True)
arrow([(d3[0] + d3[1] / 2, r2y + r2h), (d3[0] + d3[1] / 2, CY - 3)],
      label="viewer assets", lx=d3[0] + d3[1] / 2 + 8, ly=r2y + r2h + 14)

# ================================================================ RESULTS COLUMN
RX, RY, RW, RH = 1400, 90, 370, 1055
add(f'<rect x="{RX}" y="{RY}" width="{RW}" height="{RH}" rx="14" fill="{RES[0]}" '
    f'stroke="{RES[1]}" stroke-width="1.6"/>')
add(f'<rect x="{RX}" y="{RY}" width="7" height="{RH}" rx="3" fill="{RES[1]}"/>')
yy = RY + 32


def rt(txt, size=13, weight="400", col=INK, dy=20, x=RX + 22, anchor="start"):
    global yy
    add(f'<text x="{x}" y="{yy}" font-size="{size}" font-weight="{weight}" fill="{col}" '
        f'text-anchor="{anchor}">{esc(txt)}</text>')
    yy += dy


rt("D · Measured results", 18, "800", RES[1], 22)
rt("40 held-out GAMUS test tiles, real LiDAR truth", 12, col=DIM, dy=30)
rt("Overall error (RMSE, lower = better)", 13.5, "700", dy=22)
rows = [("Predict zero (floor)", 9.35, "#d5dae0"), ("RS3DAda (measured)", 6.74, "#9aa5b1"),
        ("Ours, 480 train tiles", 4.96, "#52606d"), ("Ours, 1,500 train tiles", 4.69, COPPER)]
for name, v, col in rows:
    add(f'<text x="{RX + 22}" y="{yy}" font-size="12.5" fill="{INK}">{esc(name)}</text>')
    bwid = v / 9.35 * 105
    add(f'<rect x="{RX + 185}" y="{yy - 12}" width="{bwid:.1f}" height="15" rx="3" fill="{col}"/>')
    add(f'<text x="{RX + 190 + bwid:.1f}" y="{yy}" font-size="12.5" font-weight="700" '
        f'fill="{INK}">{v:.2f} m</text>')
    yy += 25
yy += 2
rt("Ours (1,500): MAE 2.29 m · correlation r 0.81", 12.5, "700", dy=18)
rt("86.6% of pixels within 5 m of LiDAR", 12.5, col=DIM, dy=30)

rt("By land type (RMSE)", 13.5, "700", dy=22)
hdr = [("", RX + 22), ("RS3DAda", RX + 190), ("Ours", RX + 280)]
for t, x in hdr:
    add(f'<text x="{x}" y="{yy}" font-size="12" font-weight="700" fill="{DIM}">{t}</text>')
yy += 20
for lt, a, b_ in [("Urban", "5.11", "4.79"), ("Sparse", "2.83", "2.15"), ("Forest", "11.87", "6.03")]:
    add(f'<text x="{RX + 22}" y="{yy}" font-size="12.5" fill="{INK}">{lt}</text>')
    add(f'<text x="{RX + 190}" y="{yy}" font-size="12.5" fill="{INK}">{a} m</text>')
    add(f'<text x="{RX + 280}" y="{yy}" font-size="12.5" font-weight="700" fill="{COPPER}">{b_} m</text>')
    yy += 20
yy += 12

rt("Input-resolution robustness (ours)", 13.5, "700", dy=22)
rt("0.30 / 0.50 / 0.75 / 1.00 m per pixel", 12.5, col=DIM, dy=18)
rt("RMSE 4.55 / 4.40 / 4.58 / 5.63 m", 12.5, "700", dy=30)

rt("Speed — Chungthang, 625 × 625 m", 13.5, "700", dy=22)
rt("RTX 5060 laptop GPU, end to end:", 12.5, col=DIM, dy=18)
rt("RS3DAda 18.3 s · ours (480) 14.4 s", 12.5, "700", dy=30)

rt("Indian demo — Chungthang, Sikkim", 13.5, "700", dy=22)
rt("height above ground, 95th percentile:", 12.5, col=DIM, dy=18)
rt("RS3DAda 6.4 m · ours (480) 8.9 m", 12.5, "700", dy=18)
rt("ours predicts taller canopy and buildings", 12.5, col=DIM, dy=30)

rt("Rigour", 13.5, "700", dy=22)
for t in ["same 40 test tiles for every model", "model chosen on validation only",
          "harness self-test: 0.13 m noise floor", "negative results reported"]:
    rt("• " + t, 12.5, dy=18)
yy += 12
rt("Stack", 13.5, "700", dy=22)
for t in ["PyTorch · HuggingFace Transformers", "rasterio / GDAL · NumPy · SciPy",
          "Three.js (WebGL) viewer", "runs on a laptop GPU; trains on free Colab"]:
    rt("• " + t, 12.5, dy=18)

# ---------------------------------------------------------------- legend
ly = H - 42
add(f'<line x1="30" y1="{ly}" x2="70" y2="{ly}" stroke="{INK}" stroke-width="1.8" marker-end="url(#ah)"/>')
add(f'<text x="80" y="{ly + 4}" font-size="12.5" fill="{DIM}">data flow</text>')
add(f'<rect x="180" y="{ly - 9}" width="30" height="18" rx="4" fill="{OURS}" stroke="{TRAIN[1]}"/>')
add(f'<text x="218" y="{ly + 4}" font-size="12.5" fill="{DIM}">our model components</text>')
add(f'<rect x="390" y="{ly - 9}" width="30" height="18" rx="4" fill="#fff" stroke="{LINE}" stroke-dasharray="6 4"/>')
add(f'<text x="428" y="{ly + 4}" font-size="12.5" fill="{DIM}">optional (off by default)</text>')
add(f'<text x="{W - 30}" y="{ly + 4}" font-size="12" fill="{DIM}" text-anchor="end">'
    'Credits: GAMUS (CC BY 4.0) · DINOv3 (Meta) · RS3DAda/SynRS3D (MIT) · Copernicus DEM GLO-30 '
    '© DLR/Airbus · imagery © Maxar Open Data (CC BY-NC 4.0) · Three.js (MIT)</text>')
add("</svg>")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(svg))
print("wrote", OUT)
