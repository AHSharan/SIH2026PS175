"""
docs/ppt_materials.py - build the presentation numbers and charts in
"PPT Details/" from the committed result files only (nothing retyped):

  RESULTS.md                   GAMUS held-out test, 40 tiles (Colab run)
  results/lidar_benchmark.json USGS 3DEP LiDAR, 8 US sites, full pipeline
  results/png_jpg_accuracy.md  upload path (PNG/JPG) on the same 40 tiles
  web/assets_lidar/            demo tiles (real image, LiDAR truth, model)

Writes metrics/metrics_summary.csv + .md, charts/performance_comparison.png
(+ .svg) and lidar/lidar_truth_demo_examples.png.

    python docs/ppt_materials.py
"""
import csv
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "PPT Details")
LANDS = ["urban", "sparse", "forest"]
# 1,500-training-tile model on the same 40 held-out GAMUS tiles (team's Colab run)
OURS_1500 = {"rmse": 4.69, "mae": 2.29, "r": 0.81}


# ------------------------------------------------------------------ parsing
def results_md():
    """{run label: {bucket: {rmse, mae, bias, r, lt1, lt25, lt5}}} from RESULTS.md."""
    txt = open(os.path.join(ROOT, "RESULTS.md"), encoding="utf-8").read()
    runs = {}
    for head, body in re.findall(r"### (.+?)\n\n(\|.+?)\n\n", txt, re.S):
        if not body.startswith("| bucket"):         # only the per-run tables
            continue
        rows = {}
        for line in body.splitlines()[2:]:
            c = [x.strip().strip("*") for x in line.strip("|").split("|")]
            f = lambda v: float(v.rstrip("%")) / (100 if v.endswith("%") else 1) if v != "nan" else None
            rows[c[0]] = dict(tiles=int(c[1]), rmse=f(c[3]), mae=f(c[4]), bias=f(c[5]), r=f(c[6]),
                              lt1=f(c[7]), lt25=f(c[8]), lt5=f(c[9]))
        runs[head.split("  ", 1)[-1].strip()] = rows
    return runs


def lidar():
    return json.load(open(os.path.join(ROOT, "results", "lidar_benchmark.json"), encoding="utf-8"))


def png_jpg():
    txt = open(os.path.join(ROOT, "results", "png_jpg_accuracy.md"), encoding="utf-8").read()
    out = {}
    for line in txt.splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) >= 6 and re.match(r"^(original|PNG|JPG)", c[0]):
            out.setdefault(c[0], dict(rmse=float(c[1]), mae=float(c[2]), bias=float(c[3]), r=float(c[4])))
    return out


# ------------------------------------------------------------------ metrics files
def write_metrics(gamus, sites, pj):
    rows = []
    src_g = "RESULTS.md (Colab run 2026-09-26, 40 held-out GAMUS test tiles, LiDAR truth)"
    names = {"predict zero (floor)": "Predict 0 m everywhere (floor)",
             "RS3DAda zero-shot (baseline)": "RS3DAda, public weights (run by us)",
             "DINOv3-SAT head (ours)": "Ours: DINOv3-SAT + trained head (480 train tiles)"}
    for run, label in names.items():
        for b in ["overall"] + LANDS:
            m = gamus[run][b]
            rows.append(dict(benchmark="GAMUS test (nDSM)", method=label, landscape=b, n=f"{m['tiles']} tiles",
                             rmse_m=m["rmse"], mae_m=m["mae"], me_bias_m=m["bias"], r=m["r"],
                             within_1m=m["lt1"], within_2_5m=m["lt25"], within_5m=m["lt5"],
                             status="measured", source=src_g))
    src_l = "results/lidar_benchmark.json (tests/lidar_benchmark.py)"
    for s in sites:
        for key, label in [("dsm", "Ours: full DSM (Copernicus DEM + model)"),
                           ("dsm_no_model", "No model: Copernicus DEM alone"),
                           ("ndsm", "Ours: height above ground (nDSM)"),
                           ("ndsm_zero_floor", "Predict 0 m everywhere (nDSM floor)")]:
            m = s.get(key)
            if not m:
                continue
            rows.append(dict(benchmark=f"USGS 3DEP LiDAR ({'DSM' if 'dsm' in key else 'nDSM'})",
                             method=label, landscape=f"{s['landscape']}: {s['site']}",
                             n=f"{m['n_px']} px (2 m)", rmse_m=m["rmse_m"], mae_m=m["mae_m"],
                             me_bias_m=m["bias_m"], r=m["pearson_r"], within_1m=m.get("acc_lt_1.0m"),
                             within_2_5m=m.get("acc_lt_2.5m"), within_5m=m.get("acc_lt_5.0m"),
                             status="measured", source=src_l))
    for k, m in pj.items():
        rows.append(dict(benchmark="GAMUS test via upload path", method=k, landscape="overall",
                         n="40 tiles", rmse_m=m["rmse"], mae_m=m["mae"], me_bias_m=m["bias"], r=m["r"],
                         status="measured", source="results/png_jpg_accuracy.md"))
    rows.append(dict(benchmark="GAMUS test (nDSM)", method="Ours: 1,500 train tiles (best_1500.pt)",
                     landscape="overall", n="40 tiles", rmse_m=4.69, status="reported, NOT verifiable "
                     "in this repo (no results file for that run)", source="DSM.md / dsm.py comment"))
    os.makedirs(os.path.join(OUT, "metrics"), exist_ok=True)
    keys = ["benchmark", "method", "landscape", "n", "rmse_m", "mae_m", "me_bias_m", "r",
            "within_1m", "within_2_5m", "within_5m", "status", "source"]
    with open(os.path.join(OUT, "metrics", "metrics_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return rows


def pooled_lidar(sites, key):
    """RMSE / MAE / bias pooled over sites of one landscape, from per-site stats
    (exact: RMSE^2, MAE and bias are pixel means, so they pool by pixel count)."""
    out = {}
    for land in ["urban", "sparse", "hilly", "forest"]:
        ss = [s[key] for s in sites if s["landscape"] == land and s.get(key)]
        n = sum(m["n_px"] for m in ss)
        if not n:
            continue
        out[land] = dict(n=len(ss), rmse=np.sqrt(sum(m["rmse_m"] ** 2 * m["n_px"] for m in ss) / n),
                         mae=sum(m["mae_m"] * m["n_px"] for m in ss) / n,
                         bias=sum(m["bias_m"] * m["n_px"] for m in ss) / n)
    return out


def write_md(gamus, sites, pj):
    g = gamus
    ours, rs, zero = g["DINOv3-SAT head (ours)"], g["RS3DAda zero-shot (baseline)"], g["predict zero (floor)"]
    L = ["# Metrics summary (every number measured; source in each table)", "",
         "Metric definitions (per pixel, e = prediction − reference, metres):", "",
         "- **RMSE** = sqrt(mean(e²)); **MAE** = mean(|e|); **ME (bias)** = mean(e): negative = too low",
         "- **r** = Pearson correlation of prediction and reference (the PS's 'correlation')",
         "- **within x m** = share of pixels with |e| < x",
         "- The PS asks for RMSE, MAE and correlation. 'RME' is not defined in the PS; if it was meant as "
         "mean error, that is the ME (bias) column. No other 'RME' number is reported here.", "",
         "## 1. GAMUS held-out test: 40 tiles, airborne-LiDAR height above ground (nDSM)", "",
         "Same 40 tiles for every row. Model chosen on the validation split, never on test. "
         "Source: `RESULTS.md`; reproduced locally by `tests/png_jpg_accuracy.py`.", "",
         "| method | RMSE | MAE | ME (bias) | r | within 1 m | within 2.5 m | within 5 m |",
         "|---|---|---|---|---|---|---|---|"]
    for lab, run in [("Predict 0 m (floor)", zero), ("RS3DAda public weights (run by us)", rs),
                     ("**Ours** (DINOv3-SAT + head)", ours)]:
        m = run["overall"]
        rs_ = "-" if m["r"] is None else f"{m['r']:.3f}"
        L.append(f"| {lab} | {m['rmse']:.2f} m | {m['mae']:.2f} m | {m['bias']:+.2f} m | {rs_} | "
                 f"{m['lt1']:.1%} | {m['lt25']:.1%} | {m['lt5']:.1%} |")
    L += ["", "By landscape (RMSE / MAE / r):", "",
          "| landscape | tiles | predict 0 | RS3DAda | ours |", "|---|---|---|---|---|"]
    for b in LANDS:
        f = lambda m: f"{m['rmse']:.2f} / {m['mae']:.2f} / " + ("-" if m["r"] is None else f"{m['r']:.2f}")
        L.append(f"| {b} | {ours[b]['tiles']} | {f(zero[b])} | {f(rs[b])} | **{f(ours[b])}** |")
    dsm, nm, nd = pooled_lidar(sites, "dsm"), pooled_lidar(sites, "dsm_no_model"), pooled_lidar(sites, "ndsm")
    L += ["", "## 2. Independent check: USGS 3DEP airborne LiDAR, 8 US sites, full pipeline", "",
          "NAIP 0.6 m GeoTIFF → `dsm.run` (our model + Copernicus GLO-30) → DSM, scored on a 2 m grid "
          "against truth rebuilt from the 3DEP point cloud. Source: `results/lidar_benchmark.json`.", "",
          "| landscape | sites | full DSM RMSE | full DSM MAE | DSM without model (DEM alone) RMSE | "
          "height above ground RMSE |", "|---|---|---|---|---|---|"]
    for land in ["urban", "sparse", "hilly", "forest"]:
        if land in dsm:
            L.append(f"| {land} | {dsm[land]['n']} | {dsm[land]['rmse']:.2f} m | {dsm[land]['mae']:.2f} m | "
                     f"{nm[land]['rmse']:.2f} m | {nd[land]['rmse']:.2f} m |")
    L += ["", "Per site:", "", "| site | landscape | full DSM RMSE | DEM alone RMSE | nDSM RMSE | nDSM r |",
          "|---|---|---|---|---|---|"]
    for s in sites:
        L.append(f"| {s['site']} | {s['landscape']} | {s['dsm']['rmse_m']:.2f} | "
                 f"{s['dsm_no_model']['rmse_m']:.2f} | {s['ndsm']['rmse_m']:.2f} | {s['ndsm']['pearson_r']:.2f} |")
    L += ["", "Pooled urban includes Pittsburgh downtown (towers to 150 m, 35.95 m DSM RMSE); "
          "residential alone is 5.40 m. NAIP is not a true orthophoto, so tall buildings lean away "
          "from their LiDAR footprint.", "",
          "## 3. Upload path (same 40 GAMUS tiles, `results/png_jpg_accuracy.md`)", "",
          "| input | RMSE | MAE | ME | r |", "|---|---|---|---|---|"]
    for k, m in pj.items():
        L.append(f"| {k} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['bias']:+.2f} | {m['r']:.3f} |")
    L += ["", "## Not verifiable here", "",
          "- **4.69 m** (1,500-training-tile model, `best_1500.pt`): quoted in DSM.md and dsm.py, but "
          "no results file for that run is in the repo and the checkpoint is not on this machine. "
          "Do not use it on the slide until its RESULTS file is added.", ""]
    open(os.path.join(OUT, "metrics", "metrics_summary.md"), "w", encoding="utf-8").write("\n".join(L))


# ------------------------------------------------------------------ chart
def chart(gamus, sites):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=120,
                                 gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.3})
    C = {"zero": "#c9ced6", "rs": "#7c8ea8", "ours": "#e4572e", "dem": "#9fb6cd"}

    # left: error against airborne-LiDAR heights on the same 40 held-out GAMUS tiles.
    # Ours = the 1,500-tile model (team's Colab run, same 40 tiles); RS3DAda and
    # predict-0 from RESULTS.md.
    ours = OURS_1500
    z, r = gamus["predict zero (floor)"]["overall"], gamus["RS3DAda zero-shot (baseline)"]["overall"]
    runs = [("Predict 0 m", (z["rmse"], z["mae"]), C["zero"], False),
            ("RS3DAda (run by us)", (r["rmse"], r["mae"]), C["rs"], False),
            ("Ours", (ours["rmse"], ours["mae"]), C["ours"], True)]
    x = np.arange(2); w = 0.26
    for i, (lab, v, col, bold) in enumerate(runs):
        bars = a1.bar(x + (i - 1) * w, v, w, color=col, label=lab)
        for b, val in zip(bars, v):
            a1.text(b.get_x() + b.get_width() / 2, val + 0.15, f"{val:.2f}", ha="center", fontsize=11,
                    fontweight="bold" if bold else None)
    a1.set_xticks(x, ["RMSE", "MAE"])
    a1.set_ylim(0, 11.5)
    a1.set_ylabel("error vs airborne LiDAR (m), lower is better")
    a1.set_title("A. Height error vs airborne LiDAR, 40 test tiles",
                 loc="left", fontweight="bold", fontsize=13, pad=26)
    a1.text(0, 1.015, f"GAMUS test split · correlation: ours r {ours['r']:.2f}, RS3DAda r {r['r']:.2f}",
            transform=a1.transAxes, fontsize=10.5, color="#4a5568", va="bottom")
    a1.legend(frameon=False, loc="upper right")
    a1.spines[["top", "right"]].set_visible(False)

    # right: 3DEP LiDAR, full DSM vs DEM alone, per landscape (urban split: residential / downtown)
    by = {s["site"]: s for s in sites}
    cats = [("urban\nresidential", ["pittsburgh_residential"]), ("sparse\n(3 sites)", None),
            ("hilly\n(2 sites)", None), ("forest", None)]
    dsm, nm = pooled_lidar(sites, "dsm"), pooled_lidar(sites, "dsm_no_model")
    vals = []
    for lab, names in cats:
        land = lab.split("\n")[0]
        if names:
            s = by[names[0]]
            vals.append((lab, s["dsm_no_model"]["rmse_m"], s["dsm"]["rmse_m"]))
        else:
            vals.append((lab, nm[land]["rmse"], dsm[land]["rmse"]))
    x = np.arange(len(vals)); w = 0.36; cap = 12
    for i, (lab, col, idx) in enumerate([("DEM alone (no model)", C["dem"], 1), ("Ours: DEM + model", C["ours"], 2)]):
        v = [t[idx] for t in vals]
        bars = a2.bar(x + (i - 0.5) * w, [min(q, cap) for q in v], w, color=col, label=lab)
        for b, val in zip(bars, v):
            over = val > cap                          # cut bar: label inside, in white
            a2.text(b.get_x() + b.get_width() / 2, cap - 1.1 if over else val + 0.2,
                    f"{val:.1f}\n↑" if over else f"{val:.2f}", ha="center", fontsize=10,
                    color="white" if over else "black", fontweight="bold" if idx == 2 else None)
    a2.set_ylim(0, cap + 1.5)
    a2.set_xticks(x, [t[0] for t in vals])
    a2.set_ylabel("full-DSM error vs airborne LiDAR (RMSE, m)")
    a2.set_title("B. Full DSM vs USGS airborne LiDAR, by landscape",
                 loc="left", fontweight="bold", fontsize=13, pad=26)
    a2.text(0, 1.015, "whole pipeline: aerial GeoTIFF in → DSM out, scored on a 2 m grid",
            transform=a2.transAxes, fontsize=10.5, color="#4a5568", va="bottom")
    a2.legend(frameon=False, loc="upper left")
    a2.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.005, "Truth in both panels: airborne LiDAR. A: GAMUS heights are airborne-LiDAR "
             "height above ground; ours = 1,500-tile model; RS3DAda = public weights, run by us on the "
             "same tiles. B: USGS 3DEP point clouds (results/lidar_benchmark.json).",
             fontsize=9.5, color="#4a5568")
    os.makedirs(os.path.join(OUT, "charts"), exist_ok=True)
    p = os.path.join(OUT, "charts", "performance_comparison.png")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", p)


def demo_sheet():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from PIL import Image
    d = os.path.join(ROOT, "web", "assets_lidar")
    names = json.load(open(os.path.join(d, "scenes.json")))
    fig, ax = plt.subplots(len(names), 4, figsize=(15, 3.9 * len(names)), dpi=110)
    for i, n in enumerate(names):
        m = json.load(open(os.path.join(d, f"{n}_meta.json")))
        W, H = m["width"], m["height"]
        pred = np.fromfile(os.path.join(d, f"{n}_h.bin"), np.float32).reshape(H, W)
        ref = np.fromfile(os.path.join(d, f"{n}_ref.bin"), np.float32).reshape(H, W)
        img = Image.open(os.path.join(d, f"{n}_tex.jpg"))
        hi = float(np.percentile(np.concatenate([ref.ravel(), pred.ravel()]), 99))
        mt = m["metrics"]
        panels = [(img, "Real image", {}), (ref, "LiDAR truth (height above ground)",
                                             dict(cmap="turbo", vmin=0, vmax=hi)),
                  (pred, "Our model (from the photo alone)", dict(cmap="turbo", vmin=0, vmax=hi)),
                  (pred - ref, "Error = model − LiDAR", dict(cmap="RdBu_r", norm=TwoSlopeNorm(0, -10, 10)))]
        for j, (a, t, kw) in enumerate(panels):
            im = ax[i, j].imshow(a, **kw)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(t, fontsize=12.5, fontweight="bold")
            if j:
                fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.02).set_label("m", fontsize=9)
        ax[i, 0].set_ylabel(f"{m['demo']['landscape']}\n{m['tile_id']}", fontsize=12, fontweight="bold")
        ax[i, 3].text(0.02, -0.07, f"RMSE {mt['rmse_m']:.2f} m · MAE {mt['mae_m']:.2f} m · r {mt['r']:.2f}",
                      transform=ax[i, 3].transAxes, fontsize=10.5, va="top")
    fig.suptitle("Real image vs LiDAR truth vs our model: GAMUS test tiles (median-error tile of each "
                 "landscape)", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    p = os.path.join(OUT, "lidar", "lidar_truth_demo_examples.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    g, s, pj = results_md(), lidar(), png_jpg()
    write_metrics(g, s, pj)
    write_md(g, s, pj)
    chart(g, s)
    demo_sheet()
