"""
GAMUS data loader for DepthWizard (SIH 26175).

VERIFIED FACTS (measured 2026-09-19 from earthflow/GAMUS, not assumed):
  - Files are HDF5 (.h5), NOT GeoTIFF. Every file holds one dataset, key "image".
  - Layout: {images,heights,classes}/{train,val,test}/{STEM}_{RGB|IMG,AGL,CLS}.h5
  - !! RGB suffix is NOT uniform: DC/PHL use "_RGB", NYC uses "_IMG".
    Pairing on "_RGB" alone silently drops ALL 1000 NYC test tiles (and 1167
    train tiles) - i.e. exactly the city with the tall buildings.
  - Counts: train 5004, val 859, test 2861 tiles (x3 modalities = 26172 files).
  - Cities: DC, NYC, PHL (DFC2019 US3D lineage).
  - RGB    : (1024,1024,3) uint8
  - AGL    : (1024,1024) float32, metres above ground (this IS nDSM)
  - CLS    : (1024,1024) float32, 7 codes 0..6
  - NO geotransform, NO CRS, NO nodata attribute exists in these files.
  - NaN: none observed in 30 sampled tiles.
  - Height range: min -5.0, max 146.4 m observed. PHL has 100m+ towers.
  - nodata: there is no declared nodata. DC tiles use an exact -5.0 sentinel
    (spikes of 5-7% of pixels at precisely -5.0). NYC has genuine small
    negatives (to -3.1) that are real LiDAR noise, NOT nodata. PHL has none.
    => we treat exact -5.0 as invalid, and clamp other negatives to 0.

GSD: CANNOT be read from these files (no transform). DFC2019 US3D, from which
GAMUS derives, is 0.3 m GSD imagery. GSD_M below is therefore an ASSERTED
constant, flagged as such. Do not silently trust it.
"""
from __future__ import annotations

import os
import glob as _glob
from dataclasses import dataclass, field

import numpy as np

# --- asserted, not measured -------------------------------------------------
GAMUS_GSD_M = 0.3   # DFC2019 US3D spec. NOT readable from the .h5 files.
GAMUS_GSD_IS_ASSERTED = True

NODATA_SENTINEL = -5.0   # measured: exact-valued spike in DC tiles

CLS_NAMES = {  # inferred by correlating class code against AGL height
    0: "unlabeled", 1: "ground", 2: "low_veg", 3: "building",
    4: "water", 5: "road_or_lowobj", 6: "tree",
}
CLS_BUILDING, CLS_WATER, CLS_TREE = 3, 4, 6


@dataclass
class Tile:
    tile_id: str
    rgb: np.ndarray          # (H,W,3) float32 in [0,1]
    height: np.ndarray       # (H,W) float32 metres, invalid -> np.nan
    mask: np.ndarray         # (H,W) bool, True = valid
    cls: np.ndarray | None = None
    gsd: float = GAMUS_GSD_M
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def _read_h5(path: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        key = "image" if "image" in f else list(f.keys())[0]
        return f[key][:]


def _read_rasterio(path: str):
    """GeoTIFF path, used for ISRO/other data later. Returns (arr, gsd, nodata)."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read()
        arr = arr[0] if arr.shape[0] == 1 else np.transpose(arr, (1, 2, 0))
        gsd = abs(src.transform.a) if src.transform else None
        return arr, gsd, src.nodata


def read_array(path: str):
    """Dispatch on extension. Returns (array, gsd_or_None, nodata_or_None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5"):
        return _read_h5(path), None, None
    return _read_rasterio(path)


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------
def stem_of(path: str, strip_suffixes=("_RGB", "_IMG", "_AGL", "_CLS")) -> str:
    """Pairing key: basename minus extension minus a known modality suffix.

    GAMUS names differ per modality (DC_03_26_RGB / DC_03_26_AGL), so a plain
    filename stem does NOT pair. Suffix stripping is what makes it work.
    """
    s = os.path.splitext(os.path.basename(path))[0]
    for suf in strip_suffixes:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def pair_dirs(rgb_dir: str, height_dir: str, cls_dir: str | None = None,
              rgb_glob: str = "*", height_glob: str = "*",
              cls_glob: str = "*", strip_suffixes=("_RGB", "_IMG", "_AGL", "_CLS")):
    """Pair RGB and height files by stem across directories. Configurable glob
    so a different layout still works. Returns sorted list of dicts."""
    def index(d, g):
        if not d:
            return {}
        return {stem_of(p, strip_suffixes): p
                for p in sorted(_glob.glob(os.path.join(d, g)))
                if os.path.isfile(p)}

    r, h, c = index(rgb_dir, rgb_glob), index(height_dir, height_glob), index(cls_dir, cls_glob)
    common = sorted(set(r) & set(h))
    only_r, only_h = sorted(set(r) - set(h)), sorted(set(h) - set(r))
    if only_r or only_h:
        print(f"[data] WARNING unpaired: {len(only_r)} rgb-only, {len(only_h)} height-only"
              f" (e.g. {(only_r + only_h)[:3]})")
    return [{"tile_id": k, "rgb": r[k], "height": h[k], "cls": c.get(k)} for k in common]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_tile(rec: dict, gsd: float = GAMUS_GSD_M,
              nodata_sentinel: float = NODATA_SENTINEL,
              clamp_negatives: bool = True) -> Tile:
    rgb_raw, _, _ = read_array(rec["rgb"])
    h_raw, h_gsd, h_nodata = read_array(rec["height"])

    rgb = rgb_raw.astype(np.float32)
    if rgb.max() > 1.5:          # uint8 -> [0,1]
        rgb /= 255.0
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)

    h = np.asarray(h_raw, dtype=np.float32)
    mask = np.isfinite(h)
    if nodata_sentinel is not None:
        mask &= (h != np.float32(nodata_sentinel))
    if h_nodata is not None:
        mask &= (h != np.float32(h_nodata))

    h = np.where(mask, h, np.nan)
    if clamp_negatives:
        # real small negatives (NYC LiDAR noise) -> 0; nodata already NaN
        h = np.where(mask & (h < 0), 0.0, h)

    cls = None
    if rec.get("cls"):
        cls = read_array(rec["cls"])[0].astype(np.int16)

    return Tile(tile_id=rec["tile_id"], rgb=rgb, height=h, mask=mask, cls=cls,
                gsd=(h_gsd if h_gsd else gsd),
                meta={"gsd_asserted": h_gsd is None,
                      "rgb_path": rec["rgb"], "height_path": rec["height"]})


# --------------------------------------------------------------------------
# bucketing (derived from CLS, so no hand-written buckets.csv needed)
# --------------------------------------------------------------------------
def bucket_of(tile: Tile) -> str:
    """urban / forest / sparse / hilly. 'hilly' is NOT derivable from nDSM
    (terrain is already removed), so it is never returned here - it needs a DEM.
    Thresholds are heuristic; they only stratify reporting."""
    if tile.cls is None:
        return "unknown"
    valid = tile.mask
    n = max(valid.sum(), 1)
    b = ((tile.cls == CLS_BUILDING) & valid).sum() / n
    t = ((tile.cls == CLS_TREE) & valid).sum() / n
    if b >= 0.20:
        return "urban"
    if t >= 0.35:
        return "forest"
    return "sparse"


# --------------------------------------------------------------------------
# describe
# --------------------------------------------------------------------------
def describe(records, n: int = 5, gsd: float = GAMUS_GSD_M):
    """Print dtype, nodata, GSD and height percentiles for a sample of tiles."""
    recs = records[:n]
    print("=" * 78)
    print(f"describe(): {len(recs)} of {len(records)} paired tiles")
    print(f"GSD = {gsd} m  <-- ASSERTED (DFC2019 US3D spec); .h5 files carry no transform")
    print(f"nodata: no declared value; treating exact {NODATA_SENTINEL} as invalid")
    print("=" * 78)
    for rec in recs:
        rgb_raw, _, _ = read_array(rec["rgb"])
        h_raw, h_gsd, h_nd = read_array(rec["height"])
        t = load_tile(rec, gsd=gsd)
        hv = t.height[t.mask]
        pct = np.percentile(hv, [0, 1, 50, 90, 99, 100])
        print(f"\n[{t.tile_id}]  bucket={bucket_of(t)}")
        print(f"   rgb    : shape={rgb_raw.shape} dtype={rgb_raw.dtype} "
              f"range=[{rgb_raw.min()},{rgb_raw.max()}]")
        print(f"   height : shape={h_raw.shape} dtype={h_raw.dtype} "
              f"file_nodata={h_nd} file_gsd={h_gsd}")
        print(f"   valid  : {100*t.mask.mean():.2f}%   "
              f"invalid={int((~t.mask).sum())} px   NaN_in_raw={int(np.isnan(h_raw).sum())}")
        print(f"   h pct  : p0={pct[0]:.2f} p1={pct[1]:.2f} p50={pct[2]:.2f} "
              f"p90={pct[3]:.2f} p99={pct[4]:.2f} max={pct[5]:.2f} m")
        if t.cls is not None:
            u, c = np.unique(t.cls[t.mask], return_counts=True)
            frac = {CLS_NAMES.get(int(k), k): f"{100*v/c.sum():.1f}%"
                    for k, v in zip(u, c) if 100*v/c.sum() >= 1.0}
            print(f"   classes: {frac}")
    print("\n" + "=" * 78)


# --------------------------------------------------------------------------
# convenience: fetch a subset from HuggingFace without pulling all 80 GB
# --------------------------------------------------------------------------
def gamus_index(split: str = "test", n: int = 20, repo: str = "earthflow/GAMUS",
                city: str | None = None, seed: int = 0, with_cls: bool = True,
                files=None, verbose: bool = True):
    """Deterministic tile selection WITHOUT downloading anything.

    Returns [{tile_id, rgb, height[, cls]}] with repo-relative paths. The
    selection logic is exactly what fetch_gamus_subset has always used, so the
    same (split, n, seed, city, with_cls) always yields the same tiles - which is
    what makes results from different machines (Kaggle, Colab) comparable.
    """
    if files is None:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo, repo_type="dataset")

    # The modality dirs have equal counts but are NOT stem-aligned (verified:
    # heights/test/NYC_12714_AGL.h5 exists, images/test/NYC_12714_RGB.h5 404s).
    # So intersect stems instead of assuming.
    def stems_in(mod):
        return {stem_of(f) for f in files if f.startswith(f"{mod}/{split}/")}
    stems_set = stems_in("heights") & stems_in("images")
    if with_cls:
        stems_set &= stems_in("classes")
    if city:
        stems_set = {s for s in stems_set if s.startswith(city)}
    stems = sorted(stems_set)
    if verbose:
        print(f"[data] {len(stems)} stems present in all modalities for split={split}")
    rng = np.random.default_rng(seed)
    stems = list(rng.choice(stems, min(n, len(stems)), replace=False)) if n < len(stems) else stems

    # RGB suffix varies by city (_RGB vs _IMG) -> resolve from the file list
    rgb_suffix = {}
    for f in files:
        if f.startswith(f"images/{split}/"):
            base = os.path.splitext(os.path.basename(f))[0]
            rgb_suffix[stem_of(base)] = base.rsplit("_", 1)[-1]

    out = []
    for s in stems:
        s = str(s)
        rec = {"tile_id": s,
               "rgb": f"images/{split}/{s}_{rgb_suffix.get(s, 'RGB')}.h5",
               "height": f"heights/{split}/{s}_AGL.h5"}
        if with_cls:
            rec["cls"] = f"classes/{split}/{s}_CLS.h5"
        out.append(rec)
    return out


def fetch_gamus_subset(split: str = "test", n: int = 20, out_dir: str = "gamus",
                       repo: str = "earthflow/GAMUS", city: str | None = None,
                       seed: int = 0, with_cls: bool = True):
    """hf_hub_download individual tiles only. Returns dirs (rgb, height, cls)."""
    from huggingface_hub import hf_hub_download
    import shutil
    recs = gamus_index(split, n, repo, city, seed, with_cls)

    dirs = {m: os.path.join(out_dir, split, m) for m in ("images", "heights", "classes")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    for rec in recs:
        keys = [("images", rec["rgb"]), ("heights", rec["height"])]
        if with_cls:
            keys.append(("classes", rec["cls"]))
        for mod, rp in keys:
            dst = os.path.join(dirs[mod], os.path.basename(rp))
            if os.path.exists(dst):
                continue
            src = hf_hub_download(repo, rp, repo_type="dataset")
            shutil.copyfile(src, dst)
    print(f"[data] fetched {len(recs)} tiles ({split}) -> {out_dir}")
    return dirs["images"], dirs["heights"], dirs["classes"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, default=0, help="download N tiles first")
    ap.add_argument("--split", default="test")
    ap.add_argument("--city", default=None)
    ap.add_argument("--root", default="gamus")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    if a.fetch:
        rgb_d, h_d, c_d = fetch_gamus_subset(a.split, a.fetch, a.root, city=a.city)
    else:
        rgb_d = os.path.join(a.root, a.split, "images")
        h_d = os.path.join(a.root, a.split, "heights")
        c_d = os.path.join(a.root, a.split, "classes")

    recs = pair_dirs(rgb_d, h_d, c_d, "*.h5", "*_AGL.h5", "*_CLS.h5")
    print(f"[data] paired {len(recs)} tiles")
    describe(recs, n=a.n)
