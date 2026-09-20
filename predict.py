"""
predict.py - model-agnostic tiled inference engine for height prediction.

Core contract:
    predict_fn(chw: np.ndarray float32 (3,H,W)) -> np.ndarray (H,W) heights

Everything else here (GSD normalisation, tiling, feather blending, TTA,
post-processing) is model-independent and reusable for the DINOv3 head later.

-------------------------------------------------------------------------
RS3DAda interface - READ FROM THE REPO, NOT GUESSED
  source: github.com/JTRNEO/SynRS3D  infer_height.py + models/dpt.py
  model      : DPT_DINOv2(encoder='vitl', head_configs=HEAD_CONFIGS,
                          pretrained=False)
  head_configs: [{'name':'regression','nclass':1},
                 {'name':'segmentation','nclass':8}]
  checkpoint : model.load_state_dict(torch.load(path))   # raw state_dict
               HF JTRNEO/RS3DAda / RS3DAda_vitl_DPT_height.pth (1.47 GB, MIT)
  output     : dict; height = outputs['regression'].squeeze()
  patch_size : 1022  (MUST be divisible by 14 - DINOv2 patch size)
  train GSD  : SynRS3D spans 0.05 - 1.0 m

  !! NORMALISATION TRAP !!
  albumentations Normalize(mean=(123.675,116.28,103.53),
                           std=(58.395,57.12,57.375), max_pixel_value=1)
  max_pixel_value=1 means the image is NOT divided by 255 first.
  The model expects raw 0-255 floats, then (x-mean)/std.
  Passing [0,1] input (as data.py returns) makes every height wrong.
-------------------------------------------------------------------------
"""
from __future__ import annotations

import numpy as np

RS3DADA_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
RS3DADA_STD = np.array([58.395, 57.12, 57.375], np.float32)
RS3DADA_PATCH = 1022          # divisible by 14
RS3DADA_TRAIN_GSD = 0.5       # midpoint of 0.05-1.0 m; see note in main()


# ==========================================================================
# resampling
# ==========================================================================
def resample_hw(arr: np.ndarray, out_hw, order: int = 1) -> np.ndarray:
    """Resample (H,W) or (H,W,C) to out_hw with pixel-centre alignment."""
    from scipy.ndimage import map_coordinates
    H, W = out_hw
    h, w = arr.shape[:2]
    if (h, w) == (H, W):
        return arr.astype(np.float32, copy=False)
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    xs = (np.arange(W) + 0.5) * w / W - 0.5
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    if arr.ndim == 2:
        return map_coordinates(arr.astype(np.float32), [gy, gx],
                               order=order, mode="nearest").astype(np.float32)
    chans = [map_coordinates(arr[..., c].astype(np.float32), [gy, gx],
                             order=order, mode="nearest")
             for c in range(arr.shape[2])]
    return np.stack(chans, -1).astype(np.float32)


# ==========================================================================
# feather weights
# ==========================================================================
def gaussian_weight(h: int, w: int, sigma_frac: float = 0.25) -> np.ndarray:
    """Gaussian-feathered tile weight, ~1 at centre, ->0 at edges.

    Never exactly 0: a zero-weight border pixel would divide by zero where
    only one tile covers it (image corners)."""
    yy = (np.arange(h) - (h - 1) / 2) / (sigma_frac * h)
    xx = (np.arange(w) - (w - 1) / 2) / (sigma_frac * w)
    g = np.exp(-0.5 * yy[:, None] ** 2) * np.exp(-0.5 * xx[None, :] ** 2)
    return (g + 1e-3).astype(np.float32)


# ==========================================================================
# tiled inference with TTA
# ==========================================================================
def _tta_variants(tile_chw, tta: bool):
    """Yield (transformed_input, inverse_fn)."""
    yield tile_chw, lambda a: a
    if tta:
        yield tile_chw[:, :, ::-1].copy(), lambda a: a[:, ::-1]
        yield tile_chw[:, ::-1, :].copy(), lambda a: a[::-1, :]


def tiled_predict(img_chw: np.ndarray, predict_fn, tile: int = RS3DADA_PATCH,
                  overlap: float = 0.25, tta: bool = True,
                  verbose: bool = False) -> np.ndarray:
    """Run predict_fn over overlapping tiles, Gaussian-blend the results."""
    C, H, W = img_chw.shape
    tile = min(tile, H, W)
    stride = max(1, int(round(tile * (1.0 - overlap))))

    def starts(total):
        if total <= tile:
            return [0]
        s = list(range(0, total - tile + 1, stride))
        if s[-1] != total - tile:
            s.append(total - tile)      # always cover the far edge exactly
        return s

    ys, xs = starts(H), starts(W)
    acc = np.zeros((H, W), np.float64)
    wacc = np.zeros((H, W), np.float64)
    wt = gaussian_weight(tile, tile)

    n = 0
    for y in ys:
        for x in xs:
            patch = img_chw[:, y:y + tile, x:x + tile]
            preds = []
            for inp, inv in _tta_variants(patch, tta):
                out = predict_fn(inp)
                out = np.asarray(out, np.float32)
                if out.shape != (tile, tile):
                    out = resample_hw(out, (tile, tile))
                preds.append(inv(out))
            p = np.mean(preds, axis=0)
            acc[y:y + tile, x:x + tile] += p * wt
            wacc[y:y + tile, x:x + tile] += wt
            n += 1
    if verbose:
        print(f"[predict] {n} tiles ({len(ys)}x{len(xs)}), tile={tile}, "
              f"stride={stride}, tta={tta}")
    return (acc / np.maximum(wacc, 1e-8)).astype(np.float32)


# ==========================================================================
# top level
# ==========================================================================
def predict_height(rgb_hwc: np.ndarray, predict_fn, *, input_gsd: float,
                   model_gsd: float, tile: int = RS3DADA_PATCH,
                   overlap: float = 0.25, tta: bool = True,
                   clamp_min: float | None = 0.0, median3: bool = False,
                   gsd_normalise: bool = True, patch_multiple: int = 14,
                   verbose: bool = False) -> np.ndarray:
    """RGB (H,W,3) -> height (H,W) on the ORIGINAL grid.

    input_gsd / model_gsd are REQUIRED keyword args - there is no sane default.
    Getting these wrong scales every predicted height by the wrong factor,
    which is the single most common way to produce confidently wrong metres.
    """
    if input_gsd is None or model_gsd is None:
        raise ValueError("input_gsd and model_gsd are required")

    H0, W0 = rgb_hwc.shape[:2]
    work = rgb_hwc

    if gsd_normalise and abs(input_gsd - model_gsd) > 1e-9:
        # make 1 output px equal model_gsd metres on the ground
        scale = input_gsd / model_gsd
        Hn = max(patch_multiple, int(round(H0 * scale)))
        Wn = max(patch_multiple, int(round(W0 * scale)))
        Hn -= Hn % patch_multiple
        Wn -= Wn % patch_multiple
        work = resample_hw(rgb_hwc, (Hn, Wn), order=1)
        if verbose:
            print(f"[predict] GSD {input_gsd}->{model_gsd} m : "
                  f"{H0}x{W0} -> {Hn}x{Wn} (scale {scale:.3f})")
    else:
        Hn = H0 - H0 % patch_multiple
        Wn = W0 - W0 % patch_multiple
        if (Hn, Wn) != (H0, W0):
            work = resample_hw(rgb_hwc, (Hn, Wn), order=1)
            if verbose:
                print(f"[predict] cropped to patch multiple: {H0}x{W0}->{Hn}x{Wn}")

    chw = np.transpose(work, (2, 0, 1)).astype(np.float32)
    height = tiled_predict(chw, predict_fn, tile=tile, overlap=overlap,
                           tta=tta, verbose=verbose)

    if height.shape != (H0, W0):
        height = resample_hw(height, (H0, W0), order=1)

    if median3:
        from scipy.ndimage import median_filter
        height = median_filter(height, size=3)
    if clamp_min is not None:
        height = np.maximum(height, clamp_min)
    return height.astype(np.float32)


# ==========================================================================
# RS3DAda backend
# ==========================================================================
class RS3DAda:
    """Wraps SynRS3D's DPT_DINOv2 height model.

    Requires the SynRS3D repo on sys.path:
        git clone https://github.com/JTRNEO/SynRS3D
    and the checkpoint:
        huggingface_hub.hf_hub_download('JTRNEO/RS3DAda',
                                        'RS3DAda_vitl_DPT_height.pth')
    """

    HEAD_CONFIGS = [{"name": "regression", "nclass": 1},
                    {"name": "segmentation", "nclass": 8}]

    def __init__(self, ckpt: str, synrs3d_dir: str = "SynRS3D",
                 device: str = None, encoder: str = "vitl", amp: bool = True):
        import sys
        import torch
        if synrs3d_dir and synrs3d_dir not in sys.path:
            sys.path.insert(0, synrs3d_dir)
        from models.dpt import DPT_DINOv2

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp and self.device == "cuda"

        self.model = DPT_DINOv2(encoder=encoder, head_configs=self.HEAD_CONFIGS,
                                pretrained=False)
        state = torch.load(ckpt, map_location="cpu")
        # tolerate common wrappers without guessing silently
        if isinstance(state, dict) and "state_dict" in state:
            print("[RS3DAda] note: checkpoint wrapped in 'state_dict'")
            state = state["state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[RS3DAda] !! load_state_dict: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys")
            print(f"   missing[:5]={list(missing)[:5]}")
            print(f"   unexpected[:5]={list(unexpected)[:5]}")
            if len(missing) > 50:
                raise RuntimeError("too many missing keys - wrong checkpoint "
                                   "or wrong encoder size")
        self.model.eval().to(self.device)
        print(f"[RS3DAda] loaded on {self.device}")

    def _forward(self, chw: np.ndarray):
        """Returns the raw output dict. chw is raw 0-255 float RGB."""
        torch = self.torch
        x = (np.transpose(chw, (1, 2, 0)) - RS3DADA_MEAN) / RS3DADA_STD
        x = np.transpose(x, (2, 0, 1))[None]
        t = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(self.device)
        with torch.no_grad():
            if self.amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    return self.model(t)
            return self.model(t)

    def predict_seg(self, chw: np.ndarray) -> np.ndarray:
        """Predicted segmentation class ids (H,W) int16, from the model's own
        8-class head. Used for class-wise correction, so we never need GT
        labels at test time."""
        out = self._forward(chw)
        seg = out["segmentation"] if isinstance(out, dict) else out
        return seg[0].float().argmax(0).cpu().numpy().astype(np.int16)

    def predict_fn(self, chw: np.ndarray) -> np.ndarray:
        """chw is raw 0-255 float RGB. Normalisation happens HERE."""
        torch = self.torch
        x = (np.transpose(chw, (1, 2, 0)) - RS3DADA_MEAN) / RS3DADA_STD
        x = np.transpose(x, (2, 0, 1))[None]
        t = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(self.device)
        with torch.no_grad():
            if self.amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.model(t)
            else:
                out = self.model(t)
        h = out["regression"] if isinstance(out, dict) else out
        return h.squeeze().float().cpu().numpy()


def dummy_backend(constant: float = 0.0, from_image: bool = True):
    """CPU smoke-test backend. Exercises the whole pipeline with no weights.

    from_image=True derives a fake height from image darkness, so the output
    has real spatial structure and blending/TTA bugs actually show up.
    """
    def fn(chw):
        if not from_image:
            return np.full(chw.shape[1:], constant, np.float32)
        lum = chw.mean(0) / 255.0
        return ((1.0 - lum) * 20.0).astype(np.float32)
    return fn


# ==========================================================================
# CLI
# ==========================================================================
def main():
    import argparse
    import glob
    import os
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--rgb-glob", default="*.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["rs3dada", "dummy"], default="rs3dada")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--synrs3d-dir", default="SynRS3D")
    ap.add_argument("--input-gsd", type=float, required=True,
                    help="GSD of the input imagery in metres (GAMUS = 0.3)")
    ap.add_argument("--model-gsd", type=float, default=RS3DADA_TRAIN_GSD,
                    help="GSD the model expects; RS3DAda spans 0.05-1.0 m")
    ap.add_argument("--no-gsd-norm", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--tile", type=int, default=RS3DADA_PATCH)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--median3", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.rgb_dir, a.rgb_glob)))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"no inputs matched {a.rgb_dir}/{a.rgb_glob}")

    if a.backend == "rs3dada":
        if not a.ckpt:
            from huggingface_hub import hf_hub_download
            a.ckpt = hf_hub_download("JTRNEO/RS3DAda",
                                     "RS3DAda_vitl_DPT_height.pth")
        fn = RS3DAda(a.ckpt, a.synrs3d_dir).predict_fn
    else:
        fn = dummy_backend()

    import data as D
    t0 = time.time()
    for i, p in enumerate(files):
        arr, gsd, _ = D.read_array(p)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, 2)
        arr = arr[..., :3].astype(np.float32)
        if arr.max() <= 1.5:              # [0,1] -> 0-255 for RS3DAda
            arr = arr * 255.0

        h = predict_height(arr, fn, input_gsd=(gsd or a.input_gsd),
                           model_gsd=a.model_gsd, tile=a.tile,
                           overlap=a.overlap, tta=not a.no_tta,
                           median3=a.median3,
                           gsd_normalise=not a.no_gsd_norm,
                           verbose=(i == 0))
        stem = D.stem_of(p)
        np.save(os.path.join(a.out, f"{stem}_pred.npy"), h)
        if i % 5 == 0 or i == len(files) - 1:
            el = time.time() - t0
            print(f"[predict] {i+1}/{len(files)} {stem} "
                  f"range=[{h.min():.2f},{h.max():.2f}] m  "
                  f"{el:.1f}s ({el/(i+1):.1f}s/tile)")
    print(f"[predict] done -> {a.out}")


if __name__ == "__main__":
    main()
