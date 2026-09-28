"""
docs/arch_features.py - two more real thumbnails for the diagram:

  india_tiles.jpg     the Chungthang image with the EXACT tiles inference uses
                      (0.305 m -> 0.5 m: 2048 px -> 1248 px; 608-px tiles,
                      stride 456 -> starts 0 / 456 / 640 on each axis)
  india_features.jpg  "what the AI sees": the first 3 principal components of
                      the frozen DINOv3-SAT features on the same image, as RGB.
                      Real encoder output (needs the cached DINOv3 weights).

    HF_HUB_OFFLINE=1 python docs/arch_features.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "thumbs")
PX = 420


def load_rgb():
    import rasterio
    with rasterio.open(os.path.join(ROOT, "samples", "chungthang_wv2.tif")) as src:
        return np.transpose(src.read([1, 2, 3]), (1, 2, 0))


def tiles_thumb(rgb):
    im = Image.fromarray(rgb).resize((PX, PX), Image.LANCZOS).convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    n, tile = 1248, 608
    starts = [0, 456, 640]
    cols = [(255, 196, 0), (0, 200, 255), (255, 80, 160)]
    for j, y in enumerate(starts):
        for i, x in enumerate(starts):
            c = cols[(i + j) % 3]
            x0, y0 = x / n * PX, y / n * PX
            x1, y1 = (x + tile) / n * PX - 1, (y + tile) / n * PX - 1
            d.rectangle([x0, y0, x1, y1], outline=c + (255,), width=4)   # outlines only: overlaps stay readable
    Image.alpha_composite(im, over).convert("RGB").save(os.path.join(OUT, "india_tiles.jpg"),
                                                        quality=90)
    print("  wrote india_tiles.jpg")


def features_thumb(rgb):
    import train_head as TH
    enc = TH.DinoV3Encoder(log=lambda *a: None)
    x = np.asarray(Image.fromarray(rgb).resize((1024, 1024), Image.BILINEAR), np.float32)
    f = enc.encode(np.transpose(x, (2, 0, 1)))[0, 3].float().cpu().numpy()   # deepest level
    C, h, w = f.shape
    X = f.reshape(C, -1).T
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pcs = X @ vt[:3].T                                                        # (h*w, 3)
    lo, hi = np.percentile(pcs, 2, axis=0), np.percentile(pcs, 98, axis=0)
    rgbf = np.clip((pcs - lo) / (hi - lo), 0, 1).reshape(h, w, 3)
    img = Image.fromarray((rgbf * 255).astype(np.uint8)).resize((PX, PX), Image.BICUBIC)
    img.save(os.path.join(OUT, "india_features.jpg"), quality=90)
    print(f"  wrote india_features.jpg (features {C} x {h} x {w}, device {enc.device})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rgb = load_rgb()
    tiles_thumb(rgb)
    features_thumb(rgb)
