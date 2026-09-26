"""
train_head.py - frozen DINOv3-SAT ViT-L/16 encoder + small trainable DPT-style
height head, trained on CACHED encoder features.

Design, including where it deliberately departs from the original spec:

  * The encoder is frozen, so its features are computed ONCE and cached in fp16.
    Training then runs only the small head: fast, and fits a free T4 easily.

  * One fixed training GSD of 0.5 m. GAMUS 1024 px @ 0.3 m is resampled to
    608 px (38 x 38 patches of 16 px). 0.5 m sits inside the Cartosat 0.3-1 m
    range and makes the cache ~2.8x smaller than native resolution. At inference
    every input is GSD-normalised to 0.5 m by predict.predict_height - for THIS
    model (unlike RS3DAda) GSD normalisation is required, not optional.

  * DEVIATION - augmentation is flips + 90-degree rotations only, applied to the
    cached feature grid and the target together. The spec also asked for
    brightness/contrast and random-GSD augmentation, but those act on PIXELS
    before the encoder and cannot be applied to cached features. Cost: less
    robustness to colour/resolution shift. colab_train.py measures the GSD side
    of that with a sweep instead of assuming.

  * DEVIATION - heights are clipped to [0, 150] m for the loss, not [0, 80].
    GAMUS PHL tiles contain 146 m towers (measured); an 80 m clip would teach
    the model those buildings are 80 m tall.

  * Normalisation: DINOv3 SAT-493M weights use mean (0.430, 0.411, 0.296),
    std (0.213, 0.156, 0.143) per the official facebookresearch/dinov3 README -
    NOT ImageNet statistics. The HF preprocessor config is gated, so it is read
    at runtime and logged; the official SAT values are always used.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-sat493m"
SAT_MEAN = (0.430, 0.411, 0.296)        # official dinov3 README, SAT-493M
SAT_STD = (0.213, 0.156, 0.143)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATCH = 16
SRC_GSD = 0.3          # GAMUS (asserted from the DFC2019 spec, see data.py)
TRAIN_GSD = 0.5
HCLIP = 150.0


# ==========================================================================
# grids and IO
# ==========================================================================
def work_size(hw, src_gsd=SRC_GSD, gsd=TRAIN_GSD, mult=PATCH):
    """Grid the model runs on. MUST match predict.predict_height's arithmetic
    exactly, otherwise training and inference see differently-resampled images."""
    H0, W0 = int(hw[0]), int(hw[1])
    if abs(src_gsd - gsd) <= 1e-9:
        return H0 - H0 % mult, W0 - W0 % mult
    s = src_gsd / gsd
    Hn = max(mult, int(round(H0 * s)))
    Wn = max(mult, int(round(W0 * s)))
    return Hn - Hn % mult, Wn - Wn % mult


def load_rgb_255(path):
    """RGB as float32 0-255 (H,W,3) - the same convention kaggle_run.py uses."""
    import data as D
    a, _, _ = D.read_array(path)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, 2)
    a = a[..., :3].astype(np.float32)
    return a * 255.0 if a.max() <= 1.5 else a


def load_height(path, sentinel=-5.0):
    """GT heights with the same rules as data.load_tile: invalid -> NaN, small
    negatives (LiDAR noise) -> 0."""
    import data as D
    h, _, nd = D.read_array(path)
    h = np.asarray(h, np.float32)
    m = np.isfinite(h)
    if sentinel is not None:
        m &= h != np.float32(sentinel)
    if nd is not None:
        m &= h != np.float32(nd)
    h = np.where(m, h, np.nan)
    return np.where(m & (h < 0), 0.0, h).astype(np.float32)


def resample_height(h, out_hw):
    """NaN-aware bilinear resample: invalid pixels never leak into valid ones."""
    import predict as P
    out_hw = (int(out_hw[0]), int(out_hw[1]))
    if tuple(h.shape) == out_hw:
        return h.astype(np.float32)
    valid = np.isfinite(h).astype(np.float32)
    filled = np.where(valid > 0, h, 0.0).astype(np.float32)
    num = P.resample_hw(filled, out_hw, order=1)
    den = P.resample_hw(valid, out_hw, order=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / np.maximum(den, 1e-6)
    return np.where(den > 0.5, out, np.nan).astype(np.float32)


def _atomic_np_save(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def _atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def autocast_ctx(device, amp):
    dev = device if isinstance(device, str) else device.type
    if amp and dev == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ==========================================================================
# encoders
# ==========================================================================
class DinoV3Encoder:
    """Frozen DINOv3 ViT. encode(chw 0-255) -> (B, 4, C, h, w) fp16 on device."""

    def __init__(self, repo=DINOV3_REPO, token=None, device=None, amp=True,
                 layers=None, log=print):
        from transformers import AutoModel
        self.log = log
        self.repo = repo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(amp and self.device == "cuda")
        self.n_fp32_fallback = 0
        self.n_clamped = 0
        try:
            model = AutoModel.from_pretrained(repo, token=token)
        except Exception as e:                                  # noqa: BLE001
            raise RuntimeError(self._load_hint(e)) from e
        # fp32 weights + fp16 autocast; .float() also undoes any "auto" dtype a
        # newer transformers release might pick on load
        self.model = model.float().eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        self.hidden = int(cfg.hidden_size)
        self.n_layers = int(cfg.num_hidden_layers)
        self.n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.patch = int(getattr(cfg, "patch_size", PATCH) or PATCH)
        L = self.n_layers
        self.layers = list(layers) if layers else [L // 4, L // 2, (3 * L) // 4, L]
        self.mean = (np.asarray(SAT_MEAN, np.float32) * 255.0)[:, None, None]
        self.std = (np.asarray(SAT_STD, np.float32) * 255.0)[:, None, None]
        self.hs_path = None
        self.report = {
            "repo": repo, "model_class": type(self.model).__name__,
            "hidden_size": self.hidden, "num_hidden_layers": L,
            "num_register_tokens": self.n_reg, "patch_size": self.patch,
            "layers_used": self.layers,
            "normalisation_used": {"mean": SAT_MEAN, "std": SAT_STD,
                                   "source": "facebookresearch/dinov3 README, SAT-493M"},
        }
        self._check_processor(token)
        self._check_layout()

    # -------------------------------------------------------------- checks
    @staticmethod
    def _load_hint(e):
        msg = f"{type(e).__name__}: {e}"
        low = msg.lower()
        if "dinov3" in low and any(k in low for k in ("recogn", "keyerror", "does not", "unknown")):
            hint = ("transformers is too old for DINOv3. Run:  !pip install -U "
                    "\"transformers>=4.56\"  then Runtime -> Restart session, "
                    "then run cell 1 and cell 2 again.")
        elif any(k in low for k in ("401", "403", "gated", "unauthorized",
                                    "access to model", "restricted", "awaiting")):
            hint = ("no access to the gated DINOv3 repo. Check that the Colab "
                    "secret HF_TOKEN exists, that its 'Notebook access' switch is "
                    "ON, and that the token's account accepted the licence at "
                    "huggingface.co/" + DINOV3_REPO)
        else:
            hint = "unexpected - send this whole error to the team lead"
        return f"could not load {DINOV3_REPO}\n  {msg[:600]}\n  HINT: {hint}"

    def _check_processor(self, token):
        try:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(self.repo, token=token)
            pm = [float(v) for v in (getattr(proc, "image_mean", None) or [])]
            ps = [float(v) for v in (getattr(proc, "image_std", None) or [])]
        except Exception as e:                                  # noqa: BLE001
            self.report["processor_check"] = f"could not read processor ({type(e).__name__})"
            self.log(f"[dinov3] could not read the HF processor ({type(e).__name__}); "
                     f"using official SAT normalisation")
            return

        def close(a, b):
            return len(a) == 3 and bool(np.allclose(a, b, atol=2e-3))
        if close(pm, SAT_MEAN) and close(ps, SAT_STD):
            verdict = "matches the official SAT-493M values (good)"
        elif close(pm, IMAGENET_MEAN) and close(ps, IMAGENET_STD):
            verdict = "ImageNet values - DISAGREES with the official SAT guidance"
        else:
            verdict = "unrecognised values"
        self.report["processor_check"] = {"mean": pm, "std": ps, "verdict": verdict}
        self.log(f"[dinov3] HF processor mean={pm} std={ps} -> {verdict}")
        if "good" not in verdict:
            self.log("[dinov3] !! using the official SAT values regardless: "
                     "mean=(0.430,0.411,0.296) std=(0.213,0.156,0.143)")

    def _check_layout(self):
        s = 4 * self.patch                          # 64x64 -> 4x4 patch grid
        x = torch.zeros(1, 3, s, s, device=self.device)
        hs = self._hidden_states(x, amp=False)
        n = (s // self.patch) ** 2
        seq = int(hs[-1].shape[1])
        prefix = seq - n
        expect = 1 + self.n_reg
        self.report["layout"] = {"hidden_states_path": self.hs_path,
                                 "seq_len_at_64px": seq, "patch_tokens": n,
                                 "prefix_tokens": prefix, "expected_prefix": expect}
        self.log(f"[dinov3] hidden states via {self.hs_path}: {len(hs)} entries, "
                 f"seq={seq}, patch tokens={n}, prefix={prefix} "
                 f"(expected 1 cls + {self.n_reg} registers = {expect})")
        if prefix < 0:
            raise RuntimeError("fewer tokens than patches - unexpected DINOv3 layout")
        if prefix != expect:
            self.log("[dinov3] !! prefix differs from 1+registers; still taking the "
                     "LAST patch tokens (DINOv3 orders tokens cls, registers, patches)")

    # ------------------------------------------------------------- forward
    def _hidden_states(self, x, amp):
        with torch.no_grad(), autocast_ctx(self.device, amp):
            out = self.model(pixel_values=x, output_hidden_states=True)
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            hs = list(hs)
            if len(hs) == self.n_layers + 1:
                self.hs_path = "output_hidden_states (embeddings + blocks)"
                return hs
            if len(hs) == self.n_layers:
                self.hs_path = "output_hidden_states (blocks only)"
                return [None] + hs
        self.hs_path = "forward hooks on the transformer blocks"
        return self._hooked(x, amp)

    def _hooked(self, x, amp):
        blocks = None
        for _, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) == self.n_layers:
                blocks = mod
                break
        if blocks is None:
            raise RuntimeError("could not locate the transformer blocks for hooks")
        outs = []

        def grab(_m, _i, o):
            outs.append(o[0] if isinstance(o, (tuple, list)) else o)
        hooks = [b.register_forward_hook(grab) for b in blocks]
        try:
            with torch.no_grad(), autocast_ctx(self.device, amp):
                self.model(pixel_values=x)
        finally:
            for h in hooks:
                h.remove()
        return [None] + outs

    def _levels(self, t, h, w, amp):
        hs = self._hidden_states(t, amp)
        n = h * w
        lv = []
        for li in self.layers:
            z = hs[li][:, -n:, :].float()          # drop cls + register tokens
            lv.append(z.transpose(1, 2).reshape(z.shape[0], z.shape[2], h, w))
        return torch.stack(lv, 1)                  # (B,4,C,h,w) float32

    def encode(self, chw):
        """chw: (3,H,W) or (B,3,H,W) float RGB 0-255, H and W multiples of 16."""
        x = np.asarray(chw, np.float32)
        if x.ndim == 3:
            x = x[None]
        _, _, H, W = x.shape
        if H % self.patch or W % self.patch:
            raise ValueError(f"input {H}x{W} is not a multiple of {self.patch}")
        x = (x - self.mean[None]) / self.std[None]
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        h, w = H // self.patch, W // self.patch
        f = self._levels(t, h, w, self.amp)
        if self.amp and not bool(torch.isfinite(f).all()):
            self.n_fp32_fallback += 1               # fp16 overflow inside the ViT
            f = self._levels(t, h, w, False)
        big = f.abs() > 65000
        if bool(big.any()):
            self.n_clamped += int(big.sum())
            f = f.clamp(-65000, 65000)
        return f.to(torch.float16)


class FakeEncoder:
    """CPU stand-in with the SAME interface as DinoV3Encoder. TESTS ONLY.

    Features are a fixed nonlinear function of per-patch colour statistics, so
    a head can genuinely learn something and the plumbing is exercised for real.
    """

    def __init__(self, hidden=32, patch=PATCH, seed=0):
        self.device = "cpu"
        self.amp = False
        self.hidden = hidden
        self.patch = patch
        self.n_layers = 4
        self.n_reg = 4
        self.layers = [1, 2, 3, 4]
        self.n_fp32_fallback = 0
        self.n_clamped = 0
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, 1, (6, hidden)).astype(np.float32) for _ in range(4)]
        self.report = {"model_class": "FakeEncoder (TEST ONLY - not DINOv3)"}

    def encode(self, chw):
        x = np.asarray(chw, np.float32)
        if x.ndim == 3:
            x = x[None]
        B, C, H, W = x.shape
        p = self.patch
        h, w = H // p, W // p
        xb = x[:, :, :h * p, :w * p].reshape(B, C, h, p, w, p) / 255.0
        stats = np.concatenate([xb.mean(axis=(3, 5)), xb.std(axis=(3, 5))], 1)
        lv = [np.tanh(np.einsum("bkhw,kc->bchw", stats, Wm) * (i + 1))
              for i, Wm in enumerate(self.W)]
        return torch.from_numpy(np.stack(lv, 1).astype(np.float16))


# ==========================================================================
# head
# ==========================================================================
class ChannelLN(nn.Module):
    """LayerNorm over channels of a (B,C,H,W) map - per token, like the ViT."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class RCU(nn.Module):
    """Residual conv unit (DPT)."""

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class HeightHead(nn.Module):
    """4 ViT levels -> linear projection -> DPT-style reassemble at 4x/2x/1x/0.5x
    -> top-down progressive upsample-and-fuse -> 1-channel height map."""

    def __init__(self, in_dim=1024, dim=256, n_levels=4):
        super().__init__()
        self.in_dim, self.dim = in_dim, dim
        self.proj = nn.ModuleList([nn.Sequential(ChannelLN(in_dim), nn.Conv2d(in_dim, dim, 1))
                                   for _ in range(n_levels)])
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(dim, dim, 4, stride=4),
            nn.ConvTranspose2d(dim, dim, 2, stride=2),
            nn.Identity(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1)])
        self.rcu = nn.ModuleList([RCU(dim) for _ in range(n_levels)])
        self.fuse = nn.ModuleList([RCU(dim) for _ in range(n_levels)])
        self.out1 = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1), nn.ReLU())
        self.out2 = nn.Sequential(nn.Conv2d(dim // 2, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 1, 1))

    def forward(self, feats, out_hw):
        xs = [self.rcu[i](self.resample[i](self.proj[i](feats[:, i].float())))
              for i in range(feats.shape[1])]
        f = self.fuse[3](xs[3])
        for i in (2, 1, 0):
            f = F.interpolate(f, size=xs[i].shape[-2:], mode="bilinear", align_corners=False)
            f = self.fuse[i](f + xs[i])
        f = self.out1(f)
        f = F.interpolate(f, size=tuple(int(v) for v in out_hw), mode="bilinear",
                          align_corners=False)
        return self.out2(f)[:, 0]


def load_head(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    head = HeightHead(ck["in_dim"], ck["head_dim"])
    head.load_state_dict(ck["head"])
    return head.eval().to(device), ck


def make_predict_fn(encoder, head):
    """predict.py contract: chw float 0-255 (3,H,W) -> (H,W) metres."""
    dev = encoder.device

    def fn(chw):
        H, W = chw.shape[-2:]
        f = encoder.encode(chw).to(dev)
        with torch.no_grad(), autocast_ctx(dev, encoder.amp):
            p = head(f, (H, W))
        return p[0].float().cpu().numpy()
    return fn


# ==========================================================================
# feature cache
# ==========================================================================
def cache_paths(cache_dir, stem):
    return (os.path.join(cache_dir, f"{stem}_f.npy"),
            os.path.join(cache_dir, f"{stem}_t.npy"))


def is_cached(cache_dir, stem):
    fp, tp = cache_paths(cache_dir, stem)
    return os.path.exists(fp) and os.path.exists(tp)


def cache_tile(rec, encoder, cache_dir, src_gsd=SRC_GSD, gsd=TRAIN_GSD):
    """Encode one tile at the training GSD and store features + target.
    Returns False if it was already cached (resume-safe)."""
    import predict as P
    os.makedirs(cache_dir, exist_ok=True)
    fp, tp = cache_paths(cache_dir, rec["tile_id"])
    if os.path.exists(fp) and os.path.exists(tp):
        return False
    rgb = load_rgb_255(rec["rgb"])
    h = load_height(rec["height"])
    hw = work_size(h.shape, src_gsd, gsd, encoder.patch)
    rgb_w = P.resample_hw(rgb, hw, order=1)      # the exact call predict_height makes
    feats = encoder.encode(np.transpose(rgb_w, (2, 0, 1)))[0].cpu().numpy().astype(np.float16)
    tgt = resample_height(h, hw)
    _atomic_np_save(tp, tgt)
    _atomic_np_save(fp, feats)                    # written LAST = completion marker
    return True


def verify_cache(cache_dir, log=print):
    """Drop unreadable / inconsistent entries so a crash mid-write cannot poison
    training. Returns the number of good tiles."""
    if not os.path.isdir(cache_dir):
        return 0
    good = bad = 0
    ref_c = None
    for fn in sorted(os.listdir(cache_dir)):
        path = os.path.join(cache_dir, fn)
        if fn.endswith(".tmp"):
            os.remove(path)
            continue
        if not fn.endswith("_f.npy"):
            continue
        stem = fn[:-len("_f.npy")]
        fp, tp = cache_paths(cache_dir, stem)
        try:
            f = np.load(fp, mmap_mode="r")
            t = np.load(tp, mmap_mode="r")
            ok = (f.ndim == 4 and t.ndim == 2 and f.shape[0] == 4
                  and t.shape == (f.shape[2] * PATCH, f.shape[3] * PATCH))
            ref_c = f.shape[1] if ref_c is None else ref_c
            ok = ok and f.shape[1] == ref_c
            del f, t
        except Exception:                                       # noqa: BLE001
            ok = False
        if ok:
            good += 1
        else:
            bad += 1
            for q in (fp, tp):
                if os.path.exists(q):
                    os.remove(q)
    if bad:
        log(f"[cache] removed {bad} corrupt/inconsistent entries in {cache_dir}")
    return good


def sync_dir(src, dst):
    """Copy new/changed files src -> dst (Drive cache -> fast local disk)."""
    os.makedirs(dst, exist_ok=True)
    n = 0
    for fn in os.listdir(src):
        if fn.endswith(".tmp"):
            continue
        s, d = os.path.join(src, fn), os.path.join(dst, fn)
        if os.path.isfile(s) and (not os.path.exists(d)
                                  or os.path.getsize(d) != os.path.getsize(s)):
            shutil.copyfile(s, d)
            n += 1
    return n


class CachedTiles(torch.utils.data.Dataset):
    def __init__(self, cache_dir, augment=False):
        self.dir = cache_dir
        stems = sorted(fn[:-len("_f.npy")] for fn in os.listdir(cache_dir)
                       if fn.endswith("_f.npy")) if os.path.isdir(cache_dir) else []
        self.stems = [s for s in stems if is_cached(cache_dir, s)]
        self.augment = augment

    def __len__(self):
        return len(self.stems)

    def feature_dim(self):
        fp, _ = cache_paths(self.dir, self.stems[0])
        return int(np.load(fp, mmap_mode="r").shape[1])

    def __getitem__(self, i):
        fp, tp = cache_paths(self.dir, self.stems[i])
        f = torch.from_numpy(np.load(fp))            # (4,C,h,w) fp16
        t = torch.from_numpy(np.load(tp))            # (H,W) fp32, NaN = invalid
        if self.augment:
            # the SAME spatial transform on features and target: 608 = 38*16
            # exactly, so patch/pixel correspondence survives rot90 and flips
            k = int(torch.randint(0, 4, (1,)))
            if k:
                f = torch.rot90(f, k, dims=(-2, -1))
                t = torch.rot90(t, k, dims=(-2, -1))
            if bool(torch.randint(0, 2, (1,))):
                f = torch.flip(f, dims=(-1,))
                t = torch.flip(t, dims=(-1,))
        return f.contiguous(), t.contiguous()


# ==========================================================================
# training
# ==========================================================================
def masked_huber(pred, target, beta=1.0, hclip=HCLIP):
    m = torch.isfinite(target).float()
    t = torch.clamp(torch.nan_to_num(target, nan=0.0), 0.0, hclip)
    loss = F.smooth_l1_loss(pred.float(), t, beta=beta, reduction="none")
    return (loss * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def validate(head, loader, device, amp):
    head.eval()
    se = ae = be = 0.0
    n = 0
    for f, t in loader:
        f, t = f.to(device), t.to(device)
        with autocast_ctx(device, amp):
            p = head(f, t.shape[-2:])
        p = p.float().clamp_min(0.0)
        m = torch.isfinite(t)
        e = (p - torch.nan_to_num(t))[m]
        se += float((e ** 2).sum())
        ae += float(e.abs().sum())
        be += float(e.sum())
        n += int(m.sum())
    head.train()
    n = max(n, 1)
    return {"rmse": (se / n) ** 0.5, "mae": ae / n, "bias": be / n, "n_px": n}


def _grad_scaler(amp):
    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp)


def train(train_dir, val_dir, ckpt_dir, epochs=30, batch=4, lr=1e-4,
          weight_decay=1e-4, warmup_frac=0.05, head_dim=256, hclip=HCLIP,
          beta=1.0, num_workers=2, device=None, seed=0, log=print):
    """Train the head on cached features. Resumes from ckpt_dir/last.pt when it
    exists, so a Colab disconnect costs at most one epoch."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = device == "cuda"
    torch.manual_seed(seed)

    tr = CachedTiles(train_dir, augment=True)
    va = CachedTiles(val_dir, augment=False)
    if len(tr) == 0:
        raise RuntimeError(f"no cached training tiles in {train_dir}")
    if len(va) == 0:
        raise RuntimeError(f"no cached validation tiles in {val_dir}")
    in_dim = tr.feature_dim()
    head = HeightHead(in_dim, head_dim).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    kw = dict(num_workers=num_workers, pin_memory=(device == "cuda"))
    if num_workers > 0:
        kw["persistent_workers"] = True
    tl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True,
                                     drop_last=len(tr) >= batch, **kw)
    vl = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, **kw)

    steps = max(1, len(tl))
    total = epochs * steps
    warm = max(1, int(warmup_frac * total))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = _grad_scaler(amp)

    os.makedirs(ckpt_dir, exist_ok=True)
    last_p = os.path.join(ckpt_dir, "last.pt")
    best_p = os.path.join(ckpt_dir, "best.pt")
    hist_p = os.path.join(ckpt_dir, "history.json")
    start, best, history = 0, float("inf"), []

    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device, weights_only=False)
        if ck.get("in_dim") != in_dim or ck.get("head_dim") != head_dim:
            log(f"[train] !! checkpoint dims {ck.get('in_dim')}/{ck.get('head_dim')} "
                f"do not match data {in_dim}/{head_dim}; starting fresh "
                f"(old file kept as last.pt.stale)")
            os.replace(last_p, last_p + ".stale")
        else:
            head.load_state_dict(ck["head"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            start = int(ck["epoch"]) + 1
            best = float(ck["best_rmse"])
            history = list(ck.get("history", []))
            if ck.get("total_steps") != total:
                log(f"[train] note: schedule length changed ({ck.get('total_steps')} "
                    f"-> {total} steps); continuing")
            log(f"[train] RESUMED at epoch {start + 1}/{epochs} "
                f"(best val RMSE so far {best:.3f} m)")

    log(f"[train] {len(tr)} train / {len(va)} val tiles, feature dim {in_dim}, "
        f"head {n_params / 1e6:.2f}M params, {steps} steps/epoch, device {device}, "
        f"amp={amp}")
    if start >= epochs:
        log("[train] already finished")
        return best_p, history

    for ep in range(start, epochs):
        t0 = time.time()
        head.train()
        run, nb = 0.0, 0
        for f, t in tl:
            f = f.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            with autocast_ctx(device, amp):
                p = head(f, t.shape[-2:])
            loss = masked_huber(p, t, beta, hclip)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            nb += 1
        v = validate(head, vl, device, amp)
        rec = {"epoch": ep + 1, "train_loss": run / max(nb, 1), "val_rmse": v["rmse"],
               "val_mae": v["mae"], "val_bias": v["bias"],
               "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(rec)
        improved = v["rmse"] < best
        if improved:
            best = v["rmse"]
            _atomic_torch_save({"head": head.state_dict(), "in_dim": in_dim,
                                "head_dim": head_dim, "epoch": ep + 1, "val": v}, best_p)
        _atomic_torch_save({"head": head.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                            "epoch": ep, "best_rmse": best, "history": history,
                            "in_dim": in_dim, "head_dim": head_dim,
                            "total_steps": total}, last_p)
        _atomic_json(history, hist_p)
        log(f"[train] epoch {ep + 1:>2}/{epochs}  loss {rec['train_loss']:.3f}  "
            f"val RMSE {v['rmse']:.3f} m  MAE {v['mae']:.3f}  bias {v['bias']:+.3f}  "
            f"{rec['sec']:.0f}s{'  <- best' if improved else ''}")
    return best_p, history
