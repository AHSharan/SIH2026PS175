# DepthWizard - DINOv3-SAT height head: results

Generated 2026-09-26 14:43 by `colab_train.py`. Every number here was measured in this run.

## Setup

- Training tiles: 480 (GAMUS train split), validation: 48 (GAMUS val split) - model selected on val, never on test
- Test tiles: 40 (GAMUS test split) - **identical to the Kaggle RS3DAda run**
- Encoder: `facebook/dinov3-vitl16-pretrain-sat493m`, frozen; features cached in fp16
- Head: DPT-style, trained 30 epochs, batch 4, AdamW lr 0.0001, cosine schedule, masked Huber loss on metres, target clipped to [0, 150] m
- Training GSD 0.5 m (GAMUS 0.3 m resampled 1024 -> 608 px); inputs are GSD-normalised to it at inference

Encoder self-checks (read at runtime, not assumed):

```json
{
  "repo": "facebook/dinov3-vitl16-pretrain-sat493m",
  "model_class": "DINOv3ViTModel",
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "num_register_tokens": 4,
  "patch_size": 16,
  "layers_used": [
    6,
    12,
    18,
    24
  ],
  "normalisation_used": {
    "mean": [
      0.43,
      0.411,
      0.296
    ],
    "std": [
      0.213,
      0.156,
      0.143
    ],
    "source": "facebookresearch/dinov3 README, SAT-493M"
  },
  "processor_check": {
    "mean": [
      0.43,
      0.411,
      0.296
    ],
    "std": [
      0.213,
      0.156,
      0.143
    ],
    "verdict": "matches the official SAT-493M values (good)"
  },
  "layout": {
    "hidden_states_path": "output_hidden_states (embeddings + blocks)",
    "seq_len_at_64px": 21,
    "patch_tokens": 16,
    "prefix_tokens": 5,
    "expected_prefix": 5
  }
}
```

## Training

Best validation RMSE **3.765 m** at epoch 22 (checkpoint `ckpt/best.pt`).

| epoch | train loss | val RMSE (m) | val MAE (m) | val bias (m) |
|---|---|---|---|---|
| 1 | 3.632 | 6.758 | 3.396 | -2.207 |
| 4 | 2.215 | 4.262 | 2.391 | -0.298 |
| 7 | 1.928 | 4.003 | 2.169 | +0.376 |
| 10 | 1.777 | 4.094 | 2.186 | -0.474 |
| 13 | 1.669 | 4.607 | 2.325 | +0.782 |
| 16 | 1.585 | 3.811 | 1.962 | -0.137 |
| 19 | 1.509 | 3.778 | 1.980 | +0.182 |
| 22 | 1.464 | 3.765 | 1.966 | -0.285 |
| 25 | 1.435 | 3.815 | 1.977 | -0.090 |
| 28 | 1.417 | 3.815 | 1.972 | -0.069 |
| 30 | 1.410 | 3.805 | 1.965 | -0.118 |

## Test results

### B0  predict zero (floor)

| bucket | tiles | px | RMSE (m) | MAE (m) | bias (m) | r | <1m | <2.5m | <5m |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 40 | 41.9M | 9.35 | 5.10 | -5.05 | nan | 49.8% | 55.4% | 63.6% |
| **urban** | 24 | 25.1M | 8.81 | 4.74 | -4.73 | nan | 47.1% | 53.1% | 62.3% |
| **sparse** | 8 | 8.4M | 4.21 | 1.94 | -1.72 | nan | 73.8% | 77.9% | 83.6% |
| **forest** | 8 | 8.4M | 13.66 | 9.35 | -9.34 | nan | 34.1% | 40.0% | 47.7% |

### C   RS3DAda zero-shot (baseline)

| bucket | tiles | px | RMSE (m) | MAE (m) | bias (m) | r | <1m | <2.5m | <5m |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 40 | 41.9M | 6.74 | 3.47 | -2.27 | 0.597 | 48.6% | 65.7% | 79.9% |
| **urban** | 24 | 25.1M | 5.11 | 2.68 | -1.33 | 0.757 | 47.4% | 66.9% | 83.8% |
| **sparse** | 8 | 8.4M | 2.83 | 1.43 | -0.32 | 0.684 | 70.6% | 83.1% | 91.5% |
| **forest** | 8 | 8.4M | 11.87 | 7.87 | -7.07 | 0.306 | 30.1% | 44.5% | 56.7% |

### H1  DINOv3-SAT head (ours)

| bucket | tiles | px | RMSE (m) | MAE (m) | bias (m) | r | <1m | <2.5m | <5m |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 40 | 41.9M | 4.96 | 2.44 | -0.52 | 0.780 | 52.5% | 70.5% | 85.4% |
| **urban** | 24 | 25.1M | 5.28 | 2.40 | -0.81 | 0.726 | 51.9% | 71.6% | 86.9% |
| **sparse** | 8 | 8.4M | 2.15 | 1.12 | +0.02 | 0.829 | 73.5% | 85.8% | 95.1% |
| **forest** | 8 | 8.4M | 5.91 | 3.87 | -0.20 | 0.807 | 32.9% | 51.8% | 71.5% |

### Overall, side by side

| run | RMSE (m) | MAE (m) | bias (m) | r | forest RMSE (m) |
|---|---|---|---|---|---|
| B0  predict zero (floor) | 9.35 | 5.10 | -5.05 | nan | 13.66 |
| C   RS3DAda zero-shot (baseline) | 6.74 | 3.47 | -2.27 | 0.597 | 11.87 |
| H1  DINOv3-SAT head (ours) | 4.96 | 2.44 | -0.52 | 0.780 | 5.91 |
| *reference: Kaggle RS3DAda run C* | 6.74 | 3.47 | -2.27 | 0.597 | 11.87 |

## GSD robustness sweep

Each test image degraded to the given GSD, predicted, and scored against ground truth resampled to the same grid. Coarser GT is also smoother, so part of any change is the target changing, not only the model. The 0.30 m row reaches the model through one resample and the others through two (slightly smoother), so read the 0.50 -> 1.00 m trend, not the first step.

| input GSD (m) | DINOv3-SAT head (ours) RMSE / r | RS3DAda zero-shot RMSE / r |
|---|---|---|
| 0.30 | 4.85 / 0.789 | 6.82 / 0.580 |
| 0.50 | 4.60 / 0.808 | 6.70 / 0.585 |
| 0.75 | 4.68 / 0.800 | 6.83 / 0.572 |
| 1.00 | 5.46 / 0.723 | 6.92 / 0.564 |

## Honest caveats (keep these in the deck)

- Trained and tested on GAMUS (US cities, aerial). The test split is held out, but ISRO evaluation imagery is a different sensor and region, so these numbers will overstate performance there.
- nDSM only: terrain is removed, so there is no 'hilly' bucket without a DEM.
- Fixed 0.5 m training GSD with flip/rotation augmentation only; colour augmentation was not possible on cached features.
- GAMUS GSD (0.3 m) is asserted from the DFC2019 spec and a car-length check, not read from the files.

## Viewer assets

`viewer_assets/` holds 3 scenes with REAL predictions (PHL_3681, DC_39_18, PHL_3844). Copy the folder's files into the repo's `web/assets/`, run `python serve.py`, open http://localhost:8777 - the badge reads MODEL PREDICTION and the Error overlay + live RMSE panel work.
