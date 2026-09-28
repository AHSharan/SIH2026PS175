# PNG / JPG accuracy check

40 GAMUS test tiles with airborne-LiDAR ground truth, model `best.pt`, pixel size 0.3 m given. Same pixels, four ways in. All numbers measured by `tests/png_jpg_accuracy.py`.

| input | RMSE (m) | MAE (m) | bias (m) | r | forest RMSE (m) | mean change vs original (m) | 99th pct change (m) | s / tile |
|---|---|---|---|---|---|---|---|---|
| original data (direct) | 4.96 | 2.44 | -0.52 | 0.780 | 5.91 | - | - | 0.9 |
| PNG via upload path | 4.96 | 2.44 | -0.52 | 0.780 | 5.91 | 0.000 | 0.00 | 1.2 |
| JPG q90 via upload path | 4.99 | 2.45 | -0.63 | 0.780 | 5.95 | 0.177 | 1.26 | 1.1 |
| JPG q75 via upload path | 4.96 | 2.46 | -0.56 | 0.782 | 5.94 | 0.260 | 1.80 | 1.1 |

Per landscape (RMSE, m):

| input | urban | sparse | forest |
|---|---|---|---|
| original data (direct) | 5.28 | 2.15 | 5.91 |
| PNG via upload path | 5.28 | 2.15 | 5.91 |
| JPG q90 via upload path | 5.30 | 2.13 | 5.95 |
| JPG q75 via upload path | 5.26 | 2.15 | 5.94 |
