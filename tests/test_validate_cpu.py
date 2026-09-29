"""CPU tests for validate.py scoring (no model, no files).

    python tests/test_validate_cpu.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import validate as V  # noqa: E402

rng = np.random.default_rng(0)
H = W = 256
base = 500 + np.linspace(0, 80, W)[None, :].repeat(H, 0)       # sloping terrain
obj = np.clip(rng.gamma(1.5, 3.0, (H, W)), 0, 40)               # object heights
ok = 0


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok += bool(cond)
    assert cond, name


# 1. perfect prediction -> zero error in every row
rows, _ = V.score(base, obj, base + obj)
check("perfect: raw RMSE 0", rows[0]["rmse_m"] < 1e-6)

# 2. a constant datum offset is removed by 'Datum shift', not by 'Raw'
rows, _ = V.score(base, obj, base + obj + 7.0)
check("offset: raw bias -7", abs(rows[0]["bias_m"] + 7.0) < 1e-6)
check("offset: datum shift removes it", rows[1]["rmse_m"] < 1e-6)

# 3. model heights 1.3x too small: affine recovers the scale, shift cannot
rows, _ = V.score(base, obj / 1.3, base + obj)
check("scale: affine beats shift", rows[2]["rmse_m"] < 0.2 * rows[1]["rmse_m"])
check("scale: affine finds ~1.3", abs(rows[2]["scale"] - 1.3) < 0.05)

# 4. fitted rows are scored on held-out pixels: pure noise cannot be fitted away
noise = rng.normal(0, 2.0, (H, W))
rows, _ = V.score(base, obj, base + obj + noise)
check("noise: affine not better than raw by fitting noise",
      rows[2]["rmse_m"] > 0.95 * rows[0]["rmse_m"])

# 5. reference gaps (NaN) are ignored, not counted as errors
ref = base + obj
ref[:, :100] = np.nan
rows, valid = V.score(base, obj, ref)
check("gaps: only covered pixels scored", rows[0]["n_px"] == valid.sum() == H * (W - 100))

print(f"\n{ok}/7 passed")
