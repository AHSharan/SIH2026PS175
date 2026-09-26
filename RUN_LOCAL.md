# Train on your own GPU machine (e.g. the university RTX A4000)

This is the same training as `COLAB.md`, on a machine you control. It doesn't
disconnect, and the A4000 (16 GB) is faster than Colab's T4. It takes about
30–45 minutes, and everything is saved under `dw_run/` in the repo folder.

Run the commands in a terminal (Windows: *Anaconda Prompt* or *PowerShell*;
Linux: any terminal).

## 1. Get the code

```bash
git clone https://github.com/AHSharan/SIH2026PS175
```

```bash
cd SIH2026PS175
```

## 2. Check that PyTorch can see the GPU

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

- If it prints `True ... RTX A4000`, go to step 3.
- If it prints `False`, or says there is no module named torch: open
  https://pytorch.org/get-started/locally/, choose *Stable / your OS / Pip /
  Python / CUDA 12.x*, run the `pip install` command it shows, then run the
  check again. (`nvidia-smi` shows the driver's CUDA version. Any 12.x wheel
  works with a 12.x driver.)

## 3. Install the rest and log in to HuggingFace (once)

```bash
pip install -r requirements.txt
```

```bash
hf auth login
```

Paste a **Read** token from an account that has accepted the DINOv3 licence.
Ask the team lead for one if yours hasn't, and never share it in a group chat.
Older installs use `huggingface-cli login` instead of `hf auth login`.

## 4. Run it

```bash
python colab_train.py
```

It prints the same stages as the Colab version: setup checks → cache → 30
training epochs → evaluation. If it stops for any reason, run the same command
again. It resumes; it does not start over.

## 5. When it prints `ALL DONE`

Send the team lead:

- `dw_run/RESULTS.md` — the numbers
- the `dw_run/viewer_assets/` folder (zip it)

To look at the predictions in 3D yourself: copy the files from
`dw_run/viewer_assets/` into `web/assets_pred/`, run `python serve.py`, and open
http://localhost:8777/?assets=assets_pred

## Errors

Copy the last 30 lines and send them. The common ones are in the table in
`COLAB.md`. On your own machine, `CUDA out of memory` is fixed by setting a
smaller batch:

- Windows PowerShell: `$env:DW_BATCH=2; python colab_train.py`
- Linux: `DW_BATCH=2 python colab_train.py`
