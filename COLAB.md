# Train the DepthWizard model on Google Colab

You do **not** need to know how to code. You copy two short blocks, press play,
and wait about an hour. If anything goes wrong, you copy the red text and send it
to the team lead.

**What this does:** trains our height model (a satellite-pretrained DINOv3 encoder
plus a small head we train ourselves) on real aerial imagery with laser-measured
heights, then measures its accuracy on 40 held-out test images. Those are the
same 40 images our baseline was scored on, so the comparison is fair.

---

## Before you start (5 minutes, once)

You need:

1. **A Google account** with about **1 GB free** in Google Drive.
2. **A HuggingFace token from the team lead.** The model we use is gated, so
   the team lead's account has access and you use their token.
   - Team lead: at https://huggingface.co/settings/tokens click **Create new
     token**, choose type **Read**, name it `colab-teammate`, copy it, and send
     it to your teammate **privately** (not in a group chat). Delete that token
     after the hackathon.

---

## Step 1 — Open Colab with a GPU

1. Go to https://colab.research.google.com and click **New notebook**.
2. Top menu: **Runtime → Change runtime type → T4 GPU → Save**.

## Step 2 — Add the token as a secret (never paste it into a cell)

1. In the left sidebar, click the **key icon** (Secrets).
2. Click **Add new secret**.
3. **Name:** `HF_TOKEN` (exactly this, capital letters).
4. **Value:** paste the token the team lead sent you.
5. Turn **ON** the **Notebook access** switch next to it.

## Step 3 — Cell 1: paste this, press play

```
from google.colab import drive
drive.mount('/content/drive')
!pip -q install -U "transformers>=4.56" h5py "huggingface_hub[hf_xet]"
!rm -rf /content/code && git clone -q --depth 1 https://github.com/AHSharan/SIH2026PS175 /content/code
```

A Google pop-up will ask to connect to Drive. Click **Connect** and allow it.
Wait until it finishes (about 1–2 minutes).

## Step 4 — Cell 2: click **+ Code** to add a new cell, paste this, press play

```
%run /content/code/colab_train.py
```

## Step 5 — Wait (about 45–60 minutes) and keep the tab open

It prints what it is doing. Roughly:

| Stage | What you see | Time |
|---|---|---|
| Setup | `[setup] torch ... GPU: Tesla T4`, `[dinov3] ...` checks | 2–3 min |
| 1/3 Cache | `downloaded ... files`, `encoded 25/480 ...` | ~10 min |
| 2/3 Train | `[train] epoch  1/30  loss ...  val RMSE ... m` (30 lines) | ~20–30 min |
| 3/3 Evaluate | tables of numbers, `GSD robustness sweep`, `[export] ...` | ~10 min |
| Done | `ALL DONE` | |

**Keep the browser tab open.** Free Colab disconnects if the tab is closed or
left completely idle for a long time. Clicking around the page now and then helps.

## Step 6 — When you see `ALL DONE`

1. Open Google Drive → **My Drive → depthwizard**.
2. Send the team lead:
   - **`RESULTS.md`** — the numbers (this is the important one)
   - the **`viewer_assets`** folder (right-click → Download — it downloads as a zip)

That's it.

---

## If Colab disconnects

Nothing is lost. The training checkpoint is saved to your Google Drive after
every epoch.

1. Click **Reconnect** (top right). If it asks, choose the T4 GPU again (Step 1.2).
2. Run **Cell 1** again, then **Cell 2** again.

It prints `RESUMED at epoch N` and carries on from where it stopped. If training
had already finished, it skips straight to the evaluation. The feature cache is
rebuilt automatically, which takes about 10 minutes.

## If you see an error

Copy the **last 30 lines** of the output (the red text and the lines just above it)
and send them to the team lead. Don't try to fix the code. Common ones:

| You see | Do this |
|---|---|
| `No GPU` | Step 1.2 (T4 GPU), then Cell 1 and Cell 2 again |
| `could not read the Colab secret HF_TOKEN` | Step 2 — check the name is exactly `HF_TOKEN` and **Notebook access** is ON |
| `no access to the gated DINOv3 repo` | The token is wrong or expired, or its account has not accepted the DINOv3 licence. Ask the team lead for a new token. |
| `transformers is too old for DINOv3` | Run `!pip install -U "transformers>=4.56"` in a new cell, then **Runtime → Restart session**, then Cell 1 and Cell 2 |
| `CUDA out of memory` | Add a new cell **above** Cell 2 containing `%env DW_BATCH=2`, run it, then run Cell 2 |
| `NOT ON GOOGLE DRIVE` | Cell 1 did not mount Drive — run Cell 1 again and accept the pop-up |

## Don'ts

- Don't paste the token into a code cell or a chat.
- Don't close the tab while it runs.
- Don't edit the code. If something breaks, send the error instead.
