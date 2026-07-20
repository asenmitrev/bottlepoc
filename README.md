# bottle-cap-poc

Narrow PoC: can a compact, Apache-2.0 edge detector (RF-DETR-Nano/Small)
correctly classify bottle cap condition (Broken Cap / Broken Ring / Good Cap
/ Loose Cap / No Cap), evaluated honestly on a held-out test split?

No anomaly detection, OCR, label alignment, or video pipeline here by
design -- see the parent task spec for why.

## Dataset provenance

The task originally specced pulling **"Bottle Caps"**
(`bottle-defect/bottle-caps-p3mai`, 2,069 images / 570 labeled) via the
Roboflow API. Instead, this repo trains on **"Bottle Defects"** (v28,
`product-defect-detection/bottle-defects-o4gnx`), because a labeled COCO
export of it was already sitting in the parent workspace when this was
built (`../Bottle Defects.v28i.coco/`), with:

- the identical 5-class taxonomy (Broken Cap, Broken Ring, Good Cap, Loose
  Cap, No Cap)
- 623 images total, already split into train/valid/test (398/113/112)
- CC BY 4.0 license, same as the originally-specced dataset

This was a deliberate substitution, confirmed with the requester, not a
silent scope change -- see `NOTICE.md` for the updated citation. It also
happens to sidestep needing a Roboflow API key for this run. `data/download.py`
still supports falling back to a fresh Roboflow API pull (untested, since
the local copy was available) if you need to regenerate this dataset later.

**Class balance**: Good Cap (218 total) is the modal class; Broken Cap (63)
and Broken Ring (66) are both under 30% of it. Flagged automatically by
`data/download.py` and discussed in `report.md`.

**Split composition**: the provided test split is *not* proportionally
stratified relative to train -- e.g. Loose Cap is 41% of test images vs 22%
of train, and Broken Ring is 20.5% of test vs 8.8% of train. This is
Roboflow's existing split, used as-is per the task spec ("respect whatever
split Roboflow's export provides"), but it means small-support per-class
metrics (Broken Cap: n=15, Broken Ring: n=23 in test) should be read with
that caveat -- see `report.md`.

## Environment note

This repo was built on a MacBook (Apple Silicon, no CUDA) with almost no
free disk -- there was no way to install torch/rfdetr or run real training
here. Every script below is written and syntax-checked, and
`data/download.py` has been run for real (it's stdlib-only). **Training,
evaluation, and export need to happen on the actual 2x RTX 3090 machine.**
`report.md` is structured with the real dataset stats filled in and a
placeholder for the numbers that only exist after that run.

## Setup (on the CUDA machine)

```bash
git clone/rsync this bottle-cap-poc/ directory over, then:
cd bottle-cap-poc
uv venv
source .venv/bin/activate

# Install a CUDA-matched torch/torchvision FIRST, then the rest -- letting
# `uv sync` resolve torch on its own can silently grab a CPU wheel.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .

cp .env.example .env   # only needed if you use the Roboflow-API fallback
```

## Reproduce

```bash
# 1. Materialize the dataset + print/report class counts per split.
python data/download.py

# 2. Fine-tune from the RF-DETR COCO pretrained checkpoint.
#    Small is the default (more headroom than Nano at this class count);
#    pass --model nano to compare.
python train.py --model small --epochs 60 --batch-size 8

# 3. Evaluate the trained checkpoint against the TEST split (only place
#    the test set gets touched before this point).
python evaluate.py --checkpoint runs/small_<timestamp>/checkpoint_best.pth --model small

# 4. Export to ONNX and confirm exported accuracy matches the PyTorch
#    checkpoint within tolerance (default: 0.03 absolute).
python export.py --checkpoint runs/small_<timestamp>/checkpoint_best.pth --model small

# 5. Smoke test (needs a real checkpoint from step 2):
EVAL_SMOKE_CHECKPOINT=runs/small_<timestamp>/checkpoint_best.pth pytest tests/ -v
```

## ONNX export sanity check

RF-DETR's published docs (as of writing) don't pin down the exact ONNX
output box-coordinate convention (normalized cxcywh vs. absolute-pixel
xyxy in the resized input frame). `evaluate.py`'s `OnnxPredictor` detects
this at runtime from the value range and prints the raw output shapes and
ranges on first use -- **read that printed output the first time you run
`export.py` for real** and sanity-check a couple of the annotated failure
images in `eval_onnx/failures/` against `eval_pytorch/failures/` to confirm
boxes land in the right place, not just that the accuracy numbers matched.

Similarly, `evaluate.py` can't assume whether the trained model's
`class_id` output is a raw COCO category id or a 0-indexed position into
the sorted category list -- it self-calibrates this against a train-split
sample (never test) at the start of every run and prints which mapping it
picked. If it exits with a calibration failure, that's a signal the
checkpoint itself is undertrained, not a bug in the mapping logic.

## What's NOT here (by design)

Anomaly detection, OCR, label alignment, video pipeline, DDP/multi-GPU,
CLI framework, Docker, merging the original "Bottle Caps" dataset. See the
task spec for why.
