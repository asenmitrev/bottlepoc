# PoC Report: Bottle Cap Condition Classification

**Status: dataset section complete; training/eval/export sections are
PENDING a run on the actual CUDA machine.** This repo was built on a
Mac with no GPU and almost no free disk, so `train.py`, `evaluate.py`, and
`export.py` are written and syntax-checked but not yet run for real. This
document is structured so that running the four commands in `README.md` ->
"Reproduce" fills in every `[PENDING]` below without changing the
structure. See `README.md` -> "Environment note" for why.

## 1. Dataset

Substituted the already-present **"Bottle Defects" v28** dataset for the
originally-specced "Bottle Caps" export -- see `README.md` -> "Dataset
provenance" and `NOTICE.md` for the full reasoning and citation. Same
5-class taxonomy, CC BY 4.0.

| Split | Images | Annotations |
|---|---|---|
| train | 398 | 423 |
| valid | 113 | 119 |
| test  | 112 | 124 |
| **total** | **623** | **666** |

Per-class instance counts (produced by `python data/download.py`):

| Class | train | valid | test | total | % of modal class |
|---|---|---|---|---|---|
| Good Cap | 152 | 43 | 23 | 218 | 100% |
| Loose Cap | 89 | 23 | 46 | 158 | 72% |
| No Cap | 112 | 32 | 17 | 161 | 74% |
| Broken Ring | 35 | 8 | 23 | 66 | 30% |
| Broken Cap | 35 | 13 | 15 | 63 | 29% |

**Class imbalance**: flagged. Broken Cap and Broken Ring both sit under
30% of the modal class (Good Cap). If those two classes underperform in
section 3 below, imbalance is the first suspect, not "the model isn't good
enough" -- consistent with the task's instruction to check this before
accepting a low number at face value.

**Split composition is not stratified.** The Roboflow-provided split (used
as-is, not re-split, per spec) shifts class proportions between train and
test noticeably:

| Class | % of train | % of test |
|---|---|---|
| Loose Cap | 22.4% | 41.1% |
| Broken Ring | 8.8% | 20.5% |
| No Cap | 28.2% | 15.2% |
| Good Cap | 38.2% | 20.5% |
| Broken Cap | 8.8% | 13.4% |

This matters for reading section 3: test-set precision/recall for Broken
Cap (n=15) and Broken Ring (n=23) will be noisy just from small support,
independent of model quality -- a couple of misclassifications swing those
numbers by several points. Good Cap and No Cap, despite being the modal
classes in train, are *under*-represented in test relative to train, so
strong test performance on them is a lower bar than train proportions would
suggest, and weak Loose Cap performance in test carries more weight since
it's 41% of the test set.

Images are small (256x144, grayscale/CRT-preprocessed per the dataset's
own export metadata) -- worth keeping in mind if edge-deployment images end
up being color and/or higher resolution than this training distribution.

## 2. Training configuration

**[PENDING]** -- filled in automatically from `runs/<model>_<timestamp>/training_config.json`
after `python train.py` runs. Planned defaults (see `README.md`):

- Model: RF-DETR-Small (Apache 2.0), pretrained COCO checkpoint, fine-tuned
- Epochs: 60, batch_size: 8, grad_accum_steps: 2, lr: 1e-4
- Early stopping on val metric, patience 15
- `compute_test_loss=False` -- test split is not read during training at all
- Seed: 42

| Metric | Value |
|---|---|
| Final epoch | [PENDING] |
| Best val mAP (AP50:95) | [PENDING] |
| Best val AP50 | [PENDING] |
| Training wall-clock (1x RTX 3090) | [PENDING] |

## 3. Test-set evaluation (PyTorch checkpoint)

**[PENDING]** -- filled in from `eval_results/pytorch_small/metrics_report.json`
after `python evaluate.py` runs.

### Overall

| Metric | Value | Published baseline (bottle-caps-p3mai, different dataset export) |
|---|---|---|
| COCO mAP (AP50:95) | [PENDING] | -- |
| COCO AP50 | [PENDING] | 95.6%* |
| Detection-level accuracy (matched TP / all matches) | [PENDING] | -- |
| Image-level top-1 accuracy | [PENDING] | -- |

\* The 95.6% figure is a rough sanity-check target from the *original*
`bottle-caps-p3mai` dataset (2,069 images), not this one (623 images,
different Roboflow project). Treat any gap as informative, not damning --
see the verdict in section 6 for how to interpret it given the dataset
substitution.

### Per-class precision / recall / F1

**[PENDING]** -- from `eval_results/pytorch_small/per_class_metrics.csv`.
Broken Cap and Broken Ring are the classes to scrutinize first given the
imbalance flagged in section 1.

| Class | Precision | Recall | F1 | Support (test) | AP50:95 |
|---|---|---|---|---|---|
| Broken Cap | [PENDING] | [PENDING] | [PENDING] | 15 | [PENDING] |
| Broken Ring | [PENDING] | [PENDING] | [PENDING] | 23 | [PENDING] |
| Good Cap | [PENDING] | [PENDING] | [PENDING] | 23 | [PENDING] |
| Loose Cap | [PENDING] | [PENDING] | [PENDING] | 46 | [PENDING] |
| No Cap | [PENDING] | [PENDING] | [PENDING] | 17 | [PENDING] |

### Confusion matrix

**[PENDING]** -- see `eval_results/pytorch_small/confusion_matrix.csv`
(includes a "background" row/col for missed detections and hallucinated
false positives).

### Worst failure cases

**[PENDING]** -- 12 worst cases saved as annotated images (green = ground
truth, red = prediction) in `eval_results/pytorch_small/failures/`.
Look here first for whether errors cluster on a specific class (imbalance),
a specific failure mode (occlusion, lighting, ambiguous "Loose" vs "Good"
cap boundary), or are scattered (likely just needs more data).

## 4. ONNX export accuracy check

**[PENDING]** -- from `export_accuracy_check.json` after `python export.py`
runs (default tolerance: 0.03 absolute on mAP, AP50, both accuracy
definitions, and per-class F1).

| Metric | PyTorch | ONNX | Delta | Within tolerance? |
|---|---|---|---|---|
| mAP (AP50:95) | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| AP50 | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Detection-level accuracy | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Image-level top-1 accuracy | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

If this fails, treat the ONNX export as **not validated** -- do not deploy
it -- and check `README.md` -> "ONNX export sanity check" for the specific
ambiguity (box coordinate convention) most likely to cause a silent
mismatch here, since that part of `evaluate.py`'s `OnnxPredictor` couldn't
be empirically verified without GPU access when this was written.

## 5. RF-DETR-Nano comparison

**[PENDING, optional]** -- if you also run `--model nano` through the same
pipeline, drop its section-3/4 numbers in here for a same-dataset Nano vs.
Small comparison, since the task spec treats both as valid edge candidates.

## 6. Verdict

**[PENDING]** -- write this last, after sections 2-4 are filled in. Answer,
in plain language:

1. Does test-set mAP/accuracy land in a defensible range given (a) the
   dataset substitution (worth reading against the 95.6% figure with a
   large grain of salt -- different dataset, less than a third the image
   count) and (b) the class imbalance and non-stratified test split
   documented in section 1?
2. Is Broken Cap / Broken Ring specifically weak, and if so, is that
   explained by their low support (63/66 instances total) rather than a
   fundamental model limitation?
3. Given the failure cases in section 3 -- is this "good enough to
   proceed" to a real inspection pipeline, or "needs more labeled data /
   a different approach," and specifically which of those two?
