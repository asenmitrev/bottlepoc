"""Fine-tunes RF-DETR (Nano or Small) from its pretrained COCO checkpoint on
the 5 bottle-cap-condition classes.

Only train/valid are touched here. compute_test_loss is explicitly disabled
so the test split is never read during training -- evaluate.py is the only
script allowed to touch it, and only once, at the end.

Usage:
    python train.py --model small --epochs 60
    python train.py --model nano --epochs 80 --batch-size 8

Requires a CUDA GPU in practice (RF-DETR on 570-ish images trains in
minutes on a 3090; CPU is not a realistic option here). See README.md for
the exact command used to produce the numbers in report.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "data" / "bottle-defects"

MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
}


def build_model(model_name: str):
    import rfdetr

    cls = getattr(rfdetr, MODEL_CLASSES[model_name])
    return cls()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODEL_CLASSES), default="small",
                    help="RF-DETR variant. Small is the default: a bit more headroom than "
                         "Nano while still Apache-2.0 and edge-appropriate; pass --model nano "
                         "to compare against the smaller variant.")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resolution", type=int, default=None, help="Defaults to the model's native resolution if unset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--output-dir", default=None, help="Defaults to runs/<model>_<timestamp>")
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        sys.exit(f"No dataset at {dataset_dir}. Run `python data/download.py` first.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "runs" / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model)

    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        seed=args.seed,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
        log_per_class_metrics=True,
        tensorboard=True,
        # Test split must stay untouched until evaluate.py runs. This is
        # the one train() flag that reads the test split by default.
        compute_test_loss=False,
    )
    if args.resolution is not None:
        train_kwargs["resolution"] = args.resolution
    if args.resume is not None:
        train_kwargs["resume"] = args.resume

    config_record = {
        "model": args.model,
        "timestamp_utc": timestamp,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        **{k: v for k, v in train_kwargs.items() if k not in ("dataset_dir", "output_dir")},
    }
    (output_dir / "training_config.json").write_text(json.dumps(config_record, indent=2))
    print(f"[train] Config: {json.dumps(config_record, indent=2)}")

    model.train(**train_kwargs)

    # RF-DETR writes checkpoint_best_ema.pth (if use_ema, the default) and
    # checkpoint_best_regular.pth. Prefer the EMA weights; fall back to
    # regular if EMA wasn't produced for some reason.
    candidates = ["checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint_best_total.pth"]
    best = next((output_dir / c for c in candidates if (output_dir / c).exists()), None)
    if best is None:
        sys.exit(f"[train] No best checkpoint found in {output_dir}; check training logs.")

    stable_path = output_dir / "checkpoint_best.pth"
    if best != stable_path:
        shutil.copy(best, stable_path)
    print(f"[train] Best checkpoint: {best} -> {stable_path}")

    metrics_csv = output_dir / "metrics.csv"
    if metrics_csv.exists():
        lines = metrics_csv.read_text().splitlines()
        print(f"[train] metrics.csv: {len(lines) - 1} logged rows. Last row:")
        print(f"    {lines[-1]}")
    else:
        print("[train] Warning: no metrics.csv found in output_dir; check rfdetr version's logger output.")

    print(f"\n[train] Done. Checkpoint for evaluate.py/export.py: {stable_path}")


if __name__ == "__main__":
    main()
