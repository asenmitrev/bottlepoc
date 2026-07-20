"""Exports a trained RF-DETR checkpoint to ONNX, then re-runs evaluate.py's
exact evaluation logic against BOTH the PyTorch checkpoint and the ONNX
export on the test split, and diffs the summary metrics.

This is the "does distillation to an edge format hold up" check -- the
whole point of calling this an edge-distillate validation rather than just
a training run. It is intentionally not skippable: the script exits
non-zero if accuracy drift exceeds --tolerance.

Usage:
    python export.py --checkpoint runs/small_.../checkpoint_best.pth --model small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluate import DATASET_DIR, MODEL_CLASSES, OnnxPredictor, TorchPredictor, run_evaluation

REPO_ROOT = Path(__file__).resolve().parent


def export_to_onnx(model_name: str, checkpoint_path: Path, output_dir: Path) -> Path:
    import rfdetr

    cls = getattr(rfdetr, MODEL_CLASSES[model_name])
    model = cls(pretrain_weights=str(checkpoint_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    model.export(output_dir=str(output_dir), format="onnx")

    onnx_path = output_dir / "inference_model.onnx"
    if not onnx_path.exists():
        candidates = sorted(output_dir.glob("*.onnx"))
        if not candidates:
            sys.exit(f"[export] model.export() did not produce an .onnx file in {output_dir}")
        onnx_path = candidates[0]
    print(f"[export] ONNX model written to: {onnx_path}")
    return onnx_path


def diff_summaries(pytorch_summary: dict, onnx_summary: dict, tolerance: float) -> bool:
    checks = [
        ("overall mAP", pytorch_summary["overall_map"]["AP"], onnx_summary["overall_map"]["AP"]),
        ("overall AP50", pytorch_summary["overall_map"]["AP50"], onnx_summary["overall_map"]["AP50"]),
        ("detection-level accuracy", pytorch_summary["detection_level_accuracy"], onnx_summary["detection_level_accuracy"]),
        ("image-level top-1 accuracy", pytorch_summary["image_level_top1_accuracy"], onnx_summary["image_level_top1_accuracy"]),
    ]
    print(f"\n[export] PyTorch vs ONNX (tolerance={tolerance}):")
    all_ok = True
    for name, pt_val, onnx_val in checks:
        delta = abs(pt_val - onnx_val)
        ok = delta <= tolerance
        all_ok &= ok
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] {name}: pytorch={pt_val:.4f}  onnx={onnx_val:.4f}  delta={delta:.4f}")

    for name in pytorch_summary["per_class"]:
        pt_f1 = pytorch_summary["per_class"][name]["f1"]
        onnx_f1 = onnx_summary["per_class"].get(name, {}).get("f1", 0.0)
        delta = abs(pt_f1 - onnx_f1)
        ok = delta <= tolerance
        all_ok &= ok
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] {name} F1: pytorch={pt_f1:.4f}  onnx={onnx_f1:.4f}  delta={delta:.4f}")

    return all_ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pth from train.py")
    p.add_argument("--model", choices=sorted(MODEL_CLASSES), default="small")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument("--output-dir", default=None, help="Defaults to the checkpoint's parent dir")
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--tolerance", type=float, default=0.03, help="Max allowed absolute metric drift, e.g. 0.03 = 3 points")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"No checkpoint at {checkpoint_path}")

    run_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    dataset_dir = Path(args.dataset_dir)

    onnx_path = export_to_onnx(args.model, checkpoint_path, run_dir)

    print("\n[export] Re-evaluating PyTorch checkpoint on the test split...")
    torch_predictor = TorchPredictor(args.model, str(checkpoint_path))
    pytorch_summary = run_evaluation(
        torch_predictor, dataset_dir, run_dir / "eval_pytorch", f"pytorch_{args.model}",
        conf_threshold=args.conf_threshold, iou_threshold=args.iou_threshold,
    )

    print("\n[export] Evaluating ONNX export on the test split...")
    onnx_predictor = OnnxPredictor(str(onnx_path))
    onnx_summary = run_evaluation(
        onnx_predictor, dataset_dir, run_dir / "eval_onnx", f"onnx_{args.model}",
        conf_threshold=args.conf_threshold, iou_threshold=args.iou_threshold,
    )

    all_ok = diff_summaries(pytorch_summary, onnx_summary, args.tolerance)

    result = {
        "onnx_path": str(onnx_path),
        "tolerance": args.tolerance,
        "passed": all_ok,
        "pytorch_summary": pytorch_summary,
        "onnx_summary": onnx_summary,
    }
    result_path = run_dir / "export_accuracy_check.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\n[export] Full comparison written to {result_path}")

    if not all_ok:
        sys.exit(
            "[export] ONNX export accuracy drifted beyond tolerance. Do not "
            "treat this export as validated -- see export_accuracy_check.json."
        )
    print("[export] ONNX export accuracy matches the PyTorch checkpoint within tolerance.")


if __name__ == "__main__":
    main()
