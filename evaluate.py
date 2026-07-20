"""Evaluates a trained RF-DETR checkpoint (or an ONNX export of one)
against the TEST split only, and reports:

  - overall mAP / AP50 / AP75 (COCO-style, via pycocotools)
  - detection-level accuracy + image-level top-1 accuracy
  - per-class precision / recall / F1 for all 5 classes
  - a confusion matrix (CSV), including a "background" row/col for
    false negatives (missed detections) and false positives (hallucinations)
  - the worst N failure cases, saved as annotated images

Import surface for export.py: `TorchPredictor`, `OnnxPredictor`, and
`run_evaluation(...)` are reused as-is to re-run this exact logic against
the ONNX export, per the PoC spec ("don't skip the ONNX check").

Usage:
    python evaluate.py --checkpoint runs/small_.../checkpoint_best.pth --model small
    python evaluate.py --onnx runs/small_.../inference_model.onnx --model small
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "data" / "bottle-defects"

MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Detections:
    xyxy: np.ndarray        # (N, 4) float, pixel coords in ORIGINAL image space
    confidence: np.ndarray  # (N,)
    class_id: np.ndarray    # (N,) int -- raw dataset category ids, after mapping


# --------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------

class TorchPredictor:
    """Wraps a trained RF-DETR PyTorch checkpoint."""

    def __init__(self, model_name: str, checkpoint_path: str):
        import rfdetr

        cls = getattr(rfdetr, MODEL_CLASSES[model_name])
        self.model = cls(pretrain_weights=str(checkpoint_path))

    def raw_predict(self, image_path: Path, threshold: float) -> Detections:
        det = self.model.predict(str(image_path), threshold=threshold)
        return Detections(
            xyxy=np.asarray(det.xyxy, dtype=float),
            confidence=np.asarray(det.confidence, dtype=float),
            class_id=np.asarray(det.class_id, dtype=int),
        )


class OnnxPredictor:
    """Wraps an ONNX export of a trained RF-DETR model.

    RF-DETR's documented ONNX output is `boxes, labels = session.run(...)`
    with `labels` described elsewhere in the docs as "class scores" (i.e.
    per-class logits/scores, not pre-argmaxed ids). The exact box coordinate
    convention (normalized cxcywh vs. absolute-pixel xyxy in the resized
    input frame) is NOT pinned down in the published docs at the time this
    was written, so this class defensively detects it from the output value
    range and prints diagnostics on first use -- verify those printed shapes
    against README.md "ONNX export sanity check" the first time this runs
    for real.
    """

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 560
        self.input_w = shape[3] if isinstance(shape[3], int) else 560
        self._diagnosed = False
        print(f"[onnx] input '{inp.name}' shape={shape}")
        for o in self.session.get_outputs():
            print(f"[onnx] output '{o.name}' shape={o.shape}")

    def raw_predict(self, image_path: Path, threshold: float) -> Detections:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        resized = img.resize((self.input_w, self.input_h))
        arr = np.asarray(resized).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: arr})
        boxes_raw = np.asarray(outputs[0])[0]   # (num_queries, 4)
        logits_raw = np.asarray(outputs[1])[0]  # (num_queries, num_classes)

        if not self._diagnosed:
            print(f"[onnx] boxes_raw range=({boxes_raw.min():.3f}, {boxes_raw.max():.3f}) shape={boxes_raw.shape}")
            print(f"[onnx] logits_raw range=({logits_raw.min():.3f}, {logits_raw.max():.3f}) shape={logits_raw.shape}")
            self._diagnosed = True

        scores = 1.0 / (1.0 + np.exp(-logits_raw))  # RF-DETR uses a sigmoid focal-loss head
        class_id = scores.argmax(axis=1)
        confidence = scores[np.arange(len(scores)), class_id]

        keep = confidence >= threshold
        boxes_kept = boxes_raw[keep]
        boxes_xyxy = _normalize_box_format(boxes_kept, self.input_w, self.input_h)

        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y

        return Detections(xyxy=boxes_xyxy, confidence=confidence[keep], class_id=class_id[keep])


def _normalize_box_format(boxes: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(float)
    if boxes.max() <= 1.5:
        # Normalized cxcywh -> pixel xyxy in the resized-input frame.
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) * input_w
        y1 = (cy - h / 2) * input_h
        x2 = (cx + w / 2) * input_w
        y2 = (cy + h / 2) * input_h
        out = np.stack([x1, y1, x2, y2], axis=1)
    else:
        # Already absolute-pixel xyxy in the resized-input frame.
        out = boxes.astype(float).copy()
    bad = (out[:, 2] <= out[:, 0]) | (out[:, 3] <= out[:, 1])
    if bad.any():
        raise RuntimeError(
            f"{bad.sum()} ONNX boxes decoded with x2<=x1 or y2<=y1 -- the "
            "box-format assumption in _normalize_box_format() is wrong for "
            "this rfdetr version. Print outputs[0][:5] raw and compare "
            "against https://rfdetr.roboflow.com/latest/learn/export/."
        )
    return out


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------

def load_split(dataset_dir: Path, split: str):
    ann_path = dataset_dir / split / "_annotations.coco.json"
    with ann_path.open() as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    categories = {c["id"]: c["name"] for c in coco["categories"] if c["name"] != "Defects"}
    return coco, images_by_id, anns_by_image, categories


def resolve_class_id_mapping(predictor, dataset_dir: Path, categories: dict[int, str],
                              sample_size: int = 40, threshold: float = 0.3):
    """Empirically determines whether predictor class_id values are raw COCO
    category ids or 0-indexed positions into sorted(categories), by checking
    agreement against ground truth on a TRAIN-split sample. Never touches test.
    """
    _, images_by_id, anns_by_image, _ = load_split(dataset_dir, "train")
    sorted_ids = sorted(categories.keys())
    train_dir = dataset_dir / "train"

    sample_ids = [iid for iid in anns_by_image if anns_by_image[iid][0]["category_id"] in categories][:sample_size]

    direct_hits = contiguous_hits = total = 0
    for image_id in sample_ids:
        anns = anns_by_image[image_id]
        image_path = train_dir / images_by_id[image_id]["file_name"]
        det = predictor.raw_predict(image_path, threshold=threshold)
        if len(det.class_id) == 0:
            continue
        top = int(det.class_id[int(np.argmax(det.confidence))])
        gt_class = anns[0]["category_id"]
        total += 1
        direct_hits += int(top == gt_class)
        contiguous_hits += int(0 <= top < len(sorted_ids) and sorted_ids[top] == gt_class)

    if total == 0:
        sys.exit("[evaluate] Calibration failed: predictor returned zero detections on the train sample.")

    direct_rate, contiguous_rate = direct_hits / total, contiguous_hits / total
    print(f"[evaluate] Class-id calibration on {total} train images: direct={direct_rate:.0%}, contiguous={contiguous_rate:.0%}")

    if max(direct_rate, contiguous_rate) < 0.5:
        sys.exit(
            "[evaluate] Neither class-id mapping reaches 50% agreement on the "
            "calibration sample -- this points to an undertrained checkpoint "
            "rather than a mapping bug. Inspect predictions manually before "
            "trusting metrics below."
        )
    if contiguous_rate > direct_rate:
        print("[evaluate] Using CONTIGUOUS class-id mapping.")
        return lambda cid: sorted_ids[cid]
    print("[evaluate] Using DIRECT class-id mapping.")
    return lambda cid: cid


def predict_mapped(predictor, id_map, image_path: Path, threshold: float) -> Detections:
    det = predictor.raw_predict(image_path, threshold)
    if len(det.class_id) == 0:
        return det
    mapped = np.array([id_map(int(c)) for c in det.class_id], dtype=int)
    return Detections(xyxy=det.xyxy, confidence=det.confidence, class_id=mapped)


# --------------------------------------------------------------------------
# Matching / metrics
# --------------------------------------------------------------------------

def iou_matrix(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_conf, iou_threshold):
    """Greedy IoU matching, class-agnostic (so cross-class confusions show up).
    Returns a list of (gt_class_or_None, pred_class_or_None) pairs.
    """
    n_pred = len(pred_boxes)
    order = np.argsort(-pred_conf) if n_pred else np.array([], dtype=int)
    matched_gt: set[int] = set()
    used_pred: set[int] = set()
    pairs = []

    for pi in order:
        if len(gt_boxes) == 0:
            break
        ious = iou_matrix(pred_boxes[pi], gt_boxes)
        best_gi, best_iou = None, 0.0
        for gi in np.argsort(-ious):
            if gi in matched_gt:
                continue
            best_gi, best_iou = int(gi), float(ious[gi])
            break
        if best_gi is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            used_pred.add(int(pi))
            pairs.append((gt_classes[best_gi], pred_classes[pi]))

    for gi in range(len(gt_boxes)):
        if gi not in matched_gt:
            pairs.append((gt_classes[gi], None))
    for pi in range(n_pred):
        if pi not in used_pred:
            pairs.append((None, pred_classes[pi]))
    return pairs


def build_confusion_matrix(all_pairs, class_names: list[str]):
    labels = class_names + ["background"]
    idx = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for gt, pred in all_pairs:
        gi = idx[gt] if gt is not None else idx["background"]
        pi = idx[pred] if pred is not None else idx["background"]
        matrix[gi, pi] += 1
    return matrix, labels


def per_class_prf1(matrix: np.ndarray, labels: list[str]):
    results = {}
    n = len(labels)
    for i, name in enumerate(labels[:-1]):  # exclude "background" as a class
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[i, :].sum())}
    return results


def compute_coco_map(dataset_dir: Path, predictions_coco: list[dict]):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_path = dataset_dir / "test" / "_annotations.coco.json"
    coco_gt = COCO(str(ann_path))
    if not predictions_coco:
        print("[evaluate] Warning: zero detections above threshold; mAP = 0 for all classes.")
        cat_ids = coco_gt.getCatIds()
        id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}, {name: 0.0 for name in id_to_name.values() if name != "Defects"}

    coco_dt = coco_gt.loadRes(predictions_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    overall = {"AP": float(coco_eval.stats[0]), "AP50": float(coco_eval.stats[1]), "AP75": float(coco_eval.stats[2])}

    id_to_name = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    per_class = {}
    for k, cid in enumerate(coco_eval.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        per_class[id_to_name[cid]] = float(p.mean()) if p.size else float("nan")
    return overall, per_class


# --------------------------------------------------------------------------
# Failure visualization
# --------------------------------------------------------------------------

def annotate_failure(image_path: Path, gt_anns, dets: Detections, id_to_name: dict, out_path: Path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for a in gt_anns:
        x, y, w, h = a["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(0, 200, 0), width=2)
        draw.text((x, max(0, y - 12)), f"GT: {id_to_name.get(a['category_id'], '?')}", fill=(0, 200, 0), font=font)

    for box, conf, cid in zip(dets.xyxy, dets.confidence, dets.class_id):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=2)
        label = f"{id_to_name.get(int(cid), '?')}:{conf:.2f}"
        draw.text((x1, min(img.height - 12, y2 + 2)), label, fill=(220, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


# --------------------------------------------------------------------------
# Main evaluation entry point (also called from export.py)
# --------------------------------------------------------------------------

def run_evaluation(predictor, dataset_dir: Path, output_dir: Path, run_label: str,
                    conf_threshold: float = 0.5, iou_threshold: float = 0.5,
                    num_failures: int = 12) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    coco, images_by_id, anns_by_image, categories = load_split(dataset_dir, "test")
    class_names = [categories[cid] for cid in sorted(categories)]
    id_map = resolve_class_id_mapping(predictor, dataset_dir, categories)

    all_pairs = []
    coco_predictions = []
    image_level_correct = 0
    image_level_total = 0
    per_image_results = []

    test_dir = dataset_dir / "test"
    for image_id, image_info in images_by_id.items():
        image_path = test_dir / image_info["file_name"]
        gt_anns = [a for a in anns_by_image.get(image_id, []) if a["category_id"] in categories]
        gt_boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in gt_anns]) if gt_anns else np.zeros((0, 4))
        gt_classes = [categories[a["category_id"]] for a in gt_anns]

        det = predict_mapped(predictor, id_map, image_path, conf_threshold)
        pred_classes = [categories.get(int(c), f"unknown_{c}") for c in det.class_id]

        pairs = match_image(gt_boxes, gt_classes, det.xyxy, pred_classes, det.confidence, iou_threshold)
        all_pairs.extend(pairs)

        for box, conf, cid in zip(det.xyxy, det.confidence, det.class_id):
            coco_predictions.append({
                "image_id": image_id,
                "category_id": int(cid),
                "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                "score": float(conf),
            })

        is_wrong = any((g != p) for g, p in pairs)
        if gt_anns:
            image_level_total += 1
            top_pred = pred_classes[int(np.argmax(det.confidence))] if len(pred_classes) else None
            correct = top_pred == gt_classes[0]
            image_level_correct += int(correct)
        else:
            correct = not len(pred_classes)

        per_image_results.append({
            "image_id": image_id, "file_name": image_info["file_name"], "is_wrong": is_wrong,
            "gt_anns": gt_anns, "det": det, "max_conf": float(det.confidence.max()) if len(det.confidence) else 0.0,
        })

    confusion, labels = build_confusion_matrix(all_pairs, class_names)
    prf1 = per_class_prf1(confusion, labels)
    overall_map, per_class_ap = compute_coco_map(dataset_dir, coco_predictions)

    total_matches = sum(1 for g, p in all_pairs if g is not None and p is not None)
    correct_matches = sum(1 for g, p in all_pairs if g is not None and p is not None and g == p)
    detection_level_accuracy = correct_matches / total_matches if total_matches else 0.0
    image_level_accuracy = image_level_correct / image_level_total if image_level_total else 0.0

    # --- write outputs ---
    with (output_dir / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt \\ pred"] + labels)
        for i, name in enumerate(labels):
            writer.writerow([name] + confusion[i].tolist())

    with (output_dir / "per_class_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support", "AP50_95"])
        for name in class_names:
            m = prf1[name]
            writer.writerow([name, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", m["support"], f"{per_class_ap.get(name, float('nan')):.4f}"])

    summary = {
        "run_label": run_label,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "num_test_images": len(images_by_id),
        "overall_map": overall_map,
        "detection_level_accuracy": detection_level_accuracy,
        "image_level_top1_accuracy": image_level_accuracy,
        "per_class": {name: {**prf1[name], "AP50_95": per_class_ap.get(name, float("nan"))} for name in class_names},
    }
    (output_dir / "metrics_report.json").write_text(json.dumps(summary, indent=2))

    # --- failure cases ---
    wrong = [r for r in per_image_results if r["is_wrong"]]
    wrong.sort(key=lambda r: -r["max_conf"])
    id_to_name = {cid: name for cid, name in categories.items()}
    for r in wrong[:num_failures]:
        image_path = test_dir / r["file_name"]
        out_path = output_dir / "failures" / r["file_name"]
        annotate_failure(image_path, r["gt_anns"], r["det"], id_to_name, out_path)

    print(f"\n[evaluate:{run_label}] test images={len(images_by_id)}  "
          f"mAP={overall_map['AP']:.4f}  AP50={overall_map['AP50']:.4f}  "
          f"detection-acc={detection_level_accuracy:.4f}  image-top1-acc={image_level_accuracy:.4f}")
    print(f"[evaluate:{run_label}] {len(wrong)}/{len(per_image_results)} images had at least one error; "
          f"saved {min(num_failures, len(wrong))} worst cases to {output_dir / 'failures'}")
    print(f"[evaluate:{run_label}] Reports written to {output_dir}")

    return summary


def build_predictor(args) -> tuple:
    if args.checkpoint and args.onnx:
        sys.exit("Pass exactly one of --checkpoint or --onnx, not both.")
    if args.checkpoint:
        return TorchPredictor(args.model, args.checkpoint), f"pytorch_{args.model}"
    if args.onnx:
        return OnnxPredictor(args.onnx), f"onnx_{args.model}"
    sys.exit("Pass --checkpoint <path/to/checkpoint_best.pth> or --onnx <path/to/inference_model.onnx>")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--onnx", default=None)
    p.add_argument("--model", choices=sorted(MODEL_CLASSES), default="small")
    p.add_argument("--dataset-dir", default=str(DATASET_DIR))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--failures", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictor, run_label = build_predictor(args)
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "eval_results" / run_label
    run_evaluation(
        predictor, Path(args.dataset_dir), output_dir, run_label,
        conf_threshold=args.conf_threshold, iou_threshold=args.iou_threshold, num_failures=args.failures,
    )


if __name__ == "__main__":
    main()
