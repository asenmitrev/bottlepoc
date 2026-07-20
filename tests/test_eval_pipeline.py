"""Smoke test: evaluate.py's pipeline runs on a couple of test-split images
without crashing and returns sane output shapes.

Requires a trained checkpoint. Set EVAL_SMOKE_CHECKPOINT to a
checkpoint_best.pth path (e.g. from `python train.py`) before running:

    EVAL_SMOKE_CHECKPOINT=runs/small_.../checkpoint_best.pth pytest tests/

Skips automatically if no checkpoint is configured or rfdetr isn't
installed (e.g. on a machine without a GPU/training deps) -- this is meant
to be exercised on the training machine, not this repo's dev environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_ENV = "EVAL_SMOKE_CHECKPOINT"
DATASET_DIR = REPO_ROOT / "data" / "bottle-defects"

pytestmark = pytest.mark.skipif(
    not os.environ.get(CHECKPOINT_ENV),
    reason=f"Set {CHECKPOINT_ENV}=path/to/checkpoint_best.pth to run this against a real model",
)


@pytest.fixture(scope="module")
def predictor():
    from evaluate import TorchPredictor

    checkpoint = os.environ[CHECKPOINT_ENV]
    model_name = os.environ.get("EVAL_SMOKE_MODEL", "small")
    return TorchPredictor(model_name, checkpoint)


def test_dataset_present():
    assert (DATASET_DIR / "test" / "_annotations.coco.json").exists(), (
        "No test split found -- run `python data/download.py` first."
    )


def test_predict_returns_sane_shapes(predictor):
    from evaluate import load_split

    _, images_by_id, _, _ = load_split(DATASET_DIR, "test")
    sample_images = list(images_by_id.values())[:3]
    assert sample_images, "Test split has no images"

    for image_info in sample_images:
        image_path = DATASET_DIR / "test" / image_info["file_name"]
        det = predictor.raw_predict(image_path, threshold=0.5)

        assert det.xyxy.ndim == 2 and det.xyxy.shape[-1] == 4
        assert det.confidence.ndim == 1
        assert det.class_id.ndim == 1
        n = det.xyxy.shape[0]
        assert det.confidence.shape[0] == n
        assert det.class_id.shape[0] == n
        if n:
            assert (det.confidence >= 0).all() and (det.confidence <= 1).all()
            assert (det.xyxy[:, 2] > det.xyxy[:, 0]).all()
            assert (det.xyxy[:, 3] > det.xyxy[:, 1]).all()


def test_run_evaluation_end_to_end(tmp_path, predictor):
    from evaluate import run_evaluation

    summary = run_evaluation(
        predictor, DATASET_DIR, tmp_path / "smoke_eval", "smoke_test",
        conf_threshold=0.5, iou_threshold=0.5, num_failures=2,
    )

    assert "overall_map" in summary
    assert 0.0 <= summary["detection_level_accuracy"] <= 1.0
    assert 0.0 <= summary["image_level_top1_accuracy"] <= 1.0
    assert set(summary["per_class"]) == {"Broken Cap", "Broken Ring", "Good Cap", "Loose Cap", "No Cap"}
    assert (tmp_path / "smoke_eval" / "metrics_report.json").exists()
    assert (tmp_path / "smoke_eval" / "confusion_matrix.csv").exists()
    assert (tmp_path / "smoke_eval" / "per_class_metrics.csv").exists()
