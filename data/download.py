"""Materializes the training dataset into data/bottle-defects/ and reports
per-split class counts.

Primary path: copy the already-labeled local COCO export (see README.md ->
"Dataset provenance") into this repo so it travels with it (e.g. when
rsync'd to the training machine).

Fallback path: if the local source isn't found and ROBOFLOW_API_KEY is set,
pull fresh via the Roboflow API in COCO format. This is untested (the local
source was available when this script was written) -- treat it as a
documented escape hatch, not the verified path.

Deliberately stdlib-only (no pandas/dotenv/etc.) so it runs anywhere,
including on machines where the heavier training deps aren't installed yet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT.parent / "Bottle Defects.v28i.coco"
DEST_DIR = REPO_ROOT / "data" / "bottle-defects"
SPLITS = ("train", "valid", "test")
ANN_FILENAME = "_annotations.coco.json"


def load_dotenv_manually(env_path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser so we don't need python-dotenv for this script."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def find_source_dataset(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if is_valid_coco_dataset(p) else None
    if is_valid_coco_dataset(DEFAULT_SOURCE):
        return DEFAULT_SOURCE
    return None


def is_valid_coco_dataset(path: Path) -> bool:
    if not path.is_dir():
        return False
    return all((path / split / ANN_FILENAME).is_file() for split in SPLITS)


def copy_dataset(source: Path, dest: Path) -> None:
    for split in SPLITS:
        src_split = source / split
        dst_split = dest / split
        if dst_split.exists():
            shutil.rmtree(dst_split)
        shutil.copytree(src_split, dst_split)


def fallback_roboflow_download(dest: Path, env: dict[str, str]) -> Path:
    api_key = env.get("ROBOFLOW_API_KEY") or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise SystemExit(
            "No local dataset found and no ROBOFLOW_API_KEY set. "
            "Either provide the local 'Bottle Defects.v28i.coco' export "
            "(see README.md -> Dataset provenance) or set ROBOFLOW_API_KEY "
            "in .env to pull fresh from Roboflow."
        )
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ROBOFLOW_API_KEY is set but the 'roboflow' package isn't "
            "installed. Run `uv sync` first."
        ) from exc

    print("[download] No local dataset found; falling back to Roboflow API pull.")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("product-defect-detection").project("bottle-defects-o4gnx")
    version = project.version(28)
    downloaded = version.download("coco", location=str(dest.parent / "bottle-defects-rf-download"))
    return Path(downloaded.location)


def class_distribution(ann_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    with ann_path.open() as f:
        coco = json.load(f)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    image_count = len(coco["images"])
    ann_counts = Counter(a["category_id"] for a in coco["annotations"])
    by_name = {cat_names[cid]: ann_counts.get(cid, 0) for cid in sorted(cat_names) if cat_names[cid] != "Defects"}
    return by_name, {"images": image_count, "annotations": len(coco["annotations"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Override source dataset directory")
    args = parser.parse_args()

    env = load_dotenv_manually(REPO_ROOT / ".env")
    source_override = args.source or env.get("BOTTLE_DATASET_SOURCE_DIR") or os.environ.get("BOTTLE_DATASET_SOURCE_DIR")
    source_override = source_override or None

    source = find_source_dataset(source_override)
    if source is not None:
        print(f"[download] Using local dataset at: {source}")
        copy_dataset(source, DEST_DIR)
    else:
        source = fallback_roboflow_download(DEST_DIR, env)
        copy_dataset(source, DEST_DIR)

    print(f"[download] Dataset materialized at: {DEST_DIR}\n")

    report: dict[str, dict] = {}
    header = f"{'class':<14}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'total':>10}"
    print(header)
    print("-" * len(header))

    class_names: list[str] = []
    totals_by_split: dict[str, dict] = {}
    per_class_per_split: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        ann_path = DEST_DIR / split / ANN_FILENAME
        by_name, totals = class_distribution(ann_path)
        totals_by_split[split] = totals
        for name, count in by_name.items():
            per_class_per_split.setdefault(name, {})[split] = count
            if name not in class_names:
                class_names.append(name)

    for name in class_names:
        row = per_class_per_split[name]
        total = sum(row.get(s, 0) for s in SPLITS)
        print(f"{name:<14}" + "".join(f"{row.get(s, 0):>10}" for s in SPLITS) + f"{total:>10}")

    print("-" * len(header))
    img_row = "".join(f"{totals_by_split[s]['images']:>10}" for s in SPLITS)
    print(f"{'[images]':<14}{img_row}{sum(t['images'] for t in totals_by_split.values()):>10}")

    # Flag imbalance: any class whose total count is less than half the
    # modal class's count, expressed as a fraction of the largest class.
    grand_totals = {name: sum(row.values()) for name, row in per_class_per_split.items()}
    modal = max(grand_totals.values())
    imbalanced = {n: c for n, c in grand_totals.items() if c < 0.5 * modal}
    if imbalanced:
        print("\n[download] Class imbalance flagged (< 50% of the modal class count):")
        for n, c in imbalanced.items():
            print(f"    {n}: {c} ({c / modal:.0%} of modal class, {grand_totals[max(grand_totals, key=grand_totals.get)]})")
    else:
        print("\n[download] No class is below 50% of the modal class count.")

    report["class_counts"] = per_class_per_split
    report["split_totals"] = totals_by_split
    report["imbalanced_classes"] = imbalanced
    report_path = DEST_DIR.parent / "class_distribution.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[download] Full report written to: {report_path}")


if __name__ == "__main__":
    main()
