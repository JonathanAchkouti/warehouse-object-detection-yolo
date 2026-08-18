"""Create a local Ultralytics view of LOCO without touching the source dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from dataset import DEVELOPMENT_FILES, TEST_FILES, VALIDATION_EVERY

ANNOTATIONS_DIRNAME = "rgb"
LOCO_ARCHIVE_ROOT = "/dataset"
SUBSET_PATTERN = re.compile(r"loco-sub(?P<subset>\d+)-")
FORMAT_VERSION = 1


def validate_output_dir(output_dir: Path, loco_root: Path) -> None:
    """Reject broad or source-overlapping targets before any prepared view is replaced."""
    resolved_output = output_dir.resolve()
    resolved_source = loco_root.resolve()
    if resolved_output in {Path(resolved_output.anchor), resolved_source}:
        raise ValueError(f"Unsafe YOLO output directory: {resolved_output}")
    if resolved_source in resolved_output.parents or resolved_output in resolved_source.parents:
        raise ValueError("YOLO output and source LOCO directories must not contain each other")


def resolve_loco_root(raw_dir: Path, filenames: tuple[str, ...]) -> Path:
    """Return the directory containing LOCO annotations and subset image folders."""
    for candidate in (raw_dir, raw_dir.parent):
        if all((candidate / ANNOTATIONS_DIRNAME / name).is_file() for name in filenames):
            return candidate
    expected = raw_dir / ANNOTATIONS_DIRNAME / filenames[0]
    raise FileNotFoundError(f"LOCO annotations not found; expected files such as {expected}")


def subset_number(filename: str) -> int:
    """Extract the numeric LOCO subset identifier from an annotation filename."""
    match = SUBSET_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Cannot determine LOCO subset from {filename}")
    return int(match.group("subset"))


def yolo_box(annotation: dict[str, Any], width: int, height: int) -> tuple[float, ...] | None:
    """Convert one valid COCO box to clipped normalized YOLO coordinates."""
    if annotation.get("iscrowd", 0):
        return None
    x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
    if box_width <= 0 or box_height <= 0 or width <= 0 or height <= 0:
        return None

    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + box_width, 0.0), float(width))
    y2 = min(max(y + box_height, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def reusable_view(
    output_dir: Path,
    loco_root: Path,
    *,
    include_test: bool,
    copy_images: bool,
) -> tuple[Path, dict[str, Any]] | None:
    """Return an intact compatible prepared view, avoiding repeated Drive copies."""
    dataset_yaml = output_dir / "loco.yaml"
    manifest_path = output_dir / "split_manifest.json"
    if not dataset_yaml.is_file() or not manifest_path.is_file():
        return None
    try:
        summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    expected_storage = "copy" if copy_images else "symlink"
    if (
        summary.get("format_version") != FORMAT_VERSION
        or summary.get("raw_dir") != str(loco_root)
        or summary.get("include_test") is not include_test
        or summary.get("image_storage") != expected_storage
    ):
        return None
    for split, expected_count in summary.get("splits", {}).items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        if (
            not image_dir.is_dir()
            or not label_dir.is_dir()
            or sum(path.is_file() for path in image_dir.iterdir()) != expected_count
            or sum(path.is_file() for path in label_dir.iterdir()) != expected_count
        ):
            return None
    return dataset_yaml, summary


def prepare_yolo_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    include_test: bool = False,
    copy_images: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Build deterministic train/validation views and optionally the locked test view.

    Development uses exactly the same per-subset one-in-five validation rule as
    :class:`dataset.LocoDataset`. Images can be copied to fast local Colab storage or symlinked,
    while tiny YOLO label files are generated locally. By default subsets 1 and 4 are never read
    or exposed to Ultralytics.
    """
    filenames = DEVELOPMENT_FILES + (TEST_FILES if include_test else ())
    loco_root = resolve_loco_root(raw_dir, filenames)
    validate_output_dir(output_dir, loco_root)
    reusable = reusable_view(
        output_dir,
        loco_root,
        include_test=include_test,
        copy_images=copy_images,
    )
    if reusable is not None:
        return reusable
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    split_counts: defaultdict[str, int] = defaultdict(int)
    subset_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    object_counts: defaultdict[str, int] = defaultdict(int)
    categories: list[dict[str, Any]] | None = None

    for filename in filenames:
        subset = subset_number(filename)
        annotation_path = loco_root / ANNOTATIONS_DIRNAME / filename
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        current_categories = sorted(payload["categories"], key=lambda item: item["id"])
        if categories is None:
            categories = current_categories
        elif categories != current_categories:
            raise ValueError(f"Category definitions differ in {filename}")
        category_indices = {
            category["id"]: index for index, category in enumerate(current_categories)
        }
        annotations_by_image: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)

        is_test_subset = filename in TEST_FILES
        for image_index, image in enumerate(payload["images"]):
            if is_test_subset:
                split = "test"
            else:
                split = "validation" if image_index % VALIDATION_EVERY == 0 else "train"

            source = loco_root / Path(image["path"]).relative_to(LOCO_ARCHIVE_ROOT)
            if not source.is_file():
                raise FileNotFoundError(f"LOCO image referenced by {filename} is missing: {source}")
            suffix = source.suffix.lower() or ".jpg"
            stem = f"sub{subset}_{image_index:06d}"
            image_destination = output_dir / "images" / split / f"{stem}{suffix}"
            label_destination = output_dir / "labels" / split / f"{stem}.txt"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            if copy_images:
                shutil.copy2(source, image_destination)
            else:
                image_destination.symlink_to(source.resolve())

            label_lines: list[str] = []
            for annotation in annotations_by_image.get(image["id"], []):
                box = yolo_box(annotation, int(image["width"]), int(image["height"]))
                if box is None:
                    continue
                class_index = category_indices[annotation["category_id"]]
                label_lines.append(
                    f"{class_index} " + " ".join(f"{coordinate:.8f}" for coordinate in box)
                )
                object_counts[current_categories[class_index]["name"]] += 1
            label_destination.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            split_counts[split] += 1
            subset_counts[str(subset)][split] += 1

    if categories is None:
        raise ValueError("No LOCO categories were found")

    dataset_config: dict[str, Any] = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/validation",
        "names": {index: category["name"] for index, category in enumerate(categories)},
    }
    if include_test:
        dataset_config["test"] = "images/test"
    dataset_yaml = output_dir / "loco.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(dataset_config, sort_keys=False),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "raw_dir": str(loco_root),
        "output_dir": str(output_dir),
        "include_test": include_test,
        "image_storage": "copy" if copy_images else "symlink",
        "splits": dict(sorted(split_counts.items())),
        "subsets": {
            subset: dict(sorted(counts.items())) for subset, counts in sorted(subset_counts.items())
        },
        "classes": [category["name"] for category in categories],
        "objects": dict(sorted(object_counts.items())),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_yaml, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="expose subsets 1 and 4 only after the final model has been locked",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy permitted images to fast local storage instead of creating symlinks",
    )
    args = parser.parse_args()
    dataset_yaml, summary = prepare_yolo_dataset(
        args.raw_dir,
        args.output_dir,
        include_test=args.include_test,
        copy_images=args.copy_images,
    )
    print(f"YOLO dataset: {dataset_yaml}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
