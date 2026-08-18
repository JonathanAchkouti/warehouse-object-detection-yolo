import json
from pathlib import Path

import yaml
from PIL import Image

from yolo_data import prepare_yolo_dataset, yolo_box

CATEGORIES = [
    {"id": 3, "name": "forklift"},
    {"id": 8, "name": "pallet"},
]


def write_subset(root: Path, subset: int, filename: str, image_count: int) -> None:
    images = []
    annotations = []
    for index in range(image_count):
        relative_path = Path(f"warehouse-{subset}") / f"image-{index}.jpg"
        image_path = root / f"subset-{subset}" / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 10), color=(index, 0, 0)).save(image_path)
        image_id = subset * 100 + index
        images.append(
            {
                "id": image_id,
                "path": f"/dataset/subset-{subset}/{relative_path}",
                "width": 20,
                "height": 10,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 3,
                "bbox": [2, 1, 10, 5],
                "iscrowd": 0,
            }
        )
    annotations_dir = root / "rgb"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / filename).write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": CATEGORIES,
            }
        ),
        encoding="utf-8",
    )


def make_loco(root: Path) -> None:
    for subset in (2, 3, 5):
        write_subset(root, subset, f"loco-sub{subset}-v1-train.json", 6)
    for subset in (1, 4):
        write_subset(root, subset, f"loco-sub{subset}-v1-val.json", 2)


def test_prepare_yolo_dataset_preserves_split_and_hides_test(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    make_loco(raw_dir)

    dataset_yaml, summary = prepare_yolo_dataset(raw_dir, tmp_path / "yolo")

    assert summary["splits"] == {"train": 12, "validation": 6}
    assert "test" not in yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    assert not (tmp_path / "yolo" / "images" / "test").exists()
    training_images = sorted((tmp_path / "yolo" / "images" / "train").iterdir())
    assert training_images[0].is_symlink()
    label = tmp_path / "yolo" / "labels" / "train" / training_images[0].with_suffix(".txt").name
    assert label.read_text(encoding="utf-8") == "0 0.35000000 0.35000000 0.50000000 0.50000000\n"


def test_prepare_yolo_dataset_only_includes_test_when_explicit(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    make_loco(raw_dir)

    dataset_yaml, summary = prepare_yolo_dataset(
        raw_dir,
        tmp_path / "yolo",
        include_test=True,
    )

    assert summary["splits"]["test"] == 4
    assert yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))["test"] == "images/test"


def test_prepared_local_copy_is_reused(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "yolo"
    make_loco(raw_dir)
    prepare_yolo_dataset(raw_dir, output_dir, copy_images=True)
    sentinel = output_dir / "reuse-sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    _, summary = prepare_yolo_dataset(raw_dir, output_dir, copy_images=True)

    image = next((output_dir / "images" / "train").iterdir())
    assert summary["image_storage"] == "copy"
    assert image.is_file() and not image.is_symlink()
    assert sentinel.is_file()


def test_yolo_box_clips_to_image_boundaries() -> None:
    annotation = {"bbox": [-2, 2, 7, 10], "iscrowd": 0}

    assert yolo_box(annotation, width=10, height=10) == (0.25, 0.6, 0.5, 0.8)
