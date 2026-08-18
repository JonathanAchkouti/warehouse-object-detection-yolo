"""Load LOCO data for Torchvision detectors."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

import torch
from einops import rearrange
from numpy import asarray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class DetectionTarget(TypedDict):
    boxes: Tensor
    labels: Tensor


# Each LOCO annotation file is COCO JSON. The fields used here are:
#   images:      id, path (absolute inside the archive), width, height
#   annotations: image_id, category_id, bbox as [x, y, width, height], iscrowd
#   categories:  id, name
DEVELOPMENT_FILES = (
    "loco-sub2-v1-train.json",
    "loco-sub3-v1-train.json",
    "loco-sub5-v1-train.json",
)
TEST_FILES = (
    "loco-sub1-v1-val.json",
    "loco-sub4-v1-val.json",
)
SUBSET_FILES: dict[str, tuple[str, ...]] = {
    "train": DEVELOPMENT_FILES,
    "validation": DEVELOPMENT_FILES,
    "test": TEST_FILES,
}

# One image in every five from each development subset is reserved for validation.
VALIDATION_EVERY = 5

# Image ``path`` values are absolute within the LOCO archive and start with this prefix.
LOCO_ARCHIVE_ROOT = "/dataset"
ANNOTATIONS_DIRNAME = "rgb"


class LocoDataset(Dataset[tuple[Tensor, DetectionTarget]]):
    """Load a LOCO split with boxes in absolute ``xyxy`` coordinates."""

    def __init__(
        self,
        raw_dir: Path,
        split: Literal["train", "validation", "test"],
        transform: Callable[[Tensor, DetectionTarget], tuple[Tensor, DetectionTarget]]
        | None = None,
        files: tuple[str, ...] | None = None,
        *,
        use_holdout: bool = True,
    ) -> None:
        filenames = files if files is not None else SUBSET_FILES[split]
        raw_dir = self._resolve_raw_dir(raw_dir, filenames)
        self.transform = transform
        self.images: list[dict] = []
        self.image_paths: dict[int, Path] = {}
        self.annotations: defaultdict[int, list[dict]] = defaultdict(list)
        categories: list[dict] | None = None

        for filename in filenames:
            subset = self._load_subset(raw_dir / ANNOTATIONS_DIRNAME / filename)
            subset_categories = sorted(subset["categories"], key=lambda category: category["id"])
            if categories is None:
                categories = subset_categories
            elif subset_categories != categories:
                raise ValueError(f"Category definitions differ in {filename}")

            source_image_ids = {image["id"] for image in subset["images"]}
            image_ids: dict[int, int] = {}
            for image_index, image in enumerate(subset["images"]):
                if split != "test" and use_holdout:
                    is_validation = image_index % VALIDATION_EVERY == 0
                    if (split == "validation") != is_validation:
                        continue
                image_id = len(self.images) + 1
                image_ids[image["id"]] = image_id
                self.images.append({**image, "id": image_id})
                # LOCO records an archive-absolute path; its root maps to the data directory.
                self.image_paths[image_id] = raw_dir / Path(image["path"]).relative_to(
                    LOCO_ARCHIVE_ROOT
                )

            for annotation in subset["annotations"]:
                if annotation["image_id"] not in source_image_ids:
                    raise ValueError(
                        f"Annotation {annotation.get('id')} references an unknown image "
                        f"in {filename}"
                    )
                image_id = image_ids.get(annotation["image_id"])
                if image_id is None:
                    continue
                self.annotations[image_id].append({**annotation, "image_id": image_id})

        if categories is None:
            raise ValueError(f"No LOCO annotations found for {split}")
        categories = sorted(categories, key=lambda category: category["id"])
        self.category_labels = {
            category["id"]: label for label, category in enumerate(categories, start=1)
        }
        # Label 0 is the background class and has no name.
        self.label_names: dict[int, str] = {
            label: category["name"] for label, category in enumerate(categories, start=1)
        }

    @staticmethod
    def _resolve_raw_dir(raw_dir: Path, filenames: tuple[str, ...]) -> Path:
        """Return the LOCO directory holding ``rgb`` annotations and the ``subset-*`` images."""
        for candidate in (raw_dir, raw_dir.parent):
            if all(
                (candidate / ANNOTATIONS_DIRNAME / filename).is_file() for filename in filenames
            ):
                return candidate
        expected = raw_dir / ANNOTATIONS_DIRNAME / filenames[0]
        raise FileNotFoundError(
            f"LOCO annotations not found. Expected files such as {expected}; set data.raw_dir to "
            "the directory scripts/download_loco.sh wrote, which holds rgb/ and subset-*/."
        )

    @staticmethod
    def _load_subset(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def num_classes(self) -> int:
        """Return foreground classes plus the background class."""
        return len(self.category_labels) + 1

    def __len__(self) -> int:
        return len(self.images)

    def image_labels(self, index: int) -> set[int]:
        """Return foreground labels present in an image without loading its pixels."""
        image_id = self.images[index]["id"]
        return {
            self.category_labels[annotation["category_id"]]
            for annotation in self.annotations.get(image_id, [])
            if not annotation.get("iscrowd", 0)
            and annotation["bbox"][2] > 0
            and annotation["bbox"][3] > 0
        }

    def __getitem__(self, index: int) -> tuple[Tensor, DetectionTarget]:
        image_info = self.images[index]
        with Image.open(self.image_paths[image_info["id"]]) as image:
            image_HWC = asarray(image.convert("RGB"), dtype="uint8").copy()
        image_CHW = (
            rearrange(
                torch.from_numpy(image_HWC),
                "height width channels -> channels height width",
            ).float()
            / 255
        )

        boxes_NQ: list[list[float]] = []
        labels_N: list[int] = []
        for annotation in self.annotations.get(image_info["id"], []):
            x, y, width, height = annotation["bbox"]
            if annotation.get("iscrowd", 0) or width <= 0 or height <= 0:
                continue
            boxes_NQ.append([x, y, x + width, y + height])
            labels_N.append(self.category_labels[annotation["category_id"]])

        target: DetectionTarget = {
            "boxes": torch.tensor(boxes_NQ, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels_N, dtype=torch.int64),
        }
        if self.transform is not None:
            image_CHW, target = self.transform(image_CHW, target)
        return image_CHW, target


def collate_fn(
    batch: list[tuple[Tensor, DetectionTarget]],
) -> tuple[list[Tensor], list[DetectionTarget]]:
    """Return variable-size images and targets as lists."""
    images_L, targets_L = zip(*batch, strict=True)
    return list(images_L), list(targets_L)
