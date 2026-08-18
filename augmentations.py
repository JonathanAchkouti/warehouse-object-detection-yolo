"""Detection augmentations that update images and boxes together."""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor
from torchvision.transforms.v2 import ColorJitter

from dataset import DetectionTarget

DetectionTransform = Callable[
    [Tensor, DetectionTarget],
    tuple[Tensor, DetectionTarget],
]


class Compose:
    """Apply detection transforms in order."""

    def __init__(self, transforms: Sequence[DetectionTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(
        self,
        image_CHW: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        for transform in self.transforms:
            image_CHW, target = transform(image_CHW, target)
        return image_CHW, target


class RandomHorizontalFlip:
    """Flip an image and its absolute ``xyxy`` boxes with probability ``probability``."""

    def __init__(self, probability: float) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("Horizontal-flip probability must be between zero and one")
        self.probability = probability

    def __call__(
        self,
        image_CHW: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        if torch.rand(()) >= self.probability:
            return image_CHW, target

        width = image_CHW.shape[-1]
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        transformed_target: DetectionTarget = {
            **target,
            "boxes": boxes,
        }
        return image_CHW.flip(-1), transformed_target


class ImageOnlyColorJitter:
    """Apply photometric jitter without changing target geometry."""

    def __init__(self, strength: float) -> None:
        if strength < 0:
            raise ValueError("Color-jitter strength cannot be negative")
        self.transform = ColorJitter(
            brightness=strength,
            contrast=strength,
            saturation=strength,
            hue=min(strength / 2, 0.5),
        )

    def __call__(
        self,
        image_CHW: Tensor,
        target: DetectionTarget,
    ) -> tuple[Tensor, DetectionTarget]:
        return self.transform(image_CHW), target


def build_train_transform(horizontal_flip: float, color_jitter: float) -> DetectionTransform | None:
    """Build the configured training augmentation pipeline."""
    transforms: list[DetectionTransform] = []
    if color_jitter > 0:
        transforms.append(ImageOnlyColorJitter(color_jitter))
    if horizontal_flip > 0:
        transforms.append(RandomHorizontalFlip(horizontal_flip))
    return Compose(transforms) if transforms else None
