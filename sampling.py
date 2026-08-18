"""Rare-class-aware sampling utilities."""

import math
from collections import Counter

from torch import Tensor, float32, tensor

from dataset import LocoDataset


def repeat_factor_weights(dataset: LocoDataset, threshold: float) -> Tensor:
    """Return LVIS-style per-image repeat weights based on class image frequency."""
    if threshold <= 0:
        return tensor([1.0] * len(dataset), dtype=float32)
    if not dataset:
        return tensor([], dtype=float32)

    labels_per_image = [dataset.image_labels(index) for index in range(len(dataset))]
    image_counts = Counter(label for labels in labels_per_image for label in labels)
    frequencies = {label: count / len(dataset) for label, count in image_counts.items()}
    class_factors = {
        label: max(1.0, math.sqrt(threshold / frequency))
        for label, frequency in frequencies.items()
    }
    return tensor(
        [
            max((class_factors[label] for label in labels), default=1.0)
            for labels in labels_per_image
        ],
        dtype=float32,
    )
