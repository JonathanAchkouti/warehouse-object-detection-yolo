"""Additional validation reporting that does not alter the official grading metric."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DetectionMetrics:
    """Validation mAP and its foreground-class breakdown."""

    map50: float
    per_class: dict[str, float]


@torch.no_grad()
def compute_detection_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_names: dict[int, str],
    *,
    max_batches: int | None = None,
) -> DetectionMetrics:
    """Compute mAP@0.5 and per-class AP using a separate reporting metric."""
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
    for batch_index, (images, targets) in enumerate(tqdm(loader, desc="validating", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        predictions = model([image.to(device, non_blocking=True) for image in images])
        metric.update(
            [
                {key: value.detach().cpu() for key, value in prediction.items()}
                for prediction in predictions
            ],
            [{key: value.cpu() for key, value in target.items()} for target in targets],
        )

    values = metric.compute()
    class_ids = values.get("classes", torch.empty(0, dtype=torch.int64)).reshape(-1)
    class_aps = values.get("map_per_class", torch.empty(0)).reshape(-1)
    per_class = {
        label_names.get(int(class_id), f"class_{int(class_id)}"): float(class_ap)
        for class_id, class_ap in zip(class_ids, class_aps, strict=True)
    }
    return DetectionMetrics(map50=float(values["map_50"]), per_class=per_class)
