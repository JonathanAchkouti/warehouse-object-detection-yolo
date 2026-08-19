"""Adapt an Ultralytics checkpoint to the repository's Torchvision detection interface."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn


class UltralyticsDetector(nn.Module):
    """Return YOLO detections as ``boxes``, one-based ``labels``, and ``scores`` tensors."""

    def __init__(
        self,
        weights: Path,
        *,
        input_size: int,
        score_threshold: float,
        nms_threshold: float,
        max_detections: int,
    ) -> None:
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO(str(weights))
        # Ultralytics overloads ``train()`` as its high-level training entry point. Registering
        # the YOLO wrapper as a child ``nn.Module`` would therefore make ``self.eval()`` call
        # ``yolo.train(False)`` and accidentally start a new training run. Keep only its actual
        # PyTorch network registered while retaining the prediction wrapper as a plain attribute.
        object.__setattr__(self, "_yolo", yolo)
        self.network = yolo.model
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        """Run aspect-ratio-preserving prediction on a variable-size image list."""
        if not images:
            return []
        sources = []
        for image in images:
            image_HWC = (
                image.detach()
                .cpu()
                .clamp(0, 1)
                .permute(1, 2, 0)
                .mul(255)
                .round()
                .to(torch.uint8)
                .numpy()
            )
            sources.append(Image.fromarray(image_HWC))

        device = next(self.network.parameters()).device
        results = self._yolo.predict(
            source=sources,
            imgsz=self.input_size,
            conf=self.score_threshold,
            iou=self.nms_threshold,
            max_det=self.max_detections,
            device=device,
            verbose=False,
        )
        predictions: list[dict[str, Tensor]] = []
        for result in results:
            boxes = result.boxes
            predictions.append(
                {
                    "boxes": boxes.xyxy.to(device),
                    # LOCO/Torchmetrics uses foreground IDs 1..5; YOLO uses 0..4.
                    "labels": boxes.cls.to(device=device, dtype=torch.int64) + 1,
                    "scores": boxes.conf.to(device),
                }
            )
        return predictions
