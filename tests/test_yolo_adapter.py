import sys
from pathlib import Path
from types import ModuleType

from torch import nn

from models.yolo_adapter import UltralyticsDetector


class TrainingApiYOLO(nn.Module):
    """Mimic Ultralytics' high-level ``train`` method for the regression test."""

    def __init__(self, _weights: str) -> None:
        super().__init__()
        self.model = nn.Conv2d(3, 3, kernel_size=1)

    def train(self, _mode: bool = True):  # type: ignore[override]
        raise AssertionError("the YOLO training API must not be called by detector.eval()")


def test_eval_does_not_start_ultralytics_training(monkeypatch) -> None:
    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = TrainingApiYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    detector = UltralyticsDetector(
        Path("weights.pt"),
        input_size=640,
        score_threshold=0.001,
        nms_threshold=0.7,
        max_detections=300,
    )

    assert "_yolo" not in detector._modules
    detector.eval()
    assert detector.network.training is False
