"""Configuration-driven efficient detector used by training and evaluation."""

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection import _utils as detection_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.ops.misc import Conv2dNormActivation

from utils.config import load_config

# MobileNetV3-Large block 6 is the final stride-8 block in torchvision's SSDLite backbone.
C3_INDEX = 6


class C3TapBackbone(nn.Module):
    """Expose MobileNetV3's stride-8 feature map before the stock SSD features."""

    def __init__(self, base: nn.Module, *, semantic_fusion: bool = False) -> None:
        super().__init__()
        self.stem = base.features[0][: C3_INDEX + 1]
        self.rest = base.features[0][C3_INDEX + 1 :]
        self.tail = base.features[1]
        self.extra = base.extra
        self.semantic_fusion = semantic_fusion
        if semantic_fusion:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
            self.lateral = Conv2dNormActivation(
                672,
                40,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
            self.smooth = nn.Sequential(
                Conv2dNormActivation(
                    40,
                    40,
                    kernel_size=3,
                    padding=1,
                    groups=40,
                    norm_layer=norm_layer,
                    activation_layer=nn.Hardswish,
                ),
                Conv2dNormActivation(
                    40,
                    40,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.Hardswish,
                ),
            )

    def forward(self, inputs: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Return the stride-8 tap followed by all stock SSDLite feature levels."""
        c3 = self.stem(inputs)
        c4 = self.rest(c3)
        if self.semantic_fusion:
            top_down = functional.interpolate(
                self.lateral(c4),
                size=c3.shape[-2:],
                mode="nearest",
            )
            c3 = self.smooth(c3 + top_down)
        features = [c3, c4]

        hidden = self.tail(c4)
        features.append(hidden)
        for block in self.extra:
            hidden = block(hidden)
            features.append(hidden)

        return OrderedDict((str(index), feature) for index, feature in enumerate(features))


def _model_settings(model_config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve explicit settings or fall back to the repository's final configuration."""
    if model_config is not None:
        return model_config
    return load_config(Path("config.yaml")).get("model", {})


def create_model(
    num_classes: int,
    model_config: dict[str, Any] | None = None,
    *,
    pretrained: bool = False,
) -> nn.Module:
    """Build the configured SSDLite detector; ``num_classes`` includes background.

    ``pretrained`` is intended only for training initialization. Evaluation leaves it false so a
    saved checkpoint can be loaded without downloading weights from the internet.
    """
    settings = _model_settings(model_config)
    input_size = int(settings.get("input_size", 512))
    use_p3 = bool(settings.get("p3", True))
    semantic_fusion = bool(settings.get("p3_semantic_fusion", False))
    scales = list(settings.get("anchor_scales", [0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.95, 1.0]))
    ratios = list(settings.get("aspect_ratios", [2, 3]))

    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None
    model = ssdlite320_mobilenet_v3_large(
        weights=weights,
        weights_backbone=None,
    )
    if use_p3:
        model.backbone = C3TapBackbone(
            model.backbone,
            semantic_fusion=semantic_fusion,
        )

    in_channels = detection_utils.retrieve_out_channels(
        model.backbone,
        (input_size, input_size),
    )
    level_count = len(in_channels)
    required_scales = level_count + 1
    if len(scales) < required_scales:
        raise ValueError(
            f"The {level_count} feature levels require at least {required_scales} anchor scales; "
            f"received {len(scales)}"
        )

    anchor_generator = DefaultBoxGenerator(
        [ratios] * level_count,
        scales=scales[:required_scales],
    )
    model.anchor_generator = anchor_generator
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head = SSDLiteHead(
        in_channels,
        anchor_generator.num_anchors_per_location(),
        num_classes,
        norm_layer,
    )

    # Preserve SSD's pretrained normalization statistics while changing only its resize policy.
    model.transform.min_size = (input_size,)
    model.transform.max_size = input_size
    model.transform.fixed_size = (input_size, input_size)

    model.score_thresh = float(settings.get("score_thresh", 0.01))
    model.nms_thresh = float(settings.get("nms_thresh", 0.45))
    model.detections_per_img = int(settings.get("detections_per_img", 300))
    model.topk_candidates = int(settings.get("topk_candidates", 300))
    return model
