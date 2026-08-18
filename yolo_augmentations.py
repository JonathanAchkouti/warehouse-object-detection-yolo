"""Domain-generalization augmentations for YOLO training on LOCO."""

from __future__ import annotations

from typing import Any


def _probability(settings: dict[str, Any], name: str, default: float) -> float:
    value = float(settings.get(name, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return value


def build_domain_augmentations(settings: dict[str, Any]) -> list[Any]:
    """Return image-only Albumentations that preserve detection boxes.

    Grayscale and sharpening are deliberately occasional rather than mandatory: the detector
    learns not to depend on colour while retaining useful RGB cues and COCO pretraining.
    """
    import albumentations as A

    grayscale = _probability(settings, "grayscale_probability", 0.20)
    blur = _probability(settings, "blur_probability", 0.18)
    sharpen = _probability(settings, "sharpen_probability", 0.12)
    gamma = _probability(settings, "gamma_probability", 0.20)
    brightness_contrast = _probability(
        settings,
        "brightness_contrast_probability",
        0.30,
    )
    return [
        A.ToGray(p=grayscale),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=blur,
        ),
        A.Sharpen(alpha=(0.10, 0.30), lightness=(0.8, 1.2), p=sharpen),
        A.RandomGamma(gamma_limit=(70, 140), p=gamma),
        A.RandomBrightnessContrast(
            brightness_limit=0.30,
            contrast_limit=0.30,
            p=brightness_contrast,
        ),
    ]
