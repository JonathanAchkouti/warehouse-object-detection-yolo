"""Check a pretrained YOLO candidate against the assignment's deployment budgets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from metrics.efficiency import count_parameters, measure_gflops
from models.yolo_adapter import UltralyticsDetector
from utils.config import load_config

BASELINE_PARAMETERS = 18_950_729
BASELINE_GFLOPS = 23.825


def budget_failures(
    parameters: int,
    gflops: float,
    *,
    max_parameters: int = BASELINE_PARAMETERS,
    max_gflops: float = BASELINE_GFLOPS,
) -> list[str]:
    """Describe any efficiency limits exceeded by a deployed candidate."""
    failures = []
    if parameters > max_parameters:
        failures.append(f"parameters {parameters:,} exceed {max_parameters:,}")
    if gflops > max_gflops:
        failures.append(f"GFLOPs {gflops:.6f} exceed {max_gflops:.6f}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/e13_yolo26s_896.yaml"),
    )
    parser.add_argument("--probe-height", type=int, default=1080)
    parser.add_argument("--probe-width", type=int, default=1920)
    parser.add_argument("--max-parameters", type=int, default=BASELINE_PARAMETERS)
    parser.add_argument("--max-gflops", type=float, default=BASELINE_GFLOPS)
    args = parser.parse_args()
    if args.probe_height < 1 or args.probe_width < 1:
        parser.error("probe dimensions must be positive")
    if args.max_parameters < 1 or args.max_gflops <= 0:
        parser.error("efficiency limits must be positive")

    config: dict[str, Any] = load_config(args.config)
    model_settings = config["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltralyticsDetector(
        Path(model_settings["weights"]),
        input_size=int(model_settings["input_size"]),
        score_threshold=float(model_settings["score_thresh"]),
        nms_threshold=float(model_settings["nms_thresh"]),
        max_detections=int(model_settings["detections_per_img"]),
    ).to(device)
    model.eval()
    probe = torch.zeros(3, args.probe_height, args.probe_width, dtype=torch.float32)

    # The first prediction performs Ultralytics' deployment fusion. The COCO head is slightly
    # larger than the five-class fine-tuned head, making this a conservative pre-training gate.
    with torch.no_grad():
        model([probe.to(device)])
    parameters, trainable = count_parameters(model.network)
    gflops = measure_gflops(model, probe, device)
    failures = budget_failures(
        parameters,
        gflops,
        max_parameters=args.max_parameters,
        max_gflops=args.max_gflops,
    )

    print(f"model:                  {model_settings['weights']}")
    print(f"input size:             {model_settings['input_size']}")
    print(f"probe shape:            {tuple(probe.shape)}")
    print(f"parameters:             {parameters:,} ({parameters / 1e6:.2f} M)")
    print(f"  of which trainable:   {trainable:,}")
    print(f"GFLOPs per image:       {gflops:.6f}")
    print(f"parameter budget:       {args.max_parameters:,}")
    print(f"GFLOPs budget:          {args.max_gflops:.6f}")
    if failures:
        raise SystemExit("Budget check failed: " + "; ".join(failures))
    print("budget check:           PASS")


if __name__ == "__main__":
    main()
