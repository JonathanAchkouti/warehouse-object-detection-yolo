"""Evaluate YOLO with the repository's unchanged LOCO mAP@0.5 metric."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset import LocoDataset, collate_fn
from eval import measure_latency_ms
from metrics.accuracy import compute_map50
from metrics.efficiency import count_parameters
from models.yolo_adapter import UltralyticsDetector
from reporting import compute_detection_metrics
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/e12_sharpen_20.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--class-metrics", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    config: dict[str, Any] = load_config(args.config)
    model_settings = config["model"]
    dataset = LocoDataset(Path(config["data"]["raw_dir"]), split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltralyticsDetector(
        args.weights,
        input_size=int(model_settings["input_size"]),
        score_threshold=float(model_settings["score_thresh"]),
        nms_threshold=float(model_settings["nms_thresh"]),
        max_detections=int(model_settings["detections_per_img"]),
    ).to(device)
    model.eval()

    sample_CHW = dataset[0][0]
    details = None
    if args.class_metrics:
        details = compute_detection_metrics(model, loader, device, dataset.label_names)
        map50 = details.map50
    else:
        map50 = compute_map50(model, loader, device)
    latency = measure_latency_ms(model, sample_CHW.to(device), device)
    parameters, trainable = count_parameters(model.network)
    from ultralytics.utils.torch_utils import get_flops

    gflops = float(get_flops(model.network, imgsz=int(model_settings["input_size"])))

    print(f"mAP@0.5:                {map50:.4f}")
    print(f"parameters:             {parameters:,} ({parameters / 1e6:.2f} M)")
    print(f"  of which trainable:   {trainable:,}")
    print(f"GFLOPs per image:       {gflops:.3f}")
    print(f"median batch-1 latency: {latency:.2f} ms ({device}) — machine-dependent, not graded")
    if details is not None:
        print("per-class AP@0.5:")
        for name, value in details.per_class.items():
            print(f"  {name:<24} {value:.4f}")


if __name__ == "__main__":
    main()
