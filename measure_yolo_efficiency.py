"""Measure final YOLO efficiency with the repository's unchanged grading utilities."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from dataset import LocoDataset
from metrics.efficiency import count_parameters, measure_gflops
from models.yolo_adapter import UltralyticsDetector
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/e12_sharpen_20.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")

    config: dict[str, Any] = load_config(args.config)
    model_settings = config["model"]
    dataset = LocoDataset(Path(config["data"]["raw_dir"]), split=args.split)
    sample_CHW = dataset[0][0]
    first_image_id = dataset.images[0]["id"]
    probe_path = dataset.image_paths[first_image_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UltralyticsDetector(
        args.weights,
        input_size=int(model_settings["input_size"]),
        score_threshold=float(model_settings["score_thresh"]),
        nms_threshold=float(model_settings["nms_thresh"]),
        max_detections=int(model_settings["detections_per_img"]),
    ).to(device)
    model.eval()

    # Match the template evaluator's deployment order: prediction first fuses batch normalization,
    # then the deployed network is counted and traced on the fixed first image of the chosen split.
    with torch.no_grad():
        model([sample_CHW.to(device)])
    parameters, trainable = count_parameters(model.network)
    gflops = measure_gflops(model, sample_CHW, device)

    with args.weights.open("rb") as weights_file:
        checkpoint_sha256 = hashlib.file_digest(weights_file, "sha256").hexdigest()

    print(f"checkpoint SHA-256:     {checkpoint_sha256}")
    print(f"probe:                  {probe_path}")
    print(f"probe shape:            {tuple(sample_CHW.shape)}")
    print(f"parameters:             {parameters:,} ({parameters / 1e6:.2f} M)")
    print(f"  of which trainable:   {trainable:,}")
    print(f"GFLOPs per image:       {gflops:.6f}")


if __name__ == "__main__":
    main()
