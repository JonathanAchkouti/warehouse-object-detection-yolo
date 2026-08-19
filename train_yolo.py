"""Fine-tune a COCO-pretrained YOLO detector on the permitted LOCO development subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from utils.config import load_config
from yolo_augmentations import build_domain_augmentations
from yolo_data import prepare_yolo_dataset


def json_value(value: Any) -> Any:
    """Convert common metric scalar containers into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def train(config: dict[str, Any], *, smoke: bool = False) -> Path:
    """Prepare the leak-free dataset view and train or resume YOLO."""
    from ultralytics import YOLO

    data_settings = config["data"]
    model_settings = config["model"]
    train_settings = config["train"]
    output_dir = Path(config["output_dir"])
    if smoke:
        output_dir = output_dir / "smoke"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset_yaml, dataset_summary = prepare_yolo_dataset(
        Path(data_settings["raw_dir"]),
        Path(data_settings.get("yolo_workspace", "/content/loco-yolo")),
        include_test=False,
        copy_images=bool(data_settings.get("copy_images", True)),
    )
    augmentations = build_domain_augmentations(train_settings.get("domain_augmentations", {}))
    last_weights = output_dir / "weights" / "last.pt"
    should_resume = bool(train_settings.get("resume", True)) and last_weights.is_file()
    model = YOLO(str(last_weights if should_resume else model_settings["weights"]))

    common_arguments: dict[str, Any] = {
        "data": str(dataset_yaml),
        "imgsz": int(model_settings["input_size"]),
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": int(train_settings["num_workers"]),
        "augmentations": augmentations,
    }
    if should_resume:
        print(f"Resuming YOLO checkpoint: {last_weights}", flush=True)
        results = model.train(resume=True, **common_arguments)
    else:
        epochs = 1 if smoke else int(train_settings["epochs"])
        print(f"Starting COCO fine-tuning from: {model_settings['weights']}", flush=True)
        results = model.train(
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
            epochs=epochs,
            batch=2 if smoke else int(train_settings["batch_size"]),
            patience=0 if smoke else int(train_settings["patience"]),
            seed=int(train_settings["seed"]),
            deterministic=bool(train_settings.get("deterministic", True)),
            amp=bool(train_settings.get("amp", True)),
            optimizer=str(train_settings.get("optimizer", "auto")),
            cos_lr=bool(train_settings.get("cos_lr", True)),
            close_mosaic=0 if smoke else int(train_settings.get("close_mosaic", 10)),
            fliplr=float(train_settings.get("horizontal_flip", 0.5)),
            hsv_h=float(train_settings.get("hsv_h", 0.03)),
            hsv_s=float(train_settings.get("hsv_s", 0.4)),
            hsv_v=float(train_settings.get("hsv_v", 0.4)),
            degrees=float(train_settings.get("degrees", 3.0)),
            translate=float(train_settings.get("translate", 0.1)),
            scale=float(train_settings.get("scale", 0.4)),
            shear=float(train_settings.get("shear", 1.0)),
            perspective=float(train_settings.get("perspective", 0.0005)),
            mosaic=0.0 if smoke else float(train_settings.get("mosaic", 0.5)),
            mixup=0.0 if smoke else float(train_settings.get("mixup", 0.05)),
            multi_scale=0.0 if smoke else float(train_settings.get("multi_scale", 0.25)),
            plots=not smoke,
            save=True,
            save_period=-1,
            verbose=True,
            **common_arguments,
        )

    best_weights = output_dir / "weights" / "best.pt"
    summary = {
        "model": model_settings["weights"],
        "input_size": int(model_settings["input_size"]),
        "epochs_requested": 1 if smoke else int(train_settings["epochs"]),
        "resumed": should_resume,
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "dataset": dataset_summary,
        "metrics": json_value(getattr(results, "results_dict", {})),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return best_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/e12_sharpen_20.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
