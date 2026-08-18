"""Train the configured LOCO detector with fast, resumable validation runs."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from wandb.sdk.wandb_run import Run

from augmentations import build_train_transform
from dataset import LocoDataset, collate_fn
from metrics.efficiency import count_parameters
from models.baseline import create_model
from reporting import DetectionMetrics, compute_detection_metrics
from sampling import repeat_factor_weights
from training_utils import (
    ModelEMA,
    optimizer_steps_per_epoch,
    seed_worker,
    should_evaluate,
    warmup_cosine_scheduler,
)
from utils.config import load_config


def make_loader(
    dataset: LocoDataset,
    settings: dict[str, Any],
    *,
    training: bool,
    generator: torch.Generator,
) -> DataLoader:
    """Build a portable DataLoader with fast CUDA-friendly settings."""
    workers = int(settings["num_workers"])
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(settings["batch_size"]),
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(settings.get("persistent_workers", True)),
            prefetch_factor=int(settings.get("prefetch_factor", 2)),
        )

    repeat_threshold = float(settings.get("repeat_factor_threshold", 0.0))
    if training and repeat_threshold > 0:
        weights = repeat_factor_weights(dataset, repeat_threshold)
        loader_kwargs["sampler"] = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = training
        if training:
            loader_kwargs["generator"] = generator
    return DataLoader(**loader_kwargs)


def move_batch(
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    channels_last: bool,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Move a detection batch to the training device."""
    del channels_last  # Detection images remain CHW; the model parameters use channels-last.
    moved_images = [image.to(device, non_blocking=True) for image in images]
    moved_targets = [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in targets
    ]
    return moved_images, moved_targets


def save_run_config(config: dict[str, Any], output_dir: Path) -> None:
    """Freeze the exact run configuration next to its checkpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def append_history(path: Path, record: dict[str, Any]) -> None:
    """Append one raw epoch record to the experiment history."""
    with path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, sort_keys=True) + "\n")


def save_last_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_map: float,
    config: dict[str, Any],
) -> None:
    """Write all state required to resume after a Colab disconnect."""
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_map": best_map,
            "model_config": config.get("model", {}),
        },
        path,
    )


def restore_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    model_config: dict[str, Any],
    device: torch.device,
) -> tuple[int, int, float]:
    """Restore training state and return start epoch, global step, and best validation mAP."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_config") != model_config:
        raise ValueError(
            f"Cannot resume {path}: its model configuration differs from the current run"
        )
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]), int(checkpoint["global_step"]), float(checkpoint["best_map"])


def train(config: dict[str, Any], run: Run | None, *, smoke: bool = False) -> None:
    """Train the detector and save resumable state plus the best EMA weights."""
    settings = config["train"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_transform = build_train_transform(
        float(settings.get("horizontal_flip", 0.0)),
        float(settings.get("color_jitter", 0.0)),
    )
    raw_dir = Path(config["data"]["raw_dir"])
    train_dataset = LocoDataset(raw_dir, split="train", transform=train_transform)
    val_dataset = LocoDataset(raw_dir, split="validation")
    if train_dataset.category_labels != val_dataset.category_labels:
        raise ValueError("Training and validation splits define different categories")

    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(train_dataset, settings, training=True, generator=generator)
    val_loader = make_loader(val_dataset, settings, training=False, generator=generator)

    channels_last = bool(settings.get("channels_last", True)) and device.type == "cuda"
    model = create_model(
        train_dataset.num_classes,
        config.get("model"),
        pretrained=bool(settings.get("pretrained", True)),
    ).to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    optimizer = SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings["learning_rate"]),
        momentum=float(settings["momentum"]),
        weight_decay=float(settings["weight_decay"]),
        nesterov=bool(settings.get("nesterov", True)),
    )
    epochs = 1 if smoke else int(settings["epochs"])
    max_train_batches = min(len(train_loader), 20) if smoke else len(train_loader)
    max_val_batches = 10 if smoke else None
    accumulation_steps = int(settings.get("accumulation_steps", 1))
    steps_per_epoch = optimizer_steps_per_epoch(max_train_batches, accumulation_steps)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=epochs * steps_per_epoch,
        warmup_steps=int(settings.get("warmup_iters", 500)),
    )
    amp_enabled = bool(settings.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ModelEMA(model, decay=float(settings.get("ema_decay", 0.9998)))

    output_dir = Path(config["output_dir"])
    if smoke:
        output_dir = output_dir / "smoke"
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(config, output_dir)

    last_path = weights_dir / "last.pt"
    best_path = weights_dir / "best.pt"
    start_epoch = 0
    global_step = 0
    best_map = -1.0
    resume = bool(settings.get("resume", True)) and not smoke
    history_path = output_dir / "history.jsonl"
    if not resume or not last_path.is_file():
        history_path.write_text("", encoding="utf-8")
    if resume and last_path.is_file():
        start_epoch, global_step, best_map = restore_checkpoint(
            last_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=config.get("model", {}),
            device=device,
        )
        print(f"resumed {last_path} at epoch {start_epoch + 1}, best mAP50 {best_map:.4f}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        processed_batches = 0
        progress = tqdm(
            train_loader,
            total=max_train_batches,
            desc=f"epoch {epoch + 1}/{epochs}",
            leave=False,
        )
        for batch_index, (images, targets) in enumerate(progress):
            if batch_index >= max_train_batches:
                break
            images, targets = move_batch(
                images,
                targets,
                device,
                channels_last=channels_last,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                losses = model(images, targets)
                loss = sum(losses.values())
                scaled_loss = loss / accumulation_steps

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {loss.item()}")
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index + 1
            ) % accumulation_steps == 0 or batch_index + 1 == max_train_batches
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(settings.get("grad_clip", 10.0)),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                global_step += 1

            epoch_loss += float(loss.detach())
            processed_batches += 1
            progress.set_postfix(
                loss=f"{epoch_loss / processed_batches:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        epoch_loss /= max(processed_batches, 1)
        metrics: DetectionMetrics | None = None
        if should_evaluate(epoch, epochs, int(settings.get("eval_every", 2))):
            metrics = compute_detection_metrics(
                ema.model,
                val_loader,
                device,
                val_dataset.label_names,
                max_batches=max_val_batches,
            )
            if metrics.map50 > best_map:
                best_map = metrics.map50
                torch.save(ema.state_dict(), best_path)
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "map50": metrics.map50,
                            "per_class": metrics.per_class,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"  saved new best: {best_map:.4f}")

        save_last_checkpoint(
            last_path,
            epoch=epoch + 1,
            global_step=global_step,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_map=best_map,
            config=config,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        map_text = "not evaluated" if metrics is None else f"{metrics.map50:.4f}"
        print(
            f"epoch {epoch + 1}/{epochs}  loss {epoch_loss:.4f}  "
            f"val_mAP50 {map_text}  {epoch_seconds:.0f}s"
        )
        if metrics is not None:
            print(
                "  per-class AP50: "
                + ", ".join(f"{name}={value:.4f}" for name, value in metrics.per_class.items())
            )
        append_history(
            history_path,
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": epoch_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
                "val_map50": None if metrics is None else metrics.map50,
                "per_class": {} if metrics is None else metrics.per_class,
            },
        )
        if run is not None:
            log_values: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "epoch_seconds": epoch_seconds,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if metrics is not None:
                log_values["val_mAP50"] = metrics.map50
                log_values.update(
                    {f"val_AP50/{name}": value for name, value in metrics.per_class.items()}
                )
            run.log(log_values)

    parameters, trainable = count_parameters(model)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_map50": best_map,
                "best_weights": str(best_path),
                "parameters": parameters,
                "trainable_parameters": trainable,
                "epochs": epochs,
                "global_steps": global_step,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] best weights: {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--smoke", action="store_true", help="run 20 train and 10 val batches")
    args = parser.parse_args()
    config = load_config(args.config)
    wandb_config = config["wandb"]

    if not wandb_config["enabled"]:
        train(config, run=None, smoke=args.smoke)
        return

    with wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        config=config,
    ) as run:
        train(config, run, smoke=args.smoke)


if __name__ == "__main__":
    main()
