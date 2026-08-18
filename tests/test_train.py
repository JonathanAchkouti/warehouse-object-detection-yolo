import json
from pathlib import Path

from PIL import Image

from train import train


def write_smoke_data(root: Path) -> None:
    """Create two training images and one validation image with one foreground class."""
    categories = [{"id": 7, "name": "pallet"}]
    annotations_dir = root / "rgb"
    images_dir = root / "subset-2"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    images = []
    annotations = []
    for index in range(3):
        image_id = index + 1
        filename = f"image-{image_id}.png"
        Image.new("RGB", (96, 64), color=(100 + index, 80, 40)).save(images_dir / filename)
        images.append(
            {
                "id": image_id,
                "path": f"/dataset/subset-2/{filename}",
                "width": 96,
                "height": 64,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 7,
                "bbox": [16, 12, 32, 24],
            }
        )

    populated = {"images": images, "annotations": annotations, "categories": categories}
    empty = {"images": [], "annotations": [], "categories": categories}
    (annotations_dir / "loco-sub2-v1-train.json").write_text(json.dumps(populated))
    (annotations_dir / "loco-sub3-v1-train.json").write_text(json.dumps(empty))
    (annotations_dir / "loco-sub5-v1-train.json").write_text(json.dumps(empty))


def test_train_smoke_saves_resumable_and_evaluation_weights(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    write_smoke_data(data_dir)
    config = {
        "data": {"raw_dir": str(data_dir)},
        "model": {
            "input_size": 320,
            "p3": True,
            "anchor_scales": [0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.95, 1.0],
            "aspect_ratios": [2, 3],
            "score_thresh": 0.01,
            "nms_thresh": 0.45,
            "detections_per_img": 50,
            "topk_candidates": 50,
        },
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "accumulation_steps": 1,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 0.00004,
            "num_workers": 0,
            "seed": 0,
            "pretrained": False,
            "amp": False,
            "channels_last": False,
            "warmup_iters": 1,
            "eval_every": 1,
            "grad_clip": 10.0,
            "ema_decay": 0.9,
            "horizontal_flip": 0.0,
            "color_jitter": 0.0,
            "repeat_factor_threshold": 0.0,
            "resume": False,
        },
        "output_dir": str(tmp_path / "run"),
        "wandb": {"enabled": False, "project": "test", "entity": None},
    }

    train(config, run=None, smoke=True)

    weights_dir = tmp_path / "run" / "smoke" / "weights"
    assert (weights_dir / "best.pt").is_file()
    assert (weights_dir / "last.pt").is_file()
    assert (tmp_path / "run" / "smoke" / "config.yaml").is_file()
    assert (tmp_path / "run" / "smoke" / "history.jsonl").is_file()
    assert (tmp_path / "run" / "smoke" / "summary.json").is_file()
