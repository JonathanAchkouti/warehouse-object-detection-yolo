from pathlib import Path

from utils.config import load_config


def test_load_config_recursively_merges_experiment_override(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "model:\n  input_size: 512\n  p3: true\ntrain:\n  epochs: 60\n",
        encoding="utf-8",
    )
    (tmp_path / "experiment.yaml").write_text(
        "base: base.yaml\nmodel:\n  input_size: 384\noutput_dir: runs/e01\n",
        encoding="utf-8",
    )

    config = load_config(tmp_path / "experiment.yaml")

    assert config["model"] == {"input_size": 384, "p3": True}
    assert config["train"]["epochs"] == 60
    assert config["output_dir"] == "runs/e01"


def test_e12_changes_only_sharpening_and_output_directory() -> None:
    e11 = load_config(Path("configs/e11_yolo11n_domain_aug.yaml"))
    e12 = load_config(Path("configs/e12_sharpen_20.yaml"))

    assert e12["train"]["domain_augmentations"]["sharpen_probability"] == 0.20
    assert e12["output_dir"] == "runs/e12_sharpen_20"

    e11["train"]["domain_augmentations"]["sharpen_probability"] = 0.20
    e11["output_dir"] = "runs/e12_sharpen_20"
    assert e12 == e11


def test_e13_keeps_e12_domain_recipe_while_scaling_model_and_resolution() -> None:
    e12 = load_config(Path("configs/e12_sharpen_20.yaml"))
    e13 = load_config(Path("configs/e13_yolo26s_896.yaml"))

    assert e13["model"]["weights"] == "yolo26s.pt"
    assert e13["model"]["input_size"] == 896
    assert e13["train"]["batch_size"] == 8
    assert e13["train"]["patience"] == 25
    assert e13["train"]["multi_scale"] == 0.15
    assert e13["output_dir"] == "runs/e13_yolo26s_896"

    e12["model"].update(weights="yolo26s.pt", input_size=896)
    e12["train"].update(batch_size=8, patience=25, multi_scale=0.15)
    e12["output_dir"] = "runs/e13_yolo26s_896"
    assert e13 == e12
