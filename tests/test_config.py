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
