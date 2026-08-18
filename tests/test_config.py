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
