import torch

from metrics.efficiency import count_parameters
from models.baseline import create_model

MODEL_CONFIG = {
    "input_size": 320,
    "p3": True,
    "anchor_scales": [0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 0.95, 1.0],
    "aspect_ratios": [2, 3],
    "score_thresh": 0.01,
    "nms_thresh": 0.45,
    "detections_per_img": 300,
    "topk_candidates": 300,
}


def test_p3_ssdlite_has_expected_efficiency_and_outputs() -> None:
    model = create_model(6, MODEL_CONFIG)

    total, trainable = count_parameters(model)
    assert total == 2_265_300
    assert trainable == total
    assert len(model.anchor_generator.aspect_ratios) == 7
    assert model.transform.fixed_size == (320, 320)

    model.eval()
    with torch.no_grad():
        prediction = model([torch.rand(3, 240, 320)])[0]

    assert prediction["boxes"].shape == (300, 4)
    assert prediction["scores"].shape == (300,)
    assert prediction["labels"].shape == (300,)


def test_missing_anchor_scales_fails_clearly() -> None:
    settings = {**MODEL_CONFIG, "anchor_scales": [0.1, 0.2]}

    try:
        create_model(6, settings)
    except ValueError as error:
        assert "require at least 8 anchor scales" in str(error)
    else:
        raise AssertionError("Expected an invalid anchor configuration to fail")
