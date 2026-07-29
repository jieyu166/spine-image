import json

import cv2
import numpy as np
import pytest
import torch

from inference_vertebra import (
    POINT_NAMES_6,
    VertebraInference,
    infer_points_per_vertebra,
    validate_ensemble_layout,
)
from train_vertebra_model import VertebraCornerModel, VertebraDataset
from train_vertebra_model_cv import InMemoryVertebraDataset


def point(x):
    return {"x": x, "y": x + 0.5}


def test_extract_point_slots_keeps_middle_superior_and_inferior():
    points = {
        "anteriorSuperior": point(1),
        "middleSuperior": point(2),
        "posteriorSuperior": point(3),
        "posteriorInferior": point(4),
        "middleInferior": point(5),
        "anteriorInferior": point(6),
    }

    slots = VertebraDataset.extract_point_slots(points, boundary=None)

    assert slots == [
        point(1),
        point(2),
        point(3),
        point(4),
        point(5),
        point(6),
    ]


def test_extract_point_slots_preserves_legacy_and_boundary_layouts():
    legacy = [point(1), point(3), point(4), point(6)]
    six_point_list = [point(index) for index in range(1, 7)]
    upper_boundary = {
        "anteriorSuperior": point(1),
        "middleSuperior": point(2),
        "posteriorSuperior": point(3),
    }
    lower_boundary = {
        "posteriorInferior": point(4),
        "middleInferior": point(5),
        "anteriorInferior": point(6),
    }

    assert VertebraDataset.extract_point_slots(six_point_list) == six_point_list
    assert VertebraDataset.extract_point_slots(legacy) == [
        point(1), None, point(3), point(4), None, point(6)
    ]
    assert VertebraDataset.extract_point_slots(upper_boundary, boundary="upper") == [
        point(1), point(2), point(3), None, None, None
    ]
    assert VertebraDataset.extract_point_slots(lower_boundary, boundary="lower") == [
        None, None, None, point(4), point(5), point(6)
    ]
    assert VertebraDataset.extract_point_slots(
        [point(1), point(3)], boundary="upper"
    ) == [point(1), None, point(3), None, None, None]
    assert VertebraDataset.extract_point_slots(
        [point(4), point(6)], boundary="lower"
    ) == [None, None, None, point(4), None, point(6)]


def test_dataset_targets_include_both_middle_point_heatmaps(tmp_path):
    image_path = tmp_path / "case.png"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps([{
            "image_path": "case.png",
            "vertebrae": [{
                "name": "L1",
                "points": {
                    "anteriorSuperior": point(1),
                    "middleSuperior": point(2),
                    "posteriorSuperior": point(3),
                    "posteriorInferior": point(4),
                    "middleInferior": point(5),
                    "anteriorInferior": point(6),
                },
            }],
        }]),
        encoding="utf-8",
    )
    dataset = VertebraDataset(
        data_dir=tmp_path,
        annotations_file=annotation_path,
        transform=None,
        max_vertebrae=1,
    )

    _, targets = dataset[0]

    assert targets["heatmaps"].shape == (6, 128, 128)
    assert targets["valid_mask"].tolist() == [1.0] * 6
    assert targets["keypoints"][1].tolist() == pytest.approx([0.02, 0.025])
    assert targets["keypoints"][4].tolist() == pytest.approx([0.05, 0.055])


def test_cv_dataset_allocates_six_channels_per_vertebra(tmp_path):
    dataset = InMemoryVertebraDataset(
        data_dir=tmp_path,
        annotations=[],
        max_vertebrae=2,
    )

    assert dataset.num_channels == 12


def test_model_head_outputs_six_channels_per_vertebra():
    model = VertebraCornerModel(max_vertebrae=2, pretrained=False)

    assert model.num_channels == 12
    assert model.channel_embed.shape[0] == 12
    assert model.heatmap_final.out_channels == 12


def test_inference_detects_four_or_six_points_from_checkpoint_channels():
    legacy_state = {"channel_embed": torch.zeros(32, 64)}
    six_point_state = {"channel_embed": torch.zeros(48, 64)}

    assert infer_points_per_vertebra(legacy_state, max_vertebrae=8) == 4
    assert infer_points_per_vertebra(six_point_state, max_vertebrae=8) == 6


def test_ensemble_rejects_mixed_four_and_six_point_layouts():
    validate_ensemble_layout(main_points=6, member_points=6)

    with pytest.raises(ValueError, match="incompatible point layout"):
        validate_ensemble_layout(main_points=6, member_points=4)


def test_v3_decoder_emits_middle_points_for_six_point_model():
    inference = VertebraInference.__new__(VertebraInference)
    inference.max_vertebrae = 1
    inference.points_per_vertebra = 6
    inference.decode_blur_sigma = 0
    inference.decode_blur_ksize = 1

    heatmaps = torch.full((1, 6, 8, 8), -10.0)
    for channel in range(6):
        heatmaps[0, channel, channel + 1, channel + 1] = 10.0

    vertebrae, _, _ = inference._decode_v3(
        {"heatmaps": heatmaps},
        predicted_count=1,
        slot_names=["L1"],
        boundary={"upper": [], "lower": []},
        orig_w=80,
        orig_h=80,
        confidence_threshold=0.2,
    )

    assert list(vertebrae[0]["points"]) == POINT_NAMES_6


def test_aspect_fix_scales_middle_points_with_the_corners():
    inference = VertebraInference.__new__(VertebraInference)
    vertebrae = [{
        "points": {
            "anteriorSuperior": {"x": 0.0, "y": 0.0},
            "middleSuperior": {"x": 4.0, "y": 0.0},
            "posteriorSuperior": {"x": 10.0, "y": 0.0},
            "posteriorInferior": {"x": 10.0, "y": 20.0},
            "middleInferior": {"x": 6.0, "y": 20.0},
            "anteriorInferior": {"x": 0.0, "y": 20.0},
        },
    }]

    inference._apply_aspect_fix(vertebrae, min_aspect=1.0, max_scale=2.0)

    points = vertebrae[0]["points"]
    assert points["middleSuperior"] == {"x": 3.0, "y": 0.0}
    assert points["middleInferior"] == {"x": 7.0, "y": 20.0}
