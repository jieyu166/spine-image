from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from eval_oof import (
    EvalConfig,
    build_parser,
    config_from_args,
    render_worst_overlays,
    resolve_image_path,
    run_oof,
    run_production_reference,
    validate_config,
    write_reports,
)
from oof_metrics import compare_case_landmarks


def _valid_config(tmp_path):
    project_root = tmp_path / "project"
    run_dir = project_root / "runs" / "cv5"
    annotations_dir = project_root / "annotations"
    run_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    train_annotations = annotations_dir / "train.json"
    val_annotations = annotations_dir / "val.json"
    train_annotations.write_text("[]", encoding="utf-8")
    val_annotations.write_text("[]", encoding="utf-8")
    for fold_number in range(1, 6):
        (run_dir / f"best_vertebra_model_fold{fold_number}.pth").write_bytes(
            b"checkpoint"
        )

    production_model = project_root / "best_vertebra_model.pth"
    production_model.write_bytes(b"production")
    Path(f"{production_model}.ensemble").write_text(
        "best_vertebra_model_fold3.pth\n",
        encoding="utf-8",
    )

    return EvalConfig(
        project_root=project_root,
        run_dir=run_dir,
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        output_dir=run_dir / "oof_eval",
        production_model=production_model,
    )


def test_resolve_image_path_prefers_nested_annotation_path(tmp_path):
    nested = tmp_path / "Images" / "202607" / "80145593.png"
    fallback = tmp_path / "Images" / "80145593.png"
    nested.parent.mkdir(parents=True)
    fallback.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"nested")
    fallback.write_bytes(b"fallback")

    resolved = resolve_image_path(
        {"image_path": r"Images\202607\80145593.png"},
        tmp_path,
    )

    assert resolved == nested.resolve()


def test_resolve_image_path_falls_back_to_images_basename_and_extension(tmp_path):
    fallback = tmp_path / "Images" / "80145593.jpg"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"fallback")

    resolved = resolve_image_path(
        {"image_path": r"stale\location\80145593.dcm"},
        tmp_path,
    )

    assert resolved == fallback.resolve()


def test_resolve_image_path_raises_for_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError, match="cannot resolve image"):
        resolve_image_path(
            {"image_path": r"Images\202607\missing.png"},
            tmp_path,
        )


def test_validate_config_accepts_dedicated_output_below_run_dir(tmp_path):
    config = _valid_config(tmp_path)

    validate_config(config)


def test_validate_config_rejects_missing_fold_checkpoint(tmp_path):
    config = _valid_config(tmp_path)
    (config.run_dir / "best_vertebra_model_fold4.pth").unlink()

    with pytest.raises(FileNotFoundError, match="fold4"):
        validate_config(config)


def test_validate_config_rejects_checkpoint_as_output_path(tmp_path):
    config = _valid_config(tmp_path)
    dangerous = config.run_dir / "best_vertebra_model_fold1.pth"

    with pytest.raises(ValueError, match="overwrite protected model"):
        validate_config(replace(config, output_dir=dangerous))


def test_validate_config_rejects_output_outside_run_directory(tmp_path):
    config = _valid_config(tmp_path)

    with pytest.raises(ValueError, match="inside run directory"):
        validate_config(
            replace(config, output_dir=config.project_root / "comparison")
        )


def _vertebra():
    return {
        "name": "L1",
        "points": {
            "anteriorSuperior": {"x": 0, "y": 0},
            "middleSuperior": {"x": 5, "y": 0},
            "posteriorSuperior": {"x": 10, "y": 0},
            "posteriorInferior": {"x": 10, "y": 10},
            "middleInferior": {"x": 5, "y": 10},
            "anteriorInferior": {"x": 0, "y": 10},
        },
    }


def test_run_oof_predicts_each_case_once_with_held_out_fold_and_keeps_failure(
    tmp_path,
):
    config = replace(_valid_config(tmp_path), n_folds=3, device="cpu")
    annotations = [
        {
            "image_path": r"Images\202607\80145593.png",
            "spine_type": "L",
            "vertebrae": [_vertebra()],
        },
        {
            "image_path": r"Images\202607\80145593-2.png",
            "spine_type": "L",
            "vertebrae": [_vertebra()],
        },
        {
            "image_path": r"Images\202607\80294005.png",
            "spine_type": "L",
            "vertebrae": [_vertebra()],
        },
        {
            "image_path": r"Images\202607\80655287.png",
            "spine_type": "L",
            "vertebrae": [_vertebra()],
        },
    ]
    config.train_annotations.write_text(
        json.dumps(annotations[:3]),
        encoding="utf-8",
    )
    config.val_annotations.write_text(
        json.dumps(annotations[3:]),
        encoding="utf-8",
    )
    for annotation in annotations:
        image_path = config.project_root / annotation["image_path"].replace(
            "\\", "/"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")

    factory_calls = []
    prediction_calls = []

    class FakeInference:
        points_per_vertebra = 6

        def __init__(self, model_path, **kwargs):
            self.model_path = Path(model_path)
            self.tta = None
            factory_calls.append((self.model_path, kwargs))

        def predict(self, image_path, spine_type, confidence_threshold):
            image_path = Path(image_path)
            prediction_calls.append(image_path.stem)
            if image_path.stem == "80655287":
                raise RuntimeError("synthetic inference failure")
            return {
                "vertebrae": [_vertebra()],
                "predicted_count": 1,
                "count_confidence": 0.9,
                "original_image": np.zeros((32, 32, 3), dtype=np.uint8),
            }

    result = run_oof(config, inference_factory=FakeInference)

    cases = result["cases"]
    assert sorted(case["annotation_index"] for case in cases) == [0, 1, 2, 3]
    assert prediction_calls == [
        Path(case["image_path"]).stem for case in cases
    ]
    assert {case["fold"] for case in cases} == {1, 2, 3}
    assert [case["status"] for case in cases].count("failed") == 1
    failed = next(case for case in cases if case["status"] == "failed")
    assert "synthetic inference failure" in failed["error"]
    assert all(call[1]["ensemble_paths"] == [] for call in factory_calls)
    assert all(call[1]["device"] == "cpu" for call in factory_calls)


def _four_point_vertebra():
    vertebra = _vertebra()
    del vertebra["points"]["middleSuperior"]
    del vertebra["points"]["middleInferior"]
    return vertebra


def _successful_case(image_path, prediction=None):
    prediction = prediction or _vertebra()
    return {
        "annotation_index": 0,
        "fold": 1,
        "case_id": Path(image_path).stem,
        "image_path": str(image_path),
        "spine_type": "L",
        "status": "success",
        "ground_truth": {"vertebrae": [_vertebra()]},
        "prediction": {
            "predicted_count": 1,
            "count_confidence": 0.9,
            "vertebrae": [prediction],
        },
        "metrics": compare_case_landmarks([_vertebra()], [prediction]),
    }


def test_run_production_reference_preserves_sidecar_and_four_point_layout(
    tmp_path,
):
    config = replace(_valid_config(tmp_path), device="cpu")
    image_path = config.project_root / "Images" / "80145593.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    oof_result = {"cases": [_successful_case(image_path)]}
    factory_calls = []

    class FakeProductionInference:
        points_per_vertebra = 4

        def __init__(self, model_path, **kwargs):
            factory_calls.append((Path(model_path), kwargs))
            self.tta = None

        def predict(self, image_path, spine_type, confidence_threshold):
            return {
                "vertebrae": [_four_point_vertebra()],
                "predicted_count": 1,
                "count_confidence": 0.8,
                "original_image": np.zeros((32, 32, 3), dtype=np.uint8),
            }

    cases = run_production_reference(
        config,
        oof_result,
        inference_factory=FakeProductionInference,
    )

    assert len(cases) == 1
    assert factory_calls == [(config.production_model, {"device": "cpu"})]
    assert cases[0]["metrics"]["groups"]["middle"]["n"] == 0
    assert cases[0]["metrics"]["missing_predicted_landmarks"] == 2


def test_write_reports_creates_parseable_outputs_and_labels_reference(tmp_path):
    config = _valid_config(tmp_path)
    image_path = config.project_root / "Images" / "80145593.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    oof_cases = [
        _successful_case(image_path),
        {
            "annotation_index": 1,
            "fold": 2,
            "case_id": "failed",
            "image_path": str(image_path),
            "spine_type": "L",
            "status": "failed",
            "ground_truth": {"vertebrae": [_vertebra()]},
            "error": "synthetic failure",
        },
    ]
    production_cases = [
        _successful_case(image_path, prediction=_four_point_vertebra())
    ]

    paths = write_reports(config, oof_cases, production_cases)

    oof_metrics = json.loads(paths["oof_metrics"].read_text(encoding="utf-8"))
    production_metrics = json.loads(
        paths["production_reference_metrics"].read_text(encoding="utf-8")
    )
    predictions = json.loads(
        paths["oof_predictions"].read_text(encoding="utf-8")
    )
    csv_text = paths["oof_case_metrics"].read_text(encoding="utf-8-sig")
    comparison = paths["comparison"].read_text(encoding="utf-8")
    assert oof_metrics["n_cases"] == 2
    assert production_metrics["groups"]["middle"]["n"] == 0
    assert len(predictions["cases"]) == 2
    assert "success" in csv_text and "failed" in csv_text
    assert "non-OOF reference" in comparison
    assert "training-exposed" in comparison


def test_comparison_table_uses_common_corners_not_new_model_all_points(tmp_path):
    config = _valid_config(tmp_path)
    image_path = config.project_root / "Images" / "80145593.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    middle_bad = _vertebra()
    middle_bad["points"]["middleSuperior"]["y"] = 60
    middle_bad["points"]["middleInferior"]["y"] = 70
    oof_cases = [_successful_case(image_path, prediction=middle_bad)]
    production_cases = [
        _successful_case(image_path, prediction=_four_point_vertebra())
    ]

    paths = write_reports(config, oof_cases, production_cases)

    comparison = paths["comparison"].read_text(encoding="utf-8")
    new_row = next(
        line
        for line in comparison.splitlines()
        if line.startswith("| New six-point OOF |")
    )
    assert "Corner mean px" in comparison
    assert "| New six-point OOF | 1 | 0 | 0.00 |" in new_row
    assert "Middle-only OOF" in comparison


def test_render_worst_overlays_writes_requested_pngs_and_manifest(tmp_path):
    config = replace(_valid_config(tmp_path), worst_count=2)
    first_image = config.project_root / "Images" / "80145593.png"
    second_image = config.project_root / "Images" / "80294005.png"
    first_image.parent.mkdir(parents=True)
    for image_path in (first_image, second_image):
        ok, encoded = cv2.imencode(
            ".png",
            np.zeros((64, 64, 3), dtype=np.uint8),
        )
        assert ok
        image_path.write_bytes(encoded.tobytes())

    successful = _successful_case(first_image)
    failed = {
        "annotation_index": 1,
        "fold": 2,
        "case_id": second_image.stem,
        "image_path": str(second_image),
        "spine_type": "L",
        "status": "failed",
        "ground_truth": {"vertebrae": [_vertebra()]},
        "error": "synthetic failure",
    }

    manifest_path = render_worst_overlays(config, [successful, failed])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pngs = sorted(manifest_path.parent.glob("*.png"))
    assert len(manifest["cases"]) == 2
    assert len(pngs) == 2
    assert all(cv2.imread(str(path)) is not None for path in pngs)


def test_cli_builds_paths_and_supports_all_evaluation_overrides(tmp_path):
    project_root = tmp_path / "project"
    run_dir = project_root / "runs" / "cv5"
    parser = build_parser()

    args = parser.parse_args(
        [
            "--project-root",
            str(project_root),
            "--run-dir",
            str(run_dir),
            "--train-annotations",
            "annotations/train.json",
            "--val-annotations",
            "annotations/val.json",
            "--n-folds",
            "3",
            "--seed",
            "7",
            "--group-regex",
            r"(\d{8})",
            "--device",
            "cpu",
            "--threshold",
            "0.3",
            "--no-tta",
            "--worst-count",
            "4",
            "--output-dir",
            "review",
            "--production-model",
            "weights/production.pth",
            "--skip-production-reference",
        ]
    )

    config = config_from_args(args)

    assert config.project_root == project_root.resolve()
    assert config.run_dir == run_dir.resolve()
    assert config.train_annotations == (
        project_root / "annotations" / "train.json"
    ).resolve()
    assert config.val_annotations == (
        project_root / "annotations" / "val.json"
    ).resolve()
    assert config.output_dir == (run_dir / "review").resolve()
    assert config.production_model == (
        project_root / "weights" / "production.pth"
    ).resolve()
    assert config.n_folds == 3
    assert config.seed == 7
    assert config.group_regex == r"(\d{8})"
    assert config.device == "cpu"
    assert config.threshold == pytest.approx(0.3)
    assert config.tta is False
    assert config.worst_count == 4
    assert config.skip_production_reference is True
