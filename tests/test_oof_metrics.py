import pytest

from oof_metrics import (
    compare_case_landmarks,
    rank_worst_cases,
    summarize_cases,
    summarize_distances,
    validate_fold_assignments,
)
from train_vertebra_model_cv import DEFAULT_GROUP_REGEX


def _annotations():
    return [
        {"image_path": r"Images\202607\80145593.png"},
        {"image_path": r"Images\202607\80145593-2.png"},
        {"image_path": r"Images\202607\80294005.png"},
        {"image_path": r"Images\202607\80655287.png"},
    ]


def test_validate_fold_assignments_maps_every_sample_once():
    folds = [
        ([2, 3], [0, 1]),
        ([0, 1, 3], [2]),
        ([0, 1, 2], [3]),
    ]

    assert validate_fold_assignments(
        _annotations(), folds, DEFAULT_GROUP_REGEX
    ) == {0: 1, 1: 1, 2: 2, 3: 3}


def test_validate_fold_assignments_rejects_duplicate_validation_assignment():
    folds = [
        ([2, 3], [0, 1]),
        ([0, 1, 3], [2]),
        ([0, 1], [2, 3]),
    ]

    with pytest.raises(ValueError, match="multiple validation folds"):
        validate_fold_assignments(_annotations(), folds, DEFAULT_GROUP_REGEX)


def test_validate_fold_assignments_rejects_missing_validation_assignment():
    folds = [
        ([2, 3], [0, 1]),
        ([0, 1, 3], [2]),
    ]

    with pytest.raises(ValueError, match="missing validation assignment"):
        validate_fold_assignments(_annotations(), folds, DEFAULT_GROUP_REGEX)


def test_validate_fold_assignments_rejects_patient_group_leakage():
    folds = [
        ([0, 2, 3], [1]),
        ([0, 1, 3], [2]),
        ([0, 1, 2], [3]),
        ([1, 2, 3], [0]),
    ]

    with pytest.raises(ValueError, match="patient-group leakage"):
        validate_fold_assignments(_annotations(), folds, DEFAULT_GROUP_REGEX)


def _six_point_vertebra(middle_superior_y=0, middle_inferior_y=10):
    return {
        "name": "L1",
        "points": {
            "anteriorSuperior": {"x": 0, "y": 0},
            "middleSuperior": {"x": 0, "y": middle_superior_y},
            "posteriorSuperior": {"x": 10, "y": 0},
            "posteriorInferior": {"x": 10, "y": 10},
            "middleInferior": {"x": 0, "y": middle_inferior_y},
            "anteriorInferior": {"x": 0, "y": 10},
        },
    }


def test_compare_case_landmarks_separates_middle_from_corner_error():
    gt = [_six_point_vertebra()]
    pred = [_six_point_vertebra(middle_superior_y=3, middle_inferior_y=14)]

    metrics = compare_case_landmarks(gt, pred)

    assert metrics["groups"]["all"]["mean_distance_px"] == pytest.approx(7 / 6)
    assert metrics["groups"]["corners"]["mean_distance_px"] == 0.0
    assert metrics["groups"]["middle"]["mean_distance_px"] == 3.5
    assert metrics["count_exact"] is True
    assert metrics["absolute_count_error"] == 0


def test_compare_case_landmarks_counts_missing_prediction_without_zero_error():
    gt = [_six_point_vertebra()]
    pred_vertebra = _six_point_vertebra()
    del pred_vertebra["points"]["middleInferior"]

    metrics = compare_case_landmarks(gt, [pred_vertebra])

    assert metrics["missing_predicted_landmarks"] == 1
    assert metrics["groups"]["all"]["n"] == 5
    assert all(
        row["landmark"] != "middleInferior"
        for row in metrics["distance_records"]
    )


def test_summarize_distances_calculates_literal_percentiles_and_thresholds():
    summary = summarize_distances([0, 10, 20, 30, 100])

    assert summary["n"] == 5
    assert summary["mean_distance_px"] == pytest.approx(32.0)
    assert summary["median_distance_px"] == pytest.approx(20.0)
    assert summary["p90_distance_px"] == pytest.approx(72.0)
    assert summary["p95_distance_px"] == pytest.approx(86.0)
    assert summary["max_distance_px"] == pytest.approx(100.0)
    assert summary["n_below_50_px"] == 4
    assert summary["n_below_100_px"] == 4


def test_four_point_prediction_does_not_fabricate_middle_landmarks():
    gt = [_six_point_vertebra()]
    pred = [_six_point_vertebra()]
    del pred[0]["points"]["middleSuperior"]
    del pred[0]["points"]["middleInferior"]

    metrics = compare_case_landmarks(gt, pred)

    assert metrics["groups"]["middle"]["n"] == 0
    assert metrics["missing_predicted_landmarks"] == 2


def test_summarize_cases_retains_failed_cases_in_denominator():
    successful_metrics = compare_case_landmarks(
        [_six_point_vertebra()],
        [_six_point_vertebra()],
    )
    cases = [
        {"status": "success", "metrics": successful_metrics},
        {"status": "failed", "error": "CUDA failure"},
    ]

    summary = summarize_cases(cases)

    assert summary["n_cases"] == 2
    assert summary["n_successful_predictions"] == 1
    assert summary["n_failed_predictions"] == 1
    assert summary["count_exact_match_rate"] == 0.5


def test_rank_worst_cases_puts_failures_first_then_largest_errors():
    cases = [
        {
            "case_id": "small",
            "status": "success",
            "metrics": {
                "groups": {
                    "all": {
                        "max_distance_px": 20.0,
                        "mean_distance_px": 10.0,
                    }
                }
            },
        },
        {"case_id": "failed", "status": "failed", "error": "bad image"},
        {
            "case_id": "large",
            "status": "success",
            "metrics": {
                "groups": {
                    "all": {
                        "max_distance_px": 90.0,
                        "mean_distance_px": 40.0,
                    }
                }
            },
        },
    ]

    ranked = rank_worst_cases(cases, limit=3)

    assert [case["case_id"] for case in ranked] == ["failed", "large", "small"]
