"""Pure validation and metric helpers for leakage-safe OOF evaluation."""

import math

import numpy as np

from train_vertebra_model_cv import extract_group_id


POINT_NAMES_6 = (
    "anteriorSuperior",
    "middleSuperior",
    "posteriorSuperior",
    "posteriorInferior",
    "middleInferior",
    "anteriorInferior",
)
CORNER_NAMES_4 = (
    "anteriorSuperior",
    "posteriorSuperior",
    "posteriorInferior",
    "anteriorInferior",
)
MIDDLE_NAMES_2 = ("middleSuperior", "middleInferior")


def validate_fold_assignments(annotations, folds, group_regex):
    """Validate grouped folds and return annotation-index to fold-number mapping."""
    assignments = {}
    expected = set(range(len(annotations)))

    for fold_number, (train_indices, val_indices) in enumerate(folds, start=1):
        train_set = set(train_indices)
        val_set = set(val_indices)
        if train_set & val_set:
            raise ValueError(
                f"fold {fold_number} has train/validation index overlap"
            )

        train_groups = {
            extract_group_id(annotations[index], group_regex)
            for index in train_set
        }
        val_groups = {
            extract_group_id(annotations[index], group_regex)
            for index in val_set
        }
        if train_groups & val_groups:
            raise ValueError(
                f"fold {fold_number} has patient-group leakage"
            )

        for index in val_set:
            if index in assignments:
                raise ValueError(
                    f"annotation {index} appears in multiple validation folds"
                )
            assignments[index] = fold_number

    missing = expected - set(assignments)
    if missing:
        raise ValueError(
            f"annotations missing validation assignment: {sorted(missing)}"
        )

    return assignments


def summarize_distances(values):
    """Return deterministic pixel-distance statistics for a sequence."""
    if not values:
        return {
            "n": 0,
            "mean_distance_px": None,
            "median_distance_px": None,
            "p90_distance_px": None,
            "p95_distance_px": None,
            "max_distance_px": None,
            "n_below_50_px": 0,
            "n_below_100_px": 0,
        }

    data = np.asarray(values, dtype=float)
    return {
        "n": int(data.size),
        "mean_distance_px": float(np.mean(data)),
        "median_distance_px": float(np.median(data)),
        "p90_distance_px": float(np.percentile(data, 90)),
        "p95_distance_px": float(np.percentile(data, 95)),
        "max_distance_px": float(np.max(data)),
        "n_below_50_px": int(np.sum(data < 50)),
        "n_below_100_px": int(np.sum(data < 100)),
    }


def _named_vertebrae(vertebrae):
    return {
        vertebra["name"]: vertebra
        for vertebra in vertebrae or []
        if isinstance(vertebra, dict) and vertebra.get("name")
    }


def _point(vertebra, name):
    if not vertebra:
        return None
    points = vertebra.get("points", {})
    if not isinstance(points, dict):
        return None
    value = points.get(name)
    return value if isinstance(value, dict) else None


def _distance(first, second):
    dx = float(first.get("x", 0)) - float(second.get("x", 0))
    dy = float(first.get("y", 0)) - float(second.get("y", 0))
    return math.hypot(dx, dy)


def _summaries_by_name(records, key, names):
    return {
        name: summarize_distances(
            [
                record["distance_px"]
                for record in records
                if record[key] == name
            ]
        )
        for name in names
    }


def compare_case_landmarks(gt_vertebrae, pred_vertebrae):
    """Compare one case while preserving six-point layout and missingness."""
    gt_by_name = _named_vertebrae(gt_vertebrae)
    pred_by_name = _named_vertebrae(pred_vertebrae)
    vertebral_levels = sorted(set(gt_by_name) | set(pred_by_name))

    records = []
    missing_gt = 0
    missing_predicted = 0
    for level in vertebral_levels:
        gt = gt_by_name.get(level)
        pred = pred_by_name.get(level)
        for landmark in POINT_NAMES_6:
            gt_point = _point(gt, landmark)
            pred_point = _point(pred, landmark)
            if gt_point is not None and pred_point is not None:
                records.append(
                    {
                        "vertebra": level,
                        "landmark": landmark,
                        "distance_px": _distance(gt_point, pred_point),
                        "gt": {
                            "x": float(gt_point.get("x", 0)),
                            "y": float(gt_point.get("y", 0)),
                        },
                        "pred": {
                            "x": float(pred_point.get("x", 0)),
                            "y": float(pred_point.get("y", 0)),
                        },
                    }
                )
            elif gt_point is not None:
                missing_predicted += 1
            elif pred_point is not None:
                missing_gt += 1

    all_distances = [record["distance_px"] for record in records]
    corner_distances = [
        record["distance_px"]
        for record in records
        if record["landmark"] in CORNER_NAMES_4
    ]
    middle_distances = [
        record["distance_px"]
        for record in records
        if record["landmark"] in MIDDLE_NAMES_2
    ]
    gt_count = len(gt_by_name)
    pred_count = len(pred_by_name)

    return {
        "n_gt_vertebrae": gt_count,
        "n_pred_vertebrae": pred_count,
        "n_matched_vertebrae": len(set(gt_by_name) & set(pred_by_name)),
        "count_exact": gt_count == pred_count,
        "absolute_count_error": abs(gt_count - pred_count),
        "missing_gt_landmarks": missing_gt,
        "missing_predicted_landmarks": missing_predicted,
        "groups": {
            "all": summarize_distances(all_distances),
            "corners": summarize_distances(corner_distances),
            "middle": summarize_distances(middle_distances),
        },
        "per_landmark": _summaries_by_name(
            records, "landmark", POINT_NAMES_6
        ),
        "per_vertebral_level": _summaries_by_name(
            records, "vertebra", vertebral_levels
        ),
        "distance_records": records,
    }


def summarize_cases(cases):
    """Aggregate successful case metrics while retaining failure denominators."""
    successful = [
        case
        for case in cases
        if case.get("status") == "success" and isinstance(case.get("metrics"), dict)
    ]
    records = [
        record
        for case in successful
        for record in case["metrics"].get("distance_records", [])
    ]
    levels = sorted({record["vertebra"] for record in records})
    count_exact_matches = sum(
        bool(case["metrics"].get("count_exact")) for case in successful
    )
    absolute_count_errors = [
        case["metrics"]["absolute_count_error"]
        for case in successful
        if case["metrics"].get("absolute_count_error") is not None
    ]
    n_cases = len(cases)

    return {
        "n_cases": n_cases,
        "n_successful_predictions": len(successful),
        "n_failed_predictions": n_cases - len(successful),
        "count_exact_matches": count_exact_matches,
        "count_exact_match_rate": (
            count_exact_matches / n_cases if n_cases else None
        ),
        "mean_absolute_count_error": (
            float(np.mean(absolute_count_errors))
            if absolute_count_errors
            else None
        ),
        "missing_gt_landmarks": sum(
            case["metrics"].get("missing_gt_landmarks", 0)
            for case in successful
        ),
        "missing_predicted_landmarks": sum(
            case["metrics"].get("missing_predicted_landmarks", 0)
            for case in successful
        ),
        "groups": {
            "all": summarize_distances(
                [record["distance_px"] for record in records]
            ),
            "corners": summarize_distances(
                [
                    record["distance_px"]
                    for record in records
                    if record["landmark"] in CORNER_NAMES_4
                ]
            ),
            "middle": summarize_distances(
                [
                    record["distance_px"]
                    for record in records
                    if record["landmark"] in MIDDLE_NAMES_2
                ]
            ),
        },
        "per_landmark": _summaries_by_name(
            records, "landmark", POINT_NAMES_6
        ),
        "per_vertebral_level": _summaries_by_name(
            records, "vertebra", levels
        ),
    }


def rank_worst_cases(cases, limit=10):
    """Rank explicit failures first, then successful cases by largest error."""

    def sort_key(case):
        if case.get("status") != "success":
            return (0, 0.0, 0.0)
        summary = case.get("metrics", {}).get("groups", {}).get("all", {})
        maximum = summary.get("max_distance_px")
        mean = summary.get("mean_distance_px")
        return (
            1,
            -float(maximum if maximum is not None else -1),
            -float(mean if mean is not None else -1),
        )

    return sorted(cases, key=sort_key)[: max(0, int(limit))]
