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


# ── 信賴區間：病人群集 bootstrap ─────────────────────────────────────────
# 同一病人的多張影像（例如 18971571-1/-2、50892224C/L）不是獨立樣本，
# 所以重抽單位是「病人群」而不是影像；指標計算方式與 summarize_cases 一致
# （角點/中點是 landmark 合併後的分布，count 正確率的分母含失敗案例）。

CI_METRICS = (
    "count_exact_match_rate",
    "corner_mean_px",
    "corner_median_px",
    "corner_p90_px",
    "middle_mean_px",
)


def _case_payload(case):
    metrics = case.get("metrics") if case.get("status") == "success" else None
    records = (metrics or {}).get("distance_records", []) if isinstance(metrics, dict) else []
    corners = np.asarray(
        [r["distance_px"] for r in records if r.get("landmark") in CORNER_NAMES_4], dtype=float
    )
    middle = np.asarray(
        [r["distance_px"] for r in records if r.get("landmark") in MIDDLE_NAMES_2], dtype=float
    )
    exact = bool(isinstance(metrics, dict) and metrics.get("count_exact"))
    return corners, middle, exact


def _metrics_from_payloads(payloads):
    n = len(payloads)
    corners = np.concatenate([p[0] for p in payloads]) if payloads else np.empty(0)
    middle = np.concatenate([p[1] for p in payloads]) if payloads else np.empty(0)
    return {
        "count_exact_match_rate": (sum(p[2] for p in payloads) / n) if n else float("nan"),
        "corner_mean_px": float(np.mean(corners)) if corners.size else float("nan"),
        "corner_median_px": float(np.median(corners)) if corners.size else float("nan"),
        "corner_p90_px": float(np.percentile(corners, 90)) if corners.size else float("nan"),
        "middle_mean_px": float(np.mean(middle)) if middle.size else float("nan"),
    }


def case_group_id(case, group_regex):
    """OOF case 的病人群 ID；regex 對不到時退回 case_id（視為獨立病人）。"""
    try:
        return extract_group_id({"image_path": case.get("image_path", "")}, group_regex)
    except ValueError:
        return f"case:{case.get('case_id')}"


def _group_index(keys):
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _interval(samples, alpha):
    data = np.asarray(samples, dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return None, None
    return (
        float(np.percentile(data, 100 * alpha / 2)),
        float(np.percentile(data, 100 * (1 - alpha / 2))),
    )


def _clean(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def bootstrap_ci(cases, group_regex, n_boot=2000, seed=0, alpha=0.05):
    """OOF 指標的 percentile 信賴區間，以病人群為單位重抽。"""

    payloads = [_case_payload(case) for case in cases]
    groups = _group_index([case_group_id(case, group_regex) for case in cases])
    estimate = _metrics_from_payloads(payloads)
    samples = {key: [] for key in CI_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        picks = rng.integers(0, len(groups), len(groups)) if groups else []
        resampled = [payloads[i] for g in picks for i in groups[g]]
        for key, value in _metrics_from_payloads(resampled).items():
            samples[key].append(value)
    result = {}
    for key in CI_METRICS:
        low, high = _interval(samples[key], alpha) if n_boot else (None, None)
        result[key] = {"estimate": _clean(estimate[key]), "ci_low": low, "ci_high": high}
    return {
        "method": "patient-group cluster bootstrap (percentile)",
        "confidence": 1 - alpha,
        "n_boot": int(n_boot),
        "n_cases": len(cases),
        "n_groups": len(groups),
        "seed": seed,
        "metrics": result,
    }


def paired_bootstrap_delta(baseline_cases, candidate_cases, group_regex,
                           n_boot=2000, seed=0, alpha=0.05):
    """兩輪 OOF 在「同一批案例」上的差值（candidate − baseline）與信賴區間。

    兩輪都是 OOF，每個案例各自都沒被訓練到；fold 組成不同也可比較，
    但只取兩邊都有的 case_id，並以病人群為單位同步重抽。
    """
    base = {case.get("case_id"): case for case in baseline_cases}
    cand = {case.get("case_id"): case for case in candidate_cases}
    common = sorted(set(base) & set(cand))
    base_payloads = [_case_payload(base[k]) for k in common]
    cand_payloads = [_case_payload(cand[k]) for k in common]
    groups = _group_index([case_group_id(cand[k], group_regex) for k in common])

    def delta(indices):
        b = _metrics_from_payloads([base_payloads[i] for i in indices])
        c = _metrics_from_payloads([cand_payloads[i] for i in indices])
        return {key: c[key] - b[key] for key in CI_METRICS}, b, c

    all_indices = list(range(len(common)))
    point, base_point, cand_point = delta(all_indices)
    samples = {key: [] for key in CI_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        picks = rng.integers(0, len(groups), len(groups)) if groups else []
        values, _, _ = delta([i for g in picks for i in groups[g]])
        for key, value in values.items():
            samples[key].append(value)
    metrics = {}
    for key in CI_METRICS:
        low, high = _interval(samples[key], alpha) if n_boot else (None, None)
        metrics[key] = {
            "baseline": _clean(base_point[key]),
            "candidate": _clean(cand_point[key]),
            "delta": _clean(point[key]),
            "ci_low": low,
            "ci_high": high,
            "excludes_zero": bool(low is not None and (low > 0 or high < 0)),
        }
    return {
        "method": "paired patient-group cluster bootstrap (percentile)",
        "confidence": 1 - alpha,
        "n_boot": int(n_boot),
        "n_common_cases": len(common),
        "n_groups": len(groups),
        "only_in_baseline": sorted(set(base) - set(cand)),
        "only_in_candidate": sorted(set(cand) - set(base)),
        "metrics": metrics,
    }
