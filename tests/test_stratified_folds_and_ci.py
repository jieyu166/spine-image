"""StratifiedGroupKFold 切分、病人群集 bootstrap 信賴區間、eval_oof 串接與配對比較 CLI。"""
import json
from dataclasses import replace

import pytest

import train_vertebra_model_cv as cv
from compare_oof_runs import compare, main as compare_main, to_markdown
from eval_oof import build_parser, config_from_args, format_ci_markdown, resolve_fold_strategy
from oof_metrics import bootstrap_ci, case_group_id, paired_bootstrap_delta, summarize_cases
from train_vertebra_model_cv import DEFAULT_GROUP_REGEX, extract_group_id, manual_group_kfold
from test_eval_oof import _valid_config

CORNERS = ("anteriorSuperior", "posteriorSuperior", "posteriorInferior", "anteriorInferior")


# ── 切分 ────────────────────────────────────────────────────────────────
def _stratified_cohort():
    """12 個病人群：8 腰椎（含植入物）＋ 4 頸椎，其中一位病人有兩張影像。"""
    annotations = []
    for i in range(8):
        annotations.append({"image_path": f"Images/{80000000 + i}.png", "spine_type": "L",
                            "caseFlags": {"hasImplant": i < 2}})
    annotations.append({"image_path": "Images/80000000-2.png", "spine_type": "L",
                        "caseFlags": {"hasImplant": True}})
    for i in range(4):
        annotations.append({"image_path": f"Images/{50000000 + i}C.png", "spine_type": "C"})
    return annotations


def test_stratum_label_combines_spine_type_and_implant_flag():
    assert cv.stratum_label({"spine_type": "L"}) == "L"
    assert cv.stratum_label({"spineType": "C", "caseFlags": {"hasImplant": True}}) == "C_implant"
    assert cv.stratum_label({"caseFlags": "broken"}) == "L"


def test_stratified_group_kfold_keeps_patients_together_and_spreads_cervical_cases():
    annotations = _stratified_cohort()
    folds = cv.build_group_folds(annotations, n_splits=4, seed=42, strategy="stratified")

    seen = []
    for train_indices, val_indices in folds:
        train_groups = {extract_group_id(annotations[i]) for i in train_indices}
        val_groups = {extract_group_id(annotations[i]) for i in val_indices}
        assert train_groups.isdisjoint(val_groups)
        assert sorted(train_indices + val_indices) == list(range(len(annotations)))
        seen.extend(val_indices)
    assert sorted(seen) == list(range(len(annotations)))
    # 4 個頸椎病人平均分到 4 個 fold
    assert [c.get("C", 0) for c in cv.fold_composition(annotations, folds)] == [1, 1, 1, 1]


def test_stratified_group_kfold_is_deterministic_and_seed_sensitive():
    annotations = _stratified_cohort()
    first = cv.build_group_folds(annotations, 4, seed=42, strategy="stratified")
    assert first == cv.build_group_folds(annotations, 4, seed=42, strategy="stratified")
    assert first != cv.build_group_folds(annotations, 4, seed=7, strategy="stratified")


def test_build_group_folds_keeps_legacy_greedy_and_rejects_bad_input():
    annotations = _stratified_cohort()
    assert cv.build_group_folds(annotations, 3, seed=42, strategy="greedy") == list(
        manual_group_kfold(annotations, 3, seed=42)
    )
    with pytest.raises(ValueError):
        cv.build_group_folds(annotations, 3, strategy="random")
    with pytest.raises(ValueError):
        cv.build_group_folds(annotations[:2], 3, strategy="stratified")


def test_train_cli_defaults_to_stratified(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_vertebra_model_cv.py"])
    assert cv.parse_args().fold_strategy == "stratified"
    monkeypatch.setattr("sys.argv", ["train_vertebra_model_cv.py", "--fold-strategy", "greedy"])
    assert cv.parse_args().fold_strategy == "greedy"


# ── 信賴區間 ────────────────────────────────────────────────────────────
def _case(case_id, image_path, corner_px, exact=True, status="success", spine="L"):
    if status != "success":
        return {"case_id": case_id, "image_path": image_path, "status": status, "spine_type": spine}
    records = [{"landmark": n, "vertebra": "L1", "distance_px": float(corner_px)} for n in CORNERS]
    records.append({"landmark": "middleSuperior", "vertebra": "L1", "distance_px": float(corner_px) / 2})
    return {
        "case_id": case_id, "image_path": image_path, "status": "success", "spine_type": spine,
        "metrics": {"distance_records": records, "count_exact": exact, "absolute_count_error": 0 if exact else 1},
    }


def _cohort(offset=0.0):
    cases = [
        _case(f"8000000{i}", f"C:/x/Images/8000000{i}.png", 50 + 10 * i + offset, exact=(i % 4 != 0))
        for i in range(8)
    ]
    cases.append(_case("80000001-2", "C:/x/Images/80000001-2.png", 70 + offset))
    return cases


def test_bootstrap_ci_estimates_match_summarize_cases_and_bracket_estimate():
    cases = _cohort() + [_case("80000009", "C:/x/Images/80000009.png", 0, status="failed")]
    ci = bootstrap_ci(cases, DEFAULT_GROUP_REGEX, n_boot=500, seed=1)
    summary = summarize_cases(cases)

    assert ci["n_cases"] == 10
    assert ci["n_groups"] == 9  # 80000001 與 80000001-2 是同一病人
    m = ci["metrics"]
    assert m["count_exact_match_rate"]["estimate"] == pytest.approx(summary["count_exact_match_rate"])
    assert m["corner_mean_px"]["estimate"] == pytest.approx(summary["groups"]["corners"]["mean_distance_px"])
    assert m["corner_p90_px"]["estimate"] == pytest.approx(summary["groups"]["corners"]["p90_distance_px"])
    assert m["middle_mean_px"]["estimate"] == pytest.approx(summary["groups"]["middle"]["mean_distance_px"])
    for item in m.values():
        assert item["ci_low"] <= item["estimate"] <= item["ci_high"]
    assert bootstrap_ci(cases, DEFAULT_GROUP_REGEX, n_boot=500, seed=1) == ci  # 可重現


def test_case_group_id_falls_back_to_case_id_when_regex_does_not_match():
    assert case_group_id({"case_id": "x", "image_path": "Images/80000001-2.png"}, DEFAULT_GROUP_REGEX) == "80000001"
    assert case_group_id({"case_id": "weird", "image_path": "Images/abc.png"}, DEFAULT_GROUP_REGEX) == "case:weird"


def test_paired_delta_is_zero_for_identical_runs_and_detects_uniform_shift():
    same = paired_bootstrap_delta(_cohort(), _cohort(), DEFAULT_GROUP_REGEX, n_boot=300)
    assert same["n_common_cases"] == 9
    assert same["metrics"]["corner_mean_px"]["delta"] == 0
    assert same["metrics"]["corner_mean_px"]["excludes_zero"] is False

    extra = _case("99999999", "Images/99999999.png", 1)
    worse = paired_bootstrap_delta(_cohort(), _cohort(offset=20) + [extra], DEFAULT_GROUP_REGEX, n_boot=300)
    corner = worse["metrics"]["corner_mean_px"]
    assert corner["delta"] == pytest.approx(20)
    assert corner["ci_low"] == pytest.approx(20)
    assert corner["excludes_zero"] is True
    assert worse["only_in_candidate"] == ["99999999"]


# ── eval_oof 串接 ───────────────────────────────────────────────────────
def test_resolve_fold_strategy_defaults_to_legacy_greedy_and_reads_cv_results(tmp_path):
    config = _valid_config(tmp_path)
    assert resolve_fold_strategy(config) == ("greedy", None)

    members = [["Images/a.png"], ["Images/b.png"]]
    (config.run_dir / "cv_results.json").write_text(
        json.dumps({"fold_strategy": "stratified", "fold_val_members": members}), encoding="utf-8"
    )
    assert resolve_fold_strategy(config) == ("stratified", members)
    assert resolve_fold_strategy(replace(config, fold_strategy="greedy")) == ("greedy", None)

    (config.run_dir / "cv_results.json").write_text(json.dumps({"fold_strategy": "sample-level"}), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_fold_strategy(config)


def test_eval_cli_passes_fold_strategy_and_bootstrap(tmp_path):
    config = config_from_args(build_parser().parse_args(
        ["--run-dir", str(tmp_path), "--fold-strategy", "stratified", "--bootstrap", "0"]))
    assert config.fold_strategy == "stratified"
    assert config.bootstrap == 0
    defaults = config_from_args(build_parser().parse_args(["--run-dir", str(tmp_path)]))
    assert defaults.fold_strategy is None and defaults.bootstrap == 2000


def test_format_ci_markdown_renders_rates_and_pixels():
    ci = {
        "method": "patient-group cluster bootstrap (percentile)", "n_boot": 10, "n_groups": 3,
        "metrics": {
            "count_exact_match_rate": {"estimate": 0.85, "ci_low": 0.8, "ci_high": 0.9},
            "corner_mean_px": {"estimate": 120.0, "ci_low": 110.0, "ci_high": 131.5},
            "middle_mean_px": {"estimate": None, "ci_low": None, "ci_high": None},
        },
    }
    text = "\n".join(format_ci_markdown(ci))
    assert "| Count exact | 85.0% | 80.0% – 90.0% |" in text
    assert "| Corner mean px | 120.0 | 110.0 – 131.5 |" in text
    assert "Middle" not in text
    assert format_ci_markdown(None) == []


# ── compare_oof_runs CLI ────────────────────────────────────────────────
def _run(offset):
    cases = [_case(f"8000000{i}", f"Images/8000000{i}.png", 50 + 5 * i + offset) for i in range(6)]
    cases.append(_case("5000000C", "Images/5000000C.png", 90 + offset, spine="C"))
    cases.append(_case("C00160_cervical_masks_4", "Images/20260218NHANES/C00160_cervical_masks_4.png",
                       60 + offset, spine="C"))
    return cases


def test_compare_reports_subgroups_and_marks_significant_shift():
    results = compare(_run(0), _run(30), n_boot=200)
    assert set(results) == {"全部", "腰椎", "頸椎", "醫院影像（非 NHANES）"}
    assert results["腰椎"]["n_common_cases"] == 6
    assert results["醫院影像（非 NHANES）"]["n_common_cases"] == 7
    corner = results["全部"]["metrics"]["corner_mean_px"]
    assert corner["delta"] == 30 and corner["excludes_zero"]
    text = to_markdown(results, "old", "new")
    assert "# OOF 配對比較：new vs old" in text
    assert "| Corner mean px |" in text and "✔" in text


def test_compare_cli_writes_markdown_and_json(tmp_path):
    for name in ("A", "B"):
        d = tmp_path / name / "oof_eval"
        d.mkdir(parents=True)
        (d / "oof_predictions.json").write_text(json.dumps({"cases": _run(0)}), encoding="utf-8")
    out = tmp_path / "cmp.md"
    assert compare_main(["--baseline", str(tmp_path / "A/oof_eval/oof_predictions.json"),
                         "--candidate", str(tmp_path / "B/oof_eval/oof_predictions.json"),
                         "--output", str(out), "--bootstrap", "50"]) == 0
    assert "B vs A" in out.read_text(encoding="utf-8")
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["全部"]["metrics"]["corner_mean_px"]["delta"] == 0
