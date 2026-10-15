import json

import pytest

from learning_curve import fraction_dirname, main, nested_group_subsets
from train_vertebra_model_cv import build_group_folds, extract_group_id, stratum_label


def _cohort():
    anns = []
    for i in range(16):
        anns.append({"image_path": f"Images/{80000000 + i}.png", "spine_type": "L",
                     "caseFlags": {"hasImplant": i < 4}})
    anns.append({"image_path": "Images/80000005-2.png", "spine_type": "L", "caseFlags": {"hasImplant": False}})
    for i in range(4):
        anns.append({"image_path": f"Images/{50000000 + i}C.png", "spine_type": "C"})
    return anns


def test_subsets_are_nested_group_intact_and_full_at_100():
    anns = _cohort()
    train = list(range(len(anns)))
    subsets = nested_group_subsets(anns, train, [0.25, 0.5, 0.75, 1.0], seed=1)
    assert subsets[1.0] == train
    assert set(subsets[0.25]) <= set(subsets[0.5]) <= set(subsets[0.75]) <= set(subsets[1.0])
    assert len(subsets[0.25]) < len(subsets[0.5]) < len(subsets[0.75]) < len(subsets[1.0])
    for indices in subsets.values():
        chosen = {extract_group_id(anns[i]) for i in indices}
        # 同病人兩張影像（80000005 與 80000005-2）不會被拆開
        assert ({5, 16} <= set(indices)) or ("80000005" not in chosen)


def test_subsets_are_stratified_and_keep_at_least_one_group_per_stratum():
    anns = _cohort()
    subsets = nested_group_subsets(anns, list(range(len(anns))), [0.25], seed=1)
    labels = [stratum_label(anns[i]) for i in subsets[0.25]]
    # 12 腰椎群 → 3、4 植入物群 → 1、4 頸椎群 → 1
    assert labels.count("C") == 1
    assert labels.count("L_implant") == 1
    assert len({extract_group_id(anns[i]) for i in subsets[0.25] if stratum_label(anns[i]) == "L"}) == 3


def test_subsets_are_deterministic_and_reject_bad_fraction():
    anns = _cohort()
    train = list(range(len(anns)))
    assert nested_group_subsets(anns, train, [0.5], seed=3) == nested_group_subsets(anns, train, [0.5], seed=3)
    with pytest.raises(ValueError):
        nested_group_subsets(anns, train, [0.0])
    with pytest.raises(ValueError):
        nested_group_subsets(anns, train, [1.5])


def test_subsets_only_draw_from_the_fold_training_indices():
    anns = _cohort()
    folds = build_group_folds(anns, 4, seed=42, strategy="stratified")
    for train, val in folds:
        for indices in nested_group_subsets(anns, train, [0.25, 0.5, 1.0]).values():
            assert set(indices) <= set(train)
            assert not set(indices) & set(val)


def test_fraction_dirname():
    assert fraction_dirname(0.25) == "frac_025"
    assert fraction_dirname(1.0) == "frac_100"


def test_plan_only_writes_plan_without_training(tmp_path):
    anns = _cohort()
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()
    (ann_dir / "train_annotations.json").write_text(json.dumps(anns[:15]), encoding="utf-8")
    (ann_dir / "val_annotations.json").write_text(json.dumps(anns[15:]), encoding="utf-8")
    out = tmp_path / "lc"
    assert main(["--annotations-dir", str(ann_dir), "--out-dir", str(out), "--n-folds", "4",
                 "--fractions", "0.5", "1.0", "0.25", "--plan-only"]) == 0
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["fractions"] == [0.25, 0.5, 1.0]
    sizes = plan["train_images_per_fold"]
    assert all(a < b < c for a, b, c in zip(sizes["frac_025"], sizes["frac_050"], sizes["frac_100"]))
    assert not list(out.glob("frac_*"))


def test_log_survives_chdir_with_relative_out_dir(tmp_path, monkeypatch):
    anns = _cohort()
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir()
    (ann_dir / "train_annotations.json").write_text(json.dumps(anns[:15]), encoding="utf-8")
    (ann_dir / "val_annotations.json").write_text(json.dumps(anns[15:]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    import learning_curve

    def fake_train(fraction, annotations, folds, subsets_by_fold, out_dir, args, log):
        run_dir = out_dir / learning_curve.fraction_dirname(fraction)
        run_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(run_dir)  # 模擬訓練時切換工作目錄
        log("inside run dir")
        return run_dir

    monkeypatch.setattr(learning_curve, "train_fraction", fake_train)
    assert main(["--annotations-dir", "ann", "--out-dir", "lc", "--n-folds", "4",
                 "--fractions", "0.5", "--skip-eval"]) == 0
    assert "inside run dir" in (tmp_path / "lc" / "learning_curve.log").read_text(encoding="utf-8")
