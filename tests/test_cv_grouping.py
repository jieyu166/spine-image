import json
import os

import pytest

import train_vertebra_model_cv as cv
from train_vertebra_model_cv import extract_group_id, manual_group_kfold


def test_default_group_id_comes_from_filename_not_date_directory():
    first_view = {"image_path": r"Images\20260101\80145593.png"}
    second_view = {"image_path": r"Images\20260202\80145593-2.png"}
    nhanes = {
        "image_path": r"Images\20260218NHANES\C00172_cervical_masks.png"
    }

    assert extract_group_id(first_view) == "80145593"
    assert extract_group_id(second_view) == "80145593"
    assert extract_group_id(nhanes) == "C00172"


def test_group_kfold_never_splits_patient_views_across_train_and_val():
    annotations = [
        {"image_path": r"Images\202607\80145593.png"},
        {"image_path": r"Images\202607\80294005a.png"},
        {"image_path": r"Images\202607\80294005b.png"},
        {"image_path": r"Images\202607\80655287.png"},
        {"image_path": r"Images\202607\80655287-2.png"},
        {"image_path": r"Images\20260218NHANES\C00172_cervical_masks.png"},
        {"image_path": r"Images\20260218NHANES\L12289_lumbar_masks_4.png"},
    ]

    folds = list(manual_group_kfold(annotations, n_splits=3, seed=42))

    all_val_indices = []
    for train_indices, val_indices in folds:
        train_groups = {extract_group_id(annotations[i]) for i in train_indices}
        val_groups = {extract_group_id(annotations[i]) for i in val_indices}
        assert train_groups.isdisjoint(val_groups)
        all_val_indices.extend(val_indices)

    assert sorted(all_val_indices) == list(range(len(annotations)))


def test_archive_fold_artifact_retries_a_transient_windows_lock(
    monkeypatch, tmp_path
):
    source = tmp_path / "best_vertebra_model.pth"
    destination = tmp_path / "best_vertebra_model_fold2.pth"
    source.write_bytes(b"new-fold-2")
    destination.write_bytes(b"partial-old-copy")
    real_replace = os.replace
    attempts = []

    def transient_lock(src, dst):
        attempts.append((src, dst))
        if len(attempts) == 1:
            raise PermissionError(32, "file is temporarily locked", str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(cv.os, "replace", transient_lock)

    cv.archive_fold_artifact(source, destination, attempts=3, delay_seconds=0)

    assert len(attempts) == 2
    assert not source.exists()
    assert destination.read_bytes() == b"new-fold-2"


def test_load_completed_fold_results_restores_history_for_resume(tmp_path):
    result_path = tmp_path / "cv_results.json"
    result_path.write_text(
        json.dumps({
            "n_folds_completed": 1,
            "n_folds_total": 5,
            "fold_best_val_losses": [0.98],
            "per_fold_history": [{
                "train_losses": [2.4, 1.1],
                "val_losses": [1.3, 0.98],
            }],
        }),
        encoding="utf-8",
    )

    restored = cv.load_completed_fold_results(result_path, expected_folds=5)

    assert restored == [{
        "best_val_loss": 0.98,
        "train_losses": [2.4, 1.1],
        "val_losses": [1.3, 0.98],
    }]


def test_load_completed_fold_results_rejects_inconsistent_history(tmp_path):
    result_path = tmp_path / "cv_results.json"
    result_path.write_text(
        json.dumps({
            "n_folds_completed": 2,
            "n_folds_total": 5,
            "fold_best_val_losses": [0.98],
            "per_fold_history": [{"train_losses": [], "val_losses": []}],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resume result history is inconsistent"):
        cv.load_completed_fold_results(result_path, expected_folds=5)
