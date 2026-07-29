import json
from pathlib import Path

import pytest

import regenerate_splits as splits


def write_annotation(path: Path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vertebrae": [], **extra}), encoding="utf-8")


def configure_scan(monkeypatch, root: Path):
    images_dir = root / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(splits, "ROOT", root)
    monkeypatch.setattr(splits, "IMAGES_DIR", images_dir)
    return images_dir


def test_find_paired_samples_scans_images_recursively(monkeypatch, tmp_path):
    images_dir = configure_scan(monkeypatch, tmp_path)
    nested = images_dir / "202607"
    write_annotation(nested / "case.json")
    (nested / "case.png").write_bytes(b"image")

    pairs = splits.find_paired_samples()

    assert pairs == [("Images/202607/case.png", nested / "case.json")]


def test_find_paired_samples_only_pairs_within_same_directory(monkeypatch, tmp_path):
    images_dir = configure_scan(monkeypatch, tmp_path)
    nested = images_dir / "nested"
    write_annotation(nested / "case.json")
    (images_dir / "case.png").write_bytes(b"wrong-directory")

    assert splits.find_paired_samples() == []


def test_find_paired_samples_excludes_samp_annotations(monkeypatch, tmp_path):
    images_dir = configure_scan(monkeypatch, tmp_path)
    write_annotation(images_dir / "casesamp.json")
    (images_dir / "casesamp.png").write_bytes(b"overlay")

    pairs, summary = splits.scan_samples()

    assert pairs == []
    assert summary["missing_json"] == 0


def test_scan_samples_ignores_ai_and_model_comparison_json(monkeypatch, tmp_path):
    images_dir = configure_scan(monkeypatch, tmp_path)
    write_annotation(images_dir / "case.json")
    (images_dir / "case.png").write_bytes(b"image")
    write_annotation(images_dir / "caseai.json")
    write_annotation(images_dir / "casemodel.json")

    pairs, summary = splits.scan_samples()

    assert pairs == [("Images/case.png", images_dir / "case.json")]
    assert summary["missing_image"] == 0


def test_scan_samples_keeps_legitimate_stem_ending_in_ai(monkeypatch, tmp_path):
    images_dir = configure_scan(monkeypatch, tmp_path)
    write_annotation(images_dir / "thai.json")
    (images_dir / "thai.png").write_bytes(b"image")

    pairs, summary = splits.scan_samples()

    assert pairs == [("Images/thai.png", images_dir / "thai.json")]
    assert summary["paired"] == 1


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1])
def test_split_entries_rejects_invalid_val_ratio(ratio):
    with pytest.raises(ValueError, match="val_ratio must be between 0 and 1"):
        splits.split_entries([{"image_path": "case.png"}], ratio, seed=42)


def test_split_entries_rejects_empty_dataset():
    with pytest.raises(ValueError, match="dataset is empty"):
        splits.split_entries([], 0.2, seed=42)


def test_split_entries_rejects_dataset_too_small_for_two_splits():
    with pytest.raises(ValueError, match="at least 2 entries"):
        splits.split_entries([{"image_path": "only-case.png"}], 0.2, seed=42)


def test_split_entries_caps_validation_size_to_leave_training_sample():
    entries = [
        {"image_path": "case-a.png"},
        {"image_path": "case-b.png"},
    ]

    train, val = splits.split_entries(entries, 0.9, seed=42)

    assert len(train) == 1
    assert len(val) == 1


def test_scan_samples_reports_image_and_json_orphans(monkeypatch, tmp_path):
    images_dir = configure_scan(monkeypatch, tmp_path)
    write_annotation(images_dir / "paired.json")
    (images_dir / "paired.png").write_bytes(b"paired")
    write_annotation(images_dir / "json_only.json")
    (images_dir / "image_only.jpg").write_bytes(b"orphan")

    pairs, summary = splits.scan_samples()

    assert pairs == [("Images/paired.png", images_dir / "paired.json")]
    assert summary == {"paired": 1, "missing_image": 1, "missing_json": 1, "duplicates": 0}


def test_scan_samples_accounts_for_multiple_images_with_same_stem(monkeypatch, tmp_path):
    images_dir = configure_scan(monkeypatch, tmp_path)
    write_annotation(images_dir / "case.json")
    (images_dir / "case.dcm").write_bytes(b"dicom")
    (images_dir / "case.png").write_bytes(b"png")

    pairs, summary = splits.scan_samples()

    assert pairs == [("Images/case.dcm", images_dir / "case.json")]
    assert summary == {"paired": 1, "missing_image": 0, "missing_json": 0, "duplicates": 1}


def test_scan_samples_reports_root_spinefm_image_orphan(monkeypatch, tmp_path):
    configure_scan(monkeypatch, tmp_path)
    (tmp_path / "lonely.spinefm.png").write_bytes(b"orphan")

    pairs, summary = splits.scan_samples()

    assert pairs == []
    assert summary["missing_json"] == 1


def test_deduplicate_pairs_keeps_each_resolved_image_once(monkeypatch, tmp_path):
    monkeypatch.setattr(splits, "ROOT", tmp_path)
    image = tmp_path / "Images" / "case.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    first_json = image.with_suffix(".json")
    second_json = tmp_path / "duplicate.json"

    unique, duplicate_count = splits.deduplicate_pairs([
        ("Images/case.png", first_json),
        (str(image), second_json),
    ])

    assert unique == [("Images/case.png", first_json)]
    assert duplicate_count == 1


def test_split_entries_is_reproducible_for_fixed_seed():
    entries = [{"image_path": f"case-{index}.png"} for index in range(10)]

    first = splits.split_entries(entries, 0.2, seed=123)
    second = splits.split_entries(entries, 0.2, seed=123)

    assert first == second
    assert tuple(map(len, first)) == (8, 2)
    assert entries == [{"image_path": f"case-{index}.png"} for index in range(10)]


def test_split_entries_keeps_metadata_group_together():
    entries = [
        {"image_path": "p1-a.png", "patient_id": "p1"},
        {"image_path": "p1-b.png", "patient_id": "p1"},
        {"image_path": "p2-a.png", "patient_id": "p2"},
        {"image_path": "p3-a.png", "patient_id": "p3"},
    ]

    train, val = splits.split_entries(entries, 0.5, seed=42, group_key="patient_id")

    train_groups = {entry["patient_id"] for entry in train}
    val_groups = {entry["patient_id"] for entry in val}
    assert train_groups.isdisjoint(val_groups)


def test_split_entries_supports_dotted_metadata_group_key():
    entries = [
        {"image_path": "p1-a.png", "metadata": {"patient_id": "p1"}},
        {"image_path": "p1-b.png", "metadata": {"patient_id": "p1"}},
        {"image_path": "p2-a.png", "metadata": {"patient_id": "p2"}},
    ]

    train, val = splits.split_entries(
        entries,
        0.5,
        seed=42,
        group_key="metadata.patient_id",
    )

    group = lambda entry: entry["metadata"]["patient_id"]
    assert {group(entry) for entry in train}.isdisjoint({group(entry) for entry in val})


def test_split_entries_rejects_single_group():
    entries = [
        {"image_path": "p1-a.png", "patient_id": "p1"},
        {"image_path": "p1-b.png", "patient_id": "p1"},
    ]

    with pytest.raises(ValueError, match="at least 2 groups"):
        splits.split_entries(entries, 0.5, seed=42, group_key="patient_id")


def test_grouped_split_high_ratio_keeps_one_group_for_training():
    entries = [
        {"image_path": "p1.png", "patient_id": "p1"},
        {"image_path": "p2.png", "patient_id": "p2"},
    ]

    train, val = splits.split_entries(entries, 0.9, seed=42, group_key="patient_id")

    assert len(train) == 1
    assert len(val) == 1
    assert train[0]["patient_id"] != val[0]["patient_id"]


def test_split_entries_keeps_regex_group_together():
    entries = [
        {"image_path": "Images/P001_AP.png"},
        {"image_path": "Images/P001_LAT.png"},
        {"image_path": "Images/P002_AP.png"},
        {"image_path": "Images/P003_AP.png"},
    ]

    train, val = splits.split_entries(entries, 0.5, seed=42, group_regex=r"(P\d+)_")

    group = lambda entry: Path(entry["image_path"]).stem.split("_")[0]
    assert {group(entry) for entry in train}.isdisjoint({group(entry) for entry in val})


def test_split_entries_normalizes_windows_paths_for_group_regex():
    entries = [
        {"image_path": r"Images\202607\P001_AP.png"},
        {"image_path": r"Images\202607\P001_LAT.png"},
        {"image_path": r"Images\202607\P002_AP.png"},
    ]

    train, val = splits.split_entries(
        entries,
        0.5,
        seed=42,
        group_regex=r"Images/202607/(P\d+)_",
    )

    group = lambda entry: Path(entry["image_path"]).stem.split("_")[0]
    assert {group(entry) for entry in train}.isdisjoint({group(entry) for entry in val})


def test_missing_group_configuration_warns():
    with pytest.warns(UserWarning, match="patient-level isolation is disabled"):
        splits.warn_if_no_grouping(group_key=None, group_regex=None)


def test_main_reports_pairing_summary_and_warns_without_grouping(
    monkeypatch, tmp_path, capsys
):
    images_dir = configure_scan(monkeypatch, tmp_path)
    for stem in ("paired-a", "paired-b"):
        write_annotation(images_dir / f"{stem}.json")
        (images_dir / f"{stem}.png").write_bytes(b"paired")
    write_annotation(images_dir / "json-only.json")
    (images_dir / "image-only.png").write_bytes(b"orphan")
    monkeypatch.setattr(splits.sys, "argv", ["regenerate_splits.py", "--dry-run"])

    with pytest.warns(UserWarning, match="patient-level isolation is disabled"):
        splits.main()

    output = capsys.readouterr().out
    assert "找到 2 對 image+json" in output
    assert "缺 image 1 / 缺 JSON 1 / 重複 image 0" in output


def test_main_dry_run_does_not_write_or_backup_split_files(
    monkeypatch, tmp_path
):
    images_dir = configure_scan(monkeypatch, tmp_path)
    for stem in ("case-a", "case-b"):
        write_annotation(images_dir / f"{stem}.json")
        (images_dir / f"{stem}.png").write_bytes(b"image")

    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    train_file = annotation_dir / "train_annotations.json"
    val_file = annotation_dir / "val_annotations.json"
    train_file.write_text("train-sentinel", encoding="utf-8")
    val_file.write_text("val-sentinel", encoding="utf-8")
    monkeypatch.setattr(splits, "ANN_DIR", annotation_dir)
    monkeypatch.setattr(splits, "TRAIN_FILE", train_file)
    monkeypatch.setattr(splits, "VAL_FILE", val_file)
    monkeypatch.setattr(splits.sys, "argv", ["regenerate_splits.py", "--dry-run"])

    with pytest.warns(UserWarning, match="patient-level isolation is disabled"):
        splits.main()

    assert train_file.read_text(encoding="utf-8") == "train-sentinel"
    assert val_file.read_text(encoding="utf-8") == "val-sentinel"
    assert not train_file.with_suffix(".json.bak").exists()
    assert not val_file.with_suffix(".json.bak").exists()
