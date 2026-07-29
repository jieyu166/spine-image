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
