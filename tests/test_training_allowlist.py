import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "build_training_allowlist.py"


def load_builder():
    assert SCRIPT.exists(), "build_training_allowlist.py must exist"
    spec = importlib.util.spec_from_file_location("build_training_allowlist", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_pair(root: Path, stem: str, image_bytes=b"image", json_extra=None):
    images = root / "Images"
    images.mkdir(parents=True, exist_ok=True)
    image = images / f"{stem}.png"
    annotation = images / f"{stem}.json"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(image_bytes)
    annotation.write_text(
        json.dumps({"vertebrae": [], **(json_extra or {})}),
        encoding="utf-8",
    )
    return f"Images/{stem}.png", annotation


def browser_signature(root: Path, image_path: str, json_path: str, *, mtime_ms=None):
    image = root / image_path
    annotation = root / json_path
    image_mtime = int(image.stat().st_mtime * 1000) if mtime_ms is None else mtime_ms
    json_mtime = int(annotation.stat().st_mtime * 1000) if mtime_ms is None else mtime_ms
    return (
        f"{image_path}|{image.stat().st_size}|{image_mtime}|"
        f"{json_path}|{annotation.stat().st_size}|{json_mtime}"
    )


def review_manifest(pair_rows):
    pairs = []
    reviews = {}
    for row in pair_rows:
        pair_id = f"{row['image_path']}\0{row['json_path']}"
        signature = row["signature"]
        pairs.append({
            "id": pair_id,
            "imagePath": row["image_path"],
            "jsonPath": row["json_path"],
            "signature": signature,
            "manual": False,
        })
        reviews[pair_id] = {
            "status": row["status"],
            "reason": row.get("reason", ""),
            "note": row.get("note", ""),
            "reviewedAt": "2026-08-02T00:00:00.000Z",
            "signature": signature,
        }
    return {
        "version": "1.0",
        "dataset": {"rootName": "Images"},
        "pairs": pairs,
        "reviews": reviews,
        "summary": {},
    }


def test_build_allowlist_includes_reviewed_match_and_locks_current_hashes(tmp_path):
    builder = load_builder()
    image_path, annotation = make_pair(tmp_path, "80145593")
    json_path = "Images/80145593.json"
    manifest = review_manifest([{
        "image_path": image_path,
        "json_path": json_path,
        "status": "match",
        "signature": browser_signature(tmp_path, image_path, json_path),
    }])

    result = builder.build_allowlist(
        manifest,
        [(image_path, annotation)],
        approved_stems=[],
        root=tmp_path,
        source_path="pair_review.json",
    )

    assert result["summary"] == {
        "review_pairs": 1,
        "selected": 1,
        "review_matches": 1,
        "physician_overrides": 0,
        "mtime_drift": 0,
        "untrainable_annotations": 0,
        "excluded": 0,
    }
    assert result["entries"] == [{
        "stem": "80145593",
        "image": {
            "path": image_path,
            "size": 5,
            "mtime_ms": int((tmp_path / image_path).stat().st_mtime * 1000),
            "sha256": hashlib.sha256(b"image").hexdigest(),
        },
        "annotation": {
            "path": json_path,
            "size": annotation.stat().st_size,
            "mtime_ms": int(annotation.stat().st_mtime * 1000),
            "sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
        },
        "selection": {
            "source": "review_match",
            "mtime_drift": False,
        },
        "review": {
            "status": "match",
            "reason": "",
            "note": "",
            "reviewed_at": "2026-08-02T00:00:00.000Z",
        },
    }]


def test_build_allowlist_accepts_mtime_only_drift_and_records_it(tmp_path):
    builder = load_builder()
    image_path, annotation = make_pair(tmp_path, "case")
    json_path = "Images/case.json"
    manifest = review_manifest([{
        "image_path": image_path,
        "json_path": json_path,
        "status": "match",
        "signature": browser_signature(
            tmp_path, image_path, json_path, mtime_ms=0
        ),
    }])

    result = builder.build_allowlist(
        manifest, [(image_path, annotation)], [], tmp_path, "review.json"
    )

    assert result["summary"]["selected"] == 1
    assert result["summary"]["mtime_drift"] == 1
    assert result["entries"][0]["selection"] == {
        "source": "review_match",
        "mtime_drift": True,
    }


def test_build_allowlist_excludes_size_changed_match_without_override(tmp_path):
    builder = load_builder()
    image_path, annotation = make_pair(tmp_path, "changed")
    json_path = "Images/changed.json"
    old_signature = browser_signature(tmp_path, image_path, json_path)
    annotation.write_text(json.dumps({"vertebrae": [], "fixed": True}), encoding="utf-8")
    manifest = review_manifest([{
        "image_path": image_path,
        "json_path": json_path,
        "status": "match",
        "signature": old_signature,
    }])

    result = builder.build_allowlist(
        manifest, [(image_path, annotation)], [], tmp_path, "review.json"
    )

    assert result["summary"]["selected"] == 0
    assert result["excluded"] == [{
        "stem": "changed",
        "reason": "reviewed_size_changed",
    }]


def test_build_allowlist_uses_explicit_physician_override_for_corrected_case(tmp_path):
    builder = load_builder()
    image_path, annotation = make_pair(tmp_path, "corrected")
    json_path = "Images/corrected.json"
    old_signature = browser_signature(tmp_path, image_path, json_path)
    annotation.write_text(json.dumps({"vertebrae": [], "fixed": True}), encoding="utf-8")
    manifest = review_manifest([{
        "image_path": image_path,
        "json_path": json_path,
        "status": "mismatch",
        "reason": "json_invalid",
        "signature": old_signature,
    }])

    result = builder.build_allowlist(
        manifest,
        [(image_path, annotation)],
        ["corrected"],
        tmp_path,
        "review.json",
    )

    assert result["summary"]["selected"] == 1
    assert result["summary"]["physician_overrides"] == 1
    assert result["entries"][0]["selection"]["source"] == "physician_override"
    assert result["entries"][0]["review"]["status"] == "mismatch"


def test_build_allowlist_rejects_ambiguous_override_stem(tmp_path):
    builder = load_builder()
    first_image, first_json = make_pair(tmp_path, "first/shared")
    second_image, second_json = make_pair(tmp_path, "second/shared")
    rows = []
    for image_path, annotation in (
        (first_image, first_json),
        (second_image, second_json),
    ):
        json_path = annotation.relative_to(tmp_path).as_posix()
        rows.append({
            "image_path": image_path,
            "json_path": json_path,
            "status": "mismatch",
            "signature": browser_signature(tmp_path, image_path, json_path),
        })
    manifest = review_manifest(rows)

    with pytest.raises(ValueError, match="ambiguous approved stem: shared"):
        builder.build_allowlist(
            manifest,
            [(first_image, first_json), (second_image, second_json)],
            ["shared"],
            tmp_path,
            "review.json",
        )


def test_build_allowlist_rejects_duplicate_and_unknown_approved_stems(tmp_path):
    builder = load_builder()
    image_path, annotation = make_pair(tmp_path, "known")
    json_path = "Images/known.json"
    manifest = review_manifest([{
        "image_path": image_path,
        "json_path": json_path,
        "status": "mismatch",
        "signature": browser_signature(tmp_path, image_path, json_path),
    }])

    with pytest.raises(ValueError, match="duplicate approved stems: known"):
        builder.build_allowlist(
            manifest, [(image_path, annotation)], ["known", "known"], tmp_path, "review.json"
        )
    with pytest.raises(ValueError, match="approved stems not found"):
        builder.build_allowlist(
            manifest, [(image_path, annotation)], ["unknown"], tmp_path, "review.json"
        )


def test_build_allowlist_rejects_unsupported_review_manifest_version(tmp_path):
    builder = load_builder()

    with pytest.raises(ValueError, match="unsupported review manifest version"):
        builder.build_allowlist(
            {"version": "2.0", "pairs": [], "reviews": {}},
            [],
            [],
            tmp_path,
            "review.json",
        )


def test_build_allowlist_excludes_reviewed_pair_without_vertebrae(tmp_path):
    builder = load_builder()
    image_path, annotation = make_pair(tmp_path, "measurement-only")
    annotation.write_text(json.dumps({"measurements": []}), encoding="utf-8")
    json_path = "Images/measurement-only.json"
    manifest = review_manifest([{
        "image_path": image_path,
        "json_path": json_path,
        "status": "match",
        "signature": browser_signature(tmp_path, image_path, json_path),
    }])

    result = builder.build_allowlist(
        manifest, [(image_path, annotation)], [], tmp_path, "review.json"
    )

    assert result["summary"]["selected"] == 0
    assert result["summary"]["untrainable_annotations"] == 1
    assert result["excluded"] == [{
        "stem": "measurement-only",
        "reason": "annotation_missing_vertebrae",
    }]


def test_main_writes_a_versioned_allowlist_artifact(tmp_path, capsys):
    builder = load_builder()
    image_path, annotation = make_pair(tmp_path, "approved")
    json_path = "Images/approved.json"
    manifest = review_manifest([{
        "image_path": image_path,
        "json_path": json_path,
        "status": "mismatch",
        "signature": browser_signature(tmp_path, image_path, json_path),
    }])
    review_path = tmp_path / "pair_review.json"
    review_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "manifests" / "allowlist.json"

    result = builder.main(
        [
            "--review-manifest", str(review_path),
            "--output", str(output_path),
            "--approve-stem", "approved",
        ],
        root=tmp_path,
        pairs=[(image_path, annotation)],
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["version"] == "1.0"
    assert payload["summary"]["selected"] == 1
    assert "Selected 1 / 1 reviewed pairs" in capsys.readouterr().out
