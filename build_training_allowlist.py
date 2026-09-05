#!/usr/bin/env python3
"""Build a hash-locked training allowlist from physician pair review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _normalize_relative(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(root: Path, relative_path: str) -> dict:
    normalized = _normalize_relative(relative_path)
    path = root / normalized
    stat = path.stat()
    return {
        "path": normalized,
        "size": stat.st_size,
        "mtime_ms": int(stat.st_mtime * 1000),
        "sha256": file_sha256(path),
    }


def _parse_browser_signature(signature: str) -> dict:
    parts = signature.split("|")
    if len(parts) != 6:
        raise ValueError(f"invalid pair signature: {signature!r}")
    return {
        "image_path": _normalize_relative(parts[0]),
        "image_size": int(parts[1]),
        "image_mtime_ms": int(parts[2]),
        "json_path": _normalize_relative(parts[3]),
        "json_size": int(parts[4]),
        "json_mtime_ms": int(parts[5]),
    }


def _annotation_issue(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return "annotation_invalid_json"
    if not isinstance(data, dict) or "vertebrae" not in data:
        return "annotation_missing_vertebrae"
    if not isinstance(data["vertebrae"], list):
        return "annotation_invalid_vertebrae"
    return None


def build_allowlist(
    review_manifest: dict,
    pairs: list[tuple[str, Path]],
    approved_stems: list[str],
    root: Path,
    source_path: str,
) -> dict:
    """Select reviewed pairs and lock their current contents with SHA-256."""
    if review_manifest.get("version") != "1.0":
        raise ValueError(
            f"unsupported review manifest version: {review_manifest.get('version')!r}"
        )
    duplicate_approvals = sorted(
        stem for stem, count in Counter(approved_stems).items() if count > 1
    )
    if duplicate_approvals:
        raise ValueError(
            f"duplicate approved stems: {', '.join(duplicate_approvals)}"
        )

    root = root.resolve()
    pair_index = {}
    for image_path, annotation_path in pairs:
        image_path = _normalize_relative(image_path)
        json_path = annotation_path.resolve().relative_to(root).as_posix()
        pair_index[(image_path, json_path)] = annotation_path.resolve()

    entries = []
    excluded = []
    mtime_drift = 0
    review_matches = 0
    physician_overrides = 0
    untrainable_annotations = 0
    approved = set(approved_stems)

    reviews = review_manifest.get("reviews", {})
    manifest_pairs = review_manifest.get("pairs", [])
    manifest_stem_counts = Counter(
        Path(_normalize_relative(pair.get("jsonPath", ""))).stem
        for pair in manifest_pairs
    )
    for stem in approved:
        if manifest_stem_counts[stem] > 1:
            raise ValueError(f"ambiguous approved stem: {stem}")

    for pair in manifest_pairs:
        image_path = _normalize_relative(pair.get("imagePath", ""))
        json_path = _normalize_relative(pair.get("jsonPath", ""))
        pair_id = pair.get("id", f"{image_path}\0{json_path}")
        review = reviews.get(pair_id, {})
        stem = Path(json_path).stem
        annotation_path = pair_index.get((image_path, json_path))
        if annotation_path is None:
            excluded.append({"stem": stem, "reason": "pair_not_found"})
            continue

        parsed = _parse_browser_signature(review.get("signature", pair.get("signature", "")))
        image_record = _file_record(root, image_path)
        annotation_record = _file_record(root, json_path)
        sizes_match = (
            parsed["image_path"] == image_path
            and parsed["json_path"] == json_path
            and parsed["image_size"] == image_record["size"]
            and parsed["json_size"] == annotation_record["size"]
        )

        if stem in approved:
            source = "physician_override"
            approved.remove(stem)
        elif review.get("status") == "match" and sizes_match:
            source = "review_match"
        else:
            if review.get("status") == "match":
                reason = "reviewed_size_changed"
            elif review.get("status") == "mismatch":
                reason = "review_mismatch"
            else:
                reason = "not_reviewed"
            excluded.append({"stem": stem, "reason": reason})
            continue

        annotation_issue = _annotation_issue(annotation_path)
        if annotation_issue:
            untrainable_annotations += 1
            excluded.append({"stem": stem, "reason": annotation_issue})
            continue
        if source == "physician_override":
            physician_overrides += 1
        else:
            review_matches += 1

        drift = (
            parsed["image_mtime_ms"] != image_record["mtime_ms"]
            or parsed["json_mtime_ms"] != annotation_record["mtime_ms"]
        )
        mtime_drift += int(drift)
        entries.append({
            "stem": stem,
            "image": image_record,
            "annotation": annotation_record,
            "selection": {"source": source, "mtime_drift": drift},
            "review": {
                "status": review.get("status", ""),
                "reason": review.get("reason", ""),
                "note": review.get("note", ""),
                "reviewed_at": review.get("reviewedAt", ""),
            },
        })

    if approved:
        raise ValueError(f"approved stems not found in review manifest: {sorted(approved)}")

    return {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_review": source_path,
        "policy": {
            "review_status": "match",
            "mtime_only_drift_allowed": True,
            "approved_stems": list(approved_stems),
        },
        "summary": {
            "review_pairs": len(manifest_pairs),
            "selected": len(entries),
            "review_matches": review_matches,
            "physician_overrides": physician_overrides,
            "mtime_drift": mtime_drift,
            "untrainable_annotations": untrainable_annotations,
            "excluded": len(excluded),
        },
        "entries": entries,
        "excluded": excluded,
    }


def main(argv=None, *, root: Path | None = None, pairs=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a physician-reviewed, hash-locked training allowlist"
    )
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--approve-stem", action="append", default=[])
    args = parser.parse_args(argv)

    if root is None or pairs is None:
        import regenerate_splits

        root = regenerate_splits.ROOT if root is None else root
        pairs = regenerate_splits.scan_samples()[0] if pairs is None else pairs
    root = Path(root)

    with args.review_manifest.open("r", encoding="utf-8") as stream:
        review_manifest = json.load(stream)
    payload = build_allowlist(
        review_manifest,
        pairs,
        args.approve_stem,
        root,
        str(args.review_manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = payload["summary"]
    print(
        f"Selected {summary['selected']} / {summary['review_pairs']} reviewed pairs "
        f"({summary['review_matches']} review matches + "
        f"{summary['physician_overrides']} physician overrides)"
    )
    print(
        f"mtime-only drift: {summary['mtime_drift']} / "
        f"excluded: {summary['excluded']}"
    )
    print(f"Wrote allowlist: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
