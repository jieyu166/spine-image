#!/usr/bin/env python3
"""重生 train/val annotations 從硬碟上實際存在的 paired image+json。

舊的 train/val split 寫死了已經不存在的 SpineFM mask 路徑，且漏收日期
子資料夾內的已標註樣本。此工具遞迴掃描 Images/ + repo root 的 spinefm
檔，重生 train/val split，並把舊檔備份成 .bak。

配對規則:
    - <stem>.json 配同一資料夾同名 <stem>.dcm/.png/.jpg/.jpeg。
    - 同 stem 有多種影像格式時依 IMAGE_EXTS 順序選一份，並計入 duplicates。
    - *samp 及有同名 base 樣本的 *ai/*model 是比較產物，不進訓練也不計孤兒。
    - JSON 須含 vertebrae 欄位（由 build_annotation_entry 驗證）。
    - 可用 --group-key 或 --group-regex 做病人層級 train/val 隔離。

用法:
    python regenerate_splits.py [--val-ratio 0.2] [--seed 42] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import warnings
from pathlib import Path


IMAGE_EXTS = ('.dcm', '.png', '.jpg', '.jpeg')
COMPARISON_STEM_SUFFIXES = ('ai', 'model')
ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / 'Images'
ANN_DIR = ROOT / 'endplate_training_data' / 'annotations'
TRAIN_FILE = ANN_DIR / 'train_annotations.json'
VAL_FILE = ANN_DIR / 'val_annotations.json'


def is_ignored_image_stem(stem: str, directory: Path) -> bool:
    """Return whether a stem is a physician/model comparison artifact."""
    lowered = stem.lower()
    if lowered.endswith('samp'):
        return True
    for suffix in COMPARISON_STEM_SUFFIXES:
        if not lowered.endswith(suffix):
            continue
        base_stem = stem[:-len(suffix)]
        has_base = (directory / f'{base_stem}.json').exists() or any(
            (directory / f'{base_stem}{ext}').exists() for ext in IMAGE_EXTS
        )
        if has_base:
            return True
    return False


def deduplicate_pairs(
    pairs: list[tuple[str, Path]],
) -> tuple[list[tuple[str, Path]], int]:
    """依 image resolved path 去重，保留排序後第一筆。"""
    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for image_path, json_path in pairs:
        resolved = (ROOT / image_path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((image_path, json_path))
    return unique, len(pairs) - len(unique)


def scan_samples() -> tuple[list[tuple[str, Path]], dict[str, int]]:
    """掃描可配對樣本，並統計缺 image / JSON 的孤兒檔。"""
    pairs: list[tuple[str, Path]] = []
    summary = {"paired": 0, "missing_image": 0, "missing_json": 0, "duplicates": 0}
    paired_images: set[Path] = set()

    # Images/ 遞迴配對：同資料夾內 <stem>.json 配同名圖
    for json_path in sorted(IMAGES_DIR.rglob('*.json')):
        stem = json_path.stem
        if is_ignored_image_stem(stem, json_path.parent):
            continue
        candidates = [
            json_path.with_name(f'{stem}{ext}')
            for ext in IMAGE_EXTS
            if json_path.with_name(f'{stem}{ext}').exists()
        ]
        if not candidates:
            summary["missing_image"] += 1
            continue
        image = candidates[0]
        summary["duplicates"] += len(candidates) - 1
        rel = image.relative_to(ROOT).as_posix()
        pairs.append((rel, json_path))
        paired_images.update(candidate.resolve() for candidate in candidates)

    for image in sorted(
        path for path in IMAGES_DIR.rglob('*')
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and not is_ignored_image_stem(path.stem, path.parent)
    ):
        if image.resolve() not in paired_images:
            summary["missing_json"] += 1

    # repo root 下的 spinefm 檔 (image.spinefm.png + image.spinefm.json 等)
    for json_path in sorted(ROOT.glob('*.spinefm.json')):
        stem = json_path.stem  # e.g. image2.spinefm
        candidates = [
            ROOT / f'{stem}{ext}'
            for ext in IMAGE_EXTS
            if (ROOT / f'{stem}{ext}').exists()
        ]
        if not candidates:
            summary["missing_image"] += 1
            continue
        image = candidates[0]
        summary["duplicates"] += len(candidates) - 1
        pairs.append((image.name, json_path))
        paired_images.update(candidate.resolve() for candidate in candidates)

    for ext in IMAGE_EXTS:
        for image in ROOT.glob(f'*.spinefm{ext}'):
            if image.resolve() not in paired_images:
                summary["missing_json"] += 1

    pairs, duplicate_pairs = deduplicate_pairs(pairs)
    summary["duplicates"] += duplicate_pairs
    summary["paired"] = len(pairs)
    return pairs, summary


def find_paired_samples():
    """相容舊介面：只回傳 (image_path_relative_to_root, json_full_path)。"""
    return scan_samples()[0]


def warn_if_no_grouping(group_key: str | None, group_regex: str | None) -> None:
    """沒有可靠 grouping 時發出醒目告警，不從檔名自行猜病人 ID。"""
    if not group_key and not group_regex:
        warnings.warn(
            "WARNING: patient-level isolation is disabled; provide --group-key or "
            "--group-regex because no reliable patient ID was detected.",
            UserWarning,
            stacklevel=2,
        )


def get_dotted_value(data: dict, dotted_key: str):
    """Read a top-level or dotted nested key from an annotation entry."""
    value = data
    for key in dotted_key.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def split_entries(
    entries: list[dict],
    val_ratio: float,
    seed: int,
    group_key: str | None = None,
    group_regex: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """依固定 seed 將 entries 切成 train/val，可選 metadata group 隔離。"""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive)")
    if not entries:
        raise ValueError("dataset is empty")
    if len(entries) < 2:
        raise ValueError("at least 2 entries are required for train/val split")
    if group_key and group_regex:
        raise ValueError("use only one of group_key or group_regex")
    rng = random.Random(seed)
    if group_key or group_regex:
        grouped: dict[str, list[dict]] = {}
        pattern = re.compile(group_regex) if group_regex else None
        for entry in entries:
            if group_key:
                group = get_dotted_value(entry, group_key)
            else:
                normalized_path = entry.get("image_path", "").replace("\\", "/")
                match = pattern.search(normalized_path)
                group = match.group(1) if match and match.groups() else (
                    match.group(0) if match else None
                )
            if group in (None, ""):
                source = f"key: {group_key}" if group_key else f"regex: {group_regex}"
                raise ValueError(f"missing group for {entry.get('image_path', '?')} ({source})")
            grouped.setdefault(str(group), []).append(entry)
        groups = list(grouped.values())
        if len(groups) < 2:
            raise ValueError("at least 2 groups are required for grouped train/val split")
        rng.shuffle(groups)
        n_val = max(1, int(round(len(entries) * val_ratio)))
        val: list[dict] = []
        train: list[dict] = []
        for index, group_entries in enumerate(groups):
            target = val if len(val) < n_val and index < len(groups) - 1 else train
            target.extend(group_entries)
        return train, val
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * val_ratio))))
    return shuffled[n_val:], shuffled[:n_val]


def build_annotation_entry(image_rel_path: str, json_path: Path) -> dict | None:
    """把單一 standalone JSON 轉成 dataset annotation entry。"""
    try:
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [skip] {json_path.name}: 讀檔失敗 {e}")
        return None

    if not isinstance(data, dict) or 'vertebrae' not in data:
        print(f"  [skip] {json_path.name}: 無 vertebrae")
        return None

    entry = dict(data)  # shallow copy 保留全部欄位（model 只讀它用得到的）
    # 補上 dataset 需要的欄位
    entry['source_file'] = json_path.name
    entry['image_path'] = image_rel_path.replace('/', os.sep)
    # 統一 spine_type (dataset 兩種都接，但寫齊比較乾淨)
    if 'spine_type' not in entry and 'spineType' in entry:
        entry['spine_type'] = entry['spineType']
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    grouping = ap.add_mutually_exclusive_group()
    grouping.add_argument(
        '--group-key',
        help='用 JSON key 做病人級 grouping；支援 metadata.patient_id dotted path',
    )
    grouping.add_argument(
        '--group-regex',
        help='從 image_path 擷取 group；第一個 capture group 為 group ID',
    )
    ap.add_argument('--dry-run', action='store_true', help='只印報告，不寫檔')
    args = ap.parse_args()

    print(f"Scanning {IMAGES_DIR} ...")
    pairs, scan_summary = scan_samples()
    print(f"找到 {len(pairs)} 對 image+json")
    print(
        f"配對摘要: 缺 image {scan_summary['missing_image']} / "
        f"缺 JSON {scan_summary['missing_json']} / "
        f"重複 image {scan_summary['duplicates']}"
    )

    entries: list[dict] = []
    for img_rel, json_path in pairs:
        e = build_annotation_entry(img_rel, json_path)
        if e is not None:
            entries.append(e)
    print(f"成功建立 {len(entries)} 個 annotation entries")

    # 預設維持 sample-level split；有可靠 ID 時可明確選擇病人級 grouping。
    warn_if_no_grouping(args.group_key, args.group_regex)
    train_entries, val_entries = split_entries(
        entries,
        args.val_ratio,
        args.seed,
        group_key=args.group_key,
        group_regex=args.group_regex,
    )
    print(f"Split (seed={args.seed}, val_ratio={args.val_ratio}): "
          f"{len(train_entries)} train / {len(val_entries)} val")

    print("\nTrain 樣本 stems (前 10):")
    for e in train_entries[:10]:
        print(f"  - {Path(e['image_path']).stem} (spine_type={e.get('spine_type', '?')})")
    print("Val 樣本 stems:")
    for e in val_entries:
        print(f"  - {Path(e['image_path']).stem} (spine_type={e.get('spine_type', '?')})")

    if args.dry_run:
        print("\n[dry-run] 不寫檔。")
        return

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    for src in (TRAIN_FILE, VAL_FILE):
        if src.exists():
            bak = src.with_suffix(src.suffix + '.bak')
            shutil.copy2(src, bak)
            print(f"備份 {src.name} -> {bak.name}")

    TRAIN_FILE.write_text(
        json.dumps(train_entries, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    VAL_FILE.write_text(
        json.dumps(val_entries, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f"\n寫入 {TRAIN_FILE.relative_to(ROOT)} ({len(train_entries)} 樣本)")
    print(f"寫入 {VAL_FILE.relative_to(ROOT)} ({len(val_entries)} 樣本)")


if __name__ == '__main__':
    main()
