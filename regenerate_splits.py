#!/usr/bin/env python3
"""重生 train/val annotations 從硬碟上實際存在的 paired image+json。

舊的 train/val split 寫死了已經不存在的 SpineFM mask 路徑，且漏收 7 個
Images/ 已標註樣本。此工具遞迴掃描 Images/（含日期子資料夾）+ repo root 的
spinefm 檔，重生 8:2 split (固定 seed=42)，並把舊檔備份成 .bak。

配對規則:
    - <stem>.json 配同一資料夾同名 <stem>.png/.dcm/.jpg/.jpeg。
    - 以 `samp` 結尾者為醫師標註 overlay（spinal-annotation-web.html 輸出），
      僅作 AI 輸出 eval 參考，一律排除於訓練 split 之外。
    - JSON 須含 vertebrae 欄位（由 build_annotation_entry 驗證）。

用法:
    python regenerate_splits.py [--val-ratio 0.2] [--seed 42] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


IMAGE_EXTS = ('.dcm', '.png', '.jpg', '.jpeg')
ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / 'Images'
ANN_DIR = ROOT / 'endplate_training_data' / 'annotations'
TRAIN_FILE = ANN_DIR / 'train_annotations.json'
VAL_FILE = ANN_DIR / 'val_annotations.json'


def find_paired_samples():
    """掃 Images/（遞迴，含日期子資料夾）+ ROOT 找出
    (image_path_relative_to_root, json_full_path)。"""
    pairs: list[tuple[str, Path]] = []

    # Images/ 遞迴配對：同資料夾內 <stem>.json 配同名圖
    for json_path in sorted(IMAGES_DIR.rglob('*.json')):
        stem = json_path.stem
        # samp 結尾＝醫師標註 overlay，僅供 eval，不進訓練
        if stem.endswith('samp'):
            continue
        for ext in IMAGE_EXTS:
            img = json_path.with_name(f'{stem}{ext}')
            if img.exists():
                rel = img.relative_to(ROOT).as_posix()
                pairs.append((rel, json_path))
                break

    # repo root 下的 spinefm 檔 (image.spinefm.png + image.spinefm.json 等)
    for json_path in sorted(ROOT.glob('*.spinefm.json')):
        stem = json_path.stem  # e.g. image2.spinefm
        for ext in IMAGE_EXTS:
            img = ROOT / f'{stem}{ext}'
            if img.exists():
                pairs.append((img.name, json_path))
                break

    return pairs


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
    ap.add_argument('--dry-run', action='store_true', help='只印報告，不寫檔')
    args = ap.parse_args()

    print(f"Scanning {IMAGES_DIR} ...")
    pairs = find_paired_samples()
    print(f"找到 {len(pairs)} 對 image+json")

    entries: list[dict] = []
    for img_rel, json_path in pairs:
        e = build_annotation_entry(img_rel, json_path)
        if e is not None:
            entries.append(e)
    print(f"成功建立 {len(entries)} 個 annotation entries")

    # 8:2 random split, 固定 seed
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    n_val = max(1, int(round(len(entries) * args.val_ratio)))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]
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
