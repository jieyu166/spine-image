#!/usr/bin/env python3
"""
脊椎椎體 6-point 檢測 - 5-fold Cross Validation 訓練腳本 V3.4

為什麼需要 CV:
  50 張標註資料隨機切 train/val 80/20，val 只有 10 張，雜訊極大、
  容易誤判訓練成敗。5-fold CV 讓每張圖都當過一次 val，
  平均 val loss 更能反映真實模型品質，std 能告訴妳模型對 split 的敏感度。
  預設依病歷號/資料集 subject ID 分組，避免同病人的不同影像跨 fold 洩漏。

用途:
  1. 驗證架構改動是否真的有效 (對照 mean val loss)
  2. 找到最佳 epoch 數 (避免固定 split 的 overfit 到特定幾張圖)
  3. CV 完之後用全部資料 + 最佳 epoch 數訓練最終模型

用法:
  # 標準 5-fold CV (預設 30 epoch / fold)
  python train_vertebra_model_cv.py --n-folds 5 --epochs 30

  # 快速驗證 (3 fold × 5 epoch，約 15-20 分鐘)
  python train_vertebra_model_cv.py --n-folds 3 --epochs 5

  # 跳過 stage B (只做 stage A，對照組)
  python train_vertebra_model_cv.py --n-folds 5 --epochs 30 --unfreeze-epoch -1

  # 僅重現舊實驗：逐影像切 fold（可能病人洩漏，不建議）
  python train_vertebra_model_cv.py --sample-level-folds

輸出:
  - cv_results.json          每個 fold 的 best val loss + 平均 ± 標準差
  - best_vertebra_model_fold{N}.pth   每 fold 的 best checkpoint
  - training_loss_curve_fold{N}.png   每 fold 的 loss 曲線
"""

import argparse
import json
import os
import re
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_vertebra_model import (
    VertebraDataset,
    VertebraCornerModel,
    VertebraTrainer,
    RepeatDataset,
    get_transforms,
    collate_fn,
    RADIMAGENET_PATH_DEFAULT,
)

DEFAULT_GROUP_REGEX = r'(?:^|/)(\d{8}|[CL]\d{5})[^/]*$'


class InMemoryVertebraDataset(VertebraDataset):
    """VertebraDataset 變體：直接吃 Python list (避免 JSON load)"""

    def __init__(self, data_dir, annotations, transform=None, max_vertebrae=8):
        self.data_dir = data_dir
        self.transform = transform
        self.max_vertebrae = max_vertebrae
        self.num_channels = max_vertebrae * self.POINTS_PER_VERTEBRA
        self.annotations = annotations if isinstance(annotations, list) else [annotations]


def manual_kfold(n_items, n_splits, seed=42):
    """手動 KFold (避免多一個 sklearn 依賴)

    Yields:
        (train_indices, val_indices) for each fold
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)

    fold_size = n_items // n_splits
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else n_items
        val_idx = set(indices[start:end].tolist())
        train_idx = [j for j in range(n_items) if j not in val_idx]
        yield train_idx, sorted(val_idx)


def extract_group_id(annotation, group_regex=DEFAULT_GROUP_REGEX):
    """Extract a patient/dataset subject ID from a normalized image path."""
    image_path = str(annotation.get('image_path', '')).replace('\\', '/')
    match = re.search(group_regex, image_path)
    if not match:
        raise ValueError(
            f"cannot extract patient group from image_path={image_path!r} "
            f"with regex={group_regex!r}"
        )
    return match.group(1) if match.groups() else match.group(0)


def manual_group_kfold(annotations, n_splits, seed=42,
                       group_regex=DEFAULT_GROUP_REGEX):
    """Deterministic greedy K-fold that keeps each patient in one fold."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    grouped = {}
    for index, annotation in enumerate(annotations):
        group_id = extract_group_id(annotation, group_regex)
        grouped.setdefault(group_id, []).append(index)
    if len(grouped) < n_splits:
        raise ValueError(
            f"need at least {n_splits} patient groups, found {len(grouped)}"
        )

    rng = np.random.default_rng(seed)
    group_items = sorted(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    fold_indices = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for _, indices in group_items:
        target_fold = min(range(n_splits), key=lambda i: (fold_sizes[i], i))
        fold_indices[target_fold].extend(indices)
        fold_sizes[target_fold] += len(indices)

    all_indices = set(range(len(annotations)))
    for val_indices in fold_indices:
        val_set = set(val_indices)
        train_indices = sorted(all_indices - val_set)
        yield train_indices, sorted(val_indices)


def load_all_annotations(train_path, val_path):
    """合併 train + val annotations 成單一 list"""
    with open(train_path, 'r', encoding='utf-8') as f:
        train_anns = json.load(f)
    with open(val_path, 'r', encoding='utf-8') as f:
        val_anns = json.load(f)
    if isinstance(train_anns, dict):
        train_anns = [train_anns]
    if isinstance(val_anns, dict):
        val_anns = [val_anns]
    return train_anns + val_anns


def run_one_fold(fold_idx, n_folds, train_anns, val_anns, config, device):
    print(f"\n{'=' * 60}")
    print(f"=== Fold {fold_idx + 1}/{n_folds} ===")
    print(f"Train: {len(train_anns)}  Val: {len(val_anns)}")
    print(f"{'=' * 60}")

    train_ds = InMemoryVertebraDataset(
        config['data_dir'], train_anns,
        transform=get_transforms(True),
        max_vertebrae=config['max_vertebrae'],
    )
    val_ds = InMemoryVertebraDataset(
        config['data_dir'], val_anns,
        transform=get_transforms(False),
        max_vertebrae=config['max_vertebrae'],
    )

    repeat = config.get('repeat_dataset', 1)
    if repeat > 1:
        train_ds_wrapped = RepeatDataset(train_ds, repeat=repeat)
    else:
        train_ds_wrapped = train_ds

    train_loader = DataLoader(
        train_ds_wrapped,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    model = VertebraCornerModel(
        max_vertebrae=config['max_vertebrae'],
        points_per_vertebra=config['points_per_vertebra'],
        pretrained=True,
        pretrained_path=config['pretrained_path'],
    )
    trainer = VertebraTrainer(model, train_loader, val_loader, device, config)
    trainer.train()

    # 把 fold 產物改名避免被下一 fold 覆蓋
    fold_label = f"fold{fold_idx + 1}"
    if os.path.exists('best_vertebra_model.pth'):
        shutil.move('best_vertebra_model.pth', f'best_vertebra_model_{fold_label}.pth')
    if os.path.exists('training_loss_curve.png'):
        shutil.move('training_loss_curve.png', f'training_loss_curve_{fold_label}.png')

    return {
        'best_val_loss': trainer.best_val_loss,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='V3.4 Vertebra 6-point Detection - 5-fold CV')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30,
                        help='每個 fold 訓練的 epoch 數')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--repeat-dataset', type=int, default=8)
    parser.add_argument('--pretrained-path', type=str, default=RADIMAGENET_PATH_DEFAULT)
    parser.add_argument('--unfreeze-epoch', type=int, default=5,
                        help='Stage B 起始 epoch；-1 表示永不解凍')
    parser.add_argument('--stage-b-lr', type=float, default=1e-4)
    parser.add_argument('--train-annotations', type=str,
                        default='endplate_training_data/annotations/train_annotations.json')
    parser.add_argument('--val-annotations', type=str,
                        default='endplate_training_data/annotations/val_annotations.json')
    parser.add_argument('--output', type=str, default='cv_results.json')
    parser.add_argument(
        '--group-regex',
        default=DEFAULT_GROUP_REGEX,
        help='病人/subject ID regex；第一個 capture group 為 group ID',
    )
    parser.add_argument(
        '--sample-level-folds',
        action='store_true',
        help='重現舊實驗用；停用病人隔離並逐影像切 fold',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        'data_dir': '.',
        'train_annotations': args.train_annotations,
        'val_annotations': args.val_annotations,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_workers': 0,
        'max_vertebrae': 8,
        'points_per_vertebra': VertebraDataset.POINTS_PER_VERTEBRA,
        'repeat_dataset': args.repeat_dataset,
        'pretrained_path': args.pretrained_path,
        'unfreeze_epoch': args.unfreeze_epoch if args.unfreeze_epoch >= 0 else None,
        'stage_b_lr': args.stage_b_lr,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CV config: {args.n_folds} folds × {args.epochs} epochs (seed={args.seed})")

    if not os.path.exists(config['train_annotations']):
        print(f"ERROR: {config['train_annotations']} not found")
        return 1

    all_anns = load_all_annotations(config['train_annotations'], config['val_annotations'])
    print(f"Total samples: {len(all_anns)}")

    fold_results = []
    if args.sample_level_folds:
        print("WARNING: sample-level folds enabled; patient leakage is possible")
        fold_iterator = manual_kfold(
            len(all_anns), args.n_folds, seed=args.seed
        )
    else:
        fold_iterator = manual_group_kfold(
            all_anns,
            args.n_folds,
            seed=args.seed,
            group_regex=args.group_regex,
        )

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
        train_fold = [all_anns[i] for i in train_idx]
        val_fold = [all_anns[i] for i in val_idx]
        result = run_one_fold(fold_idx, args.n_folds, train_fold, val_fold, config, device)
        fold_results.append(result)

        # 累積中間結果：跑到一半也能看目前成績
        _dump_summary(args.output, fold_results, config, args)

    # 最終結果
    best_losses = [r['best_val_loss'] for r in fold_results]
    mean = float(np.mean(best_losses))
    std = float(np.std(best_losses))

    print(f"\n{'=' * 60}")
    print(f"CV Results ({args.n_folds}-fold, {args.epochs} epochs/fold)")
    print(f"{'=' * 60}")
    for i, v in enumerate(best_losses):
        print(f"  Fold {i + 1}: best val loss = {v:.4f}")
    print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
    print(f"\nSaved: {args.output}")

    return 0


def _dump_summary(output_path, fold_results, config, args):
    best_losses = [r['best_val_loss'] for r in fold_results]
    summary = {
        'n_folds_completed': len(fold_results),
        'n_folds_total': args.n_folds,
        'epochs_per_fold': args.epochs,
        'seed': args.seed,
        'fold_grouping': (
            'sample-level' if args.sample_level_folds else args.group_regex
        ),
        'fold_best_val_losses': best_losses,
        'mean_best_val_loss': float(np.mean(best_losses)) if best_losses else None,
        'std_best_val_loss': float(np.std(best_losses)) if best_losses else None,
        'per_fold_history': [
            {'train_losses': r['train_losses'], 'val_losses': r['val_losses']}
            for r in fold_results
        ],
        'config': {
            k: v for k, v in config.items()
            if isinstance(v, (int, float, str, bool, type(None), list))
        },
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    raise SystemExit(main())
