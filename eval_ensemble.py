#!/usr/bin/env python3
"""5-fold ensemble 評估 — 各 fold 單模型 vs 5-fold ensemble，對照 baseline。

用法：python eval_ensemble.py
前置：best_vertebra_model_fold1.pth ~ fold5.pth 已由 train_vertebra_model_cv.py 產出。
全程套用 main 分支的後處理（anchor 命名 + TTA + DARK decode + aspect-fix）。
"""
import os, glob, statistics as s
import numpy as np
from inference_vertebra import VertebraInference
from compare_annotations import load_gt_json, find_matching_json, compare_vertebrae

IMAGES = r'C:\Users\jai16\OneDrive\00 放射科\0筆記\Spine\Images'
FOLDS = [f'best_vertebra_model_fold{i}.pth' for i in range(1, 6)]


def gt_images():
    out = []
    for img in sorted(glob.glob(os.path.join(IMAGES, '*.png'))):
        jp = find_matching_json(img, IMAGES)
        if not jp:
            continue
        gt = load_gt_json(jp)
        if gt and 'vertebrae' in gt:
            out.append((img, gt))
    return out


def evaluate(inf, imgs, label):
    means = []
    for img, gt in imgs:
        try:
            r = inf.predict(img, spine_type='L')
        except Exception:
            continue
        _, sm = compare_vertebrae(gt['vertebrae'], r['vertebrae'])
        if sm['mean_distance_px'] is not None:
            means.append(sm['mean_distance_px'])
    ms = sorted(means)
    n = len(ms)
    nc = sorted(m for m in ms if m < 500)
    print(f'{label:30s} overall={s.mean(ms):6.1f}  median={s.median(ms):5.1f}  '
          f'<100={sum(1 for m in ms if m<100):>2}/{n}  排cata={s.mean(nc):5.1f}')
    return s.mean(ms), s.median(ms)


if __name__ == '__main__':
    imgs = gt_images()
    print(f'評估影像數: {len(imgs)}\n')

    missing = [f for f in FOLDS if not os.path.exists(f)]
    if missing:
        print('缺少 fold 權重:', missing)
        raise SystemExit(1)

    print('=== 各 fold 單模型 ===')
    for f in FOLDS:
        inf = VertebraInference(f, device='cuda')
        evaluate(inf, imgs, os.path.basename(f))

    print('\n=== 5-fold ensemble ===')
    ens = VertebraInference(FOLDS[0], device='cuda', ensemble_paths=FOLDS[1:])
    evaluate(ens, imgs, '5-fold ensemble')

    print('\n對照 baseline 單模型 (val 0.6234) + 全後處理 = 103.5 / median 45.7')
