#!/usr/bin/env python3
"""TTA / 推論設定快速掃描 eval — 載入模型一次，對所有有 GT 的影像算 overall_mean / median。

複用 compare_annotations 的 GT 配對與距離計算，只換 inference 設定，避免每次重跑慢的 compare。
"""
import os, glob, statistics as s
import numpy as np
from inference_vertebra import VertebraInference
from compare_annotations import load_gt_json, find_matching_json, compare_vertebrae

IMAGES = r'C:\Users\jai16\OneDrive\00 放射科\0筆記\Spine\Images'


def eval_config(inf, label, configurator):
    """configurator(inf): 設定 inf 上的 TTA / aspect 屬性。回傳統計 dict。"""
    configurator(inf)
    means = []
    for img in sorted(glob.glob(os.path.join(IMAGES, '*.png'))):
        jp = find_matching_json(img, IMAGES)
        if not jp:
            continue
        gt = load_gt_json(jp)
        if gt is None or 'vertebrae' not in gt:
            continue
        try:
            r = inf.predict(img, spine_type='L')
        except Exception:
            continue
        _, summary = compare_vertebrae(gt['vertebrae'], r['vertebrae'])
        if summary['mean_distance_px'] is not None:
            means.append(summary['mean_distance_px'])
    ms = sorted(means)
    n = len(ms)
    nc = sorted(m for m in ms if m < 500)
    print(f'{label:32s} n={n}  overall={s.mean(ms):6.1f}  median={s.median(ms):5.1f}  '
          f'<100={sum(1 for m in ms if m<100):>2}  排cata mean={s.mean(nc):5.1f}')
    return ms


if __name__ == '__main__':
    inf = VertebraInference('best_vertebra_model.pth', device='cuda')

    CONFIGS = {
        'baseline (TTA off)': lambda x: setattr(x, 'tta', False),
        'TTA 預設 5 變體': lambda x: (setattr(x, 'tta', True), setattr(x, 'tta_ops', None)),
        'TTA +更多旋轉 ±3/±6/±10': lambda x: (setattr(x, 'tta', True), setattr(x, 'tta_ops', [
            ('identity',), ('rot', 3.0), ('rot', -3.0), ('rot', 6.0), ('rot', -6.0),
            ('rot', 10.0), ('rot', -10.0), ('photo', 0.85), ('photo', 1.15)])),
        'TTA 只旋轉 ±6': lambda x: (setattr(x, 'tta', True), setattr(x, 'tta_ops', [
            ('identity',), ('rot', 6.0), ('rot', -6.0)])),
        'TTA 只亮度': lambda x: (setattr(x, 'tta', True), setattr(x, 'tta_ops', [
            ('identity',), ('photo', 0.8), ('photo', 1.2)])),
    }
    for label, cfg in CONFIGS.items():
        eval_config(inf, label, cfg)
