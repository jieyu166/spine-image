#!/usr/bin/env python
"""產生 tests/fixtures/medsam_core_cases.json：用 Python（OpenCV／torch）算出標準答案，
讓 tests/medsam_core.test.js 檢查標註工具裡的 JS 移植（MedsamCore）結果一致。只用合成資料，不含影像。

用法：python tests/make_medsam_core_fixtures.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import medsam_refine as mr  # noqa: E402
from refine_stage2 import apply_affine, crop_matrix  # noqa: E402

OUT = ROOT / 'tests' / 'fixtures' / 'medsam_core_cases.json'


def rle(mask):
    """一維 run-length（從 0 開始交替）。"""
    flat = mask.astype(np.uint8).ravel()
    runs, cur, n = [], 0, 0
    for v in flat:
        if v == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = v, 1
    runs.append(n)
    return runs


def vertebra(cx, cy, w, h, angle, dip):
    local = {'anteriorSuperior': (-w / 2, -h / 2), 'middleSuperior': (0, -h / 2 + dip), 'posteriorSuperior': (w / 2, -h / 2),
             'posteriorInferior': (w / 2, h / 2), 'middleInferior': (0, h / 2 - dip), 'anteriorInferior': (-w / 2, h / 2)}
    t = np.deg2rad(angle)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return {k: dict(zip('xy', (float(v) for v in np.array([cx, cy]) + r @ np.array(p)))) for k, p in local.items()}


def main():
    rng = np.random.default_rng(7)
    cases = {'affine': [], 'warp': [], 'upsample': [], 'masks': [], 'local_axes': [], 'preprocess': None}

    for center, side, crop in [((500.0, 800.0), 660.0, 512), ((1234.5, 987.25), 420.0, 1024)]:
        m = crop_matrix(center, side, 0.0, crop=crop)
        pts = [(100.0, 200.0), (center[0], center[1]), (1500.0, 30.0)]
        cases['affine'].append({'center': center, 'side': side, 'crop': crop, 'matrix': m.ravel().tolist(),
                                'points': pts, 'mapped': [apply_affine(m, *p) for p in pts]})

    gray = rng.integers(0, 256, (60, 50), dtype=np.uint8)
    for center, side in [((25.0, 30.0), 40.0), ((10.3, 50.7), 70.0)]:
        m = crop_matrix(center, side, 0.0, crop=32)
        out = cv2.warpAffine(gray, m, (32, 32), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cases['warp'].append({'w': 50, 'h': 60, 'gray': gray.ravel().tolist(), 'matrix': m.ravel().tolist(), 'S': 32,
                              'out': out.ravel().tolist()})

    for L, S in [(16, 64), (128, 512)]:
        logits = rng.normal(0, 3, (L, L)).astype(np.float32)
        up = torch.nn.functional.interpolate(torch.from_numpy(logits)[None, None], size=(S, S), mode='bilinear', align_corners=False)
        cases['upsample'].append({'L': L, 'S': S, 'logits': logits.ravel().tolist(), 'mask_rle': rle((up[0, 0] > 0).numpy())})

    S = 512
    specs = [(256, 250, 230, 160, 0, 0), (250, 260, 240, 170, 14, 12), (260, 250, 220, 180, -22, 6),
             (255, 255, 230, 170, 52, 0), (250, 250, 200, 200, 3, 25)]
    for i, (cx, cy, w, h, ang, dip) in enumerate(specs):
        pts = vertebra(cx, cy, w, h, ang, dip)
        mask = mr.rasterize(mr.outline_polygon(pts), S)
        if i == 2:
            mask[10:20, 10:20] = 1          # 另一個小區塊：最大區塊應排除它
        if i == 4:
            noise = rng.random((S, S)) < 0.02
            mask = np.where(noise, 1 - mask, mask).astype(np.uint8)   # 邊緣雜訊
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        comp = mr.largest_component(mask)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)[:, 0, :]
        prefer = mr.local_axes(vertebra(cx, cy, w, h, ang - 8, 0))[1]
        axes_plain = mr.axes_from_mask(mask)
        axes_pref = mr.axes_from_mask(mask, prefer_v=prefer)
        axes_pts = mr.local_axes(pts)
        cases['masks'].append({
            'S': S, 'mask_rle': rle(mask), 'area': int(comp.sum()),
            'contour_len': int(len(contour)), 'contour_mean': contour.mean(0).tolist(),
            'contour_set': sorted(map(list, {tuple(map(int, p)) for p in contour})),
            'prefer_v': list(map(float, prefer)),
            'axes_plain': [list(map(float, a)) for a in axes_plain], 'axes_pref': [list(map(float, a)) for a in axes_pref],
            'axes_points': [list(map(float, a)) for a in axes_pts],
            'points_with_point_axes': mr.mask_to_points(mask, axes_pts),
            'points_with_mask_axes': mr.mask_to_points(mask, axes_plain),
            'points_with_prefer_axes': mr.mask_to_points(mask, axes_pref),
        })

    tilted = vertebra(300, 300, 200, 150, 40, 0)
    for keys in (mr.POINTS, mr.SUPERIOR, mr.INFERIOR, ('anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior')):
        sub = {k: tilted[k] for k in keys}
        cases['local_axes'].append({'points': sub, 'axes': [list(map(float, a)) for a in mr.local_axes(sub)]})

    patch = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    cases['preprocess'] = {'S': 8, 'patch': patch.ravel().tolist(),
                           'imagenet': mr.preprocess(patch, 'imagenet').ravel().tolist(),
                           'minmax': mr.preprocess(patch, 'minmax').ravel().tolist()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases), encoding='utf-8')
    print(f'寫入 {OUT}（{OUT.stat().st_size / 1e6:.1f} MB）')


if __name__ == '__main__':
    main()
