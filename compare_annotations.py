#!/usr/bin/env python3
"""
椎體手標 vs 模型推論 比較/QA 工具

用途:
  1. Labeling QA：找出手標與模型誤差大的影像 → 人工 review
  2. 半自動 labeling：沒手標的影像產出模型 JSON 當起點

設計原則:
  - 絕不修改原始 JSON。所有結果寫到 --output-dir
  - 模型本身也有誤差，「距離大」不代表手標錯，可能是模型錯，要人眼看

輸出:
  output_dir/
    compare_summary.json          全資料夾統計 + 按 max distance 排序
    <stem>_compare.json           per-image 詳細：每角點 (gt, pred, distance_px)
    <stem>_compare.png            overlay (綠=手標、紅=模型、線色照距離)
    <stem>_model.json             沒手標的影像 → 純模型 JSON 作為起點

用法:
  python compare_annotations.py \
    --input-dir "C:/Users/jai16/OneDrive/00 放射科/0筆記/Spine/Images" \
    --output-dir "./comparison_output" \
    --model best_vertebra_model.pth \
    --spine-type L
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from inference_vertebra import VertebraInference, CORNER_NAMES


COLOR_GT = (0, 255, 0)      # 手標：綠 (BGR for OpenCV)
COLOR_PRED = (0, 0, 255)    # 模型：紅
DISTANCE_BANDS = [
    (10.0, (0, 255, 0)),     # < 10 px：綠（一致）
    (25.0, (0, 255, 255)),   # 10-25 px：黃（邊緣）
    (float('inf'), (0, 0, 255)),  # > 25 px：紅（需 review）
]


def load_gt_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'vertebrae' in data:
        return data
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and 'vertebrae' in first:
            return first
    return None


def find_matching_json(image_path, json_dir):
    img = Path(image_path)
    json_dir = Path(json_dir)
    candidates = [
        json_dir / f'{img.stem}.json',
        json_dir / f'{img.stem}_annotation.json',
        json_dir / f'{img.stem}_corners.json',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def point_distance(p1, p2):
    dx = p1.get('x', 0) - p2.get('x', 0)
    dy = p1.get('y', 0) - p2.get('y', 0)
    return float(np.hypot(dx, dy))


def compare_vertebrae(gt_vertebrae, pred_vertebrae):
    gt_by_name = {v.get('name', ''): v for v in gt_vertebrae if v.get('name')}
    pred_by_name = {v.get('name', ''): v for v in pred_vertebrae if v.get('name')}
    all_names = sorted(set(gt_by_name.keys()) | set(pred_by_name.keys()))

    per_vertebra = []
    distances = []

    for name in all_names:
        gt = gt_by_name.get(name)
        pred = pred_by_name.get(name)
        item = {'name': name, 'in_gt': gt is not None, 'in_pred': pred is not None, 'corners': {}}

        if gt is None or pred is None:
            per_vertebra.append(item)
            continue

        gt_pts = gt.get('points', {})
        pred_pts = pred.get('points', {})
        if not isinstance(gt_pts, dict) or not isinstance(pred_pts, dict):
            per_vertebra.append(item)
            continue

        for cname in CORNER_NAMES:
            gt_c = gt_pts.get(cname)
            pred_c = pred_pts.get(cname)
            both = isinstance(gt_c, dict) and isinstance(pred_c, dict)
            if both:
                d = point_distance(gt_c, pred_c)
                distances.append(d)
                item['corners'][cname] = {
                    'gt': {'x': float(gt_c.get('x', 0)), 'y': float(gt_c.get('y', 0))},
                    'pred': {'x': float(pred_c.get('x', 0)), 'y': float(pred_c.get('y', 0))},
                    'distance_px': d,
                }
            elif isinstance(gt_c, dict) or isinstance(pred_c, dict):
                item['corners'][cname] = {
                    'gt': {'x': float(gt_c.get('x', 0)), 'y': float(gt_c.get('y', 0))} if isinstance(gt_c, dict) else None,
                    'pred': {'x': float(pred_c.get('x', 0)), 'y': float(pred_c.get('y', 0))} if isinstance(pred_c, dict) else None,
                    'distance_px': None,
                }
        per_vertebra.append(item)

    summary = {
        'n_gt_vertebrae': len(gt_by_name),
        'n_pred_vertebrae': len(pred_by_name),
        'n_matched_vertebrae': sum(1 for n in all_names if n in gt_by_name and n in pred_by_name),
        'n_only_in_gt': sum(1 for n in all_names if n in gt_by_name and n not in pred_by_name),
        'n_only_in_pred': sum(1 for n in all_names if n not in gt_by_name and n in pred_by_name),
        'n_matched_corners': len(distances),
        'mean_distance_px': float(np.mean(distances)) if distances else None,
        'median_distance_px': float(np.median(distances)) if distances else None,
        'max_distance_px': float(np.max(distances)) if distances else None,
        'p90_distance_px': float(np.percentile(distances, 90)) if distances else None,
    }
    return per_vertebra, summary


def band_color(dist):
    for thresh, color in DISTANCE_BANDS:
        if dist < thresh:
            return color
    return DISTANCE_BANDS[-1][1]


def draw_overlay(image_rgb, gt_vertebrae, pred_vertebrae, per_vertebra):
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = vis.shape[:2]
    radius_gt = max(4, min(w, h) // 200)
    radius_pred = max(3, min(w, h) // 250)
    line_thick = max(1, min(w, h) // 500)
    font_scale = max(0.4, min(w, h) / 1800)
    font_thick = max(1, int(font_scale * 2))

    for v in gt_vertebrae:
        pts = v.get('points', {})
        if isinstance(pts, dict):
            for p in pts.values():
                if isinstance(p, dict):
                    cv2.circle(vis, (int(p.get('x', 0)), int(p.get('y', 0))),
                               radius_gt, COLOR_GT, -1)

    for v in pred_vertebrae:
        pts = v.get('points', {})
        if isinstance(pts, dict):
            for p in pts.values():
                if isinstance(p, dict):
                    cv2.circle(vis, (int(p.get('x', 0)), int(p.get('y', 0))),
                               radius_pred, COLOR_PRED, 2)

    for item in per_vertebra:
        for cdata in item['corners'].values():
            d = cdata.get('distance_px')
            gt = cdata.get('gt')
            pred = cdata.get('pred')
            if d is None or not (gt and pred):
                continue
            cv2.line(vis, (int(gt['x']), int(gt['y'])),
                     (int(pred['x']), int(pred['y'])),
                     band_color(d), line_thick)

    for v in gt_vertebrae:
        pts = v.get('points', {})
        if isinstance(pts, dict):
            xs = [p.get('x', 0) for p in pts.values() if isinstance(p, dict)]
            ys = [p.get('y', 0) for p in pts.values() if isinstance(p, dict)]
            if xs and ys:
                cv2.putText(vis, v.get('name', ''),
                            (int(np.mean(xs)) - 20, int(np.mean(ys))),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 0), font_thick)

    legend = [
        ('GT (existing)', COLOR_GT),
        ('Pred (model)', COLOR_PRED),
        ('dist <10 px', (0, 255, 0)),
        ('dist 10-25 px', (0, 255, 255)),
        ('dist >25 px', (0, 0, 255)),
    ]
    for i, (text, color) in enumerate(legend):
        cv2.putText(vis, text, (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis


def process_one(image_path, args, analyzer, output_dir, json_dir):
    print(f'\n--- {image_path.name} ---')
    try:
        result = analyzer.predict(str(image_path),
                                  spine_type=args.spine_type,
                                  confidence_threshold=args.threshold)
    except Exception as e:
        print(f'  推論失敗：{e}')
        return None

    pred_vertebrae = result['vertebrae']
    image_rgb = result['original_image']

    json_path = find_matching_json(image_path, json_dir)

    if json_path is None:
        out = output_dir / f'{image_path.stem}_model.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump({
                'image_path': image_path.name,
                'spine_type': args.spine_type,
                'predicted_count': result['predicted_count'],
                'count_confidence': result['count_confidence'],
                'vertebrae': pred_vertebrae,
            }, f, indent=2, ensure_ascii=False, default=float)
        print(f'  無手標 → 寫 {out.name}')
        return {'image': image_path.name, 'status': 'no_gt',
                'n_pred_vertebrae': len(pred_vertebrae),
                'model_json': out.name}

    gt = load_gt_json(json_path)
    if gt is None or 'vertebrae' not in gt:
        print(f'  GT JSON 格式異常：{json_path}')
        return None

    gt_vertebrae = gt['vertebrae']
    per_vertebra, summary = compare_vertebrae(gt_vertebrae, pred_vertebrae)

    compare_path = output_dir / f'{image_path.stem}_compare.json'
    with open(compare_path, 'w', encoding='utf-8') as f:
        json.dump({
            'image': image_path.name,
            'gt_json': json_path.name,
            'spine_type': args.spine_type,
            'summary': summary,
            'per_vertebra': per_vertebra,
        }, f, indent=2, ensure_ascii=False, default=float)

    overlay = draw_overlay(image_rgb, gt_vertebrae, pred_vertebrae, per_vertebra)
    overlay_path = output_dir / f'{image_path.stem}_compare.png'
    # cv2.imwrite 在 Windows 對非 ASCII 路徑會 fail，改用 imencode + write_bytes
    ok, buf = cv2.imencode('.png', overlay)
    if ok:
        overlay_path.write_bytes(buf.tobytes())
    else:
        print(f'  WARN: 無法編碼 overlay PNG')

    if summary['mean_distance_px'] is not None:
        print(f'  matched_corners={summary["n_matched_corners"]} '
              f'mean={summary["mean_distance_px"]:.1f}px '
              f'max={summary["max_distance_px"]:.1f}px '
              f'→ {compare_path.name} + {overlay_path.name}')
    else:
        print(f'  GT/Pred 椎體完全沒交集（gt={summary["n_gt_vertebrae"]} pred={summary["n_pred_vertebrae"]}）')

    return {'image': image_path.name, 'status': 'compared',
            **summary,
            'compare_json': compare_path.name,
            'overlay': overlay_path.name}


def main():
    parser = argparse.ArgumentParser(description='椎體手標 vs 模型推論 比較工具')
    parser.add_argument('--input-dir', required=True, help='影像資料夾')
    parser.add_argument('--output-dir', default='comparison_output', help='輸出資料夾')
    parser.add_argument('--json-dir', default=None,
                        help='既有 JSON 資料夾（預設與 --input-dir 同一個）')
    parser.add_argument('--model', default='best_vertebra_model.pth')
    parser.add_argument('--spine-type', default='L', choices=['L', 'C'])
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Heatmap peak 信心門檻')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--extensions', default='dcm,png,jpg,jpeg',
                        help='以逗號分隔')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = Path(args.json_dir) if args.json_dir else input_dir

    exts = {e.strip().lower() for e in args.extensions.split(',')}
    images = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower().lstrip('.') in exts
    ])
    print(f'找到 {len(images)} 張影像於 {input_dir}')
    if not images:
        return 1

    analyzer = VertebraInference(args.model, device=args.device)

    results = [r for r in (process_one(p, args, analyzer, output_dir, json_dir)
                           for p in images) if r]

    compared = [r for r in results if r.get('status') == 'compared']
    no_gt = [r for r in results if r.get('status') == 'no_gt']

    compared_sorted = sorted(
        compared,
        key=lambda r: r.get('max_distance_px') or 0,
        reverse=True,
    )

    mean_of_means = None
    if compared:
        means = [r['mean_distance_px'] for r in compared
                 if r.get('mean_distance_px') is not None]
        if means:
            mean_of_means = float(np.mean(means))

    summary = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'model': args.model,
        'spine_type': args.spine_type,
        'n_images': len(images),
        'n_compared': len(compared),
        'n_no_gt': len(no_gt),
        'overall_mean_distance_px': mean_of_means,
        'images_compared_sorted_by_max_distance': compared_sorted,
        'images_without_gt': no_gt,
    }
    summary_path = output_dir / 'compare_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    print(f'\n=== Done ===')
    print(f'比較 {len(compared)} 張 / 無 GT {len(no_gt)} 張')
    if mean_of_means is not None:
        print(f'整體平均距離 {mean_of_means:.1f} px')
    print(f'Summary: {summary_path}')
    print('前 10 張誤差最大（優先 review）:')
    for r in compared_sorted[:10]:
        print(f'  {r["image"]}: max={r["max_distance_px"]:.1f}px  '
              f'mean={r["mean_distance_px"]:.1f}px  '
              f'matched={r["n_matched_corners"]}')

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
