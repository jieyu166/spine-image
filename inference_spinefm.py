#!/usr/bin/env python3
"""
SpineFM-based 脊椎推論腳本
SpineFM-based Spine Inference Script

使用 SpineFM (Medical-SAM-Adaptor) 取得椎體 segmentation mask，
從 mask 提取上下 endplate 輪廓，轉換為角點座標，
接入 Genant 骨折分級 + 椎間盤計算。

輸出 v2.3 相容 JSON，可直接由 spine-inference-web.html 讀取。

Usage:
    python inference_spinefm.py --input image.png --spine-type L
    python inference_spinefm.py --input image.dcm --spine-type C --output result.json
    python inference_spinefm.py --input ./images/ --output ./results/
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spinefm_wrapper import SpineFMWrapper
from spine_metrics import calculate_metrics, calculate_discs
from convert_pkl_to_contour import (
    extract_contour, split_endplates, VERTEBRA_ORDER, BOUNDARY_VERTEBRAE,
)


# 角點繪製顏色 (BGR)
CORNER_COLORS = [
    (0, 255, 0),    # anteriorSuperior - 綠
    (255, 0, 0),    # posteriorSuperior - 藍
    (0, 0, 255),    # posteriorInferior - 紅
    (0, 255, 255),  # anteriorInferior - 黃
]
CORNER_NAMES = ['anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior']


def clean_mask(mask, min_area=200):
    """清理 SpineFM 輸出的 noisy mask

    1. 形態學 closing 填補孔洞
    2. 形態學 opening 移除雜訊
    3. 只保留最大連通區域
    4. 凸包 (convex hull) 平滑化
    """
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255

    # 形態學清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    # 只保留最大連通區域
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    # 用凸包填充 → 更平滑的輪廓
    hull = cv2.convexHull(largest)
    clean = np.zeros_like(mask_u8)
    cv2.fillPoly(clean, [hull], 255)

    return clean


def mask_to_corner_points(mask, boundary_type=None):
    """從 binary mask 提取 4 個角點 + middleSuperior/middleInferior

    策略:
        1. clean_mask → 形態學清理 + 凸包平滑
        2. extract_contour → 最大輪廓
        3. split_endplates → 上/下 endplate 弧 (按 X 排序)
        4. 角點 = 各弧線前/後 15% 區段的 Y 極值點
        5. middle 點 = 弧線中央 50% 區段的 Y 極值點

    Args:
        mask: (H, W) binary numpy array
        boundary_type: None, 'upper', or 'lower'

    Returns:
        dict or None
    """
    # 清理 mask
    cleaned = clean_mask(mask)
    if cleaned is None:
        # fallback: 直接用原始 mask
        cleaned = (mask > 0.5).astype(np.uint8) * 255

    contour = extract_contour(cleaned / 255.0)
    if contour is None or len(contour) < 10:
        return None

    sup_pts, inf_pts = split_endplates(contour)

    points = {}

    def _extract_endplate_points(arc_pts, is_superior):
        """從 endplate 弧提取 anterior, posterior, middle 點"""
        n = len(arc_pts)
        # 使用 15% 邊緣區段（比 25% 更精確）
        q_front = max(1, int(n * 0.15))
        q_back = min(n - 1, int(n * 0.85))

        if is_superior:
            # 上終板: Y 最小 = 最上面
            front = arc_pts[:q_front]
            back = arc_pts[q_back:]
            mid = arc_pts[q_front:q_back]
            ant = front[front[:, 1].argmin()]
            post = back[back[:, 1].argmin()]
            mid_pt = mid[mid[:, 1].argmin()] if len(mid) > 0 else None
        else:
            # 下終板: Y 最大 = 最下面
            front = arc_pts[:q_front]
            back = arc_pts[q_back:]
            mid = arc_pts[q_front:q_back]
            ant = front[front[:, 1].argmax()]
            post = back[back[:, 1].argmax()]
            mid_pt = mid[mid[:, 1].argmax()] if len(mid) > 0 else None

        return ant, post, mid_pt

    # 上終板 (boundary='lower' 的椎體沒有 superior)
    if boundary_type != 'lower' and sup_pts is not None and len(sup_pts) >= 4:
        ant, post, mid = _extract_endplate_points(sup_pts, is_superior=True)
        points['anteriorSuperior'] = {'x': float(ant[0]), 'y': float(ant[1])}
        points['posteriorSuperior'] = {'x': float(post[0]), 'y': float(post[1])}
        if mid is not None:
            points['middleSuperior'] = {'x': float(mid[0]), 'y': float(mid[1])}

    # 下終板 (boundary='upper' 的椎體沒有 inferior)
    if boundary_type != 'upper' and inf_pts is not None and len(inf_pts) >= 4:
        ant, post, mid = _extract_endplate_points(inf_pts, is_superior=False)
        points['anteriorInferior'] = {'x': float(ant[0]), 'y': float(ant[1])}
        points['posteriorInferior'] = {'x': float(post[0]), 'y': float(post[1])}
        if mid is not None:
            points['middleInferior'] = {'x': float(mid[0]), 'y': float(mid[1])}

    return points if points else None


class SpineFMInference:
    """SpineFM-based 椎體角點推論器"""

    def __init__(self, weights_path='spinefm_weights', dataset='NHANESII', device='auto'):
        self.wrapper = SpineFMWrapper(
            weights_path=weights_path,
            dataset=dataset,
            device=device,
        )

    def predict(self, image_path, spine_type='L'):
        """對單張影像執行推論

        Args:
            image_path: 影像路徑
            spine_type: 'L' (lumbar) 或 'C' (cervical)

        Returns:
            dict: v2.3 相容的推論結果
        """
        # ── 1. SpineFM segmentation ──
        print(f"Running SpineFM on: {image_path}")
        sfm_result = self.wrapper.predict_single(image_path, spine_type=spine_type)

        masks = sfm_result['masks']         # OrderedDict[name, mask]
        img = sfm_result['image']           # PIL Image
        w, h = sfm_result['image_size']
        image_np = np.array(img)

        boundary_config = BOUNDARY_VERTEBRAE.get(spine_type, {})

        # ── 2. Mask → Corner Points ──
        vertebrae = []
        for name, mask in masks.items():
            # 確保 mask 是 2D
            if mask.ndim == 3:
                mask = mask.squeeze(0)

            boundary_type = boundary_config.get(name, None)
            points = mask_to_corner_points(mask, boundary_type=boundary_type)

            if points is None:
                print(f"  WARNING: Could not extract corners from {name} mask, skipping")
                continue

            vertebra = {
                'name': name,
                'boundaryType': boundary_type,
                'hasMiddlePoints': ('middleSuperior' in points),
                'points': points,
                'confidences': {},  # SpineFM 不提供 per-corner confidence
            }
            vertebrae.append(vertebra)

        # ── 3. 計算指標 (Genant classification) ──
        for v in vertebrae:
            calculate_metrics(v)

        # ── 4. 計算椎間盤 ──
        discs = calculate_discs(vertebrae, spine_type)

        # ── 5. 產生 combined mask (視覺化用) ──
        combined_mask = np.zeros((h, w), dtype=np.float32)
        for name, mask in masks.items():
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            # resize mask to image size if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h))
            combined_mask = np.maximum(combined_mask, mask.astype(np.float32))

        return {
            'image_path': str(image_path),
            'spine_type': spine_type,
            'image_info': {'width': w, 'height': h},
            'predicted_count': len(vertebrae),
            'count_confidence': 1.0,  # SpineFM 不提供 count confidence
            'vertebrae': vertebrae,
            'discs': discs,
            'heatmap': combined_mask,
            'channel_heatmaps': None,
            'original_image': image_np,
            'backend': 'SpineFM',
        }

    def to_json(self, result):
        """將推論結果轉為 v2.3 JSON (不含影像和 heatmap)"""
        return {
            'version': '2.3',
            'exportDate': datetime.now().isoformat(),
            'spineType': result['spine_type'],
            'backend': 'SpineFM',
            'imageInfo': result['image_info'],
            'vertebrae': [
                {
                    'name': v['name'],
                    'boundaryType': v['boundaryType'],
                    'hasMiddlePoints': v.get('hasMiddlePoints', False),
                    'points': v['points'],
                    'anteriorHeight': v.get('anteriorHeight'),
                    'middleHeight': v.get('middleHeight'),
                    'posteriorHeight': v.get('posteriorHeight'),
                    'heightRatio': v.get('heightRatio'),
                    'anteriorWedgingFracture': v.get('anteriorWedgingFracture', False),
                    'biconcaveCompressionFracture': v.get('biconcaveCompressionFracture', False),
                    'crushDeformityFracture': v.get('crushDeformityFracture', False),
                    'genantGrade': v.get('genantGrade', 0),
                    'genantType': v.get('genantType'),
                }
                for v in result['vertebrae']
            ],
            'discs': result['discs'],
        }

    def visualize(self, result, output_path=None, show=False):
        """視覺化推論結果"""
        image = result['original_image'].copy()
        vertebrae = result['vertebrae']
        combined_mask = result['heatmap']
        h, w = image.shape[:2]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(
            f"SpineFM Inference - {result['spine_type']}-spine "
            f"({len(vertebrae)} vertebrae detected)",
            fontsize=14, fontweight='bold'
        )

        # 1. 原始影像 + 角點
        vis_image = image.copy()
        for v in vertebrae:
            pts = v['points']
            corners_xy = []

            for j, corner_name in enumerate(CORNER_NAMES):
                if corner_name not in pts:
                    continue
                p = pts[corner_name]
                x, y = int(p['x']), int(p['y'])
                corners_xy.append((x, y, j))

                color = CORNER_COLORS[j]
                radius = max(3, min(w, h) // 200)
                cv2.circle(vis_image, (x, y), radius, color, -1)

            # 連線
            if len(corners_xy) >= 2:
                sorted_corners = sorted(corners_xy, key=lambda c: c[2])
                for k in range(len(sorted_corners)):
                    x1, y1, _ = sorted_corners[k]
                    x2, y2, _ = sorted_corners[(k + 1) % len(sorted_corners)]
                    cv2.line(vis_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # 名稱標記
            if corners_xy:
                avg_x = int(np.mean([c[0] for c in corners_xy]))
                avg_y = int(np.mean([c[1] for c in corners_xy]))
                cv2.putText(vis_image, v['name'], (avg_x - 15, avg_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # 骨折標記
                grade = v.get('genantGrade', 0)
                if grade > 0:
                    fx_type = v.get('genantType', '?')
                    cv2.putText(vis_image, f"G{grade}({fx_type})",
                                (avg_x - 15, avg_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        axes[0].imshow(vis_image)
        axes[0].set_title('Corner Points + Fracture Flags')
        axes[0].axis('off')

        # 2. Segmentation mask overlay
        mask_overlay = image.copy()
        if combined_mask is not None:
            mask_resized = cv2.resize(combined_mask, (w, h))
            color_mask = np.zeros_like(image)
            color_mask[:, :, 1] = (mask_resized * 200).astype(np.uint8)  # green channel
            mask_overlay = cv2.addWeighted(mask_overlay, 0.7, color_mask, 0.3, 0)

        axes[1].imshow(mask_overlay)
        axes[1].set_title('Segmentation Masks')
        axes[1].axis('off')

        # 3. Metrics table
        axes[2].axis('off')
        table_data = []
        for v in vertebrae:
            ant_h = v.get('anteriorHeight')
            mid_h = v.get('middleHeight')
            post_h = v.get('posteriorHeight')
            grade = v.get('genantGrade', 0)
            fx_type = v.get('genantType', '-')

            table_data.append([
                v['name'],
                f"{ant_h:.1f}" if ant_h else '-',
                f"{mid_h:.1f}" if mid_h else '-',
                f"{post_h:.1f}" if post_h else '-',
                f"G{grade}" if grade > 0 else 'Normal',
                fx_type if grade > 0 else '-',
            ])

        if table_data:
            table = axes[2].table(
                cellText=table_data,
                colLabels=['Vertebra', 'Ant H', 'Mid H', 'Post H', 'Genant', 'Type'],
                loc='center', cellLoc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            axes[2].set_title('Vertebra Metrics (Genant Classification)')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {output_path}")

        if show:
            plt.show()

        plt.close(fig)


def process_single(inferencer, image_path, spine_type, output_dir=None):
    """處理單張影像"""
    result = inferencer.predict(image_path, spine_type=spine_type)

    stem = Path(image_path).stem
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{stem}_spinefm.json")
        vis_path = os.path.join(output_dir, f"{stem}_spinefm.png")
    else:
        json_path = str(Path(image_path).with_suffix('.spinefm.json'))
        vis_path = str(Path(image_path).with_suffix('.spinefm.png'))

    # 儲存 JSON
    json_data = inferencer.to_json(result)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")

    # 視覺化
    inferencer.visualize(result, output_path=vis_path)

    # 印出摘要
    print(f"\n{'='*60}")
    print(f"Image: {image_path}")
    print(f"Spine type: {spine_type}")
    print(f"Vertebrae detected: {len(result['vertebrae'])}")
    for v in result['vertebrae']:
        grade = v.get('genantGrade', 0)
        if grade > 0:
            print(f"  ** {v['name']}: Genant Grade {grade} ({v.get('genantType', '?')} fracture)")
    print(f"{'='*60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='SpineFM-based Spine Vertebra Inference'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory')
    parser.add_argument('--spine-type', '-t', default='L', choices=['L', 'C'],
                        help='Spine type: L (lumbar) or C (cervical)')
    parser.add_argument('--weights', '-w', default='spinefm_weights',
                        help='SpineFM weights directory')
    parser.add_argument('--dataset', '-d', default='NHANESII',
                        choices=['NHANESII', 'CSXA', 'FINETUNED'],
                        help='SpineFM dataset/model variant')
    parser.add_argument('--device', default='auto',
                        help='Device: auto, cuda, or cpu')

    args = parser.parse_args()

    inferencer = SpineFMInference(
        weights_path=args.weights,
        dataset=args.dataset,
        device=args.device,
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        # 批次處理
        exts = {'.png', '.jpg', '.jpeg', '.dcm'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in exts]
        print(f"Found {len(images)} images in {input_path}")

        for i, img_path in enumerate(sorted(images)):
            print(f"\n[{i+1}/{len(images)}] Processing: {img_path.name}")
            try:
                process_single(inferencer, str(img_path), args.spine_type, args.output)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        process_single(inferencer, str(input_path), args.spine_type, args.output)


if __name__ == '__main__':
    main()
