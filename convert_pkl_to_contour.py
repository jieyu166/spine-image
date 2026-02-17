#!/usr/bin/env python3
"""
PKL Mask → 終板輪廓點 轉換腳本
將外部專案的 vertebra segmentation mask 轉換為終板輪廓點 JSON，
供 Web 編輯器 (pkl-contour-editor.html) 載入、由醫師篩選後匯出。

Usage:
    python convert_pkl_to_contour.py --input Images/L14156_lumbar_masks_4.pkl
    python convert_pkl_to_contour.py --input Images/L14156_lumbar_masks_4.pkl --output output.json
    python convert_pkl_to_contour.py --input folder_of_pkls/
"""

import pickle
import json
import argparse
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np


# ── 椎體解剖學順序 (由上到下) ──
VERTEBRA_ORDER = {
    'C': ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1'],
    'L': ['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'],
}

# 邊界椎體定義：只有單側終板
BOUNDARY_VERTEBRAE = {
    'L': {'T12': 'lower', 'S1': 'upper'},   # T12 只有下終板, S1 只有上終板
    'C': {'C2': 'lower', 'T1': 'upper'},
}


def detect_spine_type(vertebra_names):
    """從椎體名稱推斷 spine type"""
    names_upper = [n.upper() for n in vertebra_names]
    has_l = any(n.startswith('L') for n in names_upper)
    has_c = any(n.startswith('C') for n in names_upper)
    has_s = any(n.startswith('S') for n in names_upper)
    has_t12 = 'T12' in names_upper

    if has_l or has_s or has_t12:
        return 'L'
    if has_c:
        return 'C'
    return 'L'


def sort_vertebrae(names, spine_type):
    """按解剖學順序排序椎體名稱 (由上到下)"""
    order = VERTEBRA_ORDER.get(spine_type, [])
    order_map = {name: i for i, name in enumerate(order)}

    def sort_key(n):
        return order_map.get(n, 999)

    return sorted(names, key=sort_key)


def extract_contour(mask):
    """從二值 mask 提取最大外輪廓的座標點"""
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    # 取最大輪廓
    largest = max(contours, key=cv2.contourArea)
    # shape: (N, 1, 2) → (N, 2)
    pts = largest.reshape(-1, 2)
    return pts


def split_endplates(contour_pts):
    """將椎體輪廓分為上終板和下終板

    策略 (Left-Right Arc method):
    cv2.findContours 回傳的輪廓是有序環路。找到最左和最右端點，
    這兩點自然將環路分成上弧 (superior) 和下弧 (inferior)，
    不受椎體傾斜角度影響。

    Returns:
        (superior_pts, inferior_pts) - 各為 (N, 2) numpy array，
        都按 X 座標從小到大排序 (anterior → posterior 或反之)
    """
    if contour_pts is None or len(contour_pts) < 4:
        return None, None

    n = len(contour_pts)

    # 找最左和最右端點的索引
    idx_left = int(np.argmin(contour_pts[:, 0]))
    idx_right = int(np.argmax(contour_pts[:, 0]))

    # 沿著有序環路分成兩段弧
    if idx_left < idx_right:
        arc1_idx = list(range(idx_left, idx_right + 1))
        arc2_idx = list(range(idx_right, n)) + list(range(0, idx_left + 1))
    else:
        arc1_idx = list(range(idx_left, n)) + list(range(0, idx_right + 1))
        arc2_idx = list(range(idx_right, idx_left + 1))

    arc1 = contour_pts[arc1_idx]
    arc2 = contour_pts[arc2_idx]

    # 平均 Y 較小的 = 上終板 (影像座標系 Y 向下)
    if arc1[:, 1].mean() < arc2[:, 1].mean():
        sup_pts, inf_pts = arc1, arc2
    else:
        sup_pts, inf_pts = arc2, arc1

    # 按 X 排序
    sup_pts = sup_pts[sup_pts[:, 0].argsort()]
    inf_pts = inf_pts[inf_pts[:, 0].argsort()]

    return sup_pts, inf_pts


def downsample_contour(pts, max_points=50):
    """將輪廓點均勻降採樣到 max_points 個點

    保留首尾端點，中間均勻取樣。
    """
    if pts is None or len(pts) == 0:
        return pts
    if len(pts) <= max_points:
        return pts

    indices = np.linspace(0, len(pts) - 1, max_points, dtype=int)
    return pts[indices]


def image_to_base64_png(image_array):
    """將 numpy 影像轉為 base64 PNG"""
    if image_array.ndim == 2:
        # Grayscale → RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_array

    success, buffer = cv2.imencode('.png', image_rgb)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")

    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def process_single_pkl(pkl_path, max_points_per_endplate=50):
    """處理單一 pkl 檔案

    Returns:
        (dict, np.ndarray): JSON 結構, 原始影像 array
    """
    pkl_path = Path(pkl_path)
    print(f"Loading: {pkl_path.name}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    sample_id = data.get('sample_id', pkl_path.stem)
    image = data['image']       # (H, W) uint8
    masks = data['masks']       # dict of name → (H, W) float64

    h, w = image.shape[:2]

    # 過濾掉 combined mask 和空 mask
    vertebra_names = []
    for name, mask in masks.items():
        if name.lower() == 'vertebrae':
            continue
        if mask.max() < 0.5:
            continue
        vertebra_names.append(name)

    spine_type = detect_spine_type(vertebra_names)
    sorted_names = sort_vertebrae(vertebra_names, spine_type)
    boundary_config = BOUNDARY_VERTEBRAE.get(spine_type, {})

    print(f"  Sample ID: {sample_id}")
    print(f"  Image size: {w}x{h}")
    print(f"  Spine type: {spine_type}")
    print(f"  Vertebrae: {sorted_names}")

    # 提取每個椎體的終板輪廓
    vertebrae_data = []
    for name in sorted_names:
        mask = masks[name]
        contour = extract_contour(mask)
        if contour is None:
            print(f"  WARNING: No contour found for {name}, skipping")
            continue

        sup_pts, inf_pts = split_endplates(contour)
        boundary_type = boundary_config.get(name, None)

        entry = {
            'name': name,
            'boundaryType': boundary_type,
        }

        # 上終板
        if boundary_type != 'lower' and sup_pts is not None and len(sup_pts) > 0:
            sampled = downsample_contour(sup_pts, max_points_per_endplate)
            entry['superiorEndplate'] = [
                {'x': float(pt[0]), 'y': float(pt[1])} for pt in sampled
            ]
            print(f"  {name} superior: {len(sampled)} pts")

        # 下終板
        if boundary_type != 'upper' and inf_pts is not None and len(inf_pts) > 0:
            sampled = downsample_contour(inf_pts, max_points_per_endplate)
            entry['inferiorEndplate'] = [
                {'x': float(pt[0]), 'y': float(pt[1])} for pt in sampled
            ]
            print(f"  {name} inferior: {len(sampled)} pts")

        # 完整輪廓 (備用)
        if contour is not None:
            sampled_full = downsample_contour(contour, max_points_per_endplate * 2)
            entry['fullContour'] = [
                {'x': float(pt[0]), 'y': float(pt[1])} for pt in sampled_full
            ]

        vertebrae_data.append(entry)

    # 影像轉 base64
    print("  Encoding image to base64...")
    image_b64 = image_to_base64_png(image)

    result = {
        'version': 'pkl-contour-1.0',
        'sampleId': sample_id,
        'sourcePkl': pkl_path.name,
        'spineType': spine_type,
        'imageInfo': {'width': w, 'height': h},
        'imageBase64': image_b64,
        'vertebrae': vertebrae_data,
        'maxPointsPerEndplate': max_points_per_endplate,
    }

    return result, image


def main():
    parser = argparse.ArgumentParser(
        description='PKL Mask → 終板輪廓點 JSON 轉換器')
    parser.add_argument('--input', '-i', required=True,
                        help='輸入 pkl 檔案或含 pkl 的資料夾')
    parser.add_argument('--output', '-o', default=None,
                        help='輸出 JSON 路徑 (預設: 同名 .contour.json)')
    parser.add_argument('--max-points', type=int, default=50,
                        help='每個終板最大輪廓點數 (預設: 50)')
    parser.add_argument('--no-image', action='store_true',
                        help='不輸出 PNG 影像檔 (預設會輸出)')
    parser.add_argument('--image-format', choices=['png', 'jpg'], default='png',
                        help='影像輸出格式 (預設: png)')

    args = parser.parse_args()
    input_path = Path(args.input)

    if input_path.is_dir():
        pkl_files = list(input_path.glob('*.pkl'))
        if not pkl_files:
            print(f"No .pkl files found in {input_path}")
            sys.exit(1)
    else:
        pkl_files = [input_path]

    for pkl_file in pkl_files:
        result, raw_image = process_single_pkl(pkl_file, args.max_points)

        if args.output and len(pkl_files) == 1:
            output_path = Path(args.output)
        else:
            output_path = pkl_file.with_suffix('.contour.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)

        # 檔案大小
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  JSON: {output_path} ({size_mb:.1f} MB)")

        # 輸出原始影像
        if not args.no_image and raw_image is not None:
            ext = '.' + args.image_format
            img_path = pkl_file.with_suffix(ext)
            if args.image_format == 'jpg':
                cv2.imwrite(str(img_path), raw_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(img_path), raw_image)
            img_size_mb = img_path.stat().st_size / (1024 * 1024)
            print(f"  Image: {img_path} ({img_size_mb:.1f} MB)")

    print(f"\nDone! {len(pkl_files)} file(s) converted.")
    print("Open pkl-contour-editor.html to edit and export.")


if __name__ == '__main__':
    main()
