#!/usr/bin/env python3
"""
將 pkl-contour-1.0 格式的 .contour.json 批次轉換為 v2.3 corner-point JSON

.contour.json 格式:
  - 每個椎體有 superiorEndplate / inferiorEndplate (各 50 點，沿 endplate 從 anterior→posterior)
  - 上下 endplate 共用端點 (contour 最左/最右點)

轉換邏輯:
  - 上下 endplate 曲線共用端點 (contour 左/右極值)
  - 前後角點: 在前 25% / 後 25% 區段取 Y 極值 (sup: min Y, inf: max Y)
  - 中點: 在兩角點 X 中點附近搜尋 endplate 上的 Y 極值點
  - Boundary 椎體 (只有單側 endplate) 依現有邏輯處理

用法:
  python convert_contour_to_corners.py --input planning/nhanes_orig/ --output planning/nhanes_v23/
  python convert_contour_to_corners.py --input file.contour.json --output output.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 重用 spine_metrics 計算
from spine_metrics import calculate_metrics, calculate_discs, detect_abnormalities


def _sample_curve_at_x(pts, target_x, mode='superior', window=3):
    """在曲線上找 X 最接近 target_x 的點，取附近 window 範圍的 Y 極值"""
    best_idx = min(range(len(pts)), key=lambda i: abs(pts[i]['x'] - target_x))
    lo = max(0, best_idx - window)
    hi = min(len(pts), best_idx + window + 1)
    region = pts[lo:hi]
    if mode == 'superior':
        p = min(region, key=lambda p: p['y'])
    else:
        p = max(region, key=lambda p: p['y'])
    return {'x': p['x'], 'y': p['y']}


def extract_corners_paired(sup_pts, inf_pts):
    """從配對的上下 endplate 曲線提取 6 個角點 (X 對齊)

    在上下 endplate 的共同 X 範圍內，於 20%/50%/80% 位置取樣。
    Superior 取 min Y，Inferior 取 max Y → 確保角點 X 對齊、高度合理。

    Returns:
        dict with anteriorSuperior, middleSuperior, posteriorSuperior,
              anteriorInferior, middleInferior, posteriorInferior
    """
    # 共同 X 範圍 (排除頭尾 5% 的 edge 點)
    n = len(sup_pts)
    trim = max(n // 20, 1)
    sup_trimmed = sup_pts[trim:-trim] if n > 2 * trim else sup_pts
    inf_trimmed = inf_pts[trim:-trim] if n > 2 * trim else inf_pts

    x_min = max(min(p['x'] for p in sup_trimmed), min(p['x'] for p in inf_trimmed))
    x_max = min(max(p['x'] for p in sup_trimmed), max(p['x'] for p in inf_trimmed))
    x_span = x_max - x_min
    if x_span <= 0:
        x_min = min(p['x'] for p in sup_pts)
        x_max = max(p['x'] for p in sup_pts)
        x_span = x_max - x_min

    # 三個取樣位置 (20%, 50%, 80%)
    x_ant = x_min + x_span * 0.20
    x_mid = x_min + x_span * 0.50
    x_post = x_min + x_span * 0.80

    return {
        'anteriorSuperior': _sample_curve_at_x(sup_pts, x_ant, 'superior'),
        'middleSuperior': _sample_curve_at_x(sup_pts, x_mid, 'superior'),
        'posteriorSuperior': _sample_curve_at_x(sup_pts, x_post, 'superior'),
        'anteriorInferior': _sample_curve_at_x(inf_pts, x_ant, 'inferior'),
        'middleInferior': _sample_curve_at_x(inf_pts, x_mid, 'inferior'),
        'posteriorInferior': _sample_curve_at_x(inf_pts, x_post, 'inferior'),
    }


def extract_corner_from_single_endplate(pts, mode='superior'):
    """從單一 endplate 曲線 (boundary 椎體) 提取 3 個角點"""
    n = len(pts)
    if n < 6:
        return pts[0], pts[n // 2], pts[-1]

    trim = max(n // 20, 1)
    trimmed = pts[trim:-trim] if n > 2 * trim else pts
    x_min = min(p['x'] for p in trimmed)
    x_max = max(p['x'] for p in trimmed)
    x_span = x_max - x_min

    x_ant = x_min + x_span * 0.20
    x_mid = x_min + x_span * 0.50
    x_post = x_min + x_span * 0.80

    anterior = _sample_curve_at_x(pts, x_ant, mode)
    middle = _sample_curve_at_x(pts, x_mid, mode)
    posterior = _sample_curve_at_x(pts, x_post, mode)

    return anterior, middle, posterior


def convert_vertebra(v_contour):
    """將 contour 格式的單一椎體轉換為 v2.3 corner-point 格式"""
    name = v_contour['name']
    bt = v_contour.get('boundaryType')
    sup_pts = v_contour.get('superiorEndplate', [])
    inf_pts = v_contour.get('inferiorEndplate', [])

    points = {}

    if sup_pts and inf_pts:
        # 非 boundary 椎體: 配對提取確保 X 對齊
        points = extract_corners_paired(sup_pts, inf_pts)
    elif sup_pts:
        ant, mid, post = extract_corner_from_single_endplate(sup_pts, 'superior')
        points['anteriorSuperior'] = ant
        points['middleSuperior'] = mid
        points['posteriorSuperior'] = post
    elif inf_pts:
        ant, mid, post = extract_corner_from_single_endplate(inf_pts, 'inferior')
        points['anteriorInferior'] = ant
        points['middleInferior'] = mid
        points['posteriorInferior'] = post

    vertebra = {
        'name': name,
        'points': points,
    }
    if bt:
        vertebra['boundaryType'] = bt

    return vertebra


def convert_file(contour_path, output_path=None, include_image=False):
    """將單一 .contour.json 轉換為 v2.3 JSON

    Args:
        contour_path: 輸入 .contour.json 路徑
        output_path: 輸出 .json 路徑 (None = 自動命名)
        include_image: 是否保留 base64 影像

    Returns:
        dict: v2.3 JSON 資料
    """
    with open(contour_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    spine_type = data.get('spineType', 'L')
    sample_id = data.get('sampleId', '')

    # 轉換椎體
    vertebrae = []
    for v in data.get('vertebrae', []):
        vertebrae.append(convert_vertebra(v))

    # 計算指標
    for v in vertebrae:
        calculate_metrics(v)

    discs = calculate_discs(vertebrae, spine_type)
    abnormalities = detect_abnormalities(vertebrae, discs, spine_type)

    # 組裝 v2.3 輸出
    result = {
        'version': '2.3',
        'endplatePointMode': '6-point',
        'spineType': spine_type,
        'sampleId': sample_id,
        'sourcePkl': data.get('sourcePkl', ''),
        'vertebrae': vertebrae,
        'discs': discs,
        'abnormalities': abnormalities,
    }

    # 影像資訊
    if data.get('imageInfo'):
        result['imageInfo'] = data['imageInfo']
    if include_image and data.get('imageBase64'):
        result['imageBase64'] = data['imageBase64']

    # 寫入檔案
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def batch_convert(input_dir, output_dir, include_image=False):
    """批次轉換目錄下所有 .contour.json"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob('*.contour.json'))
    if not files:
        print(f'No .contour.json files found in {input_dir}')
        return

    total = len(files)
    success = 0
    errors = []

    for i, f in enumerate(files, 1):
        # 輸出檔名: 去掉 .contour 後綴
        out_name = f.stem.replace('.contour', '') + '.json'
        out_file = output_path / out_name

        try:
            result = convert_file(str(f), str(out_file), include_image)
            n_vert = len(result['vertebrae'])
            n_disc = len(result['discs'])
            fractures = len(result['abnormalities']['compressionFractures'])
            print(f'[{i}/{total}] {f.name} -> {out_name}  '
                  f'({n_vert} vertebrae, {n_disc} discs, {fractures} fractures)')
            success += 1
        except Exception as e:
            errors.append((f.name, str(e)))
            print(f'[{i}/{total}] {f.name} ERROR: {e}')

    print(f'\nDone: {success}/{total} converted, {len(errors)} errors')
    if errors:
        print('Errors:')
        for name, err in errors:
            print(f'  {name}: {err}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert .contour.json (pkl-contour-1.0) to v2.3 corner-point JSON')
    parser.add_argument('--input', '-i', required=True,
                        help='Input .contour.json file or directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Output .json file or directory')
    parser.add_argument('--include-image', action='store_true',
                        help='Include base64 image in output')
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        batch_convert(args.input, args.output, args.include_image)
    elif input_path.is_file():
        result = convert_file(args.input, args.output, args.include_image)
        n = len(result['vertebrae'])
        print(f'Converted: {n} vertebrae, {len(result["discs"])} discs')
    else:
        print(f'Input not found: {args.input}')
        sys.exit(1)


if __name__ == '__main__':
    main()
