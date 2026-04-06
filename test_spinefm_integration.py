#!/usr/bin/env python3
"""
SpineFM Integration Tests
測試 mask → corner point 轉換 + Genant 骨折分級

不需要 SpineFM 模型權重即可執行 (使用合成 mask)。

Usage:
    python test_spinefm_integration.py
"""

import sys
import json
import numpy as np
import cv2

# ── 測試 spine_metrics ──
from spine_metrics import calculate_metrics, calculate_discs, genant_grade

# ── 測試 mask → corner point ──
from convert_pkl_to_contour import extract_contour, split_endplates


def mask_to_corner_points(mask, boundary_type=None):
    """局部複製 inference_spinefm.mask_to_corner_points (避免匯入 torch)"""
    contour = extract_contour(mask)
    if contour is None or len(contour) < 10:
        return None

    sup_pts, inf_pts = split_endplates(contour)
    points = {}

    if boundary_type != 'lower' and sup_pts is not None and len(sup_pts) >= 2:
        points['anteriorSuperior'] = {'x': float(sup_pts[0, 0]), 'y': float(sup_pts[0, 1])}
        points['posteriorSuperior'] = {'x': float(sup_pts[-1, 0]), 'y': float(sup_pts[-1, 1])}
        mid_idx = len(sup_pts) // 2
        points['middleSuperior'] = {'x': float(sup_pts[mid_idx, 0]), 'y': float(sup_pts[mid_idx, 1])}

    if boundary_type != 'upper' and inf_pts is not None and len(inf_pts) >= 2:
        points['anteriorInferior'] = {'x': float(inf_pts[0, 0]), 'y': float(inf_pts[0, 1])}
        points['posteriorInferior'] = {'x': float(inf_pts[-1, 0]), 'y': float(inf_pts[-1, 1])}
        mid_idx = len(inf_pts) // 2
        points['middleInferior'] = {'x': float(inf_pts[mid_idx, 0]), 'y': float(inf_pts[mid_idx, 1])}

    return points if points else None


def create_vertebra_mask(h, w, center_x, center_y, width, height):
    """建立橢圓形 vertebra-like mask (寬 > 高, 模擬椎體)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (center_x, center_y), (width // 2, height // 2), 0, 0, 360, 1, -1)
    return mask


def create_tilted_mask(h, w, center_x, center_y, width, height, angle_deg):
    """建立旋轉矩形 mask"""
    mask = np.zeros((h, w), dtype=np.uint8)
    rect = ((center_x, center_y), (width, height), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [box], 1)
    return mask


def test_mask_to_corners_ellipse():
    """測試: 橢圓 mask → 角點提取 + 上下 endplate 正確分離"""
    print("Test 1: Ellipse mask (vertebra-like) -> corners")

    # 橢圓中心 (250, 250), 寬 300, 高 100
    mask = create_vertebra_mask(500, 500, 250, 250, 300, 100)
    points = mask_to_corner_points(mask, boundary_type=None)

    assert points is not None, "Should extract points from ellipse mask"
    assert 'anteriorSuperior' in points
    assert 'posteriorSuperior' in points
    assert 'anteriorInferior' in points
    assert 'posteriorInferior' in points
    assert 'middleSuperior' in points
    assert 'middleInferior' in points

    # 角點 X 排列: anterior (左) < posterior (右)
    assert points['anteriorSuperior']['x'] < points['posteriorSuperior']['x'], \
        "anteriorSuperior should be left of posteriorSuperior"
    assert points['anteriorInferior']['x'] < points['posteriorInferior']['x'], \
        "anteriorInferior should be left of posteriorInferior"

    # 上終板的中間 Y 應 < 下終板的中間 Y
    sup_mid_y = points['middleSuperior']['y']
    inf_mid_y = points['middleInferior']['y']
    assert sup_mid_y < inf_mid_y, \
        f"Superior midpoint Y ({sup_mid_y}) should be above inferior midpoint Y ({inf_mid_y})"

    # 對橢圓，角點 (leftmost/rightmost) 是共用的，所以 anterior/posterior height ≈ 0
    # 但 middleHeight 應捕捉實際椎體高度
    from spine_metrics import _dist
    mid_h = _dist(points['middleSuperior'], points['middleInferior'])
    assert mid_h > 50, f"Middle height should capture ellipse height (~100), got {mid_h}"

    print("  PASSED")


def test_mask_to_corners_tilted():
    """測試: 旋轉矩形 mask → 仍能提取角點且不 crash"""
    print("Test 2: Tilted rectangular mask -> corners")

    mask = create_tilted_mask(500, 500, 250, 250, 300, 100, 15)
    points = mask_to_corner_points(mask, boundary_type=None)

    assert points is not None, "Should extract points from tilted mask"
    assert 'anteriorSuperior' in points
    assert 'posteriorInferior' in points

    # middleSuperior Y 應 < middleInferior Y (即使傾斜)
    sup_mid_y = points['middleSuperior']['y']
    inf_mid_y = points['middleInferior']['y']
    assert sup_mid_y < inf_mid_y, \
        f"Superior midpoint Y ({sup_mid_y}) should be above inferior midpoint Y ({inf_mid_y})"

    print("  PASSED")


def test_boundary_vertebra():
    """測試: 邊界椎體只回傳單側角點"""
    print("Test 3: Boundary vertebra (upper → only superior)")

    mask = create_vertebra_mask(500, 500, 250, 250, 300, 100)

    # boundary='upper' → 只有 superior (S1 只有上終板)
    points = mask_to_corner_points(mask, boundary_type='upper')
    assert points is not None
    assert 'anteriorSuperior' in points
    assert 'posteriorSuperior' in points
    assert 'anteriorInferior' not in points
    assert 'posteriorInferior' not in points

    # boundary='lower' → 只有 inferior (T12 只有下終板)
    mask2 = create_vertebra_mask(500, 500, 250, 250, 300, 100)
    points = mask_to_corner_points(mask2, boundary_type='lower')
    assert points is not None
    assert 'anteriorInferior' in points
    assert 'posteriorInferior' in points
    assert 'anteriorSuperior' not in points
    assert 'posteriorSuperior' not in points

    print("  PASSED")


def test_genant_classification():
    """測試: Genant 分級邏輯"""
    print("Test 4: Genant classification")

    assert genant_grade(0.85) == 0, "0.85 should be normal"
    assert genant_grade(0.80) == 0, "0.80 should be normal (threshold is <0.8)"
    assert genant_grade(0.78) == 1, "0.78 should be mild"
    assert genant_grade(0.70) == 2, "0.70 should be moderate"
    assert genant_grade(0.55) == 3, "0.55 should be severe"

    print("  PASSED")


def test_calculate_metrics_wedge():
    """測試: Wedge fracture detection"""
    print("Test 5: Wedge fracture (anterior height loss)")

    vertebra = {
        'name': 'L1',
        'boundaryType': None,
        'points': {
            'anteriorSuperior': {'x': 100, 'y': 100},
            'posteriorSuperior': {'x': 400, 'y': 100},
            'posteriorInferior': {'x': 400, 'y': 300},  # post height = 200
            'anteriorInferior': {'x': 100, 'y': 230},   # ant height = 130 (ratio 0.65)
        },
    }

    calculate_metrics(vertebra)

    assert vertebra['anteriorWedgingFracture'] == True, "Should detect wedge fracture"
    assert vertebra['genantGrade'] == 2, f"Should be Grade 2, got {vertebra['genantGrade']}"
    assert vertebra['genantType'] == 'wedge', f"Should be wedge, got {vertebra['genantType']}"

    print("  PASSED")


def test_calculate_metrics_crush():
    """測試: Crush fracture detection"""
    print("Test 6: Crush fracture (posterior height loss)")

    vertebra = {
        'name': 'L2',
        'boundaryType': None,
        'points': {
            'anteriorSuperior': {'x': 100, 'y': 100},
            'posteriorSuperior': {'x': 400, 'y': 100},
            'posteriorInferior': {'x': 400, 'y': 230},  # post height = 130
            'anteriorInferior': {'x': 100, 'y': 300},   # ant height = 200
        },
    }

    calculate_metrics(vertebra)

    assert vertebra['crushDeformityFracture'] == True, "Should detect crush fracture"
    assert vertebra['genantType'] == 'crush', f"Should be crush, got {vertebra['genantType']}"

    print("  PASSED")


def test_calculate_metrics_biconcave():
    """測試: Biconcave fracture detection"""
    print("Test 7: Biconcave fracture (middle height loss)")

    vertebra = {
        'name': 'L3',
        'boundaryType': None,
        'points': {
            'anteriorSuperior': {'x': 100, 'y': 100},
            'posteriorSuperior': {'x': 400, 'y': 100},
            'posteriorInferior': {'x': 400, 'y': 300},  # post height = 200
            'anteriorInferior': {'x': 100, 'y': 300},   # ant height = 200
            # middle points: 中間凹陷
            'middleSuperior': {'x': 250, 'y': 130},
            'middleInferior': {'x': 250, 'y': 270},     # mid height = 140 (ratio 0.7)
        },
    }

    calculate_metrics(vertebra)

    assert vertebra['biconcaveCompressionFracture'] == True, "Should detect biconcave fracture"
    assert vertebra['genantType'] == 'biconcave', f"Should be biconcave, got {vertebra['genantType']}"
    assert vertebra['genantGrade'] == 2, f"Should be Grade 2, got {vertebra['genantGrade']}"

    print("  PASSED")


def test_calculate_metrics_normal():
    """測試: 正常椎體無骨折"""
    print("Test 8: Normal vertebra (no fracture)")

    vertebra = {
        'name': 'L4',
        'boundaryType': None,
        'points': {
            'anteriorSuperior': {'x': 100, 'y': 100},
            'posteriorSuperior': {'x': 400, 'y': 100},
            'posteriorInferior': {'x': 400, 'y': 300},
            'anteriorInferior': {'x': 100, 'y': 300},
        },
    }

    calculate_metrics(vertebra)

    assert vertebra['anteriorWedgingFracture'] == False
    assert vertebra['biconcaveCompressionFracture'] == False
    assert vertebra['crushDeformityFracture'] == False
    assert vertebra['genantGrade'] == 0

    print("  PASSED")


def test_calculate_discs():
    """測試: 椎間盤指標計算"""
    print("Test 9: Disc metrics calculation")

    vertebrae = [
        {
            'name': 'L1',
            'boundaryType': None,
            'points': {
                'anteriorSuperior': {'x': 100, 'y': 100},
                'posteriorSuperior': {'x': 400, 'y': 100},
                'posteriorInferior': {'x': 400, 'y': 300},
                'anteriorInferior': {'x': 100, 'y': 300},
            },
        },
        {
            'name': 'L2',
            'boundaryType': None,
            'points': {
                'anteriorSuperior': {'x': 100, 'y': 320},
                'posteriorSuperior': {'x': 400, 'y': 320},
                'posteriorInferior': {'x': 400, 'y': 520},
                'anteriorInferior': {'x': 100, 'y': 520},
            },
        },
    ]

    discs = calculate_discs(vertebrae, 'L')

    assert len(discs) == 1, f"Should have 1 disc, got {len(discs)}"
    assert discs[0]['level'] == 'L1/L2'
    assert abs(discs[0]['anteriorHeight'] - 20.0) < 1.0, \
        f"Anterior disc height should be ~20, got {discs[0]['anteriorHeight']}"

    print("  PASSED")


def test_json_output_format():
    """測試: JSON 輸出格式 v2.3 相容"""
    print("Test 10: JSON output format (v2.3 compatible)")

    # 模擬 inference 結果
    vertebrae = [
        {
            'name': 'L1',
            'boundaryType': None,
            'hasMiddlePoints': True,
            'points': {
                'anteriorSuperior': {'x': 100.5, 'y': 100.2},
                'posteriorSuperior': {'x': 400.1, 'y': 102.3},
                'posteriorInferior': {'x': 395.0, 'y': 299.8},
                'anteriorInferior': {'x': 105.0, 'y': 298.5},
                'middleSuperior': {'x': 250.3, 'y': 101.2},
                'middleInferior': {'x': 250.0, 'y': 299.1},
            },
        }
    ]

    for v in vertebrae:
        calculate_metrics(v)

    discs = calculate_discs(vertebrae, 'L')

    # 檢查 JSON 序列化
    json_data = {
        'version': '2.3',
        'spineType': 'L',
        'backend': 'SpineFM',
        'imageInfo': {'width': 512, 'height': 512},
        'vertebrae': vertebrae,
        'discs': discs,
    }

    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
    parsed = json.loads(json_str)

    assert parsed['version'] == '2.3'
    assert parsed['backend'] == 'SpineFM'
    assert len(parsed['vertebrae']) == 1
    assert 'anteriorHeight' in parsed['vertebrae'][0]
    assert 'genantGrade' in parsed['vertebrae'][0]

    print("  PASSED")


def main():
    print("=" * 60)
    print("SpineFM Integration Tests")
    print("=" * 60)
    print()

    tests = [
        test_mask_to_corners_ellipse,
        test_mask_to_corners_tilted,
        test_boundary_vertebra,
        test_genant_classification,
        test_calculate_metrics_wedge,
        test_calculate_metrics_crush,
        test_calculate_metrics_biconcave,
        test_calculate_metrics_normal,
        test_calculate_discs,
        test_json_output_format,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
