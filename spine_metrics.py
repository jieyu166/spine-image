#!/usr/bin/env python3
"""
脊椎指標計算模組 — 共用於 heatmap 模型與 SpineFM 推論
Spine Metrics Module — shared by heatmap model and SpineFM inference

骨折判斷依 Genant semi-quantitative classification:
  - Wedge:     anteriorHeight / posteriorHeight < 0.8
  - Biconcave: middleHeight   / posteriorHeight < 0.8
  - Crush:     posteriorHeight / anteriorHeight < 0.8

Genant grading:
  Grade 0: Normal
  Grade 1 (Mild):     20-25% loss  (ratio 0.75-0.80)
  Grade 2 (Moderate): 25-40% loss  (ratio 0.60-0.75)
  Grade 3 (Severe):   >40%  loss   (ratio < 0.60)
"""

import math

# ── Genant 閾值 ──
GENANT_THRESHOLD = 0.8     # ratio < 0.8 → 骨折
GENANT_MILD_MAX = 0.80
GENANT_MODERATE_MAX = 0.75
GENANT_SEVERE_MAX = 0.60


def _dist(p1, p2):
    """兩點距離"""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)


def genant_grade(ratio):
    """根據高度比值回傳 Genant grade (0, 1, 2, 3)"""
    if ratio >= GENANT_THRESHOLD:
        return 0
    elif ratio >= GENANT_MODERATE_MAX:
        return 1
    elif ratio >= GENANT_SEVERE_MAX:
        return 2
    else:
        return 3


def calculate_metrics(vertebra):
    """計算單一椎體的高度、骨折指標

    直接修改 vertebra dict（in-place）。
    支援 middleHeight（若有 middleSuperior/middleInferior 或由 mask 計算）。
    """
    pts = vertebra['points']
    bt = vertebra.get('boundaryType')

    # 邊界椎體無法計算高度
    if bt:
        vertebra['anteriorHeight'] = None
        vertebra['posteriorHeight'] = None
        vertebra['middleHeight'] = None
        vertebra['heightRatio'] = None
        vertebra['anteriorWedgingFracture'] = False
        vertebra['biconcaveCompressionFracture'] = False
        vertebra['crushDeformityFracture'] = False
        vertebra['genantGrade'] = 0
        vertebra['genantType'] = None
        return

    ant_sup = pts.get('anteriorSuperior')
    ant_inf = pts.get('anteriorInferior')
    post_sup = pts.get('posteriorSuperior')
    post_inf = pts.get('posteriorInferior')

    if not all([ant_sup, ant_inf, post_sup, post_inf]):
        vertebra['anteriorHeight'] = None
        vertebra['posteriorHeight'] = None
        vertebra['middleHeight'] = None
        vertebra['heightRatio'] = None
        vertebra['anteriorWedgingFracture'] = False
        vertebra['biconcaveCompressionFracture'] = False
        vertebra['crushDeformityFracture'] = False
        vertebra['genantGrade'] = 0
        vertebra['genantType'] = None
        return

    ant_h = _dist(ant_sup, ant_inf)
    post_h = _dist(post_sup, post_inf)

    # middleHeight: 優先用 middleSuperior/middleInferior（mask 中點）
    mid_sup = pts.get('middleSuperior')
    mid_inf = pts.get('middleInferior')
    if mid_sup and mid_inf:
        mid_h = _dist(mid_sup, mid_inf)
    else:
        # 退而求其次: 上下 endplate 中點距離
        mid_h = _dist(
            {'x': (ant_sup['x'] + post_sup['x']) / 2, 'y': (ant_sup['y'] + post_sup['y']) / 2},
            {'x': (ant_inf['x'] + post_inf['x']) / 2, 'y': (ant_inf['y'] + post_inf['y']) / 2},
        )

    # 高度比值
    wedge_ratio = ant_h / post_h if post_h > 0 else 1.0
    biconcave_ratio = mid_h / post_h if post_h > 0 else 1.0
    crush_ratio = post_h / ant_h if ant_h > 0 else 1.0

    # Genant 判斷
    is_wedge = wedge_ratio < GENANT_THRESHOLD
    is_biconcave = biconcave_ratio < GENANT_THRESHOLD
    is_crush = crush_ratio < GENANT_THRESHOLD

    # 取最嚴重的 grade
    ratios = []
    types = []
    if is_wedge:
        ratios.append(wedge_ratio)
        types.append('wedge')
    if is_biconcave:
        ratios.append(biconcave_ratio)
        types.append('biconcave')
    if is_crush:
        ratios.append(crush_ratio)
        types.append('crush')

    if ratios:
        worst_idx = ratios.index(min(ratios))
        grade = genant_grade(min(ratios))
        fracture_type = types[worst_idx]
    else:
        grade = 0
        fracture_type = None

    vertebra['anteriorHeight'] = float(ant_h)
    vertebra['posteriorHeight'] = float(post_h)
    vertebra['middleHeight'] = float(mid_h)
    vertebra['heightRatio'] = float(wedge_ratio)
    vertebra['anteriorWedgingFracture'] = bool(is_wedge)
    vertebra['biconcaveCompressionFracture'] = bool(is_biconcave)
    vertebra['crushDeformityFracture'] = bool(is_crush)
    vertebra['genantGrade'] = grade
    vertebra['genantType'] = fracture_type


def calculate_discs(vertebrae, spine_type=None):
    """計算相鄰椎體間的椎間盤指標

    Returns:
        list[dict]: disc metrics
    """
    discs = []

    for i in range(len(vertebrae) - 1):
        upper = vertebrae[i]
        lower = vertebrae[i + 1]

        if upper.get('boundaryType') == 'upper':
            continue
        if lower.get('boundaryType') == 'lower':
            continue

        upper_pts = upper['points']
        lower_pts = lower['points']

        u_ant_inf = upper_pts.get('anteriorInferior')
        u_post_inf = upper_pts.get('posteriorInferior')
        l_ant_sup = lower_pts.get('anteriorSuperior')
        l_post_sup = lower_pts.get('posteriorSuperior')

        if not all([u_ant_inf, u_post_inf, l_ant_sup, l_post_sup]):
            continue

        ant_h = _dist(l_ant_sup, u_ant_inf)
        post_h = _dist(l_post_sup, u_post_inf)

        discs.append({
            'level': f"{upper['name']}/{lower['name']}",
            'anteriorHeight': float(ant_h),
            'posteriorHeight': float(post_h),
            'middleHeight': float((ant_h + post_h) / 2),
        })

    return discs
