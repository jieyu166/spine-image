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


LISTHESIS_THRESHOLD = 10.0  # % of vertebral body width
DISC_HEIGHT_RATIO = 0.60    # disc height < 60% of neighbor avg → abnormal


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


def _wedge_angle(upper_endplate, lower_endplate):
    """計算上下 endplate 之間的 wedge angle (度)

    與 web viewer 的計算方式一致：
    angle = |atan2(post.y - ant.y, post.x - ant.x)| 差值

    Args:
        upper_endplate: [anterior, middle, posterior] 點陣列
        lower_endplate: [anterior, middle, posterior] 點陣列
    Returns:
        float: wedge angle in degrees (0-90)
    """
    u_ant, u_post = upper_endplate[0], upper_endplate[-1]
    l_ant, l_post = lower_endplate[0], lower_endplate[-1]

    upper_angle = math.atan2(u_post['y'] - u_ant['y'], u_post['x'] - u_ant['x'])
    lower_angle = math.atan2(l_post['y'] - l_ant['y'], l_post['x'] - l_ant['x'])

    angle_deg = abs(upper_angle - lower_angle) * 180 / math.pi
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    return angle_deg


def _check_disc_overlap(upper_endplate, lower_endplate):
    """檢查 disc overlap — lower endplate Y < upper endplate Y (影像座標)

    Returns:
        bool: True if overlap detected
    """
    for u_pt, l_pt in zip(upper_endplate, lower_endplate):
        if l_pt['y'] < u_pt['y'] - 1:
            return True
    return False


def calculate_discs(vertebrae, spine_type=None):
    """計算相鄰椎體間的椎間盤指標

    輸出格式與 web viewer (spinal-annotation-web.html) 一致：
    - upperEndplate / lowerEndplate: 3 點陣列 [anterior, middle, posterior]
    - anteriorHeight / middleHeight / posteriorHeight
    - wedgeAngle (度)
    - overlap (bool)

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
        u_mid_inf = upper_pts.get('middleInferior')
        l_ant_sup = lower_pts.get('anteriorSuperior')
        l_post_sup = lower_pts.get('posteriorSuperior')
        l_mid_sup = lower_pts.get('middleSuperior')

        if not all([u_ant_inf, u_post_inf, l_ant_sup, l_post_sup]):
            continue

        # 若缺 middle 點，用 anterior/posterior 中點
        if not u_mid_inf:
            u_mid_inf = {
                'x': (u_ant_inf['x'] + u_post_inf['x']) / 2,
                'y': (u_ant_inf['y'] + u_post_inf['y']) / 2,
            }
        if not l_mid_sup:
            l_mid_sup = {
                'x': (l_ant_sup['x'] + l_post_sup['x']) / 2,
                'y': (l_ant_sup['y'] + l_post_sup['y']) / 2,
            }

        upper_endplate = [u_ant_inf, u_mid_inf, u_post_inf]
        lower_endplate = [l_ant_sup, l_mid_sup, l_post_sup]

        ant_h = _dist(l_ant_sup, u_ant_inf)
        post_h = _dist(l_post_sup, u_post_inf)
        mid_h = _dist(l_mid_sup, u_mid_inf)

        wedge = _wedge_angle(upper_endplate, lower_endplate)
        overlap = _check_disc_overlap(upper_endplate, lower_endplate)

        discs.append({
            'level': f"{upper['name']}/{lower['name']}",
            'upperEndplate': [
                {'x': p['x'], 'y': p['y']} for p in upper_endplate
            ],
            'lowerEndplate': [
                {'x': p['x'], 'y': p['y']} for p in lower_endplate
            ],
            'anteriorHeight': float(ant_h),
            'posteriorHeight': float(post_h),
            'middleHeight': float(mid_h),
            'wedgeAngle': float(wedge),
            'overlap': overlap,
        })

    return discs


def detect_listhesis(vertebrae):
    """偵測椎體滑脫 (spondylolisthesis)

    比較相鄰椎體 posterior edge X 位移 / 下方椎體全寬。
    與 web viewer checkListhesis() 邏輯一致。

    Returns:
        list[dict]: [{level, type, shift}, ...]
    """
    results = []
    for i in range(len(vertebrae) - 1):
        upper = vertebrae[i]
        lower = vertebrae[i + 1]

        upper_pts = upper['points']
        lower_pts = lower['points']

        # upper 的 posteriorInferior vs lower 的 posteriorSuperior
        u_post_inf = upper_pts.get('posteriorInferior')
        l_post_sup = lower_pts.get('posteriorSuperior')
        if not u_post_inf or not l_post_sup:
            continue

        x_shift = u_post_inf['x'] - l_post_sup['x']

        # 下方椎體全寬 (所有 points 的 X 範圍)
        all_x = [p['x'] for p in lower_pts.values() if isinstance(p, dict) and 'x' in p]
        if not all_x:
            continue
        width = max(all_x) - min(all_x)
        if width == 0:
            continue

        shift_pct = abs(x_shift) / width * 100
        if shift_pct > LISTHESIS_THRESHOLD:
            # x_shift > 0: upper posterior 更往右 (posterior方向) = retrolisthesis
            # x_shift < 0: upper posterior 更往左 (anterior方向) = anterolisthesis
            slip_type = 'Retrolisthesis' if x_shift > 0 else 'Anterolisthesis'
            results.append({
                'level': f"{upper['name']} on {lower['name']}",
                'type': slip_type,
                'shift': f"{shift_pct:.1f}",
            })

    return results


def detect_disc_overlap(discs):
    """從 disc 列表中找出有 overlap 的 disc levels

    Returns:
        list[str]: overlap disc level names
    """
    return [d['level'] for d in discs if d.get('overlap', False)]


def detect_height_progression_issues(discs, spine_type='L'):
    """偵測椎間盤高度遞進異常

    Lumbar: 各 disc 高度不應 < 相鄰 disc 平均的 60% (排除 L5/S1)
    與 web viewer checkHeightProgression() 邏輯一致。

    Returns:
        list[dict]: [{level, issue, severity}, ...]
    """
    results = []
    if len(discs) < 3:
        return results

    # 使用 posteriorHeight 作為比較基準
    heights = [(d['level'], d['posteriorHeight']) for d in discs]

    for i in range(len(heights)):
        level, h = heights[i]

        # L5/S1 本身就窄，不列入偵測
        if 'S1' in level:
            continue

        # 鄰居平均
        neighbors = []
        if i > 0:
            neighbors.append(heights[i - 1][1])
        if i < len(heights) - 1 and 'S1' not in heights[i + 1][0]:
            neighbors.append(heights[i + 1][1])

        if not neighbors:
            continue

        avg_neighbor = sum(neighbors) / len(neighbors)
        if avg_neighbor > 0 and h < avg_neighbor * DISC_HEIGHT_RATIO:
            results.append({
                'level': level,
                'issue': '椎間盤高度遞進異常',
                'severity': 'warning',
            })

    return results


def detect_abnormalities(vertebrae, discs, spine_type='L'):
    """彙整所有異常偵測結果

    Returns:
        dict: 與 web viewer 輸出格式一致的 abnormalities 物件
    """
    compression = []
    biconcave = []

    for v in vertebrae:
        if v.get('boundaryType'):
            continue
        if v.get('anteriorWedgingFracture'):
            compression.append({'name': v['name'], 'type': 'anteriorWedging'})
        elif v.get('crushDeformityFracture'):
            compression.append({'name': v['name'], 'type': 'crushDeformity'})
        if v.get('biconcaveCompressionFracture'):
            biconcave.append({'name': v['name']})

    return {
        'compressionFractures': compression,
        'biconcaveCompression': biconcave,
        'listhesis': detect_listhesis(vertebrae),
        'discOverlap': detect_disc_overlap(discs),
        'heightProgressionIssues': detect_height_progression_issues(discs, spine_type),
    }
