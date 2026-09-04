#!/usr/bin/env python
"""舊版 L-spine 標註 → v2.3 慣例升級「提案」產生器。

背景：早期標註（v2.0 / 部分 v2.3）把 S1 標成上下緣、且不含 T12；現行慣例是
T12 下緣 + L1~L5 + S1 上緣。模型以現行慣例訓練，遇到舊版 GT 會多算一節 T12、
被判成 count mismatch 與大誤差，實際上是慣例不一致而非模型看錯。

本工具**不改動 Images/ 原檔**，只在 output 資料夾產生升級提案 JSON：
  - 保留醫師既有角點（不移動任何醫師標的點）
  - 移除 S1 下緣三點，S1 boundaryType 改為 'upper'
  - 補 T12 下緣三點：來源為 OOF 預測（`oof_predictions.json`），標記 pointSources
  - 補缺少的 middleSuperior / middleInferior：來源同上
  - 重新計算 metrics / discs / abnormalities
  - 加上 reviewProposal 區塊，說明每項變更；醫師審核後才可取代原檔

用法：
  python propose_legacy_upgrade.py \
      --oof-predictions endplate_training_data/runs/<run>/oof_eval/oof_predictions.json \
      --images-dir Images \
      --output-dir endplate_training_data/review_proposals/legacy_upgrade_YYYYMMDD
  未指定 --stem 時自動偵測：L-spine 且（無 T12 或 S1 含下緣）。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from spine_metrics import calculate_metrics, calculate_discs, detect_abnormalities

INFERIOR_KEYS = ('posteriorInferior', 'middleInferior', 'anteriorInferior')
SUPERIOR_KEYS = ('anteriorSuperior', 'middleSuperior', 'posteriorSuperior')
T12_KEYS = INFERIOR_KEYS
LSPINE_ORDER = {name: i for i, name in enumerate(['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'])}


def is_legacy_lspine(annotation: dict) -> bool:
    """L-spine 且（沒有 T12 或 S1 帶下緣點）即視為舊版慣例。"""
    if (annotation.get('spineType') or annotation.get('spine_type')) != 'L':
        return False
    names = [v.get('name') for v in annotation.get('vertebrae', [])]
    if 'S1' not in names:
        return False
    s1 = next(v for v in annotation['vertebrae'] if v.get('name') == 'S1')
    s1_has_inferior = any(k in s1.get('points', {}) for k in INFERIOR_KEYS)
    return ('T12' not in names) or s1_has_inferior


def find_legacy_stems(images_dir: Path, include_nhanes: bool = False) -> list[str]:
    """自動偵測舊版 L-spine 標註；預設略過 NHANES（2026-09-04 起暫停處理）。"""
    stems = []
    for json_path in sorted(images_dir.rglob('*.json')):
        stem = json_path.stem
        if stem.endswith('samp') or stem.endswith('ai') or stem.endswith('model'):
            continue
        if not include_nhanes and 'nhanes' in str(json_path).lower():
            continue
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get('vertebrae'), list) and is_legacy_lspine(data):
            stems.append(str(json_path.relative_to(images_dir).with_suffix('')).replace('\\', '/'))
    return stems


def load_oof_cases(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    cases = data['cases'] if isinstance(data, dict) else data
    return {c['case_id']: c for c in cases if c.get('status') == 'success'}


def _xy(point: dict) -> dict:
    return {'x': float(point['x']), 'y': float(point['y'])}


def t12_rejection_reason(pred_t12: dict | None, l1: dict | None) -> str | None:
    """模型預測的 T12 下緣是否幾何上合理（相對醫師標的 L1）。回傳拒絕原因，None 表示接受。

    規則（影像 y 向下為正）：
      - 三點齊全；
      - T12 下緣每一點都要在 L1 上緣之上，間距介於 L1 前高的 5% ~ 150%（椎間盤高度範圍）；
      - T12 下緣中心 x 與 L1 上緣中心 x 的偏移 < L1 上緣寬度的 50%。
    """
    if not pred_t12 or not all(k in pred_t12.get('points', {}) for k in T12_KEYS):
        return 'OOF 預測無完整 T12'
    if not l1:
        return '沒有 L1 可比對'
    lp = l1.get('points', {})
    if not all(k in lp for k in ('anteriorSuperior', 'posteriorSuperior', 'anteriorInferior')):
        return 'L1 角點不全'
    l1_height = abs(float(lp['anteriorInferior']['y']) - float(lp['anteriorSuperior']['y']))
    if l1_height <= 0:
        return 'L1 高度為零'
    l1_top_y = min(float(lp['anteriorSuperior']['y']), float(lp['posteriorSuperior']['y']))
    tp = pred_t12['points']
    for k in T12_KEYS:
        gap = l1_top_y - float(tp[k]['y'])
        if gap < 0.05 * l1_height:
            return f'模型 T12 {k} 不在 L1 之上（落在 L1 或更下方）'
        if gap > 1.5 * l1_height:
            return f'模型 T12 {k} 離 L1 太遠（>1.5 倍 L1 高）'
    l1_cx = (float(lp['anteriorSuperior']['x']) + float(lp['posteriorSuperior']['x'])) / 2
    l1_w = abs(float(lp['posteriorSuperior']['x']) - float(lp['anteriorSuperior']['x']))
    t12_cx = (float(tp['anteriorInferior']['x']) + float(tp['posteriorInferior']['x'])) / 2
    if l1_w > 0 and abs(t12_cx - l1_cx) > 0.5 * l1_w:
        return '模型 T12 水平位置與 L1 偏差過大'
    return None


def build_proposal(annotation: dict, oof_case: dict | None, stem: str,
                   source_path: str, oof_run: str) -> tuple[dict, list[str]]:
    """回傳 (提案 JSON, 變更說明列表)。不修改輸入物件。"""
    proposal = copy.deepcopy(annotation)
    changes: list[str] = []
    pred_by_name: dict[str, dict] = {}
    fold = None
    if oof_case:
        fold = oof_case.get('fold')
        for v in (oof_case.get('prediction') or {}).get('vertebrae', []):
            pred_by_name[v['name']] = v
    model_tag = f'model_oof_fold{fold}' if fold is not None else 'model_oof'

    vertebrae = proposal.get('vertebrae', [])
    names = [v.get('name') for v in vertebrae]
    bottom_up = bool(names) and names[0] == 'S1'

    for v in vertebrae:
        points = v.setdefault('points', {})
        sources = {k: 'physician' for k in points}
        v['pointSources'] = sources
        name = v.get('name')

        if name == 'S1':
            removed = [k for k in INFERIOR_KEYS if k in points]
            for k in removed:
                points.pop(k)
                sources.pop(k, None)
            if removed:
                changes.append(f'S1 移除下緣點 {removed}（新慣例 S1 只標上緣）')
            v['boundaryType'] = 'upper'

        # 缺中點：用兩角點的弦中點當起始值（比 OOF 模型點可靠；catastrophe 案例的模型點整片飄掉）。
        # 醫師審核時再拉到終板凹面。
        for mid, (a, p) in (('middleSuperior', ('anteriorSuperior', 'posteriorSuperior')),
                            ('middleInferior', ('anteriorInferior', 'posteriorInferior'))):
            if mid in points or a not in points or p not in points:
                continue
            points[mid] = {'x': (float(points[a]['x']) + float(points[p]['x'])) / 2,
                           'y': (float(points[a]['y']) + float(points[p]['y'])) / 2}
            sources[mid] = 'chord_midpoint'
            changes.append(f'{name} 補 {mid}（弦中點起始值，需醫師拉到終板凹面）')

    if 'T12' not in names:
        pred_t12 = pred_by_name.get('T12')
        l1 = next((v for v in vertebrae if v.get('name') == 'L1'), None)
        reason = t12_rejection_reason(pred_t12, l1)
        if reason:
            changes.append(f'缺 T12，未補（{reason}）；需醫師手標')
        else:
            t12 = {
                'name': 'T12',
                'boundaryType': 'lower',
                'points': {k: _xy(pred_t12['points'][k]) for k in T12_KEYS},
                'pointSources': {k: model_tag for k in T12_KEYS},
                'anteriorWedgingFracture': False,
                'biconcaveCompressionFracture': False,
                'crushDeformityFracture': False,
                'genantGrade': 0,
                'genantType': None,
            }
            vertebrae.insert(0, t12)
            changes.append(f'新增 T12 下緣三點（{model_tag}，已通過與 L1 的幾何合理性檢查，需醫師確認）')

    # 統一為由上而下（T12 → S1），與現行工具匯出及 calculate_discs 的假設一致
    vertebrae.sort(key=lambda v: LSPINE_ORDER.get(v.get('name'), 99))
    if bottom_up:
        changes.append('椎體順序由下而上改為由上而下（T12→S1）')
    proposal['vertebrae'] = vertebrae

    # 重新計算指標（會就地修改每個 vertebra）
    spine_type = proposal.get('spineType', 'L')
    for v in vertebrae:
        v.setdefault('anteriorWedgingFracture', False)
        try:
            calculate_metrics(v)
        except Exception as exc:  # noqa: BLE001 - 指標算不出來不該擋提案
            changes.append(f'{v.get("name")} metrics 重算失敗：{exc}')
    try:
        proposal['discs'] = calculate_discs(vertebrae, spine_type)
        proposal['abnormalities'] = detect_abnormalities(vertebrae, proposal['discs'], spine_type)
    except Exception as exc:  # noqa: BLE001
        changes.append(f'discs/abnormalities 重算失敗：{exc}')

    proposal['version'] = '2.3'
    proposal['endplatePointMode'] = 3
    info = proposal.setdefault('imageInfo', {})
    info.setdefault('filename', f'{Path(stem).name}.png')
    proposal['reviewProposal'] = {
        'status': 'proposed',
        'createdAt': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'sourceAnnotation': source_path,
        'sourceVersion': annotation.get('version'),
        'oofRun': oof_run,
        'oofFold': fold,
        'changes': changes,
        'note': '模型補的點只是提案，醫師審核（可用 spinal-annotation-web.html 載入調整）並匯出後，才可取代 Images/ 原檔。',
    }
    return proposal, changes


def render_overlay(image_path: Path, proposal: dict, out_path: Path) -> bool:
    """畫出提案：醫師點＝綠、模型補點＝橘（醫師審核用）。失敗回傳 False。"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    buf = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        return False
    h = img.shape[0]
    r = max(4, h // 400)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for v in proposal.get('vertebrae', []):
        pts = v.get('points', {})
        src = v.get('pointSources', {})
        for k, p in pts.items():
            color = (0, 200, 0) if src.get(k) == 'physician' else (0, 140, 255)
            cv2.circle(img, (int(p['x']), int(p['y'])), r, color, -1)
        if 'anteriorSuperior' in pts:
            anchor = pts['anteriorSuperior']
        elif 'anteriorInferior' in pts:
            anchor = pts['anteriorInferior']
        else:
            continue
        label = v['name'] + ('*' if any(s != 'physician' for s in src.values()) else '')
        cv2.putText(img, label, (int(anchor['x']) - 6 * r, int(anchor['y'])), font, r / 6, (255, 255, 255), max(1, r // 2))
    cv2.putText(img, 'green=physician  orange=model proposal (*=has model points)', (10, 30 + r), font, r / 5,
                (255, 255, 255), max(1, r // 2))
    ok, enc = cv2.imencode('.png', img)
    if not ok:
        return False
    enc.tofile(str(out_path))
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--oof-predictions', required=True)
    parser.add_argument('--images-dir', default='Images')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stem', action='append', help='指定 stem（可重複）；未指定則自動偵測舊版 L-spine')
    parser.add_argument('--include-nhanes', action='store_true', help='自動偵測時也納入 NHANES（預設略過）')
    parser.add_argument('--overlays', action='store_true', help='同時輸出 <stem>_overlay.png 供醫師快速目視')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    images_dir = Path(args.images_dir)
    oof_path = Path(args.oof_predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = args.stem or find_legacy_stems(images_dir, include_nhanes=args.include_nhanes)
    if not stems:
        print('沒有偵測到舊版 L-spine 標註。')
        return 0
    cases = load_oof_cases(oof_path)
    oof_run = str(oof_path.parent.parent.name)

    summary_lines = [f'# 舊版標註升級提案（{oof_run}）', '',
                     '| stem | 原版本 | OOF fold | 變更 |', '|---|---|---|---|']
    for stem in stems:
        src = images_dir / f'{stem}.json'
        annotation = json.loads(src.read_text(encoding='utf-8'))
        case = cases.get(Path(stem).name)
        if case is None:
            print(f'[warn] {stem}: OOF 預測中找不到，T12/middle 無法補')
        proposal, changes = build_proposal(annotation, case, stem, str(src).replace('\\', '/'), oof_run)
        dst = out_dir / f'{Path(stem).name}.json'
        dst.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{stem}: {len(changes)} 項變更 -> {dst}')
        if args.overlays:
            image = next((images_dir / f'{stem}{ext}' for ext in ('.png', '.jpg', '.jpeg')
                          if (images_dir / f'{stem}{ext}').exists()), None)
            if image and render_overlay(image, proposal, out_dir / f'{Path(stem).name}_overlay.png'):
                print(f'    overlay -> {Path(stem).name}_overlay.png')
            else:
                print('    [warn] overlay 產生失敗（找不到 png/jpg 或 cv2 不可用）')
        for c in changes:
            print(f'    - {c}')
        summary_lines.append(f"| {stem} | {annotation.get('version')} | {case.get('fold') if case else '-'} | "
                             + '<br>'.join(changes) + ' |')
    summary_lines += ['', '所有提案 JSON 都未寫回 `Images/`；醫師確認後另行覆蓋。']
    (out_dir / 'SUMMARY.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(f'摘要：{out_dir / "SUMMARY.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
