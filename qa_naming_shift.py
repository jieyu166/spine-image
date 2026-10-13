#!/usr/bin/env python
"""命名位移 QA：找出「模型與 GT 的椎體命名整條差一節」的案例，交醫師裁定。

背景（2026-09-22）：新增 27 張腰椎中，有數張模型的 T12 落在 GT 的 L1、模型的 S1 落在
GT 的 S1 下方一節，整條命名差一級。醫師逐案確認後 **4 例 GT 全部正確，是模型把薦椎當成
一節腰椎**（S1 椎體清楚可辨時模型會多數一節）。所以這支工具抓到的是「需要人看」的案例：
可能是標註命名錯（count 正確但每點都差一節，毒性大），也可能是模型錯；由醫師裁定。

判定方式：對每個案例，比較三種對應下 GT 與預測椎體「質心距離」的平均：
  same   : 同名對同名
  gt_up  : 預測第 k-1 節 對 GT 第 k 節（模型命名比 GT 高一節，例：模型 T12 = GT L1）
  gt_down: 預測第 k+1 節 對 GT 第 k 節
若位移對應的距離 < same 的 ratio 倍（預設 0.5），列為疑似命名位移。工具不判定誰對：
多數是模型錯（薦椎當腰椎），少數可能是 GT 位移或移行椎（lumbarized S1 / L6），由醫師看 S1 決定。

用法：
  python qa_naming_shift.py --oof-predictions <run>/oof_eval/oof_predictions.json \
      [--ratio 0.5] [--output-dir <run>/oof_eval/naming_shift_review] [--crops]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

ORDER = {
    'L': ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'],
    'C': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2'],
}


def centroid(vertebra: dict) -> tuple[float, float] | None:
    points = vertebra.get('points') or {}
    if not points:
        return None
    return (st.mean(float(p['x']) for p in points.values()), st.mean(float(p['y']) for p in points.values()))


def shifted_distance(gt: list[dict], pred: list[dict], shift: int, order: list[str]) -> tuple[float | None, int]:
    """預測索引 = GT 索引 + shift 時的平均質心距離；配對數 < 3 回傳 None。"""
    index = {name: i for i, name in enumerate(order)}
    pred_by_index = {index[v['name']]: centroid(v) for v in pred if v.get('name') in index}
    distances = []
    for v in gt:
        if v.get('name') not in index:
            continue
        c = centroid(v)
        p = pred_by_index.get(index[v['name']] + shift)
        if c and p:
            distances.append(((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2) ** 0.5)
    if len(distances) < 3:
        return None, len(distances)
    return st.mean(distances), len(distances)


def analyze_case(case: dict, ratio: float = 0.5) -> dict | None:
    if case.get('status') != 'success':
        return None
    gt = (case.get('ground_truth') or {}).get('vertebrae') or []
    pred = (case.get('prediction') or {}).get('vertebrae') or []
    order = ORDER.get(case.get('spine_type', 'L'), ORDER['L'])
    same, n = shifted_distance(gt, pred, 0, order)
    if same is None:
        return None
    up, _ = shifted_distance(gt, pred, -1, order)
    down, _ = shifted_distance(gt, pred, +1, order)
    verdict = 'ok'
    if up is not None and up < ratio * same:
        verdict = 'pred_one_level_higher'   # 模型 T12 = GT L1（模型多數了一節，或 GT 整條高一節）
    elif down is not None and down < ratio * same:
        verdict = 'pred_one_level_lower'
    return {
        'case_id': case.get('case_id'), 'spine_type': case.get('spine_type'), 'fold': case.get('fold'),
        'image_path': case.get('image_path'), 'n_matched': n,
        'd_same': round(same, 1), 'd_gt_up': None if up is None else round(up, 1),
        'd_gt_down': None if down is None else round(down, 1), 'verdict': verdict,
    }


def render_crop(case: dict, out_path: Path) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    image_path = case['image_path']
    buf = np.fromfile(image_path, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if im is None:
        return False
    H, W = im.shape[:2]
    rad = max(3, H // 500)

    def draw(verts, color, dx):
        for v in verts:
            pts = v.get('points') or {}
            for p in pts.values():
                cv2.circle(im, (int(p['x']), int(p['y'])), rad, color, -1)
            anchor = pts.get('anteriorSuperior') or pts.get('anteriorInferior')
            if anchor:
                cv2.putText(im, v['name'], (int(anchor['x']) + dx, int(anchor['y'])), cv2.FONT_HERSHEY_SIMPLEX, rad / 3, color, max(1, rad // 2))

    gt = case['ground_truth']['vertebrae']
    draw(gt, (0, 220, 0), -12 * rad)
    draw(case['prediction']['vertebrae'], (0, 140, 255), 4 * rad)
    cv2.putText(im, f"{case['case_id']}  green=GT  orange=model OOF", (10, 30 + rad), cv2.FONT_HERSHEY_SIMPLEX, rad / 4, (255, 255, 255), max(1, rad // 2))
    ok, enc = cv2.imencode('.png', im)
    if ok:
        enc.tofile(str(out_path))
    return bool(ok)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--oof-predictions', required=True, type=Path)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--output-dir', type=Path, help='寫 naming_shift_report.md / .json；加 --crops 另存 overlay')
    parser.add_argument('--crops', action='store_true')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    data = json.loads(args.oof_predictions.read_text(encoding='utf-8'))
    cases = data['cases'] if isinstance(data, dict) else data
    results = [r for r in (analyze_case(c, args.ratio) for c in cases) if r]
    flagged = [r for r in results if r['verdict'] != 'ok']

    lines = ['# 標註命名位移 QA', '',
             f'來源：`{args.oof_predictions}`；判定門檻 ratio={args.ratio}；分析 {len(results)} 例，疑似 {len(flagged)} 例。', '',
             '| case | type | fold | d(same) | d(模型高一節) | d(模型低一節) | 判定 |', '|---|---|---:|---:|---:|---:|---|']
    for r in sorted(results, key=lambda r: (r['verdict'] == 'ok', -r['d_same'])):
        if r['verdict'] == 'ok' and r['d_same'] < 150:
            continue
        lines.append(f"| {r['case_id']} | {r['spine_type']} | {r['fold']} | {r['d_same']} | {r['d_gt_up']} | {r['d_gt_down']} | {r['verdict']} |")
    lines += ['', '`pred_one_level_higher`：模型 T12 落在 GT L1、模型 S1 在 GT S1 下方一節。09-22 醫師確認此型態 4 例皆為模型錯（薦椎被當成腰椎）；仍請醫師看一眼 S1，排除標註命名位移或移行椎。',
              '表中只列疑似位移與 d(same) ≥ 150 px 的案例。']
    report = '\n'.join(lines)
    print(report)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'naming_shift_report.md').write_text(report + '\n', encoding='utf-8')
        (args.output_dir / 'naming_shift_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
        if args.crops:
            by_id = {c.get('case_id'): c for c in cases}
            for r in flagged:
                ok = render_crop(by_id[r['case_id']], args.output_dir / f"{r['case_id']}_overlay.png")
                print(f"  overlay {r['case_id']}: {'ok' if ok else 'failed'}")
        print(f'寫入 {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
