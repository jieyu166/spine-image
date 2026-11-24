#!/usr/bin/env python
"""Python 端的「修整全部」參考答案：對 JSON 裡每節完整椎體，用與標註工具相同的框
（點位外接框 + 0.04 × max(寬, 高)）與提示點位跑 medsam_server.Segmenter，輸出 6 點。
tests/chrome_medsam_browser_smoke.mjs 用它檢查瀏覽器推論（MedsamCore＋onnxruntime-web）與 Python 一致。

用法：python tests/medsam_reference_points.py <image.png> <annotation.json> <decoder.pth> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import medsam_refine as mr  # noqa: E402
import medsam_server as ms  # noqa: E402

BOX_MARGIN = 0.04   # = spinal-annotation-web.html MEDSAM_BOX_MARGIN


def tool_box(points):
    xs = [p['x'] for p in points.values()]
    ys = [p['y'] for p in points.values()]
    m = BOX_MARGIN * max(max(xs) - min(xs), max(ys) - min(ys))
    return [min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m]


def main(image_path, json_path, decoder, out_path):
    image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    seg = ms.Segmenter(decoder).load()
    out = {'side': mr.SIDE, 'normalize': seg.normalize, 'vertebrae': {}}
    for v in data['vertebrae']:
        pts = v['points']
        if v.get('boundaryType') or not all(k in pts for k in mr.POINTS):
            continue
        t0 = time.time()
        points, iou, _ = seg(image, tool_box(pts), hint_points=pts)
        out['vertebrae'][v['name']] = {'points': points, 'iou': iou, 'seconds': round(time.time() - t0, 3)}
        print(f"{v['name']}：IoU {iou:.3f}", flush=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main(*sys.argv[1:5])
