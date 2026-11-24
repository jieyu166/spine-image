#!/usr/bin/env python
"""MedSAM 標註輔助本機服務：在標註工具裡拉一個框，回傳該節椎體的 6 點。

只在本機 127.0.0.1 監聽。標註工具（spinal-annotation-web.html）的「🧠 MedSAM 框選」會呼叫這裡。

安全：外部測試集（endplate_training_data/manifests/holdout_*.json）的影像一律拒絕，
避免測試集的標準答案被模型輔助拉向模型本身的輸出。

端點：
  GET  /health                      模型與 decoder 狀態
  POST /image   (multipart: file, sha256?)   上傳目前影像，回傳 image_id
  POST /segment (JSON: image_id, box[x0,y0,x1,y1], boundary?, hint_points?)
        boundary = null | 'upper' | 'lower'；回傳 points（完整椎體 6 點，邊界椎體只回該終板 3 點）、iou、outline

用法：
  python medsam_server.py [--decoder endplate_training_data/runs/medsam_refine_20261124/medsam_decoder_all.pth] [--port 8003]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import medsam_refine as mr

ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / 'endplate_training_data' / 'manifests'
DEFAULT_DECODER = ROOT / 'endplate_training_data' / 'runs' / 'medsam_refine_20261124' / 'medsam_decoder_all.pth'
MAX_IMAGES = 4


def load_holdout_hashes(manifest_dir: Path = MANIFEST_DIR) -> dict[str, str]:
    """{影像 SHA-256: 清單名稱}"""
    hashes = {}
    for path in sorted(Path(manifest_dir).glob('holdout_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('kind') != 'holdout':
            continue
        for e in data.get('entries', []):
            sha = (e.get('image') or {}).get('sha256')
            if sha:
                hashes[sha.lower()] = data.get('name', path.stem)
    return hashes


def pixel_hash(gray: np.ndarray) -> str:
    """解碼後灰階像素的 SHA-256（含尺寸）。瀏覽器重新編碼成 PNG 時檔案位元組會變，像素不會。"""
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    return hashlib.sha256(f'{gray.shape[0]}x{gray.shape[1]}:'.encode() + gray.tobytes()).hexdigest()


def load_holdout_pixel_hashes(manifest_dir: Path = MANIFEST_DIR, log=print) -> dict[str, str]:
    """{灰階像素 SHA-256: 清單名稱}；讀 manifest 列出的原始影像。找不到檔案的略過並警告。"""
    hashes, missing = {}, []
    for path in sorted(Path(manifest_dir).glob('holdout_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('kind') != 'holdout':
            continue
        entries = [e for e in data.get('entries', []) if (e.get('image') or {}).get('path')]
        for i, e in enumerate(entries, start=1):
            image_path = Path(e['image']['path'])
            gray = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_GRAYSCALE) if image_path.exists() else None
            if gray is None:
                missing.append(str(image_path))
                continue
            hashes[pixel_hash(gray)] = data.get('name', path.stem)
            if i % 10 == 0 or i == len(entries):
                log(f'  外部測試集像素指紋 {i}/{len(entries)}')
    if missing:
        log(f'WARNING：{len(missing)} 張外部測試集影像讀不到，只能靠檔案雜湊阻擋：{missing[:3]}')
    return hashes


def box_geometry(box):
    """提示框 → (裁切中心, 椎體大小)。訓練時框 = 點位外接框 + 每邊 BOX_MARGIN × 大小。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('框的寬高必須大於 0')
    size = max(x1 - x0, y1 - y0) / (1 + 2 * mr.BOX_MARGIN)
    return ((x0 + x1) / 2, (y0 + y1) / 2), size


def select_points(points, boundary):
    if boundary == 'upper':
        return {k: points[k] for k in mr.SUPERIOR}
    if boundary == 'lower':
        return {k: points[k] for k in mr.INFERIOR}
    return dict(points)


class Segmenter:
    """包住 MedSAM；測試時可換成假的。"""

    def __init__(self, decoder_path=None, normalize='imagenet'):
        self.decoder_path = Path(decoder_path) if decoder_path else None
        self.normalize = normalize
        self.model = None

    def load(self):
        if self.model is None:
            path = self.decoder_path if self.decoder_path and self.decoder_path.exists() else None
            self.model = mr.load_medsam(path)
            self.loaded_decoder = str(path) if path else None
            # 推論的影像正規化必須與 decoder 訓練時相同
            self.normalize = self.model.medsam_meta.get('normalize', self.normalize)
        return self

    def info(self):
        import torch
        return {'decoder': getattr(self, 'loaded_decoder', None), 'fine_tuned': bool(getattr(self, 'loaded_decoder', None)),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'normalize': self.normalize}

    def __call__(self, image, box, hint_points=None, neighbor_points=None):
        center, size = box_geometry(box)
        model = self.load().model
        return mr.segment_vertebra(model, image, center, mr.CONTEXT * size, box,
                                   hint_points or None, self.normalize, orientation_points=neighbor_points or None)


class SegmentRequest(BaseModel):
    image_id: str
    box: list[float]
    boundary: str | None = None
    hint_points: dict[str, dict[str, float]] | None = None       # 同一節既有點位（修正時）
    neighbor_points: dict[str, dict[str, float]] | None = None   # 鄰近椎體點位（新增時，只用來判斷方向）


# 只接受本機頁面：file://（Origin: null）與 127.0.0.1／localhost
LOCAL_ORIGINS = r'^(null|https?://(127\.0\.0\.1|localhost)(:\d+)?)$'


def create_app(segmenter, holdout_hashes=None, holdout_pixel_hashes=None):
    holdout_hashes = {k.lower(): v for k, v in (holdout_hashes or {}).items()}
    holdout_pixel_hashes = dict(holdout_pixel_hashes or {})
    images: OrderedDict[str, np.ndarray] = OrderedDict()
    app = FastAPI(title='MedSAM 標註輔助')
    app.add_middleware(CORSMiddleware, allow_origin_regex=LOCAL_ORIGINS, allow_methods=['*'], allow_headers=['*'])

    @app.middleware('http')
    async def private_network(request, call_next):
        response = await call_next(request)
        # Chrome 對 file:// 或公開網頁呼叫 127.0.0.1 的 Private Network Access 預檢
        response.headers['Access-Control-Allow-Private-Network'] = 'true'
        return response

    @app.get('/health')
    def health():
        return {'ok': True, **segmenter.info(), 'holdout_images_blocked': len(holdout_hashes),
                'holdout_pixel_fingerprints': len(holdout_pixel_hashes)}

    def refuse(name):
        raise HTTPException(403, f'這張是外部測試集（{name}）的影像，不提供 MedSAM 輔助，'
                                 '以免測試集標準答案偏向模型。請手動標註。')

    @app.post('/image')
    async def upload_image(file: UploadFile = File(...), sha256: str | None = Form(None)):
        data = await file.read()
        actual = hashlib.sha256(data).hexdigest()
        for sha in filter(None, {actual, (sha256 or '').lower()}):
            if sha in holdout_hashes:
                refuse(holdout_hashes[sha])
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise HTTPException(400, '無法解碼影像')
        if holdout_pixel_hashes:
            name = holdout_pixel_hashes.get(pixel_hash(image))
            if name:
                refuse(name)
        images[actual] = image
        images.move_to_end(actual)
        while len(images) > MAX_IMAGES:
            images.popitem(last=False)
        return {'image_id': actual, 'width': int(image.shape[1]), 'height': int(image.shape[0])}

    @app.post('/segment')
    def segment(req: SegmentRequest):
        image = images.get(req.image_id)
        if image is None:
            raise HTTPException(404, '找不到影像，請重新上傳')
        images.move_to_end(req.image_id)
        if len(req.box) != 4:
            raise HTTPException(400, 'box 需為 [x0, y0, x1, y1]')
        if req.boundary not in (None, 'upper', 'lower'):
            raise HTTPException(400, "boundary 只能是 null、'upper' 或 'lower'")
        try:
            points, iou, outline = segmenter(image, req.box, req.hint_points, req.neighbor_points)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # 例如 CUDA 記憶體不足：回 JSON 錯誤（含 CORS 標頭），前端才看得到原因
            raise HTTPException(500, f'MedSAM 執行錯誤：{type(e).__name__}: {e}')
        if points is None:
            raise HTTPException(422, '框內找不到可用的椎體輪廓，請把框拉得更貼近椎體')
        return {'points': select_points(points, req.boundary), 'iou': iou, 'outline': outline,
                'boundary': req.boundary}

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--decoder', type=Path, default=DEFAULT_DECODER)
    parser.add_argument('--port', type=int, default=8003)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    import uvicorn

    segmenter = Segmenter(args.decoder).load()
    info = segmenter.info()
    print(f"MedSAM decoder：{info['decoder'] or '（找不到微調權重，使用未微調 MedSAM）'}；{info['device']}；normalize={info['normalize']}")
    holdout = load_holdout_hashes()
    print(f'外部測試集影像 {len(holdout)} 張會被拒絕；計算像素指紋（防瀏覽器重新編碼）…')
    pixels = load_holdout_pixel_hashes()
    uvicorn.run(create_app(segmenter, holdout, pixels), host='127.0.0.1', port=args.port)
    return 0


if __name__ == '__main__':
    sys.exit(main())
