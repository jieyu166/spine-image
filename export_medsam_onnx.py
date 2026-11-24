#!/usr/bin/env python
"""把 MedSAM（影像 encoder）與微調後的 mask decoder 匯出成 ONNX，供瀏覽器內推論（onnxruntime-web）。

輸出（預設 endplate_training_data/onnx/medsam_20261124/，已在 .gitignore）：
  encoder.onnx        pixel_values (1,3,1024,1024) float32 → image_embeddings (1,256,64,64)
  encoder_int8.onnx   同上，權重動態量化成 int8（CPU／WASM 用，檔案約 1/4）
  encoder_fp16.onnx   同上，fp16 權重與運算、輸入輸出仍為 float32（WebGPU 用，檔案約 1/2）
  decoder.onnx        image_embeddings (1,256,64,64) + input_boxes (1,1,4)（1024 座標）
                      → low_res_masks (1,256,256) logits、iou_scores (1,)
                      512 版 decoder（meta 記錄 side=512）的尺寸全部減半：512²、32×32、128×128、框為 512 座標
  holdout_hashes.json 外部測試集的檔案 SHA-256 與像素指紋（瀏覽器模式阻擋用）
  meta.json           前處理設定（normalize、裁切倍率、框外擴比例）、檔案 SHA-256、一致性檢查結果

一致性檢查：用一張真實影像、一節椎體，比較 PyTorch（CPU fp32）與 onnxruntime 的 embedding 差異與遮罩 IoU。

用法：
  python export_medsam_onnx.py [--decoder <medsam_decoder_all.pth>] [--out <資料夾>] [--check-image <影像> --check-json <標註>]
  （不指定時自動取 Images/ 裡第一張有完整椎體、且不在外部測試集的影像）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import medsam_refine as mr

ROOT = Path(__file__).resolve().parent
DEFAULT_DECODER = ROOT / 'endplate_training_data' / 'runs' / 'medsam_refine_20261124' / 'medsam_decoder_all.pth'
DEFAULT_OUT = ROOT / 'endplate_training_data' / 'onnx' / 'medsam_20261124'
OPSET = 18
MANIFEST_DIR = ROOT / 'endplate_training_data' / 'manifests'


def log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def wrappers(model):
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, pixel_values):
            out = self.m.get_image_embeddings(pixel_values)
            return out if torch.is_tensor(out) else out[0]

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, image_embeddings, input_boxes):
            out = self.m(image_embeddings=image_embeddings, input_boxes=input_boxes, multimask_output=False)
            return out.pred_masks[:, 0, 0], out.iou_scores[:, 0, 0]

    return Encoder().eval(), Decoder().eval()


def export(module, args, path: Path, input_names, output_names):
    """先用 dynamo 匯出器；失敗時退回 TorchScript 匯出器。單一檔案（不拆 external data），瀏覽器較好載入。"""
    import torch

    t0 = time.time()
    try:
        torch.onnx.export(module, args, str(path), input_names=input_names, output_names=output_names,
                          opset_version=OPSET, dynamo=True, external_data=False)
        how = 'dynamo'
    except Exception as e:  # noqa: BLE001
        log(f'  dynamo 匯出失敗（{type(e).__name__}: {str(e)[:200]}），改用 TorchScript 匯出器')
        torch.onnx.export(module, args, str(path), input_names=input_names, output_names=output_names,
                          opset_version=OPSET, dynamo=False, do_constant_folding=True)
        how = 'torchscript'
    log(f'  {path.name}：{path.stat().st_size / 1e6:.1f} MB，{how}，{time.time() - t0:.0f} 秒')
    return how


def rchannel_fingerprint(image) -> str | None:
    """瀏覽器也算得出來的像素指紋：SHA-256(f"{高}x{寬}:" + R 通道位元組)。
    標註工具用 canvas getImageData 取 R 通道計算同一個值（8-bit 影像；16-bit 回傳 None）。"""
    if image is None or image.dtype != np.uint8:
        return None
    r = image if image.ndim == 2 else image[..., 2]          # OpenCV 是 BGR(A)
    r = np.ascontiguousarray(r)
    return hashlib.sha256(f'{r.shape[0]}x{r.shape[1]}:'.encode() + r.tobytes()).hexdigest()


class HoldoutError(RuntimeError):
    pass


def holdout_hashes(manifest_dir: Path = MANIFEST_DIR) -> dict:
    """外部測試集的檔案 SHA-256 與像素指紋，給瀏覽器模式阻擋用（只有雜湊，沒有影像或病歷號）。

    寧可匯出失敗也不能產生不完整的清單：沒有 manifest、沒有影像、任何一張讀不到或算不出指紋，
    都丟 HoldoutError。只接受 8-bit PNG（JPEG 的 EXIF 方向、16-bit、半透明在瀏覽器與 OpenCV 解碼結果不同）。"""
    files, pixels, manifests, problems = {}, {}, [], []
    for path in sorted(Path(manifest_dir).glob('holdout_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('kind') != 'holdout':
            continue
        name = data.get('name', path.stem)
        manifests.append(name)
        for i, e in enumerate(data.get('entries', []), start=1):
            info = e.get('image') or {}
            if not info.get('sha256'):
                problems.append(f'{name} 第 {i} 筆沒有 sha256')
                continue
            files[info['sha256'].lower()] = name
            image_path = Path(info.get('path') or '')
            if not info.get('path') or not image_path.exists():
                problems.append(f'{name} 第 {i} 筆影像讀不到')
                continue
            if image_path.suffix.lower() != '.png':
                problems.append(f'{name} 第 {i} 筆不是 PNG')
                continue
            image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_UNCHANGED)
            fp = rchannel_fingerprint(image)
            if fp is None or (image.ndim == 3 and image.shape[2] == 4 and int(image[..., 3].min()) < 255):
                problems.append(f'{name} 第 {i} 筆不是不透明的 8-bit 影像')
                continue
            pixels[fp] = name
    if not manifests or not files:
        raise HoldoutError(f'找不到外部測試集清單（{manifest_dir}/holdout_*.json）；沒有清單就不能匯出瀏覽器模型')
    if problems:
        raise HoldoutError(f'外部測試集清單不完整（{len(problems)} 筆）：' + '；'.join(problems[:5]))
    return {'created': datetime.now().isoformat(timespec='seconds'), 'manifests': manifests,
            'files': files, 'pixels': pixels}


def pick_check_pair(image_dir: Path, holdout_files: dict):
    """自動挑一致性檢查用的影像：有同名 JSON、含完整椎體、檔案雜湊不在外部測試集。"""
    for image_path in sorted(Path(image_dir).glob('*.png')):
        json_path = image_path.with_suffix('.json')
        if not json_path.exists() or sha256(image_path) in holdout_files:
            continue
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if any(mr.has_full_outline(v.get('points') or {}) for v in data.get('vertebrae', [])):
            return image_path, json_path
    raise SystemExit(f'{image_dir} 找不到可做一致性檢查的影像，請用 --check-image／--check-json 指定')


def check_case(image_path: Path, json_path: Path):
    """取一節完整椎體，回傳 (預處理後 1×3×1024×1024, 1024 座標的框)。"""
    image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_GRAYSCALE)
    data = json.loads(json_path.read_text(encoding='utf-8'))
    v = next(v for v in data['vertebrae'] if mr.has_full_outline(v['points']))
    pts = v['points']
    size = mr.vertebra_size(pts)
    box = mr.points_box(pts, mr.BOX_MARGIN * size)
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    patch, m = mr.crop_1024(image, center, mr.CONTEXT * size)
    a, b = mr.apply_affine(m, box[0], box[1]), mr.apply_affine(m, box[2], box[3])
    return v['name'], mr.preprocess(patch, 'imagenet')[None].astype(np.float32), np.array([[[a[0], a[1], b[0], b[1]]]], np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--decoder', type=Path, default=DEFAULT_DECODER)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check-image', type=Path, help='一致性檢查用影像（預設自動挑選）')
    parser.add_argument('--check-json', type=Path, help='--check-image 的標註 JSON')
    parser.add_argument('--skip-int8', action='store_true')
    parser.add_argument('--skip-fp16', action='store_true')
    parser.add_argument('--allow-zeroshot', action='store_true', help='找不到 --decoder 時改匯出未微調的 MedSAM（否則停止）')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    import onnxruntime as ort
    import torch

    if not args.decoder.exists() and not args.allow_zeroshot:
        raise SystemExit(f'找不到 decoder：{args.decoder}（要匯出未微調的 MedSAM 請加 --allow-zeroshot）')
    try:
        hold = holdout_hashes()             # 先檢查，清單不完整就不要花時間匯出
    except HoldoutError as e:
        raise SystemExit(str(e))
    if (args.check_image is None) != (args.check_json is None):
        raise SystemExit('--check-image 與 --check-json 要一起指定')
    if args.check_image is None:
        args.check_image, args.check_json = pick_check_pair(ROOT / 'Images', hold['files'])
    elif sha256(args.check_image) in hold['files']:
        raise SystemExit('--check-image 是外部測試集影像，不能用')
    args.out.mkdir(parents=True, exist_ok=True)
    for stale in ('encoder.onnx', 'encoder_int8.onnx', 'encoder_fp16.onnx', 'decoder.onnx', 'meta.json', 'holdout_hashes.json'):
        (args.out / stale).unlink(missing_ok=True)     # 避免舊檔（例如另一個尺寸的 encoder）被瀏覽器選到
    log(f'[1/4] 載入 MedSAM＋decoder：{args.decoder}')
    model = mr.load_medsam(args.decoder if args.decoder.exists() else None, device=torch.device('cpu')).float()
    normalize = model.medsam_meta.get('normalize', 'imagenet')
    enc, dec = wrappers(model)

    log('[2/4] 匯出 ONNX')
    x = torch.zeros(1, 3, mr.SIDE, mr.SIDE)
    emb = torch.zeros(1, 256, mr.SIDE // 16, mr.SIDE // 16)   # 1024 → 64×64；512 → 32×32
    boxes = torch.tensor([[[300.0, 300.0, 700.0, 650.0]]])
    files = {'encoder': args.out / 'encoder.onnx', 'decoder': args.out / 'decoder.onnx'}
    exporters = {'encoder': export(enc, (x,), files['encoder'], ['pixel_values'], ['image_embeddings']),
                 'decoder': export(dec, (emb, boxes), files['decoder'], ['image_embeddings', 'input_boxes'],
                                   ['low_res_masks', 'iou_scores'])}
    if not args.skip_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        log('[3/4] encoder 動態量化 int8')
        files['encoder_int8'] = args.out / 'encoder_int8.onnx'
        # onnxruntime 量化的形狀推論在 Windows 讀不了含中文的路徑：在暫存（ASCII）資料夾做完再搬回
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'encoder.onnx', Path(tmp) / 'encoder_int8.onnx'
            shutil.copyfile(files['encoder'], src)
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
            shutil.copyfile(dst, files['encoder_int8'])
        log(f"  encoder_int8.onnx：{files['encoder_int8'].stat().st_size / 1e6:.1f} MB")

    if not args.skip_fp16:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        log('[3b/4] encoder 轉 fp16（WebGPU 用；int8 在 WebGPU 上會退回 CPU，反而慢）')
        files['encoder_fp16'] = args.out / 'encoder_fp16.onnx'
        fp16 = convert_float_to_float16(onnx.load(str(files['encoder'])), keep_io_types=True)
        onnx.save(fp16, str(files['encoder_fp16']))
        log(f"  encoder_fp16.onnx：{files['encoder_fp16'].stat().st_size / 1e6:.1f} MB")

    log('[4/4] 一致性檢查（PyTorch CPU fp32 vs onnxruntime CPU）')
    name, xin, box = check_case(args.check_image, args.check_json)
    with torch.no_grad():
        ref_emb = enc(torch.from_numpy(xin))
        ref_mask, ref_iou = dec(ref_emb, torch.from_numpy(box))
    ref_mask = ref_mask.numpy()[0] > 0
    sess_dec = ort.InferenceSession(str(files['decoder']), providers=['CPUExecutionProvider'])
    checks = {}
    for key in [k for k in ('encoder', 'encoder_int8', 'encoder_fp16') if k in files]:
        sess = ort.InferenceSession(str(files[key]), providers=['CPUExecutionProvider'])
        t0 = time.time()
        e = sess.run(None, {'pixel_values': xin})[0]
        t_enc = time.time() - t0
        mask, iou = sess_dec.run(None, {'image_embeddings': e, 'input_boxes': box})
        checks[key] = {'vertebra': name, 'embedding_max_abs_diff': float(np.abs(e - ref_emb.numpy()).max()),
                       'mask_iou_vs_torch': mr.mask_iou(mask[0] > 0, ref_mask),
                       'pred_iou': float(iou[0]), 'torch_pred_iou': float(ref_iou[0]),
                       'python_cpu_encoder_seconds': round(t_enc, 2)}
        log(f"  {key}：embedding 最大差 {checks[key]['embedding_max_abs_diff']:.4f}，遮罩 IoU {checks[key]['mask_iou_vs_torch']:.4f}，"
            f"預測 IoU {checks[key]['pred_iou']:.3f}（torch {checks[key]['torch_pred_iou']:.3f}），CPU encoder {t_enc:.2f} 秒")

    hold_path = args.out / 'holdout_hashes.json'
    hold_path.write_text(json.dumps(hold, ensure_ascii=False), encoding='utf-8')
    meta = {
        'created': datetime.now().isoformat(timespec='seconds'), 'source_decoder': str(args.decoder),
        # 瀏覽器載入時核對：清單筆數與檔案雜湊要一致，否則不啟用
        'holdout': {'files': len(hold['files']), 'pixels': len(hold['pixels']), 'manifests': hold['manifests'],
                    'created': hold['created'], 'sha256': sha256(hold_path)},
        'opset': OPSET, 'exporters': exporters, 'normalize': normalize,
        'preprocess': {'side': mr.SIDE, 'context': mr.CONTEXT, 'box_margin': mr.BOX_MARGIN, 'lowres': mr.LOWRES,
                       'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'files': {k: {'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)} for k, p in files.items()},
        'checks': checks, 'onnxruntime': ort.__version__, 'torch': torch.__version__,
    }
    (args.out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"外部測試集阻擋清單：{'、'.join(hold['manifests'])}；檔案 {len(hold['files'])}、像素指紋 {len(hold['pixels'])}")
    log(f'完成：{args.out}')
    bad = [k for k, c in checks.items() if c['mask_iou_vs_torch'] < (0.98 if k == 'encoder' else 0.9)]
    if bad:
        log(f'WARNING：{bad} 與 PyTorch 結果差異過大')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
