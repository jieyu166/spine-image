#!/usr/bin/env python
"""第二階段高解析度角點細修（實驗）。

問題：第一階段把 3408 px 高的腰椎片縮成 512 輸入、128 heatmap，一格 heatmap ≈ 27 原圖 px；
學習曲線顯示 54→70 張時角點誤差幾乎不再下降，懷疑是解析度天花板。

做法：
- 第一階段：沿用某個 OOF run 的預測（椎體數與命名完全不變，只改點位）。
- 第二階段：以每節椎體為中心，從「原圖」裁一塊 side = 2.2 × 椎體大小 的正方形，
  縮成 256×256，輸出 6 點 heatmap（64×64，一格約 10 原圖 px），再映射回原圖。
- 交叉驗證與第一階段同一組 stratified 病人隔離 5 折：fold k 的細修模型只用 fold k
  訓練集的醫師標註訓練，只細修 fold k 驗證案例 → 每例仍是 out-of-fold。
- 另輸出 oracle 版本：裁切中心改用醫師標註（模擬「第一階段完美找到椎體」），
  用來分辨誤差是卡在第一階段定位，還是卡在解析度。

產物（--out-dir）：
  fold{k}_refiner.pth、train_log.json、
  refined_pass1/oof_eval/oof_predictions.json、refined_pass2/...、oracle/...、summary.md
後兩者可直接丟給 compare_oof_runs.py 和第一階段做配對比較。

用法：
  python refine_stage2.py --stage1-run endplate_training_data/runs/learning_curve_20261013/frac_100 \
      --annotations-dir endplate_training_data/runs/sixpoint_reviewed_20260922b_cv5/annotations \
      --out-dir endplate_training_data/runs/stage2_refine_YYYYMMDD [--epochs 60]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
POINTS = ('anteriorSuperior', 'middleSuperior', 'posteriorSuperior',
          'posteriorInferior', 'middleInferior', 'anteriorInferior')
CORNERS = ('anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior')
CROP = 256
HEATMAP = 64
CONTEXT = 2.2      # 裁切邊長 = CONTEXT × 椎體大小
SIGMA = 1.5        # heatmap 高斯（heatmap 格）


# ── 幾何（純函式，有測試） ─────────────────────────────────────────────
def vertebra_center(points):
    pts = [(float(p['x']), float(p['y'])) for p in points.values()]
    return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))


def vertebra_size(points):
    """四角齊全時回傳 max(高, 寬)；邊界椎體（只有一個終板）回傳 None。"""
    if not all(k in points for k in CORNERS):
        return None
    xs = [float(points[k]['x']) for k in CORNERS]
    ys = [float(points[k]['y']) for k in CORNERS]
    height = ((points['anteriorInferior']['y'] - points['anteriorSuperior']['y']) +
              (points['posteriorInferior']['y'] - points['posteriorSuperior']['y'])) / 2
    width = max(xs) - min(xs)
    return float(max(abs(height), width, max(ys) - min(ys)))


def fill_sizes(vertebrae):
    """每節椎體的大小；邊界椎體借最近的完整椎體，全部缺時給 None。"""
    sizes = [vertebra_size(v.get('points') or {}) for v in vertebrae]
    known = [i for i, s in enumerate(sizes) if s]
    out = []
    for i, s in enumerate(sizes):
        if s:
            out.append(s)
        elif known:
            out.append(sizes[min(known, key=lambda j: abs(j - i))])
        else:
            out.append(None)
    return out


def crop_matrix(center, side, angle_deg=0.0, crop=CROP):
    """原圖 → 裁切座標的 2×3 仿射矩陣（以 center 為中心、邊長 side、旋轉 angle）。"""
    scale = crop / float(side)
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle_deg, scale)
    m[0, 2] += crop / 2.0 - center[0]
    m[1, 2] += crop / 2.0 - center[1]
    return m


def apply_affine(m, x, y):
    return (float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2]))


def make_targets(points_crop, size=HEATMAP, crop=CROP, sigma=SIGMA):
    """6 張高斯 heatmap 與有效遮罩；點不存在或落在裁切外時遮罩為 0。"""
    heat = np.zeros((len(POINTS), size, size), dtype=np.float32)
    mask = np.zeros(len(POINTS), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    ratio = size / float(crop)
    for c, name in enumerate(POINTS):
        if name not in points_crop:
            continue
        px, py = points_crop[name][0] * ratio, points_crop[name][1] * ratio
        if not (0 <= px < size and 0 <= py < size):
            continue
        heat[c] = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))
        mask[c] = 1.0
    return heat, mask


def decode_heatmap(heat, crop=CROP):
    """argmax + 對數拋物線內插（對高斯峰是精確解），回傳裁切座標 (x, y, peak)。"""
    h = cv2.GaussianBlur(heat.astype(np.float32), (5, 5), 1.0)
    iy, ix = np.unravel_index(int(np.argmax(h)), h.shape)
    lh = np.log(np.clip(h, 1e-10, None))

    def offset(left, mid, right):
        denom = left - 2 * mid + right
        return 0.0 if denom >= 0 else float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))

    fx = float(ix) + (offset(lh[iy, ix - 1], lh[iy, ix], lh[iy, ix + 1]) if 0 < ix < h.shape[1] - 1 else 0.0)
    fy = float(iy) + (offset(lh[iy - 1, ix], lh[iy, ix], lh[iy + 1, ix]) if 0 < iy < h.shape[0] - 1 else 0.0)
    scale = crop / float(h.shape[1])
    # 與 make_targets 同一慣例：heatmap 座標 = 裁切座標 × (HEATMAP / CROP)
    return fx * scale, fy * scale, float(heat[iy, ix])


# ── 影像 ─────────────────────────────────────────────────────────────
def load_gray(path):
    path = str(path)
    if path.lower().endswith('.dcm'):
        import pydicom
        img = pydicom.dcmread(path).pixel_array.astype(np.float32)
        lo, hi = np.percentile(img, (0.5, 99.5))
        return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def extract_crop(image, center, side, angle=0.0):
    m = crop_matrix(center, side, angle)
    patch = cv2.warpAffine(image, m, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return patch, m


def to_tensor_input(patch):
    p = patch.astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    return ((np.stack([p, p, p], 0) - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


# ── 模型 ─────────────────────────────────────────────────────────────
def build_model(pretrained_path):
    import torch.nn as nn
    from torchvision import models
    from train_vertebra_model import load_radimagenet_weights

    class Refiner(nn.Module):
        """SimpleBaseline：ResNet50（RadImageNet）+ 3 層反卷積 → 6 張 64×64 heatmap。"""

        def __init__(self):
            super().__init__()
            resnet = models.resnet50(weights=None)
            if pretrained_path:
                load_radimagenet_weights(resnet, pretrained_path)
            self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                          resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
            layers, ch = [], 2048
            for _ in range(3):
                layers += [nn.ConvTranspose2d(ch, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
                ch = 256
            self.deconv = nn.Sequential(*layers)
            self.head = nn.Conv2d(256, len(POINTS), 1)
            for p in list(resnet.conv1.parameters()) + list(resnet.bn1.parameters()) + list(resnet.layer1.parameters()):
                p.requires_grad = False

        def forward(self, x):
            return self.head(self.deconv(self.backbone(x)))

    return Refiner()


# ── 資料 ─────────────────────────────────────────────────────────────
class ImageCache:
    def __init__(self):
        self._cache = {}

    def get(self, path):
        key = str(path)
        if key not in self._cache:
            self._cache[key] = load_gray(key)
        return self._cache[key]


def annotation_image(annotation):
    return ROOT / str(annotation['image_path']).replace('\\', '/')


def vertebra_samples(annotations):
    samples = []
    for a in annotations:
        verts = [v for v in a.get('vertebrae', []) if v.get('points')]
        for v, size in zip(verts, fill_sizes(verts)):
            if size:
                samples.append({'image': annotation_image(a), 'points': v['points'], 'size': size})
    return samples


# 訓練增強範圍：要涵蓋第一階段實際的定位誤差（09-22b OOF：中心偏移 / 椎體大小 中位 0.26、P90 0.71；
# 估計大小 / 真實大小 P10 0.80、P90 1.36）。超出範圍的裁切細修模型沒見過，會失準。
AUG = {'center_sigma': 0.2, 'center_clip': 0.5, 'scale_lo': 0.85, 'scale_hi': 1.2}


def make_training_item(sample, cache, rng, aug=None):
    aug = aug or AUG
    image = cache.get(sample['image'])
    size = sample['size']
    cx, cy = vertebra_center(sample['points'])
    jitter = np.clip(rng.normal(0, aug['center_sigma'] * size, 2), -aug['center_clip'] * size, aug['center_clip'] * size)
    side = CONTEXT * size * rng.uniform(aug['scale_lo'], aug['scale_hi'])
    angle = rng.uniform(-10, 10)
    patch, m = extract_crop(image, (cx + jitter[0], cy + jitter[1]), side, angle)
    # 亮度／對比／gamma
    p = patch.astype(np.float32) / 255.0
    p = np.clip(p * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08), 0, 1) ** rng.uniform(0.8, 1.25)
    patch = (p * 255).astype(np.uint8)
    pts = {k: apply_affine(m, v['x'], v['y']) for k, v in sample['points'].items() if k in POINTS}
    heat, mask = make_targets(pts)
    return to_tensor_input(patch), heat, mask


# ── 訓練 / 推論 ──────────────────────────────────────────────────────
def train_fold(fold, train_annotations, args, log):
    import torch

    torch.manual_seed(args.seed + fold)
    rng = np.random.default_rng(args.seed + fold)
    samples = vertebra_samples(train_annotations)
    cache = ImageCache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.pretrained_path).to(device)
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('backbone')]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.lr * 0.1}, {'params': head, 'lr': args.lr}], weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(samples) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr * 0.1, args.lr], total_steps=steps, pct_start=0.1)
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(samples))
        losses = []
        for b in range(0, len(order), args.batch_size):
            batch = [make_training_item(samples[i], cache, rng, args.aug) for i in order[b:b + args.batch_size]]
            x = torch.from_numpy(np.stack([t[0] for t in batch])).to(device)
            y = torch.from_numpy(np.stack([t[1] for t in batch])).to(device)
            mk = torch.from_numpy(np.stack([t[2] for t in batch])).to(device)
            out = torch.sigmoid(model(x))
            per = ((out - y) ** 2).mean(dim=(2, 3))
            loss = (per * mk).sum() / mk.sum().clamp(min=1) * 100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
        history.append(float(np.mean(losses)))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"    fold {fold} epoch {epoch + 1}/{args.epochs} loss {history[-1]:.4f}（{len(samples)} 節椎體，{(time.time() - started) / 60:.1f} 分鐘）")
    torch.save({'model_state_dict': model.state_dict(), 'fold': fold, 'n_samples': len(samples)},
               args.out_dir / f'fold{fold}_refiner.pth')
    return model, history


def case_median_sizes(vertebrae):
    """整張片子各節椎體大小的中位數，套用到每一節（單節大小估錯時較穩）。"""
    # 只用四角齊全的椎體，不讓邊界椎體借來的大小重複計票
    known = [s for s in (vertebra_size(v.get('points') or {}) for v in vertebrae) if s]
    if not known:
        return [None] * len(vertebrae)
    return [float(np.median(known))] * len(vertebrae)


def refine_vertebrae(model, image, vertebrae, centers=None, sizes=None, min_peak=0.0, max_shift=None):
    """細修一組椎體；centers/sizes 給定時用它們裁切（oracle），否則由 vertebrae 本身估計。

    min_peak > 0 時，heatmap 峰值低於門檻的點保留原本（第一階段）的座標。
    max_shift 給定時，細修後椎體中心離裁切中心超過 max_shift × 椎體大小，視為鎖到鄰節，
    整節保留原本座標（細修失敗的主要型態）。
    """
    import torch

    device = next(model.parameters()).device
    sizes = sizes or fill_sizes(vertebrae)
    refined = []
    batch, metas = [], []
    for i, v in enumerate(vertebrae):
        pts = v.get('points') or {}
        size = sizes[i]
        if not pts or not size:
            refined.append(v)
            metas.append(None)
            continue
        center = centers[i] if centers else vertebra_center(pts)
        patch, m = extract_crop(image, center, CONTEXT * size)
        batch.append(to_tensor_input(patch))
        metas.append(m)
        refined.append(None)
    if batch:
        model.eval()
        with torch.no_grad():
            heat = torch.sigmoid(model(torch.from_numpy(np.stack(batch)).to(device))).cpu().numpy()
    k = 0
    for i, v in enumerate(vertebrae):
        if metas[i] is None:
            continue
        inv = cv2.invertAffineTransform(metas[i])
        new_points = {}
        for c, name in enumerate(POINTS):
            if name not in v['points']:
                continue
            x, y, peak = decode_heatmap(heat[k, c])
            if peak < min_peak:
                new_points[name] = v['points'][name]
                continue
            nx, ny = apply_affine(inv, x, y)
            new_points[name] = {'x': nx, 'y': ny}
        if max_shift is not None and sizes[i]:
            crop_center = apply_affine(inv, CROP / 2.0, CROP / 2.0)
            new_center = vertebra_center(new_points)
            if math.hypot(new_center[0] - crop_center[0], new_center[1] - crop_center[1]) > max_shift * sizes[i]:
                refined[i] = v
                k += 1
                continue
        refined[i] = {**v, 'points': new_points}
        k += 1
    return refined


def evaluate(cases_out):
    from oof_metrics import summarize_cases
    return summarize_cases(cases_out)


def write_variant(out_dir, name, cases):
    path = out_dir / name / 'oof_eval'
    path.mkdir(parents=True, exist_ok=True)
    (path / 'oof_predictions.json').write_text(json.dumps({'evaluation': f'stage2 {name}', 'cases': cases}, ensure_ascii=False), encoding='utf-8')
    return path / 'oof_predictions.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage1-run', type=Path, required=True)
    parser.add_argument('--annotations-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--folds', type=int, nargs='*', help='只跑指定 fold（除錯用）')
    parser.add_argument('--max-shift', type=float, default=0.25,
                        help='細修後椎體中心偏離裁切中心超過此比例（× 椎體大小）就整節保留第一階段')
    parser.add_argument('--aug', choices=('narrow', 'wide'), default='narrow',
                        help='narrow = 中心 σ0.2/上限0.5、縮放 0.85–1.2（預設）；wide = 中心 σ0.35/上限0.9、縮放 0.7–1.45'
                             '（2026-10-15 實測全面變差，含 oracle 26→66 px：偏移太大時中央是鄰節，模型分不清目標）')
    parser.add_argument('--reuse-models', action='store_true', help='載入 out-dir 既有的 fold*_refiner.pth，只重跑推論')
    parser.add_argument('--min-peak', type=float, default=0.3, help='細修點的 heatmap 峰值門檻；低於則保留第一階段')
    parser.add_argument('--pretrained-path', default=str(ROOT / 'pretrained' / 'RadImageNet-ResNet50.pt'))
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    args.aug = dict(AUG) if args.aug == 'narrow' else {'center_sigma': 0.35, 'center_clip': 0.9, 'scale_lo': 0.7, 'scale_hi': 1.45}
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / 'refine.log'

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    from oof_metrics import compare_case_landmarks
    from train_vertebra_model_cv import DEFAULT_GROUP_REGEX, build_group_folds, load_all_annotations

    annotations = load_all_annotations(str(args.annotations_dir / 'train_annotations.json'),
                                       str(args.annotations_dir / 'val_annotations.json'))
    folds = build_group_folds(annotations, args.n_folds, args.seed, DEFAULT_GROUP_REGEX, 'stratified')
    recorded = json.loads((args.stage1_run / 'cv_results.json').read_text(encoding='utf-8')).get('fold_val_members')
    rebuilt = [[str(annotations[i]['image_path']).replace('\\', '/') for i in val] for _, val in folds]
    if recorded and rebuilt != recorded:
        raise SystemExit('第二階段的 fold 與第一階段 run 記錄不一致，停止以免資料洩漏')
    stage1 = json.loads((args.stage1_run / 'oof_eval' / 'oof_predictions.json').read_text(encoding='utf-8'))['cases']
    by_index = {c['annotation_index']: c for c in stage1}
    log(f"{len(annotations)} 張、{len(folds)} 折；第一階段：{args.stage1_run}")

    variants = {'refined_pass1': [], 'refined_pass2': [], 'refined_median': [], 'refined_median_gated': [],
                'refined_shift_gated': [], 'oracle': []}
    train_log = {}
    for k, (train_idx, val_idx) in enumerate(folds, start=1):
        if args.folds and k not in args.folds:
            continue
        log(f"=== fold {k}：訓練 {len(train_idx)} 張，細修 {len(val_idx)} 張 ===")
        if args.reuse_models:
            import torch
            model = build_model(None)
            model.load_state_dict(torch.load(args.out_dir / f'fold{k}_refiner.pth', map_location='cpu')['model_state_dict'])
            model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            log(f"  載入 fold{k}_refiner.pth（只重跑推論）")
        else:
            model, history = train_fold(k, [annotations[i] for i in train_idx], args, log)
            train_log[k] = history
        cache = ImageCache()
        for i in val_idx:
            case = by_index[i]
            if case.get('fold') != k:
                raise SystemExit(f"annotation {i} 在第一階段屬 fold {case.get('fold')}，不是 {k}")
            gt = case['ground_truth']['vertebrae']
            if case.get('status') != 'success':
                for name in variants:
                    variants[name].append(case)
                continue
            image = cache.get(case['image_path'])
            pred = case['prediction']['vertebrae']
            pass1 = refine_vertebrae(model, image, pred)
            pass2 = refine_vertebrae(model, image, pass1)  # 以第一輪結果重新置中再細修一次
            median_sizes = case_median_sizes(pred)
            med = refine_vertebrae(model, image, pred, sizes=median_sizes)
            med_gated = refine_vertebrae(model, image, pred, sizes=median_sizes, min_peak=args.min_peak)
            shift_gated = refine_vertebrae(model, image, pred, sizes=median_sizes, min_peak=args.min_peak,
                                           max_shift=args.max_shift)
            # oracle：同名椎體以醫師標註的中心與大小裁切
            gt_by_name = {v['name']: v for v in gt}
            gt_sizes = dict(zip([v['name'] for v in gt], fill_sizes(gt)))
            centers, sizes = [], []
            for v in pred:
                ref = gt_by_name.get(v['name'])
                centers.append(vertebra_center(ref['points']) if ref and ref.get('points') else vertebra_center(v['points']))
                sizes.append(gt_sizes.get(v['name']) or (fill_sizes(pred)[pred.index(v)]))
            oracle = refine_vertebrae(model, image, pred, centers=centers, sizes=sizes)
            for name, verts in (('refined_pass1', pass1), ('refined_pass2', pass2), ('refined_median', med),
                                ('refined_median_gated', med_gated), ('refined_shift_gated', shift_gated),
                                ('oracle', oracle)):
                variants[name].append({**case, 'prediction': {**case['prediction'], 'vertebrae': verts},
                                       'metrics': compare_case_landmarks(gt, verts)})
        del model
        import torch
        torch.cuda.empty_cache()
        log(f"  fold {k} 完成")

    if train_log:
        (args.out_dir / 'train_log.json').write_text(json.dumps(train_log), encoding='utf-8')
    stage1_subset = [c for c in stage1 if not args.folds or c.get('fold') in args.folds]
    lines = ['# 第二階段高解析度細修：OOF 結果', '',
             f"第一階段：`{args.stage1_run}`；裁切 {CONTEXT}× 椎體大小 → {CROP}² 輸入、{HEATMAP}² heatmap；每折 {args.epochs} epochs。", '',
             '| 版本 | 案例 | 角點平均 px | 角點中位數 px | 角點 P90 px | 中點平均 px |', '|---|---:|---:|---:|---:|---:|']
    for name, cases in [('stage1', stage1_subset)] + list(variants.items()):
        s = evaluate(cases)['groups']
        lines.append(f"| {name} | {len(cases)} | {s['corners']['mean_distance_px']:.1f} | {s['corners']['median_distance_px']:.1f} | "
                     f"{s['corners']['p90_distance_px']:.1f} | {s['middle']['mean_distance_px']:.1f} |")
        if name != 'stage1':
            write_variant(args.out_dir, name, cases)
    lines += ['', '配對比較與信賴區間：`python compare_oof_runs.py --baseline <stage1>/oof_eval/oof_predictions.json '
              '--candidate <out>/refined_pass1/oof_eval/oof_predictions.json`']
    (args.out_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
