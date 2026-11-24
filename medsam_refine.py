#!/usr/bin/env python
"""MedSAM 第二階段細修（實驗）：用分割輪廓求 6 點，取代 heatmap 細修器。

動機（2026-10-15 第二階段實驗）：裁切位置正確時，高解析度細修可把角點中位數誤差從 85 px 降到 26 px，
表示瓶頸在解析度。這裡測試醫學影像分割基礎模型 MedSAM（SAM ViT-B，醫學影像微調）能否做得更好。

做法：
- 標準答案遮罩：每節椎體的 6 點（前上、中上、後上、後下、中下、前下）依序連成六邊形。
  T12 只標下終板、S1 只標上終板，沒有完整輪廓，不參與訓練、推論時保留第一階段座標。
- 輸入：以椎體為中心從原圖裁 2.2× 椎體大小的正方形，縮放成 1024²（MedSAM 原生解析度），
  提示框 = 椎體點位外接框（訓練時加偏移模擬第一階段誤差）。
- 微調：影像編碼器與提示編碼器凍結，只訓練 mask decoder（Dice + BCE + 預測 IoU 的 MSE）。
- 點位：取遮罩最大連通區的外輪廓，依椎體方向（由提示點位推得）找四個對角極值點當角點，
  上／下終板弧線上沿弦長一半的位置當中點。
- 交叉驗證與第一階段同一組 stratified 病人隔離 5 折，fold k 的模型只看 fold k 訓練集。

版本：
  medsam_zeroshot   未微調的 MedSAM，第一階段裁切與提示框
  medsam_ft         微調後，第一階段裁切與提示框（實際全自動流程）
  medsam_ft_gated   同上，但預測 IoU < --min-iou 或輪廓求點失敗時保留第一階段
  medsam_drawn_box  微調後，裁切與提示框取自醫師標註，四邊各加 σ = 5% 椎體大小的雜訊
                    （模擬在標註工具裡由醫師拉框；不是全自動流程）

產物（--out-dir）：fold{k}_decoder.pth、train_log.json、<版本>/oof_eval/oof_predictions.json、summary.md。
--train-final 另外用全部標註訓練 medsam_decoder_all.pth，供 medsam_server.py／標註工具使用。
--side 512：encoder 輸入改成 512²（位置編碼由 1024 內插，encoder 仍凍結），瀏覽器內推論約快 5 倍；
            一節椎體在原圖約 300 px、裁切約 660 px，512 幾乎不損失原圖解析度。decoder 權重的 meta 記錄 side，
            載入時自動切換。

用法：
  python medsam_refine.py --stage1-run endplate_training_data/runs/learning_curve_20261013/frac_100 \\
      --annotations-dir endplate_training_data/runs/sixpoint_reviewed_20260922b_cv5/annotations \\
      --out-dir endplate_training_data/runs/medsam_refine_YYYYMMDD [--epochs 20] [--train-final]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from refine_stage2 import (CONTEXT, CORNERS, POINTS, ImageCache, apply_affine, case_median_sizes, crop_matrix,
                           fill_sizes, vertebra_center, vertebra_size, write_variant)

ROOT = Path(__file__).resolve().parent
MEDSAM_DIR = Path(os.environ.get('MEDSAM_DIR', r'C:\Users\jai16\.cache\medsam-vit-base'))
SIDE = 1024          # MedSAM 輸入邊長（set_input_side 可改成 512）
LOWRES = 256         # mask decoder 輸出邊長 = SIDE / 4
BOX_MARGIN = 0.04    # 提示框外擴（× 椎體大小）
# 訓練時的模擬誤差：整框平移 σ（× 大小）、四邊各自 σ、上限
TRAIN_SHIFT_SIGMA, TRAIN_SIDE_SIGMA, TRAIN_CLIP = 0.15, 0.07, 0.4
SUPERIOR = ('anteriorSuperior', 'middleSuperior', 'posteriorSuperior')
INFERIOR = ('anteriorInferior', 'middleInferior', 'posteriorInferior')


def set_input_side(side):
    """設定 MedSAM 輸入邊長（1024 = 原生；512 = 瀏覽器版）。影響裁切、訓練目標與推論的所有函式。"""
    global SIDE, LOWRES
    side = int(side)
    if side < 256 or side > 1024 or side % 64:
        raise ValueError(f'side 必須是 256–1024 之間 64 的倍數：{side}')
    SIDE, LOWRES = side, side // 4
    return SIDE


# ── 幾何（純函式，有測試） ─────────────────────────────────────────────
def xy(p):
    return (float(p['x']), float(p['y'])) if isinstance(p, dict) else (float(p[0]), float(p[1]))


def has_full_outline(points):
    return all(k in points for k in CORNERS)


def outline_polygon(points):
    """六邊形（缺中點時退回四邊形），依 POINTS 順序繞一圈；不完整時回傳 None。"""
    if not has_full_outline(points):
        return None
    return np.array([xy(points[k]) for k in POINTS if k in points], dtype=np.float32)


def rasterize(polygon, size, scale=1.0):
    """把多邊形（座標 × scale）畫成 size×size 的 0/1 遮罩。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.round(np.asarray(polygon, dtype=np.float64) * scale * 16).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return mask


def points_box(points, margin=0.0):
    arr = np.array([xy(p) for p in points.values()], dtype=np.float64)
    (x0, y0), (x1, y1) = arr.min(0), arr.max(0)
    return np.array([x0 - margin, y0 - margin, x1 + margin, y1 + margin], dtype=np.float64)


def jitter_box(box, size, rng, shift_sigma=TRAIN_SHIFT_SIGMA, side_sigma=TRAIN_SIDE_SIGMA, clip=TRAIN_CLIP):
    """整框平移 + 四邊各自擾動；保證 x0 < x1、y0 < y1。"""
    shift = np.clip(rng.normal(0, shift_sigma * size, 2), -clip * size, clip * size)
    sides = np.clip(rng.normal(0, side_sigma * size, 4), -clip * size, clip * size)
    b = np.asarray(box, dtype=np.float64) + np.array([shift[0], shift[1], shift[0], shift[1]]) + sides
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half_w, half_h = max((b[2] - b[0]) / 2, 0.1 * size), max((b[3] - b[1]) / 2, 0.1 * size)
    return np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h])


def local_axes(points):
    """(u, v)：u 指向前方（anterior），v 指向下方（inferior），互相垂直的單位向量。"""
    sup = [xy(points[k]) for k in SUPERIOR if k in points]
    inf = [xy(points[k]) for k in INFERIOR if k in points]
    ant = [xy(points[k]) for k in ('anteriorSuperior', 'anteriorInferior') if k in points]
    post = [xy(points[k]) for k in ('posteriorSuperior', 'posteriorInferior') if k in points]
    u = np.subtract(np.mean(ant, 0), np.mean(post, 0)) if ant and post else np.array([-1.0, 0.0])
    if not (sup and inf) and ant and post and np.linalg.norm(u) > 1e-6:
        # 只有一個終板（S1 上終板、T12/C2 下終板）：終板方向就是 u，v 取垂直且朝下
        u = u / np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        return u, (v if v[1] >= 0 else -v)
    v = np.subtract(np.mean(inf, 0), np.mean(sup, 0)) if sup and inf else np.array([0.0, 1.0])
    v = v / (np.linalg.norm(v) or 1.0)
    u = u - np.dot(u, v) * v
    if np.linalg.norm(u) < 1e-6:
        u = np.array([-v[1], v[0]])
    return u / np.linalg.norm(u), v


def axes_from_mask(mask, anterior_sign=-1.0, prefer_v=None):
    """沒有提示點位時，由遮罩最小外接矩形估方向。v 取較接近垂直的邊（指向下），
    u 與之垂直且 x 分量符號 = anterior_sign（標註工具慣例：前方在影像左側，即 -1）。
    prefer_v（例如鄰近椎體的下方向）給定時，改取與它最平行的邊；椎體傾斜超過 45° 時才判得對。"""
    comp = largest_component(mask)
    if comp is None:
        return np.array([anterior_sign, 0.0]), np.array([0.0, 1.0])
    ys, xs = np.nonzero(comp)
    box = cv2.boxPoints(cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32)))
    edges = [box[1] - box[0], box[2] - box[1]]
    ref = np.array([0.0, 1.0]) if prefer_v is None else np.asarray(prefer_v, dtype=np.float64)
    v = max(edges, key=lambda e: abs(float(np.dot(e, ref))) / (np.linalg.norm(e) or 1.0)).astype(np.float64)
    v = v / (np.linalg.norm(v) or 1.0)
    if v[1] < 0:
        v = -v
    u = np.array([v[1], -v[0]])
    if np.sign(u[0]) != np.sign(anterior_sign):
        u = -u
    return u, v


def largest_component(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == keep).astype(np.uint8)


def _arc(contour, i, j):
    """輪廓上從 i 到 j 的兩條弧（含端點）。"""
    n = len(contour)
    fwd = [(i + k) % n for k in range(((j - i) % n) + 1)]
    bwd = [(i - k) % n for k in range(((i - j) % n) + 1)]
    return contour[fwd], contour[bwd]


def _mid_on_arc(arc, a, b):
    chord = b - a
    length2 = float(np.dot(chord, chord)) or 1.0
    t = (arc - a) @ chord / length2
    return arc[int(np.argmin(np.abs(t - 0.5)))]


def mask_to_points(mask, axes, min_area=50):
    """遮罩 → 6 點（遮罩座標）。axes = local_axes(提示點位)。失敗回傳 None。"""
    comp = largest_component(mask)
    if comp is None or comp.sum() < min_area:
        return None
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(contour) < 8:
        return None
    u, v = (np.asarray(a, dtype=np.float64) for a in axes)
    c = contour.mean(0)
    a, b = (contour - c) @ u, (contour - c) @ v
    wa, hb = max((a.max() - a.min()) / 2, 1e-6), max((b.max() - b.min()) / 2, 1e-6)
    an, bn = a / wa, b / hb
    idx = {
        'anteriorSuperior': int(np.argmax(an - bn)), 'posteriorSuperior': int(np.argmax(-an - bn)),
        'posteriorInferior': int(np.argmax(-an + bn)), 'anteriorInferior': int(np.argmax(an + bn)),
    }
    if len(set(idx.values())) < 4:
        return None
    out = {k: contour[i] for k, i in idx.items()}
    for name, (p, q), side in (('middleSuperior', ('anteriorSuperior', 'posteriorSuperior'), 1),
                               ('middleInferior', ('anteriorInferior', 'posteriorInferior'), -1)):
        arcs = _arc(contour, idx[p], idx[q])
        # 上終板那條弧的 b 平均較小（較上方）→ 取 min；下終板取 max
        arc = min(arcs, key=lambda s: side * float(np.mean((s - c) @ v)))
        out[name] = _mid_on_arc(arc, contour[idx[p]], contour[idx[q]])
    return {k: (float(out[k][0]), float(out[k][1])) for k in POINTS}


def mask_iou(pred, target):
    pred, target = np.asarray(pred) > 0, np.asarray(target) > 0
    union = np.logical_or(pred, target).sum()
    return float(np.logical_and(pred, target).sum() / union) if union else 0.0


# ── 影像與模型 ────────────────────────────────────────────────────────
def crop_1024(image, center, side, angle=0.0):
    m = crop_matrix(center, side, angle, crop=SIDE)
    patch = cv2.warpAffine(image, m, (SIDE, SIDE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return patch, m


def preprocess(patch, normalize='imagenet'):
    """灰階 uint8 → (3, 1024, 1024) float32。imagenet = HF processor 設定；minmax = MedSAM 原始程式的 [0,1]。"""
    p = patch.astype(np.float32)
    if normalize == 'minmax':
        lo, hi = float(p.min()), float(p.max())
        p = (p - lo) / max(hi - lo, 1e-6)
        return np.stack([p, p, p], 0)
    p = p / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    return (np.stack([p, p, p], 0) - mean) / std


def load_medsam(decoder_path=None, device=None):
    """回傳 MedSAM；有 decoder 權重時一併載入，訓練時的設定（normalize 等）放在 model.medsam_meta。"""
    import torch
    from transformers import SamModel

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(decoder_path, map_location='cpu') if decoder_path else None
    meta = dict(state.get('meta') or {}) if state else {}
    if state:
        set_input_side(meta.get('side', 1024))   # 舊權重沒有記錄 side，都是 1024
    model = SamModel.from_pretrained(str(MEDSAM_DIR)) if SIDE == 1024 else resized_sam(SIDE)
    model.medsam_meta = meta
    if state:
        model.mask_decoder.load_state_dict(state['mask_decoder'])
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith('mask_decoder.')
    return model.to(device).eval()


def resized_sam(side):
    """輸入 side² 的 MedSAM：絕對位置編碼（64×64）與全域注意力的相對位置編碼由 1024 版內插。"""
    import torch.nn.functional as F
    from transformers import SamConfig, SamModel

    src = SamModel.from_pretrained(str(MEDSAM_DIR))
    cfg = SamConfig.from_pretrained(str(MEDSAM_DIR))
    grid = side // cfg.vision_config.patch_size
    cfg.vision_config.image_size = side
    cfg.prompt_encoder_config.image_size = side
    cfg.prompt_encoder_config.image_embedding_size = grid
    model = SamModel(cfg)
    target = model.state_dict()
    state = {}
    for k, v in src.state_dict().items():
        if v.shape != target[k].shape:
            if k.endswith('pos_embed'):          # (1, 64, 64, C) → (1, g, g, C)
                v = F.interpolate(v.permute(0, 3, 1, 2), size=(grid, grid), mode='bicubic',
                                  align_corners=False).permute(0, 2, 3, 1)
            elif 'rel_pos' in k:                  # (2·64−1, C) → (2g−1, C)
                v = F.interpolate(v.T[None], size=target[k].shape[0], mode='linear', align_corners=False)[0].T
            else:
                raise ValueError(f'未預期的形狀差異：{k} {tuple(v.shape)} → {tuple(target[k].shape)}')
        state[k] = v.contiguous()
    model.load_state_dict(state)
    return model


def encode(model, batch):
    import torch

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=batch.is_cuda):
        out = model.get_image_embeddings(batch)
    emb = out if torch.is_tensor(out) else out[0]
    return emb.float()


def decode(model, embeddings, boxes):
    """boxes: (B, 4) 1024 座標 → (logits (B,256,256), iou (B,))。"""
    out = model(image_embeddings=embeddings, input_boxes=boxes[:, None, :], multimask_output=False)
    return out.pred_masks[:, 0, 0], out.iou_scores[:, 0, 0]


# ── 訓練 ─────────────────────────────────────────────────────────────
def outline_samples(annotations):
    samples = []
    for a in annotations:
        verts = [v for v in a.get('vertebrae', []) if v.get('points')]
        for v in verts:
            pts = v['points']
            size = vertebra_size(pts)
            if size and has_full_outline(pts):
                samples.append({'image': ROOT / str(a['image_path']).replace('\\', '/'), 'points': pts, 'size': size})
    return samples


def make_training_item(sample, cache, rng, normalize):
    image = cache.get(sample['image'])
    size = sample['size']
    cx, cy = vertebra_center(sample['points'])
    jitter = np.clip(rng.normal(0, 0.2 * size, 2), -0.5 * size, 0.5 * size)
    side = CONTEXT * size * rng.uniform(0.85, 1.2)
    angle = rng.uniform(-10, 10)
    patch, m = crop_1024(image, (cx + jitter[0], cy + jitter[1]), side, angle)
    p = patch.astype(np.float32) / 255.0
    p = np.clip(p * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08), 0, 1) ** rng.uniform(0.8, 1.25)
    patch = (p * 255).astype(np.uint8)
    pts = {k: apply_affine(m, *xy(v)) for k, v in sample['points'].items() if k in POINTS}
    scale = SIDE / side
    box = jitter_box(points_box(pts, BOX_MARGIN * size * scale), size * scale, rng)
    target = rasterize(outline_polygon(pts), LOWRES, LOWRES / SIDE)
    return preprocess(patch, normalize), box.astype(np.float32), target.astype(np.float32)


def train_decoder(train_annotations, args, log, tag):
    import torch
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_medsam(device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    samples = outline_samples(train_annotations)
    steps_per_epoch = math.ceil(len(samples) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    rng = np.random.default_rng(args.seed)
    cache = ImageCache()
    history = []
    log(f"  [{tag}] {len(samples)} 節完整椎體、{args.epochs} epochs、每 epoch {steps_per_epoch} 步")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.mask_decoder.train()
        order = rng.permutation(len(samples))
        losses, ious = [], []
        for s in range(steps_per_epoch):
            items = [make_training_item(samples[i], cache, rng, args.normalize)
                     for i in order[s * args.batch_size:(s + 1) * args.batch_size]]
            x = torch.from_numpy(np.stack([it[0] for it in items])).to(device)
            boxes = torch.from_numpy(np.stack([it[1] for it in items])).to(device)
            target = torch.from_numpy(np.stack([it[2] for it in items])).to(device)
            emb = encode(model, x)
            logits, iou_pred = decode(model, emb, boxes)
            prob = torch.sigmoid(logits)
            inter = (prob * target).sum((1, 2))
            dice = 1 - (2 * inter + 1) / (prob.sum((1, 2)) + target.sum((1, 2)) + 1)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean((1, 2))
            with torch.no_grad():
                hard = (logits > 0).float()
                true_iou = (hard * target).sum((1, 2)) / ((hard + target).clamp(max=1).sum((1, 2)) + 1e-6)
            loss = (dice + bce).mean() + F.mse_loss(iou_pred, true_iou)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
            ious.append(float(true_iou.mean()))
        elapsed = time.time() - t0
        eta = elapsed / epoch * (args.epochs - epoch)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'train_iou': float(np.mean(ious))})
        log(f"  [{tag}] epoch {epoch}/{args.epochs} loss {np.mean(losses):.4f} 訓練 IoU {np.mean(ious):.3f}"
            f"（已 {elapsed / 60:.1f} 分，約剩 {eta / 60:.1f} 分）")
    model.eval()
    return model, history


def save_decoder(model, path, meta):
    import torch

    torch.save({'mask_decoder': model.mask_decoder.state_dict(), 'meta': meta}, path)


# ── 推論 ─────────────────────────────────────────────────────────────
def segment_vertebra(model, image, center, side, box_orig, axes_points=None, normalize='imagenet',
                     orientation_points=None):
    """裁切 → MedSAM → 6 點（原圖座標）。回傳 (points 或 None, 預測 IoU, 遮罩多邊形（原圖座標）)。
    axes_points：同一節既有點位，直接決定方向。
    沒有 axes_points 時（醫師拉框）由遮罩本身估方向；orientation_points（鄰近椎體點位）用來在
    最小外接矩形的兩個邊之間挑出「下方向」，處理傾斜超過 45° 的椎體。"""
    import torch

    device = next(model.parameters()).device
    patch, m = crop_1024(image, center, side)
    x = torch.from_numpy(preprocess(patch, normalize)[None]).to(device)
    corners = [apply_affine(m, box_orig[0], box_orig[1]), apply_affine(m, box_orig[2], box_orig[3])]
    box = np.array([corners[0][0], corners[0][1], corners[1][0], corners[1][1]], dtype=np.float32)
    with torch.no_grad():
        emb = encode(model, x)
        logits, iou = decode(model, emb, torch.from_numpy(box[None]).to(device))
        full = torch.nn.functional.interpolate(logits[:, None], size=(SIDE, SIDE), mode='bilinear', align_corners=False)
    mask = (full[0, 0] > 0).cpu().numpy().astype(np.uint8)
    if axes_points:
        axes_crop = local_axes({k: apply_affine(m, *xy(p)) for k, p in axes_points.items()})
    elif orientation_points:
        prefer_v = local_axes({k: apply_affine(m, *xy(p)) for k, p in orientation_points.items()})[1]
        axes_crop = axes_from_mask(mask, prefer_v=prefer_v)
    else:
        axes_crop = axes_from_mask(mask)
    pts = mask_to_points(mask, axes_crop)
    inv = cv2.invertAffineTransform(m)
    outline = None
    comp = largest_component(mask)
    if comp is not None:
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = [apply_affine(inv, float(p[0][0]), float(p[0][1])) for p in max(cs, key=cv2.contourArea)]
    if pts is None:
        return None, float(iou[0]), outline
    return {k: dict(zip('xy', apply_affine(inv, *pts[k]))) for k in POINTS}, float(iou[0]), outline


def refine_case(model, image, vertebrae, normalize, min_iou=None, boxes=None, centers=None, sizes=None):
    """只改有完整四角的椎體；其他（T12 下終板、S1 上終板）保留原座標。"""
    sizes = sizes or case_median_sizes(vertebrae)
    out = []
    for i, v in enumerate(vertebrae):
        pts = v.get('points') or {}
        size = sizes[i]
        if not has_full_outline(pts) or not size:
            out.append(v)
            continue
        center = centers[i] if centers else vertebra_center(pts)
        box = boxes[i] if boxes else points_box(pts, BOX_MARGIN * size)
        new, iou, _ = segment_vertebra(model, image, center, CONTEXT * size, box, pts, normalize)
        if new is None or (min_iou is not None and iou < min_iou):
            out.append(v)
            continue
        keep = {k: new[k] for k in POINTS if k in pts}
        out.append({**v, 'points': keep, 'medsam_iou': iou})
    return out


def drawn_boxes(pred, gt, rng):
    """模擬醫師拉框：同名椎體用醫師標註外接框，四邊加 σ = 5% 大小的雜訊；裁切也以標註為準。"""
    gt_by_name = {v['name']: v for v in gt}
    boxes, centers, sizes = [], [], []
    for v in pred:
        ref = gt_by_name.get(v['name'])
        ref_pts = (ref or {}).get('points') or {}
        size = vertebra_size(ref_pts) if has_full_outline(ref_pts) else None
        if not size:
            boxes.append(None)
            centers.append(vertebra_center(v['points']) if v.get('points') else None)
            sizes.append(None)
            continue
        noise = rng.normal(0, 0.05 * size, 4)
        boxes.append(points_box(ref_pts, BOX_MARGIN * size) + noise)
        centers.append(vertebra_center(ref_pts))
        sizes.append(size)
    return boxes, centers, sizes


# ── 主程式 ───────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage1-run', type=Path, required=True)
    parser.add_argument('--annotations-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--folds', type=int, nargs='*', help='只跑指定 fold（除錯用）')
    parser.add_argument('--min-iou', type=float, default=0.8, help='medsam_ft_gated：預測 IoU 低於此值保留第一階段')
    parser.add_argument('--normalize', choices=('imagenet', 'minmax'), default='imagenet')
    parser.add_argument('--side', type=int, default=1024, help='encoder 輸入邊長（512 = 瀏覽器版）')
    parser.add_argument('--reuse-models', action='store_true', help='載入 out-dir 既有的 fold*_decoder.pth，只重跑推論')
    parser.add_argument('--skip-zeroshot', action='store_true')
    parser.add_argument('--train-final', action='store_true', help='另用全部標註訓練 medsam_decoder_all.pth（標註工具用）')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    set_input_side(args.side)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / 'refine.log'

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    import torch
    from oof_metrics import compare_case_landmarks, summarize_cases
    from train_vertebra_model_cv import DEFAULT_GROUP_REGEX, build_group_folds, load_all_annotations

    annotations = load_all_annotations(str(args.annotations_dir / 'train_annotations.json'),
                                       str(args.annotations_dir / 'val_annotations.json'))
    folds = build_group_folds(annotations, args.n_folds, args.seed, DEFAULT_GROUP_REGEX, 'stratified')
    recorded = json.loads((args.stage1_run / 'cv_results.json').read_text(encoding='utf-8')).get('fold_val_members')
    rebuilt = [[str(annotations[i]['image_path']).replace('\\', '/') for i in val] for _, val in folds]
    if recorded and rebuilt != recorded:
        raise SystemExit('fold 與第一階段 run 記錄不一致，停止以免資料洩漏')
    stage1 = json.loads((args.stage1_run / 'oof_eval' / 'oof_predictions.json').read_text(encoding='utf-8'))['cases']
    by_index = {c['annotation_index']: c for c in stage1}
    log(f"{len(annotations)} 張、{len(folds)} 折；第一階段：{args.stage1_run}；MedSAM：{MEDSAM_DIR}；normalize={args.normalize}；"
        f"輸入 {SIDE}²")

    names = ['medsam_ft', 'medsam_ft_gated', 'medsam_drawn_box'] + ([] if args.skip_zeroshot else ['medsam_zeroshot'])
    variants = {n: [] for n in names}
    train_log = {}
    zeroshot = None if args.skip_zeroshot else load_medsam()
    for k, (train_idx, val_idx) in enumerate(folds, start=1):
        if args.folds and k not in args.folds:
            continue
        log(f"=== fold {k}：訓練 {len(train_idx)} 張，細修 {len(val_idx)} 張 ===")
        path = args.out_dir / f'fold{k}_decoder.pth'
        if args.reuse_models:
            model = load_medsam(path)
            if SIDE != args.side:   # 舊權重沒有記錄 side＝1024；不能和 --side 混用
                raise SystemExit(f'{path.name} 是 {SIDE}² 的 decoder，與 --side {args.side} 不符')
            log(f"  載入 {path.name}（只重跑推論）")
        else:
            model, history = train_decoder([annotations[i] for i in train_idx], args, log, f'fold {k}')
            save_decoder(model, path, {'fold': k, 'epochs': args.epochs, 'normalize': args.normalize, 'side': SIDE})
            train_log[k] = history
        rng = np.random.default_rng(args.seed + k)
        cache = ImageCache()
        for n_done, i in enumerate(val_idx, start=1):
            case = by_index[i]
            if case.get('fold') != k:
                raise SystemExit(f"annotation {i} 在第一階段屬 fold {case.get('fold')}，不是 {k}")
            if case.get('status') != 'success':
                for n in names:
                    variants[n].append(case)
                continue
            gt = case['ground_truth']['vertebrae']
            pred = case['prediction']['vertebrae']
            image = cache.get(ROOT / str(case['image_path']).replace('\\', '/'))
            results = {
                'medsam_ft': refine_case(model, image, pred, args.normalize),
                'medsam_ft_gated': refine_case(model, image, pred, args.normalize, min_iou=args.min_iou),
            }
            boxes, centers, sizes = drawn_boxes(pred, gt, rng)
            fallback = case_median_sizes(pred)
            results['medsam_drawn_box'] = refine_case(
                model, image, pred, args.normalize,
                boxes=[b if b is not None else points_box(v['points'], BOX_MARGIN * (fallback[j] or 0)) for j, (b, v) in enumerate(zip(boxes, pred))],
                centers=centers, sizes=[s or fallback[j] for j, s in enumerate(sizes)])
            if zeroshot is not None:
                results['medsam_zeroshot'] = refine_case(zeroshot, image, pred, args.normalize)
            for n, verts in results.items():
                variants[n].append({**case, 'prediction': {**case['prediction'], 'vertebrae': verts},
                                    'metrics': compare_case_landmarks(gt, verts)})
            if n_done % 5 == 0 or n_done == len(val_idx):
                log(f"  fold {k} 推論 {n_done}/{len(val_idx)}")
        del model
        torch.cuda.empty_cache()
        log(f"  fold {k} 完成")
    if zeroshot is not None:
        del zeroshot
        torch.cuda.empty_cache()

    if train_log:
        (args.out_dir / 'train_log.json').write_text(json.dumps(train_log), encoding='utf-8')
    stage1_subset = [c for c in stage1 if not args.folds or c.get('fold') in args.folds]
    lines = ['# MedSAM 第二階段細修：OOF 結果', '',
             f"第一階段：`{args.stage1_run}`；裁切 {CONTEXT}× 椎體大小 → {SIDE}²；只訓練 mask decoder，每折 {args.epochs} epochs；"
             f"normalize={args.normalize}；gated 門檻 IoU {args.min_iou}。", '',
             '| 版本 | 案例 | 角點平均 px | 角點中位數 px | 角點 P90 px | 中點平均 px |', '|---|---:|---:|---:|---:|---:|']
    for name, cases in [('stage1', stage1_subset)] + list(variants.items()):
        s = summarize_cases(cases)['groups']
        lines.append(f"| {name} | {len(cases)} | {s['corners']['mean_distance_px']:.1f} | {s['corners']['median_distance_px']:.1f} | "
                     f"{s['corners']['p90_distance_px']:.1f} | {s['middle']['mean_distance_px']:.1f} |")
        if name != 'stage1':
            write_variant(args.out_dir, name, cases)
    lines += ['', 'medsam_drawn_box 的裁切與提示框取自醫師標註（模擬標註工具拉框），不是全自動流程。', '',
              '配對比較：`python compare_oof_runs.py --baseline <stage1>/oof_eval/oof_predictions.json '
              '--candidate <out>/medsam_ft/oof_eval/oof_predictions.json`']
    (args.out_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log('\n'.join(lines))

    if args.train_final:
        log('=== 用全部標註訓練標註工具用的 decoder ===')
        model, history = train_decoder(annotations, args, log, 'final')
        save_decoder(model, args.out_dir / 'medsam_decoder_all.pth',
                     {'fold': 'all', 'epochs': args.epochs, 'normalize': args.normalize, 'n_images': len(annotations),
                      'side': SIDE})
        log(f"  寫入 {args.out_dir / 'medsam_decoder_all.pth'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
