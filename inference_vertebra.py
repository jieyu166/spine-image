#!/usr/bin/env python3
"""
脊椎椎體頂點檢測 - 推理腳本 V3.0
Spine Vertebra Corner Detection - Inference Script V3.0

使用多通道 heatmap 模型，從每個 channel 的 peak 提取角點座標。

用法:
    # 單張影像
    python inference_vertebra.py --input image.png

    # 批次處理資料夾
    python inference_vertebra.py --input ./images/ --output ./results/

    # 指定模型和脊椎類型
    python inference_vertebra.py --input image.dcm --model best_vertebra_model.pth --spine-type L
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from train_vertebra_model import VertebraCornerModel, HEATMAP_SIZE
from spine_metrics import calculate_metrics, calculate_discs


# 椎體名稱對照表
VERTEBRA_NAMES = {
    'L': ['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'],
    'C': ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1'],
}

# 邊界椎體定義
BOUNDARY_CONFIG = {
    'L': {'upper': ['S1'], 'lower': ['T12']},
    'C': {'upper': ['T1'], 'lower': ['C2']},
}

# 命名錨定方向：解碼出 N 節椎體後，從哪一端開始對應解剖名稱
#   'bottom' = 從清單尾端往回取 names[-N:]（最底那節 = 清單最後一個）
#   'top'    = 從清單開頭取 names[:N]（最頂那節 = 清單第一個）
# L-spine：薦椎 S1 幾乎必入鏡且解剖最穩定，頂端常被裁切 → 從 S1 往上錨定
# C-spine：C2 齒突最明顯且通常入鏡，T1 常被肩膀擋住 → 從 C2 往下錨定
ANCHOR_CONFIG = {
    'L': 'bottom',
    'C': 'top',
}

# 4 個角點名稱 (固定順序)
CORNER_NAMES = ['anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior']

# 角點繪製顏色 (BGR)
CORNER_COLORS = [
    (0, 255, 0),    # anteriorSuperior - 綠
    (255, 0, 0),    # posteriorSuperior - 藍
    (0, 0, 255),    # posteriorInferior - 紅
    (0, 255, 255),  # anteriorInferior - 黃
]


def extract_peaks_from_heatmap(heatmap, threshold=0.3, blur_sigma=6.0, blur_ksize=15):
    """從單通道 heatmap 提取 peak 座標 (sub-pixel 精度)

    DARK-style decode：取峰前先對 heatmap 做 Gaussian 平滑，把多模/雜訊的
    argmax 正規化到真實高斯質心。在 128 heatmap 上 1 px ≈ 原圖 24 px，去噪後
    argmax 更貼近真值，定位精度大幅提升。blur_sigma 預設對齊訓練 label 的
    sigma=6（DARK 理論：decode kernel 應匹配 label kernel）。每個 corner 是
    獨立 channel，blur 不會污染相鄰椎體。

    Args:
        heatmap: [H, W] numpy array, sigmoid 後的值 (0~1)
        threshold: peak 最低信心度
        blur_sigma: 取峰前 Gaussian 平滑 sigma (<=0 關閉)
        blur_ksize: Gaussian kernel 大小 (奇數)

    Returns:
        (x, y, confidence) 或 None
    """
    if blur_sigma and blur_sigma > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        heatmap = cv2.GaussianBlur(heatmap, (k, k), blur_sigma)

    if heatmap.max() < threshold:
        return None

    # 找全域最大值
    max_val = heatmap.max()
    max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
    iy, ix = max_pos

    # Sub-pixel refinement (Taylor expansion)
    h, w = heatmap.shape
    if 1 <= iy < h - 1 and 1 <= ix < w - 1:
        dy = (heatmap[iy + 1, ix] - heatmap[iy - 1, ix]) / 2.0
        dx = (heatmap[iy, ix + 1] - heatmap[iy, ix - 1]) / 2.0
        dyy = heatmap[iy + 1, ix] + heatmap[iy - 1, ix] - 2 * heatmap[iy, ix]
        dxx = heatmap[iy, ix + 1] + heatmap[iy, ix - 1] - 2 * heatmap[iy, ix]

        if abs(dxx) > 1e-6 and abs(dyy) > 1e-6:
            offset_x = -dx / (dxx + 1e-8)
            offset_y = -dy / (dyy + 1e-8)
            # 限制偏移量
            offset_x = max(-0.5, min(0.5, offset_x))
            offset_y = max(-0.5, min(0.5, offset_y))
        else:
            offset_x, offset_y = 0, 0

        refined_x = ix + offset_x
        refined_y = iy + offset_y
    else:
        refined_x = float(ix)
        refined_y = float(iy)

    return (refined_x, refined_y, float(max_val))


class VertebraInference:
    """椎體頂點檢測推理器 V3.0 (Heatmap-based)"""

    def __init__(self, model_path, device='auto', max_vertebrae=8, ensemble_paths=None):
        """
        Args:
            model_path: 主模型權重路徑
            ensemble_paths: 額外權重路徑清單。給定時，推論在機率空間平均所有模型
                            （含主模型）的 heatmap → ensemble 推論，壓低訓練變異。
                            count head 仍取主模型 (避免不同模型 count 不一致造成命名亂跳)。
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.max_vertebrae = max_vertebrae
        print(f"Device: {self.device}")

        # 載入模型
        print(f"Loading model: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint.get('config', {})
        self.model_version = checkpoint.get('model_version', 'unknown')
        self.heatmap_size = checkpoint.get('heatmap_size', HEATMAP_SIZE)

        # 偵測 checkpoint 版本: V2 (backbone.*) vs V3 (layer0.*)
        state_dict = checkpoint['model_state_dict']
        is_v2_checkpoint = any(k.startswith('backbone.') for k in state_dict.keys())
        self._is_v2 = False  # 預設 V3

        if is_v2_checkpoint:
            print("  Detected V2 checkpoint -> loading with V2 model architecture")
            self.model_version = 'v2_legacy'
            self._load_v2_model(state_dict, max_vertebrae)
        else:
            self.model = VertebraCornerModel(max_vertebrae=max_vertebrae, pretrained=False)
            try:
                self.model.load_state_dict(state_dict)
                # 偵測 V3.1 (有 channel_embed) vs V3.0
                if 'channel_embed' in state_dict:
                    self.model_version = checkpoint.get('model_version', 'v3.1')
                else:
                    self.model_version = checkpoint.get('model_version', 'v3.0')
            except RuntimeError as e:
                # V3.0 舊版 checkpoint 載入到 V3.1 模型 → strict=False
                print(f"  Warning: Checkpoint key mismatch, loading with strict=False")
                print(f"  (Missing keys likely: coord_conv, channel_embed, heatmap_bn, etc.)")
                self.model.load_state_dict(state_dict, strict=False)
                self.model_version = 'v3_partial'

        self.model = self.model.to(self.device)
        self.model.eval()

        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', None)
        backbone_src = checkpoint.get('backbone_source', None)
        suffix = f", val_loss {val_loss:.4f}" if val_loss else ""
        if backbone_src:
            suffix += f", backbone={backbone_src}"
        print(f"Model V{self.model_version} loaded (epoch {epoch}{suffix})")

        # ── Ensemble：載入額外權重（須與主模型同架構 V3）──
        self.ensemble_models = []
        if ensemble_paths:
            for p in ensemble_paths:
                try:
                    ck = torch.load(p, map_location=self.device, weights_only=False)
                    sd = ck['model_state_dict']
                    m = VertebraCornerModel(max_vertebrae=max_vertebrae, pretrained=False)
                    m.load_state_dict(sd)
                    m = m.to(self.device).eval()
                    self.ensemble_models.append(m)
                    vl = ck.get('val_loss')
                    print(f"  + ensemble member: {os.path.basename(p)}"
                          + (f" (val_loss {vl:.4f})" if vl else ""))
                except Exception as e:
                    print(f"  [ensemble] 跳過 {p}: {e}")
            if self.ensemble_models:
                print(f"  Ensemble 啟用：主模型 + {len(self.ensemble_models)} 個成員，共 "
                      f"{1 + len(self.ensemble_models)} 模型")

    def _load_v2_model(self, state_dict, max_vertebrae):
        """載入 V2 舊版模型 checkpoint (backbone + heatmap_branch + regression_branch)"""
        import torch.nn as nn
        from torchvision import models

        class V2Model(nn.Module):
            def __init__(self, max_vertebrae=8):
                super().__init__()
                self.max_vertebrae = max_vertebrae
                self.num_points = max_vertebrae * 4

                resnet = models.resnet50(pretrained=False)
                self.backbone = nn.Sequential(*list(resnet.children())[:-2])

                self.heatmap_branch = nn.Sequential(
                    nn.Conv2d(2048, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(64, 1, 1), nn.Sigmoid()
                )

                self.regression_branch = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Dropout(0.3),
                    nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
                    nn.Linear(512, self.num_points * 2), nn.Sigmoid()
                )

                self.count_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(2048, 256), nn.ReLU(inplace=True),
                    nn.Linear(256, max_vertebrae + 1)
                )

            def forward(self, x):
                features = self.backbone(x)
                heatmap = self.heatmap_branch(features)
                coords = self.regression_branch(features)
                coords = coords.view(-1, self.num_points, 2)
                count_logits = self.count_head(features)
                return {
                    'heatmap': heatmap,
                    'coords': coords,
                    'count_logits': count_logits,
                    # V3 相容欄位
                    'heatmaps': heatmap,
                }

        self.model = V2Model(max_vertebrae=max_vertebrae)
        self.model.load_state_dict(state_dict)
        self._is_v2 = True

    def load_image(self, image_path):
        image_path = str(image_path)

        if image_path.lower().endswith('.dcm'):
            dcm = pydicom.dcmread(image_path)
            image = dcm.pixel_array
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            # cv2.imread 在 Windows 對含非 ASCII 字元的路徑 (中文資料夾) 會 fail，
            # 改走 np.fromfile + imdecode 讓 Python 自己處理 file IO
            try:
                buf = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
            except Exception:
                image = None
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def predict(self, image_path, spine_type='L', confidence_threshold=0.2):
        """
        預測椎體頂點座標

        V3 模型: 從多通道 heatmap peak 提取座標
        V2 模型: 從回歸分支提取座標 (向下相容)

        Args:
            image_path: 影像路徑
            spine_type: 'L' (腰椎) 或 'C' (頸椎)
            confidence_threshold: heatmap peak 最低信心度

        Returns:
            dict: 包含椎體名稱、角點座標、信心度等
        """
        image = self.load_image(image_path)
        original_h, original_w = image.shape[:2]

        # ── 預處理 ──
        # V3.4 以後改用 keep-aspect resize + pad，與訓練端 LongestMaxSize+PadIfNeeded 一致；
        # 舊版 V3.0~V3.2 weights 仍用 squash resize 維持相容。
        use_aspect_aware = str(getattr(self, 'model_version', '')).startswith('v3.4')
        if use_aspect_aware:
            scale = 512.0 / max(original_h, original_w)
            new_h = int(round(original_h * scale))
            new_w = int(round(original_w * scale))
            resized_keep = cv2.resize(image, (new_w, new_h))
            pad_top = (512 - new_h) // 2
            pad_bottom = 512 - new_h - pad_top
            pad_left = (512 - new_w) // 2
            pad_right = 512 - new_w - pad_left
            resized = cv2.copyMakeBorder(
                resized_keep, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            resized = cv2.resize(image, (512, 512))
            scale = None
            pad_top = pad_left = 0

        # ── 推理（V3 模型支援 ensemble / TTA；V2 走單次 forward）──
        is_v3 = not getattr(self, '_is_v2', False)
        use_ensemble = is_v3 and bool(getattr(self, 'ensemble_models', []))
        tta_enabled = is_v3 and getattr(self, 'tta', True)
        if use_ensemble:
            # 多模型 + 各自 TTA，機率空間平均；count 取主模型
            prob, count_logits = self._predict_heatmaps_ensemble(resized)
            eps = 1e-6
            probc = np.clip(prob, eps, 1.0 - eps)
            logit = np.log(probc / (1.0 - probc)).astype(np.float32)
            predictions = {
                'heatmaps': torch.from_numpy(logit).unsqueeze(0).to(self.device),
                'count_logits': torch.from_numpy(count_logits.astype(np.float32)).unsqueeze(0).to(self.device),
            }
        elif tta_enabled:
            # 在機率空間平均多個變體（旋轉變體會把 heatmap 逆旋轉回正規座標），
            # 再轉回 logit 餵給既有 decode（decode 內部會再 sigmoid）。
            prob, count_logits = self._predict_heatmaps_tta(resized)
            eps = 1e-6
            probc = np.clip(prob, eps, 1.0 - eps)
            logit = np.log(probc / (1.0 - probc)).astype(np.float32)
            predictions = {
                'heatmaps': torch.from_numpy(logit).unsqueeze(0).to(self.device),
                'count_logits': torch.from_numpy(count_logits.astype(np.float32)).unsqueeze(0).to(self.device),
            }
        else:
            predictions = self._forward_normalized(resized)
            count_logits = predictions['count_logits'][0].cpu().numpy()
        predicted_count = int(np.argmax(count_logits))
        count_confidence = float(np.exp(count_logits[predicted_count]) / np.exp(count_logits).sum())

        names = VERTEBRA_NAMES.get(spine_type, VERTEBRA_NAMES['L'])
        boundary = BOUNDARY_CONFIG.get(spine_type, {})

        # ── 命名錨定：把「第 i 個解碼 slot（由上而下）」對應到解剖名稱 ──
        # 模型的 32 channel 是 top-down 位置 slot（slot 0 = 最頂那節）。命名若一律
        # 從 T12 起算，當影像沒有 T12（頂端被裁）但 count 數多 1 時，整條序列往下
        # 錯位一格 → catastrophe（如 81161252、21584353，mean 700-800px）。
        # 改成依 spine_type 從穩定端錨定（L 從 S1、C 從 C2）。
        anchor = getattr(self, 'anchor_mode', None) or ANCHOR_CONFIG.get(spine_type, 'bottom')
        slot_names = self._assign_slot_names(names, predicted_count, anchor)

        # ── 根據模型版本選擇解析方式 ──
        if getattr(self, '_is_v2', False):
            vertebrae, combined_heatmap, channel_heatmaps = self._decode_v2(
                predictions, predicted_count, slot_names, boundary, original_w, original_h
            )
        else:
            vertebrae, combined_heatmap, channel_heatmaps = self._decode_v3(
                predictions, predicted_count, slot_names, boundary,
                original_w, original_h, confidence_threshold,
                aspect_scale=scale, pad_top=pad_top, pad_left=pad_left,
            )

        # ── Post-process: hard aspect-ratio constraint ──
        # V3.4 訓練後人工檢視發現「同椎體 4 corner 被學成垂直長條」(PRED aspect 0.5
        # vs GT ~1.2)，且 V3.5 / V3.6 重訓嘗試 5 輪都無法解。此處不動模型，僅在
        # 推論後對 4 角點完整的椎體做 hard fix：aspect < 0.7 時，把 width 拉到
        # 1.2*height (X 軸 scale 圍中心放大)。Y 不動 (Y 是 V3.4 較弱的軸，X 是
        # 較準的軸，調 X 風險低)。Boundary 椎體只有 2 角點，跳過。
        self._apply_aspect_fix(vertebrae,
                               min_aspect=getattr(self, 'aspect_min', 0.9),
                               max_scale=getattr(self, 'aspect_max_scale', 1.6))

        # 計算椎體指標
        for v in vertebrae:
            self._calculate_metrics(v)

        # 計算椎間盤指標
        discs = self._calculate_discs(vertebrae, spine_type)

        # ── 可信度旗標 ──
        # 實測 corner_min（所有角點 confidence 的最小值）與定位誤差相關 -0.68：
        # 好案例 corner_min ~0.67-0.69，定位失敗案例 ~0.33-0.43。低於門檻時標記
        # 「需人工複查」。不改座標，僅提供臨床 triage 訊號（剩餘 catastrophe 屬
        # 模型在難圖上的真實失敗、無法後處理修正，標記它們是負責任的作法）。
        all_confs = []
        for v in vertebrae:
            all_confs += list(v.get('confidences', {}).values())
        corner_min = float(min(all_confs)) if all_confs else 0.0
        corner_avg = float(np.mean(all_confs)) if all_confs else 0.0
        reliability_threshold = getattr(self, 'reliability_threshold', 0.5)
        low_confidence = corner_min < reliability_threshold

        return {
            'image_path': str(image_path),
            'spine_type': spine_type,
            'image_info': {'width': original_w, 'height': original_h},
            'predicted_count': predicted_count,
            'count_confidence': count_confidence,
            'corner_confidence_min': corner_min,
            'corner_confidence_avg': corner_avg,
            'low_confidence': low_confidence,   # True = 建議人工完整複查
            'vertebrae': vertebrae,
            'discs': discs,
            'heatmap': combined_heatmap,
            'channel_heatmaps': channel_heatmaps,
            'original_image': image,
        }

    # ── TTA 共用前處理常數 ──
    _IMNET_MEAN = (0.485, 0.456, 0.406)
    _IMNET_STD = (0.229, 0.224, 0.225)

    def _forward_normalized(self, resized, model=None):
        """把 aspect-aware 512×512 影像 (HxWx3, 0-255) normalize 後 forward 一次。
        model 為 None 時用主模型。回傳模型原始 predictions dict。
        """
        net = model if model is not None else self.model
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self._IMNET_MEAN).view(3, 1, 1)
        std = torch.tensor(self._IMNET_STD).view(3, 1, 1)
        t = (t - mean) / std
        t = t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            return net(t)

    def _tta_ops(self):
        """回傳 TTA 變體清單。可用 inf.tta_ops 覆寫。

        每個元素：('identity',) / ('rot', 角度) / ('photo', 對比因子)
        預設保守組合，對齊訓練 aug (Rotate ±10°, brightness ±15%)，
        不含水平翻轉 (lateral spine 前後方向有醫學意義)。
        """
        custom = getattr(self, 'tta_ops', None)
        if custom:
            return custom
        return [('identity',), ('rot', 6.0), ('rot', -6.0),
                ('photo', 0.85), ('photo', 1.15)]

    def _predict_heatmaps_tta(self, resized, model=None):
        """對多個 TTA 變體 forward，於機率空間平均。
        旋轉變體會把輸出 heatmap 逆旋轉回正規座標再累加。
        model 為 None 時用主模型。

        回傳 (avg_prob[C,h,w] np.float32, identity_count_logits[K] np.float32)
        """
        h, w = resized.shape[:2]
        acc_prob = None
        nv = 0
        identity_clog = None  # count 只取 identity 那次，不被旋轉/亮度變體擾動
        for op in self._tta_ops():
            kind = op[0]
            if kind == 'identity':
                img = resized
                inv_angle = None
            elif kind == 'rot':
                ang = float(op[1])
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
                img = cv2.warpAffine(resized, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
                inv_angle = -ang
            elif kind == 'photo':
                f = float(op[1])
                img = np.clip(resized.astype(np.float32) * f, 0, 255)
                inv_angle = None
            else:
                continue

            pred = self._forward_normalized(img, model=model)
            prob = torch.sigmoid(pred['heatmaps'][0]).cpu().numpy().astype(np.float32)  # [C,h,w]

            if kind == 'identity':
                identity_clog = pred['count_logits'][0].cpu().numpy().astype(np.float32)

            if inv_angle is not None:
                hh, ww = prob.shape[1], prob.shape[2]
                Minv = cv2.getRotationMatrix2D((ww / 2.0, hh / 2.0), inv_angle, 1.0)
                # 逆旋轉每個 channel；旋轉到框外的區域填 0 (= 無 activation)
                prob = np.stack([
                    cv2.warpAffine(prob[c], Minv, (ww, hh),
                                   flags=cv2.INTER_LINEAR, borderValue=0.0)
                    for c in range(prob.shape[0])
                ], axis=0)

            acc_prob = prob if acc_prob is None else acc_prob + prob
            nv += 1

        # count_logits 用 identity 那次（若 TTA 清單沒含 identity 則退而求其次跑一次）
        if identity_clog is None:
            pred = self._forward_normalized(resized, model=model)
            identity_clog = pred['count_logits'][0].cpu().numpy().astype(np.float32)

        return acc_prob / max(nv, 1), identity_clog

    def _predict_heatmaps_ensemble(self, resized):
        """Ensemble：對主模型 + 每個 ensemble 成員各跑 TTA，於機率空間平均所有模型。
        count_logits 只取主模型（避免不同模型 count 不一致造成 anchor 命名亂跳）。

        回傳 (avg_prob[C,h,w], main_count_logits[K])
        """
        prob_main, clog_main = self._predict_heatmaps_tta(resized, model=self.model)
        acc = prob_main.copy()
        nm = 1
        for m in self.ensemble_models:
            p, _ = self._predict_heatmaps_tta(resized, model=m)
            acc += p
            nm += 1
        return acc / nm, clog_main

    def _assign_slot_names(self, names, count, anchor='bottom'):
        """把解碼 slot（由上而下，slot 0 = 最頂那節）對應到解剖名稱清單。

        anchor='bottom'：取 names[-count:]（最底那節 = 清單最後一個，如 S1）
        anchor='top'   ：取 names[:count]（最頂那節 = 清單第一個，如 T12 / C2）

        count 超過清單長度時，多出來的 slot 命名為 V{n}。
        """
        count = max(0, int(count))
        n = len(names)
        if count <= n:
            window = list(names[-count:]) if (anchor == 'bottom' and count > 0) else list(names[:count])
        else:
            # 偵測到的椎體比清單還多（理論上少見）：以清單為基礎，超出部分補 V{n}
            base = list(names)
            extra = [f'V{n + k + 1}' for k in range(count - n)]
            window = base + extra if anchor == 'top' else extra + base
        return window

    def _decode_v2(self, predictions, predicted_count, slot_names, boundary, orig_w, orig_h):
        """V2 模型: 從回歸分支的 normalized coords 解析"""
        coords = predictions['coords'][0].cpu().numpy()  # [N*4, 2] normalized [0,1]
        heatmap_tensor = predictions['heatmap'][0, 0].cpu().numpy()  # [H, W]

        vertebrae = []
        for i in range(min(predicted_count, self.max_vertebrae)):
            name = slot_names[i] if i < len(slot_names) else f'V{i+1}'
            base_idx = i * 4

            if name in boundary.get('upper', []):
                boundary_type = 'upper'
            elif name in boundary.get('lower', []):
                boundary_type = 'lower'
            else:
                boundary_type = None

            corners = {}
            for j in range(4):
                corner_name = CORNER_NAMES[j]
                if boundary_type == 'upper' and j >= 2:
                    continue
                if boundary_type == 'lower' and j < 2:
                    continue

                x_norm, y_norm = coords[base_idx + j]
                corners[corner_name] = {
                    'x': float(x_norm * orig_w),
                    'y': float(y_norm * orig_h),
                }

            if corners:
                vertebrae.append({
                    'name': name,
                    'boundaryType': boundary_type,
                    'points': corners,
                    'confidences': {},
                })

        return vertebrae, heatmap_tensor, None

    def _decode_v3(self, predictions, predicted_count, slot_names, boundary,
                   orig_w, orig_h, confidence_threshold,
                   aspect_scale=None, pad_top=0, pad_left=0):
        """V3 模型: 從多通道 heatmap peak 提取座標

        aspect_scale 非 None 時走 V3.4 keep-aspect 反算 (heatmap → 512 input
        → 減 pad → 除以 scale → 原圖座標)；否則沿用 squash 模式 (heatmap → 原圖)。
        """
        heatmaps = torch.sigmoid(predictions['heatmaps'][0]).cpu().numpy()  # [C, H, W]
        hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]

        vertebrae = []
        for i in range(min(predicted_count, self.max_vertebrae)):
            name = slot_names[i] if i < len(slot_names) else f'V{i+1}'
            base_idx = i * 4

            if name in boundary.get('upper', []):
                boundary_type = 'upper'
            elif name in boundary.get('lower', []):
                boundary_type = 'lower'
            else:
                boundary_type = None

            corners = {}
            corner_confidences = {}

            for j in range(4):
                corner_name = CORNER_NAMES[j]
                if boundary_type == 'upper' and j >= 2:
                    continue
                if boundary_type == 'lower' and j < 2:
                    continue

                ch_idx = base_idx + j
                if ch_idx >= heatmaps.shape[0]:
                    continue

                peak = extract_peaks_from_heatmap(
                    heatmaps[ch_idx], threshold=confidence_threshold,
                    blur_sigma=getattr(self, 'decode_blur_sigma', 6.0),
                    blur_ksize=getattr(self, 'decode_blur_ksize', 15),
                )
                if peak is not None:
                    px, py, conf = peak
                    if aspect_scale is not None:
                        # V3.4: heatmap → 512 input → unpad → 除 scale → 原圖
                        input_x = (px / hm_w) * 512.0
                        input_y = (py / hm_h) * 512.0
                        orig_x = (input_x - pad_left) / aspect_scale
                        orig_y = (input_y - pad_top) / aspect_scale
                    else:
                        # V3.0~V3.2: heatmap 直接對應 squash 後的原圖
                        orig_x = (px / hm_w) * orig_w
                        orig_y = (py / hm_h) * orig_h
                    corners[corner_name] = {
                        'x': float(orig_x),
                        'y': float(orig_y),
                    }
                    corner_confidences[corner_name] = float(conf)

            if corners:
                vertebrae.append({
                    'name': name,
                    'boundaryType': boundary_type,
                    'points': corners,
                    'confidences': corner_confidences,
                })

        combined_heatmap = heatmaps.max(axis=0)
        return vertebrae, combined_heatmap, heatmaps

    def _apply_aspect_fix(self, vertebrae, min_aspect=0.9, max_scale=1.6):
        """Hard aspect-ratio fix — 對 4 角點完整的椎體，aspect < min_aspect 時
        X 軸圍中心放大到「剛好達到 min_aspect」（不是固定 target，避免過度修正）。
        額外加 max_scale 上限避免極端 case 推太遠。
        Y 軸不動。Boundary 椎體 (2 角點) 跳過。

        Args:
            min_aspect: aspect (W/H) 下限門檻，原 aspect 低於此就修
            max_scale: X 軸 scale 上限，避免極端拉伸 (e.g. aspect=0.3 不會被推成 3x)
        """
        n_fixed = 0
        for v in vertebrae:
            pts = v.get('points', {})
            if len(pts) < 4:
                continue
            xs = [p['x'] for p in pts.values()]
            ys = [p['y'] for p in pts.values()]
            cx = sum(xs) / 4.0
            W = max(xs) - min(xs)
            H = max(ys) - min(ys)
            if H <= 0 or W <= 0:
                continue
            aspect = W / H
            if aspect >= min_aspect:
                continue
            # 只拉到剛好達門檻 = min_aspect * H，且 scale 不超過 max_scale
            target_W = min_aspect * H
            scale_x = min(target_W / W, max_scale)
            for p in pts.values():
                p['x'] = cx + (p['x'] - cx) * scale_x
            n_fixed += 1
        if n_fixed > 0:
            v_aspect = getattr(self, '_aspect_fix_verbose', False)
            if v_aspect:
                print(f"  [aspect-fix] adjusted {n_fixed} vertebrae")

    def _calculate_metrics(self, vertebra):
        """委託給 spine_metrics 共用模組 (Genant classification)"""
        calculate_metrics(vertebra)

    def _calculate_discs(self, vertebrae, spine_type):
        """委託給 spine_metrics 共用模組"""
        return calculate_discs(vertebrae, spine_type)

    def visualize(self, result, output_path=None, show=False):
        """視覺化預測結果"""
        image = result['original_image'].copy()
        vertebrae = result['vertebrae']
        heatmap = result['heatmap']
        original_h, original_w = image.shape[:2]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(
            f"Vertebra Detection V3 - {result['spine_type']}-spine "
            f"({result['predicted_count']} vertebrae, "
            f"confidence: {result['count_confidence']:.1%})",
            fontsize=14, fontweight='bold'
        )

        # 1. 原始影像 + 角點
        vis_image = image.copy()
        for v in vertebrae:
            pts = v['points']
            confs = v.get('confidences', {})
            corners_xy = []

            for j, corner_name in enumerate(CORNER_NAMES):
                if corner_name not in pts:
                    continue
                p = pts[corner_name]
                x, y = int(p['x']), int(p['y'])
                corners_xy.append((x, y, j))

                color = CORNER_COLORS[j]
                radius = max(3, min(original_w, original_h) // 200)
                cv2.circle(vis_image, (x, y), radius, color, -1)

                # 信心度標記
                conf = confs.get(corner_name, 0)
                if conf > 0:
                    cv2.putText(vis_image, f"{conf:.0%}",
                               (x + radius + 2, y - 2),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               max(0.3, min(original_w, original_h) / 3000),
                               color, 1)

            # 畫椎體輪廓
            if len(corners_xy) >= 2:
                pts_arr = [(c[0], c[1]) for c in corners_xy]
                for k in range(len(pts_arr)):
                    cv2.line(vis_image, pts_arr[k], pts_arr[(k+1) % len(pts_arr)],
                             (255, 255, 255), max(1, min(original_w, original_h) // 400))

            # 標記椎體名稱
            if corners_xy:
                avg_x = int(np.mean([c[0] for c in corners_xy]))
                avg_y = int(np.mean([c[1] for c in corners_xy]))
                font_scale = max(0.4, min(original_w, original_h) / 2000)
                thickness = max(1, int(font_scale * 2))

                label = v['name']
                if v.get('anteriorWedgingFracture'):
                    label += ' [AW]'
                elif v.get('crushDeformityFracture'):
                    label += ' [Crush]'

                cv2.putText(vis_image, label, (avg_x - 20, avg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

        axes[0].imshow(vis_image)
        axes[0].set_title('Vertebra Corners (from heatmap peaks)')
        axes[0].axis('off')

        # 2. 熱圖疊加 (合併所有 channel)
        heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
        axes[1].imshow(image)
        axes[1].imshow(heatmap_resized, alpha=0.5, cmap='hot')
        axes[1].set_title('Combined Keypoint Heatmap')
        axes[1].axis('off')

        # 3. 各 channel heatmap 拼接預覽 (前幾個有效 channel)
        channel_heatmaps = result.get('channel_heatmaps', None)
        if channel_heatmaps is not None:
            # 只顯示有效的 channels (有 peak 的)
            active_channels = []
            for ch_idx in range(channel_heatmaps.shape[0]):
                if channel_heatmaps[ch_idx].max() > 0.1:
                    active_channels.append(ch_idx)

            if active_channels:
                n_show = min(len(active_channels), 16)
                rows = int(np.ceil(np.sqrt(n_show)))
                cols = int(np.ceil(n_show / rows))
                grid = np.zeros((rows * channel_heatmaps.shape[1], cols * channel_heatmaps.shape[2]))

                for idx, ch_idx in enumerate(active_channels[:n_show]):
                    r, c = idx // cols, idx % cols
                    h, w = channel_heatmaps.shape[1], channel_heatmaps.shape[2]
                    grid[r*h:(r+1)*h, c*w:(c+1)*w] = channel_heatmaps[ch_idx]

                axes[2].imshow(grid, cmap='hot')
                axes[2].set_title(f'Individual Channel Heatmaps ({len(active_channels)} active)')
            else:
                axes[2].imshow(image)
                axes[2].set_title('No active channels')
        else:
            # Disc 分析 fallback
            disc_image = image.copy()
            discs = result['discs']
            for disc in discs:
                axes[2].text(0.05, 0.95 - discs.index(disc) * 0.08,
                            f"{disc['level']}: A={disc['anteriorHeight']:.0f} P={disc['posteriorHeight']:.0f}",
                            transform=axes[2].transAxes, fontsize=10,
                            verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            axes[2].imshow(disc_image)
            axes[2].set_title('Disc Analysis')

        axes[2].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {output_path}")

        if show:
            plt.show()

        plt.close()

    def analyze(self, image_path, spine_type='L', output_dir=None, visualize=True):
        """完整分析流程"""
        print(f"\nAnalyzing: {image_path} ({spine_type}-spine)")

        result = self.predict(image_path, spine_type)

        # 打印結果
        print(f"  Predicted vertebrae: {result['predicted_count']} (confidence: {result['count_confidence']:.1%})")
        for v in result['vertebrae']:
            bt_label = f" [{v['boundaryType']}]" if v['boundaryType'] else ""
            confs = v.get('confidences', {})
            avg_conf = np.mean(list(confs.values())) if confs else 0
            ratio_label = ""
            if v.get('heightRatio') is not None:
                ratio_label = f" (A/P={v['heightRatio']:.2f})"
                if v.get('anteriorWedgingFracture'):
                    ratio_label += " !! Anterior Wedging"
                if v.get('crushDeformityFracture'):
                    ratio_label += " !! Crush Deformity"
            print(f"  {v['name']}{bt_label}: {len(v['points'])} corners, avg_conf={avg_conf:.1%}{ratio_label}")

        for d in result['discs']:
            print(f"  Disc {d['level']}: A={d['anteriorHeight']:.1f} P={d['posteriorHeight']:.1f}")

        # 儲存結果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            # JSON 結果
            json_result = {k: v for k, v in result.items()
                          if k not in ('heatmap', 'channel_heatmaps', 'original_image')}
            json_path = output_dir / f"{Path(image_path).stem}_vertebra_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            print(f"  JSON saved: {json_path}")

            # 視覺化
            if visualize:
                viz_path = output_dir / f"{Path(image_path).stem}_vertebra_result.png"
                self.visualize(result, output_path=str(viz_path))

        return result


def main():
    parser = argparse.ArgumentParser(description='Spine Vertebra Corner Detection Inference V3')
    parser.add_argument('--model', type=str, default='best_vertebra_model.pth',
                       help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory')
    parser.add_argument('--spine-type', type=str, default='L', choices=['L', 'C'],
                       help='Spine type: L (lumbar) or C (cervical)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Compute device')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='Heatmap peak confidence threshold (0-1)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')

    args = parser.parse_args()

    analyzer = VertebraInference(args.model, device=args.device)

    input_path = Path(args.input)

    if input_path.is_file():
        analyzer.analyze(
            input_path,
            spine_type=args.spine_type,
            output_dir=args.output,
            visualize=not args.no_viz
        )
    elif input_path.is_dir():
        image_files = (
            list(input_path.glob('*.dcm')) +
            list(input_path.glob('*.png')) +
            list(input_path.glob('*.jpg'))
        )
        print(f"\nBatch processing: {len(image_files)} files")

        for img_file in image_files:
            try:
                analyzer.analyze(
                    img_file,
                    spine_type=args.spine_type,
                    output_dir=args.output,
                    visualize=not args.no_viz
                )
            except Exception as e:
                print(f"ERROR processing {img_file}: {e}")
    else:
        print(f"Invalid input path: {args.input}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
