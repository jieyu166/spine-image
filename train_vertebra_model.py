#!/usr/bin/env python3
"""
脊椎椎體頂點檢測 - 機器學習訓練腳本 V3.2
Spine Vertebra Corner Detection - ML Training Script V3.2

V3.2 改進 (縮小 domain gap):
- Backbone 改用 RadImageNet ResNet50 預訓練 (1.35M 放射影像)
  取代原本 ImageNet (貓狗車) 權重，大幅降低 X-ray domain gap
- 權重下載: https://github.com/BMEII-AI/RadImageNet
  放到 pretrained/RadImageNet-ResNet50.pt (不存在時自動 fallback 到 ImageNet)
- 新增 CLI: --epochs / --pretrained-path，方便煙霧測試

V3.1 改進 (解決小樣本模式崩塌問題):
- 凍結 backbone layer0~layer2: 減少可訓練參數從 36M → ~13M
- CoordConv: 讓 decoder 直接感知空間座標
- Channel Embedding: 32 個 learnable embedding 讓每個 channel 學習不同空間位置
- RepeatDataset: 每 epoch 重複 8 次增加 augmentation 多樣性
- Dropout2d + 更強 weight_decay: 防止小數據集過擬合

V3.0 基礎:
- 多通道 heatmap: 每個角點 slot 獨立 channel (max_vertebrae*4 channels)
- UNet Decoder with skip connections
- Focal Loss: 解決熱圖正負樣本不平衡
- 保留椎體計數輔助任務

支援:
- 完整椎體: 4 個角點
- 邊界椎體 (S1/T1=上終板 2點, T12/C2=下終板 2點)

日期: 2025-2026
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pydicom
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')


# ── Heatmap 輸出解析度 ──
HEATMAP_SIZE = 128  # 輸出 128x128 heatmap (比 512 小，減少記憶體)

# ── RadImageNet 預訓練權重 ──
# V3.2: backbone 改用 RadImageNet (medical image domain) 取代 ImageNet (貓狗車)
# 下載: https://github.com/BMEII-AI/RadImageNet → 放到 pretrained/RadImageNet-ResNet50.pt
RADIMAGENET_PATH_DEFAULT = os.path.join('pretrained', 'RadImageNet-ResNet50.pt')


# nn.Sequential(*list(resnet.children())[:-2]) 時各層的 index 對應關係
# 用於把 "backbone.N.X" 格式的 RadImageNet 權重 remap 回 torchvision resnet50 的扁平 key
_SEQ_TO_RESNET = {
    '0': 'conv1',
    '1': 'bn1',
    # '2': relu (無參數), '3': maxpool (無參數)
    '4': 'layer1',
    '5': 'layer2',
    '6': 'layer3',
    '7': 'layer4',
}


def _remap_sequential_keys(state_dict, prefix):
    """把 nn.Sequential 格式 key (如 backbone.0.weight) remap 回 torchvision 扁平 key"""
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            rest = k[len(prefix):]       # e.g. "0.weight" / "4.0.conv1.weight"
            dot = rest.find('.')
            if dot < 0:
                continue
            idx, suffix = rest[:dot], rest[dot+1:]
            if idx in _SEQ_TO_RESNET:
                remapped[f'{_SEQ_TO_RESNET[idx]}.{suffix}'] = v
            # 其他 index (relu / maxpool / classifier head) 無參數或不需要
        else:
            remapped[k] = v
    return remapped


def load_radimagenet_weights(resnet, weights_path):
    """載入 RadImageNet ResNet50 權重到 torchvision resnet50

    處理常見 key 差異：
      - 外包一層 {'state_dict': ...} 或 {'model': ...} → 拆開
      - `module.` 前綴 (DataParallel 儲存) → 去掉
      - `backbone.N.X` / `encoder.N.X` / `features.N.X` (nn.Sequential 包裝) → remap
      - `fc.*` 分類頭 → 剔除 (num_classes 不同)
    """
    state_dict = torch.load(weights_path, map_location='cpu')
    if isinstance(state_dict, dict):
        for key in ('state_dict', 'model', 'model_state_dict'):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 偵測 nn.Sequential 包裝 (第一個 key 像是 "backbone.0.weight")
    seq_prefixes = ('backbone.', 'encoder.', 'features.')
    for prefix in seq_prefixes:
        if any(k.startswith(prefix) for k in state_dict):
            print(f"  [RadImageNet] Detected nn.Sequential wrapping (prefix='{prefix}'), remapping keys")
            state_dict = _remap_sequential_keys(state_dict, prefix)
            break

    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}

    missing, unexpected = resnet.load_state_dict(state_dict, strict=False)
    non_fc_missing = [k for k in missing if not k.startswith('fc.')]
    matched = len(state_dict) - len(unexpected)
    print(f"  [RadImageNet] Loaded {weights_path}  matched={matched} keys")
    if non_fc_missing:
        print(f"  [RadImageNet] Warning missing (non-fc): {non_fc_missing[:5]}...")
    if unexpected:
        print(f"  [RadImageNet] Warning unexpected: {unexpected[:5]}...")
    # 如果仍然 matched=0 → 後面整個訓練等同隨機初始化，直接拋錯比較明顯
    if matched == 0:
        raise RuntimeError(
            f"RadImageNet 權重完全沒載入任何 key。請檢查 {weights_path} 格式。"
            f"前 5 個原始 key: {list(torch.load(weights_path, map_location='cpu').keys())[:5]}"
        )


class VertebraDataset(Dataset):
    """椎體頂點檢測數據集 V3.0

    每個角點 slot 對應一個獨立的 heatmap channel。
    max_vertebrae * 4 = 32 channels (預設)。

    支援:
    - 完整椎體: 4 角點 (anteriorSuperior, posteriorSuperior, posteriorInferior, anteriorInferior)
    - 上邊界椎體 (S1/T1): 2 點 (anteriorSuperior, posteriorSuperior)
    - 下邊界椎體 (T12/C2): 2 點 (posteriorInferior, anteriorInferior)
    """

    BOUNDARY_CONFIG = {
        'L': {'upper': ['S1'], 'lower': ['T12']},
        'C': {'upper': ['T1'], 'lower': ['C2']},
    }

    def __init__(self, data_dir, annotations_file, transform=None, max_vertebrae=8):
        self.data_dir = data_dir
        self.transform = transform
        self.max_vertebrae = max_vertebrae
        self.num_channels = max_vertebrae * 4  # 每個角點一個 channel

        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        if isinstance(self.annotations, dict):
            self.annotations = [self.annotations]

        print(f"Loaded {len(self.annotations)} samples (heatmap channels: {self.num_channels})")

    def __len__(self):
        return len(self.annotations)

    def _get_boundary_type(self, name, spine_type, vertebra_data):
        bt = vertebra_data.get('boundaryType', None)
        if bt:
            return bt

        points = vertebra_data.get('points', {})
        if isinstance(points, dict):
            has_all_4 = all(k in points for k in
                ['anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior'])
            if has_all_4:
                return None
        elif isinstance(points, list) and len(points) >= 4:
            return None

        config = self.BOUNDARY_CONFIG.get(spine_type, {})
        if name in config.get('upper', []):
            return 'upper'
        if name in config.get('lower', []):
            return 'lower'
        return None

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        image = self.load_image(annotation)
        if image is None:
            # 影像載入失敗 → 隨機取另一個樣本 (避免 crash)
            fallback_idx = (idx + 1) % len(self.annotations)
            print(f"  Warning: Cannot load image for annotation {idx}, using fallback {fallback_idx}")
            return self.__getitem__(fallback_idx)

        original_h, original_w = image.shape[:2]
        spine_type = annotation.get('spine_type', annotation.get('spineType', 'L'))

        vertebrae = annotation.get('vertebrae', [])
        keypoints = []     # 所有角點座標 (每椎體固定 4 slot)
        valid_flags = []   # 每個 slot 是否有效
        vertebra_names = []

        for v in vertebrae[:self.max_vertebrae]:
            points = v.get('points', {})
            name = v.get('name', '')
            vertebra_names.append(name)
            boundary = self._get_boundary_type(name, spine_type, v)

            if isinstance(points, dict):
                if boundary == 'upper':
                    corners = [
                        points.get('anteriorSuperior', {}),
                        points.get('posteriorSuperior', {}),
                        None, None,
                    ]
                elif boundary == 'lower':
                    corners = [
                        None, None,
                        points.get('posteriorInferior', {}),
                        points.get('anteriorInferior', {}),
                    ]
                else:
                    corners = [
                        points.get('anteriorSuperior', {}),
                        points.get('posteriorSuperior', {}),
                        points.get('posteriorInferior', {}),
                        points.get('anteriorInferior', {}),
                    ]
            else:
                if boundary == 'upper' and len(points) == 2:
                    corners = [points[0], points[1], None, None]
                elif boundary == 'lower' and len(points) == 2:
                    corners = [None, None, points[0], points[1]]
                else:
                    corners = (points[:4] + [None]*4)[:4]

            for corner in corners:
                if corner is not None and corner:
                    x = corner.get('x', 0) if isinstance(corner, dict) else corner[0]
                    y = corner.get('y', 0) if isinstance(corner, dict) else corner[1]
                    keypoints.append([x, y])
                    valid_flags.append(1.0)
                else:
                    keypoints.append([0.0, 0.0])
                    valid_flags.append(0.0)

        # Augmentation (只對有效 keypoints)
        valid_kp_indices = [i for i, f in enumerate(valid_flags) if f > 0]
        valid_kp = [keypoints[i] for i in valid_kp_indices]

        if self.transform and len(valid_kp) > 0:
            transformed = self.transform(image=image, keypoints=valid_kp)
            image = transformed['image']
            transformed_valid_kp = transformed['keypoints']

            # 防禦: Albumentations 某些 transform (CoarseDropout / 大角度旋轉) 即使 remove_invisible=False
            # 也會丟掉出界的 keypoint，導致回傳數量比輸入少 → 差的標記為無效即可
            transformed_kp = [[0.0, 0.0]] * len(keypoints)
            n_out = len(transformed_valid_kp)
            for j, idx_orig in enumerate(valid_kp_indices):
                if j < n_out:
                    transformed_kp[idx_orig] = list(transformed_valid_kp[j])
                else:
                    transformed_kp[idx_orig] = [0.0, 0.0]
                    valid_flags[idx_orig] = 0.0
        else:
            image = cv2.resize(image, (512, 512))
            scale_x = 512 / original_w
            scale_y = 512 / original_h
            transformed_kp = []
            for i, kp in enumerate(keypoints):
                if valid_flags[i] > 0:
                    transformed_kp.append([kp[0] * scale_x, kp[1] * scale_y])
                else:
                    transformed_kp.append([0.0, 0.0])
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # 正規化 keypoints 到 [0, 1]
        normalized_kp = []
        for i, kp in enumerate(transformed_kp):
            if valid_flags[i] > 0:
                normalized_kp.append([kp[0] / 512.0, kp[1] / 512.0])
            else:
                normalized_kp.append([0.0, 0.0])

        # 填充到固定長度
        max_points = self.num_channels
        while len(normalized_kp) < max_points:
            normalized_kp.append([0.0, 0.0])
        while len(valid_flags) < max_points:
            valid_flags.append(0.0)

        # ── 多通道 heatmap: 每個 slot 一個 channel ──
        # sigma=6: 在 128x128 上產生 ~37x37 像素的高斯，確保足夠正樣本
        heatmaps = self.create_multi_channel_heatmap(
            transformed_kp, valid_flags, (HEATMAP_SIZE, HEATMAP_SIZE), sigma=6
        )

        targets = {
            'keypoints': torch.tensor(normalized_kp[:max_points], dtype=torch.float32),
            'valid_mask': torch.tensor(valid_flags[:max_points], dtype=torch.float32),
            'heatmaps': torch.tensor(heatmaps, dtype=torch.float32),  # [C, H, W]
            'num_vertebrae': len(vertebrae),
            'vertebra_names': vertebra_names
        }

        return image, targets

    def load_image(self, annotation):
        image_path = annotation.get('image_path', '')
        full_path = os.path.join(self.data_dir, image_path)

        if os.path.exists(full_path):
            if full_path.lower().endswith('.dcm'):
                try:
                    dcm = pydicom.dcmread(full_path)
                    image = dcm.pixel_array.astype(np.float32)  # float32 避免記憶體爆炸
                    if len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=-1)
                    elif len(image.shape) == 3 and image.shape[2] > 3:
                        image = image[:, :, :3]
                    img_min, img_max = image.min(), image.max()
                    image = ((image - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
                    return image
                except Exception as e:
                    print(f"Error loading DICOM {full_path}: {e}")
            else:
                image = cv2.imread(full_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        source_file = annotation.get('source_file', '')
        if source_file:
            base_name = os.path.splitext(source_file)[0]
            for ext in ['.dcm', '.png', '.jpg']:
                candidate = os.path.join(self.data_dir, base_name + ext)
                if os.path.exists(candidate):
                    return self.load_image({'image_path': base_name + ext})

        return None

    def create_multi_channel_heatmap(self, keypoints, valid_flags, size, sigma=6):
        """每個角點 slot 獨立一個 channel 的高斯熱圖

        sigma=6 在 128x128 上產生 ~37 像素直徑的高斯 (3*sigma 半徑)
        中心值=1.0，sigma 處值≈0.61，2*sigma 處≈0.14
        """
        h, w = size
        num_ch = self.num_channels
        heatmaps = np.zeros((num_ch, h, w), dtype=np.float32)

        # 從 512x512 座標空間映射到 heatmap 空間
        scale_x = w / 512.0
        scale_y = h / 512.0

        radius = int(sigma * 3)  # 高斯半徑

        # 預計算高斯 kernel（避免重複計算）
        diameter = 2 * radius + 1
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma)).astype(np.float32)

        for i in range(min(len(keypoints), num_ch)):
            if valid_flags[i] < 1.0:
                continue
            kp = keypoints[i]
            cx = kp[0] * scale_x
            cy = kp[1] * scale_y
            ix, iy = int(round(cx)), int(round(cy))

            # 跳過完全超出邊界的角點
            if ix < -radius or ix >= w + radius or iy < -radius or iy >= h + radius:
                continue

            # 計算 kernel 在 heatmap 上的有效範圍
            y_min = max(0, iy - radius)
            y_max = min(h, iy + radius + 1)
            x_min = max(0, ix - radius)
            x_max = min(w, ix + radius + 1)

            # 跳過空範圍
            if y_max <= y_min or x_max <= x_min:
                continue

            # 對應 kernel 中的範圍
            ky_min = y_min - (iy - radius)
            ky_max = ky_min + (y_max - y_min)
            kx_min = x_min - (ix - radius)
            kx_max = kx_min + (x_max - x_min)

            heatmaps[i, y_min:y_max, x_min:x_max] = np.maximum(
                heatmaps[i, y_min:y_max, x_min:x_max],
                gaussian_kernel[ky_min:ky_max, kx_min:kx_max]
            )

        return heatmaps


class CoordConv(nn.Module):
    """CoordConv: 在 feature map 上附加歸一化的 x, y 座標通道

    讓網路直接「看到」空間位置，避免所有 channel 收斂到同一點。
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        B, C, H, W = x.shape
        # 產生歸一化座標 [0, 1]
        yy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([x, xx, yy], dim=1)
        return self.conv(x)


class VertebraCornerModel(nn.Module):
    """椎體頂點檢測模型 V3.1

    改進 V3.0 → V3.1:
    - 凍結 backbone layer0~layer2 (只訓練 layer3, layer4, decoder)
    - CoordConv: 每個 decoder 階段注入空間座標
    - Channel Embedding: 32 個 learnable embedding 幫助每個 channel 學習不同位置
    - 更強的 Dropout 避免小數據集過擬合
    """

    def __init__(self, max_vertebrae=8, pretrained=True,
                 pretrained_path=RADIMAGENET_PATH_DEFAULT):
        """
        Args:
            pretrained: True=載入 pretrained backbone, False=隨機初始化 (inference 用)
            pretrained_path: RadImageNet 權重檔路徑。檔案不存在時 fallback 到 ImageNet。
        """
        super(VertebraCornerModel, self).__init__()

        self.max_vertebrae = max_vertebrae
        self.num_channels = max_vertebrae * 4

        # ── Backbone: ResNet50 with medical-domain pretrained weights ──
        # V3.2: 優先用 RadImageNet (1.35M 放射影像)；找不到檔案時 fallback 到 ImageNet。
        # 這是整個改造計畫核心：50 張標註不夠補 ImageNet → X-ray 的 domain gap。
        if pretrained and pretrained_path and os.path.isfile(pretrained_path):
            resnet = models.resnet50(weights=None)
            load_radimagenet_weights(resnet, pretrained_path)
            self.backbone_source = 'radimagenet'
        elif pretrained:
            if pretrained_path:
                print(f"  [Backbone] RadImageNet weights not found at '{pretrained_path}'")
                print(f"  [Backbone] Fallback to torchvision ImageNet pretrained ResNet50")
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.backbone_source = 'imagenet'
        else:
            resnet = models.resnet50(weights=None)
            self.backbone_source = 'random'
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # /4, 64ch
        self.layer1 = resnet.layer1  # /4,  256ch
        self.layer2 = resnet.layer2  # /8,  512ch
        self.layer3 = resnet.layer3  # /16, 1024ch
        self.layer4 = resnet.layer4  # /32, 2048ch

        # ── 凍結整個 backbone (layer0~layer4) ──
        # 29 張圖無法有效訓練 ResNet50 backbone (23M params)
        # 只訓練 decoder + heatmap head (~12M params)
        for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = False

        # Decoder (帶完整 skip connection + CoordConv)
        # up4: x4(2048) → 上採樣 → concat x3(1024) → conv
        self.up4_pre = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # up3: d4(256) → 上採樣 → concat x2(512) → conv
        self.up3 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # up2: d3(256) → 上採樣 → concat x1(256) → conv
        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # ── CoordConv + Channel Embedding heatmap head ──
        # CoordConv 注入 x, y 座標讓模型知道空間位置
        self.coord_conv = CoordConv(128, 64, kernel_size=3, padding=1)
        self.heatmap_bn = nn.BatchNorm2d(64)
        self.heatmap_relu = nn.ReLU(inplace=True)

        # Channel embedding: 每個 channel 學習一個獨特的空間 bias
        # 這確保不同 channel 自然傾向於不同空間位置
        self.channel_embed = nn.Parameter(torch.randn(self.num_channels, 64) * 0.02)

        # 最終 1x1 conv: 64 -> num_channels
        self.heatmap_final = nn.Conv2d(64, self.num_channels, 1)

        # 椎體計數 head
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, max_vertebrae + 1)
        )

    def forward(self, x):
        # Encoder (layer0~layer2 已凍結，但仍需 forward 產生 skip connections)
        x0 = self.layer0(x)    # [B, 64,  H/4,  W/4]
        x1 = self.layer1(x0)   # [B, 256, H/4,  W/4]
        x2 = self.layer2(x1)   # [B, 512, H/8,  W/8]
        x3 = self.layer3(x2)   # [B, 1024, H/16, W/16]
        x4 = self.layer4(x3)   # [B, 2048, H/32, W/32]

        # Decoder with full skip connections at every level
        # up4: x4 → upsample → concat x3
        d4_pre = self.up4_pre(x4)                                          # [B, 512, H/32, W/32]
        d4_up = torch.nn.functional.interpolate(d4_pre, size=x3.shape[2:],
                    mode='bilinear', align_corners=True)                   # [B, 512, H/16, W/16]
        d4 = self.up4(torch.cat([d4_up, x3], dim=1))                      # [B, 256, H/16, W/16]

        # up3: d4 → upsample → concat x2
        d3_up = torch.nn.functional.interpolate(d4, size=x2.shape[2:],
                    mode='bilinear', align_corners=True)                   # [B, 256, H/8, W/8]
        d3 = self.up3(torch.cat([d3_up, x2], dim=1))                      # [B, 256, H/8, W/8]

        # up2: d3 → upsample → concat x1
        d2_up = torch.nn.functional.interpolate(d3, size=x1.shape[2:],
                    mode='bilinear', align_corners=True)                   # [B, 256, H/4, W/4]
        d2 = self.up2(torch.cat([d2_up, x1], dim=1))                      # [B, 128, H/4, W/4]

        # ── CoordConv + Channel Embedding ──
        feat = self.coord_conv(d2)        # [B, 64, H/4, W/4]
        feat = self.heatmap_relu(self.heatmap_bn(feat))

        # Channel embedding: 每個 output channel 用獨特的 embedding 向量
        # 對 feat 做加權求和，讓不同 channel 關注不同的空間特徵
        # feat: [B, 64, H, W], channel_embed: [num_channels, 64]
        B, C_feat, H, W = feat.shape
        feat_flat = feat.view(B, C_feat, -1)                   # [B, 64, H*W]
        # einsum: 'cf,bfn->bcn' (c=num_channels, f=64, b=batch, n=H*W)
        heatmaps = torch.einsum('cf,bfn->bcn', self.channel_embed, feat_flat)
        heatmaps = heatmaps.view(B, self.num_channels, H, W)  # [B, num_channels, H, W]

        # 加上 1x1 conv refinement (捕捉 embedding 無法表達的局部模式)
        heatmaps = heatmaps + self.heatmap_final(feat)

        # 調整到目標大小
        if heatmaps.shape[2] != HEATMAP_SIZE or heatmaps.shape[3] != HEATMAP_SIZE:
            heatmaps = torch.nn.functional.interpolate(
                heatmaps, size=(HEATMAP_SIZE, HEATMAP_SIZE),
                mode='bilinear', align_corners=True
            )

        # 椎體計數
        count_logits = self.count_head(x4)

        return {
            'heatmaps': heatmaps,          # [B, C, H, W] raw logits
            'count_logits': count_logits,   # [B, max_vertebrae+1]
        }


class FocalLoss(nn.Module):
    """Modified Focal Loss for heatmap (CornerNet-style)

    解決正負樣本嚴重不平衡。
    正樣本定義: target > pos_threshold (高斯核心區域)
    負樣本: 其餘區域，離高斯中心越近權重越小 (由 (1-target)^beta 控制)
    """

    def __init__(self, alpha=2.0, beta=4.0, pos_threshold=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_threshold = pos_threshold

    def forward(self, pred, target):
        """
        pred:   [B, C, H, W] sigmoid 後的預測 (0~1)
        target: [B, C, H, W] ground truth heatmap (0~1 高斯)
        """
        pred = pred.clamp(1e-6, 1 - 1e-6)

        # 正樣本: 高斯核心區域 (值 > pos_threshold)
        # sigma=6 時，距中心 <=7.5 像素 (1.25*sigma) 的區域值 > 0.3
        pos_mask = target.ge(self.pos_threshold)
        neg_mask = target.lt(self.pos_threshold)

        # 正樣本損失: 鼓勵 pred 接近 1
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask.float()

        # 負樣本損失: 鼓勵 pred 接近 0，離高斯中心越近權重越低
        neg_weight = torch.pow(1 - target, self.beta)
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weight * neg_mask.float()

        num_pos = pos_mask.float().sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss


class VertebraLoss(nn.Module):
    """椎體頂點檢測損失函數 V3"""

    def __init__(self, heatmap_weight=1.0, count_weight=0.5):
        super(VertebraLoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.count_weight = count_weight
        self.focal = FocalLoss(alpha=2.0, beta=4.0)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # 1. 多通道 heatmap 損失 (只計算有效 channel)
        pred_heatmaps = torch.sigmoid(predictions['heatmaps'])  # [B, C, H, W]
        target_heatmaps = targets['heatmaps']                    # [B, C, H, W]
        valid_mask = targets['valid_mask']                        # [B, C]

        # Resize target 到 pred 大小 (如果不同)
        if pred_heatmaps.shape[2:] != target_heatmaps.shape[2:]:
            target_heatmaps = torch.nn.functional.interpolate(
                target_heatmaps, size=pred_heatmaps.shape[2:],
                mode='bilinear', align_corners=True
            )

        # 只計算有效 channel 的損失
        channel_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        masked_pred = pred_heatmaps * channel_mask
        masked_target = target_heatmaps * channel_mask

        heatmap_loss = self.focal(masked_pred, masked_target)

        # 2. 計數損失
        count_logits = predictions['count_logits']
        target_count = torch.tensor([t for t in targets['num_vertebrae']],
                                   dtype=torch.long, device=count_logits.device)
        count_loss = self.ce(count_logits, target_count)

        total_loss = self.heatmap_weight * heatmap_loss + self.count_weight * count_loss

        return {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'count_loss': count_loss
        }


class RepeatDataset(Dataset):
    """包裝 Dataset，讓每個 epoch 重複 N 次以增加 augmentation 多樣性"""

    def __init__(self, dataset, repeat=1):
        self.dataset = dataset
        self.repeat = repeat

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class VertebraTrainer:
    """椎體頂點檢測訓練器 V3.1"""

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # 整個 backbone 已凍結，只訓練 decoder + heatmap head
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_total = sum(p.numel() for p in trainable_params)
        frozen_total = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"  Trainable params: {trainable_total:,}  Frozen backbone: {frozen_total:,}")

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )
        self.criterion = VertebraLoss()

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        components = {'heatmap_loss': 0, 'count_loss': 0}

        progress = tqdm(self.train_loader, desc='Training')
        for images, targets in progress:
            images = images.to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in targets.items()}

            self.optimizer.zero_grad()
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)

            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += losses['total_loss'].item()
            for key in components:
                components[key] += losses[key].item()

            progress.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'HM': f"{losses['heatmap_loss'].item():.4f}"
            })

        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in components.items()}

    def validate(self):
        self.model.eval()
        total_loss = 0
        components = {'heatmap_loss': 0, 'count_loss': 0}

        with torch.no_grad():
            progress = tqdm(self.val_loader, desc='Validation')
            for images, targets in progress:
                images = images.to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in targets.items()}

                predictions = self.model(images)
                losses = self.criterion(predictions, targets)

                total_loss += losses['total_loss'].item()
                for key in components:
                    components[key] += losses[key].item()

        n = max(len(self.val_loader), 1)
        return total_loss / n, {k: v / n for k, v in components.items()}

    def train(self):
        print(f"=== Vertebra Corner Detection V3 ===")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Heatmap channels: {self.model.num_channels}")
        print(f"Heatmap size: {HEATMAP_SIZE}x{HEATMAP_SIZE}")
        print("-" * 50)

        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")

            train_loss, train_comp = self.train_epoch()
            val_loss, val_comp = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Heatmap: T={train_comp['heatmap_loss']:.4f} V={val_comp['heatmap_loss']:.4f}")
            print(f"  Count:   T={train_comp['count_loss']:.4f} V={val_comp['count_loss']:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config,
                    'model_version': 'v3.2',
                    'backbone_source': getattr(self.model, 'backbone_source', 'unknown'),
                    'heatmap_size': HEATMAP_SIZE,
                }, 'best_vertebra_model.pth')
                print("  >> Saved best model")

            self.scheduler.step()

            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'model_version': 'v3.2',
                    'backbone_source': getattr(self.model, 'backbone_source', 'unknown'),
                    'heatmap_size': HEATMAP_SIZE,
                }, f'checkpoint_vertebra_epoch_{epoch+1}.pth')

        # 儲存訓練曲線
        self._save_loss_plot()

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")

    def _save_loss_plot(self):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_losses, label='Train')
            plt.plot(self.val_losses, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('training_loss_curve.png', dpi=100)
            plt.close()
            print("Saved training_loss_curve.png")
        except Exception:
            pass


def get_transforms(is_training=True):
    """V3.1 加強版 augmentation (針對小數據集 + X-ray 影像)"""
    if is_training:
        return A.Compose([
            A.Resize(512, 512),
            # 幾何變換
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.6
            ),
            A.Affine(shear=(-8, 8), p=0.3),  # 剪切變形
            # 光照變換 (X-ray 很重要)
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.RandomGamma(gamma_limit=(60, 140), p=0.4),
            # X-ray 特有: 反轉 (模擬不同 window/level)
            A.InvertImg(p=0.15),
            # 模擬雜訊/模糊
            A.GaussNoise(var_limit=(10, 80), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            # 遮擋模擬 (增加魯棒性)
            A.CoarseDropout(max_holes=3, max_height=40, max_width=40, p=0.2),
            # 正規化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def collate_fn(batch):
    images = []
    targets = {
        'keypoints': [],
        'valid_mask': [],
        'heatmaps': [],
        'num_vertebrae': [],
        'vertebra_names': []
    }

    for image, target in batch:
        images.append(image)
        targets['keypoints'].append(target['keypoints'])
        targets['valid_mask'].append(target['valid_mask'])
        targets['heatmaps'].append(target['heatmaps'])
        targets['num_vertebrae'].append(target['num_vertebrae'])
        targets['vertebra_names'].append(target['vertebra_names'])

    return (
        torch.stack(images, 0),
        {
            'keypoints': torch.stack(targets['keypoints'], 0),
            'valid_mask': torch.stack(targets['valid_mask'], 0),
            'heatmaps': torch.stack(targets['heatmaps'], 0),
            'num_vertebrae': targets['num_vertebrae'],
            'vertebra_names': targets['vertebra_names']
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Spine Vertebra Corner Detection - Training V3.2')
    parser.add_argument('--epochs', type=int, default=200,
                        help='訓練 epoch 數 (煙霧測試建議 5)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--repeat-dataset', type=int, default=8,
                        help='每 epoch 重複 dataset 的倍數 (增加 augmentation 多樣性)')
    parser.add_argument('--pretrained-path', type=str, default=RADIMAGENET_PATH_DEFAULT,
                        help='RadImageNet ResNet50 權重檔路徑；不存在會 fallback 到 ImageNet')
    parser.add_argument('--debug', action='store_true',
                        help='Debug 模式：縮短 epoch、印詳細資訊')
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        'data_dir': '.',
        'train_annotations': 'endplate_training_data/annotations/train_annotations.json',
        'val_annotations': 'endplate_training_data/annotations/val_annotations.json',
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_workers': 0,
        'max_vertebrae': 8,
        'repeat_dataset': args.repeat_dataset,
        'pretrained_path': args.pretrained_path,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not os.path.exists(config['train_annotations']):
        print(f"ERROR: {config['train_annotations']} not found")
        print("Run prepare_endplate_data.py first")
        return

    train_dataset = VertebraDataset(
        config['data_dir'],
        config['train_annotations'],
        transform=get_transforms(True),
        max_vertebrae=config['max_vertebrae']
    )

    val_dataset = VertebraDataset(
        config['data_dir'],
        config['val_annotations'],
        transform=get_transforms(False),
        max_vertebrae=config['max_vertebrae']
    )

    # V3.1: 用 RepeatDataset 增加有效訓練量 (同張圖不同 augmentation)
    repeat = config.get('repeat_dataset', 1)
    if repeat > 1:
        print(f"Dataset repeat: {repeat}x (effective train size: {len(train_dataset) * repeat})")
        train_dataset_wrapped = RepeatDataset(train_dataset, repeat=repeat)
    else:
        train_dataset_wrapped = train_dataset

    train_loader = DataLoader(
        train_dataset_wrapped,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = VertebraCornerModel(
        max_vertebrae=config['max_vertebrae'],
        pretrained=True,
        pretrained_path=config['pretrained_path'],
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Backbone source: {model.backbone_source}")

    trainer = VertebraTrainer(model, train_loader, val_loader, device, config)
    trainer.train()


if __name__ == "__main__":
    main()
