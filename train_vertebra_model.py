#!/usr/bin/env python3
"""
脊椎椎體頂點檢測 - 機器學習訓練腳本 V2.1
Spine Vertebra Corner Detection - ML Training Script V2.1

支援:
- 完整椎體: 4 個角點
- 邊界椎體 (S1/T1=上終板 2點, T12/C2=下終板 2點)

從角點自動計算：壓迫性骨折、椎間盤高度、滑脫

日期: 2025-2026
"""

import os
import json
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


class VertebraDataset(Dataset):
    """椎體頂點檢測數據集 V2.1

    支援:
    - 完整椎體: 4 角點 (anteriorSuperior, posteriorSuperior, posteriorInferior, anteriorInferior)
    - 上邊界椎體 (S1/T1): 2 點 (anteriorSuperior, posteriorSuperior)
    - 下邊界椎體 (T12/C2): 2 點 (posteriorInferior, anteriorInferior)

    為統一模型輸入，每個椎體固定 4 個 slot:
    - 完整椎體: 4 個有效點
    - 邊界椎體: 2 個有效點 + 2 個零填充 (valid_mask=0)
    """

    # 邊界椎體定義
    BOUNDARY_CONFIG = {
        'L': {'upper': ['S1'], 'lower': ['T12']},
        'C': {'upper': ['T1'], 'lower': ['C2']},
    }

    def __init__(self, data_dir, annotations_file, transform=None, max_vertebrae=8):
        self.data_dir = data_dir
        self.transform = transform
        self.max_vertebrae = max_vertebrae  # T12-S1 = 8, C2-T1 = 8

        # 載入標註數據
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        if isinstance(self.annotations, dict):
            self.annotations = [self.annotations]

        print(f"載入 {len(self.annotations)} 個樣本")

    def __len__(self):
        return len(self.annotations)

    def _get_boundary_type(self, name, spine_type, vertebra_data):
        """判斷是否為邊界椎體

        V2.1: 使用 boundaryType 欄位
        V2.0: S1 等邊界椎體可能有完整 4 點 → 視為完整椎體
        """
        # V2.1 明確標記
        bt = vertebra_data.get('boundaryType', None)
        if bt:
            return bt

        # V2.0 相容：如果有完整 4 點就當完整椎體
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

        # 載入圖像
        image = self.load_image(annotation)

        if image is None:
            raise FileNotFoundError(f"Cannot load image for annotation {idx}")

        original_h, original_w = image.shape[:2]
        spine_type = annotation.get('spine_type', annotation.get('spineType', 'L'))

        # 提取椎體頂點
        vertebrae = annotation.get('vertebrae', [])
        keypoints = []  # 所有角點座標 (每椎體固定 4 slot)
        valid_flags = []  # 每個 slot 是否有效
        vertebra_names = []

        for v in vertebrae[:self.max_vertebrae]:
            points = v.get('points', {})
            name = v.get('name', '')
            vertebra_names.append(name)
            boundary = self._get_boundary_type(name, spine_type, v)

            # 每個椎體固定 4 slots: [anteriorSuperior, posteriorSuperior, posteriorInferior, anteriorInferior]
            if isinstance(points, dict):
                if boundary == 'upper':
                    # 上邊界 (S1/T1): 只有 anteriorSuperior + posteriorSuperior
                    corners = [
                        points.get('anteriorSuperior', {}),
                        points.get('posteriorSuperior', {}),
                        None,  # posteriorInferior 不存在
                        None,  # anteriorInferior 不存在
                    ]
                elif boundary == 'lower':
                    # 下邊界 (T12/C2): 只有 posteriorInferior + anteriorInferior
                    corners = [
                        None,  # anteriorSuperior 不存在
                        None,  # posteriorSuperior 不存在
                        points.get('posteriorInferior', {}),
                        points.get('anteriorInferior', {}),
                    ]
                else:
                    # 完整椎體
                    corners = [
                        points.get('anteriorSuperior', {}),
                        points.get('posteriorSuperior', {}),
                        points.get('posteriorInferior', {}),
                        points.get('anteriorInferior', {}),
                    ]
            else:
                # list 格式
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

        # 應用變換 - 只對有效 keypoints 做變換
        valid_kp_indices = [i for i, f in enumerate(valid_flags) if f > 0]
        valid_kp = [keypoints[i] for i in valid_kp_indices]

        if self.transform and len(valid_kp) > 0:
            transformed = self.transform(image=image, keypoints=valid_kp)
            image = transformed['image']
            transformed_valid_kp = transformed['keypoints']

            # 重建完整 keypoints 列表
            transformed_kp = [[0.0, 0.0]] * len(keypoints)
            for j, idx_orig in enumerate(valid_kp_indices):
                transformed_kp[idx_orig] = list(transformed_valid_kp[j])
        else:
            # 基本變換
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

        # 正規化關鍵點到 [0, 1]
        h, w = 512, 512
        normalized_kp = []
        for i, kp in enumerate(transformed_kp):
            if valid_flags[i] > 0:
                normalized_kp.append([kp[0] / w, kp[1] / h])
            else:
                normalized_kp.append([0.0, 0.0])

        # 填充到固定長度 (max_vertebrae * 4 個點)
        max_points = self.max_vertebrae * 4
        while len(normalized_kp) < max_points:
            normalized_kp.append([0.0, 0.0])
        while len(valid_flags) < max_points:
            valid_flags.append(0.0)

        # 創建熱圖目標 (只用有效的 keypoints)
        valid_transformed_kp = [transformed_kp[i] for i in valid_kp_indices]
        heatmap = self.create_heatmap(valid_transformed_kp, (h, w))

        targets = {
            'keypoints': torch.tensor(normalized_kp[:max_points], dtype=torch.float32),  # [N*4, 2]
            'valid_mask': torch.tensor(valid_flags[:max_points], dtype=torch.float32),  # [N*4]
            'heatmap': torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0),  # [1, H, W]
            'num_vertebrae': len(vertebrae),
            'vertebra_names': vertebra_names
        }

        return image, targets

    def load_image(self, annotation):
        """載入圖像"""
        image_path = annotation.get('image_path', '')
        full_path = os.path.join(self.data_dir, image_path)

        if os.path.exists(full_path):
            if full_path.lower().endswith('.dcm'):
                try:
                    dcm = pydicom.dcmread(full_path)
                    image = dcm.pixel_array
                    if len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=-1)
                    image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
                    return image
                except Exception as e:
                    print(f"Error loading DICOM {full_path}: {e}")
            else:
                image = cv2.imread(full_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 搜尋同名檔案
        source_file = annotation.get('source_file', '')
        if source_file:
            base_name = os.path.splitext(source_file)[0]
            for ext in ['.dcm', '.png', '.jpg']:
                candidate = os.path.join(self.data_dir, base_name + ext)
                if os.path.exists(candidate):
                    return self.load_image({'image_path': base_name + ext})

        return None

    def create_heatmap(self, keypoints, size, sigma=5):
        """創建高斯熱圖"""
        h, w = size
        heatmap = np.zeros((h, w), dtype=np.float32)

        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            if 0 <= x < w and 0 <= y < h:
                # 創建高斯點
                for dy in range(-sigma*2, sigma*2+1):
                    for dx in range(-sigma*2, sigma*2+1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            dist = (dx*dx + dy*dy) / (2 * sigma * sigma)
                            heatmap[ny, nx] = max(heatmap[ny, nx], np.exp(-dist))

        return heatmap


class VertebraCornerModel(nn.Module):
    """椎體頂點檢測模型

    採用雙分支架構:
    1. 熱圖分支: 預測角點位置的熱圖 (用於粗定位)
    2. 回歸分支: 直接回歸角點座標 (用於精確定位)
    """

    def __init__(self, max_vertebrae=8, pretrained=True):
        super(VertebraCornerModel, self).__init__()

        self.max_vertebrae = max_vertebrae
        self.num_points = max_vertebrae * 4  # 每個椎體4個 slot (邊界椎體2有效+2填充)

        # Backbone - ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 移除FC和AvgPool

        # 熱圖分支 (Heatmap Branch) - 用於粗定位
        self.heatmap_branch = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 1),  # 輸出熱圖
            nn.Sigmoid()
        )

        # 回歸分支 (Regression Branch) - 直接預測座標
        self.regression_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_points * 2),  # 輸出 [N*4, 2] 座標
            nn.Sigmoid()  # 正規化到 [0, 1]
        )

        # 椎體數量預測 (輔助任務)
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, max_vertebrae + 1)  # 0 到 max_vertebrae
        )

    def forward(self, x):
        # Backbone features
        features = self.backbone(x)  # [B, 2048, H/32, W/32]

        # 熱圖預測
        heatmap = self.heatmap_branch(features)  # [B, 1, H/2, W/2]

        # 座標回歸
        coords = self.regression_branch(features)  # [B, N*4*2]
        coords = coords.view(-1, self.num_points, 2)  # [B, N*4, 2]

        # 椎體數量預測
        count_logits = self.count_head(features)  # [B, max_vertebrae+1]

        return {
            'heatmap': heatmap,
            'coords': coords,
            'count_logits': count_logits
        }


class VertebraLoss(nn.Module):
    """椎體頂點檢測損失函數"""

    def __init__(self, alpha=1.0, beta=2.0, gamma=0.5):
        super(VertebraLoss, self).__init__()
        self.alpha = alpha  # 熱圖損失權重
        self.beta = beta    # 座標回歸損失權重
        self.gamma = gamma  # 計數損失權重

        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        batch_size = predictions['coords'].shape[0]

        # 1. 熱圖損失
        pred_heatmap = predictions['heatmap']
        target_heatmap = targets['heatmap']

        # Resize 目標熱圖到預測尺寸
        if pred_heatmap.shape[2:] != target_heatmap.shape[2:]:
            target_heatmap = torch.nn.functional.interpolate(
                target_heatmap, size=pred_heatmap.shape[2:],
                mode='bilinear', align_corners=True
            )

        heatmap_loss = self.bce(pred_heatmap, target_heatmap)

        # 2. 座標回歸損失 (只計算有效點)
        pred_coords = predictions['coords']  # [B, N, 2]
        target_coords = targets['keypoints']  # [B, N, 2]
        valid_mask = targets['valid_mask']  # [B, N]

        # 計算MSE
        coord_diff = self.mse(pred_coords, target_coords)  # [B, N, 2]
        coord_diff = coord_diff.sum(dim=-1)  # [B, N]

        # 只計算有效點的損失
        coord_loss = (coord_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # 3. 計數損失
        count_logits = predictions['count_logits']
        target_count = torch.tensor([t for t in targets['num_vertebrae']],
                                   dtype=torch.long, device=count_logits.device)
        count_loss = self.ce(count_logits, target_count)

        # 總損失
        total_loss = (self.alpha * heatmap_loss +
                     self.beta * coord_loss +
                     self.gamma * count_loss)

        return {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'coord_loss': coord_loss,
            'count_loss': count_loss
        }


class VertebraTrainer:
    """椎體頂點檢測訓練器"""

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        self.criterion = VertebraLoss()

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0
        components = {'heatmap_loss': 0, 'coord_loss': 0, 'count_loss': 0}

        progress = tqdm(self.train_loader, desc='Training')
        for images, targets in progress:
            images = images.to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in targets.items()}

            self.optimizer.zero_grad()
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)

            losses['total_loss'].backward()
            self.optimizer.step()

            total_loss += losses['total_loss'].item()
            for key in components:
                components[key] += losses[key].item()

            progress.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Coord': f"{losses['coord_loss'].item():.4f}"
            })

        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in components.items()}

    def validate(self):
        """驗證"""
        self.model.eval()
        total_loss = 0
        components = {'heatmap_loss': 0, 'coord_loss': 0, 'count_loss': 0}

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

        n = len(self.val_loader)
        return total_loss / n, {k: v / n for k, v in components.items()}

    def train(self):
        """完整訓練"""
        print(f"開始訓練椎體頂點檢測模型 V2")
        print(f"設備: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print("-" * 50)

        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")

            train_loss, train_comp = self.train_epoch()
            val_loss, val_comp = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Heatmap: {train_comp['heatmap_loss']:.4f}")
            print(f"  Coord: {train_comp['coord_loss']:.4f}")
            print(f"  Count: {train_comp['count_loss']:.4f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, 'best_vertebra_model.pth')
                print("✅ 保存最佳模型")

            self.scheduler.step()

            # 每20個epoch保存檢查點
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss
                }, f'checkpoint_vertebra_epoch_{epoch+1}.pth')

        print("\n🎉 訓練完成!")
        print(f"最佳驗證損失: {self.best_val_loss:.4f}")


def get_transforms(is_training=True):
    """數據變換"""
    if is_training:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
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
    """自定義 collate 函數"""
    images = []
    targets = {
        'keypoints': [],
        'valid_mask': [],
        'heatmap': [],
        'num_vertebrae': [],
        'vertebra_names': []
    }

    for image, target in batch:
        images.append(image)
        targets['keypoints'].append(target['keypoints'])
        targets['valid_mask'].append(target['valid_mask'])
        targets['heatmap'].append(target['heatmap'])
        targets['num_vertebrae'].append(target['num_vertebrae'])
        targets['vertebra_names'].append(target['vertebra_names'])

    return (
        torch.stack(images, 0),
        {
            'keypoints': torch.stack(targets['keypoints'], 0),
            'valid_mask': torch.stack(targets['valid_mask'], 0),
            'heatmap': torch.stack(targets['heatmap'], 0),
            'num_vertebrae': targets['num_vertebrae'],
            'vertebra_names': targets['vertebra_names']
        }
    )


def main():
    """主函數"""
    config = {
        'data_dir': '.',
        'train_annotations': 'endplate_training_data/annotations/train_annotations.json',
        'val_annotations': 'endplate_training_data/annotations/val_annotations.json',
        'batch_size': 4,
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 0,
        'max_vertebrae': 8
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 檢查標註檔案
    if not os.path.exists(config['train_annotations']):
        print(f"❌ 找不到訓練標註: {config['train_annotations']}")
        print("💡 請先執行 prepare_endplate_data.py 準備數據")
        print("💡 或使用 spinal-annotation-web.html 進行標註")
        return

    # 數據集
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

    train_loader = DataLoader(
        train_dataset,
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

    # 模型
    model = VertebraCornerModel(
        max_vertebrae=config['max_vertebrae'],
        pretrained=True
    )
    print(f"模型參數: {sum(p.numel() for p in model.parameters()):,}")

    # 訓練
    trainer = VertebraTrainer(model, train_loader, val_loader, device, config)
    trainer.train()


if __name__ == "__main__":
    main()
