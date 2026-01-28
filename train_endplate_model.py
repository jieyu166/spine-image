#!/usr/bin/env python3
"""
脊椎終板檢測 - 機器學習訓練腳本
Spine Endplate Detection - ML Training Script

專注於檢測終板前後緣，不計算角度
只輸出endplate位置和vertebra edges

作者: AI Assistant
日期: 2025
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

class EndplateDataset(Dataset):
    """終板檢測數據集"""
    
    def __init__(self, data_dir, annotations_file, transform=None, is_training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        
        # 載入標註數據
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 確保是list格式
        if isinstance(self.annotations, dict):
            self.annotations = [self.annotations]
        
        print(f"載入 {len(self.annotations)} 個樣本")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 載入圖像
        image_path = os.path.join(self.data_dir, annotation.get('image_path', ''))
        
        # 嘗試載入圖像
        image = None
        
        # 方法1: 直接讀取
        if os.path.exists(image_path):
            if image_path.lower().endswith('.dcm'):
                # DICOM檔案
                try:
                    dcm = pydicom.dcmread(image_path)
                    image = dcm.pixel_array
                    # 轉換為RGB格式
                    if len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=-1)
                    # 正規化到0-255
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                except Exception as e:
                    print(f"Error loading DICOM {image_path}: {e}")
            else:
                # 一般圖像檔案
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 方法2: 如果路徑不存在或載入失敗，搜尋同名DICOM
        if image is None:
            # 從標註檔案路徑推測
            base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else ''
            if base_name:
                # 搜尋所有可能的DICOM檔案
                for root, dirs, files in os.walk(self.data_dir):
                    for file in files:
                        if file.startswith(base_name) and file.lower().endswith('.dcm'):
                            dcm_path = os.path.join(root, file)
                            try:
                                dcm = pydicom.dcmread(dcm_path)
                                image = dcm.pixel_array
                                if len(image.shape) == 2:
                                    image = np.stack([image] * 3, axis=-1)
                                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                                break
                            except:
                                continue
                    if image is not None:
                        break
        
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path} (searched in {self.data_dir})")
        
        # 處理標註 - 新格式
        measurements = annotation.get('measurements', [])
        vertebra_edges = annotation.get('vertebra_edges', {})
        
        # 應用變換（包含圖像和遮罩）
        if self.transform:
            # 創建原始尺寸的遮罩
            original_h, original_w = image.shape[:2]
            endplate_mask, vertebra_edge_mask = self.create_masks(measurements, vertebra_edges, (original_h, original_w))
            
            # 準備關鍵點
            keypoints = []
            for m in measurements:
                for p in m.get('lowerEndplate', []):
                    keypoints.append([p['x'], p['y']])
                for p in m.get('upperEndplate', []):
                    keypoints.append([p['x'], p['y']])
            
            # 使用不含ToTensorV2的transform來處理遮罩
            mask_transform = A.Compose([
                A.Resize(512, 512)
            ])
            
            # 變換遮罩
            mask_transformed = mask_transform(
                image=image,  # 需要image參數
                masks=[endplate_mask, vertebra_edge_mask[:,:,0], vertebra_edge_mask[:,:,1]]
            )
            
            # 變換圖像和關鍵點
            transformed = self.transform(image=image, keypoints=keypoints)
            
            image = transformed['image']
            masks = mask_transformed['masks']
            
            # 組合遮罩（保持numpy格式）
            endplate_mask_resized = masks[0]
            vertebra_edge_mask_resized = np.stack([masks[1], masks[2]], axis=-1)
            
            # 創建目標
            targets = self.create_targets_from_masks(
                endplate_mask_resized,
                vertebra_edge_mask_resized,
                transformed.get('keypoints', []),
                measurements,
                image.shape[1:3]
            )
        else:
            # 無變換時直接創建
            targets = self.create_targets(measurements, vertebra_edges, image.shape[:2])
        
        return image, targets
    
    def create_masks(self, measurements, vertebra_edges, image_shape):
        """創建遮罩（用於transform）"""
        h, w = image_shape
        
        # 終板遮罩
        endplate_mask = np.zeros((h, w), dtype=np.float32)
        for m in measurements:
            if 'lowerEndplate' in m and len(m['lowerEndplate']) >= 2:
                p1, p2 = m['lowerEndplate'][0], m['lowerEndplate'][1]
                cv2.line(endplate_mask, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), 1.0, thickness=5)
            if 'upperEndplate' in m and len(m['upperEndplate']) >= 2:
                p1, p2 = m['upperEndplate'][0], m['upperEndplate'][1]
                cv2.line(endplate_mask, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), 1.0, thickness=5)
        
        # 椎體邊緣遮罩
        anterior_mask = np.zeros((h, w), dtype=np.float32)
        posterior_mask = np.zeros((h, w), dtype=np.float32)
        for vertebra_name, edges in vertebra_edges.items():
            if 'anterior' in edges and len(edges['anterior']) >= 2:
                p1, p2 = edges['anterior'][0], edges['anterior'][1]
                cv2.line(anterior_mask, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), 1.0, thickness=5)
            if 'posterior' in edges and len(edges['posterior']) >= 2:
                p1, p2 = edges['posterior'][0], edges['posterior'][1]
                cv2.line(posterior_mask, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), 1.0, thickness=5)
        
        vertebra_edge_mask = np.stack([anterior_mask, posterior_mask], axis=-1)
        return endplate_mask, vertebra_edge_mask
    
    def create_targets_from_masks(self, endplate_mask, vertebra_edge_mask, keypoints, measurements, image_shape):
        """從變換後的遮罩創建目標"""
        h, w = image_shape
        
        # 處理關鍵點
        keypoint_list = []
        for kp in keypoints:
            if len(kp) >= 2:
                keypoint_list.append([kp[0]/w, kp[1]/h, 1.0])
        
        # 邊界框（從measurements計算）
        bboxes = []
        for m in measurements:
            all_points = []
            for p in m.get('lowerEndplate', []) + m.get('upperEndplate', []):
                all_points.append([p['x'], p['y']])
            if all_points:
                all_points = np.array(all_points)
                x_min, y_min = all_points.min(axis=0)
                x_max, y_max = all_points.max(axis=0)
                x_center = (x_min + x_max) / 2 / w
                y_center = (y_min + y_max) / 2 / h
                box_w = (x_max - x_min) / w
                box_h = (y_max - y_min) / h
                bboxes.append([x_center, y_center, box_w, box_h])
        
        # 轉換為張量（遮罩總是numpy格式）
        endplate_tensor = torch.from_numpy(endplate_mask).unsqueeze(0).contiguous()
        vertebra_edge_tensor = torch.from_numpy(vertebra_edge_mask).permute(2, 0, 1).contiguous()
        
        return {
            'endplate_mask': endplate_tensor,
            'vertebra_edge_mask': vertebra_edge_tensor,
            'keypoints': torch.tensor(keypoint_list, dtype=torch.float32) if keypoint_list else torch.zeros((0, 3)),
            'bboxes': torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
            'num_measurements': len(measurements)
        }
    
    def create_targets(self, measurements, vertebra_edges, image_shape):
        """創建訓練目標 - 專注於終板檢測"""
        h, w = image_shape
        
        # 1. 終板分割遮罩 (線段形式)
        endplate_mask = np.zeros((h, w), dtype=np.float32)
        
        for m in measurements:
            # 繪製lowerEndplate線段
            if 'lowerEndplate' in m and len(m['lowerEndplate']) >= 2:
                p1 = m['lowerEndplate'][0]
                p2 = m['lowerEndplate'][1]
                cv2.line(endplate_mask, 
                        (int(p1['x']), int(p1['y'])),
                        (int(p2['x']), int(p2['y'])),
                        1.0, thickness=5)
            
            # 繪製upperEndplate線段
            if 'upperEndplate' in m and len(m['upperEndplate']) >= 2:
                p1 = m['upperEndplate'][0]
                p2 = m['upperEndplate'][1]
                cv2.line(endplate_mask,
                        (int(p1['x']), int(p1['y'])),
                        (int(p2['x']), int(p2['y'])),
                        1.0, thickness=5)
        
        # 2. 關鍵點熱圖 (每個終板端點)
        num_keypoints = 0
        keypoint_list = []
        
        for m in measurements:
            # lowerEndplate 2個點
            for p in m.get('lowerEndplate', []):
                keypoint_list.append([p['x']/w, p['y']/h, 1.0])
                num_keypoints += 1
            # upperEndplate 2個點
            for p in m.get('upperEndplate', []):
                keypoint_list.append([p['x']/w, p['y']/h, 1.0])
                num_keypoints += 1
        
        # 3. 椎體邊緣檢測目標
        vertebra_edge_mask = np.zeros((h, w, 2), dtype=np.float32)  # 2通道: anterior, posterior
        
        # 為每個通道創建獨立的遮罩（解決記憶體佈局問題）
        anterior_mask = np.zeros((h, w), dtype=np.float32)
        posterior_mask = np.zeros((h, w), dtype=np.float32)
        
        for vertebra_name, edges in vertebra_edges.items():
            # Anterior edge
            if 'anterior' in edges and len(edges['anterior']) >= 2:
                p1 = edges['anterior'][0]
                p2 = edges['anterior'][1]
                cv2.line(anterior_mask,
                        (int(p1['x']), int(p1['y'])),
                        (int(p2['x']), int(p2['y'])),
                        1.0, thickness=5)
            
            # Posterior edge  
            if 'posterior' in edges and len(edges['posterior']) >= 2:
                p1 = edges['posterior'][0]
                p2 = edges['posterior'][1]
                cv2.line(posterior_mask,
                        (int(p1['x']), int(p1['y'])),
                        (int(p2['x']), int(p2['y'])),
                        1.0, thickness=5)
        
        # 組合到多通道遮罩
        vertebra_edge_mask[:,:,0] = anterior_mask
        vertebra_edge_mask[:,:,1] = posterior_mask
        
        # 4. 椎間隙邊界框 (用於定位)
        bboxes = []
        for m in measurements:
            # 收集所有點
            all_points = []
            for p in m.get('lowerEndplate', []):
                all_points.append([p['x'], p['y']])
            for p in m.get('upperEndplate', []):
                all_points.append([p['x'], p['y']])
            
            if all_points:
                all_points = np.array(all_points)
                x_min, y_min = all_points.min(axis=0)
                x_max, y_max = all_points.max(axis=0)
                
                # 正規化
                x_center = (x_min + x_max) / 2 / w
                y_center = (y_min + y_max) / 2 / h
                box_w = (x_max - x_min) / w
                box_h = (y_max - y_min) / h
                
                bboxes.append([x_center, y_center, box_w, box_h])
        
        # 確保張量是連續的
        endplate_tensor = torch.from_numpy(endplate_mask).unsqueeze(0).contiguous()  # [1, H, W]
        vertebra_edge_tensor = torch.from_numpy(vertebra_edge_mask).permute(2, 0, 1).contiguous()  # [2, H, W]
        
        return {
            'endplate_mask': endplate_tensor,
            'vertebra_edge_mask': vertebra_edge_tensor,
            'keypoints': torch.tensor(keypoint_list, dtype=torch.float32),  # [N, 3]
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),  # [M, 4]
            'num_measurements': len(measurements)
        }
    
    def _update_keypoints(self, transformed_keypoints, image_size):
        """更新變換後的關鍵點"""
        h, w = image_size
        kp_list = []
        for kp in transformed_keypoints:
            kp_list.append([kp[0]/w, kp[1]/h, 1.0])
        return torch.tensor(kp_list, dtype=torch.float32)

class EndplateDetectionModel(nn.Module):
    """終板檢測模型 - U-Net架構"""
    
    def __init__(self, pretrained=True):
        super(EndplateDetectionModel, self).__init__()
        
        # Encoder - ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.encoder1 = nn.Sequential(*list(resnet.children())[:3])  # 64
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5])  # 256
        self.encoder3 = resnet.layer2  # 512
        self.encoder4 = resnet.layer3  # 1024
        self.encoder5 = resnet.layer4  # 2048
        
        # Decoder for endplate segmentation
        self.up1 = self._make_upconv(2048, 1024)
        self.dec1 = self._make_conv_block(2048, 1024)
        
        self.up2 = self._make_upconv(1024, 512)
        self.dec2 = self._make_conv_block(1024, 512)
        
        self.up3 = self._make_upconv(512, 256)
        self.dec3 = self._make_conv_block(512, 256)
        
        self.up4 = self._make_upconv(256, 64)
        self.dec4 = self._make_conv_block(128, 64)
        
        # Output heads
        # 1. Endplate segmentation (終板線段)
        self.endplate_out = nn.Conv2d(64, 1, 1)
        
        # 2. Vertebra edge segmentation (前後緣)
        self.vertebra_edge_out = nn.Conv2d(64, 2, 1)  # 2通道: anterior, posterior
        
        # 3. Keypoint heatmap (關鍵點)
        self.keypoint_out = nn.Conv2d(64, 1, 1)
        
    def _make_upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
    
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # [B, 64, H/2, W/2]
        e2 = self.encoder2(e1)  # [B, 256, H/4, W/4]
        e3 = self.encoder3(e2)  # [B, 512, H/8, W/8]
        e4 = self.encoder4(e3)  # [B, 1024, H/16, W/16]
        e5 = self.encoder5(e4)  # [B, 2048, H/32, W/32]
        
        # Decoder
        d1 = self.up1(e5)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)
        
        # Outputs
        endplate_seg = torch.sigmoid(self.endplate_out(d4))
        vertebra_edge_seg = torch.sigmoid(self.vertebra_edge_out(d4))
        keypoint_heatmap = torch.sigmoid(self.keypoint_out(d4))
        
        return {
            'endplate_seg': endplate_seg,
            'vertebra_edge_seg': vertebra_edge_seg,
            'keypoint_heatmap': keypoint_heatmap
        }

class EndplateLoss(nn.Module):
    """終板檢測損失函數"""
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5):
        super(EndplateLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # 獲取預測的尺寸
        pred_h, pred_w = predictions['endplate_seg'].shape[2:4]
        
        # Resize目標到預測尺寸
        import torch.nn.functional as F
        endplate_mask_resized = F.interpolate(
            targets['endplate_mask'], 
            size=(pred_h, pred_w), 
            mode='bilinear', 
            align_corners=False
        )
        vertebra_edge_mask_resized = F.interpolate(
            targets['vertebra_edge_mask'], 
            size=(pred_h, pred_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 1. Endplate segmentation loss
        endplate_loss = self.bce_loss(
            predictions['endplate_seg'],
            endplate_mask_resized
        )
        
        # 2. Vertebra edge segmentation loss
        edge_loss = self.bce_loss(
            predictions['vertebra_edge_seg'],
            vertebra_edge_mask_resized
        )
        
        # 3. Keypoint heatmap loss
        # 創建高斯熱圖
        keypoint_heatmap = self._create_keypoint_heatmap(
            targets['keypoints'],
            predictions['keypoint_heatmap'].shape[2:4]
        )
        keypoint_loss = self.mse_loss(
            predictions['keypoint_heatmap'],
            keypoint_heatmap.unsqueeze(1)
        )
        
        # 總損失
        total_loss = (self.alpha * endplate_loss +
                     self.beta * edge_loss +
                     self.gamma * keypoint_loss)
        
        return {
            'total_loss': total_loss,
            'endplate_loss': endplate_loss,
            'edge_loss': edge_loss,
            'keypoint_loss': keypoint_loss
        }
    
    def _create_keypoint_heatmap(self, keypoints, output_size):
        """創建關鍵點高斯熱圖（處理list格式）"""
        h, w = output_size
        
        # keypoints 現在是 list of tensors
        if isinstance(keypoints, list):
            batch_size = len(keypoints)
            device = keypoints[0].device if len(keypoints) > 0 and keypoints[0].numel() > 0 else 'cpu'
        else:
            batch_size = keypoints.shape[0]
            device = keypoints.device
        
        heatmap = torch.zeros(batch_size, h, w, device=device)
        
        for b in range(batch_size):
            kp_batch = keypoints[b] if isinstance(keypoints, list) else keypoints[b]
            
            if kp_batch.numel() == 0:  # 沒有關鍵點
                continue
                
            for kp in kp_batch:
                if len(kp) >= 3 and kp[2] > 0:  # visibility
                    x, y = int(kp[0] * w), int(kp[1] * h)
                    if 0 <= x < w and 0 <= y < h:
                        # 簡單的點標記
                        heatmap[b, y, x] = 1.0
        
        return heatmap

class EndplateTrainer:
    """終板檢測訓練器"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 優化器和調度器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        self.criterion = EndplateLoss()
        
        # 訓練記錄
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {
            'endplate_loss': 0,
            'edge_loss': 0,
            'keypoint_loss': 0
        }
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # 記錄損失
            total_loss += losses['total_loss'].item()
            for key in loss_components:
                loss_components[key] += losses[key].item()
            
            # 更新進度條
            progress_bar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 計算平均損失
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return avg_loss, loss_components
    
    def validate(self):
        """驗證模型"""
        self.model.eval()
        total_loss = 0
        loss_components = {
            'endplate_loss': 0,
            'edge_loss': 0,
            'keypoint_loss': 0
        }
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            for images, targets in progress_bar:
                images = images.to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in targets.items()}
                
                predictions = self.model(images)
                losses = self.criterion(predictions, targets)
                
                total_loss += losses['total_loss'].item()
                for key in loss_components:
                    loss_components[key] += losses[key].item()
                
                progress_bar.set_postfix({'Val Loss': f"{losses['total_loss'].item():.4f}"})
        
        # 計算平均損失
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        return avg_loss, loss_components
    
    def train(self):
        """完整訓練流程"""
        print(f"開始訓練終板檢測模型，共 {self.config['epochs']} 個epochs")
        print(f"設備: {self.device}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"學習率: {self.config['learning_rate']}")
        print("-" * 50)
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 訓練
            train_loss, train_components = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 驗證
            val_loss, val_components = self.validate()
            self.val_losses.append(val_loss)
            
            # 打印結果
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("Loss Components:")
            for key, value in train_components.items():
                print(f"  {key}: {value:.4f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, 'best_endplate_model.pth')
                print("✅ 保存最佳模型")
            
            # 學習率調度
            self.scheduler.step()
            
            # 每10個epoch保存檢查點
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, f'checkpoint_epoch_{epoch+1}.pth')
        
        print("\n🎉 訓練完成!")
        print(f"最佳驗證損失: {self.best_val_loss:.4f}")

def get_transforms(is_training=True):
    """獲取數據變換"""
    if is_training:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def custom_collate_fn(batch):
    """自定義collate函數，處理可變長度的關鍵點和邊界框"""
    images = []
    targets = {
        'endplate_mask': [],
        'vertebra_edge_mask': [],
        'keypoints': [],
        'bboxes': [],
        'num_measurements': []
    }
    
    for image, target in batch:
        images.append(image)
        targets['endplate_mask'].append(target['endplate_mask'])
        targets['vertebra_edge_mask'].append(target['vertebra_edge_mask'])
        targets['keypoints'].append(target['keypoints'])  # 保持為list，不stack
        targets['bboxes'].append(target['bboxes'])  # 保持為list，不stack
        targets['num_measurements'].append(target['num_measurements'])
    
    # Stack固定尺寸的張量
    batched = {
        'endplate_mask': torch.stack(targets['endplate_mask'], 0),
        'vertebra_edge_mask': torch.stack(targets['vertebra_edge_mask'], 0),
        'keypoints': targets['keypoints'],  # List of tensors
        'bboxes': targets['bboxes'],  # List of tensors
        'num_measurements': torch.tensor(targets['num_measurements'])
    }
    
    return torch.stack(images, 0), batched

def main():
    """主函數"""
    # 配置參數
    config = {
        'data_dir': '.',  # 當前目錄（執行時應在 Spine 資料夾中）
        'train_annotations': 'endplate_training_data/annotations/train_annotations.json',
        'val_annotations': 'endplate_training_data/annotations/val_annotations.json',
        'batch_size': 4,
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 0  # 設為0避免多進程問題（Windows）
    }
    
    # 設備設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建數據集
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    train_dataset = EndplateDataset(
        config['data_dir'],
        config['train_annotations'],
        transform=train_transform
    )
    val_dataset = EndplateDataset(
        config['data_dir'],
        config['val_annotations'],
        transform=val_transform
    )
    
    # 創建數據載入器（使用自定義collate）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn  # 自定義collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn  # 自定義collate
    )
    
    # 創建模型
    model = EndplateDetectionModel(pretrained=True)
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 創建訓練器
    trainer = EndplateTrainer(model, train_loader, val_loader, device, config)
    
    # 開始訓練
    trainer.train()

if __name__ == "__main__":
    main()
