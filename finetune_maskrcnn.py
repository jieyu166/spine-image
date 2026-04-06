#!/usr/bin/env python3
"""
Fine-tune Mask R-CNN for Spine Vertebra Detection
使用現有 JSON 角點標註 → 自動轉換為 bbox + mask → fine-tune

Usage:
    python finetune_maskrcnn.py
    python finetune_maskrcnn.py --epochs 30 --lr 0.001
    python finetune_maskrcnn.py --base-weights spinefm_weights/mask_rcnn_2_categories.pth
"""

import os
import json
import glob
import argparse
import random
import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime


# 用於從 4/6 角點生成 mask 的角點順序
CORNER_ORDER = ['anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior']


class SpineAnnotationDataset(Dataset):
    """將現有 JSON 角點標註轉為 Mask R-CNN 訓練格式

    每個椎體 → 1 個 instance:
        - box: 角點的 bounding box (+ padding)
        - mask: 角點連成的四邊形 filled polygon
        - label: 1 (vertebra)
    """

    def __init__(self, json_files, transform=None, box_padding_ratio=0.05):
        self.samples = []
        self.transform = transform
        self.box_padding_ratio = box_padding_ratio

        for jf in json_files:
            p = Path(jf)
            img_path = p.with_suffix('.png')
            if not img_path.exists():
                continue
            self.samples.append((str(img_path), str(jf)))

        print(f"Dataset: {len(self.samples)} images loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # 載入影像
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 載入標註
        with open(json_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)

        boxes = []
        masks = []
        labels = []

        for vert in ann.get('vertebrae', []):
            pts = vert.get('points', {})

            # 收集所有可用角點
            corners = []
            for cn in CORNER_ORDER:
                if cn in pts:
                    corners.append([pts[cn]['x'], pts[cn]['y']])

            if len(corners) < 3:
                continue

            corners = np.array(corners, dtype=np.float32)

            # Bounding box (with padding)
            x_min, y_min = corners.min(axis=0)
            x_max, y_max = corners.max(axis=0)
            bw = x_max - x_min
            bh = y_max - y_min
            pad_x = bw * self.box_padding_ratio
            pad_y = bh * self.box_padding_ratio
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w, x_max + pad_x)
            y_max = min(h, y_max + pad_y)

            # 確保 box 有效
            if x_max - x_min < 5 or y_max - y_min < 5:
                continue

            boxes.append([x_min, y_min, x_max, y_max])

            # Polygon mask
            mask = np.zeros((h, w), dtype=np.uint8)
            poly = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [poly], 1)
            masks.append(mask)

            labels.append(1)  # all vertebra

        if len(boxes) == 0:
            # 空標註: 回傳空 target
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'masks': torch.zeros((0, h, w), dtype=torch.uint8),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor(idx),
            }
        else:
            target = {
                'boxes': torch.as_tensor(np.array(boxes), dtype=torch.float32),
                'masks': torch.as_tensor(np.stack(masks), dtype=torch.uint8),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor(idx),
            }

        # Image → tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(base_weights=None, device='cpu'):
    """建立 Mask R-CNN 模型 (2 classes: background + vertebra)"""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    # 修改為 2 classes
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = (
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)
    )

    # 載入 SpineFM 預訓練權重 (若有)
    if base_weights and os.path.exists(base_weights):
        print(f"Loading base weights: {base_weights}")
        state_dict = torch.load(base_weights, map_location=device)
        try:
            model.load_state_dict(state_dict)
            print("  Base weights loaded successfully")
        except RuntimeError as e:
            print(f"  Partial load (mismatched keys): {e}")
            model.load_state_dict(state_dict, strict=False)

    return model


def evaluate_model(model, data_loader, device):
    """簡易評估: 計算平均偵測數量和 score"""
    model.eval()
    total_detected = 0
    total_gt = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                scores = output['scores'].cpu().numpy()
                detected = (scores > 0.5).sum()
                gt = len(targets[i]['boxes'])
                total_detected += detected
                total_gt += gt
                total_images += 1

    avg_detected = total_detected / max(total_images, 1)
    avg_gt = total_gt / max(total_images, 1)
    recall_approx = total_detected / max(total_gt, 1)

    return {
        'avg_detected': float(avg_detected),
        'avg_gt': float(avg_gt),
        'recall_approx': float(min(recall_approx, 1.0)),
        'total_images': total_images,
    }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 收集資料 ──
    json_files = sorted(glob.glob('Images/**/*.json', recursive=True))
    json_files = [f for f in json_files if 'spinefm' not in f]
    print(f"Found {len(json_files)} annotated images")

    # Train/val split
    random.seed(42)
    random.shuffle(json_files)
    split_idx = int(len(json_files) * 0.85)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_dataset = SpineAnnotationDataset(train_files)
    val_dataset = SpineAnnotationDataset(val_files)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # ── 建立模型 ──
    model = build_model(base_weights=args.base_weights, device=device)
    model = model.to(device)

    # ── Optimizer ──
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── 訓練前評估 ──
    print("\n--- Before fine-tuning ---")
    pre_eval = evaluate_model(model, val_loader, device)
    print(f"  Val: avg detected={pre_eval['avg_detected']:.1f}, "
          f"avg GT={pre_eval['avg_gt']:.1f}, "
          f"recall~={pre_eval['recall_approx']:.1%}")

    # ── 訓練迴圈 ──
    print(f"\n--- Training ({args.epochs} epochs) ---")
    best_recall = pre_eval['recall_approx']
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            batch_count += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / max(batch_count, 1)

        # 定期評估
        if epoch % 5 == 0 or epoch == args.epochs:
            val_eval = evaluate_model(model, val_loader, device)
            print(f"  Epoch {epoch:3d}/{args.epochs}: "
                  f"loss={avg_loss:.4f}, "
                  f"val_detected={val_eval['avg_detected']:.1f}/{val_eval['avg_gt']:.1f}, "
                  f"recall~={val_eval['recall_approx']:.1%}")

            # 儲存最佳模型
            if val_eval['recall_approx'] >= best_recall:
                best_recall = val_eval['recall_approx']
                save_path = output_dir / 'mask_rcnn_finetuned.pth'
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model (recall~={best_recall:.1%})")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f}")

    # ── 最終儲存 ──
    final_path = output_dir / 'mask_rcnn_finetuned.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    # ── 訓練後評估 ──
    print("\n--- After fine-tuning ---")
    post_eval = evaluate_model(model, val_loader, device)
    print(f"  Val: avg detected={post_eval['avg_detected']:.1f}, "
          f"avg GT={post_eval['avg_gt']:.1f}, "
          f"recall~={post_eval['recall_approx']:.1%}")
    print(f"  Improvement: {pre_eval['recall_approx']:.1%} -> {post_eval['recall_approx']:.1%}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Mask R-CNN for spine vertebra detection')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--base-weights', default='spinefm_weights/mask_rcnn_2_categories.pth',
                        help='SpineFM pre-trained Mask R-CNN weights (NHANES)')
    parser.add_argument('--output-dir', default='spinefm_weights',
                        help='Output directory for fine-tuned weights')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
