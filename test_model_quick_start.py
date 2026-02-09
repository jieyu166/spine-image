#!/usr/bin/env python3
"""
快速測試椎體頂點檢測模型 (V3 版)
Quick Test for Vertebra Corner Detection Model V3

用於測試模型架構、前向傳播、數據載入是否正常
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from train_vertebra_model import VertebraCornerModel, VertebraDataset, get_transforms, HEATMAP_SIZE
import warnings
warnings.filterwarnings('ignore')

class QuickTester:
    """快速測試器 (V3 Heatmap Model)"""

    def __init__(self, data_dir, device='cpu'):
        self.data_dir = Path(data_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        print(f"Device: {self.device}")

    def collect_annotations(self):
        """收集所有標註檔案 (支援 V1 + V2 格式)"""
        print("\nCollecting annotations...")

        json_files = list(self.data_dir.glob('**/*.json'))
        valid_annotations = []

        skip_names = {'training_data', 'dataset_info', 'train_annotations', 'val_annotations'}

        for json_file in json_files:
            if json_file.stem in skip_names:
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # V2 format
                if 'vertebrae' in data and isinstance(data['vertebrae'], list) and len(data['vertebrae']) > 0:
                    valid_annotations.append({'file': json_file, 'data': data, 'format': 'V2'})
                    print(f"  [V2] {json_file.name}: {len(data['vertebrae'])} vertebrae")
                # V1 format
                elif 'measurements' in data and len(data.get('measurements', [])) > 0:
                    valid_annotations.append({'file': json_file, 'data': data, 'format': 'V1'})
                    print(f"  [V1] {json_file.name}: {len(data['measurements'])} measurements")

            except Exception as e:
                print(f"  Skip {json_file.name}: {e}")

        print(f"\nFound {len(valid_annotations)} valid annotations")
        return valid_annotations

    def create_test_model(self):
        """建立 V3 測試模型"""
        print("\nCreating V3 model (VertebraCornerModel)...")

        self.model = VertebraCornerModel(max_vertebrae=8, pretrained=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        print(f"  Heatmap size: {HEATMAP_SIZE}x{HEATMAP_SIZE}")
        print(f"  Output channels: {8 * 4} (8 vertebrae x 4 corners)")

        return self.model

    def test_forward_pass(self, test_image_path=None):
        """測試前向傳播"""
        print("\nTesting forward pass...")

        if test_image_path and os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            print("  Using random test image")

        # 預處理
        transform = get_transforms(is_training=False)
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)

        print(f"  Input shape: {input_tensor.shape}")

        # 前向傳播
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # 檢查輸出
        print("\n  Output check:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
                print(f"      range: [{value.min():.4f}, {value.max():.4f}]")
                print(f"      mean: {value.mean():.4f}")

        return outputs

    def generate_test_report(self, annotations):
        """生成測試報告"""
        print("\nGenerating test report...")

        report = {
            "model_version": "V3 (multi-channel heatmap)",
            "device": str(self.device),
            "heatmap_size": HEATMAP_SIZE,
            "dataset_info": {
                "total_annotations": len(annotations),
                "v1_count": sum(1 for a in annotations if a['format'] == 'V1'),
                "v2_count": sum(1 for a in annotations if a['format'] == 'V2'),
            },
            "model_info": {
                "architecture": "ResNet50 + UNet Decoder (multi-channel heatmap)",
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "output_channels": 8 * 4,
            }
        }

        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("  Report saved: test_report.json")

        print("\n" + "=" * 50)
        print("Test Summary")
        print("=" * 50)
        print(f"  Annotations: {report['dataset_info']['total_annotations']} (V1: {report['dataset_info']['v1_count']}, V2: {report['dataset_info']['v2_count']})")
        print(f"  Model: {report['model_info']['architecture']}")
        print(f"  Params: {report['model_info']['total_parameters']:,}")
        print(f"  Heatmap: {HEATMAP_SIZE}x{HEATMAP_SIZE}, {report['model_info']['output_channels']} channels")
        print("=" * 50)

        return report


def main():
    print("=" * 60)
    print("Vertebra Corner Detection - Quick Test (V3)")
    print("=" * 60)

    data_dir = '.'

    tester = QuickTester(data_dir, device='cuda')

    # Step 1: Collect annotations
    annotations = tester.collect_annotations()

    if len(annotations) == 0:
        print("\nNo valid annotations found")
        return

    # Step 2: Create model
    tester.create_test_model()

    # Step 3: Forward pass test
    tester.test_forward_pass()

    # Step 4: Generate report
    tester.generate_test_report(annotations)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python train_vertebra_model.py")
    print("  2. Or double-click: 2_train_model.bat")

if __name__ == "__main__":
    main()
