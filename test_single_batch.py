#!/usr/bin/env python3
"""
測試單個批次 (V3 版)
用於驗證 V3 模型數據載入和多通道 heatmap 是否正確
"""

import os
import json
import torch
from train_vertebra_model import VertebraDataset, get_transforms, HEATMAP_SIZE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("Testing single batch (V3 multi-channel heatmap)...")
print("=" * 60)

# 配置
data_dir = '.'
train_ann = 'endplate_training_data/annotations/train_annotations.json'

# 檢查檔案
if not os.path.exists(train_ann):
    print(f"Training annotation not found: {train_ann}")
    print("Please run: python prepare_endplate_data.py")
    exit(1)

print(f"Found training annotations: {train_ann}")

# 創建數據集
transform = get_transforms(is_training=False)
dataset = VertebraDataset(data_dir, train_ann, transform=transform)

print(f"Dataset size: {len(dataset)}")

# 測試載入第一個樣本
print("\n" + "=" * 60)
print("Testing first sample...")

try:
    image, targets = dataset[0]

    print("Successfully loaded sample")
    print(f"\nImage shape: {image.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")

    print(f"\nTargets:")
    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
            if value.numel() <= 20:
                print(f"    values: {value}")
            else:
                print(f"    range: [{value.min():.4f}, {value.max():.4f}]")
                print(f"    nonzero: {(value > 0.01).sum().item()} elements")
        else:
            print(f"  {key}: {value}")

    # 視覺化
    print("\nGenerating visualization...")

    # 反正規化影像
    img = image.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # 取得 heatmap
    heatmaps = targets['heatmaps'].numpy()  # [C, H, W]
    num_channels = heatmaps.shape[0]

    # 合併 heatmap (max across channels)
    combined_heatmap = heatmaps.max(axis=0)

    # 找出有信號的 channels
    active_channels = []
    for ch in range(num_channels):
        if heatmaps[ch].max() > 0.01:
            active_channels.append(ch)

    print(f"  Heatmap channels: {num_channels} total, {len(active_channels)} active")
    print(f"  Active channels: {active_channels}")

    # 繪圖: 原圖 + 合併 heatmap + 各 channel
    n_preview = min(len(active_channels), 8)
    fig_cols = 2 + min(n_preview, 4)
    fig_rows = 1 + (n_preview > 4)

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 4, fig_rows * 4))
    if fig_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'V3 Single Batch Test - {num_channels}ch Heatmap', fontsize=14, fontweight='bold')

    # 原始圖像
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 合併 heatmap
    axes[0, 1].imshow(img)
    hm_resized = np.array(
        __import__('PIL.Image', fromlist=['Image']).Image.fromarray(
            (combined_heatmap * 255).astype(np.uint8)
        ).resize((img.shape[1], img.shape[0]))
    ) / 255.0 if combined_heatmap.shape != img.shape[:2] else combined_heatmap
    axes[0, 1].imshow(hm_resized, alpha=0.6, cmap='hot')
    axes[0, 1].set_title(f'Combined Heatmap\n({len(active_channels)} active ch)')
    axes[0, 1].axis('off')

    # 各 channel 預覽
    corner_names = ['AntSup', 'PostSup', 'PostInf', 'AntInf']
    for idx in range(n_preview):
        ch = active_channels[idx]
        vertebra_idx = ch // 4
        corner_idx = ch % 4
        r = idx // (fig_cols - 2) if (fig_cols - 2) > 0 else 0
        c = idx % (fig_cols - 2) + 2 if (fig_cols - 2) > 0 else idx + 2

        if r < fig_rows and c < fig_cols:
            axes[r, c].imshow(heatmaps[ch], cmap='hot', vmin=0, vmax=1)
            axes[r, c].set_title(f'Ch{ch}: V{vertebra_idx} {corner_names[corner_idx]}')
            axes[r, c].axis('off')

    # 隱藏多餘的子圖
    for r in range(fig_rows):
        for c in range(fig_cols):
            if not axes[r, c].has_data():
                axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig('test_single_batch_output.png', dpi=150, bbox_inches='tight')
    print("Visualization saved: test_single_batch_output.png")

    # 測試批次載入
    print("\n" + "=" * 60)
    print("Testing batch loading...")

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    for batch_idx, (images, targets_batch) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images: {images.shape}")
        for key, value in targets_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
            else:
                print(f"  {key}: {type(value)}")
        break

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nReady to train! Run:")
    print("  python train_vertebra_model.py")
    print("or")
    print("  double-click 2_train_model.bat")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nPlease check:")
    print("1. Image files exist (DICOM/PNG)")
    print("2. JSON annotations are in correct V2 format")
    print("3. Re-run: python prepare_endplate_data.py")
