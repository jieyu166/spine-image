#!/usr/bin/env python3
"""
測試單個批次
用於驗證數據載入和遮罩創建是否正確
"""

import os
import json
import torch
from train_endplate_model import EndplateDataset, get_transforms
import matplotlib.pyplot as plt
import numpy as np

print("🔬 測試單個批次...")
print("=" * 60)

# 配置
data_dir = '.'
train_ann = 'endplate_training_data/annotations/train_annotations.json'

# 檢查檔案
if not os.path.exists(train_ann):
    print(f"❌ 找不到訓練標註檔案: {train_ann}")
    print("請先執行: 1_prepare_data_FIXED.bat")
    exit(1)

print(f"✅ 找到訓練標註: {train_ann}")

# 創建數據集
transform = get_transforms(is_training=False)
dataset = EndplateDataset(data_dir, train_ann, transform=transform)

print(f"✅ 數據集大小: {len(dataset)}")

# 測試載入第一個樣本
print("\n" + "=" * 60)
print("測試第一個樣本...")

try:
    image, targets = dataset[0]
    
    print("✅ 成功載入樣本")
    print(f"\n圖像形狀: {image.shape}")
    print(f"圖像範圍: [{image.min():.3f}, {image.max():.3f}]")
    
    print(f"\n目標資訊:")
    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    # 視覺化
    print("\n生成視覺化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始圖像（反正規化）
    img = image.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('原始圖像')
    axes[0, 0].axis('off')
    
    # 終板遮罩
    endplate_mask = targets['endplate_mask'][0].numpy()
    axes[0, 1].imshow(img)
    axes[0, 1].imshow(endplate_mask, alpha=0.5, cmap='Reds')
    axes[0, 1].set_title(f'終板遮罩\n(非零像素: {(endplate_mask > 0).sum()})')
    axes[0, 1].axis('off')
    
    # 前緣遮罩
    anterior_mask = targets['vertebra_edge_mask'][0].numpy()
    axes[1, 0].imshow(img)
    axes[1, 0].imshow(anterior_mask, alpha=0.5, cmap='Blues')
    axes[1, 0].set_title(f'前緣遮罩\n(非零像素: {(anterior_mask > 0).sum()})')
    axes[1, 0].axis('off')
    
    # 後緣遮罩
    posterior_mask = targets['vertebra_edge_mask'][1].numpy()
    axes[1, 1].imshow(img)
    axes[1, 1].imshow(posterior_mask, alpha=0.5, cmap='Oranges')
    axes[1, 1].set_title(f'後緣遮罩\n(非零像素: {(posterior_mask > 0).sum()})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_single_batch_output.png', dpi=150, bbox_inches='tight')
    print("✅ 視覺化已儲存: test_single_batch_output.png")
    
    # 測試批次載入
    print("\n" + "=" * 60)
    print("測試批次載入...")
    
    from torch.utils.data import DataLoader
    from train_endplate_model import custom_collate_fn
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    for batch_idx, (images, targets_batch) in enumerate(loader):
        print(f"\n批次 {batch_idx}:")
        print(f"  圖像: {images.shape}")
        print(f"  目標數量: {len(targets_batch)}")
        
        # 只測試第一個批次
        if batch_idx == 0:
            for key, value in targets_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"  {key}: list of {len(value)} tensors, shapes: {[t.shape for t in value]}")
                else:
                    print(f"  {key}: {type(value)}")
        break
    
    print("\n" + "=" * 60)
    print("✅ 所有測試通過！")
    print("=" * 60)
    print("\n可以開始訓練了！執行:")
    print("  python train_endplate_model.py")
    print("或")
    print("  雙擊 2_train_model.bat")

except Exception as e:
    print(f"\n❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
    print("\n請檢查:")
    print("1. DICOM檔案是否存在")
    print("2. JSON標註格式是否正確")
    print("3. 重新執行 1_prepare_data_FIXED.bat")

