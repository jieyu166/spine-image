# Pretrained Backbone Weights

存放 backbone 預訓練權重。**檔案不 commit 到 repo**（太大，`.gitignore` 已排除）。

## RadImageNet ResNet50（V3.2 使用）

- **來源**: https://github.com/BMEII-AI/RadImageNet
- **預訓練資料**: 1.35M 張放射影像（CT / MRI / Ultrasound）
- **用途**: 取代 ImageNet（貓狗車）預訓練，縮小 X-ray domain gap
- **檔案位置**: `pretrained/RadImageNet-ResNet50.pt`

### 下載步驟

1. 到 RadImageNet repo 的 [Release 頁面](https://github.com/BMEII-AI/RadImageNet) 或
   [Dropbox 連結](https://www.dropbox.com/sh/hhuwq5nxtm2trj7/AABSvKQnY9Tk8W4A9yQc7Tz5a)
   下載 ResNet50 的 PyTorch 版 `.pt` 檔
2. 改名為 `RadImageNet-ResNet50.pt`
3. 放到這個資料夾：`pretrained/RadImageNet-ResNet50.pt`

### Key 相容性

`train_vertebra_model.py` 的 `load_radimagenet_weights()` 會自動處理：

- `module.` 前綴（DataParallel 儲存的權重）→ 自動去掉
- `fc.*` classification head（num_classes 不同）→ 自動剔除
- 外包一層 `{'state_dict': ...}` → 自動拆開

如果載入時有 unexpected keys，會 print 警告但繼續執行（`strict=False`）。

## Fallback 行為

若 `pretrained/RadImageNet-ResNet50.pt` 不存在：

- 訓練腳本會自動 fallback 到 torchvision 的 ImageNet ResNet50
- console 會印出警告，提醒目前用的是哪個 backbone
- checkpoint 會記錄 `backbone_source` 欄位（`radimagenet` / `imagenet` / `random`）

## 驗證載入

```python
import torch
sd = torch.load("pretrained/RadImageNet-ResNet50.pt", map_location="cpu")
print(f"Keys: {len(sd)}")
print(f"First 5: {list(sd.keys())[:5]}")
# 期望看到 'conv1.weight', 'bn1.weight', 'layer1.0.conv1.weight' 之類
```
