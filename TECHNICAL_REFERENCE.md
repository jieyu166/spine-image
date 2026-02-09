# 技術參考文件 V3 (Technical Reference)

本文件整合了：V3 模型技術細節、已解決問題、標註規範、API 參考。

---

## 目錄
1. [系統架構](#系統架構)
2. [已解決的技術問題](#已解決的技術問題)
3. [標註規範](#標註規範)
4. [API 參考](#api-參考)

---

## 系統架構

### 數據流程
```
影像 (DICOM/PNG/JPG)
  → pydicom 或 OpenCV 載入
  → NumPy 陣列 (RGB)
  → Albumentations 增強 + Resize(512,512)
  → ToTensorV2 + Normalize
  → 訓練: VertebraCornerModel (V3)
  → 推理: VertebraInference
```

### V3 模型架構詳細
```
輸入層: [B, 3, 512, 512]

Encoder (ResNet50 預訓練):
├── layer0 (conv1+bn+relu+maxpool): [B, 64, 128, 128]    ← skip1
├── layer1:                          [B, 256, 64, 64]     ← skip2
├── layer2:                          [B, 512, 32, 32]     ← skip3
├── layer3:                          [B, 1024, 16, 16]    ← skip4
└── layer4:                          [B, 2048, 8, 8]      ← bottleneck

UNet Decoder (skip connections):
├── up4: Upsample + [2048+1024, 512] → [B, 512, 16, 16]
├── up3: Upsample + [512+512, 256]   → [B, 256, 32, 32]
└── up2: Upsample + [256+256, 128]   → [B, 128, 64, 64]

Heatmap Head:
  Conv2d(128, 64, 3) → BN → ReLU
  → Upsample(scale=2)        → [B, 64, 128, 128]
  → Conv2d(64, 32, 1)        → [B, 32, 128, 128]

Count Head:
  AdaptiveAvgPool2d(1) → Flatten
  → Linear(2048, 256) → ReLU
  → Linear(256, 9)           → [B, 9] (0~8 vertebrae)
```

### 損失函數
```python
# V3 損失
L_total = Focal_Loss(heatmaps) + 0.5 * CrossEntropy(count)

# Focal Loss 參數
alpha = 2.0   # 正樣本聚焦因子
beta = 4.0    # 負樣本衰減因子
threshold = 0.01  # 正負樣本分界

# 正樣本: -(1-pred)^alpha * log(pred) * target^beta
# 負樣本: -(pred)^alpha * log(1-pred) * (1-target)^beta
```

### V2 vs V3 模型對照

| 模型 | 訓練腳本 | 推理腳本 | 模型檔案 | Checkpoint 特徵 |
|------|----------|----------|----------|----------------|
| V1 (終板) | train_endplate_model.py | inference.py | best_endplate_model.pth | endplate_seg |
| V2 (椎體回歸) | - | inference_vertebra.py | best_vertebra_model.pth | backbone.* keys |
| V3 (heatmap) | train_vertebra_model.py | inference_vertebra.py | best_vertebra_model.pth | layer0.* keys |

**自動偵測**: `inference_vertebra.py` 會檢查 checkpoint 的 key prefix，自動切換 V2/V3 解碼方式。

---

## 已解決的技術問題

### 問題 1: V2 模型推理結果集中 (核心問題)
**現象**: 所有角點聚集在影像中央一小區域
**原因**: `AdaptiveAvgPool2d(1)` 將空間資訊壓縮為 1x1，回歸分支無法學習空間位置
**修正**: V3 改用多通道 heatmap + UNet decoder with skip connections

### 問題 2: V2 checkpoint 不相容 V3 模型
**錯誤**: `RuntimeError: Error(s) in loading state_dict: Missing key(s)`
**修正**: `inference_vertebra.py` 中加入 `_load_v2_model()` 向下相容

### 問題 3: 2_train_model.bat 調用錯誤腳本
**錯誤**: 訓練完成後 checkpoint 仍為 V2 (backbone.* keys)
**原因**: bat 檔案仍呼叫 `train_endplate_model.py`
**修正**: 改為呼叫 `train_vertebra_model.py`

### 問題 4: quick_test.py 僅支援 V1 格式
**修正**: 新增 V2 格式偵測 (`vertebrae` field)

### 問題 5: Windows 多進程
**錯誤**: DataLoader worker error
**修正**: `num_workers=0`

### 問題 6: OpenCV 無法讀取 DICOM
**修正**: 使用 pydicom 載入 `.dcm` 檔案

---

## 標註規範

### JSON 標註格式 V2

```json
{
  "version": "2.0",
  "spineType": "L",
  "vertebrae": [
    {
      "name": "S1",
      "points": {
        "anteriorSuperior": {"x": 100, "y": 400},
        "posteriorSuperior": {"x": 300, "y": 410}
      },
      "isBoundary": true
    },
    {
      "name": "L5",
      "points": {
        "anteriorSuperior": {"x": 100, "y": 200},
        "posteriorSuperior": {"x": 300, "y": 210},
        "posteriorInferior": {"x": 310, "y": 350},
        "anteriorInferior": {"x": 110, "y": 340}
      }
    }
  ]
}
```

### 邊界椎體規則
| 脊椎類型 | 上邊界 (2點) | 下邊界 (2點) |
|----------|-------------|-------------|
| L-spine | S1 (只有上緣) | T12 (只有下緣) |
| C-spine | T1 (只有上緣) | C2 (只有下緣) |

### 角點名稱 (固定順序)
1. `anteriorSuperior` - 前上角
2. `posteriorSuperior` - 後上角
3. `posteriorInferior` - 後下角
4. `anteriorInferior` - 前下角

---

## API 參考

### inference_vertebra.py 參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--model` | 模型 checkpoint 路徑 | best_vertebra_model.pth |
| `--input` | 輸入影像或資料夾 | 必填 |
| `--output` | 輸出結果資料夾 | inference_results |
| `--spine-type` | 脊椎類型 L 或 C | L |
| `--device` | 計算設備 | auto |
| `--threshold` | Heatmap peak 信心度門檻 | 0.2 |
| `--no-viz` | 不生成視覺化 | False |

### API 伺服器 (api_server_vertebra.py)

```bash
python api_server_vertebra.py
# http://localhost:8001
# API docs: http://localhost:8001/docs
```

**POST /api/predict**
```bash
curl -X POST "http://localhost:8001/api/predict" \
  -F "file=@spine.png" \
  -F "spine_type=L"
```

**回應格式**:
```json
{
  "success": true,
  "spine_type": "L",
  "predicted_count": 7,
  "count_confidence": 0.85,
  "vertebrae": [
    {
      "name": "S1",
      "boundaryType": "upper",
      "points": {"anteriorSuperior": {"x": 100, "y": 400}, ...},
      "confidences": {"anteriorSuperior": 0.92, ...}
    }
  ],
  "discs": [...],
  "result_image": "data:image/png;base64,...",
  "heatmap_image": "data:image/png;base64,..."
}
```

### VertebraInference 類別

```python
from inference_vertebra import VertebraInference

# 初始化 (自動偵測 V2/V3 checkpoint)
analyzer = VertebraInference('best_vertebra_model.pth', device='cuda')

# 預測
result = analyzer.predict('spine.png', spine_type='L')

# 完整分析 (含儲存)
result = analyzer.analyze('spine.png', spine_type='L',
                          output_dir='results/', visualize=True)

# 結果結構
result['vertebrae']        # 椎體列表
result['discs']            # 椎間盤列表
result['heatmap']          # 合併 heatmap (2D numpy)
result['channel_heatmaps'] # 各通道 heatmap (V3 only)
result['original_image']   # 原始影像 (RGB numpy)
```

---

## 配置參數

### 訓練配置 (train_vertebra_model.py)
```python
config = {
    'data_dir': '.',
    'batch_size': 4,          # GPU 記憶體不足時改為 1-2
    'epochs': 150,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'num_workers': 0,         # Windows 必須為 0
    'heatmap_size': 128,      # heatmap 解析度
    'max_vertebrae': 8,       # 最多椎體數
}
```

### 數據增強配置
```python
# V3 增強策略 (train_vertebra_model.py)
A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=8, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize(...),
    ToTensorV2()
])
```

---

## 版本歷史

### v3.0 (2026-02)
- Multi-channel heatmap (32 channels)
- UNet decoder with skip connections
- Focal Loss
- Sub-pixel Taylor expansion refinement
- V2 checkpoint backward compatibility
- Enhanced augmentation (CLAHE, GaussianBlur, etc.)

### v2.0 (2025-10)
- Vertebra 4-corner annotation
- Regression branch (AdaptiveAvgPool2d)
- API server

### v1.0 (初版)
- Endplate annotation
- U-Net segmentation
