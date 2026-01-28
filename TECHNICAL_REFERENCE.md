# 技術參考文件 (Technical Reference)

本文件整合了：技術細節、已解決問題、標註規範、3D Slicer 使用指南。

---

## 目錄
1. [系統架構](#系統架構)
2. [已解決的技術問題](#已解決的技術問題)
3. [標註規範](#標註規範)
4. [3D Slicer 使用指南](#3d-slicer-使用指南)
5. [API 參考](#api-參考)

---

## 系統架構

### 數據流程
```
DICOM檔案
  → pydicom載入
  → NumPy陣列
  → 創建遮罩（原始尺寸）
  → 分離Transform
     - 遮罩：Resize(512,512)
     - 圖像：Resize+Augmentation+ToTensorV2
  → custom_collate_fn
  → 模型訓練
```

### 模型架構詳細
```
輸入層: [B, 3, 512, 512]

Encoder (ResNet50 預訓練):
├── Encoder1: [B, 64, 256, 256]
├── Encoder2: [B, 256, 128, 128]
├── Encoder3: [B, 512, 64, 64]
├── Encoder4: [B, 1024, 32, 32]
└── Encoder5: [B, 2048, 16, 16]

Decoder (U-Net style):
├── Decoder1 + Skip4: [B, 1024, 32, 32]
├── Decoder2 + Skip3: [B, 512, 64, 64]
├── Decoder3 + Skip2: [B, 256, 128, 128]
└── Decoder4 + Skip1: [B, 128, 256, 256]

輸出頭:
├── endplate_seg: [B, 1, 256, 256]
├── vertebra_edge_seg: [B, 2, 256, 256]
└── keypoint_heatmap: [B, 1, 256, 256]
```

### 損失函數
```python
L_total = α * L_endplate + β * L_edge + γ * L_keypoint

# 其中:
L_endplate = BCE_Loss(pred_endplate, target_endplate)
L_edge = BCE_Loss(pred_edge, target_edge)
L_keypoint = MSE_Loss(pred_heatmap, target_heatmap)

# 權重:
α = 1.0, β = 1.0, γ = 0.5
```

---

## 已解決的技術問題

### 問題 1: 路徑配置錯誤
**錯誤**: `找到0個有效標註檔案`
**修正**: `prepare_endplate_data.py` 第291行改為 `data_dir='.'`

### 問題 2: DICOM 路徑丟失
**錯誤**: `FileNotFoundError: Cannot load image`
**修正**: 自動搜尋對應 DICOM 檔案
```python
dcm_path = json_path.with_suffix('.dcm')
if dcm_path.exists():
    image_path = str(dcm_path.relative_to(self.input_dir))
```

### 問題 3: OpenCV 無法讀取 DICOM
**錯誤**: `cv::findDecoder imread: can't open`
**修正**: 使用 pydicom 載入
```python
if image_path.endswith('.dcm'):
    dcm = pydicom.dcmread(image_path)
    image = dcm.pixel_array
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
```

### 問題 4: NumPy 記憶體佈局
**錯誤**: `cv2.error: Layout of the output array img is incompatible`
**修正**: 使用獨立遮罩
```python
anterior_mask = np.zeros((h, w), dtype=np.float32)
posterior_mask = np.zeros((h, w), dtype=np.float32)
cv2.line(anterior_mask, ...)
cv2.line(posterior_mask, ...)
vertebra_edge_mask[:,:,0] = anterior_mask
vertebra_edge_mask[:,:,1] = posterior_mask
```

### 問題 5: Windows 多進程
**錯誤**: DataLoader worker error
**修正**: `num_workers=0`

### 問題 6: 遮罩尺寸不一致
**錯誤**: `stack expects each tensor to be equal size`
**修正**: 統一 resize 到 512x512，分離 transform

### 問題 7: ToTensorV2 類型衝突
**錯誤**: `expected np.ndarray (got Tensor)`
**修正**: 遮罩和圖像使用分離的 transform

### 問題 8: 關鍵點數量不同
**錯誤**: `stack expects equal size (keypoints)`
**修正**: 使用 custom_collate_fn，keypoints 保持為 list

### 問題 9: 損失函數尺寸不匹配
**錯誤**: `target size different to input size`
**修正**: 使用 F.interpolate 調整目標遮罩尺寸
```python
endplate_mask_resized = F.interpolate(
    targets['endplate_mask'],
    size=(pred_h, pred_w),
    mode='bilinear'
)
```

---

## 標註規範

### JSON 資料結構
```json
{
  "patient_id": "P001",
  "study_id": "S20240115001",
  "spine_type": "L",
  "image_type": "flexion",
  "image_dimensions": {
    "width": 1936,
    "height": 3408
  },
  "measurements": [
    {
      "level": "L4/L5",
      "angle": 15.3,
      "angle_raw": 164.7,
      "confidence": 0.95,
      "lowerEndplate": [
        {"x": 959.7, "y": 2016.9},
        {"x": 657.5, "y": 2037.4}
      ],
      "upperEndplate": [
        {"x": 963.8, "y": 1974.1},
        {"x": 673.8, "y": 1959.8}
      ]
    }
  ],
  "vertebra_edges": {
    "L5": {
      "anterior": [
        {"x": 657.5, "y": 2037.4},
        {"x": 633.0, "y": 2243.6}
      ],
      "posterior": [
        {"x": 959.7, "y": 2016.9},
        {"x": 947.5, "y": 2223.2}
      ]
    }
  }
}
```

### 標註標準

**終板標記**:
- 每個終板需要 2 個端點
- x 座標較小 → 前緣 (anterior)
- x 座標較大 → 後緣 (posterior)

**信心度評分**:
| 分數 | 說明 |
|------|------|
| 0.9-1.0 | 終板邊界非常清晰 |
| 0.8-0.9 | 終板邊界清晰 |
| 0.7-0.8 | 終板邊界較清晰 |
| 0.6-0.7 | 終板邊界模糊 |
| <0.6 | 測量不可靠 |

**角度計算**:
```
角度 = |atan2(y2-y1, x2-x1) - atan2(y4-y3, x4-x3)| * 180/π
```
- 正常範圍: 5-25度
- 異常範圍: >25度 或 <5度

---

## 3D Slicer 使用指南

### 載入腳本
```python
# 在 3D Slicer Python Console 執行
exec(open('slicer_export_spine_annotations.py').read())
```

### 建立範本線
```python
# 腰椎範本
create_spine_template_lines()

# 頸椎範本
create_cervical_template_lines()
```

### 標註流程
1. 載入脊椎 X 光片到 3D Slicer
2. 執行腳本載入函數
3. 選擇對應的範本線
4. 調整線條位置到終板邊界
5. 設定信心度和註記

### 匯出標註
```python
export_spine_annotations()
```

### 角度銳角化規則
```python
def _normalize_angle_to_acute(diff_deg):
    diff = abs(diff_deg) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff
```

---

## API 參考

### inference.py 參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--model` | 模型檔案路徑 | best_endplate_model.pth |
| `--input` | 輸入影像或資料夾 | 必填 |
| `--output` | 輸出結果資料夾 | inference_results |
| `--device` | 計算設備 | auto |
| `--no-viz` | 不生成視覺化 | False |

### API 端點

**POST /api/analyze**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@spine.dcm" \
  -F "threshold=0.5" \
  -F "return_image=true"
```

**回應格式**:
```json
{
  "image_file": "spine.dcm",
  "num_endplates": 12,
  "num_angles": 11,
  "endplate_lines": [...],
  "angles": [...],
  "result_image": "base64..."
}
```

### SpineAnalyzer 類別

```python
from inference import SpineAnalyzer

# 初始化
analyzer = SpineAnalyzer(model_path, device='cuda')

# 分析影像
results = analyzer.analyze(
    image_path,
    output_dir='results/',
    visualize=True
)

# 提取終板線段（自訂閾值）
endplate_lines = analyzer.extract_endplates(
    mask,
    threshold=0.3,
    min_length=50
)
```

---

## 配置參數

### 訓練配置 (train_endplate_model.py)
```python
config = {
    'data_dir': '.',
    'batch_size': 4,        # GPU 記憶體不足時改為 1
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_workers': 0        # Windows 必須為 0
}
```

### 數據增強配置
```python
A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    ToTensorV2()
])
```

---

## 性能指標

### 預期訓練結果
| Epochs | Loss | IoU | 關鍵點誤差 |
|--------|------|-----|-----------|
| 1 | ~2.5 | ~0.05 | >100px |
| 50 | ~1.0 | ~0.65 | ~30px |
| 100 | ~0.5 | ~0.80 | <20px |

### 推理性能
| 設備 | 單張影像 | 批次處理 |
|------|---------|---------|
| GPU | ~0.1-0.2秒 | ~0.05秒/張 |
| CPU | ~1-2秒 | ~0.5秒/張 |

---

## 版本歷史

### v2.0 (2025-10-10)
- ✅ 完整 ML 訓練流程
- ✅ 推理和 API 支援
- ✅ 9 個主要問題已解決
- ✅ 完整文檔系統

### v1.0 (初版)
- 基礎網頁工具
- 手動標註功能
- OpenAI API 整合
