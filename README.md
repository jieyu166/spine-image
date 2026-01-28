# 脊椎終板檢測專案 (Spine Endplate Detection)

## 專案概述

基於深度學習的脊椎終板自動檢測系統，用於：
- 自動識別脊椎X光影像中的終板位置
- 檢測椎體前後緣
- 計算椎間隙角度
- 輔助診斷脊椎不穩定性

**狀態**: ✅ Production Ready (v2.0)
**最後更新**: 2025-10-10

---

## 快速開始

### 前置需求
```bash
pip install -r requirements.txt
```

### 執行順序（雙擊批次檔）
```
1. check_dicom.bat           # 檢查DICOM-JSON配對
2. 0_quick_test.bat          # 驗證JSON格式
3. 1_prepare_data_FIXED.bat  # 準備訓練數據
4. test_single_batch.bat     # 測試數據載入
5. 2_train_model.bat         # 開始訓練
6. 3_inference.bat           # 推理預測
```

或執行 `RUN_ALL.bat` 一鍵完成。

---

## 專案結構

```
Spine/
├── 核心 Python 腳本
│   ├── train_endplate_model.py    # 主訓練程式
│   ├── inference.py               # 推論預測
│   ├── prepare_endplate_data.py   # 資料準備
│   ├── api_server.py              # FastAPI 伺服器
│   ├── quick_test.py              # JSON 驗證
│   ├── check_dicom.py             # DICOM 配對檢查
│   └── preprocess_crop_dicom.py   # 裁切 DICOM
│
├── Batch 自動化
│   ├── 0_quick_test.bat
│   ├── 1_prepare_data_FIXED.bat
│   ├── 2_train_model.bat
│   ├── 3_inference.bat
│   └── RUN_ALL.bat
│
├── 模型
│   └── best_endplate_model.pth    # 訓練好的模型
│
├── 資料夾
│   ├── Images/                    # 訓練資料 (DICOM + JSON)
│   ├── endplate_training_data/    # 生成的訓練集
│   └── inference_results/         # 推理結果
│
├── HTML 網頁工具
│   ├── spinal-angle-measurement.html
│   ├── spinal-angle-measurement-ml.html
│   └── spinal-annotation-web.html
│
└── 文件
    ├── README.md                  # 本文件（入門指南）
    ├── USAGE_GUIDE.md             # 詳細使用指南
    └── TECHNICAL_REFERENCE.md     # 技術參考文件
```

---

## 模型架構

```
輸入: [B, 3, 512, 512]
  ↓
ResNet50 Encoder (預訓練)
  ↓
U-Net Decoder
  ↓
三個輸出分支:
├── endplate_seg: [B, 1, 256, 256]      # 終板分割
├── vertebra_edge_seg: [B, 2, 256, 256] # 前後緣分割
└── keypoint_heatmap: [B, 1, 256, 256]  # 關鍵點熱圖
```

### 損失函數
```
L_total = α·L_endplate + β·L_edge + γ·L_keypoint
- L_endplate: BCE Loss (終板分割)
- L_edge: BCE Loss (前後緣分割)
- L_keypoint: MSE Loss (關鍵點熱圖)
- α=1.0, β=1.0, γ=0.5
```

---

## 使用方法

### 1. 命令行推理
```bash
python inference.py --model best_endplate_model.pth --input spine.dcm
```

### 2. API 服務
```bash
python api_server.py
# 訪問 http://localhost:8000/docs
```

### 3. Python 整合
```python
from inference import SpineAnalyzer

analyzer = SpineAnalyzer('best_endplate_model.pth')
results = analyzer.analyze('spine.dcm')
print(f"檢測到 {results['num_angles']} 個角度")
```

---

## 數據質量注意事項

**當前問題**: 訓練樣本僅3個，會導致過擬合。

**建議**:
- 最少需要: 100個樣本
- 建議數量: 300-500個樣本
- 使用 `crop_dicom.bat` 裁切病歷號區域減少干擾

---

## 文件說明

| 文件 | 內容 |
|------|------|
| `README.md` | 快速入門（本文件） |
| `USAGE_GUIDE.md` | 詳細使用指南、推理說明、問題排查 |
| `TECHNICAL_REFERENCE.md` | 技術細節、已解決問題、標註規範 |

---

## 常見問題快查

| 問題 | 解決方案 |
|------|---------|
| 找到0個標註檔案 | 確認在 Spine 資料夾執行 |
| Cannot load image | 執行 `1_prepare_data_FIXED.bat` |
| CUDA out of memory | 修改 batch_size 為 1 |
| 座標值太小 | 執行 `test_debug.bat` 診斷 |

詳細問題排查請參考 `USAGE_GUIDE.md`。
