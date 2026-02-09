# 脊椎椎體檢測專案 V3 (Spine Vertebra Detection)

## 專案概述

基於深度學習的脊椎椎體頂點自動檢測系統，用於：
- 自動識別每個椎體的 4 個角點（前上、後上、後下、前下）
- 自動計算椎間盤高度、Wedge angle
- 檢測前緣壓迫性骨折（anterior wedging compression fracture）
- 檢測椎體滑脫（spondylolisthesis / retrolisthesis）
- 驗證椎間盤高度遞進規律

**狀態**: V3.0 Active Development
**最後更新**: 2026-02

---

## V3.0 新特性

### 模型架構升級 (V2 → V3)
- **V2 (舊)**: ResNet50 + AdaptiveAvgPool2d(1) 回歸分支 → 座標精度差
- **V3 (新)**: ResNet50 + UNet Decoder + 多通道 Heatmap → 空間精度大幅提升

### V3 核心改進
| 項目 | V2 | V3 |
|------|-----|-----|
| 座標提取 | 回歸分支 (FC層) | 多通道 heatmap peak |
| 空間解析度 | 1x1 (全域池化) | 128x128 heatmap |
| 損失函數 | BCE + MSE | Focal Loss + CE |
| Decoder | 無跳躍連接 | UNet-style skip connections |
| Sub-pixel | 無 | Taylor expansion refinement |

### 標註方式
標註每個椎體的 4 個角點，系統自動推算椎間盤資訊：
```
椎體 4 個頂點：
    前上角 ●─────────────● 後上角     ← 上終板
           │             │
           │   椎體      │
           │             │
    前下角 ●─────────────● 後下角     ← 下終板
           ↑             ↑
         前緣          後緣
```

### 自動計算指標
| 指標 | 計算方式 |
|------|----------|
| **壓迫性骨折** | 前緣高度 < 後緣高度 x 0.75 |
| **椎間盤高度** | 上椎體下終板 vs 下椎體上終板的距離 |
| **Wedge Angle** | 椎間盤上下終板夾角 |
| **Spondylolisthesis** | 後緣連線偏移 > 5% |

---

## 快速開始

### 前置需求
```bash
pip install -r requirements.txt
```

### 工作流程

#### 1. 標註數據
打開 `spinal-annotation-web.html`：
1. 選擇脊椎類型（L-spine 或 C-spine）
2. 載入或貼上 X 光影像
3. 依序標註每個椎體的 4 個角點
4. 匯出 JSON 檔案

**標註順序**：
- L-spine: 由下到上（S1 → L5 → L4 → ...）
- C-spine: 由上到下（C2 → C3 → C4 → ...）

#### 2. 準備訓練數據
```bash
python prepare_endplate_data.py
```

#### 3. 訓練模型
```bash
python train_vertebra_model.py
```

#### 4. 推理預測
```bash
python inference_vertebra.py --model best_vertebra_model.pth --input spine.png --spine-type L
```

#### 5. API 伺服器
```bash
python api_server_vertebra.py
# 瀏覽器開啟 http://localhost:8001
```

---

## 專案結構

```
Spine/
├── 核心腳本
│   ├── train_vertebra_model.py      # V3 訓練腳本 (多通道 heatmap)
│   ├── inference_vertebra.py        # V3 推理腳本 (heatmap peak 提取)
│   ├── api_server_vertebra.py       # V3 FastAPI 服務 (port 8001)
│   ├── prepare_endplate_data.py     # 數據準備 (支援 V1/V2 標註格式)
│   └── quick_test.py               # JSON 標註驗證
│
├── 標註工具
│   └── spinal-annotation-web.html   # V2 標註工具 (椎體4頂點)
│
├── 測試腳本
│   ├── test_model_quick_start.py    # 模型架構快速測試
│   ├── test_single_batch.py         # 數據載入與 heatmap 測試
│   └── test_inference_debug.py      # 推理除錯測試
│
├── 批次檔
│   ├── 0_quick_test.bat             # 驗證 JSON
│   ├── 1_prepare_data.bat           # 準備數據
│   ├── 2_train_model.bat            # 訓練模型
│   ├── 3_inference.bat              # 推理預測
│   └── RUN_ALL.bat                  # 完整流程
│
├── 已棄用 (V1 Legacy)
│   ├── train_endplate_model.py      # [V1] 終板檢測訓練
│   ├── inference.py                 # [V1] 終板檢測推理
│   └── api_server.py               # [V1] 終板 API (port 8000)
│
├── 文件
│   ├── README.md                    # 本文件
│   ├── USAGE_GUIDE.md               # 詳細使用指南
│   └── TECHNICAL_REFERENCE.md       # 技術參考
│
└── 資料夾
    ├── Images/                      # 訓練影像和標註
    ├── endplate_training_data/      # 訓練數據
    └── inference_results/           # 推理結果
```

---

## 模型架構 V3

```
輸入: [B, 3, 512, 512]
  ↓
ResNet50 Backbone (預訓練):
├── layer0: [B, 64, 128, 128]   ← skip connection
├── layer1: [B, 256, 64, 64]    ← skip connection
├── layer2: [B, 512, 32, 32]    ← skip connection
├── layer3: [B, 1024, 16, 16]   ← skip connection
└── layer4: [B, 2048, 8, 8]
  ↓
UNet Decoder (skip connections):
├── up4: [B, 512, 16, 16]   (+ layer3)
├── up3: [B, 256, 32, 32]   (+ layer2)
└── up2: [B, 128, 64, 64]   (+ layer1)
  ↓
輸出:
├── heatmaps: [B, 32, 128, 128]  (8椎體 x 4角點 = 32通道)
└── count_logits: [B, 9]          (0~8椎體計數)
```

### 損失函數
```
L_total = Focal_Loss(heatmaps) + 0.5 * CrossEntropy(count)

Focal Loss: (1-p)^alpha * -log(p)  (alpha=2.0, beta=4.0)
- 專門處理正負樣本不平衡
- 背景佔 heatmap 99%+ 的像素
```

---

## JSON 標註格式 V2

```json
{
  "version": "2.0",
  "spineType": "L",
  "vertebrae": [
    {
      "name": "L5",
      "points": {
        "anteriorSuperior": {"x": 100, "y": 200},
        "posteriorSuperior": {"x": 300, "y": 210},
        "posteriorInferior": {"x": 310, "y": 350},
        "anteriorInferior": {"x": 110, "y": 340}
      },
      "anteriorHeight": 140,
      "posteriorHeight": 140,
      "anteriorWedgingFracture": false
    }
  ],
  "discs": [...],
  "abnormalities": {...}
}
```

---

## 常見問題

| 問題 | 解決方案 |
|------|----------|
| 找不到標註檔案 | 確認 JSON 放在 Images/ 資料夾 |
| CUDA out of memory | 修改 batch_size 為 1 或 2 |
| V2 模型相容性 | inference_vertebra.py 自動偵測 V2/V3 checkpoint |
| 推理結果不佳 | 確認已用 V3 模型訓練 (2_train_model.bat) |

---

## 版本歷史

### v3.0 (2026-02)
- ResNet50 + UNet Decoder 多通道 heatmap 架構
- Focal Loss 處理正負樣本不平衡
- Sub-pixel Taylor expansion 精煉
- 增強數據增強 (CLAHE, GaussianBlur, etc.)
- 向下相容 V2 checkpoint

### v2.0 (2025-10)
- 椎體 4 角點標註 + 回歸分支
- API 伺服器

### v1.0 (初版)
- 終板標註方式
- 基礎 U-Net 分割
