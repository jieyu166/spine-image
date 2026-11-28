# 使用指南 V3 (Usage Guide)

本文件整合了：推理使用說明、問題排查、數據質量指南、快速參考。

---

## 目錄
1. [推理使用方法](#推理使用方法)
2. [問題排查](#問題排查)
3. [數據質量與改進](#數據質量與改進)
4. [快速參考](#快速參考)

---

## 推理使用方法

### 方法1: 命令行推理

```bash
# 單一檔案分析 (腰椎)
python inference_vertebra.py --model best_vertebra_model.pth --input spine.png --spine-type L

# 頸椎分析
python inference_vertebra.py --model best_vertebra_model.pth --input cspine.dcm --spine-type C

# 指定輸出目錄
python inference_vertebra.py --model best_vertebra_model.pth --input spine.png --output results/

# 批次處理
python inference_vertebra.py --model best_vertebra_model.pth --input patient_data/ --no-viz

# 調整信心度門檻
python inference_vertebra.py --model best_vertebra_model.pth --input spine.png --threshold 0.3
```

### 方法2: API 伺服器

```bash
# 啟動伺服器
pip install fastapi uvicorn python-multipart
python api_server_vertebra.py

# 瀏覽器開啟 http://localhost:8001 (內建 Web 介面)
# API 文件 http://localhost:8001/docs
```

**Python 調用**:
```python
import requests

with open('spine.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/api/predict',
        files={'file': f},
        data={'spine_type': 'L'}
    )
result = response.json()
print(f"Detected {result['predicted_count']} vertebrae")
for v in result['vertebrae']:
    print(f"  {v['name']}: {len(v['points'])} corners")
```

### 方法3: Python 腳本整合

```python
from inference_vertebra import VertebraInference

analyzer = VertebraInference('best_vertebra_model.pth', device='cuda')

# 完整分析
result = analyzer.analyze('spine.png', spine_type='L',
                          output_dir='results/', visualize=True)

# 輸出椎體資訊
for v in result['vertebrae']:
    print(f"{v['name']}: {len(v['points'])} corners")
    if v.get('anteriorWedgingFracture'):
        print(f"  !! Anterior wedging fracture detected")

# 輸出椎間盤資訊
for d in result['discs']:
    print(f"Disc {d['level']}: ant={d['anteriorHeight']:.1f} post={d['posteriorHeight']:.1f}")
```

### 方法4: 批次檔 (Windows)

```
雙擊 3_inference.bat
→ 選擇推理模式 (單檔/批次/範例)
→ 選擇脊椎類型 (L/C)
→ 結果存於 inference_results/
```

---

## 問題排查

### 錯誤1: Model not loaded
**原因**: `best_vertebra_model.pth` 不存在
**解決**: 先訓練模型
```bash
python train_vertebra_model.py
```

### 錯誤2: Detected V2 checkpoint
**現象**: 推理可執行但結果品質差 (角點集中)
**原因**: 尚未用 V3 架構訓練，仍使用 V2 模型
**解決**: 重新訓練
```bash
python train_vertebra_model.py
```

### 錯誤3: CUDA out of memory
**解決**: 修改 `train_vertebra_model.py` 中的 batch_size:
```python
'batch_size': 1,  # 從 4 改為 1
```

### 錯誤4: 找到 0 個有效標註檔案
**原因**: 路徑配置錯誤或不在正確目錄
**解決**:
```bash
cd "到 Spine 資料夾"
python quick_test.py
```

### 錯誤5: FileNotFoundError: Cannot load image
**原因**: JSON 中的影像路徑無效
**解決**: 確認 PNG/DICOM 檔案與 JSON 在同一資料夾

### 錯誤6: Application startup failed (API server)
**原因**: 模型 checkpoint 格式不相容
**解決**: V3 的 `inference_vertebra.py` 已支援自動偵測 V2/V3

### 診斷命令
```powershell
python quick_test.py              # JSON 標註驗證
python test_model_quick_start.py  # 模型架構測試
python test_single_batch.py       # 數據載入測試
python test_inference_debug.py    # 推理除錯
```

---

## 數據質量與改進

### 當前狀態
- 訓練樣本: ~29 個有效標註
- V1 格式 (DICOM): 7 個
- V2 格式 (PNG): 22 個
- 總椎體數: ~190 個

### 建議數量
| 樣本數 | 預期效果 |
|--------|---------|
| 29 (目前) | 基本學習，可能過擬合 |
| 50-100 | 開始泛化 |
| 100-300 | 良好效果 |
| 300+ | 穩健表現 |

### V3 增強策略
V3 模型包含更強的數據增強，部分彌補樣本不足：
- ShiftScaleRotate (位移/縮放/旋轉)
- CLAHE (直方圖均衡)
- RandomGamma (伽瑪變換)
- GaussianBlur (高斯模糊)
- RandomBrightnessContrast

### 改進建議
1. 持續標註更多影像 (用 spinal-annotation-web.html)
2. 包含不同設備/參數的影像
3. 包含不同病理狀態 (骨折、滑脫)
4. 確保影像品質良好

---

## 快速參考

### 執行順序
```
1. 0_quick_test.bat              # 驗證 JSON 標註
2. 1_prepare_data.bat            # 準備訓練數據
3. test_single_batch.bat         # 測試數據載入 (可選)
4. 2_train_model.bat             # 訓練 V3 模型
5. 3_inference.bat               # 推理預測
```

### 或使用 RUN_ALL.bat
```
RUN_ALL.bat                      # 一鍵執行: 驗證 → 準備 → 訓練
```

### 重要檔案

| 檔案 | 用途 |
|------|------|
| `train_vertebra_model.py` | V3 訓練程式 (多通道 heatmap) |
| `inference_vertebra.py` | V3 推理預測 |
| `api_server_vertebra.py` | V3 API 伺服器 (port 8001) |
| `prepare_endplate_data.py` | 數據準備 (支援 V1/V2 格式) |
| `quick_test.py` | JSON 標註驗證 |
| `best_vertebra_model.pth` | 訓練好的模型 checkpoint |
| `spinal-annotation-web.html` | 標註工具 |

### 已棄用檔案 (V1 Legacy)

| 檔案 | 說明 |
|------|------|
| `train_endplate_model.py` | V1 終板檢測訓練 |
| `inference.py` | V1 終板檢測推理 |
| `best_endplate_model.pth` | V1 模型 checkpoint |

---

## Web 前端使用

### spine-inference-web.html
透過 API 伺服器提供的 Web 介面：
1. 啟動 `python api_server_vertebra.py`
2. 瀏覽器開啟 `http://localhost:8001`
3. 上傳 X 光影像
4. 選擇脊椎類型 (L/C)
5. 查看偵測結果 (角點、heatmap、椎間盤分析)
