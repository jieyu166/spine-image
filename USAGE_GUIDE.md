# 使用指南 (Usage Guide)

本文件整合了：推理使用說明、問題排查、數據質量指南、快速參考。

---

## 目錄
1. [推理使用方法](#推理使用方法)
2. [問題排查](#問題排查)
3. [數據質量與改進](#數據質量與改進)
4. [快速參考](#快速參考)

---

## 推理使用方法

### 方法1: 命令行推理（推薦）

```bash
# 單一檔案分析
python inference.py --model best_endplate_model.pth --input spine.dcm

# 指定輸出目錄
python inference.py --model best_endplate_model.pth --input spine.dcm --output results/

# 批次處理
python inference.py --model best_endplate_model.pth --input patient_data/ --no-viz
```

### 方法2: API 伺服器

```bash
# 啟動伺服器
pip install fastapi uvicorn python-multipart
python api_server.py

# 訪問 http://localhost:8000/docs 查看API文檔
```

**Python 調用**:
```python
import requests

with open('spine.dcm', 'rb') as f:
    response = requests.post('http://localhost:8000/api/analyze', files={'file': f})
result = response.json()
print(f"檢測到 {result['num_angles']} 個角度")
```

### 方法3: Python 腳本整合

```python
from inference import SpineAnalyzer

analyzer = SpineAnalyzer('best_endplate_model.pth', device='cuda')
results = analyzer.analyze('spine.dcm', output_dir='results/', visualize=True)

for angle in results['angles']:
    print(f"椎間隙 #{angle['level']}: {angle['angle']}°")
```

### 輸出格式

```json
{
  "image_file": "spine.dcm",
  "num_endplates": 12,
  "num_angles": 11,
  "endplate_lines": [
    {"x1": 450, "y1": 320, "x2": 1850, "y2": 335}
  ],
  "angles": [
    {"level": 1, "angle": 12.5, "lower_line": [...], "upper_line": [...]}
  ]
}
```

---

## 問題排查

### 錯誤1: 找到 0 個有效標註檔案
**原因**: 路徑配置錯誤或不在正確目錄
**解決**:
```bash
cd "0. Inbox\Spine"
python quick_test.py
```

### 錯誤2: FileNotFoundError: Cannot load image
**原因**: JSON 中 image_path 為空
**解決**: 執行 `1_prepare_data_FIXED.bat`

### 錯誤3: cv::findDecoder imread: can't open
**原因**: OpenCV 無法讀取 DICOM
**解決**: 已修正，重新執行批次檔

### 錯誤4: cv2.error: Layout incompatible
**原因**: NumPy 記憶體佈局問題
**解決**: 已修正，使用獨立遮罩

### 錯誤5: stack expects equal size
**原因**: 遮罩或關鍵點尺寸不一致
**解決**: 已修正，使用 custom_collate_fn

### 錯誤6: CUDA out of memory
**解決**: 修改 `train_endplate_model.py`:
```python
'batch_size': 1,  # 從4改為1
```

### 錯誤7: 座標值異常小
**診斷**: 執行 `test_debug.bat` 檢查座標縮放
**解決**: 調整閾值或霍夫參數

### 診斷命令
```powershell
python check_dicom.py        # DICOM 配對檢查
python quick_test.py         # JSON 驗證
python test_single_batch.py  # 數據載入測試
```

---

## 數據質量與改進

### 當前問題
- 訓練樣本: **僅3個** (嚴重不足)
- 最少需求: 100個
- 建議數量: 300-500個

### 影響
- 模型過擬合
- 可能關注錯誤區域（如病歷號）
- 測試錯誤率高

### 解決方案

#### 立即可行: 裁切病歷號區域
```bash
雙擊 crop_dicom.bat
選擇激進模式
```
**預期改善**: 10-20%

#### 根本解決: 收集更多數據
| 樣本數 | 預期準確度 |
|--------|-----------|
| 3 (裁切後) | 30-40% |
| 50 | 60-70% |
| 100 | 70-85% |
| 300+ | 85-95% |

### 數據收集建議
- 從 PACS 選擇不同患者
- 包含不同設備/參數
- 包含不同病理狀態
- 確保影像品質良好

---

## 快速參考

### 執行順序
```
1. check_dicom.bat           # 檢查配對
2. 0_quick_test.bat          # 驗證JSON
3. 1_prepare_data_FIXED.bat  # 準備數據
4. test_single_batch.bat     # 測試載入
5. 2_train_model.bat         # 訓練
6. 3_inference.bat           # 推理
```

### 預期輸出

**check_dicom.bat**:
```
✅ 198261530.json
   ✅ 找到配對DICOM: 198261530.dcm
有效配對: 3
```

**test_single_batch.bat**:
```
✅ 成功載入樣本
圖像形狀: torch.Size([3, 512, 512])
endplate_mask: torch.Size([1, 512, 512])
✅ 所有測試通過！
```

**2_train_model.bat**:
```
Epoch 1/100
Training: 100%|████| Loss: 2.345
Validation: 100%|████| Val Loss: 2.567
✅ 保存最佳模型
```

### 預期損失變化
```
Epoch 1:   Loss ~2.5
Epoch 10:  Loss ~1.8
Epoch 50:  Loss ~1.0
Epoch 100: Loss ~0.5
```

### 檢查清單

**開始前**:
- [ ] Python 3.8+ 已安裝
- [ ] `pip install -r requirements.txt`
- [ ] 在 Spine 資料夾中
- [ ] DICOM 和 JSON 在同一目錄

**數據準備**:
- [ ] check_dicom.bat 通過
- [ ] 0_quick_test.bat 通過
- [ ] 生成 endplate_training_data 資料夾

**訓練前**:
- [ ] test_single_batch.bat 通過
- [ ] 無錯誤訊息

### 重要檔案

| 檔案 | 用途 |
|------|------|
| `train_endplate_model.py` | 主訓練程式 |
| `inference.py` | 推論預測 |
| `prepare_endplate_data.py` | 資料準備 |
| `best_endplate_model.pth` | 訓練好的模型 |
| `quick_test.py` | JSON 驗證 |
| `check_dicom.py` | DICOM 配對檢查 |

---

## 網頁工具使用

### 三種檢測模式

| 模式 | 特點 | 適用場景 |
|------|------|----------|
| 手動模式 | 完全人工控制 | 需要高精度 |
| AI深度學習 | 自動檢測 | 大量處理 |
| 傳統CV | 離線可用 | 無網路環境 |

### AI 模式使用步驟
1. 點擊「載入AI模型」
2. 上傳 X 光片
3. 切換到「AI自動」模式
4. 點擊「自動檢測」
5. 檢查信心度分數

### 信心度解讀
- 🟢 >80%: 結果可信
- 🟡 60-80%: 建議確認
- 🔴 <60%: 使用手動模式
