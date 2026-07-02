# Spine-Image Repo 改造計畫 — Claude Code Handoff

> 這份文件是從 chat 討論濃縮的 handoff，給 Claude Code 接手繼續改 `jieyu166/spine-image` repo 用。
> 核心原則：**不要重寫架構，只換 pretrained backbone + 訓練策略**。

---

## 背景：為什麼寫這份文件

使用者 Jieyu（Chi Mei Medical Center 放射科醫師）目前有：

- **Repo**: `jieyu166/spine-image` — V3.1 架構，ResNet50 + UNet Decoder + Multi-channel Heatmap + Focal Loss + CoordConv + Sub-pixel refinement
- **標註資料**: 約 50 份 JSON 標註（C-spine 約一半、L-spine 約一半）
- **標註方式**: 每椎體 4 corners（anteriorSuperior / posteriorSuperior / posteriorInferior / anteriorInferior）
- **硬體**: GPU 有但不強（願意接受訓練時間變長換更好效果）
- **目標**: 微調模型讓它能準確預測台灣人群頸椎與腰椎 lateral X-ray 的椎體 4 corners

專案狀態在 chat 討論前原本是「土法煉鋼、醫師沒空繼續標註」，希望用公開資料集 + fine-tune 的方式推進。

---

## 核心判斷（重要：先讀這段）

在搜尋了十幾篇論文與多個 GitHub repo 後得出的結論：

### ✅ 使用者現有 V3 架構本身就是 SOTA，不要換掉

- ResNet50 + UNet Decoder + multi-channel heatmap 是文獻上 vertebra corner detection 最常見、效果最好的架構家族之一（參考 IRCCS Galeazzi 10,193 張 sagittal 腰椎論文）
- Keypoint R-CNN 適合「未知數量實例」（如人體姿態），**不適合**「固定脊椎數量的 landmark」，heatmap regression 才是正確選擇
- 使用者現有的標註工具（`spinal-annotation-web.html`）、API server、Docker 部署、Colab notebook 都是成熟資產，整個換架構會損失這些

### ❌ 真正的問題在：backbone 用 ImageNet pretrained（貓狗車），domain gap 太大

- 50 張資料 fine-tune 不夠補這個 domain gap
- 解法：backbone 換成 medical image pretrained（RadImageNet 或 TorchXRayVision）

### 📋 要改的是三件事（不是整個架構）

1. **Backbone pretrained weights**: ImageNet → RadImageNet ResNet50
2. **訓練策略**: 加 5-fold cross validation（50 張資料不做 CV 會嚴重 overfit 單一 split）
3. **資料增強**: 針對 X-ray 特性調整（CLAHE、mild rotation、noise；C-spine 禁 horizontal flip、L-spine 可用）

---

## 具體改動清單

### 改動 1：換 backbone pretrained weights

**檔案**: `train_vertebra_model.py`（可能也要改 `inference_vertebra.py` 的 model 建構部分）

**目前（推測的樣子）**:
```python
import torchvision.models as models
from torchvision.models import ResNet50_Weights

backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
```

**改成**:
```python
import torchvision.models as models

# RadImageNet weights (download once, cache locally)
# Source: https://github.com/BMEII-AI/RadImageNet
# Direct download: https://www.dropbox.com/s/...   (去 RadImageNet repo 拿最新連結)
RADIMAGENET_PATH = "pretrained/RadImageNet-ResNet50.pt"

backbone = models.resnet50(weights=None)  # 不要 ImageNet weights
state_dict = torch.load(RADIMAGENET_PATH, map_location='cpu')
# RadImageNet 權重的 key 可能有前綴（例如 "module."），需要處理：
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# 可能還要剔除最後的 fc layer（因為 num_classes 不同）：
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
backbone.load_state_dict(state_dict, strict=False)
```

**備選**: 如果 RadImageNet 權重下載有問題，用 `torchxrayvision`：
```python
import torchxrayvision as xrv
# DenseNet121 pretrained on 14 datasets including NIH ChestX-ray14, CheXpert...
backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
# 注意：這會讓你的架構從 ResNet50 變成 DenseNet121，skip connection 層要重接
```

**優先順序**: RadImageNet ResNet50（最小改動，保留 skip connection 結構）> TorchXRayVision DenseNet121

### 改動 2：加 5-fold cross validation

**檔案**: 可能要新建 `train_vertebra_model_cv.py` 或改 `train_vertebra_model.py`

**理由**: 50 張資料隨機切 train/val 80/20，val 只有 10 張，雜訊極大、容易誤判訓練成敗。5-fold CV 可以用全部資料驗證每個 epoch。

**骨架**:
```python
from sklearn.model_selection import KFold

all_json_files = sorted(Path("Images").glob("*.json"))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_val_losses = []
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_json_files)):
    print(f"=== Fold {fold_idx + 1}/5 ===")
    train_files = [all_json_files[i] for i in train_idx]
    val_files = [all_json_files[i] for i in val_idx]
    
    # 建 dataset / dataloader / model / optimizer (每個 fold 重新 init)
    # 訓練 → 記錄最佳 val loss
    best_val = train_one_fold(train_files, val_files, fold_idx)
    fold_val_losses.append(best_val)

print(f"CV mean val loss: {np.mean(fold_val_losses):.4f} ± {np.std(fold_val_losses):.4f}")
```

**最終模型**: CV 跑完後，用**全部 50 張資料**跑一個 final training（epoch 數用 CV 找出的平均最佳 epoch），這個 model 拿去部署。

### 改動 3：X-ray 專屬資料增強

**檔案**: `prepare_endplate_data.py` 或 `train_vertebra_model.py` 的 Dataset class

**推薦的 augmentation pipeline（Albumentations）**:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# C-spine 和 L-spine 共用
train_transform_common = A.Compose([
    A.CLAHE(clip_limit=(1, 4), p=0.5),              # X-ray 對比度
    A.RandomBrightnessContrast(
        brightness_limit=0.15, contrast_limit=0.15, p=0.5
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Rotate(limit=10, p=0.5, border_mode=0),        # 小幅旋轉
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5
    ),
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# L-spine 可以加 horizontal flip（左右對稱、不影響醫學意義）
train_transform_lspine = A.Compose([
    A.HorizontalFlip(p=0.5),
    *train_transform_common.transforms,
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# C-spine 不要 horizontal flip！頸椎 lateral 的前後方向有醫學意義，flip 會讓 C2 前緣變後緣，標註也要連動翻
# 如果想加 flip，必須同步處理標註的 anterior/posterior 對調，不建議
```

**注意**:
- keypoint 要在 `keypoint_params` 中宣告，Albumentations 會自動同步變換
- `remove_invisible=False` 很重要，avoid 有些點被轉出畫面而被丟掉
- 不要用 `VerticalFlip`（會讓 superior/inferior 顛倒）
- 不要用強烈 rotation（> 15°，椎體朝向會異常）

### 改動 4：訓練策略微調（配合 pretrained backbone）

**Two-stage unfreezing schedule**:

```python
# Stage A: 只訓練 head（UNet decoder + heatmap head），凍結整個 backbone
# 用意：先讓 head 學會如何把 RadImageNet feature map 轉成 vertebra heatmap
# 時長：~5 epochs，lr=1e-3
for param in model.backbone.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# Stage B: 解凍 backbone 的 layer3, layer4（深層）
# 用意：讓深層特徵微調到 spine X-ray 的 domain
# 時長：~15 epochs，lr=1e-4（比 stage A 小 10 倍）
for param in model.backbone.layer3.parameters():
    param.requires_grad = True
for param in model.backbone.layer4.parameters():
    param.requires_grad = True
# layer0, layer1, layer2 保持凍結（low-level feature 不需要動）
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

**為什麼這樣分**: 小資料集（50 張）做 full fine-tune 很容易把預訓練的 feature 毀掉。分階段解凍是 transfer learning 的標準做法。

---

## 評估指標建議

目前 repo 看起來主要用 loss 評估，建議補上臨床相關指標：

```python
def compute_metrics(pred_corners, gt_corners, vertebra_heights):
    """
    pred_corners, gt_corners: shape [N_vertebrae, 4, 2]
    vertebra_heights: 每個椎體的平均高度（像素），用來正規化
    
    Returns:
        mae_px: 每個 corner 的平均像素誤差
        pck_005: Percentage of Correct Keypoints within 5% of vertebra height
    """
    diff = np.linalg.norm(pred_corners - gt_corners, axis=-1)  # [N, 4]
    mae_px = diff.mean()
    
    # PCK@0.05：誤差 < 5% 椎體高度的 keypoint 比例
    threshold = 0.05 * vertebra_heights[:, None]  # [N, 1]
    pck_005 = (diff < threshold).mean()
    
    return mae_px, pck_005
```

**目標**:
- MAE < 3 pixels（512x512 影像上，相當於 ~0.6% 影像寬度）
- PCK@0.05 > 90%
- 下游 Cobb angle 預測 ICC > 0.9（和手測比較）

---

## 資料增量建議（可選，中長期）

### 短期（保持現有 50 張）

以上改動 1-4 即可。預期效果：比原本 ImageNet pretrained + 無 CV 顯著提升。

### 中期（建議補到 200-300 張）

使用者提到「沒空繼續標註」，建議用**半自動加速**：

1. 用改動後的 model 跑 inference 在 unlabeled X-ray 上
2. 只修正錯誤的 corner（每張 30 秒 vs 純手標 3-5 分鐘）
3. 修正後的加回訓練集、retrain

`pkl-contour-editor.html` 已經有基礎，擴充成這個 workflow 不難。

### 長期：加入 NHANES II 公開資料集做 pre-fine-tune

NHANES II 有 544 張 lateral C-spine + L-spine 有 landmark annotation，可當 middle-stage pretraining：

```
ImageNet → RadImageNet → NHANES II (544) → Your 50 labeled → Production
       (換掉)       (換成這個)   (中段 pre-FT)    (最終 FT)
```

這是更野心的計劃，但如果第一階段效果還不夠，這是下一步。

---

## 檔案優先順序（Claude Code 可以按這個順序動）

1. **讀懂現狀**:
   - `train_vertebra_model.py` — 確認 backbone 初始化方式
   - `inference_vertebra.py` — 確認 model 建構部分（load checkpoint 要相容新 backbone）
   - `prepare_endplate_data.py` — 確認 Dataset class 的資料流

2. **改動 1 優先**: 換 RadImageNet pretrained backbone
   - 新建 `pretrained/` 資料夾存權重
   - `.gitignore` 加入 `pretrained/*.pt`（權重檔太大，不要 commit）
   - README 加一段說明如何下載權重

3. **改動 3**: 加 Albumentations augmentation（相對獨立，可以和改動 1 並行）

4. **改動 2**: 加 5-fold CV（比較大改動，放後面）

5. **改動 4**: 兩階段 unfreeze schedule（微調，配合改動 1）

6. **加評估指標**: MAE、PCK、Cobb ICC

---

## 相容性與風險提醒

1. **checkpoint 不相容**: 換 backbone 後，舊的 `best_vertebra_model.pth` 不能直接載入新 model。`inference_vertebra.py` 的 V2/V3 auto-detect 邏輯要擴充成支援 V3.2（RadImageNet backbone）。建議用 checkpoint 的 `version` 欄位區分。

2. **Docker 映像要更新**:
   - `requirements-docker.txt` 加 `albumentations`、`torchxrayvision`（如選後者）
   - RadImageNet 權重要放進 image 或 mount volume
   - Synology NAS 部署記得測一次

3. **API server 可能要重訓後才能用**: 部署到 Synology 前先本機跑通一個 fold 確認 inference 正確。

4. **不要一次全改**: 建議先只做「改動 1」看 loss curve 變化，確認 RadImageNet backbone 有效後再疊加其他改動。每個改動 commit 一次方便 bisect。

---

## RadImageNet 權重下載與驗證

**官方 repo**: https://github.com/BMEII-AI/RadImageNet

**權重格式**: 訓練時用的是 Keras/TensorFlow，但他們有提供 PyTorch 轉換版。如果只有 TF 版，需要自行轉換：

```python
# TF → PyTorch 轉換驗證（簡版）
import torch
state_dict = torch.load("RadImageNet-ResNet50.pt")
print(f"Total keys: {len(state_dict)}")
print(f"First 5 keys: {list(state_dict.keys())[:5]}")
# 期望看到類似 "conv1.weight", "bn1.weight", "layer1.0.conv1.weight"...
```

**常見 key 差異**:
- `module.conv1.weight` → 需要去掉 `module.` 前綴（DataParallel 儲存的）
- `fc.weight` / `fc.bias` → 應該剔除（classification head 不需要）
- `downsample.0.weight` vs `downsample.conv.weight` → 不同版本 PyTorch 可能命名不同

**fallback**: 如果 RadImageNet 下載真的有問題，用 `timm` 的 pretrained ResNet50 on ChestX-ray14：

```python
import timm
backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
# 然後手動載入 ChestX-ray14 weights
```

---

## 開發流程建議

```bash
# 1. 建立 feature branch
git checkout -b feature/radimagenet-backbone

# 2. 下載 pretrained weights（不 commit）
mkdir -p pretrained
# 從 RadImageNet repo 下載 ResNet50 權重到 pretrained/RadImageNet-ResNet50.pt

# 3. 改 train_vertebra_model.py 的 backbone init
# 4. 跑 5 epoch 確認能 train 下去、loss 有下降
python train_vertebra_model.py --epochs 5 --debug

# 5. 跑 full training（用 CV）
python train_vertebra_model_cv.py --folds 5 --epochs 30

# 6. Compare vs baseline
# baseline: 原本 ImageNet pretrained + 無 CV
# new: RadImageNet + 5-fold CV + X-ray augmentation

# 7. 確認 inference 相容性
python inference_vertebra.py --model checkpoints/fold_best.pth --input test.png

# 8. 更新 Dockerfile、requirements、README
# 9. PR / merge
```

---

## 最後：Chat 討論的脈絡補充

這個 repo 改造討論是從 Claude 陪 Jieyu 做 C-spine X-ray 報告練習開始的——她上傳 C-spine lateral 圖、Claude 依照 `spine-xray` skill 產生報告、她給 ground truth、Claude 學習校正、更新 skill。

這個 `spine-image` repo 是 Jieyu 獨立進行的更大專案：**用 DL 自動偵測椎體 endplate，然後輸出 JSON 供下游計算 Cobb、wedge ratio、listhesis 等指標**。短期目標是輔助自己看片效率，長期可能走向 PACS 整合。

她的工作流程：
- 放射科醫師（Chi Mei Medical Center，奇美醫院）
- 熱衷寫 AHK + GPT API 整合的 radiology workflow tool
- Obsidian 知識管理 + PARA 系統
- 偏好結構化、務實、效率導向的討論

所以 Claude Code 接手時，請保持：
- 繁體中文 commit message / comment（她的 repo 是中英混用，以繁中為主）
- 不要大改她已經穩定的部分（標註工具、API、部署）
- 每個改動寫清楚「為什麼這樣改」
- 如果有 debug，幫她用 print / log 輸出中間結果，方便她自己接手後 iterate
