# 脊椎椎體偵測專案 (Spine Vertebra Detection)

基於深度學習的脊椎側位 X 光「椎體角點」自動偵測系統。對每節椎體標出 4 個角點（前上、後上、後下、前下），再自動算出椎間盤高度、Wedge angle、壓迫性骨折、滑脫等指標。

- **目前生產配置**：V3.4（aspect-aware）+ baseline + fold3 ensemble + 全推論後處理鏈
- **定位精度**：overall_mean **97.6 px** / median **41.9 px** / 29 of 32 影像 < 100 px
- **訓練資料**：~35 張人工標註（是的，就這麼少 —— 後面會講這件事為什麼是整個故事的主角）
- **最後更新**：2026-06-11

> 下面這份 changelog 我用上課的方式寫。不是因為想搞花樣，而是這個專案最值錢的東西不是「程式碼」，是「為什麼這樣做」。技巧本身網路上都查得到，但「在什麼情況下、為了解決什麼痛、最後付出什麼代價」—— 這個才是別人帶不走的。所以我們一個一個講。

---

## 教學式 Changelog：一個 35 張圖的模型，怎麼從 144 px 走到 97.6 px

好，各位同學大家好啊，我們今天要來搞懂一件事：**這個專案到底做了什麼，讓誤差掉了快一半。**

先講結論。一言以蔽之，這個故事其實就是在講一句話：**當你的資料只有 35 張，你能改的東西，比你想的少很多。** 大改架構的全死了，真正有用的，全部是「不重訓、只改推論」的小手術。

那這堂課我們照著時間軸走。我們先看模型長什麼樣（V3.0），再看一條走錯的岔路（V3.3），然後重點來了 —— 五個真正有效的技巧：**keep-aspect resize、命名錨定、TTA、DARK decode、ensemble**。最後講我自己怎麼看這件事。

精度是這樣一路掉下來的，你先有個印象：

```
144.0 px   起點（V3.4 + aspect-fix）
131.6 px   + 命名錨定
128.0 px   + TTA
103.5 px   + DARK decode   ← 單一最大躍進
 97.6 px   + ensemble       ← 首度跌破 100
```

median（中位數）更明顯：79 px → **41.9 px**，掉了 47%。

---

### 起手式：V3.0 ~ V3.2 —— 先把「黑盒子」講清楚

在開始之前，我們先用黑盒子的角度看這個模型。**輸入是什麼？輸出是什麼？它在 optimize 什麼？**

- 輸入：一張 512×512 的脊椎 X 光。
- 輸出：32 張 heatmap。為什麼是 32？因為 8 節椎體 × 每節 4 個角點 = 32。每一張 heatmap 負責「畫一個亮點」，亮點的位置就是那個角點的座標。
- 它在 optimize 什麼？讓亮點畫在對的地方。用的是 Focal Loss，因為背景佔了 99% 以上的像素，正負樣本超級不平衡。

所以你會發現，這個模型其實就是在玩「在 32 張紙上各點一個點」的遊戲而已。V3.0 到 V3.2 做的事情，是把 backbone 換成 RadImageNet（放射影像預訓練，比貓狗車的 ImageNet 貼近 X 光）、加 CoordConv 讓模型知道空間座標、用兩階段解凍訓練。這些把 val_loss 從 1.36 壓到 0.69。

好，到這邊都還算順。那問題就來了。

---

### 走錯的岔路：V3.3 ~ V3.6 —— 大改架構，全部陣亡

你可能會想說：要更準，那就把模型改強一點嘛。解析度調高、加 decoder、換更聰明的架構。對不對？這是每個人都會有的直覺。

我們試了。而且試了很多次。結果呢？**全部更糟。**

- **V3.3**：heatmap 解析度 128→256、高斯核縮小。結果 32 個 channel 塌縮在一起，多個角點全擠到影像同一個角落。val_loss 從 0.69 暴增到 2.22。
- **V3.5**：改成 CenterNet 那種「先找中心、再算偏移」的架構，試了三輪（sigma 4→2、NMS 3→5、區域監督）。每一輪都失敗，誤差 320~370 px。
- **V3.6**：加 geometric consistency loss，想強迫椎體形狀正確。結果 soft-argmax 在訓練初期亂給座標，反而把 heatmap 學壞，誤差衝到 508 px。

那為什麼會這樣呢？這邊是整個專案最重要的一課：**你的資料只有 35 張。** CenterNet 的 offset head 想學「每節椎體多大」，可是 35 張圖根本不夠，它最後學到的是「一個平均大小的椎體模板」套到所有人身上。不是架構爛，是資料餵不飽它。

所以這個任務**沒辦法硬 train 一發**。你硬 train 一發是 train 不起來的。資料量擺在那，再聰明的架構也補不回來。我們認賠，退回 V3.4，從此改變策略：**不動模型，只動推論。**

好，那神奇的地方來了。光是「只改推論」，我們就把 144 px 壓到了 97.6 px。

---

### 技巧一：keep-aspect resize —— 不要把人壓扁

先讓 naive 做法撞牆。

一張典型的腰椎側位片，大概是 1318 寬、3082 高。長寬比 2.34，瘦瘦長長的，對不對？那模型吃的是 512×512 的正方形。怎麼辦？

最直覺的做法叫 squash —— 直接把它「壓」成正方形。可是你想想看，2.34 倍高的東西硬壓成 1:1，垂直方向是不是被擠扁了 2.34 倍？椎體本來圓圓胖胖的，壓完變成一條一條的。模型看到的是被你捏扁的脊椎，它當然學不好 Y 方向。

怎麼辦呢？**keep-aspect resize。** 白話文就是：等比例縮，縮到最長邊剛好 512，短邊不夠的地方補黑邊（padding）。這樣椎體的形狀完整保留，不會被捏扁。

這就是 V3.4。它其實就是「縮圖的時候不要變形」這麼一件事而已。但它把 val_loss 從 0.75 壓到 0.69，是整個 V3.x 系列的轉捩點。

---

### 技巧二：命名錨定 —— 從哪一頭開始數，差很多

這個技巧最有趣，因為它修的不是「模型算錯」，是「我們會錯意」。

我們的 32 個 channel 是有順序的：channel 0~3 是「最上面那節」，4~7 是「第二節」，以此類推。模型只負責由上而下找出椎體。那「最上面那節到底叫什麼名字」—— 是我們在程式裡硬指定的。

舊的做法：從上面數，第一節就叫 T12，第二節 L1，以此類推。

那問題就來了。假設這張片子最上面其實是 L1（T12 被裁掉了，沒入鏡），可是模型的 count head 多數了一節。會發生什麼事？最上面那節被你叫成 T12，於是 L1 變 L2、L2 變 L3…… **整條脊椎的名字全部往下錯一格。** 跟正確答案一比，每一節都對到隔壁，誤差直接爆炸到 700~800 px。

你可能會想說：那把 count 數準一點不就好了？沒那麼簡單，count head 在 35 張圖下本來就不穩。

怎麼辦呢？換個地方下錨。**腰椎片裡，最穩定的解剖標誌是哪個？是 S1，薦椎。** 它是底部那塊大大的三角骨，幾乎每張片子都拍得到。反而是頂端常常被裁掉。所以我們改成「**從 S1 往上數**」—— 最底下那節一定是 S1，往上 L5、L4……數幾節算幾節。頸椎則相反，從最明顯的 C2 往下數（因為 T1 常被肩膀擋住）。

這個改完，18971571 那批從 306 px 掉到 87 px，零回歸。它其實就是「換一頭開始數名字」而已，一行邏輯，但效果立竿見影。

> 老實說一句：81161252、21584353 這兩張，名字修對了，誤差還是 700+。為什麼？因為它們是**模型真的看錯了**，heatmap 的亮點整片飄掉，不是命名問題。這種就不是後處理能救的，得靠更多資料。我們後面用一個「可信度旗標」把它們標記出來提醒醫師複查 —— 誠實面對救不了的東西，也是一種負責任。

---

### 技巧三：TTA（Test-Time Augmentation）—— 同一張圖，問模型好幾次

先講 TTA 是什麼。一言以蔽之，它其實就是：**同一張圖，我稍微轉一下、調個亮度，丟給模型問好幾次，再把答案平均起來。**

為什麼這樣有用？你可以想像，模型對某些角度、某些對比特別敏感。單問一次，可能剛好踩到它的盲點。問五次再平均，雜訊會互相抵消，剩下穩定的訊號。

實作上有兩個眉角，這是踩過坑才知道的：

第一，**heatmap 要在「機率空間」平均**。旋轉過的那次，輸出的 heatmap 也是斜的，要先把它轉正回來，才能跟其他次對齊相加。

第二，這個坑很經典 —— **count（椎體數量）不能跟著平均。** 我們一開始把 count 也平均了，結果旋轉的版本讓 count 翻了一格，連帶觸發剛剛講的「命名錯位」，18971571-1 從 87 暴衝到 291。後來改成「**count 只認沒動過的原圖那一次**」，才解決。

所以記住：TTA 是好東西，但它會搖動 count，而 count 一動，名字就亂。把這兩件事拆開，就沒事了。

對了，我們**不做水平翻轉**。為什麼？因為側位脊椎的前緣、後緣是有醫學意義的，翻過去前後就顛倒了，那是錯的。掃描下來發現，旋轉是主力，亮度幾乎沒貢獻。

---

### 技巧四：DARK decode —— 取亮點之前，先把它「糊」一下 ★最大功臣

這個是本專案單一最有效的技巧，128 px 一口氣掉到 103.5。我們慢慢講。

先讓你感受問題有多大。heatmap 是 128×128，可是原圖是 3082 高。換算一下，**heatmap 上動一格，原圖就是 24 px。** 所以你抓亮點的位置只要差個半格、一格，放回原圖就是十幾二十 px 的誤差。sub-pixel（次像素）的精度在這裡，是天大的事。

那原本怎麼抓亮點？就是找最大值那一格（argmax），再用泰勒展開微調一點點。可是如果 heatmap 有雜訊、有好幾個小峰，argmax 可能剛好跳到一個假的峰上，整個位置就歪了。怎麼辦呢？

**DARK 的做法：抓亮點之前，先用高斯把整張 heatmap 糊一下（blur）。**

你可能會想說：糊掉不是更不準嗎？欸，剛好相反。你想想看，一個對稱的高斯亮點，你把它模糊，**峰的位置不會動**，因為它本來就對稱。模糊只會把旁邊的雜訊小峰抹平、跟主峰融合，讓 argmax 老老實實落在真正的質心上。對好的亮點無害，對髒的亮點救命。

那要糊多少？這裡有個很漂亮的理論：**糊的程度（sigma）要對齊你訓練時標籤用的高斯大小。** 我們訓練時 label 的 sigma 是 6，所以 decode 也用 sigma=6。掃描下來，剛好就是 6~7 之間誤差最低。理論跟實驗對上了。

結果呢？32 張圖裡 23 張改善超過 20 px，**零回歸**。而且每一節對 GT 的距離都單調下降 —— 這點很重要，它證明亮點是「更靠近真值」，不是被你糊到塌縮在一起。

它其實就是「取點之前先模糊」這一個動作而已。一個 `cv2.GaussianBlur`。但它是整個專案最值錢的一行。

---

### 技巧五：ensemble —— 三個臭皮匠，但你要慎選皮匠

最後一招。ensemble，一言以蔽之就是：**養好幾個模型，把它們的答案平均起來。** 群體智慧，互相抵消個別的錯。

我們跑了 5-fold cross validation，訓出 5 個模型。那直覺上，5 個一起平均一定最強嘛，對不對？

**結果輸給單一模型。** 全 5-fold ensemble 是 107.6 px，比 baseline 單模型 103.5 還差。

為什麼？你看這 5 個 fold 的 val_loss：0.82、0.97、**0.62**、1.03、**1.67**。看到那個 1.67 沒有？那是個爛模型（35 張圖的 CV，分到難的驗證集就 train 不起來）。**你把一個爛模型平均進來，它不是幫忙，它是拖油瓶。** 三個臭皮匠勝過諸葛亮的前提，是三個都得是「皮匠」，不能有一個是豬隊友。

怎麼辦呢？**慎選。** 我們只挑「程度相當」的：baseline（val 0.6234）配 fold3（val 0.6230），兩個一樣強。結果 **97.6 px**，首度跌破 100。

這裡還有一個小細節：count 要用 baseline 那顆當主（它的 count head 比較準）。fold3 當主的話會退回 107。

所以 ensemble 的真正教訓不是「越多越好」，是 **「品質相當才有綜效，弱模型是毒藥」**。

> 工程上怎麼上線？我們用一個 sidecar 檔（`best_vertebra_model.pth.ensemble`），裡面就寫一行 fold3 的檔名而已。`VertebraInference` 載入時自動偵測、自動把 fold3 加進來。所以 compare、API、Colab 全部不用改一行 code。那 sidecar 就是一個文字檔而已。

---

### 好，我們來總結一下今天這堂課

今天我們走過一個 35 張圖的模型，從 144 px 到 97.6 px 的旅程。三個重點：

1. **資料少的時候，改架構基本上是死路。** V3.3/V3.5/V3.6 全軍覆沒就是證明。真正有用的是不重訓、只改推論的小手術。
2. **最值錢的常常是最簡單的。** DARK decode 就是取點前糊一下，一行；命名錨定就是換一頭數，一個邏輯。但它們加起來貢獻了大半。
3. **誠實面對救不了的東西。** 那 2 張 catastrophe 是真的看錯，後處理救不了，就老實標記「請複查」，等資料變多再說。

那如果你想再往下走 —— 攻那兩張難圖、攻側彎、攻滑脫 —— 答案不在程式裡了，在標註上。把 hard case 補到 60~80 張，故事才有下一章。那就留到下一堂課再跟大家講。

以上就是我今天想跟大家分享的內容。

---

## 推論後處理鏈：開關與參數

全部後處理都可以在 `VertebraInference` 物件上開關：

```python
from inference_vertebra import VertebraInference

inf = VertebraInference('best_vertebra_model.pth')   # 自動讀 sidecar 載入 fold3 ensemble

# 各技巧開關（預設全開）
inf.tta = False               # 關 TTA
inf.decode_blur_sigma = 0     # 關 DARK decode（預設 6.0）
inf.aspect_min = 0.0          # 關 aspect-fix（椎體長寬比修正）
inf.anchor_mode = 'top'       # 命名錨定改回從頂端數（L 預設 'bottom'）
inf.reliability_threshold = 0.5  # 可信度旗標門檻

result = inf.predict('spine.png', spine_type='L')
result['low_confidence']         # True = 建議人工完整複查
result['corner_confidence_min']  # 與誤差相關 -0.68
```

---

## 快速開始

### 前置需求
```bash
pip install -r requirements.txt
pip install albumentations==1.3.1   # 1.4+ 在 Python 3.14 有安裝問題
# RadImageNet 權重放 pretrained/RadImageNet-ResNet50.pt（找不到會 fallback 到 ImageNet）
```

### 工作流程

```bash
# 1. 標註：打開 spinal-annotation-web.html，標每節椎體 4 角點，匯出 JSON
#    L-spine 由下到上 (S1→L5→...)；C-spine 由上到下 (C2→C3→...)

# 2. 重生 train/val split（從 Images/ 掃描配對）
python regenerate_splits.py

# 3. 訓練（單模型）
python train_vertebra_model.py --epochs 30 --unfreeze-epoch 5

# 3b. 訓練（5-fold ensemble，~100 分鐘；先停睡眠 powercfg /change standby-timeout-ac 0）
python train_vertebra_model_cv.py --n-folds 5 --epochs 30 --unfreeze-epoch 5

# 4. 推論
python inference_vertebra.py --model best_vertebra_model.pth --input spine.png --spine-type L

# 5. QA：手標 vs 模型推論對照（產出 overlay + 統計）
python compare_annotations.py --input-dir Images --output-dir comparison_out --spine-type L --model best_vertebra_model.pth

# 6. 5-fold ensemble 評估
python eval_ensemble.py

# 7. API 伺服器
python api_server_vertebra.py   # http://localhost:8001
```

---

## 生產部署：三個檔缺一不可

上線（Drive / Docker / Colab）時，**這三個檔要放在同一層**，少了 sidecar 或 fold3 會 silently 退回 baseline 單模型（103.5 px，差 6 px）：

```
best_vertebra_model.pth            ← baseline 主模型 (val 0.6234)
best_vertebra_model_fold3.pth      ← ensemble 成員 (val 0.6230)
best_vertebra_model.pth.ensemble   ← sidecar 純文字（內容一行：best_vertebra_model_fold3.pth）
```

**Colab 步驟**：上傳上面 3 檔到 `MyDrive/Spine/weights/` → 開 [notebook](spine_inference_colab.ipynb) → 切 T4 GPU → 依序執行 cell。載入時務必看到 `Ensemble 啟用：主模型 + 1 個成員` 才表示 fold3 有載到。詳見 [HANDOFF_2026-06-11.md](HANDOFF_2026-06-11.md)。

---

## 模型架構 V3.4

```
輸入: [B, 3, 512, 512]   （aspect-aware：等比縮 + 黑邊 padding）
  ↓
ResNet50 Backbone (RadImageNet 預訓練):
├── layer0~4，多層 skip connection
  ↓
UNet Decoder (skip connections) + CoordConv + Channel Embedding
  ↓
輸出:
├── heatmaps: [B, 32, 128, 128]   (8 椎體 × 4 角點 = 32 通道)
└── count_logits: [B, 9]           (0~8 椎體計數)

推論後處理鏈:
  命名錨定 (S1 往上) → TTA (旋轉/亮度, count 解耦)
  → DARK decode (取峰前 σ=6 高斯平滑) → aspect-fix → ensemble 平均 → reliability 旗標
```

### 損失函數
```
L_total = Focal_Loss(heatmaps) + 0.5 × CrossEntropy(count)
Focal Loss (alpha=2.0, beta=4.0)：處理背景佔 99%+ 的正負樣本不平衡
```

---

## 專案結構（重點檔案）

```
Spine/
├── 核心
│   ├── train_vertebra_model.py      # V3.4 訓練（aspect-aware + 32-channel heatmap）
│   ├── train_vertebra_model_cv.py   # 5-fold CV 訓練
│   ├── inference_vertebra.py        # 推論 + 全後處理鏈（anchor/TTA/DARK/aspect-fix/ensemble）
│   ├── api_server_vertebra.py       # FastAPI 服務 (port 8001)
│   ├── regenerate_splits.py         # 從 Images/ 重生 train/val split
│   ├── compare_annotations.py       # QA：手標 vs 模型對照（overlay + 統計）
│   └── eval_ensemble.py             # 5-fold ensemble 評估
│
├── 標註工具
│   ├── spinal-annotation-web.html   # 椎體 4 角點標註工具
│   └── pkl-contour-editor.html      # 外部 mask → 終板輪廓點編輯器
│
├── 部署
│   ├── Dockerfile / docker-compose.yml   # Synology NAS 部署
│   └── spine_inference_colab.ipynb       # Colab 推論 (Gradio)
│
├── 交班文件
│   ├── HANDOFF_2026-06-11.md         # 最新（推論優化鏈 + ensemble + Colab 步驟）
│   ├── HANDOFF_2026-06-06.md         # V3.4 上線 + aspect-fix
│   └── HANDOFF_2026-06-05.md
│
└── 資料夾
    ├── Images/                      # 訓練影像 + 標註 JSON
    └── endplate_training_data/      # 訓練數據 split
```

---

## JSON 標註格式

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
| 推論沒看到 `Ensemble 啟用` | sidecar 或 fold3 沒放對 → 確認三個檔在同一層 |
| 找不到標註檔案 | 確認 JSON 放在 `Images/` 資料夾 |
| CUDA out of memory | `--batch-size` 改 1 或 2 |
| ensemble 推論變慢 | 正常：2 模型 × TTA 5 變體 = 10 次 forward/張，約慢 2× |
| 5-fold 訓練半夜中斷 | 電腦睡眠所致 → 先 `powercfg /change standby-timeout-ac 0` |
| V2 舊 checkpoint | `inference_vertebra.py` 自動偵測 V2/V3 架構 |

---

## 版本歷史（精簡版）

| 版本 | 重點 | overall_mean |
|------|------|--------------|
| **生產 (06-11)** | baseline + fold3 ensemble + 全後處理鏈 | **97.6 px** |
| 06-10 | 推論優化鏈：命名錨定 / TTA / DARK / reliability | 103.5 px |
| V3.4 (06-06) | keep-aspect resize + aspect-fix v2 | 144 px |
| V3.3 ~ V3.6 | 大改架構實驗（全失敗，已退回） | 321~508 px |
| V3.2 | RadImageNet backbone + 兩階段解凍 | 180 px |
| V3.0 | ResNet50 + UNet decoder + 多通道 heatmap | — |
| V2.0 | 椎體 4 角點 + 回歸分支 | — |
| V1.0 | 終板標註 + U-Net 分割 | — |

完整逐步紀錄與每個技巧的「為什麼」，見 [HANDOFF_2026-06-11.md](HANDOFF_2026-06-11.md)。
