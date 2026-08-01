# 脊椎標註病例隔離與影像－JSON 批次審查設計

日期：2026-08-02

## 目標

本次工作包含兩個彼此相依的安全性改進：

1. 強化 `spinal-annotation-web.html` 的病例切換流程，避免上一張影像的點位、選取或拖曳狀態污染下一張影像，並阻止 JSON 套用到錯誤影像。
2. 擴充 `annotation-viewer.html`，讓醫師一次選取整個 `Images` 資料夾，逐例查看影像與 JSON overlay，判定「相符」或「不相符」，並保存可續審、可稽核的獨立結果。

原始影像與醫師標註 JSON 都是唯讀來源。批次審查工具不得修改、重新命名或刪除來源檔案。

## 不在本次範圍

- 不自動修正錯配、偏移或錯誤標註。
- 不重新訓練模型或重建訓練集。
- 不以模糊病歷號自動配對不同檔名。
- 不上傳影像、JSON、hash 或審查結果到網路服務。
- 不以自動檢查結果取代醫師的最終「相符／不相符」判定。

## 實作方式比較

### A. 擴充既有 `annotation-viewer.html`（採用）

沿用已存在的影像解碼、JSON 正規化、overlay、縮放、拖曳與資訊側欄，在同一個 HTML 內加入批次掃描、配對及審查控制層。

優點：

- 醫師仍以 Chrome 直接開啟單一 HTML，不需啟動伺服器。
- 單張查看與批次審查共用同一套座標繪圖邏輯。
- 不增加 Python 或網路服務的操作負擔。

限制：

- 瀏覽器基於安全限制只能下載審查結果，不能直接寫回選取的資料夾。
- `webkitdirectory` 與本機檔案流程以 Chrome／Edge 為主要支援環境。

### B. 新增獨立 `pair-review.html`

不採用。介面可較簡單，但會複製 viewer 的影像與標註繪圖邏輯，之後兩套容易出現不同座標行為。

### C. Python 本機伺服器加網頁

不採用。雖可直接寫檔並執行更完整的檔案操作，但醫師需維護 Python 環境與啟動服務，不符合目前單檔 HTML 工作流。

## 第一部分：`spinal-annotation-web.html` 病例隔離

### 病例狀態

新增單一病例重設函式，完整清除：

- `vertebrae`
- `currentPoints`
- `annotationFinished`
- `selectedVertebrae`
- `dragInfo`
- `groupDrag`
- `selectBox`
- `isDragging`
- `wasDragging`
- `_justDragged`
- 舊影像的 contrast／processed image 狀態
- 舊影像檔名、hash 及其他 identity metadata

`clearAll()` 與成功載入新影像都必須呼叫同一個病例重設函式，避免兩條路徑清除範圍不同。函式接受重設原因：醫師按下 `clearAll()` 後將病例標成 dirty；成功切換到尚未標註的新影像則將 dirty 設為 `false`。

### 新影像載入

1. 選擇或貼上新影像。
2. 若目前病例有尚未匯出的修改，先提示醫師匯出或確認捨棄。
3. 先在背景完成 FileReader 與影像解碼。
4. 解碼失敗時保留目前病例，不改動任何狀態。
5. 解碼成功後才原子化重設病例、替換 `originalImage`、設定新 canvas 尺寸並重設 zoom/pan。

以 `annotationDirty` 追蹤是否有未匯出變更。新增、移動、刪除、清除或匯入點位後設為 `true`；成功匯出後設為 `false`。已成功匯出的病例在載入下一張時不需重複警告。

載入相同檔名後，檔案 input 必須清空 value，使使用者能再次選擇同一檔案。

### 影像 identity

每次載入影像時保存：

- `fileName`
- `width`、`height`（natural dimensions）
- 檔案大小與最後修改時間（若 File/Blob 有提供）
- 影像原始位元組的 SHA-256（瀏覽器 Web Crypto 可用時）

貼上的 Blob 若沒有檔名，使用可辨識的 clipboard label，仍計算尺寸及 hash。

### JSON 匯入驗證

JSON 套用前必須完成下列檢查，任何高風險不符都停止匯入，且不得清除或修改目前標註：

1. 目前已有成功解碼的影像。
2. `imageInfo.width/height` 與目前影像 natural dimensions 完全一致。
3. 新格式 JSON 若有 `imageInfo.fileName`，必須與目前影像檔名相同。
4. 新格式 JSON 若有 `imageInfo.sha256`，必須與目前影像 hash 相同。
5. 舊格式 JSON 沒有檔名或 hash 時，使用匯入 JSON 的檔名 stem 與影像 stem，加上尺寸共同檢查；通過後可匯入，但顯示 legacy metadata 警告。

一般 JSON 匯入與 SpineFM／AI 匯入共用同一個驗證函式。尺寸不符時不再提供「仍要繼續」選項。

帶有內嵌原圖的 JSON 先解碼內嵌影像，再以該影像完成相同尺寸檢查；不得把點位套到先前顯示的影像。

### JSON 匯出

匯出改為非同步流程，等待影像 SHA-256 計算完成。`imageInfo` 至少包含：

```json
{
  "width": 1200,
  "height": 1440,
  "fileName": "80339761.png",
  "sha256": "<64 hex characters or null>"
}
```

下載檔名使用影像完整 stem，例如 `80339761.png` 匯出為 `80339761.json`；無可靠檔名時才退回 timestamp 名稱。

## 第二部分：`annotation-viewer.html` 批次審查

### 元件邊界

在既有單檔 HTML 內加入三個清楚分工的單元：

- `PairScanner`：接收資料夾 FileList，分類正式檔案、排除檔案、精確配對、待配對及資料錯誤。
- `ReviewStore`：保存審查狀態、localStorage 續審、stale 判定、JSON/CSV 匯入匯出。
- `BatchReviewController`：控制目前案例、篩選、上一例／下一例、醫師判定及把 pair 載入既有 viewer。

現有影像解碼、JSON normalizer 與 overlay renderer 維持單一來源；批次控制層不得另寫一套座標轉換。

### 資料夾掃描

使用 `<input type="file" webkitdirectory multiple>` 讓醫師選擇整個 `Images` 資料夾。掃描遞迴包含子資料夾，但配對的第一優先範圍是同一相對資料夾。

正式配對使用完整 stem；以下都屬於 ID 的一部分，不可移除或互換：

- `-1`、`-2`
- `_1`、`_2`
- `C`、`C0`
- `L`、`L0`
- 其他不是排除規則的尾碼

例如：

```text
81312903-2.json <-> 81312903-2.png
81312903_2.json <-> 81312903_2.png
```

`81312903-2` 與 `81312903_2` 不自動視為相同。

### 排除規則

檔案 stem 結尾為下列值時，不分大小寫排除於正式審查清單之外：

- `samp`
- `ai`
- `model`

排除檔案不會消失：摘要需顯示各類數量，並可展開查看完整相對路徑。

### 自動配對規則

1. JSON 與影像必須位於同一相對資料夾且完整 stem 相同。
2. 支援的影像格式沿用 viewer 能實際解碼的格式。
3. 同 stem 只有一張影像時建立 exact pair。
4. 同 stem 同時存在多種影像格式時列為 conflict，不擅自選擇。
5. 正式影像沒有 JSON、正式 JSON 沒有影像時都列為資料錯誤。
6. 同一影像不得被多個正式 JSON 自動重複使用。

### 手動配對

無 exact pair 的項目進入「待配對」清單。工具先顯示同資料夾內尚未使用的影像或 JSON 候選，但不以病歷號距離或模糊字串自動決定。

醫師可手動建立配對，例如：

```text
80145593.json <-> 8014559.png
```

手動配對必須記錄兩個完整相對路徑、建立時間與 `manual: true`，並防止候選檔在沒有額外確認時被重複使用。

### 自動檢查

每個 pair 載入時顯示但不取代醫師判定的檢查包括：

- JSON 是否可解析。
- 是否存在 `vertebrae` 與可用點位。
- 影像實際尺寸與 JSON `imageInfo` 是否一致。
- 點位是否超出影像範圍。
- 新格式檔名與 SHA-256 是否一致。
- 椎體數與有效點位數。
- 是否為 legacy JSON（缺少檔名或 hash）。

警告分為高風險紅色與資訊性黃色。即使自動檢查全部通過，案例仍維持 pending，必須由醫師目視 overlay 後判定。

## 醫師操作流程

### 啟動與進度

1. 開啟 `annotation-viewer.html`。
2. 點選「選擇 Images 資料夾」。
3. 工具顯示正式配對、待配對、資料錯誤與排除摘要。
4. 預設開啟第一個未審查 exact pair。

頂端顯示：

- 已審查／總數
- 相符數
- 不相符數
- 待配對數
- 資料錯誤數
- 排除數

篩選包括「未審查」、「不相符」、「自動警告」、「待配對」及「全部」。

### 單例審查

中央畫面沿用既有影像與 JSON overlay，可縮放、拖曳及切換點、線、標籤。右側顯示影像與 JSON 完整相對路徑、尺寸、hash、椎體數、點位數及自動警告。

操作：

- 「相符」：直接保存並前往下一個未審查案例。
- 「不相符」：必須選擇原因，可加備註，保存後前往下一例。
- 「稍後判定」：保持 pending，前往下一例。
- 「上一例／下一例」：不改變目前判定。

不相符原因固定為：

- `image_patient_mismatch`：影像／病人錯配
- `crop_or_dimension_mismatch`：裁切或尺寸不符
- `annotation_global_offset`：標註整體偏移
- `json_invalid`：JSON 損壞／內容異常
- `image_unreadable`：影像不可判讀
- `other`：其他（必填備註）

### 快捷鍵

快捷鍵只在焦點不位於輸入框時啟用，畫面同時顯示提示：

- `M`：相符
- `X`：開啟不相符原因
- `J`／右方向鍵：下一例
- `K`／左方向鍵：上一例
- `S`：稍後判定

快捷鍵不得跳過「不相符原因必選」驗證。

## 審查資料與續審

### Pair identity 與 stale 判定

Pair ID 使用 image relative path 與 JSON relative path，不使用單純病歷號：

```text
202607/81312903-2.png::202607/81312903-2.json
```

每個來源檔保存 file signature：relative path、size、lastModified 及 SHA-256。重新選擇資料夾後：

- pair 路徑與兩邊 signature 均相同：沿用既有判定。
- pair 路徑相同但任一 signature 改變：標記 stale 並回到 pending。
- 新 pair：新增 pending。
- 已不存在的 pair：保留在匯出歷史，但不列入目前待審總數。

### localStorage dataset key

Dataset key 由根資料夾名稱與正式檔案相對路徑清單產生，不納入 size 或 lastModified，使內容修改後仍能找到原審查紀錄並進行 stale 比對。

若新增或移除檔案造成路徑清單改變，工具以根資料夾名稱尋找最近一次紀錄，僅搬移 pair 路徑相同且 signature 未變的判定；若存在多個同名資料集候選，要求使用者選擇或匯入先前的 `pair_review.json`，不得靜默合併。

localStorage 只保存文字 metadata 與審查結果，不保存影像或完整 JSON。若 quota 或瀏覽器權限造成保存失敗，顯示持續警告並要求立即匯出 JSON。

### `pair_review.json`

輸出至少包含：

```json
{
  "version": "1.0",
  "dataset": {
    "rootName": "Images",
    "datasetKey": "sha256:3d9f6f334f5b5c8dd9402a53ac756a62a58a9fce73bb2a8785f2db4f8f5b6170",
    "scannedAt": "2026-08-02T00:00:00.000Z"
  },
  "rules": {
    "excludedSuffixes": ["samp", "ai", "model"],
    "exactStemOnly": true
  },
  "summary": {},
  "manualPairs": [],
  "reviews": []
}
```

每筆 review 包含 pair ID、相對路徑、兩邊 signature、auto checks、`pending/match/mismatch`、reason、note、reviewedAt 及 stale 狀態。

重新匯入 manifest 時，以 pair ID 與 signatures 合併；不得只按陣列位置覆蓋。

### `pair_review.csv`

CSV 每個 pair 一列，包含路徑、尺寸、hash、auto warnings、判定、原因、備註、時間、manual 及 stale。輸出使用 UTF-8 BOM，方便 Windows Excel 正確顯示繁體中文。

## 錯誤處理

- 個別 JSON 解析失敗或影像解碼失敗：保留案例於清單並顯示錯誤，不中止整批。
- Folder selection 取消：保留目前工作階段。
- Pair 載入的非同步結果必須以 request token 防止快速切換時舊案例覆蓋新案例。
- SHA-256 尚在計算時顯示進度；hash 失敗時記錄原因，不假裝相符。
- 匯出前重新計算 summary，避免畫面統計與 JSON/CSV 不一致。
- 匯入不支援的 manifest version 時停止並顯示明確訊息。
- 原始檔案在審查期間改變時，需重新選取資料夾才能讀到新 File metadata；重新選取後依 stale 規則處理。

## 隱私與安全

- 所有掃描、hash、overlay、localStorage 與匯出都在本機瀏覽器執行。
- 不加入 analytics、外部字型、CDN、網路 API 或遠端錯誤回報。
- UI 顯示相對路徑，不把完整 Windows 使用者路徑寫入 manifest。
- 審查頁面不提供刪除、覆寫或重新命名來源檔案的功能。

## 測試策略

先建立會在現行版本失敗的測試，再寫正式程式碼。

### `spinal-annotation-web.html`

- A 影像已有完成點位，成功載入不同尺寸 B 後，所有 A 病例與互動狀態均為空。
- B 解碼失敗時，A 的影像與標註仍保留。
- `clearAll()` 同時清除點位、選取、拖曳與框選狀態。
- zoom/pan 後新增與拖曳點位可 round-trip 回原始影像座標。
- JSON 尺寸、檔名或 SHA-256 不符時拒絕匯入且不改變現有點位。
- legacy JSON 只有在檔名 stem 與尺寸均相符時可匯入。
- 新匯出 JSON 包含 image identity，下載檔名來自影像 stem。

### `PairScanner`

- `-2` 與 `_2` 不會互配。
- `C`、`C0`、`L0` 會正常納入且完整同名配對。
- 結尾 `samp`、`ai`、`model` 必定排除，不分大小寫。
- 缺 JSON、缺影像、多影像格式及重複使用都進入正確問題分類。
- 手動配對保存完整相對路徑且不會靜默重複使用影像。

### `ReviewStore`

- 判定後可由 localStorage 還原並跳到第一個 pending。
- 同 pair signature 改變後舊判定變 stale/pending。
- 路徑相同且 signature 未變時可安全搬移既有判定。
- manifest 匯入以 pair ID/signature 合併，不按陣列順序。
- JSON 與 CSV summary、狀態及原因一致。
- `other` 沒有備註時不能保存 mismatch。

### 整合與人工驗證

- Node VM 或等效瀏覽器腳本測試直接執行 HTML 內實際 JavaScript，不以 grep 原始碼代替行為測試。
- 執行既有 Python pytest，確認資料管線沒有回歸。
- 在 Chrome 以兩組不同尺寸真實影像驗證病例切換、zoom/pan、JSON 拒絕與匯出 round-trip。
- 以目前 `Images` 資料夾驗證 exact pairs、排除清單、待配對清單、續審、stale、JSON 與 CSV 匯出。

## 驗收條件

1. 成功載入下一張影像後，上一病例的任何點位或選取狀態都不可能被匯出到新病例。
2. C00172 類型的原圖／裁切圖尺寸錯配會在標註匯入時被阻止，在審查工具中顯示高風險警告。
3. 正式檔名所有 suffix 均保留，只有 `samp/ai/model` 被排除。
4. 每張正式影像若沒有 JSON 都明確列出；工具不自行猜測錯配。
5. 醫師可完成 match/mismatch 判定、必選原因、備註、上一例／下一例及篩選。
6. 關閉頁面後可續審；來源檔變更後既有判定會失效而不是被沿用。
7. 匯出的 JSON/CSV 可追溯每個 pair 的路徑、hash、判定、原因與時間。
8. 工具全程不修改來源影像或標註 JSON，且不進行任何網路傳輸。
