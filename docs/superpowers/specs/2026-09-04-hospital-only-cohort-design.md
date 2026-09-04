# Hospital-only training cohort experiment design

Date: 2026-09-04

## Background and decision

- 醫師目前無法投入新標註；`planning/nhanes*` 的 513 張 NHANES 影像雖已自動轉成 v2.3 角點 JSON，但使用者於 2026-09-04 裁定 NHANES 與醫院影像域差太大，**暫停 NHANES 標註與預訓練**。
- 08-02 的 OOF 報告顯示 13 筆 NHANES 的 count exact 僅 53.85%，醫師評語「影像品質差幾乎不可判讀」。
- README 記錄的 ensemble 教訓（弱成員是拖油瓶）同樣適用於訓練資料：域外樣本可能拉低醫院影像的表現。

## Goal

在**零新標註**前提下回答一個可量化問題：把 13 筆 NHANES 從訓練集拿掉，只用 48 筆醫院影像（L 40 / C 8，41 個病人群）訓練，醫院影像的 OOF 指標是否優於 08-02 的 61 筆版本？

## Method

- Run 目錄：`endplate_training_data/runs/sixpoint_hospital_only_cv5_20260904_211040/`，結構與 08-02 run 一致（`Images` junction、`code_snapshot/`、`annotations/`、`run_manifest.json`）。
- 資料：直接從 08-02 run 的 `annotations/{train,val}_annotations.json` 過濾掉 `image_path` 含 `nhanes`（不分大小寫）的條目；train 49→39、val 12→9。不改任何標註內容。
- 訓練：`train_vertebra_model_cv.py --n-folds 5 --epochs 30 --unfreeze-epoch 5`，seed 42，patient-grouped folds，RadImageNet backbone（絕對路徑，避免 08-02 的 ImageNet fallback 踩雷）。
- 評估：`eval_oof.py --run-dir .`，TTA 開；再以 `compare_to_0802.py` 對**同一批 48 筆醫院影像**做配對比較（全體、L-spine、C-spine；per-case corner mean delta）。
- 比較採每例平均再取平均（per-case），與 eval_oof 的 pooled landmark 數字略有差異，但新舊兩邊口徑一致。

## Constraints

- 不動生產模型、sidecar、allowlist、標註檔。
- fold 成員與 08-02 不同（樣本數不同，greedy group k-fold 重新分配），因此比較的是「同一批 48 筆在各自 OOF 下的表現」，不是逐 fold 對照。

## Decision rule

- L-spine corner mean 與 count exact 同時改善，且 per-case 惡化 >10px 的例數明顯少於改善例數 → 後續訓練集預設排除 NHANES。
- 差異在雜訊範圍內 → NHANES 去留不重要，資料瓶頸仍在醫院影像數量；下一步改做「模型預標註 + 醫師只審不標」。
- 明顯變差 → NHANES 雖有域差仍提供有用訊號，保留。

## Non-goals

- 不重跑 08-02 run，不改 fold 演算法，不部署。
