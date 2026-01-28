# Spine X-ray Web Analyzer

此專案提供 Spine X-ray 上傳與分析的網頁介面骨架，供後續整合椎體辨識、終板偵測與 disc 量測模型。

## 快速開始

```bash
python server.py
```

瀏覽 `http://localhost:5000` 上傳影像並查看分析流程。

## 後續整合建議

- 將 AI 模型推論結果整合至 `analyze_spine_image`。
- 為每個椎體輸出 endplate 與 disc 量測結果。
- 補上 spondylolisthesis / retrolisthesis 判斷邏輯。
