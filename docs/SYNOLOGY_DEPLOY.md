# Synology NAS 部署指南 - 脊椎椎體偵測 AI

## 前置需求

| 項目 | 最低需求 | 建議 |
|------|---------|------|
| NAS 型號 | DS220+ 以上 (x86 CPU) | DS723+, DS923+ |
| RAM | 4 GB | 8 GB+ |
| 儲存空間 | 3 GB (Docker image) | 5 GB+ |
| DSM 版本 | 7.0+ | 7.2+ |
| 套件 | Container Manager | - |

> ⚠️ **不支援 ARM 架構 NAS** (如 DS120j, DS220j)，因為 PyTorch 不提供 ARM 版本。

---

## 步驟 1：安裝 Container Manager

1. 開啟 DSM → **套件中心**
2. 搜尋 **Container Manager** (舊版叫 Docker)
3. 安裝並啟動

---

## 步驟 2：準備部署檔案

在電腦上建立部署資料夾，只包含推理所需的檔案：

```
spine-deploy/
├── Dockerfile
├── docker-compose.yml
├── requirements-docker.txt
├── .dockerignore
├── api_server_vertebra.py        ← API 伺服器
├── inference_vertebra.py          ← 推理引擎
├── train_vertebra_model.py        ← 模型定義 (推理時需 import)
├── spine-inference-web.html       ← 前端 UI
├── spinal-annotation-web.html     ← 標註編輯器
└── best_vertebra_model.pth        ← 訓練好的模型 (~140MB)
```

### 從專案目錄複製檔案

```powershell
# 在 Windows 上執行
mkdir C:\spine-deploy
cd "C:\Users\jai16\OneDrive\00 放射科\0筆記\Radiology\0. Inbox\Spine"

copy Dockerfile               C:\spine-deploy\
copy docker-compose.yml        C:\spine-deploy\
copy requirements-docker.txt   C:\spine-deploy\
copy .dockerignore             C:\spine-deploy\
copy api_server_vertebra.py    C:\spine-deploy\
copy inference_vertebra.py     C:\spine-deploy\
copy train_vertebra_model.py   C:\spine-deploy\
copy spine-inference-web.html  C:\spine-deploy\
copy spinal-annotation-web.html C:\spine-deploy\
copy best_vertebra_model.pth   C:\spine-deploy\
```

---

## 步驟 3：上傳到 NAS

### 方法 A：透過 File Station（最簡單）
1. 開啟 DSM → **File Station**
2. 建立共用資料夾或在 `docker/` 下建立子資料夾：`/docker/spine-deploy/`
3. 將 `C:\spine-deploy\` 裡所有檔案上傳到 `/docker/spine-deploy/`

### 方法 B：透過 SMB 網路磁碟
```powershell
# 對映 NAS 共用資料夾
net use Z: \\你的NAS_IP\docker

# 複製檔案
xcopy C:\spine-deploy\* Z:\spine-deploy\ /E
```

---

## 步驟 4：建置 Docker Image

### 方法 A：透過 SSH（建議）

1. **啟用 SSH**：DSM → 控制台 → 終端機和 SNMP → 勾選「啟用 SSH」
2. SSH 連線：
```bash
ssh admin@你的NAS_IP
```
3. 建置 image：
```bash
cd /volume1/docker/spine-deploy
sudo docker compose build
```
> 首次建置約需 **10-20 分鐘**（下載 Python + PyTorch CPU ~1.5GB）

4. 啟動服務：
```bash
sudo docker compose up -d
```

### 方法 B：透過 Container Manager UI

1. 開啟 **Container Manager** → **專案** → **新增**
2. 專案名稱：`spine-vertebra`
3. 路徑：選擇 `/docker/spine-deploy`
4. 來源選擇：`docker-compose.yml`
5. 點擊**建置**

---

## 步驟 5：確認服務運行

### 檢查容器狀態
```bash
sudo docker compose ps
# 應顯示 spine-vertebra-api   Up (healthy)
```

### 測試 API
```bash
curl http://localhost:8001/health
# 回傳: {"status":"healthy","model_loaded":true}
```

### 開啟前端
在同一個區域網路的任何電腦上：
```
http://你的NAS_IP:8001
```

---

## 步驟 6：設定防火牆（選用）

如果 NAS 有啟用防火牆：

1. DSM → 控制台 → 安全性 → 防火牆
2. 新增規則：允許 Port **8001** (TCP)
3. 建議只允許內網 IP 範圍（如 `192.168.1.0/24`）

---

## 日常管理

### 更新模型
重新訓練後，只需替換模型檔：
```bash
# 從電腦複製新模型到 NAS
copy best_vertebra_model.pth \\NAS_IP\docker\spine-deploy\

# SSH 到 NAS 重啟服務
ssh admin@NAS_IP
cd /volume1/docker/spine-deploy
sudo docker compose restart
```

### 查看日誌
```bash
sudo docker compose logs -f --tail 50
```

### 停止/啟動
```bash
sudo docker compose stop     # 停止
sudo docker compose start    # 啟動
sudo docker compose down     # 完全移除容器
```

### 更新程式碼
```bash
# 覆蓋檔案後重建
sudo docker compose build --no-cache
sudo docker compose up -d
```

---

## 效能預估

| NAS CPU | 單張推理時間 | 說明 |
|---------|------------|------|
| Intel Celeron J4125 | ~10-15 秒 | DS220+, DS420+ |
| Intel Celeron J4025 | ~12-18 秒 | DS220+ |
| AMD Ryzen R1600 | ~6-10 秒 | DS1621+ |
| AMD Ryzen V1500B | ~5-8 秒 | DS723+, DS923+ |
| Intel Atom C3538 | ~15-25 秒 | DS1019+ |

> 推理時間主要受限於 CPU。第一次推理會稍慢（模型載入），後續推理會快一些。

---

## 疑難排解

### 建置失敗：記憶體不足
```bash
# 增加 swap
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo mkswap /swapfile
sudo swapon /swapfile
# 重新建置
sudo docker compose build
```

### 容器啟動但無法存取
1. 確認 Port 8001 沒有被其他服務佔用
2. 確認 `best_vertebra_model.pth` 在正確位置
3. 查看日誌：`sudo docker compose logs`

### 推理時出現 OOM (Out of Memory)
- 確保 NAS 有至少 4GB 可用 RAM
- 在 `docker-compose.yml` 調整記憶體限制
- 考慮關閉其他高記憶體套件 (如 Surveillance Station)

### 無法連線
- 確認在同一個區域網路內
- 試試 `http://NAS_IP:8001/health` 確認 API 是否回應
- 檢查 NAS 防火牆設定
