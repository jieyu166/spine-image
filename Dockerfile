# ======================================================
# Spine Vertebra Detection - Docker Image
# 用於 Synology NAS / Linux Server 部署
# ======================================================

FROM python:3.11-slim

# 系統依賴 (OpenCV 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先安裝 Python 套件 (利用 Docker cache)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 複製應用程式碼
COPY api_server_vertebra.py .
COPY inference_vertebra.py .
COPY train_vertebra_model.py .
COPY spine-inference-web.html .
COPY spinal-annotation-web.html .

# 模型檔案透過 docker-compose volume 掛載 (不需 COPY 進 image)

EXPOSE 8001

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["python", "api_server_vertebra.py"]
