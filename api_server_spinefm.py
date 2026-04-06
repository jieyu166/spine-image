#!/usr/bin/env python3
"""
SpineFM API 伺服器
SpineFM API Server

啟動方式:
    python api_server_spinefm.py
    瀏覽器開啟 http://localhost:8002
"""

import os
import base64
import tempfile
import numpy as np
import cv2
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from inference_spinefm import SpineFMInference

# ==================== FastAPI App ====================
app = FastAPI(
    title="SpineFM 脊椎椎體偵測 API",
    description="Vertebra Detection with SpineFM (Medical-SAM-Adaptor)",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = None
SCRIPT_DIR = Path(__file__).parent

CORNER_COLORS_BGR = [
    (0, 200, 0),      # anteriorSuperior - green
    (200, 100, 0),    # posteriorSuperior - blue
    (0, 0, 220),      # posteriorInferior - red
    (0, 220, 220),    # anteriorInferior - yellow
]
CORNER_NAMES = ['anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior']


def draw_result_image(image_rgb, vertebrae):
    """在影像上繪製角點和椎體輪廓"""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    radius = max(4, min(w, h) // 180)
    thickness = max(1, min(w, h) // 500)
    font_scale = max(0.4, min(w, h) / 2200)
    font_thick = max(1, int(font_scale * 2))

    for v in vertebrae:
        pts = v['points']
        corners_xy = []
        for j, cn in enumerate(CORNER_NAMES):
            if cn not in pts:
                continue
            x, y = int(pts[cn]['x']), int(pts[cn]['y'])
            corners_xy.append((x, y))
            cv2.circle(img, (x, y), radius, CORNER_COLORS_BGR[j], -1)

        if len(corners_xy) >= 2:
            for k in range(len(corners_xy)):
                cv2.line(img, corners_xy[k], corners_xy[(k + 1) % len(corners_xy)],
                         (255, 255, 255), thickness)

        if corners_xy:
            cx = int(np.mean([p[0] for p in corners_xy]))
            cy = int(np.mean([p[1] for p in corners_xy]))
            label = v['name']
            grade = v.get('genantGrade', 0)
            if grade > 0:
                label += f' [G{grade}]'
                cv2.putText(img, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 255), font_thick)
            else:
                cv2.putText(img, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 255), font_thick)

    return img


def draw_mask_overlay(image_rgb, heatmap):
    """Mask overlay"""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_norm = (hm * 255).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_HOT)
    return cv2.addWeighted(img, 0.6, hm_color, 0.4, 0)


def encode_image_base64(bgr_img):
    _, buf = cv2.imencode('.png', bgr_img)
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/png;base64,{b64}"


# ==================== Endpoints ====================
@app.on_event("startup")
async def load_model():
    global analyzer
    weights_dir = SCRIPT_DIR / "spinefm_weights"
    if not weights_dir.exists():
        print(f"WARNING: SpineFM weights not found: {weights_dir}")
        print("  Run setup_spinefm.bat to download weights.")
        return
    try:
        analyzer = SpineFMInference(
            weights_path=str(weights_dir),
            dataset='NHANESII',
            device='auto',
        )
    except Exception as e:
        print(f"ERROR loading SpineFM: {e}")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = SCRIPT_DIR / "spine-inference-web.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>spine-inference-web.html not found</h1>", status_code=404)


@app.get("/health")
async def health():
    return {
        "status": "healthy" if analyzer else "model not loaded",
        "model_loaded": analyzer is not None,
        "backend": "SpineFM",
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    spine_type: str = Form("L"),
):
    """上傳影像，回傳 SpineFM 椎體偵測結果 (v2.3 JSON 格式)"""
    if analyzer is None:
        raise HTTPException(503, "SpineFM model not loaded. Run setup_spinefm.bat first.")

    if spine_type not in ('L', 'C'):
        raise HTTPException(400, "spine_type must be 'L' or 'C'")

    tmp_path = None
    try:
        contents = await file.read()
        suffix = Path(file.filename or 'image.png').suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = analyzer.predict(tmp_path, spine_type=spine_type)

        result_bgr = draw_result_image(result['original_image'], result['vertebrae'])
        heatmap_bgr = draw_mask_overlay(result['original_image'], result['heatmap'])
        original_bgr = cv2.cvtColor(result['original_image'], cv2.COLOR_RGB2BGR)

        response = {
            "success": True,
            "filename": file.filename,
            "spine_type": spine_type,
            "backend": "SpineFM",
            "image_info": result['image_info'],
            "predicted_count": result['predicted_count'],
            "count_confidence": 1.0,
            "vertebrae": result['vertebrae'],
            "discs": result['discs'],
            "original_image": encode_image_base64(original_bgr),
            "result_image": encode_image_base64(result_bgr),
            "heatmap_image": encode_image_base64(heatmap_bgr),
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/model_info")
async def model_info():
    return {
        "model_loaded": analyzer is not None,
        "backend": "SpineFM",
        "dataset": "NHANESII",
    }


# ==================== Main ====================
def main():
    print("=" * 50)
    print("  SpineFM Spine Detection API Server")
    print("  http://localhost:8002")
    print("  API docs: http://localhost:8002/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")


if __name__ == "__main__":
    main()
