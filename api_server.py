#!/usr/bin/env python3
"""
脊椎終板檢測 - API 伺服器
Spine Endplate Detection - API Server

提供REST API進行推理
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np
import cv2
import pydicom
import io
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')  # 非GUI後端
import matplotlib.pyplot as plt
from inference import SpineAnalyzer

# 創建FastAPI應用
app = FastAPI(
    title="脊椎終板檢測API",
    description="使用深度學習模型檢測脊椎終板位置和計算角度",
    version="2.0"
)

# CORS設置（允許網頁調用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局變數：模型分析器
analyzer = None

@app.on_event("startup")
async def load_model():
    """啟動時載入模型"""
    global analyzer
    model_path = "best_endplate_model.pth"
    
    if not Path(model_path).exists():
        print(f"⚠️ 模型檔案不存在: {model_path}")
        print("   請先訓練模型或提供模型檔案")
        return
    
    print(f"🔧 載入模型: {model_path}")
    analyzer = SpineAnalyzer(model_path, device='auto')
    print("✅ 模型載入完成")

@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "脊椎終板檢測API",
        "version": "2.0",
        "status": "running" if analyzer is not None else "model not loaded",
        "endpoints": {
            "analyze": "/api/analyze (POST)",
            "health": "/health (GET)",
            "docs": "/docs (Swagger UI)"
        }
    }

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy" if analyzer is not None else "model not loaded",
        "model_loaded": analyzer is not None,
        "device": str(analyzer.device) if analyzer else "N/A"
    }

@app.post("/api/analyze")
async def analyze_spine(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    return_image: bool = True
):
    """
    分析脊椎影像
    
    Args:
        file: DICOM或影像檔案
        threshold: 終板檢測閾值 (0-1)
        return_image: 是否返回視覺化影像
    
    Returns:
        JSON結果，包含終板位置和角度
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="模型未載入")
    
    try:
        # 讀取上傳的檔案
        contents = await file.read()
        
        # 儲存到臨時檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        # 執行推理
        results = analyzer.predict(tmp_path)
        
        # 提取終板
        endplate_lines = analyzer.extract_endplates(
            results['endplate_mask'],
            threshold=threshold
        )
        
        # 計算角度
        angles = analyzer.calculate_angles(endplate_lines)
        
        # 準備返回結果
        response = {
            "success": True,
            "filename": file.filename,
            "num_endplates": len(endplate_lines),
            "num_angles": len(angles),
            "endplate_lines": [
                {
                    "x1": int(l[0]),
                    "y1": int(l[1]),
                    "x2": int(l[2]),
                    "y2": int(l[3])
                }
                for l in endplate_lines
            ],
            "angles": angles
        }
        
        # 生成視覺化影像
        if return_image:
            # 創建簡單視覺化
            image = results['image']
            result_img = image.copy()
            
            # 繪製終板線段
            for i, line in enumerate(endplate_lines):
                x1, y1, x2, y2 = line
                cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # 標記角度
                if i < len(angles):
                    angle_info = angles[i]
                    mid_y = (y1 + y2) // 2
                    cv2.putText(
                        result_img,
                        f"#{angle_info['level']}: {angle_info['angle']}°",
                        (10, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            
            # 轉為base64
            import base64
            _, buffer = cv2.imencode('.png', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response['result_image'] = f"data:image/png;base64,{img_base64}"
        
        # 清理臨時檔案
        Path(tmp_path).unlink(missing_ok=True)
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失敗: {str(e)}")

@app.post("/api/batch_analyze")
async def batch_analyze(files: list[UploadFile] = File(...)):
    """批次分析多個影像"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="模型未載入")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            
            # 執行推理
            pred_results = analyzer.predict(tmp_path)
            endplate_lines = analyzer.extract_endplates(pred_results['endplate_mask'])
            angles = analyzer.calculate_angles(endplate_lines)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "num_endplates": len(endplate_lines),
                "num_angles": len(angles),
                "angles": angles
            })
            
            Path(tmp_path).unlink(missing_ok=True)
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "total_files": len(files),
        "results": results
    })

@app.get("/api/model_info")
async def model_info():
    """獲取模型資訊"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="模型未載入")
    
    return {
        "model_loaded": True,
        "device": str(analyzer.device),
        "config": analyzer.config
    }

def main():
    """啟動伺服器"""
    print("🚀 啟動脊椎終板檢測API伺服器...")
    print("📝 API文檔: http://localhost:8000/docs")
    print("🔍 健康檢查: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()

