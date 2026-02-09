@echo off
chcp 65001 >nul
echo ========================================
echo 脊椎椎體頂點檢測 - 推理 V3
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

echo 請選擇推理模式:
echo.
echo 1. 分析單一影像檔案 (DICOM/PNG/JPG)
echo 2. 批次處理資料夾
echo 3. 使用範例影像測試
echo.

set /p mode="請輸入選項 (1/2/3): "

echo.
set /p spine_type="脊椎類型 (L=腰椎, C=頸椎) [預設L]: "
if "%spine_type%"=="" set spine_type=L

if "%mode%"=="1" (
    echo.
    set /p input_file="請輸入影像檔案路徑: "
    python inference_vertebra.py --model best_vertebra_model.pth --input "%input_file%" --spine-type %spine_type%
) else if "%mode%"=="2" (
    echo.
    set /p input_dir="請輸入資料夾路徑: "
    python inference_vertebra.py --model best_vertebra_model.pth --input "%input_dir%" --spine-type %spine_type% --no-viz
) else if "%mode%"=="3" (
    echo.
    echo 使用 Images 資料夾中的範例影像...
    python inference_vertebra.py --model best_vertebra_model.pth --input "Images" --spine-type %spine_type%
) else (
    echo.
    echo ❌ 無效的選項
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 推理完成！
echo ========================================
echo.
echo 結果已儲存在 inference_results\ 資料夾
echo.
pause
