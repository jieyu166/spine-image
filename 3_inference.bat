@echo off
chcp 65001 >nul
echo ========================================
echo 脊椎終板檢測 - 推理
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

echo 請選擇推理模式:
echo.
echo 1. 分析單一DICOM檔案
echo 2. 批次處理資料夾
echo 3. 使用範例影像測試
echo.

set /p mode="請輸入選項 (1/2/3): "

if "%mode%"=="1" (
    echo.
    set /p input_file="請輸入DICOM檔案路徑: "
    python inference.py --model best_endplate_model.pth --input "%input_file%"
) else if "%mode%"=="2" (
    echo.
    set /p input_dir="請輸入資料夾路徑: "
    python inference.py --model best_endplate_model.pth --input "%input_dir%" --no-viz
) else if "%mode%"=="3" (
    echo.
    echo 使用範例影像（請確保有範例檔案）
    python inference.py --model best_endplate_model.pth --input "19826153done\198261530.dcm"
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

