@echo off
chcp 65001 >nul
echo ========================================
echo 終板檢測模型 - 完整流程
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

echo 當前目錄: %CD%
echo.

echo ========================================
echo 步驟 1/3: 驗證數據
echo ========================================
python quick_test.py
if errorlevel 1 (
    echo.
    echo ❌ 驗證失敗，請檢查JSON檔案格式
    pause
    exit /b 1
)

echo.
echo ========================================
echo 步驟 2/3: 準備訓練數據
echo ========================================
python prepare_endplate_data.py
if errorlevel 1 (
    echo.
    echo ❌ 數據準備失敗
    pause
    exit /b 1
)

echo.
echo ========================================
echo 步驟 3/3: 開始訓練
echo ========================================
echo.
echo ⚠️  訓練將需要數小時時間
echo    您可以隨時按 Ctrl+C 中斷
echo.
pause

python train_vertebra_model.py

echo.
echo ========================================
echo ✅ 完成！
echo ========================================
pause

