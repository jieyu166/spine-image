@echo off
chcp 65001 >nul
echo ========================================
echo 訓練椎體頂點檢測模型 V3
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

echo 當前目錄: %CD%
echo.

echo 開始訓練...
python train_vertebra_model.py

if errorlevel 1 (
    echo.
    echo 訓練失敗
    pause
    exit /b 1
)

echo.
echo ========================================
echo 完成！
echo ========================================
pause

