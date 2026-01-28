@echo off
chcp 65001 >nul
echo ========================================
echo 測試單個批次
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

python test_single_batch.py

if errorlevel 1 (
    echo.
    echo ❌ 測試失敗
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ 測試成功！可以開始訓練。
echo.
pause

