@echo off
chcp 65001 >nul
echo ========================================
echo 準備終板檢測訓練數據 (修正版)
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

echo 當前目錄: %CD%
echo.

REM 清理舊的訓練數據
if exist "endplate_training_data" (
    echo 🗑️  清理舊的訓練數據...
    rmdir /s /q "endplate_training_data"
    echo ✅ 清理完成
    echo.
)

echo 執行數據準備...
python prepare_endplate_data.py

if errorlevel 1 (
    echo.
    echo ❌ 數據準備失敗
    echo.
    echo 可能原因:
    echo 1. 找不到DICOM檔案
    echo 2. JSON格式錯誤
    echo 3. Python套件未安裝
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 完成！
echo ========================================
echo.
echo 生成的檔案:
dir /b "endplate_training_data\annotations"
echo.
pause

