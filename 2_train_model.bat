@echo off
echo ========================================
echo 訓練終板檢測模型
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

echo 當前目錄: %CD%
echo.

echo 開始訓練...
python train_endplate_model.py

echo.
echo ========================================
echo 完成！
echo ========================================
pause

