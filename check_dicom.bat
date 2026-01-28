@echo off
chcp 65001 >nul
echo ========================================
echo 檢查DICOM和JSON配對
echo ========================================
echo.

REM 切換到腳本所在目錄
cd /d "%~dp0"

python check_dicom.py

echo.
pause

