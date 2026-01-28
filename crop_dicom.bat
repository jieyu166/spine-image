@echo off
chcp 65001 >nul
echo ========================================
echo DICOM影像預處理 - 裁切病歷號區域
echo ========================================
echo.

cd /d "%~dp0"

echo 此工具會裁切DICOM影像的邊緣區域，移除病歷號等文字
echo.
echo 請選擇裁切模式:
echo.
echo 1. 激進模式（推薦）- 移除上方20%%、左右各10%%
echo 2. 保守模式 - 只移除檢測到文字的區域
echo 3. 測試單一檔案（含視覺化）
echo.

set /p mode="請輸入選項 (1/2/3): "

if "%mode%"=="1" (
    echo.
    echo 激進模式：批次處理
    set /p input_dir="請輸入DICOM資料夾路徑: "
    python preprocess_crop_dicom.py --input "%input_dir%" --mode aggressive
) else if "%mode%"=="2" (
    echo.
    echo 保守模式：批次處理
    set /p input_dir="請輸入DICOM資料夾路徑: "
    python preprocess_crop_dicom.py --input "%input_dir%" --mode conservative
) else if "%mode%"=="3" (
    echo.
    echo 測試模式：單一檔案
    set /p input_file="請輸入DICOM檔案路徑: "
    python preprocess_crop_dicom.py --input "%input_file%" --mode aggressive --visualize
) else (
    echo.
    echo ❌ 無效的選項
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 處理完成！
echo ========================================
echo.
echo 裁切後的檔案儲存在 cropped\ 子資料夾
echo.
pause

