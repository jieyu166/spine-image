@echo off
REM ==========================================
REM  SpineFM Setup Script
REM  初始化 submodule + 下載模型權重
REM ==========================================

echo ==========================================
echo  SpineFM Setup
echo ==========================================

REM 1. Init git submodule
echo.
echo [1/4] Initializing SpineFM submodule...
git submodule update --init
if errorlevel 1 (
    echo ERROR: Failed to init submodule. Make sure git is installed.
    pause
    exit /b 1
)

REM 2. Create weights directory
echo.
echo [2/4] Creating weights directory...
if not exist "spinefm_weights" mkdir spinefm_weights

REM 3. Download SAM base weights
echo.
echo [3/4] Downloading SAM base weights (sam_vit_b_01ec64.pth)...
if not exist "spinefm_weights\sam_vit_b_01ec64.pth" (
    echo Downloading from Meta AI...
    curl -L -o spinefm_weights\sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    if errorlevel 1 (
        echo ERROR: Failed to download SAM weights.
        echo Please manually download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
        echo and place in spinefm_weights\
    )
) else (
    echo SAM weights already exist, skipping.
)

REM 4. Remind user about SpineFM-specific weights
echo.
echo [4/4] SpineFM model weights
echo ==========================================
echo.
echo SAM base weights downloaded successfully.
echo.
echo You still need to download the SpineFM-specific weights manually:
echo.
echo   NHANES II weights:
echo     https://drive.google.com/drive/folders/17nhBgptkHA5GkxAKrmIH1nlLdRp5PDQM
echo.
echo   CSXA weights:
echo     https://drive.google.com/drive/folders/1oAt1AW6EMBXLefdn442uYoQwReslAh6X
echo.
echo Download the following files and place them in spinefm_weights\:
echo   - mask_rcnn_csxa.pth        (Mask R-CNN)
echo   - MSA_300b_CSXA.pth         (Medical-SAM-Adaptor)
echo   - point_predictor.pth       (Point Predictor)
echo   - resnet50_2_class_224px_csxa.pth  (ResNet Classifier)
echo.
echo Check spinefm\utils.py get_model() for exact filenames.
echo ==========================================

REM 5. Install dependencies
echo.
echo Installing Python dependencies...
pip install scikit-image pillow opencv-python fastapi uvicorn pydicom
if errorlevel 1 (
    echo WARNING: Some dependencies failed to install.
)

echo.
echo ==========================================
echo  Setup complete!
echo  Run: python inference_spinefm.py --input image.png --spine-type L
echo ==========================================
pause
