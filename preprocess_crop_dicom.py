#!/usr/bin/env python3
"""
DICOM影像預處理 - 裁切病歷號區域
Preprocess DICOM - Crop Patient ID Region
"""

import os
import numpy as np
import cv2
import pydicom
from pathlib import Path
import argparse

def detect_text_region(image, top_percent=0.15, left_percent=0.3):
    """
    檢測影像中的文字區域（通常在左上角）
    
    Args:
        image: 輸入影像
        top_percent: 檢測上方多少比例
        left_percent: 檢測左側多少比例
    
    Returns:
        mask: 文字區域遮罩
    """
    h, w = image.shape[:2]
    
    # 只檢測左上角區域
    roi_h = int(h * top_percent)
    roi_w = int(w * left_percent)
    roi = image[:roi_h, :roi_w]
    
    # 二值化檢測高對比度文字
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        roi_gray = roi
    
    # Otsu二值化
    _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 檢測文字（高對比度區域）
    # 假設文字是白色（或黑色）背景上的黑色（或白色）文字
    white_ratio = np.sum(binary == 255) / binary.size
    
    # 如果白色比例很高或很低，說明有文字
    has_text = white_ratio > 0.8 or white_ratio < 0.2
    
    return has_text, (roi_h, roi_w)

def auto_crop_image(image, crop_top=0.15, crop_left=0.05, crop_right=0.05, crop_bottom=0.05):
    """
    自動裁切影像邊緣
    
    Args:
        image: 輸入影像
        crop_top: 裁切上方比例（移除病歷號）
        crop_left: 裁切左側比例
        crop_right: 裁切右側比例
        crop_bottom: 裁切下方比例
    
    Returns:
        cropped_image: 裁切後的影像
        crop_info: 裁切資訊
    """
    h, w = image.shape[:2]
    
    # 計算裁切範圍
    top = int(h * crop_top)
    bottom = h - int(h * crop_bottom)
    left = int(w * crop_left)
    right = w - int(w * crop_right)
    
    # 裁切
    cropped = image[top:bottom, left:right]
    
    crop_info = {
        'original_size': (h, w),
        'crop_box': (top, bottom, left, right),
        'cropped_size': cropped.shape[:2]
    }
    
    return cropped, crop_info

def smart_crop_spine_image(image, aggressive=True):
    """
    智能裁切脊椎影像
    
    Args:
        image: 輸入影像
        aggressive: 是否使用激進裁切（移除更多邊緣）
    
    Returns:
        cropped_image, crop_info
    """
    # 檢測文字區域
    has_text, text_region_size = detect_text_region(image)
    
    if aggressive:
        # 激進裁切：移除更多邊緣
        crop_params = {
            'crop_top': 0.20,    # 移除上方20%（包含病歷號）
            'crop_left': 0.10,   # 移除左側10%
            'crop_right': 0.10,  # 移除右側10%
            'crop_bottom': 0.05  # 移除下方5%
        }
    else:
        # 保守裁切：只移除明顯的文字區域
        crop_params = {
            'crop_top': 0.15 if has_text else 0.05,
            'crop_left': 0.05,
            'crop_right': 0.05,
            'crop_bottom': 0.05
        }
    
    cropped, crop_info = auto_crop_image(image, **crop_params)
    crop_info['has_text_detected'] = has_text
    crop_info['aggressive_mode'] = aggressive
    
    return cropped, crop_info

def process_dicom_file(input_path, output_path=None, aggressive=True, visualize=False):
    """
    處理單個DICOM檔案
    
    Args:
        input_path: 輸入DICOM路徑
        output_path: 輸出路徑（如果為None，則覆蓋原檔案）
        aggressive: 是否激進裁切
        visualize: 是否顯示視覺化
    """
    # 讀取DICOM
    dcm = pydicom.dcmread(input_path)
    image = dcm.pixel_array
    
    # 轉換為RGB
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image
    
    # 正規化
    image_rgb = ((image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min()) * 255).astype(np.uint8)
    
    print(f"處理: {Path(input_path).name}")
    print(f"  原始尺寸: {image_rgb.shape[:2]}")
    
    # 智能裁切
    cropped, crop_info = smart_crop_spine_image(image_rgb, aggressive=aggressive)
    
    print(f"  裁切後尺寸: {cropped.shape[:2]}")
    print(f"  檢測到文字: {'是' if crop_info['has_text_detected'] else '否'}")
    print(f"  裁切模式: {'激進' if aggressive else '保守'}")
    
    # 視覺化
    if visualize:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原始影像（標記裁切區域）
        axes[0].imshow(image_rgb, cmap='gray')
        axes[0].set_title('原始影像（紅框=裁切區域）')
        
        top, bottom, left, right = crop_info['crop_box']
        from matplotlib.patches import Rectangle
        rect = Rectangle((left, top), right-left, bottom-top, 
                        linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].axis('off')
        
        # 裁切後影像
        axes[1].imshow(cropped, cmap='gray')
        axes[1].set_title('裁切後影像')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 儲存（更新DICOM的pixel_array）
    if output_path:
        # 更新pixel array
        if len(cropped.shape) == 3:
            dcm.pixel_array = cropped[:, :, 0]  # 只取一個通道
        else:
            dcm.pixel_array = cropped
        
        # 更新尺寸資訊
        dcm.Rows, dcm.Columns = cropped.shape[:2]
        
        # 儲存
        dcm.save_as(output_path)
        print(f"  ✅ 已儲存: {output_path}")
    
    return cropped, crop_info

def batch_process(input_dir, output_dir=None, aggressive=True, pattern="*.dcm"):
    """
    批次處理資料夾中的DICOM檔案
    
    Args:
        input_dir: 輸入資料夾
        output_dir: 輸出資料夾（如果為None，在原資料夾建立cropped子資料夾）
        aggressive: 是否激進裁切
        pattern: 檔案匹配模式
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path / "cropped"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 找到所有DICOM檔案
    dcm_files = list(input_path.glob(pattern))
    
    print(f"找到 {len(dcm_files)} 個DICOM檔案")
    print(f"輸出目錄: {output_path}")
    print("=" * 60)
    
    for dcm_file in dcm_files:
        try:
            output_file = output_path / dcm_file.name
            process_dicom_file(dcm_file, output_file, aggressive=aggressive)
            print()
        except Exception as e:
            print(f"  ❌ 處理失敗: {e}")
            print()
    
    print("=" * 60)
    print(f"✅ 批次處理完成！共處理 {len(dcm_files)} 個檔案")
    print(f"輸出位置: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='DICOM影像預處理 - 裁切病歷號區域')
    parser.add_argument('--input', type=str, required=True,
                       help='輸入DICOM檔案或資料夾')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出路徑（預設：原位置/cropped/）')
    parser.add_argument('--mode', type=str, default='aggressive',
                       choices=['aggressive', 'conservative'],
                       help='裁切模式：aggressive(激進) 或 conservative(保守)')
    parser.add_argument('--visualize', action='store_true',
                       help='顯示視覺化結果')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    aggressive = (args.mode == 'aggressive')
    
    if input_path.is_file():
        # 單一檔案
        process_dicom_file(
            args.input,
            args.output,
            aggressive=aggressive,
            visualize=args.visualize
        )
    elif input_path.is_dir():
        # 資料夾
        batch_process(
            args.input,
            args.output,
            aggressive=aggressive
        )
    else:
        print(f"❌ 路徑不存在: {args.input}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

