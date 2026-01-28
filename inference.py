#!/usr/bin/env python3
"""
脊椎終板檢測 - 推理腳本
Spine Endplate Detection - Inference Script

使用訓練好的模型進行推理
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
from train_endplate_model import EndplateDetectionModel, get_transforms

class SpineAnalyzer:
    """脊椎分析器"""
    
    def __init__(self, model_path, device='auto'):
        """
        初始化分析器
        
        Args:
            model_path: 模型檔案路徑
            device: 'auto', 'cuda', 或 'cpu'
        """
        # 設備設置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 使用設備: {self.device}")
        
        # 載入模型
        print(f"📦 載入模型: {model_path}")
        self.model = EndplateDetectionModel(pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 載入配置
        self.config = checkpoint.get('config', {})
        print(f"✅ 模型載入完成")
        print(f"   訓練epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   驗證損失: {checkpoint.get('val_loss', 'unknown'):.4f}" if 'val_loss' in checkpoint else "")
        
        # 預處理
        self.transform = get_transforms(is_training=False)
    
    def load_image(self, image_path):
        """載入影像（支援DICOM和一般影像格式）"""
        image_path = str(image_path)
        
        if image_path.lower().endswith('.dcm'):
            # DICOM檔案
            dcm = pydicom.dcmread(image_path)
            image = dcm.pixel_array
            
            # 轉換為RGB
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # 正規化到0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            # 一般影像檔案
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"無法讀取影像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def preprocess(self, image):
        """預處理影像"""
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        return input_tensor
    
    def predict(self, image_path):
        """
        預測終板位置
        
        Returns:
            dict: 包含預測結果的字典
        """
        # 載入影像
        image = self.load_image(image_path)
        original_h, original_w = image.shape[:2]
        
        print(f"   原始影像尺寸: {original_w} × {original_h}")
        
        # 預處理
        input_tensor = self.preprocess(image)
        print(f"   模型輸入尺寸: {input_tensor.shape}")
        
        # 推理
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # 提取結果
        endplate_seg = predictions['endplate_seg'][0, 0].cpu().numpy()
        vertebra_edge_seg = predictions['vertebra_edge_seg'][0].cpu().numpy()
        keypoint_heatmap = predictions['keypoint_heatmap'][0, 0].cpu().numpy()
        
        print(f"   模型輸出尺寸: {endplate_seg.shape}")
        
        # Resize回原始尺寸（確保使用正確的插值方法）
        endplate_seg = cv2.resize(endplate_seg, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        anterior_edge = cv2.resize(vertebra_edge_seg[0], (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        posterior_edge = cv2.resize(vertebra_edge_seg[1], (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        keypoint_heatmap = cv2.resize(keypoint_heatmap, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        print(f"   Resize後尺寸: {endplate_seg.shape}")
        
        return {
            'image': image,
            'endplate_mask': endplate_seg,
            'anterior_edge': anterior_edge,
            'posterior_edge': posterior_edge,
            'keypoint_heatmap': keypoint_heatmap,
            'original_size': (original_w, original_h)  # 保存原始尺寸
        }
    
    def extract_endplates(self, mask, threshold=0.5, min_length=30):
        """從遮罩提取終板線段"""
        print(f"   提取終板 - 遮罩尺寸: {mask.shape}, 閾值: {threshold}")
        
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        # 檢查有效像素
        valid_pixels = np.sum(binary_mask > 0)
        print(f"   二值化遮罩 - 有效像素數: {valid_pixels}")
        
        # 霍夫線段檢測
        lines = cv2.HoughLinesP(
            binary_mask,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=min_length,
            maxLineGap=10
        )
        
        if lines is None:
            print(f"   ⚠️ 未檢測到線段")
            return []
        
        print(f"   初步檢測到 {len(lines)} 條線段")
        
        # 合併相近的線段
        merged_lines = self._merge_lines(lines)
        print(f"   合併後剩餘 {len(merged_lines)} 條線段")
        
        # 按y座標排序（從上到下）
        merged_lines = sorted(merged_lines, key=lambda l: (l[1] + l[3]) / 2)
        
        # 輸出線段座標範圍供調試
        if merged_lines:
            x_coords = [x for line in merged_lines for x in [line[0], line[2]]]
            y_coords = [y for line in merged_lines for y in [line[1], line[3]]]
            print(f"   線段座標範圍: X[{min(x_coords)}-{max(x_coords)}], Y[{min(y_coords)}-{max(y_coords)}]")
        
        return merged_lines
    
    def _merge_lines(self, lines, angle_threshold=10, distance_threshold=20):
        """合併相近的線段"""
        if len(lines) == 0:
            return []
        
        merged = []
        lines = [line[0] for line in lines]
        
        while lines:
            line1 = lines.pop(0)
            x1, y1, x2, y2 = line1
            angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # 找到相似的線段
            similar = [line1]
            i = 0
            while i < len(lines):
                x3, y3, x4, y4 = lines[i]
                angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
                
                # 檢查角度和距離
                if abs(angle1 - angle2) < angle_threshold:
                    dist = min(
                        np.sqrt((x1-x3)**2 + (y1-y3)**2),
                        np.sqrt((x2-x4)**2 + (y2-y4)**2)
                    )
                    if dist < distance_threshold:
                        similar.append(lines.pop(i))
                        continue
                i += 1
            
            # 合併為一條線段
            if similar:
                all_points = np.array([(x1, y1, x2, y2) for x1, y1, x2, y2 in similar])
                x_min, x_max = all_points[:, [0, 2]].min(), all_points[:, [0, 2]].max()
                y_mean1 = all_points[:, 1].mean()
                y_mean2 = all_points[:, 3].mean()
                merged.append([int(x_min), int(y_mean1), int(x_max), int(y_mean2)])
        
        return merged
    
    def calculate_angles(self, endplate_lines):
        """計算椎間隙角度"""
        angles = []
        
        for i in range(len(endplate_lines) - 1):
            line1 = endplate_lines[i]
            line2 = endplate_lines[i + 1]
            
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
            
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            
            angles.append({
                'level': i + 1,
                'angle': round(angle_diff, 1),
                'lower_line': line1,
                'upper_line': line2
            })
        
        return angles
    
    def visualize(self, results, output_path=None, show=True):
        """視覺化結果"""
        image = results['image']
        endplate_mask = results['endplate_mask']
        anterior_edge = results['anterior_edge']
        posterior_edge = results['posterior_edge']
        keypoint_heatmap = results['keypoint_heatmap']
        
        # 提取終板線段
        endplate_lines = self.extract_endplates(endplate_mask)
        
        # 計算角度
        angles = self.calculate_angles(endplate_lines) if len(endplate_lines) >= 2 else []
        
        # 創建視覺化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('脊椎終板檢測結果', fontsize=16, fontweight='bold')
        
        # 原始影像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始影像')
        axes[0, 0].axis('off')
        
        # 終板檢測
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(endplate_mask, alpha=0.5, cmap='Reds')
        axes[0, 1].set_title(f'終板檢測 ({len(endplate_lines)} 條線段)')
        axes[0, 1].axis('off')
        
        # 前緣檢測
        axes[0, 2].imshow(image)
        axes[0, 2].imshow(anterior_edge, alpha=0.5, cmap='Blues')
        axes[0, 2].set_title('椎體前緣')
        axes[0, 2].axis('off')
        
        # 後緣檢測
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(posterior_edge, alpha=0.5, cmap='Oranges')
        axes[1, 0].set_title('椎體後緣')
        axes[1, 0].axis('off')
        
        # 關鍵點
        axes[1, 1].imshow(image)
        axes[1, 1].imshow(keypoint_heatmap, alpha=0.5, cmap='hot')
        axes[1, 1].set_title('關鍵點檢測')
        axes[1, 1].axis('off')
        
        # 最終結果（繪製線段和角度）
        result_img = image.copy()
        for i, line in enumerate(endplate_lines):
            x1, y1, x2, y2 = line
            cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # 標記椎間隙和角度
            if i < len(angles):
                angle_info = angles[i]
                mid_y = (y1 + y2) // 2
                cv2.putText(result_img, f"#{angle_info['level']}: {angle_info['angle']}°",
                           (10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        axes[1, 2].imshow(result_img)
        axes[1, 2].set_title(f'最終結果 ({len(angles)} 個角度)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ 視覺化結果已儲存: {output_path}")
        
        if show:
            plt.show()
        
        return angles, endplate_lines
    
    def analyze(self, image_path, output_dir=None, visualize=True):
        """完整分析流程"""
        print(f"\n📸 分析影像: {image_path}")
        
        # 預測
        results = self.predict(image_path)
        
        # 提取終板
        endplate_lines = self.extract_endplates(results['endplate_mask'])
        print(f"   檢測到 {len(endplate_lines)} 條終板線段")
        
        # 計算角度
        angles = self.calculate_angles(endplate_lines)
        print(f"   計算出 {len(angles)} 個椎間隙角度")
        
        for angle_info in angles:
            print(f"   - 椎間隙 #{angle_info['level']}: {angle_info['angle']}°")
        
        # 視覺化
        if visualize and output_dir:
            output_path = Path(output_dir) / f"{Path(image_path).stem}_result.png"
            self.visualize(results, output_path=output_path, show=False)
        elif visualize:
            self.visualize(results, show=True)
        
        # 準備輸出
        output = {
            'image_file': str(image_path),
            'num_endplates': len(endplate_lines),
            'num_angles': len(angles),
            'endplate_lines': [
                {'x1': int(l[0]), 'y1': int(l[1]), 'x2': int(l[2]), 'y2': int(l[3])}
                for l in endplate_lines
            ],
            'angles': angles
        }
        
        # 儲存JSON
        if output_dir:
            json_path = Path(output_dir) / f"{Path(image_path).stem}_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON結果已儲存: {json_path}")
        
        return output

def main():
    parser = argparse.ArgumentParser(description='脊椎終板檢測推理')
    parser.add_argument('--model', type=str, default='best_endplate_model.pth',
                       help='模型檔案路徑')
    parser.add_argument('--input', type=str, required=True,
                       help='輸入影像或資料夾路徑')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='輸出結果資料夾')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='計算設備')
    parser.add_argument('--no-viz', action='store_true',
                       help='不生成視覺化結果')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 初始化分析器
    analyzer = SpineAnalyzer(args.model, device=args.device)
    
    # 處理輸入
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 單一檔案
        analyzer.analyze(
            input_path,
            output_dir=output_dir,
            visualize=not args.no_viz
        )
    elif input_path.is_dir():
        # 資料夾（批次處理）
        image_files = list(input_path.glob('*.dcm')) + \
                     list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpg'))
        
        print(f"\n📁 批次處理: {len(image_files)} 個檔案")
        
        for img_file in image_files:
            try:
                analyzer.analyze(
                    img_file,
                    output_dir=output_dir,
                    visualize=not args.no_viz
                )
            except Exception as e:
                print(f"❌ 處理失敗 {img_file}: {e}")
                continue
        
        print(f"\n✅ 批次處理完成！結果儲存在: {output_dir}")
    else:
        print(f"❌ 輸入路徑無效: {args.input}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

