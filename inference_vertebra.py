#!/usr/bin/env python3
"""
脊椎椎體頂點檢測 - 推理腳本 V2
Spine Vertebra Corner Detection - Inference Script

使用 VertebraCornerModel 對脊椎影像進行椎體頂點檢測。

用法:
    # 單張影像
    python inference_vertebra.py --input image.png

    # 批次處理資料夾
    python inference_vertebra.py --input ./images/ --output ./results/

    # 指定模型和脊椎類型
    python inference_vertebra.py --input image.dcm --model best_vertebra_model.pth --spine-type L
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
import pydicom
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from train_vertebra_model import VertebraCornerModel


# 椎體名稱對照表
VERTEBRA_NAMES = {
    'L': ['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'],
    'C': ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1'],
}

# 邊界椎體定義
BOUNDARY_CONFIG = {
    'L': {'upper': ['S1'], 'lower': ['T12']},
    'C': {'upper': ['T1'], 'lower': ['C2']},
}

# 4 個角點名稱 (固定順序)
CORNER_NAMES = ['anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior']

# 角點繪製顏色 (BGR)
CORNER_COLORS = [
    (0, 255, 0),    # anteriorSuperior - 綠
    (255, 0, 0),    # posteriorSuperior - 藍
    (0, 0, 255),    # posteriorInferior - 紅
    (0, 255, 255),  # anteriorInferior - 黃
]


class VertebraInference:
    """椎體頂點檢測推理器"""

    def __init__(self, model_path, device='auto', max_vertebrae=8):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.max_vertebrae = max_vertebrae
        print(f"Device: {self.device}")

        # 載入模型
        print(f"Loading model: {model_path}")
        self.model = VertebraCornerModel(max_vertebrae=max_vertebrae, pretrained=False)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.config = checkpoint.get('config', {})
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', None)
        print(f"Model loaded (epoch {epoch}, val_loss {val_loss:.4f})" if val_loss else f"Model loaded (epoch {epoch})")

    def load_image(self, image_path):
        """載入影像 (支援 DICOM / PNG / JPG)"""
        image_path = str(image_path)

        if image_path.lower().endswith('.dcm'):
            dcm = pydicom.dcmread(image_path)
            image = dcm.pixel_array
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def predict(self, image_path, spine_type='L'):
        """
        預測椎體頂點座標

        Args:
            image_path: 影像路徑
            spine_type: 'L' (腰椎) 或 'C' (頸椎)

        Returns:
            dict: 包含椎體名稱、角點座標、信心度等
        """
        image = self.load_image(image_path)
        original_h, original_w = image.shape[:2]

        # 預處理: resize to 512x512, normalize
        resized = cv2.resize(image, (512, 512))
        input_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # 解析結果
        coords = predictions['coords'][0].cpu().numpy()       # [N*4, 2] normalized [0,1]
        count_logits = predictions['count_logits'][0].cpu().numpy()
        heatmap = predictions['heatmap'][0, 0].cpu().numpy()   # [H, W]

        # 預測的椎體數量
        predicted_count = int(np.argmax(count_logits))
        count_confidence = float(np.exp(count_logits[predicted_count]) / np.exp(count_logits).sum())

        # 反正規化座標到原始影像尺寸
        scale_x = original_w
        scale_y = original_h

        # 解析每個椎體的 4 個角點
        names = VERTEBRA_NAMES.get(spine_type, VERTEBRA_NAMES['L'])
        boundary = BOUNDARY_CONFIG.get(spine_type, {})

        vertebrae = []
        for i in range(min(predicted_count, self.max_vertebrae)):
            name = names[i] if i < len(names) else f'V{i+1}'
            base_idx = i * 4

            # 判斷邊界類型
            if name in boundary.get('upper', []):
                boundary_type = 'upper'
            elif name in boundary.get('lower', []):
                boundary_type = 'lower'
            else:
                boundary_type = None

            corners = {}
            for j in range(4):
                corner_name = CORNER_NAMES[j]

                # 邊界椎體: 跳過無效角點
                if boundary_type == 'upper' and j >= 2:
                    continue
                if boundary_type == 'lower' and j < 2:
                    continue

                x_norm, y_norm = coords[base_idx + j]
                corners[corner_name] = {
                    'x': float(x_norm * scale_x),
                    'y': float(y_norm * scale_y),
                }

            vertebrae.append({
                'name': name,
                'boundaryType': boundary_type,
                'points': corners,
            })

        # 計算椎體指標
        for v in vertebrae:
            self._calculate_metrics(v)

        # 計算椎間盤指標
        discs = self._calculate_discs(vertebrae, spine_type)

        return {
            'image_path': str(image_path),
            'spine_type': spine_type,
            'image_info': {'width': original_w, 'height': original_h},
            'predicted_count': predicted_count,
            'count_confidence': count_confidence,
            'vertebrae': vertebrae,
            'discs': discs,
            'heatmap': heatmap,
            'original_image': image,
        }

    def _calculate_metrics(self, vertebra):
        """計算單個椎體的前緣/後緣高度"""
        pts = vertebra['points']
        bt = vertebra['boundaryType']

        if bt:
            vertebra['anteriorHeight'] = None
            vertebra['posteriorHeight'] = None
            vertebra['heightRatio'] = None
            vertebra['anteriorWedgingFracture'] = False
            vertebra['crushDeformityFracture'] = False
            return

        ant_sup = pts.get('anteriorSuperior', {})
        ant_inf = pts.get('anteriorInferior', {})
        post_sup = pts.get('posteriorSuperior', {})
        post_inf = pts.get('posteriorInferior', {})

        ant_h = np.sqrt((ant_inf['x'] - ant_sup['x'])**2 + (ant_inf['y'] - ant_sup['y'])**2)
        post_h = np.sqrt((post_inf['x'] - post_sup['x'])**2 + (post_inf['y'] - post_sup['y'])**2)

        ratio = ant_h / post_h if post_h > 0 else 0

        vertebra['anteriorHeight'] = float(ant_h)
        vertebra['posteriorHeight'] = float(post_h)
        vertebra['heightRatio'] = float(ratio)
        vertebra['anteriorWedgingFracture'] = bool(ratio < 0.75)
        vertebra['crushDeformityFracture'] = bool(ratio > 1.25)

    def _calculate_discs(self, vertebrae, spine_type):
        """計算椎間盤指標"""
        discs = []
        boundary = BOUNDARY_CONFIG.get(spine_type, {})

        for i in range(len(vertebrae) - 1):
            upper = vertebrae[i]
            lower = vertebrae[i + 1]

            # 上方需要下終板, 下方需要上終板
            if upper.get('boundaryType') == 'upper':
                continue  # 上邊界沒有下終板
            if lower.get('boundaryType') == 'lower':
                continue  # 下邊界沒有上終板

            upper_pts = upper['points']
            lower_pts = lower['points']

            # 上方椎體的下終板
            u_ant_inf = upper_pts.get('anteriorInferior', {})
            u_post_inf = upper_pts.get('posteriorInferior', {})
            # 下方椎體的上終板
            l_ant_sup = lower_pts.get('anteriorSuperior', {})
            l_post_sup = lower_pts.get('posteriorSuperior', {})

            if not all([u_ant_inf, u_post_inf, l_ant_sup, l_post_sup]):
                continue

            ant_h = np.sqrt((l_ant_sup['x'] - u_ant_inf['x'])**2 + (l_ant_sup['y'] - u_ant_inf['y'])**2)
            post_h = np.sqrt((l_post_sup['x'] - u_post_inf['x'])**2 + (l_post_sup['y'] - u_post_inf['y'])**2)

            discs.append({
                'level': f"{upper['name']}/{lower['name']}",
                'anteriorHeight': float(ant_h),
                'posteriorHeight': float(post_h),
                'middleHeight': float((ant_h + post_h) / 2),
            })

        return discs

    def visualize(self, result, output_path=None, show=False):
        """視覺化預測結果"""
        image = result['original_image'].copy()
        vertebrae = result['vertebrae']
        heatmap = result['heatmap']
        original_h, original_w = image.shape[:2]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(
            f"Vertebra Detection - {result['spine_type']}-spine "
            f"({result['predicted_count']} vertebrae, "
            f"confidence: {result['count_confidence']:.1%})",
            fontsize=14, fontweight='bold'
        )

        # 1. 原始影像 + 角點
        vis_image = image.copy()
        for v in vertebrae:
            pts = v['points']
            corners_xy = []

            for j, corner_name in enumerate(CORNER_NAMES):
                if corner_name not in pts:
                    continue
                p = pts[corner_name]
                x, y = int(p['x']), int(p['y'])
                corners_xy.append((x, y, j))

                # 畫點
                color = CORNER_COLORS[j]
                radius = max(3, min(original_w, original_h) // 200)
                cv2.circle(vis_image, (x, y), radius, color, -1)

            # 畫椎體輪廓 (連接所有角點)
            if len(corners_xy) >= 2:
                pts_arr = [(c[0], c[1]) for c in corners_xy]
                for k in range(len(pts_arr)):
                    cv2.line(vis_image, pts_arr[k], pts_arr[(k+1) % len(pts_arr)],
                             (255, 255, 255), max(1, min(original_w, original_h) // 400))

            # 標記椎體名稱
            if corners_xy:
                avg_x = int(np.mean([c[0] for c in corners_xy]))
                avg_y = int(np.mean([c[1] for c in corners_xy]))
                font_scale = max(0.4, min(original_w, original_h) / 2000)
                thickness = max(1, int(font_scale * 2))

                # 異常標記
                label = v['name']
                if v.get('anteriorWedgingFracture'):
                    label += ' [AW]'
                elif v.get('crushDeformityFracture'):
                    label += ' [Crush]'

                cv2.putText(vis_image, label, (avg_x - 20, avg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

        axes[0].imshow(vis_image)
        axes[0].set_title('Vertebra Corners')
        axes[0].axis('off')

        # 2. 熱圖疊加
        heatmap_resized = cv2.resize(heatmap, (original_w, original_h))
        axes[1].imshow(image)
        axes[1].imshow(heatmap_resized, alpha=0.5, cmap='hot')
        axes[1].set_title('Keypoint Heatmap')
        axes[1].axis('off')

        # 3. 椎間盤分析
        disc_image = image.copy()
        discs = result['discs']
        for disc in discs:
            axes[2].text(0.05, 0.95 - discs.index(disc) * 0.08,
                        f"{disc['level']}: A={disc['anteriorHeight']:.0f} P={disc['posteriorHeight']:.0f}",
                        transform=axes[2].transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[2].imshow(disc_image)
        axes[2].set_title('Disc Analysis')
        axes[2].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {output_path}")

        if show:
            plt.show()

        plt.close()

    def analyze(self, image_path, spine_type='L', output_dir=None, visualize=True):
        """完整分析流程"""
        print(f"\nAnalyzing: {image_path} ({spine_type}-spine)")

        result = self.predict(image_path, spine_type)

        # 打印結果
        print(f"  Predicted vertebrae: {result['predicted_count']} (confidence: {result['count_confidence']:.1%})")
        for v in result['vertebrae']:
            bt_label = f" [{v['boundaryType']}]" if v['boundaryType'] else ""
            ratio_label = ""
            if v.get('heightRatio') is not None:
                ratio_label = f" (A/P={v['heightRatio']:.2f})"
                if v.get('anteriorWedgingFracture'):
                    ratio_label += " !! Anterior Wedging"
                if v.get('crushDeformityFracture'):
                    ratio_label += " !! Crush Deformity"
            print(f"  {v['name']}{bt_label}: {len(v['points'])} corners{ratio_label}")

        for d in result['discs']:
            print(f"  Disc {d['level']}: A={d['anteriorHeight']:.1f} P={d['posteriorHeight']:.1f}")

        # 儲存結果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            # JSON 結果 (排除 numpy arrays 和 image)
            json_result = {k: v for k, v in result.items()
                          if k not in ('heatmap', 'original_image')}
            json_path = output_dir / f"{Path(image_path).stem}_vertebra_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            print(f"  JSON saved: {json_path}")

            # 視覺化
            if visualize:
                viz_path = output_dir / f"{Path(image_path).stem}_vertebra_result.png"
                self.visualize(result, output_path=str(viz_path))

        return result


def main():
    parser = argparse.ArgumentParser(description='Spine Vertebra Corner Detection Inference V2')
    parser.add_argument('--model', type=str, default='best_vertebra_model.pth',
                       help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory')
    parser.add_argument('--spine-type', type=str, default='L', choices=['L', 'C'],
                       help='Spine type: L (lumbar) or C (cervical)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Compute device')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')

    args = parser.parse_args()

    # 初始化推理器
    analyzer = VertebraInference(args.model, device=args.device)

    # 處理輸入
    input_path = Path(args.input)

    if input_path.is_file():
        analyzer.analyze(
            input_path,
            spine_type=args.spine_type,
            output_dir=args.output,
            visualize=not args.no_viz
        )
    elif input_path.is_dir():
        image_files = (
            list(input_path.glob('*.dcm')) +
            list(input_path.glob('*.png')) +
            list(input_path.glob('*.jpg'))
        )
        print(f"\nBatch processing: {len(image_files)} files")

        for img_file in image_files:
            try:
                analyzer.analyze(
                    img_file,
                    spine_type=args.spine_type,
                    output_dir=args.output,
                    visualize=not args.no_viz
                )
            except Exception as e:
                print(f"ERROR processing {img_file}: {e}")
    else:
        print(f"Invalid input path: {args.input}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
