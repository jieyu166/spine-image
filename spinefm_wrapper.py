#!/usr/bin/env python3
"""
SpineFM Wrapper — 封裝 SpineFM 模型的推論流程
SpineFM Wrapper — encapsulates SpineFM inference pipeline

將 SpineFM 的 batch-oriented 設計轉為 single-image API，
回傳每個椎體的 binary mask + label。

Usage:
    wrapper = SpineFMWrapper(weights_path='spinefm_weights/', dataset='NHANESII')
    results = wrapper.predict_single('image.png')
    # results = {'masks': {label: np.ndarray}, 'centroids': {label: (x, y)}}
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

# ── 動態加入 SpineFM 路徑 ──
SPINEFM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spinefm')
if SPINEFM_DIR not in sys.path:
    sys.path.insert(0, SPINEFM_DIR)

try:
    from utils import (
        find_centroids, MSA_predict,
        compute_weighted_centroid, classify,
        iou_from_logits, binary_mask_from_logits, gaussian_smooth,
    )
    from models.sam import sam_model_registry
    from models.PointPredictor.point_predictor import PointPredictor
except ImportError as e:
    raise ImportError(
        f"無法匯入 SpineFM 模組。請確認已初始化 submodule:\n"
        f"  git submodule update --init\n"
        f"原始錯誤: {e}"
    )

import torchvision

# ── 權重檔名對照表 ──
WEIGHT_FILES = {
    'NHANESII': {
        'mask_rcnn': 'mask_rcnn_2_categories.pth',
        'msa': 'MSA_600b.pth',
        'classifier': 'resnet50_4_class_224px_50_epochs.pth',
        'point_predictor': 'point_predictor.pth',
    },
    'FINETUNED': {
        'mask_rcnn': 'mask_rcnn_finetuned.pth',
        'msa': 'MSA_600b.pth',
        'classifier': 'resnet50_4_class_224px_50_epochs.pth',
        'point_predictor': 'point_predictor.pth',
    },
    'CSXA': {
        'mask_rcnn': 'mask_rcnn_csxa.pth',
        'msa': 'MSA_300b_CSXA.pth',
        'classifier': 'resnet50_2_class_224px_csxa.pth',
        'point_predictor': 'point_predictor_CSXA.pth',
    },
}


# ── NHANES II 椎體標籤對照 ──
# SpineFM 的 classifier 輸出: 0=background, 1=regular vertebra, 2=C2, 3=S1
# 我們需要根據椎體在序列中的位置 + landmark 來推斷名稱
VERTEBRA_NAMES_L = ['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1']
VERTEBRA_NAMES_C = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1']


class SpineFMWrapper:
    """封裝 SpineFM 全推論流程"""

    def __init__(self, weights_path='spinefm_weights', dataset='NHANESII', device='auto'):
        """
        Args:
            weights_path: SpineFM 模型權重資料夾路徑
            dataset: 'NHANESII' (lumbar+cervical) 或 'CSXA' (cervical only)
            device: 'auto', 'cuda', 或 'cpu'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.weights_path = weights_path
        self.dataset = dataset

        # 依 dataset 設定 patch size 和 class 數
        if dataset in ('NHANESII', 'FINETUNED'):
            self.msa_patch_size = 600
            self.classifier_patch_size = 512
            self.num_classes = 4   # bg, regular, C2, S1
        elif dataset == 'CSXA':
            self.msa_patch_size = 300
            self.classifier_patch_size = 256
            self.num_classes = 2   # bg, vertebra
        else:
            raise ValueError(f"Invalid dataset: {dataset}. Use 'NHANESII', 'CSXA', or 'FINETUNED'.")

        print(f"[SpineFM] Loading models (dataset={dataset}, device={self.device})...")
        self._load_models()
        print(f"[SpineFM] Models loaded.")

    def _load_models(self):
        """載入所有 SpineFM 模型 (依 dataset 選擇正確權重檔)"""
        wf = WEIGHT_FILES[self.dataset]
        wp = self.weights_path

        # 1. Mask R-CNN
        self.mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn()
        in_features = self.mask_rcnn.roi_heads.box_predictor.cls_score.in_features
        self.mask_rcnn.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        )
        in_features_mask = self.mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        self.mask_rcnn.roi_heads.mask_predictor = (
            torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)
        )
        self.mask_rcnn.load_state_dict(
            torch.load(os.path.join(wp, wf['mask_rcnn']), map_location=self.device)
        )
        self.mask_rcnn = self.mask_rcnn.to(self.device)
        self.mask_rcnn.eval()
        print(f"  Mask R-CNN loaded: {wf['mask_rcnn']}")

        # 2. Medical-SAM-Adaptor
        sam_checkpoint = os.path.join(wp, 'sam_vit_b_01ec64.pth')
        msa_checkpoint = os.path.join(wp, wf['msa'])
        self.msa = sam_model_registry['default'](sam_checkpoint)
        checkpoint = torch.load(msa_checkpoint, map_location=self.device)
        self.msa.load_state_dict(checkpoint['state_dict'])
        self.msa = self.msa.to(self.device)
        self.msa.eval()
        print(f"  Medical-SAM-Adaptor loaded: {wf['msa']}")

        # 3. ResNet Classifier
        self.classifier = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        in_features_fc = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Linear(in_features_fc, self.num_classes)
        self.classifier.load_state_dict(
            torch.load(os.path.join(wp, wf['classifier']), map_location=self.device)
        )
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        print(f"  ResNet Classifier loaded: {wf['classifier']}")

        # 4. Point Predictor
        self.point_predictor = PointPredictor()
        self.point_predictor.load_state_dict(
            torch.load(os.path.join(wp, wf['point_predictor']), map_location=self.device)
        )
        self.point_predictor = self.point_predictor.to(self.device)
        self.point_predictor.eval()
        print(f"  Point Predictor loaded: {wf['point_predictor']}")

    def predict_single(self, image_path, spine_type='L', score_threshold=0.5,
                        max_vertebrae=12):
        """對單張影像執行 SpineFM 推論

        使用 Mask R-CNN 偵測 + MSA refinement + PointPredictor 擴展。
        不依賴 classifier（NHANES classifier 在現代影像上全部誤判為 background），
        改用 IoU 重疊 + 邊界檢查作為停止條件。

        Args:
            image_path: 影像路徑 (PNG, JPG, or DICOM)
            spine_type: 'L' (lumbar) 或 'C' (cervical)
            score_threshold: Mask R-CNN confidence 門檻
            max_vertebrae: 最大椎體數量上限

        Returns:
            dict: {
                'masks': OrderedDict[str, np.ndarray],
                'centroids': OrderedDict[str, tuple],
                'image': PIL.Image,
                'image_size': (width, height),
            }
        """
        from collections import OrderedDict

        # ── 1. 載入影像 ──
        img = self._load_image(image_path)
        w, h = img.size

        # ── 2. Mask R-CNN 初步偵測 ──
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            outputs = self.mask_rcnn([img_tensor.to(self.device)])[0]

        masks_tensor = outputs['masks'].cpu()
        scores_tensor = outputs['scores'].cpu()

        if len(scores_tensor) == 0:
            print("[SpineFM] WARNING: Mask-RCNN detected no vertebrae.")
            return {
                'masks': OrderedDict(),
                'centroids': OrderedDict(),
                'image': img,
                'image_size': (w, h),
            }

        # ── 3. 找 centroids 並排序 (top to bottom by Y) ──
        centroids, scores = find_centroids(masks_tensor, scores_tensor, score_threshold)
        print(f"[SpineFM] Mask R-CNN centroids: {len(centroids)} (threshold={score_threshold})")

        if len(centroids) < 3:
            print(f"[SpineFM] WARNING: Only {len(centroids)} centroids (need >= 3).")
            # 嘗試降低門檻
            centroids, scores = find_centroids(masks_tensor, scores_tensor, 0.3)
            if len(centroids) < 3:
                return {
                    'masks': OrderedDict(),
                    'centroids': OrderedDict(),
                    'image': img,
                    'image_size': (w, h),
                }

        # 沿 line of best fit 排序
        x_arr, y_arr = zip(*centroids)
        A = np.array([x_arr, np.ones_like(x_arr)]).T
        m, c = np.linalg.inv((A.T @ A)) @ (A.T @ np.array(y_arr))
        y_proj = m * (np.array(x_arr) + m * (np.array(y_arr) - c)) / (1 + m**2) + c
        sorted_indices = np.argsort(y_proj)
        centroids = [centroids[i] for i in sorted_indices]

        # ── 4. MSA refinement for ALL detected centroids ──
        all_masks = []
        all_centroids_refined = []

        for point in centroids:
            pred = MSA_predict(self.msa, img, point, patch_size=self.msa_patch_size)
            all_masks.append(pred)
            p = compute_weighted_centroid(np.array(torch.sigmoid(pred).squeeze(0)))
            all_centroids_refined.append(p)

        print(f"[SpineFM] MSA refined {len(all_centroids_refined)} vertebrae")

        # ── 5. Sequential expansion (up and down from best-3) ──
        # Find top-3 consecutive highest confidence for PointPredictor seeding
        top_avg = 0
        best_start = 0
        for i in range(len(centroids) - 2):
            avg_score = sum(scores[centroids[j]] for j in range(i, i + 3)) / 3
            if avg_score > top_avg:
                top_avg = avg_score
                best_start = i

        seed_indices = [best_start, best_start + 1, best_start + 2]
        seed_points = [all_centroids_refined[i] for i in seed_indices]
        seed_masks = [all_masks[i] for i in seed_indices]

        # Expand upward (from top of seed)
        up_masks = []
        up_centroids = []
        points = list(reversed(seed_points[:3]))
        prev_masks = list(reversed(seed_masks[:3]))
        for _ in range(max_vertebrae):
            normalised = [(px / w, py / h) for (px, py) in points]
            norm_tensor = torch.tensor(normalised, dtype=torch.float32, device=self.device)
            new_point = self.point_predictor(norm_tensor.view(-1, 6))
            new_point_np = np.array(new_point.detach().cpu().squeeze(0)) * np.array([w, h])

            # Out of image bounds check
            if new_point_np[0] < 0 or new_point_np[0] > w or new_point_np[1] < 0 or new_point_np[1] > h:
                break

            new_mask = MSA_predict(self.msa, img, tuple(new_point_np), patch_size=self.msa_patch_size)
            iou = iou_from_logits(new_mask, prev_masks[-1])
            if iou > 0.1:
                break

            new_centroid = compute_weighted_centroid(np.array(torch.sigmoid(new_mask).squeeze(0)))

            # Check mask is valid (has enough pixels)
            mask_area = (torch.sigmoid(new_mask).squeeze(0) > 0.5).sum().item()
            if mask_area < 100:
                break

            up_masks.append(new_mask)
            up_centroids.append(new_centroid)
            prev_masks.append(new_mask)
            points = points[1:] + [new_centroid]

        # Expand downward (from bottom of seed)
        down_masks = []
        down_centroids = []
        points = list(seed_points[-3:])
        prev_masks = list(seed_masks[-3:])
        for _ in range(max_vertebrae):
            normalised = [(px / w, py / h) for (px, py) in points]
            norm_tensor = torch.tensor(normalised, dtype=torch.float32, device=self.device)
            new_point = self.point_predictor(norm_tensor.view(-1, 6))
            new_point_np = np.array(new_point.detach().cpu().squeeze(0)) * np.array([w, h])

            if new_point_np[0] < 0 or new_point_np[0] > w or new_point_np[1] < 0 or new_point_np[1] > h:
                break

            new_mask = MSA_predict(self.msa, img, tuple(new_point_np), patch_size=self.msa_patch_size)
            iou = iou_from_logits(new_mask, prev_masks[-1])
            if iou > 0.1:
                break

            new_centroid = compute_weighted_centroid(np.array(torch.sigmoid(new_mask).squeeze(0)))

            mask_area = (torch.sigmoid(new_mask).squeeze(0) > 0.5).sum().item()
            if mask_area < 100:
                break

            down_masks.append(new_mask)
            down_centroids.append(new_centroid)
            prev_masks.append(new_mask)
            points = points[1:] + [new_centroid]

        # ── 6. Merge: up (reversed) + seed + down ──
        # Also include non-seed Mask R-CNN detections not overlapping with existing
        final_masks = list(reversed(up_masks)) + seed_masks + down_masks
        final_centroids = list(reversed(up_centroids)) + [all_centroids_refined[i] for i in seed_indices] + down_centroids

        # Add Mask R-CNN detections that aren't already covered
        for i, (mask, centroid) in enumerate(zip(all_masks, all_centroids_refined)):
            if i in seed_indices:
                continue
            # Check if this centroid overlaps with any existing
            overlaps = False
            for fc in final_centroids:
                dist = np.sqrt((centroid[0] - fc[0])**2 + (centroid[1] - fc[1])**2)
                if dist < 80:
                    overlaps = True
                    break
            if not overlaps:
                final_masks.append(mask)
                final_centroids.append(centroid)

        print(f"[SpineFM] After expansion: {len(final_masks)} vertebrae "
              f"(seed={len(seed_indices)}, up={len(up_masks)}, down={len(down_masks)}, "
              f"extra={len(final_masks) - len(seed_indices) - len(up_masks) - len(down_masks)})")

        # ── 7. Post-process masks ──
        processed_masks = []
        for mask in final_masks:
            bm = binary_mask_from_logits(mask, 0.9)
            if self.dataset in ('NHANESII', 'FINETUNED'):
                bm = gaussian_smooth(bm)
            processed_masks.append(bm)

        # ── 8. Sort by centroid Y (top to bottom), deduplicate ──
        mask_entries = []
        for mask, centroid in zip(processed_masks, final_centroids):
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = mask
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(0)
            cy, cx = np.where(mask_np > 0.5)
            if len(cy) > 0:
                mask_entries.append((np.mean(cx), np.mean(cy), mask_np))

        mask_entries.sort(key=lambda x: x[1])

        # Remove duplicates (masks with centroids too close together)
        deduped = []
        for entry in mask_entries:
            is_dup = False
            for existing in deduped:
                dist = np.sqrt((entry[0] - existing[0])**2 + (entry[1] - existing[1])**2)
                if dist < 60:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(entry)

        mask_entries = deduped

        # ── 9. 標記椎體名稱 (position-based) ──
        names = VERTEBRA_NAMES_L if spine_type == 'L' else VERTEBRA_NAMES_C
        n_masks = len(mask_entries)

        # For lumbar: bottom vertebra is S1, assign upward
        # For cervical: top vertebra is C2, assign downward
        labeled_masks = OrderedDict()
        labeled_centroids = OrderedDict()

        if spine_type == 'L':
            # Assign from bottom (S1) upward
            for i, (cx, cy, mask) in enumerate(reversed(mask_entries)):
                idx = len(VERTEBRA_NAMES_L) - 1 - i
                if 0 <= idx < len(VERTEBRA_NAMES_L):
                    name = VERTEBRA_NAMES_L[idx]
                    labeled_masks[name] = mask
                    labeled_centroids[name] = (cx, cy)
        else:
            # Assign from top (C2) downward
            for i, (cx, cy, mask) in enumerate(mask_entries):
                if i < len(VERTEBRA_NAMES_C):
                    name = VERTEBRA_NAMES_C[i]
                    labeled_masks[name] = mask
                    labeled_centroids[name] = (cx, cy)

        # 確保按解剖學順序排列
        sorted_masks = OrderedDict()
        sorted_centroids = OrderedDict()
        for name in names:
            if name in labeled_masks:
                sorted_masks[name] = labeled_masks[name]
                sorted_centroids[name] = labeled_centroids[name]

        print(f"[SpineFM] Detected {len(sorted_masks)} vertebrae: {list(sorted_masks.keys())}")

        return {
            'masks': sorted_masks,
            'centroids': sorted_centroids,
            'image': img,
            'image_size': (w, h),
        }

    def _load_image(self, image_path):
        """載入影像為 PIL RGB Image（支援 DICOM, PNG, JPG）"""
        image_path = str(image_path)

        if image_path.lower().endswith('.dcm'):
            import pydicom
            dcm = pydicom.dcmread(image_path)
            arr = dcm.pixel_array
            if len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            return Image.fromarray(arr)
        else:
            return Image.open(image_path).convert('RGB')
